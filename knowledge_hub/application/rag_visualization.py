"""RAG retrieval visualization payloads and standalone Three.js report HTML."""

from __future__ import annotations

from datetime import datetime, timezone
import html
import json
import math
import re
from typing import Any

RAG_VISUALIZATION_SCHEMA = "knowledge-hub.rag.visualization.result.v1"
ALL_SOURCE_TYPES = ("vault", "paper", "web", "concept")


def _clean_text(value: Any, *, limit: int = 280) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)].rstrip()}..."


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _result_metadata(result: Any) -> dict[str, Any]:
    metadata = getattr(result, "metadata", None)
    return dict(metadata or {}) if isinstance(metadata, dict) else {}


def _result_node(result: Any, *, index: int, total: int) -> dict[str, Any]:
    metadata = _result_metadata(result)
    source_type = str(metadata.get("source_type") or metadata.get("sourceType") or "unknown")
    score = max(0.0, min(1.0, _safe_float(getattr(result, "score", 0.0), 0.0)))
    semantic_score = max(0.0, min(1.0, _safe_float(getattr(result, "semantic_score", 0.0), 0.0)))
    lexical_score = max(0.0, min(1.0, _safe_float(getattr(result, "lexical_score", 0.0), 0.0)))
    angle = (2.0 * math.pi * index) / max(1, total)
    radius = 8.0 - (score * 4.0)
    z_offset = (semantic_score - lexical_score) * 4.0
    return {
        "id": str(getattr(result, "document_id", "") or metadata.get("document_id") or f"result-{index + 1}"),
        "kind": "result",
        "rank": index + 1,
        "title": _clean_text(metadata.get("title") or "Untitled", limit=140),
        "sourceType": source_type,
        "score": round(score, 6),
        "semanticScore": round(semantic_score, 6),
        "lexicalScore": round(lexical_score, 6),
        "distance": round(_safe_float(getattr(result, "distance", 1.0), 1.0), 6),
        "documentId": str(getattr(result, "document_id", "") or metadata.get("document_id") or ""),
        "parentId": str(metadata.get("resolved_parent_id") or metadata.get("parent_id") or ""),
        "parentLabel": _clean_text(metadata.get("resolved_parent_label") or metadata.get("parent_label") or "", limit=160),
        "snippet": _clean_text(getattr(result, "document", ""), limit=360),
        "position": {
            "x": round(math.cos(angle) * radius, 6),
            "y": round(math.sin(angle) * radius, 6),
            "z": round(z_offset + ((index % 3) - 1) * 0.8, 6),
        },
    }


def _build_edges(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    for node in nodes:
        if node.get("kind") != "result":
            continue
        score = _safe_float(node.get("score"), 0.0)
        edges.append(
            {
                "id": f"query->{node.get('id')}",
                "source": "query",
                "target": node.get("id"),
                "kind": "similarity",
                "weight": round(score, 6),
            }
        )
    return edges


def _search_payload(
    searcher: Any,
    *,
    query: str,
    top_k: int,
    source_type: str | None,
    retrieval_mode: str,
    alpha: float,
) -> dict[str, Any]:
    return searcher.search_with_diagnostics(
        query,
        top_k=max(1, int(top_k)),
        source_type=source_type,
        retrieval_mode=retrieval_mode,
        alpha=float(alpha),
        expand_parent_context=True,
    )


def _result_key(result: Any) -> str:
    metadata = _result_metadata(result)
    return str(
        getattr(result, "document_id", "")
        or metadata.get("document_id")
        or metadata.get("resolved_parent_id")
        or metadata.get("title")
        or getattr(result, "document", "")
    )


def _merged_all_source_search(
    searcher: Any,
    *,
    query: str,
    top_k: int,
    retrieval_mode: str,
    alpha: float,
) -> tuple[list[Any], dict[str, Any], list[str]]:
    per_source_k = max(3, math.ceil(max(1, int(top_k)) / len(ALL_SOURCE_TYPES)))
    all_results: list[Any] = []
    diagnostics_by_source: dict[str, Any] = {}
    warnings: list[str] = []

    for source in ALL_SOURCE_TYPES:
        try:
            source_payload = _search_payload(
                searcher,
                query=query,
                top_k=per_source_k,
                source_type=source,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
            )
        except Exception as error:  # keep the map useful if one optional source is unavailable
            warnings.append(f"{source} search failed: {error}")
            continue
        source_results = list(source_payload.get("results") or [])
        all_results.extend(source_results)
        diagnostics_by_source[source] = {
            "resultCount": len(source_results),
            "diagnostics": dict(source_payload.get("diagnostics") or {}),
        }

    deduped: dict[str, Any] = {}
    for result in sorted(all_results, key=lambda item: _safe_float(getattr(item, "score", 0.0), 0.0), reverse=True):
        key = _result_key(result)
        if key and key not in deduped:
            deduped[key] = result
    results = list(deduped.values())[: max(1, int(top_k))]
    diagnostics = {
        "retrievalStrategy": {
            "phase": "rag_visualization_all_source_merge",
            "sourceScope": "all",
            "sourceTypes": list(ALL_SOURCE_TYPES),
            "perSourceTopK": per_source_k,
        },
        "retrievalQuality": {
            "label": "mixed",
            "score": max([_safe_float(getattr(item, "score", 0.0), 0.0) for item in results] or [0.0]),
            "evidenceCount": len(results),
            "sourceTypes": sorted(
                {
                    str(_result_metadata(item).get("source_type") or _result_metadata(item).get("sourceType") or "unknown")
                    for item in results
                }
            ),
        },
        "answerabilityRerank": {},
        "correctiveRetrieval": {},
        "artifactHealth": {},
        "sources": diagnostics_by_source,
    }
    return results, diagnostics, warnings


def build_rag_visualization_payload(
    searcher: Any,
    *,
    query: str,
    top_k: int = 20,
    source_type: str | None = None,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.7,
    output_path: str = "",
) -> dict[str, Any]:
    """Build an inspectable graph payload for a single RAG retrieval query."""

    warnings: list[str] = []
    if source_type is None:
        results, diagnostics, warnings = _merged_all_source_search(
            searcher,
            query=query,
            top_k=max(1, int(top_k)),
            retrieval_mode=retrieval_mode,
            alpha=float(alpha),
        )
    else:
        payload = _search_payload(
            searcher,
            query=query,
            top_k=max(1, int(top_k)),
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=float(alpha),
        )
        results = list(payload.get("results") or [])
        diagnostics = dict(payload.get("diagnostics") or {})
    result_nodes = [_result_node(result, index=index, total=len(results)) for index, result in enumerate(results)]
    nodes = [
        {
            "id": "query",
            "kind": "query",
            "rank": 0,
            "title": _clean_text(query, limit=160),
            "sourceType": "query",
            "score": 1.0,
            "semanticScore": 1.0,
            "lexicalScore": 1.0,
            "distance": 0.0,
            "documentId": "",
            "parentId": "",
            "parentLabel": "",
            "snippet": _clean_text(query, limit=360),
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
        *result_nodes,
    ]
    quality = dict(diagnostics.get("retrievalQuality") or {})
    return {
        "schema": RAG_VISUALIZATION_SCHEMA,
        "status": "ok" if results else "no_results",
        "query": str(query),
        "sourceType": source_type or "all",
        "retrievalMode": str(retrieval_mode),
        "topK": max(1, int(top_k)),
        "alpha": float(alpha),
        "readOnly": True,
        "labsOnly": True,
        "runtimeApplied": False,
        "artifactWritten": bool(output_path),
        "artifactPath": str(output_path or ""),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "resultCount": len(results),
        "summary": {
            "nodeCount": len(nodes),
            "edgeCount": len(result_nodes),
            "qualityLabel": str(quality.get("label") or ""),
            "qualityScore": _safe_float(quality.get("score"), 0.0),
        },
        "retrievalStrategy": dict(diagnostics.get("retrievalStrategy") or {}),
        "retrievalQuality": quality,
        "answerabilityRerank": dict(diagnostics.get("answerabilityRerank") or {}),
        "correctiveRetrieval": dict(diagnostics.get("correctiveRetrieval") or {}),
        "artifactHealth": dict(diagnostics.get("artifactHealth") or {}),
        "graph": {
            "layout": "score_radial_v1",
            "encoding": {
                "distance": "higher retrieval score places a result closer to the query node",
                "edgeWeight": "retrieval score",
                "zAxis": "semantic score minus lexical score",
            },
            "nodes": nodes,
            "edges": _build_edges(nodes),
        },
        "warnings": warnings,
    }


def render_rag_visualization_html(payload: dict[str, Any]) -> str:
    """Render a standalone Three.js HTML report with embedded payload data."""

    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    title = html.escape(str(payload.get("query") or "RAG visualization"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Knowledge Map - {title}</title>
  <style>
    html, body {{ margin: 0; height: 100%; background: #101412; color: #f2f5ef; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; overflow: hidden; }}
    #app {{ position: fixed; inset: 0; }}
    .panel {{ position: fixed; top: 16px; left: 16px; width: min(340px, calc(100vw - 32px)); background: rgba(16, 20, 18, 0.86); border: 1px solid rgba(242, 245, 239, 0.16); border-radius: 8px; padding: 14px; backdrop-filter: blur(10px); box-sizing: border-box; }}
    .panel h1 {{ margin: 0 0 8px; font-size: 16px; line-height: 1.25; font-weight: 700; }}
    .meta {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 10px 0; font-size: 12px; color: #c8d0c5; }}
    .meta span {{ display: block; color: #f2f5ef; font-weight: 650; }}
    #details {{ margin-top: 10px; font-size: 12px; line-height: 1.45; color: #dce3d8; max-height: 150px; overflow: auto; }}
    .evidence-panel {{ position: fixed; top: 16px; right: 16px; width: min(380px, calc(100vw - 32px)); max-height: calc(100vh - 32px); overflow: auto; background: rgba(16, 20, 18, 0.88); border: 1px solid rgba(242, 245, 239, 0.16); border-radius: 8px; padding: 14px; backdrop-filter: blur(10px); box-sizing: border-box; }}
    .evidence-panel h2 {{ margin: 0 0 10px; font-size: 14px; line-height: 1.25; }}
    .evidence-item {{ border-top: 1px solid rgba(242, 245, 239, 0.12); padding: 10px 0; cursor: pointer; }}
    .evidence-item:first-of-type {{ border-top: 0; padding-top: 0; }}
    .evidence-item.active {{ color: #ffffff; }}
    .evidence-title {{ display: flex; align-items: baseline; gap: 8px; font-size: 12px; font-weight: 700; line-height: 1.3; }}
    .rank {{ min-width: 28px; color: #f3c969; font-variant-numeric: tabular-nums; }}
    .scorebar {{ height: 5px; margin: 7px 0 5px; background: rgba(242, 245, 239, 0.13); border-radius: 99px; overflow: hidden; }}
    .scorefill {{ height: 100%; width: 0; background: linear-gradient(90deg, #f3c969, #8ee6a8); }}
    .evidence-meta {{ font-size: 11px; color: #c8d0c5; }}
    .evidence-snippet {{ margin-top: 5px; font-size: 11px; line-height: 1.35; color: #aeb9aa; }}
    .legend {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; font-size: 11px; color: #c8d0c5; }}
    .key {{ display: inline-flex; align-items: center; gap: 5px; }}
    .swatch {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
    @media (max-width: 860px) {{
      .panel {{ width: calc(100vw - 32px); }}
      .evidence-panel {{ left: 16px; right: auto; top: auto; bottom: 16px; width: calc(100vw - 32px); max-height: 34vh; }}
    }}
  </style>
</head>
<body>
  <div id="app"></div>
  <aside class="panel">
    <h1>{title}</h1>
    <div class="meta">
      <div>mode<span>{html.escape(str(payload.get("retrievalMode") or ""))}</span></div>
      <div>results<span>{int(payload.get("resultCount") or 0)}</span></div>
      <div>source<span>{html.escape(str(payload.get("sourceType") or "all"))}</span></div>
      <div>layout<span>score radial</span></div>
    </div>
    <div id="details">Hover a node to inspect its source, score, and snippet.</div>
    <div class="legend">
      <span class="key"><i class="swatch" style="background:#f3c969"></i>query</span>
      <span class="key"><i class="swatch" style="background:#7bc6ff"></i>paper</span>
      <span class="key"><i class="swatch" style="background:#8ee6a8"></i>vault</span>
      <span class="key"><i class="swatch" style="background:#c99cff"></i>concept</span>
      <span class="key"><i class="swatch" style="background:#ff9d7b"></i>web</span>
    </div>
  </aside>
  <aside class="evidence-panel">
    <h2>Top Evidence</h2>
    <div id="evidenceList"></div>
  </aside>
  <script type="application/json" id="payload">{payload_json}</script>
  <script type="importmap">
    {{ "imports": {{ "three": "https://unpkg.com/three@0.164.1/build/three.module.js" }} }}
  </script>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'https://unpkg.com/three@0.164.1/examples/jsm/controls/OrbitControls.js';

    const payload = JSON.parse(document.getElementById('payload').textContent);
    const graph = payload.graph || {{ nodes: [], edges: [] }};
    const root = document.getElementById('app');
    const details = document.getElementById('details');
    const evidenceList = document.getElementById('evidenceList');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101412);
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(2.5, -18, 12);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    root.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    scene.add(new THREE.AmbientLight(0xffffff, 0.72));
    const light = new THREE.DirectionalLight(0xffffff, 1.0);
    light.position.set(4, -7, 12);
    scene.add(light);

    const colors = {{ query: 0xf3c969, paper: 0x7bc6ff, vault: 0x8ee6a8, concept: 0xc99cff, web: 0xff9d7b, unknown: 0xdce3d8 }};
    const nodeById = new Map();
    const pickables = [];
    const labelGroup = new THREE.Group();
    scene.add(labelGroup);

    function displayPosition(node) {{
      const pos = node.position || {{ x: 0, y: 0, z: 0 }};
      if (node.kind === 'query') return new THREE.Vector3(-2.4, 0, 1.2);
      const rank = Number(node.rank || 0);
      const spread = rank <= 5 ? 1.18 : 1.0;
      return new THREE.Vector3(Number(pos.x || 0) * spread + 1.6, Number(pos.y || 0), Number(pos.z || 0));
    }}

    function labelSprite(text, color, scale = 1) {{
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = 512;
      canvas.height = 96;
      ctx.font = '700 30px Inter, system-ui, sans-serif';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = 'rgba(16, 20, 18, 0.72)';
      roundRect(ctx, 0, 12, 512, 72, 12);
      ctx.fill();
      ctx.fillStyle = '#' + color.toString(16).padStart(6, '0');
      ctx.fillText(text.slice(0, 34), 18, 48);
      const texture = new THREE.CanvasTexture(canvas);
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({{ map: texture, transparent: true, depthTest: false }}));
      sprite.scale.set(3.6 * scale, 0.68 * scale, 1);
      return sprite;
    }}

    function roundRect(ctx, x, y, w, h, r) {{
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }}

    for (const node of graph.nodes || []) {{
      const pos = displayPosition(node);
      const color = colors[node.kind === 'query' ? 'query' : node.sourceType] || colors.unknown;
      const score = Math.max(0, Number(node.score || 0));
      const topBoost = Number(node.rank || 99) <= 5 ? 0.16 : 0;
      const radius = node.kind === 'query' ? 0.68 : 0.22 + score * 0.54 + topBoost;
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 32, 16),
        new THREE.MeshStandardMaterial({{ color, roughness: 0.42, metalness: 0.08, emissive: color, emissiveIntensity: node.kind === 'query' ? 0.28 : Number(node.rank || 99) <= 5 ? 0.14 : 0.04 }})
      );
      mesh.position.copy(pos);
      mesh.userData.node = node;
      scene.add(mesh);
      nodeById.set(node.id, mesh);
      pickables.push(mesh);
      if (node.kind === 'query' || Number(node.rank || 99) <= 8) {{
        const label = labelSprite(node.kind === 'query' ? 'Query' : `#${{node.rank}} ${{node.title || node.id}}`, color, node.kind === 'query' ? 1.05 : 0.86);
        label.position.copy(pos).add(new THREE.Vector3(0.45, 0.2, 0.75));
        label.userData.node = node;
        labelGroup.add(label);
      }}
    }}

    for (const edge of graph.edges || []) {{
      const a = nodeById.get(edge.source);
      const b = nodeById.get(edge.target);
      if (!a || !b) continue;
      const material = new THREE.LineBasicMaterial({{ color: 0xdce3d8, transparent: true, opacity: 0.08 + Math.max(0, Number(edge.weight || 0)) * 0.68 }});
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([a.position, b.position]), material);
      scene.add(line);
    }}

    const grid = new THREE.GridHelper(18, 18, 0x4a554e, 0x27302b);
    grid.rotation.x = Math.PI / 2;
    grid.material.opacity = 0.28;
    grid.material.transparent = true;
    scene.add(grid);

    const evidenceNodes = (graph.nodes || []).filter((node) => node.kind === 'result').slice(0, 10);
    evidenceList.innerHTML = evidenceNodes.map((node) => `
      <div class="evidence-item" data-node-id="${{escapeAttr(node.id)}}">
        <div class="evidence-title"><span class="rank">#${{node.rank}}</span><span>${{escapeHtml(node.title || node.id)}}</span></div>
        <div class="scorebar"><div class="scorefill" style="width:${{Math.round(Number(node.score || 0) * 100)}}%"></div></div>
        <div class="evidence-meta">${{escapeHtml(node.sourceType || '')}} · score ${{Number(node.score || 0).toFixed(3)}} · sem ${{Number(node.semanticScore || 0).toFixed(3)}}</div>
        <div class="evidence-snippet">${{escapeHtml(node.snippet || '')}}</div>
      </div>
    `).join('');
    evidenceList.querySelectorAll('.evidence-item').forEach((item) => {{
      item.addEventListener('pointerenter', () => {{
        const mesh = nodeById.get(item.dataset.nodeId);
        if (mesh) inspectNode(mesh.userData.node);
      }});
    }});

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2(2, 2);
    window.addEventListener('pointermove', (event) => {{
      pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
      pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;
    }});
    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    function inspectNode(node) {{
      document.querySelectorAll('.evidence-item').forEach((item) => item.classList.toggle('active', item.dataset.nodeId === node.id));
      details.innerHTML = `<strong>${{node.rank ? '#' + node.rank + ' ' : ''}}${{escapeHtml(node.title || node.id)}}</strong><br>` +
        `source=${{escapeHtml(node.sourceType || '')}} score=${{Number(node.score || 0).toFixed(3)}} semantic=${{Number(node.semanticScore || 0).toFixed(3)}} lexical=${{Number(node.lexicalScore || 0).toFixed(3)}}<br>` +
        `<span>${{escapeHtml(node.snippet || '')}}</span>`;
    }}
    function escapeAttr(value) {{
      return escapeHtml(value).replace(/`/g, '&#096;');
    }}
    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, (char) => ({{ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }}[char]));
    }}
    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(pickables, false)[0];
      if (hit && hit.object.userData.node) inspectNode(hit.object.userData.node);
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>
"""


__all__ = [
    "RAG_VISUALIZATION_SCHEMA",
    "build_rag_visualization_payload",
    "render_rag_visualization_html",
]
