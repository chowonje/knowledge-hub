from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from knowledge_hub.core.models import SearchResult
from knowledge_hub.knowledge.ontology_profiles import OntologyProfileManager


def get_active_profile(
    *,
    sqlite_db: Any,
    cached_profile: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if cached_profile is not None:
        return cached_profile
    if not sqlite_db:
        return None
    try:
        manager = OntologyProfileManager(sqlite_db)
        return manager.compile_active_profile()
    except Exception:
        return None


def resolve_topology_snapshot_path(config: Any) -> Path | None:
    vault_path = str(getattr(config, "vault_path", "") or "").strip() if config else ""
    if not vault_path:
        return None
    root = Path(vault_path).expanduser().resolve() / ".obsidian" / "khub" / "topology"
    if not root.exists():
        return None
    preferred = root / "latest.json"
    if preferred.exists():
        return preferred
    candidates = sorted(root.glob("latest*.json"), key=lambda item: item.stat().st_mtime)
    return candidates[-1] if candidates else None


def load_topology_index(
    *,
    config: Any,
    cached_topology: dict[str, Any] | None,
) -> dict[str, Any] | None:
    snapshot_path = resolve_topology_snapshot_path(config)
    if snapshot_path is None or not snapshot_path.exists():
        return None
    cache_key = f"{snapshot_path}:{int(snapshot_path.stat().st_mtime)}"
    if isinstance(cached_topology, dict) and cached_topology.get("cacheKey") == cache_key:
        return cached_topology
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    nodes = payload.get("nodes", []) if isinstance(payload.get("nodes"), list) else []
    clusters = payload.get("clusters", []) if isinstance(payload.get("clusters"), list) else []
    by_path: dict[str, dict[str, Any]] = {}
    cluster_map: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        path = str(node.get("path", "")).strip()
        if path:
            by_path[path] = node
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("id", "")).strip()
        if cluster_id:
            cluster_map[cluster_id] = cluster
    return {
        "cacheKey": cache_key,
        "snapshotPath": str(snapshot_path),
        "nodesByPath": by_path,
        "clustersById": cluster_map,
    }


def resolve_query_entities(
    query: str,
    *,
    sqlite_db: Any,
    tokenize_fn: Callable[[str], list[str]],
    max_related: int = 8,
) -> list[dict[str, Any]]:
    if not sqlite_db:
        return []
    try:
        from knowledge_hub.learning.resolver import EntityResolver

        resolver = EntityResolver(sqlite_db)
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for token in sorted(tokenize_fn(query)):
            identity = resolver.resolve(token)
            if not identity:
                continue
            entity_id = str(identity.canonical_id or "").strip()
            if not entity_id or entity_id in seen:
                continue
            seen.add(entity_id)
            entity = sqlite_db.get_ontology_entity(entity_id) or {}
            items.append(
                {
                    "entity_id": entity_id,
                    "canonical_name": str(entity.get("canonical_name", token)),
                    "entity_type": str(entity.get("entity_type", "concept")),
                    "properties": dict(entity.get("properties") or {}),
                    "knowledge_kind": str((entity.get("properties") or {}).get("knowledge_kind", "")),
                    "facets": dict((entity.get("properties") or {}).get("facets") or {}),
                }
            )
            if len(items) >= max_related:
                break
        return items
    except Exception:
        return []


def apply_profile_and_cluster_scope(
    results: list[SearchResult],
    *,
    profile: dict[str, Any] | None,
    topology: dict[str, Any] | None,
    top_k: int,
    apply_score_boosts: bool,
    safe_float_fn: Callable[[Any, float], float],
    source_label_for_result_fn: Callable[[SearchResult], str],
    merge_top_signal_items_fn: Callable[[list[dict[str, Any]], dict[str, float]], list[dict[str, Any]]],
    retrieval_sort_key_fn: Callable[[SearchResult], tuple[float, float, float, float]],
) -> tuple[list[SearchResult], list[dict[str, Any]], dict[str, Any] | None]:
    preferred_sources = {
        str(item).strip().lower()
        for item in ((profile or {}).get("retrieval_facets", {}) or {}).get("preferred_sources", [])
        if str(item).strip()
    }
    cluster_scores: dict[str, float] = {}
    cluster_hits: dict[str, int] = {}

    for item in results:
        source_type = source_label_for_result_fn(item)
        if apply_score_boosts and preferred_sources and source_type in preferred_sources:
            item.score = min(1.0, safe_float_fn(item.score, 0.0) + 0.04)
        if topology is None:
            continue
        file_path = str((item.metadata or {}).get("file_path", "")).strip()
        node = topology["nodesByPath"].get(file_path) if file_path else None
        if not node:
            continue
        cluster_id = str(node.get("clusterId", "")).strip()
        if not cluster_id:
            continue
        cluster_scores[cluster_id] = cluster_scores.get(cluster_id, 0.0) + safe_float_fn(item.score, 0.0)
        cluster_hits[cluster_id] = cluster_hits.get(cluster_id, 0) + 1
        item.metadata.setdefault("cluster_id", cluster_id)

    shortlisted = sorted(
        cluster_scores.items(),
        key=lambda item: (item[1], cluster_hits.get(item[0], 0)),
        reverse=True,
    )[:3]
    shortlisted_ids = {cluster_id for cluster_id, _ in shortlisted}
    cluster_rank_by_id = {cluster_id: index for index, (cluster_id, _) in enumerate(shortlisted)}
    cluster_boost_by_id = {
        cluster_id: max(0.0, 0.03 - (0.01 * index))
        for index, (cluster_id, _) in enumerate(shortlisted)
    }
    related_clusters: list[dict[str, Any]] = []
    for cluster_id, score in shortlisted:
        cluster_meta = (topology or {}).get("clustersById", {}).get(cluster_id, {})
        related_clusters.append(
            {
                "cluster_id": cluster_id,
                "label": str(cluster_meta.get("label", "") or cluster_id),
                "size": int(cluster_meta.get("size", 0) or 0),
                "representative_note_id": str(cluster_meta.get("representativeNoteId", "") or ""),
                "score": round(float(score), 6),
                "hit_count": int(cluster_hits.get(cluster_id, 0)),
                "boost": round(float(cluster_boost_by_id.get(cluster_id, 0.0)), 6),
            }
        )

    for item in results:
        cluster_id = str((item.metadata or {}).get("cluster_id", "")).strip()
        if not cluster_id:
            continue
        cluster_meta = (topology or {}).get("clustersById", {}).get(cluster_id, {})
        cluster_selected = cluster_id in shortlisted_ids
        cluster_rank = cluster_rank_by_id.get(cluster_id, -1)
        cluster_boost = safe_float_fn(cluster_boost_by_id.get(cluster_id, 0.0) if cluster_selected else 0.0, 0.0)
        if cluster_selected and cluster_boost > 0:
            item.score = min(1.0, safe_float_fn(item.score, 0.0) + cluster_boost)
        extras = dict(item.lexical_extras or {})
        ranking_signals = dict(extras.get("ranking_signals") or {})
        ranking_signals["cluster_id"] = cluster_id
        ranking_signals["cluster_label"] = str(cluster_meta.get("label", "") or cluster_id)
        ranking_signals["cluster_hit_count"] = int(cluster_hits.get(cluster_id, 0))
        ranking_signals["cluster_rank"] = int(cluster_rank)
        ranking_signals["cluster_selected"] = bool(cluster_selected)
        ranking_signals["cluster_proximity_boost"] = round(float(cluster_boost), 6)
        extras["cluster_id"] = cluster_id
        extras["cluster_label"] = ranking_signals["cluster_label"]
        extras["cluster_hit_count"] = ranking_signals["cluster_hit_count"]
        extras["cluster_rank"] = ranking_signals["cluster_rank"]
        extras["cluster_selected"] = ranking_signals["cluster_selected"]
        extras["cluster_proximity_boost"] = ranking_signals["cluster_proximity_boost"]
        extras["top_ranking_signals"] = merge_top_signal_items_fn(
            list(extras.get("top_ranking_signals") or []),
            {"cluster_proximity_boost": cluster_boost},
        )
        ranking_signals["top_ranking_signals"] = list(extras["top_ranking_signals"])
        extras["ranking_signals"] = ranking_signals
        item.lexical_extras = extras

    if shortlisted_ids:
        scoped = [item for item in results if str((item.metadata or {}).get("cluster_id", "")).strip() in shortlisted_ids]
        spillover = [item for item in results if item not in scoped]
        scoped.sort(key=retrieval_sort_key_fn, reverse=True)
        spillover.sort(key=retrieval_sort_key_fn, reverse=True)
        merged = scoped + spillover
        return merged[:top_k], related_clusters, profile
    return results[:top_k], related_clusters, profile
