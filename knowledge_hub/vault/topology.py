"""Vault topology snapshot builder.

Builds a note-level topology artifact from existing vault embeddings and SQLite
link metadata. The output is designed for downstream visualization layers such
as an Obsidian custom view, but the builder itself is backend-only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
from typing import Any

from knowledge_hub.core.schema_validator import annotate_schema_errors


TOPOLOGY_SCHEMA_ID = "knowledge-hub.vault.topology.snapshot.v1"

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "do",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "with",
    "그",
    "그리고",
    "것",
    "는",
    "등",
    "또는",
    "및",
    "수",
    "위해",
    "이",
    "있다",
    "저",
    "하는",
}

_GENERIC_CLUSTER_TOKENS = {
    "archive",
    "archives",
    "benchmark",
    "benchmarks",
    "concept",
    "concepts",
    "deep",
    "deep-dive",
    "dive",
    "document",
    "documents",
    "draft",
    "file",
    "files",
    "folder",
    "folders",
    "idea",
    "ideas",
    "info",
    "latest",
    "license",
    "note",
    "notes",
    "paper",
    "papers",
    "preview",
    "readme",
    "resource",
    "resources",
    "study",
    "summary",
    "temp",
    "todo",
}


class TopologyBuildError(RuntimeError):
    """Raised when topology generation cannot proceed."""


@dataclass(slots=True)
class TopologyBuildOptions:
    projection: str = "auto"
    neighbors: int = 15
    min_dist: float = 0.08
    similarity_threshold: float = 0.30
    min_cluster_size: int = 2
    spherize: bool = False
    output_path: str | None = None
    scope: str = "whole-vault"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _load_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment specific
        raise TopologyBuildError(
            "vault topology requires numpy. Install with: pip install 'knowledge-hub-cli[topology]'"
        ) from exc
    return np


def _load_projection_model(kind: str, min_dist: float):
    if kind == "umap":
        try:
            from umap import UMAP
        except ImportError as exc:  # pragma: no cover - environment specific
            raise TopologyBuildError(
                "projection=umap requires umap-learn. Install with: pip install 'knowledge-hub-cli[topology]'"
            ) from exc
        return UMAP(n_components=3, metric="cosine", min_dist=float(min_dist), random_state=42)

    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:  # pragma: no cover - environment specific
        raise TopologyBuildError(
            "projection=pca requires scikit-learn. Install with: pip install 'knowledge-hub-cli[topology]'"
        ) from exc
    return PCA(n_components=3, random_state=42)


def _normalize_path_id(value: str) -> str:
    return str(value or "").strip()


def _decode_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except Exception:
            return {}
        if isinstance(payload, dict):
            return payload
    return {}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z]{2,}|[가-힣]{2,}", str(text or ""))
    return [token.lower() for token in tokens if token.lower() not in _STOPWORDS]


def _path_tokens(path: str) -> list[str]:
    rel = Path(str(path or ""))
    parts = list(rel.with_suffix("").parts)
    out: list[str] = []
    for part in parts:
        out.extend(_tokenize(part))
    return out


def _clean_cluster_term(value: str) -> str:
    token = str(value or "").strip().lower()
    token = re.sub(r"[^0-9a-z가-힣._/+ -]+", " ", token)
    token = re.sub(r"\s+", " ", token).strip(" -_/+.").strip()
    if not token:
        return ""
    if token in _STOPWORDS or token in _GENERIC_CLUSTER_TOKENS:
        return ""
    if re.fullmatch(r"[0-9.]+", token):
        return ""
    if not re.search(r"[a-z가-힣]", token):
        return ""
    return token


class VaultTopologyBuilder:
    def __init__(self, vault_path: str, vector_db, sqlite_db):
        self.vault_path = Path(vault_path).expanduser().resolve()
        self.vector_db = vector_db
        self.sqlite_db = sqlite_db

    def build(self, options: TopologyBuildOptions | None = None) -> dict[str, Any]:
        opts = options or TopologyBuildOptions()
        if not self.vault_path.exists():
            raise TopologyBuildError(f"vault path does not exist: {self.vault_path}")

        note_vectors, chunk_coverage = self._build_note_vectors()
        notes, stale_sqlite_notes = self._load_vault_notes()

        embedded_ids = sorted(note_id for note_id in notes if note_id in note_vectors)
        if not embedded_ids:
            raise TopologyBuildError(
                "no embedded vault notes found. Run 'khub index --vault-all' before building topology."
            )

        nodes, matrix = self._materialize_nodes(notes, note_vectors, embedded_ids)
        wiki_edges = self._load_wiki_edges(set(nodes))
        embedding_edges, soft_neighbors = self._build_embedding_neighbor_graph(embedded_ids, matrix, opts)
        cluster_map = self._assign_clusters(nodes, embedding_edges, min_cluster_size=opts.min_cluster_size)
        importance = self._compute_importance(nodes, wiki_edges, embedding_edges)
        cluster_meta = self._build_cluster_metadata(nodes, cluster_map, importance)
        bridge_scores, external_cluster_weights = self._compute_bridge_scores(
            nodes,
            wiki_edges,
            soft_neighbors,
            cluster_map,
        )
        bridge_edges = self._build_bridge_hint_edges(
            nodes,
            cluster_map,
            cluster_meta,
            bridge_scores,
            external_cluster_weights,
        )
        coords = self._project(matrix, opts)

        max_importance = max(importance.values(), default=0.0) or 1.0
        node_payloads: list[dict[str, Any]] = []
        for index, note_id in enumerate(embedded_ids):
            note = nodes[note_id]
            cluster_id = cluster_map[note_id]
            wiki_degree = sum(
                1
                for edge in wiki_edges
                if edge["source"] == note_id or edge["target"] == note_id
            )
            embedding_degree = sum(
                1
                for edge in embedding_edges
                if edge["source"] == note_id or edge["target"] == note_id
            )
            bridge_score = round(bridge_scores.get(note_id, 0.0), 6)
            node_payloads.append(
                {
                    "id": note_id,
                    "path": note["path"],
                    "title": note["title"],
                    "tags": note["tags"],
                    "clusterId": cluster_id,
                    "x": round(float(coords[index][0]), 6),
                    "y": round(float(coords[index][1]), 6),
                    "z": round(float(coords[index][2]), 6),
                    "importance": round(float(importance.get(note_id, 0.0) / max_importance), 6),
                    "bridgeScore": bridge_score,
                    "bridge": bool(bridge_score >= 0.35 and (wiki_degree + embedding_degree) >= 3),
                    "orphan": wiki_degree == 0 and embedding_degree == 0,
                    "updatedAt": note["updatedAt"],
                }
            )

        payload = {
            "schema": TOPOLOGY_SCHEMA_ID,
            "generatedAt": _now_iso(),
            "vaultPath": str(self.vault_path),
            "scope": opts.scope,
            "embeddingProvider": chunk_coverage["provider"],
            "embeddingModel": chunk_coverage["model"],
            "projection": self._resolve_projection_kind(opts.projection),
            "neighbors": int(opts.neighbors),
            "minDist": float(opts.min_dist),
            "spherize": bool(opts.spherize),
            "coverage": {
                "totalVaultNotes": len(notes),
                "embeddedNotes": len(embedded_ids),
                "skippedNotes": sorted(note_id for note_id in notes if note_id not in note_vectors),
                "staleSqliteNotes": int(stale_sqlite_notes),
                "vaultChunkCount": chunk_coverage["chunk_count"],
                "embeddedChunkCount": chunk_coverage["embedded_chunk_count"],
            },
            "nodes": node_payloads,
            "edges": sorted(wiki_edges + embedding_edges + bridge_edges, key=lambda item: (item["type"], item["source"], item["target"])),
            "clusters": [cluster_meta[key] for key in sorted(cluster_meta)],
        }
        annotate_schema_errors(payload, TOPOLOGY_SCHEMA_ID, strict=False)
        return payload

    def write_snapshot(self, payload: dict[str, Any], output_path: str | None = None) -> str:
        target = Path(output_path or self._default_output_path()).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=str(target.parent),
            prefix=f".{target.stem}.",
            suffix=".tmp",
            encoding="utf-8",
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            tmp_name = handle.name
        Path(tmp_name).replace(target)
        return str(target)

    def _default_output_path(self) -> Path:
        return self.vault_path / ".obsidian" / "khub" / "topology" / "latest.json"

    def _build_note_vectors(self) -> tuple[dict[str, Any], dict[str, Any]]:
        np = _load_numpy()
        chunk_data = self.vector_db.get_documents(
            filter_dict={"source_type": "vault"},
            limit=200_000,
            include_documents=False,
            include_metadatas=True,
            include_embeddings=True,
        )
        ids = list(chunk_data.get("ids") or [])
        metadatas = list(chunk_data.get("metadatas") or [])
        embeddings = list(chunk_data.get("embeddings") or [])

        if not ids or not embeddings:
            return {}, {"chunk_count": 0, "embedded_chunk_count": 0, "provider": "", "model": ""}

        grouped: dict[str, list[Any]] = defaultdict(list)
        provider = ""
        model = ""
        for metadata, embedding in zip(metadatas, embeddings):
            metadata_dict = metadata if isinstance(metadata, dict) else {}
            file_path = _normalize_path_id(metadata_dict.get("file_path", ""))
            if not file_path or embedding is None:
                continue
            provider = provider or str(metadata_dict.get("embedding_provider", "") or "")
            model = model or str(metadata_dict.get("embedding_model", "") or "")
            vector = np.asarray(embedding, dtype=float)
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            grouped[file_path].append(vector)

        note_vectors: dict[str, Any] = {}
        for file_path, vectors in grouped.items():
            if not vectors:
                continue
            mean_vec = np.mean(np.stack(vectors), axis=0)
            norm = float(np.linalg.norm(mean_vec))
            if norm > 0:
                mean_vec = mean_vec / norm
            note_vectors[file_path] = mean_vec

        return note_vectors, {
            "chunk_count": len(ids),
            "embedded_chunk_count": len([item for item in embeddings if item is not None]),
            "provider": provider,
            "model": model,
        }

    def _load_vault_notes(self) -> tuple[dict[str, dict[str, Any]], int]:
        rows = self.sqlite_db.list_notes(source_type="vault", limit=1_000_000)
        notes: dict[str, dict[str, Any]] = {}
        stale_count = 0
        for row in rows:
            note_id = _normalize_path_id(row.get("id", ""))
            if not note_id:
                continue
            rel_path = Path(str(row.get("file_path") or note_id))
            if not (self.vault_path / rel_path).exists():
                stale_count += 1
                continue
            notes[note_id] = {
                "id": note_id,
                "path": rel_path.as_posix(),
                "title": str(row.get("title") or Path(note_id).stem),
                "updatedAt": str(row.get("updated_at") or ""),
                "tags": sorted(self.sqlite_db.get_note_tags(note_id)),
                "metadata": _decode_metadata(row.get("metadata")),
            }
        return notes, stale_count

    def _materialize_nodes(self, notes: dict[str, dict[str, Any]], note_vectors: dict[str, Any], embedded_ids: list[str]):
        np = _load_numpy()
        matrix = np.stack([note_vectors[note_id] for note_id in embedded_ids])
        embedded_notes = {note_id: notes[note_id] for note_id in embedded_ids}
        return embedded_notes, matrix

    def _load_wiki_edges(self, note_ids: set[str]) -> list[dict[str, Any]]:
        graph_data = self.sqlite_db.get_graph_data()
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for edge in graph_data.get("edges", []):
            source = _normalize_path_id(edge.get("source", ""))
            target = _normalize_path_id(edge.get("target", ""))
            if source not in note_ids or target not in note_ids:
                continue
            edge_key = (source, target, "wiki_link")
            if edge_key in seen:
                continue
            seen.add(edge_key)
            edges.append({"source": source, "target": target, "type": "wiki_link", "weight": 1.0})
        return edges

    def _build_embedding_neighbor_graph(
        self,
        embedded_ids: list[str],
        matrix,
        options: TopologyBuildOptions,
    ) -> tuple[list[dict[str, Any]], dict[str, list[tuple[str, float]]]]:
        if len(matrix) <= 1:
            return [], {note_id: [] for note_id in embedded_ids}
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as exc:  # pragma: no cover - environment specific
            raise TopologyBuildError(
                "embedding neighbor search requires scikit-learn. Install with: pip install 'knowledge-hub-cli[topology]'"
            ) from exc

        count = len(matrix)
        neighbor_count = max(2, min(count, int(options.neighbors) + 1))
        nn = NearestNeighbors(metric="cosine", n_neighbors=neighbor_count)
        nn.fit(matrix)
        distances, indices = nn.kneighbors(matrix)

        neighbor_sets: dict[int, set[int]] = defaultdict(set)
        sims: dict[tuple[int, int], float] = {}
        soft_neighbors: dict[str, list[tuple[str, float]]] = {note_id: [] for note_id in embedded_ids}
        for row_idx in range(count):
            for col_idx, dist in zip(indices[row_idx], distances[row_idx]):
                if col_idx == row_idx:
                    continue
                similarity = max(0.0, 1.0 - float(dist))
                if similarity < float(options.similarity_threshold):
                    continue
                neighbor_sets[row_idx].add(int(col_idx))
                sims[(row_idx, int(col_idx))] = similarity
                soft_neighbors[embedded_ids[row_idx]].append(
                    (embedded_ids[int(col_idx)], round(float(similarity), 6))
                )

        edges: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for left in range(count):
            for right in neighbor_sets.get(left, set()):
                if left not in neighbor_sets.get(right, set()):
                    continue
                pair = tuple(sorted((left, right)))
                if pair in seen:
                    continue
                seen.add(pair)
                weight = max(sims.get((left, right), 0.0), sims.get((right, left), 0.0))
                edges.append(
                    {
                        "source": embedded_ids[pair[0]],
                        "target": embedded_ids[pair[1]],
                        "type": "embedding_neighbor",
                        "weight": round(float(weight), 6),
                    }
                )
        return edges, soft_neighbors

    def _assign_clusters(
        self,
        nodes: dict[str, dict[str, Any]],
        embedding_edges: list[dict[str, Any]],
        *,
        min_cluster_size: int,
    ) -> dict[str, str]:
        adjacency: dict[str, set[str]] = {note_id: set() for note_id in nodes}
        for edge in embedding_edges:
            adjacency[edge["source"]].add(edge["target"])
            adjacency[edge["target"]].add(edge["source"])

        cluster_map: dict[str, str] = {}
        cluster_index = 1
        visited: set[str] = set()
        for note_id in sorted(nodes):
            if note_id in visited:
                continue
            stack = [note_id]
            members: list[str] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                members.append(current)
                stack.extend(sorted(adjacency.get(current, set()) - visited))
            prefix = "cluster" if len(members) >= max(1, min_cluster_size) else "singleton"
            cluster_id = f"{prefix}-{cluster_index:04d}"
            cluster_index += 1
            for member in members:
                cluster_map[member] = cluster_id
        return cluster_map

    def _compute_importance(
        self,
        nodes: dict[str, dict[str, Any]],
        wiki_edges: list[dict[str, Any]],
        embedding_edges: list[dict[str, Any]],
    ) -> dict[str, float]:
        scores = {note_id: 0.0 for note_id in nodes}
        for edge in wiki_edges:
            scores[edge["source"]] += float(edge["weight"])
            scores[edge["target"]] += float(edge["weight"])
        for edge in embedding_edges:
            weight = float(edge["weight"])
            scores[edge["source"]] += weight
            scores[edge["target"]] += weight
        return scores

    def _build_cluster_metadata(
        self,
        nodes: dict[str, dict[str, Any]],
        cluster_map: dict[str, str],
        importance: dict[str, float],
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for note_id, cluster_id in cluster_map.items():
            grouped[cluster_id].append(note_id)

        payload: dict[str, dict[str, Any]] = {}
        for cluster_id, members in grouped.items():
            member_tags: Counter[str] = Counter()
            member_terms: Counter[str] = Counter()
            term_support: dict[str, set[str]] = defaultdict(set)
            for note_id in members:
                note = nodes[note_id]
                note_weight = 1.0 + float(importance.get(note_id, 0.0))
                for raw_tag in note["tags"]:
                    clean_tag = _clean_cluster_term(raw_tag)
                    if not clean_tag:
                        continue
                    member_tags[clean_tag] += 1
                    member_terms[clean_tag] += 2.5 * note_weight
                    term_support[clean_tag].add(note_id)
                for token in _tokenize(note["title"]):
                    clean_token = _clean_cluster_term(token)
                    if not clean_token:
                        continue
                    member_terms[clean_token] += 2.0 * note_weight
                    term_support[clean_token].add(note_id)
                for token in _path_tokens(note["path"]):
                    clean_token = _clean_cluster_term(token)
                    if not clean_token:
                        continue
                    member_terms[clean_token] += 1.2 * note_weight
                    term_support[clean_token].add(note_id)

            representative = max(members, key=lambda item: (importance.get(item, 0.0), nodes[item]["title"]))
            minimum_support = 2 if len(members) >= 4 else 1
            ranked_terms = sorted(
                (
                    (
                        len(term_support.get(term, set())),
                        float(score),
                        term,
                    )
                    for term, score in member_terms.items()
                    if len(term_support.get(term, set())) >= minimum_support
                ),
                reverse=True,
            )
            label_parts = [term for _support, _score, term in ranked_terms[:2]]
            label = " / ".join(label_parts) if label_parts else nodes[representative]["title"]
            payload[cluster_id] = {
                "id": cluster_id,
                "label": label,
                "size": len(members),
                "representativeNoteId": representative,
                "topTags": [tag for tag, _ in member_tags.most_common(5)],
            }
        return payload

    def _compute_bridge_scores(
        self,
        nodes: dict[str, dict[str, Any]],
        wiki_edges: list[dict[str, Any]],
        soft_neighbors: dict[str, list[tuple[str, float]]],
        cluster_map: dict[str, str],
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        scores: dict[str, float] = {}
        external_cluster_weights: dict[str, dict[str, float]] = {note_id: {} for note_id in nodes}
        wiki_adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in wiki_edges:
            wiki_adjacency[edge["source"]].append(edge["target"])
            wiki_adjacency[edge["target"]].append(edge["source"])

        for note_id in nodes:
            current_cluster = cluster_map[note_id]
            total_weight = 0.0
            cross_weight = 0.0
            cluster_weights: dict[str, float] = defaultdict(float)

            for neighbor_id, similarity in soft_neighbors.get(note_id, []):
                weight = float(similarity)
                if weight <= 0:
                    continue
                neighbor_cluster = cluster_map.get(neighbor_id)
                total_weight += weight
                if neighbor_cluster and neighbor_cluster != current_cluster:
                    cross_weight += weight
                    cluster_weights[neighbor_cluster] += weight

            for neighbor_id in wiki_adjacency.get(note_id, []):
                neighbor_cluster = cluster_map.get(neighbor_id)
                weight = 1.25
                total_weight += weight
                if neighbor_cluster and neighbor_cluster != current_cluster:
                    cross_weight += weight
                    cluster_weights[neighbor_cluster] += weight

            if total_weight <= 0:
                scores[note_id] = 0.0
                continue

            external_clusters = len(cluster_weights)
            diversity_bonus = 0.12 * max(0, external_clusters - 1)
            score = min(1.0, (cross_weight / total_weight) + diversity_bonus)
            scores[note_id] = round(float(score), 6)
            external_cluster_weights[note_id] = {
                cluster_id: round(weight / total_weight, 6)
                for cluster_id, weight in cluster_weights.items()
            }
        return scores, external_cluster_weights

    def _build_bridge_hint_edges(
        self,
        nodes: dict[str, dict[str, Any]],
        cluster_map: dict[str, str],
        cluster_meta: dict[str, dict[str, Any]],
        bridge_scores: dict[str, float],
        external_cluster_weights: dict[str, dict[str, float]],
    ) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for note_id, cluster_weights in external_cluster_weights.items():
            if bridge_scores.get(note_id, 0.0) < 0.35:
                continue
            ranked_clusters = sorted(cluster_weights.items(), key=lambda item: (-item[1], item[0]))[:2]
            for cluster_id, cluster_weight in ranked_clusters:
                representative = cluster_meta.get(cluster_id, {}).get("representativeNoteId")
                if not representative or representative == note_id:
                    continue
                pair = (note_id, representative)
                if pair in seen:
                    continue
                seen.add(pair)
                edges.append(
                    {
                        "source": note_id,
                        "target": representative,
                        "type": "bridge_hint",
                        "weight": round(float(max(bridge_scores.get(note_id, 0.0), cluster_weight)), 6),
                    }
                )
        return edges

    def _resolve_projection_kind(self, requested: str) -> str:
        lowered = str(requested or "auto").strip().lower()
        if lowered == "auto":
            try:
                _load_projection_model("umap", 0.08)
                return "umap"
            except TopologyBuildError:
                return "pca"
        if lowered not in {"umap", "pca"}:
            raise TopologyBuildError(f"unsupported projection: {requested}")
        return lowered

    def _project(self, matrix, options: TopologyBuildOptions):
        np = _load_numpy()
        count = len(matrix)
        if count == 0:
            return np.zeros((0, 3))
        if count == 1:
            return np.zeros((1, 3))

        projection_kind = self._resolve_projection_kind(options.projection)
        if projection_kind == "pca":
            try:
                from sklearn.decomposition import PCA
            except ImportError as exc:  # pragma: no cover - environment specific
                raise TopologyBuildError(
                    "projection=pca requires scikit-learn. Install with: pip install 'knowledge-hub-cli[topology]'"
                ) from exc
            reducer = PCA(n_components=min(3, count, int(matrix.shape[1])), random_state=42)
        else:
            reducer = _load_projection_model(projection_kind, options.min_dist)
        reduced = reducer.fit_transform(matrix)
        reduced = np.asarray(reduced, dtype=float)
        if reduced.ndim == 1:
            reduced = reduced.reshape(-1, 1)
        if reduced.shape[1] < 3:
            padding = np.zeros((reduced.shape[0], 3 - reduced.shape[1]), dtype=float)
            reduced = np.concatenate([reduced, padding], axis=1)
        scale = float(np.max(np.abs(reduced))) if reduced.size else 0.0
        if scale > 0:
            reduced = reduced / scale
        if options.spherize:
            norms = np.linalg.norm(reduced, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            reduced = reduced / norms
        return reduced
