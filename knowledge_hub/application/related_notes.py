from __future__ import annotations

import re
from typing import Any

from knowledge_hub.ai.retrieval_fit import normalize_source_type


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_가-힣]{2,}", str(text or ""))}


def _path_dir(path: str) -> str:
    token = str(path or "").strip().replace("\\", "/")
    if "/" not in token:
        return ""
    return token.rsplit("/", 1)[0]


def build_related_note_suggestions(
    results: list[Any],
    *,
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    vault_results = [
        item
        for item in list(results or [])
        if normalize_source_type((getattr(item, "metadata", {}) or {}).get("source_type")) == "vault"
    ]
    if len(vault_results) <= 1:
        return []

    seed_results = vault_results[: min(3, len(vault_results))]
    primary_path = str((getattr(seed_results[0], "metadata", {}) or {}).get("file_path") or "").strip()
    query_tokens = _tokenize(query)
    scored: dict[str, dict[str, Any]] = {}

    for candidate in vault_results[1:]:
        metadata = dict(getattr(candidate, "metadata", {}) or {})
        file_path = str(metadata.get("file_path") or "").strip()
        title = str(metadata.get("title") or "Untitled").strip() or "Untitled"
        if not file_path or file_path == primary_path:
            continue

        score = 0.0
        reasons: list[str] = []
        candidate_dir = _path_dir(file_path)
        candidate_cluster = str(metadata.get("cluster_id") or "").strip()
        candidate_alias_tokens = _tokenize(" ".join(str(item) for item in (metadata.get("aliases") or [])))
        candidate_title_tokens = _tokenize(title)

        for seed in seed_results:
            seed_meta = dict(getattr(seed, "metadata", {}) or {})
            seed_links = {str(item).strip() for item in (seed_meta.get("links") or []) if str(item).strip()}
            seed_path = str(seed_meta.get("file_path") or "").strip()
            seed_dir = _path_dir(seed_path)
            seed_cluster = str(seed_meta.get("cluster_id") or "").strip()

            if file_path in seed_links or title in seed_links:
                score += 3.0
                reasons.append("metadata_link")
            if candidate_dir and seed_dir and candidate_dir == seed_dir:
                score += 1.0
                reasons.append("path_proximity")
            if candidate_cluster and seed_cluster and candidate_cluster == seed_cluster:
                score += 1.0
                reasons.append("cluster_overlap")

        overlap = len(query_tokens & (candidate_title_tokens | candidate_alias_tokens))
        if overlap:
            score += min(1.0, 0.35 * overlap)
            reasons.append("query_overlap")

        if score <= 0.0:
            continue

        scored[file_path] = {
            "title": title,
            "file_path": file_path,
            "source_type": "vault",
            "score": round(score, 6),
            "reasons": sorted(set(reasons)),
        }

    ranked = sorted(scored.values(), key=lambda item: (-float(item["score"]), str(item["title"])))
    return ranked[: max(0, int(limit))]


__all__ = ["build_related_note_suggestions"]
