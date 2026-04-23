from __future__ import annotations

from typing import Any, Callable

from knowledge_hub.core.models import SearchResult


def search_with_paper_memory_prefilter(
    searcher: Any,
    *,
    query: str,
    top_k: int,
    source_type: str | None,
    retrieval_mode: str,
    alpha: float,
    min_score: float,
    requested_mode: str,
    metadata_filter: dict[str, Any] | None = None,
    resolve_paper_memory_prefilter_fn: Callable[..., dict[str, Any]],
    merge_search_results_fn: Callable[[list[SearchResult]], list[SearchResult]],
) -> tuple[list[SearchResult], dict[str, Any]]:
    diagnostics = resolve_paper_memory_prefilter_fn(
        searcher.sqlite_db,
        query=query,
        source_type=source_type,
        requested_mode=requested_mode,
        limit=min(max(1, int(top_k)), 5),
    )
    scoped_filter = dict(metadata_filter or {})
    if not diagnostics.get("applied"):
        results = searcher.search(
            query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            metadata_filter=scoped_filter,
        )
        return [r for r in results if r.score >= min_score], diagnostics

    candidate_ids = [
        str(item).strip()
        for item in diagnostics.get("matchedPaperIds") or []
        if str(item).strip()
    ]
    scoped_paper_id = str(scoped_filter.get("arxiv_id") or "").strip()
    if scoped_paper_id:
        if scoped_paper_id in candidate_ids:
            candidate_ids = [scoped_paper_id]
        else:
            diagnostics["applied"] = False
            diagnostics["fallbackUsed"] = True
            diagnostics["reason"] = "prefilter_scope_mismatch"
            results = searcher.search(
                query,
                top_k=top_k,
                source_type=source_type,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
                metadata_filter=scoped_filter,
            )
            return [r for r in results if r.score >= min_score], diagnostics

    collected: list[SearchResult] = []
    for paper_id in candidate_ids:
        narrowed_filter = dict(scoped_filter)
        narrowed_filter.setdefault("source_type", "paper")
        narrowed_filter["arxiv_id"] = paper_id
        collected.extend(
            searcher.search(
                query,
                top_k=top_k,
                source_type="paper",
                retrieval_mode=retrieval_mode,
                alpha=alpha,
                metadata_filter=narrowed_filter,
            )
        )

    merged = merge_search_results_fn(collected)
    filtered = [r for r in merged if r.score >= min_score]
    if filtered:
        return filtered, diagnostics

    diagnostics["applied"] = False
    diagnostics["fallbackUsed"] = True
    diagnostics["reason"] = "prefilter_no_ranked_results"
    results = searcher.search(
        query,
        top_k=top_k,
        source_type=source_type,
        retrieval_mode=retrieval_mode,
        alpha=alpha,
        metadata_filter=scoped_filter,
    )
    return [r for r in results if r.score >= min_score], diagnostics
