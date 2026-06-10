from __future__ import annotations

from typing import Any, Final

from knowledge_hub.core.models import SearchResult


ACTIVE_VECTOR_PAPER_LOOKUP_MODE: Final = "active-vector-paper"


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _resolved_paper_ids(
    *,
    query_frame: dict[str, Any] | None,
    metadata_filter: dict[str, Any] | None,
) -> list[str]:
    frame = dict(query_frame or {})
    scoped = dict(metadata_filter or {})
    values: list[Any] = [
        scoped.get("arxiv_id"),
        scoped.get("paper_id"),
        *list(frame.get("resolved_source_ids") or []),
        *list(frame.get("resolvedSourceIds") or []),
    ]
    resolved: list[str] = []
    for value in values:
        token = _clean_text(value)
        if token and token not in resolved:
            resolved.append(token)
    return resolved


def _paper_vector_filter(paper_id: str, metadata_filter: dict[str, Any] | None) -> dict[str, Any]:
    scoped = dict(metadata_filter or {})
    return {
        "source_type": "paper",
        "arxiv_id": _clean_text(scoped.get("arxiv_id") or scoped.get("paper_id") or paper_id),
    }


def _search_result(
    *,
    doc_id: str,
    document: str,
    metadata: dict[str, Any],
    paper_id: str,
) -> SearchResult:
    normalized_metadata = dict(metadata)
    normalized_metadata["source_type"] = "paper"
    normalized_metadata["arxiv_id"] = _clean_text(normalized_metadata.get("arxiv_id") or paper_id)
    normalized_metadata["paper_id"] = _clean_text(normalized_metadata.get("paper_id") or paper_id)
    normalized_metadata["source_id"] = _clean_text(normalized_metadata.get("source_id") or paper_id)
    normalized_metadata["source_ref"] = _clean_text(normalized_metadata.get("source_ref") or paper_id)
    normalized_metadata["chunk_id"] = _clean_text(normalized_metadata.get("chunk_id") or doc_id)
    if not _clean_text(normalized_metadata.get("title")):
        normalized_metadata["title"] = paper_id
    return SearchResult(
        document=document,
        metadata=normalized_metadata,
        distance=0.0,
        score=1.0,
        document_id=doc_id,
        semantic_score=1.0,
        lexical_score=1.0,
        retrieval_mode=ACTIVE_VECTOR_PAPER_LOOKUP_MODE,
        lexical_extras={
            "quality_flag": "ok",
            "source_trust_score": 0.95,
            "ranking_signals": {"resolved_paper_active_vector": 1.0},
            "top_ranking_signals": [{"name": "resolved_paper_active_vector", "value": 1.0}],
        },
    )


def source_backed_paper_lookup_results(
    *,
    searcher: Any,
    paper_family: str,
    query_frame: dict[str, Any] | None,
    metadata_filter: dict[str, Any] | None,
    limit: int,
) -> list[SearchResult]:
    if _clean_text(paper_family).lower() != "paper_lookup":
        return []
    resolved_ids = _resolved_paper_ids(query_frame=query_frame, metadata_filter=metadata_filter)
    if len(resolved_ids) != 1:
        return []
    paper_id = resolved_ids[0]
    database = getattr(searcher, "database", None)
    get_documents = getattr(database, "get_documents", None)
    if not callable(get_documents):
        return []
    raw = get_documents(
        filter_dict=_paper_vector_filter(paper_id, metadata_filter),
        limit=max(1, int(limit)),
        include_ids=True,
        include_documents=True,
        include_metadatas=True,
    )
    documents = list(raw.get("documents") or [])
    metadatas = [dict(item or {}) for item in list(raw.get("metadatas") or [])]
    ids = [_clean_text(item) for item in list(raw.get("ids") or [])]
    results: list[SearchResult] = []
    for index, document in enumerate(documents):
        body = str(document or "").strip()
        if not body:
            continue
        metadata = metadatas[index] if index < len(metadatas) else {}
        doc_id = ids[index] if index < len(ids) and ids[index] else f"paper_{paper_id}_{index}"
        results.append(
            _search_result(
                doc_id=doc_id,
                document=body,
                metadata=metadata,
                paper_id=paper_id,
            )
        )
    return results
