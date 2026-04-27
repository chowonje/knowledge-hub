from __future__ import annotations

import hashlib
from typing import Any, Callable

from knowledge_hub.core.models import SearchResult
from knowledge_hub.knowledge.features import compute_freshness_score


_SOURCE_HASH_KEYS = (
    "source_content_hash",
    "content_hash",
    "contentHash",
    "content_sha1",
    "content_sha256",
    "source_hash",
    "sourceHash",
    "document_hash",
    "parse_hash",
    "chunk_hash",
)


def _clean_token(value: Any) -> str:
    return str(value or "").strip()


def _first_nonempty(*values: Any) -> str:
    for value in values:
        token = _clean_token(value)
        if token:
            return token
    return ""


def _snippet_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _source_content_hash(metadata: dict[str, Any]) -> str:
    return _first_nonempty(*(metadata.get(key) for key in _SOURCE_HASH_KEYS))


def _safe_text_span(value: str) -> tuple[int | None, int | None]:
    if not str(value or "").strip():
        return None, None
    return 0, len(str(value or ""))


def _derivative_source(metadata: dict[str, Any], extras: dict[str, Any]) -> dict[str, Any]:
    memory_provenance = dict(extras.get("memory_provenance") or {})
    payload = {
        "cardId": _first_nonempty(metadata.get("card_id"), metadata.get("cardId")),
        "anchorId": _first_nonempty(metadata.get("anchor_id"), metadata.get("anchorId")),
        "memoryId": _first_nonempty(metadata.get("memory_id"), metadata.get("source_memory_id")),
        "claimId": _first_nonempty(metadata.get("claim_id"), metadata.get("claimId")),
        "unitId": _first_nonempty(metadata.get("unit_id"), metadata.get("unitId")),
        "memoryProvenance": memory_provenance,
    }
    return {key: value for key, value in payload.items() if value}


def _evidence_kind(*, derivative_source: dict[str, Any], metadata: dict[str, Any], extras: dict[str, Any]) -> str:
    if dict(extras.get("memory_provenance") or {}):
        return "memory_hint"
    if derivative_source:
        return "derived_anchor"
    if _first_nonempty(metadata.get("span_locator"), metadata.get("chunk_id"), metadata.get("chunk_span")):
        return "raw_span"
    return "raw_source"


def answer_evidence_item(
    result: SearchResult,
    *,
    parent_ctx_by_result: dict[str, dict[str, Any]],
    result_id_fn: Callable[[SearchResult], str],
    normalize_source_type_fn: Callable[[str], str],
    safe_float_fn: Callable[[Any, float], float],
) -> dict[str, Any]:
    extras = dict(result.lexical_extras or {})
    parent_ctx = parent_ctx_by_result.get(result_id_fn(result), {})
    metadata = dict(result.metadata or {})
    arxiv_id = str(metadata.get("arxiv_id") or metadata.get("paper_id") or "").strip()
    file_path = str(metadata.get("file_path") or "").strip()
    source_url = str(metadata.get("url") or metadata.get("source_url") or metadata.get("canonical_url") or "").strip()
    citation_target = arxiv_id or file_path or source_url or str(metadata.get("title") or "Untitled")
    source_ref = _first_nonempty(
        metadata.get("source_ref"),
        metadata.get("sourceRef"),
        arxiv_id,
        file_path,
        source_url,
        metadata.get("canonical_url"),
        metadata.get("document_id"),
        result.document_id,
        metadata.get("title"),
    )
    source_id = _first_nonempty(
        metadata.get("source_id"),
        metadata.get("sourceId"),
        metadata.get("document_id"),
        metadata.get("note_id"),
        metadata.get("paper_id"),
        arxiv_id,
        result.document_id,
        source_ref,
    )
    span_locator = _first_nonempty(
        metadata.get("span_locator"),
        metadata.get("spanLocator"),
        parent_ctx.get("chunk_span"),
        metadata.get("chunk_span"),
        metadata.get("chunk_id"),
        metadata.get("unit_id"),
        metadata.get("section_path"),
        metadata.get("section_title"),
        result.document_id,
    )
    source_hash = _source_content_hash(metadata)
    source_hash_kind = "source"
    if not source_hash:
        source_hash = _snippet_hash(result.document)
        source_hash_kind = "retrieved_document"
    char_start = metadata.get("char_start")
    char_end = metadata.get("char_end")
    if (char_start is None or char_end is None) and source_hash_kind == "retrieved_document":
        char_start, char_end = _safe_text_span(result.document)
    snippet_hash = _first_nonempty(metadata.get("snippet_hash"), metadata.get("snippetHash"), _snippet_hash(result.document))
    derivative_source = _derivative_source(metadata, extras)
    evidence_kind = _evidence_kind(derivative_source=derivative_source, metadata=metadata, extras=extras)
    source_trace = {
        "sourceRef": source_ref,
        "sourceId": source_id,
        "contentHash": source_hash,
        "spanLocator": span_locator,
        "snippetHash": snippet_hash,
        "evidenceKind": evidence_kind,
        "derivativeSource": derivative_source,
        "contentHashAvailable": bool(source_hash),
        "contentHashKind": source_hash_kind,
    }
    published_at = str(metadata.get("published_at") or "").strip()
    updated_at = str(metadata.get("updated_at") or "").strip()
    freshness_score = compute_freshness_score(
        published_at=published_at or metadata.get("document_date") or metadata.get("event_date"),
        updated_at=updated_at or metadata.get("observed_at"),
    )
    return {
        "title": metadata.get("title", "Untitled"),
        "file_path": file_path,
        "source_type": metadata.get("source_type", ""),
        "normalized_source_type": str(
            extras.get("normalized_source_type")
            or normalize_source_type_fn(str(metadata.get("source_type", "") or ""))
            or ""
        ),
        "score": result.score,
        "semantic_score": result.semantic_score,
        "lexical_score": result.lexical_score,
        "retrieval_mode": result.retrieval_mode,
        "parent_id": parent_ctx.get("parent_id", ""),
        "parent_label": parent_ctx.get("parent_label", ""),
        "parent_chunk_span": parent_ctx.get("chunk_span", ""),
        "quality_flag": str(extras.get("quality_flag") or "unscored"),
        "source_trust_score": safe_float_fn(extras.get("source_trust_score"), 0.0),
        "reference_role": str(extras.get("reference_role") or ""),
        "reference_tier": str(extras.get("reference_tier") or ""),
        "section_path": str(metadata.get("section_path") or metadata.get("section_title") or ""),
        "document_date": str(metadata.get("document_date") or ""),
        "event_date": str(metadata.get("event_date") or ""),
        "observed_at": str(metadata.get("observed_at") or ""),
        "published_at": published_at,
        "updated_at": updated_at,
        "evidence_window": str(metadata.get("evidence_window") or ""),
        "freshness_score": round(float(freshness_score), 6),
        "duplicate_collapsed": bool(extras.get("duplicate_collapsed")),
        "top_ranking_signals": list(extras.get("top_ranking_signals") or []),
        "ranking_signals": dict(extras.get("ranking_signals") or {}),
        "memory_provenance": dict(extras.get("memory_provenance") or {}),
        "arxiv_id": arxiv_id,
        "source_url": source_url,
        "citation_target": citation_target,
        "source_ref": source_ref,
        "source_id": source_id,
        "source_content_hash": source_hash,
        "source_content_hash_kind": source_hash_kind,
        "span_locator": span_locator,
        "char_start": char_start,
        "char_end": char_end,
        "snippet_hash": snippet_hash,
        "evidence_kind": evidence_kind,
        "derivative_source": derivative_source,
        "source_trace": source_trace,
        "excerpt": result.document[:200] + ("..." if len(result.document) > 200 else ""),
    }


def summarize_answer_signals(
    evidence: list[dict[str, Any]],
    *,
    contradicting_beliefs: list[dict[str, Any]] | None = None,
    safe_float_fn: Callable[[Any, float], float],
) -> dict[str, Any]:
    quality_counts = {"ok": 0, "needs_review": 0, "reject": 0, "unscored": 0}
    specialist_count = 0
    contradiction_sources = 0
    high_trust_count = 0
    fresh_evidence_count = 0
    temporal_grounded_count = 0
    memory_provenance_count = 0

    for item in evidence:
        flag = str(item.get("quality_flag") or "unscored").strip().lower() or "unscored"
        quality_counts[flag] = quality_counts.get(flag, 0) + 1
        if str(item.get("reference_tier") or "").strip().lower() == "specialist":
            specialist_count += 1
        if safe_float_fn((item.get("ranking_signals") or {}).get("contradiction_penalty"), 0.0) >= 0.1:
            contradiction_sources += 1
        if safe_float_fn(item.get("source_trust_score"), 0.0) >= 0.85:
            high_trust_count += 1
        if safe_float_fn(item.get("freshness_score"), 0.0) >= 0.55:
            fresh_evidence_count += 1
        if any(
            str(item.get(name) or "").strip()
            for name in ("event_date", "document_date", "published_at", "observed_at", "evidence_window")
        ):
            temporal_grounded_count += 1
        if dict(item.get("memory_provenance") or {}):
            memory_provenance_count += 1

    strongest_quality = "unscored"
    for candidate in ("ok", "needs_review", "unscored", "reject"):
        if quality_counts.get(candidate, 0):
            strongest_quality = candidate
            break

    caution_required = (
        quality_counts.get("ok", 0) == 0
        or quality_counts.get("reject", 0) > 0
        or contradiction_sources > 0
        or bool(contradicting_beliefs)
    )
    return {
        "total_sources": len(evidence),
        "quality_counts": quality_counts,
        "preferred_sources": quality_counts.get("ok", 0),
        "specialist_reference_count": specialist_count,
        "high_trust_source_count": high_trust_count,
        "fresh_evidence_count": fresh_evidence_count,
        "temporal_grounded_count": temporal_grounded_count,
        "memory_provenance_count": memory_provenance_count,
        "contradictory_source_count": contradiction_sources,
        "contradicting_belief_count": len(contradicting_beliefs or []),
        "strongest_quality_flag": strongest_quality,
        "caution_required": bool(caution_required),
    }
