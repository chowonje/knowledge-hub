from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from typing import Any

from knowledge_hub.ai.answer_contracts import NON_EVIDENCE_SOURCE_SCHEMES, NON_EVIDENCE_SOURCE_TYPES


COMPARE_PACKET_SCHEMA = "knowledge-hub.compare-packet.v1"


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _hash_text(*parts: Any, length: int = 24) -> str:
    text = "\n".join(str(part or "") for part in parts if str(part or "").strip())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[: max(8, int(length))]


def _source_scheme(source_id: Any) -> str:
    token = _clean_text(source_id).lower()
    if not token or ":" not in token:
        return ""
    scheme = token.split(":", 1)[0]
    return scheme if re.fullmatch(r"[a-z_][a-z0-9_+.-]*", scheme) else ""


def _is_non_evidence_ref(item: dict[str, Any]) -> bool:
    source_type = _clean_text(item.get("source_type") or item.get("sourceType")).lower()
    if source_type in NON_EVIDENCE_SOURCE_TYPES:
        return True
    scheme = _source_scheme(item.get("source_id") or item.get("sourceId") or item.get("spanRef"))
    return bool(scheme and scheme in NON_EVIDENCE_SOURCE_SCHEMES)


def _span_ref(item: dict[str, Any], *, fallback_index: int) -> dict[str, Any]:
    source_id = _clean_text(item.get("source_id") or item.get("sourceId") or item.get("target") or item.get("spanRef"))
    span_ref = _clean_text(item.get("span_ref") or item.get("spanRef")) or f"span:{fallback_index}"
    return {
        "spanRef": span_ref,
        "sourceId": source_id,
        "sourceType": _clean_text(item.get("source_type") or item.get("sourceType")),
        "contentHash": _clean_text(item.get("content_hash") or item.get("contentHash") or item.get("source_content_hash")),
        "quote": str(item.get("quote") or item.get("text") or item.get("excerpt") or "")[:500],
    }


def build_compare_packet_contract(
    *,
    query: str,
    dimensions: list[dict[str, Any]],
    retrieval_signals: list[dict[str, Any]] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_dimensions: list[dict[str, Any]] = []
    excluded_non_evidence = 0
    unknown_count = 0
    conflict_count = 0

    for index, raw_dimension in enumerate(dimensions, start=1):
        dimension = dict(raw_dimension or {})
        supporting_spans: list[dict[str, Any]] = []
        for span_index, raw_span in enumerate(list(dimension.get("supporting_spans") or dimension.get("supportingSpans") or []), start=1):
            span_item = dict(raw_span or {})
            if _is_non_evidence_ref(span_item):
                excluded_non_evidence += 1
                continue
            supporting_spans.append(_span_ref(span_item, fallback_index=span_index))

        status = _clean_text(dimension.get("status") or dimension.get("comparisonStatus") or "unknown").lower()
        if status not in {"supported", "conflict", "unknown", "insufficient"}:
            status = "unknown"
        if status == "conflict":
            conflict_count += 1
        if status in {"unknown", "insufficient"} or not supporting_spans:
            unknown_count += 1

        normalized_dimensions.append(
            {
                "dimensionId": _clean_text(dimension.get("dimension_id") or dimension.get("dimensionId")) or f"dim:{index}",
                "label": _clean_text(dimension.get("label") or dimension.get("name") or f"Dimension {index}"),
                "leftClaim": _clean_text(dimension.get("left_claim") or dimension.get("leftClaim")),
                "rightClaim": _clean_text(dimension.get("right_claim") or dimension.get("rightClaim")),
                "comparisonStatus": status,
                "supportingSpans": supporting_spans,
                "notes": _clean_text(dimension.get("notes")),
            }
        )

    answerable = bool(normalized_dimensions) and unknown_count < len(normalized_dimensions)
    packet_id = _hash_text(query, [item["dimensionId"] for item in normalized_dimensions], [item["comparisonStatus"] for item in normalized_dimensions])
    created_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema": COMPARE_PACKET_SCHEMA,
        "packetId": packet_id,
        "packet_id": packet_id,
        "query": str(query or ""),
        "createdAt": created_at,
        "created_at": created_at,
        "dimensions": normalized_dimensions,
        "retrievalSignals": [dict(item or {}) for item in list(retrieval_signals or [])],
        "policy": dict(policy or {}),
        "coverage": {
            "dimensionCount": len(normalized_dimensions),
            "supportedDimensionCount": sum(1 for item in normalized_dimensions if item["comparisonStatus"] == "supported"),
            "conflictDimensionCount": conflict_count,
            "unknownDimensionCount": unknown_count,
            "supportingSpanCount": sum(len(item["supportingSpans"]) for item in normalized_dimensions),
            "excludedNonEvidenceSpanCount": excluded_non_evidence,
            "answerable": bool(answerable),
        },
    }


__all__ = ["COMPARE_PACKET_SCHEMA", "build_compare_packet_contract"]
