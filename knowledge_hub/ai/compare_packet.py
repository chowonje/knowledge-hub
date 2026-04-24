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


def _claim_card_id(item: dict[str, Any]) -> str:
    return _clean_text(item.get("claimCardId") or item.get("claim_card_id") or item.get("claimId") or item.get("claim_id"))


def _group_claim_ids(group: dict[str, Any]) -> list[str]:
    return [_clean_text(value) for value in list(group.get("claimCardIds") or group.get("claim_card_ids") or []) if _clean_text(value)]


def _label_from_frame(frame: dict[str, Any]) -> str:
    for key in ("metric", "dataset", "task", "comparator"):
        value = _clean_text(frame.get(key))
        if value:
            return value
    return ""


def _claim_text(card: dict[str, Any]) -> str:
    parts = [
        _clean_text(card.get("summaryText") or card.get("summary_text")),
        _clean_text(card.get("resultValueText") or card.get("result_value_text")),
        _clean_text(card.get("resultDirection") or card.get("result_direction")),
    ]
    return " | ".join(part for part in parts if part)


def _claim_supporting_spans(card: dict[str, Any]) -> list[dict[str, Any]]:
    anchor_ids = [_clean_text(value) for value in list(card.get("evidenceAnchorIds") or card.get("evidence_anchor_ids") or [])]
    excerpts = [_clean_text(value) for value in list(card.get("anchorExcerpts") or card.get("anchor_excerpts") or [])]
    if not anchor_ids and not excerpts:
        return []
    count = max(len(anchor_ids), len(excerpts), 1)
    spans: list[dict[str, Any]] = []
    for index in range(count):
        spans.append(
            {
                "spanRef": anchor_ids[index] if index < len(anchor_ids) and anchor_ids[index] else f"{_claim_card_id(card)}:anchor:{index + 1}",
                "sourceId": _clean_text(card.get("sourceId") or card.get("source_id")),
                "sourceType": _clean_text(card.get("sourceKind") or card.get("source_kind") or "paper"),
                "quote": excerpts[index] if index < len(excerpts) else _clean_text(card.get("summaryText") or card.get("summary_text")),
            }
        )
    return spans


def _ordered_group_cards(*, group: dict[str, Any], claim_cards_by_id: dict[str, dict[str, Any]], resolved_source_ids: list[str]) -> list[dict[str, Any]]:
    cards = [claim_cards_by_id[claim_id] for claim_id in _group_claim_ids(group) if claim_id in claim_cards_by_id]
    if not cards:
        return []
    order = {source_id: index for index, source_id in enumerate(resolved_source_ids)}
    return sorted(
        cards,
        key=lambda item: (
            order.get(_clean_text(item.get("sourceId") or item.get("source_id")), len(order) + 1),
            _clean_text(item.get("sourceId") or item.get("source_id")),
            _claim_card_id(item),
        ),
    )


def build_compare_packet_from_runtime(
    *,
    query: str,
    source_type: str,
    family: str,
    runtime_execution: dict[str, Any],
    query_frame: dict[str, Any],
    claim_cards: list[dict[str, Any]],
    claim_alignment: dict[str, Any],
    evidence_policy: dict[str, Any] | None = None,
    comparison_verification: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if _clean_text(source_type).lower() != "paper":
        return None
    if _clean_text(family).lower() != "paper_compare":
        return None
    if _clean_text(runtime_execution.get("used")).lower() != "ask_v2":
        return None
    cards_by_id = {_claim_card_id(item): dict(item or {}) for item in claim_cards if _claim_card_id(dict(item or {}))}
    groups = [dict(item or {}) for item in list((claim_alignment or {}).get("groups") or [])]
    if not cards_by_id or not groups:
        return None

    resolved_source_ids = [_clean_text(value) for value in list(query_frame.get("resolved_source_ids") or query_frame.get("resolvedSourceIds") or [])]
    verification_conflicts = {
        _clean_text(item.get("groupKey"))
        for item in list((comparison_verification or {}).get("conflicts") or [])
        if _clean_text(item.get("groupKey"))
    }
    dimensions: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        group_cards = _ordered_group_cards(group=group, claim_cards_by_id=cards_by_id, resolved_source_ids=resolved_source_ids)
        if not group_cards:
            continue
        frame = dict(group.get("canonicalFrame") or group.get("frame") or {})
        supporting_spans: list[dict[str, Any]] = []
        for card in group_cards:
            supporting_spans.extend(_claim_supporting_spans(card))
        distinct_cards: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        for card in group_cards:
            source_id = _clean_text(card.get("sourceId") or card.get("source_id"))
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)
            distinct_cards.append(card)
        group_key = _clean_text(group.get("groupKey")) or f"compare-group:{index}"
        if int(group.get("conflictingClaimCount") or 0) > 0 or group_key in verification_conflicts:
            status = "conflict"
        elif len(distinct_cards) >= 2 and supporting_spans:
            status = "supported"
        else:
            status = "insufficient"
        dimensions.append(
            {
                "dimensionId": group_key,
                "label": _label_from_frame(frame) or f"Comparison {index}",
                "leftClaim": _claim_text(distinct_cards[0]) if distinct_cards else "",
                "rightClaim": _claim_text(distinct_cards[1]) if len(distinct_cards) > 1 else "",
                "comparisonStatus": status,
                "supportingSpans": supporting_spans,
                "notes": _clean_text(group.get("conditionText")),
            }
        )

    if not dimensions:
        return None
    return build_compare_packet_contract(
        query=query,
        dimensions=dimensions,
        retrieval_signals=[],
        policy=dict(evidence_policy or {}),
    )


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


__all__ = ["COMPARE_PACKET_SCHEMA", "build_compare_packet_contract", "build_compare_packet_from_runtime"]
