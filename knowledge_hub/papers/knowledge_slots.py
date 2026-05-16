"""Derived Paper Knowledge Slot contract helpers.

Paper knowledge slots are a read model over PaperCardV2, ClaimCard refs, and
evidence anchors. They are not canonical evidence and do not create storage.
"""

from __future__ import annotations

from typing import Any
import re

from knowledge_hub.core.card_v2_common import clean_text as _clean_text

PAPER_KNOWLEDGE_SLOTS_SCHEMA = "knowledge-hub.paper-knowledge-slots.v1"

_CHARS_LOCATOR_RE = re.compile(r"^chars:\d+-\d+$")
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_TEXT_TOKEN_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z_.+-]*|[\uac00-\ud7a3]+")
_TEXT_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "into",
    "that",
    "the",
    "this",
    "with",
}

_SLOT_DEFINITIONS = [
    {
        "slotType": "paper_core",
        "slotKey": "paper_core",
        "field": "paper_core",
        "coverageKey": "paperCore",
        "roles": {"paper_summary", "supporting"},
    },
    {
        "slotType": "problem",
        "slotKey": "problem_core",
        "field": "problem_core",
        "coverageKey": "problemCore",
        "roles": {"problem"},
    },
    {
        "slotType": "method",
        "slotKey": "method_core",
        "field": "method_core",
        "coverageKey": "methodCore",
        "roles": {"method"},
    },
    {
        "slotType": "result",
        "slotKey": "result_core",
        "field": "result_core",
        "coverageKey": "resultCore",
        "roles": {"result"},
    },
    {
        "slotType": "dataset",
        "slotKey": "dataset_core",
        "field": "dataset_core",
        "coverageKey": "datasetCore",
        "roles": {"dataset"},
    },
    {
        "slotType": "metric",
        "slotKey": "metric_core",
        "field": "metric_core",
        "coverageKey": "metricCore",
        "roles": {"metric"},
    },
    {
        "slotType": "limitation",
        "slotKey": "limitations_core",
        "field": "limitations_core",
        "coverageKey": "limitationsCore",
        "roles": {"limitation"},
    },
    {
        "slotType": "when_not_to_use",
        "slotKey": "when_not_to_use",
        "field": "when_not_to_use",
        "coverageKey": "whenNotToUse",
        "roles": {"when_not_to_use", "limitation"},
    },
]


def _get(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload.get(key)
    return ""


def _source_content_hash(card: dict[str, Any], anchor: dict[str, Any]) -> str:
    return _clean_text(
        _get(anchor, "sourceContentHash", "source_content_hash")
        or _get(card, "sourceContentHash", "source_content_hash")
    )


def _contains_hangul(value: Any) -> bool:
    return bool(_HANGUL_RE.search(_clean_text(value)))


def _cross_language_slot_ref(slot_text: str, quote: str) -> bool:
    if not slot_text or not quote:
        return False
    return _contains_hangul(slot_text) != _contains_hangul(quote)


def _meaningful_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for raw in _TEXT_TOKEN_RE.findall(_clean_text(value).casefold()):
        if raw in _TEXT_STOPWORDS:
            continue
        if len(raw) < 3 and not raw.isdigit():
            continue
        tokens.add(raw)
    return tokens


def _slot_text_supported_by_quote(slot_text: str, quote: str) -> bool:
    slot = _clean_text(slot_text)
    source_quote = _clean_text(quote)
    if not slot or not source_quote:
        return False
    slot_folded = slot.casefold()
    quote_folded = source_quote.casefold()
    if slot_folded in quote_folded or quote_folded in slot_folded:
        return True
    if _cross_language_slot_ref(slot, source_quote):
        return False
    slot_tokens = _meaningful_tokens(slot)
    quote_tokens = _meaningful_tokens(source_quote)
    if not slot_tokens or not quote_tokens:
        return False
    overlap = len(slot_tokens & quote_tokens)
    return overlap >= min(3, len(slot_tokens)) and overlap / max(1, min(len(slot_tokens), len(quote_tokens))) >= 0.5


def _slot_coverage(card: dict[str, Any], coverage_key: str, text: str) -> str:
    coverage = _get(card, "slotCoverage", "slot_coverage")
    if isinstance(coverage, dict):
        token = _clean_text(coverage.get(coverage_key))
        if token:
            return token
    return "complete" if text else "missing"


def _claim_ref_payload(ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "claimId": _clean_text(_get(ref, "claimId", "claim_id")),
        "role": _clean_text(ref.get("role")),
        "slotKey": _clean_text(_get(ref, "slotKey", "slot_key")),
        "confidence": ref.get("confidence"),
        "rank": ref.get("rank"),
        "reason": _clean_text(ref.get("reason")),
        "normalization": dict(ref.get("normalization") or {}),
    }


def _claim_refs_for_slot(claim_refs: list[dict[str, Any]], slot: dict[str, Any]) -> list[dict[str, Any]]:
    slot_key = _clean_text(slot["slotKey"])
    roles = {str(role) for role in slot["roles"]}
    matched: list[dict[str, Any]] = []
    for raw_ref in list(claim_refs or []):
        ref = dict(raw_ref or {})
        ref_slot_key = _clean_text(_get(ref, "slotKey", "slot_key"))
        ref_role = _clean_text(ref.get("role"))
        if ref_slot_key == slot_key or ref_role in roles:
            matched.append(_claim_ref_payload(ref))
    return matched


def _anchor_matches_slot(anchor: dict[str, Any], slot: dict[str, Any], claim_ids: set[str]) -> bool:
    role = _clean_text(_get(anchor, "evidenceRole", "evidence_role"))
    if role in slot["roles"]:
        return True
    claim_id = _clean_text(_get(anchor, "claimId", "claim_id"))
    return bool(claim_id and claim_id in claim_ids)


def _evidence_ref_payload(anchor: dict[str, Any], card: dict[str, Any], *, index: int, slot_text: str = "") -> dict[str, Any]:
    span_locator = _clean_text(_get(anchor, "spanLocator", "span_locator"))
    source_content_hash = _source_content_hash(card, anchor)
    snippet_hash = _clean_text(_get(anchor, "snippetHash", "snippet_hash"))
    quote = _clean_text(anchor.get("quote") or anchor.get("excerpt"))
    strict_span_backed = bool(source_content_hash and _CHARS_LOCATOR_RE.match(span_locator))
    strict_warning = ""
    if strict_span_backed and _cross_language_slot_ref(_clean_text(slot_text), quote):
        strict_span_backed = False
        strict_warning = "cross_language_slot_text_not_original_source_evidence"
    elif strict_span_backed and not _slot_text_supported_by_quote(_clean_text(slot_text), quote):
        strict_span_backed = False
        strict_warning = "slot_text_not_supported_by_source_quote"
    return {
        "anchorId": _clean_text(_get(anchor, "anchorId", "anchor_id")) or f"anchor:{index}",
        "claimId": _clean_text(_get(anchor, "claimId", "claim_id")),
        "sourceId": _clean_text(_get(anchor, "sourceId", "paper_id") or _get(card, "paperId", "paper_id")),
        "sourceType": _clean_text(_get(anchor, "sourceType", "source_type")) or "paper",
        "documentId": _clean_text(_get(anchor, "documentId", "document_id")),
        "chunkId": _clean_text(_get(anchor, "chunkId", "chunk_id")),
        "unitId": _clean_text(_get(anchor, "unitId", "unit_id")),
        "spanLocator": span_locator,
        "sourceContentHash": source_content_hash,
        "contentHash": snippet_hash,
        "snippetHash": snippet_hash,
        "evidenceRole": _clean_text(_get(anchor, "evidenceRole", "evidence_role")),
        "sectionPath": _clean_text(_get(anchor, "sectionPath", "section_path")),
        "quote": quote,
        "score": anchor.get("score"),
        "strictSpanBacked": strict_span_backed,
        "locatorOnly": bool(span_locator and not strict_span_backed),
        "fallbackSpan": False,
        "provenanceWarning": strict_warning,
    }


def _dedupe_evidence_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for ref in refs:
        key = (
            _clean_text(ref.get("anchorId")),
            _clean_text(ref.get("spanLocator")),
            _clean_text(ref.get("quote")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def build_paper_knowledge_slots_payload(
    *,
    card: dict[str, Any],
    claim_refs: list[dict[str, Any]] | None = None,
    anchors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an additive Paper Knowledge Slots payload from existing card surfaces."""

    source_card = dict(card or {})
    card_id = _clean_text(_get(source_card, "cardId", "card_id"))
    paper_id = _clean_text(_get(source_card, "paperId", "paper_id"))
    title = _clean_text(source_card.get("title"))
    normalized_claim_refs = [dict(item or {}) for item in list(claim_refs or [])]
    normalized_anchors = [dict(item or {}) for item in list(anchors or [])]

    slots: list[dict[str, Any]] = []
    for slot_def in _SLOT_DEFINITIONS:
        text = _clean_text(source_card.get(slot_def["field"]))
        slot_claim_refs = _claim_refs_for_slot(normalized_claim_refs, slot_def)
        claim_ids = {_clean_text(item.get("claimId")) for item in slot_claim_refs if _clean_text(item.get("claimId"))}
        evidence_refs = _dedupe_evidence_refs(
            [
                _evidence_ref_payload(anchor, source_card, index=index, slot_text=text)
                for index, anchor in enumerate(normalized_anchors, start=1)
                if _anchor_matches_slot(anchor, slot_def, claim_ids)
            ]
        )
        strict_count = sum(1 for ref in evidence_refs if ref.get("strictSpanBacked") is True)
        slots.append(
            {
                "slotId": f"{card_id or paper_id}#slot:{slot_def['slotType']}",
                "slotType": slot_def["slotType"],
                "slotKey": slot_def["slotKey"],
                "text": text,
                "coverage": _slot_coverage(source_card, slot_def["coverageKey"], text),
                "strictEvidence": strict_count > 0,
                "strictEvidenceRefCount": strict_count,
                "evidenceRefCount": len(evidence_refs),
                "claimRefs": slot_claim_refs,
                "evidenceRefs": evidence_refs,
                "derivedFrom": ["paper_cards_v2", "paper_card_claim_refs_v2", "evidence_anchors_v2"],
            }
        )

    complete_slots = sum(1 for item in slots if item.get("coverage") == "complete")
    strict_slots = sum(1 for item in slots if item.get("strictEvidence") is True)
    anchored_slots = sum(1 for item in slots if int(item.get("evidenceRefCount") or 0) > 0)
    return {
        "schema": PAPER_KNOWLEDGE_SLOTS_SCHEMA,
        "authority": "derived_read_model",
        "sourceKind": "paper",
        "paperId": paper_id,
        "cardId": card_id,
        "title": title,
        "sourceContentHash": _clean_text(_get(source_card, "sourceContentHash", "source_content_hash")),
        "slots": slots,
        "coverage": {
            "slotCount": len(slots),
            "completeSlotCount": complete_slots,
            "anchoredSlotCount": anchored_slots,
            "strictSlotCount": strict_slots,
            "strictEvidenceRefCount": sum(int(item.get("strictEvidenceRefCount") or 0) for item in slots),
        },
        "diagnostics": {
            "claimRefCount": len(normalized_claim_refs),
            "evidenceAnchorCount": len(normalized_anchors),
            "summaryOnlySlotCount": sum(
                1 for item in slots if item.get("coverage") == "complete" and not item.get("evidenceRefs")
            ),
        },
    }


def load_paper_knowledge_slots_payload(*, sqlite_db: Any, paper_id: str) -> dict[str, Any] | None:
    """Load the derived slot payload for a stored PaperCardV2 without writing."""

    token = _clean_text(paper_id)
    if not token:
        return None
    card = sqlite_db.get_paper_card_v2(token)
    if not card:
        return None
    card_id = _clean_text(_get(dict(card), "cardId", "card_id"))
    return build_paper_knowledge_slots_payload(
        card=dict(card),
        claim_refs=list(sqlite_db.list_paper_card_claim_refs_v2(card_id=card_id) or []) if card_id else [],
        anchors=list(sqlite_db.list_evidence_anchors_v2(card_id=card_id) or []) if card_id else [],
    )


__all__ = [
    "PAPER_KNOWLEDGE_SLOTS_SCHEMA",
    "build_paper_knowledge_slots_payload",
    "load_paper_knowledge_slots_payload",
]
