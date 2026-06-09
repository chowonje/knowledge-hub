from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

if TYPE_CHECKING:
    from knowledge_hub.application.research_review_loop import ClaimCandidate, Row, WeakConceptBasis


def _claim_payload(claim: "ClaimCandidate") -> dict[str, "JsonValue"]:
    payload: dict[str, JsonValue] = {
        "claimId": claim.claim_id,
        "claimCardId": claim.claim_card_id,
        "claimText": claim.claim_text,
        "state": claim.state,
        "sourceIds": claim.source_ids,
        "evidenceSpanIds": claim.evidence_span_ids,
        "canonicalEligible": claim.canonical_eligible,
        "reviewDecisionId": claim.review_decision_id,
        "warnings": claim.warnings,
    }
    return payload


def _claim_state(decision):
    if decision is None:
        return "proposed"
    match decision.get("decision"):
        case "accept":
            return "accepted"
        case "reject":
            return "rejected"
        case "unsure" | "needs_more_evidence":
            return "unsure"
        case "archive":
            return "archived"
        case _:
            return "unsure"


def _evidence_span_id(anchor: "Row", *, claim_id: str, index: int) -> str:
    explicit = str(anchor.get("evidenceSpanId") or anchor.get("anchor_id") or anchor.get("anchorId") or "").strip()
    if explicit:
        return explicit
    return f"evidence-span:{claim_id}:{index}"


def _evidence_payloads(store, *, row: "Row", claim_id: str) -> list[dict[str, JsonValue]]:
    from knowledge_hub.application.research_review_loop import _anchors_for_claim

    payloads: list[dict[str, JsonValue]] = []
    for index, anchor in enumerate(_anchors_for_claim(store, row=row, claim_id=claim_id), start=1):
        locator = _first_text(anchor, ("locator", "stable_span_locator", "source_locator"))
        text_preview = _first_text(anchor, ("quote", "excerpt", "text"))
        state = "proposed" if locator else "blocked_missing_locator"
        payloads.append(
            {
                "evidenceSpanId": _evidence_span_id(anchor, claim_id=claim_id, index=index),
                "sourceId": _first_text(anchor, ("source_id",)) or _first_text(row, ("paper_id", "source_id")),
                "locator": locator,
                "sourceContentHash": _first_text(anchor, ("sourceContentHash", "source_content_hash")) or None,
                "snippetHash": _snippet_hash(anchor, text_preview),
                "textPreview": text_preview[:240],
                "state": state,
                "warnings": [] if locator else ["missing locator"],
            }
        )
    return payloads


def _first_text(row: "Row", keys: tuple[str, ...]) -> str:
    for key in keys:
        token = str(row.get(key) or "").strip()
        if token:
            return token
    return ""


def _snippet_hash(anchor: "Row", text_preview: str) -> str:
    explicit = str(anchor.get("snippet_hash") or anchor.get("snippetHash") or "").strip()
    if explicit:
        return explicit
    return hashlib.sha1(text_preview.encode("utf-8")).hexdigest()[:16]


def _weak_basis(state: str) -> "WeakConceptBasis | None":
    match state:
        case "rejected":
            return "rejected_claim"
        case "unsure":
            return "unsure_claim"
        case "proposed" | "accepted" | "archived":
            return None
        case _:
            return None


def _weak_concepts_for(claim: "ClaimCandidate") -> list[dict[str, "JsonValue"]]:
    basis = _weak_basis(claim.state)
    if basis is None:
        if claim.evidence_span_ids:
            return []
        basis = "missing_evidence"
    label = claim.claim_text[:120] or claim.claim_id
    return [
        {
            "conceptId": f"weak:{claim.claim_id}",
            "label": label,
            "basis": basis,
            "supportingDecisionIds": [claim.review_decision_id] if claim.review_decision_id else [],
        }
    ]


def _open_questions_for(claim: "ClaimCandidate") -> list[dict[str, "JsonValue"]]:
    if claim.state not in {"proposed", "unsure"} and claim.evidence_span_ids:
        return []
    return [
        {
            "questionId": f"question:{claim.claim_id}",
            "questionText": f"What evidence resolves this claim: {claim.claim_text}",
            "basis": "missing_or_unreviewed_evidence",
            "sourceIds": claim.source_ids,
            "supportingDecisionIds": [claim.review_decision_id] if claim.review_decision_id else [],
        }
    ]


def _pack_id(paper_ids: list[str]) -> str:
    digest = hashlib.sha1(",".join(paper_ids).encode("utf-8")).hexdigest()[:12]
    return f"research-review-loop:paper:{digest}"
