from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge_hub.application.research_review_loop_types import ClaimCandidate, JsonValue, Row, WeakConceptBasis


def _claim_payload(claim: "ClaimCandidate") -> dict[str, "JsonValue"]:
    payload: dict[str, JsonValue] = {
        "claimId": claim.claim_id,
        "claimCardId": claim.claim_card_id,
        "claimText": claim.claim_text,
        "claimTextHash": claim.claim_text_hash,
        "state": claim.state,
        "sourceIds": claim.source_ids,
        "evidenceSpanIds": claim.evidence_span_ids,
        "evidenceSnippetHashes": claim.evidence_snippet_hashes,
        "canonicalEligible": claim.canonical_eligible,
        "reviewDecisionId": claim.review_decision_id,
        "warnings": claim.warnings,
        "claimOrigin": claim.claim_origin,
        "claimTrustLevel": claim.claim_trust_level,
        "claimQualityFlag": claim.claim_quality_flag,
        "reviewInputEligible": claim.review_input_eligible,
        "authorityLevel": claim.authority_level,
        "authorityStatus": claim.authority_status,
        "canonicalBlockers": claim.canonical_blockers,
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

    return _evidence_payloads_for_anchors(
        row=row,
        claim_id=claim_id,
        anchors=_anchors_for_claim(store, row=row, claim_id=claim_id),
    )


def _evidence_payloads_for_anchors(
    *,
    row: "Row",
    claim_id: str,
    anchors: list["Row"],
) -> list[dict[str, "JsonValue"]]:
    payloads: list[dict[str, JsonValue]] = []
    seen_hashes: set[str] = set()
    for index, anchor in enumerate(anchors, start=1):
        raw_locator = _first_text(anchor, ("locator", "stable_span_locator", "source_locator"))
        resolved_locator = _first_text(anchor, ("resolvedLocator", "resolved_locator"))
        locator = _effective_locator(raw_locator=raw_locator, resolved_locator=resolved_locator)
        source_content_hash = _first_text(anchor, ("sourceContentHash", "source_content_hash"))
        source_id = _first_text(anchor, ("source_id",)) or _first_text(row, ("paper_id", "source_id"))
        expected_source_id = _first_text(row, ("paper_id", "source_id"))
        text_preview = _first_text(anchor, ("quote", "excerpt", "text"))
        snippet_hash = _snippet_hash(anchor, text_preview)
        if snippet_hash in seen_hashes:
            continue
        seen_hashes.add(snippet_hash)
        authority = _evidence_authority(
            locator=locator,
            source_id=source_id,
            expected_source_id=expected_source_id,
            source_content_hash=source_content_hash,
            snippet_hash=snippet_hash,
            text_preview=text_preview,
        )
        payloads.append(
            {
                "evidenceSpanId": _evidence_span_id(anchor, claim_id=claim_id, index=index),
                "sourceId": source_id,
                "locator": locator,
                "rawLocator": raw_locator if raw_locator != locator else None,
                "sourceContentHash": source_content_hash or None,
                "snippetHash": snippet_hash,
                "textPreview": text_preview[:240],
                "state": authority["state"],
                "warnings": authority["warnings"],
                "locatorKind": authority["locatorKind"],
                "locatorAuthority": authority["locatorAuthority"],
                "provenanceStatus": authority["provenanceStatus"],
                "canonicalBlockers": authority["canonicalBlockers"],
                "sourceContentHashAvailable": authority["sourceContentHashAvailable"],
                "canonicalEligible": authority["canonicalEligible"],
            }
        )
    return payloads


def _effective_locator(*, raw_locator: str, resolved_locator: str) -> str:
    resolved = resolved_locator.strip()
    if raw_locator.strip().startswith("memory-unit:") and _valid_chars_locator(resolved):
        return resolved
    return raw_locator.strip()


def _evidence_authority(
    *,
    locator: str,
    source_id: str,
    expected_source_id: str,
    source_content_hash: str,
    snippet_hash: str,
    text_preview: str,
) -> dict[str, "JsonValue"]:
    locator_kind = _locator_kind(locator)
    source_hash_available = bool(source_content_hash.strip())
    snippet_hash_available = bool(snippet_hash.strip())
    snippet_text_available = bool(text_preview.strip())
    canonical_blockers: list[str] = []
    if source_id.strip() != expected_source_id.strip():
        canonical_blockers.append("source_outside_explicit_scope")
    match locator_kind:
        case "chars_offset":
            if not source_hash_available:
                canonical_blockers.append("missing_source_content_hash")
            if not snippet_hash_available:
                canonical_blockers.append("missing_snippet_hash")
            if snippet_hash_available and not snippet_text_available:
                canonical_blockers.append("missing_snippet_text")
        case "missing":
            canonical_blockers.append("missing_locator")
        case "memory_unit":
            canonical_blockers.append("unresolved_memory_unit_locator")
        case "non_offset":
            canonical_blockers.append("non_offset_locator")
        case unreachable:
            raise AssertionError(f"unexpected locator kind: {unreachable}")
    canonical_eligible = not canonical_blockers
    provenance_status = "source_resolved" if canonical_eligible else _blocked_status(canonical_blockers)
    return {
        "state": "proposed" if canonical_eligible else provenance_status,
        "locatorKind": locator_kind,
        "locatorAuthority": "source_text" if canonical_eligible else "blocked",
        "provenanceStatus": provenance_status,
        "canonicalBlockers": canonical_blockers,
        "sourceContentHashAvailable": source_hash_available,
        "canonicalEligible": canonical_eligible,
        "warnings": [] if canonical_eligible else canonical_blockers,
    }


def _locator_kind(locator: str) -> str:
    token = locator.strip()
    if not token:
        return "missing"
    if token.startswith("memory-unit:"):
        return "memory_unit"
    if token.startswith("chars:") and _valid_chars_locator(token):
        return "chars_offset"
    return "non_offset"


def _valid_chars_locator(locator: str) -> bool:
    _, _, span = locator.partition(":")
    start_text, separator, end_text = span.partition("-")
    if separator != "-":
        return False
    if not start_text.isdecimal() or not end_text.isdecimal():
        return False
    return int(start_text) < int(end_text)


def _blocked_status(canonical_blockers: list[str]) -> str:
    primary = canonical_blockers[0] if canonical_blockers else "missing_locator"
    match primary:
        case "missing_locator":
            return "blocked_missing_locator"
        case "unresolved_memory_unit_locator":
            return "blocked_unresolved_memory_unit_locator"
        case "non_offset_locator":
            return "blocked_non_offset_locator"
        case "missing_source_content_hash":
            return "blocked_missing_source_content_hash"
        case "missing_snippet_hash":
            return "blocked_missing_hash"
        case "missing_snippet_text":
            return "blocked_missing_snippet_text"
        case "source_outside_explicit_scope":
            return "blocked_source_scope_mismatch"
        case unreachable:
            raise AssertionError(f"unexpected canonical blocker: {unreachable}")


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
    if not text_preview.strip():
        return ""
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
