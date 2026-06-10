from __future__ import annotations

from typing import Final

from knowledge_hub.application.research_review_loop_claims_file import load_claims_file_inputs
from knowledge_hub.application.research_review_loop_decisions import (
    claim_text_hash,
    decision_applies_to_claim,
    load_review_decisions,
)
from knowledge_hub.application.research_review_loop_payloads import (
    _claim_payload,
    _claim_state,
    _evidence_payloads_for_anchors,
    _open_questions_for,
    _pack_id,
    _weak_concepts_for,
)
from knowledge_hub.application.research_review_loop_types import (
    ClaimCandidate,
    ClaimInput,
    ClaimState,
    JsonValue,
    ReviewDecision,
    ReviewLoopStore,
    Row,
)


RESEARCH_REVIEW_LOOP_SCHEMA: Final = "knowledge-hub.research-review-loop.result.v1"
RESEARCH_REVIEW_LOOP_VERSION: Final = "2026-06-10"


def build_research_review_loop_report(
    store: ReviewLoopStore,
    *,
    paper_ids: list[str],
    decision_file: str | None = None,
    claims_file: str | None = None,
) -> dict[str, JsonValue]:
    selected_paper_ids = _unique_non_empty(paper_ids)
    decisions = load_review_decisions(decision_file)
    decisions_by_target = {str(item.get("targetId") or ""): item for item in decisions if str(item.get("targetId") or "")}
    claim_inputs = _claim_inputs_from_store(store, paper_ids=selected_paper_ids)
    claim_inputs.extend(load_claims_file_inputs(claims_file, paper_ids=selected_paper_ids))
    evidence_spans: list[dict[str, JsonValue]] = []
    proposed_claims: list[dict[str, JsonValue]] = []
    weak_concepts: list[dict[str, JsonValue]] = []
    open_questions: list[dict[str, JsonValue]] = []

    for claim_input in claim_inputs:
        row = claim_input.row
        claim_id = _claim_id(row)
        evidence_payloads = _evidence_payloads_for_anchors(row=row, claim_id=claim_id, anchors=claim_input.anchors)
        claim = _build_claim_candidate(
            row=row,
            decisions_by_target=decisions_by_target,
            evidence_payloads=evidence_payloads,
        )
        proposed_claims.append(_claim_payload(claim))
        evidence_spans.extend(evidence_payloads)
        weak_concepts.extend(_weak_concepts_for(claim))
        open_questions.extend(_open_questions_for(claim))

    reviewed_claim_ids = [
        str(item["claimId"])
        for item in proposed_claims
        if str(item.get("reviewDecisionId") or "")
    ]
    excluded_unreviewed = [
        str(item["claimId"])
        for item in proposed_claims
        if not str(item.get("reviewDecisionId") or "")
    ]
    valid_review_decision_count = len(reviewed_claim_ids)
    unsupported_canonical = sum(
        1
        for item in proposed_claims
        if _is_unsupported_reviewed_canonical_candidate(item)
    )
    authoritative_assert_count = sum(1 for item in proposed_claims if _is_authoritative_assertion(item))
    blocked_evidence_span_rows = sum(1 for item in evidence_spans if not bool(item.get("canonicalEligible")))
    reviewed_blocked_claim_rows = sum(1 for item in proposed_claims if _is_reviewed_blocked_claim(item))
    status = _status_for(
        claim_count=len(proposed_claims),
        decision_count=valid_review_decision_count,
        authoritative_assert_count=authoritative_assert_count,
    )
    warnings = [] if proposed_claims else ["no claim cards found for explicit paper scope"]
    pack_id = _pack_id(selected_paper_ids)
    return {
        "schema": RESEARCH_REVIEW_LOOP_SCHEMA,
        "status": status,
        "version": RESEARCH_REVIEW_LOOP_VERSION,
        "sourceScope": {
            "sourceType": "paper",
            "explicitSourceIds": selected_paper_ids,
            "implicitScopeAllowed": False,
            "notes": _source_scope_notes(claims_file),
        },
        "proposedClaims": proposed_claims,
        "evidenceSpans": evidence_spans,
        "reviewDecisions": [dict(item) for item in decisions],
        "weakConcepts": weak_concepts,
        "openQuestions": open_questions,
        "contextPackPreview": {
            "packId": pack_id,
            "packType": "review",
            "reviewedClaimIds": reviewed_claim_ids,
            "excludedUnreviewedClaimIds": excluded_unreviewed,
            "weakConceptIds": [str(item["conceptId"]) for item in weak_concepts],
            "openQuestionIds": [str(item["questionId"]) for item in open_questions],
            "canonicalWriteAllowed": False,
            "authoritativeAssertCount": authoritative_assert_count,
            "reviewedButNoAuthoritativeAssertions": valid_review_decision_count > 0 and authoritative_assert_count == 0,
            "memoryProjectionPolicy": _memory_projection_policy(),
            "outputPath": None,
        },
        "counts": {
            "sourceCount": len(selected_paper_ids),
            "proposedClaimRows": len(proposed_claims),
            "evidenceSpanRows": len(evidence_spans),
            "reviewDecisionRows": len(decisions),
            "appliedReviewDecisionRows": valid_review_decision_count,
            "weakConceptRows": len(weak_concepts),
            "openQuestionRows": len(open_questions),
            "unsupportedCanonicalRows": unsupported_canonical,
            "authoritativeAssertRows": authoritative_assert_count,
            "authoritativeAssertionRows": authoritative_assert_count,
            "blockedEvidenceSpanRows": blocked_evidence_span_rows,
            "reviewedBlockedClaimRows": reviewed_blocked_claim_rows,
        },
        "warnings": warnings,
    }


def _claim_inputs_from_store(store: ReviewLoopStore, *, paper_ids: list[str]) -> list[ClaimInput]:
    selected = set(paper_ids)
    rows = [
        row
        for row in store.list_claim_cards(source_kind="paper", limit=20000)
        if _paper_id(row) in selected
    ]
    return [
        ClaimInput(row=row, anchors=_anchors_for_claim(store, row=row, claim_id=_claim_id(row)))
        for row in rows
    ]


def _source_scope_notes(claims_file: str | None) -> list[str]:
    if not claims_file:
        return []
    return ["claims_file_input_used"]


def _memory_projection_policy() -> dict[str, JsonValue]:
    return {
        "memoryCards": "excluded_generated_unreviewed",
        "canonicalMemoryCardWriteAllowed": False,
        "projectionAllowedFromReviewedArtifactsOnly": True,
    }


def _status_for(*, claim_count: int, decision_count: int, authoritative_assert_count: int) -> str:
    if claim_count == 0:
        return "blocked"
    if decision_count > 0:
        if authoritative_assert_count == 0:
            return "reviewed_no_authoritative_assertions"
        return "reviewed"
    return "ready_for_human_review"


def _is_unsupported_reviewed_canonical_candidate(item: dict[str, JsonValue]) -> bool:
    state = str(item.get("state") or "")
    if state != "accepted":
        return False
    if not str(item.get("reviewDecisionId") or ""):
        return False
    return not bool(item.get("canonicalEligible"))


def _is_authoritative_assertion(item: dict[str, JsonValue]) -> bool:
    return (
        str(item.get("state") or "") == "accepted"
        and bool(item.get("canonicalEligible"))
        and str(item.get("authorityStatus") or "") == "authoritative"
    )


def _is_reviewed_blocked_claim(item: dict[str, JsonValue]) -> bool:
    return (
        str(item.get("state") or "") == "accepted"
        and str(item.get("reviewDecisionId") or "") != ""
        and not bool(item.get("canonicalEligible"))
    )


def _paper_id(row: Row) -> str:
    return str(row.get("paper_id") or row.get("source_id") or "").strip()


def _unique_non_empty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        token = str(value or "").strip()
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def _claim_id(row: Row) -> str:
    return str(row.get("claim_id") or row.get("claim_card_id") or "").strip()


def _build_claim_candidate(
    *,
    row: Row,
    decisions_by_target: dict[str, ReviewDecision],
    evidence_payloads: list[dict[str, JsonValue]],
) -> ClaimCandidate:
    claim_id = _claim_id(row)
    claim_card_id = str(row.get("claim_card_id") or claim_id).strip()
    claim_text = str(row.get("claim_text") or "").strip()
    evidence_snippet_hashes = [str(item.get("snippetHash") or "") for item in evidence_payloads]
    decision = _applied_decision(
        decisions_by_target.get(claim_id) or decisions_by_target.get(claim_card_id),
        claim_id=claim_id,
        claim_text=claim_text,
        evidence_snippet_hashes=evidence_snippet_hashes,
    )
    state = _claim_state(decision)
    evidence_span_ids = [str(item.get("evidenceSpanId") or "") for item in evidence_payloads]
    all_valid_evidence_span_ids = _valid_evidence_span_ids(evidence_payloads)
    cited_evidence_payloads = _cited_evidence_payloads(evidence_payloads=evidence_payloads, decision=decision)
    cited_valid_evidence_span_ids = _valid_evidence_span_ids(cited_evidence_payloads)
    canonical_blockers = _claim_canonical_blockers(
        state=state,
        valid_evidence_span_ids=cited_valid_evidence_span_ids,
        evidence_payloads=cited_evidence_payloads,
        decision=decision,
        row=row,
    )
    canonical_eligible = _canonical_eligible(
        state=state,
        valid_evidence_span_ids=cited_valid_evidence_span_ids,
        canonical_blockers=canonical_blockers,
    )
    warnings = _claim_warnings(valid_evidence_span_ids=all_valid_evidence_span_ids, decision=decision)
    claim_origin = _claim_origin(row)
    review_input_eligible = claim_origin != "synthetic_fallback"
    return ClaimCandidate(
        claim_id=claim_id,
        claim_card_id=claim_card_id,
        claim_text=claim_text,
        claim_text_hash=claim_text_hash(claim_text),
        paper_id=_paper_id(row),
        source_ids=[_paper_id(row)],
        evidence_span_ids=evidence_span_ids,
        evidence_snippet_hashes=evidence_snippet_hashes,
        state=state,
        canonical_eligible=canonical_eligible,
        review_decision_id=str(decision.get("decisionId") or "") if decision else None,
        warnings=warnings,
        claim_origin=claim_origin,
        claim_trust_level=_claim_trust_level(
            decision=decision,
            canonical_eligible=canonical_eligible,
            review_input_eligible=review_input_eligible,
        ),
        claim_quality_flag=_claim_quality_flag(claim_origin=claim_origin, review_input_eligible=review_input_eligible),
        review_input_eligible=review_input_eligible,
        authority_level="authoritative" if canonical_eligible else "candidate",
        authority_status=_authority_status(
            state=state,
            canonical_eligible=canonical_eligible,
            decision=decision,
            review_input_eligible=review_input_eligible,
        ),
        canonical_blockers=canonical_blockers,
    )


def _applied_decision(
    decision: ReviewDecision | None,
    *,
    claim_id: str,
    claim_text: str,
    evidence_snippet_hashes: list[str],
) -> ReviewDecision | None:
    if decision_applies_to_claim(decision, claim_id=claim_id, claim_text=claim_text):
        cited_hashes = [str(item).strip() for item in list(decision.get("snippetHashes") or []) if str(item).strip()]
        if not cited_hashes:
            return decision
        actual_hashes = {item for item in evidence_snippet_hashes if item}
        if set(cited_hashes).issubset(actual_hashes):
            return decision
    return None


def _canonical_eligible(
    *,
    state: ClaimState,
    valid_evidence_span_ids: list[str],
    canonical_blockers: list[str],
) -> bool:
    return state == "accepted" and bool(valid_evidence_span_ids) and not canonical_blockers


def _claim_warnings(*, valid_evidence_span_ids: list[str], decision: ReviewDecision | None) -> list[str]:
    warnings: list[str] = []
    if not valid_evidence_span_ids:
        warnings.append("missing valid evidence span")
    if decision and not list(decision.get("snippetHashes") or []):
        warnings.append("review decision missing snippet hash")
    return warnings


def _valid_evidence_span_ids(evidence_payloads: list[dict[str, JsonValue]]) -> list[str]:
    return [
        str(item.get("evidenceSpanId") or "")
        for item in evidence_payloads
        if bool(item.get("canonicalEligible"))
    ]


def _cited_evidence_payloads(
    *,
    evidence_payloads: list[dict[str, JsonValue]],
    decision: ReviewDecision | None,
) -> list[dict[str, JsonValue]]:
    if decision is None:
        return []
    cited_hashes = set(_decision_snippet_hashes(decision))
    if not cited_hashes:
        return []
    return [
        item
        for item in evidence_payloads
        if str(item.get("snippetHash") or "") in cited_hashes
    ]


def _decision_snippet_hashes(decision: ReviewDecision) -> list[str]:
    return [str(item).strip() for item in list(decision.get("snippetHashes") or []) if str(item).strip()]


def _claim_canonical_blockers(
    *,
    state: ClaimState,
    valid_evidence_span_ids: list[str],
    evidence_payloads: list[dict[str, JsonValue]],
    decision: ReviewDecision | None,
    row: Row,
) -> list[str]:
    blockers: list[str] = []
    if _claim_origin(row) == "synthetic_fallback":
        blockers.append("synthetic_fallback_claim")
    if state != "accepted":
        return blockers
    if decision is None:
        blockers.append("missing_human_review_decision")
    if decision is not None and not _decision_snippet_hashes(decision):
        blockers.append("review_decision_missing_snippet_hash")
        return _unique_values(blockers)
    if valid_evidence_span_ids:
        return blockers
    evidence_blockers = _evidence_blockers(evidence_payloads)
    if evidence_blockers == ["missing_locator"]:
        blockers.append("missing_locatable_evidence_span")
    elif evidence_blockers:
        blockers.extend(evidence_blockers)
    else:
        blockers.append("missing_valid_evidence_span")
    return _unique_values(blockers)


def _evidence_blockers(evidence_payloads: list[dict[str, JsonValue]]) -> list[str]:
    blockers: list[str] = []
    for item in evidence_payloads:
        raw = item.get("canonicalBlockers")
        if not isinstance(raw, list):
            continue
        blockers.extend(str(value) for value in raw if str(value).strip())
    return _unique_values(blockers)


def _unique_values(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _claim_origin(row: Row) -> str:
    explicit = str(row.get("origin") or "").strip()
    if explicit:
        return explicit
    claim_card_id = str(row.get("claim_card_id") or "").strip()
    if claim_card_id.startswith("claims-file:"):
        return "claims_file"
    return "store_claim_card"


def _claim_trust_level(
    *,
    decision: ReviewDecision | None,
    canonical_eligible: bool,
    review_input_eligible: bool,
) -> str:
    if canonical_eligible:
        return "human_reviewed_authoritative"
    if decision is not None and review_input_eligible:
        return "human_reviewed_non_authoritative"
    if review_input_eligible:
        return "candidate"
    return "blocked_review_input"


def _claim_quality_flag(*, claim_origin: str, review_input_eligible: bool) -> str:
    if not review_input_eligible:
        return f"blocked_{claim_origin}"
    return "reviewable"


def _authority_status(
    *,
    state: ClaimState,
    canonical_eligible: bool,
    decision: ReviewDecision | None,
    review_input_eligible: bool,
) -> str:
    if canonical_eligible:
        return "authoritative"
    if not review_input_eligible:
        return "blocked"
    if state == "accepted" and decision is not None:
        return "blocked"
    return "candidate"


def _anchors_for_claim(store: ReviewLoopStore, *, row: Row, claim_id: str) -> list[Row]:
    claim_card_id = str(row.get("claim_card_id") or "").strip()
    anchors: list[Row] = []
    for source_ref in store.list_claim_card_source_refs(claim_card_id=claim_card_id):
        source_card_id = str(source_ref.get("source_card_id") or "").strip()
        if source_card_id:
            anchors.extend(store.list_evidence_anchors_v2(card_id=source_card_id, claim_ids=[claim_id]))
    return anchors
