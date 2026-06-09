from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final, Literal, Protocol, TypeAlias, TypedDict

from knowledge_hub.application.research_review_loop_payloads import (
    _claim_payload,
    _claim_state,
    _evidence_payloads,
    _evidence_span_id,
    _open_questions_for,
    _pack_id,
    _weak_concepts_for,
)


RESEARCH_REVIEW_LOOP_SCHEMA: Final = "knowledge-hub.research-review-loop.result.v1"
RESEARCH_REVIEW_LOOP_VERSION: Final = "2026-06-10"

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
ReviewDecisionValue: TypeAlias = Literal["accept", "reject", "unsure", "needs_more_evidence", "archive"]
ClaimState: TypeAlias = Literal["proposed", "accepted", "rejected", "unsure", "archived"]
EvidenceState: TypeAlias = Literal["proposed", "reviewed", "blocked_missing_locator", "blocked_missing_hash"]
WeakConceptBasis: TypeAlias = Literal["rejected_claim", "unsure_claim", "blocked_evidence", "missing_evidence"]


class Row(TypedDict, total=False):
    claim_card_id: str
    claim_id: str
    claim_text: str
    paper_id: str
    source_id: str
    task_canonical: str
    task: str
    dataset_canonical: str
    dataset: str
    metric_canonical: str
    metric: str
    source_card_id: str
    anchor_id: str
    anchorId: str
    evidenceSpanId: str
    locator: str
    stable_span_locator: str
    source_locator: str
    snippet_hash: str
    snippetHash: str
    sourceContentHash: str
    source_content_hash: str
    quote: str
    excerpt: str
    text: str


class ReviewDecision(TypedDict, total=False):
    decisionId: str
    targetType: str
    targetId: str
    decision: ReviewDecisionValue
    confidence: str
    reviewer: str
    reviewedAt: str
    reason: str


class ReviewLoopStore(Protocol):
    def list_claim_cards(self, *, source_kind: str, limit: int) -> list[Row]: ...

    def list_claim_card_source_refs(self, *, claim_card_id: str = "") -> list[Row]: ...

    def list_evidence_anchors_v2(self, *, card_id: str, claim_ids: list[str]) -> list[Row]: ...


@dataclass(frozen=True, slots=True)
class ClaimCandidate:
    claim_id: str
    claim_card_id: str
    claim_text: str
    paper_id: str
    source_ids: list[str]
    evidence_span_ids: list[str]
    state: ClaimState
    canonical_eligible: bool
    review_decision_id: str | None
    warnings: list[str]


def load_review_decisions(decision_file: str | None) -> list[ReviewDecision]:
    if not decision_file:
        return []
    path = Path(decision_file).expanduser()
    raw_text = path.read_text(encoding="utf-8")
    parsed = json.loads(raw_text)
    if isinstance(parsed, list):
        return [_coerce_decision(item) for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        raw_items = parsed.get("reviewDecisions", [])
        if isinstance(raw_items, list):
            return [_coerce_decision(item) for item in raw_items if isinstance(item, dict)]
    return []


def build_research_review_loop_report(
    store: ReviewLoopStore,
    *,
    paper_ids: list[str],
    decision_file: str | None = None,
) -> dict[str, JsonValue]:
    selected_paper_ids = _unique_non_empty(paper_ids)
    decisions = load_review_decisions(decision_file)
    decisions_by_target = {str(item.get("targetId") or ""): item for item in decisions if str(item.get("targetId") or "")}
    claim_cards = [
        row
        for row in store.list_claim_cards(source_kind="paper", limit=20000)
        if _paper_id(row) in selected_paper_ids
    ]
    evidence_spans: list[dict[str, JsonValue]] = []
    proposed_claims: list[dict[str, JsonValue]] = []
    weak_concepts: list[dict[str, JsonValue]] = []
    open_questions: list[dict[str, JsonValue]] = []

    for row in claim_cards:
        claim = _build_claim_candidate(store, row=row, decisions_by_target=decisions_by_target)
        proposed_claims.append(_claim_payload(claim))
        evidence_spans.extend(_evidence_payloads(store, row=row, claim_id=claim.claim_id))
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
    unsupported_canonical = sum(
        1
        for item in proposed_claims
        if bool(item.get("canonicalEligible")) and not str(item.get("reviewDecisionId") or "")
    )
    status = _status_for(claim_count=len(proposed_claims), decision_count=len(decisions))
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
            "notes": [],
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
            "outputPath": None,
        },
        "counts": {
            "sourceCount": len(selected_paper_ids),
            "proposedClaimRows": len(proposed_claims),
            "evidenceSpanRows": len(evidence_spans),
            "reviewDecisionRows": len(decisions),
            "weakConceptRows": len(weak_concepts),
            "openQuestionRows": len(open_questions),
            "unsupportedCanonicalRows": unsupported_canonical,
        },
        "warnings": warnings,
    }


def _coerce_decision(item: dict[str, JsonValue]) -> ReviewDecision:
    return {
        "decisionId": str(item.get("decisionId") or ""),
        "targetType": str(item.get("targetType") or "claim"),
        "targetId": str(item.get("targetId") or ""),
        "decision": _decision_value(str(item.get("decision") or "unsure")),
        "confidence": str(item.get("confidence") or "medium"),
        "reviewer": str(item.get("reviewer") or "human"),
        "reviewedAt": str(item.get("reviewedAt") or ""),
        "reason": str(item.get("reason") or ""),
    }


def _decision_value(value: str) -> ReviewDecisionValue:
    match value:
        case "accept":
            return "accept"
        case "reject":
            return "reject"
        case "unsure":
            return "unsure"
        case "needs_more_evidence":
            return "needs_more_evidence"
        case "archive":
            return "archive"
        case _:
            return "unsure"


def _status_for(*, claim_count: int, decision_count: int) -> str:
    if claim_count == 0:
        return "blocked"
    if decision_count > 0:
        return "reviewed"
    return "ready_for_human_review"


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


def _build_claim_candidate(
    store: ReviewLoopStore,
    *,
    row: Row,
    decisions_by_target: dict[str, ReviewDecision],
) -> ClaimCandidate:
    claim_id = str(row.get("claim_id") or row.get("claim_card_id") or "").strip()
    claim_card_id = str(row.get("claim_card_id") or claim_id).strip()
    decision = decisions_by_target.get(claim_id) or decisions_by_target.get(claim_card_id)
    state = _claim_state(decision)
    evidence_span_ids = [
        _evidence_span_id(anchor, claim_id=claim_id, index=index)
        for index, anchor in enumerate(_anchors_for_claim(store, row=row, claim_id=claim_id), start=1)
    ]
    warnings = [] if evidence_span_ids else ["missing evidence span"]
    return ClaimCandidate(
        claim_id=claim_id,
        claim_card_id=claim_card_id,
        claim_text=str(row.get("claim_text") or "").strip(),
        paper_id=_paper_id(row),
        source_ids=[_paper_id(row)],
        evidence_span_ids=evidence_span_ids,
        state=state,
        canonical_eligible=state in {"accepted", "unsure"},
        review_decision_id=str(decision.get("decisionId") or "") if decision else None,
        warnings=warnings,
    )


def _anchors_for_claim(store: ReviewLoopStore, *, row: Row, claim_id: str) -> list[Row]:
    claim_card_id = str(row.get("claim_card_id") or "").strip()
    anchors: list[Row] = []
    for source_ref in store.list_claim_card_source_refs(claim_card_id=claim_card_id):
        source_card_id = str(source_ref.get("source_card_id") or "").strip()
        if source_card_id:
            anchors.extend(store.list_evidence_anchors_v2(card_id=source_card_id, claim_ids=[claim_id]))
    return anchors
