from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, TypedDict


JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
ReviewDecisionValue: TypeAlias = Literal["accept", "reject", "unsure", "needs_more_evidence", "archive"]
ClaimState: TypeAlias = Literal["proposed", "accepted", "rejected", "unsure", "archived"]
EvidenceState: TypeAlias = Literal[
    "proposed",
    "reviewed",
    "blocked_missing_locator",
    "blocked_missing_hash",
    "blocked_unresolved_memory_unit_locator",
    "blocked_non_offset_locator",
    "blocked_missing_source_content_hash",
    "blocked_source_scope_mismatch",
]
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
    origin: str
    source_card_id: str
    anchor_id: str
    anchorId: str
    evidenceSpanId: str
    locator: str
    stable_span_locator: str
    source_locator: str
    resolvedLocator: str
    resolved_locator: str
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
    claimTextHash: str
    snippetHashes: list[str]
    decisionSource: str


class ReviewLoopStore(Protocol):
    def list_claim_cards(self, *, source_kind: str, limit: int) -> list[Row]: ...

    def list_claim_card_source_refs(self, *, claim_card_id: str = "") -> list[Row]: ...

    def list_evidence_anchors_v2(self, *, card_id: str, claim_ids: list[str]) -> list[Row]: ...


@dataclass(frozen=True, slots=True)
class ClaimInput:
    row: Row
    anchors: list[Row]


@dataclass(frozen=True, slots=True)
class ClaimCandidate:
    claim_id: str
    claim_card_id: str
    claim_text: str
    claim_text_hash: str
    paper_id: str
    source_ids: list[str]
    evidence_span_ids: list[str]
    evidence_snippet_hashes: list[str]
    state: ClaimState
    canonical_eligible: bool
    review_decision_id: str | None
    warnings: list[str]
    claim_origin: str
    claim_trust_level: str
    claim_quality_flag: str
    review_input_eligible: bool
    authority_level: str
    authority_status: str
    canonical_blockers: list[str]
