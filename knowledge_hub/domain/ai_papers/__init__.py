from __future__ import annotations

from knowledge_hub.domain.ai_papers.answer_scope import paper_family_answer_mode, paper_family_answer_scope, paper_family_query_intent
from knowledge_hub.domain.ai_papers.claim_cards import ClaimCardAlignmentService, ClaimCardBuilder, claim_alignment, rank_claim_cards
from knowledge_hub.domain.ai_papers.evidence_policy import normalize_evidence_policy, policy_for_family, policy_key_for_family, select_evidence_policy
from knowledge_hub.domain.ai_papers.families import (
    PAPER_FAMILY_COMPARE,
    PAPER_FAMILY_CONCEPT_EXPLAINER,
    PAPER_FAMILY_DISCOVER,
    PAPER_FAMILY_LOOKUP,
    PAPER_FAMILY_VALUES,
    classify_paper_family,
    explicit_paper_id,
)
from knowledge_hub.domain.ai_papers.lookup import lookup_match_strength, resolve_lookup
from knowledge_hub.domain.ai_papers.query_plan import (
    PAPER_FAMILY_CONCEPT,
    PaperQueryPlan,
    build_rule_based_query_frame,
    build_rule_based_paper_query_plan,
    build_rule_query_plan,
    maybe_apply_llm_query_planner,
    maybe_apply_planner_fallback,
    merge_query_plans,
    normalize_query_plan_dict,
    planner_fallback_payload,
    query_frame_from_query_plan,
    query_frame_to_query_plan,
    should_attempt_query_planner,
)
from knowledge_hub.domain.ai_papers.representative import expand_concept_terms, representative_hint


def classify_family(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, object] | None = None,
) -> str:
    return classify_paper_family(query, source_type=source_type, metadata_filter=metadata_filter)


def normalize(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, object] | None = None,
    sqlite_db=None,
    query_plan: dict[str, object] | None = None,
):
    if query_plan:
        return query_frame_from_query_plan(
            dict(query_plan),
            query=query,
            source_type=source_type,
            metadata_filter=metadata_filter,
            sqlite_db=sqlite_db,
        )
    return build_rule_based_query_frame(
        query,
        source_type=source_type,
        metadata_filter=metadata_filter,
        sqlite_db=sqlite_db,
    )


def build_query_plan(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, object] | None = None,
    sqlite_db=None,
) -> dict[str, object]:
    return build_rule_query_plan(
        query,
        source_type=source_type,
        metadata_filter=metadata_filter,
        sqlite_db=sqlite_db,
    ).to_dict()


__all__ = [
    "ClaimCardAlignmentService",
    "ClaimCardBuilder",
    "PAPER_FAMILY_COMPARE",
    "PAPER_FAMILY_CONCEPT",
    "PAPER_FAMILY_CONCEPT_EXPLAINER",
    "PAPER_FAMILY_DISCOVER",
    "PAPER_FAMILY_LOOKUP",
    "PAPER_FAMILY_VALUES",
    "PaperQueryPlan",
    "build_rule_based_query_frame",
    "build_rule_based_paper_query_plan",
    "build_rule_query_plan",
    "build_query_plan",
    "claim_alignment",
    "classify_family",
    "classify_paper_family",
    "expand_concept_terms",
    "explicit_paper_id",
    "lookup_match_strength",
    "maybe_apply_llm_query_planner",
    "maybe_apply_planner_fallback",
    "merge_query_plans",
    "normalize",
    "normalize_evidence_policy",
    "normalize_query_plan_dict",
    "paper_family_answer_mode",
    "paper_family_answer_scope",
    "paper_family_query_intent",
    "policy_for_family",
    "policy_key_for_family",
    "planner_fallback_payload",
    "query_frame_from_query_plan",
    "query_frame_to_query_plan",
    "rank_claim_cards",
    "representative_hint",
    "resolve_lookup",
    "select_evidence_policy",
    "should_attempt_query_planner",
]
