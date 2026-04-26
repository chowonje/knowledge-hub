"""Ask-v2 helpers for building retrieval pipeline result diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from knowledge_hub.ai.ask_v2_support import clean_text, stable_score
from knowledge_hub.ai.retrieval_pipeline import RetrievalPlan, RetrievalPipelineResult
from knowledge_hub.core.models import SearchResult
from knowledge_hub.papers.prefilter import normalize_paper_memory_mode_details


def _card_v2_candidate_sources(
    *,
    selected_cards: Sequence[Mapping[str, Any]],
    source_kind: str,
) -> list[dict[str, Any]]:
    return [
        {
            "id": clean_text(card.get("paper_id") or card.get("document_id") or card.get("note_id") or card.get("relative_path")),
            "cardId": clean_text(card.get("card_id")),
            "title": clean_text(card.get("title")),
            "selectionScore": stable_score(card.get("selection_score")),
            "qualityFlag": clean_text(card.get("quality_flag")),
            "sourceKind": source_kind,
        }
        for card in selected_cards
    ]


def _card_v2_context_expansion(
    *,
    route_mode: str,
    route_intent: str,
    context_expansion_mode: str,
) -> dict[str, Any]:
    ontology_applied = context_expansion_mode != "none" and route_mode == "ontology-first"
    return {
        "eligible": ontology_applied,
        "used": ontology_applied,
        "mode": "none" if context_expansion_mode == "none" else ("ontology" if ontology_applied else "card"),
        "reason": route_intent,
        "queryIntent": route_intent,
        "enrichmentRoute": route_mode,
        "ontologyEligible": ontology_applied,
        "clusterEligible": False,
        "ontologyUsed": ontology_applied,
        "clusterUsed": False,
    }


def build_card_v2_pipeline_result(
    *,
    results: list[SearchResult],
    plan: RetrievalPlan,
    source_kind: str,
    route_mode: str,
    route_intent: str,
    selected_cards: Sequence[Mapping[str, Any]] | None = None,
    selected_anchor_count: int = 0,
    memory_reason: str = "",
    context_expansion_mode: str = "card",
    v2_diagnostics: dict[str, Any] | None = None,
    paper_memory_mode: Any = "off",
) -> RetrievalPipelineResult:
    """Build the ask-v2 card retrieval result without duplicating contract fields."""

    normalized_source_kind = clean_text(source_kind) or "unknown"
    normalized_route_mode = clean_text(route_mode)
    normalized_route_intent = clean_text(route_intent)
    reason = clean_text(memory_reason) or normalized_route_mode
    cards = list(selected_cards or [])
    card_v2_mode = f"{normalized_source_kind}-card-v2"
    paper_requested_mode, paper_effective_mode, paper_alias_applied = normalize_paper_memory_mode_details(paper_memory_mode)
    paper_prefilter_enabled = paper_effective_mode in {"compat", "on"}
    paper_prefilter_applied = normalized_source_kind == "paper" and paper_prefilter_enabled
    if paper_prefilter_applied:
        paper_prefilter_reason = reason
    elif paper_prefilter_enabled:
        paper_prefilter_reason = "source_not_paper"
    else:
        paper_prefilter_reason = "disabled"
    matched_paper_ids = [clean_text(card.get("paper_id")) for card in cards if clean_text(card.get("paper_id"))] if paper_prefilter_applied else []
    matched_memory_ids = [clean_text(card.get("source_memory_id")) for card in cards if clean_text(card.get("source_memory_id"))] if paper_prefilter_applied else []

    return RetrievalPipelineResult(
        results=results,
        plan=plan,
        candidate_sources=_card_v2_candidate_sources(
            selected_cards=cards,
            source_kind=normalized_source_kind,
        ),
        memory_route={
            "contractRole": "ask_retrieval_memory_prefilter",
            "mode": card_v2_mode,
            "applied": True,
            "reason": reason,
        },
        memory_prefilter={
            "contractRole": "retrieval_memory_prefilter",
            "mode": card_v2_mode,
            "applied": True,
            "memoryRelationsUsed": [],
            "temporalSignals": dict(plan.temporal_signals),
        },
        paper_memory_prefilter={
            "contractRole": "paper_source_memory_prefilter",
            "requestedMode": paper_requested_mode,
            "effectiveMode": paper_effective_mode,
            "modeAliasApplied": paper_alias_applied,
            "aliasDeprecated": bool(paper_alias_applied and paper_requested_mode == "prefilter"),
            "sourceType": normalized_source_kind,
            "applied": paper_prefilter_applied,
            "fallbackUsed": False,
            "matchedPaperIds": matched_paper_ids,
            "matchedMemoryIds": matched_memory_ids,
            "reason": paper_prefilter_reason,
            "memoryInfluenceApplied": paper_prefilter_applied,
            "verificationCouplingApplied": False,
            "fallbackReason": "",
        },
        rerank_signals={
            "strategy": card_v2_mode,
            "selectedCardCount": len(cards),
            "selectedAnchorCount": int(selected_anchor_count),
        },
        context_expansion=_card_v2_context_expansion(
            route_mode=normalized_route_mode,
            route_intent=normalized_route_intent,
            context_expansion_mode=clean_text(context_expansion_mode) or "card",
        ),
        related_clusters=[],
        active_profile=None,
        source_scope_enforced=True,
        mixed_fallback_used=False,
        v2_diagnostics=dict(v2_diagnostics or {}),
    )


__all__ = ["build_card_v2_pipeline_result"]
