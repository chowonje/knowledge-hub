from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from knowledge_hub.ai.retrieval_strategy_diagnostics import build_strategy_plan_diagnostics


@dataclass(frozen=True)
class RetrievalPlanBuilderDeps:
    normalize_query_frame_dict_fn: Callable[[dict[str, Any] | None], dict[str, Any]]
    normalize_source_type_fn: Callable[[str | None], str | None]
    normalize_query_plan_dict_fn: Callable[[dict[str, Any] | None], dict[str, Any]]
    paper_family_query_intent_fn: Callable[[str, str], str]
    classify_query_intent_fn: Callable[[str], str]
    candidate_budgets_for_intent_fn: Callable[..., dict[str, int]]
    temporal_query_signals_fn: Callable[[str], dict[str, Any]]
    memory_prior_config_fn: Callable[[str, bool, int], tuple[float, int]]
    context_budget_config_fn: Callable[[str, int], tuple[int, int, int]]
    classify_enrichment_route_fn: Callable[[str], dict[str, Any]]
    derive_frame_prefilter_fn: Callable[..., dict[str, Any]]
    normalize_memory_route_mode_fn: Callable[[str], str]
    clean_text_fn: Callable[[str], str]
    retrieval_plan_type: Callable[..., Any]


class RetrievalPlanBuilder:
    def __init__(self, deps: RetrievalPlanBuilderDeps) -> None:
        self._deps = deps

    def build_plan(
        self,
        *,
        query: str,
        top_k: int,
        source_type: str | None,
        retrieval_mode: str,
        memory_route_mode: str,
        use_ontology_expansion: bool,
        metadata_filter: dict[str, Any] | None = None,
        query_plan: dict[str, Any] | None = None,
        query_frame: dict[str, Any] | None = None,
    ) -> Any:
        deps = self._deps
        query_frame_payload = deps.normalize_query_frame_dict_fn(query_frame)
        normalized_source = str(
            deps.normalize_source_type_fn(source_type or query_frame_payload.get("source_type")) or ""
        )
        query_plan_payload = deps.normalize_query_plan_dict_fn(query_plan)
        paper_family = str(query_frame_payload.get("family") or query_plan_payload.get("family") or "").strip().lower()
        intent = str(
            query_frame_payload.get("query_intent")
            or query_plan_payload.get("query_intent")
            or deps.paper_family_query_intent_fn(
                paper_family,
                fallback=deps.classify_query_intent_fn(query),
            )
        )
        if not paper_family and normalized_source == "paper":
            if intent == "definition":
                paper_family = "concept_explainer"
            elif intent == "paper_lookup":
                paper_family = "paper_lookup"
            elif intent == "comparison":
                paper_family = "paper_compare"
            elif intent == "paper_topic":
                paper_family = "paper_discover"
            else:
                paper_family = "general"
        budgets = deps.candidate_budgets_for_intent_fn(intent, top_k=top_k, source_scope=normalized_source or None)
        temporal_signals = deps.temporal_query_signals_fn(query)
        memory_prior_weight, fallback_window = deps.memory_prior_config_fn(
            intent,
            temporal_route=bool(temporal_signals.get("enabled")),
            top_k=top_k,
        )
        token_budget, memory_compression_target, chunk_expansion_threshold = deps.context_budget_config_fn(
            intent,
            top_k=top_k,
        )
        enrichment = deps.classify_enrichment_route_fn(query, query_intent=intent)
        prefilter = deps.derive_frame_prefilter_fn(
            source_scope=normalized_source,
            paper_family=paper_family,
            metadata_filter=metadata_filter,
            query_plan=query_plan_payload,
            query_frame=query_frame_payload,
        )
        strategy = build_strategy_plan_diagnostics(
            query=query,
            query_intent=intent,
            paper_family=paper_family,
            source_scope=normalized_source or "all",
            candidate_budgets=budgets,
            token_budget=token_budget,
            fallback_window=fallback_window,
            temporal_route_applied=bool(temporal_signals.get("enabled")),
            enrichment_route=str(enrichment.get("route") or "core_only"),
            resolved_source_ids=[
                deps.clean_text_fn(item)
                for item in list(query_frame_payload.get("resolved_source_ids") or query_plan_payload.get("resolved_paper_ids") or [])
                if deps.clean_text_fn(item)
            ],
            top_k=top_k,
        )
        return deps.retrieval_plan_type(
            query=deps.clean_text_fn(query),
            source_scope=normalized_source or "all",
            query_intent=intent,
            paper_family=paper_family,
            retrieval_mode=str(retrieval_mode or "hybrid").strip().lower() or "hybrid",
            memory_mode=deps.normalize_memory_route_mode_fn(memory_route_mode),
            candidate_budgets=budgets,
            query_plan=query_plan_payload,
            query_frame=query_frame_payload,
            temporal_signals=temporal_signals,
            temporal_route_applied=bool(temporal_signals.get("enabled")),
            memory_prior_weight=memory_prior_weight,
            fallback_window=fallback_window,
            token_budget=token_budget,
            memory_compression_target=memory_compression_target,
            chunk_expansion_threshold=chunk_expansion_threshold,
            context_expansion_policy="supplemental_opt_in",
            ontology_expansion_enabled=bool(use_ontology_expansion and enrichment.get("ontologyEligible")),
            enrichment_route=str(enrichment.get("route") or "core_only"),
            enrichment_reason=str(enrichment.get("reason") or "default_core_path"),
            ontology_assist_eligible=bool(enrichment.get("ontologyEligible")),
            cluster_assist_eligible=bool(enrichment.get("clusterEligible")),
            resolved_source_scope_applied=bool(prefilter.get("resolved_source_scope_applied")),
            canonical_entities_applied=tuple(prefilter.get("canonical_entities_applied") or ()),
            metadata_filter_applied=dict(prefilter.get("metadata_filter_applied") or {}),
            prefilter_reason=str(prefilter.get("prefilter_reason") or "none"),
            reference_source_applied=bool(prefilter.get("reference_source_applied")),
            watchlist_scope_applied=bool(prefilter.get("watchlist_scope_applied")),
            complexity_class=str(strategy.get("complexityClass") or "local_lookup"),
            budget_reason=str(strategy.get("budgetReason") or "default_core_retrieval"),
            retrieval_budget=dict(strategy.get("retrievalBudget") or {}),
            retry_policy=dict(strategy.get("retryPolicy") or {}),
        )
