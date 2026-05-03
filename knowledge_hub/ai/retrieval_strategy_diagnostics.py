from __future__ import annotations

from typing import Any

from knowledge_hub.ai.rag_support import clean_text, jaccard, result_id, safe_float, source_label_for_result, tokenize
from knowledge_hub.core.models import SearchResult


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, round(float(value), 6)))


def _score_label(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def build_strategy_plan_diagnostics(
    *,
    query: str,
    query_intent: str,
    paper_family: str,
    source_scope: str,
    candidate_budgets: dict[str, int],
    token_budget: int,
    fallback_window: int,
    temporal_route_applied: bool,
    enrichment_route: str,
    resolved_source_ids: list[str],
    top_k: int,
) -> dict[str, Any]:
    intent = str(query_intent or "general").strip().lower() or "general"
    family = str(paper_family or "").strip().lower()
    source = str(source_scope or "all").strip().lower() or "all"
    resolved_count = len([item for item in resolved_source_ids if str(item or "").strip()])
    query_text = clean_text(query)
    lowered = query_text.casefold()

    if family == "paper_compare" or intent == "comparison":
        complexity_class = "multi_source_compare"
        budget_reason = "compare_query_requires_source_coverage"
    elif str(enrichment_route or "") == "cluster_assist" or any(
        token in lowered for token in ("global", "across", "overall", "전체", "전반", "흐름", "그룹", "묶어")
    ):
        complexity_class = "global_sensemaking"
        budget_reason = "cross_document_synthesis_query"
    elif temporal_route_applied:
        complexity_class = "update_sensitive"
        budget_reason = "temporal_or_latest_query"
    elif family == "paper_lookup" or intent in {"paper_lookup", "topic_lookup"} or resolved_count == 1:
        complexity_class = "exact_lookup"
        budget_reason = "single_source_or_lookup_query"
    elif family == "paper_discover" or intent == "paper_topic":
        complexity_class = "discovery"
        budget_reason = "shortlist_or_topic_discovery_query"
    elif family == "concept_explainer" or intent == "definition":
        complexity_class = "local_explainer"
        budget_reason = "concept_explainer_query"
    elif intent in {"implementation", "howto"}:
        complexity_class = "procedural_lookup"
        budget_reason = "implementation_or_howto_query"
    else:
        complexity_class = "local_lookup"
        budget_reason = "default_core_retrieval"

    allowed_actions = ["broaden_query_terms", "read_parent_section", "abstain"]
    if source == "all":
        allowed_actions.insert(0, "broaden_source_mix")
    if complexity_class in {"multi_source_compare", "global_sensemaking"}:
        allowed_actions.append("increase_source_diversity")
    if complexity_class in {"global_sensemaking", "local_explainer"}:
        allowed_actions.append("graph_or_hierarchy_probe")

    budget = {
        "topK": int(top_k),
        "candidateBudgets": {str(key): int(value) for key, value in dict(candidate_budgets or {}).items()},
        "candidateBudgetTotal": sum(int(value) for value in dict(candidate_budgets or {}).values()),
        "tokenBudget": int(token_budget),
        "fallbackWindow": int(fallback_window),
    }
    return {
        "complexityClass": complexity_class,
        "budgetReason": budget_reason,
        "retrievalBudget": budget,
        "retryPolicy": {
            "mode": "diagnostics_only",
            "maxRetries": 0,
            "allowedActions": allowed_actions,
            "defaultAction": "none",
        },
    }


def build_retrieval_strategy_diagnostics(plan: Any) -> dict[str, Any]:
    retry_policy = dict(getattr(plan, "retry_policy", {}) or {})
    retrieval_budget = dict(getattr(plan, "retrieval_budget", {}) or {})
    return {
        "complexityClass": str(getattr(plan, "complexity_class", "") or "local_lookup"),
        "budgetReason": str(getattr(plan, "budget_reason", "") or "default_core_retrieval"),
        "retrievalBudget": retrieval_budget,
        "retryPolicy": retry_policy,
        "sourceScope": str(getattr(plan, "source_scope", "") or "all"),
        "queryIntent": str(getattr(plan, "query_intent", "") or "general"),
        "paperFamily": str(getattr(plan, "paper_family", "") or "general"),
        "retrievalMode": str(getattr(plan, "retrieval_mode", "") or "hybrid"),
        "memoryMode": str(getattr(plan, "memory_mode", "") or "off"),
        "phase": "phase0_diagnostics",
    }


def _source_ids_for_result(item: SearchResult) -> set[str]:
    metadata = dict(getattr(item, "metadata", {}) or {})
    values = {
        str(getattr(item, "document_id", "") or "").strip(),
        str(metadata.get("arxiv_id") or "").strip(),
        str(metadata.get("paper_id") or "").strip(),
        str(metadata.get("source_id") or "").strip(),
        str(metadata.get("record_id") or "").strip(),
        str(metadata.get("file_path") or "").strip(),
        str(metadata.get("url") or metadata.get("canonical_url") or "").strip(),
    }
    return {value for value in values if value}


def _parent_key(item: SearchResult) -> str:
    metadata = dict(getattr(item, "metadata", {}) or {})
    for key in ("parent_id", "file_path", "arxiv_id", "url", "canonical_url", "title"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return result_id(item)


def build_retrieval_quality_diagnostics(
    *,
    plan: Any,
    results: list[SearchResult],
    candidate_sources: list[dict[str, Any]],
    rerank_signals: dict[str, Any],
    memory_prefilter: dict[str, Any],
) -> dict[str, Any]:
    result_count = len(results)
    top_score = max((safe_float(getattr(item, "score", 0.0), 0.0) for item in results), default=0.0)
    top_semantic = max((safe_float(getattr(item, "semantic_score", 0.0), 0.0) for item in results), default=0.0)
    top_lexical = max((safe_float(getattr(item, "lexical_score", 0.0), 0.0) for item in results), default=0.0)
    budget_total = sum(int(item.get("budget") or 0) for item in candidate_sources)
    hit_total = sum(int(item.get("semanticHits") or 0) + int(item.get("lexicalHits") or 0) for item in candidate_sources)
    if budget_total <= 0:
        budget_total = int((getattr(plan, "retrieval_budget", {}) or {}).get("candidateBudgetTotal") or 0)
    hit_ratio = min(1.0, hit_total / max(1, budget_total))
    top_k = int((getattr(plan, "retrieval_budget", {}) or {}).get("topK") or max(1, result_count))
    evidence_ratio = min(1.0, result_count / max(1, top_k))
    source_types = sorted({source_label_for_result(item) or "unknown" for item in results})
    source_diversity = min(1.0, len(source_types) / 3.0)
    memory_bonus = 0.05 if bool((memory_prefilter or {}).get("memoryInfluenceApplied")) else 0.0

    score = (0.45 * top_score) + (0.22 * hit_ratio) + (0.18 * evidence_ratio) + (0.10 * source_diversity) + memory_bonus
    weak_signals: list[str] = []
    if result_count == 0:
        weak_signals.append("no_results")
    if top_score < 0.35:
        weak_signals.append("low_top_score")
    if hit_total == 0:
        weak_signals.append("no_candidate_hits_recorded")
    if bool((rerank_signals or {}).get("graphHeavy")) and len({_parent_key(item) for item in results}) < 2:
        weak_signals.append("graph_heavy_low_parent_diversity")

    resolved_ids = [str(item).strip() for item in list((getattr(plan, "query_frame", {}) or {}).get("resolved_source_ids") or []) if str(item).strip()]
    resolved_present = 0
    if resolved_ids:
        result_ids = set().union(*(_source_ids_for_result(item) for item in results)) if results else set()
        resolved_present = sum(1 for item in resolved_ids if item in result_ids)
    if str(getattr(plan, "paper_family", "") or "") == "paper_compare" and len(resolved_ids) >= 2 and resolved_present < 2:
        weak_signals.append("compare_source_coverage_incomplete")
        score -= 0.12

    score = _clamp_score(score)
    corrective_action = "none"
    if "no_results" in weak_signals:
        corrective_action = "broaden_search"
    elif "compare_source_coverage_incomplete" in weak_signals:
        corrective_action = "source_scope_rescue"
    elif "graph_heavy_low_parent_diversity" in weak_signals:
        corrective_action = "graph_or_hierarchy_probe"
    elif "low_top_score" in weak_signals:
        corrective_action = "broaden_query_terms"

    return {
        "score": score,
        "label": _score_label(score),
        "evidenceCount": result_count,
        "topScore": round(top_score, 6),
        "topSemanticScore": round(top_semantic, 6),
        "topLexicalScore": round(top_lexical, 6),
        "candidateBudgetTotal": int(budget_total),
        "candidateHitTotal": int(hit_total),
        "sourceTypes": source_types,
        "resolvedSourceCoverage": {
            "expected": len(resolved_ids),
            "present": int(resolved_present),
        },
        "weakSignals": weak_signals,
        "correctiveActionCandidate": corrective_action,
        "phase": "phase0_diagnostics",
    }


def build_answerability_rerank_diagnostics(
    *,
    plan: Any,
    results: list[SearchResult],
) -> dict[str, Any]:
    query_terms = tokenize(str(getattr(plan, "query", "") or ""))
    top_docs = " ".join(clean_text(getattr(item, "document", "") or "") for item in results[:3])
    doc_terms = tokenize(top_docs)
    term_coverage = jaccard(query_terms, doc_terms)
    support_count = sum(1 for item in results if safe_float(getattr(item, "score", 0.0), 0.0) >= 0.5)
    top_score = max((safe_float(getattr(item, "score", 0.0), 0.0) for item in results), default=0.0)
    unique_parents = len({_parent_key(item) for item in results})
    top_k = int((getattr(plan, "retrieval_budget", {}) or {}).get("topK") or max(1, len(results)))
    support_ratio = min(1.0, support_count / max(1, min(top_k, 3)))
    parent_ratio = min(1.0, unique_parents / max(1, min(top_k, 3)))

    resolved_ids = [str(item).strip() for item in list((getattr(plan, "query_frame", {}) or {}).get("resolved_source_ids") or []) if str(item).strip()]
    if resolved_ids:
        result_ids = set().union(*(_source_ids_for_result(item) for item in results)) if results else set()
        resolved_ratio = sum(1 for item in resolved_ids if item in result_ids) / max(1, len(resolved_ids))
    else:
        resolved_ratio = 1.0 if results else 0.0

    score = _clamp_score((0.35 * top_score) + (0.25 * term_coverage) + (0.20 * support_ratio) + (0.10 * parent_ratio) + (0.10 * resolved_ratio))
    weak_signals: list[str] = []
    if not results:
        weak_signals.append("no_evidence")
    if term_coverage < 0.12:
        weak_signals.append("low_query_term_coverage")
    if support_count == 0:
        weak_signals.append("no_high_confidence_support")
    if resolved_ids and resolved_ratio < 1.0:
        weak_signals.append("resolved_source_coverage_incomplete")

    return {
        "applied": False,
        "method": "deterministic_phase0_observation",
        "score": score,
        "label": _score_label(score),
        "termCoverage": round(term_coverage, 6),
        "supportCount": int(support_count),
        "uniqueParentCount": int(unique_parents),
        "resolvedSourceCoverage": round(resolved_ratio, 6),
        "weakSignals": weak_signals,
        "reason": "diagnostics_only_no_result_reordering",
    }


def build_artifact_health_diagnostics(*, plan: Any, results: list[SearchResult]) -> dict[str, Any]:
    source_scope = str(getattr(plan, "source_scope", "") or "all")
    missing: dict[str, int] = {}
    stale_count = 0
    derivative_count = 0
    required = ["source_type", "title"]
    paper_identifier_required = source_scope == "paper"
    if source_scope == "paper":
        required.append("paper_id_or_arxiv_id")

    for item in results:
        metadata = dict(getattr(item, "metadata", {}) or {})
        extras = dict(getattr(item, "lexical_extras", {}) or {})
        for field_name in [field for field in required if field != "paper_id_or_arxiv_id"]:
            if not str(metadata.get(field_name) or "").strip():
                missing[field_name] = missing.get(field_name, 0) + 1
        if paper_identifier_required and not any(
            str(value or "").strip()
            for value in (
                metadata.get("arxiv_id"),
                metadata.get("paper_id"),
                metadata.get("source_id"),
                getattr(item, "document_id", ""),
            )
        ):
            missing["paper_id_or_arxiv_id"] = missing.get("paper_id_or_arxiv_id", 0) + 1
        if str(metadata.get("stale") or extras.get("stale") or "").strip().lower() in {"1", "true", "yes"}:
            stale_count += 1
        if any(str(metadata.get(key) or extras.get(key) or "").strip() for key in ("card_id", "card_type", "derivative_source", "source_content_hash")):
            derivative_count += 1

    denominator = max(1, len(results) * max(1, len(required)))
    missing_ratio = sum(missing.values()) / denominator
    stale_ratio = stale_count / max(1, len(results))
    if not results:
        score = 0.0
    else:
        score = _clamp_score(1.0 - min(0.7, missing_ratio) - min(0.5, stale_ratio * 0.5))
    reasons: list[str] = []
    if missing:
        reasons.append("missing_required_metadata")
    if stale_count:
        reasons.append("stale_candidate_metadata")
    if not results:
        reasons.append("no_candidates_to_inspect")
    return {
        "score": score,
        "label": _score_label(score),
        "requiredFields": required,
        "missingMetadataFields": missing,
        "staleCandidateCount": int(stale_count),
        "derivativeCandidateCount": int(derivative_count),
        "reason": "ok" if not reasons else ",".join(reasons),
        "phase": "phase0_diagnostics",
    }


def build_corrective_retrieval_diagnostics(
    *,
    retrieval_quality: dict[str, Any],
    answerability_rerank: dict[str, Any],
    artifact_health: dict[str, Any],
) -> dict[str, Any]:
    triggers = list(dict.fromkeys(list(retrieval_quality.get("weakSignals") or []) + list(answerability_rerank.get("weakSignals") or [])))
    candidate_action = str(retrieval_quality.get("correctiveActionCandidate") or "none")
    if candidate_action == "none" and str(artifact_health.get("label") or "") == "low":
        candidate_action = "artifact_quality_review"
        triggers.append("artifact_health_low")
    return {
        "applied": False,
        "policy": "diagnostics_only",
        "maxRetries": 0,
        "retryCandidate": candidate_action != "none",
        "candidateAction": candidate_action,
        "triggers": triggers,
        "reason": "phase0_observation_only",
    }
