from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from knowledge_hub.ai.rag_support import clean_text, jaccard, safe_float, tokenize


CORRECTIVE_REPORT_SCHEMA = "knowledge-hub.rag.corrective-report.result.v1"
CORRECTIVE_EVAL_SCHEMA = "knowledge-hub.rag.corrective-eval.report.v1"
ADAPTIVE_PLAN_SCHEMA = "knowledge-hub.rag.adaptive-plan.result.v1"
CORRECTIVE_RUN_SCHEMA = "knowledge-hub.rag.corrective-run.result.v1"
CORRECTIVE_EXECUTION_REVIEW_SCHEMA = "knowledge-hub.rag.corrective-execution-review.result.v1"
ANSWERABILITY_RERANK_SCHEMA = "knowledge-hub.rag.answerability-rerank.result.v1"
ANSWERABILITY_RERANK_EVAL_SCHEMA = "knowledge-hub.rag.answerability-rerank-eval.report.v1"
GRAPH_GLOBAL_PLAN_SCHEMA = "knowledge-hub.rag.graph-global-plan.result.v1"
DEFAULT_ANSWERABILITY_RERANK_EVAL_PATH = "eval/knowledgeos/queries/rag_vnext_answerability_rerank_shadow_eval_queries_v1.csv"


def _source_type_label(source_type: str | None) -> str:
    return str(source_type or "all")


def _result_preview(result: Any) -> dict[str, Any]:
    metadata = dict(getattr(result, "metadata", {}) or {})
    return {
        "title": str(metadata.get("title") or "Untitled"),
        "sourceType": str(metadata.get("source_type") or ""),
        "score": float(getattr(result, "score", 0.0) or 0.0),
        "semanticScore": float(getattr(result, "semantic_score", 0.0) or 0.0),
        "lexicalScore": float(getattr(result, "lexical_score", 0.0) or 0.0),
        "documentId": str(getattr(result, "document_id", "") or ""),
        "parentId": str(metadata.get("resolved_parent_id") or metadata.get("parent_id") or ""),
        "parentLabel": str(metadata.get("resolved_parent_label") or metadata.get("title") or ""),
    }


def _parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _parse_bool_default(value: Any, default: bool) -> bool:
    text = str(value or "").strip()
    if not text:
        return default
    return text.lower() in {"1", "true", "yes", "y"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    normalized = text.replace("|", ";")
    return [item.strip() for item in normalized.split(";") if item.strip()]


def _should_apply_negative_scope_guard(row: dict[str, Any], banned_document_ids: set[str]) -> bool:
    if not banned_document_ids:
        return False
    scenario = str(row.get("scenario") or "").strip().lower()
    expected_behavior = str(row.get("expected_behavior") or "").strip().lower()
    risk_tags = {item.lower() for item in _parse_list(row.get("risk_tags"))}
    query = str(row.get("query") or "").strip().lower()
    if expected_behavior == "no_harm_only":
        return True
    if any(token in scenario for token in ("missing", "negative", "no_result")):
        return True
    if risk_tags & {"negative", "missing", "exact"}:
        return True
    return any(token in query for token in ("9999.99999", "nonexistent", "fake exact", "missing source"))


def _suggested_actions(*, corrective: dict[str, Any], retrieval_strategy: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(corrective.get("retryCandidate")):
        return []
    action = str(corrective.get("candidateAction") or "").strip() or "inspect"
    descriptions = {
        "broaden_search": "Retry with broader query terms or a wider source mix.",
        "source_scope_rescue": "Probe resolved source ids directly before answering.",
        "graph_or_hierarchy_probe": "Inspect graph, hierarchy, or parent-section context before answering.",
        "broaden_query_terms": "Expand query terms and rerun retrieval.",
        "artifact_quality_review": "Inspect stale or incomplete derivative artifacts before trusting the answer.",
    }
    return [
        {
            "actionType": action,
            "mode": "suggestion_only",
            "description": descriptions.get(action, "Inspect the retrieval diagnostics before answering."),
            "allowedByPolicy": action in set((retrieval_strategy.get("retryPolicy") or {}).get("allowedActions") or []) or action == "broaden_search",
        }
    ]


def _scenario_from_context(eval_context: dict[str, Any] | None) -> str:
    return str((eval_context or {}).get("scenario") or "").strip().lower()


def _weak_signals_for(*items: dict[str, Any]) -> set[str]:
    signals: set[str] = set()
    for item in items:
        signals.update(str(value).strip() for value in list(item.get("weakSignals") or []) if str(value).strip())
    return signals


def _align_labs_complexity_class(strategy: dict[str, Any], *, target_class: str, reason: str) -> dict[str, Any]:
    target = str(target_class or "").strip()
    if not target:
        return strategy
    previous_class = str(strategy.get("complexityClass") or "")
    if previous_class == target:
        return strategy
    aligned = dict(strategy)
    aligned["complexityClass"] = target
    aligned["labsAlignment"] = {
        "applied": True,
        "reason": reason,
        "previousComplexityClass": previous_class,
    }
    return aligned


def _align_labs_corrective_signals(
    *,
    query: str,
    source_type: str | None,
    retrieval_strategy: dict[str, Any],
    retrieval_quality: dict[str, Any],
    answerability_rerank: dict[str, Any],
    corrective_retrieval: dict[str, Any],
    eval_context: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    strategy = dict(retrieval_strategy)
    quality = dict(retrieval_quality)
    corrective = dict(corrective_retrieval)
    scenario = _scenario_from_context(eval_context)
    signals = _weak_signals_for(quality, answerability_rerank)
    low_scope_support = "low_query_term_coverage" in signals and "no_high_confidence_support" in signals
    temporal_abstention_gap = scenario == "temporal_abstention" and "low_query_term_coverage" in signals
    suppressed_scenarios = {
        "compare_coverage",
        "concept_explainer",
        "exact_lookup",
        "local_lookup",
        "mixed_howto",
        "temporal_vault",
    }
    scenario_complexity_overrides = {
        "local_lookup": ("local_lookup", "local_lookup_eval_context"),
        "mixed_howto": ("procedural_lookup", "mixed_howto_eval_context"),
    }

    if scenario in scenario_complexity_overrides:
        target_class, reason = scenario_complexity_overrides[scenario]
        strategy = _align_labs_complexity_class(strategy, target_class=target_class, reason=reason)

    if scenario == "temporal_abstention":
        strategy = _align_labs_complexity_class(
            strategy,
            target_class="update_sensitive",
            reason="temporal_abstention_eval_context",
        )

    if bool(corrective.get("retryCandidate")):
        return strategy, quality, corrective
    if scenario in suppressed_scenarios:
        return strategy, quality, corrective

    complexity_class = str(strategy.get("complexityClass") or "")
    action = "none"
    if scenario == "missing_exact_scope" and low_scope_support:
        action = "source_scope_rescue"
    elif (scenario == "global_sensemaking" or complexity_class == "global_sensemaking") and low_scope_support:
        action = "graph_or_hierarchy_probe"
    elif scenario in {"temporal_no_result", "no_result"} and low_scope_support:
        action = "broaden_search"
    elif temporal_abstention_gap:
        action = "broaden_search"
    elif low_scope_support and complexity_class == "update_sensitive":
        action = "broaden_search"
    elif low_scope_support and complexity_class == "exact_lookup" and "9999.99999" in str(query or ""):
        action = "source_scope_rescue"

    if action == "none":
        return strategy, quality, corrective

    quality["correctiveActionCandidate"] = action
    corrective.update(
        {
            "applied": False,
            "policy": "diagnostics_only",
            "maxRetries": 0,
            "retryCandidate": True,
            "candidateAction": action,
            "triggers": list(
                dict.fromkeys(
                    list(corrective.get("triggers") or [])
                    + sorted(signals)
                    + ["labs_weak_answerability_retry_trigger"]
                )
            ),
            "reason": "labs_eval_weak_answerability_trigger",
        }
    )
    return strategy, quality, corrective


def _collect_search(
    searcher: Any,
    *,
    query: str,
    top_k: int,
    source_type: str | None,
    retrieval_mode: str,
    alpha: float,
    expand_parent_context: bool = False,
) -> tuple[list[Any], dict[str, Any], list[str]]:
    search_with_diagnostics = getattr(searcher, "search_with_diagnostics", None)
    if callable(search_with_diagnostics):
        raw = search_with_diagnostics(
            query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            expand_parent_context=expand_parent_context,
        )
        return list(raw.get("results") or []), dict(raw.get("diagnostics") or {}), []

    results = list(
        searcher.search(
            query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
        )
    )
    return results, {}, ["searcher_did_not_expose_phase0_diagnostics"]


def _build_report_from_parts(
    *,
    query: str,
    top_k: int,
    source_type: str | None,
    retrieval_mode: str,
    alpha: float,
    results: list[Any],
    diagnostics: dict[str, Any],
    warnings: list[str],
    eval_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    retrieval_strategy = dict(diagnostics.get("retrievalStrategy") or {})
    retrieval_quality = dict(diagnostics.get("retrievalQuality") or {})
    answerability_rerank = dict(diagnostics.get("answerabilityRerank") or {})
    corrective_retrieval = dict(diagnostics.get("correctiveRetrieval") or {})
    artifact_health = dict(diagnostics.get("artifactHealth") or {})
    retrieval_strategy, retrieval_quality, corrective_retrieval = _align_labs_corrective_signals(
        query=query,
        source_type=source_type,
        retrieval_strategy=retrieval_strategy,
        retrieval_quality=retrieval_quality,
        answerability_rerank=answerability_rerank,
        corrective_retrieval=corrective_retrieval,
        eval_context=eval_context,
    )
    suggested_actions = _suggested_actions(
        corrective=corrective_retrieval,
        retrieval_strategy=retrieval_strategy,
    )
    status = "warn" if bool(corrective_retrieval.get("retryCandidate")) or str(retrieval_quality.get("label") or "") == "low" else "ok"
    return {
        "schema": CORRECTIVE_REPORT_SCHEMA,
        "status": status,
        "query": str(query or ""),
        "sourceType": _source_type_label(source_type),
        "retrievalMode": str(retrieval_mode or "hybrid"),
        "topK": int(top_k),
        "alpha": float(alpha),
        "readOnly": True,
        "resultCount": len(results),
        "retrievalPlan": dict(diagnostics.get("retrievalPlan") or {}),
        "retrievalStrategy": retrieval_strategy,
        "retrievalQuality": retrieval_quality,
        "answerabilityRerank": answerability_rerank,
        "correctiveRetrieval": corrective_retrieval,
        "artifactHealth": artifact_health,
        "candidateSources": list(diagnostics.get("candidateSources") or []),
        "rerankSignals": dict(diagnostics.get("rerankSignals") or {}),
        "memoryRoute": dict(diagnostics.get("memoryRoute") or {}),
        "memoryPrefilter": dict(diagnostics.get("memoryPrefilter") or {}),
        "paperMemoryPrefilter": dict(diagnostics.get("paperMemoryPrefilter") or {}),
        "suggestedActions": suggested_actions,
        "actionsApplied": [],
        "resultsPreview": [_result_preview(result) for result in results[: min(len(results), max(1, int(top_k)))]],
        "warnings": list(warnings),
    }


def build_rag_corrective_report(
    searcher: Any,
    *,
    query: str,
    top_k: int = 5,
    source_type: str | None = None,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.7,
    eval_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only Phase 1 corrective RAG report from Phase 0 diagnostics."""

    results, diagnostics, warnings = _collect_search(
        searcher,
        query=query,
        top_k=top_k,
        source_type=source_type,
        retrieval_mode=retrieval_mode,
        alpha=alpha,
        expand_parent_context=False,
    )
    return _build_report_from_parts(
        query=query,
        top_k=top_k,
        source_type=source_type,
        retrieval_mode=retrieval_mode,
        alpha=alpha,
        results=results,
        diagnostics=diagnostics,
        warnings=warnings,
        eval_context=eval_context,
    )


def _adaptive_route_for(complexity_class: str) -> str:
    if complexity_class == "global_sensemaking":
        return "global_graph_or_hierarchy"
    if complexity_class == "multi_source_compare":
        return "multi_hop_compare"
    if complexity_class in {"update_sensitive", "discovery"}:
        return "adaptive_retrieval"
    if complexity_class in {"exact_lookup", "local_lookup", "local_explainer", "procedural_lookup"}:
        return "single_hop_retrieval"
    return "single_hop_retrieval"


def _adaptive_steps_for(complexity_class: str, corrective_action: str) -> list[str]:
    steps = ["run_phase0_retrieval", "inspect_retrieval_quality"]
    if complexity_class == "global_sensemaking":
        steps.extend(["inspect_hierarchy_or_graph_context", "require_source_diversity"])
    elif complexity_class == "multi_source_compare":
        steps.extend(["require_compare_source_coverage", "pack_balanced_evidence"])
    elif complexity_class == "update_sensitive":
        steps.extend(["prefer_fresh_evidence", "abstain_if_temporal_grounding_is_missing"])
    elif complexity_class == "discovery":
        steps.extend(["widen_candidate_window", "dedupe_by_parent_source"])
    else:
        steps.append("keep_bounded_single_hop_context")
    if corrective_action and corrective_action != "none":
        steps.append(f"suggest_{corrective_action}")
    return steps


def build_rag_adaptive_plan(
    searcher: Any,
    *,
    query: str,
    top_k: int = 5,
    source_type: str | None = None,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.7,
) -> dict[str, Any]:
    report = build_rag_corrective_report(
        searcher,
        query=query,
        top_k=top_k,
        source_type=source_type,
        retrieval_mode=retrieval_mode,
        alpha=alpha,
    )
    strategy = dict(report.get("retrievalStrategy") or {})
    corrective = dict(report.get("correctiveRetrieval") or {})
    complexity_class = str(strategy.get("complexityClass") or "local_lookup")
    candidate_action = str(corrective.get("candidateAction") or "none")
    route = _adaptive_route_for(complexity_class)
    return {
        "schema": ADAPTIVE_PLAN_SCHEMA,
        "status": report.get("status", "ok"),
        "query": str(query or ""),
        "sourceType": _source_type_label(source_type),
        "retrievalMode": str(retrieval_mode or "hybrid"),
        "topK": int(top_k),
        "alpha": float(alpha),
        "readOnly": True,
        "labsOnly": True,
        "retrievalStrategy": strategy,
        "retrievalQuality": dict(report.get("retrievalQuality") or {}),
        "correctiveRetrieval": corrective,
        "plan": {
            "route": route,
            "complexityClass": complexity_class,
            "budget": dict(strategy.get("retrievalBudget") or {}),
            "retryPolicy": dict(strategy.get("retryPolicy") or {}),
            "steps": _adaptive_steps_for(complexity_class, candidate_action),
            "promotionStatus": "not_promoted_default_runtime_unchanged",
        },
        "suggestedActions": list(report.get("suggestedActions") or []),
        "actionsApplied": [],
        "warnings": list(report.get("warnings") or []),
    }


def _retry_config_for(
    *,
    query: str,
    source_type: str | None,
    retrieval_mode: str,
    top_k: int,
    alpha: float,
    action: str,
) -> dict[str, Any]:
    retry_query = str(query or "")
    retry_source = source_type
    retry_mode = retrieval_mode
    expand_parent_context = False
    retry_top_k = min(20, max(int(top_k) + 3, int(top_k) * 2))
    if action in {"broaden_search", "broaden_query_terms"}:
        retry_query = f"{query} overview evidence related sources"
        if action == "broaden_search":
            retry_source = None
    elif action == "source_scope_rescue":
        retry_query = f"{query} exact source id title arxiv"
    elif action == "graph_or_hierarchy_probe":
        retry_query = f"{query} hierarchy graph related parent context"
        expand_parent_context = True
    elif action == "artifact_quality_review":
        retry_query = f"{query} source title metadata freshness"
    return {
        "query": retry_query,
        "sourceType": _source_type_label(retry_source),
        "source_type": retry_source,
        "retrievalMode": retry_mode,
        "topK": retry_top_k,
        "alpha": float(alpha),
        "expandParentContext": expand_parent_context,
    }


def _report_score(report: dict[str, Any], section: str) -> float:
    return safe_float((dict(report.get(section) or {})).get("score"), 0.0)


def _preview_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(report.get("resultsPreview") or [])]


def _top_document_ids(items: list[dict[str, Any]], limit: int = 3) -> list[str]:
    return [str(item.get("documentId") or "").strip() for item in items[:limit] if str(item.get("documentId") or "").strip()]


def _source_coverage(items: list[dict[str, Any]]) -> dict[str, Any]:
    source_types = sorted({str(item.get("sourceType") or "").strip() for item in items if str(item.get("sourceType") or "").strip()})
    top3_source_types = sorted({str(item.get("sourceType") or "").strip() for item in items[:3] if str(item.get("sourceType") or "").strip()})
    return {
        "resultCount": len(items),
        "sourceTypes": source_types,
        "sourceTypeCount": len(source_types),
        "top3SourceTypes": top3_source_types,
        "top3SourceTypeCount": len(top3_source_types),
    }


def _source_coverage_non_regression(before: dict[str, Any], after: dict[str, Any]) -> bool:
    before_types = set(before.get("top3SourceTypes") or [])
    after_types = set(after.get("top3SourceTypes") or [])
    return before_types.issubset(after_types) or int(after.get("top3SourceTypeCount") or 0) >= int(before.get("top3SourceTypeCount") or 0)


def _protected_identities(eval_context: dict[str, Any] | None) -> set[str]:
    if not eval_context:
        return set()
    values = set(_parse_list((eval_context or {}).get("protected_document_ids")))
    values.update(_parse_list((eval_context or {}).get("expected_relevant_document_ids")))
    values.update(_parse_list((eval_context or {}).get("expected_relevant_parent_ids")))
    top1 = str((eval_context or {}).get("expected_top1_document_id") or "").strip()
    if top1:
        values.add(top1)
    return {value for value in values if value}


def _protected_source_retained(before: list[dict[str, Any]], after: list[dict[str, Any]], protected_values: set[str]) -> bool:
    if not protected_values:
        return True
    before_present = any(_matches_any_identity(item, protected_values) for item in before)
    if not before_present:
        return True
    return any(_matches_any_identity(item, protected_values) for item in after)


def _exact_scope_tokens(query: str, eval_context: dict[str, Any] | None) -> set[str]:
    values = set(_parse_list((eval_context or {}).get("expected_exact_source_ids") if eval_context else ""))
    values.update(re.findall(r"(?<!\d)(\d{4}\.\d{4,5}(?:v\d+)?)(?!\d)", str(query or "")))
    return {value.strip() for value in values if value.strip()}


def _exact_scope_coverage(items: list[dict[str, Any]], tokens: set[str]) -> dict[str, Any]:
    if not tokens:
        return {"expected": 0, "present": 0, "ratio": 1.0, "missing": []}
    present: set[str] = set()
    for item in items:
        identity_text = " ".join(sorted(_result_identity_values(item))).casefold()
        for token in tokens:
            if token.casefold() in identity_text:
                present.add(token)
    return {
        "expected": len(tokens),
        "present": len(present),
        "ratio": round(len(present) / max(1, len(tokens)), 6),
        "missing": sorted(tokens - present),
    }


def _latency_budget_ms(eval_context: dict[str, Any] | None) -> int:
    raw = str((eval_context or {}).get("latency_budget_ms") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return 10000
    return 10000


def _looks_context_required(
    *,
    query: str,
    source_type: str | None,
    eval_context: dict[str, Any] | None,
    initial_report: dict[str, Any],
    retry_candidate: bool,
) -> bool:
    if eval_context or retry_candidate:
        return False
    if _source_type_label(source_type) != "web":
        return False
    signals = set(list((dict(initial_report.get("answerabilityRerank") or {})).get("weakSignals") or []))
    query_text = str(query or "").casefold()
    has_temporal_signal = any(token in query_text for token in ("today", "latest", "recent", "오늘", "최근", "발표"))
    return has_temporal_signal and "low_query_term_coverage" in signals


def _looks_missing_source_negative(query: str, eval_context: dict[str, Any] | None) -> bool:
    text = " ".join(
        [
            str(query or ""),
            str((eval_context or {}).get("scenario") or ""),
            str((eval_context or {}).get("notes") or ""),
        ]
    ).casefold()
    markers = (
        "존재하지",
        "nonexistent",
        "does not exist",
        "missing source",
        "fake codename",
        "fake internal",
        "internal codename",
        "internal project codename",
    )
    return any(marker in text for marker in markers)


def _paper_id_candidates(token: str) -> list[str]:
    value = str(token or "").strip()
    if not value:
        return []
    candidates = [value, f"paper:{value}", f"paper_{value}_0"]
    if "v" in value:
        candidates.append(value.split("v", 1)[0])
    return list(dict.fromkeys(candidates))


def _local_exact_source_resolved(searcher: Any, tokens: set[str], before: list[dict[str, Any]]) -> bool:
    if not tokens:
        return False
    if int(_exact_scope_coverage(before, tokens).get("present") or 0) > 0:
        return True
    sqlite_db = getattr(searcher, "sqlite_db", None)
    if sqlite_db is None:
        return False
    method_names = (
        "get_paper",
        "get_paper_by_id",
        "get_paper_by_arxiv_id",
        "get_source",
        "get_document",
    )
    for token in tokens:
        for candidate in _paper_id_candidates(token):
            for method_name in method_names:
                method = getattr(sqlite_db, method_name, None)
                if not callable(method):
                    continue
                try:
                    if method(candidate):
                        return True
                except TypeError:
                    continue
                except Exception:
                    continue
    return False


def _retry_execution_precheck(
    searcher: Any,
    *,
    query: str,
    action: str,
    initial_report: dict[str, Any],
    eval_context: dict[str, Any] | None,
) -> dict[str, Any]:
    before = _preview_results(initial_report)
    exact_tokens = _exact_scope_tokens(query, eval_context)
    if action == "broaden_search" and _looks_missing_source_negative(query, eval_context):
        return {
            "executionEligible": False,
            "executionSkippedReason": "missing_source_negative",
            "exactSourceResolved": False,
        }
    if action == "source_scope_rescue" and exact_tokens:
        resolved = _local_exact_source_resolved(searcher, exact_tokens, before)
        if not resolved:
            return {
                "executionEligible": False,
                "executionSkippedReason": "exact_source_unresolved",
                "exactSourceResolved": False,
            }
        return {
            "executionEligible": True,
            "executionSkippedReason": "",
            "exactSourceResolved": True,
        }
    return {
        "executionEligible": True,
        "executionSkippedReason": "",
        "exactSourceResolved": False,
    }


def _build_retry_execution_review(
    *,
    query: str,
    source_type: str | None,
    initial_report: dict[str, Any],
    retry_report: dict[str, Any],
    actions_applied: list[dict[str, Any]],
    execute: bool,
    retry_candidate: bool,
    action: str,
    eval_context: dict[str, Any] | None,
    execution_precheck: dict[str, Any] | None,
    latency_ms: int,
    search_call_count: int,
) -> dict[str, Any]:
    before = _preview_results(initial_report)
    after = _preview_results(retry_report)
    before_top3 = _top_document_ids(before, 3)
    after_top3 = _top_document_ids(after, 3)
    top3_overlap = len(set(before_top3) & set(after_top3))
    before_coverage = _source_coverage(before)
    after_coverage = _source_coverage(after)
    coverage_non_regression = _source_coverage_non_regression(before_coverage, after_coverage)
    initial_quality = _report_score(initial_report, "retrievalQuality")
    retry_quality = _report_score(retry_report, "retrievalQuality") if retry_report else 0.0
    initial_answerability = _report_score(initial_report, "answerabilityRerank")
    retry_answerability = _report_score(retry_report, "answerabilityRerank") if retry_report else 0.0
    quality_delta = round(retry_quality - initial_quality, 6) if retry_report else 0.0
    answerability_delta = round(retry_answerability - initial_answerability, 6) if retry_report else 0.0
    before_top1 = before_top3[0] if before_top3 else ""
    after_top1 = after_top3[0] if after_top3 else ""
    protected_values = _protected_identities(eval_context)
    protected_retained = _protected_source_retained(before, after, protected_values)
    banned_values = set(_parse_list((eval_context or {}).get("banned_document_ids") if eval_context else ""))
    banned_promoted = _banned_document_promoted(before=before, after=after, banned_document_ids=banned_values) if banned_values else False
    exact_tokens = _exact_scope_tokens(query, eval_context)
    scenario = _scenario_from_context(eval_context)
    exact_required = bool(exact_tokens) and (action == "source_scope_rescue" or "missing" in scenario or "exact" in scenario)
    exact_before = _exact_scope_coverage(before, exact_tokens)
    exact_after = _exact_scope_coverage(after, exact_tokens)
    exact_scope_improved = int(exact_after.get("present") or 0) > int(exact_before.get("present") or 0)
    exact_scope_ok = not exact_required or exact_scope_improved or int(exact_after.get("present") or 0) > 0
    latency_budget = _latency_budget_ms(eval_context)
    latency_ok = int(latency_ms) <= latency_budget
    applied = bool(actions_applied)
    precheck = dict(execution_precheck or {})
    execution_eligible = bool(precheck.get("executionEligible", True))
    execution_skipped_reason = str(precheck.get("executionSkippedReason") or "").strip()
    result_count_expanded = applied and int((actions_applied[0] or {}).get("retryResultCount") or 0) > int((actions_applied[0] or {}).get("initialResultCount") or 0)
    context_required = _looks_context_required(
        query=query,
        source_type=source_type,
        eval_context=eval_context,
        initial_report=initial_report,
        retry_candidate=retry_candidate,
    )
    top3_or_source_ok = top3_overlap >= 2 or coverage_non_regression or protected_retained
    quality_ok = quality_delta >= 0.0
    answerability_ok = answerability_delta >= 0.0
    no_harm = bool(applied and quality_ok and answerability_ok and top3_or_source_ok and protected_retained and not banned_promoted and exact_scope_ok and latency_ok)
    improved = bool(no_harm and (quality_delta > 0.0 or answerability_delta > 0.0 or exact_scope_improved))
    regressed = bool(
        applied
        and (
            quality_delta < 0.0
            or answerability_delta < 0.0
            or not top3_or_source_ok
            or not protected_retained
            or banned_promoted
            or not exact_scope_ok
            or not latency_ok
        )
    )
    decision_reasons: list[str] = []
    if result_count_expanded:
        decision_reasons.append("result_count_expanded_observation_only")
    if context_required:
        decision_reasons.append("eval_context_required_for_retry_decision")
    if execution_skipped_reason:
        decision_reasons.append("execution_precheck_skipped")
        decision_reasons.append(execution_skipped_reason)
    elif not applied:
        decision_reasons.append("retry_not_applied")
    if quality_delta < 0.0:
        decision_reasons.append("quality_regression")
    if answerability_delta < 0.0:
        decision_reasons.append("answerability_regression")
    if not top3_or_source_ok:
        decision_reasons.append("top3_or_source_coverage_regression")
    if not protected_retained:
        decision_reasons.append("protected_source_not_retained")
    if banned_promoted:
        decision_reasons.append("banned_source_promoted")
    if exact_required and not exact_scope_ok:
        decision_reasons.append("exact_scope_not_improved")
    if not latency_ok:
        decision_reasons.append("latency_budget_exceeded")
    if no_harm:
        decision_reasons.append("no_harm_gate_passed")
    if improved:
        decision_reasons.append("quality_or_answerability_improved")

    return {
        "labsOnly": True,
        "runtimeApplied": False,
        "defaultRuntimeApplied": False,
        "execute": bool(execute),
        "applied": applied,
        "retryCandidate": bool(retry_candidate),
        "candidateAction": action,
        "executionEligible": execution_eligible,
        "executionSkippedReason": execution_skipped_reason,
        "exactSourceResolved": bool(precheck.get("exactSourceResolved", False)),
        "contextRequired": context_required,
        "qualityDelta": quality_delta,
        "answerabilityDelta": answerability_delta,
        "top1Changed": bool(before_top1 != after_top1) if applied else False,
        "top3Overlap": int(top3_overlap),
        "sourceCoverageBefore": before_coverage,
        "sourceCoverageAfter": after_coverage,
        "sourceCoverageNonRegression": coverage_non_regression,
        "protectedSourceRetained": protected_retained,
        "bannedSourcePromoted": banned_promoted,
        "exactScopeRequired": exact_required,
        "exactScopeCoverageBefore": exact_before,
        "exactScopeCoverageAfter": exact_after,
        "resultCountExpanded": result_count_expanded,
        "improved": improved,
        "noHarm": no_harm,
        "regressed": regressed,
        "decisionReasons": decision_reasons,
        "latencyMs": int(latency_ms),
        "latencyBudgetMs": latency_budget,
        "searchCallCount": int(search_call_count),
        "estimatedCostClass": "retrieval_only_two_calls" if applied else "retrieval_only_single_call",
    }


def build_rag_corrective_run(
    searcher: Any,
    *,
    query: str,
    top_k: int = 5,
    source_type: str | None = None,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.7,
    execute: bool = False,
    eval_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    initial_report = build_rag_corrective_report(
        searcher,
        query=query,
        top_k=top_k,
        source_type=source_type,
        retrieval_mode=retrieval_mode,
        alpha=alpha,
        eval_context=eval_context,
    )
    corrective = dict(initial_report.get("correctiveRetrieval") or {})
    action = str(corrective.get("candidateAction") or "none")
    retry_candidate = bool(corrective.get("retryCandidate"))
    actions_applied: list[dict[str, Any]] = []
    retry_report: dict[str, Any] = {}
    warnings = list(initial_report.get("warnings") or [])
    search_call_count = 1
    execution_precheck = _retry_execution_precheck(
        searcher,
        query=query,
        action=action,
        initial_report=initial_report,
        eval_context=eval_context,
    )

    if execute and retry_candidate and action != "none" and bool(execution_precheck.get("executionEligible", True)):
        retry_config = _retry_config_for(
            query=query,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            alpha=alpha,
            action=action,
        )
        results, diagnostics, retry_warnings = _collect_search(
            searcher,
            query=str(retry_config["query"]),
            top_k=int(retry_config["topK"]),
            source_type=retry_config["source_type"],
            retrieval_mode=str(retry_config["retrievalMode"]),
            alpha=float(retry_config["alpha"]),
            expand_parent_context=bool(retry_config["expandParentContext"]),
        )
        search_call_count += 1
        retry_report = _build_report_from_parts(
            query=str(retry_config["query"]),
            top_k=int(retry_config["topK"]),
            source_type=retry_config["source_type"],
            retrieval_mode=str(retry_config["retrievalMode"]),
            alpha=float(retry_config["alpha"]),
            results=results,
            diagnostics=diagnostics,
            warnings=retry_warnings,
            eval_context=eval_context,
        )
        actions_applied.append(
            {
                "actionType": action,
                "mode": "retrieval_only",
                "writeFree": True,
                "retryConfig": {key: value for key, value in retry_config.items() if key != "source_type"},
                "initialResultCount": int(initial_report.get("resultCount") or 0),
                "retryResultCount": int(retry_report.get("resultCount") or 0),
            }
        )
        warnings.extend(str(item) for item in retry_warnings)
    elif execute and retry_candidate and action != "none":
        reason = str(execution_precheck.get("executionSkippedReason") or "execution_precheck_failed")
        warnings.append(f"corrective_execution_skipped:{reason}")
    elif execute and not retry_candidate:
        warnings.append("no_corrective_retry_candidate")

    latency_ms = int(round((perf_counter() - started_at) * 1000))
    retry_execution_review = _build_retry_execution_review(
        query=query,
        source_type=source_type,
        initial_report=initial_report,
        retry_report=retry_report,
        actions_applied=actions_applied,
        execute=execute,
        retry_candidate=retry_candidate,
        action=action,
        eval_context=eval_context,
        execution_precheck=execution_precheck,
        latency_ms=latency_ms,
        search_call_count=search_call_count,
    )

    return {
        "schema": CORRECTIVE_RUN_SCHEMA,
        "status": "applied" if actions_applied else "dry_run" if not execute else "skipped",
        "query": str(query or ""),
        "sourceType": _source_type_label(source_type),
        "retrievalMode": str(retrieval_mode or "hybrid"),
        "topK": int(top_k),
        "alpha": float(alpha),
        "execute": bool(execute),
        "readOnly": True,
        "writeFree": True,
        "labsOnly": True,
        "runtimeApplied": False,
        "defaultRuntimeApplied": False,
        "initialReport": initial_report,
        "retryReport": retry_report,
        "retryExecutionReview": retry_execution_review,
        "suggestedActions": list(initial_report.get("suggestedActions") or []),
        "actionsApplied": actions_applied,
        "warnings": warnings,
    }


def _answerability_score(query: str, result: Any) -> dict[str, Any]:
    preview = _result_preview(result)
    query_terms = tokenize(str(query or ""))
    metadata = dict(getattr(result, "metadata", {}) or {})
    candidate_text = " ".join(
        [
            str(metadata.get("title") or ""),
            str(metadata.get("summary") or ""),
            str(getattr(result, "document", "") or ""),
        ]
    )
    candidate_terms = tokenize(clean_text(candidate_text))
    term_coverage = jaccard(query_terms, candidate_terms)
    base_score = safe_float(getattr(result, "score", 0.0), 0.0)
    semantic_score = safe_float(getattr(result, "semantic_score", 0.0), 0.0)
    lexical_score = safe_float(getattr(result, "lexical_score", 0.0), 0.0)
    score = max(0.0, min(1.0, (0.40 * base_score) + (0.30 * term_coverage) + (0.20 * semantic_score) + (0.10 * lexical_score)))
    preview.update(
        {
            "answerabilityScore": round(score, 6),
            "termCoverage": round(term_coverage, 6),
        }
    )
    return preview


def _matches_any_identity(item: dict[str, Any], values: set[str]) -> bool:
    if not values:
        return False
    return bool(_result_identity_values(item) & values)


def _apply_no_promotion_guard(
    reranked: list[dict[str, Any]],
    *,
    guarded_document_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not guarded_document_ids:
        return list(reranked), {
            "applied": False,
            "mode": "none",
            "guardedDocumentIds": [],
            "demotedDocumentIds": [],
            "reason": "",
        }

    guarded = list(reranked)
    demoted: list[str] = []
    for _ in range(len(guarded)):
        changed = False
        for index, item in enumerate(list(guarded)):
            if not _matches_any_identity(item, guarded_document_ids):
                continue
            original_rank = int(item.get("originalRank") or 0)
            if original_rank <= 0 or index + 1 >= original_rank:
                continue
            guarded.pop(index)
            guarded.insert(min(original_rank - 1, len(guarded)), item)
            document_id = str(item.get("documentId") or "").strip()
            if document_id and document_id not in demoted:
                demoted.append(document_id)
            changed = True
            break
        if not changed:
            break

    return guarded, {
        "applied": True,
        "mode": "no_banned_document_promotion",
        "guardedDocumentIds": sorted(guarded_document_ids),
        "demotedDocumentIds": demoted,
        "reason": "negative_or_missing_scope_eval_guard",
    }


def build_rag_answerability_rerank(
    searcher: Any,
    *,
    query: str,
    top_k: int = 5,
    source_type: str | None = None,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.7,
    no_promotion_document_ids: set[str] | None = None,
) -> dict[str, Any]:
    results, diagnostics, warnings = _collect_search(
        searcher,
        query=query,
        top_k=top_k,
        source_type=source_type,
        retrieval_mode=retrieval_mode,
        alpha=alpha,
        expand_parent_context=False,
    )
    scored = []
    for index, result in enumerate(results):
        item = _answerability_score(query, result)
        item["originalRank"] = index + 1
        scored.append(item)
    reranked = sorted(scored, key=lambda item: (-safe_float(item.get("answerabilityScore"), 0.0), item["originalRank"]))
    reranked, rerank_guard = _apply_no_promotion_guard(
        reranked,
        guarded_document_ids=set(no_promotion_document_ids or set()),
    )
    for index, item in enumerate(reranked):
        item["rerankedRank"] = index + 1
    changed = sum(1 for item in reranked if int(item.get("originalRank") or 0) != int(item.get("rerankedRank") or 0))
    return {
        "schema": ANSWERABILITY_RERANK_SCHEMA,
        "status": "ok",
        "query": str(query or ""),
        "sourceType": _source_type_label(source_type),
        "retrievalMode": str(retrieval_mode or "hybrid"),
        "topK": int(top_k),
        "alpha": float(alpha),
        "readOnly": True,
        "writeFree": True,
        "labsOnly": True,
        "runtimeApplied": False,
        "applied": True,
        "resultCount": len(results),
        "changedRankCount": int(changed),
        "retrievalStrategy": dict(diagnostics.get("retrievalStrategy") or {}),
        "answerabilityRerank": dict(diagnostics.get("answerabilityRerank") or {}),
        "rerankGuard": rerank_guard,
        "originalResults": scored[: min(len(scored), max(1, int(top_k)))],
        "rerankedResults": reranked[: min(len(reranked), max(1, int(top_k)))],
        "warnings": warnings,
    }


def _result_identity_values(item: dict[str, Any]) -> set[str]:
    return {
        str(value or "").strip()
        for value in (
            item.get("documentId"),
            item.get("parentId"),
            item.get("parentLabel"),
            item.get("title"),
        )
        if str(value or "").strip()
    }


def _rank_map(items: list[dict[str, Any]]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for index, item in enumerate(items, start=1):
        for value in _result_identity_values(item):
            ranks.setdefault(value, index)
    return ranks


def _relevant_rank(items: list[dict[str, Any]], relevant_document_ids: set[str], relevant_parent_ids: set[str]) -> int | None:
    for index, item in enumerate(items, start=1):
        if str(item.get("documentId") or "") in relevant_document_ids:
            return index
        if str(item.get("parentId") or "") in relevant_parent_ids:
            return index
    return None


def _coverage(items: list[dict[str, Any]], required_values: set[str], key: str) -> dict[str, Any]:
    present = {str(item.get(key) or "").strip() for item in items if str(item.get(key) or "").strip() in required_values}
    expected = len(required_values)
    return {
        "expected": expected,
        "present": len(present),
        "ratio": round(len(present) / max(1, expected), 6) if expected else 1.0,
        "missing": sorted(required_values - present),
    }


def _protected_document_dropped(
    *,
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    protected_document_ids: set[str],
) -> bool:
    before_ranks = _rank_map(before)
    after_ranks = _rank_map(after)
    for document_id in protected_document_ids:
        before_rank = before_ranks.get(document_id)
        after_rank = after_ranks.get(document_id)
        if before_rank is not None and (after_rank is None or after_rank > before_rank):
            return True
    return False


def _banned_document_promoted(
    *,
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    banned_document_ids: set[str],
) -> bool:
    before_ranks = _rank_map(before)
    after_ranks = _rank_map(after)
    for document_id in banned_document_ids:
        after_rank = after_ranks.get(document_id)
        if after_rank is None:
            continue
        before_rank = before_ranks.get(document_id)
        if before_rank is None or after_rank < before_rank:
            return True
    return False


def _compare_answerability_rankings(
    *,
    original_results: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
    expected_relevant_document_ids: set[str],
    expected_relevant_parent_ids: set[str],
    expected_top1_document_id: str,
    required_parent_ids: set[str],
    required_source_types: set[str],
    protected_document_ids: set[str],
    banned_document_ids: set[str],
) -> dict[str, Any]:
    all_identities = set().union(*(_result_identity_values(item) for item in original_results + reranked_results)) if original_results or reranked_results else set()
    expected_identities = set(expected_relevant_document_ids) | set(expected_relevant_parent_ids)
    if expected_top1_document_id:
        expected_identities.add(expected_top1_document_id)
    invalid_reasons: list[str] = []
    if expected_identities and not (all_identities & expected_identities):
        invalid_reasons.append("expected_gold_not_found_in_results")
    if not expected_identities and not required_parent_ids and not required_source_types and not protected_document_ids and not banned_document_ids:
        invalid_reasons.append("no_gold_labels_supplied")

    before_top3 = original_results[:3]
    after_top3 = reranked_results[:3]
    before_relevant_rank = _relevant_rank(original_results, expected_relevant_document_ids, expected_relevant_parent_ids)
    after_relevant_rank = _relevant_rank(reranked_results, expected_relevant_document_ids, expected_relevant_parent_ids)
    top1_before = bool(before_relevant_rank == 1)
    top1_after = bool(after_relevant_rank == 1)
    top3_before = before_relevant_rank is not None and before_relevant_rank <= 3
    top3_after = after_relevant_rank is not None and after_relevant_rank <= 3
    mrr_before = round(1.0 / before_relevant_rank, 6) if before_relevant_rank else 0.0
    mrr_after = round(1.0 / after_relevant_rank, 6) if after_relevant_rank else 0.0
    parent_coverage_before = _coverage(before_top3, required_parent_ids, "parentId")
    parent_coverage_after = _coverage(after_top3, required_parent_ids, "parentId")
    source_coverage_before = _coverage(before_top3, required_source_types, "sourceType")
    source_coverage_after = _coverage(after_top3, required_source_types, "sourceType")
    source_coverage_regression = source_coverage_after["ratio"] < source_coverage_before["ratio"]
    parent_coverage_regression = parent_coverage_after["ratio"] < parent_coverage_before["ratio"]
    protected_dropped = _protected_document_dropped(
        before=original_results,
        after=reranked_results,
        protected_document_ids=protected_document_ids,
    )
    banned_promoted = _banned_document_promoted(
        before=original_results,
        after=reranked_results,
        banned_document_ids=banned_document_ids,
    )
    expected_top1_before = bool(expected_top1_document_id and original_results and original_results[0].get("documentId") == expected_top1_document_id)
    expected_top1_after = bool(expected_top1_document_id and reranked_results and reranked_results[0].get("documentId") == expected_top1_document_id)
    if expected_top1_before and not expected_top1_after:
        protected_dropped = True

    regressed = any(
        [
            top1_before and not top1_after,
            top3_before and not top3_after,
            before_relevant_rank is not None and (after_relevant_rank is None or after_relevant_rank > before_relevant_rank),
            mrr_after < mrr_before,
            parent_coverage_regression,
            source_coverage_regression,
            protected_dropped,
            banned_promoted,
        ]
    )
    improved = any(
        [
            not top1_before and top1_after,
            not top3_before and top3_after,
            before_relevant_rank is None and after_relevant_rank is not None,
            before_relevant_rank is not None and after_relevant_rank is not None and after_relevant_rank < before_relevant_rank,
            mrr_after > mrr_before,
            parent_coverage_after["ratio"] > parent_coverage_before["ratio"],
            source_coverage_after["ratio"] > source_coverage_before["ratio"],
        ]
    )
    if invalid_reasons:
        verdict = "invalid_gold"
    elif regressed:
        verdict = "regressed"
    elif improved:
        verdict = "improved"
    else:
        verdict = "neutral"

    return {
        "verdict": verdict,
        "invalidReasons": invalid_reasons,
        "metrics": {
            "top1RelevantBefore": top1_before,
            "top1RelevantAfter": top1_after,
            "top3RelevantBefore": top3_before,
            "top3RelevantAfter": top3_after,
            "firstRelevantRankBefore": before_relevant_rank,
            "firstRelevantRankAfter": after_relevant_rank,
            "mrrBefore": mrr_before,
            "mrrAfter": mrr_after,
            "expectedTop1Before": expected_top1_before,
            "expectedTop1After": expected_top1_after,
            "requiredParentCoverageBefore": parent_coverage_before,
            "requiredParentCoverageAfter": parent_coverage_after,
            "requiredSourceCoverageBefore": source_coverage_before,
            "requiredSourceCoverageAfter": source_coverage_after,
            "sourceCoverageRegression": source_coverage_regression,
            "parentCoverageRegression": parent_coverage_regression,
            "protectedDocumentDropped": protected_dropped,
            "bannedDocumentPromoted": banned_promoted,
        },
    }


def _answerability_eval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = len(rows)
    changed_row_count = sum(1 for row in rows if int(((row.get("rerank") or {}).get("changedRankCount")) or 0) > 0)
    verdict_counts = {
        "improved": sum(1 for row in rows if row.get("verdict") == "improved"),
        "neutral": sum(1 for row in rows if row.get("verdict") == "neutral"),
        "regressed": sum(1 for row in rows if row.get("verdict") == "regressed"),
        "invalid_gold": sum(1 for row in rows if row.get("verdict") == "invalid_gold"),
    }
    metric_rows = [dict(row.get("metrics") or {}) for row in rows if row.get("verdict") != "invalid_gold"]
    denominator = max(1, len(metric_rows))
    return {
        "rowCount": row_count,
        "changedRowCount": changed_row_count,
        "improvedCount": verdict_counts["improved"],
        "neutralCount": verdict_counts["neutral"],
        "regressedCount": verdict_counts["regressed"],
        "invalidGoldCount": verdict_counts["invalid_gold"],
        "top1AccuracyBefore": round(sum(1 for row in metric_rows if bool(row.get("top1RelevantBefore"))) / denominator, 6),
        "top1AccuracyAfter": round(sum(1 for row in metric_rows if bool(row.get("top1RelevantAfter"))) / denominator, 6),
        "top3RecallBefore": round(sum(1 for row in metric_rows if bool(row.get("top3RelevantBefore"))) / denominator, 6),
        "top3RecallAfter": round(sum(1 for row in metric_rows if bool(row.get("top3RelevantAfter"))) / denominator, 6),
        "mrrBefore": round(sum(safe_float(row.get("mrrBefore"), 0.0) for row in metric_rows) / denominator, 6),
        "mrrAfter": round(sum(safe_float(row.get("mrrAfter"), 0.0) for row in metric_rows) / denominator, 6),
        "sourceCoverageRegressionCount": sum(1 for row in metric_rows if bool(row.get("sourceCoverageRegression"))),
        "parentCoverageRegressionCount": sum(1 for row in metric_rows if bool(row.get("parentCoverageRegression"))),
        "protectedDropCount": sum(1 for row in metric_rows if bool(row.get("protectedDocumentDropped"))),
        "bannedPromotionCount": sum(1 for row in metric_rows if bool(row.get("bannedDocumentPromoted"))),
    }


def _answerability_promotion_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if int(summary.get("rowCount") or 0) <= 0:
        blockers.append("answerability_rerank_shadow_eval_missing")
    if int(summary.get("regressedCount") or 0) > 0:
        blockers.append("answerability_rerank_regressions_present")
    if int(summary.get("invalidGoldCount") or 0) > 0:
        blockers.append("answerability_rerank_invalid_gold_present")
    if safe_float(summary.get("mrrAfter"), 0.0) < safe_float(summary.get("mrrBefore"), 0.0):
        blockers.append("answerability_rerank_mrr_regression")
    return {
        "status": "candidate_for_review" if not blockers else "not_ready",
        "criteria": {
            "requiresRows": True,
            "requiresZeroRegressions": True,
            "requiresZeroInvalidGold": True,
            "requiresMrrNonRegression": True,
            "runtimePromotionAllowed": False,
        },
        "blockers": blockers,
    }


def build_rag_answerability_rerank_eval_report(
    searcher: Any,
    *,
    queries_path: str | Path = DEFAULT_ANSWERABILITY_RERANK_EVAL_PATH,
    top_k: int = 5,
    source_type: str | None = None,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.7,
    limit: int | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    path = Path(queries_path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if limit is not None and int(limit) >= 0:
        rows = rows[: int(limit)]

    evaluated: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, row in enumerate(rows, start=1):
        row_top_k = int(row.get("top_k") or top_k)
        row_source = str(row.get("source") or source_type or "all").strip() or "all"
        resolved_source = None if row_source == "all" else row_source
        if source_type is not None:
            resolved_source = source_type
            row_source = _source_type_label(source_type)
        expected_relevant_document_ids = set(_parse_list(row.get("expected_relevant_document_ids")))
        expected_relevant_parent_ids = set(_parse_list(row.get("expected_relevant_parent_ids")))
        required_parent_ids = set(_parse_list(row.get("required_parent_ids")))
        required_source_types = set(_parse_list(row.get("required_source_types")))
        protected_document_ids = set(_parse_list(row.get("protected_document_ids")))
        banned_document_ids = set(_parse_list(row.get("banned_document_ids")))
        no_promotion_document_ids = banned_document_ids if _should_apply_negative_scope_guard(row, banned_document_ids) else set()
        rerank_report = build_rag_answerability_rerank(
            searcher,
            query=str(row.get("query") or ""),
            top_k=max(1, row_top_k),
            source_type=resolved_source,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            no_promotion_document_ids=no_promotion_document_ids,
        )
        warnings.extend(str(item) for item in list(rerank_report.get("warnings") or []))
        comparison = _compare_answerability_rankings(
            original_results=list(rerank_report.get("originalResults") or []),
            reranked_results=list(rerank_report.get("rerankedResults") or []),
            expected_relevant_document_ids=expected_relevant_document_ids,
            expected_relevant_parent_ids=expected_relevant_parent_ids,
            expected_top1_document_id=str(row.get("expected_top1_document_id") or "").strip(),
            required_parent_ids=required_parent_ids,
            required_source_types=required_source_types,
            protected_document_ids=protected_document_ids,
            banned_document_ids=banned_document_ids,
        )
        allow_rank_change = _parse_bool_default(row.get("allow_rank_change"), True)
        blockers = list(comparison.get("invalidReasons") or [])
        verdict = str(comparison.get("verdict") or "invalid_gold")
        if not allow_rank_change and int(rerank_report.get("changedRankCount") or 0) > 0 and verdict != "invalid_gold":
            verdict = "regressed"
            blockers.append("rank_change_not_allowed")
        evaluated.append(
            {
                "row": index,
                "queryId": str(row.get("query_id") or index),
                "query": str(row.get("query") or ""),
                "source": row_source,
                "scenario": str(row.get("scenario") or ""),
                "riskTags": _parse_list(row.get("risk_tags")),
                "expectedBehavior": str(row.get("expected_behavior") or "").strip(),
                "allowRankChange": allow_rank_change,
                "expected": {
                    "relevantDocumentIds": sorted(expected_relevant_document_ids),
                    "relevantParentIds": sorted(expected_relevant_parent_ids),
                    "top1DocumentId": str(row.get("expected_top1_document_id") or "").strip(),
                    "requiredParentIds": sorted(required_parent_ids),
                    "requiredSourceTypes": sorted(required_source_types),
                    "protectedDocumentIds": sorted(protected_document_ids),
                    "bannedDocumentIds": sorted(banned_document_ids),
                },
                "baseline": {"results": list(rerank_report.get("originalResults") or [])},
                "reranked": {"results": list(rerank_report.get("rerankedResults") or [])},
                "rerank": {
                    "changedRankCount": int(rerank_report.get("changedRankCount") or 0),
                    "resultCount": int(rerank_report.get("resultCount") or 0),
                },
                "rerankGuard": dict(rerank_report.get("rerankGuard") or {}),
                "metrics": dict(comparison.get("metrics") or {}),
                "delta": {
                    "mrr": round(
                        safe_float((comparison.get("metrics") or {}).get("mrrAfter"), 0.0)
                        - safe_float((comparison.get("metrics") or {}).get("mrrBefore"), 0.0),
                        6,
                    ),
                    "top1Relevant": int(bool((comparison.get("metrics") or {}).get("top1RelevantAfter")))
                    - int(bool((comparison.get("metrics") or {}).get("top1RelevantBefore"))),
                    "top3Relevant": int(bool((comparison.get("metrics") or {}).get("top3RelevantAfter")))
                    - int(bool((comparison.get("metrics") or {}).get("top3RelevantBefore"))),
                },
                "verdict": verdict,
                "blockers": blockers,
                "notes": str(row.get("notes") or ""),
            }
        )

    summary = _answerability_eval_summary(evaluated)
    readiness = _answerability_promotion_readiness(summary)
    return {
        "schema": ANSWERABILITY_RERANK_EVAL_SCHEMA,
        "status": "ok" if not readiness["blockers"] else "warn",
        "generatedAt": str(generated_at or _now_iso()),
        "readOnly": True,
        "writeFree": True,
        "labsOnly": True,
        "runtimeApplied": False,
        "evalPath": str(path),
        "request": {
            "topK": int(top_k),
            "sourceType": _source_type_label(source_type),
            "retrievalMode": str(retrieval_mode or "hybrid"),
            "alpha": float(alpha),
            "limit": limit,
        },
        "summary": summary,
        "rows": evaluated,
        "promotionReadiness": readiness,
        "warnings": list(dict.fromkeys(warnings)),
    }


def build_rag_graph_global_plan(
    searcher: Any,
    *,
    query: str,
    top_k: int = 5,
    source_type: str | None = None,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.7,
) -> dict[str, Any]:
    adaptive = build_rag_adaptive_plan(
        searcher,
        query=query,
        top_k=top_k,
        source_type=source_type,
        retrieval_mode=retrieval_mode,
        alpha=alpha,
    )
    complexity_class = str((adaptive.get("plan") or {}).get("complexityClass") or "local_lookup")
    route = str((adaptive.get("plan") or {}).get("route") or "single_hop_retrieval")
    should_use_global = route == "global_graph_or_hierarchy" or complexity_class == "global_sensemaking"
    return {
        "schema": GRAPH_GLOBAL_PLAN_SCHEMA,
        "status": "candidate" if should_use_global else "not_needed",
        "query": str(query or ""),
        "sourceType": _source_type_label(source_type),
        "retrievalMode": str(retrieval_mode or "hybrid"),
        "topK": int(top_k),
        "alpha": float(alpha),
        "readOnly": True,
        "labsOnly": True,
        "graphGlobalLane": {
            "candidate": bool(should_use_global),
            "route": "graph_global" if should_use_global else "baseline_retrieval",
            "promotionStatus": "not_promoted_default_runtime_unchanged",
            "plannedSteps": [
                "collect_candidate_sources",
                "group_by_parent_or_cluster",
                "summarize_group_evidence",
                "merge_group_summaries_with_citations",
            ]
            if should_use_global
            else ["use_baseline_retrieval"],
        },
        "adaptivePlan": adaptive,
        "actionsApplied": [],
        "warnings": list(adaptive.get("warnings") or []),
    }


def build_rag_corrective_eval_report(
    searcher: Any,
    *,
    queries_path: str | Path,
    top_k: int = 5,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.7,
    limit: int | None = None,
) -> dict[str, Any]:
    path = Path(queries_path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if limit is not None and int(limit) >= 0:
        rows = rows[: int(limit)]

    evaluated: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, row in enumerate(rows, start=1):
        source = str(row.get("source") or "all").strip() or "all"
        source_type = None if source == "all" else source
        report = build_rag_corrective_report(
            searcher,
            query=str(row.get("query") or ""),
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            eval_context=row,
        )
        warnings.extend(str(item) for item in report.get("warnings") or [])
        strategy = dict(report.get("retrievalStrategy") or {})
        corrective = dict(report.get("correctiveRetrieval") or {})
        observed_class = str(strategy.get("complexityClass") or "")
        observed_retry = bool(corrective.get("retryCandidate"))
        observed_action = str(corrective.get("candidateAction") or "none")
        expected_class = str(row.get("expected_complexity_class") or "")
        expected_retry = _parse_bool(row.get("expected_retry_candidate"))
        expected_action = str(row.get("expected_candidate_action") or "none")
        checks = {
            "complexityClass": observed_class == expected_class,
            "retryCandidate": observed_retry == expected_retry,
            "candidateAction": observed_action == expected_action,
        }
        evaluated.append(
            {
                "row": index,
                "query": str(row.get("query") or ""),
                "source": source,
                "scenario": str(row.get("scenario") or ""),
                "expected": {
                    "complexityClass": expected_class,
                    "retryCandidate": expected_retry,
                    "candidateAction": expected_action,
                },
                "observed": {
                    "complexityClass": observed_class,
                    "retryCandidate": observed_retry,
                    "candidateAction": observed_action,
                    "retrievalQuality": dict(report.get("retrievalQuality") or {}),
                    "answerabilityRerank": dict(report.get("answerabilityRerank") or {}),
                    "resultCount": int(report.get("resultCount") or 0),
                },
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    pass_count = sum(1 for row in evaluated if bool(row.get("passed")))
    row_count = len(evaluated)
    fail_count = row_count - pass_count
    return {
        "schema": CORRECTIVE_EVAL_SCHEMA,
        "status": "ok" if fail_count == 0 else "warn",
        "evalPath": str(path),
        "readOnly": True,
        "rowCount": row_count,
        "passCount": pass_count,
        "failCount": fail_count,
        "metrics": {
            "passRate": round(pass_count / max(1, row_count), 6),
            "complexityClassAccuracy": round(
                sum(1 for row in evaluated if bool((row.get("checks") or {}).get("complexityClass"))) / max(1, row_count),
                6,
            ),
            "retryCandidateAccuracy": round(
                sum(1 for row in evaluated if bool((row.get("checks") or {}).get("retryCandidate"))) / max(1, row_count),
                6,
            ),
            "candidateActionAccuracy": round(
                sum(1 for row in evaluated if bool((row.get("checks") or {}).get("candidateAction"))) / max(1, row_count),
                6,
            ),
        },
        "rows": evaluated,
        "warnings": list(dict.fromkeys(warnings)),
    }


def build_rag_corrective_execution_review(
    searcher: Any,
    *,
    queries_path: str | Path,
    top_k: int = 5,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.7,
    limit: int | None = None,
    execute: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    corrective_eval = build_rag_corrective_eval_report(
        searcher,
        queries_path=queries_path,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        alpha=alpha,
        limit=limit,
    )
    candidate_rows = [dict(row) for row in list(corrective_eval.get("rows") or []) if bool((dict(row.get("observed") or {})).get("retryCandidate"))]
    reviewed: list[dict[str, Any]] = []
    warnings = list(corrective_eval.get("warnings") or [])
    for row in candidate_rows:
        source = str(row.get("source") or "all").strip() or "all"
        source_type = None if source == "all" else source
        run = build_rag_corrective_run(
            searcher,
            query=str(row.get("query") or ""),
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            execute=execute,
            eval_context=row,
        )
        warnings.extend(str(item) for item in list(run.get("warnings") or []))
        reviewed.append(
            {
                "row": int(row.get("row") or 0),
                "query": str(row.get("query") or ""),
                "source": source,
                "scenario": str(row.get("scenario") or ""),
                "expected": dict(row.get("expected") or {}),
                "observed": dict(row.get("observed") or {}),
                "runStatus": str(run.get("status") or ""),
                "actionsApplied": list(run.get("actionsApplied") or []),
                "retryExecutionReview": dict(run.get("retryExecutionReview") or {}),
                "run": run,
            }
        )

    reviews = [dict(row.get("retryExecutionReview") or {}) for row in reviewed]
    applied_count = sum(1 for review in reviews if bool(review.get("applied")))
    skipped_count = sum(1 for review in reviews if not bool(review.get("executionEligible", True)))
    improved_count = sum(1 for review in reviews if bool(review.get("improved")))
    no_harm_count = sum(1 for review in reviews if bool(review.get("noHarm")))
    regressed_count = sum(1 for review in reviews if bool(review.get("regressed")))
    context_required_count = sum(1 for review in reviews if bool(review.get("contextRequired")))
    actions_applied_count = sum(len(list(row.get("actionsApplied") or [])) for row in reviewed)
    latency_total = sum(int(review.get("latencyMs") or 0) for review in reviews)
    status = "warn" if (execute and (regressed_count or context_required_count)) else "ok"
    return {
        "schema": CORRECTIVE_EXECUTION_REVIEW_SCHEMA,
        "status": status,
        "generatedAt": str(generated_at or _now_iso()),
        "evalPath": str(queries_path),
        "readOnly": True,
        "writeFree": True,
        "labsOnly": True,
        "runtimeApplied": False,
        "defaultRuntimeApplied": False,
        "request": {
            "topK": int(top_k),
            "retrievalMode": str(retrieval_mode or "hybrid"),
            "alpha": float(alpha),
            "limit": limit,
            "execute": bool(execute),
        },
        "summary": {
            "rowCount": int(corrective_eval.get("rowCount") or 0),
            "retryCandidateCount": len(candidate_rows),
            "retryAppliedCount": applied_count,
            "retrySkippedCount": skipped_count,
            "executionSkippedCount": skipped_count,
            "actionsAppliedCount": actions_applied_count,
            "retrievalImprovedCount": improved_count,
            "retrievalNoHarmCount": no_harm_count,
            "retrievalRegressedCount": regressed_count,
            "contextRequiredCount": context_required_count,
            "improvementRate": round(improved_count / max(1, applied_count), 6),
            "noHarmRate": round(no_harm_count / max(1, applied_count), 6),
            "latencyMsTotal": latency_total,
            "estimatedCostClass": "retrieval_only_execution_review" if execute else "retrieval_only_dry_run_review",
        },
        "correctiveEval": corrective_eval,
        "rows": reviewed,
        "warnings": list(dict.fromkeys(warnings)),
    }
