"""RAG vNext observation loop helpers.

The loop is intentionally observational: it inspects labs-only retry candidates,
but it does not write source data, change default ranking, or generate answers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_hub.application.rag_corrective_report import (
    DEFAULT_ANSWERABILITY_RERANK_EVAL_PATH,
    build_rag_answerability_rerank,
    build_rag_answerability_rerank_eval_report,
    build_rag_corrective_eval_report,
    build_rag_corrective_run,
    build_rag_graph_global_plan,
)


RAG_VNEXT_OBSERVATION_SCHEMA = "knowledge-hub.rag.vnext-observation.report.v1"
DEFAULT_CORRECTIVE_EVAL_PATH = "eval/knowledgeos/queries/rag_vnext_phase1_corrective_eval_queries_v1.csv"


def _source_type(source: str) -> str | None:
    return None if str(source or "all").strip() == "all" else str(source or "").strip()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_query(row: dict[str, Any]) -> str:
    return str(row.get("query") or "").strip()


def _row_source(row: dict[str, Any]) -> str:
    return str(row.get("source") or "all").strip() or "all"


def _observed_retry_candidate(row: dict[str, Any]) -> bool:
    return bool(dict(row.get("observed") or {}).get("retryCandidate"))


def _retry_improved(run: dict[str, Any]) -> bool:
    review = dict(run.get("retryExecutionReview") or {})
    if "improved" in review:
        return bool(review.get("improved"))
    action = next(iter(list(run.get("actionsApplied") or [])), {})
    initial_count = int(action.get("initialResultCount") or 0)
    retry_count = int(action.get("retryResultCount") or 0)
    initial_quality = _float(
        ((run.get("initialReport") or {}).get("retrievalQuality") or {}).get("score"),
        0.0,
    )
    retry_quality = _float(
        ((run.get("retryReport") or {}).get("retrievalQuality") or {}).get("score"),
        0.0,
    )
    return retry_count > initial_count or retry_quality > initial_quality


def _promotion_readiness(
    *,
    observation_count: int,
    corrective_eval: dict[str, Any],
    retry_runs: list[dict[str, Any]],
    rerank_reports: list[dict[str, Any]],
    rerank_shadow_eval: dict[str, Any],
    graph_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = dict(corrective_eval.get("metrics") or {})
    pass_rate = _float(metrics.get("passRate"), 0.0)
    applied_retry_runs = [item for item in retry_runs if list(item.get("actionsApplied") or [])]
    retry_improvement_count = sum(1 for item in applied_retry_runs if _retry_improved(item))
    retry_improvement_rate = retry_improvement_count / max(1, len(applied_retry_runs))
    rerank_changed_count = sum(int(item.get("changedRankCount") or 0) for item in rerank_reports)
    graph_candidate_count = sum(1 for item in graph_reports if bool(((item.get("graphGlobalLane") or {}).get("candidate"))))
    blockers: list[str] = []
    if observation_count < 7:
        blockers.append("needs_at_least_7_observations")
    if pass_rate < 0.8:
        blockers.append("corrective_eval_pass_rate_below_0_8")
    if applied_retry_runs and retry_improvement_rate < 0.6:
        blockers.append("corrective_retry_improvement_rate_below_0_6")
    shadow_summary = dict(rerank_shadow_eval.get("summary") or {})
    if int(shadow_summary.get("rowCount") or 0) <= 0:
        blockers.append("answerability_rerank_shadow_eval_missing")
    if int(shadow_summary.get("regressedCount") or 0) > 0:
        blockers.append("answerability_rerank_regressions_present")
    if rerank_changed_count and not rerank_reports:
        blockers.append("answerability_rerank_uninspected")
    if graph_candidate_count and not graph_reports:
        blockers.append("graph_global_candidates_uninspected")
    return {
        "status": "candidate_for_review" if not blockers else "not_ready",
        "observationCount": int(observation_count),
        "minimumObservationCount": 7,
        "criteria": {
            "passRateAtLeast": 0.8,
            "retryImprovementRateAtLeast": 0.6,
            "requiresNoUninspectedRerankRegression": True,
            "requiresGraphCandidatesExplained": True,
        },
        "observed": {
            "passRate": pass_rate,
            "retryImprovementRate": round(retry_improvement_rate, 6),
            "rerankChangedRankCount": int(rerank_changed_count),
            "rerankShadowRegressedCount": int(shadow_summary.get("regressedCount") or 0),
            "graphCandidateCount": int(graph_candidate_count),
        },
        "blockers": blockers,
    }


def build_rag_vnext_observation_report(
    searcher: Any,
    *,
    queries_path: str | Path = DEFAULT_CORRECTIVE_EVAL_PATH,
    top_k: int = 5,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.7,
    limit: int | None = None,
    retry_limit: int | None = None,
    rerank_limit: int | None = None,
    graph_limit: int | None = None,
    observation_count: int = 1,
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
    rows = [dict(row) for row in list(corrective_eval.get("rows") or [])]
    retry_rows = [row for row in rows if _observed_retry_candidate(row)]
    if retry_limit is not None:
        retry_rows = retry_rows[: max(0, int(retry_limit))]
    rerank_rows = rows[: max(0, int(rerank_limit))] if rerank_limit is not None else rows
    graph_rows = rows[: max(0, int(graph_limit))] if graph_limit is not None else rows

    corrective_runs = [
        build_rag_corrective_run(
            searcher,
            query=_row_query(row),
            top_k=top_k,
            source_type=_source_type(_row_source(row)),
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            execute=False,
            eval_context=row,
        )
        for row in retry_rows
    ]
    rerank_reports = [
        build_rag_answerability_rerank(
            searcher,
            query=_row_query(row),
            top_k=top_k,
            source_type=_source_type(_row_source(row)),
            retrieval_mode=retrieval_mode,
            alpha=alpha,
        )
        for row in rerank_rows
    ]
    rerank_shadow_eval = build_rag_answerability_rerank_eval_report(
        searcher,
        queries_path=DEFAULT_ANSWERABILITY_RERANK_EVAL_PATH,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        alpha=alpha,
        limit=rerank_limit,
    )
    graph_reports = [
        build_rag_graph_global_plan(
            searcher,
            query=_row_query(row),
            top_k=top_k,
            source_type=_source_type(_row_source(row)),
            retrieval_mode=retrieval_mode,
            alpha=alpha,
        )
        for row in graph_rows
    ]
    retry_applied_count = sum(1 for item in corrective_runs if list(item.get("actionsApplied") or []))
    retry_improved_count = sum(1 for item in corrective_runs if _retry_improved(item))
    rerank_changed_count = sum(int(item.get("changedRankCount") or 0) for item in rerank_reports)
    graph_candidate_count = sum(1 for item in graph_reports if bool(((item.get("graphGlobalLane") or {}).get("candidate"))))
    warnings = list(corrective_eval.get("warnings") or [])
    warnings.extend(str(item) for item in list(rerank_shadow_eval.get("warnings") or []))
    for report in corrective_runs + rerank_reports + graph_reports:
        warnings.extend(str(item) for item in list(report.get("warnings") or []))

    row_count = int(corrective_eval.get("rowCount") or 0)
    status = "warn" if int(corrective_eval.get("failCount") or 0) else "ok"
    payload = {
        "schema": RAG_VNEXT_OBSERVATION_SCHEMA,
        "status": status,
        "generatedAt": str(generated_at or _now_iso()),
        "readOnly": True,
        "writeFree": True,
        "queriesPath": str(queries_path),
        "request": {
            "topK": int(top_k),
            "retrievalMode": str(retrieval_mode or "hybrid"),
            "alpha": float(alpha),
            "limit": limit,
            "retryLimit": retry_limit,
            "rerankLimit": rerank_limit,
            "graphLimit": graph_limit,
        },
        "summary": {
            "rowCount": row_count,
            "correctivePassRate": (corrective_eval.get("metrics") or {}).get("passRate"),
            "correctiveFailCount": int(corrective_eval.get("failCount") or 0),
            "retryCandidateCount": len(retry_rows),
            "retryAppliedCount": int(retry_applied_count),
            "retryImprovedCount": int(retry_improved_count),
            "retryImprovementRate": round(retry_improved_count / max(1, retry_applied_count), 6),
            "rerankReportCount": len(rerank_reports),
            "rerankChangedRankCount": int(rerank_changed_count),
            "answerabilityRerankShadowEval": dict(rerank_shadow_eval.get("summary") or {}),
            "graphReportCount": len(graph_reports),
            "graphCandidateCount": int(graph_candidate_count),
            "graphCandidateRate": round(graph_candidate_count / max(1, len(graph_reports)), 6),
        },
        "correctiveEval": corrective_eval,
        "correctiveRuns": corrective_runs,
        "answerabilityReranks": rerank_reports,
        "answerabilityRerankShadowEval": rerank_shadow_eval,
        "graphGlobalPlans": graph_reports,
        "promotionReadiness": _promotion_readiness(
            observation_count=observation_count,
            corrective_eval=corrective_eval,
            retry_runs=corrective_runs,
            rerank_reports=rerank_reports,
            rerank_shadow_eval=rerank_shadow_eval,
            graph_reports=graph_reports,
        ),
        "warnings": list(dict.fromkeys(warnings)),
    }
    return payload


__all__ = [
    "DEFAULT_CORRECTIVE_EVAL_PATH",
    "RAG_VNEXT_OBSERVATION_SCHEMA",
    "build_rag_vnext_observation_report",
]
