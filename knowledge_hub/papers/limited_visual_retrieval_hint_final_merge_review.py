"""Final merge review for the visual retrieval-hint candidate-discovery tranche.

This review closes the report-only visual retrieval-hint sequence by checking
that the runtime route design, production-vector quality eval, and apply-gate
preview all agree on the same 125 candidate-discovery-only rows. It does not
write routes, query operational indexes, or promote visual hints as evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    _contains_private_path,
    normalize_text,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run import (
    load_json,
    sanitized_report_ref,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_integration_apply_executor import (
    APPLIED_DECISION as APPLY_APPLIED_DECISION,
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_APPLY_EXECUTOR_SCHEMA_ID,
    READY_DECISION as APPLY_READY_DECISION,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_search_quality_eval import (
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_SEARCH_QUALITY_EVAL_SCHEMA_ID,
    READY_DECISION as SEARCH_READY_DECISION,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_runtime_candidate_discovery_route_design import (
    LIMITED_VISUAL_RETRIEVAL_HINT_RUNTIME_CANDIDATE_DISCOVERY_ROUTE_DESIGN_SCHEMA_ID,
    READY_DECISION as ROUTE_READY_DECISION,
    READY_ROW_STATUS,
)


LIMITED_VISUAL_RETRIEVAL_HINT_FINAL_MERGE_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-final-merge-review.v1"
)

READY_DECISION = "visual_retrieval_hint_candidate_discovery_tranche_complete"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE_READY = "paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run"
NEXT_TRANCHE_HOLD = "visual_retrieval_hint_final_merge_review_blocked"

EXPECTED_HINT_ROWS = 125
EXPECTED_QUERY_ROWS = 250
EXPECTED_PRODUCTION_VECTOR_HIT_AT_5_ROWS = 234
EXPECTED_HYBRID_HIT_AT_5_ROWS = 241
EXPECTED_HYBRID_HIT_AT_5_LIFT_ROWS = 108
BASE_REF = "refs/remotes/origin/main"
STALE_WORKTREE_DO_NOT_REUSE_REF = "KnowledgeOS/.worktrees/knowledge-hub-next-implementation-20260522"

UNSAFE_COUNTER_FIELDS = (
    "candidateStoreWriteRows",
    "embeddingCallRows",
    "embeddingVectorWriteRows",
    "vectorIndexWriteRows",
    "productionVectorIndexWriteRows",
    "databaseMutationRows",
    "indexMutationRows",
    "operationalSearchIndexQueryRows",
    "runtimeRouteWriteRows",
    "runtimeConfigMutationRows",
    "runtimeVisibleRows",
    "strictEvidenceRows",
    "citationGradeRows",
    "answerableWithoutTextEvidenceRows",
    "answerGenerationRows",
    "graphDbWriteRows",
    "ontologyWriteRows",
    "memoryCardWriteRows",
    "clusterWriteRows",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _counts(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("counts") or {})


def _summary(report: dict[str, Any], *, report_ref: str, count_fields: tuple[str, ...]) -> dict[str, Any]:
    counts = _counts(report)
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "nextRecommendedTranche": normalize_text(report.get("nextRecommendedTranche")),
        "reportRef": normalize_text(report_ref),
        "counts": {field: _int(counts.get(field)) for field in count_fields},
    }


def _unsafe_counter_blockers(prefix: str, counts: dict[str, Any]) -> list[str]:
    return [f"{prefix}_has_{field}" for field in UNSAFE_COUNTER_FIELDS if _int(counts.get(field)) != 0]


def _route_design_blockers(report: dict[str, Any]) -> list[str]:
    counts = _counts(report)
    rows = list(report.get("routeBindingRowsDetail") or [])
    planned = list(report.get("plannedRuntimeRouteBindings") or [])
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_RUNTIME_CANDIDATE_DISCOVERY_ROUTE_DESIGN_SCHEMA_ID:
        blockers.append("invalid_runtime_candidate_discovery_route_design_schema")
    if report.get("status") != "ready":
        blockers.append("runtime_candidate_discovery_route_design_not_ready")
    if report.get("decision") != ROUTE_READY_DECISION:
        blockers.append("runtime_candidate_discovery_route_design_invalid_decision")
    if _int(counts.get("plannedRouteBindingRows")) != EXPECTED_HINT_ROWS:
        blockers.append("planned_route_binding_rows_not_125")
    if _int(counts.get("candidateDiscoveryOnlyRows")) != EXPECTED_HINT_ROWS:
        blockers.append("candidate_discovery_only_rows_not_125")
    if _int(counts.get("routeBindingRows")) != EXPECTED_HINT_ROWS:
        blockers.append("route_binding_rows_not_125")
    if len(rows) != EXPECTED_HINT_ROWS:
        blockers.append("route_binding_rows_detail_not_125")
    if len(planned) != EXPECTED_HINT_ROWS:
        blockers.append("planned_runtime_route_bindings_not_125")
    if any(row.get("status") != READY_ROW_STATUS for row in rows):
        blockers.append("route_binding_row_not_ready")
    if _int(counts.get("qualityEvalProductionVectorHitAt5Rows")) != EXPECTED_PRODUCTION_VECTOR_HIT_AT_5_ROWS:
        blockers.append("quality_eval_production_vector_hit_at_5_not_234")
    if _int(counts.get("qualityEvalHybridHitAt5Rows")) != EXPECTED_HYBRID_HIT_AT_5_ROWS:
        blockers.append("quality_eval_hybrid_hit_at_5_not_241")
    if _int(counts.get("qualityEvalHybridHitAt5LiftRows")) != EXPECTED_HYBRID_HIT_AT_5_LIFT_ROWS:
        blockers.append("quality_eval_hybrid_hit_at_5_lift_not_108")
    if _int(counts.get("qualityEvalRankRegressedRows")) != 0:
        blockers.append("quality_eval_rank_regressions_present")
    for field in ("blockedRows", "policyViolationRows", "privatePathLeakRows", "schemaViolationCount"):
        if _int(counts.get(field)) != 0:
            blockers.append(f"runtime_candidate_discovery_route_design_has_{field}")
    blockers.extend(_unsafe_counter_blockers("runtime_candidate_discovery_route_design", counts))
    if _contains_private_path(report):
        blockers.append("runtime_candidate_discovery_route_design_has_private_path_leak")
    return blockers


def _search_quality_blockers(report: dict[str, Any]) -> list[str]:
    counts = _counts(report)
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_SEARCH_QUALITY_EVAL_SCHEMA_ID:
        blockers.append("invalid_production_vector_search_quality_eval_schema")
    if report.get("status") != "ready":
        blockers.append("production_vector_search_quality_eval_not_ready")
    if report.get("decision") != SEARCH_READY_DECISION:
        blockers.append("production_vector_search_quality_eval_invalid_decision")
    if dict(report.get("qualityGate") or {}).get("passed") is not True:
        blockers.append("production_vector_search_quality_gate_not_passed")
    if _int(counts.get("sourceProductionVectorRecordRows")) != EXPECTED_HINT_ROWS:
        blockers.append("source_production_vector_record_rows_not_125")
    if _int(counts.get("queryRows")) != EXPECTED_QUERY_ROWS:
        blockers.append("search_quality_query_rows_not_250")
    if _int(counts.get("productionVectorHitAt5Rows")) != EXPECTED_PRODUCTION_VECTOR_HIT_AT_5_ROWS:
        blockers.append("production_vector_hit_at_5_rows_not_234")
    if _int(counts.get("hybridHitAt5Rows")) != EXPECTED_HYBRID_HIT_AT_5_ROWS:
        blockers.append("hybrid_hit_at_5_rows_not_241")
    if _int(counts.get("hybridHitAt5LiftRows")) != EXPECTED_HYBRID_HIT_AT_5_LIFT_ROWS:
        blockers.append("hybrid_hit_at_5_lift_rows_not_108")
    if _int(counts.get("rankRegressedRows")) != 0:
        blockers.append("search_quality_rank_regressions_present")
    for field in ("blockedRows", "policyViolationRows", "privatePathLeakRows", "schemaViolationCount"):
        if _int(counts.get(field)) != 0:
            blockers.append(f"production_vector_search_quality_eval_has_{field}")
    blockers.extend(_unsafe_counter_blockers("production_vector_search_quality_eval", counts))
    if _contains_private_path(report):
        blockers.append("production_vector_search_quality_eval_has_private_path_leak")
    return blockers


def _apply_executor_blockers(report: dict[str, Any]) -> list[str]:
    counts = _counts(report)
    records = list(report.get("productionVectorIndexRecordPreviews") or [])
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_APPLY_EXECUTOR_SCHEMA_ID:
        blockers.append("invalid_production_vector_apply_executor_schema")
    if report.get("status") not in {"ready", "applied"}:
        blockers.append("production_vector_apply_executor_not_ready_or_applied")
    if report.get("decision") not in {APPLY_READY_DECISION, APPLY_APPLIED_DECISION}:
        blockers.append("production_vector_apply_executor_invalid_decision")
    ready_or_applied = _int(counts.get("plannedProductionVectorRecordRows")) + _int(
        counts.get("appliedProductionVectorRecordRows")
    )
    if ready_or_applied != EXPECTED_HINT_ROWS:
        blockers.append("production_vector_ready_or_applied_rows_not_125")
    if _int(counts.get("candidateDiscoveryOnlyRows")) != EXPECTED_HINT_ROWS:
        blockers.append("production_vector_candidate_discovery_only_rows_not_125")
    if len(records) != EXPECTED_HINT_ROWS:
        blockers.append("production_vector_record_previews_not_125")
    for field in ("blockedRows", "policyViolationRows", "privatePathLeakRows", "schemaViolationCount"):
        if _int(counts.get(field)) != 0:
            blockers.append(f"production_vector_apply_executor_has_{field}")
    blockers.extend(_unsafe_counter_blockers("production_vector_apply_executor", counts))
    if _contains_private_path(report):
        blockers.append("production_vector_apply_executor_has_private_path_leak")
    return blockers


def _cross_check_blockers(
    *,
    route_design: dict[str, Any],
    search_quality_eval: dict[str, Any],
    apply_executor_report: dict[str, Any],
) -> list[str]:
    route_counts = _counts(route_design)
    search_counts = _counts(search_quality_eval)
    apply_counts = _counts(apply_executor_report)
    blockers: list[str] = []
    if _int(route_counts.get("qualityEvalProductionVectorHitAt5Rows")) != _int(
        search_counts.get("productionVectorHitAt5Rows")
    ):
        blockers.append("route_design_search_quality_production_hit_at_5_mismatch")
    if _int(route_counts.get("qualityEvalHybridHitAt5Rows")) != _int(search_counts.get("hybridHitAt5Rows")):
        blockers.append("route_design_search_quality_hybrid_hit_at_5_mismatch")
    if _int(route_counts.get("qualityEvalHybridHitAt5LiftRows")) != _int(
        search_counts.get("hybridHitAt5LiftRows")
    ):
        blockers.append("route_design_search_quality_hybrid_lift_mismatch")
    if _int(route_counts.get("qualityEvalRankRegressedRows")) != _int(search_counts.get("rankRegressedRows")):
        blockers.append("route_design_search_quality_rank_regression_mismatch")
    if _int(route_counts.get("sourceApplyPreviewRows")) != len(
        list(apply_executor_report.get("productionVectorIndexRecordPreviews") or [])
    ):
        blockers.append("route_design_apply_preview_rows_mismatch")
    if _int(route_counts.get("sourceApplyPlannedRows")) != _int(apply_counts.get("plannedProductionVectorRecordRows")):
        blockers.append("route_design_apply_planned_rows_mismatch")
    return blockers


def _review_counts(
    *,
    route_design: dict[str, Any],
    search_quality_eval: dict[str, Any],
    apply_executor_report: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    route_counts = _counts(route_design)
    search_counts = _counts(search_quality_eval)
    apply_counts = _counts(apply_executor_report)
    return {
        "reviewRows": EXPECTED_HINT_ROWS,
        "sourceRouteBindingRows": _int(route_counts.get("routeBindingRows")),
        "plannedRouteBindingRows": _int(route_counts.get("plannedRouteBindingRows")),
        "candidateDiscoveryOnlyRows": _int(route_counts.get("candidateDiscoveryOnlyRows")),
        "sourceProductionVectorRecordRows": _int(search_counts.get("sourceProductionVectorRecordRows")),
        "sourceApplyPreviewRows": len(list(apply_executor_report.get("productionVectorIndexRecordPreviews") or [])),
        "qualityEvalProductionVectorHitAt5Rows": _int(search_counts.get("productionVectorHitAt5Rows")),
        "qualityEvalHybridHitAt5Rows": _int(search_counts.get("hybridHitAt5Rows")),
        "qualityEvalHybridHitAt5LiftRows": _int(search_counts.get("hybridHitAt5LiftRows")),
        "qualityEvalRankRegressedRows": _int(search_counts.get("rankRegressedRows")),
        "sourceApplyPlannedRows": _int(apply_counts.get("plannedProductionVectorRecordRows")),
        "sourceApplyAppliedRows": _int(apply_counts.get("appliedProductionVectorRecordRows")),
        "blockedRows": len(blockers),
        "privatePathLeakRows": 0 if not blockers else sum(1 for blocker in blockers if "private_path" in blocker),
        "schemaViolationCount": 0,
        "branchDeletionRows": 0,
        "githubPrMutationRows": 0,
        "runtimeRouteWriteRows": 0,
        "runtimeConfigMutationRows": 0,
        "operationalSearchIndexQueryRows": 0,
        "runtimeVisibleRows": 0,
        "answerVisibleRows": 0,
        "answerGenerationRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "candidateStoreWriteRows": 0,
        "embeddingCallRows": 0,
        "embeddingVectorWriteRows": 0,
        "vectorIndexWriteRows": 0,
        "productionVectorIndexWriteRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "graphDbWriteRows": 0,
        "ontologyWriteRows": 0,
        "memoryCardWriteRows": 0,
        "clusterWriteRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
    }


def _gate(counts: dict[str, Any], blockers: list[str]) -> dict[str, Any]:
    zero_fields = (
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
        "branchDeletionRows",
        "githubPrMutationRows",
        "runtimeRouteWriteRows",
        "runtimeConfigMutationRows",
        "operationalSearchIndexQueryRows",
        "runtimeVisibleRows",
        "answerVisibleRows",
        "answerGenerationRows",
        "strictEvidenceRows",
        "citationGradeRows",
        "answerableWithoutTextEvidenceRows",
        "candidateStoreWriteRows",
        "embeddingCallRows",
        "embeddingVectorWriteRows",
        "vectorIndexWriteRows",
        "productionVectorIndexWriteRows",
        "databaseMutationRows",
        "indexMutationRows",
        "graphDbWriteRows",
        "ontologyWriteRows",
        "memoryCardWriteRows",
        "clusterWriteRows",
        "vaultScanRows",
        "externalDownloadRows",
    )
    checks = {
        "sourceReportsReady": not blockers,
        "plannedRouteBindingRowsExactly125": _int(counts.get("plannedRouteBindingRows")) == EXPECTED_HINT_ROWS,
        "candidateDiscoveryOnlyRowsExactly125": _int(counts.get("candidateDiscoveryOnlyRows")) == EXPECTED_HINT_ROWS,
        "qualityEvalProductionVectorHitAt5RowsExact": _int(counts.get("qualityEvalProductionVectorHitAt5Rows"))
        == EXPECTED_PRODUCTION_VECTOR_HIT_AT_5_ROWS,
        "qualityEvalHybridHitAt5RowsExact": _int(counts.get("qualityEvalHybridHitAt5Rows"))
        == EXPECTED_HYBRID_HIT_AT_5_ROWS,
        "qualityEvalHybridHitAt5LiftRowsExact": _int(counts.get("qualityEvalHybridHitAt5LiftRows"))
        == EXPECTED_HYBRID_HIT_AT_5_LIFT_ROWS,
        "qualityEvalNoRankRegression": _int(counts.get("qualityEvalRankRegressedRows")) == 0,
        "noMutationOrRuntimeExposure": all(_int(counts.get(field)) == 0 for field in zero_fields),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expectedRows": EXPECTED_HINT_ROWS,
        "observed": {
            "plannedRouteBindingRows": _int(counts.get("plannedRouteBindingRows")),
            "candidateDiscoveryOnlyRows": _int(counts.get("candidateDiscoveryOnlyRows")),
            "qualityEvalHybridHitAt5Rows": _int(counts.get("qualityEvalHybridHitAt5Rows")),
            "qualityEvalRankRegressedRows": _int(counts.get("qualityEvalRankRegressedRows")),
            "blockedRows": _int(counts.get("blockedRows")),
        },
    }


def build_limited_visual_retrieval_hint_final_merge_review(
    *,
    runtime_candidate_discovery_route_design: dict[str, Any],
    production_vector_search_quality_eval: dict[str, Any],
    production_vector_apply_executor_report: dict[str, Any],
    source_runtime_candidate_discovery_route_design_ref: str,
    source_production_vector_search_quality_eval_ref: str,
    source_production_vector_apply_executor_report_ref: str,
    generated_at: str | None = None,
    base_ref: str = BASE_REF,
    stale_worktree_do_not_reuse_ref: str = STALE_WORKTREE_DO_NOT_REUSE_REF,
) -> dict[str, Any]:
    blockers = sorted(
        set(
            [
                *_route_design_blockers(runtime_candidate_discovery_route_design),
                *_search_quality_blockers(production_vector_search_quality_eval),
                *_apply_executor_blockers(production_vector_apply_executor_report),
                *_cross_check_blockers(
                    route_design=runtime_candidate_discovery_route_design,
                    search_quality_eval=production_vector_search_quality_eval,
                    apply_executor_report=production_vector_apply_executor_report,
                ),
            ]
        )
    )
    counts = _review_counts(
        route_design=runtime_candidate_discovery_route_design,
        search_quality_eval=production_vector_search_quality_eval,
        apply_executor_report=production_vector_apply_executor_report,
        blockers=blockers,
    )
    gate = _gate(counts, blockers)
    status = "ready" if gate.get("passed") and not blockers else "blocked"
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_FINAL_MERGE_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_HOLD,
        "input": {
            "sourceRuntimeCandidateDiscoveryRouteDesignRef": normalize_text(
                source_runtime_candidate_discovery_route_design_ref
            ),
            "sourceProductionVectorSearchQualityEvalRef": normalize_text(
                source_production_vector_search_quality_eval_ref
            ),
            "sourceProductionVectorApplyExecutorReportRef": normalize_text(
                source_production_vector_apply_executor_report_ref
            ),
            "expectedHintRows": EXPECTED_HINT_ROWS,
            "baseRef": normalize_text(base_ref),
            "staleWorktreeDoNotReuseRef": normalize_text(stale_worktree_do_not_reuse_ref),
        },
        "policy": {
            "reviewOnly": True,
            "branchCleanupAuditOnly": True,
            "candidateDiscoveryOnly": True,
            "runtimeRouteWrite": False,
            "runtimeConfigMutation": False,
            "operationalSearchIndexQuery": False,
            "runtimeVisible": False,
            "answerVisible": False,
            "answerGeneration": False,
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "candidateStoreWrite": False,
            "embeddingCalls": False,
            "vectorIndexWrite": False,
            "productionVectorIndexWrite": False,
            "databaseMutation": False,
            "indexMutation": False,
            "githubMutation": False,
            "branchDeletion": False,
        },
        "sourceRuntimeCandidateDiscoveryRouteDesign": _summary(
            runtime_candidate_discovery_route_design,
            report_ref=source_runtime_candidate_discovery_route_design_ref,
            count_fields=(
                "plannedRouteBindingRows",
                "candidateDiscoveryOnlyRows",
                "qualityEvalProductionVectorHitAt5Rows",
                "qualityEvalHybridHitAt5Rows",
                "qualityEvalHybridHitAt5LiftRows",
                "qualityEvalRankRegressedRows",
                "blockedRows",
                "privatePathLeakRows",
                "schemaViolationCount",
            ),
        ),
        "sourceProductionVectorSearchQualityEval": _summary(
            production_vector_search_quality_eval,
            report_ref=source_production_vector_search_quality_eval_ref,
            count_fields=(
                "sourceProductionVectorRecordRows",
                "queryRows",
                "textOnlyHitAt5Rows",
                "productionVectorHitAt5Rows",
                "hybridHitAt5Rows",
                "hybridHitAt5LiftRows",
                "rankRegressedRows",
                "blockedRows",
                "privatePathLeakRows",
                "schemaViolationCount",
            ),
        ),
        "sourceProductionVectorApplyExecutorReport": _summary(
            production_vector_apply_executor_report,
            report_ref=source_production_vector_apply_executor_report_ref,
            count_fields=(
                "plannedProductionVectorRecordRows",
                "appliedProductionVectorRecordRows",
                "candidateDiscoveryOnlyRows",
                "productionVectorIndexWriteRows",
                "blockedRows",
                "privatePathLeakRows",
                "schemaViolationCount",
            ),
        ),
        "branchCleanupAudit": {
            "baseRef": normalize_text(base_ref),
            "baseIsOriginMain": normalize_text(base_ref) == BASE_REF,
            "staleWorktreeDoNotReuseRef": normalize_text(stale_worktree_do_not_reuse_ref),
            "staleWorktreeReuseAllowed": False,
            "deleteBranches": False,
            "closePullRequests": False,
            "pushBranches": False,
            "githubMutationRows": 0,
            "branchDeletionRows": 0,
        },
        "method": {
            "name": "visual_retrieval_hint_final_merge_review_v1",
            "description": "Revalidates the final visual retrieval-hint report-only route design before branch cleanup.",
            "completionBoundary": "candidate_discovery_only",
            "nextEvidenceWork": NEXT_TRANCHE_READY,
        },
        "counts": counts,
        "gate": gate,
        "technicalBlockers": blockers,
        "warnings": [],
    }


def render_limited_visual_retrieval_hint_final_merge_review_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    branch = dict(report.get("branchCleanupAudit") or {})
    lines = [
        "# Limited Visual Retrieval Hint Final Merge Review 005",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- plannedRouteBindingRows: `{counts.get('plannedRouteBindingRows')}`",
        f"- candidateDiscoveryOnlyRows: `{counts.get('candidateDiscoveryOnlyRows')}`",
        f"- qualityEvalProductionVectorHitAt5Rows: `{counts.get('qualityEvalProductionVectorHitAt5Rows')}`",
        f"- qualityEvalHybridHitAt5Rows: `{counts.get('qualityEvalHybridHitAt5Rows')}`",
        f"- qualityEvalHybridHitAt5LiftRows: `{counts.get('qualityEvalHybridHitAt5LiftRows')}`",
        f"- qualityEvalRankRegressedRows: `{counts.get('qualityEvalRankRegressedRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        f"- runtimeRouteWriteRows: `{counts.get('runtimeRouteWriteRows')}`",
        f"- runtimeVisibleRows: `{counts.get('runtimeVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        f"- answerVisibleRows: `{counts.get('answerVisibleRows')}`",
        f"- databaseMutationRows: `{counts.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{counts.get('indexMutationRows')}`",
        "",
        "## Branch Cleanup Audit",
        "",
        f"- baseRef: `{branch.get('baseRef')}`",
        f"- staleWorktreeDoNotReuseRef: `{branch.get('staleWorktreeDoNotReuseRef')}`",
        f"- branchDeletionRows: `{counts.get('branchDeletionRows')}`",
        f"- githubPrMutationRows: `{counts.get('githubPrMutationRows')}`",
        "",
        "## Gate",
        "",
        f"- passed: `{dict(report.get('gate') or {}).get('passed')}`",
        f"- noMutationOrRuntimeExposure: `{dict(dict(report.get('gate') or {}).get('checks') or {}).get('noMutationOrRuntimeExposure')}`",
        "",
        "## Non-Scope",
        "",
        "- No runtime route write.",
        "- No operational search index query.",
        "- No vector DB apply.",
        "- No answer-visible exposure.",
        "- No strict or citation-grade evidence promotion.",
        "- No branch deletion, PR close, push, or GitHub mutation.",
    ]
    blockers = list(report.get("technicalBlockers") or [])
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def write_limited_visual_retrieval_hint_final_merge_review(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_limited_visual_retrieval_hint_final_merge_review_markdown(report), encoding="utf-8")
    return {"json": str(report_json), "md": str(report_md)}


__all__ = [
    "LIMITED_VISUAL_RETRIEVAL_HINT_FINAL_MERGE_REVIEW_SCHEMA_ID",
    "READY_DECISION",
    "build_limited_visual_retrieval_hint_final_merge_review",
    "load_json",
    "render_limited_visual_retrieval_hint_final_merge_review_markdown",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_final_merge_review",
]
