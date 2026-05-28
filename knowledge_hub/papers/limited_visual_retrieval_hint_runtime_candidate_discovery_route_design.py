"""Runtime candidate-discovery route design for visual retrieval hints.

This final report-only phase consumes the production-vector search-quality eval
and the production-vector apply-gate preview, then projects the 125 visual
retrieval hints into runtime route-binding plans. It does not mutate runtime
search, does not query an operational index, and does not expose hints as answer
evidence.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
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
    PRODUCTION_VECTOR_INDEX_RECORD_SCHEMA_ID,
    READY_DECISION as APPLY_READY_DECISION,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_search_quality_eval import (
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_SEARCH_QUALITY_EVAL_SCHEMA_ID,
    READY_DECISION as SEARCH_QUALITY_READY_DECISION,
)


LIMITED_VISUAL_RETRIEVAL_HINT_RUNTIME_CANDIDATE_DISCOVERY_ROUTE_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-runtime-candidate-discovery-route-design.v1"
)
RUNTIME_ROUTE_BINDING_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-runtime-candidate-discovery-route-binding-row.v1"
)

READY_DECISION = "ready_for_visual_retrieval_hint_final_merge_review"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE_READY = "final_merge_review_and_branch_cleanup"
NEXT_TRANCHE_HOLD = "visual_retrieval_hint_runtime_candidate_discovery_route_design_review"

EXPECTED_HINT_ROWS = 125
TARGET_RUNTIME_BOUNDARY = "knowledge_hub.ai.rag_search_runtime.RAGSearchRuntime.search_with_diagnostics"
TARGET_PIPELINE_BOUNDARY = "knowledge_hub.ai.retrieval_pipeline.RetrievalPipelineService.execute"
TARGET_COLLECTION_NAME = "knowledge_hub_visual_retrieval_hints"
TARGET_VECTOR_STORE_REF = "config.vector_db_path/visual_retrieval_hints"
ROUTE_NAME = "visual_retrieval_hint_candidate_discovery"
ROUTE_MODE = "candidate_discovery_only"

READY_ROW_STATUS = "planned_runtime_candidate_discovery_route_binding"
BLOCKED_SOURCE_STATUS = "blocked_source_report_gate"
BLOCKED_POLICY_STATUS = "blocked_policy_violation"
BLOCKED_CONTRACT_STATUS = "blocked_route_contract_violation"

MUTATION_COUNTER_FIELDS = (
    "candidateStoreWriteRows",
    "embeddingCallRows",
    "embeddingVectorWriteRows",
    "vectorIndexWriteRows",
    "productionVectorIndexWriteRows",
    "databaseMutationRows",
    "indexMutationRows",
    "operationalSearchIndexQueryRows",
    "runtimeVisibleRows",
    "strictEvidenceRows",
    "citationGradeRows",
    "answerableWithoutTextEvidenceRows",
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


def _short_hash(value: str, *, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _counts(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("counts") or {})


def _policy_ok(record: dict[str, Any]) -> bool:
    policy = dict(record.get("policy") or {})
    metadata = dict(record.get("metadata") or {})
    return (
        normalize_text(policy.get("allowedUse")) == "retrieval_hint_only"
        and policy.get("strictEvidence") is False
        and policy.get("citationGrade") is False
        and policy.get("answerableWithoutTextEvidence") is False
        and policy.get("runtimeVisible") is False
        and policy.get("indexEligible") is False
        and policy.get("productionIndexEligible") is False
        and policy.get("candidateDiscoveryOnly") is True
        and normalize_text(metadata.get("allowedUse") or metadata.get("allowed_use")) == "retrieval_hint_only"
        and (metadata.get("strictEvidence") is False or metadata.get("strict_evidence") is False)
        and (metadata.get("runtimeVisible") is False or metadata.get("runtime_visible") is False)
    )


def _search_quality_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = _counts(report)
    gate = dict(report.get("qualityGate") or {})
    observed = dict(gate.get("observed") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "qualityGatePassed": gate.get("passed") is True,
        "layoutCandidateRows": _int(counts.get("layoutCandidateRows")),
        "sourceProductionVectorRecordRows": _int(counts.get("sourceProductionVectorRecordRows")),
        "queryRows": _int(counts.get("queryRows")),
        "textOnlyHitAt5Rows": _int(counts.get("textOnlyHitAt5Rows")),
        "productionVectorHitAt5Rows": _int(counts.get("productionVectorHitAt5Rows") or observed.get("productionVectorHitAt5Rows")),
        "hybridHitAt5Rows": _int(counts.get("hybridHitAt5Rows")),
        "hybridHitAt5LiftRows": _int(counts.get("hybridHitAt5LiftRows") or observed.get("hybridHitAt5LiftRows")),
        "rankRegressedRows": _int(counts.get("rankRegressedRows") or observed.get("rankRegressedRows")),
        "blockedRows": _int(counts.get("blockedRows")),
        "privatePathLeakRows": _int(counts.get("privatePathLeakRows")),
        "schemaViolationCount": _int(counts.get("schemaViolationCount")),
    }


def _apply_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = _counts(report)
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "plannedProductionVectorRecordRows": _int(counts.get("plannedProductionVectorRecordRows")),
        "appliedProductionVectorRecordRows": _int(counts.get("appliedProductionVectorRecordRows")),
        "readbackValidatedRows": _int(counts.get("readbackValidatedRows")),
        "candidateDiscoveryOnlyRows": _int(counts.get("candidateDiscoveryOnlyRows")),
        "productionVectorIndexWriteRows": _int(counts.get("productionVectorIndexWriteRows")),
        "databaseMutationRows": _int(counts.get("databaseMutationRows")),
        "blockedRows": _int(counts.get("blockedRows")),
        "policyViolationRows": _int(counts.get("policyViolationRows")),
        "privatePathLeakRows": _int(counts.get("privatePathLeakRows")),
        "schemaViolationCount": _int(counts.get("schemaViolationCount")),
    }


def _search_quality_blockers(report: dict[str, Any]) -> list[str]:
    counts = _counts(report)
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_SEARCH_QUALITY_EVAL_SCHEMA_ID:
        blockers.append("invalid_production_vector_search_quality_eval_schema")
    if report.get("status") != "ready":
        blockers.append("production_vector_search_quality_eval_not_ready")
    if report.get("decision") != SEARCH_QUALITY_READY_DECISION:
        blockers.append("production_vector_search_quality_eval_invalid_decision")
    if dict(report.get("qualityGate") or {}).get("passed") is not True:
        blockers.append("production_vector_search_quality_gate_not_passed")
    if _int(counts.get("sourceProductionVectorRecordRows")) != EXPECTED_HINT_ROWS:
        blockers.append("source_production_vector_record_rows_not_125")
    if _int(counts.get("queryRows")) != EXPECTED_HINT_ROWS * 2:
        blockers.append("search_quality_query_rows_not_250")
    if _int(counts.get("rankRegressedRows")) != 0:
        blockers.append("search_quality_rank_regressions_present")
    for field in (
        "blockedRows",
        "policyViolationRows",
        "privatePathLeakRows",
        "schemaViolationCount",
        *MUTATION_COUNTER_FIELDS,
    ):
        if _int(counts.get(field)) != 0:
            blockers.append(f"production_vector_search_quality_eval_has_{field}")
    if _contains_private_path(report):
        blockers.append("production_vector_search_quality_eval_has_private_path_leak")
    return blockers


def _apply_blockers(report: dict[str, Any]) -> list[str]:
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
    if len(records) != EXPECTED_HINT_ROWS:
        blockers.append("production_vector_record_previews_not_125")
    if _int(counts.get("candidateDiscoveryOnlyRows")) != EXPECTED_HINT_ROWS:
        blockers.append("production_vector_candidate_discovery_only_rows_not_125")
    for field in (
        "blockedRows",
        "policyViolationRows",
        "privatePathLeakRows",
        "schemaViolationCount",
        "externalEmbeddingCallRows",
        "embeddingCallRows",
        "runtimeVisibleRows",
        "strictEvidenceRows",
        "citationGradeRows",
        "answerableWithoutTextEvidenceRows",
    ):
        if _int(counts.get(field)) != 0:
            blockers.append(f"production_vector_apply_executor_has_{field}")
    if _contains_private_path(report):
        blockers.append("production_vector_apply_executor_has_private_path_leak")
    return blockers


def _record_blockers(record: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if record.get("schema") != PRODUCTION_VECTOR_INDEX_RECORD_SCHEMA_ID:
        blockers.append("invalid_production_vector_index_record_schema")
    for field in (
        "vectorDocumentId",
        "hintCandidateId",
        "sourceCandidateId",
        "sourceContentHash",
        "paperId",
        "page",
        "bbox",
        "candidateType",
        "documentText",
        "embeddingText",
        "metadata",
        "policy",
        "productionVectorRecordSha256",
    ):
        if not record.get(field):
            blockers.append(f"missing_{field}")
    if normalize_text(record.get("collectionName")) != TARGET_COLLECTION_NAME:
        blockers.append("unexpected_collection_name")
    if not _policy_ok(record):
        blockers.append("policy_not_retrieval_hint_only")
    if _contains_private_path(record):
        blockers.append("private_path_leak")
    return blockers


def _route_binding_row(record: dict[str, Any], *, source_blockers: list[str]) -> dict[str, Any]:
    record_blockers = _record_blockers(record)
    if source_blockers:
        status = BLOCKED_SOURCE_STATUS
    elif any("policy" in blocker for blocker in record_blockers):
        status = BLOCKED_POLICY_STATUS
    elif record_blockers:
        status = BLOCKED_CONTRACT_STATUS
    else:
        status = READY_ROW_STATUS
    basis = "|".join(
        [
            ROUTE_NAME,
            normalize_text(record.get("vectorDocumentId")),
            normalize_text(record.get("hintCandidateId")),
            normalize_text(record.get("sourceCandidateId")),
        ]
    )
    row = {
        "schema": RUNTIME_ROUTE_BINDING_ROW_SCHEMA_ID,
        "rowId": "limited-visual-retrieval-hint-runtime-candidate-discovery-route-binding:" + _short_hash(basis),
        "status": status,
        "routeName": ROUTE_NAME,
        "routeMode": ROUTE_MODE,
        "targetRuntimeBoundary": TARGET_RUNTIME_BOUNDARY,
        "targetPipelineBoundary": TARGET_PIPELINE_BOUNDARY,
        "targetVectorStoreRef": TARGET_VECTOR_STORE_REF,
        "targetCollectionName": normalize_text(record.get("collectionName")),
        "vectorDocumentId": normalize_text(record.get("vectorDocumentId")),
        "hintCandidateId": normalize_text(record.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(record.get("sourceCandidateId")),
        "sourceContentHash": normalize_text(record.get("sourceContentHash")),
        "paperId": normalize_text(record.get("paperId")),
        "page": _int(record.get("page")),
        "bbox": list(record.get("bbox") or []),
        "candidateType": normalize_text(record.get("candidateType")),
        "candidateDiscoveryPolicy": {
            "candidateMayExpandSearchCandidates": True,
            "candidateMaySupplyAnswerEvidence": False,
            "strictTextEvidenceRequiredAfterDiscovery": True,
            "citationGradePromotionAllowed": False,
            "answerVisibleByDefault": False,
            "runtimeVisibleByDefault": False,
        },
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
            "productionIndexEligible": False,
            "candidateDiscoveryOnly": True,
        },
        "executionPlan": {
            "designOnly": True,
            "futureRuntimeApplyGateRequired": True,
            "actualCandidateStoreWrite": False,
            "actualEmbeddingCall": False,
            "actualVectorIndexWrite": False,
            "actualProductionVectorIndexWrite": False,
            "actualDatabaseMutation": False,
            "actualOperationalSearchQuery": False,
            "actualRuntimeExposure": False,
            "actualAnswerGeneration": False,
            "actualEvidencePromotion": False,
            "actualGraphDbWrite": False,
            "actualOntologyWrite": False,
            "actualMemoryCardWrite": False,
            "actualClusterWrite": False,
        },
        "sourceRefs": {
            "productionVectorRecordSha256": normalize_text(record.get("productionVectorRecordSha256")),
            "sourcePreviewRecordSha256": normalize_text(record.get("sourcePreviewRecordSha256")),
        },
        "blockers": [*source_blockers, *record_blockers],
    }
    row["routeBindingSha256"] = _sha256_json(row)
    return row


def _count_rows(rows: list[dict[str, Any]], search_quality_report: dict[str, Any], apply_report: dict[str, Any]) -> dict[str, Any]:
    ready_rows = [row for row in rows if row.get("status") == READY_ROW_STATUS]
    blocked_rows = [row for row in rows if row.get("status") != READY_ROW_STATUS]
    search_counts = _counts(search_quality_report)
    apply_counts = _counts(apply_report)
    return {
        "sourceProductionVectorRecordRows": _int(search_counts.get("sourceProductionVectorRecordRows")),
        "sourceApplyPreviewRows": len(list(apply_report.get("productionVectorIndexRecordPreviews") or [])),
        "sourceSearchQualityQueryRows": _int(search_counts.get("queryRows")),
        "plannedRouteBindingRows": len(ready_rows),
        "candidateDiscoveryOnlyRows": len(ready_rows),
        "routeBindingRows": len(rows),
        "blockedRows": len(blocked_rows),
        "policyViolationRows": sum(
            1 for row in rows if row.get("status") == BLOCKED_POLICY_STATUS or "policy" in " ".join(row.get("blockers") or [])
        ),
        "contractViolationRows": sum(1 for row in rows if row.get("status") == BLOCKED_CONTRACT_STATUS),
        "privatePathLeakRows": sum(1 for row in rows if "private_path_leak" in list(row.get("blockers") or [])),
        "qualityEvalProductionVectorHitAt5Rows": _int(search_counts.get("productionVectorHitAt5Rows")),
        "qualityEvalHybridHitAt5Rows": _int(search_counts.get("hybridHitAt5Rows")),
        "qualityEvalHybridHitAt5LiftRows": _int(search_counts.get("hybridHitAt5LiftRows")),
        "qualityEvalRankRegressedRows": _int(search_counts.get("rankRegressedRows")),
        "sourceApplyPlannedRows": _int(apply_counts.get("plannedProductionVectorRecordRows")),
        "sourceApplyAppliedRows": _int(apply_counts.get("appliedProductionVectorRecordRows")),
        "candidateStoreWriteRows": 0,
        "embeddingCallRows": 0,
        "embeddingVectorWriteRows": 0,
        "vectorIndexWriteRows": 0,
        "productionVectorIndexWriteRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "operationalSearchIndexQueryRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "answerGenerationRows": 0,
        "runtimeRouteWriteRows": 0,
        "runtimeConfigMutationRows": 0,
        "graphDbWriteRows": 0,
        "ontologyWriteRows": 0,
        "memoryCardWriteRows": 0,
        "clusterWriteRows": 0,
        "schemaViolationCount": 0,
        "byCandidateType": dict(sorted(Counter(normalize_text(row.get("candidateType")) for row in rows).items())),
    }


def _gate(counts: dict[str, Any], source_blockers: list[str]) -> dict[str, Any]:
    mutation_fields = (
        "candidateStoreWriteRows",
        "embeddingCallRows",
        "embeddingVectorWriteRows",
        "vectorIndexWriteRows",
        "productionVectorIndexWriteRows",
        "databaseMutationRows",
        "indexMutationRows",
        "operationalSearchIndexQueryRows",
        "runtimeVisibleRows",
        "strictEvidenceRows",
        "citationGradeRows",
        "answerableWithoutTextEvidenceRows",
        "answerGenerationRows",
        "runtimeRouteWriteRows",
        "runtimeConfigMutationRows",
        "graphDbWriteRows",
        "ontologyWriteRows",
        "memoryCardWriteRows",
        "clusterWriteRows",
    )
    checks = {
        "sourceReportsReady": not source_blockers,
        "sourceProductionVectorRowsExactly125": _int(counts.get("sourceProductionVectorRecordRows")) == EXPECTED_HINT_ROWS,
        "sourceApplyPreviewsExactly125": _int(counts.get("sourceApplyPreviewRows")) == EXPECTED_HINT_ROWS,
        "plannedRouteBindingsExactly125": _int(counts.get("plannedRouteBindingRows")) == EXPECTED_HINT_ROWS,
        "candidateDiscoveryOnlyRowsExactly125": _int(counts.get("candidateDiscoveryOnlyRows")) == EXPECTED_HINT_ROWS,
        "searchQualityQueriesExactly250": _int(counts.get("sourceSearchQualityQueryRows")) == EXPECTED_HINT_ROWS * 2,
        "searchQualityNoRankRegression": _int(counts.get("qualityEvalRankRegressedRows")) == 0,
        "noBlockedRows": _int(counts.get("blockedRows")) == 0,
        "noPolicyViolations": _int(counts.get("policyViolationRows")) == 0,
        "noPrivatePathLeaks": _int(counts.get("privatePathLeakRows")) == 0,
        "noMutationOrRuntimeExposure": all(_int(counts.get(field)) == 0 for field in mutation_fields),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expectedRows": EXPECTED_HINT_ROWS,
        "observed": {
            "plannedRouteBindingRows": _int(counts.get("plannedRouteBindingRows")),
            "candidateDiscoveryOnlyRows": _int(counts.get("candidateDiscoveryOnlyRows")),
            "blockedRows": _int(counts.get("blockedRows")),
            "qualityEvalHybridHitAt5Rows": _int(counts.get("qualityEvalHybridHitAt5Rows")),
            "qualityEvalRankRegressedRows": _int(counts.get("qualityEvalRankRegressedRows")),
        },
    }


def build_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design(
    *,
    production_vector_search_quality_eval: dict[str, Any],
    production_vector_apply_executor_report: dict[str, Any],
    source_production_vector_search_quality_eval_ref: str,
    source_production_vector_apply_executor_report_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_blockers = [
        *_search_quality_blockers(production_vector_search_quality_eval),
        *_apply_blockers(production_vector_apply_executor_report),
    ]
    records = [dict(row) for row in production_vector_apply_executor_report.get("productionVectorIndexRecordPreviews") or []]
    rows = [_route_binding_row(record, source_blockers=source_blockers) for record in records]
    counts = _count_rows(rows, production_vector_search_quality_eval, production_vector_apply_executor_report)
    gate = _gate(counts, source_blockers)
    technical_blockers = sorted(set(source_blockers + [blocker for row in rows for blocker in row.get("blockers", [])]))
    status = "ready" if gate.get("passed") and not technical_blockers else "blocked"
    decision = READY_DECISION if status == "ready" else BLOCKED_DECISION
    next_tranche = NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_HOLD
    planned_rows = [row for row in rows if row.get("status") == READY_ROW_STATUS]
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_RUNTIME_CANDIDATE_DISCOVERY_ROUTE_DESIGN_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": decision,
        "nextRecommendedTranche": next_tranche,
        "sourceProductionVectorSearchQualityEval": _search_quality_summary(
            production_vector_search_quality_eval,
            report_ref=source_production_vector_search_quality_eval_ref,
        ),
        "sourceProductionVectorApplyExecutorReport": _apply_summary(
            production_vector_apply_executor_report,
            report_ref=source_production_vector_apply_executor_report_ref,
        ),
        "input": {
            "sourceProductionVectorSearchQualityEvalRef": normalize_text(source_production_vector_search_quality_eval_ref),
            "sourceProductionVectorApplyExecutorReportRef": normalize_text(
                source_production_vector_apply_executor_report_ref
            ),
            "expectedHintRows": EXPECTED_HINT_ROWS,
            "routeName": ROUTE_NAME,
            "routeMode": ROUTE_MODE,
            "targetRuntimeBoundary": TARGET_RUNTIME_BOUNDARY,
            "targetPipelineBoundary": TARGET_PIPELINE_BOUNDARY,
            "targetVectorStoreRef": TARGET_VECTOR_STORE_REF,
            "targetCollectionName": TARGET_COLLECTION_NAME,
        },
        "policy": {
            "designOnly": True,
            "candidateDiscoveryOnly": True,
            "futureRuntimeApplyGateRequired": True,
            "candidateStoreWrite": False,
            "embeddingCalls": False,
            "vectorIndexWrite": False,
            "productionVectorIndexWrite": False,
            "operationalSearchIndexQuery": False,
            "runtimeRouteWrite": False,
            "runtimeConfigMutation": False,
            "runtimeVisible": False,
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "answerGeneration": False,
            "graphDbWrite": False,
            "ontologyWrite": False,
            "memoryCardWrite": False,
            "clusterWrite": False,
        },
        "method": {
            "name": "visual_retrieval_hint_runtime_candidate_discovery_route_design_v1",
            "description": (
                "Plans a future runtime candidate-discovery route that may use visual retrieval hints "
                "to expand candidates while preserving strict text-evidence answer gates."
            ),
            "candidateDiscoveryRoute": ROUTE_NAME,
            "runtimeBoundaryNotChanged": TARGET_RUNTIME_BOUNDARY,
            "answerEvidenceGate": "strict_text_evidence_required_after_candidate_discovery",
            "limitations": [
                "No runtime route is changed in this phase.",
                "No operational search index is queried.",
                "No visual retrieval hint becomes answer-visible evidence.",
                "No graph DB, ontology, memory card, or clustering write is planned here.",
            ],
        },
        "counts": counts,
        "gate": gate,
        "routeBindingRowsDetail": rows,
        "plannedRuntimeRouteBindings": planned_rows,
        "sourceBlockers": sorted(set(source_blockers)),
        "technicalBlockers": technical_blockers,
        "warnings": [],
    }


def render_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Limited Visual Retrieval Hint Runtime Candidate Discovery Route Design 005",
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
        f"- productionVectorIndexWriteRows: `{counts.get('productionVectorIndexWriteRows')}`",
        f"- operationalSearchIndexQueryRows: `{counts.get('operationalSearchIndexQueryRows')}`",
        f"- runtimeRouteWriteRows: `{counts.get('runtimeRouteWriteRows')}`",
        f"- runtimeVisibleRows: `{counts.get('runtimeVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        f"- answerableWithoutTextEvidenceRows: `{counts.get('answerableWithoutTextEvidenceRows')}`",
        "",
        "## Gate",
        "",
        f"- passed: `{gate.get('passed')}`",
        f"- plannedRouteBindingsExactly125: `{dict(gate.get('checks') or {}).get('plannedRouteBindingsExactly125')}`",
        f"- noMutationOrRuntimeExposure: `{dict(gate.get('checks') or {}).get('noMutationOrRuntimeExposure')}`",
        "",
        "## Runtime Placement",
        "",
        f"- routeName: `{dict(report.get('input') or {}).get('routeName')}`",
        f"- routeMode: `{dict(report.get('input') or {}).get('routeMode')}`",
        f"- targetRuntimeBoundary: `{dict(report.get('input') or {}).get('targetRuntimeBoundary')}`",
        f"- targetCollectionName: `{dict(report.get('input') or {}).get('targetCollectionName')}`",
        "",
        "## Non-Scope",
        "",
        "- No runtime route write.",
        "- No operational search index query.",
        "- No answer-visible exposure.",
        "- No evidence promotion.",
        "- No graph DB, ontology, memory card, or clustering write.",
    ]
    blockers = list(report.get("technicalBlockers") or [])
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def write_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design_markdown(report),
        encoding="utf-8",
    )
    return {"json": str(report_json), "md": str(report_md)}


__all__ = [
    "LIMITED_VISUAL_RETRIEVAL_HINT_RUNTIME_CANDIDATE_DISCOVERY_ROUTE_DESIGN_SCHEMA_ID",
    "READY_DECISION",
    "READY_ROW_STATUS",
    "build_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design",
    "load_json",
    "render_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design_markdown",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design",
]
