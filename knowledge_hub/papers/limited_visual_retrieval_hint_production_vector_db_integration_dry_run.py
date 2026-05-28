"""Production vector DB integration dry-run for visual retrieval hints.

This helper consumes the Phase 11 production-vector integration design and
materializes the exact production vector record preview that a later apply gate
may write. It is intentionally report-only: it never calls an embedder, never
opens Chroma/SQLite/vector state, and never exposes hints as answer evidence.
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
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_integration_design import (
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DESIGN_SCHEMA_ID,
    PLANNED_STATUS as DESIGN_PLANNED_STATUS,
    PRODUCTION_NAMESPACE,
    READY_DECISION as DESIGN_READY_DECISION,
)


LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-production-vector-db-integration-dry-run.v1"
)
PRODUCTION_VECTOR_DRY_RUN_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-production-vector-db-integration-dry-run-row.v1"
)
PRODUCTION_VECTOR_RECORD_PREVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-production-vector-record-preview.v1"
)

READY_DECISION = "ready_for_limited_visual_retrieval_hint_production_vector_db_integration_apply_gate"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE_READY = "limited_visual_retrieval_hint_production_vector_db_integration_apply_gate"
NEXT_TRANCHE_HOLD = "limited_visual_retrieval_hint_production_vector_db_integration_dry_run_review"

EXPECTED_HINT_ROWS = 125
TARGET_VECTOR_DATABASE_CLASS = "knowledge_hub.infrastructure.persistence.vector.VectorDatabase"
TARGET_WRITE_METHOD = "VectorDatabase.add_documents"
TARGET_COLLECTION_NAME = "knowledge_hub_visual_retrieval_hints"
TARGET_STORE_REF = "config.vector_db_path/visual_retrieval_hints"

DRY_RUN_READY_STATUS = "dry_run_ready_production_vector_record"
BLOCKED_SOURCE_STATUS = "blocked_source_design_gate"
BLOCKED_POLICY_STATUS = "blocked_policy_violation"
BLOCKED_CONTRACT_STATUS = "blocked_contract_violation"

SOURCE_MUTATION_COUNTER_FIELDS = (
    "productionVectorIndexWriteRows",
    "databaseMutationRows",
    "indexMutationRows",
    "reindexOrReembedRows",
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


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_hash(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _counts(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("counts") or {})


def _bbox(value: Any) -> list[Any]:
    return list(value or [])


def _flag(container: dict[str, Any], camel: str, snake: str | None = None) -> Any:
    if camel in container:
        return container.get(camel)
    if snake and snake in container:
        return container.get(snake)
    return None


def _policy_flags_ok(container: dict[str, Any]) -> bool:
    return (
        normalize_text(container.get("allowedUse") or container.get("allowed_use")) == "retrieval_hint_only"
        and _flag(container, "strictEvidence", "strict_evidence") is False
        and _flag(container, "citationGrade", "citation_grade") is False
        and _flag(container, "answerableWithoutTextEvidence", "answerable_without_text_evidence") is False
        and _flag(container, "runtimeVisible", "runtime_visible") is False
        and _flag(container, "indexEligible", "index_eligible") is False
    )


def _source_design_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = _counts(report)
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "productionVectorIntegrationCandidateRows": _int(counts.get("productionVectorIntegrationCandidateRows")),
        "plannedProductionVectorRecordRows": _int(counts.get("plannedProductionVectorRecordRows")),
        "candidateDiscoveryOnlyRows": _int(counts.get("candidateDiscoveryOnlyRows")),
        "blockedRows": _int(counts.get("blockedRows")),
        "policyViolationRows": _int(counts.get("policyViolationRows")),
        "privatePathLeakRows": _int(counts.get("privatePathLeakRows")),
        "schemaViolationCount": _int(counts.get("schemaViolationCount")),
        "qualityEvalHybridHitAt5Rows": _int(counts.get("qualityEvalHybridHitAt5Rows")),
        "qualityEvalHybridHitAt5LiftRows": _int(counts.get("qualityEvalHybridHitAt5LiftRows")),
    }


def _source_design_blockers(report: dict[str, Any]) -> list[str]:
    counts = _counts(report)
    rows = list(report.get("rows") or [])
    previews = list(report.get("productionVectorRecordPreviews") or [])
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DESIGN_SCHEMA_ID:
        blockers.append("invalid_production_vector_db_integration_design_schema")
    if report.get("status") != "ready":
        blockers.append("production_vector_db_integration_design_not_ready")
    if report.get("decision") != DESIGN_READY_DECISION:
        blockers.append("production_vector_db_integration_design_invalid_decision")
    if _int(counts.get("productionVectorIntegrationCandidateRows")) != EXPECTED_HINT_ROWS:
        blockers.append("source_design_candidate_rows_not_125")
    if _int(counts.get("plannedProductionVectorRecordRows")) != EXPECTED_HINT_ROWS:
        blockers.append("source_design_planned_rows_not_125")
    if _int(counts.get("candidateDiscoveryOnlyRows")) != EXPECTED_HINT_ROWS:
        blockers.append("source_design_candidate_discovery_only_rows_not_125")
    if len(rows) != EXPECTED_HINT_ROWS:
        blockers.append("source_design_rows_not_125")
    if len(previews) != EXPECTED_HINT_ROWS:
        blockers.append("source_design_previews_not_125")
    for field in (
        "blockedRows",
        "policyViolationRows",
        "privatePathLeakRows",
        "schemaViolationCount",
        *SOURCE_MUTATION_COUNTER_FIELDS,
    ):
        if _int(counts.get(field)) != 0:
            blockers.append(f"source_design_has_{field}")
    if _contains_private_path(report):
        blockers.append("source_design_has_private_path_leak")
    return blockers


def _row_contract_blockers(row: dict[str, Any]) -> list[str]:
    metadata = dict(row.get("plannedMetadata") or {})
    policy = dict(row.get("policy") or {})
    execution_plan = dict(row.get("executionPlan") or {})
    route_design = dict(row.get("routeDesign") or {})
    blockers: list[str] = []
    if row.get("status") != DESIGN_PLANNED_STATUS:
        blockers.append("source_design_row_not_planned")
    for field in (
        "hintCandidateId",
        "sourceCandidateId",
        "sourceContentHash",
        "paperId",
        "page",
        "bbox",
        "candidateType",
        "derivedTextForRetrieval",
        "retrievalKeywords",
        "documentTextHash",
        "embeddingTextHash",
        "plannedDocumentText",
        "plannedEmbeddingText",
        "plannedMetadata",
        "productionVectorDocumentId",
        "productionNamespace",
    ):
        if not row.get(field):
            blockers.append(f"missing_{field}")
    if normalize_text(row.get("productionNamespace")) != PRODUCTION_NAMESPACE:
        blockers.append("unexpected_production_namespace")
    if route_design and route_design.get("defaultRuntimeExposure") is not False:
        blockers.append("route_design_default_runtime_exposure_not_false")
    if not _policy_flags_ok(policy):
        blockers.append("row_policy_not_retrieval_hint_only")
    if not _policy_flags_ok(metadata):
        blockers.append("planned_metadata_policy_not_retrieval_hint_only")
    if policy.get("productionIndexEligible") is not False:
        blockers.append("row_policy_production_index_eligible_not_false")
    if policy.get("candidateDiscoveryOnly") is not True:
        blockers.append("row_policy_candidate_discovery_only_not_true")
    for field, expected in (
        ("actualCandidateStoreWrite", False),
        ("actualEmbeddingCall", False),
        ("actualEmbeddingVectorWrite", False),
        ("actualVectorIndexWrite", False),
        ("actualProductionVectorIndexWrite", False),
        ("actualDatabaseMutation", False),
        ("actualRuntimeExposure", False),
        ("actualEvidencePromotion", False),
    ):
        if execution_plan.get(field) is not expected:
            blockers.append(f"execution_plan_{field}_not_false")
    if _contains_private_path(row):
        blockers.append("private_path_leak")
    return blockers


def _preview_record(row: dict[str, Any]) -> dict[str, Any]:
    route_design = dict(row.get("routeDesign") or {})
    collection = normalize_text(route_design.get("proposedCollectionName")) or TARGET_COLLECTION_NAME
    store_ref = normalize_text(route_design.get("proposedStoreRef")) or TARGET_STORE_REF
    vector_document_id = normalize_text(row.get("productionVectorDocumentId"))
    idempotency_basis = "|".join(
        [
            PRODUCTION_NAMESPACE,
            vector_document_id,
            normalize_text(row.get("hintCandidateId")),
            normalize_text(row.get("sourceCandidateId")),
            normalize_text(row.get("embeddingTextHash")),
        ]
    )
    idempotency_key = "production-visual-retrieval-hint-vector:" + _short_hash(idempotency_basis)
    payload = {
        "schema": PRODUCTION_VECTOR_RECORD_PREVIEW_SCHEMA_ID,
        "productionNamespace": normalize_text(row.get("productionNamespace")) or PRODUCTION_NAMESPACE,
        "targetVectorDatabaseClass": TARGET_VECTOR_DATABASE_CLASS,
        "targetWriteMethod": TARGET_WRITE_METHOD,
        "targetVectorStoreRef": store_ref,
        "targetCollectionName": collection,
        "routingMode": "candidate_discovery_only",
        "vectorDocumentId": vector_document_id,
        "idempotencyKey": idempotency_key,
        "hintCandidateId": normalize_text(row.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(row.get("sourceCandidateId")),
        "sourceContentHash": normalize_text(row.get("sourceContentHash")),
        "paperId": normalize_text(row.get("paperId")),
        "paperRef": normalize_text(row.get("paperRef")),
        "page": _int(row.get("page")),
        "bbox": _bbox(row.get("bbox")),
        "candidateType": normalize_text(row.get("candidateType")),
        "derivedTextForRetrieval": normalize_text(row.get("derivedTextForRetrieval")),
        "retrievalKeywords": list(row.get("retrievalKeywords") or []),
        "documentText": normalize_text(row.get("plannedDocumentText")),
        "documentTextHash": normalize_text(row.get("documentTextHash")),
        "embeddingText": normalize_text(row.get("plannedEmbeddingText")),
        "embeddingTextHash": normalize_text(row.get("embeddingTextHash")),
        "metadata": dict(row.get("plannedMetadata") or {}),
        "sourceRefs": dict(row.get("sourceRefs") or {}),
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
            "dryRunOnly": True,
            "futureApplyGateRequired": True,
            "actualCandidateStoreWrite": False,
            "actualEmbeddingCall": False,
            "actualEmbeddingVectorWrite": False,
            "actualVectorIndexWrite": False,
            "actualProductionVectorIndexWrite": False,
            "actualDatabaseMutation": False,
            "actualIndexMutation": False,
            "actualRuntimeExposure": False,
            "actualEvidencePromotion": False,
            "actualGraphDbWrite": False,
            "actualOntologyWrite": False,
            "actualMemoryCardWrite": False,
            "actualClusterWrite": False,
        },
    }
    payload["plannedVectorRecordSha256"] = _record_hash(payload)
    return payload


def _dry_run_row(row: dict[str, Any], *, source_blockers: list[str]) -> dict[str, Any]:
    row_blockers = _row_contract_blockers(row)
    if source_blockers:
        status = BLOCKED_SOURCE_STATUS
    elif any("policy" in blocker for blocker in row_blockers):
        status = BLOCKED_POLICY_STATUS
    elif row_blockers:
        status = BLOCKED_CONTRACT_STATUS
    else:
        status = DRY_RUN_READY_STATUS
    preview = _preview_record(row)
    row_id_basis = "|".join(
        [
            normalize_text(preview.get("idempotencyKey")),
            normalize_text(preview.get("plannedVectorRecordSha256")),
        ]
    )
    return {
        "schema": PRODUCTION_VECTOR_DRY_RUN_ROW_SCHEMA_ID,
        "rowId": "limited-visual-retrieval-hint-production-vector-db-integration-dry-run:"
        + _short_hash(row_id_basis),
        "status": status,
        "productionVectorDocumentId": normalize_text(preview.get("vectorDocumentId")),
        "idempotencyKey": normalize_text(preview.get("idempotencyKey")),
        "plannedVectorRecordSha256": normalize_text(preview.get("plannedVectorRecordSha256")),
        "hintCandidateId": normalize_text(preview.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(preview.get("sourceCandidateId")),
        "sourceContentHash": normalize_text(preview.get("sourceContentHash")),
        "paperId": normalize_text(preview.get("paperId")),
        "page": _int(preview.get("page")),
        "bbox": _bbox(preview.get("bbox")),
        "candidateType": normalize_text(preview.get("candidateType")),
        "policy": dict(preview.get("policy") or {}),
        "executionPlan": dict(preview.get("executionPlan") or {}),
        "productionVectorRecordPreview": preview,
        "blockers": [*source_blockers, *row_blockers],
    }


def _scope(planned_rows: int) -> dict[str, Any]:
    return {
        "writes": "report_only_dry_run",
        "plannedProductionVectorRecordRows": int(planned_rows),
        "futureApplyCandidateRows": int(planned_rows),
        "candidateStoreWriteRows": 0,
        "embeddingCallRows": 0,
        "embeddingVectorWriteRows": 0,
        "vectorIndexWriteRows": 0,
        "productionVectorIndexWriteRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "operationalSearchIndexQueryRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidencePromotionRows": 0,
        "runtimeAnswerVisibleExposureRows": 0,
        "answerGenerationRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "modelCallRows": 0,
        "apiCallRows": 0,
        "graphDbWriteRows": 0,
        "ontologyWriteRows": 0,
        "memoryCardWriteRows": 0,
        "clusterWriteRows": 0,
    }


def _policy() -> dict[str, Any]:
    return {
        "dryRunOnly": True,
        "reportOnly": True,
        "candidateDiscoveryOnly": True,
        "futureApplyGateRequired": True,
        "candidateStoreWrite": False,
        "embeddingCalls": False,
        "embeddingVectorWrite": False,
        "vectorIndexWrite": False,
        "productionVectorIndexWrite": False,
        "sourceSpanCreated": False,
        "strictEvidenceCreated": False,
        "citationGradeEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "vaultScan": False,
        "externalDownload": False,
        "reindexOrReembed": False,
        "graphDbWrite": False,
        "ontologyWrite": False,
        "memoryCardWrite": False,
        "clusterWrite": False,
        "allowedUse": "retrieval_hint_only",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "runtimeVisible": False,
        "indexEligible": False,
        "productionIndexEligible": False,
    }


def _count_rows(rows: list[dict[str, Any]], source_design: dict[str, Any]) -> dict[str, Any]:
    ready_rows = [row for row in rows if row.get("status") == DRY_RUN_READY_STATUS]
    blocked_rows = [row for row in rows if row.get("status") != DRY_RUN_READY_STATUS]
    by_type = Counter(normalize_text(row.get("candidateType")) for row in rows)
    source_counts = _counts(source_design)
    return {
        "sourceDesignRows": len(list(source_design.get("rows") or [])),
        "sourceProductionVectorRecordPreviewRows": len(list(source_design.get("productionVectorRecordPreviews") or [])),
        "sourcePlannedProductionVectorRecordRows": _int(source_counts.get("plannedProductionVectorRecordRows")),
        "sourceCandidateDiscoveryOnlyRows": _int(source_counts.get("candidateDiscoveryOnlyRows")),
        "dryRunRows": len(rows),
        "dryRunReadyProductionVectorRecordRows": len(ready_rows),
        "plannedProductionVectorRecordRows": len(ready_rows),
        "futureApplyCandidateRows": len(ready_rows),
        "candidateDiscoveryOnlyRows": len(ready_rows),
        "blockedRows": len(blocked_rows),
        "policyViolationRows": sum(
            1
            for row in rows
            if row.get("status") == BLOCKED_POLICY_STATUS or "policy" in " ".join(row.get("blockers") or [])
        ),
        "contractViolationRows": sum(1 for row in rows if row.get("status") == BLOCKED_CONTRACT_STATUS),
        "privatePathLeakRows": sum(1 for row in rows if "private_path_leak" in list(row.get("blockers") or [])),
        "candidateStoreWriteRows": 0,
        "embeddingCallRows": 0,
        "embeddingVectorWriteRows": 0,
        "vectorIndexWriteRows": 0,
        "productionVectorIndexWriteRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "operationalSearchIndexQueryRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "graphDbWriteRows": 0,
        "ontologyWriteRows": 0,
        "memoryCardWriteRows": 0,
        "clusterWriteRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "modelCallRows": 0,
        "apiCallRows": 0,
        "schemaViolationCount": 0,
        "byCandidateType": dict(sorted(by_type.items())),
    }


def _gate(counts: dict[str, Any], source_blockers: list[str]) -> dict[str, Any]:
    checks = {
        "sourceDesignReady": not source_blockers,
        "sourceRowsExactly125": _int(counts.get("sourceDesignRows")) == EXPECTED_HINT_ROWS,
        "sourcePreviewsExactly125": _int(counts.get("sourceProductionVectorRecordPreviewRows")) == EXPECTED_HINT_ROWS,
        "plannedRowsExactly125": _int(counts.get("plannedProductionVectorRecordRows")) == EXPECTED_HINT_ROWS,
        "allRowsDryRunReady": _int(counts.get("dryRunReadyProductionVectorRecordRows")) == EXPECTED_HINT_ROWS,
        "candidateDiscoveryOnlyRowsExactly125": _int(counts.get("candidateDiscoveryOnlyRows")) == EXPECTED_HINT_ROWS,
        "noBlockedRows": _int(counts.get("blockedRows")) == 0,
        "noPolicyViolations": _int(counts.get("policyViolationRows")) == 0,
        "noPrivatePathLeaks": _int(counts.get("privatePathLeakRows")) == 0,
        "noProductionMutation": all(
            _int(counts.get(field)) == 0
            for field in (
                "candidateStoreWriteRows",
                "embeddingCallRows",
                "embeddingVectorWriteRows",
                "vectorIndexWriteRows",
                "productionVectorIndexWriteRows",
                "databaseMutationRows",
                "indexMutationRows",
                "reindexOrReembedRows",
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
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expectedRows": EXPECTED_HINT_ROWS,
        "observed": {
            "plannedProductionVectorRecordRows": _int(counts.get("plannedProductionVectorRecordRows")),
            "dryRunReadyProductionVectorRecordRows": _int(counts.get("dryRunReadyProductionVectorRecordRows")),
            "blockedRows": _int(counts.get("blockedRows")),
        },
    }


def build_limited_visual_retrieval_hint_production_vector_db_integration_dry_run(
    *,
    production_vector_db_integration_design: dict[str, Any],
    source_production_vector_db_integration_design_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_blockers = _source_design_blockers(production_vector_db_integration_design)
    source_rows = list(production_vector_db_integration_design.get("productionVectorRecordPreviews") or [])
    rows = [_dry_run_row(dict(row), source_blockers=source_blockers) for row in source_rows]
    planned_rows = [row for row in rows if row.get("status") == DRY_RUN_READY_STATUS]
    planned_records = [dict(row.get("productionVectorRecordPreview") or {}) for row in planned_rows]
    counts = _count_rows(rows, production_vector_db_integration_design)
    gate = _gate(counts, source_blockers)
    technical_blockers = sorted(set(source_blockers + [blocker for row in rows for blocker in row.get("blockers", [])]))
    status = "ready" if gate.get("passed") and not technical_blockers else "blocked"
    decision = READY_DECISION if status == "ready" else BLOCKED_DECISION
    next_tranche = NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_HOLD
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": decision,
        "nextRecommendedTranche": next_tranche,
        "sourceProductionVectorDbIntegrationDesign": _source_design_summary(
            production_vector_db_integration_design,
            report_ref=source_production_vector_db_integration_design_ref,
        ),
        "input": {
            "sourceProductionVectorDbIntegrationDesignRef": normalize_text(
                source_production_vector_db_integration_design_ref
            ),
            "expectedHintRows": EXPECTED_HINT_ROWS,
            "productionNamespace": PRODUCTION_NAMESPACE,
            "targetVectorDatabaseClass": TARGET_VECTOR_DATABASE_CLASS,
            "targetWriteMethod": TARGET_WRITE_METHOD,
            "targetVectorStoreRef": TARGET_STORE_REF,
            "targetCollectionName": TARGET_COLLECTION_NAME,
        },
        "scope": _scope(len(planned_rows)),
        "policy": _policy(),
        "method": {
            "name": "visual_retrieval_hint_production_vector_db_integration_dry_run_v1",
            "description": (
                "Converts the ready production-vector integration design into deterministic production "
                "vector record previews without mutating any vector, database, runtime, or evidence store."
            ),
            "writeBoundaryNotCalled": TARGET_WRITE_METHOD,
            "runtimeSearchBoundaryNotChanged": "knowledge_hub.ai.rag_search_runtime.RAGSearchRuntime.search_with_diagnostics",
            "limitations": [
                "No production vector DB write is performed.",
                "No embedder is called.",
                "No runtime search route is changed.",
                "No visual hint is promoted to citation-grade evidence.",
            ],
        },
        "counts": counts,
        "gate": gate,
        "dryRunRowsDetail": rows,
        "plannedProductionVectorRecords": planned_records,
        "sourceBlockers": sorted(set(source_blockers)),
        "technicalBlockers": technical_blockers,
        "warnings": [],
    }


def render_limited_visual_retrieval_hint_production_vector_db_integration_dry_run_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Limited Visual Retrieval Hint Production Vector DB Integration Dry Run 005",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- plannedProductionVectorRecordRows: `{counts.get('plannedProductionVectorRecordRows')}`",
        f"- dryRunReadyProductionVectorRecordRows: `{counts.get('dryRunReadyProductionVectorRecordRows')}`",
        f"- futureApplyCandidateRows: `{counts.get('futureApplyCandidateRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- policyViolationRows: `{counts.get('policyViolationRows')}`",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- embeddingCallRows: `{counts.get('embeddingCallRows')}`",
        f"- vectorIndexWriteRows: `{counts.get('vectorIndexWriteRows')}`",
        f"- productionVectorIndexWriteRows: `{counts.get('productionVectorIndexWriteRows')}`",
        f"- databaseMutationRows: `{counts.get('databaseMutationRows')}`",
        f"- runtimeVisibleRows: `{counts.get('runtimeVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        f"- answerableWithoutTextEvidenceRows: `{counts.get('answerableWithoutTextEvidenceRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Gate",
        "",
        f"- passed: `{gate.get('passed')}`",
        f"- plannedRowsExactly125: `{dict(gate.get('checks') or {}).get('plannedRowsExactly125')}`",
        f"- allRowsDryRunReady: `{dict(gate.get('checks') or {}).get('allRowsDryRunReady')}`",
        f"- noProductionMutation: `{dict(gate.get('checks') or {}).get('noProductionMutation')}`",
        "",
        "## Placement Preview",
        "",
        f"- targetVectorDatabaseClass: `{dict(report.get('input') or {}).get('targetVectorDatabaseClass')}`",
        f"- targetWriteMethod: `{dict(report.get('input') or {}).get('targetWriteMethod')}`",
        f"- targetVectorStoreRef: `{dict(report.get('input') or {}).get('targetVectorStoreRef')}`",
        f"- targetCollectionName: `{dict(report.get('input') or {}).get('targetCollectionName')}`",
        "",
        "## Non-Scope",
        "",
        "- No production vector DB write.",
        "- No Chroma/SQLite/knowledge DB mutation.",
        "- No embedding/model/API call.",
        "- No runtime answer exposure.",
        "- No evidence promotion.",
        "- No graph DB, ontology, memory card, or clustering write.",
    ]
    blockers = list(report.get("technicalBlockers") or [])
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def write_limited_visual_retrieval_hint_production_vector_db_integration_dry_run(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_limited_visual_retrieval_hint_production_vector_db_integration_dry_run_markdown(report),
        encoding="utf-8",
    )
    return {"json": str(report_json), "md": str(report_md)}


__all__ = [
    "BLOCKED_DECISION",
    "DRY_RUN_READY_STATUS",
    "LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DRY_RUN_SCHEMA_ID",
    "READY_DECISION",
    "build_limited_visual_retrieval_hint_production_vector_db_integration_dry_run",
    "load_json",
    "render_limited_visual_retrieval_hint_production_vector_db_integration_dry_run_markdown",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_production_vector_db_integration_dry_run",
]
