"""Production vector DB integration design for visual retrieval hints.

This helper consumes the ready labs search-quality eval and source apply
reports, then projects 125 visual retrieval hints into a production-vector
integration design. It is report-only: it never writes Chroma/SQLite/vector
state, never calls an embedder, and never exposes records to runtime answers.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_EXECUTOR_SCHEMA_ID,
    _contains_private_path,
    normalize_text,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_SCHEMA_ID,
    load_json,
    sanitized_report_ref,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_SEARCH_QUALITY_EVAL_SCHEMA_ID,
    READY_DECISION as SEARCH_QUALITY_READY_DECISION,
)
from knowledge_hub.papers.visual_retrieval_hint_search_eval import (
    VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID,
)


LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-production-vector-db-integration-design.v1"
)
PRODUCTION_VECTOR_INTEGRATION_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-production-vector-integration-design-row.v1"
)

READY_DECISION = "ready_for_limited_visual_retrieval_hint_production_vector_db_integration_dry_run"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE_READY = "limited_visual_retrieval_hint_production_vector_db_integration_dry_run"
NEXT_TRANCHE_HOLD = "limited_visual_retrieval_hint_production_vector_db_integration_design_review"

EXPECTED_HINT_ROWS = 125
PRODUCTION_NAMESPACE = "production_visual_retrieval_hint_candidates_v1"
PROPOSED_COLLECTION_NAME = "knowledge_hub_visual_retrieval_hints"
PROPOSED_STORE_REF = "config.vector_db_path/visual_retrieval_hints"

PLANNED_STATUS = "planned_production_vector_integration_candidate"
BLOCKED_SOURCE_STATUS = "blocked_source_report_gate"
BLOCKED_POLICY_STATUS = "blocked_policy_violation"
BLOCKED_CONTRACT_STATUS = "blocked_record_contract_violation"

MUTATION_COUNTER_FIELDS = (
    "productionVectorIndexWriteRows",
    "databaseMutationRows",
    "indexMutationRows",
    "reindexOrReembedRows",
    "runtimeVisibleRows",
    "strictEvidenceRows",
    "citationGradeRows",
    "answerableWithoutTextEvidenceRows",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _short_hash(value: str, *, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bbox(value: Any) -> list[Any]:
    return list(value or [])


def _counts(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("counts") or {})


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _is_false(value: Any) -> bool:
    return value is False


def _summary(report: dict[str, Any], *, report_ref: str, fields: tuple[str, ...]) -> dict[str, Any]:
    counts = _counts(report)
    payload: dict[str, Any] = {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
    }
    for field in fields:
        payload[field] = _int(counts.get(field))
    return payload


def _source_quality_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = _counts(report)
    quality_gate = dict(report.get("qualityGate") or {})
    observed = dict(quality_gate.get("observed") or {})
    payload = _summary(
        report,
        report_ref=report_ref,
        fields=(
            "sourcePlannedVectorUpsertRows",
            "sourceCandidateRowsCoveredByTextBaseline",
            "sourceCandidateRowsMissingFromTextBaseline",
            "actualLabsVectorIndexRows",
            "matchedVectorRecordRows",
            "queryRows",
            "textOnlyHitAt5Rows",
            "labsVectorHitAt5Rows",
            "hybridHitAt5Rows",
            "hybridHitAt5LiftRows",
            "rankRegressedRows",
            "blockedRows",
            "privatePathLeakRows",
            "schemaViolationCount",
        ),
    )
    payload["qualityGatePassed"] = quality_gate.get("passed") is True
    payload["hybridHitAt5LiftRows"] = _int(counts.get("hybridHitAt5LiftRows") or observed.get("hybridHitAt5LiftRows"))
    payload["textOnlyMrr"] = float(counts.get("textOnlyMrr") or observed.get("textOnlyMrr") or 0.0)
    payload["labsVectorMrr"] = float(counts.get("labsVectorMrr") or observed.get("labsVectorMrr") or 0.0)
    payload["hybridMrr"] = float(counts.get("hybridMrr") or observed.get("hybridMrr") or 0.0)
    return payload


def _layout_summary(layout_candidate_report: dict[str, Any] | None, *, report_ref: str) -> dict[str, Any]:
    if not layout_candidate_report:
        return {
            "schema": "",
            "status": "not_loaded",
            "reportRef": normalize_text(report_ref),
            "candidateRows": 0,
            "loaded": False,
        }
    rows = list(layout_candidate_report.get("candidateRowsDetail") or [])
    counts = _counts(layout_candidate_report)
    return {
        "schema": normalize_text(layout_candidate_report.get("schema")),
        "status": normalize_text(layout_candidate_report.get("status")),
        "reportRef": normalize_text(report_ref),
        "candidateRows": _int(counts.get("candidateRows") or len(rows)),
        "loaded": True,
    }


def _quality_eval_blockers(report: dict[str, Any]) -> list[str]:
    counts = _counts(report)
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_SEARCH_QUALITY_EVAL_SCHEMA_ID:
        blockers.append("invalid_labs_vector_index_search_quality_eval_schema")
    if report.get("status") != "ready":
        blockers.append("labs_vector_index_search_quality_eval_not_ready")
    if report.get("decision") != SEARCH_QUALITY_READY_DECISION:
        blockers.append("labs_vector_index_search_quality_eval_invalid_decision")
    if dict(report.get("qualityGate") or {}).get("passed") is not True:
        blockers.append("labs_vector_index_search_quality_gate_not_passed")
    if _int(counts.get("sourcePlannedVectorUpsertRows")) != EXPECTED_HINT_ROWS:
        blockers.append("source_planned_vector_upsert_rows_not_125")
    if _int(counts.get("matchedVectorRecordRows")) != EXPECTED_HINT_ROWS:
        blockers.append("matched_vector_record_rows_not_125")
    if _int(counts.get("sourceCandidateRowsMissingFromTextBaseline")) != 0:
        blockers.append("source_baseline_coverage_incomplete")
    if _int(counts.get("rankRegressedRows")) != 0:
        blockers.append("search_quality_rank_regressions_present")
    for field in (
        "candidateStoreWriteRows",
        "embeddingCallRows",
        "embeddingVectorWriteRows",
        "vectorIndexWriteRows",
        "indexEligibleRows",
        *MUTATION_COUNTER_FIELDS,
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
    ):
        if _int(counts.get(field)) != 0:
            blockers.append(f"labs_vector_index_search_quality_eval_has_{field}")
    return blockers


def _candidate_store_apply_blockers(report: dict[str, Any]) -> list[str]:
    counts = _counts(report)
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_EXECUTOR_SCHEMA_ID:
        blockers.append("invalid_candidate_store_apply_executor_schema")
    if report.get("status") != "applied":
        blockers.append("candidate_store_apply_executor_not_applied")
    if report.get("decision") != "applied_limited_visual_retrieval_hint_candidate_store_apply":
        blockers.append("candidate_store_apply_executor_invalid_decision")
    if _int(counts.get("appliedCandidateRecordRows")) != EXPECTED_HINT_ROWS:
        blockers.append("candidate_store_applied_rows_not_125")
    if _int(counts.get("readbackValidatedRows")) != EXPECTED_HINT_ROWS:
        blockers.append("candidate_store_readback_rows_not_125")
    for field in (
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
        *MUTATION_COUNTER_FIELDS,
    ):
        if _int(counts.get(field)) != 0:
            blockers.append(f"candidate_store_apply_executor_has_{field}")
    return blockers


def _labs_vector_apply_blockers(report: dict[str, Any]) -> list[str]:
    counts = _counts(report)
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_SCHEMA_ID:
        blockers.append("invalid_labs_vector_index_apply_executor_schema")
    if report.get("status") != "applied":
        blockers.append("labs_vector_index_apply_executor_not_applied")
    if report.get("decision") != "applied_limited_visual_retrieval_hint_candidate_store_labs_vector_index":
        blockers.append("labs_vector_index_apply_executor_invalid_decision")
    if _int(counts.get("appliedLabsVectorRecordRows")) != EXPECTED_HINT_ROWS:
        blockers.append("labs_vector_applied_rows_not_125")
    if _int(counts.get("readbackValidatedRows")) != EXPECTED_HINT_ROWS:
        blockers.append("labs_vector_readback_rows_not_125")
    if _int(counts.get("vectorIndexWriteRows")) != EXPECTED_HINT_ROWS:
        blockers.append("labs_vector_write_rows_not_125")
    for field in (
        "candidateStoreWriteRows",
        "externalEmbeddingCallRows",
        "productionVectorIndexWriteRows",
        "databaseMutationRows",
        "indexMutationRows",
        "reindexOrReembedRows",
        "indexEligibleRows",
        "runtimeVisibleRows",
        "strictEvidenceRows",
        "citationGradeRows",
        "answerableWithoutTextEvidenceRows",
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
    ):
        if _int(counts.get(field)) != 0:
            blockers.append(f"labs_vector_index_apply_executor_has_{field}")
    return blockers


def _dry_run_blockers(report: dict[str, Any]) -> list[str]:
    counts = _counts(report)
    records = list(report.get("plannedVectorUpsertRecords") or [])
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID:
        blockers.append("invalid_labs_vector_index_apply_executor_dry_run_schema")
    if report.get("status") != "ready":
        blockers.append("labs_vector_index_apply_executor_dry_run_not_ready")
    if _int(counts.get("plannedVectorUpsertRows")) != EXPECTED_HINT_ROWS:
        blockers.append("dry_run_planned_vector_upsert_rows_not_125")
    if len(records) != EXPECTED_HINT_ROWS:
        blockers.append("dry_run_planned_vector_upsert_records_not_125")
    for field in (
        "candidateStoreWriteRows",
        "embeddingCallRows",
        "embeddingVectorWriteRows",
        "vectorIndexWriteRows",
        "databaseMutationRows",
        "indexMutationRows",
        "reindexOrReembedRows",
        "indexEligibleRows",
        "runtimeVisibleRows",
        "strictEvidenceRows",
        "citationGradeRows",
        "answerableWithoutTextEvidenceRows",
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
    ):
        if _int(counts.get(field)) != 0:
            blockers.append(f"labs_vector_index_apply_executor_dry_run_has_{field}")
    return blockers


def _layout_blockers(layout_candidate_report: dict[str, Any] | None) -> list[str]:
    if not layout_candidate_report:
        return []
    blockers: list[str] = []
    if layout_candidate_report.get("schema") != VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID:
        blockers.append("invalid_layout_candidate_report_schema")
    rows = list(layout_candidate_report.get("candidateRowsDetail") or [])
    counts = _counts(layout_candidate_report)
    if _int(counts.get("candidateRows") or len(rows)) < EXPECTED_HINT_ROWS:
        blockers.append("layout_candidate_report_too_small_for_source_coverage")
    return blockers


def _policy_flags_ok(value: dict[str, Any]) -> bool:
    return (
        normalize_text(value.get("allowedUse")) == "retrieval_hint_only"
        and _is_false(value.get("strictEvidence"))
        and _is_false(value.get("citationGrade"))
        and _is_false(value.get("answerableWithoutTextEvidence"))
        and _is_false(value.get("runtimeVisible"))
        and _is_false(value.get("indexEligible"))
    )


def _row_contract_blockers(record: dict[str, Any]) -> list[str]:
    metadata = dict(record.get("metadata") or {})
    policy = dict(record.get("policy") or {})
    blockers: list[str] = []
    if record.get("schema") != "knowledge-hub.paper.visual-retrieval-hint-labs-vector-upsert-record.v1":
        blockers.append("invalid_source_upsert_record_schema")
    for field in (
        "hintCandidateId",
        "sourceCandidateId",
        "sourceContentHash",
        "paperId",
        "page",
        "bbox",
        "candidateType",
        "documentText",
        "documentTextHash",
        "embeddingText",
        "embeddingTextHash",
    ):
        if not record.get(field):
            blockers.append(f"missing_{field}")
    if not _policy_flags_ok(metadata):
        blockers.append("metadata_policy_not_retrieval_hint_only")
    if not _policy_flags_ok(policy):
        blockers.append("row_policy_not_retrieval_hint_only")
    if policy.get("productionIndexEligible") is not False:
        blockers.append("row_policy_production_index_eligible_not_false")
    if _contains_private_path(record):
        blockers.append("private_path_leak")
    return blockers


def _extract_keywords(record: dict[str, Any]) -> list[str]:
    embedding_text = normalize_text(record.get("embeddingText"))
    marker = "keywords="
    if marker not in embedding_text:
        return []
    fragment = embedding_text.split(marker, 1)[1].split("|", 1)[0]
    output: list[str] = []
    for item in fragment.split(","):
        token = normalize_text(item)
        if token and token not in output:
            output.append(token)
    return output[:24]


def _production_document_id(record: dict[str, Any]) -> str:
    basis = "|".join(
        [
            normalize_text(record.get("hintCandidateId")),
            normalize_text(record.get("sourceCandidateId")),
            normalize_text(record.get("embeddingTextHash")),
        ]
    )
    return "visual-retrieval-hint-production-vector-doc:" + _short_hash(basis)


def _production_metadata(record: dict[str, Any], *, production_document_id: str, keywords: list[str]) -> dict[str, Any]:
    return {
        "retrieval_unit_schema": "visual_retrieval_hint_production_vector_document_design.v1",
        "namespace": PRODUCTION_NAMESPACE,
        "source_type": "visual_retrieval_hint",
        "retrieval_unit_kind": "candidate_discovery_signal",
        "allowed_use": "retrieval_hint_only",
        "allowedUse": "retrieval_hint_only",
        "document_id": production_document_id,
        "title": f"{normalize_text(record.get('paperId'))} {normalize_text(record.get('candidateType'))} visual hint",
        "field": "paper_visual_retrieval_hint",
        "keywords": ", ".join(keywords),
        "hint_candidate_id": normalize_text(record.get("hintCandidateId")),
        "source_candidate_id": normalize_text(record.get("sourceCandidateId")),
        "paper_id": normalize_text(record.get("paperId")),
        "paper_ref": normalize_text(record.get("paperRef")),
        "source_content_hash": normalize_text(record.get("sourceContentHash")),
        "page": _int(record.get("page")),
        "bbox": _bbox(record.get("bbox")),
        "candidate_type": normalize_text(record.get("candidateType")),
        "document_text_hash": normalize_text(record.get("documentTextHash")),
        "embedding_text_hash": normalize_text(record.get("embeddingTextHash")),
        "strict_evidence": False,
        "strictEvidence": False,
        "citation_grade": False,
        "citationGrade": False,
        "answerable_without_text_evidence": False,
        "answerableWithoutTextEvidence": False,
        "runtime_visible": False,
        "runtimeVisible": False,
        "index_eligible": False,
        "indexEligible": False,
        "production_index_eligible": False,
        "candidate_discovery_only": True,
    }


def _source_refs(
    *,
    source_quality_eval_ref: str,
    source_candidate_store_apply_ref: str,
    source_labs_vector_index_apply_ref: str,
    source_labs_vector_index_apply_executor_dry_run_ref: str,
    source_layout_candidate_report_ref: str,
) -> dict[str, str]:
    return {
        "sourceQualityEvalReportRef": normalize_text(source_quality_eval_ref),
        "sourceCandidateStoreApplyReportRef": normalize_text(source_candidate_store_apply_ref),
        "sourceLabsVectorIndexApplyReportRef": normalize_text(source_labs_vector_index_apply_ref),
        "sourceLabsVectorIndexApplyExecutorDryRunRef": normalize_text(
            source_labs_vector_index_apply_executor_dry_run_ref
        ),
        "sourceLayoutCandidateReportRef": normalize_text(source_layout_candidate_report_ref),
    }


def _integration_row(
    record: dict[str, Any],
    *,
    source_blockers: list[str],
    source_refs: dict[str, str],
) -> dict[str, Any]:
    row_blockers = _row_contract_blockers(record)
    keywords = _extract_keywords(record)
    production_document_id = _production_document_id(record)
    if source_blockers:
        status = BLOCKED_SOURCE_STATUS
    elif any(blocker.endswith("policy_not_retrieval_hint_only") or "policy_" in blocker for blocker in row_blockers):
        status = BLOCKED_POLICY_STATUS
    elif row_blockers:
        status = BLOCKED_CONTRACT_STATUS
    else:
        status = PLANNED_STATUS
    row_id_basis = "|".join(
        [
            production_document_id,
            normalize_text(record.get("hintCandidateId")),
            normalize_text(record.get("sourceCandidateId")),
        ]
    )
    metadata = _production_metadata(record, production_document_id=production_document_id, keywords=keywords)
    return {
        "schema": PRODUCTION_VECTOR_INTEGRATION_ROW_SCHEMA_ID,
        "rowId": "limited-visual-retrieval-hint-production-vector-db-integration-design:" + _short_hash(row_id_basis),
        "status": status,
        "productionNamespace": PRODUCTION_NAMESPACE,
        "productionVectorDocumentId": production_document_id,
        "hintCandidateId": normalize_text(record.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(record.get("sourceCandidateId")),
        "sourceContentHash": normalize_text(record.get("sourceContentHash")),
        "paperId": normalize_text(record.get("paperId")),
        "paperRef": normalize_text(record.get("paperRef")),
        "page": _int(record.get("page")),
        "bbox": _bbox(record.get("bbox")),
        "candidateType": normalize_text(record.get("candidateType")),
        "derivedTextForRetrieval": normalize_text(record.get("documentText")),
        "retrievalKeywords": keywords,
        "documentTextHash": normalize_text(record.get("documentTextHash")),
        "embeddingTextHash": normalize_text(record.get("embeddingTextHash")),
        "plannedDocumentText": normalize_text(record.get("documentText")),
        "plannedEmbeddingText": normalize_text(record.get("embeddingText")),
        "plannedMetadata": metadata,
        "sourceRefs": source_refs,
        "routeDesign": {
            "targetVectorDatabaseClass": "knowledge_hub.infrastructure.persistence.vector.VectorDatabase",
            "writeMethod": "VectorDatabase.add_documents",
            "proposedCollectionName": PROPOSED_COLLECTION_NAME,
            "proposedStoreRef": PROPOSED_STORE_REF,
            "routingMode": "candidate_discovery_only",
            "mergePolicy": "visual_hint_candidates_may_expand_candidates_but_must_not_be_citation_evidence",
            "answerEvidenceGate": "strict_text_evidence_required_after_candidate_discovery",
            "defaultRuntimeExposure": False,
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
            "futureApplyGateRequired": True,
            "actualCandidateStoreWrite": False,
            "actualEmbeddingCall": False,
            "actualEmbeddingVectorWrite": False,
            "actualVectorIndexWrite": False,
            "actualProductionVectorIndexWrite": False,
            "actualDatabaseMutation": False,
            "actualRuntimeExposure": False,
            "actualEvidencePromotion": False,
        },
        "blockers": [*source_blockers, *row_blockers],
    }


def _scope(planned_rows: int) -> dict[str, Any]:
    return {
        "writes": "report_only_design",
        "plannedProductionVectorRecordRows": planned_rows,
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
        "reportOnly": True,
        "designOnly": True,
        "candidateDiscoveryOnly": True,
        "candidateStoreWrite": False,
        "embeddingCalls": False,
        "vectorIndexWrite": False,
        "productionVectorIndexWrite": False,
        "operationalSearchIndexQuery": False,
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
    }


def _gate(counts: dict[str, Any], source_blockers: list[str]) -> dict[str, Any]:
    checks = {
        "sourceQualityEvalReady": "labs_vector_index_search_quality_eval_not_ready" not in source_blockers
        and "labs_vector_index_search_quality_eval_invalid_decision" not in source_blockers,
        "sourceQualityGatePassed": "labs_vector_index_search_quality_gate_not_passed" not in source_blockers,
        "plannedRowsExactly125": _int(counts.get("productionVectorIntegrationCandidateRows")) == EXPECTED_HINT_ROWS,
        "allRowsPlanned": _int(counts.get("plannedProductionVectorRecordRows")) == EXPECTED_HINT_ROWS,
        "baselineCoverageComplete": _int(counts.get("sourceCandidateRowsMissingFromTextBaseline")) == 0,
        "noPolicyViolations": _int(counts.get("policyViolationRows")) == 0,
        "noPrivatePathLeaks": _int(counts.get("privatePathLeakRows")) == 0,
        "noProductionMutation": all(_int(counts.get(field)) == 0 for field in MUTATION_COUNTER_FIELDS),
        "noVectorOrDatabaseWrite": all(
            _int(counts.get(field)) == 0
            for field in (
                "candidateStoreWriteRows",
                "embeddingCallRows",
                "embeddingVectorWriteRows",
                "vectorIndexWriteRows",
                "productionVectorIndexWriteRows",
                "databaseMutationRows",
                "indexMutationRows",
            )
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expectedRows": EXPECTED_HINT_ROWS,
        "observed": {
            "plannedProductionVectorRecordRows": _int(counts.get("plannedProductionVectorRecordRows")),
            "blockedRows": _int(counts.get("blockedRows")),
            "sourceCandidateRowsMissingFromTextBaseline": _int(
                counts.get("sourceCandidateRowsMissingFromTextBaseline")
            ),
        },
    }


def _count_rows(rows: list[dict[str, Any]], source_quality_eval: dict[str, Any]) -> dict[str, Any]:
    by_type = Counter(normalize_text(row.get("candidateType")) for row in rows)
    planned_rows = [row for row in rows if row.get("status") == PLANNED_STATUS]
    blocked_rows = [row for row in rows if row.get("status") != PLANNED_STATUS]
    counts = _counts(source_quality_eval)
    observed = dict(dict(source_quality_eval.get("qualityGate") or {}).get("observed") or {})
    return {
        "sourceQualityEvalRows": _int(counts.get("sourcePlannedVectorUpsertRows")),
        "sourceCandidateStoreAppliedRows": EXPECTED_HINT_ROWS,
        "sourceLabsVectorAppliedRows": _int(counts.get("actualLabsVectorIndexRows")),
        "sourcePlannedVectorUpsertRows": _int(counts.get("sourcePlannedVectorUpsertRows")),
        "sourceCandidateRowsCoveredByTextBaseline": _int(counts.get("sourceCandidateRowsCoveredByTextBaseline")),
        "sourceCandidateRowsMissingFromTextBaseline": _int(
            counts.get("sourceCandidateRowsMissingFromTextBaseline")
        ),
        "qualityEvalQueryRows": _int(counts.get("queryRows")),
        "qualityEvalTextOnlyHitAt5Rows": _int(counts.get("textOnlyHitAt5Rows")),
        "qualityEvalLabsVectorHitAt5Rows": _int(counts.get("labsVectorHitAt5Rows")),
        "qualityEvalHybridHitAt5Rows": _int(counts.get("hybridHitAt5Rows")),
        "qualityEvalHybridHitAt5LiftRows": _int(
            counts.get("hybridHitAt5LiftRows") or observed.get("hybridHitAt5LiftRows")
        ),
        "productionVectorIntegrationCandidateRows": len(rows),
        "plannedProductionVectorRecordRows": len(planned_rows),
        "candidateDiscoveryOnlyRows": len(planned_rows),
        "blockedRows": len(blocked_rows),
        "policyViolationRows": sum(
            1 for row in rows if row.get("status") == BLOCKED_POLICY_STATUS or "policy" in " ".join(row.get("blockers") or [])
        ),
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
        "schemaViolationCount": 0,
        "byCandidateType": dict(sorted(by_type.items())),
    }


def build_limited_visual_retrieval_hint_production_vector_db_integration_design(
    *,
    search_quality_eval_report: dict[str, Any],
    candidate_store_apply_report: dict[str, Any],
    labs_vector_index_apply_report: dict[str, Any],
    labs_vector_index_apply_executor_dry_run: dict[str, Any],
    source_search_quality_eval_report_ref: str,
    source_candidate_store_apply_report_ref: str,
    source_labs_vector_index_apply_report_ref: str,
    source_labs_vector_index_apply_executor_dry_run_ref: str,
    layout_candidate_report: dict[str, Any] | None = None,
    source_layout_candidate_report_ref: str = "",
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_blockers = [
        *_quality_eval_blockers(search_quality_eval_report),
        *_candidate_store_apply_blockers(candidate_store_apply_report),
        *_labs_vector_apply_blockers(labs_vector_index_apply_report),
        *_dry_run_blockers(labs_vector_index_apply_executor_dry_run),
        *_layout_blockers(layout_candidate_report),
    ]
    refs = _source_refs(
        source_quality_eval_ref=source_search_quality_eval_report_ref,
        source_candidate_store_apply_ref=source_candidate_store_apply_report_ref,
        source_labs_vector_index_apply_ref=source_labs_vector_index_apply_report_ref,
        source_labs_vector_index_apply_executor_dry_run_ref=source_labs_vector_index_apply_executor_dry_run_ref,
        source_layout_candidate_report_ref=source_layout_candidate_report_ref,
    )
    rows = [
        _integration_row(dict(record), source_blockers=source_blockers, source_refs=refs)
        for record in list(labs_vector_index_apply_executor_dry_run.get("plannedVectorUpsertRecords") or [])
    ]
    counts = _count_rows(rows, search_quality_eval_report)
    gate = _gate(counts, source_blockers)
    technical_blockers = sorted(set(source_blockers + [blocker for row in rows for blocker in row.get("blockers", [])]))
    status = "ready" if gate.get("passed") and not technical_blockers else "blocked"
    decision = READY_DECISION if status == "ready" else BLOCKED_DECISION
    next_tranche = NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_HOLD
    planned_rows = [row for row in rows if row.get("status") == PLANNED_STATUS]
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DESIGN_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": decision,
        "nextRecommendedTranche": next_tranche,
        "sourceSearchQualityEvalReport": _source_quality_summary(
            search_quality_eval_report,
            report_ref=source_search_quality_eval_report_ref,
        ),
        "sourceCandidateStoreApplyReport": _summary(
            candidate_store_apply_report,
            report_ref=source_candidate_store_apply_report_ref,
            fields=(
                "appliedCandidateRecordRows",
                "candidateStoreWriteRows",
                "readbackValidatedRows",
                "blockedRows",
                "privatePathLeakRows",
                "schemaViolationCount",
            ),
        ),
        "sourceLabsVectorIndexApplyReport": _summary(
            labs_vector_index_apply_report,
            report_ref=source_labs_vector_index_apply_report_ref,
            fields=(
                "appliedLabsVectorRecordRows",
                "readbackValidatedRows",
                "vectorIndexWriteRows",
                "productionVectorIndexWriteRows",
                "candidateStoreWriteRows",
                "blockedRows",
                "privatePathLeakRows",
                "schemaViolationCount",
            ),
        ),
        "sourceLabsVectorIndexApplyExecutorDryRun": _summary(
            labs_vector_index_apply_executor_dry_run,
            report_ref=source_labs_vector_index_apply_executor_dry_run_ref,
            fields=(
                "executorDryRunRows",
                "plannedVectorUpsertRows",
                "embeddingInputRows",
                "candidateStoreWriteRows",
                "vectorIndexWriteRows",
                "blockedRows",
                "privatePathLeakRows",
                "schemaViolationCount",
            ),
        ),
        "sourceLayoutCandidateReport": _layout_summary(
            layout_candidate_report,
            report_ref=source_layout_candidate_report_ref,
        ),
        "input": {
            "expectedHintRows": EXPECTED_HINT_ROWS,
            "productionNamespace": PRODUCTION_NAMESPACE,
            "proposedCollectionName": PROPOSED_COLLECTION_NAME,
            "proposedStoreRef": PROPOSED_STORE_REF,
            "baselineCoverageSource": "source_search_quality_eval_counts",
        },
        "productionVectorIntegrationDesign": {
            "targetVectorDatabaseClass": "knowledge_hub.infrastructure.persistence.vector.VectorDatabase",
            "writeMethod": "VectorDatabase.add_documents",
            "physicalStorePlacement": PROPOSED_STORE_REF,
            "collectionPlacement": PROPOSED_COLLECTION_NAME,
            "runtimeRoute": "separate_candidate_discovery_route_after_explicit_dry_run_and_apply_gate",
            "defaultRuntimeExposure": False,
            "evidenceBoundary": "visual_hint_vectors_return_candidate_refs_only; answers still require strict text evidence",
            "whySeparateFromDefaultCollection": (
                "The default RAG route opens config.collection_name and would search any documents in that "
                "collection unless a runtime filter is added; the first production design therefore keeps "
                "visual hints behind an isolated candidate-discovery placement."
            ),
        },
        "scope": _scope(len(planned_rows)),
        "policy": _policy(),
        "method": {
            "name": "visual_retrieval_hint_production_vector_db_integration_design_v1",
            "description": (
                "Projects applied labs visual retrieval-hint vector records into a production vector DB "
                "integration design without mutating any production store or runtime route."
            ),
            "implementationFinding": {
                "writeBoundary": "knowledge_hub.infrastructure.persistence.vector.VectorDatabase.add_documents",
                "runtimeSearchBoundary": "knowledge_hub.ai.rag_search_runtime.RAGSearchRuntime.search_with_diagnostics",
                "defaultIndexCommandBoundary": "knowledge_hub.interfaces.cli.commands.index_cmd",
            },
            "limitations": [
                "No production vector DB write is performed.",
                "No embedder is called.",
                "No runtime search route is changed.",
                "No visual hint is promoted to citation-grade evidence.",
            ],
        },
        "counts": counts,
        "gate": gate,
        "rows": rows,
        "productionVectorRecordPreviews": planned_rows,
        "sourceBlockers": sorted(set(source_blockers)),
        "technicalBlockers": technical_blockers,
        "warnings": [],
    }


def render_limited_visual_retrieval_hint_production_vector_db_integration_design_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    design = dict(report.get("productionVectorIntegrationDesign") or {})
    lines = [
        "# Limited Visual Retrieval Hint Production Vector DB Integration Design 005",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- productionVectorIntegrationCandidateRows: `{counts.get('productionVectorIntegrationCandidateRows')}`",
        f"- plannedProductionVectorRecordRows: `{counts.get('plannedProductionVectorRecordRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- sourceCandidateRowsMissingFromTextBaseline: `{counts.get('sourceCandidateRowsMissingFromTextBaseline')}`",
        f"- qualityEvalHybridHitAt5Rows: `{counts.get('qualityEvalHybridHitAt5Rows')}`",
        f"- qualityEvalHybridHitAt5LiftRows: `{counts.get('qualityEvalHybridHitAt5LiftRows')}`",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- embeddingCallRows: `{counts.get('embeddingCallRows')}`",
        f"- vectorIndexWriteRows: `{counts.get('vectorIndexWriteRows')}`",
        f"- productionVectorIndexWriteRows: `{counts.get('productionVectorIndexWriteRows')}`",
        f"- databaseMutationRows: `{counts.get('databaseMutationRows')}`",
        f"- runtimeVisibleRows: `{counts.get('runtimeVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Gate",
        "",
        f"- passed: `{gate.get('passed')}`",
        f"- plannedRowsExactly125: `{dict(gate.get('checks') or {}).get('plannedRowsExactly125')}`",
        f"- noProductionMutation: `{dict(gate.get('checks') or {}).get('noProductionMutation')}`",
        f"- noVectorOrDatabaseWrite: `{dict(gate.get('checks') or {}).get('noVectorOrDatabaseWrite')}`",
        "",
        "## Placement",
        "",
        f"- writeBoundary: `{dict(report.get('method') or {}).get('implementationFinding', {}).get('writeBoundary')}`",
        f"- proposedStoreRef: `{design.get('physicalStorePlacement')}`",
        f"- proposedCollectionName: `{design.get('collectionPlacement')}`",
        f"- runtimeRoute: `{design.get('runtimeRoute')}`",
        f"- evidenceBoundary: `{design.get('evidenceBoundary')}`",
        "",
        "## Non-Scope",
        "",
        "- No production vector DB write.",
        "- No Chroma/SQLite/knowledge DB mutation.",
        "- No operational index rebuild.",
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


def write_limited_visual_retrieval_hint_production_vector_db_integration_design(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_limited_visual_retrieval_hint_production_vector_db_integration_design_markdown(report),
        encoding="utf-8",
    )
    return {"json": str(report_json), "md": str(report_md)}


__all__ = [
    "BLOCKED_DECISION",
    "LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DESIGN_SCHEMA_ID",
    "PRODUCTION_NAMESPACE",
    "READY_DECISION",
    "build_limited_visual_retrieval_hint_production_vector_db_integration_design",
    "load_json",
    "render_limited_visual_retrieval_hint_production_vector_db_integration_design_markdown",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_production_vector_db_integration_design",
]
