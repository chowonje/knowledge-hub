"""Apply-executor dry-run for labs visual retrieval-hint vector indexing.

This module previews labs vector upsert records from reviewed visual
retrieval-hint vector documents. It does not call an embedder, produce vectors,
write a vector DB, mutate indexes, expose hints at runtime, or promote evidence.
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
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run import (
    LABS_NAMESPACE,
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_DRY_RUN_SCHEMA_ID,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_review import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_REVIEW_SCHEMA_ID,
    READY_DECISION as LABS_VECTOR_INDEX_REVIEW_READY_DECISION,
    REVIEW_STATUS_READY,
)


LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-labs-vector-index-apply-executor-dry-run.v1"
)

EXECUTOR_DRY_RUN_STATUS_READY = "dry_run_ready_labs_vector_upsert"
EXECUTOR_DRY_RUN_STATUS_BLOCKED_NON_READY_REVIEW_ROW = "blocked_non_ready_review_row"
EXECUTOR_DRY_RUN_STATUS_BLOCKED_MISSING_VECTOR_DOCUMENT = "blocked_missing_vector_document"
EXECUTOR_DRY_RUN_STATUS_BLOCKED_VECTOR_DOCUMENT_CONTRACT = "blocked_vector_document_contract"
EXECUTOR_DRY_RUN_STATUS_BLOCKED_POLICY_VIOLATION = "blocked_policy_violation"

READY_DECISION = "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_review"
BLOCKED_DECISION = "blocked"
NEXT_RECOMMENDED_TRANCHE = "limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_review"

PLANNED_LABS_VECTOR_INDEX_REF = "labs_vector_index/labs_visual_retrieval_hint_candidates_v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitized_report_ref(path: Path, *, project_root: Path | None = None) -> str:
    resolved = path.expanduser()
    if project_root is not None:
        try:
            return resolved.resolve().relative_to(project_root.resolve()).as_posix()
        except Exception:
            pass
    return f"input_reports/{resolved.name}"


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_hash(record: dict[str, Any]) -> str:
    return _sha256_text(_canonical_json(record))


def _bbox(value: Any) -> list[Any]:
    return list(value or [])


def _policy_flags_ok(container: dict[str, Any]) -> bool:
    return (
        normalize_text(container.get("allowedUse")) == "retrieval_hint_only"
        and container.get("strictEvidence") is False
        and container.get("citationGrade") is False
        and container.get("answerableWithoutTextEvidence") is False
        and container.get("runtimeVisible") is False
        and container.get("indexEligible") is False
    )


def _source_review_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "reviewRows": int(counts.get("reviewRows") or 0),
        "reviewReadyRows": int(counts.get("reviewReadyRows") or 0),
        "applyExecutorDryRunCandidateRows": int(counts.get("applyExecutorDryRunCandidateRows") or 0),
        "qualityGatePassedRows": int(counts.get("qualityGatePassedRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "privatePathLeakRows": int(counts.get("privatePathLeakRows") or 0),
        "schemaViolationCount": int(counts.get("schemaViolationCount") or 0),
    }


def _source_dry_run_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "plannedVectorDocumentRows": int(counts.get("plannedVectorDocumentRows") or 0),
        "plannedNamespaceRows": int(counts.get("plannedNamespaceRows") or 0),
        "labsQueryRows": int(counts.get("labsQueryRows") or 0),
        "labsHitAt5Rows": int(counts.get("labsHitAt5Rows") or 0),
        "labsHitAt10Rows": int(counts.get("labsHitAt10Rows") or 0),
        "candidateStoreWriteRows": int(counts.get("candidateStoreWriteRows") or 0),
        "embeddingCallRows": int(counts.get("embeddingCallRows") or 0),
        "vectorIndexWriteRows": int(counts.get("vectorIndexWriteRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "privatePathLeakRows": int(counts.get("privatePathLeakRows") or 0),
        "schemaViolationCount": int(counts.get("schemaViolationCount") or 0),
    }


def _source_review_blockers(review: dict[str, Any]) -> list[str]:
    counts = dict(review.get("counts") or {})
    blockers: list[str] = []
    if review.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_REVIEW_SCHEMA_ID:
        blockers.append("invalid_labs_vector_index_review_schema")
    if review.get("status") != "ready":
        blockers.append("labs_vector_index_review_not_ready")
    if review.get("decision") != LABS_VECTOR_INDEX_REVIEW_READY_DECISION:
        blockers.append("labs_vector_index_review_invalid_decision")
    for field_name in (
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
        "candidateStoreWriteRows",
        "embeddingCallRows",
        "vectorIndexWriteRows",
        "databaseMutationRows",
        "indexMutationRows",
        "reindexOrReembedRows",
        "indexEligibleRows",
        "runtimeVisibleRows",
        "strictEvidenceRows",
        "citationGradeRows",
        "answerableWithoutTextEvidenceRows",
    ):
        if int(counts.get(field_name) or 0) != 0:
            blockers.append(f"labs_vector_index_review_has_{field_name}")
    if int(counts.get("reviewReadyRows") or 0) <= 0:
        blockers.append("labs_vector_index_review_has_no_ready_rows")
    if int(counts.get("reviewRows") or 0) != int(counts.get("reviewReadyRows") or 0):
        blockers.append("labs_vector_index_review_ready_count_mismatch")
    return blockers


def _source_dry_run_blockers(dry_run: dict[str, Any]) -> list[str]:
    counts = dict(dry_run.get("counts") or {})
    blockers: list[str] = []
    if dry_run.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_DRY_RUN_SCHEMA_ID:
        blockers.append("invalid_labs_vector_index_dry_run_schema")
    if dry_run.get("status") != "ready":
        blockers.append("labs_vector_index_dry_run_not_ready")
    for field_name in (
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
        "candidateStoreWriteRows",
        "embeddingCallRows",
        "vectorIndexWriteRows",
        "databaseMutationRows",
        "indexMutationRows",
        "reindexOrReembedRows",
        "indexEligibleRows",
        "runtimeVisibleRows",
        "strictEvidenceRows",
        "citationGradeRows",
        "answerableWithoutTextEvidenceRows",
    ):
        if int(counts.get(field_name) or 0) != 0:
            blockers.append(f"labs_vector_index_dry_run_has_{field_name}")
    return blockers


def _scope(planned_rows: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "plannedVectorUpsertRows": int(planned_rows),
        "candidateStoreWriteRows": 0,
        "embeddingCallRows": 0,
        "embeddingVectorWriteRows": 0,
        "vectorIndexWriteRows": 0,
        "vectorIndexing": False,
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidencePromotionRows": 0,
        "runtimeAnswerVisibleExposureRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "answerabilityGateBypassRows": 0,
        "operationalSearchIndexQueryRows": 0,
        "answerGenerationRows": 0,
    }


def _policy() -> dict[str, Any]:
    return {
        "dryRunOnly": True,
        "labsOnly": True,
        "separateExplicitLabsApplyRequired": True,
        "candidateStoreWrite": False,
        "embeddingCalls": False,
        "embeddingVectorWrite": False,
        "vectorIndexWrite": False,
        "sourceSpanCreated": False,
        "strictEvidenceCreated": False,
        "citationGradeEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "vaultScan": False,
        "reindexOrReembed": False,
        "allowedUse": "retrieval_hint_only",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "runtimeVisible": False,
        "indexEligible": False,
    }


def _planned_vector_upsert_record(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-labs-vector-upsert-record.v1",
        "namespace": LABS_NAMESPACE,
        "plannedVectorIndexRef": PLANNED_LABS_VECTOR_INDEX_REF,
        "vectorDocumentId": normalize_text(document.get("vectorDocumentId")),
        "hintCandidateId": normalize_text(document.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(document.get("sourceCandidateId")),
        "paperId": normalize_text(document.get("paperId")),
        "paperRef": normalize_text(document.get("paperRef")),
        "sourceContentHash": normalize_text(document.get("sourceContentHash")),
        "page": int(document.get("page") or 0),
        "bbox": _bbox(document.get("bbox")),
        "candidateType": normalize_text(document.get("candidateType")),
        "sourceRecordSha256": normalize_text(document.get("sourceRecordSha256")),
        "documentText": str(document.get("documentText") or ""),
        "documentTextHash": normalize_text(document.get("documentTextHash")),
        "embeddingText": str(document.get("embeddingText") or ""),
        "embeddingTextHash": normalize_text(document.get("embeddingTextHash")),
        "metadata": dict(document.get("metadata") or {}),
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
            "productionIndexEligible": False,
            "labsOnly": True,
        },
        "executionPlan": {
            "embeddingProviderRef": "deferred_to_separate_explicit_labs_apply",
            "embeddingVectorPresent": False,
            "wouldCallEmbedderOnSeparateExplicitApply": True,
            "wouldWriteVectorIndexOnSeparateExplicitApply": True,
            "actualEmbeddingCall": False,
            "actualEmbeddingVectorWrite": False,
            "actualVectorIndexWrite": False,
        },
    }


def _document_contract_checks(
    *,
    review_row: dict[str, Any],
    document: dict[str, Any],
    upsert_record: dict[str, Any] | None,
) -> dict[str, bool]:
    metadata = dict(document.get("metadata") or {})
    policy = dict(document.get("policy") or {})
    return {
        "reviewRowReady": normalize_text(review_row.get("reviewStatus")) == REVIEW_STATUS_READY,
        "reviewApplyExecutorDryRunCandidate": review_row.get("applyExecutorDryRunCandidate") is True,
        "documentPresent": bool(document),
        "namespaceMatchesLabs": normalize_text(document.get("namespace")) == LABS_NAMESPACE,
        "vectorDocumentIdMatches": normalize_text(document.get("vectorDocumentId"))
        == normalize_text(review_row.get("vectorDocumentId")),
        "hintCandidateIdMatches": normalize_text(document.get("hintCandidateId"))
        == normalize_text(review_row.get("hintCandidateId")),
        "sourceCandidateIdMatches": normalize_text(document.get("sourceCandidateId"))
        == normalize_text(review_row.get("sourceCandidateId")),
        "sourceContentHashMatches": normalize_text(document.get("sourceContentHash"))
        == normalize_text(review_row.get("sourceContentHash")),
        "pageMatches": int(document.get("page") or 0) == int(review_row.get("page") or 0),
        "bboxMatches": _bbox(document.get("bbox")) == _bbox(review_row.get("bbox")),
        "candidateTypeMatches": normalize_text(document.get("candidateType"))
        == normalize_text(review_row.get("candidateType")),
        "documentTextPresent": bool(normalize_text(document.get("documentText"))),
        "embeddingTextPresent": bool(normalize_text(document.get("embeddingText"))),
        "documentTextHashMatches": normalize_text(document.get("documentTextHash"))
        == normalize_text(review_row.get("documentTextHash"))
        == _sha256_text(str(document.get("documentText") or "")),
        "embeddingTextHashMatches": normalize_text(document.get("embeddingTextHash"))
        == normalize_text(review_row.get("embeddingTextHash"))
        == _sha256_text(str(document.get("embeddingText") or "")),
        "metadataSchemaMatches": normalize_text(metadata.get("retrieval_unit_schema"))
        == "visual_retrieval_hint_vector_document.v1",
        "metadataPolicyRetrievalHintOnly": _policy_flags_ok(metadata),
        "documentPolicyRetrievalHintOnly": _policy_flags_ok(policy),
        "documentNotPrivatePathLeaking": not _contains_private_path(document),
        "upsertRecordPresent": bool(upsert_record),
        "upsertRecordNoEmbeddingVector": bool(upsert_record)
        and dict(upsert_record.get("executionPlan") or {}).get("embeddingVectorPresent") is False,
        "upsertActualEmbeddingCallFalse": bool(upsert_record)
        and dict(upsert_record.get("executionPlan") or {}).get("actualEmbeddingCall") is False,
        "upsertActualVectorIndexWriteFalse": bool(upsert_record)
        and dict(upsert_record.get("executionPlan") or {}).get("actualVectorIndexWrite") is False,
        "upsertPolicyRetrievalHintOnly": bool(upsert_record) and _policy_flags_ok(dict(upsert_record.get("policy") or {})),
    }


def _execution_status(
    *,
    review_row: dict[str, Any],
    document: dict[str, Any] | None,
    blockers: list[str],
) -> str:
    if normalize_text(review_row.get("reviewStatus")) != REVIEW_STATUS_READY:
        return EXECUTOR_DRY_RUN_STATUS_BLOCKED_NON_READY_REVIEW_ROW
    if not document:
        return EXECUTOR_DRY_RUN_STATUS_BLOCKED_MISSING_VECTOR_DOCUMENT
    policy_blockers = {
        "metadataPolicyRetrievalHintOnly",
        "documentPolicyRetrievalHintOnly",
        "documentNotPrivatePathLeaking",
        "upsertPolicyRetrievalHintOnly",
    }
    if any(blocker in policy_blockers for blocker in blockers):
        return EXECUTOR_DRY_RUN_STATUS_BLOCKED_POLICY_VIOLATION
    if blockers:
        return EXECUTOR_DRY_RUN_STATUS_BLOCKED_VECTOR_DOCUMENT_CONTRACT
    return EXECUTOR_DRY_RUN_STATUS_READY


def build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run(
    *,
    labs_vector_index_review: dict[str, Any],
    labs_vector_index_dry_run: dict[str, Any],
    source_labs_vector_index_review_ref: str,
    source_labs_vector_index_dry_run_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_blockers = sorted(
        set(_source_review_blockers(labs_vector_index_review) + _source_dry_run_blockers(labs_vector_index_dry_run))
    )
    review_rows = [
        dict(row) for row in labs_vector_index_review.get("reviewRowsDetail") or [] if isinstance(row, dict)
    ]
    documents_by_id = {
        normalize_text(row.get("vectorDocumentId")): dict(row)
        for row in labs_vector_index_dry_run.get("plannedVectorDocuments") or []
        if isinstance(row, dict)
    }
    executor_rows: list[dict[str, Any]] = []
    planned_records: list[dict[str, Any]] = []
    for index, review_row in enumerate(review_rows, start=1):
        vector_doc_id = normalize_text(review_row.get("vectorDocumentId"))
        document = documents_by_id.get(vector_doc_id)
        upsert_record = _planned_vector_upsert_record(document) if document else None
        checks = _document_contract_checks(
            review_row=review_row,
            document=document or {},
            upsert_record=upsert_record,
        )
        blockers = [name for name, passed in checks.items() if not passed]
        status = _execution_status(review_row=review_row, document=document, blockers=blockers)
        if upsert_record and status == EXECUTOR_DRY_RUN_STATUS_READY:
            planned_records.append(upsert_record)
        executor_rows.append(
            {
                "executorDryRunRowId": (
                    "limited-visual-retrieval-hint-candidate-store-labs-vector-index-apply-executor-dry-run:"
                    f"{index:04d}"
                ),
                "sourceReviewRowId": normalize_text(review_row.get("reviewRowId")),
                "sourcePlanRowId": normalize_text(review_row.get("sourcePlanRowId")),
                "vectorDocumentId": vector_doc_id,
                "namespace": LABS_NAMESPACE if document else "",
                "hintCandidateId": normalize_text(review_row.get("hintCandidateId")),
                "sourceCandidateId": normalize_text(review_row.get("sourceCandidateId")),
                "paperId": normalize_text(review_row.get("paperId")),
                "paperRef": normalize_text(review_row.get("paperRef")),
                "sourceContentHash": normalize_text(review_row.get("sourceContentHash")),
                "page": int(review_row.get("page") or 0),
                "bbox": _bbox(review_row.get("bbox")),
                "candidateType": normalize_text(review_row.get("candidateType")),
                "documentTextHash": normalize_text(review_row.get("documentTextHash")),
                "embeddingTextHash": normalize_text(review_row.get("embeddingTextHash")),
                "plannedVectorUpsertRecordSha256": _record_hash(upsert_record) if upsert_record else "",
                "plannedVectorIndexRef": PLANNED_LABS_VECTOR_INDEX_REF if upsert_record else "",
                "wouldCallEmbedderOnSeparateExplicitApply": bool(upsert_record) and status == EXECUTOR_DRY_RUN_STATUS_READY,
                "wouldWriteVectorIndexOnSeparateExplicitApply": bool(upsert_record)
                and status == EXECUTOR_DRY_RUN_STATUS_READY,
                "actualEmbeddingCall": False,
                "actualEmbeddingVectorWrite": False,
                "actualVectorIndexWrite": False,
                "indexEligible": False,
                "runtimeVisible": False,
                "strictEvidence": False,
                "citationGrade": False,
                "answerableWithoutTextEvidence": False,
                "executionStatus": status,
                "executionBlockers": sorted(set(blockers)),
                "checks": checks,
            }
        )

    if int(dict(labs_vector_index_review.get("counts") or {}).get("reviewReadyRows") or 0) != len(planned_records):
        source_blockers.append("review_ready_count_does_not_match_planned_upsert_records")
    private_path_leak_rows = 1 if _contains_private_path(executor_rows) or _contains_private_path(planned_records) else 0
    if private_path_leak_rows:
        source_blockers.append("private_path_leak")
    source_blockers = sorted(set(source_blockers))
    by_status = Counter(row["executionStatus"] for row in executor_rows)
    by_type = Counter(row["candidateType"] for row in executor_rows)
    blocked_rows = sum(1 for row in executor_rows if row["executionStatus"] != EXECUTOR_DRY_RUN_STATUS_READY)
    counts = {
        "sourceReviewRows": len(review_rows),
        "sourceReviewReadyRows": int(dict(labs_vector_index_review.get("counts") or {}).get("reviewReadyRows") or 0),
        "sourcePlannedVectorDocumentRows": int(
            dict(labs_vector_index_dry_run.get("counts") or {}).get("plannedVectorDocumentRows") or 0
        ),
        "executorDryRunRows": len(executor_rows),
        "plannedVectorUpsertRows": len(planned_records),
        "plannedLabsNamespaceRows": 1 if planned_records else 0,
        "embeddingInputRows": len(planned_records),
        "candidateStoreWriteRows": 0,
        "embeddingCallRows": 0,
        "embeddingVectorWriteRows": 0,
        "vectorIndexWriteRows": 0,
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "blockedRows": blocked_rows,
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(source_blockers),
        "byCandidateType": dict(by_type),
        "byExecutionStatus": dict(by_status),
    }
    status = "ready" if not source_blockers and not blocked_rows and planned_records else "blocked"
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "sourceLabsVectorIndexReview": _source_review_summary(
            labs_vector_index_review,
            report_ref=source_labs_vector_index_review_ref,
        ),
        "sourceLabsVectorIndexDryRun": _source_dry_run_summary(
            labs_vector_index_dry_run,
            report_ref=source_labs_vector_index_dry_run_ref,
        ),
        "input": {
            "sourceReviewReportRef": normalize_text(source_labs_vector_index_review_ref),
            "sourceDryRunReportRef": normalize_text(source_labs_vector_index_dry_run_ref),
            "labsNamespace": LABS_NAMESPACE,
            "plannedVectorIndexRef": PLANNED_LABS_VECTOR_INDEX_REF,
        },
        "scope": _scope(len(planned_records)),
        "policy": _policy(),
        "method": {
            "name": "labs_visual_retrieval_hint_vector_index_apply_executor_dry_run_v1",
            "description": (
                "Previews labs vector upsert records from reviewed visual retrieval-hint vector documents "
                "without embedding calls or vector-index writes."
            ),
            "limitations": [
                "No embedding model is called in this dry-run.",
                "No vector DB, lexical DB, runtime route, or evidence store is mutated.",
                "Passing this dry-run requires a later labs apply review before any explicit labs vector write.",
            ],
        },
        "counts": counts,
        "gate": {
            "readyForLabsVectorIndexApplyReview": status == "ready",
            "candidateStoreWriteAllowed": False,
            "embeddingCallsAllowed": False,
            "vectorIndexingAllowed": False,
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "productionIndexingAllowed": False,
            "schemaViolations": source_blockers,
        },
        "executorDryRunRowsDetail": executor_rows,
        "plannedVectorUpsertRecords": planned_records,
        "warnings": [
            "This apply-executor dry-run does not call embeddings or write vectors.",
            "Visual retrieval hints remain labs-only, non-evidence, and not answer-visible.",
        ],
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Limited Visual Retrieval Hint Candidate Store Labs Vector Index Apply-Executor Dry-Run",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- executorDryRunRows: `{counts.get('executorDryRunRows')}`",
        f"- plannedVectorUpsertRows: `{counts.get('plannedVectorUpsertRows')}`",
        f"- plannedLabsNamespaceRows: `{counts.get('plannedLabsNamespaceRows')}`",
        f"- embeddingInputRows: `{counts.get('embeddingInputRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- embeddingCallRows: `{counts.get('embeddingCallRows')}`",
        f"- embeddingVectorWriteRows: `{counts.get('embeddingVectorWriteRows')}`",
        f"- vectorIndexWriteRows: `{counts.get('vectorIndexWriteRows')}`",
        f"- indexMutationRows: `{counts.get('indexMutationRows')}`",
        f"- runtimeVisibleRows: `{counts.get('runtimeVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        "",
        "## Candidate Types",
        "",
    ]
    for candidate_type, count in sorted(dict(counts.get("byCandidateType") or {}).items()):
        lines.append(f"- `{candidate_type}`: `{count}`")
    lines.extend(["", "## Execution Status", ""])
    for execution_status, count in sorted(dict(counts.get("byExecutionStatus") or {}).items()):
        lines.append(f"- `{execution_status}`: `{count}`")
    return "\n".join(lines).rstrip() + "\n"


def write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown_report(report), encoding="utf-8")
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "EXECUTOR_DRY_RUN_STATUS_BLOCKED_MISSING_VECTOR_DOCUMENT",
    "EXECUTOR_DRY_RUN_STATUS_BLOCKED_NON_READY_REVIEW_ROW",
    "EXECUTOR_DRY_RUN_STATUS_BLOCKED_POLICY_VIOLATION",
    "EXECUTOR_DRY_RUN_STATUS_BLOCKED_VECTOR_DOCUMENT_CONTRACT",
    "EXECUTOR_DRY_RUN_STATUS_READY",
    "LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID",
    "build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run",
    "load_json",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run",
]
