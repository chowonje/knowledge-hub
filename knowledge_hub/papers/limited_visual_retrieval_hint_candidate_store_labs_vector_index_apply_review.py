"""Review gate for labs visual retrieval-hint vector-index apply dry-runs.

The review validates planned labs vector upsert records before any explicit labs
apply executor is allowed. It is report-only: no embedding calls, vector writes,
runtime exposure, or evidence promotion are performed.
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
    EXECUTOR_DRY_RUN_STATUS_READY,
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
    PLANNED_LABS_VECTOR_INDEX_REF,
    READY_DECISION as APPLY_EXECUTOR_DRY_RUN_READY_DECISION,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run import (
    LABS_NAMESPACE,
)


LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-labs-vector-index-apply-review.v1"
)

APPLY_REVIEW_STATUS_READY = "review_ready_labs_vector_upsert"
APPLY_REVIEW_STATUS_BLOCKED_NON_READY_EXECUTOR_ROW = "blocked_non_ready_executor_row"
APPLY_REVIEW_STATUS_BLOCKED_MISSING_UPSERT_RECORD = "blocked_missing_upsert_record"
APPLY_REVIEW_STATUS_BLOCKED_UPSERT_CONTRACT = "blocked_upsert_record_contract"
APPLY_REVIEW_STATUS_BLOCKED_POLICY = "blocked_policy_violation"

READY_DECISION = "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor"
BLOCKED_DECISION = "blocked"
NEXT_RECOMMENDED_TRANCHE = "limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor"


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


def _source_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "executorDryRunRows": int(counts.get("executorDryRunRows") or 0),
        "plannedVectorUpsertRows": int(counts.get("plannedVectorUpsertRows") or 0),
        "plannedLabsNamespaceRows": int(counts.get("plannedLabsNamespaceRows") or 0),
        "embeddingInputRows": int(counts.get("embeddingInputRows") or 0),
        "candidateStoreWriteRows": int(counts.get("candidateStoreWriteRows") or 0),
        "embeddingCallRows": int(counts.get("embeddingCallRows") or 0),
        "embeddingVectorWriteRows": int(counts.get("embeddingVectorWriteRows") or 0),
        "vectorIndexWriteRows": int(counts.get("vectorIndexWriteRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "privatePathLeakRows": int(counts.get("privatePathLeakRows") or 0),
        "schemaViolationCount": int(counts.get("schemaViolationCount") or 0),
    }


def _source_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID:
        blockers.append("invalid_labs_vector_index_apply_executor_dry_run_schema")
    if report.get("status") != "ready":
        blockers.append("labs_vector_index_apply_executor_dry_run_not_ready")
    if report.get("decision") != APPLY_EXECUTOR_DRY_RUN_READY_DECISION:
        blockers.append("labs_vector_index_apply_executor_dry_run_invalid_decision")
    for field_name in (
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
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
    ):
        if int(counts.get(field_name) or 0) != 0:
            blockers.append(f"labs_vector_index_apply_executor_dry_run_has_{field_name}")
    if int(counts.get("plannedVectorUpsertRows") or 0) <= 0:
        blockers.append("labs_vector_index_apply_executor_dry_run_has_no_upsert_rows")
    if int(counts.get("executorDryRunRows") or 0) != int(counts.get("plannedVectorUpsertRows") or 0):
        blockers.append("labs_vector_index_apply_executor_dry_run_count_mismatch")
    return sorted(set(blockers))


def _policy_flags_ok(policy: dict[str, Any]) -> bool:
    return (
        normalize_text(policy.get("allowedUse")) == "retrieval_hint_only"
        and policy.get("strictEvidence") is False
        and policy.get("citationGrade") is False
        and policy.get("answerableWithoutTextEvidence") is False
        and policy.get("runtimeVisible") is False
        and policy.get("indexEligible") is False
        and policy.get("productionIndexEligible") is False
        and policy.get("labsOnly") is True
    )


def _metadata_policy_ok(metadata: dict[str, Any]) -> bool:
    return (
        normalize_text(metadata.get("allowedUse")) == "retrieval_hint_only"
        and metadata.get("strictEvidence") is False
        and metadata.get("citationGrade") is False
        and metadata.get("answerableWithoutTextEvidence") is False
        and metadata.get("runtimeVisible") is False
        and metadata.get("indexEligible") is False
    )


def _scope(review_rows: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "reviewRows": int(review_rows),
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
        "reviewOnly": True,
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


def _review_row(
    *,
    index: int,
    executor_row: dict[str, Any],
    upsert_record: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict((upsert_record or {}).get("metadata") or {})
    policy = dict((upsert_record or {}).get("policy") or {})
    execution_plan = dict((upsert_record or {}).get("executionPlan") or {})
    checks = {
        "sourceExecutorRowReady": normalize_text(executor_row.get("executionStatus")) == EXECUTOR_DRY_RUN_STATUS_READY,
        "sourceExecutorNoBlockers": not list(executor_row.get("executionBlockers") or []),
        "sourceWouldCallEmbedderOnlyOnExplicitApply": executor_row.get("wouldCallEmbedderOnSeparateExplicitApply") is True,
        "sourceWouldWriteVectorIndexOnlyOnExplicitApply": executor_row.get("wouldWriteVectorIndexOnSeparateExplicitApply")
        is True,
        "sourceActualEmbeddingCallFalse": executor_row.get("actualEmbeddingCall") is False,
        "sourceActualEmbeddingVectorWriteFalse": executor_row.get("actualEmbeddingVectorWrite") is False,
        "sourceActualVectorIndexWriteFalse": executor_row.get("actualVectorIndexWrite") is False,
        "sourceNotIndexEligible": executor_row.get("indexEligible") is False,
        "sourceNotRuntimeVisible": executor_row.get("runtimeVisible") is False,
        "upsertRecordPresent": bool(upsert_record),
        "upsertRecordSchemaMatches": normalize_text((upsert_record or {}).get("schema"))
        == "knowledge-hub.paper.visual-retrieval-hint-labs-vector-upsert-record.v1",
        "namespaceMatchesLabs": normalize_text((upsert_record or {}).get("namespace")) == LABS_NAMESPACE,
        "plannedVectorIndexRefMatches": normalize_text((upsert_record or {}).get("plannedVectorIndexRef"))
        == PLANNED_LABS_VECTOR_INDEX_REF,
        "vectorDocumentIdMatches": normalize_text((upsert_record or {}).get("vectorDocumentId"))
        == normalize_text(executor_row.get("vectorDocumentId")),
        "hintCandidateIdMatches": normalize_text((upsert_record or {}).get("hintCandidateId"))
        == normalize_text(executor_row.get("hintCandidateId")),
        "sourceCandidateIdMatches": normalize_text((upsert_record or {}).get("sourceCandidateId"))
        == normalize_text(executor_row.get("sourceCandidateId")),
        "sourceContentHashMatches": normalize_text((upsert_record or {}).get("sourceContentHash"))
        == normalize_text(executor_row.get("sourceContentHash")),
        "pageMatches": int((upsert_record or {}).get("page") or 0) == int(executor_row.get("page") or 0),
        "bboxMatches": _bbox((upsert_record or {}).get("bbox")) == _bbox(executor_row.get("bbox")),
        "candidateTypeMatches": normalize_text((upsert_record or {}).get("candidateType"))
        == normalize_text(executor_row.get("candidateType")),
        "documentTextPresent": bool(normalize_text((upsert_record or {}).get("documentText"))),
        "embeddingTextPresent": bool(normalize_text((upsert_record or {}).get("embeddingText"))),
        "documentTextHashMatches": normalize_text((upsert_record or {}).get("documentTextHash"))
        == normalize_text(executor_row.get("documentTextHash"))
        == _sha256_text(str((upsert_record or {}).get("documentText") or "")),
        "embeddingTextHashMatches": normalize_text((upsert_record or {}).get("embeddingTextHash"))
        == normalize_text(executor_row.get("embeddingTextHash"))
        == _sha256_text(str((upsert_record or {}).get("embeddingText") or "")),
        "plannedUpsertRecordHashMatches": bool(upsert_record)
        and _record_hash(upsert_record) == normalize_text(executor_row.get("plannedVectorUpsertRecordSha256")),
        "metadataSchemaMatches": normalize_text(metadata.get("retrieval_unit_schema"))
        == "visual_retrieval_hint_vector_document.v1",
        "metadataNamespaceMatches": normalize_text(metadata.get("namespace")) == LABS_NAMESPACE,
        "metadataPolicyRetrievalHintOnly": _metadata_policy_ok(metadata),
        "upsertPolicyRetrievalHintOnly": _policy_flags_ok(policy),
        "embeddingVectorAbsent": execution_plan.get("embeddingVectorPresent") is False
        and "embeddingVector" not in (upsert_record or {})
        and "vector" not in (upsert_record or {}),
        "executionPlanDefersEmbedderToExplicitApply": execution_plan.get("embeddingProviderRef")
        == "deferred_to_separate_explicit_labs_apply",
        "executionPlanWouldCallEmbedderOnExplicitApply": execution_plan.get("wouldCallEmbedderOnSeparateExplicitApply")
        is True,
        "executionPlanWouldWriteVectorIndexOnExplicitApply": execution_plan.get("wouldWriteVectorIndexOnSeparateExplicitApply")
        is True,
        "executionPlanActualEmbeddingCallFalse": execution_plan.get("actualEmbeddingCall") is False,
        "executionPlanActualEmbeddingVectorWriteFalse": execution_plan.get("actualEmbeddingVectorWrite") is False,
        "executionPlanActualVectorIndexWriteFalse": execution_plan.get("actualVectorIndexWrite") is False,
        "upsertRecordNotPrivatePathLeaking": bool(upsert_record) and not _contains_private_path(upsert_record),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    policy_blockers = {
        "metadataPolicyRetrievalHintOnly",
        "upsertPolicyRetrievalHintOnly",
        "upsertRecordNotPrivatePathLeaking",
        "sourceNotIndexEligible",
        "sourceNotRuntimeVisible",
    }
    if normalize_text(executor_row.get("executionStatus")) != EXECUTOR_DRY_RUN_STATUS_READY:
        status = APPLY_REVIEW_STATUS_BLOCKED_NON_READY_EXECUTOR_ROW
    elif not upsert_record:
        status = APPLY_REVIEW_STATUS_BLOCKED_MISSING_UPSERT_RECORD
    elif any(blocker in policy_blockers for blocker in blockers):
        status = APPLY_REVIEW_STATUS_BLOCKED_POLICY
    elif blockers:
        status = APPLY_REVIEW_STATUS_BLOCKED_UPSERT_CONTRACT
    else:
        status = APPLY_REVIEW_STATUS_READY
    return {
        "applyReviewRowId": f"limited-visual-retrieval-hint-candidate-store-labs-vector-index-apply-review:{index:04d}",
        "sourceExecutorDryRunRowId": normalize_text(executor_row.get("executorDryRunRowId")),
        "sourceReviewRowId": normalize_text(executor_row.get("sourceReviewRowId")),
        "sourcePlanRowId": normalize_text(executor_row.get("sourcePlanRowId")),
        "vectorDocumentId": normalize_text(executor_row.get("vectorDocumentId")),
        "namespace": normalize_text((upsert_record or {}).get("namespace") or executor_row.get("namespace")),
        "hintCandidateId": normalize_text(executor_row.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(executor_row.get("sourceCandidateId")),
        "paperId": normalize_text(executor_row.get("paperId")),
        "paperRef": normalize_text(executor_row.get("paperRef")),
        "sourceContentHash": normalize_text(executor_row.get("sourceContentHash")),
        "page": int(executor_row.get("page") or 0),
        "bbox": _bbox(executor_row.get("bbox")),
        "candidateType": normalize_text(executor_row.get("candidateType")),
        "documentTextHash": normalize_text(executor_row.get("documentTextHash")),
        "embeddingTextHash": normalize_text(executor_row.get("embeddingTextHash")),
        "plannedVectorUpsertRecordSha256": normalize_text(executor_row.get("plannedVectorUpsertRecordSha256")),
        "plannedVectorIndexRef": normalize_text(executor_row.get("plannedVectorIndexRef")),
        "labsApplyExecutorCandidate": status == APPLY_REVIEW_STATUS_READY,
        "indexEligible": False,
        "runtimeVisible": False,
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "applyReviewStatus": status,
        "applyReviewBlockers": sorted(set(blockers)),
        "checks": checks,
    }


def review_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply(
    *,
    labs_vector_index_apply_executor_dry_run: dict[str, Any],
    source_labs_vector_index_apply_executor_dry_run_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_blockers = _source_blockers(labs_vector_index_apply_executor_dry_run)
    executor_rows = [
        dict(row)
        for row in labs_vector_index_apply_executor_dry_run.get("executorDryRunRowsDetail") or []
        if isinstance(row, dict)
    ]
    records_by_id = {
        normalize_text(row.get("vectorDocumentId")): dict(row)
        for row in labs_vector_index_apply_executor_dry_run.get("plannedVectorUpsertRecords") or []
        if isinstance(row, dict)
    }
    review_rows = [
        _review_row(
            index=index,
            executor_row=executor_row,
            upsert_record=records_by_id.get(normalize_text(executor_row.get("vectorDocumentId"))),
        )
        for index, executor_row in enumerate(executor_rows, start=1)
    ]
    planned_record_count = len(records_by_id)
    if int(dict(labs_vector_index_apply_executor_dry_run.get("counts") or {}).get("plannedVectorUpsertRows") or 0) != planned_record_count:
        source_blockers.append("planned_vector_upsert_record_count_mismatch")
    private_path_leak_rows = 1 if _contains_private_path(review_rows) else 0
    if private_path_leak_rows:
        source_blockers.append("private_path_leak")
    source_blockers = sorted(set(source_blockers))
    by_status = Counter(row["applyReviewStatus"] for row in review_rows)
    by_type = Counter(row["candidateType"] for row in review_rows)
    blocked_rows = sum(1 for row in review_rows if row["applyReviewStatus"] != APPLY_REVIEW_STATUS_READY)
    counts = {
        "sourceExecutorDryRunRows": len(executor_rows),
        "sourcePlannedVectorUpsertRows": int(
            dict(labs_vector_index_apply_executor_dry_run.get("counts") or {}).get("plannedVectorUpsertRows") or 0
        ),
        "reviewRows": len(review_rows),
        "reviewReadyRows": by_status.get(APPLY_REVIEW_STATUS_READY, 0),
        "labsApplyExecutorCandidateRows": by_status.get(APPLY_REVIEW_STATUS_READY, 0),
        "plannedLabsNamespaceRows": int(
            dict(labs_vector_index_apply_executor_dry_run.get("counts") or {}).get("plannedLabsNamespaceRows") or 0
        ),
        "embeddingInputRows": int(
            dict(labs_vector_index_apply_executor_dry_run.get("counts") or {}).get("embeddingInputRows") or 0
        ),
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
        "byApplyReviewStatus": dict(by_status),
    }
    status = "ready" if not source_blockers and not blocked_rows and review_rows else "blocked"
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "sourceLabsVectorIndexApplyExecutorDryRun": _source_summary(
            labs_vector_index_apply_executor_dry_run,
            report_ref=source_labs_vector_index_apply_executor_dry_run_ref,
        ),
        "input": {
            "sourceReportRef": normalize_text(source_labs_vector_index_apply_executor_dry_run_ref),
            "labsNamespace": LABS_NAMESPACE,
            "plannedVectorIndexRef": PLANNED_LABS_VECTOR_INDEX_REF,
        },
        "scope": _scope(len(review_rows)),
        "policy": _policy(),
        "method": {
            "name": "labs_visual_retrieval_hint_vector_index_apply_review_v1",
            "description": (
                "Reviews planned labs vector upsert records for source-row readiness, policy quarantine, "
                "hash integrity, and no-write guarantees before a later explicit labs apply executor."
            ),
            "limitations": [
                "No embedding model is called by this review.",
                "No vector DB, lexical DB, runtime route, or evidence store is mutated.",
                "Passing this review permits only designing a separate explicit labs apply executor, not production indexing.",
            ],
        },
        "counts": counts,
        "gate": {
            "readyForLabsVectorIndexApplyExecutor": status == "ready",
            "candidateStoreWriteAllowed": False,
            "embeddingCallsAllowed": False,
            "embeddingVectorWriteAllowed": False,
            "vectorIndexingAllowed": False,
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "productionIndexingAllowed": False,
            "schemaViolations": source_blockers,
        },
        "applyReviewRowsDetail": review_rows,
        "warnings": [
            "This apply review does not call embeddings or write vectors.",
            "Visual retrieval hints remain labs-only, non-evidence, and not answer-visible.",
        ],
    }


build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_review = (
    review_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply
)


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Limited Visual Retrieval Hint Candidate Store Labs Vector Index Apply Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- reviewRows: `{counts.get('reviewRows')}`",
        f"- reviewReadyRows: `{counts.get('reviewReadyRows')}`",
        f"- labsApplyExecutorCandidateRows: `{counts.get('labsApplyExecutorCandidateRows')}`",
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
    lines.extend(["", "## Apply Review Status", ""])
    for apply_status, count in sorted(dict(counts.get("byApplyReviewStatus") or {}).items()):
        lines.append(f"- `{apply_status}`: `{count}`")
    return "\n".join(lines).rstrip() + "\n"


def write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_review(
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
    "APPLY_REVIEW_STATUS_BLOCKED_MISSING_UPSERT_RECORD",
    "APPLY_REVIEW_STATUS_BLOCKED_NON_READY_EXECUTOR_ROW",
    "APPLY_REVIEW_STATUS_BLOCKED_POLICY",
    "APPLY_REVIEW_STATUS_BLOCKED_UPSERT_CONTRACT",
    "APPLY_REVIEW_STATUS_READY",
    "LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_REVIEW_SCHEMA_ID",
    "build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_review",
    "load_json",
    "review_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_review",
]
