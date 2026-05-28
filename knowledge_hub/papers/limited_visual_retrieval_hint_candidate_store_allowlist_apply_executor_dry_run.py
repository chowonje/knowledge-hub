"""Dry-run a future allowlisted visual retrieval-hint candidate-store apply.

This helper consumes the Phase 8 allowlist/holdout review plus the original
candidate-store dry-run reports. It reconstructs the future JSONL apply plan
for allowlisted rows only, verifies each row against the original dry-run
preview/hash/idempotency data, and writes no candidate store or index.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_allowlist_review import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_ALLOWLIST_REVIEW_SCHEMA_ID,
    READY_DECISION as ALLOWLIST_REVIEW_READY_DECISION,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_design import (
    PLANNED_STORE_REF,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_ROW_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_dry_run import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_dry_run import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
)


LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_ALLOWLIST_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-"
    "allowlist-apply-executor-dry-run.v1"
)
LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_ALLOWLIST_APPLY_EXECUTOR_DRY_RUN_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-"
    "allowlist-apply-executor-dry-run-row.v1"
)

READY_DECISION = "ready_for_limited_visual_retrieval_hint_candidate_store_apply_executor_review"
BLOCKED_DECISION = "blocked"
NEXT_RECOMMENDED_TRANCHE = "limited_visual_retrieval_hint_candidate_store_apply_executor_review"

ALLOWED_SOURCE_DRY_RUN_SCHEMAS = {
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
}

ALLOWLIST_STATUS = "allowlisted_for_apply_executor_dry_run"
HOLDOUT_STATUS = "holdout_pending_search_gate_review"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitized_report_ref(path: Path, *, project_root: Path | None = None) -> str:
    resolved = path.expanduser()
    if project_root is not None:
        try:
            rel = resolved.resolve().relative_to(project_root.resolve())
            return rel.as_posix()
        except Exception:
            pass
    return f"input_reports/{resolved.name}"


def _short_hash(value: str, *, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _source_allowlist_review_report_row(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "reviewRows": int(counts.get("reviewRows") or 0),
        "allowlistRows": int(counts.get("allowlistRows") or 0),
        "holdoutRows": int(counts.get("holdoutRows") or 0),
        "applyExecutorDryRunCandidateRows": int(counts.get("applyExecutorDryRunCandidateRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "candidateStoreWriteRows": int(counts.get("candidateStoreWriteRows") or 0),
        "privatePathLeakRows": int(counts.get("privatePathLeakRows") or 0),
        "schemaViolationCount": int(counts.get("schemaViolationCount") or 0),
    }


def _source_dry_run_report_row(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "dryRunRows": int(counts.get("dryRunRows") or 0),
        "plannedWriteRows": int(counts.get("plannedWriteRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "candidateStoreWriteRows": int(counts.get("candidateStoreWriteRows") or 0),
        "privatePathLeakRows": int(counts.get("privatePathLeakRows") or 0),
        "schemaViolationCount": int(counts.get("schemaViolationCount") or 0),
    }


def _scope(row_count: int, *, source_dry_run_rows: int, excluded_holdout_rows: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "sourceDryRunRows": int(source_dry_run_rows),
        "executorDryRunRows": int(row_count),
        "excludedHoldoutRows": int(excluded_holdout_rows),
        "plannedCandidateStoreRows": int(row_count),
        "candidateStoreWriteRows": 0,
        "candidateStoreApplyRows": 0,
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
        "cropWriteRows": 0,
        "pageImageWriteRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
    }


def _dry_run_policy() -> dict[str, Any]:
    return {
        "plannedStoreRef": PLANNED_STORE_REF,
        "dryRunKind": "allowlist_apply_executor_preview_only",
        "actualStoreWrite": False,
        "candidateStoreWriteAuthorizedByThisReport": False,
        "candidateStoreApplyRows": 0,
        "jsonlWritePreviewOnly": True,
        "requiresSeparateExplicitApplyGateForStoreMutation": True,
        "requiresSeparateIndexingGate": True,
        "runtimeVisibilityAfterDryRun": "not_runtime_visible",
        "indexingAfterDryRun": "not_indexed",
        "allowedUse": "retrieval_hint_only",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "runtimeVisible": False,
        "indexEligible": False,
    }


def _dry_run_rows(dry_run_reports: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_ref, report in dry_run_reports:
        for row in list(report.get("dryRunRowsDetail") or []):
            if not isinstance(row, dict):
                continue
            copied = dict(row)
            copied["_sourceDryRunReportRef"] = report_ref
            rows.append(copied)
    return rows


def _source_report_blockers(
    allowlist_review_report: dict[str, Any],
    dry_run_reports: list[tuple[str, dict[str, Any]]],
) -> list[str]:
    blockers: list[str] = []
    allowlist_counts = dict(allowlist_review_report.get("counts") or {})
    if allowlist_review_report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_ALLOWLIST_REVIEW_SCHEMA_ID:
        blockers.append("invalid_allowlist_review_schema")
    if allowlist_review_report.get("status") != "ready":
        blockers.append("allowlist_review_not_ready")
    if allowlist_review_report.get("decision") != ALLOWLIST_REVIEW_READY_DECISION:
        blockers.append("allowlist_review_invalid_decision")
    if int(allowlist_counts.get("candidateStoreWriteRows") or 0) != 0:
        blockers.append("allowlist_review_has_store_writes")
    if int(allowlist_counts.get("blockedRows") or 0) != 0:
        blockers.append("allowlist_review_has_blocked_rows")
    if int(allowlist_counts.get("privatePathLeakRows") or 0) != 0:
        blockers.append("allowlist_review_has_private_path_leaks")
    if int(allowlist_counts.get("schemaViolationCount") or 0) != 0:
        blockers.append("allowlist_review_has_schema_violations")

    for report_ref, report in dry_run_reports:
        counts = dict(report.get("counts") or {})
        if normalize_text(report.get("schema")) not in ALLOWED_SOURCE_DRY_RUN_SCHEMAS:
            blockers.append(f"invalid_dry_run_schema:{report_ref}")
        if report.get("status") != "ready":
            blockers.append(f"dry_run_report_not_ready:{report_ref}")
        if int(counts.get("candidateStoreWriteRows") or 0) != 0:
            blockers.append(f"dry_run_has_store_writes:{report_ref}")
        if int(counts.get("blockedRows") or 0) != 0:
            blockers.append(f"dry_run_has_blocked_rows:{report_ref}")
        if int(counts.get("privatePathLeakRows") or 0) != 0:
            blockers.append(f"dry_run_has_private_path_leaks:{report_ref}")
        if int(counts.get("schemaViolationCount") or 0) != 0:
            blockers.append(f"dry_run_has_schema_violations:{report_ref}")
    return blockers


def _source_indexes(dry_rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_report_and_row: dict[tuple[str, str], dict[str, Any]] = {}
    by_row_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_hint_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dry_rows:
        report_ref = normalize_text(row.get("_sourceDryRunReportRef"))
        row_id = normalize_text(row.get("dryRunRowId"))
        hint_id = normalize_text(row.get("hintCandidateId"))
        if report_ref and row_id:
            by_report_and_row[(report_ref, row_id)] = row
        if row_id:
            by_row_id[row_id].append(row)
        if hint_id:
            by_hint_id[hint_id].append(row)
    return by_report_and_row, by_row_id, by_hint_id


def _match_source_dry_run_row(
    review_row: dict[str, Any],
    *,
    by_report_and_row: dict[tuple[str, str], dict[str, Any]],
    by_row_id: dict[str, list[dict[str, Any]]],
    by_hint_id: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    report_ref = normalize_text(review_row.get("sourceDryRunReportRef"))
    row_id = normalize_text(review_row.get("sourceDryRunRowId"))
    hint_id = normalize_text(review_row.get("hintCandidateId"))
    matched = by_report_and_row.get((report_ref, row_id))
    if matched:
        return matched
    row_matches = by_row_id.get(row_id) or []
    if len(row_matches) == 1:
        return row_matches[0]
    hint_matches = by_hint_id.get(hint_id) or []
    if len(hint_matches) == 1:
        return hint_matches[0]
    return None


def _record_policy(record: dict[str, Any]) -> dict[str, Any]:
    return dict(record.get("policy") or {})


def _recomputed_record_hash(record: dict[str, Any]) -> str:
    if not record:
        return ""
    return _sha256_text(_canonical_json(record))


def _record_checks(
    *,
    review_row: dict[str, Any],
    source_dry_row: dict[str, Any] | None,
    record: dict[str, Any],
    recomputed_hash: str,
) -> dict[str, bool]:
    review_plan = dict(review_row.get("reviewPlan") or {})
    review_policy = dict(review_row.get("policy") or {})
    dry_result = dict((source_dry_row or {}).get("dryRunResult") or {})
    record_policy = _record_policy(record)
    source_ref = normalize_text((source_dry_row or {}).get("_sourceDryRunReportRef"))
    return {
        "reviewAllowlisted": review_row.get("reviewStatus") == ALLOWLIST_STATUS,
        "reviewPlanExecutorCandidate": review_plan.get("applyExecutorDryRunCandidate") is True,
        "reviewPlanDidNotWriteStore": review_plan.get("candidateStoreWrite") is False,
        "reviewPlanDidNotAuthorizeApply": review_plan.get("applyAllowedByThisReport") is False,
        "reviewRequiresSeparateExecutor": review_plan.get("requiresSeparateApplyExecutor") is True,
        "reviewRequiresSeparateIndexingGate": review_plan.get("requiresSeparateIndexingGate") is True,
        "sourceDryRunMatched": source_dry_row is not None,
        "sourceDryRunReportRefMatches": source_ref == normalize_text(review_row.get("sourceDryRunReportRef")),
        "sourceDryRunRowIdMatches": normalize_text((source_dry_row or {}).get("dryRunRowId")) == normalize_text(review_row.get("sourceDryRunRowId")),
        "sourceDryRunWouldWriteOnApply": dry_result.get("wouldWriteOnApply") is True,
        "sourceDryRunActualStoreWriteFalse": dry_result.get("actualStoreWrite") is False,
        "sourceDryRunJsonlSerializable": dry_result.get("jsonlSerializable") is True,
        "sourceDryRunPolicyCompliant": dry_result.get("policyCompliant") is True,
        "sourceDryRunNotIndexEligible": dry_result.get("indexEligible") is False,
        "sourceDryRunNotRuntimeVisible": dry_result.get("runtimeVisible") is False,
        "sourceDryRunStrictEvidenceFalse": dry_result.get("strictEvidence") is False,
        "sourceDryRunCitationGradeFalse": dry_result.get("citationGrade") is False,
        "sourceDryRunAnswerableWithoutTextEvidenceFalse": dry_result.get("answerableWithoutTextEvidence") is False,
        "hintCandidateIdMatches": normalize_text(record.get("hintCandidateId")) == normalize_text(review_row.get("hintCandidateId")),
        "sourceCandidateIdMatches": normalize_text(record.get("sourceCandidateId")) == normalize_text(review_row.get("sourceCandidateId")),
        "paperIdMatches": normalize_text(record.get("paperId")) == normalize_text(review_row.get("paperId")),
        "paperRefMatches": normalize_text(record.get("paperRef")) == normalize_text(review_row.get("paperRef")),
        "sourceContentHashMatches": normalize_text(record.get("sourceContentHash")) == normalize_text(review_row.get("sourceContentHash")),
        "pageMatches": int(record.get("page") or 0) == int(review_row.get("page") or 0),
        "bboxMatches": list(record.get("bbox") or []) == list(review_row.get("bbox") or []),
        "candidateTypeMatches": normalize_text(record.get("candidateType")) == normalize_text(review_row.get("candidateType")),
        "plannedStoreRefMatches": normalize_text((source_dry_row or {}).get("plannedStoreRef")) == PLANNED_STORE_REF,
        "idempotencyKeyMatches": normalize_text((source_dry_row or {}).get("idempotencyKey")) == normalize_text(review_row.get("idempotencyKey")),
        "plannedJsonlRecordHashMatches": normalize_text((source_dry_row or {}).get("plannedJsonlRecordSha256")) == normalize_text(review_row.get("plannedJsonlRecordSha256")),
        "recomputedJsonlRecordHashMatches": recomputed_hash == normalize_text(review_row.get("plannedJsonlRecordSha256")),
        "recordSchemaMatches": normalize_text(record.get("schema")) == VISUAL_RETRIEVAL_HINT_CANDIDATE_ROW_SCHEMA_ID,
        "recordHasDerivedText": bool(normalize_text(record.get("derivedTextForRetrieval"))),
        "recordHasVisibleText": bool(normalize_text(record.get("visibleText"))),
        "recordHasRetrievalKeywords": bool(list(record.get("retrievalKeywords") or [])),
        "reviewPolicyRetrievalHintOnly": review_policy.get("allowedUse") == "retrieval_hint_only",
        "recordPolicyRetrievalHintOnly": record_policy.get("allowedUse") == "retrieval_hint_only",
        "strictEvidenceFalse": review_policy.get("strictEvidence") is False and record_policy.get("strictEvidence") is False,
        "citationGradeFalse": review_policy.get("citationGrade") is False and record_policy.get("citationGrade") is False,
        "answerableWithoutTextEvidenceFalse": (
            review_policy.get("answerableWithoutTextEvidence") is False
            and record_policy.get("answerableWithoutTextEvidence") is False
        ),
        "runtimeVisibleFalse": review_policy.get("runtimeVisible") is False and record_policy.get("runtimeVisible") is False,
        "indexEligibleFalse": review_policy.get("indexEligible") is False and record_policy.get("indexEligible") is False,
        "answerabilityGateBypassFalse": record_policy.get("answerabilityGateBypassAllowed") is False,
        "noPrivatePathLeak": not _contains_private_path(review_row) and not _contains_private_path(source_dry_row or {}) and not _contains_private_path(record),
    }


def _executor_dry_run_row(
    review_row: dict[str, Any],
    *,
    source_dry_row: dict[str, Any] | None,
) -> dict[str, Any]:
    record = dict((source_dry_row or {}).get("plannedJsonlRecordPreview") or {})
    recomputed_hash = _recomputed_record_hash(record)
    checks = _record_checks(
        review_row=review_row,
        source_dry_row=source_dry_row,
        record=record,
        recomputed_hash=recomputed_hash,
    )
    failed_checks = [name for name, passed in checks.items() if not passed]
    basis = "|".join(
        [
            normalize_text(review_row.get("reviewRowId")),
            normalize_text(review_row.get("hintCandidateId")),
            normalize_text(review_row.get("plannedJsonlRecordSha256")),
            recomputed_hash,
        ]
    )
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_ALLOWLIST_APPLY_EXECUTOR_DRY_RUN_ROW_SCHEMA_ID,
        "executorDryRunRowId": "limited-visual-retrieval-hint-allowlist-apply-executor-dry-run:" + _short_hash(basis),
        "sourceReviewRowId": normalize_text(review_row.get("reviewRowId")),
        "sourceApplyDesignRowId": normalize_text(review_row.get("sourceApplyDesignRowId")),
        "sourceDryRunReportRef": normalize_text(review_row.get("sourceDryRunReportRef")),
        "sourceDryRunRowId": normalize_text(review_row.get("sourceDryRunRowId")),
        "hintCandidateId": normalize_text(review_row.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(review_row.get("sourceCandidateId")),
        "paperId": normalize_text(review_row.get("paperId")),
        "paperRef": normalize_text(review_row.get("paperRef")),
        "sourceContentHash": normalize_text(review_row.get("sourceContentHash")),
        "page": int(review_row.get("page") or 0),
        "bbox": list(review_row.get("bbox") or []),
        "candidateType": normalize_text(review_row.get("candidateType")),
        "plannedStoreRef": PLANNED_STORE_REF,
        "idempotencyKey": normalize_text(review_row.get("idempotencyKey")),
        "plannedJsonlRecordSha256": normalize_text(review_row.get("plannedJsonlRecordSha256")),
        "recomputedPlannedJsonlRecordSha256": recomputed_hash,
        "plannedJsonlRecordPreview": record,
        "checks": checks,
        "executorDryRunResult": {
            "wouldWriteOnSeparateExplicitApply": not failed_checks,
            "actualStoreWrite": False,
            "candidateStoreWrite": False,
            "jsonlSerializable": checks.get("sourceDryRunJsonlSerializable") is True,
            "policyCompliant": checks.get("sourceDryRunPolicyCompliant") is True and not failed_checks,
            "applyExecutorPreviewOnly": True,
            "indexEligible": False,
            "runtimeVisible": False,
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
        },
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
            "answerabilityGateBypassAllowed": False,
        },
        "provenance": {
            "sourceAllowlistReviewRowId": normalize_text(review_row.get("reviewRowId")),
            "sourceApplyDesignRowId": normalize_text(review_row.get("sourceApplyDesignRowId")),
            "sourceDryRunReportRef": normalize_text(review_row.get("sourceDryRunReportRef")),
            "sourceDryRunRowId": normalize_text(review_row.get("sourceDryRunRowId")),
            "sourceCandidateId": normalize_text(review_row.get("sourceCandidateId")),
            "sourceContentHash": normalize_text(review_row.get("sourceContentHash")),
            "page": int(review_row.get("page") or 0),
            "bbox": list(review_row.get("bbox") or []),
            "extractionMethod": "limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run_v1",
        },
        "blockerReason": ";".join(failed_checks),
    }


def _type_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[normalize_text(row.get("candidateType"))].append(row)
    summary = []
    for candidate_type in sorted(grouped):
        items = grouped[candidate_type]
        summary.append(
            {
                "candidateType": candidate_type,
                "executorDryRunRows": len(items),
                "plannedWriteRows": sum(
                    1 for row in items if dict(row.get("executorDryRunResult") or {}).get("wouldWriteOnSeparateExplicitApply")
                ),
                "jsonlSerializableRows": sum(
                    1 for row in items if dict(row.get("executorDryRunResult") or {}).get("jsonlSerializable")
                ),
                "blockedRows": sum(1 for row in items if row.get("blockerReason")),
            }
        )
    return summary


def _counts(
    rows: list[dict[str, Any]],
    *,
    source_allowlist_review_rows: int,
    source_allowlist_rows: int,
    source_holdout_rows: int,
    source_dry_run_report_rows: int,
    source_dry_run_rows: int,
    excluded_holdout_rows: int,
    private_path_leak_rows: int,
) -> dict[str, int]:
    hint_ids = [normalize_text(row.get("hintCandidateId")) for row in rows]
    source_ids = [normalize_text(row.get("sourceCandidateId")) for row in rows]
    duplicate_hint_ids = {item for item, count in Counter(hint_ids).items() if item and count > 1}
    duplicate_source_ids = {item for item, count in Counter(source_ids).items() if item and count > 1}
    blocked_rows = sum(1 for row in rows if row.get("blockerReason"))
    planned_write_rows = sum(
        1 for row in rows if dict(row.get("executorDryRunResult") or {}).get("wouldWriteOnSeparateExplicitApply")
    )
    return {
        "sourceAllowlistReviewRows": int(source_allowlist_review_rows),
        "sourceAllowlistRows": int(source_allowlist_rows),
        "sourceHoldoutRows": int(source_holdout_rows),
        "sourceDryRunReportRows": int(source_dry_run_report_rows),
        "sourceDryRunRows": int(source_dry_run_rows),
        "executorDryRunRows": len(rows),
        "plannedWriteRows": int(planned_write_rows),
        "candidateStoreWriteRows": 0,
        "candidateStoreApplyRows": 0,
        "jsonlSerializableRows": sum(
            1 for row in rows if dict(row.get("executorDryRunResult") or {}).get("jsonlSerializable")
        ),
        "excludedHoldoutRows": int(excluded_holdout_rows),
        "duplicateHintCandidateIdRows": len(duplicate_hint_ids),
        "duplicateSourceCandidateIdRows": len(duplicate_source_ids),
        "blockedRows": int(blocked_rows + len(duplicate_hint_ids) + len(duplicate_source_ids)),
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": 0,
    }


def build_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run(
    allowlist_review_report: dict[str, Any],
    dry_run_reports: list[tuple[str, dict[str, Any]]],
    *,
    source_allowlist_review_report_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    review_rows = [
        row for row in list(allowlist_review_report.get("reviewRowsDetail") or []) if isinstance(row, dict)
    ]
    allowlist_rows = [
        row
        for row in review_rows
        if row.get("reviewStatus") == ALLOWLIST_STATUS
        and dict(row.get("reviewPlan") or {}).get("applyExecutorDryRunCandidate") is True
    ]
    holdout_rows = [row for row in review_rows if row.get("reviewStatus") == HOLDOUT_STATUS]
    dry_rows = _dry_run_rows(dry_run_reports)
    by_report_and_row, by_row_id, by_hint_id = _source_indexes(dry_rows)

    rows = [
        _executor_dry_run_row(
            review_row,
            source_dry_row=_match_source_dry_run_row(
                review_row,
                by_report_and_row=by_report_and_row,
                by_row_id=by_row_id,
                by_hint_id=by_hint_id,
            ),
        )
        for review_row in allowlist_rows
    ]

    source_blockers = _source_report_blockers(allowlist_review_report, dry_run_reports)
    private_path_leak_rows = 1 if _contains_private_path(rows) else 0
    allowlist_counts = dict(allowlist_review_report.get("counts") or {})
    source_dry_run_reports = [
        _source_dry_run_report_row(report, report_ref=report_ref) for report_ref, report in dry_run_reports
    ]
    report_counts = _counts(
        rows,
        source_allowlist_review_rows=int(allowlist_counts.get("reviewRows") or len(review_rows)),
        source_allowlist_rows=int(allowlist_counts.get("allowlistRows") or len(allowlist_rows)),
        source_holdout_rows=int(allowlist_counts.get("holdoutRows") or len(holdout_rows)),
        source_dry_run_report_rows=len(dry_run_reports),
        source_dry_run_rows=len(dry_rows),
        excluded_holdout_rows=len(holdout_rows),
        private_path_leak_rows=private_path_leak_rows,
    )
    report: dict[str, Any] = {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_ALLOWLIST_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "sourceAllowlistReviewReport": _source_allowlist_review_report_row(
            allowlist_review_report,
            report_ref=source_allowlist_review_report_ref,
        ),
        "sourceDryRunReports": source_dry_run_reports,
        "scope": _scope(
            len(rows),
            source_dry_run_rows=len(dry_rows),
            excluded_holdout_rows=len(holdout_rows),
        ),
        "dryRunPolicy": _dry_run_policy(),
        "counts": report_counts,
        "typeSummary": _type_summary(rows),
        "executorDryRunRowsDetail": rows,
        "excludedHoldoutHintCandidateIds": [
            normalize_text(row.get("hintCandidateId")) for row in holdout_rows if normalize_text(row.get("hintCandidateId"))
        ],
        "warnings": [
            "This dry-run reconstructs allowlisted future JSONL records but writes no candidate store.",
            "The five holdout rows remain excluded from this executor dry-run.",
            "A separate explicit apply gate is still required before any store mutation, and a later gate is required before indexing.",
        ],
    }
    if (
        source_blockers
        or private_path_leak_rows
        or report_counts["blockedRows"]
        or report_counts["executorDryRunRows"] == 0
        or report_counts["executorDryRunRows"] != report_counts["sourceAllowlistRows"]
        or report_counts["plannedWriteRows"] != report_counts["executorDryRunRows"]
    ):
        report["status"] = "blocked"
        report["decision"] = BLOCKED_DECISION
        report["sourceBlockers"] = source_blockers
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    source = dict(report.get("sourceAllowlistReviewReport") or {})
    lines = [
        "# Limited Visual Retrieval Hint Candidate Store Allowlist Apply Executor Dry Run",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- sourceAllowlistReviewReport: `{source.get('reportRef')}`",
        f"- sourceAllowlistRows: `{counts.get('sourceAllowlistRows')}`",
        f"- excludedHoldoutRows: `{counts.get('excludedHoldoutRows')}`",
        f"- executorDryRunRows: `{counts.get('executorDryRunRows')}`",
        f"- plannedWriteRows: `{counts.get('plannedWriteRows')}`",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- candidateStoreApplyRows: `{counts.get('candidateStoreApplyRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- candidateStoreWriteRows: `{scope.get('candidateStoreWriteRows')}`",
        f"- candidateStoreApplyRows: `{scope.get('candidateStoreApplyRows')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- operationalSearchIndexQueryRows: `{scope.get('operationalSearchIndexQueryRows')}`",
        f"- answerGenerationRows: `{scope.get('answerGenerationRows')}`",
        f"- databaseMutationRows: `{scope.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{scope.get('indexMutationRows')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        f"- vaultScanRows: `{scope.get('vaultScanRows')}`",
        f"- externalDownloadRows: `{scope.get('externalDownloadRows')}`",
        "",
        "## Type Summary",
        "",
        "| type | executor dry-run rows | planned writes | jsonl serializable | blocked |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report.get("typeSummary", []):
        lines.append(
            "| {candidateType} | {rows} | {planned} | {serializable} | {blocked} |".format(
                candidateType=row.get("candidateType"),
                rows=row.get("executorDryRunRows"),
                planned=row.get("plannedWriteRows"),
                serializable=row.get("jsonlSerializableRows"),
                blocked=row.get("blockedRows"),
            )
        )
    lines.extend(
        [
            "",
            "## Sample Executor Dry Run Rows",
            "",
            "| # | paperId | type | page | would write on separate apply | sourceCandidateId |",
            "|---:|---|---|---:|---|---|",
        ]
    )
    for index, row in enumerate(list(report.get("executorDryRunRowsDetail") or [])[:32], start=1):
        result = dict(row.get("executorDryRunResult") or {})
        lines.append(
            "| {index} | {paperId} | {candidateType} | {page} | `{wouldWrite}` | `{sourceCandidateId}` |".format(
                index=index,
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                page=row.get("page"),
                wouldWrite=result.get("wouldWriteOnSeparateExplicitApply"),
                sourceCandidateId=row.get("sourceCandidateId"),
            )
        )
    holdout_ids = list(report.get("excludedHoldoutHintCandidateIds") or [])
    if holdout_ids:
        lines.extend(["", "## Excluded Holdouts", ""])
        for item in holdout_ids[:32]:
            lines.append(f"- `{item}`")
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run(
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
    "LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_ALLOWLIST_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID",
    "LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_ALLOWLIST_APPLY_EXECUTOR_DRY_RUN_ROW_SCHEMA_ID",
    "build_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run",
    "load_json",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run",
]
