"""Review gate for visual retrieval-hint candidate-store expansion previews.

This report-only helper consumes the expansion dry-run report and checks whether
the planned JSONL preview rows are ready for human/product review. It does not
approve rows, write the candidate store, index vectors, promote evidence, or
expose hints at answer runtime.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_design import (
    PLANNED_STORE_REF,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_dry_run import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
)


VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-review.v1"
)
VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-review-row.v1"
)

READY_DECISION = "ready_for_visual_retrieval_hint_candidate_store_expansion_human_product_review"
NEXT_RECOMMENDED_TRANCHE = (
    "visual_retrieval_hint_candidate_store_expansion_human_product_decision_record"
)

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


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _snippet(value: Any, *, limit: int = 240) -> str:
    text = normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _scope(row_count: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "sourceDryRunRows": int(row_count),
        "reviewRowsProjected": int(row_count),
        "humanApprovalRows": 0,
        "candidateStoreWriteRows": 0,
        "vectorIndexing": False,
        "indexEligibleRows": 0,
        "strictEvidencePromotionRows": 0,
        "runtimeAnswerVisibleExposureRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "answerabilityGateBypassRows": 0,
        "cropWriteRows": 0,
        "pageImageWriteRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
    }


def _review_policy() -> dict[str, Any]:
    return {
        "reviewType": "human_product_review_gate",
        "plannedStoreRef": PLANNED_STORE_REF,
        "actualApproval": False,
        "applyAllowedByThisReport": False,
        "requiresHumanProductDecisionRecord": True,
        "allowedHumanDecisions": [
            "approve_store_candidate_only",
            "hold_pending_more_context",
            "reject_visual_hint_candidate",
            "request_recrop_or_reannotation",
        ],
        "reviewQuestion": (
            "Should this retrieval-hint-only visual annotation be allowed into a future quarantined "
            "candidate store, while remaining unindexed, non-runtime-visible, and non-evidence?"
        ),
    }


def _row_checks(row: dict[str, Any]) -> dict[str, bool]:
    dry_run = dict(row.get("dryRunResult") or {})
    preview = dict(row.get("plannedJsonlRecordPreview") or {})
    policy = dict(preview.get("policy") or {})
    return {
        "hasStableHintCandidateId": bool(normalize_text(row.get("hintCandidateId"))),
        "hasIdempotencyKey": bool(normalize_text(row.get("idempotencyKey"))),
        "hasPlannedJsonlRecordSha256": normalize_text(row.get("plannedJsonlRecordSha256")).startswith("sha256:"),
        "jsonlSerializable": dry_run.get("jsonlSerializable") is True,
        "wouldWriteOnlyOnFutureApply": dry_run.get("wouldWriteOnApply") is True
        and dry_run.get("actualStoreWrite") is False,
        "allowedUseRetrievalHintOnly": policy.get("allowedUse") == "retrieval_hint_only",
        "strictEvidenceDisabled": policy.get("strictEvidence") is False and dry_run.get("strictEvidence") is False,
        "citationGradeDisabled": policy.get("citationGrade") is False and dry_run.get("citationGrade") is False,
        "answerableWithoutTextEvidenceDisabled": (
            policy.get("answerableWithoutTextEvidence") is False
            and dry_run.get("answerableWithoutTextEvidence") is False
        ),
        "runtimeVisibleDisabled": policy.get("runtimeVisible") is False and dry_run.get("runtimeVisible") is False,
        "indexEligibleDisabled": policy.get("indexEligible") is False and dry_run.get("indexEligible") is False,
        "noPrivatePathLeak": not _contains_private_path(row),
        "noSourceBlocker": not normalize_text(row.get("blockerReason")),
    }


def _blocker_reasons(checks: dict[str, bool]) -> list[str]:
    return [name for name, passed in checks.items() if not passed]


def _review_row(row: dict[str, Any], *, source_dry_run_report_ref: str) -> dict[str, Any]:
    preview = dict(row.get("plannedJsonlRecordPreview") or {})
    checks = _row_checks(row)
    blockers = _blocker_reasons(checks)
    review_row_id = "visual-retrieval-hint-expansion-review:" + _short_hash(
        "|".join(
            [
                normalize_text(row.get("hintCandidateId")),
                normalize_text(row.get("plannedJsonlRecordSha256")),
                normalize_text(source_dry_run_report_ref),
            ]
        )
    )
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_ROW_SCHEMA_ID,
        "reviewRowId": review_row_id,
        "hintCandidateId": normalize_text(row.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(row.get("sourceCandidateId")),
        "paperId": normalize_text(row.get("paperId")),
        "paperRef": normalize_text(row.get("paperRef")),
        "sourceContentHash": normalize_text(row.get("sourceContentHash")),
        "page": int(row.get("page") or 0),
        "bbox": list(row.get("bbox") or []),
        "candidateType": normalize_text(row.get("candidateType")),
        "plannedStoreRef": normalize_text(row.get("plannedStoreRef")) or PLANNED_STORE_REF,
        "idempotencyKey": normalize_text(row.get("idempotencyKey")),
        "plannedJsonlRecordSha256": normalize_text(row.get("plannedJsonlRecordSha256")),
        "derivedTextForRetrievalSnippet": _snippet(preview.get("derivedTextForRetrieval")),
        "visibleTextSnippet": _snippet(preview.get("visibleText")),
        "retrievalKeywordCount": len(list(preview.get("retrievalKeywords") or [])),
        "reviewStatus": "ready_for_human_product_review" if not blockers else "blocked",
        "defaultHumanDecision": "hold_pending_human_product_review",
        "allowedHumanDecisions": list(_review_policy()["allowedHumanDecisions"]),
        "checks": checks,
        "policySummary": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
            "applyAllowedByThisReport": False,
        },
        "provenance": {
            "sourceDryRunReportSchema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
            "sourceDryRunReportRef": normalize_text(source_dry_run_report_ref),
            "hintCandidateId": normalize_text(row.get("hintCandidateId")),
            "sourceCandidateId": normalize_text(row.get("sourceCandidateId")),
            "sourceContentHash": normalize_text(row.get("sourceContentHash")),
            "page": int(row.get("page") or 0),
            "bbox": list(row.get("bbox") or []),
            "extractionMethod": "visual_retrieval_hint_candidate_store_expansion_review_v1",
        },
        "blockerReason": ";".join(blockers),
    }


def _counts(rows: list[dict[str, Any]], *, source_dry_run_rows: int, private_path_leak_rows: int) -> dict[str, int]:
    hint_ids = [normalize_text(row.get("hintCandidateId")) for row in rows]
    source_ids = [normalize_text(row.get("sourceCandidateId")) for row in rows]
    duplicate_hint_ids = {item for item, count in Counter(hint_ids).items() if item and count > 1}
    duplicate_source_ids = {item for item, count in Counter(source_ids).items() if item and count > 1}
    blocked_rows = sum(1 for row in rows if row.get("blockerReason"))
    return {
        "sourceDryRunRows": int(source_dry_run_rows),
        "reviewRows": len(rows),
        "reviewReadyRows": sum(1 for row in rows if row.get("reviewStatus") == "ready_for_human_product_review"),
        "humanDecisionRows": 0,
        "applyReadyRows": 0,
        "candidateStoreWriteRows": 0,
        "policyViolationRows": int(blocked_rows),
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


def build_visual_retrieval_hint_candidate_store_expansion_review(
    dry_run_report: dict[str, Any],
    *,
    source_dry_run_report_ref: str = (
        "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_dry_run.v1.json"
    ),
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_rows = [row for row in list(dry_run_report.get("dryRunRowsDetail") or []) if isinstance(row, dict)]
    rows = [
        _review_row(row, source_dry_run_report_ref=source_dry_run_report_ref)
        for row in source_rows
    ]
    private_path_leak_rows = 1 if _contains_private_path(rows) else 0
    dry_counts = dict(dry_run_report.get("counts") or {})
    report: dict[str, Any] = {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "sourceDryRunReport": {
            "schema": normalize_text(dry_run_report.get("schema")),
            "status": normalize_text(dry_run_report.get("status")),
            "reportRef": normalize_text(source_dry_run_report_ref),
            "dryRunRows": len(source_rows),
            "blockedRows": int(dry_counts.get("blockedRows") or 0),
            "candidateStoreWriteRows": int(dry_counts.get("candidateStoreWriteRows") or 0),
        },
        "scope": _scope(len(rows)),
        "reviewPolicy": _review_policy(),
        "counts": {},
        "reviewRowsDetail": rows,
        "warnings": [
            "This report is a review gate only; it is not human approval.",
            "A later human/product decision record is required before any candidate-store apply design.",
            "Rows remain retrieval-hint-only, unindexed, not runtime-visible, and non-evidence.",
        ],
    }
    report["counts"] = _counts(
        rows,
        source_dry_run_rows=len(source_rows),
        private_path_leak_rows=private_path_leak_rows,
    )
    if (
        dry_run_report.get("schema") != VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID
        or dry_run_report.get("status") != "ready"
        or dry_counts.get("blockedRows")
        or dry_counts.get("candidateStoreWriteRows")
        or not rows
        or report["counts"]["blockedRows"]
        or private_path_leak_rows
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    source = dict(report.get("sourceDryRunReport") or {})
    policy = dict(report.get("reviewPolicy") or {})
    lines = [
        "# Visual Retrieval Hint Candidate Store Expansion Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- sourceDryRunReport: `{source.get('reportRef')}`",
        f"- reviewRows: `{counts.get('reviewRows')}`",
        f"- reviewReadyRows: `{counts.get('reviewReadyRows')}`",
        f"- humanDecisionRows: `{counts.get('humanDecisionRows')}`",
        f"- applyReadyRows: `{counts.get('applyReadyRows')}`",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        "",
        "## Review Boundary",
        "",
        f"- reviewType: `{policy.get('reviewType')}`",
        f"- actualApproval: `{policy.get('actualApproval')}`",
        f"- applyAllowedByThisReport: `{policy.get('applyAllowedByThisReport')}`",
        f"- requiresHumanProductDecisionRecord: `{policy.get('requiresHumanProductDecisionRecord')}`",
        f"- plannedStoreRef: `{policy.get('plannedStoreRef')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- candidateStoreWriteRows: `{scope.get('candidateStoreWriteRows')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- indexMutationRows: `{scope.get('indexMutationRows')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        f"- answerabilityGateBypassRows: `{scope.get('answerabilityGateBypassRows')}`",
        "",
        "## Review Rows",
        "",
        "| # | paperId | type | page | reviewStatus | hintCandidateId | recordSha256 |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for index, row in enumerate(report.get("reviewRowsDetail", []), start=1):
        lines.append(
            "| {index} | {paperId} | {candidateType} | {page} | `{status}` | `{hintCandidateId}` | `{sha}` |".format(
                index=index,
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                page=row.get("page"),
                status=row.get("reviewStatus"),
                hintCandidateId=row.get("hintCandidateId"),
                sha=row.get("plannedJsonlRecordSha256"),
            )
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_visual_retrieval_hint_candidate_store_expansion_review(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown_report(report), encoding="utf-8")
    return {
        "json": str(report_json),
        "markdown": str(report_md),
    }


__all__ = [
    "NEXT_RECOMMENDED_TRANCHE",
    "READY_DECISION",
    "VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_ROW_SCHEMA_ID",
    "VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_SCHEMA_ID",
    "build_visual_retrieval_hint_candidate_store_expansion_review",
    "load_json",
    "render_markdown_report",
    "sanitized_report_ref",
    "write_visual_retrieval_hint_candidate_store_expansion_review",
]
