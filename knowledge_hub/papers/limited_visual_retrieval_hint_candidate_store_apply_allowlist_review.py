"""Report-only allowlist/holdout review for visual retrieval-hint apply design.

This helper consumes a limited apply-design report that may be blocked because
some rows failed the search gate. It splits rows into an allowlist for a future
apply-executor dry-run and a holdout set for later review. It does not write
candidate-store JSONL, index vectors, promote evidence, or expose visual
derived text at answer runtime.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_design import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_DESIGN_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_design import PLANNED_STORE_REF


LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_ALLOWLIST_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-allowlist-review.v1"
)
LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_ALLOWLIST_REVIEW_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-allowlist-review-row.v1"
)

READY_DECISION = "ready_for_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run"
BLOCKED_DECISION = "blocked"
NEXT_RECOMMENDED_TRANCHE = "limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run"
TOLERATED_SOURCE_BLOCKERS = {"search_eval_has_regressions"}
HOLDOUT_ROW_BLOCKERS = {"search_eval_gate_not_passed"}

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


def _source_apply_design_report_row(apply_design_report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(apply_design_report.get("counts") or {})
    return {
        "schema": normalize_text(apply_design_report.get("schema")),
        "status": normalize_text(apply_design_report.get("status")),
        "decision": normalize_text(apply_design_report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "sourceBlockers": [
            normalize_text(item)
            for item in list(apply_design_report.get("sourceBlockers") or [])
            if normalize_text(item)
        ],
        "applyDesignRows": int(counts.get("applyDesignRows") or 0),
        "limitedApplyDesignCandidateRows": int(counts.get("limitedApplyDesignCandidateRows") or 0),
        "plannedSeparateApplyWriteRows": int(counts.get("plannedSeparateApplyWriteRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "candidateStoreWriteRows": int(counts.get("candidateStoreWriteRows") or 0),
    }


def _scope(row_count: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "reviewRows": int(row_count),
        "candidateStoreWriteRows": 0,
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
        "candidateStoreApplyExecutorRows": 0,
    }


def _policy() -> dict[str, Any]:
    return {
        "plannedStoreRef": PLANNED_STORE_REF,
        "reviewKind": "limited_apply_allowlist_holdout_review_only",
        "actualStoreWrite": False,
        "candidateStoreWriteAuthorizedByThisReport": False,
        "allowlistRowsMayProceedToApplyExecutorDryRun": True,
        "holdoutRowsExcludedFromApplyExecutorDryRun": True,
        "separateExplicitApplyRequiredForStoreMutation": True,
        "requiresSeparateIndexingGate": True,
        "allowedUse": "retrieval_hint_only",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "runtimeVisible": False,
        "indexEligible": False,
    }


def _split_blocker_reason(value: Any) -> list[str]:
    return [item for item in normalize_text(value).split(";") if item]


def _base_checks(row: dict[str, Any]) -> dict[str, bool]:
    apply_plan = dict(row.get("applyPlan") or {})
    policy = dict(row.get("policy") or {})
    return {
        "sourceReportDoesNotAllowApply": apply_plan.get("applyAllowedByThisReport") is False,
        "sourceReportDidNotWriteStore": apply_plan.get("candidateStoreWrite") is False,
        "requiresSeparateApplyExecutor": apply_plan.get("requiresSeparateApplyExecutor") is True,
        "requiresSeparateIndexingGate": apply_plan.get("requiresSeparateIndexingGate") is True,
        "dryRunMatched": bool(normalize_text(row.get("sourceDryRunRowId"))),
        "hasIdempotencyKey": bool(normalize_text(row.get("idempotencyKey"))),
        "hasPlannedJsonlRecordHash": bool(normalize_text(row.get("plannedJsonlRecordSha256"))),
        "policyRetrievalHintOnly": policy.get("allowedUse") == "retrieval_hint_only",
        "strictEvidenceFalse": policy.get("strictEvidence") is False,
        "citationGradeFalse": policy.get("citationGrade") is False,
        "answerableWithoutTextEvidenceFalse": policy.get("answerableWithoutTextEvidence") is False,
        "runtimeVisibleFalse": policy.get("runtimeVisible") is False,
        "indexEligibleFalse": policy.get("indexEligible") is False,
        "noPrivatePathLeak": not _contains_private_path(row),
    }


def _checks(row: dict[str, Any]) -> dict[str, bool]:
    search = dict(row.get("searchEvalSummary") or {})
    apply_plan = dict(row.get("applyPlan") or {})
    checks = _base_checks(row)
    checks.update(
        {
            "sourceApplyDesignCandidate": apply_plan.get("limitedApplyDesignCandidate") is True,
            "sourceWouldWriteOnlyOnSeparateApply": apply_plan.get("wouldWriteOnSeparateExplicitApply") is True,
            "searchGatePassed": search.get("passesLimitedApplySearchGate") is True,
            "noSearchRegression": int(search.get("regressedQueryRows") or 0) == 0,
            "noSourceBlocker": not normalize_text(row.get("blockerReason")),
        }
    )
    return checks


def _review_status(row: dict[str, Any], checks: dict[str, bool]) -> tuple[str, str]:
    if all(checks.values()):
        return "allowlisted_for_apply_executor_dry_run", ""
    blockers = _split_blocker_reason(row.get("blockerReason"))
    base_checks = _base_checks(row)
    search = dict(row.get("searchEvalSummary") or {})
    apply_plan = dict(row.get("applyPlan") or {})
    is_search_holdout = (
        blockers
        and set(blockers).issubset(HOLDOUT_ROW_BLOCKERS)
        and all(base_checks.values())
        and apply_plan.get("limitedApplyDesignCandidate") is False
        and apply_plan.get("wouldWriteOnSeparateExplicitApply") is False
        and search.get("passesLimitedApplySearchGate") is False
    )
    if is_search_holdout:
        return "holdout_pending_search_gate_review", ";".join(dict.fromkeys(blockers))
    failed = [name for name, passed in checks.items() if not passed]
    return "blocked", ";".join(failed)


def _review_row(row: dict[str, Any]) -> dict[str, Any]:
    checks = _checks(row)
    review_status, blocker_reason = _review_status(row, checks)
    allowlisted = review_status == "allowlisted_for_apply_executor_dry_run"
    holdout = review_status == "holdout_pending_search_gate_review"
    basis = "|".join(
        [
            normalize_text(row.get("applyDesignRowId")),
            normalize_text(row.get("hintCandidateId")),
            normalize_text(row.get("plannedJsonlRecordSha256")),
            review_status,
        ]
    )
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_ALLOWLIST_REVIEW_ROW_SCHEMA_ID,
        "reviewRowId": "limited-visual-retrieval-hint-apply-allowlist-review:" + _short_hash(basis),
        "sourceApplyDesignRowId": normalize_text(row.get("applyDesignRowId")),
        "hintCandidateId": normalize_text(row.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(row.get("sourceCandidateId")),
        "paperId": normalize_text(row.get("paperId")),
        "paperRef": normalize_text(row.get("paperRef")),
        "sourceContentHash": normalize_text(row.get("sourceContentHash")),
        "page": int(row.get("page") or 0),
        "bbox": list(row.get("bbox") or []),
        "candidateType": normalize_text(row.get("candidateType")),
        "plannedStoreRef": normalize_text(row.get("plannedStoreRef")),
        "sourceDryRunReportRef": normalize_text(row.get("sourceDryRunReportRef")),
        "sourceDryRunRowId": normalize_text(row.get("sourceDryRunRowId")),
        "idempotencyKey": normalize_text(row.get("idempotencyKey")),
        "plannedJsonlRecordSha256": normalize_text(row.get("plannedJsonlRecordSha256")),
        "reviewStatus": review_status,
        "checks": checks,
        "searchEvalSummary": dict(row.get("searchEvalSummary") or {}),
        "reviewPlan": {
            "applyExecutorDryRunCandidate": allowlisted,
            "holdoutFromApplyExecutorDryRun": holdout,
            "candidateStoreWrite": False,
            "applyAllowedByThisReport": False,
            "requiresSeparateApplyExecutor": True,
            "requiresSeparateIndexingGate": True,
        },
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
        },
        "blockerReason": blocker_reason,
    }


def _type_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[normalize_text(row.get("candidateType"))].append(row)
    output = []
    for candidate_type in sorted(grouped):
        items = grouped[candidate_type]
        output.append(
            {
                "candidateType": candidate_type,
                "reviewRows": len(items),
                "allowlistRows": sum(
                    1 for row in items if row.get("reviewStatus") == "allowlisted_for_apply_executor_dry_run"
                ),
                "holdoutRows": sum(
                    1 for row in items if row.get("reviewStatus") == "holdout_pending_search_gate_review"
                ),
                "blockedRows": sum(1 for row in items if row.get("reviewStatus") == "blocked"),
                "improvedQueryRows": sum(
                    int(row.get("searchEvalSummary", {}).get("improvedQueryRows") or 0) for row in items
                ),
                "strongLiftQueryRows": sum(
                    int(row.get("searchEvalSummary", {}).get("strongLiftQueryRows") or 0) for row in items
                ),
            }
        )
    return output


def _counts(
    rows: list[dict[str, Any]],
    *,
    source_apply_design_rows: int,
    private_path_leak_rows: int,
) -> dict[str, Any]:
    hint_ids = [normalize_text(row.get("hintCandidateId")) for row in rows]
    duplicate_hint_ids = {item for item, count in Counter(hint_ids).items() if item and count > 1}
    blocked_rows = sum(1 for row in rows if row.get("reviewStatus") == "blocked")
    allowlist_rows = sum(
        1 for row in rows if row.get("reviewStatus") == "allowlisted_for_apply_executor_dry_run"
    )
    holdout_rows = sum(1 for row in rows if row.get("reviewStatus") == "holdout_pending_search_gate_review")
    return {
        "sourceApplyDesignRows": int(source_apply_design_rows),
        "reviewRows": len(rows),
        "allowlistRows": allowlist_rows,
        "holdoutRows": holdout_rows,
        "applyExecutorDryRunCandidateRows": allowlist_rows,
        "plannedSeparateApplyWriteRows": allowlist_rows,
        "candidateStoreWriteRows": 0,
        "duplicateHintCandidateIdRows": len(duplicate_hint_ids),
        "blockedRows": int(blocked_rows + len(duplicate_hint_ids)),
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": 0,
    }


def build_limited_visual_retrieval_hint_candidate_store_apply_allowlist_review(
    apply_design_report: dict[str, Any],
    *,
    source_apply_design_report_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_blockers: list[str] = []
    counts = dict(apply_design_report.get("counts") or {})
    source_report_blockers = {
        normalize_text(item)
        for item in list(apply_design_report.get("sourceBlockers") or [])
        if normalize_text(item)
    }
    disallowed_source_blockers = sorted(source_report_blockers - TOLERATED_SOURCE_BLOCKERS)

    if apply_design_report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_DESIGN_SCHEMA_ID:
        source_blockers.append("invalid_apply_design_schema")
    if apply_design_report.get("status") not in {"ready", "blocked"}:
        source_blockers.append("apply_design_invalid_status")
    if apply_design_report.get("decision") not in {
        "ready_for_limited_visual_retrieval_hint_candidate_store_apply_review",
        "blocked",
    }:
        source_blockers.append("apply_design_invalid_decision")
    for blocker in disallowed_source_blockers:
        source_blockers.append(f"untolerated_source_blocker:{blocker}")
    if int(counts.get("candidateStoreWriteRows") or 0) != 0:
        source_blockers.append("apply_design_has_store_writes")
    if int(counts.get("privatePathLeakRows") or 0) != 0:
        source_blockers.append("apply_design_has_private_path_leaks")

    source_rows = [
        row
        for row in list(apply_design_report.get("applyDesignRowsDetail") or [])
        if isinstance(row, dict)
    ]
    rows = [_review_row(row) for row in source_rows]
    private_path_leak_rows = 1 if _contains_private_path(rows) else 0
    report_counts = _counts(
        rows,
        source_apply_design_rows=int(counts.get("applyDesignRows") or len(source_rows)),
        private_path_leak_rows=private_path_leak_rows,
    )
    allowlist_ids = [
        normalize_text(row.get("hintCandidateId"))
        for row in rows
        if row.get("reviewStatus") == "allowlisted_for_apply_executor_dry_run"
    ]
    holdout_ids = [
        normalize_text(row.get("hintCandidateId"))
        for row in rows
        if row.get("reviewStatus") == "holdout_pending_search_gate_review"
    ]
    report: dict[str, Any] = {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_ALLOWLIST_REVIEW_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "sourceApplyDesignReport": _source_apply_design_report_row(
            apply_design_report,
            report_ref=source_apply_design_report_ref,
        ),
        "toleratedSourceBlockers": sorted(source_report_blockers & TOLERATED_SOURCE_BLOCKERS),
        "scope": _scope(len(rows)),
        "policy": _policy(),
        "counts": report_counts,
        "typeSummary": _type_summary(rows),
        "allowlistHintCandidateIds": allowlist_ids,
        "holdoutHintCandidateIds": holdout_ids,
        "reviewRowsDetail": rows,
        "warnings": [
            "Only allowlisted rows may feed a future apply-executor dry-run.",
            "Holdout rows remain excluded until a later search-gate review, recrop, or reannotation tranche.",
            "This report does not authorize or perform candidate-store writes, vector indexing, or runtime exposure.",
        ],
    }
    if (
        source_blockers
        or private_path_leak_rows
        or report_counts["blockedRows"]
        or report_counts["allowlistRows"] == 0
        or not rows
    ):
        report["status"] = "blocked"
        report["decision"] = BLOCKED_DECISION
        report["sourceBlockers"] = source_blockers
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    lines = [
        "# Limited Visual Retrieval Hint Candidate Store Apply Allowlist Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- reviewRows: `{counts.get('reviewRows')}`",
        f"- allowlistRows: `{counts.get('allowlistRows')}`",
        f"- holdoutRows: `{counts.get('holdoutRows')}`",
        f"- applyExecutorDryRunCandidateRows: `{counts.get('applyExecutorDryRunCandidateRows')}`",
        f"- plannedSeparateApplyWriteRows: `{counts.get('plannedSeparateApplyWriteRows')}`",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- candidateStoreWriteRows: `{scope.get('candidateStoreWriteRows')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- operationalSearchIndexQueryRows: `{scope.get('operationalSearchIndexQueryRows')}`",
        f"- answerGenerationRows: `{scope.get('answerGenerationRows')}`",
        f"- databaseMutationRows: `{scope.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{scope.get('indexMutationRows')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        "",
        "## Type Summary",
        "",
        "| type | rows | allowlist | holdout | blocked | improved queries | strong-lift queries |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("typeSummary", []):
        lines.append(
            "| {candidateType} | {rows} | {allowlist} | {holdout} | {blocked} | {improved} | {strong} |".format(
                candidateType=row.get("candidateType"),
                rows=row.get("reviewRows"),
                allowlist=row.get("allowlistRows"),
                holdout=row.get("holdoutRows"),
                blocked=row.get("blockedRows"),
                improved=row.get("improvedQueryRows"),
                strong=row.get("strongLiftQueryRows"),
            )
        )
    lines.extend(
        [
            "",
            "## Sample Review Rows",
            "",
            "| # | paperId | type | status | improved queries | strong lift | sourceCandidateId |",
            "|---:|---|---|---|---:|---:|---|",
        ]
    )
    for index, row in enumerate(list(report.get("reviewRowsDetail") or [])[:24], start=1):
        summary = dict(row.get("searchEvalSummary") or {})
        lines.append(
            "| {index} | {paperId} | {candidateType} | {status} | {improved} | {strong} | `{sourceId}` |".format(
                index=index,
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                status=row.get("reviewStatus"),
                improved=summary.get("improvedQueryRows"),
                strong=summary.get("strongLiftQueryRows"),
                sourceId=row.get("sourceCandidateId"),
            )
        )
    holdout_ids = list(report.get("holdoutHintCandidateIds") or [])
    if holdout_ids:
        lines.extend(["", "## Holdouts", ""])
        for item in holdout_ids[:24]:
            lines.append(f"- `{item}`")
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_limited_visual_retrieval_hint_candidate_store_apply_allowlist_review(
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
    "LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_ALLOWLIST_REVIEW_SCHEMA_ID",
    "build_limited_visual_retrieval_hint_candidate_store_apply_allowlist_review",
    "load_json",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_candidate_store_apply_allowlist_review",
]
