"""Report-only limited apply design for visual retrieval-hint candidates.

This helper consumes the targeted search eval plus the existing dry-run record
previews and projects the rows that are safe to review for a future limited
candidate-store apply. It does not write the candidate store, index vectors,
promote evidence, or expose derived text at answer runtime.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.visual_retrieval_hint_candidate_store_design import PLANNED_STORE_REF
from knowledge_hub.papers.visual_retrieval_hint_search_eval import (
    VISUAL_RETRIEVAL_HINT_SEARCH_EVAL_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_usefulness_eval import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
)


LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-design.v1"
)
LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_DESIGN_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-design-row.v1"
)

READY_DECISION = "ready_for_limited_visual_retrieval_hint_candidate_store_apply_review"
BLOCKED_DECISION = "blocked"
NEXT_RECOMMENDED_TRANCHE = "limited_visual_retrieval_hint_candidate_store_apply_review"

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


def _valid_dry_run_schema(schema: str) -> bool:
    return schema in {
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID,
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
    }


def _dry_run_rows(dry_run_report: dict[str, Any], *, source_ref: str) -> list[dict[str, Any]]:
    rows = []
    for row in list(dry_run_report.get("dryRunRowsDetail") or []):
        if isinstance(row, dict):
            copied = dict(row)
            copied["_sourceDryRunReportRef"] = source_ref
            rows.append(copied)
    return rows


def _source_search_report_row(search_eval_report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(search_eval_report.get("counts") or {})
    return {
        "schema": normalize_text(search_eval_report.get("schema")),
        "status": normalize_text(search_eval_report.get("status")),
        "decision": normalize_text(search_eval_report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "inputHintRows": int(counts.get("inputHintRows") or 0),
        "queryRows": int(counts.get("queryRows") or 0),
        "augmentedHitAt5Rows": int(counts.get("augmentedHitAt5Rows") or 0),
        "rankRegressedRows": int(counts.get("rankRegressedRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "candidateStoreWriteRows": int(counts.get("candidateStoreWriteRows") or 0),
    }


def _source_dry_run_report_row(dry_run_report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(dry_run_report.get("counts") or {})
    return {
        "schema": normalize_text(dry_run_report.get("schema")),
        "status": normalize_text(dry_run_report.get("status")),
        "reportRef": normalize_text(report_ref),
        "dryRunRows": int(counts.get("dryRunRows") or 0),
        "plannedWriteRows": int(counts.get("plannedWriteRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "candidateStoreWriteRows": int(counts.get("candidateStoreWriteRows") or 0),
    }


def _scope(row_count: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "applyDesignRows": int(row_count),
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
        "designKind": "limited_candidate_store_apply_design_only",
        "actualStoreWrite": False,
        "candidateStoreWriteAuthorizedByThisReport": False,
        "requiresSeparateApplyExecutor": True,
        "requiresSeparateIndexingGate": True,
        "allowedUse": "retrieval_hint_only",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "runtimeVisible": False,
        "indexEligible": False,
    }


def _rank_value(rank: Any) -> int | None:
    return rank if isinstance(rank, int) and rank > 0 else None


def _candidate_search_summaries(search_eval_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in list(search_eval_report.get("queryRowsDetail") or []):
        if isinstance(row, dict):
            grouped[normalize_text(row.get("hintCandidateId"))].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for hint_id, rows in grouped.items():
        text_ranks = [
            _rank_value(dict(row.get("textOnlyResult") or {}).get("rank"))
            for row in rows
        ]
        augmented_ranks = [
            _rank_value(dict(row.get("visualHintAugmentedResult") or {}).get("rank"))
            for row in rows
        ]
        improved_rows = sum(1 for row in rows if dict(row.get("comparison") or {}).get("improved"))
        regressed_rows = sum(1 for row in rows if dict(row.get("comparison") or {}).get("regressed"))
        strong_lift_rows = sum(
            1 for row in rows if dict(row.get("comparison") or {}).get("liftBucket") == "strong_lift"
        )
        augmented_hit5_rows = sum(
            1 for row in rows if dict(row.get("visualHintAugmentedResult") or {}).get("hitAt5")
        )
        blocked_rows = sum(1 for row in rows if normalize_text(row.get("blockerReason")))
        summaries[hint_id] = {
            "sourceSearchEvalQueryRowIds": [
                normalize_text(row.get("queryRowId")) for row in rows if normalize_text(row.get("queryRowId"))
            ],
            "queryRows": len(rows),
            "textOnlyBestRank": min([rank for rank in text_ranks if rank is not None], default=None),
            "visualHintAugmentedBestRank": min(
                [rank for rank in augmented_ranks if rank is not None],
                default=None,
            ),
            "augmentedHitAt5Rows": augmented_hit5_rows,
            "improvedQueryRows": improved_rows,
            "regressedQueryRows": regressed_rows,
            "strongLiftQueryRows": strong_lift_rows,
            "blockedQueryRows": blocked_rows,
            "passesLimitedApplySearchGate": (
                len(rows) > 0
                and augmented_hit5_rows == len(rows)
                and regressed_rows == 0
                and blocked_rows == 0
            ),
        }
    return summaries


def _record_preview(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "derivedTextForRetrievalPreview": normalize_text(record.get("derivedTextForRetrieval"))[:360],
        "visibleTextPreview": normalize_text(record.get("visibleText"))[:240],
        "retrievalKeywords": [
            normalize_text(item)
            for item in list(record.get("retrievalKeywords") or [])[:12]
            if normalize_text(item)
        ],
        "limitations": normalize_text(record.get("limitations")),
    }


def _dry_row_safe(row: dict[str, Any]) -> bool:
    result = dict(row.get("dryRunResult") or {})
    return (
        not normalize_text(row.get("blockerReason"))
        and result.get("wouldWriteOnApply") is True
        and result.get("actualStoreWrite") is False
        and result.get("jsonlSerializable") is True
        and result.get("policyCompliant") is True
        and result.get("indexEligible") is False
        and result.get("runtimeVisible") is False
        and result.get("strictEvidence") is False
        and result.get("citationGrade") is False
        and result.get("answerableWithoutTextEvidence") is False
    )


def _apply_design_row(dry_row: dict[str, Any], search_summary: dict[str, Any] | None) -> dict[str, Any]:
    record = dict(dry_row.get("plannedJsonlRecordPreview") or {})
    hint_id = normalize_text(dry_row.get("hintCandidateId") or record.get("hintCandidateId"))
    source_id = normalize_text(dry_row.get("sourceCandidateId") or record.get("sourceCandidateId"))
    blocker_reasons: list[str] = []
    if not search_summary:
        blocker_reasons.append("missing_search_eval_summary")
        search_summary = {
            "sourceSearchEvalQueryRowIds": [],
            "queryRows": 0,
            "textOnlyBestRank": None,
            "visualHintAugmentedBestRank": None,
            "augmentedHitAt5Rows": 0,
            "improvedQueryRows": 0,
            "regressedQueryRows": 0,
            "strongLiftQueryRows": 0,
            "blockedQueryRows": 0,
            "passesLimitedApplySearchGate": False,
        }
    if not search_summary.get("passesLimitedApplySearchGate"):
        blocker_reasons.append("search_eval_gate_not_passed")
    if not _dry_row_safe(dry_row):
        blocker_reasons.append("dry_run_row_not_apply_candidate_safe")
    if _contains_private_path(dry_row) or _contains_private_path(search_summary):
        blocker_reasons.append("private_path_leak")

    apply_candidate = not blocker_reasons
    basis = "|".join(
        [
            hint_id,
            source_id,
            normalize_text(dry_row.get("plannedJsonlRecordSha256")),
            normalize_text(dry_row.get("_sourceDryRunReportRef")),
        ]
    )
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_DESIGN_ROW_SCHEMA_ID,
        "applyDesignRowId": "limited-visual-retrieval-hint-apply-design:" + _short_hash(basis),
        "hintCandidateId": hint_id,
        "sourceCandidateId": source_id,
        "paperId": normalize_text(dry_row.get("paperId") or record.get("paperId")),
        "paperRef": normalize_text(dry_row.get("paperRef") or record.get("paperRef")),
        "sourceContentHash": normalize_text(dry_row.get("sourceContentHash") or record.get("sourceContentHash")),
        "page": int(dry_row.get("page") or record.get("page") or 0),
        "bbox": list(dry_row.get("bbox") or record.get("bbox") or []),
        "candidateType": normalize_text(dry_row.get("candidateType") or record.get("candidateType")),
        "sourceDryRunReportRef": normalize_text(dry_row.get("_sourceDryRunReportRef")),
        "sourceDryRunRowId": normalize_text(dry_row.get("dryRunRowId")),
        "plannedStoreRef": normalize_text(dry_row.get("plannedStoreRef") or PLANNED_STORE_REF),
        "idempotencyKey": normalize_text(dry_row.get("idempotencyKey")),
        "plannedJsonlRecordSha256": normalize_text(dry_row.get("plannedJsonlRecordSha256")),
        "recordPreview": _record_preview(record),
        "searchEvalSummary": search_summary,
        "applyPlan": {
            "limitedApplyDesignCandidate": apply_candidate,
            "wouldWriteOnSeparateExplicitApply": apply_candidate,
            "applyAllowedByThisReport": False,
            "candidateStoreWrite": False,
            "requiresSeparateApplyExecutor": True,
            "requiresSeparateIndexingGate": True,
            "writeModeIfLaterApproved": "append_jsonl_to_visual_retrieval_hint_candidate_store",
        },
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
        },
        "blockerReason": ";".join(dict.fromkeys(blocker_reasons)),
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
                "applyDesignRows": len(items),
                "limitedApplyDesignCandidateRows": sum(
                    1 for row in items if row.get("applyPlan", {}).get("limitedApplyDesignCandidate")
                ),
                "improvedQueryRows": sum(
                    int(row.get("searchEvalSummary", {}).get("improvedQueryRows") or 0) for row in items
                ),
                "strongLiftQueryRows": sum(
                    int(row.get("searchEvalSummary", {}).get("strongLiftQueryRows") or 0) for row in items
                ),
                "blockedRows": sum(1 for row in items if row.get("blockerReason")),
            }
        )
    return summary


def _counts(
    rows: list[dict[str, Any]],
    *,
    source_search_query_rows: int,
    source_dry_run_reports: int,
    source_dry_run_rows: int,
    private_path_leak_rows: int,
) -> dict[str, Any]:
    hint_ids = [normalize_text(row.get("hintCandidateId")) for row in rows]
    duplicate_hint_ids = {item for item, count in Counter(hint_ids).items() if item and count > 1}
    blocked_rows = sum(1 for row in rows if row.get("blockerReason"))
    return {
        "sourceSearchQueryRows": int(source_search_query_rows),
        "sourceDryRunReportRows": int(source_dry_run_reports),
        "sourceDryRunRows": int(source_dry_run_rows),
        "inputHintRows": len(rows),
        "applyDesignRows": len(rows),
        "limitedApplyDesignCandidateRows": sum(
            1 for row in rows if row.get("applyPlan", {}).get("limitedApplyDesignCandidate")
        ),
        "plannedSeparateApplyWriteRows": sum(
            1 for row in rows if row.get("applyPlan", {}).get("wouldWriteOnSeparateExplicitApply")
        ),
        "candidateStoreWriteRows": 0,
        "jsonlSerializableRows": sum(1 for row in rows if normalize_text(row.get("plannedJsonlRecordSha256"))),
        "searchGatePassedRows": sum(
            1 for row in rows if row.get("searchEvalSummary", {}).get("passesLimitedApplySearchGate")
        ),
        "dryRunMatchedRows": sum(1 for row in rows if normalize_text(row.get("sourceDryRunRowId"))),
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


def build_limited_visual_retrieval_hint_candidate_store_apply_design(
    search_eval_report: dict[str, Any],
    dry_run_reports: list[tuple[str, dict[str, Any]]],
    *,
    source_search_eval_report_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_blockers: list[str] = []
    search_counts = dict(search_eval_report.get("counts") or {})
    if search_eval_report.get("schema") != VISUAL_RETRIEVAL_HINT_SEARCH_EVAL_SCHEMA_ID:
        source_blockers.append("invalid_search_eval_schema")
    if search_eval_report.get("status") != "ready":
        source_blockers.append("search_eval_not_ready")
    if search_eval_report.get("decision") != "ready_for_limited_visual_retrieval_hint_candidate_store_apply_design":
        source_blockers.append("search_eval_not_ready_for_apply_design")
    if int(search_counts.get("blockedRows") or 0) != 0:
        source_blockers.append("search_eval_has_blocked_rows")
    if int(search_counts.get("rankRegressedRows") or 0) != 0:
        source_blockers.append("search_eval_has_regressions")
    if int(search_counts.get("candidateStoreWriteRows") or 0) != 0:
        source_blockers.append("search_eval_has_store_writes")

    dry_rows: list[dict[str, Any]] = []
    source_dry_run_reports = []
    for report_ref, report in dry_run_reports:
        source_dry_run_reports.append(_source_dry_run_report_row(report, report_ref=report_ref))
        if not _valid_dry_run_schema(normalize_text(report.get("schema"))):
            source_blockers.append(f"invalid_dry_run_schema:{report_ref}")
        if report.get("status") != "ready":
            source_blockers.append(f"dry_run_not_ready:{report_ref}")
        if int(dict(report.get("counts") or {}).get("blockedRows") or 0) != 0:
            source_blockers.append(f"dry_run_has_blocked_rows:{report_ref}")
        if int(dict(report.get("counts") or {}).get("candidateStoreWriteRows") or 0) != 0:
            source_blockers.append(f"dry_run_has_store_writes:{report_ref}")
        dry_rows.extend(_dry_run_rows(report, source_ref=report_ref))

    search_summaries = _candidate_search_summaries(search_eval_report)
    selected_dry_rows = [
        row
        for row in dry_rows
        if normalize_text(row.get("hintCandidateId")) in search_summaries
    ]
    rows = [
        _apply_design_row(row, search_summaries.get(normalize_text(row.get("hintCandidateId"))))
        for row in selected_dry_rows
    ]
    private_path_leak_rows = 1 if _contains_private_path(rows) or _contains_private_path(source_dry_run_reports) else 0
    counts = _counts(
        rows,
        source_search_query_rows=int(search_counts.get("queryRows") or 0),
        source_dry_run_reports=len(source_dry_run_reports),
        source_dry_run_rows=len(dry_rows),
        private_path_leak_rows=private_path_leak_rows,
    )
    report: dict[str, Any] = {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_DESIGN_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "sourceSearchEvalReport": _source_search_report_row(
            search_eval_report,
            report_ref=source_search_eval_report_ref,
        ),
        "sourceDryRunReports": source_dry_run_reports,
        "scope": _scope(len(rows)),
        "policy": _policy(),
        "counts": counts,
        "typeSummary": _type_summary(rows),
        "applyDesignRowsDetail": rows,
        "warnings": [
            "This report only designs a future limited candidate-store apply; it does not write the store.",
            "Visual derived text remains retrieval-hint-only and cannot become strict or citation-grade evidence.",
            "Indexing, runtime visibility, and answer generation require separate gates after any future apply.",
        ],
    }
    if (
        source_blockers
        or private_path_leak_rows
        or counts["blockedRows"]
        or counts["limitedApplyDesignCandidateRows"] != counts["inputHintRows"]
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
        "# Limited Visual Retrieval Hint Candidate Store Apply Design",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- inputHintRows: `{counts.get('inputHintRows')}`",
        f"- limitedApplyDesignCandidateRows: `{counts.get('limitedApplyDesignCandidateRows')}`",
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
        "| type | rows | limited candidates | improved queries | strong-lift queries | blocked |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("typeSummary", []):
        lines.append(
            "| {candidateType} | {rows} | {candidates} | {improved} | {strong} | {blocked} |".format(
                candidateType=row.get("candidateType"),
                rows=row.get("applyDesignRows"),
                candidates=row.get("limitedApplyDesignCandidateRows"),
                improved=row.get("improvedQueryRows"),
                strong=row.get("strongLiftQueryRows"),
                blocked=row.get("blockedRows"),
            )
        )
    lines.extend(
        [
            "",
            "## Sample Apply Design Rows",
            "",
            "| # | paperId | type | candidate | improved queries | strong lift | sourceCandidateId |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for index, row in enumerate(list(report.get("applyDesignRowsDetail") or [])[:24], start=1):
        summary = dict(row.get("searchEvalSummary") or {})
        lines.append(
            "| {index} | {paperId} | {candidateType} | {candidate} | {improved} | {strong} | `{sourceId}` |".format(
                index=index,
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                candidate=row.get("applyPlan", {}).get("limitedApplyDesignCandidate"),
                improved=summary.get("improvedQueryRows"),
                strong=summary.get("strongLiftQueryRows"),
                sourceId=row.get("sourceCandidateId"),
            )
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_limited_visual_retrieval_hint_candidate_store_apply_design(
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
    "LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_DESIGN_SCHEMA_ID",
    "build_limited_visual_retrieval_hint_candidate_store_apply_design",
    "load_json",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_candidate_store_apply_design",
]
