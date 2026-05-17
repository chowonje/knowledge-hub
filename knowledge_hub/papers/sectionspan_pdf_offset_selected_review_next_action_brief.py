"""Report-only next-action brief for selected SectionSpan review rows.

The brief makes the manual review gate reproducible. It reads the selected
manual sheet plus validation/decision-record reports and tells the operator
which draft file needs human edits. It does not approve rows, record decisions,
create strict evidence, route parsers, write canonical artifacts, mutate DB
state, reindex, reembed, or change answer behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_NEXT_ACTION_BRIEF_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-next-action-brief.v1"
)
SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_MANUAL_SHEET_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-manual-sheet.v1"
)
SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_VALIDATION_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-file-validation.v1"
)
SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-record.v1"
)

ALLOWED_DECISIONS = [
    "needs_review",
    "approve_for_later_promotion_design",
    "reject_keep_candidate_only",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _append_safety_flags(prefix: str, report: dict[str, Any], *, expected_schema: str) -> list[str]:
    flags: list[str] = []
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    policy = dict(report.get("policy") or {})
    if report.get("schema") != expected_schema:
        flags.append(f"{prefix}_schema_mismatch")
    if str(report.get("status") or "") == "blocked":
        flags.append(f"{prefix}_blocked")
    for key in ("strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"{prefix}_{key}_nonzero")
    for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(gate.get(key)):
            flags.append(f"{prefix}_{key}_true")
    for key in (
        "strictEvidenceCreated",
        "runtimePromotionAllowed",
        "parserRoutingChanged",
        "canonicalParsedArtifactsWritten",
        "databaseMutation",
        "reindexOrReembed",
        "answerIntegrationChanged",
    ):
        if bool(policy.get(key)):
            flags.append(f"{prefix}_{key}_true")
    return flags


def _unsafe_flags(manual_sheet: dict[str, Any], validation: dict[str, Any], decision_record: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    flags.extend(
        _append_safety_flags(
            "selectedReviewManualSheet",
            manual_sheet,
            expected_schema=SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_MANUAL_SHEET_SCHEMA_ID,
        )
    )
    flags.extend(
        _append_safety_flags(
            "selectedDecisionFileValidation",
            validation,
            expected_schema=SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_VALIDATION_SCHEMA_ID,
        )
    )
    flags.extend(
        _append_safety_flags(
            "selectedDecisionRecord",
            decision_record,
            expected_schema=SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_RECORD_SCHEMA_ID,
        )
    )
    return list(dict.fromkeys(flags))


def _brief_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "brief_row_id": f"sectionspan-pdf-offset-selected-review-next-action:{index:04d}",
        "source_decision_row_id": str(row.get("source_decision_row_id") or ""),
        "source_sectionspan_candidate_id": str(row.get("source_sectionspan_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "candidate_text": str(row.get("candidate_text") or ""),
        "section_type": str(row.get("section_type") or ""),
        "review_priority": str(row.get("review_priority") or ""),
        "current_decision": str(row.get("current_decision") or "needs_review"),
        "review_suggestion": str(row.get("review_suggestion") or "needs_review"),
        "review_suggestion_reason": str(row.get("review_suggestion_reason") or ""),
        "validation_status": str(row.get("validation_status") or ""),
        "review_context_status": str(row.get("review_context_status") or ""),
        "page_text_match": bool(row.get("page_text_match")),
        "context_match_method": str(row.get("context_match_method") or ""),
        "matched_text": str(row.get("matched_text") or ""),
        "context_before": str(row.get("context_before") or ""),
        "context_after": str(row.get("context_after") or ""),
        "allowed_decisions": list(row.get("allowed_decisions") or ALLOWED_DECISIONS),
        "required_review_checks": list(row.get("required_review_checks") or []),
        "safe_default_decision": "needs_review",
        "decision_note": "Do not change from needs_review unless a human reviewer confirms the required checks.",
        "required_for_non_needs_review_decision": ["reviewer", "notes"],
        "evidence_tier": "sectionspan_pdf_offset_selected_review_next_action_brief_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "next_action_brief_only",
            "manual_human_review_required_before_decision_record_completion",
            "strict_promotion_requires_later_explicit_apply_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "brief_rows_are_not_human_review_decisions",
            "brief_rows_do_not_authorize_runtime_use",
            "brief_rows_do_not_create_strict_evidence",
        ],
    }


def _counts(rows: list[dict[str, Any]], validation: dict[str, Any], decision_record: dict[str, Any], unsafe_flags: list[str]) -> dict[str, Any]:
    by_decision = Counter(str(row.get("current_decision") or "") for row in rows)
    by_suggestion = Counter(str(row.get("review_suggestion") or "") for row in rows)
    by_paper = Counter(str(row.get("paper_id") or "") for row in rows)
    by_section_type = Counter(str(row.get("section_type") or "") for row in rows)
    by_review_priority = Counter(str(row.get("review_priority") or "") for row in rows)
    validation_counts = dict(validation.get("counts") or {})
    record_counts = dict(decision_record.get("counts") or {})
    return {
        "briefRows": len(rows),
        "needsReviewRows": by_decision.get("needs_review", 0),
        "nonNeedsReviewRows": len(rows) - by_decision.get("needs_review", 0),
        "suggestedApproveForLaterPromotionDesignRows": by_suggestion.get("approve_for_later_promotion_design", 0),
        "suggestedNeedsReviewRows": by_suggestion.get("needs_review", 0),
        "validationValidRows": _safe_int(validation_counts.get("validRows")),
        "validationInvalidRows": _safe_int(validation_counts.get("invalidRows")),
        "validationMissingRows": _safe_int(validation_counts.get("missingRows")),
        "decisionRecordNeedsReviewRows": _safe_int(record_counts.get("needsReviewRows")),
        "decisionRecordApprovedForLaterPromotionDesignRows": _safe_int(
            record_counts.get("approvedForLaterPromotionDesignRows")
        ),
        "decisionRecordRejectedRows": _safe_int(record_counts.get("rejectedRows")),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byDecision": dict(by_decision),
        "bySuggestion": dict(by_suggestion),
        "byPaper": dict(by_paper),
        "bySectionType": dict(by_section_type),
        "byReviewPriority": dict(by_review_priority),
    }


def build_sectionspan_pdf_offset_selected_review_next_action_brief(
    *,
    sectionspan_pdf_offset_selected_review_manual_sheet_report: str | Path,
    sectionspan_pdf_offset_selected_review_decision_file_validation_report: str | Path,
    sectionspan_pdf_offset_selected_review_decision_record_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only next-action brief for selected SectionSpan review."""

    manual_path = Path(str(sectionspan_pdf_offset_selected_review_manual_sheet_report)).expanduser()
    validation_path = Path(str(sectionspan_pdf_offset_selected_review_decision_file_validation_report)).expanduser()
    record_path = Path(str(sectionspan_pdf_offset_selected_review_decision_record_report)).expanduser()
    manual_sheet = _read_json(manual_path)
    validation = _read_json(validation_path)
    decision_record = _read_json(record_path)
    unsafe_flags = _unsafe_flags(manual_sheet, validation, decision_record)
    rows = [
        _brief_row(index, dict(item))
        for index, item in enumerate(list(manual_sheet.get("manualRows") or []), start=1)
        if isinstance(item, dict)
    ]
    counts = _counts(rows, validation, decision_record, unsafe_flags)
    record_needs_review = _safe_int(counts.get("decisionRecordNeedsReviewRows"))
    status = "blocked" if unsafe_flags else "manual_review_required" if record_needs_review else "manual_review_recorded_non_runtime"
    decision = "blocked" if unsafe_flags else "manual_edit_required" if record_needs_review else "manual_decisions_recorded_non_runtime"
    inputs = dict(manual_sheet.get("inputs") or {})
    edit_target = str(inputs.get("selectedReviewDecisionsFile") or "")
    return {
        "schema": SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_NEXT_ACTION_BRIEF_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "sectionspanPdfOffsetSelectedReviewManualSheetReport": str(manual_path),
            "sectionspanPdfOffsetSelectedReviewManualSheetSchema": str(manual_sheet.get("schema") or ""),
            "sectionspanPdfOffsetSelectedReviewDecisionFileValidationReport": str(validation_path),
            "sectionspanPdfOffsetSelectedReviewDecisionFileValidationSchema": str(validation.get("schema") or ""),
            "sectionspanPdfOffsetSelectedReviewDecisionRecordReport": str(record_path),
            "sectionspanPdfOffsetSelectedReviewDecisionRecordSchema": str(decision_record.get("schema") or ""),
            "selectedReviewDecisionsFile": edit_target,
        },
        "counts": counts,
        "gate": {
            "nextActionBriefReady": bool(rows) and not unsafe_flags,
            "manualReviewRequired": bool(rows) and not unsafe_flags and record_needs_review > 0,
            "autoApprovalAllowed": False,
            "humanReviewComplete": bool(rows) and not unsafe_flags and record_needs_review == 0,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "nextEditTarget": edit_target,
            "recommendedNextTranche": "manual_edit_selected_sectionspan_review_decision_file"
            if record_needs_review
            else "sectionspan_selected_decision_record_review_before_any_apply_design",
        },
        "policy": {
            "reportOnly": True,
            "nextActionBriefOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "allowedDecisionValues": ALLOWED_DECISIONS,
        "requiredForNonNeedsReviewDecision": ["reviewer", "notes"],
        "warnings": [
            "next_action_brief_does_not_approve_rows",
            "non_needs_review_decisions_require_human_reviewer_and_notes",
            "approved_rows_only_authorize_later_design_review_not_runtime_promotion",
        ],
        "briefRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "inputs",
            "counts",
            "gate",
            "policy",
            "allowedDecisionValues",
            "requiredForNonNeedsReviewDecision",
            "warnings",
        )
        if key in report
    }


def render_sectionspan_pdf_offset_selected_review_next_action_brief_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan PDF Offset Selected Review Next Action Brief",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Rows: `{int(counts.get('briefRows') or 0)}`",
        f"- `needs_review` rows: `{int(counts.get('needsReviewRows') or 0)}`",
        f"- Suggested approve-for-later-design rows: `{int(counts.get('suggestedApproveForLaterPromotionDesignRows') or 0)}`",
        f"- Decision-record `needs_review` rows: `{int(counts.get('decisionRecordNeedsReviewRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This brief is local review metadata only. It does not approve rows, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Edit Target",
        "",
        f"- Decision file: `{gate.get('nextEditTarget', '')}`",
        "",
        "Allowed decisions: `needs_review`, `approve_for_later_promotion_design`, `reject_keep_candidate_only`.",
        "For any non-`needs_review` decision, fill `reviewer` and `notes`.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By priority: `{json.dumps(counts.get('byReviewPriority') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By decision: `{json.dumps(counts.get('byDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By suggestion: `{json.dumps(counts.get('bySuggestion') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Rows",
        "",
    ]
    for row in list(report.get("briefRows") or []):
        lines.extend(
            [
                f"### {row.get('source_decision_row_id', '')}",
                "",
                f"- Paper: `{row.get('paper_id', '')}`",
                f"- Candidate: `{row.get('candidate_text', '')}`",
                f"- Type: `{row.get('section_type', '')}` / `{row.get('review_priority', '')}`",
                f"- Current decision: `{row.get('current_decision', '')}`",
                f"- Suggestion: `{row.get('review_suggestion', '')}`",
                f"- Validation/context: `{row.get('validation_status', '')}` / `{row.get('review_context_status', '')}` / page text match `{row.get('page_text_match', '')}`",
                f"- Required checks: `{'; '.join(str(item) for item in list(row.get('required_review_checks') or []))}`",
                f"- Matched text: `{row.get('matched_text', '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_sectionspan_pdf_offset_selected_review_next_action_brief_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    brief_path = root / "sectionspan-pdf-offset-selected-review-next-action-brief.json"
    summary_path = root / "sectionspan-pdf-offset-selected-review-next-action-brief-summary.json"
    markdown_path = root / "sectionspan-pdf-offset-selected-review-next-action-brief.md"
    brief_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_sectionspan_pdf_offset_selected_review_next_action_brief_markdown(report),
        encoding="utf-8",
    )
    return {"brief": str(brief_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only selected SectionSpan next-action brief.")
    parser.add_argument("--sectionspan-pdf-offset-selected-review-manual-sheet-report", required=True)
    parser.add_argument("--sectionspan-pdf-offset-selected-review-decision-file-validation-report", required=True)
    parser.add_argument("--sectionspan-pdf-offset-selected-review-decision-record-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_pdf_offset_selected_review_next_action_brief(
        sectionspan_pdf_offset_selected_review_manual_sheet_report=(
            args.sectionspan_pdf_offset_selected_review_manual_sheet_report
        ),
        sectionspan_pdf_offset_selected_review_decision_file_validation_report=(
            args.sectionspan_pdf_offset_selected_review_decision_file_validation_report
        ),
        sectionspan_pdf_offset_selected_review_decision_record_report=(
            args.sectionspan_pdf_offset_selected_review_decision_record_report
        ),
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_pdf_offset_selected_review_next_action_brief_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_NEXT_ACTION_BRIEF_SCHEMA_ID",
    "build_sectionspan_pdf_offset_selected_review_next_action_brief",
    "render_sectionspan_pdf_offset_selected_review_next_action_brief_markdown",
    "write_sectionspan_pdf_offset_selected_review_next_action_brief_reports",
]
