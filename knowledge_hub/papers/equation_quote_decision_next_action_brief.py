"""Report-only next-action brief for EquationQuote review decisions.

The brief makes the current EquationQuote manual decision gate reproducible. It
reads the manual review sheet, decision-file validation report, and decision
record, then points the operator at the editable decision file. It does not
approve rows, create source spans, interpret equations, promote strict evidence,
route parsers, write canonical artifacts, mutate DB state, reindex, reembed, or
change answer behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


EQUATION_QUOTE_DECISION_NEXT_ACTION_BRIEF_SCHEMA_ID = (
    "knowledge-hub.paper.equation-quote-decision-next-action-brief.v1"
)
EQUATION_QUOTE_MANUAL_REVIEW_SHEET_SCHEMA_ID = "knowledge-hub.paper.equation-quote-manual-review-sheet.v1"
EQUATION_QUOTE_DECISION_FILE_VALIDATION_SCHEMA_ID = "knowledge-hub.paper.equation-quote-decision-file-validation.v1"
EQUATION_QUOTE_DECISION_RECORD_SCHEMA_ID = "knowledge-hub.paper.equation-quote-decision-record.v1"

ALLOWED_DECISIONS = [
    "needs_review",
    "accept_diagnostic_context_for_later_reextract_design",
    "reject_equation_quote_candidate",
    "request_equation_quote_reextraction",
    "keep_blocked",
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
        return int(value or 0)
    except Exception:
        return 0


def _append_safety_flags(
    prefix: str,
    report: dict[str, Any],
    *,
    expected_schema: str,
    allowed_statuses: set[str],
) -> list[str]:
    flags: list[str] = []
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    policy = dict(report.get("policy") or {})
    if report.get("schema") != expected_schema:
        flags.append(f"{prefix}_schema_mismatch")
    status = str(report.get("status") or "")
    if status == "blocked":
        flags.append(f"{prefix}_blocked")
    if status and status not in allowed_statuses:
        flags.append(f"{prefix}_status_unexpected")
    for key in (
        "sourceSpanCreatedRows",
        "originalPdfOffsetRecoveredRows",
        "equationSemanticsInterpretedRows",
        "strictEligibleRows",
        "citationGradeRows",
        "runtimeEvidenceRows",
        "invalidRows",
        "missingRows",
    ):
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
            "equationQuoteManualReviewSheet",
            manual_sheet,
            expected_schema=EQUATION_QUOTE_MANUAL_REVIEW_SHEET_SCHEMA_ID,
            allowed_statuses={"manual_review_sheet_ready"},
        )
    )
    flags.extend(
        _append_safety_flags(
            "equationQuoteDecisionFileValidation",
            validation,
            expected_schema=EQUATION_QUOTE_DECISION_FILE_VALIDATION_SCHEMA_ID,
            allowed_statuses={"decision_file_validated"},
        )
    )
    flags.extend(
        _append_safety_flags(
            "equationQuoteDecisionRecord",
            decision_record,
            expected_schema=EQUATION_QUOTE_DECISION_RECORD_SCHEMA_ID,
            allowed_statuses={"decision_record_required", "decision_recorded"},
        )
    )
    return list(dict.fromkeys(flags))


def _record_map(decision_record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for item in list(decision_record.get("decisionRecords") or []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("source_review_sheet_row_id") or "")
        if key:
            mapped[key] = dict(item)
    return mapped


def _required_checks(row: dict[str, Any]) -> list[str]:
    action_type = str(row.get("action_type") or "")
    if action_type == "reject_or_reextract_unmatched_equation_quote":
        return [
            "confirm_candidate_has_no_usable_source_span_or_diagnostic_context",
            "choose_reject_or_reextract_only_after_human_review",
            "keep_needs_review_if_not_sure",
        ]
    return [
        "inspect_diagnostic_page_candidates_against_original_pdf",
        "do_not_treat_diagnostic_context_as_source_span",
        "choose_accept_only_as_later_reextract_design_context_not_runtime_evidence",
        "keep_needs_review_if_not_sure",
    ]


def _brief_row(index: int, row: dict[str, Any], records: dict[str, dict[str, Any]], edit_target: str) -> dict[str, Any]:
    row_id = str(row.get("review_sheet_row_id") or "")
    record = records.get(row_id) or {}
    recorded_decision = str(record.get("recorded_decision") or row.get("current_decision") or "needs_review")
    return {
        "brief_row_id": f"equation-quote-decision-next-action:{index:04d}",
        "source_review_sheet_row_id": row_id,
        "source_action_card_id": str(row.get("source_action_card_id") or ""),
        "source_equation_quote_candidate_id": str(row.get("source_equation_quote_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "candidate_text": str(row.get("candidate_text") or ""),
        "equation_label": str(row.get("equation_label") or ""),
        "action_type": str(row.get("action_type") or ""),
        "action_status": str(row.get("action_status") or ""),
        "priority": str(row.get("priority") or ""),
        "current_decision": recorded_decision,
        "recommended_review_action": str(row.get("recommended_review_action") or ""),
        "review_prompt": str(row.get("review_prompt") or ""),
        "canonical_alignment_status": str(row.get("canonical_alignment_status") or ""),
        "canonical_alignment_method": str(row.get("canonical_alignment_method") or ""),
        "alignment_feasibility_status": str(row.get("alignment_feasibility_status") or ""),
        "pdf_offset_feasibility_status": str(row.get("pdf_offset_feasibility_status") or ""),
        "diagnostic_terms": list(row.get("diagnostic_terms") or []),
        "diagnostic_page_candidates": list(row.get("diagnostic_page_candidates") or []),
        "best_diagnostic_page_coverage": float(row.get("best_diagnostic_page_coverage") or 0.0),
        "layout_element_count": _safe_int(row.get("layout_element_count")),
        "bbox_available": bool(row.get("bbox_available")),
        "sourceContentHash": str(row.get("sourceContentHash") or ""),
        "allowed_decisions": list(row.get("allowed_decisions") or ALLOWED_DECISIONS),
        "required_review_checks": _required_checks(row),
        "safe_default_decision": "needs_review",
        "decision_note": "Do not change from needs_review unless a human reviewer confirms the required checks.",
        "required_for_non_needs_review_decision": ["reviewer", "notes"],
        "next_edit_target": edit_target,
        "source_span_created": False,
        "original_pdf_offset_recovered": False,
        "equation_semantics_interpreted": False,
        "evidence_tier": "equation_quote_decision_next_action_brief_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "equation_quote_decision_next_action_brief_only",
            "manual_human_review_required_before_decision_record_completion",
            "source_span_not_created",
            "equation_semantics_not_interpreted",
            "strict_promotion_requires_later_explicit_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "brief_rows_are_not_human_review_decisions",
            "brief_rows_do_not_create_source_spans",
            "brief_rows_do_not_interpret_equations",
            "brief_rows_do_not_authorize_runtime_use",
        ],
    }


def _counts(rows: list[dict[str, Any]], validation: dict[str, Any], decision_record: dict[str, Any], unsafe_flags: list[str]) -> dict[str, Any]:
    by_decision = Counter(str(row.get("current_decision") or "") for row in rows)
    by_paper = Counter(str(row.get("paper_id") or "") for row in rows)
    by_action_type = Counter(str(row.get("action_type") or "") for row in rows)
    by_action_status = Counter(str(row.get("action_status") or "") for row in rows)
    validation_counts = dict(validation.get("counts") or {})
    record_counts = dict(decision_record.get("counts") or {})
    return {
        "briefRows": len(rows),
        "needsReviewRows": by_decision.get("needs_review", 0),
        "nonNeedsReviewRows": len(rows) - by_decision.get("needs_review", 0),
        "acceptedDiagnosticContextRows": by_decision.get("accepted_diagnostic_context_for_later_reextract_design", 0),
        "rejectedRows": by_decision.get("rejected_equation_quote_candidate", 0),
        "reextractRequestRows": by_decision.get("requested_equation_quote_reextraction", 0),
        "keptBlockedRows": by_decision.get("kept_blocked", 0),
        "diagnosticPageContextRows": by_action_type.get("review_diagnostic_page_context", 0),
        "unmatchedEquationQuoteRows": by_action_type.get("reject_or_reextract_unmatched_equation_quote", 0),
        "validationValidRows": _safe_int(validation_counts.get("validRows")),
        "validationInvalidRows": _safe_int(validation_counts.get("invalidRows")),
        "validationMissingRows": _safe_int(validation_counts.get("missingRows")),
        "decisionRecordNeedsReviewRows": _safe_int(record_counts.get("needsReviewRows")),
        "decisionRecordAcceptedDiagnosticContextRows": _safe_int(record_counts.get("acceptedDiagnosticContextRows")),
        "decisionRecordRejectedRows": _safe_int(record_counts.get("rejectedRows")),
        "decisionRecordReextractRequestRows": _safe_int(record_counts.get("reextractRequestRows")),
        "decisionRecordKeptBlockedRows": _safe_int(record_counts.get("keptBlockedRows")),
        "sourceSpanCreatedRows": 0,
        "originalPdfOffsetRecoveredRows": 0,
        "equationSemanticsInterpretedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byDecision": dict(by_decision),
        "byPaper": dict(by_paper),
        "byActionType": dict(by_action_type),
        "byActionStatus": dict(by_action_status),
    }


def build_equation_quote_decision_next_action_brief(
    *,
    equation_quote_manual_review_sheet_report: str | Path,
    equation_quote_decision_file_validation_report: str | Path,
    equation_quote_decision_record_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only next-action brief for EquationQuote review decisions."""

    manual_path = Path(str(equation_quote_manual_review_sheet_report)).expanduser()
    validation_path = Path(str(equation_quote_decision_file_validation_report)).expanduser()
    record_path = Path(str(equation_quote_decision_record_report)).expanduser()
    manual_sheet = _read_json(manual_path)
    validation = _read_json(validation_path)
    decision_record = _read_json(record_path)
    unsafe_flags = _unsafe_flags(manual_sheet, validation, decision_record)
    inputs = dict(decision_record.get("inputs") or {})
    edit_target = str(inputs.get("reviewDecisionsReport") or "")
    records = _record_map(decision_record)
    rows = [
        _brief_row(index, dict(item), records, edit_target)
        for index, item in enumerate(list(manual_sheet.get("reviewRows") or []), start=1)
        if isinstance(item, dict)
    ]
    counts = _counts(rows, validation, decision_record, unsafe_flags)
    record_needs_review = _safe_int(counts.get("decisionRecordNeedsReviewRows"))
    status = "blocked" if unsafe_flags else "manual_review_required" if record_needs_review else "manual_review_recorded_non_runtime"
    decision = (
        "blocked"
        if unsafe_flags
        else "manual_equation_quote_decision_edit_required"
        if record_needs_review
        else "manual_equation_quote_decisions_recorded_non_runtime"
    )
    return {
        "schema": EQUATION_QUOTE_DECISION_NEXT_ACTION_BRIEF_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "equationQuoteManualReviewSheetReport": str(manual_path),
            "equationQuoteManualReviewSheetSchema": str(manual_sheet.get("schema") or ""),
            "equationQuoteDecisionFileValidationReport": str(validation_path),
            "equationQuoteDecisionFileValidationSchema": str(validation.get("schema") or ""),
            "equationQuoteDecisionRecordReport": str(record_path),
            "equationQuoteDecisionRecordSchema": str(decision_record.get("schema") or ""),
            "equationQuoteDecisionsFile": edit_target,
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
            "recommendedNextTranche": "manual_edit_equation_quote_decision_file"
            if record_needs_review
            else "equation_quote_decision_record_review_before_any_reextract_or_rejection_apply_design",
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
            "diagnostic_context_acceptance_is_for_later_reextract_design_only",
            "reextract_or_reject_decisions_do_not_create_runtime_evidence",
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


def render_equation_quote_decision_next_action_brief_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# EquationQuote Decision Next Action Brief",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Rows: `{int(counts.get('briefRows') or 0)}`",
        f"- `needs_review` rows: `{int(counts.get('needsReviewRows') or 0)}`",
        f"- Diagnostic page-context rows: `{int(counts.get('diagnosticPageContextRows') or 0)}`",
        f"- Unmatched equation rows: `{int(counts.get('unmatchedEquationQuoteRows') or 0)}`",
        f"- Decision-record `needs_review` rows: `{int(counts.get('decisionRecordNeedsReviewRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This brief is local review metadata only. It does not approve rows, create source spans, interpret equations, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Edit Target",
        "",
        f"- Decision file: `{gate.get('nextEditTarget', '')}`",
        "",
        "Allowed decisions: `needs_review`, `accept_diagnostic_context_for_later_reextract_design`, `reject_equation_quote_candidate`, `request_equation_quote_reextraction`, `keep_blocked`.",
        "For any non-`needs_review` decision, fill `reviewer` and `notes`.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By action type: `{json.dumps(counts.get('byActionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By action status: `{json.dumps(counts.get('byActionStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By decision: `{json.dumps(counts.get('byDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Rows",
        "",
    ]
    for row in list(report.get("briefRows") or []):
        terms = ", ".join(str(item) for item in list(row.get("diagnostic_terms") or []))
        lines.extend(
            [
                f"### {row.get('source_review_sheet_row_id', '')}",
                "",
                f"- Paper: `{row.get('paper_id', '')}`",
                f"- Equation: `{row.get('equation_label', '')}`",
                f"- Action: `{row.get('action_type', '')}` / `{row.get('priority', '')}` / `{row.get('action_status', '')}`",
                f"- Current decision: `{row.get('current_decision', '')}`",
                f"- Alignment/offset: `{row.get('alignment_feasibility_status', '')}` / `{row.get('pdf_offset_feasibility_status', '')}`",
                f"- Diagnostic terms: `{terms}`",
                f"- Best diagnostic page coverage: `{row.get('best_diagnostic_page_coverage', '')}`",
                f"- Required checks: `{'; '.join(str(item) for item in list(row.get('required_review_checks') or []))}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_equation_quote_decision_next_action_brief_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    brief_path = root / "equation-quote-decision-next-action-brief.json"
    summary_path = root / "equation-quote-decision-next-action-brief-summary.json"
    markdown_path = root / "equation-quote-decision-next-action-brief.md"
    brief_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_quote_decision_next_action_brief_markdown(report), encoding="utf-8")
    return {"brief": str(brief_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only EquationQuote decision next-action brief.")
    parser.add_argument("--equation-quote-manual-review-sheet-report", required=True)
    parser.add_argument("--equation-quote-decision-file-validation-report", required=True)
    parser.add_argument("--equation-quote-decision-record-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_equation_quote_decision_next_action_brief(
        equation_quote_manual_review_sheet_report=args.equation_quote_manual_review_sheet_report,
        equation_quote_decision_file_validation_report=args.equation_quote_decision_file_validation_report,
        equation_quote_decision_record_report=args.equation_quote_decision_record_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_equation_quote_decision_next_action_brief_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EQUATION_QUOTE_DECISION_NEXT_ACTION_BRIEF_SCHEMA_ID",
    "build_equation_quote_decision_next_action_brief",
    "render_equation_quote_decision_next_action_brief_markdown",
    "write_equation_quote_decision_next_action_brief_reports",
]
