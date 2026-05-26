"""Report-only validation for EquationQuote decision files.

This helper validates a human-edited EquationQuote decision file before any
later decision-record tranche. It does not record decisions, create source
spans, interpret equations, promote strict evidence, route parsers, write
canonical artifacts, mutate DB state, reindex, reembed, or change answer
behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


EQUATION_QUOTE_DECISION_FILE_VALIDATION_SCHEMA_ID = "knowledge-hub.paper.equation-quote-decision-file-validation.v1"
EQUATION_QUOTE_MANUAL_REVIEW_SHEET_SCHEMA_ID = "knowledge-hub.paper.equation-quote-manual-review-sheet.v1"


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


def _review_rows(sheet: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(sheet.get("reviewRows") or []) if isinstance(item, dict)]


def _decision_rows(decision_file: dict[str, Any]) -> list[dict[str, Any]]:
    rows = decision_file.get("decisions")
    if rows is None:
        rows = decision_file.get("decisionRows")
    if rows is None:
        rows = decision_file.get("reviewDecisions")
    return [dict(item) for item in list(rows or []) if isinstance(item, dict)]


def _decision_value(item: dict[str, Any]) -> str:
    return str(item.get("decision") or item.get("review_decision") or item.get("reviewDecision") or "")


def _review_keys(row: dict[str, Any]) -> set[str]:
    return {
        str(row.get("review_sheet_row_id") or ""),
        str(row.get("source_action_card_id") or ""),
        str(row.get("source_equation_quote_candidate_id") or ""),
    } - {""}


def _decision_keys(row: dict[str, Any]) -> set[str]:
    return {
        str(row.get("source_review_sheet_row_id") or ""),
        str(row.get("review_sheet_row_id") or ""),
        str(row.get("source_action_card_id") or ""),
        str(row.get("action_card_id") or ""),
        str(row.get("source_equation_quote_candidate_id") or ""),
        str(row.get("equation_quote_candidate_id") or ""),
    } - {""}


def _review_key_map(sheet: dict[str, Any]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for row in _review_rows(sheet):
        primary = str(row.get("review_sheet_row_id") or "")
        for key in _review_keys(row):
            mapped[key] = primary
    return mapped


def _unsafe_sheet_flags(sheet: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(sheet.get("counts") or {})
    gate = dict(sheet.get("gate") or {})
    policy = dict(sheet.get("policy") or {})
    if sheet.get("schema") != EQUATION_QUOTE_MANUAL_REVIEW_SHEET_SCHEMA_ID:
        flags.append("equation_quote_manual_review_sheet_schema_mismatch")
    if str(sheet.get("status") or "") == "blocked":
        flags.append("equation_quote_manual_review_sheet_blocked")
    for key in (
        "sourceSpanCreatedRows",
        "originalPdfOffsetRecoveredRows",
        "equationSemanticsInterpretedRows",
        "strictEligibleRows",
        "citationGradeRows",
        "runtimeEvidenceRows",
    ):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"manualReviewSheet_{key}_nonzero")
    for key in (
        "humanReviewComplete",
        "strictEvidenceReady",
        "parserRoutingReady",
        "answerIntegrationReady",
        "runtimePromotionAllowed",
    ):
        if bool(gate.get(key)):
            flags.append(f"manualReviewSheet_{key}_true")
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
            flags.append(f"manualReviewSheet_{key}_true")
    return list(dict.fromkeys(flags))


def _submitted_decisions(
    sheet: dict[str, Any],
    decision_file: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    review_keys = _review_key_map(sheet)
    mapped: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    seen: set[str] = set()
    for item in _decision_rows(decision_file):
        keys = _decision_keys(item)
        if not keys:
            errors.append("equation_quote_decision_row_id_missing")
            continue
        matched = [review_keys[key] for key in keys if key in review_keys]
        primary_id = matched[0] if matched else ""
        if not primary_id:
            errors.append("equation_quote_decision_unknown_review_row_id")
            continue
        if primary_id in seen:
            errors.append("equation_quote_decision_duplicate_row_id")
        seen.add(primary_id)
        if primary_id not in mapped:
            mapped[primary_id] = dict(item)
    return mapped, list(dict.fromkeys(errors))


def _allowed_decisions(review_row: dict[str, Any]) -> list[str]:
    allowed = [str(item) for item in list(review_row.get("allowed_decisions") or []) if item]
    if not allowed:
        allowed = [
            "needs_review",
            "reject_equation_quote_candidate",
            "request_equation_quote_reextraction",
            "keep_blocked",
        ]
    if "needs_review" not in allowed:
        allowed = ["needs_review", *allowed]
    return allowed


def _validation_row(index: int, review_row: dict[str, Any], submitted: dict[str, Any] | None) -> dict[str, Any]:
    row_id = str(review_row.get("review_sheet_row_id") or "")
    allowed = _allowed_decisions(review_row)
    decision = _decision_value(submitted or {}) if submitted else ""
    reviewer = str((submitted or {}).get("reviewer") or "")
    notes = str((submitted or {}).get("notes") or "")
    errors: list[str] = []
    if not submitted:
        submitted_decision = "needs_review"
        validation_status = "missing"
        errors.append("decision_missing")
    else:
        submitted_decision = decision or "needs_review"
        if submitted_decision not in allowed:
            errors.append("equation_quote_decision_not_allowed")
        if submitted_decision != "needs_review" and not reviewer:
            errors.append("reviewer_required_for_non_needs_review_decision")
        if submitted_decision != "needs_review" and not notes:
            errors.append("notes_required_for_non_needs_review_decision")
        validation_status = "valid" if not errors else "invalid"
    return {
        "validation_row_id": f"equation-quote-decision-file-validation:{index:04d}",
        "source_review_sheet_row_id": row_id,
        "source_action_card_id": str(review_row.get("source_action_card_id") or ""),
        "source_equation_quote_candidate_id": str(review_row.get("source_equation_quote_candidate_id") or ""),
        "paper_id": str(review_row.get("paper_id") or ""),
        "candidate_text": str(review_row.get("candidate_text") or ""),
        "equation_label": str(review_row.get("equation_label") or ""),
        "action_type": str(review_row.get("action_type") or ""),
        "action_status": str(review_row.get("action_status") or ""),
        "priority": str(review_row.get("priority") or ""),
        "allowed_decisions": allowed,
        "submitted_decision": submitted_decision,
        "reviewer": reviewer,
        "notes": notes,
        "validation_status": validation_status,
        "validation_errors": errors,
        "source_span_created": False,
        "original_pdf_offset_recovered": False,
        "equation_semantics_interpreted": False,
        "decision_scope": "equation_quote_decision_file_validation_only_no_runtime_or_strict_promotion",
        "evidence_tier": "equation_quote_decision_file_validation_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "equation_quote_decision_file_validation_only",
            "decisions_not_recorded_by_validation_report",
            "source_span_not_created",
            "equation_semantics_not_interpreted",
            "strict_promotion_requires_later_explicit_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "validation_rows_are_not_human_review_decision_records",
            "validation_rows_do_not_create_source_spans",
            "validation_rows_do_not_interpret_equations",
            "validation_rows_do_not_authorize_runtime_use",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str], file_errors: list[str]) -> dict[str, Any]:
    by_status = Counter(str(row.get("validation_status") or "") for row in rows)
    by_decision = Counter(str(row.get("submitted_decision") or "") for row in rows)
    by_paper = Counter(str(row.get("paper_id") or "") for row in rows)
    by_action_type = Counter(str(row.get("action_type") or "") for row in rows)
    return {
        "validationRows": len(rows),
        "validRows": by_status.get("valid", 0),
        "invalidRows": by_status.get("invalid", 0),
        "missingRows": by_status.get("missing", 0),
        "needsReviewRows": by_decision.get("needs_review", 0),
        "submittedRejectRows": by_decision.get("reject_equation_quote_candidate", 0),
        "submittedReextractRequestRows": by_decision.get("request_equation_quote_reextraction", 0),
        "submittedKeepBlockedRows": by_decision.get("keep_blocked", 0),
        "nonNeedsReviewRows": len(rows) - by_decision.get("needs_review", 0),
        "fileErrorCount": len(file_errors),
        "sourceSpanCreatedRows": 0,
        "originalPdfOffsetRecoveredRows": 0,
        "equationSemanticsInterpretedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byStatus": dict(by_status),
        "byDecision": dict(by_decision),
        "byPaper": dict(by_paper),
        "byActionType": dict(by_action_type),
    }


def build_equation_quote_decision_file_validation(
    *,
    equation_quote_manual_review_sheet_report: str | Path,
    review_decisions_report: str | Path | None = None,
) -> dict[str, Any]:
    """Validate an EquationQuote decision file without recording it."""

    sheet_path = Path(str(equation_quote_manual_review_sheet_report)).expanduser()
    decisions_path = Path(str(review_decisions_report)).expanduser() if review_decisions_report else None
    sheet = _read_json(sheet_path)
    decision_file = _read_json(decisions_path) if decisions_path else {}
    submitted, file_errors = _submitted_decisions(sheet, decision_file)
    unsafe_flags = _unsafe_sheet_flags(sheet)
    rows = [
        _validation_row(index, row, submitted.get(str(row.get("review_sheet_row_id") or "")))
        for index, row in enumerate(_review_rows(sheet), start=1)
    ]
    counts = _counts(rows, unsafe_flags, file_errors)
    if unsafe_flags or file_errors:
        status = "blocked"
        decision = "blocked"
    elif not decisions_path:
        status = "decision_file_required"
        decision = "manual_equation_quote_decision_file_missing"
    elif _safe_int(counts.get("missingRows")) or _safe_int(counts.get("invalidRows")):
        status = "decision_file_incomplete"
        decision = "manual_equation_quote_decision_file_incomplete"
    else:
        status = "decision_file_validated"
        decision = "manual_equation_quote_decision_file_validated_non_runtime"
    return {
        "schema": EQUATION_QUOTE_DECISION_FILE_VALIDATION_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "equationQuoteManualReviewSheetReport": str(sheet_path),
            "equationQuoteManualReviewSheetSchema": str(sheet.get("schema") or ""),
            "reviewDecisionsReport": str(decisions_path or ""),
            "reviewDecisionInputRows": len(_decision_rows(decision_file)),
        },
        "counts": counts,
        "gate": {
            "decisionFileValidationReady": bool(rows) and not unsafe_flags,
            "decisionFileComplete": status == "decision_file_validated",
            "containsRecordedDecisions": False,
            "humanReviewRecordComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "fileValidationErrors": file_errors,
            "recommendedNextTranche": "manual_fill_equation_quote_decision_file"
            if status != "decision_file_validated"
            else "equation_quote_decision_record_from_validated_file_requires_explicit_review",
        },
        "policy": {
            "reportOnly": True,
            "decisionFileValidationOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "decision_file_validation_does_not_record_human_review_decisions",
            "valid_decision_rows_do_not_authorize_runtime_use",
            "diagnostic_context_does_not_create_source_spans",
            "equation_semantics_are_not_interpreted",
            "strict_or_runtime_promotion_requires_a_separate_explicit_tranche",
        ],
        "validationRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_equation_quote_decision_file_validation_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# EquationQuote Decision File Validation",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Validation rows: `{int(counts.get('validationRows') or 0)}`",
        f"- Valid rows: `{int(counts.get('validRows') or 0)}`",
        f"- Missing rows: `{int(counts.get('missingRows') or 0)}`",
        f"- Invalid rows: `{int(counts.get('invalidRows') or 0)}`",
        f"- `needs_review` rows: `{int(counts.get('needsReviewRows') or 0)}`",
        f"- Reject rows: `{int(counts.get('submittedRejectRows') or 0)}`",
        f"- Re-extraction request rows: `{int(counts.get('submittedReextractRequestRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This validation report is report-only. It does not record human review decisions, create source spans, interpret equations, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By status: `{json.dumps(counts.get('byStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By decision: `{json.dumps(counts.get('byDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By action type: `{json.dumps(counts.get('byActionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_equation_quote_decision_file_validation_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    validation_path = root / "equation-quote-decision-file-validation.json"
    summary_path = root / "equation-quote-decision-file-validation-summary.json"
    markdown_path = root / "equation-quote-decision-file-validation.md"
    validation_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_quote_decision_file_validation_markdown(report), encoding="utf-8")
    return {"validation": str(validation_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Validate an EquationQuote decision file without applying it.")
    parser.add_argument("--equation-quote-manual-review-sheet-report", required=True)
    parser.add_argument("--review-decisions-report", default="")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_equation_quote_decision_file_validation(
        equation_quote_manual_review_sheet_report=args.equation_quote_manual_review_sheet_report,
        review_decisions_report=args.review_decisions_report or None,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_equation_quote_decision_file_validation_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EQUATION_QUOTE_DECISION_FILE_VALIDATION_SCHEMA_ID",
    "build_equation_quote_decision_file_validation",
    "render_equation_quote_decision_file_validation_markdown",
    "write_equation_quote_decision_file_validation_reports",
]
