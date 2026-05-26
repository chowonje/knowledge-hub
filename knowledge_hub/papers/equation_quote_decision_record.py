"""Report-only decision record for EquationQuote review decisions.

This helper turns a validated EquationQuote decision file into a review record.
It does not create source spans, interpret equations, promote strict evidence,
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


EQUATION_QUOTE_DECISION_RECORD_SCHEMA_ID = "knowledge-hub.paper.equation-quote-decision-record.v1"
EQUATION_QUOTE_MANUAL_REVIEW_SHEET_SCHEMA_ID = "knowledge-hub.paper.equation-quote-manual-review-sheet.v1"
EQUATION_QUOTE_DECISION_FILE_VALIDATION_SCHEMA_ID = "knowledge-hub.paper.equation-quote-decision-file-validation.v1"

_VALID_DECISIONS = {
    "needs_review": "needs_review",
    "accept_diagnostic_context_for_later_reextract_design": "accepted_diagnostic_context_for_later_reextract_design",
    "reject_equation_quote_candidate": "rejected_equation_quote_candidate",
    "request_equation_quote_reextraction": "requested_equation_quote_reextraction",
    "keep_blocked": "kept_blocked",
}


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


def _decision_rows(decisions_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = decisions_report.get("decisions")
    if rows is None:
        rows = decisions_report.get("decisionRows")
    if rows is None:
        rows = decisions_report.get("reviewDecisions")
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


def _unsafe_flags(sheet: dict[str, Any], validation: dict[str, Any], decisions_report: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    decision_rows = _decision_rows(decisions_report)
    sheet_counts = dict(sheet.get("counts") or {})
    sheet_gate = dict(sheet.get("gate") or {})
    sheet_policy = dict(sheet.get("policy") or {})
    validation_counts = dict(validation.get("counts") or {})
    validation_gate = dict(validation.get("gate") or {})
    validation_policy = dict(validation.get("policy") or {})

    if sheet.get("schema") != EQUATION_QUOTE_MANUAL_REVIEW_SHEET_SCHEMA_ID:
        flags.append("equation_quote_manual_review_sheet_schema_mismatch")
    if validation and validation.get("schema") != EQUATION_QUOTE_DECISION_FILE_VALIDATION_SCHEMA_ID:
        flags.append("equation_quote_decision_file_validation_schema_mismatch")
    if str(sheet.get("status") or "") == "blocked":
        flags.append("equation_quote_manual_review_sheet_blocked")
    if decision_rows and not validation:
        flags.append("equation_quote_decision_file_validation_missing")
    if str(validation.get("status") or "") == "blocked":
        flags.append("equation_quote_decision_file_validation_blocked")
    if validation and str(validation.get("status") or "") != "decision_file_validated":
        flags.append("equation_quote_decision_file_validation_not_validated")
    for key in ("missingRows", "invalidRows"):
        if _safe_int(validation_counts.get(key)) > 0:
            flags.append(f"decisionValidation_{key}_nonzero")

    for prefix, counts in (("manualReviewSheet", sheet_counts), ("decisionValidation", validation_counts)):
        for key in (
            "sourceSpanCreatedRows",
            "originalPdfOffsetRecoveredRows",
            "equationSemanticsInterpretedRows",
            "strictEligibleRows",
            "citationGradeRows",
            "runtimeEvidenceRows",
        ):
            if _safe_int(counts.get(key)) > 0:
                flags.append(f"{prefix}_{key}_nonzero")

    for prefix, gate in (("manualReviewSheet", sheet_gate), ("decisionValidation", validation_gate)):
        for key in (
            "strictEvidenceReady",
            "parserRoutingReady",
            "answerIntegrationReady",
            "runtimePromotionAllowed",
        ):
            if bool(gate.get(key)):
                flags.append(f"{prefix}_{key}_true")

    for prefix, policy in (("manualReviewSheet", sheet_policy), ("decisionValidation", validation_policy)):
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

    seen: set[str] = set()
    review_keys = _review_key_map(sheet)
    for item in decision_rows:
        keys = _decision_keys(item)
        decision = _decision_value(item)
        if not keys:
            flags.append("equation_quote_decision_row_id_missing")
            continue
        matched = [review_keys[key] for key in keys if key in review_keys]
        primary_id = matched[0] if matched else ""
        if not primary_id:
            flags.append("equation_quote_decision_unknown_review_row_id")
        if primary_id and primary_id in seen:
            flags.append("equation_quote_decision_duplicate_row_id")
        if primary_id:
            seen.add(primary_id)
        if decision not in _VALID_DECISIONS:
            flags.append("equation_quote_decision_invalid_value")
    return list(dict.fromkeys(flags))


def _decision_map(sheet: dict[str, Any], decisions_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    review_keys = _review_key_map(sheet)
    mapped: dict[str, dict[str, Any]] = {}
    for item in _decision_rows(decisions_report):
        keys = _decision_keys(item)
        matched = [review_keys[key] for key in keys if key in review_keys]
        primary_id = matched[0] if matched else ""
        decision = _decision_value(item)
        if primary_id and decision in _VALID_DECISIONS and primary_id not in mapped:
            mapped[primary_id] = dict(item)
    return mapped


def _record_row(index: int, review_row: dict[str, Any], decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row_id = str(review_row.get("review_sheet_row_id") or "")
    decision_item = decisions.get(row_id) or {}
    raw_decision = _decision_value(decision_item)
    recorded_decision = _VALID_DECISIONS.get(raw_decision, "needs_review")
    strict_blockers = [
        "equation_quote_decision_record_only",
        "source_span_not_created",
        "equation_semantics_not_interpreted",
        "strict_promotion_requires_later_explicit_tranche",
        "runtime_promotion_disabled_for_tranche",
    ]
    if recorded_decision == "needs_review":
        strict_blockers.append("human_review_decision_missing")
    elif recorded_decision == "accepted_diagnostic_context_for_later_reextract_design":
        strict_blockers.append("diagnostic_context_acceptance_is_for_later_design_only")
    elif recorded_decision == "rejected_equation_quote_candidate":
        strict_blockers.append("human_review_rejected_equation_quote_candidate")
    elif recorded_decision == "requested_equation_quote_reextraction":
        strict_blockers.append("human_review_requested_reextraction")
    elif recorded_decision == "kept_blocked":
        strict_blockers.append("human_review_kept_blocked")
    return {
        "record_row_id": f"equation-quote-decision-record:{index:04d}",
        "source_review_sheet_row_id": row_id,
        "source_action_card_id": str(review_row.get("source_action_card_id") or ""),
        "source_equation_quote_candidate_id": str(review_row.get("source_equation_quote_candidate_id") or ""),
        "paper_id": str(review_row.get("paper_id") or ""),
        "candidate_text": str(review_row.get("candidate_text") or ""),
        "equation_label": str(review_row.get("equation_label") or ""),
        "action_type": str(review_row.get("action_type") or ""),
        "action_status": str(review_row.get("action_status") or ""),
        "priority": str(review_row.get("priority") or ""),
        "recorded_decision": recorded_decision,
        "decision_scope": "equation_quote_decision_record_only_no_runtime_or_strict_promotion",
        "reviewer": str(decision_item.get("reviewer") or ""),
        "notes": str(decision_item.get("notes") or ""),
        "source_span_created": False,
        "original_pdf_offset_recovered": False,
        "equation_semantics_interpreted": False,
        "evidence_tier": "equation_quote_decision_record_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": [
            "equation_quote_decision_records_are_not_strict_evidence",
            "decision_records_do_not_create_source_spans",
            "decision_records_do_not_interpret_equations",
            "decision_records_do_not_authorize_runtime_use",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    by_decision = Counter(str(row.get("recorded_decision") or "") for row in rows)
    return {
        "recordRows": len(rows),
        "needsReviewRows": by_decision.get("needs_review", 0),
        "acceptedDiagnosticContextRows": by_decision.get("accepted_diagnostic_context_for_later_reextract_design", 0),
        "rejectedRows": by_decision.get("rejected_equation_quote_candidate", 0),
        "reextractRequestRows": by_decision.get("requested_equation_quote_reextraction", 0),
        "keptBlockedRows": by_decision.get("kept_blocked", 0),
        "sourceSpanCreatedRows": 0,
        "originalPdfOffsetRecoveredRows": 0,
        "equationSemanticsInterpretedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byPaper": dict(Counter(str(row.get("paper_id") or "") for row in rows)),
        "byActionType": dict(Counter(str(row.get("action_type") or "") for row in rows)),
        "byDecision": dict(by_decision),
    }


def build_equation_quote_decision_record(
    *,
    equation_quote_manual_review_sheet_report: str | Path,
    equation_quote_decision_file_validation_report: str | Path | None = None,
    review_decisions_report: str | Path | None = None,
) -> dict[str, Any]:
    """Build a report-only EquationQuote decision record."""

    sheet_path = Path(str(equation_quote_manual_review_sheet_report)).expanduser()
    validation_path = (
        Path(str(equation_quote_decision_file_validation_report)).expanduser()
        if equation_quote_decision_file_validation_report
        else None
    )
    decisions_path = Path(str(review_decisions_report)).expanduser() if review_decisions_report else None
    sheet = _read_json(sheet_path)
    validation = _read_json(validation_path)
    decisions_report = _read_json(decisions_path) if decisions_path else {}
    unsafe_flags = _unsafe_flags(sheet, validation, decisions_report)
    decisions = _decision_map(sheet, decisions_report)
    rows = [_record_row(index, row, decisions) for index, row in enumerate(_review_rows(sheet), start=1)]
    counts = _counts(rows, unsafe_flags)
    needs_review = _safe_int(counts.get("needsReviewRows"))
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif needs_review:
        status = "decision_record_required"
        decision = "manual_equation_quote_decisions_still_required"
    else:
        status = "decision_recorded"
        decision = "manual_equation_quote_decisions_recorded_non_runtime"
    return {
        "schema": EQUATION_QUOTE_DECISION_RECORD_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "equationQuoteManualReviewSheetReport": str(sheet_path),
            "equationQuoteManualReviewSheetSchema": str(sheet.get("schema") or ""),
            "equationQuoteDecisionFileValidationReport": str(validation_path or ""),
            "equationQuoteDecisionFileValidationSchema": str(validation.get("schema") or ""),
            "reviewDecisionsReport": str(decisions_path or ""),
            "reviewDecisionInputRows": len(_decision_rows(decisions_report)),
        },
        "counts": counts,
        "gate": {
            "decisionRecordReady": bool(rows) and not unsafe_flags,
            "humanReviewComplete": bool(rows) and not unsafe_flags and needs_review == 0,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_record_equation_quote_decisions"
            if needs_review
            else "equation_quote_reextract_or_rejection_apply_design_requires_explicit_approval",
        },
        "policy": {
            "reportOnly": True,
            "decisionRecordOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "equation_quote_decision_records_are_not_strict_evidence",
            "decision_records_do_not_create_source_spans",
            "decision_records_do_not_interpret_equations",
            "strict_or_runtime_promotion_requires_a_separate_explicit_tranche",
        ],
        "decisionRecords": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_equation_quote_decision_record_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# EquationQuote Decision Record",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Record rows: `{int(counts.get('recordRows') or 0)}`",
        f"- Needs review: `{int(counts.get('needsReviewRows') or 0)}`",
        f"- Accepted diagnostic context: `{int(counts.get('acceptedDiagnosticContextRows') or 0)}`",
        f"- Rejected: `{int(counts.get('rejectedRows') or 0)}`",
        f"- Re-extraction requests: `{int(counts.get('reextractRequestRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This record is report-only. It does not create source spans, interpret equations, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By action type: `{json.dumps(counts.get('byActionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By decision: `{json.dumps(counts.get('byDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_equation_quote_decision_record_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / "equation-quote-decision-record.json"
    summary_path = root / "equation-quote-decision-record-summary.json"
    markdown_path = root / "equation-quote-decision-record.md"
    record_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_quote_decision_record_markdown(report), encoding="utf-8")
    return {"record": str(record_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only EquationQuote decision record.")
    parser.add_argument("--equation-quote-manual-review-sheet-report", required=True)
    parser.add_argument("--equation-quote-decision-file-validation-report", default="")
    parser.add_argument("--review-decisions-report", default="")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_equation_quote_decision_record(
        equation_quote_manual_review_sheet_report=args.equation_quote_manual_review_sheet_report,
        equation_quote_decision_file_validation_report=args.equation_quote_decision_file_validation_report or None,
        review_decisions_report=args.review_decisions_report or None,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_equation_quote_decision_record_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EQUATION_QUOTE_DECISION_RECORD_SCHEMA_ID",
    "build_equation_quote_decision_record",
    "render_equation_quote_decision_record_markdown",
    "write_equation_quote_decision_record_reports",
]
