"""Report-only manual review sheet for selected SectionSpan PDF offset rows.

This helper joins the selected decision template, original-PDF evidence pack,
editable decision file, and decision-file validation report into one
operator-readable sheet. It does not record human decisions, create strict
evidence, route parsers, write canonical artifacts, mutate DB state, reindex,
reembed, or change answer behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_MANUAL_SHEET_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-manual-sheet.v1"
)
SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_TEMPLATE_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-template.v1"
)
SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_EVIDENCE_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-evidence-pack.v1"
)
SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_VALIDATION_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-file-validation.v1"
)


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


def _decision_rows(decisions_file: dict[str, Any]) -> list[dict[str, Any]]:
    rows = decisions_file.get("decisions")
    if rows is None:
        rows = decisions_file.get("decisionRows")
    if rows is None:
        rows = decisions_file.get("reviewDecisions")
    return [dict(item) for item in list(rows or []) if isinstance(item, dict)]


def _decision_id(item: dict[str, Any]) -> str:
    return str(
        item.get("source_decision_row_id")
        or item.get("sourceDecisionRowId")
        or item.get("decision_row_id")
        or item.get("decisionRowId")
        or ""
    )


def _decision_value(item: dict[str, Any]) -> str:
    return str(item.get("decision") or item.get("review_decision") or item.get("reviewDecision") or "")


def _template_rows(template: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(template.get("decisionRows") or []) if isinstance(item, dict)]


def _by_id(rows: list[dict[str, Any]], key_name: str = "source_decision_row_id") -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get(key_name) or "")
        if row_id and row_id not in mapped:
            mapped[row_id] = row
    return mapped


def _decision_file_by_id(decisions_file: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for row in _decision_rows(decisions_file):
        row_id = _decision_id(row)
        if row_id and row_id not in mapped:
            mapped[row_id] = row
    return mapped


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


def _unsafe_flags(template: dict[str, Any], evidence: dict[str, Any], validation: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    flags.extend(
        _append_safety_flags(
            "selectedDecisionTemplate",
            template,
            expected_schema=SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_TEMPLATE_SCHEMA_ID,
        )
    )
    flags.extend(
        _append_safety_flags(
            "selectedReviewEvidencePack",
            evidence,
            expected_schema=SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_EVIDENCE_PACK_SCHEMA_ID,
        )
    )
    flags.extend(
        _append_safety_flags(
            "selectedDecisionFileValidation",
            validation,
            expected_schema=SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_VALIDATION_SCHEMA_ID,
        )
    )
    return list(dict.fromkeys(flags))


def _manual_row(
    index: int,
    template_row: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
    decision_by_id: dict[str, dict[str, Any]],
    validation_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row_id = str(template_row.get("decision_row_id") or "")
    evidence_row = evidence_by_id.get(row_id) or {}
    decision_row = decision_by_id.get(row_id) or {}
    validation_row = validation_by_id.get(row_id) or {}
    current_decision = _decision_value(decision_row) or "needs_review"
    return {
        "manual_sheet_row_id": f"sectionspan-pdf-offset-selected-review-manual-sheet:{index:04d}",
        "source_decision_row_id": row_id,
        "source_selected_review_card_id": str(template_row.get("source_selected_review_card_id") or ""),
        "source_priority_card_id": str(template_row.get("source_priority_card_id") or ""),
        "source_original_decision_row_id": str(template_row.get("source_decision_row_id") or ""),
        "source_gate_row_id": str(template_row.get("source_gate_row_id") or ""),
        "source_review_card_id": str(template_row.get("source_review_card_id") or ""),
        "source_sectionspan_candidate_id": str(template_row.get("source_sectionspan_candidate_id") or ""),
        "paper_id": str(template_row.get("paper_id") or ""),
        "candidate_text": str(template_row.get("candidate_text") or ""),
        "section_type": str(template_row.get("section_type") or ""),
        "section_level": _safe_int(template_row.get("section_level")),
        "review_priority": str(template_row.get("review_priority") or ""),
        "canonical_span": dict(template_row.get("canonical_span") or {}),
        "original_pdf_span": dict(template_row.get("original_pdf_span") or {}),
        "page_agreement": bool(template_row.get("page_agreement")),
        "source_hash_agreement": bool(template_row.get("source_hash_agreement")),
        "review_context_status": str(evidence_row.get("review_context_status") or ""),
        "page_text_match": bool(evidence_row.get("page_text_match")),
        "context_match_method": str(evidence_row.get("context_match_method") or ""),
        "matched_text": str(evidence_row.get("matched_text") or ""),
        "context_before": str(evidence_row.get("context_before") or ""),
        "context_after": str(evidence_row.get("context_after") or ""),
        "review_suggestion": str(evidence_row.get("review_suggestion") or "needs_review"),
        "review_suggestion_reason": str(evidence_row.get("review_suggestion_reason") or ""),
        "current_decision": current_decision,
        "draft_reviewer": str(decision_row.get("reviewer") or ""),
        "draft_notes": str(decision_row.get("notes") or ""),
        "validation_status": str(validation_row.get("validation_status") or ""),
        "validation_errors": list(validation_row.get("validation_errors") or []),
        "allowed_decisions": list(template_row.get("allowed_decisions") or []),
        "required_review_checks": list(template_row.get("required_review_checks") or []),
        "decision_note": "Edit the selected decision draft only after manual review; leave needs_review when unsure.",
        "decision_scope": "sectionspan_pdf_offset_selected_review_manual_sheet_only_no_runtime_or_strict_promotion",
        "evidence_tier": "sectionspan_pdf_offset_selected_review_manual_sheet_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "selected_review_manual_sheet_only",
            "manual_sheet_rows_are_not_decision_records",
            "strict_promotion_requires_later_explicit_apply_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "manual_sheet_rows_are_not_human_review_decisions",
            "manual_sheet_rows_do_not_authorize_runtime_use",
            "manual_sheet_rows_do_not_create_strict_evidence",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    by_decision = Counter(str(row.get("current_decision") or "") for row in rows)
    by_validation = Counter(str(row.get("validation_status") or "") for row in rows)
    by_context = Counter(str(row.get("review_context_status") or "") for row in rows)
    by_paper = Counter(str(row.get("paper_id") or "") for row in rows)
    by_section_type = Counter(str(row.get("section_type") or "") for row in rows)
    by_review_priority = Counter(str(row.get("review_priority") or "") for row in rows)
    return {
        "manualSheetRows": len(rows),
        "needsReviewRows": by_decision.get("needs_review", 0),
        "nonNeedsReviewRows": len(rows) - by_decision.get("needs_review", 0),
        "validationValidRows": by_validation.get("valid", 0),
        "validationInvalidRows": by_validation.get("invalid", 0),
        "validationMissingRows": by_validation.get("missing", 0),
        "reviewContextReadyRows": by_context.get("review_context_ready", 0),
        "pageTextMatchRows": sum(1 for row in rows if row.get("page_text_match")),
        "suggestedApproveForLaterPromotionDesignRows": sum(
            1 for row in rows if row.get("review_suggestion") == "approve_for_later_promotion_design"
        ),
        "suggestedNeedsReviewRows": sum(1 for row in rows if row.get("review_suggestion") == "needs_review"),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byDecision": dict(by_decision),
        "byValidationStatus": dict(by_validation),
        "byContextStatus": dict(by_context),
        "byPaper": dict(by_paper),
        "bySectionType": dict(by_section_type),
        "byReviewPriority": dict(by_review_priority),
    }


def build_sectionspan_pdf_offset_selected_review_manual_sheet(
    *,
    sectionspan_pdf_offset_selected_review_decision_template_report: str | Path,
    sectionspan_pdf_offset_selected_review_evidence_pack_report: str | Path,
    selected_review_decisions_file: str | Path,
    sectionspan_pdf_offset_selected_review_decision_file_validation_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only manual sheet for selected SectionSpan review rows."""

    template_path = Path(str(sectionspan_pdf_offset_selected_review_decision_template_report)).expanduser()
    evidence_path = Path(str(sectionspan_pdf_offset_selected_review_evidence_pack_report)).expanduser()
    decisions_path = Path(str(selected_review_decisions_file)).expanduser()
    validation_path = Path(str(sectionspan_pdf_offset_selected_review_decision_file_validation_report)).expanduser()
    template = _read_json(template_path)
    evidence = _read_json(evidence_path)
    decisions_file = _read_json(decisions_path)
    validation = _read_json(validation_path)
    unsafe_flags = _unsafe_flags(template, evidence, validation)
    evidence_by_id = _by_id([dict(item) for item in list(evidence.get("evidenceRows") or []) if isinstance(item, dict)])
    decision_by_id = _decision_file_by_id(decisions_file)
    validation_by_id = _by_id([dict(item) for item in list(validation.get("validationRows") or []) if isinstance(item, dict)])
    rows = [
        _manual_row(index, row, evidence_by_id, decision_by_id, validation_by_id)
        for index, row in enumerate(_template_rows(template), start=1)
    ]
    counts = _counts(rows, unsafe_flags)
    status = "blocked" if unsafe_flags else "selected_manual_sheet_ready"
    return {
        "schema": SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_MANUAL_SHEET_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "sectionspanPdfOffsetSelectedReviewDecisionTemplateReport": str(template_path),
            "sectionspanPdfOffsetSelectedReviewDecisionTemplateSchema": str(template.get("schema") or ""),
            "sectionspanPdfOffsetSelectedReviewEvidencePackReport": str(evidence_path),
            "sectionspanPdfOffsetSelectedReviewEvidencePackSchema": str(evidence.get("schema") or ""),
            "selectedReviewDecisionsFile": str(decisions_path),
            "sectionspanPdfOffsetSelectedReviewDecisionFileValidationReport": str(validation_path),
            "sectionspanPdfOffsetSelectedReviewDecisionFileValidationSchema": str(validation.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "selectedManualSheetReady": bool(rows) and not unsafe_flags,
            "containsNonNeedsReviewDraftValues": _safe_int(counts.get("nonNeedsReviewRows")) > 0,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "blocked" if unsafe_flags else "selected_manual_sheet_ready_for_human_edit",
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_edit_selected_sectionspan_review_decision_file",
        },
        "policy": {
            "reportOnly": True,
            "selectedManualSheetOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "manual_sheet_rows_are_not_human_review_decisions",
            "manual_sheet_does_not_authorize_runtime_use",
            "approval_requires_a_decision_record_and_later_explicit_apply_tranche",
        ],
        "manualRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_sectionspan_pdf_offset_selected_review_manual_sheet_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    inputs = dict(report.get("inputs") or {})
    lines = [
        "# SectionSpan PDF Offset Selected Manual Review Sheet",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Rows: `{int(counts.get('manualSheetRows') or 0)}`",
        f"- `needs_review` rows: `{int(counts.get('needsReviewRows') or 0)}`",
        f"- Non-`needs_review` draft values: `{int(counts.get('nonNeedsReviewRows') or 0)}`",
        f"- Context-ready rows: `{int(counts.get('reviewContextReadyRows') or 0)}`",
        f"- Suggested approve-for-later-design rows: `{int(counts.get('suggestedApproveForLaterPromotionDesignRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This sheet is local review metadata only. It does not record approvals, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Edit Target",
        "",
        f"- Decision file: `{inputs.get('selectedReviewDecisionsFile', '')}`",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By priority: `{json.dumps(counts.get('byReviewPriority') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By decision: `{json.dumps(counts.get('byDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By validation: `{json.dumps(counts.get('byValidationStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By context: `{json.dumps(counts.get('byContextStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Rows",
        "",
    ]
    for row in list(report.get("manualRows") or []):
        lines.extend(
            [
                f"### {row.get('source_decision_row_id', '')}",
                "",
                f"- Paper: `{row.get('paper_id', '')}`",
                f"- Candidate: `{row.get('candidate_text', '')}`",
                f"- Type: `{row.get('section_type', '')}` / `{row.get('review_priority', '')}`",
                f"- Current decision: `{row.get('current_decision', '')}`",
                f"- Validation: `{row.get('validation_status', '')}`",
                f"- Context: `{row.get('review_context_status', '')}` / `{row.get('context_match_method', '')}`",
                f"- Suggestion: `{row.get('review_suggestion', '')}`",
                f"- Required checks: `{'; '.join(str(item) for item in list(row.get('required_review_checks') or []))}`",
                f"- Context before: `{row.get('context_before', '')}`",
                f"- Matched text: `{row.get('matched_text', '')}`",
                f"- Context after: `{row.get('context_after', '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_sectionspan_pdf_offset_selected_review_manual_sheet_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    sheet_path = root / "sectionspan-pdf-offset-selected-review-manual-sheet.json"
    summary_path = root / "sectionspan-pdf-offset-selected-review-manual-sheet-summary.json"
    markdown_path = root / "sectionspan-pdf-offset-selected-review-manual-sheet.md"
    sheet_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_sectionspan_pdf_offset_selected_review_manual_sheet_markdown(report),
        encoding="utf-8",
    )
    return {"sheet": str(sheet_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only selected SectionSpan manual review sheet.")
    parser.add_argument("--sectionspan-pdf-offset-selected-review-decision-template-report", required=True)
    parser.add_argument("--sectionspan-pdf-offset-selected-review-evidence-pack-report", required=True)
    parser.add_argument("--selected-review-decisions-file", required=True)
    parser.add_argument("--sectionspan-pdf-offset-selected-review-decision-file-validation-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_pdf_offset_selected_review_manual_sheet(
        sectionspan_pdf_offset_selected_review_decision_template_report=(
            args.sectionspan_pdf_offset_selected_review_decision_template_report
        ),
        sectionspan_pdf_offset_selected_review_evidence_pack_report=(
            args.sectionspan_pdf_offset_selected_review_evidence_pack_report
        ),
        selected_review_decisions_file=args.selected_review_decisions_file,
        sectionspan_pdf_offset_selected_review_decision_file_validation_report=(
            args.sectionspan_pdf_offset_selected_review_decision_file_validation_report
        ),
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_pdf_offset_selected_review_manual_sheet_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_MANUAL_SHEET_SCHEMA_ID",
    "build_sectionspan_pdf_offset_selected_review_manual_sheet",
    "render_sectionspan_pdf_offset_selected_review_manual_sheet_markdown",
    "write_sectionspan_pdf_offset_selected_review_manual_sheet_reports",
]
