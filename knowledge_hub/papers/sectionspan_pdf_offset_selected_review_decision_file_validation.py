"""Report-only validation for selected SectionSpan review decision files.

This helper validates a human-edited selected SectionSpan decision file before
it is consumed by the selected decision-record helper. It does not record
decisions, create strict evidence, route parsers, write canonical artifacts,
mutate DB state, reindex, reembed, or change answer behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_VALIDATION_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-file-validation.v1"
)
SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_TEMPLATE_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-template.v1"
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


def _template_rows(template: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(template.get("decisionRows") or []) if isinstance(item, dict)]


def _decision_rows(decision_file: dict[str, Any]) -> list[dict[str, Any]]:
    rows = decision_file.get("decisions")
    if rows is None:
        rows = decision_file.get("decisionRows")
    if rows is None:
        rows = decision_file.get("reviewDecisions")
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


def _template_keys(row: dict[str, Any]) -> set[str]:
    return {
        str(row.get("decision_row_id") or ""),
        str(row.get("source_decision_row_id") or ""),
    } - {""}


def _template_key_map(template: dict[str, Any]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for row in _template_rows(template):
        primary = str(row.get("decision_row_id") or "")
        for key in _template_keys(row):
            mapped[key] = primary
    return mapped


def _unsafe_template_flags(template: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(template.get("counts") or {})
    gate = dict(template.get("gate") or {})
    policy = dict(template.get("policy") or {})
    if template.get("schema") != SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_TEMPLATE_SCHEMA_ID:
        flags.append("sectionspan_pdf_offset_selected_review_decision_template_schema_mismatch")
    if str(template.get("status") or "") == "blocked":
        flags.append("sectionspan_pdf_offset_selected_review_decision_template_blocked")
    for key in ("approvedRows", "rejectedRows", "strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"selectedDecisionTemplate_{key}_nonzero")
    for key in (
        "humanReviewComplete",
        "strictEvidenceReady",
        "parserRoutingReady",
        "answerIntegrationReady",
        "runtimePromotionAllowed",
    ):
        if bool(gate.get(key)):
            flags.append(f"selectedDecisionTemplate_{key}_true")
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
            flags.append(f"selectedDecisionTemplate_{key}_true")
    return list(dict.fromkeys(flags))


def _submitted_decisions(
    template: dict[str, Any],
    decision_file: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    template_keys = _template_key_map(template)
    mapped: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    seen: set[str] = set()
    for item in _decision_rows(decision_file):
        raw_id = _decision_id(item)
        if not raw_id:
            errors.append("selected_review_decision_row_id_missing")
            continue
        primary_id = template_keys.get(raw_id, "")
        if not primary_id:
            errors.append("selected_review_decision_unknown_template_row_id")
            continue
        if primary_id in seen:
            errors.append("selected_review_decision_duplicate_row_id")
        seen.add(primary_id)
        if primary_id not in mapped:
            mapped[primary_id] = dict(item)
    return mapped, list(dict.fromkeys(errors))


def _allowed_decisions(template_row: dict[str, Any]) -> list[str]:
    allowed = [str(item) for item in list(template_row.get("allowed_decisions") or []) if item]
    if not allowed:
        allowed = ["needs_review", "approve_for_later_promotion_design", "reject_keep_candidate_only"]
    if "needs_review" not in allowed:
        allowed = ["needs_review", *allowed]
    return allowed


def _validation_row(index: int, template_row: dict[str, Any], submitted: dict[str, Any] | None) -> dict[str, Any]:
    row_id = str(template_row.get("decision_row_id") or "")
    allowed = _allowed_decisions(template_row)
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
            errors.append("selected_review_decision_not_allowed")
        if submitted_decision != "needs_review" and not reviewer:
            errors.append("reviewer_required_for_non_needs_review_decision")
        if submitted_decision != "needs_review" and not notes:
            errors.append("notes_required_for_non_needs_review_decision")
        validation_status = "valid" if not errors else "invalid"
    return {
        "validation_row_id": f"sectionspan-pdf-offset-selected-review-decision-file-validation:{index:04d}",
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
        "allowed_decisions": allowed,
        "submitted_decision": submitted_decision,
        "reviewer": reviewer,
        "notes": notes,
        "validation_status": validation_status,
        "validation_errors": errors,
        "decision_scope": "sectionspan_pdf_offset_selected_review_decision_file_validation_only_no_runtime_or_strict_promotion",
        "evidence_tier": "sectionspan_pdf_offset_selected_review_decision_file_validation_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "selected_review_decision_file_validation_only",
            "decisions_not_recorded_by_validation_report",
            "strict_promotion_requires_later_explicit_apply_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "validation_rows_are_not_human_review_decision_records",
            "validation_rows_do_not_authorize_runtime_use",
            "validation_rows_do_not_create_strict_evidence",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str], file_errors: list[str]) -> dict[str, Any]:
    by_status = Counter(str(row.get("validation_status") or "") for row in rows)
    by_decision = Counter(str(row.get("submitted_decision") or "") for row in rows)
    by_paper = Counter(str(row.get("paper_id") or "") for row in rows)
    by_section_type = Counter(str(row.get("section_type") or "") for row in rows)
    by_review_priority = Counter(str(row.get("review_priority") or "") for row in rows)
    return {
        "validationRows": len(rows),
        "validRows": by_status.get("valid", 0),
        "invalidRows": by_status.get("invalid", 0),
        "missingRows": by_status.get("missing", 0),
        "needsReviewRows": by_decision.get("needs_review", 0),
        "submittedApprovedForLaterPromotionDesignRows": by_decision.get("approve_for_later_promotion_design", 0),
        "submittedRejectedRows": by_decision.get("reject_keep_candidate_only", 0),
        "nonNeedsReviewRows": len(rows) - by_decision.get("needs_review", 0),
        "fileErrorCount": len(file_errors),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byStatus": dict(by_status),
        "byDecision": dict(by_decision),
        "byPaper": dict(by_paper),
        "bySectionType": dict(by_section_type),
        "byReviewPriority": dict(by_review_priority),
    }


def build_sectionspan_pdf_offset_selected_review_decision_file_validation(
    *,
    sectionspan_pdf_offset_selected_review_decision_template_report: str | Path,
    review_decisions_report: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a selected SectionSpan decision file without recording it."""

    template_path = Path(str(sectionspan_pdf_offset_selected_review_decision_template_report)).expanduser()
    decisions_path = Path(str(review_decisions_report)).expanduser() if review_decisions_report else None
    template = _read_json(template_path)
    decision_file = _read_json(decisions_path) if decisions_path else {}
    submitted, file_errors = _submitted_decisions(template, decision_file)
    unsafe_flags = _unsafe_template_flags(template)
    rows = [
        _validation_row(index, row, submitted.get(str(row.get("decision_row_id") or "")))
        for index, row in enumerate(_template_rows(template), start=1)
    ]
    counts = _counts(rows, unsafe_flags, file_errors)
    if unsafe_flags or file_errors:
        status = "blocked"
        decision = "blocked"
    elif not decisions_path:
        status = "selected_decision_file_required"
        decision = "manual_selected_review_decision_file_missing"
    elif _safe_int(counts.get("missingRows")) or _safe_int(counts.get("invalidRows")):
        status = "selected_decision_file_incomplete"
        decision = "manual_selected_review_decision_file_incomplete"
    else:
        status = "selected_decision_file_validated"
        decision = "manual_selected_review_decision_file_validated_non_runtime"
    return {
        "schema": SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_VALIDATION_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "sectionspanPdfOffsetSelectedReviewDecisionTemplateReport": str(template_path),
            "sectionspanPdfOffsetSelectedReviewDecisionTemplateSchema": str(template.get("schema") or ""),
            "reviewDecisionsReport": str(decisions_path or ""),
            "reviewDecisionInputRows": len(_decision_rows(decision_file)),
        },
        "counts": counts,
        "gate": {
            "selectedDecisionFileValidationReady": bool(rows) and not unsafe_flags,
            "selectedDecisionFileComplete": status == "selected_decision_file_validated",
            "containsRecordedDecisions": False,
            "humanReviewRecordComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "fileValidationErrors": file_errors,
            "recommendedNextTranche": "manual_fill_selected_sectionspan_review_decision_file"
            if status != "selected_decision_file_validated"
            else "sectionspan_selected_decision_record_from_validated_file_requires_explicit_review",
        },
        "policy": {
            "reportOnly": True,
            "selectedDecisionFileValidationOnly": True,
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
            "approved_rows_are_for_later_promotion_design_only",
            "strict_or_runtime_promotion_requires_a_separate_explicit_apply_tranche",
        ],
        "validationRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_sectionspan_pdf_offset_selected_review_decision_file_validation_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan PDF Offset Selected Review Decision File Validation",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Validation rows: `{int(counts.get('validationRows') or 0)}`",
        f"- Valid rows: `{int(counts.get('validRows') or 0)}`",
        f"- Missing rows: `{int(counts.get('missingRows') or 0)}`",
        f"- Invalid rows: `{int(counts.get('invalidRows') or 0)}`",
        f"- `needs_review` rows: `{int(counts.get('needsReviewRows') or 0)}`",
        f"- Approved for later promotion design rows: `{int(counts.get('submittedApprovedForLaterPromotionDesignRows') or 0)}`",
        f"- Rejected rows: `{int(counts.get('submittedRejectedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This validation report is report-only. It does not record human review decisions, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By status: `{json.dumps(counts.get('byStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By decision: `{json.dumps(counts.get('byDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By priority: `{json.dumps(counts.get('byReviewPriority') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_sectionspan_pdf_offset_selected_review_decision_file_validation_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    validation_path = root / "sectionspan-pdf-offset-selected-review-decision-file-validation.json"
    summary_path = root / "sectionspan-pdf-offset-selected-review-decision-file-validation-summary.json"
    markdown_path = root / "sectionspan-pdf-offset-selected-review-decision-file-validation.md"
    validation_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_sectionspan_pdf_offset_selected_review_decision_file_validation_markdown(report),
        encoding="utf-8",
    )
    return {"validation": str(validation_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Validate a SectionSpan selected-review decision file without applying it."
    )
    parser.add_argument("--sectionspan-pdf-offset-selected-review-decision-template-report", required=True)
    parser.add_argument("--review-decisions-report", default="")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_pdf_offset_selected_review_decision_file_validation(
        sectionspan_pdf_offset_selected_review_decision_template_report=(
            args.sectionspan_pdf_offset_selected_review_decision_template_report
        ),
        review_decisions_report=args.review_decisions_report or None,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_pdf_offset_selected_review_decision_file_validation_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_VALIDATION_SCHEMA_ID",
    "build_sectionspan_pdf_offset_selected_review_decision_file_validation",
    "render_sectionspan_pdf_offset_selected_review_decision_file_validation_markdown",
    "write_sectionspan_pdf_offset_selected_review_decision_file_validation_reports",
]
