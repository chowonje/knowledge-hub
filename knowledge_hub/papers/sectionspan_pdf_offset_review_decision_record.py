"""Report-only decision record for SectionSpan PDF offset review templates."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-review-decision-record.v1"
)
SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_TEMPLATE_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-review-decision-template.v1"
)

_VALID_DECISIONS = {
    "approve_for_later_promotion_design": "approved_for_later_promotion_design",
    "reject_keep_candidate_only": "rejected_keep_candidate_only",
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
        return int(value)
    except Exception:
        return 0


def _decision_rows(decisions_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = decisions_report.get("decisions")
    if rows is None:
        rows = decisions_report.get("decisionRows")
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


def _unsafe_flags(template: dict[str, Any], decisions_report: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(template.get("counts") or {})
    gate = dict(template.get("gate") or {})
    policy = dict(template.get("policy") or {})
    if template.get("schema") != SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_TEMPLATE_SCHEMA_ID:
        flags.append("sectionspan_pdf_offset_review_decision_template_schema_mismatch")
    if template.get("status") == "blocked":
        flags.append("sectionspan_pdf_offset_review_decision_template_blocked")
    for key in ("strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"decisionTemplate_{key}_nonzero")
    for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(gate.get(key)):
            flags.append(f"decisionTemplate_{key}_true")
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
            flags.append(f"decisionTemplate_{key}_true")

    seen: set[str] = set()
    template_ids = {str(row.get("decision_row_id") or "") for row in list(template.get("decisionRows") or [])}
    for item in _decision_rows(decisions_report):
        row_id = _decision_id(item)
        decision = _decision_value(item)
        if not row_id:
            flags.append("review_decision_row_id_missing")
            continue
        if row_id in seen:
            flags.append("review_decision_duplicate_row_id")
        seen.add(row_id)
        if row_id not in template_ids:
            flags.append("review_decision_unknown_template_row_id")
        if decision not in _VALID_DECISIONS:
            flags.append("review_decision_invalid_value")
    return list(dict.fromkeys(flags))


def _decision_map(decisions_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for item in _decision_rows(decisions_report):
        row_id = _decision_id(item)
        decision = _decision_value(item)
        if row_id and decision in _VALID_DECISIONS and row_id not in mapped:
            mapped[row_id] = dict(item)
    return mapped


def _record_row(index: int, template_row: dict[str, Any], decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row_id = str(template_row.get("decision_row_id") or "")
    decision_item = decisions.get(row_id) or {}
    raw_decision = _decision_value(decision_item)
    recorded_decision = _VALID_DECISIONS.get(raw_decision, "needs_review")
    strict_blockers = [
        "decision_record_only",
        "strict_promotion_requires_later_explicit_apply_tranche",
        "runtime_promotion_disabled_for_tranche",
    ]
    if recorded_decision == "needs_review":
        strict_blockers.append("human_review_decision_missing")
    elif recorded_decision == "approved_for_later_promotion_design":
        strict_blockers.append("approval_is_for_later_design_only_not_runtime_evidence")
    elif recorded_decision == "rejected_keep_candidate_only":
        strict_blockers.append("human_review_rejected")
    return {
        "record_row_id": f"sectionspan-pdf-offset-review-decision-record:{index:04d}",
        "source_decision_row_id": row_id,
        "source_gate_row_id": str(template_row.get("source_gate_row_id") or ""),
        "source_review_card_id": str(template_row.get("source_review_card_id") or ""),
        "source_sectionspan_candidate_id": str(template_row.get("source_sectionspan_candidate_id") or ""),
        "paper_id": str(template_row.get("paper_id") or ""),
        "candidate_text": str(template_row.get("candidate_text") or ""),
        "section_type": str(template_row.get("section_type") or ""),
        "section_level": _safe_int(template_row.get("section_level")),
        "canonical_span": dict(template_row.get("canonical_span") or {}),
        "original_pdf_span": dict(template_row.get("original_pdf_span") or {}),
        "page_agreement": bool(template_row.get("page_agreement")),
        "source_hash_agreement": bool(template_row.get("source_hash_agreement")),
        "recorded_decision": recorded_decision,
        "decision_scope": "decision_record_only_no_runtime_or_strict_promotion",
        "reviewer": str(decision_item.get("reviewer") or ""),
        "notes": str(decision_item.get("notes") or ""),
        "evidence_tier": "sectionspan_pdf_offset_review_decision_record_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": [
            "decision_record_rows_are_not_strict_evidence",
            "decision_record_rows_do_not_authorize_runtime_use",
            "later_explicit_apply_tranche_required_for_any_promotion",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    return {
        "recordRows": len(rows),
        "needsReviewRows": sum(1 for item in rows if item.get("recorded_decision") == "needs_review"),
        "approvedForLaterPromotionDesignRows": sum(
            1 for item in rows if item.get("recorded_decision") == "approved_for_later_promotion_design"
        ),
        "rejectedRows": sum(1 for item in rows if item.get("recorded_decision") == "rejected_keep_candidate_only"),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in rows)),
        "bySectionType": dict(Counter(str(item.get("section_type") or "") for item in rows)),
        "byDecision": dict(Counter(str(item.get("recorded_decision") or "") for item in rows)),
    }


def build_sectionspan_pdf_offset_review_decision_record(
    *,
    sectionspan_pdf_offset_review_decision_template_report: str | Path,
    review_decisions_report: str | Path | None = None,
) -> dict[str, Any]:
    """Build a report-only record from a SectionSpan decision template and optional decisions."""

    template_path = Path(str(sectionspan_pdf_offset_review_decision_template_report)).expanduser()
    decisions_path = Path(str(review_decisions_report)).expanduser() if review_decisions_report else None
    template = _read_json(template_path)
    decisions_report = _read_json(decisions_path) if decisions_path else {}
    unsafe_flags = _unsafe_flags(template, decisions_report)
    decisions = _decision_map(decisions_report)
    template_rows = [dict(item) for item in list(template.get("decisionRows") or []) if isinstance(item, dict)]
    rows = [_record_row(index, row, decisions) for index, row in enumerate(template_rows, start=1)]
    counts = _counts(rows, unsafe_flags)
    needs_review = _safe_int(counts.get("needsReviewRows"))
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif needs_review:
        status = "decision_record_required"
        decision = "manual_review_decisions_still_required"
    else:
        status = "decision_recorded"
        decision = "manual_review_decisions_recorded_non_strict"
    return {
        "schema": SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_RECORD_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "sectionspanPdfOffsetReviewDecisionTemplateReport": str(template_path),
            "sectionspanPdfOffsetReviewDecisionTemplateSchema": str(template.get("schema") or ""),
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
            "recommendedNextTranche": "manual_sectionspan_pdf_offset_review_decision_recording"
            if needs_review
            else "sectionspan_promotion_apply_design_requires_explicit_approval",
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
            "decision_records_are_not_strict_evidence",
            "approved_rows_only_authorize_later_design_review_not_runtime_promotion",
            "strict_promotion_requires_a_separate_explicit_apply_tranche",
        ],
        "decisionRecords": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_sectionspan_pdf_offset_review_decision_record_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan PDF Offset Review Decision Record",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Record rows: `{int(counts.get('recordRows') or 0)}`",
        f"- Needs review: `{int(counts.get('needsReviewRows') or 0)}`",
        f"- Approved for later promotion design: `{int(counts.get('approvedForLaterPromotionDesignRows') or 0)}`",
        f"- Rejected: `{int(counts.get('rejectedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This record is report-only. It does not create strict evidence, runtime citations, parser routing, canonical parsed artifacts, DB mutations, reindex, reembed, or answer integration.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By decision: `{json.dumps(counts.get('byDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_sectionspan_pdf_offset_review_decision_record_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / "sectionspan-pdf-offset-review-decision-record.json"
    summary_path = root / "sectionspan-pdf-offset-review-decision-record-summary.json"
    markdown_path = root / "sectionspan-pdf-offset-review-decision-record.md"
    record_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sectionspan_pdf_offset_review_decision_record_markdown(report), encoding="utf-8")
    return {"record": str(record_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only SectionSpan PDF offset review decision record.")
    parser.add_argument("--sectionspan-pdf-offset-review-decision-template-report", required=True)
    parser.add_argument("--review-decisions-report", default="")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_pdf_offset_review_decision_record(
        sectionspan_pdf_offset_review_decision_template_report=args.sectionspan_pdf_offset_review_decision_template_report,
        review_decisions_report=args.review_decisions_report or None,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_pdf_offset_review_decision_record_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_RECORD_SCHEMA_ID",
    "build_sectionspan_pdf_offset_review_decision_record",
    "render_sectionspan_pdf_offset_review_decision_record_markdown",
    "write_sectionspan_pdf_offset_review_decision_record_reports",
]
