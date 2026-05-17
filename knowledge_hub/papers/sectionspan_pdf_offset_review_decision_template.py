"""Report-only decision template for SectionSpan PDF offset human review."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_TEMPLATE_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-review-decision-template.v1"
)
SECTIONSPAN_PDF_OFFSET_HUMAN_REVIEW_GATE_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-human-review-gate.v1"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
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


def _unsafe_flags(gate_report: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(gate_report.get("counts") or {})
    gate = dict(gate_report.get("gate") or {})
    policy = dict(gate_report.get("policy") or {})
    if gate_report.get("schema") != SECTIONSPAN_PDF_OFFSET_HUMAN_REVIEW_GATE_SCHEMA_ID:
        flags.append("sectionspan_pdf_offset_human_review_gate_schema_mismatch")
    for key in ("strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"humanReviewGate_{key}_nonzero")
    for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(gate.get(key)):
            flags.append(f"humanReviewGate_{key}_true")
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
            flags.append(f"humanReviewGate_{key}_true")
    if gate_report.get("status") == "blocked":
        flags.append("sectionspan_pdf_offset_human_review_gate_blocked")
    return list(dict.fromkeys(flags))


def _decision_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    original_span = dict(row.get("original_pdf_span") or {})
    return {
        "decision_row_id": f"sectionspan-pdf-offset-review-decision:{index:04d}",
        "source_gate_row_id": str(row.get("gate_row_id") or ""),
        "source_review_card_id": str(row.get("source_review_card_id") or ""),
        "source_sectionspan_candidate_id": str(row.get("source_sectionspan_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "candidate_text": str(row.get("candidate_text") or ""),
        "section_type": str(row.get("section_type") or ""),
        "section_level": _safe_int(row.get("section_level")),
        "canonical_span": dict(row.get("canonical_span") or {}),
        "original_pdf_span": original_span,
        "page_agreement": bool(row.get("page_agreement")),
        "source_hash_agreement": bool(row.get("source_hash_agreement")),
        "match_method": str(original_span.get("matchMethod") or ""),
        "match_confidence": original_span.get("matchConfidence"),
        "default_decision": "needs_review",
        "allowed_decisions": [
            "needs_review",
            "approve_for_later_promotion_design",
            "reject_keep_candidate_only",
        ],
        "required_review_checks": [
            "confirm_heading_text_matches_original_pdf_at_recorded_page_and_offset",
            "confirm_section_boundary_is_not_title_or_table_of_contents_navigation",
            "confirm_page_and_source_hash_agreement_are_expected",
            "record_rejection_if_boundary_is_ambiguous_or_overbroad",
        ],
        "decision_scope": "template_only_no_runtime_or_strict_promotion",
        "evidence_tier": "sectionspan_pdf_offset_review_decision_template_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "decision_template_only",
            "human_review_decision_not_recorded",
            "strict_promotion_requires_later_explicit_apply_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "template_rows_are_not_human_review_decisions",
            "template_rows_do_not_authorize_runtime_use",
            "template_rows_do_not_create_strict_evidence",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    return {
        "templateRows": len(rows),
        "pendingDecisionRows": len(rows),
        "approvedRows": 0,
        "rejectedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in rows)),
        "bySectionType": dict(Counter(str(item.get("section_type") or "") for item in rows)),
        "byMatchMethod": dict(Counter(str(item.get("match_method") or "") for item in rows)),
    }


def build_sectionspan_pdf_offset_review_decision_template(
    *,
    sectionspan_pdf_offset_human_review_gate_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only decision template for pending SectionSpan review rows."""

    gate_path = Path(str(sectionspan_pdf_offset_human_review_gate_report)).expanduser()
    gate_report = _read_json(gate_path)
    unsafe_flags = _unsafe_flags(gate_report)
    pending_rows = [
        dict(item)
        for item in list(gate_report.get("gateRows") or [])
        if isinstance(item, dict) and item.get("human_review_status") == "pending_human_review"
    ]
    rows = [_decision_row(index, row) for index, row in enumerate(pending_rows, start=1)]
    counts = _counts(rows, unsafe_flags)
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif rows:
        status = "decision_template_ready"
        decision = "manual_review_decision_template_ready"
    else:
        status = "no_pending_decisions"
        decision = "no_pending_human_review_rows"
    return {
        "schema": SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_TEMPLATE_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "sectionspanPdfOffsetHumanReviewGateReport": str(gate_path),
            "sectionspanPdfOffsetHumanReviewGateSchema": str(gate_report.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "decisionTemplateReady": bool(rows) and not unsafe_flags,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_sectionspan_pdf_offset_review_decision_recording"
            if rows
            else "sectionspan_review_gate_refresh",
        },
        "policy": {
            "reportOnly": True,
            "decisionTemplateOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "decision_template_rows_are_not_human_review_decisions",
            "decision_template_rows_do_not_authorize_strict_or_runtime_evidence",
            "approval_requires_a_separate_review_decision_file_and_later_apply_tranche",
        ],
        "decisionRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_sectionspan_pdf_offset_review_decision_template_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan PDF Offset Review Decision Template",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Template rows: `{int(counts.get('templateRows') or 0)}`",
        f"- Pending decisions: `{int(counts.get('pendingDecisionRows') or 0)}`",
        f"- Approved rows: `{int(counts.get('approvedRows') or 0)}`",
        f"- Rejected rows: `{int(counts.get('rejectedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This template is a review worksheet only. It does not record approvals, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By match method: `{json.dumps(counts.get('byMatchMethod') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_sectionspan_pdf_offset_review_decision_template_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    template_path = root / "sectionspan-pdf-offset-review-decision-template.json"
    summary_path = root / "sectionspan-pdf-offset-review-decision-template-summary.json"
    markdown_path = root / "sectionspan-pdf-offset-review-decision-template.md"
    template_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sectionspan_pdf_offset_review_decision_template_markdown(report), encoding="utf-8")
    return {"template": str(template_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only SectionSpan PDF offset review decision template.")
    parser.add_argument("--sectionspan-pdf-offset-human-review-gate-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_pdf_offset_review_decision_template(
        sectionspan_pdf_offset_human_review_gate_report=args.sectionspan_pdf_offset_human_review_gate_report
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_pdf_offset_review_decision_template_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_TEMPLATE_SCHEMA_ID",
    "build_sectionspan_pdf_offset_review_decision_template",
    "render_sectionspan_pdf_offset_review_decision_template_markdown",
    "write_sectionspan_pdf_offset_review_decision_template_reports",
]
