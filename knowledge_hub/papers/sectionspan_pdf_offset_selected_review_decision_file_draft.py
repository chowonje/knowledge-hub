"""Report-only draft decision file for selected SectionSpan PDF offset rows.

The draft is an editable starting point for human review. Every row defaults to
``needs_review`` and the helper records no approval, rejection, strict evidence,
parser routing, canonical artifact write, DB mutation, reindex, reembed, or
answer integration.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_DRAFT_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-file-draft.v1"
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


def _unsafe_flags(template: dict[str, Any]) -> list[str]:
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


def _draft_row(index: int, template_row: dict[str, Any]) -> dict[str, Any]:
    allowed_decisions = list(template_row.get("allowed_decisions") or [])
    if "needs_review" not in allowed_decisions:
        allowed_decisions = ["needs_review", *allowed_decisions]
    return {
        "draft_row_id": f"sectionspan-pdf-offset-selected-review-decision-file-draft:{index:04d}",
        "source_decision_row_id": str(template_row.get("decision_row_id") or ""),
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
        "allowed_decisions": allowed_decisions,
        "decision": "needs_review",
        "reviewer": "",
        "notes": "",
        "draft_only": True,
        "decision_scope": "sectionspan_pdf_offset_selected_review_decision_file_draft_only_no_runtime_or_strict_promotion",
        "evidence_tier": "sectionspan_pdf_offset_selected_review_decision_file_draft_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "selected_review_decision_file_draft_only",
            "human_review_decision_not_recorded",
            "strict_promotion_requires_later_explicit_apply_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "draft_rows_are_not_human_review_decisions",
            "draft_rows_default_to_needs_review",
            "draft_rows_do_not_authorize_runtime_use",
            "draft_rows_do_not_create_strict_evidence",
        ],
    }


def _decision_file_from_drafts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "draftOnly": True,
        "instructions": [
            "Edit a copy of this file before using it as a selected SectionSpan review decision file.",
            "Keep decision=needs_review unless a human has explicitly approved or rejected the row.",
            "Valid non-needs_review decisions are approve_for_later_promotion_design and reject_keep_candidate_only.",
            "Non-needs_review decisions require reviewer and notes.",
            "Approvals are for later promotion design only and do not authorize strict evidence or runtime use.",
        ],
        "decisions": [
            {
                "source_decision_row_id": str(row.get("source_decision_row_id") or ""),
                "decision": "needs_review",
                "reviewer": "",
                "notes": "",
                "allowed_decisions": list(row.get("allowed_decisions") or []),
                "paper_id": str(row.get("paper_id") or ""),
                "candidate_text": str(row.get("candidate_text") or ""),
                "section_type": str(row.get("section_type") or ""),
                "review_priority": str(row.get("review_priority") or ""),
            }
            for row in rows
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    by_paper = Counter(str(row.get("paper_id") or "") for row in rows)
    by_section_type = Counter(str(row.get("section_type") or "") for row in rows)
    by_review_priority = Counter(str(row.get("review_priority") or "") for row in rows)
    return {
        "draftRows": len(rows),
        "needsReviewRows": len(rows),
        "approvedForLaterPromotionDesignRows": 0,
        "rejectedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byPaper": dict(by_paper),
        "bySectionType": dict(by_section_type),
        "byReviewPriority": dict(by_review_priority),
    }


def build_sectionspan_pdf_offset_selected_review_decision_file_draft(
    *,
    sectionspan_pdf_offset_selected_review_decision_template_report: str | Path,
) -> dict[str, Any]:
    """Build a needs-review-only draft file for selected SectionSpan decisions."""

    template_path = Path(str(sectionspan_pdf_offset_selected_review_decision_template_report)).expanduser()
    template = _read_json(template_path)
    unsafe_flags = _unsafe_flags(template)
    rows = [_draft_row(index, row) for index, row in enumerate(_template_rows(template), start=1)]
    counts = _counts(rows, unsafe_flags)
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif rows:
        status = "selected_decision_file_draft_ready"
        decision = "needs_review_draft_ready_for_manual_edit"
    else:
        status = "no_selected_decision_rows"
        decision = "no_selected_decision_rows"
    return {
        "schema": SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_DRAFT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "sectionspanPdfOffsetSelectedReviewDecisionTemplateReport": str(template_path),
            "sectionspanPdfOffsetSelectedReviewDecisionTemplateSchema": str(template.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "selectedDecisionFileDraftReady": bool(rows) and not unsafe_flags,
            "containsOnlyNeedsReviewDefaults": True,
            "containsApprovedRows": False,
            "containsRejectedRows": False,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_edit_selected_sectionspan_review_decision_file"
            if rows
            else "sectionspan_pdf_offset_selected_review_packet_refresh",
        },
        "policy": {
            "reportOnly": True,
            "selectedDecisionFileDraftOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "draft_rows_are_not_human_review_decisions",
            "draft_decision_file_defaults_every_row_to_needs_review",
            "approved_rows_would_only_authorize_later_design_review_not_runtime_promotion",
            "strict_or_runtime_promotion_requires_a_separate_explicit_apply_tranche",
        ],
        "decisionFileDraft": _decision_file_from_drafts(rows),
        "draftRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_sectionspan_pdf_offset_selected_review_decision_file_draft_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan PDF Offset Selected Review Decision File Draft",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Draft rows: `{int(counts.get('draftRows') or 0)}`",
        f"- `needs_review` rows: `{int(counts.get('needsReviewRows') or 0)}`",
        f"- Approved for later promotion design: `{int(counts.get('approvedForLaterPromotionDesignRows') or 0)}`",
        f"- Rejected rows: `{int(counts.get('rejectedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This draft is an editable starting point only. It does not record human review decisions, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By priority: `{json.dumps(counts.get('byReviewPriority') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_sectionspan_pdf_offset_selected_review_decision_file_draft_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    draft_report_path = root / "sectionspan-pdf-offset-selected-review-decision-file-draft.json"
    decision_file_path = root / "sectionspan-pdf-offset-selected-review-decisions.draft.json"
    summary_path = root / "sectionspan-pdf-offset-selected-review-decision-file-draft-summary.json"
    markdown_path = root / "sectionspan-pdf-offset-selected-review-decision-file-draft.md"
    draft_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    decision_file_path.write_text(
        json.dumps(report.get("decisionFileDraft") or {}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_sectionspan_pdf_offset_selected_review_decision_file_draft_markdown(report),
        encoding="utf-8",
    )
    return {
        "draftReport": str(draft_report_path),
        "decisionFileDraft": str(decision_file_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Generate a needs-review-only SectionSpan selected-review decision file draft."
    )
    parser.add_argument("--sectionspan-pdf-offset-selected-review-decision-template-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_pdf_offset_selected_review_decision_file_draft(
        sectionspan_pdf_offset_selected_review_decision_template_report=(
            args.sectionspan_pdf_offset_selected_review_decision_template_report
        )
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_pdf_offset_selected_review_decision_file_draft_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_DRAFT_SCHEMA_ID",
    "build_sectionspan_pdf_offset_selected_review_decision_file_draft",
    "render_sectionspan_pdf_offset_selected_review_decision_file_draft_markdown",
    "write_sectionspan_pdf_offset_selected_review_decision_file_draft_reports",
]
