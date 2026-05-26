"""Report-only decision proposal for selected SectionSpan PDF offset review rows."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_PROPOSAL_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-proposal.v1"
)
SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_EVIDENCE_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-evidence-pack.v1"
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


def _unsafe_flags(evidence_pack: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(evidence_pack.get("counts") or {})
    gate = dict(evidence_pack.get("gate") or {})
    policy = dict(evidence_pack.get("policy") or {})
    if evidence_pack.get("schema") != SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_EVIDENCE_PACK_SCHEMA_ID:
        flags.append("sectionspan_pdf_offset_selected_review_evidence_pack_schema_mismatch")
    if evidence_pack.get("status") == "blocked":
        flags.append("sectionspan_pdf_offset_selected_review_evidence_pack_blocked")
    for key in ("strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"selectedReviewEvidencePack_{key}_nonzero")
    for key in ("humanReviewComplete", "strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(gate.get(key)):
            flags.append(f"selectedReviewEvidencePack_{key}_true")
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
            flags.append(f"selectedReviewEvidencePack_{key}_true")
    return list(dict.fromkeys(flags))


def _proposal_decision(row: dict[str, Any]) -> tuple[str, str]:
    if (
        row.get("review_context_status") == "review_context_ready"
        and row.get("page_text_match")
        and row.get("review_suggestion") == "approve_for_later_promotion_design"
    ):
        return (
            "approve_for_later_promotion_design",
            "original_pdf_context_ready_and_evidence_pack_suggests_later_promotion_design",
        )
    return ("needs_review", "manual_review_required_or_context_not_ready")


def _proposal_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    proposed_decision, proposed_reason = _proposal_decision(row)
    return {
        "proposal_row_id": f"sectionspan-pdf-offset-selected-review-decision-proposal:{index:04d}",
        "source_evidence_row_id": str(row.get("review_evidence_row_id") or ""),
        "source_decision_row_id": str(row.get("source_decision_row_id") or ""),
        "source_selected_review_card_id": str(row.get("source_selected_review_card_id") or ""),
        "source_sectionspan_candidate_id": str(row.get("source_sectionspan_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "candidate_text": str(row.get("candidate_text") or ""),
        "section_type": str(row.get("section_type") or ""),
        "section_level": _safe_int(row.get("section_level")),
        "review_priority": str(row.get("review_priority") or ""),
        "review_context_status": str(row.get("review_context_status") or ""),
        "context_match_method": str(row.get("context_match_method") or ""),
        "matched_text": str(row.get("matched_text") or ""),
        "canonical_span": dict(row.get("canonical_span") or {}),
        "original_pdf_span": dict(row.get("original_pdf_span") or {}),
        "sourceContentHash": str(row.get("sourceContentHash") or ""),
        "proposed_decision": proposed_decision,
        "proposed_decision_reason": proposed_reason,
        "human_decision_required": True,
        "accepted_as_human_decision": False,
        "decision_record_input_hint": {
            "source_decision_row_id": str(row.get("source_decision_row_id") or ""),
            "decision": proposed_decision,
            "reviewer": "",
            "notes": "",
        },
        "evidence_tier": "sectionspan_pdf_offset_selected_review_decision_proposal_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "decision_proposal_only",
            "human_review_decision_not_recorded",
            "strict_promotion_requires_later_explicit_apply_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "decision_proposals_are_not_human_review_decisions",
            "decision_proposals_do_not_authorize_runtime_use",
            "decision_proposals_do_not_create_strict_evidence",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    return {
        "proposalRows": len(rows),
        "proposedApproveForLaterPromotionDesignRows": sum(
            1 for item in rows if item.get("proposed_decision") == "approve_for_later_promotion_design"
        ),
        "proposedNeedsReviewRows": sum(1 for item in rows if item.get("proposed_decision") == "needs_review"),
        "acceptedHumanDecisionRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in rows)),
        "bySectionType": dict(Counter(str(item.get("section_type") or "") for item in rows)),
        "byReviewPriority": dict(Counter(str(item.get("review_priority") or "") for item in rows)),
        "byProposedDecision": dict(Counter(str(item.get("proposed_decision") or "") for item in rows)),
    }


def build_sectionspan_pdf_offset_selected_review_decision_proposal(
    *,
    sectionspan_pdf_offset_selected_review_evidence_pack_report: str | Path,
) -> dict[str, Any]:
    """Build a non-binding decision proposal from selected review evidence."""

    evidence_path = Path(str(sectionspan_pdf_offset_selected_review_evidence_pack_report)).expanduser()
    evidence_pack = _read_json(evidence_path)
    unsafe_flags = _unsafe_flags(evidence_pack)
    evidence_rows = [dict(item) for item in list(evidence_pack.get("evidenceRows") or []) if isinstance(item, dict)]
    rows = [_proposal_row(index, row) for index, row in enumerate(evidence_rows, start=1)]
    counts = _counts(rows, unsafe_flags)
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif rows:
        status = "decision_proposal_ready"
        decision = "manual_decision_file_required"
    else:
        status = "no_proposal_rows"
        decision = "no_selected_review_evidence_rows_for_proposal"
    return {
        "schema": SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_PROPOSAL_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "sectionspanPdfOffsetSelectedReviewEvidencePackReport": str(evidence_path),
            "sectionspanPdfOffsetSelectedReviewEvidencePackSchema": str(evidence_pack.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "decisionProposalReady": bool(rows) and not unsafe_flags,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_record_selected_sectionspan_review_decisions",
        },
        "policy": {
            "reportOnly": True,
            "decisionProposalOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "decision_proposals_are_not_human_review_decisions",
            "proposal_rows_are_not_named_decisions_and_are_not_consumed_by_decision_record_helpers",
            "approval_requires_a_separate_review_decision_file_and_later_apply_tranche",
        ],
        "proposalRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_sectionspan_pdf_offset_selected_review_decision_proposal_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan PDF Offset Selected Review Decision Proposal",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Proposal rows: `{int(counts.get('proposalRows') or 0)}`",
        f"- Proposed approve-for-later-promotion-design rows: `{int(counts.get('proposedApproveForLaterPromotionDesignRows') or 0)}`",
        f"- Proposed needs-review rows: `{int(counts.get('proposedNeedsReviewRows') or 0)}`",
        f"- Accepted human decision rows: `{int(counts.get('acceptedHumanDecisionRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This proposal is not a human decision file. It does not create strict evidence, runtime citations, parser routing, canonical parsed artifacts, DB mutations, reindex, reembed, or answer integration.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By priority: `{json.dumps(counts.get('byReviewPriority') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By proposed decision: `{json.dumps(counts.get('byProposedDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_sectionspan_pdf_offset_selected_review_decision_proposal_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "sectionspan-pdf-offset-selected-review-decision-proposal.json"
    summary_path = root / "sectionspan-pdf-offset-selected-review-decision-proposal-summary.json"
    markdown_path = root / "sectionspan-pdf-offset-selected-review-decision-proposal.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sectionspan_pdf_offset_selected_review_decision_proposal_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only SectionSpan PDF offset selected review decision proposal.")
    parser.add_argument("--sectionspan-pdf-offset-selected-review-evidence-pack-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_pdf_offset_selected_review_decision_proposal(
        sectionspan_pdf_offset_selected_review_evidence_pack_report=(
            args.sectionspan_pdf_offset_selected_review_evidence_pack_report
        )
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_pdf_offset_selected_review_decision_proposal_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_PROPOSAL_SCHEMA_ID",
    "build_sectionspan_pdf_offset_selected_review_decision_proposal",
    "render_sectionspan_pdf_offset_selected_review_decision_proposal_markdown",
    "write_sectionspan_pdf_offset_selected_review_decision_proposal_reports",
]
