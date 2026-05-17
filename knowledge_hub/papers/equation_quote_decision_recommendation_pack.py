"""Report-only nonbinding recommendations for EquationQuote review decisions.

This helper reads the EquationQuote decision next-action brief and proposes
human-review decision values. It never writes the editable decision file and its
rows are intentionally not accepted as decisions by the decision-record helper.
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


EQUATION_QUOTE_DECISION_RECOMMENDATION_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.equation-quote-decision-recommendation-pack.v1"
)
EQUATION_QUOTE_DECISION_NEXT_ACTION_BRIEF_SCHEMA_ID = (
    "knowledge-hub.paper.equation-quote-decision-next-action-brief.v1"
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
        return int(value or 0)
    except Exception:
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _unsafe_flags(next_action_brief: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(next_action_brief.get("counts") or {})
    gate = dict(next_action_brief.get("gate") or {})
    policy = dict(next_action_brief.get("policy") or {})
    if next_action_brief.get("schema") != EQUATION_QUOTE_DECISION_NEXT_ACTION_BRIEF_SCHEMA_ID:
        flags.append("equation_quote_decision_next_action_brief_schema_mismatch")
    status = str(next_action_brief.get("status") or "")
    if status == "blocked":
        flags.append("equation_quote_decision_next_action_brief_blocked")
    if status and status not in {"manual_review_required", "manual_review_recorded_non_runtime"}:
        flags.append("equation_quote_decision_next_action_brief_status_unexpected")
    for key in (
        "sourceSpanCreatedRows",
        "originalPdfOffsetRecoveredRows",
        "equationSemanticsInterpretedRows",
        "strictEligibleRows",
        "citationGradeRows",
        "runtimeEvidenceRows",
    ):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"decisionNextActionBrief_{key}_nonzero")
    for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(gate.get(key)):
            flags.append(f"decisionNextActionBrief_{key}_true")
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
            flags.append(f"decisionNextActionBrief_{key}_true")
    return list(dict.fromkeys(flags))


def _recommended_decision(row: dict[str, Any]) -> tuple[str, str, str]:
    action_type = str(row.get("action_type") or "")
    action_status = str(row.get("action_status") or "")
    coverage = _safe_float(row.get("best_diagnostic_page_coverage"))
    diagnostic_pages = list(row.get("diagnostic_page_candidates") or [])
    if (
        action_type == "review_diagnostic_page_context"
        and action_status == "ready_for_human_review"
        and coverage >= 0.5
        and diagnostic_pages
    ):
        return (
            "accept_diagnostic_context_for_later_reextract_design",
            "diagnostic_page_context_is_available_but_not_a_source_span_or_runtime_evidence",
            "medium",
        )
    if action_type == "reject_or_reextract_unmatched_equation_quote":
        if _safe_int(row.get("layout_element_count")) or bool(row.get("bbox_available")):
            return (
                "request_equation_quote_reextraction",
                "candidate_has_layout_signal_but_no_canonical_or_original_pdf_source_span",
                "low",
            )
        return (
            "keep_blocked",
            "candidate_lacks_source_span_diagnostic_context_and_reextract_signal",
            "low",
        )
    return ("needs_review", "no_safe_nonbinding_recommendation", "low")


def _recommendation_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    proposed_decision, proposed_reason, confidence = _recommended_decision(row)
    return {
        "recommendation_row_id": f"equation-quote-decision-recommendation:{index:04d}",
        "source_brief_row_id": str(row.get("brief_row_id") or ""),
        "source_review_sheet_row_id": str(row.get("source_review_sheet_row_id") or ""),
        "source_action_card_id": str(row.get("source_action_card_id") or ""),
        "source_equation_quote_candidate_id": str(row.get("source_equation_quote_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "candidate_text": str(row.get("candidate_text") or ""),
        "equation_label": str(row.get("equation_label") or ""),
        "action_type": str(row.get("action_type") or ""),
        "action_status": str(row.get("action_status") or ""),
        "priority": str(row.get("priority") or ""),
        "current_decision": str(row.get("current_decision") or "needs_review"),
        "recommended_decision": proposed_decision,
        "recommended_decision_reason": proposed_reason,
        "recommendation_confidence": confidence,
        "diagnostic_terms": list(row.get("diagnostic_terms") or []),
        "diagnostic_page_candidates": list(row.get("diagnostic_page_candidates") or []),
        "best_diagnostic_page_coverage": _safe_float(row.get("best_diagnostic_page_coverage")),
        "canonical_alignment_status": str(row.get("canonical_alignment_status") or ""),
        "alignment_feasibility_status": str(row.get("alignment_feasibility_status") or ""),
        "pdf_offset_feasibility_status": str(row.get("pdf_offset_feasibility_status") or ""),
        "layout_element_count": _safe_int(row.get("layout_element_count")),
        "bbox_available": bool(row.get("bbox_available")),
        "sourceContentHash": str(row.get("sourceContentHash") or ""),
        "human_decision_required": True,
        "accepted_as_human_decision": False,
        "decision_record_input_hint": {
            "source_review_sheet_row_id": str(row.get("source_review_sheet_row_id") or ""),
            "decision": proposed_decision,
            "reviewer": "",
            "notes": "",
        },
        "source_span_created": False,
        "original_pdf_offset_recovered": False,
        "equation_semantics_interpreted": False,
        "evidence_tier": "equation_quote_decision_recommendation_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "equation_quote_decision_recommendation_only",
            "human_review_decision_not_recorded",
            "source_span_not_created",
            "equation_semantics_not_interpreted",
            "strict_promotion_requires_later_explicit_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "recommendation_rows_are_not_human_review_decisions",
            "recommendation_rows_are_not_consumed_by_decision_record_helpers",
            "recommendation_rows_do_not_create_source_spans",
            "recommendation_rows_do_not_interpret_equations",
            "recommendation_rows_do_not_authorize_runtime_use",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    by_decision = Counter(str(row.get("recommended_decision") or "") for row in rows)
    return {
        "recommendationRows": len(rows),
        "proposedAcceptDiagnosticContextRows": by_decision.get(
            "accept_diagnostic_context_for_later_reextract_design", 0
        ),
        "proposedRejectRows": by_decision.get("reject_equation_quote_candidate", 0),
        "proposedReextractRequestRows": by_decision.get("request_equation_quote_reextraction", 0),
        "proposedKeepBlockedRows": by_decision.get("keep_blocked", 0),
        "proposedNeedsReviewRows": by_decision.get("needs_review", 0),
        "acceptedHumanDecisionRows": 0,
        "sourceSpanCreatedRows": 0,
        "originalPdfOffsetRecoveredRows": 0,
        "equationSemanticsInterpretedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byPaper": dict(Counter(str(row.get("paper_id") or "") for row in rows)),
        "byActionType": dict(Counter(str(row.get("action_type") or "") for row in rows)),
        "byActionStatus": dict(Counter(str(row.get("action_status") or "") for row in rows)),
        "byRecommendedDecision": dict(by_decision),
        "byRecommendationConfidence": dict(Counter(str(row.get("recommendation_confidence") or "") for row in rows)),
    }


def build_equation_quote_decision_recommendation_pack(
    *,
    equation_quote_decision_next_action_brief_report: str | Path,
) -> dict[str, Any]:
    """Build a nonbinding recommendation pack for EquationQuote decisions."""

    brief_path = Path(str(equation_quote_decision_next_action_brief_report)).expanduser()
    next_action_brief = _read_json(brief_path)
    unsafe_flags = _unsafe_flags(next_action_brief)
    brief_rows = [dict(item) for item in list(next_action_brief.get("briefRows") or []) if isinstance(item, dict)]
    rows = [_recommendation_row(index, row) for index, row in enumerate(brief_rows, start=1)]
    counts = _counts(rows, unsafe_flags)
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif rows:
        status = "recommendation_pack_ready"
        decision = "manual_decision_file_still_required"
    else:
        status = "no_recommendation_rows"
        decision = "no_equation_quote_review_rows_for_recommendation"
    return {
        "schema": EQUATION_QUOTE_DECISION_RECOMMENDATION_PACK_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "equationQuoteDecisionNextActionBriefReport": str(brief_path),
            "equationQuoteDecisionNextActionBriefSchema": str(next_action_brief.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "recommendationPackReady": bool(rows) and not unsafe_flags,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_edit_equation_quote_decision_file",
        },
        "policy": {
            "reportOnly": True,
            "decisionRecommendationOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "recommendation_rows_are_not_human_review_decisions",
            "recommendation_rows_are_not_named_decisions_and_are_not_consumed_by_decision_record_helpers",
            "accepting_diagnostic_context_does_not_create_source_spans_or_runtime_evidence",
            "reextract_or_reject_recommendations_require_manual_reviewer_and_notes",
        ],
        "recommendationRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_equation_quote_decision_recommendation_pack_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# EquationQuote Decision Recommendation Pack",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Recommendation rows: `{int(counts.get('recommendationRows') or 0)}`",
        f"- Proposed accept diagnostic context: `{int(counts.get('proposedAcceptDiagnosticContextRows') or 0)}`",
        f"- Proposed re-extraction requests: `{int(counts.get('proposedReextractRequestRows') or 0)}`",
        f"- Accepted human decision rows: `{int(counts.get('acceptedHumanDecisionRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This recommendation pack is not a human decision file. It does not create source spans, interpret equations, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By action type: `{json.dumps(counts.get('byActionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By recommendation: `{json.dumps(counts.get('byRecommendedDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By confidence: `{json.dumps(counts.get('byRecommendationConfidence') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_equation_quote_decision_recommendation_pack_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "equation-quote-decision-recommendation-pack.json"
    summary_path = root / "equation-quote-decision-recommendation-summary.json"
    markdown_path = root / "equation-quote-decision-recommendation-pack.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_quote_decision_recommendation_pack_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only EquationQuote decision recommendation pack.")
    parser.add_argument("--equation-quote-decision-next-action-brief-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_equation_quote_decision_recommendation_pack(
        equation_quote_decision_next_action_brief_report=args.equation_quote_decision_next_action_brief_report
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_equation_quote_decision_recommendation_pack_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EQUATION_QUOTE_DECISION_RECOMMENDATION_PACK_SCHEMA_ID",
    "build_equation_quote_decision_recommendation_pack",
    "render_equation_quote_decision_recommendation_pack_markdown",
    "write_equation_quote_decision_recommendation_pack_reports",
]
