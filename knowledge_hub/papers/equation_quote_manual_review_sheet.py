"""Report-only manual review sheet for EquationQuote next-action cards.

The sheet turns EquationQuote diagnostic action cards into operator-readable
review rows. It does not record decisions, interpret equations, create source
spans, promote strict evidence, route parsers, write canonical artifacts,
mutate DB state, reindex, reembed, or change answer behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


EQUATION_QUOTE_MANUAL_REVIEW_SHEET_SCHEMA_ID = "knowledge-hub.paper.equation-quote-manual-review-sheet.v1"
EQUATION_QUOTE_NEXT_ACTION_GATE_SCHEMA_ID = "knowledge-hub.paper.equation-quote-next-action-gate.v1"


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


def _unsafe_flags(next_action_gate: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(next_action_gate.get("counts") or {})
    gate = dict(next_action_gate.get("gate") or {})
    policy = dict(next_action_gate.get("policy") or {})

    if next_action_gate.get("schema") != EQUATION_QUOTE_NEXT_ACTION_GATE_SCHEMA_ID:
        flags.append("equation_quote_next_action_gate_schema_mismatch")
    if str(next_action_gate.get("status") or "") == "blocked":
        flags.append("equation_quote_next_action_gate_blocked")

    for key in (
        "sourceSpanCreatedCards",
        "originalPdfOffsetRecoveredCards",
        "equationSemanticsInterpretedCards",
        "strictEligibleCards",
        "citationGradeCards",
        "runtimeEvidenceCards",
    ):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"nextActionGate_{key}_nonzero")

    for key in (
        "sourceSpanCreationReady",
        "equationSemanticsReady",
        "strictEvidenceReady",
        "parserRoutingReady",
        "answerIntegrationReady",
        "runtimePromotionAllowed",
    ):
        if bool(gate.get(key)):
            flags.append(f"nextActionGate_{key}_true")

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
            flags.append(f"nextActionGate_{key}_true")

    return list(dict.fromkeys(flags))


def _allowed_decisions(action_type: str) -> list[str]:
    if action_type == "reject_or_reextract_unmatched_equation_quote":
        return [
            "needs_review",
            "reject_equation_quote_candidate",
            "request_equation_quote_reextraction",
            "keep_blocked",
        ]
    return [
        "needs_review",
        "accept_diagnostic_context_for_later_reextract_design",
        "reject_equation_quote_candidate",
        "request_equation_quote_reextraction",
        "keep_blocked",
    ]


def _recommended_action(action_type: str) -> str:
    if action_type == "reject_or_reextract_unmatched_equation_quote":
        return "reject_or_request_reextraction_for_unmatched_equation_quote"
    return "review_diagnostic_page_context_before_any_later_reextract_or_source_span_design"


def _review_row(index: int, card: dict[str, Any]) -> dict[str, Any]:
    action_type = str(card.get("action_type") or "")
    diagnostic_pages = [dict(item) for item in list(card.get("diagnostic_page_candidates") or []) if isinstance(item, dict)]
    return {
        "review_sheet_row_id": f"equation-quote-manual-review:{index:04d}",
        "source_action_card_id": str(card.get("action_card_id") or ""),
        "source_equation_quote_candidate_id": str(card.get("source_equation_quote_candidate_id") or ""),
        "paper_id": str(card.get("paper_id") or ""),
        "candidate_text": str(card.get("candidate_text") or ""),
        "equation_label": str(card.get("equation_label") or ""),
        "action_type": action_type,
        "action_status": str(card.get("action_status") or ""),
        "priority": str(card.get("priority") or ""),
        "current_decision": "needs_review",
        "allowed_decisions": _allowed_decisions(action_type),
        "recommended_review_action": _recommended_action(action_type),
        "review_prompt": "Inspect the diagnostic context, then leave needs_review unless a human explicitly rejects or requests re-extraction.",
        "canonical_alignment_status": str(card.get("canonical_alignment_status") or ""),
        "canonical_alignment_method": str(card.get("canonical_alignment_method") or ""),
        "alignment_feasibility_status": str(card.get("alignment_feasibility_status") or ""),
        "pdf_offset_feasibility_status": str(card.get("pdf_offset_feasibility_status") or ""),
        "diagnostic_terms": [str(item) for item in list(card.get("diagnostic_terms") or [])],
        "diagnostic_term_coverage": _safe_float(card.get("diagnostic_term_coverage")),
        "diagnostic_page_candidates": diagnostic_pages[:5],
        "best_diagnostic_page_coverage": _safe_float(card.get("best_diagnostic_page_coverage")),
        "layout_element_count": _safe_int(card.get("layout_element_count")),
        "bbox_available": bool(card.get("bbox_available")),
        "sourceContentHash": str(card.get("sourceContentHash") or ""),
        "source_span_created": False,
        "original_pdf_offset_recovered": False,
        "equation_semantics_interpreted": False,
        "quote_only": True,
        "evidence_tier": "equation_quote_manual_review_sheet_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": list(
            dict.fromkeys(
                [
                    "equation_quote_manual_review_sheet_only",
                    *[str(item) for item in list(card.get("strict_blockers") or [])],
                    "strict_promotion_requires_later_explicit_tranche",
                    "runtime_promotion_disabled_for_tranche",
                ]
            )
        ),
        "non_strict_reason": [
            "manual_review_sheet_rows_are_not_decisions",
            "diagnostic_context_does_not_create_source_spans",
            "equation_semantics_are_not_interpreted",
            "manual_review_sheet_rows_do_not_authorize_runtime_use",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    by_action_type = Counter(str(row.get("action_type") or "") for row in rows)
    by_action_status = Counter(str(row.get("action_status") or "") for row in rows)
    by_paper = Counter(str(row.get("paper_id") or "") for row in rows)
    by_decision = Counter(str(row.get("current_decision") or "") for row in rows)
    return {
        "reviewRows": len(rows),
        "needsReviewRows": by_decision.get("needs_review", 0),
        "nonNeedsReviewRows": len(rows) - by_decision.get("needs_review", 0),
        "humanReviewRows": by_action_status.get("ready_for_human_review", 0),
        "blockedRows": by_action_status.get("blocked_no_source_span_or_diagnostic_context", 0),
        "diagnosticPageContextRows": by_action_type.get("review_diagnostic_page_context", 0),
        "unmatchedEquationQuoteRows": by_action_type.get("reject_or_reextract_unmatched_equation_quote", 0),
        "sourceSpanCreatedRows": 0,
        "originalPdfOffsetRecoveredRows": 0,
        "equationSemanticsInterpretedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byPaper": dict(by_paper),
        "byActionType": dict(by_action_type),
        "byActionStatus": dict(by_action_status),
        "byDecision": dict(by_decision),
    }


def build_equation_quote_manual_review_sheet(
    *,
    equation_quote_next_action_gate_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only manual review sheet over EquationQuote action cards."""

    gate_path = Path(str(equation_quote_next_action_gate_report)).expanduser()
    next_action_gate = _read_json(gate_path)
    unsafe_flags = _unsafe_flags(next_action_gate)
    rows = [
        _review_row(index, card)
        for index, card in enumerate(list(next_action_gate.get("actionCards") or []), start=1)
        if isinstance(card, dict)
    ]
    counts = _counts(rows, unsafe_flags)
    status = "blocked" if unsafe_flags else "manual_review_sheet_ready"
    return {
        "schema": EQUATION_QUOTE_MANUAL_REVIEW_SHEET_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "equationQuoteNextActionGateReport": str(gate_path),
            "equationQuoteNextActionGateSchema": str(next_action_gate.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "manualReviewSheetReady": bool(rows) and not unsafe_flags,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "blocked" if unsafe_flags else "equation_quote_manual_review_required",
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_record_equation_quote_next_action_decisions",
        },
        "policy": {
            "reportOnly": True,
            "manualReviewSheetOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "manual_review_sheet_rows_are_not_recorded_decisions",
            "diagnostic_context_does_not_create_source_spans",
            "equation_semantics_are_not_interpreted",
            "strict_or_runtime_promotion_requires_a_separate_explicit_tranche",
        ],
        "reviewRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_equation_quote_manual_review_sheet_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# EquationQuote Manual Review Sheet",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Review rows: `{int(counts.get('reviewRows') or 0)}`",
        f"- `needs_review` rows: `{int(counts.get('needsReviewRows') or 0)}`",
        f"- Diagnostic page-context rows: `{int(counts.get('diagnosticPageContextRows') or 0)}`",
        f"- Unmatched equation quote rows: `{int(counts.get('unmatchedEquationQuoteRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This sheet is local review metadata only. It does not record decisions, create source spans, interpret equations, create strict evidence, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By action type: `{json.dumps(counts.get('byActionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By action status: `{json.dumps(counts.get('byActionStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Rows",
        "",
    ]
    for row in list(report.get("reviewRows") or []):
        text = str(row.get("candidate_text") or "").replace("\n", " ")
        if len(text) > 180:
            text = text[:177] + "..."
        lines.extend(
            [
                f"### {row.get('review_sheet_row_id', '')}",
                "",
                f"- Paper: `{row.get('paper_id', '')}`",
                f"- Candidate: `{row.get('source_equation_quote_candidate_id', '')}`",
                f"- Action type: `{row.get('action_type', '')}`",
                f"- Current decision: `{row.get('current_decision', '')}`",
                f"- Recommended action: `{row.get('recommended_review_action', '')}`",
                f"- Best diagnostic page coverage: `{float(row.get('best_diagnostic_page_coverage') or 0.0):.3f}`",
                f"- Text: {text}",
                "",
            ]
        )
    return "\n".join(lines)


def write_equation_quote_manual_review_sheet_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    sheet_path = root / "equation-quote-manual-review-sheet.json"
    summary_path = root / "equation-quote-manual-review-summary.json"
    markdown_path = root / "equation-quote-manual-review-sheet.md"
    sheet_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_quote_manual_review_sheet_markdown(report), encoding="utf-8")
    return {"sheet": str(sheet_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only EquationQuote manual review sheet.")
    parser.add_argument("--equation-quote-next-action-gate-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_equation_quote_manual_review_sheet(
        equation_quote_next_action_gate_report=args.equation_quote_next_action_gate_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_equation_quote_manual_review_sheet_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EQUATION_QUOTE_MANUAL_REVIEW_SHEET_SCHEMA_ID",
    "build_equation_quote_manual_review_sheet",
    "render_equation_quote_manual_review_sheet_markdown",
    "write_equation_quote_manual_review_sheet_reports",
]
