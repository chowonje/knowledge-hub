"""Report-only EquationQuote next-action gate.

This helper consolidates EquationQuote alignment and original-PDF offset
feasibility reports into explicit operator next actions. It does not interpret
equations, create source spans, create strict evidence, route parsers, write
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


EQUATION_QUOTE_NEXT_ACTION_GATE_SCHEMA_ID = "knowledge-hub.paper.equation-quote-next-action-gate.v1"
EQUATION_ALIGNMENT_FEASIBILITY_AUDIT_SCHEMA_ID = "knowledge-hub.paper.equation-alignment-feasibility-audit.v1"
EQUATION_QUOTE_PDF_OFFSET_FEASIBILITY_SCHEMA_ID = "knowledge-hub.paper.equation-quote-pdf-offset-feasibility.v1"


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


def _strict_zero_violations(prefix: str, report: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    policy = dict(report.get("policy") or {})
    for key in (
        "canonicalSourceSpanCreatedCandidates",
        "originalPdfOffsetRecoveredRows",
        "sourceSpanCreatedRows",
        "equationSemanticsInterpretedCandidates",
        "equationSemanticsInterpretedRows",
        "strictEligibleCandidates",
        "strictEligibleRows",
        "citationGradeCandidates",
        "citationGradeRows",
        "runtimeEvidenceCandidates",
        "runtimeEvidenceRows",
    ):
        if _safe_int(counts.get(key)) > 0:
            violations.append(f"{prefix}_{key}_nonzero")
    for key in (
        "sourceSpanCreationReady",
        "equationSemanticsReady",
        "strictEvidenceReady",
        "parserRoutingReady",
        "answerIntegrationReady",
        "runtimePromotionAllowed",
    ):
        if bool(gate.get(key)):
            violations.append(f"{prefix}_{key}_true")
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
            violations.append(f"{prefix}_{key}_true")
    return list(dict.fromkeys(violations))


def _schema_violations(alignment: dict[str, Any], pdf_offset: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if alignment.get("schema") != EQUATION_ALIGNMENT_FEASIBILITY_AUDIT_SCHEMA_ID:
        violations.append("equation_alignment_feasibility_audit_schema_mismatch")
    if pdf_offset.get("schema") != EQUATION_QUOTE_PDF_OFFSET_FEASIBILITY_SCHEMA_ID:
        violations.append("equation_quote_pdf_offset_feasibility_schema_mismatch")
    return violations


def _pdf_rows_by_candidate(pdf_offset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for row in list(pdf_offset.get("feasibilityRows") or []):
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("source_equation_quote_candidate_id") or "")
        if candidate_id and candidate_id not in mapped:
            mapped[candidate_id] = dict(row)
    return mapped


def _action_type(alignment_row: dict[str, Any], pdf_row: dict[str, Any]) -> str:
    if pdf_row.get("original_pdf_offset_recovered") or alignment_row.get("source_span_created"):
        return "review_unexpected_source_span_before_any_promotion"
    if str(pdf_row.get("feasibility_status") or "") == "diagnostic_page_context_candidate_only":
        return "review_diagnostic_page_context"
    if str(alignment_row.get("feasibility_status") or "") == "diagnostic_term_context_candidate_only":
        return "review_diagnostic_term_context"
    return "reject_or_reextract_unmatched_equation_quote"


def _action_status(action_type: str) -> str:
    if action_type.startswith("review_"):
        return "ready_for_human_review"
    return "blocked_no_source_span_or_diagnostic_context"


def _action_card(index: int, alignment_row: dict[str, Any], pdf_row: dict[str, Any]) -> dict[str, Any]:
    action_type = _action_type(alignment_row, pdf_row)
    diagnostic_pages = list(pdf_row.get("diagnostic_page_candidates") or [])
    return {
        "action_card_id": f"equation-quote-next-action:{index:04d}",
        "source_alignment_audit_id": str(alignment_row.get("audit_id") or ""),
        "source_pdf_offset_feasibility_row_id": str(pdf_row.get("feasibility_row_id") or ""),
        "source_equation_quote_candidate_id": str(
            alignment_row.get("candidate_id") or pdf_row.get("source_equation_quote_candidate_id") or ""
        ),
        "paper_id": str(alignment_row.get("paper_id") or pdf_row.get("paper_id") or ""),
        "candidate_text": str(alignment_row.get("candidate_text") or pdf_row.get("candidate_text") or ""),
        "equation_label": str(pdf_row.get("equation_label") or ""),
        "action_type": action_type,
        "action_status": _action_status(action_type),
        "priority": "high" if action_type != "review_unexpected_source_span_before_any_promotion" else "blocked",
        "reason": "equation quote has diagnostic page/term context but no canonical or original-PDF source span"
        if action_type.startswith("review_diagnostic")
        else "equation quote has no reliable source span and needs rejection or later re-extraction",
        "canonical_alignment_status": str(alignment_row.get("existing_alignment_status") or ""),
        "canonical_alignment_method": str(alignment_row.get("existing_alignment_method") or ""),
        "alignment_feasibility_status": str(alignment_row.get("feasibility_status") or ""),
        "pdf_offset_feasibility_status": str(pdf_row.get("feasibility_status") or ""),
        "diagnostic_terms": list(alignment_row.get("diagnostic_terms") or pdf_row.get("diagnostic_terms") or []),
        "diagnostic_term_matches": list(alignment_row.get("diagnostic_term_matches") or []),
        "diagnostic_term_coverage": _safe_float(alignment_row.get("diagnostic_term_coverage")),
        "diagnostic_page_candidates": diagnostic_pages[:5],
        "best_diagnostic_page_coverage": _safe_float(
            diagnostic_pages[0].get("coverage") if diagnostic_pages and isinstance(diagnostic_pages[0], dict) else 0.0
        ),
        "layout_element_count": _safe_int(alignment_row.get("layout_element_count")),
        "bbox_available": bool(alignment_row.get("bbox_available") or pdf_row.get("bbox")),
        "sourceContentHash": str(alignment_row.get("sourceContentHash") or pdf_row.get("sourceContentHash") or ""),
        "source_span_created": False,
        "original_pdf_offset_recovered": False,
        "equation_semantics_interpreted": False,
        "quote_only": True,
        "evidence_tier": "equation_quote_next_action_gate_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "equation_quote_next_action_gate_only",
            "equation_alignment_missing",
            "original_pdf_offset_not_recovered",
            "equation_semantics_not_interpreted",
            "runtime_promotion_disabled_for_tranche",
            "strict_promotion_requires_explicit_later_tranche",
        ],
        "non_strict_reason": [
            "action_cards_are_not_evidence",
            "diagnostic_context_does_not_create_source_spans",
            "equation_semantics_are_not_interpreted",
        ],
    }


def _counts(cards: list[dict[str, Any]], violations: list[str]) -> dict[str, Any]:
    blockers: Counter[str] = Counter()
    for card in cards:
        blockers.update(str(item) for item in list(card.get("strict_blockers") or []))
    return {
        "nextActionCards": len(cards),
        "humanReviewCards": sum(1 for card in cards if card.get("action_status") == "ready_for_human_review"),
        "blockedCards": sum(1 for card in cards if str(card.get("action_status") or "").startswith("blocked")),
        "diagnosticPageContextCards": sum(1 for card in cards if card.get("action_type") == "review_diagnostic_page_context"),
        "diagnosticTermContextCards": sum(1 for card in cards if card.get("action_type") == "review_diagnostic_term_context"),
        "unmatchedEquationQuoteCards": sum(
            1 for card in cards if card.get("action_type") == "reject_or_reextract_unmatched_equation_quote"
        ),
        "sourceSpanCreatedCards": 0,
        "originalPdfOffsetRecoveredCards": 0,
        "equationSemanticsInterpretedCards": 0,
        "strictEligibleCards": 0,
        "citationGradeCards": 0,
        "runtimeEvidenceCards": 0,
        "schemaViolationCount": len([item for item in violations if item.endswith("_mismatch")]),
        "unsafeUpstreamFlagCount": len([item for item in violations if not item.endswith("_mismatch")]),
        "byPaper": dict(Counter(str(card.get("paper_id") or "") for card in cards)),
        "byActionType": dict(Counter(str(card.get("action_type") or "") for card in cards)),
        "byActionStatus": dict(Counter(str(card.get("action_status") or "") for card in cards)),
        "strictBlockerSummary": dict(blockers),
    }


def build_equation_quote_next_action_gate(
    *,
    equation_alignment_feasibility_audit: str | Path,
    equation_quote_pdf_offset_feasibility: str | Path,
) -> dict[str, Any]:
    """Build a report-only next-action gate for EquationQuote blockers."""

    alignment_path = Path(str(equation_alignment_feasibility_audit)).expanduser()
    pdf_path = Path(str(equation_quote_pdf_offset_feasibility)).expanduser()
    alignment = _read_json(alignment_path)
    pdf_offset = _read_json(pdf_path)
    violations = [
        *_schema_violations(alignment, pdf_offset),
        *_strict_zero_violations("alignment", alignment),
        *_strict_zero_violations("pdfOffset", pdf_offset),
    ]
    pdf_by_candidate = _pdf_rows_by_candidate(pdf_offset)
    cards = [
        _action_card(index, dict(row), pdf_by_candidate.get(str(row.get("candidate_id") or ""), {}))
        for index, row in enumerate(list(alignment.get("rows") or []), start=1)
        if isinstance(row, dict)
    ]
    counts = _counts(cards, violations)
    return {
        "schema": EQUATION_QUOTE_NEXT_ACTION_GATE_SCHEMA_ID,
        "status": "blocked" if violations or not cards else "next_action_ready",
        "generatedAt": _now(),
        "inputs": {
            "equationAlignmentFeasibilityAudit": str(alignment_path),
            "equationAlignmentFeasibilityAuditSchema": str(alignment.get("schema") or ""),
            "equationQuotePdfOffsetFeasibility": str(pdf_path),
            "equationQuotePdfOffsetFeasibilitySchema": str(pdf_offset.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "nextActionGateReady": bool(cards) and not violations,
            "humanReviewRequired": bool(cards) and not violations,
            "sourceSpanCreationReady": False,
            "equationSemanticsReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "blocked" if violations or not cards else "equation_quote_next_action_review_required",
            "schemaViolations": [item for item in violations if item.endswith("_mismatch")],
            "unsafeUpstreamFlags": [item for item in violations if not item.endswith("_mismatch")],
            "recommendedNextTranche": "manual_review_equation_quote_diagnostic_context_or_reextract",
        },
        "policy": {
            "reportOnly": True,
            "nextActionGateOnly": True,
            "quoteOnly": True,
            "equationSemanticsInterpreted": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "next_action_cards_are_not_equation_evidence",
            "diagnostic_page_context_does_not_create_source_spans",
            "equation_semantics_are_not_interpreted",
        ],
        "actionCards": cards,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_equation_quote_next_action_gate_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# EquationQuote Next Action Gate",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Action cards: `{int(counts.get('nextActionCards') or 0)}`",
        f"- Human-review cards: `{int(counts.get('humanReviewCards') or 0)}`",
        f"- Diagnostic page-context cards: `{int(counts.get('diagnosticPageContextCards') or 0)}`",
        f"- Unmatched cards: `{int(counts.get('unmatchedEquationQuoteCards') or 0)}`",
        f"- Strict eligible cards: `{int(counts.get('strictEligibleCards') or 0)}`",
        "",
        "## Boundary",
        "",
        "This gate is report-only. It does not interpret equations, create source spans, create strict evidence, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Cards",
        "",
    ]
    for card in list(report.get("actionCards") or []):
        lines.extend(
            [
                f"### {card.get('action_card_id', '')}",
                "",
                f"- Paper: `{card.get('paper_id', '')}`",
                f"- Candidate: `{card.get('source_equation_quote_candidate_id', '')}`",
                f"- Action: `{card.get('action_type', '')}` / `{card.get('action_status', '')}`",
                f"- Diagnostic terms: `{', '.join(str(item) for item in list(card.get('diagnostic_terms') or []))}`",
                f"- Best page coverage: `{card.get('best_diagnostic_page_coverage', 0)}`",
                f"- Reason: `{card.get('reason', '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_equation_quote_next_action_gate_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    gate_path = root / "equation-quote-next-action-gate.json"
    summary_path = root / "equation-quote-next-action-gate-summary.json"
    markdown_path = root / "equation-quote-next-action-gate.md"
    gate_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_quote_next_action_gate_markdown(report), encoding="utf-8")
    return {"gate": str(gate_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only EquationQuote next-action gate.")
    parser.add_argument("--equation-alignment-feasibility-audit", required=True)
    parser.add_argument("--equation-quote-pdf-offset-feasibility", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_equation_quote_next_action_gate(
        equation_alignment_feasibility_audit=args.equation_alignment_feasibility_audit,
        equation_quote_pdf_offset_feasibility=args.equation_quote_pdf_offset_feasibility,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_equation_quote_next_action_gate_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EQUATION_QUOTE_NEXT_ACTION_GATE_SCHEMA_ID",
    "build_equation_quote_next_action_gate",
    "render_equation_quote_next_action_gate_markdown",
    "write_equation_quote_next_action_gate_reports",
]
