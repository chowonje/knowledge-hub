"""Report-only edit plan for EquationQuote decision files.

This helper combines the editable EquationQuote decision draft with the
nonbinding recommendation pack. It emits manual edit hints only. It does not
write the decision file, accept recommendations as human decisions, create
source spans, interpret equations, promote strict evidence, route parsers,
write canonical artifacts, mutate DB state, reindex, reembed, or change answer
behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


EQUATION_QUOTE_DECISION_EDIT_PLAN_SCHEMA_ID = "knowledge-hub.paper.equation-quote-decision-edit-plan.v1"
EQUATION_QUOTE_DECISION_RECOMMENDATION_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.equation-quote-decision-recommendation-pack.v1"
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
        return int(value or 0)
    except Exception:
        return 0


def _decision_rows(decision_file: dict[str, Any]) -> list[dict[str, Any]]:
    rows = decision_file.get("decisions")
    if rows is None:
        rows = decision_file.get("decisionRows")
    if rows is None:
        rows = decision_file.get("reviewDecisions")
    return [dict(item) for item in list(rows or []) if isinstance(item, dict)]


def _recommendation_rows(pack: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(pack.get("recommendationRows") or []) if isinstance(item, dict)]


def _decision_key(row: dict[str, Any]) -> str:
    return str(
        row.get("source_review_sheet_row_id")
        or row.get("review_sheet_row_id")
        or row.get("sourceReviewSheetRowId")
        or row.get("reviewSheetRowId")
        or ""
    )


def _decision_by_review_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _decision_key(row)
        if key and key not in result:
            result[key] = row
    return result


def _allowed_decisions(row: dict[str, Any]) -> list[str]:
    allowed = [str(item) for item in list(row.get("allowed_decisions") or []) if item]
    if "needs_review" not in allowed:
        allowed = ["needs_review", *allowed]
    return allowed


def _current_decision(row: dict[str, Any]) -> str:
    return str(row.get("decision") or row.get("review_decision") or row.get("reviewDecision") or "needs_review")


def _unsafe_flags(recommendation_pack: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(recommendation_pack.get("counts") or {})
    gate = dict(recommendation_pack.get("gate") or {})
    policy = dict(recommendation_pack.get("policy") or {})
    if recommendation_pack.get("schema") != EQUATION_QUOTE_DECISION_RECOMMENDATION_PACK_SCHEMA_ID:
        flags.append("equation_quote_decision_recommendation_pack_schema_mismatch")
    if str(recommendation_pack.get("status") or "") == "blocked":
        flags.append("equation_quote_decision_recommendation_pack_blocked")
    for key in (
        "acceptedHumanDecisionRows",
        "sourceSpanCreatedRows",
        "originalPdfOffsetRecoveredRows",
        "equationSemanticsInterpretedRows",
        "strictEligibleRows",
        "citationGradeRows",
        "runtimeEvidenceRows",
    ):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"recommendationPack_{key}_nonzero")
    for key in (
        "humanReviewComplete",
        "strictEvidenceReady",
        "parserRoutingReady",
        "answerIntegrationReady",
        "runtimePromotionAllowed",
    ):
        if bool(gate.get(key)):
            flags.append(f"recommendationPack_{key}_true")
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
            flags.append(f"recommendationPack_{key}_true")
    return list(dict.fromkeys(flags))


def _edit_row(index: int, recommendation: dict[str, Any], decision_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    review_row_id = str(recommendation.get("source_review_sheet_row_id") or "")
    decision_row = decision_by_id.get(review_row_id) or {}
    allowed = _allowed_decisions(decision_row)
    recommended_decision = str(recommendation.get("recommended_decision") or "needs_review")
    missing_decision_row = not bool(decision_row)
    recommendation_allowed = bool(decision_row) and recommended_decision in allowed
    if missing_decision_row:
        edit_status = "blocked_missing_decision_file_row"
    elif not recommendation_allowed:
        edit_status = "blocked_recommendation_not_allowed"
    else:
        edit_status = "ready_for_manual_edit"
    return {
        "edit_row_id": f"equation-quote-decision-edit-plan:{index:04d}",
        "source_recommendation_row_id": str(recommendation.get("recommendation_row_id") or ""),
        "source_review_sheet_row_id": review_row_id,
        "source_action_card_id": str(recommendation.get("source_action_card_id") or ""),
        "source_equation_quote_candidate_id": str(recommendation.get("source_equation_quote_candidate_id") or ""),
        "paper_id": str(recommendation.get("paper_id") or ""),
        "candidate_text": str(recommendation.get("candidate_text") or ""),
        "equation_label": str(recommendation.get("equation_label") or ""),
        "action_type": str(recommendation.get("action_type") or ""),
        "action_status": str(recommendation.get("action_status") or ""),
        "priority": str(recommendation.get("priority") or ""),
        "current_decision": _current_decision(decision_row),
        "recommended_decision": recommended_decision,
        "recommended_decision_reason": str(recommendation.get("recommended_decision_reason") or ""),
        "recommendation_confidence": str(recommendation.get("recommendation_confidence") or ""),
        "allowed_decisions": allowed,
        "edit_status": edit_status,
        "reviewer_required": recommended_decision != "needs_review",
        "notes_required": recommended_decision != "needs_review",
        "human_decision_required": True,
        "accepted_as_human_decision": False,
        "decision_file_patch_hint": {
            "source_review_sheet_row_id": review_row_id,
            "decision": recommended_decision,
            "reviewer": "",
            "notes": "",
        },
        "source_span_created": False,
        "original_pdf_offset_recovered": False,
        "equation_semantics_interpreted": False,
        "evidence_tier": "equation_quote_decision_edit_plan_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "equation_quote_decision_edit_plan_only",
            "human_review_decision_not_recorded",
            "source_span_not_created",
            "equation_semantics_not_interpreted",
            "strict_promotion_requires_later_explicit_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "edit_plan_rows_are_not_human_review_decisions",
            "edit_plan_rows_do_not_modify_the_decision_file",
            "edit_plan_rows_do_not_create_source_spans",
            "edit_plan_rows_do_not_interpret_equations",
            "edit_plan_rows_do_not_authorize_runtime_use",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    by_status = Counter(str(row.get("edit_status") or "") for row in rows)
    by_current = Counter(str(row.get("current_decision") or "") for row in rows)
    by_recommended = Counter(str(row.get("recommended_decision") or "") for row in rows)
    by_paper = Counter(str(row.get("paper_id") or "") for row in rows)
    by_action_type = Counter(str(row.get("action_type") or "") for row in rows)
    return {
        "editRows": len(rows),
        "readyForManualEditRows": by_status.get("ready_for_manual_edit", 0),
        "blockedMissingDecisionFileRows": by_status.get("blocked_missing_decision_file_row", 0),
        "blockedRecommendationNotAllowedRows": by_status.get("blocked_recommendation_not_allowed", 0),
        "currentNeedsReviewRows": by_current.get("needs_review", 0),
        "currentNonNeedsReviewRows": len(rows) - by_current.get("needs_review", 0),
        "proposedAcceptDiagnosticContextRows": by_recommended.get(
            "accept_diagnostic_context_for_later_reextract_design", 0
        ),
        "proposedRejectRows": by_recommended.get("reject_equation_quote_candidate", 0),
        "proposedReextractRequestRows": by_recommended.get("request_equation_quote_reextraction", 0),
        "proposedKeepBlockedRows": by_recommended.get("keep_blocked", 0),
        "proposedNeedsReviewRows": by_recommended.get("needs_review", 0),
        "acceptedHumanDecisionRows": 0,
        "sourceSpanCreatedRows": 0,
        "originalPdfOffsetRecoveredRows": 0,
        "equationSemanticsInterpretedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byEditStatus": dict(by_status),
        "byCurrentDecision": dict(by_current),
        "byRecommendedDecision": dict(by_recommended),
        "byPaper": dict(by_paper),
        "byActionType": dict(by_action_type),
    }


def build_equation_quote_decision_edit_plan(
    *,
    equation_quote_decision_recommendation_pack_report: str | Path,
    equation_quote_decisions_file: str | Path,
) -> dict[str, Any]:
    """Build a report-only manual edit plan for EquationQuote decisions."""

    recommendation_path = Path(str(equation_quote_decision_recommendation_pack_report)).expanduser()
    decisions_path = Path(str(equation_quote_decisions_file)).expanduser()
    recommendation_pack = _read_json(recommendation_path)
    decision_file = _read_json(decisions_path)
    unsafe_flags = _unsafe_flags(recommendation_pack)
    decision_by_id = _decision_by_review_id(_decision_rows(decision_file))
    rows = [
        _edit_row(index, row, decision_by_id)
        for index, row in enumerate(_recommendation_rows(recommendation_pack), start=1)
    ]
    counts = _counts(rows, unsafe_flags)
    blocked = bool(unsafe_flags) or _safe_int(counts.get("blockedMissingDecisionFileRows")) > 0 or _safe_int(
        counts.get("blockedRecommendationNotAllowedRows")
    ) > 0
    if blocked:
        status = "blocked"
        decision = "blocked"
    elif rows:
        status = "edit_plan_ready"
        decision = "manual_edit_still_required"
    else:
        status = "no_edit_rows"
        decision = "no_equation_quote_decision_recommendations"
    return {
        "schema": EQUATION_QUOTE_DECISION_EDIT_PLAN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "equationQuoteDecisionRecommendationPackReport": str(recommendation_path),
            "equationQuoteDecisionRecommendationPackSchema": str(recommendation_pack.get("schema") or ""),
            "equationQuoteDecisionsFile": str(decisions_path),
            "equationQuoteDecisionInputRows": len(_decision_rows(decision_file)),
        },
        "counts": counts,
        "gate": {
            "editPlanReady": bool(rows) and not blocked,
            "manualDecisionFileEditRequired": bool(rows) and not blocked,
            "decisionFileModified": False,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_edit_equation_quote_decision_file" if rows else "refresh_recommendations",
        },
        "policy": {
            "reportOnly": True,
            "decisionEditPlanOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "edit_plan_rows_are_not_human_review_decisions",
            "edit_plan_does_not_write_or_modify_the_decision_file",
            "patch_hints_must_be_reviewed_by_a_human_before_validation_or_decision_record_generation",
            "diagnostic_context_does_not_create_source_spans_or_runtime_evidence",
        ],
        "editRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_equation_quote_decision_edit_plan_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# EquationQuote Decision Edit Plan",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Edit rows: `{int(counts.get('editRows') or 0)}`",
        f"- Ready for manual edit: `{int(counts.get('readyForManualEditRows') or 0)}`",
        f"- Proposed diagnostic-context acceptances: `{int(counts.get('proposedAcceptDiagnosticContextRows') or 0)}`",
        f"- Proposed re-extraction requests: `{int(counts.get('proposedReextractRequestRows') or 0)}`",
        f"- Accepted human decisions: `{int(counts.get('acceptedHumanDecisionRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This edit plan is not a decision file. It does not modify the draft, record human decisions, create source spans, interpret equations, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By current decision: `{json.dumps(counts.get('byCurrentDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By recommended decision: `{json.dumps(counts.get('byRecommendedDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By edit status: `{json.dumps(counts.get('byEditStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_equation_quote_decision_edit_plan_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "equation-quote-decision-edit-plan.json"
    summary_path = root / "equation-quote-decision-edit-plan-summary.json"
    markdown_path = root / "equation-quote-decision-edit-plan.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_quote_decision_edit_plan_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only EquationQuote decision edit plan.")
    parser.add_argument("--equation-quote-decision-recommendation-pack-report", required=True)
    parser.add_argument("--equation-quote-decisions-file", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_equation_quote_decision_edit_plan(
        equation_quote_decision_recommendation_pack_report=args.equation_quote_decision_recommendation_pack_report,
        equation_quote_decisions_file=args.equation_quote_decisions_file,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_equation_quote_decision_edit_plan_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EQUATION_QUOTE_DECISION_EDIT_PLAN_SCHEMA_ID",
    "build_equation_quote_decision_edit_plan",
    "render_equation_quote_decision_edit_plan_markdown",
    "write_equation_quote_decision_edit_plan_reports",
]
