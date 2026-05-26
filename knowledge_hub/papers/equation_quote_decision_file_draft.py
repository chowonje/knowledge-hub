"""Report-only draft decision file for EquationQuote manual review rows.

The draft is an editable starting point for human review. Every row defaults to
``needs_review`` and the helper records no rejection, re-extraction request,
source span, equation semantics, strict evidence, parser routing, canonical
artifact write, DB mutation, reindex, reembed, or answer integration.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


EQUATION_QUOTE_DECISION_FILE_DRAFT_SCHEMA_ID = "knowledge-hub.paper.equation-quote-decision-file-draft.v1"
EQUATION_QUOTE_MANUAL_REVIEW_SHEET_SCHEMA_ID = "knowledge-hub.paper.equation-quote-manual-review-sheet.v1"


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


def _review_rows(sheet: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(sheet.get("reviewRows") or []) if isinstance(item, dict)]


def _unsafe_flags(sheet: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(sheet.get("counts") or {})
    gate = dict(sheet.get("gate") or {})
    policy = dict(sheet.get("policy") or {})

    if sheet.get("schema") != EQUATION_QUOTE_MANUAL_REVIEW_SHEET_SCHEMA_ID:
        flags.append("equation_quote_manual_review_sheet_schema_mismatch")
    if str(sheet.get("status") or "") == "blocked":
        flags.append("equation_quote_manual_review_sheet_blocked")

    for key in (
        "sourceSpanCreatedRows",
        "originalPdfOffsetRecoveredRows",
        "equationSemanticsInterpretedRows",
        "strictEligibleRows",
        "citationGradeRows",
        "runtimeEvidenceRows",
    ):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"manualReviewSheet_{key}_nonzero")

    for key in (
        "humanReviewComplete",
        "strictEvidenceReady",
        "parserRoutingReady",
        "answerIntegrationReady",
        "runtimePromotionAllowed",
    ):
        if bool(gate.get(key)):
            flags.append(f"manualReviewSheet_{key}_true")

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
            flags.append(f"manualReviewSheet_{key}_true")

    return list(dict.fromkeys(flags))


def _draft_row(index: int, review_row: dict[str, Any]) -> dict[str, Any]:
    allowed_decisions = [str(item) for item in list(review_row.get("allowed_decisions") or []) if item]
    if "needs_review" not in allowed_decisions:
        allowed_decisions = ["needs_review", *allowed_decisions]
    return {
        "draft_row_id": f"equation-quote-decision-file-draft:{index:04d}",
        "source_review_sheet_row_id": str(review_row.get("review_sheet_row_id") or ""),
        "source_action_card_id": str(review_row.get("source_action_card_id") or ""),
        "source_equation_quote_candidate_id": str(review_row.get("source_equation_quote_candidate_id") or ""),
        "paper_id": str(review_row.get("paper_id") or ""),
        "candidate_text": str(review_row.get("candidate_text") or ""),
        "equation_label": str(review_row.get("equation_label") or ""),
        "action_type": str(review_row.get("action_type") or ""),
        "action_status": str(review_row.get("action_status") or ""),
        "priority": str(review_row.get("priority") or ""),
        "allowed_decisions": allowed_decisions,
        "decision": "needs_review",
        "reviewer": "",
        "notes": "",
        "draft_only": True,
        "decision_scope": "equation_quote_decision_file_draft_only_no_runtime_or_strict_promotion",
        "evidence_tier": "equation_quote_decision_file_draft_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "equation_quote_decision_file_draft_only",
            "human_review_decision_not_recorded",
            "source_span_not_created",
            "equation_semantics_not_interpreted",
            "strict_promotion_requires_later_explicit_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "draft_rows_are_not_human_review_decisions",
            "draft_rows_default_to_needs_review",
            "draft_rows_do_not_create_source_spans",
            "draft_rows_do_not_authorize_runtime_use",
        ],
    }


def _decision_file_from_drafts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "draftOnly": True,
        "instructions": [
            "Edit a copy of this file before using it as an EquationQuote decision file.",
            "Keep decision=needs_review unless a human explicitly rejects the candidate or requests re-extraction.",
            "Diagnostic context does not create source spans and does not interpret equations.",
            "Non-needs_review decisions require reviewer and notes.",
            "This file does not authorize strict evidence, parser routing, runtime citations, or canonical artifact writes.",
        ],
        "decisions": [
            {
                "source_review_sheet_row_id": str(row.get("source_review_sheet_row_id") or ""),
                "source_action_card_id": str(row.get("source_action_card_id") or ""),
                "source_equation_quote_candidate_id": str(row.get("source_equation_quote_candidate_id") or ""),
                "paper_id": str(row.get("paper_id") or ""),
                "candidate_text": str(row.get("candidate_text") or ""),
                "action_type": str(row.get("action_type") or ""),
                "decision": "needs_review",
                "reviewer": "",
                "notes": "",
                "allowed_decisions": list(row.get("allowed_decisions") or []),
            }
            for row in rows
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    by_paper = Counter(str(row.get("paper_id") or "") for row in rows)
    by_action_type = Counter(str(row.get("action_type") or "") for row in rows)
    by_action_status = Counter(str(row.get("action_status") or "") for row in rows)
    return {
        "draftRows": len(rows),
        "needsReviewRows": len(rows),
        "nonNeedsReviewRows": 0,
        "rejectDecisionRows": 0,
        "reextractRequestRows": 0,
        "keepBlockedRows": 0,
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
    }


def build_equation_quote_decision_file_draft(
    *,
    equation_quote_manual_review_sheet_report: str | Path,
) -> dict[str, Any]:
    """Build a needs-review-only draft file for EquationQuote decisions."""

    sheet_path = Path(str(equation_quote_manual_review_sheet_report)).expanduser()
    sheet = _read_json(sheet_path)
    unsafe_flags = _unsafe_flags(sheet)
    rows = [_draft_row(index, row) for index, row in enumerate(_review_rows(sheet), start=1)]
    counts = _counts(rows, unsafe_flags)
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif rows:
        status = "decision_file_draft_ready"
        decision = "needs_review_draft_ready_for_manual_edit"
    else:
        status = "no_equation_quote_review_rows"
        decision = "no_equation_quote_review_rows"
    return {
        "schema": EQUATION_QUOTE_DECISION_FILE_DRAFT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "equationQuoteManualReviewSheetReport": str(sheet_path),
            "equationQuoteManualReviewSheetSchema": str(sheet.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "decisionFileDraftReady": bool(rows) and not unsafe_flags,
            "containsOnlyNeedsReviewDefaults": True,
            "containsRejectRows": False,
            "containsReextractRequestRows": False,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_edit_equation_quote_decision_file"
            if rows
            else "equation_quote_manual_review_sheet_refresh",
        },
        "policy": {
            "reportOnly": True,
            "decisionFileDraftOnly": True,
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
            "diagnostic_context_does_not_create_source_spans",
            "equation_semantics_are_not_interpreted",
            "strict_or_runtime_promotion_requires_a_separate_explicit_tranche",
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


def render_equation_quote_decision_file_draft_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# EquationQuote Decision File Draft",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Draft rows: `{int(counts.get('draftRows') or 0)}`",
        f"- `needs_review` rows: `{int(counts.get('needsReviewRows') or 0)}`",
        f"- Reject decisions: `{int(counts.get('rejectDecisionRows') or 0)}`",
        f"- Re-extraction requests: `{int(counts.get('reextractRequestRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This draft is an editable starting point only. It does not record human review decisions, create source spans, interpret equations, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By action type: `{json.dumps(counts.get('byActionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By action status: `{json.dumps(counts.get('byActionStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_equation_quote_decision_file_draft_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    draft_report_path = root / "equation-quote-decision-file-draft.json"
    decision_file_path = root / "equation-quote-decisions.draft.json"
    summary_path = root / "equation-quote-decision-file-draft-summary.json"
    markdown_path = root / "equation-quote-decision-file-draft.md"
    draft_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    decision_file_path.write_text(
        json.dumps(report.get("decisionFileDraft") or {}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_quote_decision_file_draft_markdown(report), encoding="utf-8")
    return {
        "draftReport": str(draft_report_path),
        "decisionFileDraft": str(decision_file_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a needs-review-only EquationQuote decision file draft.")
    parser.add_argument("--equation-quote-manual-review-sheet-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_equation_quote_decision_file_draft(
        equation_quote_manual_review_sheet_report=args.equation_quote_manual_review_sheet_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_equation_quote_decision_file_draft_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EQUATION_QUOTE_DECISION_FILE_DRAFT_SCHEMA_ID",
    "build_equation_quote_decision_file_draft",
    "render_equation_quote_decision_file_draft_markdown",
    "write_equation_quote_decision_file_draft_reports",
]
