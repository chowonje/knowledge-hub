"""Report-only manual review sheet for candidate-layer blocker decisions.

This helper consolidates the decision draft, template context, and current
decision-record status into an operator-readable sheet. It does not record
decisions, execute operator actions, create evidence, or authorize promotion.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CANDIDATE_LAYER_BLOCKER_MANUAL_DECISION_REVIEW_SHEET_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-manual-decision-review-sheet.v1"
)
CANDIDATE_LAYER_BLOCKER_DECISION_TEMPLATE_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-template.v1"
)
CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-record.v1"
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


def _decision_rows(decisions_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = decisions_report.get("decisions")
    if rows is None:
        rows = decisions_report.get("decisionRows")
    if rows is None:
        rows = decisions_report.get("decisionRecords")
    return [dict(item) for item in list(rows or []) if isinstance(item, dict)]


def _template_rows(template: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(template.get("decisionRows") or []) if isinstance(item, dict)]


def _decision_id(item: dict[str, Any]) -> str:
    return str(
        item.get("source_decision_row_id")
        or item.get("sourceDecisionRowId")
        or item.get("decision_row_id")
        or item.get("decisionRowId")
        or ""
    )


def _decision_value(item: dict[str, Any]) -> str:
    return str(
        item.get("decision")
        or item.get("recorded_decision")
        or item.get("recordedDecision")
        or item.get("review_decision")
        or item.get("reviewDecision")
        or ""
    )


def _by_decision_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = _decision_id(row)
        if row_id and row_id not in result:
            result[row_id] = row
    return result


def _unsafe_flags(template: dict[str, Any], record: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    template_counts = dict(template.get("counts") or {})
    template_gate = dict(template.get("gate") or {})
    template_policy = dict(template.get("policy") or {})
    record_counts = dict(record.get("counts") or {})
    record_gate = dict(record.get("gate") or {})
    record_policy = dict(record.get("policy") or {})

    if template.get("schema") != CANDIDATE_LAYER_BLOCKER_DECISION_TEMPLATE_SCHEMA_ID:
        flags.append("candidate_layer_blocker_decision_template_schema_mismatch")
    if record and record.get("schema") != CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID:
        flags.append("candidate_layer_blocker_decision_record_schema_mismatch")
    if str(template.get("status") or "") == "blocked":
        flags.append("candidate_layer_blocker_decision_template_blocked")
    if str(record.get("status") or "") == "blocked":
        flags.append("candidate_layer_blocker_decision_record_blocked")

    for key in ("strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
        if _safe_int(template_counts.get(key)) > 0:
            flags.append(f"decisionTemplate_{key}_nonzero")
        if _safe_int(record_counts.get(key)) > 0:
            flags.append(f"decisionRecord_{key}_nonzero")

    for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(template_gate.get(key)):
            flags.append(f"decisionTemplate_{key}_true")
        if bool(record_gate.get(key)):
            flags.append(f"decisionRecord_{key}_true")

    for key in (
        "strictEvidenceCreated",
        "runtimePromotionAllowed",
        "parserRoutingChanged",
        "canonicalParsedArtifactsWritten",
        "databaseMutation",
        "reindexOrReembed",
        "answerIntegrationChanged",
    ):
        if bool(template_policy.get(key)):
            flags.append(f"decisionTemplate_{key}_true")
        if bool(record_policy.get(key)):
            flags.append(f"decisionRecord_{key}_true")
    return list(dict.fromkeys(flags))


def _review_row(
    index: int,
    template_row: dict[str, Any],
    draft_by_id: dict[str, dict[str, Any]],
    record_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row_id = str(template_row.get("decision_row_id") or "")
    draft_row = draft_by_id.get(row_id) or {}
    record_row = record_by_id.get(row_id) or {}
    current_decision = _decision_value(draft_row) or "needs_review"
    allowed_decisions = [str(item) for item in list(template_row.get("allowed_decisions") or [])]
    return {
        "review_sheet_row_id": f"candidate-layer-blocker-manual-review:{index:04d}",
        "source_decision_row_id": row_id,
        "source_review_card_id": str(template_row.get("source_review_card_id") or ""),
        "source_backlog_id": str(template_row.get("source_backlog_id") or ""),
        "blocker": str(template_row.get("blocker") or ""),
        "priority": str(template_row.get("priority") or ""),
        "review_bucket": str(template_row.get("review_bucket") or ""),
        "affected_layers": list(template_row.get("affected_layers") or []),
        "affected_candidate_count": _safe_int(template_row.get("affected_candidate_count")),
        "affected_eval_question_count": _safe_int(template_row.get("affected_eval_question_count")),
        "current_decision": current_decision,
        "allowed_decisions": allowed_decisions,
        "draft_reviewer": str(draft_row.get("reviewer") or ""),
        "draft_notes": str(draft_row.get("notes") or ""),
        "recorded_decision": str(record_row.get("recorded_decision") or ""),
        "recommended_next_tranche": str(template_row.get("recommended_next_tranche") or ""),
        "recommended_review_action": str(template_row.get("recommended_review_action") or ""),
        "required_review_checks": [str(item) for item in list(template_row.get("required_review_checks") or [])],
        "decision_note": "Edit the draft decision file only after human/operator review; leave needs_review when unsure.",
        "decision_scope": "candidate_layer_blocker_manual_decision_review_sheet_only_no_runtime_or_strict_promotion",
        "evidence_tier": "candidate_layer_blocker_manual_decision_review_sheet_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "manual_decision_review_sheet_only",
            "strict_promotion_requires_later_explicit_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "manual_review_sheet_rows_are_not_decisions",
            "manual_review_sheet_rows_do_not_authorize_runtime_use",
            "manual_review_sheet_rows_do_not_create_strict_evidence",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    by_bucket = Counter(str(row.get("review_bucket") or "") for row in rows)
    by_priority = Counter(str(row.get("priority") or "") for row in rows)
    by_layer = Counter(layer for row in rows for layer in list(row.get("affected_layers") or []))
    by_decision = Counter(str(row.get("current_decision") or "") for row in rows)
    return {
        "reviewRows": len(rows),
        "needsReviewRows": by_decision.get("needs_review", 0),
        "nonNeedsReviewRows": len(rows) - by_decision.get("needs_review", 0),
        "manualBucketRows": by_bucket.get("manual_decision_required", 0),
        "operatorBucketRows": by_bucket.get("operator_approval_required", 0),
        "technicalBucketRows": by_bucket.get("technical_feasibility_blocked", 0),
        "policyBucketRows": by_bucket.get("policy_review_only", 0),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byBucket": dict(by_bucket),
        "byPriority": dict(by_priority),
        "byLayer": dict(by_layer),
        "byDecision": dict(by_decision),
    }


def build_candidate_layer_blocker_manual_decision_review_sheet(
    *,
    candidate_layer_blocker_decision_template_report: str | Path,
    candidate_layer_blocker_decisions_file: str | Path,
    candidate_layer_blocker_decision_record_report: str | Path | None = None,
) -> dict[str, Any]:
    """Build a report-only manual decision review sheet."""

    template_path = Path(str(candidate_layer_blocker_decision_template_report)).expanduser()
    decisions_path = Path(str(candidate_layer_blocker_decisions_file)).expanduser()
    record_path = (
        Path(str(candidate_layer_blocker_decision_record_report)).expanduser()
        if candidate_layer_blocker_decision_record_report
        else None
    )
    template = _read_json(template_path)
    decisions = _read_json(decisions_path)
    record = _read_json(record_path)
    unsafe_flags = _unsafe_flags(template, record)
    rows = [
        _review_row(index, row, _by_decision_id(_decision_rows(decisions)), _by_decision_id(_decision_rows(record)))
        for index, row in enumerate(_template_rows(template), start=1)
    ]
    counts = _counts(rows, unsafe_flags)
    status = "blocked" if unsafe_flags else "manual_review_sheet_ready"
    return {
        "schema": CANDIDATE_LAYER_BLOCKER_MANUAL_DECISION_REVIEW_SHEET_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "candidateLayerBlockerDecisionTemplateReport": str(template_path),
            "candidateLayerBlockerDecisionTemplateSchema": str(template.get("schema") or ""),
            "candidateLayerBlockerDecisionsFile": str(decisions_path),
            "candidateLayerBlockerDecisionRecordReport": str(record_path or ""),
            "candidateLayerBlockerDecisionRecordSchema": str(record.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "manualReviewSheetReady": bool(rows) and not unsafe_flags,
            "containsNonNeedsReviewDraftValues": _safe_int(counts.get("nonNeedsReviewRows")) > 0,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "blocked" if unsafe_flags else "manual_review_sheet_ready_for_human_edit",
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_edit_candidate_layer_blocker_decision_file_draft",
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
            "manual_review_sheet_does_not_authorize_runtime_use",
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


def render_candidate_layer_blocker_manual_decision_review_sheet_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    inputs = dict(report.get("inputs") or {})
    lines = [
        "# Candidate Layer Blocker Manual Decision Review Sheet",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Review rows: `{int(counts.get('reviewRows') or 0)}`",
        f"- `needs_review` rows: `{int(counts.get('needsReviewRows') or 0)}`",
        f"- Non-`needs_review` draft values: `{int(counts.get('nonNeedsReviewRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This sheet is local review metadata only. It does not record approvals, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Edit Target",
        "",
        f"- Decision file: `{inputs.get('candidateLayerBlockerDecisionsFile', '')}`",
        "",
        "## Counts",
        "",
        f"- By bucket: `{json.dumps(counts.get('byBucket') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By priority: `{json.dumps(counts.get('byPriority') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By layer: `{json.dumps(counts.get('byLayer') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By decision: `{json.dumps(counts.get('byDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Rows",
        "",
    ]
    for row in list(report.get("reviewRows") or []):
        lines.extend(
            [
                f"### {row.get('source_decision_row_id', '')} - {row.get('blocker', '')}",
                "",
                f"- Priority: `{row.get('priority', '')}`",
                f"- Bucket: `{row.get('review_bucket', '')}`",
                f"- Layers: `{', '.join(list(row.get('affected_layers') or []))}`",
                f"- Current decision: `{row.get('current_decision', '')}`",
                f"- Allowed decisions: `{', '.join(list(row.get('allowed_decisions') or []))}`",
                f"- Recommended action: `{row.get('recommended_review_action', '')}`",
                f"- Recommended next tranche: `{row.get('recommended_next_tranche', '')}`",
                f"- Required checks: `{'; '.join(list(row.get('required_review_checks') or []))}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_candidate_layer_blocker_manual_decision_review_sheet_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    sheet_path = root / "candidate-layer-blocker-manual-decision-review-sheet.json"
    summary_path = root / "candidate-layer-blocker-manual-decision-review-summary.json"
    markdown_path = root / "candidate-layer-blocker-manual-decision-review-sheet.md"
    sheet_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_candidate_layer_blocker_manual_decision_review_sheet_markdown(report),
        encoding="utf-8",
    )
    return {"sheet": str(sheet_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only candidate-layer blocker manual decision review sheet.")
    parser.add_argument("--candidate-layer-blocker-decision-template-report", required=True)
    parser.add_argument("--candidate-layer-blocker-decisions-file", required=True)
    parser.add_argument("--candidate-layer-blocker-decision-record-report", default="")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_candidate_layer_blocker_manual_decision_review_sheet(
        candidate_layer_blocker_decision_template_report=args.candidate_layer_blocker_decision_template_report,
        candidate_layer_blocker_decisions_file=args.candidate_layer_blocker_decisions_file,
        candidate_layer_blocker_decision_record_report=args.candidate_layer_blocker_decision_record_report or None,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_candidate_layer_blocker_manual_decision_review_sheet_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_LAYER_BLOCKER_MANUAL_DECISION_REVIEW_SHEET_SCHEMA_ID",
    "build_candidate_layer_blocker_manual_decision_review_sheet",
    "render_candidate_layer_blocker_manual_decision_review_sheet_markdown",
    "write_candidate_layer_blocker_manual_decision_review_sheet_reports",
]
