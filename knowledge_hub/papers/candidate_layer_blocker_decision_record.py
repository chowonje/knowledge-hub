"""Report-only decision record for candidate-layer blocker decision templates.

The record can consume an explicit external decision file, but without one it
keeps every blocker row pending. Recorded decisions remain review metadata only:
they do not create evidence, run parsers, mutate stores, or authorize runtime
promotion.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-record.v1"
)
CANDIDATE_LAYER_BLOCKER_DECISION_TEMPLATE_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-template.v1"
)

_RECORDED_DECISION_BY_INPUT = {
    "needs_review": "needs_review",
    "record_manual_approval_in_separate_decision_file": "manual_approval_recorded_for_later_design_only",
    "record_manual_rejection_in_separate_decision_file": "manual_rejection_recorded_keep_blocked",
    "approve_diagnostic_operator_action_in_separate_decision_file": "operator_diagnostic_action_approved_report_only",
    "decline_diagnostic_operator_action_keep_blocked": "operator_diagnostic_action_declined_keep_blocked",
    "accept_technical_blocker_as_open": "technical_blocker_accepted_open",
    "defer_technical_followup": "technical_followup_deferred",
    "close_as_not_needed": "technical_blocker_closed_as_not_needed",
    "accept_policy_blocker_as_guardrail": "policy_blocker_accepted_guardrail",
    "defer_policy_review": "policy_review_deferred",
    "keep_blocked": "keep_blocked",
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
    if rows is None:
        rows = decisions_report.get("decisionRecords")
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
    return str(
        item.get("decision")
        or item.get("recorded_decision")
        or item.get("recordedDecision")
        or item.get("review_decision")
        or item.get("reviewDecision")
        or ""
    )


def _template_rows(template: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(template.get("decisionRows") or []) if isinstance(item, dict)]


def _allowed_decision_by_row_id(template_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        str(row.get("decision_row_id") or ""): {str(item) for item in list(row.get("allowed_decisions") or [])}
        for row in template_rows
    }


def _unsafe_flags(template: dict[str, Any], decisions_report: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(template.get("counts") or {})
    gate = dict(template.get("gate") or {})
    policy = dict(template.get("policy") or {})
    template_rows = _template_rows(template)
    allowed_by_id = _allowed_decision_by_row_id(template_rows)
    if template.get("schema") != CANDIDATE_LAYER_BLOCKER_DECISION_TEMPLATE_SCHEMA_ID:
        flags.append("candidate_layer_blocker_decision_template_schema_mismatch")
    if str(template.get("status") or "") == "blocked":
        flags.append("candidate_layer_blocker_decision_template_blocked")
    for key in (
        "acceptedDecisionRows",
        "operatorApprovedRows",
        "strictEligibleRows",
        "citationGradeRows",
        "runtimeEvidenceRows",
    ):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"decisionTemplate_{key}_nonzero")
    for key in (
        "strictEvidenceReady",
        "parserRoutingReady",
        "answerIntegrationReady",
        "runtimePromotionAllowed",
    ):
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
    for item in _decision_rows(decisions_report):
        row_id = _decision_id(item)
        decision = _decision_value(item)
        if not row_id:
            flags.append("blocker_decision_row_id_missing")
            continue
        if row_id in seen:
            flags.append("blocker_decision_duplicate_row_id")
        seen.add(row_id)
        if row_id not in allowed_by_id:
            flags.append("blocker_decision_unknown_template_row_id")
            continue
        if decision not in allowed_by_id[row_id]:
            flags.append("blocker_decision_invalid_for_review_bucket")
        if decision not in _RECORDED_DECISION_BY_INPUT:
            flags.append("blocker_decision_unknown_value")
    return list(dict.fromkeys(flags))


def _decision_map(template_rows: list[dict[str, Any]], decisions_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    allowed_by_id = _allowed_decision_by_row_id(template_rows)
    mapped: dict[str, dict[str, Any]] = {}
    for item in _decision_rows(decisions_report):
        row_id = _decision_id(item)
        decision = _decision_value(item)
        if row_id and row_id not in mapped and decision in allowed_by_id.get(row_id, set()):
            mapped[row_id] = dict(item)
    return mapped


def _record_row(index: int, template_row: dict[str, Any], decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row_id = str(template_row.get("decision_row_id") or "")
    decision_item = decisions.get(row_id) or {}
    raw_decision = _decision_value(decision_item) or "needs_review"
    recorded_decision = _RECORDED_DECISION_BY_INPUT.get(raw_decision, "needs_review")
    strict_blockers = [
        "candidate_layer_blocker_decision_record_only",
        "strict_promotion_requires_later_explicit_tranche",
        "runtime_promotion_disabled_for_tranche",
    ]
    if recorded_decision == "needs_review":
        strict_blockers.append("candidate_layer_blocker_decision_missing")
    elif recorded_decision == "manual_approval_recorded_for_later_design_only":
        strict_blockers.append("manual_approval_is_for_later_design_only_not_runtime_evidence")
    elif recorded_decision == "operator_diagnostic_action_approved_report_only":
        strict_blockers.append("operator_approval_is_report_only_and_does_not_execute_action")
    elif recorded_decision.endswith("_keep_blocked") or recorded_decision == "keep_blocked":
        strict_blockers.append("blocker_kept_open_or_rejected")
    else:
        strict_blockers.append("recorded_decision_is_non_runtime_review_metadata")
    return {
        "record_row_id": f"candidate-layer-blocker-decision-record:{index:04d}",
        "source_decision_row_id": row_id,
        "source_review_card_id": str(template_row.get("source_review_card_id") or ""),
        "source_backlog_id": str(template_row.get("source_backlog_id") or ""),
        "blocker": str(template_row.get("blocker") or ""),
        "priority": str(template_row.get("priority") or ""),
        "review_bucket": str(template_row.get("review_bucket") or ""),
        "affected_layers": list(template_row.get("affected_layers") or []),
        "affected_candidate_count": _safe_int(template_row.get("affected_candidate_count")),
        "affected_eval_question_count": _safe_int(template_row.get("affected_eval_question_count")),
        "recommended_next_tranche": str(template_row.get("recommended_next_tranche") or ""),
        "recommended_review_action": str(template_row.get("recommended_review_action") or ""),
        "raw_decision": raw_decision,
        "recorded_decision": recorded_decision,
        "decision_scope": "candidate_layer_blocker_decision_record_only_no_runtime_or_strict_promotion",
        "reviewer": str(decision_item.get("reviewer") or ""),
        "notes": str(decision_item.get("notes") or ""),
        "evidence_tier": "candidate_layer_blocker_decision_record_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": [
            "decision_record_rows_are_review_metadata_only",
            "decision_record_rows_do_not_authorize_runtime_use",
            "decision_record_rows_do_not_create_strict_evidence",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    by_bucket = Counter(str(row.get("review_bucket") or "") for row in rows)
    by_priority = Counter(str(row.get("priority") or "") for row in rows)
    by_layer = Counter(layer for row in rows for layer in list(row.get("affected_layers") or []))
    by_decision = Counter(str(row.get("recorded_decision") or "") for row in rows)
    return {
        "recordRows": len(rows),
        "needsReviewRows": by_decision.get("needs_review", 0),
        "manualApprovalRows": by_decision.get("manual_approval_recorded_for_later_design_only", 0),
        "manualRejectionRows": by_decision.get("manual_rejection_recorded_keep_blocked", 0),
        "operatorApprovedRows": by_decision.get("operator_diagnostic_action_approved_report_only", 0),
        "operatorDeclinedRows": by_decision.get("operator_diagnostic_action_declined_keep_blocked", 0),
        "technicalAcceptedOpenRows": by_decision.get("technical_blocker_accepted_open", 0),
        "technicalDeferredRows": by_decision.get("technical_followup_deferred", 0),
        "policyAcceptedGuardrailRows": by_decision.get("policy_blocker_accepted_guardrail", 0),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byBucket": dict(by_bucket),
        "byPriority": dict(by_priority),
        "byLayer": dict(by_layer),
        "byDecision": dict(by_decision),
    }


def _all_non_needs_review(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(str(row.get("recorded_decision") or "") != "needs_review" for row in rows)


def _bucket_complete(rows: list[dict[str, Any]], bucket: str) -> bool:
    bucket_rows = [row for row in rows if row.get("review_bucket") == bucket]
    return bool(bucket_rows) and all(str(row.get("recorded_decision") or "") != "needs_review" for row in bucket_rows)


def build_candidate_layer_blocker_decision_record(
    *,
    candidate_layer_blocker_decision_template_report: str | Path,
    blocker_decisions_report: str | Path | None = None,
) -> dict[str, Any]:
    """Build a report-only decision record from blocker template rows."""

    template_path = Path(str(candidate_layer_blocker_decision_template_report)).expanduser()
    decisions_path = Path(str(blocker_decisions_report)).expanduser() if blocker_decisions_report else None
    template = _read_json(template_path)
    decisions_report = _read_json(decisions_path) if decisions_path else {}
    template_rows = _template_rows(template)
    unsafe_flags = _unsafe_flags(template, decisions_report)
    decisions = _decision_map(template_rows, decisions_report)
    rows = [_record_row(index, row, decisions) for index, row in enumerate(template_rows, start=1)]
    counts = _counts(rows, unsafe_flags)
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif _safe_int(counts.get("needsReviewRows")):
        status = "decision_record_required"
        decision = "manual_or_operator_decisions_still_required"
    else:
        status = "decision_recorded"
        decision = "blocker_decisions_recorded_non_strict"
    return {
        "schema": CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "candidateLayerBlockerDecisionTemplateReport": str(template_path),
            "candidateLayerBlockerDecisionTemplateSchema": str(template.get("schema") or ""),
            "blockerDecisionsReport": str(decisions_path or ""),
            "blockerDecisionInputRows": len(_decision_rows(decisions_report)),
        },
        "counts": counts,
        "gate": {
            "decisionRecordReady": bool(rows) and not unsafe_flags,
            "allDecisionRowsComplete": _all_non_needs_review(rows) and not unsafe_flags,
            "humanReviewComplete": _bucket_complete(rows, "manual_decision_required") and not unsafe_flags,
            "operatorApprovalComplete": _bucket_complete(rows, "operator_approval_required") and not unsafe_flags,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_record_candidate_layer_blocker_decisions"
            if _safe_int(counts.get("needsReviewRows"))
            else "candidate_layer_blocker_resolution_review_requires_explicit_approval",
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
            "operator_approval_records_do_not_execute_diagnostic_actions",
            "manual_approval_records_only_authorize_later_design_review_not_runtime_promotion",
            "strict_or_runtime_promotion_requires_a_separate_explicit_tranche",
        ],
        "decisionRecords": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_candidate_layer_blocker_decision_record_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Candidate Layer Blocker Decision Record",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Record rows: `{int(counts.get('recordRows') or 0)}`",
        f"- Needs review: `{int(counts.get('needsReviewRows') or 0)}`",
        f"- Manual approvals: `{int(counts.get('manualApprovalRows') or 0)}`",
        f"- Operator approvals: `{int(counts.get('operatorApprovedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This record is report-only. It does not create strict evidence, runtime citations, parser routing, canonical parsed artifacts, DB mutations, reindex, reembed, or answer integration.",
        "",
        "## Counts",
        "",
        f"- By bucket: `{json.dumps(counts.get('byBucket') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By priority: `{json.dumps(counts.get('byPriority') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By layer: `{json.dumps(counts.get('byLayer') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By decision: `{json.dumps(counts.get('byDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_candidate_layer_blocker_decision_record_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / "candidate-layer-blocker-decision-record.json"
    summary_path = root / "candidate-layer-blocker-decision-record-summary.json"
    markdown_path = root / "candidate-layer-blocker-decision-record.md"
    record_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_candidate_layer_blocker_decision_record_markdown(report), encoding="utf-8")
    return {"record": str(record_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only candidate-layer blocker decision record.")
    parser.add_argument("--candidate-layer-blocker-decision-template-report", required=True)
    parser.add_argument("--blocker-decisions-report", default="")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_candidate_layer_blocker_decision_record(
        candidate_layer_blocker_decision_template_report=args.candidate_layer_blocker_decision_template_report,
        blocker_decisions_report=args.blocker_decisions_report or None,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_candidate_layer_blocker_decision_record_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID",
    "build_candidate_layer_blocker_decision_record",
    "render_candidate_layer_blocker_decision_record_markdown",
    "write_candidate_layer_blocker_decision_record_reports",
]
