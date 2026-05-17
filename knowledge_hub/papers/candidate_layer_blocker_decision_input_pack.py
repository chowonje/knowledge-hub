"""Report-only input pack for candidate-layer blocker decisions.

The input pack prepares a schema-backed file a human/operator can copy into a
separate decision file later. It deliberately keeps every decision at
``needs_review`` and records no approvals.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CANDIDATE_LAYER_BLOCKER_DECISION_INPUT_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-input-pack.v1"
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


def _template_rows(template: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(template.get("decisionRows") or []) if isinstance(item, dict)]


def _record_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(record.get("decisionRecords") or []) if isinstance(item, dict)]


def _record_by_source_decision_id(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("source_decision_row_id") or ""): item for item in _record_rows(record)}


def _unsafe_flags(template: dict[str, Any], record: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    template_counts = dict(template.get("counts") or {})
    template_gate = dict(template.get("gate") or {})
    template_policy = dict(template.get("policy") or {})
    if template.get("schema") != CANDIDATE_LAYER_BLOCKER_DECISION_TEMPLATE_SCHEMA_ID:
        flags.append("candidate_layer_blocker_decision_template_schema_mismatch")
    if str(template.get("status") or "") == "blocked":
        flags.append("candidate_layer_blocker_decision_template_blocked")
    for key in ("acceptedDecisionRows", "operatorApprovedRows", "strictEligibleRows", "runtimeEvidenceRows"):
        if _safe_int(template_counts.get(key)) > 0:
            flags.append(f"decisionTemplate_{key}_nonzero")
    for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(template_gate.get(key)):
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
        if bool(template_policy.get(key)):
            flags.append(f"decisionTemplate_{key}_true")

    if record:
        record_counts = dict(record.get("counts") or {})
        record_gate = dict(record.get("gate") or {})
        record_policy = dict(record.get("policy") or {})
        if record.get("schema") != CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID:
            flags.append("candidate_layer_blocker_decision_record_schema_mismatch")
        if str(record.get("status") or "") == "blocked":
            flags.append("candidate_layer_blocker_decision_record_blocked")
        for key in ("strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
            if _safe_int(record_counts.get(key)) > 0:
                flags.append(f"decisionRecord_{key}_nonzero")
        for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
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
            if bool(record_policy.get(key)):
                flags.append(f"decisionRecord_{key}_true")
    return list(dict.fromkeys(flags))


def _pending_template_rows(template: dict[str, Any], record: dict[str, Any]) -> list[dict[str, Any]]:
    by_source_id = _record_by_source_decision_id(record)
    rows: list[dict[str, Any]] = []
    for row in _template_rows(template):
        row_id = str(row.get("decision_row_id") or "")
        if not record:
            rows.append(row)
            continue
        record_row = by_source_id.get(row_id)
        if not record_row or str(record_row.get("recorded_decision") or "") == "needs_review":
            rows.append(row)
    return rows


def _input_row(index: int, template_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_row_id": f"candidate-layer-blocker-decision-input:{index:04d}",
        "source_decision_row_id": str(template_row.get("decision_row_id") or ""),
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
        "allowed_decisions": list(template_row.get("allowed_decisions") or []),
        "decision": "needs_review",
        "reviewer": "",
        "notes": "",
        "decision_scope": "candidate_layer_blocker_decision_input_pack_only_no_runtime_or_strict_promotion",
        "evidence_tier": "candidate_layer_blocker_decision_input_pack_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "candidate_layer_blocker_decision_input_pack_only",
            "decision_not_recorded",
            "strict_promotion_requires_later_explicit_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "decision_input_rows_are_not_decisions",
            "decision_input_rows_do_not_authorize_runtime_use",
            "decision_input_rows_do_not_create_strict_evidence",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    by_bucket = Counter(str(row.get("review_bucket") or "") for row in rows)
    by_priority = Counter(str(row.get("priority") or "") for row in rows)
    by_layer = Counter(layer for row in rows for layer in list(row.get("affected_layers") or []))
    return {
        "inputRows": len(rows),
        "pendingSourceDecisionRows": len(rows),
        "defaultNeedsReviewRows": sum(1 for row in rows if row.get("decision") == "needs_review"),
        "manualDecisionInputRows": by_bucket.get("manual_decision_required", 0),
        "operatorApprovalInputRows": by_bucket.get("operator_approval_required", 0),
        "technicalDecisionInputRows": by_bucket.get("technical_feasibility_blocked", 0),
        "policyDecisionInputRows": by_bucket.get("policy_review_only", 0),
        "acceptedDecisionRows": 0,
        "operatorApprovedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byBucket": dict(by_bucket),
        "byPriority": dict(by_priority),
        "byLayer": dict(by_layer),
    }


def build_candidate_layer_blocker_decision_input_pack(
    *,
    candidate_layer_blocker_decision_template_report: str | Path,
    candidate_layer_blocker_decision_record_report: str | Path | None = None,
) -> dict[str, Any]:
    """Build a report-only pending-decision input pack for blocker rows."""

    template_path = Path(str(candidate_layer_blocker_decision_template_report)).expanduser()
    record_path = (
        Path(str(candidate_layer_blocker_decision_record_report)).expanduser()
        if candidate_layer_blocker_decision_record_report
        else None
    )
    template = _read_json(template_path)
    record = _read_json(record_path) if record_path else {}
    unsafe_flags = _unsafe_flags(template, record)
    rows = [_input_row(index, row) for index, row in enumerate(_pending_template_rows(template, record), start=1)]
    counts = _counts(rows, unsafe_flags)
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif rows:
        status = "decision_input_pack_ready"
        decision = "pending_decision_inputs_ready"
    else:
        status = "no_pending_decision_inputs"
        decision = "no_pending_decision_rows"
    return {
        "schema": CANDIDATE_LAYER_BLOCKER_DECISION_INPUT_PACK_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "candidateLayerBlockerDecisionTemplateReport": str(template_path),
            "candidateLayerBlockerDecisionTemplateSchema": str(template.get("schema") or ""),
            "candidateLayerBlockerDecisionRecordReport": str(record_path or ""),
            "candidateLayerBlockerDecisionRecordSchema": str(record.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "decisionInputPackReady": bool(rows) and not unsafe_flags,
            "containsAcceptedDecisions": False,
            "containsOperatorApprovals": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_fill_candidate_layer_blocker_decision_input_file"
            if rows
            else "candidate_layer_blocker_backlog_refresh",
        },
        "policy": {
            "reportOnly": True,
            "decisionInputPackOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "decision_input_rows_are_not_decisions",
            "decision_input_rows_default_to_needs_review",
            "operator_approval_inputs_do_not_execute_diagnostic_actions",
            "strict_or_runtime_promotion_requires_a_separate_explicit_tranche",
        ],
        "decisionInputs": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_candidate_layer_blocker_decision_input_pack_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Candidate Layer Blocker Decision Input Pack",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Input rows: `{int(counts.get('inputRows') or 0)}`",
        f"- Default `needs_review` rows: `{int(counts.get('defaultNeedsReviewRows') or 0)}`",
        f"- Accepted decisions: `{int(counts.get('acceptedDecisionRows') or 0)}`",
        f"- Operator approvals: `{int(counts.get('operatorApprovedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This input pack is a worksheet source only. It does not record decisions, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By bucket: `{json.dumps(counts.get('byBucket') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By priority: `{json.dumps(counts.get('byPriority') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By layer: `{json.dumps(counts.get('byLayer') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_candidate_layer_blocker_decision_input_pack_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    pack_path = root / "candidate-layer-blocker-decision-input-pack.json"
    summary_path = root / "candidate-layer-blocker-decision-input-pack-summary.json"
    markdown_path = root / "candidate-layer-blocker-decision-input-pack.md"
    pack_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_candidate_layer_blocker_decision_input_pack_markdown(report), encoding="utf-8")
    return {"inputPack": str(pack_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only candidate-layer blocker decision input pack.")
    parser.add_argument("--candidate-layer-blocker-decision-template-report", required=True)
    parser.add_argument("--candidate-layer-blocker-decision-record-report", default="")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_candidate_layer_blocker_decision_input_pack(
        candidate_layer_blocker_decision_template_report=args.candidate_layer_blocker_decision_template_report,
        candidate_layer_blocker_decision_record_report=args.candidate_layer_blocker_decision_record_report or None,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_candidate_layer_blocker_decision_input_pack_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_LAYER_BLOCKER_DECISION_INPUT_PACK_SCHEMA_ID",
    "build_candidate_layer_blocker_decision_input_pack",
    "render_candidate_layer_blocker_decision_input_pack_markdown",
    "write_candidate_layer_blocker_decision_input_pack_reports",
]
