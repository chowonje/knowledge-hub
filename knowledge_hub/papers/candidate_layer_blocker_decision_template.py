"""Report-only decision template for candidate-layer blocker review cards.

The template is a worksheet for humans/operators. It does not record accepted
decisions, approve dependency work, create evidence, change parser routing,
write canonical artifacts, mutate SQLite, or change answer behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CANDIDATE_LAYER_BLOCKER_DECISION_TEMPLATE_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-template.v1"
)
CANDIDATE_LAYER_BLOCKER_REVIEW_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-review-pack.v1"
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


def _unsafe_flags(review_pack: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(review_pack.get("counts") or {})
    gate = dict(review_pack.get("gate") or {})
    policy = dict(review_pack.get("policy") or {})
    if review_pack.get("schema") != CANDIDATE_LAYER_BLOCKER_REVIEW_PACK_SCHEMA_ID:
        flags.append("candidate_layer_blocker_review_pack_schema_mismatch")
    if str(review_pack.get("status") or "") == "blocked":
        flags.append("candidate_layer_blocker_review_pack_blocked")
    for key in ("strictEligibleCards", "citationGradeCards", "runtimeEvidenceCards"):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"reviewPack_{key}_nonzero")
    for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(gate.get(key)):
            flags.append(f"reviewPack_{key}_true")
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
            flags.append(f"reviewPack_{key}_true")
    return list(dict.fromkeys(flags))


def _allowed_decisions(bucket: str) -> list[str]:
    if bucket == "manual_decision_required":
        return [
            "needs_review",
            "record_manual_approval_in_separate_decision_file",
            "record_manual_rejection_in_separate_decision_file",
            "keep_blocked",
        ]
    if bucket == "operator_approval_required":
        return [
            "needs_review",
            "approve_diagnostic_operator_action_in_separate_decision_file",
            "decline_diagnostic_operator_action_keep_blocked",
            "keep_blocked",
        ]
    if bucket == "technical_feasibility_blocked":
        return [
            "needs_review",
            "accept_technical_blocker_as_open",
            "defer_technical_followup",
            "close_as_not_needed",
        ]
    return [
        "needs_review",
        "accept_policy_blocker_as_guardrail",
        "defer_policy_review",
    ]


def _required_checks(bucket: str) -> list[str]:
    common = [
        "confirm_no_runtime_or_strict_evidence_is_authorized_by_this_row",
        "confirm_parser_routing_and_answer_integration_remain_out_of_scope",
    ]
    if bucket == "manual_decision_required":
        return [
            *common,
            "record reviewer and notes in a separate explicit decision file",
            "keep default needs_review if source evidence has not been inspected",
        ]
    if bucket == "operator_approval_required":
        return [
            *common,
            "approve only bounded diagnostic execution if explicitly intended",
            "confirm no global package install or canonical artifact write is implied",
        ]
    if bucket == "technical_feasibility_blocked":
        return [
            *common,
            "verify blocker still reflects the latest feasibility report",
            "choose a report-only followup or leave the blocker open",
        ]
    return [
        *common,
        "confirm candidate-only policy remains the desired guardrail",
        "defer any runtime promotion to a later explicit tranche",
    ]


def _decision_row(index: int, card: dict[str, Any]) -> dict[str, Any]:
    bucket = str(card.get("review_bucket") or "")
    source_card_id = str(card.get("review_card_id") or "")
    return {
        "decision_row_id": f"candidate-layer-blocker-decision:{index:04d}",
        "source_review_card_id": source_card_id,
        "source_backlog_id": str(card.get("source_backlog_id") or ""),
        "blocker": str(card.get("blocker") or ""),
        "priority": str(card.get("priority") or ""),
        "review_bucket": bucket,
        "affected_layers": list(card.get("affected_layers") or []),
        "affected_candidate_count": _safe_int(card.get("affected_candidate_count")),
        "affected_eval_question_count": _safe_int(card.get("affected_eval_question_count")),
        "recommended_next_tranche": str(card.get("recommended_next_tranche") or ""),
        "recommended_review_action": str(card.get("recommended_review_action") or ""),
        "default_decision": "needs_review",
        "allowed_decisions": _allowed_decisions(bucket),
        "manual_decision_input": {
            "source_review_card_id": source_card_id,
            "decision": "needs_review",
            "reviewer": "",
            "notes": "",
        },
        "required_review_checks": _required_checks(bucket),
        "decision_scope": "candidate_layer_blocker_decision_template_only_no_runtime_or_strict_promotion",
        "evidence_tier": "candidate_layer_blocker_decision_template_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "candidate_layer_blocker_decision_template_only",
            "decision_not_recorded",
            "strict_promotion_requires_later_explicit_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "decision_template_rows_are_not_human_or_operator_decisions",
            "decision_template_rows_do_not_authorize_runtime_use",
            "decision_template_rows_do_not_create_strict_evidence",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    by_bucket = Counter(str(row.get("review_bucket") or "") for row in rows)
    by_priority = Counter(str(row.get("priority") or "") for row in rows)
    by_layer = Counter(layer for row in rows for layer in list(row.get("affected_layers") or []))
    return {
        "templateRows": len(rows),
        "pendingDecisionRows": len(rows),
        "acceptedDecisionRows": 0,
        "rejectedDecisionRows": 0,
        "operatorApprovedRows": 0,
        "operatorDeclinedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byBucket": dict(by_bucket),
        "byPriority": dict(by_priority),
        "byLayer": dict(by_layer),
    }


def build_candidate_layer_blocker_decision_template(
    *,
    candidate_layer_blocker_review_pack_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only decision template from blocker review cards."""

    review_pack_path = Path(str(candidate_layer_blocker_review_pack_report)).expanduser()
    review_pack = _read_json(review_pack_path)
    unsafe_flags = _unsafe_flags(review_pack)
    cards = [dict(item) for item in list(review_pack.get("reviewCards") or []) if isinstance(item, dict)]
    rows = [_decision_row(index, card) for index, card in enumerate(cards, start=1)]
    counts = _counts(rows, unsafe_flags)
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif rows:
        status = "decision_template_ready"
        decision = "manual_or_operator_decision_template_ready"
    else:
        status = "no_decision_rows"
        decision = "no_review_cards_for_decision_template"
    return {
        "schema": CANDIDATE_LAYER_BLOCKER_DECISION_TEMPLATE_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "candidateLayerBlockerReviewPackReport": str(review_pack_path),
            "candidateLayerBlockerReviewPackSchema": str(review_pack.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "decisionTemplateReady": bool(rows) and not unsafe_flags,
            "humanReviewComplete": False,
            "operatorApprovalComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_record_candidate_layer_blocker_decisions"
            if rows
            else "candidate_layer_blocker_review_pack_refresh",
        },
        "policy": {
            "reportOnly": True,
            "decisionTemplateOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "decision_template_rows_are_not_decisions",
            "decision_template_rows_do_not_authorize_strict_or_runtime_evidence",
            "operator_approval_rows_require_separate_explicit_user_approval",
            "manual_decision_rows_require_a_separate_review_decision_file",
        ],
        "decisionRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_candidate_layer_blocker_decision_template_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Candidate Layer Blocker Decision Template",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Template rows: `{int(counts.get('templateRows') or 0)}`",
        f"- Pending decisions: `{int(counts.get('pendingDecisionRows') or 0)}`",
        f"- Accepted decisions: `{int(counts.get('acceptedDecisionRows') or 0)}`",
        f"- Operator-approved rows: `{int(counts.get('operatorApprovedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This template is a worksheet only. It does not record approvals, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By bucket: `{json.dumps(counts.get('byBucket') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By priority: `{json.dumps(counts.get('byPriority') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By layer: `{json.dumps(counts.get('byLayer') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_candidate_layer_blocker_decision_template_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    template_path = root / "candidate-layer-blocker-decision-template.json"
    summary_path = root / "candidate-layer-blocker-decision-template-summary.json"
    markdown_path = root / "candidate-layer-blocker-decision-template.md"
    template_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_candidate_layer_blocker_decision_template_markdown(report), encoding="utf-8")
    return {
        "template": str(template_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only candidate-layer blocker decision template.")
    parser.add_argument("--candidate-layer-blocker-review-pack-report", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = build_candidate_layer_blocker_decision_template(
        candidate_layer_blocker_review_pack_report=args.candidate_layer_blocker_review_pack_report
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_candidate_layer_blocker_decision_template_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_LAYER_BLOCKER_DECISION_TEMPLATE_SCHEMA_ID",
    "build_candidate_layer_blocker_decision_template",
    "render_candidate_layer_blocker_decision_template_markdown",
    "write_candidate_layer_blocker_decision_template_reports",
]
