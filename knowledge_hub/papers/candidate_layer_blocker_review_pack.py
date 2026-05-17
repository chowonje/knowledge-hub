"""Report-only review pack for candidate-layer blocker backlog items.

This module converts the blocker backlog into operator-review cards. It does
not approve decisions, create evidence, change parser routing, write canonical
parsed artifacts, mutate SQLite, or change answer behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CANDIDATE_LAYER_BLOCKER_REVIEW_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-review-pack.v1"
)
CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID = "knowledge-hub.paper.candidate-layer-blocker-backlog.v1"

_MANUAL_DECISION_BLOCKERS = {
    "sectionspan_pdf_offset_human_review_pending",
    "sectionspan_selected_review_decision_file_required",
    "sectionspan_selected_review_manual_edit_required",
    "sectionspan_pdf_offsets_require_human_review_before_strict_promotion",
    "equation_quote_decision_manual_edit_required",
    "candidate_layer_blocker_decision_record_pending",
}
_OPERATOR_APPROVAL_BLOCKERS = {
    "table_cell_isolated_extractor_approval_required",
}
_POLICY_REVIEW_BLOCKERS = {
    "candidate_layers_are_report_only",
    "runtime_promotion_disabled_for_tranche",
}


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


def _review_bucket(blocker: str) -> str:
    if blocker in _MANUAL_DECISION_BLOCKERS:
        return "manual_decision_required"
    if blocker in _OPERATOR_APPROVAL_BLOCKERS:
        return "operator_approval_required"
    if blocker in _POLICY_REVIEW_BLOCKERS:
        return "policy_review_only"
    return "technical_feasibility_blocked"


def _review_action(blocker: str, fallback: str) -> str:
    actions = {
        "sectionspan_selected_review_decision_file_required": "provide_selected_sectionspan_decision_file_or_keep_pending",
        "sectionspan_selected_review_manual_edit_required": "manually_edit_selected_sectionspan_decision_file_or_keep_pending",
        "sectionspan_pdf_offset_human_review_pending": "review_sectionspan_original_pdf_offset_rows",
        "sectionspan_pdf_offsets_require_human_review_before_strict_promotion": "run_or_refresh_sectionspan_human_review_gate",
        "table_cell_isolated_extractor_approval_required": "approve_or_decline_isolated_table_cell_extractor_pilot",
        "equation_quote_alignment_missing": "continue_equation_alignment_or_normalization_audit",
        "table_cell_row_column_bbox_provenance_missing": "continue_table_cell_provenance_review_without_promoting_cells",
        "figure_region_link_unverified": "continue_figure_region_link_review_without_promoting_figures",
        "non_sectionspan_layers_lack_original_pdf_offsets": "continue_non_sectionspan_pdf_offset_recovery_review",
        "figure_caption_pdf_offsets_require_region_link_review": "review_caption_source_span_to_figure_region_links",
        "table_caption_pdf_offsets_require_cell_provenance_review": "review_table_caption_offsets_before_cell_provenance",
        "equation_quote_decision_manual_edit_required": "manually_edit_equation_quote_decision_file_or_keep_pending",
        "candidate_layer_blocker_decision_record_pending": "record_candidate_layer_blocker_decisions_or_keep_pending",
        "candidate_layers_are_report_only": "keep_candidate_layer_contract_report_only",
        "runtime_promotion_disabled_for_tranche": "defer_runtime_promotion_to_explicit_later_tranche",
    }
    return actions.get(blocker, fallback or "manual_review")


def _card(index: int, item: dict[str, Any]) -> dict[str, Any]:
    blocker = str(item.get("blocker") or "")
    bucket = _review_bucket(blocker)
    strict_blockers = list(dict.fromkeys(str(value) for value in list(item.get("strict_blockers") or []) if value))
    if "candidate_layer_blocker_review_pack_only" not in strict_blockers:
        strict_blockers.append("candidate_layer_blocker_review_pack_only")
    if "runtime_promotion_disabled_for_tranche" not in strict_blockers:
        strict_blockers.append("runtime_promotion_disabled_for_tranche")
    return {
        "review_card_id": f"candidate-layer-blocker-review:{index:03d}",
        "source_backlog_id": str(item.get("backlog_id") or ""),
        "blocker": blocker,
        "priority": str(item.get("priority") or ""),
        "review_bucket": bucket,
        "affected_layers": list(item.get("affected_layers") or []),
        "affected_candidate_count": _safe_int(item.get("affected_candidate_count")),
        "affected_eval_question_count": _safe_int(item.get("affected_eval_question_count")),
        "recommended_next_tranche": str(item.get("recommendedNextTranche") or ""),
        "recommended_review_action": _review_action(blocker, str(item.get("recommendedNextTranche") or "")),
        "requires_human_decision": bucket == "manual_decision_required",
        "requires_operator_approval": bucket == "operator_approval_required",
        "requires_technical_followup": bucket == "technical_feasibility_blocked",
        "policy_only": bucket == "policy_review_only",
        "allowed_actions": list(item.get("allowedActions") or []),
        "disallowed_actions": list(item.get("disallowedActions") or []),
        "stop_rule": str(item.get("stopRule") or ""),
        "evidence_needed_before_promotion": list(item.get("evidenceNeededBeforePromotion") or []),
        "evidence_tier": "candidate_layer_blocker_review_card_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": [
            "review_cards_are_not_evidence",
            "review_cards_do_not_authorize_runtime_use",
            "explicit_later_promotion_tranche_required",
        ],
    }


def _schema_violations(backlog: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if backlog.get("schema") != CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID:
        violations.append("candidate_layer_blocker_backlog_schema_mismatch")
    if str(backlog.get("status") or "") != "ok":
        violations.append("candidate_layer_blocker_backlog_not_ok")
    for violation in list((backlog.get("gate") or {}).get("schemaViolations") or []):
        violations.append(f"candidate_layer_blocker_backlog_upstream:{violation}")
    return list(dict.fromkeys(violations))


def build_candidate_layer_blocker_review_pack(*, candidate_layer_blocker_backlog_report: str | Path) -> dict[str, Any]:
    """Build report-only operator review cards from a candidate-layer backlog."""

    backlog_path = Path(str(candidate_layer_blocker_backlog_report)).expanduser()
    backlog = _read_json(backlog_path)
    violations = _schema_violations(backlog)
    cards = [_card(index, item) for index, item in enumerate(list(backlog.get("backlog") or []), start=1) if isinstance(item, dict)]
    by_bucket = Counter(str(card.get("review_bucket") or "") for card in cards)
    by_priority = Counter(str(card.get("priority") or "") for card in cards)
    by_layer = Counter(layer for card in cards for layer in list(card.get("affected_layers") or []))
    counts = {
        "reviewCardCount": len(cards),
        "manualDecisionRequiredCards": _safe_int(by_bucket.get("manual_decision_required")),
        "operatorApprovalRequiredCards": _safe_int(by_bucket.get("operator_approval_required")),
        "technicalFeasibilityBlockedCards": _safe_int(by_bucket.get("technical_feasibility_blocked")),
        "policyReviewOnlyCards": _safe_int(by_bucket.get("policy_review_only")),
        "affectedCandidateCount": sum(_safe_int(card.get("affected_candidate_count")) for card in cards),
        "affectedEvalQuestionCount": sum(_safe_int(card.get("affected_eval_question_count")) for card in cards),
        "strictEligibleCards": 0,
        "citationGradeCards": 0,
        "runtimeEvidenceCards": 0,
        "schemaViolationCount": len(violations),
        "byBucket": dict(by_bucket),
        "byPriority": dict(by_priority),
        "byLayer": dict(by_layer),
    }
    return {
        "schema": CANDIDATE_LAYER_BLOCKER_REVIEW_PACK_SCHEMA_ID,
        "status": "review_pack_ready" if cards and not violations else "blocked",
        "generatedAt": _now(),
        "inputs": {
            "candidateLayerBlockerBacklogReport": str(backlog_path),
            "candidateLayerBlockerBacklogSchema": str(backlog.get("schema") or ""),
            "expectedCandidateLayerBlockerBacklogSchema": CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID,
        },
        "counts": counts,
        "gate": {
            "reviewPackReady": bool(cards and not violations),
            "manualDecisionRequired": counts["manualDecisionRequiredCards"] > 0,
            "operatorApprovalRequired": counts["operatorApprovalRequiredCards"] > 0,
            "technicalFollowupRequired": counts["technicalFeasibilityBlockedCards"] > 0,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "review_pack_ready" if cards and not violations else "blocked",
            "schemaViolations": violations,
            "recommendedNextTranche": "manual_or_operator_review_of_blocker_cards",
        },
        "policy": {
            "reviewPackOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "review_cards_are_not_evidence",
            "review_cards_do_not_authorize_runtime_candidate_use",
            "affected_candidate_counts_are_overlapping_backlog_references_not_unique_candidates",
            "manual_decision_cards_require_explicit_human_input",
            "operator_approval_cards_require_explicit_user_approval",
            "parser_routing_and_answer_integration_remain_out_of_scope",
        ],
        "reviewCards": cards,
    }


def render_candidate_layer_blocker_review_pack_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Candidate Layer Blocker Review Pack",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Review cards: `{int(counts.get('reviewCardCount') or 0)}`",
        f"- Manual decision required: `{int(counts.get('manualDecisionRequiredCards') or 0)}`",
        f"- Operator approval required: `{int(counts.get('operatorApprovalRequiredCards') or 0)}`",
        f"- Technical feasibility blocked: `{int(counts.get('technicalFeasibilityBlockedCards') or 0)}`",
        f"- Policy review only: `{int(counts.get('policyReviewOnlyCards') or 0)}`",
        f"- Strict eligible cards: `{int(counts.get('strictEligibleCards') or 0)}`",
        f"- Runtime evidence cards: `{int(counts.get('runtimeEvidenceCards') or 0)}`",
        "",
        "## Safety",
        "",
        "This review pack is report-only. It does not create strict evidence, parser routing, answer integration, canonical parsed artifacts, DB mutations, or index/embedding changes.",
        "",
        "## Cards",
        "",
    ]
    for card in list(report.get("reviewCards") or []):
        lines.extend(
            [
                f"### `{card.get('review_card_id', '')}`",
                "",
                f"- Blocker: `{card.get('blocker', '')}`",
                f"- Priority: `{card.get('priority', '')}`",
                f"- Bucket: `{card.get('review_bucket', '')}`",
                f"- Affected candidates: `{int(card.get('affected_candidate_count') or 0)}`",
                f"- Review action: `{card.get('recommended_review_action', '')}`",
                f"- Stop rule: `{card.get('stop_rule', '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_candidate_layer_blocker_review_pack_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    cards_path = root / "candidate-layer-blocker-review-cards.json"
    summary_path = root / "candidate-layer-blocker-review-summary.json"
    markdown_path = root / "candidate-layer-blocker-review-pack.md"
    cards_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_payload = {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_candidate_layer_blocker_review_pack_markdown(report), encoding="utf-8")
    return {
        "cards": str(cards_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only candidate-layer blocker review pack.")
    parser.add_argument("--candidate-layer-blocker-backlog-report", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = build_candidate_layer_blocker_review_pack(
        candidate_layer_blocker_backlog_report=args.candidate_layer_blocker_backlog_report
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_candidate_layer_blocker_review_pack_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_LAYER_BLOCKER_REVIEW_PACK_SCHEMA_ID",
    "build_candidate_layer_blocker_review_pack",
    "render_candidate_layer_blocker_review_pack_markdown",
    "write_candidate_layer_blocker_review_pack_reports",
]
