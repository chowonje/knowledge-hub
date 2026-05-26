"""Report-only edit plan for candidate-layer blocker decision files.

This helper joins nonbinding blocker decision recommendations with an editable
review JSON and emits manual edit hints only. It does not write the decision
file, record human/operator decisions, create strict evidence, route parsers,
write canonical parsed artifacts, mutate SQLite, reindex, reembed, or change
answer behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CANDIDATE_LAYER_BLOCKER_DECISION_EDIT_PLAN_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-edit-plan.v1"
)
CANDIDATE_LAYER_BLOCKER_NONBINDING_DECISION_RECOMMENDATIONS_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-nonbinding-decision-recommendations.v1"
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
    return [dict(row) for row in list(rows or []) if isinstance(row, dict)]


def _recommendation_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in list(report.get("recommendationRows") or []) if isinstance(row, dict)]


def _decision_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("source_decision_row_id") or "")
        if row_id and row_id not in result:
            result[row_id] = row
    return result


def _allowed_decisions(decision_row: dict[str, Any], recommendation: dict[str, Any]) -> list[str]:
    allowed = [str(item) for item in list(decision_row.get("allowed_decisions") or []) if item]
    if not allowed:
        allowed = [str(item) for item in list(recommendation.get("allowed_decisions") or []) if item]
    if "needs_review" not in allowed:
        allowed = ["needs_review", *allowed]
    return list(dict.fromkeys(allowed))


def _current_decision(decision_row: dict[str, Any], recommendation: dict[str, Any]) -> str:
    return str(decision_row.get("decision") or recommendation.get("current_decision") or "needs_review")


def _unsafe_flags(recommendations: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(recommendations.get("counts") or {})
    gate = dict(recommendations.get("gate") or {})
    policy = dict(recommendations.get("policy") or {})
    if recommendations.get("schema") != CANDIDATE_LAYER_BLOCKER_NONBINDING_DECISION_RECOMMENDATIONS_SCHEMA_ID:
        flags.append("candidate_layer_blocker_nonbinding_recommendations_schema_mismatch")
    if recommendations and recommendations.get("status") != "nonbinding_recommendations_ready":
        flags.append("candidate_layer_blocker_nonbinding_recommendations_not_ready")
    for key in ("decisionRowsModified", "strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"recommendations_{key}_nonzero")
    for key in (
        "containsRecordedDecisions",
        "reviewCopyModified",
        "strictEvidenceReady",
        "parserRoutingReady",
        "answerIntegrationReady",
        "runtimePromotionAllowed",
    ):
        if bool(gate.get(key)):
            flags.append(f"recommendations_{key}_true")
    for key in (
        "decisionFileModified",
        "decisionRecordCreated",
        "strictEvidenceCreated",
        "runtimePromotionAllowed",
        "parserRoutingChanged",
        "canonicalParsedArtifactsWritten",
        "databaseMutation",
        "reindexOrReembed",
        "answerIntegrationChanged",
    ):
        if bool(policy.get(key)):
            flags.append(f"recommendations_{key}_true")
    return list(dict.fromkeys(flags))


def _edit_row(index: int, recommendation: dict[str, Any], decision_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row_id = str(recommendation.get("source_decision_row_id") or "")
    decision_row = decision_by_id.get(row_id) or {}
    allowed = _allowed_decisions(decision_row, recommendation)
    recommended_decision = str(recommendation.get("recommended_decision") or "needs_review")
    current_decision = _current_decision(decision_row, recommendation)
    reviewer_present = bool(str(decision_row.get("reviewer") or ""))
    notes_present = bool(str(decision_row.get("notes") or ""))
    missing_decision_row = not bool(decision_row)
    recommendation_allowed = bool(decision_row) and recommended_decision in allowed
    if missing_decision_row:
        edit_status = "blocked_missing_decision_file_row"
    elif not recommendation_allowed:
        edit_status = "blocked_recommendation_not_allowed"
    else:
        edit_status = "ready_for_manual_edit"
    reviewer_required = recommended_decision != "needs_review"
    notes_required = recommended_decision != "needs_review"
    manual_edit_required = (
        edit_status == "ready_for_manual_edit"
        and (
            recommended_decision != current_decision
            or (reviewer_required and not reviewer_present)
            or (notes_required and not notes_present)
        )
    )
    return {
        "edit_row_id": f"candidate-layer-blocker-decision-edit-plan:{index:04d}",
        "source_recommendation_row_id": str(recommendation.get("recommendation_row_id") or ""),
        "source_input_row_id": str(recommendation.get("source_input_row_id") or ""),
        "source_decision_row_id": row_id,
        "source_review_card_id": str(recommendation.get("source_review_card_id") or ""),
        "source_backlog_id": str(recommendation.get("source_backlog_id") or ""),
        "blocker": str(recommendation.get("blocker") or ""),
        "priority": str(recommendation.get("priority") or ""),
        "review_bucket": str(recommendation.get("review_bucket") or ""),
        "affected_layers": list(recommendation.get("affected_layers") or []),
        "current_decision": current_decision,
        "recommended_decision": recommended_decision,
        "recommendation_rationale": str(recommendation.get("recommendation_rationale") or ""),
        "recommendation_confidence": str(recommendation.get("recommendation_confidence") or ""),
        "allowed_decisions": allowed,
        "edit_status": edit_status,
        "manual_edit_required": manual_edit_required,
        "reviewer_required": reviewer_required,
        "notes_required": notes_required,
        "reviewer_present": reviewer_present,
        "notes_present": notes_present,
        "accepted_as_human_decision": False,
        "decision_file_modified": False,
        "decision_file_patch_hint": {
            "source_decision_row_id": row_id,
            "decision": recommended_decision,
            "reviewer": "",
            "notes": "",
        },
        "evidence_tier": "candidate_layer_blocker_decision_edit_plan_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "candidate_layer_blocker_decision_edit_plan_only",
            "human_or_operator_decision_not_recorded",
            "decision_file_not_modified",
            "strict_promotion_requires_later_explicit_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "edit_plan_rows_are_not_human_or_operator_decisions",
            "edit_plan_rows_do_not_modify_the_decision_file",
            "edit_plan_rows_do_not_record_decisions",
            "edit_plan_rows_do_not_create_strict_evidence",
            "edit_plan_rows_do_not_authorize_runtime_use",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    by_status = Counter(str(row.get("edit_status") or "") for row in rows)
    by_current = Counter(str(row.get("current_decision") or "") for row in rows)
    by_recommended = Counter(str(row.get("recommended_decision") or "") for row in rows)
    by_bucket = Counter(str(row.get("review_bucket") or "") for row in rows)
    by_layer = Counter(layer for row in rows for layer in list(row.get("affected_layers") or []))
    return {
        "editRows": len(rows),
        "readyForManualEditRows": by_status.get("ready_for_manual_edit", 0),
        "blockedMissingDecisionFileRows": by_status.get("blocked_missing_decision_file_row", 0),
        "blockedRecommendationNotAllowedRows": by_status.get("blocked_recommendation_not_allowed", 0),
        "manualEditRequiredRows": sum(1 for row in rows if row.get("manual_edit_required")),
        "noEditRequiredRows": sum(
            1 for row in rows if row.get("edit_status") == "ready_for_manual_edit" and not row.get("manual_edit_required")
        ),
        "currentNeedsReviewRows": by_current.get("needs_review", 0),
        "currentNonNeedsReviewRows": len(rows) - by_current.get("needs_review", 0),
        "proposedAcceptTechnicalOpenRows": by_recommended.get("accept_technical_blocker_as_open", 0),
        "proposedAcceptPolicyGuardrailRows": by_recommended.get("accept_policy_blocker_as_guardrail", 0),
        "proposedNeedsReviewRows": by_recommended.get("needs_review", 0),
        "decisionRowsModified": 0,
        "acceptedHumanDecisionRows": 0,
        "operatorApprovedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byEditStatus": dict(by_status),
        "byCurrentDecision": dict(by_current),
        "byRecommendedDecision": dict(by_recommended),
        "byBucket": dict(by_bucket),
        "byLayer": dict(by_layer),
    }


def build_candidate_layer_blocker_decision_edit_plan(
    *,
    candidate_layer_blocker_nonbinding_decision_recommendations_report: str | Path,
    candidate_layer_blocker_decisions_file: str | Path,
) -> dict[str, Any]:
    """Build a report-only manual edit plan for blocker decisions."""

    recommendations_path = Path(str(candidate_layer_blocker_nonbinding_decision_recommendations_report)).expanduser()
    decisions_path = Path(str(candidate_layer_blocker_decisions_file)).expanduser()
    recommendations = _read_json(recommendations_path)
    decision_file = _read_json(decisions_path)
    unsafe_flags = _unsafe_flags(recommendations)
    decision_by_id = _decision_by_id(_decision_rows(decision_file))
    rows = [
        _edit_row(index, row, decision_by_id)
        for index, row in enumerate(_recommendation_rows(recommendations), start=1)
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
        decision = "manual_edit_still_required" if _safe_int(counts.get("manualEditRequiredRows")) else "no_manual_edits_required"
    else:
        status = "no_edit_rows"
        decision = "no_candidate_layer_blocker_recommendations"
    return {
        "schema": CANDIDATE_LAYER_BLOCKER_DECISION_EDIT_PLAN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "candidateLayerBlockerNonbindingDecisionRecommendationsReport": str(recommendations_path),
            "candidateLayerBlockerNonbindingDecisionRecommendationsSchema": str(recommendations.get("schema") or ""),
            "candidateLayerBlockerDecisionsFile": str(decisions_path),
            "candidateLayerBlockerDecisionRows": len(_decision_rows(decision_file)),
        },
        "counts": counts,
        "gate": {
            "editPlanReady": bool(rows) and not blocked,
            "manualDecisionFileEditRequired": bool(rows) and not blocked and _safe_int(counts.get("manualEditRequiredRows")) > 0,
            "decisionFileModified": False,
            "decisionsRecorded": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_edit_candidate_layer_blocker_decisions_review_json",
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
            "edit_plan_rows_are_not_human_or_operator_decisions",
            "edit_plan_does_not_write_or_modify_the_decision_file",
            "patch_hints_must_be_reviewed_by_a_human_or_operator_before_validation_or_decision_record_generation",
            "strict_or_runtime_promotion_requires_a_separate_explicit_tranche",
        ],
        "editRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_candidate_layer_blocker_decision_edit_plan_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Candidate Layer Blocker Decision Edit Plan",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Edit rows: `{int(counts.get('editRows') or 0)}`",
        f"- Manual edit required rows: `{int(counts.get('manualEditRequiredRows') or 0)}`",
        f"- No edit required rows: `{int(counts.get('noEditRequiredRows') or 0)}`",
        f"- Proposed technical-open acceptances: `{int(counts.get('proposedAcceptTechnicalOpenRows') or 0)}`",
        f"- Proposed policy guardrails: `{int(counts.get('proposedAcceptPolicyGuardrailRows') or 0)}`",
        f"- Accepted human/operator decisions: `0`",
        f"- Strict/citation/runtime evidence rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This edit plan is not a decision file. It does not modify the review JSON, record decisions, create strict evidence, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By edit status: `{json.dumps(counts.get('byEditStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By current decision: `{json.dumps(counts.get('byCurrentDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By recommended decision: `{json.dumps(counts.get('byRecommendedDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By bucket: `{json.dumps(counts.get('byBucket') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_candidate_layer_blocker_decision_edit_plan_reports(
    report: dict[str, Any], output_dir: str | Path
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "candidate-layer-blocker-decision-edit-plan.json"
    summary_path = root / "candidate-layer-blocker-decision-edit-plan-summary.json"
    markdown_path = root / "candidate-layer-blocker-decision-edit-plan.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_candidate_layer_blocker_decision_edit_plan_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only candidate-layer blocker decision edit plan.")
    parser.add_argument("--candidate-layer-blocker-nonbinding-decision-recommendations-report", required=True)
    parser.add_argument("--candidate-layer-blocker-decisions-file", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = build_candidate_layer_blocker_decision_edit_plan(
        candidate_layer_blocker_nonbinding_decision_recommendations_report=(
            args.candidate_layer_blocker_nonbinding_decision_recommendations_report
        ),
        candidate_layer_blocker_decisions_file=args.candidate_layer_blocker_decisions_file,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_candidate_layer_blocker_decision_edit_plan_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_LAYER_BLOCKER_DECISION_EDIT_PLAN_SCHEMA_ID",
    "build_candidate_layer_blocker_decision_edit_plan",
    "render_candidate_layer_blocker_decision_edit_plan_markdown",
    "write_candidate_layer_blocker_decision_edit_plan_reports",
]
