"""Report-only nonbinding recommendations for candidate-layer blocker decisions.

This module suggests manual decision values for an editable blocker decision
file, but it does not edit the file, record decisions, create evidence, change
parser routing, write canonical parsed artifacts, mutate SQLite, or change
answer behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CANDIDATE_LAYER_BLOCKER_NONBINDING_DECISION_RECOMMENDATIONS_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-nonbinding-decision-recommendations.v1"
)
CANDIDATE_LAYER_BLOCKER_DECISION_INPUT_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-input-pack.v1"
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


def _decision_rows_by_id(decisions_file: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = list(decisions_file.get("decisions") or [])
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("source_decision_row_id") or "")
        if row_id and row_id not in result:
            result[row_id] = row
    return result


def _decision_index_by_id(decisions_file: dict[str, Any]) -> dict[str, int]:
    rows = list(decisions_file.get("decisions") or [])
    result: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("source_decision_row_id") or "")
        if row_id and row_id not in result:
            result[row_id] = index
    return result


def _recommendation_for_bucket(bucket: str, allowed_decisions: list[str]) -> tuple[str, str]:
    if bucket == "technical_feasibility_blocked":
        decision = "accept_technical_blocker_as_open"
        if decision in allowed_decisions:
            return decision, "acknowledge_the_technical_blocker_as_open_without_resolving_or_promoting"
    if bucket == "policy_review_only":
        decision = "accept_policy_blocker_as_guardrail"
        if decision in allowed_decisions:
            return decision, "preserve_candidate_only_runtime_disabled_policy_as_guardrail"
    if bucket == "operator_approval_required":
        return "needs_review", "requires_explicit_operator_approval_before_any_operator_action_is_recorded"
    if bucket == "manual_decision_required":
        return "needs_review", "requires_human_review_of_source_evidence_before_any_non_needs_review_decision"
    return "needs_review", "no_safe_nonbinding_recommendation_available"


def _row(
    index: int,
    input_row: dict[str, Any],
    current_decisions: dict[str, dict[str, Any]],
    decision_indexes: dict[str, int],
) -> dict[str, Any]:
    row_id = str(input_row.get("source_decision_row_id") or "")
    current_row = current_decisions.get(row_id, {})
    decision_index = decision_indexes.get(row_id, index - 1)
    allowed_decisions = [str(value) for value in list(input_row.get("allowed_decisions") or []) if value]
    bucket = str(input_row.get("review_bucket") or "")
    recommended_decision, rationale = _recommendation_for_bucket(bucket, allowed_decisions)
    current_decision = str(current_row.get("decision") or input_row.get("decision") or "needs_review")
    current_reviewer = str(current_row.get("reviewer") or "")
    current_notes = str(current_row.get("notes") or "")
    manual_edit_required = recommended_decision != current_decision
    return {
        "recommendation_row_id": f"candidate-layer-blocker-decision-recommendation:{index:04d}",
        "source_input_row_id": str(input_row.get("input_row_id") or ""),
        "source_decision_row_id": row_id,
        "source_review_card_id": str(input_row.get("source_review_card_id") or ""),
        "source_backlog_id": str(input_row.get("source_backlog_id") or ""),
        "blocker": str(input_row.get("blocker") or ""),
        "priority": str(input_row.get("priority") or ""),
        "review_bucket": bucket,
        "affected_layers": list(input_row.get("affected_layers") or []),
        "affected_candidate_count": _safe_int(input_row.get("affected_candidate_count")),
        "affected_eval_question_count": _safe_int(input_row.get("affected_eval_question_count")),
        "current_decision": current_decision,
        "current_reviewer_present": bool(current_reviewer),
        "current_notes_present": bool(current_notes),
        "recommended_decision": recommended_decision,
        "recommendation_rationale": rationale,
        "recommendation_confidence": "medium" if recommended_decision != "needs_review" else "low",
        "allowed_decisions": allowed_decisions,
        "manual_edit_required": manual_edit_required,
        "decision_pointer": f"/decisions/{decision_index}/decision",
        "reviewer_pointer": f"/decisions/{decision_index}/reviewer",
        "notes_pointer": f"/decisions/{decision_index}/notes",
        "decision_file_patch_hint": {
            "source_decision_row_id": row_id,
            "decision": recommended_decision,
            "reviewer": "",
            "notes": "",
        },
        "evidence_tier": "candidate_layer_blocker_nonbinding_recommendation_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "candidate_layer_blocker_nonbinding_recommendation_only",
            "recommendation_is_not_a_human_decision",
            "decision_file_not_modified",
            "strict_promotion_requires_later_explicit_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "recommendation_rows_are_not_decisions",
            "recommendation_rows_do_not_modify_the_decision_file",
            "recommendation_rows_do_not_authorize_runtime_use",
            "recommendation_rows_do_not_create_strict_evidence",
        ],
    }


def _unsafe_flags(input_pack: dict[str, Any], decisions_file: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if input_pack.get("schema") != CANDIDATE_LAYER_BLOCKER_DECISION_INPUT_PACK_SCHEMA_ID:
        flags.append("candidate_layer_blocker_decision_input_pack_schema_mismatch")
    if input_pack and input_pack.get("status") != "decision_input_pack_ready":
        flags.append("candidate_layer_blocker_decision_input_pack_not_ready")
    counts = dict(input_pack.get("counts") or {})
    if _safe_int(counts.get("strictEligibleRows")):
        flags.append("candidate_layer_blocker_decision_input_pack_strict_rows_present")
    if _safe_int(counts.get("runtimeEvidenceRows")):
        flags.append("candidate_layer_blocker_decision_input_pack_runtime_rows_present")
    if decisions_file:
        rows = list(decisions_file.get("decisions") or [])
        ids = [str(row.get("source_decision_row_id") or "") for row in rows if isinstance(row, dict)]
        if len(ids) != len(set(ids)):
            flags.append("candidate_layer_blocker_decision_file_duplicate_row_id")
    return list(dict.fromkeys(flags))


def build_candidate_layer_blocker_nonbinding_decision_recommendations(
    *,
    candidate_layer_blocker_decision_input_pack_report: str | Path,
    blocker_decisions_file: str | Path | None = None,
) -> dict[str, Any]:
    """Build nonbinding recommendations for blocker decision inputs."""

    input_pack_path = Path(str(candidate_layer_blocker_decision_input_pack_report)).expanduser()
    decisions_path = Path(str(blocker_decisions_file)).expanduser() if blocker_decisions_file else None
    input_pack = _read_json(input_pack_path)
    decisions_file = _read_json(decisions_path)
    flags = _unsafe_flags(input_pack, decisions_file)
    decision_rows = _decision_rows_by_id(decisions_file)
    decision_indexes = _decision_index_by_id(decisions_file)
    rows = [
        _row(index, input_row, decision_rows, decision_indexes)
        for index, input_row in enumerate(list(input_pack.get("decisionInputs") or []), start=1)
        if isinstance(input_row, dict)
    ]
    by_recommendation = Counter(str(row.get("recommended_decision") or "") for row in rows)
    by_bucket = Counter(str(row.get("review_bucket") or "") for row in rows)
    by_priority = Counter(str(row.get("priority") or "") for row in rows)
    by_layer = Counter(layer for row in rows for layer in list(row.get("affected_layers") or []))
    counts = {
        "recommendationRows": len(rows),
        "manualEditSuggestedRows": sum(1 for row in rows if row.get("manual_edit_required")),
        "leaveNeedsReviewRows": sum(1 for row in rows if row.get("recommended_decision") == "needs_review"),
        "decisionRowsModified": 0,
        "recommendationOnlyRows": len(rows),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(flags),
        "byRecommendation": dict(by_recommendation),
        "byBucket": dict(by_bucket),
        "byPriority": dict(by_priority),
        "byLayer": dict(by_layer),
    }
    return {
        "schema": CANDIDATE_LAYER_BLOCKER_NONBINDING_DECISION_RECOMMENDATIONS_SCHEMA_ID,
        "status": "nonbinding_recommendations_ready" if rows and not flags else "blocked",
        "generatedAt": _now(),
        "inputs": {
            "candidateLayerBlockerDecisionInputPackReport": str(input_pack_path),
            "candidateLayerBlockerDecisionInputPackSchema": str(input_pack.get("schema") or ""),
            "blockerDecisionsFile": str(decisions_path or ""),
            "candidateLayerBlockerDecisionsFile": str(decisions_path or ""),
            "blockerDecisionRows": len(list(decisions_file.get("decisions") or [])),
        },
        "counts": counts,
        "gate": {
            "recommendationsReady": bool(rows and not flags),
            "containsRecordedDecisions": False,
            "reviewCopyModified": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "recommendations_ready" if rows and not flags else "blocked",
            "unsafeUpstreamFlags": flags,
            "recommendedNextTranche": "manual_edit_candidate_layer_blocker_decisions_review_json",
        },
        "policy": {
            "reportOnly": True,
            "recommendationsOnly": True,
            "decisionFileModified": False,
            "decisionRecordCreated": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "recommendations_are_not_decisions",
            "recommendations_do_not_modify_decision_files",
            "manual_or_operator_review_is_required_before_any_non_needs_review_value_is_recorded",
            "strict_or_runtime_promotion_requires_a_separate_explicit_tranche",
        ],
        "recommendationRows": rows,
    }


def render_candidate_layer_blocker_nonbinding_decision_recommendations_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Candidate Layer Blocker Nonbinding Decision Recommendations",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Recommendation rows: `{int(counts.get('recommendationRows') or 0)}`",
        f"- Manual edit suggested rows: `{int(counts.get('manualEditSuggestedRows') or 0)}`",
        f"- Leave as needs_review rows: `{int(counts.get('leaveNeedsReviewRows') or 0)}`",
        f"- Strict/citation/runtime evidence rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "These rows are recommendations only. They do not edit the review JSON, record decisions, create strict evidence, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Rows",
        "",
    ]
    for row in list(report.get("recommendationRows") or []):
        lines.extend(
            [
                f"### `{row.get('source_decision_row_id', '')}` - `{row.get('blocker', '')}`",
                "",
                f"- Bucket: `{row.get('review_bucket', '')}`",
                f"- Current decision: `{row.get('current_decision', '')}`",
                f"- Recommended decision: `{row.get('recommended_decision', '')}`",
                f"- Manual edit required: `{str(bool(row.get('manual_edit_required'))).lower()}`",
                f"- Rationale: `{row.get('recommendation_rationale', '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_candidate_layer_blocker_nonbinding_decision_recommendations_reports(
    report: dict[str, Any], output_dir: str | Path
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "candidate-layer-blocker-nonbinding-decision-recommendations.json"
    summary_path = root / "candidate-layer-blocker-nonbinding-decision-recommendations-summary.json"
    markdown_path = root / "candidate-layer-blocker-nonbinding-decision-recommendations.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_payload = {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_candidate_layer_blocker_nonbinding_decision_recommendations_markdown(report),
        encoding="utf-8",
    )
    return {
        "recommendations": str(report_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only candidate-layer blocker decision recommendations.")
    parser.add_argument("--candidate-layer-blocker-decision-input-pack-report", required=True)
    parser.add_argument(
        "--blocker-decisions-file",
        "--candidate-layer-blocker-decisions-file",
        dest="blocker_decisions_file",
        default="",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = build_candidate_layer_blocker_nonbinding_decision_recommendations(
        candidate_layer_blocker_decision_input_pack_report=args.candidate_layer_blocker_decision_input_pack_report,
        blocker_decisions_file=args.blocker_decisions_file or None,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_candidate_layer_blocker_nonbinding_decision_recommendations_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_LAYER_BLOCKER_NONBINDING_DECISION_RECOMMENDATIONS_SCHEMA_ID",
    "build_candidate_layer_blocker_nonbinding_decision_recommendations",
    "render_candidate_layer_blocker_nonbinding_decision_recommendations_markdown",
    "write_candidate_layer_blocker_nonbinding_decision_recommendations_reports",
]
