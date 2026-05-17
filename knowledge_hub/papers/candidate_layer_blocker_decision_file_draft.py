"""Report-only draft decision file for candidate-layer blocker rows.

The draft file is an editable starting point for a human/operator. Every row is
left at ``needs_review`` and no approval, rejection, or operator action is
recorded by this helper.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CANDIDATE_LAYER_BLOCKER_DECISION_FILE_DRAFT_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-file-draft.v1"
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


def _input_rows(input_pack: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(input_pack.get("decisionInputs") or []) if isinstance(item, dict)]


def _unsafe_flags(input_pack: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(input_pack.get("counts") or {})
    gate = dict(input_pack.get("gate") or {})
    policy = dict(input_pack.get("policy") or {})
    if input_pack.get("schema") != CANDIDATE_LAYER_BLOCKER_DECISION_INPUT_PACK_SCHEMA_ID:
        flags.append("candidate_layer_blocker_decision_input_pack_schema_mismatch")
    if str(input_pack.get("status") or "") == "blocked":
        flags.append("candidate_layer_blocker_decision_input_pack_blocked")
    for key in ("acceptedDecisionRows", "operatorApprovedRows", "strictEligibleRows", "runtimeEvidenceRows"):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"decisionInputPack_{key}_nonzero")
    for key in (
        "containsAcceptedDecisions",
        "containsOperatorApprovals",
        "strictEvidenceReady",
        "parserRoutingReady",
        "answerIntegrationReady",
        "runtimePromotionAllowed",
    ):
        if bool(gate.get(key)):
            flags.append(f"decisionInputPack_{key}_true")
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
            flags.append(f"decisionInputPack_{key}_true")
    return list(dict.fromkeys(flags))


def _draft_row(index: int, input_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "draft_row_id": f"candidate-layer-blocker-decision-file-draft:{index:04d}",
        "source_input_row_id": str(input_row.get("input_row_id") or ""),
        "source_decision_row_id": str(input_row.get("source_decision_row_id") or ""),
        "source_review_card_id": str(input_row.get("source_review_card_id") or ""),
        "source_backlog_id": str(input_row.get("source_backlog_id") or ""),
        "blocker": str(input_row.get("blocker") or ""),
        "priority": str(input_row.get("priority") or ""),
        "review_bucket": str(input_row.get("review_bucket") or ""),
        "affected_layers": list(input_row.get("affected_layers") or []),
        "allowed_decisions": list(input_row.get("allowed_decisions") or []),
        "decision": "needs_review",
        "reviewer": "",
        "notes": "",
        "draft_only": True,
        "decision_scope": "candidate_layer_blocker_decision_file_draft_only_no_runtime_or_strict_promotion",
        "evidence_tier": "candidate_layer_blocker_decision_file_draft_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "candidate_layer_blocker_decision_file_draft_only",
            "decision_not_recorded",
            "strict_promotion_requires_later_explicit_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "draft_rows_are_not_decisions",
            "draft_rows_default_to_needs_review",
            "draft_rows_do_not_authorize_runtime_use",
        ],
    }


def _decision_file_from_drafts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "draftOnly": True,
        "instructions": [
            "Edit a copy of this file before using it as a decision file.",
            "Keep decision=needs_review unless a human/operator has made an explicit decision.",
            "Non-needs_review decisions require reviewer and notes.",
            "This draft does not authorize strict evidence, parser routing, DB/index writes, or runtime answer use.",
        ],
        "decisions": [
            {
                "source_decision_row_id": str(row.get("source_decision_row_id") or ""),
                "decision": "needs_review",
                "reviewer": "",
                "notes": "",
                "allowed_decisions": list(row.get("allowed_decisions") or []),
                "review_bucket": str(row.get("review_bucket") or ""),
                "blocker": str(row.get("blocker") or ""),
            }
            for row in rows
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    by_bucket = Counter(str(row.get("review_bucket") or "") for row in rows)
    by_priority = Counter(str(row.get("priority") or "") for row in rows)
    by_layer = Counter(layer for row in rows for layer in list(row.get("affected_layers") or []))
    return {
        "draftRows": len(rows),
        "needsReviewRows": len(rows),
        "nonNeedsReviewRows": 0,
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


def build_candidate_layer_blocker_decision_file_draft(
    *,
    candidate_layer_blocker_decision_input_pack_report: str | Path,
) -> dict[str, Any]:
    """Build a needs-review-only draft decision file report."""

    input_pack_path = Path(str(candidate_layer_blocker_decision_input_pack_report)).expanduser()
    input_pack = _read_json(input_pack_path)
    unsafe_flags = _unsafe_flags(input_pack)
    rows = [_draft_row(index, row) for index, row in enumerate(_input_rows(input_pack), start=1)]
    counts = _counts(rows, unsafe_flags)
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif rows:
        status = "decision_file_draft_ready"
        decision = "needs_review_draft_ready_for_manual_edit"
    else:
        status = "no_draft_rows"
        decision = "no_pending_decision_inputs"
    return {
        "schema": CANDIDATE_LAYER_BLOCKER_DECISION_FILE_DRAFT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "candidateLayerBlockerDecisionInputPackReport": str(input_pack_path),
            "candidateLayerBlockerDecisionInputPackSchema": str(input_pack.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "decisionFileDraftReady": bool(rows) and not unsafe_flags,
            "containsOnlyNeedsReviewDefaults": True,
            "containsAcceptedDecisions": False,
            "containsOperatorApprovals": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_edit_candidate_layer_blocker_decision_file_draft"
            if rows
            else "candidate_layer_blocker_decision_input_pack_refresh",
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
            "draft_rows_are_not_recorded_decisions",
            "draft_decision_file_defaults_every_row_to_needs_review",
            "operator_approval_drafts_do_not_execute_diagnostic_actions",
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


def render_candidate_layer_blocker_decision_file_draft_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Candidate Layer Blocker Decision File Draft",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Draft rows: `{int(counts.get('draftRows') or 0)}`",
        f"- `needs_review` rows: `{int(counts.get('needsReviewRows') or 0)}`",
        f"- Accepted decisions: `{int(counts.get('acceptedDecisionRows') or 0)}`",
        f"- Operator approvals: `{int(counts.get('operatorApprovedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This draft is an editable starting point only. It does not record decisions, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By bucket: `{json.dumps(counts.get('byBucket') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By priority: `{json.dumps(counts.get('byPriority') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By layer: `{json.dumps(counts.get('byLayer') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_candidate_layer_blocker_decision_file_draft_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    draft_report_path = root / "candidate-layer-blocker-decision-file-draft.json"
    decision_file_path = root / "candidate-layer-blocker-decisions.draft.json"
    summary_path = root / "candidate-layer-blocker-decision-file-draft-summary.json"
    markdown_path = root / "candidate-layer-blocker-decision-file-draft.md"
    draft_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    decision_file_path.write_text(
        json.dumps(report.get("decisionFileDraft") or {}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_candidate_layer_blocker_decision_file_draft_markdown(report), encoding="utf-8")
    return {
        "draftReport": str(draft_report_path),
        "decisionFileDraft": str(decision_file_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a needs-review-only candidate-layer blocker decision file draft.")
    parser.add_argument("--candidate-layer-blocker-decision-input-pack-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_candidate_layer_blocker_decision_file_draft(
        candidate_layer_blocker_decision_input_pack_report=args.candidate_layer_blocker_decision_input_pack_report
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_candidate_layer_blocker_decision_file_draft_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_LAYER_BLOCKER_DECISION_FILE_DRAFT_SCHEMA_ID",
    "build_candidate_layer_blocker_decision_file_draft",
    "render_candidate_layer_blocker_decision_file_draft_markdown",
    "write_candidate_layer_blocker_decision_file_draft_reports",
]
