"""Report-only preview for candidate-layer blocker decision resolution.

The preview shows how a decision record relates to the current blocker backlog.
It never removes blockers, applies decisions, creates evidence, routes parsers,
or changes answer behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CANDIDATE_LAYER_BLOCKER_RESOLUTION_PREVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-resolution-preview.v1"
)
CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-record.v1"
)
CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID = "knowledge-hub.paper.candidate-layer-blocker-backlog.v1"


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


def _unsafe_flags(decision_record: dict[str, Any], backlog: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    record_counts = dict(decision_record.get("counts") or {})
    record_gate = dict(decision_record.get("gate") or {})
    record_policy = dict(decision_record.get("policy") or {})
    backlog_counts = dict(backlog.get("counts") or {})
    backlog_gate = dict(backlog.get("gate") or {})
    backlog_policy = dict(backlog.get("policy") or {})

    if decision_record.get("schema") != CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID:
        flags.append("candidate_layer_blocker_decision_record_schema_mismatch")
    if backlog.get("schema") != CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID:
        flags.append("candidate_layer_blocker_backlog_schema_mismatch")
    if str(decision_record.get("status") or "") == "blocked":
        flags.append("candidate_layer_blocker_decision_record_blocked")
    if str(backlog.get("status") or "") == "blocked":
        flags.append("candidate_layer_blocker_backlog_blocked")

    for key in ("strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
        if _safe_int(record_counts.get(key)) > 0:
            flags.append(f"decisionRecord_{key}_nonzero")
    for key in ("strictEligibleCandidates", "citationGradeCandidates", "runtimeEvidenceCandidates"):
        if _safe_int(backlog_counts.get(key)) > 0:
            flags.append(f"backlog_{key}_nonzero")

    for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(record_gate.get(key)):
            flags.append(f"decisionRecord_{key}_true")
        if bool(backlog_gate.get(key)):
            flags.append(f"backlog_{key}_true")

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
        if bool(backlog_policy.get(key)):
            flags.append(f"backlog_{key}_true")
    return list(dict.fromkeys(flags))


def _backlog_items(backlog: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(backlog.get("backlog") or []) if isinstance(item, dict)]


def _preview_row(index: int, backlog_item: dict[str, Any], decision_counts: dict[str, Any]) -> dict[str, Any]:
    blocker = str(backlog_item.get("blocker") or "")
    needs_review = _safe_int(decision_counts.get("needsReviewRows"))
    if blocker == "candidate_layer_blocker_decision_record_pending":
        if needs_review:
            preview_status = "still_blocked"
            preview_reason = "decision_record_rows_still_need_review"
            next_action = "manual_record_candidate_layer_blocker_decisions"
        else:
            preview_status = "decision_record_complete_report_only"
            preview_reason = "decision_record_has_no_needs_review_rows_but_resolution_requires_later_explicit_tranche"
            next_action = "candidate_layer_blocker_backlog_refresh_after_explicit_approval"
    else:
        preview_status = "unchanged_open_blocker"
        preview_reason = "this_preview_only_checks_the_decision_record_pending_blocker"
        next_action = str(backlog_item.get("recommendedNextTranche") or "")

    return {
        "preview_row_id": f"candidate-layer-blocker-resolution-preview:{index:04d}",
        "source_backlog_id": str(backlog_item.get("backlog_id") or ""),
        "blocker": blocker,
        "priority": str(backlog_item.get("priority") or ""),
        "category": str(backlog_item.get("category") or ""),
        "affected_layers": list(backlog_item.get("affected_layers") or []),
        "affected_candidate_count": _safe_int(backlog_item.get("affected_candidate_count")),
        "affected_eval_question_count": _safe_int(backlog_item.get("affected_eval_question_count")),
        "preview_status": preview_status,
        "preview_reason": preview_reason,
        "recommended_next_tranche": next_action,
        "decision_scope": "candidate_layer_blocker_resolution_preview_only_no_runtime_or_strict_promotion",
        "evidence_tier": "candidate_layer_blocker_resolution_preview_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "resolution_preview_only",
            "strict_promotion_requires_later_explicit_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "resolution_preview_rows_are_not_backlog_mutations",
            "resolution_preview_rows_do_not_authorize_runtime_use",
            "resolution_preview_rows_do_not_create_strict_evidence",
        ],
    }


def _counts(rows: list[dict[str, Any]], decision_record: dict[str, Any], backlog: dict[str, Any], unsafe_flags: list[str]) -> dict[str, Any]:
    decision_counts = dict(decision_record.get("counts") or {})
    backlog_counts = dict(backlog.get("counts") or {})
    by_status = Counter(str(row.get("preview_status") or "") for row in rows)
    by_priority = Counter(str(row.get("priority") or "") for row in rows)
    by_layer = Counter(layer for row in rows for layer in list(row.get("affected_layers") or []))
    return {
        "previewRows": len(rows),
        "stillBlockedRows": by_status.get("still_blocked", 0),
        "unchangedOpenBlockerRows": by_status.get("unchanged_open_blocker", 0),
        "decisionRecordCompleteReportOnlyRows": by_status.get("decision_record_complete_report_only", 0),
        "backlogItemCount": _safe_int(backlog_counts.get("backlogItemCount")),
        "openBacklogItemCount": _safe_int(backlog_counts.get("openBacklogItemCount")),
        "decisionRecordRows": _safe_int(decision_counts.get("recordRows")),
        "decisionRecordNeedsReviewRows": _safe_int(decision_counts.get("needsReviewRows")),
        "decisionRecordManualApprovalRows": _safe_int(decision_counts.get("manualApprovalRows")),
        "decisionRecordOperatorApprovedRows": _safe_int(decision_counts.get("operatorApprovedRows")),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byPreviewStatus": dict(by_status),
        "byPriority": dict(by_priority),
        "byLayer": dict(by_layer),
    }


def build_candidate_layer_blocker_resolution_preview(
    *,
    candidate_layer_blocker_decision_record_report: str | Path,
    candidate_layer_blocker_backlog_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only preview of decision-record/backlog resolution state."""

    decision_record_path = Path(str(candidate_layer_blocker_decision_record_report)).expanduser()
    backlog_path = Path(str(candidate_layer_blocker_backlog_report)).expanduser()
    decision_record = _read_json(decision_record_path)
    backlog = _read_json(backlog_path)
    unsafe_flags = _unsafe_flags(decision_record, backlog)
    decision_counts = dict(decision_record.get("counts") or {})
    rows = [
        _preview_row(index, item, decision_counts)
        for index, item in enumerate(_backlog_items(backlog), start=1)
    ]
    counts = _counts(rows, decision_record, backlog, unsafe_flags)
    needs_review = _safe_int(decision_counts.get("needsReviewRows"))
    status = "blocked" if unsafe_flags else "resolution_preview_ready"
    return {
        "schema": CANDIDATE_LAYER_BLOCKER_RESOLUTION_PREVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "candidateLayerBlockerDecisionRecordReport": str(decision_record_path),
            "candidateLayerBlockerDecisionRecordSchema": str(decision_record.get("schema") or ""),
            "candidateLayerBlockerBacklogReport": str(backlog_path),
            "candidateLayerBlockerBacklogSchema": str(backlog.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "resolutionPreviewReady": bool(rows) and not unsafe_flags,
            "decisionRecordPending": bool(needs_review),
            "decisionRecordCompleteReportOnly": not bool(needs_review) and not unsafe_flags,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "blocked"
            if unsafe_flags
            else (
                "manual_decisions_still_required"
                if needs_review
                else "decision_record_complete_but_report_only"
            ),
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_record_candidate_layer_blocker_decisions"
            if needs_review
            else "candidate_layer_blocker_backlog_refresh_after_explicit_approval",
        },
        "policy": {
            "reportOnly": True,
            "resolutionPreviewOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "resolution_preview_rows_are_not_backlog_mutations",
            "decision_record_completion_does_not_authorize_runtime_use",
            "strict_or_runtime_promotion_requires_a_separate_explicit_tranche",
        ],
        "previewRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_candidate_layer_blocker_resolution_preview_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Candidate Layer Blocker Resolution Preview",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Preview rows: `{int(counts.get('previewRows') or 0)}`",
        f"- Decision-record `needs_review` rows: `{int(counts.get('decisionRecordNeedsReviewRows') or 0)}`",
        f"- Still blocked rows: `{int(counts.get('stillBlockedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This preview is report-only. It does not mutate the backlog, record decisions, create strict evidence, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By preview status: `{json.dumps(counts.get('byPreviewStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By priority: `{json.dumps(counts.get('byPriority') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By layer: `{json.dumps(counts.get('byLayer') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_candidate_layer_blocker_resolution_preview_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    preview_path = root / "candidate-layer-blocker-resolution-preview.json"
    summary_path = root / "candidate-layer-blocker-resolution-preview-summary.json"
    markdown_path = root / "candidate-layer-blocker-resolution-preview.md"
    preview_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_candidate_layer_blocker_resolution_preview_markdown(report), encoding="utf-8")
    return {"preview": str(preview_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only candidate-layer blocker resolution preview.")
    parser.add_argument("--candidate-layer-blocker-decision-record-report", required=True)
    parser.add_argument("--candidate-layer-blocker-backlog-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_candidate_layer_blocker_resolution_preview(
        candidate_layer_blocker_decision_record_report=args.candidate_layer_blocker_decision_record_report,
        candidate_layer_blocker_backlog_report=args.candidate_layer_blocker_backlog_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_candidate_layer_blocker_resolution_preview_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_LAYER_BLOCKER_RESOLUTION_PREVIEW_SCHEMA_ID",
    "build_candidate_layer_blocker_resolution_preview",
    "render_candidate_layer_blocker_resolution_preview_markdown",
    "write_candidate_layer_blocker_resolution_preview_reports",
]
