"""Report-only review pack for remaining candidate-layer manual blockers.

This helper narrows a blocker decision record to the rows that still require
human review. It does not record decisions, close blockers, create evidence,
run parsers, mutate stores, or authorize runtime promotion.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CANDIDATE_LAYER_REMAINING_MANUAL_BLOCKER_REVIEW_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-remaining-manual-blocker-review-pack.v1"
)
CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-record.v1"
)
CANDIDATE_LAYER_BLOCKER_RESOLUTION_PREVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-resolution-preview.v1"
)

_ALLOWED_DECISIONS_BY_BUCKET = {
    "manual_decision_required": [
        "needs_review",
        "record_manual_approval_in_separate_decision_file",
        "record_manual_rejection_in_separate_decision_file",
        "keep_blocked",
    ],
    "operator_approval_required": [
        "needs_review",
        "approve_diagnostic_operator_action_in_separate_decision_file",
        "decline_diagnostic_operator_action_keep_blocked",
        "keep_blocked",
    ],
    "technical_feasibility_blocked": [
        "needs_review",
        "accept_technical_blocker_as_open",
        "defer_technical_followup",
        "close_as_not_needed",
    ],
    "policy_review_only": [
        "needs_review",
        "accept_policy_blocker_as_guardrail",
        "defer_policy_review",
    ],
}

_MANUAL_REVIEW_REASON_BY_BLOCKER = {
    "sectionspan_pdf_offsets_require_human_review_before_strict_promotion": (
        "SectionSpan PDF offsets are a later-promotion prerequisite and must be reviewed by a human before any strict evidence design."
    ),
    "sectionspan_selected_review_manual_edit_required": (
        "Selected SectionSpan rows still need explicit reviewer, decision, and notes in their own decision file."
    ),
    "equation_quote_decision_manual_edit_required": (
        "EquationQuote rows require human decision-file edits because equation alignment and original-PDF offsets are still incomplete."
    ),
    "candidate_layer_blocker_decision_record_pending": (
        "The blocker decision record is still pending until all decision rows are explicitly recorded or kept blocked."
    ),
}

_EVIDENCE_TO_INSPECT_BY_BLOCKER = {
    "sectionspan_pdf_offsets_require_human_review_before_strict_promotion": [
        "sectionspan-pdf-offset-human-review-gate",
        "sectionspan-pdf-offset-selected-review-evidence-pack",
        "sectionspan-pdf-offset-selected-review-manual-sheet",
    ],
    "sectionspan_selected_review_manual_edit_required": [
        "sectionspan-pdf-offset-selected-review-decision-file-draft",
        "sectionspan-pdf-offset-selected-review-decision-file-validation",
        "sectionspan-pdf-offset-selected-review-next-action-brief",
    ],
    "equation_quote_decision_manual_edit_required": [
        "equation-quote-decision-file-draft",
        "equation-quote-decision-file-validation",
        "equation-quote-decision-next-action-brief",
    ],
    "candidate_layer_blocker_decision_record_pending": [
        "candidate-layer-blocker-decisions.applied-2026-05-18.json",
        "candidate-layer-blocker-decision-record",
        "candidate-layer-blocker-resolution-preview",
    ],
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


def _record_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(record.get("decisionRecords") or []) if isinstance(item, dict)]


def _preview_rows(preview: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(preview.get("previewRows") or []) if isinstance(item, dict)]


def _unsafe_flags(decision_record: dict[str, Any], resolution_preview: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    record_counts = dict(decision_record.get("counts") or {})
    record_gate = dict(decision_record.get("gate") or {})
    record_policy = dict(decision_record.get("policy") or {})
    if decision_record.get("schema") != CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID:
        flags.append("candidate_layer_blocker_decision_record_schema_mismatch")
    if str(decision_record.get("status") or "") == "blocked":
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

    if resolution_preview:
        preview_counts = dict(resolution_preview.get("counts") or {})
        preview_gate = dict(resolution_preview.get("gate") or {})
        preview_policy = dict(resolution_preview.get("policy") or {})
        if resolution_preview.get("schema") != CANDIDATE_LAYER_BLOCKER_RESOLUTION_PREVIEW_SCHEMA_ID:
            flags.append("candidate_layer_blocker_resolution_preview_schema_mismatch")
        if str(resolution_preview.get("status") or "") == "blocked":
            flags.append("candidate_layer_blocker_resolution_preview_blocked")
        for key in ("strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
            if _safe_int(preview_counts.get(key)) > 0:
                flags.append(f"resolutionPreview_{key}_nonzero")
        for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
            if bool(preview_gate.get(key)):
                flags.append(f"resolutionPreview_{key}_true")
        for key in (
            "strictEvidenceCreated",
            "runtimePromotionAllowed",
            "parserRoutingChanged",
            "canonicalParsedArtifactsWritten",
            "databaseMutation",
            "reindexOrReembed",
            "answerIntegrationChanged",
        ):
            if bool(preview_policy.get(key)):
                flags.append(f"resolutionPreview_{key}_true")
    return list(dict.fromkeys(flags))


def _remaining_manual_rows(decision_record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in _record_rows(decision_record)
        if str(row.get("recorded_decision") or "") == "needs_review"
    ]


def _review_row(index: int, record_row: dict[str, Any], preview_by_blocker: dict[str, dict[str, Any]]) -> dict[str, Any]:
    blocker = str(record_row.get("blocker") or "")
    bucket = str(record_row.get("review_bucket") or "")
    preview_row = preview_by_blocker.get(blocker) or {}
    allowed_decisions = _ALLOWED_DECISIONS_BY_BUCKET.get(bucket, ["needs_review"])
    return {
        "review_row_id": f"candidate-layer-remaining-manual-blocker-review:{index:04d}",
        "source_decision_record_row_id": str(record_row.get("record_row_id") or ""),
        "source_decision_row_id": str(record_row.get("source_decision_row_id") or ""),
        "source_review_card_id": str(record_row.get("source_review_card_id") or ""),
        "source_backlog_id": str(record_row.get("source_backlog_id") or preview_row.get("source_backlog_id") or ""),
        "blocker": blocker,
        "priority": str(record_row.get("priority") or ""),
        "review_bucket": bucket,
        "affected_layers": list(record_row.get("affected_layers") or []),
        "affected_candidate_count": _safe_int(record_row.get("affected_candidate_count")),
        "affected_eval_question_count": _safe_int(record_row.get("affected_eval_question_count")),
        "recorded_decision": str(record_row.get("recorded_decision") or ""),
        "raw_decision": str(record_row.get("raw_decision") or ""),
        "reviewer": str(record_row.get("reviewer") or ""),
        "notes": str(record_row.get("notes") or ""),
        "resolution_preview_status": str(preview_row.get("preview_status") or ""),
        "resolution_preview_reason": str(preview_row.get("preview_reason") or ""),
        "why_manual_review_required": _MANUAL_REVIEW_REASON_BY_BLOCKER.get(
            blocker,
            "This blocker remains needs_review and cannot be auto-closed by a report-only helper.",
        ),
        "evidence_to_inspect": list(_EVIDENCE_TO_INSPECT_BY_BLOCKER.get(blocker, [])),
        "recommended_next_tranche": str(record_row.get("recommended_next_tranche") or ""),
        "recommended_review_action": str(record_row.get("recommended_review_action") or ""),
        "allowed_decisions": allowed_decisions,
        "safe_default_decision": "needs_review",
        "decision_input_hint": {
            "source_decision_row_id": str(record_row.get("source_decision_row_id") or ""),
            "decision": "needs_review",
            "reviewer": "",
            "notes": "",
        },
        "decision_scope": "candidate_layer_remaining_manual_blocker_review_pack_only_no_runtime_or_strict_promotion",
        "evidence_tier": "candidate_layer_remaining_manual_blocker_review_pack_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "remaining_manual_blocker_requires_human_decision",
            "strict_promotion_requires_later_explicit_tranche",
            "runtime_promotion_disabled_for_tranche",
            "candidate_layer_outputs_are_report_only",
        ],
        "non_strict_reason": [
            "remaining_manual_blocker_review_rows_are_not_decisions",
            "remaining_manual_blocker_review_rows_do_not_authorize_runtime_use",
            "remaining_manual_blocker_review_rows_do_not_create_strict_evidence",
        ],
    }


def _counts(rows: list[dict[str, Any]], decision_record: dict[str, Any], resolution_preview: dict[str, Any], unsafe_flags: list[str]) -> dict[str, Any]:
    by_bucket = Counter(str(row.get("review_bucket") or "") for row in rows)
    by_priority = Counter(str(row.get("priority") or "") for row in rows)
    by_layer = Counter(layer for row in rows for layer in list(row.get("affected_layers") or []))
    by_blocker = Counter(str(row.get("blocker") or "") for row in rows)
    record_counts = dict(decision_record.get("counts") or {})
    preview_counts = dict(resolution_preview.get("counts") or {})
    return {
        "reviewRows": len(rows),
        "remainingNeedsReviewRows": len(rows),
        "decisionRecordRows": _safe_int(record_counts.get("recordRows")),
        "decisionRecordNeedsReviewRows": _safe_int(record_counts.get("needsReviewRows")),
        "decisionRecordTechnicalAcceptedOpenRows": _safe_int(record_counts.get("technicalAcceptedOpenRows")),
        "decisionRecordPolicyAcceptedGuardrailRows": _safe_int(record_counts.get("policyAcceptedGuardrailRows")),
        "resolutionPreviewStillBlockedRows": _safe_int(preview_counts.get("stillBlockedRows")),
        "manualDecisionRows": by_bucket.get("manual_decision_required", 0),
        "operatorApprovalRows": by_bucket.get("operator_approval_required", 0),
        "technicalDecisionRows": by_bucket.get("technical_feasibility_blocked", 0),
        "policyDecisionRows": by_bucket.get("policy_review_only", 0),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byBucket": dict(by_bucket),
        "byPriority": dict(by_priority),
        "byLayer": dict(by_layer),
        "byBlocker": dict(by_blocker),
    }


def build_candidate_layer_remaining_manual_blocker_review_pack(
    *,
    candidate_layer_blocker_decision_record_report: str | Path,
    candidate_layer_blocker_resolution_preview_report: str | Path | None = None,
) -> dict[str, Any]:
    """Build a report-only pack for remaining manual blocker rows."""

    record_path = Path(str(candidate_layer_blocker_decision_record_report)).expanduser()
    preview_path = (
        Path(str(candidate_layer_blocker_resolution_preview_report)).expanduser()
        if candidate_layer_blocker_resolution_preview_report
        else None
    )
    decision_record = _read_json(record_path)
    resolution_preview = _read_json(preview_path) if preview_path else {}
    unsafe_flags = _unsafe_flags(decision_record, resolution_preview)
    preview_by_blocker = {
        str(row.get("blocker") or ""): row
        for row in _preview_rows(resolution_preview)
    }
    rows = [
        _review_row(index, row, preview_by_blocker)
        for index, row in enumerate(_remaining_manual_rows(decision_record), start=1)
    ]
    counts = _counts(rows, decision_record, resolution_preview, unsafe_flags)
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif rows:
        status = "manual_review_required"
        decision = "remaining_manual_blockers_require_human_decision"
    else:
        status = "no_remaining_manual_blockers"
        decision = "no_remaining_manual_blockers_report_only"
    return {
        "schema": CANDIDATE_LAYER_REMAINING_MANUAL_BLOCKER_REVIEW_PACK_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "candidateLayerBlockerDecisionRecordReport": str(record_path),
            "candidateLayerBlockerDecisionRecordSchema": str(decision_record.get("schema") or ""),
            "candidateLayerBlockerResolutionPreviewReport": str(preview_path or ""),
            "candidateLayerBlockerResolutionPreviewSchema": str(resolution_preview.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "reviewPackReady": bool(rows) and not unsafe_flags,
            "manualReviewRequired": bool(rows),
            "allDecisionRowsComplete": not bool(rows) and not unsafe_flags,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_decide_remaining_candidate_layer_blockers"
            if rows
            else "candidate_layer_blocker_resolution_review_requires_explicit_approval",
        },
        "policy": {
            "reportOnly": True,
            "remainingManualBlockerReviewPackOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "remaining_manual_blocker_review_rows_are_not_decisions",
            "this_pack_does_not_edit_the_decision_file",
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


def render_candidate_layer_remaining_manual_blocker_review_pack_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Candidate Layer Remaining Manual Blocker Review Pack",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Remaining manual review rows: `{int(counts.get('remainingNeedsReviewRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        f"- Runtime evidence rows: `{int(counts.get('runtimeEvidenceRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This pack is report-only. It does not edit decision files, record decisions, create strict evidence, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By bucket: `{json.dumps(counts.get('byBucket') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By layer: `{json.dumps(counts.get('byLayer') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By blocker: `{json.dumps(counts.get('byBlocker') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Remaining Rows",
        "",
    ]
    for row in list(report.get("reviewRows") or []):
        lines.extend(
            [
                f"### {row.get('source_decision_row_id', '')}",
                "",
                f"- Blocker: `{row.get('blocker', '')}`",
                f"- Priority: `{row.get('priority', '')}`",
                f"- Layers: `{json.dumps(row.get('affected_layers') or [], ensure_ascii=False)}`",
                f"- Why manual: {row.get('why_manual_review_required', '')}",
                f"- Allowed decisions: `{json.dumps(row.get('allowed_decisions') or [], ensure_ascii=False)}`",
                f"- Safe default: `{row.get('safe_default_decision', '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_candidate_layer_remaining_manual_blocker_review_pack_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    pack_path = root / "candidate-layer-remaining-manual-blocker-review-pack.json"
    summary_path = root / "candidate-layer-remaining-manual-blocker-review-pack-summary.json"
    markdown_path = root / "candidate-layer-remaining-manual-blocker-review-pack.md"
    pack_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_candidate_layer_remaining_manual_blocker_review_pack_markdown(report), encoding="utf-8")
    return {"reviewPack": str(pack_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only remaining manual blocker review pack.")
    parser.add_argument("--candidate-layer-blocker-decision-record-report", required=True)
    parser.add_argument("--candidate-layer-blocker-resolution-preview-report", default="")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_candidate_layer_remaining_manual_blocker_review_pack(
        candidate_layer_blocker_decision_record_report=args.candidate_layer_blocker_decision_record_report,
        candidate_layer_blocker_resolution_preview_report=args.candidate_layer_blocker_resolution_preview_report or None,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_candidate_layer_remaining_manual_blocker_review_pack_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_LAYER_REMAINING_MANUAL_BLOCKER_REVIEW_PACK_SCHEMA_ID",
    "build_candidate_layer_remaining_manual_blocker_review_pack",
    "render_candidate_layer_remaining_manual_blocker_review_pack_markdown",
    "write_candidate_layer_remaining_manual_blocker_review_pack_reports",
]
