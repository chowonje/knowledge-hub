"""Cleanup/archive plan for canonical dirty rows in the text-evidence RC."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.figure_caption_artifact_vertical_slice import utc_now_iso

TEXT_EVIDENCE_CANONICAL_DIRTY_CLEANUP_PLAN_SCHEMA_ID = (
    "knowledge-hub.paper.text-evidence-canonical-dirty-cleanup-plan.v1"
)

BUCKET_DECISION_REPORT_REF = "text_evidence_canonical_dirty_bucket_decision.v1.json"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)

CLEANUP_ACTIONS: dict[str, dict[str, str]] = {
    "answer_runtime_or_query_stack": {
        "cleanupAction": "archive_or_discard_after_text_rc_branch_review",
        "sequenceGroup": "post_rc_candidate_review",
        "approvalRequired": "yes",
        "reason": "answer runtime changes are excluded from text RC and should not remain as canonical dirty state",
    },
    "cli_mcp_public_surface_stack": {
        "cleanupAction": "compare_then_discard_or_clean_replay",
        "sequenceGroup": "public_surface_review",
        "approvalRequired": "yes",
        "reason": "CLI/MCP changes need comparison against the public operator cleanup before any replay",
    },
    "docs_governance_stack": {
        "cleanupAction": "selective_record_reconciliation_then_discard_remainder",
        "sequenceGroup": "record_reconciliation",
        "approvalRequired": "yes",
        "reason": "docs changes may contain useful records, but dirty docs cannot be copied wholesale into the RC",
    },
    "eval_or_test_support_stack": {
        "cleanupAction": "map_to_owning_feature_then_archive_or_discard",
        "sequenceGroup": "owning_feature_review",
        "approvalRequired": "yes",
        "reason": "eval/test files should move only with the feature stack they verify",
    },
    "evidence_spine_or_source_contract_stack": {
        "cleanupAction": "preserve_for_later_clean_replay_then_clear_canonical_dirty",
        "sequenceGroup": "clean_replay_candidate",
        "approvalRequired": "yes",
        "reason": "evidence/source contract changes are valuable but too broad for direct text RC inclusion",
    },
    "infrastructure_or_core_stack": {
        "cleanupAction": "preserve_for_later_clean_replay_then_clear_canonical_dirty",
        "sequenceGroup": "clean_replay_candidate",
        "approvalRequired": "yes",
        "reason": "core infrastructure changes need isolated replay and broad verification",
    },
    "parser_artifact_side_stack": {
        "cleanupAction": "hold_for_parser_branch_or_archive",
        "sequenceGroup": "held_side_track",
        "approvalRequired": "yes",
        "reason": "parser/artifact work is deferred from the text RC line",
    },
    "provider_hint_side_stack": {
        "cleanupAction": "hold_or_discard_shadow_experiment",
        "sequenceGroup": "held_side_track",
        "approvalRequired": "yes",
        "reason": "provider hint shadow work is not needed for the current RC",
    },
    "research_objects_side_stack": {
        "cleanupAction": "archive_research_objects_side_stack",
        "sequenceGroup": "held_side_track",
        "approvalRequired": "yes",
        "reason": "research-object graph planning is outside the current text RC",
    },
    "source_ingest_or_library_stack": {
        "cleanupAction": "preserve_for_later_clean_replay_then_clear_canonical_dirty",
        "sequenceGroup": "clean_replay_candidate",
        "approvalRequired": "yes",
        "reason": "source ingest/library changes need storage-focused replay checks",
    },
    "workspace_process_record": {
        "cleanupAction": "move_to_workspace_records_or_delete_from_product_checkout",
        "sequenceGroup": "public_rc_exclusion",
        "approvalRequired": "yes",
        "reason": "local process records are not product behavior and should not ship in the public RC",
    },
}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _private_path_leak_count(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return 1 if PRIVATE_PATH_RE.search(encoded) else 0


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _plan_rows(bucket_decision: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket_row in bucket_decision.get("buckets", []):
        if not isinstance(bucket_row, dict):
            continue
        bucket = _clean_text(bucket_row.get("bucket"))
        spec = CLEANUP_ACTIONS.get(bucket)
        if spec is None:
            spec = {
                "cleanupAction": "manual_cleanup_plan_required",
                "sequenceGroup": "blocked",
                "approvalRequired": "yes",
                "reason": "bucket has no registered cleanup action",
            }
        rows.append(
            {
                "bucket": bucket,
                "dirtyRows": int(bucket_row.get("dirtyRows") or 0),
                "priorDecision": _clean_text(bucket_row.get("decision")),
                "publicRcAction": _clean_text(bucket_row.get("publicRcAction")),
                "cleanupAction": _clean_text(spec["cleanupAction"]),
                "sequenceGroup": _clean_text(spec["sequenceGroup"]),
                "approvalRequired": _clean_text(spec["approvalRequired"]),
                "reason": _clean_text(spec["reason"]),
            }
        )
    return rows


def build_text_evidence_canonical_dirty_cleanup_plan_report(
    *,
    reports_root: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    bucket_decision = _load_json(reports_root / BUCKET_DECISION_REPORT_REF)
    rows = _plan_rows(bucket_decision)
    sequence_counts = dict(sorted(Counter(row["sequenceGroup"] for row in rows).items()))
    action_counts = dict(sorted(Counter(row["cleanupAction"] for row in rows).items()))
    manual_plan_rows = sum(1 for row in rows if row["cleanupAction"] == "manual_cleanup_plan_required")
    dirty_rows = sum(row["dirtyRows"] for row in rows)

    report: dict[str, Any] = {
        "schema": TEXT_EVIDENCE_CANONICAL_DIRTY_CLEANUP_PLAN_SCHEMA_ID,
        "status": "ready" if bucket_decision and manual_plan_rows == 0 else "blocked",
        "generatedAt": generated_at or utc_now_iso(),
        "scope": {
            "writes": "report_only",
            "canonicalCheckoutEdited": False,
            "fileContentReadRows": 0,
            "vaultScanRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "externalDownloadRows": 0,
            "mergeRows": 0,
            "cherryPickRows": 0,
            "worktreeDeletionRows": 0,
            "destructiveCleanupRows": 0,
        },
        "bucketDecisionReportRef": BUCKET_DECISION_REPORT_REF,
        "bucketDecisionStatus": _clean_text(bucket_decision.get("status")),
        "dirtyRows": dirty_rows,
        "planRows": len(rows),
        "manualPlanRows": manual_plan_rows,
        "executionPolicy": {
            "requiresExplicitApproval": True,
            "requiresPr149ResolvedFirst": True,
            "requiresSnapshotBeforeCleanup": True,
            "allowsDirectTextRcMerge": False,
        },
        "sequenceCounts": sequence_counts,
        "actionCounts": action_counts,
        "rows": rows,
        "nextAction": "request_approval_to_close_pr149_then_snapshot_and_clean_canonical_checkout",
        "mutationCounters": {
            "canonicalParsedArtifactWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "mergeRows": 0,
            "cherryPickRows": 0,
            "worktreeDeletionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
            "strictEvidencePromotionRows": 0,
            "destructiveCleanupRows": 0,
        },
        "schemaViolationCount": 0,
        "privatePathLeakRows": 0,
        "reportHash": "",
        "warnings": [
            "This cleanup plan performs no cleanup and no destructive operation.",
            "Physical cleanup requires explicit approval and a snapshot/checkpoint first.",
        ],
        "schemaErrors": [],
    }
    report["privatePathLeakRows"] = _private_path_leak_count(report)
    if report["privatePathLeakRows"]:
        report["status"] = "blocked"
    payload_for_hash = dict(report)
    payload_for_hash["reportHash"] = ""
    report["reportHash"] = _sha256_json(payload_for_hash)
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    policy = dict(report.get("executionPolicy") or {})
    lines = [
        "# Text Evidence Canonical Dirty Cleanup Plan",
        "",
        f"- status: `{report.get('status')}`",
        f"- dirtyRows: `{report.get('dirtyRows')}`",
        f"- planRows: `{report.get('planRows')}`",
        f"- manualPlanRows: `{report.get('manualPlanRows')}`",
        f"- requiresExplicitApproval: `{policy.get('requiresExplicitApproval')}`",
        f"- requiresPr149ResolvedFirst: `{policy.get('requiresPr149ResolvedFirst')}`",
        f"- requiresSnapshotBeforeCleanup: `{policy.get('requiresSnapshotBeforeCleanup')}`",
        f"- allowsDirectTextRcMerge: `{policy.get('allowsDirectTextRcMerge')}`",
        f"- nextAction: `{report.get('nextAction')}`",
        "",
        "## Sequence Counts",
        "",
        "| sequence | rows |",
        "|---|---:|",
    ]
    for name, count in dict(report.get("sequenceCounts") or {}).items():
        lines.append(f"| `{name}` | `{count}` |")
    lines.extend(
        [
            "",
            "## Cleanup Rows",
            "",
            "| bucket | dirty rows | sequence | cleanup action |",
            "|---|---:|---|---|",
        ]
    )
    for row in report.get("rows", []):
        lines.append(
            f"| `{row.get('bucket')}` | `{row.get('dirtyRows')}` | "
            f"`{row.get('sequenceGroup')}` | `{row.get('cleanupAction')}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


__all__ = [
    "TEXT_EVIDENCE_CANONICAL_DIRTY_CLEANUP_PLAN_SCHEMA_ID",
    "build_text_evidence_canonical_dirty_cleanup_plan_report",
    "render_markdown_report",
    "write_report",
]
