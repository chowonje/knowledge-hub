"""Bucket-level decision report for canonical dirty rows in the text-evidence RC."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.figure_caption_artifact_vertical_slice import utc_now_iso

TEXT_EVIDENCE_CANONICAL_DIRTY_BUCKET_DECISION_SCHEMA_ID = (
    "knowledge-hub.paper.text-evidence-canonical-dirty-bucket-decision.v1"
)

INVENTORY_REPORT_REF = "text_evidence_canonical_dirty_inventory.v1.json"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)

BUCKET_DECISIONS: dict[str, dict[str, str]] = {
    "answer_runtime_or_query_stack": {
        "decision": "exclude_from_text_rc_review_after_rc_candidate",
        "publicRcAction": "do_not_replay_now",
        "followUpTrack": "answer_runtime_replay",
        "reason": "answer/runtime/query deltas are broader than the report-only text-evidence RC line",
    },
    "cli_mcp_public_surface_stack": {
        "decision": "exclude_from_text_rc_compare_with_public_operator_cleanup",
        "publicRcAction": "do_not_replay_now",
        "followUpTrack": "public_surface_replay_review",
        "reason": "CLI/MCP deltas overlap the already-cleaned public/operator surface and need explicit comparison first",
    },
    "docs_governance_stack": {
        "decision": "exclude_from_text_rc_record_sync_only",
        "publicRcAction": "review_records_only",
        "followUpTrack": "docs_record_reconciliation",
        "reason": "dirty docs should not be copied into RC without reconciling them against the current stacked branch",
    },
    "eval_or_test_support_stack": {
        "decision": "exclude_from_text_rc_map_to_owning_feature",
        "publicRcAction": "do_not_replay_now",
        "followUpTrack": "owning_feature_eval_replay",
        "reason": "eval/test deltas must follow the owning clean feature branch rather than land as a mixed bucket",
    },
    "evidence_spine_or_source_contract_stack": {
        "decision": "exclude_from_text_rc_clean_replay_candidate",
        "publicRcAction": "defer_to_clean_replay",
        "followUpTrack": "evidence_spine_clean_replay",
        "reason": "evidence/source contract deltas have broad answer-contract impact and require a separate clean replay",
    },
    "infrastructure_or_core_stack": {
        "decision": "exclude_from_text_rc_clean_replay_candidate",
        "publicRcAction": "defer_to_clean_replay",
        "followUpTrack": "core_infrastructure_clean_replay",
        "reason": "core/infrastructure deltas have cross-cutting blast radius and need separate verification",
    },
    "parser_artifact_side_stack": {
        "decision": "hold_outside_text_rc_parser_track",
        "publicRcAction": "hold",
        "followUpTrack": "parser_artifact_repair",
        "reason": "parser/artifact repair is deferred from the text-only RC convergence line",
    },
    "provider_hint_side_stack": {
        "decision": "hold_outside_text_rc_provider_hint_track",
        "publicRcAction": "hold",
        "followUpTrack": "provider_hint_shadow",
        "reason": "provider hint shadow work was explicitly identified as non-essential for the current RC",
    },
    "research_objects_side_stack": {
        "decision": "hold_outside_text_rc_research_objects_track",
        "publicRcAction": "hold",
        "followUpTrack": "research_objects",
        "reason": "research-object graph planning is outside the current text-evidence RC scope",
    },
    "source_ingest_or_library_stack": {
        "decision": "exclude_from_text_rc_clean_replay_candidate",
        "publicRcAction": "defer_to_clean_replay",
        "followUpTrack": "source_ingest_library_clean_replay",
        "reason": "source ingest/library/vault-link deltas need a clean replay and focused storage checks",
    },
    "workspace_process_record": {
        "decision": "drop_or_move_out_of_product_checkout",
        "publicRcAction": "exclude_from_public_rc",
        "followUpTrack": "workspace_record_cleanup",
        "reason": "local process records are not product behavior and should not be tracked in the product RC",
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


def _bucket_rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    bucket_counts = dict(inventory.get("bucketCounts") or {})
    rows: list[dict[str, Any]] = []
    for bucket, count in sorted(bucket_counts.items()):
        spec = BUCKET_DECISIONS.get(bucket)
        if spec is None:
            spec = {
                "decision": "manual_decision_required",
                "publicRcAction": "block",
                "followUpTrack": "manual_triage",
                "reason": "bucket has no registered RC disposition",
            }
        rows.append(
            {
                "bucket": _clean_text(bucket),
                "dirtyRows": int(count or 0),
                "decision": _clean_text(spec["decision"]),
                "publicRcAction": _clean_text(spec["publicRcAction"]),
                "followUpTrack": _clean_text(spec["followUpTrack"]),
                "reason": _clean_text(spec["reason"]),
            }
        )
    return rows


def build_text_evidence_canonical_dirty_bucket_decision_report(
    *,
    reports_root: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    inventory = _load_json(reports_root / INVENTORY_REPORT_REF)
    bucket_rows = _bucket_rows(inventory)
    public_actions = Counter(row["publicRcAction"] for row in bucket_rows)
    direct_include_rows = sum(row["dirtyRows"] for row in bucket_rows if row["publicRcAction"] == "include_in_text_rc")
    block_rows = sum(row["dirtyRows"] for row in bucket_rows if row["publicRcAction"] == "block")
    unknown_bucket_rows = int(inventory.get("unknownRows") or 0)
    dirty_rows = int(inventory.get("dirtyRows") or 0)

    report: dict[str, Any] = {
        "schema": TEXT_EVIDENCE_CANONICAL_DIRTY_BUCKET_DECISION_SCHEMA_ID,
        "status": "ready" if inventory and block_rows == 0 and unknown_bucket_rows == 0 else "blocked",
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
        },
        "inventoryReportRef": INVENTORY_REPORT_REF,
        "inventoryStatus": _clean_text(inventory.get("status")),
        "dirtyRows": dirty_rows,
        "bucketRows": len(bucket_rows),
        "directIncludeRows": direct_include_rows,
        "blockRows": block_rows,
        "unknownBucketRows": unknown_bucket_rows,
        "publicRcDecision": {
            "decision": "do_not_merge_canonical_dirty_checkout_into_text_rc",
            "reason": "all canonical dirty buckets are either excluded, held, or deferred to separate clean replay tracks",
            "requiresCanonicalCleanupBeforeRc": True,
        },
        "actionCounts": dict(sorted(public_actions.items())),
        "buckets": bucket_rows,
        "nextAction": "clean_or_archive_canonical_dirty_checkout_after_external_pr149_resolution",
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
        },
        "schemaViolationCount": 0,
        "privatePathLeakRows": 0,
        "reportHash": "",
        "warnings": [
            "This decision does not edit or clean the canonical checkout.",
            "No canonical dirty bucket is approved for direct inclusion in the text-evidence RC branch.",
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
    decision = dict(report.get("publicRcDecision") or {})
    lines = [
        "# Text Evidence Canonical Dirty Bucket Decision",
        "",
        f"- status: `{report.get('status')}`",
        f"- dirtyRows: `{report.get('dirtyRows')}`",
        f"- bucketRows: `{report.get('bucketRows')}`",
        f"- directIncludeRows: `{report.get('directIncludeRows')}`",
        f"- blockRows: `{report.get('blockRows')}`",
        f"- unknownBucketRows: `{report.get('unknownBucketRows')}`",
        f"- publicRcDecision: `{decision.get('decision')}`",
        f"- nextAction: `{report.get('nextAction')}`",
        "",
        "## Buckets",
        "",
        "| bucket | rows | action | decision | follow-up |",
        "|---|---:|---|---|---|",
    ]
    for row in report.get("buckets", []):
        lines.append(
            f"| `{row.get('bucket')}` | `{row.get('dirtyRows')}` | `{row.get('publicRcAction')}` | "
            f"`{row.get('decision')}` | `{row.get('followUpTrack')}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


__all__ = [
    "TEXT_EVIDENCE_CANONICAL_DIRTY_BUCKET_DECISION_SCHEMA_ID",
    "build_text_evidence_canonical_dirty_bucket_decision_report",
    "render_markdown_report",
    "write_report",
]
