"""Approval packet for the external RC blocker actions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence

from knowledge_hub.papers.figure_caption_artifact_vertical_slice import utc_now_iso

TEXT_EVIDENCE_RC_EXTERNAL_ACTION_APPROVAL_PACKET_SCHEMA_ID = (
    "knowledge-hub.paper.text-evidence-rc-external-action-approval-packet.v1"
)

PR149_DISPOSITION_REPORT_REF = "text_evidence_pr149_disposition.v1.json"
CANONICAL_DIRTY_CLEANUP_PLAN_REPORT_REF = "text_evidence_canonical_dirty_cleanup_plan.v1.json"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _private_path_leak_count(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return 1 if PRIVATE_PATH_RE.search(encoded) else 0


def _run_text(args: Sequence[str], *, cwd: Path | None = None, timeout: int = 8) -> str:
    try:
        result = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.rstrip("\n")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _pr149_state(*, repo: Path, include_gh: bool) -> dict[str, Any]:
    fallback = {
        "state": "UNKNOWN",
        "isDraft": True,
        "mergeable": "CONFLICTING",
        "mergeStateStatus": "DIRTY",
        "headRefOid": "",
        "title": "Add complex QA real strict evidence availability bridge audit",
        "url": "https://github.com/chowonje/knowledge-hub/pull/149",
    }
    if not include_gh:
        return fallback
    output = _run_text(
        [
            "gh",
            "pr",
            "view",
            "149",
            "--repo",
            "chowonje/knowledge-hub",
            "--json",
            "state,isDraft,mergeable,mergeStateStatus,headRefOid,title,url",
        ],
        cwd=repo,
        timeout=10,
    )
    if not output:
        return fallback
    try:
        payload = json.loads(output)
    except Exception:
        return fallback
    return {
        "state": _clean_text(payload.get("state")),
        "isDraft": bool(payload.get("isDraft")),
        "mergeable": _clean_text(payload.get("mergeable")),
        "mergeStateStatus": _clean_text(payload.get("mergeStateStatus")),
        "headRefOid": _clean_text(payload.get("headRefOid")),
        "title": _clean_text(payload.get("title")),
        "url": _clean_text(payload.get("url")),
    }


def _action_rows(*, pr149: dict[str, Any], cleanup_plan: dict[str, Any]) -> list[dict[str, Any]]:
    cleanup_policy = dict(cleanup_plan.get("executionPolicy") or {})
    return [
        {
            "actionId": "close_pr149_without_merge",
            "actionType": "external_pr_mutation",
            "targetRef": "github_pr_149",
            "recommended": True,
            "approvalStatus": "pending_user_approval",
            "executionStatus": "not_executed",
            "sequenceOrder": 1,
            "requiresSnapshotFirst": False,
            "requiresPr149ResolvedFirst": False,
            "commandPreview": "gh pr close 149 --repo chowonje/knowledge-hub --comment <sanitized closeout>",
            "reason": (
                "PR #149 is open, draft, and conflicting; the disposition report recommends abandoning "
                "it before public RC rather than merging or recutting it into the text-evidence stack."
            ),
            "currentState": {
                "state": _clean_text(pr149.get("state")),
                "isDraft": bool(pr149.get("isDraft")),
                "mergeable": _clean_text(pr149.get("mergeable")),
                "mergeStateStatus": _clean_text(pr149.get("mergeStateStatus")),
            },
        },
        {
            "actionId": "snapshot_and_clean_canonical_dirty_checkout",
            "actionType": "local_checkout_cleanup",
            "targetRef": "canonical_product_checkout",
            "recommended": True,
            "approvalStatus": "pending_user_approval",
            "executionStatus": "not_executed",
            "sequenceOrder": 2,
            "requiresSnapshotFirst": bool(cleanup_policy.get("requiresSnapshotBeforeCleanup", True)),
            "requiresPr149ResolvedFirst": bool(cleanup_policy.get("requiresPr149ResolvedFirst", True)),
            "commandPreview": "snapshot current dirty state, then clean/archive by cleanup-plan bucket",
            "reason": (
                "The cleanup plan excludes direct dirty-checkout inclusion from the text RC and requires "
                "a checkpoint before any physical cleanup."
            ),
            "currentState": {
                "cleanupPlanStatus": _clean_text(cleanup_plan.get("status")),
                "dirtyRows": int(cleanup_plan.get("dirtyRows") or 0),
                "planRows": int(cleanup_plan.get("planRows") or 0),
            },
        },
    ]


def build_text_evidence_rc_external_action_approval_packet(
    *,
    project_root: Path,
    reports_root: Path,
    include_gh: bool = True,
    generated_at: str | None = None,
) -> dict[str, Any]:
    pr_disposition = _load_json(reports_root / PR149_DISPOSITION_REPORT_REF)
    cleanup_plan = _load_json(reports_root / CANONICAL_DIRTY_CLEANUP_PLAN_REPORT_REF)
    pr149 = _pr149_state(repo=project_root, include_gh=include_gh)
    actions = _action_rows(pr149=pr149, cleanup_plan=cleanup_plan)
    pending_rows = sum(1 for row in actions if row["approvalStatus"] == "pending_user_approval")
    executed_rows = sum(1 for row in actions if row["executionStatus"] != "not_executed")

    report: dict[str, Any] = {
        "schema": TEXT_EVIDENCE_RC_EXTERNAL_ACTION_APPROVAL_PACKET_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "scope": {
            "writes": "report_only",
            "pullRequestMutationPerformed": False,
            "canonicalCheckoutEdited": False,
            "destructiveCleanupRows": 0,
            "vaultScanRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "externalDownloadRows": 0,
            "mergeRows": 0,
            "cherryPickRows": 0,
            "worktreeDeletionRows": 0,
        },
        "inputReports": {
            "pr149DispositionReportRef": PR149_DISPOSITION_REPORT_REF,
            "pr149DispositionStatus": _clean_text(pr_disposition.get("status")),
            "canonicalCleanupPlanReportRef": CANONICAL_DIRTY_CLEANUP_PLAN_REPORT_REF,
            "canonicalCleanupPlanStatus": _clean_text(cleanup_plan.get("status")),
        },
        "approvalPolicy": {
            "requiresExplicitUserApproval": True,
            "noAutoExecution": True,
            "closePr149BeforeCanonicalCleanup": True,
            "snapshotBeforeCanonicalCleanup": True,
            "allowsDirectTextRcMerge": False,
        },
        "actionRows": len(actions),
        "pendingApprovalRows": pending_rows,
        "executedRows": executed_rows,
        "actions": actions,
        "nextAction": "await_explicit_user_approval_to_close_pr149_without_merge",
        "mutationCounters": {
            "externalPrMutationRows": 0,
            "canonicalCleanupRows": 0,
            "destructiveCleanupRows": 0,
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
            "This approval packet does not close PR #149 and does not clean the canonical checkout.",
            "The first executable mutation remains explicit user approval to close PR #149 without merge.",
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
    policy = dict(report.get("approvalPolicy") or {})
    lines = [
        "# Text Evidence RC External Action Approval Packet",
        "",
        f"- status: `{report.get('status')}`",
        f"- actionRows: `{report.get('actionRows')}`",
        f"- pendingApprovalRows: `{report.get('pendingApprovalRows')}`",
        f"- executedRows: `{report.get('executedRows')}`",
        f"- requiresExplicitUserApproval: `{policy.get('requiresExplicitUserApproval')}`",
        f"- noAutoExecution: `{policy.get('noAutoExecution')}`",
        f"- nextAction: `{report.get('nextAction')}`",
        "",
        "## Actions",
        "",
        "| order | action | approval | execution | target |",
        "|---:|---|---|---|---|",
    ]
    for row in report.get("actions", []):
        lines.append(
            f"| `{row.get('sequenceOrder')}` | `{row.get('actionId')}` | `{row.get('approvalStatus')}` | "
            f"`{row.get('executionStatus')}` | `{row.get('targetRef')}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


__all__ = [
    "TEXT_EVIDENCE_RC_EXTERNAL_ACTION_APPROVAL_PACKET_SCHEMA_ID",
    "build_text_evidence_rc_external_action_approval_packet",
    "render_markdown_report",
    "write_report",
]
