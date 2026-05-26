"""Read-only receipt check for PR #149 close execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence

from knowledge_hub.papers.figure_caption_artifact_vertical_slice import utc_now_iso

TEXT_EVIDENCE_RC_PR149_CLOSE_RECEIPT_SCHEMA_ID = "knowledge-hub.paper.text-evidence-rc-pr149-close-receipt.v1"

PR149_CLOSE_APPROVAL_REQUEST_REPORT_REF = "text_evidence_rc_pr149_close_approval_request.v1.json"

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


def _pr149_state(*, project_root: Path, include_gh: bool) -> dict[str, Any]:
    fallback = {
        "available": False,
        "state": "UNKNOWN",
        "isDraft": True,
        "mergeable": "CONFLICTING",
        "mergeStateStatus": "DIRTY",
        "headRefName": "codex/complex-qa-real-strict-evidence-availability-bridge-audit-20260520",
        "baseRefName": "main",
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
            "state,isDraft,mergeable,mergeStateStatus,headRefName,baseRefName,url",
        ],
        cwd=project_root,
        timeout=10,
    )
    if not output:
        return fallback
    try:
        payload = json.loads(output)
    except Exception:
        return fallback
    return {
        "available": True,
        "state": _clean_text(payload.get("state")),
        "isDraft": bool(payload.get("isDraft")),
        "mergeable": _clean_text(payload.get("mergeable")),
        "mergeStateStatus": _clean_text(payload.get("mergeStateStatus")),
        "headRefName": _clean_text(payload.get("headRefName")),
        "baseRefName": _clean_text(payload.get("baseRefName")),
        "url": _clean_text(payload.get("url")),
    }


def build_text_evidence_rc_pr149_close_receipt(
    *,
    project_root: Path,
    reports_root: Path,
    include_gh: bool = True,
    generated_at: str | None = None,
) -> dict[str, Any]:
    approval_request = _load_json(reports_root / PR149_CLOSE_APPROVAL_REQUEST_REPORT_REF)
    request = dict(approval_request.get("approvalRequest") or {})
    pr149 = _pr149_state(project_root=project_root, include_gh=include_gh)
    approval_ready = (
        _clean_text(approval_request.get("status")) == "ready_for_user_decision"
        and bool(request.get("safeToExecuteAfterApproval"))
    )
    close_verified = _clean_text(pr149.get("state")) == "CLOSED"
    status = "closed_verified" if approval_ready and close_verified else "pending_close_execution"
    blockers: list[dict[str, str]] = []
    if not approval_ready:
        blockers.append(
            {
                "blockerId": "pr149_close_approval_request_not_ready",
                "reason": "close receipt requires a ready approval request before close execution can be verified",
            }
        )
    if approval_ready and not close_verified:
        blockers.append(
            {
                "blockerId": "pr149_not_closed",
                "reason": "PR #149 is not closed yet; explicit user approval and close execution remain pending",
            }
        )

    report: dict[str, Any] = {
        "schema": TEXT_EVIDENCE_RC_PR149_CLOSE_RECEIPT_SCHEMA_ID,
        "status": status,
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
            "approvalRequestReportRef": PR149_CLOSE_APPROVAL_REQUEST_REPORT_REF,
            "approvalRequestStatus": _clean_text(approval_request.get("status")),
            "approvalRequestSafeToExecute": bool(request.get("safeToExecuteAfterApproval")),
            "approvalRequestExecutionStatus": _clean_text(request.get("executionStatus")),
        },
        "pullRequest149": {
            "available": bool(pr149.get("available")),
            "state": _clean_text(pr149.get("state")),
            "isDraft": bool(pr149.get("isDraft")),
            "mergeable": _clean_text(pr149.get("mergeable")),
            "mergeStateStatus": _clean_text(pr149.get("mergeStateStatus")),
            "headRefName": _clean_text(pr149.get("headRefName")),
            "baseRefName": _clean_text(pr149.get("baseRefName")),
            "closedVerified": close_verified,
        },
        "receipt": {
            "actionId": "close_pr149_without_merge",
            "executionVerified": close_verified,
            "mergePerformed": False,
            "nextAllowedAction": (
                "proceed_to_canonical_dirty_cleanup_snapshot"
                if close_verified
                else "await_user_yes_to_close_pr149_without_merge"
            ),
        },
        "blockerRows": len(blockers),
        "blockers": blockers,
        "nextAction": (
            "proceed_to_canonical_dirty_cleanup_snapshot"
            if close_verified
            else "await_user_yes_to_close_pr149_without_merge"
        ),
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
            "This receipt check is read-only and does not close PR #149.",
            "Until PR #149 is closed, canonical dirty cleanup must remain pending.",
        ],
        "schemaErrors": [],
    }
    report["privatePathLeakRows"] = _private_path_leak_count(report)
    if report["privatePathLeakRows"]:
        report["status"] = "pending_close_execution"
    payload_for_hash = dict(report)
    payload_for_hash["reportHash"] = ""
    report["reportHash"] = _sha256_json(payload_for_hash)
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    pr149 = dict(report.get("pullRequest149") or {})
    receipt = dict(report.get("receipt") or {})
    lines = [
        "# Text Evidence RC PR149 Close Receipt",
        "",
        f"- status: `{report.get('status')}`",
        f"- executionVerified: `{receipt.get('executionVerified')}`",
        f"- pr149State: `{pr149.get('state')}`",
        f"- mergePerformed: `{receipt.get('mergePerformed')}`",
        f"- blockerRows: `{report.get('blockerRows')}`",
        f"- nextAction: `{report.get('nextAction')}`",
        "",
        "## Blockers",
    ]
    for row in report.get("blockers") or []:
        lines.append(f"- `{row.get('blockerId')}`: {row.get('reason')}")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


__all__ = [
    "TEXT_EVIDENCE_RC_PR149_CLOSE_RECEIPT_SCHEMA_ID",
    "build_text_evidence_rc_pr149_close_receipt",
    "render_markdown_report",
    "write_report",
]
