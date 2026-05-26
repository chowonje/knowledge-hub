"""Approval request packet for closing PR #149 without merge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.figure_caption_artifact_vertical_slice import utc_now_iso

TEXT_EVIDENCE_RC_PR149_CLOSE_APPROVAL_REQUEST_SCHEMA_ID = (
    "knowledge-hub.paper.text-evidence-rc-pr149-close-approval-request.v1"
)

EXTERNAL_ACTION_PREFLIGHT_REPORT_REF = "text_evidence_rc_external_action_preflight.v1.json"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)

CLOSE_COMMENT = (
    "Closing this draft without merge for the v0.1 text-evidence RC convergence path. "
    "The branch is preserved as later side-track material; the current RC uses the "
    "text-only evidence stack and keeps strict-evidence promotion out of scope."
)


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


def build_text_evidence_rc_pr149_close_approval_request(
    *,
    reports_root: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    preflight = _load_json(reports_root / EXTERNAL_ACTION_PREFLIGHT_REPORT_REF)
    pr149 = dict(preflight.get("pullRequest149") or {})
    ready = (
        _clean_text(preflight.get("status")) == "ready_for_user_approval"
        and bool(preflight.get("readyForUserApproval"))
        and bool(pr149.get("expectedHoldState"))
    )
    blockers: list[dict[str, str]] = []
    if not ready:
        blockers.append(
            {
                "blockerId": "external_action_preflight_not_ready",
                "reason": "PR #149 close request requires a ready external action preflight.",
            }
        )

    report: dict[str, Any] = {
        "schema": TEXT_EVIDENCE_RC_PR149_CLOSE_APPROVAL_REQUEST_SCHEMA_ID,
        "status": "ready_for_user_decision" if ready else "blocked",
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
            "externalActionPreflightReportRef": EXTERNAL_ACTION_PREFLIGHT_REPORT_REF,
            "externalActionPreflightStatus": _clean_text(preflight.get("status")),
            "externalActionPreflightReady": bool(preflight.get("readyForUserApproval")),
        },
        "approvalRequest": {
            "actionId": "close_pr149_without_merge",
            "targetRef": "github_pr_149",
            "requiresExplicitUserApproval": True,
            "userDecisionRequired": True,
            "executionStatus": "not_executed",
            "recommendedDecision": "approve_close_without_merge",
            "commandPreview": "gh pr close 149 --repo chowonje/knowledge-hub --comment <sanitized closeout>",
            "closeComment": CLOSE_COMMENT,
            "safeToExecuteAfterApproval": ready,
        },
        "pullRequest149": {
            "state": _clean_text(pr149.get("state")),
            "isDraft": bool(pr149.get("isDraft")),
            "mergeable": _clean_text(pr149.get("mergeable")),
            "mergeStateStatus": _clean_text(pr149.get("mergeStateStatus")),
            "headRefName": _clean_text(pr149.get("headRefName")),
            "baseRefName": _clean_text(pr149.get("baseRefName")),
            "expectedHoldState": bool(pr149.get("expectedHoldState")),
        },
        "blockerRows": len(blockers),
        "blockers": blockers,
        "nextAction": (
            "await_user_yes_to_close_pr149_without_merge"
            if ready
            else "refresh_external_action_preflight_before_close_request"
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
            "This packet is an approval request only; it does not close PR #149.",
            "Closing PR #149 is an external mutation and must wait for explicit user approval.",
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
    request = dict(report.get("approvalRequest") or {})
    pr149 = dict(report.get("pullRequest149") or {})
    lines = [
        "# Text Evidence RC PR149 Close Approval Request",
        "",
        f"- status: `{report.get('status')}`",
        f"- actionId: `{request.get('actionId')}`",
        f"- executionStatus: `{request.get('executionStatus')}`",
        f"- recommendedDecision: `{request.get('recommendedDecision')}`",
        f"- safeToExecuteAfterApproval: `{request.get('safeToExecuteAfterApproval')}`",
        f"- pr149State: `{pr149.get('state')}`",
        f"- pr149Mergeable: `{pr149.get('mergeable')}`",
        f"- pr149MergeStateStatus: `{pr149.get('mergeStateStatus')}`",
        f"- nextAction: `{report.get('nextAction')}`",
        "",
        "## Close Comment",
        "",
        str(request.get("closeComment") or ""),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


__all__ = [
    "TEXT_EVIDENCE_RC_PR149_CLOSE_APPROVAL_REQUEST_SCHEMA_ID",
    "build_text_evidence_rc_pr149_close_approval_request",
    "render_markdown_report",
    "write_report",
]
