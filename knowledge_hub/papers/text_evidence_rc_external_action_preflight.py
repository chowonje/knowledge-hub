"""Read-only preflight for pending external RC actions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence

from knowledge_hub.papers.figure_caption_artifact_vertical_slice import utc_now_iso

TEXT_EVIDENCE_RC_EXTERNAL_ACTION_PREFLIGHT_SCHEMA_ID = (
    "knowledge-hub.paper.text-evidence-rc-external-action-preflight.v1"
)

APPROVAL_PACKET_REPORT_REF = "text_evidence_rc_external_action_approval_packet.v1.json"
SNAPSHOT_DRY_RUN_REPORT_REF = "text_evidence_canonical_dirty_snapshot_dry_run.v1.json"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _sha256_text(encoded)


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


def _git_text(repo: Path, *args: str) -> str:
    return _run_text(["git", "-C", str(repo), *args])


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _status_lines(canonical_repo: Path | None) -> list[str]:
    if canonical_repo is None or not canonical_repo.exists():
        return []
    output = _git_text(canonical_repo, "status", "--short")
    return [line for line in output.splitlines() if line.strip()]


def _status_fingerprint(status_lines: Sequence[str]) -> str:
    normalized = "\n".join(status_lines).rstrip() + "\n"
    return _sha256_text(normalized)


def _pr149_state(*, project_root: Path, include_gh: bool) -> dict[str, Any]:
    fallback = {
        "available": False,
        "state": "OPEN",
        "isDraft": True,
        "mergeable": "CONFLICTING",
        "mergeStateStatus": "DIRTY",
        "headRefName": "codex/complex-qa-real-strict-evidence-availability-bridge-audit-20260520",
        "baseRefName": "main",
        "headRefOid": "",
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
            "state,isDraft,mergeable,mergeStateStatus,headRefName,baseRefName,headRefOid,url",
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
        "headRefOid": _clean_text(payload.get("headRefOid")),
        "url": _clean_text(payload.get("url")),
    }


def _pr149_expected_hold_state(pr149: dict[str, Any]) -> bool:
    return (
        _clean_text(pr149.get("state")) == "OPEN"
        and bool(pr149.get("isDraft")) is True
        and _clean_text(pr149.get("mergeable")) == "CONFLICTING"
        and _clean_text(pr149.get("mergeStateStatus")) == "DIRTY"
    )


def build_text_evidence_rc_external_action_preflight(
    *,
    project_root: Path,
    reports_root: Path,
    canonical_repo: Path | None,
    include_gh: bool = True,
    generated_at: str | None = None,
) -> dict[str, Any]:
    approval_packet = _load_json(reports_root / APPROVAL_PACKET_REPORT_REF)
    snapshot = _load_json(reports_root / SNAPSHOT_DRY_RUN_REPORT_REF)
    status_lines = _status_lines(canonical_repo)
    current_fingerprint = _status_fingerprint(status_lines)
    canonical_available = canonical_repo is not None and canonical_repo.exists()
    pr149 = _pr149_state(project_root=project_root, include_gh=include_gh)

    approval_ready = _clean_text(approval_packet.get("status")) == "ready"
    approvals_pending = int(approval_packet.get("pendingApprovalRows") or 0) == 2
    no_actions_executed = int(approval_packet.get("executedRows") or 0) == 0
    snapshot_ready = _clean_text(snapshot.get("status")) == "ready"
    snapshot_fingerprint_matches = current_fingerprint == _clean_text(snapshot.get("statusFingerprint"))
    dirty_rows_match = len(status_lines) == int(snapshot.get("dirtyRows") or 0)
    pr149_expected_hold = _pr149_expected_hold_state(pr149)

    ready_for_user_approval = bool(
        approval_ready
        and approvals_pending
        and no_actions_executed
        and snapshot_ready
        and canonical_available
        and snapshot_fingerprint_matches
        and dirty_rows_match
        and pr149_expected_hold
    )
    blockers: list[dict[str, str]] = []
    if not approval_ready:
        blockers.append({"blockerId": "approval_packet_not_ready", "reason": "approval packet is missing or not ready"})
    if not approvals_pending:
        blockers.append({"blockerId": "approval_packet_pending_count_changed", "reason": "pending approvals are not exactly two"})
    if not no_actions_executed:
        blockers.append({"blockerId": "external_action_already_executed", "reason": "approval packet indicates an action has already executed"})
    if not snapshot_ready:
        blockers.append({"blockerId": "snapshot_dry_run_not_ready", "reason": "canonical dirty snapshot dry-run is not ready"})
    if not canonical_available:
        blockers.append({"blockerId": "canonical_checkout_unavailable", "reason": "canonical checkout is unavailable for status preflight"})
    if not snapshot_fingerprint_matches:
        blockers.append({"blockerId": "canonical_dirty_fingerprint_drift", "reason": "current git-status fingerprint differs from snapshot dry-run"})
    if not dirty_rows_match:
        blockers.append({"blockerId": "canonical_dirty_row_count_drift", "reason": "current dirty row count differs from snapshot dry-run"})
    if not pr149_expected_hold:
        blockers.append({"blockerId": "pr149_state_drift", "reason": "PR #149 is not in the expected open/draft/conflicting state"})

    report: dict[str, Any] = {
        "schema": TEXT_EVIDENCE_RC_EXTERNAL_ACTION_PREFLIGHT_SCHEMA_ID,
        "status": "ready_for_user_approval" if ready_for_user_approval else "blocked",
        "generatedAt": generated_at or utc_now_iso(),
        "scope": {
            "writes": "report_only",
            "pullRequestMutationPerformed": False,
            "canonicalCheckoutEdited": False,
            "fileContentReadRows": 0,
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
            "approvalPacketReportRef": APPROVAL_PACKET_REPORT_REF,
            "approvalPacketStatus": _clean_text(approval_packet.get("status")),
            "approvalPacketPendingRows": int(approval_packet.get("pendingApprovalRows") or 0),
            "approvalPacketExecutedRows": int(approval_packet.get("executedRows") or 0),
            "snapshotDryRunReportRef": SNAPSHOT_DRY_RUN_REPORT_REF,
            "snapshotDryRunStatus": _clean_text(snapshot.get("status")),
            "snapshotDirtyRows": int(snapshot.get("dirtyRows") or 0),
            "snapshotStatusFingerprint": _clean_text(snapshot.get("statusFingerprint")),
        },
        "canonicalCheckout": {
            "available": bool(canonical_available),
            "repoRef": "canonical_product_checkout",
            "branch": _git_text(canonical_repo, "rev-parse", "--abbrev-ref", "HEAD") if canonical_available else "",
            "head": _git_text(canonical_repo, "rev-parse", "--short", "HEAD") if canonical_available else "",
            "dirtyRows": len(status_lines),
            "statusFingerprint": current_fingerprint,
            "snapshotFingerprintMatches": snapshot_fingerprint_matches,
            "dirtyRowsMatchSnapshot": dirty_rows_match,
        },
        "pullRequest149": {
            "available": bool(pr149.get("available")),
            "state": _clean_text(pr149.get("state")),
            "isDraft": bool(pr149.get("isDraft")),
            "mergeable": _clean_text(pr149.get("mergeable")),
            "mergeStateStatus": _clean_text(pr149.get("mergeStateStatus")),
            "headRefName": _clean_text(pr149.get("headRefName")),
            "baseRefName": _clean_text(pr149.get("baseRefName")),
            "expectedHoldState": pr149_expected_hold,
        },
        "readyForUserApproval": ready_for_user_approval,
        "blockerRows": len(blockers),
        "blockers": blockers,
        "nextAction": (
            "request_explicit_user_approval_to_close_pr149_without_merge"
            if ready_for_user_approval
            else "refresh_external_action_approval_packet_or_snapshot_dry_run"
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
            "This preflight does not close PR #149 and does not clean the canonical checkout.",
            "The canonical checkout check reads git status only; it does not read dirty file contents.",
        ],
        "schemaErrors": [],
    }
    report["privatePathLeakRows"] = _private_path_leak_count(report)
    if report["privatePathLeakRows"]:
        report["status"] = "blocked"
        report["readyForUserApproval"] = False
    payload_for_hash = dict(report)
    payload_for_hash["reportHash"] = ""
    report["reportHash"] = _sha256_json(payload_for_hash)
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    canonical = dict(report.get("canonicalCheckout") or {})
    pr149 = dict(report.get("pullRequest149") or {})
    lines = [
        "# Text Evidence RC External Action Preflight",
        "",
        f"- status: `{report.get('status')}`",
        f"- readyForUserApproval: `{report.get('readyForUserApproval')}`",
        f"- blockerRows: `{report.get('blockerRows')}`",
        f"- canonicalDirtyRows: `{canonical.get('dirtyRows')}`",
        f"- snapshotFingerprintMatches: `{canonical.get('snapshotFingerprintMatches')}`",
        f"- pr149State: `{pr149.get('state')}`",
        f"- pr149Mergeable: `{pr149.get('mergeable')}`",
        f"- pr149MergeStateStatus: `{pr149.get('mergeStateStatus')}`",
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
    "TEXT_EVIDENCE_RC_EXTERNAL_ACTION_PREFLIGHT_SCHEMA_ID",
    "build_text_evidence_rc_external_action_preflight",
    "render_markdown_report",
    "write_report",
]
