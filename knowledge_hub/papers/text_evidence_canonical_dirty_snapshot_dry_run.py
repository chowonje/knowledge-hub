"""Dry-run snapshot manifest for canonical dirty cleanup readiness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence

from knowledge_hub.papers.figure_caption_artifact_vertical_slice import utc_now_iso

TEXT_EVIDENCE_CANONICAL_DIRTY_SNAPSHOT_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.text-evidence-canonical-dirty-snapshot-dry-run.v1"
)

EXTERNAL_ACTION_APPROVAL_PACKET_REPORT_REF = "text_evidence_rc_external_action_approval_packet.v1.json"

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


def _status_lines(canonical_repo: Path | None) -> list[str]:
    if canonical_repo is None or not canonical_repo.exists():
        return []
    output = _git_text(canonical_repo, "status", "--short")
    return [line for line in output.splitlines() if line.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _status_fingerprint(status_lines: Sequence[str]) -> str:
    normalized = "\n".join(status_lines).rstrip() + "\n"
    return _sha256_text(normalized)


def _bucket_summary_from_status(status_lines: Sequence[str]) -> dict[str, int]:
    tracked = 0
    untracked = 0
    for line in status_lines:
        if line.startswith("??"):
            untracked += 1
        else:
            tracked += 1
    return {"trackedDirtyRows": tracked, "untrackedRows": untracked}


def build_text_evidence_canonical_dirty_snapshot_dry_run_report(
    *,
    project_root: Path,
    canonical_repo: Path | None,
    reports_root: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    approval_packet = _load_json(reports_root / EXTERNAL_ACTION_APPROVAL_PACKET_REPORT_REF)
    status_lines = _status_lines(canonical_repo)
    bucket_summary = _bucket_summary_from_status(status_lines)
    canonical_available = canonical_repo is not None and canonical_repo.exists()

    report: dict[str, Any] = {
        "schema": TEXT_EVIDENCE_CANONICAL_DIRTY_SNAPSHOT_DRY_RUN_SCHEMA_ID,
        "status": "ready" if canonical_available and approval_packet else "blocked",
        "generatedAt": generated_at or utc_now_iso(),
        "scope": {
            "writes": "report_only",
            "snapshotWritten": False,
            "canonicalCheckoutEdited": False,
            "fileContentReadRows": 0,
            "vaultScanRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "externalDownloadRows": 0,
            "destructiveCleanupRows": 0,
        },
        "approvalPacketReportRef": EXTERNAL_ACTION_APPROVAL_PACKET_REPORT_REF,
        "approvalPacketStatus": _clean_text(approval_packet.get("status")),
        "canonicalCheckout": {
            "available": bool(canonical_available),
            "repoRef": "canonical_product_checkout",
            "branch": _git_text(canonical_repo, "rev-parse", "--abbrev-ref", "HEAD") if canonical_available else "",
            "head": _git_text(canonical_repo, "rev-parse", "--short", "HEAD") if canonical_available else "",
            "dirtyRows": len(status_lines),
        },
        "dirtyRows": len(status_lines),
        "trackedDirtyRows": int(bucket_summary["trackedDirtyRows"]),
        "untrackedRows": int(bucket_summary["untrackedRows"]),
        "statusFingerprint": _status_fingerprint(status_lines),
        "snapshotPlan": {
            "snapshotMode": "status_fingerprint_only",
            "snapshotTargetRef": "operator_selected_location_after_approval",
            "requiresExplicitApproval": True,
            "requiresPr149ResolvedFirst": True,
            "capturesFileContent": False,
        },
        "nextAction": "close_pr149_without_merge_before_writing_cleanup_snapshot",
        "mutationCounters": {
            "snapshotWriteRows": 0,
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
            "This dry-run records only a git-status fingerprint, not file contents.",
            "No snapshot file is written outside the report and no cleanup is executed.",
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
    checkout = dict(report.get("canonicalCheckout") or {})
    plan = dict(report.get("snapshotPlan") or {})
    lines = [
        "# Text Evidence Canonical Dirty Snapshot Dry Run",
        "",
        f"- status: `{report.get('status')}`",
        f"- dirtyRows: `{report.get('dirtyRows')}`",
        f"- trackedDirtyRows: `{report.get('trackedDirtyRows')}`",
        f"- untrackedRows: `{report.get('untrackedRows')}`",
        f"- canonicalBranch: `{checkout.get('branch')}`",
        f"- canonicalHead: `{checkout.get('head')}`",
        f"- statusFingerprint: `{report.get('statusFingerprint')}`",
        f"- snapshotMode: `{plan.get('snapshotMode')}`",
        f"- capturesFileContent: `{plan.get('capturesFileContent')}`",
        f"- nextAction: `{report.get('nextAction')}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


__all__ = [
    "TEXT_EVIDENCE_CANONICAL_DIRTY_SNAPSHOT_DRY_RUN_SCHEMA_ID",
    "build_text_evidence_canonical_dirty_snapshot_dry_run_report",
    "render_markdown_report",
    "write_report",
]
