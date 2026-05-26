from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.text_evidence_rc_external_action_preflight import (
    TEXT_EVIDENCE_RC_EXTERNAL_ACTION_PREFLIGHT_SCHEMA_ID,
    build_text_evidence_rc_external_action_preflight,
    write_report,
)


def _sha256_empty_status() -> str:
    import hashlib

    return "sha256:" + hashlib.sha256("\n".encode("utf-8")).hexdigest()


def _write_inputs(root: Path, *, fingerprint: str | None = None) -> None:
    (root / "text_evidence_rc_external_action_approval_packet.v1.json").write_text(
        json.dumps(
            {
                "schema": "x",
                "status": "ready",
                "pendingApprovalRows": 2,
                "executedRows": 0,
            }
        ),
        encoding="utf-8",
    )
    (root / "text_evidence_canonical_dirty_snapshot_dry_run.v1.json").write_text(
        json.dumps(
            {
                "schema": "x",
                "status": "ready",
                "dirtyRows": 0,
                "statusFingerprint": fingerprint or _sha256_empty_status(),
            }
        ),
        encoding="utf-8",
    )


def test_external_action_preflight_ready_when_snapshot_and_pr_state_match(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    _write_inputs(reports)
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / ".git").mkdir()

    report = build_text_evidence_rc_external_action_preflight(
        project_root=tmp_path,
        reports_root=reports,
        canonical_repo=canonical,
        include_gh=False,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "ready_for_user_approval"
    assert report["readyForUserApproval"] is True
    assert report["blockerRows"] == 0
    assert report["canonicalCheckout"]["dirtyRows"] == 0
    assert report["canonicalCheckout"]["snapshotFingerprintMatches"] is True
    assert report["pullRequest149"]["expectedHoldState"] is True
    assert report["scope"]["fileContentReadRows"] == 0
    assert report["mutationCounters"]["externalPrMutationRows"] == 0
    assert report["mutationCounters"]["canonicalCleanupRows"] == 0
    assert report["privatePathLeakRows"] == 0
    assert validate_payload(report, TEXT_EVIDENCE_RC_EXTERNAL_ACTION_PREFLIGHT_SCHEMA_ID, strict=True).ok


def test_external_action_preflight_blocks_on_snapshot_drift(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    _write_inputs(reports, fingerprint="sha256:" + "0" * 64)
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / ".git").mkdir()

    report = build_text_evidence_rc_external_action_preflight(
        project_root=tmp_path,
        reports_root=reports,
        canonical_repo=canonical,
        include_gh=False,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["readyForUserApproval"] is False
    assert any(row["blockerId"] == "canonical_dirty_fingerprint_drift" for row in report["blockers"])


def test_external_action_preflight_writer_is_path_sanitized(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    _write_inputs(reports)
    report = build_text_evidence_rc_external_action_preflight(
        project_root=tmp_path,
        reports_root=reports,
        canonical_repo=None,
        include_gh=False,
        generated_at="2026-05-26T00:00:00Z",
    )
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_report(report, json_path=json_path, markdown_path=md_path)

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert json.loads(json_path.read_text(encoding="utf-8"))["privatePathLeakRows"] == 0
