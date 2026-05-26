from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.text_evidence_canonical_dirty_snapshot_dry_run import (
    TEXT_EVIDENCE_CANONICAL_DIRTY_SNAPSHOT_DRY_RUN_SCHEMA_ID,
    build_text_evidence_canonical_dirty_snapshot_dry_run_report,
    write_report,
)


def _write_approval_packet(root: Path) -> None:
    (root / "text_evidence_rc_external_action_approval_packet.v1.json").write_text(
        json.dumps({"schema": "x", "status": "ready"}), encoding="utf-8"
    )


def test_snapshot_dry_run_records_status_fingerprint_without_snapshot_write(tmp_path: Path) -> None:
    _write_approval_packet(tmp_path)

    report = build_text_evidence_canonical_dirty_snapshot_dry_run_report(
        project_root=tmp_path,
        canonical_repo=None,
        reports_root=tmp_path,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["scope"]["snapshotWritten"] is False
    assert report["scope"]["fileContentReadRows"] == 0
    assert report["snapshotPlan"]["snapshotMode"] == "status_fingerprint_only"
    assert report["snapshotPlan"]["capturesFileContent"] is False
    assert report["mutationCounters"]["snapshotWriteRows"] == 0
    assert report["mutationCounters"]["canonicalCleanupRows"] == 0


def test_snapshot_dry_run_schema_with_empty_git_repo(tmp_path: Path) -> None:
    _write_approval_packet(tmp_path)
    canonical = tmp_path / "repo"
    canonical.mkdir()
    (canonical / ".git").mkdir()

    report = build_text_evidence_canonical_dirty_snapshot_dry_run_report(
        project_root=tmp_path,
        canonical_repo=canonical,
        reports_root=tmp_path,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["dirtyRows"] == 0
    assert report["trackedDirtyRows"] == 0
    assert report["untrackedRows"] == 0
    assert report["privatePathLeakRows"] == 0
    assert validate_payload(report, TEXT_EVIDENCE_CANONICAL_DIRTY_SNAPSHOT_DRY_RUN_SCHEMA_ID, strict=True).ok


def test_snapshot_dry_run_writer_is_path_sanitized(tmp_path: Path) -> None:
    _write_approval_packet(tmp_path)
    report = build_text_evidence_canonical_dirty_snapshot_dry_run_report(
        project_root=tmp_path,
        canonical_repo=None,
        reports_root=tmp_path,
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
