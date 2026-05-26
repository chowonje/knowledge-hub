from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers import text_evidence_rc_pr149_close_receipt as receipt
from knowledge_hub.papers.text_evidence_rc_pr149_close_receipt import (
    TEXT_EVIDENCE_RC_PR149_CLOSE_RECEIPT_SCHEMA_ID,
    build_text_evidence_rc_pr149_close_receipt,
    write_report,
)


def _write_approval(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "text_evidence_rc_pr149_close_approval_request.v1.json").write_text(
        json.dumps(
            {
                "schema": "knowledge-hub.paper.text-evidence-rc-pr149-close-approval-request.v1",
                "status": "ready_for_user_decision",
                "approvalRequest": {
                    "safeToExecuteAfterApproval": True,
                    "executionStatus": "not_executed",
                },
            }
        ),
        encoding="utf-8",
    )


def test_pr149_close_receipt_is_pending_before_close_execution(tmp_path: Path) -> None:
    _write_approval(tmp_path)

    report = build_text_evidence_rc_pr149_close_receipt(
        project_root=tmp_path,
        reports_root=tmp_path,
        include_gh=False,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "pending_close_execution"
    assert report["receipt"]["executionVerified"] is False
    assert report["receipt"]["mergePerformed"] is False
    assert report["nextAction"] == "await_user_yes_to_close_pr149_without_merge"
    assert report["blockerRows"] == 1
    assert report["mutationCounters"]["externalPrMutationRows"] == 0
    assert report["mutationCounters"]["canonicalCleanupRows"] == 0
    assert report["privatePathLeakRows"] == 0
    assert validate_payload(report, TEXT_EVIDENCE_RC_PR149_CLOSE_RECEIPT_SCHEMA_ID, strict=True).ok


def test_pr149_close_receipt_writer_is_path_sanitized(tmp_path: Path) -> None:
    _write_approval(tmp_path)
    report = build_text_evidence_rc_pr149_close_receipt(
        project_root=tmp_path,
        reports_root=tmp_path,
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


def test_pr149_close_receipt_verifies_closed_pr(tmp_path: Path, monkeypatch) -> None:
    _write_approval(tmp_path)

    def fake_run_text(args, *, cwd=None, timeout=8):
        if args[:3] == ["gh", "pr", "view"]:
            return json.dumps(
                {
                    "state": "CLOSED",
                    "isDraft": True,
                    "mergeable": "CONFLICTING",
                    "mergeStateStatus": "DIRTY",
                    "headRefName": "codex/complex-qa-real-strict-evidence-availability-bridge-audit-20260520",
                    "baseRefName": "main",
                    "url": "https://github.com/chowonje/knowledge-hub/pull/149",
                }
            )
        return ""

    monkeypatch.setattr(receipt, "_run_text", fake_run_text)

    report = build_text_evidence_rc_pr149_close_receipt(
        project_root=tmp_path,
        reports_root=tmp_path,
        include_gh=True,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "closed_verified"
    assert report["receipt"]["executionVerified"] is True
    assert report["receipt"]["mergePerformed"] is False
    assert report["nextAction"] == "proceed_to_canonical_dirty_cleanup_snapshot"
    assert report["blockerRows"] == 0
    assert validate_payload(report, TEXT_EVIDENCE_RC_PR149_CLOSE_RECEIPT_SCHEMA_ID, strict=True).ok
