from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.text_evidence_rc_pr149_close_approval_request import (
    TEXT_EVIDENCE_RC_PR149_CLOSE_APPROVAL_REQUEST_SCHEMA_ID,
    build_text_evidence_rc_pr149_close_approval_request,
    write_report,
)


def _write_preflight(root: Path, *, ready: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("text_evidence_rc_external_action_preflight.v1.json").write_text(
        json.dumps(
            {
                "schema": "knowledge-hub.paper.text-evidence-rc-external-action-preflight.v1",
                "status": "ready_for_user_approval" if ready else "blocked",
                "readyForUserApproval": ready,
                "pullRequest149": {
                    "state": "OPEN",
                    "isDraft": True,
                    "mergeable": "CONFLICTING",
                    "mergeStateStatus": "DIRTY",
                    "headRefName": "codex/complex-qa-real-strict-evidence-availability-bridge-audit-20260520",
                    "baseRefName": "main",
                    "expectedHoldState": ready,
                },
            }
        ),
        encoding="utf-8",
    )


def test_pr149_close_approval_request_is_ready_but_not_executed(tmp_path: Path) -> None:
    _write_preflight(tmp_path)

    report = build_text_evidence_rc_pr149_close_approval_request(
        reports_root=tmp_path,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "ready_for_user_decision"
    assert report["approvalRequest"]["requiresExplicitUserApproval"] is True
    assert report["approvalRequest"]["executionStatus"] == "not_executed"
    assert report["approvalRequest"]["safeToExecuteAfterApproval"] is True
    assert report["approvalRequest"]["recommendedDecision"] == "approve_close_without_merge"
    assert report["nextAction"] == "await_user_yes_to_close_pr149_without_merge"
    assert report["mutationCounters"]["externalPrMutationRows"] == 0
    assert report["mutationCounters"]["canonicalCleanupRows"] == 0
    assert report["privatePathLeakRows"] == 0
    assert validate_payload(report, TEXT_EVIDENCE_RC_PR149_CLOSE_APPROVAL_REQUEST_SCHEMA_ID, strict=True).ok


def test_pr149_close_approval_request_blocks_without_ready_preflight(tmp_path: Path) -> None:
    _write_preflight(tmp_path, ready=False)

    report = build_text_evidence_rc_pr149_close_approval_request(
        reports_root=tmp_path,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["approvalRequest"]["safeToExecuteAfterApproval"] is False
    assert report["blockerRows"] == 1


def test_pr149_close_approval_request_writer_is_path_sanitized(tmp_path: Path) -> None:
    _write_preflight(tmp_path)
    report = build_text_evidence_rc_pr149_close_approval_request(
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
