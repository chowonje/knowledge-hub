from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.text_evidence_rc_external_action_approval_packet import (
    TEXT_EVIDENCE_RC_EXTERNAL_ACTION_APPROVAL_PACKET_SCHEMA_ID,
    build_text_evidence_rc_external_action_approval_packet,
    write_report,
)


def _write_inputs(root: Path) -> None:
    (root / "text_evidence_pr149_disposition.v1.json").write_text(
        json.dumps({"schema": "x", "status": "ready"}), encoding="utf-8"
    )
    (root / "text_evidence_canonical_dirty_cleanup_plan.v1.json").write_text(
        json.dumps(
            {
                "schema": "x",
                "status": "ready",
                "dirtyRows": 168,
                "planRows": 11,
                "executionPolicy": {
                    "requiresPr149ResolvedFirst": True,
                    "requiresSnapshotBeforeCleanup": True,
                },
            }
        ),
        encoding="utf-8",
    )


def test_external_action_packet_requires_approval_and_executes_nothing(tmp_path: Path) -> None:
    _write_inputs(tmp_path)

    report = build_text_evidence_rc_external_action_approval_packet(
        project_root=tmp_path,
        reports_root=tmp_path,
        include_gh=False,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["approvalPolicy"]["requiresExplicitUserApproval"] is True
    assert report["approvalPolicy"]["noAutoExecution"] is True
    assert report["approvalPolicy"]["closePr149BeforeCanonicalCleanup"] is True
    assert report["actionRows"] == 2
    assert report["pendingApprovalRows"] == 2
    assert report["executedRows"] == 0
    assert report["actions"][0]["actionId"] == "close_pr149_without_merge"
    assert report["actions"][1]["requiresPr149ResolvedFirst"] is True
    assert report["mutationCounters"]["externalPrMutationRows"] == 0
    assert report["mutationCounters"]["canonicalCleanupRows"] == 0
    assert report["privatePathLeakRows"] == 0
    assert validate_payload(report, TEXT_EVIDENCE_RC_EXTERNAL_ACTION_APPROVAL_PACKET_SCHEMA_ID, strict=True).ok


def test_external_action_packet_writer_is_path_sanitized(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    report = build_text_evidence_rc_external_action_approval_packet(
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
