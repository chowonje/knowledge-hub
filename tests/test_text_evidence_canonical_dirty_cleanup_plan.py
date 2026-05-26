from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.text_evidence_canonical_dirty_cleanup_plan import (
    TEXT_EVIDENCE_CANONICAL_DIRTY_CLEANUP_PLAN_SCHEMA_ID,
    build_text_evidence_canonical_dirty_cleanup_plan_report,
    write_report,
)


def _write_bucket_decision(root: Path) -> None:
    report_path = root / "text_evidence_canonical_dirty_bucket_decision.v1.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "knowledge-hub.paper.text-evidence-canonical-dirty-bucket-decision.v1",
                "status": "ready",
                "buckets": [
                    {
                        "bucket": "evidence_spine_or_source_contract_stack",
                        "dirtyRows": 2,
                        "decision": "exclude_from_text_rc_clean_replay_candidate",
                        "publicRcAction": "defer_to_clean_replay",
                    },
                    {
                        "bucket": "workspace_process_record",
                        "dirtyRows": 1,
                        "decision": "drop_or_move_out_of_product_checkout",
                        "publicRcAction": "exclude_from_public_rc",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_cleanup_plan_requires_approval_and_keeps_mutations_zero(tmp_path: Path) -> None:
    _write_bucket_decision(tmp_path)

    report = build_text_evidence_canonical_dirty_cleanup_plan_report(
        reports_root=tmp_path,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["dirtyRows"] == 3
    assert report["planRows"] == 2
    assert report["manualPlanRows"] == 0
    assert report["executionPolicy"]["requiresExplicitApproval"] is True
    assert report["executionPolicy"]["requiresPr149ResolvedFirst"] is True
    assert report["executionPolicy"]["requiresSnapshotBeforeCleanup"] is True
    assert report["executionPolicy"]["allowsDirectTextRcMerge"] is False
    assert report["mutationCounters"]["destructiveCleanupRows"] == 0
    assert report["mutationCounters"]["vaultScanRows"] == 0
    assert report["privatePathLeakRows"] == 0
    assert validate_payload(report, TEXT_EVIDENCE_CANONICAL_DIRTY_CLEANUP_PLAN_SCHEMA_ID, strict=True).ok


def test_cleanup_plan_writer_is_path_sanitized(tmp_path: Path) -> None:
    _write_bucket_decision(tmp_path)
    report = build_text_evidence_canonical_dirty_cleanup_plan_report(
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
