from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.text_evidence_canonical_dirty_inventory import (
    TEXT_EVIDENCE_CANONICAL_DIRTY_INVENTORY_SCHEMA_ID,
    build_text_evidence_canonical_dirty_inventory_report,
    write_report,
)


def test_canonical_dirty_inventory_buckets_status_rows_without_file_reads() -> None:
    report = build_text_evidence_canonical_dirty_inventory_report(
        canonical_repo=None,
        status_lines=[
            " M knowledge_hub/ai/answer_verification.py",
            "?? tasks/2026-05-26-local-note.md",
            "?? knowledge_hub/research_objects/planner.py",
            "?? tests/test_provider_hint_shadow_hook.py",
            " M docs/PROJECT_STATE.md",
        ],
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["dirtyRows"] == 5
    assert report["trackedDirtyRows"] == 2
    assert report["untrackedRows"] == 3
    assert report["scope"]["fileContentReadRows"] == 0
    assert report["bucketCounts"]["evidence_spine_or_source_contract_stack"] == 1
    assert report["bucketCounts"]["workspace_process_record"] == 1
    assert report["bucketCounts"]["research_objects_side_stack"] == 1
    assert report["bucketCounts"]["provider_hint_side_stack"] == 1
    assert report["bucketCounts"]["docs_governance_stack"] == 1
    assert report["mutationCounters"]["vaultScanRows"] == 0
    assert report["privatePathLeakRows"] == 0
    assert validate_payload(report, TEXT_EVIDENCE_CANONICAL_DIRTY_INVENTORY_SCHEMA_ID, strict=True).ok


def test_canonical_dirty_inventory_writer_is_path_sanitized(tmp_path: Path) -> None:
    report = build_text_evidence_canonical_dirty_inventory_report(
        canonical_repo=None,
        status_lines=["?? artifacts/local.md"],
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
