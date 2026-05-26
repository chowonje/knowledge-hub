from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.text_evidence_canonical_dirty_bucket_decision import (
    TEXT_EVIDENCE_CANONICAL_DIRTY_BUCKET_DECISION_SCHEMA_ID,
    build_text_evidence_canonical_dirty_bucket_decision_report,
    write_report,
)


def _write_inventory(root: Path) -> None:
    report_path = root / "text_evidence_canonical_dirty_inventory.v1.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "knowledge-hub.paper.text-evidence-canonical-dirty-inventory.v1",
                "status": "ready",
                "dirtyRows": 6,
                "unknownRows": 0,
                "bucketCounts": {
                    "answer_runtime_or_query_stack": 2,
                    "evidence_spine_or_source_contract_stack": 2,
                    "research_objects_side_stack": 1,
                    "workspace_process_record": 1,
                },
            }
        ),
        encoding="utf-8",
    )


def test_bucket_decision_excludes_all_dirty_buckets_from_text_rc(tmp_path: Path) -> None:
    _write_inventory(tmp_path)

    report = build_text_evidence_canonical_dirty_bucket_decision_report(
        reports_root=tmp_path,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["dirtyRows"] == 6
    assert report["bucketRows"] == 4
    assert report["directIncludeRows"] == 0
    assert report["blockRows"] == 0
    assert report["unknownBucketRows"] == 0
    assert report["publicRcDecision"]["decision"] == "do_not_merge_canonical_dirty_checkout_into_text_rc"
    assert report["publicRcDecision"]["requiresCanonicalCleanupBeforeRc"] is True
    assert report["mutationCounters"]["vaultScanRows"] == 0
    assert report["privatePathLeakRows"] == 0
    assert validate_payload(report, TEXT_EVIDENCE_CANONICAL_DIRTY_BUCKET_DECISION_SCHEMA_ID, strict=True).ok


def test_bucket_decision_writer_is_path_sanitized(tmp_path: Path) -> None:
    _write_inventory(tmp_path)
    report = build_text_evidence_canonical_dirty_bucket_decision_report(
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
