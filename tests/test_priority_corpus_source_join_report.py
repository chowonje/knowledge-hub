"""Tests for priority corpus source join report artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from knowledge_hub.core.schema_validator import validate_payload


def _load_builder() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    path = root / "eval/knowledgeos/scripts/build_priority_corpus_source_join_report.py"
    spec = importlib.util.spec_from_file_location("priority_corpus_source_join_report_builder", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_join_builder_excludes_already_manifest_sources_from_allowlist(tmp_path: Path) -> None:
    builder = _load_builder()
    hash_a = "sha256:" + ("a" * 64)
    hash_b = "sha256:" + ("b" * 64)
    ledger_path = tmp_path / "ledger.json"
    manifest_path = tmp_path / "manifest.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema": "knowledge-hub.priority-corpus-candidate-ledger.v1",
                "generated_at": "2026-05-21",
                "scope_note": "test fixture",
                "source_pools": {},
                "status_vocabulary": {},
                "counts": {
                    "candidate_rows": 2,
                    "in_manifest": 1,
                    "not_in_manifest": 1,
                    "source_verification_pending": 2,
                },
                "candidates": [
                    {
                        "source_id": "paper-a",
                        "title": "Registered paper",
                        "year": 2026,
                        "provider": "fixture",
                        "candidate_tier": "eval_critical",
                        "current_manifest_status": "in_manifest",
                        "source_artifact_status": "join_pending",
                    },
                    {
                        "source_id": "paper-b",
                        "title": "Expansion paper",
                        "year": 2026,
                        "provider": "fixture",
                        "candidate_tier": "recent_ai",
                        "current_manifest_status": "not_in_manifest",
                        "source_artifact_status": "join_pending",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "artifactId": "artifact-a",
                        "sourceIds": ["paper-a"],
                        "expectedFilename": "paper-a.pdf",
                        "expectedSourceContentHash": hash_a,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    inventory_payload = {
        "schema": "knowledge-hub.paper.corpus-source-artifact-inventory.v1",
        "generatedAt": "2026-05-21T00:00:00Z",
        "counts": {"inventoryRows": 2},
        "items": [
            {
                "sourceId": "paper-a",
                "filename": "paper-a.pdf",
                "corpusLocationRef": "papers_dir/paper-a.pdf",
                "sourceType": "pdf",
                "byteLength": 10,
                "sha256": hash_a,
            },
            {
                "sourceId": "paper-b",
                "filename": "paper-b.pdf",
                "corpusLocationRef": "papers_dir/paper-b.pdf",
                "sourceType": "pdf",
                "byteLength": 20,
                "sha256": hash_b,
            },
        ],
    }

    report, allowlist = builder.build_join_report(
        ledger_path=ledger_path,
        manifest_path=manifest_path,
        inventory_payload=inventory_payload,
        papers_dir=tmp_path / "papers",
    )

    assert validate_payload(report, builder.JOIN_SCHEMA, strict=True).errors == []
    assert validate_payload(allowlist, builder.ALLOWLIST_SCHEMA, strict=True).errors == []
    assert report["counts"]["already_in_manifest_count"] == 1
    assert report["counts"]["expansion_allowlist_count"] == 1
    assert [row["source_id"] for row in allowlist["allowlist"]] == ["paper-b"]
    assert all(
        row["current_manifest_status"] == "not_in_manifest"
        for row in report["rows"]
        if row["expansion_allowlist_eligible"]
    )
