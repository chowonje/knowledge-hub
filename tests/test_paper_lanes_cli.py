from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands.paper_labs_cmd import paper_labs_group
from knowledge_hub.papers.paper_lanes import seed_lane_metadata


class _StubConfig:
    def __init__(self, *, sqlite_path: str, vault_path: str):
        self.sqlite_path = sqlite_path
        self.vault_path = vault_path


class _StubKhub:
    def __init__(self, db: SQLiteDatabase, *, vault_path: str):
        self._db = db
        self.config = _StubConfig(sqlite_path=db.db_path, vault_path=vault_path)

    def sqlite_db(self):
        return self._db


def _seed(db: SQLiteDatabase) -> None:
    db.upsert_paper(
        {
            "arxiv_id": "2501.12345",
            "title": "TurboQuant for KV Cache Compression",
            "authors": "A. Researcher",
            "year": 2025,
            "field": "AI Systems",
            "importance": 5,
            "notes": "kv cache compression throughput serving",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_paper_memory_card(
        card={
            "memory_id": "pm:2501.12345",
            "paper_id": "2501.12345",
            "source_note_id": "",
            "title": "TurboQuant for KV Cache Compression",
            "paper_core": "KV cache compression for faster serving.",
            "problem_context": "Bandwidth bottleneck during inference.",
            "method_core": "Quantization-aware KV cache packing.",
            "evidence_core": "Improves throughput and memory use.",
            "limitations": "Applies to inference systems with KV cache.",
            "concept_links": ["kv", "compression", "serving"],
            "claim_refs": [],
            "published_at": "2025-01-01",
            "evidence_window": "",
            "search_text": "kv cache compression throughput serving inference",
            "quality_flag": "ok",
            "version": "paper-memory-v1",
        }
    )


def _seed_non_ai(db: SQLiteDatabase) -> None:
    db.upsert_paper(
        {
            "arxiv_id": "math-001",
            "title": "A Theorem on Compact Symplectic Manifolds",
            "authors": "B. Mathematician",
            "year": 2024,
            "field": "Mathematics",
            "importance": 2,
            "notes": "Proof of a topology result for manifolds and curvature bounds.",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )


def test_seed_lane_metadata_leaves_non_ai_paper_unassigned():
    seeded = seed_lane_metadata(
        {
            "title": "A Theorem on Compact Symplectic Manifolds",
            "field": "Mathematics",
            "notes": "Proof of a topology result for manifolds and curvature bounds.",
        }
    )
    assert seeded["primary_lane"] is None
    assert seeded["secondary_tags"] == []


def test_paper_lane_store_roundtrip(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    db.upsert_paper(
        {
            "arxiv_id": "2601.00001",
            "title": "Agent Planning",
            "authors": "A",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
            "primary_lane": "agent",
            "secondary_tags": ["agent", "planning"],
            "lane_review_status": "locked",
        }
    )
    row = db.get_paper("2601.00001")
    assert row is not None
    assert row["primary_lane"] == "agent"
    assert row["secondary_tags"] == ["agent", "planning"]
    assert row["lane_review_status"] == "locked"
    assert row["lane_updated_at"]


def test_paper_lanes_backfill_review_and_sync_hubs(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    _seed_non_ai(db)
    vault_root = tmp_path / "vault"
    (vault_root / "Projects" / "AI" / "AI_Papers").mkdir(parents=True, exist_ok=True)
    khub = _StubKhub(db, vault_path=str(vault_root))
    runner = CliRunner()

    backfill = runner.invoke(paper_labs_group, ["lanes-backfill", "--json"], obj={"khub": khub})
    assert backfill.exit_code == 0
    backfill_payload = json.loads(backfill.output)
    assert backfill_payload["schema"] == "knowledge-hub.paper-lanes.backfill.result.v1"
    assert backfill_payload["updated"] == 1
    assert backfill_payload["unclassified"] == 1
    updated_item = next(item for item in backfill_payload["items"] if item["paperId"] == "2501.12345")
    unclassified_item = next(item for item in backfill_payload["items"] if item["paperId"] == "math-001")
    assert updated_item["primaryLane"] == "memory_inference"
    assert unclassified_item["status"] == "unclassified"
    non_ai = db.get_paper("math-001")
    assert non_ai is not None
    assert non_ai["primary_lane"] is None

    review_csv = tmp_path / "paper_lanes_review.csv"
    review_summary = tmp_path / "paper_lanes_review_summary.json"
    review = runner.invoke(
        paper_labs_group,
        ["lanes-review", "--out", str(review_csv), "--summary-out", str(review_summary), "--json"],
        obj={"khub": khub},
    )
    assert review.exit_code == 0
    review_payload = json.loads(review.output)
    assert review_payload["rowCount"] == 2
    assert review_payload["summary"]["statusCounts"]["seeded"] == 2
    assert review_payload["summary"]["laneCounts"]["memory_inference"] == 1
    assert review_summary.exists()
    assert review_csv.exists()
    review_text = review_csv.read_text(encoding="utf-8")
    assert "primary_lane" in review_text
    assert "memory_inference" in review_text

    lane_review = runner.invoke(
        paper_labs_group,
        ["lanes-review", "--lane", "memory_inference", "--json"],
        obj={"khub": khub},
    )
    assert lane_review.exit_code == 0
    lane_review_payload = json.loads(lane_review.output)
    assert lane_review_payload["laneFilter"] == "memory_inference"
    assert lane_review_payload["rowCount"] == 1

    sync = runner.invoke(paper_labs_group, ["lanes-sync-hubs", "--json"], obj={"khub": khub})
    assert sync.exit_code == 0
    sync_payload = json.loads(sync.output)
    assert sync_payload["writtenCount"] == 6
    lanes_dir = Path(sync_payload["paths"][0]).parent
    files = sorted(path.name for path in lanes_dir.glob("*.md"))
    assert len(files) == 6
    memory_hub = lanes_dir / "Memory Inference Lane.md"
    first = memory_hub.read_text(encoding="utf-8")
    assert "TurboQuant for KV Cache Compression" in first

    sync_again = runner.invoke(paper_labs_group, ["lanes-sync-hubs", "--json"], obj={"khub": khub})
    assert sync_again.exit_code == 0
    second = memory_hub.read_text(encoding="utf-8")
    assert first == second


def test_paper_lanes_backfill_force_clears_seeded_non_ai_and_preserves_locked(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_non_ai(db)
    db.update_paper_lane_metadata(
        arxiv_id="math-001",
        primary_lane="architecture",
        secondary_tags=["attention"],
        lane_review_status="seeded",
    )
    db.upsert_paper(
        {
            "arxiv_id": "locked-agent",
            "title": "Locked Agent Benchmark",
            "authors": "A. Reviewer",
            "year": 2026,
            "field": "AI Systems",
            "importance": 4,
            "notes": "agent benchmark",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
            "primary_lane": "agent",
            "secondary_tags": ["agent", "benchmark"],
            "lane_review_status": "locked",
        }
    )
    vault_root = tmp_path / "vault"
    (vault_root / "Projects" / "AI" / "AI_Papers").mkdir(parents=True, exist_ok=True)
    khub = _StubKhub(db, vault_path=str(vault_root))
    runner = CliRunner()

    result = runner.invoke(paper_labs_group, ["lanes-backfill", "--force", "--json"], obj={"khub": khub})
    assert result.exit_code == 0
    payload = json.loads(result.output)
    cleared_item = next(item for item in payload["items"] if item["paperId"] == "math-001")
    locked_item = next(item for item in payload["items"] if item["paperId"] == "locked-agent")
    assert cleared_item["status"] == "cleared_non_ai"
    assert locked_item["status"] == "skipped_locked"

    non_ai = db.get_paper("math-001")
    assert non_ai is not None
    assert non_ai["primary_lane"] is None
    assert non_ai["secondary_tags"] == []

    locked = db.get_paper("locked-agent")
    assert locked is not None
    assert locked["primary_lane"] == "agent"
    assert locked["lane_review_status"] == "locked"
