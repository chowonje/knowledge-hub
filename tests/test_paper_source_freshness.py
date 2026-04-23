from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.application.paper_source_freshness import audit_paper_source_freshness
from knowledge_hub.core.config import Config
from knowledge_hub.core.database import SQLiteDatabase
from knowledge_hub.document_memory import DocumentMemoryBuilder
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group
from knowledge_hub.papers.source_text import source_hash_for_path


class _StubKhub:
    def __init__(self, config: Config):
        self.config = config


def _config(tmp_path: Path) -> Config:
    config = Config()
    config.set_nested("storage", "sqlite", str(tmp_path / "knowledge.db"))
    config.set_nested("storage", "papers_dir", str(tmp_path / "papers"))
    config.set_nested("storage", "vector_db", str(tmp_path / "vector_db"))
    return config


def _seed_paper(db: SQLiteDatabase, *, paper_id: str, text_path: Path) -> None:
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Freshness Paper",
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "",
            "pdf_path": "",
            "text_path": str(text_path),
            "translated_path": "",
        }
    )


def test_document_memory_paper_source_hash_uses_current_text_file(tmp_path: Path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2604.23001"
    text_path = tmp_path / "paper.txt"
    text_path.write_text("Abstract This paper studies source freshness for grounded retrieval.", encoding="utf-8")
    _seed_paper(db, paper_id=paper_id, text_path=text_path)

    rows = DocumentMemoryBuilder(db).build_and_store_paper(paper_id=paper_id)

    assert rows
    expected_hash = source_hash_for_path(str(text_path))
    assert rows[0]["source_content_hash"] == expected_hash
    assert db.get_document_memory_summary(f"paper:{paper_id}")["source_content_hash"] == expected_hash


def test_paper_source_freshness_audit_marks_stale_only_with_apply(tmp_path: Path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2604.23002"
    text_path = tmp_path / "paper.txt"
    text_path.write_text("Abstract Version one source text for freshness audit.", encoding="utf-8")
    _seed_paper(db, paper_id=paper_id, text_path=text_path)
    DocumentMemoryBuilder(db).build_and_store_paper(paper_id=paper_id)

    assert audit_paper_source_freshness(db, paper_ids=[paper_id])["counts"]["fresh"] == 1

    text_path.write_text("Abstract Version two source text for freshness audit.", encoding="utf-8")
    dry_run = audit_paper_source_freshness(db, paper_ids=[paper_id])

    assert dry_run["dryRun"] is True
    assert dry_run["counts"]["staleCandidate"] == 1
    assert db.get_document_memory_summary(f"paper:{paper_id}")["stale"] is False

    applied = audit_paper_source_freshness(db, paper_ids=[paper_id], apply=True)

    assert applied["applied"] is True
    assert applied["markedStaleCount"] >= 1
    summary = db.get_document_memory_summary(f"paper:{paper_id}")
    assert summary["stale"] is True
    assert summary["stale_reason"] == "source_content_hash_changed"
    assert db.search_document_memory_units("Version one", limit=5) == []
    assert db.search_document_memory_units("Version one", limit=5, include_stale=True)


def test_paper_source_freshness_cli_json_preview(tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    paper_id = "2604.23003"
    text_path = tmp_path / "paper.txt"
    text_path.write_text("Abstract Initial source text for CLI audit.", encoding="utf-8")
    _seed_paper(db, paper_id=paper_id, text_path=text_path)
    DocumentMemoryBuilder(db).build_and_store_paper(paper_id=paper_id)
    text_path.write_text("Abstract Changed source text for CLI audit.", encoding="utf-8")
    db.close()

    result = CliRunner().invoke(
        paper_group,
        ["source-freshness", "--paper-id", paper_id, "--json"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.paper-source-freshness.result.v1"
    assert payload["dryRun"] is True
    assert payload["counts"]["staleCandidate"] == 1
    assert payload["sampleItems"][0]["paperId"] == paper_id
