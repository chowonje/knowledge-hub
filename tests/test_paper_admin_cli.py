from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group


class _StubConfig:
    def __init__(self, tmp_path: Path):
        self.papers_dir = str(tmp_path / "papers")

    def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
        _ = args
        return default


class _StubKhub:
    def __init__(self, db: SQLiteDatabase, tmp_path: Path):
        self._db = db
        self.config = _StubConfig(tmp_path)

    def sqlite_db(self):
        return self._db


def test_paper_list_renders_seeded_rows(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    db.upsert_paper(
        {
            "arxiv_id": "2603.13017",
            "title": "Personalized Agent Memory",
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "summary text " * 4,
            "pdf_path": "/tmp/paper.pdf",
            "text_path": "/tmp/paper.txt",
            "translated_path": "/tmp/paper.ko.md",
            "indexed": 1,
        }
    )
    runner = CliRunner()

    result = runner.invoke(paper_group, ["list"], obj={"khub": _StubKhub(db, tmp_path)})

    assert result.exit_code == 0
    assert "논문 목록" in result.output
    assert "2603.13017" in result.output
    assert "Personalized" in result.output
    assert "Agent Memory" in result.output


def test_paper_list_handles_empty_store(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    runner = CliRunner()

    result = runner.invoke(paper_group, ["list"], obj={"khub": _StubKhub(db, tmp_path)})

    assert result.exit_code == 0
    assert "수집된 논문이 없습니다" in result.output


def test_paper_info_renders_metadata_and_summary(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    db.upsert_paper(
        {
            "arxiv_id": "2603.13017",
            "title": "Personalized Agent Memory",
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "# Summary\n\nThis paper compresses long-running agent sessions into memory cards.",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
            "indexed": 1,
        }
    )
    runner = CliRunner()

    result = runner.invoke(paper_group, ["info", "2603.13017"], obj={"khub": _StubKhub(db, tmp_path)})

    assert result.exit_code == 0
    assert "논문 정보: 2603.13017" in result.output
    assert "https://arxiv.org/abs/2603.13017" in result.output
    assert "Personalized Agent Memory" in result.output
    assert "This paper compresses long-running agent sessions into memory cards." in result.output


def test_paper_review_filters_bad_summaries(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    db.upsert_paper(
        {
            "arxiv_id": "2603.13017",
            "title": "Healthy Paper",
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "good summary body",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
            "indexed": 1,
        }
    )
    db.upsert_paper(
        {
            "arxiv_id": "2603.13018",
            "title": "Needs Rewrite",
            "authors": "B. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "bad summary body",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
            "indexed": 0,
        }
    )

    def _fake_assess_summary_quality(notes: str):
        if "bad" in notes:
            return {"score": 20, "label": "미흡", "color": "yellow", "reasons": ["길이 부족"]}
        return {"score": 90, "label": "우수", "color": "green", "reasons": []}

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._assess_summary_quality",
        _fake_assess_summary_quality,
    )

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        ["review", "--bad-only", "--threshold", "50"],
        obj={"khub": _StubKhub(db, tmp_path)},
    )

    assert result.exit_code == 0
    assert "논문 요약 품질 리뷰" in result.output
    assert "2603.13018" in result.output
    assert "Needs Rewrite" in result.output
    assert "2603.13017" not in result.output
    assert "재요약 가이드" in result.output


def test_paper_review_show_summary_renders_preview(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    db.upsert_paper(
        {
            "arxiv_id": "2603.13019",
            "title": "Preview Paper",
            "authors": "C. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "preview summary line " * 12,
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
            "indexed": 0,
        }
    )

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._assess_summary_quality",
        lambda notes: {"score": 35, "label": "미흡", "color": "yellow", "reasons": ["근거 부족"]},
    )

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        ["review", "--bad-only", "--show-summary"],
        obj={"khub": _StubKhub(db, tmp_path)},
    )

    assert result.exit_code == 0
    assert "요약 미리보기" in result.output
    assert "2603.13019" in result.output
    assert "preview summary line" in result.output


def test_paper_review_bad_only_shows_guidance_and_preview(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    db.upsert_paper(
        {
            "arxiv_id": "2603.13017",
            "title": "Weak Summary Paper",
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "짧은 요약",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
            "indexed": 0,
        }
    )
    runner = CliRunner()

    result = runner.invoke(
        paper_group,
        ["review", "--bad-only", "--show-summary"],
        obj={"khub": _StubKhub(db, tmp_path)},
    )

    assert result.exit_code == 0
    assert "논문 요약 품질 리뷰" in result.output
    assert "Weak Summary Paper" in result.output
    assert "요약 미리보기" in result.output
    assert "재요약 가이드" in result.output
