from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.core.config import Config
from knowledge_hub.core.database import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group


class _StubKhub:
    def __init__(self, config: Config):
        self.config = config


def _config(tmp_path: Path) -> Config:
    config = Config()
    config.set_nested("storage", "sqlite", str(tmp_path / "knowledge.db"))
    config.set_nested("storage", "papers_dir", str(tmp_path / "papers"))
    config.set_nested("storage", "vector_db", str(tmp_path / "vector_db"))
    return config


def _seed_paper(config: Config) -> None:
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_paper(
        {
            "arxiv_id": "2501.00001",
            "title": "Transformer Test Paper",
            "authors": "A",
            "year": 2025,
            "field": "AI",
            "importance": 3,
            "notes": "seed notes",
            "pdf_path": None,
            "text_path": None,
            "translated_path": None,
        }
    )
    db.close()


def test_paper_download_updates_existing_record(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    _seed_paper(config)

    class _FakeDownloader:
        def __init__(self, papers_dir: str):
            self.papers_dir = Path(papers_dir)

        def download_single(self, arxiv_id: str, title: str):
            pdf_path = self.papers_dir / f"{arxiv_id}.pdf"
            text_path = self.papers_dir / f"{arxiv_id}.txt"
            return {
                "success": True,
                "pdf": str(pdf_path),
                "text": str(text_path),
                "title": title,
            }

    monkeypatch.setattr("knowledge_hub.papers.downloader.PaperDownloader", _FakeDownloader)

    runner = CliRunner()
    result = runner.invoke(paper_group, ["download", "2501.00001"], obj={"khub": _StubKhub(config)})

    assert result.exit_code == 0
    assert "다운로드 완료" in result.output

    db = SQLiteDatabase(config.sqlite_path)
    paper = db.get_paper("2501.00001")
    db.close()

    assert paper["title"] == "Transformer Test Paper"
    assert paper["pdf_path"].endswith("2501.00001.pdf")
    assert paper["text_path"].endswith("2501.00001.txt")
    assert paper["notes"] == "seed notes"


def test_paper_embed_marks_paper_indexed(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    _seed_paper(config)
    db = SQLiteDatabase(config.sqlite_path)
    db.conn.execute("UPDATE papers SET notes = ? WHERE arxiv_id = ?", ("summary body", "2501.00001"))
    db.conn.commit()
    db.close()

    class _FakeEmbedder:
        def embed_text(self, text: str):
            assert "Transformer Test Paper" in text
            return [0.1, 0.2, 0.3]

    class _FakeVectorDb:
        def __init__(self):
            self.docs = []

        def add_documents(self, documents, embeddings, metadatas, ids):  # noqa: ANN001
            self.docs.append((documents, embeddings, metadatas, ids))

        def count(self):
            return len(self.docs)

    fake_vector_db = _FakeVectorDb()
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._build_embedder",
        lambda *args, **kwargs: _FakeEmbedder(),
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._vector_db",
        lambda *args, **kwargs: fake_vector_db,
    )

    runner = CliRunner()
    result = runner.invoke(paper_group, ["embed", "2501.00001"], obj={"khub": _StubKhub(config)})

    assert result.exit_code == 0
    assert "임베딩 완료" in result.output

    db = SQLiteDatabase(config.sqlite_path)
    paper = db.get_paper("2501.00001")
    db.close()
    assert int(paper["indexed"]) == 1
    assert fake_vector_db.count() == 1


def test_paper_translate_all_updates_missing_translations(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    text_path = tmp_path / "paper.txt"
    text_path.write_text("Transformer uses attention. " * 40, encoding="utf-8")
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_paper(
        {
            "arxiv_id": "2501.00001",
            "title": "Translate Me",
            "authors": "A",
            "year": 2025,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": None,
            "text_path": str(text_path),
            "translated_path": None,
        }
    )
    db.close()

    class _FakeLLM:
        def translate(self, text: str, source_lang: str = "en", target_lang: str = "ko"):  # noqa: ARG002
            return "번역 결과"

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._build_llm",
        lambda *args, **kwargs: _FakeLLM(),
    )

    runner = CliRunner()
    result = runner.invoke(paper_group, ["translate-all"], obj={"khub": _StubKhub(config)})

    assert result.exit_code == 0
    assert "1/1편 번역 완료" in result.output

    db = SQLiteDatabase(config.sqlite_path)
    paper = db.get_paper("2501.00001")
    db.close()
    assert paper["translated_path"].endswith("Translate Me_translated.md")


def test_paper_summarize_all_bad_only_filters_scores(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    text_path = tmp_path / "paper.txt"
    text_path.write_text("Transformer uses attention. " * 40, encoding="utf-8")
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_paper(
        {
            "arxiv_id": "2501.00001",
            "title": "Bad Summary Paper",
            "authors": "A",
            "year": 2025,
            "field": "AI",
            "importance": 3,
            "notes": "bad summary",
            "pdf_path": None,
            "text_path": str(text_path),
            "translated_path": None,
        }
    )
    db.upsert_paper(
        {
            "arxiv_id": "2501.00002",
            "title": "Healthy Summary Paper",
            "authors": "A",
            "year": 2025,
            "field": "AI",
            "importance": 3,
            "notes": "good summary " * 20,
            "pdf_path": None,
            "text_path": str(text_path),
            "translated_path": None,
        }
    )
    db.close()

    class _FakeLLM:
        def summarize(self, text: str, language: str = "ko", max_sentences: int = 5):  # noqa: ARG002
            return "요약"

        def summarize_paper(self, text: str, title: str = "", language: str = "ko"):  # noqa: ARG002
            return f"요약:{title}"

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._build_llm",
        lambda *args, **kwargs: _FakeLLM(),
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._assess_summary_quality",
        lambda notes: {"score": 10, "label": "미흡", "color": "yellow", "reasons": ["짧음"]}
        if "bad" in notes
        else {"score": 90, "label": "우수", "color": "green", "reasons": []},
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._update_obsidian_summary",
        lambda *args, **kwargs: None,
    )

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        ["summarize-all", "--bad-only", "--threshold", "50"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0
    assert "품질 점수 50점 미만 논문 필터링" in result.output
    assert "1편" in result.output

    db = SQLiteDatabase(config.sqlite_path)
    bad_paper = db.get_paper("2501.00001")
    good_paper = db.get_paper("2501.00002")
    db.close()
    assert bad_paper["notes"] == "요약:Bad Summary Paper"
    assert good_paper["notes"] == "good summary " * 20


def test_paper_embed_all_indexes_unindexed_batch(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    for arxiv_id in ("2501.00001", "2501.00002"):
        db.upsert_paper(
            {
                "arxiv_id": arxiv_id,
                "title": f"Paper {arxiv_id}",
                "authors": "A",
                "year": 2025,
                "field": "AI",
                "importance": 3,
                "notes": "summary",
                "pdf_path": None,
                "text_path": None,
                "translated_path": None,
                "indexed": 0,
            }
        )
    db.close()

    class _FakeEmbedder:
        def embed_batch(self, texts, show_progress=False):  # noqa: ANN001
            assert show_progress is False
            return [[0.1, 0.2] for _ in texts]

    class _FakeVectorDb:
        def add_documents(self, documents, embeddings, metadatas, ids):  # noqa: ANN001
            assert len(documents) == 2
            assert len(embeddings) == 2
            assert len(metadatas) == 2
            assert len(ids) == 2

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._build_embedder",
        lambda *args, **kwargs: _FakeEmbedder(),
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._vector_db",
        lambda *args, **kwargs: _FakeVectorDb(),
    )

    runner = CliRunner()
    result = runner.invoke(paper_group, ["embed-all"], obj={"khub": _StubKhub(config)})

    assert result.exit_code == 0
    assert "2/2편 인덱싱 완료" in result.output

    db = SQLiteDatabase(config.sqlite_path)
    papers = db.list_papers(limit=10)
    db.close()
    assert all(int(paper["indexed"]) == 1 for paper in papers)
