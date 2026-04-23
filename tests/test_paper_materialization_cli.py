from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.core.config import Config
from knowledge_hub.core.database import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group
from knowledge_hub.interfaces.cli.commands.paper_shared_runtime import (
    _build_paper_embedding_text,
    _render_structured_summary_notes,
)


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


class _PreviewDownloader:
    def __init__(self, papers_dir: str, *, preview_text: str):
        self.papers_dir = Path(papers_dir)
        self.preview_text = preview_text

    def download_single(self, arxiv_id: str, title: str):
        paper_dir = self.papers_dir / str(arxiv_id)
        paper_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = paper_dir / "paper.pdf"
        text_path = paper_dir / "paper.txt"
        pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
        text_path.write_text(self.preview_text or f"{title}\nPreview", encoding="utf-8")
        return {
            "success": True,
            "pdf": str(pdf_path),
            "text": str(text_path),
            "title": title,
        }


def _structured_summary_payload(*, paper_id: str, title: str, route: str = "local") -> dict:
    return {
        "schema": "knowledge-hub.paper-summary.build.result.v1",
        "status": "ok",
        "paperId": paper_id,
        "paperTitle": title,
        "parserUsed": "raw",
        "fallbackUsed": False,
        "llmRoute": route,
        "warnings": [],
        "summary": {
            "oneLine": f"{title}의 핵심 요약",
            "problem": f"{title}가 다루는 문제",
            "coreIdea": f"{title}의 핵심 아이디어",
            "methodSteps": [f"{title}의 방법"],
            "keyResults": [f"{title}의 결과"],
            "limitations": [f"{title}의 한계"],
        },
    }


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


def test_paper_download_blocks_guarded_source_and_preserves_existing_row(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    canonical_pdf = tmp_path / "canonical-tr.pdf"
    canonical_text = tmp_path / "canonical-tr.txt"
    canonical_pdf.write_bytes(b"%PDF-1.4\n%fake\n")
    canonical_text.write_text(
        "Transformer Reasoning with Memory\nWe study reasoning chains with persistent memory.",
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": "9999.0001",
            "title": "Transformer Reasoning with Memory",
            "authors": "A",
            "year": 2024,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": str(canonical_pdf),
            "text_path": str(canonical_text),
            "translated_path": None,
        }
    )
    db.upsert_paper(
        {
            "arxiv_id": "2501.00001",
            "title": "Transformer Reasoning",
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

    monkeypatch.setattr(
        "knowledge_hub.papers.downloader.PaperDownloader",
        lambda papers_dir: _PreviewDownloader(
            papers_dir,
            preview_text="Chapter 1 Introduction to Statistics and Probability",
        ),
    )

    runner = CliRunner()
    result = runner.invoke(paper_group, ["download", "2501.00001"], obj={"khub": _StubKhub(config)})

    assert result.exit_code == 0
    assert "source guard blocked import" in result.output

    db = SQLiteDatabase(config.sqlite_path)
    paper = db.get_paper("2501.00001")
    db.close()
    assert paper["pdf_path"] in {None, ""}
    assert paper["text_path"] in {None, ""}


def test_paper_add_blocks_unknown_collision_and_keeps_artifacts_empty(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    canonical_pdf = tmp_path / "canonical-tr.pdf"
    canonical_text = tmp_path / "canonical-tr.txt"
    canonical_pdf.write_bytes(b"%PDF-1.4\n%fake\n")
    canonical_text.write_text(
        "Transformer Reasoning with Memory\nWe study reasoning chains with persistent memory.",
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": "9999.0001",
            "title": "Transformer Reasoning with Memory",
            "authors": "A",
            "year": 2024,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": str(canonical_pdf),
            "text_path": str(canonical_text),
            "translated_path": None,
        }
    )
    db.close()

    monkeypatch.setattr(
        "knowledge_hub.papers.url_resolver.resolve_url",
        lambda url: type(
            "ResolvedPaper",
            (),
            {
                "arxiv_id": "Transformer_Reasoning_abcd1234",
                "title": "Transformer Reasoning",
                "authors": "A. Author",
                "year": 2026,
                "citation_count": 3,
                "fields_of_study": ["AI"],
                "abstract": "abstract",
                "source": "custom",
            },
        )(),
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.downloader.PaperDownloader",
        lambda papers_dir: _PreviewDownloader(
            papers_dir,
            preview_text="Chapter 1 Introduction to Statistics and Probability",
        ),
    )

    runner = CliRunner()
    result = runner.invoke(paper_group, ["add", "https://example.com/tr"], obj={"khub": _StubKhub(config)})

    assert result.exit_code == 0
    assert "source guard blocked import" in result.output

    db = SQLiteDatabase(config.sqlite_path)
    paper = db.get_paper("Transformer_Reasoning_abcd1234")
    db.close()
    assert paper is not None
    assert paper["pdf_path"] in {None, ""}
    assert paper["text_path"] in {None, ""}


def test_paper_embed_marks_paper_indexed(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    _seed_paper(config)
    db = SQLiteDatabase(config.sqlite_path)
    db.conn.execute("UPDATE papers SET notes = ? WHERE arxiv_id = ?", ("summary body", "2501.00001"))
    db.conn.commit()
    db.close()

    class _FakeEmbedder:
        def embed_text(self, text: str):
            assert text == "EMBED TEXT"
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
        "knowledge_hub.interfaces.cli.commands.paper_materialization_runtime._build_paper_embedding_text",
        lambda *args, **kwargs: "EMBED TEXT",
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

    class _FakeSummaryService:
        def __init__(self, sqlite_db, config):  # noqa: ANN001
            self.sqlite_db = sqlite_db
            self.config = config

        def load_artifact(self, *, paper_id: str):  # noqa: ARG002
            return {}

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd.StructuredPaperSummaryService",
        _FakeSummaryService,
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._build_public_summary_card",
        lambda khub, paper_id: {"quality": {"score": 10 if paper_id == "2501.00001" else 90}},  # noqa: ARG005
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._run_structured_summary_batch_worker",
        lambda **kwargs: _structured_summary_payload(
            paper_id=kwargs["paper_id"],
            title="Bad Summary Paper" if kwargs["paper_id"] == "2501.00001" else "Healthy Summary Paper",
            route="local",
        ),
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
    assert bad_paper["notes"] == _render_structured_summary_notes(
        _structured_summary_payload(
            paper_id="2501.00001",
            title="Bad Summary Paper",
            route="local",
        )
    )
    assert good_paper["notes"] == "good summary " * 20


def test_paper_summarize_all_records_timeout_checkpoint_and_continues(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    for arxiv_id, title in (
        ("2501.00001", "Slow Summary Paper"),
        ("2501.00002", "Fast Summary Paper"),
    ):
        db.upsert_paper(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": "A",
                "year": 2025,
                "field": "AI",
                "importance": 3,
                "notes": "",
                "pdf_path": None,
                "text_path": None,
                "translated_path": None,
            }
        )
    db.close()

    class _FakeSummaryService:
        def __init__(self, sqlite_db, config):  # noqa: ANN001
            self.sqlite_db = sqlite_db
            self.config = config

        def load_artifact(self, *, paper_id: str):  # noqa: ARG002
            return {}

    def _worker(**kwargs):  # noqa: ANN003
        if kwargs["paper_id"] == "2501.00001":
            raise TimeoutError("summary worker timed out after 1s")
        return _structured_summary_payload(
            paper_id="2501.00002",
            title="Fast Summary Paper",
            route="strong",
        )

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd.StructuredPaperSummaryService",
        _FakeSummaryService,
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._run_structured_summary_batch_worker",
        _worker,
    )

    checkpoint = tmp_path / "summaries-checkpoint.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        [
            "summarize-all",
            "--resummary",
            "--limit",
            "2",
            "--paper-timeout-sec",
            "1",
            "--checkpoint-file",
            str(checkpoint),
        ],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0
    assert "TIMEOUT" in result.output
    assert "1/2편 요약 완료" in result.output
    assert checkpoint.exists()

    records = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [record["status"] for record in records] == ["timeout", "ok"]
    assert records[0]["paperId"] == "2501.00001"
    assert records[1]["paperId"] == "2501.00002"

    db = SQLiteDatabase(config.sqlite_path)
    slow_paper = db.get_paper("2501.00001")
    fast_paper = db.get_paper("2501.00002")
    db.close()
    assert slow_paper["notes"] == ""
    assert fast_paper["notes"] == _render_structured_summary_notes(
        _structured_summary_payload(
            paper_id="2501.00002",
            title="Fast Summary Paper",
            route="strong",
        )
    )


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
            assert texts == ["EMBED 2501.00001", "EMBED 2501.00002"]
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
        "knowledge_hub.interfaces.cli.commands.paper_materialization_runtime._build_paper_embedding_text",
        lambda _sqlite_db, *, paper, config: f"EMBED {paper['arxiv_id']}",
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


def test_paper_embed_all_recovers_failed_batch_items_individually(monkeypatch, tmp_path: Path):
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
            return [None, [0.2, 0.3]]

        def embed_text(self, text: str):
            if text == "EMBED 2501.00001":
                return [0.1, 0.2]
            raise RuntimeError("unexpected fallback call")

    class _FakeVectorDb:
        def __init__(self):
            self.calls = []

        def add_documents(self, documents, embeddings, metadatas, ids):  # noqa: ANN001
            self.calls.append((documents, embeddings, metadatas, ids))

    fake_vector_db = _FakeVectorDb()
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._build_embedder",
        lambda *args, **kwargs: _FakeEmbedder(),
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_materialization_runtime._build_paper_embedding_text",
        lambda _sqlite_db, *, paper, config: f"EMBED {paper['arxiv_id']}",
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._vector_db",
        lambda *args, **kwargs: fake_vector_db,
    )

    runner = CliRunner()
    result = runner.invoke(paper_group, ["embed-all"], obj={"khub": _StubKhub(config)})

    assert result.exit_code == 0
    assert "2/2편 인덱싱 완료" in result.output
    assert len(fake_vector_db.calls) == 1

    db = SQLiteDatabase(config.sqlite_path)
    papers = db.list_papers(limit=10)
    db.close()
    assert all(int(paper["indexed"]) == 1 for paper in papers)


def test_build_paper_embedding_text_prefers_structured_summary_and_memory(tmp_path: Path):
    config = _config(tmp_path)
    _seed_paper(config)
    db = SQLiteDatabase(config.sqlite_path)
    db.conn.execute("UPDATE papers SET notes = ? WHERE arxiv_id = ?", ("raw notes should be fallback only", "2501.00001"))
    db.upsert_paper_memory_card(
        card={
            "memory_id": "paper-memory:2501.00001:test",
            "paper_id": "2501.00001",
            "title": "Transformer Test Paper",
            "paper_core": "memory paper core",
            "problem_context": "memory problem context",
            "method_core": "memory method core",
            "evidence_core": "memory evidence core",
            "limitations": "memory limitations",
            "concept_links": ["attention"],
            "claim_refs": [],
            "search_text": "memory search text",
            "quality_flag": "ok",
        }
    )
    summary_dir = Path(config.papers_dir) / "summaries" / "2501.00001"
    summary_dir.mkdir(parents=True, exist_ok=True)
    payload = _structured_summary_payload(paper_id="2501.00001", title="Transformer Test Paper", route="strong")
    (summary_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    paper = db.get_paper("2501.00001")
    text = _build_paper_embedding_text(db, paper=paper, config=config, keywords=["transformer", "attention"])
    db.close()

    assert "Structured Summary" in text
    assert "Transformer Test Paper의 핵심 요약" in text
    assert "Paper Memory" in text
    assert "memory method core" in text
    assert "Keywords: transformer, attention" in text
