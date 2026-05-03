from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands import paper_import_support as import_module
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group


class _StubConfig:
    def __init__(self, tmp_path: Path):
        self.papers_dir = str(tmp_path / "papers")
        self.embedding_provider = "stub"
        self.embedding_model = "stub-embed"
        self.vector_db_path = str(tmp_path / "vector")
        self.collection_name = "knowledge-hub-test"

    def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
        return default


class _StubKhub:
    def __init__(self, db: SQLiteDatabase, tmp_path: Path):
        self._db = db
        self.config = _StubConfig(tmp_path)

    def sqlite_db(self):
        return self._db


class _FakeEmbedder:
    def __init__(self, *, fail: bool = False):
        self.fail = fail

    def embed_text(self, text: str):
        if self.fail:
            raise RuntimeError("embed boom")
        _ = text
        return [0.1, 0.2, 0.3]


class _FakeVectorDB:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def add_documents(self, *, documents, embeddings, metadatas, ids):  # noqa: ANN001
        self.calls.append(
            {
                "documents": list(documents),
                "embeddings": list(embeddings),
                "metadatas": list(metadatas),
                "ids": list(ids),
            }
        )


class _FakeDownloader:
    def __init__(self, papers_dir: str, *, should_raise: bool = False):
        self.papers_dir = Path(papers_dir)
        self.should_raise = should_raise

    def download_single(self, arxiv_id, title):  # noqa: ANN001
        if self.should_raise:
            raise AssertionError("download should have been skipped")
        paper_dir = self.papers_dir / str(arxiv_id)
        paper_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = paper_dir / "paper.pdf"
        text_path = paper_dir / "paper.txt"
        pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
        text_path.write_text(f"{title}\nFull paper text", encoding="utf-8")
        return {"success": True, "pdf": str(pdf_path), "text": str(text_path)}


class _PreviewDownloader:
    def __init__(self, papers_dir: str, *, preview_text: str):
        self.papers_dir = Path(papers_dir)
        self.preview_text = preview_text

    def download_single(self, arxiv_id, title):  # noqa: ANN001
        paper_dir = self.papers_dir / str(arxiv_id)
        paper_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = paper_dir / "paper.pdf"
        text_path = paper_dir / "paper.txt"
        pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
        text_path.write_text(self.preview_text or f"{title}\nPreview", encoding="utf-8")
        return {"success": True, "pdf": str(pdf_path), "text": str(text_path)}


class _FakeResponse:
    def __init__(self, content: bytes, *, content_type: str = "application/pdf", status_code: int = 200):
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    content = "\n".join([",".join(["bucket_ko", "title", "year", "theme_ko", "priority", "why_selected", "source_url"]), *[",".join(row) for row in rows]]) + "\n"
    path.write_text(content, encoding="utf-8-sig")


def test_paper_import_csv_help_exposes_command():
    runner = CliRunner()
    result = runner.invoke(paper_group, ["import-csv", "--help"])
    assert result.exit_code == 0
    assert "--csv" in result.output
    assert "--min-priority" in result.output
    assert "--document-memory-parser" in result.output


def test_paper_import_csv_filters_priority_and_records_manifest(monkeypatch, tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    khub = _StubKhub(db, tmp_path)
    runner = CliRunner()
    csv_path = tmp_path / "papers.csv"
    _write_csv(
        csv_path,
        [
            ["최신/이슈", "Test Paper", "2026", "에이전트", "5", "selected", "https://example.com/paper-1"],
            ["최신/이슈", "Low Priority", "2026", "에이전트", "4", "filtered", "https://example.com/paper-2"],
            ["최신/이슈", "Missing URL", "2026", "에이전트", "5", "invalid", ""],
        ],
    )
    vector_db = _FakeVectorDB()

    monkeypatch.setattr(
        import_module,
        "resolve_url",
        lambda url: SimpleNamespace(
            arxiv_id="2603.99991",
            title="Test Paper",
            authors="A. Author",
            year=2026,
            citation_count=5,
            fields_of_study=["AI"],
        )
        if url.endswith("paper-1")
        else None,
    )
    monkeypatch.setattr(import_module, "_build_embedder", lambda config, khub=None: _FakeEmbedder())
    monkeypatch.setattr(import_module, "_vector_db", lambda config, khub=None: vector_db)
    monkeypatch.setattr(import_module, "PaperDownloader", lambda papers_dir: _FakeDownloader(papers_dir))
    monkeypatch.setattr(import_module.PaperMemoryBuilder, "build_and_store", lambda self, paper_id: {"memory_id": f"pm:{paper_id}"})
    monkeypatch.setattr(import_module.DocumentMemoryBuilder, "build_and_store_paper", lambda self, paper_id, **kwargs: [{"document_id": f"paper:{paper_id}"}])

    result = runner.invoke(paper_group, ["import-csv", "--csv", str(csv_path), "--json"], obj={"khub": khub})

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["counts"]["selected"] == 1
    assert payload["counts"]["completed"] == 1
    assert payload["counts"]["filtered"] == 1
    assert payload["counts"]["invalid"] == 1
    assert db.get_paper("2603.99991")["indexed"] == 1
    assert vector_db.calls[0]["ids"] == ["paper_2603.99991_0"]

    manifest = json.loads(Path(payload["manifestPath"]).read_text(encoding="utf-8"))
    statuses = {item["title"]: item["status"] for item in manifest["entries"]}
    assert statuses["Test Paper"] == "completed"
    assert statuses["Low Priority"] == "filtered"
    assert statuses["Missing URL"] == "skipped"


def test_paper_import_csv_falls_back_to_arxiv_id_when_resolver_fails(monkeypatch, tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    khub = _StubKhub(db, tmp_path)
    runner = CliRunner()
    csv_path = tmp_path / "papers.csv"
    _write_csv(
        csv_path,
        [["최신/이슈", "Fallback Arxiv Paper", "2026", "에이전트", "5", "selected", "https://arxiv.org/abs/2603.05706"]],
    )
    vector_db = _FakeVectorDB()

    monkeypatch.setattr(import_module, "resolve_url", lambda url: None)
    monkeypatch.setattr(import_module, "_build_embedder", lambda config, khub=None: _FakeEmbedder())
    monkeypatch.setattr(import_module, "_vector_db", lambda config, khub=None: vector_db)
    monkeypatch.setattr(import_module, "PaperDownloader", lambda papers_dir: _FakeDownloader(papers_dir))
    monkeypatch.setattr(import_module.PaperMemoryBuilder, "build_and_store", lambda self, paper_id: {"memory_id": f"pm:{paper_id}"})

    result = runner.invoke(
        paper_group,
        ["import-csv", "--csv", str(csv_path), "--steps", "register,download,embed,paper-memory", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    item = payload["items"][0]
    assert item["resolvedPaperId"] == "2603.05706"
    assert item["completedSteps"] == ["register", "download", "embed", "paper-memory"]
    assert db.get_paper("2603.05706")["indexed"] == 1
    assert vector_db.calls[0]["ids"] == ["paper_2603.05706_0"]


def test_paper_import_csv_text_summary_uses_skipped_label(monkeypatch, tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    khub = _StubKhub(db, tmp_path)
    runner = CliRunner()
    csv_path = tmp_path / "papers.csv"
    _write_csv(
        csv_path,
        [["최신/이슈", "Existing Paper", "2026", "에이전트", "5", "selected", "https://example.com/paper-1"]],
    )
    pdf_path = tmp_path / "seed.pdf"
    text_path = tmp_path / "seed.txt"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    text_path.write_text("Existing full text", encoding="utf-8")
    db.upsert_paper(
        {
            "arxiv_id": "2603.99990",
            "title": "Existing Paper",
            "authors": "A. Author",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "summary text",
            "pdf_path": str(pdf_path),
            "text_path": str(text_path),
            "translated_path": "",
            "indexed": 1,
        }
    )
    monkeypatch.setattr(
        import_module,
        "resolve_url",
        lambda url: SimpleNamespace(
            arxiv_id="2603.99990",
            title="Existing Paper",
            authors="A. Author",
            year=2026,
            citation_count=5,
            fields_of_study=["AI"],
        ),
    )
    monkeypatch.setattr(import_module, "PaperDownloader", lambda papers_dir: _FakeDownloader(papers_dir, should_raise=True))
    monkeypatch.setattr(import_module.PaperMemoryBuilder, "build_and_store", lambda self, paper_id: {"memory_id": f"pm:{paper_id}"})
    monkeypatch.setattr(import_module.DocumentMemoryBuilder, "build_and_store_paper", lambda self, paper_id, **kwargs: [{"document_id": f"paper:{paper_id}"}])

    result = runner.invoke(
        paper_group,
        ["import-csv", "--csv", str(csv_path), "--steps", "register,download"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    assert "skipped=1" in result.output


def test_paper_import_csv_existing_paper_skips_download_and_runs_missing_steps(monkeypatch, tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    khub = _StubKhub(db, tmp_path)
    runner = CliRunner()
    csv_path = tmp_path / "papers.csv"
    pdf_path = tmp_path / "seed.pdf"
    text_path = tmp_path / "seed.txt"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    text_path.write_text("Existing full text", encoding="utf-8")
    db.upsert_paper(
        {
            "arxiv_id": "2603.99992",
            "title": "Existing Paper",
            "authors": "A. Author",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "summary text",
            "pdf_path": str(pdf_path),
            "text_path": str(text_path),
            "translated_path": "",
        }
    )
    _write_csv(
        csv_path,
        [["최신/이슈", "Existing Paper", "2026", "에이전트", "5", "selected", "https://example.com/paper-1"]],
    )
    vector_db = _FakeVectorDB()

    monkeypatch.setattr(
        import_module,
        "resolve_url",
        lambda url: SimpleNamespace(
            arxiv_id="2603.99992",
            title="Existing Paper",
            authors="A. Author",
            year=2026,
            citation_count=5,
            fields_of_study=["AI"],
        ),
    )
    monkeypatch.setattr(import_module, "_build_embedder", lambda config, khub=None: _FakeEmbedder())
    monkeypatch.setattr(import_module, "_vector_db", lambda config, khub=None: vector_db)
    monkeypatch.setattr(import_module, "PaperDownloader", lambda papers_dir: _FakeDownloader(papers_dir, should_raise=True))
    monkeypatch.setattr(import_module.PaperMemoryBuilder, "build_and_store", lambda self, paper_id: {"memory_id": f"pm:{paper_id}"})
    monkeypatch.setattr(import_module.DocumentMemoryBuilder, "build_and_store_paper", lambda self, paper_id, **kwargs: [{"document_id": f"paper:{paper_id}"}])

    result = runner.invoke(paper_group, ["import-csv", "--csv", str(csv_path), "--json"], obj={"khub": khub})

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["counts"]["completed"] == 1
    item = payload["items"][0]
    assert item["resolvedPaperId"] == "2603.99992"
    assert "download" in item["completedSteps"]
    assert "download" not in item["executedSteps"]
    assert "embed" in item["executedSteps"]
    assert db.get_paper("2603.99992")["indexed"] == 1


def test_paper_import_csv_resumes_from_failed_step(monkeypatch, tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    khub = _StubKhub(db, tmp_path)
    runner = CliRunner()
    csv_path = tmp_path / "papers.csv"
    _write_csv(
        csv_path,
        [["최신/이슈", "Resume Paper", "2026", "에이전트", "5", "selected", "https://example.com/paper-1"]],
    )
    vector_db = _FakeVectorDB()

    monkeypatch.setattr(
        import_module,
        "resolve_url",
        lambda url: SimpleNamespace(
            arxiv_id="2603.99993",
            title="Resume Paper",
            authors="A. Author",
            year=2026,
            citation_count=5,
            fields_of_study=["AI"],
        ),
    )
    monkeypatch.setattr(import_module, "_vector_db", lambda config, khub=None: vector_db)
    monkeypatch.setattr(import_module, "PaperDownloader", lambda papers_dir: _FakeDownloader(papers_dir))
    monkeypatch.setattr(import_module.PaperMemoryBuilder, "build_and_store", lambda self, paper_id: {"memory_id": f"pm:{paper_id}"})
    monkeypatch.setattr(import_module.DocumentMemoryBuilder, "build_and_store_paper", lambda self, paper_id, **kwargs: [{"document_id": f"paper:{paper_id}"}])

    first_embedder = _FakeEmbedder(fail=True)
    monkeypatch.setattr(import_module, "_build_embedder", lambda config, khub=None: first_embedder)
    first = runner.invoke(paper_group, ["import-csv", "--csv", str(csv_path), "--json"], obj={"khub": khub})
    assert first.exit_code == 0, first.output
    first_payload = json.loads(first.output)
    assert first_payload["status"] == "failed"
    first_item = first_payload["items"][0]
    assert first_item["failedStep"] == "embed"
    assert first_item["completedSteps"] == ["register", "download"]

    monkeypatch.setattr(import_module, "PaperDownloader", lambda papers_dir: _FakeDownloader(papers_dir, should_raise=True))
    monkeypatch.setattr(import_module, "_build_embedder", lambda config, khub=None: _FakeEmbedder())
    second = runner.invoke(paper_group, ["import-csv", "--csv", str(csv_path), "--json"], obj={"khub": khub})
    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.output)
    assert second_payload["status"] == "ok"
    second_item = second_payload["items"][0]
    assert second_item["completedSteps"] == ["register", "download", "embed", "paper-memory", "document-memory"]
    assert second_item["executedSteps"] == ["embed", "paper-memory", "document-memory"]


def test_paper_import_csv_rebuilds_document_memory_when_parser_changes(monkeypatch, tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    khub = _StubKhub(db, tmp_path)
    runner = CliRunner()
    csv_path = tmp_path / "papers.csv"
    _write_csv(
        csv_path,
        [["최신/이슈", "Parser Paper", "2026", "에이전트", "5", "selected", "https://example.com/paper-1"]],
    )
    vector_db = _FakeVectorDB()
    summary_state = {"ready": False}
    parser_calls: list[str] = []

    monkeypatch.setattr(
        import_module,
        "resolve_url",
        lambda url: SimpleNamespace(
            arxiv_id="2603.99994",
            title="Parser Paper",
            authors="A. Author",
            year=2026,
            citation_count=5,
            fields_of_study=["AI"],
        ),
    )
    monkeypatch.setattr(import_module, "_build_embedder", lambda config, khub=None: _FakeEmbedder())
    monkeypatch.setattr(import_module, "_vector_db", lambda config, khub=None: vector_db)
    monkeypatch.setattr(import_module, "PaperDownloader", lambda papers_dir: _FakeDownloader(papers_dir))
    monkeypatch.setattr(import_module.PaperMemoryBuilder, "build_and_store", lambda self, paper_id: {"memory_id": f"pm:{paper_id}"})
    monkeypatch.setattr(
        db,
        "get_document_memory_summary",
        lambda document_id: {"document_id": document_id} if summary_state["ready"] else None,
    )

    def _build_document_memory(self, paper_id, **kwargs):  # noqa: ANN001
        parser_calls.append(str(kwargs.get("paper_parser")))
        summary_state["ready"] = True
        return [{"document_id": f"paper:{paper_id}"}]

    monkeypatch.setattr(import_module.DocumentMemoryBuilder, "build_and_store_paper", _build_document_memory)

    first = runner.invoke(paper_group, ["import-csv", "--csv", str(csv_path), "--json"], obj={"khub": khub})
    assert first.exit_code == 0, first.output
    first_payload = json.loads(first.output)
    assert first_payload["status"] == "ok"
    assert parser_calls == ["raw"]

    second = runner.invoke(
        paper_group,
        ["import-csv", "--csv", str(csv_path), "--document-memory-parser", "opendataloader", "--json"],
        obj={"khub": khub},
    )
    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.output)
    second_item = second_payload["items"][0]
    assert second_item["status"] == "completed"
    assert second_item["executedSteps"] == ["document-memory"]
    assert second_item["artifacts"]["documentMemoryParser"] == "opendataloader"
    assert parser_calls == ["raw", "opendataloader"]


def test_paper_import_csv_uses_summary_provenance_when_manifest_parser_missing(monkeypatch, tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    khub = _StubKhub(db, tmp_path)
    runner = CliRunner()
    csv_path = tmp_path / "papers.csv"
    _write_csv(
        csv_path,
        [["최신/이슈", "Provenance Paper", "2026", "에이전트", "5", "selected", "https://example.com/paper-1"]],
    )
    vector_db = _FakeVectorDB()
    summary_state = {"ready": False, "parser": ""}
    parser_calls: list[str] = []

    monkeypatch.setattr(
        import_module,
        "resolve_url",
        lambda url: SimpleNamespace(
            arxiv_id="2603.99995",
            title="Provenance Paper",
            authors="A. Author",
            year=2026,
            citation_count=5,
            fields_of_study=["AI"],
        ),
    )
    monkeypatch.setattr(import_module, "_build_embedder", lambda config, khub=None: _FakeEmbedder())
    monkeypatch.setattr(import_module, "_vector_db", lambda config, khub=None: vector_db)
    monkeypatch.setattr(import_module, "PaperDownloader", lambda papers_dir: _FakeDownloader(papers_dir))
    monkeypatch.setattr(import_module.PaperMemoryBuilder, "build_and_store", lambda self, paper_id: {"memory_id": f"pm:{paper_id}"})
    monkeypatch.setattr(
        db,
        "get_document_memory_summary",
        lambda document_id: (
            {"document_id": document_id, "provenance": {"parser": summary_state["parser"]}}
            if summary_state["ready"]
            else None
        ),
    )

    def _build_document_memory(self, paper_id, **kwargs):  # noqa: ANN001
        parser_token = str(kwargs.get("paper_parser"))
        parser_calls.append(parser_token)
        summary_state["ready"] = True
        summary_state["parser"] = parser_token
        return [{"document_id": f"paper:{paper_id}"}]

    monkeypatch.setattr(import_module.DocumentMemoryBuilder, "build_and_store_paper", _build_document_memory)

    first = runner.invoke(
        paper_group,
        ["import-csv", "--csv", str(csv_path), "--document-memory-parser", "opendataloader", "--json"],
        obj={"khub": khub},
    )
    assert first.exit_code == 0, first.output
    manifest_path = Path(json.loads(first.output)["manifestPath"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["artifacts"].pop("documentMemoryParser", None)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    second = runner.invoke(
        paper_group,
        ["import-csv", "--csv", str(csv_path), "--document-memory-parser", "raw", "--json"],
        obj={"khub": khub},
    )
    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.output)
    second_item = second_payload["items"][0]
    assert second_item["executedSteps"] == ["document-memory"]
    assert second_item["artifacts"]["documentMemoryParser"] == "raw"
    assert parser_calls == ["opendataloader", "raw"]


def test_paper_import_csv_downloads_non_arxiv_resolved_pdf(monkeypatch, tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    khub = _StubKhub(db, tmp_path)
    runner = CliRunner()
    csv_path = tmp_path / "papers.csv"
    _write_csv(
        csv_path,
        [["최신/이슈", "OpenReview Paper", "2026", "에이전트", "5", "selected", "https://openreview.net/forum?id=abc"]],
    )
    vector_db = _FakeVectorDB()

    monkeypatch.setattr(
        import_module,
        "resolve_url",
        lambda url: SimpleNamespace(
            arxiv_id="openreview-abc",
            title="OpenReview Paper",
            authors="A. Author",
            year=2026,
            citation_count=5,
            fields_of_study=["AI"],
            source="openreview",
            pdf_url="https://openreview.net/pdf?id=abc",
        ),
    )
    monkeypatch.setattr(import_module, "_build_embedder", lambda config, khub=None: _FakeEmbedder())
    monkeypatch.setattr(import_module, "_vector_db", lambda config, khub=None: vector_db)
    monkeypatch.setattr(import_module, "requests", SimpleNamespace(get=lambda *args, **kwargs: _FakeResponse(b"%PDF-1.4\n%fake\n")))
    monkeypatch.setattr(import_module, "PaperDownloader", lambda papers_dir: (_ for _ in ()).throw(AssertionError("arXiv downloader should not be used")))
    monkeypatch.setattr(import_module.PaperMemoryBuilder, "build_and_store", lambda self, paper_id: {"memory_id": f"pm:{paper_id}"})
    monkeypatch.setattr(import_module.DocumentMemoryBuilder, "build_and_store_paper", lambda self, paper_id, **kwargs: [{"document_id": f"paper:{paper_id}"}])

    result = runner.invoke(
        paper_group,
        ["import-csv", "--csv", str(csv_path), "--steps", "register,download,embed,paper-memory"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    stored = db.get_paper("openreview-abc")
    assert stored is not None
    assert stored["pdf_path"]
    assert Path(stored["pdf_path"]).exists()


def test_paper_import_csv_resumes_non_arxiv_download_from_manifest(monkeypatch, tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    khub = _StubKhub(db, tmp_path)
    runner = CliRunner()
    csv_path = tmp_path / "papers.csv"
    _write_csv(
        csv_path,
        [["최신/이슈", "Resumed OpenReview", "2026", "에이전트", "5", "selected", "https://openreview.net/forum?id=resume"]],
    )

    monkeypatch.setattr(
        import_module,
        "resolve_url",
        lambda url: SimpleNamespace(
            arxiv_id="openreview-resume",
            title="Resumed OpenReview",
            authors="A. Author",
            year=2026,
            citation_count=5,
            fields_of_study=["AI"],
            source="openreview",
            pdf_url="https://openreview.net/pdf?id=resume",
        ),
    )
    first = runner.invoke(
        paper_group,
        ["import-csv", "--csv", str(csv_path), "--steps", "register", "--json"],
        obj={"khub": khub},
    )
    assert first.exit_code == 0, first.output

    monkeypatch.setattr(import_module, "resolve_url", lambda url: (_ for _ in ()).throw(AssertionError("register should not rerun")))
    monkeypatch.setattr(import_module, "requests", SimpleNamespace(get=lambda *args, **kwargs: _FakeResponse(b"%PDF-1.4\n%fake\n")))
    monkeypatch.setattr(import_module, "PaperDownloader", lambda papers_dir: (_ for _ in ()).throw(AssertionError("arXiv downloader should not be used")))

    second = runner.invoke(
        paper_group,
        ["import-csv", "--csv", str(csv_path), "--steps", "register,download", "--json"],
        obj={"khub": khub},
    )
    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.output)
    second_item = second_payload["items"][0]
    assert second_item["completedSteps"] == ["register", "download"]
    assert second_item["executedSteps"] == ["download"]
    assert Path(db.get_paper("openreview-resume")["pdf_path"]).exists()


def test_paper_import_csv_fails_non_arxiv_without_pdf_url(monkeypatch, tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    khub = _StubKhub(db, tmp_path)
    runner = CliRunner()
    csv_path = tmp_path / "papers.csv"
    _write_csv(
        csv_path,
        [["최신/이슈", "No PDF URL", "2026", "에이전트", "5", "selected", "https://openreview.net/forum?id=no-pdf"]],
    )

    monkeypatch.setattr(
        import_module,
        "resolve_url",
        lambda url: SimpleNamespace(
            arxiv_id="openreview-no-pdf",
            title="No PDF URL",
            authors="A. Author",
            year=2026,
            citation_count=5,
            fields_of_study=["AI"],
            source="openreview",
            pdf_url="",
        ),
    )
    monkeypatch.setattr(import_module, "PaperDownloader", lambda papers_dir: (_ for _ in ()).throw(AssertionError("arXiv downloader should not be used")))

    result = runner.invoke(
        paper_group,
        ["import-csv", "--csv", str(csv_path), "--steps", "register,download", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["items"][0]["failedStep"] == "download"
    assert "pdf_url" in payload["items"][0]["error"]


def test_paper_import_csv_relinks_known_contaminated_title_collision(monkeypatch, tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    khub = _StubKhub(db, tmp_path)
    runner = CliRunner()
    csv_path = tmp_path / "papers.csv"
    _write_csv(
        csv_path,
        [["최신/이슈", "Batch Normalization", "2026", "에이전트", "5", "selected", "https://example.com/paper-bn"]],
    )
    canonical_pdf = tmp_path / "canonical-bn.pdf"
    canonical_text = tmp_path / "canonical-bn.txt"
    canonical_pdf.write_bytes(b"%PDF-1.4\n%fake\n")
    canonical_text.write_text(
        "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift\n"
        "Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training.",
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": "1502.03167",
            "title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
            "authors": "A. Author",
            "year": 2015,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": str(canonical_pdf),
            "text_path": str(canonical_text),
            "translated_path": "",
        }
    )
    monkeypatch.setattr(
        import_module,
        "resolve_url",
        lambda url: SimpleNamespace(
            arxiv_id="Batch_Normalization_c72acd36",
            title="Batch Normalization",
            authors="A. Author",
            year=2026,
            citation_count=5,
            fields_of_study=["AI"],
        ),
    )
    monkeypatch.setattr(
        import_module,
        "PaperDownloader",
        lambda papers_dir: _PreviewDownloader(
            papers_dir,
            preview_text="CHAPTER 1 DESCRIPTIVE STATISTICS AND PROBABILITY THEORY\nArtificial Neural Network and Deep Learning",
        ),
    )

    result = runner.invoke(
        paper_group,
        ["import-csv", "--csv", str(csv_path), "--steps", "register,download", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    item = payload["items"][0]
    assert item["status"] == "completed"
    assert item["sourceGuard"]["decision"] == "relink_to_canonical"
    assert item["sourceGuard"]["canonicalPaperId"] == "1502.03167"
    stored = db.get_paper("Batch_Normalization_c72acd36")
    assert stored["pdf_path"] == str(canonical_pdf)
    assert stored["text_path"] == str(canonical_text)


def test_paper_import_csv_blocks_unknown_title_collision_when_preview_mismatches(monkeypatch, tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    khub = _StubKhub(db, tmp_path)
    runner = CliRunner()
    csv_path = tmp_path / "papers.csv"
    _write_csv(
        csv_path,
        [["최신/이슈", "Transformer Reasoning", "2026", "에이전트", "5", "selected", "https://example.com/paper-tr"]],
    )
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
            "authors": "A. Author",
            "year": 2024,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": str(canonical_pdf),
            "text_path": str(canonical_text),
            "translated_path": "",
        }
    )
    monkeypatch.setattr(
        import_module,
        "resolve_url",
        lambda url: SimpleNamespace(
            arxiv_id="Transformer_Reasoning_abcd1234",
            title="Transformer Reasoning",
            authors="A. Author",
            year=2026,
            citation_count=1,
            fields_of_study=["AI"],
        ),
    )
    monkeypatch.setattr(
        import_module,
        "PaperDownloader",
        lambda papers_dir: _PreviewDownloader(
            papers_dir,
            preview_text="Chapter 1 Introduction to Statistics and Probability",
        ),
    )

    result = runner.invoke(
        paper_group,
        ["import-csv", "--csv", str(csv_path), "--steps", "register,download", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    item = payload["items"][0]
    assert payload["status"] == "failed"
    assert item["failedStep"] == "download"
    assert item["sourceGuard"]["decision"] == "block_manual_review"
    stored = db.get_paper("Transformer_Reasoning_abcd1234")
    assert stored["pdf_path"] in {None, ""}
    assert stored["text_path"] in {None, ""}


def test_paper_import_csv_keeps_clean_duplicate_when_preview_matches(monkeypatch, tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    khub = _StubKhub(db, tmp_path)
    runner = CliRunner()
    csv_path = tmp_path / "papers.csv"
    _write_csv(
        csv_path,
        [["최신/이슈", "NeRF", "2026", "에이전트", "5", "selected", "https://example.com/paper-nerf"]],
    )
    canonical_pdf = tmp_path / "canonical-nerf.pdf"
    canonical_text = tmp_path / "canonical-nerf.txt"
    canonical_pdf.write_bytes(b"%PDF-1.4\n%fake\n")
    canonical_text.write_text(
        "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis\nWe present a method that achieves state-of-the-art results for synthesizing novel views.",
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": "2003.08934",
            "title": "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
            "authors": "A. Author",
            "year": 2020,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": str(canonical_pdf),
            "text_path": str(canonical_text),
            "translated_path": "",
        }
    )
    monkeypatch.setattr(
        import_module,
        "resolve_url",
        lambda url: SimpleNamespace(
            arxiv_id="NeRF_1b9d0d11",
            title="NeRF",
            authors="A. Author",
            year=2026,
            citation_count=5,
            fields_of_study=["AI"],
        ),
    )
    monkeypatch.setattr(
        import_module,
        "PaperDownloader",
        lambda papers_dir: _PreviewDownloader(
            papers_dir,
            preview_text="NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis\nWe present a method that achieves state-of-the-art results for synthesizing novel views of complex scenes.",
        ),
    )

    result = runner.invoke(
        paper_group,
        ["import-csv", "--csv", str(csv_path), "--steps", "register,download", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    item = payload["items"][0]
    assert item["status"] == "completed"
    assert item["sourceGuard"]["decision"] == "allow_duplicate"
    stored = db.get_paper("NeRF_1b9d0d11")
    assert stored["pdf_path"] != str(canonical_pdf)
    assert "NeRF_1b9d0d11" in str(stored["pdf_path"] or "")
