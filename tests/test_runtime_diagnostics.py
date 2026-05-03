from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from knowledge_hub.application.runtime_diagnostics import build_runtime_diagnostics
from knowledge_hub.application.runtime_diagnostics import parser_runtime_status


class _FakeDatabase:
    def __init__(self, stats: dict[str, object]):
        self._stats = stats

    def get_stats(self):  # noqa: ANN001
        return dict(self._stats)


def test_build_runtime_diagnostics_marks_empty_vector_corpus_degraded():
    searcher = SimpleNamespace(
        config=SimpleNamespace(
            translation_provider="openai",
            translation_model="gpt-4o-mini",
            summarization_provider="ollama",
            summarization_model="qwen3:14b",
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
        ),
        database=_FakeDatabase(
            {
                "collection_name": "knowledge_hub",
                "total_documents": 0,
                "db_path": "/tmp/vector",
                "lexical_db_path": "/tmp/vector-fts",
            }
        ),
        embedder=SimpleNamespace(get_last_status=lambda: {}),
    )

    payload = build_runtime_diagnostics(searcher.config, searcher=searcher)

    assert payload["status"] == "degraded"
    assert payload["degraded"] is True
    assert payload["vectorCorpus"]["reasons"] == ["vector_corpus_empty"]
    assert payload["vectorCorpus"]["available"] is False
    assert "vector corpus degraded" in payload["warnings"][0]


def test_build_runtime_diagnostics_marks_missing_vector_corpus_unavailable():
    payload = build_runtime_diagnostics(None, searcher=None)

    assert payload["status"] == "degraded"
    assert payload["vectorCorpus"]["reasons"] == ["vector_corpus_unavailable"]


def test_build_runtime_diagnostics_uses_non_chroma_vector_inspection(monkeypatch):
    config = SimpleNamespace(
        translation_provider="openai",
        translation_model="gpt-4o-mini",
        summarization_provider="openai",
        summarization_model="gpt-4o-mini",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        vector_db_path="/tmp/vector",
        collection_name="knowledge_hub_bge_m3_v1",
        get_provider_config=lambda provider: {},
    )
    monkeypatch.setattr("knowledge_hub.application.runtime_diagnostics.provider_runtime_probe", lambda config, provider_name: {})
    monkeypatch.setattr("knowledge_hub.application.runtime_diagnostics.resolve_api_key", lambda provider_name, raw_key: "ok")
    monkeypatch.setattr(
        "knowledge_hub.application.runtime_diagnostics.inspect_vector_store",
        lambda db_path, collection_name: {
            "available": False,
            "degraded": True,
            "reasons": ["vector_corpus_empty"],
            "collection_name": collection_name,
            "total_documents": 0,
            "db_path": db_path,
            "lexical_db_path": "/tmp/vector/_lexical.sqlite3",
            "recovery_backup": {"path": "/tmp/vector.corrupt.1", "total_documents": 2436},
        },
    )

    payload = build_runtime_diagnostics(config)

    assert payload["vectorCorpus"]["collection_name"] == "knowledge_hub_bge_m3_v1"
    assert payload["vectorCorpus"]["recovery_backup"]["total_documents"] == 2436
    assert any("vector corpus backup available" in warning for warning in payload["warnings"])


def test_build_runtime_diagnostics_marks_lexical_only_store_degraded_without_searcher(monkeypatch, tmp_path):
    vector_root = tmp_path / "vector"
    vector_root.mkdir(parents=True, exist_ok=True)
    lexical = sqlite3.connect(vector_root / "_lexical.sqlite3")
    lexical.execute(
        """
        CREATE VIRTUAL TABLE lexical_documents_fts
        USING fts5(
            doc_id UNINDEXED,
            title,
            section_title,
            contextual_summary,
            keywords,
            field,
            document,
            searchable_text
        )
        """
    )
    lexical.execute(
        """
        INSERT INTO lexical_documents_fts
        (doc_id, title, section_title, contextual_summary, keywords, field, document, searchable_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("doc-1", "title", "", "", "", "", "body", "title body"),
    )
    lexical.commit()
    lexical.close()

    config = SimpleNamespace(
        translation_provider="openai",
        translation_model="gpt-4o-mini",
        summarization_provider="openai",
        summarization_model="gpt-4o-mini",
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        vector_db_path=str(vector_root),
        collection_name="knowledge_hub",
    )
    monkeypatch.setattr("knowledge_hub.application.runtime_diagnostics.provider_runtime_probe", lambda config, provider_name: {})
    monkeypatch.setattr("knowledge_hub.application.runtime_diagnostics.resolve_api_key", lambda provider_name, raw_key: "ok")

    payload = build_runtime_diagnostics(config, searcher=None)

    assert payload["status"] == "degraded"
    assert payload["vectorCorpus"]["available"] is False
    assert payload["vectorCorpus"]["total_documents"] == 0
    assert payload["vectorCorpus"]["lexical_documents"] == 1
    assert payload["vectorCorpus"]["chroma_embeddings"] == 0
    assert payload["vectorCorpus"]["reasons"] == ["vector_corpus_empty", "vector_embeddings_missing"]


def test_build_runtime_diagnostics_marks_summarization_provider_runtime_unavailable(monkeypatch):
    searcher = SimpleNamespace(
        config=SimpleNamespace(
            translation_provider="openai",
            translation_model="gpt-4o-mini",
            summarization_provider="ollama",
            summarization_model="qwen3:14b",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        ),
        database=_FakeDatabase(
            {
                "collection_name": "knowledge_hub",
                "total_documents": 10,
                "db_path": "/tmp/vector",
                "lexical_db_path": "/tmp/vector-fts",
            }
        ),
        embedder=SimpleNamespace(get_last_status=lambda: {}),
    )
    monkeypatch.setattr(
        "knowledge_hub.application.runtime_diagnostics._ollama_check",
        lambda config: {
            "available": False,
            "status": "blocked",
            "summary": "Ollama 서버에 연결할 수 없습니다.",
            "detail": "connection refused",
            "fixCommand": "ollama serve",
        },
    )

    payload = build_runtime_diagnostics(searcher.config, searcher=searcher)

    summarization = next(item for item in payload["providers"] if item["role"] == "summarization")
    assert payload["status"] == "degraded"
    assert summarization["available"] is False
    assert "provider_runtime_unavailable" in summarization["reasons"]
    assert any("summarization degraded: provider_runtime_unavailable" in warning for warning in payload["warnings"])


def test_build_runtime_diagnostics_does_not_degrade_on_zero_embedder_runtime_status(monkeypatch):
    searcher = SimpleNamespace(
        config=SimpleNamespace(
            translation_provider="openai",
            translation_model="gpt-4o-mini",
            summarization_provider="openai",
            summarization_model="gpt-4o-mini",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        ),
        database=_FakeDatabase(
            {
                "collection_name": "knowledge_hub",
                "total_documents": 10,
                "db_path": "/tmp/vector",
                "lexical_db_path": "/tmp/vector-fts",
            }
        ),
        embedder=SimpleNamespace(get_last_status=lambda: {"retries": 0, "failures": []}),
    )
    monkeypatch.setattr("knowledge_hub.application.runtime_diagnostics.provider_runtime_probe", lambda config, provider_name: {})
    monkeypatch.setattr("knowledge_hub.application.runtime_diagnostics.resolve_api_key", lambda provider_name, raw_key: "ok")

    payload = build_runtime_diagnostics(searcher.config, searcher=searcher)

    assert payload["status"] == "ok"
    assert payload["degraded"] is False
    assert payload["warnings"] == []


def test_parser_runtime_status_blocks_opendataloader_when_java_missing(monkeypatch):
    monkeypatch.setattr("knowledge_hub.application.runtime_diagnostics.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(
        "knowledge_hub.application.runtime_diagnostics._java_runtime_status",
        lambda: {
            "available": False,
            "status": "blocked",
            "detail": "Unable to locate a Java Runtime.",
            "fixCommand": 'export PATH="$(brew --prefix openjdk)/bin:$PATH"',
        },
    )

    payload = parser_runtime_status("opendataloader")

    assert payload["available"] is False
    assert payload["status"] == "blocked"
    assert "Java runtime" in payload["detail"]


def test_parser_runtime_status_blocks_pymupdf_when_package_missing(monkeypatch):
    real_find_spec = __import__("importlib.util").util.find_spec

    def _fake_find_spec(name: str):  # noqa: ANN001
        if name == "fitz":
            return None
        return real_find_spec(name)

    monkeypatch.setattr("knowledge_hub.application.runtime_diagnostics.importlib.util.find_spec", _fake_find_spec)

    payload = parser_runtime_status("pymupdf")

    assert payload["available"] is False
    assert payload["status"] == "needs_setup"
    assert "PyMuPDF" in payload["detail"]


def test_parser_runtime_status_degrades_pymupdf_when_ocr_missing(monkeypatch):
    real_find_spec = __import__("importlib.util").util.find_spec

    def _fake_find_spec(name: str):  # noqa: ANN001
        if name == "fitz":
            return object()
        return real_find_spec(name)

    def _fake_which(name: str):  # noqa: ANN001
        if name == "tesseract":
            return "/opt/homebrew/bin/tesseract"
        return None

    monkeypatch.setattr("knowledge_hub.application.runtime_diagnostics.importlib.util.find_spec", _fake_find_spec)
    monkeypatch.setattr("knowledge_hub.application.runtime_diagnostics.shutil.which", _fake_which)

    payload = parser_runtime_status("pymupdf")

    assert payload["available"] is True
    assert payload["status"] == "degraded"
    assert "scanned PDF OCR" in payload["detail"]


def test_parser_runtime_status_blocks_mineru_when_runtime_dependencies_are_broken(monkeypatch):
    monkeypatch.setattr("knowledge_hub.application.runtime_diagnostics.shutil.which", lambda name: "/tmp/mineru" if name == "mineru" else None)
    monkeypatch.setattr(
        "knowledge_hub.application.runtime_diagnostics.importlib.metadata.version",
        lambda name: "3.0.1" if name == "mineru" else "5.3.0",
    )
    monkeypatch.setattr(
        "knowledge_hub.application.runtime_diagnostics._mineru_dependency_status",
        lambda: {
            "available": False,
            "status": "blocked",
            "detail": "MinerU runtime dependency check failed: transformers=5.3.0 cannot import helper",
            "fixCommand": "python -m pip install -e '.[mineru]'",
        },
    )

    payload = parser_runtime_status("mineru")

    assert payload["available"] is False
    assert payload["status"] == "blocked"
    assert "transformers=5.3.0" in payload["detail"]
