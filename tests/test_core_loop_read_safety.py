from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from knowledge_hub.application import context as context_module
from knowledge_hub.application.context import AppContext
from knowledge_hub.infrastructure.persistence.vector import VectorDatabase, inspect_vector_store
from knowledge_hub.interfaces.cli.commands import doctor_cmd, status_cmd


class _DummyEmbedder:
    def embed_text(self, text: str) -> list[float]:
        _ = text
        return [0.0]


class _DummyLLM:
    def generate(self, prompt: str, context: str = "") -> str:
        _ = (prompt, context)
        return "ok"


class _StatsSQLite:
    def get_stats(self) -> dict[str, int]:
        return {
            "papers": 1,
            "notes": 1,
            "tags": 1,
            "links": 1,
        }


class _ReadOnlyKhub:
    def __init__(self, config):
        self.config = config

    def sqlite_db(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        return _StatsSQLite()

    def vector_db(self, *, repair_on_init: bool = True):
        raise AssertionError(f"read-only diagnostics should not open VectorDatabase directly: {repair_on_init=}")

    def searcher(self):
        raise AssertionError("read-only diagnostics should not build a searcher")


def _default_config(tmp_path: Path, vector_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config_path=str(tmp_path / "config.yaml"),
        sqlite_path=str(tmp_path / "knowledge.sqlite3"),
        vector_db_path=str(vector_root),
        collection_name="knowledge_hub",
        translation_provider="openai",
        translation_model="gpt-4o-mini",
        summarization_provider="openai",
        summarization_model="gpt-4o-mini",
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        vault_enabled=False,
        vault_path=str(tmp_path / "vault"),
        get_provider_config=lambda provider: {},
        get_nested=lambda *path, default=None: default,
    )


def test_seeded_vector_corpus_survives_status_doctor_and_search(monkeypatch, tmp_path: Path):
    vector_root = tmp_path / "vector"
    db = VectorDatabase(str(vector_root), "knowledge_hub")
    db.add_documents(
        ["alpha retrieval memory"],
        [[0.0]],
        [{"title": "Alpha", "source_type": "vault", "file_path": "alpha.md"}],
        ids=["doc-1"],
    )

    config = _default_config(tmp_path, vector_root)
    khub = _ReadOnlyKhub(config)
    repair_flags: list[bool] = []
    real_vector_db = context_module.VectorDatabase

    monkeypatch.setattr(
        "knowledge_hub.application.runtime_diagnostics.provider_runtime_probe",
        lambda config, provider_name: {},
    )
    monkeypatch.setattr(
        "knowledge_hub.application.runtime_diagnostics.resolve_api_key",
        lambda provider_name, raw_key: "ok",
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.status_cmd.reranker_runtime_status",
        lambda config: {
            "enabled": False,
            "model": "BAAI/bge-reranker-v2-m3",
            "candidate_window": 12,
            "timeout_ms": 1500,
            "ready": False,
            "reason": "disabled",
            "reasons": ["disabled"],
        },
    )
    monkeypatch.setattr(
        "knowledge_hub.infrastructure.providers.list_providers",
        lambda: {"openai": SimpleNamespace(display_name="OpenAI"), "ollama": SimpleNamespace(display_name="Ollama")},
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.doctor_cmd._ollama_check",
        lambda config: {
            "area": "Ollama",
            "status": "ok",
            "summary": "ok",
            "detail": "ok",
            "fixCommand": "",
        },
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.doctor_cmd._parser_check",
        lambda config: {
            "area": "parser",
            "status": "ok",
            "summary": "ok",
            "detail": "ok",
            "fixCommand": "",
        },
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.doctor_cmd._storage_check",
        lambda config: {
            "area": "storage",
            "status": "ok",
            "summary": "ok",
            "detail": "ok",
            "fixCommand": "",
        },
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.doctor_cmd._reranker_check",
        lambda config: {
            "area": "reranker",
            "status": "ok",
            "summary": "ok",
            "detail": "ok",
            "fixCommand": "",
        },
    )
    monkeypatch.setattr(context_module, "get_embedder", lambda provider, model=None, **kwargs: _DummyEmbedder())
    monkeypatch.setattr(context_module, "get_llm", lambda provider, model=None, **kwargs: _DummyLLM())

    def _recording_vector_db(db_path: str, collection_name: str, repair_on_init: bool = True):
        repair_flags.append(bool(repair_on_init))
        return real_vector_db(db_path, collection_name, repair_on_init=repair_on_init)

    monkeypatch.setattr(context_module, "VectorDatabase", _recording_vector_db)

    backups_before = sorted(path.name for path in tmp_path.iterdir() if ".corrupt." in path.name)
    baseline = inspect_vector_store(str(vector_root), "knowledge_hub")
    assert baseline["available"] is True
    assert baseline["total_documents"] == 1

    status_cmd.run_status(khub)
    doctor_payload = doctor_cmd.build_doctor_payload(khub)

    app = AppContext(project_root=tmp_path)
    app._config = config
    results = app.searcher.search("alpha", top_k=1)

    backups_after = sorted(path.name for path in tmp_path.iterdir() if ".corrupt." in path.name)
    inspected_after = inspect_vector_store(str(vector_root), "knowledge_hub")

    assert doctor_payload["status"] == "ok"
    assert doctor_payload["nextActions"] == []
    assert repair_flags == [False]
    assert len(results) == 1
    assert results[0].metadata["title"] == "Alpha"
    assert backups_after == backups_before
    assert inspected_after["available"] is True
    assert inspected_after["total_documents"] == 1
