from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from knowledge_hub.interfaces.cli.commands import doctor_cmd as doctor_module

setup_module = None


class _FakeKhub:
    def __init__(self, config):
        self.config = config

    def searcher(self):
        return SimpleNamespace(config=self.config)


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config_path=str(tmp_path / "config.yaml"),
        translation_provider="openai",
        translation_model="gpt-5-nano",
        summarization_provider="openai",
        summarization_model="gpt-5-nano",
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        paper_summary_parser="auto",
        papers_dir=str(tmp_path / "papers"),
        sqlite_path=str(tmp_path / "knowledge.db"),
        get_provider_config=lambda provider: {"base_url": "http://localhost:11434"} if provider == "ollama" else {},
        get_nested=lambda *args, default=None: default,
    )


def test_build_doctor_payload_translates_runtime_state(monkeypatch, tmp_path):
    config = _config(tmp_path)
    monkeypatch.setattr(
        doctor_module,
        "build_runtime_diagnostics",
        lambda config, searcher=None, searcher_error="": {
            "providers": [
                {"role": "translation", "provider": "openai", "model": "gpt-5-nano", "installed": True, "requires_api_key": True, "api_key_status": "missing", "degraded": True, "reasons": ["missing_api_key"], "available": False},
                {"role": "summarization", "provider": "openai", "model": "gpt-5-nano", "installed": True, "requires_api_key": True, "api_key_status": "missing", "degraded": True, "reasons": ["missing_api_key"], "available": False},
                {"role": "embedding", "provider": "ollama", "model": "nomic-embed-text", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": False, "reasons": [], "available": True},
            ],
            "vectorCorpus": {"available": False, "reasons": ["vector_corpus_empty"], "total_documents": 0, "collection_name": "knowledge_hub"},
            "warnings": [],
        },
    )
    monkeypatch.setattr(doctor_module, "_module_available", lambda name: False)
    monkeypatch.setattr(
        doctor_module,
        "parser_runtime_status",
        lambda name: {"available": False, "status": "needs_setup", "detail": f"{name} missing", "fixCommand": "install parser"},
    )
    monkeypatch.setattr(doctor_module, "_ollama_ok", lambda url: False)
    monkeypatch.setattr(doctor_module, "_ollama_available", lambda url: False)
    monkeypatch.setattr(doctor_module.registry, "get_provider_info", lambda name: SimpleNamespace(requires_api_key=True, display_name=name, is_local=(name == "ollama")))

    payload = doctor_module.build_doctor_payload(_FakeKhub(config))
    assert payload["schema"] == "knowledge-hub.doctor.result.v1"
    assert payload["status"] in {"needs_setup", "blocked"}
    assert any(item["area"] == "summary" for item in payload["checks"])
    assert any(item["area"] == "paper parser" for item in payload["checks"])
    assert any(item["area"] == "vector corpus" for item in payload["checks"])
    assert payload["nextActions"]


def test_build_doctor_payload_prioritizes_actionable_ollama_recovery(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config.translation_provider = "ollama"
    config.translation_model = "qwen3:14b"
    config.summarization_provider = "ollama"
    config.summarization_model = "qwen3:14b"
    config.embedding_provider = "ollama"
    config.embedding_model = "nomic-embed-text"
    monkeypatch.setattr(
        doctor_module,
        "build_runtime_diagnostics",
        lambda config, searcher=None, searcher_error="": {
            "providers": [
                {"role": "translation", "provider": "ollama", "model": "qwen3:14b", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": True, "reasons": ["provider_runtime_unavailable"], "available": False},
                {"role": "summarization", "provider": "ollama", "model": "qwen3:14b", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": True, "reasons": ["provider_runtime_unavailable"], "available": False},
                {"role": "embedding", "provider": "ollama", "model": "nomic-embed-text", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": True, "reasons": ["provider_runtime_unavailable"], "available": False},
            ],
            "vectorCorpus": {"available": False, "reasons": ["vector_corpus_empty"], "total_documents": 0, "collection_name": "knowledge_hub"},
            "warnings": ["semantic retrieval degraded: provider_runtime_unavailable"],
        },
    )
    monkeypatch.setattr(
        doctor_module,
        "parser_runtime_status",
        lambda name: {"available": True, "status": "ok", "detail": f"{name} ok", "fixCommand": ""},
    )
    monkeypatch.setattr(doctor_module, "_ollama_ok", lambda url: False)
    monkeypatch.setattr(doctor_module, "_ollama_available", lambda url: False)
    monkeypatch.setattr(doctor_module.registry, "get_provider_info", lambda name: SimpleNamespace(requires_api_key=False, display_name=name, is_local=(name == "ollama")))

    payload = doctor_module.build_doctor_payload(_FakeKhub(config))

    checks = {item["area"]: item for item in payload["checks"]}
    assert checks["Ollama"]["status"] == "blocked"
    assert "local runtime unavailable" in checks["embedding"]["detail"]
    assert payload["nextActions"][:3] == [
        "ollama serve  # start the local runtime at http://localhost:11434",
        "ollama pull qwen3:14b",
        "ollama pull nomic-embed-text",
    ]
    assert payload["nextActions"][3].startswith("python -m knowledge_hub.interfaces.cli.main doctor")


def test_doctor_cmd_json_outputs_public_shape(monkeypatch, tmp_path):
    config = _config(tmp_path)
    Path(config.config_path).write_text("x: 1\n", encoding="utf-8")
    Path(config.papers_dir).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        doctor_module,
        "build_runtime_diagnostics",
        lambda config, searcher=None, searcher_error="": {
            "providers": [
                {"role": "translation", "provider": "openai", "model": "gpt-5-nano", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": False, "reasons": [], "available": True},
                {"role": "summarization", "provider": "openai", "model": "gpt-5-nano", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": False, "reasons": [], "available": True},
                {"role": "embedding", "provider": "ollama", "model": "nomic-embed-text", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": False, "reasons": [], "available": True},
            ],
            "vectorCorpus": {"available": True, "reasons": [], "total_documents": 5, "collection_name": "knowledge_hub"},
            "warnings": [],
        },
    )
    monkeypatch.setattr(doctor_module, "_module_available", lambda name: True)
    monkeypatch.setattr(
        doctor_module,
        "parser_runtime_status",
        lambda name: {"available": True, "status": "ok", "detail": f"{name} ok", "fixCommand": ""},
    )
    monkeypatch.setattr(doctor_module, "_ollama_ok", lambda url: True)
    monkeypatch.setattr(doctor_module, "_ollama_available", lambda url: True)
    monkeypatch.setattr(doctor_module.registry, "get_provider_info", lambda name: SimpleNamespace(requires_api_key=False, display_name=name, is_local=(name == "ollama")))

    runner = CliRunner()
    result = runner.invoke(doctor_module.doctor_cmd, ["--json"], obj={"khub": _FakeKhub(config)})
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.doctor.result.v1"
    assert payload["status"] == "ok"
    assert isinstance(payload["checks"], list)
    assert isinstance(payload["nextActions"], list)
