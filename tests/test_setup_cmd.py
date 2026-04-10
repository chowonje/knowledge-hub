from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.core import config as core_config_module
from knowledge_hub.infrastructure import config as config_module
from knowledge_hub.interfaces.cli.commands import setup_cmd as setup_mod


def test_setup_local_profile_writes_public_defaults(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(core_config_module, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(setup_mod, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(setup_mod, "_ollama_runtime_available", lambda config: False)

    runner = CliRunner()
    result = runner.invoke(setup_mod.setup_cmd, ["--profile", "local", "--non-interactive"])

    assert result.exit_code == 0
    cfg = config_module.Config(str(config_path))
    assert cfg.translation_provider == "ollama"
    assert cfg.translation_model == "qwen3:14b"
    assert cfg.summarization_provider == "ollama"
    assert cfg.embedding_provider == "ollama"
    assert cfg.paper_summary_parser == "auto"
    assert "blocked/degraded" in result.output
    assert "ollama serve" in result.output
    assert "python -m knowledge_hub.interfaces.cli.main doctor" in result.output


def test_setup_quick_alias_matches_local_profile(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(core_config_module, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(setup_mod, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(setup_mod, "_ollama_runtime_available", lambda config: False)

    runner = CliRunner()
    result = runner.invoke(setup_mod.setup_cmd, ["--quick"])

    assert result.exit_code == 0
    cfg = config_module.Config(str(config_path))
    assert cfg.translation_provider == "ollama"
    assert cfg.summarization_model == "qwen3:14b"
    assert cfg.paper_summary_parser == "auto"


def test_setup_hybrid_falls_back_embedding_when_ollama_unavailable(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(core_config_module, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(setup_mod, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(setup_mod.registry, "is_provider_available", lambda provider: False if provider == "ollama" else True)
    monkeypatch.setattr(setup_mod, "_ollama_runtime_available", lambda config: False)

    runner = CliRunner()
    result = runner.invoke(setup_mod.setup_cmd, ["--profile", "hybrid", "--non-interactive"])

    assert result.exit_code == 0
    cfg = config_module.Config(str(config_path))
    assert cfg.translation_provider == "openai"
    assert cfg.translation_model == "gpt-5-nano"
    assert cfg.embedding_provider == "openai"
    assert cfg.embedding_model == "text-embedding-3-small"
    assert "fallback으로 저장" in result.output
    assert "knowledge_hub.interfaces.cli.main" in result.output
    assert "doctor" in result.output
