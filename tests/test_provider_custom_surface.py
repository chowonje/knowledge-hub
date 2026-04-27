from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.providers import get_llm, get_provider_info
from knowledge_hub.interfaces.cli.commands.provider_cmd import provider_group
from knowledge_hub.interfaces.cli.main import cli


def _config(tmp_path: Path) -> Config:
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    return Config(str(path))


def test_custom_openai_compatible_provider_is_available_to_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    config = _config(tmp_path)
    config.set_nested("providers", "deepseek", "adapter", "openai-compatible")
    config.set_nested("providers", "deepseek", "base_url", "https://api.deepseek.com/v1")
    config.set_nested("providers", "deepseek", "api_key_env", "DEEPSEEK_API_KEY")
    config.set_nested("providers", "deepseek", "supports", {"llm": True, "embedding": False})
    config.set_nested("providers", "deepseek", "models", {"llm": ["deepseek-chat"], "embedding": []})
    config.set_nested("providers", "deepseek", "default_llm_model", "deepseek-chat")

    provider_config = config.get_provider_config("deepseek")
    assert provider_config["api_key"] == "test-key"

    info = get_provider_info("deepseek", config=config)
    assert info is not None
    assert info.supports_llm is True
    assert info.supports_embedding is False
    assert info.default_llm_model == "deepseek-chat"

    llm = get_llm("deepseek", model="deepseek-chat", **provider_config)
    assert llm.base_url == "https://api.deepseek.com/v1"
    assert llm.api_key == "test-key"
    assert llm.provider_name == "deepseek"


def test_provider_add_and_use_configures_custom_answer_role(tmp_path):
    config = _config(tmp_path)
    result = CliRunner().invoke(
        provider_group,
        [
            "add",
            "deepseek",
            "--from-service",
            "deepseek",
            "--use-for",
            "answer",
            "--json",
        ],
        obj={"khub": SimpleNamespace(config=config)},
    )

    assert result.exit_code == 0, result.output
    assert config.summarization_provider == "deepseek"
    assert config.summarization_model == "deepseek-chat"
    assert config.get_nested("providers", "deepseek", "api_key_env") == "DEEPSEEK_API_KEY"
    assert config.get_nested("providers", "deepseek", "adapter") == "openai-compatible"

    result = CliRunner().invoke(
        provider_group,
        ["use", "translation", "deepseek/deepseek-reasoner", "--json"],
        obj={"khub": SimpleNamespace(config=config)},
    )

    assert result.exit_code == 0, result.output
    assert config.translation_provider == "deepseek"
    assert config.translation_model == "deepseek-reasoner"


def test_provider_key_uses_env_reference_not_raw_secret(tmp_path):
    config = _config(tmp_path)
    result = CliRunner().invoke(
        provider_group,
        ["key", "deepseek", "--env", "DEEPSEEK_API_KEY", "--json"],
        obj={"khub": SimpleNamespace(config=config)},
    )

    assert result.exit_code == 0, result.output
    assert config.get_nested("providers", "deepseek", "api_key_env") == "DEEPSEEK_API_KEY"
    assert config.get_nested("providers", "deepseek", "api_key", default="") == ""
    assert "DEEPSEEK_API_KEY" in result.output


def test_provider_setup_codex_mcp_sets_answer_backend(tmp_path):
    config = _config(tmp_path)
    result = CliRunner().invoke(
        provider_group,
        ["setup", "--profile", "codex-mcp", "--json"],
        obj={"khub": SimpleNamespace(config=config)},
    )

    assert result.exit_code == 0, result.output
    assert config.get_nested("routing", "llm", "tasks", "rag_answer", "preferred_backend") == "codex_mcp"
    assert config.get_nested("routing", "llm", "tasks", "rag_answer", "codex", "transport") == "mcp"
    assert config.get_nested("routing", "llm", "tasks", "rag_answer", "codex", "command") == "codex"
    assert config.get_nested("routing", "llm", "tasks", "rag_answer", "codex", "args") == "mcp-server"


def test_root_help_promotes_provider_surface():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "provider" in result.output
