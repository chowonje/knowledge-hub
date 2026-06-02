from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.interfaces.cli.commands.models_cmd import chat_model_options, chat_provider_options, models_group
from knowledge_hub.learning.task_router import decide_task_route


def _config(tmp_path: Path) -> Config:
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    return Config(str(path))


def test_models_status_redacts_api_key_env_value(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-openai-value")
    config = _config(tmp_path)
    config.set_nested("providers", "openai", "api_key_env", "OPENAI_API_KEY")
    config.set_nested("routing", "llm", "tasks", "chat", "provider", "openai")
    config.set_nested("routing", "llm", "tasks", "chat", "model", "gpt-5.4")

    result = CliRunner().invoke(
        models_group,
        ["status", "--json"],
        obj={"khub": SimpleNamespace(config=config)},
    )

    assert result.exit_code == 0, result.output
    assert "unit-test-openai-value" not in result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.models.result.v1"
    assert payload["chat"]["provider"] == "openai"
    assert payload["chat"]["model"] == "gpt-5.4"
    assert payload["chat"]["apiKeyEnv"] == "OPENAI_API_KEY"
    assert payload["chat"]["apiKeyStatus"] == "set"


def test_models_use_updates_chat_task_only(tmp_path):
    config = _config(tmp_path)
    config.set_nested("summarization", "provider", "ollama")
    config.set_nested("summarization", "model", "qwen3:14b")
    config.set_nested("routing", "llm", "tasks", "local", "provider", "local-test-provider")
    config.set_nested("routing", "llm", "tasks", "local", "model", "local-test-model")
    before = decide_task_route(config, task_type="rag_answer", allow_external=False, query="short ask")

    result = CliRunner().invoke(
        models_group,
        ["use", "openai/gpt-5.4", "--json"],
        obj={"khub": SimpleNamespace(config=config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["target"] == "routing.llm.tasks.chat"
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-5.4"
    assert config.get_nested("routing", "llm", "tasks", "chat", "provider") == "openai"
    assert config.get_nested("routing", "llm", "tasks", "chat", "model") == "gpt-5.4"
    assert config.summarization_provider == "ollama"
    assert config.summarization_model == "qwen3:14b"
    after = decide_task_route(config, task_type="rag_answer", allow_external=False, query="short ask")
    assert after.provider == before.provider == "local-test-provider"
    assert after.model == before.model == "local-test-model"


def test_models_select_prompts_for_provider_and_model(tmp_path):
    config = _config(tmp_path)

    result = CliRunner().invoke(
        models_group,
        ["select", "--provider", "openai"],
        input="4\n",
        obj={"khub": SimpleNamespace(config=config)},
    )

    assert result.exit_code == 0, result.output
    assert "Choose Model (openai)" in result.output
    assert "routing.llm.tasks.chat -> openai/gpt-5.4" in result.output
    assert config.get_nested("routing", "llm", "tasks", "chat", "provider") == "openai"
    assert config.get_nested("routing", "llm", "tasks", "chat", "model") == "gpt-5.4"


def test_models_select_provider_options_include_openai_compatible_catalog(tmp_path):
    config = _config(tmp_path)

    options = chat_provider_options(config)
    names = {item["name"] for item in options}

    assert "codex" in names
    assert {"deepseek", "groq", "mistral", "openrouter", "perplexity", "xai"}.issubset(names)
    assert chat_model_options(config, "deepseek") == ["deepseek-chat", "deepseek-reasoner"]
    assert chat_model_options(config, "codex")[0] == "gpt-5.5"


def test_models_select_materializes_openai_compatible_provider_alias(tmp_path):
    config = _config(tmp_path)

    result = CliRunner().invoke(
        models_group,
        ["select", "--provider", "deepseek"],
        input="1\n",
        obj={"khub": SimpleNamespace(config=config)},
    )

    assert result.exit_code == 0, result.output
    assert "Choose Model (deepseek)" in result.output
    assert "routing.llm.tasks.chat -> deepseek/deepseek-chat" in result.output
    assert config.get_nested("routing", "llm", "tasks", "chat", "provider") == "deepseek"
    assert config.get_nested("routing", "llm", "tasks", "chat", "model") == "deepseek-chat"
    assert config.get_nested("providers", "deepseek", "adapter") == "openai-compatible"
    assert config.get_nested("providers", "deepseek", "base_url") == "https://api.deepseek.com/v1"
    assert config.get_nested("providers", "deepseek", "api_key_env") == "DEEPSEEK_API_KEY"


def test_models_login_stores_env_reference_not_raw_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-openai-value")
    config = _config(tmp_path)

    result = CliRunner().invoke(
        models_group,
        ["login", "openai", "--env", "OPENAI_API_KEY", "--json"],
        obj={"khub": SimpleNamespace(config=config)},
    )

    assert result.exit_code == 0, result.output
    assert "unit-test-openai-value" not in result.output
    payload = json.loads(result.output)
    assert payload["provider"] == "openai"
    assert payload["apiKeyEnv"] == "OPENAI_API_KEY"
    assert payload["apiKeyStatus"] == "set"
    assert payload["storedSecret"] is False
    assert config.get_nested("providers", "openai", "api_key_env") == "OPENAI_API_KEY"
    saved = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "api_key_env: OPENAI_API_KEY" in saved
    assert "unit-test-openai-value" not in saved


def test_models_status_falls_back_to_summarization_route(tmp_path):
    config = _config(tmp_path)
    config.set_nested("summarization", "provider", "ollama")
    config.set_nested("summarization", "model", "qwen3:14b")

    result = CliRunner().invoke(
        models_group,
        ["status", "--json"],
        obj={"khub": SimpleNamespace(config=config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["chat"]["source"] == "summarization"
    assert payload["chat"]["provider"] == "ollama"
    assert payload["chat"]["model"] == "qwen3:14b"


def test_models_use_can_set_codex_delegated_chat_provider(tmp_path):
    config = _config(tmp_path)

    result = CliRunner().invoke(
        models_group,
        ["use", "codex/gpt-5.5", "--json"],
        obj={"khub": SimpleNamespace(config=config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["target"] == "routing.llm.tasks.chat"
    assert payload["provider"] == "codex"
    assert payload["model"] == "gpt-5.5"
    assert config.get_nested("routing", "llm", "tasks", "chat", "provider") == "codex"
    assert config.get_nested("routing", "llm", "tasks", "chat", "model") == "gpt-5.5"
