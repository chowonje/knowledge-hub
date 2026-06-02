from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.interfaces.cli.commands.auth_cmd import auth_group


def _config(tmp_path: Path) -> Config:
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    return Config(str(path))


def test_auth_status_shows_codex_oauth_boundary_and_redacts_env_value(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-openai-value")
    config = _config(tmp_path)
    config.set_nested("providers", "openai", "api_key_env", "OPENAI_API_KEY")
    config.set_nested("routing", "llm", "tasks", "chat", "provider", "openai")
    config.set_nested("routing", "llm", "tasks", "chat", "model", "gpt-5.4")

    result = CliRunner().invoke(
        auth_group,
        ["status", "--json"],
        obj={"khub": SimpleNamespace(config=config)},
    )

    assert result.exit_code == 0, result.output
    assert "unit-test-openai-value" not in result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.auth.status.v1"
    assert payload["mode"] == "env_reference_only"
    assert payload["chat"]["provider"] == "openai"
    assert payload["chat"]["apiKeyEnv"] == "OPENAI_API_KEY"
    assert payload["chat"]["apiKeyStatus"] == "set"
    assert payload["codex"]["authType"] == "codex_cli_delegated_oauth"
    assert payload["codex"]["status"] in {"delegated_available", "delegated_unavailable"}
    assert payload["codex"]["directTokenReuse"] is False
    assert payload["codex"]["secretRead"] is False
    assert payload["codex"]["delegatedBackend"]["provider"] == "codex"
    assert payload["secretPolicy"]["rawKeyAccepted"] is False
    assert payload["secretPolicy"]["storedSecret"] is False


def test_auth_login_stores_env_reference_not_raw_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unit-test-anthropic-value")
    config = _config(tmp_path)

    result = CliRunner().invoke(
        auth_group,
        ["login", "anthropic", "--env", "ANTHROPIC_API_KEY", "--json"],
        obj={"khub": SimpleNamespace(config=config)},
    )

    assert result.exit_code == 0, result.output
    assert "unit-test-anthropic-value" not in result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.auth.login.result.v1"
    assert payload["provider"] == "anthropic"
    assert payload["apiKeyEnv"] == "ANTHROPIC_API_KEY"
    assert payload["apiKeyStatus"] == "set"
    assert payload["storedSecret"] is False
    assert payload["rawKeyAccepted"] is False
    assert config.get_nested("providers", "anthropic", "api_key_env") == "ANTHROPIC_API_KEY"
    saved = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "api_key_env: ANTHROPIC_API_KEY" in saved
    assert "unit-test-anthropic-value" not in saved


def test_auth_codex_status_does_not_read_or_claim_codex_connection(monkeypatch):
    monkeypatch.setenv("CODEX_FAKE_SECRET", "codex-secret-value")

    result = CliRunner().invoke(auth_group, ["codex", "--json"])

    assert result.exit_code == 0, result.output
    assert "codex-secret-value" not in result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.auth.status.v1"
    assert payload["codex"]["status"] in {"delegated_available", "delegated_unavailable"}
    assert payload["codex"]["connected"] is False
    assert payload["codex"]["secretRead"] is False
    assert payload["codex"]["secretPathRead"] is False
    assert payload["codex"]["directTokenReuse"] is False
    assert payload["codex"]["delegatedBackend"]["secretRead"] is False
