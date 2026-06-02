from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

from click.testing import CliRunner

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.interfaces.cli.commands.tui_cmd import _run_loop, tui_cmd


class _FakeLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        _ = (context, max_tokens)
        self.prompts.append(prompt)
        return f"fake tui response: {prompt}"


class _FakeSearcher:
    def __init__(self):
        self.config = None
        self.sqlite_db = None
        self.calls: list[tuple[str, dict]] = []

    def generate_answer(self, query: str, **kwargs):
        self.calls.append((query, kwargs))
        return {
            "answer": f"paper tui answer: {query}",
            "sources": [{"title": "Attention Is All You Need", "id": "1706.03762"}],
            "citations": [{"sourceId": "1706.03762"}],
            "warnings": ["paper tui warning"],
            "router": {"selected": {"route": "local", "provider": "fake-rag", "model": "fake-paper-model"}},
        }


class _FakeFactory:
    def __init__(self, searcher: _FakeSearcher):
        self.searcher = searcher

    def get_searcher(self):
        return self.searcher


class _FakeKhub:
    def __init__(self, config):
        self.config = config
        self.llm = _FakeLLM()
        self.searcher = _FakeSearcher()
        self.factory = _FakeFactory(self.searcher)
        self.build_calls: list[tuple[str, str]] = []

    def build_llm(self, provider: str, model: str):
        self.build_calls.append((provider, model))
        return self.llm


def _config(tmp_path: Path) -> Config:
    path = tmp_path / "config.yaml"
    sqlite_path = tmp_path / "knowledge.db"
    path.write_text(f"storage:\n  sqlite: {sqlite_path}\n", encoding="utf-8")
    return Config(str(path))


def _sqlite_events(session_dir: Path) -> list[dict]:
    db_path = session_dir / "sessions.sqlite"
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT payload_json FROM assistant_session_events ORDER BY id").fetchall()
    return [json.loads(row[0]) for row in rows]


def test_tui_startup_screen_exits_without_calling_llm(tmp_path):
    khub = _FakeKhub(_config(tmp_path))

    result = CliRunner().invoke(
        tui_cmd,
        ["--provider", "fake"],
        input="/exit\n",
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    assert "Knowledge Hub TUI" in result.output
    assert "Model" in result.output
    assert "fake/" in result.output
    assert "/auth" in result.output
    assert "/models" in result.output
    assert "/paper" in result.output
    assert "memory-only" in result.output
    assert khub.build_calls == []


def test_tui_plain_chat_uses_memory_only_history(tmp_path):
    khub = _FakeKhub(_config(tmp_path))

    result = CliRunner().invoke(
        tui_cmd,
        ["--provider", "fake"],
        input="hello\nagain\n/exit\n",
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    assert "fake tui response: hello" in result.output
    assert "fake tui response: Conversation so far:" in result.output
    assert khub.llm.prompts[0] == "hello"
    assert "Conversation so far:" in khub.llm.prompts[1]
    assert "User: hello" in khub.llm.prompts[1]
    assert "Assistant: fake tui response: hello" in khub.llm.prompts[1]


def test_tui_models_auth_help_and_unknown_enrich_are_local_to_shell(tmp_path):
    khub = _FakeKhub(_config(tmp_path))

    result = CliRunner().invoke(
        tui_cmd,
        ["--provider", "fake"],
        input="/commands\n/auth\n/model\n/enrich\n/chat\n/exit\n",
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    assert "TUI Commands" in result.output
    assert "Codex OAuth" in result.output
    assert "directTokenReuse" in result.output
    assert "Chat Model" in result.output
    assert "apiKeyStatus" in result.output
    assert "compat alias for /paper" in result.output
    assert "unknown command" in result.output
    assert "chat -> fake/" in result.output
    assert khub.build_calls == []


def test_tui_auth_login_and_model_use_update_session_config_without_printing_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-sensitive-value")
    config = _config(tmp_path)
    khub = _FakeKhub(config)

    result = CliRunner().invoke(
        tui_cmd,
        ["--provider", "fake"],
        input="/auth login openai --env OPENAI_API_KEY\n/models use openai/gpt-5.4\n/model\n/chat\n/exit\n",
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    assert "unit-test-sensitive-value" not in result.output
    assert "api_key_env=OPENAI_API_KEY (set)" in result.output
    assert "routing.llm.tasks.chat -> openai/gpt-5.4" in result.output
    assert "chat -> openai/gpt-5.4" in result.output
    assert config.get_nested("providers", "openai", "api_key_env") == "OPENAI_API_KEY"
    assert config.get_nested("routing", "llm", "tasks", "chat", "provider") == "openai"
    assert config.get_nested("routing", "llm", "tasks", "chat", "model") == "gpt-5.4"
    assert khub.build_calls == []


def test_tui_model_select_prompts_for_provider_and_model(tmp_path):
    config = _config(tmp_path)
    khub = _FakeKhub(config)
    from knowledge_hub.interfaces.cli.commands.models_cmd import chat_model_options, chat_provider_options

    providers = chat_provider_options(config)
    provider_index = [item["name"] for item in providers].index("openai") + 1
    model_index = chat_model_options(config, "openai").index("gpt-5.4") + 1

    result = CliRunner().invoke(
        tui_cmd,
        ["--provider", "fake"],
        input=f"/model select\n{provider_index}\n{model_index}\n/model\n/exit\n",
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    assert "Choose Provider" in result.output
    assert "Choose Model (openai)" in result.output
    assert "routing.llm.tasks.chat -> openai/gpt-5.4" in result.output
    assert "provider" in result.output
    assert "openai" in result.output
    assert config.get_nested("routing", "llm", "tasks", "chat", "provider") == "openai"
    assert config.get_nested("routing", "llm", "tasks", "chat", "model") == "gpt-5.4"


def test_tui_reset_clears_in_memory_history(tmp_path):
    khub = _FakeKhub(_config(tmp_path))

    result = CliRunner().invoke(
        tui_cmd,
        ["--provider", "fake"],
        input="hello\n/reset\nagain\n/exit\n",
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    assert "chat history cleared" in result.output
    assert khub.llm.prompts == ["hello", "again"]


def test_tui_paper_and_ask_slash_use_paper_evidence_runtime(tmp_path):
    khub = _FakeKhub(_config(tmp_path))

    result = CliRunner().invoke(
        tui_cmd,
        ["--provider", "fake"],
        input="/paper transformer\n/ask rag\n/exit\n",
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    assert "paper tui answer: transformer" in result.output
    assert "paper tui answer: rag" in result.output
    assert "Paper Evidence" in result.output
    assert "Attention Is All You Need" in result.output
    assert "paper tui warning" in result.output
    assert khub.build_calls == []
    assert [call[0] for call in khub.searcher.calls] == ["transformer", "rag"]
    assert all(call[1]["source_type"] == "paper" for call in khub.searcher.calls)


def test_tui_no_allow_external_blocks_external_provider_without_building_llm(tmp_path):
    khub = _FakeKhub(_config(tmp_path))

    result = CliRunner().invoke(
        tui_cmd,
        ["--provider", "openai", "--model", "gpt-5.4", "--no-allow-external"],
        input="hello\n/exit\n",
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    assert "External" in result.output
    assert "blocked" in result.output
    assert "external provider blocked" in result.output
    assert khub.build_calls == []


def test_tui_save_session_writes_metadata_only_without_raw_prompt_or_answer(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    monkeypatch.setenv("KHUB_SESSION_DIR", str(session_dir))
    khub = _FakeKhub(_config(tmp_path))

    result = CliRunner().invoke(
        tui_cmd,
        ["--provider", "fake", "--save-session"],
        input="hello unit-test-sensitive-value\n/exit\n",
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    assert str(session_dir) not in result.output
    events = _sqlite_events(session_dir)
    event_types = [event["type"] for event in events]
    assert event_types == ["session_start", "user_message", "assistant_message", "route_metadata", "session_end"]
    raw_sqlite = "\n".join(json.dumps(event, ensure_ascii=False) for event in events)
    assert "hello unit-test-sensitive-value" not in raw_sqlite
    assert "fake tui response" not in raw_sqlite
    assert "unit-test-sensitive-value" not in raw_sqlite
    assert events[1]["rawContentStored"] is False
    assert events[1]["contentClassification"] == "UNKNOWN"
    assert events[1]["persistencePolicy"] == "metadata_only_fail_closed"
    assert events[2]["rawContentStored"] is False
    assert events[2]["contentClassification"] == "UNKNOWN"
    assert events[2]["persistencePolicy"] == "metadata_only_fail_closed"


def test_tui_keyboard_interrupt_exits_cleanly(tmp_path, monkeypatch):
    khub = _FakeKhub(_config(tmp_path))

    class _InterruptingStdin:
        def readline(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(sys, "stdin", _InterruptingStdin())

    _run_loop(
        khub,
        provider="fake",
        model="qwen3:14b",
        allow_external=True,
        paper_allow_external=None,
        provider_status={"name": "fake", "isLocal": False, "apiKeyStatus": "not_required"},
    )

    assert khub.build_calls == []
