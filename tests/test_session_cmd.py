from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from click.testing import CliRunner

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.interfaces.cli.commands.chat_cmd import chat_cmd


class _FakeLLM:
    def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        _ = (context, max_tokens)
        return f"fake response containing {prompt}"


class _FakeSearcher:
    def __init__(self):
        self.config = None
        self.sqlite_db = None

    def generate_answer(self, query: str, **_kwargs):
        return {
            "answer": f"paper answer containing {query}",
            "sources": [{"title": "Source Paper", "id": "paper-1"}],
            "citations": [{"sourceId": "paper-1"}],
            "warnings": [],
            "router": {"selected": {"route": "local", "provider": "fake-rag", "model": "fake-paper-model"}},
        }


class _FakeFactory:
    def __init__(self):
        self.searcher = _FakeSearcher()

    def get_searcher(self):
        return self.searcher


class _FakeKhub:
    def __init__(self, config):
        self.config = config
        self.factory = _FakeFactory()

    def build_llm(self, provider: str, model: str):
        _ = (provider, model)
        return _FakeLLM()


def _config(tmp_path: Path) -> Config:
    config_path = tmp_path / "config.yaml"
    sqlite_path = tmp_path / "knowledge.db"
    config_path.write_text(f"storage:\n  sqlite: {sqlite_path}\n", encoding="utf-8")
    return Config(str(config_path))


def _jsonl_events(session_dir: Path) -> list[dict]:
    paths = sorted(session_dir.glob("*.jsonl"))
    assert len(paths) == 1
    return [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()]


def _sqlite_events(session_dir: Path) -> list[dict]:
    db_path = session_dir / "sessions.sqlite"
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT payload_json FROM assistant_session_events ORDER BY id"
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def test_chat_without_save_session_does_not_create_session_store(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    monkeypatch.setenv("KHUB_SESSION_DIR", str(session_dir))
    khub = _FakeKhub(_config(tmp_path))

    result = CliRunner().invoke(
        chat_cmd,
        ["hello", "--provider", "fake", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["historyPersisted"] is False
    assert not session_dir.exists()


def test_chat_save_session_writes_metadata_only_sqlite_and_jsonl_mirror(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    monkeypatch.setenv("KHUB_SESSION_DIR", str(session_dir))
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-sensitive-value")
    khub = _FakeKhub(_config(tmp_path))

    result = CliRunner().invoke(
        chat_cmd,
        ["hello unit-test-sensitive-value", "--provider", "fake", "--save-session", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["historyPersisted"] is True
    assert payload["session"]["persisted"] is True
    assert payload["session"]["historyMode"] == "sqlite-redacted-events"
    events = _sqlite_events(session_dir)
    event_types = [event["type"] for event in events]
    assert event_types == ["session_start", "user_message", "assistant_message", "route_metadata", "session_end"]
    raw_sqlite = "\n".join(json.dumps(event, ensure_ascii=False) for event in events)
    assert "hello unit-test-sensitive-value" not in raw_sqlite
    assert "fake response containing" not in raw_sqlite
    assert "unit-test-sensitive-value" not in raw_sqlite
    assert events[1]["textHash"].startswith("sha256:")
    assert events[2]["answerHash"].startswith("sha256:")
    mirror_events = _jsonl_events(session_dir)
    raw_jsonl = "\n".join(json.dumps(event, ensure_ascii=False) for event in mirror_events)
    assert "hello unit-test-sensitive-value" not in raw_jsonl
    assert "fake response containing" not in raw_jsonl
    assert "unit-test-sensitive-value" not in raw_jsonl
