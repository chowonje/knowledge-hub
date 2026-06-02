from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.interfaces.cli.commands.assistant_runtime import build_assist_usage
from knowledge_hub.interfaces.cli.commands.chat_cmd import chat_cmd


class _FakeLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        _ = (context, max_tokens)
        self.prompts.append(prompt)
        return f"fake response: {prompt}"


class _FakeSearcher:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.config = None
        self.sqlite_db = None

    def generate_answer(self, query: str, **kwargs):
        self.calls.append((query, kwargs))
        return {
            "answer": f"paper answer: {query}",
            "sources": [{"title": "Attention Is All You Need", "id": "1706.03762"}],
            "citations": [{"sourceId": "1706.03762"}],
            "warnings": ["paper warning"],
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


def test_chat_single_turn_json_uses_fake_llm(tmp_path):
    config = _config(tmp_path)
    khub = _FakeKhub(config)

    result = CliRunner().invoke(
        chat_cmd,
        ["hello", "--provider", "fake", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.chat.result.v1"
    assert payload["status"] == "ok"
    assert payload["mode"] == "single"
    assert payload["provider"] == "fake"
    assert payload["layerUsed"] == "interface"
    assert payload["route"] == "plain_llm"
    assert payload["providerApplied"] == "fake"
    assert payload["modelApplied"] == config.summarization_model
    assert payload["sessionId"].startswith("chat_")
    assert payload["turnId"] == "turn_0001"
    assert payload["externalCallAllowed"] is True
    assert payload["policyBlocked"] is False
    assert payload["promptChars"] == len("hello")
    assert "prompt" not in payload
    assert payload["answer"] == "fake response: hello"
    assert payload["historyPersisted"] is False
    assert khub.build_calls == [("fake", config.summarization_model)]


def test_chat_paper_slash_json_uses_paper_evidence_runtime_without_building_llm(tmp_path):
    config = _config(tmp_path)
    khub = _FakeKhub(config)

    result = CliRunner().invoke(
        chat_cmd,
        ["/paper transformer contributions", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.chat.result.v1"
    assert payload["status"] == "ok"
    assert payload["mode"] == "paper"
    assert payload["route"] == "paper"
    assert payload["layerUsed"] == "core"
    assert payload["providerApplied"] == "fake-rag"
    assert payload["modelApplied"] == "fake-paper-model"
    assert payload["sessionId"].startswith("chat_")
    assert payload["turnId"] == "turn_0001"
    assert payload["externalCallAllowed"] is False
    assert payload["policyBlocked"] is False
    assert payload["sourceType"] == "paper"
    assert payload["retrievalMode"] == "hybrid"
    assert payload["topK"] == 8
    assert payload["answer"] == "paper answer: transformer contributions"
    assert payload["sources"][0]["id"] == "1706.03762"
    assert payload["warnings"] == ["paper warning"]
    assert payload["assistUsage"]["paperEvidence"] == "used"
    assert payload["assistUsage"]["paperMemory"] == "off"
    assert payload["assistUsage"]["ontology"] == "not_used"
    assert payload["assistUsage"]["enrichRecommended"] is False
    assert payload["answerRouteApplied"] == "local"
    assert payload["answerProviderApplied"] == "fake-rag"
    assert payload["answerModelApplied"] == "fake-paper-model"
    assert payload["promptChars"] == len("/paper transformer contributions")
    assert payload["questionChars"] == len("transformer contributions")
    assert "prompt" not in payload
    assert "question" not in payload
    assert khub.build_calls == []
    assert khub.searcher.calls == [
        (
            "transformer contributions",
            {
                "top_k": 8,
                "source_type": "paper",
                "retrieval_mode": "hybrid",
                "alpha": 0.7,
                "allow_external": False,
                "memory_route_mode": "off",
                "paper_memory_mode": "off",
                "answer_route_override": None,
            },
        )
    ]


def test_assist_usage_marks_graph_signal_as_ontology_signal():
    payload = build_assist_usage(
        graph_query_signal={"kind": "relation", "confidence": 0.8},
        paper_evidence_used=True,
    )

    assert payload["paperEvidence"] == "used"
    assert payload["ontology"] == "signal_only"
    assert payload["cluster"] == "not_used"


def test_chat_repl_keeps_history_in_process_memory_only(tmp_path):
    config = _config(tmp_path)
    khub = _FakeKhub(config)

    result = CliRunner().invoke(
        chat_cmd,
        ["--provider", "fake"],
        input="hello\nagain\n/exit\n",
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    assert "khub chat (fake/" in result.output
    assert "fake response: hello" in result.output
    assert "fake response: Conversation so far:" in result.output
    assert khub.llm.prompts[0] == "hello"
    assert "Conversation so far:" in khub.llm.prompts[1]
    assert "User: hello" in khub.llm.prompts[1]
    assert "Assistant: fake response: hello" in khub.llm.prompts[1]


def test_chat_repl_handles_paper_slash_without_adding_to_plain_chat_history(tmp_path):
    khub = _FakeKhub(_config(tmp_path))

    result = CliRunner().invoke(
        chat_cmd,
        ["--provider", "fake"],
        input="/paper retrieval augmented generation\nhello\n/exit\n",
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    assert "paper answer: retrieval augmented generation" in result.output
    assert "Attention Is All You Need" in result.output
    assert "fake response: hello" in result.output
    assert khub.llm.prompts == ["hello"]
    assert khub.searcher.calls[0][0] == "retrieval augmented generation"


def test_chat_no_allow_external_blocks_external_provider_without_building_llm(tmp_path):
    config = _config(tmp_path)
    khub = _FakeKhub(config)

    result = CliRunner().invoke(
        chat_cmd,
        ["hello", "--provider", "openai", "--model", "gpt-5.4", "--no-allow-external", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "blocked"
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-5.4"
    assert payload["allowExternal"] is False
    assert payload["externalCallAllowed"] is False
    assert payload["policyBlocked"] is True
    assert payload["route"] == "plain_llm"
    assert payload["layerUsed"] == "interface"
    assert payload["providerApplied"] == "openai"
    assert payload["modelApplied"] == "gpt-5.4"
    assert payload["turnId"] == "turn_0001"
    assert payload["promptChars"] == len("hello")
    assert "prompt" not in payload
    assert "external provider blocked" in payload["warnings"][0]
    assert khub.build_calls == []


def test_chat_json_without_prompt_is_rejected(tmp_path):
    result = CliRunner().invoke(
        chat_cmd,
        ["--provider", "fake", "--json"],
        obj={"khub": SimpleNamespace(config=_config(tmp_path), build_llm=lambda *_args: _FakeLLM())},
    )

    assert result.exit_code != 0
    assert "--json requires PROMPT" in result.output


def test_chat_text_mode_prints_diagnostics_footer(tmp_path):
    config = _config(tmp_path)
    khub = _FakeKhub(config)

    result = CliRunner().invoke(
        chat_cmd,
        ["hello", "--provider", "fake"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    assert "fake response: hello" in result.output
    assert "diagnostics: layer=interface route=plain_llm provider=fake" in result.output
    assert "policyBlocked=False" in result.output
