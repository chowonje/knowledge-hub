from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from knowledge_hub.providers import codex_provider
from knowledge_hub.providers.codex_provider import CodexDelegatedLLM
from knowledge_hub.providers.policy_guard import OutboundPolicyError
from knowledge_hub.providers.registry import get_provider_info


def test_codex_provider_info_is_delegated_llm_without_api_key():
    info = get_provider_info("codex")

    assert info is not None
    assert info.supports_llm is True
    assert info.supports_embedding is False
    assert info.requires_api_key is False
    assert info.is_local is False
    assert info.default_llm_model == "gpt-5.5"


def test_codex_delegated_llm_uses_codex_backend_without_reading_secret(monkeypatch):
    calls = []

    def _fake_run_codex_tool_sync(**kwargs):
        calls.append(dict(kwargs))
        return {
            "isError": False,
            "threadId": "",
            "content": "codex delegated answer",
            "structuredContent": {"transport": "exec", "returncode": 0},
        }

    monkeypatch.setattr(
        "knowledge_hub.application.codex_backend.run_codex_tool_sync",
        _fake_run_codex_tool_sync,
    )
    config = SimpleNamespace(get_nested=lambda *_args, default=None: default)
    llm = CodexDelegatedLLM(model="gpt-5.5", config=config)

    answer = llm.generate("hello", context="")

    assert answer == "codex delegated answer"
    assert calls[0]["model"] == "gpt-5.5"
    assert calls[0]["sandbox"] == "read-only"
    assert calls[0]["approval_policy"] == "never"
    assert "hello" in calls[0]["prompt"]


def test_codex_delegated_llm_does_not_emit_policy_warning_to_chat(monkeypatch, caplog):
    calls = []

    class _Decision:
        classification = "P1"
        trace_id = "policy_test"
        warnings = ["P1 structured facts detected", "provider=codex", "model=gpt-5.5"]

        def to_dict(self):
            return {"classification": self.classification, "traceId": self.trace_id, "warnings": self.warnings}

    def _fake_policy(**kwargs):
        calls.append(dict(kwargs))
        return _Decision()

    def _fake_run_codex_tool_sync(**kwargs):
        _ = kwargs
        return {
            "isError": False,
            "threadId": "",
            "content": "codex delegated answer",
            "structuredContent": {"transport": "exec", "returncode": 0},
        }

    monkeypatch.setattr(codex_provider, "enforce_outbound_policy", _fake_policy)
    monkeypatch.setattr(
        "knowledge_hub.application.codex_backend.run_codex_tool_sync",
        _fake_run_codex_tool_sync,
    )
    config = SimpleNamespace(get_nested=lambda *_args, default=None: default)
    llm = CodexDelegatedLLM(model="gpt-5.5", config=config)

    with caplog.at_level(logging.WARNING):
        answer = llm.generate("hello", context="")

    assert answer == "codex delegated answer"
    assert calls[0]["provider"] == "codex"
    assert "Provider outbound warning" not in caplog.text


def test_codex_delegated_llm_blocks_p0_before_backend_call():
    config = SimpleNamespace(get_nested=lambda *_args, default=None: default)
    llm = CodexDelegatedLLM(model="gpt-5.5", config=config)
    calls = []

    def _fake_generate(*args, **kwargs):
        calls.append((args, kwargs))
        return "should not be called"

    llm.backend.generate = _fake_generate

    with pytest.raises(OutboundPolicyError) as exc:
        llm.generate("contact test@example.com", context="")

    assert exc.value.decision.classification == "P0"
    assert exc.value.decision.allowed is False
    assert calls == []
