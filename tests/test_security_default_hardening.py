"""Regression guards for the 2026-06-11 security default hardening tranche.

Covers two fail-closed defaults (the vault-writeback fail-closed fix lives in the
vault-write-hazards tranche, PR #218):
1. paper.memory.allow_external must default False (config + runtime fallback + CLI opt-in only)
2. Provider outbound policy guard must hard-deny when the caller set allow_external=False
"""

from types import SimpleNamespace

import pytest

from knowledge_hub.infrastructure.config import DEFAULT_CONFIG
from knowledge_hub.papers import memory_runtime
from knowledge_hub.providers.policy_guard import (
    OutboundPolicyError,
    enforce_outbound_policy,
    evaluate_outbound_policy,
    evaluate_outbound_policy_batch,
)


# ---------------------------------------------------------------------------
# 1. paper.memory.allow_external defaults
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, values: dict | None = None):
        self._values = values or {}

    def get_nested(self, *keys, default=None):
        return self._values.get(".".join(keys), default)

    def get_provider_config(self, provider):  # presence enables the routed path
        return {}


def test_default_config_paper_memory_allow_external_is_false():
    assert DEFAULT_CONFIG["paper"]["memory"]["allow_external"] is False


def test_memory_runtime_defaults_local_when_config_key_missing(monkeypatch):
    captured = {}

    def fake_get_llm_for_task(config, *, task_type, allow_external, **kwargs):
        captured["allow_external"] = allow_external
        return None, SimpleNamespace(model=""), []

    monkeypatch.setattr(memory_runtime, "get_llm_for_task", fake_get_llm_for_task)
    memory_runtime.build_paper_memory_builder(object(), config=_FakeConfig())
    assert captured["allow_external"] is False


def test_memory_runtime_explicit_opt_in_still_works(monkeypatch):
    captured = {}

    def fake_get_llm_for_task(config, *, task_type, allow_external, **kwargs):
        captured["allow_external"] = allow_external
        return None, SimpleNamespace(model=""), []

    monkeypatch.setattr(memory_runtime, "get_llm_for_task", fake_get_llm_for_task)
    memory_runtime.build_paper_memory_builder(object(), config=_FakeConfig(), allow_external=True)
    assert captured["allow_external"] is True


def test_paper_memory_cli_threads_allow_external(monkeypatch):
    from click.testing import CliRunner

    from knowledge_hub.interfaces.cli.commands import paper_memory_cmd

    captured = {}

    class _FakeBuilder:
        def build_and_store(self, *, paper_id):
            return {"paperId": paper_id}

    def fake_build(sqlite_db, *, config=None, allow_external=None, **kwargs):
        captured["allow_external"] = allow_external
        return _FakeBuilder()

    monkeypatch.setattr(paper_memory_cmd, "build_paper_memory_builder", fake_build)
    monkeypatch.setattr(paper_memory_cmd, "_validate_cli_payload", lambda *a, **k: None)
    monkeypatch.setattr(paper_memory_cmd, "_compact_item", lambda item: item)

    khub = SimpleNamespace(sqlite_db=lambda: object(), config=None)
    runner = CliRunner()

    result = runner.invoke(
        paper_memory_cmd.paper_memory_group,
        ["build", "--paper-id", "p1", "--json"],
        obj={"khub": khub},
    )
    assert result.exit_code == 0, result.output
    assert captured["allow_external"] is None  # config-driven (default False)

    result = runner.invoke(
        paper_memory_cmd.paper_memory_group,
        ["build", "--paper-id", "p1", "--allow-external", "--json"],
        obj={"khub": khub},
    )
    assert result.exit_code == 0, result.output
    assert captured["allow_external"] is True


# ---------------------------------------------------------------------------
# 2. outbound policy guard hard-deny on allow_external=False
# ---------------------------------------------------------------------------


def test_outbound_policy_denies_when_local_only():
    decision = evaluate_outbound_policy(
        provider="openai", model="gpt-x", prompt="ordinary text", allow_external=False
    )
    assert decision.allowed is False
    assert decision.rule == "deny_external_local_only"

    with pytest.raises(OutboundPolicyError):
        enforce_outbound_policy(
            provider="openai", model="gpt-x", prompt="ordinary text", allow_external=False
        )


def test_outbound_policy_legacy_none_keeps_previous_behavior():
    decision = evaluate_outbound_policy(provider="openai", model="gpt-x", prompt="ordinary text")
    assert decision.allowed is True


def test_outbound_policy_batch_denies_all_when_local_only():
    report = evaluate_outbound_policy_batch(
        provider="openai", model="emb", texts=["a", "b"], allow_external=False
    )
    assert report.blocked_count == 2
    assert report.allowed_count == 0
    assert report.blocked_indices == [0, 1]


def test_external_adapter_blocks_before_any_network():
    from knowledge_hub.providers.openai_provider import OpenAILLM

    llm = OpenAILLM(model="gpt-x", allow_external=False)
    with pytest.raises(OutboundPolicyError):
        llm.generate("ordinary text")  # raises at the guard, before any HTTP client use


def test_task_router_threads_allow_external_into_provider(monkeypatch):
    from knowledge_hub.learning import task_router

    captured = {}

    def fake_get_llm(provider, model=None, **kwargs):
        captured["provider"] = provider
        captured["allow_external"] = kwargs.get("allow_external")
        return SimpleNamespace(model=model)

    monkeypatch.setattr(task_router, "get_llm", fake_get_llm)
    monkeypatch.setattr(task_router, "provider_runtime_probe", lambda config, provider: None)

    config = _FakeConfig(
        {
            "routing.llm.tasks.defaults": {},
        }
    )
    llm, decision, warnings = task_router.get_llm_for_task(
        config, task_type="summarization", allow_external=False, query="q"
    )
    assert llm is not None
    assert captured["allow_external"] is False
    assert decision.allow_external_effective is False
