from __future__ import annotations

from knowledge_hub.application import codex_backend


class _Config:
    def __init__(self, values: dict | None = None):
        self._values = values or {}

    def get_nested(self, *keys, default=None):  # noqa: ANN002, ANN003
        current = self._values
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current


def test_resolve_preferred_codex_backend_selects_configured_backend(monkeypatch):
    config = _Config(
        {
            "routing": {
                "llm": {
                    "tasks": {
                        "rag_answer": {
                            "preferred_backend": "codex_mcp",
                            "preferred_backend_model": "gpt-5.4-codex",
                        }
                    }
                }
            }
        }
    )
    monkeypatch.setattr(
        codex_backend,
        "codex_backend_readiness",
        lambda config, task_type="rag_answer": {  # noqa: ARG005
            "available": True,
            "provider": "codex_mcp",
            "transport": "exec",
            "command": "codex",
            "reason": "",
            "summary": "ready",
            "timeoutSeconds": 180,
        },
    )

    llm, decision, warnings = codex_backend.resolve_preferred_codex_backend(
        config=config,
        allow_external=True,
    )

    assert isinstance(llm, codex_backend.CodexPromptLLM)
    assert decision == {
        "route": "api",
        "provider": "codex_mcp",
        "model": "gpt-5.4-codex",
        "reasons": [
            "preferred_backend=codex_mcp",
            "transport=exec",
        ],
        "timeoutSec": 180,
        "fallbackUsed": False,
    }
    assert warnings == []


def test_resolve_preferred_codex_backend_skips_when_external_disabled():
    config = _Config(
        {
            "routing": {
                "llm": {
                    "tasks": {
                        "rag_answer": {
                            "preferred_backend": "codex_mcp",
                        }
                    }
                }
            }
        }
    )

    llm, decision, warnings = codex_backend.resolve_preferred_codex_backend(
        config=config,
        allow_external=False,
    )

    assert llm is None
    assert decision is None
    assert warnings == ["codex_mcp backend skipped: allow_external disabled"]


def test_resolve_preferred_codex_backend_surfaces_unavailable_warning(monkeypatch):
    monkeypatch.setattr(
        codex_backend,
        "codex_backend_readiness",
        lambda config, task_type="rag_answer": {  # noqa: ARG005
            "available": False,
            "provider": "codex_mcp",
            "transport": "exec",
            "command": "codex",
            "reason": "command_not_found",
            "summary": "codex command not found: codex",
        },
    )

    llm, decision, warnings = codex_backend.resolve_preferred_codex_backend(
        config=None,
        allow_external=True,
        force_route="codex",
    )

    assert llm is None
    assert decision is None
    assert warnings == ["codex_mcp backend unavailable: codex command not found: codex"]


def test_codex_prompt_llm_generate_uses_runner_and_sanitizes(monkeypatch):
    observed: dict[str, str] = {}

    def _fake_run(**kwargs):  # noqa: ANN003
        observed["prompt"] = kwargs["prompt"]
        observed["model"] = kwargs["model"]
        observed["task_type"] = kwargs["task_type"]
        return {
            "isError": False,
            "threadId": "thread-123",
            "content": "Codex answer [1]\n\n(Source 1)",
            "structuredContent": {"transport": "exec"},
        }

    monkeypatch.setattr(codex_backend, "run_codex_tool_sync", _fake_run)
    llm = codex_backend.CodexPromptLLM(config=object(), model="gpt-5.4-codex", task_type="rag_answer")

    text = llm.generate("Question", context="Grounded context")

    assert text == "Codex answer"
    assert observed["prompt"] == "Question\n\nContext:\nGrounded context"
    assert observed["model"] == "gpt-5.4-codex"
    assert observed["task_type"] == "rag_answer"
    assert llm.last_policy == {
        "provider": "codex_mcp",
        "transport": "exec",
        "threadId": "thread-123",
    }
