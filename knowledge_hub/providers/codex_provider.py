"""Delegated Codex CLI LLM provider.

This provider never reads Codex OAuth files or tokens. It delegates generation
to the installed `codex` CLI, which owns its own authentication state.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Generator

from knowledge_hub.application.codex_backend import CodexPromptLLM, codex_backend_readiness
from knowledge_hub.providers.base import BaseLLM, ProviderInfo
from knowledge_hub.providers.policy_guard import enforce_outbound_policy

log = logging.getLogger("khub.providers.codex")

DEFAULT_CODEX_CHAT_MODEL = "gpt-5.5"
CODEX_CHAT_MODELS = [
    "gpt-5.5",
    "gpt-5.4-codex",
    "gpt-5.4",
    "gpt-5.3-codex-spark",
    "gpt-5.3-codex",
    "gpt-5-codex",
]


class CodexDelegatedLLM(BaseLLM):
    """LLM wrapper around `codex exec` for assistant chat."""

    def __init__(
        self,
        model: str = DEFAULT_CODEX_CHAT_MODEL,
        *,
        config: Any | None = None,
        _khub_config: Any | None = None,
        task_type: str = "chat",
        **kwargs: Any,
    ):
        super().__init__(model or DEFAULT_CODEX_CHAT_MODEL, **kwargs)
        self.config = _khub_config or config or {}
        self.task_type = str(task_type or "chat")
        self.backend = CodexPromptLLM(config=self.config, model=self.model, task_type=self.task_type)
        self.last_policy: dict[str, Any] = {}
        self.last_response: dict[str, Any] = {}

    def _prompt(self, prompt: str, context: str) -> str:
        body = str(prompt or "").strip()
        ctx = str(context or "").strip()
        if not ctx:
            return body
        if not body:
            return ctx
        return f"{body}\n\nContext:\n{ctx}"

    def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        decision = enforce_outbound_policy(provider="codex", model=self.model, prompt=prompt, context=context)
        self.last_policy = decision.to_dict()
        if decision.classification == "P1":
            log.debug("Provider outbound warning trace_id=%s warnings=%s", decision.trace_id, decision.warnings)

        # Codex CLI manages auth and transport. khub only provides prompt/model.
        text = self.backend.generate(self._prompt(prompt, context), max_tokens=max_tokens)
        self.last_response = dict(self.backend.last_response or {})
        return text

    def stream_generate(self, prompt: str, context: str = "") -> Generator[str, None, None]:
        text = self.generate(prompt, context)
        for chunk in re.findall(r".{1,256}", text, flags=re.DOTALL):
            yield chunk

    @classmethod
    def provider_info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="codex",
            display_name="Codex CLI delegated OAuth",
            supports_llm=True,
            supports_embedding=False,
            requires_api_key=False,
            is_local=False,
            default_llm_model=DEFAULT_CODEX_CHAT_MODEL,
            available_models=list(CODEX_CHAT_MODELS),
        )


def codex_provider_status(config: Any, *, task_type: str = "chat") -> dict[str, Any]:
    """Return redacted delegated Codex provider readiness."""

    readiness = codex_backend_readiness(config, task_type=task_type)
    return {
        "provider": "codex",
        "authType": "codex_cli_delegated_oauth",
        "directTokenReuse": False,
        "secretRead": False,
        "secretPathRead": False,
        "available": bool(readiness.get("available", False)),
        "transport": str(readiness.get("transport") or ""),
        "command": str(readiness.get("command") or ""),
        "reason": str(readiness.get("reason") or ""),
        "summary": str(readiness.get("summary") or ""),
        "timeoutSeconds": int(readiness.get("timeoutSeconds") or 0),
    }
