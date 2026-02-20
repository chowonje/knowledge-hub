"""
Anthropic Claude LLM 프로바이더

pip install knowledge-hub[anthropic]
"""

from __future__ import annotations

import os
from typing import Generator

from knowledge_hub.providers.base import BaseLLM, ProviderInfo


class AnthropicLLM(BaseLLM):
    """Anthropic Claude 모델"""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic 패키지 필요: pip install knowledge-hub[anthropic]")
        return self._client

    def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        system = ""
        if context:
            system = f"참고 문서:\n{context}"

        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system if system else "You are a helpful research assistant.",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return message.content[0].text

    def stream_generate(self, prompt: str, context: str = "") -> Generator[str, None, None]:
        system = ""
        if context:
            system = f"참고 문서:\n{context}"

        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system if system else "You are a helpful research assistant.",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        ) as stream:
            for text in stream.text_stream:
                yield text

    @classmethod
    def provider_info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="anthropic",
            display_name="Anthropic (Claude)",
            supports_llm=True,
            supports_embedding=False,
            requires_api_key=True,
            is_local=False,
            default_llm_model="claude-sonnet-4-20250514",
            available_models=[
                "claude-sonnet-4-20250514", "claude-opus-4-20250514",
                "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",
            ],
        )
