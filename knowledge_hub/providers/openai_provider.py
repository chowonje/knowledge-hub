"""
OpenAI LLM/Embedder 프로바이더

pip install knowledge-hub[openai]
"""

from __future__ import annotations

import os
from typing import Generator, List, Optional

from knowledge_hub.providers.base import BaseLLM, BaseEmbedder, ProviderInfo


class OpenAILLM(BaseLLM):
    """OpenAI GPT 모델"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai 패키지 필요: pip install knowledge-hub[openai]")
        return self._client

    def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        messages = []
        if context:
            messages.append({"role": "system", "content": f"참고 문서:\n{context}"})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def stream_generate(self, prompt: str, context: str = "") -> Generator[str, None, None]:
        messages = []
        if context:
            messages.append({"role": "system", "content": f"참고 문서:\n{context}"})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @classmethod
    def provider_info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="openai",
            display_name="OpenAI",
            supports_llm=True,
            supports_embedding=True,
            requires_api_key=True,
            is_local=False,
            default_llm_model="gpt-4o-mini",
            default_embed_model="text-embedding-3-small",
            available_models=[
                "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1",
                "o3-mini",
            ],
        )


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI 임베딩"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                import httpx
                self._client = OpenAI(
                    api_key=self.api_key,
                    timeout=httpx.Timeout(30.0, connect=10.0),
                )
            except ImportError:
                raise ImportError("openai 패키지 필요: pip install knowledge-hub[openai]")
        return self._client

    def _embed_via_requests(self, texts: List[str]) -> List[List[float]]:
        """openai 라이브러리 실패 시 requests로 직접 호출"""
        import requests
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "input": texts},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("빈 텍스트는 임베딩할 수 없습니다")
        try:
            response = self.client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        except Exception:
            return self._embed_via_requests([text])[0]

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[Optional[List[float]]]:
        clean_texts = [t for t in texts if t and t.strip()]
        if not clean_texts:
            return [None] * len(texts)

        try:
            embeddings = self._embed_via_requests(clean_texts)
        except Exception:
            response = self.client.embeddings.create(model=self.model, input=clean_texts)
            embeddings = [e.embedding for e in sorted(response.data, key=lambda x: x.index)]

        results = []
        clean_idx = 0
        for t in texts:
            if t and t.strip():
                results.append(embeddings[clean_idx] if clean_idx < len(embeddings) else None)
                clean_idx += 1
            else:
                results.append(None)
        return results

    @classmethod
    def provider_info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="openai",
            display_name="OpenAI",
            supports_llm=False,
            supports_embedding=True,
            requires_api_key=True,
            is_local=False,
            default_embed_model="text-embedding-3-small",
            available_models=["text-embedding-3-small", "text-embedding-3-large"],
        )
