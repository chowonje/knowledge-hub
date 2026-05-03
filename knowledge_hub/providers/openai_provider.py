"""
OpenAI LLM/Embedder 프로바이더

pip install knowledge-hub[openai]
"""

from __future__ import annotations

import os
import logging
from typing import Generator, List, Optional

from knowledge_hub.providers.base import BaseLLM, BaseEmbedder, ProviderInfo
from knowledge_hub.providers.policy_guard import enforce_outbound_policy, evaluate_outbound_policy_batch

log = logging.getLogger("khub.providers.openai")


def _embed_blocked_with_local_fallback(texts: List[str]) -> List[Optional[List[float]]]:
    if not texts:
        return []
    try:
        from knowledge_hub.infrastructure.config import Config
        from knowledge_hub.infrastructure.providers import get_embedder, get_provider_info

        config = Config()
        provider = str(config.embedding_provider or "").strip()
        info = get_provider_info(provider)
        if not info or not info.is_local:
            return [None] * len(texts)
        embedder = get_embedder(provider, model=config.embedding_model, **config.get_provider_config(provider))
        return embedder.embed_batch(texts, show_progress=False)
    except Exception as error:
        log.warning("local embed fallback unavailable: %s", error)
        return [None] * len(texts)


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

    def _is_gpt5_model(self) -> bool:
        return str(self.model or "").startswith("gpt-5")

    def _gpt5_reasoning_effort(self) -> str:
        token = str(self.model or "").strip().lower()
        # Live API behavior differs across GPT-5 family variants; gpt-5.4 rejects
        # minimal, so keep the fast path for the smaller family and use low there.
        if token.startswith("gpt-5.4"):
            return "low"
        return "minimal"

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai 패키지 필요: pip install knowledge-hub[openai]")
        return self._client

    def _completion_token_kwargs(self, max_tokens: int | None = None) -> dict[str, int]:
        limit = int(max_tokens or self.max_tokens)
        if self._is_gpt5_model():
            return {"max_completion_tokens": limit}
        return {"max_tokens": limit}

    def _generation_kwargs(self, max_tokens: int | None = None) -> dict[str, int | float]:
        payload: dict[str, int | float] = dict(self._completion_token_kwargs(max_tokens))
        if not self._is_gpt5_model():
            payload["temperature"] = self.temperature
        return payload

    def _extract_response_text(self, response) -> str:  # noqa: ANN001
        output_text = str(getattr(response, "output_text", "") or "").strip()
        if output_text:
            return output_text
        segments: list[str] = []
        for item in list(getattr(response, "output", []) or []):
            for content in list(getattr(item, "content", []) or []):
                text_value = getattr(content, "text", None)
                if isinstance(text_value, str) and text_value.strip():
                    segments.append(text_value.strip())
        return "\n".join(segments).strip()

    def _generate_via_responses(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            instructions=f"참고 문서:\n{context}" if context else None,
            max_output_tokens=int(max_tokens or self.max_tokens),
            reasoning={"effort": self._gpt5_reasoning_effort()},
        )
        return self._extract_response_text(response)

    def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        decision = enforce_outbound_policy(provider="openai", model=self.model, prompt=prompt, context=context)
        self.last_policy = decision.to_dict()
        if decision.classification == "P1":
            log.warning("Provider outbound warning trace_id=%s warnings=%s", decision.trace_id, decision.warnings)
        if self._is_gpt5_model():
            return self._generate_via_responses(prompt, context=context, max_tokens=max_tokens)
        messages = []
        if context:
            messages.append({"role": "system", "content": f"참고 문서:\n{context}"})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self._generation_kwargs(max_tokens),
        )
        return response.choices[0].message.content or ""

    def stream_generate(self, prompt: str, context: str = "") -> Generator[str, None, None]:
        decision = enforce_outbound_policy(provider="openai", model=self.model, prompt=prompt, context=context)
        self.last_policy = decision.to_dict()
        if decision.classification == "P1":
            log.warning("Provider outbound warning trace_id=%s warnings=%s", decision.trace_id, decision.warnings)
        if self._is_gpt5_model():
            text = self._generate_via_responses(prompt, context=context, max_tokens=self.max_tokens)
            if text:
                yield text
            return
        messages = []
        if context:
            messages.append({"role": "system", "content": f"참고 문서:\n{context}"})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self._generation_kwargs(self.max_tokens),
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
                "gpt-5-mini", "gpt-5", "gpt-5.4", "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-4.1-nano",
                "o1", "o1-mini", "o1-pro", "o3", "o3-mini", "o4-mini",
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
        decision = enforce_outbound_policy(provider="openai", model=self.model, prompt=text, context="")
        self.last_policy = decision.to_dict()
        if decision.classification == "P1":
            log.warning("Provider outbound warning trace_id=%s warnings=%s", decision.trace_id, decision.warnings)
        try:
            response = self.client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        except Exception:
            return self._embed_via_requests([text])[0]

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[Optional[List[float]]]:
        clean_pairs = [(idx, text) for idx, text in enumerate(texts) if text and text.strip()]
        if not clean_pairs:
            return [None] * len(texts)

        clean_texts = [text for _, text in clean_pairs]
        report = evaluate_outbound_policy_batch(provider="openai", model=self.model, texts=clean_texts)
        blocked_positions = set(report.blocked_indices)
        safe_texts = [text for pos, text in enumerate(clean_texts) if pos not in blocked_positions]
        self.last_policy = report.to_dict()
        if report.blocked_count or any("P1 warning" in warning for warning in report.warnings):
            log.warning(
                "Provider batch outbound trace_id=%s allowed=%s blocked=%s warnings=%s",
                report.trace_id,
                report.allowed_count,
                report.blocked_count,
                report.warnings,
            )

        safe_embeddings: List[Optional[List[float]]] = []
        if safe_texts:
            try:
                safe_embeddings = self._embed_via_requests(safe_texts)
            except Exception:
                response = self.client.embeddings.create(model=self.model, input=safe_texts)
                safe_embeddings = [e.embedding for e in sorted(response.data, key=lambda x: x.index)]

        blocked_texts = [text for pos, text in enumerate(clean_texts) if pos in blocked_positions]
        blocked_embeddings = _embed_blocked_with_local_fallback(blocked_texts)

        results: List[Optional[List[float]]] = [None] * len(texts)
        safe_idx = 0
        blocked_idx = 0
        for pos, (original_idx, _) in enumerate(clean_pairs):
            if pos in blocked_positions:
                results[original_idx] = blocked_embeddings[blocked_idx] if blocked_idx < len(blocked_embeddings) else None
                blocked_idx += 1
            else:
                results[original_idx] = safe_embeddings[safe_idx] if safe_idx < len(safe_embeddings) else None
                safe_idx += 1
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
