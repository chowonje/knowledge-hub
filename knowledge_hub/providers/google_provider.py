"""
Google Gemini LLM/Embedder 프로바이더

pip install knowledge-hub[google]
"""

from __future__ import annotations

import os
import logging
from typing import Generator, List, Optional

from knowledge_hub.providers.base import BaseLLM, BaseEmbedder, ProviderInfo
from knowledge_hub.providers.policy_guard import enforce_outbound_policy, evaluate_outbound_policy_batch

log = logging.getLogger("khub.providers.google")


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


class GoogleLLM(BaseLLM):
    """Google Gemini 모델"""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._model_instance = None

    @property
    def model_instance(self):
        if self._model_instance is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._model_instance = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError("google-generativeai 패키지 필요: pip install knowledge-hub[google]")
        return self._model_instance

    def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        decision = enforce_outbound_policy(provider="google", model=self.model, prompt=prompt, context=context)
        self.last_policy = decision.to_dict()
        if decision.classification == "P1":
            log.warning("Provider outbound warning trace_id=%s warnings=%s", decision.trace_id, decision.warnings)
        full_prompt = prompt
        if context:
            full_prompt = f"참고 문서:\n{context}\n\n---\n\n{prompt}"

        response = self.model_instance.generate_content(
            full_prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": max_tokens or self.max_tokens,
            },
        )
        return response.text

    def stream_generate(self, prompt: str, context: str = "") -> Generator[str, None, None]:
        decision = enforce_outbound_policy(provider="google", model=self.model, prompt=prompt, context=context)
        self.last_policy = decision.to_dict()
        if decision.classification == "P1":
            log.warning("Provider outbound warning trace_id=%s warnings=%s", decision.trace_id, decision.warnings)
        full_prompt = prompt
        if context:
            full_prompt = f"참고 문서:\n{context}\n\n---\n\n{prompt}"

        response = self.model_instance.generate_content(
            full_prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    @classmethod
    def provider_info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="google",
            display_name="Google (Gemini)",
            supports_llm=True,
            supports_embedding=True,
            requires_api_key=True,
            is_local=False,
            default_llm_model="gemini-2.0-flash",
            default_embed_model="text-embedding-004",
            available_models=[
                "gemini-2.0-flash", "gemini-2.0-flash-lite",
                "gemini-2.5-pro", "gemini-2.5-flash",
            ],
        )


class GoogleEmbedder(BaseEmbedder):
    """Google Gemini 임베딩"""

    def __init__(
        self,
        model: str = "text-embedding-004",
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self._configured = False

    def _ensure_configured(self):
        if not self._configured:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._configured = True
            except ImportError:
                raise ImportError("google-generativeai 패키지 필요: pip install knowledge-hub[google]")

    def embed_text(self, text: str) -> List[float]:
        decision = enforce_outbound_policy(provider="google", model=self.model, prompt=text, context="")
        self.last_policy = decision.to_dict()
        if decision.classification == "P1":
            log.warning("Provider outbound warning trace_id=%s warnings=%s", decision.trace_id, decision.warnings)
        self._ensure_configured()
        import google.generativeai as genai
        result = genai.embed_content(model=f"models/{self.model}", content=text)
        return result["embedding"]

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[Optional[List[float]]]:
        clean_pairs = [(idx, text) for idx, text in enumerate(texts) if text and text.strip()]
        clean_texts = [text for _, text in clean_pairs]
        report = evaluate_outbound_policy_batch(provider="google", model=self.model, texts=clean_texts)
        blocked_positions = set(report.blocked_indices)
        self.last_policy = report.to_dict()
        if report.blocked_count or any("P1 warning" in warning for warning in report.warnings):
            log.warning(
                "Provider batch outbound trace_id=%s allowed=%s blocked=%s warnings=%s",
                report.trace_id,
                report.allowed_count,
                report.blocked_count,
                report.warnings,
            )
        self._ensure_configured()
        import google.generativeai as genai
        results: List[Optional[List[float]]] = [None] * len(texts)
        blocked_texts = [text for pos, text in enumerate(clean_texts) if pos in blocked_positions]
        blocked_embeddings = _embed_blocked_with_local_fallback(blocked_texts)
        blocked_idx = 0
        for pos, (original_idx, text) in enumerate(clean_pairs):
            if pos in blocked_positions:
                results[original_idx] = blocked_embeddings[blocked_idx] if blocked_idx < len(blocked_embeddings) else None
                blocked_idx += 1
                continue
            try:
                result = genai.embed_content(model=f"models/{self.model}", content=text)
                results[original_idx] = result["embedding"]
            except Exception:
                results[original_idx] = None
        return results

    @classmethod
    def provider_info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="google",
            display_name="Google (Gemini)",
            supports_llm=False,
            supports_embedding=True,
            requires_api_key=True,
            is_local=False,
            default_embed_model="text-embedding-004",
            available_models=["text-embedding-004"],
        )
