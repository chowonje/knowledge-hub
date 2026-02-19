"""
Google Gemini LLM/Embedder 프로바이더

pip install knowledge-hub[google]
"""

from __future__ import annotations

import os
from typing import Generator, List, Optional

from knowledge_hub.providers.base import BaseLLM, BaseEmbedder, ProviderInfo


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
                "gemini-2.0-flash", "gemini-2.5-pro", "gemini-2.5-flash",
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
        self._ensure_configured()
        import google.generativeai as genai
        result = genai.embed_content(model=f"models/{self.model}", content=text)
        return result["embedding"]

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[Optional[List[float]]]:
        self._ensure_configured()
        import google.generativeai as genai
        results = []
        for text in texts:
            try:
                result = genai.embed_content(model=f"models/{self.model}", content=text)
                results.append(result["embedding"])
            except Exception:
                results.append(None)
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
