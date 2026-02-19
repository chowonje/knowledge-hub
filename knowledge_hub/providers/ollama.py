"""
Ollama 로컬 LLM/Embedder 프로바이더

pip install knowledge-hub[ollama]
"""

from __future__ import annotations

from typing import Generator, List, Optional

from knowledge_hub.providers.base import BaseLLM, BaseEmbedder, ProviderInfo


class OllamaLLM(BaseLLM):
    """Ollama 로컬 LLM"""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise ImportError("ollama 패키지 필요: pip install knowledge-hub[ollama]")
        return self._client

    def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        full_prompt = prompt
        if context:
            full_prompt = (
                f"다음은 관련 문서들입니다:\n\n{context}\n\n---\n\n"
                f"위 문서들을 참고하여 다음 질문에 답변해주세요:\n{prompt}\n\n"
                f"답변 시 문서의 내용을 직접 인용하면서 설명해주세요."
            )
        response = self.client.generate(
            model=self.model,
            prompt=full_prompt,
            options={"temperature": self.temperature, "num_predict": max_tokens or self.max_tokens},
        )
        return response["response"]

    def stream_generate(self, prompt: str, context: str = "") -> Generator[str, None, None]:
        full_prompt = prompt
        if context:
            full_prompt = (
                f"다음은 관련 문서들입니다:\n\n{context}\n\n---\n\n"
                f"위 문서들을 참고하여 다음 질문에 답변해주세요:\n{prompt}\n\n"
                f"답변 시 문서의 내용을 직접 인용하면서 설명해주세요."
            )
        stream = self.client.generate(
            model=self.model,
            prompt=full_prompt,
            stream=True,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )
        for chunk in stream:
            yield chunk["response"]

    @classmethod
    def provider_info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="ollama",
            display_name="Ollama (Local)",
            supports_llm=True,
            supports_embedding=True,
            requires_api_key=False,
            is_local=True,
            default_llm_model="qwen2.5:7b",
            default_embed_model="nomic-embed-text",
            available_models=[
                "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b",
                "llama3.3:latest", "gemma3:12b",
                "deepseek-r1:14b", "deepseek-r1:32b",
            ],
        )


class OllamaEmbedder(BaseEmbedder):
    """Ollama 로컬 임베딩"""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise ImportError("ollama 패키지 필요: pip install knowledge-hub[ollama]")
        return self._client

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("빈 텍스트는 임베딩할 수 없습니다")
        response = self.client.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[Optional[List[float]]]:
        from tqdm import tqdm
        results = []
        iterator = tqdm(texts, desc="임베딩 생성 중") if show_progress else texts
        for text in iterator:
            try:
                results.append(self.embed_text(text))
            except Exception:
                results.append(None)
        return results

    @classmethod
    def provider_info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="ollama",
            display_name="Ollama (Local)",
            supports_llm=False,
            supports_embedding=True,
            requires_api_key=False,
            is_local=True,
            default_embed_model="nomic-embed-text",
            available_models=["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
        )
