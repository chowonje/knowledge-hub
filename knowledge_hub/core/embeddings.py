"""
임베딩 & LLM 모듈 (레거시 호환 래퍼)

이 모듈은 이전 코드와의 호환성을 위해 유지됩니다.
실제 프로바이더 구현은 knowledge_hub.providers 패키지에 있습니다.

새 코드에서는 knowledge_hub.providers.registry 의
get_llm() / get_embedder()를 사용하세요.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class OllamaEmbedder:
    """Ollama 기반 텍스트 임베딩 생성기 (레거시 호환)"""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise ImportError(
                    "ollama 패키지가 필요합니다: pip install knowledge-hub[ollama]\n"
                    "또는 다른 프로바이더를 사용하세요: khub config set embedding.provider openai"
                )
        return self._client

    def check_available(self) -> bool:
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]
            return any(self.model in name for name in model_names)
        except Exception:
            return False

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("빈 텍스트는 임베딩할 수 없습니다")
        response = self.client.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    def embed_batch(self, texts: List[str], show_progress: bool = True) -> List[Optional[List[float]]]:
        embeddings = []
        try:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="임베딩 생성 중") if show_progress else texts
        except ImportError:
            iterator = texts
        for text in iterator:
            try:
                embeddings.append(self.embed_text(text))
            except Exception:
                embeddings.append(None)
        return embeddings

    def get_embedding_dimension(self) -> int:
        test_embedding = self.embed_text("test")
        return len(test_embedding)


class OllamaLLM:
    """Ollama LLM 응답 생성기 (레거시 호환)"""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        self.model = model
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
                raise ImportError(
                    "ollama 패키지가 필요합니다: pip install knowledge-hub[ollama]\n"
                    "또는 다른 프로바이더를 사용하세요: khub config set summarization.provider openai"
                )
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

    def stream_generate(self, prompt: str, context: str = ""):
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
