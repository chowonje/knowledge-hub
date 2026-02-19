"""
임베딩 & LLM 모듈

Ollama를 사용하여 로컬에서 텍스트 임베딩 생성 및 LLM 응답을 생성합니다.
"""

from typing import List, Optional
import ollama
from tqdm import tqdm


class OllamaEmbedder:
    """Ollama 기반 텍스트 임베딩 생성기"""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)

    def check_available(self) -> bool:
        """Ollama 서버와 모델이 사용 가능한지 확인"""
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]
            return any(self.model in name for name in model_names)
        except Exception:
            return False

    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트를 벡터로 변환"""
        if not text or not text.strip():
            raise ValueError("빈 텍스트는 임베딩할 수 없습니다")
        response = self.client.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    def embed_batch(self, texts: List[str], show_progress: bool = True) -> List[Optional[List[float]]]:
        """여러 텍스트를 벡터로 일괄 변환"""
        embeddings = []
        iterator = tqdm(texts, desc="임베딩 생성 중") if show_progress else texts
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
    """Ollama LLM 응답 생성기"""

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
        self.client = ollama.Client(host=base_url)

    def generate(self, prompt: str, context: str = "") -> str:
        """프롬프트와 컨텍스트를 기반으로 응답 생성"""
        full_prompt = f"""다음은 관련 문서들입니다:

{context}

---

위 문서들을 참고하여 다음 질문에 답변해주세요:
{prompt}

답변 시 문서의 내용을 직접 인용하면서 설명해주세요."""

        response = self.client.generate(
            model=self.model,
            prompt=full_prompt,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )
        return response["response"]

    def stream_generate(self, prompt: str, context: str = ""):
        """스트리밍 방식으로 응답 생성"""
        full_prompt = f"""다음은 관련 문서들입니다:

{context}

---

위 문서들을 참고하여 다음 질문에 답변해주세요:
{prompt}

답변 시 문서의 내용을 직접 인용하면서 설명해주세요."""

        stream = self.client.generate(
            model=self.model,
            prompt=full_prompt,
            stream=True,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )
        for chunk in stream:
            yield chunk["response"]
