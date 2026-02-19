"""
AI 프로바이더 추상 베이스 클래스

모든 LLM/Embedder 프로바이더는 이 인터페이스를 구현합니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generator, List, Optional


@dataclass
class ProviderInfo:
    """프로바이더 메타데이터"""
    name: str
    display_name: str
    supports_llm: bool = True
    supports_embedding: bool = False
    requires_api_key: bool = False
    is_local: bool = False
    default_llm_model: str = ""
    default_embed_model: str = ""
    available_models: list[str] = field(default_factory=list)


class BaseLLM(ABC):
    """LLM 프로바이더 추상 클래스"""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self._kwargs = kwargs

    @abstractmethod
    def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        """프롬프트로 텍스트 생성"""
        ...

    def stream_generate(self, prompt: str, context: str = "") -> Generator[str, None, None]:
        """스트리밍 텍스트 생성 (기본: 일반 생성 fallback)"""
        yield self.generate(prompt, context)

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "ko") -> str:
        """텍스트 번역"""
        prompt = (
            f"Translate the following {source_lang} text to {target_lang}. "
            f"Output only the translation, no explanations.\n\n{text}"
        )
        return self.generate(prompt)

    def summarize(self, text: str, language: str = "ko", max_sentences: int = 5) -> str:
        """텍스트 요약 (간단 버전 — discover 파이프라인에서 사용)"""
        prompt = (
            f"다음 논문 내용을 {language}로 {max_sentences}문장 이내로 핵심만 요약해주세요. "
            f"불필요한 서론 없이 바로 요약 내용만 출력하세요.\n\n{text}"
        )
        return self.generate(prompt)

    def summarize_paper(self, text: str, title: str = "", language: str = "ko") -> str:
        """구조화된 논문 심층 요약 — Markdown 섹션별 분석 결과를 반환"""
        prompt = f"""당신은 AI/ML 논문 분석 전문가입니다.
아래 논문을 읽고 다음 형식으로 **{language}** 심층 요약을 작성하세요.
각 섹션을 빠짐없이 채우고, 구체적인 수치·모델명·데이터셋을 반드시 포함하세요.

### 한줄 요약
(이 논문이 해결하는 문제와 핵심 아이디어를 1문장으로)

### 핵심 기여 (Contributions)
- (이 논문이 기존 연구 대비 새롭게 제안/발견한 점 3~5개, 각각 1~2문장)

### 방법론 (Methodology)
- (제안하는 모델/알고리즘/프레임워크의 구조와 작동 원리를 단계별로 설명)
- (핵심 수식이나 아키텍처가 있다면 간략히 서술)

### 주요 실험 결과 (Key Results)
- (어떤 데이터셋/벤치마크에서 테스트했는지)
- (기존 SOTA 대비 성능 향상 수치 — 정확도, F1, BLEU 등 구체적 숫자)
- (Ablation study 핵심 발견)

### 한계 및 향후 과제
- (저자가 밝힌 한계점 또는 논문에서 발견되는 약점)
- (향후 연구 방향 제안)

### 읽을 가치 / 시사점
- (이 논문이 연구자/실무자에게 주는 인사이트)
- (어떤 상황에서 이 기법을 적용할 수 있는지)

---
논문 제목: {title}

{text}"""
        return self.generate(prompt, max_tokens=4000)

    @classmethod
    def provider_info(cls) -> ProviderInfo:
        """프로바이더 정보 반환 (서브클래스에서 오버라이드)"""
        return ProviderInfo(name="base", display_name="Base")


class BaseEmbedder(ABC):
    """임베딩 프로바이더 추상 클래스"""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self._kwargs = kwargs

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트를 벡터로 변환"""
        ...

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[Optional[List[float]]]:
        """여러 텍스트를 벡터로 일괄 변환 (기본: 순차 처리)"""
        results = []
        for text in texts:
            try:
                results.append(self.embed_text(text))
            except Exception:
                results.append(None)
        return results

    @classmethod
    def provider_info(cls) -> ProviderInfo:
        return ProviderInfo(name="base", display_name="Base", supports_llm=False, supports_embedding=True)
