"""
프로바이더 레지스트리

설치된 프로바이더를 자동으로 감지하고, 설정에 따라 인스턴스를 반환합니다.
"""

from __future__ import annotations

from typing import Optional

from knowledge_hub.providers.base import BaseLLM, BaseEmbedder, ProviderInfo

_LLM_REGISTRY: dict[str, type[BaseLLM]] = {}
_EMBEDDER_REGISTRY: dict[str, type[BaseEmbedder]] = {}
_PROVIDER_INFO: dict[str, ProviderInfo] = {}


def _discover_providers():
    """설치된 프로바이더를 자동 탐색하여 레지스트리에 등록"""
    if _LLM_REGISTRY:
        return

    try:
        from knowledge_hub.providers.ollama import OllamaLLM, OllamaEmbedder
        _LLM_REGISTRY["ollama"] = OllamaLLM
        _EMBEDDER_REGISTRY["ollama"] = OllamaEmbedder
        _PROVIDER_INFO["ollama"] = OllamaLLM.provider_info()
    except ImportError:
        pass

    try:
        from knowledge_hub.providers.openai_provider import OpenAILLM, OpenAIEmbedder
        _LLM_REGISTRY["openai"] = OpenAILLM
        _EMBEDDER_REGISTRY["openai"] = OpenAIEmbedder
        _PROVIDER_INFO["openai"] = OpenAILLM.provider_info()
    except ImportError:
        pass

    try:
        from knowledge_hub.providers.anthropic_provider import AnthropicLLM
        _LLM_REGISTRY["anthropic"] = AnthropicLLM
        _PROVIDER_INFO["anthropic"] = AnthropicLLM.provider_info()
    except ImportError:
        pass

    try:
        from knowledge_hub.providers.google_provider import GoogleLLM, GoogleEmbedder
        _LLM_REGISTRY["google"] = GoogleLLM
        _EMBEDDER_REGISTRY["google"] = GoogleEmbedder
        _PROVIDER_INFO["google"] = GoogleLLM.provider_info()
    except ImportError:
        pass

    try:
        from knowledge_hub.providers.openai_compat import OpenAICompatLLM, OpenAICompatEmbedder
        _LLM_REGISTRY["openai-compat"] = OpenAICompatLLM
        _EMBEDDER_REGISTRY["openai-compat"] = OpenAICompatEmbedder
        _PROVIDER_INFO["openai-compat"] = OpenAICompatLLM.provider_info()
    except ImportError:
        pass


def get_llm(provider: str, model: Optional[str] = None, **kwargs) -> BaseLLM:
    """프로바이더 이름으로 LLM 인스턴스 생성"""
    _discover_providers()
    provider = provider.lower().strip()

    if provider not in _LLM_REGISTRY:
        available = ", ".join(_LLM_REGISTRY.keys()) or "(없음)"
        raise ValueError(
            f"LLM 프로바이더 '{provider}' 사용 불가. "
            f"설치된 프로바이더: {available}\n"
            f"설치: pip install knowledge-hub[{provider}]"
        )

    cls = _LLM_REGISTRY[provider]
    info = _PROVIDER_INFO.get(provider)
    if model is None and info:
        model = info.default_llm_model
    return cls(model=model or "", **kwargs)


def get_embedder(provider: str, model: Optional[str] = None, **kwargs) -> BaseEmbedder:
    """프로바이더 이름으로 Embedder 인스턴스 생성"""
    _discover_providers()
    provider = provider.lower().strip()

    if provider not in _EMBEDDER_REGISTRY:
        available = ", ".join(_EMBEDDER_REGISTRY.keys()) or "(없음)"
        raise ValueError(
            f"Embedder 프로바이더 '{provider}' 사용 불가. "
            f"설치된 프로바이더: {available}\n"
            f"설치: pip install knowledge-hub[{provider}]"
        )

    cls = _EMBEDDER_REGISTRY[provider]
    info = _PROVIDER_INFO.get(provider)
    if model is None and info:
        model = info.default_embed_model
    return cls(model=model or "", **kwargs)


def list_providers() -> dict[str, ProviderInfo]:
    """등록된 모든 프로바이더 정보 반환"""
    _discover_providers()
    return dict(_PROVIDER_INFO)


def get_provider_info(name: str) -> Optional[ProviderInfo]:
    """특정 프로바이더 정보 조회"""
    _discover_providers()
    return _PROVIDER_INFO.get(name.lower().strip())


def is_provider_available(name: str) -> bool:
    """프로바이더 사용 가능 여부"""
    _discover_providers()
    name = name.lower().strip()
    return name in _LLM_REGISTRY or name in _EMBEDDER_REGISTRY
