"""
OpenAI-compatible API provider — works with any endpoint that implements
the OpenAI chat/completions and embeddings API format.

Confirmed compatible services:
  - DeepSeek API       (api.deepseek.com)
  - Groq               (api.groq.com)
  - Together AI        (api.together.xyz)
  - Fireworks AI       (api.fireworks.ai)
  - Mistral AI         (api.mistral.ai)
  - xAI (Grok)         (api.x.ai)
  - Perplexity         (api.perplexity.ai)
  - OpenRouter         (openrouter.ai/api)
  - vLLM / TGI         (localhost)
  - LM Studio          (localhost:1234)
  - Ollama /v1         (localhost:11434/v1)

Usage in config.yaml:
  translation:
    provider: openai-compat
    model: deepseek-chat
	  providers:
	    openai-compat:
	      base_url: https://api.deepseek.com/v1
	      api_key_env: DEEPSEEK_API_KEY
"""

from __future__ import annotations

import os
import logging
from typing import Generator, List, Optional

import requests

from knowledge_hub.providers.base import BaseLLM, BaseEmbedder, ProviderInfo
from knowledge_hub.providers.policy_guard import enforce_outbound_policy, evaluate_outbound_policy_batch

log = logging.getLogger("khub.providers.openai_compat")


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


KNOWN_SERVICES: dict[str, dict] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "llm_models": ["deepseek-chat", "deepseek-reasoner"],
        "embed_models": [],
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "llm_models": [
            "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
            "gemma2-9b-it", "mixtral-8x7b-32768",
            "deepseek-r1-distill-llama-70b",
        ],
        "embed_models": [],
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
        "llm_models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-R1",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
        ],
        "embed_models": [
            "togethercomputer/m2-bert-80M-8k-retrieval",
            "BAAI/bge-large-en-v1.5",
        ],
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env_key": "FIREWORKS_API_KEY",
        "llm_models": [
            "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "accounts/fireworks/models/qwen2p5-72b-instruct",
            "accounts/fireworks/models/deepseek-r1",
        ],
        "embed_models": [
            "nomic-ai/nomic-embed-text-v1.5",
        ],
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
        "llm_models": [
            "mistral-large-latest", "mistral-small-latest",
            "codestral-latest", "open-mistral-nemo",
        ],
        "embed_models": ["mistral-embed"],
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "env_key": "XAI_API_KEY",
        "llm_models": ["grok-3", "grok-3-mini", "grok-2"],
        "embed_models": [],
    },
    "perplexity": {
        "base_url": "https://api.perplexity.ai",
        "env_key": "PERPLEXITY_API_KEY",
        "llm_models": [
            "sonar-pro", "sonar", "sonar-reasoning-pro", "sonar-reasoning",
        ],
        "embed_models": [],
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "llm_models": [
            "openai/gpt-4o-mini", "anthropic/claude-sonnet-4",
            "google/gemini-2.5-flash", "deepseek/deepseek-r1",
            "meta-llama/llama-3.3-70b-instruct",
        ],
        "embed_models": [],
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "env_key": "",
        "llm_models": [],
        "embed_models": [],
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "env_key": "",
        "llm_models": [],
        "embed_models": [],
    },
}


class OpenAICompatLLM(BaseLLM):
    """Any OpenAI API-compatible LLM endpoint."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        provider_name: str = "openai-compat",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or self._resolve_api_key(base_url)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider_name = str(provider_name or "openai-compat").strip() or "openai-compat"

    @staticmethod
    def _resolve_api_key(base_url: str) -> str:
        for svc in KNOWN_SERVICES.values():
            if svc["base_url"].rstrip("/") in base_url and svc["env_key"]:
                val = os.getenv(svc["env_key"], "")
                if val:
                    return val
        return os.getenv("OPENAI_COMPAT_API_KEY", "")

    def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        decision = enforce_outbound_policy(provider=self.provider_name, model=self.model, prompt=prompt, context=context)
        self.last_policy = decision.to_dict()
        if decision.classification == "P1":
            log.warning("Provider outbound warning trace_id=%s warnings=%s", decision.trace_id, decision.warnings)
        messages = []
        if context:
            messages.append({"role": "system", "content": f"참고 문서:\n{context}"})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def stream_generate(self, prompt: str, context: str = "") -> Generator[str, None, None]:
        decision = enforce_outbound_policy(provider=self.provider_name, model=self.model, prompt=prompt, context=context)
        self.last_policy = decision.to_dict()
        if decision.classification == "P1":
            log.warning("Provider outbound warning trace_id=%s warnings=%s", decision.trace_id, decision.warnings)
        messages = []
        if context:
            messages.append({"role": "system", "content": f"참고 문서:\n{context}"})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
        import json
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if text.startswith("data: "):
                text = text[6:]
            if text.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(text)
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    yield delta
            except (json.JSONDecodeError, IndexError, KeyError):
                continue

    @classmethod
    def provider_info(cls) -> ProviderInfo:
        all_models = []
        for svc in KNOWN_SERVICES.values():
            all_models.extend(svc["llm_models"])
        return ProviderInfo(
            name="openai-compat",
            display_name="OpenAI-Compatible (DeepSeek/Groq/Together/Mistral/xAI/...)",
            supports_llm=True,
            supports_embedding=True,
            requires_api_key=True,
            is_local=False,
            default_llm_model="deepseek-chat",
            default_embed_model="",
            available_models=all_models,
        )


class OpenAICompatEmbedder(BaseEmbedder):
    """Any OpenAI API-compatible embeddings endpoint."""

    def __init__(
        self,
        model: str = "mistral-embed",
        base_url: str = "https://api.mistral.ai/v1",
        api_key: str | None = None,
        provider_name: str = "openai-compat",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or OpenAICompatLLM._resolve_api_key(base_url)
        self.provider_name = str(provider_name or "openai-compat").strip() or "openai-compat"

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("빈 텍스트는 임베딩할 수 없습니다")
        decision = enforce_outbound_policy(provider=self.provider_name, model=self.model, prompt=text, context="")
        self.last_policy = decision.to_dict()
        if decision.classification == "P1":
            log.warning("Provider outbound warning trace_id=%s warnings=%s", decision.trace_id, decision.warnings)
        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "input": [text]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[Optional[List[float]]]:
        clean_pairs = [(idx, text) for idx, text in enumerate(texts) if text and text.strip()]
        clean = [text for _, text in clean_pairs]
        if not clean:
            return [None] * len(texts)
        report = evaluate_outbound_policy_batch(provider=self.provider_name, model=self.model, texts=clean)
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

        safe_texts = [text for pos, text in enumerate(clean) if pos not in blocked_positions]
        embeddings: List[Optional[List[float]]] = []
        if safe_texts:
            resp = requests.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "input": safe_texts},
                timeout=60,
            )
            resp.raise_for_status()
            embeddings = [x["embedding"] for x in sorted(resp.json()["data"], key=lambda x: x["index"])]

        blocked_texts = [text for pos, text in enumerate(clean) if pos in blocked_positions]
        blocked_embeddings = _embed_blocked_with_local_fallback(blocked_texts)

        results: List[Optional[List[float]]] = [None] * len(texts)
        safe_idx = 0
        blocked_idx = 0
        for pos, (original_idx, _) in enumerate(clean_pairs):
            if pos in blocked_positions:
                results[original_idx] = blocked_embeddings[blocked_idx] if blocked_idx < len(blocked_embeddings) else None
                blocked_idx += 1
            else:
                results[original_idx] = embeddings[safe_idx] if safe_idx < len(embeddings) else None
                safe_idx += 1
        return results

    @classmethod
    def provider_info(cls) -> ProviderInfo:
        all_models = []
        for svc in KNOWN_SERVICES.values():
            all_models.extend(svc["embed_models"])
        return ProviderInfo(
            name="openai-compat",
            display_name="OpenAI-Compatible Embeddings",
            supports_llm=False,
            supports_embedding=True,
            requires_api_key=True,
            is_local=False,
            default_embed_model="mistral-embed",
            available_models=all_models,
        )
