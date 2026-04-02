"""
Perplexity Embeddings local provider via TEI (Text Embeddings Inference).

Expected server:
  - TEI endpoint: POST /embed
  - Optional OpenAI-compatible endpoint fallback: POST /v1/embeddings

Example config:
  embedding:
    provider: pplx-local
    model: perplexity-ai/pplx-embed-v1-0.6b
  providers:
    pplx-local:
      base_url: http://localhost:8080
      timeout: 60
"""

from __future__ import annotations

from typing import List, Optional

import requests

from knowledge_hub.providers.base import BaseEmbedder, ProviderInfo


class PPLXLocalEmbedder(BaseEmbedder):
    """Local embedding provider backed by a TEI server."""

    def __init__(
        self,
        model: str = "perplexity-ai/pplx-embed-v1-0.6b",
        base_url: str = "http://localhost:8080",
        timeout: int = 60,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)

    def _post_tei_embed(self, inputs: str | list[str]) -> list[list[float]]:
        """Call TEI /embed first; fallback to OpenAI-compatible embeddings path."""
        try:
            resp = requests.post(
                f"{self.base_url}/embed",
                json={"inputs": inputs, "model": self.model},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(inputs, str):
                # TEI single response can be a flat list
                return [data] if data and isinstance(data[0], (int, float)) else data
            return data
        except Exception:
            payload = {"model": self.model, "input": [inputs] if isinstance(inputs, str) else inputs}
            resp = requests.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            parsed = resp.json()
            return [item["embedding"] for item in sorted(parsed["data"], key=lambda x: x["index"])]

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("빈 텍스트는 임베딩할 수 없습니다")
        return self._post_tei_embed(text)[0]

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[Optional[List[float]]]:
        clean = [t for t in texts if t and t.strip()]
        if not clean:
            return [None] * len(texts)

        embeddings = self._post_tei_embed(clean)
        results, clean_idx = [], 0
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
            name="pplx-local",
            display_name="Perplexity Embeddings (Local/TEI)",
            supports_llm=False,
            supports_embedding=True,
            requires_api_key=False,
            is_local=True,
            default_embed_model="perplexity-ai/pplx-embed-v1-0.6b",
            available_models=[
                "perplexity-ai/pplx-embed-v1-0.6b",
                "perplexity-ai/pplx-embed-v1-4b",
                "perplexity-ai/pplx-embed-context-v1-0.6b",
                "perplexity-ai/pplx-embed-context-v1-4b",
            ],
        )
