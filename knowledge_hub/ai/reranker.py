from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import time
from typing import Any

from knowledge_hub.core.models import SearchResult


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    candidate_window: int = 12
    timeout_ms: int = 1500
    fallback_on_error: bool = True

    @classmethod
    def from_config(cls, config: Any) -> "RerankerConfig":
        if config is None or not hasattr(config, "get_nested"):
            return cls()
        return cls(
            enabled=bool(config.get_nested("labs", "retrieval", "reranker", "enabled", default=False)),
            model=str(
                config.get_nested(
                    "labs",
                    "retrieval",
                    "reranker",
                    "model",
                    default="BAAI/bge-reranker-v2-m3",
                )
                or "BAAI/bge-reranker-v2-m3"
            ).strip(),
            candidate_window=max(
                1,
                _safe_int(
                    config.get_nested("labs", "retrieval", "reranker", "candidate_window", default=12),
                    12,
                ),
            ),
            timeout_ms=max(
                1,
                _safe_int(
                    config.get_nested("labs", "retrieval", "reranker", "timeout_ms", default=1500),
                    1500,
                ),
            ),
            fallback_on_error=_safe_bool(
                config.get_nested("labs", "retrieval", "reranker", "fallback_on_error", default=True),
                True,
            ),
        )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def reranker_runtime_status(config: Any) -> dict[str, Any]:
    runtime = RerankerConfig.from_config(config)
    sentence_transformers_available = _module_available("sentence_transformers")
    reasons: list[str] = []
    if not runtime.enabled:
        reasons.append("disabled")
    if not sentence_transformers_available:
        reasons.append("sentence_transformers_missing")
    ready = sentence_transformers_available
    return {
        "enabled": bool(runtime.enabled),
        "model": str(runtime.model),
        "candidate_window": int(runtime.candidate_window),
        "timeout_ms": int(runtime.timeout_ms),
        "fallback_on_error": bool(runtime.fallback_on_error),
        "available": bool(sentence_transformers_available),
        "ready": bool(ready),
        "reason": "ok" if ready else (reasons[-1] if reasons else "unknown"),
        "reasons": reasons,
    }


@dataclass(frozen=True)
class RerankerExecution:
    results: list[SearchResult]
    diagnostics: dict[str, Any]


class SentenceTransformerReranker:
    def __init__(self, model: str):
        self.model = model
        self._cross_encoder = None

    @property
    def client(self):
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError("sentence-transformers 패키지 필요: pip install 'knowledge-hub-cli[st]'") from exc
            self._cross_encoder = CrossEncoder(self.model)
        return self._cross_encoder

    def rerank(
        self,
        *,
        query: str,
        results: list[SearchResult],
        config: RerankerConfig,
    ) -> RerankerExecution:
        diagnostics: dict[str, Any] = {
            "rerankerApplied": False,
            "rerankerModel": self.model,
            "rerankerWindow": min(len(results), int(config.candidate_window)),
            "rerankerLatencyMs": 0,
            "rerankerFallbackUsed": False,
            "rerankerReason": "disabled",
        }
        if not config.enabled:
            return RerankerExecution(results=list(results), diagnostics=diagnostics)

        window = min(len(results), int(config.candidate_window))
        if window <= 1:
            diagnostics["rerankerReason"] = "insufficient_candidates"
            return RerankerExecution(results=list(results), diagnostics=diagnostics)

        started = time.perf_counter()
        try:
            pairs = [(query, item.document or "") for item in results[:window]]
            raw_scores = self.client.predict(pairs)
            latency_ms = int((time.perf_counter() - started) * 1000)
            diagnostics["rerankerLatencyMs"] = latency_ms
            if latency_ms > int(config.timeout_ms):
                diagnostics["rerankerFallbackUsed"] = True
                diagnostics["rerankerReason"] = "timeout"
                return RerankerExecution(results=list(results), diagnostics=diagnostics)

            scored_pairs = list(zip(results[:window], [float(score) for score in raw_scores], strict=False))
            scored_pairs.sort(
                key=lambda pair: (
                    pair[1],
                    float((pair[0].lexical_extras or {}).get("retrieval_sort_score", pair[0].score)),
                    pair[0].score,
                ),
                reverse=True,
            )
            reranked = [item for item, _ in scored_pairs] + list(results[window:])
            for rank, (item, score) in enumerate(scored_pairs, start=1):
                extras = dict(item.lexical_extras or {})
                ranking_signals = dict(extras.get("ranking_signals") or {})
                reranker_boost = max(0.0, min(0.08, (float(score) / 20.0)))
                item.score = max(0.0, min(1.0, float(item.score) + reranker_boost))
                extras["reranker_score"] = round(float(score), 6)
                extras["reranker_rank"] = rank
                extras["retrieval_sort_score"] = round(
                    float(extras.get("retrieval_sort_score", item.score)) + reranker_boost,
                    6,
                )
                ranking_signals["cross_encoder_reranker_score"] = round(float(score), 6)
                ranking_signals["cross_encoder_reranker_rank"] = rank
                ranking_signals["cross_encoder_reranker_boost"] = round(reranker_boost, 6)
                extras["ranking_signals"] = ranking_signals
                item.lexical_extras = extras
            diagnostics["rerankerApplied"] = True
            diagnostics["rerankerReason"] = "applied"
            return RerankerExecution(results=reranked, diagnostics=diagnostics)
        except Exception:
            diagnostics["rerankerLatencyMs"] = int((time.perf_counter() - started) * 1000)
            diagnostics["rerankerFallbackUsed"] = bool(config.fallback_on_error)
            diagnostics["rerankerReason"] = "error"
            if not config.fallback_on_error:
                raise
            return RerankerExecution(results=list(results), diagnostics=diagnostics)


def build_reranker(config: RerankerConfig) -> SentenceTransformerReranker | None:
    if not config.enabled:
        return None
    if not reranker_runtime_status(config).get("available"):
        return None
    return SentenceTransformerReranker(config.model)


__all__ = [
    "RerankerConfig",
    "RerankerExecution",
    "SentenceTransformerReranker",
    "build_reranker",
    "reranker_runtime_status",
]
