from __future__ import annotations

from dataclasses import dataclass
import inspect
import importlib.util
import importlib.metadata
import os
import time
from typing import Any

from packaging.version import InvalidVersion, Version

from knowledge_hub.core.models import SearchResult

DEFAULT_RERANKER_MODEL = "cross-encoder/ettin-reranker-17m-v1"
DEFAULT_RERANKER_CANDIDATE_WINDOW = 8
DEFAULT_RERANKER_TIMEOUT_MS = 1200
DEFAULT_RERANKER_MAX_LENGTH = 512
RECOMMENDED_RERANKER_MODELS = (
    DEFAULT_RERANKER_MODEL,
    "cross-encoder/ettin-reranker-32m-v1",
    "cross-encoder/ettin-reranker-68m-v1",
    "BAAI/bge-reranker-v2-m3",
)
ETTIN_RERANKER_MIN_SENTENCE_TRANSFORMERS = "5.4.1"
ETTIN_RERANKER_MIN_TRANSFORMERS = "5.7.0"


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
    model: str = DEFAULT_RERANKER_MODEL
    candidate_window: int = DEFAULT_RERANKER_CANDIDATE_WINDOW
    timeout_ms: int = DEFAULT_RERANKER_TIMEOUT_MS
    fallback_on_error: bool = True
    allow_download: bool = False
    max_length: int = DEFAULT_RERANKER_MAX_LENGTH
    trust_remote_code: bool = False
    cache_folder: str = ""

    @classmethod
    def from_config(cls, config: Any) -> "RerankerConfig":
        if isinstance(config, cls):
            return config
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
                    default=DEFAULT_RERANKER_MODEL,
                )
                or DEFAULT_RERANKER_MODEL
            ).strip(),
            candidate_window=max(
                1,
                _safe_int(
                    config.get_nested(
                        "labs",
                        "retrieval",
                        "reranker",
                        "candidate_window",
                        default=DEFAULT_RERANKER_CANDIDATE_WINDOW,
                    ),
                    DEFAULT_RERANKER_CANDIDATE_WINDOW,
                ),
            ),
            timeout_ms=max(
                1,
                _safe_int(
                    config.get_nested(
                        "labs",
                        "retrieval",
                        "reranker",
                        "timeout_ms",
                        default=DEFAULT_RERANKER_TIMEOUT_MS,
                    ),
                    DEFAULT_RERANKER_TIMEOUT_MS,
                ),
            ),
            fallback_on_error=_safe_bool(
                config.get_nested("labs", "retrieval", "reranker", "fallback_on_error", default=True),
                True,
            ),
            allow_download=_safe_bool(
                config.get_nested("labs", "retrieval", "reranker", "allow_download", default=False),
                False,
            ),
            max_length=max(
                1,
                _safe_int(
                    config.get_nested(
                        "labs",
                        "retrieval",
                        "reranker",
                        "max_length",
                        default=DEFAULT_RERANKER_MAX_LENGTH,
                    ),
                    DEFAULT_RERANKER_MAX_LENGTH,
                ),
            ),
            trust_remote_code=_safe_bool(
                config.get_nested("labs", "retrieval", "reranker", "trust_remote_code", default=False),
                False,
            ),
            cache_folder=str(
                config.get_nested("labs", "retrieval", "reranker", "cache_folder", default="")
                or ""
            ).strip(),
        )

    def cache_key(self) -> tuple[Any, ...]:
        return (
            self.model,
            self.allow_download,
            self.max_length,
            self.trust_remote_code,
            self.cache_folder,
        )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _package_version_at_least(package: str, minimum: str) -> bool | None:
    try:
        current = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        return Version(current) >= Version(minimum)
    except InvalidVersion:
        return None


def _is_ettin_reranker(model: str) -> bool:
    return str(model or "").strip().startswith("cross-encoder/ettin-reranker-")


def _model_config_cached(model: str) -> bool | None:
    try:
        from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache
    except Exception:
        return None
    try:
        cached = try_to_load_from_cache(str(model or "").strip(), "config.json")
    except Exception:
        return None
    if cached is None or cached is _CACHED_NO_EXIST:
        return False
    return True


def _prepare_cross_encoder_import_env() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def reranker_runtime_status(config: Any) -> dict[str, Any]:
    runtime = RerankerConfig.from_config(config)
    sentence_transformers_available = _module_available("sentence_transformers")
    model_config_cached = _model_config_cached(runtime.model)
    sentence_transformers_version_ok = True
    transformers_version_ok = True
    if _is_ettin_reranker(runtime.model):
        sentence_transformers_version_ok = bool(
            _package_version_at_least(
                "sentence-transformers",
                ETTIN_RERANKER_MIN_SENTENCE_TRANSFORMERS,
            )
        )
        transformers_version_ok = bool(
            _package_version_at_least(
                "transformers",
                ETTIN_RERANKER_MIN_TRANSFORMERS,
            )
        )
    reasons: list[str] = []
    if not runtime.enabled:
        reasons.append("disabled")
    if not sentence_transformers_available:
        reasons.append("sentence_transformers_missing")
    if not sentence_transformers_version_ok:
        reasons.append("sentence_transformers_version_too_old")
    if not transformers_version_ok:
        reasons.append("transformers_version_too_old")
    if not runtime.allow_download and model_config_cached is False:
        reasons.append("model_not_cached")
    if not runtime.allow_download and model_config_cached is None:
        reasons.append("model_cache_unknown")
    ready = bool(
        sentence_transformers_available
        and sentence_transformers_version_ok
        and transformers_version_ok
        and (bool(runtime.allow_download) or model_config_cached is True)
    )
    return {
        "enabled": bool(runtime.enabled),
        "model": str(runtime.model),
        "candidate_window": int(runtime.candidate_window),
        "timeout_ms": int(runtime.timeout_ms),
        "fallback_on_error": bool(runtime.fallback_on_error),
        "allow_download": bool(runtime.allow_download),
        "local_files_only": not bool(runtime.allow_download),
        "max_length": int(runtime.max_length),
        "trust_remote_code": bool(runtime.trust_remote_code),
        "cache_folder": str(runtime.cache_folder),
        "model_config_cached": model_config_cached,
        "sentence_transformers_version_ok": sentence_transformers_version_ok,
        "transformers_version_ok": transformers_version_ok,
        "min_sentence_transformers_version": (
            ETTIN_RERANKER_MIN_SENTENCE_TRANSFORMERS if _is_ettin_reranker(runtime.model) else ""
        ),
        "min_transformers_version": (
            ETTIN_RERANKER_MIN_TRANSFORMERS if _is_ettin_reranker(runtime.model) else ""
        ),
        "recommended_models": list(RECOMMENDED_RERANKER_MODELS),
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
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model = config.model
        self._cross_encoder = None

    @property
    def client(self):
        if self._cross_encoder is None:
            _prepare_cross_encoder_import_env()
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError("sentence-transformers 패키지 필요: pip install 'knowledge-hub-cli[st]'") from exc
            signature = inspect.signature(CrossEncoder)
            supported = set(signature.parameters)
            if not self.config.allow_download and "local_files_only" not in supported:
                raise RuntimeError("sentence-transformers CrossEncoder runtime cannot enforce local_files_only")
            kwargs: dict[str, Any] = {}
            if "local_files_only" in supported:
                kwargs["local_files_only"] = not bool(self.config.allow_download)
            if "max_length" in supported:
                kwargs["max_length"] = int(self.config.max_length)
            if "trust_remote_code" in supported:
                kwargs["trust_remote_code"] = bool(self.config.trust_remote_code)
            if "cache_folder" in supported and self.config.cache_folder:
                kwargs["cache_folder"] = self.config.cache_folder
            self._cross_encoder = CrossEncoder(self.model, **kwargs)
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
    if not reranker_runtime_status(config).get("ready"):
        return None
    return SentenceTransformerReranker(config)


__all__ = [
    "DEFAULT_RERANKER_CANDIDATE_WINDOW",
    "DEFAULT_RERANKER_MAX_LENGTH",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_RERANKER_TIMEOUT_MS",
    "ETTIN_RERANKER_MIN_SENTENCE_TRANSFORMERS",
    "ETTIN_RERANKER_MIN_TRANSFORMERS",
    "RECOMMENDED_RERANKER_MODELS",
    "RerankerConfig",
    "RerankerExecution",
    "SentenceTransformerReranker",
    "build_reranker",
    "reranker_runtime_status",
]
