"""Helpers for user-defined provider aliases."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

from knowledge_hub.providers.base import ProviderInfo

OPENAI_COMPATIBLE_ADAPTERS = {
    "openai-compatible",
    "openai_compatible",
    "openai-compat",
    "openai_compat",
}

CUSTOM_PROVIDER_META_KEYS = {
    "adapter",
    "api_key_env",
    "display_name",
    "supports",
    "supports_llm",
    "supports_embedding",
    "is_local",
    "local",
    "no_api_key",
    "requires_api_key",
    "default_llm_model",
    "default_embed_model",
    "llm_model",
    "embedding_model",
    "models",
    "region",
    "data_boundary",
}


def normalize_adapter(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def is_openai_compatible_config(config: dict[str, Any] | None) -> bool:
    return normalize_adapter((config or {}).get("adapter")) in OPENAI_COMPATIBLE_ADAPTERS


def _bool_value(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _model_list(config: dict[str, Any], key: str) -> list[str]:
    models = config.get("models") if isinstance(config.get("models"), dict) else {}
    values = models.get(key) or models.get("embedding" if key == "embed" else key) or []
    if isinstance(values, str):
        values = [values]
    return [str(item).strip() for item in values if str(item).strip()]


def _base_url_is_local(base_url: str) -> bool:
    host = (urlsplit(str(base_url or "")).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def custom_provider_info(name: str, config: dict[str, Any] | None) -> ProviderInfo | None:
    cfg = dict(config or {})
    if not is_openai_compatible_config(cfg):
        return None

    supports = cfg.get("supports") if isinstance(cfg.get("supports"), dict) else {}
    llm_models = _model_list(cfg, "llm")
    embed_models = _model_list(cfg, "embed")
    default_llm_model = str(cfg.get("default_llm_model") or cfg.get("llm_model") or (llm_models[0] if llm_models else "")).strip()
    default_embed_model = str(
        cfg.get("default_embed_model") or cfg.get("embedding_model") or (embed_models[0] if embed_models else "")
    ).strip()

    supports_llm = _bool_value(
        cfg.get("supports_llm", supports.get("llm")),
        default=bool(default_llm_model or llm_models or not embed_models),
    )
    supports_embedding = _bool_value(
        cfg.get("supports_embedding", supports.get("embedding")),
        default=bool(default_embed_model or embed_models),
    )
    is_local = _bool_value(
        cfg.get("is_local", cfg.get("local")),
        default=_base_url_is_local(str(cfg.get("base_url") or "")),
    )
    no_api_key = _bool_value(cfg.get("no_api_key"), default=False)
    requires_api_key = _bool_value(
        cfg.get("requires_api_key"),
        default=not is_local and not no_api_key,
    )

    return ProviderInfo(
        name=str(name).strip(),
        display_name=str(cfg.get("display_name") or name).strip(),
        supports_llm=supports_llm,
        supports_embedding=supports_embedding,
        requires_api_key=requires_api_key,
        is_local=is_local,
        default_llm_model=default_llm_model,
        default_embed_model=default_embed_model,
        available_models=list(dict.fromkeys([*llm_models, *embed_models])),
    )


def custom_provider_runtime_kwargs(name: str, config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    kwargs = {
        key: value
        for key, value in cfg.items()
        if key not in CUSTOM_PROVIDER_META_KEYS and value is not None
    }
    api_key_env = str(cfg.get("api_key_env") or "").strip()
    if api_key_env and not str(kwargs.get("api_key") or "").strip():
        kwargs["api_key"] = os.environ.get(api_key_env, "")
    kwargs["provider_name"] = str(name).strip()
    return kwargs


def configured_custom_provider_infos(config: Any) -> dict[str, ProviderInfo]:
    if config is None or not hasattr(config, "get_nested"):
        return {}
    try:
        providers = config.get_nested("providers", default={}) or {}
    except Exception:
        return {}
    if not isinstance(providers, dict):
        return {}
    infos: dict[str, ProviderInfo] = {}
    for name, provider_config in providers.items():
        if not isinstance(provider_config, dict):
            continue
        info = custom_provider_info(str(name), provider_config)
        if info is not None:
            infos[str(name)] = info
    return infos
