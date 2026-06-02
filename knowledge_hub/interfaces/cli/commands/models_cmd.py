"""User-facing model setup commands for the future assistant chat surface."""

from __future__ import annotations

import os
import re
from typing import Any

import click
from rich.console import Console
from rich.table import Table

console = Console()

MODELS_RESULT_SCHEMA = "knowledge-hub.models.result.v1"
PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_API_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}
OPENAI_COMPAT_ADAPTER = "openai-compatible"


def _config_get(config: Any, *keys: str, default: Any = "") -> Any:
    getter = getattr(config, "get_nested", None)
    if callable(getter):
        return getter(*keys, default=default)
    return default


def _normalize_provider_name(value: str) -> str:
    name = str(value or "").strip().lower()
    if not PROVIDER_NAME_RE.match(name):
        raise click.BadParameter("provider name must use letters, numbers, dot, underscore, or hyphen")
    return name


def _split_provider_model(provider_model: str, model: str | None = None) -> tuple[str, str]:
    raw = str(provider_model or "").strip()
    if not raw:
        raise click.BadParameter("provider is required")
    if model:
        return _normalize_provider_name(raw), str(model).strip()
    if "/" in raw:
        provider, _, model_part = raw.partition("/")
        return _normalize_provider_name(provider), model_part.strip()
    return _normalize_provider_name(raw), ""


def _save(config: Any) -> None:
    saver = getattr(config, "save", None)
    if callable(saver):
        saver()


def _provider_info(config: Any, provider: str):
    from knowledge_hub.infrastructure.providers import get_provider_info

    try:
        return get_provider_info(provider, config=config)
    except TypeError:
        return get_provider_info(provider)


def _default_llm_model(config: Any, provider: str) -> str:
    info = _provider_info(config, provider)
    return str(getattr(info, "default_llm_model", "") or "") if info else ""


def _raw_provider_config(config: Any, provider: str) -> dict[str, Any]:
    raw = _config_get(config, "providers", provider, default={})
    return dict(raw) if isinstance(raw, dict) else {}


def _known_openai_compat_services() -> dict[str, dict[str, Any]]:
    from knowledge_hub.providers.openai_compat import KNOWN_SERVICES

    return {str(name): dict(service) for name, service in KNOWN_SERVICES.items()}


def _service_is_local(service: dict[str, Any]) -> bool:
    base_url = str(service.get("base_url") or "")
    return "localhost" in base_url or "127.0.0.1" in base_url or "0.0.0.0" in base_url


def _catalog_provider_payload(config: Any, provider: str, service: dict[str, Any]) -> dict[str, Any]:
    llm_models = [str(model) for model in service.get("llm_models", []) if str(model).strip()]
    embed_models = [str(model) for model in service.get("embed_models", []) if str(model).strip()]
    env_name = str(_raw_provider_config(config, provider).get("api_key_env") or service.get("env_key") or "").strip()
    is_local = _service_is_local(service)
    requires_api_key = bool(env_name) and not is_local
    if not requires_api_key:
        status = "not_required"
    elif os.environ.get(env_name):
        status = "set"
    else:
        status = "missing"
    display = {
        "xai": "xAI",
        "vllm": "vLLM",
        "lmstudio": "LM Studio",
    }.get(provider, provider.replace("-", " ").title())
    return {
        "name": provider,
        "displayName": f"{display} (OpenAI-compatible)",
        "supportsLLM": True,
        "supportsEmbedding": bool(embed_models),
        "requiresApiKey": requires_api_key,
        "isLocal": is_local,
        "defaultLLMModel": llm_models[0] if llm_models else "",
        "defaultEmbeddingModel": embed_models[0] if embed_models else "",
        "models": list(dict.fromkeys([*llm_models, *embed_models])),
        "apiKeyEnv": env_name,
        "apiKeyEnvSource": "catalog_default" if env_name else "",
        "apiKeyStatus": status,
        "inlineSecretPresent": False,
        "adapter": OPENAI_COMPAT_ADAPTER,
        "baseUrl": str(service.get("base_url") or ""),
        "catalog": "openai-compatible",
    }


def _set_nested_if_missing(config: Any, *keys_and_value: Any) -> None:
    *keys, value = keys_and_value
    current = _config_get(config, *[str(key) for key in keys], default=None)
    if current in (None, "", {}, []):
        config.set_nested(*[str(key) for key in keys], value)


def ensure_catalog_provider_config(config: Any, provider: str) -> bool:
    """Materialize a known OpenAI-compatible service as a custom provider alias."""
    provider_name = _normalize_provider_name(provider)
    service = _known_openai_compat_services().get(provider_name)
    if service is None:
        return False
    llm_models = [str(model) for model in service.get("llm_models", []) if str(model).strip()]
    embed_models = [str(model) for model in service.get("embed_models", []) if str(model).strip()]
    is_local = _service_is_local(service)
    env_name = str(service.get("env_key") or "").strip()

    _set_nested_if_missing(config, "providers", provider_name, "adapter", OPENAI_COMPAT_ADAPTER)
    _set_nested_if_missing(config, "providers", provider_name, "base_url", str(service.get("base_url") or ""))
    _set_nested_if_missing(config, "providers", provider_name, "display_name", _catalog_provider_payload(config, provider_name, service)["displayName"])
    _set_nested_if_missing(config, "providers", provider_name, "supports", "llm", True)
    _set_nested_if_missing(config, "providers", provider_name, "supports", "embedding", bool(embed_models))
    _set_nested_if_missing(config, "providers", provider_name, "is_local", is_local)
    _set_nested_if_missing(config, "providers", provider_name, "no_api_key", not bool(env_name))
    _set_nested_if_missing(config, "providers", provider_name, "requires_api_key", bool(env_name) and not is_local)
    if env_name:
        _set_nested_if_missing(config, "providers", provider_name, "api_key_env", env_name)
    if llm_models:
        _set_nested_if_missing(config, "providers", provider_name, "models", "llm", llm_models)
        _set_nested_if_missing(config, "providers", provider_name, "default_llm_model", llm_models[0])
    if embed_models:
        _set_nested_if_missing(config, "providers", provider_name, "models", "embed", embed_models)
        _set_nested_if_missing(config, "providers", provider_name, "default_embed_model", embed_models[0])
    return True


def _api_key_env(config: Any, provider: str) -> tuple[str, str]:
    cfg = _raw_provider_config(config, provider)
    configured = str(cfg.get("api_key_env") or "").strip()
    if configured:
        return configured, "config"
    default_env = DEFAULT_API_KEY_ENVS.get(provider, "")
    if default_env:
        return default_env, "provider_default"
    return "", ""


def _api_key_status(config: Any, provider: str, *, requires_api_key: bool) -> dict[str, str | bool]:
    cfg = _raw_provider_config(config, provider)
    env_name, env_source = _api_key_env(config, provider)
    inline_secret_present = bool(str(cfg.get("api_key") or "").strip())
    if not requires_api_key:
        status = "not_required"
    elif env_name and os.environ.get(env_name):
        status = "set"
    elif env_name:
        status = "missing"
    elif inline_secret_present:
        status = "inline_secret_present"
    else:
        status = "missing"
    return {
        "apiKeyEnv": env_name,
        "apiKeyEnvSource": env_source,
        "apiKeyStatus": status,
        "inlineSecretPresent": inline_secret_present,
    }


def _provider_payload(config: Any, provider: str) -> dict[str, Any]:
    info = _provider_info(config, provider)
    cfg = _raw_provider_config(config, provider)
    requires_api_key = bool(getattr(info, "requires_api_key", False))
    payload = {
        "name": provider,
        "displayName": getattr(info, "display_name", provider) if info else provider,
        "supportsLLM": bool(getattr(info, "supports_llm", False)),
        "supportsEmbedding": bool(getattr(info, "supports_embedding", False)),
        "requiresApiKey": requires_api_key,
        "isLocal": bool(getattr(info, "is_local", False)),
        "defaultLLMModel": str(getattr(info, "default_llm_model", "") or ""),
        "defaultEmbeddingModel": str(getattr(info, "default_embed_model", "") or ""),
        "models": list(getattr(info, "available_models", []) or []),
        "adapter": str(cfg.get("adapter") or "").strip(),
        "baseUrl": str(cfg.get("base_url") or "").strip(),
    }
    payload.update(_api_key_status(config, provider, requires_api_key=requires_api_key))
    return payload


def _chat_route(config: Any) -> dict[str, Any]:
    chat_provider = str(_config_get(config, "routing", "llm", "tasks", "chat", "provider", default="") or "").strip().lower()
    chat_model = str(_config_get(config, "routing", "llm", "tasks", "chat", "model", default="") or "").strip()
    summary_provider = str(
        getattr(config, "summarization_provider", "") or _config_get(config, "summarization", "provider", default="ollama") or "ollama"
    ).strip().lower()
    summary_model = str(
        getattr(config, "summarization_model", "") or _config_get(config, "summarization", "model", default="qwen3:14b") or "qwen3:14b"
    ).strip()
    provider = chat_provider or summary_provider
    model = chat_model if chat_provider else summary_model
    if chat_provider and not model:
        model = _default_llm_model(config, provider)
    return {
        "provider": provider,
        "model": model,
        "source": "routing.llm.tasks.chat" if chat_provider else "summarization",
        "target": "routing.llm.tasks.chat",
    }


def _list_providers(config: Any) -> dict[str, Any]:
    from knowledge_hub.infrastructure.providers import list_providers

    try:
        return list_providers(config=config)
    except TypeError:
        return list_providers()


def chat_provider_options(config: Any) -> list[dict[str, Any]]:
    """Return LLM-capable providers for interactive selection."""
    providers = _list_providers(config)
    options = [_provider_payload(config, name) for name in sorted(providers)]
    provider_names = {str(item.get("name") or "") for item in options}
    for name, service in sorted(_known_openai_compat_services().items()):
        if name not in provider_names:
            options.append(_catalog_provider_payload(config, name, service))
    return [item for item in options if item.get("supportsLLM")]


def chat_model_options(config: Any, provider: str) -> list[str]:
    """Return known model ids for a provider, default first when available."""
    provider_name = _normalize_provider_name(provider)
    payload = _provider_payload(config, provider_name)
    if not payload.get("models"):
        service = _known_openai_compat_services().get(provider_name)
        if service is not None:
            payload = _catalog_provider_payload(config, provider_name, service)
    models = [str(item) for item in payload.get("models", []) if str(item).strip()]
    default_model = str(payload.get("defaultLLMModel") or "").strip()
    ordered: list[str] = []
    if default_model:
        ordered.append(default_model)
    for model in models:
        if model not in ordered:
            ordered.append(model)
    return ordered


def _select_number(raw: str, *, max_value: int) -> int:
    value = str(raw or "").strip()
    try:
        number = int(value)
    except ValueError as exc:
        raise click.BadParameter("selection must be a number") from exc
    if number < 1 or number > max_value:
        raise click.BadParameter(f"selection must be between 1 and {max_value}")
    return number


def _print_provider_options(options: list[dict[str, Any]]) -> None:
    table = Table(title="Choose Provider")
    table.add_column("#", justify="right")
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Type")
    table.add_column("Default")
    table.add_column("Local", justify="center")
    table.add_column("API key")
    for index, item in enumerate(options, start=1):
        table.add_row(
            str(index),
            str(item.get("name") or "-"),
            str(item.get("catalog") or item.get("adapter") or "native"),
            str(item.get("defaultLLMModel") or "-"),
            "Y" if item.get("isLocal") else "-",
            str(item.get("apiKeyStatus") or "-"),
        )
    console.print(table)


def _print_model_options(provider: str, options: list[str]) -> None:
    table = Table(title=f"Choose Model ({provider})")
    table.add_column("#", justify="right")
    table.add_column("Model", style="cyan")
    for index, model in enumerate(options, start=1):
        table.add_row(str(index), model)
    console.print(table)


def set_chat_model_config(config: Any, provider_model: str, *, model: str | None = "") -> dict[str, Any]:
    """Set the chat model route and return a redacted payload."""
    provider, model_id = _split_provider_model(provider_model, model=model)
    ensure_catalog_provider_config(config, provider)
    if not model_id:
        model_id = _default_llm_model(config, provider)
    if not model_id:
        raise click.BadParameter("model is required when the provider has no default model")

    config.set_nested("routing", "llm", "tasks", "chat", "provider", provider)
    config.set_nested("routing", "llm", "tasks", "chat", "model", model_id)
    _save(config)
    return {
        "schema": MODELS_RESULT_SCHEMA,
        "status": "ok",
        "target": "routing.llm.tasks.chat",
        "provider": provider,
        "model": model_id,
        "nextCommands": ["khub models status"],
        "chatCommandStatus": "pending_chat_tranche",
    }


@click.group("models")
def models_group():
    """List and configure the user-facing chat model."""


@models_group.command("list")
@click.option("--models", is_flag=True, help="Show model inventory.")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def models_list(ctx, models, as_json):
    """List available providers and models."""
    config = ctx.obj["khub"].config
    providers = _list_providers(config)
    payload = {
        "schema": MODELS_RESULT_SCHEMA,
        "status": "ok",
        "providers": [_provider_payload(config, name) for name in sorted(providers)],
    }
    if as_json:
        console.print_json(data=payload)
        return

    table = Table(title="Models")
    table.add_column("Provider", style="cyan")
    table.add_column("LLM", justify="center")
    table.add_column("Local", justify="center")
    table.add_column("Default")
    table.add_column("API key")
    for item in payload["providers"]:
        table.add_row(
            item["name"],
            "Y" if item["supportsLLM"] else "-",
            "Y" if item["isLocal"] else "-",
            item["defaultLLMModel"] or "-",
            str(item["apiKeyStatus"]),
        )
        if models and item["models"]:
            table.add_row("", "", "", ", ".join(item["models"][:8]), "")
    console.print(table)


@models_group.command("status")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def models_status(ctx, as_json):
    """Show current chat model status without exposing secrets."""
    config = ctx.obj["khub"].config
    route = _chat_route(config)
    provider = str(route["provider"])
    provider_payload = _provider_payload(config, provider) if provider else {}
    payload = {
        "schema": MODELS_RESULT_SCHEMA,
        "status": "ok",
        "chat": {
            "target": route["target"],
            "source": route["source"],
            "provider": provider,
            "model": route["model"],
            "isLocal": bool(provider_payload.get("isLocal", False)),
            "requiresApiKey": bool(provider_payload.get("requiresApiKey", False)),
            "apiKeyEnv": provider_payload.get("apiKeyEnv", ""),
            "apiKeyEnvSource": provider_payload.get("apiKeyEnvSource", ""),
            "apiKeyStatus": provider_payload.get("apiKeyStatus", "missing"),
        },
        "provider": provider_payload,
    }
    if as_json:
        console.print_json(data=payload)
        return

    chat = payload["chat"]
    console.print(f"[cyan]chat[/cyan] -> {chat['provider']}/{chat['model']} ({chat['source']})")
    console.print(f"[dim]local={chat['isLocal']} api_key={chat['apiKeyStatus']} env={chat['apiKeyEnv'] or '-'}[/dim]")


@models_group.command("use")
@click.argument("provider_model")
@click.option("--model", default="", help="Model id when not using provider/model syntax.")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def models_use(ctx, provider_model, model, as_json):
    """Set routing.llm.tasks.chat without changing ask/summarization roles."""
    config = ctx.obj["khub"].config
    payload = set_chat_model_config(config, provider_model, model=model)
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[green]routing.llm.tasks.chat[/green] -> {payload['provider']}/{payload['model']}")


@models_group.command("select")
@click.option("--provider", default="", help="Skip provider prompt and choose a model for this provider.")
@click.pass_context
def models_select(ctx, provider):
    """Interactively choose provider and model for routing.llm.tasks.chat."""
    config = ctx.obj["khub"].config
    provider_name = str(provider or "").strip().lower()
    if not provider_name:
        providers = chat_provider_options(config)
        if not providers:
            raise click.ClickException("no LLM-capable providers are available")
        _print_provider_options(providers)
        selected = _select_number(click.prompt("Provider #"), max_value=len(providers))
        provider_name = str(providers[selected - 1]["name"])
    else:
        provider_name = _normalize_provider_name(provider_name)

    models = chat_model_options(config, provider_name)
    if models:
        _print_model_options(provider_name, models)
        selected_model = _select_number(click.prompt("Model #"), max_value=len(models))
        model_id = models[selected_model - 1]
    else:
        model_id = str(click.prompt("Model id")).strip()
    payload = set_chat_model_config(config, f"{provider_name}/{model_id}")
    _print_provider_options([_provider_payload(config, provider_name)])
    console.print(f"[green]routing.llm.tasks.chat[/green] -> {payload['provider']}/{payload['model']}")


@models_group.command("login")
@click.argument("provider")
@click.option("--env", "env_name", required=True, help="Environment variable containing the API key.")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def models_login(ctx, provider, env_name, as_json):
    """Store an API-key environment variable reference for a provider."""
    provider_name = _normalize_provider_name(provider)
    env_name = str(env_name or "").strip()
    if not ENV_NAME_RE.match(env_name):
        raise click.BadParameter("--env must be a valid environment variable name")

    config = ctx.obj["khub"].config
    config.set_nested("providers", provider_name, "api_key_env", env_name)
    _save(config)
    status = "set" if os.environ.get(env_name) else "missing"
    payload = {
        "schema": MODELS_RESULT_SCHEMA,
        "status": "ok",
        "provider": provider_name,
        "apiKeyEnv": env_name,
        "apiKeyStatus": status,
        "storedSecret": False,
    }
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[green]{provider_name}[/green] api_key_env={env_name} ({status})")
