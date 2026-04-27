"""Provider setup helpers for local, API, Codex MCP, and custom models."""

from __future__ import annotations

import re
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from knowledge_hub.application.runtime_diagnostics import build_runtime_diagnostics
from knowledge_hub.providers.custom_provider import custom_provider_info

console = Console()
PROVIDER_RESULT_SCHEMA = "knowledge-hub.provider.result.v1"
PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")


def _normalize_provider_name(value: str) -> str:
    name = str(value or "").strip().lower()
    if not PROVIDER_NAME_RE.match(name):
        raise click.BadParameter("provider name must use letters, numbers, dot, underscore, or hyphen")
    return name


def _as_model_list(values: tuple[str, ...]) -> list[str]:
    return [str(item).strip() for item in values if str(item).strip()]


def _save(config) -> None:
    config.save()


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


def _provider_config(config, provider: str) -> dict[str, Any]:
    try:
        return dict(config.get_provider_config(provider) or {})
    except Exception:
        return {}


def _default_model_for_role(config, provider: str, role: str) -> str:
    from knowledge_hub.infrastructure.providers import get_provider_info

    try:
        info = get_provider_info(provider, config=config)
    except TypeError:
        info = get_provider_info(provider)
    if info is None:
        return ""
    if role == "embedding":
        return str(getattr(info, "default_embed_model", "") or "")
    return str(getattr(info, "default_llm_model", "") or "")


def _set_role(config, role: str, provider: str, model: str) -> tuple[str, str]:
    role_key = str(role or "").strip().lower()
    provider = _normalize_provider_name(provider)
    if not model:
        model = _default_model_for_role(config, provider, "embedding" if role_key == "embedding" else "llm")
    if not model and role_key not in {"codex", "codex-mcp"}:
        raise click.BadParameter("model is required when the provider has no default model")

    if role_key in {"answer", "summary", "summarization"}:
        config.set_nested("summarization", "provider", provider)
        config.set_nested("summarization", "model", model)
        return "summarization", model
    if role_key == "translation":
        config.set_nested("translation", "provider", provider)
        config.set_nested("translation", "model", model)
        return "translation", model
    if role_key == "embedding":
        config.set_nested("embedding", "provider", provider)
        config.set_nested("embedding", "model", model)
        return "embedding", model
    if role_key in {"local", "mini", "strong"}:
        config.set_nested("routing", "llm", "tasks", role_key, "provider", provider)
        config.set_nested("routing", "llm", "tasks", role_key, "model", model)
        return f"routing.llm.tasks.{role_key}", model
    raise click.BadParameter("role must be answer, summary, translation, embedding, local, mini, or strong")


def _service_defaults(service_name: str) -> dict[str, Any]:
    if not service_name:
        return {}
    from knowledge_hub.providers.openai_compat import KNOWN_SERVICES

    service = KNOWN_SERVICES.get(str(service_name).strip().lower())
    if service is None:
        raise click.BadParameter(f"unknown service preset: {service_name}")
    return dict(service)


def _provider_payload(config, provider: str) -> dict[str, Any]:
    from knowledge_hub.infrastructure.providers import get_provider_info

    cfg = _provider_config(config, provider)
    try:
        info = get_provider_info(provider, config=config)
    except TypeError:
        info = get_provider_info(provider)
    return {
        "name": provider,
        "displayName": getattr(info, "display_name", provider) if info else provider,
        "adapter": str(cfg.get("adapter") or ("builtin" if info else "")),
        "baseUrl": str(cfg.get("base_url") or ""),
        "apiKeyEnv": str(cfg.get("api_key_env") or ""),
        "supportsLLM": bool(getattr(info, "supports_llm", False)),
        "supportsEmbedding": bool(getattr(info, "supports_embedding", False)),
        "requiresApiKey": bool(getattr(info, "requires_api_key", False)),
        "isLocal": bool(getattr(info, "is_local", False)),
        "defaultLLMModel": str(getattr(info, "default_llm_model", "") or ""),
        "defaultEmbeddingModel": str(getattr(info, "default_embed_model", "") or ""),
        "models": list(getattr(info, "available_models", []) or []),
    }


@click.group("provider")
def provider_group():
    """Provider setup for local, API, Codex MCP, and custom models."""


@provider_group.command("recommend")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
def provider_recommend(as_json):
    """Show recommended provider profiles by role."""
    payload = {
        "schema": PROVIDER_RESULT_SCHEMA,
        "status": "ok",
        "recommendations": [
            {
                "profile": "local",
                "summary": "private/local-first",
                "embedding": "ollama/nomic-embed-text or pplx-st",
                "answer": "ollama/qwen3:14b",
            },
            {
                "profile": "balanced",
                "summary": "local embeddings, API generation",
                "embedding": "ollama/nomic-embed-text",
                "answer": "openai/gpt-5-nano",
            },
            {
                "profile": "quality",
                "summary": "API-backed high quality generation and embeddings",
                "embedding": "openai/text-embedding-3-large",
                "answer": "openai/gpt-5.4",
            },
            {
                "profile": "codex-mcp",
                "summary": "use Codex MCP as answer backend",
                "answer": "codex_mcp via khub ask --answer-route codex --allow-external",
            },
        ],
    }
    if as_json:
        console.print_json(data=payload)
        return
    table = Table(title="Provider Recommendations")
    table.add_column("Profile", style="cyan")
    table.add_column("Use")
    table.add_column("Embedding")
    table.add_column("Answer")
    for item in payload["recommendations"]:
        table.add_row(item["profile"], item["summary"], item.get("embedding", "-"), item.get("answer", "-"))
    console.print(table)


@provider_group.command("list")
@click.option("--models", is_flag=True, help="show model inventory")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def provider_list(ctx, models, as_json):
    """List built-in and custom providers."""
    from knowledge_hub.infrastructure.providers import list_providers

    config = ctx.obj["khub"].config
    try:
        providers = list_providers(config=config)
    except TypeError:
        providers = list_providers()
    payload = {
        "schema": PROVIDER_RESULT_SCHEMA,
        "status": "ok",
        "providers": [_provider_payload(config, name) for name in sorted(providers)],
    }
    if as_json:
        console.print_json(data=payload)
        return
    table = Table(title="Providers")
    table.add_column("Name", style="cyan")
    table.add_column("LLM", justify="center")
    table.add_column("Embed", justify="center")
    table.add_column("Local", justify="center")
    table.add_column("Default LLM")
    table.add_column("Default Embed")
    for item in payload["providers"]:
        table.add_row(
            item["name"],
            "Y" if item["supportsLLM"] else "-",
            "Y" if item["supportsEmbedding"] else "-",
            "Y" if item["isLocal"] else "-",
            item["defaultLLMModel"] or "-",
            item["defaultEmbeddingModel"] or "-",
        )
        if models and item["models"]:
            table.add_row("", "", "", "", ", ".join(item["models"][:8]), "")
    console.print(table)


@provider_group.command("add")
@click.argument("name")
@click.option("--from-service", default="", help="known OpenAI-compatible service preset, e.g. deepseek or openrouter.")
@click.option("--adapter", default="openai-compatible", show_default=True, help="adapter type. currently: openai-compatible.")
@click.option("--base-url", default="", help="OpenAI-compatible API base URL.")
@click.option("--api-key-env", default="", help="environment variable that contains the API key.")
@click.option("--no-api-key", is_flag=True, default=False, help="mark endpoint as requiring no API key.")
@click.option("--llm-model", multiple=True, help="LLM model id. Can be repeated.")
@click.option("--embedding-model", multiple=True, help="embedding model id. Can be repeated.")
@click.option("--display-name", default="", help="human-readable provider name.")
@click.option("--region", default="", help="optional region label, e.g. cn/us/eu/local.")
@click.option("--local/--external", "is_local", default=None, help="data boundary hint.")
@click.option("--use-for", multiple=True, help="also set role: answer, summary, translation, embedding, local, mini, strong.")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def provider_add(
    ctx,
    name,
    from_service,
    adapter,
    base_url,
    api_key_env,
    no_api_key,
    llm_model,
    embedding_model,
    display_name,
    region,
    is_local,
    use_for,
    as_json,
):
    """Add a named custom OpenAI-compatible provider."""
    provider = _normalize_provider_name(name)
    service = _service_defaults(from_service)
    base_url = str(base_url or service.get("base_url") or "").strip()
    if not base_url:
        raise click.BadParameter("--base-url is required unless --from-service supplies one")
    api_key_env = str(api_key_env or service.get("env_key") or "").strip()
    llm_models = _as_model_list(tuple(llm_model)) or _as_model_list(tuple(service.get("llm_models") or []))
    embed_models = _as_model_list(tuple(embedding_model)) or _as_model_list(tuple(service.get("embed_models") or []))
    if not llm_models and not embed_models:
        raise click.BadParameter("at least one --llm-model or --embedding-model is required")

    config = ctx.obj["khub"].config
    provider_cfg = {
        "adapter": adapter,
        "base_url": base_url,
        "display_name": display_name or provider,
        "api_key_env": api_key_env,
        "no_api_key": bool(no_api_key),
        "supports": {
            "llm": bool(llm_models),
            "embedding": bool(embed_models),
        },
        "models": {
            "llm": llm_models,
            "embedding": embed_models,
        },
        "default_llm_model": llm_models[0] if llm_models else "",
        "default_embed_model": embed_models[0] if embed_models else "",
    }
    if region:
        provider_cfg["region"] = str(region).strip()
    if is_local is not None:
        provider_cfg["is_local"] = bool(is_local)
        provider_cfg["data_boundary"] = "local" if is_local else "external"

    if custom_provider_info(provider, provider_cfg) is None:
        raise click.BadParameter("only openai-compatible custom providers are supported right now")
    for key, value in provider_cfg.items():
        config.set_nested("providers", provider, key, value)

    role_updates = []
    for role in use_for:
        _, model = _set_role(
            config,
            role,
            provider,
            provider_cfg["default_embed_model"] if str(role).strip().lower() == "embedding" else provider_cfg["default_llm_model"],
        )
        role_updates.append({"role": role, "provider": provider, "model": model})

    _save(config)
    payload = {
        "schema": PROVIDER_RESULT_SCHEMA,
        "status": "ok",
        "provider": _provider_payload(config, provider),
        "roleUpdates": role_updates,
        "nextCommands": [
            f"khub provider use answer {provider}/{provider_cfg['default_llm_model']}" if provider_cfg["default_llm_model"] else "",
            f"khub provider use embedding {provider}/{provider_cfg['default_embed_model']}" if provider_cfg["default_embed_model"] else "",
            "khub provider doctor",
        ],
    }
    payload["nextCommands"] = [item for item in payload["nextCommands"] if item]
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[green]provider added:[/green] {provider}")
    console.print(f"[dim]adapter={adapter} base_url={base_url} api_key_env={api_key_env or '-'}[/dim]")
    for item in role_updates:
        console.print(f"[green]role updated:[/green] {item['role']} -> {item['provider']}/{item['model']}")


@provider_group.command("use")
@click.argument("role")
@click.argument("provider_model")
@click.option("--model", default="", help="model id when not using provider/model syntax.")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def provider_use(ctx, role, provider_model, model, as_json):
    """Assign a provider/model to a role."""
    provider, model_id = _split_provider_model(provider_model, model=model)
    config = ctx.obj["khub"].config
    target, model_id = _set_role(config, role, provider, model_id)
    _save(config)
    payload = {
        "schema": PROVIDER_RESULT_SCHEMA,
        "status": "ok",
        "role": role,
        "target": target,
        "provider": provider,
        "model": model_id,
    }
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[green]{target}[/green] -> {provider}/{model_id}")


@provider_group.command("key")
@click.argument("name")
@click.option("--env", "env_name", required=True, help="environment variable containing the API key.")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def provider_key(ctx, name, env_name, as_json):
    """Store an API-key environment variable reference for a provider."""
    provider = _normalize_provider_name(name)
    env_name = str(env_name or "").strip()
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", env_name):
        raise click.BadParameter("--env must be a valid environment variable name")
    config = ctx.obj["khub"].config
    config.set_nested("providers", provider, "api_key_env", env_name)
    _save(config)
    payload = {
        "schema": PROVIDER_RESULT_SCHEMA,
        "status": "ok",
        "provider": provider,
        "apiKeyEnv": env_name,
    }
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[green]{provider}[/green] api_key_env={env_name}")


@provider_group.command("setup")
@click.option(
    "--profile",
    type=click.Choice(["local", "balanced", "quality", "codex-mcp"], case_sensitive=False),
    required=True,
    help="provider profile to apply.",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def provider_setup(ctx, profile, as_json):
    """Apply a recommended provider profile."""
    config = ctx.obj["khub"].config
    profile = str(profile or "").strip().lower()
    updates: list[dict[str, str]] = []

    def set_role(role: str, provider: str, model: str) -> None:
        target, model_id = _set_role(config, role, provider, model)
        updates.append({"role": role, "target": target, "provider": provider, "model": model_id})

    if profile == "local":
        set_role("translation", "ollama", "qwen3:14b")
        set_role("summary", "ollama", "qwen3:14b")
        set_role("embedding", "ollama", "nomic-embed-text")
        set_role("local", "ollama", "qwen3:14b")
    elif profile == "balanced":
        set_role("translation", "openai", "gpt-5-nano")
        set_role("summary", "openai", "gpt-5-nano")
        set_role("embedding", "ollama", "nomic-embed-text")
        set_role("local", "ollama", "qwen3:14b")
        set_role("mini", "openai", "gpt-5-nano")
        set_role("strong", "openai", "gpt-5.4")
    elif profile == "quality":
        set_role("translation", "openai", "gpt-5-nano")
        set_role("summary", "openai", "gpt-5.4")
        set_role("embedding", "openai", "text-embedding-3-large")
        set_role("mini", "openai", "gpt-5-nano")
        set_role("strong", "openai", "gpt-5.4")
    elif profile == "codex-mcp":
        config.set_nested("routing", "llm", "tasks", "rag_answer", "preferred_backend", "codex_mcp")
        config.set_nested("routing", "llm", "tasks", "rag_answer", "codex", "transport", "mcp")
        config.set_nested("routing", "llm", "tasks", "rag_answer", "codex", "command", "codex")
        config.set_nested("routing", "llm", "tasks", "rag_answer", "codex", "args", "mcp-server")
        updates.append({"role": "answer", "target": "routing.llm.tasks.rag_answer", "provider": "codex_mcp", "model": ""})

    _save(config)
    payload = {
        "schema": PROVIDER_RESULT_SCHEMA,
        "status": "ok",
        "profile": profile,
        "updates": updates,
        "nextCommands": ["khub provider doctor", 'khub ask "질문" --allow-external'],
    }
    if profile == "codex-mcp":
        payload["nextCommands"] = ["khub provider doctor", 'khub ask "질문" --answer-route codex --allow-external']
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[green]profile applied:[/green] {profile}")
    for item in updates:
        model_suffix = f"/{item['model']}" if item["model"] else ""
        console.print(f"  - {item['target']} -> {item['provider']}{model_suffix}")


@provider_group.command("doctor")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def provider_doctor(ctx, as_json):
    """Inspect configured provider roles without exposing secrets."""
    config = ctx.obj["khub"].config
    diagnostics = build_runtime_diagnostics(config)
    payload = {
        "schema": PROVIDER_RESULT_SCHEMA,
        "status": "ok" if not diagnostics.get("degraded") else "degraded",
        "providers": diagnostics.get("providers", []),
        "warnings": diagnostics.get("warnings", []),
    }
    if as_json:
        console.print_json(data=payload)
        return
    table = Table(title="Configured Provider Roles")
    table.add_column("Role", style="cyan")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Available")
    table.add_column("Reasons")
    for item in payload["providers"]:
        table.add_row(
            str(item.get("role") or ""),
            str(item.get("provider") or ""),
            str(item.get("model") or ""),
            "Y" if item.get("available") else "-",
            ", ".join(str(reason) for reason in item.get("reasons") or []) or "-",
        )
    console.print(table)
