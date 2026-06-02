"""User-facing auth setup surface for khub assistant chat."""

from __future__ import annotations

import os
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from knowledge_hub.interfaces.cli.commands.models_cmd import (
    DEFAULT_API_KEY_ENVS,
    ENV_NAME_RE,
    _chat_route,
    _default_llm_model,
    _list_providers,
    _normalize_provider_name,
    _provider_payload,
    _save,
)

console = Console()

AUTH_STATUS_SCHEMA = "knowledge-hub.auth.status.v1"
AUTH_LOGIN_SCHEMA = "knowledge-hub.auth.login.result.v1"


def _codex_oauth_payload(config: Any | None = None) -> dict[str, Any]:
    try:
        from knowledge_hub.providers.codex_provider import codex_provider_status

        delegated = codex_provider_status(config or {}, task_type="chat")
    except Exception as error:
        delegated = {
            "provider": "codex",
            "authType": "codex_cli_delegated_oauth",
            "directTokenReuse": False,
            "secretRead": False,
            "secretPathRead": False,
            "available": False,
            "transport": "",
            "command": "",
            "reason": "readiness_error",
            "summary": f"{type(error).__name__}: {error}",
            "timeoutSeconds": 0,
        }
    return {
        "name": "codex",
        "displayName": "Codex CLI delegated OAuth",
        "authType": "codex_cli_delegated_oauth",
        "status": "delegated_available" if delegated.get("available") else "delegated_unavailable",
        "connected": False,
        "secretRead": False,
        "secretPathRead": False,
        "directTokenReuse": False,
        "delegatedBackend": delegated,
        "message": "khub does not read or reuse Codex OAuth tokens; it can delegate chat generation to the logged-in codex CLI.",
        "nextStep": "Run `khub models use codex/gpt-5.5`; chat execution is reserved for the later chat tranche.",
    }


def _safe_provider_payload(config: Any, provider: str) -> dict[str, Any]:
    try:
        return _provider_payload(config, provider)
    except Exception:
        return {
            "name": provider,
            "displayName": provider,
            "supportsLLM": False,
            "supportsEmbedding": False,
            "requiresApiKey": False,
            "isLocal": False,
            "defaultLLMModel": "",
            "defaultEmbeddingModel": "",
            "models": [],
            "apiKeyEnv": DEFAULT_API_KEY_ENVS.get(provider, ""),
            "apiKeyEnvSource": "provider_default" if provider in DEFAULT_API_KEY_ENVS else "",
            "apiKeyStatus": "unknown",
            "inlineSecretPresent": False,
        }


def _provider_names(config: Any, *, chat_provider: str = "") -> list[str]:
    names = set(DEFAULT_API_KEY_ENVS)
    try:
        names.update(_list_providers(config))
    except Exception:
        pass
    if chat_provider:
        names.add(chat_provider)
    return sorted(name for name in names if name)


def build_auth_status_payload(
    config: Any,
    *,
    provider_override: str = "",
    model_override: str = "",
) -> dict[str, Any]:
    """Build a redacted auth setup payload without reading raw secrets."""
    route = _chat_route(config)
    provider = str(provider_override or route.get("provider") or "").strip().lower()
    model = str(model_override or route.get("model") or "").strip()
    source = "override" if provider_override or model_override else str(route.get("source") or "")
    if provider and not model:
        model = _default_llm_model(config, provider)
    provider_payload = _safe_provider_payload(config, provider) if provider else {}
    providers = [_safe_provider_payload(config, name) for name in _provider_names(config, chat_provider=provider)]
    return {
        "schema": AUTH_STATUS_SCHEMA,
        "status": "ok",
        "mode": "env_reference_only",
        "codex": _codex_oauth_payload(config),
        "chat": {
            "provider": provider,
            "model": model,
            "source": source,
            "isLocal": bool(provider_payload.get("isLocal", False)),
            "requiresApiKey": bool(provider_payload.get("requiresApiKey", False)),
            "apiKeyEnv": provider_payload.get("apiKeyEnv", ""),
            "apiKeyEnvSource": provider_payload.get("apiKeyEnvSource", ""),
            "apiKeyStatus": provider_payload.get("apiKeyStatus", "missing"),
        },
        "providers": providers,
        "secretPolicy": {
            "rawKeyAccepted": False,
            "storedSecret": False,
            "envReferenceOnly": True,
            "codexTokenRead": False,
        },
        "nextCommands": [
            "khub auth login openai --env OPENAI_API_KEY",
            "khub models use openai/gpt-5.4",
            "khub auth codex",
        ],
    }


def store_api_key_env_reference(config: Any, provider: str, env_name: str) -> dict[str, Any]:
    """Store an API-key environment-variable reference without storing the key."""
    provider_name = _normalize_provider_name(provider)
    env_name = str(env_name or "").strip()
    if not ENV_NAME_RE.match(env_name):
        raise click.BadParameter("env var name must be a valid environment variable name")

    config.set_nested("providers", provider_name, "api_key_env", env_name)
    _save(config)
    status = "set" if os.environ.get(env_name) else "missing"
    return {
        "schema": AUTH_LOGIN_SCHEMA,
        "status": "ok",
        "provider": provider_name,
        "apiKeyEnv": env_name,
        "apiKeyStatus": status,
        "storedSecret": False,
        "rawKeyAccepted": False,
        "nextCommands": [
            f"khub models use {provider_name}/MODEL",
            "khub models status",
        ],
    }


def _print_auth_payload(payload: dict[str, Any]) -> None:
    chat = payload.get("chat") if isinstance(payload.get("chat"), dict) else {}
    codex = payload.get("codex") if isinstance(payload.get("codex"), dict) else {}
    codex_table = Table.grid(padding=(0, 2))
    codex_table.add_column(style="cyan", no_wrap=True)
    codex_table.add_column()
    codex_table.add_row("authType", str(codex.get("authType") or "-"))
    codex_table.add_row("status", str(codex.get("status") or "-"))
    codex_table.add_row("connected", str(bool(codex.get("connected", False))))
    codex_table.add_row("directTokenReuse", str(bool(codex.get("directTokenReuse", False))))
    codex_table.add_row("secretRead", str(bool(codex.get("secretRead", False))))
    codex_table.add_row("next", str(codex.get("nextStep") or "-"))
    console.print(Panel(codex_table, title="Codex OAuth", border_style="orange3"))

    chat_table = Table(title="Chat Auth")
    chat_table.add_column("Field", style="cyan", no_wrap=True)
    chat_table.add_column("Value")
    chat_table.add_row("provider", str(chat.get("provider") or "-"))
    chat_table.add_row("model", str(chat.get("model") or "-"))
    chat_table.add_row("source", str(chat.get("source") or "-"))
    chat_table.add_row("apiKeyStatus", str(chat.get("apiKeyStatus") or "-"))
    chat_table.add_row("apiKeyEnv", str(chat.get("apiKeyEnv") or "-"))
    chat_table.add_row("secretPolicy", "env reference only; raw keys are not stored")
    console.print(chat_table)

    providers = payload.get("providers") if isinstance(payload.get("providers"), list) else []
    provider_table = Table(title="Provider Keys")
    provider_table.add_column("Provider", style="cyan", no_wrap=True)
    provider_table.add_column("Local", justify="center")
    provider_table.add_column("Requires key", justify="center")
    provider_table.add_column("Env")
    provider_table.add_column("Status")
    for item in providers:
        provider_table.add_row(
            str(item.get("name") or "-"),
            "Y" if item.get("isLocal") else "-",
            "Y" if item.get("requiresApiKey") else "-",
            str(item.get("apiKeyEnv") or "-"),
            str(item.get("apiKeyStatus") or "-"),
        )
    console.print(provider_table)
    console.print("[dim]next: khub auth login openai --env OPENAI_API_KEY; khub models use openai/gpt-5.4; khub models status[/dim]")


@click.group("auth")
def auth_group():
    """Show assistant auth status and store API-key env references."""


@auth_group.command("status")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def auth_status(ctx, as_json):
    """Show a redacted assistant auth setup page."""
    payload = build_auth_status_payload(ctx.obj["khub"].config)
    if as_json:
        console.print_json(data=payload)
        return
    _print_auth_payload(payload)


@auth_group.command("codex")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
def auth_codex(as_json):
    """Explain Codex OAuth status without reading Codex secrets."""
    ctx = click.get_current_context(silent=True)
    config = ((ctx.obj or {}).get("khub").config if ctx is not None and isinstance(ctx.obj, dict) and (ctx.obj or {}).get("khub") else None)
    payload = {
        "schema": AUTH_STATUS_SCHEMA,
        "status": "ok",
        "codex": _codex_oauth_payload(config),
    }
    if as_json:
        console.print_json(data=payload)
        return
    _print_auth_payload(
        {
            "schema": AUTH_STATUS_SCHEMA,
            "status": "ok",
            "chat": {},
            "codex": payload["codex"],
            "providers": [],
        }
    )


@auth_group.command("login")
@click.argument("provider")
@click.option("--env", "env_name", required=True, help="Environment variable containing the API key.")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def auth_login(ctx, provider, env_name, as_json):
    """Store an API-key environment variable reference for a provider."""
    config = ctx.obj["khub"].config
    payload = store_api_key_env_reference(config, provider, env_name)
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[green]{payload['provider']}[/green] api_key_env={payload['apiKeyEnv']} ({payload['apiKeyStatus']})")
