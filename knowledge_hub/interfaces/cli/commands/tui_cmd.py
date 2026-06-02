"""Interactive shell for the hidden khub chat Interface."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from knowledge_hub.interfaces.cli.commands.assistant_runtime import (
    compact_source_label,
    generate_paper_answer_payload,
    parse_paper_slash,
)
from knowledge_hub.interfaces.cli.commands.auth_cmd import (
    _print_auth_payload as _print_auth_status_payload,
    build_auth_status_payload,
    store_api_key_env_reference,
)
from knowledge_hub.interfaces.cli.commands.chat_cmd import (
    _allow_external_value,
    _build_turn_prompt,
    _generate_chat_answer,
    _resolve_chat_route,
)
from knowledge_hub.interfaces.cli.commands.models_cmd import (
    _provider_payload,
    chat_model_options,
    chat_provider_options,
    set_chat_model_config,
)
from knowledge_hub.interfaces.cli.commands.session_runtime import (
    SESSION_HISTORY_MODE,
    SessionRecorder,
    new_session_id,
)
from knowledge_hub.interfaces.cli.commands.slash_registry import help_rows, parse_slash_input, startup_rows

console = Console()


def _provider_status(config: Any, provider: str) -> dict[str, Any]:
    try:
        return _provider_payload(config, provider)
    except Exception:
        return {
            "name": provider,
            "isLocal": False,
            "requiresApiKey": False,
            "apiKeyEnv": "",
            "apiKeyStatus": "unknown",
        }


def _external_label(provider_status: dict[str, Any], allow_external: bool) -> str:
    if provider_status.get("isLocal"):
        return "local"
    return "allowed" if allow_external else "blocked"


def _startup_panel(
    *,
    provider: str,
    model: str,
    allow_external: bool,
    session_id: str,
    workspace: Path,
    provider_status: dict[str, Any],
    history_mode: str,
) -> Panel:
    status = Table.grid(padding=(0, 2))
    status.add_column(style="bold cyan", no_wrap=True)
    status.add_column()
    status.add_row("Mode", "read-only Interface")
    status.add_row("Route", "plain chat / paper evidence")
    status.add_row("Model", f"{provider}/{model}")
    status.add_row("External", _external_label(provider_status, allow_external))
    status.add_row("API key", f"{provider_status.get('apiKeyStatus', 'unknown')} env={provider_status.get('apiKeyEnv') or '-'}")
    status.add_row("History", history_mode)
    status.add_row("Session", session_id)
    status.add_row("Workspace", str(workspace))

    commands = Table.grid(padding=(0, 2))
    commands.add_column(style="bold cyan", no_wrap=True)
    commands.add_column()
    commands.add_row("Commands", "")
    for label, summary in startup_rows():
        commands.add_row(label, summary)

    body = Table.grid(expand=True)
    body.add_column(ratio=5)
    body.add_column(ratio=5)
    body.add_row(status, commands)
    return Panel(body, title="Knowledge Hub TUI", subtitle="plain chat first; /paper for evidence", border_style="cyan")


def _print_startup(
    *,
    provider: str,
    model: str,
    allow_external: bool,
    session_id: str,
    workspace: Path,
    provider_status: dict[str, Any],
    history_mode: str,
) -> None:
    console.print(
        _startup_panel(
            provider=provider,
            model=model,
            allow_external=allow_external,
            session_id=session_id,
            workspace=workspace,
            provider_status=provider_status,
            history_mode=history_mode,
        )
    )
    console.print("[dim]Type a message, /help, /paper QUESTION, /models, /auth, /clear, or /exit.[/dim]")


def _print_help() -> None:
    table = Table(title="TUI Commands")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Use")
    for label, summary in help_rows():
        table.add_row(label, summary)
    console.print(table)


def _print_models_status(*, provider: str, model: str, allow_external: bool, provider_status: dict[str, Any]) -> None:
    table = Table(title="Chat Model")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("provider", provider)
    table.add_row("model", model)
    table.add_row("local", str(bool(provider_status.get("isLocal", False))))
    table.add_row("external", _external_label(provider_status, allow_external))
    table.add_row("apiKeyStatus", str(provider_status.get("apiKeyStatus", "unknown")))
    table.add_row("apiKeyEnv", str(provider_status.get("apiKeyEnv") or "-"))
    console.print(table)


def _print_model_use(payload: dict[str, Any]) -> None:
    console.print(f"[green]routing.llm.tasks.chat[/green] -> {payload['provider']}/{payload['model']}")


def _prompt_model_selection(config: Any) -> tuple[str, str] | None:
    providers = chat_provider_options(config)
    if not providers:
        console.print("[yellow]no LLM-capable providers are available[/yellow]")
        return None

    provider_table = Table(title="Choose Provider")
    provider_table.add_column("#", justify="right")
    provider_table.add_column("Provider", style="cyan", no_wrap=True)
    provider_table.add_column("Type")
    provider_table.add_column("Default")
    provider_table.add_column("Local", justify="center")
    provider_table.add_column("API key")
    for index, item in enumerate(providers, start=1):
        provider_table.add_row(
            str(index),
            str(item.get("name") or "-"),
            str(item.get("catalog") or item.get("adapter") or "native"),
            str(item.get("defaultLLMModel") or "-"),
            "Y" if item.get("isLocal") else "-",
            str(item.get("apiKeyStatus") or "-"),
        )
    console.print(provider_table)

    console.print("Provider # (or q to cancel): ", end="")
    provider_raw = sys.stdin.readline()
    if provider_raw == "":
        return None
    provider_choice = provider_raw.strip().lower()
    if provider_choice in {"q", "quit", "cancel"}:
        console.print("[dim]model selection cancelled.[/dim]")
        return None
    try:
        provider_index = int(provider_choice)
    except ValueError:
        console.print("[yellow]selection must be a number[/yellow]")
        return None
    if provider_index < 1 or provider_index > len(providers):
        console.print(f"[yellow]selection must be between 1 and {len(providers)}[/yellow]")
        return None
    provider_name = str(providers[provider_index - 1]["name"])

    models = chat_model_options(config, provider_name)
    if not models:
        console.print(f"Model id for {provider_name}: ", end="")
        model_raw = sys.stdin.readline()
        if model_raw == "":
            return None
        model_id = model_raw.strip()
        return (provider_name, model_id) if model_id else None

    model_table = Table(title=f"Choose Model ({provider_name})")
    model_table.add_column("#", justify="right")
    model_table.add_column("Model", style="cyan")
    for index, model_id in enumerate(models, start=1):
        model_table.add_row(str(index), model_id)
    console.print(model_table)

    console.print("Model # (or q to cancel): ", end="")
    model_raw = sys.stdin.readline()
    if model_raw == "":
        return None
    model_choice = model_raw.strip().lower()
    if model_choice in {"q", "quit", "cancel"}:
        console.print("[dim]model selection cancelled.[/dim]")
        return None
    try:
        model_index = int(model_choice)
    except ValueError:
        console.print("[yellow]selection must be a number[/yellow]")
        return None
    if model_index < 1 or model_index > len(models):
        console.print(f"[yellow]selection must be between 1 and {len(models)}[/yellow]")
        return None
    return provider_name, models[model_index - 1]


def _print_auth_login(payload: dict[str, Any]) -> None:
    console.print(f"[green]{payload['provider']}[/green] api_key_env={payload['apiKeyEnv']} ({payload['apiKeyStatus']})")


def _print_paper_payload(payload: dict[str, Any]) -> None:
    status = str(payload.get("status") or "unknown")
    if status != "ok":
        warnings = list(payload.get("warnings") or [])
        console.print(f"[yellow]paper route {status}:[/yellow] {warnings[0] if warnings else 'no answer'}")
        return

    answer = str(payload.get("answer") or "").strip()
    console.print(answer or "[yellow]No paper answer was generated.[/yellow]")

    sources = list(payload.get("sources") or [])
    warnings = list(payload.get("warnings") or [])
    route = str(payload.get("answerRouteApplied") or "unknown")
    provider = str(payload.get("answerProviderApplied") or "-")
    model = str(payload.get("answerModelApplied") or "-")

    details = Table(title="Paper Evidence", show_header=False)
    details.add_column("Field", style="cyan", no_wrap=True)
    details.add_column("Value")
    details.add_row("route", f"{route} / {provider}/{model}")
    details.add_row("allowExternal", str(bool(payload.get("allowExternal", False))))
    if sources:
        details.add_row("sources", "\n".join(f"- {compact_source_label(source)}" for source in sources[:5]))
    if warnings:
        details.add_row("warnings", "\n".join(f"- {warning}" for warning in warnings[:5]))
    console.print(details)


def _paper_turn(khub: Any, *, user_prompt: str, allow_external: bool | None) -> dict[str, Any]:
    command = parse_paper_slash(user_prompt, allow_ask_alias=True)
    if command is None:
        console.print("[yellow]paper route requires /paper QUESTION[/yellow]")
        return {"status": "blocked", "route": "paper", "answer": "", "warnings": ["paper route requires /paper QUESTION"]}
    with console.status("paper evidence answer..."):
        payload = generate_paper_answer_payload(
            khub,
            original_prompt=user_prompt,
            question=command.question,
            allow_external_override=allow_external,
        )
    _print_paper_payload(payload)
    return payload


def _chat_turn(
    khub: Any,
    *,
    history: list[tuple[str, str]],
    user_prompt: str,
    provider: str,
    model: str,
    allow_external: bool,
) -> dict[str, Any]:
    turn_prompt = _build_turn_prompt(history, user_prompt)
    answer, blocked_reason = _generate_chat_answer(
        khub,
        prompt=turn_prompt,
        provider=provider,
        model=model,
        allow_external=allow_external,
    )
    if blocked_reason:
        console.print(f"[red]blocked:[/red] {blocked_reason}")
        return {
            "status": "blocked",
            "route": "plain",
            "answer": "",
            "warnings": [blocked_reason],
            "provider": provider,
            "model": model,
            "allowExternal": bool(allow_external),
        }
    console.print(answer)
    history.append(("User", user_prompt))
    history.append(("Assistant", answer))
    return {
        "status": "ok",
        "route": "plain",
        "answer": answer,
        "warnings": [],
        "provider": provider,
        "model": model,
        "allowExternal": bool(allow_external),
    }


def _record_tui_turn(recorder: SessionRecorder | None, *, route: str, text: str, payload: dict[str, Any]) -> None:
    if recorder is None:
        return
    recorder.user_message(route=route, text=text)
    recorder.assistant_message(route=route, answer=str(payload.get("answer") or ""))
    metadata: dict[str, Any] = {
        "status": str(payload.get("status") or "unknown"),
        "warningCount": len(list(payload.get("warnings") or [])),
    }
    if "assistUsage" in payload:
        metadata["assistUsage"] = payload.get("assistUsage")
    for key in ("provider", "model", "allowExternal", "answerRouteApplied", "answerProviderApplied", "answerModelApplied"):
        if key in payload:
            metadata[key] = payload.get(key)
    recorder.route_metadata(route=route, metadata=metadata)


def _run_loop(
    khub: Any,
    *,
    provider: str,
    model: str,
    allow_external: bool,
    paper_allow_external: bool | None,
    provider_status: dict[str, Any],
    recorder: SessionRecorder | None = None,
) -> None:
    history: list[tuple[str, str]] = []
    current_provider = provider
    current_model = model
    current_allow_external = allow_external
    current_provider_status = dict(provider_status)
    try:
        while True:
            console.print("> ", end="")
            line = sys.stdin.readline()
            if line == "":
                console.print("")
                break
            text = line.strip()
            if not text:
                continue
            lower = text.lower()
            if lower in {"exit", "quit"}:
                break
            parsed_slash = parse_slash_input(text)
            if parsed_slash is None and text.startswith("/"):
                console.print(f"[yellow]unknown command:[/yellow] {text}. Type /help.")
                continue
            if parsed_slash is None:
                payload = _chat_turn(
                    khub,
                    history=history,
                    user_prompt=text,
                    provider=current_provider,
                    model=current_model,
                    allow_external=current_allow_external,
                )
                _record_tui_turn(recorder, route="plain", text=text, payload=payload)
                continue

            command, args = parsed_slash
            if command.name == "exit":
                break
            if command.name == "help":
                _print_help()
                continue
            if command.name == "auth":
                if recorder is not None:
                    recorder.user_message(route="auth", text=text)
                args_lower = args.lower()
                if args_lower.startswith("login "):
                    parts = args.split()
                    if len(parts) == 3:
                        login_provider, env_name = parts[1], parts[2]
                    elif len(parts) == 4 and parts[2] == "--env":
                        login_provider, env_name = parts[1], parts[3]
                    else:
                        console.print("[yellow]usage:[/yellow] /auth login PROVIDER ENV_VAR")
                        console.print("[dim]also accepted: /auth login PROVIDER --env ENV_VAR[/dim]")
                        continue
                    try:
                        payload = store_api_key_env_reference(khub.config, login_provider, env_name)
                    except click.BadParameter as exc:
                        console.print(f"[yellow]auth login blocked:[/yellow] {exc.message}")
                        continue
                    _print_auth_login(payload)
                    current_provider_status = _provider_status(khub.config, current_provider)
                else:
                    payload = build_auth_status_payload(khub.config, provider_override=current_provider, model_override=current_model)
                    _print_auth_status_payload(payload)
                if recorder is not None:
                    recorder.route_metadata(
                        route="auth",
                        metadata={
                            "status": str(payload.get("status") or "unknown"),
                            "schema": payload.get("schema"),
                            "codexStatus": ((payload.get("codex") or {}).get("status") if isinstance(payload.get("codex"), dict) else ""),
                        },
                    )
                continue
            if command.name == "models":
                args_lower = args.lower()
                if args_lower in {"select", "choose"}:
                    selected = _prompt_model_selection(khub.config)
                    if selected is None:
                        continue
                    selected_provider, selected_model = selected
                    provider_model = f"{selected_provider}/{selected_model}"
                elif args_lower.startswith("use "):
                    provider_model = args.split(maxsplit=1)[1].strip()
                else:
                    provider_model = ""
                if provider_model:
                    try:
                        payload = set_chat_model_config(khub.config, provider_model)
                    except click.BadParameter as exc:
                        console.print(f"[yellow]model use blocked:[/yellow] {exc.message}")
                        continue
                    current_provider = str(payload["provider"])
                    current_model = str(payload["model"])
                    current_allow_external = _allow_external_value(khub.config, current_provider, paper_allow_external)
                    current_provider_status = _provider_status(khub.config, current_provider)
                    _print_model_use(payload)
                    if recorder is not None:
                        recorder.route_metadata(
                            route="models",
                            metadata={
                                "status": str(payload.get("status") or "unknown"),
                                "schema": payload.get("schema"),
                                "provider": current_provider,
                                "model": current_model,
                            },
                        )
                    continue
                _print_models_status(
                    provider=current_provider,
                    model=current_model,
                    allow_external=current_allow_external,
                    provider_status=current_provider_status,
                )
                continue
            if command.name == "clear":
                history.clear()
                console.print("[dim]chat history cleared for this process.[/dim]")
                continue
            if command.name == "chat":
                chat_text = args
                if not chat_text:
                    console.print(f"[cyan]chat[/cyan] -> {current_provider}/{current_model}")
                    continue
                payload = _chat_turn(
                    khub,
                    history=history,
                    user_prompt=chat_text,
                    provider=current_provider,
                    model=current_model,
                    allow_external=current_allow_external,
                )
                _record_tui_turn(recorder, route="plain", text=chat_text, payload=payload)
                continue
            if command.name in {"paper", "ask"}:
                user_prompt = f"/{command.name} {args}".strip()
                payload = _paper_turn(khub, user_prompt=user_prompt, allow_external=paper_allow_external)
                _record_tui_turn(recorder, route="paper", text=text, payload=payload)
                continue
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
    finally:
        if recorder is not None:
            recorder.end()


def run_tui(
    khub: Any,
    *,
    provider: str = "",
    model: str = "",
    allow_external: bool | None = None,
    save_session: bool = False,
) -> None:
    """Open the interactive assistant shell."""
    route_provider, route_model = _resolve_chat_route(khub.config, provider_override=provider, model_override=model)
    effective_allow_external = _allow_external_value(khub.config, route_provider, allow_external)
    provider_status = _provider_status(khub.config, route_provider)
    session_id = new_session_id("tui")
    history_mode = SESSION_HISTORY_MODE if save_session else "memory-only process"
    recorder = None
    if save_session:
        recorder = SessionRecorder(session_id=session_id, surface="tui")
        recorder.start(
            provider=route_provider,
            model=route_model,
            allow_external=effective_allow_external,
            history_mode=SESSION_HISTORY_MODE,
        )
    _print_startup(
        provider=route_provider,
        model=route_model,
        allow_external=effective_allow_external,
        session_id=session_id,
        workspace=Path.cwd(),
        provider_status=provider_status,
        history_mode=history_mode,
    )
    _run_loop(
        khub,
        provider=route_provider,
        model=route_model,
        allow_external=effective_allow_external,
        paper_allow_external=allow_external,
        provider_status=provider_status,
        recorder=recorder,
    )


@click.command("tui")
@click.option("--provider", default="", help="Override the chat provider.")
@click.option("--model", default="", help="Override the chat model.")
@click.option(
    "--allow-external/--no-allow-external",
    default=None,
    help="Allow or block non-local provider calls for this TUI session.",
)
@click.option("--save-session/--no-save-session", default=False, show_default=True, help="Persist metadata-only session events.")
@click.pass_context
def tui_cmd(ctx, provider, model, allow_external, save_session):
    """Open the hidden khub chat Interface TUI."""
    run_tui(
        ctx.obj["khub"],
        provider=provider,
        model=model,
        allow_external=allow_external,
        save_session=save_session,
    )
