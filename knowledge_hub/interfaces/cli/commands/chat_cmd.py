"""Explicit LLM chat command for the khub assistant Interface."""

from __future__ import annotations

import sys
from typing import Any

import click
from rich.console import Console

from knowledge_hub.interfaces.cli.commands.assistant_runtime import (
    generate_paper_answer_payload,
    paper_payload_text_lines,
    parse_paper_slash,
)
from knowledge_hub.interfaces.cli.commands.session_runtime import SESSION_HISTORY_MODE, SessionRecorder, new_session_id

console = Console()

CHAT_RESULT_SCHEMA = "knowledge-hub.chat.result.v1"
LOCAL_PROVIDER_NAMES = {"ollama", "pplx-local", "pplx-st", "pplx_st"}


def _config_get(config: Any, *keys: str, default: Any = "") -> Any:
    getter = getattr(config, "get_nested", None)
    if callable(getter):
        return getter(*keys, default=default)
    return default


def _provider_info(config: Any, provider: str):
    from knowledge_hub.infrastructure.providers import get_provider_info

    try:
        return get_provider_info(provider, config=config)
    except TypeError:
        return get_provider_info(provider)


def _default_llm_model(config: Any, provider: str) -> str:
    info = _provider_info(config, provider)
    return str(getattr(info, "default_llm_model", "") or "") if info else ""


def _is_local_provider(config: Any, provider: str) -> bool:
    provider_name = str(provider or "").strip().lower()
    if provider_name in LOCAL_PROVIDER_NAMES:
        return True
    info = _provider_info(config, provider_name)
    return bool(getattr(info, "is_local", False)) if info else False


def _summarization_provider(config: Any) -> str:
    return str(getattr(config, "summarization_provider", "") or _config_get(config, "summarization", "provider", default="ollama") or "ollama")


def _summarization_model(config: Any) -> str:
    return str(getattr(config, "summarization_model", "") or _config_get(config, "summarization", "model", default="qwen3:14b") or "qwen3:14b")


def _resolve_chat_route(config: Any, provider_override: str = "", model_override: str = "") -> tuple[str, str]:
    provider_override = str(provider_override or "").strip().lower()
    model_override = str(model_override or "").strip()
    chat_provider = str(_config_get(config, "routing", "llm", "tasks", "chat", "provider", default="") or "").strip().lower()
    chat_model = str(_config_get(config, "routing", "llm", "tasks", "chat", "model", default="") or "").strip()
    summary_provider = _summarization_provider(config).strip().lower()
    summary_model = _summarization_model(config).strip()

    provider = provider_override or chat_provider or summary_provider
    if not provider:
        raise click.ClickException("chat provider is not configured")

    if model_override:
        model = model_override
    elif provider_override:
        model = chat_model if provider_override == chat_provider else ""
        model = model or _default_llm_model(config, provider) or summary_model
    elif chat_provider:
        model = chat_model or _default_llm_model(config, provider) or summary_model
    else:
        model = summary_model or _default_llm_model(config, provider)

    if not model:
        raise click.ClickException("chat model is not configured")
    return provider, model


def _allow_external_value(config: Any, provider: str, value: bool | None) -> bool:
    if value is not None:
        return bool(value)
    return not _is_local_provider(config, provider)


def _blocked_payload(
    *,
    prompt: str,
    provider: str,
    model: str,
    allow_external: bool,
    mode: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema": CHAT_RESULT_SCHEMA,
        "status": "blocked",
        "mode": mode,
        "provider": provider,
        "model": model,
        "allowExternal": bool(allow_external),
        "historyPersisted": False,
        "promptChars": len(prompt),
        "answer": "",
        "warnings": [reason],
    }


def _ok_payload(
    *,
    prompt: str,
    answer: str,
    provider: str,
    model: str,
    allow_external: bool,
    mode: str,
) -> dict[str, Any]:
    return {
        "schema": CHAT_RESULT_SCHEMA,
        "status": "ok",
        "mode": mode,
        "provider": provider,
        "model": model,
        "allowExternal": bool(allow_external),
        "historyPersisted": False,
        "promptChars": len(prompt),
        "answer": answer,
        "warnings": [],
    }


def _new_recorder(
    *, save_session: bool, surface: str, session_id: str, provider: str, model: str, allow_external: bool
) -> SessionRecorder | None:
    if not save_session:
        return None
    recorder = SessionRecorder(session_id=session_id, surface=surface)
    recorder.start(provider=provider, model=model, allow_external=allow_external, history_mode=SESSION_HISTORY_MODE)
    return recorder


def _diagnostics(
    *,
    layer_used: str,
    route: str,
    session_id: str,
    turn_id: str,
    provider_applied: str = "",
    model_applied: str = "",
    external_call_allowed: bool,
    policy_blocked: bool,
) -> dict[str, Any]:
    # Mapping: plain chat is an Interface/provider turn; /paper delegates to the Core ask/paper runtime.
    return {
        "layerUsed": layer_used,
        "route": route,
        "providerApplied": str(provider_applied or ""),
        "modelApplied": str(model_applied or ""),
        "sessionId": session_id,
        "turnId": turn_id,
        "externalCallAllowed": bool(external_call_allowed),
        "policyBlocked": bool(policy_blocked),
    }


def _attach_diagnostics(payload: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    payload.update(diagnostics)
    return payload


def _footer(payload: dict[str, Any]) -> str:
    provider = str(payload.get("providerApplied") or payload.get("provider") or "-")
    model = str(payload.get("modelApplied") or payload.get("model") or "-")
    return (
        f"diagnostics: layer={payload.get('layerUsed') or '-'} route={payload.get('route') or '-'} "
        f"provider={provider} model={model} external={bool(payload.get('externalCallAllowed'))} "
        f"policyBlocked={bool(payload.get('policyBlocked'))} session={payload.get('sessionId') or '-'} "
        f"turn={payload.get('turnId') or '-'}"
    )


def _attach_session(payload: dict[str, Any], recorder: SessionRecorder | None) -> dict[str, Any]:
    if recorder is None:
        return payload
    payload["historyPersisted"] = True
    payload["session"] = recorder.session_payload()
    return payload


def _record_route_metadata(recorder: SessionRecorder | None, *, route: str, payload: dict[str, Any]) -> None:
    if recorder is None:
        return
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


def _build_turn_prompt(history: list[tuple[str, str]], user_prompt: str) -> str:
    if not history:
        return user_prompt
    transcript = "\n".join(f"{role}: {content}" for role, content in history)
    return f"Conversation so far:\n{transcript}\n\nUser: {user_prompt}\nAssistant:"


def _generate_chat_answer(
    khub: Any,
    *,
    prompt: str,
    provider: str,
    model: str,
    allow_external: bool,
) -> tuple[str, str | None]:
    if not allow_external and not _is_local_provider(khub.config, provider):
        return "", f"external provider blocked: {provider}/{model} requires --allow-external"
    llm = khub.build_llm(provider, model)
    try:
        return str(llm.generate(prompt) or ""), None
    except Exception as error:
        if getattr(error, "code", "") == "POLICY_BLOCKED_OUTBOUND":
            decision = getattr(error, "decision", None)
            classification = str(getattr(decision, "classification", "") or "unknown")
            trace_id = str(getattr(decision, "trace_id", "") or "-")
            return "", f"outbound policy blocked: classification={classification} trace_id={trace_id}"
        raise


def _single_turn(
    ctx: click.Context,
    prompt: str,
    provider: str,
    model: str,
    allow_external: bool,
    as_json: bool,
    session_id: str,
    turn_id: str,
    recorder: SessionRecorder | None = None,
) -> None:
    khub = ctx.obj["khub"]
    if recorder is not None:
        recorder.user_message(route="plain", text=prompt)
    answer, blocked_reason = _generate_chat_answer(
        khub,
        prompt=prompt,
        provider=provider,
        model=model,
        allow_external=allow_external,
    )
    if blocked_reason:
        payload = _blocked_payload(
            prompt=prompt,
            provider=provider,
            model=model,
            allow_external=allow_external,
            mode="single",
            reason=blocked_reason,
        )
        _attach_diagnostics(
            payload,
            _diagnostics(
                layer_used="interface",
                route="plain_llm",
                session_id=session_id,
                turn_id=turn_id,
                provider_applied=provider,
                model_applied=model,
                external_call_allowed=allow_external,
                policy_blocked=True,
            ),
        )
        _attach_session(payload, recorder)
        _record_route_metadata(recorder, route="plain", payload=payload)
        if as_json:
            console.print_json(data=payload)
            return
        raise click.ClickException(blocked_reason)

    payload = _ok_payload(
        prompt=prompt,
        answer=answer,
        provider=provider,
        model=model,
        allow_external=allow_external,
        mode="single",
    )
    _attach_diagnostics(
        payload,
        _diagnostics(
            layer_used="interface",
            route="plain_llm",
            session_id=session_id,
            turn_id=turn_id,
            provider_applied=provider,
            model_applied=model,
            external_call_allowed=allow_external,
            policy_blocked=False,
        ),
    )
    if recorder is not None:
        recorder.assistant_message(route="plain", answer=answer)
    _attach_session(payload, recorder)
    _record_route_metadata(recorder, route="plain", payload=payload)
    if as_json:
        console.print_json(data=payload)
        return
    click.echo(answer)
    click.echo(_footer(payload))


def _single_turn_paper(
    ctx: click.Context,
    prompt: str,
    allow_external: bool | None,
    as_json: bool,
    session_id: str,
    turn_id: str,
    recorder: SessionRecorder | None = None,
) -> None:
    khub = ctx.obj["khub"]
    command = parse_paper_slash(prompt)
    if command is None:
        raise click.ClickException("paper route requires /paper QUESTION")
    if recorder is not None:
        recorder.user_message(route="paper", text=prompt)
    payload = generate_paper_answer_payload(
        khub,
        original_prompt=prompt,
        question=command.question,
        allow_external_override=allow_external,
    )
    _attach_diagnostics(
        payload,
        _diagnostics(
            layer_used="core",
            route=str(payload.get("route") or "paper"),
            session_id=session_id,
            turn_id=turn_id,
            provider_applied=str(payload.get("answerProviderApplied") or ""),
            model_applied=str(payload.get("answerModelApplied") or ""),
            external_call_allowed=bool(payload.get("allowExternal")),
            policy_blocked=payload.get("status") == "blocked",
        ),
    )
    if recorder is not None:
        recorder.assistant_message(route="paper", answer=str(payload.get("answer") or ""))
    _attach_session(payload, recorder)
    _record_route_metadata(recorder, route="paper", payload=payload)
    if as_json:
        console.print_json(data=payload)
        return
    if payload.get("status") not in {"ok"} and not payload.get("answer"):
        warnings = list(payload.get("warnings") or [])
        raise click.ClickException(str(warnings[0] if warnings else payload.get("status") or "paper route failed"))
    click.echo("\n".join(paper_payload_text_lines(payload)))
    click.echo(_footer(payload))


def _repl(
    ctx: click.Context,
    provider: str,
    model: str,
    allow_external: bool,
    paper_allow_external: bool | None,
    session_id: str,
    recorder: SessionRecorder | None = None,
) -> None:
    khub = ctx.obj["khub"]
    try:
        if not allow_external and not _is_local_provider(khub.config, provider):
            raise click.ClickException(f"external provider blocked: {provider}/{model} requires --allow-external")

        history: list[tuple[str, str]] = []
        turn_index = 0
        click.echo(f"khub chat ({provider}/{model}); type /exit to quit")
        while True:
            click.echo("you> ", nl=False)
            line = sys.stdin.readline()
            if line == "":
                click.echo("")
                break
            user_prompt = line.strip()
            if not user_prompt:
                continue
            if user_prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
                break
            paper_command = parse_paper_slash(user_prompt)
            if paper_command is not None:
                turn_index += 1
                turn_id = f"turn_{turn_index:04d}"
                if recorder is not None:
                    recorder.user_message(route="paper", text=user_prompt)
                payload = generate_paper_answer_payload(
                    khub,
                    original_prompt=user_prompt,
                    question=paper_command.question,
                    allow_external_override=paper_allow_external,
                )
                _attach_diagnostics(
                    payload,
                    _diagnostics(
                        layer_used="core",
                        route=str(payload.get("route") or "paper"),
                        session_id=session_id,
                        turn_id=turn_id,
                        provider_applied=str(payload.get("answerProviderApplied") or ""),
                        model_applied=str(payload.get("answerModelApplied") or ""),
                        external_call_allowed=bool(payload.get("allowExternal")),
                        policy_blocked=payload.get("status") == "blocked",
                    ),
                )
                if recorder is not None:
                    recorder.assistant_message(route="paper", answer=str(payload.get("answer") or ""))
                _record_route_metadata(recorder, route="paper", payload=payload)
                click.echo("\n".join(paper_payload_text_lines(payload)))
                click.echo(_footer(payload))
                continue
            turn_index += 1
            turn_id = f"turn_{turn_index:04d}"
            if recorder is not None:
                recorder.user_message(route="plain", text=user_prompt)
            turn_prompt = _build_turn_prompt(history, user_prompt)
            answer, blocked_reason = _generate_chat_answer(
                khub,
                prompt=turn_prompt,
                provider=provider,
                model=model,
                allow_external=allow_external,
            )
            if blocked_reason:
                payload = _blocked_payload(
                    prompt=user_prompt,
                    provider=provider,
                    model=model,
                    allow_external=allow_external,
                    mode="repl",
                    reason=blocked_reason,
                )
                _attach_diagnostics(
                    payload,
                    _diagnostics(
                        layer_used="interface",
                        route="plain_llm",
                        session_id=session_id,
                        turn_id=turn_id,
                        provider_applied=provider,
                        model_applied=model,
                        external_call_allowed=allow_external,
                        policy_blocked=True,
                    ),
                )
                _record_route_metadata(recorder, route="plain", payload=payload)
                click.echo(f"blocked: {blocked_reason}")
                click.echo(_footer(payload))
                continue
            if recorder is not None:
                recorder.assistant_message(route="plain", answer=answer)
            payload = _ok_payload(
                prompt=user_prompt,
                answer=answer,
                provider=provider,
                model=model,
                allow_external=allow_external,
                mode="repl",
            )
            _attach_diagnostics(
                payload,
                _diagnostics(
                    layer_used="interface",
                    route="plain_llm",
                    session_id=session_id,
                    turn_id=turn_id,
                    provider_applied=provider,
                    model_applied=model,
                    external_call_allowed=allow_external,
                    policy_blocked=False,
                ),
            )
            _record_route_metadata(recorder, route="plain", payload=payload)
            click.echo(answer)
            click.echo(_footer(payload))
            history.append(("User", user_prompt))
            history.append(("Assistant", answer))
    finally:
        if recorder is not None:
            recorder.end()


@click.command("chat")
@click.argument("prompt", required=False)
@click.option("--provider", default="", help="Override the chat provider.")
@click.option("--model", default="", help="Override the chat model.")
@click.option(
    "--allow-external/--no-allow-external",
    default=None,
    help="Allow or block non-local provider calls for this chat request.",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True, help="Emit a JSON result for single-turn chat.")
@click.option("--save-session/--no-save-session", default=False, show_default=True, help="Persist metadata-only SQLite session events.")
@click.pass_context
def chat_cmd(ctx, prompt, provider, model, allow_external, as_json, save_session):
    """Plain LLM chat. Use /paper QUESTION for paper evidence answers."""
    khub = ctx.obj["khub"]
    route_provider, route_model = _resolve_chat_route(khub.config, provider_override=provider, model_override=model)
    effective_allow_external = _allow_external_value(khub.config, route_provider, allow_external)
    session_id = new_session_id("chat")
    if prompt is None:
        if as_json:
            raise click.ClickException("--json requires PROMPT for khub chat")
        recorder = _new_recorder(
            save_session=bool(save_session),
            surface="chat",
            session_id=session_id,
            provider=route_provider,
            model=route_model,
            allow_external=effective_allow_external,
        )
        _repl(ctx, route_provider, route_model, effective_allow_external, allow_external, session_id, recorder=recorder)
        return

    recorder = _new_recorder(
        save_session=bool(save_session),
        surface="chat",
        session_id=session_id,
        provider=route_provider,
        model=route_model,
        allow_external=effective_allow_external,
    )
    try:
        if parse_paper_slash(str(prompt)) is not None:
            _single_turn_paper(ctx, str(prompt), allow_external, as_json, session_id, "turn_0001", recorder=recorder)
            return
        _single_turn(
            ctx,
            str(prompt),
            route_provider,
            route_model,
            effective_allow_external,
            as_json,
            session_id,
            "turn_0001",
            recorder=recorder,
        )
    finally:
        if recorder is not None:
            recorder.end()
