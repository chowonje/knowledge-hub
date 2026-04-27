from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Sequence
from uuid import uuid4

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from knowledge_hub.application.agent.foundry_bridge import run_foundry_agent_goal
from knowledge_hub.application.mcp.agent_payloads import (
    build_fallback_agent_payload,
    coerce_foundry_payload,
    normalize_foundry_payload,
    sanitize_agent_packet_value,
    transition_code,
    write_agent_run_report,
)
from knowledge_hub.application.mcp.jobs import ACTIVE_MCP_JOBS, run_async_tool
from knowledge_hub.application.mcp.responses import (
    CORE_ONLY_TOOL_NAMES,
    JOB_TOOLS,
    MCP_TOOL_STATUS_BLOCKED,
    MCP_TOOL_STATUS_DONE,
    MCP_TOOL_STATUS_EXPIRED,
    MCP_TOOL_STATUS_FAILED,
    MCP_TOOL_STATUS_OK,
    MCP_TOOL_STATUS_QUEUED,
    MCP_TOOL_STATUS_RUNNING,
    build_mcp_tool_response,
    build_text_response,
    build_verify_block,
    collect_source_refs,
    evaluate_policy_gate,
    infer_classification,
    normalize_source,
    now_iso,
    to_bool,
    to_float,
    to_int,
    to_tool_status,
)
from knowledge_hub.application.mcp.runtime import ensure_tool_runtime, initialize_core_runtime, initialize_search_runtime
from knowledge_hub.core.sanitizer import redact_payload
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.learning import LearningCoachService
from knowledge_hub.mcp.dispatch import dispatch_tool
from knowledge_hub.web import WebIngestService

SERVER_STATE = SimpleNamespace(
    config=None,
    sqlite_db=None,
    searcher=None,
    config_path=None,
    active_jobs=ACTIVE_MCP_JOBS,
)

app = Server("knowledge-hub")


def initialize(state: Any = SERVER_STATE):
    return initialize_search_runtime(state)


def initialize_core_only(state: Any = SERVER_STATE):
    return initialize_core_runtime(state)


_run_foundry_agent_goal = run_foundry_agent_goal
_coerce_foundry_payload = coerce_foundry_payload
_write_agent_run_report = write_agent_run_report
_to_bool = to_bool
_to_int = to_int
_to_float = to_float
_normalize_source = normalize_source


def _normalize_foundry_payload(payload: dict[str, object], goal: str, max_rounds: int, dry_run: bool) -> dict[str, object]:
    return normalize_foundry_payload(
        payload,
        goal=goal,
        max_rounds=max_rounds,
        dry_run=dry_run,
        default_source="knowledge-hub/interfaces.mcp.server",
        evaluate_policy_gate_fn=evaluate_policy_gate,
        transition_code_fn=transition_code,
    )


def _build_fallback_agent_payload(
    goal: str,
    max_rounds: int,
    dry_run: bool,
    plan: list[str],
    artifact,
    verify_ok: bool,
    errors: list[str],
    trace: list[dict],
    role: str | None = None,
    orchestrator_mode: str | None = None,
    playbook: dict[str, object] | None = None,
) -> str:
    return build_fallback_agent_payload(
        goal=goal,
        max_rounds=max_rounds,
        dry_run=dry_run,
        plan=plan,
        artifact=artifact,
        verify_ok=verify_ok,
        errors=errors,
        trace=trace,
        role=role,
        orchestrator_mode=orchestrator_mode,
        playbook=playbook,
        source="knowledge-hub/interfaces.mcp.server",
        evaluate_policy_gate_fn=evaluate_policy_gate,
        transition_code_fn=transition_code,
    )


def _build_verify_block(payload: dict[str, Any] | Any, status: str, tool_name: str) -> dict[str, Any]:
    def _validate_strict(current_payload: dict[str, Any], schema_id: str):
        try:
            return validate_payload(current_payload, schema_id, strict=True)
        except TypeError:
            return validate_payload(current_payload, schema_id)

    return build_verify_block(
        payload,
        status,
        tool_name,
        validate_payload_fn=_validate_strict,
    )


def _build_mcp_tool_response(
    *,
    tool: str,
    status: str,
    payload: dict[str, Any] | Any,
    request_echo: dict[str, Any] | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    job_id: str | None = None,
    status_message: str | None = None,
    artifact: Any | None = None,
    source_refs: Sequence[str] | None = None,
) -> dict[str, Any]:
    return build_mcp_tool_response(
        tool=tool,
        status=status,
        payload=payload,
        request_echo=request_echo,
        started_at=started_at,
        finished_at=finished_at,
        job_id=job_id,
        status_message=status_message,
        artifact=artifact,
        source_refs=source_refs,
        build_verify_block_fn=_build_verify_block,
    )


async def _run_async_tool(
    tool: str,
    request_echo: dict[str, Any],
    sync_job: Any,
    started_at: str | None = None,
) -> tuple[str, dict[str, Any]]:
    return await run_async_tool(
        sqlite_db=SERVER_STATE.sqlite_db,
        tool=tool,
        request_echo=request_echo,
        sync_job=sync_job,
        started_at=started_at,
        build_verify_block_fn=_build_verify_block,
        build_mcp_tool_response_fn=_build_mcp_tool_response,
        collect_source_refs_fn=collect_source_refs,
        infer_classification_fn=infer_classification,
        to_float_fn=to_float,
        now_iso_fn=now_iso,
        status_done=MCP_TOOL_STATUS_DONE,
        status_failed=MCP_TOOL_STATUS_FAILED,
        status_blocked=MCP_TOOL_STATUS_BLOCKED,
        status_queued=MCP_TOOL_STATUS_QUEUED,
        status_running=MCP_TOOL_STATUS_RUNNING,
        status_expired=MCP_TOOL_STATUS_EXPIRED,
    )


async def list_tools_impl() -> list[Tool]:
    from knowledge_hub.mcp.tool_specs import build_tools

    return build_tools()


def _mcp_tool_profile_access(name: str) -> tuple[str, bool, bool]:
    from knowledge_hub.mcp.tool_specs import build_tools, resolve_tool_profile

    profile = resolve_tool_profile()
    active_tool_names = {tool.name for tool in build_tools(profile)}
    if name in active_tool_names:
        return profile, True, True
    all_tool_names = {tool.name for tool in build_tools("all")}
    return profile, False, name in all_tool_names


async def call_tool_impl(state: Any, name: str, arguments: Any) -> Sequence[TextContent]:
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return build_text_response(
            _build_mcp_tool_response(
                tool=name,
                status=MCP_TOOL_STATUS_FAILED,
                payload={"error": "arguments는 object(dict)여야 합니다."},
            ),
            compact=False,
        )

    compact = to_bool(arguments.get("compact"), default=False)
    started_at = now_iso()
    echo_arguments = redact_payload(dict(arguments))
    if name.startswith("agent_"):
        sanitized_echo, _ = sanitize_agent_packet_value(echo_arguments)
        echo_arguments = sanitized_echo if isinstance(sanitized_echo, dict) else echo_arguments
    if name in {"agent_policy_check", "agent_stage_memory"} and isinstance(echo_arguments, dict) and "payload" in echo_arguments:
        echo_arguments["payload"] = {
            "omitted": True,
            "reason": f"{name} does not echo inspected payload",
        }
    request_echo = {
        "tool": name,
        "arguments": echo_arguments,
    }
    profile, profile_allowed, known_tool = _mcp_tool_profile_access(name)
    if not known_tool:
        return build_text_response(
            _build_mcp_tool_response(
                tool=name,
                status=MCP_TOOL_STATUS_FAILED,
                payload={"error": f"알 수 없는 도구: {name}"},
                started_at=started_at,
                request_echo=request_echo,
                status_message="unknown tool",
            ),
            compact=compact,
        )
    if not profile_allowed:
        profile_hint = "Set KHUB_MCP_PROFILE=labs or KHUB_MCP_PROFILE=all to use experimental/operator tools."
        if name.startswith("agent_"):
            profile_hint = "Set KHUB_MCP_PROFILE=agent, KHUB_MCP_PROFILE=labs, or KHUB_MCP_PROFILE=all to use agent-safe tools."
        return build_text_response(
            _build_mcp_tool_response(
                tool=name,
                status=MCP_TOOL_STATUS_BLOCKED,
                payload={
                    "blockReason": "profile",
                    "error": f"tool is not available in current MCP profile: {name}",
                    "profile": profile,
                    "hint": profile_hint,
                },
                started_at=started_at,
                request_echo=request_echo,
                status_message="tool blocked by MCP profile",
            ),
            compact=compact,
        )
    initialize_fn = getattr(state, "initialize", None)
    if not callable(initialize_fn):
        def initialize_fn() -> None:
            initialize(state)

    initialize_core_only_fn = getattr(state, "initialize_core_only", None)
    if not callable(initialize_core_only_fn):
        def initialize_core_only_fn() -> None:
            initialize_core_only(state)

    learning_service_cls = getattr(state, "LearningCoachService", LearningCoachService)
    web_ingest_service_cls = getattr(state, "WebIngestService", WebIngestService)
    from knowledge_hub.notes import KoNoteEnricher, KoNoteMaterializer

    ko_note_materializer_cls = getattr(state, "KoNoteMaterializer", KoNoteMaterializer)
    ko_note_enricher_cls = getattr(state, "KoNoteEnricher", KoNoteEnricher)
    run_foundry_agent_goal_fn = getattr(state, "_run_foundry_agent_goal", None) or getattr(
        state,
        "run_foundry_agent_goal",
        run_foundry_agent_goal,
    )
    coerce_foundry_payload_fn = getattr(state, "_coerce_foundry_payload", None) or getattr(
        state,
        "coerce_foundry_payload",
        coerce_foundry_payload,
    )
    normalize_foundry_payload_fn = getattr(state, "_normalize_foundry_payload", None)
    if not callable(normalize_foundry_payload_fn):
        def normalize_foundry_payload_fn(payload, goal, max_rounds, dry_run):  # noqa: ANN001
            return normalize_foundry_payload(
                payload,
                goal=goal,
                max_rounds=max_rounds,
                dry_run=dry_run,
                default_source="knowledge-hub/interfaces.mcp.server",
                evaluate_policy_gate_fn=evaluate_policy_gate,
                transition_code_fn=transition_code,
            )

    write_agent_run_report_fn = getattr(state, "_write_agent_run_report", None) or getattr(
        state,
        "write_agent_run_report",
        write_agent_run_report,
    )
    build_fallback_agent_payload_fn = getattr(state, "_build_fallback_agent_payload", None)
    if not callable(build_fallback_agent_payload_fn):
        def build_fallback_agent_payload_fn(**kwargs):  # noqa: ANN003
            return _build_fallback_agent_payload(**kwargs)

    ensure_tool_runtime(
        state,
        tool_name=name,
        initialize_fn=initialize_fn,
        initialize_core_only_fn=initialize_core_only_fn,
    )

    def emit(status: str, payload: dict[str, Any], **kwargs: Any) -> list[TextContent]:
        response = _build_mcp_tool_response(
            tool=name,
            status=status,
            payload=payload,
            started_at=kwargs.get("started_at", started_at),
            finished_at=kwargs.get("finished_at"),
            job_id=kwargs.get("job_id"),
            request_echo=kwargs.get("request_echo", request_echo),
            status_message=kwargs.get("status_message"),
            artifact=kwargs.get("artifact"),
            source_refs=kwargs.get("source_refs"),
        )
        return build_text_response(response, compact=compact)

    async def run_async_tool_bound(
        tool: str | None = None,
        request_echo_payload: dict[str, Any] | None = None,
        sync_job: Any | None = None,
        started_at_value: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        tool = tool or kwargs.get("name")
        request_echo_payload = request_echo_payload or kwargs.get("request_echo")
        sync_job = sync_job or kwargs.get("sync_job")
        if started_at_value is None:
            started_at_value = kwargs.get("started_at")
        return await run_async_tool(
            sqlite_db=state.sqlite_db,
            tool=str(tool or ""),
            request_echo=dict(request_echo_payload or {}),
            sync_job=sync_job,
            started_at=started_at_value,
            build_verify_block_fn=_build_verify_block,
            build_mcp_tool_response_fn=_build_mcp_tool_response,
            collect_source_refs_fn=collect_source_refs,
            infer_classification_fn=infer_classification,
            to_float_fn=to_float,
            now_iso_fn=now_iso,
            status_done=MCP_TOOL_STATUS_DONE,
            status_failed=MCP_TOOL_STATUS_FAILED,
            status_blocked=MCP_TOOL_STATUS_BLOCKED,
            status_queued=MCP_TOOL_STATUS_QUEUED,
            status_running=MCP_TOOL_STATUS_RUNNING,
            status_expired=MCP_TOOL_STATUS_EXPIRED,
        )
    run_async_tool_fn = getattr(state, "_run_async_tool", None) or getattr(
        state,
        "run_async_tool",
        None,
    )
    if not callable(run_async_tool_fn):
        run_async_tool_fn = run_async_tool_bound

    dispatch_context = {
        "emit": emit,
        "searcher": state.searcher,
        "sqlite_db": state.sqlite_db,
        "get_sqlite_db": lambda: state.sqlite_db,
        "config": state.config,
        "initialize": initialize_fn,
        "initialize_core_only": initialize_core_only_fn,
        "to_bool": to_bool,
        "to_int": to_int,
        "to_float": to_float,
        "to_tool_status": to_tool_status,
        "normalize_source": normalize_source,
        "redact_payload": redact_payload,
        "started_at": started_at,
        "request_echo": request_echo,
        "config_path": getattr(state, "config_path", None),
        "active_jobs": getattr(state, "active_jobs", ACTIVE_MCP_JOBS),
        "LearningCoachService": learning_service_cls,
        "WebIngestService": web_ingest_service_cls,
        "KoNoteMaterializer": ko_note_materializer_cls,
        "KoNoteEnricher": ko_note_enricher_cls,
        "run_async_tool": run_async_tool_fn,
        "now_iso": now_iso,
        "uuid4": uuid4,
        "run_foundry_agent_goal": run_foundry_agent_goal_fn,
        "coerce_foundry_payload": coerce_foundry_payload_fn,
        "normalize_foundry_payload": normalize_foundry_payload_fn,
        "write_agent_run_report": write_agent_run_report_fn,
        "build_fallback_agent_payload": build_fallback_agent_payload_fn,
        "MCP_TOOL_STATUS_OK": MCP_TOOL_STATUS_OK,
        "MCP_TOOL_STATUS_FAILED": MCP_TOOL_STATUS_FAILED,
        "MCP_TOOL_STATUS_BLOCKED": MCP_TOOL_STATUS_BLOCKED,
        "MCP_TOOL_STATUS_QUEUED": MCP_TOOL_STATUS_QUEUED,
        "MCP_TOOL_STATUS_RUNNING": MCP_TOOL_STATUS_RUNNING,
    }
    dispatched = await dispatch_tool(name, arguments, dispatch_context)
    if dispatched is not None:
        state.config = dispatch_context.get("config", state.config)
        state.sqlite_db = dispatch_context.get("sqlite_db", state.sqlite_db)
        state.searcher = dispatch_context.get("searcher", state.searcher)
        return dispatched

    try:
        return emit(
            MCP_TOOL_STATUS_FAILED,
            {"error": f"알 수 없는 도구: {name}"},
            status_message="unknown tool",
        )
    except Exception as error:
        return build_text_response(
            _build_mcp_tool_response(
                tool=name,
                status=MCP_TOOL_STATUS_FAILED,
                payload={"error": str(error)},
                started_at=started_at,
                request_echo=request_echo,
                status_message="tool execution failed",
            ),
            compact=compact,
        )


@app.list_tools()
async def list_tools() -> list[Tool]:
    return await list_tools_impl()


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    return await call_tool_impl(SERVER_STATE, name, arguments)


async def _async_main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        return


__all__ = [
    "SERVER_STATE",
    "ACTIVE_MCP_JOBS",
    "CORE_ONLY_TOOL_NAMES",
    "JOB_TOOLS",
    "MCP_TOOL_STATUS_BLOCKED",
    "MCP_TOOL_STATUS_DONE",
    "MCP_TOOL_STATUS_EXPIRED",
    "MCP_TOOL_STATUS_FAILED",
    "MCP_TOOL_STATUS_OK",
    "MCP_TOOL_STATUS_QUEUED",
    "MCP_TOOL_STATUS_RUNNING",
    "app",
    "call_tool",
    "call_tool_impl",
    "initialize",
    "initialize_core_only",
    "list_tools",
    "list_tools_impl",
    "main",
    "_build_fallback_agent_payload",
    "_build_mcp_tool_response",
    "_build_verify_block",
    "_coerce_foundry_payload",
    "_normalize_foundry_payload",
    "_run_async_tool",
    "_run_foundry_agent_goal",
    "_to_bool",
    "_to_float",
    "_to_int",
    "_write_agent_run_report",
]
