from __future__ import annotations

import json
import sys
import asyncio
from types import SimpleNamespace
from typing import Any, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from knowledge_hub.mcp.handlers.search import handle_tool as handle_search_tool
from knowledge_hub.mcp.tool_specs import build_tools

MCP_TOOL_STATUS_OK = "ok"
MCP_TOOL_STATUS_FAILED = "failed"

SERVER_STATE = SimpleNamespace(config=None, sqlite_db=None, searcher=None)
app = Server("knowledge-hub")


def to_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return default


def to_int(value: Any, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def to_float(value: Any, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def normalize_source(value: Any) -> str:
    source = str(value or "all").strip().lower()
    if source in {"all", "*"}:
        return "all"
    if source == "note":
        return "note"
    if source in {"paper", "web"}:
        return source
    return "all"


def _response(
    tool: str,
    status: str,
    payload: dict[str, Any] | Any,
    *,
    status_message: str | None = None,
    artifact: Any | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "tool": tool,
        "status": status,
        "payload": payload,
        "verify": {"status": "pass" if status == MCP_TOOL_STATUS_OK else "fail"},
    }
    if status_message:
        data["status_message"] = status_message
    if artifact is not None:
        data["artifact"] = artifact
    return data


def _text_response(data: dict[str, Any]) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False))]


async def list_tools() -> list[Tool]:
    return build_tools()


async def call_tool(name: str, arguments: Any | None = None) -> Sequence[TextContent]:
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return _text_response(_response(name, MCP_TOOL_STATUS_FAILED, {"error": "arguments는 object(dict)여야 합니다."}))

    initialize = getattr(SERVER_STATE, "initialize", None)
    if callable(initialize):
        initialize()

    def emit(
        status: str,
        payload: dict[str, Any] | Any,
        *,
        status_message: str | None = None,
        artifact: Any | None = None,
    ):
        return _text_response(_response(name, status, payload, status_message=status_message, artifact=artifact))

    result = await handle_search_tool(
        name,
        arguments,
        {
            "emit": emit,
            "searcher": SERVER_STATE.searcher,
            "sqlite_db": SERVER_STATE.sqlite_db,
            "normalize_source": normalize_source,
            "to_bool": to_bool,
            "to_int": to_int,
            "to_float": to_float,
            "MCP_TOOL_STATUS_OK": MCP_TOOL_STATUS_OK,
            "MCP_TOOL_STATUS_FAILED": MCP_TOOL_STATUS_FAILED,
        },
    )
    if result is not None:
        return result
    return _text_response(_response(name, MCP_TOOL_STATUS_FAILED, {"error": f"unknown tool: {name}"}))


@app.list_tools()
async def list_tools_impl() -> list[Tool]:
    return await list_tools()


@app.call_tool()
async def call_tool_impl(name: str, arguments: Any) -> Sequence[TextContent]:
    return await call_tool(name, arguments)


async def _async_main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        sys.exit(0)
