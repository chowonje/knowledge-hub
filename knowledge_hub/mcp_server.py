#!/usr/bin/env python3
"""Legacy MCP server shim.

The canonical MCP surface lives at ``knowledge_hub.interfaces.mcp.server``.
Keep this module as a thin compatibility layer for older import paths.
"""

from knowledge_hub.interfaces.mcp.server import (
    ACTIVE_MCP_JOBS,
    CORE_ONLY_TOOL_NAMES,
    JOB_TOOLS,
    MCP_TOOL_STATUS_BLOCKED,
    MCP_TOOL_STATUS_DONE,
    MCP_TOOL_STATUS_EXPIRED,
    MCP_TOOL_STATUS_FAILED,
    MCP_TOOL_STATUS_OK,
    MCP_TOOL_STATUS_QUEUED,
    MCP_TOOL_STATUS_RUNNING,
    SERVER_STATE,
    _build_fallback_agent_payload,
    _build_mcp_tool_response,
    _build_verify_block,
    _coerce_foundry_payload,
    _normalize_foundry_payload,
    _run_async_tool,
    _run_foundry_agent_goal,
    _to_bool,
    _to_float,
    _to_int,
    _write_agent_run_report,
    app,
    call_tool,
    call_tool_impl,
    initialize,
    initialize_core_only,
    list_tools,
    list_tools_impl,
    main,
)

__all__ = [
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
    "SERVER_STATE",
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


if __name__ == "__main__":
    main()
