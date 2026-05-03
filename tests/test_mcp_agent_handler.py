from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest

from knowledge_hub.mcp.handlers import agent as agent_handler


def _import_mcp_server():
    try:
        return importlib.import_module("knowledge_hub.interfaces.mcp.server")
    except SystemExit:
        pytest.skip("mcp dependency is unavailable in test environment")


def _emit(status, payload, **kwargs):  # noqa: ANN001
    return {"status": status, "payload": payload, "meta": kwargs}


def test_agent_handler_requires_goal():
    module = _import_mcp_server()
    ctx = {
        "emit": _emit,
        "to_bool": module._to_bool,
        "to_int": module._to_int,
        "run_async_tool": None,
        "request_echo": {"tool": "run_agentic_query"},
        "searcher": SimpleNamespace(search=lambda *_a, **_k: [], generate_answer=lambda *_a, **_k: {"answer": "ok"}),
        "run_foundry_agent_goal": module._run_foundry_agent_goal,
        "coerce_foundry_payload": module._coerce_foundry_payload,
        "normalize_foundry_payload": module._normalize_foundry_payload,
        "write_agent_run_report": module._write_agent_run_report,
        "build_fallback_agent_payload": module._build_fallback_agent_payload,
        "MCP_TOOL_STATUS_FAILED": module.MCP_TOOL_STATUS_FAILED,
        "MCP_TOOL_STATUS_QUEUED": module.MCP_TOOL_STATUS_QUEUED,
    }
    result = asyncio.run(agent_handler.handle_tool("run_agentic_query", {}, ctx))
    assert result["status"] == "failed"


def test_agent_handler_uses_delegated_foundry_payload():
    module = _import_mcp_server()
    captured = {}

    async def _fake_run_async_tool(name, request_echo, sync_job):  # noqa: ANN001
        _ = (name, request_echo)
        captured["normalized"] = await sync_job()
        return "job1", {"payload": {"message": "queued"}}

    def _fake_run_foundry_agent_goal(**kwargs):  # noqa: ANN003
        _ = kwargs
        return json.dumps(
            {
                "runId": "run_001",
                "status": "completed",
                "goal": "rag",
                "verify": {"allowed": True, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
                "transitions": [{"stage": "PLAN", "status": "PLAN", "message": "plan"}],
            }
        ), None

    ctx = {
        "emit": _emit,
        "to_bool": module._to_bool,
        "to_int": module._to_int,
        "run_async_tool": _fake_run_async_tool,
        "request_echo": {"tool": "run_agentic_query"},
        "searcher": SimpleNamespace(search=lambda *_a, **_k: [], generate_answer=lambda *_a, **_k: {"answer": "ok"}),
        "run_foundry_agent_goal": _fake_run_foundry_agent_goal,
        "coerce_foundry_payload": module._coerce_foundry_payload,
        "normalize_foundry_payload": module._normalize_foundry_payload,
        "write_agent_run_report": module._write_agent_run_report,
        "build_fallback_agent_payload": module._build_fallback_agent_payload,
        "MCP_TOOL_STATUS_FAILED": module.MCP_TOOL_STATUS_FAILED,
        "MCP_TOOL_STATUS_QUEUED": module.MCP_TOOL_STATUS_QUEUED,
    }
    result = asyncio.run(agent_handler.handle_tool("run_agentic_query", {"goal": "rag"}, ctx))
    assert result["status"] == "queued"
    assert result["meta"]["job_id"] == "job1"
    assert captured["normalized"]["source"] == "foundry-core/cli-agent"
    assert "gateway" not in captured["normalized"]
    assert module.validate_payload(captured["normalized"], captured["normalized"]["schema"], strict=True).ok


def test_agent_handler_dry_run_adds_gateway_metadata_on_delegated_payload():
    module = _import_mcp_server()
    captured = {}

    async def _fake_run_async_tool(name, request_echo, sync_job):  # noqa: ANN001
        _ = (name, request_echo)
        captured["normalized"] = await sync_job()
        return "job1b", {"payload": {"message": "queued"}}

    def _fake_run_foundry_agent_goal(**kwargs):  # noqa: ANN003
        assert kwargs["dry_run"] is True
        return json.dumps(
            {
                "runId": "run_001b",
                "status": "blocked",
                "goal": "rag",
                "dryRun": True,
                "verify": {"allowed": False, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
                "transitions": [{"stage": "PLAN", "status": "PLAN", "message": "plan"}],
            }
        ), None

    ctx = {
        "emit": _emit,
        "to_bool": module._to_bool,
        "to_int": module._to_int,
        "run_async_tool": _fake_run_async_tool,
        "request_echo": {"tool": "run_agentic_query"},
        "searcher": SimpleNamespace(search=lambda *_a, **_k: [], generate_answer=lambda *_a, **_k: {"answer": "ok"}),
        "run_foundry_agent_goal": _fake_run_foundry_agent_goal,
        "coerce_foundry_payload": module._coerce_foundry_payload,
        "normalize_foundry_payload": module._normalize_foundry_payload,
        "write_agent_run_report": module._write_agent_run_report,
        "build_fallback_agent_payload": module._build_fallback_agent_payload,
        "MCP_TOOL_STATUS_FAILED": module.MCP_TOOL_STATUS_FAILED,
        "MCP_TOOL_STATUS_QUEUED": module.MCP_TOOL_STATUS_QUEUED,
    }
    result = asyncio.run(agent_handler.handle_tool("run_agentic_query", {"goal": "rag", "dry_run": True}, ctx))
    assert result["status"] == "queued"
    assert captured["normalized"]["gateway"]["surface"] == "agent_run"
    assert captured["normalized"]["gateway"]["mode"] == "dry_run"


def test_agent_handler_fallback_executes_search_and_answer():
    module = _import_mcp_server()
    captured = {}

    class _FakeSearcher:
        def search(self, *_args, **_kwargs):
            return [SimpleNamespace(metadata={"title": "doc", "source_type": "note"}, score=0.9)]

        def generate_answer(self, *_args, **_kwargs):
            return {"answer": "fallback answer", "sources": []}

    async def _fake_run_async_tool(name, request_echo, sync_job):  # noqa: ANN001
        _ = (name, request_echo)
        captured["normalized"] = await sync_job()
        return "job2", {"payload": {"message": "queued"}}

    ctx = {
        "emit": _emit,
        "to_bool": module._to_bool,
        "to_int": module._to_int,
        "run_async_tool": _fake_run_async_tool,
        "request_echo": {"tool": "run_agentic_query"},
        "searcher": _FakeSearcher(),
        "run_foundry_agent_goal": lambda **_kwargs: (None, "bridge unavailable"),
        "coerce_foundry_payload": module._coerce_foundry_payload,
        "normalize_foundry_payload": module._normalize_foundry_payload,
        "write_agent_run_report": module._write_agent_run_report,
        "build_fallback_agent_payload": module._build_fallback_agent_payload,
        "MCP_TOOL_STATUS_FAILED": module.MCP_TOOL_STATUS_FAILED,
        "MCP_TOOL_STATUS_QUEUED": module.MCP_TOOL_STATUS_QUEUED,
    }
    result = asyncio.run(
        agent_handler.handle_tool(
            "run_agentic_query",
            {"goal": "search and compare", "orchestratorMode": "strict", "dry_run": True},
            ctx,
        )
    )
    assert result["status"] == "queued"
    assert captured["normalized"]["source"] == "knowledge-hub/interfaces.mcp.server"
    assert captured["normalized"]["gateway"]["surface"] == "agent_run"
    assert captured["normalized"]["gateway"]["mode"] == "dry_run"
    assert "verify" in captured["normalized"]
    assert module.validate_payload(captured["normalized"], captured["normalized"]["schema"], strict=True).ok


def test_agent_handler_coding_goal_builds_task_context_before_answer():
    module = _import_mcp_server()
    captured = {}

    class _FakeSearcher:
        def search(self, *_args, **_kwargs):
            return [SimpleNamespace(metadata={"title": "doc", "source_type": "note"}, score=0.9, document="note evidence")]

        def generate_answer(self, *_args, **_kwargs):
            return {"answer": "fallback answer", "sources": []}

    async def _fake_run_async_tool(name, request_echo, sync_job):  # noqa: ANN001
        _ = (name, request_echo)
        captured["normalized"] = await sync_job()
        return "job3", {"payload": {"message": "queued"}}

    ctx = {
        "emit": _emit,
        "to_bool": module._to_bool,
        "to_int": module._to_int,
        "run_async_tool": _fake_run_async_tool,
        "request_echo": {"tool": "run_agentic_query"},
        "searcher": _FakeSearcher(),
        "run_foundry_agent_goal": lambda **_kwargs: (None, "bridge unavailable"),
        "coerce_foundry_payload": module._coerce_foundry_payload,
        "normalize_foundry_payload": module._normalize_foundry_payload,
        "write_agent_run_report": module._write_agent_run_report,
        "build_fallback_agent_payload": module._build_fallback_agent_payload,
        "MCP_TOOL_STATUS_FAILED": module.MCP_TOOL_STATUS_FAILED,
        "MCP_TOOL_STATUS_QUEUED": module.MCP_TOOL_STATUS_QUEUED,
    }
    result = asyncio.run(
        agent_handler.handle_tool(
            "run_agentic_query",
            {
                "goal": "Implement task context for agent runtime",
                "repo_path": "/tmp/nonexistent",
                "dry_run": True,
            },
            ctx,
        )
    )
    assert result["status"] == "queued"
    assert captured["normalized"]["plan"] == ["build_task_context", "ask_knowledge"]
    artifact = captured["normalized"]["artifact"]["jsonContent"]
    assert artifact["taskContext"]["mode"] == "coding"
    assert artifact["taskContext"]["gateway"]["surface"] == "task_context"
    assert artifact["taskContext"]["gateway"]["mode"] == "context"
    assert artifact["persistentKnowledgeEvidence"][0]["title"] == "doc"
    assert "workspace context skipped" in "\n".join(artifact["taskContext"]["warnings"])
    assert module.validate_payload(captured["normalized"], captured["normalized"]["schema"], strict=True).ok
