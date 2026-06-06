from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from knowledge_hub.mcp.handlers import agent as agent_handler
from knowledge_hub.infrastructure.persistence.vector import VectorDatabase
from tests.remote_index_test_support import DeterministicEmbedder, embedding_vector, remote_index_config


def _import_mcp_server():
    try:
        return importlib.import_module("knowledge_hub.interfaces.mcp.server")
    except SystemExit:
        pytest.skip("mcp dependency is unavailable in test environment")


def _emit(status, payload, **kwargs):  # noqa: ANN001
    return {"status": status, "payload": payload, "meta": kwargs}


def _agent_handler_ctx(module, *, captured: dict, searcher):  # noqa: ANN001
    async def _fake_run_async_tool(name, request_echo, sync_job):  # noqa: ANN001
        _ = (name, request_echo)
        captured["normalized"] = await sync_job()
        return "job-qwen8", {"payload": {"message": "queued"}}

    return {
        "emit": _emit,
        "to_bool": module._to_bool,
        "to_int": module._to_int,
        "run_async_tool": _fake_run_async_tool,
        "request_echo": {"tool": "run_agentic_query"},
        "searcher": searcher,
        "run_foundry_agent_goal": lambda **_kwargs: (None, "bridge unavailable"),
        "coerce_foundry_payload": module._coerce_foundry_payload,
        "normalize_foundry_payload": module._normalize_foundry_payload,
        "write_agent_run_report": module._write_agent_run_report,
        "build_fallback_agent_payload": module._build_fallback_agent_payload,
        "MCP_TOOL_STATUS_FAILED": module.MCP_TOOL_STATUS_FAILED,
        "MCP_TOOL_STATUS_QUEUED": module.MCP_TOOL_STATUS_QUEUED,
    }


def _write_query_embedding_output(run_dir: Path, *, query_text: str = "retrieval planning") -> None:
    output_path = run_dir / "query_embeddings.qwen3-8b.jsonl"
    row = {
        "query_id": "query:0001",
        "query": query_text,
        "query_text_hash": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
        "embedding_model": "qwen3-embedding:8b",
        "embedding_dim": 4096,
        "embedding": embedding_vector(4096),
    }
    output_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    (run_dir / "checksums.output.sha256").write_text(f"{digest}  {output_path.name}\n", encoding="utf-8")


def _seed_qwen8_agent_runtime(tmp_path: Path):
    from knowledge_hub.application.paper_query_export import export_query_embedding_bundle

    config = remote_index_config(tmp_path)
    baseline = VectorDatabase(config.vector_db_path, config.collection_name, repair_on_init=False)
    baseline.add_documents(
        documents=["Agentic retrieval planning paper baseline evidence chunk."],
        embeddings=[DeterministicEmbedder("nomic-embed-text").embed_text("retrieval planning")],
        metadatas=[{"paper_id": "paper-bge", "chunk_id": "paper-bge:1", "source_type": "paper", "stale": 0}],
        ids=["bge:paper-bge:0001"],
    )
    qwen = VectorDatabase(config.vector_db_path, "qwen3_8b_full_candidate", repair_on_init=False)
    qwen.add_documents(
        documents=["Qwen8 retrieval planning paper side evidence chunk."],
        embeddings=[embedding_vector(4096)],
        metadatas=[{"paper_id": "paper-qwen", "chunk_id": "paper-qwen:1", "source_type": "paper", "stale": 0}],
        ids=["qwen3_8b_full_candidate:paper-qwen:0001"],
    )
    run_dir = tmp_path / "query-run"
    export_query_embedding_bundle(queries=("retrieval planning",), out_dir=run_dir)
    _write_query_embedding_output(run_dir)
    return config, run_dir


class _Qwen8FakeSearcher:
    def __init__(self, config):
        self.config = config

    def build_embedder(self, provider: str, model: str | None = None):  # noqa: ANN001
        if str(model or "").startswith("qwen3-embedding:"):
            raise AssertionError("qwen embedding must come from artifact, not local runtime")
        return DeterministicEmbedder(model or "nomic-embed-text")

    def search(self, *_args, **_kwargs):
        return [SimpleNamespace(metadata={"title": "doc", "source_type": "paper"}, score=0.9)]

    def generate_answer(self, *_args, **_kwargs):
        return {"answer": "fallback answer", "sources": []}


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


def test_agent_handler_qwen8_paper_harness_off_by_default(tmp_path):
    module = _import_mcp_server()
    config, _run_dir = _seed_qwen8_agent_runtime(tmp_path)
    captured = {}
    ctx = _agent_handler_ctx(module, captured=captured, searcher=_Qwen8FakeSearcher(config))

    result = asyncio.run(
        agent_handler.handle_tool(
            "run_agentic_query",
            {"goal": "search retrieval planning papers", "orchestratorMode": "strict", "dry_run": True},
            ctx,
        )
    )

    assert result["status"] == "queued"
    artifact = captured["normalized"]["artifact"]["jsonContent"]
    assert "paperEvidencePack" not in artifact
    assert module.validate_payload(captured["normalized"], captured["normalized"]["schema"], strict=True).ok


def test_agent_handler_attaches_qwen8_paper_evidence_pack_when_query_run_supplied(tmp_path):
    module = _import_mcp_server()
    config, run_dir = _seed_qwen8_agent_runtime(tmp_path)
    captured = {}
    ctx = _agent_handler_ctx(module, captured=captured, searcher=_Qwen8FakeSearcher(config))

    result = asyncio.run(
        agent_handler.handle_tool(
            "run_agentic_query",
            {
                "goal": "search retrieval planning papers",
                "orchestratorMode": "strict",
                "dry_run": True,
                "paperQueryRun": str(run_dir),
                "paperQueryId": "query:0001",
            },
            ctx,
        )
    )

    assert result["status"] == "queued"
    artifact = captured["normalized"]["artifact"]["jsonContent"]
    assert artifact["paperEvidencePack"]["schema"] == "knowledge-hub.labs.paper-retrieval-harness.v1"
    assert artifact["paperEvidencePack"]["status"] == "ok"
    assert {arm["name"]: arm["status"] for arm in artifact["paperEvidencePack"]["arms"]}["qwen8"] == "ok"
    assert module.validate_payload(captured["normalized"], captured["normalized"]["schema"], strict=True).ok


def test_agent_handler_attaches_qwen8_paper_evidence_pack_to_delegated_payload(tmp_path):
    module = _import_mcp_server()
    config, run_dir = _seed_qwen8_agent_runtime(tmp_path)
    captured = {}

    async def _fake_run_async_tool(name, request_echo, sync_job):  # noqa: ANN001
        _ = (name, request_echo)
        captured["normalized"] = await sync_job()
        return "job-qwen8-delegated", {"payload": {"message": "queued"}}

    def _fake_run_foundry_agent_goal(**kwargs):  # noqa: ANN003
        _ = kwargs
        return json.dumps(
            {
                "runId": "run_qwen8_delegated",
                "status": "completed",
                "goal": "search retrieval planning papers",
                "artifact": {"jsonContent": {"answer": "delegated answer"}, "classification": "P2"},
                "verify": {"allowed": True, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
                "transitions": [{"stage": "PLAN", "status": "PLAN", "message": "plan"}],
            }
        ), None

    ctx = _agent_handler_ctx(module, captured=captured, searcher=_Qwen8FakeSearcher(config))
    ctx["run_async_tool"] = _fake_run_async_tool
    ctx["run_foundry_agent_goal"] = _fake_run_foundry_agent_goal

    result = asyncio.run(
        agent_handler.handle_tool(
            "run_agentic_query",
            {
                "goal": "search retrieval planning papers",
                "paperQueryRun": str(run_dir),
                "paperQueryId": "query:0001",
            },
            ctx,
        )
    )

    assert result["status"] == "queued"
    artifact = captured["normalized"]["artifact"]["jsonContent"]
    assert artifact["answer"] == "delegated answer"
    assert artifact["paperEvidencePack"]["schema"] == "knowledge-hub.labs.paper-retrieval-harness.v1"
    assert {arm["name"]: arm["status"] for arm in artifact["paperEvidencePack"]["arms"]}["qwen8"] == "ok"
    assert module.validate_payload(captured["normalized"], captured["normalized"]["schema"], strict=True).ok


def test_agent_handler_blocks_qwen8_missing_query_artifact_without_external_call(tmp_path):
    from knowledge_hub.application.paper_query_export import export_query_embedding_bundle

    module = _import_mcp_server()
    config = remote_index_config(tmp_path)
    run_dir = tmp_path / "query-run"
    export_query_embedding_bundle(queries=("retrieval planning",), out_dir=run_dir)
    captured = {}
    ctx = _agent_handler_ctx(module, captured=captured, searcher=_Qwen8FakeSearcher(config))

    result = asyncio.run(
        agent_handler.handle_tool(
            "run_agentic_query",
            {
                "goal": "search retrieval planning papers",
                "orchestratorMode": "strict",
                "dry_run": True,
                "paperQueryRun": str(run_dir),
                "paperQueryId": "query:0001",
            },
            ctx,
        )
    )

    assert result["status"] == "queued"
    artifact = captured["normalized"]["artifact"]["jsonContent"]
    assert artifact["answer"] == "fallback answer"
    assert artifact["paperEvidencePack"]["status"] == "blocked"
    assert "query_embedding_file_missing" in artifact["paperEvidencePack"]["blockers"]
    assert module.validate_payload(captured["normalized"], captured["normalized"]["schema"], strict=True).ok
