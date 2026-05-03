from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace


def _import_mcp_server():
    return importlib.import_module("knowledge_hub.interfaces.mcp.server")


def _decode_response(contents):
    assert contents
    return json.loads(contents[0].text)


class _FakeSQLiteDB:
    def get_stats(self):
        return {"notes": 2, "papers": 1, "tags": 3, "links": 4}


class _FakeSearcher:
    def __init__(self):
        self.database = SimpleNamespace(get_stats=lambda: {"total_documents": 3, "collection_name": "knowledge_hub"})
        self.sqlite_db = _FakeSQLiteDB()

    def search(self, query, top_k=5, source_type="all", retrieval_mode="hybrid", alpha=0.7, expand_parent_context=True):
        _ = (query, top_k, source_type, retrieval_mode, alpha, expand_parent_context)
        return [
            SimpleNamespace(
                metadata={
                    "title": "RAG Note",
                    "source_type": "note",
                    "resolved_parent_id": "p1",
                    "resolved_parent_label": "Section A",
                    "resolved_parent_chunk_span": "0-2",
                },
                score=0.92,
                semantic_score=0.9,
                lexical_score=0.8,
                retrieval_mode="hybrid",
                lexical_extras={
                    "quality_flag": "ok",
                    "source_trust_score": 0.94,
                    "reference_role": "glossary_reference",
                    "reference_tier": "specialist",
                    "ranking_signals": {
                        "quality_flag": "ok",
                        "reference_role": "glossary_reference",
                        "reference_tier": "specialist",
                    },
                },
                document="retrieval augmented generation",
            )
        ]

    def generate_answer(self, question, **kwargs):
        _ = (question, kwargs)
        return {"answer": "RAG answer", "sources": []}


def _setup_fakes(module):
    module.SERVER_STATE.config = object()
    module.SERVER_STATE.sqlite_db = _FakeSQLiteDB()
    module.SERVER_STATE.searcher = _FakeSearcher()


def test_search_knowledge_requires_query_and_returns_result_shape():
    module = _import_mcp_server()
    _setup_fakes(module)

    failed = _decode_response(asyncio.run(module.call_tool("search_knowledge", {})))
    assert failed["status"] == "failed"
    assert "query" in failed["payload"]["error"]

    ok = _decode_response(asyncio.run(module.call_tool("search_knowledge", {"query": "rag"})))
    assert ok["status"] == "ok"
    assert ok["payload"]["result_count"] == 1
    assert ok["payload"]["results"][0]["title"] == "RAG Note"
    assert ok["payload"]["results"][0]["quality_flag"] == "ok"
    assert ok["payload"]["results"][0]["reference_tier"] == "specialist"
    assert ok["payload"]["results"][0]["normalized_source_type"] == "vault"
    assert ok["payload"]["runtimeDiagnostics"]["schema"] == "knowledge-hub.runtime.diagnostics.v1"


def test_build_task_context_returns_schema_valid_payload(tmp_path):
    module = _import_mcp_server()
    _setup_fakes(module)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("- Preserve boundaries\n", encoding="utf-8")
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "agent.ts").write_text("export const enabled = true;\n", encoding="utf-8")

    ok = _decode_response(
        asyncio.run(
            module.call_tool(
                "build_task_context",
                {"goal": "Implement src/agent.ts update", "repo_path": str(repo)},
            )
        )
    )
    assert ok["status"] == "ok"
    assert ok["payload"]["schema"] == "knowledge-hub.task-context.result.v1"
    assert ok["payload"]["mode"] == "coding"
    assert ok["payload"]["workspace_files"][0]["source_type"] == "project"
    assert ok["payload"]["gateway"]["surface"] == "task_context"
    assert ok["payload"]["runtimeDiagnostics"]["schema"] == "knowledge-hub.runtime.diagnostics.v1"


def test_list_tools_contains_only_core_context_serving_tools():
    module = _import_mcp_server()

    tools = asyncio.run(module.list_tools())
    names = {tool.name for tool in tools}
    assert names == {"search_knowledge", "ask_knowledge", "build_task_context", "get_hub_stats"}
