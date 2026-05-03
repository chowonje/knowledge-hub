from __future__ import annotations

import asyncio

from knowledge_hub.mcp.handlers import search as search_handler


def _emit(status, payload, **kwargs):  # noqa: ANN001
    return {"status": status, "payload": payload, "meta": kwargs}


class _FakeSearcher:
    def __init__(self):
        self.sqlite_db = object()
        self.config = object()

    def search(self, *args, **kwargs):  # noqa: ANN001
        _ = (args, kwargs)
        return [
            type(
                "Result",
                (),
                {
                    "metadata": {
                        "title": "RAG Note",
                        "source_type": "vault",
                        "file_path": "Projects/AI/RAG Note.md",
                        "links": ["Projects/AI/RAG Architecture.md"],
                        "resolved_parent_id": "vault:rag-note",
                        "resolved_parent_label": "RAG Note",
                        "resolved_parent_chunk_span": "0",
                    },
                    "score": 0.91,
                    "semantic_score": 0.88,
                    "lexical_score": 0.77,
                    "retrieval_mode": "hybrid",
                    "lexical_extras": {},
                    "document": "retrieval augmented generation note",
                },
            )(),
            type(
                "Result",
                (),
                {
                    "metadata": {
                        "title": "RAG Architecture",
                        "source_type": "vault",
                        "file_path": "Projects/AI/RAG Architecture.md",
                        "cluster_id": "cluster-rag",
                        "resolved_parent_id": "vault:rag-architecture",
                        "resolved_parent_label": "RAG Architecture",
                        "resolved_parent_chunk_span": "0",
                    },
                    "score": 0.83,
                    "semantic_score": 0.79,
                    "lexical_score": 0.61,
                    "retrieval_mode": "hybrid",
                    "lexical_extras": {},
                    "document": "rag architecture note",
                },
            )(),
        ]

    def generate_answer(self, *args, **kwargs):  # noqa: ANN001
        _ = (args, kwargs)
        return {
            "answer": "RAG answer",
            "sources": [
                {
                    "title": "RAG Note",
                    "source_type": "vault",
                    "score": 0.91,
                    "semantic_score": 0.88,
                    "lexical_score": 0.77,
                }
            ],
            "citations": [{"label": "S1", "title": "RAG Note", "target": "Projects/AI/RAG Note.md", "kind": "file"}],
            "answerGeneration": {"status": "fallback", "fallbackUsed": True, "stage": "initial_answer", "errorType": "TimeoutError"},
            "claimVerification": [{"claimId": "claim:1", "status": "supported"}],
            "claimConsensus": {"supportCount": 1, "claimVerificationSummary": "supported"},
            "paperAnswerScope": {"applied": False},
            "evidenceBudget": {"maxSources": 6},
        }


def test_mcp_search_handler_adds_related_notes_and_graph_signal(monkeypatch):
    monkeypatch.setattr(search_handler, "_runtime_diagnostics", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(
        search_handler,
        "_graph_query_signal",
        lambda searcher, query: {"is_graph_heavy": True, "recommended_mode": "graph"},
    )

    ctx = {
        "emit": _emit,
        "searcher": _FakeSearcher(),
        "sqlite_db": object(),
        "normalize_source": lambda value: value,
        "to_int": lambda value, default=None, minimum=None, maximum=None: default if value is None else int(value),
        "to_float": lambda value, default=None, minimum=None, maximum=None: default if value is None else float(value),
        "MCP_TOOL_STATUS_OK": "ok",
        "MCP_TOOL_STATUS_FAILED": "failed",
    }

    result = asyncio.run(search_handler.handle_tool("search_knowledge", {"query": "rag"}, ctx))

    assert result["status"] == "ok"
    payload = result["payload"]
    assert payload["graph_query_signal"]["is_graph_heavy"] is True
    assert payload["related_notes"]
    assert payload["results"][0]["title"] == "RAG Note"


def test_mcp_ask_handler_exposes_graph_signal_and_answer_diagnostics(monkeypatch):
    monkeypatch.setattr(search_handler, "_runtime_diagnostics", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(
        search_handler,
        "_graph_query_signal",
        lambda searcher, query: {"is_graph_heavy": False, "recommended_mode": "baseline"},
    )

    ctx = {
        "emit": _emit,
        "searcher": _FakeSearcher(),
        "sqlite_db": object(),
        "normalize_source": lambda value: value,
        "to_int": lambda value, default=None, minimum=None, maximum=None: default if value is None else int(value),
        "to_float": lambda value, default=None, minimum=None, maximum=None: default if value is None else float(value),
        "MCP_TOOL_STATUS_OK": "ok",
        "MCP_TOOL_STATUS_FAILED": "failed",
    }

    result = asyncio.run(search_handler.handle_tool("ask_knowledge", {"question": "What is RAG?"}, ctx))

    assert result["status"] == "ok"
    payload = result["payload"]
    assert payload["graph_query_signal"]["recommended_mode"] == "baseline"
    assert payload["citations"][0]["label"] == "S1"
    assert payload["answer_generation"]["fallbackUsed"] is True
    assert payload["claim_verification"][0]["claimId"] == "claim:1"
    assert payload["claim_consensus"]["supportCount"] == 1
    assert payload["paper_answer_scope"]["applied"] is False
    assert payload["evidence_budget"]["maxSources"] == 6
