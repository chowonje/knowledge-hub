from __future__ import annotations

import asyncio

from knowledge_hub.core.config import Config
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.mcp.handlers import paper as paper_handler


def _emit(status, payload, **kwargs):  # noqa: ANN001
    return {"status": status, "payload": payload, "meta": kwargs}


async def _run_async_tool(*, name, request_echo, sync_job):  # noqa: ANN001
    _ = (name, request_echo)
    payload = await sync_job()
    return "job_1", {"payload": payload}


def test_discover_and_ingest_handler_passes_judge_options(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeManager:
        def __init__(self, **kwargs):  # noqa: ANN003
            _ = kwargs

        def discover_and_ingest(self, **kwargs):  # noqa: ANN003
            captured.update(kwargs)
            return {
                "schema": "knowledge-hub.paper.discover.result.v1",
                "status": "ok",
                "topic": kwargs["topic"],
                "discovered": 3,
                "duplicates_skipped": 0,
                "ingested": [],
                "failed": [],
                "obsidian_notes_created": [],
                "warnings": [],
                "judge": {
                    "enabled": True,
                    "backend": "rule_llm_v1",
                    "threshold": 0.62,
                    "candidateCount": 6,
                    "selectedCount": 2,
                    "degraded": False,
                    "warnings": [],
                    "items": [],
                },
            }

    monkeypatch.setattr("knowledge_hub.papers.manager.PaperManager", _FakeManager)

    ctx = {
        "emit": _emit,
        "config": Config(),
        "sqlite_db": object(),
        "searcher": type("Searcher", (), {"database": object(), "embedder": object(), "llm": object()})(),
        "to_bool": lambda value, default=False: default if value is None else bool(value),
        "to_int": lambda value, default=None, minimum=None, maximum=None: default if value is None else int(value),
        "run_async_tool": _run_async_tool,
        "request_echo": {},
        "MCP_TOOL_STATUS_OK": "ok",
        "MCP_TOOL_STATUS_FAILED": "failed",
        "MCP_TOOL_STATUS_QUEUED": "queued",
    }

    result = asyncio.run(
        paper_handler.handle_tool(
            "discover_and_ingest",
            {
                "topic": "agent retrieval",
                "judge_enabled": True,
                "judge_threshold": 0.7,
                "judge_candidates": 12,
                "allow_external": True,
            },
            ctx,
        )
    )

    assert result["status"] == "queued"
    assert captured["judge_enabled"] is True
    assert captured["judge_threshold"] == 0.7
    assert captured["judge_candidates"] == 12
    assert captured["allow_external"] is True
    assert validate_payload(result["payload"], result["payload"]["schema"], strict=True).ok


def test_paper_topic_synthesize_handler_returns_schema_payload(monkeypatch):
    payload = {
        "schema": "knowledge-hub.paper-topic-synthesis.result.v1",
        "status": "ok",
        "query": "transformer alternatives",
        "sourceMode": "local",
        "effectiveSourceMode": "local",
        "candidatePapers": [{"paperId": "2401.00001", "title": "Mamba"}],
        "selectedPapers": [{"paperId": "2401.00001", "title": "Mamba", "decision": "keep", "rationale": "fit", "groupLabel": "architecture"}],
        "excludedPapers": [],
        "selectionDiagnostics": {},
        "topicSummary": "summary",
        "architectureGroups": [],
        "comparisonPoints": [],
        "limitations": [],
        "gaps": [],
        "citations": [],
        "verification": {"status": "verified", "summary": "ok", "warnings": [], "claims": []},
        "warnings": [],
    }

    monkeypatch.setattr(
        "knowledge_hub.mcp.handlers.paper.PaperTopicSynthesisService",
        type(
            "_StubService",
            (),
            {
                "__init__": lambda self, **kwargs: None,  # noqa: ARG005
                "synthesize": lambda self, **kwargs: payload,  # noqa: ARG005
            },
        ),
    )

    ctx = {
        "emit": _emit,
        "config": Config(),
        "sqlite_db": object(),
        "searcher": object(),
        "to_bool": lambda value, default=False: default if value is None else bool(value),
        "to_int": lambda value, default=None, minimum=None, maximum=None: default if value is None else int(value),
        "run_async_tool": _run_async_tool,
        "request_echo": {},
        "MCP_TOOL_STATUS_OK": "ok",
        "MCP_TOOL_STATUS_FAILED": "failed",
        "MCP_TOOL_STATUS_QUEUED": "queued",
    }

    result = asyncio.run(
        paper_handler.handle_tool(
            "paper_topic_synthesize",
            {"query": "transformer alternatives papers"},
            ctx,
        )
    )

    assert result["status"] == "ok"
    assert result["payload"]["selectedPapers"][0]["paperId"] == "2401.00001"
    assert validate_payload(result["payload"], result["payload"]["schema"], strict=True).ok
