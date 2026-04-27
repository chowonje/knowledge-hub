from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest

from knowledge_hub.application.mcp.agent_payloads import build_agent_context_packet, default_agent_policy
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.mcp.handlers import agent as agent_handler
from knowledge_hub.mcp import tool_specs


def _decode_response(contents):
    assert contents
    return json.loads(contents[0].text)


def _import_mcp_server():
    try:
        return importlib.import_module("knowledge_hub.interfaces.mcp.server")
    except SystemExit:
        pytest.skip("mcp dependency is unavailable in test environment")


def _emit(status, payload, **kwargs):  # noqa: ANN001
    return {"status": status, "payload": payload, "meta": kwargs}


def _ctx(searcher=None):
    return {
        "emit": _emit,
        "searcher": searcher or _FakeSearcher(),
        "sqlite_db": object(),
        "normalize_source": lambda value: str(value or "all"),
        "to_bool": lambda value, default=False: default if value is None else bool(value),
        "to_int": lambda value, default=None, minimum=None, maximum=None: default if value is None else int(value),
        "to_float": lambda value, default=None, minimum=None, maximum=None: default if value is None else float(value),
        "redact_payload": lambda value: value,
        "uuid4": lambda: "agent-test-request",
        "MCP_TOOL_STATUS_OK": "ok",
        "MCP_TOOL_STATUS_FAILED": "failed",
    }


class _FakeSearcher:
    def __init__(self, verdict: str = "pass"):
        self.sqlite_db = object()
        self.config = object()
        self.database = SimpleNamespace(get_stats=lambda: {"total_documents": 1, "collection_name": "test"})
        self.verdict = verdict

    def search(self, *args, **kwargs):  # noqa: ANN001
        _ = (args, kwargs)
        return [
            SimpleNamespace(
                metadata={
                    "title": "RAG Note",
                    "source_type": "note",
                    "resolved_parent_id": "note:rag",
                    "resolved_parent_label": "RAG Note",
                    "resolved_parent_chunk_span": "0",
                },
                score=0.9,
                semantic_score=0.88,
                lexical_score=0.7,
                retrieval_mode="hybrid",
                lexical_extras={},
                document="RAG uses retrieved local evidence before answering.",
            )
        ]

    def generate_answer(self, *args, **kwargs):  # noqa: ANN001
        _ = (args, kwargs)
        unsupported = 1 if self.verdict == "fail" else 0
        return {
            "answer": "RAG uses retrieved local evidence.",
            "sources": [
                {
                    "title": "RAG Note",
                    "source_type": "note",
                    "score": 0.9,
                    "semantic_score": 0.88,
                    "lexical_score": 0.7,
                }
            ],
            "citations": [{"label": "S1", "title": "RAG Note", "target": "vault/RAG.md"}],
            "evidencePacketContract": {"schema": "knowledge-hub.evidence-packet.v1", "packet_id": "evidence:1"},
            "answerContract": {"schema": "knowledge-hub.answer-contract.v1", "answer_id": "answer:1"},
            "verificationVerdict": {
                "schema": "knowledge-hub.verification-verdict.v1",
                "verdict": self.verdict,
                "recommended_action": "abstain" if self.verdict == "abstain" else "use",
                "unsupportedClaimCount": unsupported,
            },
        }


def test_agent_context_packet_schema_validates():
    packet = build_agent_context_packet(
        request_id="req-1",
        tool="agent_search_knowledge",
        goal="find RAG notes",
        query="RAG",
        policy=default_agent_policy(),
        context={"results": []},
        next_actions=["agent_ask_knowledge"],
    )

    result = validate_payload(packet, "knowledge-hub.agent.context-packet.v1", strict=True)

    assert result.ok, result.errors


def test_agent_profile_exposes_agent_tools_but_default_hides():
    default_names = {tool.name for tool in tool_specs.build_tools(profile="default")}
    agent_names = {tool.name for tool in tool_specs.build_tools(profile="agent")}
    labs_names = {tool.name for tool in tool_specs.build_tools(profile="labs")}

    assert "agent_ask_knowledge" not in default_names
    assert "agent_ask_knowledge" in agent_names
    assert "agent_policy_check" in agent_names
    assert "ask_knowledge" in agent_names
    assert "mcp_job_status" in agent_names
    assert "ops_action_list" not in agent_names
    assert "agent_ask_knowledge" in labs_names


def test_default_profile_blocks_agent_tool_direct_call(monkeypatch):
    monkeypatch.delenv("KHUB_MCP_PROFILE", raising=False)
    module = _import_mcp_server()

    blocked = _decode_response(asyncio.run(module.call_tool("agent_policy_check", {"payload": {"ok": True}})))

    assert blocked["status"] == "blocked"
    assert blocked["payload"]["profile"] == "default"
    assert "KHUB_MCP_PROFILE=agent" in blocked["payload"]["hint"]


def test_agent_profile_policy_check_returns_schema_valid_mcp_response(monkeypatch):
    monkeypatch.setenv("KHUB_MCP_PROFILE", "agent")
    module = _import_mcp_server()
    module.SERVER_STATE.config = object()
    module.SERVER_STATE.sqlite_db = object()
    module.SERVER_STATE.searcher = None
    module.SERVER_STATE.initialize_core_only = lambda: None
    module.SERVER_STATE.initialize = lambda: pytest.fail("agent_policy_check should not initialize search runtime")

    ok = _decode_response(
        asyncio.run(module.call_tool("agent_policy_check", {"payload": {"summary": "public local note"}}))
    )

    assert ok["status"] == "ok"
    assert ok["payload"]["schema"] == "knowledge-hub.agent.context-packet.v1"
    assert ok["payload"]["policy"]["policyMode"] == "local-only"
    assert ok["verify"]["schemaValid"] is True
    assert ok["verify"]["policyAllowed"] is True


def test_agent_profile_policy_check_p0_preserves_packet_and_omits_payload_echo(monkeypatch):
    monkeypatch.setenv("KHUB_MCP_PROFILE", "agent")
    module = _import_mcp_server()
    module.SERVER_STATE.config = object()
    module.SERVER_STATE.sqlite_db = object()
    module.SERVER_STATE.searcher = None
    module.SERVER_STATE.initialize_core_only = lambda: None
    module.SERVER_STATE.initialize = lambda: pytest.fail("agent_policy_check should not initialize search runtime")

    result = _decode_response(
        asyncio.run(
            module.call_tool(
                "agent_policy_check",
                {"payload": {"classification": "P0", "body": "private material", "api_key": "sk-test"}},
            )
        )
    )
    rendered = json.dumps(result, ensure_ascii=False)

    assert result["status"] == "ok"
    assert result["payload"]["schema"] == "knowledge-hub.agent.context-packet.v1"
    assert result["payload"]["policy"]["classification"] == "P0"
    assert result["payload"]["safeToUse"] is False
    assert result["payload"]["requiredHumanReview"] is True
    assert result["verify"]["schemaValid"] is True
    assert result["verify"]["policyAllowed"] is False
    assert result["requestEcho"]["arguments"]["payload"]["omitted"] is True
    assert result["artifact"]["jsonContent"]["context"]["payload"]["omitted"] is True
    assert "private material" not in rendered
    assert "sk-test" not in rendered


def test_agent_ask_knowledge_wraps_contracts_and_local_policy(monkeypatch):
    monkeypatch.setattr(agent_handler, "_runtime_diagnostics", lambda *args, **kwargs: {"schema": "runtime", "status": "ok"})
    monkeypatch.setattr(agent_handler, "_graph_query_signal", lambda *args, **kwargs: {"recommended_mode": "baseline"})

    result = asyncio.run(
        agent_handler.handle_tool("agent_ask_knowledge", {"question": "What is RAG?"}, _ctx())
    )

    assert result["status"] == "ok"
    packet = result["payload"]
    assert packet["schema"] == "knowledge-hub.agent.context-packet.v1"
    assert packet["policy"]["allowExternal"] is False
    assert packet["policy"]["policyMode"] == "local-only"
    assert packet["evidencePacketContract"]["packet_id"] == "evidence:1"
    assert packet["answerContract"]["answer_id"] == "answer:1"
    assert packet["verificationVerdict"]["verdict"] == "pass"
    assert packet["safeToUse"] is True
    assert packet["requiredHumanReview"] is False
    assert packet["context"]["citations"][0]["label"] == "S1"


def test_agent_ask_knowledge_marks_failed_verdict_unsafe(monkeypatch):
    monkeypatch.setattr(agent_handler, "_runtime_diagnostics", lambda *args, **kwargs: {"schema": "runtime", "status": "ok"})
    monkeypatch.setattr(agent_handler, "_graph_query_signal", lambda *args, **kwargs: {})

    result = asyncio.run(
        agent_handler.handle_tool(
            "agent_ask_knowledge",
            {"question": "What is RAG?"},
            _ctx(searcher=_FakeSearcher(verdict="fail")),
        )
    )

    packet = result["payload"]
    assert packet["safeToUse"] is False
    assert packet["requiredHumanReview"] is True
    assert any("verification verdict" in warning for warning in packet["warnings"])


def test_agent_policy_check_blocks_p0_payload():
    result = asyncio.run(
        agent_handler.handle_tool(
            "agent_policy_check",
            {"payload": {"classification": "P0", "body": "private material"}},
            _ctx(),
        )
    )

    packet = result["payload"]
    assert result["status"] == "ok"
    assert packet["policy"]["classification"] == "P0"
    assert packet["safeToUse"] is False
    assert packet["requiredHumanReview"] is True
    assert packet["policy"]["externalSendAllowed"] is False


def test_agent_stage_memory_is_stage_only():
    result = asyncio.run(
        agent_handler.handle_tool(
            "agent_stage_memory",
            {"goal": "remember RAG summary", "payload": {"summary": "local only"}, "sourceId": "note:rag"},
            _ctx(),
        )
    )

    packet = result["payload"]
    stage = packet["context"]["stage"]
    assert result["status"] == "ok"
    assert stage["status"] == "proposal"
    assert stage["applySkipped"] is True
    assert stage["finalApplyAllowed"] is False
    assert packet["policy"]["stageAllowed"] is True
    assert packet["policy"]["writebackAllowed"] is False
    assert packet["requiredHumanReview"] is True
