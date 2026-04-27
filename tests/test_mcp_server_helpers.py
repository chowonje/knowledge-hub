from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest


def _import_mcp_server():
    try:
        return importlib.import_module("knowledge_hub.interfaces.mcp.server")
    except SystemExit:
        pytest.skip("mcp dependency is unavailable in test environment")


def test_coerce_foundry_payload_supports_json_and_line_fallback():
    module = _import_mcp_server()

    payload = module._coerce_foundry_payload('{"status":"ok","runId":"r1"}')
    assert payload["status"] == "ok"
    assert payload["runId"] == "r1"

    mixed = "log line\n{\"status\":\"completed\",\"runId\":\"r2\"}\n"
    payload2 = module._coerce_foundry_payload(mixed)
    assert payload2["status"] == "completed"
    assert payload2["runId"] == "r2"


def test_normalize_foundry_payload_blocks_p0_artifact():
    module = _import_mcp_server()
    raw = {
        "status": "completed",
        "runId": "run_001",
        "goal": "test",
        "transitions": [{"stage": "PLAN", "status": "PLAN", "message": "init"}],
        "verify": {"allowed": True, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
        "artifact": {"jsonContent": "contact at secret@example.com", "classification": "P0"},
    }
    normalized = module._normalize_foundry_payload(raw, goal="test", max_rounds=2, dry_run=False)
    assert normalized["status"] == "blocked"
    assert normalized["verify"]["allowed"] is False
    assert normalized["artifact"]["classification"] == "P0"
    assert normalized["writeback"]["ok"] is False


def test_normalize_foundry_payload_blocks_delegated_external_policy_gaps():
    module = _import_mcp_server()
    missing = module._normalize_foundry_payload(
        {
            "status": "completed",
            "source": "foundry-core/cli-agent",
            "runId": "run_missing_policy",
            "goal": "test",
            "verify": {"allowed": True, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
            "transitions": [{"stage": "PLAN", "status": "PLAN", "message": "init"}],
        },
        goal="test",
        max_rounds=2,
        dry_run=False,
    )
    external = module._normalize_foundry_payload(
        {
            "status": "completed",
            "source": "foundry-core/cli-agent",
            "runId": "run_external_policy",
            "goal": "test",
            "externalPolicy": {
                "allowExternal": True,
                "externalSendAllowed": True,
                "policyMode": "external-allowed",
                "decisionSource": "test",
            },
            "verify": {"allowed": True, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
            "transitions": [{"stage": "PLAN", "status": "PLAN", "message": "init"}],
        },
        goal="test",
        max_rounds=2,
        dry_run=False,
    )

    assert missing["status"] == "blocked"
    assert missing["verify"]["allowed"] is False
    assert any("missing local-only externalPolicy" in item for item in missing["verify"]["schemaErrors"])
    assert external["status"] == "blocked"
    assert external["verify"]["allowed"] is False
    assert any("external calls are disabled" in item for item in external["verify"]["schemaErrors"])


def test_build_fallback_agent_payload_includes_verify_and_trace():
    module = _import_mcp_server()
    payload_text = module._build_fallback_agent_payload(
        goal="compare rag methods",
        max_rounds=2,
        dry_run=True,
        plan=["search_knowledge", "ask_knowledge"],
        artifact={"answer": "ok"},
        verify_ok=True,
        errors=[],
        trace=[{"stage": "PLAN", "step": "search_knowledge"}],
        role="planner",
        orchestrator_mode="adaptive",
        playbook={"schema": "knowledge-hub.agent.playbook.v1"},
    )
    payload = json.loads(payload_text)
    assert payload["schema"] == "knowledge-hub.foundry.agent.run.result.v1"
    assert payload["dryRun"] is True
    assert isinstance(payload["transitions"], list)
    assert payload["verify"]["policyAllowed"] in {True, False}
    assert payload["playbook"]["schema"] == "knowledge-hub.foundry.agent.run.playbook.v1"
    assert module.validate_payload(payload, payload["schema"], strict=True).ok


def test_build_verify_block_collects_schema_errors(monkeypatch):
    module = _import_mcp_server()

    monkeypatch.setattr(
        module,
        "validate_payload",
        lambda payload, schema_id: SimpleNamespace(ok=False, errors=["schema mismatch"]),  # noqa: ARG005
    )
    verify = module._build_verify_block({"schema": "x"}, module.MCP_TOOL_STATUS_OK, "crawl_web_ingest")
    assert verify["schemaValid"] is False
    assert verify["allowed"] is False
    assert "schema mismatch" in verify["schemaErrors"]


def test_build_mcp_tool_response_blocks_policy_payload():
    module = _import_mcp_server()
    response = module._build_mcp_tool_response(
        tool="ask_knowledge",
        status=module.MCP_TOOL_STATUS_OK,
        payload={"answer": "send to private@example.com"},
        request_echo={"question": "q"},
    )
    assert response["status"] == module.MCP_TOOL_STATUS_BLOCKED
    assert response["verify"]["allowed"] is False
    assert response["artifact"]["classification"] == "P0"


def test_infer_classification_handles_scalar_payloads():
    module = _import_mcp_server()

    assert module.infer_classification("plain text") == "P2"
    assert module.infer_classification("contact secret@example.com") == "P0"


def test_build_mcp_tool_response_accepts_scalar_artifact():
    module = _import_mcp_server()

    response = module._build_mcp_tool_response(
        tool="ask_knowledge",
        status=module.MCP_TOOL_STATUS_OK,
        payload={"answer": "ok"},
        artifact="plain text artifact",
    )

    assert response["status"] == module.MCP_TOOL_STATUS_OK
    assert response["artifact"]["jsonContent"] == "plain text artifact"
    assert response["artifact"]["classification"] == "P2"


def test_build_mcp_tool_response_blocks_policy_artifact_even_when_payload_is_safe():
    module = _import_mcp_server()

    response = module._build_mcp_tool_response(
        tool="ask_knowledge",
        status=module.MCP_TOOL_STATUS_OK,
        payload={"answer": "ok"},
        artifact={"jsonContent": "contact secret@example.com"},
    )

    assert response["status"] == module.MCP_TOOL_STATUS_BLOCKED
    assert response["verify"]["policyAllowed"] is False
    assert response["verify"]["allowed"] is False
    assert response["artifact"]["classification"] == "P0"
    assert response["artifact"]["jsonContent"] == "[REDACTED_BY_POLICY]"


def test_normalize_foundry_payload_preserves_trace_entries():
    module = _import_mcp_server()
    payload = module._normalize_foundry_payload(
        {
            "status": "completed",
            "runId": "r1",
            "goal": "goal",
            "role": "planner",
            "orchestratorMode": "adaptive",
            "plan": ["a", "b"],
            "verify": {"allowed": True, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
            "transitions": [{"at": "now", "stage": "PLAN", "status": "PLAN", "message": "step"}],
            "artifact": {"jsonContent": {"answer": "ok"}},
        },
        goal="goal",
        max_rounds=2,
        dry_run=False,
    )
    assert payload["runId"] == "r1"
    assert isinstance(payload["transitions"], list)
    assert payload["transitions"][0]["stage"] == "PLAN"
    assert module.validate_payload(payload, payload["schema"], strict=True).ok


def test_normalize_foundry_payload_fills_required_contract_fields():
    module = _import_mcp_server()
    payload = module._normalize_foundry_payload(
        {
            "status": "completed",
            "runId": "r2",
            "goal": "goal",
            "verify": {"allowed": True, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
            "transitions": [{"stage": "PLAN", "status": "PLAN", "message": "planned", "tool": None}],
            "artifact": {"jsonContent": {"answer": "ok"}, "classification": "P2"},
        },
        goal="goal",
        max_rounds=2,
        dry_run=False,
    )
    assert payload["role"] == "planner"
    assert payload["orchestratorMode"] == "adaptive"
    assert payload["playbook"]["schema"] == "knowledge-hub.foundry.agent.run.playbook.v1"
    assert "tool" not in payload
    assert "tool" not in payload["transitions"][0]
    assert module.validate_payload(payload, payload["schema"], strict=True).ok


def test_normalize_foundry_payload_adds_gateway_metadata_for_dry_run():
    module = _import_mcp_server()
    payload = module._normalize_foundry_payload(
        {
            "status": "blocked",
            "runId": "r2-dry",
            "goal": "goal",
            "dryRun": True,
            "verify": {"allowed": False, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
            "transitions": [{"stage": "PLAN", "status": "PLAN", "message": "planned"}],
            "artifact": None,
        },
        goal="goal",
        max_rounds=2,
        dry_run=True,
    )
    assert payload["gateway"]["surface"] == "agent_run"
    assert payload["gateway"]["mode"] == "dry_run"
    assert module.validate_payload(payload, payload["schema"], strict=True).ok


def test_build_verify_block_uses_payload_schema_when_tool_mapping_is_missing():
    module = _import_mcp_server()
    verify = module._build_verify_block(
        {
            "schema": "knowledge-hub.foundry.agent.run.result.v1",
            "runId": "r3",
            "goal": "missing fields",
        },
        module.MCP_TOOL_STATUS_OK,
        "run_agentic_query",
    )
    assert verify["schemaValid"] is False
    assert verify["allowed"] is False
    assert any("required property" in item for item in verify["schemaErrors"])


def test_mcp_server_exports_explicit_compatibility_surface():
    module = _import_mcp_server()
    exported = set(getattr(module, "__all__", []))
    assert {"app", "call_tool", "list_tools", "main", "initialize", "_run_async_tool"} <= exported
    assert {"_build_verify_block", "_build_mcp_tool_response", "_coerce_foundry_payload"} <= exported


def test_legacy_mcp_server_shim_re_exports_canonical_surface():
    legacy = importlib.import_module("knowledge_hub.mcp_server")
    canonical = _import_mcp_server()
    assert legacy.app is canonical.app
    assert legacy.call_tool is canonical.call_tool
    assert legacy.list_tools is canonical.list_tools
    assert legacy.main is canonical.main
