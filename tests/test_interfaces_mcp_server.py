from __future__ import annotations

import asyncio
import importlib
import json

import pytest


def _import_interfaces_server():
    try:
        return importlib.import_module("knowledge_hub.interfaces.mcp.server")
    except SystemExit:
        pytest.skip("mcp dependency is unavailable in test environment")


def test_interfaces_mcp_server_imports_with_standalone_app():
    module = _import_interfaces_server()
    assert module.app is not None
    assert callable(module.main)


def test_interfaces_mcp_server_rejects_non_object_arguments():
    module = _import_interfaces_server()
    response = asyncio.run(module.call_tool_impl(module.SERVER_STATE, "search_knowledge", []))
    assert response
    payload = json.loads(response[0].text)
    assert payload["status"] == "failed"
    assert "arguments" in payload["payload"]["error"]
