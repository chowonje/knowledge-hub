from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

from knowledge_hub.application.evidence_registry import register_context_pack, register_packet
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase


def _import_mcp_server():
    try:
        return importlib.import_module("knowledge_hub.interfaces.mcp.server")
    except SystemExit:
        pytest.skip("mcp dependency is unavailable in test environment")


def test_mcp_lists_khub_resources_and_templates():
    module = _import_mcp_server()

    resources = [str(item.uri) for item in module.list_khub_resources()]
    templates = [str(item.uriTemplate) for item in module.list_khub_resource_templates()]

    assert "khub://corpus/status" in resources
    assert "khub://corpus/contract" in resources
    assert "khub://source/{source_id}" in templates
    assert "khub://chunk/{chunk_id}" in templates
    assert "khub://packet/{packet_id}" in templates
    assert "khub://context/{context_pack_id}" in templates


def test_mcp_read_corpus_status_resource_returns_json(tmp_path):
    module = _import_mcp_server()
    state = SimpleNamespace(
        config=SimpleNamespace(
            vector_db_path=str(tmp_path / "chroma_db"),
            sqlite_path=str(tmp_path / "knowledge.db"),
            collection_name="knowledge_hub_test",
        ),
        sqlite_db=SimpleNamespace(get_stats=lambda: {"notes": 1}),
    )

    contents = module.read_khub_resource(state, "khub://corpus/status")
    payload = json.loads(contents[0].content)

    assert payload["schema"] == "knowledge-hub.inspect.result.v1"
    assert payload["contract"]["primaryWorkflow"] == "Codex context + evidence-backed local knowledge"
    assert payload["stores"]["sqlite"]["stats"]["notes"] == 1
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_mcp_read_packet_resource_is_stable_not_found():
    module = _import_mcp_server()
    state = SimpleNamespace(config=object(), sqlite_db=None)

    contents = module.read_khub_resource(state, "khub://packet/epkt_missing")
    payload = json.loads(contents[0].content)

    assert payload["schema"] == "knowledge-hub.mcp.resource.result.v1"
    assert payload["status"] == "not_found"
    assert payload["resourceKind"] == "packet"
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_mcp_read_packet_resource_resolves_registry_record(tmp_path):
    module = _import_mcp_server()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    register_packet(
        db,
        {
            "schema": "knowledge-hub.evidence-packet.v1",
            "packet_id": "epkt_1",
            "spans": [{"span_id": "span_1", "source_id": "src_1", "content_hash": "sha256:aaa"}],
        },
    )
    state = SimpleNamespace(config=object(), sqlite_db=db)

    contents = module.read_khub_resource(state, "khub://packet/epkt_1")
    payload = json.loads(contents[0].content)

    assert payload["schema"] == "knowledge-hub.mcp.resource.result.v1"
    assert payload["status"] == "ok"
    assert payload["registryLookup"]["payload"]["packet_id"] == "epkt_1"
    assert payload["registryLookup"]["lineage"]["evidenceSpanIds"] == ["span_1"]
    assert validate_payload(payload["registryLookup"], payload["registryLookup"]["schema"], strict=True).ok


def test_mcp_read_context_resource_resolves_registry_record(tmp_path):
    module = _import_mcp_server()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    register_context_pack(
        db,
        {
            "schema": "knowledge-hub.context-pack.result.v1",
            "contextPackId": "ctx_1",
            "status": "ok",
            "target": "task",
            "query": "mcp context",
            "sources": [{"source_id": "src_1", "source_content_hash": "sha256:aaa"}],
            "counts": {"total": 1},
            "warnings": [],
        },
    )
    state = SimpleNamespace(config=object(), sqlite_db=db)

    contents = module.read_khub_resource(state, "khub://context/ctx_1")
    payload = json.loads(contents[0].content)

    assert payload["status"] == "ok"
    assert payload["registryLookup"]["payload"]["contextPackId"] == "ctx_1"
    assert validate_payload(payload["registryLookup"], payload["registryLookup"]["schema"], strict=True).ok
