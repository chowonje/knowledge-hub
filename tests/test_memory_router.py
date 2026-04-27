from __future__ import annotations

import json

from click.testing import CliRunner

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.interfaces.cli.commands.memory_cmd import memory_group
from knowledge_hub.knowledge.memory_router import build_memory_route


class _StubConfig:
    def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
        _ = args
        return default


class _StubKhub:
    def __init__(self):
        self.config = _StubConfig()


def test_build_memory_route_prefers_claims_for_comparison():
    payload = build_memory_route(query="RAG와 Mem0 논문 결과를 비교해줘", source_type="paper")
    assert payload["recommendedDecisionOrder"] == "memory_form_first"
    assert payload["contractRole"] == "read_only_memory_form_route_explainer"
    assert payload["retrievalExecuted"] is False
    assert payload["canonicalWritePerformed"] is False
    assert payload["queryIntent"] == "comparison"
    assert payload["route"]["primaryForm"] == "claim_evidence"
    assert payload["route"]["verifierForm"] == "chunk"
    assert payload["route"]["preferredForms"][0]["name"] == "claim_evidence"
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_build_memory_route_prefers_document_memory_for_definition():
    payload = build_memory_route(query="RAG가 뭐야?", source_type="all")
    assert payload["queryIntent"] in {"definition", "topic_lookup"}
    assert payload["route"]["primaryForm"] == "document_memory"
    names = [item["name"] for item in payload["route"]["preferredForms"]]
    assert "ontology_cluster" in names
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_memory_route_cli_json_payload():
    runner = CliRunner()
    result = runner.invoke(
        memory_group,
        ["route", "--query", "논문 결과를 비교해줘", "--source-type", "paper", "--explain-budget", "--json"],
        obj={"khub": _StubKhub()},
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.memory-route.result.v1"
    assert payload["contractRole"] == "read_only_memory_form_route_explainer"
    assert payload["retrievalExecuted"] is False
    assert payload["canonicalWritePerformed"] is False
    assert payload["recommendedDecisionOrder"] == "memory_form_first"
    assert payload["route"]["primaryForm"] == "claim_evidence"
    assert any(item["name"] == "retrieval_path_first" for item in payload["strategyComparison"])
    assert payload["budgetHints"]["tokenBudget"] > 0
    assert payload["budgetHints"]["chunkExpansionThreshold"] > 0.0
    core_payload = dict(payload)
    core_payload.pop("budgetHints", None)
    assert validate_payload(core_payload, core_payload["schema"], strict=True).ok


def test_memory_route_schema_requires_read_only_contract_markers():
    payload = build_memory_route(query="논문 결과를 비교해줘", source_type="paper")

    for field in ("contractRole", "retrievalExecuted", "canonicalWritePerformed"):
        invalid = dict(payload)
        invalid.pop(field)
        assert validate_payload(invalid, invalid["schema"], strict=True).ok is False
