from __future__ import annotations

import pytest

from knowledge_hub.ai.ask_v2_pipeline_result import build_card_v2_pipeline_result
from knowledge_hub.ai.retrieval_pipeline import RetrievalPlan
from knowledge_hub.core.models import SearchResult


def _plan(**overrides):
    payload = {
        "query": "rag retrieval",
        "source_scope": "paper",
        "query_intent": "comparison",
        "paper_family": "general",
        "retrieval_mode": "hybrid",
        "memory_mode": "paper-card-v2",
        "candidate_budgets": {"paper": 6},
        "query_plan": {"family": "general"},
        "query_frame": {"family": "general", "query_intent": "comparison"},
        "temporal_signals": {"enabled": False, "mode": "none", "targetYear": ""},
        "temporal_route_applied": False,
        "memory_prior_weight": 0.0,
        "fallback_window": 3,
        "token_budget": 720,
        "memory_compression_target": 0.4,
        "chunk_expansion_threshold": 0.7,
    }
    payload.update(overrides)
    return RetrievalPlan(**payload)


def test_build_card_v2_pipeline_result_preserves_scoped_no_result_contract():
    diagnostics = {
        "runtimeExecution": {"used": "ask_v2", "fallbackReason": "scoped_vault_no_result"},
        "answerProvenance": {"mode": "scoped_no_result"},
        "fallback": {"used": False, "reason": "scoped_vault_no_result"},
    }

    result = build_card_v2_pipeline_result(
        results=[],
        plan=_plan(source_scope="vault", memory_mode="vault-card-v2"),
        source_kind="vault",
        route_mode="scoped",
        route_intent="lookup",
        selected_cards=[],
        selected_anchor_count=0,
        memory_reason="scoped_vault_no_result",
        context_expansion_mode="none",
        v2_diagnostics=diagnostics,
    )

    assert result.results == []
    assert result.candidate_sources == []
    assert result.context_expansion == {
        "eligible": False,
        "used": False,
        "mode": "none",
        "reason": "lookup",
        "queryIntent": "lookup",
        "enrichmentRoute": "scoped",
        "ontologyEligible": False,
        "clusterEligible": False,
        "ontologyUsed": False,
        "clusterUsed": False,
    }
    assert result.memory_route == {
        "contractRole": "ask_retrieval_memory_prefilter",
        "mode": "vault-card-v2",
        "applied": True,
        "reason": "scoped_vault_no_result",
    }
    assert result.memory_prefilter["contractRole"] == "retrieval_memory_prefilter"
    assert result.paper_memory_prefilter["contractRole"] == "paper_source_memory_prefilter"
    assert result.paper_memory_prefilter["requestedMode"] == "off"
    assert result.paper_memory_prefilter["effectiveMode"] == "off"
    assert result.paper_memory_prefilter["applied"] is False
    assert result.paper_memory_prefilter["reason"] == "disabled"
    assert result.paper_memory_prefilter["matchedPaperIds"] == []
    assert result.paper_memory_prefilter["matchedMemoryIds"] == []
    assert result.rerank_signals["selectedCardCount"] == 0
    assert result.rerank_signals["selectedAnchorCount"] == 0
    assert result.v2_diagnostics["answerProvenance"]["mode"] == "scoped_no_result"


def test_build_card_v2_pipeline_result_preserves_card_candidate_and_ontology_contract():
    search_result = SearchResult(
        document="anchor text",
        metadata={"title": "RAG Paper", "source_type": "paper"},
        distance=0.1,
        score=0.9,
        document_id="paper:rag",
    )
    plan = _plan(
        temporal_signals={"enabled": True, "mode": "latest", "targetYear": "2026"},
        temporal_route_applied=True,
    )
    selected_cards = [
        {
            "paper_id": "paper:rag",
            "source_memory_id": "pmem:rag",
            "card_id": "card:rag",
            "title": "RAG Paper",
            "selection_score": "0.84",
            "quality_flag": "strong",
        }
    ]

    result = build_card_v2_pipeline_result(
        results=[search_result],
        plan=plan,
        source_kind="paper",
        route_mode="ontology-first",
        route_intent="relation",
        selected_cards=selected_cards,
        selected_anchor_count=2,
        memory_reason="ontology-first",
        context_expansion_mode="card",
        paper_memory_mode="prefilter",
    )

    assert result.results == [search_result]
    assert result.candidate_sources == [
        {
            "id": "paper:rag",
            "cardId": "card:rag",
            "title": "RAG Paper",
            "selectionScore": 0.84,
            "qualityFlag": "strong",
            "sourceKind": "paper",
        }
    ]
    assert result.memory_route["contractRole"] == "ask_retrieval_memory_prefilter"
    assert result.memory_route["mode"] == "paper-card-v2"
    assert result.memory_prefilter["temporalSignals"] == {"enabled": True, "mode": "latest", "targetYear": "2026"}
    assert result.paper_memory_prefilter["requestedMode"] == "prefilter"
    assert result.paper_memory_prefilter["effectiveMode"] == "compat"
    assert result.paper_memory_prefilter["modeAliasApplied"] is True
    assert result.paper_memory_prefilter["applied"] is True
    assert result.paper_memory_prefilter["matchedPaperIds"] == ["paper:rag"]
    assert result.paper_memory_prefilter["matchedMemoryIds"] == ["pmem:rag"]
    assert result.context_expansion["mode"] == "ontology"
    assert result.context_expansion["ontologyEligible"] is True
    assert result.context_expansion["ontologyUsed"] is True
    assert result.rerank_signals == {
        "strategy": "paper-card-v2",
        "selectedCardCount": 1,
        "selectedAnchorCount": 2,
    }


def test_build_card_v2_pipeline_result_disables_paper_prefilter_for_non_paper_sources():
    result = build_card_v2_pipeline_result(
        results=[],
        plan=_plan(source_scope="web", memory_mode="web-card-v2"),
        source_kind="web",
        route_mode="card-first",
        route_intent="reference",
        selected_cards=[
            {
                "document_id": "web:rag",
                "source_memory_id": "wmem:rag",
                "card_id": "web-card:rag",
                "title": "RAG Web",
                "selection_score": 0.8,
            }
        ],
        selected_anchor_count=1,
        memory_reason="card-first",
        paper_memory_mode="prefilter",
    )

    assert result.candidate_sources[0]["sourceKind"] == "web"
    assert result.memory_prefilter["mode"] == "web-card-v2"
    assert result.memory_prefilter["applied"] is True
    assert result.paper_memory_prefilter["requestedMode"] == "prefilter"
    assert result.paper_memory_prefilter["effectiveMode"] == "compat"
    assert result.paper_memory_prefilter["applied"] is False
    assert result.paper_memory_prefilter["reason"] == "source_not_paper"
    assert result.paper_memory_prefilter["matchedPaperIds"] == []
    assert result.paper_memory_prefilter["matchedMemoryIds"] == []


@pytest.mark.parametrize("source_kind", ["vault", "web", "project"])
def test_build_card_v2_pipeline_result_defaults_non_paper_prefilter_to_public_off(source_kind):
    result = build_card_v2_pipeline_result(
        results=[],
        plan=_plan(source_scope=source_kind, memory_mode=f"{source_kind}-card-v2"),
        source_kind=source_kind,
        route_mode="card-first",
        route_intent="lookup",
        selected_cards=[],
        selected_anchor_count=0,
        memory_reason="card-first",
    )

    assert result.memory_prefilter["mode"] == f"{source_kind}-card-v2"
    assert result.memory_prefilter["applied"] is True
    assert result.paper_memory_prefilter["contractRole"] == "paper_source_memory_prefilter"
    assert result.paper_memory_prefilter["requestedMode"] == "off"
    assert result.paper_memory_prefilter["effectiveMode"] == "off"
    assert result.paper_memory_prefilter["applied"] is False
    assert result.paper_memory_prefilter["reason"] == "disabled"
    assert result.paper_memory_prefilter["matchedPaperIds"] == []
    assert result.paper_memory_prefilter["matchedMemoryIds"] == []
