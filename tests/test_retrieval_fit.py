from __future__ import annotations

import types

from knowledge_hub.ai.paper_query_plan import classify_paper_family
from knowledge_hub.ai.retrieval_fit import (
    _query_ontology_tokens,
    apply_query_fit_reranking,
    build_related_note_suggestions,
    classify_query_intent,
)
from knowledge_hub.core.models import SearchResult


def _result(
    *,
    title: str,
    source_type: str,
    document: str,
    score: float,
    distance: float = 0.2,
    file_path: str = "",
    cluster_id: str = "",
    links: list[str] | None = None,
    canonical_name: str = "",
    aliases: list[str] | None = None,
    entity_id: str = "",
) -> SearchResult:
    metadata = {
        "title": title,
        "source_type": source_type,
    }
    if file_path:
        metadata["file_path"] = file_path
    if cluster_id:
        metadata["cluster_id"] = cluster_id
    if links:
        metadata["links"] = links
    if canonical_name:
        metadata["canonical_name"] = canonical_name
    if aliases:
        metadata["aliases"] = aliases
    if entity_id:
        metadata["entity_id"] = entity_id
    return SearchResult(
        document=document,
        metadata=metadata,
        distance=distance,
        score=score,
        semantic_score=max(0.0, score - 0.05),
        lexical_score=max(0.0, score - 0.1),
        retrieval_mode="hybrid",
        lexical_extras={},
    )


def test_classify_query_intent_distinguishes_topic_howto_and_evaluation():
    assert classify_query_intent("Transformer architecture") == "topic_lookup"
    assert classify_query_intent("How to set up notebook workbench") == "howto"
    assert classify_query_intent("safety evaluation benchmark") == "evaluation"
    assert classify_query_intent("RAG paper 2401.12345") == "paper_lookup"
    assert classify_query_intent("트랜스포머를 대체할 차세대 아키텍처 논문들을 찾아서 정리해줘") == "paper_topic"
    assert classify_query_intent("관련 papers about state space models 정리해줘") == "paper_topic"
    assert classify_query_intent("논문 요약해줘") == "paper_lookup"
    assert classify_query_intent("what is RAG") == "definition"


def test_classify_paper_family_uses_four_default_routes_for_paper_queries():
    assert classify_paper_family("CNN을 쉽게 설명해줘", source_type="paper") == "concept_explainer"
    assert classify_paper_family("AlexNet 요약해줘", source_type="paper") == "paper_lookup"
    assert classify_paper_family("CNN vs ViT 비교해줘", source_type="paper") == "paper_compare"
    assert classify_paper_family("RAG 관련 논문 찾아줘", source_type="paper") == "paper_discover"


def test_build_related_note_suggestions_uses_existing_metadata():
    results = [
        _result(
            title="Transformer Architecture",
            source_type="vault",
            document="transformer architecture notes",
            score=0.82,
            file_path="Notes/Transformer Architecture.md",
            cluster_id="c1",
            links=["Attention Mechanism"],
        ),
        _result(
            title="Attention Mechanism",
            source_type="vault",
            document="attention mechanism notes",
            score=0.79,
            file_path="Notes/Attention Mechanism.md",
            cluster_id="c1",
        ),
        _result(
            title="Unrelated Survey",
            source_type="concept",
            document="unrelated survey note",
            score=0.4,
            file_path="Concepts/Unrelated Survey.md",
            cluster_id="c9",
        ),
    ]

    suggestions = build_related_note_suggestions(results, limit=2)

    assert "Notes/Transformer Architecture.md" in suggestions
    transformer_suggestions = suggestions["Notes/Transformer Architecture.md"]
    assert transformer_suggestions[0]["title"] == "Attention Mechanism"
    assert "same_cluster" in transformer_suggestions[0]["reasons"]
    assert "linked_note" in transformer_suggestions[0]["reasons"]


def test_apply_query_fit_reranking_adds_related_notes_and_prefers_concrete_vault():
    results = [
        _result(
            title="Transformer",
            source_type="concept",
            document="transformer overview note",
            score=0.41,
            canonical_name="Transformer",
            aliases=["transformer"],
            entity_id="concept_transformer",
        ),
        _result(
            title="Transformer Architecture",
            source_type="vault",
            document="transformer architecture details and implementation notes",
            score=0.39,
            file_path="Notes/Transformer Architecture.md",
            cluster_id="c1",
            links=["Transformer"],
        ),
    ]

    reranked = apply_query_fit_reranking(results, query="Transformer architecture")

    assert reranked[0].metadata["title"] == "Transformer Architecture"
    assert reranked[0].lexical_extras["query_intent"] == "topic_lookup"
    assert reranked[0].lexical_extras["related_notes"] == ["Transformer"]
    assert reranked[0].lexical_extras["related_note_suggestions"][0]["title"] == "Transformer"


def test_apply_query_fit_reranking_uses_rescue_query_forms_for_paper_title_match():
    results = [
        _result(
            title="Brain Tumor Identification and Classification of MRI Images Using Deep Learning Techniques",
            source_type="paper",
            document="application paper using deep learning for MRI classification",
            score=0.53,
        ),
        _result(
            title="ImageNet Classification with Deep Convolutional Neural Networks",
            source_type="paper",
            document="AlexNet introduced a deep convolutional neural network for ImageNet classification.",
            score=0.5,
        ),
    ]

    reranked = apply_query_fit_reranking(
        results,
        query="CNN을 쉽게 설명해줘",
        query_forms=["CNN", "convolutional neural network"],
    )

    assert reranked[0].metadata["title"] == "ImageNet Classification with Deep Convolutional Neural Networks"
    assert reranked[0].lexical_extras["ranking_signals"]["near_title_overlap_boost"] > 0.0


def test_apply_query_fit_reranking_penalizes_generic_vault_for_technical_definition():
    results = [
        _result(
            title="용어",
            source_type="vault",
            document="generic glossary note",
            score=0.54,
            file_path="Projects/AI/용어.md",
        ),
        _result(
            title="SoK: Agentic Retrieval-Augmented Generation (RAG)",
            source_type="paper",
            document="rag taxonomy architectures evaluation and research directions",
            score=0.51,
        ),
    ]

    reranked = apply_query_fit_reranking(results, query="RAG의 핵심 장단점은 무엇인가?")

    assert reranked[0].metadata["source_type"] == "paper"
    assert reranked[0].metadata["title"].startswith("SoK: Agentic Retrieval-Augmented Generation")
    assert reranked[1].lexical_extras["ranking_signals"]["generic_vault_penalty"] > 0.0


def test_apply_query_fit_reranking_penalizes_refusal_excerpt_and_paper_source_mismatch():
    results = [
        _result(
            title="Interpretability Note",
            source_type="vault",
            document="claim evidence graph overview",
            score=0.58,
            file_path="Notes/Interpretability Note.md",
        ),
        _result(
            title="Paper2Agent",
            source_type="paper",
            document="I'm unable to access or read external documents. Upload the PDF for a precise summary.",
            score=0.61,
        ),
        _result(
            title="Agent Memory Evaluation",
            source_type="paper",
            document="paper memory card stores compact paper-level memory for retrieval and lookup",
            score=0.57,
        ),
    ]

    reranked = apply_query_fit_reranking(results, query="paper memory card의 목적을 설명해줘")

    assert reranked[0].metadata["title"] == "Agent Memory Evaluation"
    refusal_item = next(item for item in reranked if item.metadata["title"] == "Paper2Agent")
    refusal_signals = refusal_item.lexical_extras["ranking_signals"]
    assert refusal_signals["refusal_excerpt_penalty"] > 0.0


def test_apply_query_fit_reranking_penalizes_vault_hub_noise_for_explainer_queries():
    results = [
        _result(
            title="Obsidian 전체 마인드맵 및 정리 아틀라스",
            source_type="vault",
            document="이 노트는 전체 Vault를 한 번에 정리할 때 사용하는 큰 그림 지도입니다.",
            score=0.62,
            file_path="Atlas/Obsidian 전체 마인드맵 및 정리 아틀라스.md",
        ),
        _result(
            title="memory-first retrieval 설명",
            source_type="vault",
            document="memory-first retrieval은 먼저 memory unit을 좁혀 retrieval noise를 줄이는 전략이다.",
            score=0.58,
            file_path="Projects/AI/memory-first retrieval 설명.md",
        ),
    ]

    reranked = apply_query_fit_reranking(results, query="memory-first retrieval의 목적을 한 문장으로 설명해줘")

    assert reranked[0].metadata["title"] == "memory-first retrieval 설명"
    hub_item = next(item for item in reranked if item.metadata["title"] == "Obsidian 전체 마인드맵 및 정리 아틀라스")
    assert hub_item.lexical_extras["ranking_signals"]["vault_hub_penalty"] > 0.0


def test_apply_query_fit_reranking_still_penalizes_hub_noise_for_assertion_queries():
    results = [
        _result(
            title="Obsidian 전체 마인드맵 및 정리 아틀라스",
            source_type="vault",
            document="이 노트는 전체 Vault를 정리하는 큰 그림 지도입니다.",
            score=0.63,
            file_path="Atlas/Obsidian 전체 마인드맵 및 정리 아틀라스.md",
        ),
        _result(
            title="Reranker 구현 메모",
            source_type="vault",
            document="현재 구조에서 cross-encoder reranker를 기본값으로 쓰지 않는다고 명시한다.",
            score=0.58,
            file_path="Projects/AI/Reranker 구현 메모.md",
        ),
    ]

    reranked = apply_query_fit_reranking(results, query="지금 구조가 cross-encoder reranker를 기본으로 사용한다고 단정할 수 있나?")

    assert reranked[0].metadata["title"] == "Reranker 구현 메모"
    hub_item = next(item for item in reranked if item.metadata["title"] == "Obsidian 전체 마인드맵 및 정리 아틀라스")
    assert hub_item.lexical_extras["ranking_signals"]["vault_hub_penalty"] >= 0.08


def test_apply_query_fit_reranking_preserves_ranking_signals_per_chunk_same_parent():
    """Same vault parent can yield multiple chunks; signals must not use the last chunk's contributions."""
    query = "Specific Topic implementation in RetrievalPipelineService"
    intro = SearchResult(
        document="unrelated intro fluff",
        metadata={
            "title": "Shared",
            "source_type": "vault",
            "file_path": "Notes/same.md",
            "section_path": "Introduction",
        },
        distance=0.2,
        score=0.52,
        semantic_score=0.47,
        lexical_score=0.42,
        retrieval_mode="hybrid",
        lexical_extras={},
    )
    impl = SearchResult(
        document="specific topic implementation RetrievalPipelineService orchestration",
        metadata={
            "title": "Shared",
            "source_type": "vault",
            "file_path": "Notes/same.md",
            "section_path": "Implementation",
        },
        distance=0.2,
        score=0.48,
        semantic_score=0.43,
        lexical_score=0.38,
        retrieval_mode="hybrid",
        lexical_extras={},
    )
    # Implementation chunk last in the first pass — previously overwrote parent-keyed contributions.
    reranked = apply_query_fit_reranking([intro, impl], query=query)
    intro_out = next(r for r in reranked if r.metadata.get("section_path") == "Introduction")
    impl_out = next(r for r in reranked if r.metadata.get("section_path") == "Implementation")
    assert float(intro_out.lexical_extras["ranking_signals"].get("direct_answer_bonus") or 0.0) == 0.0
    assert float(impl_out.lexical_extras["ranking_signals"].get("direct_answer_bonus") or 0.0) > 0.0


def test_apply_query_fit_reranking_adds_direct_answer_bonus_for_implementation_notes():
    results = [
        _result(
            title="memory-first retrieval 개요",
            source_type="vault",
            document="memory-first retrieval 관련 일반 개요 노트",
            score=0.58,
            file_path="Projects/AI/memory-first retrieval 개요.md",
        ),
        SearchResult(
            document="RetrievalPipelineService implementation details and orchestration flow",
            metadata={
                "title": "RetrievalPipelineService implementation",
                "source_type": "vault",
                "file_path": "Projects/AI/RetrievalPipelineService implementation.md",
                "unit_type": "method",
                "section_path": "Implementation > RetrievalPipelineService",
            },
            distance=0.2,
            score=0.55,
            semantic_score=0.5,
            lexical_score=0.45,
            retrieval_mode="hybrid",
            lexical_extras={},
        ),
    ]

    reranked = apply_query_fit_reranking(results, query="RetrievalPipelineService는 어떻게 구현되는가?")

    assert reranked[0].metadata["title"] == "RetrievalPipelineService implementation"
    assert reranked[0].lexical_extras["ranking_signals"]["direct_answer_bonus"] > 0.0


def test_query_ontology_tokens_skips_fuzzy_resolver_for_long_title_queries(monkeypatch):
    class _ResolverShouldNotRun:
        def __init__(self, db):  # noqa: D401, ANN001
            raise AssertionError("resolver should be skipped for long title-heavy queries")

    fake_module = types.SimpleNamespace(EntityResolver=_ResolverShouldNotRun)
    monkeypatch.setitem(__import__("sys").modules, "knowledge_hub.learning.resolver", fake_module)

    query = (
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks와 "
        "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering을 비교해줘"
    )
    tokens = _query_ontology_tokens(query, sqlite_db=object())

    assert "retrieval" in tokens
    assert "generation" in tokens


def test_query_ontology_tokens_keeps_resolver_for_short_concept_queries(monkeypatch):
    class _Resolver:
        def __init__(self, db):  # noqa: ANN001
            self.db = db

        def resolve(self, token):  # noqa: ANN001
            if token == "cnn":
                return types.SimpleNamespace(
                    display_name="Convolutional Neural Network",
                    aliases=["convnet"],
                    canonical_id="cnn",
                )
            return None

    fake_module = types.SimpleNamespace(EntityResolver=_Resolver)
    monkeypatch.setitem(__import__("sys").modules, "knowledge_hub.learning.resolver", fake_module)

    tokens = _query_ontology_tokens("CNN", sqlite_db=object())

    assert "cnn" in tokens
    assert "convolutional" in tokens
    assert "convnet" in tokens
