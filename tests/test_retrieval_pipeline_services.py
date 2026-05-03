from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from knowledge_hub.application.query_frame import build_query_frame
from knowledge_hub.ai.paper_query_plan import build_rule_based_query_frame, build_rule_query_plan
from knowledge_hub.ai.evidence_assembly import EvidenceAssemblyService
from knowledge_hub.ai.rag import RAGSearcher
from knowledge_hub.ai.retrieval_pipeline import RetrievalPipelineService
from knowledge_hub.core.models import SearchResult
from tests.test_rag_search import DummyEmbedder, DummyFeatureSQLite, DummyVectorDB, FakeLLM


class _GraphAnalysis:
    def to_dict(self):
        return {
            "is_graph_heavy": True,
            "candidate_hints": [
                {
                    "entity_id": "concept_rag",
                    "canonical_name": "RAG",
                    "aliases": ["Retrieval Augmented Generation"],
                    "score": 0.95,
                }
            ],
            "diagnostics": {
                "candidate_reduction_eligible": True,
                "query_kind": "entity",
            },
        }


class _ConfigStub:
    def __init__(self, values=None):
        self._values = values or {}

    def get_nested(self, *path, default=None):  # noqa: ANN002, ANN003
        cursor = self._values
        for token in path:
            if not isinstance(cursor, dict) or token not in cursor:
                return default
            cursor = cursor[token]
        return cursor


class _PrefixPaperSQLite(DummyFeatureSQLite):
    def search_papers(self, query, limit=20):
        _ = limit
        if str(query or "").strip().casefold() == "dino":
            return [
                {
                    "arxiv_id": "dinov3-local",
                    "title": "DINOv3",
                },
                {
                    "arxiv_id": "grounding-dino",
                    "title": "Grounding DINO",
                },
            ]
        return []


class _LocalTitleLookupSQLite(DummyFeatureSQLite):
    def search_paper_cards_v2(self, query, limit=5):
        _ = (query, limit)
        return []

    def search_papers(self, query, limit=20):
        _ = limit
        if str(query or "").strip() == "Auto-Encoding Variational Bayes":
            return [
                {
                    "arxiv_id": "aevb-local",
                    "title": "Auto-Encoding Variational Bayes",
                }
            ]
        return []


class _CompareCardSQLite(DummyFeatureSQLite):
    def __init__(self, snapshots, *, paper_cards=None, paper_memory_cards=None, papers=None):
        super().__init__(snapshots)
        self.paper_cards = dict(paper_cards or {})
        self.paper_memory_cards = dict(paper_memory_cards or {})
        self.papers = dict(papers or {})

    def get_paper_card_v2(self, paper_id):
        return self.paper_cards.get(str(paper_id or "").strip())

    def get_paper_memory_card(self, paper_id):
        return self.paper_memory_cards.get(str(paper_id or "").strip())

    def get_paper(self, paper_id):
        return self.papers.get(str(paper_id or "").strip())
        return []


class _PaperCardSearchSQLite(_CompareCardSQLite):
    def __init__(self, snapshots, *, paper_cards=None, paper_memory_cards=None, papers=None, search_rows=None):
        super().__init__(
            snapshots,
            paper_cards=paper_cards,
            paper_memory_cards=paper_memory_cards,
            papers=papers,
        )
        self.search_rows = dict(search_rows or {})

    def search_paper_cards_v2(self, query, limit=5):
        token = str(query or "").strip()
        return list(self.search_rows.get(token, []))[:limit]


def _build_topology_snapshot(vault_path, *, file_path: str, cluster_id: str = "c1") -> None:
    root = vault_path / ".obsidian" / "khub" / "topology"
    root.mkdir(parents=True, exist_ok=True)
    snapshot_path = root / "latest.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema": "knowledge-hub.vault.topology.snapshot.v1",
                "nodes": [
                    {
                        "id": file_path,
                        "path": file_path,
                        "title": "RAG Note",
                        "clusterId": cluster_id,
                    }
                ],
                "clusters": [
                    {
                        "id": cluster_id,
                        "label": "rag cluster",
                        "size": 1,
                        "representativeNoteId": file_path,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_retrieval_pipeline_uses_graph_candidate_boost_from_rag_patchpoint():
    records = [
        {
            "id": "generic",
            "document": "general implementation note",
            "distance": 0.12,
            "metadata": {
                "title": "Implementation Notes",
                "source_type": "vault",
                "file_path": "Projects/AI/Implementation Notes.md",
            },
        },
        {
            "id": "rag-note",
            "document": "rag transformer attention note",
            "distance": 0.16,
            "metadata": {
                "title": "RAG Transformer Note",
                "source_type": "vault",
                "file_path": "Projects/AI/RAG Transformer Note.md",
                "related_concepts": ["RAG"],
            },
        },
    ]
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB(records),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    with patch("knowledge_hub.ai.rag.analyze_graph_query", lambda query, repository: _GraphAnalysis()):
        result = RetrievalPipelineService(searcher).execute(
            query="RAG transformer attention",
            top_k=2,
            retrieval_mode="semantic",
            use_ontology_expansion=False,
        )

    ranking = dict(result.results[0].lexical_extras or {}).get("ranking_signals") or {}
    assert result.results[0].metadata["title"] == "RAG Transformer Note"
    assert ranking["graph_candidate_boost"] > 0.0
    assert ranking["graph_candidate_reduction_applied"] is True
    assert "RAG" in ranking["graph_candidate_matches"]


def test_retrieval_pipeline_graph_candidate_boost_can_match_document_text():
    class _TransformerGraphAnalysis:
        def to_dict(self):
            return {
                "is_graph_heavy": True,
                "candidate_hints": [
                    {
                        "entity_id": "transformer",
                        "canonical_name": "Transformer",
                        "aliases": ["Transformers"],
                        "score": 0.95,
                    }
                ],
                "diagnostics": {
                    "candidate_reduction_eligible": True,
                    "query_kind": "entity",
                },
            }

    records = [
        {
            "id": "generic",
            "document": "general sequence modeling note",
            "distance": 0.12,
            "metadata": {
                "title": "Sequence Modeling Notes",
                "source_type": "paper",
                "arxiv_id": "generic",
            },
        },
        {
            "id": "attn-is-all-you-need",
            "document": "The Transformer uses only attention mechanisms, removing recurrence and convolutions.",
            "distance": 0.16,
            "metadata": {
                "title": "Attention Is All You Need",
                "source_type": "paper",
                "arxiv_id": "1706.03762",
            },
        },
    ]
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB(records),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    with patch("knowledge_hub.ai.rag.analyze_graph_query", lambda query, repository: _TransformerGraphAnalysis()):
        result = RetrievalPipelineService(searcher).execute(
            query="Transformer의 핵심 아이디어를 설명해줘",
            top_k=2,
            source_type="paper",
            retrieval_mode="semantic",
            use_ontology_expansion=False,
        )

    ranking = dict(result.results[0].lexical_extras or {}).get("ranking_signals") or {}
    assert result.results[0].metadata["title"] == "Attention Is All You Need"
    assert ranking["graph_candidate_boost"] > 0.0
    assert ranking["graph_candidate_reduction_applied"] is True
    assert "Transformer" in ranking["graph_candidate_matches"]


def test_retrieval_pipeline_keeps_cross_encoder_reranker_disabled_by_default():
    records = [
        {
            "id": "vault-a",
            "document": "compact paper memory description",
            "distance": 0.05,
            "metadata": {"title": "Paper Memory A", "source_type": "vault", "file_path": "AI/Paper Memory A.md"},
        },
        {
            "id": "vault-b",
            "document": "broader retrieval notes",
            "distance": 0.06,
            "metadata": {"title": "Paper Memory B", "source_type": "vault", "file_path": "AI/Paper Memory B.md"},
        },
    ]
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB(records),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
        config=_ConfigStub(),
    )

    result = RetrievalPipelineService(searcher).execute(
        query="paper memory card의 목적",
        top_k=2,
        retrieval_mode="semantic",
        use_ontology_expansion=False,
    )

    diagnostics = result.diagnostics()["rerankSignals"]
    assert diagnostics["rerankerApplied"] is False
    assert diagnostics["rerankerFallbackUsed"] is False
    assert diagnostics["rerankerReason"] == "disabled"


def test_retrieval_pipeline_uses_local_prefix_title_rescue_for_versioned_paper_queries():
    records = [
        {
            "id": "blo-inst",
            "document": "Grounding DINO alignment for segmentation and detection.",
            "distance": 0.08,
            "metadata": {
                "title": "BLO-Inst YOLO and SAM Alignment",
                "source_type": "paper",
                "arxiv_id": "blo-inst",
            },
        },
        {
            "id": "dinov3",
            "document": "DINOv3 is a self-supervised vision foundation model.",
            "distance": 0.18,
            "metadata": {
                "title": "DINOv3",
                "source_type": "paper",
                "arxiv_id": "dinov3-local",
            },
        },
    ]
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB(records),
        llm=FakeLLM(),
        sqlite_db=_PrefixPaperSQLite({}),
    )

    result = RetrievalPipelineService(searcher).execute(
        query="DINO에 대해서 설명해줘",
        top_k=2,
        source_type="paper",
        retrieval_mode="hybrid",
        use_ontology_expansion=False,
        query_plan=build_rule_query_plan(
            "DINO에 대해서 설명해줘",
            source_type="paper",
            sqlite_db=searcher.sqlite_db,
        ).to_dict(),
        query_frame=build_rule_based_query_frame(
            "DINO에 대해서 설명해줘",
            source_type="paper",
            sqlite_db=searcher.sqlite_db,
        ).to_dict(),
    )

    diagnostics = result.diagnostics()["rerankSignals"]
    assert "DINOv3" in diagnostics["queryRescueForms"]
    assert "DINOv3" in diagnostics["lexicalQueryForms"]
    assert result.results[0].metadata["title"] == "DINOv3"


def test_retrieval_pipeline_promotes_local_prefix_title_rescue_for_cross_source_definition_queries():
    records = [
        {
            "id": "blo-inst",
            "document": "Grounding DINO alignment for segmentation and detection.",
            "distance": 0.08,
            "metadata": {
                "title": "BLO-Inst YOLO and SAM Alignment",
                "source_type": "paper",
                "arxiv_id": "blo-inst",
            },
        },
        {
            "id": "dinov3",
            "document": "DINOv3 is a self-supervised vision foundation model.",
            "distance": 0.18,
            "metadata": {
                "title": "DINOv3",
                "source_type": "paper",
                "arxiv_id": "dinov3-local",
            },
        },
    ]
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB(records),
        llm=FakeLLM(),
        sqlite_db=_PrefixPaperSQLite({}),
    )

    result = RetrievalPipelineService(searcher).execute(
        query="DINO에 대해서 설명해줘",
        top_k=2,
        retrieval_mode="hybrid",
        use_ontology_expansion=False,
    )

    diagnostics = result.diagnostics()["rerankSignals"]
    assert "DINOv3" in diagnostics["queryRescueForms"]
    assert diagnostics["lexicalQueryForms"][0] == "DINOv3"
    assert result.results[0].metadata["title"] == "DINOv3"


def test_retrieval_pipeline_applies_cross_encoder_reranker_when_enabled():
    records = [
        {
            "id": "vault-a",
            "document": "generic retrieval background",
            "distance": 0.05,
            "metadata": {"title": "Generic Retrieval", "source_type": "vault", "file_path": "AI/Generic Retrieval.md"},
        },
        {
            "id": "vault-b",
            "document": "paper memory card stores compact paper-level memory for retrieval",
            "distance": 0.06,
            "metadata": {"title": "Paper Memory Card", "source_type": "vault", "file_path": "AI/Paper Memory Card.md"},
        },
    ]
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB(records),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
        config=_ConfigStub(
            {
                "labs": {
                    "retrieval": {
                        "reranker": {
                            "enabled": True,
                            "model": "test-reranker",
                            "candidate_window": 2,
                            "timeout_ms": 1500,
                            "fallback_on_error": True,
                        }
                    }
                }
            }
        ),
    )

    class _FakeReranker:
        def rerank(self, *, query, results, config):  # noqa: ANN001
            _ = (query, config)
            reranked = [results[1], results[0]]
            return type(
                "_Execution",
                (),
                {
                    "results": reranked,
                    "diagnostics": {
                        "rerankerApplied": True,
                        "rerankerModel": "test-reranker",
                        "rerankerWindow": 2,
                        "rerankerLatencyMs": 12,
                        "rerankerFallbackUsed": False,
                        "rerankerReason": "applied",
                    },
                },
            )()

    with patch.object(RetrievalPipelineService, "_get_reranker", return_value=_FakeReranker()):
        result = RetrievalPipelineService(searcher).execute(
            query="paper memory card의 목적",
            top_k=2,
            retrieval_mode="semantic",
            use_ontology_expansion=False,
        )

    diagnostics = result.diagnostics()["rerankSignals"]
    assert diagnostics["rerankerApplied"] is True
    assert diagnostics["rerankerModel"] == "test-reranker"
    assert diagnostics["rerankerFallbackUsed"] is False
    assert result.results[0].metadata["title"] == "Paper Memory Card"


def test_retrieval_pipeline_records_cross_encoder_fallback_when_timeout_occurs():
    records = [
        {
            "id": "vault-a",
            "document": "generic retrieval background",
            "distance": 0.05,
            "metadata": {"title": "Generic Retrieval", "source_type": "vault", "file_path": "AI/Generic Retrieval.md"},
        },
        {
            "id": "vault-b",
            "document": "paper memory card stores compact paper-level memory for retrieval",
            "distance": 0.06,
            "metadata": {"title": "Paper Memory Card", "source_type": "vault", "file_path": "AI/Paper Memory Card.md"},
        },
    ]
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB(records),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
        config=_ConfigStub(
            {
                "labs": {
                    "retrieval": {
                        "reranker": {
                            "enabled": True,
                            "model": "test-reranker",
                            "candidate_window": 2,
                            "timeout_ms": 10,
                            "fallback_on_error": True,
                        }
                    }
                }
            }
        ),
    )

    class _FallbackReranker:
        def rerank(self, *, query, results, config):  # noqa: ANN001
            _ = (query, config)
            return type(
                "_Execution",
                (),
                {
                    "results": list(results),
                    "diagnostics": {
                        "rerankerApplied": False,
                        "rerankerModel": "test-reranker",
                        "rerankerWindow": 2,
                        "rerankerLatencyMs": 25,
                        "rerankerFallbackUsed": True,
                        "rerankerReason": "timeout",
                    },
                },
            )()

    with patch.object(RetrievalPipelineService, "_get_reranker", return_value=_FallbackReranker()):
        result = RetrievalPipelineService(searcher).execute(
            query="paper memory card의 목적",
            top_k=2,
            retrieval_mode="semantic",
            use_ontology_expansion=False,
        )

    diagnostics = result.diagnostics()["rerankSignals"]
    assert diagnostics["rerankerApplied"] is False
    assert diagnostics["rerankerFallbackUsed"] is True
    assert diagnostics["rerankerReason"] == "timeout"


def test_evidence_assembly_builds_citation_target_from_result_metadata():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )
    result = SearchResult(
        document="paper abstract",
        metadata={"title": "Paper A", "source_type": "paper", "arxiv_id": "2501.00001"},
        distance=0.1,
        score=0.95,
        semantic_score=0.94,
        lexical_score=0.93,
        retrieval_mode="hybrid",
        lexical_extras={},
        document_id="paper-a",
    )

    searcher._collect_claim_context = lambda filtered: ([], [], [], [])
    searcher._resolve_parent_context = lambda result, doc_cache: {
        "parent_id": "",
        "parent_label": "",
        "chunk_span": "",
        "text": result.document,
    }
    searcher._build_answer_context = lambda filtered, parent_ctx_by_result: "\n".join(
        item.document for item in filtered
    )
    searcher._summarize_answer_signals = lambda evidence, contradicting_beliefs: {
        "preferred_sources": len(evidence)
    }
    searcher._answer_evidence_item = lambda result, parent_ctx_by_result: {
        "title": result.metadata["title"],
        "source_type": result.metadata["source_type"],
        "score": result.score,
        "semantic_score": result.semantic_score,
        "lexical_score": result.lexical_score,
    }

    packet = EvidenceAssemblyService.from_searcher(searcher).assemble(
        query="이 논문의 핵심 기여는?",
        source_type="paper",
        results=[result],
        paper_memory_prefilter={
            "requestedMode": "prefilter",
            "applied": True,
            "fallbackUsed": False,
            "matchedPaperIds": ["2501.00001"],
            "matchedMemoryIds": ["memory-a"],
            "memoryRelationsUsed": [],
            "temporalSignals": {},
            "reason": "matched_cards",
        },
        metadata_filter=None,
    )

    assert packet.paper_answer_scope["applied"] is True
    assert packet.evidence_budget["selectedCount"] == 1
    assert packet.citations[0]["target"] == "2501.00001"
    assert packet.citations[0]["kind"] == "arxiv"


def test_retrieval_pipeline_marks_temporal_route_and_fallback_window():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    plan = RetrievalPipelineService(searcher).build_plan(
        query="latest RAG benchmark update",
        top_k=3,
        source_type="paper",
        retrieval_mode="hybrid",
        memory_route_mode="prefilter",
        use_ontology_expansion=False,
    )

    assert plan.temporal_route_applied is True
    assert plan.temporal_signals["mode"] == "latest"
    assert plan.memory_prior_weight >= 0.1
    assert plan.fallback_window >= 3


def test_retrieval_pipeline_default_paper_lookup_is_memory_first_without_enrichment():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    plan = RetrievalPipelineService(searcher).build_plan(
        query="paper memory card의 목적을 설명해줘",
        top_k=3,
        source_type="paper",
        retrieval_mode="hybrid",
        memory_route_mode="prefilter",
        use_ontology_expansion=True,
    )

    assert plan.query_intent == "paper_lookup"
    assert plan.enrichment_route == "memory_heavy"
    assert plan.ontology_assist_eligible is False
    assert plan.cluster_assist_eligible is False
    assert plan.ontology_expansion_enabled is False


def test_retrieval_plan_applies_hard_scope_for_single_resolved_paper_lookup():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    plan = RetrievalPipelineService(searcher).build_plan(
        query="AlexNet 논문 요약해줘",
        top_k=3,
        source_type="paper",
        retrieval_mode="hybrid",
        memory_route_mode="off",
        use_ontology_expansion=False,
        query_frame={
            "source_type": "paper",
            "family": "paper_lookup",
            "query_intent": "paper_lookup",
            "canonical_entity_ids": ["deep_convolutional_neural_networks"],
            "resolved_source_ids": ["alexnet-2012"],
        },
    )

    assert plan.resolved_source_scope_applied is True
    assert plan.metadata_filter_applied["source_type"] == "paper"
    assert plan.metadata_filter_applied["arxiv_id"] == "alexnet-2012"
    assert plan.prefilter_reason == "resolved_source_id"
    assert list(plan.canonical_entities_applied) == ["deep_convolutional_neural_networks"]


def test_retrieval_plan_keeps_concept_prefilter_soft_when_only_canonical_entities_exist():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    plan = RetrievalPipelineService(searcher).build_plan(
        query="CNN을 쉽게 설명해줘",
        top_k=3,
        source_type="paper",
        retrieval_mode="hybrid",
        memory_route_mode="off",
        use_ontology_expansion=False,
        query_frame={
            "source_type": "paper",
            "family": "concept_explainer",
            "query_intent": "definition",
            "canonical_entity_ids": ["deep_convolutional_neural_networks"],
            "resolved_source_ids": ["alexnet-2012", "2010.11929"],
        },
    )

    assert plan.resolved_source_scope_applied is False
    assert plan.prefilter_reason == "representative_candidate_narrowing"
    assert plan.metadata_filter_applied["source_type"] == "paper"


def test_retrieval_pipeline_topic_query_does_not_auto_enable_ontology_or_cluster():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    plan = RetrievalPipelineService(searcher).build_plan(
        query="트랜스포머를 대체할 차세대 아키텍처 논문들을 찾아서 정리해줘",
        top_k=4,
        source_type="paper",
        retrieval_mode="hybrid",
        memory_route_mode="prefilter",
        use_ontology_expansion=True,
    )

    assert plan.query_intent == "paper_topic"
    assert plan.enrichment_route == "memory_heavy"
    assert plan.ontology_assist_eligible is False
    assert plan.cluster_assist_eligible is False


def test_retrieval_pipeline_definition_query_can_enable_ontology_assist():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    plan = RetrievalPipelineService(searcher).build_plan(
        query="RAG와 GraphRAG의 차이와 개념을 정의해줘",
        top_k=4,
        source_type=None,
        retrieval_mode="hybrid",
        memory_route_mode="off",
        use_ontology_expansion=True,
    )

    assert plan.enrichment_route == "ontology_assist"
    assert plan.ontology_assist_eligible is True
    assert plan.cluster_assist_eligible is False
    assert plan.ontology_expansion_enabled is True


def test_retrieval_pipeline_grouping_query_can_enable_cluster_assist(tmp_path):
    records = [
        {
            "id": "rag-note",
            "document": "RAG note for grouping",
            "distance": 0.08,
            "metadata": {"title": "RAG Note", "source_type": "vault", "file_path": "AI/rag.md"},
        }
    ]
    _build_topology_snapshot(tmp_path, file_path="AI/rag.md")
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB(records),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
        config=SimpleNamespace(vault_path=str(tmp_path)),
    )

    plan = RetrievalPipelineService(searcher).build_plan(
        query="RAG 관련 논문들을 그룹으로 추천해줘",
        top_k=3,
        source_type=None,
        retrieval_mode="semantic",
        memory_route_mode="off",
        use_ontology_expansion=True,
    )
    result = RetrievalPipelineService(searcher).execute(
        query="RAG 관련 논문들을 그룹으로 추천해줘",
        top_k=3,
        retrieval_mode="semantic",
        use_ontology_expansion=True,
        memory_route_mode="off",
    )

    assert plan.enrichment_route == "cluster_assist"
    assert result.diagnostics()["contextExpansion"]["clusterEligible"] is True
    assert result.diagnostics()["contextExpansion"]["clusterUsed"] is True
    assert result.related_clusters[0]["cluster_id"] == "c1"


def test_retrieval_pipeline_core_query_exposes_enrichment_diagnostics_without_usage():
    records = [
        {
            "id": "paper-doc",
            "document": "paper memory card is a compact paper-level memory representation used for retrieval",
            "distance": 0.06,
            "metadata": {
                "title": "Paper Memory Card",
                "source_type": "paper",
                "arxiv_id": "2501.00001",
            },
        },
    ]
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB(records),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    result = RetrievalPipelineService(searcher).execute(
        query="paper memory card의 목적을 설명해줘",
        top_k=2,
        source_type="paper",
        retrieval_mode="semantic",
        use_ontology_expansion=True,
        memory_route_mode="off",
    )

    diagnostics = result.diagnostics()["contextExpansion"]
    assert diagnostics["eligible"] is False
    assert diagnostics["used"] is False
    assert diagnostics["mode"] == "none"
    assert diagnostics["reason"] == "memory_first_query_family"


def test_evidence_assembly_tracks_temporal_grounding_and_memory_provenance():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )
    result = SearchResult(
        document="updated paper abstract",
        metadata={
            "title": "Paper B",
            "source_type": "paper",
            "arxiv_id": "2603.00002",
            "published_at": "2026-03-01T00:00:00+00:00",
            "section_path": "Paper B > Results",
        },
        distance=0.1,
        score=0.95,
        semantic_score=0.94,
        lexical_score=0.93,
        retrieval_mode="hybrid",
        lexical_extras={
            "source_trust_score": 0.91,
            "memory_provenance": {"requestedMode": "prefilter", "reason": "matched_cards"},
        },
        document_id="paper-b",
    )

    searcher._collect_claim_context = lambda filtered: ([], [], [], [])
    searcher._resolve_parent_context = lambda result, doc_cache: {
        "parent_id": "",
        "parent_label": "",
        "chunk_span": "",
        "text": result.document,
    }
    searcher._build_answer_context = lambda filtered, parent_ctx_by_result: "\n".join(
        item.document for item in filtered
    )

    packet = EvidenceAssemblyService.from_searcher(searcher).assemble(
        query="latest result for paper b",
        source_type="paper",
        results=[result],
        paper_memory_prefilter={
            "requestedMode": "prefilter",
            "applied": True,
            "fallbackUsed": False,
            "matchedPaperIds": ["2603.00002"],
            "matchedMemoryIds": ["memory-b"],
            "memoryRelationsUsed": [],
            "temporalSignals": {"enabled": True, "mode": "latest"},
            "reason": "matched_cards",
        },
        metadata_filter=None,
    )

    assert packet.evidence[0]["published_at"] == "2026-03-01T00:00:00+00:00"
    assert packet.evidence[0]["memory_provenance"]["requestedMode"] == "prefilter"
    assert packet.evidence_packet["freshness"]["temporalGroundedCount"] == 1
    assert packet.evidence_packet["freshness"]["memoryProvenanceCount"] == 1
    assert packet.evidence_packet["answerable"] is True


def test_retrieval_pipeline_enforces_explicit_source_scope_when_matches_exist():
    records = [
        {
            "id": "vault-paper-note",
            "document": "paper memory card stores compact paper level memory",
            "distance": 0.05,
            "metadata": {
                "title": "Paper Memory Note",
                "source_type": "vault",
                "file_path": "AI/Paper Memory Note.md",
                "arxiv_id": "2501.00001",
            },
        },
        {
            "id": "paper-doc",
            "document": "paper memory card is a compact paper-level memory representation used for retrieval",
            "distance": 0.06,
            "metadata": {
                "title": "Paper Memory Card",
                "source_type": "paper",
                "arxiv_id": "2501.00001",
            },
        },
    ]
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB(records),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    result = RetrievalPipelineService(searcher).execute(
        query="paper memory card의 목적을 설명해줘",
        top_k=2,
        source_type="paper",
        retrieval_mode="semantic",
        use_ontology_expansion=False,
        memory_route_mode="off",
    )

    assert result.results
    assert all((item.metadata or {}).get("source_type") == "paper" for item in result.results)
    assert result.diagnostics()["sourceScopeEnforced"] is True


def test_retrieval_plan_hard_scopes_single_resolved_paper_lookup():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )
    query_frame = build_query_frame(
        domain_key="ai_papers",
        source_type="paper",
        family="paper_lookup",
        query_intent="paper_lookup",
        answer_mode="paper_scoped_answer",
        entities=["AlexNet"],
        canonical_entity_ids=[],
        expanded_terms=["AlexNet"],
        resolved_source_ids=["2501.00001"],
        confidence=0.91,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="paper_lookup_policy",
        metadata_filter={},
    ).to_dict()

    plan = RetrievalPipelineService(searcher).build_plan(
        query="AlexNet 논문 요약해줘",
        top_k=3,
        source_type="paper",
        retrieval_mode="hybrid",
        memory_route_mode="off",
        use_ontology_expansion=False,
        metadata_filter=None,
        query_frame=query_frame,
    )

    assert plan.resolved_source_scope_applied is True
    assert plan.metadata_filter_applied["arxiv_id"] == "2501.00001"
    assert plan.prefilter_reason == "resolved_source_id"


def test_retrieval_pipeline_uses_resolved_source_ids_for_representative_narrowing(monkeypatch):
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )
    query_frame = build_query_frame(
        domain_key="ai_papers",
        source_type="paper",
        family="concept_explainer",
        query_intent="definition",
        answer_mode="representative_paper_explainer",
        entities=["CNN"],
        canonical_entity_ids=["deep_convolutional_neural_networks"],
        expanded_terms=["CNN", "Deep Convolutional Neural Networks"],
        resolved_source_ids=["alexnet-2012"],
        confidence=0.9,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="concept_explainer_policy",
        metadata_filter={},
    ).to_dict()
    lexical_filters: list[dict[str, object]] = []

    monkeypatch.setattr("knowledge_hub.ai.retrieval_pipeline.expand_query_with_ontology", lambda *_args, **_kwargs: ["CNN을 쉽게 설명해줘"])
    monkeypatch.setattr("knowledge_hub.ai.retrieval_pipeline.semantic_search", lambda *args, **kwargs: [])

    def _fake_lexical(_searcher, _query, top_k=5, filter_dict=None):
        _ = top_k
        lexical_filters.append(dict(filter_dict or {}))
        return []

    monkeypatch.setattr("knowledge_hub.ai.retrieval_pipeline.lexical_search", _fake_lexical)

    result = RetrievalPipelineService(searcher).execute(
        query="CNN을 쉽게 설명해줘",
        top_k=2,
        source_type="paper",
        retrieval_mode="keyword",
        use_ontology_expansion=False,
        memory_route_mode="off",
        query_frame=query_frame,
    )

    assert any(item.get("arxiv_id") == "alexnet-2012" for item in lexical_filters)
    assert result.diagnostics()["retrievalPlan"]["prefilterReason"] == "representative_candidate_narrowing"


def test_retrieval_pipeline_hard_scopes_local_exact_title_lookup_without_cards():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=_LocalTitleLookupSQLite({}),
    )

    plan = RetrievalPipelineService(searcher).build_plan(
        query="Auto-Encoding Variational Bayes 논문 설명해줘",
        top_k=3,
        source_type="paper",
        retrieval_mode="hybrid",
        memory_route_mode="off",
        use_ontology_expansion=False,
        query_frame=build_rule_based_query_frame(
            "Auto-Encoding Variational Bayes 논문 설명해줘",
            source_type="paper",
            sqlite_db=_LocalTitleLookupSQLite({}),
        ).to_dict(),
    )

    assert plan.query_frame["family"] == "paper_lookup"
    assert plan.query_frame["resolved_source_ids"] == ["aevb-local"]
    assert plan.resolved_source_scope_applied is True
    assert plan.metadata_filter_applied["arxiv_id"] == "aevb-local"
    assert plan.prefilter_reason == "resolved_source_id"


def test_retrieval_pipeline_marks_mixed_fallback_for_all_source_prefilter():
    records = [
        {
            "id": "web-guide",
            "document": "latest retrieval pipeline guide explains evidence-first answering",
            "distance": 0.08,
            "metadata": {
                "title": "Retrieval Pipeline Guide",
                "source_type": "web",
                "url": "https://example.com/guide",
            },
        }
    ]
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB(records),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    result = RetrievalPipelineService(searcher).execute(
        query="latest retrieval pipeline guide",
        top_k=2,
        source_type="all",
        retrieval_mode="semantic",
        use_ontology_expansion=False,
        memory_route_mode="prefilter",
    )

    diagnostics = result.diagnostics()
    assert diagnostics["mixedFallbackUsed"] is True
    assert diagnostics["memoryPrefilter"]["reason"] == "mixed_fallback_ranked"
    assert result.results


def test_evidence_assembly_marks_refusal_excerpt_as_non_substantive():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )
    result = SearchResult(
        document="I'm unable to access external documents. Upload the PDF to continue.",
        metadata={"title": "Paper2Agent", "source_type": "paper", "arxiv_id": "2501.00002"},
        distance=0.1,
        score=0.72,
        semantic_score=0.72,
        lexical_score=0.70,
        retrieval_mode="hybrid",
        lexical_extras={"source_trust_score": 0.91},
        document_id="paper2agent",
    )

    searcher._collect_claim_context = lambda filtered: ([], [], [], [])
    searcher._resolve_parent_context = lambda result, doc_cache: {
        "parent_id": "",
        "parent_label": "",
        "chunk_span": "",
        "text": result.document,
    }
    searcher._build_answer_context = lambda filtered, parent_ctx_by_result: "\n".join(
        item.document for item in filtered
    )

    packet = EvidenceAssemblyService.from_searcher(searcher).assemble(
        query="paper summary path에서 evidence packet이 왜 필요한가?",
        source_type="paper",
        results=[result],
        paper_memory_prefilter={
            "requestedMode": "prefilter",
            "applied": True,
            "fallbackUsed": False,
            "matchedPaperIds": ["2501.00002"],
            "matchedMemoryIds": ["memory-c"],
            "memoryRelationsUsed": [],
            "temporalSignals": {},
            "reason": "matched_cards",
        },
        metadata_filter=None,
    )

    assert packet.evidence_packet["answerable"] is False
    assert "non_substantive_evidence" in packet.evidence_packet["insufficientEvidenceReasons"]


def test_evidence_assembly_reselects_substantive_top1_after_hub_noise():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )
    noisy = SearchResult(
        document="# Obsidian 전체 마인드맵 및 정리 아틀라스\n이 노트는 전체 Vault를 정리하는 지도다.",
        metadata={
            "title": "Obsidian 전체 마인드맵 및 정리 아틀라스",
            "source_type": "vault",
            "file_path": "Atlas/Obsidian 전체 마인드맵 및 정리 아틀라스.md",
        },
        distance=0.1,
        score=0.81,
        semantic_score=0.80,
        lexical_score=0.79,
        retrieval_mode="hybrid",
        lexical_extras={"source_trust_score": 0.5},
        document_id="vault-hub",
    )
    substantive = SearchResult(
        document="memory-first retrieval은 먼저 memory unit을 좁혀 retrieval noise를 줄인다.",
        metadata={
            "title": "memory-first retrieval 설명",
            "source_type": "vault",
            "file_path": "Projects/AI/memory-first retrieval 설명.md",
        },
        distance=0.11,
        score=0.79,
        semantic_score=0.78,
        lexical_score=0.77,
        retrieval_mode="hybrid",
        lexical_extras={"source_trust_score": 0.55, "memory_provenance": {"reason": "vault_chunk_fallback"}},
        document_id="vault-substantive",
    )

    searcher._collect_claim_context = lambda filtered: ([], [], [], [])
    searcher._resolve_parent_context = lambda result, doc_cache: {
        "parent_id": "",
        "parent_label": "",
        "chunk_span": "",
        "text": result.document,
    }
    searcher._build_answer_context = lambda filtered, parent_ctx_by_result: "\n".join(item.document for item in filtered)

    packet = EvidenceAssemblyService.from_searcher(searcher).assemble(
        query="memory-first retrieval의 목적을 한 문장으로 설명해줘",
        source_type="vault",
        results=[noisy, substantive],
        paper_memory_prefilter={"requestedMode": "prefilter", "matchedPaperIds": [], "matchedMemoryIds": [], "memoryRelationsUsed": [], "temporalSignals": {}, "reason": "vault_chunk_fallback"},
        metadata_filter=None,
    )

    assert packet.evidence[0]["title"] == "memory-first retrieval 설명"
    assert packet.evidence_packet["validation"]["top1Reselected"] is True
    assert packet.evidence_packet["validation"]["top1RejectedReason"] == "vault_hub_noise"
    assert packet.evidence_packet["answerable"] is True


def test_evidence_assembly_keeps_web_temporal_observed_at_only_as_not_answerable():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )
    result = SearchResult(
        document="This article was observed recently and discusses retrieval guidance in broad terms.",
        metadata={
            "title": "Retrieval Guide",
            "source_type": "web",
            "url": "https://example.com/latest-guide",
            "observed_at": "2026-03-23T16:00:09+00:00",
        },
        distance=0.1,
        score=0.84,
        semantic_score=0.83,
        lexical_score=0.82,
        retrieval_mode="hybrid",
        lexical_extras={"source_trust_score": 0.9},
        document_id="web-latest-guide",
    )

    searcher._collect_claim_context = lambda filtered: ([], [], [], [])
    searcher._resolve_parent_context = lambda result, doc_cache: {
        "parent_id": "",
        "parent_label": "",
        "chunk_span": "",
        "text": result.document,
    }
    searcher._build_answer_context = lambda filtered, parent_ctx_by_result: "\n".join(item.document for item in filtered)

    packet = EvidenceAssemblyService.from_searcher(searcher).assemble(
        query="latest vector database retrieval best practice는 무엇인가?",
        source_type="web",
        results=[result],
        paper_memory_prefilter={"requestedMode": "prefilter", "matchedPaperIds": [], "matchedMemoryIds": [], "memoryRelationsUsed": [], "temporalSignals": {"enabled": True, "mode": "latest"}, "reason": "web_chunk_fallback_success"},
        metadata_filter=None,
    )

    assert packet.evidence_packet["answerable"] is False
    assert "missing_temporal_grounding" in packet.evidence_packet["insufficientEvidenceReasons"]
    assert packet.evidence_packet["answerableDecisionReason"] == "weak_web_temporal_grounding"


def test_evidence_assembly_uses_strict_threshold_for_abstention_like_queries():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )
    result = SearchResult(
        document="현재 구조에서는 cross-encoder reranker를 기본으로 쓴다는 근거가 부족하다.",
        metadata={
            "title": "Reranker 메모",
            "source_type": "vault",
            "file_path": "Projects/AI/Reranker 메모.md",
        },
        distance=0.1,
        score=0.82,
        semantic_score=0.81,
        lexical_score=0.8,
        retrieval_mode="hybrid",
        lexical_extras={"source_trust_score": 0.7},
        document_id="reranker-note",
    )

    searcher._collect_claim_context = lambda filtered: ([], [], [], [])
    searcher._resolve_parent_context = lambda result, doc_cache: {
        "parent_id": "",
        "parent_label": "",
        "chunk_span": "",
        "text": result.document,
    }
    searcher._build_answer_context = lambda filtered, parent_ctx_by_result: "\n".join(item.document for item in filtered)

    packet = EvidenceAssemblyService.from_searcher(searcher).assemble(
        query="지금 구조가 cross-encoder reranker를 기본으로 사용한다고 단정할 수 있나?",
        source_type="vault",
        results=[result],
        paper_memory_prefilter={"requestedMode": "prefilter", "matchedPaperIds": [], "matchedMemoryIds": [], "memoryRelationsUsed": [], "temporalSignals": {}, "reason": "vault_chunk_fallback"},
        metadata_filter=None,
    )

    assert packet.evidence_packet["answerable"] is False
    assert packet.evidence_packet["answerableDecisionReason"] == "strict_abstention_threshold_not_met"


def test_evidence_assembly_candidate_v6_can_accept_mixed_queries_with_direct_overlap():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )
    searcher._eval_answer_profile = "candidate-v6"
    result = SearchResult(
        document="이 노트는 시스템 전반에 대한 배경 메모이며 질문에 대한 직접 설명은 없다.",
        metadata={
            "title": "시스템 배경 메모",
            "source_type": "vault",
            "file_path": "Projects/AI/시스템 배경 메모.md",
            "unit_type": "background",
            "section_path": "Overview",
        },
        distance=0.1,
        score=0.82,
        semantic_score=0.81,
        lexical_score=0.8,
        retrieval_mode="hybrid",
        lexical_extras={"source_trust_score": 0.7},
        document_id="retrieval-overview",
    )

    searcher._collect_claim_context = lambda filtered: ([], [], [], [])
    searcher._resolve_parent_context = lambda result, doc_cache: {
        "parent_id": "",
        "parent_label": "",
        "chunk_span": "",
        "text": result.document,
    }
    searcher._build_answer_context = lambda filtered, parent_ctx_by_result: "\n".join(item.document for item in filtered)

    packet = EvidenceAssemblyService.from_searcher(searcher).assemble(
        query="KnowledgeOS에서 evidence packet이 중요한 이유를 전체 구조 관점에서 설명해줘",
        source_type=None,
        results=[result],
        paper_memory_prefilter={"requestedMode": "prefilter", "matchedPaperIds": [], "matchedMemoryIds": [], "memoryRelationsUsed": [], "temporalSignals": {}, "reason": "mixed_fallback_ranked"},
        metadata_filter=None,
    )

    assert packet.evidence_packet["answerable"] is True
    assert packet.evidence_packet["answerableDecisionReason"] == "substantive_evidence_found"


def test_evidence_assembly_candidate_v6_keeps_direct_temporal_evidence_answerable():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )
    searcher._eval_answer_profile = "candidate-v6"
    result = SearchResult(
        document="Updated benchmark v2 release notes compare the new metric emphasis and changed evaluation criteria.",
        metadata={
            "title": "Updated benchmark v2 release notes",
            "source_type": "paper",
            "published_at": "2026-02-01T00:00:00+00:00",
            "unit_type": "result",
            "section_path": "Results > Benchmark update",
        },
        distance=0.1,
        score=0.88,
        semantic_score=0.87,
        lexical_score=0.86,
        retrieval_mode="hybrid",
        lexical_extras={"source_trust_score": 0.9},
        document_id="benchmark-v2",
    )

    searcher._collect_claim_context = lambda filtered: ([], [], [], [])
    searcher._resolve_parent_context = lambda result, doc_cache: {
        "parent_id": "",
        "parent_label": "",
        "chunk_span": "",
        "text": result.document,
    }
    searcher._build_answer_context = lambda filtered, parent_ctx_by_result: "\n".join(item.document for item in filtered)

    packet = EvidenceAssemblyService.from_searcher(searcher).assemble(
        query="updated benchmark v2가 baseline v1보다 더 강조한 metric은 무엇인가?",
        source_type="paper",
        results=[result],
        paper_memory_prefilter={"requestedMode": "prefilter", "matchedPaperIds": [], "matchedMemoryIds": [], "memoryRelationsUsed": [], "temporalSignals": {"enabled": True, "mode": "latest"}, "reason": "memory_prefilter_chunk_fallback"},
        metadata_filter=None,
    )

    assert packet.evidence_packet["answerable"] is True
    assert packet.evidence_packet["answerableDecisionReason"] == "substantive_evidence_found"


def test_retrieval_pipeline_marks_cross_encoder_unavailable_when_enabled_but_runtime_missing():
    records = [
        {
            "id": "vault-a",
            "document": "generic retrieval background",
            "distance": 0.05,
            "metadata": {"title": "Generic Retrieval", "source_type": "vault", "file_path": "AI/Generic Retrieval.md"},
        },
        {
            "id": "vault-b",
            "document": "paper memory card stores compact paper-level memory for retrieval",
            "distance": 0.06,
            "metadata": {"title": "Paper Memory Card", "source_type": "vault", "file_path": "AI/Paper Memory Card.md"},
        },
    ]
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB(records),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
        config=_ConfigStub({"labs": {"retrieval": {"reranker": {"enabled": True, "model": "test-reranker", "candidate_window": 2, "timeout_ms": 1500, "fallback_on_error": True}}}}),
    )

    with patch.object(RetrievalPipelineService, "_get_reranker", return_value=None):
        result = RetrievalPipelineService(searcher).execute(
            query="paper memory card의 목적",
            top_k=2,
            retrieval_mode="semantic",
            use_ontology_expansion=False,
        )

    diagnostics = result.diagnostics()["rerankSignals"]
    assert diagnostics["rerankerApplied"] is False
    assert diagnostics["rerankerFallbackUsed"] is True
    assert diagnostics["rerankerReason"] == "unavailable"


def test_retrieval_pipeline_limits_concept_explainer_fanout_for_paper_legacy_path():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    semantic_calls: list[dict[str, object]] = []
    lexical_calls: list[dict[str, object]] = []

    def _semantic_stub(searcher_obj, *, query_embedding, top_k, filter_dict):  # noqa: ANN001
        _ = (searcher_obj, query_embedding)
        semantic_calls.append({"top_k": top_k, "filter_dict": dict(filter_dict or {})})
        return []

    def _lexical_stub(searcher_obj, query, *, top_k, filter_dict):  # noqa: ANN001
        _ = searcher_obj
        lexical_calls.append({"query": query, "top_k": top_k, "filter_dict": dict(filter_dict or {})})
        return []

    with patch("knowledge_hub.ai.retrieval_pipeline.semantic_search", _semantic_stub), patch(
        "knowledge_hub.ai.retrieval_pipeline.lexical_search",
        _lexical_stub,
    ):
        result = RetrievalPipelineService(searcher).execute(
            query="CNN을 쉽게 설명해줘",
            top_k=3,
            source_type="paper",
            retrieval_mode="hybrid",
            use_ontology_expansion=False,
            query_plan={
                "family": "concept_explainer",
                "expanded_terms": [
                    "CNN",
                    "Deep Convolutional Neural Networks",
                    "ImageNet Classification with Deep Convolutional Neural Networks",
                    "extra ignored term",
                ],
            },
            query_frame={
                "source_type": "paper",
                "family": "concept_explainer",
                "query_intent": "definition",
                "entities": ["CNN"],
                "canonical_entity_ids": ["deep_convolutional_neural_networks"],
                "resolved_source_ids": ["alexnet-2012", "ignored-paper"],
            },
        )

    signals = result.rerank_signals
    assert signals["expandedQueryCount"] <= 4
    assert len(signals["lexicalQueryForms"]) <= 2
    assert signals["representativeScopeIdsUsed"] == ["alexnet-2012"]
    assert signals["extraSourceFanoutSkipped"] is True
    assert "CNN을 쉽게 설명해줘" not in signals["lexicalQueryForms"]
    assert any(item == "CNN" for item in signals["lexicalQueryForms"])
    assert all(call["filter_dict"].get("source_type") == "paper" for call in semantic_calls if call["filter_dict"])
    assert all(call["filter_dict"].get("source_type") == "paper" for call in lexical_calls if call["filter_dict"])


def test_retrieval_pipeline_compare_lexical_forms_drop_task_words():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    lexical_queries: list[str] = []

    def _lexical_stub(searcher_obj, query, *, top_k, filter_dict):  # noqa: ANN001
        _ = (searcher_obj, top_k, filter_dict)
        lexical_queries.append(query)
        return []

    with patch("knowledge_hub.ai.retrieval_pipeline.lexical_search", _lexical_stub):
        result = RetrievalPipelineService(searcher).execute(
            query="CNN vs ViT 비교해줘",
            top_k=3,
            source_type="paper",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            query_plan={
                "family": "paper_compare",
                "expanded_terms": ["CNN", "ViT", "An Image is Worth 16x16 Words"],
            },
            query_frame={
                "source_type": "paper",
                "family": "paper_compare",
                "query_intent": "comparison",
                "entities": ["CNN", "ViT"],
                "resolved_source_ids": ["alexnet-2012", "2010.11929"],
            },
        )

    assert "CNN vs ViT 비교해줘" not in lexical_queries
    assert "비교" not in " ".join(result.rerank_signals["lexicalQueryForms"])
    assert "vs" not in " ".join(result.rerank_signals["lexicalQueryForms"]).lower()
    assert "An Image is Worth 16x16 Words" in result.rerank_signals["lexicalQueryForms"]
    assert "CNN" in result.rerank_signals["lexicalQueryForms"]
    assert "ViT" in result.rerank_signals["lexicalQueryForms"]


def test_retrieval_pipeline_compare_lexical_forms_prefer_title_rescue_over_short_acronyms():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    lexical_queries: list[str] = []

    def _lexical_stub(searcher_obj, query, *, top_k, filter_dict):  # noqa: ANN001
        _ = (searcher_obj, top_k, filter_dict)
        lexical_queries.append(query)
        return []

    with patch("knowledge_hub.ai.retrieval_pipeline.lexical_search", _lexical_stub):
        result = RetrievalPipelineService(searcher).execute(
            query="BERT와 GPT 계열의 차이를 논문 기준으로 비교해줘",
            top_k=3,
            source_type="paper",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            query_plan={
                "family": "paper_compare",
                "expanded_terms": [
                    "BERT",
                    "GPT",
                    "Language Models are Few-Shot Learners",
                    "BatGPT: A Bidirectional Autoregessive Talker from Generative Pre-trained Transformer",
                ],
            },
            query_frame={
                "source_type": "paper",
                "family": "paper_compare",
                "query_intent": "comparison",
                "entities": ["BERT", "GPT"],
                "resolved_source_ids": ["1810.04805", "2005.14165", "2307.00360"],
            },
        )

    assert result.rerank_signals["lexicalQueryForms"][0] == "Language Models are Few-Shot Learners"
    assert result.rerank_signals["lexicalQueryForms"].index("BERT") > 0
    assert result.rerank_signals["lexicalQueryForms"].index("GPT") > 0


def test_retrieval_pipeline_compare_lexical_forms_keep_compare_alias_rescues():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    with patch("knowledge_hub.ai.retrieval_pipeline.lexical_search", lambda *args, **kwargs: []):
        result = RetrievalPipelineService(searcher).execute(
            query="CNN이랑 ViT를 논문 관점에서 비교해서 핵심 차이와 각각 잘하는 상황을 설명해줘",
            top_k=3,
            source_type="paper",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            query_plan={
                "family": "paper_compare",
                "expanded_terms": [
                    "CNN",
                    "ViT",
                    "ImageNet Classification with Deep Convolutional Neural Networks",
                    "An Image is Worth 16x16 Words",
                    "Convolutional Neural Networks",
                    "Vision Transformer",
                ],
            },
            query_frame={
                "source_type": "paper",
                "family": "paper_compare",
                "query_intent": "comparison",
                "entities": ["CNN", "ViT"],
                "resolved_source_ids": ["alexnet-2012", "2010.11929"],
            },
        )

    forms = list(result.rerank_signals["lexicalQueryForms"])
    assert "CNN이랑 ViT를 논문 관점에서 비교해서 핵심 차이와 각각 잘하는 상황을 설명해줘" not in forms
    assert "Vision Transformer" in forms
    assert "Convolutional Neural Networks" in forms


def test_retrieval_pipeline_compare_promotes_resolved_paper_pair_into_results():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    def _result(doc_id: str, title: str, paper_id: str, *, score: float) -> SearchResult:
        return SearchResult(
            document=f"{title} abstract",
            metadata={"title": title, "source_type": "paper", "arxiv_id": paper_id},
            distance=max(0.0, 1.0 - score),
            score=score,
            semantic_score=score,
            lexical_score=score,
            retrieval_mode="keyword",
            lexical_extras={"query": title},
            document_id=doc_id,
        )

    def _lexical_stub(searcher_obj, query, *, top_k, filter_dict):  # noqa: ANN001
        _ = (searcher_obj, query, top_k)
        paper_id = str((filter_dict or {}).get("arxiv_id") or "").strip()
        if paper_id == "2005.11401":
            return [_result("rag", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "2005.11401", score=0.66)]
        if paper_id == "2007.01282":
            return [_result("fid", "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", "2007.01282", score=0.65)]
        generic_hits = [
            _result("fid", "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", "2007.01282", score=0.91),
            _result("realm", "REALM: Retrieval-Augmented Language Model Pre-Training", "2002.08909", score=0.89),
            _result("atlas", "Atlas: Few-shot Learning with Retrieval Augmented Language Models", "2208.03299", score=0.88),
        ]
        return generic_hits[:top_k]

    with patch("knowledge_hub.ai.retrieval_pipeline.lexical_search", _lexical_stub), patch(
        "knowledge_hub.ai.retrieval_pipeline.semantic_search",
        lambda *args, **kwargs: [],
    ):
        result = RetrievalPipelineService(searcher).execute(
            query="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks와 Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering를 비교해줘",
            top_k=3,
            source_type="paper",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            memory_route_mode="off",
            query_frame={
                "source_type": "paper",
                "family": "paper_compare",
                "query_intent": "comparison",
                "entities": ["Retrieval-Augmented Generation", "Leveraging Passage Retrieval"],
                "resolved_source_ids": ["2005.11401", "2007.01282", "2510.15682"],
            },
        )

    assert [item.metadata.get("arxiv_id") for item in result.results[:2]] == ["2005.11401", "2007.01282"]
    assert result.rerank_signals["resolvedSourceIds"][:2] == ["2005.11401", "2007.01282"]


def test_retrieval_pipeline_compare_preserves_resolved_pair_after_memory_merge():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    def _result(doc_id: str, title: str, paper_id: str, *, score: float) -> SearchResult:
        return SearchResult(
            document=f"{title} abstract",
            metadata={"title": title, "source_type": "paper", "arxiv_id": paper_id},
            distance=max(0.0, 1.0 - score),
            score=score,
            semantic_score=score,
            lexical_score=score,
            retrieval_mode="keyword",
            lexical_extras={"query": title},
            document_id=doc_id,
        )

    def _lexical_stub(searcher_obj, query, *, top_k, filter_dict):  # noqa: ANN001
        _ = (searcher_obj, query, top_k)
        paper_id = str((filter_dict or {}).get("arxiv_id") or "").strip()
        if paper_id == "2005.11401":
            return [_result("rag", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "2005.11401", score=0.66)]
        if paper_id == "2007.01282":
            return [_result("fid", "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", "2007.01282", score=0.65)]
        generic_hits = [
            _result("fid", "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", "2007.01282", score=0.91),
            _result("realm", "REALM: Retrieval-Augmented Language Model Pre-Training", "2002.08909", score=0.89),
            _result("atlas", "Atlas: Few-shot Learning with Retrieval Augmented Language Models", "2208.03299", score=0.88),
        ]
        return generic_hits[:top_k]

    compatibility_results = [
        _result("fid", "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", "2007.01282", score=0.91),
        _result("realm", "REALM: Retrieval-Augmented Language Model Pre-Training", "2002.08909", score=0.89),
        _result("atlas", "Atlas: Few-shot Learning with Retrieval Augmented Language Models", "2208.03299", score=0.88),
    ]

    with patch("knowledge_hub.ai.retrieval_pipeline.lexical_search", _lexical_stub), patch(
        "knowledge_hub.ai.retrieval_pipeline.semantic_search",
        lambda *args, **kwargs: [],
    ), patch.object(
        searcher,
        "_search_with_paper_memory_prefilter",
        return_value=(
            compatibility_results,
            {
                "requestedMode": "off",
                "applied": False,
                "fallbackUsed": True,
                "matchedPaperIds": [],
                "matchedMemoryIds": [],
                "reason": "fallback_generic_search",
            },
        ),
    ):
        result = RetrievalPipelineService(searcher).execute(
            query="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks와 Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering를 비교해줘",
            top_k=3,
            source_type="paper",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            memory_route_mode="off",
            query_frame={
                "source_type": "paper",
                "family": "paper_compare",
                "query_intent": "comparison",
                "entities": ["Retrieval-Augmented Generation", "Leveraging Passage Retrieval"],
                "resolved_source_ids": ["2005.11401", "2007.01282", "2510.15682"],
            },
        )

    assert [item.metadata.get("arxiv_id") for item in result.results[:2]] == ["2005.11401", "2007.01282"]


def test_retrieval_pipeline_does_not_apply_memory_boost_when_memory_mode_is_off_with_compat_fallback():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    def _result(doc_id: str, title: str, paper_id: str, *, score: float) -> SearchResult:
        return SearchResult(
            document=f"{title} abstract",
            metadata={"title": title, "source_type": "paper", "arxiv_id": paper_id},
            distance=max(0.0, 1.0 - score),
            score=score,
            semantic_score=score,
            lexical_score=score,
            retrieval_mode="keyword",
            lexical_extras={"query": title},
            document_id=doc_id,
        )

    base_results = [
        _result("target", "Target Paper", "1111.11111", score=0.84),
        _result("distractor", "Distractor Paper", "2222.22222", score=0.80),
    ]
    memory_results = [
        _result("distractor", "Distractor Paper", "2222.22222", score=0.80),
    ]

    with patch.object(
        RetrievalPipelineService,
        "_run_base_search",
        return_value=(
            base_results,
            {"lexicalQueryForms": ["target"]},
            [{"sourceType": "paper", "budget": 3, "semanticHits": 0, "lexicalHits": 2}],
        ),
    ), patch.object(
        searcher,
        "_search_with_paper_memory_prefilter",
        return_value=(
            memory_results,
            {
                "requestedMode": "off",
                "applied": False,
                "fallbackUsed": True,
                "matchedPaperIds": [],
                "matchedMemoryIds": [],
                "reason": "fallback_generic_search",
            },
        ),
    ):
        result = RetrievalPipelineService(searcher).execute(
            query="target paper 설명",
            top_k=2,
            source_type="paper",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            memory_route_mode="off",
        )

    assert result.results
    assert result.results[0].metadata.get("arxiv_id") == "1111.11111"
    top_signals = dict(result.results[0].lexical_extras or {}).get("ranking_signals") or {}
    assert "memory_prior_boost" not in top_signals


def test_retrieval_pipeline_memory_mode_compat_rescues_without_memory_prior_boost():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    def _result(doc_id: str, title: str, paper_id: str, *, score: float) -> SearchResult:
        return SearchResult(
            document=f"{title} abstract",
            metadata={"title": title, "source_type": "paper", "arxiv_id": paper_id},
            distance=max(0.0, 1.0 - score),
            score=score,
            semantic_score=score,
            lexical_score=score,
            retrieval_mode="keyword",
            lexical_extras={"query": title},
            document_id=doc_id,
        )

    base_results = [
        _result("target", "Target Paper", "1111.11111", score=0.84),
    ]
    memory_results = [
        _result("memory", "Memory Paper", "2222.22222", score=0.80),
    ]

    with patch.object(
        RetrievalPipelineService,
        "_run_base_search",
        return_value=(
            base_results,
            {"lexicalQueryForms": ["target"]},
            [{"sourceType": "paper", "budget": 3, "semanticHits": 0, "lexicalHits": 1}],
        ),
    ), patch.object(
        searcher,
        "_search_with_paper_memory_prefilter",
        return_value=(
            memory_results,
            {
                "requestedMode": "prefilter",
                "effectiveMode": "compat",
                "applied": True,
                "fallbackUsed": False,
                "matchedPaperIds": ["2222.22222"],
                "matchedMemoryIds": ["memory-a"],
                "reason": "matched_cards",
            },
        ),
    ):
        result = RetrievalPipelineService(searcher).execute(
            query="memory paper 설명",
            top_k=2,
            source_type="paper",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            memory_route_mode="prefilter",
        )

    paper_ids = [item.metadata.get("arxiv_id") for item in result.results]
    assert paper_ids == ["1111.11111", "2222.22222"]
    diagnostics = result.diagnostics()
    assert diagnostics["memoryRoute"]["requestedMode"] == "prefilter"
    assert diagnostics["memoryRoute"]["effectiveMode"] == "compat"
    assert diagnostics["memoryRoute"]["memoryInfluenceApplied"] is True
    for item in result.results:
        extras = dict(item.lexical_extras or {})
        signals = dict(extras.get("ranking_signals") or {})
        assert "memory_prior_boost" not in extras
        assert "memory_prior_boost" not in signals
        assert "memory_provenance" not in extras


def test_retrieval_pipeline_memory_mode_on_prefers_memory_results_more_strongly():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    def _result(doc_id: str, title: str, paper_id: str, *, score: float) -> SearchResult:
        return SearchResult(
            document=f"{title} abstract",
            metadata={"title": title, "source_type": "paper", "arxiv_id": paper_id},
            distance=max(0.0, 1.0 - score),
            score=score,
            semantic_score=score,
            lexical_score=score,
            retrieval_mode="keyword",
            lexical_extras={"query": title},
            document_id=doc_id,
        )

    base_results = [
        _result("target", "Target Paper", "1111.11111", score=0.84),
        _result("memory", "Memory Paper", "2222.22222", score=0.80),
    ]
    memory_results = [
        _result("memory", "Memory Paper", "2222.22222", score=0.80),
    ]

    with patch.object(
        RetrievalPipelineService,
        "_run_base_search",
        return_value=(
            base_results,
            {"lexicalQueryForms": ["target"]},
            [{"sourceType": "paper", "budget": 3, "semanticHits": 0, "lexicalHits": 2}],
        ),
    ), patch.object(
        searcher,
        "_search_with_paper_memory_prefilter",
        return_value=(
            memory_results,
            {
                "requestedMode": "on",
                "effectiveMode": "on",
                "applied": True,
                "fallbackUsed": False,
                "matchedPaperIds": ["2222.22222"],
                "matchedMemoryIds": ["memory-a"],
                "reason": "matched_cards",
            },
        ),
    ):
        result = RetrievalPipelineService(searcher).execute(
            query="memory paper 설명",
            top_k=2,
            source_type="paper",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            memory_route_mode="on",
        )

    assert result.results
    assert result.results[0].metadata.get("arxiv_id") == "2222.22222"
    diagnostics = result.diagnostics()
    assert diagnostics["memoryRoute"]["effectiveMode"] == "on"
    assert diagnostics["memoryRoute"]["memoryInfluenceApplied"] is True
    top_signals = dict(result.results[0].lexical_extras or {}).get("ranking_signals") or {}
    assert top_signals["memory_prior_boost"] >= 0.12


def test_retrieval_pipeline_override_preserves_paper_prefilter_shim_path():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    def _result(doc_id: str, title: str, paper_id: str, *, score: float) -> SearchResult:
        return SearchResult(
            document=f"{title} abstract",
            metadata={"title": title, "source_type": "paper", "arxiv_id": paper_id},
            distance=max(0.0, 1.0 - score),
            score=score,
            semantic_score=score,
            lexical_score=score,
            retrieval_mode="keyword",
            lexical_extras={"query": title},
            document_id=doc_id,
        )

    shim_results = [_result("memory", "Memory Paper", "2222.22222", score=0.8)]
    base_results = [_result("target", "Target Paper", "1111.11111", score=0.84)]

    with patch.object(
        RetrievalPipelineService,
        "_run_base_search",
        return_value=(
            base_results,
            {"lexicalQueryForms": ["target"]},
            [{"sourceType": "paper", "budget": 3, "semanticHits": 0, "lexicalHits": 1}],
        ),
    ), patch.object(
        searcher,
        "_search_with_paper_memory_prefilter",
        return_value=(
            shim_results,
            {
                "requestedMode": "prefilter",
                "applied": True,
                "fallbackUsed": False,
                "matchedPaperIds": ["2222.22222"],
                "matchedMemoryIds": ["memory-a"],
                "reason": "matched_cards",
            },
        ),
    ) as shim_mock, patch.object(
        RetrievalPipelineService,
        "_search_with_paper_memory_prefilter_direct",
        side_effect=AssertionError("direct paper prefilter path should not run when shim is overridden"),
    ):
        result = RetrievalPipelineService(searcher).execute(
            query="memory paper 설명",
            top_k=2,
            source_type="paper",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            memory_route_mode="prefilter",
        )

    assert shim_mock.call_count == 1
    assert result.results


def test_retrieval_pipeline_override_preserves_generic_prefilter_shim_path():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    def _result(doc_id: str, title: str, *, score: float) -> SearchResult:
        return SearchResult(
            document=f"{title} note",
            metadata={"title": title, "source_type": "vault", "file_path": f"AI/{doc_id}.md"},
            distance=max(0.0, 1.0 - score),
            score=score,
            semantic_score=score,
            lexical_score=score,
            retrieval_mode="keyword",
            lexical_extras={"query": title},
            document_id=doc_id,
        )

    shim_results = [_result("memory", "Memory Note", score=0.8)]
    base_results = [_result("target", "Target Note", score=0.84)]

    with patch.object(
        RetrievalPipelineService,
        "_run_base_search",
        return_value=(
            base_results,
            {"lexicalQueryForms": ["target"]},
            [{"sourceType": "vault", "budget": 3, "semanticHits": 0, "lexicalHits": 1}],
        ),
    ), patch.object(
        searcher,
        "_search_with_memory_prefilter",
        return_value=(
            shim_results,
            {
                "requestedMode": "compat",
                "effectiveMode": "compat",
                "applied": True,
                "fallbackUsed": False,
                "matchedMemoryIds": ["memory-a"],
                "matchedDocumentIds": ["AI/memory.md"],
                "reason": "matched_document_memory",
            },
        ),
    ) as shim_mock, patch.object(
        RetrievalPipelineService,
        "_search_with_memory_prefilter_direct",
        side_effect=AssertionError("direct generic prefilter path should not run when shim is overridden"),
    ):
        result = RetrievalPipelineService(searcher).execute(
            query="memory note 설명",
            top_k=2,
            source_type="vault",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            memory_route_mode="compat",
        )

    assert shim_mock.call_count == 1
    assert result.results


def test_retrieval_pipeline_non_override_paper_prefilter_uses_direct_helper():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )

    def _result(doc_id: str, title: str, paper_id: str, *, score: float) -> SearchResult:
        return SearchResult(
            document=f"{title} abstract",
            metadata={"title": title, "source_type": "paper", "arxiv_id": paper_id},
            distance=max(0.0, 1.0 - score),
            score=score,
            semantic_score=score,
            lexical_score=score,
            retrieval_mode="keyword",
            lexical_extras={"query": title},
            document_id=doc_id,
        )

    base_results = [_result("target", "Target Paper", "1111.11111", score=0.84)]
    direct_results = [_result("memory", "Memory Paper", "2222.22222", score=0.8)]

    with patch.object(
        RetrievalPipelineService,
        "_run_base_search",
        return_value=(
            base_results,
            {"lexicalQueryForms": ["target"]},
            [{"sourceType": "paper", "budget": 3, "semanticHits": 0, "lexicalHits": 1}],
        ),
    ), patch.object(
        RetrievalPipelineService,
        "_search_with_paper_memory_prefilter_direct",
        return_value=(
            direct_results,
            {
                "requestedMode": "prefilter",
                "applied": True,
                "fallbackUsed": False,
                "matchedPaperIds": ["2222.22222"],
                "matchedMemoryIds": ["memory-a"],
                "reason": "matched_cards",
            },
        ),
    ) as direct_mock:
        result = RetrievalPipelineService(searcher).execute(
            query="memory paper 설명",
            top_k=2,
            source_type="paper",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            memory_route_mode="prefilter",
        )

    assert direct_mock.call_count == 1
    assert [item.metadata.get("arxiv_id") for item in result.results] == ["1111.11111", "2222.22222"]


def test_retrieval_pipeline_cluster_scope_uses_direct_ctx_helpers(tmp_path):
    records = [
        {
            "id": "rag-note",
            "document": "RAG note for grouping",
            "distance": 0.08,
            "metadata": {"title": "RAG Note", "source_type": "vault", "file_path": "AI/rag.md"},
        }
    ]
    _build_topology_snapshot(tmp_path, file_path="AI/rag.md")
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB(records),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
        config=SimpleNamespace(vault_path=str(tmp_path)),
    )
    searcher._get_active_profile = lambda: (_ for _ in ()).throw(AssertionError("legacy profile shim should not be used"))
    searcher._load_topology_index = lambda: (_ for _ in ()).throw(AssertionError("legacy topology shim should not be used"))

    result = RetrievalPipelineService(searcher).execute(
        query="RAG 관련 논문들을 그룹으로 추천해줘",
        top_k=3,
        retrieval_mode="semantic",
        use_ontology_expansion=True,
        memory_route_mode="off",
    )

    assert result.diagnostics()["contextExpansion"]["clusterUsed"] is True
    assert result.related_clusters[0]["cluster_id"] == "c1"


def test_retrieval_pipeline_compare_uses_card_fallback_when_resolved_paper_is_not_indexed():
    sqlite_db = _CompareCardSQLite(
        {},
        paper_cards={
            "2005.11401": {
                "paper_id": "2005.11401",
                "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "method_core": "RAG combines dense passage retrieval with seq2seq generation.",
            },
            "2007.01282": {
                "paper_id": "2007.01282",
                "title": "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
                "method_core": "FiD encodes retrieved passages independently and fuses them in the decoder.",
            },
        },
        papers={
            "2005.11401": {"arxiv_id": "2005.11401", "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"},
            "2007.01282": {"arxiv_id": "2007.01282", "title": "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering"},
        },
    )
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=sqlite_db,
    )

    def _result(doc_id: str, title: str, paper_id: str, *, score: float) -> SearchResult:
        return SearchResult(
            document=f"{title} abstract",
            metadata={"title": title, "source_type": "paper", "arxiv_id": paper_id},
            distance=max(0.0, 1.0 - score),
            score=score,
            semantic_score=0.0,
            lexical_score=score,
            retrieval_mode="keyword",
            lexical_extras={"query": title},
            document_id=doc_id,
        )

    def _lexical_stub(searcher_obj, query, *, top_k, filter_dict):  # noqa: ANN001
        _ = (searcher_obj, query, top_k)
        paper_id = str((filter_dict or {}).get("arxiv_id") or "").strip()
        if paper_id:
            return []
        return [
            _result("realm", "REALM: Retrieval-Augmented Language Model Pre-Training", "2002.08909", score=0.89),
            _result("atlas", "Atlas: Few-shot Learning with Retrieval Augmented Language Models", "2208.03299", score=0.88),
        ]

    with patch("knowledge_hub.ai.retrieval_pipeline.lexical_search", _lexical_stub), patch(
        "knowledge_hub.ai.retrieval_pipeline.semantic_search",
        lambda *args, **kwargs: [],
    ):
        result = RetrievalPipelineService(searcher).execute(
            query="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks와 Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering를 비교해줘",
            top_k=3,
            source_type="paper",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            memory_route_mode="off",
            query_frame={
                "source_type": "paper",
                "family": "paper_compare",
                "query_intent": "comparison",
                "entities": ["Retrieval-Augmented Generation", "Leveraging Passage Retrieval"],
                "resolved_source_ids": ["2005.11401", "2007.01282", "2510.15682"],
            },
        )

    assert [item.metadata.get("arxiv_id") for item in result.results[:2]] == ["2005.11401", "2007.01282"]
    assert result.results[0].retrieval_mode == "paper-card-v2-fallback"
    assert dict(result.results[0].lexical_extras or {}).get("compare_resolved_paper_fallback") is True


def test_retrieval_pipeline_concept_explainer_promotes_resolved_paper_card_fallback():
    sqlite_db = _CompareCardSQLite(
        {},
        paper_cards={
            "alexnet-2012": {
                "paper_id": "alexnet-2012",
                "title": "ImageNet Classification with Deep Convolutional Neural Networks",
                "paper_core": "AlexNet established deep convolutional neural networks for ImageNet classification.",
            },
        },
        papers={
            "alexnet-2012": {
                "arxiv_id": "alexnet-2012",
                "title": "ImageNet Classification with Deep Convolutional Neural Networks",
            }
        },
    )
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=sqlite_db,
    )

    def _result(doc_id: str, title: str, paper_id: str, *, score: float) -> SearchResult:
        return SearchResult(
            document=f"{title} overview",
            metadata={"title": title, "source_type": "paper", "arxiv_id": paper_id},
            distance=max(0.0, 1.0 - score),
            score=score,
            semantic_score=0.0,
            lexical_score=score,
            retrieval_mode="keyword",
            lexical_extras={"query": title},
            document_id=doc_id,
        )

    def _lexical_stub(searcher_obj, query, *, top_k, filter_dict):  # noqa: ANN001
        _ = (searcher_obj, query, top_k)
        if str((filter_dict or {}).get("arxiv_id") or "").strip():
            return []
        return [
            _result("faster-rcnn", "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", "1506.01497", score=0.89),
        ]

    with patch("knowledge_hub.ai.retrieval_pipeline.lexical_search", _lexical_stub), patch(
        "knowledge_hub.ai.retrieval_pipeline.semantic_search",
        lambda *args, **kwargs: [],
    ):
        result = RetrievalPipelineService(searcher).execute(
            query="CNN을 쉽게 설명해줘",
            top_k=3,
            source_type="paper",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            memory_route_mode="off",
            query_frame={
                "source_type": "paper",
                "family": "concept_explainer",
                "query_intent": "definition",
                "entities": ["CNN"],
                "expanded_terms": ["ImageNet Classification with Deep Convolutional Neural Networks", "CNN"],
                "resolved_source_ids": ["alexnet-2012"],
            },
        )

    assert result.results[0].metadata.get("arxiv_id") == "alexnet-2012"
    assert result.results[0].retrieval_mode == "paper-card-v2-fallback"
    assert dict(result.results[0].lexical_extras or {}).get("concept_resolved_paper_fallback") is True


def test_retrieval_pipeline_paper_lookup_uses_card_fallback_when_resolved_paper_has_no_vectors():
    sqlite_db = _CompareCardSQLite(
        {},
        paper_cards={
            "1706.03762": {
                "paper_id": "1706.03762",
                "title": "Attention Is All You Need",
                "method_core": "Transformer replaces recurrence with multi-head self-attention.",
            },
        },
        papers={
            "1706.03762": {
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
            }
        },
    )
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=sqlite_db,
    )

    with patch("knowledge_hub.ai.retrieval_pipeline.lexical_search", lambda *args, **kwargs: []), patch(
        "knowledge_hub.ai.retrieval_pipeline.semantic_search",
        lambda *args, **kwargs: [],
    ):
        result = RetrievalPipelineService(searcher).execute(
            query="Attention Is All You Need 논문 설명해줘",
            top_k=2,
            source_type="paper",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            memory_route_mode="off",
            query_frame={
                "source_type": "paper",
                "family": "paper_lookup",
                "query_intent": "paper_lookup",
                "entities": ["Attention Is All You Need"],
                "expanded_terms": ["Attention Is All You Need"],
                "resolved_source_ids": ["1706.03762"],
            },
        )

    assert result.results[0].metadata.get("arxiv_id") == "1706.03762"
    assert result.results[0].retrieval_mode == "paper-card-v2-fallback"
    assert dict(result.results[0].lexical_extras or {}).get("paper_lookup_resolved_paper_fallback") is True


def test_retrieval_pipeline_paper_discover_uses_card_search_fallback_when_vector_search_is_empty():
    sqlite_db = _PaperCardSearchSQLite(
        {},
        paper_cards={
            "2504.10449": {
                "paper_id": "2504.10449",
                "title": "M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models",
                "paper_core": "M1 studies Mamba-style reasoning models as a Transformer alternative.",
            },
        },
        papers={
            "2504.10449": {
                "arxiv_id": "2504.10449",
                "title": "M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models",
            }
        },
        search_rows={
            "Mamba": [
                {
                    "paper_id": "2504.10449",
                    "title": "M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models",
                    "paper_core": "M1 studies Mamba-style reasoning models as a Transformer alternative.",
                    "match_score": 8,
                }
            ]
        },
    )
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=sqlite_db,
    )

    with patch("knowledge_hub.ai.retrieval_pipeline.lexical_search", lambda *args, **kwargs: []), patch(
        "knowledge_hub.ai.retrieval_pipeline.semantic_search",
        lambda *args, **kwargs: [],
    ):
        result = RetrievalPipelineService(searcher).execute(
            query="state space model 계열 논문들을 찾아 정리해줘",
            top_k=3,
            source_type="paper",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            memory_route_mode="off",
            query_plan={
                "family": "paper_discover",
                "expanded_terms": ["Mamba", "RetNet", "state", "space", "model"],
            },
            query_frame={
                "source_type": "paper",
                "family": "paper_discover",
                "query_intent": "paper_topic",
                "entities": ["state", "space", "model"],
                "expanded_terms": ["Mamba", "RetNet", "state", "space", "model"],
                "resolved_source_ids": [],
            },
        )

    assert result.results
    assert result.results[0].metadata.get("arxiv_id") == "2504.10449"
    assert result.results[0].retrieval_mode == "paper-card-v2-fallback"
    assert dict(result.results[0].lexical_extras or {}).get("paper_discover_card_fallback") is True


def test_retrieval_pipeline_applies_web_reference_prefilter_from_query_frame():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )
    query_frame = build_query_frame(
        domain_key="web_knowledge",
        source_type="web",
        family="reference_explainer",
        query_intent="definition",
        answer_mode="concise_summary",
        expanded_terms=["vector search rerank", "example.com"],
        resolved_source_ids=["https://example.com/vector-rerank", "web_1234abcd"],
        confidence=0.9,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="web_reference_explainer_policy",
        metadata_filter={
            "source_type": "web",
            "reference_only": True,
        },
    )

    plan = RetrievalPipelineService(searcher).build_plan(
        query="example.com rerank guide 설명",
        top_k=4,
        source_type="web",
        retrieval_mode="hybrid",
        memory_route_mode="off",
        use_ontology_expansion=False,
        query_frame=query_frame.to_dict(),
    )

    assert plan.source_scope == "web"
    assert plan.paper_family == "reference_explainer"
    assert plan.resolved_source_scope_applied is True
    assert plan.reference_source_applied is True
    assert plan.metadata_filter_applied["url"] == "https://example.com/vector-rerank"
    assert "canonical_url" not in plan.metadata_filter_applied
    assert plan.metadata_filter_applied["document_id"] == "web_1234abcd"


def test_retrieval_pipeline_applies_web_temporal_watchlist_diagnostics():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )
    query_frame = build_query_frame(
        domain_key="web_knowledge",
        source_type="web",
        family="temporal_update",
        query_intent="temporal",
        answer_mode="timeline_compare",
        expanded_terms=["latest rerank update", "version grounding"],
        resolved_source_ids=[],
        confidence=0.9,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="web_temporal_update_policy",
        metadata_filter={
            "source_type": "web",
            "latest_only": True,
            "temporal_required": True,
            "watchlist_scope": "latest_feed",
        },
    )

    plan = RetrievalPipelineService(searcher).build_plan(
        query="최근 rerank 업데이트는 뭐가 changed?",
        top_k=4,
        source_type="web",
        retrieval_mode="hybrid",
        memory_route_mode="off",
        use_ontology_expansion=False,
        query_frame=query_frame.to_dict(),
    )

    serialized = plan.to_dict()
    assert plan.paper_family == "temporal_update"
    assert plan.watchlist_scope_applied is True
    assert plan.prefilter_reason == "temporal_grounding_required"
    assert serialized["referenceSourceApplied"] is False
    assert serialized["watchlistScopeApplied"] is True


def test_retrieval_pipeline_web_reference_lexical_forms_use_alias_terms_not_generic_prompt():
    searcher = RAGSearcher(
        DummyEmbedder(),
        DummyVectorDB([]),
        llm=FakeLLM(),
        sqlite_db=DummyFeatureSQLite({}),
    )
    lexical_queries: list[str] = []

    def _lexical_stub(searcher_obj, query, *, top_k, filter_dict):  # noqa: ANN001
        _ = (searcher_obj, top_k, filter_dict)
        lexical_queries.append(query)
        return []

    with patch("knowledge_hub.ai.retrieval_pipeline.lexical_search", _lexical_stub):
        result = RetrievalPipelineService(searcher).execute(
            query="web card v2에서 version grounding이 필요한 이유는 무엇인가?",
            top_k=3,
            source_type="web",
            retrieval_mode="keyword",
            use_ontology_expansion=False,
            query_plan={
                "family": "reference_explainer",
                "expanded_terms": ["web card v2", "version grounding", "guide", "reference"],
            },
            query_frame={
                "source_type": "web",
                "family": "reference_explainer",
                "query_intent": "definition",
                "entities": ["web card v2", "version grounding"],
                "metadata_filter": {"source_type": "web", "reference_only": True},
            },
        )

    assert "guide" not in result.rerank_signals["lexicalQueryForms"]
    assert "reference" not in result.rerank_signals["lexicalQueryForms"]
    assert "web card v2" in result.rerank_signals["lexicalQueryForms"]
    assert "version grounding" in result.rerank_signals["lexicalQueryForms"]
