"""RAG 검색 모듈 단위 테스트."""

from __future__ import annotations

import sqlite3

import pytest

from knowledge_hub.application.rag_reports import build_rag_ops_report
from knowledge_hub.ai.rag import RAGSearcher
from knowledge_hub.ai.retrieval_fit import normalize_source_type
from knowledge_hub.core.config import Config
from knowledge_hub.core.models import SearchResult
from knowledge_hub.core.rag_answer_log_store import RAGAnswerLogStore


class DummyEmbedder:
    def embed_text(self, text: str):
        return [0.0]


class FakeLLM:
    def __init__(self):
        self.last_context: str | None = None
        self.last_prompt: str | None = None
        self.calls = 0

    def generate(self, prompt: str, context: str = ""):
        self.calls += 1
        self.last_prompt = prompt
        self.last_context = context
        return "요약 답변"

    def stream_generate(self, prompt: str, context: str = ""):
        self.last_prompt = prompt
        self.last_context = context
        yield "요약 답변"


class StaticLLM(FakeLLM):
    def __init__(self, response: str):
        super().__init__()
        self.response = response

    def generate(self, prompt: str, context: str = ""):
        self.calls += 1
        self.last_prompt = prompt
        self.last_context = context
        return self.response

    def stream_generate(self, prompt: str, context: str = ""):
        self.calls += 1
        self.last_prompt = prompt
        self.last_context = context
        yield self.response


class FailingLLM(FakeLLM):
    def __init__(self, error: Exception):
        super().__init__()
        self.error = error

    def generate(self, prompt: str, context: str = ""):
        self.calls += 1
        self.last_prompt = prompt
        self.last_context = context
        raise self.error

    def stream_generate(self, prompt: str, context: str = ""):
        self.calls += 1
        self.last_prompt = prompt
        self.last_context = context
        raise self.error


class DummyHybridDecision:
    def __init__(self, route: str, provider: str = "ollama", model: str = "qwen2.5:7b", timeout_sec: int = 45):
        self.route = route
        self.provider = provider
        self.model = model
        self.timeout_sec = timeout_sec

    def to_dict(self):
        return {
            "route": self.route,
            "provider": self.provider,
            "model": self.model,
            "reasons": [],
            "timeoutSec": self.timeout_sec,
            "complexityScore": 0,
            "threshold": 0,
            "fallbackUsed": False,
        }


class DummyVectorDB:
    def __init__(self, records):
        self.records = records

    def search(self, query_embedding, top_k: int, filter_dict=None):
        candidates = self._filter_records(filter_dict)
        candidates = sorted(candidates, key=lambda item: item["distance"])
        candidates = candidates[:top_k]

        return {
            "documents": [[r["document"] for r in candidates]],
            "metadatas": [[r["metadata"] for r in candidates]],
            "distances": [[r["distance"] for r in candidates]],
            "ids": [[r["id"] for r in candidates]],
        }

    def get_documents(self, filter_dict=None, limit=500, include_ids=True, include_documents=True, include_metadatas=True):
        candidates = self._filter_records(filter_dict)
        candidates = candidates[:limit]

        return {
            "documents": [r["document"] for r in candidates],
            "metadatas": [r["metadata"] for r in candidates],
            "ids": [r["id"] for r in candidates],
        }

    def _filter_records(self, filter_dict):
        if not filter_dict:
            return self.records
        filtered = self.records
        for key, value in (filter_dict or {}).items():
            if isinstance(value, dict) and "$eq" in value:
                expected = value["$eq"]
            else:
                expected = value
            filtered = [r for r in filtered if r["metadata"].get(key) == expected]
        return filtered


class DummyVectorDBWithLexical(DummyVectorDB):
    def lexical_search(self, query, top_k=5, filter_dict=None):
        _ = query
        candidates = self._filter_records(filter_dict)
        hits = [
            {
                "id": item["id"],
                "document": item["document"],
                "metadata": item["metadata"],
                "score": 0.9 if item["metadata"].get("title") == "Paper B" else 0.4,
                "rank": 0.1 if item["metadata"].get("title") == "Paper B" else 0.9,
            }
            for item in candidates[:top_k]
        ]
        hits.sort(key=lambda row: row["score"], reverse=True)
        return hits


class DummyVectorDBWithLexicalRescue(DummyVectorDB):
    def lexical_search(self, query, top_k=5, filter_dict=None):
        _ = query
        candidates = self._filter_records(filter_dict)
        hits = []
        for item in candidates[:top_k]:
            title = str(item["metadata"].get("title") or "")
            lexical_match = "강화 학습" in title
            hits.append(
                {
                    "id": item["id"],
                    "document": item["document"],
                    "metadata": item["metadata"],
                    "score": 1.0 if lexical_match else 0.2,
                    "rank": -12.0 if lexical_match else -1.0,
                }
            )
        hits.sort(key=lambda row: row["score"], reverse=True)
        return hits


class DummyVectorDBWithEmptyLexical(DummyVectorDB):
    def lexical_search(self, query, top_k=5, filter_dict=None):
        _ = query, top_k, filter_dict
        return []

    def get_documents(self, *args, **kwargs):
        raise AssertionError("brute-force lexical fallback should not run when lexical_search is available")


class DummyFeatureSQLite:
    def __init__(self, snapshots, *, notes=None, ko_note_items=None, ontology_entities=None, entity_aliases=None, concept_papers=None):
        self.snapshots = snapshots
        self.notes = notes or {}
        self.ko_note_items = ko_note_items or {}
        self.ontology_entities = list(ontology_entities or [])
        self.entity_aliases = dict(entity_aliases or {})
        self.concept_papers = dict(concept_papers or {})

    def find_source_feature_snapshot(self, note_id="", record_id="", canonical_url="", source_item_id=""):
        for candidate in [source_item_id, record_id, canonical_url, note_id]:
            token = str(candidate or "").strip()
            if token and token in self.snapshots:
                return self.snapshots[token]
        return None

    def get_note(self, note_id):
        return self.notes.get(str(note_id or "").strip())

    def find_ko_note_item_by_final_path(self, *, final_path, item_type=None, statuses=("approved", "applied")):
        _ = (item_type, statuses)
        return self.ko_note_items.get(str(final_path or "").strip())

    def list_claims_by_note(self, note_id, limit=20):
        _ = (note_id, limit)
        return []

    def list_claims_by_record(self, record_id, limit=20):
        _ = (record_id, limit)
        return []

    def list_beliefs_by_claim_ids(self, claim_ids, limit=200):
        _ = (claim_ids, limit)
        return []

    def list_ontology_entities(self, limit=5000):
        _ = limit
        return list(self.ontology_entities)

    def get_entity_aliases(self, entity_id):
        return list(self.entity_aliases.get(str(entity_id or "").strip(), []))

    def get_concept_papers(self, concept_id):
        return list(self.concept_papers.get(str(concept_id or "").strip(), []))


class DummyCompareFallbackSQLite(DummyFeatureSQLite):
    def __init__(self, snapshots, *, paper_cards=None, papers=None):
        super().__init__(snapshots)
        self.paper_cards = dict(paper_cards or {})
        self.papers = dict(papers or {})

    def search_paper_cards_v2(self, query, limit=5):
        _ = (query, limit)
        return []

    def search_papers(self, query, limit=20):
        _ = limit
        token = str(query or "").strip()
        rows = []
        for paper_id, row in self.papers.items():
            title = str((row or {}).get("title") or "").strip()
            if token == title:
                rows.append({"arxiv_id": paper_id, "title": title})
        return rows

    def get_paper_card_v2(self, paper_id):
        return self.paper_cards.get(str(paper_id or "").strip())

    def get_paper(self, paper_id):
        return self.papers.get(str(paper_id or "").strip())


class DummyClaimSQLite(DummyFeatureSQLite):
    def __init__(self, snapshots, *, record_claims=None, beliefs=None, notes=None, ko_note_items=None):
        super().__init__(snapshots, notes=notes, ko_note_items=ko_note_items)
        self.record_claims = record_claims or {}
        self.beliefs = beliefs or []

    def list_claims_by_record(self, record_id, limit=20):
        return list(self.record_claims.get(str(record_id or "").strip(), []))[:limit]

    def list_beliefs_by_claim_ids(self, claim_ids, limit=200):
        wanted = set(claim_ids)
        return [
            belief
            for belief in self.beliefs
            if wanted & {str(item).strip() for item in belief.get("derived_from_claim_ids", [])}
        ][:limit]


class DummyClaimNormalizationSQLite(DummyClaimSQLite):
    def __init__(self, snapshots, *, record_claims=None, beliefs=None, notes=None, ko_note_items=None, normalizations=None):
        super().__init__(snapshots, record_claims=record_claims, beliefs=beliefs, notes=notes, ko_note_items=ko_note_items)
        self.normalizations = normalizations or {}

    def list_claim_normalizations(self, claim_ids=None, status=None, limit=100, **kwargs):  # noqa: ANN001
        _ = (status, limit, kwargs)
        rows = []
        for claim_id in claim_ids or []:
            payload = self.normalizations.get(str(claim_id).strip())
            if payload:
                rows.append({"claim_id": str(claim_id).strip(), **payload})
        return rows


class DummyFeatureSQLiteWithLogs(DummyFeatureSQLite):
    def __init__(self, snapshots, *, notes=None, ko_note_items=None):
        super().__init__(snapshots, notes=notes, ko_note_items=ko_note_items)
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._log_store = RAGAnswerLogStore(self._conn)
        self._log_store.ensure_schema()

    def add_rag_answer_log(self, **kwargs):
        return self._log_store.add_log(**kwargs)

    def list_rag_answer_logs(self, *, limit=100, days=0):
        return self._log_store.list_logs(limit=limit, days=days)


def test_normalize_source_type_treats_all_as_unscoped():
    assert normalize_source_type("all") == ""
    assert normalize_source_type("*") == ""
    assert normalize_source_type("note") == "vault"


def _build_records():
    return [
        {
            "id": "paper-a",
            "document": "attention is needed for seq2seq training.",
            "distance": 0.2,
            "metadata": {
                "title": "Paper A",
                "source_type": "paper",
                "contextual_summary": "[Paper A] attention is needed for seq2seq training.",
                "section_title": "Introduction",
            },
        },
        {
            "id": "paper-b",
            "document": "transformer uses attention mechanism repeatedly.",
            "distance": 0.5,
            "metadata": {
                "title": "Paper B",
                "source_type": "paper",
                "contextual_summary": "[Paper B] transformer uses attention mechanism repeatedly.",
                "section_title": "Method",
            },
        },
        {
            "id": "paper-c",
            "document": "finance is unrelated to attention mechanism.",
            "distance": 0.9,
            "metadata": {
                "title": "Paper C",
                "source_type": "web",
                "contextual_summary": "[Paper C] finance is unrelated.",
            },
        },
    ]


def test_search_hybrid_mode_balances_semantic_and_lexical_scores():
    db = DummyVectorDB(_build_records())
    searcher = RAGSearcher(DummyEmbedder(), db, llm=None)

    results = searcher.search("attention mechanism", top_k=2, source_type=None, retrieval_mode="hybrid", alpha=0.6)

    assert len(results) == 2
    assert results[0].metadata["title"] == "Paper B"
    assert results[0].score >= results[1].score
    assert results[0].semantic_score >= 0
    assert results[0].lexical_score >= 0


def test_search_hybrid_without_metadata_keeps_near_current_behavior():
    db = DummyVectorDB(_build_records())
    sqlite_db = DummyFeatureSQLite({})
    searcher = RAGSearcher(DummyEmbedder(), db, llm=None, sqlite_db=sqlite_db)

    results = searcher.search("attention mechanism", top_k=2, source_type=None, retrieval_mode="hybrid", alpha=0.6)

    assert len(results) == 2
    assert results[0].metadata["title"] == "Paper B"
    assert results[0].lexical_extras["quality_flag"] == "unscored"
    assert results[0].lexical_extras["reference_prior_boost"] == 0.0


def test_search_hybrid_adds_paper_candidates_when_unfiltered_pool_is_vault_heavy():
    records = [
        {
            "id": "vault-1",
            "document": "generic glossary note",
            "distance": 0.34,
            "metadata": {"title": "용어", "source_type": "vault", "file_path": "Projects/AI/용어.md"},
        },
        {
            "id": "vault-2",
            "document": "generic reference summary",
            "distance": 0.35,
            "metadata": {"title": "요약", "source_type": "vault", "file_path": "Projects/AI/요약.md"},
        },
        {
            "id": "vault-3",
            "document": "generic overview note",
            "distance": 0.36,
            "metadata": {"title": "web", "source_type": "vault", "file_path": "Projects/AI/web.md"},
        },
        {
            "id": "vault-4",
            "document": "generic principles note",
            "distance": 0.37,
            "metadata": {"title": "디자인 원칙", "source_type": "vault", "file_path": "Projects/AI/원칙.md"},
        },
        {
            "id": "paper-rag",
            "document": "agentic retrieval augmented generation taxonomy architectures evaluation and research directions",
            "distance": 0.41,
            "metadata": {"title": "SoK: Agentic Retrieval-Augmented Generation (RAG)", "source_type": "paper"},
        },
    ]
    db = DummyVectorDBWithEmptyLexical(records)
    searcher = RAGSearcher(DummyEmbedder(), db)

    results = searcher.search("RAG의 핵심 장단점은 무엇인가?", top_k=1, retrieval_mode="hybrid", alpha=0.7)

    assert len(results) == 1
    assert results[0].metadata["source_type"] == "paper"
    assert results[0].metadata["title"].startswith("SoK: Agentic Retrieval-Augmented Generation")


def test_search_hybrid_rescues_strong_lexical_only_hit():
    records = [
        {
            "id": "doc-semantic-a",
            "document": "generic machine learning overview",
            "distance": 0.12,
            "metadata": {
                "title": "머신러닝 개요",
                "source_type": "vault",
                "file_path": "Projects/ai/ml.md",
            },
        },
        {
            "id": "doc-semantic-b",
            "document": "another generic overview",
            "distance": 0.14,
            "metadata": {
                "title": "일반 개요",
                "source_type": "vault",
                "file_path": "Projects/ai/general.md",
            },
        },
        {
            "id": "doc-lexical",
            "document": "강화 학습 핵심 요약",
            "distance": 1.0,
            "metadata": {
                "title": "강화 학습 핵심 요약",
                "source_type": "vault",
                "file_path": "Projects/ai/강화학습.md",
            },
        },
    ]
    db = DummyVectorDBWithLexicalRescue(records)
    searcher = RAGSearcher(DummyEmbedder(), db, llm=None)

    results = searcher.search("강화 학습", top_k=3, source_type=None, retrieval_mode="hybrid", alpha=0.7)

    assert results[0].metadata["title"] == "강화 학습 핵심 요약"
    assert results[0].lexical_extras["hybrid_keyword_rescue_score"] > 0.0


def test_search_hybrid_diversifies_duplicate_lexical_chunks_across_parents(tmp_path):
    records = [
        {
            "id": "doc-1",
            "document": "강화 학습 핵심 요약 첫 번째 청크",
            "distance": 1.0,
            "metadata": {
                "title": "강화 학습 핵심 요약",
                "source_type": "vault",
                "file_path": "Projects/ai/강화학습.md",
                "parent_id": "rl-note",
                "document_scope_id": "vault:Projects/ai/강화학습.md::document:강화 학습 핵심 요약",
                "record_id": "rec-1",
            },
        },
        {
            "id": "doc-2",
            "document": "강화 학습 핵심 요약 두 번째 청크",
            "distance": 1.0,
            "metadata": {
                "title": "강화 학습 핵심 요약",
                "source_type": "vault",
                "file_path": "Projects/ai/강화학습.md",
                "parent_id": "rl-note",
                "document_scope_id": "vault:Projects/ai/강화학습.md::document:강화 학습 핵심 요약",
                "record_id": "rec-2",
            },
        },
        {
            "id": "doc-3",
            "document": "강화 학습 개요와 비교 포인트",
            "distance": 0.21,
            "metadata": {
                "title": "강화 학습 개요",
                "source_type": "vault",
                "file_path": "Projects/ai/강화학습-개요.md",
                "parent_id": "rl-overview",
                "record_id": "rec-3",
            },
        },
    ]
    sqlite_db = DummyFeatureSQLite(
        {
            "rec-1": {"importance_score": 0.75, "freshness_score": 0.7, "source_trust_score": 0.85},
            "rec-2": {"importance_score": 0.75, "freshness_score": 0.7, "source_trust_score": 0.85},
            "rec-3": {"importance_score": 0.72, "freshness_score": 0.7, "source_trust_score": 0.84},
        }
    )
    config = Config()
    config.set_nested("obsidian", "vault_path", str((tmp_path / "vault").resolve()))
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDBWithLexicalRescue(records), sqlite_db=sqlite_db, config=config)

    results = searcher.search("강화 학습", top_k=3, retrieval_mode="hybrid", alpha=0.7)

    assert {results[0].metadata["parent_id"], results[1].metadata["parent_id"]} == {"rl-note", "rl-overview"}
    assert results[2].metadata["parent_id"] == "rl-note"
    assert results[2].lexical_extras["duplicate_collapsed"] is True


def test_search_hybrid_applies_feature_boosts_to_rank_results():
    records = [
        {
            "id": "doc-low",
            "document": "rag retrieves relevant chunks before generation.",
            "distance": 0.3,
            "metadata": {
                "title": "Lower Priority",
                "source_type": "web",
                "url": "https://example.com/low",
                "record_id": "rec-low",
            },
        },
        {
            "id": "doc-high",
            "document": "rag retrieves relevant chunks before generation.",
            "distance": 0.32,
            "metadata": {
                "title": "Higher Priority",
                "source_type": "web",
                "url": "https://example.com/high",
                "record_id": "rec-high",
            },
        },
    ]
    sqlite_db = DummyFeatureSQLite(
        {
            "rec-high": {
                "importance_score": 1.0,
                "freshness_score": 1.0,
                "claim_density": 1.0,
                "support_doc_count": 10,
            }
        }
    )
    db = DummyVectorDB(records)
    searcher = RAGSearcher(DummyEmbedder(), db, sqlite_db=sqlite_db)

    results = searcher.search("rag generation", top_k=2, source_type=None, retrieval_mode="hybrid", alpha=0.6)

    assert len(results) == 2
    assert results[0].metadata["title"] == "Higher Priority"
    assert results[0].lexical_extras["feature_boost"] > 0
    assert results[0].lexical_extras["feature_normalized_support_doc_count"] == 1.0
    assert results[0].score > results[1].score
    assert results[0].lexical_extras["retrieval_adjusted_score"] == pytest.approx(results[0].score, abs=1e-6)


def test_search_hybrid_penalizes_contradictory_sources():
    records = [
        {
            "id": "doc-contradict",
            "document": "rag retrieves relevant chunks before generation.",
            "distance": 0.3,
            "metadata": {
                "title": "Contradictory Source",
                "source_type": "web",
                "url": "https://example.com/contradict",
                "record_id": "rec-contradict",
            },
        },
        {
            "id": "doc-clean",
            "document": "rag retrieves relevant chunks before generation.",
            "distance": 0.31,
            "metadata": {
                "title": "Clean Source",
                "source_type": "web",
                "url": "https://example.com/clean",
                "record_id": "rec-clean",
            },
        },
    ]
    sqlite_db = DummyFeatureSQLite(
        {
            "rec-contradict": {
                "importance_score": 0.9,
                "freshness_score": 0.9,
                "claim_density": 0.8,
                "contradiction_score": 1.0,
            },
            "rec-clean": {
                "importance_score": 0.9,
                "freshness_score": 0.9,
                "claim_density": 0.8,
                "contradiction_score": 0.0,
            },
        }
    )
    db = DummyVectorDB(records)
    searcher = RAGSearcher(DummyEmbedder(), db, sqlite_db=sqlite_db)

    results = searcher.search("rag generation", top_k=2, source_type=None, retrieval_mode="hybrid", alpha=0.6)

    assert len(results) == 2
    assert results[0].metadata["title"] == "Clean Source"
    assert results[1].lexical_extras["feature_penalty"] > 0
    assert results[1].lexical_extras["feature_contradiction"] == 1.0


def test_search_with_paper_memory_prefilter_scopes_candidate_papers(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=None, sqlite_db=DummyFeatureSQLite({}))
    calls: list[dict[str, object]] = []

    def _fake_search(self, query, top_k=5, source_type=None, retrieval_mode="hybrid", alpha=0.7, metadata_filter=None, **kwargs):
        _ = kwargs
        calls.append({"query": query, "source_type": source_type, "metadata_filter": dict(metadata_filter or {})})
        arxiv_id = str((metadata_filter or {}).get("arxiv_id") or "")
        if arxiv_id == "2603.13017":
            return [
                SearchResult(
                    document="paper-memory prefiltered result A",
                    metadata={"title": "Paper A", "source_type": "paper", "arxiv_id": "2603.13017"},
                    distance=0.1,
                    score=0.93,
                    semantic_score=0.92,
                    lexical_score=0.91,
                    retrieval_mode="hybrid",
                    lexical_extras={},
                    document_id="paper-a",
                )
            ]
        if arxiv_id == "2603.13018":
            return [
                SearchResult(
                    document="paper-memory prefiltered result B",
                    metadata={"title": "Paper B", "source_type": "paper", "arxiv_id": "2603.13018"},
                    distance=0.2,
                    score=0.72,
                    semantic_score=0.71,
                    lexical_score=0.7,
                    retrieval_mode="hybrid",
                    lexical_extras={},
                    document_id="paper-b",
                )
            ]
        return []

    monkeypatch.setattr(
        "knowledge_hub.ai.rag.resolve_paper_memory_prefilter",
        lambda sqlite_db, *, query, source_type, requested_mode, limit=3: {
            "requestedMode": requested_mode,
            "applied": True,
            "fallbackUsed": False,
            "matchedPaperIds": ["2603.13017", "2603.13018"],
            "matchedMemoryIds": ["memory-a", "memory-b"],
            "reason": "matched_cards",
        },
    )
    monkeypatch.setattr(RAGSearcher, "search", _fake_search)

    results, diagnostics = searcher._search_with_paper_memory_prefilter(
        query="memory retrieval",
        top_k=3,
        min_score=0.0,
        source_type="paper",
        retrieval_mode="hybrid",
        alpha=0.7,
        requested_mode="prefilter",
    )

    assert diagnostics["applied"] is True
    assert diagnostics["matchedPaperIds"] == ["2603.13017", "2603.13018"]
    assert [item.metadata["arxiv_id"] for item in results] == ["2603.13017", "2603.13018"]
    assert [call["metadata_filter"]["arxiv_id"] for call in calls] == ["2603.13017", "2603.13018"]


def test_search_with_paper_memory_prefilter_falls_back_when_not_applied(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=None, sqlite_db=DummyFeatureSQLite({}))
    calls: list[dict[str, object]] = []

    def _fake_search(self, query, top_k=5, source_type=None, retrieval_mode="hybrid", alpha=0.7, metadata_filter=None, **kwargs):
        _ = kwargs
        calls.append({"query": query, "source_type": source_type, "metadata_filter": dict(metadata_filter or {})})
        return [
            SearchResult(
                document="fallback result",
                metadata={"title": "Paper A", "source_type": "paper", "arxiv_id": "2603.13017"},
                distance=0.1,
                score=0.81,
                semantic_score=0.8,
                lexical_score=0.79,
                retrieval_mode="hybrid",
                lexical_extras={},
                document_id="paper-a",
            )
        ]

    monkeypatch.setattr(
        "knowledge_hub.ai.rag.resolve_paper_memory_prefilter",
        lambda sqlite_db, *, query, source_type, requested_mode, limit=3: {
            "requestedMode": requested_mode,
            "applied": False,
            "fallbackUsed": True,
            "matchedPaperIds": [],
            "matchedMemoryIds": [],
            "reason": "no_memory_hits",
        },
    )
    monkeypatch.setattr(RAGSearcher, "search", _fake_search)

    results, diagnostics = searcher._search_with_paper_memory_prefilter(
        query="memory retrieval",
        top_k=3,
        min_score=0.0,
        source_type="paper",
        retrieval_mode="hybrid",
        alpha=0.7,
        requested_mode="prefilter",
    )

    assert diagnostics["applied"] is False
    assert diagnostics["fallbackUsed"] is True
    assert diagnostics["reason"] == "no_memory_hits"
    assert len(results) == 1
    assert calls == [{"query": "memory retrieval", "source_type": "paper", "metadata_filter": {}}]


def test_generate_answer_includes_paper_memory_prefilter_payload(monkeypatch):
    llm = FakeLLM()
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=llm, sqlite_db=DummyFeatureSQLite({}))
    result_item = SearchResult(
        document="paper-memory answer evidence",
        metadata={"title": "Paper A", "source_type": "paper", "arxiv_id": "2603.13017"},
        distance=0.1,
        score=0.95,
        semantic_score=0.94,
        lexical_score=0.93,
        retrieval_mode="hybrid",
        lexical_extras={},
        document_id="paper-a",
    )

    monkeypatch.setattr(
        searcher,
        "_search_with_paper_memory_prefilter",
        lambda **kwargs: (
            [result_item],
            {
                "requestedMode": "prefilter",
                "applied": True,
                "fallbackUsed": False,
                "matchedPaperIds": ["2603.13017"],
                "matchedMemoryIds": ["memory-a"],
                "reason": "matched_cards",
            },
        ),
    )
    monkeypatch.setattr(searcher, "_apply_profile_and_cluster_scope", lambda filtered, query, top_k, apply_score_boosts=False: (filtered, [], None))
    monkeypatch.setattr(searcher, "_resolve_query_entities", lambda query: [])
    monkeypatch.setattr(searcher, "_collect_claim_context", lambda filtered: ([], [], [], []))
    monkeypatch.setattr(searcher, "_resolve_parent_context", lambda result, doc_cache: {"parent_id": "", "parent_label": "", "chunk_span": "", "text": result.document})
    monkeypatch.setattr(searcher, "_build_answer_context", lambda filtered, parent_ctx_by_result: "\n".join(item.document for item in filtered))
    monkeypatch.setattr(searcher, "_resolve_llm_for_request", lambda **kwargs: (llm, {"route": "fixed", "provider": "", "model": ""}, []))
    monkeypatch.setattr(searcher, "_build_answer_prompt", lambda **kwargs: "prompt")
    monkeypatch.setattr(searcher, "_verify_answer", lambda **kwargs: {"status": "verified", "supportedClaimCount": 1, "unsupportedClaimCount": 0, "uncertainClaimCount": 0, "conflictMentioned": True, "needsCaution": False, "warnings": []})
    monkeypatch.setattr(searcher, "_rewrite_answer", lambda **kwargs: (kwargs["answer"], {"attempted": False, "applied": False, "finalAnswerSource": "original", "warnings": []}))
    monkeypatch.setattr(searcher, "_apply_conservative_fallback_if_needed", lambda **kwargs: (kwargs["answer"], kwargs["rewrite_meta"], kwargs["verification"]))
    monkeypatch.setattr(searcher, "_answer_evidence_item", lambda result, parent_ctx_by_result: {"title": result.metadata["title"], "source_type": result.metadata["source_type"], "score": result.score, "semantic_score": result.semantic_score, "lexical_score": result.lexical_score})
    monkeypatch.setattr(searcher, "_record_answer_log", lambda **kwargs: None)

    payload = searcher.generate_answer("memory retrieval", top_k=1, source_type="paper", paper_memory_mode="prefilter")

    assert payload["status"] == "ok"
    assert payload["paperMemoryPrefilter"]["applied"] is True
    assert payload["paperMemoryPrefilter"]["matchedPaperIds"] == ["2603.13017"]


def test_generate_answer_ignores_paper_memory_prefilter_for_non_paper_source(monkeypatch):
    llm = FakeLLM()
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=llm, sqlite_db=DummyFeatureSQLite({}))
    result_item = SearchResult(
        document="web answer evidence",
        metadata={"title": "Web A", "source_type": "web"},
        distance=0.1,
        score=0.95,
        semantic_score=0.94,
        lexical_score=0.93,
        retrieval_mode="hybrid",
        lexical_extras={},
        document_id="web-a",
    )

    monkeypatch.setattr(
        searcher,
        "_search_with_paper_memory_prefilter",
        lambda **kwargs: (
            [result_item],
            {
                "requestedMode": "prefilter",
                "applied": False,
                "fallbackUsed": False,
                "matchedPaperIds": [],
                "matchedMemoryIds": [],
                "reason": "source_not_paper",
            },
        ),
    )
    monkeypatch.setattr(searcher, "_apply_profile_and_cluster_scope", lambda filtered, query, top_k, apply_score_boosts=False: (filtered, [], None))
    monkeypatch.setattr(searcher, "_resolve_query_entities", lambda query: [])
    monkeypatch.setattr(searcher, "_collect_claim_context", lambda filtered: ([], [], [], []))
    monkeypatch.setattr(searcher, "_resolve_parent_context", lambda result, doc_cache: {"parent_id": "", "parent_label": "", "chunk_span": "", "text": result.document})
    monkeypatch.setattr(searcher, "_build_answer_context", lambda filtered, parent_ctx_by_result: "\n".join(item.document for item in filtered))
    monkeypatch.setattr(searcher, "_resolve_llm_for_request", lambda **kwargs: (llm, {"route": "fixed", "provider": "", "model": ""}, []))
    monkeypatch.setattr(searcher, "_summarize_answer_signals", lambda evidence, contradicting_beliefs: {"preferred_sources": len(evidence)})
    monkeypatch.setattr(searcher, "_build_answer_prompt", lambda **kwargs: "prompt")
    monkeypatch.setattr(searcher, "_verify_answer", lambda **kwargs: {"status": "verified", "supportedClaimCount": 1, "unsupportedClaimCount": 0, "uncertainClaimCount": 0, "conflictMentioned": True, "needsCaution": False, "warnings": []})
    monkeypatch.setattr(searcher, "_rewrite_answer", lambda **kwargs: (kwargs["answer"], {"attempted": False, "applied": False, "finalAnswerSource": "original", "warnings": []}))
    monkeypatch.setattr(searcher, "_apply_conservative_fallback_if_needed", lambda **kwargs: (kwargs["answer"], kwargs["rewrite_meta"], kwargs["verification"]))
    monkeypatch.setattr(searcher, "_answer_evidence_item", lambda result, parent_ctx_by_result: {"title": result.metadata["title"], "source_type": result.metadata["source_type"], "score": result.score, "semantic_score": result.semantic_score, "lexical_score": result.lexical_score})
    monkeypatch.setattr(searcher, "_record_answer_log", lambda **kwargs: None)

    payload = searcher.generate_answer("memory retrieval", top_k=1, source_type="web", paper_memory_mode="prefilter")

    assert payload["status"] == "ok"
    assert payload["paperMemoryPrefilter"]["applied"] is False
    assert payload["paperMemoryPrefilter"]["reason"] == "source_not_paper"


def test_generate_answer_applies_single_paper_scope_and_builds_citations(monkeypatch):
    llm = FakeLLM()
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=llm, sqlite_db=DummyFeatureSQLite({}))
    paper_a = SearchResult(
        document="paper a abstract",
        metadata={"title": "Paper A", "source_type": "paper", "arxiv_id": "2501.00001"},
        distance=0.1,
        score=0.95,
        semantic_score=0.94,
        lexical_score=0.93,
        retrieval_mode="hybrid",
        lexical_extras={},
        document_id="paper-a",
    )
    paper_b = SearchResult(
        document="paper b abstract",
        metadata={"title": "Paper B", "source_type": "paper", "arxiv_id": "2501.00002"},
        distance=0.2,
        score=0.74,
        semantic_score=0.73,
        lexical_score=0.72,
        retrieval_mode="hybrid",
        lexical_extras={},
        document_id="paper-b",
    )

    monkeypatch.setattr(
        searcher,
        "_search_with_paper_memory_prefilter",
        lambda **kwargs: (
            [paper_a, paper_b],
            {
                "requestedMode": "prefilter",
                "applied": True,
                "fallbackUsed": False,
                "matchedPaperIds": ["2501.00001"],
                "matchedMemoryIds": ["memory-a"],
                "reason": "matched_cards",
            },
        ),
    )
    monkeypatch.setattr(searcher, "_apply_profile_and_cluster_scope", lambda filtered, query, top_k, apply_score_boosts=False: (filtered, [], None))
    monkeypatch.setattr(searcher, "_resolve_query_entities", lambda query: [])
    monkeypatch.setattr(searcher, "_collect_claim_context", lambda filtered: ([], [], [], []))
    monkeypatch.setattr(searcher, "_resolve_parent_context", lambda result, doc_cache: {"parent_id": "", "parent_label": "", "chunk_span": "", "text": result.document})
    monkeypatch.setattr(searcher, "_build_answer_context", lambda filtered, parent_ctx_by_result: "\n".join(item.document for item in filtered))
    monkeypatch.setattr(searcher, "_resolve_llm_for_request", lambda **kwargs: (llm, {"route": "fixed", "provider": "", "model": ""}, []))
    monkeypatch.setattr(searcher, "_summarize_answer_signals", lambda evidence, contradicting_beliefs: {"preferred_sources": len(evidence)})
    monkeypatch.setattr(searcher, "_build_answer_prompt", lambda **kwargs: "prompt")
    monkeypatch.setattr(searcher, "_verify_answer", lambda **kwargs: {"status": "verified", "supportedClaimCount": 1, "unsupportedClaimCount": 0, "uncertainClaimCount": 0, "conflictMentioned": True, "needsCaution": False, "warnings": []})
    monkeypatch.setattr(searcher, "_rewrite_answer", lambda **kwargs: (kwargs["answer"], {"attempted": False, "applied": False, "finalAnswerSource": "original", "warnings": []}))
    monkeypatch.setattr(searcher, "_apply_conservative_fallback_if_needed", lambda **kwargs: (kwargs["answer"], kwargs["rewrite_meta"], kwargs["verification"]))
    monkeypatch.setattr(searcher, "_record_answer_log", lambda **kwargs: None)

    payload = searcher.generate_answer("이 논문의 핵심 기여는?", top_k=5, source_type="paper", paper_memory_mode="prefilter")

    assert payload["paperAnswerScope"]["applied"] is True
    assert payload["paperAnswerScope"]["selectedPaperId"] == "2501.00001"
    assert [item["title"] for item in payload["sources"]] == ["Paper A"]
    assert payload["evidenceBudget"]["selectedCount"] == 1
    assert payload["citations"][0]["target"] == "2501.00001"
    assert payload["sources"][0]["citation_label"] == "S1"


def test_generate_answer_does_not_single_scope_broad_paper_definition_without_identity_signal(monkeypatch):
    llm = FakeLLM()
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=llm, sqlite_db=DummyFeatureSQLite({}))
    paper_a = SearchResult(
        document="paper a abstract",
        metadata={"title": "FlexPrefill", "source_type": "paper", "arxiv_id": "2501.00001"},
        distance=0.1,
        score=0.95,
        semantic_score=0.94,
        lexical_score=0.93,
        retrieval_mode="hybrid",
        lexical_extras={},
        document_id="paper-a",
    )
    paper_b = SearchResult(
        document="paper b abstract",
        metadata={"title": "ImageNet Classification with Deep Convolutional Neural Networks", "source_type": "paper", "arxiv_id": "2501.00002"},
        distance=0.2,
        score=0.74,
        semantic_score=0.73,
        lexical_score=0.72,
        retrieval_mode="hybrid",
        lexical_extras={},
        document_id="paper-b",
    )

    monkeypatch.setattr(
        searcher,
        "_search_with_paper_memory_prefilter",
        lambda **kwargs: (
            [paper_a, paper_b],
            {
                "requestedMode": "prefilter",
                "applied": True,
                "fallbackUsed": False,
                "matchedPaperIds": ["2501.00001"],
                "matchedMemoryIds": ["memory-a"],
                "reason": "matched_cards",
            },
        ),
    )
    monkeypatch.setattr(searcher, "_apply_profile_and_cluster_scope", lambda filtered, query, top_k, apply_score_boosts=False: (filtered, [], None))
    monkeypatch.setattr(searcher, "_resolve_query_entities", lambda query: [])
    monkeypatch.setattr(searcher, "_collect_claim_context", lambda filtered: ([], [], [], []))
    monkeypatch.setattr(searcher, "_resolve_parent_context", lambda result, doc_cache: {"parent_id": "", "parent_label": "", "chunk_span": "", "text": result.document})
    monkeypatch.setattr(searcher, "_build_answer_context", lambda filtered, parent_ctx_by_result: "\n".join(item.document for item in filtered))
    monkeypatch.setattr(searcher, "_resolve_llm_for_request", lambda **kwargs: (llm, {"route": "fixed", "provider": "", "model": ""}, []))
    monkeypatch.setattr(searcher, "_summarize_answer_signals", lambda evidence, contradicting_beliefs: {"preferred_sources": len(evidence)})
    monkeypatch.setattr(searcher, "_build_answer_prompt", lambda **kwargs: "prompt")
    monkeypatch.setattr(searcher, "_verify_answer", lambda **kwargs: {"status": "verified", "supportedClaimCount": 1, "unsupportedClaimCount": 0, "uncertainClaimCount": 0, "conflictMentioned": True, "needsCaution": False, "warnings": []})
    monkeypatch.setattr(searcher, "_rewrite_answer", lambda **kwargs: (kwargs["answer"], {"attempted": False, "applied": False, "finalAnswerSource": "original", "warnings": []}))
    monkeypatch.setattr(searcher, "_apply_conservative_fallback_if_needed", lambda **kwargs: (kwargs["answer"], kwargs["rewrite_meta"], kwargs["verification"]))
    monkeypatch.setattr(searcher, "_record_answer_log", lambda **kwargs: None)

    payload = searcher.generate_answer("CNN을 쉽게 설명해줘", top_k=5, source_type="paper", paper_memory_mode="prefilter")

    assert payload["paperFamily"] == "concept_explainer"
    assert payload["queryPlan"]["family"] == "concept_explainer"
    assert payload["queryFrame"]["family"] == "concept_explainer"
    assert payload["queryFrame"]["domain_key"] == "ai_papers"
    assert "CNN" in payload["queryPlan"]["expandedTerms"]
    assert "CNN" in payload["queryFrame"]["expanded_terms"]
    assert payload["familyRouteDiagnostics"]["answerMode"] == "representative_paper_explainer_beginner"
    assert payload["familyRouteDiagnostics"]["resolvedSourceScopeApplied"] is False
    assert payload["familyRouteDiagnostics"]["prefilterReason"] in {
        "canonical_entity_linking",
        "metadata_filter",
        "representative_candidate_narrowing",
        "source_scope",
    }
    assert isinstance(payload["familyRouteDiagnostics"]["canonicalEntitiesApplied"], list)
    assert payload["plannerFallback"]["attempted"] is False
    assert payload["answerSignals"]["answer_mode"] == "representative_paper_explainer_beginner"
    assert payload["answerSignals"]["representative_selection"]["reason"] in {
        "resolved_anchor_seed",
        "resolved_source_and_title_match",
        "strong_title_match",
        "retrieval_score_lead",
    }
    assert payload["retrievalObjectsAvailable"] == ["RawEvidenceUnit", "DocSummary", "SectionCard"]
    assert payload["retrievalObjectsUsed"] == ["RawEvidenceUnit"]
    assert payload["representativeRole"] == "anchor"
    assert payload["evidencePolicy"]["policyKey"] == "concept_explainer_policy"
    assert payload["evidencePolicy"]["singleScopeRequired"] is False
    assert payload["paperAnswerScope"]["applied"] is False
    assert payload["paperAnswerScope"]["reason"] in {"not_applicable", "selected_paper_missing"}
    assert [item["title"] for item in payload["sources"]] == ["FlexPrefill", "ImageNet Classification with Deep Convolutional Neural Networks"]


def test_search_distills_broad_paper_definition_query_into_concept_rescue_form(monkeypatch):
    records = [
        {
            "id": "paper-a",
            "document": "AlexNet introduced a deep convolutional neural network for ImageNet classification.",
            "metadata": {
                "title": "ImageNet Classification with Deep Convolutional Neural Networks",
                "source_type": "paper",
                "arxiv_id": "alexnet-2012",
            },
            "distance": 0.12,
        }
    ]
    sqlite_db = DummyFeatureSQLite(
        {},
        ontology_entities=[
            {
                "entity_id": "deep_convolutional_neural_networks",
                "entity_type": "concept",
                "canonical_name": "Deep Convolutional Neural Networks",
            }
        ],
        concept_papers={
            "deep_convolutional_neural_networks": [
                {"arxiv_id": "alexnet-2012", "title": "ImageNet Classification with Deep Convolutional Neural Networks"}
            ]
        },
    )
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records), llm=FakeLLM(), sqlite_db=sqlite_db)
    lexical_queries: list[str] = []

    monkeypatch.setattr("knowledge_hub.ai.retrieval_pipeline.expand_query_with_ontology", lambda searcher, query: [query])
    monkeypatch.setattr("knowledge_hub.ai.retrieval_pipeline.semantic_search", lambda *args, **kwargs: [])

    def _fake_lexical(searcher_obj, query_text, top_k=5, filter_dict=None):
        _ = (searcher_obj, top_k, filter_dict)
        lexical_queries.append(query_text)
        if query_text == "CNN":
            return [
                SearchResult(
                    document=records[0]["document"],
                    metadata=records[0]["metadata"],
                    distance=records[0]["distance"],
                    score=0.0,
                    semantic_score=0.0,
                    lexical_score=0.98,
                    retrieval_mode="keyword",
                    lexical_extras={"query": query_text},
                    document_id=records[0]["id"],
                )
            ]
        return []

    monkeypatch.setattr("knowledge_hub.ai.retrieval_pipeline.lexical_search", _fake_lexical)

    results = searcher.search("CNN을 쉽게 설명해줘", top_k=5, source_type="paper", retrieval_mode="hybrid")

    assert lexical_queries[0] == "CNN"
    assert "CNN" in lexical_queries
    assert "CNN을 쉽게 설명해줘" not in lexical_queries
    assert [item.metadata.get("title") for item in results] == ["ImageNet Classification with Deep Convolutional Neural Networks"]
    assert results[0].lexical_extras["ranking_signals"]["query_intent"] == "definition"


def test_generate_answer_limits_evidence_budget_for_paper_scope(monkeypatch):
    llm = FakeLLM()
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=llm, sqlite_db=DummyFeatureSQLite({}))
    items = [
        SearchResult(
            document=f"paper abstract {idx}",
            metadata={"title": f"Paper A-{idx}", "source_type": "paper", "arxiv_id": "2501.00001"},
            distance=0.1 + (idx * 0.01),
            score=0.95 - (idx * 0.01),
            semantic_score=0.94 - (idx * 0.01),
            lexical_score=0.93 - (idx * 0.01),
            retrieval_mode="hybrid",
            lexical_extras={},
            document_id=f"paper-a-{idx}",
        )
        for idx in range(6)
    ]
    monkeypatch.setattr(
        searcher,
        "_search_with_paper_memory_prefilter",
        lambda **kwargs: (
            items,
            {"requestedMode": "off", "applied": False, "fallbackUsed": False, "matchedPaperIds": [], "matchedMemoryIds": [], "reason": "disabled"},
        ),
    )
    monkeypatch.setattr(searcher, "_apply_profile_and_cluster_scope", lambda filtered, query, top_k, apply_score_boosts=False: (filtered, [], None))
    monkeypatch.setattr(searcher, "_resolve_query_entities", lambda query: [])
    monkeypatch.setattr(searcher, "_collect_claim_context", lambda filtered: ([], [], [], []))
    monkeypatch.setattr(searcher, "_resolve_parent_context", lambda result, doc_cache: {"parent_id": "", "parent_label": "", "chunk_span": "", "text": result.document})
    monkeypatch.setattr(searcher, "_build_answer_context", lambda filtered, parent_ctx_by_result: "\n".join(item.document for item in filtered))
    monkeypatch.setattr(searcher, "_resolve_llm_for_request", lambda **kwargs: (llm, {"route": "fixed", "provider": "", "model": ""}, []))
    monkeypatch.setattr(searcher, "_summarize_answer_signals", lambda evidence, contradicting_beliefs: {"preferred_sources": len(evidence)})
    monkeypatch.setattr(searcher, "_build_answer_prompt", lambda **kwargs: "prompt")
    monkeypatch.setattr(searcher, "_verify_answer", lambda **kwargs: {"status": "verified", "supportedClaimCount": 1, "unsupportedClaimCount": 0, "uncertainClaimCount": 0, "conflictMentioned": True, "needsCaution": False, "warnings": []})
    monkeypatch.setattr(searcher, "_rewrite_answer", lambda **kwargs: (kwargs["answer"], {"attempted": False, "applied": False, "finalAnswerSource": "original", "warnings": []}))
    monkeypatch.setattr(searcher, "_apply_conservative_fallback_if_needed", lambda **kwargs: (kwargs["answer"], kwargs["rewrite_meta"], kwargs["verification"]))
    monkeypatch.setattr(searcher, "_record_answer_log", lambda **kwargs: None)

    payload = searcher.generate_answer("논문 비교", top_k=8, source_type="paper")

    assert payload["queryFrame"]["family"] == "paper_compare"
    assert payload["evidencePolicy"]["policyKey"] == "paper_compare_policy"
    assert payload["evidencePolicy"]["requiresMultipleSources"] is True
    assert payload["evidenceBudget"]["maxSources"] == 4
    assert payload["evidenceBudget"]["selectedCount"] == 4
    assert len(payload["sources"]) == 4


def test_generate_answer_compare_uses_paper_card_fallback_for_missing_vector_papers(monkeypatch):
    llm = FakeLLM()
    sqlite_db = DummyCompareFallbackSQLite(
        {},
        paper_cards={
            "2005.11401": {
                "paper_id": "2005.11401",
                "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "method_core": "RAG combines dense passage retrieval with sequence-to-sequence generation.",
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
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=llm, sqlite_db=sqlite_db)

    monkeypatch.setattr(searcher, "_resolve_query_entities", lambda query: [])
    monkeypatch.setattr(searcher, "_collect_claim_context", lambda filtered: ([], [], [], []))
    monkeypatch.setattr(searcher, "_resolve_parent_context", lambda result, doc_cache: {"parent_id": "", "parent_label": "", "chunk_span": "", "text": result.document})
    monkeypatch.setattr(searcher, "_resolve_llm_for_request", lambda **kwargs: (llm, {"route": "fixed", "provider": "", "model": ""}, []))
    monkeypatch.setattr(searcher, "_summarize_answer_signals", lambda evidence, contradicting_beliefs: {"preferred_sources": len(evidence)})
    monkeypatch.setattr(searcher, "_build_answer_prompt", lambda **kwargs: "prompt")
    monkeypatch.setattr(searcher, "_verify_answer", lambda **kwargs: {"status": "verified", "supportedClaimCount": 1, "unsupportedClaimCount": 0, "uncertainClaimCount": 0, "conflictMentioned": False, "needsCaution": False, "warnings": []})
    monkeypatch.setattr(searcher, "_rewrite_answer", lambda **kwargs: (kwargs["answer"], {"attempted": False, "applied": False, "finalAnswerSource": "original", "warnings": []}))
    monkeypatch.setattr(searcher, "_apply_conservative_fallback_if_needed", lambda **kwargs: (kwargs["answer"], kwargs["rewrite_meta"], kwargs["verification"]))
    monkeypatch.setattr(searcher, "_record_answer_log", lambda **kwargs: None)

    payload = searcher.generate_answer(
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks와 Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering을 비교해줘",
        top_k=6,
        source_type="paper",
    )

    assert payload["queryFrame"]["family"] == "paper_compare"
    assert [item.get("paper_id") or item.get("arxiv_id") for item in payload["sources"][:2]] == ["2005.11401", "2007.01282"]
    assert payload["paperAnswerScope"]["candidatePaperIds"][:2] == ["2005.11401", "2007.01282"]
    assert payload["familyRouteDiagnostics"]["runtimeUsed"] == "legacy"


def test_generate_answer_uses_concept_first_structured_context_for_paper_definition_queries(monkeypatch):
    llm = FakeLLM()
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=llm, sqlite_db=DummyFeatureSQLite({}))
    representative = SearchResult(
        document=(
            "Title: Attention Is All You Need\n"
            "## 한줄 요약\n\n"
            "The Transformer uses only attention mechanisms, removing recurrence.\n\n"
            "## 핵심 아이디어\n\n"
            "Use self-attention to model token relations in parallel.\n\n"
            "## 방법\n\n"
            "An encoder-decoder stack built entirely from attention and feed-forward layers.\n"
        ),
        metadata={"title": "Attention Is All You Need", "source_type": "paper", "arxiv_id": "1706.03762"},
        distance=0.1,
        score=0.95,
        semantic_score=0.94,
        lexical_score=0.93,
        retrieval_mode="hybrid",
        lexical_extras={},
        document_id="paper-a",
    )
    support = SearchResult(
        document=(
            "Title: An Image is Worth 16x16 Words\n"
            "## 한줄 요약\n\n"
            "Apply the transformer encoder to image patches.\n"
        ),
        metadata={"title": "An Image is Worth 16x16 Words", "source_type": "paper", "arxiv_id": "2010.11929"},
        distance=0.2,
        score=0.74,
        semantic_score=0.73,
        lexical_score=0.72,
        retrieval_mode="hybrid",
        lexical_extras={},
        document_id="paper-b",
    )

    monkeypatch.setattr(
        searcher,
        "_search_with_paper_memory_prefilter",
        lambda **kwargs: (
            [representative, support],
            {"requestedMode": "off", "applied": False, "fallbackUsed": False, "matchedPaperIds": [], "matchedMemoryIds": [], "reason": "disabled"},
        ),
    )
    monkeypatch.setattr(searcher, "_apply_profile_and_cluster_scope", lambda filtered, query, top_k, apply_score_boosts=False: (filtered, [], None))
    monkeypatch.setattr(searcher, "_resolve_query_entities", lambda query: [])
    monkeypatch.setattr(searcher, "_collect_claim_context", lambda filtered: ([], [], [], []))
    monkeypatch.setattr(searcher, "_resolve_parent_context", lambda result, doc_cache: {"parent_id": "", "parent_label": "", "chunk_span": "", "text": result.document})
    monkeypatch.setattr(searcher, "_resolve_llm_for_request", lambda **kwargs: (llm, {"route": "fixed", "provider": "", "model": ""}, []))
    monkeypatch.setattr(searcher, "_verify_answer", lambda **kwargs: {"status": "verified", "supportedClaimCount": 1, "unsupportedClaimCount": 0, "uncertainClaimCount": 0, "conflictMentioned": False, "needsCaution": False, "warnings": []})
    monkeypatch.setattr(searcher, "_rewrite_answer", lambda **kwargs: (kwargs["answer"], {"attempted": False, "applied": False, "finalAnswerSource": "original", "warnings": []}))
    monkeypatch.setattr(searcher, "_apply_conservative_fallback_if_needed", lambda **kwargs: (kwargs["answer"], kwargs["rewrite_meta"], kwargs["verification"]))
    monkeypatch.setattr(searcher, "_record_answer_log", lambda **kwargs: None)

    payload = searcher.generate_answer("Transformer의 핵심 아이디어를 설명해줘", top_k=6, source_type="paper")

    assert payload["status"] == "ok"
    assert payload["paperFamily"] == "concept_explainer"
    assert payload["queryPlan"]["family"] == "concept_explainer"
    assert payload["queryFrame"]["family"] == "concept_explainer"
    assert payload["representativePaper"]["paperId"] == "1706.03762"
    assert payload["evidencePolicy"]["policyKey"] == "concept_explainer_policy"
    assert payload["familyRouteDiagnostics"]["paperFamily"] == "concept_explainer"
    assert payload["familyRouteDiagnostics"]["runtimeUsed"] == "legacy"
    assert payload["familyRouteDiagnostics"]["resolvedSourceScopeApplied"] is False
    assert isinstance(payload["familyRouteDiagnostics"]["canonicalEntitiesApplied"], list)
    assert payload["plannerFallback"]["attempted"] is False
    assert "한줄 정의 -> 작동 원리 -> 왜 중요한지 -> 대표 사례" in (llm.last_prompt or "")
    assert "audience=general" in (llm.last_prompt or "")
    assert "Audience: general" in (llm.last_context or "")
    assert "=== Concept Core Evidence ===" in (llm.last_context or "")
    assert "=== Representative Paper Example 1 ===" in (llm.last_context or "")
    assert (llm.last_context or "").index("=== Concept Core Evidence ===") < (llm.last_context or "").index("=== Representative Paper Example 1 ===")
    assert "mechanism_summary=Use self-attention to model token relations in parallel." in (llm.last_context or "")
    assert payload["paperAnswerScope"]["applied"] is False
    assert payload["answerSignals"]["supporting_paper_count"] == 1
    assert "support_note=Apply the transformer encoder to image patches." in (llm.last_context or "")


def test_generate_answer_returns_conservative_fallback_when_initial_generation_times_out(monkeypatch):
    llm = FailingLLM(TimeoutError("timed out"))
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=llm, sqlite_db=DummyFeatureSQLite({}))
    paper = SearchResult(
        document="paper a abstract",
        metadata={"title": "Paper A", "source_type": "paper", "arxiv_id": "2501.00001"},
        distance=0.1,
        score=0.95,
        semantic_score=0.94,
        lexical_score=0.93,
        retrieval_mode="hybrid",
        lexical_extras={},
        document_id="paper-a",
    )
    monkeypatch.setattr(
        searcher,
        "_search_with_paper_memory_prefilter",
        lambda **kwargs: (
            [paper],
            {"requestedMode": "prefilter", "applied": True, "fallbackUsed": False, "matchedPaperIds": ["2501.00001"], "matchedMemoryIds": ["memory-a"], "reason": "matched_cards"},
        ),
    )
    monkeypatch.setattr(searcher, "_apply_profile_and_cluster_scope", lambda filtered, query, top_k, apply_score_boosts=False: (filtered, [], None))
    monkeypatch.setattr(searcher, "_resolve_query_entities", lambda query: [])
    monkeypatch.setattr(searcher, "_collect_claim_context", lambda filtered: ([], [], [], []))
    monkeypatch.setattr(searcher, "_resolve_parent_context", lambda result, doc_cache: {"parent_id": "", "parent_label": "", "chunk_span": "", "text": result.document})
    monkeypatch.setattr(searcher, "_build_answer_context", lambda filtered, parent_ctx_by_result: "\n".join(item.document for item in filtered))
    monkeypatch.setattr(searcher, "_resolve_llm_for_request", lambda **kwargs: (llm, {"route": "local", "provider": "ollama", "model": "qwen3:14b"}, []))
    monkeypatch.setattr(searcher, "_summarize_answer_signals", lambda evidence, contradicting_beliefs: {"preferred_sources": len(evidence)})
    monkeypatch.setattr(searcher, "_build_answer_prompt", lambda **kwargs: "prompt")
    monkeypatch.setattr(searcher, "_record_answer_log", lambda **kwargs: None)

    payload = searcher.generate_answer("이 논문의 핵심 기여는?", top_k=3, source_type="paper", paper_memory_mode="prefilter")

    assert payload["status"] == "ok"
    assert payload["answerGeneration"]["fallbackUsed"] is True
    assert payload["answerGeneration"]["errorType"] == "TimeoutError"
    assert payload["answerRewrite"]["finalAnswerSource"] == "generation_fallback"
    assert payload["answerVerification"]["status"] == "skipped"
    assert payload["citations"][0]["target"] == "2501.00001"
    assert "보수적으로 정리합니다" in payload["answer"]
    assert any("answer generation fallback applied" in warning for warning in payload["warnings"])


def test_generate_answer_returns_conservative_fallback_when_no_llm_route_is_available(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=None, sqlite_db=DummyFeatureSQLite({}))
    paper = SearchResult(
        document="paper a abstract",
        metadata={"title": "Paper A", "source_type": "paper", "arxiv_id": "2501.00001"},
        distance=0.1,
        score=0.95,
        semantic_score=0.94,
        lexical_score=0.93,
        retrieval_mode="hybrid",
        lexical_extras={},
        document_id="paper-a",
    )
    monkeypatch.setattr(
        searcher,
        "_search_with_paper_memory_prefilter",
        lambda **kwargs: (
            [paper],
            {"requestedMode": "prefilter", "applied": True, "fallbackUsed": False, "matchedPaperIds": ["2501.00001"], "matchedMemoryIds": ["memory-a"], "reason": "matched_cards"},
        ),
    )
    monkeypatch.setattr(searcher, "_apply_profile_and_cluster_scope", lambda filtered, query, top_k, apply_score_boosts=False: (filtered, [], None))
    monkeypatch.setattr(searcher, "_resolve_query_entities", lambda query: [])
    monkeypatch.setattr(searcher, "_collect_claim_context", lambda filtered: ([], [], [], []))
    monkeypatch.setattr(searcher, "_resolve_parent_context", lambda result, doc_cache: {"parent_id": "", "parent_label": "", "chunk_span": "", "text": result.document})
    monkeypatch.setattr(searcher, "_build_answer_context", lambda filtered, parent_ctx_by_result: "\n".join(item.document for item in filtered))
    monkeypatch.setattr(
        searcher,
        "_resolve_llm_for_request",
        lambda **kwargs: (
            None,
            {"route": "fallback-only", "provider": "", "model": "", "reasons": ["routing_failed_no_available_llm"], "fallbackUsed": True},
            ["route unavailable (ollama/qwen3:14b): runtime unavailable"],
        ),
    )
    monkeypatch.setattr(searcher, "_summarize_answer_signals", lambda evidence, contradicting_beliefs: {"preferred_sources": len(evidence)})
    monkeypatch.setattr(searcher, "_build_answer_prompt", lambda **kwargs: "prompt")
    monkeypatch.setattr(searcher, "_record_answer_log", lambda **kwargs: None)

    payload = searcher.generate_answer("이 논문의 핵심 기여는?", top_k=3, source_type="paper", paper_memory_mode="prefilter")

    assert payload["status"] == "ok"
    assert payload["answerGeneration"]["fallbackUsed"] is True
    assert payload["answerGeneration"]["errorType"] == "RuntimeError"
    assert payload["router"]["selected"]["route"] == "fallback-only"
    assert any("route unavailable" in warning for warning in payload["warnings"])


def test_stream_answer_returns_conservative_fallback_when_initial_stream_generation_times_out(monkeypatch):
    llm = FailingLLM(TimeoutError("timed out"))
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=llm, sqlite_db=DummyFeatureSQLite({}))
    paper = SearchResult(
        document="paper a abstract",
        metadata={"title": "Paper A", "source_type": "paper", "arxiv_id": "2501.00001"},
        distance=0.1,
        score=0.95,
        semantic_score=0.94,
        lexical_score=0.93,
        retrieval_mode="hybrid",
        lexical_extras={},
        document_id="paper-a",
    )
    monkeypatch.setattr(
        searcher,
        "_search_with_paper_memory_prefilter",
        lambda **kwargs: (
            [paper],
            {"requestedMode": "prefilter", "applied": True, "fallbackUsed": False, "matchedPaperIds": ["2501.00001"], "matchedMemoryIds": ["memory-a"], "reason": "matched_cards"},
        ),
    )
    monkeypatch.setattr(searcher, "_resolve_parent_context", lambda result, doc_cache: {"parent_id": "", "parent_label": "", "chunk_span": "", "text": result.document})
    monkeypatch.setattr(searcher, "_build_answer_context", lambda filtered, parent_ctx_by_result: "\n".join(item.document for item in filtered))
    monkeypatch.setattr(searcher, "_resolve_llm_for_request", lambda **kwargs: (llm, {"route": "local", "provider": "ollama", "model": "qwen3:14b"}, []))
    monkeypatch.setattr(searcher, "_record_answer_log", lambda **kwargs: None)

    output = "".join(searcher.stream_answer("이 논문의 핵심 기여는?", top_k=3, source_type="paper", paper_memory_mode="prefilter"))

    assert "보수적으로 정리합니다" in output
    assert "Paper A" in output


def test_search_applies_bounded_graph_candidate_boost_for_entity_heavy_query(monkeypatch):
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

    records = [
        {
            "id": "generic",
            "document": "general implementation note",
            "distance": 0.12,
            "metadata": {"title": "Implementation Notes", "source_type": "vault", "file_path": "Projects/AI/Implementation Notes.md"},
        },
        {
            "id": "rag-note",
            "document": "rag transformer attention note",
            "distance": 0.16,
            "metadata": {"title": "RAG Transformer Note", "source_type": "vault", "file_path": "Projects/AI/RAG Transformer Note.md", "related_concepts": ["RAG"]},
        },
    ]
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records), llm=FakeLLM(), sqlite_db=DummyFeatureSQLite({}))
    monkeypatch.setattr("knowledge_hub.ai.rag.analyze_graph_query", lambda query, repository: _GraphAnalysis())

    results = searcher.search("RAG transformer attention", top_k=2, retrieval_mode="semantic", use_ontology_expansion=False)

    assert results[0].metadata["title"] == "RAG Transformer Note"
    ranking = dict(results[0].lexical_extras or {}).get("ranking_signals") or {}
    assert ranking["graph_candidate_boost"] > 0.0
    assert ranking["graph_candidate_reduction_applied"] is True
    assert "RAG" in ranking["graph_candidate_matches"]


def test_search_prefers_ok_quality_when_base_scores_are_close(tmp_path):
    records = [
        {
            "id": "doc-review",
            "document": "rag retrieves relevant chunks before generation.",
            "distance": 0.3,
            "metadata": {
                "title": "Needs Review Note",
                "source_type": "vault",
                "file_path": "LearningHub/ai/source_review.md",
                "record_id": "rec-review",
            },
        },
        {
            "id": "doc-ok",
            "document": "rag retrieves relevant chunks before generation.",
            "distance": 0.31,
            "metadata": {
                "title": "OK Note",
                "source_type": "vault",
                "file_path": "LearningHub/ai/source_ok.md",
                "record_id": "rec-ok",
            },
        },
    ]
    sqlite_db = DummyFeatureSQLite(
        {
            "rec-review": {"importance_score": 0.7, "freshness_score": 0.7, "source_trust_score": 0.82},
            "rec-ok": {"importance_score": 0.7, "freshness_score": 0.7, "source_trust_score": 0.82},
        },
        ko_note_items={
            str((tmp_path / "vault" / "LearningHub/ai/source_review.md").resolve()): {"payload_json": {"quality": {"flag": "needs_review"}}},
            str((tmp_path / "vault" / "LearningHub/ai/source_ok.md").resolve()): {"payload_json": {"quality": {"flag": "ok"}}},
        },
    )
    config = Config()
    config.set_nested("obsidian", "vault_path", str((tmp_path / "vault").resolve()))
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records), sqlite_db=sqlite_db, config=config)

    results = searcher.search("rag generation", top_k=2, retrieval_mode="hybrid", alpha=0.6)

    assert len(results) == 2
    assert results[0].metadata["title"] == "OK Note"
    assert results[0].lexical_extras["quality_flag"] == "ok"
    assert results[1].lexical_extras["quality_flag"] == "needs_review"
    assert results[0].lexical_extras["quality_boost"] > results[1].lexical_extras["quality_boost"]


def test_search_adds_small_reference_prior_for_specialist_sources():
    records = [
        {
            "id": "doc-generic",
            "document": "rag glossary note",
            "distance": 0.3,
            "metadata": {
                "title": "Generic Web Note",
                "source_type": "web",
                "url": "https://example.com/generic",
                "record_id": "rec-generic",
            },
        },
        {
            "id": "doc-specialist",
            "document": "rag glossary note",
            "distance": 0.305,
            "metadata": {
                "title": "Specialist Reference",
                "source_type": "web",
                "url": "https://example.com/specialist",
                "record_id": "rec-specialist",
            },
        },
    ]
    sqlite_db = DummyFeatureSQLite(
        {
            "rec-generic": {"importance_score": 0.75, "freshness_score": 0.6, "source_trust_score": 0.82},
            "rec-specialist": {
                "importance_score": 0.78,
                "freshness_score": 0.6,
                "source_trust_score": 0.82,
                "payload_json": {
                    "referenceRole": "glossary_reference",
                    "referenceTier": "specialist",
                    "referencePriorBoost": 0.075,
                },
            },
        },
    )
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records), sqlite_db=sqlite_db)

    results = searcher.search("rag glossary", top_k=2, retrieval_mode="hybrid", alpha=0.6)

    assert results[0].metadata["title"] == "Specialist Reference"
    assert results[0].lexical_extras["reference_tier"] == "specialist"
    assert results[0].lexical_extras["reference_role"] == "glossary_reference"
    assert 0.0 < results[0].lexical_extras["reference_prior_boost"] <= 0.08


def test_search_adds_ontology_entity_overlap_signal():
    records = [
        {
            "id": "doc-rag",
            "document": "retrieval augmented generation note",
            "distance": 0.26,
            "metadata": {
                "title": "RAG Concepts",
                "source_type": "concept",
                "canonical_name": "Retrieval Augmented Generation",
                "aliases": ["RAG", "retrieval augmented generation"],
                "entity_id": "concept_rag",
                "related_concepts": ["retrieval augmented generation", "grounded retrieval"],
            },
        },
        {
            "id": "doc-generic",
            "document": "generic concept note",
            "distance": 0.27,
            "metadata": {
                "title": "Generic Overview",
                "source_type": "concept",
                "canonical_name": "Generic Overview",
                "aliases": ["approach", "overview"],
                "entity_id": "concept_generic",
            },
        },
    ]
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records))

    results = searcher.search("RAG", top_k=2, retrieval_mode="hybrid", alpha=0.6)

    assert results[0].metadata["title"] == "RAG Concepts"
    assert results[0].lexical_extras["ranking_signals"]["ontology_entity_overlap_boost"] > 0.0
    assert results[0].lexical_extras["ranking_signals"]["ontology_entity_exact_match_boost"] >= 0.0
    assert any(item["name"] == "ontology_entity_overlap_boost" for item in results[0].lexical_extras["top_ranking_signals"])


def test_search_applies_cluster_proximity_signal_from_topology(monkeypatch):
    records = [
        {
            "id": "doc-a",
            "document": "rag cluster alpha",
            "distance": 0.24,
            "metadata": {
                "title": "Cluster Alpha",
                "source_type": "vault",
                "file_path": "AI/alpha.md",
                "record_id": "rec-a",
            },
        },
        {
            "id": "doc-b",
            "document": "rag cluster beta",
            "distance": 0.25,
            "metadata": {
                "title": "Cluster Beta",
                "source_type": "vault",
                "file_path": "AI/beta.md",
                "record_id": "rec-b",
            },
        },
        {
            "id": "doc-c",
            "document": "rag cluster gamma",
            "distance": 0.22,
            "metadata": {
                "title": "Cluster Gamma",
                "source_type": "vault",
                "file_path": "AI/gamma.md",
                "record_id": "rec-c",
            },
        },
    ]
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records))
    monkeypatch.setattr(
        searcher,
        "_load_topology_index",
        lambda: {
            "cacheKey": "fake",
            "snapshotPath": "fake",
            "nodesByPath": {
                "AI/alpha.md": {"clusterId": "c1"},
                "AI/beta.md": {"clusterId": "c1"},
                "AI/gamma.md": {"clusterId": "c2"},
            },
            "clustersById": {
                "c1": {"label": "RAG Cluster", "size": 2, "representativeNoteId": "AI/alpha.md"},
                "c2": {"label": "Other Cluster", "size": 1, "representativeNoteId": "AI/gamma.md"},
            },
        },
    )

    results = searcher.search("rag cluster", top_k=3, retrieval_mode="hybrid", alpha=0.6)

    c1_result = next(result for result in results if result.metadata.get("cluster_id") == "c1")
    assert c1_result.lexical_extras["ranking_signals"]["cluster_selected"] is True
    assert c1_result.lexical_extras["ranking_signals"]["cluster_proximity_boost"] > 0.0
    assert any(item["name"] == "cluster_proximity_boost" for item in c1_result.lexical_extras["top_ranking_signals"])
    assert any(result.metadata.get("cluster_id") == "c2" for result in results)


def test_apply_profile_and_cluster_scope_sorts_cluster_scoped_results_without_name_error(monkeypatch):
    records = [
        {
            "id": "doc-a",
            "document": "rag cluster alpha",
            "distance": 0.24,
            "metadata": {
                "title": "Cluster Alpha",
                "source_type": "vault",
                "file_path": "AI/alpha.md",
                "record_id": "rec-a",
            },
        },
        {
            "id": "doc-b",
            "document": "rag cluster beta",
            "distance": 0.25,
            "metadata": {
                "title": "Cluster Beta",
                "source_type": "vault",
                "file_path": "AI/beta.md",
                "record_id": "rec-b",
            },
        },
        {
            "id": "doc-c",
            "document": "rag cluster gamma",
            "distance": 0.22,
            "metadata": {
                "title": "Cluster Gamma",
                "source_type": "vault",
                "file_path": "AI/gamma.md",
                "record_id": "rec-c",
            },
        },
        {
            "id": "doc-d",
            "document": "rag cluster delta",
            "distance": 0.6,
            "metadata": {
                "title": "Cluster Delta",
                "source_type": "vault",
                "file_path": "AI/delta.md",
                "record_id": "rec-d",
            },
        },
    ]
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records))
    monkeypatch.setattr(searcher, "_get_active_profile", lambda: None)
    monkeypatch.setattr(
        searcher,
        "_load_topology_index",
        lambda: {
            "cacheKey": "fake",
            "snapshotPath": "fake",
            "nodesByPath": {
                "AI/alpha.md": {"clusterId": "c1"},
                "AI/beta.md": {"clusterId": "c1"},
                "AI/gamma.md": {"clusterId": "c2"},
                "AI/delta.md": {"clusterId": "c4"},
            },
            "clustersById": {
                "c1": {"label": "RAG Cluster", "size": 2, "representativeNoteId": "AI/alpha.md"},
                "c2": {"label": "Other Cluster", "size": 1, "representativeNoteId": "AI/gamma.md"},
                "c4": {"label": "Spillover Cluster", "size": 1, "representativeNoteId": "AI/delta.md"},
            },
        },
    )

    results = searcher.search("rag cluster", top_k=4, retrieval_mode="hybrid", alpha=0.6)
    scoped_results, related_clusters, profile = searcher._apply_profile_and_cluster_scope(
        results,
        query="rag cluster",
        top_k=4,
    )

    assert profile is None
    assert all(str((result.metadata or {}).get("cluster_id", "")).strip() != "c4" for result in scoped_results[:3])
    assert str((scoped_results[-1].metadata or {}).get("cluster_id", "")).strip() == "c4"
    assert related_clusters[0]["cluster_id"] == "c1"


def test_search_normalizes_note_source_and_prefers_exact_title_match_over_generic_concept():
    records = [
        {
            "id": "doc-concept",
            "document": "transformer explanation and high-level background.",
            "distance": 0.18,
            "metadata": {
                "title": "Transformer",
                "source_type": "concept",
                "file_path": "Projects/AI/AI_Papers/Concepts/Transformer.md",
                "record_id": "rec-concept",
            },
        },
        {
            "id": "doc-note",
            "document": "transformer architecture details and implementation notes.",
            "distance": 0.2,
            "metadata": {
                "title": "Transformer Architecture",
                "source_type": "note",
                "file_path": "Projects/AI/Notes/Transformer Architecture.md",
                "record_id": "rec-note",
            },
        },
    ]
    sqlite_db = DummyFeatureSQLite(
        {
            "rec-concept": {"importance_score": 0.9, "freshness_score": 0.7, "source_trust_score": 0.8},
            "rec-note": {"importance_score": 0.8, "freshness_score": 0.7, "source_trust_score": 0.82},
        }
    )
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records), sqlite_db=sqlite_db)

    results = searcher.search("Transformer Architecture", top_k=2, retrieval_mode="hybrid", alpha=0.7)

    assert results[0].metadata["title"] == "Transformer Architecture"
    assert results[0].metadata["source_type"] == "vault"
    assert results[0].lexical_extras["normalized_source_type"] == "vault"
    assert results[0].lexical_extras["ranking_signals"]["exact_title_match_boost"] > 0.0
    assert results[1].lexical_extras["ranking_signals"]["generic_concept_penalty"] >= 0.0


def test_search_penalizes_duplicate_parent_exposure(tmp_path):
    records = [
        {
            "id": "doc-1",
            "document": "rag implementation details chunk one.",
            "distance": 0.2,
            "metadata": {
                "title": "RAG Implementation",
                "source_type": "vault",
                "file_path": "Projects/AI/RAG Implementation.md",
                "parent_id": "rag-impl",
                "record_id": "rec-1",
            },
        },
        {
            "id": "doc-2",
            "document": "rag implementation details chunk two.",
            "distance": 0.205,
            "metadata": {
                "title": "RAG Implementation",
                "source_type": "vault",
                "file_path": "Projects/AI/RAG Implementation.md",
                "parent_id": "rag-impl",
                "record_id": "rec-2",
            },
        },
        {
            "id": "doc-3",
            "document": "rag architecture tradeoffs overview.",
            "distance": 0.215,
            "metadata": {
                "title": "RAG Architecture Overview",
                "source_type": "vault",
                "file_path": "Projects/AI/RAG Architecture Overview.md",
                "parent_id": "rag-arch",
                "record_id": "rec-3",
            },
        },
    ]
    sqlite_db = DummyFeatureSQLite(
        {
            "rec-1": {"importance_score": 0.75, "freshness_score": 0.7, "source_trust_score": 0.85},
            "rec-2": {"importance_score": 0.75, "freshness_score": 0.7, "source_trust_score": 0.85},
            "rec-3": {"importance_score": 0.72, "freshness_score": 0.7, "source_trust_score": 0.84},
        }
    )
    config = Config()
    config.set_nested("obsidian", "vault_path", str((tmp_path / "vault").resolve()))
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records), sqlite_db=sqlite_db, config=config)

    results = searcher.search("rag implementation", top_k=3, retrieval_mode="hybrid", alpha=0.7)

    assert results[0].metadata["title"] == "RAG Implementation"
    assert results[1].metadata["title"] == "RAG Architecture Overview"
    assert results[2].lexical_extras["duplicate_collapsed"] is True
    assert results[2].lexical_extras["ranking_signals"]["duplicate_exposure_penalty"] > 0.0


def test_search_contradiction_penalty_outweighs_quality_boost(tmp_path):
    records = [
        {
            "id": "doc-contradict",
            "document": "rag retrieves relevant chunks before generation.",
            "distance": 0.3,
            "metadata": {
                "title": "Contradictory but high quality",
                "source_type": "vault",
                "file_path": "LearningHub/ai/contradict.md",
                "record_id": "rec-contradict",
            },
        },
        {
            "id": "doc-clean",
            "document": "rag retrieves relevant chunks before generation.",
            "distance": 0.31,
            "metadata": {
                "title": "Clean lower quality",
                "source_type": "vault",
                "file_path": "LearningHub/ai/clean.md",
                "record_id": "rec-clean",
            },
        },
    ]
    sqlite_db = DummyFeatureSQLite(
        {
            "rec-contradict": {
                "importance_score": 0.9,
                "freshness_score": 0.9,
                "claim_density": 0.8,
                "source_trust_score": 0.9,
                "contradiction_score": 1.0,
            },
            "rec-clean": {
                "importance_score": 0.85,
                "freshness_score": 0.85,
                "claim_density": 0.7,
                "source_trust_score": 0.85,
                "contradiction_score": 0.0,
            },
        },
        ko_note_items={
            str((tmp_path / "vault" / "LearningHub/ai/contradict.md").resolve()): {"payload_json": {"quality": {"flag": "ok"}}},
            str((tmp_path / "vault" / "LearningHub/ai/clean.md").resolve()): {"payload_json": {"quality": {"flag": "needs_review"}}},
        },
    )
    config = Config()
    config.set_nested("obsidian", "vault_path", str((tmp_path / "vault").resolve()))
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records), sqlite_db=sqlite_db, config=config)

    results = searcher.search("rag generation", top_k=2, retrieval_mode="hybrid", alpha=0.6)

    assert results[0].metadata["title"] == "Clean lower quality"
    assert results[1].lexical_extras["feature_penalty"] >= 0.15
    assert results[1].lexical_extras["quality_flag"] == "ok"


def test_generate_answer_exposes_ranking_signals_in_evidence():
    llm = FakeLLM()
    records = [
        {
            "id": "doc-signal",
            "document": "rag retrieves relevant chunks before generation.",
            "distance": 0.2,
            "metadata": {
                "title": "Signal Note",
                "source_type": "web",
                "url": "https://example.com/signal",
                "record_id": "rec-signal",
            },
        }
    ]
    sqlite_db = DummyFeatureSQLite(
        {
            "rec-signal": {
                "importance_score": 0.9,
                "freshness_score": 0.8,
                "source_trust_score": 0.94,
                "payload_json": {
                    "referenceRole": "standard_reference",
                    "referenceTier": "specialist",
                    "referencePriorBoost": 0.08,
                },
            }
        }
    )
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records), llm=llm, sqlite_db=sqlite_db)

    answer = searcher.generate_answer("rag", top_k=1, retrieval_mode="hybrid")

    assert answer["sources"][0]["quality_flag"] == "unscored"
    assert answer["sources"][0]["reference_tier"] == "specialist"
    assert "ranking_signals" in answer["sources"][0]
    assert answer["sources"][0]["ranking_signals"]["reference_role"] == "standard_reference"
    assert answer["sources"][0]["normalized_source_type"] == "web"
    assert isinstance(answer["sources"][0]["top_ranking_signals"], list)
    assert answer["answerSignals"]["specialist_reference_count"] == 1


def test_generate_answer_prompt_marks_low_quality_and_contradictions(monkeypatch, tmp_path):
    llm = FakeLLM()
    records = [
        {
            "id": "doc-review",
            "document": "rag has caveats when evidence quality is weak.",
            "distance": 0.2,
            "metadata": {
                "title": "Weak Note",
                "source_type": "vault",
                "file_path": "LearningHub/ai/weak.md",
                "record_id": "rec-review",
            },
        }
    ]
    sqlite_db = DummyFeatureSQLite(
        {
            "rec-review": {
                "importance_score": 0.7,
                "freshness_score": 0.7,
                "source_trust_score": 0.8,
                "contradiction_score": 1.0,
            }
        },
        ko_note_items={
            str((tmp_path / "vault" / "LearningHub/ai/weak.md").resolve()): {"payload_json": {"quality": {"flag": "needs_review"}}},
        },
    )
    config = Config()
    config.set_nested("obsidian", "vault_path", str((tmp_path / "vault").resolve()))
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records), llm=llm, sqlite_db=sqlite_db, config=config)
    monkeypatch.setattr(searcher, "_resolve_llm_for_request", lambda **kwargs: (llm, {"route": "fixed", "provider": "", "model": ""}, []))

    answer = searcher.generate_answer("rag 주의점", top_k=1, retrieval_mode="hybrid")

    assert answer["status"] == "ok"
    assert answer["answerSignals"]["caution_required"] is True
    assert llm.last_prompt is not None
    assert "needs_review/reject/unscored" in llm.last_prompt
    assert "contradictory_sources=1" in llm.last_prompt
    assert "quality=needs_review" in llm.last_context


def test_generate_answer_includes_claim_adjudication_in_payload_and_context(monkeypatch):
    llm = FakeLLM()
    records = [
        {
            "id": "paper-claim-1",
            "document": "MMLU accuracy reaches 71.5 compared with prior-work baseline.",
            "distance": 0.2,
            "metadata": {
                "title": "Claimed Paper",
                "source_type": "paper",
                "arxiv_id": "2501.00001",
                "record_id": "2501.00001",
                "contextual_summary": "[Claimed Paper] MMLU accuracy reaches 71.5.",
                "section_title": "Results",
            },
        }
    ]
    sqlite_db = DummyClaimNormalizationSQLite(
        {},
        record_claims={
            "2501.00001": [
                {
                    "claim_id": "claim:mmlu-acc",
                    "claim_text": "The paper improves MMLU accuracy to 71.5.",
                    "confidence": 0.91,
                }
            ]
        },
        normalizations={
            "claim:mmlu-acc": {
                "dataset": "MMLU",
                "metric": "Accuracy",
                "comparator": "prior-work",
                "result_direction": "better",
                "result_value_text": "71.5",
                "result_value_numeric": 71.5,
                "evidence_strength": "strong",
            }
        },
    )
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records), llm=llm, sqlite_db=sqlite_db)
    monkeypatch.setattr(searcher, "_resolve_llm_for_request", lambda **kwargs: (llm, {"route": "fixed", "provider": "", "model": ""}, []))
    monkeypatch.setattr(searcher, "_verify_answer", lambda **kwargs: {"status": "verified", "supportedClaimCount": 1, "unsupportedClaimCount": 0, "uncertainClaimCount": 0, "conflictMentioned": True, "needsCaution": False, "warnings": []})
    monkeypatch.setattr(searcher, "_rewrite_answer", lambda **kwargs: (kwargs["answer"], {"attempted": False, "applied": False, "warnings": [], "finalAnswerSource": "original"}))
    monkeypatch.setattr(searcher, "_apply_conservative_fallback_if_needed", lambda **kwargs: (kwargs["answer"], kwargs["rewrite_meta"], kwargs["verification"]))

    answer = searcher.generate_answer("What does this paper report on MMLU?", top_k=1, retrieval_mode="hybrid")

    assert answer["claimConsensus"]["supportCount"] == 1
    assert answer["claim_consensus"]["claimVerificationSummary"] == "supported"
    assert answer["claimVerification"][0]["normalized"]["metric"] == "Accuracy"
    assert "Claim adjudication summary:" in (llm.last_context or "")
    assert "summary=supported" in (llm.last_context or "")


def test_claim_consensus_keeps_single_paper_lookup_claims_advisory_in_answer_verification(monkeypatch):
    llm = FakeLLM()
    records = [
        {
            "id": "paper-claim-unsupported",
            "document": "The paper discusses MMLU benchmarking but does not provide a concrete score in this excerpt.",
            "distance": 0.2,
            "metadata": {
                "title": "Claimed Paper",
                "source_type": "paper",
                "arxiv_id": "2501.99999",
                "record_id": "2501.99999",
                "contextual_summary": "[Claimed Paper] discusses MMLU benchmarking.",
                "section_title": "Results",
            },
        }
    ]
    sqlite_db = DummyClaimNormalizationSQLite(
        {},
        record_claims={
            "2501.99999": [
                {
                    "claim_id": "claim:unsupported-mmlu",
                    "claim_text": "The paper reaches 99.9 accuracy on MMLU.",
                    "confidence": 0.91,
                }
            ]
        },
        normalizations={
            "claim:unsupported-mmlu": {
                "dataset": "MMLU",
                "metric": "Accuracy",
                "comparator": "prior-work",
                "result_direction": "better",
                "result_value_text": "99.9",
                "result_value_numeric": 99.9,
                "evidence_strength": "strong",
            }
        },
    )
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records), llm=llm, sqlite_db=sqlite_db)
    captured = {}
    monkeypatch.setattr(searcher, "_resolve_llm_for_request", lambda **kwargs: (llm, {"route": "fixed", "provider": "", "model": ""}, []))
    monkeypatch.setattr(searcher, "_verify_answer", lambda **kwargs: {"status": "verified", "supportedClaimCount": 1, "unsupportedClaimCount": 0, "uncertainClaimCount": 0, "conflictMentioned": True, "needsCaution": False, "warnings": []})

    def _capture_rewrite(**kwargs):
        captured["verification"] = dict(kwargs["verification"])
        return kwargs["answer"], {"attempted": False, "applied": False, "warnings": [], "finalAnswerSource": "original"}

    monkeypatch.setattr(searcher, "_rewrite_answer", _capture_rewrite)
    monkeypatch.setattr(searcher, "_apply_conservative_fallback_if_needed", lambda **kwargs: (kwargs["answer"], kwargs["rewrite_meta"], kwargs["verification"]))

    answer = searcher.generate_answer("What does this paper report on MMLU?", top_k=1, retrieval_mode="hybrid")

    assert captured["verification"]["claimWeakCount"] == 1
    assert captured["verification"]["claimConsensusMode"] == "advisory"
    assert captured["verification"]["needsCaution"] is False
    assert captured["verification"]["status"] == "verified"
    assert answer["answerVerification"]["claimWeakCount"] == 1
    assert answer["answerVerification"]["claimConsensusMode"] == "advisory"
    assert answer["answerVerification"]["needsCaution"] is False
    assert answer["answerVerification"]["status"] == "verified"


def test_search_prefers_vector_db_fts_when_available():
    db = DummyVectorDBWithLexical(_build_records())
    searcher = RAGSearcher(DummyEmbedder(), db)

    results = searcher.search("attention mechanism", top_k=2, source_type=None, retrieval_mode="keyword")

    assert len(results) == 2
    assert results[0].metadata["title"] == "Paper B"
    assert results[0].lexical_extras["fts_rank"] == 0.1
    assert results[0].document_id == "paper-b"


def test_search_does_not_fallback_to_bruteforce_when_fts_returns_no_hits():
    db = DummyVectorDBWithEmptyLexical(_build_records())
    searcher = RAGSearcher(DummyEmbedder(), db)

    results = searcher.search("attention mechanism", top_k=2, source_type=None, retrieval_mode="keyword")

    assert results == []


@pytest.mark.parametrize("mode", ["keyword", "semantic"])
def test_search_modes_work_with_filters(mode):
    db = DummyVectorDB(_build_records())
    searcher = RAGSearcher(DummyEmbedder(), db)

    if mode == "keyword":
        results = searcher.search("finance", top_k=2, source_type="paper", retrieval_mode=mode, alpha=0.7)
    else:
        results = searcher.search("finance", top_k=2, source_type="paper", retrieval_mode=mode)

    assert isinstance(results, list)
    assert all(r.metadata.get("source_type") == "paper" for r in results)


def test_generate_answer_includes_contextual_summary_in_context():
    llm = FakeLLM()
    db = DummyVectorDB(_build_records())
    searcher = RAGSearcher(DummyEmbedder(), db, llm=llm)

    answer = searcher.generate_answer("attention mechanism", top_k=1, retrieval_mode="hybrid")

    assert answer["answer"] == "요약 답변"
    assert llm.last_context is not None
    assert "section=Introduction" in llm.last_context
    assert "summary=[Paper" in llm.last_context
    assert len(answer["sources"]) == 1
    assert answer["sources"][0]["semantic_score"] >= 0


def test_generate_answer_expands_parent_context_from_child_hit():
    llm = FakeLLM()
    records = [
        {
            "id": "vault-0",
            "document": "RAG retrieves relevant chunks before generation.",
            "distance": 0.45,
            "metadata": {
                "title": "RAG Guide",
                "source_type": "vault",
                "file_path": "notes/rag.md",
                "chunk_index": 0,
                "parent_id": "vault:notes/rag.md::section:RAG Foundations",
                "parent_title": "RAG Foundations",
                "section_title": "RAG Foundations",
                "contextual_summary": "[RAG Foundations] retrieval overview",
            },
        },
        {
            "id": "vault-1",
            "document": "Parent-child retrieval keeps context while staying token efficient.",
            "distance": 0.15,
            "metadata": {
                "title": "RAG Guide",
                "source_type": "vault",
                "file_path": "notes/rag.md",
                "chunk_index": 1,
                "parent_id": "vault:notes/rag.md::section:RAG Foundations",
                "parent_title": "RAG Foundations",
                "section_title": "RAG Foundations",
                "contextual_summary": "[RAG Foundations] parent child retrieval",
            },
        },
        {
            "id": "vault-2",
            "document": "Section-level evidence should be merged with the hit chunk.",
            "distance": 0.35,
            "metadata": {
                "title": "RAG Guide",
                "source_type": "vault",
                "file_path": "notes/rag.md",
                "chunk_index": 2,
                "parent_id": "vault:notes/rag.md::section:RAG Foundations",
                "parent_title": "RAG Foundations",
                "section_title": "RAG Foundations",
                "contextual_summary": "[RAG Foundations] section evidence",
            },
        },
        {
            "id": "vault-3",
            "document": "Appendix: unrelated benchmark table.",
            "distance": 0.2,
            "metadata": {
                "title": "RAG Guide",
                "source_type": "vault",
                "file_path": "notes/rag.md",
                "chunk_index": 3,
                "parent_id": "vault:notes/rag.md::section:Appendix",
                "parent_title": "Appendix",
                "section_title": "Appendix",
                "contextual_summary": "[Appendix] unrelated benchmark",
            },
        },
    ]
    db = DummyVectorDB(records)
    searcher = RAGSearcher(DummyEmbedder(), db, llm=llm)

    answer = searcher.generate_answer("parent child retrieval", top_k=1, retrieval_mode="hybrid")

    assert answer["answer"] == "요약 답변"
    assert llm.last_context is not None
    assert "RAG retrieves relevant chunks before generation." in llm.last_context
    assert "Section-level evidence should be merged with the hit chunk." in llm.last_context
    assert "Appendix: unrelated benchmark table." not in llm.last_context
    assert answer["sources"][0]["parent_id"] == "vault:notes/rag.md::section:RAG Foundations"
    assert answer["sources"][0]["parent_label"] == "RAG Foundations"


def test_search_can_expand_parent_context_from_child_hit():
    records = [
        {
            "id": "vault-0",
            "document": "RAG retrieves relevant chunks before generation.",
            "distance": 0.45,
            "metadata": {
                "title": "RAG Guide",
                "source_type": "vault",
                "file_path": "notes/rag.md",
                "chunk_index": 0,
                "parent_id": "vault:notes/rag.md::section:RAG Foundations",
                "parent_title": "RAG Foundations",
                "section_title": "RAG Foundations",
                "contextual_summary": "[RAG Foundations] retrieval overview",
            },
        },
        {
            "id": "vault-1",
            "document": "Parent-child retrieval keeps context while staying token efficient.",
            "distance": 0.15,
            "metadata": {
                "title": "RAG Guide",
                "source_type": "vault",
                "file_path": "notes/rag.md",
                "chunk_index": 1,
                "parent_id": "vault:notes/rag.md::section:RAG Foundations",
                "parent_title": "RAG Foundations",
                "section_title": "RAG Foundations",
                "contextual_summary": "[RAG Foundations] parent child retrieval",
            },
        },
        {
            "id": "vault-2",
            "document": "Section-level evidence should be merged with the hit chunk.",
            "distance": 0.35,
            "metadata": {
                "title": "RAG Guide",
                "source_type": "vault",
                "file_path": "notes/rag.md",
                "chunk_index": 2,
                "parent_id": "vault:notes/rag.md::section:RAG Foundations",
                "parent_title": "RAG Foundations",
                "section_title": "RAG Foundations",
                "contextual_summary": "[RAG Foundations] section evidence",
            },
        },
        {
            "id": "vault-3",
            "document": "Appendix: unrelated benchmark table.",
            "distance": 0.2,
            "metadata": {
                "title": "RAG Guide",
                "source_type": "vault",
                "file_path": "notes/rag.md",
                "chunk_index": 3,
                "parent_id": "vault:notes/rag.md::section:Appendix",
                "parent_title": "Appendix",
                "section_title": "Appendix",
                "contextual_summary": "[Appendix] unrelated benchmark",
            },
        },
    ]
    db = DummyVectorDB(records)
    searcher = RAGSearcher(DummyEmbedder(), db)

    results = searcher.search(
        "parent child retrieval",
        top_k=1,
        retrieval_mode="hybrid",
        expand_parent_context=True,
    )

    assert len(results) == 1
    assert "RAG retrieves relevant chunks before generation." in results[0].document
    assert "Section-level evidence should be merged with the hit chunk." in results[0].document
    assert "Appendix: unrelated benchmark table." not in results[0].document
    assert results[0].metadata["resolved_parent_id"] == "vault:notes/rag.md::section:RAG Foundations"
    assert results[0].metadata["resolved_parent_label"] == "RAG Foundations"


def test_generate_answer_blocks_external_call_when_p0_present():
    llm = FakeLLM()
    records = [
        {
            "id": "note-p0",
            "document": "문의: private@example.com 로 연락하세요.",
            "distance": 0.1,
            "metadata": {
                "title": "Sensitive Note",
                "source_type": "note",
                "file_path": "notes/private.md",
            },
        }
    ]
    db = DummyVectorDB(records)
    searcher = RAGSearcher(DummyEmbedder(), db, llm=llm)

    result = searcher.generate_answer("연락처 알려줘", top_k=1, retrieval_mode="hybrid")

    assert result["status"] == "blocked"
    assert result["policy"]["originalClassification"] == "P0"
    assert llm.calls == 0


def test_generate_answer_includes_p1_warning_metadata():
    llm = FakeLLM()
    records = [
        {
            "id": "note-p1",
            "document": '{"entity_id":"paper_123","confidence":0.92,"status":"ok"}',
            "distance": 0.1,
            "metadata": {
                "title": "Structured Facts",
                "source_type": "note",
                "file_path": "notes/structured.md",
            },
        }
    ]
    db = DummyVectorDB(records)
    searcher = RAGSearcher(DummyEmbedder(), db, llm=llm)

    result = searcher.generate_answer("상태 요약", top_k=1, retrieval_mode="hybrid")

    assert result["status"] == "ok"
    assert result["policy"]["originalClassification"] == "P1"
    assert result["policy"]["warnings"]
    assert llm.calls == 1


def test_generate_answer_uses_api_llm_when_hybrid_router_selects_api(monkeypatch):
    local_llm = FakeLLM()
    api_llm = FakeLLM()
    db = DummyVectorDB(_build_records())
    config = Config()
    searcher = RAGSearcher(DummyEmbedder(), db, llm=local_llm, config=config)

    def _fake_router(*args, **kwargs):
        return api_llm, DummyHybridDecision(route="api", provider="openai-compat", model="sonar-pro"), []

    monkeypatch.setattr("knowledge_hub.ai.rag.get_llm_for_hybrid_routing", _fake_router)
    result = searcher.generate_answer("attention mechanism", top_k=1, retrieval_mode="hybrid")

    assert result["status"] == "ok"
    assert api_llm.calls == 1
    assert local_llm.calls == 0
    assert result["router"]["selected"]["route"] == "api"
    assert "quality_flag=ok" in api_llm.last_prompt or "ok=" in api_llm.last_prompt


def test_generate_answer_allows_p0_when_hybrid_router_selects_local(monkeypatch):
    local_llm = FakeLLM()
    records = [
        {
            "id": "note-p0-local",
            "document": "문의: private@example.com 로 연락하세요.",
            "distance": 0.1,
            "metadata": {
                "title": "Sensitive Local Note",
                "source_type": "note",
                "file_path": "notes/private-local.md",
            },
        }
    ]
    db = DummyVectorDB(records)
    config = Config()
    searcher = RAGSearcher(DummyEmbedder(), db, llm=local_llm, config=config)

    def _fake_router(*args, **kwargs):
        return local_llm, DummyHybridDecision(route="local", provider="ollama", model="qwen2.5:7b"), []

    monkeypatch.setattr("knowledge_hub.ai.rag.get_llm_for_hybrid_routing", _fake_router)
    result = searcher.generate_answer("연락처 알려줘", top_k=1, retrieval_mode="hybrid")

    assert result["status"] == "ok"
    assert result["router"]["selected"]["route"] == "local"
    assert result["policy"]["externalCall"]["mode"] == "rag-local"
    assert local_llm.calls == 1


def test_generate_answer_does_not_reuse_stale_init_llm_for_local_route(monkeypatch):
    init_llm = FakeLLM()
    routed_local_llm = FakeLLM()
    db = DummyVectorDB(_build_records())
    config = Config()
    config.set_nested("routing", "llm", "tasks", "local", "provider", "ollama")
    config.set_nested("routing", "llm", "tasks", "local", "model", "qwen3:14b")
    config.set_nested("routing", "llm", "tasks", "local", "timeout_sec", 45)
    searcher = RAGSearcher(DummyEmbedder(), db, llm=init_llm, config=config)

    def _fake_router(*args, **kwargs):
        return routed_local_llm, DummyHybridDecision(route="local", provider="ollama", model="qwen3:14b", timeout_sec=45), []

    monkeypatch.setattr("knowledge_hub.ai.rag.get_llm_for_hybrid_routing", _fake_router)
    monkeypatch.setattr(searcher, "_verify_answer", lambda **_kwargs: {"status": "verified", "needsCaution": False, "warnings": [], "unsupportedClaimCount": 0})
    monkeypatch.setattr(searcher, "_rewrite_answer", lambda **kwargs: (kwargs["answer"], {"attempted": False, "applied": False, "finalAnswerSource": "original"}))
    monkeypatch.setattr(
        searcher,
        "_apply_conservative_fallback_if_needed",
        lambda **kwargs: (kwargs["answer"], kwargs["rewrite_meta"], kwargs["verification"]),
    )
    result = searcher.generate_answer("attention mechanism", top_k=1, retrieval_mode="hybrid")

    assert result["status"] == "ok"
    assert routed_local_llm.calls == 1
    assert init_llm.calls == 0
    assert result["router"]["selected"]["timeoutSec"] == 45


def test_verify_answer_local_only_uses_heuristic_warning_path():
    config = Config()
    config.set_nested("routing", "llm", "tasks", "local", "provider", "ollama")
    config.set_nested("routing", "llm", "tasks", "local", "model", "qwen3:14b")
    config.set_nested("routing", "llm", "tasks", "local", "timeout_sec", 45)
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=StaticLLM("unused"), config=config)

    verification = searcher._verify_answer(
        query="attention mechanism",
        answer="Attention은 seq2seq training에 필요합니다.",
        evidence=[{"title": "Paper A", "excerpt": "attention is needed for seq2seq training."}],
        answer_signals={"contradictory_source_count": 0},
        contradicting_beliefs=[],
        allow_external=False,
    )

    assert verification["status"] == "caution"
    assert verification["route"]["route"] == "local"
    assert verification["route"]["mode"] == "heuristic"
    assert "answer verification fell back to heuristic: local verifier route is disabled by default" in verification["warnings"]
    assert "answer verification used heuristic fallback" in verification["warnings"]


def test_rewrite_answer_local_only_skips_with_route_unavailable_warning():
    config = Config()
    config.set_nested("routing", "llm", "tasks", "local", "provider", "ollama")
    config.set_nested("routing", "llm", "tasks", "local", "model", "qwen3:14b")
    config.set_nested("routing", "llm", "tasks", "local", "timeout_sec", 45)
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=StaticLLM("unused"), config=config)
    answer = "RAG는 2028년에 처음 제안되었습니다."

    rewritten, rewrite_meta = searcher._rewrite_answer(
        query="attention mechanism",
        answer=answer,
        evidence=[{"title": "Paper A", "excerpt": "attention is needed for seq2seq training."}],
        answer_signals={"contradictory_source_count": 0},
        verification={
            "needsCaution": True,
            "unsupportedClaimCount": 1,
            "supportedClaimCount": 0,
            "claimUnsupportedCount": 1,
            "claimConflictCount": 0,
            "claimWeakCount": 0,
            "conflictMentioned": True,
        },
        contradicting_beliefs=[],
        allow_external=False,
    )

    assert rewritten == answer
    assert rewrite_meta["attempted"] is False
    assert rewrite_meta["applied"] is False
    assert rewrite_meta["finalAnswerSource"] == "original"
    assert rewrite_meta["requiresConservativeFallback"] is True
    assert "answer rewrite skipped: unsupported claims require conservative fallback" in rewrite_meta["warnings"]


def test_conservative_fallback_can_apply_when_unsupported_rewrite_was_skipped(monkeypatch):
    config = Config()
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=StaticLLM("unused"), config=config)
    answer = "RAG는 2028년에 처음 제안되었습니다."

    monkeypatch.setattr(
        "knowledge_hub.ai.answer_rewrite.verify_answer",
        lambda *_args, **_kwargs: {
            "status": "verified",
            "needsCaution": False,
            "warnings": [],
            "supportedClaimCount": 1,
            "unsupportedClaimCount": 0,
            "uncertainClaimCount": 0,
            "conflictMentioned": True,
        },
    )

    final_answer, rewrite_meta, final_verification = searcher._apply_conservative_fallback_if_needed(
        query="attention mechanism",
        answer=answer,
        rewrite_meta={
            "attempted": False,
            "applied": False,
            "requiresConservativeFallback": True,
            "warnings": ["answer rewrite skipped: unsupported claims require conservative fallback"],
        },
        verification={
            "needsCaution": True,
            "unsupportedClaimCount": 1,
            "supportedClaimCount": 0,
            "claimUnsupportedCount": 1,
            "claimConflictCount": 0,
            "claimWeakCount": 0,
            "claims": [{"claim": "RAG는 2028년에 처음 제안되었습니다.", "verdict": "unsupported"}],
        },
        evidence=[{"title": "Paper A", "excerpt": "attention is needed for seq2seq training."}],
        answer_signals={"contradictory_source_count": 0},
        contradicting_beliefs=[],
        allow_external=False,
    )

    assert final_answer != answer
    assert final_answer.startswith("제공된 근거만으로는")
    assert rewrite_meta["applied"] is True
    assert rewrite_meta["finalAnswerSource"] == "conservative_fallback"
    assert final_verification["status"] == "verified"


def test_generate_answer_local_only_surfaces_verification_and_rewrite_warnings(monkeypatch):
    answer_llm = StaticLLM("RAG는 검색된 근거를 바탕으로 답변을 생성합니다.")
    config = Config()
    config.set_nested("routing", "llm", "tasks", "local", "provider", "ollama")
    config.set_nested("routing", "llm", "tasks", "local", "model", "qwen3:14b")
    config.set_nested("routing", "llm", "tasks", "local", "timeout_sec", 45)
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=answer_llm, config=config)

    def _fake_router(*args, **kwargs):
        return answer_llm, DummyHybridDecision(route="local", provider="ollama", model="qwen3:14b", timeout_sec=45), []

    monkeypatch.setattr("knowledge_hub.ai.rag.get_llm_for_hybrid_routing", _fake_router)

    result = searcher.generate_answer("attention mechanism", top_k=1, retrieval_mode="hybrid", allow_external=False)

    assert result["status"] == "ok"
    assert result["router"]["selected"]["route"] == "local"
    assert result["initialAnswerVerification"]["route"]["route"] == "local"
    assert result["initialAnswerVerification"]["route"]["mode"] == "heuristic"
    assert result["answerRewrite"]["requiresConservativeFallback"] is True
    assert result["answerRewrite"]["applied"] is True
    assert result["answerRewrite"]["finalAnswerSource"] == "conservative_fallback"
    assert any(
        "answer verification fell back to heuristic: local verifier route is disabled by default" in warning
        for warning in result["warnings"]
    )
    assert any("answer rewrite skipped: unsupported claims require conservative fallback" in warning for warning in result["warnings"])


def test_generate_answer_returns_verified_answer_verification(monkeypatch):
    answer_llm = StaticLLM("RAG는 검색된 근거를 바탕으로 답변을 생성합니다.")
    verifier_llm = StaticLLM(
        '{"claims":[{"claim":"RAG는 검색된 근거를 바탕으로 답변을 생성합니다.","verdict":"supported","evidenceTitles":["Paper A"],"reason":"근거 excerpt가 답변을 직접 지지합니다."}],"conflictMentioned":true,"needsCaution":false,"summary":"모든 핵심 claim이 제공된 근거와 직접 연결됩니다."}'
    )
    db = DummyVectorDB(_build_records())
    searcher = RAGSearcher(DummyEmbedder(), db, llm=answer_llm)

    monkeypatch.setattr(
        searcher,
        "_resolve_llm_for_verification",
        lambda **_kwargs: (
            verifier_llm,
            {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": ["task_default=strong"], "fallbackUsed": False},
            [],
        ),
    )

    result = searcher.generate_answer("attention mechanism", top_k=1, retrieval_mode="hybrid")

    assert result["status"] == "ok"
    assert result["answerVerification"]["status"] == "verified"
    assert result["answerVerification"]["supportedClaimCount"] == 1
    assert result["answerVerification"]["unsupportedClaimCount"] == 0
    assert result["answerVerification"]["route"]["route"] == "strong"
    assert result["router"]["selected"]["route"] == "fixed"


def test_generate_answer_marks_unsupported_claims_in_verification(monkeypatch):
    answer_llm = StaticLLM("RAG는 2028년에 처음 제안되었고 모든 태스크에서 최고 성능을 냅니다.")
    verifier_llm = StaticLLM(
        '{"claims":[{"claim":"RAG는 2028년에 처음 제안되었다.","verdict":"unsupported","evidenceTitles":[],"reason":"제공된 근거 어디에도 연도 정보가 없습니다."},{"claim":"RAG는 모든 태스크에서 최고 성능을 낸다.","verdict":"unsupported","evidenceTitles":[],"reason":"제공된 근거 어디에도 범용 SOTA 주장이 없습니다."}],"conflictMentioned":true,"needsCaution":true,"summary":"근거에 없는 구체적 사실이 포함되어 있습니다."}'
    )
    db = DummyVectorDB(_build_records())
    searcher = RAGSearcher(DummyEmbedder(), db, llm=answer_llm)

    monkeypatch.setattr(
        searcher,
        "_resolve_llm_for_verification",
        lambda **_kwargs: (
            verifier_llm,
            {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": ["task_default=strong"], "fallbackUsed": False},
            [],
        ),
    )

    result = searcher.generate_answer("attention mechanism", top_k=1, retrieval_mode="hybrid")

    assert result["answerVerification"]["status"] == "caution"
    assert result["answerVerification"]["unsupportedClaimCount"] >= 1
    assert result["answerVerification"]["claims"][0]["verdict"] == "unsupported"


def test_generate_answer_verification_detects_unmentioned_conflicts_with_heuristic(monkeypatch):
    answer_llm = StaticLLM("RAG는 항상 정확한 답변을 제공합니다.")
    records = [
        {
            "id": "doc-conflict",
            "document": "RAG answers may still hallucinate when retrieval misses key evidence.",
            "distance": 0.2,
            "metadata": {
                "title": "Conflict Note",
                "source_type": "web",
                "record_id": "rec-conflict",
            },
        }
    ]
    sqlite_db = DummyClaimSQLite(
        {
            "rec-conflict": {
                "importance_score": 0.7,
                "freshness_score": 0.7,
                "source_trust_score": 0.84,
                "contradiction_score": 1.0,
            }
        },
        record_claims={
            "rec-conflict": [{"claim_id": "c1", "claim_text": "RAG may hallucinate when retrieval is weak.", "confidence": 0.8}]
        },
        beliefs=[
            {
                "belief_id": "b1",
                "statement": "RAG can still hallucinate under weak retrieval.",
                "status": "rejected",
                "contradiction_ids": ["b2"],
                "derived_from_claim_ids": ["c1"],
            }
        ],
    )
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records), llm=answer_llm, sqlite_db=sqlite_db)

    monkeypatch.setattr(
        searcher,
        "_resolve_llm_for_verification",
        lambda **_kwargs: (
            None,
            {"route": "fallback-only", "provider": "", "model": "", "reasons": ["forced_heuristic"], "fallbackUsed": True},
            ["forced heuristic verification"],
        ),
    )

    result = searcher.generate_answer("RAG 장단점", top_k=1, retrieval_mode="hybrid")

    assert result["answerVerification"]["status"] == "caution"
    assert result["initialAnswerVerification"]["conflictMentioned"] is False
    assert result["answerVerification"]["conflictMentioned"] is True
    assert result["answerVerification"]["needsCaution"] is True


def test_generate_answer_verification_skips_without_breaking_payload(monkeypatch):
    answer_llm = StaticLLM("요약 답변")
    db = DummyVectorDB(_build_records())
    searcher = RAGSearcher(DummyEmbedder(), db, llm=answer_llm)

    monkeypatch.setattr(
        searcher,
        "_resolve_llm_for_verification",
        lambda **_kwargs: (
            None,
            {"route": "fallback-only", "provider": "", "model": "", "reasons": ["config_missing"], "fallbackUsed": True},
            ["verification unavailable"],
        ),
    )

    result = searcher.generate_answer("attention mechanism", top_k=1, retrieval_mode="hybrid")

    assert result["status"] == "ok"
    assert result["answer"] == "요약 답변"
    assert result["answerVerification"]["status"] == "skipped"
    assert result["answerVerification"]["route"]["route"] == "fallback-only"
    assert result["warnings"]


def test_generate_answer_skips_rewrite_when_unsupported_claims_exist(monkeypatch):
    answer_llm = StaticLLM("RAG는 2028년에 처음 제안되었고 모든 태스크에서 최고 성능을 냅니다.")
    verifier_initial = StaticLLM(
        '{"claims":[{"claim":"RAG는 2028년에 처음 제안되었다.","verdict":"unsupported","evidenceTitles":[],"reason":"근거에 없는 연도 정보입니다."}],"conflictMentioned":true,"needsCaution":true,"summary":"근거 밖 claim이 있습니다."}'
    )
    verifier_after_fallback = StaticLLM(
        '{"claims":[],"conflictMentioned":true,"needsCaution":false,"summary":"보수적 fallback 답변은 단정적 claim을 제거했습니다."}'
    )
    db = DummyVectorDB(_build_records())
    searcher = RAGSearcher(DummyEmbedder(), db, llm=answer_llm)

    verify_calls = iter(
        [
            (verifier_initial, {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False}, []),
            (verifier_after_fallback, {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False}, []),
        ]
    )
    monkeypatch.setattr(searcher, "_resolve_llm_for_verification", lambda **_kwargs: next(verify_calls))
    monkeypatch.setattr(
        searcher,
        "_resolve_llm_for_rewrite",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unsupported claims must not call rewrite route")),
    )

    result = searcher.generate_answer("attention mechanism", top_k=1, retrieval_mode="hybrid")

    assert "단정적인 결론을 내리기 어렵습니다" in result["answer"]
    assert result["answerRewrite"]["attempted"] is True
    assert result["answerRewrite"]["applied"] is True
    assert result["answerRewrite"]["finalAnswerSource"] == "conservative_fallback"
    assert result["answerRewrite"]["requiresConservativeFallback"] is True
    assert result["initialAnswerVerification"]["unsupportedClaimCount"] == 1
    assert result["answerVerification"]["status"] == "verified"
    assert any("answer rewrite skipped: unsupported claims require conservative fallback" in warning for warning in result["warnings"])


def test_generate_answer_rewrites_when_conflict_language_missing(monkeypatch):
    answer_llm = StaticLLM("RAG는 검색 근거를 활용하지만 주의점은 따로 언급하지 않았습니다.")
    verifier_initial = StaticLLM(
        '{"claims":[{"claim":"RAG는 검색 근거를 활용합니다.","verdict":"supported","evidenceTitles":["Conflict Note"],"reason":"검색 근거를 활용한다는 점은 지지됩니다."}],"conflictMentioned":false,"needsCaution":true,"summary":"상충 가능성 언급이 없습니다."}'
    )
    rewrite_llm = StaticLLM("RAG는 유용하지만, 검색이 약하면 환각이 남을 수 있어 결과를 주의해서 해석해야 합니다.")
    verifier_final = StaticLLM(
        '{"claims":[{"claim":"RAG는 유용하지만, 검색이 약하면 환각이 남을 수 있다.","verdict":"supported","evidenceTitles":["Conflict Note"],"reason":"근거가 직접 지지합니다."}],"conflictMentioned":true,"needsCaution":false,"summary":"재작성 후 주의 문구가 반영되었습니다."}'
    )
    records = [
        {
            "id": "doc-conflict",
            "document": "RAG answers may still hallucinate when retrieval misses key evidence.",
            "distance": 0.2,
            "metadata": {"title": "Conflict Note", "source_type": "web", "record_id": "rec-conflict"},
        }
    ]
    sqlite_db = DummyClaimSQLite(
        {"rec-conflict": {"importance_score": 0.7, "freshness_score": 0.7, "source_trust_score": 0.84, "contradiction_score": 1.0}},
        record_claims={"rec-conflict": [{"claim_id": "c1", "claim_text": "RAG may hallucinate when retrieval is weak.", "confidence": 0.8}]},
        beliefs=[{"belief_id": "b1", "statement": "RAG can still hallucinate under weak retrieval.", "status": "rejected", "contradiction_ids": ["b2"], "derived_from_claim_ids": ["c1"]}],
    )
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records), llm=answer_llm, sqlite_db=sqlite_db)

    verify_calls = iter(
        [
            (verifier_initial, {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False}, []),
            (verifier_final, {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False}, []),
        ]
    )
    monkeypatch.setattr(searcher, "_resolve_llm_for_verification", lambda **_kwargs: next(verify_calls))
    monkeypatch.setattr(
        searcher,
        "_resolve_llm_for_rewrite",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("caution must not call rewrite route")),
    )

    result = searcher.generate_answer("RAG 장단점", top_k=1, retrieval_mode="hybrid")

    assert result["answerRewrite"]["attempted"] is True
    assert result["answerRewrite"]["applied"] is True
    assert result["answerRewrite"]["finalAnswerSource"] == "conservative_fallback"
    assert result["initialAnswerVerification"]["conflictMentioned"] is False
    assert result["answerVerification"]["conflictMentioned"] is True
    assert result["answerVerification"]["needsCaution"] is False
    assert any("answer rewrite skipped: caution requires conservative fallback" in warning for warning in result["warnings"])


def test_generate_answer_uses_conservative_fallback_for_uncertain_only_claims(monkeypatch):
    answer_llm = StaticLLM("RAG는 항상 정확한 답변을 제공합니다.")
    verifier_initial = StaticLLM(
        '{"claims":[{"claim":"RAG는 항상 정확한 답변을 제공합니다.","verdict":"uncertain","evidenceTitles":["Conflict Note"],"reason":"상충 근거가 있어 단정할 수 없습니다."}],"conflictMentioned":true,"needsCaution":true,"summary":"근거가 불확실합니다."}'
    )
    verifier_after_fallback = StaticLLM(
        '{"claims":[{"claim":"RAG는 항상 정확하다고 단정하기 어렵습니다.","verdict":"supported","evidenceTitles":["Conflict Note"],"reason":"보수적 fallback은 단정 표현을 제거했습니다."}],"conflictMentioned":true,"needsCaution":false,"summary":"보수적 fallback 답변입니다."}'
    )
    records = [
        {
            "id": "doc-conflict",
            "document": "RAG answers may still hallucinate when retrieval misses key evidence.",
            "distance": 0.2,
            "metadata": {"title": "Conflict Note", "source_type": "web", "record_id": "rec-conflict"},
        }
    ]
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(records), llm=answer_llm)

    verify_calls = iter(
        [
            (verifier_initial, {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False}, []),
            (verifier_after_fallback, {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False}, []),
        ]
    )
    monkeypatch.setattr(searcher, "_resolve_llm_for_verification", lambda **_kwargs: next(verify_calls))
    monkeypatch.setattr(
        searcher,
        "_resolve_llm_for_rewrite",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("uncertain-only claims must not call rewrite route")),
    )

    result = searcher.generate_answer("RAG 장단점", top_k=1, retrieval_mode="hybrid")

    assert "단정적인 결론을 내리기 어렵습니다" in result["answer"]
    assert result["answerRewrite"]["requiresConservativeFallback"] is True
    assert result["answerRewrite"]["finalAnswerSource"] == "conservative_fallback"
    assert result["initialAnswerVerification"]["supportedClaimCount"] == 0
    assert result["initialAnswerVerification"]["uncertainClaimCount"] == 1
    assert any("answer rewrite skipped: uncertain claims require conservative fallback" in warning for warning in result["warnings"])


def test_generate_answer_does_not_call_rewrite_route_for_unsupported_claims(monkeypatch):
    answer_llm = StaticLLM("RAG는 2028년에 처음 제안되었습니다.")
    verifier_initial = StaticLLM(
        '{"claims":[{"claim":"RAG는 2028년에 처음 제안되었다.","verdict":"unsupported","evidenceTitles":[],"reason":"근거에 없는 연도 정보입니다."}],"conflictMentioned":true,"needsCaution":true,"summary":"근거 밖 claim이 있습니다."}'
    )
    db = DummyVectorDB(_build_records())
    searcher = RAGSearcher(DummyEmbedder(), db, llm=answer_llm)

    verifier_after_fallback = StaticLLM(
        '{"claims":[],"conflictMentioned":true,"needsCaution":false,"summary":"보수적 fallback 답변은 단정적 claim을 제거했습니다."}'
    )
    verify_calls = iter(
        [
            (verifier_initial, {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False}, []),
            (verifier_after_fallback, {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False}, []),
        ]
    )
    monkeypatch.setattr(searcher, "_resolve_llm_for_verification", lambda **_kwargs: next(verify_calls))
    monkeypatch.setattr(
        searcher,
        "_resolve_llm_for_rewrite",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unsupported claims must not call rewrite route")),
    )

    result = searcher.generate_answer("attention mechanism", top_k=1, retrieval_mode="hybrid")

    assert "단정적인 결론을 내리기 어렵습니다" in result["answer"]
    assert result["answerRewrite"]["attempted"] is True
    assert result["answerRewrite"]["applied"] is True
    assert result["answerRewrite"]["finalAnswerSource"] == "conservative_fallback"
    assert any("answer rewrite skipped: unsupported claims require conservative fallback" in warning for warning in result["warnings"])


def test_stream_answer_buffers_and_emits_conservative_fallback_for_unsupported_claim(monkeypatch):
    answer_llm = StaticLLM("RAG는 2028년에 처음 제안되었습니다.")
    verifier_initial = StaticLLM(
        '{"claims":[{"claim":"RAG는 2028년에 처음 제안되었다.","verdict":"unsupported","evidenceTitles":[],"reason":"근거에 없는 연도 정보입니다."}],"conflictMentioned":true,"needsCaution":true,"summary":"근거 밖 claim이 있습니다."}'
    )
    verifier_after_fallback = StaticLLM(
        '{"claims":[],"conflictMentioned":true,"needsCaution":false,"summary":"보수적 fallback 답변은 단정적 claim을 제거했습니다."}'
    )
    db = DummyVectorDB(_build_records())
    searcher = RAGSearcher(DummyEmbedder(), db, llm=answer_llm)
    verify_calls = iter(
        [
            (verifier_initial, {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False}, []),
            (verifier_after_fallback, {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False}, []),
        ]
    )
    monkeypatch.setattr(searcher, "_resolve_llm_for_verification", lambda **_kwargs: next(verify_calls))
    monkeypatch.setattr(
        searcher,
        "_resolve_llm_for_rewrite",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unsupported claims must not call rewrite route")),
    )

    output = "".join(searcher.stream_answer("attention mechanism", top_k=1, retrieval_mode="hybrid"))

    assert "단정적인 결론을 내리기 어렵습니다" in output
    assert "2028" not in output


def test_generate_answer_applies_conservative_fallback_when_caution_remains(monkeypatch):
    answer_llm = StaticLLM("RAG는 2028년에 처음 제안되었고 모든 태스크에서 최고 성능을 냅니다.")
    verifier_initial = StaticLLM(
        '{"claims":[{"claim":"RAG는 2028년에 처음 제안되었다.","verdict":"unsupported","evidenceTitles":[],"reason":"근거에 없는 연도 정보입니다."}],"conflictMentioned":true,"needsCaution":true,"summary":"근거 밖 claim이 있습니다."}'
    )
    verifier_after_fallback = StaticLLM(
        '{"claims":[],"conflictMentioned":true,"needsCaution":false,"summary":"보수적 fallback 답변은 단정적 claim을 제거했습니다."}'
    )
    db = DummyVectorDB(_build_records())
    searcher = RAGSearcher(DummyEmbedder(), db, llm=answer_llm)

    verify_calls = iter(
        [
            (verifier_initial, {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False}, []),
            (verifier_after_fallback, {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False}, []),
        ]
    )
    monkeypatch.setattr(searcher, "_resolve_llm_for_verification", lambda **_kwargs: next(verify_calls))
    monkeypatch.setattr(
        searcher,
        "_resolve_llm_for_rewrite",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unsupported claims must not call rewrite route")),
    )

    result = searcher.generate_answer("attention mechanism", top_k=1, retrieval_mode="hybrid")

    assert result["answerRewrite"]["applied"] is True
    assert result["answerRewrite"]["finalAnswerSource"] == "conservative_fallback"
    assert "단정적인 결론을 내리기 어렵습니다" in result["answer"]
    assert result["answerVerification"]["status"] == "verified"
    assert any("answer conservative fallback applied" in warning for warning in result["warnings"])


def test_stream_answer_emits_conservative_fallback_when_caution_remains(monkeypatch):
    answer_llm = StaticLLM("RAG는 2028년에 처음 제안되었고 모든 태스크에서 최고 성능을 냅니다.")
    verifier_initial = StaticLLM(
        '{"claims":[{"claim":"RAG는 2028년에 처음 제안되었다.","verdict":"unsupported","evidenceTitles":[],"reason":"근거에 없는 연도 정보입니다."}],"conflictMentioned":true,"needsCaution":true,"summary":"근거 밖 claim이 있습니다."}'
    )
    verifier_after_fallback = StaticLLM(
        '{"claims":[],"conflictMentioned":true,"needsCaution":false,"summary":"보수적 fallback 답변은 단정적 claim을 제거했습니다."}'
    )
    db = DummyVectorDB(_build_records())
    searcher = RAGSearcher(DummyEmbedder(), db, llm=answer_llm)

    verify_calls = iter(
        [
            (verifier_initial, {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False}, []),
            (verifier_after_fallback, {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False}, []),
        ]
    )
    monkeypatch.setattr(searcher, "_resolve_llm_for_verification", lambda **_kwargs: next(verify_calls))
    monkeypatch.setattr(
        searcher,
        "_resolve_llm_for_rewrite",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unsupported claims must not call rewrite route")),
    )

    output = "".join(searcher.stream_answer("attention mechanism", top_k=1, retrieval_mode="hybrid"))

    assert "단정적인 결론을 내리기 어렵습니다" in output
    assert "모든 태스크에서 최고 성능" not in output


def test_generate_answer_persists_operator_safe_log(monkeypatch):
    sqlite_db = DummyFeatureSQLiteWithLogs({})
    answer_llm = StaticLLM("Attention은 토큰 간 중요도를 계산합니다.")
    verifier_llm = StaticLLM(
        '{"claims":[{"claim":"Attention은 토큰 간 중요도를 계산한다.","verdict":"supported","evidenceTitles":["Paper A"],"reason":"검색 근거가 동일한 설명을 포함합니다."}],"conflictMentioned":true,"needsCaution":false,"summary":"답변이 근거와 일치합니다."}'
    )
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=answer_llm, sqlite_db=sqlite_db)
    monkeypatch.setattr(
        searcher,
        "_resolve_llm_for_verification",
        lambda **_kwargs: (
            verifier_llm,
            {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False},
            [],
        ),
    )

    query = (
        "attention mechanism의 동작 원리를 자세히 설명해줘. "
        "이 문장은 로그에 원문 전체가 저장되면 안 되고 마지막 토큰 UNIQUE_SENSITIVE_TRAILER 는 남으면 안 된다."
    )
    result = searcher.generate_answer(query, top_k=1, retrieval_mode="hybrid")

    rows = sqlite_db.list_rag_answer_logs(limit=10, days=7)
    assert result["answerVerification"]["status"] == "verified"
    assert len(rows) == 1
    row = rows[0]
    assert row["query_hash"]
    assert row["query_digest"]
    assert row["query_digest"] != query
    serialized = " ".join(str(value) for value in row.values())
    assert "UNIQUE_SENSITIVE_TRAILER" not in serialized
    assert row["verification_status"] == "verified"
    assert row["result_status"] == "ok"
    assert row["final_answer_source"] == "original"


def test_stream_answer_persists_final_log_after_buffered_completion(monkeypatch):
    sqlite_db = DummyFeatureSQLiteWithLogs({})
    answer_llm = StaticLLM("Attention은 입력 간 중요도를 계산합니다.")
    verifier_initial = StaticLLM(
        '{"claims":[{"claim":"Attention은 입력 간 중요도를 계산한다.","verdict":"supported","evidenceTitles":["Paper A"],"reason":"검색 근거와 일치합니다."}],"conflictMentioned":true,"needsCaution":false,"summary":"답변이 근거와 일치합니다."}'
    )
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=answer_llm, sqlite_db=sqlite_db)
    monkeypatch.setattr(
        searcher,
        "_resolve_llm_for_verification",
        lambda **_kwargs: (
            verifier_initial,
            {"route": "strong", "provider": "openai", "model": "gpt-5.4", "reasons": [], "fallbackUsed": False},
            [],
        ),
    )

    output = "".join(searcher.stream_answer("attention mechanism", top_k=1, retrieval_mode="hybrid"))

    rows = sqlite_db.list_rag_answer_logs(limit=10, days=7)
    assert "Attention은 입력 간 중요도를 계산합니다." in output
    assert len(rows) == 1
    assert rows[0]["verification_status"] == "verified"
    assert rows[0]["warning_count"] == 0


def test_build_ops_report_aggregates_recent_answer_logs():
    sqlite_db = DummyFeatureSQLiteWithLogs({})
    sqlite_db.add_rag_answer_log(
        query_hash="hash-a",
        query_digest="RAG 장단점",
        source_type="all",
        retrieval_mode="hybrid",
        allow_external=False,
        result_status="ok",
        verification_status="caution",
        needs_caution=True,
        supported_claim_count=1,
        uncertain_claim_count=1,
        unsupported_claim_count=1,
        conflict_mentioned=False,
        rewrite_attempted=True,
        rewrite_applied=True,
        final_answer_source="rewritten",
        warning_count=1,
        source_count=2,
        evidence_count=2,
        answer_route={"route": "strong"},
        verification_route={"route": "strong"},
        rewrite_route={"route": "strong"},
        warnings=["answer verification caution"],
    )
    sqlite_db.add_rag_answer_log(
        query_hash="hash-b",
        query_digest="Attention 요약",
        source_type="all",
        retrieval_mode="hybrid",
        allow_external=False,
        result_status="ok",
        verification_status="verified",
        needs_caution=False,
        supported_claim_count=2,
        uncertain_claim_count=0,
        unsupported_claim_count=0,
        conflict_mentioned=True,
        rewrite_attempted=False,
        rewrite_applied=False,
        final_answer_source="original",
        warning_count=0,
        source_count=1,
        evidence_count=1,
        answer_route={"route": "strong"},
        verification_route={"route": "strong"},
        rewrite_route={},
        warnings=[],
    )

    report = build_rag_ops_report(sqlite_db, limit=10, days=7)

    assert report["status"] == "ok"
    assert report["counts"]["total"] == 2
    assert report["counts"]["needsCaution"] == 1
    assert report["counts"]["rewriteApplied"] == 1
    assert report["verification"]["caution"] == 1
    assert report["rates"]["unsupportedClaimRate"] == pytest.approx(0.5, abs=1e-6)
    assert report["topWarningPatterns"][0]["warning"] == "answer verification caution"
    assert report["samples"][0]["queryDigest"]
    assert "answer" not in report["samples"][0]
    assert report["alerts"] == []
    assert report["recommendedActions"] == []


def test_build_ops_report_emits_threshold_alerts_for_high_caution_window():
    sqlite_db = DummyFeatureSQLiteWithLogs({})
    for index in range(10):
        sqlite_db.add_rag_answer_log(
            query_hash=f"hash-{index}",
            query_digest=f"Query {index}",
            source_type="all",
            retrieval_mode="hybrid",
            allow_external=False,
            result_status="ok",
            verification_status="caution" if index < 3 else "verified",
            needs_caution=index < 3,
            supported_claim_count=1,
            uncertain_claim_count=1 if index < 3 else 0,
            unsupported_claim_count=1 if index < 2 else 0,
            conflict_mentioned=index >= 3,
            rewrite_attempted=index < 3,
            rewrite_applied=index < 2,
            final_answer_source="conservative_fallback" if index == 0 else ("rewritten" if index == 1 else "original"),
            warning_count=1 if index < 3 else 0,
            source_count=2,
            evidence_count=2,
            answer_route={"route": "strong"},
            verification_route={"route": "strong"},
            rewrite_route={"route": "strong"} if index < 3 else {},
            warnings=["answer verification caution"] if index < 3 else [],
        )

    report = build_rag_ops_report(sqlite_db, limit=20, days=7)

    codes = {item["code"] for item in report["alerts"]}
    action_types = {item["actionType"] for item in report["recommendedActions"]}
    assert "rag_high_caution_rate" in codes
    assert "rag_high_unsupported_rate" in codes
    assert "rag_high_conservative_fallback_rate" in codes
    assert "inspect_rag_samples" in action_types
    assert "review_answer_routes" in action_types


def test_build_ops_report_emits_verification_route_recommendation_without_volume_threshold():
    sqlite_db = DummyFeatureSQLiteWithLogs({})
    sqlite_db.add_rag_answer_log(
        query_hash="hash-failed",
        query_digest="Verification failed case",
        source_type="all",
        retrieval_mode="hybrid",
        allow_external=False,
        result_status="ok",
        verification_status="failed",
        needs_caution=False,
        supported_claim_count=0,
        uncertain_claim_count=0,
        unsupported_claim_count=0,
        conflict_mentioned=False,
        rewrite_attempted=False,
        rewrite_applied=False,
        final_answer_source="original",
        warning_count=1,
        source_count=1,
        evidence_count=1,
        answer_route={"route": "strong"},
        verification_route={"route": "fallback-only"},
        rewrite_route={},
        warnings=["verification unavailable"],
    )

    report = build_rag_ops_report(sqlite_db, limit=10, days=7)

    assert report["alerts"][0]["code"] == "rag_verification_failed_or_skipped"
    assert report["recommendedActions"][0]["actionType"] == "inspect_verification_routes"
