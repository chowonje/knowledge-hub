from __future__ import annotations

from knowledge_hub.ai.memory_prefilter import (
    execute_memory_prefilter,
    memory_route_payload,
    normalize_memory_route_mode,
    normalize_memory_route_mode_details,
)
from knowledge_hub.core.models import SearchResult
from knowledge_hub.papers.prefilter import normalize_paper_memory_mode, normalize_paper_memory_mode_details


class _DummySQLite:
    def get_note(self, note_id):
        if note_id == "vault:Doc.md":
            return {"id": note_id, "file_path": "Doc.md", "source_type": "vault"}
        return None

    def list_memory_relations(self, **kwargs):
        _ = kwargs
        return []


class _DummyDatabase:
    def get_documents(
        self,
        *,
        filter_dict=None,
        limit=500,
        include_ids=True,
        include_documents=True,
        include_metadatas=True,
    ):
        _ = (limit, include_ids, include_documents, include_metadatas)
        if dict(filter_dict or {}) in ({"file_path": "Doc.md"}, {"source_type": "vault"}):
            return {
                "ids": ["doc-1"],
                "documents": ["memory route diagnostics are emitted for ask"],
                "metadatas": [
                    {
                        "title": "Doc",
                        "source_type": "vault",
                        "file_path": "Doc.md",
                        "section_title": "Memory Route",
                    }
                ],
            }
        return {"ids": [], "documents": [], "metadatas": []}


class _DummySearcher:
    def __init__(self):
        self.sqlite_db = _DummySQLite()
        self.database = _DummyDatabase()

    def search(self, *args, **kwargs):
        _ = (args, kwargs)
        return []


class _RankedVaultSearcher(_DummySearcher):
    def search(self, query, top_k=10, source_type=None, retrieval_mode="hybrid", alpha=0.7, metadata_filter=None):  # noqa: ARG002
        payload = dict(metadata_filter or {})
        if payload.get("file_path") == "Doc.md":
            return [
                SearchResult(
                    document="memory route diagnostics explain the purpose of memory-first retrieval.",
                    metadata={"title": "Doc", "source_type": "vault", "file_path": "Doc.md"},
                    distance=0.1,
                    score=0.82,
                    semantic_score=0.82,
                    lexical_score=0.0,
                    retrieval_mode="hybrid",
                    lexical_extras={},
                    document_id="doc-ranked-1",
                ),
                SearchResult(
                    document="extra verifier evidence that should be trimmed after memory-only gating.",
                    metadata={"title": "Doc Extra", "source_type": "vault", "file_path": "Doc.md"},
                    distance=0.2,
                    score=0.74,
                    semantic_score=0.74,
                    lexical_score=0.0,
                    retrieval_mode="hybrid",
                    lexical_extras={},
                    document_id="doc-ranked-2",
                ),
            ]
        return []


def test_memory_mode_normalization_supports_aliases_and_invalid_values():
    assert normalize_memory_route_mode("prefilter") == "compat"
    assert normalize_memory_route_mode("compat") == "compat"
    assert normalize_memory_route_mode("on") == "on"
    assert normalize_memory_route_mode("bogus") == "off"

    requested, effective, alias_applied = normalize_memory_route_mode_details("prefilter")
    assert requested == "prefilter"
    assert effective == "compat"
    assert alias_applied is True
    payload = memory_route_payload(requested_mode="prefilter", source_type="paper")
    assert payload["contractRole"] == "ask_retrieval_memory_prefilter"
    assert payload["requestedMode"] == "prefilter"
    assert payload["effectiveMode"] == "compat"
    assert payload["modeAliasApplied"] is True
    assert payload["aliasDeprecated"] is True

    assert normalize_paper_memory_mode("prefilter") == "compat"
    paper_requested, paper_effective, paper_alias_applied = normalize_paper_memory_mode_details("prefilter")
    assert paper_requested == "prefilter"
    assert paper_effective == "compat"
    assert paper_alias_applied is True


def test_execute_memory_prefilter_uses_chunk_fallback_when_prefilter_hits_but_ranked_results_are_empty(monkeypatch):
    monkeypatch.setattr(
        "knowledge_hub.ai.memory_prefilter.DocumentMemoryRetriever.search",
        lambda self, query, limit=10: [
            {
                "documentId": "vault:Doc.md",
                "sourceType": "vault",
                "sourceRef": "vault:Doc.md",
                "documentSummary": {"unitId": "memory-unit:vault:Doc.md"},
                "retrievalSignals": {
                    "temporalSignals": {"enabled": True, "matchedField": "document_date", "matchedValue": "2026-03-01T00:00:00+00:00"},
                    "updatesPreferred": True,
                },
            }
        ],
    )

    execution = execute_memory_prefilter(
        _DummySearcher(),
        query="memory route diagnostics",
        top_k=3,
        source_type="vault",
        retrieval_mode="hybrid",
        alpha=0.7,
        min_score=0.0,
        requested_mode="prefilter",
        metadata_filter=None,
        result_id_fn=lambda item: getattr(item, "document_id", ""),
    )

    assert execution.diagnostics["applied"] is True
    assert execution.diagnostics["requestedMode"] == "prefilter"
    assert execution.diagnostics["effectiveMode"] == "compat"
    assert execution.diagnostics["modeAliasApplied"] is True
    assert execution.diagnostics["fallbackUsed"] is True
    assert execution.diagnostics["reason"] == "memory_prefilter_chunk_fallback"
    assert execution.diagnostics["memoryInfluenceApplied"] is True
    assert execution.diagnostics["verificationCouplingApplied"] is False
    assert execution.diagnostics["temporalRouteApplied"] is True
    assert execution.diagnostics["updatesPreferred"] is True
    assert execution.diagnostics["memoryConfidence"] > 0.0
    assert execution.diagnostics["gatingDecision"] == "memory_plus_verify"
    assert execution.diagnostics["chunkExpansionTriggered"] is True
    assert execution.diagnostics["temporalSignals"]["matchedField"] == "document_date"
    assert len(execution.results) == 1
    assert execution.results[0].document_id == "doc-1"


def test_execute_memory_prefilter_uses_mixed_fallback_for_all_source_scope():
    class _MixedSearcher(_DummySearcher):
        def search(self, *args, **kwargs):
            _ = (args, kwargs)
            return [
                SearchResult(
                    document="kept",
                    metadata={"source_type": "web"},
                    distance=0.1,
                    score=0.6,
                    semantic_score=0.6,
                    lexical_score=0.0,
                    retrieval_mode="semantic",
                    lexical_extras={},
                    document_id="kept",
                )
            ]

    execution = execute_memory_prefilter(
        _MixedSearcher(),
        query="latest retrieval pipeline guide",
        top_k=3,
        source_type=None,
        retrieval_mode="hybrid",
        alpha=0.7,
        min_score=0.0,
        requested_mode="prefilter",
        metadata_filter=None,
        result_id_fn=lambda item: str(item),
    )

    assert len(execution.results) == 1
    assert execution.results[0].document == "kept"
    assert execution.diagnostics["fallbackUsed"] is True
    assert execution.diagnostics["mixedFallbackUsed"] is True
    assert execution.diagnostics["reason"] == "mixed_fallback_ranked"
    assert execution.diagnostics["applied"] is True
    assert execution.diagnostics["gatingDecision"] == "full_fallback"


def test_execute_memory_prefilter_uses_vault_chunk_fallback_when_memory_hits_are_missing(monkeypatch):
    monkeypatch.setattr(
        "knowledge_hub.ai.memory_prefilter.DocumentMemoryRetriever.search",
        lambda self, query, limit=10: [],
    )

    execution = execute_memory_prefilter(
        _DummySearcher(),
        query="memory route diagnostics",
        top_k=3,
        source_type="vault",
        retrieval_mode="hybrid",
        alpha=0.7,
        min_score=0.0,
        requested_mode="prefilter",
        metadata_filter=None,
        result_id_fn=lambda item: getattr(item, "document_id", ""),
    )

    assert execution.diagnostics["applied"] is True
    assert execution.diagnostics["fallbackUsed"] is True
    assert execution.diagnostics["reason"] == "vault_chunk_fallback"
    assert execution.diagnostics["staleMemorySignals"] == ["no_memory_hits"]
    assert len(execution.results) == 1
    assert execution.results[0].metadata["source_type"] == "vault"


def test_execute_memory_prefilter_uses_web_chunk_fallback_after_ranked_miss(monkeypatch):
    class _WebSQLite(_DummySQLite):
        def get_note(self, note_id):
            _ = note_id
            return None

    class _WebDatabase(_DummyDatabase):
        def get_documents(self, *, filter_dict=None, limit=500, include_ids=True, include_documents=True, include_metadatas=True):
            _ = (limit, include_ids, include_documents, include_metadatas)
            if dict(filter_dict or {}) == {"source_type": "web"}:
                return {
                    "ids": ["web-1"],
                    "documents": ["reference source watchlist update guide"],
                    "metadatas": [
                        {
                            "title": "Reference Source Watchlist",
                            "source_type": "web",
                            "url": "https://example.com/watchlist",
                            "source_ref": "https://example.com/watchlist",
                        }
                    ],
                }
            return {"ids": [], "documents": [], "metadatas": []}

    class _WebSearcher(_DummySearcher):
        def __init__(self):
            self.sqlite_db = _WebSQLite()
            self.database = _WebDatabase()

        def search(self, *args, **kwargs):
            _ = (args, kwargs)
            return []

    monkeypatch.setattr(
        "knowledge_hub.ai.memory_prefilter.DocumentMemoryRetriever.search",
        lambda self, query, limit=10: [
            {
                "documentId": "web:watchlist",
                "sourceType": "web",
                "sourceRef": "https://example.com/watchlist",
                "documentSummary": {"unitId": "memory-unit:web:watchlist"},
                "retrievalSignals": {"temporalSignals": {"enabled": True, "matchedField": "observed_at", "matchedValue": "2026-03-01T00:00:00+00:00"}},
            }
        ],
    )

    execution = execute_memory_prefilter(
        _WebSearcher(),
        query="latest reference source watchlist",
        top_k=3,
        source_type="web",
        retrieval_mode="hybrid",
        alpha=0.7,
        min_score=0.0,
        requested_mode="prefilter",
        metadata_filter=None,
        result_id_fn=lambda item: getattr(item, "document_id", ""),
    )

    assert execution.diagnostics["applied"] is True
    assert execution.diagnostics["fallbackUsed"] is True
    assert execution.diagnostics["reason"] == "web_chunk_fallback_success"
    assert len(execution.results) == 1
    assert execution.results[0].metadata["source_type"] == "web"


def test_execute_memory_prefilter_promotes_strong_ranked_memory_hit_to_memory_only(monkeypatch):
    monkeypatch.setattr(
        "knowledge_hub.ai.memory_prefilter.DocumentMemoryRetriever.search",
        lambda self, query, limit=10: [
            {
                "documentId": "vault:Doc.md",
                "sourceType": "vault",
                "sourceRef": "vault:Doc.md",
                "documentSummary": {"unitId": "memory-unit:vault:Doc.md"},
                "retrievalSignals": {
                    "temporalSignals": {"enabled": False, "matchedField": "", "matchedValue": ""},
                    "updatesPreferred": True,
                },
            },
            {
                "documentId": "vault:Doc.md",
                "sourceType": "vault",
                "sourceRef": "vault:Doc.md",
                "documentSummary": {"unitId": "memory-unit:vault:Doc.md:duplicate"},
                "retrievalSignals": {
                    "temporalSignals": {"enabled": False, "matchedField": "", "matchedValue": ""},
                    "updatesPreferred": True,
                },
            },
        ],
    )

    execution = execute_memory_prefilter(
        _RankedVaultSearcher(),
        query="memory-first retrieval의 목적을 설명해줘",
        top_k=4,
        source_type="vault",
        retrieval_mode="hybrid",
        alpha=0.7,
        min_score=0.0,
        requested_mode="prefilter",
        metadata_filter=None,
        result_id_fn=lambda item: getattr(item, "document_id", ""),
    )

    assert execution.diagnostics["reason"] == "matched_document_memory"
    assert execution.diagnostics["gatingDecision"] == "memory_only"
    assert execution.diagnostics["chunkExpansionTriggered"] is False
    assert execution.diagnostics["verifierBudgetUsed"] == 0
    assert len(execution.results) == 1
    assert execution.results[0].document_id == "doc-ranked-1"


def test_execute_memory_prefilter_keeps_how_query_in_verify_mode(monkeypatch):
    monkeypatch.setattr(
        "knowledge_hub.ai.memory_prefilter.DocumentMemoryRetriever.search",
        lambda self, query, limit=10: [
            {
                "documentId": "vault:Doc.md",
                "sourceType": "vault",
                "sourceRef": "vault:Doc.md",
                "documentSummary": {"unitId": "memory-unit:vault:Doc.md"},
                "retrievalSignals": {
                    "temporalSignals": {"enabled": False, "matchedField": "", "matchedValue": ""},
                    "updatesPreferred": True,
                },
            },
            {
                "documentId": "vault:Doc.md",
                "sourceType": "vault",
                "sourceRef": "vault:Doc.md",
                "documentSummary": {"unitId": "memory-unit:vault:Doc.md:duplicate"},
                "retrievalSignals": {
                    "temporalSignals": {"enabled": False, "matchedField": "", "matchedValue": ""},
                    "updatesPreferred": True,
                },
            },
        ],
    )

    execution = execute_memory_prefilter(
        _RankedVaultSearcher(),
        query="memory-first retrieval이 어떻게 동작하나?",
        top_k=4,
        source_type="vault",
        retrieval_mode="hybrid",
        alpha=0.7,
        min_score=0.0,
        requested_mode="prefilter",
        metadata_filter=None,
        result_id_fn=lambda item: getattr(item, "document_id", ""),
    )

    assert execution.diagnostics["gatingDecision"] == "memory_plus_verify"
    assert execution.diagnostics["chunkExpansionTriggered"] is True
    assert execution.diagnostics["verifierBudgetUsed"] >= 1


def test_execute_memory_prefilter_penalizes_vault_hub_noise_in_chunk_fallback(monkeypatch):
    class _VaultDatabase(_DummyDatabase):
        def get_documents(self, *, filter_dict=None, limit=500, include_ids=True, include_documents=True, include_metadatas=True):
            _ = (limit, include_ids, include_documents, include_metadatas)
            if dict(filter_dict or {}) == {"source_type": "vault"}:
                return {
                    "ids": ["hub-1", "note-1"],
                    "documents": [
                        "# Obsidian 전체 마인드맵 및 정리 아틀라스\n이 노트는 전체 Vault를 정리하는 지도다.",
                        "memory-first retrieval 설명: 먼저 memory unit을 좁혀 retrieval noise를 줄인다.",
                    ],
                    "metadatas": [
                        {"title": "Obsidian 전체 마인드맵 및 정리 아틀라스", "source_type": "vault", "file_path": "Atlas/Obsidian 전체 마인드맵 및 정리 아틀라스.md"},
                        {"title": "memory-first retrieval 설명", "source_type": "vault", "file_path": "Projects/AI/memory-first retrieval 설명.md"},
                    ],
                }
            return {"ids": [], "documents": [], "metadatas": []}

    class _VaultSearcher(_DummySearcher):
        def __init__(self):
            self.sqlite_db = _DummySQLite()
            self.database = _VaultDatabase()

    monkeypatch.setattr(
        "knowledge_hub.ai.memory_prefilter.DocumentMemoryRetriever.search",
        lambda self, query, limit=10: [],
    )

    execution = execute_memory_prefilter(
        _VaultSearcher(),
        query="memory-first retrieval의 목적을 한 문장으로 설명해줘",
        top_k=3,
        source_type="vault",
        retrieval_mode="hybrid",
        alpha=0.7,
        min_score=0.0,
        requested_mode="prefilter",
        metadata_filter=None,
        result_id_fn=lambda item: getattr(item, "document_id", ""),
    )

    assert execution.diagnostics["reason"] == "vault_chunk_fallback"
    assert execution.results[0].metadata["title"] == "memory-first retrieval 설명"


def test_execute_memory_prefilter_penalizes_refusal_paper_in_chunk_fallback(monkeypatch):
    class _PaperDatabase(_DummyDatabase):
        def get_documents(self, *, filter_dict=None, limit=500, include_ids=True, include_documents=True, include_metadatas=True):
            _ = (limit, include_ids, include_documents, include_metadatas)
            payload = dict(filter_dict or {})
            if payload == {"arxiv_id": "2509.06917"}:
                return {
                    "ids": ["paper2agent-1"],
                    "documents": ["제가 논문 전문을 직접 열람할 수 없어서 원문을 업로드해 주시면 정확한 요약을 드릴 수 있습니다."],
                    "metadatas": [{"title": "Paper2Agent", "source_type": "paper", "arxiv_id": "2509.06917"}],
                }
            if payload == {"arxiv_id": "2501.12948"}:
                return {
                    "ids": ["deepseek-1"],
                    "documents": ["paper summary path에서는 evidence packet이 요약 근거를 정리하고 인용을 고정하는 역할을 한다."],
                    "metadatas": [{"title": "DeepSeek-R1", "source_type": "paper", "arxiv_id": "2501.12948"}],
                }
            return {"ids": [], "documents": [], "metadatas": []}

    class _PaperSearcher(_DummySearcher):
        def __init__(self):
            self.sqlite_db = _DummySQLite()
            self.database = _PaperDatabase()

    monkeypatch.setattr(
        "knowledge_hub.ai.memory_prefilter.resolve_paper_memory_prefilter",
        lambda *args, **kwargs: {
            "requestedMode": "prefilter",
            "applied": True,
            "fallbackUsed": False,
            "matchedMemoryIds": ["paper-memory:2509.06917:x", "paper-memory:2501.12948:y"],
            "matchedPaperIds": ["2509.06917", "2501.12948"],
            "memoryRelationsUsed": [],
            "temporalSignals": {},
            "reason": "matched_cards",
        },
    )
    monkeypatch.setattr(
        "knowledge_hub.ai.memory_prefilter.PaperMemoryRetriever.search",
        lambda self, query, limit=10, include_refs=False: [
            {"paperId": "2509.06917", "memoryId": "paper-memory:2509.06917:x"},
            {"paperId": "2501.12948", "memoryId": "paper-memory:2501.12948:y"},
        ],
    )

    execution = execute_memory_prefilter(
        _PaperSearcher(),
        query="paper summary path에서 evidence packet이 왜 필요한가?",
        top_k=3,
        source_type="paper",
        retrieval_mode="hybrid",
        alpha=0.7,
        min_score=0.0,
        requested_mode="prefilter",
        metadata_filter=None,
        result_id_fn=lambda item: getattr(item, "document_id", ""),
        search_fn=lambda *args, **kwargs: [],
    )

    assert execution.results
    assert execution.results[0].metadata["title"] == "DeepSeek-R1"
