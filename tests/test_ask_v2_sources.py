from __future__ import annotations

from datetime import datetime, timezone

from knowledge_hub.ai.ask_v2_support import build_project_cards
from knowledge_hub.application.query_frame import build_query_frame
from knowledge_hub.document_memory import DocumentMemoryBuilder
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.ai.ask_v2 import AskV2Service
from knowledge_hub.ai.rag import RAGSearcher
from knowledge_hub.web.ingest import make_web_note_id
from tests.test_rag_search import DummyEmbedder, DummyVectorDB, FakeLLM
from tests.test_paper_ask_v2 import _seed_web_document_memory


class _SearchForbiddenVectorDB:
    def __init__(self):
        self.search_called = False

    def search(self, query_embedding, top_k: int, filter_dict=None):  # noqa: ANN001
        _ = (query_embedding, top_k, filter_dict)
        self.search_called = True
        raise AssertionError("ask v2 should not use broad vector search")

    def get_documents(self, filter_dict=None, limit=500, include_ids=True, include_documents=True, include_metadatas=True):  # noqa: ANN001
        _ = (filter_dict, limit, include_ids, include_documents, include_metadatas)
        return {"documents": [], "metadatas": [], "ids": []}


def _build_searcher(db: SQLiteDatabase) -> tuple[RAGSearcher, _SearchForbiddenVectorDB]:
    vector_db = _SearchForbiddenVectorDB()
    searcher = RAGSearcher(
        DummyEmbedder(),
        vector_db,
        llm=FakeLLM(),
        sqlite_db=db,
    )
    searcher._verify_answer = lambda **kwargs: {  # type: ignore[method-assign]
        "status": "passed",
        "supportedClaimCount": 1,
        "unsupportedClaimCount": 0,
        "uncertainClaimCount": 0,
        "conflictMentioned": False,
        "needsCaution": False,
        "warnings": [],
    }
    searcher._rewrite_answer = lambda **kwargs: (kwargs["answer"], {"attempted": False, "applied": False, "warnings": []})  # type: ignore[method-assign]
    searcher._apply_conservative_fallback_if_needed = lambda **kwargs: (  # type: ignore[method-assign]
        kwargs["answer"],
        kwargs["rewrite_meta"],
        kwargs["verification"],
    )
    return searcher, vector_db


def test_generate_answer_uses_web_card_v2_and_flags_observed_at_only_temporal(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    url = "https://example.com/reference-watchlist"
    note_id = make_web_note_id(url)
    db.upsert_note(
        note_id=note_id,
        title="Reference Source Watchlist",
        content=(
            "# Reference Source Watchlist\n\n"
            "## Overview\n\n"
            "This page tracks stable reference sources for retrieval evaluation.\n"
        ),
        source_type="web",
        metadata={
            "canonical_url": url,
            "observed_at": "2026-03-01T00:00:00+00:00",
        },
    )
    DocumentMemoryBuilder(db).build_and_store_web(canonical_url=url)
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="latest reference watchlist",
        source_type="web",
        top_k=3,
    )

    assert vector_db.search_called is False
    assert payload["status"] == "no_result"
    assert payload["v2"]["routing"]["sourceKind"] == "web"
    assert payload["v2"]["evidenceVerification"]["anchorIdsUsed"]
    assert "temporal_version_grounding" in payload["v2"]["evidenceVerification"]["unsupportedFields"]
    assert payload["v2"]["fallback"]["reason"].startswith("ask_v2_weak_evidence:")
    assert payload["queryFrame"]["domain_key"] == "web_knowledge"
    assert payload["queryFrame"]["family"] == "temporal_update"
    assert payload["evidencePolicy"]["policyKey"] == "web_temporal_update_policy"
    assert payload["familyRouteDiagnostics"]["temporalSignalsApplied"] is True


def test_generate_answer_rebuilds_stale_web_card_when_document_summary_is_newer(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    url = "https://example.com/reference-watchlist"
    note_id = make_web_note_id(url)
    db.upsert_note(
        note_id=note_id,
        title="Reference Source Watchlist",
        content="# Reference Source Watchlist\n\nBaseline summary.\n",
        source_type="web",
        metadata={"canonical_url": url},
    )
    DocumentMemoryBuilder(db).build_and_store_web(canonical_url=url)
    searcher, vector_db = _build_searcher(db)

    initial = searcher.generate_answer(
        query="reference watchlist summary",
        source_type="web",
        top_k=3,
        metadata_filter={"canonical_url": url},
    )
    assert initial["v2"]["routing"]["sourceKind"] == "web"

    db.conn.execute(
        "UPDATE web_cards_v2 SET updated_at = ? WHERE document_id = ?",
        ("2026-03-01T00:00:00+00:00", note_id),
    )
    db.conn.execute(
        """
        UPDATE document_memory_units
        SET contextual_summary = ?, updated_at = ?
        WHERE document_id = ? AND unit_type = 'document_summary'
        """,
        (
            "Refreshed web summary with newer guidance.",
            datetime(2026, 3, 28, tzinfo=timezone.utc).isoformat(),
            note_id,
        ),
    )
    db.conn.commit()

    payload = searcher.generate_answer(
        query="reference watchlist summary",
        source_type="web",
        top_k=3,
        metadata_filter={"canonical_url": url},
    )

    refreshed = db.get_web_card_v2_by_url(url)
    assert vector_db.search_called is False
    assert refreshed is not None
    assert refreshed["page_core"] == "Refreshed web summary with newer guidance."
    assert payload["v2"]["routing"]["selected_card_ids"]


def test_select_web_cards_rebuilds_from_resolved_doc_id_metadata(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    url = "https://example.com/reference/vector-search-rerank"
    note_id = make_web_note_id(url)
    db.upsert_note(
        note_id=note_id,
        title="Vector Search Rerank Guide",
        content="# Vector Search Rerank Guide\n\nThis guide explains rerank as a precision layer.\n",
        source_type="web",
        metadata={"canonical_url": url},
    )
    DocumentMemoryBuilder(db).build_and_store_web(canonical_url=url)
    searcher, _ = _build_searcher(db)
    service = AskV2Service(searcher)
    frame = build_query_frame(
        domain_key="web_knowledge",
        source_type="web",
        family="reference_explainer",
        query_intent="definition",
        answer_mode="concise_summary",
        resolved_source_ids=[note_id],
        expanded_terms=["rerank", "guide"],
        confidence=0.9,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="web_reference_explainer_policy",
        metadata_filter={"source_type": "web", "document_id": note_id, "reference_only": True},
    )
    route = service._route(query="rerank guide", source_type="web", query_frame=frame.to_dict())

    cards = service._select_web_cards(
        query="rerank guide",
        route=route,
        limit=3,
        metadata_filter={"document_id": note_id},
        query_frame=frame.to_dict(),
    )

    assert cards
    assert cards[0]["canonical_url"] == url
    assert db.get_web_card_v2_by_url(url) is not None


def test_execute_web_definition_prefers_stored_card_anchors_before_claim_cards(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    url = "https://example.com/reference/vector-search-rerank"
    _seed_web_document_memory(db, url, with_temporal_markers=False)
    searcher, _ = _build_searcher(db)
    service = AskV2Service(searcher)
    frame = build_query_frame(
        domain_key="web_knowledge",
        source_type="web",
        family="reference_explainer",
        query_intent="definition",
        answer_mode="concise_summary",
        expanded_terms=["rerank", "guide"],
        confidence=0.9,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="web_reference_explainer_policy",
        metadata_filter={"source_type": "web", "canonical_url": url, "reference_only": True},
    )

    monkeypatch.setattr(
        service,
        "_load_claim_cards",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("stored web card anchors should be used before claim cards")),
    )

    pipeline_result, evidence_packet = service.execute(
        query="vector search rerank guide",
        top_k=3,
        source_type="web",
        retrieval_mode="hybrid",
        alpha=0.5,
        allow_external=False,
        metadata_filter=frame.metadata_filter,
        query_frame=frame.to_dict(),
    )

    assert pipeline_result.results
    assert evidence_packet.evidence


def test_select_web_cards_rebuilds_from_resolved_doc_id_metadata_url_fallback(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    url = "https://example.com/blog/vector-search-rerank"
    note_id = make_web_note_id(url)
    db.upsert_note(
        note_id=note_id,
        title="Vector Search Rerank Guide",
        content="# Vector Search Rerank Guide\n\nThis guide explains rerank as a precision layer.\n",
        source_type="web",
        metadata={"url": url},
    )
    DocumentMemoryBuilder(db).build_and_store_web(canonical_url=url)
    searcher, _ = _build_searcher(db)
    service = AskV2Service(searcher)
    frame = build_query_frame(
        domain_key="web_knowledge",
        source_type="web",
        family="reference_explainer",
        query_intent="definition",
        answer_mode="concise_summary",
        resolved_source_ids=[note_id],
        expanded_terms=["rerank", "guide"],
        confidence=0.9,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="web_reference_explainer_policy",
        metadata_filter={"source_type": "web", "document_id": note_id, "reference_only": True},
    )
    route = service._route(query="rerank guide", source_type="web", query_frame=frame.to_dict())

    cards = service._select_web_cards(
        query="rerank guide",
        route=route,
        limit=3,
        metadata_filter={"document_id": note_id},
        query_frame=frame.to_dict(),
    )

    assert cards
    assert cards[0]["canonical_url"] == url


def test_select_web_cards_merges_materialized_fallback_cards_when_candidates_exist(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    searcher, _ = _build_searcher(db)
    service = AskV2Service(searcher)
    frame = build_query_frame(
        domain_key="web_knowledge",
        source_type="web",
        family="reference_explainer",
        query_intent="definition",
        answer_mode="concise_summary",
        expanded_terms=["rerank", "version grounding"],
        confidence=0.9,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="web_reference_explainer_policy",
        metadata_filter={"source_type": "web", "reference_only": True},
    )
    route = service._route(query="version grounding rerank", source_type="web", query_frame=frame.to_dict())
    generic_card = {
        "document_id": "web_generic",
        "title": "General Search Notes",
        "page_core": "General retrieval notes.",
        "topic_core": "Search concepts.",
        "result_core": "",
        "version_core": "",
        "search_text": "general notes",
    }
    fallback_card = {
        "document_id": "web_direct",
        "canonical_url": "https://example.com/vector-search-rerank",
        "title": "Version Grounding Rerank Guide",
        "page_core": "Version grounding explains why rerank needs anchored evidence.",
        "topic_core": "Rerank and version grounding guide.",
        "result_core": "",
        "version_core": "updated 2026-04-01",
        "search_text": "version grounding rerank guide",
    }

    monkeypatch.setattr(service.sqlite_db, "search_web_cards_v2", lambda *args, **kwargs: [dict(generic_card)])
    monkeypatch.setattr(service, "_fallback_web_urls", lambda **kwargs: [fallback_card["canonical_url"]])
    monkeypatch.setattr(service, "_ensure_cards_for_web", lambda urls: [dict(fallback_card)])
    monkeypatch.setattr(
        service,
        "_score_card",
        lambda card, **kwargs: 10.0 if card.get("document_id") == "web_direct" else 1.0,
    )

    cards = service._select_web_cards(
        query="version grounding rerank",
        route=route,
        limit=3,
        metadata_filter={"source_type": "web"},
        query_frame=frame.to_dict(),
    )

    assert [item["document_id"] for item in cards][:2] == ["web_direct", "web_generic"]


def test_select_web_cards_prefers_materialized_fallback_for_duplicate_document_id(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    searcher, _ = _build_searcher(db)
    service = AskV2Service(searcher)
    frame = build_query_frame(
        domain_key="web_knowledge",
        source_type="web",
        family="temporal_update",
        query_intent="temporal",
        answer_mode="timeline_compare",
        expanded_terms=["observed_at", "version grounding"],
        confidence=0.9,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="web_temporal_update_policy",
        metadata_filter={"source_type": "web", "latest_only": True, "temporal_required": True},
    )
    route = service._route(query="version grounding update", source_type="web", query_frame=frame.to_dict())
    generic_card = {
        "document_id": "web_same",
        "canonical_url": "https://example.com/web-card-v2",
        "title": "Generic Product Notes",
        "page_core": "Generic notes.",
        "topic_core": "Overview.",
        "result_core": "",
        "version_core": "",
        "search_text": "generic notes",
    }
    fallback_card = {
        "document_id": "web_same",
        "canonical_url": "https://example.com/web-card-v2",
        "title": "Version Grounding Update Notes",
        "page_core": "Observed_at alone is weak for latest answers without version grounding.",
        "topic_core": "Version grounding update notes.",
        "result_core": "",
        "version_core": "observed_at 2026-04-01",
        "search_text": "version grounding observed_at update",
    }

    monkeypatch.setattr(service.sqlite_db, "search_web_cards_v2", lambda *args, **kwargs: [dict(generic_card)])
    monkeypatch.setattr(service, "_fallback_web_urls", lambda **kwargs: [fallback_card["canonical_url"]])
    monkeypatch.setattr(service, "_ensure_cards_for_web", lambda urls: [dict(fallback_card)])
    monkeypatch.setattr(
        service,
        "_score_card",
        lambda card, **kwargs: 10.0 if "Version Grounding" in str(card.get("title")) else 1.0,
    )

    cards = service._select_web_cards(
        query="version grounding update",
        route=route,
        limit=3,
        metadata_filter={"source_type": "web"},
        query_frame=frame.to_dict(),
    )

    assert len([item for item in cards if item["document_id"] == "web_same"]) == 1
    assert cards[0]["title"] == "Version Grounding Update Notes"


def test_fallback_web_urls_uses_like_scan_and_on_demand_memory_materialization(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    url = "https://example.com/blog/vector-search-rerank"
    note_id = make_web_note_id(url)
    db.upsert_note(
        note_id=note_id,
        title="Vector Search Rerank Guide",
        content=(
            "# Vector Search Rerank Guide\n\n"
            "This guide explains rerank as a precision layer and why version grounding matters.\n"
        ),
        source_type="web",
        metadata={"url": url},
    )
    searcher, _ = _build_searcher(db)
    service = AskV2Service(searcher)
    frame = build_query_frame(
        domain_key="web_knowledge",
        source_type="web",
        family="reference_explainer",
        query_intent="definition",
        answer_mode="concise_summary",
        expanded_terms=["rerank", "version grounding"],
        confidence=0.9,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="web_reference_explainer_policy",
        metadata_filter={"source_type": "web", "reference_only": True},
    )
    route = service._route(query="rerank precision", source_type="web", query_frame=frame.to_dict())

    monkeypatch.setattr(service.sqlite_db, "search_notes", lambda *args, **kwargs: [])

    urls = service._fallback_web_urls(
        query="rerank precision",
        route=route,
        limit=3,
        lookup_forms=["rerank", "version grounding"],
    )
    cards = service._ensure_cards_for_web(urls)

    assert url in urls
    assert cards
    assert cards[0]["canonical_url"] == url
    assert db.get_document_memory_summary(note_id) is not None
    assert db.get_web_card_v2_by_url(url) is not None


def test_select_web_cards_includes_internal_reference_cards_for_product_specific_queries(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    searcher, _ = _build_searcher(db)
    service = AskV2Service(searcher)
    frame = build_query_frame(
        domain_key="web_knowledge",
        source_type="web",
        family="relation_explainer",
        query_intent="relation",
        answer_mode="concept_explainer",
        expanded_terms=["ontology-first routing", "ontology", "routing"],
        confidence=0.9,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="web_relation_explainer_policy",
        metadata_filter={"source_type": "web", "reference_only": True, "internal_reference_preferred": True},
    )
    route = service._route(
        query="web ask v2에서 ontology-first routing은 언제 도움이 되나?",
        source_type="web",
        query_frame=frame.to_dict(),
    )

    cards = service._select_web_cards(
        query="web ask v2에서 ontology-first routing은 언제 도움이 되나?",
        route=route,
        limit=3,
        metadata_filter=frame.metadata_filter,
        query_frame=frame.to_dict(),
    )

    assert cards
    assert cards[0]["diagnostics"]["internalReference"] is True
    assert "ontology-first routing" in cards[0]["title"].casefold()


def test_generate_answer_uses_internal_web_reference_cards_for_product_specific_queries(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="web ask v2에서 ontology-first routing은 언제 도움이 되나?",
        source_type="web",
        top_k=3,
    )

    assert vector_db.search_called is False
    assert payload["queryFrame"]["family"] == "relation_explainer"
    assert payload["familyRouteDiagnostics"]["internalReferenceApplied"] is True
    assert payload["sources"]
    assert "ontology-first routing" in payload["sources"][0]["title"].casefold()


def test_execute_prefers_internal_web_reference_anchor_over_external_claim_anchor(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    searcher, _ = _build_searcher(db)
    service = AskV2Service(searcher)

    internal_card = {
        "card_id": "web-internal-ref:web_ontology_routing",
        "document_id": "web_internal_ref:web_ontology_routing",
        "canonical_url": "internal://docs/PROJECT_STATE.md#L1",
        "title": "Web ask v2 notes: ontology-first routing helps entity-heavy web queries",
        "page_core": "Ontology-first routing helps entity-heavy web queries.",
        "topic_core": "Ontology-first routing helps entity-heavy web queries.",
        "result_core": "",
        "version_core": "internal reference",
        "search_text": "ontology-first routing entity-heavy web queries",
        "quality_flag": "ok",
        "document_date": "2026-04-05T00:00:00+00:00",
        "observed_at": "2026-04-05T00:00:00+00:00",
        "updated_at": "2026-04-05T00:00:00+00:00",
        "diagnostics": {"internalReference": True},
        "anchors": [
            {
                "anchor_id": "web-internal-ref:web_ontology_routing:anchor",
                "card_id": "web-internal-ref:web_ontology_routing",
                "excerpt": "Ontology-first routing helps entity-heavy web queries.",
                "score": 0.99,
                "source_url": "internal://docs/PROJECT_STATE.md#L1",
                "title": "Web ask v2 notes: ontology-first routing helps entity-heavy web queries",
                "document_date": "2026-04-05T00:00:00+00:00",
                "observed_at": "2026-04-05T00:00:00+00:00",
            }
        ],
    }
    external_card = {
        "card_id": "web-card:external",
        "document_id": "web:external",
        "canonical_url": "https://example.com/external",
        "title": "External Retrieval Article",
        "page_core": "External retrieval notes.",
        "topic_core": "External retrieval notes.",
        "result_core": "",
        "version_core": "",
        "search_text": "external retrieval notes",
        "quality_flag": "ok",
    }
    external_claim_cards = [
        {
            "claim_card_id": "claim-card:external",
            "claim_id": "claim:external",
            "source_kind": "web",
            "source_id": "web:external",
            "source_card_id": "web-card:external",
            "claim_type": "definition",
            "origin": "materialized_v1",
            "trust_level": "high",
            "anchors": [
                {
                    "anchor_id": "claim-anchor:external",
                    "claim_id": "claim:external",
                    "card_id": "web-card:external",
                    "excerpt": "External article says routing matters.",
                    "score": 0.99,
                    "source_url": "https://example.com/external",
                    "title": "External Retrieval Article",
                }
            ],
        }
    ]

    monkeypatch.setattr(
        AskV2Service,
        "_select_web_cards",
        lambda self, **kwargs: [internal_card, external_card],
    )
    monkeypatch.setattr(
        AskV2Service,
        "_load_claim_cards",
        lambda self, **kwargs: external_claim_cards,
    )

    frame = build_query_frame(
        domain_key="web_knowledge",
        source_type="web",
        family="relation_explainer",
        query_intent="relation",
        answer_mode="concept_explainer",
        expanded_terms=["ontology-first routing", "ontology", "routing"],
        confidence=0.9,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="web_relation_explainer_policy",
        metadata_filter={"source_type": "web", "reference_only": True, "internal_reference_preferred": True},
    )

    pipeline_result, evidence_packet = service.execute(
        query="web ask v2에서 ontology-first routing은 언제 도움이 되나?",
        top_k=3,
        source_type="web",
        retrieval_mode="hybrid",
        alpha=0.5,
        allow_external=False,
        metadata_filter=frame.metadata_filter,
        query_frame=frame.to_dict(),
    )

    assert pipeline_result.results
    assert "ontology-first routing" in pipeline_result.results[0].metadata["title"].casefold()
    assert evidence_packet.evidence
    assert "ontology-first routing" in evidence_packet.evidence[0]["title"].casefold()


def test_generate_answer_uses_vault_card_v2(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    note_id = "vault:Projects/PolicyEngineAudit.md"
    db.upsert_note(
        note_id=note_id,
        title="Policy Engine Audit",
        content=(
            "# Policy Engine Audit\n\n"
            "## Design\n\n"
            "This note explains retrieval diagnostics and audit logging for the policy engine.\n"
        ),
        file_path="Projects/PolicyEngineAudit.md",
        source_type="vault",
        metadata={"tags": ["policy-engine", "audit-log"]},
    )
    DocumentMemoryBuilder(db).build_and_store_note(note_id=note_id)
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="PolicyEngineAudit audit-log retrieval",
        source_type="vault",
        top_k=3,
    )

    assert vector_db.search_called is False
    assert payload["v2"]["routing"]["sourceKind"] == "vault"
    assert payload["v2"]["routing"]["mode"] == "card-first"
    assert payload["v2"]["evidenceVerification"]["anchorIdsUsed"]


def test_generate_answer_uses_ephemeral_project_cards_without_persistence(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("- Keep repo context ephemeral\n", encoding="utf-8")
    (repo / "README.md").write_text("# Repo\n\nProject overview.\n", encoding="utf-8")
    src = repo / "src"
    src.mkdir()
    (src / "agent.py").write_text(
        "from helpers import fetch_context\n\n"
        "def build_context():\n"
        "    return fetch_context()\n",
        encoding="utf-8",
    )

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="Explain src/agent.py build_context fetch_context architecture",
        source_type="project",
        top_k=3,
        metadata_filter={"repo_path": str(repo)},
    )

    assert vector_db.search_called is False
    assert payload["v2"]["routing"]["sourceKind"] == "project"
    assert payload["v2"]["routing"]["mode"] == "card-first"
    assert payload["v2"]["evidenceVerification"]["anchorIdsUsed"]
    assert any(item["slotCoverage"]["symbolOwnerCore"] == "complete" for item in payload["v2"]["cardSelection"]["selected"])
    assert any(item["slotCoverage"]["callFlowCore"] == "complete" for item in payload["v2"]["cardSelection"]["selected"])
    assert db.search_notes("agent.py", limit=5) == []


def test_generate_answer_project_path_does_not_require_claim_card_builder(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Repo\n\nProject overview.\n", encoding="utf-8")
    src = repo / "src"
    src.mkdir()
    (src / "agent.py").write_text(
        "from helpers import fetch_context\n\n"
        "def build_context():\n"
        "    return fetch_context()\n",
        encoding="utf-8",
    )

    class _ForbiddenClaimCardBuilder:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("project path should not instantiate ClaimCardBuilder")

    monkeypatch.setattr("knowledge_hub.ai.ask_v2.ClaimCardBuilder", _ForbiddenClaimCardBuilder)

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="Explain src/agent.py build_context fetch_context architecture",
        source_type="project",
        top_k=3,
        metadata_filter={"repo_path": str(repo)},
    )

    assert vector_db.search_called is False
    assert payload["v2"]["routing"]["sourceKind"] == "project"
    assert payload["claimCards"]
    assert all(item["sourceKind"] == "project" for item in payload["claimCards"])


def test_build_project_cards_extracts_file_role_and_stable_anchors():
    workspace_files = [
        {
            "path": "/tmp/repo/src/main.py",
            "relative_path": "src/main.py",
            "reason": "goal matched architecture file",
            "snippet": (
                "from helpers import fetch_context\n\n"
                "def build_context():\n"
                "    return fetch_context()\n"
            ),
        }
    ]

    first = build_project_cards(workspace_files=workspace_files, repo_path="/tmp/repo")
    second = build_project_cards(workspace_files=workspace_files, repo_path="/tmp/repo")

    assert first[0]["file_role_core"] == "entrypoint"
    assert first[0]["slot_coverage"]["fileRoleCore"] == "complete"
    assert first[0]["slot_coverage"]["symbolOwnerCore"] == "complete"
    assert first[0]["slot_coverage"]["callFlowCore"] == "complete"
    assert first[0]["slot_coverage"]["integrationBoundaryCore"] == "complete"
    assert [item["anchor_id"] for item in first[0]["anchors"]] == [item["anchor_id"] for item in second[0]["anchors"]]


def test_generate_answer_project_architecture_prefers_entrypoint_over_readme_and_tests(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Repo\n\nArchitecture overview.\n", encoding="utf-8")
    src = repo / "src"
    src.mkdir()
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (src / "main.py").write_text(
        "from service import build_pipeline\n\n"
        "def main():\n"
        "    return build_pipeline()\n",
        encoding="utf-8",
    )
    (src / "service.py").write_text(
        "def build_pipeline():\n"
        "    return 'pipeline'\n",
        encoding="utf-8",
    )
    (tests_dir / "test_service.py").write_text(
        "from src.service import build_pipeline\n\n"
        "def test_build_pipeline():\n"
        "    assert build_pipeline()\n",
        encoding="utf-8",
    )

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="service architecture entrypoint pipeline",
        source_type="project",
        top_k=3,
        metadata_filter={"repo_path": str(repo)},
    )

    assert vector_db.search_called is False
    assert payload["v2"]["routing"]["sourceKind"] == "project"
    assert payload["v2"]["cardSelection"]["selected"][0]["title"] == "src/main.py"
    assert payload["v2"]["projectSignals"]["selectedFileRoles"][0] == "entrypoint"


def test_generate_answer_project_symbol_owner_prefers_definition_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "src"
    src.mkdir()
    (repo / "README.md").write_text("# Repo\n\nHelper docs.\n", encoding="utf-8")
    (src / "helpers.py").write_text(
        "def build_context():\n"
        "    return 'ctx'\n",
        encoding="utf-8",
    )
    (src / "agent.py").write_text(
        "from helpers import build_context\n\n"
        "def run_agent():\n"
        "    return build_context()\n",
        encoding="utf-8",
    )

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="where is build_context defined",
        source_type="project",
        top_k=3,
        metadata_filter={"repo_path": str(repo)},
    )

    assert vector_db.search_called is False
    assert payload["v2"]["cardSelection"]["selected"][0]["title"] == "src/helpers.py"
    assert "symbol_owner" in payload["v2"]["evidenceVerification"]["anchorRolesUsed"]


def test_generate_answer_project_weak_repo_cards_are_marked_weak(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Repo\n\nGeneral notes only.\n", encoding="utf-8")

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="service architecture pipeline integration",
        source_type="project",
        top_k=3,
        metadata_filter={"repo_path": str(repo)},
    )

    assert vector_db.search_called is False
    assert payload["status"] == "no_result"
    assert payload["v2"]["routing"]["sourceKind"] == "project"
    assert payload["v2"]["evidenceVerification"]["verificationStatus"] == "weak"
    assert "weak_project_slots" in payload["v2"]["evidenceVerification"]["unsupportedFields"]
    assert payload["v2"]["fallback"]["used"] is True
    assert payload["v2"]["fallback"]["reason"].startswith("ask_v2_weak_evidence:")


def test_generate_answer_youtube_summary_uses_youtube_scope_and_payload_contract(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    url = "https://www.youtube.com/watch?v=s4xnZMiEIJc"
    note_id = make_web_note_id(url)
    db.upsert_note(
        note_id=note_id,
        title="DSBA Study Agent AI 2주차 Cognitive Architectures (2)",
        content=(
            "# DSBA Study Agent AI 2주차 Cognitive Architectures (2)\n\n"
            "## Description\n\n"
            "Agent AI 2주차 Cognitive Architectures (2)\n\n"
            "## Transcript\n\n"
            "[00:00:03] 메모리와 planning을 소개한다.\n"
        ),
        source_type="web",
        metadata={
            "canonical_url": url,
            "video_id": "s4xnZMiEIJc",
            "media_platform": "youtube",
            "source_channel_type": "youtube_video",
            "channel_name": "서울대학교 산업공학과 DSBA 연구실",
        },
    )
    DocumentMemoryBuilder(db).build_and_store_web(canonical_url=url)
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query=f"{url} 영상 요약해줘",
        source_type="youtube",
        top_k=3,
    )

    assert vector_db.search_called is False
    assert payload["queryFrame"]["domain_key"] == "youtube_knowledge"
    assert payload["queryFrame"]["family"] == "video_lookup"
    assert payload["evidencePolicy"]["policyKey"] == "youtube_video_lookup_policy"
    assert payload["familyRouteDiagnostics"]["videoScopeApplied"] is True
    assert payload["familyRouteDiagnostics"]["runtimeUsed"] == "ask_v2"
    assert payload["retrievalObjectsAvailable"] == ["RawEvidenceUnit", "DocSummary", "SectionCard"]
    assert payload["sources"]
    assert payload["sources"][0]["title"] == "DSBA Study Agent AI 2주차 Cognitive Architectures (2)"


def test_select_web_cards_source_youtube_filters_out_generic_web_cards(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    youtube_url = "https://www.youtube.com/watch?v=s4xnZMiEIJc"
    youtube_note_id = make_web_note_id(youtube_url)
    db.upsert_note(
        note_id=youtube_note_id,
        title="DSBA Study Agent AI 2주차 Cognitive Architectures (2)",
        content="# Video\n\n## Transcript\n[00:00:03] 메모리를 설명한다.\n",
        source_type="web",
        metadata={
            "canonical_url": youtube_url,
            "video_id": "s4xnZMiEIJc",
            "media_platform": "youtube",
            "source_channel_type": "youtube_video",
        },
    )
    other_url = "https://example.com/blog/memory-planning"
    other_note_id = make_web_note_id(other_url)
    db.upsert_note(
        note_id=other_note_id,
        title="Memory Planning Blog",
        content="# Blog\n\nThis is not a youtube video.\n",
        source_type="web",
        metadata={
            "canonical_url": other_url,
            "media_platform": "web",
            "source_channel_type": "reference_watchlist",
        },
    )
    DocumentMemoryBuilder(db).build_and_store_web(canonical_url=youtube_url)
    DocumentMemoryBuilder(db).build_and_store_web(canonical_url=other_url)
    searcher, _ = _build_searcher(db)
    service = AskV2Service(searcher)
    frame = build_query_frame(
        domain_key="youtube_knowledge",
        source_type="youtube",
        family="video_lookup",
        query_intent="video_lookup",
        answer_mode="video_scoped_answer",
        resolved_source_ids=[youtube_note_id, youtube_url, "s4xnZMiEIJc"],
        expanded_terms=["memory", "planning", "video summary"],
        confidence=0.9,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="youtube_video_lookup_policy",
        metadata_filter={
            "source_type": "web",
            "media_platform": "youtube",
            "document_id": youtube_note_id,
            "canonical_url": youtube_url,
            "youtube_only": True,
        },
    )
    route = service._route(query="memory planning 영상 요약해줘", source_type="youtube", query_frame=frame.to_dict())

    cards = service._select_web_cards(
        query="memory planning 영상 요약해줘",
        route=route,
        limit=3,
        metadata_filter=frame.metadata_filter,
        query_frame=frame.to_dict(),
    )

    assert cards
    assert cards[0]["document_id"] == youtube_note_id
    assert all(card["canonical_url"] == youtube_url for card in cards)


def test_generate_answer_vault_summary_uses_vault_scope_and_payload_contract(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    note_id = "vault:rag-quality"
    file_path = "AI/KnowledgeOS/rag-quality.md"
    db.upsert_note(
        note_id=note_id,
        title="RAG Retrieval Quality",
        content=(
            "# RAG Retrieval Quality\n\n"
            "## Overview\n\n"
            "The most common problem is weak query understanding and poor evidence prioritization.\n"
        ),
        source_type="vault",
        file_path=file_path,
        metadata={"tags": ["rag", "retrieval", "quality"]},
    )
    DocumentMemoryBuilder(db).build_and_store_note(note_id=note_id)
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="RAG 검색 품질을 떨어뜨리는 가장 흔한 원인은 무엇인가?",
        source_type="vault",
        top_k=3,
    )

    assert vector_db.search_called is False
    assert payload["v2"]["routing"]["sourceKind"] == "vault"
    assert payload["queryFrame"]["domain_key"] == "vault_knowledge"
    assert payload["queryFrame"]["family"] == "vault_explainer"
    assert payload["evidencePolicy"]["policyKey"] == "vault_explainer_policy"
    assert payload["retrievalObjectsAvailable"] == ["RawEvidenceUnit", "DocSummary"]
    assert payload["familyRouteDiagnostics"]["runtimeUsed"] == "ask_v2"
    assert payload["sources"]
    assert payload["sources"][0]["title"] == "RAG Retrieval Quality"


def test_generate_answer_vault_missing_explicit_path_returns_no_result(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    db.upsert_note(
        note_id="vault:sheet",
        title="Sheet",
        content="# Sheet\n\nUnrelated reparameterization trick notes.\n",
        source_type="vault",
        file_path="Notes/sheet.md",
        metadata={},
    )
    DocumentMemoryBuilder(db).build_and_store_note(note_id="vault:sheet")
    vector_db = DummyVectorDB(
        [
            {
                "id": "unrelated",
                "document": "Unrelated vault evidence",
                "metadata": {"source_type": "vault", "file_path": "Notes/sheet.md", "title": "Sheet"},
                "distance": 0.01,
            }
        ]
    )
    searcher = RAGSearcher(DummyEmbedder(), vector_db, llm=FakeLLM(), sqlite_db=db)

    payload = searcher.generate_answer(
        query="Missing/NoSuchNote-999.md 파일의 핵심 내용을 보여줘",
        source_type="vault",
        top_k=3,
    )

    assert payload["status"] == "no_result"
    assert payload["sources"] == []
    assert payload["queryFrame"]["family"] == "note_lookup"
    assert payload["queryFrame"]["answer_mode"] == "abstain"
    assert payload["familyRouteDiagnostics"]["runtimeUsed"] == "ask_v2"
    assert payload["v2"]["runtimeExecution"]["used"] == "ask_v2"
    assert payload["v2"]["runtimeExecution"]["fallbackReason"] == "scoped_vault_no_result"
    assert payload["familyRouteDiagnostics"]["vaultScopeApplied"] is True
    assert payload["familyRouteDiagnostics"]["metadataFilterApplied"]["vault_scope_missing"] is True


def test_generate_answer_vault_missing_explicit_note_id_returns_no_result(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    db.upsert_note(
        note_id="vault:sheet",
        title="Sheet",
        content="# Sheet\n\nUnrelated reparameterization trick notes.\n",
        source_type="vault",
        file_path="Notes/sheet.md",
        metadata={},
    )
    DocumentMemoryBuilder(db).build_and_store_note(note_id="vault:sheet")
    vector_db = DummyVectorDB(
        [
            {
                "id": "unrelated",
                "document": "Unrelated vault evidence",
                "metadata": {"source_type": "vault", "file_path": "Notes/sheet.md", "title": "Sheet"},
                "distance": 0.01,
            }
        ]
    )
    searcher = RAGSearcher(DummyEmbedder(), vector_db, llm=FakeLLM(), sqlite_db=db)

    payload = searcher.generate_answer(
        query="vault:missing-note 파일의 핵심 내용을 보여줘",
        source_type="vault",
        top_k=3,
    )

    assert payload["status"] == "no_result"
    assert payload["sources"] == []
    assert payload["queryFrame"]["family"] == "note_lookup"
    assert payload["queryFrame"]["answer_mode"] == "abstain"
    assert payload["familyRouteDiagnostics"]["runtimeUsed"] == "ask_v2"
    assert payload["v2"]["runtimeExecution"]["used"] == "ask_v2"
    assert payload["v2"]["runtimeExecution"]["fallbackReason"] == "scoped_vault_no_result"
    assert payload["familyRouteDiagnostics"]["vaultScopeApplied"] is True
    assert payload["familyRouteDiagnostics"]["metadataFilterApplied"]["vault_scope_missing"] is True


def test_generate_answer_infers_vault_for_temporal_missing_explicit_note_id(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    db.upsert_note(
        note_id="vault:sheet",
        title="Sheet",
        content="# Sheet\n\nUnrelated reparameterization trick notes.\n",
        source_type="vault",
        file_path="Notes/sheet.md",
        metadata={},
    )
    DocumentMemoryBuilder(db).build_and_store_note(note_id="vault:sheet")
    vector_db = DummyVectorDB(
        [
            {
                "id": "unrelated",
                "document": "Unrelated vault evidence",
                "metadata": {"source_type": "vault", "file_path": "Notes/sheet.md", "title": "Sheet"},
                "distance": 0.01,
            }
        ]
    )
    searcher = RAGSearcher(DummyEmbedder(), vector_db, llm=FakeLLM(), sqlite_db=db)

    payload = searcher.generate_answer(
        query="latest vault:missing-note changes?",
        top_k=3,
    )

    assert payload["status"] == "no_result"
    assert payload["sources"] == []
    assert payload["queryFrame"]["source_type"] == "vault"
    assert payload["queryFrame"]["family"] == "note_lookup"
    assert payload["queryFrame"]["answer_mode"] == "abstain"
    assert payload["familyRouteDiagnostics"]["runtimeUsed"] == "ask_v2"
    assert payload["v2"]["runtimeExecution"]["fallbackReason"] == "scoped_vault_no_result"
