from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import pytest

from knowledge_hub.ai.ask_v2 import AskV2FallbackToLegacy, PaperAskV2Service, _paper_claim_card_gate_relaxation
from knowledge_hub.ai.ask_v2_support import AskV2Route, classify_intent
from knowledge_hub.ai.ask_v2_verification import AskV2Verifier
from knowledge_hub.ai.answer_orchestrator import AnswerOrchestrator
from knowledge_hub.ai.claim_cards import ClaimCardBuilder
from knowledge_hub.application.query_frame import build_query_frame
from knowledge_hub.ai.rag import RAGSearcher
from knowledge_hub.ai.section_card_materializer import PaperSectionCardMaterializer
from knowledge_hub.ai.section_cards import assess_section_source_quality, project_section_cards, rank_section_cards, section_coverage
from knowledge_hub.core.section_card_v1_store import SectionCardV1Store
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.card_v2_builder import PaperCardV2Builder
from knowledge_hub.web.ingest import make_web_note_id
from tests.test_paper_memory import _seed_paper_with_note
from tests.test_rag_search import DummyEmbedder, FakeLLM


class _SearchForbiddenVectorDB:
    def __init__(self):
        self.search_called = False

    def search(self, query_embedding, top_k: int, filter_dict=None):  # noqa: ANN001
        _ = (query_embedding, top_k, filter_dict)
        self.search_called = True
        raise AssertionError("paper ask v2 should not use broad vector search")

    def get_documents(self, filter_dict=None, limit=500, include_ids=True, include_documents=True, include_metadatas=True):  # noqa: ANN001
        _ = (filter_dict, limit, include_ids, include_documents, include_metadatas)
        return {"documents": [], "metadatas": [], "ids": []}


def test_paper_lookup_unsupported_claim_gate_does_not_relax_without_pre_gate_answerability():
    diagnostics = _paper_claim_card_gate_relaxation(
        hard_gate_reason="ask_v2_unsupported_claim_cards",
        route=SimpleNamespace(source_kind="paper"),
        paper_family="paper_lookup",
        pre_hard_gate_answerable=False,
        evidence_packet=SimpleNamespace(filtered_results=[object()], evidence=[]),
        selected_cards=[{"paper_id": "2603.13017"}],
        query_frame=SimpleNamespace(resolved_source_ids=["2603.13017"]),
        card_selection_diagnostics={},
    )

    assert diagnostics["claimCardGateRelaxed"] is False
    assert diagnostics["claimCardGateRelaxationReason"] == ""
    assert diagnostics["originalHardGateReason"] == "ask_v2_unsupported_claim_cards"


def test_paper_compare_unsupported_claim_gate_does_not_relax_when_evidence_collapses_to_one_paper():
    diagnostics = _paper_claim_card_gate_relaxation(
        hard_gate_reason="ask_v2_unsupported_claim_cards",
        route=SimpleNamespace(source_kind="paper"),
        paper_family="paper_compare",
        pre_hard_gate_answerable=True,
        evidence_packet=SimpleNamespace(
            filtered_results=[object()],
            evidence=[],
            evidence_packet={"uniquePaperCount": 1},
        ),
        selected_cards=[{"paper_id": "2603.13017"}, {"paper_id": "2603.13018"}],
        query_frame=SimpleNamespace(resolved_source_ids=["2603.13017", "2603.13018"]),
        card_selection_diagnostics={
            "resolvedPaperIds": ["2603.13017", "2603.13018"],
            "resolvedPairPreserved": True,
        },
    )

    assert diagnostics["claimCardGateRelaxed"] is False
    assert diagnostics["claimCardGateRelaxationReason"] == ""
    assert diagnostics["compareTargetGuardPassed"] is False


class _NoAliasStoreDB:
    def __init__(self, db: SQLiteDatabase):
        self._db = db

    def __getattr__(self, name: str):
        if name in {"list_normalization_aliases", "upsert_normalization_alias"}:
            raise AttributeError(f"'_NoAliasStoreDB' object has no attribute '{name}'")
        return getattr(self._db, name)


def _seed_document_memory(db: SQLiteDatabase, paper_id: str) -> None:
    document_id = f"paper:{paper_id}"
    db.replace_document_memory_units(
        document_id=document_id,
        units=[
            {
                "unit_id": f"{document_id}:summary",
                "document_id": document_id,
                "document_title": "Personalized Agent Memory",
                "source_type": "paper",
                "source_ref": paper_id,
                "unit_type": "document_summary",
                "title": "Summary",
                "section_path": "Summary",
                "contextual_summary": "This paper summarizes personalized agent memory as a compressed retrieval card for long-term interactions.",
                "source_excerpt": "Compressed retrieval memory card for long-term agent interactions.",
                "document_thesis": "Personalized memory improves retrieval quality.",
                "claims": ["claim_memory_1"],
                "concepts": ["concept_memory"],
                "search_text": "personalized agent memory compressed retrieval card",
            },
            {
                "unit_id": f"{document_id}:method",
                "document_id": document_id,
                "document_title": "Personalized Agent Memory",
                "source_type": "paper",
                "source_ref": paper_id,
                "unit_type": "section",
                "title": "Method",
                "section_path": "Method",
                "contextual_summary": "The method distills exchange history into retrieval-oriented cards.",
                "source_excerpt": "Exchange history is distilled into retrieval cards.",
                "claims": ["claim_memory_1"],
                "concepts": ["concept_memory"],
                "search_text": "method retrieval cards distillation",
            },
            {
                "unit_id": f"{document_id}:result",
                "document_id": document_id,
                "document_title": "Personalized Agent Memory",
                "source_type": "paper",
                "source_ref": paper_id,
                "unit_type": "section",
                "title": "Results",
                "section_path": "Results",
                "contextual_summary": "Evaluation reports 11x compression on agent exchange logs.",
                "source_excerpt": "The method achieves 11x compression.",
                "claims": ["claim_memory_1"],
                "concepts": ["concept_memory"],
                "search_text": "results 11x compression evaluation",
            },
            {
                "unit_id": f"{document_id}:limitations",
                "document_id": document_id,
                "document_title": "Personalized Agent Memory",
                "source_type": "paper",
                "source_ref": paper_id,
                "unit_type": "section",
                "title": "Limitations",
                "section_path": "Limitations",
                "contextual_summary": "The paper is only evaluated in a single-user setting.",
                "source_excerpt": "Single-user setting only.",
                "claims": [],
                "concepts": ["concept_memory"],
                "search_text": "limitations single-user setting",
            },
        ],
    )


def _seed_unique_compare_paper(db: SQLiteDatabase, paper_id: str, title: str) -> None:
    claim_id = f"claim_memory_{paper_id.replace('.', '_')}"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": title,
            "authors": "B. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": f"{title} comparison note",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_ontology_entity(
        entity_id=f"paper:{paper_id}",
        entity_type="paper",
        canonical_name=title,
        source="test",
    )
    db.upsert_note(
        note_id=f"paper:{paper_id}",
        title=f"[논문] {title}",
        content=f"# {title}\n\n## 요약\n\n비교 가능한 메모리 논문이다.\n\n## 결과\n\n독립적인 결과 근거를 제공한다.\n",
        source_type="paper",
        para_category="resource",
        metadata={"arxiv_id": paper_id, "quality_flag": "ok"},
    )
    db.upsert_claim(
        claim_id=claim_id,
        claim_text=f"{title} reports comparable memory retrieval results.",
        subject_entity_id=f"paper:{paper_id}",
        predicate="reports",
        object_literal="memory retrieval results",
        confidence=0.86,
        evidence_ptrs=[{"note_id": f"paper:{paper_id}"}],
        source="test",
    )


def _seed_web_document_memory(db: SQLiteDatabase, url: str, *, with_temporal_markers: bool = False) -> None:
    document_id = make_web_note_id(url)
    db.upsert_note(
        document_id,
        title="Memory Systems Guide",
        content="A guide to memory systems. Updated recently.",
        file_path=url,
        source_type="web",
        metadata={"canonical_url": url},
    )
    db.replace_document_memory_units(
        document_id=document_id,
        units=[
            {
                "unit_id": f"{document_id}:summary",
                "document_id": document_id,
                "document_title": "Memory Systems Guide",
                "source_type": "web",
                "source_ref": url,
                "unit_type": "document_summary",
                "title": "Summary",
                "section_path": "Summary",
                "contextual_summary": "This guide explains memory systems and update policies.",
                "source_excerpt": "Memory systems guide and update policy.",
                "document_thesis": "Memory systems need version-aware evidence.",
                "search_text": "memory systems guide update policy",
                "document_date": "2026-03-20T00:00:00+00:00" if with_temporal_markers else "",
                "observed_at": "2026-03-28T00:00:00+00:00",
            },
            {
                "unit_id": f"{document_id}:result",
                "document_id": document_id,
                "document_title": "Memory Systems Guide",
                "source_type": "web",
                "source_ref": url,
                "unit_type": "section",
                "title": "Release Notes" if with_temporal_markers else "Guide",
                "section_path": "Release Notes" if with_temporal_markers else "Guide",
                "contextual_summary": "Version 2.1 updated the memory policy." if with_temporal_markers else "The guide covers memory policy basics.",
                "source_excerpt": "Updated in version 2.1." if with_temporal_markers else "Memory policy basics.",
                "search_text": "memory policy version 2.1 updated" if with_temporal_markers else "memory policy basics",
                "document_date": "2026-03-20T00:00:00+00:00" if with_temporal_markers else "",
                "observed_at": "2026-03-28T00:00:00+00:00",
            },
        ],
    )


def _seed_weak_paper_document_memory(db: SQLiteDatabase, paper_id: str) -> None:
    document_id = f"paper:{paper_id}"
    db.replace_document_memory_units(
        document_id=document_id,
        units=[
            {
                "unit_id": f"{document_id}:summary",
                "document_id": document_id,
                "document_title": "Weak Paper",
                "source_type": "paper",
                "source_ref": paper_id,
                "unit_type": "document_summary",
                "title": "Weak Summary",
                "section_path": "Summary",
                "contextual_summary": "요청 감사합니다 — 논문 전문을 읽고 정확한 심층 요약을 작성하려면 논문 원문(또는 링크)이 필요합니다. 현재 제공하신 건 제목·저자 정보뿐이라 원문을 확인할 수 없어 요약을 바로 작성할 수 없습니다.",
                "source_excerpt": "현재 제공하신 건 제목·저자 정보뿐이라 원문을 확인할 수 없어 요약을 바로 작성할 수 없습니다.",
                "document_thesis": "",
                "search_text": "paper text unavailable original paper required",
                "confidence": 0.85,
            },
            {
                "unit_id": f"{document_id}:block",
                "document_id": document_id,
                "document_title": "Weak Paper",
                "source_type": "paper",
                "source_ref": paper_id,
                "unit_type": "summary",
                "title": "Block 1",
                "section_path": "Block 1",
                "contextual_summary": "현재 제공하신 건 제목·저자 정보뿐이라 원문을 확인할 수 없어 요약을 바로 작성할 수 없습니다.",
                "source_excerpt": "원문 또는 링크가 필요합니다.",
                "search_text": "원문 필요 요약 불가",
                "confidence": 0.72,
            },
        ],
    )


def _seed_vault_document_memory(db: SQLiteDatabase, note_id: str) -> None:
    db.upsert_note(
        note_id,
        title="Memory Design Note",
        content="# Overview\nMemory design note\n\n## Decisions\nUse cards first.",
        file_path="vault/Memory Design Note.md",
        source_type="note",
    )
    db.replace_document_memory_units(
        document_id=note_id,
        units=[
            {
                "unit_id": f"{note_id}:summary",
                "document_id": note_id,
                "document_title": "Memory Design Note",
                "source_type": "note",
                "source_ref": note_id,
                "unit_type": "document_summary",
                "title": "Summary",
                "section_path": "Summary",
                "contextual_summary": "This note summarizes a card-first memory design.",
                "source_excerpt": "Card-first memory design.",
                "document_thesis": "Cards first, scoped evidence last.",
                "search_text": "card first memory design",
            },
            {
                "unit_id": f"{note_id}:decision",
                "document_id": note_id,
                "document_title": "Memory Design Note",
                "source_type": "note",
                "source_ref": note_id,
                "unit_type": "section",
                "title": "Decisions",
                "section_path": "Decisions",
                "contextual_summary": "Use cards first and scoped evidence later.",
                "source_excerpt": "Use cards first.",
                "search_text": "cards first scoped evidence",
            },
        ],
    )


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


def test_paper_card_v2_builder_populates_slots_refs_and_stable_anchors(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    db.upsert_claim_normalization(
        claim_id="claim_memory_1",
        normalization_version="v1",
        status="normalized",
        dataset="MemoryBench",
        metric="compression ratio",
        evidence_strength="strong",
    )

    first = PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")
    second = PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")

    assert first["paper_core"]
    assert first["method_core"]
    assert first["result_core"]
    assert first["dataset_core"] == "MemoryBench"
    assert first["metric_core"] == "compression ratio"
    assert first["claim_refs"][0]["claim_id"] == "claim_memory_1"
    assert first["anchors"]
    assert [item["anchor_id"] for item in first["anchors"]] == [item["anchor_id"] for item in second["anchors"]]


def test_paper_card_v2_builder_dedupes_duplicate_concept_entity_refs(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")

    db.add_relation(
        "paper",
        "2603.13017",
        "uses",
        "concept",
        "concept_memory",
        confidence=0.9,
        evidence_text='{"source":"test_source_a","detail":"duplicate relation from source a"}',
    )
    db.add_relation(
        "paper",
        "2603.13017",
        "uses",
        "concept",
        "concept_memory",
        confidence=0.89,
        evidence_text='{"source":"test_source_b","detail":"duplicate relation from source b"}',
    )

    paper_card = PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")
    entity_ids = [item["entity_id"] for item in paper_card["entity_refs"]]

    assert entity_ids.count("paper:2603.13017") == 1
    assert entity_ids.count("concept_memory") == 1
    assert len(entity_ids) == len(set(entity_ids))


def test_paper_card_v2_builder_drops_heuristic_title_fallback_entity_refs_when_real_concepts_exist(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")

    db.upsert_ontology_entity(
        entity_id="concept_jacobian",
        entity_type="concept",
        canonical_name="Jacobian",
        source="test",
    )
    db.add_relation(
        "paper",
        "2603.13017",
        "uses",
        "concept",
        "concept_jacobian",
        confidence=0.95,
        evidence_text='{"note_id":"AI/AI_Papers/Papers/Math Bridge - Sample.md"}',
    )
    db.upsert_ontology_entity(
        entity_id="feedforward_neural_networks",
        entity_type="concept",
        canonical_name="feedforward neural networks",
        source="paper_memory_title_fallback",
    )
    db.add_relation(
        "paper",
        "2603.13017",
        "uses",
        "concept",
        "feedforward_neural_networks",
        confidence=0.42,
        evidence_text='{"source":"paper_memory_title_fallback","reason":"title-derived fallback concept links"}',
    )

    paper_card = PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")
    entity_names = [item["entity_name"] for item in paper_card["entity_refs"]]

    assert "Jacobian" in entity_names
    assert "feedforward neural networks" not in entity_names


def test_projected_section_cards_map_into_understanding_roles(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")

    paper_card = PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")
    units = list(db.list_document_memory_units("paper:2603.13017", limit=50))
    projected = project_section_cards(source_kind="paper", source_card=paper_card, units=units)
    ranked = rank_section_cards(query="2603.13017 방법과 결과를 설명해줘", section_cards=projected, intent="implementation")
    coverage = section_coverage(section_cards=ranked)

    assert {item["role"] for item in projected} >= {"problem", "method", "results", "limitations"}
    assert "method" in coverage["selectedRoles"]
    assert "results" in coverage["selectedRoles"]


def test_load_section_cards_falls_back_to_paper_memory_adapter_when_document_units_missing(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    db.upsert_paper_memory_card(
        card={
            "memory_id": "paper-memory:2603.13017",
            "paper_id": "2603.13017",
            "source_note_id": "paper:2603.13017",
            "title": "Personalized Agent Memory",
            "paper_core": "Personalized agent memory compresses long-running interactions into reusable cards.",
            "problem_context": "Long-running agent sessions are hard to revisit consistently.",
            "method_core": "The method distills interactions into retrieval-oriented memory cards.",
            "evidence_core": "The paper reports 11x compression on agent exchange logs.",
            "limitations": "The evaluation focuses on a single-user setting.",
            "concept_links": ["agent memory"],
            "claim_refs": ["claim_memory_1"],
            "search_text": "personalized agent memory retrieval cards compression",
            "quality_flag": "ok",
        }
    )

    paper_card = PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")
    service = PaperAskV2Service(_build_searcher(db)[0])
    route = service._route(query="2603.13017 방법을 설명해줘", source_type="paper")

    section_cards = service._load_section_cards(
        query="2603.13017 방법을 설명해줘",
        cards=[paper_card],
        route=route,
    )

    assert {item["role"] for item in section_cards} >= {"problem", "method", "results", "limitations"}
    assert any(item.get("origin") == "paper_memory_adapter_v1" for item in section_cards)


def test_section_source_quality_gate_blocks_problem_only_placeholder_sections(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_weak_paper_document_memory(db, "2603.13017")

    paper_card = PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")
    units = list(db.list_document_memory_units("paper:2603.13017", limit=50))
    projected = project_section_cards(source_kind="paper", source_card=paper_card, units=units)
    ranked = rank_section_cards(query="2603.13017 방법을 설명해줘", section_cards=projected, intent="implementation")
    coverage = section_coverage(section_cards=ranked)
    gate = assess_section_source_quality(section_cards=ranked, coverage=coverage)

    assert coverage["status"] == "weak"
    assert gate["allowed"] is False
    assert gate["reason"] in {"problem_only_sections", "placeholder_majority", "all_sections_meta_only", "weak_missing_method_or_results"}


def test_claim_card_builder_persists_paper_claim_cards_and_alignment_refs(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    db.upsert_claim(
        claim_id="claim_memory_2",
        claim_text="The method performs worse than the baseline on MemoryBench.",
        subject_entity_id="paper:2603.13017",
        predicate="reports",
        object_literal="baseline",
        confidence=0.81,
        evidence_ptrs=[],
        source="test",
    )
    db.upsert_claim_normalization(
        claim_id="claim_memory_1",
        normalization_version="v1",
        status="normalized",
        task="retrieval_augmented_generation",
        dataset="MemoryBench",
        metric="compression ratio",
        comparator="baseline",
        result_direction="better",
        result_value_text="11x compression",
        result_value_numeric=11.0,
        evidence_strength="strong",
    )
    db.upsert_claim_normalization(
        claim_id="claim_memory_2",
        normalization_version="v1",
        status="normalized",
        task="retrieval_augmented_generation",
        dataset="MemoryBench",
        metric="compression ratio",
        comparator="baseline",
        result_direction="worse",
        result_value_text="worse than baseline",
        result_value_numeric=9.0,
        evidence_strength="strong",
    )

    paper_card = PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")
    claim_cards = ClaimCardBuilder(db).build_and_store_for_source_card(source_kind="paper", source_card=paper_card)

    assert len(claim_cards) >= 2
    stored = db.list_claim_cards(source_kind="paper", source_id="2603.13017", limit=10)
    assert {item["claim_id"] for item in stored} >= {"claim_memory_1", "claim_memory_2"}
    assert any(item["claim_type"] == "comparison" for item in stored)
    refs = db.list_claim_card_source_refs(source_card_id=paper_card["card_id"])
    assert len(refs) >= 2
    alignment = db.list_claim_card_alignment_refs(claim_card_id="claim-card-v1:paper:claim_memory_1")
    assert any(item["alignment_type"] == "conflict" for item in alignment)


def test_claim_card_builder_loads_stored_cards_without_alias_store_methods(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    db.upsert_claim_normalization(
        claim_id="claim_memory_1",
        normalization_version="v1",
        status="normalized",
        task="retrieval_augmented_generation",
        dataset="MemoryBench",
        metric="compression ratio",
        comparator="baseline",
        result_direction="better",
        result_value_text="11x compression",
        result_value_numeric=11.0,
        evidence_strength="strong",
    )

    paper_card = PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")
    stored = ClaimCardBuilder(db).build_and_store_for_source_card(source_kind="paper", source_card=paper_card)

    proxy_db = _NoAliasStoreDB(db)
    loaded = ClaimCardBuilder(proxy_db).load_or_build_for_source_card(source_kind="paper", source_card=paper_card)

    assert {item["claim_card_id"] for item in loaded} == {item["claim_card_id"] for item in stored}
    assert all(item["anchors"] for item in loaded)


def test_paper_ask_v2_router_classifies_summary_relation_and_temporal(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    db.add_entity_alias("memory", "concept_memory")

    service = PaperAskV2Service(_build_searcher(db)[0])
    summary_route = service._route(query="2603.13017 논문 요약", metadata_filter=None)
    relation_route = service._route(query="memory 관계를 설명해줘", metadata_filter=None)
    temporal_route = service._route(query="latest memory paper", metadata_filter=None)

    assert summary_route.intent == "paper_lookup"
    assert summary_route.mode == "card-first"
    assert relation_route.intent == "relation"
    assert relation_route.mode == "ontology-first"
    assert temporal_route.intent == "temporal"
    assert temporal_route.mode == "card-first"


def test_execute_paper_concept_falls_back_only_when_no_candidates_exist(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    service = PaperAskV2Service(_build_searcher(db)[0])
    frame = build_query_frame(
        domain_key="ai_papers",
        source_type="paper",
        family="concept_explainer",
        query_intent="definition",
        answer_mode="representative_paper_explainer",
        confidence=0.9,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="concept_explainer_policy",
        metadata_filter={},
    )

    with pytest.raises(AskV2FallbackToLegacy, match="no_concept_candidates"):
        service.execute(
            query="Transformer의 핵심 아이디어를 설명해줘",
            top_k=3,
            source_type="paper",
            retrieval_mode="hybrid",
            alpha=0.7,
            allow_external=False,
            query_frame=frame.to_dict(),
        )


def test_execute_paper_discover_prefers_doc_summary_anchors_without_claim_cards(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")
    service = PaperAskV2Service(_build_searcher(db)[0])
    frame = build_query_frame(
        domain_key="ai_papers",
        source_type="paper",
        family="paper_discover",
        query_intent="paper_topic",
        answer_mode="paper_shortlist_summary",
        expanded_terms=["agent memory", "personalized agent memory"],
        confidence=0.86,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="paper_discover_policy",
        metadata_filter={},
    )

    monkeypatch.setattr(
        service,
        "_load_claim_cards",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("paper_discover should stay retrieval-only in ask_v2")),
    )

    pipeline_result, _evidence_packet = service.execute(
        query="agent memory 관련 논문들을 추천해줘",
        top_k=3,
        source_type="paper",
        retrieval_mode="hybrid",
        alpha=0.7,
        allow_external=False,
        query_frame=frame.to_dict(),
    )

    assert pipeline_result.results
    assert pipeline_result.results[0].metadata["unit_type"] == "document_summary"
    diagnostics = pipeline_result.diagnostics()
    assert diagnostics["candidateSources"]
    assert diagnostics["candidateSources"][0]["sourceKind"] == "paper"
    assert diagnostics["memoryRoute"]["contractRole"] == "ask_retrieval_memory_prefilter"
    assert diagnostics["memoryPrefilter"]["contractRole"] == "retrieval_memory_prefilter"
    assert diagnostics["paperMemoryPrefilter"]["contractRole"] == "paper_source_memory_prefilter"
    assert diagnostics["paperMemoryPrefilter"]["requestedMode"] == "off"
    assert diagnostics["paperMemoryPrefilter"]["applied"] is False
    assert diagnostics["paperMemoryPrefilter"]["matchedPaperIds"] == []
    assert "2603.13017" in [item["id"] for item in diagnostics["candidateSources"]]
    assert diagnostics["contextExpansion"]["mode"] in {"card", "ontology"}
    assert pipeline_result.v2_diagnostics["runtimeExecution"]["used"] == "ask_v2"


def test_execute_paper_discover_keeps_claim_builder_lazy_for_read_only_path(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")

    class _ForbiddenClaimCardBuilder:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("claim card builder should stay lazy for paper_discover")

    monkeypatch.setattr("knowledge_hub.ai.ask_v2.ClaimCardBuilder", _ForbiddenClaimCardBuilder)

    service = PaperAskV2Service(_build_searcher(db)[0])
    frame = build_query_frame(
        domain_key="ai_papers",
        source_type="paper",
        family="paper_discover",
        query_intent="paper_topic",
        answer_mode="paper_shortlist_summary",
        expanded_terms=["agent memory", "personalized agent memory"],
        confidence=0.86,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="paper_discover_policy",
        metadata_filter={},
    )

    pipeline_result, _evidence_packet = service.execute(
        query="agent memory 관련 논문들을 추천해줘",
        top_k=3,
        source_type="paper",
        retrieval_mode="hybrid",
        alpha=0.7,
        allow_external=False,
        query_frame=frame.to_dict(),
    )

    assert service._claim_card_builder is None
    assert pipeline_result.results
    assert pipeline_result.v2_diagnostics["runtimeExecution"]["used"] == "ask_v2"


def test_paper_ask_v2_classifies_core_idea_queries_as_definition():
    assert classify_intent("Transformer의 핵심 아이디어를 설명해줘") == "definition"
    assert classify_intent("self-attention의 원리를 설명해줘") == "definition"
    assert classify_intent("CNN을 쉽게 설명해줘") == "definition"


def test_paper_ask_v2_keeps_implementation_queries_out_of_definition_lane():
    assert classify_intent("RAG 파이프라인을 설명해줘") == "implementation"


def test_ask_v2_route_preserves_concept_definition_when_classifier_is_impl_or_eval(tmp_path):
    """Rule-based frame says definition; regex classifier can win on impl/eval keywords first."""
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    service = PaperAskV2Service(_build_searcher(db)[0])
    frame = {"family": "concept_explainer", "query_intent": "definition"}
    q_impl = "Transformer 아키텍처와 파이프라인을 쉽게 설명해줘"
    q_eval = "Transformer의 성능과 지표를 초심자에게 개념 설명해줘"
    assert classify_intent(q_impl, None) == "implementation"
    assert classify_intent(q_eval, None) == "evaluation"
    assert service._route(query=q_impl, source_type="paper", metadata_filter=None, query_frame=frame).intent == "definition"
    assert service._route(query=q_eval, source_type="paper", metadata_filter=None, query_frame=frame).intent == "definition"


def test_ask_v2_route_preserves_paper_compare_when_classifier_is_impl_or_eval(tmp_path):
    """Planner frame says comparison; regex can match implementation/eval without compare cue words."""
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    service = PaperAskV2Service(_build_searcher(db)[0])
    frame = {"family": "paper_compare", "query_intent": "comparison"}
    q_impl = "AlexNet and ViT implementation details"
    q_eval = "AlexNet and ViT benchmark metrics"
    assert classify_intent(q_impl, None) == "implementation"
    assert classify_intent(q_eval, None) == "evaluation"
    assert service._route(query=q_impl, source_type="paper", metadata_filter=None, query_frame=frame).intent == "comparison"
    assert service._route(query=q_eval, source_type="paper", metadata_filter=None, query_frame=frame).intent == "comparison"


def test_generate_answer_compare_with_stored_claim_cards_does_not_require_alias_store(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    db.upsert_claim(
        claim_id="claim_memory_2",
        claim_text="The method performs worse than the baseline on MemoryBench.",
        subject_entity_id="paper:2603.13017",
        predicate="reports",
        object_literal="baseline",
        confidence=0.81,
        evidence_ptrs=[],
        source="test",
    )
    db.upsert_claim_normalization(
        claim_id="claim_memory_1",
        normalization_version="v1",
        status="normalized",
        task="retrieval_augmented_generation",
        dataset="MemoryBench",
        metric="compression ratio",
        comparator="baseline",
        result_direction="better",
        result_value_text="11x compression",
        result_value_numeric=11.0,
        evidence_strength="strong",
    )
    db.upsert_claim_normalization(
        claim_id="claim_memory_2",
        normalization_version="v1",
        status="normalized",
        task="retrieval_augmented_generation",
        dataset="MemoryBench",
        metric="compression ratio",
        comparator="baseline",
        result_direction="worse",
        result_value_text="worse than baseline",
        result_value_numeric=9.0,
        evidence_strength="strong",
    )

    paper_card = PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")
    ClaimCardBuilder(db).build_and_store_for_source_card(source_kind="paper", source_card=paper_card)
    searcher, vector_db = _build_searcher(_NoAliasStoreDB(db))

    payload = searcher.generate_answer(
        query="2603.13017 metric comparison",
        source_type="paper",
        top_k=3,
    )

    assert vector_db.search_called is False
    assert payload["v2"]["runtimeExecution"]["used"] == "ask_v2"
    assert payload["claimCards"]
    assert payload["answerProvenance"]["mode"] == "claim_cards_conflicted"


def test_generate_answer_rebuilds_stale_paper_card_when_upstream_memory_is_newer(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")
    db.conn.execute(
        "UPDATE paper_cards_v2 SET updated_at = ? WHERE paper_id = ?",
        ("2026-03-01T00:00:00+00:00", "2603.13017"),
    )
    db.conn.execute(
        """
        UPDATE paper_memory_cards
        SET paper_core = ?, updated_at = ?
        WHERE paper_id = ?
        """,
        (
            "Rebuilt paper core from fresher memory card.",
            datetime(2026, 3, 28, tzinfo=timezone.utc).isoformat(),
            "2603.13017",
        ),
    )
    db.conn.commit()
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="2603.13017 논문 요약",
        source_type="paper",
        top_k=3,
    )

    refreshed = db.get_paper_card_v2("2603.13017")
    assert vector_db.search_called is False
    assert refreshed is not None
    assert refreshed["paper_core"] == "Rebuilt paper core from fresher memory card."
    assert payload["v2"]["routing"]["selected_card_ids"]


def test_generate_answer_uses_card_first_v2_and_anchor_scoped_evidence(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="2603.13017 논문 요약",
        source_type="paper",
        top_k=3,
    )

    assert vector_db.search_called is False
    assert payload["v2"]["routing"]["mode"] == "card-first"
    assert payload["v2"]["routing"]["selected_card_ids"]
    assert payload["v2"]["evidenceVerification"]["anchorIdsUsed"]
    assert payload["paperMemoryPrefilter"]["requestedMode"] == "off"
    assert payload["paperMemoryPrefilter"]["effectiveMode"] == "off"
    assert payload["paperMemoryPrefilter"]["applied"] is False
    assert payload["queryFrame"]["family"] == "paper_lookup"
    assert payload["evidencePolicy"]["policyKey"] == "paper_lookup_policy"


def test_generate_answer_v2_fallback_does_not_trigger_on_weak_claim_only_when_verification_is_strong(tmp_path, monkeypatch):
    """Weak-only claim consensus should remain diagnostic when answer verification is strong."""
    _orig_cv = AskV2Verifier.claim_verification
    _orig_vs = AskV2Verifier.verification_summary

    def _claim_cv(self, *, selected_claims, anchors):
        cv, cc = _orig_cv(self, selected_claims=selected_claims, anchors=anchors)
        merged = dict(cc)
        merged["weakClaimCount"] = int(merged.get("weakClaimCount") or 0) + 1
        return cv, merged

    def _ver_sum(self, **kwargs):
        out = _orig_vs(self, **kwargs)
        merged = dict(out)
        merged["verificationStatus"] = "strong"
        return merged

    monkeypatch.setattr(AskV2Verifier, "claim_verification", _claim_cv)
    monkeypatch.setattr(AskV2Verifier, "verification_summary", _ver_sum)

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="2603.13017 논문 요약",
        source_type="paper",
        top_k=3,
    )

    assert vector_db.search_called is False
    assert int(payload["v2"]["consensus"].get("weakClaimCount") or 0) >= 1
    assert payload["v2"]["evidenceVerification"].get("verificationStatus") == "strong"
    assert payload["v2"]["fallback"]["used"] is False


def test_generate_answer_paper_lookup_relaxes_unsupported_claim_gate_when_pre_gate_answerable(tmp_path, monkeypatch):
    _orig_cv = AskV2Verifier.claim_verification

    def _claim_cv(self, *, selected_claims, anchors):
        cv, cc = _orig_cv(self, selected_claims=selected_claims, anchors=anchors)
        merged = dict(cc)
        merged["unsupportedClaimCount"] = int(merged.get("unsupportedClaimCount") or 0) + 1
        return [
            *cv,
            {
                "claimId": "claim_without_anchor",
                "status": "unsupported",
                "verdict": "unsupported",
                "reasons": ["no_anchor_backed_evidence"],
            },
        ], merged

    monkeypatch.setattr(AskV2Verifier, "claim_verification", _claim_cv)

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="2603.13017 논문 요약",
        source_type="paper",
        top_k=3,
    )

    v2 = payload["v2"]
    assert vector_db.search_called is False
    assert payload["status"] != "no_result"
    assert payload["evidencePacket"]["answerable"] is True
    assert payload["evidencePacket"]["askV2HardGate"] is False
    assert payload["evidencePacket"]["claimCardGateRelaxed"] is True
    assert payload["evidencePacket"]["askV2OriginalHardGateReason"] == "ask_v2_unsupported_claim_cards"
    assert payload["evidencePacket"]["originalHardGateReason"] == "ask_v2_unsupported_claim_cards"
    assert payload["evidencePacket"]["compareTargetGuardPassed"] is None
    assert v2["preHardGateAnswerable"] is True
    assert v2["preHardGateReason"] == "substantive_evidence_found"
    assert v2["v2ConsensusUnsupportedClaimCount"] == v2["consensus"]["unsupportedClaimCount"]
    assert v2["v2ConsensusWeakClaimCount"] == v2["consensus"]["weakClaimCount"]
    assert v2["v2ConsensusSupportedClaimCount"] == v2["consensus"]["supportCount"]
    assert "unsupported:no_anchor_backed_evidence=1" in v2["v2ClaimReasonSummary"]
    assert v2["askV2OriginalHardGateReason"] == "ask_v2_unsupported_claim_cards"
    assert v2["originalHardGateReason"] == "ask_v2_unsupported_claim_cards"
    assert v2["compareTargetGuardPassed"] is None
    assert v2["claimCardGateRelaxed"] is True
    assert v2["claimCardGateRelaxationReason"] == "claim_card_unsupported_relaxed_for_paper_family"
    assert v2["claimCardGateRelaxation"]["claimCardGateRelaxationDetail"] == "paper_lookup_answerable_evidence_present"
    assert v2["fallback"]["reason"] == "claim_card_unsupported_relaxed_for_paper_family"


def test_generate_answer_paper_compare_relaxes_unsupported_claim_gate_when_compare_evidence_present(tmp_path, monkeypatch):
    _orig_cv = AskV2Verifier.claim_verification

    def _claim_cv(self, *, selected_claims, anchors):
        cv, cc = _orig_cv(self, selected_claims=selected_claims, anchors=anchors)
        merged = dict(cc)
        merged["unsupportedClaimCount"] = int(merged.get("unsupportedClaimCount") or 0) + 2
        return [
            *cv,
            {
                "claimId": "compare_claim_without_anchor",
                "status": "unsupported",
                "verdict": "unsupported",
                "reasons": ["no_anchor_backed_evidence"],
            },
        ], merged

    monkeypatch.setattr(AskV2Verifier, "claim_verification", _claim_cv)

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path, paper_id="2603.13017")
    _seed_document_memory(db, "2603.13017")
    db.upsert_paper(
        {
            "arxiv_id": "2603.13018",
            "title": "Second Agent Memory",
            "authors": "B. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "second memory paper",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_ontology_entity(
        entity_id="paper:2603.13018",
        entity_type="paper",
        canonical_name="Second Agent Memory",
        source="test",
    )
    db.upsert_note(
        note_id="paper:2603.13018",
        title="[논문] Second Agent Memory",
        content="# Second Agent Memory\n\n## 요약\n\n두 번째 메모리 논문이다.\n\n## 결과\n\n비교 가능한 결과를 제공한다.\n",
        source_type="paper",
        para_category="resource",
        metadata={"arxiv_id": "2603.13018", "quality_flag": "ok"},
    )
    _seed_document_memory(db, "2603.13018")
    PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")
    PaperCardV2Builder(db).build_and_store(paper_id="2603.13018")
    searcher, vector_db = _build_searcher(db)
    service = PaperAskV2Service(searcher)
    frame = build_query_frame(
        domain_key="ai_papers",
        source_type="paper",
        family="paper_compare",
        query_intent="comparison",
        answer_mode="paper_comparison",
        resolved_source_ids=["2603.13017", "2603.13018"],
        confidence=0.9,
        planner_status="not_attempted",
        planner_reason="test_frame",
        evidence_policy_key="paper_compare_policy",
        metadata_filter={},
    )
    pipeline_result, evidence_packet = service.execute(
        query="2603.13017와 2603.13018을 비교해줘",
        top_k=3,
        source_type="paper",
        retrieval_mode="hybrid",
        alpha=0.7,
        allow_external=False,
        query_frame=frame.to_dict(),
    )

    payload = AnswerOrchestrator(searcher).generate(
        query="2603.13017와 2603.13018을 비교해줘",
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
        pipeline_result=pipeline_result,
        evidence_packet=evidence_packet,
    )

    assert vector_db.search_called is False
    assert payload["status"] != "no_result"
    assert payload["evidencePacket"]["answerable"] is True
    assert payload["evidencePacket"]["uniquePaperCount"] >= 2
    assert payload["evidencePacket"]["askV2HardGate"] is False
    assert payload["evidencePacket"]["claimCardGateRelaxed"] is True
    assert payload["evidencePacket"]["compareTargetGuardPassed"] is True
    assert payload["v2"]["askV2OriginalHardGateReason"] == "ask_v2_unsupported_claim_cards"
    assert payload["v2"]["originalHardGateReason"] == "ask_v2_unsupported_claim_cards"
    assert payload["v2"]["claimCardGateRelaxed"] is True
    assert payload["v2"]["claimCardGateRelaxationReason"] == "claim_card_unsupported_relaxed_for_paper_family"
    assert payload["v2"]["claimCardGateRelaxation"]["claimCardGateRelaxationDetail"] == "paper_compare_target_guard_passed"
    assert payload["v2"]["compareTargetGuardPassed"] is True
    assert payload["v2"]["v2ConsensusUnsupportedClaimCount"] > 0
    assert "unsupported:no_anchor_backed_evidence=" in payload["v2"]["v2ClaimReasonSummary"]


def test_generate_answer_paper_compare_keeps_claim_gate_when_selected_target_drifts(tmp_path, monkeypatch):
    _orig_cv = AskV2Verifier.claim_verification

    def _claim_cv(self, *, selected_claims, anchors):
        cv, cc = _orig_cv(self, selected_claims=selected_claims, anchors=anchors)
        merged = dict(cc)
        merged["unsupportedClaimCount"] = int(merged.get("unsupportedClaimCount") or 0) + 1
        return [
            *cv,
            {
                "claimId": "drifted_compare_claim_without_anchor",
                "status": "unsupported",
                "verdict": "unsupported",
                "reasons": ["no_anchor_backed_evidence"],
            },
        ], merged

    monkeypatch.setattr(AskV2Verifier, "claim_verification", _claim_cv)

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path, paper_id="2603.13017")
    _seed_document_memory(db, "2603.13017")
    _seed_unique_compare_paper(db, "2603.13018", "Second Agent Memory")
    _seed_document_memory(db, "2603.13018")
    _seed_unique_compare_paper(db, "2603.13019", "Drifted Agent Memory")
    _seed_document_memory(db, "2603.13019")
    card_a = PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")
    PaperCardV2Builder(db).build_and_store(paper_id="2603.13018")
    card_wrong = PaperCardV2Builder(db).build_and_store(paper_id="2603.13019")

    drifted_cards = [dict(card_a), dict(card_wrong)]
    for card in drifted_cards:
        card["selection_diagnostics"] = {
            "resolvedPaperIds": ["2603.13017", "2603.13018"],
            "candidateCountBeforeRerank": 3,
            "candidateCountAfterRerank": 3,
            "resolvedPairPreserved": False,
            "stage": "compare_ranked_diversity",
            "reason": "test_selected_target_drift",
        }

    searcher, vector_db = _build_searcher(db)
    service = PaperAskV2Service(searcher)
    monkeypatch.setattr(service, "_select_cards", lambda **_kwargs: drifted_cards)
    frame = build_query_frame(
        domain_key="ai_papers",
        source_type="paper",
        family="paper_compare",
        query_intent="comparison",
        answer_mode="paper_comparison",
        resolved_source_ids=["2603.13017", "2603.13018"],
        confidence=0.9,
        planner_status="not_attempted",
        planner_reason="test_frame",
        evidence_policy_key="paper_compare_policy",
        metadata_filter={},
    )
    pipeline_result, evidence_packet = service.execute(
        query="2603.13017와 2603.13018을 비교해줘",
        top_k=3,
        source_type="paper",
        retrieval_mode="hybrid",
        alpha=0.7,
        allow_external=False,
        query_frame=frame.to_dict(),
    )

    payload = AnswerOrchestrator(searcher).generate(
        query="2603.13017와 2603.13018을 비교해줘",
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
        pipeline_result=pipeline_result,
        evidence_packet=evidence_packet,
    )

    assert vector_db.search_called is False
    assert payload["status"] == "no_result"
    assert payload["evidencePacket"]["answerable"] is False
    assert payload["evidencePacket"]["answerableDecisionReason"] == "ask_v2_unsupported_claim_cards"
    assert payload["evidencePacket"]["askV2HardGate"] is True
    assert payload["evidencePacket"]["claimCardGateRelaxed"] is False
    assert payload["evidencePacket"]["compareTargetGuardPassed"] is False
    assert payload["v2"]["cardSelection"]["resolvedPairPreserved"] is False
    assert payload["v2"]["compareTargetGuardPassed"] is False
    assert payload["v2"]["claimCardGateRelaxed"] is False
    assert payload["v2"]["originalHardGateReason"] == "ask_v2_unsupported_claim_cards"
    assert {item["paperId"] for item in payload["v2"]["cardSelection"]["selected"]} == {"2603.13017", "2603.13019"}
    assert payload["v2"]["v2ConsensusUnsupportedClaimCount"] > 0


def test_generate_answer_missing_verification_still_hard_gates_paper_lookup(tmp_path, monkeypatch):
    _orig_vs = AskV2Verifier.verification_summary

    def _ver_sum(self, **kwargs):
        out = dict(_orig_vs(self, **kwargs))
        out["verificationStatus"] = "missing"
        out["unsupportedFields"] = []
        return out

    monkeypatch.setattr(AskV2Verifier, "verification_summary", _ver_sum)

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    searcher, _vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="2603.13017 논문 요약",
        source_type="paper",
        top_k=3,
    )

    assert payload["status"] == "no_result"
    assert payload["evidencePacket"]["askV2HardGate"] is True
    assert payload["evidencePacket"]["answerableDecisionReason"] == "ask_v2_missing"
    assert payload["evidencePacket"]["claimCardGateRelaxed"] is False
    assert payload["v2"]["claimCardGateRelaxed"] is False


def test_generate_answer_non_paper_unsupported_claim_gate_still_hard_gates(tmp_path, monkeypatch):
    _orig_cv = AskV2Verifier.claim_verification

    def _claim_cv(self, *, selected_claims, anchors):
        cv, cc = _orig_cv(self, selected_claims=selected_claims, anchors=anchors)
        merged = dict(cc)
        merged["unsupportedClaimCount"] = int(merged.get("unsupportedClaimCount") or 0) + 1
        return [
            *cv,
            {
                "claimId": "web_claim_without_anchor",
                "status": "unsupported",
                "verdict": "unsupported",
                "reasons": ["no_anchor_backed_evidence"],
            },
        ], merged

    monkeypatch.setattr(AskV2Verifier, "claim_verification", _claim_cv)

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    url = "https://example.com/memory-guide"
    _seed_web_document_memory(db, url)
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="memory systems guide",
        source_type="web",
        top_k=3,
        metadata_filter={"canonical_url": url},
    )

    assert vector_db.search_called is False
    assert payload["status"] == "no_result"
    assert payload["evidencePacket"]["askV2HardGate"] is True
    assert payload["evidencePacket"]["answerableDecisionReason"] == "ask_v2_unsupported_claim_cards"
    assert payload["evidencePacket"]["claimCardGateRelaxed"] is False
    assert payload["v2"]["routing"]["sourceKind"] == "web"
    assert payload["v2"]["claimCardGateRelaxed"] is False
    assert payload["v2"]["askV2OriginalHardGateReason"] == "ask_v2_unsupported_claim_cards"


def test_select_paper_cards_prefers_resolved_compare_scope_over_broad_search(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    service = PaperAskV2Service(_build_searcher(db)[0])
    route = service._route(query="CNN vs ViT 비교해줘", metadata_filter=None)

    monkeypatch.setattr(
        service,
        "_ensure_cards_for_papers",
        lambda paper_ids: [
            {"paper_id": "alexnet-2012", "card_id": "paper-card-v2:alexnet-2012", "title": "AlexNet", "quality_flag": "ok"},
            {"paper_id": "2010.11929", "card_id": "paper-card-v2:2010.11929", "title": "ViT", "quality_flag": "ok"},
        ]
        if paper_ids == ["alexnet-2012", "2010.11929"]
        else [],
    )
    monkeypatch.setattr(
        db,
        "search_paper_cards_v2",
        lambda *_args, **_kwargs: [
            {"paper_id": "2507.07957", "card_id": "paper-card-v2:2507.07957", "title": "Noise", "quality_flag": "ok"}
        ],
    )

    selected = service._select_paper_cards(
        query="CNN vs ViT 비교해줘",
        route=route,
        limit=2,
        metadata_filter=None,
        query_plan={
            "family": "paper_compare",
            "resolvedPaperIds": ["alexnet-2012", "2010.11929"],
            "expandedTerms": ["CNN", "ViT", "An Image is Worth 16x16 Words"],
        },
        query_frame={
            "family": "paper_compare",
            "resolved_source_ids": ["alexnet-2012", "2010.11929"],
            "expanded_terms": ["CNN", "ViT", "An Image is Worth 16x16 Words"],
        },
    )

    assert {item["paper_id"] for item in selected} == {"alexnet-2012", "2010.11929"}
    diagnostics = [dict(item.get("selection_diagnostics") or {}) for item in selected]
    assert [item["resolvedPaperIds"] for item in diagnostics] == [
        ["alexnet-2012", "2010.11929"],
        ["alexnet-2012", "2010.11929"],
    ]
    assert {item["stage"] for item in diagnostics} == {"compare_resolved_paper_id"}
    before_counts = {item["candidateCountBeforeRerank"] for item in diagnostics}
    assert len(before_counts) == 1
    assert next(iter(before_counts)) >= 3
    assert {item["candidateCountAfterRerank"] for item in diagnostics} == {3}
    assert {item["resolvedPairPreserved"] for item in diagnostics} == {True}


def test_select_paper_cards_preserves_resolved_compare_scope_when_rerank_truncates_targets(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    service = PaperAskV2Service(_build_searcher(db)[0])
    route = service._route(query="RAG와 FiD를 비교해줘", metadata_filter=None)
    resolved_cards = [
        {
            "paper_id": "2005.11401",
            "card_id": "paper-card-v2:2005.11401",
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "quality_flag": "ok",
        },
        {
            "paper_id": "2007.01282",
            "card_id": "paper-card-v2:2007.01282",
            "title": "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
            "quality_flag": "ok",
        },
    ]
    noise_cards = [
        {
            "paper_id": f"noise-{idx}",
            "card_id": f"paper-card-v2:noise-{idx}",
            "title": f"RAG distractor {idx}",
            "quality_flag": "ok",
        }
        for idx in range(6)
    ]

    monkeypatch.setattr(
        service,
        "_ensure_cards_for_papers",
        lambda paper_ids: resolved_cards if paper_ids == ["2005.11401", "2007.01282"] else [],
    )
    monkeypatch.setattr(db, "search_paper_cards_v2", lambda *_args, **_kwargs: noise_cards)
    monkeypatch.setattr(
        service,
        "_dedupe_and_score",
        lambda candidates, **_kwargs: [
            card for card in candidates if str(card.get("paper_id") or "").startswith("noise-")
        ][:2],
    )

    selected = service._select_paper_cards(
        query="RAG와 FiD를 비교해줘",
        route=route,
        limit=2,
        metadata_filter=None,
        query_plan={
            "family": "paper_compare",
            "resolvedPaperIds": ["2005.11401", "2007.01282"],
            "expandedTerms": [
                "RAG",
                "FiD",
                "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
            ],
        },
        query_frame={
            "family": "paper_compare",
            "resolved_source_ids": ["2005.11401", "2007.01282"],
            "expanded_terms": [
                "RAG",
                "FiD",
                "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
            ],
        },
    )

    assert [item["paper_id"] for item in selected] == ["2005.11401", "2007.01282"]
    diagnostics = [dict(item.get("selection_diagnostics") or {}) for item in selected]
    assert {item["stage"] for item in diagnostics} <= {"compare_focus_form", "compare_resolved_paper_id"}
    assert {item["resolvedPairPreserved"] for item in diagnostics} == {True}
    assert {item["candidateCountAfterRerank"] for item in diagnostics} == {2}


def test_select_paper_cards_does_not_fill_exact_resolved_pair_with_same_axis_rescue_title(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    service = PaperAskV2Service(_build_searcher(db)[0])
    route = AskV2Route(
        source_kind="paper",
        intent="comparison",
        mode="ontology-first",
        matched_entities=[],
        entity_ids=["diffusion"],
    )
    resolved_cards = [
        {
            "paper_id": "1406.2661",
            "card_id": "paper-card-v2:1406.2661",
            "title": "Generative Adversarial Nets",
            "quality_flag": "ok",
        },
        {
            "paper_id": "2006.11239",
            "card_id": "paper-card-v2:2006.11239",
            "title": "Denoising Diffusion Probabilistic Models",
            "quality_flag": "ok",
        },
    ]
    latent_diffusion = {
        "paper_id": "2112.10752",
        "card_id": "paper-card-v2:2112.10752",
        "title": "High-Resolution Image Synthesis with Latent Diffusion Models",
        "quality_flag": "ok",
    }

    monkeypatch.setattr(
        service,
        "_ensure_cards_for_papers",
        lambda paper_ids: resolved_cards if paper_ids == ["1406.2661", "2006.11239"] else [],
    )
    monkeypatch.setattr(
        db,
        "search_paper_cards_v2",
        lambda form, **_kwargs: [latent_diffusion]
        if form == "High-Resolution Image Synthesis with Latent Diffusion Models"
        else [],
    )
    monkeypatch.setattr(
        db,
        "list_paper_cards_v2_by_entity_ids",
        lambda **_kwargs: [
            resolved_cards[1],
            latent_diffusion,
        ],
    )
    monkeypatch.setattr(
        service,
        "_dedupe_and_score",
        lambda candidates, **_kwargs: [
            card for card in candidates if card.get("paper_id") in {"2006.11239", "2112.10752"}
        ],
    )

    selected = service._select_paper_cards(
        query="GAN과 Diffusion 모델 논문을 비교해줘",
        route=route,
        limit=2,
        metadata_filter=None,
        query_plan={
            "family": "paper_compare",
            "resolvedPaperIds": ["1406.2661", "2006.11239"],
            "expandedTerms": [
                "GAN",
                "Diffusion",
                "Generative Adversarial Nets",
                "Denoising Diffusion Probabilistic Models",
                "High-Resolution Image Synthesis with Latent Diffusion Models",
            ],
        },
        query_frame={
            "family": "paper_compare",
            "resolved_source_ids": ["1406.2661", "2006.11239"],
            "expanded_terms": [
                "GAN",
                "Diffusion",
                "Generative Adversarial Nets",
                "Denoising Diffusion Probabilistic Models",
                "High-Resolution Image Synthesis with Latent Diffusion Models",
            ],
        },
    )

    assert {item["paper_id"] for item in selected} == {"1406.2661", "2006.11239"}
    assert "2112.10752" not in {item["paper_id"] for item in selected}
    diagnostics = [dict(item.get("selection_diagnostics") or {}) for item in selected]
    assert {item["resolvedPairPreserved"] for item in diagnostics} == {True}


def test_select_paper_cards_preserves_paper_lookup_scope_for_title_queries_with_architecture_keywords(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    service = PaperAskV2Service(_build_searcher(db)[0])
    frame = build_query_frame(
        domain_key="ai_papers",
        source_type="paper",
        family="paper_lookup",
        query_intent="paper_lookup",
        answer_mode="paper_scoped_answer",
        entities=["Compound", "AI", "Architecture"],
        canonical_entity_ids=["architecture"],
        expanded_terms=["A Compound AI Architecture for Scientific Discovery"],
        resolved_source_ids=["compound-local"],
        confidence=0.86,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key="paper_lookup_policy",
        metadata_filter={},
    )
    route = service._route(
        query="A Compound AI Architecture for Scientific Discovery 논문 설명해줘",
        source_type="paper",
        metadata_filter=None,
        query_frame=frame.to_dict(),
    )

    monkeypatch.setattr(
        service,
        "_ensure_cards_for_papers",
        lambda paper_ids: [
            {
                "paper_id": "compound-local",
                "card_id": "paper-card-v2:compound-local",
                "title": "A Compound AI Architecture for Scientific Discovery",
                "quality_flag": "ok",
            }
        ]
        if paper_ids == ["compound-local"]
        else [],
    )
    monkeypatch.setattr(
        db,
        "search_paper_cards_v2",
        lambda *_args, **_kwargs: [
            {
                "paper_id": "2504.14191",
                "card_id": "paper-card-v2:2504.14191",
                "title": "AI Idea Bench 2025: AI Research Idea Generation Benchmark",
                "quality_flag": "ok",
            }
        ],
    )

    selected = service._select_paper_cards(
        query="A Compound AI Architecture for Scientific Discovery 논문 설명해줘",
        route=route,
        limit=2,
        metadata_filter=None,
        query_plan=frame.to_query_plan_dict(),
        query_frame=frame.to_dict(),
    )

    assert route.intent == "paper_lookup"
    assert [item["paper_id"] for item in selected] == ["compound-local"]


def test_select_paper_cards_falls_back_to_legacy_when_compare_scope_has_insufficient_cards(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    service = PaperAskV2Service(_build_searcher(db)[0])
    route = service._route(query="CNN vs ViT 비교해줘", metadata_filter=None)

    monkeypatch.setattr(
        service,
        "_ensure_cards_for_papers",
        lambda paper_ids: [{"paper_id": "2010.11929", "card_id": "paper-card-v2:2010.11929", "title": "ViT", "quality_flag": "ok"}]
        if paper_ids == ["alexnet-2012", "2010.11929"]
        else [],
    )

    with pytest.raises(AskV2FallbackToLegacy):
        service._select_paper_cards(
            query="CNN vs ViT 비교해줘",
            route=route,
            limit=2,
            metadata_filter=None,
            query_plan={
                "family": "paper_compare",
                "resolvedPaperIds": ["alexnet-2012", "2010.11929"],
                "expandedTerms": ["CNN", "ViT", "An Image is Worth 16x16 Words"],
            },
            query_frame={
                "family": "paper_compare",
                "resolved_source_ids": ["alexnet-2012", "2010.11929"],
                "expanded_terms": ["CNN", "ViT", "An Image is Worth 16x16 Words"],
            },
        )


def test_select_paper_cards_prefers_explicit_compare_titles_over_broad_resolved_aliases(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    service = PaperAskV2Service(_build_searcher(db)[0])
    route = service._route(query="RAG와 FiD를 비교해줘", metadata_filter=None)

    monkeypatch.setattr(
        service,
        "_ensure_cards_for_papers",
        lambda paper_ids: [
            {
                "paper_id": "rag-generic",
                "card_id": "paper-card-v2:rag-generic",
                "title": "Generic RAG Variant",
                "quality_flag": "ok",
            },
            {
                "paper_id": "rag-base",
                "card_id": "paper-card-v2:rag-base",
                "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "quality_flag": "ok",
            },
            {
                "paper_id": "rag-other",
                "card_id": "paper-card-v2:rag-other",
                "title": "Another RAG Paper",
                "quality_flag": "ok",
            },
        ]
        if paper_ids == ["rag-generic", "rag-base", "rag-other"]
        else [],
    )

    def _search(form: str, **_kwargs):
        if form == "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering":
            return [
                {
                    "paper_id": "fid-paper",
                    "card_id": "paper-card-v2:fid-paper",
                    "title": "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
                    "quality_flag": "ok",
                }
            ]
        if form == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks":
            return [
                {
                    "paper_id": "rag-base",
                    "card_id": "paper-card-v2:rag-base",
                    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                    "quality_flag": "ok",
                }
            ]
        return []

    monkeypatch.setattr(db, "search_paper_cards_v2", _search)

    selected = service._select_paper_cards(
        query="RAG와 FiD를 비교해줘",
        route=route,
        limit=2,
        metadata_filter=None,
        query_plan={
            "family": "paper_compare",
            "resolvedPaperIds": ["rag-generic", "rag-base", "rag-other"],
            "expandedTerms": [
                "RAG",
                "FiD",
                "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
            ],
        },
        query_frame={
            "family": "paper_compare",
            "resolved_source_ids": ["rag-generic", "rag-base", "rag-other"],
            "expanded_terms": [
                "RAG",
                "FiD",
                "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
            ],
        },
    )

    assert [item["paper_id"] for item in selected] == ["rag-base", "fid-paper"]


def test_select_paper_cards_limits_compare_scope_to_target_pair(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    service = PaperAskV2Service(_build_searcher(db)[0])
    route = service._route(query="RAG와 Self-RAG를 비교해줘", metadata_filter=None)

    monkeypatch.setattr(
        service,
        "_ensure_cards_for_papers",
        lambda paper_ids: [
            {
                "paper_id": "rag-base",
                "card_id": "paper-card-v2:rag-base",
                "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "quality_flag": "ok",
            },
            {
                "paper_id": "self-rag",
                "card_id": "paper-card-v2:self-rag",
                "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
                "quality_flag": "ok",
            },
            {
                "paper_id": "squai",
                "card_id": "paper-card-v2:squai",
                "title": "SQuAI: Scientific Question-Answering with Multi-Agent Retrieval-Augmented Generation",
                "quality_flag": "ok",
            },
        ]
        if paper_ids == ["rag-base", "self-rag", "squai"]
        else [],
    )

    monkeypatch.setattr(
        db,
        "search_paper_cards_v2",
        lambda *_args, **_kwargs: [
            {
                "paper_id": "survey-rag",
                "card_id": "paper-card-v2:survey-rag",
                "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
                "quality_flag": "ok",
            }
        ],
    )

    selected = service._select_paper_cards(
        query="RAG와 Self-RAG를 비교해줘",
        route=route,
        limit=6,
        metadata_filter=None,
        query_plan={
            "family": "paper_compare",
            "resolvedPaperIds": ["rag-base", "self-rag", "squai"],
            "expandedTerms": [
                "RAG",
                "Self-RAG",
                "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
            ],
        },
        query_frame={
            "family": "paper_compare",
            "resolved_source_ids": ["rag-base", "self-rag", "squai"],
            "expanded_terms": [
                "RAG",
                "Self-RAG",
                "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
            ],
        },
    )

    assert [item["paper_id"] for item in selected] == ["rag-base", "self-rag"]


def test_anchor_results_preserve_compare_source_diversity(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    service = PaperAskV2Service(_build_searcher(db)[0])
    route = service._route(query="BERT와 GPT 비교해줘", source_type="paper", metadata_filter=None)
    cards = [
        {
            "paper_id": "bert",
            "card_id": "paper-card-v2:bert",
            "title": "BERT",
            "quality_flag": "ok",
        },
        {
            "paper_id": "gpt",
            "card_id": "paper-card-v2:gpt",
            "title": "GPT",
            "quality_flag": "ok",
        },
    ]
    anchors = [
        {
            "anchor_id": f"bert-{idx}",
            "card_id": "paper-card-v2:bert",
            "paper_id": "bert",
            "excerpt": "bert anchor",
            "score": 0.9 - (0.01 * idx),
            "evidence_role": "result",
        }
        for idx in range(8)
    ]
    anchors.append(
        {
            "anchor_id": "gpt-0",
            "card_id": "paper-card-v2:gpt",
            "paper_id": "gpt",
            "excerpt": "gpt anchor",
            "score": 0.4,
            "evidence_role": "result",
        }
    )

    results = service._anchor_results(cards=cards, anchors=anchors, route=route)

    assert {str(item.metadata.get("paper_id") or "") for item in results[:2]} == {"bert", "gpt"}


def test_generate_answer_uses_section_native_prompt_for_explanation_query(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="2603.13017 방법을 설명해줘",
        source_type="paper",
        top_k=3,
    )

    assert vector_db.search_called is False
    assert payload["v2"]["routing"]["memoryForm"] == "section_cards"
    assert payload["sectionCards"]
    assert payload["answerProvenance"]["mode"].startswith("section_cards")
    assert "section-first" in (searcher.llm.last_prompt or "")
    assert payload["v2"]["runtimeExecution"]["used"] == "ask_v2"
    assert payload["v2"]["runtimeExecution"]["sectionDecision"] == "selected"


def test_section_card_materializer_builds_materialized_cards(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")

    class _StubDecision:
        route = "mini"
        provider = "openai"
        model = "gpt-5-mini"

        def to_dict(self):
            return {"route": self.route, "provider": self.provider, "model": self.model}

    class _StubLLM:
        model = "gpt-5-mini"

        def generate(self, prompt, max_tokens=0):  # noqa: ANN001
            _ = (prompt, max_tokens)
            return """{
              "title": "Section Summary",
              "sectionPath": "Method",
              "contextualSummary": "The paper distills exchange history into retrieval-oriented memory cards.",
              "sourceExcerpt": "Exchange history is distilled into retrieval cards.",
              "keyPoints": ["Distills exchange history", "Builds retrieval-oriented cards"],
              "scopeNotes": ["Single-user setting"],
              "confidence": 0.81
            }"""

    monkeypatch.setattr(
        "knowledge_hub.ai.section_card_materializer.get_llm_for_task",
        lambda *args, **kwargs: (_StubLLM(), _StubDecision(), []),
    )

    class _StubConfig:
        def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
            return default

        def get_provider_config(self, provider):  # noqa: ANN001
            _ = provider
            return {}

    payload = PaperSectionCardMaterializer(db, _StubConfig()).build_and_store(
        paper_id="2603.13017",
        allow_external=True,
        llm_mode="mini",
    )

    assert payload["status"] == "ok"
    assert payload["count"] >= 3
    rows = SectionCardV1Store(db.conn).list_paper_cards("2603.13017")
    assert {row["role"] for row in rows} >= {"problem", "method", "results"}
    assert any("retrieval-oriented memory cards" in row["contextual_summary"] for row in rows)


def test_section_card_materializer_blocks_weak_placeholder_sources(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_weak_paper_document_memory(db, "2603.13017")
    PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")

    class _StubConfig:
        def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
            return default

        def get_provider_config(self, provider):  # noqa: ANN001
            _ = provider
            return {}

    payload = PaperSectionCardMaterializer(db, _StubConfig()).build_and_store(
        paper_id="2603.13017",
        allow_external=False,
        llm_mode="fallback-only",
    )

    assert payload["status"] == "blocked"
    assert payload["blockReason"]
    assert payload["items"] == []
    assert SectionCardV1Store(db.conn).list_paper_cards("2603.13017") == []


def test_generate_answer_prefers_materialized_section_cards_when_present(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    PaperCardV2Builder(db).build_and_store(paper_id="2603.13017")
    SectionCardV1Store(db.conn).replace_paper_cards(
        paper_id="2603.13017",
        cards=[
            {
                "section_card_id": "paper-section-card-materialized:2603.13017:method",
                "paper_id": "2603.13017",
                "document_id": "paper:2603.13017",
                "role": "method",
                "title": "Materialized Method",
                "section_path": "Method",
                "unit_type": "section",
                "unit_ids": ["paper:2603.13017:method"],
                "contextual_summary": "Materialized method summary for retrieval card construction.",
                "source_excerpt": "Materialized excerpt.",
                "document_thesis": "Personalized memory improves retrieval quality.",
                "key_points": ["Retrieval card construction"],
                "scope_notes": ["Single-user setting"],
                "claims": ["claim_memory_1"],
                "concepts": ["concept_memory"],
                "confidence": 0.88,
                "provenance": {"builder": "test"},
                "search_text": "materialized method retrieval card construction",
                "origin": "materialized_v1",
                "generator_model": "test-model",
            }
        ],
    )

    searcher, vector_db = _build_searcher(db)
    payload = searcher.generate_answer(
        query="2603.13017 방법을 설명해줘",
        source_type="paper",
        top_k=3,
    )

    assert vector_db.search_called is False
    assert payload["v2"]["routing"]["memoryForm"] == "section_cards"
    assert any(item.get("origin") == "materialized_v1" for item in payload["sectionCards"])
    assert any("Materialized method summary" in str(item.get("contextualSummary")) for item in payload["sectionCards"])


def test_generate_answer_blocks_placeholder_section_cards_and_falls_back(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_weak_paper_document_memory(db, "2603.13017")
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="2603.13017 방법을 설명해줘",
        source_type="paper",
        top_k=3,
    )

    assert vector_db.search_called is False
    assert payload["v2"]["runtimeExecution"]["used"] == "ask_v2"
    assert payload["v2"]["runtimeExecution"]["sectionDecision"] == "blocked"
    assert payload["v2"]["runtimeExecution"]["sectionBlockReason"]
    assert payload["v2"]["routing"]["memoryForm"] != "section_cards"


def test_generate_answer_uses_ontology_first_route_for_relation_query(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    db.add_entity_alias("memory", "concept_memory")
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="memory 관계를 설명해줘",
        source_type="paper",
        top_k=3,
    )

    assert vector_db.search_called is False
    assert payload["v2"]["routing"]["mode"] == "ontology-first"
    assert payload["v2"]["evidenceVerification"]["anchorIdsUsed"]


def test_generate_answer_temporal_query_flags_weak_evidence_without_temporal_support(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    searcher, _vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="latest memory paper",
        source_type="paper",
        top_k=3,
    )

    assert payload["status"] == "no_result"
    assert payload["v2"]["routing"]["intent"] == "temporal"
    assert "temporal" in payload["v2"]["evidenceVerification"]["unsupportedFields"]
    assert payload["v2"]["fallback"]["used"] is True
    assert payload["v2"]["fallback"]["reason"] == "ask_v2_missing"


def test_generate_answer_claim_conflict_marks_consensus_and_fallback(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    db.upsert_claim(
        claim_id="claim_memory_2",
        claim_text="The method performs worse than the baseline on MemoryBench.",
        subject_entity_id="paper:2603.13017",
        predicate="reports",
        object_literal="baseline",
        confidence=0.81,
        evidence_ptrs=[],
        source="test",
    )
    db.upsert_claim_normalization(
        claim_id="claim_memory_1",
        normalization_version="v1",
        status="normalized",
        dataset="MemoryBench",
        metric="compression ratio",
        comparator="baseline",
        result_direction="better",
        result_value_text="11x compression",
        evidence_strength="strong",
    )
    db.upsert_claim_normalization(
        claim_id="claim_memory_2",
        normalization_version="v1",
        status="normalized",
        dataset="MemoryBench",
        metric="compression ratio",
        comparator="baseline",
        result_direction="worse",
        result_value_text="worse than baseline",
        evidence_strength="strong",
    )
    searcher, _vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="2603.13017 metric comparison",
        source_type="paper",
        top_k=3,
    )

    assert payload["v2"]["consensus"]["conflictCount"] >= 1
    assert payload["claimCards"]
    assert payload["claimAlignment"]["groups"]
    assert payload["answerProvenance"]["mode"] == "claim_cards_conflicted"
    assert payload["v2"]["fallback"]["used"] is True


def test_generate_answer_paper_claim_verification_exposes_supported_and_contradicted_claims(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    _seed_document_memory(db, "2603.13017")
    db.upsert_claim(
        claim_id="claim_memory_conflict",
        claim_text="The method achieves 13x compression.",
        subject_entity_id="paper:2603.13017",
        predicate="reports",
        confidence=0.72,
        evidence_ptrs=[],
        source="test",
    )
    db.upsert_claim_normalization(
        claim_id="claim_memory_1",
        normalization_version="v1",
        status="normalized",
        dataset="MemoryBench",
        metric="compression ratio",
        result_value_numeric=11.0,
        evidence_strength="strong",
    )
    db.upsert_claim_normalization(
        claim_id="claim_memory_conflict",
        normalization_version="v1",
        status="normalized",
        metric="compression ratio",
        result_value_numeric=13.0,
        evidence_strength="strong",
    )
    searcher, _vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="2603.13017 결과와 지표를 설명해줘",
        source_type="paper",
        top_k=3,
    )

    statuses = {item["claimId"]: item["status"] for item in payload["v2"]["claimVerification"]}
    assert statuses["claim_memory_1"] == "supported"
    assert statuses["claim_memory_conflict"] == "contradicted"
    assert payload["v2"]["fallback"]["used"] is True


def test_generate_answer_web_temporal_query_requires_stronger_temporal_grounding(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    url = "https://example.com/memory-guide"
    _seed_web_document_memory(db, url, with_temporal_markers=False)
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="latest memory guide",
        source_type="web",
        top_k=3,
        metadata_filter={"canonical_url": url},
    )

    assert vector_db.search_called is False
    assert payload["status"] == "no_result"
    assert payload["v2"]["routing"]["sourceKind"] == "web"
    assert "temporal_version_grounding" in payload["v2"]["evidenceVerification"]["unsupportedFields"]
    assert payload["v2"]["fallback"]["used"] is True
    assert payload["v2"]["fallback"]["reason"].startswith("ask_v2_weak_evidence:")


def test_generate_answer_rebuilds_stale_web_card_when_upstream_memory_is_newer(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    url = "https://example.com/memory-guide"
    _seed_web_document_memory(db, url, with_temporal_markers=True)
    searcher, vector_db = _build_searcher(db)

    searcher.generate_answer(
        query="latest memory guide",
        source_type="web",
        top_k=3,
        metadata_filter={"canonical_url": url},
    )
    db.conn.execute(
        "UPDATE web_cards_v2 SET updated_at = ? WHERE canonical_url = ?",
        ("2026-03-01T00:00:00+00:00", url),
    )
    db.conn.execute(
        """
        UPDATE document_memory_units
        SET contextual_summary = ?, updated_at = ?
        WHERE document_id = ? AND unit_type = 'document_summary'
        """,
        (
            "This guide now includes fresher version-aware update policy guidance.",
            datetime(2026, 3, 28, tzinfo=timezone.utc).isoformat(),
            make_web_note_id(url),
        ),
    )
    db.conn.commit()

    payload = searcher.generate_answer(
        query="latest memory guide",
        source_type="web",
        top_k=3,
        metadata_filter={"canonical_url": url},
    )

    refreshed = db.get_web_card_v2_by_url(url)
    assert vector_db.search_called is False
    assert refreshed is not None
    assert refreshed["page_core"] == "This guide now includes fresher version-aware update policy guidance."
    assert payload["v2"]["routing"]["selected_card_ids"]


def test_generate_answer_vault_query_uses_vault_card_lane(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_vault_document_memory(db, "vault:memory-design")
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="memory design note summary",
        source_type="vault",
        top_k=3,
        metadata_filter={"note_id": "vault:memory-design"},
    )

    assert vector_db.search_called is False
    assert payload["v2"]["routing"]["sourceKind"] == "vault"
    assert payload["v2"]["routing"]["selected_card_ids"]
    assert payload["v2"]["evidenceVerification"]["anchorIdsUsed"]


def test_generate_answer_vault_file_path_scope_accepts_note_source_type(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    note_id = "vault:memory-design"
    _seed_vault_document_memory(db, note_id)
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="Memory Design Note.md 요약",
        source_type="vault",
        top_k=3,
        metadata_filter={"file_path": "vault/Memory Design Note.md"},
    )

    assert vector_db.search_called is False
    assert payload["v2"]["routing"]["sourceKind"] == "vault"
    assert payload["v2"]["routing"]["selected_card_ids"] == [f"vault-card-v2:{note_id}"]


def test_generate_answer_project_query_uses_ephemeral_repo_cards(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("# Repo\nService architecture summary.\n", encoding="utf-8")
    (repo_root / "service.py").write_text("def build_pipeline():\n    return 'pipeline'\n", encoding="utf-8")
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    searcher, vector_db = _build_searcher(db)

    payload = searcher.generate_answer(
        query="service architecture pipeline",
        source_type="project",
        top_k=3,
        metadata_filter={"repo_path": str(repo_root)},
    )

    assert vector_db.search_called is False
    assert payload["v2"]["routing"]["sourceKind"] == "project"
    assert payload["v2"]["routing"]["selected_card_ids"]
    assert payload["v2"]["evidenceVerification"]["anchorIdsUsed"]
    assert payload["claimCards"]
    assert all(item["sourceKind"] == "project" for item in payload["claimCards"])
    assert db.list_claim_cards(source_kind="project", limit=10) == []
