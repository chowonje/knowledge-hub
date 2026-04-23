from __future__ import annotations
import json
from pathlib import Path

from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.document_memory import DocumentMemoryBuilder
from knowledge_hub.papers.card_v2_builder import PaperCardV2Builder
from knowledge_hub.papers import memory_builder as memory_builder_module
from knowledge_hub.papers.memory_builder import PaperMemoryBuilder
from knowledge_hub.papers.memory_extraction import PaperMemoryExtractionError, PaperMemorySchemaExtractor
from knowledge_hub.papers.memory_eval import PaperMemoryEvalCase, PaperMemoryEvalHarness
from knowledge_hub.papers.memory_projection import PROJECTED_ENRICHED_VERSION, PROJECTED_VERSION, PaperMemoryProjector
from knowledge_hub.papers.memory_retriever import PaperMemoryRetriever


def _seed_paper_with_note(db: SQLiteDatabase, tmp_path: Path, *, paper_id: str = "2603.13017") -> None:
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Personalized Agent Memory",
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 5,
            "notes": "compressed retrieval memory for agent history",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_ontology_entity(
        entity_id=f"paper:{paper_id}",
        entity_type="paper",
        canonical_name="Personalized Agent Memory",
        source="test",
    )
    db.upsert_ontology_entity(
        entity_id="concept_memory",
        entity_type="concept",
        canonical_name="agent memory",
        source="test",
    )
    db.add_relation("paper", paper_id, "paper_uses_concept", "concept", "concept_memory", confidence=0.91)
    db.upsert_note(
        note_id=f"paper:{paper_id}",
        title="[논문] Personalized Agent Memory",
        content=(
            "# Personalized Agent Memory\n\n"
            "## 요약\n\n"
            "개인화된 에이전트 메모리를 검색 가능한 압축 계층으로 정리한다.\n\n"
            "## Abstract\n\n"
            "The paper studies long-term memory retrieval for developer-agent interactions.\n\n"
            "## 방법\n\n"
            "- exchange를 구조화된 card로 증류한다.\n\n"
            "## 결과\n\n"
            "- 11배 압축을 달성한다.\n\n"
            "## 한계\n\n"
            "- 단일 사용자 환경 중심 평가다.\n"
        ),
        source_type="paper",
        para_category="resource",
        metadata={"arxiv_id": paper_id, "quality_flag": "ok"},
    )
    db.upsert_claim(
        claim_id="claim_memory_1",
        claim_text="The approach achieves around 11x compression over raw exchanges.",
        subject_entity_id=f"paper:{paper_id}",
        predicate="improves",
        object_literal="compression efficiency",
        confidence=0.93,
        evidence_ptrs=[{"note_id": f"paper:{paper_id}"}],
        source="test",
    )


def test_build_paper_memory_card_from_note_claim_and_concepts(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)

    item = PaperMemoryBuilder(db).build_and_store(paper_id="2603.13017")

    assert item["paper_id"] == "2603.13017"
    assert item["quality_flag"] == "ok"
    assert "압축 계층" in item["paper_core"]
    assert item["concept_links"] == ["agent memory"]
    assert item["claim_refs"] == ["claim_memory_1"]


def test_build_paper_memory_card_prioritizes_bridge_math_concepts_in_visible_links(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2312.10997"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
            "authors": "RAG Authors",
            "year": 2023,
            "field": "AI",
            "importance": 7,
            "notes": "survey note",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_note(
        note_id=f"paper:{paper_id}",
        title="[논문] RAG Survey",
        content="# RAG Survey\n\n## 요약\n\nretrieval plus generation.\n",
        source_type="paper",
        para_category="resource",
        metadata={"arxiv_id": paper_id, "quality_flag": "ok"},
    )

    for concept_id, name in [
        ("concept_llm", "Large Language Model"),
        ("concept_rag", "Retrieval-Augmented Generation"),
        ("concept_generation", "Generation"),
        ("concept_retrieval", "Retrieval"),
        ("concept_inner_product", "Inner Product"),
        ("concept_norm", "Norm"),
        ("concept_vector", "Vector"),
    ]:
        db.upsert_ontology_entity(concept_id, "concept", name, source="test")

    for concept_id, snippet in [
        ("concept_llm", "llm keyword"),
        ("concept_rag", "rag keyword"),
        ("concept_generation", "generation keyword"),
        ("concept_retrieval", "retrieval keyword"),
    ]:
        db.add_relation(
            "paper",
            paper_id,
            "uses",
            "concept",
            concept_id,
            evidence_text=json.dumps(
                {
                    "source": "paper_sync_keywords_targeted",
                    "evidence_ptrs": [{"type": "note", "path": f"/tmp/{snippet}.md"}],
                }
            ),
            confidence=1.0,
        )

    for concept_id, name in [
        ("concept_inner_product", "Inner Product"),
        ("concept_norm", "Norm"),
        ("concept_vector", "Vector"),
    ]:
        db.add_relation(
            "paper",
            paper_id,
            "uses",
            "concept",
            concept_id,
            evidence_text=json.dumps(
                {
                    "note_id": "AI/AI_Papers/Papers/Math Bridge - Retrieval-Augmented Generation for Large Language Models A Survey.md",
                    "paper_title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
                    "concept_name": name,
                }
            ),
            confidence=1.0,
        )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert item["concept_links"][:3] == ["Inner Product", "Norm", "Vector"]
    assert "Large Language Model" in item["concept_links"]


def test_concept_names_drop_heuristic_title_fallback_when_real_concepts_exist():
    rows = [
        {
            "canonical_name": "feedforward neural networks",
            "source": "paper_memory_title_fallback",
            "confidence": 1.0,
        },
        {
            "canonical_name": "Jacobian",
            "source": "vault_math_bridge_sync",
            "confidence": 1.0,
            "reason_json": json.dumps(
                {
                    "legacy_evidence_text": json.dumps(
                        {"note_id": "AI/AI_Papers/Papers/Math Bridge - Understanding Difficulty of Training Deep Networks.md"},
                        ensure_ascii=False,
                    )
                },
                ensure_ascii=False,
            ),
        },
        {
            "canonical_name": "Variance and Covariance",
            "source": "vault_math_bridge_sync",
            "confidence": 1.0,
            "reason_json": json.dumps(
                {
                    "legacy_evidence_text": json.dumps(
                        {"note_id": "AI/AI_Papers/Papers/Math Bridge - Understanding Difficulty of Training Deep Networks.md"},
                        ensure_ascii=False,
                    )
                },
                ensure_ascii=False,
            ),
        },
    ]

    names = memory_builder_module._concept_names(rows, limit=8)

    assert names == ["Jacobian", "Variance and Covariance"]


def test_paper_memory_projector_is_deterministic_for_same_document_memory_units(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    DocumentMemoryBuilder(db).build_and_store_paper(paper_id="2603.13017")
    projector = PaperMemoryProjector(db)

    first = projector.project(paper_id="2603.13017")
    second = projector.project(paper_id="2603.13017")

    assert first is not None
    assert second is not None
    assert first.to_record() == second.to_record()


def test_paper_memory_builder_needs_rebuild_tracks_document_memory_freshness(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    DocumentMemoryBuilder(db).build_and_store_paper(paper_id="2603.13017")
    builder = PaperMemoryBuilder(db)
    builder.build_and_store(paper_id="2603.13017")

    assert builder.needs_rebuild("2603.13017") is False

    db.conn.execute(
        "UPDATE document_memory_units SET updated_at = ? WHERE document_id = ?",
        ("2099-01-01 00:00:00", "paper:2603.13017"),
    )
    db.conn.commit()

    assert builder.needs_rebuild("2603.13017") is True


def test_cutover_builder_preserves_paper_memory_schema_and_payload_keys(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    item = PaperMemoryBuilder(db).build_and_store(paper_id="2603.13017")

    expected_record_keys = {
        "memory_id",
        "paper_id",
        "source_note_id",
        "title",
        "paper_core",
        "problem_context",
        "method_core",
        "evidence_core",
        "limitations",
        "concept_links",
        "claim_refs",
        "published_at",
        "evidence_window",
        "search_text",
        "quality_flag",
        "version",
    }
    assert expected_record_keys.issubset(set(item))
    assert item["version"] in {PROJECTED_VERSION, PROJECTED_ENRICHED_VERSION}


def test_paper_memory_store_roundtrip_preserves_hidden_cause_slots(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    card = memory_builder_module.PaperMemoryCard(
        memory_id="paper-memory:hidden-cause",
        paper_id="2603.13017",
        title="Hidden Cause Paper",
        paper_core="Compact retrieval memory.",
        problem_context="Long-session recall drifts without structure.",
        method_core="Distill exchanges into compact cards.",
        evidence_core="Improves retrieval precision on longitudinal tasks.",
        limitations="Single-user evaluation only.",
        concept_links=["agent memory"],
        claim_refs=["claim:1"],
        formal_cause={
            "summary": "A card-structured episodic memory schema.",
            "basis": "author_stated",
            "confidence": 0.93,
            "coverage": "complete",
            "evidence_refs": ["method.section"],
            "warnings": [],
        },
        final_cause={
            "authorStatedSummary": "Support durable long-horizon developer-agent memory.",
            "inferredSummary": "",
            "basis": "author_stated",
            "confidence": 0.88,
            "coverage": "complete",
            "evidenceRefs": ["abstract.goal"],
            "warnings": [],
        },
        quality_flag="ok",
    )

    stored = db.upsert_paper_memory_card(card=card.to_record())

    assert stored["formal_cause"]["summary"] == "A card-structured episodic memory schema."
    assert stored["formal_cause"]["basis"] == "author_stated"
    assert stored["final_cause"]["author_stated_summary"] == "Support durable long-horizon developer-agent memory."
    assert stored["final_cause"]["basis"] == "author_stated"
    public_payload = memory_builder_module.PaperMemoryCard.from_row(stored).to_payload()
    assert "formalCause" not in public_payload
    assert "finalCause" not in public_payload


def test_paper_memory_card_from_legacy_row_defaults_empty_cause_slots():
    card = memory_builder_module.PaperMemoryCard.from_row(
        {
            "memory_id": "paper-memory:legacy",
            "paper_id": "legacy-paper",
            "title": "Legacy Paper",
            "paper_core": "Legacy core.",
        }
    )

    assert card is not None
    assert card.formal_cause == {}
    assert card.final_cause == {}


def test_projected_enrichment_only_fills_empty_slots_without_overwrite(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    builder = PaperMemoryBuilder(db)
    projected = memory_builder_module.PaperMemoryCard(
        memory_id="paper-memory:test",
        paper_id="2603.13017",
        source_note_id="paper:2603.13017",
        title="Projected Memory Card",
        paper_core="Projected paper core",
        problem_context="Projected problem context",
        method_core="",
        evidence_core="",
        limitations="Projected limitations",
        concept_links=["agent memory"],
        claim_refs=["claim_memory_1"],
        quality_flag="needs_review",
        version=PROJECTED_VERSION,
    )
    extraction = memory_builder_module.PaperMemoryExtractionV1.from_dict(
        {
            "thesis": "Extraction thesis should not overwrite projected paper core.",
            "methodCore": "Extraction fills empty method slot.",
            "evidenceCore": "Extraction fills empty evidence slot.",
            "limitations": "Extraction limitations should not overwrite projected limitations.",
            "conceptLinks": ["episodic memory"],
            "claimRefs": ["claim_memory_2"],
            "qualityFlag": "ok",
        }
    )

    merged = builder._merge_extraction_into_card(
        card=projected,
        extraction=extraction,
        has_explicit_limitations=True,
    )

    assert merged.paper_core == "Projected paper core"
    assert merged.limitations == "Projected limitations"
    assert merged.method_core == "Extraction fills empty method slot."
    assert merged.evidence_core == "Extraction fills empty evidence slot."
    assert merged.quality_flag == "needs_review"
    assert merged.version == PROJECTED_ENRICHED_VERSION


def test_projected_enrichment_prefers_legacy_when_projected_slots_are_polluted(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    builder = PaperMemoryBuilder(db)
    projected = memory_builder_module.PaperMemoryCard(
        memory_id="paper-memory:phi4",
        paper_id="2412.08905",
        source_note_id="paper:2412.08905",
        title="Phi-4 Technical Report",
        paper_core="Phi-4 Technical Report Marah Abdin Jyoti Aneja Harkirat Behl and many author names.",
        problem_context="Phi-4 Technical Report Marah Abdin Jyoti Aneja Harkirat Behl and many author names.",
        method_core="Table 4: Ablations on the allocation of 75% of training tokens to synthetic data.",
        evidence_core="MMLU 84.8 GPQA 56.1 MATH 80.4 HumanEval 82.6 DROP 75.5 IFEval 63.0.",
        limitations="Phi-4 Technical Report Marah Abdin Jyoti Aneja Harkirat Behl and many author names.",
        concept_links=["phi-4"],
        claim_refs=[],
        quality_flag="needs_review",
        version=PROJECTED_VERSION,
    )
    legacy = memory_builder_module.PaperMemoryCard(
        memory_id="paper-memory:phi4",
        paper_id="2412.08905",
        source_note_id="paper:2412.08905",
        title="Phi-4 Technical Report",
        paper_core="Synthetic-data-heavy training strengthens small-model reasoning.",
        problem_context="Small models need stronger reasoning without losing latency and cost advantages.",
        method_core="Use synthetic data, multi-agent prompting, self-revision, and two-stage DPO post-training.",
        evidence_core="phi-4 reaches 84.8 MMLU, 56.1 GPQA, and 82.6 HumanEval on the reported benchmark suite.",
        limitations="SimpleQA remains weak and the model is not the best system on every benchmark.",
        concept_links=["phi-4", "synthetic data"],
        claim_refs=[],
        quality_flag="ok",
        version=PROJECTED_ENRICHED_VERSION,
    )

    merged = builder._merge_additive_card_enrichment(base=projected, enrichment=legacy)

    assert merged.paper_core == legacy.paper_core
    assert merged.problem_context == legacy.problem_context
    assert merged.method_core == legacy.method_core
    assert "MMLU" in merged.evidence_core
    assert merged.limitations == legacy.limitations
    assert merged.quality_flag == "ok"


def test_projected_enrichment_prefers_legacy_over_front_matter_and_table_captions(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    builder = PaperMemoryBuilder(db)
    projected = memory_builder_module.PaperMemoryCard(
        memory_id="paper-memory:deepseek-r1",
        paper_id="2501.12948",
        source_note_id="paper:2501.12948",
        title="DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning",
        paper_core="DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning",
        problem_context="DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning research@deepseek.com Abstract General reasoning represents a long-standing challenge.",
        method_core="0 2000 4000 6000 Steps Figure 1 | AIME accuracy of DeepSeek-R1-Zero during training.",
        evidence_core="Table 3 | Experimental results at each stage of DeepSeek-R1. Numbers in bold denote statistical significance.",
        limitations="limitations of current human priors. Specifically, we first engage human annotators to convert the reasoning trace.",
        concept_links=["DeepSeek-R1"],
        claim_refs=[],
        quality_flag="needs_review",
        version=PROJECTED_VERSION,
    )
    legacy = memory_builder_module.PaperMemoryCard(
        memory_id="paper-memory:deepseek-r1",
        paper_id="2501.12948",
        source_note_id="paper:2501.12948",
        title="DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning",
        paper_core="강화학습으로 언어모델의 추론 능력을 키우는 접근이다.",
        problem_context="기존 추론 학습은 사람이 제공한 예시에 많이 의존해 더 나은 비인간적 추론 경로를 탐색하기 어렵다는 문제를 다룬다.",
        method_core="GRPO 기반 강화학습으로 정책을 업데이트하고, DeepSeek-R1-Zero에서는 강화학습만으로 초기 추론 능력을 끌어올린다.",
        evidence_core="논문은 단계별 실험 결과를 제시하며 일부 성능 차이가 통계적으로 유의하다고 보고한다.",
        limitations="일부 단계에서는 추론 흔적을 사람이 더 자연스러운 대화체로 바꾸는 과정이 들어가므로 인간 priors의 한계에서 완전히 자유롭지 않다.",
        concept_links=["DeepSeek-R1", "GRPO"],
        claim_refs=[],
        quality_flag="ok",
        version=PROJECTED_ENRICHED_VERSION,
    )

    merged = builder._merge_additive_card_enrichment(base=projected, enrichment=legacy)

    assert merged.problem_context == legacy.problem_context
    assert merged.method_core == legacy.method_core
    assert merged.evidence_core == legacy.evidence_core
    assert merged.limitations == legacy.limitations
    assert merged.quality_flag == "ok"


def test_structured_summary_slot_values_prefer_key_results_over_polluted_when_it_matters():
    slots = memory_builder_module._structured_summary_slot_values(
        {
            "summary": {
                "oneLine": "강화학습으로 추론 능력을 키우는 접근이다.",
                "keyResults": [
                    "논문은 단계별 실험 결과를 제시하며 일부 성능 차이가 통계적으로 유의하다고 보고한다."
                ],
                "whenItMatters": "Junxiao Song proposed the GRPO algorithm, implemented the initial version, and introduced rule-based rewards for math tasks.",
            }
        }
    )

    assert slots["evidence_core"] == "논문은 단계별 실험 결과를 제시하며 일부 성능 차이가 통계적으로 유의하다고 보고한다."
    assert "Junxiao Song" not in slots["evidence_core"]


def test_projected_enrichment_replaces_author_stub_and_acknowledgement_evidence(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    builder = PaperMemoryBuilder(db)
    projected = memory_builder_module.PaperMemoryCard(
        memory_id="paper-memory:reward",
        paper_id="RewardIsEnough",
        source_note_id="paper:RewardIsEnough",
        title="Reward is Enough",
        paper_core="Reward maximization can unify intelligence.",
        problem_context="Different capabilities are often framed with different objectives.",
        method_core="D. Silver, S.",
        evidence_core=(
            "This work was supported in part by institutional grants and gifts from industry partners; "
            "we thank legal advisory for feedback."
        ),
        limitations="The paper is more programmatic than implementation-specific.",
        concept_links=["reinforcement learning"],
        claim_refs=[],
        quality_flag="needs_review",
        version=PROJECTED_VERSION,
    )
    legacy = memory_builder_module.PaperMemoryCard(
        memory_id="paper-memory:reward",
        paper_id="RewardIsEnough",
        source_note_id="paper:RewardIsEnough",
        title="Reward is Enough",
        paper_core="Reward maximization is proposed as a unifying objective for intelligence.",
        problem_context="The paper asks whether separate cognitive abilities need separate objectives.",
        method_core="The paper frames reinforcement learning as the general algorithmic route to maximize reward across tasks.",
        evidence_core="The paper argues from experimental reinforcement-learning progress across games, robotics, and large-scale control benchmarks.",
        limitations="It leaves the nature of the agent and embodiment assumptions under-specified.",
        concept_links=["reinforcement learning", "agi"],
        claim_refs=[],
        quality_flag="ok",
        version=PROJECTED_ENRICHED_VERSION,
    )

    merged = builder._merge_additive_card_enrichment(base=projected, enrichment=legacy)

    assert merged.method_core == legacy.method_core
    assert merged.evidence_core == legacy.evidence_core
    assert merged.quality_flag == "ok"


def test_projected_enrichment_promotes_fully_supported_card_without_ok_candidate(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    builder = PaperMemoryBuilder(db)
    projected = memory_builder_module.PaperMemoryCard(
        memory_id="paper-memory:world-models",
        paper_id="1803.10122",
        source_note_id="paper:1803.10122",
        title="World Models",
        paper_core="World Models",
        problem_context="Page excerpt only.",
        method_core="",
        evidence_core="",
        limitations="",
        concept_links=[],
        claim_refs=[],
        quality_flag="needs_review",
        version=PROJECTED_VERSION,
    )
    enrichment = memory_builder_module.PaperMemoryCard(
        memory_id="paper-memory:world-models",
        paper_id="1803.10122",
        source_note_id="paper:1803.10122",
        title="World Models",
        paper_core="The paper studies whether compact latent world models can support control in reinforcement learning.",
        problem_context="Model-based agents need compressed state and future prediction to act in image-based environments.",
        method_core="A VAE compresses observations, an RNN predicts latent dynamics, and a controller acts from the latent state and hidden state.",
        evidence_core="Experiments on CarRacing and Doom show the learned world model supports competitive control performance in simulation.",
        limitations="Transfer back to the real environment depends on calibrated stochasticity and limited model capacity.",
        concept_links=["world models"],
        claim_refs=[],
        quality_flag="unscored",
        version=PROJECTED_ENRICHED_VERSION,
    )

    merged = builder._merge_additive_card_enrichment(base=projected, enrichment=enrichment)

    assert merged.paper_core == enrichment.paper_core
    assert merged.method_core == enrichment.method_core
    assert merged.evidence_core == enrichment.evidence_core
    assert merged.quality_flag == "ok"


def test_build_paper_memory_card_materializes_paper_card_v2(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13017"
    _seed_paper_with_note(db, tmp_path, paper_id=paper_id)

    assert db.get_paper_card_v2(paper_id) is None
    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert item["paper_id"] == paper_id
    assert "card_v2_materialize_error" not in item
    card = db.get_paper_card_v2(paper_id)
    assert card is not None
    assert str(card.get("card_id") or "") == f"paper-card-v2:{paper_id}"


def test_build_and_store_surfaces_card_v2_failure_when_v2_builder_raises(tmp_path, monkeypatch):
    """Regression: card v2 stage must not fail silently after paper_memory persists."""
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13017"
    _seed_paper_with_note(db, tmp_path, paper_id=paper_id)

    def _boom(self, *, paper_id: str) -> dict:
        raise RuntimeError("v2 persistence simulated failure")

    monkeypatch.setattr(
        "knowledge_hub.papers.card_v2_builder.PaperCardV2Builder.build_and_store",
        _boom,
    )
    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)
    assert item["paper_id"] == paper_id
    assert db.get_paper_memory_card(paper_id) is not None
    assert db.get_paper_card_v2(paper_id) is None
    assert "RuntimeError" in str(item.get("card_v2_materialize_error") or "")
    assert "simulated failure" in str(item.get("card_v2_materialize_error") or "")


def test_build_paper_memory_card_falls_back_to_claims_when_note_missing(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13018"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Fallback Memory Paper",
            "authors": "B. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "fallback notes",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_ontology_entity(entity_id=f"paper:{paper_id}", entity_type="paper", canonical_name="Fallback Memory Paper")
    db.upsert_claim(
        claim_id="claim_fallback_1",
        claim_text="Claim-only fallback still captures retrieval-oriented memory compression.",
        subject_entity_id=f"paper:{paper_id}",
        predicate="related_to",
        object_literal="memory compression",
        confidence=0.81,
        evidence_ptrs=[],
        source="test",
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert item["paper_id"] == paper_id
    assert item["claim_refs"] == ["claim_fallback_1"]
    assert "Claim-only fallback" in item["paper_core"] or "Claim-only fallback" in item["evidence_core"]


def test_build_paper_memory_card_falls_back_to_paper_metadata_and_translated_text(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13019"
    translated_path = tmp_path / "2603.13019_translated.md"
    translated_path.write_text(
        "This paper proposes a compact memory card for paper retrieval. It improves search efficiency.",
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Metadata Only Paper",
            "authors": "C. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "metadata fallback note",
            "pdf_path": "",
            "text_path": "",
            "translated_path": str(translated_path),
        }
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert item["paper_id"] == paper_id
    assert item["paper_core"]
    assert "compact memory card" in item["paper_core"] or "compact memory card" in item["problem_context"]


def test_build_paper_memory_card_uses_pdf_excerpt_when_only_pdf_path_exists(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13019_pdf"
    pdf_path = tmp_path / "2603.13019.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(
        memory_builder_module,
        "extract_pdf_text_excerpt",
        lambda *_args, **_kwargs: (
            "Residual connections stabilize optimization for very deep networks. "
            "The method enables substantially deeper image models. "
            "Experiments improve ImageNet accuracy over prior plain networks."
        ),
    )
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "PDF Fallback Paper",
            "authors": "C. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": str(pdf_path),
            "text_path": "",
            "translated_path": "",
        }
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert item["paper_id"] == paper_id
    assert "Residual connections stabilize optimization" in item["paper_core"]
    assert "deeper image models" in item["method_core"]
    assert "ImageNet accuracy" in item["evidence_core"]


def test_build_paper_memory_card_derives_title_based_concepts_when_relations_missing(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13019b"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications",
            "authors": "C. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "Long-horizon memory benchmark for agent systems.",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert "AMA-Bench" in item["concept_links"]
    assert "Long-Horizon Memory" in item["concept_links"]
    assert "Agentic Applications" in item["concept_links"]


def test_title_concept_candidates_reject_file_and_math_junk():
    candidates = memory_builder_module._title_concept_candidates(
        "AGENTS.md: $π_0$ for Large Language Models"
    )

    assert "AGENTS.md" not in candidates
    assert "$π_0$" not in candidates
    assert "Large Language Model" in candidates


def test_title_concept_candidates_reject_sentence_like_spans():
    candidates = memory_builder_module._title_concept_candidates(
        "Across Diverse Tasks: Affordable and Efficient Robotics"
    )

    assert "Across Diverse Tasks" not in candidates
    assert "Affordable and Efficient Robotics" not in candidates


def test_title_concept_candidates_drop_generic_junk_and_canonicalize_trusted_titles():
    candidates = memory_builder_module._title_concept_candidates(
        "Benchmark: AI Agents for Large Language Models"
    )

    assert "Benchmark" not in candidates
    assert "AI Agent" in candidates
    assert "Large Language Model" in candidates


def test_title_concept_candidates_promote_curated_general_concepts():
    candidates = memory_builder_module._title_concept_candidates(
        "Prompt Injection Attacks in Reinforcement Learning"
    )

    assert "Prompt Injection" in candidates
    assert "Reinforcement Learning" in candidates


def test_raw_fallback_summary_with_usable_slots_is_trusted():
    payload = {
        "status": "ok",
        "parserUsed": "raw",
        "fallbackUsed": True,
        "documentMemoryDiagnostics": {
            "structuredSectionsDetected": 0,
            "parseArtifactPath": "",
        },
        "summary": {
            "oneLine": "Temporal knowledge graph memory layer for AI agents.",
            "problem": "Agents need evolving memory over long-running sessions.",
            "coreIdea": "Store episodic and semantic memories in a temporal graph.",
            "methodSteps": ["Represent episodes and facts in the same temporal graph."],
            "keyResults": ["Improves long-horizon retrieval against a baseline memory layer."],
            "limitations": ["Evaluation breadth is still limited."],
        },
    }

    assert memory_builder_module._structured_summary_payload_is_trustworthy(payload) is True


def test_citation_stub_summary_value_is_unusable():
    assert memory_builder_module._summary_value_is_unusable("citations: 0") is True


def test_build_paper_memory_card_backfills_title_based_concepts_to_ontology(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13019c"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Emergent Introspective Awareness in Large Language Models",
            "authors": "C. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "Study of introspection in large language models.",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )

    PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)
    concept_names = [row.get("canonical_name") for row in db.get_paper_concepts(paper_id)]

    assert "Emergent Introspective Awareness" in concept_names
    assert "Large Language Model" in concept_names


def test_build_paper_memory_card_promotes_trusted_title_concepts_to_seed_source(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13019seed"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "AI Agents for Large Language Models",
            "authors": "C. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "Trusted title concepts should be promoted.",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )

    PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)
    concept_rows = db.get_paper_concepts(paper_id)

    by_name = {row.get("canonical_name"): row for row in concept_rows}
    assert "AI Agent" in by_name
    assert "Large Language Model" in by_name
    assert by_name["AI Agent"]["source"] == "paper_title_seed"
    assert by_name["Large Language Model"]["source"] == "paper_title_seed"


def test_build_paper_memory_card_reuses_existing_concept_entity_by_canonical_name(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13019d"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Analyzing and Improving Chain-of-Thought Monitorability Through Information Theory",
            "authors": "C. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "Monitorability study.",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_ontology_entity(
        entity_id="existing_information_theory",
        entity_type="concept",
        canonical_name="Information Theory",
        source="test",
    )

    PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)
    rows = db.get_paper_concepts(paper_id)

    assert any(row.get("entity_id") == "existing_information_theory" for row in rows)


def test_build_paper_memory_card_adds_section_headings_to_search_text(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13020"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Temporal Routing For Memory Retrieval",
            "authors": "D. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "temporal routing study",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_note(
        note_id=f"paper:{paper_id}",
        title="[논문] Temporal Routing For Memory Retrieval",
        content=(
            "# Temporal Routing For Memory Retrieval\n\n"
            "## Abstract\n\n"
            "The paper studies date-aware memory routing.\n\n"
            "## Temporal Routing\n\n"
            "A temporal router prioritizes event-date grounded evidence.\n\n"
            "## Implementation Notes\n\n"
            "The builder adds retrieval hints for routing and updates.\n"
        ),
        source_type="paper",
        metadata={"arxiv_id": paper_id},
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)
    results = PaperMemoryRetriever(db).search("temporal routing", limit=3, include_refs=False)

    assert "temporal routing" in item["search_text"].casefold()
    assert results
    assert results[0]["paperId"] == paper_id


def test_build_paper_memory_card_prefers_structured_summary_artifact_when_source_text_is_latex_heavy(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13021"
    translated_dir = tmp_path / "translated"
    translated_dir.mkdir(parents=True, exist_ok=True)
    translated_path = translated_dir / f"{paper_id}.md"
    translated_path.write_text(
        "\\documentclass{article}\n\\usepackage{graphicx}\n\\section{Intro}\n\\hypersetup{colorlinks=true}\n",
        encoding="utf-8",
    )
    summary_dir = tmp_path / "summaries" / paper_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "oneLine": "This paper compresses paper understanding into grounded memory cards.",
                    "problem": "The paper targets unreliable paper understanding caused by noisy extracted text.",
                    "coreIdea": "It combines parser-backed summaries with memory-card construction.",
                    "methodSteps": ["Use structured summaries as the preferred fallback for weak raw text."],
                    "keyResults": ["The rebuilt card keeps method and evidence slots grounded for residual papers."],
                    "limitations": ["Quality still depends on parser/source quality for badly scanned PDFs."],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Summary Bridged Memory Paper",
            "authors": "E. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "",
            "pdf_path": "",
            "text_path": "",
            "translated_path": str(translated_path),
        }
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert "grounded memory cards" in item["paper_core"]
    assert "noisy extracted text" in item["problem_context"]
    assert "preferred fallback" in item["method_core"]
    assert "residual papers" in item["evidence_core"]
    assert "badly scanned PDFs" in item["limitations"]


def test_build_paper_memory_card_promotes_strong_structured_summary_without_claims(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13029"
    translated_dir = tmp_path / "translated"
    translated_dir.mkdir(parents=True, exist_ok=True)
    translated_path = translated_dir / f"{paper_id}.md"
    translated_path.write_text("placeholder", encoding="utf-8")
    summary_dir = tmp_path / "summaries" / paper_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "parserUsed": "pymupdf",
                "fallbackUsed": False,
                "summary": {
                    "oneLine": "This paper distills multimodal pretraining lessons into a reusable memory card.",
                    "problem": "Operators need compact grounded memory slots instead of noisy page fragments.",
                    "coreIdea": "It bridges parser-backed summaries into retrieval-oriented paper memory slots.",
                    "methodSteps": ["Prefer structured summary slots before raw keyword-window fallbacks."],
                    "keyResults": ["The rebuilt card keeps benchmark evidence and method summaries retrievable."],
                    "limitations": ["Performance still depends on parser quality for badly scanned PDFs."],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Structured Summary Quality Paper",
            "authors": "L. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "",
            "pdf_path": "",
            "text_path": "",
            "translated_path": str(translated_path),
        }
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert item["quality_flag"] == "ok"
    assert "compact grounded memory slots" in item["problem_context"]
    assert "structured summary slots" in item["method_core"]


def test_build_paper_memory_card_keeps_fallback_summary_at_needs_review(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13030"
    text_dir = tmp_path / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    raw_path = text_dir / f"{paper_id}.txt"
    raw_path.write_text("placeholder", encoding="utf-8")
    summary_dir = tmp_path / "summaries" / paper_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "parserUsed": "raw",
                "fallbackUsed": True,
                "documentMemoryDiagnostics": {
                    "structuredSectionsDetected": 1,
                    "parseArtifactPath": "/tmp/parsed/document.md",
                },
                "summary": {
                    "oneLine": "This fallback summary still contains enough structure to be readable.",
                    "problem": "The task is to recover usable slots when only partial parsing succeeded.",
                    "coreIdea": "Use fallback summaries conservatively instead of treating them as fully trusted.",
                    "methodSteps": ["Bridge the best available summary into the memory card."],
                    "keyResults": ["The card remains readable even when the parse path degrades."],
                    "limitations": ["Parser degradation can still require manual review."],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Fallback Summary Quality Paper",
            "authors": "M. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "",
            "pdf_path": "",
            "text_path": str(raw_path),
            "translated_path": "",
        }
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert item["quality_flag"] == "needs_review"


def test_build_compact_extraction_input_uses_structured_summary_artifact(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13022"
    raw_dir = tmp_path / "text"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{paper_id}.txt"
    raw_path.write_text("\\documentclass{article}\n\\usepackage{amsmath}\n", encoding="utf-8")
    summary_dir = tmp_path / "summaries" / paper_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "oneLine": "A parser-aware summary artifact can repair compact extraction inputs.",
                    "problem": "The task is to recover usable paper slots when raw text starts with LaTeX boilerplate.",
                    "coreIdea": "Inject summary artifacts before keyword-window fallback.",
                    "methodSteps": ["Prefer summary-derived method text for schema extraction input."],
                    "keyResults": ["Evidence excerpts stay grounded even when the raw text is unusable."],
                    "limitations": ["Residual low-quality scans may still need manual review."],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Compact Extraction Bridge",
            "authors": "F. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "",
            "pdf_path": "",
            "text_path": str(raw_path),
            "translated_path": "",
        }
    )

    payload = PaperMemoryBuilder(db).build_compact_extraction_input(paper_id=paper_id)

    assert "parser-aware summary artifact" in payload["summaryExcerpt"]
    assert "LaTeX boilerplate" in payload["problemExcerpt"]
    assert "summary-derived method text" in payload["methodExcerpt"]
    assert "raw text is unusable" in payload["findingsExcerpt"]
    assert "manual review" in payload["limitationsExcerpt"]


def test_build_compact_extraction_input_includes_formal_and_final_cause_excerpts(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)

    payload = PaperMemoryBuilder(db).build_compact_extraction_input(paper_id="2603.13017")

    assert "구조화된 card" in payload["formalCauseExcerpt"]
    assert "long-term memory retrieval" in payload["finalCauseExcerpt"]


def test_build_paper_memory_ignores_unusable_structured_summary_fields(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13023"
    translated_dir = tmp_path / "translated"
    translated_dir.mkdir(parents=True, exist_ok=True)
    translated_path = translated_dir / f"{paper_id}.md"
    translated_path.write_text("", encoding="utf-8")
    summary_dir = tmp_path / "summaries" / paper_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "oneLine": "This paper studies summary-bridged memory recovery.",
                    "problem": "현재 주신 정보는 제목·저자뿐이라 원문 내용을 직접 확인할 수 없어 구체적 수치를 제공할 수 없습니다.",
                    "coreIdea": "Use structured summaries only when they are actually usable.",
                    "methodSteps": ["\\documentclass{article} \\usepackage{hyperref}"],
                    "keyResults": ["insufficient information to determine limitations"],
                    "limitations": ["Residual scans may still need parser repair."],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Summary Filtering Paper",
            "authors": "G. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "",
            "pdf_path": "",
            "text_path": "",
            "translated_path": str(translated_path),
        }
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert "summary-bridged memory recovery" in item["paper_core"]
    assert "summary-bridged memory recovery" in item["problem_context"]
    assert "structured summaries" in item["method_core"]
    assert "원문 내용을 직접 확인할 수 없어" not in item["problem_context"]
    assert "\\documentclass" not in item["method_core"]


def test_build_paper_memory_ignores_refusal_like_note_fallback(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13025"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Refusal Filter Paper",
            "authors": "H. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "논문 원문(PDF)이 필요합니다. 현재 주신 정보만으로는 정확한 요약을 제공할 수 없습니다.",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert item["paper_core"] == "Refusal Filter Paper"
    assert "원문(PDF)이 필요합니다" not in item["paper_core"]


def test_build_paper_memory_ignores_latex_raw_text_fallback(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13026"
    raw_path = tmp_path / f"{paper_id}.txt"
    raw_path.write_text("\\documentclass{article}\n\\usepackage{hyperref}\n\\includepdf[pages=1-last]{paper.pdf}\n", encoding="utf-8")
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Latex Filter Paper",
            "authors": "I. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": "",
            "text_path": str(raw_path),
            "translated_path": "",
        }
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert item["paper_core"] == "Latex Filter Paper"
    assert "\\documentclass" not in item["paper_core"]


def test_build_paper_memory_card_prefers_structured_summary_over_polluted_note_sections(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13024"
    summary_dir = tmp_path / "summaries" / paper_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "oneLine": "Structured summary should replace polluted note sections during recovery.",
                    "problem": "The paper studies how to recover cards when notes contain author-block or LaTeX noise.",
                    "coreIdea": "Prefer parser-backed summaries over stale note fragments.",
                    "methodSteps": ["Use parser-backed summary slots as the primary card input during recovery."],
                    "keyResults": ["Recovered cards keep evidence grounded instead of reusing noisy note text."],
                    "limitations": ["Source quality can still cap recovery for badly parsed PDFs."],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Polluted Notes Paper",
            "authors": "H. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "",
            "pdf_path": "",
            "text_path": "",
            "translated_path": str((tmp_path / "translated" / f"{paper_id}.md").resolve()),
        }
    )
    db.upsert_note(
        note_id=f"paper:{paper_id}",
        title="[논문] Polluted Notes Paper",
        content=(
            "# Polluted Notes Paper\n\n"
            "## 요약\n\n"
            "Ashish Vaswani\\\\thanks{noise@example.com} \\author{Noise block}\n\n"
            "## Abstract\n\n"
            "Ashish Vaswani\\\\thanks{noise@example.com} \\author{Noise block}\n\n"
            "## 방법\n\n"
            "\\documentclass{article} \\usepackage{hyperref}\n\n"
            "## 결과\n\n"
            "\\hypersetup{colorlinks=true}\n"
        ),
        source_type="paper",
        metadata={"arxiv_id": paper_id},
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert "Structured summary should replace polluted note sections" in item["paper_core"]
    assert "recover cards when notes contain author-block" in item["problem_context"]
    assert "primary card input during recovery" in item["method_core"]
    assert "Recovered cards keep evidence grounded" in item["evidence_core"]
    assert "Ashish Vaswani" not in item["paper_core"]
    assert "\\documentclass" not in item["method_core"]


def test_structured_summary_lookup_prefers_local_root_over_default_cache(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13025"
    default_root = tmp_path / "default-cache"
    local_root = tmp_path / "runtime"
    translated_dir = local_root / "translated"
    translated_dir.mkdir(parents=True, exist_ok=True)
    translated_path = translated_dir / f"{paper_id}.md"
    translated_path.write_text("", encoding="utf-8")
    (default_root / "papers" / "summaries" / paper_id).mkdir(parents=True, exist_ok=True)
    (default_root / "papers" / "summaries" / paper_id / "summary.json").write_text(
        json.dumps({"summary": {"oneLine": "Stale shared cache summary.", "problem": "stale cache"}}),
        encoding="utf-8",
    )
    (local_root / "summaries" / paper_id).mkdir(parents=True, exist_ok=True)
    (local_root / "summaries" / paper_id / "summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "oneLine": "Fresh local summary should win over the shared cache.",
                    "problem": "The local artifact is newer and should be selected first.",
                    "coreIdea": "Prefer local summary roots before the global cache.",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(memory_builder_module, "DEFAULT_CONFIG_DIR", default_root)
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Local Summary Preference",
            "authors": "I. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "",
            "pdf_path": "",
            "text_path": "",
            "translated_path": str(translated_path),
        }
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert "Fresh local summary should win" in item["paper_core"]
    assert "newer and should be selected first" in item["problem_context"]
    assert "Stale shared cache summary" not in item["paper_core"]


def test_build_paper_memory_ignores_fabricated_note_only_summary(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13026"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Real Source Missing Paper",
            "authors": "J. Researcher",
            "year": 2026,
            "field": "AI Safety",
            "importance": 3,
            "notes": (
                "죄송합니다. 제공된 텍스트는 논문에 대한 세부 정보를 충분히 제공하지 않아서 요청하신 형식으로 심층 요약을 작성하기 어려운 상황입니다. "
                "다만, 일반적인 논문 요약 형식을 기반으로 가상의 요약을 제공해 드리겠습니다. "
                "이 요약은 가정에 기반하며, 실제 논문의 내용을 반영하지 않습니다."
            ),
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert item["paper_core"] == "Real Source Missing Paper"
    assert "가상의 요약" not in item["paper_core"]
    assert "실제 논문의 내용을 반영하지 않습니다" not in item["paper_core"]


def test_build_paper_memory_ignores_untrusted_raw_fallback_summary_artifact(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13027"
    summary_dir = tmp_path / "summaries" / paper_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "parserUsed": "raw",
                "fallbackUsed": True,
                "documentMemoryDiagnostics": {
                    "structuredSectionsDetected": 0,
                    "parseArtifactPath": "",
                },
                "summary": {
                    "oneLine": "Synthetic fallback summary that should not be trusted.",
                    "problem": "This text was generated without a real structured parse.",
                    "coreIdea": "It should be ignored by the memory bridge.",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Parser Trust Gate Paper",
            "authors": "K. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert item["paper_core"] == "Parser Trust Gate Paper"
    assert "Synthetic fallback summary" not in item["paper_core"]


def test_build_paper_memory_uses_parsed_markdown_when_summary_is_untrusted(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13028"
    text_dir = tmp_path / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    raw_path = text_dir / f"{paper_id}.txt"
    raw_path.write_text("\\documentclass{article}\n\\hypersetup{colorlinks=true}\n", encoding="utf-8")
    summary_dir = tmp_path / "summaries" / paper_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "parserUsed": "raw",
                "fallbackUsed": True,
                "documentMemoryDiagnostics": {
                    "structuredSectionsDetected": 0,
                    "parseArtifactPath": "",
                },
                "summary": {
                    "oneLine": "Synthetic fallback summary that should not win.",
                    "problem": "Fallback problem text.",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    parsed_dir = tmp_path / "parsed" / paper_id
    parsed_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.joinpath("document.md").write_text(
        "# Paper Title\n\n"
        "## Abstract\n\n"
        "We study reliable parser-backed paper memory recovery under noisy raw text.\n\n"
        "## Method\n\n"
        "Our approach uses parsed markdown sections as a deterministic fallback.\n\n"
        "## Results\n\n"
        "Experiments show grounded evidence extraction improves downstream card quality by 12%.\n",
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Parsed Markdown Bridge Paper",
            "authors": "L. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": "",
            "text_path": str(raw_path),
            "translated_path": "",
        }
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert "parser-backed paper memory recovery" in item["paper_core"]
    assert "deterministic fallback" in item["method_core"]
    assert "grounded evidence extraction improves downstream card quality" in item["evidence_core"]
    assert "Synthetic fallback summary" not in item["paper_core"]


def test_parsed_markdown_slot_values_anchor_headings_after_front_matter():
    parsed = (
        "Frontier Memory Systems Jane Researcher author@example.com\n\n"
        "Abstract This paper studies robust paper-memory recovery from noisy parser output. "
        "It shows that heading-anchored extraction avoids title and author spillover.\n\n"
        "Introduction Existing summary pipelines can fail when front matter overwhelms the first page. "
        "The main problem is preserving grounded problem context instead of page headers.\n\n"
        "Results Across held-out benchmarks, the repaired extraction keeps evidence grounded and improves card quality by 12%.\n\n"
        "Limitations The approach still depends on readable PDF text and can degrade on severely corrupted scans.\n"
    )

    slots = memory_builder_module._parsed_markdown_slot_values(parsed)

    assert slots["paper_core"].startswith("This paper studies robust paper-memory recovery")
    assert "author@example.com" not in slots["paper_core"]
    assert "front matter overwhelms the first page" in slots["problem_context"]
    assert "improves card quality by 12%" in slots["evidence_core"]
    assert "readable PDF text" in slots["limitations"]


def test_build_paper_memory_promotes_imported_paper_when_parsed_markdown_supplies_four_core_slots(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13029"
    text_dir = tmp_path / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    raw_path = text_dir / f"{paper_id}.txt"
    raw_path.write_text("noisy pdf text placeholder", encoding="utf-8")
    summary_dir = tmp_path / "summaries" / paper_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "parserUsed": "pymupdf",
                "fallbackUsed": True,
                "summary": {
                    "oneLine": "Imported Canon Paper Jane Researcher author@example.com Abstract page header noise.",
                    "problem": "[2603.13029 > Page 2] page header noise only.",
                    "coreIdea": "Imported Canon Paper Jane Researcher author@example.com Abstract page header noise.",
                    "keyResults": ["Table 2: page header noise only."],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    parsed_dir = tmp_path / "parsed" / paper_id
    parsed_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.joinpath("document.md").write_text(
        "Imported Canon Paper Jane Researcher author@example.com\n\n"
        "Abstract This paper studies imported-card recovery under noisy parser output. "
        "It uses heading-aware extraction to keep the core thesis grounded.\n\n"
        "Introduction Existing imported papers can degrade when title blocks and page headers dominate the extracted summary. "
        "The main problem is recovering the actual research context from parser noise.\n\n"
        "Results Across the reported benchmark suite, the repaired extraction restores grounded evidence and improves card quality by 12%.\n",
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Imported Canon Paper",
            "authors": "Jane Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": "",
            "text_path": str(raw_path),
            "translated_path": "",
        }
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert item["quality_flag"] == "ok"
    assert "author@example.com" not in item["paper_core"]
    assert "recovering the actual research context" in item["problem_context"]
    assert "improves card quality by 12%" in item["evidence_core"]


def test_paper_memory_rebuild_is_idempotent_per_paper(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    builder = PaperMemoryBuilder(db)

    first = builder.build_and_store(paper_id="2603.13017")
    second = builder.build_and_store(paper_id="2603.13017")

    cards = db.list_paper_memory_cards(limit=10)
    assert first["paper_id"] == second["paper_id"] == "2603.13017"
    assert len(cards) == 1


def test_search_paper_memory_returns_hydrated_card(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    PaperMemoryBuilder(db).build_and_store(paper_id="2603.13017")

    results = PaperMemoryRetriever(db).search("압축 계층", limit=5)

    assert len(results) == 1
    assert results[0]["paperId"] == "2603.13017"
    assert results[0]["sourceNote"]["id"] == "paper:2603.13017"
    assert results[0]["claims"][0]["claimId"] == "claim_memory_1"


def test_search_paper_memory_prefers_updated_card_for_temporal_query(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    db.upsert_paper_memory_card(
        card={
            "memory_id": "paper-memory:old",
            "paper_id": "2501.00001",
            "title": "Agent Memory Benchmark",
            "paper_core": "Agent memory benchmark baseline",
            "problem_context": "long-term retrieval",
            "method_core": "memory card ranking",
            "evidence_core": "benchmark report",
            "limitations": "",
            "concept_links": ["agent memory"],
            "claim_refs": [],
            "published_at": "2025-01-01T00:00:00+00:00",
            "evidence_window": "2025-01-01T00:00:00+00:00",
            "search_text": "Agent Memory Benchmark long-term retrieval benchmark",
            "quality_flag": "ok",
        }
    )
    db.upsert_paper_memory_card(
        card={
            "memory_id": "paper-memory:new",
            "paper_id": "2601.00001",
            "title": "Agent Memory Benchmark v2",
            "paper_core": "Updated agent memory benchmark",
            "problem_context": "long-term retrieval",
            "method_core": "memory card ranking",
            "evidence_core": "updated benchmark report",
            "limitations": "",
            "concept_links": ["agent memory"],
            "claim_refs": [],
            "published_at": "2026-01-01T00:00:00+00:00",
            "evidence_window": "2026-01-01T00:00:00+00:00",
            "search_text": "Agent Memory Benchmark v2 updated long-term retrieval benchmark",
            "quality_flag": "ok",
        }
    )
    db.upsert_memory_relation(
        relation_id="rel:paper-updates",
        src_form="paper_memory",
        src_id="paper-memory:old",
        dst_form="paper_memory",
        dst_id="paper-memory:new",
        relation_type="updates",
        confidence=0.91,
        provenance={"rule": "test"},
    )

    results = PaperMemoryRetriever(db).search("latest agent memory benchmark", limit=2, include_refs=False)

    assert [item["paperId"] for item in results[:2]] == ["2601.00001", "2501.00001"]
    assert results[0]["retrievalSignals"]["updatesPreferred"] is True
    assert results[0]["retrievalSignals"]["timeRelevance"] > 0
    assert results[0]["retrievalSignals"]["fallbackLexicalMatch"] > 0


def test_build_paper_memory_dedupes_duplicate_claims_from_note_and_entity(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    db.upsert_claim(
        claim_id="claim_memory_1",
        claim_text="The approach achieves around 11x compression over raw exchanges.",
        subject_entity_id="paper:2603.13017",
        predicate="improves",
        object_literal="compression efficiency",
        confidence=0.97,
        evidence_ptrs=[{"note_id": "paper:2603.13017"}],
        source="test",
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id="2603.13017")

    assert item["claim_refs"] == ["claim_memory_1"]


def test_build_paper_memory_tolerates_malformed_note_metadata(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    db.conn.execute("UPDATE notes SET metadata = ? WHERE id = ?", ("[]", "paper:2603.13017"))
    db.conn.commit()

    item = PaperMemoryBuilder(db).build_and_store(paper_id="2603.13017")

    assert item["paper_id"] == "2603.13017"
    assert item["quality_flag"] == "needs_review"


def test_build_paper_memory_handles_unreadable_and_missing_paths(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13020"
    unreadable_path = tmp_path / "directory_instead_of_file"
    unreadable_path.mkdir()
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Unreadable Fallback Paper",
            "authors": "D. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 2,
            "notes": "falls back to metadata when paths fail",
            "pdf_path": "",
            "text_path": str(tmp_path / "missing.txt"),
            "translated_path": str(unreadable_path),
        }
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert item["paper_id"] == paper_id
    assert "falls back to metadata" in item["paper_core"] or "Unreadable Fallback Paper" in item["paper_core"]


def test_build_paper_memory_without_matching_sections_degrades_to_claims(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13021"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Flat Note Paper",
            "authors": "E. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "flat note fallback",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_note(
        note_id=f"paper:{paper_id}",
        title="[논문] Flat Note Paper",
        content="문단만 있고 구조화된 ## 섹션은 없다.",
        source_type="paper",
        para_category="resource",
        metadata={"arxiv_id": paper_id},
    )
    db.upsert_claim(
        claim_id="claim_flat_1",
        claim_text="Flat-note papers still expose claim-level retrieval evidence.",
        subject_entity_id=f"paper:{paper_id}",
        predicate="supports",
        object_literal="claim fallback",
        confidence=0.74,
        evidence_ptrs=[{"note_id": f"paper:{paper_id}"}],
        source="test",
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert item["paper_id"] == paper_id
    assert "Flat-note papers" in item["paper_core"] or "Flat-note papers" in item["evidence_core"]


def test_quality_flag_does_not_overpromote_single_strong_claim(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13022"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Weak Note Strong Claim",
            "authors": "F. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "weak note",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_note(
        note_id=f"paper:{paper_id}",
        title="[논문] Weak Note Strong Claim",
        content="# Weak Note\n\n짧은 설명만 있다.",
        source_type="paper",
        para_category="resource",
        metadata={},
    )
    db.upsert_claim(
        claim_id="claim_strong_only",
        claim_text="A single strong claim should not auto-promote the whole card.",
        subject_entity_id=f"paper:{paper_id}",
        predicate="suggests",
        object_literal="conservative quality inheritance",
        confidence=0.92,
        evidence_ptrs=[{"note_id": f"paper:{paper_id}"}],
        source="test",
    )

    item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)

    assert item["quality_flag"] == "needs_review"


def test_quality_flag_promotes_imported_paper_without_note_when_two_strong_claims_exist(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13023"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Imported Strong Claim Paper",
            "authors": "G. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "imported paper without a local paper note",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_claim(
        claim_id="claim_imported_1",
        claim_text="The imported paper improves retrieval latency by 30 percent.",
        subject_entity_id=f"paper:{paper_id}",
        predicate="improves",
        object_literal="retrieval latency",
        confidence=0.91,
        evidence_ptrs=[],
        source="test",
    )
    db.upsert_claim(
        claim_id="claim_imported_2",
        claim_text="The imported paper reduces memory overhead by 20 percent.",
        subject_entity_id=f"paper:{paper_id}",
        predicate="reduces",
        object_literal="memory overhead",
        confidence=0.88,
        evidence_ptrs=[],
        source="test",
    )

    memory_item = PaperMemoryBuilder(db).build_and_store(paper_id=paper_id)
    card_item = PaperCardV2Builder(db).build_and_store(paper_id=paper_id)

    assert memory_item["quality_flag"] == "ok"
    assert card_item["quality_flag"] == "ok"
    assert card_item["diagnostics"]["acceptedClaimCount"] == 2


def test_paper_memory_eval_harness_reports_surface_quality(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path, paper_id="2603.13017")

    translated_path = tmp_path / "2603.13019_translated.md"
    translated_path.write_text(
        "This paper proposes a compact memory card for paper retrieval. It improves search efficiency.",
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": "2603.13019",
            "title": "Metadata Only Paper",
            "authors": "C. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "metadata fallback note",
            "pdf_path": "",
            "text_path": "",
            "translated_path": str(translated_path),
        }
    )
    PaperMemoryBuilder(db).build_and_store(paper_id="2603.13017")
    PaperMemoryBuilder(db).build_and_store(paper_id="2603.13019")

    harness = PaperMemoryEvalHarness(db)
    report = harness.evaluate_cases(
        [
            PaperMemoryEvalCase(
                case_id="exact_title",
                query="Personalized Agent Memory",
                expected_paper_id="2603.13017",
                category="exact_metadata_lookup",
                artifact_profile="note_rich_claim_rich",
            ),
            PaperMemoryEvalCase(
                case_id="thematic_korean",
                query="검색 가능한 압축 계층",
                expected_paper_id="2603.13017",
                category="thematic_recall",
                artifact_profile="note_rich_claim_rich",
            ),
            PaperMemoryEvalCase(
                case_id="metadata_fallback",
                query="compact memory card for paper retrieval",
                expected_paper_id="2603.13019",
                category="note_poor_metadata_fallback",
                artifact_profile="metadata_only",
            ),
        ]
    )

    assert report["summary"]["caseCount"] == 3
    thematic_case = next(case for case in report["cases"] if case["caseId"] == "thematic_korean")
    assert thematic_case["surfaces"]["paper_memory_search"]["top1Match"] is True
    assert thematic_case["surfaces"]["search_papers"]["noResult"] is True
    fallback_case = next(case for case in report["cases"] if case["caseId"] == "metadata_fallback")
    assert fallback_case["surfaces"]["paper_memory_search"]["top1Match"] is True


class _FakePaperSchemaExtractor:
    def __init__(self, payload):
        self.payload = payload

    def extract(self, *, paper):  # noqa: ANN001
        _ = paper
        return self.payload


class _RecordingPaperSchemaExtractor:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def extract(self, *, paper):  # noqa: ANN001
        self.calls.append(paper)
        return self.payload


def test_paper_memory_schema_mode_maps_internal_payload_without_changing_card_shape(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    builder = PaperMemoryBuilder(
        db,
        extraction_mode="schema",
        schema_extractor=_FakePaperSchemaExtractor(
            {
                "paperCore": "Schema-backed paper core.",
                "problemContext": "Schema-backed problem context.",
                "methodCore": "Schema-backed method core.",
                "evidenceCore": "Schema-backed evidence core.",
                "limitations": "Schema-backed limitation.",
                "conceptLinks": ["structured memory"],
                "claimRefs": ["claim_schema_extra"],
                "qualityFlag": "ok",
                "coverageStatusByField": {"paperCore": "complete"},
            }
        ),
    )

    item = builder.build_and_store(paper_id="2603.13017")

    assert item["paper_core"] == "개인화된 에이전트 메모리를 검색 가능한 압축 계층으로 정리한다."
    assert item["problem_context"] == "The paper studies long-term memory retrieval for developer-agent interactions."
    assert item["method_core"] == "exchange를 구조화된 card로 증류한다."
    assert item["evidence_core"] == "11배 압축을 달성한다."
    assert "structured memory" in item["concept_links"]
    assert "claim_schema_extra" in item["claim_refs"]
    diagnostics = builder.get_last_extraction_diagnostics("2603.13017")
    assert diagnostics["applied"] is True
    assert diagnostics["fallbackUsed"] is False


def test_paper_memory_schema_mode_uses_sanitized_text_for_extraction_input(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13023"
    raw_path = tmp_path / "2603.13023.txt"
    raw_path.write_text(
        "\\documentclass{article}\n"
        "\\usepackage{amsmath}\n"
        "\\title{Sanitized Input Paper}\n"
        "\\begin{document}\n"
        "\\maketitle\n\n"
        "Abstract\n"
        "This paper studies agent memory retrieval with enough prose to survive sanitation. "
        "It introduces compressed evidence cards for longitudinal queries and benchmark evaluation.\n",
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Sanitized Input Paper",
            "authors": "G. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "sanitization coverage",
            "pdf_path": "",
            "text_path": str(raw_path),
            "translated_path": "",
        }
    )
    extractor = _RecordingPaperSchemaExtractor({"paperCore": "Structured paper core."})
    builder = PaperMemoryBuilder(
        db,
        extraction_mode="shadow",
        schema_extractor=extractor,
    )

    item = builder.build_and_store(paper_id=paper_id)

    assert item["paper_core"]
    assert extractor.calls
    payload = extractor.calls[0]
    assert "\\documentclass" not in payload["summaryExcerpt"]
    assert payload["summaryExcerpt"].startswith("Abstract")
    assert "agent memory retrieval" in payload["problemExcerpt"]
    diagnostics = builder.get_last_extraction_diagnostics(paper_id)
    assert diagnostics["textSanitation"]["raw"]["startsWithLatex"] is True
    assert diagnostics["textSanitation"]["preferredSource"] == "raw"


def test_paper_memory_schema_mode_prefers_numeric_findings_excerpt_for_evidence_input(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "2603.13024"
    raw_path = tmp_path / "2603.13024.txt"
    raw_path.write_text(
        "Abstract\n"
        "This paper studies agent memory retrieval.\n\n"
        "Results\n"
        "The system improves hit rate by 12% on MemoryArena and achieves 0.81 AUPRC on longitudinal retrieval.\n"
        "It also outperforms baseline routing on benchmark tasks.\n",
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Evidence Heavy Paper",
            "authors": "H. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "evidence coverage",
            "pdf_path": "",
            "text_path": str(raw_path),
            "translated_path": "",
        }
    )
    extractor = _RecordingPaperSchemaExtractor({"paperCore": "Structured paper core."})
    builder = PaperMemoryBuilder(
        db,
        extraction_mode="shadow",
        schema_extractor=extractor,
    )

    builder.build_and_store(paper_id=paper_id)

    payload = extractor.calls[0]
    assert "12%" in payload["findingsExcerpt"]
    assert "0.81 AUPRC" in payload["findingsExcerpt"]
    assert "MemoryArena" in payload["findingsExcerpt"]


def test_paper_memory_shadow_mode_keeps_deterministic_card_and_records_diagnostics(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    builder = PaperMemoryBuilder(
        db,
        extraction_mode="shadow",
        schema_extractor=_FakePaperSchemaExtractor({"paperCore": "Shadow paper core."}),
    )

    item = builder.build_and_store(paper_id="2603.13017")

    assert item["paper_core"] != "Shadow paper core."
    diagnostics = builder.get_last_extraction_diagnostics("2603.13017")
    assert diagnostics["attempted"] is True
    assert diagnostics["applied"] is False
    assert diagnostics["fallbackUsed"] is False


def test_paper_memory_schema_mode_accepts_partial_payload_and_keeps_missing_fields_deterministic(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    builder = PaperMemoryBuilder(
        db,
        extraction_mode="schema",
        schema_extractor=_FakePaperSchemaExtractor(
            {
                "thesis": "Short structured thesis.",
                "conceptLinks": ["paper memory"],
                "formalCause": {"summary": "", "basis": "missing"},
                "finalCause": {"authorStatedSummary": "", "inferredSummary": "", "basis": "missing"},
                "coverageStatusByField": {"thesis": "complete"},
                "fieldConfidence": {"thesis": 0.93},
            }
        ),
    )

    item = builder.build_and_store(paper_id="2603.13017")

    assert item["paper_core"] == "개인화된 에이전트 메모리를 검색 가능한 압축 계층으로 정리한다."
    assert item["problem_context"]
    assert item["method_core"]
    assert "paper memory" in item["concept_links"]
    stored = db.get_paper_memory_card("2603.13017")
    assert stored["formal_cause"] == {}
    assert stored["final_cause"] == {}
    diagnostics = builder.get_last_extraction_diagnostics("2603.13017")
    assert diagnostics["applied"] is True
    assert diagnostics["coverageByField"]["thesis"] == "complete"
    assert diagnostics["fieldConfidence"]["thesis"] == 0.93


def test_paper_memory_schema_mode_stores_cause_shadow_without_public_shape_change(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    builder = PaperMemoryBuilder(
        db,
        extraction_mode="schema",
        schema_extractor=_FakePaperSchemaExtractor(
            {
                "formalCause": {
                    "summary": "A typed episodic-card schema for replayable memory.",
                    "basis": "author_stated",
                    "confidence": 0.91,
                    "coverage": "complete",
                    "evidenceRefs": ["method.formal"],
                },
                "finalCause": {
                    "authorStatedSummary": "Enable durable long-horizon developer-agent memory retrieval.",
                    "inferredSummary": "",
                    "basis": "author_stated",
                    "confidence": 0.87,
                    "coverage": "complete",
                    "evidenceRefs": ["abstract.goal"],
                },
                "coverageStatusByField": {
                    "formalCause": "complete",
                    "finalCause": "complete",
                },
                "fieldConfidence": {
                    "formalCause": 0.91,
                    "finalCause": 0.87,
                },
            }
        ),
    )

    item = builder.build_and_store(paper_id="2603.13017")
    stored = db.get_paper_memory_card("2603.13017")
    public_payload = PaperMemoryRetriever(db).get("2603.13017", include_refs=False)

    assert stored["formal_cause"]["basis"] == "author_stated"
    assert stored["formal_cause"]["summary"] == "A typed episodic-card schema for replayable memory."
    assert stored["final_cause"]["author_stated_summary"] == "Enable durable long-horizon developer-agent memory retrieval."
    assert stored["final_cause"]["inferred_summary"] == ""
    assert "A typed episodic-card schema for replayable memory." not in item["search_text"]
    assert "Enable durable long-horizon developer-agent memory retrieval." not in item["search_text"]
    assert "formalCause" not in public_payload
    assert "finalCause" not in public_payload
    diagnostics = builder.get_last_extraction_diagnostics("2603.13017")
    assert diagnostics["coverageByField"]["formalCause"] == "complete"
    assert diagnostics["coverageByField"]["finalCause"] == "complete"


def test_paper_memory_schema_mode_keeps_author_stated_final_cause_empty_when_only_inferred(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    builder = PaperMemoryBuilder(
        db,
        extraction_mode="schema",
        schema_extractor=_FakePaperSchemaExtractor(
            {
                "finalCause": {
                    "authorStatedSummary": "",
                    "inferredSummary": "Reduce retrieval drift across long-running agent sessions.",
                    "basis": "inferred",
                    "confidence": 0.74,
                    "coverage": "partial",
                    "evidenceRefs": ["abstract.inference"],
                }
            }
        ),
    )

    builder.build_and_store(paper_id="2603.13017")
    stored = db.get_paper_memory_card("2603.13017")

    assert stored["final_cause"]["author_stated_summary"] == ""
    assert stored["final_cause"]["inferred_summary"] == "Reduce retrieval drift across long-running agent sessions."
    assert stored["final_cause"]["basis"] == "inferred"


def test_paper_memory_schema_mode_uses_structured_method_and_evidence_when_present(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    builder = PaperMemoryBuilder(
        db,
        extraction_mode="schema",
        schema_extractor=_FakePaperSchemaExtractor(
            {
                "thesis": "Structured thesis.",
                "methodCore": "Structured method summary.",
                "evidenceCore": "Structured evidence summary.",
                "coverageStatusByField": {
                    "thesis": "complete",
                    "methodCore": "complete",
                    "evidenceCore": "complete",
                },
                "fieldConfidence": {
                    "thesis": 0.94,
                    "methodCore": 0.89,
                    "evidenceCore": 0.91,
                },
            }
        ),
    )

    item = builder.build_and_store(paper_id="2603.13017")

    assert item["paper_core"] == "개인화된 에이전트 메모리를 검색 가능한 압축 계층으로 정리한다."
    assert item["method_core"] == "exchange를 구조화된 card로 증류한다."
    assert item["evidence_core"] == "11배 압축을 달성한다."
    diagnostics = builder.get_last_extraction_diagnostics("2603.13017")
    assert diagnostics["applied"] is True
    assert diagnostics["coverageByField"]["methodCore"] == "complete"
    assert diagnostics["fieldConfidence"]["methodCore"] == 0.89
    assert diagnostics["fieldConfidence"]["evidenceCore"] == 0.91


def test_paper_memory_schema_mode_replaces_polluted_problem_context_when_structured_value_present(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    db.upsert_note(
        note_id="paper:2603.13017",
        title="[논문] Personalized Agent Memory",
        content=(
            "# Personalized Agent Memory\n\n"
            "## Abstract\n\n"
            "Ashish Vaswani\\\\thanks{contact@example.com} \\author{Noise}\n\n"
            "## 방법\n\n"
            "- exchange를 구조화된 card로 증류한다.\n\n"
            "## 결과\n\n"
            "- 11배 압축을 달성한다.\n\n"
            "## 한계\n\n"
            "- 단일 사용자 환경 중심 평가다.\n"
        ),
        source_type="paper",
        para_category="resource",
        metadata={"arxiv_id": "2603.13017", "quality_flag": "ok"},
    )
    builder = PaperMemoryBuilder(
        db,
        extraction_mode="schema",
        schema_extractor=_FakePaperSchemaExtractor(
            {
                "thesis": "Structured thesis.",
                "problemContext": "Long-term memory retrieval for developer-agent interactions.",
            }
        ),
    )

    item = builder.build_and_store(paper_id="2603.13017")

    assert item["problem_context"] == "Long-term memory retrieval for developer-agent interactions."
    assert "Ashish Vaswani" not in item["search_text"]
    assert "Long-term memory retrieval" in item["search_text"]


def test_paper_memory_schema_mode_makes_limitations_conservative_without_explicit_support(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    db.upsert_note(
        note_id="paper:2603.13017",
        title="[논문] Personalized Agent Memory",
        content=(
            "# Personalized Agent Memory\n\n"
            "## 요약\n\n"
            "개인화된 에이전트 메모리를 검색 가능한 압축 계층으로 정리한다.\n\n"
            "## 방법\n\n"
            "- exchange를 구조화된 card로 증류한다.\n\n"
            "## 결과\n\n"
            "- 11배 압축을 달성한다.\n"
        ),
        source_type="paper",
        para_category="resource",
        metadata={"arxiv_id": "2603.13017", "quality_flag": "ok"},
    )
    builder = PaperMemoryBuilder(
        db,
        extraction_mode="schema",
        schema_extractor=_FakePaperSchemaExtractor(
            {
                "thesis": "Structured thesis.",
                "limitations": "Applies only to a single-user setting.",
            }
        ),
    )

    item = builder.build_and_store(paper_id="2603.13017")

    assert item["limitations"] == "limitations not explicit in visible excerpt"


def test_paper_memory_schema_mode_keeps_explicit_limitations_when_supported(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    builder = PaperMemoryBuilder(
        db,
        extraction_mode="schema",
        schema_extractor=_FakePaperSchemaExtractor(
            {
                "thesis": "Structured thesis.",
                "limitations": "Limited to a single-user evaluation setting.",
            }
        ),
    )

    item = builder.build_and_store(paper_id="2603.13017")

    assert item["limitations"] == "단일 사용자 환경 중심 평가다."


def test_paper_memory_schema_mode_falls_back_when_payload_is_invalid(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)
    builder = PaperMemoryBuilder(
        db,
        extraction_mode="schema",
        schema_extractor=_FakePaperSchemaExtractor({"warnings": ["empty"]}),
    )

    item = builder.build_and_store(paper_id="2603.13017")

    assert item["paper_core"]
    diagnostics = builder.get_last_extraction_diagnostics("2603.13017")
    assert diagnostics["fallbackUsed"] is True
    assert "invalid_or_empty_payload" in diagnostics["warnings"]


def test_paper_memory_schema_mode_records_parse_preview_on_extractor_error(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper_with_note(db, tmp_path)

    class _ExplodingSchemaExtractor:
        def extract_with_metadata(self, *, paper):  # noqa: ANN001
            raise PaperMemoryExtractionError(
                "bad payload",
                raw_preview='{"thesis":',
                raw_payload_bytes=9,
                parse_stage="json_parse",
            )

    builder = PaperMemoryBuilder(
        db,
        extraction_mode="shadow",
        schema_extractor=_ExplodingSchemaExtractor(),
    )

    builder.build_and_store(paper_id="2603.13017")

    diagnostics = builder.get_last_extraction_diagnostics("2603.13017")
    assert diagnostics["fallbackUsed"] is True
    assert diagnostics["parseStage"] == "json_parse"
    assert diagnostics["rawOutputPreview"] == '{"thesis":'
    assert diagnostics["rawPayloadBytes"] == 9


def test_paper_memory_schema_extractor_parses_json_payload():
    class _FakeLLM:
        def generate(self, prompt: str, context: str = "", max_tokens: int | None = None):  # noqa: ARG002
            return '{"paperCore":"Structured paper core","methodCore":"Method summary"}'

    extractor = PaperMemorySchemaExtractor(_FakeLLM(), model="exaone3.5:7.8b")
    payload = extractor.extract(paper={"paperId": "2603.13017"})

    assert payload["paperCore"] == "Structured paper core"


def test_paper_memory_schema_extractor_parses_think_wrapped_json_payload():
    class _FakeLLM:
        def generate(self, prompt: str, context: str = "", max_tokens: int | None = None):  # noqa: ARG002
            return '<think>reasoning</think>```json\n{"paperCore":"Structured paper core","methodCore":"Method summary"}\n```'

    extractor = PaperMemorySchemaExtractor(_FakeLLM(), model="exaone3.5:7.8b")
    payload = extractor.extract(paper={"paperId": "2603.13017"})

    assert payload["paperCore"] == "Structured paper core"


def test_paper_memory_schema_extractor_with_metadata_reports_payload_size():
    class _FakeLLM:
        def generate(self, prompt: str, context: str = "", max_tokens: int | None = None):  # noqa: ARG002
            return '{"thesis":"Structured thesis","claims":["c1"]}'

    extractor = PaperMemorySchemaExtractor(_FakeLLM(), model="exaone3.5:7.8b")
    payload, metadata = extractor.extract_with_metadata(paper={"paperId": "2603.13017"})

    assert payload["thesis"] == "Structured thesis"
    assert metadata["rawPayloadBytes"] > 0
    assert "thesis" in metadata["parsedFields"]


def test_paper_memory_schema_extraction_v1_accepts_method_and_evidence_only_payload():
    from knowledge_hub.papers.memory_extraction import PaperMemoryExtractionV1

    extraction = PaperMemoryExtractionV1.from_dict(
        {
            "methodCore": "Retriever over compressed cards.",
            "evidenceCore": "Improves hit rate on longitudinal tasks.",
        }
    )

    assert extraction is not None
    assert extraction.method_core == "Retriever over compressed cards."
    assert extraction.evidence_core == "Improves hit rate on longitudinal tasks."


def test_paper_memory_schema_extractor_reports_raw_preview_on_parse_failure():
    class _FakeLLM:
        def generate(self, prompt: str, context: str = "", max_tokens: int | None = None):  # noqa: ARG002
            return "Not JSON at all. thesis: maybe here."

    extractor = PaperMemorySchemaExtractor(_FakeLLM(), model="exaone3.5:7.8b")
    try:
        extractor.extract(paper={"paperId": "2603.13017"})
    except PaperMemoryExtractionError as exc:
        assert exc.parse_stage == "json_parse"
        assert "Not JSON at all" in exc.raw_preview
        assert exc.raw_payload_bytes > 0
    else:
        raise AssertionError("expected PaperMemoryExtractionError")
