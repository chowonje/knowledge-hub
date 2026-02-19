"""Knowledge Graph 핵심 테스트 — concepts, aliases, kg_relations, 쿼리"""

from __future__ import annotations

import pytest

from knowledge_hub.core.database import SQLiteDatabase


@pytest.fixture()
def db(tmp_path):
    return SQLiteDatabase(str(tmp_path / "test.db"))


class TestConceptsCRUD:
    def test_upsert_and_get(self, db):
        db.upsert_concept("transformer", "Transformer", "A sequence model based on attention.")
        c = db.get_concept("transformer")
        assert c is not None
        assert c["canonical_name"] == "Transformer"
        assert "attention" in c["description"]

    def test_get_by_name(self, db):
        db.upsert_concept("attention_mechanism", "Attention Mechanism")
        c = db.get_concept_by_name("Attention Mechanism")
        assert c is not None
        assert c["id"] == "attention_mechanism"

    def test_list(self, db):
        db.upsert_concept("a", "Alpha")
        db.upsert_concept("b", "Beta")
        db.upsert_concept("c", "Gamma")
        concepts = db.list_concepts()
        assert len(concepts) == 3

    def test_upsert_updates_description(self, db):
        db.upsert_concept("x", "X", "old desc")
        db.upsert_concept("x", "X", "new desc")
        assert db.get_concept("x")["description"] == "new desc"

    def test_delete_cascades(self, db):
        db.upsert_concept("t", "T")
        db.add_alias("t_alias", "t")
        db.add_relation("concept", "t", "concept_related_to", "concept", "other")
        db.delete_concept("t")
        assert db.get_concept("t") is None
        assert db.get_aliases("t") == []
        assert db.get_relations("concept", "t") == []


class TestAliases:
    def test_add_and_resolve(self, db):
        db.upsert_concept("neural_network", "Neural Network")
        db.add_alias("Neural Networks", "neural_network")
        db.add_alias("NN", "neural_network")

        assert db.resolve_concept("Neural Network") == "Neural Network"
        assert db.resolve_concept("Neural Networks") == "Neural Network"
        assert db.resolve_concept("NN") == "Neural Network"

    def test_resolve_unknown_returns_none(self, db):
        assert db.resolve_concept("NonExistent") is None

    def test_get_aliases(self, db):
        db.upsert_concept("llm", "Large Language Model")
        db.add_alias("LLM", "llm")
        db.add_alias("Large Language Models", "llm")
        aliases = db.get_aliases("llm")
        assert set(aliases) == {"LLM", "Large Language Models"}


class TestKGRelations:
    def test_add_and_get(self, db):
        db.upsert_concept("transformer", "Transformer")
        db.upsert_paper({
            "arxiv_id": "1706.03762", "title": "Attention Is All You Need",
            "authors": "Vaswani et al.", "year": 2017, "field": "CS",
            "importance": 5, "notes": "", "pdf_path": None,
            "text_path": None, "translated_path": None,
        })
        db.add_relation(
            "paper", "1706.03762",
            "paper_uses_concept",
            "concept", "transformer",
            evidence_text="We propose the Transformer, based entirely on attention.",
            confidence=0.95,
        )

        rels = db.get_relations("paper", "1706.03762")
        assert len(rels) == 1
        assert rels[0]["relation"] == "paper_uses_concept"
        assert rels[0]["evidence_text"].startswith("We propose")

    def test_upsert_updates_evidence(self, db):
        db.add_relation("concept", "a", "concept_related_to", "concept", "b",
                         evidence_text="old", confidence=0.3)
        db.add_relation("concept", "a", "concept_related_to", "concept", "b",
                         evidence_text="new", confidence=0.9)
        rels = db.get_relations("concept", "a")
        assert len(rels) == 1
        assert rels[0]["evidence_text"] == "new"
        assert rels[0]["confidence"] == 0.9

    def test_concept_papers_query(self, db):
        db.upsert_concept("gan", "GAN")
        for i in range(3):
            aid = f"2501.0000{i}"
            db.upsert_paper({
                "arxiv_id": aid, "title": f"GAN Paper {i}",
                "authors": "", "year": 2025, "field": "CS",
                "importance": 3, "notes": "", "pdf_path": None,
                "text_path": None, "translated_path": None,
            })
            db.add_relation("paper", aid, "paper_uses_concept", "concept", "gan",
                             evidence_text=f"paper {i} uses GAN", confidence=0.8)

        papers = db.get_concept_papers("gan")
        assert len(papers) == 3

    def test_paper_concepts_query(self, db):
        aid = "2501.99999"
        db.upsert_paper({
            "arxiv_id": aid, "title": "Multi Concept Paper",
            "authors": "", "year": 2025, "field": "CS",
            "importance": 3, "notes": "", "pdf_path": None,
            "text_path": None, "translated_path": None,
        })
        for name in ["Transformer", "Attention", "BERT"]:
            cid = name.lower()
            db.upsert_concept(cid, name)
            db.add_relation("paper", aid, "paper_uses_concept", "concept", cid)

        concepts = db.get_paper_concepts(aid)
        assert len(concepts) == 3
        names = {c["canonical_name"] for c in concepts}
        assert "Transformer" in names

    def test_related_concepts_query(self, db):
        db.upsert_concept("transformer", "Transformer")
        db.upsert_concept("attention", "Attention Mechanism")
        db.upsert_concept("bert", "BERT")
        db.add_relation("concept", "transformer", "concept_related_to",
                         "concept", "attention", confidence=0.8)
        db.add_relation("concept", "transformer", "concept_related_to",
                         "concept", "bert", confidence=0.7)

        related = db.get_related_concepts("transformer")
        assert len(related) == 2


class TestKGStats:
    def test_stats_counts(self, db):
        db.upsert_concept("a", "A")
        db.upsert_concept("b", "B")
        db.add_alias("b_alias", "b")
        db.add_relation("concept", "a", "concept_related_to", "concept", "b")

        stats = db.get_kg_stats()
        assert stats["concepts"] == 2
        assert stats["aliases"] == 1
        assert stats["relations"] == 1
        assert stats["isolated_concepts"] == 0

    def test_isolated_concepts(self, db):
        db.upsert_concept("lonely", "Lonely Concept")
        db.upsert_concept("connected", "Connected")
        db.add_relation("concept", "connected", "concept_related_to",
                         "concept", "other")

        stats = db.get_kg_stats()
        assert stats["isolated_concepts"] == 1


class TestConceptIdHelper:
    def test_concept_id(self):
        from knowledge_hub.cli.paper_cmd import _concept_id
        assert _concept_id("Neural Network") == "neural_network"
        assert _concept_id("  BERT  ") == "bert"
        assert _concept_id("Attention Mechanism") == "attention_mechanism"
