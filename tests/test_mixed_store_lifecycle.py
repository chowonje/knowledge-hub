from __future__ import annotations

import json

from knowledge_hub.core.database import SQLiteDatabase
from knowledge_hub.infrastructure.persistence.stores.derivative_lifecycle import (
    mark_derivatives_stale_for_document,
)


def _bootstrap_note_backed_mixed_rows(db: SQLiteDatabase) -> None:
    db.upsert_note(
        "note-1",
        "Mixed Store Note",
        "note body",
        file_path="vault/mixed-store-note.md",
        metadata={"source_content_hash": "hash-note-old", "canonical_url": "https://example.com/mixed"},
    )
    db.upsert_ontology_entity(
        entity_id="concept-a",
        entity_type="concept",
        canonical_name="Concept A",
        source="test",
    )
    db.upsert_claim(
        claim_id="claim-1",
        claim_text="Concept A is mentioned in note-1.",
        subject_entity_id="concept-a",
        predicate="mentions",
        confidence=0.7,
        evidence_ptrs=[{"note_id": "note-1", "path": "vault/mixed-store-note.md"}],
        source="test",
    )
    db.add_relation(
        "note",
        "note-1",
        "note_mentions_concept",
        "concept",
        "concept-a",
        evidence_text=json.dumps(
            {
                "note_id": "note-1",
                "evidence_ptrs": [{"note_id": "note-1", "path": "vault/mixed-store-note.md"}],
            },
            ensure_ascii=False,
        ),
        confidence=0.8,
    )
    db.upsert_memory_relation(
        relation_id="rel-derived-note",
        src_form="document_memory",
        src_id="note-1",
        dst_form="document_memory",
        dst_id="note-1-newer",
        relation_type="updates",
        confidence=0.75,
        provenance={"note_id": "note-1", "path": "vault/mixed-store-note.md"},
    )
    db.upsert_learning_graph_node(
        node_id="lg-node-1",
        entity_id="concept-a",
        node_type="concept",
        canonical_name="Concept A",
        difficulty_level="intermediate",
        difficulty_score=0.5,
        stage="intermediate",
        confidence=0.8,
        provenance={"noteId": "note-1", "path": "vault/mixed-store-note.md"},
    )
    db.upsert_learning_graph_edge(
        edge_id="lg-edge-1",
        source_node_id="lg-node-1",
        edge_type="prerequisite",
        target_node_id="lg-node-2",
        confidence=0.7,
        status="approved",
        provenance={"noteId": "note-1"},
        evidence={"noteId": "note-1"},
    )
    db.upsert_learning_graph_resource_link(
        link_id="lg-link-1",
        concept_node_id="lg-node-1",
        resource_node_id="paper-node-1",
        link_type="introduced_by",
        reading_stage="beginner",
        confidence=0.7,
        status="approved",
        provenance={"noteId": "note-1"},
    )
    db.upsert_learning_graph_path(
        path_id="lg-path-1",
        topic_slug="rag",
        status="approved",
        version=1,
        path_payload={
            "pathId": "lg-path-1",
            "topicSlug": "rag",
            "nodes": ["lg-node-1"],
            "stages": {},
            "status": "approved",
            "noteId": "note-1",
        },
        score_payload={},
        provenance={"noteId": "note-1"},
    )


def test_note_source_change_marks_mixed_derivatives_stale_and_default_reads_exclude_them(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "mixed-lifecycle-note.db"))
    _bootstrap_note_backed_mixed_rows(db)

    assert db.list_ontology_claims(limit=10)
    assert db.list_relations(limit=10, source_type="note")
    assert db.list_memory_relations(src_form="document_memory", src_id="note-1", relation_type="updates", limit=10)
    assert db.list_learning_graph_nodes(limit=10)
    assert db.list_learning_graph_edges(status="approved", limit=10)
    assert db.list_learning_graph_resource_links(status="approved", limit=10)
    assert db.get_latest_learning_graph_path(topic_slug="rag", status="approved") is not None

    db.merge_note_metadata("note-1", {"source_content_hash": "hash-note-new"})

    claim_row = db.list_ontology_claims(limit=10, include_stale=True)[0]
    relation_row = db.list_relations(limit=10, source_type="note", include_stale=True)[0]
    memory_row = db.list_memory_relations(
        src_form="document_memory",
        src_id="note-1",
        relation_type="updates",
        limit=10,
        include_stale=True,
    )[0]
    node_row = db.list_learning_graph_nodes(limit=10, include_stale=True)[0]
    edge_row = db.list_learning_graph_edges(status="approved", limit=10, include_stale=True)[0]
    resource_row = db.list_learning_graph_resource_links(status="approved", limit=10, include_stale=True)[0]
    path_row = db.get_latest_learning_graph_path(topic_slug="rag", status="approved", include_stale=True)

    assert claim_row["stale"] is True
    assert relation_row["stale"] is True
    assert memory_row["stale"] is True
    assert node_row["stale"] is True
    assert edge_row["stale"] is True
    assert resource_row["stale"] is True
    assert path_row is not None and path_row["stale"] is True

    assert db.list_ontology_claims(limit=10) == []
    assert db.list_relations(limit=10, source_type="note") == []
    assert (
        db.list_memory_relations(
            src_form="document_memory",
            src_id="note-1",
            relation_type="updates",
            limit=10,
        )
        == []
    )
    assert db.list_learning_graph_nodes(limit=10) == []
    assert db.list_learning_graph_edges(status="approved", limit=10) == []
    assert db.list_learning_graph_resource_links(status="approved", limit=10) == []
    assert db.get_latest_learning_graph_path(topic_slug="rag", status="approved") is None


def test_manual_memory_relations_are_not_auto_invalidated(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "mixed-lifecycle-manual.db"))
    db.upsert_note(
        "note-1",
        "Manual Relation Note",
        "body",
        file_path="vault/manual-relation.md",
        metadata={"source_content_hash": "hash-note-old"},
    )
    db.upsert_memory_relation(
        relation_id="rel-manual",
        src_form="document_memory",
        src_id="note-1",
        dst_form="document_memory",
        dst_id="note-1-other",
        relation_type="updates",
        confidence=0.9,
        provenance={"note_id": "note-1"},
        origin="manual",
    )

    db.merge_note_metadata("note-1", {"source_content_hash": "hash-note-new"})

    rows = db.list_memory_relations(
        src_form="document_memory",
        src_id="note-1",
        relation_type="updates",
        limit=10,
        include_stale=True,
    )
    assert len(rows) == 1
    assert rows[0]["origin"] == "manual"
    assert rows[0]["stale"] is False
    assert len(
        db.list_memory_relations(
            src_form="document_memory",
            src_id="note-1",
            relation_type="updates",
            limit=10,
        )
    ) == 1


def test_paper_memory_relations_go_stale_when_linked_paper_memory_row_is_invalidated(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "mixed-lifecycle-paper.db"))
    paper_id = "2401.00001"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Paper One",
            "authors": "A",
            "year": 2024,
            "field": "cs",
            "importance": 3,
            "notes": "",
            "pdf_path": None,
            "text_path": None,
            "translated_path": None,
            "source_content_hash": "hash-paper-old",
        }
    )
    db.upsert_paper_memory_card(
        card={
            "memory_id": "pm-1",
            "paper_id": paper_id,
            "title": "Paper One",
            "paper_core": "core",
            "problem_context": "problem",
            "method_core": "method",
            "evidence_core": "evidence",
            "limitations": "limits",
            "search_text": "paper one",
            "source_content_hash": "hash-paper-old",
        }
    )
    db.upsert_memory_relation(
        relation_id="rel-derived-paper",
        src_form="paper_memory",
        src_id="pm-1",
        dst_form="paper_memory",
        dst_id="pm-2",
        relation_type="updates",
        confidence=0.8,
        provenance={"paperId": paper_id},
    )

    changed = mark_derivatives_stale_for_document(
        db.conn,
        document_id=f"paper:{paper_id}",
        source_content_hash="hash-paper-new",
        source_type="paper",
    )

    assert changed >= 2
    rows = db.list_memory_relations(
        src_form="paper_memory",
        src_id="pm-1",
        relation_type="updates",
        limit=10,
        include_stale=True,
    )
    assert len(rows) == 1
    assert rows[0]["stale"] is True
    assert (
        db.list_memory_relations(
            src_form="paper_memory",
            src_id="pm-1",
            relation_type="updates",
            limit=10,
        )
        == []
    )
