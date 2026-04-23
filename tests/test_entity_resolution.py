from __future__ import annotations

from knowledge_hub.core.config import Config
from knowledge_hub.core.database import SQLiteDatabase
from knowledge_hub.knowledge.entity_resolution import build_entity_merge_proposals_for_note


def _make_db(tmp_path) -> SQLiteDatabase:
    config = Config()
    config.set_nested("storage", "sqlite", str(tmp_path / "knowledge.db"))
    return SQLiteDatabase(config.sqlite_path)


def test_build_entity_merge_proposals_for_note_queues_duplicate_concepts(tmp_path):
    db = _make_db(tmp_path)
    db.upsert_ontology_entity("concept_rag", "concept", "Retrieval Augmented Generation")
    db.add_entity_alias("RAG", "concept_rag")
    db.upsert_ontology_entity("concept_rag_dup", "concept", "retrieval augmented generation")
    db.upsert_ontology_entity("concept_docs", "concept", "Documents")

    db.upsert_note(
        note_id="note:canonical",
        title="Canonical RAG",
        content="RAG overview",
        source_type="web",
        metadata={"url": "https://example.com/canonical", "topic": "rag"},
    )
    db.add_relation(
        source_type="note",
        source_id="note:canonical",
        relation="mentions",
        target_type="concept",
        target_id="concept_rag",
        evidence_text="RAG overview",
        confidence=0.9,
    )
    db.upsert_claim(
        claim_id="claim:rag-uses-docs",
        claim_text="RAG uses documents",
        subject_entity_id="concept_rag",
        predicate="uses",
        object_entity_id="concept_docs",
        confidence=0.86,
        evidence_ptrs=[{"note_id": "note:canonical"}],
    )

    db.upsert_note(
        note_id="note:dup",
        title="Duplicate Mention",
        content="retrieval augmented generation duplicates RAG",
        source_type="web",
        metadata={"url": "https://example.com/dup", "topic": "rag"},
    )
    db.add_relation(
        source_type="note",
        source_id="note:dup",
        relation="mentions",
        target_type="concept",
        target_id="concept_rag_dup",
        evidence_text="dup mention",
        confidence=0.88,
    )

    created = build_entity_merge_proposals_for_note(
        db,
        topic_slug="rag",
        note_id="note:dup",
        source_url="https://example.com/dup",
    )

    assert created
    proposals = db.list_entity_merge_proposals(topic_slug="rag", status="pending", limit=10)
    assert proposals
    assert proposals[0]["source_entity_id"] == "concept_rag_dup"
    assert proposals[0]["target_entity_id"] == "concept_rag"
    assert proposals[0]["reason_json"]["note_id"] == "note:dup"
    assert proposals[0]["reason_json"]["precision_first"]["suppressed"] is False
    assert proposals[0]["reason_json"]["precision_first"]["alias_overlap"]["strength"] == "strong"
    assert proposals[0]["source_entity"]["canonical_name"] == "retrieval augmented generation"
    assert proposals[0]["target_entity"]["canonical_name"] == "Retrieval Augmented Generation"
    assert proposals[0]["duplicate_cluster"]["size"] >= 2
    assert proposals[0]["proposal_provenance"]["topic_slug"] == "rag"

    db.close()


def test_add_entity_merge_proposal_suppresses_entity_type_mismatch(tmp_path):
    db = _make_db(tmp_path)
    db.upsert_ontology_entity("person_claude", "person", "Claude")
    db.upsert_ontology_entity("org_claude", "organization", "Claude")

    proposal_id = db.entity_resolution_store.add_entity_merge_proposal(
        source_entity_id="person_claude",
        target_entity_id="org_claude",
        topic_slug="precision",
        confidence=0.995,
        match_method="normalized_exact",
        reason={
            "source_display_name": "Claude",
            "target_display_name": "Claude",
            "normalized_source": ["claude"],
            "normalized_target": ["claude"],
        },
    )

    assert proposal_id == 0
    assert not db.list_entity_merge_proposals(topic_slug="precision", status="pending", limit=10)

    db.close()


def test_build_entity_merge_proposals_for_note_suppresses_false_positive_and_emits_split_signal(tmp_path):
    db = _make_db(tmp_path)
    db.upsert_ontology_entity("concept_context_engineering", "concept", "Context Engineering")
    db.upsert_ontology_entity("concept_content_engineering", "concept", "Content Engineering")
    db.upsert_ontology_entity("concept_prompts", "concept", "Prompts")
    db.upsert_ontology_entity("concept_agent_memory", "concept", "Agent Memory")
    db.upsert_ontology_entity("concept_cnc", "concept", "CNC Calibration")
    db.upsert_ontology_entity("concept_editorial_workflows", "concept", "Editorial Workflows")

    db.upsert_note(
        note_id="note:context-ai",
        title="Context Engineering for Agents",
        content="Context engineering coordinates prompts and memory for agents.",
        source_type="web",
        metadata={"url": "https://ai.example.com/context", "topic": "llm"},
    )
    db.add_relation(
        source_type="note",
        source_id="note:context-ai",
        relation="mentions",
        target_type="concept",
        target_id="concept_context_engineering",
        evidence_text="context engineering for agents",
        confidence=0.93,
    )
    db.upsert_claim(
        claim_id="claim:context-prompts",
        claim_text="Context engineering organizes prompts",
        subject_entity_id="concept_context_engineering",
        predicate="uses",
        object_entity_id="concept_prompts",
        confidence=0.87,
        evidence_ptrs=[{"note_id": "note:context-ai", "source_url": "https://ai.example.com/context"}],
    )
    db.upsert_claim(
        claim_id="claim:context-memory",
        claim_text="Context engineering improves agent memory",
        subject_entity_id="concept_context_engineering",
        predicate="improves",
        object_entity_id="concept_agent_memory",
        confidence=0.83,
        evidence_ptrs=[{"note_id": "note:context-ai", "source_url": "https://ai.example.com/context"}],
    )

    db.upsert_note(
        note_id="note:context-factory",
        title="Context Engineering in Factories",
        content="Context engineering also appears in factory calibration contexts.",
        source_type="web",
        metadata={"url": "https://ops.example.net/context", "topic": "industrial"},
    )
    db.add_relation(
        source_type="note",
        source_id="note:context-factory",
        relation="mentions",
        target_type="concept",
        target_id="concept_context_engineering",
        evidence_text="factory context engineering",
        confidence=0.86,
    )
    db.upsert_claim(
        claim_id="claim:context-cnc",
        claim_text="Context engineering calibrates CNC machines",
        subject_entity_id="concept_context_engineering",
        predicate="uses",
        object_entity_id="concept_cnc",
        confidence=0.72,
        evidence_ptrs=[{"note_id": "note:context-factory", "source_url": "https://ops.example.net/context"}],
    )

    db.upsert_note(
        note_id="note:content",
        title="Content Engineering",
        content="Content engineering drives editorial workflows.",
        source_type="web",
        metadata={"url": "https://cms.example.org/content", "topic": "publishing"},
    )
    db.add_relation(
        source_type="note",
        source_id="note:content",
        relation="mentions",
        target_type="concept",
        target_id="concept_content_engineering",
        evidence_text="content engineering",
        confidence=0.9,
    )
    db.upsert_claim(
        claim_id="claim:content-editorial",
        claim_text="Content engineering supports editorial workflows",
        subject_entity_id="concept_content_engineering",
        predicate="uses",
        object_entity_id="concept_editorial_workflows",
        confidence=0.85,
        evidence_ptrs=[{"note_id": "note:content", "source_url": "https://cms.example.org/content"}],
    )

    created = build_entity_merge_proposals_for_note(
        db,
        topic_slug="precision",
        note_id="note:context-factory",
        source_url="https://ops.example.net/context",
        fuzzy_threshold=0.88,
    )

    assert created == []
    assert not db.list_entity_merge_proposals(topic_slug="precision", status="pending", limit=10)

    split_proposals = db.entity_resolution_store.list_entity_split_proposals(
        topic_slug="precision",
        status="pending",
        limit=10,
    )
    assert split_proposals
    assert split_proposals[0]["source_entity_id"] == "concept_context_engineering"
    assert "concept_content_engineering" in split_proposals[0]["candidate_entities_json"]
    assert "topic_diversity" in split_proposals[0]["reason_json"]["overload_signals"]
    assert split_proposals[0]["reason_json"]["topic_diversity"] >= 2
    assert split_proposals[0]["reason_json"]["domain_diversity"] >= 2

    db.close()


def test_apply_entity_merge_proposal_records_side_effects(tmp_path):
    db = _make_db(tmp_path)
    db.upsert_ontology_entity("concept_rag", "concept", "Retrieval Augmented Generation")
    db.upsert_ontology_entity("concept_rag_dup", "concept", "RAG")
    db.upsert_ontology_entity("concept_docs", "concept", "Documents")
    db.add_entity_alias("RAG", "concept_rag_dup")
    db.upsert_claim(
        claim_id="claim:rag-uses-docs",
        claim_text="RAG uses documents",
        subject_entity_id="concept_rag_dup",
        predicate="uses",
        object_entity_id="concept_docs",
        confidence=0.84,
        evidence_ptrs=[{"note_id": "note:dup"}],
    )
    db.add_relation(
        source_type="concept",
        source_id="concept_rag_dup",
        relation="related_to",
        target_type="concept",
        target_id="concept_docs",
        evidence_text="RAG is related to documents",
        confidence=0.8,
    )
    db.upsert_feature_snapshot(
        topic_slug="rag",
        feature_kind="concept",
        feature_key="concept_rag_dup",
        feature_name="RAG",
        entity_id="concept_rag_dup",
        note_id=None,
        record_id=None,
        freshness_score=0.6,
        importance_score=0.8,
        support_doc_count=2.0,
        relation_degree=1.0,
        claim_density=0.5,
        source_trust_score=0.7,
        concept_activity_score=0.6,
        contradiction_score=0.0,
        payload={},
    )
    db.upsert_learning_graph_node(
        node_id="node:rag-dup",
        entity_id="concept_rag_dup",
        node_type="concept",
        canonical_name="RAG",
        difficulty_level="intermediate",
        difficulty_score=0.5,
        stage="intermediate",
        confidence=0.8,
        provenance={"topicSlug": "rag"},
    )
    db.upsert_learning_graph_edge(
        edge_id="edge:rag-dup",
        source_node_id="node:rag-dup",
        edge_type="recommended_before",
        target_node_id="node:rag-dup",
        confidence=0.7,
        status="approved",
        provenance={"topicSlug": "rag"},
        evidence={},
    )
    db.upsert_learning_graph_path(
        path_id="path:rag",
        topic_slug="rag",
        status="approved",
        version=1,
        path_payload={"nodes": ["node:rag-dup"]},
        score_payload={},
        provenance={"topicSlug": "rag"},
    )
    db.add_learning_graph_pending(
        item_type="edge",
        topic_slug="rag",
        payload={
            "sourceNodeId": "node:rag-dup",
            "edgeType": "recommended_before",
            "targetNodeId": "node:rag-dup",
            "provenance": {"topicSlug": "rag"},
        },
        confidence=0.6,
        reason="test",
        provenance={"topicSlug": "rag"},
    )

    proposal_id = db.add_entity_merge_proposal(
        source_entity_id="concept_rag_dup",
        target_entity_id="concept_rag",
        topic_slug="rag",
        confidence=0.9,
        match_method="alias_exact",
        reason={"note_id": "note:dup"},
    )
    assert proposal_id > 0
    assert db.apply_entity_merge_proposal(proposal_id) is True

    item = db.get_entity_merge_proposal(proposal_id)
    assert item is not None
    assert item["status"] == "approved"
    side_effects = item["reason_json"]["apply_side_effects"]
    assert side_effects["aliases_added"] >= 1
    assert side_effects["subject_claims_moved"] == 1
    assert side_effects["source_relations_moved"] >= 1
    assert side_effects["feature_snapshots_deleted"] >= 1
    assert side_effects["learning_graph_nodes_deleted"] >= 1
    assert "rag" in side_effects["affected_topics"]
    assert side_effects["source_entity_deleted"] == 1
    assert db.get_ontology_entity("concept_rag_dup") is None
    claims = db.list_claims_by_entity("concept_rag")
    assert any(claim["claim_id"] == "claim:rag-uses-docs" for claim in claims)
    assert db.get_feature_snapshot(topic_slug="rag", feature_kind="concept", feature_key="concept_rag_dup") is None
    assert db.get_latest_learning_graph_path(topic_slug="rag", status="approved") is None
    assert not db.list_learning_graph_pending(topic_slug="rag", status="pending", limit=10)
    events = db.list_learning_graph_events(topic_slug="rag", limit=10)
    assert any(event["event_type"] == "entity_merge_invalidation" for event in events)
    ontology_events = db.list_ontology_events(entity_id="concept_rag_dup", event_type="entity_deleted", limit=10)
    assert any(event["event_type"] == "entity_deleted" for event in ontology_events)

    db.close()
