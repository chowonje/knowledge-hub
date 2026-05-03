from __future__ import annotations

import sqlite3

from knowledge_hub.core.database import SQLiteDatabase
from knowledge_hub.infrastructure.persistence.stores.epistemic_store import EpistemicStore
from knowledge_hub.infrastructure.persistence.stores.learning_graph_store import LearningGraphStore
from knowledge_hub.infrastructure.persistence.stores.memory_relation_store import MemoryRelationStore
from knowledge_hub.infrastructure.persistence.stores.ontology_store import OntologyStore


def _connect_memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def test_sqlite_database_bootstraps_mixed_store_authority_columns(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "mixed-store-schema.db"))

    assert "origin" in _column_names(db.conn, "memory_relations")
    assert "origin" in _column_names(db.conn, "learning_graph_edges")
    assert "origin" in _column_names(db.conn, "learning_graph_resource_links")
    assert "contributor_hashes" in _column_names(db.conn, "concepts")
    assert "contributor_hashes" in _column_names(db.conn, "ontology_entities")
    assert "origin" in _column_names(db.conn, "ontology_claims")
    assert "origin" in _column_names(db.conn, "ontology_relations")
    assert "supersedes" in _column_names(db.conn, "beliefs")
    assert "superseded_by" in _column_names(db.conn, "beliefs")
    assert "supersedes" in _column_names(db.conn, "decisions")
    assert "superseded_by" in _column_names(db.conn, "decisions")


def test_memory_relation_store_adds_origin_to_legacy_rows():
    conn = _connect_memory_db()
    conn.execute(
        """
        CREATE TABLE memory_relations (
            relation_id TEXT PRIMARY KEY,
            src_form TEXT NOT NULL,
            src_id TEXT NOT NULL,
            dst_form TEXT NOT NULL,
            dst_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.0,
            provenance_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO memory_relations (
            relation_id, src_form, src_id, dst_form, dst_id, relation_type, confidence, provenance_json
        ) VALUES ('rel-1', 'paper', 'paper:1', 'paper', 'paper:2', 'updates', 0.8, '{}')
        """
    )
    MemoryRelationStore(conn).ensure_schema()

    row = conn.execute("SELECT origin FROM memory_relations WHERE relation_id = 'rel-1'").fetchone()
    assert row is not None
    assert row["origin"] == "derived"


def test_learning_graph_store_adds_origin_to_legacy_rows():
    conn = _connect_memory_db()
    conn.execute(
        """
        CREATE TABLE learning_graph_edges (
            edge_id TEXT PRIMARY KEY,
            source_node_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            target_node_id TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            status TEXT NOT NULL DEFAULT 'pending',
            provenance_json TEXT NOT NULL DEFAULT '{}',
            evidence_json TEXT NOT NULL DEFAULT '{}',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_node_id, edge_type, target_node_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE learning_graph_resource_links (
            link_id TEXT PRIMARY KEY,
            concept_node_id TEXT NOT NULL,
            resource_node_id TEXT NOT NULL,
            link_type TEXT NOT NULL,
            reading_stage TEXT NOT NULL DEFAULT 'intermediate',
            confidence REAL NOT NULL DEFAULT 0.5,
            status TEXT NOT NULL DEFAULT 'pending',
            provenance_json TEXT NOT NULL DEFAULT '{}',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(concept_node_id, resource_node_id, link_type)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO learning_graph_edges (
            edge_id, source_node_id, edge_type, target_node_id, confidence, status, provenance_json, evidence_json
        ) VALUES ('edge-1', 'node-a', 'requires', 'node-b', 0.7, 'approved', '{}', '{}')
        """
    )
    conn.execute(
        """
        INSERT INTO learning_graph_resource_links (
            link_id, concept_node_id, resource_node_id, link_type, reading_stage, confidence, status, provenance_json
        ) VALUES ('link-1', 'node-a', 'node-r', 'explains', 'intermediate', 0.6, 'approved', '{}')
        """
    )
    LearningGraphStore(conn).ensure_schema()

    edge_row = conn.execute("SELECT origin FROM learning_graph_edges WHERE edge_id = 'edge-1'").fetchone()
    link_row = conn.execute("SELECT origin FROM learning_graph_resource_links WHERE link_id = 'link-1'").fetchone()
    assert edge_row is not None and edge_row["origin"] == "derived"
    assert link_row is not None and link_row["origin"] == "derived"


def test_epistemic_store_adds_supersede_columns_to_legacy_rows():
    conn = _connect_memory_db()
    conn.execute(
        """
        CREATE TABLE beliefs (
            belief_id TEXT PRIMARY KEY,
            statement TEXT NOT NULL,
            scope TEXT NOT NULL DEFAULT 'global',
            status TEXT NOT NULL DEFAULT 'proposed',
            confidence REAL NOT NULL DEFAULT 0.5,
            derived_from_claim_ids_json TEXT NOT NULL DEFAULT '[]',
            support_ids_json TEXT NOT NULL DEFAULT '[]',
            contradiction_ids_json TEXT NOT NULL DEFAULT '[]',
            last_validated_at TEXT,
            review_due_at TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE decisions (
            decision_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            summary TEXT DEFAULT '',
            related_belief_ids_json TEXT NOT NULL DEFAULT '[]',
            chosen_option TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'open',
            review_due_at TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("INSERT INTO beliefs (belief_id, statement) VALUES ('belief-1', 'Belief statement')")
    conn.execute("INSERT INTO decisions (decision_id, title) VALUES ('decision-1', 'Decision title')")
    EpistemicStore(conn).ensure_schema()

    belief_row = conn.execute(
        "SELECT supersedes, superseded_by FROM beliefs WHERE belief_id = 'belief-1'"
    ).fetchone()
    decision_row = conn.execute(
        "SELECT supersedes, superseded_by FROM decisions WHERE decision_id = 'decision-1'"
    ).fetchone()
    assert belief_row is not None and belief_row["supersedes"] == "" and belief_row["superseded_by"] == ""
    assert decision_row is not None and decision_row["supersedes"] == "" and decision_row["superseded_by"] == ""


def test_ontology_store_adds_mixed_authority_columns_to_legacy_tables():
    conn = _connect_memory_db()
    conn.execute(
        """
        CREATE TABLE concepts (
            id TEXT PRIMARY KEY,
            canonical_name TEXT UNIQUE NOT NULL,
            description TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE ontology_entities (
            entity_id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            description TEXT DEFAULT '',
            properties_json TEXT DEFAULT '{}',
            confidence REAL DEFAULT 1.0,
            source TEXT DEFAULT 'system',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE ontology_claims (
            claim_id TEXT PRIMARY KEY,
            claim_text TEXT NOT NULL,
            subject_entity_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_entity_id TEXT,
            object_literal TEXT,
            confidence REAL DEFAULT 0.5,
            evidence_ptrs_json TEXT DEFAULT '[]',
            source TEXT DEFAULT 'extraction',
            valid_from TEXT,
            valid_to TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE ontology_relations (
            relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_entity_id TEXT NOT NULL,
            predicate_id TEXT NOT NULL,
            target_entity_id TEXT NOT NULL,
            source_type TEXT NOT NULL DEFAULT '',
            source_id TEXT NOT NULL DEFAULT '',
            target_type TEXT NOT NULL DEFAULT '',
            target_id TEXT NOT NULL DEFAULT '',
            confidence REAL DEFAULT 0.5,
            evidence_ptrs_json TEXT DEFAULT '[]',
            reason_json TEXT DEFAULT '{}',
            source TEXT DEFAULT 'system',
            valid_from TEXT,
            valid_to TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE kg_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            evidence_text TEXT DEFAULT '',
            confidence REAL DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_type, source_id, relation, target_type, target_id)
        )
        """
    )
    conn.execute("INSERT INTO concepts (id, canonical_name, description) VALUES ('concept-1', 'Concept One', 'desc')")
    conn.execute(
        """
        INSERT INTO ontology_entities (
            entity_id, entity_type, canonical_name, description, properties_json, confidence, source
        ) VALUES ('entity-1', 'concept', 'Entity One', 'desc', '{}', 1.0, 'system')
        """
    )
    conn.execute(
        """
        INSERT INTO ontology_entities (
            entity_id, entity_type, canonical_name, description, properties_json, confidence, source
        ) VALUES ('entity-2', 'concept', 'Entity Two', 'desc', '{}', 1.0, 'system')
        """
    )
    conn.execute(
        """
        INSERT INTO ontology_claims (
            claim_id, claim_text, subject_entity_id, predicate, object_entity_id, object_literal,
            confidence, evidence_ptrs_json, source
        ) VALUES ('claim-1', 'Entity One uses Entity Two', 'entity-1', 'uses', 'entity-2', '', 0.8, '[]', 'extraction')
        """
    )
    conn.execute(
        """
        INSERT INTO ontology_relations (
            source_entity_id, predicate_id, target_entity_id, source_type, source_id, target_type, target_id,
            confidence, evidence_ptrs_json, reason_json, source
        ) VALUES ('entity-1', 'uses', 'entity-2', 'concept', 'entity-1', 'concept', 'entity-2', 0.8, '[]', '{}', 'system')
        """
    )
    conn.execute(
        """
        INSERT INTO kg_relations (
            source_type, source_id, relation, target_type, target_id, evidence_text, confidence
        ) VALUES ('concept', 'entity-1', 'uses', 'concept', 'entity-2', '', 0.8)
        """
    )

    store = OntologyStore(conn)
    store.ensure_schema()
    store.ensure_legacy_kg_relations_schema()

    concept_row = conn.execute(
        "SELECT contributor_hashes FROM concepts WHERE id = 'concept-1'"
    ).fetchone()
    entity_row = conn.execute(
        "SELECT contributor_hashes FROM ontology_entities WHERE entity_id = 'entity-1'"
    ).fetchone()
    claim_row = conn.execute(
        "SELECT origin FROM ontology_claims WHERE claim_id = 'claim-1'"
    ).fetchone()
    relation_row = conn.execute(
        "SELECT origin FROM ontology_relations WHERE relation_id = 1"
    ).fetchone()
    kg_row = conn.execute("SELECT origin FROM kg_relations WHERE id = 1").fetchone()

    assert concept_row is not None and concept_row["contributor_hashes"] == "[]"
    assert entity_row is not None and entity_row["contributor_hashes"] == "[]"
    assert claim_row is not None and claim_row["origin"] == "derived"
    assert relation_row is not None and relation_row["origin"] == "derived"
    assert kg_row is not None and kg_row["origin"] == "derived"
