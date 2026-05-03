from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.database import SQLiteDatabase
from knowledge_hub.infrastructure.persistence.stores.ontology_store import OntologyStore
from knowledge_hub.infrastructure.persistence.stores.section_card_v1_store import SectionCardV1Store


INVENTORY_PATH = Path("docs/store_authority_inventory.json")
LIFECYCLE_COLUMNS = {"source_content_hash", "stale", "stale_reason", "invalidated_at"}
REQUIRED_FIELDS = {"table", "authority", "requires_lifecycle", "evidence_role", "manual_origin_allowed"}


def _columns(conn, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _table_exists(conn, table: str) -> bool:
    return conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table,)).fetchone() is not None


def _inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def test_store_authority_inventory_is_machine_readable_and_unique():
    payload = _inventory()
    assert payload["schema"] == "knowledge-hub.store-authority-inventory.v1"
    tables = payload["tables"]
    assert tables
    names = [item["table"] for item in tables]
    assert len(names) == len(set(names))
    for item in tables:
        assert REQUIRED_FIELDS.issubset(item)
        assert isinstance(item["requires_lifecycle"], bool)
        assert item["evidence_role"] != "citation_endpoint"


def test_lifecycle_required_inventory_tables_have_lifecycle_columns(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "store-authority.db"))
    SectionCardV1Store(db.conn).ensure_schema()
    ontology_store = OntologyStore(db.conn)
    ontology_store.ensure_legacy_kg_relations_schema()
    ontology_store.ensure_schema()

    for item in _inventory()["tables"]:
        table = item["table"]
        assert _table_exists(db.conn, table), table
        if item["requires_lifecycle"]:
            assert LIFECYCLE_COLUMNS.issubset(_columns(db.conn, table)), table


def test_inventory_marks_signal_tables_as_non_citation_evidence():
    signal_tables = {
        "concepts",
        "ontology_entities",
        "ontology_claims",
        "ontology_relations",
        "kg_relations",
        "memory_relations",
        "learning_graph_nodes",
        "learning_graph_edges",
        "learning_graph_resource_links",
        "learning_graph_paths",
    }
    by_table = {item["table"]: item for item in _inventory()["tables"]}

    for table in signal_tables:
        assert by_table[table]["evidence_role"] == "retrieval_signal_only"
