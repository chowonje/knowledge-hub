from __future__ import annotations

from datetime import datetime, timezone

from knowledge_hub.core.database import SQLiteDatabase


def _make_db(tmp_path) -> SQLiteDatabase:
    return SQLiteDatabase(str(tmp_path / "knowledge.db"))


def test_repair_missing_entity_events_backfills_orphans(tmp_path):
    db = _make_db(tmp_path)
    db.conn.execute(
        """
        INSERT INTO ontology_entities
            (entity_id, entity_type, canonical_name, description, properties_json, confidence, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "note:web_orphan",
            "event",
            "Orphan Note",
            "/tmp/orphan.md",
            '{"note_id":"web_orphan","source_type":"web"}',
            0.8,
            "relation_runtime",
        ),
    )
    db.conn.commit()

    sample = db.event_store.list_entities_without_events(limit=10)
    assert [item["entity_id"] for item in sample] == ["note:web_orphan"]

    payload = db.event_store.repair_missing_entity_events(run_id="test-repair")

    assert payload["repaired_count"] == 1
    assert payload["remaining_count"] == 0
    assert payload["entity_ids"] == ["note:web_orphan"]
    repaired_events = db.list_ontology_events(entity_id="note:web_orphan", event_type="entity_created", limit=10)
    assert any(event["event_type"] == "entity_created" for event in repaired_events)
    snapshot = db.event_store.snapshot_at(datetime.now(timezone.utc).isoformat())
    assert snapshot["entity_count"] == 1
    db.close()


def test_relation_runtime_entity_creation_appends_event(tmp_path):
    db = _make_db(tmp_path)
    db.upsert_ontology_entity("concept_agents", "concept", "Agents")

    db.add_relation(
        source_type="note",
        source_id="web_relation_runtime",
        relation="mentions",
        target_type="concept",
        target_id="concept_agents",
        evidence_text="mentions",
        confidence=0.7,
    )

    events = db.list_ontology_events(entity_id="note:web_relation_runtime", event_type="entity_created", limit=10)
    assert any(event["event_type"] == "entity_created" for event in events)
    db.close()
