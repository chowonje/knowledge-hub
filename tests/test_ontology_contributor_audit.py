from __future__ import annotations

from click.testing import CliRunner

from knowledge_hub.application.ontology_contributor_audit import audit_ontology_contributor_hashes
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands.ontology_cmd import ontology_group


class _StubKhub:
    def __init__(self, sqlite_db):
        self._sqlite_db = sqlite_db

    def sqlite_db(self):
        return self._sqlite_db


def _json_list(raw: str) -> list[str]:
    import json

    loaded = json.loads(raw or "[]")
    return [str(value) for value in loaded]


def test_ontology_contributor_audit_backfills_legacy_empty_concept_hashes(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "contributors.db"))
    db.upsert_ontology_entity("concept-a", "concept", "Concept A", source="test")
    db.conn.execute(
        "INSERT OR IGNORE INTO concepts (id, canonical_name, description) VALUES ('concept-a', 'Concept A', '')"
    )
    db.conn.execute(
        "UPDATE ontology_entities SET contributor_hashes = '[]' WHERE entity_id = 'concept-a'"
    )
    db.conn.execute("UPDATE concepts SET contributor_hashes = '[]' WHERE id = 'concept-a'")
    db.conn.execute(
        """
        INSERT INTO ontology_claims
          (claim_id, claim_text, subject_entity_id, predicate, object_entity_id,
           object_literal, confidence, evidence_ptrs_json, source, origin,
           source_content_hash, stale, stale_reason, invalidated_at)
        VALUES
          ('claim-legacy', 'Concept A appears in legacy evidence.', 'concept-a', 'mentions', NULL,
           NULL, 0.8, '[]', 'test', 'derived', 'hash-legacy', 0, '', '')
        """
    )
    db.conn.commit()

    dry_run = audit_ontology_contributor_hashes(db.conn)

    assert dry_run["apply"] is False
    assert dry_run["counts"]["missingContributorEntityCount"] == 1
    assert dry_run["counts"]["updatedEntityCount"] == 0
    assert dry_run["items"][0]["missingContributorHashes"] == ["hash-legacy"]
    row = db.conn.execute(
        "SELECT contributor_hashes FROM ontology_entities WHERE entity_id = 'concept-a'"
    ).fetchone()
    assert _json_list(row["contributor_hashes"]) == []

    applied = audit_ontology_contributor_hashes(db.conn, apply=True)

    assert applied["apply"] is True
    assert applied["counts"]["updatedEntityCount"] == 1
    assert applied["counts"]["updatedConceptMirrorCount"] == 1
    entity_row = db.conn.execute(
        "SELECT contributor_hashes FROM ontology_entities WHERE entity_id = 'concept-a'"
    ).fetchone()
    concept_row = db.conn.execute("SELECT contributor_hashes FROM concepts WHERE id = 'concept-a'").fetchone()
    assert _json_list(entity_row["contributor_hashes"]) == ["hash-legacy"]
    assert _json_list(concept_row["contributor_hashes"]) == ["hash-legacy"]


def test_ontology_contributor_audit_ignores_stale_and_manual_support(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "contributors-ignore.db"))
    db.upsert_ontology_entity("concept-a", "concept", "Concept A", source="test")
    db.conn.execute(
        """
        INSERT INTO ontology_claims
          (claim_id, claim_text, subject_entity_id, predicate, object_entity_id,
           object_literal, confidence, evidence_ptrs_json, source, origin,
           source_content_hash, stale, stale_reason, invalidated_at)
        VALUES
          ('claim-stale', 'Stale support.', 'concept-a', 'mentions', NULL,
           NULL, 0.8, '[]', 'test', 'derived', 'hash-stale', 1, 'test', CURRENT_TIMESTAMP),
          ('claim-manual', 'Manual support.', 'concept-a', 'mentions', NULL,
           NULL, 0.8, '[]', 'test', 'manual', 'hash-manual', 0, '', '')
        """
    )
    db.conn.commit()

    payload = audit_ontology_contributor_hashes(db.conn)

    assert payload["counts"]["contributorCandidateCount"] == 0
    assert payload["counts"]["missingContributorEntityCount"] == 0
    assert payload["items"] == []


def test_ontology_contributor_audit_cli_json(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "contributors-cli.db"))
    db.upsert_ontology_entity("concept-a", "concept", "Concept A", source="test")
    db.conn.execute(
        "UPDATE ontology_entities SET contributor_hashes = '[]' WHERE entity_id = 'concept-a'"
    )
    db.conn.execute(
        """
        INSERT INTO ontology_claims
          (claim_id, claim_text, subject_entity_id, predicate, object_entity_id,
           object_literal, confidence, evidence_ptrs_json, source, origin,
           source_content_hash, stale, stale_reason, invalidated_at)
        VALUES
          ('claim-cli', 'Concept A appears in CLI evidence.', 'concept-a', 'mentions', NULL,
           NULL, 0.8, '[]', 'test', 'derived', 'hash-cli', 0, '', '')
        """
    )
    db.conn.commit()

    result = CliRunner().invoke(
        ontology_group,
        ["contributor-audit", "--json"],
        obj={"khub": _StubKhub(db)},
    )

    assert result.exit_code == 0, result.output
    assert '"schema": "knowledge-hub.ontology-contributors.audit.v1"' in result.output
    assert '"missingContributorEntityCount": 1' in result.output
