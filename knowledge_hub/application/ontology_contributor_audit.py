"""Audit and backfill ontology concept contributor hashes."""

from __future__ import annotations

from typing import Any

from knowledge_hub.infrastructure.persistence.stores.derivative_lifecycle import (
    clean_token,
    column_names,
    merge_token_json,
    normalize_token_list,
    table_exists,
)


def _live_source_where(columns: set[str]) -> str:
    clauses = ["source_content_hash != ''"]
    if "stale" in columns:
        clauses.append("COALESCE(stale, 0) = 0")
    if "origin" in columns:
        clauses.append("COALESCE(origin, 'derived') = 'derived'")
    return " AND ".join(clauses)


def _add_expected(expected: dict[str, set[str]], entity_id: Any, source_hash: Any, concept_ids: set[str]) -> None:
    entity_token = clean_token(entity_id)
    hash_token = clean_token(source_hash)
    if not entity_token or not hash_token or entity_token not in concept_ids:
        return
    expected.setdefault(entity_token, set()).add(hash_token)


def _load_concepts(conn: Any) -> dict[str, dict[str, Any]]:
    concepts: dict[str, dict[str, Any]] = {}
    if table_exists(conn, "ontology_entities"):
        columns = column_names(conn, "ontology_entities")
        selected = ["entity_id", "canonical_name"]
        if "contributor_hashes" in columns:
            selected.append("contributor_hashes")
        if "stale" in columns:
            selected.append("stale")
        rows = conn.execute(
            f"""
            SELECT {', '.join(selected)}
            FROM ontology_entities
            WHERE entity_type = 'concept'
            """
        ).fetchall()
        for row in rows:
            item = dict(row)
            entity_id = clean_token(item.get("entity_id"))
            if not entity_id:
                continue
            concepts[entity_id] = {
                "entityId": entity_id,
                "canonicalName": clean_token(item.get("canonical_name")),
                "currentContributorHashes": normalize_token_list(item.get("contributor_hashes", "")),
                "stale": bool(item.get("stale", 0)),
                "hasOntologyEntity": True,
                "hasConceptMirror": False,
            }
    if table_exists(conn, "concepts"):
        columns = column_names(conn, "concepts")
        selected = ["id", "canonical_name"]
        if "contributor_hashes" in columns:
            selected.append("contributor_hashes")
        rows = conn.execute(f"SELECT {', '.join(selected)} FROM concepts").fetchall()
        for row in rows:
            item = dict(row)
            entity_id = clean_token(item.get("id"))
            if not entity_id:
                continue
            existing = concepts.setdefault(
                entity_id,
                {
                    "entityId": entity_id,
                    "canonicalName": clean_token(item.get("canonical_name")),
                    "currentContributorHashes": normalize_token_list(item.get("contributor_hashes", "")),
                    "stale": False,
                    "hasOntologyEntity": False,
                    "hasConceptMirror": True,
                },
            )
            existing["hasConceptMirror"] = True
            if not existing.get("canonicalName"):
                existing["canonicalName"] = clean_token(item.get("canonical_name"))
            if not existing.get("currentContributorHashes"):
                existing["currentContributorHashes"] = normalize_token_list(item.get("contributor_hashes", ""))
    return concepts


def _collect_expected_hashes(conn: Any, *, concept_ids: set[str]) -> dict[str, set[str]]:
    expected: dict[str, set[str]] = {}
    if table_exists(conn, "ontology_claims"):
        columns = column_names(conn, "ontology_claims")
        if {"subject_entity_id", "source_content_hash"}.issubset(columns):
            optional_object = "object_entity_id" in columns
            select_cols = ["subject_entity_id", "source_content_hash"]
            if optional_object:
                select_cols.append("object_entity_id")
            rows = conn.execute(
                f"""
                SELECT {', '.join(select_cols)}
                FROM ontology_claims
                WHERE {_live_source_where(columns)}
                """
            ).fetchall()
            for row in rows:
                item = dict(row)
                source_hash = item.get("source_content_hash")
                _add_expected(expected, item.get("subject_entity_id"), source_hash, concept_ids)
                if optional_object:
                    _add_expected(expected, item.get("object_entity_id"), source_hash, concept_ids)
    if table_exists(conn, "ontology_relations"):
        columns = column_names(conn, "ontology_relations")
        if {"source_entity_id", "target_entity_id", "source_content_hash"}.issubset(columns):
            rows = conn.execute(
                f"""
                SELECT source_entity_id, target_entity_id, source_content_hash
                FROM ontology_relations
                WHERE {_live_source_where(columns)}
                """
            ).fetchall()
            for row in rows:
                item = dict(row)
                source_hash = item.get("source_content_hash")
                _add_expected(expected, item.get("source_entity_id"), source_hash, concept_ids)
                _add_expected(expected, item.get("target_entity_id"), source_hash, concept_ids)
    if table_exists(conn, "kg_relations"):
        columns = column_names(conn, "kg_relations")
        if {"source_type", "source_id", "target_type", "target_id", "source_content_hash"}.issubset(columns):
            rows = conn.execute(
                f"""
                SELECT source_type, source_id, target_type, target_id, source_content_hash
                FROM kg_relations
                WHERE {_live_source_where(columns)}
                """
            ).fetchall()
            for row in rows:
                item = dict(row)
                source_hash = item.get("source_content_hash")
                if clean_token(item.get("source_type")) == "concept":
                    _add_expected(expected, item.get("source_id"), source_hash, concept_ids)
                if clean_token(item.get("target_type")) == "concept":
                    _add_expected(expected, item.get("target_id"), source_hash, concept_ids)
    return expected


def audit_ontology_contributor_hashes(conn: Any, *, apply: bool = False, sample_limit: int = 50) -> dict[str, Any]:
    concepts = _load_concepts(conn)
    concept_ids = set(concepts)
    expected = _collect_expected_hashes(conn, concept_ids=concept_ids)
    items: list[dict[str, Any]] = []
    updated_entity_count = 0
    updated_concept_mirror_count = 0
    missing_hash_count = 0
    extra_entity_count = 0

    ontology_entity_columns = column_names(conn, "ontology_entities")
    concept_columns = column_names(conn, "concepts")
    can_update_entity = table_exists(conn, "ontology_entities") and "contributor_hashes" in ontology_entity_columns
    can_update_concept = table_exists(conn, "concepts") and "contributor_hashes" in concept_columns

    for entity_id in sorted(concepts):
        concept = concepts[entity_id]
        current = normalize_token_list(concept.get("currentContributorHashes", []))
        expected_hashes = sorted(expected.get(entity_id, set()))
        missing = [value for value in expected_hashes if value not in set(current)]
        extra = [value for value in current if value not in set(expected_hashes)]
        if extra:
            extra_entity_count += 1
        if not missing:
            continue
        missing_hash_count += len(missing)
        merged = merge_token_json(current, missing)
        action = "would_update"
        if apply:
            if can_update_entity and concept.get("hasOntologyEntity"):
                conn.execute(
                    """
                    UPDATE ontology_entities
                    SET contributor_hashes = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE entity_id = ?
                    """,
                    (merged, entity_id),
                )
                updated_entity_count += 1
            if can_update_concept and concept.get("hasConceptMirror"):
                conn.execute("UPDATE concepts SET contributor_hashes = ? WHERE id = ?", (merged, entity_id))
                updated_concept_mirror_count += 1
            action = "updated"
        if len(items) < max(1, int(sample_limit)):
            items.append(
                {
                    "entityId": entity_id,
                    "canonicalName": concept.get("canonicalName", ""),
                    "currentContributorHashes": current,
                    "expectedContributorHashes": expected_hashes,
                    "missingContributorHashes": missing,
                    "extraContributorHashes": extra,
                    "stale": bool(concept.get("stale", False)),
                    "action": action,
                }
            )
    if apply:
        conn.commit()
    return {
        "schema": "knowledge-hub.ontology-contributors.audit.v1",
        "status": "ok",
        "apply": bool(apply),
        "counts": {
            "conceptEntityCount": len(concepts),
            "contributorCandidateCount": len(expected),
            "missingContributorEntityCount": sum(1 for entity_id in concepts if expected.get(entity_id) and set(expected[entity_id]) - set(normalize_token_list(concepts[entity_id].get("currentContributorHashes", [])))),
            "missingContributorHashCount": missing_hash_count,
            "extraContributorEntityCount": extra_entity_count,
            "updatedEntityCount": updated_entity_count,
            "updatedConceptMirrorCount": updated_concept_mirror_count,
        },
        "items": items,
    }


__all__ = ["audit_ontology_contributor_hashes"]
