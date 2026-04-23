"""Canonical ontology relation/query store helpers."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from knowledge_hub.core.models import OntologyEvent

log = logging.getLogger("khub.ontology_store")

CORE_PREDICATES = {
    "mentions",
    "uses",
    "causes",
    "enables",
    "part_of",
    "contrasts",
    "example_of",
    "requires",
    "improves",
    "related_to",
    "unknown_relation",
}

MINIMUM_FORMAL_SEMANTICS_PREDICATES = frozenset(
    {
        "causes",
        "enables",
        "part_of",
        "contrasts",
        "requires",
        "improves",
    }
)


def _safe_json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_predicate(relation: str, evidence_text: str = "") -> str:
    token = str(relation or "").strip()
    if token == "note_mentions_concept":
        return "mentions"
    if token == "paper_uses_concept":
        return "uses"
    if token == "concept_related_to":
        candidate = str(_safe_json_dict(evidence_text).get("relation_norm", "")).strip().lower()
        if candidate in CORE_PREDICATES:
            return candidate
        return "related_to"
    if token in CORE_PREDICATES:
        return token
    # extension predicate candidates are validated by _ensure_predicate()
    return token or "related_to"


class OntologyStore:
    """Relation/claim/event query helpers bound to a sqlite connection."""

    def __init__(self, conn, event_store=None, db_path: str | Path | None = None):
        self.conn = conn
        self.event_store = event_store
        self.db_path = Path(db_path) if db_path else None
        self._table_exists_cache: dict[str, bool] = {}
        self._table_columns_cache: dict[str, set[str]] = {}
        self._entity_type_support_cache: dict[str, bool] = {}

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                canonical_name TEXT UNIQUE NOT NULL,
                description TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS concept_aliases (
                alias TEXT PRIMARY KEY,
                concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL
                    CHECK(entity_type IN ('concept','claim','paper','person','organization','event','note')),
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
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_oe_type_name
            ON ontology_entities(entity_type, canonical_name)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_oe_type
            ON ontology_entities(entity_type)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_aliases (
                alias TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL REFERENCES ontology_entities(entity_id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entity_aliases_entity
            ON entity_aliases(entity_id)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_claims (
                claim_id TEXT PRIMARY KEY,
                claim_text TEXT NOT NULL,
                subject_entity_id TEXT NOT NULL REFERENCES ontology_entities(entity_id),
                predicate TEXT NOT NULL,
                object_entity_id TEXT REFERENCES ontology_entities(entity_id),
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
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_oc_subject
            ON ontology_claims(subject_entity_id)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_oc_predicate
            ON ontology_claims(predicate)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_oc_object
            ON ontology_claims(object_entity_id)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_predicates (
                predicate_id TEXT PRIMARY KEY,
                parent_predicate_id TEXT REFERENCES ontology_predicates(predicate_id),
                status TEXT NOT NULL DEFAULT 'core'
                    CHECK(status IN ('core','approved_ext','deprecated')),
                description TEXT DEFAULT '',
                source TEXT DEFAULT 'system',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_relations (
                relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity_id TEXT NOT NULL REFERENCES ontology_entities(entity_id),
                predicate_id TEXT NOT NULL REFERENCES ontology_predicates(predicate_id),
                target_entity_id TEXT NOT NULL REFERENCES ontology_entities(entity_id),
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
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_entity_id, predicate_id, target_entity_id, source)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_or_source_entity
            ON ontology_relations(source_entity_id, predicate_id)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_or_target_entity
            ON ontology_relations(target_entity_id, predicate_id)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_or_legacy_source
            ON ontology_relations(source_type, source_id)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_or_legacy_target
            ON ontology_relations(target_type, target_id)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_pending (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pending_type TEXT NOT NULL
                    CHECK(pending_type IN ('concept','relation','claim','predicate_ext')),
                run_id TEXT NOT NULL,
                topic_slug TEXT NOT NULL DEFAULT '',
                note_id TEXT NOT NULL DEFAULT '',
                source_url TEXT NOT NULL DEFAULT '',
                source_entity_id TEXT NOT NULL DEFAULT '',
                predicate_id TEXT NOT NULL DEFAULT '',
                target_entity_id TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                evidence_ptrs_json TEXT NOT NULL DEFAULT '[]',
                reason_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP,
                CHECK(status IN ('pending', 'approved', 'rejected'))
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ontology_pending_status
            ON ontology_pending(status, created_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ontology_pending_type
            ON ontology_pending(pending_type, status, created_at DESC)
            """
        )
        self.conn.commit()

    def ensure_core_predicates(self) -> None:
        predicate_defaults = {
            "mentions": {},
            "uses": {},
            "causes": {"domain_source_type": "concept", "range_target_type": "concept"},
            "enables": {"domain_source_type": "concept", "range_target_type": "concept"},
            "part_of": {
                "domain_source_type": "concept",
                "range_target_type": "concept",
                "is_transitive": True,
                "is_antisymmetric": True,
            },
            "contrasts": {
                "domain_source_type": "concept",
                "range_target_type": "concept",
                "is_symmetric": True,
            },
            "example_of": {},
            "requires": {
                "domain_source_type": "concept",
                "range_target_type": "concept",
                "is_transitive": True,
            },
            "improves": {"domain_source_type": "concept", "range_target_type": "concept"},
            "related_to": {},
            "unknown_relation": {},
        }
        for predicate_id in CORE_PREDICATES:
            self.upsert_predicate(
                predicate_id=predicate_id,
                parent_predicate_id=None,
                status="core",
                description="",
                source="system",
                **predicate_defaults.get(predicate_id, {}),
            )

    def _table_exists(self, table_name: str) -> bool:
        if table_name in self._table_exists_cache:
            return self._table_exists_cache[table_name]
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        exists = bool(row)
        self._table_exists_cache[table_name] = exists
        return exists

    def _table_columns(self, table_name: str) -> set[str]:
        if table_name in self._table_columns_cache:
            return self._table_columns_cache[table_name]
        if not self._table_exists(table_name):
            self._table_columns_cache[table_name] = set()
            return set()
        rows = self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        cols = {str(row["name"]) for row in rows}
        self._table_columns_cache[table_name] = cols
        return cols

    @staticmethod
    def _decode_entity_row(row: Any) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        try:
            item["properties"] = json.loads(item.get("properties_json") or "{}")
        except Exception:
            item["properties"] = {}
        return item

    def _upsert_entity(
        self,
        entity_id: str,
        entity_type: str,
        canonical_name: str,
        *,
        description: str = "",
        properties_json: str = "{}",
        confidence: float = 1.0,
        source: str = "relation_runtime",
        emit_create_event: bool = False,
    ) -> None:
        if not self._table_exists("ontology_entities"):
            return
        columns = self._table_columns("ontology_entities")
        if "entity_id" not in columns:
            return
        existing = self.get_ontology_entity(entity_id) if emit_create_event and self.event_store else None

        target_entity_type = str(entity_type)
        if not self._ontology_entity_type_supported(target_entity_type):
            target_entity_type = "event"

        insert_payload: dict[str, Any] = {"entity_id": entity_id}
        if "entity_type" in columns:
            insert_payload["entity_type"] = target_entity_type
        if "canonical_name" in columns:
            insert_payload["canonical_name"] = canonical_name
        if "description" in columns:
            insert_payload["description"] = description
        if "properties_json" in columns:
            insert_payload["properties_json"] = properties_json
        if "confidence" in columns:
            insert_payload["confidence"] = float(confidence)
        if "source" in columns:
            insert_payload["source"] = source

        cols_sql = ", ".join(insert_payload.keys())
        placeholders = ", ".join(["?"] * len(insert_payload))
        self.conn.execute(
            f"INSERT OR IGNORE INTO ontology_entities ({cols_sql}) VALUES ({placeholders})",
            tuple(insert_payload.values()),
        )

        update_parts: list[str] = []
        update_params: list[Any] = []
        if "entity_type" in columns:
            update_parts.append("entity_type = ?")
            update_params.append(target_entity_type)
        if "canonical_name" in columns:
            update_parts.append("canonical_name = ?")
            update_params.append(canonical_name)
        if "description" in columns:
            update_parts.append("description = ?")
            update_params.append(description)
        if "properties_json" in columns:
            update_parts.append("properties_json = ?")
            update_params.append(properties_json)
        if "confidence" in columns:
            update_parts.append("confidence = ?")
            update_params.append(float(confidence))
        if "source" in columns:
            update_parts.append("source = ?")
            update_params.append(source)
        if "updated_at" in columns:
            update_parts.append("updated_at = CURRENT_TIMESTAMP")

        if update_parts:
            self.conn.execute(
                f"UPDATE ontology_entities SET {', '.join(update_parts)} WHERE entity_id = ?",
                (*update_params, entity_id),
            )
        if emit_create_event and self.event_store and existing is None:
            properties: dict[str, Any]
            try:
                properties = json.loads(properties_json or "{}")
            except Exception:
                properties = {}
            event = OntologyEvent(
                event_id=f"evt_{uuid4().hex}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="entity_created",
                entity_id=str(entity_id),
                entity_type=str(target_entity_type),
                actor=str(source),
                data={
                    "canonical_name": str(canonical_name or ""),
                    "description": str(description or ""),
                    "properties": properties,
                    "confidence": float(confidence),
                    "entity_type": str(target_entity_type),
                    "source": str(source),
                },
                policy_class="P2",
            )
            try:
                self.event_store.append(event)
            except Exception as error:
                log.error("Event append failed for implicit entity create (%s): %s", entity_id, error)

    def _get_system_meta(self, key: str, default: str = "") -> str:
        if not self._table_exists("system_meta"):
            return default
        row = self.conn.execute(
            "SELECT value FROM system_meta WHERE key = ?",
            (str(key),),
        ).fetchone()
        if not row:
            return default
        return str(row["value"] or default)

    def _set_system_meta(self, key: str, value: str) -> None:
        if not self._table_exists("system_meta"):
            return
        self.conn.execute(
            """INSERT INTO system_meta(key, value, updated_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(key) DO UPDATE SET
                 value=excluded.value,
                 updated_at=CURRENT_TIMESTAMP""",
            (str(key), str(value)),
        )
        self.conn.commit()

    def _backup_db_snapshot(self) -> str:
        if not self.db_path:
            raise RuntimeError("db_path is required for ontology core backup")
        backup_dir = self.db_path.parent / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"knowledge_{stamp}.db"
        idx = 0
        while backup_path.exists():
            idx += 1
            backup_path = backup_dir / f"knowledge_{stamp}_{idx}.db"

        backup_conn = sqlite3.connect(str(backup_path))
        try:
            self.conn.backup(backup_conn)
            backup_conn.commit()
        finally:
            backup_conn.close()
        return str(backup_path)

    def _ontology_entity_type_supported(self, entity_type: str) -> bool:
        token = str(entity_type or "").strip()
        if not token:
            return False
        if token in self._entity_type_support_cache:
            return self._entity_type_support_cache[token]
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='ontology_entities'"
        ).fetchone()
        ddl = str(row["sql"] if row and row["sql"] else "")
        supported = True
        if "entity_type IN" in ddl:
            supported = f"'{token}'" in ddl
        self._entity_type_support_cache[token] = supported
        return supported

    def ensure_legacy_kg_relations_schema(self) -> None:
        """Compatibility helper for legacy migration fixtures only."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kg_relations (
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
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kg_source
            ON kg_relations(source_type, source_id)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kg_target
            ON kg_relations(target_type, target_id)
            """
        )

    def run_core_migration(self) -> None:
        if self._get_system_meta("ontology_core_v2_migrated", "") == "1":
            return

        kg_count = 0
        if self._table_exists("kg_relations"):
            kg_count_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM kg_relations").fetchone()
            kg_count = int(kg_count_row["cnt"]) if kg_count_row else 0

        if kg_count > 0:
            backup_path = self._backup_db_snapshot()
            self._set_system_meta("ontology_core_v2_backup_path", backup_path)

        if self._table_exists("ontology_entities"):
            legacy_papers = self.conn.execute(
                "SELECT * FROM ontology_entities WHERE entity_type = 'paper'"
            ).fetchall()
            for row in legacy_papers:
                old_id = str(row["entity_id"])
                new_id = self._paper_entity_id(old_id)
                if old_id == new_id:
                    continue
                existing_new = self.get_ontology_entity(new_id)
                if not existing_new:
                    self.conn.execute(
                        "UPDATE ontology_entities SET entity_id = ? WHERE entity_id = ?",
                        (new_id, old_id),
                    )
                else:
                    self._upsert_entity(
                        entity_id=new_id,
                        entity_type="paper",
                        canonical_name=str(row["canonical_name"] or new_id),
                        description=str(row["description"] or ""),
                        properties_json=json.dumps(
                            _safe_json_dict(row["properties_json"]),
                            ensure_ascii=False,
                        ),
                        confidence=float(row["confidence"] or 1.0),
                        source=str(row["source"] or "migration_paper_id"),
                    )
                if self._table_exists("entity_aliases"):
                    self.conn.execute(
                        "UPDATE entity_aliases SET entity_id = ? WHERE entity_id = ?",
                        (new_id, old_id),
                    )
                if self._table_exists("ontology_claims"):
                    self.conn.execute(
                        "UPDATE ontology_claims SET subject_entity_id = ? WHERE subject_entity_id = ?",
                        (new_id, old_id),
                    )
                    self.conn.execute(
                        "UPDATE ontology_claims SET object_entity_id = ? WHERE object_entity_id = ?",
                        (new_id, old_id),
                    )
                if self._table_exists("ontology_relations"):
                    self.conn.execute(
                        "UPDATE ontology_relations SET source_entity_id = ? WHERE source_entity_id = ?",
                        (new_id, old_id),
                    )
                    self.conn.execute(
                        "UPDATE ontology_relations SET target_entity_id = ? WHERE target_entity_id = ?",
                        (new_id, old_id),
                    )
                if existing_new:
                    self.conn.execute("DELETE FROM ontology_entities WHERE entity_id = ?", (old_id,))

        if self._table_exists("papers"):
            paper_rows = self.conn.execute(
                "SELECT * FROM papers ORDER BY year DESC, importance DESC LIMIT 50000"
            ).fetchall()
            for paper_row in paper_rows:
                paper = dict(paper_row)
                arxiv_id = str(paper.get("arxiv_id", "")).strip()
                if not arxiv_id:
                    continue
                self._upsert_entity(
                    entity_id=self._paper_entity_id(arxiv_id),
                    entity_type="paper",
                    canonical_name=str(paper.get("title", "") or self._paper_entity_id(arxiv_id)),
                    description=f"{paper.get('authors', '')} ({paper.get('year', '')})".strip(),
                    properties_json=json.dumps(
                        {
                            "arxiv_id": arxiv_id,
                            "authors": paper.get("authors", ""),
                            "year": paper.get("year", 0),
                            "field": paper.get("field", ""),
                        },
                        ensure_ascii=False,
                    ),
                    source="migration_papers_table",
                )

        if kg_count > 0 and self._table_exists("ontology_relations"):
            kg_rows = self.conn.execute("SELECT * FROM kg_relations ORDER BY id ASC").fetchall()
            for row in kg_rows:
                source_type = str(row["source_type"] or "")
                source_id = str(row["source_id"] or "")
                target_type = str(row["target_type"] or "")
                target_id = str(row["target_id"] or "")
                if not source_id or not target_id:
                    continue
                source_entity_id, source_type_norm, source_id_norm = self._ensure_entity_ref(source_type, source_id)
                target_entity_id, target_type_norm, target_id_norm = self._ensure_entity_ref(target_type, target_id)
                if not source_entity_id or not target_entity_id:
                    continue

                legacy_relation = str(row["relation"] or "")
                evidence_text = str(row["evidence_text"] or "")
                evidence_json = _safe_json_dict(evidence_text)
                predicate_id = _normalize_predicate(legacy_relation, evidence_text=evidence_text)
                confidence = float(row["confidence"] or 0.5)
                reason = {"legacy_relation": legacy_relation}
                if evidence_json:
                    reason["legacy_evidence"] = evidence_json
                self.conn.execute(
                    """INSERT INTO ontology_relations
                         (source_entity_id, predicate_id, target_entity_id,
                          source_type, source_id, target_type, target_id,
                          confidence, evidence_ptrs_json, reason_json, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(source_entity_id, predicate_id, target_entity_id, source) DO UPDATE SET
                         confidence=excluded.confidence,
                         evidence_ptrs_json=excluded.evidence_ptrs_json,
                         reason_json=excluded.reason_json,
                         source_type=excluded.source_type,
                         source_id=excluded.source_id,
                         target_type=excluded.target_type,
                         target_id=excluded.target_id,
                         updated_at=CURRENT_TIMESTAMP""",
                    (
                        source_entity_id,
                        predicate_id,
                        target_entity_id,
                        source_type_norm,
                        source_id_norm,
                        target_type_norm,
                        target_id_norm,
                        confidence,
                        json.dumps(evidence_json.get("evidence_ptrs", []), ensure_ascii=False),
                        json.dumps(reason, ensure_ascii=False),
                        str(evidence_json.get("source", "") or "legacy_kg_migration"),
                    ),
                )

        self.conn.commit()
        self._set_system_meta("ontology_core_v2_migrated", "1")

    @staticmethod
    def _paper_entity_id(source_id: str) -> str:
        token = str(source_id or "").strip()
        if token.startswith("paper:"):
            return token
        if token.startswith("paper_"):
            token = token[len("paper_"):]
        return f"paper:{token}"

    @staticmethod
    def _note_entity_id(source_id: str) -> str:
        token = str(source_id or "").strip()
        if token.startswith("note:"):
            return token
        return f"note:{token}"

    def _ensure_predicate(self, predicate_id: str) -> str:
        token = str(predicate_id or "").strip() or "related_to"
        if not self._table_exists("ontology_predicates"):
            return token if token in CORE_PREDICATES else "related_to"
        row = self.conn.execute(
            "SELECT predicate_id, status FROM ontology_predicates WHERE predicate_id = ?",
            (token,),
        ).fetchone()
        if row:
            status = str(row["status"] or "approved_ext")
            if status == "deprecated":
                return "related_to"
            return token
        # Unknown predicates are not auto-promoted to active semantics.
        return "related_to"

    def _get_predicate_semantics(self, predicate_id: str) -> dict[str, Any]:
        if not self._table_exists("ontology_predicates"):
            return {}
        row = self.conn.execute(
            "SELECT * FROM ontology_predicates WHERE predicate_id = ?",
            (str(predicate_id or "").strip(),),
        ).fetchone()
        return dict(row) if row else {}

    def _validate_predicate_usage(
        self,
        predicate_id: str,
        source_type: str,
        target_type: str,
    ) -> dict[str, Any]:
        semantics = self._get_predicate_semantics(predicate_id)
        if not semantics:
            return {}
        issues: list[str] = []
        expected_source = str(semantics.get("domain_source_type", "") or "").strip().lower()
        expected_target = str(semantics.get("range_target_type", "") or "").strip().lower()
        if expected_source and expected_source != str(source_type or "").strip().lower():
            issues.append(f"domain mismatch: expected {expected_source}, got {source_type}")
        if expected_target and expected_target != str(target_type or "").strip().lower():
            issues.append(f"range mismatch: expected {expected_target}, got {target_type}")
        if not issues:
            return {}
        return {
            "predicate_id": predicate_id,
            "issues": issues,
            "domain_source_type": expected_source,
            "range_target_type": expected_target,
        }

    def validate_minimum_predicate_usage(
        self,
        predicate_id: str,
        source_type: str,
        target_type: str,
    ) -> dict[str, Any]:
        token = str(predicate_id or "").strip()
        if token not in MINIMUM_FORMAL_SEMANTICS_PREDICATES:
            return {}
        validation = self._validate_predicate_usage(token, source_type, target_type)
        if not validation:
            return {}
        return {
            **validation,
            "enforcement": "minimum_domain_range",
            "source_type": str(source_type or "").strip().lower(),
            "target_type": str(target_type or "").strip().lower(),
        }

    def _ensure_entity_ref(self, entity_type: str, entity_id: str) -> tuple[str, str, str]:
        et = str(entity_type or "").strip().lower()
        eid = str(entity_id or "").strip()
        if not eid:
            return "", "", ""

        if et == "paper":
            canonical = self._paper_entity_id(eid)
            paper_row = None
            if self._table_exists("papers"):
                paper_row = self.conn.execute(
                    "SELECT * FROM papers WHERE arxiv_id = ?",
                    (eid.replace("paper:", "").replace("paper_", ""),),
                ).fetchone()
            title = str((dict(paper_row) if paper_row else {}).get("title", "")).strip() or canonical
            self._upsert_entity(
                canonical,
                "paper",
                title,
                description=f"{(dict(paper_row) if paper_row else {}).get('authors', '')} ({(dict(paper_row) if paper_row else {}).get('year', '')})".strip(),
                properties_json=json.dumps(
                    {
                        "arxiv_id": eid.replace("paper:", "").replace("paper_", ""),
                        "authors": (dict(paper_row) if paper_row else {}).get("authors", ""),
                        "year": (dict(paper_row) if paper_row else {}).get("year", 0),
                        "field": (dict(paper_row) if paper_row else {}).get("field", ""),
                    },
                    ensure_ascii=False,
                ),
                emit_create_event=True,
            )
            return canonical, "paper", eid.replace("paper:", "").replace("paper_", "")

        if et == "note":
            canonical = self._note_entity_id(eid)
            note_row = None
            if self._table_exists("notes"):
                note_row = self.conn.execute("SELECT * FROM notes WHERE id = ?", (eid,)).fetchone()
            title = str((dict(note_row) if note_row else {}).get("title", "")).strip() or canonical
            self._upsert_entity(
                canonical,
                "note",
                title,
                description=str((dict(note_row) if note_row else {}).get("file_path", "")),
                properties_json=json.dumps(
                    {
                        "note_id": eid,
                        "source_type": str((dict(note_row) if note_row else {}).get("source_type", "")),
                    },
                    ensure_ascii=False,
                ),
                emit_create_event=True,
            )
            return canonical, "note", eid

        normalized_type = et if et in {"concept", "claim", "person", "organization", "event", "note"} else "event"
        if self._table_exists("ontology_entities"):
            existing = self.conn.execute(
                "SELECT entity_id FROM ontology_entities WHERE entity_id = ?",
                (eid,),
            ).fetchone()
            if existing:
                return eid, et or normalized_type, eid
        self._upsert_entity(eid, normalized_type, eid, emit_create_event=True)
        return eid, et or normalized_type, eid

    def _legacy_relation_alias(self, predicate_id: str, source_type: str, target_type: str) -> str:
        if predicate_id == "mentions" and source_type == "note" and target_type == "concept":
            return "note_mentions_concept"
        if predicate_id == "uses" and source_type == "paper" and target_type == "concept":
            return "paper_uses_concept"
        if source_type == "concept" and target_type == "concept":
            return "concept_related_to"
        return predicate_id

    def _format_relation_row(self, row: Any) -> dict[str, Any]:
        item = dict(row)
        if "predicate_id" not in item:
            legacy_relation = str(item.get("relation", "related_to"))
            legacy_evidence_text = str(item.get("evidence_text", "") or "")
            predicate_id = _normalize_predicate(legacy_relation, legacy_evidence_text)
            parsed_legacy = _safe_json_dict(legacy_evidence_text)
            item["predicate_id"] = predicate_id
            item["relation"] = legacy_relation
            item["id"] = item.get("id")
            item["evidence_json"] = {
                "source": str(parsed_legacy.get("source", "")),
                "relation_norm": predicate_id,
                "evidence_ptrs": parsed_legacy.get("evidence_ptrs", [])
                if isinstance(parsed_legacy.get("evidence_ptrs"), list)
                else [],
                "reason": {"legacy_relation": legacy_relation},
            }
            return item

        reason_json = _safe_json_dict(item.get("reason_json"))
        evidence_ptrs = []
        try:
            evidence_ptrs = json.loads(item.get("evidence_ptrs_json") or "[]")
        except Exception:
            evidence_ptrs = []

        relation_alias = str(reason_json.get("legacy_relation", "")).strip()
        if not relation_alias:
            relation_alias = self._legacy_relation_alias(
                str(item.get("predicate_id", "related_to")),
                str(item.get("source_type", "")),
                str(item.get("target_type", "")),
            )

        evidence_json = {
            "source": str(item.get("source", "")),
            "relation_norm": str(item.get("predicate_id", "related_to")),
            "evidence_ptrs": evidence_ptrs,
            "reason": reason_json,
        }
        item["id"] = item.get("relation_id")
        item["relation"] = relation_alias
        item["evidence_json"] = evidence_json
        legacy_evidence_text = str(reason_json.get("legacy_evidence_text", "")).strip()
        if legacy_evidence_text:
            item["evidence_text"] = legacy_evidence_text
        else:
            item["evidence_text"] = json.dumps(evidence_json, ensure_ascii=False)
        return item

    def _attach_predicate_semantics(self, item: dict[str, Any]) -> dict[str, Any]:
        predicate_id = str(item.get("predicate_id") or "").strip()
        if not predicate_id:
            return item
        semantics = self._get_predicate_semantics(predicate_id)
        if not semantics:
            return item
        item["predicate_semantics"] = semantics
        for key in ("domain_source_type", "range_target_type", "is_transitive", "is_symmetric", "is_antisymmetric"):
            if key in semantics and key not in item:
                item[key] = semantics[key]
        validation = self._runtime_validate_relation(item, semantics=semantics)
        if validation:
            item["predicate_validation"] = validation
        return item

    def _runtime_validate_relation(
        self,
        item: dict[str, Any],
        *,
        semantics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        predicate_id = str(item.get("predicate_id") or "").strip()
        if not predicate_id:
            return {}
        semantics = semantics or self._get_predicate_semantics(predicate_id)
        if not semantics:
            return {}

        source_type = str(item.get("source_type") or "").strip().lower()
        target_type = str(item.get("target_type") or "").strip().lower()
        source_entity_id = str(item.get("source_entity_id") or item.get("source_id") or "").strip()
        target_entity_id = str(item.get("target_entity_id") or item.get("target_id") or "").strip()
        relation_id = str(item.get("relation_id") or item.get("id") or "").strip()

        issues: list[str] = []
        base_validation = self._validate_predicate_usage(predicate_id, source_type, target_type)
        issues.extend(base_validation.get("issues", []))

        if source_entity_id and target_entity_id and source_entity_id != target_entity_id and self._table_exists("ontology_relations"):
            reverse_row = self.conn.execute(
                """
                SELECT relation_id
                FROM ontology_relations
                WHERE source_entity_id = ? AND target_entity_id = ? AND predicate_id = ?
                LIMIT 1
                """,
                (target_entity_id, source_entity_id, predicate_id),
            ).fetchone()
            reverse_relation_id = str((dict(reverse_row) if reverse_row else {}).get("relation_id", "")).strip() if reverse_row else ""
            reverse_exists = bool(reverse_relation_id and reverse_relation_id != relation_id)
            if bool(semantics.get("is_antisymmetric")) and reverse_exists:
                issues.append("antisymmetric misuse: reverse relation exists")
            if bool(semantics.get("is_symmetric")) and not reverse_exists:
                issues.append("symmetric misuse: reverse relation missing")

        if not issues:
            return {}
        return {
            "predicate_id": predicate_id,
            "issues": issues,
            "domain_source_type": str(semantics.get("domain_source_type", "") or ""),
            "range_target_type": str(semantics.get("range_target_type", "") or ""),
        }

    def _record_invalid_relation_attempt(
        self,
        *,
        source_entity_id: str,
        predicate_id: str,
        target_entity_id: str,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        confidence: float,
        evidence_ptrs: list[dict[str, Any]],
        legacy_relation: str,
        legacy_evidence_text: str,
        evidence_json: dict[str, Any],
        validation: dict[str, Any],
    ) -> int:
        if not self._table_exists("ontology_pending"):
            return 0
        reason_json: dict[str, Any] = {
            "kind": "relation",
            "blocked_by": "predicate_semantics",
            "semantic_validation": validation,
            "source_type": source_type,
            "source_id": source_id,
            "target_type": target_type,
            "target_id": target_id,
            "legacy_relation": legacy_relation,
            "legacy_evidence_text": legacy_evidence_text,
            "write_path": "ontology_store.add_relation",
        }
        if evidence_json:
            reason_json["legacy_evidence"] = evidence_json
        note_id = source_id if source_type == "note" else str(evidence_json.get("note_id", "") or "")
        source_url = str(evidence_json.get("source_url", "") or "")
        return self.add_ontology_pending(
            pending_type="relation",
            run_id="ontology_store.add_relation",
            topic_slug=str(evidence_json.get("topic_slug", "") or ""),
            note_id=note_id,
            source_url=source_url,
            source_entity_id=source_entity_id,
            predicate_id=predicate_id,
            target_entity_id=target_entity_id,
            confidence=confidence,
            evidence_ptrs=evidence_ptrs,
            reason=reason_json,
            status="rejected",
        )

    def add_relation(
        self,
        source_type: str,
        source_id: str,
        relation: str,
        target_type: str,
        target_id: str,
        evidence_text: str = "",
        confidence: float = 0.5,
    ) -> None:
        source_entity_id, source_type_norm, source_id_norm = self._ensure_entity_ref(source_type, source_id)
        target_entity_id, target_type_norm, target_id_norm = self._ensure_entity_ref(target_type, target_id)
        if not source_id_norm or not target_id_norm:
            return

        predicate_raw = _normalize_predicate(relation, evidence_text=evidence_text)
        predicate_id = self._ensure_predicate(predicate_raw)
        evidence_json = _safe_json_dict(evidence_text)
        evidence_ptrs = evidence_json.get("evidence_ptrs", []) if isinstance(evidence_json.get("evidence_ptrs"), list) else []
        reason_json = {
            "legacy_relation": str(relation or ""),
            "legacy_evidence_text": str(evidence_text or ""),
        }
        if evidence_json:
            reason_json["legacy_evidence"] = evidence_json
        validation = self.validate_minimum_predicate_usage(predicate_id, source_type_norm, target_type_norm)
        if validation:
            reason_json["semantic_validation"] = validation
            log.warning(
                "Predicate semantics rejected for %s (%s -> %s): %s",
                predicate_id,
                source_type_norm,
                target_type_norm,
                "; ".join(validation.get("issues", [])),
            )
            self._record_invalid_relation_attempt(
                source_entity_id=source_entity_id or source_id_norm,
                predicate_id=predicate_id,
                target_entity_id=target_entity_id or target_id_norm,
                source_type=source_type_norm,
                source_id=source_id_norm,
                target_type=target_type_norm,
                target_id=target_id_norm,
                confidence=float(confidence),
                evidence_ptrs=evidence_ptrs,
                legacy_relation=str(relation or ""),
                legacy_evidence_text=str(evidence_text or ""),
                evidence_json=evidence_json,
                validation=validation,
            )
            return

        relation_alias = self._legacy_relation_alias(predicate_id, source_type_norm, target_type_norm)
        source_value = str(evidence_json.get("source", "")).strip()
        if not source_value:
            source_value = "web" if source_type_norm in {"concept", "note"} else "system"

        if self._table_exists("ontology_relations"):
            self.conn.execute(
                """INSERT INTO ontology_relations
                     (source_entity_id, predicate_id, target_entity_id,
                      source_type, source_id, target_type, target_id,
                      confidence, evidence_ptrs_json, reason_json, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_entity_id, predicate_id, target_entity_id, source) DO UPDATE SET
                     confidence=excluded.confidence,
                     evidence_ptrs_json=excluded.evidence_ptrs_json,
                     reason_json=excluded.reason_json,
                     source_type=excluded.source_type,
                     source_id=excluded.source_id,
                     target_type=excluded.target_type,
                     target_id=excluded.target_id,
                     updated_at=CURRENT_TIMESTAMP""",
                (
                    source_entity_id or source_id_norm,
                    predicate_id,
                    target_entity_id or target_id_norm,
                    source_type_norm,
                    source_id_norm,
                    target_type_norm,
                    target_id_norm,
                    float(confidence),
                    json.dumps(evidence_ptrs, ensure_ascii=False),
                    json.dumps(reason_json, ensure_ascii=False),
                    source_value,
                ),
            )

        self.conn.commit()

        if self.event_store:
            event = OntologyEvent(
                event_id=f"evt_{uuid4().hex}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="relation_added",
                entity_id=f"{source_entity_id or source_id_norm}_{target_entity_id or target_id_norm}",
                entity_type="relation",
                actor="system",
                data={
                    "source_id": source_id_norm,
                    "source_entity_id": source_entity_id or source_id_norm,
                    "source_type": source_type_norm,
                    "relation": relation_alias,
                    "predicate_id": predicate_id,
                    "target_id": target_id_norm,
                    "target_entity_id": target_entity_id or target_id_norm,
                    "target_type": target_type_norm,
                    "confidence": confidence,
                },
                policy_class="P2",
            )
            try:
                self.event_store.append(event)
            except Exception as error:
                log.error(
                    "Event append failed for relation add (%s:%s -> %s:%s): %s",
                    source_type_norm,
                    source_id_norm,
                    target_type_norm,
                    target_id_norm,
                    error,
                )

    def get_relations(self, entity_type: str, entity_id: str) -> list[dict]:
        et = str(entity_type or "").strip()
        eid = str(entity_id or "").strip()
        if not self._table_exists("ontology_relations"):
            return []
        query = """
            SELECT * FROM ontology_relations
            WHERE (source_type=? AND source_id=?) OR (target_type=? AND target_id=?)
            ORDER BY confidence DESC, relation_id DESC
        """
        params: list[Any] = [et, eid, et, eid]
        # Entity-id fallback for paper/note canonical ids.
        if et == "paper":
            canonical = self._paper_entity_id(eid)
            query = """
                SELECT * FROM ontology_relations
                WHERE (source_type=? AND source_id=?) OR (target_type=? AND target_id=?)
                   OR (source_entity_id=?) OR (target_entity_id=?)
                ORDER BY confidence DESC, relation_id DESC
            """
            params.extend([canonical, canonical])
        elif et == "note":
            canonical = self._note_entity_id(eid)
            query = """
                SELECT * FROM ontology_relations
                WHERE (source_type=? AND source_id=?) OR (target_type=? AND target_id=?)
                   OR (source_entity_id=?) OR (target_entity_id=?)
                ORDER BY confidence DESC, relation_id DESC
            """
            params.extend([canonical, canonical])
        rows = self.conn.execute(query, params).fetchall()
        return [self._attach_predicate_semantics(self._format_relation_row(row)) for row in rows]

    def list_legacy_kg_relations(
        self,
        relation: str | None = None,
        source_type: str | None = None,
        target_type: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
        limit: int = 2000,
    ) -> list[dict]:
        if not self._table_exists("kg_relations"):
            return []
        query = "SELECT * FROM kg_relations WHERE 1=1"
        params: list[Any] = []
        if source_type:
            query += " AND source_type = ?"
            params.append(str(source_type))
        if target_type:
            query += " AND target_type = ?"
            params.append(str(target_type))
        if source_id:
            query += " AND source_id = ?"
            params.append(str(source_id))
        if target_id:
            query += " AND target_id = ?"
            params.append(str(target_id))
        query += " ORDER BY confidence DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        items = [self._attach_predicate_semantics(self._format_relation_row(row)) for row in rows]
        if not relation:
            return items
        relation_token = str(relation).strip()
        return [
            item
            for item in items
            if item.get("relation") == relation_token or item.get("predicate_id") == relation_token
        ]

    def list_kg_relations(
        self,
        relation: str | None = None,
        source_type: str | None = None,
        target_type: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
        limit: int = 2000,
    ) -> list[dict]:
        if not self._table_exists("ontology_relations"):
            return []
        query = "SELECT * FROM ontology_relations WHERE 1=1"
        params: list[Any] = []
        if source_type:
            query += " AND source_type = ?"
            params.append(str(source_type))
        if target_type:
            query += " AND target_type = ?"
            params.append(str(target_type))
        if source_id:
            query += " AND source_id = ?"
            params.append(str(source_id))
        if target_id:
            query += " AND target_id = ?"
            params.append(str(target_id))
        query += " ORDER BY confidence DESC, relation_id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        items = [self._attach_predicate_semantics(self._format_relation_row(row)) for row in rows]
        if not relation:
            return items
        relation_token = str(relation).strip()
        return [item for item in items if item.get("relation") == relation_token or item.get("predicate_id") == relation_token]

    def list_relations(
        self,
        limit: int = 2000,
        updated_after: str | None = None,
        relation: str | None = None,
        source_type: str | None = None,
        target_type: str | None = None,
        predicate_id: str | None = None,
    ) -> list[dict]:
        if not self._table_exists("ontology_relations"):
            return []
        query = "SELECT * FROM ontology_relations WHERE 1=1"
        params: list[Any] = []
        if updated_after:
            query += " AND created_at > ?"
            params.append(str(updated_after))
        if source_type:
            query += " AND source_type = ?"
            params.append(str(source_type))
        if target_type:
            query += " AND target_type = ?"
            params.append(str(target_type))
        if predicate_id:
            query += " AND predicate_id = ?"
            params.append(str(predicate_id))
        query += " ORDER BY created_at ASC, relation_id ASC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        items = [self._attach_predicate_semantics(self._format_relation_row(row)) for row in rows]
        if not relation:
            return items
        relation_token = str(relation).strip()
        return [item for item in items if item.get("relation") == relation_token or item.get("predicate_id") == relation_token]

    def list_predicate_validation_issues(
        self,
        *,
        limit: int = 200,
        predicate_id: str | None = None,
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        for relation in self.list_relations(limit=max(limit * 5, 500), predicate_id=predicate_id):
            validation = relation.get("predicate_validation")
            if not isinstance(validation, dict):
                continue
            validation_issues = validation.get("issues") or []
            if not validation_issues:
                continue
            issues.append(
                {
                    "relation_id": relation.get("relation_id"),
                    "predicate_id": relation.get("predicate_id"),
                    "source_entity_id": relation.get("source_entity_id"),
                    "target_entity_id": relation.get("target_entity_id"),
                    "source_name": relation.get("source_name") or relation.get("source_entity_id"),
                    "target_name": relation.get("target_name") or relation.get("target_entity_id"),
                    "issues": list(validation_issues),
                    "validation": validation,
                }
            )
            if len(issues) >= max(1, int(limit)):
                break
        if len(issues) >= max(1, int(limit)):
            return issues
        for pending in self.list_ontology_pending(
            pending_type="relation",
            status=None,
            limit=max(limit * 5, 500),
        ):
            if predicate_id and str(pending.get("predicate_id") or "").strip() != str(predicate_id).strip():
                continue
            reason_json = pending.get("reason_json") if isinstance(pending.get("reason_json"), dict) else {}
            validation = reason_json.get("semantic_validation") if isinstance(reason_json.get("semantic_validation"), dict) else {}
            validation_issues = validation.get("issues") or []
            if not validation_issues:
                continue
            issues.append(
                {
                    "relation_id": None,
                    "pending_id": pending.get("id"),
                    "status": pending.get("status"),
                    "predicate_id": pending.get("predicate_id"),
                    "source_entity_id": pending.get("source_entity_id"),
                    "target_entity_id": pending.get("target_entity_id"),
                    "source_name": reason_json.get("source_id") or pending.get("source_entity_id"),
                    "target_name": reason_json.get("target_id") or pending.get("target_entity_id"),
                    "issues": list(validation_issues),
                    "validation": validation,
                }
            )
            if len(issues) >= max(1, int(limit)):
                break
        return issues

    def upsert_ontology_entity(
        self,
        entity_id: str,
        entity_type: str,
        canonical_name: str,
        description: str = "",
        properties: dict[str, Any] | None = None,
        confidence: float = 1.0,
        source: str = "system",
    ) -> None:
        existing = self.get_ontology_entity(entity_id)
        is_update = existing is not None
        properties_json = json.dumps(properties or {}, ensure_ascii=False)
        self.conn.execute(
            """INSERT INTO ontology_entities
               (entity_id, entity_type, canonical_name, description, properties_json, confidence, source)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(entity_id) DO UPDATE SET
                 entity_type=excluded.entity_type,
                 canonical_name=excluded.canonical_name,
                 description=excluded.description,
                 properties_json=excluded.properties_json,
                 confidence=excluded.confidence,
                 source=excluded.source,
                 updated_at=CURRENT_TIMESTAMP""",
            (
                str(entity_id),
                str(entity_type),
                str(canonical_name),
                str(description or ""),
                properties_json,
                float(confidence),
                str(source),
            ),
        )
        self.conn.commit()

        if not self.event_store:
            return
        event = OntologyEvent(
            event_id=f"evt_{uuid4().hex}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="entity_updated" if is_update else "entity_created",
            entity_id=str(entity_id),
            entity_type=str(entity_type),
            actor=str(source),
            data={
                "canonical_name": canonical_name,
                "description": description,
                "properties": properties or {},
                "confidence": confidence,
                "entity_type": entity_type,
            },
            policy_class="P2",
        )
        try:
            self.event_store.append(event)
        except Exception as error:
            log.error("Event append failed for entity upsert (%s): %s", entity_id, error)

    def get_ontology_entity(self, entity_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM ontology_entities WHERE entity_id = ?",
            (str(entity_id),),
        ).fetchone()
        return self._decode_entity_row(row)

    def list_ontology_entities(
        self,
        entity_type: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if entity_type:
            rows = self.conn.execute(
                "SELECT * FROM ontology_entities WHERE entity_type = ? ORDER BY canonical_name LIMIT ?",
                (str(entity_type), max(1, int(limit))),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM ontology_entities ORDER BY entity_type, canonical_name LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [item for item in (self._decode_entity_row(row) for row in rows) if item]

    def add_entity_alias(self, alias: str, entity_id: str) -> None:
        alias_token = str(alias).strip()
        entity_token = str(entity_id).strip()
        if not alias_token or not entity_token:
            return
        previous = self.conn.execute(
            "SELECT entity_id FROM entity_aliases WHERE alias = ?",
            (alias_token,),
        ).fetchone()
        previous_entity_id = str(previous["entity_id"]) if previous else None
        self.conn.execute(
            "INSERT OR REPLACE INTO entity_aliases (alias, entity_id) VALUES (?, ?)",
            (alias_token, entity_token),
        )
        self.conn.commit()

        if not self.event_store:
            return
        event_type = "alias_added"
        if previous_entity_id and previous_entity_id != entity_token:
            event_type = "alias_reassigned"
        elif previous_entity_id == entity_token:
            event_type = "alias_updated"
        event = OntologyEvent(
            event_id=f"evt_{uuid4().hex}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            entity_id=entity_token,
            entity_type="alias",
            actor="system",
            data={
                "alias": alias_token,
                "entity_id": entity_token,
                "previous_entity_id": previous_entity_id,
            },
            policy_class="P2",
        )
        try:
            self.event_store.append(event)
        except Exception as error:
            log.error("Event append failed for alias update (%s -> %s): %s", alias_token, entity_token, error)

    def migrate_concepts_to_entities(self) -> int:
        concepts = self.conn.execute("SELECT * FROM concepts").fetchall()
        count = 0
        for concept in concepts:
            concept_id = concept["id"]
            canonical_name = concept["canonical_name"]
            description = concept["description"] if "description" in concept.keys() else ""
            self.upsert_ontology_entity(
                entity_id=concept_id,
                entity_type="concept",
                canonical_name=canonical_name,
                description=description or "",
                source="migration_from_concepts",
            )
            alias_rows = self.conn.execute(
                "SELECT alias FROM concept_aliases WHERE concept_id = ?",
                (concept_id,),
            ).fetchall()
            for row in alias_rows:
                alias = str(row[0] if row and row[0] else "").strip()
                if alias:
                    self.add_entity_alias(alias, concept_id)
            count += 1
        return count

    def create_concepts_view(self) -> None:
        self.conn.execute("DROP VIEW IF EXISTS concepts_view")
        self.conn.execute(
            """
            CREATE VIEW IF NOT EXISTS concepts_view AS
            SELECT
                entity_id AS id,
                canonical_name,
                description,
                created_at
            FROM ontology_entities
            WHERE entity_type = 'concept'
            """
        )
        self.conn.commit()

    def sync_paper_entities(self) -> int:
        papers = self.conn.execute("SELECT * FROM papers").fetchall()
        count = 0
        for paper in papers:
            arxiv_id = paper["arxiv_id"]
            title = paper["title"]
            authors = paper["authors"] if "authors" in paper.keys() else ""
            year = paper["year"] if "year" in paper.keys() else 0
            field = paper["field"] if "field" in paper.keys() else ""
            properties = {
                "arxiv_id": arxiv_id,
                "authors": authors,
                "year": year,
                "field": field,
            }
            self.upsert_ontology_entity(
                entity_id=self._paper_entity_id(arxiv_id),
                entity_type="paper",
                canonical_name=title,
                description=f"{authors} ({year})".strip(),
                properties=properties,
                source="papers_table",
            )
            count += 1
        return count

    def get_entity_aliases(self, entity_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT alias FROM entity_aliases WHERE entity_id = ?",
            (str(entity_id),),
        ).fetchall()
        return [str(row[0]) for row in rows if row and row[0]]

    def resolve_entity(self, name_or_alias: str, entity_type: str | None = None) -> dict[str, Any] | None:
        query = "SELECT * FROM ontology_entities WHERE (canonical_name = ? OR entity_id = ?)"
        params: list[Any] = [str(name_or_alias), str(name_or_alias)]
        if entity_type:
            query += " AND entity_type = ?"
            params.append(str(entity_type))
        row = self.conn.execute(query, params).fetchone()
        if row:
            return self._decode_entity_row(row)

        alias_query = """
            SELECT e.* FROM entity_aliases a
            JOIN ontology_entities e ON a.entity_id = e.entity_id
            WHERE a.alias = ?
        """
        alias_params: list[Any] = [str(name_or_alias)]
        if entity_type:
            alias_query += " AND e.entity_type = ?"
            alias_params.append(str(entity_type))
        alias_row = self.conn.execute(alias_query, alias_params).fetchone()
        return self._decode_entity_row(alias_row)

    def delete_ontology_entity(self, entity_id: str) -> None:
        entity_token = str(entity_id).strip()
        if not entity_token:
            return
        existing = self.get_ontology_entity(entity_token)
        aliases = self.get_entity_aliases(entity_token)
        claim_count_row = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM ontology_claims WHERE subject_entity_id = ? OR object_entity_id = ?",
            (entity_token, entity_token),
        ).fetchone()
        removed_claim_count = int(claim_count_row["cnt"]) if claim_count_row else 0

        self.conn.execute("DELETE FROM entity_aliases WHERE entity_id = ?", (entity_token,))
        self.conn.execute(
            "DELETE FROM ontology_claims WHERE subject_entity_id = ? OR object_entity_id = ?",
            (entity_token, entity_token),
        )
        self.conn.execute(
            "DELETE FROM ontology_relations WHERE source_entity_id = ? OR target_entity_id = ?",
            (entity_token, entity_token),
        )
        self.conn.execute("DELETE FROM ontology_entities WHERE entity_id = ?", (entity_token,))
        self.conn.commit()

        if not self.event_store or not existing:
            return
        event = OntologyEvent(
            event_id=f"evt_{uuid4().hex}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="entity_deleted",
            entity_id=entity_token,
            entity_type=str(existing.get("entity_type", "entity")),
            actor="system",
            data={
                "canonical_name": str(existing.get("canonical_name", "")),
                "description": str(existing.get("description", "")),
                "aliases": aliases,
                "removed_claim_count": removed_claim_count,
            },
            policy_class="P2",
        )
        try:
            self.event_store.append(event)
        except Exception as error:
            log.error("Event append failed for entity delete (%s): %s", entity_token, error)

    def upsert_concept(self, concept_id: str, canonical_name: str, description: str = "") -> None:
        self.upsert_ontology_entity(
            entity_id=str(concept_id),
            entity_type="concept",
            canonical_name=str(canonical_name),
            description=str(description or ""),
            source="legacy_concept_api",
        )

    def get_concept(self, concept_id: str) -> dict[str, Any] | None:
        entity = self.get_ontology_entity(concept_id)
        if not entity or str(entity.get("entity_type") or "") != "concept":
            return None
        return {
            "id": entity["entity_id"],
            "canonical_name": entity.get("canonical_name", ""),
            "description": entity.get("description", ""),
            "created_at": entity.get("created_at"),
        }

    def legacy_lookup_concept_by_name(self, name: str) -> dict[str, Any] | None:
        entity = self.resolve_entity(name, entity_type="concept")
        if not entity or str(entity.get("canonical_name") or "") != str(name):
            return None
        return {
            "id": entity["entity_id"],
            "canonical_name": entity.get("canonical_name", ""),
            "description": entity.get("description", ""),
            "created_at": entity.get("created_at"),
        }

    def list_concepts(self, limit: int = 500) -> list[dict[str, Any]]:
        return [
            {
                "id": item["entity_id"],
                "canonical_name": item.get("canonical_name", ""),
                "description": item.get("description", ""),
                "created_at": item.get("created_at"),
            }
            for item in self.list_ontology_entities(entity_type="concept", limit=limit)
        ]

    def add_alias(self, alias: str, concept_id: str) -> None:
        self.add_entity_alias(alias, concept_id)

    def get_aliases(self, concept_id: str) -> list[str]:
        return self.get_entity_aliases(concept_id)

    def resolve_concept(self, name_or_alias: str) -> str | None:
        entity = self.resolve_entity(name_or_alias, entity_type="concept")
        if not entity:
            return None
        return str(entity.get("canonical_name") or "")

    def delete_concept(self, concept_id: str) -> None:
        self.delete_ontology_entity(concept_id)

    def list_ontology_claims(
        self,
        limit: int = 2000,
        updated_after: str | None = None,
        subject_entity_id: str | None = None,
        predicate: str | None = None,
        object_entity_id: str | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM ontology_claims WHERE 1=1"
        params: list[Any] = []
        if updated_after:
            query += " AND created_at > ?"
            params.append(str(updated_after))
        if subject_entity_id:
            query += " AND subject_entity_id = ?"
            params.append(str(subject_entity_id))
        if predicate:
            query += " AND predicate = ?"
            params.append(str(predicate))
        if object_entity_id:
            query += " AND object_entity_id = ?"
            params.append(str(object_entity_id))
        query += " ORDER BY created_at ASC, claim_id ASC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()

        result: list[dict] = []
        for row in rows:
            item = dict(row)
            raw = item.get("evidence_ptrs_json")
            try:
                item["evidence_ptrs"] = json.loads(raw) if raw else []
            except Exception:
                item["evidence_ptrs"] = []
            result.append(item)
        return result

    def list_ontology_events(
        self,
        limit: int = 2000,
        updated_after: str | None = None,
        entity_id: str | None = None,
        event_type: str | None = None,
    ) -> list[dict]:
        table_row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ontology_events'"
        ).fetchone()
        if not table_row:
            return []

        query = "SELECT * FROM ontology_events WHERE 1=1"
        params: list[Any] = []
        if updated_after:
            query += " AND created_at > ?"
            params.append(str(updated_after))
        if entity_id:
            query += " AND entity_id = ?"
            params.append(str(entity_id))
        if event_type:
            query += " AND event_type = ?"
            params.append(str(event_type))
        query += " ORDER BY created_at ASC, id ASC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_related_concepts(self, concept_id: str) -> list[dict]:
        if not self._table_exists("ontology_relations"):
            return []
        rows = self.conn.execute(
            """SELECT e.*, r.confidence, r.predicate_id
               FROM ontology_relations r
               JOIN ontology_entities e ON (
                 CASE WHEN r.source_entity_id = ? THEN r.target_entity_id ELSE r.source_entity_id END
               ) = e.entity_id
               WHERE e.entity_type='concept'
                 AND (
                   (r.source_type='concept' AND r.source_id=?)
                   OR (r.target_type='concept' AND r.target_id=?)
                   OR (r.source_entity_id=?)
                   OR (r.target_entity_id=?)
                 )
               ORDER BY r.confidence DESC""",
            (concept_id, concept_id, concept_id, concept_id, concept_id),
        ).fetchall()
        result: list[dict] = []
        for row in rows:
            item = dict(row)
            item.setdefault("id", item.get("entity_id", ""))
            result.append(item)
        return result

    def count_relations(self) -> int:
        if not self._table_exists("ontology_relations"):
            return 0
        return self.conn.execute("SELECT COUNT(*) FROM ontology_relations").fetchone()[0]

    def count_concepts(self) -> int:
        if not self._table_exists("ontology_entities"):
            return 0
        return self.conn.execute(
            "SELECT COUNT(*) FROM ontology_entities WHERE entity_type = 'concept'"
        ).fetchone()[0]

    def get_kg_stats(self) -> dict[str, Any]:
        concept_count = self.count_concepts()
        relation_count = self.count_relations()
        alias_count = self.conn.execute("SELECT COUNT(*) FROM entity_aliases").fetchone()[0] if self._table_exists("entity_aliases") else 0
        paper_count = self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] if self._table_exists("papers") else 0

        if not self._table_exists("ontology_relations"):
            isolated_concepts = concept_count
            relation_types: dict[str, int] = {}
        else:
            isolated_concepts = self.conn.execute(
                """SELECT COUNT(*) FROM ontology_entities e
                   WHERE e.entity_type='concept'
                     AND NOT EXISTS (
                     SELECT 1 FROM ontology_relations r
                     WHERE r.source_entity_id=e.entity_id OR r.target_entity_id=e.entity_id
                   )"""
            ).fetchone()[0]
            rel_type_counts = self.conn.execute(
                "SELECT predicate_id, COUNT(*) as cnt FROM ontology_relations GROUP BY predicate_id ORDER BY cnt DESC"
            ).fetchall()
            relation_types = {r["predicate_id"]: r["cnt"] for r in rel_type_counts}

        return {
            "concepts": concept_count,
            "aliases": alias_count,
            "papers": paper_count,
            "relations": relation_count,
            "isolated_concepts": isolated_concepts,
            "relation_types": relation_types,
        }

    def upsert_predicate(
        self,
        predicate_id: str,
        parent_predicate_id: str | None = None,
        status: str = "approved_ext",
        description: str = "",
        source: str = "system",
        domain_source_type: str = "",
        range_target_type: str = "",
        is_transitive: bool = False,
        is_symmetric: bool = False,
        is_antisymmetric: bool = False,
    ) -> None:
        token = str(predicate_id or "").strip()
        if not token or not self._table_exists("ontology_predicates"):
            return
        status_value = str(status or "approved_ext").strip()
        if status_value not in {"core", "approved_ext", "deprecated"}:
            status_value = "approved_ext"
        parent_value = str(parent_predicate_id).strip() if parent_predicate_id else None
        columns = {
            "predicate_id": token,
            "parent_predicate_id": parent_value,
            "status": status_value,
            "description": str(description or ""),
            "source": str(source or "system"),
        }
        predicate_cols = self._table_columns("ontology_predicates")
        if "domain_source_type" in predicate_cols:
            columns["domain_source_type"] = str(domain_source_type or "")
        if "range_target_type" in predicate_cols:
            columns["range_target_type"] = str(range_target_type or "")
        if "is_transitive" in predicate_cols:
            columns["is_transitive"] = 1 if is_transitive else 0
        if "is_symmetric" in predicate_cols:
            columns["is_symmetric"] = 1 if is_symmetric else 0
        if "is_antisymmetric" in predicate_cols:
            columns["is_antisymmetric"] = 1 if is_antisymmetric else 0

        update_parts = [
            "parent_predicate_id=excluded.parent_predicate_id",
            "status=excluded.status",
            "description=excluded.description",
            "source=excluded.source",
        ]
        for key in ("domain_source_type", "range_target_type", "is_transitive", "is_symmetric", "is_antisymmetric"):
            if key in columns:
                update_parts.append(f"{key}=excluded.{key}")
        update_parts.append("updated_at=CURRENT_TIMESTAMP")

        cols_sql = ", ".join(columns.keys())
        placeholders = ", ".join(["?"] * len(columns))
        self.conn.execute(
            f"""INSERT INTO ontology_predicates ({cols_sql})
                VALUES ({placeholders})
                ON CONFLICT(predicate_id) DO UPDATE SET
                  {", ".join(update_parts)}""",
            tuple(columns.values()),
        )
        self.conn.commit()
        self._table_columns_cache.pop("ontology_predicates", None)

    def get_predicate(self, predicate_id: str) -> dict[str, Any] | None:
        if not self._table_exists("ontology_predicates"):
            return None
        row = self.conn.execute(
            "SELECT * FROM ontology_predicates WHERE predicate_id = ?",
            (str(predicate_id or "").strip(),),
        ).fetchone()
        return dict(row) if row else None

    def list_predicates(self, status: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        if not self._table_exists("ontology_predicates"):
            return []
        query = "SELECT * FROM ontology_predicates WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(str(status))
        query += " ORDER BY predicate_id ASC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def add_ontology_pending(
        self,
        pending_type: str,
        run_id: str,
        topic_slug: str = "",
        note_id: str = "",
        source_url: str = "",
        source_entity_id: str = "",
        predicate_id: str = "",
        target_entity_id: str = "",
        confidence: float = 0.0,
        evidence_ptrs: list[dict] | None = None,
        reason: dict | None = None,
        status: str = "pending",
    ) -> int:
        if not self._table_exists("ontology_pending"):
            return 0
        pending_value = str(pending_type or "").strip()
        if pending_value not in {"concept", "relation", "claim", "predicate_ext"}:
            raise ValueError(f"unsupported pending_type: {pending_value}")
        status_value = str(status or "pending").strip()
        if status_value not in {"pending", "approved", "rejected"}:
            status_value = "pending"
        cursor = self.conn.execute(
            """INSERT INTO ontology_pending
                 (pending_type, run_id, topic_slug, note_id, source_url, source_entity_id, predicate_id,
                  target_entity_id, confidence, evidence_ptrs_json, reason_json, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pending_value,
                str(run_id or ""),
                str(topic_slug or ""),
                str(note_id or ""),
                str(source_url or ""),
                str(source_entity_id or ""),
                str(predicate_id or ""),
                str(target_entity_id or ""),
                float(confidence or 0.0),
                json.dumps(evidence_ptrs or [], ensure_ascii=False),
                json.dumps(reason or {}, ensure_ascii=False),
                status_value,
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def get_ontology_pending(self, pending_id: int) -> dict[str, Any] | None:
        if not self._table_exists("ontology_pending"):
            return None
        row = self.conn.execute(
            "SELECT * FROM ontology_pending WHERE id = ?",
            (int(pending_id),),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["evidence_ptrs_json"] = json.loads(item.get("evidence_ptrs_json") or "[]")
        except Exception:
            item["evidence_ptrs_json"] = []
        try:
            item["reason_json"] = json.loads(item.get("reason_json") or "{}")
        except Exception:
            item["reason_json"] = {}
        return item

    def list_ontology_pending(
        self,
        pending_type: str | None = None,
        topic_slug: str | None = None,
        status: str | None = "pending",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if not self._table_exists("ontology_pending"):
            return []
        query = "SELECT * FROM ontology_pending WHERE 1=1"
        params: list[Any] = []
        if pending_type:
            query += " AND pending_type = ?"
            params.append(str(pending_type))
        if topic_slug:
            query += " AND topic_slug = ?"
            params.append(str(topic_slug))
        if status:
            query += " AND status = ?"
            params.append(str(status))
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        return [item for item in (self.get_ontology_pending(int(row["id"])) for row in rows) if item]

    def update_ontology_pending_status(self, pending_id: int, status: str) -> bool:
        if not self._table_exists("ontology_pending"):
            return False
        status_value = str(status or "pending").strip()
        if status_value not in {"pending", "approved", "rejected"}:
            status_value = "pending"
        cursor = self.conn.execute(
            """UPDATE ontology_pending
               SET status = ?,
                   reviewed_at = CASE WHEN ? IN ('approved', 'rejected') THEN CURRENT_TIMESTAMP ELSE reviewed_at END
               WHERE id = ?""",
            (status_value, status_value, int(pending_id)),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def update_ontology_pending_reason(
        self,
        pending_id: int,
        reason: dict[str, Any] | None,
        *,
        replace: bool = False,
    ) -> bool:
        if not self._table_exists("ontology_pending"):
            return False
        current = self.get_ontology_pending(int(pending_id))
        if not current:
            return False
        payload = dict(reason or {})
        if not replace:
            merged = current.get("reason_json") if isinstance(current.get("reason_json"), dict) else {}
            payload = {**merged, **payload}
        cursor = self.conn.execute(
            "UPDATE ontology_pending SET reason_json = ? WHERE id = ?",
            (json.dumps(payload, ensure_ascii=False), int(pending_id)),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def add_web_ontology_pending(
        self,
        run_id: str,
        topic_slug: str,
        note_id: str,
        source_url: str,
        source_canonical_id: str,
        relation_norm: str,
        target_canonical_id: str,
        confidence: float,
        evidence_ptrs: list[dict] | None = None,
        reason: dict | None = None,
        status: str = "pending",
    ) -> int:
        reason_payload = dict(reason or {})
        pending_type = "concept" if relation_norm == "concept_candidate" else "relation"
        source_type = str(reason_payload.get("source_type", "concept") or "concept")
        target_type = str(reason_payload.get("target_type", "concept") or "concept")
        reason_payload.setdefault("source_type", source_type)
        reason_payload.setdefault("source_id", source_canonical_id)
        reason_payload.setdefault("target_type", target_type)
        reason_payload.setdefault("target_id", target_canonical_id)
        reason_payload.setdefault("relation_norm", relation_norm)
        reason_payload.setdefault("kind", "concept" if pending_type == "concept" else "relation")
        return self.add_ontology_pending(
            pending_type=pending_type,
            run_id=run_id,
            topic_slug=topic_slug,
            note_id=note_id,
            source_url=source_url,
            source_entity_id=source_canonical_id,
            predicate_id=relation_norm if pending_type == "relation" else "related_to",
            target_entity_id=target_canonical_id,
            confidence=confidence,
            evidence_ptrs=evidence_ptrs,
            reason=reason_payload,
            status=status,
        )

    def get_web_ontology_pending(self, pending_id: int) -> dict[str, Any] | None:
        item = self.get_ontology_pending(pending_id)
        if not item:
            return None
        reason_json = item.get("reason_json") if isinstance(item.get("reason_json"), dict) else {}
        relation_norm = str(reason_json.get("relation_norm", "")).strip()
        if not relation_norm:
            relation_norm = (
                "concept_candidate"
                if item.get("pending_type") == "concept"
                else str(item.get("predicate_id", "related_to"))
            )
        return {
            "id": item.get("id"),
            "run_id": item.get("run_id"),
            "topic_slug": item.get("topic_slug"),
            "note_id": item.get("note_id"),
            "source_url": item.get("source_url"),
            "source_canonical_id": item.get("source_entity_id"),
            "relation_norm": relation_norm,
            "target_canonical_id": item.get("target_entity_id"),
            "confidence": item.get("confidence"),
            "evidence_ptrs_json": item.get("evidence_ptrs_json")
            if isinstance(item.get("evidence_ptrs_json"), list)
            else [],
            "reason_json": reason_json,
            "status": item.get("status"),
            "created_at": item.get("created_at"),
            "reviewed_at": item.get("reviewed_at"),
        }

    def list_web_ontology_pending(
        self,
        topic_slug: str | None = None,
        status: str | None = "pending",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        items = self.list_ontology_pending(
            pending_type=None,
            topic_slug=topic_slug,
            status=status,
            limit=limit,
        )
        result: list[dict[str, Any]] = []
        for item in items:
            if item.get("pending_type") not in {"concept", "relation"}:
                continue
            normalized = self.get_web_ontology_pending(int(item.get("id", 0)))
            if normalized:
                result.append(normalized)
        return result
