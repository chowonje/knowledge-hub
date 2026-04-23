"""Canonical claim store implementation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from knowledge_hub.core.models import OntologyEvent
from knowledge_hub.infrastructure.persistence.stores.derivative_lifecycle import (
    add_derivative_lifecycle_columns,
    source_hash_from_payload,
)

log = logging.getLogger("khub.claim_store")


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


def _add_column_if_missing(conn, table: str, column_name: str, column_sql: str) -> None:
    columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column_name in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_sql}")


class ClaimStore:
    def __init__(self, conn, event_store=None):
        self.conn = conn
        self.event_store = event_store

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS claim_normalizations (
                claim_id TEXT NOT NULL REFERENCES ontology_claims(claim_id) ON DELETE CASCADE,
                normalization_version TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'failed'
                    CHECK(status IN ('normalized', 'partial', 'failed')),
                task TEXT DEFAULT '',
                dataset TEXT DEFAULT '',
                metric TEXT DEFAULT '',
                comparator TEXT DEFAULT '',
                result_direction TEXT NOT NULL DEFAULT 'unknown'
                    CHECK(result_direction IN ('better', 'worse', 'neutral', 'unknown')),
                result_value_text TEXT DEFAULT '',
                result_value_numeric REAL,
                condition_text TEXT DEFAULT '',
                scope_text TEXT DEFAULT '',
                limitation_text TEXT DEFAULT '',
                negative_scope_text TEXT DEFAULT '',
                evidence_strength TEXT NOT NULL DEFAULT 'weak'
                    CHECK(evidence_strength IN ('strong', 'medium', 'weak')),
                normalized_payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (claim_id, normalization_version)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_claim_normalizations_status
            ON claim_normalizations(status, normalization_version, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_claim_normalizations_task_metric
            ON claim_normalizations(task, metric, dataset, updated_at DESC)
            """
        )
        columns = {str(row[1]) for row in self.conn.execute("PRAGMA table_info(claim_normalizations)").fetchall()}
        if "negative_scope_text" not in columns:
            self.conn.execute("ALTER TABLE claim_normalizations ADD COLUMN negative_scope_text TEXT DEFAULT ''")
        if self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ontology_claims'").fetchone():
            add_derivative_lifecycle_columns(self.conn, "ontology_claims", _add_column_if_missing)
        self.conn.commit()

    def _decode_claim(self, row) -> Optional[dict[str, Any]]:
        if not row:
            return None
        item = dict(row)
        try:
            item["evidence_ptrs"] = json.loads(item.get("evidence_ptrs_json") or "[]")
        except Exception:
            item["evidence_ptrs"] = []
        item["stale"] = bool(item.get("stale"))
        return item

    def _decode_claim_normalization(self, row) -> Optional[dict[str, Any]]:
        if not row:
            return None
        item = dict(row)
        item["result_value_numeric"] = (
            float(item["result_value_numeric"])
            if item.get("result_value_numeric") is not None
            else None
        )
        item["normalized_payload"] = _safe_json_dict(item.get("normalized_payload_json"))
        return item

    def upsert_claim(
        self,
        *,
        claim_id: str,
        claim_text: str,
        subject_entity_id: str,
        predicate: str,
        object_entity_id: str | None = None,
        object_literal: str | None = None,
        confidence: float = 0.5,
        evidence_ptrs: list[dict[str, str]] | None = None,
        source: str = "extraction",
        valid_from: str | None = None,
        valid_to: str | None = None,
    ) -> None:
        existing = self.get_claim(claim_id)
        is_update = existing is not None
        evidence_json = json.dumps(evidence_ptrs or [], ensure_ascii=False)
        source_hash = source_hash_from_payload({"evidence_ptrs": evidence_ptrs or []})
        self.conn.execute(
            """INSERT INTO ontology_claims
               (claim_id, claim_text, subject_entity_id, predicate, object_entity_id,
                object_literal, confidence, evidence_ptrs_json, source, valid_from, valid_to,
                origin, source_content_hash, stale, stale_reason, invalidated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'derived', ?, 0, '', '')
               ON CONFLICT(claim_id) DO UPDATE SET
                 claim_text=excluded.claim_text,
                 subject_entity_id=excluded.subject_entity_id,
                 predicate=excluded.predicate,
                 object_entity_id=excluded.object_entity_id,
                 object_literal=excluded.object_literal,
                 confidence=excluded.confidence,
                 evidence_ptrs_json=excluded.evidence_ptrs_json,
                 source=excluded.source,
                 valid_from=excluded.valid_from,
                 valid_to=excluded.valid_to,
                 origin=excluded.origin,
                 source_content_hash=excluded.source_content_hash,
                 stale=excluded.stale,
                 stale_reason=excluded.stale_reason,
                 invalidated_at=excluded.invalidated_at""",
            (
                str(claim_id),
                str(claim_text),
                str(subject_entity_id),
                str(predicate),
                object_entity_id,
                object_literal,
                float(confidence),
                evidence_json,
                str(source),
                valid_from,
                valid_to,
                source_hash,
            ),
        )
        self.conn.commit()

        if not self.event_store:
            return
        event = OntologyEvent(
            event_id=f"evt_{uuid4().hex}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="claim_updated" if is_update else "claim_added",
            entity_id=str(claim_id),
            entity_type="claim",
            actor=str(source),
            data={
                "claim_id": claim_id,
                "claim_text": claim_text,
                "subject_entity_id": subject_entity_id,
                "predicate": predicate,
                "object_entity_id": object_entity_id,
                "object_literal": object_literal,
                "confidence": confidence,
                "evidence_ptrs": evidence_ptrs or [],
            },
            policy_class="P2",
        )
        try:
            self.event_store.append(event)
        except Exception as error:
            log.error("Event append failed for claim upsert (%s): %s", claim_id, error)

    def get_claim(self, claim_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM ontology_claims WHERE claim_id = ?",
            (str(claim_id),),
        ).fetchone()
        return self._decode_claim(row)

    def upsert_claim_normalization(
        self,
        *,
        claim_id: str,
        normalization_version: str,
        status: str,
        task: str = "",
        dataset: str = "",
        metric: str = "",
        comparator: str = "",
        result_direction: str = "unknown",
        result_value_text: str = "",
        result_value_numeric: float | None = None,
        condition_text: str = "",
        scope_text: str = "",
        limitation_text: str = "",
        negative_scope_text: str = "",
        evidence_strength: str = "weak",
        normalized_payload: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO claim_normalizations
            (
                claim_id,
                normalization_version,
                status,
                task,
                dataset,
                metric,
                comparator,
                result_direction,
                result_value_text,
                result_value_numeric,
                condition_text,
                scope_text,
                limitation_text,
                negative_scope_text,
                evidence_strength,
                normalized_payload_json,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(claim_id, normalization_version) DO UPDATE SET
                status=excluded.status,
                task=excluded.task,
                dataset=excluded.dataset,
                metric=excluded.metric,
                comparator=excluded.comparator,
                result_direction=excluded.result_direction,
                result_value_text=excluded.result_value_text,
                result_value_numeric=excluded.result_value_numeric,
                condition_text=excluded.condition_text,
                scope_text=excluded.scope_text,
                limitation_text=excluded.limitation_text,
                negative_scope_text=excluded.negative_scope_text,
                evidence_strength=excluded.evidence_strength,
                normalized_payload_json=excluded.normalized_payload_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                str(claim_id).strip(),
                str(normalization_version).strip(),
                str(status).strip(),
                str(task or "").strip(),
                str(dataset or "").strip(),
                str(metric or "").strip(),
                str(comparator or "").strip(),
                str(result_direction or "unknown").strip(),
                str(result_value_text or "").strip(),
                float(result_value_numeric) if result_value_numeric is not None else None,
                str(condition_text or "").strip(),
                str(scope_text or "").strip(),
                str(limitation_text or "").strip(),
                str(negative_scope_text or "").strip(),
                str(evidence_strength or "weak").strip(),
                json.dumps(normalized_payload or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def get_claim_normalization(
        self,
        claim_id: str,
        *,
        normalization_version: str,
    ) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            """
            SELECT * FROM claim_normalizations
            WHERE claim_id = ? AND normalization_version = ?
            """,
            (str(claim_id).strip(), str(normalization_version).strip()),
        ).fetchone()
        return self._decode_claim_normalization(row)

    def list_claim_normalizations(
        self,
        *,
        normalization_version: str | None = None,
        claim_ids: list[str] | None = None,
        status: str | None = None,
        task: str | None = None,
        dataset: str | None = None,
        metric: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM claim_normalizations WHERE 1=1"
        params: list[Any] = []
        if normalization_version:
            query += " AND normalization_version = ?"
            params.append(str(normalization_version).strip())
        if status:
            query += " AND status = ?"
            params.append(str(status).strip())
        if task:
            query += " AND task = ?"
            params.append(str(task).strip())
        if dataset:
            query += " AND dataset = ?"
            params.append(str(dataset).strip())
        if metric:
            query += " AND metric = ?"
            params.append(str(metric).strip())
        if claim_ids:
            normalized_ids = [str(item).strip() for item in claim_ids if str(item).strip()]
            if normalized_ids:
                placeholders = ", ".join("?" for _ in normalized_ids)
                query += f" AND claim_id IN ({placeholders})"
                params.extend(normalized_ids)
        query += " ORDER BY updated_at DESC, claim_id ASC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        return [item for item in (self._decode_claim_normalization(row) for row in rows) if item]

    def list_claims(
        self,
        *,
        subject_id: str | None = None,
        predicate: str | None = None,
        object_id: str | None = None,
        limit: int = 500,
        include_stale: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM ontology_claims WHERE 1=1"
        params: list[Any] = []
        columns = {str(row[1]) for row in self.conn.execute("PRAGMA table_info(ontology_claims)").fetchall()}
        if subject_id:
            query += " AND subject_entity_id = ?"
            params.append(str(subject_id))
        if predicate:
            query += " AND predicate = ?"
            params.append(str(predicate))
        if object_id:
            query += " AND object_entity_id = ?"
            params.append(str(object_id))
        if "stale" in columns and not include_stale:
            query += " AND COALESCE(stale, 0) = 0"
        query += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        return [item for item in (self._decode_claim(row) for row in rows) if item]

    def _get_note(self, note_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM notes WHERE id = ?",
            (str(note_id),),
        ).fetchone()
        return dict(row) if row else None

    def list_claims_by_note(self, note_id: str, limit: int = 200) -> list[dict[str, Any]]:
        note_id = str(note_id or "").strip()
        if not note_id:
            return []
        note = self._get_note(note_id) or {}
        metadata = _safe_json_dict(note.get("metadata"))
        file_path = str(note.get("file_path") or "").strip()
        url = str(metadata.get("canonical_url") or metadata.get("url") or "").strip()
        record_id = str(metadata.get("record_id") or "").strip()
        source_item_id = str(metadata.get("source_item_id") or "").strip()
        rows = self.list_claims(limit=max(1, int(limit * 4)))
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            evidence_ptrs = row.get("evidence_ptrs") if isinstance(row.get("evidence_ptrs"), list) else []
            for ptr in evidence_ptrs:
                if not isinstance(ptr, dict):
                    continue
                ptr_note_id = str(ptr.get("note_id") or "").strip()
                ptr_record_id = str(ptr.get("record_id") or "").strip()
                ptr_source_item_id = str(ptr.get("source_item_id") or "").strip()
                ptr_path = str(ptr.get("path") or "").strip()
                ptr_url = str(ptr.get("source_url") or ptr.get("url") or "").strip()
                if (
                    ptr_note_id == note_id
                    or (record_id and ptr_record_id == record_id)
                    or (source_item_id and ptr_source_item_id == source_item_id)
                ):
                    if row["claim_id"] not in seen:
                        seen.add(str(row["claim_id"]))
                        result.append(row)
                    break
                if file_path and ptr_path and ptr_path == file_path:
                    if row["claim_id"] not in seen:
                        seen.add(str(row["claim_id"]))
                        result.append(row)
                    break
                if url and ptr_url and ptr_url == url:
                    if row["claim_id"] not in seen:
                        seen.add(str(row["claim_id"]))
                        result.append(row)
                    break
            if len(result) >= limit:
                break
        return result[:limit]

    def list_claims_by_record(self, record_id: str, limit: int = 200) -> list[dict[str, Any]]:
        record_id = str(record_id or "").strip()
        if not record_id:
            return []
        rows = self.conn.execute(
            "SELECT id, metadata FROM notes WHERE metadata LIKE ? ORDER BY updated_at DESC LIMIT 50",
            (f'%\"record_id\": \"{record_id}\"%',),
        ).fetchall()
        note_ids = [str(row["id"]) for row in rows]
        source_item_ids: set[str] = set()
        for row in rows:
            metadata = _safe_json_dict(row["metadata"] if "metadata" in row.keys() else None)
            token = str(metadata.get("source_item_id") or "").strip()
            if token:
                source_item_ids.add(token)
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for note_id in note_ids:
            for row in self.list_claims_by_note(note_id, limit=limit):
                claim_id = str(row.get("claim_id") or "")
                if claim_id and claim_id not in seen:
                    seen.add(claim_id)
                    result.append(row)
                if len(result) >= limit:
                    return result[:limit]
        for row in self.list_claims(limit=max(1, int(limit * 4))):
            evidence_ptrs = row.get("evidence_ptrs") if isinstance(row.get("evidence_ptrs"), list) else []
            if any(
                (
                    str(ptr.get("record_id") or "").strip() == record_id
                    or (
                        source_item_ids
                        and str(ptr.get("source_item_id") or "").strip() in source_item_ids
                    )
                )
                for ptr in evidence_ptrs
                if isinstance(ptr, dict)
            ):
                claim_id = str(row.get("claim_id") or "")
                if claim_id and claim_id not in seen:
                    seen.add(claim_id)
                    result.append(row)
                if len(result) >= limit:
                    break
        return result[:limit]

    def list_claims_by_entity(self, entity_id: str, limit: int = 200) -> list[dict[str, Any]]:
        entity_id = str(entity_id or "").strip()
        if not entity_id:
            return []
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in self.list_claims(subject_id=entity_id, limit=limit):
            claim_id = str(row.get("claim_id") or "")
            if claim_id and claim_id not in seen:
                seen.add(claim_id)
                result.append(row)
        for row in self.list_claims(object_id=entity_id, limit=limit):
            claim_id = str(row.get("claim_id") or "")
            if claim_id and claim_id not in seen:
                seen.add(claim_id)
                result.append(row)
        result.sort(
            key=lambda item: (
                float(item.get("confidence", 0.0) or 0.0),
                str(item.get("created_at") or ""),
            ),
            reverse=True,
        )
        return result[:limit]

    def delete_claim(self, claim_id: str) -> None:
        claim_id = str(claim_id).strip()
        if not claim_id:
            return
        existing = self.get_claim(claim_id)
        self.conn.execute("DELETE FROM ontology_claims WHERE claim_id = ?", (claim_id,))
        self.conn.commit()
        if not self.event_store or not existing:
            return
        event = OntologyEvent(
            event_id=f"evt_{uuid4().hex}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="claim_deleted",
            entity_id=claim_id,
            entity_type="claim",
            actor=str(existing.get("source", "system")),
            data={
                "claim_id": claim_id,
                "claim_text": str(existing.get("claim_text", "")),
                "subject_entity_id": str(existing.get("subject_entity_id", "")),
                "predicate": str(existing.get("predicate", "")),
                "object_entity_id": str(existing.get("object_entity_id", "")),
                "object_literal": str(existing.get("object_literal", "")),
                "confidence": float(existing.get("confidence", 0.0) or 0.0),
            },
            policy_class="P2",
        )
        try:
            self.event_store.append(event)
        except Exception as error:
            log.error("Event append failed for claim delete (%s): %s", claim_id, error)
