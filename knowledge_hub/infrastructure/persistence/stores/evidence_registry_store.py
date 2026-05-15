"""SQLite-backed registry for evidence packet and context lookup artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from typing import Any


REGISTRY_RECORD_SCHEMA = "knowledge-hub.evidence-registry.record.v1"
VALID_RECORD_KINDS = {"packet", "context", "trace"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_loads(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value or "")
    except Exception:
        return fallback


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = _json_dumps(payload).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class EvidenceRegistryStore:
    """Stores derived lookup records without becoming source authority.

    Records persist packet/context/trace payloads for URI resolution, but each
    row carries source revision and lineage metadata so callers can treat it as
    a derived lookup artifact rather than factual source truth.
    """

    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_registry_records (
                record_kind TEXT NOT NULL,
                registry_id TEXT NOT NULL,
                payload_schema TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'ok',
                source_revision_hash TEXT NOT NULL DEFAULT '',
                source_refs_json TEXT NOT NULL DEFAULT '[]',
                lineage_json TEXT NOT NULL DEFAULT '{}',
                authority_json TEXT NOT NULL DEFAULT '{}',
                payload_json TEXT NOT NULL DEFAULT '{}',
                payload_hash TEXT NOT NULL DEFAULT '',
                token_count INTEGER NOT NULL DEFAULT 0,
                expires_at TEXT NOT NULL DEFAULT '',
                stale INTEGER NOT NULL DEFAULT 0,
                stale_reason TEXT NOT NULL DEFAULT '',
                deletion_policy TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (record_kind, registry_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_evidence_registry_kind_status
            ON evidence_registry_records(record_kind, status, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_evidence_registry_expiry
            ON evidence_registry_records(expires_at)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_evidence_registry_payload_hash
            ON evidence_registry_records(payload_hash)
            """
        )
        self.conn.commit()

    def upsert_record(
        self,
        *,
        registry_id: str,
        record_kind: str,
        payload: dict[str, Any],
        payload_schema: str = "",
        status: str = "ok",
        source_revision_hash: str = "",
        source_refs: list[dict[str, Any]] | None = None,
        lineage: dict[str, Any] | None = None,
        authority: dict[str, Any] | None = None,
        token_count: int = 0,
        expires_at: str = "",
        stale: bool = False,
        stale_reason: str = "",
        deletion_policy: str = "",
    ) -> dict[str, Any]:
        kind = str(record_kind or "").strip().lower()
        if kind not in VALID_RECORD_KINDS:
            raise ValueError(f"unsupported evidence registry record kind: {record_kind}")
        identifier = str(registry_id or "").strip()
        if not identifier:
            raise ValueError("evidence registry id is required")
        if not isinstance(payload, dict):
            raise ValueError("evidence registry payload must be an object")

        self.ensure_schema()
        now = _now_iso()
        resolved_schema = str(payload_schema or payload.get("schema") or "")
        resolved_status = "stale" if stale else str(status or "ok")
        resolved_payload_hash = _payload_hash(payload)
        self.conn.execute(
            """
            INSERT INTO evidence_registry_records (
                record_kind, registry_id, payload_schema, status,
                source_revision_hash, source_refs_json, lineage_json, authority_json,
                payload_json, payload_hash, token_count, expires_at, stale,
                stale_reason, deletion_policy, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_kind, registry_id) DO UPDATE SET
                payload_schema = excluded.payload_schema,
                status = excluded.status,
                source_revision_hash = excluded.source_revision_hash,
                source_refs_json = excluded.source_refs_json,
                lineage_json = excluded.lineage_json,
                authority_json = excluded.authority_json,
                payload_json = excluded.payload_json,
                payload_hash = excluded.payload_hash,
                token_count = excluded.token_count,
                expires_at = excluded.expires_at,
                stale = excluded.stale,
                stale_reason = excluded.stale_reason,
                deletion_policy = excluded.deletion_policy,
                updated_at = excluded.updated_at
            """,
            (
                kind,
                identifier,
                resolved_schema,
                resolved_status,
                str(source_revision_hash or ""),
                _json_dumps(source_refs or []),
                _json_dumps(lineage or {}),
                _json_dumps(authority or {}),
                _json_dumps(payload),
                resolved_payload_hash,
                max(0, int(token_count or 0)),
                str(expires_at or ""),
                1 if stale else 0,
                str(stale_reason or ""),
                str(deletion_policy or ""),
                now,
                now,
            ),
        )
        self.conn.commit()
        record = self.get_record(kind, identifier)
        if record is None:  # pragma: no cover - defensive sqlite failure guard
            raise RuntimeError("evidence registry record write did not round-trip")
        return record

    def get_record(self, record_kind: str, registry_id: str) -> dict[str, Any] | None:
        kind = str(record_kind or "").strip().lower()
        identifier = str(registry_id or "").strip()
        if kind not in VALID_RECORD_KINDS or not identifier:
            return None
        try:
            row = self.conn.execute(
                """
                SELECT * FROM evidence_registry_records
                WHERE record_kind = ? AND registry_id = ?
                """,
                (kind, identifier),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if not row:
            return None
        return self._row_to_record(row)

    def list_records(self, *, record_kind: str = "", limit: int = 100, include_stale: bool = True) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        kind = str(record_kind or "").strip().lower()
        if kind:
            if kind not in VALID_RECORD_KINDS:
                return []
            where.append("record_kind = ?")
            params.append(kind)
        if not include_stale:
            where.append("stale = 0")
        where_clause = f" WHERE {' AND '.join(where)}" if where else ""
        params.append(max(1, int(limit or 100)))
        try:
            rows = self.conn.execute(
                f"SELECT * FROM evidence_registry_records{where_clause} ORDER BY updated_at DESC LIMIT ?",
                params,
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [self._row_to_record(row) for row in rows]

    def delete_record(self, record_kind: str, registry_id: str) -> bool:
        kind = str(record_kind or "").strip().lower()
        identifier = str(registry_id or "").strip()
        if kind not in VALID_RECORD_KINDS or not identifier:
            return False
        try:
            cursor = self.conn.execute(
                "DELETE FROM evidence_registry_records WHERE record_kind = ? AND registry_id = ?",
                (kind, identifier),
            )
        except sqlite3.OperationalError:
            return False
        self.conn.commit()
        return cursor.rowcount > 0

    def prune_expired_records(self, *, now: str | None = None) -> int:
        cutoff = str(now or _now_iso())
        try:
            cursor = self.conn.execute(
                """
                DELETE FROM evidence_registry_records
                WHERE expires_at != '' AND expires_at <= ?
                """,
                (cutoff,),
            )
        except sqlite3.OperationalError:
            return 0
        self.conn.commit()
        return int(cursor.rowcount or 0)

    def _row_to_record(self, row: Any) -> dict[str, Any]:
        item = dict(row)
        stale = bool(item.get("stale"))
        status = str(item.get("status") or "ok")
        if stale and status == "ok":
            status = "stale"
        return {
            "schema": REGISTRY_RECORD_SCHEMA,
            "status": status,
            "registryId": str(item.get("registry_id") or ""),
            "recordKind": str(item.get("record_kind") or ""),
            "payloadSchema": str(item.get("payload_schema") or ""),
            "sourceRevisionHash": str(item.get("source_revision_hash") or ""),
            "sourceRefs": _json_loads(str(item.get("source_refs_json") or "[]"), []),
            "lineage": _json_loads(str(item.get("lineage_json") or "{}"), {}),
            "authority": _json_loads(str(item.get("authority_json") or "{}"), {}),
            "payload": _json_loads(str(item.get("payload_json") or "{}"), {}),
            "payloadHash": str(item.get("payload_hash") or ""),
            "tokenCount": int(item.get("token_count") or 0),
            "expiresAt": str(item.get("expires_at") or ""),
            "stale": stale,
            "staleReason": str(item.get("stale_reason") or ""),
            "deletionPolicy": str(item.get("deletion_policy") or ""),
            "createdAt": str(item.get("created_at") or ""),
            "updatedAt": str(item.get("updated_at") or ""),
        }
