"""Foundry sync conflict store helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class SyncConflictStore:
    """CRUD helpers for foundry dual-write conflict queue."""

    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS foundry_sync_pending_conflicts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conflict_key TEXT NOT NULL,
                conflict_type TEXT NOT NULL,
                connector_id TEXT NOT NULL,
                source_filter TEXT NOT NULL DEFAULT 'all',
                reason TEXT NOT NULL DEFAULT '',
                payload_hash TEXT NOT NULL,
                existing_hash TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}',
                existing_payload_json TEXT DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP,
                reviewer TEXT,
                resolution_note TEXT,
                CHECK(status IN ('pending', 'approved', 'rejected'))
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_foundry_conflicts_status
            ON foundry_sync_pending_conflicts(status, created_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_foundry_conflicts_connector
            ON foundry_sync_pending_conflicts(connector_id, source_filter, status)
            """
        )
        self.conn.commit()

    @staticmethod
    def hash_payload(payload: Any) -> str:
        serialized = json.dumps(payload if payload is not None else {}, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def add_conflict(
        self,
        *,
        conflict_key: str,
        conflict_type: str,
        connector_id: str,
        source_filter: str = "all",
        reason: str = "",
        payload: dict | None = None,
        existing_payload: dict | None = None,
    ) -> int:
        payload_value = payload or {}
        existing_value = existing_payload or {}
        payload_hash = self.hash_payload(payload_value)
        existing_hash = self.hash_payload(existing_value) if existing_payload is not None else None

        duplicate = self.conn.execute(
            """
            SELECT id FROM foundry_sync_pending_conflicts
            WHERE status = 'pending'
              AND conflict_key = ?
              AND connector_id = ?
              AND source_filter = ?
              AND payload_hash = ?
              AND IFNULL(existing_hash, '') = IFNULL(?, '')
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                str(conflict_key),
                str(connector_id),
                str(source_filter),
                payload_hash,
                existing_hash,
            ),
        ).fetchone()
        if duplicate:
            return int(duplicate["id"])

        cursor = self.conn.execute(
            """
            INSERT INTO foundry_sync_pending_conflicts (
                conflict_key, conflict_type, connector_id, source_filter, reason,
                payload_hash, existing_hash, payload_json, existing_payload_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                str(conflict_key),
                str(conflict_type or "unknown"),
                str(connector_id or "unknown"),
                str(source_filter or "all"),
                str(reason or ""),
                payload_hash,
                existing_hash,
                json.dumps(payload_value, ensure_ascii=False),
                json.dumps(existing_value, ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def get_conflict(self, conflict_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM foundry_sync_pending_conflicts WHERE id = ?",
            (int(conflict_id),),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        for key in ("payload_json", "existing_payload_json"):
            try:
                item[key] = json.loads(item.get(key) or "{}")
            except Exception:
                item[key] = {}
        return item

    def list_conflicts(
        self,
        *,
        status: str | None = "pending",
        connector_id: str | None = None,
        source_filter: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        query = "SELECT * FROM foundry_sync_pending_conflicts WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(str(status))
        if connector_id:
            query += " AND connector_id = ?"
            params.append(str(connector_id))
        if source_filter:
            query += " AND source_filter = ?"
            params.append(str(source_filter))
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))

        rows = self.conn.execute(query, params).fetchall()
        items: list[dict] = []
        for row in rows:
            item = dict(row)
            for key in ("payload_json", "existing_payload_json"):
                try:
                    item[key] = json.loads(item.get(key) or "{}")
                except Exception:
                    item[key] = {}
            items.append(item)
        return items

    def update_conflict_status(
        self,
        conflict_id: int,
        *,
        status: str,
        reviewer: str = "system",
        resolution_note: str = "",
    ) -> bool:
        status_value = str(status).strip().lower()
        if status_value not in {"pending", "approved", "rejected"}:
            status_value = "pending"
        cursor = self.conn.execute(
            """
            UPDATE foundry_sync_pending_conflicts
            SET status = ?,
                reviewer = ?,
                resolution_note = ?,
                reviewed_at = CASE
                    WHEN ? IN ('approved', 'rejected') THEN CURRENT_TIMESTAMP
                    ELSE reviewed_at
                END
            WHERE id = ?
            """,
            (
                status_value,
                reviewer,
                resolution_note,
                status_value,
                int(conflict_id),
            ),
        )
        self.conn.commit()
        return cursor.rowcount > 0
