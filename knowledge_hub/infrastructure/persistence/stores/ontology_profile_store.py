"""Canonical runtime state and proposal storage for ontology profiles."""

from __future__ import annotations

import json
from typing import Any


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, default=str)


def _json_loads(raw: Any, default: Any) -> Any:
    if raw in (None, ""):
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        parsed = json.loads(raw)
    except Exception:
        return default
    return parsed if isinstance(parsed, type(default)) else default


class OntologyProfileStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_profile_state (
                kind TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                source_path TEXT DEFAULT '',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_profile_runtime (
                runtime_key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_profile_proposals (
                proposal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposal_type TEXT NOT NULL,
                target_profile TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL DEFAULT '{}',
                source TEXT NOT NULL DEFAULT 'user',
                status TEXT NOT NULL DEFAULT 'pending',
                reason_json TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_profile_overlays (
                overlay_id INTEGER PRIMARY KEY AUTOINCREMENT,
                overlay_type TEXT NOT NULL,
                target_profile TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL DEFAULT '{}',
                source TEXT NOT NULL DEFAULT 'system',
                status TEXT NOT NULL DEFAULT 'approved',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def set_active_profile(self, kind: str, profile_id: str, source_path: str = "") -> None:
        self.conn.execute(
            """INSERT INTO ontology_profile_state(kind, profile_id, source_path, updated_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(kind) DO UPDATE SET
                 profile_id=excluded.profile_id,
                 source_path=excluded.source_path,
                 updated_at=CURRENT_TIMESTAMP""",
            (str(kind), str(profile_id), str(source_path)),
        )
        self.conn.commit()

    def get_active_profile(self, kind: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM ontology_profile_state WHERE kind = ?",
            (str(kind),),
        ).fetchone()
        return dict(row) if row else None

    def list_active_profiles(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM ontology_profile_state ORDER BY kind ASC").fetchall()
        return [dict(row) for row in rows]

    def set_runtime_json(self, key: str, value: Any) -> None:
        self.conn.execute(
            """INSERT INTO ontology_profile_runtime(runtime_key, value_json, updated_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(runtime_key) DO UPDATE SET
                 value_json=excluded.value_json,
                 updated_at=CURRENT_TIMESTAMP""",
            (str(key), _json_dumps(value)),
        )
        self.conn.commit()

    def get_runtime_json(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute(
            "SELECT value_json FROM ontology_profile_runtime WHERE runtime_key = ?",
            (str(key),),
        ).fetchone()
        if not row:
            return default
        return _json_loads(row["value_json"], default)

    def add_profile_proposal(
        self,
        proposal_type: str,
        target_profile: str,
        payload: dict[str, Any],
        *,
        source: str = "user",
        status: str = "pending",
        reason: dict[str, Any] | None = None,
    ) -> int:
        cursor = self.conn.execute(
            """INSERT INTO ontology_profile_proposals
               (proposal_type, target_profile, payload_json, source, status, reason_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(proposal_type),
                str(target_profile),
                _json_dumps(payload or {}),
                str(source or "user"),
                str(status or "pending"),
                _json_dumps(reason or {}),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def get_profile_proposal(self, proposal_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM ontology_profile_proposals WHERE proposal_id = ?",
            (int(proposal_id),),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["payload"] = _json_loads(item.get("payload_json"), {})
        item["reason"] = _json_loads(item.get("reason_json"), {})
        return item

    def list_profile_proposals(
        self,
        *,
        status: str | None = None,
        proposal_type: str | None = None,
        target_profile: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM ontology_profile_proposals WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(str(status))
        if proposal_type:
            query += " AND proposal_type = ?"
            params.append(str(proposal_type))
        if target_profile:
            query += " AND target_profile = ?"
            params.append(str(target_profile))
        query += " ORDER BY proposal_id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = _json_loads(item.get("payload_json"), {})
            item["reason"] = _json_loads(item.get("reason_json"), {})
            result.append(item)
        return result

    def update_profile_proposal_status(self, proposal_id: int, status: str, reason: dict[str, Any] | None = None) -> bool:
        current = self.get_profile_proposal(proposal_id)
        if not current:
            return False
        reason_payload = current.get("reason") if isinstance(current.get("reason"), dict) else {}
        if isinstance(reason, dict):
            reason_payload.update(reason)
        cursor = self.conn.execute(
            """UPDATE ontology_profile_proposals
               SET status = ?, reason_json = ?, updated_at = CURRENT_TIMESTAMP
               WHERE proposal_id = ?""",
            (str(status), _json_dumps(reason_payload), int(proposal_id)),
        )
        self.conn.commit()
        return bool(cursor.rowcount)

    def add_profile_overlay(
        self,
        overlay_type: str,
        target_profile: str,
        payload: dict[str, Any],
        *,
        source: str = "proposal_apply",
        status: str = "approved",
    ) -> int:
        cursor = self.conn.execute(
            """INSERT INTO ontology_profile_overlays
               (overlay_type, target_profile, payload_json, source, status)
               VALUES (?, ?, ?, ?, ?)""",
            (
                str(overlay_type),
                str(target_profile),
                _json_dumps(payload or {}),
                str(source or "proposal_apply"),
                str(status or "approved"),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def list_profile_overlays(
        self,
        *,
        status: str | None = "approved",
        target_profile: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM ontology_profile_overlays WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(str(status))
        if target_profile:
            query += " AND target_profile = ?"
            params.append(str(target_profile))
        query += " ORDER BY overlay_id ASC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = _json_loads(item.get("payload_json"), {})
            result.append(item)
        return result
