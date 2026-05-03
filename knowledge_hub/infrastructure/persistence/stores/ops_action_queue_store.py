"""Canonical persistent operator action queue store."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _loads(raw: Any, fallback):
    if raw is None or raw == "":
        return fallback
    if isinstance(raw, (dict, list)):
        return raw
    try:
        parsed = json.loads(raw)
    except Exception:
        return fallback
    if isinstance(fallback, list):
        return parsed if isinstance(parsed, list) else fallback
    return parsed if isinstance(parsed, dict) else fallback


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class OpsActionQueueStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ops_action_queue (
                action_id TEXT PRIMARY KEY,
                scope TEXT NOT NULL DEFAULT '',
                action_type TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                target_kind TEXT NOT NULL DEFAULT '',
                target_key TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                reason_codes_json TEXT NOT NULL DEFAULT '[]',
                command TEXT NOT NULL DEFAULT '',
                args_json TEXT NOT NULL DEFAULT '[]',
                alert_json TEXT NOT NULL DEFAULT '[]',
                action_json TEXT NOT NULL DEFAULT '{}',
                first_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                seen_count INTEGER NOT NULL DEFAULT 1,
                acked_at TIMESTAMP,
                acked_by TEXT NOT NULL DEFAULT '',
                resolved_at TIMESTAMP,
                resolved_by TEXT NOT NULL DEFAULT '',
                note TEXT NOT NULL DEFAULT '',
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(scope, action_type, target_kind, target_key)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ops_action_queue_status
            ON ops_action_queue(status, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ops_action_queue_scope
            ON ops_action_queue(scope, status, updated_at DESC)
            """
        )
        self.conn.commit()

    def _row_to_item(self, row) -> dict[str, Any]:
        item = dict(row)
        item["reason_codes_json"] = _loads(item.get("reason_codes_json"), [])
        item["args_json"] = _loads(item.get("args_json"), [])
        item["alert_json"] = _loads(item.get("alert_json"), [])
        item["action_json"] = _loads(item.get("action_json"), {})
        return item

    def get_action(self, action_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM ops_action_queue WHERE action_id = ?", (str(action_id),)).fetchone()
        if not row:
            return None
        return self._row_to_item(row)

    def get_action_by_identity(
        self,
        *,
        scope: str,
        action_type: str,
        target_kind: str,
        target_key: str,
    ) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT * FROM ops_action_queue
            WHERE scope = ? AND action_type = ? AND target_kind = ? AND target_key = ?
            """,
            (str(scope), str(action_type), str(target_kind), str(target_key)),
        ).fetchone()
        if not row:
            return None
        return self._row_to_item(row)

    def upsert_action(
        self,
        *,
        scope: str,
        action_type: str,
        target_kind: str,
        target_key: str,
        summary: str,
        reason_codes: list[str] | None = None,
        command: str = "",
        args: list[str] | None = None,
        alerts: list[dict[str, Any]] | None = None,
        action: dict[str, Any] | None = None,
        seen_at: str | None = None,
    ) -> dict[str, Any]:
        timestamp = str(seen_at or _now_iso())
        existing = self.get_action_by_identity(
            scope=scope,
            action_type=action_type,
            target_kind=target_kind,
            target_key=target_key,
        )
        normalized_reason_codes = [str(item) for item in (reason_codes or []) if str(item).strip()]
        normalized_args = [str(item) for item in (args or []) if str(item).strip()]
        if existing is None:
            action_id = f"ops_action_{uuid4().hex[:12]}"
            self.conn.execute(
                """
                INSERT INTO ops_action_queue (
                    action_id, scope, action_type, status, target_kind, target_key,
                    summary, reason_codes_json, command, args_json, alert_json, action_json,
                    first_seen_at, last_seen_at, seen_count, updated_at
                ) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    action_id,
                    str(scope),
                    str(action_type),
                    str(target_kind),
                    str(target_key),
                    str(summary or ""),
                    json.dumps(normalized_reason_codes, ensure_ascii=False),
                    str(command or ""),
                    json.dumps(normalized_args, ensure_ascii=False),
                    json.dumps(alerts or [], ensure_ascii=False),
                    json.dumps(action or {}, ensure_ascii=False),
                    timestamp,
                    timestamp,
                    timestamp,
                ),
            )
            self.conn.commit()
            return {"operation": "created", "item": self.get_action(action_id)}

        next_status = str(existing.get("status") or "pending")
        operation = "updated"
        if next_status == "resolved":
            next_status = "pending"
            operation = "reopened"
        self.conn.execute(
            """
            UPDATE ops_action_queue
            SET status = ?,
                summary = ?,
                reason_codes_json = ?,
                command = ?,
                args_json = ?,
                alert_json = ?,
                action_json = ?,
                last_seen_at = ?,
                seen_count = ?,
                updated_at = ?
            WHERE action_id = ?
            """,
            (
                next_status,
                str(summary or ""),
                json.dumps(normalized_reason_codes, ensure_ascii=False),
                str(command or ""),
                json.dumps(normalized_args, ensure_ascii=False),
                json.dumps(alerts or [], ensure_ascii=False),
                json.dumps(action or {}, ensure_ascii=False),
                timestamp,
                int(existing.get("seen_count") or 0) + 1,
                timestamp,
                str(existing.get("action_id") or ""),
            ),
        )
        self.conn.commit()
        return {"operation": operation, "item": self.get_action(str(existing.get("action_id") or ""))}

    def list_actions(
        self,
        *,
        status: str | None = None,
        scope: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if status:
            where.append("status = ?")
            params.append(str(status))
        if scope:
            where.append("scope = ?")
            params.append(str(scope))
        query = "SELECT * FROM ops_action_queue"
        if where:
            query += f" WHERE {' AND '.join(where)}"
        query += " ORDER BY last_seen_at DESC, updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_item(row) for row in rows]

    def action_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            """
            SELECT status, COUNT(*) AS total
            FROM ops_action_queue
            GROUP BY status
            """
        ).fetchall()
        counts = {"pending": 0, "acked": 0, "resolved": 0, "total": 0}
        for row in rows:
            status = str(row["status"] or "")
            if status in counts:
                counts[status] = int(row["total"] or 0)
        counts["total"] = int(counts["pending"]) + int(counts["acked"]) + int(counts["resolved"])
        return counts

    def set_action_status(
        self,
        action_id: str,
        *,
        status: str,
        actor: str = "",
        note: str = "",
        changed_at: str | None = None,
    ) -> dict[str, Any] | None:
        if str(status) not in {"acked", "resolved"}:
            raise ValueError(f"unsupported action status: {status}")
        existing = self.get_action(action_id)
        if existing is None:
            return None
        timestamp = str(changed_at or _now_iso())
        fields = {
            "status": str(status),
            "note": str(note or ""),
            "updated_at": timestamp,
        }
        if str(status) == "acked":
            fields["acked_at"] = timestamp
            fields["acked_by"] = str(actor or "")
        if str(status) == "resolved":
            fields["resolved_at"] = timestamp
            fields["resolved_by"] = str(actor or "")
        assignments = ", ".join(f"{column} = ?" for column in fields)
        params = [*fields.values(), str(action_id)]
        self.conn.execute(f"UPDATE ops_action_queue SET {assignments} WHERE action_id = ?", params)
        self.conn.commit()
        return self.get_action(action_id)
