"""Persistent execution receipt store for ops actions."""

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
    if isinstance(fallback, dict):
        return parsed if isinstance(parsed, dict) else fallback
    return parsed


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_artifact(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"items": value}
    return {"text": str(value)[:4000]}


class OpsActionReceiptStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ops_action_receipts (
                receipt_id TEXT PRIMARY KEY,
                action_id TEXT NOT NULL,
                executed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                mode TEXT NOT NULL DEFAULT 'sync',
                status TEXT NOT NULL DEFAULT 'started',
                runner TEXT NOT NULL DEFAULT 'cli',
                command TEXT NOT NULL DEFAULT '',
                args_json TEXT NOT NULL DEFAULT '[]',
                mcp_job_id TEXT NOT NULL DEFAULT '',
                result_summary TEXT NOT NULL DEFAULT '',
                error_summary TEXT NOT NULL DEFAULT '',
                artifact_json TEXT NOT NULL DEFAULT '{}',
                actor TEXT NOT NULL DEFAULT '',
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ops_action_receipts_action
            ON ops_action_receipts(action_id, executed_at DESC, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ops_action_receipts_job
            ON ops_action_receipts(mcp_job_id)
            """
        )
        self.conn.commit()

    def _row_to_item(self, row) -> dict[str, Any]:
        item = dict(row)
        item["args_json"] = _loads(item.get("args_json"), [])
        item["artifact_json"] = _loads(item.get("artifact_json"), {})
        return item

    def get_receipt(self, receipt_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM ops_action_receipts WHERE receipt_id = ?",
            (str(receipt_id),),
        ).fetchone()
        if not row:
            return None
        return self._row_to_item(row)

    def create_receipt(
        self,
        *,
        action_id: str,
        mode: str,
        status: str,
        runner: str,
        command: str,
        args: list[str] | None = None,
        mcp_job_id: str = "",
        result_summary: str = "",
        error_summary: str = "",
        artifact: Any = None,
        actor: str = "",
        executed_at: str | None = None,
    ) -> dict[str, Any]:
        receipt_id = f"ops_receipt_{uuid4().hex[:12]}"
        timestamp = str(executed_at or _now_iso())
        self.conn.execute(
            """
            INSERT INTO ops_action_receipts (
                receipt_id, action_id, executed_at, mode, status, runner, command,
                args_json, mcp_job_id, result_summary, error_summary, artifact_json,
                actor, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                str(action_id),
                timestamp,
                str(mode),
                str(status),
                str(runner),
                str(command or ""),
                json.dumps([str(arg) for arg in (args or [])], ensure_ascii=False),
                str(mcp_job_id or ""),
                str(result_summary or ""),
                str(error_summary or ""),
                json.dumps(_coerce_artifact(artifact), ensure_ascii=False),
                str(actor or ""),
                timestamp,
            ),
        )
        self.conn.commit()
        return self.get_receipt(receipt_id) or {}

    def update_receipt(
        self,
        receipt_id: str,
        *,
        status: str | None = None,
        mcp_job_id: str | None = None,
        result_summary: str | None = None,
        error_summary: str | None = None,
        artifact: Any = None,
        actor: str | None = None,
        updated_at: str | None = None,
    ) -> dict[str, Any] | None:
        existing = self.get_receipt(receipt_id)
        if existing is None:
            return None
        timestamp = str(updated_at or _now_iso())
        fields: dict[str, Any] = {"updated_at": timestamp}
        if status is not None:
            fields["status"] = str(status)
        if mcp_job_id is not None:
            fields["mcp_job_id"] = str(mcp_job_id)
        if result_summary is not None:
            fields["result_summary"] = str(result_summary)
        if error_summary is not None:
            fields["error_summary"] = str(error_summary)
        if artifact is not None:
            fields["artifact_json"] = json.dumps(_coerce_artifact(artifact), ensure_ascii=False)
        if actor is not None:
            fields["actor"] = str(actor)
        assignments = ", ".join(f"{column} = ?" for column in fields)
        params = [*fields.values(), str(receipt_id)]
        self.conn.execute(f"UPDATE ops_action_receipts SET {assignments} WHERE receipt_id = ?", params)
        self.conn.commit()
        return self.get_receipt(receipt_id)

    def list_receipts(self, *, action_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM ops_action_receipts
            WHERE action_id = ?
            ORDER BY executed_at DESC, updated_at DESC, rowid DESC
            LIMIT ?
            """,
            (str(action_id), max(1, int(limit))),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def latest_receipt(self, action_id: str) -> dict[str, Any] | None:
        rows = self.list_receipts(action_id=str(action_id), limit=1)
        return rows[0] if rows else None
