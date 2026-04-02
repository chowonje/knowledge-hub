"""MCP job queue store helpers for SQLiteDatabase facade."""

from __future__ import annotations

import json
from typing import Any


class MCPJobStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mcp_jobs (
                job_id TEXT PRIMARY KEY,
                tool TEXT NOT NULL,
                actor TEXT DEFAULT 'system',
                status TEXT NOT NULL DEFAULT 'queued',
                request_json TEXT NOT NULL DEFAULT '{}',
                request_echo_json TEXT,
                artifact_json TEXT,
                classification TEXT NOT NULL DEFAULT 'P2',
                policy_result TEXT NOT NULL DEFAULT 'allowed',
                source_refs_json TEXT DEFAULT '[]',
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                finished_at TIMESTAMP,
                run_time_ms INTEGER,
                progress INTEGER DEFAULT 0
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mcp_jobs_status
            ON mcp_jobs(status, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mcp_jobs_tool
            ON mcp_jobs(tool, status)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mcp_jobs_finished
            ON mcp_jobs(finished_at)
            """
        )
        self.conn.commit()

    def create_mcp_job(
        self,
        job_id: str,
        tool: str,
        request: dict,
        actor: str | None = None,
        request_echo: dict | None = None,
        classification: str = "P2",
        status: str = "queued",
        progress: int = 0,
    ) -> str:
        self.conn.execute(
            """
            INSERT INTO mcp_jobs (
                job_id, tool, actor, status, request_json, request_echo_json,
                classification, progress
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(job_id),
                str(tool),
                actor or "system",
                status,
                json.dumps(request or {}, ensure_ascii=False),
                json.dumps(request_echo or {}, ensure_ascii=False),
                classification,
                int(progress),
            ),
        )
        self.conn.commit()
        return str(job_id)

    def update_mcp_job(
        self,
        job_id: str,
        status: str | None = None,
        finished_at: str | None = None,
        started_at: str | None = None,
        updated_at: str | None = None,
        run_time_ms: int | None = None,
        progress: int | None = None,
        policy_result: str | None = None,
        error: str | None = None,
        artifact: Any | None = None,
        source_refs: list[str] | None = None,
    ) -> bool:
        sets: list[str] = []
        params: list[object] = []
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if finished_at is not None:
            sets.append("finished_at = ?")
            params.append(finished_at)
        if started_at is not None:
            sets.append("started_at = ?")
            params.append(started_at)
        if updated_at is not None:
            sets.append("updated_at = ?")
            params.append(updated_at)
        if run_time_ms is not None:
            sets.append("run_time_ms = ?")
            params.append(int(run_time_ms))
        if progress is not None:
            sets.append("progress = ?")
            params.append(int(progress))
        if policy_result is not None:
            sets.append("policy_result = ?")
            params.append(str(policy_result))
        if error is not None:
            sets.append("error = ?")
            params.append(str(error))
        if artifact is not None:
            sets.append("artifact_json = ?")
            params.append(json.dumps(artifact, ensure_ascii=False))
        if source_refs is not None:
            sets.append("source_refs_json = ?")
            params.append(json.dumps(source_refs, ensure_ascii=False))
        if not sets:
            return False
        sql = f"UPDATE mcp_jobs SET {', '.join(sets)} WHERE job_id = ?"
        params.append(str(job_id))
        cursor = self.conn.execute(sql, params)
        self.conn.commit()
        return cursor.rowcount > 0

    def get_mcp_job(self, job_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM mcp_jobs WHERE job_id = ?", (str(job_id),)).fetchone()
        if not row:
            return None
        item = dict(row)
        for key in ("request_json", "request_echo_json", "artifact_json", "source_refs_json"):
            try:
                if key in item:
                    item[key] = json.loads(item[key]) if item[key] is not None else None
            except Exception:
                item[key] = {} if key != "source_refs_json" else []
        return item

    def list_mcp_jobs(
        self,
        status: str | None = None,
        tool: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        where: list[str] = []
        params: list[object] = []
        if status is not None:
            where.append("status = ?")
            params.append(status)
        if tool is not None:
            where.append("tool = ?")
            params.append(tool)
        where_clause = f" WHERE {' AND '.join(where)}" if where else ""
        query = f"SELECT * FROM mcp_jobs{where_clause} ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()

        result: list[dict] = []
        for row in rows:
            item = dict(row)
            for key in ("request_json", "request_echo_json", "artifact_json", "source_refs_json"):
                try:
                    if key in item:
                        item[key] = json.loads(item[key]) if item[key] is not None else None
                except Exception:
                    item[key] = {} if key != "source_refs_json" else []
            result.append(item)
        return result

    def cancel_mcp_job(self, job_id: str) -> bool:
        cursor = self.conn.execute(
            """
            UPDATE mcp_jobs
            SET status = 'expired', updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ? AND status IN ('queued', 'running')
            """,
            (str(job_id),),
        )
        self.conn.commit()
        return cursor.rowcount > 0
