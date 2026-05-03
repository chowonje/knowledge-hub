"""Crawl pipeline and crawl-domain policy store extracted from SQLiteDatabase."""

from __future__ import annotations

import json
from typing import Any, Optional


class CrawlPipelineStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS crawl_domain_policy (
                domain TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('approved','pending','rejected')),
                reason TEXT DEFAULT '',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_crawl_domain_policy_status
            ON crawl_domain_policy(status, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS crawl_pipeline_jobs (
                job_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running'
                    CHECK(status IN ('queued','running','completed','partial','failed','blocked')),
                profile TEXT NOT NULL DEFAULT 'safe',
                source_policy TEXT NOT NULL DEFAULT 'hybrid',
                storage_root TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'web',
                topic TEXT NOT NULL DEFAULT '',
                sources_json TEXT NOT NULL DEFAULT '[]',
                requested_count INTEGER NOT NULL DEFAULT 0,
                processed_count INTEGER NOT NULL DEFAULT 0,
                normalized_count INTEGER NOT NULL DEFAULT 0,
                indexed_count INTEGER NOT NULL DEFAULT 0,
                pending_domain_count INTEGER NOT NULL DEFAULT 0,
                failed_count INTEGER NOT NULL DEFAULT 0,
                skipped_count INTEGER NOT NULL DEFAULT 0,
                retry_count INTEGER NOT NULL DEFAULT 0,
                dedupe_count INTEGER NOT NULL DEFAULT 0,
                warnings_json TEXT NOT NULL DEFAULT '[]',
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                finished_at TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_crawl_pipeline_jobs_status
            ON crawl_pipeline_jobs(status, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS crawl_pipeline_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES crawl_pipeline_jobs(job_id) ON DELETE CASCADE,
                record_id TEXT NOT NULL,
                source TEXT NOT NULL,
                source_url TEXT NOT NULL,
                canonical_url TEXT NOT NULL,
                domain TEXT NOT NULL,
                canonical_url_hash TEXT NOT NULL,
                content_sha256 TEXT NOT NULL DEFAULT '',
                state TEXT NOT NULL DEFAULT 'queued'
                    CHECK(state IN ('queued','downloading','normalized','indexed','pending_domain','failed','skipped')),
                retries INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                raw_path TEXT DEFAULT '',
                normalized_path TEXT DEFAULT '',
                indexed_path TEXT DEFAULT '',
                fetched_at TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(job_id, record_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_crawl_pipeline_records_state
            ON crawl_pipeline_records(job_id, state, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_crawl_pipeline_records_hash
            ON crawl_pipeline_records(job_id, canonical_url_hash, content_sha256)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS crawl_pipeline_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES crawl_pipeline_jobs(job_id) ON DELETE CASCADE,
                step TEXT NOT NULL,
                cursor TEXT NOT NULL DEFAULT '',
                last_record_id TEXT NOT NULL DEFAULT '',
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(job_id, step)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_crawl_pipeline_checkpoints_job
            ON crawl_pipeline_checkpoints(job_id, step)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS crawl_pipeline_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES crawl_pipeline_jobs(job_id) ON DELETE CASCADE,
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                phase TEXT NOT NULL DEFAULT '',
                memory_ratio REAL NOT NULL DEFAULT 0.0,
                cpu_ratio REAL NOT NULL DEFAULT 0.0,
                step_latency_ms REAL NOT NULL DEFAULT 0.0,
                retry_count INTEGER NOT NULL DEFAULT 0,
                dedupe_count INTEGER NOT NULL DEFAULT 0,
                details_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_crawl_pipeline_metrics_job
            ON crawl_pipeline_metrics(job_id, ts DESC)
            """
        )
        self.conn.commit()

    def upsert_crawl_domain_policy(self, domain: str, status: str, reason: str = "") -> dict[str, Any]:
        domain_token = str(domain or "").strip().lower()
        if not domain_token:
            raise ValueError("domain is required")
        status_token = str(status or "pending").strip().lower()
        if status_token not in {"approved", "pending", "rejected"}:
            status_token = "pending"
        self.conn.execute(
            """INSERT INTO crawl_domain_policy(domain, status, reason, updated_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(domain) DO UPDATE SET
                 status=excluded.status,
                 reason=excluded.reason,
                 updated_at=CURRENT_TIMESTAMP""",
            (domain_token, status_token, str(reason or "")),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT * FROM crawl_domain_policy WHERE domain = ?",
            (domain_token,),
        ).fetchone()
        return dict(row) if row else {"domain": domain_token, "status": status_token, "reason": str(reason or "")}

    def get_crawl_domain_policy(self, domain: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM crawl_domain_policy WHERE domain = ?",
            (str(domain or "").strip().lower(),),
        ).fetchone()
        return dict(row) if row else None

    def list_crawl_domain_policy(self, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        query = "SELECT * FROM crawl_domain_policy WHERE 1=1"
        params: list[Any] = []
        status_token = str(status or "").strip().lower()
        if status_token:
            query += " AND status = ?"
            params.append(status_token)
        query += " ORDER BY updated_at DESC, domain ASC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def create_crawl_pipeline_job(
        self,
        *,
        job_id: str,
        run_id: str,
        profile: str,
        source_policy: str,
        storage_root: str,
        source: str = "web",
        topic: str = "",
        sources: list[str] | None = None,
        status: str = "running",
    ) -> None:
        status_token = str(status or "running").strip().lower()
        if status_token not in {"queued", "running", "completed", "partial", "failed", "blocked"}:
            status_token = "running"
        self.conn.execute(
            """INSERT INTO crawl_pipeline_jobs
                 (job_id, run_id, status, profile, source_policy, storage_root, source, topic, sources_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(job_id) DO UPDATE SET
                 run_id=excluded.run_id,
                 status=excluded.status,
                 profile=excluded.profile,
                 source_policy=excluded.source_policy,
                 storage_root=excluded.storage_root,
                 source=excluded.source,
                 topic=excluded.topic,
                 sources_json=excluded.sources_json,
                 updated_at=CURRENT_TIMESTAMP""",
            (
                str(job_id),
                str(run_id),
                status_token,
                str(profile or "safe"),
                str(source_policy or "hybrid"),
                str(storage_root),
                str(source or "web"),
                str(topic or ""),
                json.dumps(sources or [], ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def get_crawl_pipeline_job(self, job_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM crawl_pipeline_jobs WHERE job_id = ?",
            (str(job_id),),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["sources_json"] = json.loads(item.get("sources_json") or "[]")
        except Exception:
            item["sources_json"] = []
        try:
            item["warnings_json"] = json.loads(item.get("warnings_json") or "[]")
        except Exception:
            item["warnings_json"] = []
        return item

    def update_crawl_pipeline_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        requested_count: int | None = None,
        processed_count: int | None = None,
        normalized_count: int | None = None,
        indexed_count: int | None = None,
        pending_domain_count: int | None = None,
        failed_count: int | None = None,
        skipped_count: int | None = None,
        retry_count: int | None = None,
        dedupe_count: int | None = None,
        warnings: list[str] | None = None,
        finished: bool = False,
    ) -> bool:
        updates: list[str] = []
        params: list[Any] = []
        status_token = str(status or "").strip().lower()
        if status_token:
            if status_token not in {"queued", "running", "completed", "partial", "failed", "blocked"}:
                status_token = "running"
            updates.append("status = ?")
            params.append(status_token)
        numeric_fields = {
            "requested_count": requested_count,
            "processed_count": processed_count,
            "normalized_count": normalized_count,
            "indexed_count": indexed_count,
            "pending_domain_count": pending_domain_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "retry_count": retry_count,
            "dedupe_count": dedupe_count,
        }
        for field, value in numeric_fields.items():
            if value is None:
                continue
            updates.append(f"{field} = ?")
            params.append(max(0, int(value)))
        if warnings is not None:
            updates.append("warnings_json = ?")
            params.append(json.dumps(list(warnings), ensure_ascii=False))
        updates.append("updated_at = CURRENT_TIMESTAMP")
        if finished:
            updates.append("finished_at = CURRENT_TIMESTAMP")
        if not updates:
            return False
        params.append(str(job_id))
        cursor = self.conn.execute(
            f"UPDATE crawl_pipeline_jobs SET {', '.join(updates)} WHERE job_id = ?",
            params,
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def upsert_crawl_pipeline_record(
        self,
        *,
        job_id: str,
        record_id: str,
        source: str,
        source_url: str,
        canonical_url: str,
        domain: str,
        canonical_url_hash: str,
        content_sha256: str = "",
        state: str = "queued",
        retries: int = 0,
        error: str = "",
        raw_path: str = "",
        normalized_path: str = "",
        indexed_path: str = "",
        fetched_at: str = "",
    ) -> None:
        state_token = str(state or "queued").strip().lower()
        if state_token not in {"queued", "downloading", "normalized", "indexed", "pending_domain", "failed", "skipped"}:
            state_token = "queued"
        self.conn.execute(
            """INSERT INTO crawl_pipeline_records
                 (job_id, record_id, source, source_url, canonical_url, domain, canonical_url_hash,
                  content_sha256, state, retries, error, raw_path, normalized_path, indexed_path, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(job_id, record_id) DO UPDATE SET
                 source=excluded.source,
                 source_url=excluded.source_url,
                 canonical_url=excluded.canonical_url,
                 domain=excluded.domain,
                 canonical_url_hash=excluded.canonical_url_hash,
                 content_sha256=excluded.content_sha256,
                 state=excluded.state,
                 retries=excluded.retries,
                 error=excluded.error,
                 raw_path=excluded.raw_path,
                 normalized_path=excluded.normalized_path,
                 indexed_path=excluded.indexed_path,
                 fetched_at=excluded.fetched_at,
                 updated_at=CURRENT_TIMESTAMP""",
            (
                str(job_id),
                str(record_id),
                str(source),
                str(source_url),
                str(canonical_url),
                str(domain),
                str(canonical_url_hash),
                str(content_sha256 or ""),
                state_token,
                max(0, int(retries)),
                str(error or ""),
                str(raw_path or ""),
                str(normalized_path or ""),
                str(indexed_path or ""),
                str(fetched_at or ""),
            ),
        )
        self.conn.commit()

    def get_crawl_pipeline_record(self, job_id: str, record_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM crawl_pipeline_records WHERE job_id = ? AND record_id = ?",
            (str(job_id), str(record_id)),
        ).fetchone()
        return dict(row) if row else None

    def list_crawl_pipeline_records(
        self,
        job_id: str,
        *,
        state: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM crawl_pipeline_records WHERE job_id = ?"
        params: list[Any] = [str(job_id)]
        state_token = str(state or "").strip().lower()
        if state_token:
            query += " AND state = ?"
            params.append(state_token)
        query += " ORDER BY id ASC LIMIT ? OFFSET ?"
        params.extend([max(1, int(limit)), max(0, int(offset))])
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def count_crawl_pipeline_records(self, job_id: str) -> dict[str, int]:
        rows = self.conn.execute(
            """SELECT state, COUNT(*) AS cnt
               FROM crawl_pipeline_records
               WHERE job_id = ?
               GROUP BY state""",
            (str(job_id),),
        ).fetchall()
        counts = {str(row["state"]): int(row["cnt"]) for row in rows}
        counts["total"] = sum(counts.values())
        return counts

    def update_crawl_pipeline_record_state(
        self,
        job_id: str,
        record_id: str,
        *,
        state: str,
        error: str | None = None,
        retries: int | None = None,
        content_sha256: str | None = None,
        raw_path: str | None = None,
        normalized_path: str | None = None,
        indexed_path: str | None = None,
        fetched_at: str | None = None,
    ) -> bool:
        updates = ["state = ?", "updated_at = CURRENT_TIMESTAMP"]
        params: list[Any] = [str(state).strip().lower()]
        if error is not None:
            updates.append("error = ?")
            params.append(str(error))
        if retries is not None:
            updates.append("retries = ?")
            params.append(max(0, int(retries)))
        if content_sha256 is not None:
            updates.append("content_sha256 = ?")
            params.append(str(content_sha256))
        if raw_path is not None:
            updates.append("raw_path = ?")
            params.append(str(raw_path))
        if normalized_path is not None:
            updates.append("normalized_path = ?")
            params.append(str(normalized_path))
        if indexed_path is not None:
            updates.append("indexed_path = ?")
            params.append(str(indexed_path))
        if fetched_at is not None:
            updates.append("fetched_at = ?")
            params.append(str(fetched_at))
        params.extend([str(job_id), str(record_id)])
        cursor = self.conn.execute(
            f"UPDATE crawl_pipeline_records SET {', '.join(updates)} WHERE job_id = ? AND record_id = ?",
            params,
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def upsert_crawl_pipeline_checkpoint(
        self,
        job_id: str,
        step: str,
        cursor_value: str = "",
        last_record_id: str = "",
    ) -> None:
        self.conn.execute(
            """INSERT INTO crawl_pipeline_checkpoints(job_id, step, cursor, last_record_id, ts)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(job_id, step) DO UPDATE SET
                 cursor=excluded.cursor,
                 last_record_id=excluded.last_record_id,
                 ts=CURRENT_TIMESTAMP""",
            (str(job_id), str(step), str(cursor_value or ""), str(last_record_id or "")),
        )
        self.conn.commit()

    def list_crawl_pipeline_checkpoints(self, job_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM crawl_pipeline_checkpoints WHERE job_id = ? ORDER BY step ASC",
            (str(job_id),),
        ).fetchall()
        return [dict(row) for row in rows]

    def append_crawl_pipeline_metric(
        self,
        job_id: str,
        *,
        phase: str,
        memory_ratio: float = 0.0,
        cpu_ratio: float = 0.0,
        step_latency_ms: float = 0.0,
        retry_count: int = 0,
        dedupe_count: int = 0,
        details: dict | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO crawl_pipeline_metrics
                 (job_id, phase, memory_ratio, cpu_ratio, step_latency_ms, retry_count, dedupe_count, details_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(job_id),
                str(phase or ""),
                float(memory_ratio or 0.0),
                float(cpu_ratio or 0.0),
                float(step_latency_ms or 0.0),
                max(0, int(retry_count)),
                max(0, int(dedupe_count)),
                json.dumps(details or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def list_crawl_pipeline_metrics(self, job_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT * FROM crawl_pipeline_metrics
               WHERE job_id = ?
               ORDER BY id ASC
               LIMIT ?""",
            (str(job_id), max(1, int(limit))),
        ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["details_json"] = json.loads(item.get("details_json") or "{}")
            except Exception:
                item["details_json"] = {}
            items.append(item)
        return items

    def get_latest_crawl_pipeline_job(self) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT job_id FROM crawl_pipeline_jobs ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return self.get_crawl_pipeline_job(str(row["job_id"]))
