"""Persistence helpers for external notebook bridge sync state."""

from __future__ import annotations

from typing import Any


class NotebookBridgeStore:
    """Stores idempotent topic notebook bindings and synced source manifests."""

    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS external_notebook_topics (
                provider TEXT NOT NULL,
                topic_slug TEXT NOT NULL,
                topic TEXT NOT NULL DEFAULT '',
                remote_notebook_id TEXT NOT NULL DEFAULT '',
                remote_notebook_url TEXT DEFAULT '',
                remote_notebook_title TEXT DEFAULT '',
                last_sync_status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (provider, topic_slug)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS external_notebook_sources (
                provider TEXT NOT NULL,
                topic_slug TEXT NOT NULL,
                local_source_id TEXT NOT NULL,
                local_source_type TEXT NOT NULL,
                source_title TEXT DEFAULT '',
                content_hash TEXT NOT NULL DEFAULT '',
                remote_notebook_id TEXT DEFAULT '',
                remote_source_id TEXT DEFAULT '',
                last_synced_at TIMESTAMP,
                last_sync_status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (provider, topic_slug, local_source_type, local_source_id)
            )
            """
        )
        self.conn.commit()

    def get_topic_binding(self, provider: str, topic_slug: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT provider, topic_slug, topic, remote_notebook_id, remote_notebook_url,
                   remote_notebook_title, last_sync_status, created_at, updated_at
            FROM external_notebook_topics
            WHERE provider = ? AND topic_slug = ?
            """,
            (provider, topic_slug),
        ).fetchone()
        return dict(row) if row else None

    def upsert_topic_binding(
        self,
        *,
        provider: str,
        topic_slug: str,
        topic: str,
        remote_notebook_id: str,
        remote_notebook_url: str = "",
        remote_notebook_title: str = "",
        last_sync_status: str = "pending",
    ) -> dict[str, Any]:
        self.conn.execute(
            """
            INSERT INTO external_notebook_topics (
                provider, topic_slug, topic, remote_notebook_id, remote_notebook_url,
                remote_notebook_title, last_sync_status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(provider, topic_slug) DO UPDATE SET
                topic = excluded.topic,
                remote_notebook_id = excluded.remote_notebook_id,
                remote_notebook_url = excluded.remote_notebook_url,
                remote_notebook_title = excluded.remote_notebook_title,
                last_sync_status = excluded.last_sync_status,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                provider,
                topic_slug,
                topic,
                remote_notebook_id,
                remote_notebook_url,
                remote_notebook_title,
                last_sync_status,
            ),
        )
        self.conn.commit()
        return self.get_topic_binding(provider, topic_slug) or {}

    def list_topic_bindings(self, provider: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query = """
            SELECT provider, topic_slug, topic, remote_notebook_id, remote_notebook_url,
                   remote_notebook_title, last_sync_status, created_at, updated_at
            FROM external_notebook_topics
        """
        params: list[Any] = []
        if provider:
            query += " WHERE provider = ?"
            params.append(provider)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_source_binding(
        self,
        *,
        provider: str,
        topic_slug: str,
        local_source_type: str,
        local_source_id: str,
    ) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT provider, topic_slug, local_source_id, local_source_type, source_title,
                   content_hash, remote_notebook_id, remote_source_id, last_synced_at,
                   last_sync_status, created_at, updated_at
            FROM external_notebook_sources
            WHERE provider = ? AND topic_slug = ? AND local_source_type = ? AND local_source_id = ?
            """,
            (provider, topic_slug, local_source_type, local_source_id),
        ).fetchone()
        return dict(row) if row else None

    def upsert_source_binding(
        self,
        *,
        provider: str,
        topic_slug: str,
        local_source_id: str,
        local_source_type: str,
        source_title: str,
        content_hash: str,
        remote_notebook_id: str,
        remote_source_id: str = "",
        last_sync_status: str = "pending",
        last_synced_at: str | None = None,
    ) -> dict[str, Any]:
        synced_at = last_synced_at or "CURRENT_TIMESTAMP"
        if last_synced_at:
            self.conn.execute(
                """
                INSERT INTO external_notebook_sources (
                    provider, topic_slug, local_source_id, local_source_type, source_title,
                    content_hash, remote_notebook_id, remote_source_id, last_synced_at,
                    last_sync_status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(provider, topic_slug, local_source_type, local_source_id) DO UPDATE SET
                    source_title = excluded.source_title,
                    content_hash = excluded.content_hash,
                    remote_notebook_id = excluded.remote_notebook_id,
                    remote_source_id = excluded.remote_source_id,
                    last_synced_at = excluded.last_synced_at,
                    last_sync_status = excluded.last_sync_status,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    provider,
                    topic_slug,
                    local_source_id,
                    local_source_type,
                    source_title,
                    content_hash,
                    remote_notebook_id,
                    remote_source_id,
                    last_synced_at,
                    last_sync_status,
                ),
            )
        else:
            self.conn.execute(
                f"""
                INSERT INTO external_notebook_sources (
                    provider, topic_slug, local_source_id, local_source_type, source_title,
                    content_hash, remote_notebook_id, remote_source_id, last_synced_at,
                    last_sync_status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, {synced_at}, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(provider, topic_slug, local_source_type, local_source_id) DO UPDATE SET
                    source_title = excluded.source_title,
                    content_hash = excluded.content_hash,
                    remote_notebook_id = excluded.remote_notebook_id,
                    remote_source_id = excluded.remote_source_id,
                    last_synced_at = excluded.last_synced_at,
                    last_sync_status = excluded.last_sync_status,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    provider,
                    topic_slug,
                    local_source_id,
                    local_source_type,
                    source_title,
                    content_hash,
                    remote_notebook_id,
                    remote_source_id,
                    last_sync_status,
                ),
            )
        self.conn.commit()
        return self.get_source_binding(
            provider=provider,
            topic_slug=topic_slug,
            local_source_type=local_source_type,
            local_source_id=local_source_id,
        ) or {}

    def list_source_bindings(
        self,
        *,
        provider: str | None = None,
        topic_slug: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT provider, topic_slug, local_source_id, local_source_type, source_title,
                   content_hash, remote_notebook_id, remote_source_id, last_synced_at,
                   last_sync_status, created_at, updated_at
            FROM external_notebook_sources
            WHERE 1 = 1
        """
        params: list[Any] = []
        if provider:
            query += " AND provider = ?"
            params.append(provider)
        if topic_slug:
            query += " AND topic_slug = ?"
            params.append(topic_slug)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
