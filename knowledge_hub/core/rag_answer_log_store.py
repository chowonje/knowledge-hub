"""Persistent operator-safe RAG answer telemetry store."""

from __future__ import annotations

import json
from typing import Any


class RAGAnswerLogStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_answer_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                query_hash TEXT NOT NULL DEFAULT '',
                query_digest TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL DEFAULT '',
                retrieval_mode TEXT NOT NULL DEFAULT 'hybrid',
                allow_external INTEGER NOT NULL DEFAULT 0,
                result_status TEXT NOT NULL DEFAULT 'ok',
                verification_status TEXT NOT NULL DEFAULT '',
                needs_caution INTEGER NOT NULL DEFAULT 0,
                supported_claim_count INTEGER NOT NULL DEFAULT 0,
                uncertain_claim_count INTEGER NOT NULL DEFAULT 0,
                unsupported_claim_count INTEGER NOT NULL DEFAULT 0,
                conflict_mentioned INTEGER NOT NULL DEFAULT 0,
                rewrite_attempted INTEGER NOT NULL DEFAULT 0,
                rewrite_applied INTEGER NOT NULL DEFAULT 0,
                final_answer_source TEXT NOT NULL DEFAULT 'original',
                warning_count INTEGER NOT NULL DEFAULT 0,
                source_count INTEGER NOT NULL DEFAULT 0,
                evidence_count INTEGER NOT NULL DEFAULT 0,
                answer_route_json TEXT NOT NULL DEFAULT '{}',
                verification_route_json TEXT NOT NULL DEFAULT '{}',
                rewrite_route_json TEXT NOT NULL DEFAULT '{}',
                warnings_json TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rag_answer_logs_created_at
            ON rag_answer_logs(created_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rag_answer_logs_status
            ON rag_answer_logs(result_status, verification_status, created_at DESC)
            """
        )
        self.conn.commit()

    def add_log(
        self,
        *,
        query_hash: str,
        query_digest: str,
        source_type: str = "",
        retrieval_mode: str = "hybrid",
        allow_external: bool = False,
        result_status: str = "ok",
        verification_status: str = "",
        needs_caution: bool = False,
        supported_claim_count: int = 0,
        uncertain_claim_count: int = 0,
        unsupported_claim_count: int = 0,
        conflict_mentioned: bool = False,
        rewrite_attempted: bool = False,
        rewrite_applied: bool = False,
        final_answer_source: str = "original",
        warning_count: int = 0,
        source_count: int = 0,
        evidence_count: int = 0,
        answer_route: dict[str, Any] | None = None,
        verification_route: dict[str, Any] | None = None,
        rewrite_route: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO rag_answer_logs (
                query_hash, query_digest, source_type, retrieval_mode, allow_external,
                result_status, verification_status, needs_caution,
                supported_claim_count, uncertain_claim_count, unsupported_claim_count,
                conflict_mentioned, rewrite_attempted, rewrite_applied,
                final_answer_source, warning_count, source_count, evidence_count,
                answer_route_json, verification_route_json, rewrite_route_json, warnings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(query_hash or ""),
                str(query_digest or ""),
                str(source_type or ""),
                str(retrieval_mode or "hybrid"),
                1 if allow_external else 0,
                str(result_status or "ok"),
                str(verification_status or ""),
                1 if needs_caution else 0,
                int(supported_claim_count or 0),
                int(uncertain_claim_count or 0),
                int(unsupported_claim_count or 0),
                1 if conflict_mentioned else 0,
                1 if rewrite_attempted else 0,
                1 if rewrite_applied else 0,
                str(final_answer_source or "original"),
                int(warning_count or 0),
                int(source_count or 0),
                int(evidence_count or 0),
                json.dumps(answer_route or {}, ensure_ascii=False),
                json.dumps(verification_route or {}, ensure_ascii=False),
                json.dumps(rewrite_route or {}, ensure_ascii=False),
                json.dumps(warnings or [], ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def list_logs(self, *, limit: int = 100, days: int = 0) -> list[dict[str, Any]]:
        query = "SELECT * FROM rag_answer_logs"
        params: list[Any] = []
        if int(days or 0) > 0:
            query += " WHERE created_at >= datetime('now', ?)"
            params.append(f"-{int(days)} days")
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("answer_route_json", "verification_route_json", "rewrite_route_json", "warnings_json"):
                try:
                    item[key] = json.loads(item.get(key) or ("[]" if key == "warnings_json" else "{}"))
                except Exception:
                    item[key] = [] if key == "warnings_json" else {}
            items.append(item)
        return items
