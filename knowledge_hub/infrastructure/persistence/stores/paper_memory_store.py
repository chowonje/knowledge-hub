"""Paper memory card storage helpers."""

from __future__ import annotations

import json
from typing import Any

from knowledge_hub.infrastructure.persistence.stores.derivative_lifecycle import (
    add_derivative_lifecycle_columns,
    fresh_stale_value,
    invalidated_at_value,
    resolve_source_content_hash,
    stale_reason_value,
)


def _add_column_if_missing(conn, table: str, column_name: str, column_sql: str) -> None:
    columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column_name in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_sql}")


def _loads_list(raw: Any) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item or "").strip()]
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item or "").strip()]


def _loads_dict(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): value for key, value in parsed.items()}


class PaperMemoryStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_memory_cards (
                memory_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL UNIQUE,
                source_note_id TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                paper_core TEXT NOT NULL DEFAULT '',
                problem_context TEXT NOT NULL DEFAULT '',
                method_core TEXT NOT NULL DEFAULT '',
                evidence_core TEXT NOT NULL DEFAULT '',
                limitations TEXT NOT NULL DEFAULT '',
                concept_links_json TEXT NOT NULL DEFAULT '[]',
                claim_refs_json TEXT NOT NULL DEFAULT '[]',
                formal_cause_json TEXT NOT NULL DEFAULT '{}',
                final_cause_json TEXT NOT NULL DEFAULT '{}',
                published_at TEXT NOT NULL DEFAULT '',
                evidence_window TEXT NOT NULL DEFAULT '',
                search_text TEXT NOT NULL DEFAULT '',
                quality_flag TEXT NOT NULL DEFAULT 'unscored',
                version TEXT NOT NULL DEFAULT 'paper-memory-v1',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _add_column_if_missing(
            self.conn,
            "paper_memory_cards",
            "published_at",
            "published_at TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "paper_memory_cards",
            "evidence_window",
            "evidence_window TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "paper_memory_cards",
            "formal_cause_json",
            "formal_cause_json TEXT NOT NULL DEFAULT '{}'",
        )
        _add_column_if_missing(
            self.conn,
            "paper_memory_cards",
            "final_cause_json",
            "final_cause_json TEXT NOT NULL DEFAULT '{}'",
        )
        add_derivative_lifecycle_columns(self.conn, "paper_memory_cards", _add_column_if_missing)
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_memory_cards_updated_at
            ON paper_memory_cards(updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_memory_cards_quality
            ON paper_memory_cards(quality_flag, updated_at DESC)
            """
        )
        self.conn.commit()

    def _row_to_item(self, row) -> dict[str, Any]:
        item = dict(row)
        item["concept_links"] = _loads_list(item.get("concept_links_json"))
        item["claim_refs"] = _loads_list(item.get("claim_refs_json"))
        item["formal_cause"] = _loads_dict(item.get("formal_cause_json"))
        item["final_cause"] = _loads_dict(item.get("final_cause_json"))
        item["stale"] = bool(item.get("stale"))
        return item

    def upsert_card(self, *, card: dict[str, Any]) -> dict[str, Any]:
        payload = dict(card or {})
        paper_id = str(payload.get("paper_id") or "")
        source_hash = resolve_source_content_hash(self.conn, payload, document_id=f"paper:{paper_id}" if paper_id else "")
        self.conn.execute(
            """
            INSERT INTO paper_memory_cards (
                memory_id, paper_id, source_note_id, title, paper_core, problem_context,
                method_core, evidence_core, limitations, concept_links_json, claim_refs_json,
                formal_cause_json, final_cause_json, published_at, evidence_window,
                search_text, quality_flag, version, source_content_hash, stale, stale_reason,
                invalidated_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), CURRENT_TIMESTAMP)
            ON CONFLICT(paper_id) DO UPDATE SET
                memory_id=excluded.memory_id,
                source_note_id=excluded.source_note_id,
                title=excluded.title,
                paper_core=excluded.paper_core,
                problem_context=excluded.problem_context,
                method_core=excluded.method_core,
                evidence_core=excluded.evidence_core,
                limitations=excluded.limitations,
                concept_links_json=excluded.concept_links_json,
                claim_refs_json=excluded.claim_refs_json,
                formal_cause_json=excluded.formal_cause_json,
                final_cause_json=excluded.final_cause_json,
                published_at=excluded.published_at,
                evidence_window=excluded.evidence_window,
                search_text=excluded.search_text,
                quality_flag=excluded.quality_flag,
                version=excluded.version,
                source_content_hash=excluded.source_content_hash,
                stale=excluded.stale,
                stale_reason=excluded.stale_reason,
                invalidated_at=excluded.invalidated_at,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                str(payload.get("memory_id") or ""),
                paper_id,
                str(payload.get("source_note_id") or ""),
                str(payload.get("title") or ""),
                str(payload.get("paper_core") or ""),
                str(payload.get("problem_context") or ""),
                str(payload.get("method_core") or ""),
                str(payload.get("evidence_core") or ""),
                str(payload.get("limitations") or ""),
                json.dumps(list(payload.get("concept_links") or []), ensure_ascii=False),
                json.dumps(list(payload.get("claim_refs") or []), ensure_ascii=False),
                json.dumps(dict(payload.get("formal_cause") or {}), ensure_ascii=False),
                json.dumps(dict(payload.get("final_cause") or {}), ensure_ascii=False),
                str(payload.get("published_at") or ""),
                str(payload.get("evidence_window") or ""),
                str(payload.get("search_text") or ""),
                str(payload.get("quality_flag") or "unscored"),
                str(payload.get("version") or "paper-memory-v1"),
                source_hash,
                fresh_stale_value(payload),
                stale_reason_value(payload),
                invalidated_at_value(payload),
                str(payload.get("created_at") or "") or None,
            ),
        )
        self.conn.commit()
        return self.get_card(str(payload.get("paper_id") or ""))

    def get_card(self, paper_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM paper_memory_cards WHERE paper_id = ?",
            (str(paper_id or ""),),
        ).fetchone()
        return self._row_to_item(row) if row else None

    def list_cards(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM paper_memory_cards ORDER BY updated_at DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def search_cards(self, query: str, *, limit: int = 20, include_stale: bool = False) -> list[dict[str, Any]]:
        token = str(query or "").strip()
        if not token:
            return []
        rows = self.conn.execute(
            """
            SELECT *,
                   (
                       CASE WHEN title LIKE ? THEN 4 ELSE 0 END +
                       CASE WHEN paper_core LIKE ? THEN 3 ELSE 0 END +
                       CASE WHEN problem_context LIKE ? THEN 2 ELSE 0 END +
                       CASE WHEN method_core LIKE ? THEN 2 ELSE 0 END +
                       CASE WHEN evidence_core LIKE ? THEN 2 ELSE 0 END +
                       CASE WHEN search_text LIKE ? THEN 5 ELSE 0 END
                   ) AS match_score
            FROM paper_memory_cards
            WHERE (? = 1 OR COALESCE(stale, 0) = 0)
              AND (
                  title LIKE ?
               OR paper_core LIKE ?
               OR problem_context LIKE ?
               OR method_core LIKE ?
               OR evidence_core LIKE ?
               OR search_text LIKE ?
              )
            ORDER BY match_score DESC, updated_at DESC
            LIMIT ?
            """,
            tuple([f"%{token}%"] * 6 + [1 if include_stale else 0, *[f"%{token}%"] * 6, max(1, int(limit))]),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]
