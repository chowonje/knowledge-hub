"""Labs-only store for materialized paper SectionCards."""

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


class SectionCardV1Store:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_section_cards_v1 (
                section_card_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'other',
                title TEXT NOT NULL DEFAULT '',
                section_path TEXT NOT NULL DEFAULT '',
                unit_type TEXT NOT NULL DEFAULT 'section',
                unit_ids_json TEXT NOT NULL DEFAULT '[]',
                contextual_summary TEXT NOT NULL DEFAULT '',
                source_excerpt TEXT NOT NULL DEFAULT '',
                document_thesis TEXT NOT NULL DEFAULT '',
                key_points_json TEXT NOT NULL DEFAULT '[]',
                scope_notes_json TEXT NOT NULL DEFAULT '[]',
                claims_json TEXT NOT NULL DEFAULT '[]',
                concepts_json TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0.0,
                provenance_json TEXT NOT NULL DEFAULT '{}',
                search_text TEXT NOT NULL DEFAULT '',
                origin TEXT NOT NULL DEFAULT 'materialized_v1',
                generator_model TEXT NOT NULL DEFAULT '',
                built_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _add_column_if_missing(
            self.conn,
            "paper_section_cards_v1",
            "generator_model",
            "generator_model TEXT NOT NULL DEFAULT ''",
        )
        add_derivative_lifecycle_columns(self.conn, "paper_section_cards_v1", _add_column_if_missing)
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_section_cards_v1_paper
            ON paper_section_cards_v1(paper_id, role, updated_at DESC)
            """
        )
        self.conn.commit()

    def _row_to_item(self, row) -> dict[str, Any]:
        item = dict(row)
        item["unit_ids"] = _loads_list(item.get("unit_ids_json"))
        item["key_points"] = _loads_list(item.get("key_points_json"))
        item["scope_notes"] = _loads_list(item.get("scope_notes_json"))
        item["claims"] = _loads_list(item.get("claims_json"))
        item["concepts"] = _loads_list(item.get("concepts_json"))
        item["provenance"] = _loads_dict(item.get("provenance_json"))
        item["stale"] = bool(item.get("stale"))
        return item

    def replace_paper_cards(self, *, paper_id: str, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.ensure_schema()
        token = str(paper_id or "").strip()
        if not token:
            raise ValueError("paper_id is required")
        self.conn.execute("DELETE FROM paper_section_cards_v1 WHERE paper_id = ?", (token,))
        for payload in cards:
            payload = dict(payload or {})
            source_hash = resolve_source_content_hash(self.conn, payload, document_id=str(payload.get("document_id") or f"paper:{token}"))
            self.conn.execute(
                """
                INSERT INTO paper_section_cards_v1 (
                    section_card_id, paper_id, document_id, role, title, section_path, unit_type,
                    unit_ids_json, contextual_summary, source_excerpt, document_thesis,
                    key_points_json, scope_notes_json, claims_json, concepts_json, confidence,
                    provenance_json, search_text, origin, generator_model, source_content_hash,
                    stale, stale_reason, invalidated_at, built_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), CURRENT_TIMESTAMP)
                """,
                (
                    str(payload.get("section_card_id") or ""),
                    token,
                    str(payload.get("document_id") or f"paper:{token}"),
                    str(payload.get("role") or "other"),
                    str(payload.get("title") or ""),
                    str(payload.get("section_path") or ""),
                    str(payload.get("unit_type") or "section"),
                    json.dumps(list(payload.get("unit_ids") or []), ensure_ascii=False),
                    str(payload.get("contextual_summary") or ""),
                    str(payload.get("source_excerpt") or ""),
                    str(payload.get("document_thesis") or ""),
                    json.dumps(list(payload.get("key_points") or []), ensure_ascii=False),
                    json.dumps(list(payload.get("scope_notes") or []), ensure_ascii=False),
                    json.dumps(list(payload.get("claims") or []), ensure_ascii=False),
                    json.dumps(list(payload.get("concepts") or []), ensure_ascii=False),
                    float(payload.get("confidence") or 0.0),
                    json.dumps(dict(payload.get("provenance") or {}), ensure_ascii=False),
                    str(payload.get("search_text") or ""),
                    str(payload.get("origin") or "materialized_v1"),
                    str(payload.get("generator_model") or ""),
                    source_hash,
                    fresh_stale_value(payload),
                    stale_reason_value(payload),
                    invalidated_at_value(payload),
                    str(payload.get("built_at") or "") or None,
                ),
            )
        self.conn.commit()
        return self.list_paper_cards(token)

    def list_paper_cards(self, paper_id: str) -> list[dict[str, Any]]:
        self.ensure_schema()
        rows = self.conn.execute(
            """
            SELECT * FROM paper_section_cards_v1
            WHERE paper_id = ?
              AND COALESCE(stale, 0) = 0
            ORDER BY
                CASE role
                    WHEN 'problem' THEN 0
                    WHEN 'method' THEN 1
                    WHEN 'results' THEN 2
                    WHEN 'limitations' THEN 3
                    ELSE 4
                END ASC,
                updated_at DESC
            """,
            (str(paper_id or "").strip(),),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]


__all__ = ["SectionCardV1Store"]
