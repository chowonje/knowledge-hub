"""Generic document-memory storage helpers."""

from __future__ import annotations

import json
import re
from typing import Any


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


class DocumentMemoryStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_memory_units (
                unit_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                document_title TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL DEFAULT '',
                source_ref TEXT NOT NULL DEFAULT '',
                unit_type TEXT NOT NULL DEFAULT 'section',
                title TEXT NOT NULL DEFAULT '',
                section_path TEXT NOT NULL DEFAULT '',
                contextual_summary TEXT NOT NULL DEFAULT '',
                source_excerpt TEXT NOT NULL DEFAULT '',
                context_header TEXT NOT NULL DEFAULT '',
                document_thesis TEXT NOT NULL DEFAULT '',
                parent_unit_id TEXT NOT NULL DEFAULT '',
                scope_id TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                provenance_json TEXT NOT NULL DEFAULT '{}',
                order_index INTEGER NOT NULL DEFAULT 0,
                content_type TEXT NOT NULL DEFAULT 'plain',
                links_json TEXT NOT NULL DEFAULT '[]',
                tags_json TEXT NOT NULL DEFAULT '[]',
                claims_json TEXT NOT NULL DEFAULT '[]',
                concepts_json TEXT NOT NULL DEFAULT '[]',
                search_text TEXT NOT NULL DEFAULT '',
                version TEXT NOT NULL DEFAULT 'document-memory-v1',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _add_column_if_missing(
            self.conn,
            "document_memory_units",
            "context_header",
            "context_header TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "document_memory_units",
            "document_thesis",
            "document_thesis TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "document_memory_units",
            "document_date",
            "document_date TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "document_memory_units",
            "event_date",
            "event_date TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "document_memory_units",
            "observed_at",
            "observed_at TEXT NOT NULL DEFAULT ''",
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_document_memory_document
            ON document_memory_units(document_id, order_index ASC, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_document_memory_unit_type
            ON document_memory_units(unit_type, source_type, updated_at DESC)
            """
        )
        self.conn.commit()

    def _row_to_item(self, row) -> dict[str, Any]:
        item = dict(row)
        item["links"] = _loads_list(item.get("links_json"))
        item["tags"] = _loads_list(item.get("tags_json"))
        item["claims"] = _loads_list(item.get("claims_json"))
        item["concepts"] = _loads_list(item.get("concepts_json"))
        item["provenance"] = _loads_dict(item.get("provenance_json"))
        return item

    def replace_units(self, *, document_id: str, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        token = str(document_id or "").strip()
        if not token:
            raise ValueError("document_id is required")
        self.conn.execute("DELETE FROM document_memory_units WHERE document_id = ?", (token,))
        for payload in units:
            self.conn.execute(
                """
                INSERT INTO document_memory_units (
                    unit_id, document_id, document_title, source_type, source_ref, unit_type,
                    title, section_path, contextual_summary, source_excerpt, context_header,
                    document_thesis, parent_unit_id, document_date, event_date, observed_at,
                    scope_id, confidence, provenance_json, order_index, content_type,
                    links_json, tags_json, claims_json, concepts_json, search_text, version,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), CURRENT_TIMESTAMP)
                """,
                (
                    str(payload.get("unit_id") or ""),
                    token,
                    str(payload.get("document_title") or ""),
                    str(payload.get("source_type") or ""),
                    str(payload.get("source_ref") or ""),
                    str(payload.get("unit_type") or "section"),
                    str(payload.get("title") or ""),
                    str(payload.get("section_path") or ""),
                    str(payload.get("contextual_summary") or ""),
                    str(payload.get("source_excerpt") or ""),
                    str(payload.get("context_header") or ""),
                    str(payload.get("document_thesis") or ""),
                    str(payload.get("parent_unit_id") or ""),
                    str(payload.get("document_date") or ""),
                    str(payload.get("event_date") or ""),
                    str(payload.get("observed_at") or ""),
                    str(payload.get("scope_id") or ""),
                    float(payload.get("confidence") or 0.0),
                    json.dumps(dict(payload.get("provenance") or {}), ensure_ascii=False),
                    int(payload.get("order_index") or 0),
                    str(payload.get("content_type") or "plain"),
                    json.dumps(list(payload.get("links") or []), ensure_ascii=False),
                    json.dumps(list(payload.get("tags") or []), ensure_ascii=False),
                    json.dumps(list(payload.get("claims") or []), ensure_ascii=False),
                    json.dumps(list(payload.get("concepts") or []), ensure_ascii=False),
                    str(payload.get("search_text") or ""),
                    str(payload.get("version") or "document-memory-v1"),
                    str(payload.get("created_at") or "") or None,
                ),
            )
        self.conn.commit()
        return self.list_document_units(token, limit=max(1, len(units) + 4))

    def get_unit(self, unit_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM document_memory_units WHERE unit_id = ?",
            (str(unit_id or "").strip(),),
        ).fetchone()
        return self._row_to_item(row) if row else None

    def get_document_summary(self, document_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT * FROM document_memory_units
            WHERE document_id = ? AND unit_type = 'document_summary'
            ORDER BY order_index ASC, updated_at DESC
            LIMIT 1
            """,
            (str(document_id or "").strip(),),
        ).fetchone()
        return self._row_to_item(row) if row else None

    def list_document_units(self, document_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM document_memory_units
            WHERE document_id = ?
            ORDER BY order_index ASC, updated_at DESC
            LIMIT ?
            """,
            (str(document_id or "").strip(), max(1, int(limit))),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def search_units(
        self,
        query: str,
        *,
        limit: int = 20,
        unit_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        token = str(query or "").strip()
        if not token:
            return []
        type_tokens = [str(item or "").strip() for item in unit_types or [] if str(item or "").strip()]
        where_parts: list[str] = []
        base_params: list[Any] = []
        if type_tokens:
            placeholders = ", ".join("?" for _ in type_tokens)
            where_parts.append(f"unit_type IN ({placeholders})")
            base_params.extend(type_tokens)

        def _select_rows(extra_clause: str = "", extra_params: list[Any] | None = None, row_limit: int = 0):
            query_sql = "SELECT * FROM document_memory_units"
            params = list(base_params)
            clauses = list(where_parts)
            if extra_clause:
                clauses.append(extra_clause)
            if clauses:
                query_sql += " WHERE " + " AND ".join(clauses)
            query_sql += " ORDER BY updated_at DESC"
            if row_limit > 0:
                query_sql += f" LIMIT {int(row_limit)}"
            params.extend(list(extra_params or []))
            return self.conn.execute(query_sql, tuple(params)).fetchall()

        per_source_limit = max(300, max(1, int(limit)) * 80)
        rows: list[Any] = []
        seen_unit_ids: set[str] = set()
        for source_type in ("paper", "vault", "web"):
            for row in _select_rows("source_type = ?", [source_type], row_limit=per_source_limit):
                unit_id = str(row["unit_id"]) if "unit_id" in row.keys() else str(row[0])
                if unit_id in seen_unit_ids:
                    continue
                seen_unit_ids.add(unit_id)
                rows.append(row)
        for row in _select_rows("source_type NOT IN ('paper', 'vault', 'web')", row_limit=max(60, int(limit) * 10)):
            unit_id = str(row["unit_id"]) if "unit_id" in row.keys() else str(row[0])
            if unit_id in seen_unit_ids:
                continue
            seen_unit_ids.add(unit_id)
            rows.append(row)

        terms = [part.casefold() for part in re.split(r"\s+", token) if part.strip()]
        scored: list[tuple[int, float, dict[str, Any]]] = []
        for row in rows:
            item = self._row_to_item(row)
            haystack = " ".join(
                [
                    str(item.get("document_title") or ""),
                    str(item.get("title") or ""),
                    str(item.get("source_type") or ""),
                    str(item.get("source_ref") or ""),
                    str((item.get("provenance") or {}).get("file_path") or ""),
                    " ".join(str(part or "") for part in list(((item.get("provenance") or {}).get("heading_path") or [])) if str(part or "").strip()),
                    str(item.get("contextual_summary") or ""),
                    str(item.get("section_path") or ""),
                    str(item.get("source_excerpt") or ""),
                    str(item.get("search_text") or ""),
                    str(item.get("document_date") or ""),
                    str(item.get("event_date") or ""),
                    str(item.get("observed_at") or ""),
                    " ".join(str(tag or "") for tag in list(item.get("tags") or []) if str(tag or "").strip()),
                    " ".join(str(link or "") for link in list(item.get("links") or []) if str(link or "").strip()),
                ]
            ).casefold()
            score = 0
            for term in terms:
                if term in haystack:
                    score += 2
            if token.casefold() in haystack:
                score += 4
            if not score:
                continue
            scored.append((score, float(item.get("confidence") or 0.0), item))

        scored.sort(key=lambda item: (-item[0], -item[1], str(item[2].get("document_id") or "")))
        return [item for _, _, item in scored[: max(1, int(limit))]]
