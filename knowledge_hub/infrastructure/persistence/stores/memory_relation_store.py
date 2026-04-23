"""Internal memory-relation storage helpers."""

from __future__ import annotations

import json
from typing import Any


def _loads_dict(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _add_column_if_missing(conn, table: str, column_name: str, column_sql: str) -> None:
    columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column_name in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_sql}")


class MemoryRelationStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_relations (
                relation_id TEXT PRIMARY KEY,
                src_form TEXT NOT NULL,
                src_id TEXT NOT NULL,
                dst_form TEXT NOT NULL,
                dst_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                provenance_json TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_relations_src
            ON memory_relations(src_form, src_id, relation_type, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_relations_dst
            ON memory_relations(dst_form, dst_id, relation_type, updated_at DESC)
            """
        )
        _add_column_if_missing(
            self.conn,
            "memory_relations",
            "origin",
            "origin TEXT NOT NULL DEFAULT 'derived' CHECK(origin IN ('derived','manual','pending'))",
        )
        self.conn.commit()

    def _row_to_item(self, row) -> dict[str, Any]:
        item = dict(row)
        item["provenance"] = _loads_dict(item.get("provenance_json"))
        return item

    def upsert_relation(
        self,
        *,
        relation_id: str,
        src_form: str,
        src_id: str,
        dst_form: str,
        dst_id: str,
        relation_type: str,
        confidence: float = 0.0,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        self.conn.execute(
            """
            INSERT INTO memory_relations (
                relation_id, src_form, src_id, dst_form, dst_id, relation_type,
                confidence, provenance_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(relation_id) DO UPDATE SET
                src_form=excluded.src_form,
                src_id=excluded.src_id,
                dst_form=excluded.dst_form,
                dst_id=excluded.dst_id,
                relation_type=excluded.relation_type,
                confidence=excluded.confidence,
                provenance_json=excluded.provenance_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                str(relation_id or "").strip(),
                str(src_form or "").strip(),
                str(src_id or "").strip(),
                str(dst_form or "").strip(),
                str(dst_id or "").strip(),
                str(relation_type or "").strip(),
                float(confidence or 0.0),
                json.dumps(dict(provenance or {}), ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return self.get_relation(str(relation_id or "").strip())

    def get_relation(self, relation_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM memory_relations WHERE relation_id = ?",
            (str(relation_id or "").strip(),),
        ).fetchone()
        return self._row_to_item(row) if row else None

    def list_relations(
        self,
        *,
        src_form: str = "",
        src_id: str = "",
        dst_form: str = "",
        dst_id: str = "",
        relation_type: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if str(src_form or "").strip():
            clauses.append("src_form = ?")
            params.append(str(src_form).strip())
        if str(src_id or "").strip():
            clauses.append("src_id = ?")
            params.append(str(src_id).strip())
        if str(dst_form or "").strip():
            clauses.append("dst_form = ?")
            params.append(str(dst_form).strip())
        if str(dst_id or "").strip():
            clauses.append("dst_id = ?")
            params.append(str(dst_id).strip())
        if str(relation_type or "").strip():
            clauses.append("relation_type = ?")
            params.append(str(relation_type).strip())
        where_clause = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM memory_relations{where_clause} ORDER BY updated_at DESC, relation_id ASC LIMIT ?",
            tuple([*params, max(1, int(limit))]),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def delete_relations_for_node(
        self,
        *,
        form: str,
        node_id: str,
        relation_type: str = "",
        direction: str = "src",
    ) -> int:
        field = "src" if str(direction or "src").strip().lower() != "dst" else "dst"
        clauses = [f"{field}_form = ?", f"{field}_id = ?"]
        params: list[Any] = [str(form or "").strip(), str(node_id or "").strip()]
        if str(relation_type or "").strip():
            clauses.append("relation_type = ?")
            params.append(str(relation_type).strip())
        cursor = self.conn.execute(
            f"DELETE FROM memory_relations WHERE {' AND '.join(clauses)}",
            tuple(params),
        )
        self.conn.commit()
        return int(cursor.rowcount or 0)


__all__ = ["MemoryRelationStore"]
