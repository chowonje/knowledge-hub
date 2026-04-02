"""Paper-focused SQLite store helpers.

This module isolates paper CRUD and paper-concept projection queries from
``SQLiteDatabase`` so the facade can stay compact.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from knowledge_hub.core.paper_lanes import (
    normalize_lane_review_status,
    normalize_primary_lane,
    normalize_secondary_tags,
    now_lane_timestamp,
    serialize_secondary_tags,
)


def _add_column_if_missing(conn, table: str, column_name: str, column_sql: str) -> None:
    columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column_name in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_sql}")


class PaperStore:
    """Paper read/write operations bound to a sqlite connection."""

    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                arxiv_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                year INTEGER,
                field TEXT,
                importance INTEGER DEFAULT 3,
                notes TEXT,
                pdf_path TEXT,
                text_path TEXT,
                translated_path TEXT,
                indexed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                primary_lane TEXT,
                secondary_tags_json TEXT DEFAULT '[]',
                lane_review_status TEXT DEFAULT 'seeded',
                lane_updated_at TIMESTAMP
            )
            """
        )
        _add_column_if_missing(self.conn, "papers", "primary_lane", "primary_lane TEXT")
        _add_column_if_missing(
            self.conn,
            "papers",
            "secondary_tags_json",
            "secondary_tags_json TEXT DEFAULT '[]'",
        )
        _add_column_if_missing(
            self.conn,
            "papers",
            "lane_review_status",
            "lane_review_status TEXT DEFAULT 'seeded'",
        )
        _add_column_if_missing(
            self.conn,
            "papers",
            "lane_updated_at",
            "lane_updated_at TIMESTAMP",
        )
        self.conn.commit()

    def _table_exists(self, table_name: str) -> bool:
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return bool(row)

    def _row_to_item(self, row) -> dict[str, Any]:
        item = dict(row)
        try:
            item["primary_lane"] = normalize_primary_lane(item.get("primary_lane"))
        except ValueError:
            item["primary_lane"] = None
        item["secondary_tags"] = normalize_secondary_tags(item.get("secondary_tags_json") or "[]")
        item["secondary_tags_json"] = serialize_secondary_tags(item.get("secondary_tags"))
        try:
            item["lane_review_status"] = normalize_lane_review_status(item.get("lane_review_status"))
        except ValueError:
            item["lane_review_status"] = "seeded"
        item["lane_updated_at"] = str(item.get("lane_updated_at") or "")
        return item

    def _normalized_lane_payload(self, paper: dict[str, Any]) -> dict[str, Any]:
        payload = dict(paper or {})
        existing = self.get_paper(str(payload.get("arxiv_id") or "")) if payload.get("arxiv_id") else None
        if "primary_lane" in payload:
            primary_lane = normalize_primary_lane(payload.get("primary_lane"))
        elif existing is not None:
            primary_lane = normalize_primary_lane(existing.get("primary_lane"))
        else:
            primary_lane = None

        if "secondary_tags" in payload:
            secondary_tags_json = serialize_secondary_tags(payload.get("secondary_tags"))
        elif "secondary_tags_json" in payload:
            secondary_tags_json = serialize_secondary_tags(payload.get("secondary_tags_json"))
        elif existing is not None:
            secondary_tags_json = serialize_secondary_tags(existing.get("secondary_tags") or existing.get("secondary_tags_json"))
        else:
            secondary_tags_json = "[]"

        if "lane_review_status" in payload:
            lane_review_status = normalize_lane_review_status(payload.get("lane_review_status"))
        elif existing is not None:
            lane_review_status = normalize_lane_review_status(existing.get("lane_review_status"))
        else:
            lane_review_status = "seeded"

        lane_keys_present = any(
            key in payload for key in ("primary_lane", "secondary_tags", "secondary_tags_json", "lane_review_status")
        )
        lane_updated_at = payload.get("lane_updated_at")
        if lane_updated_at is None:
            if lane_keys_present:
                lane_updated_at = now_lane_timestamp()
            elif existing is not None:
                lane_updated_at = existing.get("lane_updated_at") or ""
            else:
                lane_updated_at = ""
        payload["primary_lane"] = primary_lane
        payload["secondary_tags_json"] = secondary_tags_json
        payload["lane_review_status"] = lane_review_status
        payload["lane_updated_at"] = str(lane_updated_at or "")
        return payload

    def upsert_paper(self, paper: dict[str, Any]) -> None:
        payload = self._normalized_lane_payload(paper)
        self.conn.execute(
            """INSERT INTO papers (
                   arxiv_id, title, authors, year, field, importance, notes, pdf_path, text_path, translated_path,
                   primary_lane, secondary_tags_json, lane_review_status, lane_updated_at
               )
               VALUES (
                   :arxiv_id, :title, :authors, :year, :field, :importance, :notes, :pdf_path, :text_path, :translated_path,
                   :primary_lane, :secondary_tags_json, :lane_review_status, :lane_updated_at
               )
               ON CONFLICT(arxiv_id) DO UPDATE SET
                 title=excluded.title, authors=excluded.authors, year=excluded.year,
                 field=excluded.field, importance=excluded.importance, notes=excluded.notes,
                 pdf_path=excluded.pdf_path, text_path=excluded.text_path,
                 translated_path=excluded.translated_path,
                 primary_lane=COALESCE(excluded.primary_lane, primary_lane),
                 secondary_tags_json=COALESCE(excluded.secondary_tags_json, secondary_tags_json),
                 lane_review_status=COALESCE(excluded.lane_review_status, lane_review_status),
                 lane_updated_at=COALESCE(excluded.lane_updated_at, lane_updated_at)""",
            payload,
        )
        self.conn.commit()

    def get_paper(self, arxiv_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
        return self._row_to_item(row) if row else None

    def list_papers(self, field: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query = "SELECT * FROM papers WHERE 1=1"
        params: list[Any] = []
        if field:
            query += " AND field = ?"
            params.append(field)
        query += " ORDER BY year DESC, importance DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_item(r) for r in rows]

    def search_papers(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE title LIKE ? OR authors LIKE ? OR field LIKE ? ORDER BY importance DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def update_paper_lane_metadata(
        self,
        *,
        arxiv_id: str,
        primary_lane: str | None,
        secondary_tags: Any = None,
        lane_review_status: str = "seeded",
        lane_updated_at: str | None = None,
    ) -> dict[str, Any] | None:
        normalized_lane = normalize_primary_lane(primary_lane)
        normalized_tags_json = serialize_secondary_tags(secondary_tags)
        normalized_status = normalize_lane_review_status(lane_review_status)
        updated_at = str(lane_updated_at or now_lane_timestamp())
        self.conn.execute(
            """
            UPDATE papers
               SET primary_lane = ?,
                   secondary_tags_json = ?,
                   lane_review_status = ?,
                   lane_updated_at = ?
             WHERE arxiv_id = ?
            """,
            (
                normalized_lane,
                normalized_tags_json,
                normalized_status,
                updated_at,
                str(arxiv_id or ""),
            ),
        )
        self.conn.commit()
        return self.get_paper(arxiv_id)

    def get_concept_papers(self, concept_id: str) -> list[dict[str, Any]]:
        if not self._table_exists("ontology_relations"):
            return []
        rows = self.conn.execute(
            """SELECT p.*, r.confidence, r.reason_json, r.evidence_ptrs_json
               FROM ontology_relations r
               JOIN papers p ON r.source_id = p.arxiv_id
               WHERE r.source_type='paper'
                 AND r.predicate_id='uses'
                 AND r.target_type='concept'
                 AND r.target_id=?
               ORDER BY r.confidence DESC""",
            (concept_id,),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["evidence_text"] = ""
            if item.get("reason_json"):
                try:
                    reason = json.loads(item.get("reason_json") or "{}")
                except Exception:
                    reason = {}
                item["evidence_text"] = str(
                    reason.get("legacy_evidence_text") or reason.get("legacy_evidence") or ""
                )[:1000]
            result.append(item)
        return result

    def get_paper_concepts(self, arxiv_id: str) -> list[dict[str, Any]]:
        if not self._table_exists("ontology_relations"):
            return []
        paper_entity_id = f"paper:{arxiv_id}"
        rows = self.conn.execute(
            """SELECT e.*, r.confidence, r.reason_json, r.evidence_ptrs_json
               FROM ontology_relations r
               JOIN ontology_entities e ON r.target_entity_id = e.entity_id
               WHERE r.source_type='paper'
                 AND r.predicate_id='uses'
                 AND r.target_type='concept'
                 AND e.entity_type='concept'
                 AND (r.source_id=? OR r.source_entity_id=?)
               ORDER BY r.confidence DESC""",
            (arxiv_id, paper_entity_id),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item.setdefault("id", item.get("entity_id", ""))
            result.append(item)
        return result
