"""Additive v2 web-card projections for ask-path routing and evidence lookup."""

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


def _loads_dict(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return {str(key): value for key, value in parsed.items()} if isinstance(parsed, dict) else {}


class WebCardV2Store:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS web_cards_v2 (
                card_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL UNIQUE,
                canonical_url TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                page_core TEXT NOT NULL DEFAULT '',
                topic_core TEXT NOT NULL DEFAULT '',
                result_core TEXT NOT NULL DEFAULT '',
                limitations_core TEXT NOT NULL DEFAULT '',
                version_core TEXT NOT NULL DEFAULT '',
                when_not_to_use TEXT NOT NULL DEFAULT '',
                search_text TEXT NOT NULL DEFAULT '',
                quality_flag TEXT NOT NULL DEFAULT 'unscored',
                document_date TEXT NOT NULL DEFAULT '',
                event_date TEXT NOT NULL DEFAULT '',
                observed_at TEXT NOT NULL DEFAULT '',
                version TEXT NOT NULL DEFAULT 'web-card-v2',
                slot_coverage_json TEXT NOT NULL DEFAULT '{}',
                diagnostics_json TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS web_evidence_anchors_v2 (
                anchor_id TEXT PRIMARY KEY,
                card_id TEXT NOT NULL,
                claim_id TEXT NOT NULL DEFAULT '',
                document_id TEXT NOT NULL DEFAULT '',
                unit_id TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL DEFAULT 'web',
                source_url TEXT NOT NULL DEFAULT '',
                section_path TEXT NOT NULL DEFAULT '',
                span_locator TEXT NOT NULL DEFAULT '',
                snippet_hash TEXT NOT NULL DEFAULT '',
                evidence_role TEXT NOT NULL DEFAULT 'supporting',
                excerpt TEXT NOT NULL DEFAULT '',
                score REAL NOT NULL DEFAULT 0.0,
                document_date TEXT NOT NULL DEFAULT '',
                event_date TEXT NOT NULL DEFAULT '',
                observed_at TEXT NOT NULL DEFAULT '',
                updated_at_marker TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS web_card_claim_refs_v2 (
                card_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'supporting',
                confidence REAL NOT NULL DEFAULT 0.0,
                rank INTEGER NOT NULL DEFAULT 0,
                reason TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (card_id, claim_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS web_card_entity_refs_v2 (
                card_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                entity_name TEXT NOT NULL DEFAULT '',
                entity_type TEXT NOT NULL DEFAULT '',
                weight REAL NOT NULL DEFAULT 0.0,
                role TEXT NOT NULL DEFAULT 'concept',
                PRIMARY KEY (card_id, entity_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_web_cards_v2_updated_at
            ON web_cards_v2(updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_web_evidence_anchors_v2_card
            ON web_evidence_anchors_v2(card_id, claim_id, score DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_web_card_claim_refs_v2_card
            ON web_card_claim_refs_v2(card_id, rank ASC, confidence DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_web_card_entity_refs_v2_entity
            ON web_card_entity_refs_v2(entity_id, weight DESC)
            """
        )
        _add_column_if_missing(
            self.conn,
            "web_cards_v2",
            "slot_coverage_json",
            "slot_coverage_json TEXT NOT NULL DEFAULT '{}'",
        )
        _add_column_if_missing(
            self.conn,
            "web_cards_v2",
            "diagnostics_json",
            "diagnostics_json TEXT NOT NULL DEFAULT '{}'",
        )
        _add_column_if_missing(
            self.conn,
            "web_evidence_anchors_v2",
            "claim_id",
            "claim_id TEXT NOT NULL DEFAULT ''",
        )
        add_derivative_lifecycle_columns(self.conn, "web_cards_v2", _add_column_if_missing)
        self.conn.commit()

    def _row_to_card(self, row) -> dict[str, Any]:
        item = dict(row)
        item["slot_coverage"] = _loads_dict(item.get("slot_coverage_json"))
        item["diagnostics"] = _loads_dict(item.get("diagnostics_json"))
        item["stale"] = bool(item.get("stale"))
        return item

    def upsert_card(self, *, card: dict[str, Any]) -> dict[str, Any]:
        payload = dict(card or {})
        document_id = str(payload.get("document_id") or "")
        source_hash = resolve_source_content_hash(self.conn, payload, document_id=document_id)
        self.conn.execute(
            """
            INSERT INTO web_cards_v2 (
                card_id, document_id, canonical_url, title, page_core, topic_core, result_core,
                limitations_core, version_core, when_not_to_use, search_text, quality_flag,
                document_date, event_date, observed_at, version, slot_coverage_json, diagnostics_json,
                source_content_hash, stale, stale_reason, invalidated_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), CURRENT_TIMESTAMP)
            ON CONFLICT(document_id) DO UPDATE SET
                card_id=excluded.card_id,
                canonical_url=excluded.canonical_url,
                title=excluded.title,
                page_core=excluded.page_core,
                topic_core=excluded.topic_core,
                result_core=excluded.result_core,
                limitations_core=excluded.limitations_core,
                version_core=excluded.version_core,
                when_not_to_use=excluded.when_not_to_use,
                search_text=excluded.search_text,
                quality_flag=excluded.quality_flag,
                document_date=excluded.document_date,
                event_date=excluded.event_date,
                observed_at=excluded.observed_at,
                version=excluded.version,
                slot_coverage_json=excluded.slot_coverage_json,
                diagnostics_json=excluded.diagnostics_json,
                source_content_hash=excluded.source_content_hash,
                stale=excluded.stale,
                stale_reason=excluded.stale_reason,
                invalidated_at=excluded.invalidated_at,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                str(payload.get("card_id") or ""),
                document_id,
                str(payload.get("canonical_url") or ""),
                str(payload.get("title") or ""),
                str(payload.get("page_core") or ""),
                str(payload.get("topic_core") or ""),
                str(payload.get("result_core") or ""),
                str(payload.get("limitations_core") or ""),
                str(payload.get("version_core") or ""),
                str(payload.get("when_not_to_use") or ""),
                str(payload.get("search_text") or ""),
                str(payload.get("quality_flag") or "unscored"),
                str(payload.get("document_date") or ""),
                str(payload.get("event_date") or ""),
                str(payload.get("observed_at") or ""),
                str(payload.get("version") or "web-card-v2"),
                json.dumps(dict(payload.get("slot_coverage") or {}), ensure_ascii=False),
                json.dumps(dict(payload.get("diagnostics") or {}), ensure_ascii=False),
                source_hash,
                fresh_stale_value(payload),
                stale_reason_value(payload),
                invalidated_at_value(payload),
                str(payload.get("created_at") or "") or None,
            ),
        )
        self.conn.commit()
        return self.get_card(str(payload.get("document_id") or "")) or {}

    def get_card(self, document_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM web_cards_v2 WHERE document_id = ?",
            (str(document_id or "").strip(),),
        ).fetchone()
        return self._row_to_card(row) if row else None

    def get_card_by_url(self, canonical_url: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM web_cards_v2 WHERE canonical_url = ?",
            (str(canonical_url or "").strip(),),
        ).fetchone()
        return self._row_to_card(row) if row else None

    def list_cards(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM web_cards_v2 ORDER BY updated_at DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [self._row_to_card(row) for row in rows]

    def search_cards(self, query: str, *, limit: int = 20, document_ids: list[str] | None = None) -> list[dict[str, Any]]:
        token = str(query or "").strip()
        if not token and not document_ids:
            return []
        clauses: list[str] = ["COALESCE(stale, 0) = 0"]
        params: list[Any] = []
        if document_ids:
            normalized_ids = [str(item).strip() for item in document_ids if str(item).strip()]
            if normalized_ids:
                placeholders = ", ".join("?" for _ in normalized_ids)
                clauses.append(f"document_id IN ({placeholders})")
                params.extend(normalized_ids)
        if token:
            clauses.append(
                """
                (
                    title LIKE ?
                    OR page_core LIKE ?
                    OR topic_core LIKE ?
                    OR result_core LIKE ?
                    OR limitations_core LIKE ?
                    OR version_core LIKE ?
                    OR when_not_to_use LIKE ?
                    OR search_text LIKE ?
                    OR canonical_url LIKE ?
                )
                """
            )
            params.extend([f"%{token}%"] * 9)
        sql = """
            SELECT *,
                   (
                       CASE WHEN title LIKE ? THEN 6 ELSE 0 END +
                       CASE WHEN page_core LIKE ? THEN 4 ELSE 0 END +
                       CASE WHEN topic_core LIKE ? THEN 4 ELSE 0 END +
                       CASE WHEN result_core LIKE ? THEN 3 ELSE 0 END +
                       CASE WHEN limitations_core LIKE ? THEN 2 ELSE 0 END +
                       CASE WHEN version_core LIKE ? THEN 2 ELSE 0 END +
                       CASE WHEN when_not_to_use LIKE ? THEN 1 ELSE 0 END +
                       CASE WHEN search_text LIKE ? THEN 8 ELSE 0 END +
                       CASE WHEN canonical_url LIKE ? THEN 5 ELSE 0 END
                   ) AS match_score
            FROM web_cards_v2
        """
        match_params = [f"%{token}%"] * 9 if token else [""] * 9
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY match_score DESC, updated_at DESC LIMIT ?"
        rows = self.conn.execute(sql, tuple([*match_params, *params, max(1, int(limit))])).fetchall()
        return [self._row_to_card(row) for row in rows]

    def replace_claim_refs(self, *, card_id: str, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        token = str(card_id or "").strip()
        self.conn.execute("DELETE FROM web_card_claim_refs_v2 WHERE card_id = ?", (token,))
        for rank, ref in enumerate(list(refs or []), 1):
            payload = dict(ref or {})
            self.conn.execute(
                """
                INSERT INTO web_card_claim_refs_v2 (card_id, claim_id, role, confidence, rank, reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    token,
                    str(payload.get("claim_id") or ""),
                    str(payload.get("role") or "supporting"),
                    float(payload.get("confidence") or 0.0),
                    int(payload.get("rank") or rank),
                    str(payload.get("reason") or ""),
                ),
            )
        self.conn.commit()
        return self.list_claim_refs(card_id=token)

    def list_claim_refs(self, *, card_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM web_card_claim_refs_v2
            WHERE card_id = ?
            ORDER BY rank ASC, confidence DESC, claim_id ASC
            """,
            (str(card_id or "").strip(),),
        ).fetchall()
        return [dict(row) for row in rows]

    def replace_anchors(self, *, card_id: str, anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        token = str(card_id or "").strip()
        self.conn.execute("DELETE FROM web_evidence_anchors_v2 WHERE card_id = ?", (token,))
        for anchor in list(anchors or []):
            payload = dict(anchor or {})
            self.conn.execute(
                """
                INSERT INTO web_evidence_anchors_v2 (
                    anchor_id, card_id, claim_id, document_id, unit_id, title, source_type, source_url, section_path,
                    span_locator, snippet_hash, evidence_role, excerpt, score, document_date, event_date,
                    observed_at, updated_at_marker, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), CURRENT_TIMESTAMP)
                """,
                (
                    str(payload.get("anchor_id") or ""),
                    token,
                    str(payload.get("claim_id") or ""),
                    str(payload.get("document_id") or ""),
                    str(payload.get("unit_id") or ""),
                    str(payload.get("title") or ""),
                    str(payload.get("source_type") or "web"),
                    str(payload.get("source_url") or ""),
                    str(payload.get("section_path") or ""),
                    str(payload.get("span_locator") or ""),
                    str(payload.get("snippet_hash") or ""),
                    str(payload.get("evidence_role") or "supporting"),
                    str(payload.get("excerpt") or ""),
                    float(payload.get("score") or 0.0),
                    str(payload.get("document_date") or ""),
                    str(payload.get("event_date") or ""),
                    str(payload.get("observed_at") or ""),
                    str(payload.get("updated_at_marker") or ""),
                    str(payload.get("created_at") or "") or None,
                ),
            )
        self.conn.commit()
        return self.list_anchors(card_id=token)

    def list_anchors(self, *, card_id: str, claim_ids: list[str] | None = None) -> list[dict[str, Any]]:
        clauses = ["card_id = ?"]
        params: list[Any] = [str(card_id or "").strip()]
        normalized_claim_ids = [str(item).strip() for item in claim_ids or [] if str(item).strip()]
        if normalized_claim_ids:
            placeholders = ", ".join("?" for _ in normalized_claim_ids)
            clauses.append(f"(claim_id = '' OR claim_id IN ({placeholders}))")
            params.extend(normalized_claim_ids)
        rows = self.conn.execute(
            f"""
            SELECT * FROM web_evidence_anchors_v2
            WHERE {' AND '.join(clauses)}
            ORDER BY score DESC, section_path ASC, anchor_id ASC
            """,
            tuple(params),
        ).fetchall()
        return [dict(row) for row in rows]

    def replace_entity_refs(self, *, card_id: str, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        token = str(card_id or "").strip()
        self.conn.execute("DELETE FROM web_card_entity_refs_v2 WHERE card_id = ?", (token,))
        for ref in list(refs or []):
            payload = dict(ref or {})
            self.conn.execute(
                """
                INSERT INTO web_card_entity_refs_v2 (card_id, entity_id, entity_name, entity_type, weight, role)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    token,
                    str(payload.get("entity_id") or ""),
                    str(payload.get("entity_name") or ""),
                    str(payload.get("entity_type") or ""),
                    float(payload.get("weight") or 0.0),
                    str(payload.get("role") or "concept"),
                ),
            )
        self.conn.commit()
        return self.list_entity_refs(card_id=token)

    def list_entity_refs(self, *, card_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM web_card_entity_refs_v2
            WHERE card_id = ?
            ORDER BY weight DESC, entity_name ASC, entity_id ASC
            """,
            (str(card_id or "").strip(),),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_cards_by_entity_ids(self, *, entity_ids: list[str], limit: int = 20) -> list[dict[str, Any]]:
        normalized_ids = [str(item).strip() for item in entity_ids if str(item).strip()]
        if not normalized_ids:
            return []
        placeholders = ", ".join("?" for _ in normalized_ids)
        rows = self.conn.execute(
            f"""
            SELECT c.*, MAX(r.weight) AS entity_match_weight
            FROM web_cards_v2 c
            JOIN web_card_entity_refs_v2 r ON r.card_id = c.card_id
            WHERE r.entity_id IN ({placeholders})
              AND COALESCE(c.stale, 0) = 0
            GROUP BY c.card_id
            ORDER BY entity_match_weight DESC, c.updated_at DESC
            LIMIT ?
            """,
            tuple([*normalized_ids, max(1, int(limit))]),
        ).fetchall()
        return [self._row_to_card(row) for row in rows]
