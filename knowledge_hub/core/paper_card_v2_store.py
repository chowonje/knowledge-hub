"""Additive v2 paper-card projections for ask-path routing and evidence lookup."""

from __future__ import annotations

import json
from typing import Any


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
    if not isinstance(parsed, dict):
        return {}
    return {str(key): value for key, value in parsed.items()}


def _loads_list(raw: Any) -> list[Any]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return list(raw)
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    return list(parsed) if isinstance(parsed, list) else []


class PaperCardV2Store:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_cards_v2 (
                card_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL DEFAULT '',
                paper_core TEXT NOT NULL DEFAULT '',
                problem_core TEXT NOT NULL DEFAULT '',
                method_core TEXT NOT NULL DEFAULT '',
                result_core TEXT NOT NULL DEFAULT '',
                limitations_core TEXT NOT NULL DEFAULT '',
                dataset_core TEXT NOT NULL DEFAULT '',
                metric_core TEXT NOT NULL DEFAULT '',
                when_not_to_use TEXT NOT NULL DEFAULT '',
                source_memory_id TEXT NOT NULL DEFAULT '',
                search_text TEXT NOT NULL DEFAULT '',
                quality_flag TEXT NOT NULL DEFAULT 'unscored',
                published_at TEXT NOT NULL DEFAULT '',
                version TEXT NOT NULL DEFAULT 'paper-card-v2',
                slot_coverage_json TEXT NOT NULL DEFAULT '{}',
                diagnostics_json TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _add_column_if_missing(
            self.conn,
            "paper_cards_v2",
            "slot_coverage_json",
            "slot_coverage_json TEXT NOT NULL DEFAULT '{}'",
        )
        _add_column_if_missing(
            self.conn,
            "paper_cards_v2",
            "diagnostics_json",
            "diagnostics_json TEXT NOT NULL DEFAULT '{}'",
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_card_claim_refs_v2 (
                card_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'supporting',
                slot_key TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                rank INTEGER NOT NULL DEFAULT 0,
                reason TEXT NOT NULL DEFAULT '',
                normalization_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (card_id, claim_id)
            )
            """
        )
        _add_column_if_missing(
            self.conn,
            "paper_card_claim_refs_v2",
            "slot_key",
            "slot_key TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "paper_card_claim_refs_v2",
            "normalization_json",
            "normalization_json TEXT NOT NULL DEFAULT '{}'",
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_anchors_v2 (
                anchor_id TEXT PRIMARY KEY,
                card_id TEXT NOT NULL,
                claim_id TEXT NOT NULL DEFAULT '',
                paper_id TEXT NOT NULL DEFAULT '',
                document_id TEXT NOT NULL DEFAULT '',
                unit_id TEXT NOT NULL DEFAULT '',
                chunk_id TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL DEFAULT 'paper',
                section_path TEXT NOT NULL DEFAULT '',
                span_locator TEXT NOT NULL DEFAULT '',
                snippet_hash TEXT NOT NULL DEFAULT '',
                evidence_role TEXT NOT NULL DEFAULT 'supporting',
                excerpt TEXT NOT NULL DEFAULT '',
                score REAL NOT NULL DEFAULT 0.0,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_card_entity_refs_v2 (
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
            CREATE INDEX IF NOT EXISTS idx_paper_cards_v2_updated_at
            ON paper_cards_v2(updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_card_claim_refs_v2_card
            ON paper_card_claim_refs_v2(card_id, rank ASC, confidence DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_evidence_anchors_v2_card
            ON evidence_anchors_v2(card_id, claim_id, score DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_card_entity_refs_v2_entity
            ON paper_card_entity_refs_v2(entity_id, weight DESC)
            """
        )
        self.conn.commit()

    def _row_to_card(self, row) -> dict[str, Any]:
        item = dict(row)
        item["slot_coverage"] = _loads_dict(item.get("slot_coverage_json"))
        item["diagnostics"] = _loads_dict(item.get("diagnostics_json"))
        return item

    def upsert_card(self, *, card: dict[str, Any]) -> dict[str, Any]:
        payload = dict(card or {})
        self.conn.execute(
            """
            INSERT INTO paper_cards_v2 (
                card_id, paper_id, title, paper_core, problem_core, method_core, result_core,
                limitations_core, dataset_core, metric_core, when_not_to_use, source_memory_id,
                search_text, quality_flag, published_at, version, slot_coverage_json,
                diagnostics_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), CURRENT_TIMESTAMP)
            ON CONFLICT(paper_id) DO UPDATE SET
                card_id=excluded.card_id,
                title=excluded.title,
                paper_core=excluded.paper_core,
                problem_core=excluded.problem_core,
                method_core=excluded.method_core,
                result_core=excluded.result_core,
                limitations_core=excluded.limitations_core,
                dataset_core=excluded.dataset_core,
                metric_core=excluded.metric_core,
                when_not_to_use=excluded.when_not_to_use,
                source_memory_id=excluded.source_memory_id,
                search_text=excluded.search_text,
                quality_flag=excluded.quality_flag,
                published_at=excluded.published_at,
                version=excluded.version,
                slot_coverage_json=excluded.slot_coverage_json,
                diagnostics_json=excluded.diagnostics_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                str(payload.get("card_id") or ""),
                str(payload.get("paper_id") or ""),
                str(payload.get("title") or ""),
                str(payload.get("paper_core") or ""),
                str(payload.get("problem_core") or ""),
                str(payload.get("method_core") or ""),
                str(payload.get("result_core") or ""),
                str(payload.get("limitations_core") or ""),
                str(payload.get("dataset_core") or ""),
                str(payload.get("metric_core") or ""),
                str(payload.get("when_not_to_use") or ""),
                str(payload.get("source_memory_id") or ""),
                str(payload.get("search_text") or ""),
                str(payload.get("quality_flag") or "unscored"),
                str(payload.get("published_at") or ""),
                str(payload.get("version") or "paper-card-v2"),
                json.dumps(dict(payload.get("slot_coverage") or {}), ensure_ascii=False),
                json.dumps(dict(payload.get("diagnostics") or {}), ensure_ascii=False),
                str(payload.get("created_at") or "") or None,
            ),
        )
        self.conn.commit()
        return self.get_card(str(payload.get("paper_id") or "")) or {}

    def get_card(self, paper_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM paper_cards_v2 WHERE paper_id = ?",
            (str(paper_id or "").strip(),),
        ).fetchone()
        return self._row_to_card(row) if row else None

    def list_cards(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM paper_cards_v2 ORDER BY updated_at DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [self._row_to_card(row) for row in rows]

    def search_cards(
        self,
        query: str,
        *,
        limit: int = 20,
        paper_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        token = str(query or "").strip()
        if not token and not paper_ids:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if paper_ids:
            normalized_ids = [str(item).strip() for item in paper_ids if str(item).strip()]
            if normalized_ids:
                placeholders = ", ".join("?" for _ in normalized_ids)
                clauses.append(f"paper_id IN ({placeholders})")
                params.extend(normalized_ids)
        if token:
            match_clause = """
            (
                title LIKE ?
                OR paper_core LIKE ?
                OR problem_core LIKE ?
                OR method_core LIKE ?
                OR result_core LIKE ?
                OR limitations_core LIKE ?
                OR dataset_core LIKE ?
                OR metric_core LIKE ?
                OR when_not_to_use LIKE ?
                OR search_text LIKE ?
            )
            """
            clauses.append(match_clause)
            params.extend([f"%{token}%"] * 10)
        query_sql = """
            SELECT *,
                   (
                       CASE WHEN title LIKE ? THEN 6 ELSE 0 END +
                       CASE WHEN paper_core LIKE ? THEN 4 ELSE 0 END +
                       CASE WHEN problem_core LIKE ? THEN 2 ELSE 0 END +
                       CASE WHEN method_core LIKE ? THEN 3 ELSE 0 END +
                       CASE WHEN result_core LIKE ? THEN 3 ELSE 0 END +
                       CASE WHEN limitations_core LIKE ? THEN 2 ELSE 0 END +
                       CASE WHEN dataset_core LIKE ? THEN 2 ELSE 0 END +
                       CASE WHEN metric_core LIKE ? THEN 2 ELSE 0 END +
                       CASE WHEN when_not_to_use LIKE ? THEN 1 ELSE 0 END +
                       CASE WHEN search_text LIKE ? THEN 8 ELSE 0 END
                   ) AS match_score
            FROM paper_cards_v2
        """
        match_params = [f"%{token}%"] * 10 if token else [""] * 10
        if clauses:
            query_sql += " WHERE " + " AND ".join(clauses)
        query_sql += " ORDER BY match_score DESC, updated_at DESC LIMIT ?"
        rows = self.conn.execute(query_sql, tuple([*match_params, *params, max(1, int(limit))])).fetchall()
        return [self._row_to_card(row) for row in rows]

    def replace_claim_refs(self, *, card_id: str, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        token = str(card_id or "").strip()
        self.conn.execute("DELETE FROM paper_card_claim_refs_v2 WHERE card_id = ?", (token,))
        for rank, ref in enumerate(list(refs or []), 1):
            payload = dict(ref or {})
            self.conn.execute(
                """
                INSERT INTO paper_card_claim_refs_v2 (card_id, claim_id, role, slot_key, confidence, rank, reason, normalization_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token,
                    str(payload.get("claim_id") or ""),
                    str(payload.get("role") or "supporting"),
                    str(payload.get("slot_key") or ""),
                    float(payload.get("confidence") or 0.0),
                    int(payload.get("rank") or rank),
                    str(payload.get("reason") or ""),
                    json.dumps(dict(payload.get("normalization") or {}), ensure_ascii=False),
                ),
            )
        self.conn.commit()
        return self.list_claim_refs(card_id=token)

    def list_claim_refs(self, *, card_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM paper_card_claim_refs_v2
            WHERE card_id = ?
            ORDER BY rank ASC, confidence DESC, claim_id ASC
            """,
            (str(card_id or "").strip(),),
        ).fetchall()
        payload: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["normalization"] = _loads_dict(item.get("normalization_json"))
            payload.append(item)
        return payload

    def replace_anchors(self, *, card_id: str, anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        token = str(card_id or "").strip()
        self.conn.execute("DELETE FROM evidence_anchors_v2 WHERE card_id = ?", (token,))
        for anchor in list(anchors or []):
            payload = dict(anchor or {})
            self.conn.execute(
                """
                INSERT INTO evidence_anchors_v2 (
                    anchor_id, card_id, claim_id, paper_id, document_id, unit_id, chunk_id, title,
                    source_type, section_path, span_locator, snippet_hash, evidence_role, excerpt, score,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), CURRENT_TIMESTAMP)
                """,
                (
                    str(payload.get("anchor_id") or ""),
                    token,
                    str(payload.get("claim_id") or ""),
                    str(payload.get("paper_id") or ""),
                    str(payload.get("document_id") or ""),
                    str(payload.get("unit_id") or ""),
                    str(payload.get("chunk_id") or ""),
                    str(payload.get("title") or ""),
                    str(payload.get("source_type") or "paper"),
                    str(payload.get("section_path") or ""),
                    str(payload.get("span_locator") or ""),
                    str(payload.get("snippet_hash") or ""),
                    str(payload.get("evidence_role") or "supporting"),
                    str(payload.get("excerpt") or ""),
                    float(payload.get("score") or 0.0),
                    str(payload.get("created_at") or "") or None,
                ),
            )
        self.conn.commit()
        return self.list_anchors(card_id=token)

    def list_anchors(self, *, card_id: str, claim_ids: list[str] | None = None) -> list[dict[str, Any]]:
        token = str(card_id or "").strip()
        query = "SELECT * FROM evidence_anchors_v2 WHERE card_id = ?"
        params: list[Any] = [token]
        if claim_ids:
            normalized_ids = [str(item).strip() for item in claim_ids if str(item).strip()]
            if normalized_ids:
                placeholders = ", ".join("?" for _ in normalized_ids)
                query += f" AND (claim_id = '' OR claim_id IN ({placeholders}))"
                params.extend(normalized_ids)
        query += " ORDER BY score DESC, section_path ASC, anchor_id ASC"
        rows = self.conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def replace_entity_refs(self, *, card_id: str, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        token = str(card_id or "").strip()
        self.conn.execute("DELETE FROM paper_card_entity_refs_v2 WHERE card_id = ?", (token,))
        for ref in list(refs or []):
            payload = dict(ref or {})
            self.conn.execute(
                """
                INSERT INTO paper_card_entity_refs_v2 (card_id, entity_id, entity_name, entity_type, weight, role)
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
            SELECT * FROM paper_card_entity_refs_v2
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
            FROM paper_cards_v2 c
            JOIN paper_card_entity_refs_v2 r ON r.card_id = c.card_id
            WHERE r.entity_id IN ({placeholders})
            GROUP BY c.card_id
            ORDER BY entity_match_weight DESC, c.updated_at DESC
            LIMIT ?
            """,
            tuple([*normalized_ids, max(1, int(limit))]),
        ).fetchall()
        return [self._row_to_card(row) for row in rows]
