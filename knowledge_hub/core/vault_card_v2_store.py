"""Additive vault-card v2 projections for ask-path routing and evidence lookup."""

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
    return {str(key): value for key, value in parsed.items()} if isinstance(parsed, dict) else {}


class VaultCardV2Store:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vault_cards_v2 (
                card_id TEXT PRIMARY KEY,
                note_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL DEFAULT '',
                note_core TEXT NOT NULL DEFAULT '',
                concept_core TEXT NOT NULL DEFAULT '',
                decision_core TEXT NOT NULL DEFAULT '',
                action_core TEXT NOT NULL DEFAULT '',
                when_not_to_use TEXT NOT NULL DEFAULT '',
                search_text TEXT NOT NULL DEFAULT '',
                quality_flag TEXT NOT NULL DEFAULT 'unscored',
                file_path TEXT NOT NULL DEFAULT '',
                version TEXT NOT NULL DEFAULT 'vault-card-v2',
                slot_coverage_json TEXT NOT NULL DEFAULT '{}',
                diagnostics_json TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vault_card_claim_refs_v2 (
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
            CREATE TABLE IF NOT EXISTS vault_evidence_anchors_v2 (
                anchor_id TEXT PRIMARY KEY,
                card_id TEXT NOT NULL,
                claim_id TEXT NOT NULL DEFAULT '',
                note_id TEXT NOT NULL DEFAULT '',
                unit_id TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL DEFAULT 'vault',
                section_path TEXT NOT NULL DEFAULT '',
                span_locator TEXT NOT NULL DEFAULT '',
                snippet_hash TEXT NOT NULL DEFAULT '',
                evidence_role TEXT NOT NULL DEFAULT 'supporting',
                excerpt TEXT NOT NULL DEFAULT '',
                score REAL NOT NULL DEFAULT 0.0,
                file_path TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _add_column_if_missing(self.conn, "vault_cards_v2", "slot_coverage_json", "slot_coverage_json TEXT NOT NULL DEFAULT '{}'")
        _add_column_if_missing(self.conn, "vault_cards_v2", "diagnostics_json", "diagnostics_json TEXT NOT NULL DEFAULT '{}'")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_vault_cards_v2_updated_at ON vault_cards_v2(updated_at DESC)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_vault_card_claim_refs_v2_card ON vault_card_claim_refs_v2(card_id, rank ASC, confidence DESC)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_vault_evidence_anchors_v2_card ON vault_evidence_anchors_v2(card_id, claim_id, score DESC)")
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
            INSERT INTO vault_cards_v2 (
                card_id, note_id, title, note_core, concept_core, decision_core, action_core,
                when_not_to_use, search_text, quality_flag, file_path, version,
                slot_coverage_json, diagnostics_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), CURRENT_TIMESTAMP)
            ON CONFLICT(note_id) DO UPDATE SET
                card_id=excluded.card_id,
                title=excluded.title,
                note_core=excluded.note_core,
                concept_core=excluded.concept_core,
                decision_core=excluded.decision_core,
                action_core=excluded.action_core,
                when_not_to_use=excluded.when_not_to_use,
                search_text=excluded.search_text,
                quality_flag=excluded.quality_flag,
                file_path=excluded.file_path,
                version=excluded.version,
                slot_coverage_json=excluded.slot_coverage_json,
                diagnostics_json=excluded.diagnostics_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                str(payload.get("card_id") or ""),
                str(payload.get("note_id") or ""),
                str(payload.get("title") or ""),
                str(payload.get("note_core") or ""),
                str(payload.get("concept_core") or ""),
                str(payload.get("decision_core") or ""),
                str(payload.get("action_core") or ""),
                str(payload.get("when_not_to_use") or ""),
                str(payload.get("search_text") or ""),
                str(payload.get("quality_flag") or "unscored"),
                str(payload.get("file_path") or ""),
                str(payload.get("version") or "vault-card-v2"),
                json.dumps(dict(payload.get("slot_coverage") or {}), ensure_ascii=False),
                json.dumps(dict(payload.get("diagnostics") or {}), ensure_ascii=False),
                str(payload.get("created_at") or "") or None,
            ),
        )
        self.conn.commit()
        return self.get_card(str(payload.get("note_id") or "")) or {}

    def get_card(self, note_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM vault_cards_v2 WHERE note_id = ?", (str(note_id or "").strip(),)).fetchone()
        return self._row_to_card(row) if row else None

    def list_cards(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM vault_cards_v2 ORDER BY updated_at DESC LIMIT ?", (max(1, int(limit)),)).fetchall()
        return [self._row_to_card(row) for row in rows]

    def search_cards(self, query: str, *, limit: int = 20, note_ids: list[str] | None = None) -> list[dict[str, Any]]:
        token = str(query or "").strip()
        if not token and not note_ids:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if note_ids:
            normalized_ids = [str(item).strip() for item in note_ids if str(item).strip()]
            if normalized_ids:
                placeholders = ", ".join("?" for _ in normalized_ids)
                clauses.append(f"note_id IN ({placeholders})")
                params.extend(normalized_ids)
        if token:
            clauses.append(
                """
                (
                    title LIKE ? OR note_core LIKE ? OR concept_core LIKE ? OR decision_core LIKE ?
                    OR action_core LIKE ? OR when_not_to_use LIKE ? OR search_text LIKE ? OR file_path LIKE ?
                )
                """
            )
            params.extend([f"%{token}%"] * 8)
        sql = """
            SELECT *,
                   (
                       CASE WHEN title LIKE ? THEN 6 ELSE 0 END +
                       CASE WHEN note_core LIKE ? THEN 4 ELSE 0 END +
                       CASE WHEN concept_core LIKE ? THEN 3 ELSE 0 END +
                       CASE WHEN decision_core LIKE ? THEN 3 ELSE 0 END +
                       CASE WHEN action_core LIKE ? THEN 2 ELSE 0 END +
                       CASE WHEN when_not_to_use LIKE ? THEN 1 ELSE 0 END +
                       CASE WHEN search_text LIKE ? THEN 8 ELSE 0 END +
                       CASE WHEN file_path LIKE ? THEN 4 ELSE 0 END
                   ) AS match_score
            FROM vault_cards_v2
        """
        match_params = [f"%{token}%"] * 8 if token else [""] * 8
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY match_score DESC, updated_at DESC LIMIT ?"
        rows = self.conn.execute(sql, tuple([*match_params, *params, max(1, int(limit))])).fetchall()
        return [self._row_to_card(row) for row in rows]

    def replace_claim_refs(self, *, card_id: str, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        token = str(card_id or "").strip()
        self.conn.execute("DELETE FROM vault_card_claim_refs_v2 WHERE card_id = ?", (token,))
        for rank, ref in enumerate(list(refs or []), 1):
            payload = dict(ref or {})
            self.conn.execute(
                """
                INSERT INTO vault_card_claim_refs_v2 (card_id, claim_id, role, confidence, rank, reason)
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
            "SELECT * FROM vault_card_claim_refs_v2 WHERE card_id = ? ORDER BY rank ASC, confidence DESC, claim_id ASC",
            (str(card_id or "").strip(),),
        ).fetchall()
        return [dict(row) for row in rows]

    def replace_anchors(self, *, card_id: str, anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        token = str(card_id or "").strip()
        self.conn.execute("DELETE FROM vault_evidence_anchors_v2 WHERE card_id = ?", (token,))
        for anchor in anchors or []:
            payload = dict(anchor or {})
            self.conn.execute(
                """
                INSERT INTO vault_evidence_anchors_v2 (
                    anchor_id, card_id, claim_id, note_id, unit_id, title, source_type,
                    section_path, span_locator, snippet_hash, evidence_role, excerpt, score,
                    file_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), CURRENT_TIMESTAMP)
                """,
                (
                    str(payload.get("anchor_id") or ""),
                    token,
                    str(payload.get("claim_id") or ""),
                    str(payload.get("note_id") or ""),
                    str(payload.get("unit_id") or ""),
                    str(payload.get("title") or ""),
                    str(payload.get("source_type") or "vault"),
                    str(payload.get("section_path") or ""),
                    str(payload.get("span_locator") or ""),
                    str(payload.get("snippet_hash") or ""),
                    str(payload.get("evidence_role") or "supporting"),
                    str(payload.get("excerpt") or ""),
                    float(payload.get("score") or 0.0),
                    str(payload.get("file_path") or ""),
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
            f"SELECT * FROM vault_evidence_anchors_v2 WHERE {' AND '.join(clauses)} ORDER BY score DESC, anchor_id ASC",
            tuple(params),
        ).fetchall()
        return [dict(row) for row in rows]


__all__ = ["VaultCardV2Store"]
