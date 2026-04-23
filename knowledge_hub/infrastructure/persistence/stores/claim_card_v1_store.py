"""Additive claim-card v1 projections for claim-first ask-time reasoning."""

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


class ClaimCardV1Store:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS claim_cards_v1 (
                claim_card_id TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                source_id TEXT NOT NULL DEFAULT '',
                document_id TEXT NOT NULL DEFAULT '',
                paper_id TEXT NOT NULL DEFAULT '',
                claim_text TEXT NOT NULL DEFAULT '',
                claim_type TEXT NOT NULL DEFAULT 'unknown',
                status TEXT NOT NULL DEFAULT 'unknown',
                summary_text TEXT NOT NULL DEFAULT '',
                scope_text TEXT NOT NULL DEFAULT '',
                condition_text TEXT NOT NULL DEFAULT '',
                limitation_text TEXT NOT NULL DEFAULT '',
                negative_scope_text TEXT NOT NULL DEFAULT '',
                task TEXT NOT NULL DEFAULT '',
                dataset TEXT NOT NULL DEFAULT '',
                metric TEXT NOT NULL DEFAULT '',
                comparator TEXT NOT NULL DEFAULT '',
                result_direction TEXT NOT NULL DEFAULT 'unknown',
                result_value_text TEXT NOT NULL DEFAULT '',
                result_value_numeric REAL,
                evidence_strength TEXT NOT NULL DEFAULT 'weak',
                evidence_anchor_ids_json TEXT NOT NULL DEFAULT '[]',
                section_paths_json TEXT NOT NULL DEFAULT '[]',
                matched_entity_ids_json TEXT NOT NULL DEFAULT '[]',
                search_text TEXT NOT NULL DEFAULT '',
                quality_flag TEXT NOT NULL DEFAULT 'unscored',
                confidence REAL NOT NULL DEFAULT 0.0,
                origin TEXT NOT NULL DEFAULT 'extracted',
                trust_level TEXT NOT NULL DEFAULT 'high',
                built_at TEXT NOT NULL DEFAULT '',
                source_updated_at_snapshot TEXT NOT NULL DEFAULT '',
                normalization_updated_at_snapshot TEXT NOT NULL DEFAULT '',
                task_canonical TEXT NOT NULL DEFAULT '',
                dataset_canonical TEXT NOT NULL DEFAULT '',
                dataset_family TEXT NOT NULL DEFAULT '',
                dataset_version TEXT NOT NULL DEFAULT '',
                metric_canonical TEXT NOT NULL DEFAULT '',
                comparator_canonical TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                stored_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (source_kind, claim_id, source_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS claim_card_source_refs_v1 (
                source_card_id TEXT NOT NULL,
                claim_card_id TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                source_id TEXT NOT NULL DEFAULT '',
                document_id TEXT NOT NULL DEFAULT '',
                paper_id TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT 'source_card',
                rank INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (source_card_id, claim_card_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS claim_card_alignment_refs_v1 (
                claim_card_id TEXT NOT NULL,
                aligned_claim_card_id TEXT NOT NULL,
                task TEXT NOT NULL DEFAULT '',
                dataset TEXT NOT NULL DEFAULT '',
                metric TEXT NOT NULL DEFAULT '',
                comparator TEXT NOT NULL DEFAULT '',
                task_canonical TEXT NOT NULL DEFAULT '',
                dataset_canonical TEXT NOT NULL DEFAULT '',
                dataset_family TEXT NOT NULL DEFAULT '',
                dataset_version TEXT NOT NULL DEFAULT '',
                metric_canonical TEXT NOT NULL DEFAULT '',
                comparator_canonical TEXT NOT NULL DEFAULT '',
                condition_text TEXT NOT NULL DEFAULT '',
                group_key TEXT NOT NULL DEFAULT '',
                family_relation_note TEXT NOT NULL DEFAULT '',
                alignment_type TEXT NOT NULL DEFAULT 'aligned',
                value_delta REAL,
                source_diversity INTEGER NOT NULL DEFAULT 0,
                evidence_order INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (claim_card_id, aligned_claim_card_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS normalization_aliases (
                canonical TEXT NOT NULL,
                alias TEXT NOT NULL,
                alias_type TEXT NOT NULL CHECK(alias_type IN ('dataset', 'metric', 'task', 'comparator')),
                dataset_family TEXT NOT NULL DEFAULT '',
                dataset_version TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (alias_type, alias)
            )
            """
        )
        _add_column_if_missing(
            self.conn,
            "claim_cards_v1",
            "updated_at",
            "updated_at TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(self.conn, "claim_cards_v1", "origin", "origin TEXT NOT NULL DEFAULT 'extracted'")
        _add_column_if_missing(self.conn, "claim_cards_v1", "trust_level", "trust_level TEXT NOT NULL DEFAULT 'high'")
        _add_column_if_missing(self.conn, "claim_cards_v1", "built_at", "built_at TEXT NOT NULL DEFAULT ''")
        _add_column_if_missing(
            self.conn,
            "claim_cards_v1",
            "source_updated_at_snapshot",
            "source_updated_at_snapshot TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "claim_cards_v1",
            "normalization_updated_at_snapshot",
            "normalization_updated_at_snapshot TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(self.conn, "claim_cards_v1", "task_canonical", "task_canonical TEXT NOT NULL DEFAULT ''")
        _add_column_if_missing(self.conn, "claim_cards_v1", "dataset_canonical", "dataset_canonical TEXT NOT NULL DEFAULT ''")
        _add_column_if_missing(self.conn, "claim_cards_v1", "dataset_family", "dataset_family TEXT NOT NULL DEFAULT ''")
        _add_column_if_missing(self.conn, "claim_cards_v1", "dataset_version", "dataset_version TEXT NOT NULL DEFAULT ''")
        _add_column_if_missing(self.conn, "claim_cards_v1", "metric_canonical", "metric_canonical TEXT NOT NULL DEFAULT ''")
        _add_column_if_missing(
            self.conn,
            "claim_cards_v1",
            "comparator_canonical",
            "comparator_canonical TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "claim_card_alignment_refs_v1",
            "task_canonical",
            "task_canonical TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "claim_card_alignment_refs_v1",
            "dataset_canonical",
            "dataset_canonical TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "claim_card_alignment_refs_v1",
            "dataset_family",
            "dataset_family TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "claim_card_alignment_refs_v1",
            "dataset_version",
            "dataset_version TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "claim_card_alignment_refs_v1",
            "metric_canonical",
            "metric_canonical TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "claim_card_alignment_refs_v1",
            "comparator_canonical",
            "comparator_canonical TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "claim_card_alignment_refs_v1",
            "condition_text",
            "condition_text TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "claim_card_alignment_refs_v1",
            "group_key",
            "group_key TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "claim_card_alignment_refs_v1",
            "family_relation_note",
            "family_relation_note TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "normalization_aliases",
            "dataset_family",
            "dataset_family TEXT NOT NULL DEFAULT ''",
        )
        _add_column_if_missing(
            self.conn,
            "normalization_aliases",
            "dataset_version",
            "dataset_version TEXT NOT NULL DEFAULT ''",
        )
        add_derivative_lifecycle_columns(self.conn, "claim_cards_v1", _add_column_if_missing)
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_claim_cards_v1_source
            ON claim_cards_v1(source_kind, source_id, stored_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_claim_cards_v1_frame
            ON claim_cards_v1(task, dataset, metric, comparator, stored_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_claim_cards_v1_canonical_frame
            ON claim_cards_v1(task_canonical, dataset_canonical, metric_canonical, comparator_canonical, stored_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_claim_card_source_refs_v1_card
            ON claim_card_source_refs_v1(source_card_id, rank ASC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_claim_card_alignment_refs_v1_card
            ON claim_card_alignment_refs_v1(claim_card_id, alignment_type, evidence_order ASC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_normalization_aliases_lookup
            ON normalization_aliases(alias_type, alias)
            """
        )
        self.conn.commit()

    def _row_to_claim_card(self, row) -> dict[str, Any]:
        item = dict(row)
        item["evidence_anchor_ids"] = _loads_list(item.get("evidence_anchor_ids_json"))
        item["section_paths"] = _loads_list(item.get("section_paths_json"))
        item["matched_entity_ids"] = _loads_list(item.get("matched_entity_ids_json"))
        item["stale"] = bool(item.get("stale"))
        return item

    def upsert_claim_card(self, *, card: dict[str, Any]) -> dict[str, Any]:
        payload = dict(card or {})
        document_id = str(payload.get("document_id") or "")
        if not document_id and str(payload.get("paper_id") or "").strip():
            document_id = f"paper:{str(payload.get('paper_id') or '').strip()}"
        source_hash = resolve_source_content_hash(self.conn, payload, document_id=document_id)
        self.conn.execute(
            """
            INSERT INTO claim_cards_v1 (
                claim_card_id, claim_id, source_kind, source_id, document_id, paper_id,
                claim_text, claim_type, status, summary_text, scope_text, condition_text,
                limitation_text, negative_scope_text, task, dataset, metric, comparator,
                result_direction, result_value_text, result_value_numeric, evidence_strength,
                evidence_anchor_ids_json, section_paths_json, matched_entity_ids_json,
                search_text, quality_flag, confidence, origin, trust_level, built_at,
                source_updated_at_snapshot, normalization_updated_at_snapshot,
                task_canonical, dataset_canonical, dataset_family, dataset_version,
                metric_canonical, comparator_canonical, updated_at, source_content_hash,
                stale, stale_reason, invalidated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_kind, claim_id, source_id) DO UPDATE SET
                claim_card_id=excluded.claim_card_id,
                document_id=excluded.document_id,
                paper_id=excluded.paper_id,
                claim_text=excluded.claim_text,
                claim_type=excluded.claim_type,
                status=excluded.status,
                summary_text=excluded.summary_text,
                scope_text=excluded.scope_text,
                condition_text=excluded.condition_text,
                limitation_text=excluded.limitation_text,
                negative_scope_text=excluded.negative_scope_text,
                task=excluded.task,
                dataset=excluded.dataset,
                metric=excluded.metric,
                comparator=excluded.comparator,
                result_direction=excluded.result_direction,
                result_value_text=excluded.result_value_text,
                result_value_numeric=excluded.result_value_numeric,
                evidence_strength=excluded.evidence_strength,
                evidence_anchor_ids_json=excluded.evidence_anchor_ids_json,
                section_paths_json=excluded.section_paths_json,
                matched_entity_ids_json=excluded.matched_entity_ids_json,
                search_text=excluded.search_text,
                quality_flag=excluded.quality_flag,
                confidence=excluded.confidence,
                origin=excluded.origin,
                trust_level=excluded.trust_level,
                built_at=excluded.built_at,
                source_updated_at_snapshot=excluded.source_updated_at_snapshot,
                normalization_updated_at_snapshot=excluded.normalization_updated_at_snapshot,
                task_canonical=excluded.task_canonical,
                dataset_canonical=excluded.dataset_canonical,
                dataset_family=excluded.dataset_family,
                dataset_version=excluded.dataset_version,
                metric_canonical=excluded.metric_canonical,
                comparator_canonical=excluded.comparator_canonical,
                updated_at=excluded.updated_at,
                source_content_hash=excluded.source_content_hash,
                stale=excluded.stale,
                stale_reason=excluded.stale_reason,
                invalidated_at=excluded.invalidated_at,
                stored_at=CURRENT_TIMESTAMP
            """,
            (
                str(payload.get("claim_card_id") or ""),
                str(payload.get("claim_id") or ""),
                str(payload.get("source_kind") or ""),
                str(payload.get("source_id") or ""),
                document_id,
                str(payload.get("paper_id") or ""),
                str(payload.get("claim_text") or ""),
                str(payload.get("claim_type") or "unknown"),
                str(payload.get("status") or "unknown"),
                str(payload.get("summary_text") or ""),
                str(payload.get("scope_text") or ""),
                str(payload.get("condition_text") or ""),
                str(payload.get("limitation_text") or ""),
                str(payload.get("negative_scope_text") or ""),
                str(payload.get("task") or ""),
                str(payload.get("dataset") or ""),
                str(payload.get("metric") or ""),
                str(payload.get("comparator") or ""),
                str(payload.get("result_direction") or "unknown"),
                str(payload.get("result_value_text") or ""),
                float(payload.get("result_value_numeric")) if payload.get("result_value_numeric") is not None else None,
                str(payload.get("evidence_strength") or "weak"),
                json.dumps(list(payload.get("evidence_anchor_ids") or []), ensure_ascii=False),
                json.dumps(list(payload.get("section_paths") or []), ensure_ascii=False),
                json.dumps(list(payload.get("matched_entity_ids") or []), ensure_ascii=False),
                str(payload.get("search_text") or ""),
                str(payload.get("quality_flag") or "unscored"),
                float(payload.get("confidence") or 0.0),
                str(payload.get("origin") or "extracted"),
                str(payload.get("trust_level") or "high"),
                str(payload.get("built_at") or ""),
                str(payload.get("source_updated_at_snapshot") or ""),
                str(payload.get("normalization_updated_at_snapshot") or ""),
                str(payload.get("task_canonical") or ""),
                str(payload.get("dataset_canonical") or ""),
                str(payload.get("dataset_family") or ""),
                str(payload.get("dataset_version") or ""),
                str(payload.get("metric_canonical") or ""),
                str(payload.get("comparator_canonical") or ""),
                str(payload.get("updated_at") or ""),
                source_hash,
                fresh_stale_value(payload),
                stale_reason_value(payload),
                invalidated_at_value(payload),
            ),
        )
        self.conn.commit()
        return self.get_claim_card(str(payload.get("claim_card_id") or "")) or {}

    def get_claim_card(self, claim_card_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM claim_cards_v1 WHERE claim_card_id = ?",
            (str(claim_card_id or "").strip(),),
        ).fetchone()
        return self._row_to_claim_card(row) if row else None

    def list_claim_cards(
        self,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
        claim_ids: list[str] | None = None,
        claim_card_ids: list[str] | None = None,
        task: str | None = None,
        dataset: str | None = None,
        metric: str | None = None,
        comparator: str | None = None,
        task_canonical: str | None = None,
        dataset_canonical: str | None = None,
        metric_canonical: str | None = None,
        comparator_canonical: str | None = None,
        limit: int = 500,
        include_stale: bool = False,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM claim_cards_v1 WHERE 1=1"
        params: list[Any] = []
        if not include_stale:
            sql += " AND COALESCE(stale, 0) = 0"
        if source_kind:
            sql += " AND source_kind = ?"
            params.append(str(source_kind).strip())
        if source_id:
            sql += " AND source_id = ?"
            params.append(str(source_id).strip())
        if task:
            sql += " AND task = ?"
            params.append(str(task).strip())
        if dataset:
            sql += " AND dataset = ?"
            params.append(str(dataset).strip())
        if metric:
            sql += " AND metric = ?"
            params.append(str(metric).strip())
        if comparator is not None:
            sql += " AND comparator = ?"
            params.append(str(comparator).strip())
        if task_canonical is not None:
            sql += " AND task_canonical = ?"
            params.append(str(task_canonical).strip())
        if dataset_canonical is not None:
            sql += " AND dataset_canonical = ?"
            params.append(str(dataset_canonical).strip())
        if metric_canonical is not None:
            sql += " AND metric_canonical = ?"
            params.append(str(metric_canonical).strip())
        if comparator_canonical is not None:
            sql += " AND comparator_canonical = ?"
            params.append(str(comparator_canonical).strip())
        if claim_ids:
            normalized = [str(item).strip() for item in claim_ids if str(item).strip()]
            if normalized:
                placeholders = ", ".join("?" for _ in normalized)
                sql += f" AND claim_id IN ({placeholders})"
                params.extend(normalized)
        if claim_card_ids:
            normalized = [str(item).strip() for item in claim_card_ids if str(item).strip()]
            if normalized:
                placeholders = ", ".join("?" for _ in normalized)
                sql += f" AND claim_card_id IN ({placeholders})"
                params.extend(normalized)
        sql += " ORDER BY confidence DESC, stored_at DESC, claim_card_id ASC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(sql, tuple(params)).fetchall()
        return [self._row_to_claim_card(row) for row in rows]

    def replace_claim_card_source_refs(self, *, source_card_id: str, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        token = str(source_card_id or "").strip()
        self.conn.execute("DELETE FROM claim_card_source_refs_v1 WHERE source_card_id = ?", (token,))
        for rank, ref in enumerate(list(refs or []), 1):
            payload = dict(ref or {})
            self.conn.execute(
                """
                INSERT INTO claim_card_source_refs_v1 (
                    source_card_id, claim_card_id, source_kind, source_id, document_id, paper_id, role, rank
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token,
                    str(payload.get("claim_card_id") or ""),
                    str(payload.get("source_kind") or ""),
                    str(payload.get("source_id") or ""),
                    str(payload.get("document_id") or ""),
                    str(payload.get("paper_id") or ""),
                    str(payload.get("role") or "source_card"),
                    int(payload.get("rank") or rank),
                ),
            )
        self.conn.commit()
        return self.list_claim_card_source_refs(source_card_id=token)

    def list_claim_card_source_refs(
        self,
        *,
        source_card_id: str | None = None,
        claim_card_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM claim_card_source_refs_v1 WHERE 1=1"
        params: list[Any] = []
        if source_card_id:
            sql += " AND source_card_id = ?"
            params.append(str(source_card_id).strip())
        if claim_card_id:
            sql += " AND claim_card_id = ?"
            params.append(str(claim_card_id).strip())
        sql += " ORDER BY rank ASC, claim_card_id ASC"
        rows = self.conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def replace_claim_card_alignment_refs(self, *, claim_card_id: str, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        token = str(claim_card_id or "").strip()
        self.conn.execute("DELETE FROM claim_card_alignment_refs_v1 WHERE claim_card_id = ?", (token,))
        for rank, ref in enumerate(list(refs or []), 1):
            payload = dict(ref or {})
            self.conn.execute(
                """
                INSERT INTO claim_card_alignment_refs_v1 (
                    claim_card_id, aligned_claim_card_id, task, dataset, metric, comparator,
                    task_canonical, dataset_canonical, dataset_family, dataset_version,
                    metric_canonical, comparator_canonical, condition_text, group_key, family_relation_note,
                    alignment_type, value_delta, source_diversity, evidence_order
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token,
                    str(payload.get("aligned_claim_card_id") or ""),
                    str(payload.get("task") or ""),
                    str(payload.get("dataset") or ""),
                    str(payload.get("metric") or ""),
                    str(payload.get("comparator") or ""),
                    str(payload.get("task_canonical") or ""),
                    str(payload.get("dataset_canonical") or ""),
                    str(payload.get("dataset_family") or ""),
                    str(payload.get("dataset_version") or ""),
                    str(payload.get("metric_canonical") or ""),
                    str(payload.get("comparator_canonical") or ""),
                    str(payload.get("condition_text") or ""),
                    str(payload.get("group_key") or ""),
                    str(payload.get("family_relation_note") or ""),
                    str(payload.get("alignment_type") or "aligned"),
                    float(payload.get("value_delta")) if payload.get("value_delta") is not None else None,
                    int(payload.get("source_diversity") or 0),
                    int(payload.get("evidence_order") or rank),
                ),
            )
        self.conn.commit()
        return self.list_claim_card_alignment_refs(claim_card_id=token)

    def list_claim_card_alignment_refs(
        self,
        *,
        claim_card_id: str | None = None,
        claim_card_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM claim_card_alignment_refs_v1 WHERE 1=1"
        params: list[Any] = []
        if claim_card_id:
            sql += " AND claim_card_id = ?"
            params.append(str(claim_card_id).strip())
        if claim_card_ids:
            normalized = [str(item).strip() for item in claim_card_ids if str(item).strip()]
            if normalized:
                placeholders = ", ".join("?" for _ in normalized)
                sql += f" AND claim_card_id IN ({placeholders})"
                params.extend(normalized)
        sql += " ORDER BY evidence_order ASC, aligned_claim_card_id ASC"
        rows = self.conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def delete_claim_cards(self, *, claim_card_ids: list[str]) -> None:
        tokens = [str(item).strip() for item in list(claim_card_ids or []) if str(item).strip()]
        if not tokens:
            return
        placeholders = ", ".join("?" for _ in tokens)
        self.conn.execute(
            f"DELETE FROM claim_card_alignment_refs_v1 WHERE claim_card_id IN ({placeholders}) OR aligned_claim_card_id IN ({placeholders})",
            tuple(tokens + tokens),
        )
        self.conn.execute(
            f"DELETE FROM claim_card_source_refs_v1 WHERE claim_card_id IN ({placeholders})",
            tuple(tokens),
        )
        self.conn.execute(
            f"DELETE FROM claim_cards_v1 WHERE claim_card_id IN ({placeholders})",
            tuple(tokens),
        )
        self.conn.commit()

    def upsert_normalization_alias(
        self,
        *,
        alias_type: str,
        alias: str,
        canonical: str,
        dataset_family: str = "",
        dataset_version: str = "",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO normalization_aliases (canonical, alias, alias_type, dataset_family, dataset_version, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(alias_type, alias) DO UPDATE SET
                canonical=excluded.canonical,
                dataset_family=excluded.dataset_family,
                dataset_version=excluded.dataset_version,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                str(canonical or "").strip(),
                str(alias or "").strip(),
                str(alias_type or "").strip(),
                str(dataset_family or "").strip(),
                str(dataset_version or "").strip(),
            ),
        )
        self.conn.commit()

    def list_normalization_aliases(self, *, alias_type: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM normalization_aliases"
        params: list[Any] = []
        if alias_type:
            sql += " WHERE alias_type = ?"
            params.append(str(alias_type).strip())
        sql += " ORDER BY alias_type ASC, alias ASC"
        rows = self.conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]


__all__ = ["ClaimCardV1Store"]
