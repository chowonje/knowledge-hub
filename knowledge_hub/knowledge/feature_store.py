"""Feature snapshot helpers for claim/feature layer v1."""

from __future__ import annotations

import json
from typing import Any

from knowledge_hub.core.models import FeatureSnapshot


def _loads(raw: Any, fallback):
    if raw is None or raw == "":
        return fallback
    if isinstance(raw, (dict, list)):
        return raw
    try:
        parsed = json.loads(raw)
    except Exception:
        return fallback
    if isinstance(fallback, list):
        return parsed if isinstance(parsed, list) else fallback
    return parsed if isinstance(parsed, dict) else fallback


class FeatureStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_feature_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_slug TEXT NOT NULL DEFAULT '',
                feature_kind TEXT NOT NULL,
                feature_key TEXT NOT NULL,
                feature_name TEXT NOT NULL DEFAULT '',
                entity_id TEXT NOT NULL DEFAULT '',
                note_id TEXT NOT NULL DEFAULT '',
                record_id TEXT NOT NULL DEFAULT '',
                canonical_url TEXT NOT NULL DEFAULT '',
                source_item_id TEXT NOT NULL DEFAULT '',
                freshness_score REAL NOT NULL DEFAULT 0.0,
                importance_score REAL NOT NULL DEFAULT 0.0,
                support_doc_count INTEGER NOT NULL DEFAULT 0,
                relation_degree REAL NOT NULL DEFAULT 0.0,
                claim_density REAL NOT NULL DEFAULT 0.0,
                source_trust_score REAL NOT NULL DEFAULT 0.0,
                concept_activity_score REAL NOT NULL DEFAULT 0.0,
                contradiction_score REAL NOT NULL DEFAULT 0.0,
                payload_json TEXT NOT NULL DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(topic_slug, feature_kind, feature_key)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feature_snapshots_topic_kind
            ON ontology_feature_snapshots(topic_slug, feature_kind, importance_score DESC)
            """
        )
        self.conn.commit()

    def upsert_snapshot(
        self,
        *,
        snapshot: FeatureSnapshot | None = None,
        topic_slug: str = "",
        feature_kind: str = "",
        feature_key: str = "",
        feature_name: str = "",
        entity_id: str = "",
        note_id: str = "",
        record_id: str = "",
        canonical_url: str = "",
        source_item_id: str = "",
        freshness_score: float = 0.0,
        importance_score: float = 0.0,
        support_doc_count: int = 0,
        relation_degree: float = 0.0,
        claim_density: float = 0.0,
        source_trust_score: float = 0.0,
        concept_activity_score: float = 0.0,
        contradiction_score: float = 0.0,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if snapshot is not None:
            payload_dict = snapshot.to_dict()
            topic_slug = payload_dict["topic_slug"]
            feature_kind = payload_dict["feature_kind"]
            feature_key = payload_dict["feature_key"]
            feature_name = payload_dict["feature_name"]
            entity_id = payload_dict["entity_id"]
            note_id = payload_dict["note_id"]
            record_id = payload_dict["record_id"]
            canonical_url = payload_dict["canonical_url"]
            source_item_id = payload_dict["source_item_id"]
            freshness_score = payload_dict["freshness_score"]
            importance_score = payload_dict["importance_score"]
            support_doc_count = payload_dict["support_doc_count"]
            relation_degree = payload_dict["relation_degree"]
            claim_density = payload_dict["claim_density"]
            source_trust_score = payload_dict["source_trust_score"]
            concept_activity_score = payload_dict["concept_activity_score"]
            contradiction_score = payload_dict["contradiction_score"]
            payload = payload_dict["payload"]
        self.conn.execute(
            """
            INSERT INTO ontology_feature_snapshots (
                topic_slug, feature_kind, feature_key, feature_name, entity_id, note_id, record_id,
                canonical_url, source_item_id, freshness_score, importance_score, support_doc_count,
                relation_degree, claim_density, source_trust_score, concept_activity_score,
                contradiction_score, payload_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(topic_slug, feature_kind, feature_key) DO UPDATE SET
                feature_name=excluded.feature_name,
                entity_id=excluded.entity_id,
                note_id=excluded.note_id,
                record_id=excluded.record_id,
                canonical_url=excluded.canonical_url,
                source_item_id=excluded.source_item_id,
                freshness_score=excluded.freshness_score,
                importance_score=excluded.importance_score,
                support_doc_count=excluded.support_doc_count,
                relation_degree=excluded.relation_degree,
                claim_density=excluded.claim_density,
                source_trust_score=excluded.source_trust_score,
                concept_activity_score=excluded.concept_activity_score,
                contradiction_score=excluded.contradiction_score,
                payload_json=excluded.payload_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                str(topic_slug or ""),
                str(feature_kind or ""),
                str(feature_key or ""),
                str(feature_name or ""),
                str(entity_id or ""),
                str(note_id or ""),
                str(record_id or ""),
                str(canonical_url or ""),
                str(source_item_id or ""),
                float(freshness_score or 0.0),
                float(importance_score or 0.0),
                int(support_doc_count or 0),
                float(relation_degree or 0.0),
                float(claim_density or 0.0),
                float(source_trust_score or 0.0),
                float(concept_activity_score or 0.0),
                float(contradiction_score or 0.0),
                json.dumps(payload or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def _decode_row(self, row) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        item["payload_json"] = _loads(item.get("payload_json"), {})
        return item

    def get_snapshot(self, *, topic_slug: str, feature_kind: str, feature_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT * FROM ontology_feature_snapshots
            WHERE topic_slug = ? AND feature_kind = ? AND feature_key = ?
            """,
            (str(topic_slug or ""), str(feature_kind or ""), str(feature_key or "")),
        ).fetchone()
        return self._decode_row(row)

    def find_source_snapshot(
        self,
        *,
        topic_slug: str = "",
        note_id: str = "",
        record_id: str = "",
        canonical_url: str = "",
        source_item_id: str = "",
    ) -> dict[str, Any] | None:
        candidates = [
            ("note_id", str(note_id or "").strip()),
            ("record_id", str(record_id or "").strip()),
            ("source_item_id", str(source_item_id or "").strip()),
            ("canonical_url", str(canonical_url or "").strip()),
        ]
        for field, value in candidates:
            if not value:
                continue
            query = f"SELECT * FROM ontology_feature_snapshots WHERE feature_kind = 'source' AND {field} = ?"
            params: list[Any] = [value]
            if topic_slug:
                query += " AND topic_slug = ?"
                params.append(str(topic_slug))
            query += " ORDER BY importance_score DESC, updated_at DESC LIMIT 1"
            row = self.conn.execute(query, params).fetchone()
            decoded = self._decode_row(row)
            if decoded:
                return decoded
        return None

    def list_snapshots(
        self,
        *,
        topic_slug: str | None = None,
        feature_kind: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM ontology_feature_snapshots WHERE 1=1"
        params: list[Any] = []
        if topic_slug is not None:
            query += " AND topic_slug = ?"
            params.append(str(topic_slug))
        if feature_kind:
            query += " AND feature_kind = ?"
            params.append(str(feature_kind))
        query += " ORDER BY importance_score DESC, updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        return [item for item in (self._decode_row(row) for row in rows) if item]

    def list_top(
        self,
        *,
        topic_slug: str,
        feature_kind: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM ontology_feature_snapshots
            WHERE topic_slug = ? AND feature_kind = ?
            ORDER BY importance_score DESC, freshness_score DESC, updated_at DESC
            LIMIT ?
            """,
            (str(topic_slug or ""), str(feature_kind or ""), max(1, int(limit))),
        ).fetchall()
        return [item for item in (self._decode_row(row) for row in rows) if item]
