"""Canonical SQLite store for learning graph projections."""

from __future__ import annotations

import json
from typing import Any


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


class LearningGraphStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_graph_nodes (
                node_id TEXT PRIMARY KEY,
                entity_id TEXT,
                node_type TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                difficulty_level TEXT NOT NULL DEFAULT 'intermediate',
                difficulty_score REAL NOT NULL DEFAULT 0.5,
                stage TEXT NOT NULL DEFAULT 'intermediate',
                confidence REAL NOT NULL DEFAULT 0.5,
                provenance_json TEXT NOT NULL DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learning_graph_nodes_entity
            ON learning_graph_nodes(entity_id, node_type)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_graph_edges (
                edge_id TEXT PRIMARY KEY,
                source_node_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                status TEXT NOT NULL DEFAULT 'pending',
                provenance_json TEXT NOT NULL DEFAULT '{}',
                evidence_json TEXT NOT NULL DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_node_id, edge_type, target_node_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learning_graph_edges_status
            ON learning_graph_edges(status, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_graph_paths (
                path_id TEXT PRIMARY KEY,
                topic_slug TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                version INTEGER NOT NULL DEFAULT 1,
                path_json TEXT NOT NULL DEFAULT '{}',
                score_json TEXT NOT NULL DEFAULT '{}',
                provenance_json TEXT NOT NULL DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learning_graph_paths_topic
            ON learning_graph_paths(topic_slug, status, version DESC)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_graph_resource_links (
                link_id TEXT PRIMARY KEY,
                concept_node_id TEXT NOT NULL,
                resource_node_id TEXT NOT NULL,
                link_type TEXT NOT NULL,
                reading_stage TEXT NOT NULL DEFAULT 'intermediate',
                confidence REAL NOT NULL DEFAULT 0.5,
                status TEXT NOT NULL DEFAULT 'pending',
                provenance_json TEXT NOT NULL DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(concept_node_id, resource_node_id, link_type)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learning_graph_resource_links_status
            ON learning_graph_resource_links(status, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_graph_pending (
                pending_id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_type TEXT NOT NULL,
                topic_slug TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                confidence REAL NOT NULL DEFAULT 0.5,
                reason TEXT NOT NULL DEFAULT '',
                provenance_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                reviewed_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learning_graph_pending_topic
            ON learning_graph_pending(topic_slug, item_type, status, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_graph_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                topic_slug TEXT NOT NULL,
                event_type TEXT NOT NULL,
                actor TEXT NOT NULL DEFAULT 'system',
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learning_graph_events_topic
            ON learning_graph_events(topic_slug, created_at DESC)
            """
        )
        self.conn.commit()

    def upsert_node(
        self,
        node_id: str,
        entity_id: str | None,
        node_type: str,
        canonical_name: str,
        difficulty_level: str,
        difficulty_score: float,
        stage: str,
        confidence: float,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO learning_graph_nodes
                 (node_id, entity_id, node_type, canonical_name, difficulty_level,
                  difficulty_score, stage, confidence, provenance_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(node_id) DO UPDATE SET
                 entity_id=excluded.entity_id,
                 node_type=excluded.node_type,
                 canonical_name=excluded.canonical_name,
                 difficulty_level=excluded.difficulty_level,
                 difficulty_score=excluded.difficulty_score,
                 stage=excluded.stage,
                 confidence=excluded.confidence,
                 provenance_json=excluded.provenance_json,
                 updated_at=CURRENT_TIMESTAMP""",
            (
                node_id,
                entity_id,
                node_type,
                canonical_name,
                difficulty_level,
                float(difficulty_score),
                stage,
                float(confidence),
                json.dumps(provenance or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM learning_graph_nodes WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["provenance_json"] = _loads(item.get("provenance_json"), {})
        return item

    def list_nodes(self, node_type: str | None = None, limit: int = 2000) -> list[dict[str, Any]]:
        query = "SELECT * FROM learning_graph_nodes"
        params: list[Any] = []
        if node_type:
            query += " WHERE node_type = ?"
            params.append(node_type)
        query += " ORDER BY canonical_name ASC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["provenance_json"] = _loads(item.get("provenance_json"), {})
            result.append(item)
        return result

    def upsert_edge(
        self,
        edge_id: str,
        source_node_id: str,
        edge_type: str,
        target_node_id: str,
        confidence: float,
        status: str,
        provenance: dict[str, Any] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO learning_graph_edges
                 (edge_id, source_node_id, edge_type, target_node_id, confidence, status,
                  provenance_json, evidence_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(source_node_id, edge_type, target_node_id) DO UPDATE SET
                 edge_id=excluded.edge_id,
                 confidence=excluded.confidence,
                 status=excluded.status,
                 provenance_json=excluded.provenance_json,
                 evidence_json=excluded.evidence_json,
                 updated_at=CURRENT_TIMESTAMP""",
            (
                edge_id,
                source_node_id,
                edge_type,
                target_node_id,
                float(confidence),
                status,
                json.dumps(provenance or {}, ensure_ascii=False),
                json.dumps(evidence or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def list_edges(self, status: str | None = None, limit: int = 5000) -> list[dict[str, Any]]:
        query = "SELECT * FROM learning_graph_edges"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["provenance_json"] = _loads(item.get("provenance_json"), {})
            item["evidence_json"] = _loads(item.get("evidence_json"), {})
            result.append(item)
        return result

    def upsert_path(
        self,
        path_id: str,
        topic_slug: str,
        status: str,
        version: int,
        path_payload: dict[str, Any],
        score_payload: dict[str, Any],
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO learning_graph_paths
                 (path_id, topic_slug, status, version, path_json, score_json, provenance_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(path_id) DO UPDATE SET
                 topic_slug=excluded.topic_slug,
                 status=excluded.status,
                 version=excluded.version,
                 path_json=excluded.path_json,
                 score_json=excluded.score_json,
                 provenance_json=excluded.provenance_json,
                 updated_at=CURRENT_TIMESTAMP""",
            (
                path_id,
                topic_slug,
                status,
                int(version),
                json.dumps(path_payload, ensure_ascii=False),
                json.dumps(score_payload, ensure_ascii=False),
                json.dumps(provenance or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def get_latest_path(self, topic_slug: str, status: str | None = None) -> dict[str, Any] | None:
        query = "SELECT * FROM learning_graph_paths WHERE topic_slug = ?"
        params: list[Any] = [topic_slug]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY version DESC, updated_at DESC LIMIT 1"
        row = self.conn.execute(query, params).fetchone()
        if not row:
            return None
        item = dict(row)
        item["path_json"] = _loads(item.get("path_json"), {})
        item["score_json"] = _loads(item.get("score_json"), {})
        item["provenance_json"] = _loads(item.get("provenance_json"), {})
        return item

    def upsert_resource_link(
        self,
        link_id: str,
        concept_node_id: str,
        resource_node_id: str,
        link_type: str,
        reading_stage: str,
        confidence: float,
        status: str,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO learning_graph_resource_links
                 (link_id, concept_node_id, resource_node_id, link_type, reading_stage,
                  confidence, status, provenance_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(concept_node_id, resource_node_id, link_type) DO UPDATE SET
                 link_id=excluded.link_id,
                 reading_stage=excluded.reading_stage,
                 confidence=excluded.confidence,
                 status=excluded.status,
                 provenance_json=excluded.provenance_json,
                 updated_at=CURRENT_TIMESTAMP""",
            (
                link_id,
                concept_node_id,
                resource_node_id,
                link_type,
                reading_stage,
                float(confidence),
                status,
                json.dumps(provenance or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def list_resource_links(self, status: str | None = None, limit: int = 5000) -> list[dict[str, Any]]:
        query = "SELECT * FROM learning_graph_resource_links"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["provenance_json"] = _loads(item.get("provenance_json"), {})
            result.append(item)
        return result

    def add_pending(
        self,
        item_type: str,
        topic_slug: str,
        payload: dict[str, Any],
        confidence: float,
        reason: str,
        provenance: dict[str, Any] | None = None,
    ) -> int:
        cursor = self.conn.execute(
            """INSERT INTO learning_graph_pending
                 (item_type, topic_slug, payload_json, confidence, reason, provenance_json, status, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', CURRENT_TIMESTAMP)""",
            (
                item_type,
                topic_slug,
                json.dumps(payload, ensure_ascii=False),
                float(confidence),
                reason,
                json.dumps(provenance or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def list_pending(
        self,
        topic_slug: str | None = None,
        item_type: str | None = None,
        status: str = "pending",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM learning_graph_pending WHERE status = ?"
        params: list[Any] = [status]
        if topic_slug:
            query += " AND topic_slug = ?"
            params.append(topic_slug)
        if item_type and item_type != "all":
            query += " AND item_type = ?"
            params.append(item_type)
        query += " ORDER BY updated_at DESC, pending_id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload_json"] = _loads(item.get("payload_json"), {})
            item["provenance_json"] = _loads(item.get("provenance_json"), {})
            result.append(item)
        return result

    def get_pending(self, pending_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM learning_graph_pending WHERE pending_id = ?",
            (int(pending_id),),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["payload_json"] = _loads(item.get("payload_json"), {})
        item["provenance_json"] = _loads(item.get("provenance_json"), {})
        return item

    def set_pending_status(self, pending_id: int, status: str) -> None:
        self.conn.execute(
            """UPDATE learning_graph_pending
               SET status = ?, reviewed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
               WHERE pending_id = ?""",
            (status, int(pending_id)),
        )
        self.conn.commit()

    def append_event(
        self,
        event_type: str,
        topic_slug: str,
        payload: dict[str, Any],
        actor: str = "system",
    ) -> None:
        self.conn.execute(
            """INSERT INTO learning_graph_events
                 (event_id, topic_slug, event_type, actor, payload_json, created_at)
               VALUES (hex(randomblob(16)), ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (
                topic_slug,
                event_type,
                actor,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def list_events(self, topic_slug: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        query = "SELECT * FROM learning_graph_events"
        params: list[Any] = []
        if topic_slug:
            query += " WHERE topic_slug = ?"
            params.append(topic_slug)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload_json"] = _loads(item.get("payload_json"), {})
            result.append(item)
        return result
