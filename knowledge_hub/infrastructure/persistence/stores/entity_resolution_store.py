"""Entity resolution merge proposal store."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from knowledge_hub.core.models import OntologyEvent

log = logging.getLogger("khub.entity_resolution_store")


class EntityResolutionStore:
    """Store and apply entity merge proposals.

    The store owns SQL for proposal persistence while delegating higher-level
    merge operations back to the SQLite facade for compatibility with the
    existing ontology helpers.
    """

    def __init__(self, conn, db):
        self.conn = conn
        self.db = db

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_merge_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_slug TEXT NOT NULL DEFAULT '',
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                match_method TEXT NOT NULL DEFAULT '',
                reason_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'approved', 'rejected')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP,
                UNIQUE(source_entity_id, target_entity_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entity_merge_proposals_topic
            ON entity_merge_proposals(topic_slug, status, created_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_split_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_slug TEXT NOT NULL DEFAULT '',
                source_entity_id TEXT NOT NULL,
                candidate_entities_json TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0.0,
                reason_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'approved', 'rejected')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entity_split_proposals_topic
            ON entity_split_proposals(topic_slug, status, created_at DESC)
            """
        )
        self.conn.commit()

    def _load_entity_summary(self, entity_id: str) -> dict[str, Any]:
        token = str(entity_id or "").strip()
        if not token:
            return {}
        entity = self.db.get_ontology_entity(token) or {}
        entity_type = str(entity.get("entity_type") or "").strip()
        aliases = sorted(self.db.get_entity_aliases(token))
        relation_count = len(self.db.get_relations(entity_type or "concept", token)) if entity_type else 0
        paper_count = len(self.db.get_concept_papers(token)) if entity_type == "concept" else 0
        return {
            "entity_id": token,
            "canonical_name": str(entity.get("canonical_name") or token),
            "entity_type": entity_type,
            "source": str(entity.get("source") or ""),
            "aliases": aliases,
            "alias_count": len(aliases),
            "relation_count": relation_count,
            "claim_count": len(self.db.list_claims_by_entity(token, limit=500)),
            "paper_count": paper_count,
        }

    def _build_duplicate_cluster(self, source_entity_id: str, target_entity_id: str) -> dict[str, Any]:
        source_token = str(source_entity_id or "").strip()
        target_token = str(target_entity_id or "").strip()
        if not source_token or not target_token:
            return {"entity_ids": [], "size": 0, "proposals": []}
        rows = self.conn.execute(
            """
            SELECT id, source_entity_id, target_entity_id, confidence, status, match_method, reason_json
            FROM entity_merge_proposals
            WHERE source_entity_id IN (?, ?) OR target_entity_id IN (?, ?)
            ORDER BY confidence DESC, created_at DESC, id DESC
            """,
            (source_token, target_token, source_token, target_token),
        ).fetchall()
        entity_ids = {source_token, target_token}
        proposals: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["reason_json"] = json.loads(item.get("reason_json") or "{}")
            except Exception:
                item["reason_json"] = {}
            proposals.append(item)
            entity_ids.add(str(item.get("source_entity_id") or "").strip())
            entity_ids.add(str(item.get("target_entity_id") or "").strip())
        summaries = [self._load_entity_summary(entity_id) for entity_id in sorted(entity_ids) if entity_id]
        return {
            "entity_ids": [item["entity_id"] for item in summaries if item],
            "size": len([item for item in summaries if item]),
            "entities": summaries,
            "proposals": proposals,
        }

    def _enrich_proposal(self, item: dict[str, Any]) -> dict[str, Any]:
        source_entity_id = str(item.get("source_entity_id") or "").strip()
        target_entity_id = str(item.get("target_entity_id") or "").strip()
        reason = item.get("reason_json") or {}
        item["source_entity"] = self._load_entity_summary(source_entity_id)
        item["target_entity"] = self._load_entity_summary(target_entity_id)
        item["duplicate_cluster"] = self._build_duplicate_cluster(source_entity_id, target_entity_id)
        item["proposal_provenance"] = {
            "topic_slug": str(item.get("topic_slug") or ""),
            "match_method": str(item.get("match_method") or ""),
            "note_id": str(reason.get("note_id") or ""),
            "source_url": str(reason.get("source_url") or ""),
            "source_strength": float(reason.get("source_strength") or 0.0),
            "target_strength": float(reason.get("target_strength") or 0.0),
        }
        return item

    def _enrich_split_proposal(self, item: dict[str, Any]) -> dict[str, Any]:
        source_entity_id = str(item.get("source_entity_id") or "").strip()
        candidate_entity_ids = item.get("candidate_entities_json") or []
        if not isinstance(candidate_entity_ids, list):
            candidate_entity_ids = []
        item["source_entity"] = self._load_entity_summary(source_entity_id)
        item["candidate_entities"] = [
            self._load_entity_summary(str(entity_id).strip())
            for entity_id in candidate_entity_ids
            if str(entity_id).strip()
        ]
        return item

    @staticmethod
    def _normalize_status(status: str) -> str:
        status_token = str(status or "pending").strip().lower()
        if status_token not in {"pending", "approved", "rejected"}:
            return "pending"
        return status_token

    @staticmethod
    def _clamp_confidence(value: float) -> float:
        return max(0.0, min(0.999, float(value or 0.0)))

    def _apply_precision_first_merge_policy(
        self,
        *,
        source_entity_id: str,
        target_entity_id: str,
        confidence: float,
        match_method: str,
        reason: dict | None,
    ) -> tuple[float, dict[str, Any], bool]:
        reason_json = dict(reason or {})
        precision = reason_json.get("precision_first")
        if not isinstance(precision, dict):
            precision = {}

        source_summary = self._load_entity_summary(source_entity_id)
        target_summary = self._load_entity_summary(target_entity_id)
        penalties = [str(item).strip() for item in precision.get("penalties") or [] if str(item).strip()]
        suppress_reasons = [
            str(item).strip()
            for item in precision.get("suppress_reasons") or []
            if str(item).strip()
        ]

        adjusted_confidence = self._clamp_confidence(precision.get("adjusted_confidence", confidence))
        source_type = str(source_summary.get("entity_type") or "").strip()
        target_type = str(target_summary.get("entity_type") or "").strip()
        if source_type and target_type and source_type != target_type and "entity_type_mismatch" not in suppress_reasons:
            suppress_reasons.append("entity_type_mismatch")

        if match_method == "normalized_exact" and adjusted_confidence < 0.94 and "entity_type_mismatch" not in suppress_reasons:
            adjusted_confidence = 0.94

        suppressed = bool(precision.get("suppressed")) or bool(suppress_reasons)
        precision.update(
            {
                "base_confidence": round(self._clamp_confidence(confidence), 4),
                "adjusted_confidence": round(adjusted_confidence, 4),
                "suppressed": suppressed,
                "suppress_reasons": suppress_reasons,
                "penalties": penalties,
                "source_entity_type": source_type,
                "target_entity_type": target_type,
            }
        )
        reason_json["precision_first"] = precision
        return adjusted_confidence, reason_json, suppressed

    def add_entity_merge_proposal(
        self,
        *,
        source_entity_id: str,
        target_entity_id: str,
        topic_slug: str = "",
        confidence: float = 0.0,
        match_method: str = "",
        reason: dict | None = None,
        status: str = "pending",
    ) -> int:
        source_token = str(source_entity_id or "").strip()
        target_token = str(target_entity_id or "").strip()
        if not source_token or not target_token or source_token == target_token:
            return 0
        status_token = self._normalize_status(status)
        adjusted_confidence, reason_json, suppressed = self._apply_precision_first_merge_policy(
            source_entity_id=source_token,
            target_entity_id=target_token,
            confidence=float(confidence or 0.0),
            match_method=str(match_method or ""),
            reason=reason,
        )
        if suppressed:
            return 0
        cursor = self.conn.execute(
            """INSERT INTO entity_merge_proposals
                 (topic_slug, source_entity_id, target_entity_id, confidence, match_method, reason_json, status)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source_entity_id, target_entity_id) DO UPDATE SET
                 topic_slug=excluded.topic_slug,
                 confidence=excluded.confidence,
                 match_method=excluded.match_method,
                 reason_json=excluded.reason_json,
                 status=CASE
                   WHEN entity_merge_proposals.status='approved' THEN entity_merge_proposals.status
                   ELSE excluded.status
                 END,
                 created_at=CURRENT_TIMESTAMP""",
            (
                str(topic_slug or ""),
                source_token,
                target_token,
                adjusted_confidence,
                str(match_method or ""),
                json.dumps(reason_json, ensure_ascii=False),
                status_token,
            ),
        )
        self.conn.commit()
        proposal_id = int(cursor.lastrowid or 0)
        if proposal_id:
            return proposal_id
        row = self.conn.execute(
            """SELECT id FROM entity_merge_proposals
               WHERE source_entity_id = ? AND target_entity_id = ?""",
            (source_token, target_token),
        ).fetchone()
        return int(row["id"]) if row else 0

    def add_entity_split_proposal(
        self,
        *,
        source_entity_id: str,
        candidate_entity_ids: list[str],
        topic_slug: str = "",
        confidence: float = 0.0,
        reason: dict | None = None,
        status: str = "pending",
    ) -> int:
        source_token = str(source_entity_id or "").strip()
        candidate_tokens = sorted(
            {
                str(candidate_id).strip()
                for candidate_id in candidate_entity_ids or []
                if str(candidate_id).strip() and str(candidate_id).strip() != source_token
            }
        )
        if not source_token or not candidate_tokens:
            return 0
        status_token = self._normalize_status(status)
        confidence_value = self._clamp_confidence(confidence)
        row = self.conn.execute(
            """SELECT id, status FROM entity_split_proposals
               WHERE topic_slug = ? AND source_entity_id = ?
               ORDER BY created_at DESC, id DESC LIMIT 1""",
            (str(topic_slug or ""), source_token),
        ).fetchone()
        payload = (
            json.dumps(candidate_tokens, ensure_ascii=False),
            confidence_value,
            json.dumps(reason or {}, ensure_ascii=False),
        )
        if row:
            self.conn.execute(
                """UPDATE entity_split_proposals
                   SET candidate_entities_json = ?,
                       confidence = ?,
                       reason_json = ?,
                       status = CASE
                         WHEN status = 'approved' THEN status
                         ELSE ?
                       END,
                       created_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (*payload, status_token, int(row["id"])),
            )
            self.conn.commit()
            return int(row["id"])

        cursor = self.conn.execute(
            """INSERT INTO entity_split_proposals
                 (topic_slug, source_entity_id, candidate_entities_json, confidence, reason_json, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(topic_slug or ""),
                source_token,
                payload[0],
                payload[1],
                payload[2],
                status_token,
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def list_entity_split_proposals(
        self,
        *,
        topic_slug: str | None = None,
        status: str | None = "pending",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM entity_split_proposals WHERE 1=1"
        params: list[Any] = []
        if topic_slug:
            query += " AND topic_slug = ?"
            params.append(str(topic_slug))
        status_token = str(status or "").strip().lower()
        if status_token:
            query += " AND status = ?"
            params.append(status_token)
        query += " ORDER BY confidence DESC, created_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["candidate_entities_json"] = json.loads(item.get("candidate_entities_json") or "[]")
            except Exception:
                item["candidate_entities_json"] = []
            try:
                item["reason_json"] = json.loads(item.get("reason_json") or "{}")
            except Exception:
                item["reason_json"] = {}
            items.append(self._enrich_split_proposal(item))
        return items

    def get_entity_split_proposal(self, proposal_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM entity_split_proposals WHERE id = ?",
            (int(proposal_id),),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["candidate_entities_json"] = json.loads(item.get("candidate_entities_json") or "[]")
        except Exception:
            item["candidate_entities_json"] = []
        try:
            item["reason_json"] = json.loads(item.get("reason_json") or "{}")
        except Exception:
            item["reason_json"] = {}
        return self._enrich_split_proposal(item)

    def list_entity_merge_proposals(
        self,
        *,
        topic_slug: str | None = None,
        status: str | None = "pending",
        limit: int = 200,
    ) -> list[dict]:
        query = "SELECT * FROM entity_merge_proposals WHERE 1=1"
        params: list[Any] = []
        if topic_slug:
            query += " AND topic_slug = ?"
            params.append(str(topic_slug))
        status_token = str(status or "").strip().lower()
        if status_token:
            query += " AND status = ?"
            params.append(status_token)
        query += " ORDER BY confidence DESC, created_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        items: list[dict] = []
        for row in rows:
            item = dict(row)
            try:
                item["reason_json"] = json.loads(item.get("reason_json") or "{}")
            except Exception:
                item["reason_json"] = {}
            items.append(self._enrich_proposal(item))
        return items

    def get_entity_merge_proposal(self, proposal_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM entity_merge_proposals WHERE id = ?",
            (int(proposal_id),),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["reason_json"] = json.loads(item.get("reason_json") or "{}")
        except Exception:
            item["reason_json"] = {}
        return self._enrich_proposal(item)

    def update_entity_merge_proposal_status(self, proposal_id: int, status: str) -> bool:
        status_token = str(status or "pending").strip().lower()
        if status_token not in {"pending", "approved", "rejected"}:
            status_token = "pending"
        cursor = self.conn.execute(
            """UPDATE entity_merge_proposals
               SET status = ?,
                   reviewed_at = CASE WHEN ? IN ('approved', 'rejected') THEN CURRENT_TIMESTAMP ELSE reviewed_at END
               WHERE id = ?""",
            (status_token, status_token, int(proposal_id)),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def reject_entity_merge_proposal(self, proposal_id: int) -> bool:
        return self.update_entity_merge_proposal_status(proposal_id, "rejected")

    def apply_entity_merge_proposal(self, proposal_id: int) -> bool:
        item = self.get_entity_merge_proposal(int(proposal_id))
        if not item:
            return False
        source_entity_id = str(item.get("source_entity_id") or "").strip()
        target_entity_id = str(item.get("target_entity_id") or "").strip()
        if not source_entity_id or not target_entity_id or source_entity_id == target_entity_id:
            return False
        source_entity = self.db.get_ontology_entity(source_entity_id) or {}
        target_entity = self.db.get_ontology_entity(target_entity_id) or {}
        if not source_entity or not target_entity:
            return False

        existing_target_aliases = set(self.db.get_entity_aliases(target_entity_id))
        aliases_to_add = {str(source_entity.get("canonical_name") or "").strip()}
        aliases_to_add.update(self.db.get_entity_aliases(source_entity_id))
        aliases_to_add.discard("")
        for alias in sorted(aliases_to_add):
            self.db.add_entity_alias(alias, target_entity_id)
        aliases_added = max(0, len(set(self.db.get_entity_aliases(target_entity_id)) - existing_target_aliases))

        subject_claims_cursor = self.conn.execute(
            "UPDATE ontology_claims SET subject_entity_id = ? WHERE subject_entity_id = ?",
            (target_entity_id, source_entity_id),
        )
        object_claims_cursor = self.conn.execute(
            "UPDATE ontology_claims SET object_entity_id = ? WHERE object_entity_id = ?",
            (target_entity_id, source_entity_id),
        )
        source_relations_cursor = self.conn.execute(
            "UPDATE ontology_relations SET source_entity_id = ?, source_id = ? WHERE source_entity_id = ?",
            (target_entity_id, target_entity_id, source_entity_id),
        )
        target_relations_cursor = self.conn.execute(
            "UPDATE ontology_relations SET target_entity_id = ?, target_id = ? WHERE target_entity_id = ?",
            (target_entity_id, target_entity_id, source_entity_id),
        )
        self_loop_cursor = self.conn.execute(
            """DELETE FROM ontology_relations
               WHERE source_entity_id = target_entity_id
                 AND predicate_id IN ('related_to', 'example_of')"""
        )
        feature_rows = self.conn.execute(
            """SELECT topic_slug
               FROM ontology_feature_snapshots
               WHERE entity_id IN (?, ?)""",
            (source_entity_id, target_entity_id),
        ).fetchall()
        affected_topics = {
            str(row["topic_slug"] or "").strip()
            for row in feature_rows
            if str(row["topic_slug"] or "").strip()
        }
        feature_delete_cursor = self.conn.execute(
            """DELETE FROM ontology_feature_snapshots
               WHERE entity_id IN (?, ?)""",
            (source_entity_id, target_entity_id),
        )
        graph_node_rows = self.conn.execute(
            """SELECT node_id, provenance_json
               FROM learning_graph_nodes
               WHERE entity_id IN (?, ?)""",
            (source_entity_id, target_entity_id),
        ).fetchall()
        graph_node_ids = {str(row["node_id"] or "").strip() for row in graph_node_rows if str(row["node_id"] or "").strip()}
        for row in graph_node_rows:
            try:
                provenance = json.loads(row["provenance_json"] or "{}")
            except Exception:
                provenance = {}
            topic_slug = str(provenance.get("topicSlug") or "").strip()
            if topic_slug:
                affected_topics.add(topic_slug)
        for table_name, column_name in (
            ("learning_graph_pending", "payload_json"),
            ("learning_graph_paths", "path_json"),
        ):
            rows = self.conn.execute(
                f"SELECT topic_slug, {column_name} AS payload FROM {table_name}"
            ).fetchall()
            for row in rows:
                payload_text = str(row["payload"] or "")
                if source_entity_id in payload_text or target_entity_id in payload_text:
                    topic_slug = str(row["topic_slug"] or "").strip()
                    if topic_slug:
                        affected_topics.add(topic_slug)
        edge_delete_cursor = self.conn.execute(
            """DELETE FROM learning_graph_edges
               WHERE source_node_id IN ({placeholders})
                  OR target_node_id IN ({placeholders})""".format(
                placeholders=", ".join("?" for _ in graph_node_ids) if graph_node_ids else "''"
            ),
            tuple(graph_node_ids) * 2 if graph_node_ids else tuple(),
        )
        resource_delete_cursor = self.conn.execute(
            """DELETE FROM learning_graph_resource_links
               WHERE concept_node_id IN ({placeholders})
                  OR resource_node_id IN ({placeholders})""".format(
                placeholders=", ".join("?" for _ in graph_node_ids) if graph_node_ids else "''"
            ),
            tuple(graph_node_ids) * 2 if graph_node_ids else tuple(),
        )
        node_delete_cursor = self.conn.execute(
            """DELETE FROM learning_graph_nodes
               WHERE node_id IN ({placeholders})""".format(
                placeholders=", ".join("?" for _ in graph_node_ids) if graph_node_ids else "''"
            ),
            tuple(graph_node_ids) if graph_node_ids else tuple(),
        )
        pending_deleted = 0
        path_deleted = 0
        event_count = 0
        for topic_slug in sorted(affected_topics):
            pending_deleted += max(
                0,
                int(
                    (
                        self.conn.execute(
                            "DELETE FROM learning_graph_pending WHERE topic_slug = ?",
                            (topic_slug,),
                        ).rowcount
                        or 0
                    )
                ),
            )
            path_deleted += max(
                0,
                int(
                    (
                        self.conn.execute(
                            "DELETE FROM learning_graph_paths WHERE topic_slug = ?",
                            (topic_slug,),
                        ).rowcount
                        or 0
                    )
                ),
            )
            self.conn.execute(
                """INSERT INTO learning_graph_events
                   (event_id, topic_slug, event_type, actor, payload_json, created_at)
                   VALUES (?, ?, 'entity_merge_invalidation', 'system', ?, CURRENT_TIMESTAMP)""",
                (
                    f"merge_invalidation_{proposal_id}_{topic_slug}",
                    topic_slug,
                    json.dumps(
                        {
                            "proposalId": int(proposal_id),
                            "sourceEntityId": source_entity_id,
                            "targetEntityId": target_entity_id,
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
            event_count += 1
        source_entity = self.db.get_ontology_entity(source_entity_id) or {}
        source_aliases = list(self.db.get_entity_aliases(source_entity_id)) if source_entity else []
        delete_entity_cursor = self.conn.execute(
            "DELETE FROM ontology_entities WHERE entity_id = ?",
            (source_entity_id,),
        )
        if delete_entity_cursor.rowcount and getattr(self.db, "event_store", None):
            event = OntologyEvent(
                event_id=f"evt_{uuid4().hex}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="entity_deleted",
                entity_id=source_entity_id,
                entity_type=str(source_entity.get("entity_type") or "entity"),
                actor="entity_resolution_merge",
                data={
                    "canonical_name": str(source_entity.get("canonical_name") or source_entity_id),
                    "description": str(source_entity.get("description") or ""),
                    "merge_target_entity_id": target_entity_id,
                    "proposal_id": int(proposal_id),
                },
                policy_class="P2",
            )
            try:
                self.db.event_store.append(event)
            except Exception as error:
                log.error("Event append failed for entity merge delete (%s): %s", source_entity_id, error)
        side_effects = {
            "aliases_added": aliases_added,
            "subject_claims_moved": max(0, int(subject_claims_cursor.rowcount or 0)),
            "object_claims_moved": max(0, int(object_claims_cursor.rowcount or 0)),
            "source_relations_moved": max(0, int(source_relations_cursor.rowcount or 0)),
            "target_relations_moved": max(0, int(target_relations_cursor.rowcount or 0)),
            "self_loops_deleted": max(0, int(self_loop_cursor.rowcount or 0)),
            "feature_snapshots_deleted": max(0, int(feature_delete_cursor.rowcount or 0)),
            "learning_graph_edges_deleted": max(0, int(edge_delete_cursor.rowcount or 0)),
            "learning_graph_resource_links_deleted": max(0, int(resource_delete_cursor.rowcount or 0)),
            "learning_graph_nodes_deleted": max(0, int(node_delete_cursor.rowcount or 0)),
            "learning_graph_pending_deleted": pending_deleted,
            "learning_graph_paths_deleted": path_deleted,
            "learning_graph_events_added": event_count,
            "affected_topics": sorted(affected_topics),
            "source_entity_deleted": max(0, int(delete_entity_cursor.rowcount or 0)),
        }
        reason_json = dict(item.get("reason_json") or {})
        reason_json["apply_side_effects"] = side_effects
        self.conn.commit()
        self.conn.execute(
            """UPDATE entity_merge_proposals
               SET status = 'approved',
                   reviewed_at = CURRENT_TIMESTAMP,
                   reason_json = ?
               WHERE id = ?""",
            (json.dumps(reason_json, ensure_ascii=False), int(proposal_id)),
        )
        self.conn.commit()
        event_store = getattr(self.db, "event_store", None)
        if event_store and source_entity and int(delete_entity_cursor.rowcount or 0) > 0:
            event = OntologyEvent(
                event_id=f"evt_{uuid4().hex}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="entity_deleted",
                entity_id=str(source_entity_id),
                entity_type=str(source_entity.get("entity_type") or "entity"),
                actor="entity_resolution_merge",
                data={
                    "canonical_name": str(source_entity.get("canonical_name") or ""),
                    "description": str(source_entity.get("description") or ""),
                    "aliases": source_aliases,
                    "merge_target_entity_id": str(target_entity_id),
                    "merge_proposal_id": int(proposal_id),
                },
                policy_class="P2",
                run_id=f"entity_merge:{proposal_id}",
            )
            event_store.append(event)
        return True
