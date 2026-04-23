"""Canonical learning/web-pending store helpers."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


class LearningStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_sessions (
                session_id TEXT PRIMARY KEY,
                topic_slug TEXT NOT NULL,
                target_trunk_ids_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'draft',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_session_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES learning_sessions(session_id) ON DELETE CASCADE,
                source_canonical_id TEXT NOT NULL,
                relation_norm TEXT NOT NULL,
                target_canonical_id TEXT NOT NULL,
                evidence_ptrs_json TEXT NOT NULL DEFAULT '[]',
                confidence REAL DEFAULT 3.0,
                is_valid INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learning_edges_session
            ON learning_session_edges(session_id)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_progress (
                session_id TEXT PRIMARY KEY REFERENCES learning_sessions(session_id) ON DELETE CASCADE,
                topic_slug TEXT NOT NULL,
                score_final REAL NOT NULL DEFAULT 0,
                score_edge_accuracy REAL NOT NULL DEFAULT 0,
                score_coverage REAL NOT NULL DEFAULT 0,
                score_explanation_quality REAL NOT NULL DEFAULT 0,
                gate_passed INTEGER NOT NULL DEFAULT 0,
                gate_status TEXT NOT NULL DEFAULT 'insufficient',
                weaknesses_json TEXT NOT NULL DEFAULT '[]',
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                run_id TEXT,
                request_id TEXT,
                session_id TEXT,
                event_type TEXT NOT NULL,
                logical_step TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL DEFAULT '{}',
                policy_class TEXT NOT NULL DEFAULT 'P2',
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, event_type, logical_step)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learning_events_session
            ON learning_events(session_id)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS web_ontology_pending (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                topic_slug TEXT NOT NULL,
                note_id TEXT NOT NULL,
                source_url TEXT NOT NULL,
                source_canonical_id TEXT NOT NULL,
                relation_norm TEXT NOT NULL,
                target_canonical_id TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                evidence_ptrs_json TEXT NOT NULL DEFAULT '[]',
                reason_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP,
                CHECK(status IN ('pending', 'approved', 'rejected')),
                UNIQUE(run_id, note_id, source_canonical_id, relation_norm, target_canonical_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_web_pending_topic_status
            ON web_ontology_pending(topic_slug, status, created_at DESC)
            """
        )
        self.conn.commit()

    def ensure_learning_events_schema(self) -> None:
        self._ensure_learning_events_columns()
        self._ensure_learning_events_constraints()
        event_columns = {
            row["name"] for row in self.conn.execute("PRAGMA table_info(learning_events)").fetchall()
        }
        if "run_id" in event_columns:
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_learning_events_run
                ON learning_events(run_id)
                """
            )
        self.conn.commit()

    def _ensure_learning_events_columns(self) -> None:
        existing = {
            row["name"] for row in self.conn.execute("PRAGMA table_info(learning_events)").fetchall()
        }
        migrations: dict[str, str] = {
            "run_id": "ALTER TABLE learning_events ADD COLUMN run_id TEXT",
            "request_id": "ALTER TABLE learning_events ADD COLUMN request_id TEXT",
            "source": "ALTER TABLE learning_events ADD COLUMN source TEXT",
        }
        for column_name, sql in migrations.items():
            if column_name in existing:
                continue
            try:
                self.conn.execute(sql)
            except sqlite3.OperationalError:
                # already exists or not alterable in unusual states.
                pass
        self.conn.commit()

    def _ensure_learning_events_constraints(self) -> None:
        index_rows = self.conn.execute("PRAGMA index_list(learning_events)").fetchall()
        has_old_unique = False
        for row in index_rows:
            name = str(row["name"])
            if not name.startswith("sqlite_autoindex_learning_events_"):
                continue
            if int(row["unique"]) != 1:
                continue
            columns = [
                item["name"]
                for item in self.conn.execute(f"PRAGMA index_info({name})").fetchall()
            ]
            if columns == ["session_id", "event_type", "logical_step"]:
                has_old_unique = True
                break

        if not has_old_unique:
            return

        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE learning_events_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                run_id TEXT,
                request_id TEXT,
                session_id TEXT,
                event_type TEXT NOT NULL,
                logical_step TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL DEFAULT '{}',
                policy_class TEXT NOT NULL DEFAULT 'P2',
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, run_id, event_type, logical_step)
            )
            """
        )
        cursor.execute(
            """
            INSERT OR IGNORE INTO learning_events_v2 (
                event_id,
                run_id,
                request_id,
                session_id,
                event_type,
                logical_step,
                payload_json,
                policy_class,
                source,
                created_at
            )
            SELECT
                event_id,
                run_id,
                request_id,
                session_id,
                event_type,
                logical_step,
                payload_json,
                policy_class,
                source,
                created_at
            FROM learning_events
            """
        )
        cursor.execute("DROP TABLE learning_events")
        cursor.execute("ALTER TABLE learning_events_v2 RENAME TO learning_events")
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learning_events_session
            ON learning_events(session_id)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learning_events_run
            ON learning_events(run_id)
            """
        )
        self.conn.commit()

    def upsert_learning_session(
        self,
        session_id: str,
        topic_slug: str,
        target_trunk_ids: list[str],
        status: str = "draft",
    ) -> None:
        self.conn.execute(
            """INSERT INTO learning_sessions
                 (session_id, topic_slug, target_trunk_ids_json, status, updated_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(session_id) DO UPDATE SET
                 topic_slug=excluded.topic_slug,
                 target_trunk_ids_json=excluded.target_trunk_ids_json,
                 status=excluded.status,
                 updated_at=CURRENT_TIMESTAMP""",
            (session_id, topic_slug, json.dumps(target_trunk_ids, ensure_ascii=False), status),
        )
        self.conn.commit()

    def get_learning_session(self, session_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM learning_sessions WHERE session_id = ?", (session_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["target_trunk_ids_json"] = json.loads(item.get("target_trunk_ids_json") or "[]")
        except Exception:
            item["target_trunk_ids_json"] = []
        return item

    def list_learning_sessions(
        self,
        topic_slug: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        predicates: list[str] = []
        params: list[Any] = []
        if topic_slug:
            predicates.append("topic_slug = ?")
            params.append(str(topic_slug))
        if status:
            predicates.append("status = ?")
            params.append(str(status))
        where_clause = f" WHERE {' AND '.join(predicates)}" if predicates else ""
        rows = self.conn.execute(
            f"""SELECT * FROM learning_sessions
                {where_clause}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?""",
            [*params, max(1, int(limit))],
        ).fetchall()
        items: list[dict] = []
        for row in rows:
            item = dict(row)
            try:
                item["target_trunk_ids_json"] = json.loads(item.get("target_trunk_ids_json") or "[]")
            except Exception:
                item["target_trunk_ids_json"] = []
            items.append(item)
        return items

    def replace_learning_session_edges(self, session_id: str, edges: list[dict]) -> None:
        self.conn.execute("DELETE FROM learning_session_edges WHERE session_id = ?", (session_id,))
        if edges:
            rows = [
                (
                    session_id,
                    edge.get("source_canonical_id", "unknown"),
                    edge.get("relation_norm", "unknown_relation"),
                    edge.get("target_canonical_id", "unknown"),
                    json.dumps(edge.get("evidence_ptrs", []), ensure_ascii=False),
                    float(edge.get("confidence", 3.0)),
                    1 if edge.get("is_valid", False) else 0,
                )
                for edge in edges
            ]
            self.conn.executemany(
                """INSERT INTO learning_session_edges
                     (session_id, source_canonical_id, relation_norm, target_canonical_id,
                      evidence_ptrs_json, confidence, is_valid)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        self.conn.commit()

    def list_learning_session_edges(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM learning_session_edges
               WHERE session_id = ?
               ORDER BY id ASC""",
            (session_id,),
        ).fetchall()
        result: list[dict] = []
        for row in rows:
            item = dict(row)
            try:
                item["evidence_ptrs_json"] = json.loads(item.get("evidence_ptrs_json") or "[]")
            except Exception:
                item["evidence_ptrs_json"] = []
            result.append(item)
        return result

    def upsert_learning_progress(
        self,
        session_id: str,
        topic_slug: str,
        score_final: float,
        score_edge_accuracy: float,
        score_coverage: float,
        score_explanation_quality: float,
        gate_passed: bool,
        gate_status: str,
        weaknesses: list[dict],
        details: dict,
    ) -> None:
        self.conn.execute(
            """INSERT INTO learning_progress
                 (session_id, topic_slug, score_final, score_edge_accuracy, score_coverage,
                  score_explanation_quality, gate_passed, gate_status, weaknesses_json,
                  details_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(session_id) DO UPDATE SET
                 topic_slug=excluded.topic_slug,
                 score_final=excluded.score_final,
                 score_edge_accuracy=excluded.score_edge_accuracy,
                 score_coverage=excluded.score_coverage,
                 score_explanation_quality=excluded.score_explanation_quality,
                 gate_passed=excluded.gate_passed,
                 gate_status=excluded.gate_status,
                 weaknesses_json=excluded.weaknesses_json,
                 details_json=excluded.details_json,
                 updated_at=CURRENT_TIMESTAMP""",
            (
                session_id,
                topic_slug,
                float(score_final),
                float(score_edge_accuracy),
                float(score_coverage),
                float(score_explanation_quality),
                1 if gate_passed else 0,
                gate_status,
                json.dumps(weaknesses, ensure_ascii=False),
                json.dumps(details, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def get_learning_progress(self, session_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM learning_progress WHERE session_id = ?", (session_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["weaknesses_json"] = json.loads(item.get("weaknesses_json") or "[]")
        except Exception:
            item["weaknesses_json"] = []
        try:
            item["details_json"] = json.loads(item.get("details_json") or "{}")
        except Exception:
            item["details_json"] = {}
        return item

    def append_learning_event(
        self,
        event_id: str,
        event_type: str,
        logical_step: str,
        session_id: str | None,
        payload: dict,
        policy_class: str = "P2",
        run_id: str | None = None,
        request_id: str | None = None,
        source: str | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO learning_events
                 (event_id, run_id, request_id, session_id, event_type, logical_step, payload_json, policy_class, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                run_id,
                request_id,
                session_id,
                event_type,
                logical_step,
                json.dumps(payload, ensure_ascii=False),
                policy_class,
                source,
            ),
        )
        self.conn.commit()

    def list_learning_events(
        self,
        session_id: str | None = None,
        run_id: str | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        params: list[object] = []
        predicates: list[str] = []
        if session_id is not None:
            predicates.append("session_id = ?")
            params.append(session_id)
        if run_id is not None:
            predicates.append("run_id = ?")
            params.append(run_id)
        if source is not None:
            predicates.append("source = ?")
            params.append(source)

        where_clause = " WHERE " + " AND ".join(predicates) if predicates else ""
        sql = f"SELECT * FROM learning_events{where_clause} ORDER BY id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(sql, params).fetchall()

        result: list[dict] = []
        for row in rows:
            item = dict(row)
            try:
                item["payload_json"] = json.loads(item.get("payload_json") or "{}")
            except Exception:
                item["payload_json"] = {}
            result.append(item)
        return result

    def add_web_ontology_pending(
        self,
        run_id: str,
        topic_slug: str,
        note_id: str,
        source_url: str,
        source_canonical_id: str,
        relation_norm: str,
        target_canonical_id: str,
        confidence: float,
        evidence_ptrs: list[dict] | None = None,
        reason: dict | None = None,
        status: str = "pending",
    ) -> int:
        cursor = self.conn.execute(
            """INSERT OR IGNORE INTO web_ontology_pending
                 (run_id, topic_slug, note_id, source_url, source_canonical_id, relation_norm,
                  target_canonical_id, confidence, evidence_ptrs_json, reason_json, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                topic_slug,
                note_id,
                source_url,
                source_canonical_id,
                relation_norm,
                target_canonical_id,
                float(confidence),
                json.dumps(evidence_ptrs or [], ensure_ascii=False),
                json.dumps(reason or {}, ensure_ascii=False),
                status if status in {"pending", "approved", "rejected"} else "pending",
            ),
        )
        self.conn.commit()
        if cursor.lastrowid:
            return int(cursor.lastrowid)
        row = self.conn.execute(
            """SELECT id FROM web_ontology_pending
               WHERE run_id = ? AND note_id = ? AND source_canonical_id = ? AND relation_norm = ? AND target_canonical_id = ?""",
            (run_id, note_id, source_canonical_id, relation_norm, target_canonical_id),
        ).fetchone()
        return int(row["id"]) if row else 0

    def get_web_ontology_pending(self, pending_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM web_ontology_pending WHERE id = ?", (int(pending_id),)).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["evidence_ptrs_json"] = json.loads(item.get("evidence_ptrs_json") or "[]")
        except Exception:
            item["evidence_ptrs_json"] = []
        try:
            item["reason_json"] = json.loads(item.get("reason_json") or "{}")
        except Exception:
            item["reason_json"] = {}
        return item

    def list_web_ontology_pending(
        self,
        topic_slug: str | None = None,
        status: str | None = "pending",
        limit: int = 50,
    ) -> list[dict]:
        query = "SELECT * FROM web_ontology_pending WHERE 1=1"
        params: list[Any] = []
        if topic_slug:
            query += " AND topic_slug = ?"
            params.append(topic_slug)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        items: list[dict] = []
        for row in rows:
            item = dict(row)
            try:
                item["evidence_ptrs_json"] = json.loads(item.get("evidence_ptrs_json") or "[]")
            except Exception:
                item["evidence_ptrs_json"] = []
            try:
                item["reason_json"] = json.loads(item.get("reason_json") or "{}")
            except Exception:
                item["reason_json"] = {}
            items.append(item)
        return items

    def update_web_ontology_pending_status(self, pending_id: int, status: str) -> bool:
        status_value = status if status in {"pending", "approved", "rejected"} else "pending"
        cursor = self.conn.execute(
            """UPDATE web_ontology_pending
               SET status = ?, reviewed_at = CASE WHEN ? IN ('approved', 'rejected') THEN CURRENT_TIMESTAMP ELSE reviewed_at END
               WHERE id = ?""",
            (status_value, status_value, int(pending_id)),
        )
        self.conn.commit()
        return cursor.rowcount > 0
