"""Belief / decision / outcome store for single-user epistemic loop."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4


BELIEF_STATUSES = {"proposed", "reviewed", "trusted", "stale", "rejected", "superseded"}
DECISION_STATUSES = {"open", "committed", "reviewed", "closed"}
OUTCOME_STATUSES = {"observed", "confirmed", "invalidated"}


def _json_dumps_list(values: list[str] | None) -> str:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        token = str(raw or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return json.dumps(normalized, ensure_ascii=False)


def _json_loads_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item or "").strip()]
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item or "").strip()]


def _add_column_if_missing(conn, table: str, column_name: str, column_sql: str) -> None:
    columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column_name in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_sql}")


def _normalize_status(raw_status: str | None, *, allowed: set[str], default: str) -> str:
    value = str(raw_status or default).strip().lower()
    if value not in allowed:
        return default
    return value


def _new_version_id(prefix: str, explicit_id: str | None = None) -> str:
    token = str(explicit_id or "").strip()
    if token:
        return token
    return f"{prefix}_{uuid4().hex[:12]}"


class EpistemicStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS beliefs (
                belief_id TEXT PRIMARY KEY,
                statement TEXT NOT NULL,
                scope TEXT NOT NULL DEFAULT 'global',
                status TEXT NOT NULL DEFAULT 'proposed',
                confidence REAL NOT NULL DEFAULT 0.5,
                derived_from_claim_ids_json TEXT NOT NULL DEFAULT '[]',
                support_ids_json TEXT NOT NULL DEFAULT '[]',
                contradiction_ids_json TEXT NOT NULL DEFAULT '[]',
                last_validated_at TEXT,
                review_due_at TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT DEFAULT '',
                related_belief_ids_json TEXT NOT NULL DEFAULT '[]',
                chosen_option TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'open',
                review_due_at TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS outcomes (
                outcome_id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'observed',
                summary TEXT DEFAULT '',
                recorded_at TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _add_column_if_missing(self.conn, "beliefs", "supersedes", "supersedes TEXT NOT NULL DEFAULT ''")
        _add_column_if_missing(self.conn, "beliefs", "superseded_by", "superseded_by TEXT NOT NULL DEFAULT ''")
        _add_column_if_missing(self.conn, "decisions", "supersedes", "supersedes TEXT NOT NULL DEFAULT ''")
        _add_column_if_missing(self.conn, "decisions", "superseded_by", "superseded_by TEXT NOT NULL DEFAULT ''")
        self.conn.commit()

    def _decode_belief(self, row) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        item["derived_from_claim_ids"] = _json_loads_list(item.get("derived_from_claim_ids_json"))
        item["support_ids"] = _json_loads_list(item.get("support_ids_json"))
        item["contradiction_ids"] = _json_loads_list(item.get("contradiction_ids_json"))
        return item

    def _resolve_latest_belief_id(self, belief_id: str) -> str | None:
        current_id = str(belief_id or "").strip()
        if not current_id:
            return None
        seen: set[str] = set()
        while current_id and current_id not in seen:
            seen.add(current_id)
            row = self.conn.execute(
                "SELECT belief_id, superseded_by FROM beliefs WHERE belief_id = ?",
                (current_id,),
            ).fetchone()
            if not row:
                return None
            successor_id = str(row["superseded_by"] or "").strip()
            if not successor_id:
                return str(row["belief_id"])
            current_id = successor_id
        return current_id or None

    def _resolve_latest_decision_id(self, decision_id: str) -> str | None:
        current_id = str(decision_id or "").strip()
        if not current_id:
            return None
        seen: set[str] = set()
        while current_id and current_id not in seen:
            seen.add(current_id)
            row = self.conn.execute(
                "SELECT decision_id, superseded_by FROM decisions WHERE decision_id = ?",
                (current_id,),
            ).fetchone()
            if not row:
                return None
            successor_id = str(row["superseded_by"] or "").strip()
            if not successor_id:
                return str(row["decision_id"])
            current_id = successor_id
        return current_id or None

    def upsert_belief(
        self,
        *,
        belief_id: str,
        statement: str,
        scope: str = "global",
        status: str = "proposed",
        confidence: float = 0.5,
        derived_from_claim_ids: list[str] | None = None,
        support_ids: list[str] | None = None,
        contradiction_ids: list[str] | None = None,
        last_validated_at: str | None = None,
        review_due_at: str | None = None,
    ) -> None:
        status_value = _normalize_status(status, allowed=BELIEF_STATUSES, default="proposed")
        self.conn.execute(
            """INSERT INTO beliefs
               (belief_id, statement, scope, status, confidence,
                derived_from_claim_ids_json, support_ids_json, contradiction_ids_json,
                last_validated_at, review_due_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(belief_id) DO UPDATE SET
                 statement=excluded.statement,
                 scope=excluded.scope,
                 status=excluded.status,
                 confidence=excluded.confidence,
                 derived_from_claim_ids_json=excluded.derived_from_claim_ids_json,
                 support_ids_json=excluded.support_ids_json,
                 contradiction_ids_json=excluded.contradiction_ids_json,
                 last_validated_at=excluded.last_validated_at,
                 review_due_at=excluded.review_due_at,
                 updated_at=CURRENT_TIMESTAMP""",
            (
                str(belief_id),
                str(statement),
                str(scope or "global"),
                status_value,
                float(max(0.0, min(1.0, confidence))),
                _json_dumps_list(derived_from_claim_ids),
                _json_dumps_list(support_ids),
                _json_dumps_list(contradiction_ids),
                last_validated_at,
                review_due_at,
            ),
        )
        self.conn.commit()

    def get_belief(self, belief_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM beliefs WHERE belief_id = ?", (str(belief_id),)).fetchone()
        return self._decode_belief(row)

    def list_beliefs(
        self,
        *,
        status: str | None = None,
        scope: str | None = None,
        limit: int = 200,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM beliefs WHERE 1=1"
        params: list[Any] = []
        if not include_superseded:
            query += " AND COALESCE(status, '') != 'superseded'"
        if status:
            query += " AND status = ?"
            params.append(str(status))
        if scope:
            query += " AND scope = ?"
            params.append(str(scope))
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        return [item for item in (self._decode_belief(row) for row in rows) if item]

    def list_beliefs_by_claim_ids(
        self,
        claim_ids: list[str],
        limit: int = 200,
        *,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        wanted = {str(item or "").strip() for item in claim_ids or [] if str(item or "").strip()}
        if not wanted:
            return []
        result: list[dict[str, Any]] = []
        for item in self.list_beliefs(limit=max(1, int(limit * 4)), include_superseded=include_superseded):
            if wanted & set(item.get("derived_from_claim_ids", [])):
                result.append(item)
            if len(result) >= limit:
                break
        return result[:limit]

    def supersede_belief(
        self,
        belief_id: str,
        *,
        status: str,
        successor_belief_id: str | None = None,
        last_validated_at: str | None = None,
        review_due_at: str | None = None,
    ) -> dict[str, Any] | None:
        latest_belief_id = self._resolve_latest_belief_id(belief_id)
        if not latest_belief_id:
            return None
        belief = self.get_belief(latest_belief_id)
        if not belief:
            return None
        successor_id = _new_version_id("belief", successor_belief_id)
        if successor_id == latest_belief_id:
            raise ValueError("successor belief_id must differ from the superseded belief_id")
        if self.get_belief(successor_id):
            raise ValueError(f"belief already exists: {successor_id}")
        status_value = _normalize_status(status, allowed=BELIEF_STATUSES - {"superseded"}, default="proposed")
        with self.conn:
            self.conn.execute(
                """
                UPDATE beliefs
                   SET status = 'superseded',
                       superseded_by = ?,
                       updated_at = CURRENT_TIMESTAMP
                 WHERE belief_id = ?
                """,
                (successor_id, latest_belief_id),
            )
            self.conn.execute(
                """INSERT INTO beliefs
                   (belief_id, statement, scope, status, confidence,
                    derived_from_claim_ids_json, support_ids_json, contradiction_ids_json,
                    last_validated_at, review_due_at, supersedes, superseded_by, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', CURRENT_TIMESTAMP)""",
                (
                    successor_id,
                    str(belief.get("statement", "")),
                    str(belief.get("scope", "global")),
                    status_value,
                    float(belief.get("confidence", 0.5) or 0.5),
                    _json_dumps_list(list(belief.get("derived_from_claim_ids", []))),
                    _json_dumps_list(list(belief.get("support_ids", []))),
                    _json_dumps_list(list(belief.get("contradiction_ids", []))),
                    last_validated_at or belief.get("last_validated_at"),
                    review_due_at or belief.get("review_due_at"),
                    latest_belief_id,
                ),
            )
        return self.get_belief(successor_id)

    def review_belief(
        self,
        belief_id: str,
        *,
        status: str,
        last_validated_at: str | None = None,
        review_due_at: str | None = None,
        successor_belief_id: str | None = None,
    ) -> dict[str, Any] | None:
        return self.supersede_belief(
            belief_id,
            status=status,
            successor_belief_id=successor_belief_id,
            last_validated_at=last_validated_at,
            review_due_at=review_due_at,
        )

    def _decode_decision(self, row) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        item["related_belief_ids"] = _json_loads_list(item.get("related_belief_ids_json"))
        return item

    def upsert_decision(
        self,
        *,
        decision_id: str,
        title: str,
        summary: str = "",
        related_belief_ids: list[str] | None = None,
        chosen_option: str = "",
        status: str = "open",
        review_due_at: str | None = None,
    ) -> None:
        status_value = _normalize_status(status, allowed=DECISION_STATUSES, default="open")
        self.conn.execute(
            """INSERT INTO decisions
               (decision_id, title, summary, related_belief_ids_json, chosen_option, status, review_due_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(decision_id) DO UPDATE SET
                 title=excluded.title,
                 summary=excluded.summary,
                 related_belief_ids_json=excluded.related_belief_ids_json,
                 chosen_option=excluded.chosen_option,
                 status=excluded.status,
                 review_due_at=excluded.review_due_at,
                 updated_at=CURRENT_TIMESTAMP""",
            (
                str(decision_id),
                str(title),
                str(summary or ""),
                _json_dumps_list(related_belief_ids),
                str(chosen_option or ""),
                status_value,
                review_due_at,
            ),
        )
        self.conn.commit()

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM decisions WHERE decision_id = ?", (str(decision_id),)).fetchone()
        return self._decode_decision(row)

    def list_decisions(
        self,
        *,
        status: str | None = None,
        limit: int = 200,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM decisions WHERE 1=1"
        params: list[Any] = []
        if not include_superseded:
            query += " AND COALESCE(superseded_by, '') = ''"
        if status:
            query += " AND status = ?"
            params.append(str(status))
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        return [item for item in (self._decode_decision(row) for row in rows) if item]

    def supersede_decision(
        self,
        decision_id: str,
        *,
        status: str,
        successor_decision_id: str | None = None,
        review_due_at: str | None = None,
    ) -> dict[str, Any] | None:
        latest_decision_id = self._resolve_latest_decision_id(decision_id)
        if not latest_decision_id:
            return None
        item = self.get_decision(latest_decision_id)
        if not item:
            return None
        successor_id = _new_version_id("decision", successor_decision_id)
        if successor_id == latest_decision_id:
            raise ValueError("successor decision_id must differ from the superseded decision_id")
        if self.get_decision(successor_id):
            raise ValueError(f"decision already exists: {successor_id}")
        status_value = _normalize_status(status, allowed=DECISION_STATUSES, default="open")
        with self.conn:
            self.conn.execute(
                """
                UPDATE decisions
                   SET superseded_by = ?,
                       updated_at = CURRENT_TIMESTAMP
                 WHERE decision_id = ?
                """,
                (successor_id, latest_decision_id),
            )
            self.conn.execute(
                """INSERT INTO decisions
                   (decision_id, title, summary, related_belief_ids_json, chosen_option,
                    status, review_due_at, supersedes, superseded_by, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', CURRENT_TIMESTAMP)""",
                (
                    successor_id,
                    str(item.get("title", "")),
                    str(item.get("summary", "")),
                    _json_dumps_list(list(item.get("related_belief_ids", []))),
                    str(item.get("chosen_option", "")),
                    status_value,
                    review_due_at or item.get("review_due_at"),
                    latest_decision_id,
                ),
            )
        return self.get_decision(successor_id)

    def review_decision(
        self,
        decision_id: str,
        *,
        status: str,
        review_due_at: str | None = None,
        successor_decision_id: str | None = None,
    ) -> dict[str, Any] | None:
        return self.supersede_decision(
            decision_id,
            status=status,
            successor_decision_id=successor_decision_id,
            review_due_at=review_due_at,
        )

    def record_outcome(
        self,
        *,
        outcome_id: str,
        decision_id: str,
        status: str = "observed",
        summary: str = "",
        recorded_at: str,
    ) -> None:
        status_value = str(status or "observed").strip().lower()
        if status_value not in OUTCOME_STATUSES:
            status_value = "observed"
        self.conn.execute(
            """INSERT INTO outcomes
               (outcome_id, decision_id, status, summary, recorded_at, updated_at)
               VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(outcome_id) DO UPDATE SET
                 decision_id=excluded.decision_id,
                 status=excluded.status,
                 summary=excluded.summary,
                 recorded_at=excluded.recorded_at,
                 updated_at=CURRENT_TIMESTAMP""",
            (
                str(outcome_id),
                str(decision_id),
                status_value,
                str(summary or ""),
                str(recorded_at),
            ),
        )
        self.conn.commit()

    def get_outcome(self, outcome_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM outcomes WHERE outcome_id = ?", (str(outcome_id),)).fetchone()
        return dict(row) if row else None

    def list_outcomes(self, *, decision_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        query = "SELECT * FROM outcomes WHERE 1=1"
        params: list[Any] = []
        if decision_id:
            query += " AND decision_id = ?"
            params.append(str(decision_id))
        query += " ORDER BY recorded_at DESC, updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
