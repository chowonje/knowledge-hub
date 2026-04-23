"""Belief / decision / outcome store for single-user epistemic loop."""

from __future__ import annotations

import json
from typing import Any


BELIEF_STATUSES = {"proposed", "reviewed", "trusted", "stale", "rejected"}
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
        status_value = str(status or "proposed").strip().lower()
        if status_value not in BELIEF_STATUSES:
            status_value = "proposed"
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
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM beliefs WHERE 1=1"
        params: list[Any] = []
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

    def list_beliefs_by_claim_ids(self, claim_ids: list[str], limit: int = 200) -> list[dict[str, Any]]:
        wanted = {str(item or "").strip() for item in claim_ids or [] if str(item or "").strip()}
        if not wanted:
            return []
        result: list[dict[str, Any]] = []
        for item in self.list_beliefs(limit=max(1, int(limit * 4))):
            if wanted & set(item.get("derived_from_claim_ids", [])):
                result.append(item)
            if len(result) >= limit:
                break
        return result[:limit]

    def review_belief(
        self,
        belief_id: str,
        *,
        status: str,
        last_validated_at: str | None = None,
        review_due_at: str | None = None,
    ) -> dict[str, Any] | None:
        belief = self.get_belief(belief_id)
        if not belief:
            return None
        self.upsert_belief(
            belief_id=belief_id,
            statement=str(belief.get("statement", "")),
            scope=str(belief.get("scope", "global")),
            status=status,
            confidence=float(belief.get("confidence", 0.5) or 0.5),
            derived_from_claim_ids=list(belief.get("derived_from_claim_ids", [])),
            support_ids=list(belief.get("support_ids", [])),
            contradiction_ids=list(belief.get("contradiction_ids", [])),
            last_validated_at=last_validated_at or belief.get("last_validated_at"),
            review_due_at=review_due_at or belief.get("review_due_at"),
        )
        return self.get_belief(belief_id)

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
        status_value = str(status or "open").strip().lower()
        if status_value not in DECISION_STATUSES:
            status_value = "open"
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

    def list_decisions(self, *, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        query = "SELECT * FROM decisions WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(str(status))
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        return [item for item in (self._decode_decision(row) for row in rows) if item]

    def review_decision(self, decision_id: str, *, status: str, review_due_at: str | None = None) -> dict[str, Any] | None:
        item = self.get_decision(decision_id)
        if not item:
            return None
        self.upsert_decision(
            decision_id=decision_id,
            title=str(item.get("title", "")),
            summary=str(item.get("summary", "")),
            related_belief_ids=list(item.get("related_belief_ids", [])),
            chosen_option=str(item.get("chosen_option", "")),
            status=status,
            review_due_at=review_due_at or item.get("review_due_at"),
        )
        return self.get_decision(decision_id)

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
