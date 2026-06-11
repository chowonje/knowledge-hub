"""Persistent user-judgment ledger (V0 judgment loop).

JSONL (``judgments.jsonl`` next to the sqlite db) is the append-only ground
truth and is always written FIRST; the ``judgments_v1`` sqlite table is a
query index mirror, following the EventStore jsonl-first pattern.

Decision vocabulary vs the review-loop ``ReviewDecision`` enum
(``application/research_review_loop_types.py``):

- ``accept``  -> review-loop ``accept``
- ``thin``    -> roughly ``needs_more_evidence`` but the artifact is still usable
- ``reject``  -> review-loop ``reject``
- ``unsure``  -> review-loop ``unsure``
- ``abstain`` -> no review-loop equivalent; judges that the system's
  fail-closed/abstain outcome was the correct behavior
- ``archive`` -> review-loop ``archive``

Judgments are never mutated: corrections are recorded as a new judgment via
``supersede_judgment`` which appends a new jsonl line carrying ``supersedes``
and only updates the old mirror row's ``superseded_by`` pointer.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

VALID_TARGET_TYPES = frozenset({"claim", "answer", "brief"})
VALID_DECISIONS = frozenset({"accept", "thin", "reject", "unsure", "abstain", "archive"})
VALID_CONFIDENCES = frozenset({"low", "medium", "high"})
VALID_DECISION_SOURCES = frozenset({"cli", "mcp", "decision_file", "manual_worklog"})

_JSON_LIST_FIELDS = ("snippet_hashes", "evidence_span_ids", "source_ids")
_JSON_DICT_FIELDS = ("verification", "payload")


def _hash_target_text(text: str) -> str:
    """Same convention as research_review_loop_decisions.claim_text_hash."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _normalize_iso_timestamp(raw: Any, *, field: str) -> str:
    token = str(raw or "").strip()
    if not token:
        return datetime.now(timezone.utc).isoformat()
    normalized = token.replace("Z", "+00:00").replace(" ", "T")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp, got {token!r}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def _string_list(value: Any) -> list[str]:
    return [str(item).strip() for item in (value or []) if str(item or "").strip()]


class JudgmentStore:
    """Append-only judgment ledger: jsonl ground truth + sqlite mirror."""

    def __init__(self, conn: sqlite3.Connection, *, db_path: str | Path):
        self.conn = conn
        self.db_path = Path(db_path)
        self.jsonl_path = self.db_path.parent / "judgments.jsonl"

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS judgments_v1 (
                judgment_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                confidence TEXT NOT NULL DEFAULT 'medium',
                reviewer TEXT NOT NULL,
                reviewed_at TEXT NOT NULL,
                reason TEXT NOT NULL,
                target_text_hash TEXT NOT NULL DEFAULT '',
                snippet_hashes_json TEXT NOT NULL DEFAULT '[]',
                evidence_span_ids_json TEXT NOT NULL DEFAULT '[]',
                source_ids_json TEXT NOT NULL DEFAULT '[]',
                query_hash TEXT NOT NULL DEFAULT '',
                query_digest TEXT NOT NULL DEFAULT '',
                rag_answer_log_id INTEGER,
                runtime_used TEXT NOT NULL DEFAULT '',
                verification_json TEXT NOT NULL DEFAULT '{}',
                decision_source TEXT NOT NULL DEFAULT 'manual_worklog',
                supersedes TEXT NOT NULL DEFAULT '',
                superseded_by TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_judgments_v1_created_at ON judgments_v1(created_at DESC)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_judgments_v1_target ON judgments_v1(target_type, target_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_judgments_v1_query_hash ON judgments_v1(query_hash)"
        )
        self.conn.commit()

    def _validate(self, record: dict[str, Any]) -> None:
        if record["target_type"] not in VALID_TARGET_TYPES:
            raise ValueError(
                f"target_type must be one of {sorted(VALID_TARGET_TYPES)}, got {record['target_type']!r}"
            )
        if record["decision"] not in VALID_DECISIONS:
            raise ValueError(
                f"decision must be one of {sorted(VALID_DECISIONS)}, got {record['decision']!r}"
            )
        if record["confidence"] not in VALID_CONFIDENCES:
            raise ValueError(
                f"confidence must be one of {sorted(VALID_CONFIDENCES)}, got {record['confidence']!r}"
            )
        if record["decision_source"] not in VALID_DECISION_SOURCES:
            raise ValueError(
                f"decision_source must be one of {sorted(VALID_DECISION_SOURCES)}, "
                f"got {record['decision_source']!r}"
            )
        for field in ("target_id", "reviewer", "reason"):
            if not str(record[field] or "").strip():
                raise ValueError(f"{field} must be a non-empty string")

    def record_judgment(
        self,
        *,
        target_type: str,
        target_id: str,
        decision: str,
        reviewer: str,
        reason: str,
        confidence: str = "medium",
        reviewed_at: str | None = None,
        target_text: str | None = None,
        target_text_hash: str = "",
        snippet_hashes: list[str] | None = None,
        evidence_span_ids: list[str] | None = None,
        source_ids: list[str] | None = None,
        query_hash: str = "",
        query_digest: str = "",
        rag_answer_log_id: int | None = None,
        runtime_used: str = "",
        verification: dict[str, Any] | None = None,
        decision_source: str = "manual_worklog",
        supersedes: str = "",
        judgment_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if target_text is not None and str(target_text):
            target_text_hash = _hash_target_text(str(target_text))
        record: dict[str, Any] = {
            "judgment_id": str(judgment_id or "").strip() or f"judgment_{uuid4().hex[:12]}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_type": str(target_type or "").strip(),
            "target_id": str(target_id or "").strip(),
            "decision": str(decision or "").strip(),
            "confidence": str(confidence or "").strip() or "medium",
            "reviewer": str(reviewer or "").strip(),
            "reviewed_at": _normalize_iso_timestamp(reviewed_at, field="reviewed_at"),
            "reason": str(reason or "").strip(),
            "target_text_hash": str(target_text_hash or "").strip(),
            "snippet_hashes": _string_list(snippet_hashes),
            "evidence_span_ids": _string_list(evidence_span_ids),
            "source_ids": _string_list(source_ids),
            "query_hash": str(query_hash or "").strip(),
            "query_digest": str(query_digest or "").strip(),
            "rag_answer_log_id": int(rag_answer_log_id) if rag_answer_log_id is not None else None,
            "runtime_used": str(runtime_used or "").strip(),
            "verification": dict(verification or {}),
            "decision_source": str(decision_source or "").strip(),
            "supersedes": str(supersedes or "").strip(),
            "superseded_by": "",
            "payload": dict(payload or {}),
        }
        self._validate(record)

        # 1) JSONL append (ground truth, written first).
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 2) sqlite index mirror.
        self.conn.execute(
            """
            INSERT INTO judgments_v1 (
                judgment_id, created_at, target_type, target_id, decision, confidence,
                reviewer, reviewed_at, reason, target_text_hash,
                snippet_hashes_json, evidence_span_ids_json, source_ids_json,
                query_hash, query_digest, rag_answer_log_id, runtime_used,
                verification_json, decision_source, supersedes, superseded_by, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["judgment_id"],
                record["created_at"],
                record["target_type"],
                record["target_id"],
                record["decision"],
                record["confidence"],
                record["reviewer"],
                record["reviewed_at"],
                record["reason"],
                record["target_text_hash"],
                json.dumps(record["snippet_hashes"], ensure_ascii=False),
                json.dumps(record["evidence_span_ids"], ensure_ascii=False),
                json.dumps(record["source_ids"], ensure_ascii=False),
                record["query_hash"],
                record["query_digest"],
                record["rag_answer_log_id"],
                record["runtime_used"],
                json.dumps(record["verification"], ensure_ascii=False),
                record["decision_source"],
                record["supersedes"],
                record["superseded_by"],
                json.dumps(record["payload"], ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return record

    @staticmethod
    def _decode_row(row: Any) -> dict[str, Any]:
        item = dict(row)
        for field in _JSON_LIST_FIELDS:
            raw = item.pop(f"{field}_json", "[]")
            try:
                parsed = json.loads(raw or "[]")
            except Exception:
                parsed = []
            item[field] = parsed if isinstance(parsed, list) else []
        for field in _JSON_DICT_FIELDS:
            raw = item.pop(f"{field}_json", "{}")
            try:
                parsed = json.loads(raw or "{}")
            except Exception:
                parsed = {}
            item[field] = parsed if isinstance(parsed, dict) else {}
        return item

    def list_judgments(
        self,
        *,
        limit: int = 50,
        target_type: str | None = None,
        decision: str | None = None,
        query_hash: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM judgments_v1"
        clauses: list[str] = []
        params: list[Any] = []
        if target_type:
            clauses.append("target_type = ?")
            params.append(str(target_type))
        if decision:
            clauses.append("decision = ?")
            params.append(str(decision))
        if query_hash:
            clauses.append("query_hash = ?")
            params.append(str(query_hash))
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, judgment_id DESC LIMIT ?"
        params.append(max(1, int(limit or 50)))
        try:
            rows = self.conn.execute(query, params).fetchall()
        except sqlite3.OperationalError:
            # Read-only connections skip bootstrap; tolerate a missing table.
            return []
        return [self._decode_row(row) for row in rows]

    def get_judgment(self, judgment_id: str) -> dict[str, Any] | None:
        try:
            row = self.conn.execute(
                "SELECT * FROM judgments_v1 WHERE judgment_id = ?",
                (str(judgment_id or "").strip(),),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return self._decode_row(row) if row else None

    def supersede_judgment(self, old_judgment_id: str, **new_fields: Any) -> dict[str, Any]:
        old_id = str(old_judgment_id or "").strip()
        old_record = self.get_judgment(old_id)
        if old_record is None:
            raise ValueError(f"judgment not found: {old_id!r}")
        new_fields.setdefault("target_type", old_record["target_type"])
        new_fields.setdefault("target_id", old_record["target_id"])
        new_fields["supersedes"] = old_id
        new_record = self.record_judgment(**new_fields)
        # jsonl stays append-only; only the old mirror row gains the pointer.
        self.conn.execute(
            "UPDATE judgments_v1 SET superseded_by = ? WHERE judgment_id = ?",
            (new_record["judgment_id"], old_id),
        )
        self.conn.commit()
        return new_record
