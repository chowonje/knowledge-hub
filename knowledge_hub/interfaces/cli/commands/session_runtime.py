"""Opt-in assistant session metadata helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

SESSION_EVENT_SCHEMA = "knowledge-hub.assistant.session.event.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_session_id(surface: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{surface}_{stamp}_{uuid4().hex[:6]}"


def text_hash(value: str) -> str:
    digest = hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def default_session_dir() -> Path:
    configured = str(os.environ.get("KHUB_SESSION_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".khub" / "sessions"


class SessionRecorder:
    """Record metadata-only assistant session events with SQLite as authority."""

    def __init__(self, *, session_id: str, surface: str, session_dir: Path | None = None):
        self.session_id = str(session_id)
        self.surface = str(surface)
        self.session_dir = session_dir or default_session_dir()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = self.session_dir / f"{self.session_id}.jsonl"
        self.metadata_path = self.session_dir / "sessions.sqlite"
        self._event_count = 0

    def start(self, *, provider: str, model: str, allow_external: bool, history_mode: str) -> None:
        self._ensure_metadata()
        created_at = utc_now()
        with sqlite3.connect(self.metadata_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO assistant_sessions (
                    session_id, surface, created_at, updated_at, transcript_path, history_mode, event_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.session_id,
                    self.surface,
                    created_at,
                    created_at,
                    str(self.transcript_path),
                    str(history_mode),
                    self._event_count,
                ),
            )
        self._append(
            {
                "type": "session_start",
                "sessionId": self.session_id,
                "createdAt": created_at,
                "surface": self.surface,
                "provider": str(provider),
                "model": str(model),
                "allowExternal": bool(allow_external),
                "historyMode": str(history_mode),
            }
        )

    def user_message(self, *, route: str, text: str) -> None:
        self._append(
            {
                "type": "user_message",
                "sessionId": self.session_id,
                "createdAt": utc_now(),
                "surface": self.surface,
                "route": str(route),
                "textHash": text_hash(text),
                "textChars": len(str(text or "")),
            }
        )

    def assistant_message(self, *, route: str, answer: str) -> None:
        self._append(
            {
                "type": "assistant_message",
                "sessionId": self.session_id,
                "createdAt": utc_now(),
                "surface": self.surface,
                "route": str(route),
                "answerHash": text_hash(answer),
                "answerChars": len(str(answer or "")),
            }
        )

    def route_metadata(self, *, route: str, metadata: dict[str, Any]) -> None:
        safe_metadata = dict(metadata or {})
        self._append(
            {
                "type": "route_metadata",
                "sessionId": self.session_id,
                "createdAt": utc_now(),
                "surface": self.surface,
                "route": str(route),
                **safe_metadata,
            }
        )

    def end(self) -> None:
        ended_at = utc_now()
        self._append(
            {
                "type": "session_end",
                "sessionId": self.session_id,
                "createdAt": ended_at,
                "surface": self.surface,
            }
        )
        self._update_metadata(updated_at=ended_at)

    def session_payload(self) -> dict[str, Any]:
        return {
            "id": self.session_id,
            "persisted": True,
            "historyMode": "sqlite-redacted-events",
            "metadataPath": str(self.metadata_path),
            "transcriptMirrorPath": str(self.transcript_path),
        }

    def _append(self, payload: dict[str, Any]) -> None:
        payload = {"schema": SESSION_EVENT_SCHEMA, **payload}
        self._ensure_metadata()
        created_at = str(payload.get("createdAt") or utc_now())
        event_type = str(payload.get("type") or "unknown")
        route = str(payload.get("route") or "")
        with self.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        with sqlite3.connect(self.metadata_path) as conn:
            conn.execute(
                """
                INSERT INTO assistant_session_events (
                    session_id, created_at, event_type, route, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    self.session_id,
                    created_at,
                    event_type,
                    route,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )
        self._event_count += 1
        self._update_metadata(updated_at=created_at)

    def _ensure_metadata(self) -> None:
        with sqlite3.connect(self.metadata_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assistant_sessions (
                    session_id TEXT PRIMARY KEY,
                    surface TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    transcript_path TEXT NOT NULL,
                    history_mode TEXT NOT NULL,
                    event_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assistant_session_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    route TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES assistant_sessions(session_id)
                )
                """
            )

    def _update_metadata(self, *, updated_at: str) -> None:
        if not self.metadata_path.exists():
            return
        with sqlite3.connect(self.metadata_path) as conn:
            conn.execute(
                """
                UPDATE assistant_sessions
                SET updated_at = ?, event_count = ?
                WHERE session_id = ?
                """,
                (updated_at, self._event_count, self.session_id),
            )
