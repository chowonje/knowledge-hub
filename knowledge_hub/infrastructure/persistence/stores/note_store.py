"""Canonical note/tag/link/PARA store helpers."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from knowledge_hub.infrastructure.persistence.stores.derivative_lifecycle import (
    mark_derivatives_stale_for_document,
    source_hash_from_content,
    source_hash_from_payload,
)


class NoteStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT,
                file_path TEXT,
                source_type TEXT DEFAULT 'note',
                para_category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                starred INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                color TEXT DEFAULT '#6366f1'
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS note_tags (
                note_id TEXT REFERENCES notes(id) ON DELETE CASCADE,
                tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
                PRIMARY KEY (note_id, tag_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS links (
                source_id TEXT REFERENCES notes(id) ON DELETE CASCADE,
                target_id TEXT REFERENCES notes(id) ON DELETE CASCADE,
                link_type TEXT DEFAULT 'related',
                strength REAL DEFAULT 0.5,
                PRIMARY KEY (source_id, target_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS para_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                color TEXT DEFAULT '#6366f1',
                icon TEXT,
                sort_order INTEGER DEFAULT 0
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS system_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT '',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def upsert_note(
        self,
        note_id: str,
        title: str,
        content: str = "",
        file_path: str = "",
        source_type: str = "note",
        para_category: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        metadata_payload = dict(metadata or {})
        source_hash = source_hash_from_content(
            content=content,
            metadata=metadata_payload,
            identity=note_id,
        )
        if source_hash and not str(metadata_payload.get("source_content_hash") or "").strip():
            metadata_payload["source_content_hash"] = source_hash
        if source_hash:
            mark_derivatives_stale_for_document(
                self.conn,
                document_id=str(note_id or "").strip(),
                source_content_hash=source_hash,
                source_type=str(source_type or "").strip(),
            )
        self.conn.execute(
            """INSERT INTO notes (id, title, content, file_path, source_type, para_category, metadata, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(id) DO UPDATE SET
                 title=excluded.title, content=excluded.content,
                 file_path=excluded.file_path, source_type=excluded.source_type,
                 para_category=excluded.para_category, metadata=excluded.metadata,
                 updated_at=CURRENT_TIMESTAMP""",
            (
                note_id,
                title,
                content,
                file_path,
                source_type,
                para_category,
                json.dumps(metadata_payload, ensure_ascii=False, default=str),
            ),
        )
        self.conn.commit()

    def get_note(self, note_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return dict(row) if row else None

    def list_notes(
        self,
        source_type: str | None = None,
        para_category: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        query = "SELECT * FROM notes WHERE 1=1"
        params: list[Any] = []
        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)
        if para_category:
            query += " AND para_category = ?"
            params.append(para_category)
        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def delete_note(self, note_id: str) -> None:
        self.conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self.conn.commit()

    def delete_notes(self, note_ids: list[str]) -> int:
        ids = [str(note_id or "").strip() for note_id in note_ids if str(note_id or "").strip()]
        if not ids:
            return 0
        placeholders = ", ".join(["?"] * len(ids))
        params = list(ids)
        self.conn.execute(f"DELETE FROM note_tags WHERE note_id IN ({placeholders})", params)
        self.conn.execute(
            f"DELETE FROM links WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
            params + params,
        )
        cursor = self.conn.execute(f"DELETE FROM notes WHERE id IN ({placeholders})", params)
        self.conn.commit()
        return int(cursor.rowcount or 0)

    def merge_note_metadata(self, note_id: str, patch: dict) -> bool:
        row = self.conn.execute("SELECT metadata, content, source_type FROM notes WHERE id = ?", (note_id,)).fetchone()
        if not row:
            return False
        current_raw = row["metadata"] if isinstance(row, sqlite3.Row) else row[0]
        try:
            current = json.loads(current_raw or "{}")
        except Exception:
            current = {}
        if not isinstance(current, dict):
            current = {}
        current.update(patch or {})
        patched_hash = source_hash_from_payload(dict(patch or {}))
        if patched_hash:
            current["source_content_hash"] = patched_hash
            mark_derivatives_stale_for_document(
                self.conn,
                document_id=str(note_id or "").strip(),
                source_content_hash=patched_hash,
                source_type=str(row["source_type"] if isinstance(row, sqlite3.Row) else row[2] or "").strip(),
            )
        self.conn.execute(
            "UPDATE notes SET metadata = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (json.dumps(current, ensure_ascii=False, default=str), note_id),
        )
        self.conn.commit()
        return True

    def search_notes(self, query: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM notes WHERE title LIKE ? OR content LIKE ? ORDER BY updated_at DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def ensure_tag(self, name: str) -> int:
        row = self.conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()
        if row:
            return row[0]
        cursor = self.conn.execute("INSERT INTO tags (name) VALUES (?)", (name,))
        self.conn.commit()
        return cursor.lastrowid

    def add_note_tag(self, note_id: str, tag_name: str) -> None:
        tag_id = self.ensure_tag(tag_name)
        self.conn.execute(
            "INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)",
            (note_id, tag_id),
        )
        self.conn.commit()

    def replace_note_tags(self, note_id: str, tag_names: list[str]) -> None:
        tags = []
        seen: set[str] = set()
        for raw in tag_names or []:
            tag = str(raw or "").strip()
            if not tag or tag in seen:
                continue
            seen.add(tag)
            tags.append(tag)

        self.conn.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
        for tag in tags:
            tag_id = self.ensure_tag(tag)
            self.conn.execute(
                "INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)",
                (note_id, tag_id),
            )
        self.conn.commit()

    def get_note_tags(self, note_id: str) -> list[str]:
        rows = self.conn.execute(
            """SELECT t.name FROM tags t
               JOIN note_tags nt ON t.id = nt.tag_id
               WHERE nt.note_id = ?""",
            (note_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def list_tags(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT t.name, t.color, COUNT(nt.note_id) as count
               FROM tags t LEFT JOIN note_tags nt ON t.id = nt.tag_id
               GROUP BY t.id ORDER BY count DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def add_link(self, source_id: str, target_id: str, link_type: str = "related") -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO links (source_id, target_id, link_type) VALUES (?, ?, ?)",
            (source_id, target_id, link_type),
        )
        self.conn.commit()

    def clear_links_for_source(self, source_id: str, link_type: str | None = None) -> int:
        if link_type:
            cursor = self.conn.execute(
                "DELETE FROM links WHERE source_id = ? AND link_type = ?",
                (source_id, link_type),
            )
        else:
            cursor = self.conn.execute("DELETE FROM links WHERE source_id = ?", (source_id,))
        self.conn.commit()
        return int(cursor.rowcount or 0)

    def replace_links_for_source(self, source_id: str, target_ids: list[str], link_type: str = "related") -> None:
        self.clear_links_for_source(source_id, link_type=link_type)
        targets = []
        seen: set[str] = set()
        for raw in target_ids or []:
            target = str(raw or "").strip()
            if not target or target in seen:
                continue
            seen.add(target)
            targets.append(target)

        for target in targets:
            self.conn.execute(
                "INSERT OR IGNORE INTO links (source_id, target_id, link_type) VALUES (?, ?, ?)",
                (source_id, target, link_type),
            )
        self.conn.commit()

    def delete_links_for_note_ids(self, note_ids: list[str]) -> int:
        ids = [str(note_id or "").strip() for note_id in note_ids if str(note_id or "").strip()]
        if not ids:
            return 0
        placeholders = ", ".join(["?"] * len(ids))
        cursor = self.conn.execute(
            f"DELETE FROM links WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
            ids + ids,
        )
        self.conn.commit()
        return int(cursor.rowcount or 0)

    def get_links(self, note_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM links WHERE source_id = ? OR target_id = ?",
            (note_id, note_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_para_categories(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM para_categories ORDER BY sort_order").fetchall()
        return [dict(r) for r in rows]

    def ensure_default_para(self) -> None:
        row = self.conn.execute("SELECT COUNT(*) FROM para_categories").fetchone()
        count = int(row[0] if row else 0)
        if count > 0:
            return
        defaults = [
            ("project", "Projects", "활발히 진행 중인 프로젝트", "#3b82f6", "folder"),
            ("area", "Areas", "지속적으로 관리하는 영역", "#10b981", "layers"),
            ("resource", "Resources", "참고 자료와 리소스", "#f59e0b", "book-open"),
            ("archive", "Archives", "완료되거나 보관된 항목", "#6b7280", "archive"),
        ]
        self.conn.executemany(
            "INSERT INTO para_categories (type, name, description, color, icon) VALUES (?, ?, ?, ?, ?)",
            defaults,
        )
        self.conn.commit()

    def set_system_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            """INSERT INTO system_meta(key, value, updated_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(key) DO UPDATE SET
                 value=excluded.value,
                 updated_at=CURRENT_TIMESTAMP""",
            (str(key), str(value)),
        )
        self.conn.commit()

    def get_system_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM system_meta WHERE key = ?", (str(key),)).fetchone()
        if not row:
            return default
        return str(row["value"] or default)

    def get_para_stats(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT para_category, COUNT(*) as cnt FROM notes WHERE para_category IS NOT NULL GROUP BY para_category"
        ).fetchall()
        return {r["para_category"]: r["cnt"] for r in rows}

    def get_graph_data(self) -> dict[str, Any]:
        notes = self.conn.execute("SELECT id, title, source_type, para_category FROM notes").fetchall()
        links = self.conn.execute("SELECT * FROM links").fetchall()
        nodes = [
            {
                "id": n["id"],
                "label": n["title"],
                "type": n["source_type"],
                "group": n["para_category"] or "none",
            }
            for n in notes
        ]
        edges = [
            {
                "source": link["source_id"],
                "target": link["target_id"],
                "type": link["link_type"],
            }
            for link in links
        ]
        return {"nodes": nodes, "edges": edges}

    def get_stats(self) -> dict[str, Any]:
        note_count = self.conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        paper_count = self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        tag_count = self.conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
        link_count = self.conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
        return {
            "notes": note_count,
            "papers": paper_count,
            "tags": tag_count,
            "links": link_count,
        }
