"""database.py 핵심 실패 경로 테스트"""

from __future__ import annotations

import sqlite3

import pytest

from knowledge_hub.core.database import SQLiteDatabase


class TestSQLiteWALMode:
    def test_wal_mode_enabled(self, tmp_path):
        db = SQLiteDatabase(str(tmp_path / "test.db"))
        mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_busy_timeout_set(self, tmp_path):
        db = SQLiteDatabase(str(tmp_path / "test.db"))
        timeout = db.conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout >= 5000


class TestSQLiteTransaction:
    def test_transaction_commits(self, tmp_path):
        db = SQLiteDatabase(str(tmp_path / "test.db"))
        with db.transaction():
            db.conn.execute(
                "INSERT INTO papers (arxiv_id, title) VALUES (?, ?)",
                ("2501.00001", "Test Paper"),
            )
        paper = db.get_paper("2501.00001")
        assert paper is not None
        assert paper["title"] == "Test Paper"

    def test_transaction_rollback_on_error(self, tmp_path):
        db = SQLiteDatabase(str(tmp_path / "test.db"))
        with pytest.raises(sqlite3.IntegrityError):
            with db.transaction():
                db.conn.execute(
                    "INSERT INTO papers (arxiv_id, title) VALUES (?, ?)",
                    ("2501.00002", "Paper A"),
                )
                db.conn.execute(
                    "INSERT INTO papers (arxiv_id, title) VALUES (?, ?)",
                    ("2501.00002", "Paper B Duplicate PK"),
                )
        paper = db.get_paper("2501.00002")
        assert paper is None


class TestSQLitePaperCRUD:
    def test_upsert_and_get(self, tmp_path):
        db = SQLiteDatabase(str(tmp_path / "test.db"))
        db.upsert_paper({
            "arxiv_id": "2501.12345",
            "title": "My Paper",
            "authors": "Author A",
            "year": 2025,
            "field": "CS",
            "importance": 3,
            "notes": "",
            "pdf_path": None,
            "text_path": None,
            "translated_path": None,
        })
        paper = db.get_paper("2501.12345")
        assert paper["title"] == "My Paper"
        assert paper["year"] == 2025

    def test_get_nonexistent_returns_none(self, tmp_path):
        db = SQLiteDatabase(str(tmp_path / "test.db"))
        assert db.get_paper("9999.99999") is None
