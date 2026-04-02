"""Minimal migration framework for foundation schema evolution.

The current project still creates most tables in ``_init_tables`` for
compatibility. This manager adds a durable version table and explicit migration
markers so new schema changes can move out of application logic incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import sqlite3


@dataclass(frozen=True)
class Migration:
    version: str
    description: str
    apply: Callable[[any], None]


def _baseline(_conn) -> None:
    return


def _foundation_refactor_v1(_conn) -> None:
    return


def _add_column_if_missing(conn, table: str, column_name: str, column_sql: str) -> None:
    columns = {
        str(row["name"]) if isinstance(row, sqlite3.Row) else str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column_name in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_sql}")


def _predicate_semantics_columns(conn) -> None:
    _add_column_if_missing(conn, "ontology_predicates", "domain_source_type", "domain_source_type TEXT NOT NULL DEFAULT ''")
    _add_column_if_missing(conn, "ontology_predicates", "range_target_type", "range_target_type TEXT NOT NULL DEFAULT ''")
    _add_column_if_missing(conn, "ontology_predicates", "is_transitive", "is_transitive INTEGER NOT NULL DEFAULT 0")
    _add_column_if_missing(conn, "ontology_predicates", "is_symmetric", "is_symmetric INTEGER NOT NULL DEFAULT 0")
    _add_column_if_missing(conn, "ontology_predicates", "is_antisymmetric", "is_antisymmetric INTEGER NOT NULL DEFAULT 0")


MIGRATIONS: tuple[Migration, ...] = (
    Migration("20260307_000001", "baseline schema tracking", _baseline),
    Migration("20260307_000002", "foundation refactor v1 marker", _foundation_refactor_v1),
    Migration("20260307_000003", "predicate semantics columns", _predicate_semantics_columns),
)


class MigrationManager:
    def __init__(self, conn, db_path: Path):
        self.conn = conn
        self.db_path = Path(db_path)
        self.sql_dir = Path(__file__).resolve().parent / "migrations_sql"

    def ensure_version_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                description TEXT NOT NULL DEFAULT '',
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def applied_versions(self) -> set[str]:
        self.ensure_version_table()
        rows = self.conn.execute("SELECT version FROM schema_migrations").fetchall()
        return {str(row["version"]) for row in rows}

    def apply_pending_migrations(self) -> list[str]:
        self.ensure_version_table()
        applied = self.applied_versions()
        ran: list[str] = []
        for migration in MIGRATIONS:
            if migration.version in applied:
                continue
            migration.apply(self.conn)
            self.conn.execute(
                "INSERT INTO schema_migrations(version, description) VALUES (?, ?)",
                (migration.version, migration.description),
            )
            ran.append(migration.version)
            applied.add(migration.version)
        for sql_file in sorted(self.sql_dir.glob("*.sql")):
            version = sql_file.stem
            if version in applied:
                continue
            sql_text = sql_file.read_text(encoding="utf-8")
            self.conn.executescript(sql_text)
            self.conn.execute(
                "INSERT INTO schema_migrations(version, description) VALUES (?, ?)",
                (version, f"sql file {sql_file.name}"),
            )
            ran.append(version)
            applied.add(version)
        self.conn.commit()
        return ran
