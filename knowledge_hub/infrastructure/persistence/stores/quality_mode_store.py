from __future__ import annotations

import sqlite3
from datetime import datetime


class QualityModeStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quality_mode_usage (
                usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                month_key TEXT NOT NULL,
                item_kind TEXT NOT NULL,
                route TEXT NOT NULL,
                estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
                topic_slug TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_quality_mode_usage_month
            ON quality_mode_usage(month_key, item_kind, route)
            """
        )
        self.conn.commit()

    def record_usage(
        self,
        item_kind: str,
        route: str,
        estimated_cost_usd: float,
        topic_slug: str = "",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO quality_mode_usage (
                month_key,
                item_kind,
                route,
                estimated_cost_usd,
                topic_slug
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                self._month_key(),
                item_kind,
                route,
                float(estimated_cost_usd),
                topic_slug or "",
            ),
        )
        self.conn.commit()

    def get_monthly_spend(self, month_key: str | None = None) -> float:
        row = self.conn.execute(
            """
            SELECT COALESCE(SUM(estimated_cost_usd), 0.0)
            FROM quality_mode_usage
            WHERE month_key = ?
            """,
            (self._month_key(month_key),),
        ).fetchone()
        return float(row[0] if row else 0.0)

    @staticmethod
    def _month_key(month_key: str | None = None) -> str:
        if month_key:
            return month_key
        return datetime.now().strftime("%Y-%m")
