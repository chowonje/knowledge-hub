from __future__ import annotations

from collections import Counter
from pathlib import Path
import sqlite3
from typing import Any

from knowledge_hub.infrastructure.persistence.vector import list_vector_documents


SOURCE_TYPES = ("paper", "vault", "web")


def _normalize_source_type(value: Any) -> str:
    token = str(value or "").strip().lower()
    return "vault" if token == "note" else token


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def collect_sqlite_source_counts(sqlite_path: str | Path) -> dict[str, int]:
    path = Path(sqlite_path).expanduser()
    if not path.exists():
        return {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1)
    try:
        counts: Counter[str] = Counter()
        if _table_exists(conn, "papers"):
            counts["paper"] += int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] or 0)
        if _table_exists(conn, "notes"):
            for source_type, count in conn.execute("SELECT COALESCE(source_type, ''), COUNT(*) FROM notes GROUP BY source_type"):
                normalized = _normalize_source_type(source_type)
                if normalized in {"vault", "web"}:
                    counts[normalized] += int(count or 0)
        if _table_exists(conn, "web_cards_v2"):
            counts["web"] = max(counts["web"], int(conn.execute("SELECT COUNT(*) FROM web_cards_v2").fetchone()[0] or 0))
        if _table_exists(conn, "vault_cards_v2"):
            counts["vault"] = max(counts["vault"], int(conn.execute("SELECT COUNT(*) FROM vault_cards_v2").fetchone()[0] or 0))
        return {source_type: int(counts.get(source_type, 0)) for source_type in SOURCE_TYPES}
    finally:
        conn.close()


def collect_vector_source_counts(vector_db_path: str | Path, *, limit: int = 20000) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in list_vector_documents(vector_db_path, limit=limit):
        source_type = _normalize_source_type(row.get("source_type"))
        if source_type in SOURCE_TYPES:
            counts[source_type] += 1
    return {source_type: int(counts.get(source_type, 0)) for source_type in SOURCE_TYPES}


def build_index_freshness_report(
    *,
    sqlite_source_counts: dict[str, int],
    vector_source_counts: dict[str, int],
) -> dict[str, Any]:
    missing_types = [
        source_type
        for source_type in SOURCE_TYPES
        if int(sqlite_source_counts.get(source_type, 0) or 0) > 0 and int(vector_source_counts.get(source_type, 0) or 0) <= 0
    ]
    stale_risk_types = [
        source_type
        for source_type in SOURCE_TYPES
        if int(sqlite_source_counts.get(source_type, 0) or 0) > 0
        and 0 < int(vector_source_counts.get(source_type, 0) or 0) < int(sqlite_source_counts.get(source_type, 0) or 0)
    ]
    if missing_types:
        status = "degraded"
        summary = "source records exist without matching vector coverage."
        fix = "khub index --all"
    elif stale_risk_types:
        status = "ok"
        summary = "source records and vector coverage both exist; partial-count differences are informational."
        fix = ""
    else:
        status = "ok"
        summary = "source/vector coverage is present for indexed source types."
        fix = ""
    return {
        "status": status,
        "summary": summary,
        "sqliteSourceCounts": {source_type: int(sqlite_source_counts.get(source_type, 0) or 0) for source_type in SOURCE_TYPES},
        "vectorSourceCounts": {source_type: int(vector_source_counts.get(source_type, 0) or 0) for source_type in SOURCE_TYPES},
        "missingVectorSourceTypes": missing_types,
        "partialVectorSourceTypes": stale_risk_types,
        "fixCommand": fix,
    }


def build_index_freshness_check(config: Any) -> dict[str, Any]:
    try:
        sqlite_counts = collect_sqlite_source_counts(getattr(config, "sqlite_path", ""))
        vector_counts = collect_vector_source_counts(getattr(config, "vector_db_path", ""))
        report = build_index_freshness_report(sqlite_source_counts=sqlite_counts, vector_source_counts=vector_counts)
    except Exception as exc:  # noqa: BLE001
        return {
            "area": "index freshness",
            "status": "degraded",
            "summary": "index freshness check could not inspect local stores.",
            "detail": str(exc),
            "fixCommand": "",
        }
    return {
        "area": "index freshness",
        "status": report["status"],
        "summary": report["summary"],
        "detail": (
            f"sqlite={report['sqliteSourceCounts']} vector={report['vectorSourceCounts']} "
            f"missing={report['missingVectorSourceTypes']} partial={report['partialVectorSourceTypes']}"
        ),
        "fixCommand": report["fixCommand"],
        "diagnostics": report,
    }


__all__ = [
    "build_index_freshness_check",
    "build_index_freshness_report",
    "collect_sqlite_source_counts",
    "collect_vector_source_counts",
]
