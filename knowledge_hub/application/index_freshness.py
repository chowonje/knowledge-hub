from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from knowledge_hub.infrastructure.persistence.vector import list_vector_documents


SOURCE_TYPES = ("paper", "vault", "web")
SOURCE_ID_SAMPLE_LIMIT = 20


def _normalize_source_type(value: Any) -> str:
    token = str(value or "").strip().lower()
    return "vault" if token == "note" else token


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _empty_source_id_sets() -> dict[str, set[str]]:
    return {source_type: set() for source_type in SOURCE_TYPES}


def _strip_scope_suffix(value: str) -> str:
    token = str(value or "").strip().replace("\\", "/")
    for marker in ("::section:", "::document:", "::chunk:", "::"):
        if marker in token:
            token = token.split(marker, 1)[0]
    if "#" in token and not token.lower().startswith(("http://", "https://")):
        token = token.split("#", 1)[0]
    return token.strip()


def _canonical_source_id(source_type: Any, value: Any) -> str:
    normalized_type = _normalize_source_type(source_type)
    token = _strip_scope_suffix(str(value or ""))
    if normalized_type not in SOURCE_TYPES or not token:
        return ""

    if normalized_type == "paper":
        if token.startswith("paper:"):
            token = token[len("paper:") :]
        match = re.match(r"^paper_(.+)_\d+$", token)
        if match:
            token = match.group(1)
        return f"paper:{token.strip()}" if token.strip() else ""

    if normalized_type == "vault":
        for prefix in ("vault:", "note:"):
            if token.startswith(prefix):
                token = token[len(prefix) :]
                break
        return f"vault:{token.strip()}" if token.strip() else ""

    if normalized_type == "web":
        for prefix in ("web:", "note:"):
            if token.startswith(prefix):
                token = token[len(prefix) :]
                break
        return f"web:{token.strip()}" if token.strip() else ""

    return ""


def _normalize_source_id_map(source_ids: dict[str, Any] | None) -> dict[str, set[str]]:
    normalized = _empty_source_id_sets()
    if not source_ids:
        return normalized
    for source_type, values in source_ids.items():
        normalized_type = _normalize_source_type(source_type)
        if normalized_type not in SOURCE_TYPES:
            continue
        if isinstance(values, (str, bytes)):
            iterable = [values]
        else:
            try:
                iterable = list(values or [])
            except TypeError:
                iterable = [values]
        for value in iterable:
            source_id = _canonical_source_id(normalized_type, value)
            if source_id:
                normalized[normalized_type].add(source_id)
    return normalized


def _sorted_source_id_map(source_ids: dict[str, set[str]]) -> dict[str, list[str]]:
    return {source_type: sorted(source_ids.get(source_type, set())) for source_type in SOURCE_TYPES}


def _sample_source_id_map(source_ids: dict[str, set[str]], *, limit: int = SOURCE_ID_SAMPLE_LIMIT) -> dict[str, list[str]]:
    sample_limit = max(1, int(limit))
    return {source_type: sorted(source_ids.get(source_type, set()))[:sample_limit] for source_type in SOURCE_TYPES}


def _json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    token = str(raw or "").strip()
    if not token:
        return {}
    try:
        parsed = json.loads(token)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _row_value(row: sqlite3.Row, key: str) -> str:
    try:
        return str(row[key] or "")
    except Exception:
        return ""


def _vector_row_source_id(*, source_type: str, metadata: dict[str, Any], doc_id: str) -> str:
    # Span-level freshness is not reliable from the current vector sidecar: paper,
    # vault, and web rows do not all expose stable expected span locators. Use the
    # best document/source-level identifier available instead.
    keys_by_type = {
        "paper": ("document_id", "source_id", "arxiv_id", "paper_id"),
        "vault": ("document_scope_id", "document_id", "source_id", "note_id", "file_path", "stable_scope_id"),
        "web": ("document_id", "source_id", "note_id", "canonical_url", "url", "source_url"),
    }
    for key in keys_by_type.get(source_type, ()):
        source_id = _canonical_source_id(source_type, metadata.get(key))
        if source_id:
            return source_id
    return _canonical_source_id(source_type, doc_id)


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


def collect_sqlite_source_ids(sqlite_path: str | Path, *, limit: int = 20000) -> dict[str, list[str]]:
    path = Path(sqlite_path).expanduser()
    if not path.exists():
        return {source_type: [] for source_type in SOURCE_TYPES}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1)
    conn.row_factory = sqlite3.Row
    try:
        ids = _empty_source_id_sets()
        row_limit = max(1, int(limit))

        if _table_exists(conn, "papers") and "arxiv_id" in _table_columns(conn, "papers"):
            for row in conn.execute(
                "SELECT arxiv_id FROM papers WHERE COALESCE(arxiv_id, '') != '' ORDER BY arxiv_id LIMIT ?",
                (row_limit,),
            ):
                source_id = _canonical_source_id("paper", row["arxiv_id"])
                if source_id:
                    ids["paper"].add(source_id)

        if _table_exists(conn, "notes"):
            columns = _table_columns(conn, "notes")
            if {"id", "source_type"}.issubset(columns):
                selected_columns = ["id", "source_type"]
                if "file_path" in columns:
                    selected_columns.append("file_path")
                query = (
                    f"SELECT {', '.join(selected_columns)} FROM notes "
                    "WHERE COALESCE(source_type, '') != '' ORDER BY id LIMIT ?"
                )
                for row in conn.execute(query, (row_limit,)):
                    source_type = _normalize_source_type(_row_value(row, "source_type"))
                    if source_type not in {"vault", "web"}:
                        continue
                    source_id = _canonical_source_id(source_type, _row_value(row, "id"))
                    if not source_id:
                        source_id = _canonical_source_id(source_type, _row_value(row, "file_path"))
                    if source_id:
                        ids[source_type].add(source_id)

        if _table_exists(conn, "web_cards_v2"):
            columns = _table_columns(conn, "web_cards_v2")
            if "document_id" in columns:
                for row in conn.execute(
                    "SELECT document_id FROM web_cards_v2 WHERE COALESCE(document_id, '') != '' ORDER BY document_id LIMIT ?",
                    (row_limit,),
                ):
                    source_id = _canonical_source_id("web", row["document_id"])
                    if source_id:
                        ids["web"].add(source_id)

        if _table_exists(conn, "vault_cards_v2"):
            columns = _table_columns(conn, "vault_cards_v2")
            if "note_id" in columns:
                for row in conn.execute(
                    "SELECT note_id FROM vault_cards_v2 WHERE COALESCE(note_id, '') != '' ORDER BY note_id LIMIT ?",
                    (row_limit,),
                ):
                    source_id = _canonical_source_id("vault", row["note_id"])
                    if source_id:
                        ids["vault"].add(source_id)

        return _sorted_source_id_map(ids)
    finally:
        conn.close()


def collect_vector_source_counts(vector_db_path: str | Path, *, limit: int = 20000) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in list_vector_documents(vector_db_path, limit=limit):
        source_type = _normalize_source_type(row.get("source_type"))
        if source_type in SOURCE_TYPES:
            counts[source_type] += 1
    return {source_type: int(counts.get(source_type, 0)) for source_type in SOURCE_TYPES}


def collect_vector_source_ids(vector_db_path: str | Path, *, limit: int = 20000) -> dict[str, list[str]]:
    root = Path(vector_db_path).expanduser()
    lexical_db_path = root / "_lexical.sqlite3"
    if not lexical_db_path.exists():
        return {source_type: [] for source_type in SOURCE_TYPES}
    conn = sqlite3.connect(f"file:{lexical_db_path}?mode=ro", uri=True, timeout=1)
    conn.row_factory = sqlite3.Row
    try:
        ids = _empty_source_id_sets()
        if not _table_exists(conn, "lexical_documents_fts"):
            return _sorted_source_id_map(ids)
        if _table_exists(conn, "lexical_documents_meta"):
            rows = conn.execute(
                """
                SELECT f.doc_id, m.metadata_json
                FROM lexical_documents_fts AS f
                LEFT JOIN lexical_documents_meta AS m ON m.doc_id = f.doc_id
                ORDER BY f.doc_id
                LIMIT ?
                """,
                (max(1, int(limit)),),
            )
        else:
            rows = conn.execute(
                """
                SELECT f.doc_id, '' AS metadata_json
                FROM lexical_documents_fts AS f
                ORDER BY f.doc_id
                LIMIT ?
                """,
                (max(1, int(limit)),),
            )
        for row in rows:
            metadata = _json_dict(row["metadata_json"])
            source_type = _normalize_source_type(metadata.get("source_type"))
            if source_type not in SOURCE_TYPES:
                continue
            source_id = _vector_row_source_id(source_type=source_type, metadata=metadata, doc_id=str(row["doc_id"] or ""))
            if source_id:
                ids[source_type].add(source_id)
        return _sorted_source_id_map(ids)
    finally:
        conn.close()


def _build_source_id_coverage(
    *,
    sqlite_source_ids: dict[str, Any] | None,
    vector_source_ids: dict[str, Any] | None,
    vector_source_counts: dict[str, int],
) -> dict[str, Any]:
    if sqlite_source_ids is None or vector_source_ids is None:
        return {
            "status": "unavailable",
            "scope": "source_id",
            "spanCoverageAvailable": False,
            "limitation": (
                "Exact span coverage is unavailable because current vector metadata does not expose a complete "
                "expected span inventory for paper, vault, and web sources."
            ),
            "sqliteSourceIdCounts": {source_type: 0 for source_type in SOURCE_TYPES},
            "vectorSourceIdCounts": {source_type: 0 for source_type in SOURCE_TYPES},
            "missingVectorSourceIds": {source_type: [] for source_type in SOURCE_TYPES},
            "missingVectorSourceIdCounts": {source_type: 0 for source_type in SOURCE_TYPES},
            "missingVectorSourceIdTypes": [],
            "sourceIdMetadataUnavailableTypes": [],
        }

    sqlite_ids = _normalize_source_id_map(sqlite_source_ids)
    vector_ids = _normalize_source_id_map(vector_source_ids)
    missing_ids = _empty_source_id_sets()
    metadata_unavailable_types: list[str] = []
    for source_type in SOURCE_TYPES:
        expected = sqlite_ids[source_type]
        observed = vector_ids[source_type]
        vector_count = int(vector_source_counts.get(source_type, 0) or 0)
        if expected and vector_count > 0 and not observed:
            metadata_unavailable_types.append(source_type)
            continue
        missing_ids[source_type] = set(expected - observed)

    missing_types = [source_type for source_type in SOURCE_TYPES if missing_ids[source_type]]
    coverage_status = "degraded" if missing_types else "unavailable" if metadata_unavailable_types else "ok"
    return {
        "status": coverage_status,
        "scope": "source_id",
        "spanCoverageAvailable": False,
        "limitation": (
            "Exact span coverage is unavailable because current vector metadata does not expose a complete "
            "expected span inventory for paper, vault, and web sources; this audit compares canonical source IDs."
        ),
        "sqliteSourceIdCounts": {source_type: len(sqlite_ids[source_type]) for source_type in SOURCE_TYPES},
        "vectorSourceIdCounts": {source_type: len(vector_ids[source_type]) for source_type in SOURCE_TYPES},
        "missingVectorSourceIds": _sample_source_id_map(missing_ids),
        "missingVectorSourceIdCounts": {source_type: len(missing_ids[source_type]) for source_type in SOURCE_TYPES},
        "missingVectorSourceIdTypes": missing_types,
        "sourceIdMetadataUnavailableTypes": metadata_unavailable_types,
    }


def build_index_freshness_report(
    *,
    sqlite_source_counts: dict[str, int],
    vector_source_counts: dict[str, int],
    sqlite_source_ids: dict[str, Any] | None = None,
    vector_source_ids: dict[str, Any] | None = None,
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
    source_id_coverage = _build_source_id_coverage(
        sqlite_source_ids=sqlite_source_ids,
        vector_source_ids=vector_source_ids,
        vector_source_counts=vector_source_counts,
    )
    missing_source_id_types = list(source_id_coverage.get("missingVectorSourceIdTypes") or [])
    if missing_types or missing_source_id_types:
        status = "degraded"
        if missing_types:
            summary = "source records exist without matching vector coverage."
        else:
            summary = "canonical source IDs exist without matching vector metadata coverage."
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
        "sourceIdCoverage": source_id_coverage,
        "fixCommand": fix,
    }


def build_index_freshness_check(config: Any) -> dict[str, Any]:
    try:
        sqlite_counts = collect_sqlite_source_counts(getattr(config, "sqlite_path", ""))
        vector_counts = collect_vector_source_counts(getattr(config, "vector_db_path", ""))
        sqlite_source_ids = collect_sqlite_source_ids(getattr(config, "sqlite_path", ""))
        vector_source_ids = collect_vector_source_ids(getattr(config, "vector_db_path", ""))
        report = build_index_freshness_report(
            sqlite_source_counts=sqlite_counts,
            vector_source_counts=vector_counts,
            sqlite_source_ids=sqlite_source_ids,
            vector_source_ids=vector_source_ids,
        )
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
            f"missing={report['missingVectorSourceTypes']} partial={report['partialVectorSourceTypes']} "
            f"sourceIdMissing={report['sourceIdCoverage']['missingVectorSourceIdCounts']}"
        ),
        "fixCommand": report["fixCommand"],
        "diagnostics": report,
    }


__all__ = [
    "build_index_freshness_check",
    "build_index_freshness_report",
    "collect_sqlite_source_counts",
    "collect_sqlite_source_ids",
    "collect_vector_source_counts",
    "collect_vector_source_ids",
]
