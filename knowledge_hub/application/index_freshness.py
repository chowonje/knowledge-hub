from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from knowledge_hub.core.prepared_source_record import PREPARED_SOURCE_RECORD_SCHEMA

SOURCE_TYPES = ("paper", "vault", "web")
SOURCE_ID_SAMPLE_LIMIT = 20
SOURCE_ID_SAMPLE_MODE = "redacted_sha256"
PAPER_REPAIR_COMMAND = "khub index --all"
VAULT_REPAIR_COMMAND = "khub index --vault-all"
WEB_REPAIR_COMMAND = "khub crawl reindex-approved --include-unrated --json"
WEB_PREPARED_REPAIR_COMMAND = "khub crawl reindex-approved --include-unrated --prepared-metadata-only --json"
PREPARED_RECORD_REPAIRABLE_TYPES = {"web"}


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


def _redacted_source_id(source_type: str, source_id: str) -> str:
    digest = hashlib.sha256(str(source_id or "").encode("utf-8")).hexdigest()[:12]
    return f"{source_type}-redacted-{digest}"


def _sample_source_id_map(source_ids: dict[str, set[str]], *, limit: int = SOURCE_ID_SAMPLE_LIMIT) -> dict[str, list[str]]:
    sample_limit = max(1, int(limit))
    return {
        source_type: [
            _redacted_source_id(source_type, source_id)
            for source_id in sorted(source_ids.get(source_type, set()))[:sample_limit]
        ]
        for source_type in SOURCE_TYPES
    }


def _source_id_coverage_ratios(
    sqlite_ids: dict[str, set[str]],
    missing_ids: dict[str, set[str]],
    *,
    metadata_unavailable_types: list[str],
) -> dict[str, float | None]:
    unavailable = set(metadata_unavailable_types)
    ratios: dict[str, float | None] = {}
    for source_type in SOURCE_TYPES:
        expected_count = len(sqlite_ids.get(source_type, set()))
        if expected_count <= 0:
            ratios[source_type] = 1.0
        elif source_type in unavailable:
            ratios[source_type] = None
        else:
            observed_count = max(0, expected_count - len(missing_ids.get(source_type, set())))
            ratios[source_type] = round(observed_count / expected_count, 6)
    return ratios


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


def _lexical_source_type_sql(alias: str = "m") -> str:
    return (
        f"LOWER(TRIM(CASE WHEN json_valid({alias}.metadata_json) "
        f"THEN json_extract({alias}.metadata_json, '$.source_type') ELSE '' END))"
    )


def _source_type_aliases(source_type: str) -> tuple[str, ...]:
    if source_type == "vault":
        return ("vault", "note")
    return (source_type,)


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
    root = Path(vector_db_path).expanduser()
    lexical_db_path = root / "_lexical.sqlite3"
    if not lexical_db_path.exists():
        return {source_type: 0 for source_type in SOURCE_TYPES}
    conn = sqlite3.connect(f"file:{lexical_db_path}?mode=ro", uri=True, timeout=1)
    conn.row_factory = sqlite3.Row
    try:
        if not _table_exists(conn, "lexical_documents_fts") or not _table_exists(conn, "lexical_documents_meta"):
            return {source_type: 0 for source_type in SOURCE_TYPES}

        source_type_expr = _lexical_source_type_sql("m")
        rows = conn.execute(
            f"""
            SELECT {source_type_expr} AS source_type, COUNT(*) AS count
            FROM lexical_documents_fts AS f
            LEFT JOIN lexical_documents_meta AS m ON m.doc_id = f.doc_id
            GROUP BY source_type
            """
        ).fetchall()
        counts: Counter[str] = Counter()
        for row in rows:
            source_type = _normalize_source_type(row["source_type"])
            if source_type in SOURCE_TYPES:
                counts[source_type] += int(row["count"] or 0)
        return {source_type: int(counts.get(source_type, 0)) for source_type in SOURCE_TYPES}
    finally:
        conn.close()


def _collect_vector_source_rows_by_type(
    vector_db_path: str | Path,
    *,
    source_type: str,
    limit: int | None,
) -> list[sqlite3.Row]:
    root = Path(vector_db_path).expanduser()
    lexical_db_path = root / "_lexical.sqlite3"
    if not lexical_db_path.exists():
        return []
    conn = sqlite3.connect(f"file:{lexical_db_path}?mode=ro", uri=True, timeout=1)
    conn.row_factory = sqlite3.Row
    try:
        if not _table_exists(conn, "lexical_documents_fts") or not _table_exists(conn, "lexical_documents_meta"):
            return []
        source_type_expr = _lexical_source_type_sql("m")
        aliases = _source_type_aliases(source_type)
        placeholders = ", ".join("?" for _ in aliases)
        limit_clause = ""
        params: tuple[Any, ...] = aliases
        if limit is not None:
            limit_clause = "LIMIT ?"
            params = (*aliases, max(1, int(limit)))
        return list(
            conn.execute(
                f"""
                SELECT f.doc_id, m.metadata_json
                FROM lexical_documents_fts AS f
                LEFT JOIN lexical_documents_meta AS m ON m.doc_id = f.doc_id
                WHERE {source_type_expr} IN ({placeholders})
                ORDER BY f.doc_id
                {limit_clause}
                """,
                params,
            ).fetchall()
        )
    finally:
        conn.close()


def collect_vector_source_ids(vector_db_path: str | Path, *, limit: int | None = None) -> dict[str, list[str]]:
    ids = _empty_source_id_sets()
    for source_type in SOURCE_TYPES:
        for row in _collect_vector_source_rows_by_type(vector_db_path, source_type=source_type, limit=limit):
            metadata = _json_dict(row["metadata_json"])
            source_id = _vector_row_source_id(source_type=source_type, metadata=metadata, doc_id=str(row["doc_id"] or ""))
            if source_id:
                ids[source_type].add(source_id)
    return _sorted_source_id_map(ids)


def _prepared_record_path_is_indexable(
    path_value: Any,
    *,
    vector_root: Path,
    cache: dict[str, bool],
) -> bool:
    token = str(path_value or "").strip()
    if not token:
        return False
    if token in cache:
        return cache[token]
    raw_path = Path(token).expanduser()
    candidates = [raw_path] if raw_path.is_absolute() else [vector_root.parent / raw_path, raw_path]
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
        lifecycle = payload.get("lifecycle") if isinstance(payload.get("lifecycle"), dict) else {}
        ok = (
            str(payload.get("schema") or "").strip() == PREPARED_SOURCE_RECORD_SCHEMA
            and quality.get("passed") is True
            and lifecycle.get("stale") is False
        )
        cache[token] = ok
        return ok
    cache[token] = False
    return False


def collect_vector_prepared_counts(vector_db_path: str | Path, *, limit: int | None = None) -> dict[str, dict[str, int]] | None:
    root = Path(vector_db_path).expanduser()
    lexical_db_path = root / "_lexical.sqlite3"
    if not lexical_db_path.exists():
        return None
    conn = sqlite3.connect(f"file:{lexical_db_path}?mode=ro", uri=True, timeout=1)
    conn.row_factory = sqlite3.Row
    try:
        if not _table_exists(conn, "lexical_documents_fts"):
            return None
        if not _table_exists(conn, "lexical_documents_meta"):
            return {source_type: {"vectorRows": 0, "preparedRows": 0} for source_type in SOURCE_TYPES}
        counts = {source_type: {"vectorRows": 0, "preparedRows": 0} for source_type in SOURCE_TYPES}
        source_type_expr = _lexical_source_type_sql("m")
        path_cache: dict[str, bool] = {}
        rows = conn.execute(
            f"""
            SELECT {source_type_expr} AS source_type,
                   m.metadata_json AS metadata_json
            FROM lexical_documents_fts AS f
            LEFT JOIN lexical_documents_meta AS m ON m.doc_id = f.doc_id
            ORDER BY f.doc_id
            """
        )
        for row in rows:
            source_type = _normalize_source_type(row["source_type"])
            if source_type not in SOURCE_TYPES:
                continue
            counts[source_type]["vectorRows"] += 1
            metadata = _json_dict(row["metadata_json"])
            prepared_path = (
                metadata.get("prepared_record_path")
                or metadata.get("prepared_source_record_path")
                or metadata.get("preparedRecordPath")
            )
            if _prepared_record_path_is_indexable(prepared_path, vector_root=root, cache=path_cache):
                counts[source_type]["preparedRows"] += 1
        return counts
    finally:
        conn.close()


def _prepared_record_coverage(
    prepared_counts: dict[str, dict[str, int]] | None,
    vector_source_counts: dict[str, int],
    sqlite_source_counts: dict[str, int],
) -> dict[str, Any]:
    expected_types = {
        source_type
        for source_type in SOURCE_TYPES
        if source_type in PREPARED_RECORD_REPAIRABLE_TYPES
        and int(sqlite_source_counts.get(source_type, 0) or 0) > 0
    }
    if prepared_counts is None:
        counts = {source_type: {"vectorRows": int(vector_source_counts.get(source_type, 0) or 0), "preparedRows": 0} for source_type in SOURCE_TYPES}
        unavailable_types = [
            source_type
            for source_type in SOURCE_TYPES
            if source_type in expected_types and int(vector_source_counts.get(source_type, 0) or 0) > 0
        ]
        return {
            "status": "unavailable" if unavailable_types else "ok",
            "scope": "vector_row",
            "counts": counts,
            "preparedRowCoverageRatios": {source_type: None for source_type in SOURCE_TYPES},
            "missingPreparedRecordTypes": [],
            "metadataUnavailableTypes": unavailable_types,
            "preparedRepairCommands": {},
        }
    counts = prepared_counts
    ratios: dict[str, float | None] = {}
    missing_types: list[str] = []
    for source_type in SOURCE_TYPES:
        vector_rows = int((counts.get(source_type) or {}).get("vectorRows", 0) or 0)
        prepared_rows = int((counts.get(source_type) or {}).get("preparedRows", 0) or 0)
        if vector_rows <= 0:
            ratios[source_type] = 1.0
        elif source_type not in expected_types:
            ratios[source_type] = None
        else:
            ratios[source_type] = round(prepared_rows / vector_rows, 6)
            if prepared_rows < vector_rows:
                missing_types.append(source_type)
    return {
        "status": "degraded" if missing_types else "ok",
        "scope": "vector_row",
        "counts": counts,
        "preparedRowCoverageRatios": ratios,
        "missingPreparedRecordTypes": missing_types,
        "metadataUnavailableTypes": [],
        "preparedRepairCommands": _prepared_repair_commands(missing_types),
    }


def _prepared_repair_commands(source_types: list[str]) -> dict[str, str]:
    requested = {str(source_type or "").strip().lower() for source_type in source_types}
    commands: dict[str, str] = {}
    if "web" in requested:
        commands["web"] = WEB_PREPARED_REPAIR_COMMAND
    return commands


def _fix_commands_for_source_types(source_types: list[str]) -> list[str]:
    requested = {str(source_type or "").strip().lower() for source_type in source_types}
    commands: list[str] = []
    needs_paper = "paper" in requested
    needs_vault = "vault" in requested
    if needs_paper and needs_vault:
        commands.append(f"{PAPER_REPAIR_COMMAND} --vault-all")
    elif needs_paper:
        commands.append(PAPER_REPAIR_COMMAND)
    elif needs_vault:
        commands.append(VAULT_REPAIR_COMMAND)
    if "web" in requested:
        commands.append(WEB_REPAIR_COMMAND)
    return commands


def _source_repair_commands(source_types: list[str]) -> dict[str, str]:
    requested = {str(source_type or "").strip().lower() for source_type in source_types}
    commands: dict[str, str] = {}
    if "paper" in requested:
        commands["paper"] = PAPER_REPAIR_COMMAND
    if "vault" in requested:
        commands["vault"] = VAULT_REPAIR_COMMAND
    if "web" in requested:
        commands["web"] = WEB_REPAIR_COMMAND
    return commands


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
            "sourceIdCoverageRatios": {source_type: None for source_type in SOURCE_TYPES},
            "missingVectorSourceIdSampleMode": SOURCE_ID_SAMPLE_MODE,
            "missingVectorSourceIdSampleLimit": SOURCE_ID_SAMPLE_LIMIT,
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
        "sourceIdCoverageRatios": _source_id_coverage_ratios(
            sqlite_ids,
            missing_ids,
            metadata_unavailable_types=metadata_unavailable_types,
        ),
        "missingVectorSourceIdSampleMode": SOURCE_ID_SAMPLE_MODE,
        "missingVectorSourceIdSampleLimit": SOURCE_ID_SAMPLE_LIMIT,
    }


def build_index_freshness_report(
    *,
    sqlite_source_counts: dict[str, int],
    vector_source_counts: dict[str, int],
    sqlite_source_ids: dict[str, Any] | None = None,
    vector_source_ids: dict[str, Any] | None = None,
    vector_prepared_counts: dict[str, dict[str, int]] | None = None,
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
    prepared_coverage = _prepared_record_coverage(vector_prepared_counts, vector_source_counts, sqlite_source_counts)
    missing_source_id_types = list(source_id_coverage.get("missingVectorSourceIdTypes") or [])
    missing_prepared_types = list(prepared_coverage.get("missingPreparedRecordTypes") or [])
    prepared_repair_commands = dict(prepared_coverage.get("preparedRepairCommands") or {})
    repair_source_types = list(dict.fromkeys([*missing_types, *missing_source_id_types]))
    fix_commands = _fix_commands_for_source_types(repair_source_types)
    source_repair_commands = _source_repair_commands(repair_source_types)
    prepared_fix_commands = [
        command
        for command in prepared_repair_commands.values()
        if command and command not in fix_commands
    ]
    all_fix_commands = [*fix_commands, *prepared_fix_commands]
    if missing_types or missing_source_id_types:
        status = "degraded"
        if missing_types:
            summary = "source records exist without matching vector coverage."
        else:
            summary = "canonical source IDs exist without matching vector metadata coverage."
        fix = " && ".join(all_fix_commands) if all_fix_commands else PAPER_REPAIR_COMMAND
    elif missing_prepared_types:
        status = "degraded"
        summary = "vector rows are missing prepared-source metadata."
        fix = " && ".join(all_fix_commands) if all_fix_commands else WEB_PREPARED_REPAIR_COMMAND
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
        "preparedRecordCoverage": prepared_coverage,
        "sourceRepairCommands": (
            {**source_repair_commands, **{key: value for key, value in prepared_repair_commands.items() if key not in source_repair_commands}}
            if status == "degraded"
            else {}
        ),
        "fixCommand": fix,
        "fixCommands": all_fix_commands if status == "degraded" else [],
    }


def build_index_freshness_check(config: Any) -> dict[str, Any]:
    try:
        sqlite_counts = collect_sqlite_source_counts(getattr(config, "sqlite_path", ""))
        vector_counts = collect_vector_source_counts(getattr(config, "vector_db_path", ""))
        sqlite_source_ids = collect_sqlite_source_ids(getattr(config, "sqlite_path", ""))
        vector_source_ids = collect_vector_source_ids(getattr(config, "vector_db_path", ""))
        vector_prepared_counts = collect_vector_prepared_counts(getattr(config, "vector_db_path", ""))
        report = build_index_freshness_report(
            sqlite_source_counts=sqlite_counts,
            vector_source_counts=vector_counts,
            sqlite_source_ids=sqlite_source_ids,
            vector_source_ids=vector_source_ids,
            vector_prepared_counts=vector_prepared_counts,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "area": "index freshness",
            "status": "degraded",
            "summary": "index freshness check could not inspect local stores.",
            "detail": str(exc),
            "fixCommand": "",
            "fixCommands": [],
        }
    return {
        "area": "index freshness",
        "status": report["status"],
        "summary": report["summary"],
        "detail": (
            f"sqlite={report['sqliteSourceCounts']} vector={report['vectorSourceCounts']} "
            f"missing={report['missingVectorSourceTypes']} partial={report['partialVectorSourceTypes']} "
            f"sourceIdMissing={report['sourceIdCoverage']['missingVectorSourceIdCounts']} "
            f"preparedMissing={report['preparedRecordCoverage']['missingPreparedRecordTypes']}"
        ),
        "fixCommand": report["fixCommand"],
        "fixCommands": report.get("fixCommands", []),
        "diagnostics": report,
    }


__all__ = [
    "build_index_freshness_check",
    "build_index_freshness_report",
    "collect_sqlite_source_counts",
    "collect_sqlite_source_ids",
    "collect_vector_source_counts",
    "collect_vector_source_ids",
    "collect_vector_prepared_counts",
]
