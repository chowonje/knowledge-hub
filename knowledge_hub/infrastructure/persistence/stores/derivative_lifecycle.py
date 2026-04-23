"""Shared lifecycle helpers for rebuildable semantic projections."""

from __future__ import annotations

import hashlib
from typing import Any, Callable


SOURCE_HASH_KEYS = (
    "source_content_hash",
    "content_hash",
    "contentHash",
    "content_sha1",
    "content_sha256",
    "source_hash",
    "sourceHash",
    "document_hash",
    "parse_hash",
    "chunk_hash",
)

DERIVATIVE_SOURCE_COLUMNS = (
    ("source_content_hash", "source_content_hash TEXT NOT NULL DEFAULT ''"),
    ("stale", "stale INTEGER NOT NULL DEFAULT 0"),
    ("stale_reason", "stale_reason TEXT NOT NULL DEFAULT ''"),
    ("invalidated_at", "invalidated_at TEXT NOT NULL DEFAULT ''"),
)


def clean_token(value: Any) -> str:
    return str(value or "").strip()


def table_exists(conn: Any, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (str(table or "").strip(),),
    ).fetchone()
    return row is not None


def column_names(conn: Any, table: str) -> set[str]:
    if not table_exists(conn, table):
        return set()
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def add_derivative_lifecycle_columns(conn: Any, table: str, add_column_fn: Callable[[Any, str, str, str], None]) -> None:
    for column_name, column_sql in DERIVATIVE_SOURCE_COLUMNS:
        add_column_fn(conn, table, column_name, column_sql)


def source_hash_from_payload(payload: dict[str, Any]) -> str:
    for key in SOURCE_HASH_KEYS:
        token = clean_token(payload.get(key))
        if token:
            return token
    for nested_key in ("provenance", "diagnostics", "metadata", "source_trace", "sourceTrace"):
        nested = payload.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in SOURCE_HASH_KEYS:
            token = clean_token(nested.get(key))
            if token:
                return token
    return ""


def fallback_source_hash(*parts: Any) -> str:
    text = "\n".join(clean_token(part) for part in parts if clean_token(part))
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_hash_from_content(*, content: Any = "", metadata: dict[str, Any] | None = None, identity: Any = "") -> str:
    """Resolve a canonical-ish source hash for raw note/web/vault writes."""

    payload = dict(metadata or {})
    explicit = source_hash_from_payload(payload)
    if explicit:
        return explicit
    return fallback_source_hash(content, identity)


def _document_id_variants(document_id: str, *, source_type: str = "") -> list[str]:
    token = clean_token(document_id)
    if not token:
        return []
    source = clean_token(source_type).lower()
    variants: list[str] = []

    def add(value: str) -> None:
        clean = clean_token(value)
        if clean and clean not in variants:
            variants.append(clean)

    add(token)
    if source and not token.startswith(f"{source}:"):
        add(f"{source}:{token}")
    if ":" in token:
        add(token.split(":", 1)[1])
    return variants


def document_source_hash(document_id: str, units: list[dict[str, Any]]) -> str:
    for payload in units:
        token = source_hash_from_payload(dict(payload or {}))
        if token:
            return token
    digest_parts: list[str] = [clean_token(document_id)]
    for payload in units:
        item = dict(payload or {})
        digest_parts.extend(
            [
                item.get("unit_id"),
                item.get("document_title"),
                item.get("title"),
                item.get("section_path"),
                item.get("contextual_summary"),
                item.get("source_excerpt"),
                item.get("search_text"),
            ]
        )
    return fallback_source_hash(*digest_parts)


def document_memory_source_hash(conn: Any, document_id: str) -> str:
    if not document_id or "document_memory_units" not in _tables_with_column(conn, "source_content_hash"):
        return ""
    row = conn.execute(
        """
        SELECT source_content_hash FROM document_memory_units
        WHERE document_id = ? AND source_content_hash != ''
        ORDER BY
            CASE WHEN unit_type = 'document_summary' THEN 0 ELSE 1 END ASC,
            updated_at DESC
        LIMIT 1
        """,
        (str(document_id or "").strip(),),
    ).fetchone()
    return clean_token(row["source_content_hash"] if row else "")


def resolve_source_content_hash(conn: Any, payload: dict[str, Any], *, document_id: str = "") -> str:
    explicit = source_hash_from_payload(payload)
    if explicit:
        return explicit
    return document_memory_source_hash(conn, document_id)


def fresh_stale_value(payload: dict[str, Any]) -> int:
    return 1 if bool(payload.get("stale")) else 0


def stale_reason_value(payload: dict[str, Any]) -> str:
    return clean_token(payload.get("stale_reason") or payload.get("staleReason"))


def invalidated_at_value(payload: dict[str, Any]) -> str:
    return clean_token(payload.get("invalidated_at") or payload.get("invalidatedAt"))


def _tables_with_column(conn: Any, column_name: str) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    matched: set[str] = set()
    for row in rows:
        table = str(row["name"] if hasattr(row, "keys") and "name" in row.keys() else row[0])
        if column_name in column_names(conn, table):
            matched.add(table)
    return matched


def mark_derivatives_stale_for_document(
    conn: Any,
    *,
    document_id: str,
    source_content_hash: str,
    source_type: str = "",
) -> int:
    doc_id = clean_token(document_id)
    new_hash = clean_token(source_content_hash)
    if not doc_id or not new_hash:
        return 0

    paper_id = doc_id.removeprefix("paper:") if doc_id.startswith("paper:") else ""
    variants = _document_id_variants(doc_id, source_type=source_type)

    def _in_clause(column: str, values: list[str]) -> tuple[str, tuple[Any, ...]]:
        placeholders = ", ".join("?" for _ in values)
        return f"{column} IN ({placeholders})", tuple(values)

    doc_clause, doc_params = _in_clause("document_id", variants)
    note_clause, note_params = _in_clause("note_id", variants)
    claim_clause = " OR ".join(
        [
            _in_clause("document_id", variants)[0],
            _in_clause("source_id", variants)[0],
        ]
    )
    claim_params = tuple([*variants, *variants])
    targets: list[tuple[str, str, tuple[Any, ...]]] = [
        ("document_memory_units", doc_clause, doc_params),
        ("web_cards_v2", doc_clause, doc_params),
        ("vault_cards_v2", note_clause, note_params),
        ("claim_cards_v1", f"({claim_clause})", claim_params),
    ]
    if paper_id:
        targets.extend(
            [
                ("paper_memory_cards", "paper_id = ?", (paper_id,)),
                ("paper_cards_v2", "paper_id = ?", (paper_id,)),
                ("paper_section_cards_v1", "paper_id = ?", (paper_id,)),
                (
                    "claim_cards_v1",
                    "(paper_id = ? OR document_id = ? OR source_id = ?)",
                    (paper_id, doc_id, paper_id),
                ),
            ]
        )

    changed = 0
    for table, identity_clause, identity_params in targets:
        columns = column_names(conn, table)
        required = {"source_content_hash", "stale", "stale_reason", "invalidated_at"}
        if not required.issubset(columns):
            continue
        cursor = conn.execute(
            f"""
            UPDATE {table}
            SET stale = 1,
                stale_reason = 'source_content_hash_changed',
                invalidated_at = CURRENT_TIMESTAMP
            WHERE {identity_clause}
              AND (source_content_hash = '' OR source_content_hash != ?)
              AND COALESCE(stale, 0) = 0
            """,
            tuple([*identity_params, new_hash]),
        )
        changed += int(cursor.rowcount or 0)
    return changed
