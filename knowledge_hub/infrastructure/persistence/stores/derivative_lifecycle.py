"""Shared lifecycle helpers for rebuildable semantic projections."""

from __future__ import annotations

import hashlib
import json
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


def normalize_token_list(values: Any) -> list[str]:
    parsed: list[Any]
    if isinstance(values, list):
        parsed = values
    elif isinstance(values, tuple):
        parsed = list(values)
    elif not values:
        parsed = []
    else:
        try:
            loaded = json.loads(values)
        except Exception:
            loaded = []
        parsed = loaded if isinstance(loaded, list) else []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in parsed:
        token = clean_token(raw)
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def merge_token_json(existing: Any, new_values: list[str] | tuple[str, ...] | None) -> str:
    merged = normalize_token_list(existing)
    seen = set(merged)
    for raw in list(new_values or []):
        token = clean_token(raw)
        if not token or token in seen:
            continue
        seen.add(token)
        merged.append(token)
    return json.dumps(merged, ensure_ascii=False)


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
    seen: set[int] = set()
    stack: list[Any] = [dict(payload or {})]
    while stack:
        current = stack.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        if isinstance(current, dict):
            for key in SOURCE_HASH_KEYS:
                token = clean_token(current.get(key))
                if token:
                    return token
            for value in current.values():
                if isinstance(value, (dict, list, tuple)):
                    stack.append(value)
        elif isinstance(current, (list, tuple)):
            for value in current:
                if isinstance(value, (dict, list, tuple)):
                    stack.append(value)
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


def _safe_json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _document_identity_tokens(conn: Any, document_id: str, *, source_type: str = "") -> list[str]:
    variants = _document_id_variants(document_id, source_type=source_type)
    tokens: list[str] = []

    def add(value: Any) -> None:
        token = clean_token(value)
        if token and token not in tokens:
            tokens.append(token)

    for variant in variants:
        add(variant)
    if table_exists(conn, "notes"):
        for variant in variants:
            row = conn.execute(
                "SELECT id, file_path, metadata FROM notes WHERE id = ? OR file_path = ? LIMIT 1",
                (variant, variant),
            ).fetchone()
            if not row:
                continue
            add(row["id"] if hasattr(row, "keys") and "id" in row.keys() else row[0])
            add(row["file_path"] if hasattr(row, "keys") and "file_path" in row.keys() else row[1])
            metadata_raw = row["metadata"] if hasattr(row, "keys") and "metadata" in row.keys() else row[2]
            metadata = _safe_json_dict(metadata_raw)
            for key in ("canonical_url", "url", "source_url", "record_id", "source_item_id", "source_content_hash"):
                add(metadata.get(key))
    if source_type == "paper":
        paper_id = clean_token(document_id.removeprefix("paper:") if str(document_id).startswith("paper:") else document_id)
        add(paper_id)
        add(f"paper:{paper_id}" if paper_id else "")
    return tokens


def _like_any_clause(column: str, values: list[str]) -> tuple[str, tuple[Any, ...]]:
    normalized = [clean_token(value) for value in values if clean_token(value)]
    if not normalized:
        return "", ()
    return (
        "(" + " OR ".join(f"{column} LIKE ?" for _ in normalized) + ")",
        tuple(f"%{value}%" for value in normalized),
    )


def _mark_table_rows_stale(
    conn: Any,
    *,
    table: str,
    identity_clause: str,
    identity_params: tuple[Any, ...],
    source_content_hash: str,
    allow_manual_rows: bool = False,
) -> int:
    columns = column_names(conn, table)
    required = {"source_content_hash", "stale", "stale_reason", "invalidated_at"}
    if not required.issubset(columns):
        return 0
    where_parts = [f"({identity_clause})", "(source_content_hash = '' OR source_content_hash != ?)", "COALESCE(stale, 0) = 0"]
    params: list[Any] = [*identity_params, source_content_hash]
    if "origin" in columns and not allow_manual_rows:
        where_parts.append("COALESCE(origin, 'derived') != 'manual'")
    cursor = conn.execute(
        f"""
        UPDATE {table}
        SET stale = 1,
            stale_reason = 'source_content_hash_changed',
            invalidated_at = CURRENT_TIMESTAMP
        WHERE {' AND '.join(where_parts)}
        """,
        tuple(params),
    )
    return int(cursor.rowcount or 0)


def _collect_affected_source_hashes(
    conn: Any,
    *,
    table: str,
    identity_clause: str,
    identity_params: tuple[Any, ...],
    source_content_hash: str,
    allow_manual_rows: bool = False,
) -> set[str]:
    columns = column_names(conn, table)
    required = {"source_content_hash", "stale"}
    if not required.issubset(columns):
        return set()
    where_parts = [f"({identity_clause})", "source_content_hash != ''", "source_content_hash != ?", "COALESCE(stale, 0) = 0"]
    params: list[Any] = [*identity_params, source_content_hash]
    if "origin" in columns and not allow_manual_rows:
        where_parts.append("COALESCE(origin, 'derived') != 'manual'")
    rows = conn.execute(
        f"""
        SELECT DISTINCT source_content_hash
        FROM {table}
        WHERE {' AND '.join(where_parts)}
        """,
        tuple(params),
    ).fetchall()
    return {clean_token(row["source_content_hash"] if hasattr(row, "keys") else row[0]) for row in rows if clean_token(row["source_content_hash"] if hasattr(row, "keys") else row[0])}


def _has_live_contributor_support(
    conn: Any,
    contributor_hashes: list[str],
    *,
    supporting_tables: tuple[str, ...] = ("ontology_claims", "ontology_relations", "kg_relations"),
) -> bool:
    hashes = [clean_token(value) for value in contributor_hashes if clean_token(value)]
    if not hashes:
        return False
    placeholders = ", ".join("?" for _ in hashes)
    for table in supporting_tables:
        columns = column_names(conn, table)
        if not {"source_content_hash", "stale"}.issubset(columns):
            continue
        row = conn.execute(
            f"""
            SELECT 1
            FROM {table}
            WHERE source_content_hash IN ({placeholders})
              AND source_content_hash != ''
              AND COALESCE(stale, 0) = 0
            LIMIT 1
            """,
            tuple(hashes),
        ).fetchone()
        if row:
            return True
    return False


def _mark_contributor_gated_rows_stale(
    conn: Any,
    *,
    table: str,
    id_column: str,
    affected_source_hashes: set[str],
    entity_type: str | None = None,
) -> int:
    columns = column_names(conn, table)
    required = {"contributor_hashes", "stale", "stale_reason", "invalidated_at"}
    if not required.issubset(columns):
        return 0
    query = f"SELECT {id_column}, contributor_hashes FROM {table} WHERE COALESCE(stale, 0) = 0"
    params: list[Any] = []
    if entity_type and "entity_type" in columns:
        query += " AND entity_type = ?"
        params.append(clean_token(entity_type))
    rows = conn.execute(query, tuple(params)).fetchall()
    changed = 0
    for row in rows:
        contributor_hashes = normalize_token_list(row["contributor_hashes"] if hasattr(row, "keys") else row[1])
        if not contributor_hashes:
            continue
        if not set(contributor_hashes) & affected_source_hashes:
            continue
        if _has_live_contributor_support(conn, contributor_hashes):
            continue
        row_id = clean_token(row[id_column] if hasattr(row, "keys") else row[0])
        if not row_id:
            continue
        cursor = conn.execute(
            f"""
            UPDATE {table}
            SET stale = 1,
                stale_reason = 'contributors_stale',
                invalidated_at = CURRENT_TIMESTAMP
            WHERE {id_column} = ?
              AND COALESCE(stale, 0) = 0
            """,
            (row_id,),
        )
        changed += int(cursor.rowcount or 0)
    return changed


def mark_derivatives_stale_for_document(
    conn: Any,
    *,
    document_id: str,
    source_content_hash: str,
    source_type: str = "",
) -> int:
    doc_id = clean_token(document_id)
    new_hash = clean_token(source_content_hash)
    source_kind = clean_token(source_type).lower()
    if not doc_id or not new_hash:
        return 0

    paper_id = doc_id.removeprefix("paper:") if doc_id.startswith("paper:") else ""
    variants = _document_id_variants(doc_id, source_type=source_type)
    tokens = _document_identity_tokens(conn, doc_id, source_type=source_type)

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
        changed += _mark_table_rows_stale(
            conn,
            table=table,
            identity_clause=identity_clause,
            identity_params=identity_params,
            source_content_hash=new_hash,
        )
    invalidate_mixed_derivatives = source_kind in {"note", "paper", "web", "vault"}
    affected_ontology_hashes: set[str] = set()
    if tokens and invalidate_mixed_derivatives:
        claim_like_clause, claim_like_params = _like_any_clause("evidence_ptrs_json", tokens)
        if claim_like_clause:
            affected_ontology_hashes.update(
                _collect_affected_source_hashes(
                    conn,
                    table="ontology_claims",
                    identity_clause=claim_like_clause,
                    identity_params=claim_like_params,
                    source_content_hash=new_hash,
                )
            )
            changed += _mark_table_rows_stale(
                conn,
                table="ontology_claims",
                identity_clause=claim_like_clause,
                identity_params=claim_like_params,
                source_content_hash=new_hash,
            )

        relation_clauses: list[str] = []
        relation_params: list[Any] = []
        if variants:
            placeholders = ", ".join("?" for _ in variants)
            relation_clauses.append(f"source_id IN ({placeholders})")
            relation_params.extend(variants)
            relation_clauses.append(f"target_id IN ({placeholders})")
            relation_params.extend(variants)
        reason_like_clause, reason_like_params = _like_any_clause("reason_json", tokens)
        if reason_like_clause:
            relation_clauses.append(reason_like_clause)
            relation_params.extend(reason_like_params)
        if relation_clauses:
            affected_ontology_hashes.update(
                _collect_affected_source_hashes(
                    conn,
                    table="ontology_relations",
                    identity_clause=" OR ".join(f"({clause})" for clause in relation_clauses),
                    identity_params=tuple(relation_params),
                    source_content_hash=new_hash,
                )
            )
            changed += _mark_table_rows_stale(
                conn,
                table="ontology_relations",
                identity_clause=" OR ".join(f"({clause})" for clause in relation_clauses),
                identity_params=tuple(relation_params),
                source_content_hash=new_hash,
            )

        legacy_clauses: list[str] = []
        legacy_params: list[Any] = []
        if variants:
            placeholders = ", ".join("?" for _ in variants)
            legacy_clauses.append(f"source_id IN ({placeholders})")
            legacy_params.extend(variants)
            legacy_clauses.append(f"target_id IN ({placeholders})")
            legacy_params.extend(variants)
        evidence_like_clause, evidence_like_params = _like_any_clause("evidence_text", tokens)
        if evidence_like_clause:
            legacy_clauses.append(evidence_like_clause)
            legacy_params.extend(evidence_like_params)
        if legacy_clauses:
            affected_ontology_hashes.update(
                _collect_affected_source_hashes(
                    conn,
                    table="kg_relations",
                    identity_clause=" OR ".join(f"({clause})" for clause in legacy_clauses),
                    identity_params=tuple(legacy_params),
                    source_content_hash=new_hash,
                )
            )
            changed += _mark_table_rows_stale(
                conn,
                table="kg_relations",
                identity_clause=" OR ".join(f"({clause})" for clause in legacy_clauses),
                identity_params=tuple(legacy_params),
                source_content_hash=new_hash,
            )

        memory_clauses: list[str] = []
        memory_params: list[Any] = []
        if variants:
            placeholders = ", ".join("?" for _ in variants)
            memory_clauses.append(f"(src_form = 'document_memory' AND src_id IN ({placeholders}))")
            memory_params.extend(variants)
            memory_clauses.append(f"(dst_form = 'document_memory' AND dst_id IN ({placeholders}))")
            memory_params.extend(variants)
        if paper_id:
            memory_clauses.extend(
                [
                    "(src_form = 'paper_memory' AND src_id IN (SELECT memory_id FROM paper_memory_cards WHERE paper_id = ?))",
                    "(dst_form = 'paper_memory' AND dst_id IN (SELECT memory_id FROM paper_memory_cards WHERE paper_id = ?))",
                ]
            )
            memory_params.extend([paper_id, paper_id])
        provenance_like_clause, provenance_like_params = _like_any_clause("provenance_json", tokens)
        if provenance_like_clause:
            memory_clauses.append(provenance_like_clause)
            memory_params.extend(provenance_like_params)
        if memory_clauses:
            changed += _mark_table_rows_stale(
                conn,
                table="memory_relations",
                identity_clause=" OR ".join(f"({clause})" for clause in memory_clauses),
                identity_params=tuple(memory_params),
                source_content_hash=new_hash,
            )

        learning_targets = (
            ("learning_graph_nodes", ("provenance_json",)),
            ("learning_graph_edges", ("provenance_json", "evidence_json")),
            ("learning_graph_paths", ("provenance_json", "path_json")),
            ("learning_graph_resource_links", ("provenance_json",)),
        )
        for table, text_columns in learning_targets:
            clauses: list[str] = []
            params: list[Any] = []
            for column in text_columns:
                clause, clause_params = _like_any_clause(column, tokens)
                if clause:
                    clauses.append(clause)
                    params.extend(clause_params)
            if clauses:
                changed += _mark_table_rows_stale(
                    conn,
                    table=table,
                    identity_clause=" OR ".join(f"({clause})" for clause in clauses),
                    identity_params=tuple(params),
                    source_content_hash=new_hash,
                )
        if affected_ontology_hashes:
            changed += _mark_contributor_gated_rows_stale(
                conn,
                table="ontology_entities",
                id_column="entity_id",
                affected_source_hashes=affected_ontology_hashes,
                entity_type="concept",
            )
            changed += _mark_contributor_gated_rows_stale(
                conn,
                table="concepts",
                id_column="id",
                affected_source_hashes=affected_ontology_hashes,
            )
    return changed
