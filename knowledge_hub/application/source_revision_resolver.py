"""Read-only source revision resolution for persisted evidence registry records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from knowledge_hub.infrastructure.persistence.stores.derivative_lifecycle import (
    source_hash_from_content,
    source_hash_from_payload,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _first_key(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _conn(sqlite_db: Any) -> Any | None:
    conn = getattr(sqlite_db, "conn", None)
    if conn is not None:
        return conn
    registry = getattr(sqlite_db, "registry", None)
    return getattr(registry, "conn", None)


def _metadata_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _source_variants(source_id: str) -> list[str]:
    token = _clean_text(source_id)
    base = token.split("#", 1)[0].strip()
    variants: list[str] = []
    for item in (token, base):
        if item and item not in variants:
            variants.append(item)
    if ":" in base:
        scheme, rest = base.split(":", 1)
        rest = rest.strip()
        if rest and rest not in variants:
            variants.append(rest)
        if scheme == "paper" and rest:
            paper_doc = f"paper:{rest}"
            if paper_doc not in variants:
                variants.append(paper_doc)
    return variants


def _document_memory_hash(conn: Any, variants: list[str]) -> tuple[str, str]:
    for document_id in variants:
        try:
            row = conn.execute(
                """
                SELECT source_content_hash FROM document_memory_units
                WHERE document_id = ?
                  AND source_content_hash != ''
                  AND COALESCE(stale, 0) = 0
                ORDER BY
                    CASE WHEN unit_type = 'document_summary' THEN 0 ELSE 1 END ASC,
                    updated_at DESC
                LIMIT 1
                """,
                (document_id,),
            ).fetchone()
        except Exception:
            row = None
        source_hash = _clean_text(row["source_content_hash"] if row else "")
        if source_hash:
            return source_hash, document_id
    return "", ""


def _note_hash(sqlite_db: Any, variants: list[str]) -> tuple[str, str]:
    getter = getattr(sqlite_db, "get_note", None)
    if not callable(getter):
        return "", ""
    for note_id in variants:
        try:
            note = getter(note_id)
        except Exception:
            note = None
        if not note:
            continue
        metadata = _metadata_dict(note.get("metadata"))
        source_hash = _clean_text(
            metadata.get("source_content_hash")
            or source_hash_from_content(
                content=note.get("content"),
                metadata=metadata,
                identity=note.get("id") or note_id,
            )
        )
        if source_hash:
            return source_hash, note_id
    return "", ""


def _paper_hash(sqlite_db: Any, variants: list[str]) -> tuple[str, str]:
    getter = getattr(sqlite_db, "get_paper", None)
    if not callable(getter):
        return "", ""
    paper_ids = []
    for token in variants:
        candidate = token.split("paper:", 1)[1] if token.startswith("paper:") else token
        if candidate and candidate not in paper_ids:
            paper_ids.append(candidate)
    for paper_id in paper_ids:
        try:
            paper = getter(paper_id)
        except Exception:
            paper = None
        if not paper:
            continue
        source_hash = source_hash_from_payload(dict(paper))
        if source_hash:
            return source_hash, paper_id
    return "", ""


def _resolve_one(sqlite_db: Any, source_id: str) -> dict[str, Any]:
    variants = _source_variants(source_id)
    source_hash, matched_id = _note_hash(sqlite_db, variants)
    if source_hash:
        return {
            "sourceId": source_id,
            "sourceContentHash": source_hash,
            "resolvedFrom": "note",
            "resolvedId": matched_id,
        }
    source_hash, matched_id = _paper_hash(sqlite_db, variants)
    if source_hash:
        return {
            "sourceId": source_id,
            "sourceContentHash": source_hash,
            "resolvedFrom": "paper",
            "resolvedId": matched_id,
        }
    conn = _conn(sqlite_db)
    if conn is not None:
        source_hash, matched_id = _document_memory_hash(conn, variants)
        if source_hash:
            return {
                "sourceId": source_id,
                "sourceContentHash": source_hash,
                "resolvedFrom": "document_memory",
                "resolvedId": matched_id,
            }
    return {"sourceId": source_id, "sourceContentHash": "", "resolvedFrom": "", "resolvedId": ""}


def resolve_current_source_refs(sqlite_db: Any, source_refs: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Resolve current source hashes without mutating registry or index state."""

    checked_at = _now_iso()
    if sqlite_db is None:
        return {
            "status": "unchecked",
            "checkedAt": checked_at,
            "reason": "sqlite_unavailable",
            "currentSourceRefs": [],
            "missingSourceIds": [],
            "unresolvedSourceIds": [],
        }

    expected: dict[str, str] = {}
    for ref in source_refs or []:
        source_id = _first_key(ref, "sourceId", "source_id")
        recorded_hash = _first_key(ref, "sourceContentHash", "source_content_hash")
        if source_id:
            expected[source_id] = recorded_hash
    if not expected:
        return {
            "status": "unchecked",
            "checkedAt": checked_at,
            "reason": "registry_record_has_no_source_refs",
            "currentSourceRefs": [],
            "missingSourceIds": [],
            "unresolvedSourceIds": [],
        }

    current_refs: list[dict[str, Any]] = []
    missing: list[str] = []
    mismatched: list[str] = []
    matched: list[str] = []
    for source_id, recorded_hash in expected.items():
        resolved = _resolve_one(sqlite_db, source_id)
        current_hash = _clean_text(resolved.get("sourceContentHash"))
        if not current_hash:
            missing.append(source_id)
            continue
        current_refs.append(resolved)
        if recorded_hash and current_hash != recorded_hash:
            mismatched.append(source_id)
        elif recorded_hash:
            matched.append(source_id)

    if mismatched:
        status = "stale"
        reason = "source_revision_mismatch"
    elif missing:
        status = "source_missing"
        reason = "source_hash_not_found"
    elif matched and len(matched) == len(expected):
        status = "fresh"
        reason = "source_hash_match"
    else:
        status = "unchecked"
        reason = "recorded_source_hash_missing"

    return {
        "status": status,
        "checkedAt": checked_at,
        "reason": reason,
        "currentSourceRefs": current_refs,
        "missingSourceIds": missing,
        "unresolvedSourceIds": missing,
        "matchedSourceIds": matched,
        "mismatchedSourceIds": mismatched,
    }


__all__ = ["resolve_current_source_refs"]
