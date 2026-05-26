"""Shared helpers for Storage / Source Ledger records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE_LEDGER_RECORD_SCHEMA = "knowledge-hub.source-ledger-record.v1"
DEFAULT_SOURCE_LEDGER_POLICY = {
    "classification": "UNKNOWN",
    "external_allowed": False,
    "redaction_required": True,
    "reasons": ["classification_not_evaluated"],
}


def source_ledger_archive_dir(sqlite_path: str | Path) -> Path:
    base = Path(sqlite_path).expanduser().resolve().parent
    path = base / "source_ledger"
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_ledger_record_path(archive: str | Path, record: dict[str, Any]) -> Path:
    source_type = _slugify(str(record.get("source_type") or "source"))
    ledger_id = _slugify(str(record.get("ledger_id") or record.get("source_id") or "source-ledger"))
    return Path(archive) / source_type / f"{ledger_id}.json"


def write_source_ledger_record(archive: str | Path, record: dict[str, Any]) -> Path:
    path = source_ledger_record_path(archive, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_source_ledger_record(
    *,
    ledger_id: str,
    source_id: str,
    source_type: str,
    canonical_uri: str,
    source_content_hash: str,
    raw_ref: str = "",
    normalized_ref: str = "",
    prepared_ref: str = "",
    indexed_ref: str = "",
    policy: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    created_at: str = "",
) -> dict[str, Any]:
    return {
        "schema": SOURCE_LEDGER_RECORD_SCHEMA,
        "ledger_id": str(ledger_id or source_id),
        "source_id": str(source_id or ledger_id),
        "source_type": str(source_type or "source"),
        "canonical_uri": str(canonical_uri or raw_ref or source_id),
        "source_content_hash": str(source_content_hash or ""),
        "artifacts": {
            "raw_ref": str(raw_ref or ""),
            "normalized_ref": str(normalized_ref or ""),
            "prepared_ref": str(prepared_ref or ""),
            "indexed_ref": str(indexed_ref or ""),
        },
        "policy": dict(policy or DEFAULT_SOURCE_LEDGER_POLICY),
        "metadata": dict(metadata or {}),
        "created_at": str(created_at or "").strip() or datetime.now(timezone.utc).isoformat(),
    }


def _slugify(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = "".join(ch if ch.isalnum() else "-" for ch in lowered)
    while "--" in lowered:
        lowered = lowered.replace("--", "-")
    return lowered.strip("-") or "untitled"


__all__ = [
    "DEFAULT_SOURCE_LEDGER_POLICY",
    "SOURCE_LEDGER_RECORD_SCHEMA",
    "build_source_ledger_record",
    "source_ledger_archive_dir",
    "source_ledger_record_path",
    "write_source_ledger_record",
]
