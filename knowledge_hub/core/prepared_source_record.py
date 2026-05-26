"""Shared helpers for Processing / Preparation output records."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PREPARED_SOURCE_RECORD_SCHEMA = "knowledge-hub.prepared-source-record.v1"


def prepared_archive_dir(sqlite_path: str | Path) -> Path:
    base = Path(sqlite_path).expanduser().resolve().parent
    path = base / "prepared_sources"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepared_record_path(archive: str | Path, record: dict[str, Any]) -> Path:
    source_type = _slugify(str(record.get("source_type") or "source"))
    record_id = _slugify(str(record.get("record_id") or record.get("source_id") or "prepared-source"))
    return Path(archive) / source_type / f"{record_id}.json"


def write_prepared_record(archive: str | Path, record: dict[str, Any]) -> Path:
    path = prepared_record_path(archive, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    record["storage_ref"] = str(path)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def prepared_vector_metadata(record: dict[str, Any]) -> dict[str, Any]:
    prepared = record.get("prepared") if isinstance(record.get("prepared"), dict) else {}
    segments = prepared.get("segments") if isinstance(prepared.get("segments"), list) else []
    metadata: dict[str, Any] = {}
    for source_key, meta_key in (
        ("record_id", "prepared_record_id"),
        ("schema", "prepared_record_schema"),
        ("storage_ref", "prepared_record_path"),
        ("source_type", "prepared_source_type"),
    ):
        value = str(record.get(source_key) or "").strip()
        if value:
            metadata[meta_key] = value
    text_hash = str(prepared.get("text_hash") or "").strip()
    if text_hash:
        metadata["prepared_text_hash"] = text_hash
    metadata["prepared_segment_count"] = len(segments)
    return metadata


def build_prepared_source_record_from_text(
    *,
    source_id: str,
    source_type: str,
    canonical_uri: str,
    text: str,
    title: str = "",
    source_content_hash: str = "",
    ledger_id: str = "",
    raw_ref: str = "",
    metadata: dict[str, Any] | None = None,
    processor: str = "text_preparer",
    parser: str = "plain_text",
    created_at: str = "",
    quality_passed: bool = True,
    quality_score: float = 1.0,
    quality_flags: list[str] | None = None,
    locator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prepared_text = str(text or "").strip()
    source_hash = str(source_content_hash or "").strip() or f"sha256:{_sha256(prepared_text)}"
    source_type_value = str(source_type or "source").strip() or "source"
    source_id_value = str(source_id or "").strip()
    canonical_value = str(canonical_uri or raw_ref or source_id_value).strip()
    created = str(created_at or "").strip() or datetime.now(timezone.utc).isoformat()
    segment_locator = dict(locator or {})
    if not segment_locator:
        segment_locator = {
            "section": "Document",
            "source_ref": canonical_value,
        }
    segments = []
    if prepared_text:
        segments.append(
            {
                "segment_id": f"{source_id_value}:seg:0000",
                "text": prepared_text,
                "char_start": 0,
                "char_end": len(prepared_text),
                "locator": segment_locator,
            }
        )
    flags = [str(item).strip() for item in (quality_flags or []) if str(item).strip()]
    return {
        "schema": PREPARED_SOURCE_RECORD_SCHEMA,
        "record_id": _record_id(source_type_value, source_id_value, source_hash),
        "source_id": source_id_value,
        "source_type": source_type_value,
        "canonical_uri": canonical_value,
        "source_content_hash": source_hash,
        "ledger_id": str(ledger_id or source_id_value),
        "raw_ref": str(raw_ref or canonical_value),
        "title": str(title or "").strip(),
        "metadata": dict(metadata or {}),
        "prepared": {
            "text": prepared_text,
            "text_hash": f"sha256:{_sha256(prepared_text)}",
            "language": str((metadata or {}).get("language") or "").strip(),
            "segments": segments,
        },
        "processing": {
            "processor": str(processor or "text_preparer"),
            "processor_version": "prepared-source-record-v1",
            "parser": str(parser or "plain_text"),
            "extraction_mode": "deterministic",
            "fallback_used": False,
            "fallback_chain": [str(parser or "plain_text")],
            "diagnostics": {
                "text_length": len(prepared_text),
                "segment_count": len(segments),
            },
            "warnings": [],
            "errors": [],
        },
        "quality": {
            "passed": bool(quality_passed),
            "score": float(quality_score),
            "flags": flags,
            "reasons": flags,
        },
        "lifecycle": {
            "stale": False,
            "stale_reason": None,
            "invalidated_at": None,
        },
        "created_at": created,
    }


def build_prepared_source_record_from_segments(
    *,
    source_id: str,
    source_type: str,
    canonical_uri: str,
    text: str,
    segments: list[dict[str, Any]],
    title: str = "",
    source_content_hash: str = "",
    ledger_id: str = "",
    raw_ref: str = "",
    metadata: dict[str, Any] | None = None,
    processor: str = "text_preparer",
    parser: str = "plain_text",
    created_at: str = "",
    quality_passed: bool = True,
    quality_score: float = 1.0,
    quality_flags: list[str] | None = None,
    fallback_used: bool = False,
    fallback_chain: list[str] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prepared_text = str(text or "").strip()
    source_hash = str(source_content_hash or "").strip() or f"sha256:{_sha256(prepared_text)}"
    source_type_value = str(source_type or "source").strip() or "source"
    source_id_value = str(source_id or "").strip()
    canonical_value = str(canonical_uri or raw_ref or source_id_value).strip()
    created = str(created_at or "").strip() or datetime.now(timezone.utc).isoformat()
    normalized_segments: list[dict[str, Any]] = []
    for index, raw_segment in enumerate(list(segments or [])):
        segment_text = str(raw_segment.get("text") or "").strip()
        if not segment_text:
            continue
        segment_id = str(raw_segment.get("segment_id") or "").strip() or f"{source_id_value}:seg:{index:04d}"
        locator = dict(raw_segment.get("locator") or {})
        locator.setdefault("source_ref", canonical_value)
        normalized_segments.append(
            {
                "segment_id": segment_id,
                "text": segment_text,
                "char_start": raw_segment.get("char_start"),
                "char_end": raw_segment.get("char_end"),
                "locator": locator,
            }
        )
    flags = [str(item).strip() for item in (quality_flags or []) if str(item).strip()]
    fallback_items = [str(item).strip() for item in (fallback_chain or [parser]) if str(item).strip()]
    processing_diagnostics = {
        "text_length": len(prepared_text),
        "segment_count": len(normalized_segments),
    }
    processing_diagnostics.update(dict(diagnostics or {}))
    return {
        "schema": PREPARED_SOURCE_RECORD_SCHEMA,
        "record_id": _record_id(source_type_value, source_id_value, source_hash),
        "source_id": source_id_value,
        "source_type": source_type_value,
        "canonical_uri": canonical_value,
        "source_content_hash": source_hash,
        "ledger_id": str(ledger_id or source_id_value),
        "raw_ref": str(raw_ref or canonical_value),
        "title": str(title or "").strip(),
        "metadata": dict(metadata or {}),
        "prepared": {
            "text": prepared_text,
            "text_hash": f"sha256:{_sha256(prepared_text)}",
            "language": str((metadata or {}).get("language") or "").strip(),
            "segments": normalized_segments,
        },
        "processing": {
            "processor": str(processor or "text_preparer"),
            "processor_version": "prepared-source-record-v1",
            "parser": str(parser or "plain_text"),
            "extraction_mode": "deterministic",
            "fallback_used": bool(fallback_used),
            "fallback_chain": fallback_items,
            "diagnostics": processing_diagnostics,
            "warnings": [],
            "errors": [],
        },
        "quality": {
            "passed": bool(quality_passed),
            "score": float(quality_score),
            "flags": flags,
            "reasons": flags,
        },
        "lifecycle": {
            "stale": False,
            "stale_reason": None,
            "invalidated_at": None,
        },
        "created_at": created,
    }


def _sha256(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _record_id(source_type: str, source_id: str, source_content_hash: str) -> str:
    digest = hashlib.sha1(f"{source_type}|{source_id}|{source_content_hash}".encode("utf-8")).hexdigest()[:16]
    return f"prepared:{source_type}:{digest}"


def _slugify(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = "".join(ch if ch.isalnum() else "-" for ch in lowered)
    while "--" in lowered:
        lowered = lowered.replace("--", "-")
    return lowered.strip("-") or "untitled"


__all__ = [
    "PREPARED_SOURCE_RECORD_SCHEMA",
    "build_prepared_source_record_from_text",
    "build_prepared_source_record_from_segments",
    "prepared_archive_dir",
    "prepared_record_path",
    "prepared_vector_metadata",
    "write_prepared_record",
]
