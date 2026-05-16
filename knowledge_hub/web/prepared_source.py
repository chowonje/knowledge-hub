"""Prepared-source adapters for web-family ingestion."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from knowledge_hub.core.prepared_source_record import PREPARED_SOURCE_RECORD_SCHEMA
from knowledge_hub.web.quality import QualityDoc
from knowledge_hub.web.youtube_extractor import normalize_youtube_segments_for_indexing


def build_prepared_source_record_from_quality_doc(
    quality_doc: QualityDoc,
    *,
    source_id: str,
    topic: str = "",
    run_id: str = "",
    created_at: str = "",
    ledger_id: str = "",
    raw_ref: str = "",
) -> dict[str, Any]:
    """Build the common Preparation output envelope for a cleaned web document."""

    doc = quality_doc.doc
    source_metadata = dict(doc.source_metadata or {})
    canonical_uri = str(quality_doc.canonical_url or doc.url or "").strip()
    prepared_text = str(quality_doc.cleaned_content or doc.content or "").strip()
    source_content_hash = str(quality_doc.content_hash or "").strip() or _sha256(prepared_text)
    source_type = _prepared_source_type(source_metadata)
    created = str(created_at or doc.fetched_at or "").strip() or datetime.now(timezone.utc).isoformat()
    warnings = _string_list(source_metadata.get("warnings"))
    if doc.error:
        warnings.append(str(doc.error).strip())

    segments = _prepared_segments(
        prepared_text=prepared_text,
        canonical_uri=canonical_uri,
        source_id=source_id,
        source_metadata=source_metadata,
    )
    parser = _parser_name(doc.engine, source_metadata)
    fallback_used = bool(
        parser == "asr"
        or source_metadata.get("fallback_used")
        or any("fallback" in item or "failed" in item for item in warnings)
    )

    record = {
        "schema": PREPARED_SOURCE_RECORD_SCHEMA,
        "record_id": _record_id(source_type, source_id, source_content_hash),
        "source_id": source_id,
        "source_type": source_type,
        "canonical_uri": canonical_uri,
        "source_content_hash": source_content_hash,
        "ledger_id": str(ledger_id or source_id),
        "raw_ref": str(raw_ref or canonical_uri),
        "title": str(doc.title or "").strip(),
        "metadata": _metadata(doc, source_metadata),
        "prepared": {
            "text": prepared_text,
            "text_hash": f"sha256:{_sha256(prepared_text)}",
            "language": str(source_metadata.get("language") or "").strip(),
            "segments": segments,
        },
        "processing": {
            "processor": "youtube_extractor" if source_type == "youtube" else "web_preparer",
            "processor_version": "prepared-source-record-v1",
            "parser": parser,
            "extraction_mode": "deterministic",
            "fallback_used": fallback_used,
            "fallback_chain": _fallback_chain(parser, source_metadata),
            "diagnostics": {
                "run_id": run_id,
                "topic": topic,
                "engine": doc.engine,
                "text_length": len(prepared_text),
                "segment_count": len(segments),
                "quality": quality_doc.assessment.to_dict(),
            },
            "warnings": warnings,
            "errors": [str(doc.error).strip()] if doc.error else [],
        },
        "quality": {
            "passed": bool(quality_doc.assessment.approved),
            "score": float(quality_doc.assessment.score),
            "flags": list(quality_doc.assessment.reasons),
            "reasons": list(quality_doc.assessment.reasons),
        },
        "lifecycle": {
            "stale": False,
            "stale_reason": None,
            "invalidated_at": None,
        },
        "created_at": created,
    }
    return record


def _prepared_source_type(source_metadata: dict[str, Any]) -> str:
    media_platform = str(source_metadata.get("media_platform") or "").strip().lower()
    if media_platform == "youtube":
        return "youtube"
    return str(source_metadata.get("source_type") or "web").strip() or "web"


def _prepared_segments(
    *,
    prepared_text: str,
    canonical_uri: str,
    source_id: str,
    source_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    media_platform = str(source_metadata.get("media_platform") or "").strip().lower()
    if media_platform == "youtube":
        transcript_segments = normalize_youtube_segments_for_indexing(
            source_metadata.get("transcript_segments") if isinstance(source_metadata.get("transcript_segments"), list) else []
        )
        if transcript_segments:
            return _youtube_segments(
                transcript_segments,
                prepared_text=prepared_text,
                canonical_uri=canonical_uri,
                source_id=source_id,
            )

    if not prepared_text:
        return []
    return [
        {
            "segment_id": f"{source_id}:seg:0000",
            "text": prepared_text,
            "char_start": 0,
            "char_end": len(prepared_text),
            "locator": {
                "section": "Document",
                "source_ref": canonical_uri,
            },
        }
    ]


def _youtube_segments(
    transcript_segments: list[dict[str, Any]],
    *,
    prepared_text: str,
    canonical_uri: str,
    source_id: str,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    cursor = 0
    for index, item in enumerate(transcript_segments):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        start_sec = _float_or_none(item.get("start_sec"))
        end_sec = _float_or_none(item.get("end_sec"))
        char_start, char_end, cursor = _locate_text(prepared_text, text, cursor)
        locator: dict[str, Any] = {
            "section": "Transcript",
            "timestamp_start_sec": start_sec,
            "timestamp_end_sec": end_sec,
            "source_ref": _timestamp_ref(canonical_uri, start_sec),
        }
        prepared.append(
            {
                "segment_id": f"{source_id}:seg:{index:04d}",
                "text": text,
                "char_start": char_start,
                "char_end": char_end,
                "locator": locator,
            }
        )
    return prepared


def _locate_text(prepared_text: str, text: str, cursor: int) -> tuple[int | None, int | None, int]:
    if not prepared_text or not text:
        return None, None, cursor
    start = prepared_text.find(text, max(0, cursor))
    if start < 0:
        start = prepared_text.find(text)
    if start < 0:
        return None, None, cursor
    end = start + len(text)
    return start, end, end


def _timestamp_ref(canonical_uri: str, start_sec: float | None) -> str:
    if start_sec is None:
        return canonical_uri
    return f"{canonical_uri}#t={max(0, start_sec):.3f}"


def _metadata(doc: Any, source_metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "title": str(doc.title or "").strip(),
        "authors": [str(doc.author).strip()] if str(doc.author or "").strip() else [],
        "published_at": str(doc.published_at or "").strip(),
        "language": str(source_metadata.get("language") or "").strip(),
        "source_name": str(source_metadata.get("source_name") or "").strip(),
        "source_vendor": str(source_metadata.get("source_vendor") or "").strip(),
        "source_channel": str(source_metadata.get("source_channel") or "").strip(),
        "source_item_id": str(source_metadata.get("source_item_id") or "").strip(),
        "tags": list(doc.tags or []),
    }
    for key in (
        "media_platform",
        "media_type",
        "video_id",
        "channel_name",
        "channel_id",
        "duration_sec",
        "transcript_source",
    ):
        if key in source_metadata:
            metadata[key] = source_metadata.get(key)
    return metadata


def _parser_name(engine: str, source_metadata: dict[str, Any]) -> str:
    transcript_source = str(source_metadata.get("transcript_source") or "").strip()
    if transcript_source:
        return transcript_source
    return str(engine or "web").strip() or "web"


def _fallback_chain(parser: str, source_metadata: dict[str, Any]) -> list[str]:
    chain = _string_list(source_metadata.get("fallback_chain"))
    if chain:
        return chain
    return [parser] if parser else []


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _float_or_none(value: Any) -> float | None:
    try:
        return round(float(value), 3)
    except Exception:
        return None


def _sha256(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _record_id(source_type: str, source_id: str, source_content_hash: str) -> str:
    digest = hashlib.sha1(f"{source_type}|{source_id}|{source_content_hash}".encode("utf-8")).hexdigest()[:16]
    return f"prepared:{source_type}:{digest}"
