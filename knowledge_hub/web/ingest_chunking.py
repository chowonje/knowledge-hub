"""Chunking and contextual-summary helpers for web ingest."""

from __future__ import annotations

import re
from typing import Any

from knowledge_hub.core.chunking import chunk_text_with_offsets as canonical_chunk_text_with_offsets, infer_content_type
from knowledge_hub.web.youtube_extractor import normalize_youtube_segments_for_indexing


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    return [
        str(item.get("text") or "")
        for item in canonical_chunk_text_with_offsets(
            text,
            content_type=infer_content_type(text=text, hint="plain"),
            chunk_size=chunk_size,
            overlap=overlap,
        )
    ]


def build_context_summary(title: str, chunk_text: str, section_title: str, section_path: str) -> str:
    normalized = re.sub(r"\s+", " ", (chunk_text or "").strip())
    if not normalized:
        return title

    first_sentence = normalized.split(". ")[0].strip()
    if len(first_sentence) > 180:
        first_sentence = f"{first_sentence[:177]}..."

    if section_title:
        if section_path and section_path != section_title:
            return f"[{section_path}] {first_sentence}"
        return f"[{section_title}] {first_sentence}"
    return f"[{title}] {first_sentence}"


def format_seconds_label(value: float | int | None) -> str:
    if value is None:
        return ""
    total = int(max(0, float(value)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_time_window_label(start_sec: float | int | None, end_sec: float | int | None) -> str:
    start_label = format_seconds_label(start_sec)
    end_label = format_seconds_label(end_sec)
    if start_label and end_label:
        return f"{start_label}-{end_label}"
    return start_label or end_label


def youtube_context_summary(
    *,
    chunk_text: str,
    chapter_title: str,
    start_sec: float | int | None,
    end_sec: float | int | None,
) -> str:
    normalized = re.sub(r"\s+", " ", (chunk_text or "").strip())
    if not normalized:
        return chapter_title or format_time_window_label(start_sec, end_sec)

    first_sentence = normalized.split(". ")[0].strip()
    if len(first_sentence) > 180:
        first_sentence = f"{first_sentence[:177]}..."

    timestamp_label = format_time_window_label(start_sec, end_sec)
    if chapter_title:
        return f"[Chapter: {chapter_title} | {timestamp_label}] {first_sentence}".strip()
    return f"[{timestamp_label}] {first_sentence}".strip()


def build_parent_metadata(
    source_type: str,
    document_id: str,
    title: str,
    section_title: str,
    section_path: str,
) -> dict[str, str]:
    parent_type = "section" if section_title else "document"
    parent_scope = (section_path or section_title or title or "__document__").strip()
    parent_scope = re.sub(r"\s+", " ", parent_scope) or "__document__"
    doc_key = (document_id or title or "unknown").strip() or "unknown"
    return {
        "document_id": f"{source_type}:{doc_key}",
        "parent_id": f"{source_type}:{doc_key}::{parent_type}:{parent_scope}",
        "parent_title": section_title or title,
        "parent_type": parent_type,
    }


def resolve_youtube_chapter(
    chapters: list[dict[str, Any]],
    *,
    point_sec: float | int | None,
) -> tuple[int, str]:
    if point_sec is None:
        return -1, ""
    current = float(point_sec)
    for index, chapter in enumerate(chapters):
        if not isinstance(chapter, dict):
            continue
        start_sec = chapter.get("start_sec")
        end_sec = chapter.get("end_sec")
        try:
            start_value = float(start_sec) if start_sec is not None else None
        except Exception:
            start_value = None
        try:
            end_value = float(end_sec) if end_sec is not None else None
        except Exception:
            end_value = None
        if start_value is None:
            continue
        if current < start_value:
            continue
        if end_value is not None and current >= end_value:
            continue
        return index, str(chapter.get("title") or "").strip()
    return -1, ""


def build_youtube_chunks_from_segments(
    transcript_segments: list[dict[str, Any]],
    *,
    chapters: list[dict[str, Any]] | None,
    title: str,
    source_type: str = "web",
    document_id: str = "",
    target_chars: int = 1200,
    max_chars: int = 1400,
    target_seconds: int = 60,
    max_seconds: int = 90,
    overlap_segments: int = 1,
) -> list[dict[str, Any]]:
    segments = normalize_youtube_segments_for_indexing(transcript_segments)
    if not segments:
        return []

    chapter_items = [item for item in (chapters or []) if isinstance(item, dict)]
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    chunk_index = 0

    def _chunk_chars(items: list[dict[str, Any]]) -> int:
        return sum(len(str(item.get("text") or "")) for item in items) + max(0, len(items) - 1)

    def _chunk_duration(items: list[dict[str, Any]]) -> float:
        if not items:
            return 0.0
        start_value = float(items[0].get("start_sec") or 0.0)
        end_value = float(items[-1].get("end_sec") or start_value)
        return max(0.0, end_value - start_value)

    def _build_chunk(items: list[dict[str, Any]], *, index: int) -> dict[str, Any]:
        chunk_text = "\n".join(str(item.get("text") or "").strip() for item in items if str(item.get("text") or "").strip()).strip()
        start_sec = float(items[0].get("start_sec") or 0.0)
        end_sec = float(items[-1].get("end_sec") or start_sec)
        chapter_index, chapter_title = resolve_youtube_chapter(chapter_items, point_sec=start_sec)
        timestamp_label = format_time_window_label(start_sec, end_sec)
        section_title = chapter_title or timestamp_label
        section_path = chapter_title or timestamp_label
        parent_type = "chapter" if chapter_title else "video_window"
        doc_key = (document_id or title or "unknown").strip() or "unknown"
        parent_scope = re.sub(r"\s+", " ", section_title).strip() or timestamp_label or "__video__"
        return {
            "text": chunk_text,
            "chunk_index": index,
            "start": 0,
            "end": len(chunk_text),
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "timestamp_label": timestamp_label,
            "chapter_title": chapter_title,
            "chapter_index": chapter_index,
            "section_title": section_title,
            "section_path": section_path,
            "document_id": f"{source_type}:{doc_key}",
            "parent_id": f"{source_type}:{doc_key}::{parent_type}:{parent_scope}",
            "parent_title": chapter_title or title,
            "parent_type": parent_type,
            "summary": youtube_context_summary(
                chunk_text=chunk_text,
                chapter_title=chapter_title,
                start_sec=start_sec,
                end_sec=end_sec,
            ),
        }

    def _flush(*, allow_overlap: bool) -> None:
        nonlocal current, chunk_index
        if not current:
            return
        chunk = _build_chunk(current, index=chunk_index)
        if chunk.get("text"):
            chunks.append(chunk)
            chunk_index += 1
        if allow_overlap and overlap_segments > 0 and len(current) > 1:
            current = [dict(item) for item in current[-overlap_segments:]]
        else:
            current = []

    for segment in segments:
        if not current:
            current = [dict(segment)]
            continue

        current_chapter_index, _ = resolve_youtube_chapter(chapter_items, point_sec=current[0].get("start_sec"))
        next_chapter_index, _ = resolve_youtube_chapter(chapter_items, point_sec=segment.get("start_sec"))
        if current_chapter_index != next_chapter_index:
            _flush(allow_overlap=False)
            current = [dict(segment)]
            continue

        current.append(dict(segment))
        current_chars = _chunk_chars(current)
        current_duration = _chunk_duration(current)
        if (
            current_duration >= max(1, target_seconds)
            or current_chars >= max(128, target_chars)
            or current_duration >= max(1, max_seconds)
            or current_chars >= max(256, max_chars)
        ):
            _flush(allow_overlap=True)

    _flush(allow_overlap=False)
    return chunks


def chunk_text_with_offsets(
    text: str,
    title: str,
    source_type: str = "web",
    document_id: str = "",
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return []

    chunks: list[dict[str, Any]] = []
    content_type = infer_content_type(text=text, hint="markdown")
    for item in canonical_chunk_text_with_offsets(
        text,
        content_type=content_type,
        chunk_size=chunk_size,
        overlap=overlap,
    ):
        section_title = str(item.get("section_title") or "")
        section_path = str(item.get("section_path") or "")
        chunk_text = str(item.get("text") or "")
        chunks.append(
            {
                "text": chunk_text,
                "chunk_index": int(item.get("chunk_index", 0)),
                "start": int(item.get("start", 0)),
                "end": int(item.get("end", 0)),
                "section_title": section_title,
                "section_path": section_path,
                **build_parent_metadata(
                    source_type=source_type,
                    document_id=document_id,
                    title=title,
                    section_title=section_title,
                    section_path=section_path,
                ),
                "summary": build_context_summary(
                    title=title,
                    chunk_text=chunk_text,
                    section_title=section_title,
                    section_path=section_path,
                ),
            }
        )
    return chunks


def resolve_web_chunks(
    row: dict[str, Any],
    *,
    title: str,
    source_type: str,
    document_id: str,
    content: str,
) -> list[dict[str, Any]]:
    media_platform = str(row.get("media_platform") or "").strip().lower()
    if media_platform == "youtube":
        raw_segments = row.get("transcript_segments")
        chapters = row.get("chapters")
        if isinstance(raw_segments, list):
            youtube_chunks = build_youtube_chunks_from_segments(
                raw_segments,
                chapters=chapters if isinstance(chapters, list) else [],
                title=title,
                source_type=source_type,
                document_id=document_id,
            )
            if youtube_chunks:
                return youtube_chunks

    return chunk_text_with_offsets(
        content,
        title=title,
        source_type=source_type,
        document_id=document_id,
    )


__all__ = [
    "build_context_summary",
    "build_parent_metadata",
    "build_youtube_chunks_from_segments",
    "chunk_text",
    "chunk_text_with_offsets",
    "format_seconds_label",
    "format_time_window_label",
    "resolve_web_chunks",
    "resolve_youtube_chapter",
    "youtube_context_summary",
]
