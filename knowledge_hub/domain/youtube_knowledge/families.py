from __future__ import annotations

import re
from typing import Any

from knowledge_hub.domain.registry import normalize_domain_source


YOUTUBE_FAMILY_VIDEO_LOOKUP = "video_lookup"
YOUTUBE_FAMILY_VIDEO_EXPLAINER = "video_explainer"
YOUTUBE_FAMILY_SECTION_LOOKUP = "section_lookup"
YOUTUBE_FAMILY_TIMESTAMP_LOOKUP = "timestamp_lookup"
YOUTUBE_FAMILY_VALUES = {
    YOUTUBE_FAMILY_VIDEO_LOOKUP,
    YOUTUBE_FAMILY_VIDEO_EXPLAINER,
    YOUTUBE_FAMILY_SECTION_LOOKUP,
    YOUTUBE_FAMILY_TIMESTAMP_LOOKUP,
}

_URL_RE = re.compile(r"https?://(?:www\.)?(?:youtube\.com|m\.youtube\.com|youtu\.be)/[^\s]+", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(
    r"\b(when|timestamp|timecode|minute|minutes|second|seconds|where in the video)\b|언제|몇분|몇 초|타임스탬프|시점|시간대",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(
    r"\b(transcript|chapter|section|description|intro|outro)\b|트랜스크립트|자막|챕터|섹션|설명란|소개",
    re.IGNORECASE,
)
_LOOKUP_RE = re.compile(
    r"\b(video|youtube|watch|watch\\?v=|summary|summarize|recap)\b|영상|유튜브|요약|정리",
    re.IGNORECASE,
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def explicit_youtube_scope(query: str, metadata_filter: dict[str, Any] | None = None) -> str:
    scoped = dict(metadata_filter or {})
    for key in ("canonical_url", "url", "source_url", "document_id", "video_id"):
        token = _clean_text(scoped.get(key))
        if token:
            return token
    match = _URL_RE.search(str(query or ""))
    return _clean_text(match.group(0) if match else "")


def classify_youtube_family(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> str:
    normalized_source = normalize_domain_source(source_type)
    body = _clean_text(query)
    if normalized_source != "youtube" and not explicit_youtube_scope(body, metadata_filter=metadata_filter):
        return ""
    if _TIMESTAMP_RE.search(body):
        return YOUTUBE_FAMILY_TIMESTAMP_LOOKUP
    if _SECTION_RE.search(body):
        return YOUTUBE_FAMILY_SECTION_LOOKUP
    if explicit_youtube_scope(body, metadata_filter=metadata_filter):
        return YOUTUBE_FAMILY_VIDEO_LOOKUP
    if _LOOKUP_RE.search(body):
        return YOUTUBE_FAMILY_VIDEO_LOOKUP
    return YOUTUBE_FAMILY_VIDEO_EXPLAINER


__all__ = [
    "YOUTUBE_FAMILY_SECTION_LOOKUP",
    "YOUTUBE_FAMILY_TIMESTAMP_LOOKUP",
    "YOUTUBE_FAMILY_VALUES",
    "YOUTUBE_FAMILY_VIDEO_EXPLAINER",
    "YOUTUBE_FAMILY_VIDEO_LOOKUP",
    "classify_youtube_family",
    "explicit_youtube_scope",
]
