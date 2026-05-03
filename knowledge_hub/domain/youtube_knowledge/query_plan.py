from __future__ import annotations

import json
import re
from typing import Any

from knowledge_hub.application.query_frame import (
    QUERY_FRAME_FAMILY_FROM_FRAME,
    QUERY_FRAME_FAMILY_FROM_PACK,
    NormalizedQueryFrame,
    build_query_frame,
    family_supported_for_source,
    normalize_query_frame_dict,
    query_frame_lock_mask,
    query_intent_supported_for_family,
)
from knowledge_hub.domain.registry import normalize_domain_source
from knowledge_hub.domain.youtube_knowledge.evidence_policy import policy_key_for_family
from knowledge_hub.domain.youtube_knowledge.families import (
    YOUTUBE_FAMILY_SECTION_LOOKUP,
    YOUTUBE_FAMILY_TIMESTAMP_LOOKUP,
    YOUTUBE_FAMILY_VIDEO_EXPLAINER,
    YOUTUBE_FAMILY_VIDEO_LOOKUP,
    classify_youtube_family,
    explicit_youtube_scope,
)
from knowledge_hub.web.ingest import make_web_note_id
from knowledge_hub.web.youtube_extractor import is_youtube_url, youtube_video_id_from_url, youtube_watch_url


_DOMAIN_KEY = "youtube_knowledge"
_TOKEN_RE = re.compile(r"https?://[^\s]+|[A-Za-z0-9._+-]+|[가-힣]+")
_ENTITY_STOPWORDS = {
    "youtube",
    "video",
    "영상",
    "유튜브",
    "요약",
    "정리",
    "설명",
    "뭐야",
    "무엇",
    "이",
    "은",
    "는",
    "가",
    "을",
    "를",
    "에서",
    "어디",
    "언제",
    "몇분",
    "트랜스크립트",
    "자막",
    "챕터",
    "section",
    "transcript",
    "chapter",
}
_TIMESTAMP_RE = re.compile(
    r"\b(when|timestamp|timecode|minute|minutes|second|seconds|where in the video)\b|언제|몇분|몇 초|타임스탬프|시점|시간대",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(
    r"\b(transcript|chapter|section|description|intro|outro)\b|트랜스크립트|자막|챕터|섹션|설명란|소개",
    re.IGNORECASE,
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _dedupe(values: list[Any], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = _clean_text(raw)
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
        if limit is not None and len(result) >= limit:
            break
    return result


def _query_terms(query: str) -> list[str]:
    result: list[str] = []
    for raw in _TOKEN_RE.findall(_clean_text(query)):
        token = _clean_text(raw)
        lowered = token.casefold()
        if not token or lowered in _ENTITY_STOPWORDS:
            continue
        if token.startswith("http://") or token.startswith("https://"):
            continue
        if re.fullmatch(r"[A-Za-z0-9._+-]+", token) and len(token) < 3:
            continue
        result.append(token)
    return _dedupe(result, limit=6)


def _query_intent(query: str, family: str) -> str:
    if family == YOUTUBE_FAMILY_TIMESTAMP_LOOKUP:
        return "temporal"
    if family == YOUTUBE_FAMILY_SECTION_LOOKUP:
        return "section_lookup"
    if family == YOUTUBE_FAMILY_VIDEO_LOOKUP:
        return "video_lookup"
    return "definition"


def _answer_mode(query: str, family: str) -> str:
    if family == YOUTUBE_FAMILY_TIMESTAMP_LOOKUP:
        return "timeline_grounded"
    if family == YOUTUBE_FAMILY_SECTION_LOOKUP:
        return "section_summary"
    if family == YOUTUBE_FAMILY_VIDEO_LOOKUP:
        return "video_scoped_answer"
    return "concise_summary"


def _parse_note_metadata(row: dict[str, Any] | None) -> dict[str, Any]:
    raw = (row or {}).get("metadata")
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw or "{}")
    except Exception:
        parsed = {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _youtube_note_candidates(sqlite_db: Any | None) -> list[dict[str, Any]]:
    if not sqlite_db:
        return []
    rows = list(sqlite_db.list_notes(source_type="web", limit=1000) or [])
    result: list[dict[str, Any]] = []
    for row in rows:
        note = dict(row or {})
        metadata = _parse_note_metadata(note)
        if _clean_text(metadata.get("media_platform")).casefold() != "youtube":
            continue
        note["__meta"] = metadata
        result.append(note)
    return result


def _resolve_youtube_scope(
    query: str,
    *,
    sqlite_db: Any | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    scoped = _clean_text(explicit_youtube_scope(query, metadata_filter=metadata_filter))
    if scoped:
        if is_youtube_url(scoped):
            canonical_url = youtube_watch_url(scoped)
            video_id = youtube_video_id_from_url(canonical_url)
            resolved = [canonical_url, make_web_note_id(canonical_url)]
            if video_id:
                resolved.append(video_id)
            return _dedupe(resolved, limit=3), [canonical_url, video_id], {
                "canonical_url": canonical_url,
                "document_id": make_web_note_id(canonical_url),
                "video_id": video_id,
            }
        return [scoped], [scoped], {"document_id": scoped}
    best_note: dict[str, Any] | None = None
    best_score = 0.0
    for row in _youtube_note_candidates(sqlite_db):
        metadata = dict(row.get("__meta") or {})
        title = _clean_text(row.get("title"))
        canonical_url = _clean_text(metadata.get("canonical_url") or metadata.get("url"))
        video_id = _clean_text(metadata.get("video_id") or metadata.get("source_item_id"))
        channel = _clean_text(metadata.get("channel_name") or metadata.get("source_channel"))
        haystack = " ".join([title, canonical_url, video_id, channel]).casefold()
        score = 0.0
        if canonical_url and canonical_url.casefold() in _clean_text(query).casefold():
            score += 4.0
        if video_id and video_id.casefold() in _clean_text(query).casefold():
            score += 4.0
        for term in _query_terms(query)[:4]:
            if term.casefold() in haystack:
                score += 1.0
        if score > best_score and score >= 2.0:
            best_score = score
            best_note = row
    if not best_note:
        return [], [], {}
    metadata = dict(best_note.get("__meta") or {})
    canonical_url = _clean_text(metadata.get("canonical_url") or metadata.get("url"))
    note_id = _clean_text(best_note.get("id"))
    video_id = _clean_text(metadata.get("video_id") or metadata.get("source_item_id"))
    resolved = [canonical_url, note_id, video_id]
    expanded = [best_note.get("title"), canonical_url, video_id, metadata.get("channel_name")]
    return _dedupe(resolved, limit=3), _dedupe(expanded, limit=4), {
        "canonical_url": canonical_url,
        "document_id": note_id,
        "video_id": video_id,
    }


def _canonical_entity_ids(query: str, *, sqlite_db: Any | None = None) -> list[str]:
    if not sqlite_db:
        return []
    try:
        from knowledge_hub.learning.resolver import EntityResolver

        resolver = EntityResolver(sqlite_db)
        result: list[str] = []
        for token in _query_terms(query)[:4]:
            identity = resolver.resolve(token, entity_type="concept")
            if identity is None:
                continue
            result.append(str(identity.canonical_id or ""))
        return _dedupe(result, limit=6)
    except Exception:
        return []


def _expanded_terms(query: str, family: str, scoped_terms: list[str]) -> list[str]:
    result = list(scoped_terms)
    if family == YOUTUBE_FAMILY_TIMESTAMP_LOOKUP:
        result.extend(["timestamp", "timecode", "transcript"])
    elif family == YOUTUBE_FAMILY_SECTION_LOOKUP:
        result.extend(["transcript", "section", "description"])
    elif family == YOUTUBE_FAMILY_VIDEO_LOOKUP:
        result.extend(["video summary", "transcript summary"])
    return _dedupe([*result, *_query_terms(query)], limit=8)


def build_rule_based_query_frame(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    sqlite_db: Any | None = None,
) -> NormalizedQueryFrame:
    family = classify_youtube_family(query, source_type=source_type, metadata_filter=metadata_filter)
    resolved_source_ids, scoped_terms, scoped_meta = _resolve_youtube_scope(
        query,
        sqlite_db=sqlite_db,
        metadata_filter=metadata_filter,
    )
    query_intent = _query_intent(query, family)
    answer_mode = _answer_mode(query, family)
    merged_filter = {
        "source_type": "web",
        "media_platform": "youtube",
        "youtube_only": True,
        **dict(metadata_filter or {}),
    }
    merged_filter["source_type"] = "web"
    merged_filter["media_platform"] = "youtube"
    if family == YOUTUBE_FAMILY_TIMESTAMP_LOOKUP:
        merged_filter["timeline_required"] = True
    if family == YOUTUBE_FAMILY_SECTION_LOOKUP:
        merged_filter["section_preferred"] = True
    for key in ("canonical_url", "document_id", "video_id"):
        token = _clean_text(scoped_meta.get(key))
        if token:
            merged_filter[key] = token
    return build_query_frame(
        domain_key=_DOMAIN_KEY,
        source_type=str(source_type or "youtube"),
        family=family or YOUTUBE_FAMILY_VIDEO_EXPLAINER,
        query_intent=query_intent,
        answer_mode=answer_mode,
        entities=_query_terms(query),
        canonical_entity_ids=_canonical_entity_ids(query, sqlite_db=sqlite_db),
        expanded_terms=_expanded_terms(query, family, scoped_terms),
        resolved_source_ids=resolved_source_ids,
        confidence=0.92 if resolved_source_ids else 0.74,
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key=policy_key_for_family(family or YOUTUBE_FAMILY_VIDEO_EXPLAINER),
        metadata_filter=merged_filter,
    )


def build_rule_query_plan(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    sqlite_db: Any | None = None,
) -> NormalizedQueryFrame:
    return build_rule_based_query_frame(
        query,
        source_type=source_type,
        metadata_filter=metadata_filter,
        sqlite_db=sqlite_db,
    )


def query_frame_from_query_plan(
    query_plan: dict[str, Any],
    *,
    query: str = "",
    source_type: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    sqlite_db: Any | None = None,
) -> NormalizedQueryFrame:
    base_frame = build_rule_based_query_frame(
        query,
        source_type=source_type,
        metadata_filter=metadata_filter,
        sqlite_db=sqlite_db,
    )
    payload = normalize_query_frame_dict(query_plan)
    if not payload:
        return base_frame
    effective_source = normalize_domain_source(payload.get("source_type") or source_type or base_frame.source_type)
    family = str(payload.get("family") or "").strip().lower()
    family_source = QUERY_FRAME_FAMILY_FROM_FRAME if family else QUERY_FRAME_FAMILY_FROM_PACK
    overrides = list(payload.get("overrides_applied") or [])
    if family and not family_supported_for_source(effective_source, family):
        family = base_frame.family
        family_source = QUERY_FRAME_FAMILY_FROM_PACK
        overrides.append("INVALID_FAMILY")
    elif not family:
        family = base_frame.family
        family_source = str(base_frame.family_source or QUERY_FRAME_FAMILY_FROM_PACK)
    query_intent = str(payload.get("query_intent") or "").strip()
    if query_intent and not query_intent_supported_for_family(effective_source, family, query_intent):
        query_intent = base_frame.query_intent
        overrides.append("UNSUPPORTED_INTENT")
    elif not query_intent:
        query_intent = base_frame.query_intent
    answer_mode = str(payload.get("answer_mode") or "").strip()
    if not answer_mode or {"INVALID_FAMILY", "UNSUPPORTED_INTENT"} & set(overrides):
        answer_mode = base_frame.answer_mode
    effective_metadata_filter = dict(base_frame.metadata_filter or {})
    effective_metadata_filter.update(dict(metadata_filter or {}))
    effective_metadata_filter.update(dict(payload.get("metadata_filter") or {}))
    return build_query_frame(
        domain_key=str(payload.get("domain_key") or base_frame.domain_key or _DOMAIN_KEY),
        source_type=effective_source or base_frame.source_type,
        family=family or YOUTUBE_FAMILY_VIDEO_EXPLAINER,
        query_intent=query_intent or "definition",
        answer_mode=answer_mode or "concise_summary",
        entities=_dedupe([*list(payload.get("entities") or []), *list(base_frame.entities or [])], limit=6),
        canonical_entity_ids=list(payload.get("canonical_entity_ids") or base_frame.canonical_entity_ids),
        expanded_terms=_dedupe([*list(payload.get("expanded_terms") or []), *list(base_frame.expanded_terms or [])], limit=8),
        resolved_source_ids=_dedupe([*list(payload.get("resolved_source_ids") or []), *list(base_frame.resolved_source_ids or [])], limit=6),
        confidence=float(payload.get("confidence") or base_frame.confidence or 0.0),
        planner_status=str(payload.get("planner_status") or base_frame.planner_status or "not_attempted"),
        planner_reason=str(payload.get("planner_reason") or base_frame.planner_reason or "rule_based"),
        evidence_policy_key=str(payload.get("evidence_policy_key") or base_frame.evidence_policy_key or policy_key_for_family(family)),
        metadata_filter=effective_metadata_filter,
        frame_provenance=str(payload.get("frame_provenance") or base_frame.frame_provenance or "derived"),
        trusted=bool(payload.get("trusted")),
        lock_mask=list(payload.get("lock_mask") or query_frame_lock_mask(payload)),
        family_source=family_source,
        overrides_applied=overrides,
    )


__all__ = [
    "build_rule_based_query_frame",
    "build_rule_query_plan",
    "query_frame_from_query_plan",
]
