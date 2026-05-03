from __future__ import annotations

import json
from typing import Any

from knowledge_hub.domain.youtube_knowledge.evidence_policy import (
    policy_for_family,
    policy_key_for_family,
    select_evidence_policy,
)
from knowledge_hub.domain.youtube_knowledge.families import (
    YOUTUBE_FAMILY_SECTION_LOOKUP,
    YOUTUBE_FAMILY_TIMESTAMP_LOOKUP,
    YOUTUBE_FAMILY_VALUES,
    YOUTUBE_FAMILY_VIDEO_EXPLAINER,
    YOUTUBE_FAMILY_VIDEO_LOOKUP,
    classify_youtube_family,
    explicit_youtube_scope,
)
from knowledge_hub.domain.youtube_knowledge.query_plan import (
    build_rule_based_query_frame,
    build_rule_query_plan,
    query_frame_from_query_plan,
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _parse_note_metadata(row: dict[str, Any] | None) -> dict[str, Any]:
    raw = (row or {}).get("metadata")
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw or "{}")
    except Exception:
        parsed = {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def classify_family(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, object] | None = None,
) -> str:
    return classify_youtube_family(query, source_type=source_type, metadata_filter=metadata_filter)


def normalize(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, object] | None = None,
    sqlite_db=None,
    query_plan: dict[str, object] | None = None,
):
    if query_plan:
        return query_frame_from_query_plan(
            dict(query_plan),
            query=query,
            source_type=source_type,
            metadata_filter=metadata_filter,
            sqlite_db=sqlite_db,
        )
    return build_rule_based_query_frame(
        query,
        source_type=source_type,
        metadata_filter=metadata_filter,
        sqlite_db=sqlite_db,
    )


def build_query_plan(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, object] | None = None,
    sqlite_db=None,
) -> dict[str, object]:
    return build_rule_query_plan(
        query,
        source_type=source_type,
        metadata_filter=metadata_filter,
        sqlite_db=sqlite_db,
    ).to_dict()


def resolve_lookup(
    entities: list[str],
    *,
    sqlite_db: Any | None = None,
) -> tuple[list[str], list[str]]:
    if not sqlite_db:
        return [], []
    ids: list[str] = []
    titles: list[str] = []
    lowered_entities = [str(item or "").strip().casefold() for item in entities if str(item or "").strip()]
    rows = list(sqlite_db.list_notes(source_type="web", limit=1000) or [])
    for row in rows:
        note = dict(row or {})
        metadata = _parse_note_metadata(note)
        if _clean_text(metadata.get("media_platform")).casefold() != "youtube":
            continue
        title = _clean_text(note.get("title"))
        canonical_url = _clean_text(metadata.get("canonical_url") or metadata.get("url"))
        video_id = _clean_text(metadata.get("video_id") or metadata.get("source_item_id"))
        haystack = " ".join([title, canonical_url, video_id]).casefold()
        if lowered_entities and not any(item in haystack for item in lowered_entities):
            continue
        for candidate in (canonical_url, _clean_text(note.get("id")), video_id):
            token = _clean_text(candidate)
            if token and token not in ids:
                ids.append(token)
        if title and title not in titles:
            titles.append(title)
        if len(ids) >= 4:
            break
    return ids[:4], titles[:4]


def representative_hint(
    entities: list[str],
    *,
    sqlite_db: Any | None = None,
) -> list[dict[str, Any]]:
    resolved_ids, titles = resolve_lookup(entities, sqlite_db=sqlite_db)
    hints: list[dict[str, Any]] = []
    for source_id, title in zip(resolved_ids, titles or resolved_ids, strict=False):
        hints.append(
            {
                "source_id": source_id,
                "title": title,
                "score": 0.8,
            }
        )
    return hints


def claim_alignment(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(cards or [])[:6]]


__all__ = [
    "YOUTUBE_FAMILY_SECTION_LOOKUP",
    "YOUTUBE_FAMILY_TIMESTAMP_LOOKUP",
    "YOUTUBE_FAMILY_VALUES",
    "YOUTUBE_FAMILY_VIDEO_EXPLAINER",
    "YOUTUBE_FAMILY_VIDEO_LOOKUP",
    "build_query_plan",
    "build_rule_based_query_frame",
    "build_rule_query_plan",
    "classify_family",
    "classify_youtube_family",
    "claim_alignment",
    "explicit_youtube_scope",
    "normalize",
    "policy_for_family",
    "policy_key_for_family",
    "representative_hint",
    "resolve_lookup",
    "select_evidence_policy",
]
