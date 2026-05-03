from __future__ import annotations

from typing import Any

from knowledge_hub.domain.web_knowledge.evidence_policy import policy_for_family, policy_key_for_family, select_evidence_policy
from knowledge_hub.domain.web_knowledge.families import (
    WEB_FAMILY_REFERENCE_EXPLAINER,
    WEB_FAMILY_RELATION_EXPLAINER,
    WEB_FAMILY_SOURCE_DISAMBIGUATION,
    WEB_FAMILY_TEMPORAL_UPDATE,
    WEB_FAMILY_VALUES,
    classify_web_family,
    explicit_web_scope,
    host_hint,
)
from knowledge_hub.domain.web_knowledge.query_plan import build_rule_based_query_frame, build_rule_query_plan, query_frame_from_query_plan


def classify_family(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, object] | None = None,
) -> str:
    return classify_web_family(query, source_type=source_type, metadata_filter=metadata_filter)


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
    titles: list[str] = []
    ids: list[str] = []
    rows = list(sqlite_db.list_notes(source_type="web", limit=1000) or [])
    lowered_entities = [str(item or "").strip().casefold() for item in entities if str(item or "").strip()]
    for row in rows:
        note = dict(row or {})
        metadata = dict(note.get("metadata") or {})
        title = str(note.get("title") or "").strip()
        canonical_url = str(metadata.get("canonical_url") or metadata.get("source_url") or "").strip()
        haystack = " ".join([title, canonical_url]).casefold()
        if lowered_entities and not any(item in haystack for item in lowered_entities):
            continue
        if canonical_url and canonical_url not in ids:
            ids.append(canonical_url)
        note_id = str(note.get("id") or "").strip()
        if note_id and note_id not in ids:
            ids.append(note_id)
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
    "WEB_FAMILY_REFERENCE_EXPLAINER",
    "WEB_FAMILY_RELATION_EXPLAINER",
    "WEB_FAMILY_SOURCE_DISAMBIGUATION",
    "WEB_FAMILY_TEMPORAL_UPDATE",
    "WEB_FAMILY_VALUES",
    "build_query_plan",
    "build_rule_based_query_frame",
    "build_rule_query_plan",
    "classify_family",
    "classify_web_family",
    "claim_alignment",
    "explicit_web_scope",
    "host_hint",
    "normalize",
    "policy_for_family",
    "policy_key_for_family",
    "representative_hint",
    "resolve_lookup",
    "select_evidence_policy",
]
