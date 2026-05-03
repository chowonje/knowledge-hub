from __future__ import annotations

from typing import Any

from knowledge_hub.domain.vault_knowledge.evidence_policy import (
    policy_for_family,
    policy_key_for_family,
    select_evidence_policy,
)
from knowledge_hub.domain.vault_knowledge.families import (
    VAULT_FAMILY_NOTE_LOOKUP,
    VAULT_FAMILY_VALUES,
    VAULT_FAMILY_VAULT_COMPARE,
    VAULT_FAMILY_VAULT_EXPLAINER,
    VAULT_FAMILY_VAULT_TIMELINE,
    classify_vault_family,
    explicit_vault_scope,
)
from knowledge_hub.domain.vault_knowledge.query_plan import (
    build_rule_based_query_frame,
    build_rule_query_plan,
    query_frame_from_query_plan,
)
from knowledge_hub.domain.vault_knowledge.scope import (
    vault_scope_from_filter,
    vault_scope_from_query,
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def classify_family(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, object] | None = None,
) -> str:
    return classify_vault_family(query, source_type=source_type, metadata_filter=metadata_filter)


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
    rows = list(sqlite_db.list_notes(source_type="vault", limit=1000) or [])
    for row in rows:
        note = dict(row or {})
        title = _clean_text(note.get("title"))
        file_path = _clean_text(note.get("file_path"))
        haystack = " ".join([title, file_path]).casefold()
        if lowered_entities and not any(item in haystack for item in lowered_entities):
            continue
        note_id = _clean_text(note.get("id"))
        if note_id and note_id not in ids:
            ids.append(note_id)
        if file_path and file_path not in ids:
            ids.append(file_path)
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
    "VAULT_FAMILY_NOTE_LOOKUP",
    "VAULT_FAMILY_VALUES",
    "VAULT_FAMILY_VAULT_COMPARE",
    "VAULT_FAMILY_VAULT_EXPLAINER",
    "VAULT_FAMILY_VAULT_TIMELINE",
    "build_query_plan",
    "build_rule_based_query_frame",
    "build_rule_query_plan",
    "classify_family",
    "classify_vault_family",
    "claim_alignment",
    "explicit_vault_scope",
    "normalize",
    "policy_for_family",
    "policy_key_for_family",
    "query_frame_from_query_plan",
    "representative_hint",
    "resolve_lookup",
    "select_evidence_policy",
    "vault_scope_from_filter",
    "vault_scope_from_query",
]
