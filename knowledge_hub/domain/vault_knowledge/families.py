from __future__ import annotations

import re
from typing import Any

from knowledge_hub.domain.registry import normalize_domain_source
from knowledge_hub.domain.vault_knowledge.scope import explicit_vault_scope


VAULT_FAMILY_NOTE_LOOKUP = "note_lookup"
VAULT_FAMILY_VAULT_EXPLAINER = "vault_explainer"
VAULT_FAMILY_VAULT_COMPARE = "vault_compare"
VAULT_FAMILY_VAULT_TIMELINE = "vault_timeline"
VAULT_FAMILY_VALUES = {
    VAULT_FAMILY_NOTE_LOOKUP,
    VAULT_FAMILY_VAULT_EXPLAINER,
    VAULT_FAMILY_VAULT_COMPARE,
    VAULT_FAMILY_VAULT_TIMELINE,
}

_TEMPORAL_RE = re.compile(r"\b(latest|recent|updated|newest|before|after|since|changed?)\b|최근|최신|업데이트|이전|이후|변경", re.IGNORECASE)
_COMPARE_RE = re.compile(r"\b(compare|comparison|difference|versus|vs)\b|비교|차이", re.IGNORECASE)
_LOOKUP_RE = re.compile(r"\b(note|notes|page|document|doc|abstract|summary)\b|노트|문서|페이지|요약|정리", re.IGNORECASE)
_EXPLAIN_RE = re.compile(r"\b(what is|define|definition|meaning|concept|core idea|principle|intuition|explain)\b|무엇|뭐야|정의|개념|원리|핵심|설명", re.IGNORECASE)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def classify_vault_family(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> str:
    normalized_source = normalize_domain_source(source_type)
    body = _clean_text(query)
    scoped = explicit_vault_scope(body, metadata_filter=metadata_filter)
    if normalized_source != "vault" and not scoped:
        return ""
    if scoped:
        return VAULT_FAMILY_NOTE_LOOKUP
    if _TEMPORAL_RE.search(body):
        return VAULT_FAMILY_VAULT_TIMELINE
    if _COMPARE_RE.search(body):
        return VAULT_FAMILY_VAULT_COMPARE
    if _LOOKUP_RE.search(body) and _EXPLAIN_RE.search(body):
        return VAULT_FAMILY_NOTE_LOOKUP
    return VAULT_FAMILY_VAULT_EXPLAINER


__all__ = [
    "VAULT_FAMILY_NOTE_LOOKUP",
    "VAULT_FAMILY_VAULT_COMPARE",
    "VAULT_FAMILY_VAULT_EXPLAINER",
    "VAULT_FAMILY_VAULT_TIMELINE",
    "VAULT_FAMILY_VALUES",
    "classify_vault_family",
    "explicit_vault_scope",
]
