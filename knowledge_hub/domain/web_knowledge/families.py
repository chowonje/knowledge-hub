from __future__ import annotations

import re
from typing import Any

from knowledge_hub.domain.registry import normalize_domain_source


WEB_FAMILY_REFERENCE_EXPLAINER = "reference_explainer"
WEB_FAMILY_TEMPORAL_UPDATE = "temporal_update"
WEB_FAMILY_RELATION_EXPLAINER = "relation_explainer"
WEB_FAMILY_SOURCE_DISAMBIGUATION = "source_disambiguation"
WEB_FAMILY_VALUES = {
    WEB_FAMILY_REFERENCE_EXPLAINER,
    WEB_FAMILY_TEMPORAL_UPDATE,
    WEB_FAMILY_RELATION_EXPLAINER,
    WEB_FAMILY_SOURCE_DISAMBIGUATION,
}

_TEMPORAL_RE = re.compile(r"\b(latest|recent|updated?|newest|before|after|since|release|changed?)\b|최근|최신|업데이트|이전|이후|변경", re.IGNORECASE)
_RELATION_RE = re.compile(r"\b(related|relationship|connected|link|dependency|depends on|ontology)\b|관계|연결|의존|온톨로지", re.IGNORECASE)
_REFERENCE_RE = re.compile(r"\b(guide|guideline|reference|overview|summary|definition|what is|meaning)\b|가이드|레퍼런스|참고|개요|요약|정의|설명|무엇", re.IGNORECASE)
_DISAMBIGUATION_RE = re.compile(
    r"\b(versus|vs|difference|which|choose|distinguish|between)\b|구분|구별|차이|어디|무엇이 더 맞|뭐가 더 맞",
    re.IGNORECASE,
)
_DISAMBIGUATION_SOURCE_RE = re.compile(
    r"\b(reference|latest|watchlist|feed|source|sources)\b|"
    r"(?:reference|latest|watchlist|feed|source|sources)(?=[가-힣])|"
    r"reference source|latest news|watchlist|레퍼런스|피드|소스|최신",
    re.IGNORECASE,
)
_SOURCE_CLASS_HINT_RE = re.compile(
    r"reference|latest|watchlist|feed|source|sources|article|tutorial|레퍼런스|피드|소스|최신|아티클|튜토리얼",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_HOST_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE)
_ABSTAIN_RE = re.compile(
    r"\b(can we|can i|should we|is it safe to|can we conclude|can we claim)\b|단정할 수 있나|답해야 하나|출처 없이|결론을 단정",
    re.IGNORECASE,
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _has_source_disambiguation_target(body: str) -> bool:
    if _DISAMBIGUATION_SOURCE_RE.search(body):
        return True
    hint_tokens = {_clean_text(match.group(0)).casefold() for match in _SOURCE_CLASS_HINT_RE.finditer(body)}
    article_like = {"article", "tutorial", "아티클", "튜토리얼"}
    return len(hint_tokens & article_like) >= 2


def explicit_web_scope(query: str, metadata_filter: dict[str, Any] | None = None) -> str:
    scoped = dict(metadata_filter or {})
    for key in ("canonical_url", "url", "source_url", "document_id", "note_id"):
        token = _clean_text(scoped.get(key))
        if token:
            return token
    url_match = _URL_RE.search(str(query or ""))
    if url_match:
        return _clean_text(url_match.group(0))
    return ""


def host_hint(query: str) -> str:
    match = _HOST_RE.search(str(query or ""))
    return _clean_text(match.group(0) if match else "")


def classify_web_family(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> str:
    normalized_source = normalize_domain_source(source_type)
    body = _clean_text(query)
    if normalized_source != "web" and not explicit_web_scope(body, metadata_filter=metadata_filter):
        return ""
    if _DISAMBIGUATION_RE.search(body) and _has_source_disambiguation_target(body):
        return WEB_FAMILY_SOURCE_DISAMBIGUATION
    if _RELATION_RE.search(body):
        return WEB_FAMILY_RELATION_EXPLAINER
    if _TEMPORAL_RE.search(body) or _ABSTAIN_RE.search(body):
        return WEB_FAMILY_TEMPORAL_UPDATE
    if _DISAMBIGUATION_RE.search(body) and _has_source_disambiguation_target(body):
        return WEB_FAMILY_SOURCE_DISAMBIGUATION
    if explicit_web_scope(body, metadata_filter=metadata_filter):
        return WEB_FAMILY_REFERENCE_EXPLAINER
    if _REFERENCE_RE.search(body):
        return WEB_FAMILY_REFERENCE_EXPLAINER
    return WEB_FAMILY_REFERENCE_EXPLAINER


__all__ = [
    "WEB_FAMILY_REFERENCE_EXPLAINER",
    "WEB_FAMILY_TEMPORAL_UPDATE",
    "WEB_FAMILY_RELATION_EXPLAINER",
    "WEB_FAMILY_SOURCE_DISAMBIGUATION",
    "WEB_FAMILY_VALUES",
    "classify_web_family",
    "explicit_web_scope",
    "host_hint",
]
