from __future__ import annotations

import re
from typing import Any

from knowledge_hub.domain.registry import normalize_domain_source


PAPER_FAMILY_CONCEPT_EXPLAINER = "concept_explainer"
PAPER_FAMILY_LOOKUP = "paper_lookup"
PAPER_FAMILY_COMPARE = "paper_compare"
PAPER_FAMILY_DISCOVER = "paper_discover"
PAPER_FAMILY_VALUES = {
    PAPER_FAMILY_CONCEPT_EXPLAINER,
    PAPER_FAMILY_LOOKUP,
    PAPER_FAMILY_COMPARE,
    PAPER_FAMILY_DISCOVER,
}

_COMPARE_RE = re.compile(r"\b(compare|comparison|versus|vs|difference|tradeoff|trade-off)\b|비교|차이|장단점", re.IGNORECASE)
_DISCOVER_RE = re.compile(
    r"\b(find|search|collect|curate|organize|overview|survey|related|representative|alternatives?)\b|찾아|찾아줘|정리|정리해줘|추천|관련|대표|대체|차세대|계열",
    re.IGNORECASE,
)
_LOOKUP_RE = re.compile(
    r"\b(paper|papers|arxiv|doi|abstract|summary|summarize|citation|citations)\b|논문|초록|요약",
    re.IGNORECASE,
)
_LOOKUP_ACTION_RE = re.compile(
    r"\b(abstract|summary|summarize|describe|explain)\b|초록|요약|정리|설명",
    re.IGNORECASE,
)
_EXPLAIN_RE = re.compile(
    r"\b(what is|define|definition|meaning|concept|core idea|main idea|principle|intuition|explain|explainer)\b|정의|개념|무엇|뭐야|뭐지|뭔지|핵심\s*아이디어|원리|직관|설명",
    re.IGNORECASE,
)
_ARXIV_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", re.IGNORECASE)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def explicit_paper_id(query: str, metadata_filter: dict[str, Any] | None = None) -> str:
    scoped = dict(metadata_filter or {})
    for key in ("paper_id", "arxiv_id", "doi"):
        token = _clean_text(scoped.get(key))
        if token:
            return token
    match = _ARXIV_RE.search(str(query or ""))
    if match:
        return _clean_text(match.group(0))
    doi_match = _DOI_RE.search(str(query or ""))
    return _clean_text(doi_match.group(0) if doi_match else "")


def _single_paper_lookup_signal(query: str) -> bool:
    body = _clean_text(query)
    if not body:
        return False
    if explicit_paper_id(body):
        return True
    has_lookup = bool(_LOOKUP_RE.search(body))
    has_lookup_action = bool(_LOOKUP_ACTION_RE.search(body))
    has_broad_discover = bool(
        re.search(r"\b(find|search|recommend|related|alternatives?)\b|찾아|찾아줘|추천|관련|대체|차세대|계열", body, re.IGNORECASE)
    )
    if has_lookup and has_lookup_action and not has_broad_discover:
        return True
    if "논문" in body and "설명" in body and not has_broad_discover:
        return True
    return False


def classify_paper_family(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> str:
    normalized_source = normalize_domain_source(source_type)
    scoped_paper_id = explicit_paper_id(query, metadata_filter=metadata_filter)
    body = _clean_text(query)

    if normalized_source != "paper" and not scoped_paper_id:
        return ""
    if scoped_paper_id:
        return PAPER_FAMILY_LOOKUP
    if _COMPARE_RE.search(body):
        return PAPER_FAMILY_COMPARE
    if _single_paper_lookup_signal(body):
        return PAPER_FAMILY_LOOKUP
    if _DISCOVER_RE.search(body):
        return PAPER_FAMILY_DISCOVER
    if _LOOKUP_RE.search(body) and not _EXPLAIN_RE.search(body):
        return PAPER_FAMILY_LOOKUP
    if _EXPLAIN_RE.search(body):
        return PAPER_FAMILY_CONCEPT_EXPLAINER
    return PAPER_FAMILY_DISCOVER


__all__ = [
    "PAPER_FAMILY_COMPARE",
    "PAPER_FAMILY_CONCEPT_EXPLAINER",
    "PAPER_FAMILY_DISCOVER",
    "PAPER_FAMILY_LOOKUP",
    "PAPER_FAMILY_VALUES",
    "classify_paper_family",
    "explicit_paper_id",
]
