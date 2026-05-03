from __future__ import annotations


from knowledge_hub.domain.ai_papers.families import (
    PAPER_FAMILY_COMPARE,
    PAPER_FAMILY_CONCEPT_EXPLAINER,
    PAPER_FAMILY_DISCOVER,
    PAPER_FAMILY_LOOKUP,
)
import re


_NOVICE_EXPLAINER_RE = re.compile(
    r"\b(beginner|beginners|novice|intro|introduction|easy|simply|simple)\b|쉽게|초심자|입문|처음",
    re.IGNORECASE,
)


def paper_family_answer_mode(family: str, *, query: str | None = None) -> str:
    normalized = str(family or "").strip().lower()
    if normalized == PAPER_FAMILY_CONCEPT_EXPLAINER:
        if _NOVICE_EXPLAINER_RE.search(str(query or "")):
            return "representative_paper_explainer_beginner"
        return "representative_paper_explainer"
    if normalized == PAPER_FAMILY_LOOKUP:
        return "paper_scoped_answer"
    if normalized == PAPER_FAMILY_COMPARE:
        return "paper_comparison"
    if normalized == PAPER_FAMILY_DISCOVER:
        return "paper_shortlist_summary"
    return ""


def paper_family_query_intent(family: str, *, fallback: str = "definition") -> str:
    normalized = str(family or "").strip().lower()
    if normalized == PAPER_FAMILY_CONCEPT_EXPLAINER:
        return "definition"
    if normalized == PAPER_FAMILY_LOOKUP:
        return "paper_lookup"
    if normalized == PAPER_FAMILY_COMPARE:
        return "comparison"
    if normalized == PAPER_FAMILY_DISCOVER:
        return "paper_topic"
    return fallback


def paper_family_answer_scope(family: str, *, query: str | None = None) -> dict[str, str]:
    normalized = str(family or "").strip().lower()
    return {
        "paperFamily": normalized,
        "answerMode": paper_family_answer_mode(normalized, query=query),
        "queryIntent": paper_family_query_intent(normalized, fallback="definition"),
    }


__all__ = [
    "paper_family_answer_mode",
    "paper_family_answer_scope",
    "paper_family_query_intent",
]
