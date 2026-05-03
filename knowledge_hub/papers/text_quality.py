"""Shared text-quality heuristics for paper summary and memory surfaces."""

from __future__ import annotations

import re
from typing import Any

_PAGE_REF_RE = re.compile(r"\[[^\]]*page\s+\d+\]", re.IGNORECASE)
_PAGE_EXCERPT_RE = re.compile(r"\[[^\]]+>\s*Page\s+\d+\]", re.IGNORECASE)
_TABLE_FIGURE_MARKER_RE = re.compile(r"\b(?:table|figure)\s+\d+\b", re.IGNORECASE)
_EMAIL_MARKER_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b", re.IGNORECASE)
_AUTHOR_INITIAL_RE = re.compile(r"\b[A-Z]\.")
_RAW_ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9.+-]*")
_RAW_ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "with",
}


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def word_like_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z-]+|[가-힣]{2,}", str(text or "")))


def numeric_token_count(text: str) -> int:
    return len(re.findall(r"\b\d+(?:\.\d+)?%?\b", str(text or "")))


def contains_hangul(text: Any) -> bool:
    return bool(re.search(r"[가-힣]", str(text or "")))


def looks_author_contribution(text: str) -> bool:
    token = clean_text(text)
    lowered = token.casefold()
    if ":" not in token:
        return False
    contribution_markers = (
        "co-implemented",
        "implemented support",
        "assisted with",
        "performed",
        "collected",
        "annotated",
        "wrote",
        "maintained",
        "edited",
    )
    return token.count(":") >= 2 and any(marker in lowered for marker in contribution_markers)


def looks_caption_stub(text: str) -> bool:
    token = clean_text(text)
    lowered = token.casefold()
    if not token:
        return False
    if "|" in token:
        return True
    if lowered.startswith(("table ", "figure ", "fig. ", "fig ", "a comparison of ")):
        return True
    if "numbers in bold denote" in lowered or "experimental results at each stage" in lowered:
        return True
    if re.search(r"\b[A-Za-z]{2,}- [A-Za-z]{2,}\b", token):
        return True
    return False


def looks_page_stub(text: str) -> bool:
    token = clean_text(text)
    lowered = token.casefold()
    if not (_PAGE_REF_RE.search(token) or _PAGE_EXCERPT_RE.search(token)):
        return False
    if token.lstrip().startswith("["):
        return True
    if "et al." in lowered:
        return True
    return word_like_count(token) <= 8


def looks_table_heavy(text: str) -> bool:
    token = clean_text(text)
    numeric_count = numeric_token_count(token)
    word_count = word_like_count(token)
    return numeric_count >= 6 and numeric_count >= max(6, word_count)


def looks_author_stub(text: str) -> bool:
    token = clean_text(text)
    words = token.split()
    if len(words) > 8 or "," not in token:
        return False
    initials = len(_AUTHOR_INITIAL_RE.findall(token))
    capitalized = sum(1 for word in words if re.fullmatch(r"[A-Z][a-z]+\.?", word))
    return initials >= 2 and (initials + capitalized) >= 3


def is_front_matter_spillover(text: Any, *, title: str = "") -> bool:
    token = clean_text(text)
    if not token:
        return False
    lowered = token.casefold()
    title_token = clean_text(title).casefold()
    if _EMAIL_MARKER_RE.search(token):
        return True
    if looks_author_contribution(token) or looks_author_stub(token) or looks_page_stub(token):
        return True
    if title_token and lowered.startswith(title_token) and token.count(",") >= 3:
        return True
    if lowered.startswith(("authors:", "author:", "affiliation:", "affiliations:", "email:", "emails:")):
        return True
    if any(marker in lowered for marker in ("arxiv:", "corresponding author", "equal contribution", "all rights reserved")):
        return True
    return False


def is_table_caption_spillover(text: Any) -> bool:
    token = clean_text(text)
    lowered = token.casefold()
    if not token:
        return False
    if looks_caption_stub(token) or looks_table_heavy(token):
        return True
    if _TABLE_FIGURE_MARKER_RE.search(lowered):
        return True
    if any(
        marker in lowered
        for marker in (
            "numbers in bold",
            "higher is better",
            "lower is better",
            "standard deviation",
            "significance level",
            "p-value",
            "caption",
        )
    ):
        return True
    return False


def is_raw_english_spillover(text: Any) -> bool:
    token = clean_text(text)
    if not token or contains_hangul(token):
        return False
    english_tokens = _RAW_ENGLISH_TOKEN_RE.findall(token)
    if len(english_tokens) < 7:
        return False
    if len(token) < 48:
        return False
    stopword_hits = sum(1 for item in english_tokens if item.casefold() in _RAW_ENGLISH_STOPWORDS)
    sentence_like = bool(re.search(r"[.?!,:;]", token) or token.count(" ") >= 7)
    if not sentence_like or stopword_hits < 2:
        return False
    if all(re.fullmatch(r"[A-Z0-9][A-Za-z0-9.+-]*", item) for item in english_tokens) and len(english_tokens) <= 10:
        return False
    if token.count(" ") <= 3 and all("-" in item or item[0].isupper() for item in english_tokens):
        return False
    return True


def spillover_issues(text: Any, *, title: str = "") -> list[str]:
    token = clean_text(text)
    if not token:
        return []
    issues: list[str] = []
    if is_front_matter_spillover(token, title=title):
        issues.append("front_matter_spillover")
    if is_table_caption_spillover(token):
        issues.append("table_caption_spillover")
    if is_raw_english_spillover(token):
        issues.append("raw_english_spillover")
    return issues


__all__ = [
    "clean_text",
    "contains_hangul",
    "is_front_matter_spillover",
    "is_raw_english_spillover",
    "is_table_caption_spillover",
    "looks_author_contribution",
    "looks_author_stub",
    "looks_caption_stub",
    "looks_page_stub",
    "looks_table_heavy",
    "numeric_token_count",
    "spillover_issues",
    "word_like_count",
]
