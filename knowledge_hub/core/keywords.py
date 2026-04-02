"""Shared keyword extraction helpers."""

from __future__ import annotations

import re
from collections import Counter

STOP_KOREAN = {
    "하는",
    "있는",
    "있다",
    "있고",
    "않고",
    "없이",
    "또는",
    "그리고",
    "에서",
    "대한",
    "에 대한",
    "같은",
    "및",
    "또한",
    "하지만",
    "뿐",
    "수",
    "등",
    "의",
    "을",
    "를",
    "에",
    "과",
    "와",
    "로",
    "으로",
    "으로써",
    "또",
    "뿐만",
    "때문에",
    "않은",
    "없는",
    "이",
    "그",
    "그녀",
    "본",
    "있습니다",
    "있을",
    "있으며",
    "있는지",
    "있음",
}

STOP_ENGLISH = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "were",
    "have",
    "has",
    "are",
    "was",
    "which",
    "their",
    "when",
    "then",
    "there",
    "about",
    "used",
    "using",
    "based",
    "models",
    "model",
    "learn",
    "learning",
    "dataset",
    "datasets",
    "results",
    "method",
    "methods",
    "paper",
    "research",
    "analysis",
    "different",
    "new",
    "more",
    "less",
    "one",
    "two",
    "three",
    "also",
    "first",
    "four",
    "five",
    "use",
    "show",
    "shown",
    "shows",
}


def _clean_text(text: str) -> str:
    body = (text or "").replace("\r\n", "\n")
    if body.startswith("#"):
        lines = body.splitlines()
        body = "\n".join(lines[4:]) if len(lines) > 4 else body
    body = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", body)
    body = re.sub(r"`[^`]*`", " ", body)
    body = re.sub(r"[\W_]+", " ", body)
    return body


def extract_keywords_from_text(text: str, max_keywords: int = 12) -> list[str]:
    """Extract lightweight multilingual keywords from text."""
    body = _clean_text(text)
    if not body.strip():
        return []

    en_tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", body)
        if len(token) >= 3 and token.lower() not in STOP_ENGLISH
    ]
    ko_tokens = [
        token
        for token in re.findall(r"[가-힣]{2,}", body)
        if token not in STOP_KOREAN and len(token) >= 2
    ]

    phrases: list[str] = []
    for match in re.findall(
        r"[A-Za-z][A-Za-z0-9\-\+]+(?:\s+[A-Za-z][A-Za-z0-9\-\+]+){0,2}",
        body,
    ):
        raw = match.strip()
        lowered = raw.lower()
        if len(raw.split()) >= 2 and lowered not in STOP_ENGLISH:
            phrases.append(lowered)

    counter: Counter[str] = Counter()
    for token in [*en_tokens, *ko_tokens, *phrases]:
        counter[token] += len(token)

    return [token for token, _ in counter.most_common(max(1, int(max_keywords)))]
