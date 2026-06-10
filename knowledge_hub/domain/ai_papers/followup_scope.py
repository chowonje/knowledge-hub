from __future__ import annotations

import re
from typing import Final


TRANSFORMER_CANONICAL_PAPER_ID: Final = "1706.03762"
TRANSFORMER_CANONICAL_TITLE: Final = "Attention Is All You Need"

_TRANSFORMER_PAPER_SCOPE_RE: Final = re.compile(
    r"\b(?:the\s+transformer|transformer\s+paper)\b",
    re.IGNORECASE,
)
_VISION_TRANSFORMER_SCOPE_RE: Final = re.compile(
    r"\b(?:vision\s+transformer|vit)\b",
    re.IGNORECASE,
)
_TRANSFORMER_FOLLOWUP_CUE_RE: Final = re.compile(
    r"\b(?:"
    r"remov(?:e|es|ed|ing)\s+recurrence|"
    r"recurrence|recurrent|"
    r"convolution(?:s|al)?|"
    r"self[-\s]?attention|"
    r"attention[-\s]?only|"
    r"paralleliz(?:e|es|ed|ing|able)|"
    r"sequence\s+transduction"
    r")\b",
    re.IGNORECASE,
)


def _clean_text(value: str | None) -> str:
    return " ".join(str(value or "").strip().split())


def canonical_followup_title_candidate(query: str) -> str:
    body = _clean_text(query)
    if not body:
        return ""
    if _VISION_TRANSFORMER_SCOPE_RE.search(body):
        return ""
    if not _TRANSFORMER_PAPER_SCOPE_RE.search(body):
        return ""
    if not _TRANSFORMER_FOLLOWUP_CUE_RE.search(body):
        return ""
    return TRANSFORMER_CANONICAL_TITLE


__all__ = [
    "TRANSFORMER_CANONICAL_PAPER_ID",
    "TRANSFORMER_CANONICAL_TITLE",
    "canonical_followup_title_candidate",
]
