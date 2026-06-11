"""Page-1 identity gate for parsed-paper acceptance.

The 2026-06-11 parsed-store audit found 13 papers whose on-disk PDF was a
different document than the registered metadata, and no pipeline step ever
compared parsed content against that metadata — hashes were internally
consistent with the wrong bytes. This gate runs at parse acceptance and
checks two page-1 signals:

- arXiv watermark id vs the expected id (authoritative in both directions:
  a mismatch fails even when the title loosely matches; a match passes even
  when the registered title is stale).
- registered-title token containment in the page-1 text (decides when no
  watermark is available).

Cases with no checkable signal (placeholder titles, scanned pages, non-arXiv
sources without a watermark) stay ``inconclusive`` and are accepted, so the
gate only blocks positively-detected wrong documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .source_guard import _meaningful_title_tokens, _normalize_title
from .source_text import extract_pdf_text_excerpt

IDENTITY_GATE_SCHEMA_ID = "knowledge-hub.paper-identity-gate.v1"
IDENTITY_GATE_FAIL_REASON_PREFIX = "identity_gate_wrong_document"

_ARXIV_WATERMARK_RE = re.compile(r"arxiv:\s*(\d{4}\.\d{4,5})", re.IGNORECASE)
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
_PLACEHOLDER_TITLE_RE = re.compile(r"^arxiv \d{4} \d{4,5}(?: v\d+)?$")

_TITLE_PASS_RATIO = 0.6
_TITLE_FAIL_RATIO = 0.25
_MAX_TITLE_TOKENS = 8


class PaperIdentityGateError(RuntimeError):
    """Parse acceptance rejected: page-1 content does not match registered metadata."""

    def __init__(self, *, paper_id: str, reason: str, signals: dict[str, Any] | None = None):
        self.paper_id = str(paper_id)
        self.reason = str(reason)
        self.signals = dict(signals or {})
        super().__init__(f"{IDENTITY_GATE_FAIL_REASON_PREFIX}: {self.paper_id}: {self.reason}")


@dataclass
class IdentityGateResult:
    status: str  # "pass" | "fail" | "inconclusive"
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status != "fail"


def extract_first_page_text(pdf_path: str, *, max_chars: int = 8000) -> str:
    return extract_pdf_text_excerpt(str(pdf_path or ""), max_pages=1, max_chars=max_chars)


def check_page1_identity(
    *,
    page1_text: Any,
    registered_title: Any,
    expected_arxiv_id: Any = "",
) -> IdentityGateResult:
    text = str(page1_text or "")
    normalized_text = _normalize_title(text)
    title = str(registered_title or "").strip()
    expected = str(expected_arxiv_id or "").strip()
    expected_is_arxiv = bool(_ARXIV_ID_RE.match(expected))

    watermark_ids = list(dict.fromkeys(_ARXIV_WATERMARK_RE.findall(text)))
    signals: dict[str, Any] = {
        "schema": IDENTITY_GATE_SCHEMA_ID,
        "expectedArxivId": expected if expected_is_arxiv else "",
        "watermarkIds": watermark_ids,
        "watermarkMatch": None,
        "titleTokens": [],
        "titleTokensMatched": 0,
        "titleMatchRatio": None,
        "titlePlaceholder": False,
        "page1TextChars": len(text),
    }

    if not normalized_text:
        return IdentityGateResult("inconclusive", "page1_text_unavailable", signals)

    if expected_is_arxiv and watermark_ids:
        watermark_match = expected in watermark_ids
        signals["watermarkMatch"] = watermark_match
        if not watermark_match:
            return IdentityGateResult("fail", f"watermark_mismatch:{watermark_ids[0]}", signals)
        return IdentityGateResult("pass", "watermark_match", signals)

    normalized_title = _normalize_title(title)
    title_placeholder = (not normalized_title) or bool(_PLACEHOLDER_TITLE_RE.match(normalized_title))
    signals["titlePlaceholder"] = title_placeholder

    if not title_placeholder:
        tokens = _meaningful_title_tokens(title)[:_MAX_TITLE_TOKENS]
        signals["titleTokens"] = tokens
        if tokens:
            page1_words = set(normalized_text.split())
            matched = sum(1 for token in tokens if token in page1_words)
            ratio = matched / len(tokens)
            signals["titleTokensMatched"] = matched
            signals["titleMatchRatio"] = round(ratio, 3)
            if ratio >= _TITLE_PASS_RATIO:
                return IdentityGateResult("pass", "title_token_containment", signals)
            if ratio <= _TITLE_FAIL_RATIO:
                return IdentityGateResult(
                    "fail",
                    f"title_containment_too_low:{matched}/{len(tokens)}",
                    signals,
                )
            return IdentityGateResult("inconclusive", "title_containment_ambiguous", signals)

    return IdentityGateResult("inconclusive", "no_checkable_identity_signals", signals)


def enforce_parse_identity(
    *,
    paper_id: str,
    pdf_path: str,
    registered_title: str,
    expected_arxiv_id: str = "",
    page1_text: str | None = None,
) -> IdentityGateResult:
    """Run the page-1 identity check and raise on a positive wrong-document verdict.

    ``page1_text`` overrides PDF extraction (tests / callers that already hold
    the text). When ``expected_arxiv_id`` is omitted, an arXiv-shaped
    ``paper_id`` is used as the expected watermark id.
    """

    expected = str(expected_arxiv_id or "").strip()
    token = str(paper_id or "").strip()
    if not expected and _ARXIV_ID_RE.match(token):
        expected = token
    text = page1_text if page1_text is not None else extract_first_page_text(pdf_path)
    result = check_page1_identity(
        page1_text=text,
        registered_title=registered_title,
        expected_arxiv_id=expected,
    )
    if result.status == "fail":
        raise PaperIdentityGateError(paper_id=token, reason=result.reason, signals=result.signals)
    return result


__all__ = [
    "IDENTITY_GATE_FAIL_REASON_PREFIX",
    "IDENTITY_GATE_SCHEMA_ID",
    "IdentityGateResult",
    "PaperIdentityGateError",
    "check_page1_identity",
    "enforce_parse_identity",
    "extract_first_page_text",
]
