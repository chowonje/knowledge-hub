"""Shared sanitization and P0 classification helpers.

This module keeps raw-sensitive handling centralized so web/learning paths do not
re-implement policy-related redaction rules.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge_hub.learning.models import EvidencePointer

P0_PATTERNS = (
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    # Phone-like patterns. Avoid broad digit-hyphen matches that can classify ISO dates as P0.
    re.compile(r"\b(?:\+?\d{1,3}[-\s]?)?(?:\(?\d{2,4}\)?[-\s]?)\d{3,4}[-\s]?\d{4}\b"),
    re.compile(r"\b\d{13,19}\b"),
    re.compile(r"(?i)\b(password|passwd|api[_-]?key|access[_-]?token|refresh[_-]?token|secret(?:[_-]?key)?|주민등록번호|계좌번호)\b"),
)

P3_PUBLIC_HINT_PATTERNS = (
    re.compile(r"(?i)\bhttps?://"),
    re.compile(r"(?i)\barxiv\b"),
    re.compile(r"(?i)\bdoi\b"),
    re.compile(r"(?i)\bwikipedia\b"),
    re.compile(r"(?i)\bgithub\b"),
    re.compile(r"(?i)\bopen source\b"),
)

P1_STRUCTURED_HINT_PATTERNS = (
    re.compile(r"^\s*[\{\[]"),  # JSON-like payload
    re.compile(r"(?m)^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*.+$"),  # key: value lines
    re.compile(r"(?i)\b(id|uuid|arxiv_id|paper_id|source_id|entity_id|claim_id)\b"),
    re.compile(r"(?i)\b(status|score|confidence|count|timestamp|created_at|updated_at)\b"),
)


def detect_p0(text: str | None) -> bool:
    """Return True when a text fragment should be treated as P0 raw data."""
    if not text:
        return False
    target = str(text)
    return any(pattern.search(target) for pattern in P0_PATTERNS)


def classify_text_level(text: str | None) -> str:
    """Classify a text fragment into P0/P1/P2/P3.

    Classification policy:
    - P0: raw sensitive tokens detected.
    - P3: clearly public references (URL/arXiv/DOI/open-source style) with short non-sensitive text.
    - P1: structured fact-like payloads (JSON, key-value logs, IDs/metrics).
    - P2: default summary/free-form text.
    """
    if detect_p0(text):
        return "P0"

    raw = str(text or "").strip()
    if not raw:
        return "P3"

    lowered = raw.lower()
    if len(raw) <= 400 and any(pattern.search(raw) for pattern in P3_PUBLIC_HINT_PATTERNS):
        return "P3"
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return "P3"

    if any(pattern.search(raw) for pattern in P1_STRUCTURED_HINT_PATTERNS):
        return "P1"

    return "P2"


def classify_payload_level(records: list[str]) -> str:
    """Classify a collection of fragments by most-sensitive level."""
    level_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    best_level = "P3"
    best_rank = level_rank[best_level]
    for item in records:
        if not item:
            continue
        level = classify_text_level(item)
        rank = level_rank.get(level, 2)
        if rank < best_rank:
            best_rank = rank
            best_level = level
            if best_level == "P0":
                break
    return best_level


def classify_payload(records: list[str]) -> bool:
    """Backward compatible helper: return True when any P0 is present."""
    return classify_payload_level(records) == "P0"


def redact_p0(text: str | None) -> str:
    """Redact obvious P0 tokens inside a free-form string.

    Note: this is a lightweight guard for logs and diagnostics. It is not a
    cryptographic redaction mechanism.
    """
    if not text:
        return ""
    output = str(text)
    for pattern in P0_PATTERNS:
        output = pattern.sub("[REDACTED]", output)
    return output


def redact_payload(value: Any) -> Any:
    """Recursively sanitize plain values used for external-facing payloads."""
    if isinstance(value, str):
        return redact_p0(value)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            result[str(key)] = redact_payload(nested)
        return result
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_payload(item) for item in value)
    return value


def pointer_from_snippet(
    raw: str,
    fallback_path: str,
) -> "EvidencePointer":
    """Parse a free-form pointer expression into EvidencePointer fields.

    This parser keeps only pointer metadata and strips any raw snippet text from
    writeback, so local notes can be referenced without persisting sensitive
    sentence fragments.
    """
    raw = (raw or "").strip()
    path = fallback_path
    heading = ""
    block_id = ""

    # Supports: key=value;key=value style inputs.
    if "=" in raw and ";" in raw:
        pairs: dict[str, str] = {}
        for part in raw.split(";"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            pairs[key.strip().lower()] = value.strip()
        path = pairs.get("path", path)
        heading = pairs.get("heading", "")
        block_id = pairs.get("block_id", pairs.get("blockid", ""))

    snippet_seed = raw or path
    snippet_hash = hashlib.sha256(snippet_seed.encode("utf-8")).hexdigest()[:16]
    from knowledge_hub.learning.models import EvidencePointer

    return EvidencePointer(
        type="note",
        path=path,
        heading=heading,
        block_id=block_id,
        snippet_hash=snippet_hash,
    )
