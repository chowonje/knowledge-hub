"""Quality-first preprocessing for web crawl documents.

This module keeps heuristics deterministic and local-first:
- canonical URL normalization (for dedupe/idempotency)
- content cleaning (boilerplate/noise reduction)
- quality scoring + approval decision
- batch-level dedupe and sample gate evaluation
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from knowledge_hub.web.crawl4ai_adapter import CrawlDocument


_TRACKING_QUERY_PREFIXES = (
    "utm_",
    "fbclid",
    "gclid",
    "mc_",
    "ref",
    "igshid",
)

_AI_SIGNAL_TERMS = {
    "ai",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural",
    "transformer",
    "llm",
    "language model",
    "rag",
    "retrieval",
    "embedding",
    "inference",
    "fine-tuning",
    "agent",
    "alignment",
    "safety",
}

_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*(cookie|privacy policy|all rights reserved)\b", re.IGNORECASE),
    re.compile(r"^\s*(sign in|log in|subscribe|newsletter)\b", re.IGNORECASE),
    re.compile(r"^\s*(share this|follow us|back to top)\b", re.IGNORECASE),
    re.compile(r"^\s*(advertisement|sponsored)\b", re.IGNORECASE),
]


def canonicalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    scheme = (parts.scheme or "https").lower()
    netloc = (parts.netloc or "").lower()
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
        if not path:
            path = "/"

    query_pairs = parse_qsl(parts.query, keep_blank_values=False)
    filtered: list[tuple[str, str]] = []
    for key, value in query_pairs:
        lowered = key.lower()
        if any(lowered.startswith(prefix) for prefix in _TRACKING_QUERY_PREFIXES):
            continue
        filtered.append((key, value))
    filtered.sort(key=lambda item: (item[0], item[1]))
    query = urlencode(filtered, doseq=True)

    return urlunsplit((scheme, netloc, path, query, ""))


def clean_web_text(text: str) -> str:
    lines = re.split(r"\r?\n", text or "")
    cleaned_lines: list[str] = []
    for line in lines:
        candidate = re.sub(r"\s+", " ", (line or "").strip())
        if not candidate:
            continue
        if len(candidate) <= 2:
            continue
        if any(pattern.search(candidate) for pattern in _BOILERPLATE_PATTERNS):
            continue
        cleaned_lines.append(candidate)
    cleaned = "\n\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9가-힣][A-Za-z0-9가-힣\-_/]*", text.lower())


@dataclass
class QualityAssessment:
    score: float
    approved: bool
    min_tokens: int
    token_count: int
    unique_ratio: float
    ai_signal_hits: int
    noise_ratio: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "approved": self.approved,
            "minTokens": self.min_tokens,
            "tokenCount": self.token_count,
            "uniqueRatio": self.unique_ratio,
            "aiSignalHits": self.ai_signal_hits,
            "noiseRatio": self.noise_ratio,
            "reasons": list(self.reasons),
        }


@dataclass
class QualityDoc:
    doc: CrawlDocument
    canonical_url: str
    cleaned_content: str
    content_hash: str
    assessment: QualityAssessment


@dataclass
class QualityBatchResult:
    items: list[QualityDoc]
    duplicates: list[dict[str, str]]
    rejected: int
    approved: int

    @property
    def evaluated(self) -> int:
        return len(self.items)


def assess_quality(
    text: str,
    *,
    threshold: float,
    min_tokens: int,
) -> QualityAssessment:
    cleaned = clean_web_text(text)
    tokens = _tokenize(cleaned)
    token_count = len(tokens)
    unique_ratio = (len(set(tokens)) / token_count) if token_count else 0.0

    lower_text = cleaned.lower()
    ai_signal_hits = sum(1 for term in _AI_SIGNAL_TERMS if term in lower_text)

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    boilerplate_hits = 0
    for line in lines:
        if any(pattern.search(line) for pattern in _BOILERPLATE_PATTERNS):
            boilerplate_hits += 1
    noise_ratio = (boilerplate_hits / max(1, len(lines))) if lines else 0.0

    length_score = min(1.0, token_count / 500.0)
    diversity_score = max(0.0, min(1.0, (unique_ratio - 0.2) / 0.5))
    ai_score = min(1.0, ai_signal_hits / 6.0)
    noise_score = 1.0 - min(1.0, noise_ratio * 1.5)

    score = (
        0.35 * length_score
        + 0.25 * diversity_score
        + 0.25 * ai_score
        + 0.15 * noise_score
    )
    score = round(max(0.0, min(1.0, score)), 4)

    reasons: list[str] = []
    if token_count < min_tokens:
        reasons.append("too_short")
    if ai_signal_hits == 0:
        reasons.append("no_ai_signal")
    if noise_ratio > 0.35:
        reasons.append("high_noise")
    if unique_ratio < 0.28:
        reasons.append("low_diversity")

    approved = bool(token_count >= min_tokens and score >= threshold)
    if not approved and not reasons:
        reasons.append("low_quality_score")

    return QualityAssessment(
        score=score,
        approved=approved,
        min_tokens=int(min_tokens),
        token_count=token_count,
        unique_ratio=round(unique_ratio, 4),
        ai_signal_hits=ai_signal_hits,
        noise_ratio=round(noise_ratio, 4),
        reasons=reasons,
    )


def evaluate_batch(
    docs: list[CrawlDocument],
    *,
    threshold: float = 0.62,
    min_tokens: int = 80,
) -> QualityBatchResult:
    items: list[QualityDoc] = []
    duplicates: list[dict[str, str]] = []
    rejected = 0
    approved = 0

    seen_url: dict[str, str] = {}
    seen_hash: dict[str, str] = {}

    for doc in docs:
        canonical = canonicalize_url(doc.url) or (doc.url or "").strip()
        cleaned_content = clean_web_text(doc.content)
        content_hash = hashlib.sha1(cleaned_content.encode("utf-8")).hexdigest()

        if canonical in seen_url:
            duplicates.append(
                {
                    "url": doc.url,
                    "error": f"duplicate canonical url (kept {seen_url[canonical]})",
                }
            )
            continue
        if content_hash in seen_hash:
            duplicates.append(
                {
                    "url": doc.url,
                    "error": f"duplicate content (same as {seen_hash[content_hash]})",
                }
            )
            continue

        assessment = assess_quality(
            cleaned_content,
            threshold=threshold,
            min_tokens=min_tokens,
        )
        if assessment.approved:
            approved += 1
        else:
            rejected += 1

        items.append(
            QualityDoc(
                doc=doc,
                canonical_url=canonical,
                cleaned_content=cleaned_content,
                content_hash=content_hash,
                assessment=assessment,
            )
        )
        seen_url[canonical] = canonical
        seen_hash[content_hash] = canonical

    return QualityBatchResult(
        items=items,
        duplicates=duplicates,
        rejected=rejected,
        approved=approved,
    )


def evaluate_sample_gate(
    batch: QualityBatchResult,
    *,
    sample_size: int = 12,
    min_pass_rate: float = 0.7,
    strict_score_floor: float = 0.55,
) -> dict[str, Any]:
    total = len(batch.items)
    if total == 0:
        return {
            "enabled": True,
            "sampleSize": 0,
            "passRate": 1.0,
            "minPassRate": min_pass_rate,
            "allowed": False,
            "failedItems": 0,
            "reason": "empty_batch",
        }

    bounded_size = max(1, min(int(sample_size), total))
    sorted_items = sorted(batch.items, key=lambda item: item.assessment.score)
    sample = sorted_items[:bounded_size]

    failed = 0
    for item in sample:
        score = item.assessment.score
        if not item.assessment.approved or score < float(strict_score_floor):
            failed += 1

    pass_rate = 1.0 - (failed / max(1, bounded_size))
    allowed = pass_rate >= float(min_pass_rate)
    return {
        "enabled": True,
        "sampleSize": bounded_size,
        "passRate": round(pass_rate, 4),
        "minPassRate": round(float(min_pass_rate), 4),
        "allowed": bool(allowed),
        "failedItems": failed,
        "reason": "" if allowed else "sample_gate_failed",
    }
