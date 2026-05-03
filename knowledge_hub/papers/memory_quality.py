"""Quality heuristics for paper-memory cards and rebuild diagnostics."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

_LATEX_MARKERS = (
    "\\documentclass",
    "\\usepackage",
    "\\title{",
    "\\author{",
    "\\begin{document}",
    "\\maketitle",
    "\\section{",
    "\\subsection{",
)
_GENERIC_LIMITATION_MARKERS = (
    "limited information provided",
    "insufficient information",
    "limitations are not clearly stated",
    "the text does not provide",
    "not enough information",
    "details are sparse",
)
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣][0-9A-Za-z가-힣'_-]{2,}")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _tokenize(value: Any) -> set[str]:
    return {
        token
        for token in (match.group(0).casefold() for match in _TOKEN_RE.finditer(str(value or "")))
        if len(token) >= 3
    }


def contains_latex_markers(value: Any) -> bool:
    body = _clean_text(value).casefold()
    if not body:
        return False
    return any(marker in body for marker in _LATEX_MARKERS)


def is_generic_limitation(value: Any) -> bool:
    body = _clean_text(value).casefold()
    if not body:
        return False
    return any(marker in body for marker in _GENERIC_LIMITATION_MARKERS)


def title_overlap_ratio(title: Any, body: Any) -> float:
    title_tokens = _tokenize(title)
    body_tokens = _tokenize(body)
    if not title_tokens or not body_tokens:
        return 0.0
    return len(title_tokens & body_tokens) / max(1, len(title_tokens))


def evaluate_paper_memory_quality(
    *,
    title: Any,
    paper_core: Any,
    method_core: Any,
    evidence_core: Any,
    limitations: Any,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(diagnostics or {})
    sanitation = dict(metadata.get("textSanitation") or {})
    translated = dict(sanitation.get("translated") or {})
    raw = dict(sanitation.get("raw") or {})

    paper_core_text = _clean_text(paper_core)
    method_core_text = _clean_text(method_core)
    evidence_core_text = _clean_text(evidence_core)
    limitations_text = _clean_text(limitations)
    fallback_used = bool(metadata.get("fallbackUsed"))
    paper_core_has_latex = contains_latex_markers(paper_core_text)
    empty_method_core = not bool(method_core_text)
    empty_evidence_core = not bool(evidence_core_text)
    generic_limitation = is_generic_limitation(limitations_text)
    source_starts_latex = bool(translated.get("startsWithLatex")) or bool(raw.get("startsWithLatex"))
    weak_sanitized_content = bool(sanitation.get("weakContent"))
    semantic_mismatch_likely = bool(
        paper_core_text
        and title_overlap_ratio(title, paper_core_text) < 0.2
        and not paper_core_has_latex
        and len(_tokenize(paper_core_text)) >= 6
    )

    weak_reasons: list[str] = []
    if paper_core_has_latex:
        weak_reasons.append("paper_core_has_latex")
    if empty_method_core:
        weak_reasons.append("empty_method_core")
    if empty_evidence_core:
        weak_reasons.append("empty_evidence_core")
    if generic_limitation:
        weak_reasons.append("generic_limitation")
    if fallback_used:
        weak_reasons.append("fallback_used")

    auxiliary_reasons: list[str] = []
    if source_starts_latex:
        auxiliary_reasons.append("source_starts_latex")
    if weak_sanitized_content:
        auxiliary_reasons.append("weak_sanitized_content")
    if semantic_mismatch_likely:
        auxiliary_reasons.append("semantic_mismatch_likely")

    return {
        "weakCard": bool(weak_reasons),
        "needsReview": bool(weak_reasons or auxiliary_reasons),
        "weakReasons": weak_reasons,
        "auxiliaryReviewReasons": auxiliary_reasons,
        "paperCoreHasLatex": paper_core_has_latex,
        "emptyMethodCore": empty_method_core,
        "emptyEvidenceCore": empty_evidence_core,
        "genericLimitation": generic_limitation,
        "fallbackUsed": fallback_used,
        "sourceStartsLatex": source_starts_latex,
        "weakSanitizedContent": weak_sanitized_content,
        "semanticMismatchLikely": semantic_mismatch_likely,
        "titleOverlapRatio": round(title_overlap_ratio(title, paper_core_text), 4) if paper_core_text else 0.0,
    }


def summarize_quality_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {}
    reason_counts: Counter[str] = Counter()
    auxiliary_counts: Counter[str] = Counter()
    weak_card_count = 0
    needs_review_count = 0
    paper_core_has_latex_count = 0
    empty_method_count = 0
    empty_evidence_count = 0
    generic_limitation_count = 0
    fallback_count = 0
    source_starts_latex_count = 0
    weak_sanitized_content_count = 0
    semantic_mismatch_count = 0
    for report in reports:
        if bool(report.get("weakCard")):
            weak_card_count += 1
        if bool(report.get("needsReview")):
            needs_review_count += 1
        if bool(report.get("paperCoreHasLatex")):
            paper_core_has_latex_count += 1
        if bool(report.get("emptyMethodCore")):
            empty_method_count += 1
        if bool(report.get("emptyEvidenceCore")):
            empty_evidence_count += 1
        if bool(report.get("genericLimitation")):
            generic_limitation_count += 1
        if bool(report.get("fallbackUsed")):
            fallback_count += 1
        if bool(report.get("sourceStartsLatex")):
            source_starts_latex_count += 1
        if bool(report.get("weakSanitizedContent")):
            weak_sanitized_content_count += 1
        if bool(report.get("semanticMismatchLikely")):
            semantic_mismatch_count += 1
        reason_counts.update(str(item) for item in list(report.get("weakReasons") or []) if str(item).strip())
        auxiliary_counts.update(str(item) for item in list(report.get("auxiliaryReviewReasons") or []) if str(item).strip())
    total = len(reports)
    return {
        "weakCardCount": weak_card_count,
        "weakCardRate": round(weak_card_count / total, 4),
        "needsReviewCount": needs_review_count,
        "needsReviewRate": round(needs_review_count / total, 4),
        "paperCoreHasLatexCount": paper_core_has_latex_count,
        "emptyMethodCoreCount": empty_method_count,
        "emptyEvidenceCoreCount": empty_evidence_count,
        "genericLimitationCount": generic_limitation_count,
        "fallbackUsedCount": fallback_count,
        "sourceStartsLatexCount": source_starts_latex_count,
        "sanitationHitRate": round(source_starts_latex_count / total, 4),
        "weakSanitizedContentCount": weak_sanitized_content_count,
        "semanticMismatchLikelyCount": semantic_mismatch_count,
        "weakReasonCounts": dict(reason_counts),
        "auxiliaryReviewReasonCounts": dict(auxiliary_counts),
    }


__all__ = [
    "contains_latex_markers",
    "evaluate_paper_memory_quality",
    "is_generic_limitation",
    "summarize_quality_reports",
    "title_overlap_ratio",
]
