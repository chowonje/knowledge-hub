"""Candidate scoring helpers for Korean note materialization."""

from __future__ import annotations

from datetime import datetime, timezone
from difflib import SequenceMatcher
import re
from typing import Any

from knowledge_hub.notes.models import KoNoteQuality, KoNoteRemediation, KoNoteReview


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))


def normalize_count(value: int | float, baseline: int | float) -> float:
    baseline_value = max(1.0, float(baseline or 1.0))
    return clamp01(float(value or 0.0) / baseline_value)


def text_similarity(left: str, right: str) -> float:
    a = str(left or "").strip().lower()
    b = str(right or "").strip().lower()
    if not a or not b:
        return 0.0
    return clamp01(SequenceMatcher(None, a, b).ratio())


def compute_source_novelty(
    *,
    title: str,
    source_url: str,
    existing_items: list[dict[str, Any]],
) -> float:
    source_token = str(source_url or "").strip()
    highest_similarity = 0.0
    for item in existing_items:
        urls = item.get("source_urls_json") or []
        if source_token and source_token in urls:
            return 0.0
        title_score = max(
            text_similarity(title, str(item.get("title_en", ""))),
            text_similarity(title, str(item.get("title_ko", ""))),
        )
        highest_similarity = max(highest_similarity, title_score)
    return clamp01(1.0 - highest_similarity)


def compute_ontology_density(
    *,
    entity_count: int,
    relation_count: int,
    claim_count: int,
    token_count: int,
) -> float:
    token_norm = max(1.0, float(token_count or 0) / 180.0)
    raw = (float(entity_count or 0) + float(relation_count or 0) * 1.5 + float(claim_count or 0) * 2.0) / token_norm
    return clamp01(raw / 5.0)


def compute_evidence_quality(
    *,
    title: str,
    content_text: str,
    metadata: dict[str, Any] | None = None,
) -> float:
    paragraphs = [part.strip() for part in str(content_text or "").split("\n\n") if part.strip()]
    meta = metadata or {}
    score = 0.0
    if str(title or "").strip():
        score += 0.25
    length = len(str(content_text or "").strip())
    if length >= 600:
        score += 0.35
    elif length >= 250:
        score += 0.20
    if len(paragraphs) >= 3:
        score += 0.20
    elif len(paragraphs) >= 2:
        score += 0.10
    completeness_fields = ("url", "domain", "record_id", "fetched_at", "quality_score")
    completeness = sum(1 for field in completeness_fields if str(meta.get(field, "")).strip())
    score += normalize_count(completeness, len(completeness_fields)) * 0.20
    return clamp01(score)


def compute_source_score(
    *,
    quality_score: float,
    ontology_density: float,
    novelty: float,
    domain_trust: float,
    evidence_quality: float,
) -> float:
    return clamp01(
        0.35 * clamp01(quality_score)
        + 0.25 * clamp01(ontology_density)
        + 0.20 * clamp01(novelty)
        + 0.10 * clamp01(domain_trust)
        + 0.10 * clamp01(evidence_quality)
    )


def compute_concept_score(
    *,
    support_doc_count_norm: float,
    evidence_diversity: float,
    relation_degree: float,
    avg_confidence: float,
) -> float:
    return clamp01(
        0.35 * clamp01(support_doc_count_norm)
        + 0.30 * clamp01(evidence_diversity)
        + 0.20 * clamp01(relation_degree)
        + 0.15 * clamp01(avg_confidence)
    )


def translation_level_for_score(score: float, *, key_excerpt_threshold: float = 0.82) -> str:
    return "T2" if float(score or 0.0) >= float(key_excerpt_threshold or 0.82) else "T1"


CONCEPT_QUALITY_VERSION = "concept-quality-v1"
SOURCE_QUALITY_VERSION = "source-quality-v1"
KO_NOTE_REVIEW_VERSION = "ko-note-review-v1"
CONCEPT_REMEDIATION_SECTIONS = (
    "summary",
    "summary_line",
    "core_summary",
    "why_it_matters",
    "relation_lines",
    "claim_lines",
    "support_lines",
    "related_sources",
    "related_concepts",
    "key_excerpts_ko",
)
SOURCE_REMEDIATION_SECTIONS = (
    "summary",
    "summary_line",
    "document_type",
    "thesis",
    "top_claims",
    "contributions",
    "methodology",
    "results_or_findings",
    "insights",
    "limitations",
    "core_concepts",
    "key_excerpts_ko",
    "related_concepts",
    "sources",
)
CONCEPT_BANNED_PHRASES = (
    "state-of-the-art",
    "모든 태스크에서",
    "혁신적으로",
    "최초로",
    "완벽하게 해결",
)
_CONCEPT_PLACEHOLDERS = {
    "핵심 요약 없음",
    "핵심 정의 없음",
    "관련 개념 없음",
    "근거 문서 없음",
    "관련 소스 없음",
    "자동 추출 관계 없음",
    "대표 주장 없음",
}
_SOURCE_PLACEHOLDERS = {
    "핵심 요약 없음",
    "원문 URL 없음",
    "관련 개념 없음",
    "핵심 용어 없음",
    "발췌 근거 없음",
    "핵심 발췌 요약 없음",
}
_SOURCE_GENERIC_HINTS = (
    "함께 읽어야",
    "활용할 수 있습니다",
    "출발점으로 볼 수 있습니다",
    "추가 검토가 필요합니다",
    "추가 근거가 제한적입니다",
    "원문 기준으로 확인해야 합니다",
)
_DUPLICATE_SIMILARITY_THRESHOLD = 0.94
_SECTION_REMEDIATION_ESCALATION_THRESHOLD = 2

_CONCEPT_LINE_MINIMUMS = {
    "why_it_matters": 3,
    "relation_lines": 2,
    "claim_lines": 2,
    "support_lines": 2,
    "related_sources": 2,
    "related_concepts": 2,
    "key_excerpts_ko": 2,
}
_SOURCE_LINE_MINIMUMS = {
    "top_claims": 2,
    "contributions": 3,
    "methodology": 3,
    "results_or_findings": 3,
    "insights": 3,
    "limitations": 2,
    "core_concepts": 2,
    "key_excerpts_ko": 2,
    "related_concepts": 2,
    "representative_sources": 1,
    "sources": 1,
}


def _clean_markdown_line(line: str) -> str:
    token = str(line or "").strip()
    token = re.sub(r"^[\-\*\u2022]+\s*", "", token).strip()
    return token


def _is_placeholder_line(line: str) -> bool:
    token = _clean_markdown_line(line)
    return token in (_CONCEPT_PLACEHOLDERS | _SOURCE_PLACEHOLDERS)


def _split_markdown_sections(markdown: str) -> dict[str, list[str]]:
    from knowledge_hub.notes.templates import split_frontmatter

    _frontmatter, body = split_frontmatter(markdown)
    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in str(body or "").splitlines():
        line = raw_line.rstrip()
        if line.startswith("#"):
            current = line.strip()
            sections.setdefault(current, [])
            continue
        if current:
            sections.setdefault(current, []).append(line)
    return sections


def _meaningful_lines(lines: list[str], *, bullets_only: bool = False) -> list[str]:
    result: list[str] = []
    for raw_line in lines:
        line = str(raw_line or "").strip()
        if not line or line.startswith("<!--"):
            continue
        if bullets_only and not line.startswith("- "):
            continue
        if _is_placeholder_line(line):
            continue
        cleaned = _clean_markdown_line(line) if bullets_only else line
        if cleaned:
            result.append(cleaned)
    return result


def _meaningful_source_lines(lines: list[str], *, bullets_only: bool = False) -> list[str]:
    result: list[str] = []
    for raw_line in lines:
        line = str(raw_line or "").strip()
        if not line or line.startswith("<!--"):
            continue
        if bullets_only and not line.startswith("- "):
            continue
        token = _clean_markdown_line(line) if bullets_only else line.strip()
        if not token or _is_placeholder_line(token):
            continue
        result.append(token)
    return result


def _source_low_signal_checks(values: list[str]) -> tuple[int, list[str]]:
    from knowledge_hub.notes.source_profile import (
        _GENERIC_CLAIM_PATTERNS,
        _GENERIC_OPENINGS,
        _LOW_SIGNAL_PATTERNS,
        _LOW_SIGNAL_SUBSTRINGS,
    )

    hits: list[str] = []
    for raw_value in values:
        token = _clean_markdown_line(raw_value)
        lowered = token.casefold()
        if not lowered:
            continue
        matched = False
        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _LOW_SIGNAL_PATTERNS):
            matched = True
        elif any(substring in token for substring in _LOW_SIGNAL_SUBSTRINGS):
            matched = True
        elif any(re.match(pattern, lowered, flags=re.IGNORECASE) for pattern in _GENERIC_CLAIM_PATTERNS):
            matched = True
        elif any(lowered.startswith(opening.casefold()) for opening in _GENERIC_OPENINGS) and len(lowered.split()) <= 12:
            matched = True
        elif any(hint.casefold() in lowered for hint in _SOURCE_GENERIC_HINTS):
            matched = True
        if matched:
            hits.append(token)
    return len(hits), hits


def _normalized_line_token(line: str) -> str:
    token = _clean_markdown_line(line)
    token = re.sub(r"\s+", " ", token).strip().casefold()
    return token


def _concept_line_is_weak(line: str) -> tuple[bool, bool]:
    token = _clean_markdown_line(line)
    if not token:
        return True, False
    if _is_placeholder_line(token):
        return True, True
    lowered = token.casefold()
    banned = any(phrase.casefold() in lowered for phrase in CONCEPT_BANNED_PHRASES)
    return bool(banned), False


def _source_line_is_weak(line: str) -> tuple[bool, bool, bool]:
    token = _clean_markdown_line(line)
    if not token:
        return True, False, True
    if _is_placeholder_line(token):
        return True, True, True
    low_signal_count, _examples = _source_low_signal_checks([token])
    return bool(low_signal_count), False, bool(low_signal_count)


def analyze_line_quality(lines: list[str], *, item_type: str) -> dict[str, Any]:
    placeholder_indexes: list[int] = []
    weak_indexes: list[int] = []
    low_signal_indexes: list[int] = []
    duplicate_indexes: list[int] = []
    seen_tokens: list[str] = []
    meaningful_count = 0

    for index, raw_line in enumerate(lines or []):
        token = _clean_markdown_line(raw_line)
        normalized = _normalized_line_token(raw_line)
        if normalized:
            meaningful_count += 1
        if str(item_type or "") == "source":
            is_weak, is_placeholder, is_low_signal = _source_line_is_weak(raw_line)
        else:
            is_weak, is_placeholder = _concept_line_is_weak(raw_line)
            is_low_signal = is_weak and not is_placeholder

        is_duplicate = False
        if normalized:
            if normalized in seen_tokens:
                is_duplicate = True
            else:
                for existing in seen_tokens:
                    if text_similarity(normalized, existing) >= _DUPLICATE_SIMILARITY_THRESHOLD:
                        is_duplicate = True
                        break
                seen_tokens.append(normalized)

        if is_placeholder:
            placeholder_indexes.append(index)
        if is_low_signal:
            low_signal_indexes.append(index)
        if is_duplicate:
            duplicate_indexes.append(index)
        if is_weak or is_duplicate:
            weak_indexes.append(index)

    return {
        "line_count": len(lines or []),
        "meaningful_line_count": meaningful_count,
        "weak_line_indexes": weak_indexes,
        "placeholder_line_indexes": placeholder_indexes,
        "low_signal_line_count": len(low_signal_indexes),
        "low_signal_line_indexes": low_signal_indexes,
        "duplicate_line_indexes": duplicate_indexes,
    }


def line_diagnostics_for_payload_field(
    *,
    item_type: str,
    field_name: str,
    values: list[str] | None,
) -> dict[str, Any]:
    diagnostics = analyze_line_quality(list(values or []), item_type=item_type)
    diagnostics["field"] = str(field_name or "")
    return diagnostics


def remediation_line_minimum(*, item_type: str, field_name: str) -> int:
    field = str(field_name or "").strip()
    if str(item_type or "") == "source":
        return int(_SOURCE_LINE_MINIMUMS.get(field, 1))
    return int(_CONCEPT_LINE_MINIMUMS.get(field, 1))


def should_replace_scalar(
    existing: str | None,
    generated: str | None,
    *,
    missing: bool = False,
) -> bool:
    if not str(generated or "").strip():
        return False
    existing_token = str(existing or "").strip()
    if missing:
        return True
    if not existing_token:
        return True
    if _is_placeholder_line(existing_token):
        return True
    return False


def merge_targeted_lines(
    existing: list[str] | None,
    generated: list[str] | None,
    *,
    item_type: str,
    weak_indexes: list[int] | None = None,
    min_count: int = 1,
    max_count: int | None = None,
) -> tuple[list[str], int, int]:
    existing_lines = [str(item) for item in (existing or [])]
    generated_lines = [str(item) for item in (generated or []) if str(item).strip()]
    diagnostics = analyze_line_quality(existing_lines, item_type=item_type)
    weak_set = {int(index) for index in (weak_indexes or diagnostics.get("weak_line_indexes") or [])}
    meaningful_indexes = [
        index
        for index, line in enumerate(existing_lines)
        if _normalized_line_token(line)
    ]
    if meaningful_indexes and weak_set.issuperset(set(meaningful_indexes)):
        merged = []
    else:
        merged = [
            line
            for index, line in enumerate(existing_lines)
            if index not in weak_set and _normalized_line_token(line)
        ]
    preserved_count = len(merged)
    patched_count = 0
    seen = {_normalized_line_token(line) for line in merged if _normalized_line_token(line)}
    limit = int(max_count) if max_count is not None else max(len(existing_lines), len(generated_lines), int(min_count or 1))
    for line in generated_lines:
        normalized = _normalized_line_token(line)
        if not normalized or normalized in seen:
            continue
        merged.append(line)
        seen.add(normalized)
        patched_count += 1
        if len(merged) >= max(int(min_count or 1), limit):
            break
    return merged[: max(1, limit)], patched_count, preserved_count


def _ordered_targets(targets: list[str], catalog: tuple[str, ...]) -> list[str]:
    seen = {str(item).strip() for item in targets if str(item).strip()}
    return [key for key in catalog if key in seen]


def _append_ordered_targets(targets: list[str], *candidates: str) -> None:
    seen = {str(item).strip() for item in targets if str(item).strip()}
    for candidate in candidates:
        token = str(candidate or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        targets.append(token)


def _trim_target_list(targets: list[str], catalog: tuple[str, ...], *, maximum: int) -> list[str]:
    ordered = _ordered_targets(targets, catalog)
    return ordered[: max(1, int(maximum or 1))]


def _section_target_catalog(item_type: str) -> tuple[str, ...]:
    return CONCEPT_REMEDIATION_SECTIONS if str(item_type or "") == "concept" else SOURCE_REMEDIATION_SECTIONS


def _source_layout_title_map(document_type: str) -> dict[str, str]:
    from knowledge_hub.notes.templates import _source_layout

    layout = _source_layout(str(document_type or "method_paper"))
    mapping: dict[str, str] = {}
    for section in layout["sections"]:
        keys = tuple(section.get("keys") or ())
        canonical = ""
        for key in keys:
            token = str(key or "").strip()
            if not token:
                continue
            canonical = "results_or_findings" if token == "key_results" else token
            break
        if canonical:
            mapping[str(section.get("title") or "")] = canonical
    return mapping


def build_note_remediation_targets(
    *,
    item_type: str,
    quality: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> list[str]:
    checks = dict((quality or {}).get("checks") or {})
    missing_sections = [str(item).strip() for item in ((quality or {}).get("missing_sections") or []) if str(item).strip()]
    source_payload = dict(payload or {})
    targets: list[str] = []

    if str(item_type or "") == "concept":
        missing_map = {
            "summary": "summary",
            "summary_line": "summary_line",
            "core_summary": "core_summary",
            "why_it_matters": "why_it_matters",
            "relation_lines": "relation_lines",
            "claim_lines": "claim_lines",
            "support_lines": "support_lines",
            "related_sources": "related_sources",
            "related_concepts": "related_concepts",
        }
        for key in missing_sections:
            mapped = missing_map.get(key)
            if mapped:
                _append_ordered_targets(targets, mapped)
        # Prefer evidence/graph gaps first; only widen into descriptive sections when little else is flagged.
        if int(checks.get("support_bullets") or 0) < 2:
            _append_ordered_targets(targets, "support_lines")
        if int(checks.get("source_bullets") or 0) < 2:
            _append_ordered_targets(targets, "related_sources")
        if int(checks.get("related_bullets") or 0) < 2:
            _append_ordered_targets(targets, "related_concepts")
        if len(targets) < 2 and int(checks.get("claim_bullets") or 0) < 2:
            _append_ordered_targets(targets, "claim_lines")
        if len(targets) < 2 and int(checks.get("why_it_matters_bullets") or 0) < 2:
            _append_ordered_targets(targets, "why_it_matters")
        if len(targets) < 2 and int(checks.get("relation_bullets") or 0) < 2:
            _append_ordered_targets(targets, "relation_lines")
        if not str(source_payload.get("summary_line_ko") or "").strip():
            _append_ordered_targets(targets, "summary_line")
        if not str(source_payload.get("core_summary") or source_payload.get("summary_ko") or "").strip():
            _append_ordered_targets(targets, "core_summary")
        if not [str(item).strip() for item in (source_payload.get("key_excerpts_ko") or []) if str(item).strip()]:
            _append_ordered_targets(targets, "key_excerpts_ko")
        return _trim_target_list(targets, CONCEPT_REMEDIATION_SECTIONS, maximum=4)

    document_type = str(checks.get("document_type") or source_payload.get("document_type") or "method_paper")
    title_map = _source_layout_title_map(document_type)
    missing_map = {
        "summary": "summary",
        "summary_line": "summary_line",
        "document_type": "document_type",
        "top_claims": "top_claims",
        "limitations": "limitations",
        "core_concepts": "core_concepts",
        "evidence_excerpts": "key_excerpts_ko",
        "related_concepts": "related_concepts",
        "sources": "sources",
    }
    for key in missing_sections:
        mapped = missing_map.get(key) or title_map.get(key)
        if mapped:
            _append_ordered_targets(targets, mapped)
    if int(checks.get("claim_bullets") or 0) < 1:
        _append_ordered_targets(targets, "top_claims")
    if int(checks.get("limitation_bullets") or 0) < 1:
        _append_ordered_targets(targets, "limitations")
    if int(checks.get("key_term_bullets") or 0) < 1:
        _append_ordered_targets(targets, "core_concepts")
    if int(checks.get("excerpt_block_count") or 0) < 1:
        _append_ordered_targets(targets, "key_excerpts_ko")
    if int(checks.get("related_bullets") or 0) < 1:
        _append_ordered_targets(targets, "related_concepts")
    if not bool(checks.get("metadata_source_present")) or int(checks.get("source_bullets") or 0) < 1:
        _append_ordered_targets(targets, "sources")
    if int(checks.get("low_signal_bullet_count") or 0) > 0:
        _append_ordered_targets(targets, "top_claims")
        weakest_type_section: tuple[int, str] | None = None
        for title, mapped in title_map.items():
            count = int(checks.get(f"type_section::{title}") or 0)
            if count >= 2:
                continue
            candidate = (count, mapped)
            if weakest_type_section is None or candidate < weakest_type_section:
                weakest_type_section = candidate
        if weakest_type_section is not None:
            _append_ordered_targets(targets, weakest_type_section[1])
    if not str(source_payload.get("thesis") or "").strip():
        _append_ordered_targets(targets, "thesis")
    if not str(source_payload.get("summary_line_ko") or "").strip():
        _append_ordered_targets(targets, "summary_line")
    if not str(source_payload.get("core_summary") or source_payload.get("summary_ko") or "").strip():
        _append_ordered_targets(targets, "summary")
    return _trim_target_list(targets, SOURCE_REMEDIATION_SECTIONS, maximum=5)


def remediation_preserve_sections(*, item_type: str, target_sections: list[str]) -> list[str]:
    catalog = _section_target_catalog(item_type)
    target_set = {str(item).strip() for item in (target_sections or []) if str(item).strip()}
    return [key for key in catalog if key not in target_set]


def score_concept_note_markdown(markdown: str, concept_type: str) -> dict[str, Any]:
    from knowledge_hub.notes.templates import _concept_layout

    layout = _concept_layout(str(concept_type or "generic"))
    sections = _split_markdown_sections(markdown)

    checks = {
        "summary_section_present": "## 요약" in sections,
        "summary_line_present": False,
        "core_section_present": False,
        "why_it_matters_present": False,
        "relation_section_present": False,
        "claim_section_present": False,
        "support_section_present": False,
        "source_section_present": False,
        "related_section_present": False,
        "why_it_matters_bullets": 0,
        "relation_bullets": 0,
        "claim_bullets": 0,
        "support_bullets": 0,
        "source_bullets": 0,
        "related_bullets": 0,
        "excerpt_bullets": 0,
        "banned_phrase_count": 0,
        "concept_type": str(concept_type or "generic"),
        "core_heading": layout["core_heading"],
        "relation_heading": layout["relation_heading"],
        "claim_heading": layout["claim_heading"],
        "support_heading": layout["support_heading"],
        "source_heading": layout["source_heading"],
        "related_heading": layout["related_heading"],
    }
    missing_sections: list[str] = []

    summary_line_lines = _meaningful_lines(sections.get("### 한줄 요약", []))
    core_lines = _meaningful_lines(sections.get(layout["core_heading"], []))
    why_lines = _meaningful_lines(sections.get("### 왜 중요한가", []), bullets_only=True)
    relation_lines = _meaningful_lines(sections.get(layout["relation_heading"], []), bullets_only=True)
    claim_lines = _meaningful_lines(sections.get(layout["claim_heading"], []), bullets_only=True)
    support_lines = _meaningful_lines(sections.get(layout["support_heading"], []), bullets_only=True)
    source_lines = _meaningful_lines(sections.get(layout["source_heading"], []), bullets_only=True)
    related_lines = _meaningful_lines(sections.get(layout["related_heading"], []), bullets_only=True)
    excerpt_lines = [line for line in claim_lines if "근거" in line or "evidence" in line.casefold()]
    line_diagnostics = {
        "why_it_matters": analyze_line_quality(why_lines, item_type="concept"),
        "relation_lines": analyze_line_quality(relation_lines, item_type="concept"),
        "claim_lines": analyze_line_quality(claim_lines, item_type="concept"),
        "support_lines": analyze_line_quality(support_lines, item_type="concept"),
        "related_sources": analyze_line_quality(source_lines, item_type="concept"),
        "related_concepts": analyze_line_quality(related_lines, item_type="concept"),
        "key_excerpts_ko": analyze_line_quality(excerpt_lines, item_type="concept"),
    }

    checks["summary_line_present"] = bool(summary_line_lines)
    checks["core_section_present"] = bool(core_lines)
    checks["why_it_matters_present"] = bool(why_lines)
    checks["relation_section_present"] = bool(relation_lines)
    checks["claim_section_present"] = bool(claim_lines)
    checks["support_section_present"] = bool(support_lines)
    checks["source_section_present"] = bool(source_lines)
    checks["related_section_present"] = bool(related_lines)
    checks["why_it_matters_bullets"] = len(why_lines)
    checks["relation_bullets"] = len(relation_lines)
    checks["claim_bullets"] = len(claim_lines)
    checks["support_bullets"] = len(support_lines)
    checks["source_bullets"] = len(source_lines)
    checks["related_bullets"] = len(related_lines)
    checks["excerpt_bullets"] = len(excerpt_lines)
    checks["line_diagnostics"] = line_diagnostics

    if not checks["summary_section_present"]:
        missing_sections.append("summary")
    if not checks["summary_line_present"]:
        missing_sections.append("summary_line")
    if not checks["core_section_present"]:
        missing_sections.append("core_summary")
    if not checks["why_it_matters_present"]:
        missing_sections.append("why_it_matters")
    if not checks["relation_section_present"]:
        missing_sections.append("relation_lines")
    if not checks["claim_section_present"]:
        missing_sections.append("claim_lines")
    if not checks["support_section_present"]:
        missing_sections.append("support_lines")
    if not checks["source_section_present"]:
        missing_sections.append("related_sources")
    if not checks["related_section_present"]:
        missing_sections.append("related_concepts")

    lowered = str(markdown or "").casefold()
    banned_phrase_hits = [phrase for phrase in CONCEPT_BANNED_PHRASES if phrase.casefold() in lowered]
    checks["banned_phrase_count"] = len(banned_phrase_hits)

    score = sum(
        int(value)
        for value in (
            checks["summary_section_present"],
            checks["summary_line_present"],
            checks["core_section_present"],
            checks["why_it_matters_present"],
            checks["relation_section_present"],
            checks["claim_section_present"],
            checks["support_section_present"],
            checks["source_section_present"],
            checks["related_section_present"],
            not banned_phrase_hits,
        )
    )
    max_score = 10
    severe_failures = sum(
        int(value)
        for value in (
            not checks["summary_line_present"],
            not checks["core_section_present"],
            not checks["support_section_present"],
            not checks["source_section_present"],
            not checks["related_section_present"],
            len(banned_phrase_hits) >= 2,
        )
    )
    if severe_failures >= 3 or score <= 4:
        flag = "reject"
    elif missing_sections or banned_phrase_hits or score < 8:
        flag = "needs_review"
    else:
        flag = "ok"

    return {
        "score": int(score),
        "max_score": int(max_score),
        "flag": str(flag),
        "missing_sections": missing_sections,
        "banned_phrase_hits": banned_phrase_hits,
        "checks": checks,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "version": CONCEPT_QUALITY_VERSION,
    }


def concept_quality_warnings(quality: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    flag = str((quality or {}).get("flag") or "").strip()
    missing_sections = [str(item).strip() for item in ((quality or {}).get("missing_sections") or []) if str(item).strip()]
    banned_hits = [str(item).strip() for item in ((quality or {}).get("banned_phrase_hits") or []) if str(item).strip()]
    if flag in {"needs_review", "reject"}:
        summary = f"concept-note-quality:{flag}"
        if missing_sections:
            summary += f" missing={','.join(missing_sections)}"
        warnings.append(summary)
    if banned_hits:
        warnings.append(f"concept-note-quality:banned-phrases={','.join(banned_hits)}")
    return warnings


def score_source_note_markdown(markdown: str, document_type: str) -> dict[str, Any]:
    from knowledge_hub.notes.templates import _source_layout, split_frontmatter

    layout = _source_layout(str(document_type or "method_paper"))
    sections = _split_markdown_sections(markdown)
    _frontmatter, body = split_frontmatter(markdown)

    summary_lines = _meaningful_source_lines(sections.get("### 한줄 요약", []))
    claim_lines = _meaningful_source_lines(sections.get(str(layout["claim_heading"]), []), bullets_only=True)
    limitation_lines = _meaningful_source_lines(sections.get(str(layout["limitation_heading"]), []), bullets_only=True)
    key_term_lines = _meaningful_source_lines(sections.get("### 핵심 용어", []), bullets_only=True)
    related_lines = _meaningful_source_lines(sections.get("## 관련 개념", []), bullets_only=True)
    source_lines = _meaningful_source_lines(sections.get("## 출처", []), bullets_only=True)

    type_section_titles = [str(section["title"]) for section in layout["sections"]]
    type_section_bullets: dict[str, list[str]] = {
        title: _meaningful_source_lines(sections.get(title, []), bullets_only=True)
        for title in type_section_titles
    }
    excerpt_summary_count = len(
        [
            item
            for item in re.findall(r"^- 한국어 요약:\s*(.+)$", str(body or ""), flags=re.MULTILINE)
            if item.strip() and not _is_placeholder_line(item)
        ]
    )
    excerpt_quote_count = len(
        [item for item in re.findall(r"^> (.+)$", str(body or ""), flags=re.MULTILINE) if item.strip() != "원문 발췌"]
    )
    excerpt_blocks = min(excerpt_summary_count, excerpt_quote_count)

    all_scored_bullets = [
        *summary_lines,
        *claim_lines,
        *limitation_lines,
        *key_term_lines,
        *related_lines,
        *[line for bullets in type_section_bullets.values() for line in bullets],
    ]
    low_signal_count, low_signal_examples = _source_low_signal_checks(all_scored_bullets)
    low_signal_examples = low_signal_examples[:4]
    low_signal_density_ok = low_signal_count <= max(1, len(all_scored_bullets) // 4) if all_scored_bullets else False
    excerpt_summary_lines = [
        item
        for item in re.findall(r"^- 한국어 요약:\s*(.+)$", str(body or ""), flags=re.MULTILINE)
        if item.strip() and not _is_placeholder_line(item)
    ]
    representative_source_lines = [line for line in source_lines if "http" not in line]
    line_diagnostics = {
        "top_claims": analyze_line_quality(claim_lines, item_type="source"),
        "limitations": analyze_line_quality(limitation_lines, item_type="source"),
        "core_concepts": analyze_line_quality(key_term_lines, item_type="source"),
        "key_excerpts_ko": analyze_line_quality(excerpt_summary_lines, item_type="source"),
        "related_concepts": analyze_line_quality(related_lines, item_type="source"),
        "representative_sources": analyze_line_quality(representative_source_lines, item_type="source"),
    }
    for title, bullets in type_section_bullets.items():
        mapped = _source_layout_title_map(document_type).get(title)
        if mapped:
            line_diagnostics[mapped] = analyze_line_quality(bullets, item_type="source")

    checks = {
        "summary_section_present": "## 요약" in sections,
        "summary_line_present": bool(summary_lines),
        "document_type_marker_present": bool(
            re.search(r"^- 문서 타입: `[^`]+`$", str(body or ""), flags=re.MULTILINE)
        ),
        "document_type_matches_layout": bool(
            re.search(rf"^- 문서 타입: `{re.escape(str(document_type or 'method_paper'))}`$", str(body or ""), flags=re.MULTILINE)
        ),
        "claim_section_present": bool(claim_lines),
        "limitation_section_present": bool(limitation_lines),
        "key_terms_present": bool(key_term_lines),
        "excerpt_section_present": str(layout["excerpt_title"]) in sections and excerpt_blocks > 0,
        "related_section_present": bool(related_lines),
        "source_section_present": bool(source_lines),
        "metadata_source_present": any("http" in line for line in source_lines),
        "low_signal_density_ok": bool(low_signal_density_ok),
        "low_signal_bullet_count": int(low_signal_count),
        "excerpt_block_count": int(excerpt_blocks),
        "summary_line_count": len(summary_lines),
        "claim_bullets": len(claim_lines),
        "limitation_bullets": len(limitation_lines),
        "key_term_bullets": len(key_term_lines),
        "related_bullets": len(related_lines),
        "source_bullets": len(source_lines),
        "document_type": str(document_type or "method_paper"),
        "claim_heading": str(layout["claim_heading"]),
        "limitation_heading": str(layout["limitation_heading"]),
        "excerpt_title": str(layout["excerpt_title"]),
        "type_sections_present": {},
        "line_diagnostics": line_diagnostics,
    }
    missing_sections: list[str] = []

    if not checks["summary_section_present"]:
        missing_sections.append("summary")
    if not checks["summary_line_present"]:
        missing_sections.append("summary_line")
    if not checks["document_type_marker_present"]:
        missing_sections.append("document_type")
    if not checks["claim_section_present"]:
        missing_sections.append("top_claims")
    for title in type_section_titles:
        present = bool(type_section_bullets[title])
        checks["type_sections_present"][title] = present
        checks[f"type_section::{title}"] = len(type_section_bullets[title])
        if not present:
            missing_sections.append(title)
    if not checks["limitation_section_present"]:
        missing_sections.append("limitations")
    if not checks["key_terms_present"]:
        missing_sections.append("core_concepts")
    if not checks["excerpt_section_present"]:
        missing_sections.append("evidence_excerpts")
    if not checks["related_section_present"]:
        missing_sections.append("related_concepts")
    if not checks["source_section_present"]:
        missing_sections.append("sources")

    score = sum(
        int(value)
        for value in (
            checks["summary_section_present"],
            checks["summary_line_present"],
            checks["document_type_marker_present"],
            checks["claim_section_present"],
            *checks["type_sections_present"].values(),
            checks["limitation_section_present"],
            checks["key_terms_present"],
            checks["excerpt_section_present"],
            checks["related_section_present"],
            checks["source_section_present"],
            checks["metadata_source_present"],
            checks["low_signal_density_ok"],
        )
    )
    max_score = 12 + len(type_section_titles)
    severe_failures = sum(
        int(value)
        for value in (
            not checks["summary_line_present"],
            not checks["claim_section_present"],
            not checks["excerpt_section_present"],
            not checks["source_section_present"],
            not checks["key_terms_present"],
            sum(int(not present) for present in checks["type_sections_present"].values()) >= 2,
            low_signal_count >= max(2, len(all_scored_bullets) // 3) if all_scored_bullets else True,
        )
    )
    if severe_failures >= 3 or score <= max_score // 2:
        flag = "reject"
    elif missing_sections or not low_signal_density_ok or low_signal_count > 0 or score < (max_score - 1):
        flag = "needs_review"
    else:
        flag = "ok"

    return {
        "score": int(score),
        "max_score": int(max_score),
        "flag": str(flag),
        "missing_sections": missing_sections,
        "checks": {
            **checks,
            "low_signal_examples": low_signal_examples,
        },
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "version": SOURCE_QUALITY_VERSION,
    }


def source_quality_warnings(quality: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    flag = str((quality or {}).get("flag") or "").strip()
    missing_sections = [str(item).strip() for item in ((quality or {}).get("missing_sections") or []) if str(item).strip()]
    checks = dict((quality or {}).get("checks") or {})
    low_signal_count = int(checks.get("low_signal_bullet_count") or 0)
    low_signal_examples = [str(item).strip() for item in (checks.get("low_signal_examples") or []) if str(item).strip()]
    if flag in {"needs_review", "reject"}:
        summary = f"source-note-quality:{flag}"
        if missing_sections:
            summary += f" missing={','.join(missing_sections)}"
        warnings.append(summary)
    if low_signal_count:
        preview = ",".join(low_signal_examples[:2]) if low_signal_examples else str(low_signal_count)
        warnings.append(f"source-note-quality:low-signal={preview}")
    return warnings


def _review_flag(quality: dict[str, Any]) -> str:
    flag = str((quality or {}).get("flag") or "").strip()
    return flag if flag in {"ok", "needs_review", "reject", "unscored"} else "unscored"


def _concept_review_messages(quality: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    from knowledge_hub.notes.templates import _concept_layout

    checks = dict((quality or {}).get("checks") or {})
    missing_sections = [str(item).strip() for item in ((quality or {}).get("missing_sections") or []) if str(item).strip()]
    banned_hits = [str(item).strip() for item in ((quality or {}).get("banned_phrase_hits") or []) if str(item).strip()]
    concept_type = str(checks.get("concept_type") or "generic")
    layout = _concept_layout(concept_type)
    reasons: list[str] = []
    patch_hints: list[str] = []
    actions: list[str] = []

    missing_map = {
        "summary": ("요약 섹션이 비어 있습니다.", "`## 요약` 아래에 개념 전체를 2~4문장으로 다시 정리하세요."),
        "summary_line": ("한 줄 요약이 없습니다.", "`### 한줄 요약`에 상위범주와 핵심 기능이 드러나는 1문장을 추가하세요."),
        "core_summary": ("핵심 설명이 부족합니다.", f"`{layout['core_heading']}` 아래에 정의·작동 방식·경계를 더 구체적으로 채우세요."),
        "why_it_matters": ("중요성 설명이 약합니다.", "`### 왜 중요한가`에 검색/학습/설계 맥락의 의미를 근거 기반 bullet로 보강하세요."),
        "relation_lines": ("개념 관계 정보가 부족합니다.", f"`{layout['relation_heading']}`에 인접 개념과의 관계를 최소 2개 이상 정리하세요."),
        "claim_lines": ("대표 근거 bullet이 부족합니다.", f"`{layout['claim_heading']}`에 claim 또는 evidence 기반 bullet을 보강하세요."),
        "support_lines": ("대표 문서 근거가 비어 있습니다.", f"`{layout['support_heading']}`에 대표 supporting document를 추가하세요."),
        "related_sources": ("관련 소스가 비어 있습니다.", f"`{layout['source_heading']}`에 URL 또는 source note 기준 근거를 연결하세요."),
        "related_concepts": ("연결 개념이 부족합니다.", f"`{layout['related_heading']}`에 상위/병렬/혼동 개념을 더 연결하세요."),
    }
    for key in missing_sections:
        if key in missing_map:
            reason, hint = missing_map[key]
            reasons.append(reason)
            patch_hints.append(hint)

    if banned_hits:
        reasons.append(f"과장되거나 금지된 표현이 포함되어 있습니다: {', '.join(banned_hits[:3])}")
        patch_hints.append("금지 표현을 제거하고, 연도·근거·조건이 있는 서술로 다시 쓰세요.")

    if checks.get("support_bullets", 0) < 2:
        reasons.append("근거 문서 수가 적어 개념 설명의 신뢰성이 약합니다.")
        patch_hints.append("supporting document를 2개 이상 확보하거나 대표 문서를 더 명시적으로 연결하세요.")

    if checks.get("related_bullets", 0) < 2:
        reasons.append("인접 개념 구분이 약해 재사용성이 떨어집니다.")
        patch_hints.append("혼동되는 개념과의 차이를 `연결 개념`에 추가하세요.")

    if reasons:
        actions.append("staging note를 열어 patch hint 기준으로 요약/근거/관계 섹션을 보강하세요.")
    if banned_hits:
        actions.append("generic/과장 표현을 제거한 뒤 다시 enrich하거나 수동 수정하세요.")
    if "support_lines" in missing_sections or "related_sources" in missing_sections:
        actions.append("관련 source note 또는 대표 문서를 추가로 연결하세요.")
    return reasons[:6], patch_hints[:6], list(dict.fromkeys(actions))[:4]


def _source_review_messages(quality: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    from knowledge_hub.notes.templates import _source_layout

    checks = dict((quality or {}).get("checks") or {})
    missing_sections = [str(item).strip() for item in ((quality or {}).get("missing_sections") or []) if str(item).strip()]
    document_type = str(checks.get("document_type") or "method_paper")
    layout = _source_layout(document_type)
    low_signal_count = int(checks.get("low_signal_bullet_count") or 0)
    low_signal_examples = [str(item).strip() for item in (checks.get("low_signal_examples") or []) if str(item).strip()]
    reasons: list[str] = []
    patch_hints: list[str] = []
    actions: list[str] = []

    missing_map = {
        "summary": ("요약 섹션이 비어 있습니다.", "`## 요약` 아래에 문서의 논지와 활용 맥락을 2~4문장으로 보강하세요."),
        "summary_line": ("한 줄 요약이 없습니다.", "`### 한줄 요약`에 문서가 실제로 주장하는 핵심을 1문장으로 다시 쓰세요."),
        "document_type": ("문서 타입 마커가 없습니다.", f"`- 문서 타입: `{document_type}`` 줄이 유지되도록 payload의 `document_type`를 확인하세요."),
        "top_claims": ("핵심 주장 섹션이 약합니다.", f"`{layout['claim_heading']}`에 claim/evidence 기반 bullet을 최소 2개 이상 보강하세요."),
        "limitations": ("한계 섹션이 부족합니다.", f"`{layout['limitation_heading']}`에 scope/risk/coverage 한계를 추가하세요."),
        "core_concepts": ("핵심 용어가 부족합니다.", "`### 핵심 용어`에 실제 entity 또는 개념 용어를 더 채우세요."),
        "evidence_excerpts": ("근거 발췌가 부족합니다.", f"`{layout['excerpt_title']}` 아래에 대표 발췌와 한국어 요약을 보강하세요."),
        "related_concepts": ("관련 개념 연결이 약합니다.", "`## 관련 개념`에 같이 읽어야 할 concept를 더 연결하세요."),
        "sources": ("출처 섹션이 비어 있습니다.", "`## 출처`에 원문 URL과 representative source를 명시하세요."),
    }
    for key in missing_sections:
        if key in missing_map:
            reason, hint = missing_map[key]
            reasons.append(reason)
            patch_hints.append(hint)
        elif key.startswith("### "):
            reasons.append(f"{key} 섹션이 비어 있습니다.")
            patch_hints.append(f"`{key}` 아래에 문서 타입에 맞는 근거 bullet을 보강하세요.")

    if low_signal_count:
        reasons.append(f"generic 또는 저신호 bullet이 {low_signal_count}개 감지되었습니다.")
        example_suffix = f" 예: {', '.join(low_signal_examples[:2])}" if low_signal_examples else ""
        patch_hints.append(f"\"This paper presents...\" 같은 generic bullet을 제거하고 claim/evidence 중심으로 다시 쓰세요.{example_suffix}")

    if not bool(checks.get("metadata_source_present")):
        reasons.append("원문 URL 또는 대표 source 연결이 약합니다.")
        patch_hints.append("source_url과 representative_sources를 확인해 출처를 명시적으로 연결하세요.")

    if reasons:
        actions.append("staging source note를 열어 generic bullet을 evidence 기반 문장으로 교체하세요.")
    if low_signal_count:
        actions.append("LLM 재실행 전 대표 claim과 limitation을 수동으로 정리한 뒤 enrich를 다시 시도하세요.")
    if "evidence_excerpts" in missing_sections or not bool(checks.get("excerpt_section_present")):
        actions.append("대표 excerpt 2개 이상을 확보한 뒤 source note를 다시 보강하세요.")
    return reasons[:6], patch_hints[:6], list(dict.fromkeys(actions))[:4]


def build_note_review_payload(
    *,
    item_type: str,
    quality: dict[str, Any],
    payload: dict[str, Any] | None = None,
    existing_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quality_meta = KoNoteQuality.from_payload(quality)
    existing = KoNoteReview.from_payload(existing_review)
    quality_payload = quality_meta.to_payload()
    flag = _review_flag(quality_payload)
    checks = dict(quality_meta.checks or {})
    line_diagnostics = dict(checks.get("line_diagnostics") or {})
    target_sections = build_note_remediation_targets(
        item_type=item_type,
        quality=quality_payload,
        payload=payload,
    )
    if str(item_type or "") == "concept":
        reasons, patch_hints, actions = _concept_review_messages(quality_payload)
    else:
        reasons, patch_hints, actions = _source_review_messages(quality_payload)
    queued = flag in {"needs_review", "reject", "unscored"}
    if flag == "ok":
        reasons = []
        patch_hints = []
        actions = []
    remediation = existing.remediation
    remediation.target_sections = list(target_sections)
    field_diagnostics: dict[str, Any] = {}
    field_aliases = {"sources": "representative_sources"}
    for section in target_sections:
        section_key = str(section or "").strip()
        if not section_key:
            continue
        diagnostic = dict(
            line_diagnostics.get(section_key)
            or line_diagnostics.get(field_aliases.get(section_key, ""))
            or {}
        )
        if diagnostic:
            field_diagnostics[section_key] = diagnostic
    if field_diagnostics:
        remediation.field_diagnostics = field_diagnostics
    existing.queue = bool(queued)
    existing.reasons = list(reasons)
    existing.patch_hints = list(patch_hints)
    existing.suggested_actions = list(actions)
    existing.remediation = remediation
    existing.generated_at = datetime.now(timezone.utc).isoformat()
    existing.version = KO_NOTE_REVIEW_VERSION
    return existing.to_payload()


def update_review_remediation(
    existing_review: dict[str, Any] | None,
    *,
    run_id: str,
    status: str,
    warnings: list[str] | None = None,
    before_quality: dict[str, Any] | None = None,
    after_quality: dict[str, Any] | None = None,
    strategy: str | None = None,
    target_sections: list[str] | None = None,
    patched_sections: list[str] | None = None,
    preserved_sections_count: int | None = None,
    patched_line_count: int | None = None,
    preserved_line_count: int | None = None,
    recommended_strategy: str | None = None,
) -> dict[str, Any]:
    review = KoNoteReview.from_payload(existing_review)
    remediation = review.remediation
    before = KoNoteQuality.from_payload(before_quality)
    after = KoNoteQuality.from_payload(after_quality)
    before_score = float(before.score or 0.0)
    before_max = max(1.0, float(before.max_score or 1.0))
    after_score = float(after.score or 0.0)
    after_max = max(1.0, float(after.max_score or 1.0))
    before_norm = before_score / before_max
    after_norm = after_score / after_max
    strategy_token = str(strategy or remediation.strategy or "")
    improved = bool(
        (str(after.flag or "") != str(before.flag or ""))
        or (after_norm > before_norm + 1e-9)
    )
    no_improvement_count = int(remediation.section_no_improvement_count or 0)
    if strategy_token == "section":
        if improved:
            no_improvement_count = 0
        elif str(status or "") in {"remediated", "failed"}:
            no_improvement_count += 1
    if recommended_strategy is None:
        recommended_strategy = (
            "full"
            if strategy_token == "section" and no_improvement_count >= _SECTION_REMEDIATION_ESCALATION_THRESHOLD
            else ""
        )
    remediation.attempt_count = int(remediation.attempt_count or 0) + 1
    remediation.last_attempt_at = datetime.now(timezone.utc).isoformat()
    remediation.last_attempt_status = str(status or "")
    remediation.last_attempt_warnings = [str(item).strip() for item in (warnings or []) if str(item).strip()][:8]
    remediation.last_attempt_quality_flag = str(after.flag or "unscored")
    remediation.last_attempt_score = round(after_score, 6)
    remediation.last_improved = improved
    remediation.last_run_id = str(run_id or "")
    remediation.strategy = strategy_token
    remediation.target_sections = [str(item).strip() for item in (target_sections or remediation.target_sections or []) if str(item).strip()]
    remediation.patched_sections = [str(item).strip() for item in (patched_sections or []) if str(item).strip()]
    remediation.preserved_sections_count = int(preserved_sections_count or remediation.preserved_sections_count or 0)
    remediation.last_patched_line_count = int(patched_line_count or 0)
    remediation.last_preserved_line_count = int(preserved_line_count or 0)
    remediation.section_no_improvement_count = int(no_improvement_count)
    remediation.recommended_strategy = str(recommended_strategy or "")
    review.remediation = remediation
    return review.to_payload()
