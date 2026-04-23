"""Shared payload composition for source and concept notes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from knowledge_hub.notes.source_profile import (
    DOCUMENT_TYPES,
    extract_thesis,
    filter_low_signal_evidence,
    infer_document_type,
    representative_sources,
    synthesize_evidence_sections,
)


_SOURCE_SECTION_LIMITS = {
    "top_claims": 4,
    "contributions": 5,
    "methodology": 5,
    "results_or_findings": 5,
    "limitations": 4,
    "insights": 5,
}

CONCEPT_TYPES = (
    "model",
    "method",
    "metric",
    "task",
    "benchmark",
    "safety_risk",
    "generic",
)

_CONCEPT_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "model": (
        "model",
        "architecture",
        "transformer",
        "attention",
        "encoder",
        "decoder",
        "network",
        "backbone",
        "agent architecture",
        "아키텍처",
        "모델",
        "구조",
    ),
    "method": (
        "method",
        "approach",
        "algorithm",
        "pipeline",
        "workflow",
        "retrieval",
        "generation",
        "prompting",
        "fine-tuning",
        "finetuning",
        "optimizer",
        "방법",
        "절차",
        "파이프라인",
        "알고리즘",
        "최적화",
    ),
    "metric": (
        "metric",
        "metrics",
        "score",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "faithfulness",
        "calibration",
        "error rate",
        "지표",
        "점수",
        "정확도",
        "재현율",
    ),
    "task": (
        "task",
        "question answering",
        "qa",
        "summarization",
        "translation",
        "classification",
        "planning",
        "reasoning task",
        "problem setting",
        "objective",
        "태스크",
        "문제",
        "질의응답",
        "요약",
        "번역",
        "분류",
        "계획",
    ),
    "benchmark": (
        "benchmark",
        "leaderboard",
        "dataset",
        "evaluation suite",
        "arena",
        "test set",
        "corpus",
        "벤치마크",
        "리더보드",
        "데이터셋",
        "평가 셋",
    ),
    "safety_risk": (
        "risk",
        "safety",
        "alignment",
        "harm",
        "jailbreak",
        "failure mode",
        "mitigation",
        "red team",
        "preparedness",
        "guardrail",
        "위험",
        "안전",
        "정렬",
        "완화",
        "실패 모드",
        "가드레일",
    ),
}

_CONCEPT_TYPE_BODY_HINTS: dict[str, tuple[str, ...]] = {
    "model": ("uses", "requires", "component", "mechanism", "system"),
    "method": ("step", "procedure", "apply", "train", "retrieve", "generate"),
    "metric": ("measure", "evaluate", "compare", "threshold", "interpret"),
    "task": ("solve", "goal", "input", "output", "setting"),
    "benchmark": ("leaderboard", "evaluate", "coverage", "annotation", "split"),
    "safety_risk": ("mitigate", "monitor", "governance", "failure", "deployment"),
}


def _clean_line(value: Any) -> str:
    token = str(value or "").strip()
    if token.startswith("- "):
        token = token[2:].strip()
    return " ".join(token.split())


def _joined_text(*parts: Any) -> str:
    values: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, (list, tuple, set)):
            values.extend(str(item or "") for item in part)
        else:
            values.append(str(part or ""))
    return " ".join(" ".join(value.split()) for value in values if str(value or "").strip()).strip()


def _clean_list(values: Any, *, limit: int | None = None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        candidates = [values]
    else:
        candidates = list(values)
    result: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        token = _clean_line(raw)
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
        if limit is not None and len(result) >= limit:
            break
    return result


def _as_bullets(values: Sequence[Any], *, limit: int | None = None) -> list[str]:
    return [f"- {item}" for item in _clean_list(values, limit=limit)]


def _extract_domain(source_url: str, domain_hint: str) -> str:
    if str(domain_hint or "").strip():
        return str(domain_hint or "").strip().lower()
    try:
        return (urlsplit(str(source_url or "")).netloc or "").strip().lower()
    except Exception:
        return ""


def _source_metadata_lines(payload: Mapping[str, Any], *, source_url: str, domain: str, explicit: Sequence[str] | None) -> list[str]:
    metadata_lines = _clean_list(explicit, limit=8) if explicit is not None else _clean_list(payload.get("metadata_lines"), limit=8)
    if metadata_lines:
        return metadata_lines
    generated: list[str] = []
    if source_url:
        generated.append(f"url={source_url}")
    if domain:
        generated.append(f"domain={domain}")
    published = str(payload.get("published_or_fetched") or payload.get("fetched_at") or "").strip()
    if published:
        generated.append(f"published_or_fetched={published}")
    quality = payload.get("quality_score")
    if quality is not None and str(quality).strip():
        generated.append(f"quality={quality}")
    return generated


def _select_section(
    payload: Mapping[str, Any],
    key: str,
    *,
    synthesized: list[str],
    fallbacks: Mapping[str, Sequence[str]] | None,
    limit: int,
) -> list[str]:
    existing = filter_low_signal_evidence(_clean_list(payload.get(key), limit=limit), limit=limit)
    if existing:
        return existing
    fallback_values = filter_low_signal_evidence(_clean_list((fallbacks or {}).get(key), limit=limit), limit=limit)
    if synthesized:
        return filter_low_signal_evidence(synthesized, limit=limit) or synthesized[:limit]
    if fallback_values:
        return fallback_values[:limit]
    return []


def compose_source_note_payload(
    payload: Mapping[str, Any],
    *,
    content_text: str = "",
    entity_names: Sequence[str] | None = None,
    metadata_lines: Sequence[str] | None = None,
    section_fallbacks: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    composed = dict(payload)
    title_en = str(composed.get("title_en") or "").strip()
    title_ko = str(composed.get("title_ko") or title_en).strip()
    source_url = str(composed.get("source_url") or "").strip()
    domain = _extract_domain(source_url, str(composed.get("domain") or ""))
    metadata = _source_metadata_lines(composed, source_url=source_url, domain=domain, explicit=metadata_lines)
    visible_entity_names = _clean_list(
        entity_names
        or composed.get("entity_names")
        or composed.get("core_concepts")
        or composed.get("entity_lines")
        or composed.get("related_concepts"),
        limit=12,
    )
    visible_relation_lines = _clean_list(composed.get("relation_lines"), limit=12)
    visible_related_concepts = _clean_list(composed.get("related_concepts") or visible_entity_names, limit=12)
    claim_lines = _clean_list(composed.get("claim_lines"), limit=8)
    key_excerpts_en = _clean_list(
        composed.get("source_key_excerpts_en") or composed.get("key_excerpts_en"),
        limit=4,
    )
    filtered_key_excerpts_en = filter_low_signal_evidence(key_excerpts_en, limit=4) or key_excerpts_en[:4]
    effective_content = str(
        content_text
        or composed.get("source_content_text")
        or composed.get("original_content_text")
        or composed.get("content_text")
        or composed.get("core_summary")
        or composed.get("summary_ko")
        or ""
    )
    document_type = str(composed.get("document_type") or "").strip()
    if document_type not in DOCUMENT_TYPES:
        document_type = infer_document_type(
            title=title_en,
            source_url=source_url,
            domain=domain,
            content_text=effective_content,
            key_excerpts=filtered_key_excerpts_en,
            metadata_lines=metadata,
            claim_lines=claim_lines,
        )
    thesis = str(composed.get("thesis") or "").strip()
    if not thesis:
        thesis = extract_thesis(
            title=title_en,
            document_type=document_type,
            content_text=effective_content,
            claim_lines=claim_lines,
            key_excerpts=filtered_key_excerpts_en,
        )
    synthesized = synthesize_evidence_sections(
        document_type=document_type,
        title=title_en,
        thesis=thesis,
        content_text=effective_content,
        entity_names=visible_entity_names,
        relation_lines=visible_relation_lines,
        claim_lines=claim_lines,
        related_concepts=visible_related_concepts,
        key_excerpts=filtered_key_excerpts_en,
        metadata_lines=metadata,
    )
    composed.update(
        {
            "title_en": title_en,
            "title_ko": title_ko,
            "core_summary": str(composed.get("core_summary") or composed.get("summary_ko") or "").strip(),
            "summary_ko": str(composed.get("summary_ko") or composed.get("core_summary") or "").strip(),
            "summary_line_ko": str(composed.get("summary_line_ko") or "").strip(),
            "source_url": source_url,
            "domain": domain,
            "metadata_lines": metadata,
            "entity_lines": _as_bullets(composed.get("entity_lines") or visible_entity_names, limit=12),
            "relation_lines": _as_bullets(composed.get("relation_lines"), limit=12),
            "claim_lines": _as_bullets(claim_lines, limit=8),
            "key_excerpts_ko": _as_bullets(composed.get("key_excerpts_ko"), limit=4),
            "key_excerpts_en": filtered_key_excerpts_en[:4],
            "related_concepts": _as_bullets(visible_related_concepts, limit=12) or ["- 관련 개념 없음"],
            "document_type": document_type,
            "thesis": thesis,
            "top_claims": _select_section(
                composed,
                "top_claims",
                synthesized=synthesized["top_claims"],
                fallbacks=section_fallbacks,
                limit=_SOURCE_SECTION_LIMITS["top_claims"],
            ),
            "core_concepts": _clean_list(
                composed.get("core_concepts") or synthesized["core_concepts"] or visible_entity_names,
                limit=8,
            ),
            "contributions": _select_section(
                composed,
                "contributions",
                synthesized=synthesized["contributions"],
                fallbacks=section_fallbacks,
                limit=_SOURCE_SECTION_LIMITS["contributions"],
            ),
            "methodology": _select_section(
                composed,
                "methodology",
                synthesized=synthesized["methodology"],
                fallbacks=section_fallbacks,
                limit=_SOURCE_SECTION_LIMITS["methodology"],
            ),
            "results_or_findings": _select_section(
                composed,
                "results_or_findings",
                synthesized=synthesized["results_or_findings"],
                fallbacks=section_fallbacks,
                limit=_SOURCE_SECTION_LIMITS["results_or_findings"],
            ),
            "limitations": _select_section(
                composed,
                "limitations",
                synthesized=synthesized["limitations"],
                fallbacks=section_fallbacks,
                limit=_SOURCE_SECTION_LIMITS["limitations"],
            ),
            "insights": _select_section(
                composed,
                "insights",
                synthesized=synthesized["insights"],
                fallbacks=section_fallbacks,
                limit=_SOURCE_SECTION_LIMITS["insights"],
            ),
            "representative_sources": _clean_list(
                composed.get("representative_sources")
                or representative_sources(
                    title=title_en,
                    source_url=source_url,
                    domain=domain,
                    metadata_lines=metadata,
                ),
                limit=5,
            ),
            "source_content_text": effective_content,
        }
    )
    return composed


def _default_why_it_matters(
    *,
    canonical_name: str,
    concept_type: str,
    support_doc_count: int,
    relation_lines: Sequence[str],
    claim_lines: Sequence[str],
    related_concepts: Sequence[str],
    support_lines: Sequence[str],
) -> list[str]:
    type_reason = {
        "model": f"{canonical_name}는 여러 시스템 설계에서 재사용되는 핵심 아키텍처/모델 개념입니다.",
        "method": f"{canonical_name}는 반복 가능한 절차와 적용 조건을 정리할 때 기준이 되는 방법 개념입니다.",
        "metric": f"{canonical_name}는 점수 자체보다 무엇을 어떻게 측정하는지 해석하게 만드는 기준 지표입니다.",
        "task": f"{canonical_name}는 시스템이 무엇을 풀어야 하는지 정의하는 문제 설정입니다.",
        "benchmark": f"{canonical_name}는 모델/시스템의 강점과 약점을 비교 가능하게 드러내는 평가 기준입니다.",
        "safety_risk": f"{canonical_name}는 실패 모드와 완화 조건을 같이 읽어야 하는 안전 위험 개념입니다.",
    }.get(concept_type, "")
    reasons = [
        type_reason,
        f"{canonical_name}는 {support_doc_count}개 근거 문서에서 반복적으로 확인된 핵심 개념입니다."
        if support_doc_count
        else "",
        f"{', '.join(_clean_list(related_concepts, limit=3))}와 직접 연결돼 후속 개념 노트를 읽는 기준점이 됩니다."
        if _clean_list(related_concepts, limit=3)
        else "",
        "주요 관계와 대표 근거를 함께 보면 개념의 적용 범위와 전제 조건을 빠르게 파악할 수 있습니다."
        if _clean_list(relation_lines, limit=1) or _clean_list(claim_lines, limit=1)
        else "",
        f"대표 근거 문서: {_clean_list(support_lines, limit=1)[0]}" if _clean_list(support_lines, limit=1) else "",
    ]
    return _clean_list(reasons, limit=4)


def infer_concept_type(
    *,
    title: str,
    summary_text: str,
    relation_lines: Sequence[str],
    claim_lines: Sequence[str],
    related_concepts: Sequence[str],
    support_lines: Sequence[str],
    evidence_lines: Sequence[str],
) -> tuple[str, float]:
    title_lower = _joined_text(title).casefold()
    body_lower = _joined_text(summary_text, relation_lines, claim_lines, related_concepts, support_lines, evidence_lines).casefold()
    scores: dict[str, float] = {}
    for concept_type in CONCEPT_TYPES:
        if concept_type == "generic":
            continue
        keywords = _CONCEPT_TYPE_KEYWORDS.get(concept_type, ())
        title_hits = sum(1 for keyword in keywords if keyword in title_lower)
        body_hits = sum(1 for keyword in keywords if keyword in body_lower)
        body_hint_hits = sum(1 for keyword in _CONCEPT_TYPE_BODY_HINTS.get(concept_type, ()) if keyword in body_lower)
        score = (title_hits * 2.4) + (min(body_hits, 4) * 1.1) + (min(body_hint_hits, 2) * 0.6)
        if concept_type == "benchmark" and any(token in body_lower for token in ("metric", "score", "leaderboard", "gap", "지표", "점수", "격차")):
            score += 0.9
        if concept_type == "metric" and any(token in body_lower for token in ("compare", "threshold", "faithfulness", "precision", "정확도", "비교")):
            score += 0.9
        if concept_type == "safety_risk" and any(token in body_lower for token in ("mitigation", "guardrail", "failure", "완화", "실패")):
            score += 1.1
        scores[concept_type] = score

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if not ranked:
        return "generic", 0.0
    concept_type, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_score < 3.0 or (top_score - second_score) < 1.0:
        return "generic", min(top_score / 6.0, 0.55)
    return concept_type, min(top_score / 6.0, 0.98)


def _default_concept_claims(
    *,
    claim_lines: Sequence[str],
    evidence_lines: Sequence[str],
    support_lines: Sequence[str],
) -> list[str]:
    explicit = _clean_list(claim_lines, limit=4)
    if explicit:
        return explicit
    evidence = _clean_list(evidence_lines, limit=4)
    if evidence:
        return evidence
    support = _clean_list(support_lines, limit=4)
    if support:
        return [f"{item}에서 개념의 사용 맥락이 확인됩니다." for item in support[:4]]
    return []


def compose_concept_note_payload(
    payload: Mapping[str, Any],
    *,
    aliases: Sequence[str] | None = None,
) -> dict[str, Any]:
    composed = dict(payload)
    title = str(composed.get("title") or "").strip()
    title_ko = str(composed.get("title_ko") or title).strip()
    relation_lines = _clean_list(composed.get("relation_lines"), limit=12)
    support_lines = _clean_list(composed.get("support_lines"), limit=10)
    related_sources = _clean_list(composed.get("related_sources") or support_lines, limit=8)
    related_concepts = _clean_list(composed.get("related_concepts"), limit=12)
    evidence_lines = _clean_list(composed.get("key_excerpts_ko"), limit=5)
    claim_lines = _default_concept_claims(
        claim_lines=composed.get("claim_lines") or [],
        evidence_lines=evidence_lines,
        support_lines=support_lines,
    )
    support_doc_count = int(composed.get("support_doc_count") or len(support_lines) or len(related_sources) or 0)
    summary_text = str(composed.get("core_summary") or composed.get("summary_ko") or "").strip()
    concept_type = str(composed.get("concept_type") or "").strip()
    concept_type_confidence = composed.get("concept_type_confidence")
    if concept_type not in CONCEPT_TYPES:
        concept_type, inferred_confidence = infer_concept_type(
            title=title,
            summary_text=summary_text,
            relation_lines=relation_lines,
            claim_lines=claim_lines,
            related_concepts=related_concepts,
            support_lines=support_lines or related_sources,
            evidence_lines=evidence_lines,
        )
        concept_type_confidence = inferred_confidence
    else:
        try:
            concept_type_confidence = float(concept_type_confidence)
        except (TypeError, ValueError):
            concept_type_confidence = 0.75 if concept_type != "generic" else 0.5
    why_it_matters = _clean_list(composed.get("why_it_matters"), limit=4) or _default_why_it_matters(
        canonical_name=title,
        concept_type=concept_type,
        support_doc_count=support_doc_count,
        relation_lines=relation_lines,
        claim_lines=claim_lines,
        related_concepts=related_concepts,
        support_lines=support_lines or related_sources,
    )
    composed.update(
        {
            "title": title,
            "title_ko": title_ko,
            "core_summary": summary_text,
            "summary_ko": str(composed.get("summary_ko") or composed.get("core_summary") or "").strip(),
            "summary_line_ko": str(composed.get("summary_line_ko") or "").strip(),
            "aliases": _clean_list(aliases or composed.get("aliases"), limit=12),
            "concept_type": concept_type,
            "concept_type_confidence": round(float(concept_type_confidence or 0.0), 3),
            "why_it_matters": why_it_matters,
            "relation_lines": _as_bullets(relation_lines, limit=12),
            "claim_lines": _as_bullets(claim_lines, limit=4),
            "support_lines": _as_bullets(support_lines or related_sources, limit=10),
            "key_excerpts_ko": _as_bullets(evidence_lines or claim_lines, limit=5),
            "key_excerpts_en": _clean_list(composed.get("key_excerpts_en"), limit=5),
            "related_sources": _as_bullets(related_sources or support_lines, limit=8),
            "related_concepts": _as_bullets(related_concepts, limit=12) or ["- 관련 개념 없음"],
            "support_doc_count": support_doc_count,
        }
    )
    return composed
