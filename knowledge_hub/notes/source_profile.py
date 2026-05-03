"""Heuristics for source-note profiling and evidence-pack synthesis."""

from __future__ import annotations

import re
from typing import Any


DOCUMENT_TYPES = (
    "survey_taxonomy",
    "benchmark",
    "system_card_safety_report",
    "method_paper",
    "blog_tutorial",
)

_LIMITATION_KEYWORDS = (
    "limit",
    "limitation",
    "constraint",
    "scope",
    "caveat",
    "trade-off",
    "tradeoff",
    "future work",
    "underperform",
    "fail",
    "fails",
    "failure",
    "risk",
    "mitigation",
    "coverage",
    "bias",
    "generaliz",
    "benchmark",
    "retrieval quality",
    "한계",
    "제약",
    "범위",
    "위험",
)

_LOW_SIGNAL_PATTERNS = (
    r"^view a pdf",
    r"^title:\s*$",
    r"^authors?:\s*$",
    r"^computer science\s*>",
    r"^\(cs\)$",
    r"^\[submitted on",
    r"^submitted on",
    r"^back to articles",
    r"^update on github",
    r"^follow$",
    r"^(url|domain|source|published(?:_at)?|published_or_fetched|updated|quality)\s*[:=]",
    r"^https?://",
    r"^doi:\s*",
)

_LOW_SIGNAL_SUBSTRINGS = (
    "문서의 핵심 기여를 자동 추출하지 못했습니다.",
    "문서의 접근 방식을 자동 추출하지 못했습니다.",
    "핵심 결과를 자동 추출하지 못했습니다.",
    "문서의 한계를 자동 추출하지 못했습니다.",
    "후속 개념 탐색을 위한 요약 소스로 활용할 가치가 있습니다.",
    "문서의 접근 방식은 본문 설명 흐름과 관계 추출 결과를 기준으로 재구성했습니다.",
    "정량 실험이 명시되지 않은 경우",
    "핵심 발췌 요약 없음",
    "핵심 용어 없음",
    "관련 개념 없음",
    "발췌 근거 없음",
)

_GENERIC_OPENINGS = (
    "this paper presents",
    "this paper introduces",
    "this paper describes",
    "this paper discusses",
    "this work presents",
    "this work introduces",
    "this work describes",
    "this work discusses",
    "the paper presents",
    "the paper introduces",
    "the paper describes",
    "the paper discusses",
    "this tutorial shows",
    "this tutorial explains",
    "this guide shows",
    "this guide explains",
    "이 문서는",
    "이 글은",
    "이 튜토리얼은",
    "이 가이드는",
)

_GENERIC_CLAIM_PATTERNS = (
    r"^this (paper|work|study) (presents|introduces|describes|discusses) (a|an) (method|approach|framework|system|pipeline)\.?$",
    r"^the (paper|work|study) (presents|introduces|describes|discusses) (a|an) (method|approach|framework|system|pipeline)\.?$",
    r"^this benchmark (evaluates|measures) [a-z0-9 -]{0,48}(systems|models|tasks|benchmarks)\.?$",
    r"^the benchmark (evaluates|measures) [a-z0-9 -]{0,48}(systems|models|tasks|benchmarks)\.?$",
    r"^this (tutorial|guide) (shows|explains) .{0,32}$",
    r"^이 (문서|글|튜토리얼|가이드)는 .{0,40}$",
)

_SUBSTANTIVE_SIGNAL_KEYWORDS = (
    "taxonomy",
    "framework",
    "benchmark",
    "leaderboard",
    "dataset",
    "task",
    "metric",
    "score",
    "accuracy",
    "error",
    "gap",
    "risk",
    "mitigation",
    "safety",
    "deployment",
    "red team",
    "preparedness",
    "workflow",
    "step",
    "example",
    "code",
    "pipeline",
    "retrieve",
    "retrieval",
    "grounded",
    "faithfulness",
    "hallucination",
    "outperform",
    "improve",
    "reduce",
    "increase",
    "evaluate",
    "evaluation",
    "ablation",
    "compare",
    "comparison",
    "criteria",
    "coverage",
    "bias",
    "failure",
    "limitation",
    "constraint",
    "범주",
    "분류",
    "분석",
    "평가",
    "지표",
    "격차",
    "위험",
    "완화",
    "배포",
    "절차",
    "예제",
    "코드",
    "환각",
    "근거",
    "개선",
    "감소",
    "증가",
    "한계",
    "제약",
)

_COMPARATIVE_SIGNAL_KEYWORDS = (
    "compared with",
    "compared to",
    "relative to",
    "versus",
    "vs.",
    "vs ",
    "outperform",
    "outperforms",
    "improve",
    "improves",
    "improved",
    "improving",
    "reduce",
    "reduces",
    "reduced",
    "increase",
    "increases",
    "higher than",
    "lower than",
    "better than",
    "worse than",
    "gap",
    "lift",
    "gain",
    "drop",
    "improvement",
    "decrease",
    "개선",
    "감소",
    "증가",
    "격차",
    "비해",
    "대비",
)

_EVALUATION_SIGNAL_KEYWORDS = (
    "benchmark",
    "dataset",
    "task",
    "metric",
    "metrics",
    "score",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "leaderboard",
    "evaluate",
    "evaluation",
    "measured",
    "measures",
    "faithfulness",
    "hallucination",
    "grounded qa",
    "grounding",
    "test set",
    "benchmark coverage",
    "지표",
    "점수",
    "평가",
    "측정",
    "벤치마크",
    "데이터셋",
)

_CLAIM_PRIORITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "survey_taxonomy": (
        "taxonomy",
        "framework",
        "criteria",
        "dimension",
        "compare",
        "comparison",
        "classif",
        "gap",
        "systematic",
        "survey",
        "review",
        "분류",
        "비교",
        "기준",
        "공백",
    ),
    "benchmark": (
        "benchmark",
        "dataset",
        "task",
        "metric",
        "score",
        "accuracy",
        "error",
        "leaderboard",
        "pass",
        "gap",
        "evaluate",
        "evaluation",
        "지표",
        "점수",
        "격차",
        "평가",
    ),
    "system_card_safety_report": (
        "risk",
        "mitigation",
        "safety",
        "capability",
        "deployment",
        "preparedness",
        "red team",
        "monitor",
        "governance",
        "failure",
        "위험",
        "완화",
        "배포",
        "안전",
    ),
    "blog_tutorial": (
        "tutorial",
        "guide",
        "workflow",
        "step",
        "example",
        "quickstart",
        "code",
        "build",
        "setup",
        "practical",
        "절차",
        "예제",
        "구현",
        "실무",
        "코드",
    ),
    "method_paper": (
        "method",
        "approach",
        "pipeline",
        "architecture",
        "retrieve",
        "generation",
        "train",
        "evaluate",
        "ablation",
        "improve",
        "outperform",
        "faithfulness",
        "hallucination",
        "방법",
        "개선",
        "성능",
        "환각",
    ),
}


def _normalize_text(*parts: Any) -> str:
    return re.sub(r"\s+", " ", " ".join(str(part or "") for part in parts)).strip()


def _split_sentences(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", normalized)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _truncate(text: str, limit: int = 220) -> str:
    token = str(text or "").strip()
    if len(token) <= limit:
        return token
    return f"{token[: max(0, limit - 3)].rstrip()}..."


def _normalize_bullet(text: str, *, limit: int = 220) -> str:
    token = _normalize_text(text)
    token = re.sub(r"^[\-\*\u2022]+\s*", "", token)
    if not token:
        return ""
    return _truncate(token, limit=limit)


def _dedupe(values: list[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= limit:
            break
    return result


def _has_substantive_signal(text: str) -> bool:
    token = _normalize_text(text).lower()
    if not token:
        return False
    if any(keyword in token for keyword in _SUBSTANTIVE_SIGNAL_KEYWORDS):
        return True
    if re.search(r"\b\d+(?:\.\d+)?%?\b", token):
        return True
    if re.search(r"\b[a-z]{2,}\d+\b", token):
        return True
    return bool(re.search(r"\b[A-Z]{2,}\b", str(text or "")))


def _has_numeric_signal(text: str) -> bool:
    token = _normalize_text(text).lower()
    if not token:
        return False
    if re.search(r"\b\d+(?:\.\d+)?%?\b", token):
        return True
    return bool(re.search(r"\b\d+(?:\.\d+)?x\b", token))


def _has_comparative_signal(text: str) -> bool:
    token = _normalize_text(text).lower()
    return bool(token) and any(keyword in token for keyword in _COMPARATIVE_SIGNAL_KEYWORDS)


def _has_evaluation_signal(text: str) -> bool:
    token = _normalize_text(text).lower()
    if not token:
        return False
    if any(keyword in token for keyword in _EVALUATION_SIGNAL_KEYWORDS):
        return True
    return bool(
        re.search(
            r"\b(on|under|across)\b.{0,40}\b(benchmark|dataset|task|evaluation|qa|leaderboard|metric|test set)\b",
            token,
        )
    )


def _has_limitation_signal(text: str) -> bool:
    token = _normalize_text(text).lower()
    if not token:
        return False
    if any(keyword in token for keyword in _LIMITATION_KEYWORDS):
        return True
    return any(
        phrase in token
        for phrase in (
            "depends on",
            "dependent on",
            "limited by",
            "fails on",
            "underperforms on",
            "sensitive to",
            "only when",
            "constrained by",
            "좌우된다",
            "의존한다",
            "제한된다",
        )
    )


def is_low_signal_text(text: str) -> bool:
    token = _normalize_bullet(text, limit=320).lower()
    if not token:
        return True
    if len(token) < 18 and not _has_substantive_signal(token):
        return True
    if any(re.match(pattern, token) for pattern in _LOW_SIGNAL_PATTERNS):
        return True
    if any(fragment in token for fragment in _LOW_SIGNAL_SUBSTRINGS):
        return True
    if any(token.startswith(prefix) for prefix in _GENERIC_OPENINGS) and not _has_substantive_signal(token):
        return True
    metadata_like = re.match(r"^[a-z0-9_ -]{1,32}\s*[:=]\s*\S+\s*$", token)
    if metadata_like:
        return True
    return False


def filter_low_signal_evidence(values: list[str], *, limit: int) -> list[str]:
    kept = [_normalize_bullet(value, limit=260) for value in values if not is_low_signal_text(str(value or ""))]
    return _dedupe([item for item in kept if item], limit=limit)


def _claim_priority_score(text: str, *, document_type: str, source_kind: str) -> float:
    token = _normalize_bullet(text, limit=320)
    lowered = token.lower()
    score = min(len(lowered) / 110.0, 2.4)
    if source_kind == "claim":
        score += 2.0
    if _has_numeric_signal(lowered):
        score += 2.3
    if _has_comparative_signal(lowered):
        score += 1.9
    if _has_evaluation_signal(lowered):
        score += 1.4
    if _has_limitation_signal(lowered):
        score += 1.0
    if any(
        phrase in lowered
        for phrase in (
            "depends on",
            "limited by",
            "coverage",
            "annotation",
            "scope",
            "의존",
            "제약",
        )
    ):
        score += 0.7
    if any(
        keyword in lowered
        for keyword in (
            "show",
            "shows",
            "demonstrate",
            "demonstrates",
            "reveal",
            "reveals",
            "risk",
            "mitigation",
            "failure",
            "향상",
            "위험",
            "완화",
            "보여",
            "드러낸",
        )
    ):
        score += 1.1
    if any(keyword in lowered for keyword in _CLAIM_PRIORITY_KEYWORDS.get(document_type, ())):
        score += 2.2
    if any(re.match(pattern, lowered) for pattern in _GENERIC_CLAIM_PATTERNS):
        score -= 3.2
    elif any(lowered.startswith(prefix) for prefix in _GENERIC_OPENINGS):
        score -= 0.9 if (_has_numeric_signal(lowered) or _has_comparative_signal(lowered) or _has_evaluation_signal(lowered)) else 2.1
    if len(lowered) < 48 and not (_has_numeric_signal(lowered) or _has_comparative_signal(lowered) or _has_evaluation_signal(lowered)):
        score -= 0.8
    if not _has_substantive_signal(token):
        score -= 1.8
    return score


def _rank_claim_candidates(
    values: list[str],
    *,
    document_type: str,
    source_kind: str,
    limit: int,
) -> list[str]:
    ranked: list[tuple[float, int, str]] = []
    for index, value in enumerate(values):
        token = _normalize_bullet(value, limit=260)
        if not token or is_low_signal_text(token):
            continue
        ranked.append((_claim_priority_score(token, document_type=document_type, source_kind=source_kind), index, token))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    return _dedupe([token for _, _, token in ranked], limit=limit)


def infer_document_type(
    *,
    title: str,
    source_url: str,
    domain: str,
    content_text: str,
    key_excerpts: list[str],
    metadata_lines: list[str] | None = None,
    claim_lines: list[str] | None = None,
) -> str:
    title_lower = _normalize_text(title).lower()
    body_lower = _normalize_text(content_text, " ".join(key_excerpts), " ".join(metadata_lines or []), " ".join(claim_lines or [])).lower()
    url_lower = str(source_url or "").lower()
    domain_lower = str(domain or "").lower()

    survey_hits = sum(
        1
        for token in (
            "survey",
            "systematic review",
            "literature review",
            "taxonomy",
            "framework",
            "comparison",
            "evaluation criteria",
            "research gap",
            "gap",
        )
        if token in title_lower or token in body_lower
    )
    system_card_hits = sum(
        1
        for token in (
            "system card",
            "model card",
            "safety report",
            "preparedness",
            "red team",
            "red teaming",
            "risk",
            "mitigation",
            "capability",
            "deployment",
        )
        if token in title_lower or token in body_lower or token in url_lower
    )
    benchmark_hits = sum(
        1
        for token in (
            "benchmark",
            "leaderboard",
            "evaluation suite",
            "dataset",
            "task",
            "metric",
            "human-model gap",
        )
        if token in title_lower or token in body_lower
    )
    tutorial_hits = sum(
        1
        for token in (
            "tutorial",
            "guide",
            "how to",
            "walkthrough",
            "quickstart",
            "getting started",
            "introduction",
            "example",
        )
        if token in title_lower or token in body_lower or token in url_lower
    )
    blog_like = (
        "/blog/" in url_lower
        or domain_lower.startswith("blog.")
        or any(token in domain_lower for token in ("medium.com", "substack.com", "huggingface.co", "openai.com"))
    )
    has_explicit_survey_title = any(token in title_lower for token in ("survey", "systematic review", "literature review", "review"))
    has_explicit_system_card_title = any(token in title_lower for token in ("system card", "model card", "safety report", "preparedness"))
    has_explicit_tutorial_title = any(token in title_lower for token in ("tutorial", "guide", "how to", "quickstart"))

    if has_explicit_survey_title or survey_hits >= 3:
        return "survey_taxonomy"
    if has_explicit_system_card_title and system_card_hits >= 2:
        return "system_card_safety_report"
    if has_explicit_tutorial_title or (blog_like and any(token in body_lower for token in ("step", "example", "workflow", "code", "lines of code", "build"))):
        return "blog_tutorial"
    if benchmark_hits >= 2 and any(token in body_lower for token in ("evaluate", "evaluation", "results", "score", "accuracy", "pass")):
        return "benchmark"
    if tutorial_hits >= 2:
        return "blog_tutorial"
    return "method_paper"


def extract_thesis(
    *,
    title: str,
    document_type: str,
    content_text: str,
    claim_lines: list[str],
    key_excerpts: list[str],
) -> str:
    ranked_claims = _rank_claim_candidates(
        [item for item in claim_lines if len(str(item or "").strip()) >= 24],
        document_type=document_type,
        source_kind="claim",
        limit=1,
    )
    if ranked_claims:
        return _normalize_bullet(ranked_claims[0], limit=200)
    candidates = _split_sentences(_normalize_text(" ".join(key_excerpts[:2]), content_text))
    prioritized = [
        sentence
        for sentence in candidates
        if any(token in sentence.lower() for token in ("we introduce", "we present", "this paper", "this work", "this review", "this tutorial"))
    ]
    ranked_candidates = _rank_claim_candidates(
        prioritized or candidates,
        document_type=document_type,
        source_kind="sentence",
        limit=1,
    )
    if ranked_candidates:
        return _normalize_bullet(ranked_candidates[0], limit=200)
    fallback_prefix = {
        "survey_taxonomy": f"{title}는 기존 연구를 구조화해 읽는 기준틀을 제공하는 survey 문서다.",
        "benchmark": f"{title}는 평가 셋업과 성능 격차를 드러내는 benchmark 문서다.",
        "system_card_safety_report": f"{title}는 capability, risk, mitigation, evaluation을 함께 정리하는 system card 계열 문서다.",
        "blog_tutorial": f"{title}는 실무 절차와 예시를 빠르게 따라갈 수 있게 정리한 tutorial 계열 문서다.",
        "method_paper": f"{title}는 특정 방법과 그 효과를 제안하거나 검증하는 method paper다.",
    }
    return fallback_prefix.get(document_type, f"{title}의 핵심 논지를 추가 근거와 함께 확인해야 합니다.")


def extract_top_claims(
    *,
    content_text: str,
    claim_lines: list[str],
    key_excerpts: list[str],
    document_type: str,
    limit: int = 4,
) -> list[str]:
    bullets = _rank_claim_candidates(
        claim_lines,
        document_type=document_type,
        source_kind="claim",
        limit=max(limit * 2, limit),
    )
    candidate_sentences = [
        sentence
        for sentence in _split_sentences(_normalize_text(" ".join(key_excerpts[:3]), content_text))
        if any(
            token in sentence.lower()
            for token in (
                "we introduce",
                "we present",
                "we evaluate",
                "we show",
                "we review",
                "we provide",
                "outperform",
                "improve",
                "gap",
                "risk",
                "mitigation",
                "compared to",
                "compared with",
                "relative to",
                "accuracy",
                "precision",
                "recall",
                "faithfulness",
                "hallucination",
                "benchmark",
                "dataset",
                "metric",
                "leaderboard",
                "depends on",
                "limited by",
                "failure",
            )
        )
        or _has_numeric_signal(sentence)
    ]
    ranked_sentences = _rank_claim_candidates(
        candidate_sentences,
        document_type=document_type,
        source_kind="sentence",
        limit=max(limit * 3, limit),
    )
    if document_type == "survey_taxonomy":
        ranked_sentences = _rank_claim_candidates(
            [
                *candidate_sentences,
                *[
                    item
                    for item in candidate_sentences
                    if any(token in item.lower() for token in ("taxonomy", "framework", "criteria", "gap"))
                ],
            ],
            document_type=document_type,
            source_kind="sentence",
            limit=max(limit * 3, limit),
        )
    return _dedupe([*bullets, *ranked_sentences], limit=limit)


def extract_core_concepts(entity_names: list[str], related_concepts: list[str], *, limit: int = 8) -> list[str]:
    normalized_related = [str(item).lstrip("- ").strip() for item in related_concepts if str(item).strip()]
    return _dedupe(
        [str(item).strip() for item in entity_names if str(item).strip()] + normalized_related,
        limit=limit,
    )


def representative_sources(
    *,
    title: str,
    source_url: str,
    domain: str,
    metadata_lines: list[str],
) -> list[str]:
    result: list[str] = []
    if source_url:
        result.append(f"{title} ({source_url})")
    if domain:
        result.append(f"source domain: {domain}")
    for item in metadata_lines[:3]:
        token = _normalize_bullet(item, limit=180)
        if token:
            result.append(token)
    return _dedupe(result, limit=4)


def synthesize_evidence_sections(
    *,
    document_type: str,
    title: str,
    thesis: str,
    content_text: str,
    entity_names: list[str],
    relation_lines: list[str],
    claim_lines: list[str],
    related_concepts: list[str],
    key_excerpts: list[str],
    metadata_lines: list[str],
) -> dict[str, list[str]]:
    sentences = [
        _normalize_bullet(item)
        for item in _split_sentences(_normalize_text(" ".join(key_excerpts[:4]), content_text))
        if _normalize_bullet(item) and not is_low_signal_text(item)
    ]
    top_claims = extract_top_claims(
        content_text=content_text,
        claim_lines=claim_lines,
        key_excerpts=key_excerpts,
        document_type=document_type,
        limit=4,
    )
    concepts = extract_core_concepts(entity_names, related_concepts, limit=8)
    relation_bullets = _dedupe(
        [
            _normalize_bullet(f"관계 단서: {str(item).replace('-[', ' ').replace(']->', ' ').replace('->', ' ')}")
            for item in relation_lines
            if str(item).strip()
        ],
        limit=4,
    )
    limitations = _dedupe(
        [
            sentence
            for sentence in sentences + top_claims
            if any(token in sentence.lower() for token in _LIMITATION_KEYWORDS)
        ],
        limit=4,
    )

    type_specific: dict[str, list[str]] = {
        "contributions": [],
        "methodology": [],
        "results_or_findings": [],
        "limitations": limitations,
        "insights": [],
    }
    if document_type == "survey_taxonomy":
        type_specific["contributions"] = _dedupe(
            [
                thesis,
                *[item for item in top_claims if any(token in item.lower() for token in ("taxonomy", "framework", "criteria", "gap"))],
                *[item for item in sentences if any(token in item.lower() for token in ("taxonomy", "systematic", "framework", "compare"))],
            ],
            limit=5,
        )
        type_specific["methodology"] = _dedupe(
            [
                *[item for item in sentences if any(token in item.lower() for token in ("review", "survey", "compare", "classif", "dimension", "criteria"))],
                *relation_bullets,
            ],
            limit=5,
        )
        type_specific["results_or_findings"] = _dedupe(
            [
                *[item for item in top_claims if any(token in item.lower() for token in ("gap", "benchmark", "risk", "control", "propensit", "capabilit"))],
                *sentences[:3],
            ],
            limit=5,
        )
        type_specific["insights"] = _dedupe(
            [
                f"핵심 비교 축: {', '.join(concepts[:5])}" if concepts else "",
                "개별 benchmark보다 상위 taxonomy를 먼저 이해해야 후속 source note 해석이 쉬워집니다.",
                *[item for item in sentences if any(token in item.lower() for token in ("criteria", "framework", "governance", "interpret"))],
            ],
            limit=5,
        )
    elif document_type == "system_card_safety_report":
        type_specific["contributions"] = _dedupe(
            [thesis, *[item for item in top_claims if any(token in item.lower() for token in ("capability", "risk", "mitigation", "safety"))]],
            limit=5,
        )
        type_specific["methodology"] = _dedupe(
            [
                *[item for item in sentences if any(token in item.lower() for token in ("evaluation", "red team", "preparedness", "deployment", "scope"))],
                *relation_bullets,
            ],
            limit=5,
        )
        type_specific["results_or_findings"] = _dedupe(
            [
                *[item for item in top_claims if any(token in item.lower() for token in ("risk", "mitigation", "deployment", "behavior"))],
                *sentences[:3],
            ],
            limit=5,
        )
        type_specific["insights"] = _dedupe(
            [
                "capability, risk, mitigation, evaluation을 한 문서에서 같이 보는 기준점으로 읽을 가치가 있습니다.",
                *[item for item in sentences if any(token in item.lower() for token in ("scope", "governance", "monitor"))],
            ],
            limit=5,
        )
    elif document_type == "benchmark":
        type_specific["contributions"] = _dedupe(
            [thesis, *[item for item in top_claims if any(token in item.lower() for token in ("benchmark", "dataset", "task", "metric"))]],
            limit=5,
        )
        type_specific["methodology"] = _dedupe(
            [
                *[item for item in sentences if any(token in item.lower() for token in ("evaluate", "benchmark", "dataset", "metric", "task", "annotat"))],
                *relation_bullets,
            ],
            limit=5,
        )
        type_specific["results_or_findings"] = _dedupe(
            [
                *[item for item in top_claims if any(token in item.lower() for token in ("gap", "accuracy", "performance", "score", "pass"))],
                *sentences[:3],
            ],
            limit=5,
        )
        type_specific["insights"] = _dedupe(
            [
                "모델 성능보다 평가 설정이 무엇을 측정하는지 먼저 확인해야 하는 benchmark 문서입니다.",
                *[item for item in sentences if any(token in item.lower() for token in ("gap", "error", "sample efficient", "generalization"))],
            ],
            limit=5,
        )
    elif document_type == "blog_tutorial":
        type_specific["contributions"] = _dedupe(
            [thesis, *[item for item in top_claims if any(token in item.lower() for token in ("workflow", "tutorial", "guide", "example", "step"))]],
            limit=5,
        )
        type_specific["methodology"] = _dedupe(
            [
                *[item for item in sentences if any(token in item.lower() for token in ("step", "example", "workflow", "build", "use"))],
                *relation_bullets,
            ],
            limit=5,
        )
        type_specific["results_or_findings"] = _dedupe(
            [
                *[item for item in top_claims if any(token in item.lower() for token in ("improve", "result", "performance", "practical"))],
                *sentences[:3],
            ],
            limit=5,
        )
        type_specific["insights"] = _dedupe(
            [
                "구현 순서와 실무 적용 맥락을 빠르게 잡는 입문용 source note로 유용합니다.",
                *[item for item in sentences if any(token in item.lower() for token in ("step", "practical", "example", "workflow"))],
            ],
            limit=5,
        )
    else:
        type_specific["contributions"] = _dedupe([thesis, *top_claims, *sentences[:2]], limit=5)
        type_specific["methodology"] = _dedupe(
            [*relation_bullets, *[item for item in sentences if any(token in item.lower() for token in ("method", "approach", "train", "retrieve", "generate", "evaluate"))]],
            limit=5,
        )
        type_specific["results_or_findings"] = _dedupe(
            [*[item for item in top_claims if any(token in item.lower() for token in ("improve", "outperform", "result", "generalization", "accuracy"))], *sentences[:3]],
            limit=5,
        )
        type_specific["insights"] = _dedupe(
            [
                f"핵심 연결 개념: {', '.join(concepts[:5])}" if concepts else "",
                *[item for item in sentences if any(token in item.lower() for token in ("implication", "suggest", "practical", "future work"))],
            ],
            limit=5,
        )

    if not type_specific["limitations"]:
        type_specific["limitations"] = _dedupe(
            [
                *[item for item in metadata_lines if "quality" in str(item).lower()],
                *[item for item in sentences if any(token in item.lower() for token in ("scope", "limit", "risk", "coverage"))],
                "세부 수치나 적용 범위는 원문과 대표 근거 문장을 함께 확인해야 합니다.",
            ],
            limit=4,
        )
    return {
        "top_claims": top_claims,
        "core_concepts": concepts,
        "contributions": type_specific["contributions"],
        "methodology": type_specific["methodology"],
        "results_or_findings": type_specific["results_or_findings"],
        "limitations": type_specific["limitations"],
        "insights": type_specific["insights"],
        "representative_sources": representative_sources(
            title=title,
            source_url="",
            domain="",
            metadata_lines=metadata_lines,
        ),
    }
