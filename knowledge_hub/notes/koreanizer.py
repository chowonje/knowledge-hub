"""Local-first Korean note generation helpers."""

from __future__ import annotations

import json
import os
import re
import requests
from typing import Any

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.core.sanitizer import redact_p0
from knowledge_hub.learning.policy import evaluate_policy_for_payload
from knowledge_hub.learning.task_router import get_llm_for_task
from knowledge_hub.notes.glossary import load_glossary, protect_terms, restore_terms
from knowledge_hub.notes.source_profile import (
    extract_core_concepts,
    representative_sources,
    synthesize_evidence_sections,
)
from knowledge_hub.infrastructure.providers import get_llm


_FALLBACK_HEAVY_PHRASES = (
    "추가 근거가 제한적입니다",
    "함께 읽어야",
    "추가 검토가 필요합니다",
    "원문 근거를 추가로 검토해야 합니다",
    "출발점으로 볼 수 있습니다",
    "대표 근거를 기준으로 해석해야 합니다",
)
_CONCEPT_BANNED_PHRASES = (
    "state-of-the-art",
    "모든 태스크에서",
    "혁신적으로",
    "최초로",
    "완벽하게 해결",
)
_SOURCE_BANNED_PHRASES = (
    "this paper presents a method",
    "this paper presents",
    "this work presents",
    "this tutorial shows",
    "this guide explains",
    "이 문서는",
    "이 글은",
    "이 튜토리얼은",
    "함께 읽어야 한다",
)


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if not normalized:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", normalized)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _truncate(text: str, limit: int) -> str:
    token = str(text or "").strip()
    if len(token) <= limit:
        return token
    return f"{token[: max(0, limit - 3)].rstrip()}..."


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


def _ensure_bullets(values: list[str], *, minimum: int, fallback: str) -> list[str]:
    cleaned = [str(value).strip() for value in values if str(value or "").strip()]
    if len(cleaned) >= minimum:
        return cleaned
    while len(cleaned) < minimum:
        cleaned.append(fallback)
    return cleaned


def _clean_title(text: str, fallback: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return fallback
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return fallback
    title = lines[0]
    title = re.sub(r"^[#>*`\-\s]+", "", title).strip()
    title = re.sub(r"^(제목|title)\s*:\s*", "", title, flags=re.IGNORECASE).strip()
    title = re.sub(r"^\[[0-9]{4}\.[0-9]{4,5}(?:v\d+)?\]\s*", "", title).strip()
    title = title.replace("**", "").strip()
    title = re.sub(r"\s+", " ", title)
    if not title:
        return fallback
    if len(title) > 120:
        title = title[:120].rstrip(" .-_")
    return title


def _clean_summary(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = raw.replace("\r\n", "\n")
    raw = re.sub(r"^\s*#+\s*제목\s*:.*?$", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^\s*\*\*요약:?\*\*\s*$", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^\s*요약\s*:?\s*$", "", raw, flags=re.MULTILINE)
    cleaned_lines: list[str] = []
    pending_blank = False
    for line in raw.splitlines():
        token = line.strip()
        if not token:
            if cleaned_lines:
                pending_blank = True
            continue
        if re.match(r"^#{1,6}\s+", token):
            continue
        token = re.sub(r"^\*\*([^*]+?)\*\*:?\s*", r"\1", token).strip()
        token = token.rstrip(":").strip()
        if re.match(r"^(관련 문서|관련 문서 및 지지 근거|관련 문서 및 인용|핵심 기술 용어)\s*:?\s*$", token):
            continue
        if token in {"-", "*"}:
            continue
        if pending_blank and cleaned_lines:
            cleaned_lines.append("")
        pending_blank = False
        cleaned_lines.append(token)
    raw = "\n".join(cleaned_lines)
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
    return raw


def _extract_summary_line(text: str, fallback: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if not normalized:
        return fallback
    if len(normalized) <= 180:
        return normalized
    sentences = _split_sentences(normalized)
    if sentences:
        head = sentences[0].strip()
        if head:
            return _truncate(head, 180)
    return _truncate(normalized, 180)


def _normalize_bullet(text: str, *, limit: int = 180) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    normalized = re.sub(r"^[\-\*\u2022]+\s*", "", normalized)
    if not normalized:
        return ""
    return _truncate(normalized, limit)


def _extract_claim_bullets(claim_lines: list[str], *, limit: int = 4) -> list[str]:
    return _dedupe(
        [
            _normalize_bullet(line)
            for line in claim_lines
            if _normalize_bullet(line)
        ],
        limit=limit,
    )


def _extract_relation_bullets(relation_lines: list[str], *, limit: int = 4) -> list[str]:
    bullets: list[str] = []
    for line in relation_lines:
        token = str(line or "").strip()
        if not token:
            continue
        token = token.replace("-[", " ").replace("]->", " ").replace("->", " ")
        token = re.sub(r"\s+", " ", token).strip()
        normalized = _normalize_bullet(f"관계 단서: {token}")
        if normalized:
            bullets.append(normalized)
    return _dedupe(bullets, limit=limit)


def _extract_content_bullets(content_text: str, *, limit: int = 6) -> list[str]:
    sentences = _split_sentences(content_text)
    bullets: list[str] = []
    for sentence in sentences:
        normalized = _normalize_bullet(sentence)
        if normalized:
            bullets.append(normalized)
    return _dedupe(bullets, limit=limit)


def _build_grounded_source_sections(
    *,
    content_text: str,
    entity_names: list[str],
    relation_lines: list[str],
    claim_lines: list[str],
    related_concepts: list[str],
    key_excerpts_en: list[str],
    metadata_lines: list[str],
) -> dict[str, list[str]]:
    claim_bullets = _extract_claim_bullets(claim_lines, limit=4)
    relation_bullets = _extract_relation_bullets(relation_lines, limit=4)
    excerpt_bullets = _dedupe(
        [
            _normalize_bullet(f"핵심 근거: {excerpt}", limit=220)
            for excerpt in key_excerpts_en
            if _normalize_bullet(excerpt, limit=220)
        ],
        limit=4,
    )
    content_bullets = _extract_content_bullets(content_text, limit=6)

    limitation_keywords = (
        "limit",
        "limitation",
        "scope",
        "constraint",
        "depend",
        "benchmark",
        "quality",
        "한계",
        "제약",
        "의존",
        "범위",
    )
    limitation_bullets = _dedupe(
        [
            bullet
            for bullet in [*excerpt_bullets, *content_bullets]
            if any(keyword in bullet.lower() for keyword in limitation_keywords)
        ],
        limit=4,
    )

    entity_summary = ""
    if entity_names:
        entity_summary = _normalize_bullet(
            f"핵심 개체는 {', '.join(entity_names[:4])} 중심으로 전개됩니다."
        )
    concept_summary = ""
    if related_concepts:
        concept_summary = _normalize_bullet(
            f"관련 개념은 {', '.join(related_concepts[:4])}와 직접 연결됩니다."
        )
    metadata_summary = ""
    if metadata_lines:
        metadata_summary = _normalize_bullet(
            f"메타데이터 단서: {'; '.join(str(item).strip() for item in metadata_lines[:2] if str(item).strip())}",
            limit=220,
        )

    contributions = _dedupe(
        [*claim_bullets, *excerpt_bullets, *content_bullets[:2], entity_summary],
        limit=5,
    )
    methodology = _dedupe(
        [*relation_bullets, *content_bullets[1:4], *excerpt_bullets[:2], concept_summary],
        limit=5,
    )
    key_results = _dedupe(
        [*claim_bullets[1:], *excerpt_bullets, *content_bullets[:3], metadata_summary],
        limit=5,
    )
    limitations = _dedupe(
        [*limitation_bullets, metadata_summary, concept_summary],
        limit=5,
    )
    insights = _dedupe(
        [concept_summary, entity_summary, metadata_summary, *excerpt_bullets[:2], *relation_bullets[:2]],
        limit=5,
    )
    return {
        "contributions": [item for item in contributions if item],
        "methodology": [item for item in methodology if item],
        "key_results": [item for item in key_results if item],
        "limitations": [item for item in limitations if item],
        "insights": [item for item in insights if item],
    }


def _evidence_list(payload: dict[str, Any], key: str, *, limit: int = 5) -> list[str]:
    return _dedupe([_normalize_bullet(item, limit=220) for item in (payload.get(key) or []) if _normalize_bullet(item, limit=220)], limit=limit)


def _merge_bullets(*groups: list[str], limit: int) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for item in group:
            token = _normalize_bullet(item, limit=220)
            if token:
                merged.append(token)
    return _dedupe(merged, limit=limit)


def _detect_fallback_heavy(*groups: list[str]) -> list[str]:
    reasons: list[str] = []
    flattened = [str(item or "").strip() for group in groups for item in group if str(item or "").strip()]
    if not flattened:
        return ["empty_sections"]
    fallback_hits = sum(
        1
        for item in flattened
        if any(phrase in item for phrase in _FALLBACK_HEAVY_PHRASES)
    )
    if fallback_hits >= max(3, len(flattened) // 2):
        reasons.append("fallback_phrase_density_high")
    if len(set(flattened)) <= max(2, len(flattened) // 2):
        reasons.append("low_unique_bullet_diversity")
    return reasons


def _looks_generic_source_summary_line(text: str) -> bool:
    token = re.sub(r"\s+", " ", str(text or "").strip())
    generic_patterns = (
        "이 문서는 AI 문서를 중심으로 설명합니다.",
        "이 문서는 핵심 내용을 중심으로 설명합니다.",
        "이 문서는 관련 내용을 중심으로 설명합니다.",
    )
    return token in generic_patterns or (token.startswith("이 문서는 ") and token.endswith("중심으로 설명합니다."))


def _contains_source_banned_phrase(text: str) -> bool:
    token = str(text or "").casefold()
    return any(phrase.casefold() in token for phrase in _SOURCE_BANNED_PHRASES)


def _sanitize_source_banned_phrases(text: str) -> str:
    cleaned = str(text or "")
    for phrase in _SOURCE_BANNED_PHRASES:
        cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" ,.;:-")


def _contains_banned_phrase(text: str) -> bool:
    token = str(text or "").casefold()
    return any(phrase.casefold() in token for phrase in _CONCEPT_BANNED_PHRASES)


def _sanitize_banned_phrases(text: str) -> str:
    cleaned = str(text or "")
    for phrase in _CONCEPT_BANNED_PHRASES:
        cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" ,.;:-")


def _filter_grounded_bullets(values: list[str], *, limit: int) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        token = _normalize_bullet(_sanitize_banned_phrases(value), limit=220)
        if not token:
            continue
        if _contains_banned_phrase(token):
            continue
        cleaned.append(token)
    return _dedupe(cleaned, limit=limit)


def _filter_grounded_source_bullets(values: list[str], *, limit: int) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        token = _normalize_bullet(_sanitize_source_banned_phrases(value), limit=220)
        if not token:
            continue
        lowered = token.casefold()
        if _contains_source_banned_phrase(token):
            continue
        if lowered.startswith(("this paper presents", "this work presents", "this tutorial shows", "this guide explains")):
            continue
        if token.startswith("이 문서는 ") and len(token) <= 60:
            continue
        cleaned.append(token)
    return _dedupe(cleaned, limit=limit)


def _looks_generic_concept_summary_line(text: str, canonical_name: str) -> bool:
    token = re.sub(r"\s+", " ", str(text or "").strip())
    generic_patterns = (
        f"{canonical_name} 개념 요약입니다.",
        f"{canonical_name} 개념을 여러 근거에서 통합한 요약입니다.",
        "개념을 여러 문서 근거로 통합한 요약입니다.",
    )
    return token in generic_patterns or token.endswith("개념 요약입니다.")


def _grounded_concept_importance(
    *,
    canonical_name: str,
    relation_lines: list[str],
    related_concepts: list[str],
    compressed_support_docs: list[dict[str, Any]],
) -> list[str]:
    doc_count = len([item for item in compressed_support_docs if str(item.get("source_title") or "").strip()])
    domains = _dedupe([str(item.get("domain") or "").strip() for item in compressed_support_docs if str(item.get("domain") or "").strip()], limit=6)
    concepts = _dedupe([str(item).strip() for item in related_concepts if str(item).strip()], limit=4)
    relation_hints = _filter_grounded_bullets(relation_lines, limit=3)
    bullets = [
        f"{canonical_name}는 {max(1, doc_count)}개 근거 문서에서 반복적으로 등장해 설명 우선순위가 높은 개념입니다.",
    ]
    if concepts:
        bullets.append(
            f"{', '.join(concepts[:3])}와 직접 연결되어 후속 개념 노트 탐색의 기준점이 됩니다."
        )
    if domains:
        bullets.append(
            f"{', '.join(domains[:3])} 등 서로 다른 출처에서 공통 맥락이 확인되어 검색과 학습 재사용성이 높습니다."
        )
    if relation_hints:
        bullets.append(relation_hints[0])
    return _filter_grounded_bullets(bullets, limit=4)


def _grounded_concept_relations(
    relation_lines: list[str],
    compressed_support_docs: list[dict[str, Any]],
) -> list[str]:
    merged = list(relation_lines)
    for item in compressed_support_docs[:8]:
        merged.extend(str(line).strip() for line in (item.get("relation_lines") or []) if str(line).strip())
    return _filter_grounded_bullets(merged, limit=6)


def _grounded_related_sources(compressed_support_docs: list[dict[str, Any]]) -> list[str]:
    bullets: list[str] = []
    for item in compressed_support_docs[:8]:
        title = str(item.get("source_title") or "").strip()
        why_relevant = str(item.get("why_relevant") or item.get("summary_line_ko") or "").strip()
        if title and why_relevant:
            bullets.append(f"{title} - {why_relevant}")
        elif title:
            bullets.append(title)
    return _filter_grounded_bullets(bullets, limit=8)


def _grounded_related_concepts(related_concepts: list[str]) -> list[str]:
    return _filter_grounded_bullets([str(item).strip() for item in related_concepts if str(item).strip()], limit=12)


class Koreanizer:
    def __init__(
        self,
        config: Config,
        *,
        allow_external: bool = False,
        llm_mode: str = "auto",
        local_timeout_sec: int | None = None,
        api_fallback_on_timeout: bool = True,
    ):
        self.config = config
        self.allow_external = bool(allow_external)
        self.llm_mode = str(llm_mode or "auto").strip().lower() or "auto"
        self.local_timeout_sec = int(local_timeout_sec or config.get_nested("routing", "llm", "tasks", "local", "timeout_sec", default=45) or 45)
        self.api_fallback_on_timeout = bool(api_fallback_on_timeout)
        glossary_path = self.config.get_nested("materialization", "glossary_path", default="")
        self.glossary = load_glossary(str(glossary_path or ""))
        self._llm_cache: dict[tuple[str, str, int], Any] = {}

    def _get_cached_llm(self, provider: str, model: str, timeout_sec: int):
        provider_token = str(provider or "").strip()
        model_token = str(model or "").strip()
        if provider_token == "ollama":
            model_token = self._resolve_ollama_model(model_token)
        cache_key = (provider_token, model_token, int(timeout_sec))
        if cache_key not in self._llm_cache:
            provider_cfg = dict(self.config.get_provider_config(provider_token))
            provider_cfg.setdefault("timeout", float(timeout_sec))
            provider_cfg.setdefault("request_timeout", float(timeout_sec))
            self._llm_cache[cache_key] = get_llm(provider_token, model=model_token, **provider_cfg)
        return self._llm_cache[cache_key], model_token

    def _resolve_ollama_model(self, configured_model: str) -> str:
        preferred = [
            str(configured_model or "").strip(),
            "exaone3.5:7.8b",
            "mistral:latest",
            "llama3:latest",
            "qwen3:14b",
        ]
        try:
            response = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
            response.raise_for_status()
            data = response.json()
            installed = {
                str(item.get("name") or "").strip()
                for item in data.get("models", [])
                if str(item.get("name") or "").strip()
            }
        except Exception:
            return str(configured_model or "").strip()
        for candidate in preferred:
            if candidate and candidate in installed:
                return candidate
        return str(configured_model or "").strip()

    def _safe_json_object(self, raw: str) -> dict[str, Any]:
        text = str(raw or "").strip()
        if not text:
            return {}
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _run_task(
        self,
        *,
        task_type: str,
        prompt: str,
        context: str,
        allow_external: bool,
        force_route: str | None = None,
        max_tokens: int = 900,
    ) -> tuple[str, list[str]]:
        warnings: list[str] = []
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return "", warnings
        protected_prompt, prompt_terms = protect_terms(prompt, self.glossary)
        protected_context, context_terms = protect_terms(context, self.glossary)
        decision_allow_external = bool(allow_external and self.allow_external)
        llm, decision, route_warnings = get_llm_for_task(
            self.config,
            task_type=task_type,  # type: ignore[arg-type]
            allow_external=decision_allow_external,
            query=restore_terms(protected_prompt, prompt_terms),
            context=restore_terms(protected_context, context_terms),
            source_count=1,
            force_route=(force_route or self.llm_mode),  # type: ignore[arg-type]
            timeout_sec=self.local_timeout_sec if (force_route or self.llm_mode) == "local" else None,
        )
        warnings.extend(route_warnings)
        chain = [route for route in decision.fallback_chain if route != "fallback-only"]
        if not chain and llm is not None:
            chain = [decision.route]

        for idx, route in enumerate(chain):
            active_llm = llm
            active_decision = decision
            if idx > 0 or active_llm is None or active_decision.route != route:
                active_llm, active_decision, extra = get_llm_for_task(
                    self.config,
                    task_type=task_type,  # type: ignore[arg-type]
                    allow_external=decision_allow_external,
                    query=restore_terms(protected_prompt, prompt_terms),
                    context=restore_terms(protected_context, context_terms),
                    source_count=1,
                    force_route=route,  # type: ignore[arg-type]
                    timeout_sec=self.local_timeout_sec if route == "local" else None,
                )
                warnings.extend(extra)
            if active_llm is None:
                continue

            prompt_value = restore_terms(protected_prompt, prompt_terms)
            context_value = restore_terms(protected_context, context_terms)
            if route in {"mini", "strong"}:
                sanitized_context = redact_p0(context_value)
                policy = evaluate_policy_for_payload(
                    allow_external=True,
                    raw_texts=[prompt_value, sanitized_context],
                    mode="ko-materializer-external",
                )
                if not policy.allowed:
                    warnings.extend(policy.warnings or [])
                    warnings.extend(policy.policy_errors or [])
                    break
                warnings.extend(policy.warnings or [])
                context_value = sanitized_context

            try:
                client, _resolved_model = self._get_cached_llm(
                    active_decision.provider,
                    active_decision.model,
                    self.local_timeout_sec if route == "local" else active_decision.timeout_sec,
                )
                output = str(client.generate(prompt_value, context=context_value, max_tokens=max_tokens) or "").strip()
                if output:
                    return restore_terms(output, {**prompt_terms, **context_terms}), warnings
            except Exception as error:
                if route == "local" and decision_allow_external and self.api_fallback_on_timeout:
                    warnings.append(f"local-timeout-fallback-to-mini: {error}")
                    continue
                if route == "local":
                    warnings.append(f"local-timeout-fallback-deterministic: {error}")
                else:
                    warnings.append(f"external-llm-fallback: {error}")
        return "", warnings

    def _fallback_summary(self, title: str, text: str, *, max_sentences: int = 4) -> str:
        sentences = _split_sentences(text)
        if not sentences:
            if title:
                return f"{title} 문서의 핵심 정보를 자동 요약할 충분한 본문이 없습니다."
            return "자동 요약에 사용할 본문이 충분하지 않습니다."
        selected = sentences[: max(1, max_sentences)]
        joined = " ".join(_truncate(sentence, 180) for sentence in selected)
        if title:
            return f"{title}에 대한 핵심 요약입니다. {joined}"
        return joined

    def _fallback_source_summary(
        self,
        *,
        title: str,
        entity_names: list[str],
        relation_lines: list[str],
        key_excerpts_en: list[str],
    ) -> str:
        entities = ", ".join(_dedupe(entity_names, limit=4)) or "주요 AI 개념"
        relations = ", ".join(
            _dedupe(
                [line.split("-[", 1)[1].split("]->", 1)[0] for line in relation_lines if "-[" in line],
                limit=4,
            )
        ) or "핵심 연관 관계"
        excerpt = _truncate((key_excerpts_en or [""])[0], 180)
        return (
            f"이 문서는 {entities}를 중심으로 설명합니다. "
            f"온톨로지 추출 기준으로 {relations} 관계가 확인되었습니다. "
            f"원문 제목은 '{title}'이며, 대표 근거는 \"{excerpt}\" 입니다."
        )

    def _fallback_concept_summary(
        self,
        *,
        canonical_name: str,
        support_lines: list[str],
        relation_lines: list[str],
        key_excerpts_en: list[str],
    ) -> str:
        support_count = len([line for line in support_lines if str(line).strip()])
        relations = ", ".join(
            _dedupe(
                [line.split("-[", 1)[1].split("]->", 1)[0] for line in relation_lines if "-[" in line],
                limit=4,
            )
        ) or "관련 관계"
        excerpt = _truncate((key_excerpts_en or [""])[0], 160)
        return (
            f"{canonical_name} 개념은 현재 수집된 문서 {support_count}건에서 반복적으로 등장합니다. "
            f"이 개념은 {relations} 관계를 통해 다른 개념 및 문서와 연결됩니다. "
            f"대표 근거는 \"{excerpt}\" 입니다."
        )

    def _structured_source_summary(
        self,
        *,
        title_en: str,
        entity_names: list[str],
        relation_lines: list[str],
        content_text: str,
        allow_external: bool,
    ) -> tuple[dict[str, Any], list[str]]:
        prompt = (
            "다음 웹 문서를 현재 source note 템플릿에 맞는 한국어 지식 노트로 정리하세요. "
            "엄격한 JSON object만 반환하세요.\n"
            "Schema:\n"
            "{"
            '"title_ko":"string",'
            '"summary_line_ko":"string",'
            '"core_summary":"string",'
            '"contributions":["string"],'
            '"methodology":["string"],'
            '"key_results":["string"],'
            '"limitations":["string"],'
            '"insights":["string"]'
            "}\n"
            "Rules:\n"
            "- 기술 용어는 원문 표기를 유지\n"
            "- output은 현재 source note의 한줄 요약, 핵심 주장, 문서 타입별 섹션, 한계 섹션에 그대로 들어간다고 가정\n"
            "- summary_line_ko는 1문장으로 쓰고 '이 문서는 ... 설명합니다' 같은 generic opener 금지\n"
            "- contributions/methodology/key_results/limitations/insights는 각각 2~4개 bullet 수준\n"
            f"- 다음 표현 금지: {', '.join(_SOURCE_BANNED_PHRASES[:5])}\n"
            "- metadata를 다시 읽어주는 문장 대신 claim, evidence, limitation을 우선 정리"
        )
        context = (
            f"Title: {title_en}\n"
            f"Entities: {', '.join(entity_names[:8])}\n"
            f"Relations: {'; '.join(relation_lines[:6])}\n\n"
            f"{_truncate(content_text, 3200)}"
        )
        raw, warnings = self._run_task(
            task_type="materialization_summary",
            prompt=prompt,
            context=context,
            allow_external=allow_external,
            max_tokens=1200,
        )
        return self._safe_json_object(raw), warnings

    def _structured_concept_summary(
        self,
        *,
        canonical_name: str,
        aliases: list[str],
        support_lines: list[str],
        relation_lines: list[str],
        key_excerpts_en: list[str],
        allow_external: bool,
    ) -> tuple[dict[str, Any], list[str]]:
        prompt = (
            "다음 AI 개념을 한국어 통합 노트로 정리하세요. 엄격한 JSON object만 반환하세요.\n"
            "Schema:\n"
            "{"
            '"summary_line_ko":"string",'
            '"core_summary":"string",'
            '"why_it_matters":["string"]'
            "}\n"
            "Rules:\n"
            "- 기술 용어는 원문 표기를 유지\n"
            "- 여러 문서의 공통 의미를 통합해서 설명\n"
            "- why_it_matters는 2~4개\n"
            "- summary_line_ko는 1문장, 막연한 표현 금지\n"
            f"- 다음 표현 금지: {', '.join(_CONCEPT_BANNED_PHRASES)}"
        )
        context = (
            f"Concept: {canonical_name}\n"
            f"Aliases: {', '.join(aliases[:8])}\n"
            f"Support: {'; '.join(support_lines[:5])}\n"
            f"Relations: {'; '.join(relation_lines[:8])}\n\n"
            f"{_truncate(' '.join(key_excerpts_en[:2]), 2400)}"
        )
        raw, warnings = self._run_task(
            task_type="materialization_summary",
            prompt=prompt,
            context=context,
            allow_external=allow_external,
            max_tokens=1000,
        )
        return self._safe_json_object(raw), warnings

    def build_source_summary(
        self,
        *,
        title_en: str,
        content_text: str,
        entity_names: list[str],
        relation_lines: list[str],
        key_excerpts_en: list[str],
        candidate_score: float,
        translation_level: str,
    ) -> dict[str, Any]:
        structured, warnings = self._structured_source_summary(
            title_en=title_en,
            entity_names=entity_names,
            relation_lines=relation_lines,
            content_text=content_text,
            allow_external=self.allow_external,
        )
        summary_ko = _clean_summary(str(structured.get("core_summary", "")).strip())
        if _contains_source_banned_phrase(summary_ko):
            summary_ko = ""
        if not summary_ko:
            summary_ko = self._fallback_source_summary(
                title=title_en,
                entity_names=entity_names,
                relation_lines=relation_lines,
                key_excerpts_en=key_excerpts_en,
            )
        summary_ko = _clean_summary(_sanitize_source_banned_phrases(summary_ko))

        title_ko = _clean_title(str(structured.get("title_ko", "")).strip(), title_en)
        summary_line_ko = _extract_summary_line(
            _sanitize_source_banned_phrases(str(structured.get("summary_line_ko", "")).strip()) or summary_ko,
            f"{title_ko}의 핵심 내용을 정리한 문서입니다.",
        )
        if _looks_generic_source_summary_line(summary_line_ko) or _contains_source_banned_phrase(summary_line_ko):
            summary_line_ko = _extract_summary_line(summary_ko, f"{title_ko}의 핵심 주장과 근거를 정리한 문서입니다.")

        excerpts_ko: list[str] = []
        if translation_level == "T2":
            for excerpt in key_excerpts_en[:2]:
                translated, excerpt_warnings = self._run_task(
                    task_type="translation",
                    prompt="다음 기술 문단을 한국어로 정확히 옮기되, 기술 용어는 원문 표기를 유지하세요. 번역문만 출력하세요.",
                    context=_truncate(excerpt, 1200),
                    allow_external=self.allow_external,
                    max_tokens=700,
                )
                warnings.extend(excerpt_warnings)
                excerpts_ko.append(translated or f"핵심 근거: {_truncate(excerpt, 220)}")
        else:
            excerpts_ko = [f"원문 핵심 구절: {_truncate(excerpt, 220)}" for excerpt in key_excerpts_en[:2]]

        contributions = _filter_grounded_source_bullets(
            [str(item).strip() for item in (structured.get("contributions") or []) if str(item).strip()],
            limit=5,
        )
        methodology = _filter_grounded_source_bullets(
            [str(item).strip() for item in (structured.get("methodology") or []) if str(item).strip()],
            limit=5,
        )
        key_results = _filter_grounded_source_bullets(
            [str(item).strip() for item in (structured.get("key_results") or []) if str(item).strip()],
            limit=5,
        )
        limitations = _filter_grounded_source_bullets(
            [str(item).strip() for item in (structured.get("limitations") or []) if str(item).strip()],
            limit=5,
        )
        insights = _filter_grounded_source_bullets(
            [str(item).strip() for item in (structured.get("insights") or []) if str(item).strip()],
            limit=5,
        )

        return {
            "title_ko": title_ko.strip(),
            "summary_ko": summary_ko.strip(),
            "summary_line_ko": summary_line_ko.strip(),
            "core_summary": summary_ko.strip(),
            "contributions": contributions,
            "methodology": methodology,
            "key_results": key_results,
            "limitations": limitations,
            "insights": insights,
            "entity_lines": [f"- {name}" for name in entity_names[:10]],
            "relation_lines": [f"- {line}" for line in relation_lines[:10]],
            "key_excerpts_ko": excerpts_ko,
            "key_excerpts_en": key_excerpts_en[:2],
            "translation_level": translation_level,
            "candidate_score": float(candidate_score or 0.0),
            "warnings": warnings,
        }

    def _structured_source_enrichment(
        self,
        *,
        title_en: str,
        metadata_lines: list[str],
        entity_names: list[str],
        relation_lines: list[str],
        claim_lines: list[str],
        related_concepts: list[str],
        key_excerpts_en: list[str],
        content_text: str,
        evidence_pack: dict[str, Any] | None,
        allow_external: bool,
        target_sections: list[str] | None = None,
        preserve_sections: list[str] | None = None,
        field_diagnostics: dict[str, dict[str, Any]] | None = None,
        review_reasons: list[str] | None = None,
        review_patch_hints: list[str] | None = None,
        remediation_attempt: int = 0,
    ) -> tuple[dict[str, Any], list[str]]:
        document_type = str((evidence_pack or {}).get("document_type") or "method_paper").strip() or "method_paper"
        remediation_active = bool(review_reasons or review_patch_hints or remediation_attempt or target_sections or preserve_sections)
        prompt = (
            "다음 AI/ML 문서를 현재 source note 템플릿에 맞는 한국어 지식 노트로 깊이 있게 확장하세요. "
            "엄격한 JSON object만 반환하세요.\n"
            "Schema:\n"
            "{"
            '"title_ko":"string",'
            '"summary_line_ko":"string",'
            '"core_summary":"string",'
            '"contributions":["string"],'
            '"methodology":["string"],'
            '"key_results":["string"],'
            '"limitations":["string"],'
            '"insights":["string"],'
            '"key_excerpts_ko":["string"]'
            "}\n"
            "Rules:\n"
            "- 기술 용어는 원문 표기를 유지\n"
            "- 과장 금지, 근거 없는 추정 금지\n"
            "- 이 출력은 source note의 한줄 요약, 핵심 주장, 문서 타입별 섹션, 한계 섹션으로 바로 렌더링된다\n"
            "- summary_line_ko는 1문장으로 쓰고 metadata restatement나 '이 문서는 ... 설명합니다' 패턴 금지\n"
            "- top-level thesis를 반복하지 말고 근거 기반 bullet을 우선 작성\n"
            "- contributions/methodology/key_results/limitations/insights는 각각 3~5개 bullet\n"
            "- core_summary는 2~4개의 짧은 단락으로 작성\n"
            "- key_excerpts_ko는 근거 요약 3~4개\n"
            f"- document_type={document_type}를 우선 존중\n"
            "- survey_taxonomy면 taxonomy, comparison frame, gaps, evaluation criteria를 우선 반영\n"
            "- system_card_safety_report면 capability, risk, mitigation, scope, evaluation을 우선 반영\n"
            "- fallback 문장이나 메타 코멘트 대신 실제 내용 bullet을 사용\n"
            "- 이미 충분히 grounded한 섹션은 유지하고, review에서 지적된 약한 부분을 우선 보강\n"
            "- missing/weak section을 메우기 위해 generic filler를 쓰지 말고 claim/excerpt/evidence pack에서 직접 근거를 끌어오세요\n"
            "- section remediation이면 target section 내부의 약한 bullet/문장만 대체하고 preserve section은 유지하세요\n"
            "- preserve된 bullet과 거의 같은 문장을 반복하지 마세요\n"
            "- target section이 비어 있지 않다면 전체 섹션을 다시 쓰기보다 weak slot을 메우는 replacement bullet을 우선 만드세요\n"
            "- target section에 근거가 부족하면 빈말을 채우지 말고 sparse but grounded하게 작성하세요\n"
            f"- 다음 표현 금지: {', '.join(_SOURCE_BANNED_PHRASES[:5])}"
        )
        evidence_json = json.dumps(evidence_pack or {}, ensure_ascii=False, indent=2)
        context = (
            f"Title: {title_en}\n"
            f"Document type: {document_type}\n"
            f"Metadata: {'; '.join(metadata_lines[:6])}\n"
            f"Entities: {', '.join(entity_names[:12])}\n"
            f"Relations: {'; '.join(relation_lines[:12])}\n"
            f"Claims: {'; '.join(claim_lines[:8])}\n"
            f"Related concepts: {', '.join(related_concepts[:12])}\n"
            f"Evidence excerpts: {'; '.join(_truncate(item, 320) for item in key_excerpts_en[:4])}\n\n"
            f"Evidence pack:\n{evidence_json}\n\n"
            f"{_truncate(content_text, 14000)}"
        )
        if remediation_active:
            field_guidance = []
            for key, diagnostic in list((field_diagnostics or {}).items())[:8]:
                weak_count = len([int(item) for item in (diagnostic.get("weak_line_indexes") or [])])
                line_count = int(diagnostic.get("line_count") or 0)
                low_signal_count = int(diagnostic.get("low_signal_line_count") or 0)
                preserved_count = max(0, line_count - weak_count)
                field_guidance.append(
                    f"{key}: weak={weak_count}, preserved={preserved_count}, desired_min=3, low_signal={low_signal_count}"
                )
            context += (
                "\n\nRemediation guidance:\n"
                f"- attempt: {int(remediation_attempt or 0)}\n"
                f"- target_sections: {'; '.join(str(item).strip() for item in (target_sections or [])[:12])}\n"
                f"- preserve_sections: {'; '.join(str(item).strip() for item in (preserve_sections or [])[:12])}\n"
                f"- review_reasons: {'; '.join(str(item).strip() for item in (review_reasons or [])[:6])}\n"
                f"- review_patch_hints: {'; '.join(str(item).strip() for item in (review_patch_hints or [])[:6])}\n"
                f"- field_diagnostics: {'; '.join(field_guidance)}\n"
            )
        raw, warnings = self._run_task(
            task_type="materialization_source_enrichment",
            prompt=prompt,
            context=context,
            allow_external=allow_external,
            max_tokens=2200,
        )
        return self._safe_json_object(raw), warnings

    def build_source_enrichment(
        self,
        *,
        title_en: str,
        content_text: str,
        entity_names: list[str],
        relation_lines: list[str],
        claim_lines: list[str],
        related_concepts: list[str],
        key_excerpts_en: list[str],
        metadata_lines: list[str],
        evidence_pack: dict[str, Any] | None = None,
        candidate_score: float,
        translation_level: str,
        minimum_bullets_per_section: int = 3,
        target_sections: list[str] | None = None,
        preserve_sections: list[str] | None = None,
        field_diagnostics: dict[str, dict[str, Any]] | None = None,
        review_reasons: list[str] | None = None,
        review_patch_hints: list[str] | None = None,
        remediation_attempt: int = 0,
    ) -> dict[str, Any]:
        structured, warnings = self._structured_source_enrichment(
            title_en=title_en,
            metadata_lines=metadata_lines,
            entity_names=entity_names,
            relation_lines=relation_lines,
            claim_lines=claim_lines,
            related_concepts=related_concepts,
            key_excerpts_en=key_excerpts_en,
            content_text=content_text,
            evidence_pack=evidence_pack,
            allow_external=self.allow_external,
            target_sections=target_sections,
            preserve_sections=preserve_sections,
            field_diagnostics=field_diagnostics,
            review_reasons=review_reasons,
            review_patch_hints=review_patch_hints,
            remediation_attempt=remediation_attempt,
        )
        pack = dict(evidence_pack or {})
        grounded_sections = _build_grounded_source_sections(
            metadata_lines=metadata_lines,
            entity_names=entity_names,
            relation_lines=relation_lines,
            claim_lines=claim_lines,
            related_concepts=related_concepts,
            key_excerpts_en=key_excerpts_en,
            content_text=content_text,
        )
        synthesized_sections = synthesize_evidence_sections(
            document_type=str(pack.get("document_type") or "method_paper"),
            title=title_en,
            thesis=str(pack.get("thesis") or ""),
            content_text=content_text,
            entity_names=entity_names,
            relation_lines=relation_lines,
            claim_lines=claim_lines,
            related_concepts=related_concepts,
            key_excerpts=key_excerpts_en,
            metadata_lines=metadata_lines,
        )
        if not structured:
            base = self.build_source_summary(
                title_en=title_en,
                content_text=content_text,
                entity_names=entity_names,
                relation_lines=relation_lines,
                key_excerpts_en=key_excerpts_en,
                candidate_score=candidate_score,
                translation_level=translation_level,
            )
            thesis = str(pack.get("thesis") or "") or _extract_summary_line(base.get("summary_ko") or "", f"{title_en}의 핵심 논지를 정리한 문서입니다.")
            summary_ko = _clean_summary(_sanitize_source_banned_phrases(str(base.get("summary_ko") or "").strip())) or thesis
            if _looks_generic_source_summary_line(summary_ko) and thesis:
                summary_ko = thesis
            summary_line = _extract_summary_line(
                thesis if thesis else str(base.get("summary_line_ko") or "").strip(),
                f"{title_en}의 핵심 논지를 정리한 문서입니다.",
            )
            base["contributions"] = _ensure_bullets(
                _merge_bullets(
                    _evidence_list(pack, "contributions"),
                    _evidence_list(synthesized_sections, "contributions"),
                    list(base.get("contributions") or []),
                    grounded_sections["contributions"],
                    limit=max(minimum_bullets_per_section, 5),
                ),
                minimum=minimum_bullets_per_section,
                fallback=f"{title_en}의 핵심 기여는 추출된 claim과 근거 문단을 함께 읽어 정리해야 합니다.",
            )
            base["methodology"] = _ensure_bullets(
                _merge_bullets(
                    _evidence_list(pack, "methodology"),
                    _evidence_list(synthesized_sections, "methodology"),
                    list(base.get("methodology") or []),
                    grounded_sections["methodology"],
                    limit=max(minimum_bullets_per_section, 5),
                ),
                minimum=minimum_bullets_per_section,
                fallback=f"{title_en}의 접근 방식은 relation과 evidence excerpt를 함께 읽어야 선명해집니다.",
            )
            base["key_results"] = _ensure_bullets(
                _merge_bullets(
                    _evidence_list(pack, "results_or_findings"),
                    _evidence_list(synthesized_sections, "results_or_findings"),
                    list(base.get("key_results") or []),
                    grounded_sections["key_results"],
                    limit=max(minimum_bullets_per_section, 5),
                ),
                minimum=minimum_bullets_per_section,
                fallback=f"{title_en}의 결과 해석은 claim과 핵심 근거를 함께 확인해야 합니다.",
            )
            base["limitations"] = _ensure_bullets(
                _merge_bullets(
                    _evidence_list(pack, "limitations"),
                    _evidence_list(synthesized_sections, "limitations"),
                    list(base.get("limitations") or []),
                    grounded_sections["limitations"],
                    limit=max(minimum_bullets_per_section, 5),
                ),
                minimum=minimum_bullets_per_section,
                fallback=f"{title_en}의 적용 범위와 한계는 원문 근거를 추가로 검토해야 합니다.",
            )
            base["insights"] = _ensure_bullets(
                _merge_bullets(
                    _evidence_list(pack, "insights"),
                    _evidence_list(synthesized_sections, "insights"),
                    list(base.get("insights") or []),
                    grounded_sections["insights"],
                    limit=max(minimum_bullets_per_section, 5),
                ),
                minimum=minimum_bullets_per_section,
                fallback=f"{title_en}는 관련 개념과 후속 문서를 연결하는 출발점으로 볼 수 있습니다.",
            )
            base["summary_ko"] = summary_ko
            base["summary_line_ko"] = summary_line
            base["core_summary"] = summary_ko
            base["document_type"] = str(pack.get("document_type") or "method_paper")
            base["thesis"] = thesis
            base["top_claims"] = _filter_grounded_source_bullets(
                _evidence_list(pack, "top_claims") or synthesized_sections["top_claims"],
                limit=4,
            )
            base["core_concepts"] = list(pack.get("core_concepts") or []) or extract_core_concepts(entity_names, related_concepts)
            base["results_or_findings"] = list(base["key_results"])
            base["representative_sources"] = list(pack.get("representative_sources") or representative_sources(title=title_en, source_url="", domain="", metadata_lines=metadata_lines))
            base["needs_reinforcement"] = bool(_detect_fallback_heavy(base["contributions"], base["methodology"], base["key_results"], base["limitations"], base["insights"]))
            base["reinforcement_reasons"] = _detect_fallback_heavy(base["contributions"], base["methodology"], base["key_results"], base["limitations"], base["insights"])
            base["warnings"] = [*list(base.get("warnings") or []), *warnings]
            return base

        summary_ko = _clean_summary(_sanitize_source_banned_phrases(str(structured.get("core_summary", "")).strip()))
        if not summary_ko:
            summary_ko = str(pack.get("thesis") or "").strip() or self._fallback_summary(title_en, content_text, max_sentences=6)
        if _looks_generic_source_summary_line(summary_ko) and str(pack.get("thesis") or "").strip():
            summary_ko = str(pack.get("thesis") or "").strip()
        title_ko = _clean_title(str(structured.get("title_ko", "")).strip(), title_en)
        summary_line_ko = _extract_summary_line(
            str(pack.get("thesis") or "").strip()
            or ("" if _looks_generic_source_summary_line(str(structured.get("summary_line_ko", "")).strip()) else _sanitize_source_banned_phrases(str(structured.get("summary_line_ko", "")).strip()))
            or summary_ko,
            f"{title_ko}의 핵심 내용을 구조적으로 정리한 노트입니다.",
        )
        contributions = _ensure_bullets(
            _filter_grounded_source_bullets(_merge_bullets(
                _evidence_list(pack, "contributions"),
                _evidence_list(synthesized_sections, "contributions"),
                [str(item).strip() for item in (structured.get("contributions") or []) if str(item).strip()],
                grounded_sections["contributions"],
                limit=max(minimum_bullets_per_section, 5),
            ), limit=max(minimum_bullets_per_section, 5)),
            minimum=minimum_bullets_per_section,
            fallback=f"{title_en}의 핵심 기여는 claim과 evidence excerpt를 함께 확인해야 합니다.",
        )
        methodology = _ensure_bullets(
            _filter_grounded_source_bullets(_merge_bullets(
                _evidence_list(pack, "methodology"),
                _evidence_list(synthesized_sections, "methodology"),
                [str(item).strip() for item in (structured.get("methodology") or []) if str(item).strip()],
                grounded_sections["methodology"],
                limit=max(minimum_bullets_per_section, 5),
            ), limit=max(minimum_bullets_per_section, 5)),
            minimum=minimum_bullets_per_section,
            fallback=f"{title_en}의 접근 방식은 relation과 본문 근거를 함께 읽어 정리해야 합니다.",
        )
        key_results = _ensure_bullets(
            _filter_grounded_source_bullets(_merge_bullets(
                _evidence_list(pack, "results_or_findings"),
                _evidence_list(synthesized_sections, "results_or_findings"),
                [str(item).strip() for item in (structured.get("key_results") or []) if str(item).strip()],
                grounded_sections["key_results"],
                limit=max(minimum_bullets_per_section, 5),
            ), limit=max(minimum_bullets_per_section, 5)),
            minimum=minimum_bullets_per_section,
            fallback=f"{title_en}의 결과는 claim과 대표 근거를 기준으로 해석해야 합니다.",
        )
        limitations = _ensure_bullets(
            _filter_grounded_source_bullets(_merge_bullets(
                _evidence_list(pack, "limitations"),
                _evidence_list(synthesized_sections, "limitations"),
                [str(item).strip() for item in (structured.get("limitations") or []) if str(item).strip()],
                grounded_sections["limitations"],
                limit=max(minimum_bullets_per_section, 5),
            ), limit=max(minimum_bullets_per_section, 5)),
            minimum=minimum_bullets_per_section,
            fallback=f"{title_en}의 한계는 적용 범위와 근거 문단을 함께 검토해야 합니다.",
        )
        insights = _ensure_bullets(
            _filter_grounded_source_bullets(_merge_bullets(
                _evidence_list(pack, "insights"),
                _evidence_list(synthesized_sections, "insights"),
                [str(item).strip() for item in (structured.get("insights") or []) if str(item).strip()],
                grounded_sections["insights"],
                limit=max(minimum_bullets_per_section, 5),
            ), limit=max(minimum_bullets_per_section, 5)),
            minimum=minimum_bullets_per_section,
            fallback=f"{title_en}는 관련 개념과 후속 문서를 연결하는 핵심 참고 문서입니다.",
        )
        excerpts_ko = [str(item).strip() for item in (structured.get("key_excerpts_ko") or []) if str(item).strip()]
        if len(excerpts_ko) < 3:
            excerpts_ko.extend(
                [f"원문 핵심 구절: {_truncate(excerpt, 260)}" for excerpt in key_excerpts_en[: max(0, 3 - len(excerpts_ko))]]
            )
        reinforcement_reasons = _detect_fallback_heavy(contributions, methodology, key_results, limitations, insights)
        return {
            "title_ko": title_ko.strip(),
            "summary_ko": summary_ko.strip(),
            "summary_line_ko": summary_line_ko.strip(),
            "core_summary": summary_ko.strip(),
            "document_type": str(pack.get("document_type") or "method_paper"),
            "thesis": str(pack.get("thesis") or summary_line_ko).strip(),
            "top_claims": _filter_grounded_source_bullets(
                _evidence_list(pack, "top_claims") or synthesized_sections["top_claims"],
                limit=4,
            ),
            "core_concepts": list(pack.get("core_concepts") or []) or extract_core_concepts(entity_names, related_concepts),
            "contributions": contributions,
            "methodology": methodology,
            "key_results": key_results,
            "results_or_findings": key_results,
            "limitations": limitations,
            "insights": insights,
            "representative_sources": list(pack.get("representative_sources") or representative_sources(title=title_en, source_url="", domain="", metadata_lines=metadata_lines)),
            "entity_lines": [f"- {name}" for name in entity_names[:12]],
            "relation_lines": [f"- {line}" for line in relation_lines[:12]],
            "key_excerpts_ko": excerpts_ko[:4],
            "key_excerpts_en": key_excerpts_en[:4],
            "translation_level": translation_level,
            "candidate_score": float(candidate_score or 0.0),
            "needs_reinforcement": bool(reinforcement_reasons),
            "reinforcement_reasons": reinforcement_reasons,
            "warnings": warnings,
        }

    def compress_source_evidence_for_concept(
        self,
        *,
        source_title: str,
        source_url: str,
        domain: str,
        summary_text: str,
        relation_lines: list[str],
        key_excerpts_en: list[str],
    ) -> tuple[dict[str, Any], list[str]]:
        prompt = (
            "다음 문서를 AI 개념 노트용 근거 카드로 압축하세요. 엄격한 JSON object만 반환하세요.\n"
            "Schema:\n"
            "{"
            '"source_title":"string",'
            '"summary_line_ko":"string",'
            '"facts":["string"],'
            '"why_relevant":"string"'
            "}\n"
            "Rules:\n"
            "- facts는 2~3개\n"
            "- 기술 용어는 원문 유지\n"
            "- 과장 금지"
        )
        context = (
            f"Title: {source_title}\n"
            f"URL: {source_url}\n"
            f"Domain: {domain}\n"
            f"Relations: {'; '.join(relation_lines[:6])}\n"
            f"Summary: {_truncate(summary_text, 1200)}\n"
            f"Excerpts: {'; '.join(_truncate(item, 220) for item in key_excerpts_en[:2])}"
        )
        raw, warnings = self._run_task(
            task_type="materialization_source_enrichment",
            prompt=prompt,
            context=context,
            allow_external=self.allow_external,
            force_route="mini",
            max_tokens=700,
        )
        parsed = self._safe_json_object(raw)
        if not parsed:
            parsed = {
                "source_title": source_title,
                "summary_line_ko": _extract_summary_line(summary_text, f"{source_title} 요약"),
                "facts": [
                    f"관련 관계: {relation_lines[0]}" if relation_lines else "관련 개념 연결이 확인된 문서입니다.",
                    f"도메인: {domain}" if domain else "출처 도메인을 확인해야 합니다.",
                ],
                "why_relevant": f"{source_title}는 개념 통합의 근거 문서로 사용할 수 있습니다.",
            }
        return parsed, warnings

    def build_concept_summary(
        self,
        *,
        canonical_name: str,
        aliases: list[str],
        support_lines: list[str],
        relation_lines: list[str],
        key_excerpts_en: list[str],
        candidate_score: float,
        translation_level: str,
    ) -> dict[str, Any]:
        structured, warnings = self._structured_concept_summary(
            canonical_name=canonical_name,
            aliases=aliases,
            support_lines=support_lines,
            relation_lines=relation_lines,
            key_excerpts_en=key_excerpts_en,
            allow_external=self.allow_external,
        )
        summary_ko = _clean_summary(str(structured.get("core_summary", "")).strip())
        if _contains_banned_phrase(summary_ko):
            summary_ko = ""
        if not summary_ko:
            summary_ko = self._fallback_concept_summary(
                canonical_name=canonical_name,
                support_lines=support_lines,
                relation_lines=relation_lines,
                key_excerpts_en=key_excerpts_en,
            )
        summary_ko = _clean_summary(_sanitize_banned_phrases(summary_ko))

        title_ko = canonical_name
        summary_line_ko = _extract_summary_line(
            _sanitize_banned_phrases(str(structured.get("summary_line_ko", "")).strip()) or summary_ko,
            f"{canonical_name} 개념 요약입니다.",
        )
        if _looks_generic_concept_summary_line(summary_line_ko, canonical_name) or _contains_banned_phrase(summary_line_ko):
            summary_line_ko = _extract_summary_line(summary_ko, f"{canonical_name}는 여러 문서에서 반복 등장하는 핵심 개념입니다.")

        excerpts_ko: list[str] = []
        if translation_level == "T2":
            for excerpt in key_excerpts_en[:2]:
                translated, excerpt_warnings = self._run_task(
                    task_type="translation",
                    prompt="다음 근거 문장을 한국어로 옮기세요. 기술 용어는 유지하세요.",
                    context=_truncate(excerpt, 1200),
                    allow_external=self.allow_external,
                    max_tokens=700,
                )
                warnings.extend(excerpt_warnings)
                excerpts_ko.append(translated or f"핵심 근거: {_truncate(excerpt, 220)}")
        else:
            excerpts_ko = [f"원문 핵심 구절: {_truncate(excerpt, 220)}" for excerpt in key_excerpts_en[:2]]

        why_it_matters = _filter_grounded_bullets(
            [str(item).strip() for item in (structured.get("why_it_matters") or []) if str(item).strip()],
            limit=4,
        )

        return {
            "title_ko": title_ko.strip(),
            "summary_ko": summary_ko.strip(),
            "summary_line_ko": summary_line_ko.strip(),
            "core_summary": summary_ko.strip(),
            "why_it_matters": why_it_matters,
            "aliases": aliases,
            "support_lines": [f"- {line}" for line in support_lines[:10]],
            "relation_lines": [f"- {line}" for line in relation_lines[:10]],
            "key_excerpts_ko": excerpts_ko,
            "key_excerpts_en": key_excerpts_en[:2],
            "translation_level": translation_level,
            "candidate_score": float(candidate_score or 0.0),
            "warnings": warnings,
        }

    def _structured_concept_enrichment(
        self,
        *,
        canonical_name: str,
        aliases: list[str],
        relation_lines: list[str],
        related_concepts: list[str],
        compressed_support_docs: list[dict[str, Any]],
        existing_summary: str,
        allow_external: bool,
        target_sections: list[str] | None = None,
        preserve_sections: list[str] | None = None,
        field_diagnostics: dict[str, dict[str, Any]] | None = None,
        review_reasons: list[str] | None = None,
        review_patch_hints: list[str] | None = None,
        remediation_attempt: int = 0,
    ) -> tuple[dict[str, Any], list[str]]:
        remediation_active = bool(review_reasons or review_patch_hints or remediation_attempt or target_sections or preserve_sections)
        prompt = (
            "다음 AI 개념을 여러 문서 근거를 통합해 현재 concept note 렌더 구조에 맞는 한국어 개념 노트로 심화하세요. "
            "엄격한 JSON object만 반환하세요.\n"
            "Schema:\n"
            "{"
            '"summary_line_ko":"string",'
            '"core_summary":"string",'
            '"why_it_matters":["string"],'
            '"relation_lines":["string"],'
            '"evidence_synthesis":["string"],'
            '"related_sources":["string"],'
            '"related_concepts":["string"]'
            "}\n"
            "Rules:\n"
            "- core_summary는 현재 노트의 핵심 구조/정의 섹션에 들어갈 2~4단락 설명\n"
            "- summary_line_ko는 1문장, 120자 내외, 막연한 문구 금지\n"
            "- why_it_matters/relation_lines/evidence_synthesis는 각각 3~5개\n"
            "- why_it_matters는 왜 중요한지와 어디에 연결되는지를 근거 기반으로 설명\n"
            "- relation_lines는 실제 관계/의존/영향이 드러나는 구체 문장 우선\n"
            "- related_sources는 source_title 또는 왜 relevant한지를 반영한 근거 중심 문장으로 작성\n"
            "- related_concepts는 이 개념과 구분되거나 함께 읽어야 할 인접 개념만 포함\n"
            "- 기술 용어는 원문 유지\n"
            "- 서로 다른 근거 문서의 공통점과 차이를 드러내기\n"
            "- 기존 요약에서 이미 grounded한 설명은 유지하고, review에서 지적된 약한 부분을 우선 보강\n"
            "- missing support/relation/source 섹션은 compressed support docs의 근거를 직접 사용해 메우기\n"
            "- section remediation이면 target section 내부의 약한 bullet/문장만 대체하고 preserve section은 그대로 유지하세요\n"
            "- preserve된 bullet과 거의 같은 문장을 반복하지 마세요\n"
            "- target section이 비어 있지 않다면 전체 섹션을 다시 쓰기보다 weak slot을 메우는 replacement bullet을 우선 만드세요\n"
            "- target section에 근거가 부족하면 과장하지 말고 sparse but grounded하게 작성하세요\n"
            f"- 다음 표현 금지: {', '.join(_CONCEPT_BANNED_PHRASES)}"
        )
        support_json = json.dumps(compressed_support_docs[:8], ensure_ascii=False, indent=2)
        context = (
            f"Concept: {canonical_name}\n"
            f"Aliases: {', '.join(aliases[:10])}\n"
            f"Existing summary: {_truncate(existing_summary, 1800)}\n"
            f"Relations: {'; '.join(relation_lines[:12])}\n"
            f"Related concepts: {', '.join(related_concepts[:12])}\n"
            f"Compressed support docs:\n{support_json}"
        )
        if remediation_active:
            field_guidance = []
            for key, diagnostic in list((field_diagnostics or {}).items())[:8]:
                weak_count = len([int(item) for item in (diagnostic.get("weak_line_indexes") or [])])
                line_count = int(diagnostic.get("line_count") or 0)
                low_signal_count = int(diagnostic.get("low_signal_line_count") or 0)
                preserved_count = max(0, line_count - weak_count)
                field_guidance.append(
                    f"{key}: weak={weak_count}, preserved={preserved_count}, desired_min=3, low_signal={low_signal_count}"
                )
            context += (
                "\n\nRemediation guidance:\n"
                f"- attempt: {int(remediation_attempt or 0)}\n"
                f"- target_sections: {'; '.join(str(item).strip() for item in (target_sections or [])[:12])}\n"
                f"- preserve_sections: {'; '.join(str(item).strip() for item in (preserve_sections or [])[:12])}\n"
                f"- review_reasons: {'; '.join(str(item).strip() for item in (review_reasons or [])[:6])}\n"
                f"- review_patch_hints: {'; '.join(str(item).strip() for item in (review_patch_hints or [])[:6])}\n"
                f"- field_diagnostics: {'; '.join(field_guidance)}\n"
            )
        raw, warnings = self._run_task(
            task_type="materialization_concept_enrichment",
            prompt=prompt,
            context=context,
            allow_external=allow_external,
            max_tokens=2200,
        )
        return self._safe_json_object(raw), warnings

    def build_concept_enrichment(
        self,
        *,
        canonical_name: str,
        aliases: list[str],
        relation_lines: list[str],
        related_concepts: list[str],
        compressed_support_docs: list[dict[str, Any]],
        existing_summary: str,
        candidate_score: float,
        translation_level: str,
        minimum_support_docs: int = 4,
        target_sections: list[str] | None = None,
        preserve_sections: list[str] | None = None,
        field_diagnostics: dict[str, dict[str, Any]] | None = None,
        review_reasons: list[str] | None = None,
        review_patch_hints: list[str] | None = None,
        remediation_attempt: int = 0,
    ) -> dict[str, Any]:
        structured, warnings = self._structured_concept_enrichment(
            canonical_name=canonical_name,
            aliases=aliases,
            relation_lines=relation_lines,
            related_concepts=related_concepts,
            compressed_support_docs=compressed_support_docs[: max(minimum_support_docs, 8)],
            existing_summary=existing_summary,
            allow_external=self.allow_external,
            target_sections=target_sections,
            preserve_sections=preserve_sections,
            field_diagnostics=field_diagnostics,
            review_reasons=review_reasons,
            review_patch_hints=review_patch_hints,
            remediation_attempt=remediation_attempt,
        )
        if not structured:
            base = self.build_concept_summary(
                canonical_name=canonical_name,
                aliases=aliases,
                support_lines=[str(item.get("source_title") or "") for item in compressed_support_docs if str(item.get("source_title") or "").strip()],
                relation_lines=relation_lines,
                key_excerpts_en=[
                    str(item.get("summary_line_ko") or "")
                    for item in compressed_support_docs
                    if str(item.get("summary_line_ko") or "").strip()
                ],
                candidate_score=candidate_score,
                translation_level=translation_level,
            )
            return {
                **base,
                "related_sources": [
                    f"- {item.get('source_title')} - {item.get('why_relevant')}"
                    for item in compressed_support_docs[:8]
                    if str(item.get("source_title") or "").strip()
                ],
            }

        summary_ko = _clean_summary(str(structured.get("core_summary", "")).strip())
        if _contains_banned_phrase(summary_ko):
            summary_ko = ""
        if not summary_ko:
            summaries = [str(item.get("summary_line_ko") or "").strip() for item in compressed_support_docs if str(item.get("summary_line_ko") or "").strip()]
            summary_ko = "\n\n".join(summaries[:3]) or self._fallback_concept_summary(
                canonical_name=canonical_name,
                support_lines=[str(item.get("source_title") or "") for item in compressed_support_docs],
                relation_lines=relation_lines,
                key_excerpts_en=[str(item.get("summary_line_ko") or "") for item in compressed_support_docs],
            )
        summary_ko = _clean_summary(_sanitize_banned_phrases(summary_ko))
        summary_line_ko = _extract_summary_line(
            _sanitize_banned_phrases(str(structured.get("summary_line_ko", "")).strip()) or summary_ko,
            f"{canonical_name} 개념을 여러 근거에서 통합한 요약입니다.",
        )
        if _looks_generic_concept_summary_line(summary_line_ko, canonical_name) or _contains_banned_phrase(summary_line_ko):
            summary_line_ko = _extract_summary_line(summary_ko, f"{canonical_name}는 여러 문서에서 반복 등장하는 핵심 개념입니다.")
        why_it_matters = _ensure_bullets(
            _filter_grounded_bullets(
                [str(item).strip() for item in (structured.get("why_it_matters") or []) if str(item).strip()],
                limit=4,
            )
            or _grounded_concept_importance(
                canonical_name=canonical_name,
                relation_lines=relation_lines,
                related_concepts=related_concepts,
                compressed_support_docs=compressed_support_docs,
            ),
            minimum=3,
            fallback=f"{canonical_name}는 여러 문서에서 반복 등장하며 다른 개념과의 연결이 확인됩니다.",
        )
        rel_lines = _ensure_bullets(
            _filter_grounded_bullets(
                [str(item).strip() for item in (structured.get("relation_lines") or []) if str(item).strip()],
                limit=6,
            )
            or _grounded_concept_relations(relation_lines, compressed_support_docs),
            minimum=3,
            fallback="관계 해석은 원문 근거와 온톨로지 연결을 함께 확인해야 합니다.",
        )
        evidence_synthesis = _ensure_bullets(
            _filter_grounded_bullets(
                [str(item).strip() for item in (structured.get("evidence_synthesis") or []) if str(item).strip()],
                limit=5,
            ),
            minimum=3,
            fallback="대표 근거 문서를 함께 읽어 개념의 범위를 확인할 필요가 있습니다.",
        )
        related_sources = [
            f"- {item}"
            for item in _filter_grounded_bullets(
                [str(value).strip() for value in (structured.get("related_sources") or []) if str(value).strip()],
                limit=8,
            )
        ]
        if not related_sources:
            related_sources = [
                f"- {item}"
                for item in _grounded_related_sources(compressed_support_docs)
            ]
        related_concept_lines = [
            f"- {item}" if not str(item).strip().startswith("- ") else str(item).strip()
            for item in _filter_grounded_bullets(
                [str(value).strip() for value in (structured.get("related_concepts") or []) if str(value).strip()],
                limit=12,
            )
        ]
        if not related_concept_lines:
            related_concept_lines = [f"- {item}" for item in _grounded_related_concepts(related_concepts)]
        support_lines = [
            f"- {item.get('source_title')} ({item.get('why_relevant')})"
            for item in compressed_support_docs[:8]
            if str(item.get("source_title") or "").strip()
        ]
        return {
            "title_ko": canonical_name.strip(),
            "summary_ko": summary_ko.strip(),
            "summary_line_ko": summary_line_ko.strip(),
            "core_summary": summary_ko.strip(),
            "why_it_matters": why_it_matters,
            "aliases": aliases,
            "support_lines": support_lines,
            "relation_lines": rel_lines,
            "key_excerpts_ko": evidence_synthesis[:5],
            "key_excerpts_en": [str(item.get("summary_line_ko") or "") for item in compressed_support_docs[:4]],
            "translation_level": translation_level,
            "candidate_score": float(candidate_score or 0.0),
            "warnings": warnings,
            "related_sources": related_sources,
            "related_concepts": related_concept_lines,
        }
