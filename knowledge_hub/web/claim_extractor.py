"""Web claim extraction helpers using deterministic priors + optional LLM refinement."""

from __future__ import annotations

import re
from typing import Any

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.core.models import ClaimCandidate
from knowledge_hub.core.sanitizer import redact_payload
from knowledge_hub.learning.task_router import get_llm_for_task
from knowledge_hub.learning.resolver import normalize_term
from knowledge_hub.papers.claim_extractor import extract_claim_candidates as extract_llm_claim_candidates

_CLAIM_PATTERNS: list[tuple[re.Pattern[str], str, float]] = [
    (re.compile(r"\b(require|requires|required|need|needs|prerequisite)\b|필요|요구"), "requires", 0.78),
    (re.compile(r"\b(enable|enables|enabled|allow|allows)\b|가능하게|지원"), "enables", 0.76),
    (re.compile(r"\b(improve|improves|improved|boost|enhance|enhances|better)\b|향상|개선"), "improves", 0.8),
    (re.compile(r"\b(reduce|reduces|reduced|mitigate|mitigates)\b|감소|줄인다"), "reduces", 0.78),
    (re.compile(r"\b(cause|causes|caused|lead to|leads to|trigger)\b|원인|유발"), "causes", 0.78),
    (re.compile(r"\b(use|uses|used|based on|builds on)\b|사용|기반"), "uses", 0.72),
]


def _split_sentences(text: str) -> list[str]:
    return [token.strip() for token in re.split(r"(?<=[.!?])\s+|\n+", text or "") if token and token.strip()]


def _find_mentions(sentence: str, related_entities: list[dict[str, Any]]) -> list[tuple[str, int]]:
    normalized_sentence = normalize_term(sentence)
    mentions: list[tuple[str, int]] = []
    for entity in related_entities:
        canonical_id = str(entity.get("canonical_id") or entity.get("entity_id") or "").strip()
        if not canonical_id:
            continue
        names = [str(entity.get("display_name") or "").strip()]
        names.extend(str(alias).strip() for alias in (entity.get("aliases") or []) if str(alias).strip())
        best_pos: int | None = None
        for name in names:
            needle = normalize_term(name)
            if not needle:
                continue
            pos = normalized_sentence.find(needle)
            if pos >= 0 and (best_pos is None or pos < best_pos):
                best_pos = pos
        if best_pos is not None:
            mentions.append((canonical_id, best_pos))
    mentions.sort(key=lambda item: item[1])
    return mentions


def _deterministic_candidates(
    *,
    title: str,
    text: str,
    related_entities: list[dict[str, Any]],
    max_claims: int,
) -> list[ClaimCandidate]:
    candidates: list[ClaimCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for sentence in _split_sentences(text):
        normalized_sentence = normalize_term(sentence)
        if len(normalized_sentence) < 20:
            continue
        predicate = ""
        confidence = 0.0
        for pattern, relation, base_conf in _CLAIM_PATTERNS:
            if pattern.search(normalized_sentence):
                predicate = relation
                confidence = base_conf
                break
        if not predicate:
            continue
        mentions = _find_mentions(sentence, related_entities)
        if len(mentions) < 2:
            continue
        subject_id = mentions[0][0]
        object_id = mentions[1][0]
        key = (subject_id, predicate, object_id)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            ClaimCandidate(
                claim_text=sentence[:600],
                subject=subject_id,
                predicate=predicate,
                object_value=object_id,
                evidence=sentence[:600],
                llm_confidence=confidence,
            )
        )
        if len(candidates) >= max(1, int(max_claims)):
            break
    return candidates


def extract_web_claim_candidates(
    *,
    config: Config | None,
    title: str,
    text: str,
    source_metadata: dict[str, Any] | None,
    related_entities: list[dict[str, Any]],
    allow_external: bool,
    max_claims: int = 6,
    max_input_chars: int = 3500,
) -> tuple[list[ClaimCandidate], list[str]]:
    deterministic = _deterministic_candidates(
        title=title,
        text=text,
        related_entities=related_entities,
        max_claims=max_claims,
    )
    warnings: list[str] = []
    if not allow_external or config is None:
        return deterministic, warnings

    excerpt = str(text or "")[: max(1, int(max_input_chars))]
    sanitized_context = redact_payload(
        {
            "title": title,
            "source": source_metadata or {},
            "related_entities": [
                {
                    "canonical_id": str(item.get("canonical_id") or item.get("entity_id") or ""),
                    "display_name": str(item.get("display_name") or ""),
                }
                for item in related_entities[:12]
            ],
            "excerpt": excerpt,
        }
    )
    llm, _decision, route_warnings = get_llm_for_task(
        config,
        task_type="claim_extraction",
        allow_external=allow_external,
        query=str(title or ""),
        context=str(sanitized_context),
        source_count=max(1, len(related_entities)),
    )
    warnings.extend(route_warnings)
    if llm is None:
        return deterministic, warnings

    try:
        llm_candidates = extract_llm_claim_candidates(
            llm,
            title=str(title or ""),
            text=str(sanitized_context),
            max_claims=max_claims,
            max_input_chars=max_input_chars,
        )
    except Exception as error:
        warnings.append(f"claim extraction fallback used: {error}")
        return deterministic, warnings

    merged: list[ClaimCandidate] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for candidate in [*llm_candidates, *deterministic]:
        key = (
            str(candidate.subject or "").strip(),
            str(candidate.predicate or "").strip(),
            str(candidate.object_value or "").strip(),
        )
        if not key[0] or not key[1] or key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(candidate)
        if len(merged) >= max(1, int(max_claims)):
            break
    return merged or deterministic, warnings
