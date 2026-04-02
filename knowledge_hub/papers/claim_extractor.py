"""Paper claim extraction helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from knowledge_hub.core.models import ClaimCandidate

_SPECIFIC_SIGNAL_RE = re.compile(
    r"\b\d+(?:\.\d+)?(?:%|x|ms|s|m|k|b)?\b|"
    r"\b(?:accuracy|precision|recall|f1|auc|bleu|rouge|benchmark|dataset|baseline|experiment|ablation|"
    r"latency|throughput|error rate|top[- ]?\d+|compared to|versus|vs)\b",
    re.IGNORECASE,
)
_GENERIC_CLAIM_TOKENS = {
    "approach",
    "cases",
    "capability",
    "effective",
    "generally",
    "helpful",
    "helps",
    "important",
    "improvement",
    "many",
    "method",
    "methods",
    "often",
    "overall",
    "potentially",
    "promising",
    "results",
    "several",
    "sometimes",
    "solution",
    "strategy",
    "system",
    "systems",
    "technique",
    "typically",
    "useful",
    "various",
    "경우",
    "대체로",
    "때때로",
    "방법",
    "방식",
    "시스템",
    "여러",
    "일반적",
}
_HEDGE_TOKENS = {
    "can",
    "could",
    "frequently",
    "generally",
    "inconsistent",
    "may",
    "might",
    "often",
    "possibly",
    "sometimes",
    "typically",
    "varies",
    "경우에",
    "때때로",
    "불안정",
    "일반적으로",
    "종종",
}
_CONTRADICTION_CUES = {
    "although",
    "but",
    "conflict",
    "conflicting",
    "depends",
    "despite",
    "however",
    "inconsistent",
    "mixed",
    "nevertheless",
    "unlike",
    "whereas",
    "while",
    "그러나",
    "다만",
    "반면",
    "하지만",
}
_NEGATION_TOKENS = {
    "barely",
    "cannot",
    "cant",
    "didnt",
    "doesnt",
    "fail",
    "failed",
    "fails",
    "hardly",
    "never",
    "no",
    "not",
    "without",
}
_PREDICATE_STEMS = {
    "causes": "caus",
    "enables": "enabl",
    "improves": "improv",
    "reduces": "reduc",
    "requires": "requir",
    "uses": "use",
}


def _safe_json_list(raw: str) -> list[dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def estimate_evidence_quality(evidence: str) -> float:
    text = str(evidence or "").strip()
    if not text:
        return 0.0
    token_len = len(text)
    length_score = min(1.0, token_len / 220.0)
    signal_bonus = 0.0
    if any(marker in text.lower() for marker in ("because", "therefore", "shows", "improves", "reduces", "increase", "decrease")):
        signal_bonus = 0.15
    return max(0.0, min(1.0, length_score + signal_bonus))


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _normalize_score_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _tokenize_score_text(value: str) -> list[str]:
    return re.findall(r"[\w%]+", _normalize_score_text(value))


def estimate_generic_claim_penalty(
    claim_text: str,
    evidence: str,
    *,
    subject: str = "",
    object_value: str = "",
) -> float:
    claim_raw = str(claim_text or "").strip()
    evidence_raw = str(evidence or "").strip()
    combined = f"{claim_raw} {evidence_raw}".strip()
    tokens = _tokenize_score_text(combined)
    if not tokens:
        return 0.18

    penalty = 0.0
    if len(tokens) <= 8:
        penalty += 0.06
    elif len(tokens) <= 14:
        penalty += 0.03

    if not _SPECIFIC_SIGNAL_RE.search(combined):
        penalty += 0.04

    generic_hits = sum(1 for token in tokens if token in _GENERIC_CLAIM_TOKENS)
    penalty += min(0.06, generic_hits * 0.02)

    if any(token in _HEDGE_TOKENS for token in tokens):
        penalty += 0.03

    if evidence_raw and _normalize_score_text(claim_raw) == _normalize_score_text(evidence_raw) and len(evidence_raw) < 120:
        penalty += 0.02

    subject_norm = _normalize_score_text(subject)
    object_norm = _normalize_score_text(object_value)
    if subject_norm and object_norm and subject_norm == object_norm:
        penalty += 0.08

    return round(max(0.0, min(0.18, penalty)), 6)


def estimate_contradiction_hint(claim_text: str, evidence: str, *, predicate: str = "") -> float:
    combined = f"{claim_text or ''} {evidence or ''}".strip()
    tokens = _tokenize_score_text(combined)
    if not tokens:
        return 0.0

    hint = 0.0
    token_set = set(tokens)
    if token_set & _CONTRADICTION_CUES:
        hint += 0.35
    if token_set & _HEDGE_TOKENS:
        hint += 0.15
    if token_set & _NEGATION_TOKENS:
        hint += 0.12

    stem = _PREDICATE_STEMS.get(str(predicate or "").strip().lower(), "")
    if stem:
        for idx, token in enumerate(tokens):
            if token.startswith(stem):
                window = token_set.intersection(tokens[max(0, idx - 3) : idx])
                if window & _NEGATION_TOKENS:
                    hint += 0.28
                    break

    return round(_clamp01(hint), 6)


def score_claim_with_breakdown(
    llm_confidence: float,
    entity_resolve_conf: float,
    evidence_quality: float,
    *,
    claim_text: str = "",
    evidence: str = "",
    subject: str = "",
    predicate: str = "",
    object_value: str = "",
) -> tuple[float, dict[str, float]]:
    llm_score = _clamp01(llm_confidence)
    entity_score = _clamp01(entity_resolve_conf)
    evidence_score = _clamp01(evidence_quality)
    base_score = (0.6 * llm_score) + (0.2 * entity_score) + (0.2 * evidence_score)
    generic_penalty = estimate_generic_claim_penalty(
        claim_text,
        evidence,
        subject=subject,
        object_value=object_value,
    )
    contradiction_hint = estimate_contradiction_hint(claim_text, evidence, predicate=predicate)
    contradiction_penalty = round(0.12 * contradiction_hint, 6)
    weak_evidence_penalty = 0.0
    if evidence_score < 0.28:
        weak_evidence_penalty = 0.05
    elif evidence_score < 0.45:
        weak_evidence_penalty = 0.02

    final_score = _clamp01(base_score - generic_penalty - contradiction_penalty - weak_evidence_penalty)
    breakdown = {
        "llm_confidence": round(llm_score, 6),
        "evidence_quality": round(evidence_score, 6),
        "entity_resolution_confidence": round(entity_score, 6),
        "generic_claim_penalty": round(generic_penalty, 6),
        "contradiction_hint": round(contradiction_hint, 6),
        "base_score": round(base_score, 6),
        "weak_evidence_penalty": round(weak_evidence_penalty, 6),
        "contradiction_penalty": round(contradiction_penalty, 6),
        "final_score": round(final_score, 6),
    }
    return final_score, breakdown


def score_claim(
    llm_confidence: float,
    entity_resolve_conf: float,
    evidence_quality: float,
    *,
    claim_text: str = "",
    evidence: str = "",
    subject: str = "",
    predicate: str = "",
    object_value: str = "",
) -> float:
    score, _breakdown = score_claim_with_breakdown(
        llm_confidence,
        entity_resolve_conf,
        evidence_quality,
        claim_text=claim_text,
        evidence=evidence,
        subject=subject,
        predicate=predicate,
        object_value=object_value,
    )
    return score


def extract_claim_candidates(
    llm,
    title: str,
    text: str,
    max_claims: int = 8,
    max_input_chars: int = 3500,
) -> list[ClaimCandidate]:
    excerpt = (text or "")[: max(1, int(max_input_chars))]
    prompt = (
        "Extract high-value factual claims from this AI/ML paper text. "
        "Return strict JSON array only.\n"
        "Schema: [{\"claim_text\":\"...\",\"subject\":\"...\",\"predicate\":\"...\","
        "\"object\":\"...\",\"evidence\":\"...\",\"confidence\":0.0}]\n"
        "Rules:\n"
        "- Keep 3-8 claims\n"
        "- subject/predicate/object must be concise\n"
        "- predicate should be relation-like (improves/requires/causes/enables/uses/reduces/etc)\n"
        "- confidence in [0,1]\n"
        "- evidence must be a sentence from or faithful paraphrase of the text\n\n"
        f"Title: {title}\n\nText:\n{excerpt}"
    )
    raw = llm.generate(prompt).strip()
    items = _safe_json_list(raw)
    results: list[ClaimCandidate] = []
    for item in items[: max(1, int(max_claims))]:
        claim_text = str(item.get("claim_text", "")).strip()
        subject = str(item.get("subject", "")).strip()
        predicate = str(item.get("predicate", "")).strip().lower()
        object_value = str(item.get("object", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        if not claim_text or not subject or not predicate:
            continue
        try:
            llm_conf = float(item.get("confidence", 0.0))
        except Exception:
            llm_conf = 0.0
        llm_conf = max(0.0, min(1.0, llm_conf))
        results.append(
            ClaimCandidate(
                claim_text=claim_text[:600],
                subject=subject[:160],
                predicate=predicate[:80],
                object_value=object_value[:160],
                evidence=evidence[:600],
                llm_confidence=llm_conf,
            )
        )
    return results
