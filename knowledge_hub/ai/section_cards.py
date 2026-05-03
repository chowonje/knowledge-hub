from __future__ import annotations

import re
from typing import Any

from knowledge_hub.ai.ask_v2_support import clean_text, query_terms, stable_score, text_overlap


_METHOD_RE = re.compile(r"\b(method|methods|approach|architecture|pipeline|implementation|방법|접근|구현)\b", re.IGNORECASE)
_RESULT_RE = re.compile(r"\b(result|results|finding|findings|evaluation|experiment|experiments|benchmark|metric|결과|평가|실험)\b", re.IGNORECASE)
_LIMITATION_RE = re.compile(r"\b(limit|limits|limitation|limitations|future work|risk|caveat|한계)\b", re.IGNORECASE)
_PROBLEM_RE = re.compile(r"\b(problem|motivation|background|introduction|abstract|문제|배경|요약|초록)\b", re.IGNORECASE)
_APPENDIX_RE = re.compile(r"\b(appendix|supplement|부록)\b", re.IGNORECASE)
_PLACEHOLDER_PATTERNS = (
    re.compile(r"원문.*필요", re.IGNORECASE),
    re.compile(r"원문.*확인할 수 없", re.IGNORECASE),
    re.compile(r"제목.?저자 정보뿐", re.IGNORECASE),
    re.compile(r"요약을 바로 작성할 수 없", re.IGNORECASE),
    re.compile(r"paper text unavailable", re.IGNORECASE),
    re.compile(r"original paper required", re.IGNORECASE),
    re.compile(r"cannot access the paper", re.IGNORECASE),
)


def _role_for_unit(unit: dict[str, Any]) -> str:
    unit_type = clean_text(unit.get("unit_type")).casefold()
    if unit_type == "document_summary":
        return "problem"
    if unit_type in {"summary", "background"}:
        return "problem"
    if unit_type == "method":
        return "method"
    if unit_type in {"result", "table_block"}:
        return "results"
    if unit_type == "limitation":
        return "limitations"
    haystack = " ".join(
        [
            clean_text(unit.get("title")),
            clean_text(unit.get("section_path")),
            clean_text(unit.get("contextual_summary")),
        ]
    )
    if _LIMITATION_RE.search(haystack):
        return "limitations"
    if _RESULT_RE.search(haystack):
        return "results"
    if _METHOD_RE.search(haystack):
        return "method"
    if _PROBLEM_RE.search(haystack):
        return "problem"
    return "other"


def _appendix_like(unit: dict[str, Any]) -> bool:
    if _APPENDIX_RE.search(" ".join([clean_text(unit.get("title")), clean_text(unit.get("section_path"))])):
        return True
    quality = dict((unit.get("provenance") or {}).get("quality_signals") or {})
    return bool(quality.get("appendix_like"))


def project_section_cards(*, source_kind: str, source_card: dict[str, Any], units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if source_kind != "paper":
        return []
    source_card_id = clean_text(source_card.get("card_id"))
    paper_id = clean_text(source_card.get("paper_id"))
    section_cards: list[dict[str, Any]] = []
    for unit in units:
        unit_type = clean_text(unit.get("unit_type")).casefold()
        if not unit_type:
            continue
        unit_id = clean_text(unit.get("unit_id"))
        if not unit_id:
            continue
        role = _role_for_unit(unit)
        confidence = stable_score(unit.get("confidence"))
        section_cards.append(
            {
                "section_card_id": f"paper-section-card:{unit_id}",
                "source_kind": source_kind,
                "source_card_id": source_card_id,
                "source_id": paper_id,
                "paper_id": paper_id,
                "document_id": clean_text(unit.get("document_id")),
                "unit_id": unit_id,
                "title": clean_text(unit.get("title")),
                "section_path": clean_text(unit.get("section_path")),
                "unit_type": unit_type,
                "role": role,
                "order_index": int(unit.get("order_index") or 0),
                "contextual_summary": clean_text(unit.get("contextual_summary")),
                "source_excerpt": clean_text(unit.get("source_excerpt")),
                "document_thesis": clean_text(unit.get("document_thesis")),
                "confidence": confidence,
                "claims": list(unit.get("claims") or []),
                "concepts": list(unit.get("concepts") or []),
                "provenance": dict(unit.get("provenance") or {}),
                "appendix_like": _appendix_like(unit),
                "search_text": clean_text(
                    " ".join(
                        [
                            clean_text(source_card.get("title")),
                            role,
                            clean_text(unit.get("title")),
                            clean_text(unit.get("section_path")),
                            clean_text(unit.get("contextual_summary")),
                            clean_text(unit.get("source_excerpt")),
                            clean_text(unit.get("document_thesis")),
                        ]
                    )
                ),
            }
        )
    return section_cards


def rank_section_cards(*, query: str, section_cards: list[dict[str, Any]], intent: str) -> list[dict[str, Any]]:
    role_bias = {
        "paper_summary": {"problem": 1.0, "method": 0.9, "results": 0.8, "limitations": 0.4},
        "paper_lookup": {"problem": 1.0, "method": 0.8, "results": 0.7, "limitations": 0.3},
        "definition": {"problem": 1.2, "method": 0.9, "results": 0.3, "limitations": 0.2},
        "relation": {"problem": 1.0, "method": 0.9, "results": 0.4, "limitations": 0.2},
        "implementation": {"method": 1.4, "problem": 0.7, "results": 0.5, "limitations": 0.2},
        "temporal": {"results": 0.8, "problem": 0.7, "method": 0.5, "limitations": 0.3},
    }
    query_token_set = {term for term in query_terms(query) if term}
    ranked: list[dict[str, Any]] = []
    for item in section_cards:
        role = clean_text(item.get("role")).casefold() or "other"
        overlap = text_overlap(
            query,
            item.get("title"),
            item.get("section_path"),
            item.get("contextual_summary"),
            item.get("source_excerpt"),
            item.get("document_thesis"),
            item.get("search_text"),
        )
        query_bonus = 0.0
        reasons: list[str] = []
        for field in ("title", "section_path", "role"):
            token = clean_text(item.get(field))
            if token and any(term in token.casefold() or token.casefold() in term for term in query_token_set):
                query_bonus += 0.8
                reasons.append(f"{field}_match")
        bias = role_bias.get(intent, {}).get(role, 0.0)
        confidence = stable_score(item.get("confidence"))
        penalty = 0.0
        if bool(item.get("appendix_like")):
            penalty += 1.2
            reasons.append("appendix_penalty")
        if confidence < 0.3:
            penalty += 0.7
            reasons.append("low_confidence_penalty")
        score = round(overlap + query_bonus + bias + (confidence * 0.6) - penalty, 4)
        enriched = dict(item)
        enriched["selection_score"] = score
        enriched["ranking_reasons"] = [*reasons, f"role:{role}"]
        ranked.append(enriched)
    ranked.sort(key=lambda item: (-stable_score(item.get("selection_score")), -stable_score(item.get("confidence")), clean_text(item.get("section_card_id"))))
    return ranked


def section_coverage(*, section_cards: list[dict[str, Any]]) -> dict[str, Any]:
    selected_roles: list[str] = []
    weak_roles: list[str] = []
    for item in section_cards:
        role = clean_text(item.get("role"))
        if role and role not in selected_roles and role != "other":
            selected_roles.append(role)
        if stable_score(item.get("confidence")) < 0.3 and role and role not in weak_roles:
            weak_roles.append(role)
    core_roles = ["problem", "method", "results", "limitations"]
    missing_roles = [role for role in core_roles if role not in selected_roles]
    return {
        "selectedRoles": selected_roles,
        "missingRoles": missing_roles,
        "weakRoles": weak_roles,
        "roleCount": len(selected_roles),
        "status": "strong" if len(selected_roles) >= 2 and not missing_roles[:2] else "weak" if selected_roles else "missing",
    }


def _placeholder_like(item: dict[str, Any]) -> bool:
    haystack = " ".join(
        [
            clean_text(item.get("title")),
            clean_text(item.get("section_path")),
            clean_text(item.get("contextual_summary")),
            clean_text(item.get("source_excerpt")),
        ]
    )
    return any(pattern.search(haystack) for pattern in _PLACEHOLDER_PATTERNS)


def assess_section_source_quality(*, section_cards: list[dict[str, Any]], coverage: dict[str, Any] | None = None) -> dict[str, Any]:
    cov = dict(coverage or section_coverage(section_cards=section_cards))
    selected_roles = [clean_text(role) for role in list(cov.get("selectedRoles") or []) if clean_text(role)]
    role_set = set(selected_roles)
    cards = list(section_cards or [])
    placeholder_count = sum(1 for item in cards if _placeholder_like(item))
    appendix_count = sum(1 for item in cards if bool(item.get("appendix_like")))
    meta_only_count = 0
    for item in cards:
        summary = clean_text(item.get("contextual_summary"))
        excerpt = clean_text(item.get("source_excerpt"))
        claims = list(item.get("claims") or [])
        concepts = list(item.get("concepts") or [])
        if (not summary and not excerpt) or (_placeholder_like(item) and not claims and not concepts):
            meta_only_count += 1
    majority_threshold = (len(cards) // 2) + 1 if cards else 1
    blocked = False
    reason = ""
    if role_set == {"problem"} and cards:
        blocked = True
        reason = "problem_only_sections"
    elif str(cov.get("status") or "").strip() == "weak" and ("method" not in role_set or "results" not in role_set):
        blocked = True
        reason = "weak_missing_method_or_results"
    elif cards and meta_only_count >= len(cards):
        blocked = True
        reason = "all_sections_meta_only"
    elif placeholder_count >= majority_threshold:
        blocked = True
        reason = "placeholder_majority"
    signals = {
        "selectedRoles": selected_roles,
        "coverageStatus": str(cov.get("status") or "missing"),
        "placeholderCount": placeholder_count,
        "metaOnlyCount": meta_only_count,
        "appendixLikeCount": appendix_count,
        "cardCount": len(cards),
    }
    return {"allowed": not blocked, "reason": reason, "signals": signals}


__all__ = ["project_section_cards", "rank_section_cards", "section_coverage", "assess_section_source_quality"]
