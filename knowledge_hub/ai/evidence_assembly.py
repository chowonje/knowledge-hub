from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from knowledge_hub.application.query_frame import normalize_query_frame_dict
from knowledge_hub.ai.evidence_answerability import AnswerabilityInputs, evaluate_answerability
from knowledge_hub.ai.evidence_collaborator import (
    EvidenceAssemblyCollaborator,
    make_evidence_assembly_collaborator,
)
from knowledge_hub.ai.retrieval_fit import (
    classify_query_intent,
    is_non_substantive_text,
    is_vault_hub_note,
    normalize_source_type,
)
from knowledge_hub.core.models import SearchResult
from knowledge_hub.domain.registry import get_domain_pack
from knowledge_hub.domain.ai_papers.representative import curated_representative_title
from knowledge_hub.domain.ai_papers.evidence_policy import normalize_evidence_policy, policy_for_family, select_evidence_policy


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _estimate_token_count(*parts: Any) -> int:
    text = " ".join(str(part or "") for part in parts if str(part or "").strip())
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return 0
    return max(1, int(round(len(compact) / 4.0)))


def _result_id(result: SearchResult) -> str:
    if result.document_id:
        return result.document_id
    metadata = dict(result.metadata or {})
    title = str(metadata.get("title") or "").strip()
    file_path = str(metadata.get("file_path") or "").strip()
    chunk = str(metadata.get("chunk_index") or "0").strip()
    if file_path:
        return f"{file_path}#{chunk}#{title}".strip("#")
    if title:
        return f"{title}#{chunk}"
    return f"{result.distance:.4f}-{len(result.document)}"


def _result_paper_id(result: SearchResult) -> str:
    metadata = dict(result.metadata or {})
    for key in ("arxiv_id", "paper_id"):
        token = str(metadata.get(key) or "").strip()
        if token:
            return token
    return ""


_PAPER_IDENTITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "with",
    "what",
    "whatis",
    "core",
    "idea",
    "ideas",
    "explain",
    "explainer",
    "concept",
    "concepts",
    "쉽게",
    "설명",
    "설명해줘",
    "설명해주세요",
    "정의",
    "개념",
    "핵심",
    "아이디어",
    "원리",
    "무엇",
}
_BEGINNER_EXPLAINER_RE = re.compile(r"\b(simple|simply|beginner|for beginners|easy|intuition)\b|쉽게|입문|초심자|직관", re.IGNORECASE)
_REPRESENTATIVE_SURVEY_RE = re.compile(r"\b(survey|overview|comparison|benchmark)\b|정리|비교|개관|서베이", re.IGNORECASE)


def _paper_identity_terms(query: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[0-9a-z가-힣]+", str(query or "").casefold())
        if len(token) >= 2 and token not in _PAPER_IDENTITY_STOPWORDS
    ]


def _paper_definition_explanation_style(*, query: str, query_plan: dict[str, Any] | None = None) -> str:
    plan_payload = dict(query_plan or {})
    answer_mode = str(plan_payload.get("answer_mode") or plan_payload.get("answerMode") or "").strip().lower()
    if "beginner" in answer_mode or _BEGINNER_EXPLAINER_RE.search(str(query or "")):
        return "novice"
    return "default"


def _paper_definition_expanded_terms(query_plan: dict[str, Any] | None = None) -> list[str]:
    plan_payload = dict(query_plan or {})
    return [
        str(item).strip()
        for item in list(plan_payload.get("expanded_terms") or plan_payload.get("expandedTerms") or [])
        if str(item).strip()
    ]


def _representative_paper_score(
    result: SearchResult,
    *,
    evidence_item: dict[str, Any],
    query: str,
    query_plan: dict[str, Any] | None = None,
) -> float:
    metadata = dict(result.metadata or {})
    title = str(metadata.get("title") or evidence_item.get("title") or "").strip().casefold()
    body = " ".join(
        [
            title,
            str(evidence_item.get("excerpt") or "").strip(),
            str(evidence_item.get("document") or "").strip(),
            str(getattr(result, "document", "") or "").strip(),
        ]
    ).casefold()
    score = _safe_float(getattr(result, "score", 0.0), 0.0)
    score += _direct_answer_score(evidence_item, query=query)
    query_terms = _query_terms(query)
    overlap = sum(1 for term in query_terms if term in title)
    if overlap:
        score += 0.9 * overlap
    expanded_overlap = 0
    for term in _paper_definition_expanded_terms(query_plan)[:6]:
        lowered = term.casefold()
        if lowered and lowered in title:
            expanded_overlap += 1
            score += 2.2
        elif lowered and lowered in body:
            score += 1.0
    score += min(1.5, _safe_float(evidence_item.get("source_trust_score"), 0.0) * 1.5)
    if str(evidence_item.get("reference_tier") or "").strip().lower() in {"specialist", "glossary", "standard", "background_reference"}:
        score += 0.8
    if int(evidence_item.get("highAuthorityCount") or 0) > 0:
        score += 0.6
    if expanded_overlap > 0:
        score += 0.5
    return round(score, 4)


def _representative_selection_reason(
    *,
    paper_id: str,
    resolved_source_ids: list[str],
    title_hits: float,
    body_hits: float,
    trust_boost: float,
    retrieval_score: float,
) -> str:
    if paper_id and paper_id in resolved_source_ids and title_hits >= 1:
        return "resolved_source_and_title_match"
    if paper_id and paper_id in resolved_source_ids:
        return "resolved_source_match"
    if title_hits >= 2:
        return "strong_title_match"
    if title_hits >= 1 and body_hits >= 2:
        return "title_and_body_match"
    if title_hits >= 1 and trust_boost > 0:
        return "title_match_with_trust"
    if body_hits >= 2 and trust_boost > 0:
        return "body_match_with_trust"
    if trust_boost >= 0.8:
        return "trusted_reference_signal"
    if retrieval_score > 0:
        return "retrieval_score_lead"
    return "candidate_score_lead"


def _has_strong_paper_identity_signal(query: str, paper_results: list[SearchResult]) -> bool:
    terms = _paper_identity_terms(query)
    if not terms:
        return False
    joined = " ".join(terms)
    for result in paper_results[:3]:
        title = re.sub(r"\s+", " ", str((result.metadata or {}).get("title") or "").casefold()).strip()
        if not title:
            continue
        if joined and joined in title:
            return True
        overlap = sum(1 for term in terms if term in title)
        if overlap >= 2:
            return True
    return False


def _normalize_family_anchor(value: Any) -> str:
    token = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    token = re.sub(r"[^0-9a-z가-힣:/#._ -]+", "", token)
    return token[:160]


def _is_temporal_query(query: str) -> bool:
    return bool(
        re.search(r"\b(latest|recent|updated|update|newest|changed since|before|after|since)\b", str(query or ""), re.IGNORECASE)
        or re.search(r"최근|최신|업데이트|이전|이후|당시", str(query or ""))
    )


def _is_abstention_query(query: str) -> bool:
    body = str(query or "")
    return bool(
        re.search(r"\b(can we|can i|should we|is it safe to|is it fair to|can we conclude|can we claim)\b", body, re.IGNORECASE)
        or re.search(r"단정할 수 있나|말해도 되나|봐도 되나|비교할 수 있나|출처 없이|강하게 답해야 하나|완전하게 비교할 수 있나", body)
    )


def _is_non_substantive_evidence(item: dict[str, Any]) -> bool:
    text = " ".join([str(item.get("title") or ""), str(item.get("excerpt") or ""), str(item.get("document") or "")]).strip()
    return is_non_substantive_text(text)


def _is_source_mismatch(item: dict[str, Any], source_type: str | None) -> bool:
    normalized_expected = normalize_source_type(source_type)
    if not normalized_expected or normalized_expected == "all":
        return False
    actual = normalize_source_type(item.get("normalized_source_type") or item.get("source_type"))
    return bool(actual and actual != normalized_expected)


def _has_explicit_temporal_marker(item: dict[str, Any]) -> bool:
    for name in ("event_date", "document_date", "published_at", "evidence_window"):
        if str(item.get(name) or "").strip():
            return True
    text = " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("section_path") or ""),
            str(item.get("excerpt") or ""),
        ]
    )
    return bool(
        re.search(r"\b(v\d+|version\s*\d+|updated?|latest|recent|newest|release)\b", text, re.IGNORECASE)
        or re.search(r"버전|업데이트|최신|개정", text)
    )


def _is_temporal_grounded(item: dict[str, Any]) -> bool:
    return _has_explicit_temporal_marker(item)


def _is_observed_at_only(item: dict[str, Any]) -> bool:
    return bool(str(item.get("observed_at") or "").strip()) and not _has_explicit_temporal_marker(item)


def _has_web_temporal_textual_marker(item: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("section_path") or ""),
            str(item.get("source_url") or ""),
        ]
    )
    return bool(
        re.search(r"\b(updated?|latest|recent|newest|version|release|guide|guideline|watchlist|reference)\b", text, re.IGNORECASE)
        or re.search(r"최신|최근|업데이트|버전|개정|가이드|레퍼런스|참고", text)
    )


def _is_temporal_web_candidate(item: dict[str, Any], *, source_type: str | None) -> bool:
    return normalize_source_type(source_type) == "web" and normalize_source_type(item.get("source_type")) == "web"


def _is_substantive_evidence(item: dict[str, Any], *, query: str, source_type: str | None) -> bool:
    if _is_non_substantive_evidence(item):
        return False
    if _is_source_mismatch(item, source_type):
        return False
    if normalize_source_type(item.get("source_type")) == "vault" and is_vault_hub_note(
        title=str(item.get("title") or ""),
        file_path=str(item.get("file_path") or ""),
        document=str(item.get("excerpt") or item.get("document") or ""),
        query=query,
    ):
        return False
    return True


def _query_terms(query: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^0-9A-Za-z가-힣]+", str(query or "").casefold())
        if token and len(token) >= 2
    }


_NOVICE_EXPLAINER_MARKERS = (
    "쉽게",
    "쉬운",
    "입문",
    "초심자",
    "처음",
    "비전공자",
    "beginner",
    "novice",
)


def _is_novice_explainer_query(query: str) -> bool:
    body = str(query or "").casefold()
    return any(marker in body for marker in _NOVICE_EXPLAINER_MARKERS)


def _direct_answer_score(item: dict[str, Any], *, query: str) -> float:
    intent = classify_query_intent(query)
    text = " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("section_path") or ""),
            str(item.get("excerpt") or ""),
            str(item.get("file_path") or ""),
        ]
    ).casefold()
    unit_type = str(item.get("unit_type") or "").strip().lower()
    score = 0.0
    if intent == "definition":
        if any(token in text for token in ("definition", "overview", "summary", "설명", "정의", "개요", "요약", "목적")):
            score += 1.3
        if unit_type in {"summary", "document_summary", "background"}:
            score += 0.8
    elif intent == "comparison":
        if any(token in text for token in ("compare", "comparison", "versus", "vs", "difference", "비교", "차이", "장단점")):
            score += 1.4
        if unit_type == "result":
            score += 0.6
    elif intent == "implementation":
        if any(token in text for token in ("implementation", "pipeline", "service", "orchestrator", "구현", "파이프라인", "서비스", "아키텍처")):
            score += 1.4
        if unit_type == "method":
            score += 0.8
    elif intent == "evaluation":
        if any(token in text for token in ("evaluation", "benchmark", "metric", "result", "finding", "평가", "벤치마크", "지표", "결과")):
            score += 1.4
        if unit_type == "result":
            score += 0.8
    elif intent == "howto":
        if any(token in text for token in ("guide", "steps", "setup", "usage", "가이드", "설정", "사용법", "방법")):
            score += 1.3
        if unit_type == "method":
            score += 0.7
    elif intent == "paper_lookup":
        if normalize_source_type(item.get("source_type")) == "paper":
            score += 0.8
        if any(token in text for token in ("abstract", "summary", "논문", "초록", "요약")):
            score += 0.8
    elif intent == "paper_topic":
        if normalize_source_type(item.get("source_type")) == "paper":
            score += 1.0
        if any(token in text for token in ("survey", "overview", "comparison", "related", "representative", "논문", "요약", "정리", "비교", "대표")):
            score += 0.9
        if unit_type in {"summary", "document_summary", "background", "result"}:
            score += 0.5
    else:
        if unit_type in {"summary", "document_summary"}:
            score += 0.5
    terms = _query_terms(query)
    overlap = sum(1 for term in terms if term in text)
    if overlap >= 2:
        score += 0.6
    elif overlap == 1:
        score += 0.2
    return round(score, 3)


def _is_direct_answer_candidate(item: dict[str, Any], *, query: str) -> bool:
    return _direct_answer_score(item, query=query) >= 1.0


def _reselection_score(item: dict[str, Any], *, query: str, source_type: str | None) -> float:
    text = " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("excerpt") or ""),
            str(item.get("file_path") or ""),
            str(item.get("section_path") or ""),
        ]
    ).casefold()
    terms = _query_terms(query)
    overlap = sum(1 for term in terms if term in text)
    score = float(overlap)
    if not _is_source_mismatch(item, source_type):
        score += 2.0
    if normalize_source_type(item.get("source_type")) == "paper":
        score += 0.5
    if normalize_source_type(item.get("source_type")) == "vault" and not is_vault_hub_note(
        title=str(item.get("title") or ""),
        file_path=str(item.get("file_path") or ""),
        document=str(item.get("excerpt") or item.get("document") or ""),
        query=query,
    ):
        score += 0.75
    if _is_temporal_grounded(item):
        score += 0.5
    score += _direct_answer_score(item, query=query)
    return score


def _top1_reselection_reason(item: dict[str, Any], *, query: str, source_type: str | None) -> str:
    if _is_non_substantive_evidence(item):
        return "non_substantive_top1"
    if _is_source_mismatch(item, source_type):
        return "source_mismatch"
    if normalize_source_type(item.get("source_type")) == "vault" and is_vault_hub_note(
        title=str(item.get("title") or ""),
        file_path=str(item.get("file_path") or ""),
        document=str(item.get("excerpt") or item.get("document") or ""),
        query=query,
    ):
        return "vault_hub_noise"
    return ""


def _derive_paper_answer_scope(
    *,
    query: str,
    source_type: str | None,
    filtered: list[SearchResult],
    paper_memory_prefilter: dict[str, Any],
    metadata_filter: dict[str, Any] | None = None,
    query_plan: dict[str, Any] | None = None,
    query_frame: dict[str, Any] | None = None,
    evidence_policy: dict[str, Any] | None = None,
) -> tuple[list[SearchResult], dict[str, Any]]:
    intent = classify_query_intent(query)
    plan_payload = dict(query_plan or {})
    frame_payload = normalize_query_frame_dict(query_frame)
    paper_family = str(frame_payload.get("family") or plan_payload.get("family") or "").strip().lower() or (
        "paper_lookup" if intent == "paper_lookup" else "paper_discover" if intent == "paper_topic" else "concept_explainer" if normalize_source_type(source_type) == "paper" and intent == "definition" else "general"
    )
    policy_payload = normalize_evidence_policy(evidence_policy, family=paper_family)
    normalized_source = normalize_source_type(source_type)
    scoped_filter = dict(metadata_filter or {})
    explicit_paper_id = str(scoped_filter.get("arxiv_id") or scoped_filter.get("paper_id") or "").strip()
    planned_paper_ids = [
        str(item).strip()
        for item in list(frame_payload.get("resolved_source_ids") or plan_payload.get("resolvedPaperIds") or plan_payload.get("resolved_paper_ids") or [])
        if str(item).strip()
    ]
    matched_ids = [str(item).strip() for item in list(paper_memory_prefilter.get("matchedPaperIds") or []) if str(item).strip()]
    paper_results = [item for item in filtered if _result_paper_id(item)]
    ranked_paper_ids = [_result_paper_id(item) for item in paper_results if _result_paper_id(item)]
    unique_paper_ids = list(dict.fromkeys(ranked_paper_ids))

    selected_paper_id = ""
    reason = ""
    allow_single_paper_narrowing = bool(policy_payload.get("singleScopeRequired"))

    if explicit_paper_id:
        selected_paper_id = explicit_paper_id
        reason = "explicit_metadata_filter"
    elif allow_single_paper_narrowing and len(planned_paper_ids) == 1:
        selected_paper_id = planned_paper_ids[0]
        reason = "query_plan_single_match"
    elif allow_single_paper_narrowing and len(matched_ids) == 1:
        selected_paper_id = matched_ids[0]
        reason = "paper_memory_single_match"

    if not selected_paper_id:
        return filtered, {
            "applied": False,
            "reason": "not_applicable",
            "selectedPaperId": "",
            "candidatePaperIds": unique_paper_ids[:5],
            "matchedPaperIds": matched_ids[:5],
            "queryIntent": intent,
            "paperFamily": paper_family,
            "evidencePolicyKey": str(policy_payload.get("policyKey") or ""),
        }

    narrowed = [item for item in filtered if _result_paper_id(item) == selected_paper_id]
    if not narrowed:
        return filtered, {
            "applied": False,
            "reason": "selected_paper_missing",
            "selectedPaperId": selected_paper_id,
            "candidatePaperIds": unique_paper_ids[:5],
            "matchedPaperIds": matched_ids[:5],
            "queryIntent": intent,
            "paperFamily": paper_family,
            "evidencePolicyKey": str(policy_payload.get("policyKey") or ""),
        }

    return narrowed, {
        "applied": True,
        "reason": reason,
        "selectedPaperId": selected_paper_id,
        "candidatePaperIds": unique_paper_ids[:5],
        "matchedPaperIds": matched_ids[:5],
        "queryIntent": intent,
        "paperFamily": paper_family,
        "filteredResultCount": len(narrowed),
        "evidencePolicyKey": str(policy_payload.get("policyKey") or ""),
    }


def _apply_answer_evidence_budget(
    *,
    query: str,
    source_type: str | None,
    filtered: list[SearchResult],
    paper_scope: dict[str, Any],
    query_plan: dict[str, Any] | None = None,
    query_frame: dict[str, Any] | None = None,
    evidence_policy: dict[str, Any] | None = None,
) -> tuple[list[SearchResult], dict[str, Any]]:
    intent = classify_query_intent(query)
    plan_payload = dict(query_plan or {})
    frame_payload = normalize_query_frame_dict(query_frame)
    paper_family = str(frame_payload.get("family") or plan_payload.get("family") or paper_scope.get("paperFamily") or "").strip().lower() or (
        "paper_lookup" if intent == "paper_lookup" else "paper_discover" if intent == "paper_topic" else "concept_explainer" if normalize_source_type(source_type) == "paper" and intent == "definition" else "general"
    )
    policy_payload = normalize_evidence_policy(evidence_policy, family=paper_family)
    normalized_source = normalize_source_type(source_type)
    original_count = len(filtered)
    max_sources = 6
    reason = "default"
    if bool(paper_scope.get("applied")):
        max_sources = 2 if paper_family in {"paper_lookup", "concept_explainer"} else 4
        reason = "paper_scope"
    elif bool(policy_payload.get("shortlistOnly")) and normalized_source == "paper":
        max_sources = 6
        reason = "policy_shortlist_only"
    elif normalized_source == "paper" and paper_family == "concept_explainer":
        max_sources = 2
        reason = "paper_concept_family"
    elif normalized_source == "paper" and paper_family == "paper_compare":
        max_sources = 4
        reason = "paper_compare_family"
    elif normalized_source == "paper":
        max_sources = 4
        reason = "paper_source"
    elif intent in {"comparison"}:
        max_sources = 5
        reason = "comparison"
    elif intent in {"implementation"}:
        max_sources = 5
        reason = "implementation"
    trimmed = list(filtered[:max_sources])
    return trimmed, {
        "applied": original_count > len(trimmed),
        "reason": reason,
        "queryIntent": intent,
        "paperFamily": paper_family,
        "evidencePolicyKey": str(policy_payload.get("policyKey") or ""),
        "maxSources": max_sources,
        "originalCount": original_count,
        "selectedCount": len(trimmed),
    }


def _build_citations(results: list[SearchResult], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for index, (result, item) in enumerate(zip(results, evidence, strict=False), 1):
        metadata = dict(getattr(result, "metadata", {}) or {})
        arxiv_id = str(item.get("arxiv_id") or metadata.get("arxiv_id") or metadata.get("paper_id") or "").strip()
        file_path = str(item.get("file_path") or metadata.get("file_path") or "").strip()
        source_url = str(item.get("source_url") or metadata.get("source_url") or metadata.get("url") or "").strip()
        title = str(item.get("title") or metadata.get("title") or "").strip()
        target = arxiv_id or file_path or source_url or title
        citations.append(
            {
                "label": f"S{index}",
                "title": title,
                "source_type": str(item.get("source_type") or metadata.get("source_type") or ""),
                "target": target,
                "kind": "arxiv" if arxiv_id else "file" if file_path else "url" if source_url else "title",
            }
        )
    return citations


def _paper_definition_signal(
    *,
    query: str,
    source_type: str | None,
    results: list[SearchResult],
    evidence: list[dict[str, Any]],
    query_plan: dict[str, Any] | None = None,
    query_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if normalize_source_type(source_type) != "paper":
        return {}
    frame_payload = normalize_query_frame_dict(query_frame)
    plan_payload = dict(query_plan or {})
    paper_family = str(frame_payload.get("family") or plan_payload.get("family") or "").strip().lower()
    if paper_family and paper_family != "concept_explainer":
        return {}
    if not paper_family and classify_query_intent(query) != "definition":
        return {}
    if not results or not evidence:
        return {}
    resolved_source_ids = [
        str(item).strip()
        for item in list(frame_payload.get("resolved_source_ids") or plan_payload.get("resolvedPaperIds") or plan_payload.get("resolved_paper_ids") or [])
        if str(item).strip()
    ]
    resolved_source_id_set = set(resolved_source_ids)
    ranking_terms = [
        str(item).strip().casefold()
        for item in list(frame_payload.get("expanded_terms") or frame_payload.get("entities") or [])
        if str(item).strip()
    ]
    if not ranking_terms:
        ranking_terms = [
            token.casefold()
            for token in _paper_identity_terms(query)
            if token
        ]

    def _candidate_score(result: SearchResult, evidence_item: dict[str, Any]) -> tuple[float, float, float, float, float]:
        paper_id = _result_paper_id(result)
        title = str((result.metadata or {}).get("title") or evidence_item.get("title") or "").casefold()
        excerpt = " ".join(
            [
                str(evidence_item.get("excerpt") or ""),
                str(evidence_item.get("document_thesis") or ""),
                str(evidence_item.get("contextual_summary") or ""),
                str(getattr(result, "document", "") or ""),
            ]
        ).casefold()
        title_hits = sum(1.0 for term in ranking_terms if term and term in title)
        body_hits = sum(1.0 for term in ranking_terms if term and term in excerpt)
        structured_sections = any(
            marker in str(getattr(result, "document", "") or "")
            for marker in ("## 한줄 요약", "## 요약", "## 핵심 아이디어", "## Core Idea", "## 방법", "## Method")
        )
        resolved_boost = 3.0 if paper_id and paper_id in resolved_source_id_set else 0.0
        trust_boost = _safe_float(evidence_item.get("source_trust_score"), 0.0)
        specialist_boost = 0.35 if str(evidence_item.get("reference_tier") or "").strip().lower() == "specialist" else 0.0
        survey_penalty = 0.0
        if paper_family == "concept_explainer" and _REPRESENTATIVE_SURVEY_RE.search(title) and not resolved_boost:
            survey_penalty = 1.75
        return (
            resolved_boost
            + (1.4 * title_hits)
            + (0.35 * body_hits)
            + (0.8 if structured_sections else 0.0)
            + trust_boost
            + specialist_boost
            - survey_penalty
            + _safe_float(getattr(result, "score", 0.0), 0.0),
            title_hits,
            body_hits,
            trust_boost,
            _safe_float(getattr(result, "score", 0.0), 0.0),
        )

    def _has_minimum_alignment(score_tuple: tuple[float, float, float, float, float], *, result: SearchResult) -> bool:
        _, title_hits, body_hits, _, _ = score_tuple
        paper_id = _result_paper_id(result)
        if paper_family == "concept_explainer":
            return bool((paper_id and paper_id in resolved_source_id_set) or title_hits >= 1)
        return bool((paper_id and paper_id in resolved_source_id_set) or title_hits >= 1 or body_hits >= 2)

    candidate_pairs = list(zip(results, evidence, strict=False))
    scored_candidates = [
        (result, evidence_item, _candidate_score(result, evidence_item))
        for result, evidence_item in candidate_pairs
    ]
    aligned_candidates = [
        item
        for item in scored_candidates
        if _has_minimum_alignment(item[2], result=item[0])
    ]
    resolved_aligned_candidates = [
        item
        for item in aligned_candidates
        if _result_paper_id(item[0]) in resolved_source_id_set
    ]
    if paper_family == "concept_explainer" and (not aligned_candidates or (resolved_source_ids and not resolved_aligned_candidates)):
        fallback_paper_id = resolved_source_ids[0] if resolved_source_ids else ""
        fallback_title = curated_representative_title(fallback_paper_id)
        if not fallback_paper_id or not fallback_title:
            return {}
        unique_paper_ids = list(dict.fromkeys(_result_paper_id(item) for item in results if _result_paper_id(item)))
        answer_mode = str(frame_payload.get("answer_mode") or plan_payload.get("answerMode") or plan_payload.get("answer_mode") or "").strip()
        novice_mode = _is_novice_explainer_query(query)
        concept_excerpt = str((evidence[0] or {}).get("excerpt") or (evidence[0] or {}).get("document") or "").strip() if evidence else ""
        return {
            "paper_definition_mode": True,
            "answer_mode": f"{answer_mode}_beginner" if answer_mode and novice_mode and not answer_mode.endswith("_beginner") else answer_mode,
            "novice_explainer_mode": novice_mode,
            "concept_core_evidence": {
                "paperId": fallback_paper_id,
                "title": fallback_title,
                "summary": concept_excerpt[:320],
                "sourceCount": len(unique_paper_ids),
            },
            "representative_paper_evidence": {
                "paperId": fallback_paper_id,
                "title": fallback_title,
                "citationLabel": "",
                "sourceCount": len(unique_paper_ids),
            },
            "representative_paper": {
                "paperId": fallback_paper_id,
                "title": fallback_title,
                "citationLabel": "",
                "sourceCount": len(unique_paper_ids),
            },
            "supporting_paper_count": max(0, len(unique_paper_ids) - 1),
            "representative_selection": {
                "paperId": fallback_paper_id,
                "title": fallback_title,
                "score": 0.0,
                "titleHits": 0,
                "bodyHits": 0,
                "trustBoost": 0.0,
                "retrievalScore": 0.0,
                "rankingTerms": ranking_terms[:6],
                "reason": "resolved_anchor_seed",
            },
        }
    candidate_pool = aligned_candidates or scored_candidates
    representative, representative_item, representative_score = max(
        candidate_pool,
        key=lambda item: item[2][0],
    )
    representative_paper_id = _result_paper_id(representative)
    if not representative_paper_id:
        return {}
    metadata = dict(representative.metadata or {})
    unique_paper_ids = list(dict.fromkeys(_result_paper_id(item) for item in results if _result_paper_id(item)))
    representative_title = str(metadata.get("title") or representative_item.get("title") or "").strip()
    concept_excerpt = str(representative_item.get("excerpt") or representative_item.get("document") or getattr(representative, "document", "") or "").strip()
    answer_mode = str(frame_payload.get("answer_mode") or plan_payload.get("answerMode") or plan_payload.get("answer_mode") or "").strip()
    novice_mode = _is_novice_explainer_query(query)
    candidate_score, title_hits, body_hits, trust_boost, retrieval_score = representative_score
    selection_reason = _representative_selection_reason(
        paper_id=representative_paper_id,
        resolved_source_ids=resolved_source_ids,
        title_hits=title_hits,
        body_hits=body_hits,
        trust_boost=trust_boost,
        retrieval_score=retrieval_score,
    )
    return {
        "paper_definition_mode": True,
        "answer_mode": f"{answer_mode}_beginner" if answer_mode and novice_mode and not answer_mode.endswith("_beginner") else answer_mode,
        "novice_explainer_mode": novice_mode,
        "concept_core_evidence": {
            "paperId": representative_paper_id,
            "title": representative_title,
            "summary": concept_excerpt[:320],
            "sourceCount": len(unique_paper_ids),
        },
        "representative_paper_evidence": {
            "paperId": representative_paper_id,
            "title": representative_title,
            "citationLabel": str(representative_item.get("citation_label") or "").strip(),
            "sourceCount": len(unique_paper_ids),
        },
        "representative_paper": {
            "paperId": representative_paper_id,
            "title": representative_title,
            "citationLabel": str(representative_item.get("citation_label") or "").strip(),
            "sourceCount": len(unique_paper_ids),
        },
        "supporting_paper_count": max(0, len(unique_paper_ids) - 1),
        "representative_selection": {
            "paperId": representative_paper_id,
            "title": representative_title,
            "score": round(candidate_score, 4),
            "titleHits": int(title_hits),
            "bodyHits": int(body_hits),
            "trustBoost": round(trust_boost, 4),
            "retrievalScore": round(retrieval_score, 4),
            "rankingTerms": ranking_terms[:6],
            "reason": selection_reason,
        },
    }


def _semantic_family_key(item: dict[str, Any], *, query: str) -> str:
    source_type = normalize_source_type(item.get("normalized_source_type") or item.get("source_type")) or "unknown"
    paper_id = str(item.get("arxiv_id") or item.get("paper_id") or "").strip()
    document_id = str(item.get("parent_id") or item.get("file_path") or item.get("citation_target") or item.get("source_url") or "").strip()
    anchor = _normalize_family_anchor(
        item.get("section_path")
        or item.get("parent_chunk_span")
        or item.get("title")
        or item.get("excerpt")
    )
    if source_type == "paper" and paper_id:
        return f"paper::{paper_id}::{anchor or classify_query_intent(query)}"
    base = paper_id or document_id or str(item.get("title") or "").strip()
    return f"{source_type}::{_normalize_family_anchor(base)}::{anchor or classify_query_intent(query)}"


def _context_budget_payload(
    *,
    pre_budget_results: list[SearchResult],
    selected_results: list[SearchResult],
    evidence: list[dict[str, Any]],
    context: str,
) -> dict[str, Any]:
    raw_chunk_tokens = sum(_estimate_token_count(getattr(item, "document", ""), dict(item.metadata or {}).get("title", "")) for item in pre_budget_results)
    selected_chunk_tokens = sum(_estimate_token_count(getattr(item, "document", ""), dict(item.metadata or {}).get("title", "")) for item in selected_results)
    memory_context_tokens = sum(
        _estimate_token_count(
            item.get("title"),
            item.get("excerpt"),
            item.get("document"),
            item.get("section_path"),
            item.get("document_thesis"),
            item.get("contextual_summary"),
        )
        for item in evidence
    )
    final_context_tokens = _estimate_token_count(context)
    dedup_saved_tokens = max(0, selected_chunk_tokens - final_context_tokens)
    gating_saved_tokens = max(0, raw_chunk_tokens - selected_chunk_tokens)
    verifier_added_tokens = max(0, final_context_tokens - memory_context_tokens)
    return {
        "rawChunkTokenEstimate": int(raw_chunk_tokens),
        "memoryContextTokenEstimate": int(memory_context_tokens),
        "finalPackedTokenEstimate": int(final_context_tokens),
        "dedupSavedTokens": int(dedup_saved_tokens),
        "gatingSavedTokens": int(gating_saved_tokens),
        "verifierAddedTokens": int(verifier_added_tokens),
    }


@dataclass
class EvidencePacket:
    filtered_results: list[SearchResult]
    evidence: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    context: str
    answer_signals: dict[str, Any]
    claims: list[dict[str, Any]]
    supporting_beliefs: list[dict[str, Any]]
    contradicting_beliefs: list[dict[str, Any]]
    belief_updates_suggested: list[dict[str, Any]]
    paper_answer_scope: dict[str, Any]
    evidence_budget: dict[str, Any]
    evidence_policy: dict[str, Any]
    evidence_packet: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["paperAnswerScope"] = payload.pop("paper_answer_scope")
        payload["evidenceBudget"] = payload.pop("evidence_budget")
        payload["evidencePolicy"] = payload.pop("evidence_policy")
        payload["answerSignals"] = payload.pop("answer_signals")
        payload["supportingBeliefs"] = payload.pop("supporting_beliefs")
        payload["contradictingBeliefs"] = payload.pop("contradicting_beliefs")
        payload["beliefUpdatesSuggested"] = payload.pop("belief_updates_suggested")
        payload["evidencePacket"] = payload.pop("evidence_packet")
        payload.pop("filtered_results", None)
        return payload


class EvidenceAssemblyService:
    def __init__(
        self,
        collaborator: EvidenceAssemblyCollaborator,
        *,
        eval_answer_profile: str | None = None,
    ):
        self.collaborator = collaborator
        self.eval_answer_profile = str(eval_answer_profile or "").strip()

    @classmethod
    def from_searcher(cls, searcher: Any) -> "EvidenceAssemblyService":
        builder = getattr(searcher, "_build_evidence_collaborator", None)
        collaborator = builder() if callable(builder) else make_evidence_assembly_collaborator(searcher)
        return cls(
            collaborator,
            eval_answer_profile=str(getattr(searcher, "_eval_answer_profile", "") or "").strip(),
        )

    def _dedupe_evidence(
        self,
        *,
        query: str,
        results: list[SearchResult],
        parent_ctx_by_result: dict[str, dict[str, Any]],
    ) -> tuple[list[SearchResult], list[dict[str, Any]], dict[str, Any]]:
        deduped_results: list[SearchResult] = []
        deduped_evidence: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        seen_families: set[str] = set()
        contradiction_count = 0
        duplicate_count = 0
        semantic_duplicate_count = 0
        collapsed_related: list[dict[str, Any]] = []
        for result in results:
            item = self.collaborator.answer_evidence_item(result, parent_ctx_by_result)
            ranking = dict(item.get("ranking_signals") or {})
            contradiction_penalty = _safe_float(ranking.get("contradiction_penalty"), 0.0)
            if contradiction_penalty >= 0.1:
                item["contradiction_flag"] = True
                contradiction_count += 1
            dedupe_key = "::".join(
                [
                    str(item.get("citation_target") or "").strip(),
                    str(item.get("parent_id") or "").strip(),
                    str(item.get("parent_chunk_span") or "").strip(),
                    str(item.get("title") or "").strip(),
                ]
            ).strip(":")
            if dedupe_key in seen_keys:
                duplicate_count += 1
                continue
            family_key = _semantic_family_key(item, query=query)
            if family_key in seen_families:
                semantic_duplicate_count += 1
                collapsed_related.append(
                    {
                        "familyKey": family_key,
                        "title": str(item.get("title") or ""),
                        "sourceType": str(item.get("source_type") or ""),
                        "citationTarget": str(item.get("citation_target") or ""),
                    }
                )
                continue
            seen_keys.add(dedupe_key)
            seen_families.add(family_key)
            item["semantic_family_key"] = family_key
            deduped_results.append(result)
            deduped_evidence.append(item)
        diagnostics = {
            "duplicatesRemoved": duplicate_count,
            "semanticDuplicatesRemoved": semantic_duplicate_count,
            "contradictionFlagCount": contradiction_count,
            "selectedCount": len(deduped_evidence),
            "nonSubstantiveEvidenceCount": sum(1 for item in deduped_evidence if _is_non_substantive_evidence(item)),
            "temporalGroundedCount": sum(
                1
                for item in deduped_evidence
                if _is_temporal_grounded(item)
            ),
            "weakObservedAtOnlyCount": sum(1 for item in deduped_evidence if _is_observed_at_only(item)),
            "highFreshnessCount": sum(1 for item in deduped_evidence if _safe_float(item.get("freshness_score"), 0.0) >= 0.55),
            "highAuthorityCount": sum(1 for item in deduped_evidence if _safe_float(item.get("source_trust_score"), 0.0) >= 0.85),
            "memoryProvenanceCount": sum(1 for item in deduped_evidence if dict(item.get("memory_provenance") or {})),
            "semanticFamilyCount": len(seen_families),
            "collapsedRelatedEvidence": collapsed_related[:10],
        }
        return deduped_results, deduped_evidence, diagnostics

    def _reselect_top1_if_needed(
        self,
        *,
        query: str,
        source_type: str | None,
        results: list[SearchResult],
        evidence: list[dict[str, Any]],
        profile: str,
    ) -> tuple[list[SearchResult], list[dict[str, Any]], dict[str, Any]]:
        if not evidence:
            return results, evidence, {
                "top1Substantive": False,
                "top1Reselected": False,
                "top1RejectedReason": "no_evidence",
            "substantiveEvidenceCount": 0,
            "directAnswerEvidenceCount": 0,
            "sourceMismatchCount": 0,
            "vaultHubPenaltyApplied": False,
        }

        substantive_flags = [
            _is_substantive_evidence(item, query=query, source_type=source_type)
            for item in evidence
        ]
        source_mismatch_count = sum(1 for item in evidence if _is_source_mismatch(item, source_type))
        direct_answer_count = sum(1 for item in evidence if _is_direct_answer_candidate(item, query=query))
        vault_hub_count = sum(
            1
            for item in evidence
            if normalize_source_type(item.get("source_type")) == "vault"
            and is_vault_hub_note(
                title=str(item.get("title") or ""),
                file_path=str(item.get("file_path") or ""),
                document=str(item.get("excerpt") or item.get("document") or ""),
                query=query,
            )
        )
        top1_substantive = bool(substantive_flags[0])
        top1_rejected_reason = "" if top1_substantive else _top1_reselection_reason(
            evidence[0], query=query, source_type=source_type
        )
        if top1_substantive or profile == "on-control":
            return results, evidence, {
                "top1Substantive": top1_substantive,
                "top1Reselected": False,
                "top1RejectedReason": top1_rejected_reason,
                "substantiveEvidenceCount": sum(1 for flag in substantive_flags if flag),
                "directAnswerEvidenceCount": direct_answer_count,
                "sourceMismatchCount": source_mismatch_count,
                "vaultHubPenaltyApplied": vault_hub_count > 0,
            }

        replacement_candidates = [
            (idx, _reselection_score(evidence[idx], query=query, source_type=source_type))
            for idx, flag in enumerate(substantive_flags[1:], 1)
            if flag
        ]
        replacement_candidates.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        replacement_index = replacement_candidates[0][0] if replacement_candidates else -1
        if replacement_index <= 0:
            return results, evidence, {
                "top1Substantive": False,
                "top1Reselected": False,
                "top1RejectedReason": top1_rejected_reason or "no_substantive_alternative",
                "substantiveEvidenceCount": sum(1 for flag in substantive_flags if flag),
                "directAnswerEvidenceCount": direct_answer_count,
                "sourceMismatchCount": source_mismatch_count,
                "vaultHubPenaltyApplied": vault_hub_count > 0,
            }

        reselection_order = [replacement_index, *[idx for idx in range(len(evidence)) if idx != replacement_index]]
        reselection_results = [results[idx] for idx in reselection_order]
        reselection_evidence = [evidence[idx] for idx in reselection_order]
        return reselection_results, reselection_evidence, {
            "top1Substantive": True,
            "top1Reselected": True,
            "top1RejectedReason": top1_rejected_reason or "non_substantive_top1",
            "substantiveEvidenceCount": sum(1 for flag in substantive_flags if flag),
            "directAnswerEvidenceCount": direct_answer_count,
            "sourceMismatchCount": source_mismatch_count,
            "vaultHubPenaltyApplied": vault_hub_count > 0,
        }

    def assemble(
        self,
        *,
        query: str,
        source_type: str | None,
        results: list[SearchResult],
        paper_memory_prefilter: dict[str, Any],
        metadata_filter: dict[str, Any] | None = None,
        query_plan: dict[str, Any] | None = None,
        query_frame: dict[str, Any] | None = None,
    ) -> EvidencePacket:
        frame_payload = normalize_query_frame_dict(query_frame)
        query_intent = classify_query_intent(query)
        paper_family = str(frame_payload.get("family") or dict(query_plan or {}).get("family") or "").strip().lower() or (
            "paper_lookup" if query_intent == "paper_lookup" else "paper_discover" if query_intent == "paper_topic" else "concept_explainer" if normalize_source_type(source_type) == "paper" and query_intent == "definition" else "general"
        )
        domain_pack = get_domain_pack(source_type=source_type)
        if domain_pack is not None and frame_payload:
            evidence_policy = dict(domain_pack.select_evidence_policy(frame_payload) or {})
        elif frame_payload:
            evidence_policy = select_evidence_policy(frame_payload)
        else:
            evidence_policy = policy_for_family(paper_family)
        narrowed_results, paper_answer_scope = _derive_paper_answer_scope(
            query=query,
            source_type=source_type,
            filtered=list(results),
            paper_memory_prefilter=paper_memory_prefilter,
            metadata_filter=metadata_filter,
            query_plan=query_plan,
            query_frame=frame_payload,
            evidence_policy=evidence_policy,
        )
        budgeted_results, evidence_budget = _apply_answer_evidence_budget(
            query=query,
            source_type=source_type,
            filtered=narrowed_results,
            paper_scope=paper_answer_scope,
            query_plan=query_plan,
            query_frame=frame_payload,
            evidence_policy=evidence_policy,
        )
        claims, supporting_beliefs, contradicting_beliefs, belief_updates_suggested = self.collaborator.collect_claim_context(
            budgeted_results
        )
        doc_cache: dict[str, list[dict[str, Any]]] = {}
        parent_ctx_by_result: dict[str, dict[str, Any]] = {}
        for result in budgeted_results:
            parent_ctx_by_result[_result_id(result)] = self.collaborator.resolve_parent_context(result, doc_cache)
        selected_results, evidence, validation = self._dedupe_evidence(
            query=query,
            results=budgeted_results,
            parent_ctx_by_result=parent_ctx_by_result,
        )
        eval_profile = str(self.eval_answer_profile or "candidate-v5").strip().lower()
        selected_results, evidence, reselection = self._reselect_top1_if_needed(
            query=query,
            source_type=source_type,
            results=selected_results,
            evidence=evidence,
            profile=eval_profile,
        )
        validation.update(reselection)
        citations = _build_citations(selected_results, evidence)
        for citation, item in zip(citations, evidence, strict=False):
            item["citation_label"] = citation["label"]
            item["citation_kind"] = citation["kind"]
        answer_signals = self.collaborator.summarize_answer_signals(
            evidence,
            contradicting_beliefs=contradicting_beliefs,
        )
        normalized_requested_source = normalize_source_type(source_type)
        answer_signals = {
            **dict(answer_signals or {}),
            "query_intent": str(dict(query_plan or {}).get("query_intent") or dict(query_plan or {}).get("queryIntent") or classify_query_intent(query)),
            "paper_family": str(dict(query_plan or {}).get("family") or paper_answer_scope.get("paperFamily") or ""),
            "requested_source_type": normalized_requested_source or "all",
            **_paper_definition_signal(
                query=query,
                source_type=source_type,
                results=selected_results,
                evidence=evidence,
                query_plan=query_plan,
                query_frame=frame_payload,
            ),
        }
        substantive_count = int(validation.get("substantiveEvidenceCount") or 0)
        direct_answer_count = int(validation.get("directAnswerEvidenceCount") or 0)
        temporal_grounded_count = int(validation.get("temporalGroundedCount") or 0)
        weak_observed_only_count = int(validation.get("weakObservedAtOnlyCount") or 0)
        source_mismatch_count = int(validation.get("sourceMismatchCount") or 0)
        high_trust_count = int(validation.get("highAuthorityCount") or 0)
        memory_provenance_count = int(validation.get("memoryProvenanceCount") or 0)
        preferred_source_count = int((answer_signals.get("quality_counts") or {}).get("ok", 0))
        top1_item = evidence[0] if evidence else {}
        top1_substantive = bool(validation.get("top1Substantive"))
        top1_rejected_reason = str(validation.get("top1RejectedReason") or "")
        top1_temporal_grounded = _is_temporal_grounded(top1_item) if top1_item else False
        top1_observed_at_only = _is_observed_at_only(top1_item) if top1_item else False
        top1_direct_score = _direct_answer_score(top1_item, query=query) if top1_item else 0.0
        top1_direct = top1_direct_score >= 1.0
        calibration_hardened = eval_profile in {"candidate-v4", "candidate-v5", "candidate-v6", "on-control"}
        paper_family = str(answer_signals.get("paper_family") or paper_answer_scope.get("paperFamily") or frame_payload.get("family") or "")
        unique_selected_paper_ids = {
            _result_paper_id(item)
            for item in selected_results
            if _result_paper_id(item)
        }
        resolved_compare_paper_ids = {
            str(item).strip()
            for item in list(frame_payload.get("resolved_source_ids") or [])
            if str(item).strip()
        }

        is_temporal = _is_temporal_query(query)
        is_abstention = _is_abstention_query(query)
        answerable, answerable_reason, insufficient_reasons = evaluate_answerability(
            AnswerabilityInputs(
                evidence_count=len(evidence),
                substantive_count=substantive_count,
                all_evidence_non_substantive=bool(evidence) and int(validation.get("nonSubstantiveEvidenceCount") or 0) >= len(evidence),
                source_mismatch_count=(
                    source_mismatch_count
                    if source_type and normalize_source_type(source_type) not in {"", "all"}
                    else 0
                ),
                unique_paper_count=len(unique_selected_paper_ids),
                resolved_compare_paper_count=len(resolved_compare_paper_ids),
                requires_multiple_sources=bool(evidence_policy.get("requiresMultipleSources")),
                is_temporal=is_temporal,
                is_abstention=is_abstention,
                calibration_hardened=calibration_hardened,
                eval_profile=eval_profile,
                normalized_requested_source=normalized_requested_source,
                top1_substantive=top1_substantive,
                top1_rejected_reason=top1_rejected_reason,
                top1_temporal_grounded=top1_temporal_grounded,
                top1_observed_at_only=top1_observed_at_only,
                top1_direct_score=top1_direct_score,
                top1_direct=top1_direct,
                direct_answer_count=direct_answer_count,
                temporal_grounded_count=temporal_grounded_count,
                weak_observed_only_count=weak_observed_only_count,
                high_trust_count=high_trust_count,
                memory_provenance_count=memory_provenance_count,
                preferred_source_count=preferred_source_count,
                contradicting_beliefs_present=bool(contradicting_beliefs),
                top1_is_temporal_web_candidate=_is_temporal_web_candidate(top1_item, source_type=source_type),
                top1_has_web_temporal_textual_marker=_has_web_temporal_textual_marker(top1_item),
                paper_top1_non_substantive_refusal=(
                    not top1_substantive
                    and top1_rejected_reason == "non_substantive_top1"
                    and normalize_source_type(top1_item.get("source_type")) == "paper"
                ),
            )
        )

        evidence_packet = {
            "answerable": bool(answerable),
            "selectedEvidenceCount": len(evidence),
            "citationCount": len(citations),
            "insufficientEvidenceReasons": insufficient_reasons,
            "validation": validation,
            "answerableDecisionReason": answerable_reason,
            "top1RejectedReason": str(validation.get("top1RejectedReason") or ""),
            "paperFamily": paper_family,
            "uniquePaperCount": len(unique_selected_paper_ids),
            "freshness": {
                "highFreshnessCount": int(validation.get("highFreshnessCount") or 0),
                "highAuthorityCount": int(validation.get("highAuthorityCount") or 0),
                "temporalGroundedCount": int(validation.get("temporalGroundedCount") or 0),
                "memoryProvenanceCount": int(validation.get("memoryProvenanceCount") or 0),
                "substantiveEvidenceCount": substantive_count,
                "directAnswerEvidenceCount": direct_answer_count,
                "weakObservedAtOnlyCount": weak_observed_only_count,
            },
        }
        context = self.collaborator.build_answer_context(
            filtered=selected_results,
            parent_ctx_by_result=parent_ctx_by_result,
        )
        context_budget = _context_budget_payload(
            pre_budget_results=budgeted_results,
            selected_results=selected_results,
            evidence=evidence,
            context=context,
        )
        return EvidencePacket(
            filtered_results=selected_results,
            evidence=evidence,
            citations=citations,
            context=context,
            answer_signals=answer_signals,
            claims=claims,
            supporting_beliefs=supporting_beliefs,
            contradicting_beliefs=contradicting_beliefs,
            belief_updates_suggested=belief_updates_suggested,
            paper_answer_scope=paper_answer_scope,
            evidence_budget={
                **dict(evidence_budget),
                "maxFamilies": int(dict(evidence_budget).get("maxSources") or 0),
                "maxChunksPerFamily": 1,
                "maxVerifierChunks": 0 if int(context_budget.get("verifierAddedTokens") or 0) <= 0 else 3,
            },
            evidence_policy=dict(evidence_policy),
            evidence_packet={
                **dict(evidence_packet),
                "contextBudget": context_budget,
                "collapsedRelatedEvidence": list(validation.get("collapsedRelatedEvidence") or []),
                "semanticFamilyCount": int(validation.get("semanticFamilyCount") or 0),
            },
        )


__all__ = ["EvidenceAssemblyService", "EvidencePacket"]
