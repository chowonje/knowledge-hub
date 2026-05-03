from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Optional

from knowledge_hub.ai.memory_prefilter import (
    MEMORY_ROUTE_MODE_OFF,
    MEMORY_ROUTE_MODE_ON,
    execute_memory_prefilter,
    memory_route_payload,
    normalize_memory_route_mode_details,
    normalize_memory_route_mode,
)
from knowledge_hub.ai.rag_paper_prefilter import search_with_paper_memory_prefilter
from knowledge_hub.ai.retrieval_pipeline_paper_coverage import (
    PaperCoverageDeps,
    PaperCoverageSearchRuntime,
    PaperCoverageService,
)
from knowledge_hub.ai.retrieval_pipeline_plan import RetrievalPlanBuilder, RetrievalPlanBuilderDeps
from knowledge_hub.ai.retrieval_pipeline_runtime import (
    MemoryPrefilterDeps,
    RetrievalPipelineRuntime,
    RetrievalPipelineRuntimeDeps,
    ScopeDeps,
)
from knowledge_hub.ai.retrieval_pipeline_search_core import RetrievalSearchCore, RetrievalSearchCoreDeps
from knowledge_hub.ai.retrieval_strategy_diagnostics import (
    build_answerability_rerank_diagnostics,
    build_artifact_health_diagnostics,
    build_corrective_retrieval_diagnostics,
    build_retrieval_quality_diagnostics,
    build_retrieval_strategy_diagnostics,
)
from knowledge_hub.ai.rag_scope import get_active_profile, load_topology_index
from knowledge_hub.ai.retrieval import (
    apply_feature_boosts,
    expand_query_with_ontology,
    lexical_search,
    semantic_search,
)
from knowledge_hub.ai.reranker import RerankerConfig, build_reranker
from knowledge_hub.ai.retrieval_fit import (
    apply_query_fit_reranking,
    classify_query_intent,
    normalize_source_type,
)
from knowledge_hub.application.query_frame import normalize_query_frame_dict
from knowledge_hub.core.models import SearchResult
from knowledge_hub.domain.ai_papers.query_plan import normalize_query_plan_dict, paper_family_query_intent
from knowledge_hub.domain.ai_papers.representative import local_title_prefix_rescue_forms
from knowledge_hub.knowledge.graph_signals import analyze_graph_query
from knowledge_hub.papers.prefilter import (
    PAPER_MEMORY_MODE_COMPAT,
    PAPER_MEMORY_MODE_OFF,
    PAPER_MEMORY_MODE_ON,
    normalize_paper_memory_mode_details,
    normalize_paper_memory_mode,
    resolve_paper_memory_prefilter,
)


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


_RESCUE_TOKEN_STOPWORDS = {
    "about",
    "concept",
    "define",
    "definition",
    "describe",
    "explain",
    "explainer",
    "idea",
    "ideas",
    "main",
    "meaning",
    "principle",
    "summary",
    "what",
    "개념",
    "설명",
    "설명해",
    "설명해줘",
    "설명해주세요",
    "쉽게",
    "아이디어",
    "알려줘",
    "원리",
    "의미",
    "정리",
    "정리해줘",
    "정의",
    "핵심",
}
_KOREAN_PARTICLE_SUFFIXES = (
    "으로",
    "에서",
    "에게",
    "한테",
    "처럼",
    "보다",
    "까지",
    "부터",
    "와",
    "과",
    "의",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "도",
    "만",
    "로",
)
_COMPARE_LEXICAL_STOPWORDS = {
    "vs",
    "versus",
    "between",
    "compare",
    "comparison",
    "difference",
    "비교",
    "비교해",
    "비교해줘",
    "차이",
    "model",
    "models",
    "논문",
    "계열",
    "기준",
    "모델",
}
_DISCOVER_LEXICAL_STOPWORDS = {
    "related",
    "recommend",
    "recommended",
    "discover",
    "find",
    "찾아",
    "찾아줘",
    "관련",
    "논문",
}
_SECTION_CARD_RETRIEVAL_OBJECT = "SectionCard"
_CLAIM_CARD_RETRIEVAL_OBJECT = "ClaimCard"
_DOC_SUMMARY_RETRIEVAL_OBJECT = "DocSummary"
_RAW_EVIDENCE_RETRIEVAL_OBJECT = "RawEvidenceUnit"
_CONCEPT_EXPLAINER_LEXICAL_STOPWORDS = {
    "아이디어",
    "핵심",
    "원리",
    "개념",
    "설명",
    "쉽게",
}
_WEB_REFERENCE_LEXICAL_STOPWORDS = {
    "guide",
    "reference",
    "overview",
    "정의",
    "설명",
    "가이드",
    "레퍼런스",
}
_WEB_RELATION_LEXICAL_STOPWORDS = {
    "relationship",
    "relation",
    "related",
    "connected",
    "관계",
    "연결",
    "의존",
}
_WEB_TEMPORAL_RESCUE_TERMS = {"latest", "recent", "updated", "version", "최근", "최신", "업데이트", "버전"}


def _strip_korean_particle(token: str) -> str:
    value = str(token or "").strip()
    for suffix in _KOREAN_PARTICLE_SUFFIXES:
        if len(value) > len(suffix) + 1 and value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _append_unique(values: list[str], candidate: str, *, limit: int | None = None) -> None:
    token = _clean_text(candidate)
    if not token:
        return
    lowered = token.casefold()
    if any(existing.casefold() == lowered for existing in values):
        return
    values.append(token)
    if limit is not None and len(values) > limit:
        del values[limit:]


def _paper_definition_rescue_queries(query: str, *, sqlite_db: Any | None = None, limit: int = 4) -> list[str]:
    text = _clean_text(query)
    if not text:
        return []

    filtered_tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9.+-]+|[가-힣]+", text):
        token = _strip_korean_particle(raw)
        lowered = token.casefold()
        if not token or lowered in _RESCUE_TOKEN_STOPWORDS:
            continue
        if re.fullmatch(r"[가-힣]+", token) and len(token) < 2:
            continue
        if re.fullmatch(r"[A-Za-z0-9.+-]+", token) and len(token) < 2:
            continue
        filtered_tokens.append(token)

    rescue_queries: list[str] = []
    if sqlite_db:
        for title in local_title_prefix_rescue_forms(filtered_tokens, sqlite_db=sqlite_db):
            _append_unique(rescue_queries, title, limit=limit)
    if filtered_tokens:
        _append_unique(rescue_queries, " ".join(filtered_tokens[:4]), limit=limit)
        for token in filtered_tokens[:3]:
            _append_unique(rescue_queries, token, limit=limit)

    if sqlite_db:
        try:
            from knowledge_hub.learning.resolver import EntityResolver

            resolver = EntityResolver(sqlite_db)
            resolved_forms = list(rescue_queries) if rescue_queries else filtered_tokens[:3]
            for form in resolved_forms:
                identity = resolver.resolve(form, entity_type="concept")
                if identity is None:
                    continue
                _append_unique(rescue_queries, str(identity.display_name or ""), limit=limit)
                for alias in list(identity.aliases or [])[:3]:
                    _append_unique(rescue_queries, str(alias or ""), limit=limit)
        except Exception:
            pass

    return rescue_queries[:limit]


def _family_query_limits(*, normalized_source: str, paper_family: str) -> dict[str, int]:
    if normalized_source == "paper" and paper_family == "concept_explainer":
        return {
            "planned_term_limit": 2,
            "rescue_query_limit": 1,
            "lexical_form_limit": 2,
            "representative_scope_limit": 1,
        }
    if normalized_source == "paper" and paper_family == "paper_compare":
        return {
            "planned_term_limit": 8,
            "rescue_query_limit": 4,
            "lexical_form_limit": 6,
            "representative_scope_limit": 3,
        }
    return {
        "planned_term_limit": 6,
        "rescue_query_limit": 4,
        "lexical_form_limit": 4,
        "representative_scope_limit": 3,
    }


def _build_expanded_queries(
    *,
    query_text: str,
    ontology_queries: list[str],
    planned_terms: list[str],
    rescue_queries: list[str],
    normalized_source: str,
    paper_family: str,
) -> list[str]:
    limits = _family_query_limits(normalized_source=normalized_source, paper_family=paper_family)
    expanded_queries: list[str] = []
    _append_unique(expanded_queries, query_text)
    if normalized_source == "paper" and paper_family == "concept_explainer":
        for term in planned_terms[: int(limits["planned_term_limit"])]:
            _append_unique(expanded_queries, term, limit=4)
        for rescue in rescue_queries[: int(limits["rescue_query_limit"])]:
            _append_unique(expanded_queries, rescue, limit=4)
        return expanded_queries[:4]

    for query_form in ontology_queries:
        _append_unique(expanded_queries, query_form, limit=8)
    for term in planned_terms[: int(limits["planned_term_limit"])]:
        _append_unique(expanded_queries, term, limit=8)
    for rescue in rescue_queries[: int(limits["rescue_query_limit"])]:
        _append_unique(expanded_queries, rescue, limit=8)
    return expanded_queries[:8]


def _is_title_like_seed(candidate: str) -> bool:
    token = _clean_text(candidate)
    if not token:
        return False
    if "/" in token or "\\" in token:
        return False
    words = token.split()
    if re.search(r"[A-Za-z]", token) and re.search(r"\d", token):
        return True
    return len(words) >= 3 or ":" in token or len(token) >= 24


def _is_compact_title_rescue(candidate: str) -> bool:
    token = _clean_text(candidate)
    if not token:
        return False
    words = token.split()
    return bool(re.search(r"[A-Za-z]", token) and re.search(r"\d", token)) or len(words) <= 2


def _lexical_seed_candidates(
    *,
    query_text: str,
    query_frame_payload: dict[str, Any],
    planned_terms: list[str],
    rescue_queries: list[str],
    normalized_source: str,
    paper_family: str,
) -> list[str]:
    entities = [
        _clean_text(item)
        for item in list(query_frame_payload.get("entities") or [])
        if _clean_text(item)
    ]
    if normalized_source == "paper" and paper_family == "concept_explainer":
        alias_candidates: list[str] = []
        compact_title_rescue_candidates: list[str] = []
        long_title_rescue_candidates: list[str] = []
        for item in entities:
            lowered = _clean_text(item).casefold()
            if lowered in _CONCEPT_EXPLAINER_LEXICAL_STOPWORDS:
                continue
            alias_candidates.append(item)
        for item in rescue_queries:
            if _is_title_like_seed(item):
                if _is_compact_title_rescue(item):
                    compact_title_rescue_candidates.append(item)
                else:
                    long_title_rescue_candidates.append(item)
                continue
            lowered = _clean_text(item).casefold()
            if lowered in _CONCEPT_EXPLAINER_LEXICAL_STOPWORDS:
                continue
            alias_candidates.append(item)
        for item in planned_terms:
            if _is_title_like_seed(item):
                if _is_compact_title_rescue(item):
                    compact_title_rescue_candidates.append(item)
                else:
                    long_title_rescue_candidates.append(item)
        seeds: list[str] = []
        lexical_limit = int(_family_query_limits(normalized_source=normalized_source, paper_family=paper_family)["lexical_form_limit"])
        max_depth = max(len(compact_title_rescue_candidates), len(alias_candidates), len(long_title_rescue_candidates))
        for idx in range(max_depth):
            if idx < len(compact_title_rescue_candidates):
                _append_unique(seeds, compact_title_rescue_candidates[idx], limit=lexical_limit)
            if idx < len(alias_candidates):
                _append_unique(seeds, alias_candidates[idx], limit=lexical_limit)
            if idx < len(long_title_rescue_candidates):
                _append_unique(seeds, long_title_rescue_candidates[idx], limit=lexical_limit)
        return seeds
    if normalized_source == "paper" and paper_family == "paper_lookup":
        title_rescue_candidates = [item for item in planned_terms if _is_title_like_seed(item)]
        non_title_candidates = [item for item in planned_terms if not _is_title_like_seed(item)]
        return [*title_rescue_candidates, query_text, *non_title_candidates, *entities]
    if normalized_source == "paper" and paper_family == "paper_compare":
        title_rescue_candidates: list[str] = []
        alias_candidates: list[str] = []
        entity_candidates: list[str] = []
        seeds: list[str] = []
        lexical_limit = int(_family_query_limits(normalized_source=normalized_source, paper_family=paper_family)["lexical_form_limit"])
        for item in planned_terms:
            lowered = _clean_text(item).casefold()
            if not lowered or lowered in _COMPARE_LEXICAL_STOPWORDS:
                continue
            if _is_title_like_seed(item):
                title_rescue_candidates.append(item)
            else:
                alias_candidates.append(item)
        for item in entities:
            lowered = _clean_text(item).casefold()
            if lowered in _COMPARE_LEXICAL_STOPWORDS:
                continue
            entity_candidates.append(item)
        for item in [*title_rescue_candidates, *alias_candidates, *entity_candidates]:
            _append_unique(seeds, item, limit=lexical_limit)
        return seeds
    if normalized_source == "paper" and paper_family == "paper_discover":
        seeds: list[str] = [query_text]
        for item in planned_terms:
            lowered = _clean_text(item).casefold()
            if lowered in _DISCOVER_LEXICAL_STOPWORDS:
                continue
            seeds.append(item)
        return seeds
    if normalized_source == "web" and paper_family == "reference_explainer":
        seeds: list[str] = []
        for item in [*entities, *planned_terms]:
            lowered = _clean_text(item).casefold()
            if not lowered or lowered in _WEB_REFERENCE_LEXICAL_STOPWORDS:
                continue
            _append_unique(seeds, item, limit=3)
        return seeds or [query_text]
    if normalized_source == "web" and paper_family == "temporal_update":
        seeds: list[str] = []
        for item in [query_text, *planned_terms]:
            token = _clean_text(item)
            if not token:
                continue
            _append_unique(seeds, token, limit=3)
        for term in _WEB_TEMPORAL_RESCUE_TERMS:
            _append_unique(seeds, term, limit=3)
        return seeds[:3]
    if normalized_source == "web" and paper_family == "relation_explainer":
        seeds: list[str] = []
        for item in [*entities, *planned_terms]:
            lowered = _clean_text(item).casefold()
            if not lowered or lowered in _WEB_RELATION_LEXICAL_STOPWORDS:
                continue
            _append_unique(seeds, item, limit=4)
        return seeds or [query_text]
    if normalized_source == "web" and paper_family == "source_disambiguation":
        seeds: list[str] = []
        for item in planned_terms:
            _append_unique(seeds, item, limit=3)
        _append_unique(seeds, "reference source", limit=3)
        _append_unique(seeds, "latest update", limit=3)
        return seeds or [query_text]
    generic_title_rescues = [item for item in rescue_queries if _is_title_like_seed(item)]
    if generic_title_rescues:
        seeds: list[str] = []
        for item in [*generic_title_rescues, query_text, *planned_terms, *rescue_queries]:
            _append_unique(seeds, item, limit=4)
        return seeds
    return [query_text, *planned_terms, *rescue_queries]


def _build_lexical_query_forms(
    *,
    query_text: str,
    query_frame_payload: dict[str, Any],
    planned_terms: list[str],
    rescue_queries: list[str],
    normalized_source: str,
    paper_family: str,
) -> list[str]:
    limits = _family_query_limits(normalized_source=normalized_source, paper_family=paper_family)
    lexical_query_forms: list[str] = []
    for candidate in _lexical_seed_candidates(
        query_text=query_text,
        query_frame_payload=query_frame_payload,
        planned_terms=planned_terms,
        rescue_queries=rescue_queries,
        normalized_source=normalized_source,
        paper_family=paper_family,
    ):
        _append_unique(lexical_query_forms, candidate, limit=int(limits["lexical_form_limit"]))
    if not lexical_query_forms:
        _append_unique(lexical_query_forms, query_text, limit=int(limits["lexical_form_limit"]))
    return lexical_query_forms[: int(limits["lexical_form_limit"])]


def _temporal_query_signals(query: str) -> dict[str, Any]:
    text = str(query or "").strip()
    before_match = re.search(r"\b(before|prior to|earlier than)\s+(20\d{2})\b|([0-9]{4})\s*이전|([0-9]{4})\s*당시", text, re.IGNORECASE)
    after_match = re.search(r"\b(after|since)\s+(20\d{2})\b|([0-9]{4})\s*이후", text, re.IGNORECASE)
    latest = bool(re.search(r"\b(latest|recent|updated|update|newest|changed since)\b|최근|최신|업데이트", text, re.IGNORECASE))
    target_year = ""
    mode = "none"
    if before_match:
        mode = "before"
        target_year = next((group for group in before_match.groups() if group and group.isdigit()), "")
    elif after_match:
        mode = "after"
        target_year = next((group for group in after_match.groups() if group and group.isdigit()), "")
    elif latest:
        mode = "latest"
    return {
        "enabled": mode != "none",
        "mode": mode,
        "targetYear": target_year,
    }


def _classify_enrichment_route(query: str, *, query_intent: str) -> dict[str, Any]:
    text = str(query or "").strip()
    lowered = text.lower()

    concept_patterns = [
        r"\bwhat is\b",
        r"\bmeaning of\b",
        r"\bdefine\b",
        r"\bdefinition\b",
        r"\bconcept\b",
        r"\bentity\b",
        r"\brelationship\b",
        r"\bgraph\b",
        r"\bdisambiguat",
    ]
    concept_tokens = ["차이", "정의", "개념", "의미", "관계", "구분"]
    cluster_patterns = [
        r"\brecommend\b",
        r"\brecommended\b",
        r"\breading list\b",
        r"\bsimilar\b",
        r"\brelated papers\b",
        r"\bgroup(?:ing)?\b",
        r"\bcluster\b",
        r"\bbucket\b",
        r"\borganize\b",
    ]
    cluster_tokens = ["추천", "관련 논문", "비슷한", "묶어", "그룹", "클러스터"]
    graph_patterns = [
        r"\bhow .* related\b",
        r"\bconnected to\b",
        r"\bdepends on\b",
        r"\blinked to\b",
        r"\brelation(?:ship)?\b",
        r"\bgraph\b",
        r"\bontology\b",
    ]
    graph_tokens = ["관계", "연결", "의존", "그래프", "온톨로지"]

    concept_match = (
        query_intent == "definition"
        or any(re.search(pattern, lowered, re.IGNORECASE) for pattern in concept_patterns)
        or any(token in text for token in concept_tokens)
    )
    graph_match = any(re.search(pattern, lowered, re.IGNORECASE) for pattern in graph_patterns) or any(
        token in text for token in graph_tokens
    )
    cluster_match = any(re.search(pattern, lowered, re.IGNORECASE) for pattern in cluster_patterns) or any(
        token in text for token in cluster_tokens
    )

    if cluster_match:
        return {
            "route": "cluster_assist",
            "ontologyEligible": False,
            "clusterEligible": True,
            "reason": "recommendation_or_grouping_query",
        }
    if concept_match or graph_match:
        return {
            "route": "ontology_assist",
            "ontologyEligible": True,
            "clusterEligible": False,
            "reason": "concept_or_disambiguation_query" if concept_match else "graph_heavy_relationship_query",
        }
    if query_intent in {"paper_lookup", "paper_topic", "comparison", "evaluation", "topic_lookup"}:
        return {
            "route": "memory_heavy",
            "ontologyEligible": False,
            "clusterEligible": False,
            "reason": "memory_first_query_family",
        }
    return {
        "route": "core_only",
        "ontologyEligible": False,
        "clusterEligible": False,
        "reason": "default_core_path",
    }


def _enrichment_mode(*, ontology_used: bool, cluster_used: bool) -> str:
    if ontology_used and cluster_used:
        return "ontology+cluster"
    if ontology_used:
        return "ontology"
    if cluster_used:
        return "cluster"
    return "none"


def _build_enrichment_diagnostics(
    plan: "RetrievalPlan",
    *,
    ontology_used: bool,
    cluster_used: bool,
    preferred_sources: list[str] | None = None,
    topology_applied: bool = False,
    cluster_ids: list[str] | None = None,
    cluster_count: int = 0,
) -> dict[str, Any]:
    return {
        "eligible": bool(plan.ontology_assist_eligible or plan.cluster_assist_eligible),
        "used": bool(ontology_used or cluster_used),
        "mode": _enrichment_mode(ontology_used=ontology_used, cluster_used=cluster_used),
        "reason": str(plan.enrichment_reason or ""),
        "queryIntent": str(plan.query_intent or ""),
        "enrichmentRoute": str(plan.enrichment_route or ""),
        "ontologyEligible": bool(plan.ontology_assist_eligible),
        "clusterEligible": bool(plan.cluster_assist_eligible),
        "ontologyUsed": bool(ontology_used),
        "clusterUsed": bool(cluster_used),
        "clusterCount": int(cluster_count),
        "clusterIds": list(cluster_ids or []),
        "preferredSources": list(preferred_sources or []),
        "topologyApplied": bool(topology_applied),
    }


def _memory_prior_config(query_intent: str, *, temporal_route: bool, top_k: int) -> tuple[float, int]:
    weight = 0.06
    fallback_window = max(2, min(8, int(top_k)))
    if query_intent in {"comparison", "evaluation", "paper_lookup", "paper_topic", "topic_lookup"}:
        weight = 0.08
    elif query_intent in {"implementation", "howto"}:
        weight = 0.065
    if temporal_route:
        weight = max(weight, 0.1)
        fallback_window = max(fallback_window, min(10, int(top_k) + 2))
    return round(weight, 3), int(fallback_window)


def _context_budget_config(query_intent: str, *, top_k: int) -> tuple[int, float, float]:
    token_budget = max(600, int(top_k) * 260)
    memory_compression_target = 0.48
    chunk_expansion_threshold = 0.62
    if query_intent in {"comparison", "evaluation", "paper_lookup", "paper_topic"}:
        token_budget = max(720, int(top_k) * 280)
        memory_compression_target = 0.42
        chunk_expansion_threshold = 0.68
    elif query_intent in {"implementation", "howto"}:
        token_budget = max(760, int(top_k) * 300)
        memory_compression_target = 0.52
        chunk_expansion_threshold = 0.58
    return int(token_budget), round(memory_compression_target, 3), round(chunk_expansion_threshold, 3)


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


def _retrieval_sort_score(result: SearchResult) -> float:
    extras = dict(getattr(result, "lexical_extras", {}) or {})
    try:
        return float(extras.get("retrieval_sort_score"))
    except Exception:
        return _safe_float(getattr(result, "score", 0.0), 0.0)


def _retrieval_sort_key(result: SearchResult, *, prefer_lexical: bool = False) -> tuple[float, float, float, float]:
    if prefer_lexical:
        return (
            _retrieval_sort_score(result),
            _safe_float(getattr(result, "score", 0.0), 0.0),
            _safe_float(getattr(result, "lexical_score", 0.0), 0.0),
            _safe_float(getattr(result, "semantic_score", 0.0), 0.0),
        )
    return (
        _retrieval_sort_score(result),
        _safe_float(getattr(result, "score", 0.0), 0.0),
        _safe_float(getattr(result, "semantic_score", 0.0), 0.0),
        _safe_float(getattr(result, "lexical_score", 0.0), 0.0),
    )


def _merge_source_filter(base_filter: Optional[dict[str, Any]], source_type: str) -> Optional[dict[str, Any]]:
    merged = dict(base_filter or {})
    current = normalize_source_type(merged.get("source_type"))
    if current and current != source_type:
        return None
    merged["source_type"] = source_type
    return merged


def _merge_filter_dicts(
    base_filter: Optional[dict[str, Any]],
    extra_filter: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    base = dict(base_filter or {})
    extra = dict(extra_filter or {})
    if not base:
        return extra or None
    if not extra:
        return base
    if "$and" not in base and "$and" not in extra:
        merged = dict(base)
        merged.update(extra)
        return merged
    clauses: list[dict[str, Any]] = []
    for payload in (base, extra):
        nested = payload.get("$and")
        if isinstance(nested, list):
            clauses.extend([dict(item) for item in nested if isinstance(item, dict)])
        else:
            clauses.append(dict(payload))
    return {"$and": clauses}


def _derive_frame_prefilter(
    *,
    source_scope: str,
    paper_family: str,
    metadata_filter: Optional[dict[str, Any]],
    query_plan: dict[str, Any],
    query_frame: dict[str, Any],
) -> dict[str, Any]:
    base_filter = _merge_filter_dicts(metadata_filter, query_frame.get("metadata_filter")) or {}
    if source_scope and "source_type" not in base_filter:
        base_filter["source_type"] = source_scope

    canonical_entities = list(
        dict.fromkeys(
            _clean_text(item)
            for item in list(query_frame.get("canonical_entity_ids") or [])
            if _clean_text(item)
        )
    )[:6]
    resolved_source_ids = list(
        dict.fromkeys(
            _clean_text(item)
            for item in [
                *list(query_frame.get("resolved_source_ids") or []),
                *list(query_plan.get("resolved_paper_ids") or []),
                *list(query_plan.get("resolvedPaperIds") or []),
            ]
            if _clean_text(item)
        )
    )[:3]
    explicit_scope_key = next(
        (
            key
            for key in ("arxiv_id", "paper_id", "doi")
            if _clean_text(base_filter.get(key))
        ),
        "",
    )
    effective_filter = dict(base_filter)
    scope_applied = False
    reason = "metadata_filter" if metadata_filter else "source_scope" if source_scope else "none"
    reference_source_applied = bool(effective_filter.get("reference_only"))
    watchlist_scope_applied = bool(_clean_text(effective_filter.get("watchlist_scope")))

    if source_scope == "paper":
        if explicit_scope_key:
            scope_applied = True
            reason = "explicit_metadata_filter"
        elif paper_family == "paper_lookup" and len(resolved_source_ids) == 1:
            effective_filter = _merge_filter_dicts(
                effective_filter,
                {
                    "source_type": "paper",
                    "arxiv_id": resolved_source_ids[0],
                },
            ) or {}
            scope_applied = True
            reason = "resolved_source_id"
        elif paper_family == "concept_explainer" and resolved_source_ids:
            reason = "representative_candidate_narrowing"
        elif paper_family == "concept_explainer" and canonical_entities:
            reason = "canonical_entity_linking"
        elif paper_family == "paper_compare" and resolved_source_ids:
            reason = "resolved_compare_candidates"
        elif paper_family == "paper_discover" and (canonical_entities or resolved_source_ids):
            reason = "lightweight_discovery_expansion"
    elif source_scope == "web":
        original_filter = dict(metadata_filter or {})
        explicit_url = _clean_text(original_filter.get("canonical_url") or original_filter.get("url") or original_filter.get("source_url"))
        explicit_doc = _clean_text(original_filter.get("document_id"))
        resolved_url = next((item for item in resolved_source_ids if item.startswith("http://") or item.startswith("https://")), "")
        resolved_doc = next((item for item in resolved_source_ids if item.startswith("web_")), "")
        if explicit_url or explicit_doc:
            scope_applied = True
            reason = "explicit_metadata_filter"
        elif resolved_url or resolved_doc:
            if resolved_url:
                effective_filter = _merge_filter_dicts(effective_filter, {"canonical_url": resolved_url}) or {}
            if resolved_doc:
                effective_filter = _merge_filter_dicts(effective_filter, {"document_id": resolved_doc}) or {}
            scope_applied = True
            reason = "resolved_source_id"
        elif bool(effective_filter.get("latest_only") or effective_filter.get("temporal_required")):
            reason = "temporal_grounding_required"
        elif watchlist_scope_applied:
            reason = "watchlist_scope"
        elif reference_source_applied:
            reason = "reference_source_bias"
        elif canonical_entities:
            reason = "canonical_entity_linking"
        web_url = _clean_text(effective_filter.get("url") or effective_filter.get("canonical_url") or effective_filter.get("source_url"))
        if web_url:
            effective_filter["url"] = web_url
        effective_filter.pop("canonical_url", None)
        effective_filter.pop("source_url", None)

    return {
        "effective_filter": effective_filter,
        "resolved_source_scope_applied": scope_applied,
        "canonical_entities_applied": canonical_entities,
        "metadata_filter_applied": effective_filter,
        "prefilter_reason": reason,
        "resolved_source_ids": resolved_source_ids,
        "reference_source_applied": reference_source_applied,
        "watchlist_scope_applied": watchlist_scope_applied,
    }


def _source_label_for_result(result: SearchResult) -> str:
    return str(normalize_source_type((result.metadata or {}).get("source_type")) or "").strip().lower()


def _result_paper_id(result: SearchResult) -> str:
    metadata = dict(getattr(result, "metadata", {}) or {})
    for key in ("arxiv_id", "paper_id"):
        token = str(metadata.get(key) or "").strip()
        if token:
            return token
    return ""


def _memory_mode_fallback_reason(*, mode: str, reason: str) -> str:
    normalized_mode = normalize_memory_route_mode(mode)
    if normalized_mode != MEMORY_ROUTE_MODE_ON:
        return ""
    token = str(reason or "").strip().lower()
    if token in {"no_memory_hits", "mixed_fallback_no_hit"}:
        return "memory_empty_fallback"
    if token in {
        "hits_not_usable",
        "vault_chunk_fallback",
        "web_chunk_fallback",
        "paper_chunk_fallback",
        "memory_prefilter_chunk_fallback",
        "source_not_paper",
        "source_not_supported",
    }:
        return "memory_hits_not_usable"
    if token in {"sqlite_unavailable", "prefilter_search_failed", "disabled"}:
        return "memory_prefilter_unavailable"
    if token.endswith("_fallback") or token.endswith("_fallback_success"):
        return "memory_empty_fallback"
    return "memory_prefilter_unavailable"


def _top_signal_items(existing: list[dict[str, Any]] | None, additions: dict[str, float], limit: int = 5) -> list[dict[str, Any]]:
    merged: dict[str, float] = {}
    for item in existing or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        try:
            merged[name] = round(float(item.get("value") or 0.0), 6)
        except Exception:
            continue
    for name, value in additions.items():
        token = str(name or "").strip()
        if not token:
            continue
        try:
            parsed = float(value)
        except Exception:
            continue
        if abs(parsed) <= 0.000001:
            continue
        merged[token] = round(parsed, 6)
    ranked = [{"name": name, "value": value} for name, value in merged.items()]
    ranked.sort(key=lambda item: abs(float(item["value"])), reverse=True)
    return ranked[: max(1, int(limit))]


def _first_nonempty(*values: Any) -> str:
    for value in values:
        token = _clean_text(value)
        if token:
            return token
    return ""


def _candidate_budgets_for_intent(intent: str, *, top_k: int, source_scope: str | None) -> dict[str, int]:
    if source_scope:
        return {str(source_scope): max(3, int(top_k) * 2)}

    if intent in {"definition", "comparison", "evaluation", "paper_lookup", "topic_lookup"}:
        return {
            "vault": max(3, int(top_k)),
            "paper": max(4, int(top_k) * 2),
            "web": max(2, int(top_k)),
            "concept": max(3, int(top_k)),
        }
    if intent == "paper_topic":
        return {
            "vault": max(2, int(top_k)),
            "paper": max(8, int(top_k) * 3),
            "web": max(2, int(top_k)),
            "concept": max(3, int(top_k)),
        }
    if intent in {"implementation", "howto"}:
        return {
            "vault": max(4, int(top_k) * 2),
            "paper": max(3, int(top_k)),
            "web": max(2, int(top_k)),
            "concept": max(2, int(top_k) // 2),
        }
    return {
        "vault": max(4, int(top_k) * 2),
        "paper": max(3, int(top_k)),
        "web": max(2, int(top_k)),
        "concept": max(2, int(top_k) // 2),
    }


def _preserve_parent_diversity(results: list[SearchResult]) -> list[SearchResult]:
    if len(results) <= 2:
        return results

    parent_order: list[str] = []
    grouped: dict[str, list[SearchResult]] = {}
    for item in results:
        parent_key = str((item.metadata or {}).get("parent_id") or "").strip()
        if not parent_key:
            parent_key = str((item.metadata or {}).get("file_path") or "").strip()
        if not parent_key:
            parent_key = str((item.metadata or {}).get("title") or "").strip()
        if parent_key not in grouped:
            parent_order.append(parent_key)
            grouped[parent_key] = []
        grouped[parent_key].append(item)

    if len(grouped) == len(results):
        return results

    diversified: list[SearchResult] = []
    depth = 0
    while len(diversified) < len(results):
        progressed = False
        for parent_key in parent_order:
            siblings = grouped[parent_key]
            if depth < len(siblings):
                diversified.append(siblings[depth])
                progressed = True
        if not progressed:
            break
        depth += 1
    return diversified


@dataclass(frozen=True)
class RetrievalPlan:
    query: str
    source_scope: str
    query_intent: str
    paper_family: str
    retrieval_mode: str
    memory_mode: str
    candidate_budgets: dict[str, int]
    query_plan: dict[str, Any]
    query_frame: dict[str, Any]
    temporal_signals: dict[str, Any]
    temporal_route_applied: bool
    memory_prior_weight: float
    fallback_window: int
    token_budget: int
    memory_compression_target: float
    chunk_expansion_threshold: float
    rerank_strategy: str = "query_fit_plus_graph"
    context_expansion_policy: str = "supplemental_opt_in"
    ontology_expansion_enabled: bool = False
    enrichment_route: str = "core_only"
    enrichment_reason: str = "default_core_path"
    ontology_assist_eligible: bool = False
    cluster_assist_eligible: bool = False
    resolved_source_scope_applied: bool = False
    canonical_entities_applied: tuple[str, ...] = ()
    metadata_filter_applied: dict[str, Any] = field(default_factory=dict)
    prefilter_reason: str = "none"
    reference_source_applied: bool = False
    watchlist_scope_applied: bool = False
    complexity_class: str = "local_lookup"
    budget_reason: str = "default_core_retrieval"
    retrieval_budget: dict[str, Any] = field(default_factory=dict)
    retry_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sourceScope"] = payload.pop("source_scope")
        payload["queryIntent"] = payload.pop("query_intent")
        payload["paperFamily"] = payload.pop("paper_family")
        payload["retrievalMode"] = payload.pop("retrieval_mode")
        payload["memoryMode"] = payload.pop("memory_mode")
        payload["candidateBudgets"] = dict(payload.pop("candidate_budgets"))
        payload["queryPlan"] = dict(payload.pop("query_plan") or {})
        payload["queryFrame"] = dict(payload.pop("query_frame") or {})
        payload["temporalSignals"] = dict(payload.pop("temporal_signals"))
        payload["temporalRouteApplied"] = bool(payload.pop("temporal_route_applied"))
        payload["memoryPriorWeight"] = float(payload.pop("memory_prior_weight"))
        payload["fallbackWindow"] = int(payload.pop("fallback_window"))
        payload["tokenBudget"] = int(payload.pop("token_budget"))
        payload["memoryCompressionTarget"] = float(payload.pop("memory_compression_target"))
        payload["chunkExpansionThreshold"] = float(payload.pop("chunk_expansion_threshold"))
        payload["rerankStrategy"] = payload.pop("rerank_strategy")
        payload["contextExpansionPolicy"] = payload.pop("context_expansion_policy")
        payload["ontologyExpansionEnabled"] = bool(payload.pop("ontology_expansion_enabled"))
        payload["enrichmentRoute"] = payload.pop("enrichment_route")
        payload["enrichmentReason"] = payload.pop("enrichment_reason")
        payload["ontologyAssistEligible"] = bool(payload.pop("ontology_assist_eligible"))
        payload["clusterAssistEligible"] = bool(payload.pop("cluster_assist_eligible"))
        payload["resolvedSourceScopeApplied"] = bool(payload.pop("resolved_source_scope_applied"))
        payload["canonicalEntitiesApplied"] = list(payload.pop("canonical_entities_applied") or [])
        payload["metadataFilterApplied"] = dict(payload.pop("metadata_filter_applied") or {})
        payload["prefilterReason"] = payload.pop("prefilter_reason")
        payload["referenceSourceApplied"] = bool(payload.pop("reference_source_applied"))
        payload["watchlistScopeApplied"] = bool(payload.pop("watchlist_scope_applied"))
        payload["complexityClass"] = payload.pop("complexity_class")
        payload["budgetReason"] = payload.pop("budget_reason")
        payload["retrievalBudget"] = dict(payload.pop("retrieval_budget") or {})
        payload["retryPolicy"] = dict(payload.pop("retry_policy") or {})
        return payload


@dataclass
class RetrievalPipelineResult:
    results: list[SearchResult]
    plan: RetrievalPlan
    candidate_sources: list[dict[str, Any]]
    memory_route: dict[str, Any]
    memory_prefilter: dict[str, Any]
    paper_memory_prefilter: dict[str, Any]
    rerank_signals: dict[str, Any]
    context_expansion: dict[str, Any]
    related_clusters: list[dict[str, Any]] = field(default_factory=list)
    active_profile: dict[str, Any] | None = None
    source_scope_enforced: bool = False
    mixed_fallback_used: bool = False
    v2_diagnostics: dict[str, Any] = field(default_factory=dict)
    retrieval_strategy: dict[str, Any] = field(default_factory=dict)
    retrieval_quality: dict[str, Any] = field(default_factory=dict)
    answerability_rerank: dict[str, Any] = field(default_factory=dict)
    corrective_retrieval: dict[str, Any] = field(default_factory=dict)
    artifact_health: dict[str, Any] = field(default_factory=dict)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "retrievalPlan": self.plan.to_dict(),
            "candidateSources": [dict(item) for item in self.candidate_sources],
            "memoryRoute": dict(self.memory_route),
            "memoryPrefilter": dict(self.memory_prefilter),
            "paperMemoryPrefilter": dict(self.paper_memory_prefilter),
            "rerankSignals": dict(self.rerank_signals),
            "contextExpansion": dict(self.context_expansion),
            "memoryRelationsUsed": list(self.memory_prefilter.get("memoryRelationsUsed") or []),
            "temporalSignals": dict(self.memory_prefilter.get("temporalSignals") or self.plan.temporal_signals),
            "sourceScopeEnforced": bool(self.source_scope_enforced),
            "mixedFallbackUsed": bool(self.mixed_fallback_used),
            "retrievalStrategy": dict(self.retrieval_strategy),
            "retrievalQuality": dict(self.retrieval_quality),
            "answerabilityRerank": dict(self.answerability_rerank),
            "correctiveRetrieval": dict(self.corrective_retrieval),
            "artifactHealth": dict(self.artifact_health),
            "v2": dict(self.v2_diagnostics),
        }


class RetrievalPipelineService:
    def __init__(self, searcher: Any):
        self.searcher = searcher
        self._ctx = getattr(searcher, "_ctx", None)
        self._caches = getattr(searcher, "_caches", None)
        self._reranker_cache: tuple[str, Any] | None = None

    def _paper_coverage_service(self) -> PaperCoverageService:
        search_runtime = PaperCoverageSearchRuntime(
            embedder=getattr(self.searcher, "embedder", None),
            database=getattr(self.searcher, "database", None),
            sqlite_db=getattr(self.searcher, "sqlite_db", None),
            build_retrieval_ranking_signals_fn=getattr(self.searcher, "_build_retrieval_ranking_signals", None),
        )
        deps = PaperCoverageDeps(
            search_runtime=search_runtime,
            apply_feature_boosts_fn=apply_feature_boosts,
            lexical_search_fn=lexical_search,
            semantic_search_fn=semantic_search,
            merge_filter_dicts_fn=_merge_filter_dicts,
            clean_text_fn=_clean_text,
            safe_float_fn=_safe_float,
            result_id_fn=_result_id,
            retrieval_sort_key_fn=_retrieval_sort_key,
            result_paper_id_fn=_result_paper_id,
            top_signal_items_fn=_top_signal_items,
            first_nonempty_fn=_first_nonempty,
        )
        return PaperCoverageService(deps)

    def _plan_builder(self) -> RetrievalPlanBuilder:
        deps = RetrievalPlanBuilderDeps(
            normalize_query_frame_dict_fn=normalize_query_frame_dict,
            normalize_source_type_fn=normalize_source_type,
            normalize_query_plan_dict_fn=normalize_query_plan_dict,
            paper_family_query_intent_fn=paper_family_query_intent,
            classify_query_intent_fn=classify_query_intent,
            candidate_budgets_for_intent_fn=_candidate_budgets_for_intent,
            temporal_query_signals_fn=_temporal_query_signals,
            memory_prior_config_fn=_memory_prior_config,
            context_budget_config_fn=_context_budget_config,
            classify_enrichment_route_fn=_classify_enrichment_route,
            derive_frame_prefilter_fn=_derive_frame_prefilter,
            normalize_memory_route_mode_fn=normalize_memory_route_mode,
            clean_text_fn=_clean_text,
            retrieval_plan_type=RetrievalPlan,
        )
        return RetrievalPlanBuilder(deps)

    def _search_core(self) -> RetrievalSearchCore:
        deps = RetrievalSearchCoreDeps(
            searcher=self.searcher,
            apply_feature_boosts_fn=apply_feature_boosts,
            expand_query_with_ontology_fn=expand_query_with_ontology,
            lexical_search_fn=lexical_search,
            semantic_search_fn=semantic_search,
            apply_query_fit_reranking_fn=apply_query_fit_reranking,
            classify_query_intent_fn=classify_query_intent,
            normalize_source_type_fn=normalize_source_type,
            candidate_budgets_for_intent_fn=_candidate_budgets_for_intent,
            paper_definition_rescue_queries_fn=_paper_definition_rescue_queries,
            build_expanded_queries_fn=_build_expanded_queries,
            build_lexical_query_forms_fn=_build_lexical_query_forms,
            family_query_limits_fn=_family_query_limits,
            merge_filter_dicts_fn=_merge_filter_dicts,
            merge_source_filter_fn=_merge_source_filter,
            safe_float_fn=_safe_float,
            safe_int_fn=_safe_int,
            clean_text_fn=_clean_text,
            result_id_fn=_result_id,
            retrieval_sort_key_fn=_retrieval_sort_key,
            preserve_parent_diversity_fn=_preserve_parent_diversity,
            source_label_for_result_fn=_source_label_for_result,
            top_signal_items_fn=_top_signal_items,
            ensure_compare_result_coverage_fn=self._ensure_compare_result_coverage,
        )
        return RetrievalSearchCore(deps)

    def _runtime(self) -> RetrievalPipelineRuntime:
        deps = RetrievalPipelineRuntimeDeps(
            searcher=self.searcher,
            ctx=self._ctx,
            caches=self._caches,
            result_id_fn=_result_id,
            retrieval_sort_key_fn=_retrieval_sort_key,
            safe_float_fn=_safe_float,
            result_paper_id_fn=_result_paper_id,
            source_label_for_result_fn=_source_label_for_result,
            top_signal_items_fn=_top_signal_items,
            preserve_parent_diversity_fn=_preserve_parent_diversity,
            normalize_source_type_fn=normalize_source_type,
            reranker_config_fn=self._reranker_config,
            get_reranker_fn=self._get_reranker,
        )
        return RetrievalPipelineRuntime(deps)

    def _scope_deps(self) -> ScopeDeps:
        return self._runtime().scope_deps()

    def _memory_prefilter_deps(self) -> MemoryPrefilterDeps:
        return self._runtime().memory_prefilter_deps()

    def _sync_profile_cache(self, profile: dict[str, Any] | None) -> dict[str, Any] | None:
        return self._runtime().sync_profile_cache(profile)

    def _sync_topology_cache(self, topology: dict[str, Any] | None) -> dict[str, Any] | None:
        return self._runtime().sync_topology_cache(topology)

    def _get_active_profile_direct(self) -> dict[str, Any] | None:
        return self._runtime().get_active_profile_direct()

    def _load_topology_index_direct(self) -> dict[str, Any] | None:
        return self._runtime().load_topology_index_direct()

    def _merge_prefilter_results(self, results: list[SearchResult], *, top_k: int) -> list[SearchResult]:
        return self._runtime().merge_prefilter_results(results, top_k=top_k)

    def _search_with_paper_memory_prefilter_direct(
        self,
        *,
        query: str,
        top_k: int,
        min_score: float,
        source_type: str | None,
        retrieval_mode: str,
        alpha: float,
        requested_mode: str,
        metadata_filter: dict[str, Any] | None = None,
        search_fn=None,
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        return self._runtime().search_with_paper_memory_prefilter_direct(
            query=query,
            top_k=top_k,
            min_score=min_score,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            requested_mode=requested_mode,
            metadata_filter=metadata_filter,
            search_fn=search_fn,
        )

    def _search_with_memory_prefilter_direct(
        self,
        *,
        query: str,
        top_k: int,
        min_score: float,
        source_type: str | None,
        retrieval_mode: str,
        alpha: float,
        requested_mode: str,
        metadata_filter: dict[str, Any] | None = None,
        query_forms: list[str] | None = None,
        search_fn=None,
    ) -> MemoryPrefilterExecution:
        return self._runtime().search_with_memory_prefilter_direct(
            query=query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            min_score=min_score,
            requested_mode=requested_mode,
            metadata_filter=metadata_filter,
            query_forms=query_forms,
            search_fn=search_fn,
        )

    def _method_overridden(self, name: str) -> bool:
        bound = getattr(self.searcher, name, None)
        class_attr = getattr(type(self.searcher), name, None)
        func = getattr(bound, "__func__", None)
        if callable(bound) and func is None:
            return True
        return callable(bound) and func is not None and func is not class_attr

    def _reranker_config(self) -> RerankerConfig:
        return RerankerConfig.from_config(getattr(self.searcher, "config", None))

    def _get_reranker(self, config: RerankerConfig):
        if not config.enabled:
            return None
        cache = self._reranker_cache
        if cache and cache[0] == config.model:
            return cache[1]
        reranker = build_reranker(config)
        self._reranker_cache = (config.model, reranker)
        return reranker

    def _apply_cross_encoder_reranking(
        self,
        results: list[SearchResult],
        *,
        query: str,
        top_k: int,
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        return self._runtime().apply_cross_encoder_reranking(
            results,
            query=query,
            top_k=top_k,
        )

    def build_plan(
        self,
        *,
        query: str,
        top_k: int,
        source_type: str | None,
        retrieval_mode: str,
        memory_route_mode: str,
        use_ontology_expansion: bool,
        metadata_filter: dict[str, Any] | None = None,
        query_plan: dict[str, Any] | None = None,
        query_frame: dict[str, Any] | None = None,
    ) -> RetrievalPlan:
        return self._plan_builder().build_plan(
            query=query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            memory_route_mode=memory_route_mode,
            use_ontology_expansion=use_ontology_expansion,
            metadata_filter=metadata_filter,
            query_plan=query_plan,
            query_frame=query_frame,
        )

    def _graph_candidate_boost(
        self,
        results: list[SearchResult],
        *,
        graph_query_signal: dict[str, Any] | None,
    ) -> list[SearchResult]:
        return self._search_core().graph_candidate_boost(
            results,
            graph_query_signal=graph_query_signal,
        )

    def _fetch_compare_scoped_result(
        self,
        *,
        query_text: str,
        paper_id: str,
        lexical_query_forms: list[str],
        filter_dict: dict[str, Any] | None,
        top_k: int,
    ) -> SearchResult | None:
        return self._paper_coverage_service().fetch_compare_scoped_result(
            query_text=query_text,
            paper_id=paper_id,
            lexical_query_forms=lexical_query_forms,
            filter_dict=filter_dict,
            top_k=top_k,
        )

    def _fetch_paper_scoped_result(
        self,
        *,
        query_text: str,
        paper_id: str,
        lexical_query_forms: list[str],
        filter_dict: dict[str, Any] | None,
        top_k: int,
        fallback_reason: str,
    ) -> SearchResult | None:
        return self._paper_coverage_service().fetch_paper_scoped_result(
            query_text=query_text,
            paper_id=paper_id,
            lexical_query_forms=lexical_query_forms,
            filter_dict=filter_dict,
            top_k=top_k,
            fallback_reason=fallback_reason,
        )

    def _build_compare_card_fallback_result(self, *, paper_id: str) -> SearchResult | None:
        return self._paper_coverage_service().build_compare_card_fallback_result(paper_id=paper_id)

    def _build_paper_card_fallback_result(
        self,
        *,
        paper_id: str,
        reason: str,
        score: float = 0.62,
        card_row: dict[str, Any] | None = None,
    ) -> SearchResult | None:
        return self._paper_coverage_service().build_paper_card_fallback_result(
            paper_id=paper_id,
            reason=reason,
            score=score,
            card_row=card_row,
        )

    def _search_paper_card_fallback_results(
        self,
        *,
        query_forms: list[str],
        limit: int,
        reason: str,
    ) -> list[SearchResult]:
        return self._paper_coverage_service().search_paper_card_fallback_results(
            query_forms=query_forms,
            limit=limit,
            reason=reason,
        )

    def _ensure_resolved_paper_result_coverage(
        self,
        *,
        results: list[SearchResult],
        query_text: str,
        filter_dict: dict[str, Any] | None,
        lexical_query_forms: list[str],
        resolved_source_ids: list[str],
        top_k: int,
        normalized_source: str,
        paper_family: str,
    ) -> list[SearchResult]:
        return self._paper_coverage_service().ensure_resolved_paper_result_coverage(
            results=results,
            query_text=query_text,
            filter_dict=filter_dict,
            lexical_query_forms=lexical_query_forms,
            resolved_source_ids=resolved_source_ids,
            top_k=top_k,
            normalized_source=normalized_source,
            paper_family=paper_family,
        )

    def _ensure_discover_result_coverage(
        self,
        *,
        results: list[SearchResult],
        lexical_query_forms: list[str],
        top_k: int,
        normalized_source: str,
        paper_family: str,
    ) -> list[SearchResult]:
        return self._paper_coverage_service().ensure_discover_result_coverage(
            results=results,
            lexical_query_forms=lexical_query_forms,
            top_k=top_k,
            normalized_source=normalized_source,
            paper_family=paper_family,
        )

    def _ensure_paper_result_coverage(
        self,
        *,
        results: list[SearchResult],
        query_text: str,
        filter_dict: dict[str, Any] | None,
        lexical_query_forms: list[str],
        resolved_source_ids: list[str],
        top_k: int,
        normalized_source: str,
        paper_family: str,
    ) -> list[SearchResult]:
        return self._paper_coverage_service().ensure_paper_result_coverage(
            results=results,
            query_text=query_text,
            filter_dict=filter_dict,
            lexical_query_forms=lexical_query_forms,
            resolved_source_ids=resolved_source_ids,
            top_k=top_k,
            normalized_source=normalized_source,
            paper_family=paper_family,
        )

    def _ensure_compare_result_coverage(
        self,
        *,
        results: list[SearchResult],
        query_text: str,
        filter_dict: dict[str, Any] | None,
        lexical_query_forms: list[str],
        resolved_source_ids: list[str],
        top_k: int,
        normalized_source: str,
        paper_family: str,
    ) -> list[SearchResult]:
        return self._paper_coverage_service().ensure_compare_result_coverage(
            results=results,
            query_text=query_text,
            filter_dict=filter_dict,
            lexical_query_forms=lexical_query_forms,
            resolved_source_ids=resolved_source_ids,
            top_k=top_k,
            normalized_source=normalized_source,
            paper_family=paper_family,
        )

    def _apply_cluster_context_expansion(
        self,
        results: list[SearchResult],
        *,
        top_k: int,
    ) -> tuple[list[SearchResult], list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
        return self._runtime().apply_cluster_context_expansion(
            results,
            top_k=top_k,
        )

    def _run_base_search(
        self,
        *,
        query: str,
        top_k: int,
        source_type: str | None,
        retrieval_mode: str,
        alpha: float,
        semantic_top_k: Optional[int] = None,
        lexical_top_k: Optional[int] = None,
        use_ontology_expansion: bool = True,
        metadata_filter: Optional[dict[str, Any]] = None,
        plan: RetrievalPlan | None = None,
    ) -> tuple[list[SearchResult], dict[str, Any], list[dict[str, Any]]]:
        return self._search_core().run_base_search(
            query=query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            semantic_top_k=semantic_top_k,
            lexical_top_k=lexical_top_k,
            use_ontology_expansion=use_ontology_expansion,
            metadata_filter=metadata_filter,
            plan=plan,
        )

    def _memory_prior_merge(
        self,
        *,
        base_results: list[SearchResult],
        memory_results: list[SearchResult],
        memory_prefilter: dict[str, Any],
        top_k: int,
        plan: RetrievalPlan,
    ) -> list[SearchResult]:
        return self._runtime().memory_prior_merge(
            base_results=base_results,
            memory_results=memory_results,
            memory_prefilter=memory_prefilter,
            top_k=top_k,
            plan=plan,
        )

    def _enforce_source_scope(
        self,
        results: list[SearchResult],
        *,
        source_type: str | None,
    ) -> tuple[list[SearchResult], bool]:
        return self._runtime().enforce_source_scope(
            results,
            source_type=source_type,
        )

    def execute(
        self,
        *,
        query: str,
        top_k: int = 5,
        source_type: str | None = None,
        retrieval_mode: str = "hybrid",
        alpha: float = 0.7,
        semantic_top_k: Optional[int] = None,
        lexical_top_k: Optional[int] = None,
        use_ontology_expansion: bool = True,
        metadata_filter: Optional[dict[str, Any]] = None,
        memory_route_mode: str = MEMORY_ROUTE_MODE_OFF,
        paper_memory_mode: str = PAPER_MEMORY_MODE_OFF,
        min_score: float = 0.0,
        query_plan: dict[str, Any] | None = None,
        query_frame: dict[str, Any] | None = None,
    ) -> RetrievalPipelineResult:
        normalized_memory_route_mode = normalize_memory_route_mode(
            memory_route_mode,
            paper_memory_mode=paper_memory_mode,
        )
        requested_memory_route_mode, _, memory_mode_alias_applied = normalize_memory_route_mode_details(
            memory_route_mode,
            paper_memory_mode=paper_memory_mode,
        )
        plan = self.build_plan(
            query=query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            memory_route_mode=normalized_memory_route_mode,
            use_ontology_expansion=use_ontology_expansion,
            metadata_filter=metadata_filter,
            query_plan=query_plan,
            query_frame=query_frame,
        )
        base_results, rerank_signals, candidate_sources = self._run_base_search(
            query=query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            semantic_top_k=semantic_top_k,
            lexical_top_k=lexical_top_k,
            use_ontology_expansion=bool(plan.ontology_expansion_enabled),
            metadata_filter=plan.metadata_filter_applied,
            plan=plan,
        )
        normalized_source = str(normalize_source_type(source_type) or "")
        paper_override_active = normalized_source == "paper" and self._method_overridden("_search_with_paper_memory_prefilter")
        requested_paper_mode, normalized_paper_mode, paper_mode_alias_applied = normalize_paper_memory_mode_details(
            paper_memory_mode if paper_memory_mode else normalized_memory_route_mode
        )

        if normalized_memory_route_mode == MEMORY_ROUTE_MODE_OFF and not paper_override_active:
            memory_prefilter = {
                "requestedMode": requested_memory_route_mode,
                "effectiveMode": normalized_memory_route_mode,
                "modeAliasApplied": memory_mode_alias_applied,
                "applied": False,
                "fallbackUsed": False,
                "matchedMemoryIds": [],
                "matchedDocumentIds": [],
                "memoryRelationsUsed": [],
                "temporalSignals": dict(plan.temporal_signals),
                "temporalRouteApplied": bool(plan.temporal_route_applied),
                "updatesPreferred": False,
                "formsTried": [],
                "reason": "disabled",
                "memoryInfluenceApplied": False,
                "verificationCouplingApplied": False,
                "fallbackReason": "",
            }
            merged_results = list(base_results)
        else:
            compatibility_results: list[SearchResult] | None = None
            compatibility_diag: dict[str, Any] | None = None
            direct_search_fn = lambda *args, **kwargs: self._run_base_search(
                query=kwargs.get("query") or (args[0] if args else query),
                top_k=int(kwargs.get("top_k", top_k)),
                source_type=kwargs.get("source_type", source_type),
                retrieval_mode=kwargs.get("retrieval_mode", retrieval_mode),
                alpha=_safe_float(kwargs.get("alpha", alpha), alpha),
                semantic_top_k=semantic_top_k,
                lexical_top_k=lexical_top_k,
                use_ontology_expansion=bool(plan.ontology_expansion_enabled),
                metadata_filter=kwargs.get("metadata_filter"),
                plan=plan,
            )[0]
            if paper_override_active:
                compatibility_results, compatibility_diag = self.searcher._search_with_paper_memory_prefilter(
                    query=query,
                    top_k=top_k,
                    min_score=min_score,
                    source_type=source_type,
                    retrieval_mode=retrieval_mode,
                    alpha=alpha,
                    requested_mode=normalize_paper_memory_mode(
                        paper_memory_mode if paper_memory_mode else normalized_memory_route_mode
                    ),
                    metadata_filter=plan.metadata_filter_applied,
                )
                compatibility_diag = dict(compatibility_diag or {})
                memory_prefilter = {
                    "requestedMode": requested_memory_route_mode,
                    "effectiveMode": normalized_memory_route_mode,
                    "modeAliasApplied": memory_mode_alias_applied,
                    "applied": bool(compatibility_diag.get("applied")),
                    "fallbackUsed": bool(compatibility_diag.get("fallbackUsed")),
                    "matchedMemoryIds": list(compatibility_diag.get("matchedMemoryIds") or []),
                    "matchedDocumentIds": list(compatibility_diag.get("matchedPaperIds") or []),
                    "memoryRelationsUsed": list(compatibility_diag.get("memoryRelationsUsed") or []),
                    "temporalSignals": dict(compatibility_diag.get("temporalSignals") or plan.temporal_signals),
                    "temporalRouteApplied": bool(compatibility_diag.get("temporalRouteApplied", plan.temporal_route_applied)),
                    "updatesPreferred": bool(compatibility_diag.get("updatesPreferred")),
                    "formsTried": ["paper_memory", "document_memory", "chunk"],
                    "reason": str(compatibility_diag.get("reason") or ""),
                    "memoryInfluenceApplied": normalized_memory_route_mode != MEMORY_ROUTE_MODE_OFF and bool(compatibility_diag.get("applied")),
                    "verificationCouplingApplied": False,
                    "fallbackReason": _memory_mode_fallback_reason(
                        mode=normalized_memory_route_mode,
                        reason=str(compatibility_diag.get("reason") or ""),
                    ),
                }
            elif normalized_source == "paper":
                compatibility_results, compatibility_diag = self._search_with_paper_memory_prefilter_direct(
                    query=query,
                    top_k=top_k,
                    min_score=min_score,
                    source_type=source_type,
                    retrieval_mode=retrieval_mode,
                    alpha=alpha,
                    requested_mode=normalize_paper_memory_mode(
                        paper_memory_mode if paper_memory_mode else normalized_memory_route_mode
                    ),
                    metadata_filter=plan.metadata_filter_applied,
                    search_fn=direct_search_fn,
                )
                compatibility_diag = dict(compatibility_diag or {})
                memory_prefilter = {
                    "requestedMode": requested_memory_route_mode,
                    "effectiveMode": normalized_memory_route_mode,
                    "modeAliasApplied": memory_mode_alias_applied,
                    "applied": bool(compatibility_diag.get("applied")),
                    "fallbackUsed": bool(compatibility_diag.get("fallbackUsed")),
                    "matchedMemoryIds": list(compatibility_diag.get("matchedMemoryIds") or []),
                    "matchedDocumentIds": list(compatibility_diag.get("matchedPaperIds") or []),
                    "memoryRelationsUsed": list(compatibility_diag.get("memoryRelationsUsed") or []),
                    "temporalSignals": dict(compatibility_diag.get("temporalSignals") or plan.temporal_signals),
                    "temporalRouteApplied": bool(compatibility_diag.get("temporalRouteApplied", plan.temporal_route_applied)),
                    "updatesPreferred": bool(compatibility_diag.get("updatesPreferred")),
                    "formsTried": ["paper_memory", "document_memory", "chunk"],
                    "reason": str(compatibility_diag.get("reason") or ""),
                    "memoryInfluenceApplied": normalized_memory_route_mode != MEMORY_ROUTE_MODE_OFF and bool(compatibility_diag.get("applied")),
                    "verificationCouplingApplied": False,
                    "fallbackReason": _memory_mode_fallback_reason(
                        mode=normalized_memory_route_mode,
                        reason=str(compatibility_diag.get("reason") or ""),
                    ),
                }
            elif normalized_source != "paper" and self._method_overridden("_search_with_memory_prefilter"):
                compatibility_results, compatibility_diag = self.searcher._search_with_memory_prefilter(
                    query=query,
                    top_k=top_k,
                    min_score=min_score,
                    source_type=source_type,
                    retrieval_mode=retrieval_mode,
                    alpha=alpha,
                    requested_mode=normalized_memory_route_mode,
                    metadata_filter=plan.metadata_filter_applied,
                )
                memory_prefilter = dict(compatibility_diag or {})
                memory_prefilter.setdefault("requestedMode", requested_memory_route_mode)
                memory_prefilter.setdefault("effectiveMode", normalized_memory_route_mode)
                memory_prefilter.setdefault("modeAliasApplied", memory_mode_alias_applied)
                memory_prefilter.setdefault("temporalSignals", dict(plan.temporal_signals))
                memory_prefilter.setdefault("temporalRouteApplied", bool(plan.temporal_route_applied))
                memory_prefilter.setdefault("updatesPreferred", False)
                memory_prefilter["memoryInfluenceApplied"] = normalized_memory_route_mode != MEMORY_ROUTE_MODE_OFF and bool(
                    memory_prefilter.get("applied")
                )
                memory_prefilter["verificationCouplingApplied"] = False
                memory_prefilter["fallbackReason"] = _memory_mode_fallback_reason(
                    mode=normalized_memory_route_mode,
                    reason=str(memory_prefilter.get("reason") or ""),
                )
            else:
                execution = self._search_with_memory_prefilter_direct(
                    query=query,
                    top_k=top_k,
                    min_score=min_score,
                    source_type=source_type,
                    retrieval_mode=retrieval_mode,
                    alpha=alpha,
                    requested_mode=normalized_memory_route_mode,
                    metadata_filter=plan.metadata_filter_applied,
                    query_forms=list(dict.fromkeys([*list(plan.query_frame.get("expanded_terms") or []), query]))[:6],
                    search_fn=direct_search_fn,
                )
                compatibility_results = list(execution.results or [])
                memory_prefilter = dict(execution.diagnostics or {})
                memory_prefilter["memoryInfluenceApplied"] = normalized_memory_route_mode != MEMORY_ROUTE_MODE_OFF and bool(
                    memory_prefilter.get("applied")
                )
                memory_prefilter["verificationCouplingApplied"] = False
                memory_prefilter["fallbackReason"] = _memory_mode_fallback_reason(
                    mode=normalized_memory_route_mode,
                    reason=str(memory_prefilter.get("reason") or ""),
                )
            merged_results = self._memory_prior_merge(
                base_results=base_results,
                memory_results=list(compatibility_results or []),
                memory_prefilter=memory_prefilter,
                top_k=top_k,
                plan=plan,
            )

        merged_results = self._ensure_paper_result_coverage(
            results=merged_results,
            query_text=query,
            filter_dict=plan.metadata_filter_applied,
            lexical_query_forms=list(rerank_signals.get("lexicalQueryForms") or []),
            resolved_source_ids=list(plan.query_frame.get("resolved_source_ids") or []),
            top_k=top_k,
            normalized_source=normalized_source,
            paper_family=plan.paper_family,
        )

        merged_results = [item for item in merged_results if float(getattr(item, "score", 0.0) or 0.0) >= min_score]
        merged_results, source_scope_enforced = self._enforce_source_scope(merged_results, source_type=source_type)
        merged_results, cross_encoder_diagnostics = self._apply_cross_encoder_reranking(
            merged_results,
            query=query,
            top_k=top_k,
        )
        merged_results = self._ensure_paper_result_coverage(
            results=merged_results,
            query_text=query,
            filter_dict=plan.metadata_filter_applied,
            lexical_query_forms=list(rerank_signals.get("lexicalQueryForms") or []),
            resolved_source_ids=list(plan.query_frame.get("resolved_source_ids") or []),
            top_k=top_k,
            normalized_source=normalized_source,
            paper_family=plan.paper_family,
        )
        ontology_used = bool(plan.ontology_expansion_enabled and getattr(self.searcher, "sqlite_db", None))
        if bool(plan.cluster_assist_eligible):
            scoped_results, related_clusters, cluster_expansion, active_profile = self._apply_cluster_context_expansion(
                merged_results,
                top_k=top_k,
            )
            context_expansion = _build_enrichment_diagnostics(
                plan,
                ontology_used=ontology_used,
                cluster_used=bool(related_clusters),
                preferred_sources=list(cluster_expansion.get("preferredSources") or []),
                topology_applied=bool(cluster_expansion.get("topologyApplied")),
                cluster_ids=list(cluster_expansion.get("clusterIds") or []),
                cluster_count=int(cluster_expansion.get("clusterCount") or 0),
            )
        else:
            merged_results.sort(key=_retrieval_sort_key, reverse=True)
            scoped_results = _preserve_parent_diversity(merged_results)[:top_k]
            related_clusters = []
            active_profile = None
            context_expansion = _build_enrichment_diagnostics(
                plan,
                ontology_used=ontology_used,
                cluster_used=False,
            )
        mixed_fallback_used = bool(memory_prefilter.get("mixedFallbackUsed"))
        retrieval_strategy = build_retrieval_strategy_diagnostics(plan)
        retrieval_quality = build_retrieval_quality_diagnostics(
            plan=plan,
            results=scoped_results,
            candidate_sources=candidate_sources,
            rerank_signals=rerank_signals,
            memory_prefilter=memory_prefilter,
        )
        answerability_rerank = build_answerability_rerank_diagnostics(
            plan=plan,
            results=scoped_results,
        )
        artifact_health = build_artifact_health_diagnostics(
            plan=plan,
            results=scoped_results,
        )
        corrective_retrieval = build_corrective_retrieval_diagnostics(
            retrieval_quality=retrieval_quality,
            answerability_rerank=answerability_rerank,
            artifact_health=artifact_health,
        )

        for item in scoped_results:
            extras = dict(item.lexical_extras or {})
            extras["retrieval_plan"] = plan.to_dict()
            extras["candidate_sources"] = list(candidate_sources)
            extras["context_expansion"] = dict(context_expansion)
            extras["memory_prefilter"] = dict(memory_prefilter)
            extras["source_scope_enforced"] = bool(source_scope_enforced)
            extras["mixed_fallback_used"] = bool(mixed_fallback_used)
            extras["retrieval_strategy"] = dict(retrieval_strategy)
            extras["retrieval_quality"] = dict(retrieval_quality)
            extras["answerability_rerank"] = dict(answerability_rerank)
            extras["corrective_retrieval"] = dict(corrective_retrieval)
            extras["artifact_health"] = dict(artifact_health)
            item.lexical_extras = extras

        memory_route = memory_route_payload(
            requested_mode=memory_route_mode,
            source_type=source_type,
            paper_memory_mode=paper_memory_mode,
        )
        memory_route.update(
            {
                "applied": bool(memory_prefilter.get("applied")),
                "matchedForms": list(memory_prefilter.get("formsTried") or []),
                "memoryInfluenceApplied": bool(memory_prefilter.get("memoryInfluenceApplied")),
                "verificationCouplingApplied": False,
                "fallbackReason": str(memory_prefilter.get("fallbackReason") or ""),
            }
        )
        if normalized_source == "paper":
            paper_memory_prefilter = {
                "requestedMode": requested_paper_mode,
                "effectiveMode": normalized_paper_mode,
                "modeAliasApplied": paper_mode_alias_applied,
                "applied": bool(memory_prefilter.get("applied")),
                "fallbackUsed": bool(memory_prefilter.get("fallbackUsed")),
                "matchedPaperIds": list(memory_prefilter.get("matchedDocumentIds") or []),
                "matchedMemoryIds": list(memory_prefilter.get("matchedMemoryIds") or []),
                "memoryRelationsUsed": list(memory_prefilter.get("memoryRelationsUsed") or []),
                "temporalSignals": dict(memory_prefilter.get("temporalSignals") or {}),
                "temporalRouteApplied": bool(memory_prefilter.get("temporalRouteApplied")),
                "updatesPreferred": bool(memory_prefilter.get("updatesPreferred")),
                "reason": str(memory_prefilter.get("reason") or ""),
                "memoryInfluenceApplied": bool(memory_prefilter.get("memoryInfluenceApplied")),
                "verificationCouplingApplied": False,
                "fallbackReason": str(memory_prefilter.get("fallbackReason") or ""),
            }
        else:
            raw_paper_mode = str(paper_memory_mode or PAPER_MEMORY_MODE_OFF).strip().lower() or PAPER_MEMORY_MODE_OFF
            paper_memory_prefilter = {
                "requestedMode": raw_paper_mode,
                "effectiveMode": normalize_paper_memory_mode(paper_memory_mode),
                "modeAliasApplied": normalize_paper_memory_mode(paper_memory_mode) != raw_paper_mode,
                "applied": False,
                "fallbackUsed": bool(memory_prefilter.get("fallbackUsed")),
                "matchedPaperIds": [],
                "matchedMemoryIds": [],
                "memoryRelationsUsed": [],
                "temporalSignals": dict(memory_prefilter.get("temporalSignals") or {}),
                "temporalRouteApplied": bool(memory_prefilter.get("temporalRouteApplied")),
                "updatesPreferred": False,
                "memoryInfluenceApplied": False,
                "verificationCouplingApplied": False,
                "fallbackReason": str(memory_prefilter.get("fallbackReason") or ""),
                "reason": (
                    "source_not_paper"
                    if normalize_paper_memory_mode(paper_memory_mode) in {PAPER_MEMORY_MODE_COMPAT, PAPER_MEMORY_MODE_ON}
                    else str(memory_prefilter.get("reason") or ("disabled" if normalized_memory_route_mode == "off" else "source_not_paper"))
                ),
            }

        memory_prefilter.setdefault("contractRole", "retrieval_memory_prefilter")
        memory_prefilter.setdefault(
            "aliasDeprecated",
            bool(memory_prefilter.get("modeAliasApplied") and memory_prefilter.get("requestedMode") == "prefilter"),
        )
        paper_memory_prefilter.setdefault("contractRole", "paper_source_memory_prefilter")
        paper_memory_prefilter.setdefault(
            "aliasDeprecated",
            bool(paper_memory_prefilter.get("modeAliasApplied") and paper_memory_prefilter.get("requestedMode") == "prefilter"),
        )

        rerank_signals = dict(rerank_signals or {})
        rerank_signals.update(
            {
                "rerankerApplied": bool(cross_encoder_diagnostics.get("rerankerApplied")),
                "rerankerModel": str(cross_encoder_diagnostics.get("rerankerModel") or ""),
                "rerankerWindow": int(cross_encoder_diagnostics.get("rerankerWindow") or 0),
                "rerankerLatencyMs": int(cross_encoder_diagnostics.get("rerankerLatencyMs") or 0),
                "rerankerFallbackUsed": bool(cross_encoder_diagnostics.get("rerankerFallbackUsed")),
                "rerankerReason": str(cross_encoder_diagnostics.get("rerankerReason") or ""),
            }
        )

        return RetrievalPipelineResult(
            results=scoped_results,
            plan=plan,
            candidate_sources=candidate_sources,
            memory_route=memory_route,
            memory_prefilter=memory_prefilter,
            paper_memory_prefilter=paper_memory_prefilter,
            rerank_signals=rerank_signals,
            context_expansion=context_expansion,
            related_clusters=related_clusters,
            active_profile=active_profile,
            source_scope_enforced=source_scope_enforced,
            mixed_fallback_used=mixed_fallback_used,
            retrieval_strategy=retrieval_strategy,
            retrieval_quality=retrieval_quality,
            answerability_rerank=answerability_rerank,
            corrective_retrieval=corrective_retrieval,
            artifact_health=artifact_health,
        )


__all__ = [
    "RetrievalPipelineResult",
    "RetrievalPipelineService",
    "RetrievalPlan",
]
