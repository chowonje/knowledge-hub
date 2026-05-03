from __future__ import annotations

import json
import re
from typing import Any, Callable

from knowledge_hub.ai.rag_support import extract_json_payload, truncate_text
from knowledge_hub.core.models import SearchResult
from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.document_memory.retriever import DocumentMemoryRetriever
from knowledge_hub.infrastructure.providers import get_llm
from knowledge_hub.learning.task_router import TaskRouteDecision, get_llm_for_task
from knowledge_hub.papers.memory_retriever import PaperMemoryRetriever
from knowledge_hub.providers.registry import get_provider_info


TOPIC_SYNTHESIS_SCHEMA = "knowledge-hub.paper-topic-synthesis.result.v1"
_LOCAL_PROVIDER_NAMES = {"ollama", "pplx-local", "pplx-st", "pplx_st"}
_GENERIC_TOPIC_QUERY_TOKENS = {
    "paper",
    "papers",
    "논문",
    "논문들",
    "찾아",
    "찾아줘",
    "정리",
    "정리해줘",
    "관련",
    "대표",
    "차세대",
    "대체",
}
_QUERY_ALIAS_MAP = {
    "트랜스포머": ["transformer", "attention"],
    "어텐션": ["attention"],
    "아키텍처": ["architecture", "model"],
    "차세대": ["next generation", "scalable"],
    "대체": ["alternative", "replacement"],
    "메모리": ["memory"],
    "에이전트": ["agent", "agents"],
    "검색": ["retrieval", "search"],
    "안전": ["safety", "evaluation"],
    "멀티모달": ["multimodal"],
}
_ROLE_MARKERS = {
    "survey": ("survey", "taxonomy", "sok", "review", "overview", "literature review", "taxonom"),
    "benchmark": ("benchmark", "bench", "evaluation", "eval", "leaderboard", "dataset"),
    "architecture": (
        "architecture",
        "replacement",
        "alternative",
        "state space",
        "ssm",
        "mamba",
        "retnet",
        "retention",
        "fast weights",
        "memory architecture",
        "sequence modeling",
    ),
    "system": ("system", "framework", "runtime", "pipeline", "orchestration", "agentic rag", "rag architecture"),
    "application": ("application", "deployment", "discovery", "robotics", "surgery", "omics", "polymer"),
}
_TOPIC_KIND_PROFILES = {
    "transformer_alternative": {
        "preferred": ("alternative", "replacement", "state space", "ssm", "mamba", "retention", "retnet", "fast weights"),
        "discouraged": ("bert", "switch transformer", "with transformers", "diffusion models with transformers", "decision transformer"),
        "must_avoid": ("attention is all you need",),
        "required_roles": ("architecture",),
    },
    "state_space_model": {
        "preferred": ("state space", "ssm", "mamba", "selective state", "linear recurrent"),
        "discouraged": ("bert", "decision transformer", "diffusion", "video", "embedding", "introspective"),
        "must_avoid": ("transformer reinforcement learning",),
        "required_roles": ("architecture",),
    },
    "agentic_rag": {
        "preferred": ("agentic rag", "retrieval architecture", "retrieval-augmented", "dynamic rag", "rag architecture"),
        "discouraged": ("knowledge-intensive nlp", "multimodal retrieval", "embedding", "sandbox"),
        "must_avoid": (),
        "required_roles": ("system", "architecture"),
    },
    "ai_scientist_eval": {
        "preferred": ("scientific discovery", "ai scientist", "benchmark", "survey", "omics", "auto-bench"),
        "discouraged": ("foundation model", "compound ai architecture"),
        "must_avoid": (),
        "required_roles": ("benchmark", "survey"),
    },
    "safety_eval": {
        "preferred": ("safety", "guardrail", "risk", "taxonomy", "benchmark", "dataset", "evaluation"),
        "discouraged": ("rag taxonomy", "industry applications", "memory evaluation", "swe-bench"),
        "must_avoid": (),
        "required_roles": ("benchmark", "survey"),
    },
    "multimodal_reasoning": {
        "preferred": ("multimodal", "audio", "video", "visual", "world learner", "self-supervised"),
        "discouraged": ("bert", "reward model", "webarbiter", "agents"),
        "must_avoid": (),
        "required_roles": ("architecture",),
    },
    "long_term_memory_application": {
        "preferred": ("long-term memory", "long horizon memory", "agentic applications", "application", "memory architecture"),
        "discouraged": ("benchmarking", "benchmark", "evaluation framework"),
        "must_avoid": (),
        "required_roles": ("application", "benchmark"),
    },
    "embodied_memory": {
        "preferred": ("embodied", "video", "long-term memory", "exploration", "world"),
        "discouraged": ("chat assistants", "llm agents", "memoryarena"),
        "must_avoid": (),
        "required_roles": ("benchmark", "architecture"),
    },
    "world_model": {
        "preferred": ("world model", "world learner", "world generation", "sequential", "prediction", "situated awareness"),
        "discouraged": ("prompt injection", "benchmarking", "agent security"),
        "must_avoid": (),
        "required_roles": ("architecture",),
    },
    "paper_to_agent": {
        "preferred": ("paper2agent", "research papers as interactive", "paper-to-agent"),
        "discouraged": ("general agents", "agent benchmark", "conceptual taxonomy"),
        "must_avoid": (),
        "required_roles": ("system", "survey"),
    },
}
_ROLE_MARKERS: dict[str, tuple[str, ...]] = {
    "survey": ("survey", "sok", "taxonomy", "review", "overview", "literature review", "systematic review"),
    "benchmark": ("benchmark", "bench", "evaluation framework", "eval", "leaderboard", "dataset"),
    "architecture": ("architecture", "alternative", "replacement", "state space", "ssm", "mamba", "retention", "retnet", "fast weights"),
    "system": ("system", "framework", "runtime", "platform", "agentic", "orchestration", "pipeline"),
    "application": ("application", "applications", "discovery", "robotics", "surgery", "omics", "polymer"),
}
_TOPIC_PROFILE_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "name": "transformer_alternatives",
        "query_markers": ("트랜스포머", "transformer"),
        "extra_markers": ("대체", "alternative", "replacement", "차세대"),
        "preferred_roles": ("architecture",),
        "required_any": ("state space", "ssm", "mamba", "retention", "retnet", "rwkv", "fast weights"),
        "positive_markers": ("state space", "ssm", "mamba", "retention", "retnet", "rwkv", "fast weights"),
        "negative_markers": (
            "switch transformer",
            "decision transformer",
            "bert",
            "gpt",
            "diffusion models with transformers",
            "sparse attention",
            "prefill",
            "attention is all you need",
        ),
        "must_avoid": ("switch transformer", "decision transformer", "bert", "diffusion models with transformers"),
    },
    {
        "name": "state_space_models_as_transformer_alternatives",
        "query_markers": ("state space", "ssm", "mamba", "상태공간"),
        "extra_markers": ("transformer", "대안", "대체", "비교"),
        "preferred_roles": ("architecture",),
        "required_roles": ("architecture",),
        "disfavored_roles": ("benchmark", "survey", "system", "application"),
        "required_any": ("state space", "ssm", "mamba", "retentive", "retention", "retnet", "rwkv", "hyena"),
        "positive_markers": ("state space", "ssm", "mamba", "retentive", "retention", "retnet", "rwkv", "hyena"),
        "negative_markers": (
            "switch transformer",
            "chain of thought",
            "direct preference optimization",
            "dpo",
            "generic agent",
            "multimodal ai agents",
            "word2vec",
            "benchmark",
        ),
        "must_avoid": ("switch transformer", "chain of thought", "dpo", "direct preference optimization", "bert"),
        "min_adjusted_score": 2.2,
        "judge_window_size": 4,
        "negative_margin": 1.0,
    },
    {
        "name": "agentic_rag_architecture",
        "query_markers": ("agentic rag", "agentic", "rag"),
        "extra_markers": ("retrieval architecture", "hierarchical retrieval", "실제로 바꾸", "바꾸는"),
        "preferred_roles": ("architecture", "system"),
        "disfavored_roles": ("survey",),
        "positive_markers": ("agentic rag", "a-rag", "jade", "hierarchical retrieval", "retrieval interface", "retrieval-infused"),
        "negative_markers": ("survey", "taxonomy", "sok", "knowledge-intensive nlp"),
        "must_avoid": ("survey", "taxonomy", "sok"),
    },
    {
        "name": "scientific_discovery_eval",
        "query_markers": ("scientific discovery", "ai scientist", "scientific"),
        "extra_markers": ("benchmark", "survey", "평가", "나눠"),
        "preferred_roles": ("benchmark", "survey"),
        "positive_markers": ("benchmark", "bench", "survey", "evaluation", "scientific discovery", "ai scientist"),
        "negative_markers": ("foundation model", "multimodal foundation model"),
    },
    {
        "name": "safety_eval_framework",
        "query_markers": ("safety", "안전"),
        "extra_markers": ("evaluation", "framework", "taxonomy", "benchmark", "guardrail"),
        "preferred_roles": ("benchmark", "survey"),
        "positive_markers": ("safety", "guardrail", "risk", "taxonomy", "evaluation", "benchmark"),
        "negative_markers": ("industry", "applications in industry", "agentic rag taxonomy"),
    },
    {
        "name": "multimodal_non_transformer_reasoning",
        "query_markers": ("multimodal", "audio-visual", "video", "multimodal understanding"),
        "extra_markers": ("reasoning", "다른", "transformer 변형보다"),
        "preferred_roles": ("architecture", "system"),
        "required_roles": ("architecture",),
        "disfavored_roles": ("benchmark", "survey", "application"),
        "required_any": ("multimodal", "audio", "video", "vision", "visual", "world model", "world learner", "jepa"),
        "positive_markers": ("multimodal", "audio", "video", "vision", "visual", "world model", "world learner", "jepa", "predict", "self-supervised", "native multimodal"),
        "negative_markers": ("chain of thought", "self-consistency", "reward model", "agent benchmark", "security benchmark", "test-time scaling"),
        "must_avoid": ("chain of thought prompting", "self-consistency", "reward model"),
        "min_adjusted_score": 2.0,
        "judge_window_size": 4,
        "negative_margin": 1.0,
    },
    {
        "name": "long_term_memory_application_vs_benchmark",
        "query_markers": ("long-term memory", "long horizon memory", "장기 기억"),
        "extra_markers": ("application", "benchmark", "응용", "평가"),
        "preferred_roles": ("application", "benchmark"),
        "required_roles": ("application", "benchmark"),
        "disfavored_roles": ("survey", "system"),
        "required_any": ("memory", "long-term", "long horizon"),
        "positive_markers": ("memory", "long-term", "long horizon", "benchmark", "application", "agentic applications", "temporal knowledge graph"),
        "negative_markers": ("prompt injection", "security benchmark", "survey", "industry"),
        "must_avoid": ("prompt injection", "security benchmark"),
        "min_adjusted_score": 1.8,
        "judge_window_size": 5,
        "negative_margin": 1.0,
    },
    {
        "name": "world_model_as_transformer_alternative",
        "query_markers": ("world model", "world learner", "alternative sequential", "sequential modeling"),
        "extra_markers": ("transformer 대체", "대체 맥락"),
        "preferred_roles": ("architecture", "system"),
        "required_roles": ("architecture",),
        "disfavored_roles": ("benchmark", "survey", "application"),
        "required_any": ("world model", "world learner", "sequential", "prediction", "planning", "situated awareness"),
        "positive_markers": ("world model", "world learner", "prediction", "planning", "sequential", "situated awareness", "state space"),
        "negative_markers": ("prompt injection", "security benchmark", "benchmark", "scientific discovery"),
        "must_avoid": ("prompt injection", "security benchmark"),
        "min_adjusted_score": 2.0,
        "judge_window_size": 4,
        "negative_margin": 1.0,
    },
    {
        "name": "paper_to_agent_system_vs_framing",
        "query_markers": ("paper-to-agent", "paper2agent", "논문을 ai agent로", "논문을 ai agent로 재구성"),
        "preferred_roles": ("system", "survey"),
        "required_roles": ("system",),
        "disfavored_roles": ("application", "benchmark"),
        "required_any": ("paper2agent", "paper-to-agent", "research papers as interactive", "reimagining research papers"),
        "positive_markers": ("paper2agent", "paper-to-agent", "research papers", "interactive", "reliable ai agents", "conceptual framing"),
        "negative_markers": ("general agent", "security benchmark", "prompt injection", "memory architecture", "agentic retrieval-augmented generation"),
        "must_avoid": ("prompt injection", "security benchmark"),
        "min_adjusted_score": 1.7,
        "judge_window_size": 4,
        "negative_margin": 1.0,
    },
)
_ROLE_KEYWORDS = {
    "benchmark": ("benchmark", "bench", "evaluation", "evaluate", "eval", "dataset", "leaderboard", "framework"),
    "survey": ("survey", "overview", "taxonomy", "sok", "review", "landscape"),
    "system": ("system", "framework", "runtime", "platform", "workflow", "orchestration", "agentic rag", "paper2agent"),
    "architecture": (
        "architecture",
        "model",
        "mechanism",
        "state space",
        "ssm",
        "mamba",
        "retention",
        "retentive",
        "rwkv",
        "fast weights",
        "sequence modeling",
    ),
    "application": ("application", "for ", "robotics", "embodied", "video", "omics", "polymer", "discovery", "assistant"),
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize_title(value: Any) -> str:
    body = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    return re.sub(r"[^0-9a-z가-힣 ]+", "", body).strip()


def _query_variants(query: str) -> list[str]:
    base = _clean_text(query)
    if not base:
        return []
    lowered = base.casefold()
    variants = [base]
    english_expansions: list[str] = []
    for marker, aliases in _QUERY_ALIAS_MAP.items():
        if marker in lowered:
            english_expansions.extend(aliases)
    token_candidates = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]+|[가-힣]{2,}", lowered)
        if token not in _GENERIC_TOPIC_QUERY_TOKENS
    ]
    english_tokens = [token for token in token_candidates if re.search(r"[a-z]", token)]
    if english_tokens:
        variants.append(" ".join(english_tokens[:4]))
    if english_expansions:
        variants.append(" ".join(dict.fromkeys(english_expansions)))
    for token in list(dict.fromkeys(english_expansions + english_tokens))[:6]:
        if token and token not in variants:
            variants.append(token)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in variants:
        normalized = _clean_text(item)
        if not normalized or normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered[:8]


def _provider_is_local(provider: str) -> bool:
    token = str(provider or "").strip().lower()
    if token in _LOCAL_PROVIDER_NAMES:
        return True
    try:
        info = get_provider_info(token)
    except Exception:
        info = None
    return bool(info and info.is_local)


def _contains_any_bool(text: str, markers: tuple[str, ...] | list[str]) -> bool:
    lowered = str(text or "").casefold()
    return any(marker and marker.casefold() in lowered for marker in markers)


def _query_wants_plural(query: str) -> bool:
    lowered = str(query or "").casefold()
    markers = ("논문들", "papers", "find papers", "찾아", "정리", "비교", "구분", "나눠", "representative", "대표")
    return _contains_any_bool(lowered, markers)


def _infer_topic_profile_v2(query: str) -> dict[str, Any]:
    lowered = _clean_text(query).casefold()
    required_roles: list[str] = []
    if _contains_any_bool(lowered, ("benchmark", "bench", "벤치마크", "평가")):
        required_roles.append("benchmark")
    if _contains_any_bool(lowered, ("survey", "taxonomy", "sok", "서베이", "taxonomy", "review")):
        required_roles.append("survey")
    if _contains_any_bool(lowered, ("system", "framework", "runtime", "system paper", "구현")):
        required_roles.append("system")
    if _contains_any_bool(lowered, ("architecture", "대체", "replacement", "alternative", "아키텍처")):
        required_roles.append("architecture")
    if _contains_any_bool(lowered, ("application", "응용", "use case")):
        required_roles.append("application")

    matched_template: dict[str, Any] | None = None
    matched_score: tuple[int, int, int] = (-1, -1, -1)
    for template in _TOPIC_PROFILE_TEMPLATES:
        query_markers = tuple(template.get("query_markers", ()) or ())
        query_hits = sum(1 for marker in query_markers if marker and marker.casefold() in lowered)
        if query_hits <= 0:
            continue
        extra = tuple(template.get("extra_markers", ()) or ())
        extra_hits = sum(1 for marker in extra if marker and marker.casefold() in lowered)
        if extra and extra_hits <= 0:
            continue
        candidate_score = (query_hits, extra_hits, len(query_markers) + len(extra))
        if candidate_score > matched_score:
            matched_template = template
            matched_score = candidate_score

    profile = {
        "name": str((matched_template or {}).get("name") or "generic_topic"),
        "preferredRoles": list((matched_template or {}).get("preferred_roles", ()) or ()),
        "disfavoredRoles": list((matched_template or {}).get("disfavored_roles", ()) or ()),
        "requiredRoles": list(
            dict.fromkeys(
                required_roles
                or list((matched_template or {}).get("required_roles", ()) or ())
                or list((matched_template or {}).get("preferred_roles", ()) or ())
            )
        ),
        "requiredAnyMarkers": list((matched_template or {}).get("required_any", ()) or ()),
        "positiveMarkers": list((matched_template or {}).get("positive_markers", ()) or ()),
        "negativeMarkers": list((matched_template or {}).get("negative_markers", ()) or ()),
        "mustAvoidMarkers": list((matched_template or {}).get("must_avoid", ()) or ()),
        "minAdjustedScore": _safe_float((matched_template or {}).get("min_adjusted_score"), 0.0),
        "judgeWindowSize": max(3, int((matched_template or {}).get("judge_window_size") or 6)),
        "negativeMargin": max(1.0, _safe_float((matched_template or {}).get("negative_margin"), 1.0)),
        "preferPlural": _query_wants_plural(query),
    }
    return profile


def _candidate_text_v2(candidate: dict[str, Any]) -> str:
    return " ".join(
        filter(
            None,
            [
                str(candidate.get("title") or ""),
                str(candidate.get("paperCore") or ""),
                str(candidate.get("methodCore") or ""),
                str(candidate.get("evidenceCore") or ""),
                str(candidate.get("limitations") or ""),
                " ".join(str(item) for item in list(candidate.get("conceptLinks") or [])),
            ],
        )
    ).casefold()


def _classify_candidate_roles_v2(candidate: dict[str, Any]) -> dict[str, Any]:
    text = _candidate_text_v2(candidate)
    scores: dict[str, int] = {}
    for role, markers in _ROLE_MARKERS.items():
        scores[role] = sum(1 for marker in markers if marker in text)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    matched_roles = [role for role, score in ordered if score > 0]
    primary = matched_roles[0] if matched_roles else "adjacent"
    return {
        "primaryRole": primary,
        "matchedRoles": matched_roles,
        "roleScores": scores,
    }


def _score_candidate_for_profile_v2(candidate: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    base = _safe_float(((candidate.get("retrievalSignals") or {}).get("candidateScore")), 0.0)
    role_info = _classify_candidate_roles_v2(candidate)
    text = _candidate_text_v2(candidate)
    positive_hits = [marker for marker in list(profile.get("positiveMarkers") or []) if marker.casefold() in text]
    negative_hits = [marker for marker in list(profile.get("negativeMarkers") or []) if marker.casefold() in text]
    must_avoid_hits = [marker for marker in list(profile.get("mustAvoidMarkers") or []) if marker.casefold() in text]
    required_any = list(profile.get("requiredAnyMarkers") or [])
    required_any_hit = any(marker.casefold() in text for marker in required_any) if required_any else True

    score = base
    score += min(len(positive_hits), 3) * 1.25
    if role_info["primaryRole"] in set(profile.get("preferredRoles") or []):
        score += 1.0
    if role_info["primaryRole"] in set(profile.get("disfavoredRoles") or []):
        score -= 1.0
    if not required_any_hit and required_any:
        score -= 2.5
    score -= min(len(negative_hits), 3) * 1.0
    score -= min(len(must_avoid_hits), 2) * 2.0
    return {
        "primaryRole": role_info["primaryRole"],
        "matchedRoles": list(role_info["matchedRoles"]),
        "positiveHits": positive_hits,
        "negativeHits": negative_hits,
        "mustAvoidHits": must_avoid_hits,
        "requiredAnyHit": required_any_hit,
        "adjustedScore": round(score, 6),
    }


def _candidate_sort_key_v2(candidate: dict[str, Any]) -> tuple[float, float, float, float, str]:
    retrieval = dict(candidate.get("retrievalSignals") or {})
    return (
        -_safe_float(retrieval.get("adjustedScore"), 0.0),
        -_safe_float(retrieval.get("candidateScore"), 0.0),
        -_safe_float(retrieval.get("paperMemoryScore"), 0.0),
        -_safe_float(retrieval.get("documentMemoryScore"), 0.0),
        str(candidate.get("title") or ""),
    )


def _prune_candidates_for_profile_v2(
    candidates: list[dict[str, Any]],
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    preferred_roles = {str(item).strip() for item in list(profile.get("preferredRoles") or []) if str(item).strip()}
    required_roles = {str(item).strip() for item in list(profile.get("requiredRoles") or []) if str(item).strip()}
    disfavored_roles = {str(item).strip() for item in list(profile.get("disfavoredRoles") or []) if str(item).strip()}
    allowed_roles = preferred_roles | required_roles
    min_adjusted_score = _safe_float(profile.get("minAdjustedScore"), 0.0)
    negative_margin = _safe_float(profile.get("negativeMargin"), 1.0)

    kept: list[dict[str, Any]] = []
    pruned: list[dict[str, Any]] = []
    prune_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        enriched = dict(candidate)
        signals = dict(enriched.get("topicSignals") or {})
        primary_role = str(signals.get("primaryRole") or "").strip()
        positive_count = len(list(signals.get("positiveHits") or []))
        negative_count = len(list(signals.get("negativeHits") or []))
        adjusted_score = _safe_float(signals.get("adjustedScore"), 0.0)
        reasons: list[str] = []
        if list(signals.get("mustAvoidHits") or []):
            reasons.append("must_avoid")
        if signals.get("requiredAnyHit") is False:
            reasons.append("missing_required_markers")
        if allowed_roles and primary_role not in allowed_roles and adjusted_score < (min_adjusted_score + 0.75):
            reasons.append("role_outside_allowed")
        if primary_role in disfavored_roles and positive_count == 0:
            reasons.append("disfavored_role_without_support")
        if (negative_count - positive_count) >= negative_margin and adjusted_score < (min_adjusted_score + 1.0):
            reasons.append("negative_signal_outweighs_positive")
        if adjusted_score < min_adjusted_score and (primary_role not in allowed_roles or not signals.get("requiredAnyHit", True)):
            reasons.append("below_min_adjusted_score")

        signals["prunePassed"] = not reasons
        signals["pruneReasons"] = list(dict.fromkeys(reasons))
        enriched["topicSignals"] = signals
        if reasons:
            enriched["selection"] = {
                "verdict": "exclude",
                "rationale": f"pruned before topic judge: {', '.join(list(dict.fromkeys(reasons)))}",
            }
            pruned.append(enriched)
            prune_rows.append(
                {
                    "paperId": str(enriched.get("paperId") or ""),
                    "title": str(enriched.get("title") or ""),
                    "primaryRole": primary_role,
                    "adjustedScore": round(adjusted_score, 6),
                    "reasons": list(dict.fromkeys(reasons)),
                }
            )
        else:
            kept.append(enriched)

    kept.sort(key=_candidate_sort_key_v2)
    pruned.sort(key=lambda item: int(item.get("rank") or 999))
    return kept, pruned, {
        "retrievedCount": len(candidates),
        "keptCount": len(kept),
        "prunedCount": len(pruned),
        "rows": prune_rows[: max(6, len(prune_rows))],
        "fallbackPoolTooSmall": len(kept) < 2,
    }


def _dynamic_selected_limit_v2(candidates: list[dict[str, Any]], requested_limit: int, profile: dict[str, Any]) -> int:
    if not candidates:
        return 0
    min_adjusted_score = _safe_float(profile.get("minAdjustedScore"), 0.0)
    above_threshold = [
        item
        for item in candidates
        if _safe_float(((item.get("topicSignals") or {}).get("adjustedScore")), 0.0) >= min_adjusted_score
    ]
    if len(above_threshold) >= 2:
        return min(max(2, len(above_threshold)), requested_limit)
    if len(candidates) >= 2:
        return min(2, requested_limit)
    return 1


def _derive_topic_labels_v2(candidate: dict[str, Any], topic_profile: dict[str, Any]) -> dict[str, str]:
    signals = dict(candidate.get("topicSignals") or {})
    primary_role = str(signals.get("primaryRole") or "adjacent").strip() or "adjacent"
    positive_hits = list(signals.get("positiveHits") or [])
    negative_hits = list(signals.get("negativeHits") or [])
    text = _candidate_text_v2(candidate)
    profile_name = str(topic_profile.get("name") or "")

    topic_fit = "core" if positive_hits and not negative_hits else "adjacent"
    if negative_hits and not positive_hits:
        topic_fit = "borderline"

    alternative_vs_optimization = "n/a"
    if "transformer" in profile_name or "alternative" in profile_name:
        alternative_vs_optimization = "alternative" if primary_role == "architecture" and positive_hits else "optimization_or_adjacent"

    application_vs_benchmark = "n/a"
    if "application_vs_benchmark" in profile_name:
        if primary_role == "application":
            application_vs_benchmark = "application"
        elif primary_role == "benchmark":
            application_vs_benchmark = "benchmark"

    system_vs_conceptual = "n/a"
    if "paper_to_agent" in profile_name:
        if primary_role == "system":
            system_vs_conceptual = "system"
        elif primary_role == "survey":
            system_vs_conceptual = "conceptual"

    if "system paper" in text or "framework" in text:
        system_vs_conceptual = "system"
    if "survey" in text or "taxonomy" in text or "framing" in text:
        system_vs_conceptual = "conceptual"

    return {
        "role": primary_role,
        "topicFit": topic_fit,
        "alternativeVsOptimization": alternative_vs_optimization,
        "applicationVsBenchmark": application_vs_benchmark,
        "systemVsConceptual": system_vs_conceptual,
    }


def _route_stub(task_type: str, reasons: list[str] | None = None) -> dict[str, Any]:
    return TaskRouteDecision(
        task_type=task_type,  # type: ignore[arg-type]
        route="fallback-only",
        provider="",
        model="",
        timeout_sec=0,
        fallback_chain=["fallback-only"],
        reasons=list(reasons or []),
        allow_external_effective=False,
        complexity_score=0,
        policy_mode="local-only",
    ).to_dict()



class PaperTopicSynthesisService:
    def __init__(
        self,
        *,
        sqlite_db: Any,
        searcher: Any,
        config: Any,
        llm_resolver: Callable[..., tuple[Any | None, dict[str, Any], list[str]]] | None = None,
        paper_memory_retriever_cls: type[PaperMemoryRetriever] = PaperMemoryRetriever,
        document_memory_retriever_cls: type[DocumentMemoryRetriever] = DocumentMemoryRetriever,
    ):
        self.sqlite_db = sqlite_db
        self.searcher = searcher
        self.config = config
        self.llm_resolver = llm_resolver
        self.paper_memory_retriever_cls = paper_memory_retriever_cls
        self.document_memory_retriever_cls = document_memory_retriever_cls

    def synthesize(
        self,
        *,
        query: str,
        source_mode: str = "local",
        candidate_limit: int = 12,
        selected_limit: int = 6,
        top_k: int = 8,
        retrieval_mode: str = "hybrid",
        alpha: float = 0.7,
        allow_external: bool = False,
        llm_mode: str = "auto",
        provider_override: str | None = None,
        model_override: str | None = None,
        timeout_sec: int | None = None,
    ) -> dict[str, Any]:
        text = str(query or "").strip()
        if not text:
            payload = {
                "schema": TOPIC_SYNTHESIS_SCHEMA,
                "status": "failed",
                "query": "",
                "sourceMode": "local",
                "candidatePapers": [],
                "selectedPapers": [],
                "excludedPapers": [],
                "selectionDiagnostics": {"reason": "query_required"},
                "topicSummary": "",
                "architectureGroups": [],
                "comparisonPoints": [],
                "limitations": [],
                "gaps": [],
                "citations": [],
                "verification": {"status": "skipped", "reason": "query_required"},
                "warnings": ["query is required"],
            }
            annotate_schema_errors(payload, TOPIC_SYNTHESIS_SCHEMA)
            return payload

        requested_source_mode = str(source_mode or "local").strip().lower() or "local"
        warnings: list[str] = []
        effective_source_mode = "local"
        if requested_source_mode not in {"local", "discover", "hybrid"}:
            requested_source_mode = "local"
        if requested_source_mode != "local":
            warnings.append(
                f"source_mode={requested_source_mode} is not implemented in v1; falling back to local corpus only."
            )
        topic_profile = _infer_topic_profile_v2(text)

        candidates, retrieval_diag = self._build_candidates(
            query=text,
            candidate_limit=max(6, int(candidate_limit)),
            top_k=max(4, int(top_k)),
            retrieval_mode=str(retrieval_mode or "hybrid").strip().lower() or "hybrid",
            alpha=float(alpha),
            topic_profile=topic_profile,
        )
        selected, excluded, selection_diag, prune_diag = self._judge_candidates(
            query=text,
            candidates=candidates,
            selected_limit=max(1, int(selected_limit)),
            topic_profile=topic_profile,
            allow_external=allow_external,
            llm_mode=llm_mode,
            provider_override=provider_override,
            model_override=model_override,
            timeout_sec=timeout_sec,
        )
        synthesis, synthesis_diag = self._synthesize_selection(
            query=text,
            selected=selected,
            topic_profile=topic_profile,
            allow_external=allow_external,
            llm_mode=llm_mode,
            provider_override=provider_override,
            model_override=model_override,
            timeout_sec=timeout_sec,
        )
        warnings.extend(selection_diag.pop("warnings", []))
        warnings.extend(synthesis_diag.pop("warnings", []))

        verification = self._verify(
            query=text,
            selected=selected,
            topic_summary=str(synthesis.get("topicSummary") or ""),
            comparison_points=list(synthesis.get("comparisonPoints") or []),
            allow_external=allow_external,
        )

        citations = [
            {
                "paperId": str(item.get("paperId") or ""),
                "title": str(item.get("title") or ""),
                "year": item.get("year"),
                "field": str(item.get("field") or ""),
            }
            for item in selected
            if str(item.get("paperId") or "").strip()
        ]

        def _public_selection_row(item: dict[str, Any]) -> dict[str, Any]:
            payload = dict(item)
            selection = dict(payload.get("selection") or {})
            if selection:
                payload.setdefault("decision", str(selection.get("verdict") or "").strip())
                payload.setdefault("rationale", str(selection.get("rationale") or "").strip())
                payload.setdefault("groupLabel", str(selection.get("group") or "").strip())
            if payload.get("topicLabels"):
                payload.setdefault("topicLabels", dict(payload.get("topicLabels") or {}))
            return payload

        payload = {
            "schema": TOPIC_SYNTHESIS_SCHEMA,
            "status": "ok" if selected else "no_result",
            "query": text,
            "sourceMode": requested_source_mode,
            "effectiveSourceMode": effective_source_mode,
            "enrichment": {
                "eligible": False,
                "used": False,
                "mode": "none",
                "reason": "topic_synthesis_memory_first",
                "queryIntent": "paper_topic",
                "enrichmentRoute": "memory_heavy",
                "ontologyEligible": False,
                "clusterEligible": False,
                "ontologyUsed": False,
                "clusterUsed": False,
            },
            "candidatePapers": candidates,
            "selectedPapers": [_public_selection_row(item) for item in selected],
            "excludedPapers": [_public_selection_row(item) for item in excluded],
            "selectionDiagnostics": {
                "requestedSourceMode": requested_source_mode,
                "effectiveSourceMode": effective_source_mode,
                "candidateCount": len(candidates),
                "selectedCount": len(selected),
                "excludedCount": len(excluded),
                "topK": int(top_k),
                "topicProfile": topic_profile,
                "retrieval": retrieval_diag,
                "prune": prune_diag,
                "judge": selection_diag,
                "synthesis": synthesis_diag,
            },
            "topicSummary": str(synthesis.get("topicSummary") or ""),
            "architectureGroups": list(synthesis.get("architectureGroups") or []),
            "comparisonPoints": list(synthesis.get("comparisonPoints") or []),
            "limitations": list(synthesis.get("limitations") or []),
            "gaps": list(synthesis.get("gaps") or []),
            "citations": citations,
            "verification": verification,
            "warnings": list(dict.fromkeys(warnings)),
        }
        annotate_schema_errors(payload, TOPIC_SYNTHESIS_SCHEMA)
        return payload

    def _build_candidates(
        self,
        *,
        query: str,
        candidate_limit: int,
        top_k: int,
        retrieval_mode: str,
        alpha: float,
        topic_profile: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        packets: dict[str, dict[str, Any]] = {}
        title_keys: dict[str, str] = {}
        query_variants = _query_variants(query)
        memory_items: list[dict[str, Any]] = []
        doc_items: list[dict[str, Any]] = []
        search_results: list[SearchResult] = []
        search_diagnostics: list[dict[str, Any]] = []
        for variant in query_variants:
            memory_items.extend(self.paper_memory_retriever_cls(self.sqlite_db).search(variant, limit=max(candidate_limit, 8), include_refs=True))
            doc_items.extend(self.document_memory_retriever_cls(self.sqlite_db).search(variant, limit=max(candidate_limit * 2, 12)))
            search_payload = self.searcher.search_with_diagnostics(
                variant,
                top_k=max(top_k * 2, candidate_limit * 2),
                source_type="paper",
                retrieval_mode=retrieval_mode,
                alpha=alpha,
            )
            search_results.extend(list(search_payload.get("results") or []))
            search_diagnostics.append(
                {
                    "query": variant,
                    "diagnostics": dict(search_payload.get("diagnostics") or {}),
                }
            )

        for item in memory_items:
            packet = self._candidate_from_memory_item(item)
            self._merge_candidate(packet, packets, title_keys)

        for item in doc_items:
            packet = self._candidate_from_document_item(item)
            if packet:
                self._merge_candidate(packet, packets, title_keys)

        for result in search_results:
            packet = self._candidate_from_search_result(result)
            if packet:
                self._merge_candidate(packet, packets, title_keys)

        paper_search_terms = [variant for variant in query_variants if len(variant) >= 3]
        for term in paper_search_terms[:6]:
            for row in list(self.sqlite_db.search_papers(term, limit=max(candidate_limit, 8)) or []):
                packet = self._candidate_from_paper_row(row, matched_via="paper_title_search")
                self._merge_candidate(packet, packets, title_keys)

        ordered = list(packets.values())
        for item in ordered:
            topic_signals = _score_candidate_for_profile_v2(item, topic_profile)
            item["topicSignals"] = topic_signals
            item.setdefault("retrievalSignals", {})["adjustedScore"] = _safe_float(topic_signals.get("adjustedScore"), 0.0)
        ordered.sort(
            key=lambda item: (
                -_safe_float((item.get("retrievalSignals") or {}).get("adjustedScore"), 0.0),
                -_safe_float((item.get("retrievalSignals") or {}).get("candidateScore"), 0.0),
                -_safe_float((item.get("retrievalSignals") or {}).get("paperMemoryScore"), 0.0),
                -_safe_float((item.get("retrievalSignals") or {}).get("documentMemoryScore"), 0.0),
                str(item.get("title") or ""),
            )
        )
        for index, item in enumerate(ordered, start=1):
            item["rank"] = index
        return ordered[:candidate_limit], {
            "memoryHits": len(memory_items),
            "documentHits": len(doc_items),
            "paperSearchHits": len(search_results),
            "dedupedCandidates": len(ordered),
            "queryVariants": query_variants,
            "topicProfile": topic_profile,
            "paperSearchDiagnostics": search_diagnostics,
        }

    def _merge_candidate(self, candidate: dict[str, Any], packets: dict[str, dict[str, Any]], title_keys: dict[str, str]) -> None:
        paper_id = str(candidate.get("paperId") or "").strip()
        title_key = _normalize_title(candidate.get("title"))
        key = paper_id or title_keys.get(title_key) or title_key
        if not key:
            return
        if key not in packets:
            packets[key] = candidate
            if title_key:
                title_keys.setdefault(title_key, key)
            return
        target = packets[key]
        target_methods = set(target.get("matchedVia") or [])
        target_methods.update(candidate.get("matchedVia") or [])
        target["matchedVia"] = sorted(target_methods)
        target_snippets = list(target.get("documentSnippets") or [])
        seen_snippets = {(str(item.get("unitType") or ""), str(item.get("excerpt") or "")) for item in target_snippets}
        for snippet in list(candidate.get("documentSnippets") or []):
            fingerprint = (str(snippet.get("unitType") or ""), str(snippet.get("excerpt") or ""))
            if fingerprint in seen_snippets:
                continue
            target_snippets.append(snippet)
            seen_snippets.add(fingerprint)
        target["documentSnippets"] = target_snippets[:3]
        for field in ("paperCore", "methodCore", "evidenceCore", "limitations", "title", "year", "field"):
            if not str(target.get(field) or "").strip() and str(candidate.get(field) or "").strip():
                target[field] = candidate[field]
        target_signals = dict(target.get("retrievalSignals") or {})
        candidate_signals = dict(candidate.get("retrievalSignals") or {})
        for name in ("paperMemoryScore", "documentMemoryScore", "paperRetrievalScore"):
            target_signals[name] = max(
                _safe_float(target_signals.get(name), 0.0),
                _safe_float(candidate_signals.get(name), 0.0),
            )
        target_signals["candidateScore"] = round(
            _safe_float(target_signals.get("paperMemoryScore"), 0.0)
            + _safe_float(target_signals.get("documentMemoryScore"), 0.0)
            + _safe_float(target_signals.get("paperRetrievalScore"), 0.0),
            6,
        )
        target["retrievalSignals"] = target_signals

    def _hydrate_paper_row(self, paper_id: str) -> dict[str, Any]:
        if not paper_id:
            return {}
        getter = getattr(self.sqlite_db, "get_paper", None)
        if callable(getter):
            row = getter(paper_id)
            if isinstance(row, dict):
                return row
        return {}

    def _base_candidate(self, *, paper_id: str, title: str, year: Any = None, field: str = "") -> dict[str, Any]:
        paper_row = self._hydrate_paper_row(paper_id)
        resolved_title = str(title or paper_row.get("title") or "").strip()
        resolved_year = year if year not in {"", None} else paper_row.get("year")
        resolved_field = str(field or paper_row.get("field") or "").strip()
        return {
            "paperId": str(paper_id or "").strip(),
            "title": resolved_title,
            "year": resolved_year,
            "field": resolved_field,
            "paperCore": "",
            "methodCore": "",
            "evidenceCore": "",
            "limitations": "",
            "conceptLinks": [],
            "documentSnippets": [],
            "matchedVia": [],
            "retrievalSignals": {
                "paperMemoryScore": 0.0,
                "documentMemoryScore": 0.0,
                "paperRetrievalScore": 0.0,
                "candidateScore": 0.0,
            },
        }

    def _candidate_from_memory_item(self, item: dict[str, Any]) -> dict[str, Any]:
        paper = dict(item.get("paper") or {})
        paper_id = str(item.get("paperId") or paper.get("paperId") or "").strip()
        packet = self._base_candidate(
            paper_id=paper_id,
            title=str(item.get("title") or paper.get("title") or "").strip(),
            year=paper.get("year"),
            field=str(paper.get("field") or ""),
        )
        packet.update(
            {
                "paperCore": str(item.get("paperCore") or ""),
                "methodCore": str(item.get("methodCore") or ""),
                "evidenceCore": str(item.get("evidenceCore") or ""),
                "limitations": str(item.get("limitations") or ""),
                "conceptLinks": list(item.get("conceptLinks") or []),
                "matchedVia": ["paper_memory"],
            }
        )
        score = _safe_float(((item.get("retrievalSignals") or {}).get("score")), 0.0)
        packet["retrievalSignals"]["paperMemoryScore"] = round(score, 6)
        packet["retrievalSignals"]["candidateScore"] = round(score, 6)
        return packet

    def _candidate_from_document_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        document_id = str(item.get("documentId") or "").strip()
        matched_unit = dict(item.get("matchedUnit") or {})
        source_ref = str(matched_unit.get("sourceRef") or document_id or "").strip()
        paper_id = ""
        if document_id.startswith("paper:"):
            paper_id = document_id.split("paper:", 1)[1].strip()
        elif source_ref and not source_ref.startswith("vault:") and not source_ref.startswith("web:"):
            paper_id = source_ref
        if not paper_id:
            return None
        packet = self._base_candidate(
            paper_id=paper_id,
            title=str(item.get("documentTitle") or item.get("title") or "").strip(),
        )
        packet["matchedVia"] = ["document_memory"]
        snippet_text = str(
            matched_unit.get("contextualSummary")
            or matched_unit.get("sourceExcerpt")
            or matched_unit.get("documentThesis")
            or ""
        ).strip()
        if snippet_text:
            packet["documentSnippets"] = [
                {
                    "unitType": str(matched_unit.get("unitType") or ""),
                    "sectionPath": str(matched_unit.get("sectionPath") or ""),
                    "excerpt": truncate_text(snippet_text, 280),
                }
            ]
        score = _safe_float(((item.get("retrievalSignals") or {}).get("score")), 0.0)
        packet["retrievalSignals"]["documentMemoryScore"] = round(score, 6)
        packet["retrievalSignals"]["candidateScore"] = round(score, 6)
        return packet

    def _candidate_from_search_result(self, result: SearchResult) -> dict[str, Any] | None:
        metadata = dict(result.metadata or {})
        paper_id = str(metadata.get("arxiv_id") or metadata.get("paper_id") or "").strip()
        if not paper_id:
            return None
        packet = self._base_candidate(
            paper_id=paper_id,
            title=str(metadata.get("title") or "").strip(),
            year=metadata.get("year"),
            field=str(metadata.get("field") or ""),
        )
        packet["matchedVia"] = ["paper_search"]
        packet["retrievalSignals"]["paperRetrievalScore"] = round(_safe_float(result.score, 0.0), 6)
        packet["retrievalSignals"]["candidateScore"] = round(_safe_float(result.score, 0.0), 6)
        excerpt = truncate_text(str(result.document or ""), 280)
        if excerpt:
            packet["documentSnippets"] = [
                {
                    "unitType": str(metadata.get("unit_type") or "retrieval"),
                    "sectionPath": str(metadata.get("section_path") or ""),
                    "excerpt": excerpt,
                }
            ]
        return packet

    def _candidate_from_paper_row(self, row: dict[str, Any], *, matched_via: str) -> dict[str, Any]:
        packet = self._base_candidate(
            paper_id=str(row.get("arxiv_id") or "").strip(),
            title=str(row.get("title") or "").strip(),
            year=row.get("year"),
            field=str(row.get("field") or ""),
        )
        packet["paperCore"] = truncate_text(str(row.get("notes") or ""), 240)
        packet["matchedVia"] = [matched_via]
        packet["retrievalSignals"]["paperRetrievalScore"] = 0.4
        packet["retrievalSignals"]["candidateScore"] = 0.4
        return packet

    def _resolve_llm(
        self,
        *,
        task_type: str,
        query: str,
        context: str,
        source_count: int,
        allow_external: bool,
        llm_mode: str,
        provider_override: str | None,
        model_override: str | None,
        timeout_sec: int,
    ) -> tuple[Any | None, dict[str, Any], list[str]]:
        if callable(self.llm_resolver):
            return self.llm_resolver(
                task_type=task_type,
                allow_external=allow_external,
                llm_mode=llm_mode,
                query=query,
                context=context,
                source_count=source_count,
                provider_override=provider_override,
                model_override=model_override,
                timeout_sec=timeout_sec,
            )
        if provider_override:
            provider = str(provider_override or "").strip()
            if not provider:
                return None, _route_stub(task_type, ["provider_override_missing"]), []
            if not _provider_is_local(provider) and not allow_external:
                return None, _route_stub(task_type, ["provider_override_blocked"]), [
                    "external provider override blocked because allow_external is false"
                ]
            provider_cfg = dict(self.config.get_provider_config(provider))
            provider_cfg["timeout"] = float(timeout_sec)
            provider_cfg["request_timeout"] = float(timeout_sec)
            llm = get_llm(provider, model=str(model_override or "").strip() or None, **provider_cfg)
            route = "local" if _provider_is_local(provider) else "strong"
            return llm, {
                "taskType": task_type,
                "route": route,
                "provider": provider,
                "model": str(model_override or getattr(llm, "model", "") or ""),
                "timeoutSec": int(timeout_sec),
                "reasons": ["provider_override"],
                "allowExternal": bool(allow_external),
            }, []

        force_route = str(llm_mode or "auto").strip().lower() or "auto"
        if not allow_external and force_route in {"auto", "mini", "strong"}:
            force_route = "local"
        llm, decision, warnings = get_llm_for_task(
            self.config,
            task_type=task_type,  # type: ignore[arg-type]
            allow_external=allow_external,
            query=query,
            context=context,
            source_count=source_count,
            force_route=force_route,  # type: ignore[arg-type]
            timeout_sec=timeout_sec,
        )
        return llm, decision.to_dict(), warnings

    def _judge_candidates(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        selected_limit: int,
        topic_profile: dict[str, Any],
        allow_external: bool,
        llm_mode: str,
        provider_override: str | None,
        model_override: str | None,
        timeout_sec: int | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        if not candidates:
            return [], [], {"fallbackUsed": True, "reason": "no_candidates", "route": _route_stub("predicate_reasoning")}, {
                "retrievedCount": 0,
                "keptCount": 0,
                "prunedCount": 0,
                "rows": [],
                "fallbackPoolTooSmall": True,
            }

        shortlist, pruned, prune_diag = _prune_candidates_for_profile_v2(candidates, topic_profile)
        dynamic_limit = _dynamic_selected_limit_v2(shortlist, selected_limit, topic_profile)
        if not shortlist:
            return [], pruned, {"fallbackUsed": True, "reason": "no_candidates_after_prune", "route": _route_stub("predicate_reasoning")}, prune_diag

        judge_window = min(max(2, int(topic_profile.get("judgeWindowSize") or 6)), len(shortlist))
        judge_candidates = list(shortlist[:judge_window])
        compact_context = json.dumps(
            [
                {
                    "paperId": item.get("paperId"),
                    "title": item.get("title"),
                    "year": item.get("year"),
                    "field": item.get("field"),
                    "primaryRole": str((item.get("topicSignals") or {}).get("primaryRole") or ""),
                    "matchedRoles": list((item.get("topicSignals") or {}).get("matchedRoles") or []),
                    "paperCore": truncate_text(str(item.get("paperCore") or ""), 140),
                    "methodCore": truncate_text(str(item.get("methodCore") or ""), 110),
                    "evidenceCore": truncate_text(str(item.get("evidenceCore") or ""), 110),
                    "positiveHits": list((item.get("topicSignals") or {}).get("positiveHits") or [])[:4],
                    "negativeHits": list((item.get("topicSignals") or {}).get("negativeHits") or [])[:3],
                    "mustAvoidHits": list((item.get("topicSignals") or {}).get("mustAvoidHits") or [])[:3],
                    "adjustedScore": _safe_float(((item.get("retrievalSignals") or {}).get("adjustedScore")), 0.0),
                    "candidateScore": _safe_float(((item.get("retrievalSignals") or {}).get("candidateScore")), 0.0),
                }
                for item in judge_candidates
            ],
            ensure_ascii=False,
        )
        llm, route, warnings = self._resolve_llm(
            task_type="predicate_reasoning",
            query=query,
            context=compact_context,
            source_count=len(judge_candidates),
            allow_external=allow_external,
            llm_mode=llm_mode,
            provider_override=provider_override,
            model_override=model_override,
            timeout_sec=int(timeout_sec or 90),
        )
        if llm is None:
            selected, excluded = self._deterministic_selection(shortlist, selected_limit=dynamic_limit, topic_profile=topic_profile)
            return selected, sorted(excluded + pruned, key=lambda item: int(item.get("rank") or 999)), {
                "fallbackUsed": True,
                "reason": "llm_unavailable",
                "route": route,
                "warnings": warnings,
                "requestedSelectedLimit": int(selected_limit),
                "effectiveSelectedLimit": int(dynamic_limit),
            }, prune_diag

        prompt = (
            "You are judging candidate papers for a topic-level literature synthesis.\n"
            "Return JSON only with keys selected and excluded.\n"
            "selected must be an array of objects with keys: paperId, verdict, rationale, group.\n"
            "excluded must be an array of objects with keys: paperId, verdict, rationale.\n"
            "Use verdict values keep, borderline, exclude.\n"
            "Be strict: include only papers that directly fit the topic, not adjacent optimizations.\n"
            "Prefer precise multi-paper selection over noisy breadth.\n"
            f"Preferred roles: {', '.join(topic_profile.get('preferredRoles') or []) or 'none'}\n"
            f"Required roles to cover when possible: {', '.join(topic_profile.get('requiredRoles') or []) or 'none'}\n"
            f"Required topical markers (at least one when applicable): {', '.join(topic_profile.get('requiredAnyMarkers') or []) or 'none'}\n"
            f"Must-avoid markers: {', '.join(topic_profile.get('mustAvoidMarkers') or []) or 'none'}\n"
            f"Topic query: {query}\n"
            f"Candidate limit: {dynamic_limit}\n"
            "Candidates:\n"
            f"{compact_context}"
        )
        try:
            raw = llm.generate(prompt, context="")
            parsed = extract_json_payload(raw)
        except Exception as error:
            selected, excluded = self._deterministic_selection(shortlist, selected_limit=dynamic_limit, topic_profile=topic_profile)
            warnings = [*warnings, f"judge failed: {error}"]
            return selected, sorted(excluded + pruned, key=lambda item: int(item.get("rank") or 999)), {
                "fallbackUsed": True,
                "reason": "judge_error",
                "route": route,
                "warnings": warnings,
                "judgeWindow": judge_window,
                "requestedSelectedLimit": int(selected_limit),
                "effectiveSelectedLimit": int(dynamic_limit),
            }, prune_diag

        decisions_by_id: dict[str, dict[str, Any]] = {}
        selected_items = list(parsed.get("selected") or parsed.get("selectedPapers") or [])
        excluded_items = list(parsed.get("excluded") or parsed.get("excludedPapers") or [])
        for item in selected_items:
            paper_id = str((item or {}).get("paperId") or "").strip()
            if paper_id:
                decisions_by_id[paper_id] = {
                    "verdict": str((item or {}).get("verdict") or (item or {}).get("decision") or "keep").strip() or "keep",
                    "rationale": str((item or {}).get("rationale") or "").strip(),
                    "group": str((item or {}).get("group") or (item or {}).get("groupLabel") or "").strip(),
                }
        excluded_by_id: dict[str, dict[str, Any]] = {}
        for item in excluded_items:
            paper_id = str((item or {}).get("paperId") or "").strip()
            if paper_id:
                excluded_by_id[paper_id] = {
                    "verdict": str((item or {}).get("verdict") or (item or {}).get("decision") or "exclude").strip() or "exclude",
                    "rationale": str((item or {}).get("rationale") or "").strip(),
                }

        if not decisions_by_id:
            selected, excluded = self._deterministic_selection(shortlist, selected_limit=dynamic_limit, topic_profile=topic_profile)
            warnings = [*warnings, "judge returned no selected papers; used deterministic fallback"]
            return selected, sorted(excluded + pruned, key=lambda item: int(item.get("rank") or 999)), {
                "fallbackUsed": True,
                "reason": "judge_empty",
                "route": route,
                "warnings": warnings,
                "judgeWindow": judge_window,
                "requestedSelectedLimit": int(selected_limit),
                "effectiveSelectedLimit": int(dynamic_limit),
            }, prune_diag

        selected: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        candidate_by_id = {str(item.get("paperId") or ""): item for item in shortlist}
        for paper_id, decision in decisions_by_id.items():
            candidate = candidate_by_id.get(paper_id)
            if not candidate:
                continue
            enriched = dict(candidate)
            enriched["selection"] = decision
            enriched["topicLabels"] = _derive_topic_labels_v2(enriched, topic_profile)
            selected.append(enriched)
        for candidate in shortlist:
            paper_id = str(candidate.get("paperId") or "")
            if paper_id in decisions_by_id:
                continue
            decision = excluded_by_id.get(paper_id) or {"verdict": "exclude", "rationale": "not selected by topic judge"}
            enriched = dict(candidate)
            enriched["selection"] = decision
            enriched["topicLabels"] = _derive_topic_labels_v2(enriched, topic_profile)
            excluded.append(enriched)
        selected.sort(key=lambda item: int(item.get("rank") or 999))
        excluded.sort(key=lambda item: int(item.get("rank") or 999))
        selected, excluded = self._ensure_role_coverage(
            selected=selected,
            excluded=excluded,
            selected_limit=dynamic_limit,
            topic_profile=topic_profile,
        )
        return selected[:dynamic_limit], sorted(excluded + pruned, key=lambda item: int(item.get("rank") or 999)), {
            "fallbackUsed": False,
            "reason": "llm_judge",
            "route": route,
            "warnings": warnings,
            "judgeWindow": judge_window,
            "requestedSelectedLimit": int(selected_limit),
            "effectiveSelectedLimit": int(dynamic_limit),
        }, prune_diag

    def _deterministic_selection(
        self,
        candidates: list[dict[str, Any]],
        *,
        selected_limit: int,
        topic_profile: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ordered = sorted(
            candidates,
            key=_candidate_sort_key_v2,
        )
        selected: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for index, candidate in enumerate(ordered):
            enriched = dict(candidate)
            if index < selected_limit:
                enriched["selection"] = {
                    "verdict": "keep",
                    "rationale": "selected by retrieval-backed fallback ranking",
                    "group": self._guess_group(candidate),
                }
                enriched["topicLabels"] = _derive_topic_labels_v2(enriched, topic_profile)
                selected.append(enriched)
            else:
                enriched["selection"] = {
                    "verdict": "exclude",
                    "rationale": "outside top fallback candidate window",
                }
                enriched["topicLabels"] = _derive_topic_labels_v2(enriched, topic_profile)
                excluded.append(enriched)
        selected, excluded = self._ensure_role_coverage(
            selected=selected,
            excluded=excluded,
            selected_limit=selected_limit,
            topic_profile=topic_profile,
        )
        return selected, excluded

    def _ensure_role_coverage(
        self,
        *,
        selected: list[dict[str, Any]],
        excluded: list[dict[str, Any]],
        selected_limit: int,
        topic_profile: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        required_roles = [str(item).strip() for item in list(topic_profile.get("requiredRoles") or []) if str(item).strip()]
        if not required_roles:
            return selected, excluded
        selected_roles = {str((item.get("topicSignals") or {}).get("primaryRole") or "").strip() for item in selected}
        remaining = list(excluded)
        for role in required_roles:
            if role in selected_roles:
                continue
            replacement_index = next(
                (
                    index
                    for index, item in enumerate(remaining)
                    if str((item.get("topicSignals") or {}).get("primaryRole") or "").strip() == role
                ),
                None,
            )
            if replacement_index is None:
                continue
            if not bool((remaining[replacement_index].get("topicSignals") or {}).get("prunePassed", True)):
                continue
            replacement = remaining.pop(replacement_index)
            replacement_selection = dict(replacement.get("selection") or {})
            replacement_selection.setdefault("verdict", "borderline")
            replacement_selection["rationale"] = (
                str(replacement_selection.get("rationale") or "").strip() or "added to preserve required topic role coverage"
            )
            replacement["selection"] = replacement_selection
            if len(selected) < selected_limit:
                selected.append(replacement)
            else:
                selected.sort(key=lambda item: _safe_float(((item.get("retrievalSignals") or {}).get("adjustedScore")), 0.0))
                demoted = selected.pop(0)
                demoted["selection"] = {"verdict": "exclude", "rationale": "demoted to preserve required role coverage"}
                remaining.append(demoted)
                selected.append(replacement)
            selected_roles.add(role)
        selected.sort(key=lambda item: int(item.get("rank") or 999))
        remaining.sort(key=lambda item: int(item.get("rank") or 999))
        return selected, remaining

    def _guess_group(self, candidate: dict[str, Any]) -> str:
        text = " ".join(
            [
                str(candidate.get("title") or ""),
                str(candidate.get("paperCore") or ""),
                " ".join(str(item) for item in list(candidate.get("conceptLinks") or [])),
            ]
        ).casefold()
        rules = (
            ("state space models", ("state space", "mamba", "ssm")),
            ("attention alternatives", ("attention", "linear attention", "retentive", "rwkv")),
            ("memory architectures", ("memory", "external memory", "retrieval memory")),
            ("mixture or modular routing", ("mixture", "moe", "routing", "experts")),
        )
        for label, markers in rules:
            if any(marker in text for marker in markers):
                return label
        return "other"

    def _synthesize_selection(
        self,
        *,
        query: str,
        selected: list[dict[str, Any]],
        topic_profile: dict[str, Any],
        allow_external: bool,
        llm_mode: str,
        provider_override: str | None,
        model_override: str | None,
        timeout_sec: int | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not selected:
            return {
                "topicSummary": "",
                "architectureGroups": [],
                "comparisonPoints": [],
                "limitations": [],
                "gaps": [],
            }, {"fallbackUsed": True, "reason": "no_selected", "route": _route_stub("rag_answer_rewrite"), "warnings": []}

        context = json.dumps(
            [
                {
                    "paperId": item.get("paperId"),
                    "title": item.get("title"),
                    "year": item.get("year"),
                    "field": item.get("field"),
                    "paperCore": truncate_text(str(item.get("paperCore") or ""), 220),
                    "methodCore": truncate_text(str(item.get("methodCore") or ""), 180),
                    "evidenceCore": truncate_text(str(item.get("evidenceCore") or ""), 180),
                    "limitations": truncate_text(str(item.get("limitations") or ""), 140),
                    "group": str((item.get("selection") or {}).get("group") or self._guess_group(item)),
                    "primaryRole": str((item.get("topicSignals") or {}).get("primaryRole") or ""),
                    "topicLabels": dict(item.get("topicLabels") or _derive_topic_labels_v2(item, topic_profile)),
                }
                for item in selected
            ],
            ensure_ascii=False,
        )
        llm, route, warnings = self._resolve_llm(
            task_type="rag_answer_rewrite",
            query=query,
            context=context,
            source_count=len(selected),
            allow_external=allow_external,
            llm_mode=llm_mode,
            provider_override=provider_override,
            model_override=model_override,
            timeout_sec=int(timeout_sec or 120),
        )
        if llm is None:
            return self._deterministic_synthesis(query=query, selected=selected, topic_profile=topic_profile), {
                "fallbackUsed": True,
                "reason": "llm_unavailable",
                "route": route,
                "warnings": warnings,
            }

        prompt = (
            "You are producing a structured multi-paper topic synthesis.\n"
            "Return JSON only with keys topicSummary, architectureGroups, comparisonPoints, limitations, gaps.\n"
            "architectureGroups must be an array of objects with keys label, summary, paperIds.\n"
            "comparisonPoints, limitations, and gaps must be short arrays of strings.\n"
            "Explicitly separate requested paper categories when the query implies them, such as benchmark vs system or survey vs architecture.\n"
            "Use topicLabels to explain why each selected paper fits the requested category and say clearly when only one side of a requested comparison is well-supported.\n"
            f"Preferred roles: {', '.join(topic_profile.get('preferredRoles') or []) or 'none'}\n"
            f"Required roles: {', '.join(topic_profile.get('requiredRoles') or []) or 'none'}\n"
            f"User topic query: {query}\n"
            "Selected papers:\n"
            f"{context}"
        )
        try:
            raw = llm.generate(prompt, context="")
            parsed = extract_json_payload(raw)
        except Exception as error:
            warnings = [*warnings, f"synthesis failed: {error}"]
            return self._deterministic_synthesis(query=query, selected=selected, topic_profile=topic_profile), {
                "fallbackUsed": True,
                "reason": "synthesis_error",
                "route": route,
                "warnings": warnings,
            }
        topic_summary = str(parsed.get("topicSummary") or "").strip()
        if not topic_summary:
            warnings = [*warnings, "synthesis returned empty topicSummary; used deterministic fallback"]
            return self._deterministic_synthesis(query=query, selected=selected, topic_profile=topic_profile), {
                "fallbackUsed": True,
                "reason": "synthesis_empty",
                "route": route,
                "warnings": warnings,
            }
        return {
            "topicSummary": topic_summary,
            "architectureGroups": list(parsed.get("architectureGroups") or []),
            "comparisonPoints": [str(item).strip() for item in list(parsed.get("comparisonPoints") or []) if str(item).strip()],
            "limitations": [str(item).strip() for item in list(parsed.get("limitations") or []) if str(item).strip()],
            "gaps": [str(item).strip() for item in list(parsed.get("gaps") or []) if str(item).strip()],
        }, {
            "fallbackUsed": False,
            "reason": "llm_synthesis",
            "route": route,
            "warnings": warnings,
        }

    def _deterministic_synthesis(
        self,
        *,
        query: str,
        selected: list[dict[str, Any]],
        topic_profile: dict[str, Any],
    ) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in selected:
            label = str((item.get("selection") or {}).get("group") or self._guess_group(item) or "other").strip()
            groups.setdefault(label, []).append(item)
        architecture_groups = []
        for label, items in groups.items():
            architecture_groups.append(
                {
                    "label": label,
                    "summary": truncate_text(
                        " / ".join(
                            filter(
                                None,
                                [
                                    str(items[0].get("paperCore") or ""),
                                    str(items[0].get("methodCore") or ""),
                                ],
                            )
                        ),
                        240,
                    ),
                    "paperIds": [str(item.get("paperId") or "") for item in items if str(item.get("paperId") or "").strip()],
                }
            )
        comparison_points = [
            truncate_text(
                f"{item.get('title')}: {str(item.get('methodCore') or item.get('paperCore') or '')}",
                220,
            )
            for item in selected[:5]
        ]
        limitations = [
            truncate_text(str(item.get("limitations") or "limitations not explicit in candidate packet"), 180)
            for item in selected[:4]
            if str(item.get("limitations") or "").strip()
        ]
        selected_roles = {str((item.get("topicLabels") or _derive_topic_labels_v2(item, topic_profile)).get("role") or "").strip() for item in selected}
        gaps = ["External discovery is disabled in v1, so this view only reflects the current local paper corpus."]
        profile_name = str(topic_profile.get("name") or "")
        if "application_vs_benchmark" in profile_name:
            if "application" not in selected_roles:
                gaps.append("The current local corpus does not yield a strong long-term-memory application paper in the final shortlist.")
            if "benchmark" not in selected_roles:
                gaps.append("The current local corpus does not yield a strong long-term-memory benchmark paper in the final shortlist.")
        if "paper_to_agent" in profile_name:
            labels = {str((item.get("topicLabels") or _derive_topic_labels_v2(item, topic_profile)).get("systemVsConceptual") or "").strip() for item in selected}
            if "system" not in labels:
                gaps.append("The current local corpus does not yield a strong system paper for paper-to-agent in the final shortlist.")
            if "conceptual" not in labels:
                gaps.append("The current local corpus does not yield a strong conceptual-framing paper for paper-to-agent in the final shortlist.")
        return {
            "topicSummary": truncate_text(
                f"Selected {len(selected)} papers for the topic '{query}' from the local paper corpus and grouped them by architecture family.",
                280,
            ),
            "architectureGroups": architecture_groups,
            "comparisonPoints": comparison_points,
            "limitations": limitations,
            "gaps": gaps,
        }

    def _verify(
        self,
        *,
        query: str,
        selected: list[dict[str, Any]],
        topic_summary: str,
        comparison_points: list[str],
        allow_external: bool,
    ) -> dict[str, Any]:
        if not selected:
            return {"status": "skipped", "reason": "no_selected_papers"}
        answer = topic_summary.strip()
        if comparison_points:
            answer = f"{answer}\n- " + "\n- ".join(str(item).strip() for item in comparison_points[:4] if str(item).strip())
        if not answer.strip():
            return {"status": "skipped", "reason": "empty_answer"}
        evidence = []
        for item in selected:
            excerpt = str(item.get("paperCore") or item.get("methodCore") or item.get("evidenceCore") or "").strip()
            if not excerpt and item.get("documentSnippets"):
                excerpt = str((item.get("documentSnippets") or [{}])[0].get("excerpt") or "").strip()
            evidence.append(
                {
                    "title": str(item.get("title") or ""),
                    "excerpt": truncate_text(excerpt, 220),
                    "source_type": "paper",
                    "citation_target": str(item.get("paperId") or item.get("title") or ""),
                }
            )
        try:
            return self.searcher._verify_answer(
                query=query,
                answer=answer,
                evidence=evidence,
                answer_signals={"queryIntent": "paper_topic", "selectedPaperCount": len(selected)},
                contradicting_beliefs=[],
                allow_external=allow_external,
            )
        except Exception as error:
            return {"status": "skipped", "reason": f"verification_failed: {error}"}
