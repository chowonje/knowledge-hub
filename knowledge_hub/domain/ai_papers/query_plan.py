from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from knowledge_hub.application.query_frame import (
    QUERY_FRAME_FAMILY_FROM_FRAME,
    QUERY_FRAME_FAMILY_FROM_PACK,
    NormalizedQueryFrame,
    build_query_frame,
    family_supported_for_source,
    normalize_query_frame_dict,
    query_frame_lock_mask,
    query_intent_supported_for_family,
)
from knowledge_hub.ai.rag_support import extract_json_payload
from knowledge_hub.domain.ai_papers.answer_scope import paper_family_answer_mode, paper_family_query_intent
from knowledge_hub.domain.ai_papers.evidence_policy import policy_key_for_family
from knowledge_hub.domain.ai_papers.families import (
    PAPER_FAMILY_COMPARE,
    PAPER_FAMILY_CONCEPT_EXPLAINER,
    PAPER_FAMILY_DISCOVER,
    PAPER_FAMILY_LOOKUP,
    PAPER_FAMILY_VALUES,
    classify_paper_family,
    explicit_paper_id,
)
from knowledge_hub.domain.ai_papers.lookup import (
    extract_compare_title_candidates,
    extract_lookup_title_candidate,
    looks_like_explicit_paper_title,
    resolve_lookup,
    resolve_lookup_from_local_titles,
)
from knowledge_hub.domain.ai_papers.representative import expand_concept_terms, representative_hint
from knowledge_hub.domain.registry import normalize_domain_source
from knowledge_hub.learning.policy import evaluate_policy_for_payload
from knowledge_hub.learning.resolver import normalize_term


_TOKEN_RE = re.compile(r"[A-Za-z0-9.+-]+|[가-힣]+")
_COMMON_ENTITY_RESCUE_FORMS = {
    "cnn": [
        "ImageNet Classification with Deep Convolutional Neural Networks",
        "Deep Convolutional Neural Networks",
        "Convolutional Neural Networks",
    ],
    "dqn": [
        "Playing Atari with Deep Reinforcement Learning",
        "Deep Q-Network",
    ],
    "rag": [
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "Retrieval-Augmented Generation",
    ],
    "fid": [
        "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
        "Fusion-in-Decoder",
    ],
    "gpt": [
        "Language Models are Few-Shot Learners",
    ],
    "mamba": [
        "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "Selective State Space",
    ],
    "gan": [
        "Generative Adversarial Nets",
        "GAN",
    ],
    "diffusion": [
        "Denoising Diffusion Probabilistic Models",
        "High-Resolution Image Synthesis with Latent Diffusion Models",
    ],
    "ppo": [
        "Proximal Policy Optimization Algorithms",
        "Proximal Policy Optimization",
    ],
    "transformer": [
        "Attention Is All You Need",
        "Transformer",
        "Transformers",
    ],
    "vit": [
        "An Image is Worth 16x16 Words",
        "Vision Transformer",
        "Vision Transformers",
    ],
}
_DISCOVER_QUERY_RESCUES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(r"\bstate\s+space\s+model\b|\bssm\b|state space|상태\s*공간", re.IGNORECASE),
        ("Mamba", "Selective State Space", "RetNet", "Hyena", "RWKV"),
    ),
    (
        re.compile(r"transformer.*(alternative|alternatives|successor)|대안\s*아키텍처|transformer\s*대안|post-transformer", re.IGNORECASE),
        ("Mamba", "RetNet", "Hyena", "RWKV", "linear attention"),
    ),
)
_EXPLICIT_TITLE_RESCUES = {
    normalize_term("Playing Atari with Deep Reinforcement Learning"): {
        "paper_id": "1312.5602",
        "title": "Playing Atari with Deep Reinforcement Learning",
    },
    normalize_term("An Image is Worth 16x16 Words"): {
        "paper_id": "2010.11929",
        "title": "An Image is Worth 16x16 Words",
    },
}
_ENTITY_STOPWORDS = {
    "and",
    "about",
    "concept",
    "compare",
    "comparison",
    "define",
    "definition",
    "describe",
    "difference",
    "explain",
    "explainer",
    "how",
    "idea",
    "ideas",
    "main",
    "meaning",
    "paper",
    "papers",
    "principle",
    "related",
    "summary",
    "tradeoff",
    "trade-off",
    "versus",
    "vs",
    "what",
    "with",
    "개념",
    "논문",
    "설명",
    "설명해",
    "설명해줘",
    "설명해주세요",
    "쉽게",
    "요약",
    "요약해줘",
    "요약해주세요",
    "원리",
    "의미",
    "장단점",
    "정리",
    "정리해줘",
    "정의",
    "찾아",
    "찾아줘",
    "차이",
    "비교",
    "비교해",
    "비교해줘",
    "비교해주세요",
    "비교해서",
    "각각",
    "관점",
    "관점에서",
    "핵심",
    "계열",
    "기준",
    "상황",
    "잘하는",
    "잘하",
    "이랑",
    "랑",
    "와",
    "과",
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
_REPRESENTATIVE_DEPRIORITIZED_TITLE_RE = re.compile(
    r"\b(survey|overview|comparison|benchmark)\b|정리|비교|개관|서베이",
    re.IGNORECASE,
)
_QUERY_PLAN_PROMPT = """
You are improving a paper query-understanding plan for a local-first paper assistant.
Return JSON only.

Rules:
- family must be one of: concept_explainer, paper_lookup, paper_compare, paper_discover
- expanded_terms should contain bounded retrieval-friendly terms, aliases, or representative paper phrases
- resolved_paper_ids should only include ids that are explicit in the question text
- do not answer the question
- do not invent citations, metrics, or paper ids

Return shape:
{
  "family": "...",
  "entities": ["..."],
  "expanded_terms": ["..."],
  "resolved_paper_ids": ["..."],
  "answer_mode": "...",
  "confidence": 0.0
}
""".strip()

_DOMAIN_KEY = "ai_papers"


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_token(token: str) -> str:
    value = _clean_text(token)
    for suffix in _KOREAN_PARTICLE_SUFFIXES:
        if len(value) > len(suffix) + 1 and value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def _dedupe_lines(values: list[Any], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = _clean_text(raw)
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


def _bounded_entity_rescue_forms(entities: list[str]) -> list[str]:
    rescue: list[str] = []
    rescue_buckets = [_COMMON_ENTITY_RESCUE_FORMS.get(entity.casefold(), []) for entity in entities[:4]]
    max_depth = max((len(bucket) for bucket in rescue_buckets), default=0)
    for idx in range(max_depth):
        for bucket in rescue_buckets:
            if idx < len(bucket):
                rescue.append(bucket[idx])
    return _dedupe_lines(rescue, limit=6)


def _discover_query_rescue_forms(query: str) -> list[str]:
    text = _clean_text(query)
    if not text:
        return []
    rescue: list[str] = []
    for pattern, terms in _DISCOVER_QUERY_RESCUES:
        if pattern.search(text):
            rescue.extend(list(terms))
    return _dedupe_lines(rescue, limit=6)


def _lookup_seed_forms(query: str, *, entities: list[str], bounded_rescue_forms: list[str], family: str) -> list[str]:
    title_candidate = extract_lookup_title_candidate(query)
    seeds: list[str] = []
    if family != PAPER_FAMILY_COMPARE and looks_like_explicit_paper_title(title_candidate):
        seeds.append(title_candidate)
    if family == PAPER_FAMILY_COMPARE:
        rescued_entities = {
            entity.casefold()
            for entity in entities
            if _COMMON_ENTITY_RESCUE_FORMS.get(entity.casefold())
        }
        seeds.extend(bounded_rescue_forms)
        seeds.extend(
            entity
            for entity in entities
            if entity.casefold() not in rescued_entities
        )
    else:
        seeds.extend([query, *entities])
    return _dedupe_lines(seeds, limit=6)


def _prefer_explainer_representative_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    role_appropriate = [
        item
        for item in candidates
        if not _REPRESENTATIVE_DEPRIORITIZED_TITLE_RE.search(_clean_text(item.get("title")))
    ]
    return role_appropriate or candidates


def _explicit_title_rescue(title_candidate: str) -> tuple[list[str], list[str]]:
    token = normalize_term(title_candidate)
    if not token:
        return [], []
    row = _EXPLICIT_TITLE_RESCUES.get(token) or {}
    paper_id = _clean_text(row.get("paper_id"))
    title = _clean_text(row.get("title"))
    return (
        [paper_id] if paper_id else [],
        [title] if title else [],
    )


def _rule_based_entities(query: str, *, family: str = "") -> list[str]:
    entities: list[str] = []
    for raw in _TOKEN_RE.findall(_clean_text(query)):
        token = _clean_token(raw)
        lowered = token.casefold()
        if not token or lowered in _ENTITY_STOPWORDS:
            continue
        if family == PAPER_FAMILY_COMPARE and lowered in {"between"}:
            continue
        if re.fullmatch(r"[가-힣]+", token) and len(token) < 2:
            continue
        if re.fullmatch(r"[A-Za-z0-9.+-]+", token) and len(token) < 2:
            continue
        entities.append(token)
    return _dedupe_lines(entities, limit=6)


def _resolve_canonical_entity_ids(
    entities: list[str],
    *,
    sqlite_db: Any | None = None,
) -> list[str]:
    if not sqlite_db or not entities:
        return []
    try:
        from knowledge_hub.learning.resolver import EntityResolver

        resolver = EntityResolver(sqlite_db)
        resolved: list[str] = []
        for entity in entities[:4]:
            identity = resolver.resolve(entity, entity_type="concept")
            if identity is None:
                continue
            resolved.append(str(identity.canonical_id or ""))
        return _dedupe_lines(resolved, limit=6)
    except Exception:
        return []


def _rule_based_confidence(query: str, family: str, *, has_explicit_paper_id: bool) -> float:
    body = _clean_text(query)
    if has_explicit_paper_id:
        return 0.99
    if family == PAPER_FAMILY_COMPARE:
        return 0.92
    if family == PAPER_FAMILY_DISCOVER:
        return 0.88 if "찾" in body or "related" in body.lower() else 0.62
    if family == PAPER_FAMILY_LOOKUP:
        return 0.86 if ("논문" in body or "paper" in body.lower() or "요약" in body) else 0.74
    if family == PAPER_FAMILY_CONCEPT_EXPLAINER:
        return 0.9 if ("설명" in body or "concept" in body.lower() or "what is" in body.lower()) else 0.66
    return 0.55


def _answer_mode_for_query(query: str, family: str) -> str:
    return paper_family_answer_mode(family, query=query)


@dataclass(frozen=True)
class PaperQueryPlan:
    family: str
    entities: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    resolved_paper_ids: tuple[str, ...]
    answer_mode: str
    confidence: float
    planner_used: bool = False
    query_intent: str = ""
    planner_reason: str = "rule_based"
    evidence_policy_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["entities"] = list(self.entities)
        payload["expanded_terms"] = list(self.expanded_terms)
        payload["resolved_paper_ids"] = list(self.resolved_paper_ids)
        payload["answerMode"] = payload["answer_mode"]
        payload["queryIntent"] = payload["query_intent"]
        payload["plannerUsed"] = payload["planner_used"]
        payload["plannerReason"] = payload["planner_reason"]
        payload["evidencePolicyKey"] = payload["evidence_policy_key"]
        payload["expandedTerms"] = list(self.expanded_terms)
        payload["resolvedPaperIds"] = list(self.resolved_paper_ids)
        return payload


def query_frame_to_query_plan(frame: NormalizedQueryFrame | dict[str, Any]) -> PaperQueryPlan:
    payload = normalize_query_frame_dict(frame)
    planner_status = _clean_text(payload.get("planner_status") or "not_attempted").lower()
    return PaperQueryPlan(
        family=_clean_text(payload.get("family")).lower(),
        entities=tuple(_dedupe_lines(list(payload.get("entities") or []), limit=6)),
        expanded_terms=tuple(_dedupe_lines(list(payload.get("expanded_terms") or []), limit=6)),
        resolved_paper_ids=tuple(_dedupe_lines(list(payload.get("resolved_source_ids") or []), limit=3)),
        answer_mode=_clean_text(payload.get("answer_mode")),
        confidence=max(0.0, min(1.0, float(payload.get("confidence") or 0.0))),
        planner_used=planner_status == "used",
        query_intent=_clean_text(payload.get("query_intent")),
        planner_reason=_clean_text(payload.get("planner_reason") or "rule_based"),
        evidence_policy_key=_clean_text(payload.get("evidence_policy_key")),
    )


def build_rule_based_query_frame(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    sqlite_db: Any | None = None,
) -> NormalizedQueryFrame:
    family = classify_paper_family(query, source_type=source_type, metadata_filter=metadata_filter)
    scoped_paper_id = explicit_paper_id(query, metadata_filter=metadata_filter)
    entities = _rule_based_entities(query, family=family)
    bounded_rescue_forms = _bounded_entity_rescue_forms(entities)
    canonical_entity_ids = _resolve_canonical_entity_ids(entities, sqlite_db=sqlite_db)
    resolved_ids = [scoped_paper_id] if scoped_paper_id else []
    expanded_terms: list[str] = []
    lookup_ids: list[str] = []
    lookup_titles: list[str] = []
    expanded_term_limit = 6
    representative_candidates = representative_hint(entities, sqlite_db=sqlite_db)
    if family == PAPER_FAMILY_CONCEPT_EXPLAINER:
        representative_candidates = _prefer_explainer_representative_candidates(representative_candidates)
    representative_ids = [
        _clean_text(item.get("paperId"))
        for item in representative_candidates
        if _clean_text(item.get("paperId"))
    ]
    representative_titles = [
        _clean_text(item.get("title"))
        for item in representative_candidates
        if _clean_text(item.get("title"))
    ]
    title_candidate = extract_lookup_title_candidate(query)
    compare_title_candidates = extract_compare_title_candidates(query) if family == PAPER_FAMILY_COMPARE else []
    explicit_compare_titles = [
        candidate
        for candidate in compare_title_candidates
        if looks_like_explicit_paper_title(candidate) and not re.fullmatch(r"[A-Z][A-Z0-9.+-]{1,12}", candidate)
    ]
    # Compare queries already get representative ids from concept-linked entities.
    # Prefer bounded rescue forms here so lookup can recover the second paper even
    # when the ontology lacks the compare-side alias. Lookup queries also try a
    # stripped title candidate before the raw query so title-like prompts stay scoped.
    lookup_seed = _lookup_seed_forms(
        query,
        entities=entities,
        bounded_rescue_forms=bounded_rescue_forms,
        family=family,
    )
    lookup_forms = _dedupe_lines([*lookup_seed, *bounded_rescue_forms], limit=6)
    if family in {PAPER_FAMILY_LOOKUP, PAPER_FAMILY_COMPARE}:
        card_lookup_ids, card_lookup_titles = resolve_lookup(lookup_forms, sqlite_db=sqlite_db)
        lookup_ids, lookup_titles = list(card_lookup_ids), list(card_lookup_titles)
        if family == PAPER_FAMILY_LOOKUP and title_candidate:
            local_lookup_ids, local_lookup_titles = resolve_lookup_from_local_titles(
                [title_candidate],
                sqlite_db=sqlite_db,
            )
            strict_title_lookup = looks_like_explicit_paper_title(title_candidate)
            if strict_title_lookup:
                rescue_ids, rescue_titles = _explicit_title_rescue(title_candidate)
                strict_card_lookup_ids, strict_card_lookup_titles = resolve_lookup([title_candidate], sqlite_db=sqlite_db)
                strict_lookup_ids = _dedupe_lines(
                    [*rescue_ids, *local_lookup_ids, *strict_card_lookup_ids],
                    limit=3,
                )
                strict_lookup_titles = _dedupe_lines(
                    [*rescue_titles, *local_lookup_titles, *strict_card_lookup_titles],
                    limit=3,
                )
                lookup_ids = strict_lookup_ids
                lookup_titles = strict_lookup_titles
            elif local_lookup_ids:
                lookup_ids = _dedupe_lines(local_lookup_ids, limit=3)
                lookup_titles = _dedupe_lines(local_lookup_titles, limit=3)
            else:
                lookup_ids = _dedupe_lines(card_lookup_ids, limit=3)
                lookup_titles = _dedupe_lines(card_lookup_titles, limit=3)
        elif family == PAPER_FAMILY_COMPARE and compare_title_candidates:
            compare_lookup_ids: list[str] = []
            compare_lookup_titles: list[str] = []
            for candidate in compare_title_candidates[:3]:
                rescue_ids, rescue_titles = _explicit_title_rescue(candidate)
                local_ids, local_titles = resolve_lookup_from_local_titles([candidate], sqlite_db=sqlite_db)
                strict_card_ids, strict_card_titles = resolve_lookup([candidate], sqlite_db=sqlite_db)
                compare_lookup_ids.extend([*rescue_ids, *local_ids, *strict_card_ids])
                compare_lookup_titles.extend([*rescue_titles, *local_titles, *strict_card_titles])
            if explicit_compare_titles:
                lookup_ids = _dedupe_lines(compare_lookup_ids, limit=3)
                lookup_titles = _dedupe_lines(compare_lookup_titles, limit=6)
            else:
                lookup_ids = _dedupe_lines([*compare_lookup_ids, *card_lookup_ids], limit=3)
                lookup_titles = _dedupe_lines([*compare_lookup_titles, *card_lookup_titles], limit=6)
    elif family == PAPER_FAMILY_CONCEPT_EXPLAINER:
        concept_lookup_forms = _dedupe_lines(
            [*bounded_rescue_forms, *representative_titles, *entities],
            limit=6,
        )
        if concept_lookup_forms:
            lookup_ids, lookup_titles = resolve_lookup(concept_lookup_forms, sqlite_db=sqlite_db)
    if family == PAPER_FAMILY_CONCEPT_EXPLAINER:
        expanded_terms = [
            *bounded_rescue_forms,
            *lookup_titles[:2],
            *representative_titles[:2],
            *expand_concept_terms(entities, sqlite_db=sqlite_db),
            *entities[:3],
        ]
        concept_anchor_ids = lookup_ids[:1] or representative_ids[:1]
        resolved_ids = [*resolved_ids, *concept_anchor_ids]
    elif family == PAPER_FAMILY_DISCOVER:
        expanded_terms = [
            *_discover_query_rescue_forms(query),
            *entities[:5],
            *bounded_rescue_forms[:2],
        ]
    elif family == PAPER_FAMILY_LOOKUP:
        resolved_ids = [*resolved_ids, *lookup_ids]
        expanded_terms = [
            *([title_candidate] if title_candidate and (looks_like_explicit_paper_title(title_candidate) or lookup_ids or lookup_titles) else []),
            *lookup_titles,
            *bounded_rescue_forms[:2],
            *entities[:3],
        ]
    elif family == PAPER_FAMILY_COMPARE:
        compare_ids: list[str] = []
        compare_id_candidates = [*lookup_ids, *representative_ids] if explicit_compare_titles else [*representative_ids, *lookup_ids]
        for paper_id in compare_id_candidates:
            if len(compare_ids) >= 2:
                break
            if paper_id not in compare_ids:
                compare_ids.append(paper_id)
        resolved_ids = [*resolved_ids, *compare_ids]
        expanded_terms = [*compare_title_candidates, *entities[:4], *bounded_rescue_forms, *lookup_titles, *representative_titles[:2]]
        expanded_term_limit = 8
    return build_query_frame(
        domain_key=_DOMAIN_KEY,
        source_type=normalize_domain_source(source_type),
        family=family,
        query_intent=paper_family_query_intent(family, fallback="paper_topic"),
        answer_mode=_answer_mode_for_query(query, family),
        entities=entities,
        canonical_entity_ids=canonical_entity_ids,
        expanded_terms=_dedupe_lines(expanded_terms, limit=expanded_term_limit),
        resolved_source_ids=_dedupe_lines(resolved_ids, limit=3),
        confidence=_rule_based_confidence(query, family, has_explicit_paper_id=bool(scoped_paper_id)),
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key=policy_key_for_family(family),
        metadata_filter=dict(metadata_filter or {}),
    )


def normalize_query_plan_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, NormalizedQueryFrame):
        return value.to_query_plan_dict()
    if isinstance(value, PaperQueryPlan):
        return value.to_dict()
    if isinstance(value, dict):
        confidence = value.get("confidence")
        try:
            normalized_confidence = max(0.0, min(1.0, float(confidence or 0.0)))
        except Exception:
            normalized_confidence = 0.0
        normalized = {
            "family": _clean_text(value.get("family")).lower(),
            "entities": _dedupe_lines(list(value.get("entities") or []), limit=6),
            "expanded_terms": _dedupe_lines(list(value.get("expanded_terms") or value.get("expandedTerms") or []), limit=6),
            "resolved_paper_ids": _dedupe_lines(list(value.get("resolved_paper_ids") or value.get("resolvedPaperIds") or []), limit=3),
            "answer_mode": _clean_text(value.get("answer_mode") or value.get("answerMode")),
            "confidence": normalized_confidence,
            "planner_used": bool(value.get("planner_used") or value.get("plannerUsed")),
            "query_intent": _clean_text(value.get("query_intent") or value.get("queryIntent")),
            "planner_reason": _clean_text(value.get("planner_reason") or value.get("plannerReason") or "rule_based"),
            "planner_status": _clean_text(value.get("planner_status") or value.get("plannerStatus")),
            "planner_warnings": list(value.get("planner_warnings") or value.get("plannerWarnings") or []),
            "planner_route": dict(value.get("planner_route") or value.get("plannerRoute") or {}),
            "evidence_policy_key": _clean_text(value.get("evidence_policy_key") or value.get("evidencePolicyKey")),
        }
        normalized["answerMode"] = normalized["answer_mode"]
        normalized["queryIntent"] = normalized["query_intent"]
        normalized["plannerUsed"] = normalized["planner_used"]
        normalized["plannerReason"] = normalized["planner_reason"]
        normalized["plannerStatus"] = normalized["planner_status"]
        normalized["plannerWarnings"] = list(normalized["planner_warnings"])
        normalized["plannerRoute"] = dict(normalized["planner_route"])
        normalized["expandedTerms"] = list(normalized["expanded_terms"])
        normalized["resolvedPaperIds"] = list(normalized["resolved_paper_ids"])
        normalized["evidencePolicyKey"] = normalized["evidence_policy_key"]
        return normalized
    return {}


def query_frame_from_query_plan(
    query_plan: dict[str, Any] | NormalizedQueryFrame | PaperQueryPlan | None,
    *,
    query: str = "",
    source_type: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    sqlite_db: Any | None = None,
) -> NormalizedQueryFrame:
    base_frame = build_rule_based_query_frame(
        query,
        source_type=source_type,
        metadata_filter=metadata_filter,
        sqlite_db=sqlite_db,
    )
    if not query_plan:
        return base_frame
    normalized = normalize_query_plan_dict(query_plan)
    payload = normalize_query_frame_dict(query_plan)
    effective_source = normalize_domain_source(payload.get("source_type") or source_type or base_frame.source_type)
    family = _clean_text(payload.get("family") or normalized.get("family") or "").lower()
    family_source = QUERY_FRAME_FAMILY_FROM_FRAME if family else QUERY_FRAME_FAMILY_FROM_PACK
    overrides = list(payload.get("overrides_applied") or [])
    if family and not family_supported_for_source(effective_source, family):
        family = base_frame.family
        family_source = QUERY_FRAME_FAMILY_FROM_PACK
        overrides.append("INVALID_FAMILY")
    elif not family:
        family = base_frame.family
        family_source = str(base_frame.family_source or QUERY_FRAME_FAMILY_FROM_PACK)
    query_intent = _clean_text(payload.get("query_intent") or normalized.get("query_intent") or normalized.get("queryIntent"))
    if query_intent and not query_intent_supported_for_family(effective_source, family, query_intent):
        query_intent = base_frame.query_intent
        overrides.append("UNSUPPORTED_INTENT")
    elif not query_intent:
        query_intent = base_frame.query_intent
    answer_mode = _clean_text(payload.get("answer_mode") or normalized.get("answer_mode") or normalized.get("answerMode"))
    if not answer_mode or {"INVALID_FAMILY", "UNSUPPORTED_INTENT"} & set(overrides):
        answer_mode = base_frame.answer_mode
    entities = _dedupe_lines([*list(payload.get("entities") or normalized.get("entities") or []), *list(base_frame.entities or [])], limit=6)
    canonical_entity_ids = list(payload.get("canonical_entity_ids") or base_frame.canonical_entity_ids)
    expanded_terms = _dedupe_lines([*list(payload.get("expanded_terms") or normalized.get("expanded_terms") or []), *list(base_frame.expanded_terms or [])], limit=6)
    resolved_source_ids = _dedupe_lines(
        [
            *list(payload.get("resolved_source_ids") or normalized.get("resolved_paper_ids") or normalized.get("resolvedPaperIds") or []),
            *list(base_frame.resolved_source_ids or []),
        ],
        limit=3,
    )
    effective_metadata_filter = dict(base_frame.metadata_filter or {})
    effective_metadata_filter.update(dict(metadata_filter or {}))
    effective_metadata_filter.update(dict(payload.get("metadata_filter") or {}))
    return build_query_frame(
        domain_key=str(payload.get("domain_key") or base_frame.domain_key or _DOMAIN_KEY),
        source_type=effective_source or base_frame.source_type,
        family=family,
        query_intent=query_intent or paper_family_query_intent(family, fallback="paper_topic"),
        answer_mode=answer_mode or paper_family_answer_mode(family),
        entities=entities,
        canonical_entity_ids=canonical_entity_ids,
        expanded_terms=expanded_terms,
        resolved_source_ids=resolved_source_ids,
        confidence=max(0.0, min(1.0, float(payload.get("confidence") or normalized.get("confidence") or base_frame.confidence or 0.0))),
        planner_status=_clean_text(payload.get("planner_status") or normalized.get("planner_status") or base_frame.planner_status or "not_attempted"),
        planner_reason=_clean_text(payload.get("planner_reason") or normalized.get("planner_reason") or base_frame.planner_reason or "rule_based"),
        evidence_policy_key=_clean_text(payload.get("evidence_policy_key") or normalized.get("evidence_policy_key") or base_frame.evidence_policy_key or policy_key_for_family(family)),
        metadata_filter=effective_metadata_filter,
        frame_provenance=str(payload.get("frame_provenance") or base_frame.frame_provenance or "derived"),
        trusted=bool(payload.get("trusted")),
        lock_mask=list(payload.get("lock_mask") or query_frame_lock_mask(payload)),
        family_source=family_source,
        overrides_applied=overrides,
    )


def build_rule_based_paper_query_plan(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    sqlite_db: Any | None = None,
) -> PaperQueryPlan:
    return query_frame_to_query_plan(
        build_rule_based_query_frame(
            query,
            source_type=source_type,
            metadata_filter=metadata_filter,
            sqlite_db=sqlite_db,
        )
    )


def merge_query_plans(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    merged = dict(normalize_query_plan_dict(base))
    update = dict(normalize_query_plan_dict(candidate))
    family = _clean_text(update.get("family")).lower()
    if family in PAPER_FAMILY_VALUES:
        merged["family"] = family
    merged["entities"] = _dedupe_lines([*list(merged.get("entities") or []), *list(update.get("entities") or [])], limit=6)
    merged["expanded_terms"] = _dedupe_lines([*list(merged.get("expanded_terms") or []), *list(update.get("expanded_terms") or [])], limit=6)
    merged["resolved_paper_ids"] = _dedupe_lines(
        [*list(merged.get("resolved_paper_ids") or []), *list(update.get("resolved_paper_ids") or [])],
        limit=3,
    )
    merged["answer_mode"] = _clean_text(update.get("answer_mode") or merged.get("answer_mode") or paper_family_answer_mode(merged.get("family", "")))
    merged["confidence"] = max(float(merged.get("confidence") or 0.0), float(update.get("confidence") or 0.0))
    merged["planner_used"] = bool(update.get("planner_used") or merged.get("planner_used"))
    merged["planner_reason"] = _clean_text(update.get("planner_reason") or merged.get("planner_reason") or "rule_based")
    merged["query_intent"] = _clean_text(update.get("query_intent") or merged.get("query_intent") or paper_family_query_intent(merged.get("family", ""), fallback="definition"))
    merged["planner_status"] = _clean_text(update.get("planner_status") or merged.get("planner_status") or "")
    merged["planner_warnings"] = list(update.get("planner_warnings") or merged.get("planner_warnings") or [])
    merged["planner_route"] = dict(update.get("planner_route") or merged.get("planner_route") or {})
    merged["answerMode"] = merged["answer_mode"]
    merged["queryIntent"] = merged["query_intent"]
    merged["plannerUsed"] = merged["planner_used"]
    merged["plannerReason"] = merged["planner_reason"]
    merged["plannerStatus"] = merged["planner_status"]
    merged["plannerWarnings"] = list(merged["planner_warnings"])
    merged["plannerRoute"] = dict(merged["planner_route"])
    merged["expandedTerms"] = list(merged["expanded_terms"])
    merged["resolvedPaperIds"] = list(merged["resolved_paper_ids"])
    merged["evidence_policy_key"] = _clean_text(update.get("evidence_policy_key") or merged.get("evidence_policy_key") or policy_key_for_family(merged.get("family", "")))
    merged["evidencePolicyKey"] = merged["evidence_policy_key"]
    return merged


def planner_fallback_payload(*, attempted: bool, used: bool, reason: str, route: dict[str, Any] | None = None, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "attempted": bool(attempted),
        "used": bool(used),
        "reason": _clean_text(reason),
        "route": dict(route or {}),
        "warnings": list(warnings or []),
    }


def should_attempt_query_planner(
    query_plan: dict[str, Any],
    *,
    reason: str,
    source_type: str | None,
) -> bool:
    normalized_source = normalize_domain_source(source_type)
    if normalized_source != "paper":
        return False
    family = _clean_text(query_plan.get("family")).lower()
    if family not in {PAPER_FAMILY_CONCEPT_EXPLAINER, PAPER_FAMILY_DISCOVER}:
        return False
    planner_status = _clean_text(query_plan.get("planner_status") or query_plan.get("plannerStatus")).lower()
    if bool(query_plan.get("planner_used")) or planner_status in {"attempted", "used"}:
        return False
    if reason == "low_confidence":
        return float(query_plan.get("confidence") or 0.0) < 0.72
    if reason == "no_result":
        return True
    return False


def maybe_apply_llm_query_planner(
    searcher: Any,
    *,
    query: str,
    source_type: str | None,
    allow_external: bool,
    base_query_plan: dict[str, Any],
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_base = normalize_query_plan_dict(base_query_plan)
    if not should_attempt_query_planner(normalized_base, reason=reason, source_type=source_type):
        return normalized_base, planner_fallback_payload(attempted=False, used=False, reason="not_applicable")

    policy = evaluate_policy_for_payload(
        allow_external=True,
        raw_texts=[query],
        mode="rag-query-planner",
    )
    if str(policy.classification or "").strip().upper() == "P0":
        return normalized_base, planner_fallback_payload(attempted=False, used=False, reason="policy_blocked_p0")

    llm, route_meta, warnings = searcher._resolve_llm_for_request(  # noqa: SLF001 - existing runtime gate
        query=query,
        context=query,
        source_count=0,
        allow_external=allow_external,
    )
    if llm is None:
        return normalized_base, planner_fallback_payload(
            attempted=True,
            used=False,
            reason="planner_llm_unavailable",
            route=route_meta,
            warnings=warnings,
        )

    try:
        raw = llm.generate(_QUERY_PLAN_PROMPT, query)
    except Exception as error:
        return normalized_base, planner_fallback_payload(
            attempted=True,
            used=False,
            reason=f"planner_error:{type(error).__name__}",
            route=route_meta,
            warnings=warnings,
        )

    parsed = extract_json_payload(str(raw or ""))
    family = _clean_text(parsed.get("family")).lower()
    if family not in PAPER_FAMILY_VALUES:
        return normalized_base, planner_fallback_payload(
            attempted=True,
            used=False,
            reason="planner_invalid_family",
            route=route_meta,
            warnings=warnings,
        )

    scoped_paper_id = explicit_paper_id(query)
    resolved_ids = _dedupe_lines(list(parsed.get("resolved_paper_ids") or []), limit=3)
    if scoped_paper_id:
        resolved_ids = _dedupe_lines([scoped_paper_id, *resolved_ids], limit=3)
    else:
        resolved_ids = list(normalized_base.get("resolved_paper_ids") or [])
    candidate = {
        "family": family,
        "entities": _dedupe_lines(list(parsed.get("entities") or []), limit=6),
        "expanded_terms": _dedupe_lines(list(parsed.get("expanded_terms") or []), limit=6),
        "resolved_paper_ids": resolved_ids,
        "answer_mode": _clean_text(parsed.get("answer_mode") or paper_family_answer_mode(family)),
        "confidence": max(0.0, min(1.0, float(parsed.get("confidence") or 0.0))),
        "planner_used": True,
        "query_intent": paper_family_query_intent(family, fallback=str(normalized_base.get("query_intent") or "definition")),
        "planner_reason": f"llm_fallback:{reason}",
    }
    merged = merge_query_plans(normalized_base, candidate)
    merged["planner_status"] = "used"
    merged["planner_warnings"] = list(warnings)
    merged["planner_route"] = dict(route_meta or {})
    merged["plannerStatus"] = "used"
    merged["plannerWarnings"] = list(warnings)
    merged["plannerRoute"] = dict(route_meta or {})
    return merged, planner_fallback_payload(
        attempted=True,
        used=True,
        reason=f"llm_query_planner:{reason}",
        route=route_meta,
        warnings=warnings,
    )


PAPER_FAMILY_CONCEPT = PAPER_FAMILY_CONCEPT_EXPLAINER


def build_rule_query_plan(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    sqlite_db: Any | None = None,
) -> PaperQueryPlan:
    return build_rule_based_paper_query_plan(
        query,
        source_type=source_type,
        metadata_filter=metadata_filter,
        sqlite_db=sqlite_db,
    )


def maybe_apply_planner_fallback(
    *,
    query: str,
    query_plan: dict[str, Any],
    searcher: Any,
    allow_external: bool,
    reason: str,
) -> dict[str, Any]:
    merged, planner_payload = maybe_apply_llm_query_planner(
        searcher,
        query=query,
        source_type="paper",
        allow_external=allow_external,
        base_query_plan=query_plan,
        reason=reason,
    )
    merged["planner_used"] = bool(planner_payload.get("used"))
    merged["planner_reason"] = _clean_text(planner_payload.get("reason") or merged.get("planner_reason") or "")
    merged["planner_status"] = "used" if planner_payload.get("used") else ("attempted" if planner_payload.get("attempted") else "not_attempted")
    merged["planner_warnings"] = list(planner_payload.get("warnings") or [])
    merged["planner_route"] = dict(planner_payload.get("route") or {})
    merged["plannerUsed"] = merged["planner_used"]
    merged["plannerReason"] = merged["planner_reason"]
    merged["plannerStatus"] = merged["planner_status"]
    merged["plannerWarnings"] = list(merged["planner_warnings"])
    merged["plannerRoute"] = dict(merged["planner_route"])
    return merged


__all__ = [
    "PAPER_FAMILY_COMPARE",
    "PAPER_FAMILY_CONCEPT",
    "PAPER_FAMILY_CONCEPT_EXPLAINER",
    "PAPER_FAMILY_DISCOVER",
    "PAPER_FAMILY_LOOKUP",
    "PAPER_FAMILY_VALUES",
    "PaperQueryPlan",
    "build_rule_based_query_frame",
    "build_rule_query_plan",
    "build_rule_based_paper_query_plan",
    "classify_paper_family",
    "maybe_apply_planner_fallback",
    "maybe_apply_llm_query_planner",
    "merge_query_plans",
    "normalize_query_plan_dict",
    "paper_family_answer_mode",
    "paper_family_query_intent",
    "planner_fallback_payload",
    "query_frame_from_query_plan",
    "query_frame_to_query_plan",
    "should_attempt_query_planner",
]
