from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import urlparse
import json
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
from knowledge_hub.domain.registry import normalize_domain_source
from knowledge_hub.domain.web_knowledge.evidence_policy import policy_key_for_family
from knowledge_hub.domain.web_knowledge.families import (
    WEB_FAMILY_REFERENCE_EXPLAINER,
    WEB_FAMILY_RELATION_EXPLAINER,
    WEB_FAMILY_SOURCE_DISAMBIGUATION,
    WEB_FAMILY_TEMPORAL_UPDATE,
    classify_web_family,
    explicit_web_scope,
    host_hint,
)
from knowledge_hub.domain.web_knowledge.internal_reference import internal_reference_requested
from knowledge_hub.learning.resolver import normalize_term
from knowledge_hub.web.ingest import make_web_note_id


_DOMAIN_KEY = "web_knowledge"
_TOKEN_RE = re.compile(r"https?://[^\s]+|[A-Za-z0-9._+-]+|[가-힣]+")
_ENTITY_STOPWORDS = {
    "what",
    "is",
    "the",
    "a",
    "an",
    "guide",
    "reference",
    "latest",
    "update",
    "updated",
    "watchlist",
    "feed",
    "source",
    "sources",
    "what's",
    "guide",
    "overview",
    "summary",
    "meaning",
    "what",
    "최근",
    "최신",
    "업데이트",
    "가이드",
    "레퍼런스",
    "참고",
    "요약",
    "설명",
    "정의",
    "무엇",
    "web",
    "card",
    "ask",
    "v2",
    "에서",
    "이",
    "은",
    "는",
    "를",
    "가",
    "을",
    "왜",
    "어떤",
    "하나",
    "글",
    "글은",
    "질문",
    "질문에서",
    "필드",
    "필드를",
    "봐야",
    "역할",
    "역할로",
    "도움",
    "도움이",
    "되나",
    "있는",
    "source",
}
_TEMPORAL_RE = re.compile(r"\b(latest|recent|updated?|newest|before|after|since|release|changed?)\b|최근|최신|업데이트|이전|이후|변경", re.IGNORECASE)
_RELATION_RE = re.compile(r"\b(related|relationship|connected|link|dependency|depends on|ontology)\b|관계|연결|의존|온톨로지", re.IGNORECASE)
_DISAMBIGUATION_RE = re.compile(r"\b(which|choose|distinguish|difference|versus|vs)\b|구분|구별|차이|어디|무엇이 더 맞", re.IGNORECASE)
_IMPLEMENTATION_RE = re.compile(r"\b(field|fields|how|implementation|steps?)\b|필드|어떻게|구현", re.IGNORECASE)
_ABSTAIN_RE = re.compile(
    r"\b(can we|should we|can i|safe to|conclude|claim)\b|단정할 수 있나|강하게 답해야 하나|출처 없이|피해야 하나|강한 최신 답변",
    re.IGNORECASE,
)
_REFERENCE_SOURCE_RE = re.compile(r"\b(reference|guide|guideline|glossary|standard|reference source)\b|reference source|가이드|레퍼런스|참고|정의", re.IGNORECASE)
_WATCHLIST_RE = re.compile(r"\b(watchlist|feed|news|latest)\b|watchlist|feed|뉴스|최신", re.IGNORECASE)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


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


def _query_terms(query: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(_clean_text(query)):
        token = _clean_text(raw)
        lowered = token.casefold()
        if not token or lowered in _ENTITY_STOPWORDS:
            continue
        if token.startswith("http://") or token.startswith("https://"):
            continue
        if re.fullmatch(r"[A-Za-z0-9._+-]+", token) and len(token) < 3:
            continue
        tokens.append(token)
    return _dedupe_lines(tokens, limit=6)


def _query_intent(query: str, family: str) -> str:
    body = _clean_text(query)
    if family == WEB_FAMILY_TEMPORAL_UPDATE:
        return "abstention" if _ABSTAIN_RE.search(body) else "temporal"
    if family == WEB_FAMILY_RELATION_EXPLAINER:
        return "relation"
    if family == WEB_FAMILY_SOURCE_DISAMBIGUATION:
        return "disambiguation"
    if _IMPLEMENTATION_RE.search(body):
        return "implementation"
    return "definition"


def _answer_mode(query: str, family: str, intent: str) -> str:
    if family == WEB_FAMILY_SOURCE_DISAMBIGUATION:
        return "disambiguation"
    if intent == "abstention":
        return "abstain"
    if _IMPLEMENTATION_RE.search(_clean_text(query)):
        return "implementation_steps"
    if family == WEB_FAMILY_RELATION_EXPLAINER:
        return "concept_explainer"
    if family == WEB_FAMILY_TEMPORAL_UPDATE:
        return "timeline_compare"
    return "concise_summary"


def _scope_from_note(note: dict[str, Any] | None) -> tuple[list[str], dict[str, Any]]:
    raw_metadata = (note or {}).get("metadata")
    if isinstance(raw_metadata, dict):
        metadata = dict(raw_metadata)
    else:
        try:
            parsed = json.loads(raw_metadata or "{}")
        except Exception:
            parsed = {}
        metadata = dict(parsed) if isinstance(parsed, dict) else {}
    canonical_url = _clean_text(metadata.get("canonical_url") or metadata.get("source_url") or metadata.get("url"))
    note_id = _clean_text((note or {}).get("id"))
    resolved: list[str] = []
    if canonical_url:
        resolved.append(canonical_url)
    if note_id:
        resolved.append(note_id)
    return _dedupe_lines(resolved, limit=2), metadata


def _resolve_web_scope_candidates(
    query: str,
    *,
    sqlite_db: Any | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    explicit = _clean_text(explicit_web_scope(query, metadata_filter=metadata_filter))
    if explicit:
        if explicit.startswith("http://") or explicit.startswith("https://"):
            return [explicit, make_web_note_id(explicit)], [_clean_text(host_hint(explicit) or urlparse(explicit).netloc)], {}
        return [explicit], [], {}
    if not sqlite_db:
        return [], [], {}
    host = _clean_text(host_hint(query))
    title_terms = _query_terms(query)
    rows = list(sqlite_db.list_notes(source_type="web", limit=1000) or [])
    best_note: dict[str, Any] | None = None
    best_score = 0.0
    for row in rows:
        note = dict(row or {})
        _, metadata = _scope_from_note(note)
        title = _clean_text(note.get("title"))
        canonical_url = _clean_text(metadata.get("canonical_url") or metadata.get("source_url") or metadata.get("url"))
        haystack = " ".join([title, canonical_url, _clean_text(metadata.get("source_channel_type")), _clean_text(metadata.get("reference_tier"))]).casefold()
        score = 0.0
        if host and host.casefold() in haystack:
            score += 3.0
        for term in title_terms[:4]:
            if normalize_term(term) and normalize_term(term) in normalize_term(haystack):
                score += 1.0
        if score > best_score and score >= 2.0:
            best_score = score
            best_note = note
    if not best_note:
        return [], [], {}
    resolved, best_meta = _scope_from_note(best_note)
    expanded = [_clean_text(best_note.get("title")), _clean_text(best_meta.get("canonical_url") or best_meta.get("source_url") or best_meta.get("url"))]
    if host:
        expanded.append(host)
    return resolved, _dedupe_lines(expanded, limit=4), best_meta


def _canonical_entity_ids(query: str, *, sqlite_db: Any | None = None) -> list[str]:
    if not sqlite_db:
        return []
    try:
        from knowledge_hub.learning.resolver import EntityResolver

        resolver = EntityResolver(sqlite_db)
        resolved: list[str] = []
        for token in _query_terms(query)[:4]:
            identity = resolver.resolve(token, entity_type="concept")
            if identity is None:
                continue
            resolved.append(str(identity.canonical_id or ""))
        return _dedupe_lines(resolved, limit=6)
    except Exception:
        return []


def _metadata_filter(
    *,
    query: str,
    family: str,
    resolved_source_ids: list[str],
    metadata_filter: dict[str, Any] | None,
    scope_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = dict(metadata_filter or {})
    payload["source_type"] = "web"
    scope_metadata = dict(scope_metadata or {})
    resolved_url = next((item for item in resolved_source_ids if item.startswith("http://") or item.startswith("https://")), "")
    resolved_doc = next((item for item in resolved_source_ids if item.startswith("web_")), "")
    if resolved_url and not _clean_text(payload.get("canonical_url")):
        payload["canonical_url"] = resolved_url
    if resolved_doc and not _clean_text(payload.get("document_id")):
        payload["document_id"] = resolved_doc
    if family in {WEB_FAMILY_REFERENCE_EXPLAINER, WEB_FAMILY_RELATION_EXPLAINER}:
        payload["reference_only"] = True
    if family == WEB_FAMILY_TEMPORAL_UPDATE:
        payload["latest_only"] = True
        payload["temporal_required"] = True
    source_channel_type = _clean_text(scope_metadata.get("source_channel_type"))
    if _WATCHLIST_RE.search(source_channel_type):
        payload["watchlist_scope"] = source_channel_type
    if internal_reference_requested(query):
        payload["internal_reference_preferred"] = True
    return payload


def _expanded_terms(
    query: str,
    *,
    family: str,
    resolved_source_ids: list[str],
    scope_expanded_terms: list[str],
) -> list[str]:
    priority_terms: list[str] = []
    terms: list[str] = []
    base_terms = _query_terms(query)
    lowered_query = _clean_text(query).casefold()
    if "rerank" in lowered_query or "reranker" in lowered_query:
        priority_terms.extend(["rerank", "vector search", "retrieval quality", "reranker"])
    if "version grounding" in lowered_query or "grounding" in lowered_query or "observed_at" in lowered_query:
        priority_terms.extend(["version grounding", "document_date", "event_date", "observed_at"])
    if "ontology-first" in lowered_query or ("ontology" in lowered_query and "routing" in lowered_query):
        priority_terms.extend(["ontology-first routing", "ontology routing", "ontology", "routing"])
    if "evidence anchor" in lowered_query or ("evidence" in lowered_query and "anchor" in lowered_query):
        priority_terms.extend(["evidence anchor", "document_date", "event_date", "observed_at", "anchor"])
    terms.extend(priority_terms)
    terms.extend(scope_expanded_terms)
    terms.extend(base_terms)
    resolved_url = next((item for item in resolved_source_ids if item.startswith("http://") or item.startswith("https://")), "")
    if resolved_url:
        parsed = urlparse(resolved_url)
        if parsed.netloc:
            terms.append(parsed.netloc)
    if family == WEB_FAMILY_TEMPORAL_UPDATE:
        terms.extend(["latest", "update", "version"])
    if family == WEB_FAMILY_REFERENCE_EXPLAINER:
        terms.extend(["guide", "reference"])
    if family == WEB_FAMILY_SOURCE_DISAMBIGUATION:
        terms.extend(["reference source", "latest update"])
    return _dedupe_lines(terms, limit=6)


def _confidence(family: str, *, resolved_source_ids: list[str], canonical_entity_ids: list[str]) -> float:
    if resolved_source_ids:
        return 0.94
    if family == WEB_FAMILY_TEMPORAL_UPDATE:
        return 0.9
    if canonical_entity_ids:
        return 0.84
    return 0.78


def build_rule_based_query_frame(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    sqlite_db: Any | None = None,
) -> NormalizedQueryFrame:
    normalized_source = normalize_domain_source(source_type) or "web"
    family = classify_web_family(query, source_type=normalized_source, metadata_filter=metadata_filter)
    resolved_source_ids, scoped_terms, scope_metadata = _resolve_web_scope_candidates(
        query,
        sqlite_db=sqlite_db,
        metadata_filter=metadata_filter,
    )
    canonical_entity_ids = _canonical_entity_ids(query, sqlite_db=sqlite_db)
    intent = _query_intent(query, family)
    answer_mode = _answer_mode(query, family, intent)
    effective_filter = _metadata_filter(
        query=query,
        family=family,
        resolved_source_ids=resolved_source_ids,
        metadata_filter=metadata_filter,
        scope_metadata=scope_metadata,
    )
    return build_query_frame(
        domain_key=_DOMAIN_KEY,
        source_type=normalized_source,
        family=family,
        query_intent=intent,
        answer_mode=answer_mode,
        entities=_query_terms(query),
        canonical_entity_ids=canonical_entity_ids,
        expanded_terms=_expanded_terms(
            query,
            family=family,
            resolved_source_ids=resolved_source_ids,
            scope_expanded_terms=scoped_terms,
        ),
        resolved_source_ids=resolved_source_ids,
        confidence=_confidence(family, resolved_source_ids=resolved_source_ids, canonical_entity_ids=canonical_entity_ids),
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key=policy_key_for_family(family),
        metadata_filter=effective_filter,
    )


def query_frame_from_query_plan(
    query_plan: dict[str, Any],
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
    payload = normalize_query_frame_dict(query_plan)
    if not payload:
        return base_frame
    effective_source = normalize_domain_source(payload.get("source_type") or source_type or base_frame.source_type)
    family = str(payload.get("family") or "").strip().lower()
    family_source = QUERY_FRAME_FAMILY_FROM_FRAME if family else QUERY_FRAME_FAMILY_FROM_PACK
    overrides = list(payload.get("overrides_applied") or [])
    if family and not family_supported_for_source(effective_source, family):
        family = base_frame.family
        family_source = QUERY_FRAME_FAMILY_FROM_PACK
        overrides.append("INVALID_FAMILY")
    elif not family:
        family = base_frame.family
        family_source = str(base_frame.family_source or QUERY_FRAME_FAMILY_FROM_PACK)
    query_intent = str(payload.get("query_intent") or "").strip()
    if query_intent and not query_intent_supported_for_family(effective_source, family, query_intent):
        query_intent = base_frame.query_intent
        overrides.append("UNSUPPORTED_INTENT")
    elif not query_intent:
        query_intent = base_frame.query_intent
    answer_mode = str(payload.get("answer_mode") or "").strip()
    if not answer_mode or {"INVALID_FAMILY", "UNSUPPORTED_INTENT"} & set(overrides):
        answer_mode = base_frame.answer_mode
    effective_metadata_filter = dict(base_frame.metadata_filter or {})
    effective_metadata_filter.update(dict(metadata_filter or {}))
    effective_metadata_filter.update(dict(payload.get("metadata_filter") or {}))
    return build_query_frame(
        domain_key=str(payload.get("domain_key") or base_frame.domain_key or _DOMAIN_KEY),
        source_type=effective_source or base_frame.source_type,
        family=family or WEB_FAMILY_REFERENCE_EXPLAINER,
        query_intent=query_intent or "definition",
        answer_mode=answer_mode or "concise_summary",
        entities=_dedupe_lines([*list(payload.get("entities") or []), *list(base_frame.entities or [])], limit=6),
        canonical_entity_ids=list(payload.get("canonical_entity_ids") or base_frame.canonical_entity_ids),
        expanded_terms=_dedupe_lines([*list(payload.get("expanded_terms") or []), *list(base_frame.expanded_terms or [])], limit=6),
        resolved_source_ids=_dedupe_lines([*list(payload.get("resolved_source_ids") or []), *list(base_frame.resolved_source_ids or [])], limit=6),
        confidence=float(payload.get("confidence") or base_frame.confidence or 0.0),
        planner_status=str(payload.get("planner_status") or base_frame.planner_status or "not_attempted"),
        planner_reason=str(payload.get("planner_reason") or base_frame.planner_reason or "rule_based"),
        evidence_policy_key=str(payload.get("evidence_policy_key") or base_frame.evidence_policy_key or ""),
        metadata_filter=effective_metadata_filter,
        frame_provenance=str(payload.get("frame_provenance") or base_frame.frame_provenance or "derived"),
        trusted=bool(payload.get("trusted")),
        lock_mask=list(payload.get("lock_mask") or query_frame_lock_mask(payload)),
        family_source=family_source,
        overrides_applied=overrides,
    )


@dataclass(frozen=True)
class WebQueryPlan:
    family: str
    entities: list[str]
    expanded_terms: list[str]
    resolved_source_ids: list[str]
    answer_mode: str
    confidence: float
    query_intent: str
    planner_used: bool
    planner_reason: str
    planner_status: str
    evidence_policy_key: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["answerMode"] = payload["answer_mode"]
        payload["queryIntent"] = payload["query_intent"]
        payload["plannerUsed"] = payload["planner_used"]
        payload["plannerReason"] = payload["planner_reason"]
        payload["plannerStatus"] = payload["planner_status"]
        payload["resolvedSourceIds"] = list(payload["resolved_source_ids"])
        payload["expandedTerms"] = list(payload["expanded_terms"])
        payload["evidencePolicyKey"] = payload["evidence_policy_key"]
        return payload


def build_rule_query_plan(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    sqlite_db: Any | None = None,
) -> WebQueryPlan:
    frame = build_rule_based_query_frame(
        query,
        source_type=source_type,
        metadata_filter=metadata_filter,
        sqlite_db=sqlite_db,
    )
    return WebQueryPlan(
        family=frame.family,
        entities=list(frame.entities),
        expanded_terms=list(frame.expanded_terms),
        resolved_source_ids=list(frame.resolved_source_ids),
        answer_mode=frame.answer_mode,
        confidence=float(frame.confidence),
        query_intent=frame.query_intent,
        planner_used=False,
        planner_reason=frame.planner_reason,
        planner_status=frame.planner_status,
        evidence_policy_key=frame.evidence_policy_key,
    )


__all__ = [
    "WEB_FAMILY_REFERENCE_EXPLAINER",
    "WEB_FAMILY_TEMPORAL_UPDATE",
    "WEB_FAMILY_RELATION_EXPLAINER",
    "WEB_FAMILY_SOURCE_DISAMBIGUATION",
    "WebQueryPlan",
    "build_rule_based_query_frame",
    "build_rule_query_plan",
    "query_frame_from_query_plan",
]
