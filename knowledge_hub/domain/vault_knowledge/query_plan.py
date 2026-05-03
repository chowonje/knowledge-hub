from __future__ import annotations

from dataclasses import asdict, dataclass
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
from knowledge_hub.domain.vault_knowledge.evidence_policy import policy_key_for_family
from knowledge_hub.domain.vault_knowledge.families import (
    VAULT_FAMILY_NOTE_LOOKUP,
    VAULT_FAMILY_VAULT_COMPARE,
    VAULT_FAMILY_VAULT_EXPLAINER,
    VAULT_FAMILY_VAULT_TIMELINE,
    classify_vault_family,
    explicit_vault_scope,
)


_DOMAIN_KEY = "vault_knowledge"
_TOKEN_RE = re.compile(r"[A-Za-z0-9._/\-]+|[가-힣]+")
_TEMPORAL_RE = re.compile(r"\b(latest|recent|updated|newest|before|after|since|changed?)\b|최근|최신|업데이트|이전|이후|변경", re.IGNORECASE)
_DEFINITION_RE = re.compile(r"\b(what is|define|definition|meaning|concept|core idea|principle|intuition|explain)\b|무엇|뭐야|정의|개념|원리|핵심|설명", re.IGNORECASE)
_IMPL_RE = re.compile(r"\b(implementation|implement|pipeline|architecture|howto|guide|steps?)\b|구현|파이프라인|아키텍처|방법|가이드|단계", re.IGNORECASE)
_RELATION_RE = re.compile(r"\b(relationship|related|dependency|link|updates?)\b|관계|연결|의존|업데이트", re.IGNORECASE)
_SUMMARY_RE = re.compile(r"\b(cause|reason|problem|issue|trade[- ]?off|pros?|cons?)\b|원인|이유|문제|쟁점|트레이드오프|장점|단점", re.IGNORECASE)
_ENTITY_STOPWORDS = {
    "vault",
    "note",
    "notes",
    "page",
    "document",
    "doc",
    "md",
    "markdown",
    "설명",
    "정의",
    "요약",
    "정리",
    "무엇",
    "뭐야",
    "차이",
    "비교",
    "최근",
    "최신",
    "업데이트",
    "이유",
    "무엇인가",
    "의미하나",
    "가장",
    "한",
    "문장",
    "으로",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에서",
}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _dedupe(values: list[Any], *, limit: int | None = None) -> list[str]:
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
    result: list[str] = []
    for raw in _TOKEN_RE.findall(_clean_text(query)):
        token = _clean_text(raw)
        lowered = token.casefold()
        if not token or lowered in _ENTITY_STOPWORDS:
            continue
        if token.endswith(".md") or token.endswith(".markdown"):
            continue
        if re.fullmatch(r"[A-Za-z0-9._/\-]+", token) and len(token) < 3:
            continue
        result.append(token)
    return _dedupe(result, limit=6)


def _parse_note_metadata(row: dict[str, Any] | None) -> dict[str, Any]:
    raw = (row or {}).get("metadata")
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw or "{}")
    except Exception:
        parsed = {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _vault_note_rows(sqlite_db: Any | None) -> list[dict[str, Any]]:
    if not sqlite_db:
        return []
    return [dict(row or {}) for row in list(sqlite_db.list_notes(source_type="vault", limit=2000) or [])]


def _scope_from_note_row(row: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    note_id = _clean_text(row.get("id"))
    file_path = _clean_text(row.get("file_path"))
    title = _clean_text(row.get("title"))
    return _dedupe([note_id, file_path], limit=2), _dedupe([title, file_path], limit=3), {
        "note_id": note_id,
        "file_path": file_path,
        "title": title,
    }


def _resolve_vault_scope(
    query: str,
    *,
    sqlite_db: Any | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    scoped = _clean_text(explicit_vault_scope(query, metadata_filter=metadata_filter))
    query_body = _clean_text(query).casefold()
    if scoped:
        scoped_lower = scoped.casefold()
        if scoped.startswith("vault:"):
            note = dict((sqlite_db.get_note(scoped) if sqlite_db else {}) or {})
            if sqlite_db is not None and not note:
                return _dedupe([scoped], limit=1), _dedupe([scoped], limit=1), {
                    "note_id": scoped,
                    "scope_missing": True,
                }
            if not note:
                return _dedupe([scoped], limit=1), _dedupe([scoped], limit=1), {
                    "note_id": scoped,
                }
            return _scope_from_note_row(note)
        for row in _vault_note_rows(sqlite_db):
            note_id = _clean_text(row.get("id"))
            file_path = _clean_text(row.get("file_path"))
            if scoped_lower in {note_id.casefold(), file_path.casefold()}:
                return _scope_from_note_row(row)
        for row in _vault_note_rows(sqlite_db):
            file_path = _clean_text(row.get("file_path"))
            if file_path and file_path.casefold() in query_body:
                return _scope_from_note_row(row)
        scope_metadata: dict[str, Any] = {"file_path": scoped}
        if sqlite_db is not None:
            scope_metadata["scope_missing"] = True
        return _dedupe([scoped], limit=1), _dedupe([scoped], limit=1), scope_metadata
    best_note: dict[str, Any] | None = None
    best_score = 0.0
    for row in _vault_note_rows(sqlite_db):
        metadata = _parse_note_metadata(row)
        title = _clean_text(row.get("title"))
        file_path = _clean_text(row.get("file_path"))
        aliases = metadata.get("aliases") if isinstance(metadata.get("aliases"), list) else []
        tags = metadata.get("tags") if isinstance(metadata.get("tags"), list) else []
        haystack = " ".join(
            [
                title,
                file_path,
                " ".join(str(item or "") for item in aliases[:8]),
                " ".join(str(item or "") for item in tags[:8]),
            ]
        ).casefold()
        score = 0.0
        for term in _query_terms(query)[:4]:
            if term.casefold() in haystack:
                score += 1.0
        if title and title.casefold() in _clean_text(query).casefold():
            score += 2.5
        if score > best_score and score >= 2.0:
            best_score = score
            best_note = row
    if not best_note:
        return [], [], {}
    metadata = _parse_note_metadata(best_note)
    note_id = _clean_text(best_note.get("id"))
    file_path = _clean_text(best_note.get("file_path"))
    title = _clean_text(best_note.get("title"))
    scoped_terms: list[Any] = [title, file_path]
    if isinstance(metadata.get("aliases"), list):
        scoped_terms.extend(list(metadata.get("aliases") or [])[:3])
    if isinstance(metadata.get("tags"), list):
        scoped_terms.extend(list(metadata.get("tags") or [])[:3])
    return _dedupe([note_id, file_path], limit=2), _dedupe(scoped_terms, limit=6), {
        "note_id": note_id,
        "file_path": file_path,
        "title": title,
    }


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
        return _dedupe(resolved, limit=6)
    except Exception:
        return []


def _query_intent(query: str, family: str) -> str:
    body = _clean_text(query)
    if family == VAULT_FAMILY_VAULT_TIMELINE:
        return "temporal"
    if family == VAULT_FAMILY_VAULT_COMPARE:
        return "comparison"
    if family == VAULT_FAMILY_NOTE_LOOKUP:
        return "note_lookup"
    if _IMPL_RE.search(body):
        return "implementation"
    if _RELATION_RE.search(body):
        return "relation"
    return "definition"


def _answer_mode(query: str, family: str, intent: str, *, scope_metadata: dict[str, Any] | None = None) -> str:
    body = _clean_text(query)
    if bool((scope_metadata or {}).get("scope_missing")):
        return "abstain"
    if family == VAULT_FAMILY_VAULT_TIMELINE:
        return "timeline_compare"
    if family == VAULT_FAMILY_VAULT_COMPARE:
        return "compare_summary"
    if family == VAULT_FAMILY_NOTE_LOOKUP:
        return "note_scoped_answer"
    if family == VAULT_FAMILY_VAULT_EXPLAINER and _SUMMARY_RE.search(body):
        return "concise_summary"
    if _DEFINITION_RE.search(body):
        return "concept_explainer"
    if intent == "implementation":
        return "implementation_steps"
    return "concise_summary"


def _expanded_terms(query: str, family: str, scoped_terms: list[str]) -> list[str]:
    result = list(scoped_terms)
    if family == VAULT_FAMILY_VAULT_TIMELINE:
        result.extend(["timeline", "updated", "event_date", "document_date"])
    elif family == VAULT_FAMILY_VAULT_COMPARE:
        result.extend(["compare", "difference"])
    elif family == VAULT_FAMILY_VAULT_EXPLAINER:
        result.extend(["overview", "summary"])
    return _dedupe([*result, *_query_terms(query)], limit=8)


def _metadata_filter(
    *,
    family: str,
    metadata_filter: dict[str, Any] | None,
    scope_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = dict(metadata_filter or {})
    payload["source_type"] = "vault"
    scoped = dict(scope_metadata or {})
    for key in ("note_id", "file_path"):
        token = _clean_text(scoped.get(key))
        if token:
            payload[key] = token
    if bool(scoped.get("scope_missing")):
        payload["vault_scope_missing"] = True
        payload["note_scope_required"] = True
    if family == VAULT_FAMILY_VAULT_TIMELINE:
        payload["timeline_required"] = True
    if family == VAULT_FAMILY_NOTE_LOOKUP:
        payload["note_scope_required"] = True
    return payload


def _confidence(family: str, *, resolved_source_ids: list[str], canonical_entity_ids: list[str]) -> float:
    base = 0.62 if family == VAULT_FAMILY_VAULT_EXPLAINER else 0.68
    if family == VAULT_FAMILY_NOTE_LOOKUP and resolved_source_ids:
        base += 0.22
    if family == VAULT_FAMILY_VAULT_COMPARE:
        base += 0.08
    if family == VAULT_FAMILY_VAULT_TIMELINE:
        base += 0.08
    if canonical_entity_ids:
        base += 0.04
    return max(0.0, min(0.98, base))


def build_rule_based_query_frame(
    query: str,
    *,
    source_type: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    sqlite_db: Any | None = None,
) -> NormalizedQueryFrame:
    normalized_source = normalize_domain_source(source_type) or "vault"
    family = classify_vault_family(query, source_type=normalized_source, metadata_filter=metadata_filter) or VAULT_FAMILY_VAULT_EXPLAINER
    resolved_source_ids, scoped_terms, scope_metadata = _resolve_vault_scope(
        query,
        sqlite_db=sqlite_db,
        metadata_filter=metadata_filter,
    )
    canonical_entity_ids = _canonical_entity_ids(query, sqlite_db=sqlite_db)
    intent = _query_intent(query, family)
    answer_mode = _answer_mode(query, family, intent, scope_metadata=scope_metadata)
    return build_query_frame(
        domain_key=_DOMAIN_KEY,
        source_type=normalized_source,
        family=family,
        query_intent=intent,
        answer_mode=answer_mode,
        entities=_query_terms(query),
        canonical_entity_ids=canonical_entity_ids,
        expanded_terms=_expanded_terms(query, family, scoped_terms),
        resolved_source_ids=resolved_source_ids,
        confidence=_confidence(family, resolved_source_ids=resolved_source_ids, canonical_entity_ids=canonical_entity_ids),
        planner_status="not_attempted",
        planner_reason="rule_based",
        evidence_policy_key=policy_key_for_family(family),
        metadata_filter=_metadata_filter(family=family, metadata_filter=metadata_filter, scope_metadata=scope_metadata),
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
        family=family or VAULT_FAMILY_VAULT_EXPLAINER,
        query_intent=query_intent or "definition",
        answer_mode=answer_mode or "concise_summary",
        entities=_dedupe([*list(payload.get("entities") or []), *list(base_frame.entities or [])], limit=6),
        canonical_entity_ids=list(payload.get("canonical_entity_ids") or base_frame.canonical_entity_ids),
        expanded_terms=_dedupe([*list(payload.get("expanded_terms") or []), *list(base_frame.expanded_terms or [])], limit=8),
        resolved_source_ids=_dedupe([*list(payload.get("resolved_source_ids") or []), *list(base_frame.resolved_source_ids or [])], limit=6),
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
class VaultQueryPlan:
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
) -> VaultQueryPlan:
    frame = build_rule_based_query_frame(
        query,
        source_type=source_type,
        metadata_filter=metadata_filter,
        sqlite_db=sqlite_db,
    )
    return VaultQueryPlan(
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
    "VAULT_FAMILY_NOTE_LOOKUP",
    "VAULT_FAMILY_VAULT_COMPARE",
    "VAULT_FAMILY_VAULT_EXPLAINER",
    "VAULT_FAMILY_VAULT_TIMELINE",
    "VaultQueryPlan",
    "build_rule_based_query_frame",
    "build_rule_query_plan",
    "query_frame_from_query_plan",
]
