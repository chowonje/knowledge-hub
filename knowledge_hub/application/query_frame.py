from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


QUERY_FRAME_PROVENANCE_DERIVED = "derived"
QUERY_FRAME_PROVENANCE_EXPLICIT = "explicit"
QUERY_FRAME_FAMILY_FROM_PACK = "from_pack"
QUERY_FRAME_FAMILY_FROM_FRAME = "from_frame"
_LOCKABLE_QUERY_FRAME_FIELDS = ("source_type", "family", "query_intent", "answer_mode")
_DOMAIN_KEY_BY_SOURCE = {
    "paper": "ai_papers",
    "web": "web_knowledge",
    "vault": "vault_knowledge",
    "youtube": "youtube_knowledge",
}
_FAMILY_VALUES_BY_SOURCE = {
    "paper": {"concept_explainer", "paper_lookup", "paper_compare", "paper_discover"},
    "web": {"reference_explainer", "temporal_update", "relation_explainer", "source_disambiguation"},
    "vault": {"note_lookup", "vault_explainer", "vault_compare", "vault_timeline"},
    "youtube": {"video_lookup", "video_explainer", "section_lookup", "timestamp_lookup"},
}
_SUPPORTED_QUERY_INTENTS_BY_FAMILY = {
    ("paper", "paper_compare"): {"comparison"},
    ("paper", "concept_explainer"): {"definition"},
    ("web", "reference_explainer"): {"definition", "implementation"},
    ("web", "relation_explainer"): {"relation"},
    ("web", "source_disambiguation"): {"disambiguation"},
    ("web", "temporal_update"): {"abstention", "temporal"},
    ("vault", "note_lookup"): {"note_lookup"},
    ("vault", "vault_compare"): {"comparison"},
    ("vault", "vault_explainer"): {"definition", "implementation", "relation"},
    ("vault", "vault_timeline"): {"temporal"},
    ("youtube", "section_lookup"): {"section_lookup"},
    ("youtube", "timestamp_lookup"): {"temporal"},
    ("youtube", "video_explainer"): {"definition"},
    ("youtube", "video_lookup"): {"video_lookup"},
}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_source_type(value: Any) -> str:
    source = _clean_text(value).lower()
    if source in {"", "all", "*"}:
        return ""
    if source == "note":
        return "vault"
    if source in {"repo", "repository", "workspace"}:
        return "project"
    return source


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


def normalize_frame_provenance(value: Any) -> str:
    token = _clean_text(value).lower()
    if token == QUERY_FRAME_PROVENANCE_EXPLICIT:
        return QUERY_FRAME_PROVENANCE_EXPLICIT
    return QUERY_FRAME_PROVENANCE_DERIVED


def normalize_family_source(value: Any) -> str:
    token = _clean_text(value).lower()
    if token == QUERY_FRAME_FAMILY_FROM_FRAME:
        return QUERY_FRAME_FAMILY_FROM_FRAME
    return QUERY_FRAME_FAMILY_FROM_PACK


def derive_domain_key(source_type: Any, fallback_domain_key: Any = "") -> str:
    normalized_source = _normalize_source_type(source_type)
    if normalized_source:
        resolved = _clean_text(_DOMAIN_KEY_BY_SOURCE.get(normalized_source))
        if resolved:
            return resolved
    return _clean_text(fallback_domain_key)


def default_query_frame_lock_mask(*, frame_provenance: str, trusted: bool) -> tuple[str, ...]:
    if frame_provenance == QUERY_FRAME_PROVENANCE_EXPLICIT and trusted:
        return _LOCKABLE_QUERY_FRAME_FIELDS
    return ()


def normalize_query_frame_lock_mask(
    values: Any,
    *,
    frame_provenance: str,
    trusted: bool,
) -> tuple[str, ...]:
    if values is None:
        return default_query_frame_lock_mask(frame_provenance=frame_provenance, trusted=trusted)
    allowed = set(_LOCKABLE_QUERY_FRAME_FIELDS)
    normalized = [
        token
        for token in _dedupe_lines(list(values or []), limit=len(_LOCKABLE_QUERY_FRAME_FIELDS))
        if token in allowed
    ]
    if normalized:
        return tuple(normalized)
    return default_query_frame_lock_mask(frame_provenance=frame_provenance, trusted=trusted)


def query_frame_is_authoritative(value: Any) -> bool:
    payload = normalize_query_frame_dict(value)
    return bool(payload) and str(payload.get("frame_provenance")) == QUERY_FRAME_PROVENANCE_EXPLICIT and bool(payload.get("trusted"))


def query_frame_lock_mask(value: Any) -> tuple[str, ...]:
    payload = normalize_query_frame_dict(value)
    return normalize_query_frame_lock_mask(
        payload.get("lock_mask"),
        frame_provenance=str(payload.get("frame_provenance") or QUERY_FRAME_PROVENANCE_DERIVED),
        trusted=bool(payload.get("trusted")),
    )


def family_values_for_source(source_type: Any) -> set[str]:
    return set(_FAMILY_VALUES_BY_SOURCE.get(_normalize_source_type(source_type), set()))


def family_supported_for_source(source_type: Any, family: Any) -> bool:
    normalized_family = _clean_text(family).lower()
    if not normalized_family:
        return False
    allowed = family_values_for_source(source_type)
    if not allowed:
        return True
    return normalized_family in allowed


def query_intent_supported_for_family(source_type: Any, family: Any, query_intent: Any) -> bool:
    normalized_source = _normalize_source_type(source_type)
    normalized_family = _clean_text(family).lower()
    normalized_intent = _clean_text(query_intent)
    if not normalized_intent:
        return True
    allowed = _SUPPORTED_QUERY_INTENTS_BY_FAMILY.get((normalized_source, normalized_family))
    if not allowed:
        return True
    return normalized_intent in allowed


@dataclass(frozen=True)
class NormalizedQueryFrame:
    domain_key: str
    source_type: str
    family: str
    query_intent: str
    answer_mode: str
    entities: tuple[str, ...]
    canonical_entity_ids: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    resolved_source_ids: tuple[str, ...]
    confidence: float
    planner_status: str
    planner_reason: str
    evidence_policy_key: str
    metadata_filter: dict[str, Any]
    frame_provenance: str = QUERY_FRAME_PROVENANCE_DERIVED
    trusted: bool = False
    lock_mask: tuple[str, ...] = ()
    family_source: str = QUERY_FRAME_FAMILY_FROM_PACK
    overrides_applied: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["entities"] = list(self.entities)
        payload["canonical_entity_ids"] = list(self.canonical_entity_ids)
        payload["expanded_terms"] = list(self.expanded_terms)
        payload["resolved_source_ids"] = list(self.resolved_source_ids)
        payload["lock_mask"] = list(self.lock_mask)
        payload["overrides_applied"] = list(self.overrides_applied)
        payload["metadata_filter"] = dict(self.metadata_filter or {})
        payload["frameProvenance"] = self.frame_provenance
        payload["frameTrusted"] = self.trusted
        payload["lockMask"] = list(self.lock_mask)
        payload["familySource"] = self.family_source
        payload["overridesApplied"] = list(self.overrides_applied)
        return payload

    def to_query_plan_dict(self) -> dict[str, Any]:
        planner_used = str(self.planner_status or "").strip().lower() == "used"
        payload = {
            "family": _clean_text(self.family).lower(),
            "entities": list(self.entities),
            "expanded_terms": list(self.expanded_terms),
            "resolved_paper_ids": list(self.resolved_source_ids),
            "answer_mode": _clean_text(self.answer_mode),
            "confidence": max(0.0, min(1.0, float(self.confidence or 0.0))),
            "planner_used": planner_used,
            "query_intent": _clean_text(self.query_intent),
            "planner_reason": _clean_text(self.planner_reason or "rule_based"),
            "planner_status": _clean_text(self.planner_status or "not_attempted"),
            "planner_warnings": [],
            "planner_route": {},
            "evidence_policy_key": _clean_text(self.evidence_policy_key),
            "frame_provenance": self.frame_provenance,
            "trusted": self.trusted,
            "lock_mask": list(self.lock_mask),
            "family_source": self.family_source,
            "overrides_applied": list(self.overrides_applied),
        }
        payload["answerMode"] = payload["answer_mode"]
        payload["queryIntent"] = payload["query_intent"]
        payload["plannerUsed"] = payload["planner_used"]
        payload["plannerReason"] = payload["planner_reason"]
        payload["plannerStatus"] = payload["planner_status"]
        payload["plannerWarnings"] = []
        payload["plannerRoute"] = {}
        payload["expandedTerms"] = list(payload["expanded_terms"])
        payload["resolvedPaperIds"] = list(payload["resolved_paper_ids"])
        payload["evidencePolicyKey"] = payload["evidence_policy_key"]
        payload["frameProvenance"] = payload["frame_provenance"]
        payload["frameTrusted"] = payload["trusted"]
        payload["lockMask"] = list(payload["lock_mask"])
        payload["familySource"] = payload["family_source"]
        payload["overridesApplied"] = list(payload["overrides_applied"])
        return payload


def normalize_query_frame_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, NormalizedQueryFrame):
        return value.to_dict()
    if not isinstance(value, dict):
        return {}
    frame_provenance = normalize_frame_provenance(value.get("frame_provenance") or value.get("frameProvenance"))
    trusted = bool(value.get("trusted") or value.get("frameTrusted"))
    source_type = _normalize_source_type(value.get("source_type") or value.get("sourceType"))
    payload = {
        "source_type": source_type,
        "family": _clean_text(value.get("family")).lower(),
        "query_intent": _clean_text(value.get("query_intent") or value.get("queryIntent")),
        "answer_mode": _clean_text(value.get("answer_mode") or value.get("answerMode")),
        "entities": _dedupe_lines(list(value.get("entities") or []), limit=6),
        "canonical_entity_ids": _dedupe_lines(list(value.get("canonical_entity_ids") or value.get("canonicalEntityIds") or []), limit=6),
        "expanded_terms": _dedupe_lines(list(value.get("expanded_terms") or value.get("expandedTerms") or []), limit=6),
        "resolved_source_ids": _dedupe_lines(
            list(value.get("resolved_source_ids") or value.get("resolvedSourceIds") or value.get("resolved_paper_ids") or value.get("resolvedPaperIds") or []),
            limit=6,
        ),
        "confidence": max(0.0, min(1.0, float(value.get("confidence") or 0.0))),
        "planner_status": _clean_text(value.get("planner_status") or value.get("plannerStatus") or "not_attempted"),
        "planner_reason": _clean_text(value.get("planner_reason") or value.get("plannerReason") or "rule_based"),
        "evidence_policy_key": _clean_text(value.get("evidence_policy_key") or value.get("evidencePolicyKey")),
        "metadata_filter": dict(value.get("metadata_filter") or value.get("metadataFilter") or {}),
        "frame_provenance": frame_provenance,
        "trusted": trusted,
        "family_source": normalize_family_source(value.get("family_source") or value.get("familySource")),
        "overrides_applied": _dedupe_lines(list(value.get("overrides_applied") or value.get("overridesApplied") or []), limit=8),
    }
    payload["domain_key"] = derive_domain_key(source_type, value.get("domain_key") or value.get("domainKey"))
    payload["lock_mask"] = list(
        normalize_query_frame_lock_mask(
            value.get("lock_mask") or value.get("lockMask"),
            frame_provenance=frame_provenance,
            trusted=trusted,
        )
    )
    return payload


def build_query_frame(
    *,
    domain_key: str,
    source_type: str,
    family: str,
    query_intent: str,
    answer_mode: str,
    entities: list[Any] | tuple[Any, ...] | None = None,
    canonical_entity_ids: list[Any] | tuple[Any, ...] | None = None,
    expanded_terms: list[Any] | tuple[Any, ...] | None = None,
    resolved_source_ids: list[Any] | tuple[Any, ...] | None = None,
    confidence: float = 0.0,
    planner_status: str = "not_attempted",
    planner_reason: str = "rule_based",
    evidence_policy_key: str = "",
    metadata_filter: dict[str, Any] | None = None,
    frame_provenance: str = QUERY_FRAME_PROVENANCE_DERIVED,
    trusted: bool = False,
    lock_mask: list[Any] | tuple[Any, ...] | None = None,
    family_source: str = QUERY_FRAME_FAMILY_FROM_PACK,
    overrides_applied: list[Any] | tuple[Any, ...] | None = None,
) -> NormalizedQueryFrame:
    normalized_source = _normalize_source_type(source_type)
    normalized_provenance = normalize_frame_provenance(frame_provenance)
    normalized_trusted = bool(trusted)
    return NormalizedQueryFrame(
        domain_key=derive_domain_key(normalized_source, domain_key),
        source_type=normalized_source,
        family=_clean_text(family).lower(),
        query_intent=_clean_text(query_intent),
        answer_mode=_clean_text(answer_mode),
        entities=tuple(_dedupe_lines(list(entities or []), limit=6)),
        canonical_entity_ids=tuple(_dedupe_lines(list(canonical_entity_ids or []), limit=6)),
        expanded_terms=tuple(_dedupe_lines(list(expanded_terms or []), limit=6)),
        resolved_source_ids=tuple(_dedupe_lines(list(resolved_source_ids or []), limit=6)),
        confidence=max(0.0, min(1.0, float(confidence or 0.0))),
        planner_status=_clean_text(planner_status or "not_attempted"),
        planner_reason=_clean_text(planner_reason or "rule_based"),
        evidence_policy_key=_clean_text(evidence_policy_key),
        metadata_filter=dict(metadata_filter or {}),
        frame_provenance=normalized_provenance,
        trusted=normalized_trusted,
        lock_mask=normalize_query_frame_lock_mask(
            lock_mask,
            frame_provenance=normalized_provenance,
            trusted=normalized_trusted,
        ),
        family_source=normalize_family_source(family_source),
        overrides_applied=tuple(_dedupe_lines(list(overrides_applied or []), limit=8)),
    )


__all__ = [
    "NormalizedQueryFrame",
    "QUERY_FRAME_FAMILY_FROM_FRAME",
    "QUERY_FRAME_FAMILY_FROM_PACK",
    "QUERY_FRAME_PROVENANCE_DERIVED",
    "QUERY_FRAME_PROVENANCE_EXPLICIT",
    "build_query_frame",
    "default_query_frame_lock_mask",
    "derive_domain_key",
    "family_supported_for_source",
    "family_values_for_source",
    "normalize_family_source",
    "normalize_frame_provenance",
    "normalize_query_frame_dict",
    "normalize_query_frame_lock_mask",
    "query_frame_is_authoritative",
    "query_frame_lock_mask",
    "query_intent_supported_for_family",
]
