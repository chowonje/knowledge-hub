"""Entity resolution helpers for crawl post-processing."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlparse

from knowledge_hub.learning.resolver import normalize_term


@dataclass
class _ConceptCandidate:
    entity_id: str
    entity_type: str
    canonical_name: str
    aliases: list[str]
    normalized_values: tuple[str, ...]


@dataclass
class _EntityContext:
    entity_id: str
    entity_type: str
    source: str
    claim_signatures: frozenset[str]
    claim_predicates: frozenset[str]
    claim_labels: frozenset[str]
    relation_signatures: frozenset[str]
    relation_predicates: frozenset[str]
    relation_labels: frozenset[str]
    counterpart_labels: frozenset[str]
    note_ids: frozenset[str]
    domains: frozenset[str]
    topics: frozenset[str]
    provenance_sources: frozenset[str]
    alias_divergence: int


def _load_alias_map(db) -> dict[str, list[str]]:
    alias_map: dict[str, list[str]] = defaultdict(list)
    conn = getattr(db, "conn", None)
    if conn is None:
        return alias_map
    try:
        rows = conn.execute("SELECT entity_id, alias FROM entity_aliases").fetchall()
    except Exception:
        return alias_map
    for row in rows:
        entity_id = str(row["entity_id"] or "").strip()
        alias = str(row["alias"] or "").strip()
        if entity_id and alias:
            alias_map[entity_id].append(alias)
    return alias_map


def _load_concept_candidates(db, *, limit: int = 5000) -> dict[str, _ConceptCandidate]:
    alias_map = _load_alias_map(db)
    concepts = db.list_ontology_entities(entity_type="concept", limit=max(1, int(limit)))
    candidates: dict[str, _ConceptCandidate] = {}
    for concept in concepts:
        entity_id = str(concept.get("entity_id") or "").strip()
        entity_type = str(concept.get("entity_type") or "concept").strip() or "concept"
        canonical_name = str(concept.get("canonical_name") or "").strip()
        if not entity_id or not canonical_name:
            continue
        aliases = sorted({str(alias).strip() for alias in alias_map.get(entity_id, []) if str(alias).strip()})
        normalized_values = {
            normalize_term(canonical_name),
            *(normalize_term(alias) for alias in aliases),
        }
        normalized_values.discard("")
        candidates[entity_id] = _ConceptCandidate(
            entity_id=entity_id,
            entity_type=entity_type,
            canonical_name=canonical_name,
            aliases=aliases,
            normalized_values=tuple(sorted(normalized_values)),
        )
    return candidates


def _similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    best = 0.0
    for source_value in left:
        if not source_value:
            continue
        for target_value in right:
            if not target_value:
                continue
            best = max(best, SequenceMatcher(None, source_value, target_value).ratio())
    return best


def _safe_json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_domain(value: str) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    if "://" not in token:
        token = f"https://{token}"
    try:
        parsed = urlparse(token)
    except Exception:
        return ""
    domain = str(parsed.netloc or "").strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _normalized_token_set(values: tuple[str, ...]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        parts = [part.strip() for part in str(value or "").split()]
        for part in parts:
            if len(part) >= 3 or part.isdigit():
                tokens.add(part)
    return tokens


def _token_overlap(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _collapsed_values(values: tuple[str, ...]) -> set[str]:
    return {value.replace(" ", "") for value in values if value}


def _load_note_metadata(db, note_id: str, note_cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    token = str(note_id or "").strip()
    if not token:
        return {}
    if token not in note_cache:
        note_cache[token] = db.get_note(token) or {}
    note = note_cache.get(token) or {}
    return _safe_json_dict(note.get("metadata"))


def _collect_note_provenance(
    db,
    note_id: str,
    *,
    note_cache: dict[str, dict[str, Any]],
    note_ids: set[str],
    domains: set[str],
    topics: set[str],
) -> None:
    token = str(note_id or "").strip()
    if not token:
        return
    note_ids.add(token)
    metadata = _load_note_metadata(db, token, note_cache)
    url = str(
        metadata.get("canonical_url")
        or metadata.get("source_url")
        or metadata.get("url")
        or ""
    ).strip()
    domain = _extract_domain(url)
    if domain:
        domains.add(domain)
    topic = normalize_term(str(metadata.get("topic_slug") or metadata.get("topic") or "").strip())
    if topic:
        topics.add(topic)


def _collect_pointer_provenance(
    db,
    pointers: list[Any],
    *,
    note_cache: dict[str, dict[str, Any]],
    note_ids: set[str],
    domains: set[str],
    topics: set[str],
) -> None:
    for pointer in pointers:
        if not isinstance(pointer, dict):
            continue
        note_id = str(pointer.get("note_id") or "").strip()
        if note_id:
            _collect_note_provenance(
                db,
                note_id,
                note_cache=note_cache,
                note_ids=note_ids,
                domains=domains,
                topics=topics,
            )
        domain = _extract_domain(str(pointer.get("source_url") or pointer.get("url") or "").strip())
        if domain:
            domains.add(domain)
        topic = normalize_term(str(pointer.get("topic_slug") or pointer.get("topic") or "").strip())
        if topic:
            topics.add(topic)


def _resolve_entity_label(db, entity_id: str, entity_name_cache: dict[str, str]) -> str:
    token = str(entity_id or "").strip()
    if not token:
        return ""
    if token not in entity_name_cache:
        entity = db.get_ontology_entity(token) or {}
        entity_name_cache[token] = normalize_term(str(entity.get("canonical_name") or token))
    return entity_name_cache[token]


def _claim_signature(db, entity_id: str, claim: dict[str, Any], entity_name_cache: dict[str, str]) -> tuple[str, str, str]:
    predicate = str(claim.get("predicate") or "").strip()
    if not predicate:
        return "", "", ""
    subject_entity_id = str(claim.get("subject_entity_id") or "").strip()
    object_entity_id = str(claim.get("object_entity_id") or "").strip()
    object_literal = normalize_term(str(claim.get("object_literal") or "").strip())

    direction = ""
    counterpart = ""
    if subject_entity_id == entity_id:
        direction = "out"
        counterpart = _resolve_entity_label(db, object_entity_id, entity_name_cache) or object_literal
    elif object_entity_id == entity_id:
        direction = "in"
        counterpart = _resolve_entity_label(db, subject_entity_id, entity_name_cache)
    if not direction or not counterpart:
        return "", predicate, counterpart
    return f"{direction}:{predicate}:{counterpart}", predicate, counterpart


def _relation_signature(
    db,
    entity_id: str,
    relation: dict[str, Any],
    entity_name_cache: dict[str, str],
) -> tuple[str, str, str]:
    predicate = str(relation.get("predicate_id") or relation.get("relation") or "").strip()
    if not predicate or predicate == "mentions":
        return "", "", ""

    source_entity_id = str(relation.get("source_entity_id") or relation.get("source_id") or "").strip()
    target_entity_id = str(relation.get("target_entity_id") or relation.get("target_id") or "").strip()
    source_type = str(relation.get("source_type") or "").strip()
    target_type = str(relation.get("target_type") or "").strip()

    direction = ""
    counterpart = ""
    if source_entity_id == entity_id:
        if target_type == "note":
            return "", predicate, ""
        direction = "out"
        counterpart = _resolve_entity_label(db, target_entity_id, entity_name_cache)
    elif target_entity_id == entity_id:
        if source_type == "note":
            return "", predicate, ""
        direction = "in"
        counterpart = _resolve_entity_label(db, source_entity_id, entity_name_cache)
    if not direction or not counterpart:
        return "", predicate, counterpart
    return f"{direction}:{predicate}:{counterpart}", predicate, counterpart


def _alias_divergence(candidate: _ConceptCandidate) -> int:
    canonical = normalize_term(candidate.canonical_name)
    if not canonical:
        return 0
    canonical_collapsed = canonical.replace(" ", "")
    divergent = 0
    for value in candidate.normalized_values:
        if not value or value == canonical:
            continue
        collapsed = value.replace(" ", "")
        if collapsed == canonical_collapsed:
            continue
        if _token_overlap(_normalized_token_set((canonical,)), _normalized_token_set((value,))) >= 0.5:
            continue
        if SequenceMatcher(None, canonical, value).ratio() >= 0.9:
            continue
        divergent += 1
    return divergent


def _build_entity_context(
    db,
    candidate: _ConceptCandidate,
    *,
    context_cache: dict[str, _EntityContext],
    note_cache: dict[str, dict[str, Any]],
    entity_name_cache: dict[str, str],
) -> _EntityContext:
    cached = context_cache.get(candidate.entity_id)
    if cached is not None:
        return cached

    entity = db.get_ontology_entity(candidate.entity_id) or {}
    entity_type = str(entity.get("entity_type") or candidate.entity_type or "concept").strip() or "concept"
    claims = db.list_claims_by_entity(candidate.entity_id, limit=80)
    relations = db.get_relations(entity_type, candidate.entity_id)

    claim_signatures: set[str] = set()
    claim_predicates: set[str] = set()
    claim_labels: set[str] = set()
    relation_signatures: set[str] = set()
    relation_predicates: set[str] = set()
    relation_labels: set[str] = set()
    counterpart_labels: set[str] = set()
    note_ids: set[str] = set()
    domains: set[str] = set()
    topics: set[str] = set()
    provenance_sources: set[str] = set()

    source_value = str(entity.get("source") or "").strip()
    if source_value:
        provenance_sources.add(source_value)

    for claim in claims:
        signature, predicate, counterpart = _claim_signature(db, candidate.entity_id, claim, entity_name_cache)
        if signature:
            claim_signatures.add(signature)
        if predicate:
            claim_predicates.add(predicate)
        if counterpart:
            claim_labels.add(counterpart)
            counterpart_labels.add(counterpart)
        source_value = str(claim.get("source") or "").strip()
        if source_value:
            provenance_sources.add(source_value)
        evidence_ptrs = claim.get("evidence_ptrs") if isinstance(claim.get("evidence_ptrs"), list) else []
        _collect_pointer_provenance(
            db,
            evidence_ptrs,
            note_cache=note_cache,
            note_ids=note_ids,
            domains=domains,
            topics=topics,
        )

    for relation in relations:
        evidence_json = relation.get("evidence_json") if isinstance(relation.get("evidence_json"), dict) else {}
        evidence_ptrs = evidence_json.get("evidence_ptrs") if isinstance(evidence_json.get("evidence_ptrs"), list) else []
        _collect_pointer_provenance(
            db,
            evidence_ptrs,
            note_cache=note_cache,
            note_ids=note_ids,
            domains=domains,
            topics=topics,
        )
        source_value = str(evidence_json.get("source") or relation.get("source") or "").strip()
        if source_value:
            provenance_sources.add(source_value)
        if str(relation.get("source_type") or "").strip() == "note":
            _collect_note_provenance(
                db,
                str(relation.get("source_id") or "").strip(),
                note_cache=note_cache,
                note_ids=note_ids,
                domains=domains,
                topics=topics,
            )
        if str(relation.get("target_type") or "").strip() == "note":
            _collect_note_provenance(
                db,
                str(relation.get("target_id") or "").strip(),
                note_cache=note_cache,
                note_ids=note_ids,
                domains=domains,
                topics=topics,
            )

        signature, predicate, counterpart = _relation_signature(db, candidate.entity_id, relation, entity_name_cache)
        if signature:
            relation_signatures.add(signature)
        if predicate:
            relation_predicates.add(predicate)
        if counterpart:
            relation_labels.add(counterpart)
            counterpart_labels.add(counterpart)

    context = _EntityContext(
        entity_id=candidate.entity_id,
        entity_type=entity_type,
        source=str(entity.get("source") or "").strip(),
        claim_signatures=frozenset(claim_signatures),
        claim_predicates=frozenset(claim_predicates),
        claim_labels=frozenset(claim_labels),
        relation_signatures=frozenset(relation_signatures),
        relation_predicates=frozenset(relation_predicates),
        relation_labels=frozenset(relation_labels),
        counterpart_labels=frozenset(counterpart_labels),
        note_ids=frozenset(note_ids),
        domains=frozenset(domains),
        topics=frozenset(topics),
        provenance_sources=frozenset(provenance_sources),
        alias_divergence=_alias_divergence(candidate),
    )
    context_cache[candidate.entity_id] = context
    return context


def _overlap_strength(
    left_signatures: frozenset[str],
    right_signatures: frozenset[str],
    left_predicates: frozenset[str],
    right_predicates: frozenset[str],
    left_labels: frozenset[str],
    right_labels: frozenset[str],
) -> dict[str, Any]:
    shared_signatures = sorted(left_signatures & right_signatures)
    shared_predicates = sorted(left_predicates & right_predicates)
    shared_labels = sorted(left_labels & right_labels)
    left_count = len(left_signatures)
    right_count = len(right_signatures)

    strength = "sparse"
    if shared_signatures or (shared_predicates and shared_labels):
        strength = "strong"
    elif min(left_count, right_count) >= 2 and (shared_predicates or shared_labels):
        strength = "moderate"
    elif min(left_count, right_count) >= 2:
        strength = "weak"

    return {
        "strength": strength,
        "shared_signatures": shared_signatures[:5],
        "shared_predicates": shared_predicates[:5],
        "shared_labels": shared_labels[:5],
        "left_count": left_count,
        "right_count": right_count,
    }


def _alias_overlap(source: _ConceptCandidate, target: _ConceptCandidate) -> dict[str, Any]:
    source_values = set(source.normalized_values)
    target_values = set(target.normalized_values)
    exact_overlap = sorted(source_values & target_values)
    collapsed_overlap = sorted(_collapsed_values(source.normalized_values) & _collapsed_values(target.normalized_values))
    source_tokens = _normalized_token_set(source.normalized_values)
    target_tokens = _normalized_token_set(target.normalized_values)
    token_jaccard = _token_overlap(source_tokens, target_tokens)
    shared_tokens = sorted(source_tokens & target_tokens)
    best_similarity = _similarity(source.normalized_values, target.normalized_values)

    strength = "weak"
    if exact_overlap or collapsed_overlap:
        strength = "strong"
    elif token_jaccard >= 0.75 or (len(shared_tokens) >= 2 and best_similarity >= 0.95):
        strength = "strong"
    elif token_jaccard >= 0.5 or (len(shared_tokens) >= 1 and best_similarity >= 0.97):
        strength = "moderate"

    return {
        "strength": strength,
        "exact_overlap": exact_overlap[:5],
        "collapsed_overlap": collapsed_overlap[:5],
        "token_jaccard": round(token_jaccard, 4),
        "shared_tokens": shared_tokens[:8],
        "best_similarity": round(best_similarity, 4),
    }


def _provenance_overlap(source: _EntityContext, target: _EntityContext) -> dict[str, Any]:
    shared_notes = sorted(source.note_ids & target.note_ids)
    shared_domains = sorted(source.domains & target.domains)
    shared_topics = sorted(source.topics & target.topics)
    shared_sources = sorted(source.provenance_sources & target.provenance_sources)

    conflict = False
    if source.topics and target.topics and not shared_topics:
        conflict = True
    elif not shared_notes and not shared_topics and source.domains and target.domains and not shared_domains:
        conflict = True

    strength = "sparse"
    if shared_notes or shared_domains or shared_topics:
        strength = "strong"
    elif source.note_ids or source.domains or source.topics or target.note_ids or target.domains or target.topics:
        strength = "weak"

    return {
        "strength": strength,
        "conflict": conflict,
        "shared_notes": shared_notes[:5],
        "shared_domains": shared_domains[:5],
        "shared_topics": shared_topics[:5],
        "shared_sources": shared_sources[:5],
        "source_note_count": len(source.note_ids),
        "target_note_count": len(target.note_ids),
        "source_domain_count": len(source.domains),
        "target_domain_count": len(target.domains),
        "source_topic_count": len(source.topics),
        "target_topic_count": len(target.topics),
    }


def _build_split_signal(
    candidate: _ConceptCandidate,
    context: _EntityContext,
    counterparty: _ConceptCandidate,
    precision: dict[str, Any],
) -> dict[str, Any] | None:
    weak_merge = bool(precision.get("suppressed")) or float(precision.get("adjusted_confidence") or 0.0) < 0.9
    if not weak_merge:
        return None

    overload_signals: list[str] = []
    confidence = 0.46
    if context.alias_divergence > 0:
        overload_signals.append("alias_divergence")
        confidence += min(0.08 * context.alias_divergence, 0.16)
    if len(context.topics) > 1:
        overload_signals.append("topic_diversity")
        confidence += 0.12
    if len(context.domains) > 1:
        overload_signals.append("domain_diversity")
        confidence += 0.08
    if len(context.counterpart_labels) >= 3:
        overload_signals.append("semantic_mix")
        confidence += 0.08
    if precision.get("provenance", {}).get("conflict"):
        overload_signals.append("provenance_conflict")
        confidence += 0.08
    if precision.get("claim_overlap", {}).get("strength") in {"weak", "sparse"}:
        overload_signals.append("weak_claim_overlap")
        confidence += 0.04

    if not overload_signals or confidence < 0.58:
        return None

    return {
        "source_entity_id": candidate.entity_id,
        "candidate_entity_ids": [counterparty.entity_id],
        "confidence": round(min(0.92, confidence), 4),
        "reason": {
            "source_display_name": candidate.canonical_name,
            "candidate_display_names": [counterparty.canonical_name],
            "overload_signals": overload_signals,
            "alias_divergence": context.alias_divergence,
            "topic_diversity": len(context.topics),
            "domain_diversity": len(context.domains),
            "semantic_neighbor_count": len(context.counterpart_labels),
            "weak_merge_confidence": round(float(precision.get("adjusted_confidence") or 0.0), 4),
            "merge_penalties": list(precision.get("penalties") or []),
        },
    }


def _evaluate_merge_pair(
    source: _ConceptCandidate,
    target: _ConceptCandidate,
    *,
    source_context: _EntityContext,
    target_context: _EntityContext,
    confidence: float,
    match_method: str,
    fuzzy_threshold: float,
) -> dict[str, Any]:
    alias_overlap = _alias_overlap(source, target)
    claim_overlap = _overlap_strength(
        source_context.claim_signatures,
        target_context.claim_signatures,
        source_context.claim_predicates,
        target_context.claim_predicates,
        source_context.claim_labels,
        target_context.claim_labels,
    )
    relation_overlap = _overlap_strength(
        source_context.relation_signatures,
        target_context.relation_signatures,
        source_context.relation_predicates,
        target_context.relation_predicates,
        source_context.relation_labels,
        target_context.relation_labels,
    )
    provenance = _provenance_overlap(source_context, target_context)

    penalties: list[str] = []
    suppress_reasons: list[str] = []
    adjusted_confidence = float(confidence)

    if source_context.entity_type and target_context.entity_type and source_context.entity_type != target_context.entity_type:
        suppress_reasons.append("entity_type_mismatch")

    alias_strength = str(alias_overlap.get("strength") or "weak")
    if alias_strength == "weak":
        penalties.append("weak_alias_overlap")
        adjusted_confidence -= 0.12
    elif alias_strength == "moderate":
        penalties.append("moderate_alias_overlap")
        adjusted_confidence -= 0.04

    if provenance.get("conflict"):
        penalties.append("provenance_conflict")
        adjusted_confidence -= 0.08 if alias_strength != "strong" else 0.03

    claim_strength = str(claim_overlap.get("strength") or "sparse")
    if claim_strength == "weak":
        penalties.append("weak_claim_overlap")
        adjusted_confidence -= 0.12
    elif claim_strength == "sparse" and alias_strength == "weak":
        penalties.append("sparse_claim_support")
        adjusted_confidence -= 0.04

    relation_strength = str(relation_overlap.get("strength") or "sparse")
    if relation_strength == "weak" and claim_strength in {"weak", "sparse"}:
        penalties.append("weak_relation_overlap")
        adjusted_confidence -= 0.05

    if alias_strength == "strong" and (
        provenance.get("strength") == "strong"
        or claim_strength == "strong"
        or relation_strength == "strong"
    ):
        adjusted_confidence = min(0.999, adjusted_confidence + 0.02)

    adjusted_confidence = max(0.0, min(0.999, adjusted_confidence))

    suppressed = bool(suppress_reasons)
    if not suppressed and alias_strength == "weak" and provenance.get("conflict") and claim_strength in {"weak", "sparse"}:
        suppress_reasons.append("precision_first_combined_conflict")
        suppressed = True
    if (
        not suppressed
        and match_method != "normalized_exact"
        and adjusted_confidence < max(0.9, fuzzy_threshold)
        and alias_strength != "strong"
        and claim_strength != "strong"
    ):
        suppress_reasons.append("precision_first_low_confidence")
        suppressed = True

    precision = {
        "base_confidence": round(float(confidence), 4),
        "adjusted_confidence": round(adjusted_confidence, 4),
        "suppressed": suppressed,
        "suppress_reasons": suppress_reasons,
        "penalties": penalties,
        "alias_overlap": alias_overlap,
        "claim_overlap": claim_overlap,
        "relation_overlap": relation_overlap,
        "provenance": provenance,
        "source_entity_type": source_context.entity_type,
        "target_entity_type": target_context.entity_type,
    }

    return {
        "precision": precision,
        "source_split_signal": _build_split_signal(source, source_context, target, precision),
        "target_split_signal": _build_split_signal(target, target_context, source, precision),
    }


def _accumulate_split_signal(
    bucket: dict[str, dict[str, Any]],
    signal: dict[str, Any] | None,
    *,
    topic_slug: str,
    note_id: str,
    source_url: str,
) -> None:
    if not signal:
        return
    source_entity_id = str(signal.get("source_entity_id") or "").strip()
    candidate_ids = sorted(
        {
            str(candidate_id).strip()
            for candidate_id in signal.get("candidate_entity_ids") or []
            if str(candidate_id).strip() and str(candidate_id).strip() != source_entity_id
        }
    )
    if not source_entity_id or not candidate_ids:
        return

    reason = signal.get("reason") if isinstance(signal.get("reason"), dict) else {}
    item = bucket.setdefault(
        source_entity_id,
        {
            "topic_slug": str(topic_slug or ""),
            "source_entity_id": source_entity_id,
            "candidate_entity_ids": set(),
            "confidence": 0.0,
            "reason": {
                "source_display_name": str(reason.get("source_display_name") or ""),
                "candidate_display_names": [],
                "overload_signals": [],
                "alias_divergence": int(reason.get("alias_divergence") or 0),
                "topic_diversity": int(reason.get("topic_diversity") or 0),
                "domain_diversity": int(reason.get("domain_diversity") or 0),
                "semantic_neighbor_count": int(reason.get("semantic_neighbor_count") or 0),
                "weak_merge_confidence": float(reason.get("weak_merge_confidence") or 0.0),
                "merge_penalties": [],
                "note_id": str(note_id or ""),
                "source_url": str(source_url or ""),
            },
        },
    )
    item["candidate_entity_ids"].update(candidate_ids)
    item["confidence"] = max(float(item.get("confidence") or 0.0), float(signal.get("confidence") or 0.0))

    candidate_display_names = item["reason"]["candidate_display_names"]
    for display_name in reason.get("candidate_display_names") or []:
        token = str(display_name or "").strip()
        if token and token not in candidate_display_names:
            candidate_display_names.append(token)

    overload_signals = item["reason"]["overload_signals"]
    for overload_signal in reason.get("overload_signals") or []:
        token = str(overload_signal or "").strip()
        if token and token not in overload_signals:
            overload_signals.append(token)

    merge_penalties = item["reason"]["merge_penalties"]
    for penalty in reason.get("merge_penalties") or []:
        token = str(penalty or "").strip()
        if token and token not in merge_penalties:
            merge_penalties.append(token)

    item["reason"]["alias_divergence"] = max(item["reason"]["alias_divergence"], int(reason.get("alias_divergence") or 0))
    item["reason"]["topic_diversity"] = max(item["reason"]["topic_diversity"], int(reason.get("topic_diversity") or 0))
    item["reason"]["domain_diversity"] = max(item["reason"]["domain_diversity"], int(reason.get("domain_diversity") or 0))
    item["reason"]["semantic_neighbor_count"] = max(
        item["reason"]["semantic_neighbor_count"],
        int(reason.get("semantic_neighbor_count") or 0),
    )
    item["reason"]["weak_merge_confidence"] = max(
        float(item["reason"]["weak_merge_confidence"] or 0.0),
        float(reason.get("weak_merge_confidence") or 0.0),
    )


def _entity_strength(db, entity_id: str, entity_type: str, *, alias_count: int = 0) -> float:
    relation_count = len(db.get_relations(entity_type or "concept", entity_id))
    claim_count = len(db.list_claims_by_entity(entity_id, limit=80))
    paper_count = len(db.get_concept_papers(entity_id)) if entity_type == "concept" else 0
    return (2.0 * relation_count) + (2.0 * claim_count) + (1.25 * paper_count) + (0.5 * alias_count)


def _note_concept_ids(db, note_id: str) -> list[str]:
    concept_ids: list[str] = []
    for relation in db.get_relations("note", note_id):
        target_type = str(relation.get("target_type") or "").strip()
        target_id = str(relation.get("target_id") or relation.get("target_entity_id") or "").strip()
        if target_type == "concept" and target_id and target_id not in concept_ids:
            concept_ids.append(target_id)
    return concept_ids


def build_entity_merge_proposals_for_note(
    db,
    *,
    topic_slug: str,
    note_id: str,
    source_url: str = "",
    fuzzy_threshold: float = 0.94,
    max_candidates: int = 20,
) -> list[dict[str, Any]]:
    """Queue merge proposals for likely duplicate concepts mentioned in a note."""
    note_token = str(note_id or "").strip()
    if not note_token:
        return []

    concept_ids = _note_concept_ids(db, note_token)
    if not concept_ids:
        return []

    candidates = _load_concept_candidates(db)
    if not candidates:
        return []

    created: list[dict[str, Any]] = []
    split_signals: dict[str, dict[str, Any]] = {}
    context_cache: dict[str, _EntityContext] = {}
    note_cache: dict[str, dict[str, Any]] = {}
    entity_name_cache: dict[str, str] = {}
    max_candidates = max(1, int(max_candidates))
    fuzzy_threshold = max(0.0, min(1.0, float(fuzzy_threshold)))

    for concept_id in concept_ids:
        source = candidates.get(concept_id)
        if not source or not source.normalized_values:
            continue

        source_context = _build_entity_context(
            db,
            source,
            context_cache=context_cache,
            note_cache=note_cache,
            entity_name_cache=entity_name_cache,
        )
        source_strength = _entity_strength(
            db,
            source.entity_id,
            source_context.entity_type,
            alias_count=len(source.aliases),
        )
        matches: list[tuple[float, str, str]] = []
        for target_id, target in candidates.items():
            if target_id == source.entity_id:
                continue
            method = ""
            score = 0.0
            if set(source.normalized_values) & set(target.normalized_values):
                method = "normalized_exact"
                score = 0.995
            else:
                score = _similarity(source.normalized_values, target.normalized_values)
                if score >= fuzzy_threshold:
                    method = "fuzzy"
            if not method:
                continue
            matches.append((score, method, target_id))

        matches.sort(key=lambda item: (-item[0], item[2]))
        for score, method, target_id in matches[:max_candidates]:
            target = candidates.get(target_id)
            if not target:
                continue
            target_context = _build_entity_context(
                db,
                target,
                context_cache=context_cache,
                note_cache=note_cache,
                entity_name_cache=entity_name_cache,
            )
            target_strength = _entity_strength(
                db,
                target.entity_id,
                target_context.entity_type,
                alias_count=len(target.aliases),
            )
            proposal_source = source.entity_id
            proposal_target = target.entity_id
            if target_strength < source_strength:
                proposal_source = target.entity_id
                proposal_target = source.entity_id

            if proposal_source == proposal_target:
                continue

            evaluation = _evaluate_merge_pair(
                source,
                target,
                source_context=source_context,
                target_context=target_context,
                confidence=float(score),
                match_method=method,
                fuzzy_threshold=fuzzy_threshold,
            )
            precision = evaluation["precision"]
            _accumulate_split_signal(
                split_signals,
                evaluation.get("source_split_signal"),
                topic_slug=topic_slug,
                note_id=note_token,
                source_url=source_url,
            )
            _accumulate_split_signal(
                split_signals,
                evaluation.get("target_split_signal"),
                topic_slug=topic_slug,
                note_id=note_token,
                source_url=source_url,
            )
            if precision.get("suppressed"):
                continue

            proposal_id = db.add_entity_merge_proposal(
                source_entity_id=proposal_source,
                target_entity_id=proposal_target,
                topic_slug=str(topic_slug or ""),
                confidence=float(precision.get("adjusted_confidence") or score),
                match_method=method,
                reason={
                    "source_display_name": source.canonical_name,
                    "target_display_name": target.canonical_name,
                    "normalized_source": sorted(source.normalized_values),
                    "normalized_target": sorted(target.normalized_values),
                    "note_id": note_token,
                    "source_url": str(source_url or ""),
                    "source_strength": round(source_strength, 4),
                    "target_strength": round(target_strength, 4),
                    "precision_first": precision,
                },
            )
            if proposal_id:
                created.append(
                    {
                        "proposalId": int(proposal_id),
                        "sourceEntityId": proposal_source,
                        "targetEntityId": proposal_target,
                        "confidence": float(precision.get("adjusted_confidence") or score),
                        "matchMethod": method,
                    }
                )
    split_store = getattr(db, "entity_resolution_store", None)
    add_split = getattr(split_store, "add_entity_split_proposal", None)
    if callable(add_split):
        for item in split_signals.values():
            add_split(
                source_entity_id=item["source_entity_id"],
                candidate_entity_ids=sorted(item["candidate_entity_ids"]),
                topic_slug=item["topic_slug"],
                confidence=float(item["confidence"]),
                reason=item["reason"],
            )
    return created
