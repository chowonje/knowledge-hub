"""Claim/feature layer v1 helpers."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_hub.core.models import FeatureSnapshot
from knowledge_hub.knowledge.contracts import FeatureComputationRepository
from knowledge_hub.learning.mapper import generate_learning_map, slugify_topic


SOURCE_TRUST_SCORES: dict[str, float] = {
    "openai": 0.98,
    "openai_news_rss": 0.97,
    "openai_developer_blog": 0.98,
    "google_deepmind": 0.96,
    "google_research": 0.94,
    "google_ml_glossary": 0.94,
    "anthropic": 0.96,
    "bair": 0.92,
    "aws_ml_blog": 0.88,
    "huggingface_blog": 0.89,
    "huggingface_transformers_docs": 0.9,
    "huggingface": 0.86,
    "arxiv": 0.9,
    "arxiv_cs_cl": 0.91,
    "arxiv_cs_lg": 0.91,
    "paperswithcode": 0.84,
    "paperswithcode_methods": 0.84,
    "nist": 0.96,
    "nist_ai_glossary": 0.96,
    "nist_ai_rmf": 0.96,
    "paper": 0.86,
    "web": 0.72,
}

REFERENCE_TIER_PRIORS: dict[str, float] = {
    "specialist": 0.04,
    "official": 0.035,
    "reference": 0.025,
    "manual_fallback": 0.0,
    "generic": 0.0,
}

REFERENCE_ROLE_PRIORS: dict[str, float] = {
    "standard_reference": 0.04,
    "glossary_reference": 0.035,
    "background_reference": 0.02,
    "benchmark_reference": 0.018,
    "method_reference": 0.015,
    "latest_discovery": 0.0,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tokenize_topic(text: str) -> set[str]:
    body = str(text or "").lower()
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9가-힣]{2,}", body)
        if len(token) >= 2
    }


def _parse_ts(value: Any) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(token)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _note_metadata(note: dict[str, Any]) -> dict[str, Any]:
    raw = note.get("metadata")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalized_claim_value(value: Any) -> str:
    token = str(value or "").strip().lower()
    token = re.sub(r"[_\-]+", " ", token)
    token = re.sub(r"[^a-z0-9가-힣\s:]+", "", token)
    token = re.sub(r"\s+", " ", token)
    return token.strip()


def _claim_predicate_polarity(predicate: str) -> int:
    token = _normalized_claim_value(predicate)
    if not token:
        return 0
    positive_markers = ("improve", "improves", "enable", "enables", "support", "supports", "boost", "increase")
    negative_markers = ("reduce", "reduces", "degrade", "degrades", "harm", "harms", "limit", "limits", "worsen", "decrease")
    if any(marker in token for marker in positive_markers) or any(marker in token for marker in ("향상", "개선", "지원")):
        return 1
    if any(marker in token for marker in negative_markers) or any(marker in token for marker in ("감소", "저하", "제한", "악화")):
        return -1
    return 0


def _claim_predicate_family(predicate: str) -> str:
    token = _normalized_claim_value(predicate)
    if not token:
        return ""
    if any(marker in token for marker in ("improve", "boost", "increase", "enhance")) or any(marker in token for marker in ("향상", "개선")):
        return "effect_positive"
    if any(marker in token for marker in ("reduce", "decrease", "harm", "worsen", "limit")) or any(marker in token for marker in ("감소", "저하", "악화", "제한")):
        return "effect_negative"
    if any(marker in token for marker in ("support", "enable", "allow")) or any(marker in token for marker in ("지원", "허용")):
        return "supportive"
    if any(marker in token for marker in ("prevent", "block", "oppose", "inhibit", "forbid")) or any(marker in token for marker in ("방지", "차단", "억제")):
        return "blocking"
    if "require" in token or "needed for" in token or "required for" in token or "필요" in token:
        return "requirement"
    return token


def _claim_object_key(claim: dict[str, Any]) -> str:
    entity_id = str(claim.get("object_entity_id") or "").strip()
    if entity_id:
        return entity_id
    return _normalized_claim_value(claim.get("object_literal") or claim.get("object_value") or "")


def _compute_contradiction_score(claims: list[dict[str, Any]]) -> float:
    if len(claims) < 2:
        return 0.0

    contradictions = 0
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        subject = str(claim.get("subject_entity_id") or "").strip()
        if subject:
            by_subject[subject].append(claim)
        object_key = _claim_object_key(claim)
        if object_key:
            by_object[object_key].append(claim)

    multi_valued_predicates = {
        "uses",
        "mentions",
        "related_to",
        "example_of",
        "part_of",
        "requires",
        "enables",
        "improves",
    }
    mutually_exclusive_families = {
        frozenset({"effect_positive", "effect_negative"}),
        frozenset({"supportive", "blocking"}),
    }

    for subject_claims in by_subject.values():
        if len(subject_claims) < 2:
            continue
        polarity_by_object: dict[str, set[int]] = defaultdict(set)
        object_values_by_predicate: dict[str, set[str]] = defaultdict(set)
        family_by_object: dict[str, set[str]] = defaultdict(set)
        object_values_by_family: dict[str, set[str]] = defaultdict(set)
        for claim in subject_claims:
            predicate = _normalized_claim_value(claim.get("predicate") or "")
            family = _claim_predicate_family(predicate)
            object_key = _claim_object_key(claim)
            polarity = _claim_predicate_polarity(predicate)
            if predicate and object_key and polarity:
                polarity_by_object[object_key].add(polarity)
            if predicate and object_key:
                object_values_by_predicate[predicate].add(object_key)
            if family and object_key:
                family_by_object[object_key].add(family)
                object_values_by_family[family].add(object_key)
        contradictions += sum(1 for values in polarity_by_object.values() if len(values) > 1)
        for families in family_by_object.values():
            normalized_families = {family for family in families if family}
            for exclusive in mutually_exclusive_families:
                if exclusive.issubset(normalized_families):
                    contradictions += 1
        for predicate, object_values in object_values_by_predicate.items():
            if predicate in multi_valued_predicates:
                continue
            if len(object_values) > 1:
                contradictions += len(object_values) - 1
        for family, object_values in object_values_by_family.items():
            if family in {"effect_positive", "effect_negative", "supportive", "blocking", "requirement"}:
                continue
            if len(object_values) > 1:
                contradictions += len(object_values) - 1

    for object_claims in by_object.values():
        if len(object_claims) < 2:
            continue
        polarity_by_subject: dict[str, set[int]] = defaultdict(set)
        family_by_subject: dict[str, set[str]] = defaultdict(set)
        for claim in object_claims:
            subject_key = str(claim.get("subject_entity_id") or "").strip()
            if not subject_key:
                subject_key = _normalized_claim_value(claim.get("subject_literal") or claim.get("subject_value") or "")
            if not subject_key:
                continue
            predicate = _normalized_claim_value(claim.get("predicate") or "")
            polarity = _claim_predicate_polarity(predicate)
            family = _claim_predicate_family(predicate)
            if predicate and polarity:
                polarity_by_subject[subject_key].add(polarity)
            if family:
                family_by_subject[subject_key].add(family)
        contradictions += sum(1 for values in polarity_by_subject.values() if len(values) > 1)
        combined_families = {family for families in family_by_subject.values() for family in families if family}
        for exclusive in mutually_exclusive_families:
            if exclusive.issubset(combined_families):
                contradictions += 1

    if contradictions <= 0:
        return 0.0
    return _clamp01(contradictions / max(1.0, float(len(claims))))


def topic_matches_text(topic: str, *parts: Any) -> bool:
    topic_terms = _tokenize_topic(topic)
    if not topic_terms:
        return True
    text = " ".join(str(part or "") for part in parts).lower()
    hits = sum(1 for term in topic_terms if term in text)
    return hits >= max(1, min(2, len(topic_terms)))


def compute_freshness_score(published_at: Any = None, updated_at: Any = None, *, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    dt = _parse_ts(published_at) or _parse_ts(updated_at)
    if not dt:
        return 0.25
    age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
    # ~30 days half-life, but never fully zero out.
    return _clamp01(0.15 + 0.85 * math.exp(-age_days / 30.0))


def source_trust_score(*, source_vendor: str = "", source_channel: str = "", source_type: str = "") -> float:
    keys = [
        str(source_channel or "").strip().lower(),
        str(source_vendor or "").strip().lower(),
        str(source_type or "").strip().lower(),
    ]
    for key in keys:
        if key and key in SOURCE_TRUST_SCORES:
            return SOURCE_TRUST_SCORES[key]
    return 0.6


def reference_prior_boost(*, reference_role: str = "", reference_tier: str = "") -> float:
    role_key = str(reference_role or "").strip().lower()
    tier_key = str(reference_tier or "").strip().lower()
    role_boost = REFERENCE_ROLE_PRIORS.get(role_key, 0.0)
    tier_boost = REFERENCE_TIER_PRIORS.get(tier_key, 0.0)
    return _clamp01(min(0.08, role_boost + tier_boost))


def compute_importance_score(
    *,
    source_trust_score_value: float,
    support_doc_count: int,
    relation_degree: float,
    claim_density: float,
    reference_prior_boost_value: float = 0.0,
) -> float:
    support_score = _clamp01(float(support_doc_count) / 8.0)
    relation_score = _clamp01(float(relation_degree) / 12.0)
    claim_score = _clamp01(float(claim_density))
    base = _clamp01(
        0.30 * _clamp01(source_trust_score_value)
        + 0.25 * support_score
        + 0.25 * relation_score
        + 0.20 * claim_score
    )
    return _clamp01(base + min(0.08, _clamp01(reference_prior_boost_value)))


def _source_feature_key(note_id: str, record_id: str, canonical_url: str, source_item_id: str) -> str:
    for candidate in [source_item_id, record_id, note_id, canonical_url]:
        token = str(candidate or "").strip()
        if token:
            return token
    return "unknown-source"


def build_source_feature_snapshots(
    db: FeatureComputationRepository,
    *,
    topic: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    topic_slug = slugify_topic(topic)
    now = datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []
    notes = db.list_notes(source_type="web", limit=max(200, int(limit)))
    for note in notes:
        metadata = _note_metadata(note)
        if not topic_matches_text(topic, note.get("title"), note.get("content"), metadata.get("topic"), metadata.get("source_name")):
            continue
        note_id = str(note.get("id") or "")
        record_id = str(metadata.get("record_id") or "")
        canonical_url = str(metadata.get("canonical_url") or metadata.get("url") or "")
        source_item_id = str(metadata.get("source_item_id") or "")
        relations = db.get_relations("note", note_id)
        entity_ids = {
            str(rel.get("target_id") or rel.get("target_entity_id") or "").strip()
            for rel in relations
            if str(rel.get("target_type") or "").strip() == "concept"
        }
        note_claims = db.list_claims_by_note(note_id, limit=100)
        claim_count = len(note_claims)
        relation_degree = float(len(relations))
        support_doc_count = max(1, len(entity_ids))
        claim_density = _clamp01(claim_count / max(1.0, support_doc_count * 2.0))
        contradiction_score = _compute_contradiction_score(note_claims)
        trust = source_trust_score(
            source_vendor=str(metadata.get("source_vendor") or ""),
            source_channel=str(metadata.get("source_channel") or ""),
            source_type=str(note.get("source_type") or metadata.get("source_type") or ""),
        )
        prior_boost = reference_prior_boost(
            reference_role=str(metadata.get("reference_role") or ""),
            reference_tier=str(metadata.get("reference_tier") or ""),
        )
        freshness = compute_freshness_score(
            metadata.get("published_at"),
            note.get("updated_at") or metadata.get("fetched_at"),
            now=now,
        )
        importance = compute_importance_score(
            source_trust_score_value=trust,
            support_doc_count=support_doc_count,
            relation_degree=relation_degree,
            claim_density=claim_density,
            reference_prior_boost_value=prior_boost,
        )
        payload = {
            "topicSlug": topic_slug,
            "noteId": note_id,
            "recordId": record_id,
            "canonicalUrl": canonical_url,
            "sourceItemId": source_item_id,
            "title": str(note.get("title") or ""),
            "entityIds": sorted(entity_ids),
            "claimCount": claim_count,
            "contradictionScore": contradiction_score,
            "referenceRole": str(metadata.get("reference_role") or ""),
            "referenceTier": str(metadata.get("reference_tier") or ""),
            "referencePriorBoost": prior_boost,
        }
        feature_key = _source_feature_key(note_id, record_id, canonical_url, source_item_id)
        snapshot = FeatureSnapshot(
            topic_slug=topic_slug,
            feature_kind="source",
            feature_key=feature_key,
            feature_name=str(note.get("title") or feature_key),
            note_id=note_id,
            record_id=record_id,
            canonical_url=canonical_url,
            source_item_id=source_item_id,
            freshness_score=freshness,
            importance_score=importance,
            support_doc_count=support_doc_count,
            relation_degree=relation_degree,
            claim_density=claim_density,
            source_trust_score=trust,
            concept_activity_score=_clamp01((relation_degree + claim_count) / 20.0),
            contradiction_score=contradiction_score,
            payload=payload,
        )
        db.upsert_feature_snapshot(snapshot=snapshot)
        results.append(snapshot.to_dict())
    return results


def build_concept_feature_snapshots(
    db: FeatureComputationRepository,
    *,
    topic: str,
    top_k: int = 12,
) -> list[dict[str, Any]]:
    topic_slug = slugify_topic(topic)
    now = datetime.now(timezone.utc)
    scoped = generate_learning_map(
        db=db,
        topic=topic,
        source="all",
        days=3650,
        top_k=max(1, int(top_k)),
        allow_external=False,
        run_id=f"feature_scope_{topic_slug}",
    )
    concept_ids: list[str] = []
    for item in (scoped.get("trunks") or []) + (scoped.get("branches") or []):
        concept_id = str(item.get("canonical_id") or "").strip()
        if concept_id and concept_id not in concept_ids:
            concept_ids.append(concept_id)

    results: list[dict[str, Any]] = []
    for concept_id in concept_ids:
        entity = db.get_ontology_entity(concept_id) or {}
        canonical_name = str(entity.get("canonical_name") or concept_id)
        relations = db.get_relations("concept", concept_id)
        papers = db.get_concept_papers(concept_id)
        claims = db.list_claims_by_entity(concept_id, limit=120)

        support_sources: set[str] = set()
        source_trusts: list[float] = []
        recent_timestamps: list[datetime] = []
        for relation in relations:
            source_type = str(relation.get("source_type") or "").strip()
            source_id = str(relation.get("source_id") or "").strip()
            if source_type == "note" and source_id:
                note = db.get_note(source_id) or {}
                metadata = _note_metadata(note)
                source_url = str(metadata.get("canonical_url") or metadata.get("url") or source_id)
                support_sources.add(source_url)
                source_trusts.append(
                    source_trust_score(
                        source_vendor=str(metadata.get("source_vendor") or ""),
                        source_channel=str(metadata.get("source_channel") or ""),
                        source_type=str(note.get("source_type") or ""),
                    )
                )
                dt = _parse_ts(metadata.get("published_at")) or _parse_ts(note.get("updated_at"))
                if dt:
                    recent_timestamps.append(dt)
        for paper in papers:
            arxiv_id = str(paper.get("arxiv_id") or "")
            if arxiv_id:
                support_sources.add(f"paper:{arxiv_id}")
                source_trusts.append(source_trust_score(source_channel="arxiv", source_type="paper"))
            year = int(paper.get("year") or 0)
            if year > 0:
                recent_timestamps.append(datetime(year, 1, 1, tzinfo=timezone.utc))

        support_doc_count = max(1, len(support_sources))
        relation_degree = float(len(relations))
        claim_density = _clamp01(len(claims) / max(1.0, support_doc_count * 2.0))
        contradiction_score = _compute_contradiction_score(claims)
        trust = sum(source_trusts) / len(source_trusts) if source_trusts else 0.65
        newest = max(recent_timestamps) if recent_timestamps else None
        freshness = compute_freshness_score(newest.isoformat() if newest else "", now=now)
        activity = _clamp01((len(claims) + len(relations) + len(papers)) / 20.0)
        importance = compute_importance_score(
            source_trust_score_value=trust,
            support_doc_count=support_doc_count,
            relation_degree=relation_degree,
            claim_density=claim_density,
        )
        payload = {
            "topicSlug": topic_slug,
            "entityId": concept_id,
            "canonicalName": canonical_name,
            "paperCount": len(papers),
            "claimCount": len(claims),
            "supportSources": sorted(support_sources)[:20],
            "contradictionScore": contradiction_score,
        }
        snapshot = FeatureSnapshot(
            topic_slug=topic_slug,
            feature_kind="concept",
            feature_key=concept_id,
            feature_name=canonical_name,
            entity_id=concept_id,
            freshness_score=freshness,
            importance_score=importance,
            support_doc_count=support_doc_count,
            relation_degree=relation_degree,
            claim_density=claim_density,
            source_trust_score=trust,
            concept_activity_score=activity,
            contradiction_score=contradiction_score,
            payload=payload,
        )
        db.upsert_feature_snapshot(snapshot=snapshot)
        results.append(snapshot.to_dict())
    return results


def snapshot_features(
    db: FeatureComputationRepository,
    *,
    topic: str,
    source_limit: int = 500,
    top_k: int = 12,
) -> dict[str, Any]:
    topic_slug = slugify_topic(topic)
    source_items = build_source_feature_snapshots(db, topic=topic, limit=source_limit)
    concept_items = build_concept_feature_snapshots(db, topic=topic, top_k=top_k)
    return {
        "topic": topic,
        "topicSlug": topic_slug,
        "sourceCount": len(source_items),
        "conceptCount": len(concept_items),
        "topSources": db.list_top_feature_snapshots(topic_slug=topic_slug, feature_kind="source", limit=10),
        "topConcepts": db.list_top_feature_snapshots(topic_slug=topic_slug, feature_kind="concept", limit=10),
    }
