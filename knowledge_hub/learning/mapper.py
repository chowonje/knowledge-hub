"""Trunk/branch map generation for Learning Coach MVP v2."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.learning.models import MAP_SCHEMA, BranchConcept, ConceptIdentity, TrunkConcept
from knowledge_hub.learning.policy import evaluate_policy_for_payload
from knowledge_hub.learning.resolver import EntityResolver, normalize_term

GENERIC_CONCEPT_TERMS = {
    "model",
    "models",
    "method",
    "methods",
    "system",
    "systems",
    "framework",
    "frameworks",
    "performance",
    "approach",
    "approaches",
    "data",
    "dataset",
}


def slugify_topic(topic: str) -> str:
    token = normalize_term(topic)
    token = re.sub(r"\s+", "-", token)
    token = re.sub(r"-+", "-", token)
    return token or "untitled-topic"


def _tokenize(value: str) -> set[str]:
    normalized = normalize_term(value)
    return {token for token in normalized.split(" ") if token}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    inter = len(left.intersection(right))
    union = len(left.union(right))
    return inter / union if union else 0.0


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        raw = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _recency_score(latest_at: datetime | None, now: datetime, half_life_days: float = 90.0) -> float:
    if latest_at is None:
        return 0.0
    delta = now - latest_at
    age_days = max(0.0, delta.total_seconds() / 86400.0)
    decay = math.exp(-math.log(2.0) * (age_days / max(1.0, half_life_days)))
    return max(0.0, min(1.0, decay))


def _semantic_similarity(topic: str, concept_name: str) -> float:
    # lightweight local-only semantic proxy for MVP
    topic_norm = normalize_term(topic)
    concept_norm = normalize_term(concept_name)
    if not topic_norm or not concept_norm:
        return 0.0
    if topic_norm in concept_norm or concept_norm in topic_norm:
        return 0.95
    topic_tokens = _tokenize(topic_norm)
    concept_tokens = _tokenize(concept_norm)
    return 0.6 * _jaccard(topic_tokens, concept_tokens) + 0.4 * (len(set(topic_norm) & set(concept_norm)) / max(1, len(set(topic_norm) | set(concept_norm))))


def _topic_proximity(topic_tokens: set[str], related_names: list[str]) -> float:
    if not related_names:
        return 0.0
    values = [_jaccard(topic_tokens, _tokenize(name)) for name in related_names]
    return max(values) if values else 0.0


def _note_mentions(notes: list[dict], term: str, source_filter: str) -> tuple[int, int]:
    needle = normalize_term(term)
    note_count = 0
    web_count = 0
    for note in notes:
        source_type = str(note.get("source_type", "")).lower()
        if source_filter == "note" and source_type == "web":
            continue
        if source_filter == "web" and source_type != "web":
            continue
        if source_filter == "paper":
            continue
        hay = str(note.get("hay") or "")
        if not hay:
            title = normalize_term(str(note.get("title", "")))
            content = normalize_term(str(note.get("content", "")))
            hay = f"{title} {content}"
        if needle and needle in hay:
            if source_type == "web":
                web_count += 1
            else:
                note_count += 1
    return note_count, web_count


def _source_diversity_score(has_note: bool, has_paper: bool, has_web: bool) -> float:
    count = int(has_note) + int(has_paper) + int(has_web)
    return count / 3.0


def _relation_quality(rels: list[dict], topic_tokens: set[str]) -> tuple[int, float]:
    if not rels:
        return 0, 0.0

    topical_rels: list[dict] = []
    for rel in rels:
        rel_name = str(rel.get("canonical_name", ""))
        if _jaccard(topic_tokens, _tokenize(rel_name)) > 0:
            topical_rels.append(rel)
    scoped = topical_rels or rels

    degree = len(scoped)
    quality = sum(float(rel.get("confidence", 0.5)) for rel in scoped) / max(1, degree)
    return degree, max(0.0, min(1.0, quality))


def _recommended_top_k(scores: list[float], min_k: int = 6, max_k: int = 20) -> int:
    if not scores:
        return min_k
    limited = scores[: max_k + 1]
    if len(limited) <= 1:
        return min_k
    max_drop = -1.0
    best_index = min_k
    for idx in range(1, len(limited)):
        drop = limited[idx - 1] - limited[idx]
        k = idx
        if k < min_k:
            continue
        if drop > max_drop:
            max_drop = drop
            best_index = k
    return max(min_k, min(best_index, max_k))


def generate_learning_map(
    db: SQLiteDatabase,
    topic: str,
    source: str = "all",
    days: int = 180,
    top_k: int = 12,
    allow_external: bool = False,
    run_id: str | None = None,
) -> dict:
    run_id = str(run_id or f"learn_map_{uuid4().hex[:12]}")
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(1, int(days)))

    policy = evaluate_policy_for_payload(
        allow_external=allow_external,
        raw_texts=[topic],
        mode="external-allowed" if allow_external else "local-only",
    )
    if not policy.allowed:
        return {
            "schema": MAP_SCHEMA,
            "runId": run_id,
            "topic": topic,
            "status": "blocked",
            "policy": policy.to_dict(),
            "trunks": [],
            "branches": [],
            "scoringDetail": {},
            "suggestedTopK": 0,
            "createdAt": now.isoformat(),
            "updatedAt": now.isoformat(),
        }

    resolver = EntityResolver(db)
    concepts = db.list_ontology_entities(entity_type="concept", limit=5000)
    notes_raw = db.list_notes(limit=5000)
    notes: list[dict] = []
    for note in notes_raw:
        ts = _parse_ts(str(note.get("updated_at", "")))
        if ts is None or ts >= cutoff:
            notes.append(
                {
                    **note,
                    "ts": ts,
                    "hay": normalize_term(f"{note.get('title', '')} {note.get('content', '')}"),
                }
            )

    topic_tokens = _tokenize(topic)
    concept_name_by_id = {
        str(item.get("entity_id")): str(item.get("canonical_name"))
        for item in concepts
        if item.get("entity_id") and item.get("canonical_name")
    }

    candidate_rows: list[dict] = []
    max_degree = 1

    for concept in concepts:
        concept_id = str(concept.get("entity_id", "")).strip()
        canonical_name = str(concept.get("canonical_name", "")).strip()
        if not concept_id or not canonical_name:
            continue

        lexical = _jaccard(topic_tokens, _tokenize(canonical_name))
        semantic = max(0.0, min(1.0, _semantic_similarity(topic, canonical_name)))
        if max(lexical, semantic) < 0.08:
            continue

        aliases = db.get_entity_aliases(concept_id)

        related = db.get_related_concepts(concept_id)
        related_names = [str(row.get("canonical_name", "")) for row in related if row.get("canonical_name")]
        proximity = _topic_proximity(topic_tokens, related_names)

        topic_relevance = max(0.0, min(1.0, 0.50 * lexical + 0.35 * semantic + 0.15 * proximity))

        degree, relation_quality = _relation_quality(related, topic_tokens=topic_tokens)
        max_degree = max(max_degree, degree)

        papers_all = db.get_concept_papers(concept_id)
        papers = []
        for paper in papers_all:
            year = paper.get("year")
            if isinstance(year, int) and year < cutoff.year:
                continue
            papers.append(paper)
        has_paper = len(papers) > 0 and source in {"all", "paper"}

        note_count, web_count = _note_mentions(notes, canonical_name, source)
        has_note = note_count > 0
        has_web = web_count > 0

        coverage = _source_diversity_score(has_note=has_note, has_paper=has_paper, has_web=has_web)

        latest_times: list[datetime] = []
        for item in papers:
            year = item.get("year")
            if isinstance(year, int) and 1900 <= year <= 2200:
                latest_times.append(datetime(year, 1, 1, tzinfo=timezone.utc))
        needle = normalize_term(canonical_name)
        for note in notes:
            if needle and needle in str(note.get("hay") or ""):
                ts = note.get("ts")
                if isinstance(ts, datetime):
                    latest_times.append(ts)

        latest_at = max(latest_times) if latest_times else None
        recency = _recency_score(latest_at, now, half_life_days=90.0)

        candidate_rows.append(
            {
                "concept_id": concept_id,
                "canonical_name": canonical_name,
                "aliases": aliases,
                "topic_relevance": topic_relevance,
                "degree": degree,
                "relation_quality": relation_quality,
                "coverage": coverage,
                "recency": recency,
                "has_note": has_note,
                "has_paper": has_paper,
                "has_web": has_web,
            }
        )

    if not candidate_rows:
        return {
            "schema": MAP_SCHEMA,
            "runId": run_id,
            "topic": topic,
            "status": "error",
            "policy": policy.to_dict(),
            "warnings": ["no concepts available"],
            "trunks": [],
            "branches": [],
            "scoringDetail": {},
            "suggestedTopK": 0,
            "createdAt": now.isoformat(),
            "updatedAt": now.isoformat(),
        }

    for row in candidate_rows:
        degree_norm = row["degree"] / max(1, max_degree)
        quality = row["relation_quality"]
        centrality = max(0.0, min(1.0, 0.65 * degree_norm + 0.35 * quality))

        generic_penalty = 0.2 if normalize_term(row["canonical_name"]) in GENERIC_CONCEPT_TERMS else 0.0
        row["graph_centrality"] = max(0.0, centrality - generic_penalty)

        row["trunk_score"] = (
            0.45 * row["topic_relevance"]
            + 0.25 * row["graph_centrality"]
            + 0.20 * row["coverage"]
            + 0.10 * row["recency"]
        )

    sorted_rows = sorted(candidate_rows, key=lambda item: item["trunk_score"], reverse=True)
    sparse = len(sorted_rows) < 6
    if not sparse:
        top_slice = sorted_rows[: min(20, len(sorted_rows))]
        avg_degree = sum(item["degree"] for item in top_slice) / max(1, len(top_slice))
        avg_quality = sum(item["relation_quality"] for item in top_slice) / max(1, len(top_slice))
        sparse = avg_degree < 1.0 or avg_quality < 0.35

    selected_rows = sorted_rows[: max(1, min(top_k, len(sorted_rows)))]
    trunks: list[TrunkConcept] = []

    for row in selected_rows:
        identity = ConceptIdentity(
            canonical_id=row["concept_id"],
            display_name=row["canonical_name"],
            aliases=row["aliases"],
            resolve_confidence=1.0,
            resolve_method="canonical",
        )
        source_names = []
        if row["has_note"]:
            source_names.append("note")
        if row["has_paper"]:
            source_names.append("paper")
        if row["has_web"]:
            source_names.append("web")

        trunks.append(
            TrunkConcept(
                identity=identity,
                trunk_score=row["trunk_score"],
                topic_relevance=row["topic_relevance"],
                graph_centrality=row["graph_centrality"],
                evidence_coverage=row["coverage"],
                recency=row["recency"],
                evidence_sources=source_names,
            )
        )

    trunk_ids = {item.identity.canonical_id for item in trunks}
    branch_candidates: dict[str, dict] = {}

    for trunk in trunks:
        related = db.get_related_concepts(trunk.identity.canonical_id)
        for relation in related:
            cid = str(relation.get("id", "")).strip()
            if not cid or cid in trunk_ids:
                continue
            if cid not in branch_candidates:
                branch_candidates[cid] = {
                    "name": str(relation.get("canonical_name", cid)),
                    "aliases": db.get_entity_aliases(cid),
                    "parent_ids": set(),
                    "confidence": float(relation.get("confidence", 0.4)),
                }
            branch_candidates[cid]["parent_ids"].add(trunk.identity.canonical_id)
            branch_candidates[cid]["confidence"] = max(
                branch_candidates[cid]["confidence"],
                float(relation.get("confidence", 0.4)),
            )

    # sparse fallback: concept co-occurrence across papers
    warnings: list[str] = []
    if sparse and len(branch_candidates) < 3:
        warnings.append("sparse concept graph detected; fallback co-occurrence mode enabled")
        warnings.append("추천: khub paper build-concepts && khub paper normalize-concepts")
        papers = db.list_papers(limit=1000)
        pair_score: defaultdict[str, float] = defaultdict(float)
        pair_parent: defaultdict[str, set[str]] = defaultdict(set)
        for paper in papers:
            concepts_in_paper = db.get_paper_concepts(str(paper.get("arxiv_id", "")))
            ids = [
                str(item.get("entity_id", item.get("id", ""))).strip()
                for item in concepts_in_paper
                if item.get("entity_id") or item.get("id")
            ]
            ids = [cid for cid in ids if cid]
            for src in ids:
                if src not in trunk_ids:
                    continue
                for tgt in ids:
                    if tgt == src or tgt in trunk_ids:
                        continue
                    key = tgt
                    pair_score[key] += 1.0
                    pair_parent[key].add(src)

        # fallback extension: note-based co-occurrence
        top_concepts = {row["concept_id"] for row in sorted_rows[:200]}
        searchable = [
            (cid, normalize_term(concept_name_by_id.get(cid, "")))
            for cid in top_concepts
            if concept_name_by_id.get(cid)
        ]
        for note in notes:
            hay = normalize_term(f"{note.get('title', '')} {note.get('content', '')}")
            if not hay:
                continue
            mentioned = [cid for cid, name in searchable if name and name in hay]
            if len(mentioned) < 2:
                continue
            for src in mentioned:
                if src not in trunk_ids:
                    continue
                for tgt in mentioned:
                    if tgt == src or tgt in trunk_ids:
                        continue
                    pair_score[tgt] += 0.7
                    pair_parent[tgt].add(src)

        if pair_score:
            max_score = max(pair_score.values())
            for cid, raw_score in pair_score.items():
                if cid in branch_candidates:
                    continue
                concept = db.get_ontology_entity(cid)
                if not concept:
                    continue
                branch_candidates[cid] = {
                    "name": str(concept.get("canonical_name", cid)),
                    "aliases": db.get_entity_aliases(cid),
                    "parent_ids": pair_parent[cid],
                    "confidence": max(0.2, min(0.55, raw_score / max(1.0, max_score))),
                }

    branches: list[BranchConcept] = []
    for cid, item in sorted(branch_candidates.items(), key=lambda row: row[1]["confidence"], reverse=True):
        identity = resolver.resolve(item["name"], entity_type="concept") or ConceptIdentity(
            canonical_id=cid,
            display_name=item["name"],
            aliases=item["aliases"],
            resolve_confidence=1.0,
            resolve_method="canonical",
        )
        branches.append(
            BranchConcept(
                identity=identity,
                parent_trunk_ids=sorted(item["parent_ids"]),
                confidence=float(item["confidence"]),
            )
        )

    suggested_top_k = _recommended_top_k([row["trunk_score"] for row in sorted_rows])
    status = "fallback" if sparse else "ok"

    return {
        "schema": MAP_SCHEMA,
        "runId": run_id,
        "topic": topic,
        "topicSlug": slugify_topic(topic),
        "status": status,
        "policy": policy.to_dict(),
        "source": source,
        "timeWindowDays": int(days),
        "trunks": [item.to_dict() for item in trunks],
        "branches": [item.to_dict() for item in branches],
        "scoringDetail": {
            "weights": {
                "topic_relevance": 0.45,
                "graph_centrality": 0.25,
                "evidence_coverage": 0.20,
                "recency": 0.10,
            },
            "topicRelevanceFormula": {
                "lexical": 0.50,
                "semantic": 0.35,
                "topicProximity": 0.15,
            },
            "recencyHalfLifeDays": 90,
            "genericPenaltyTerms": sorted(GENERIC_CONCEPT_TERMS),
        },
        "suggestedTopK": suggested_top_k,
        "sparseFallback": sparse,
        "warnings": warnings,
        "createdAt": now.isoformat(),
        "updatedAt": now.isoformat(),
    }
