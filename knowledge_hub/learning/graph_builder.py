"""Learning graph projection builder."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import yaml

from knowledge_hub.core.models import LearningGraphCandidate
from knowledge_hub.learning.contracts import LearningGraphDataRepository
from knowledge_hub.learning.difficulty import score_difficulty
from knowledge_hub.learning.graph_models import LearningEdge, LearningNode
from knowledge_hub.learning.mapper import generate_learning_map, slugify_topic
from knowledge_hub.learning.prerequisites import edge_from_relation, edges_from_claim
from knowledge_hub.learning.resource_mapper import build_resource_link

_GENERIC_SCOPE_TERMS = {
    "accuracy",
    "agent",
    "agents",
    "assistant",
    "context",
    "dynamic",
    "efficient",
    "framework",
    "human",
    "implementation",
    "interactive",
    "knowledge",
    "medical",
    "nvidia",
    "optimal solutions",
    "performance",
    "prediction",
    "probabilistic models",
    "semantic",
    "strategy",
    "structured",
    "teaching methodology",
    "time horizon",
    "user",
    "user preferences",
}

_TECHNICAL_SCOPE_TERMS = {
    "agentic",
    "attention",
    "benchmark",
    "diffusion",
    "embedding",
    "evaluation",
    "fine-tuning",
    "generation",
    "inference",
    "language modeling",
    "large language models",
    "llm",
    "moe",
    "multimodal",
    "natural language processing",
    "neural network",
    "optimization",
    "pretraining",
    "rag",
    "reasoning",
    "reinforcement",
    "reinforcement learning",
    "retrieval",
    "safety",
    "training",
    "transformer",
}

_TOPIC_RULES_DIR = Path(__file__).resolve().parents[2] / "data" / "curation" / "learning_topic_rules"


def _normalize_scope_term(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+\- ]+", " ", str(value or "").strip().lower()))
    return cleaned.strip()


def _looks_technical(value: str) -> bool:
    normalized = _normalize_scope_term(value)
    if not normalized:
        return False
    if normalized in _TECHNICAL_SCOPE_TERMS:
        return True
    if any(token in normalized for token in ("llm", "rag", "moe", "transformer", "embedding", "diffusion")):
        return True
    tokens = [token for token in normalized.split() if token]
    return len(tokens) >= 2 and any(token in _TECHNICAL_SCOPE_TERMS for token in tokens)


def _is_generic_scope_term(value: str) -> bool:
    normalized = _normalize_scope_term(value)
    if not normalized:
        return True
    return normalized in _GENERIC_SCOPE_TERMS and not _looks_technical(normalized)


@lru_cache(maxsize=32)
def _load_topic_scope_rule(topic_slug: str) -> dict[str, Any]:
    normalized_topic = str(topic_slug or "").strip().lower()
    path = _TOPIC_RULES_DIR / f"{normalized_topic}.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _topic_rule_values(topic_slug: str, key: str) -> set[str]:
    raw = _load_topic_scope_rule(topic_slug).get(key) or []
    if not isinstance(raw, list):
        return set()
    return {_normalize_scope_term(value) for value in raw if _normalize_scope_term(value)}


def _topic_rule_regexes(topic_slug: str) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    raw = _load_topic_scope_rule(topic_slug).get("deny_regex") or []
    if not isinstance(raw, list):
        return compiled
    for pattern in raw:
        try:
            compiled.append(re.compile(str(pattern), re.IGNORECASE))
        except re.error:
            continue
    return compiled


def _topic_min_support_docs(topic_slug: str) -> int:
    try:
        return max(0, int(_load_topic_scope_rule(topic_slug).get("min_support_docs", 0)))
    except Exception:
        return 0


def topic_preferred_prerequisites(topic_slug: str) -> list[str]:
    raw = _load_topic_scope_rule(topic_slug).get("preferred_prerequisites") or []
    if not isinstance(raw, list):
        return []
    return [value for value in (_normalize_scope_term(item) for item in raw) if value]


def _preferred_prerequisite_rank(topic_slug: str) -> dict[str, int]:
    return {
        value: index
        for index, value in enumerate(topic_preferred_prerequisites(topic_slug))
    }


def _preferred_prerequisite_adjustment(
    topic_slug: str,
    *,
    source_name: str,
    target_name: str,
) -> tuple[float, dict[str, Any]]:
    rank_map = _preferred_prerequisite_rank(topic_slug)
    source_key = _normalize_scope_term(source_name)
    target_key = _normalize_scope_term(target_name)
    source_rank = rank_map.get(source_key)
    target_rank = rank_map.get(target_key)
    if source_rank is None or target_rank is None or source_rank == target_rank:
        return 0.0, {}
    if source_rank < target_rank:
        return 0.08, {
            "matched": True,
            "direction": "preferred",
            "sourceRank": source_rank,
            "targetRank": target_rank,
        }
    return -0.08, {
        "matched": True,
        "direction": "deprioritized",
        "sourceRank": source_rank,
        "targetRank": target_rank,
    }


def _is_topic_anchor(topic_slug: str, value: str) -> bool:
    normalized_value = _normalize_scope_term(value)
    if not normalized_value:
        return False
    anchors = _topic_rule_values(topic_slug, "anchors")
    if normalized_value in anchors:
        return True
    return any(anchor and anchor in normalized_value for anchor in anchors)


def _passes_topic_scope_rule(topic_slug: str, value: str) -> bool:
    normalized_topic = str(topic_slug or "").strip().lower()
    normalized_value = _normalize_scope_term(value)
    if not normalized_value:
        return False
    deny_labels = _topic_rule_values(normalized_topic, "deny_labels")
    if normalized_value in deny_labels:
        return False
    if any(token and token in normalized_value for token in deny_labels):
        return False
    for pattern in _topic_rule_regexes(normalized_topic):
        if pattern.search(normalized_value):
            return False
    if _is_topic_anchor(normalized_topic, normalized_value):
        return True
    deny_contains = _topic_rule_values(normalized_topic, "deny_contains")
    if any(token and token in normalized_value for token in deny_contains):
        return False
    return True


class LearningGraphBuilder:
    def __init__(self, db: LearningGraphDataRepository):
        self.db = db

    def build_topic_candidates(self, topic: str, top_k: int) -> dict[str, Any]:
        scoped = generate_learning_map(
            db=self.db,
            topic=topic,
            source="all",
            days=3650,
            top_k=top_k,
            allow_external=False,
            run_id=f"learn_graph_scope_{uuid4().hex[:12]}",
        )
        topic_slug = str(scoped.get("topicSlug") or slugify_topic(topic))
        feature_loader = getattr(self.db, "list_feature_snapshots", None)
        feature_rows = []
        if callable(feature_loader):
            feature_rows = feature_loader(topic_slug=topic_slug, feature_kind="concept", limit=5000) or []
        concept_feature_snapshots = {
            str(item.get("entity_id") or ""): item
            for item in feature_rows
            if str(item.get("entity_id") or "").strip()
        }
        trunks = scoped.get("trunks") if isinstance(scoped.get("trunks"), list) else []
        branches = scoped.get("branches") if isinstance(scoped.get("branches"), list) else []
        candidate_ids: list[str] = []
        for item in trunks + branches:
            canonical_id = str(item.get("canonical_id") or "").strip()
            if canonical_id and canonical_id not in candidate_ids:
                candidate_ids.append(canonical_id)

        related_rows = self.db.list_relations(limit=10000, source_type="concept", target_type="concept")
        related_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        candidate_set = set(candidate_ids)
        for row in related_rows:
            sid = str(row.get("source_entity_id") or row.get("source_id") or "").strip()
            tid = str(row.get("target_entity_id") or row.get("target_id") or "").strip()
            if sid in candidate_set:
                related_by_id[sid].append(row)
            if tid in candidate_set:
                related_by_id[tid].append(row)

        papers_by_concept: dict[str, list[dict[str, Any]]] = {}
        for concept_id in candidate_ids:
            papers_by_concept[concept_id] = self.db.get_concept_papers(concept_id)

        filtered_trunks: list[dict[str, Any]] = []
        min_support_docs = _topic_min_support_docs(topic_slug)
        for item in trunks:
            canonical_id = str(item.get("canonical_id") or "").strip()
            if not canonical_id:
                continue
            display_name = str(item.get("display_name") or canonical_id)
            if not _passes_topic_scope_rule(topic_slug, display_name):
                continue
            score_breakdown = item.get("scoreBreakdown") or {}
            trunk_score = float(item.get("trunkScore") or 0.0)
            topic_relevance = float(score_breakdown.get("topicRelevance") or 0.0)
            graph_centrality = float(score_breakdown.get("graphCentrality") or 0.0)
            paper_support = len(papers_by_concept.get(canonical_id, []))
            relation_degree = len(related_by_id.get(canonical_id, []))
            anchored = _is_topic_anchor(topic_slug, display_name)
            if min_support_docs and not anchored and paper_support < min_support_docs and relation_degree < min_support_docs:
                continue
            if _is_generic_scope_term(display_name):
                if paper_support == 0 and topic_relevance < 0.25:
                    continue
                if topic_relevance < 0.16 and paper_support == 0 and relation_degree < 5:
                    continue
                if trunk_score < 0.5 and graph_centrality < 0.85 and paper_support == 0:
                    continue
            filtered_trunks.append(item)

        kept_trunk_ids = {
            str(item.get("canonical_id") or "").strip()
            for item in filtered_trunks
            if str(item.get("canonical_id") or "").strip()
        }

        filtered_branches: list[dict[str, Any]] = []
        for item in branches:
            canonical_id = str(item.get("canonical_id") or "").strip()
            if not canonical_id:
                continue
            display_name = str(item.get("display_name") or canonical_id)
            if not _passes_topic_scope_rule(topic_slug, display_name):
                continue
            parent_ids = [
                str(parent_id).strip()
                for parent_id in (item.get("parentTrunkIds") or [])
                if str(parent_id).strip() in kept_trunk_ids
            ]
            if not parent_ids:
                continue
            paper_support = len(papers_by_concept.get(canonical_id, []))
            relation_degree = len(related_by_id.get(canonical_id, []))
            confidence = float(item.get("confidence") or 0.0)
            technical = _looks_technical(display_name)
            generic = _is_generic_scope_term(display_name)
            anchored = _is_topic_anchor(topic_slug, display_name)

            if min_support_docs and not anchored and paper_support < min_support_docs and relation_degree < min_support_docs:
                continue
            if generic and paper_support == 0 and not anchored:
                continue
            if not technical and len(parent_ids) < 2 and paper_support == 0 and relation_degree < 3:
                continue
            if not technical and paper_support == 0 and len(parent_ids) < 3:
                continue
            if len(parent_ids) < 2 and paper_support == 0 and relation_degree < 2 and confidence < 0.75:
                continue

            branch_copy = dict(item)
            branch_copy["parentTrunkIds"] = parent_ids
            filtered_branches.append(branch_copy)

        trunks = filtered_trunks
        branches = filtered_branches
        concept_ids = []
        for item in trunks + branches:
            canonical_id = str(item.get("canonical_id") or "").strip()
            if canonical_id and canonical_id not in concept_ids:
                concept_ids.append(canonical_id)
        candidate_set = set(concept_ids)
        filtered_related_rows = [
            row
            for row in related_rows
            if str(row.get("source_entity_id") or row.get("source_id") or "").strip() in candidate_set
            and str(row.get("target_entity_id") or row.get("target_id") or "").strip() in candidate_set
        ]

        provisional_edges: list[tuple[str, str]] = []
        for concept_id in concept_ids:
            for row in related_by_id.get(concept_id, []):
                predicate = str(row.get("predicate_id") or row.get("relation") or "")
                sid = str(row.get("source_entity_id") or row.get("source_id") or "").strip()
                tid = str(row.get("target_entity_id") or row.get("target_id") or "").strip()
                if sid not in candidate_set or tid not in candidate_set:
                    continue
                if predicate == "requires":
                    provisional_edges.append((tid, sid))
                elif predicate in {"part_of", "enables", "improves"}:
                    provisional_edges.append((sid, tid))

        incoming_counts: dict[str, int] = defaultdict(int)
        for source_id, target_id in provisional_edges:
            incoming_counts[target_id] += 1

        max_degree = max((len(related_by_id.get(entity_id, [])) for entity_id in concept_ids), default=1)
        max_incoming = max(incoming_counts.values(), default=1)
        nodes: list[LearningNode] = []
        node_by_entity_id: dict[str, str] = {}
        display_name_by_id: dict[str, str] = {}

        for item in trunks + branches:
            entity_id = str(item.get("canonical_id") or "").strip()
            if not entity_id:
                continue
            entity = self.db.get_ontology_entity(entity_id) or {}
            display_name = str(item.get("display_name") or entity.get("canonical_name") or entity_id)
            display_name_by_id[entity_id] = display_name
            subtype = str((entity.get("properties") or {}).get("subtype") or "").strip().lower()
            node_type = "technique" if subtype == "technique" else "concept"
            paper_support = len(papers_by_concept.get(entity_id, []))
            relation_degree = len(related_by_id.get(entity_id, []))
            feature_snapshot = concept_feature_snapshots.get(entity_id, {})
            feature_support = float(feature_snapshot.get("support_doc_count") or 0.0)
            feature_importance = float(feature_snapshot.get("importance_score") or 0.0)
            feature_claim_density = float(feature_snapshot.get("claim_density") or 0.0)
            feature_freshness = float(feature_snapshot.get("freshness_score") or 0.0)
            feature_contradiction = float(feature_snapshot.get("contradiction_score") or 0.0)
            evidence_diversity = min(1.0, (paper_support + relation_degree) / 8.0)
            if feature_support > 0:
                evidence_diversity = max(evidence_diversity, min(1.0, feature_support / 8.0))
            graph_depth = incoming_counts.get(entity_id, 0) / max(1, max_incoming)
            ontology_centrality = relation_degree / max(1, max_degree)
            difficulty = score_difficulty(
                canonical_name=display_name,
                graph_depth=graph_depth,
                ontology_centrality=ontology_centrality,
                papers=papers_by_concept.get(entity_id, []),
                evidence_diversity=evidence_diversity,
            )
            scope_item = item if isinstance(item, dict) else {}
            score_breakdown = scope_item.get("scoreBreakdown") or {}
            node_confidence = min(
                0.98,
                0.45
                + 0.15 * float(score_breakdown.get("topicRelevance") or 0.0)
                + 0.10 * float(score_breakdown.get("evidenceCoverage") or 0.0)
                + 0.15 * min(1.0, feature_importance)
                + 0.10 * min(1.0, feature_claim_density)
                + 0.05 * min(1.0, feature_freshness),
            )
            node_confidence = max(0.05, node_confidence - 0.10 * min(1.0, feature_contradiction))
            node_id = f"lg_node_{entity_id}"
            node_by_entity_id[entity_id] = node_id
            nodes.append(
                LearningNode(
                    node_id=node_id,
                    entity_id=entity_id,
                    node_type=node_type,
                    canonical_name=display_name,
                    difficulty_level=str(difficulty["difficulty_level"]),
                    difficulty_score=float(difficulty["difficulty_score"]),
                    stage=str(difficulty["stage"]),
                    confidence=float(node_confidence),
                    provenance={
                        "topicSlug": topic_slug,
                        "kind": "topic_scope",
                        "isTrunk": entity_id in kept_trunk_ids,
                        "topicRelevance": float(score_breakdown.get("topicRelevance") or 0.0),
                        "graphCentrality": float(score_breakdown.get("graphCentrality") or 0.0),
                        "evidenceCoverage": float(score_breakdown.get("evidenceCoverage") or 0.0),
                        "paperSupport": paper_support,
                        "relationDegree": relation_degree,
                        "parentTrunkCount": len(scope_item.get("parentTrunkIds") or []),
                        "featureImportance": feature_importance,
                        "featureSupportDocCount": feature_support,
                        "featureClaimDensity": feature_claim_density,
                        "featureFreshness": feature_freshness,
                        "featureContradiction": feature_contradiction,
                        "components": difficulty["components"],
                    },
                )
            )

        return {
            "topicSlug": topic_slug,
            "scope": scoped,
            "nodes": nodes,
            "nodeByEntityId": node_by_entity_id,
            "displayNameByEntityId": display_name_by_id,
            "conceptFeatureSnapshots": concept_feature_snapshots,
            "papersByConcept": papers_by_concept,
            "relatedRows": filtered_related_rows,
            "scopeSummary": {
                "inputTrunks": len(scoped.get("trunks") or []),
                "keptTrunks": len(trunks),
                "inputBranches": len(scoped.get("branches") or []),
                "keptBranches": len(branches),
            },
        }

    def generate_edge_candidates(
        self,
        topic: str,
        top_k: int,
        *,
        candidate_data: dict[str, Any] | None = None,
    ) -> list[LearningEdge]:
        candidate_data = candidate_data or self.build_topic_candidates(topic=topic, top_k=top_k)
        topic_slug = candidate_data["topicSlug"]
        node_by_entity_id = candidate_data["nodeByEntityId"]
        display_name_by_id = candidate_data.get("displayNameByEntityId") or {}
        entity_ids = set(node_by_entity_id.keys())
        concept_feature_snapshots = candidate_data.get("conceptFeatureSnapshots") or {}
        result: list[LearningEdge] = []

        for relation in candidate_data["relatedRows"]:
            sid = str(relation.get("source_entity_id") or relation.get("source_id") or "").strip()
            tid = str(relation.get("target_entity_id") or relation.get("target_id") or "").strip()
            if sid not in entity_ids or tid not in entity_ids:
                continue
            edge = edge_from_relation(relation=relation, topic_slug=topic_slug, node_by_entity_id=node_by_entity_id)
            if edge:
                source_snapshot = concept_feature_snapshots.get(sid, {})
                target_snapshot = concept_feature_snapshots.get(tid, {})
                preferred_delta, preferred_meta = _preferred_prerequisite_adjustment(
                    topic_slug,
                    source_name=str(display_name_by_id.get(sid) or sid),
                    target_name=str(display_name_by_id.get(tid) or tid),
                )
                boost = min(
                    0.12,
                    0.06 * min(1.0, float(source_snapshot.get("importance_score") or 0.0))
                    + 0.03 * min(1.0, float(target_snapshot.get("importance_score") or 0.0))
                    + 0.03 * min(1.0, float(source_snapshot.get("claim_density") or 0.0)),
                )
                contradiction_penalty = min(
                    0.10,
                    0.06 * min(1.0, float(source_snapshot.get("contradiction_score") or 0.0))
                    + 0.04 * min(1.0, float(target_snapshot.get("contradiction_score") or 0.0)),
                )
                edge.confidence = max(
                    0.05,
                    min(0.99, float(edge.confidence) + boost - contradiction_penalty + preferred_delta),
                )
                edge.provenance["featureBoost"] = round(boost, 6)
                edge.provenance["contradictionPenalty"] = round(contradiction_penalty, 6)
                if preferred_meta:
                    edge.provenance["preferredPrerequisiteAdjustment"] = {
                        **preferred_meta,
                        "delta": round(preferred_delta, 6),
                    }
                result.append(edge)

        claims = self.db.list_ontology_claims(limit=5000)
        for claim in claims:
            subject_id = str(claim.get("subject_entity_id") or "").strip()
            object_id = str(claim.get("object_entity_id") or "").strip()
            if subject_id not in entity_ids or object_id not in entity_ids:
                continue
            source_snapshot = concept_feature_snapshots.get(subject_id, {})
            target_snapshot = concept_feature_snapshots.get(object_id, {})
            claim_edges = edges_from_claim(claim=claim, topic_slug=topic_slug, node_by_entity_id=node_by_entity_id)
            for edge in claim_edges:
                preferred_delta, preferred_meta = _preferred_prerequisite_adjustment(
                    topic_slug,
                    source_name=str(display_name_by_id.get(subject_id) or subject_id),
                    target_name=str(display_name_by_id.get(object_id) or object_id),
                )
                boost = min(
                    0.10,
                    0.06 * min(1.0, float(source_snapshot.get("claim_density") or 0.0))
                    + 0.04 * min(1.0, float(source_snapshot.get("importance_score") or 0.0)),
                )
                contradiction_penalty = min(
                    0.08,
                    0.05 * min(1.0, float(source_snapshot.get("contradiction_score") or 0.0))
                    + 0.03 * min(1.0, float(target_snapshot.get("contradiction_score") or 0.0)),
                )
                edge.confidence = max(
                    0.05,
                    min(0.99, float(edge.confidence) + boost - contradiction_penalty + preferred_delta),
                )
                edge.provenance["featureBoost"] = round(boost, 6)
                edge.provenance["contradictionPenalty"] = round(contradiction_penalty, 6)
                if preferred_meta:
                    edge.provenance["preferredPrerequisiteAdjustment"] = {
                        **preferred_meta,
                        "delta": round(preferred_delta, 6),
                    }
                result.append(edge)

        deduped: dict[tuple[str, str, str], LearningEdge] = {}
        for edge in result:
            key = (edge.source_node_id, edge.edge_type, edge.target_node_id)
            current = deduped.get(key)
            if current is None or edge.confidence > current.confidence:
                deduped[key] = edge
        return sorted(
            deduped.values(),
            key=lambda item: (-item.confidence, item.edge_type, item.source_node_id, item.target_node_id),
        )

    def generate_resource_links(
        self,
        topic: str,
        top_k: int,
        *,
        candidate_data: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        candidate_data = candidate_data or self.build_topic_candidates(topic=topic, top_k=top_k)
        topic_slug = candidate_data["topicSlug"]
        node_by_entity_id = candidate_data["nodeByEntityId"]
        nodes = {node.node_id: node for node in candidate_data["nodes"]}
        result: list[dict[str, Any]] = []

        for concept_id, papers in candidate_data["papersByConcept"].items():
            concept_node_id = node_by_entity_id.get(concept_id)
            if not concept_node_id:
                continue
            difficulty_level = nodes[concept_node_id].difficulty_level
            for paper in papers[:8]:
                arxiv_id = str(paper.get("arxiv_id") or "").strip()
                if not arxiv_id:
                    continue
                paper_node_id = f"lg_node_paper:{arxiv_id}"
                result.append(
                    build_resource_link(
                        concept_node_id=concept_node_id,
                        resource_node_id=paper_node_id,
                        paper=paper,
                        difficulty_level=difficulty_level,
                        topic_slug=topic_slug,
                    )
                )
        return result

    def queue_pending(
        self,
        topic: str,
        items: list[LearningGraphCandidate | dict[str, Any]],
    ) -> dict[str, Any]:
        topic_slug = slugify_topic(topic)
        queued_ids: list[int] = []
        counts: dict[str, int] = defaultdict(int)
        for item in items:
            item_dict = item.to_dict() if isinstance(item, LearningGraphCandidate) else item
            item_type = str(item_dict.get("itemType") or item_dict.get("item_type") or "").strip()
            if not item_type:
                continue
            pending_id = self.db.add_learning_graph_pending(
                item_type=item_type,
                topic_slug=topic_slug,
                payload=item_dict.get("payload") or {},
                confidence=float(item_dict.get("confidence", 0.5)),
                reason=str(item_dict.get("reason") or ""),
                provenance=item_dict.get("provenance") or {},
            )
            queued_ids.append(pending_id)
            counts[item_type] += 1
        return {
            "topicSlug": topic_slug,
            "queuedIds": queued_ids,
            "counts": dict(counts),
        }
