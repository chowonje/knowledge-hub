"""Deterministic learning edge candidate generation."""

from __future__ import annotations

from collections import defaultdict, deque
import re
from uuid import uuid4

from knowledge_hub.learning.graph_models import LearningEdge


_CLAIM_BUILDS_ON = re.compile(r"\bbuilds on\b", flags=re.IGNORECASE)
_CLAIM_REQUIRES = re.compile(r"\brequires?\b|\bneeded for\b|\brequired for\b", flags=re.IGNORECASE)


def edge_from_relation(
    relation: dict,
    topic_slug: str,
    node_by_entity_id: dict[str, str],
) -> LearningEdge | None:
    validation = relation.get("predicate_validation")
    if not isinstance(validation, dict):
        reason_json = relation.get("reason_json") if isinstance(relation.get("reason_json"), dict) else {}
        validation = reason_json.get("validation") if isinstance(reason_json.get("validation"), dict) else {}
    issues = validation.get("issues") if isinstance(validation, dict) else []
    if any("mismatch" in str(issue).lower() or "antisymmetric" in str(issue).lower() for issue in (issues or [])):
        return None

    predicate = str(relation.get("predicate_id") or relation.get("relation") or "").strip()
    source_entity_id = str(relation.get("source_entity_id") or relation.get("source_id") or "").strip()
    target_entity_id = str(relation.get("target_entity_id") or relation.get("target_id") or "").strip()
    if not source_entity_id or not target_entity_id:
        return None
    source_node_id = node_by_entity_id.get(source_entity_id)
    target_node_id = node_by_entity_id.get(target_entity_id)
    if not source_node_id or not target_node_id or source_node_id == target_node_id:
        return None

    edge_type: str | None = None
    learning_source = source_node_id
    learning_target = target_node_id
    confidence = float(relation.get("confidence", 0.5))

    predicate_semantics = relation.get("predicate_semantics") or {}
    supports_transitive_recommended = False

    if predicate == "requires":
        edge_type = "prerequisite"
        learning_source, learning_target = target_node_id, source_node_id
        confidence = max(confidence, 0.85)
    elif predicate == "part_of":
        edge_type = "recommended_before"
        confidence = max(confidence, 0.60)
        supports_transitive_recommended = True
    elif predicate == "enables":
        edge_type = "recommended_before"
        confidence = max(confidence, 0.70)
    elif predicate == "improves":
        edge_type = "builds_on"
        confidence = max(confidence, 0.55)
    elif predicate == "example_of":
        edge_type = "example_of"
        confidence = max(confidence, 0.55)
    elif predicate == "related_to":
        return None

    if edge_type is None:
        return None

    return LearningEdge(
        edge_id=f"lg_edge_{uuid4().hex[:12]}",
        source_node_id=learning_source,
        edge_type=edge_type,
        target_node_id=learning_target,
        confidence=min(1.0, confidence),
        status="pending",
        provenance={
            "topicSlug": topic_slug,
            "derivedFrom": "ontology_relation",
            "predicateId": predicate,
            "sourceEntityId": source_entity_id,
            "targetEntityId": target_entity_id,
            "predicateSemantics": predicate_semantics,
            "validation": validation if isinstance(validation, dict) else {},
            "transitiveSupport": bool(
                edge_type == "prerequisite"
                or supports_transitive_recommended
                or bool(predicate_semantics.get("is_transitive"))
            ),
        },
        evidence={
            "relationId": relation.get("relation_id") or relation.get("id"),
            "predicateId": predicate,
        },
    )


def edges_from_claim(
    claim: dict,
    topic_slug: str,
    node_by_entity_id: dict[str, str],
) -> list[LearningEdge]:
    subject_entity_id = str(claim.get("subject_entity_id") or "").strip()
    object_entity_id = str(claim.get("object_entity_id") or "").strip()
    text = str(claim.get("claim_text") or "")
    if not subject_entity_id or not object_entity_id:
        return []
    subject_node_id = node_by_entity_id.get(subject_entity_id)
    object_node_id = node_by_entity_id.get(object_entity_id)
    if not subject_node_id or not object_node_id or subject_node_id == object_node_id:
        return []

    matches: list[tuple[str, str, str, float]] = []
    if _CLAIM_BUILDS_ON.search(text):
        matches.append(("builds_on", object_node_id, subject_node_id, 0.62))
    if _CLAIM_REQUIRES.search(text):
        matches.append(("prerequisite", object_node_id, subject_node_id, 0.76))

    result: list[LearningEdge] = []
    for edge_type, source_node_id, target_node_id, floor in matches:
        result.append(
            LearningEdge(
                edge_id=f"lg_edge_{uuid4().hex[:12]}",
                source_node_id=source_node_id,
                edge_type=edge_type,
                target_node_id=target_node_id,
                confidence=max(floor, float(claim.get("confidence", 0.5))),
                status="pending",
                provenance={
                    "topicSlug": topic_slug,
                    "derivedFrom": "ontology_claim",
                    "claimId": claim.get("claim_id"),
                    "subjectEntityId": subject_entity_id,
                    "objectEntityId": object_entity_id,
                },
                evidence={
                    "claimId": claim.get("claim_id"),
                    "claimText": text[:400],
                    "evidencePtrs": claim.get("evidence_ptrs") or [],
                },
            )
        )
    return result


def transitive_prerequisite_map(
    topic_nodes: list[str],
    edges: list[LearningEdge],
    *,
    include_recommended: bool = True,
) -> dict[str, list[str]]:
    """Compute prerequisite closure for each topic node."""

    allowed_edge_types = {"prerequisite"}

    topic_node_set = set(topic_nodes)
    reverse_adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.edge_type == "prerequisite":
            pass
        elif edge.edge_type == "recommended_before":
            if not include_recommended:
                continue
            provenance = edge.provenance or {}
            if not bool(provenance.get("transitiveSupport")):
                continue
        else:
            continue
        if edge.source_node_id == edge.target_node_id:
            continue
        if edge.source_node_id not in topic_node_set or edge.target_node_id not in topic_node_set:
            continue
        reverse_adjacency[edge.target_node_id].append(edge.source_node_id)

    result: dict[str, list[str]] = {}
    for node_id in topic_nodes:
        queue = deque(reverse_adjacency.get(node_id, []))
        seen: set[str] = set()
        ordered: list[str] = []
        while queue:
            candidate = queue.popleft()
            if candidate in seen or candidate == node_id:
                continue
            seen.add(candidate)
            ordered.append(candidate)
            for upstream in reverse_adjacency.get(candidate, []):
                if upstream not in seen:
                    queue.append(upstream)
        result[node_id] = ordered
    return result
