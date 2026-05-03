"""Learning graph domain models and schema constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

LEARNING_GRAPH_BUILD_SCHEMA = "knowledge-hub.learning.graph.build.result.v1"
LEARNING_GRAPH_PENDING_SCHEMA = "knowledge-hub.learning.graph.pending.result.v1"
LEARNING_PATH_SCHEMA = "knowledge-hub.learning.path.result.v1"
LEARNING_REVIEW_SCHEMA = "knowledge-hub.learning.review.result.v1"

NODE_TYPES = ("concept", "paper", "technique", "topic_index")
EDGE_TYPES = (
    "prerequisite",
    "recommended_before",
    "builds_on",
    "introduced_by",
    "deepened_by",
    "example_of",
)
LEVELS = ("beginner", "intermediate", "advanced")
PENDING_ITEM_TYPES = ("edge", "path", "difficulty", "resource_link")
PENDING_STATUSES = ("pending", "approved", "rejected", "deprecated")


@dataclass
class LearningNode:
    node_id: str
    entity_id: str | None
    node_type: str
    canonical_name: str
    difficulty_level: str
    difficulty_score: float
    stage: str
    confidence: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodeId": self.node_id,
            "entityId": self.entity_id,
            "nodeType": self.node_type,
            "canonicalName": self.canonical_name,
            "difficultyLevel": self.difficulty_level,
            "difficultyScore": round(float(self.difficulty_score), 6),
            "stage": self.stage,
            "confidence": round(float(self.confidence), 6),
            "provenance": self.provenance,
        }


@dataclass
class LearningEdge:
    edge_id: str
    source_node_id: str
    edge_type: str
    target_node_id: str
    confidence: float
    status: str
    provenance: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edgeId": self.edge_id,
            "sourceNodeId": self.source_node_id,
            "edgeType": self.edge_type,
            "targetNodeId": self.target_node_id,
            "confidence": round(float(self.confidence), 6),
            "status": self.status,
            "provenance": self.provenance,
            "evidence": self.evidence,
        }


@dataclass
class LearningPath:
    path_id: str
    topic_slug: str
    nodes: list[str]
    stages: dict[str, list[dict[str, Any]]]
    score: dict[str, Any]
    status: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pathId": self.path_id,
            "topicSlug": self.topic_slug,
            "nodes": self.nodes,
            "stages": self.stages,
            "score": self.score,
            "status": self.status,
            "provenance": self.provenance,
        }
