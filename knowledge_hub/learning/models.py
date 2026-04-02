"""Learning Coach MVP v2 domain models and shared constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RELATION_ENUM = (
    "causes",
    "enables",
    "part_of",
    "contrasts",
    "example_of",
    "requires",
    "improves",
    "related_to",
    "unknown_relation",
)

MAP_SCHEMA = "knowledge-hub.learning.map.result.v1"
GRADE_SCHEMA = "knowledge-hub.learning.grade.result.v1"
NEXT_SCHEMA = "knowledge-hub.learning.next.result.v1"
TEMPLATE_SCHEMA = "knowledge-hub.learning.template.result.v1"
RUN_SCHEMA = "knowledge-hub.learning.run.result.v1"
GAP_SCHEMA = "knowledge-hub.learning.gap.result.v1"
QUIZ_GENERATE_SCHEMA = "knowledge-hub.learning.quiz.generate.result.v1"
QUIZ_GRADE_SCHEMA = "knowledge-hub.learning.quiz.grade.result.v1"
PATCH_SUGGEST_SCHEMA = "knowledge-hub.learning.patch.suggest.result.v1"


@dataclass
class PolicyStatus:
    mode: str = "local-only"
    allowed: bool = True
    classification: str = "P2"
    rule: str = "allow_non_p0"
    trace_id: str = ""
    blocked_reason: str | None = None
    policy_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "allowed": self.allowed,
            "classification": self.classification,
            "rule": self.rule,
            "traceId": self.trace_id,
            "trace_id": self.trace_id,
            "blockedReason": self.blocked_reason,
            "policyErrors": self.policy_errors,
            "warnings": self.warnings,
        }


@dataclass
class EvidencePointer:
    type: str = "note"
    path: str = ""
    heading: str = ""
    block_id: str = ""
    snippet_hash: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "type": self.type,
            "path": self.path,
            "heading": self.heading,
            "block_id": self.block_id,
            "snippet_hash": self.snippet_hash,
        }


@dataclass
class ConceptIdentity:
    canonical_id: str
    display_name: str
    aliases: list[str] = field(default_factory=list)
    resolve_confidence: float = 1.0
    resolve_method: str = "exact"

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "display_name": self.display_name,
            "aliases": self.aliases,
            "resolveConfidence": round(float(self.resolve_confidence), 4),
            "resolveMethod": self.resolve_method,
        }


@dataclass
class TrunkConcept:
    identity: ConceptIdentity
    trunk_score: float
    topic_relevance: float
    graph_centrality: float
    evidence_coverage: float
    recency: float
    evidence_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity.to_dict(),
            "trunkScore": round(float(self.trunk_score), 6),
            "scoreBreakdown": {
                "topicRelevance": round(float(self.topic_relevance), 6),
                "graphCentrality": round(float(self.graph_centrality), 6),
                "evidenceCoverage": round(float(self.evidence_coverage), 6),
                "recency": round(float(self.recency), 6),
            },
            "evidenceSources": self.evidence_sources,
        }


@dataclass
class BranchConcept:
    identity: ConceptIdentity
    parent_trunk_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity.to_dict(),
            "parentTrunkIds": self.parent_trunk_ids,
            "confidence": round(float(self.confidence), 6),
        }


@dataclass
class AssessmentEdge:
    source_canonical_id: str
    relation_raw: str
    relation_norm: str
    target_canonical_id: str
    evidence_ptrs: list[EvidencePointer] = field(default_factory=list)
    confidence: float = 3.0
    is_valid: bool = False
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceCanonicalId": self.source_canonical_id,
            "relationRaw": self.relation_raw,
            "relationNorm": self.relation_norm,
            "targetCanonicalId": self.target_canonical_id,
            "evidencePtrs": [ptr.to_dict() for ptr in self.evidence_ptrs],
            "confidence": round(float(self.confidence), 4),
            "isValid": self.is_valid,
            "issues": self.issues,
        }


@dataclass
class AssessmentScore:
    coverage: float
    edge_accuracy: float
    explanation_quality: float
    final: float
    total_edges: int
    valid_edges: int
    min_edges: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": round(float(self.coverage), 6),
            "edgeAccuracy": round(float(self.edge_accuracy), 6),
            "explanationQuality": round(float(self.explanation_quality), 6),
            "final": round(float(self.final), 6),
            "totalEdges": int(self.total_edges),
            "validEdges": int(self.valid_edges),
            "minEdges": int(self.min_edges),
        }


@dataclass
class ProgressGateDecision:
    passed: bool
    status: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status,
            "reasons": self.reasons,
        }
