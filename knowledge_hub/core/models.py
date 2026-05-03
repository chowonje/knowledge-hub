"""
공유 데이터 모델

모든 모듈에서 사용하는 기본 데이터 클래스입니다.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class SourceType(str, Enum):
    """지식 소스 유형"""
    VAULT = "vault"
    PAPER = "paper"
    WEB = "web"
    NOTE = "note"


class ParaCategory(str, Enum):
    """PARA 분류 체계"""
    PROJECT = "project"
    AREA = "area"
    RESOURCE = "resource"
    ARCHIVE = "archive"


class EntityType(str, Enum):
    """온톨로지 엔티티 유형"""
    CONCEPT = "concept"
    CLAIM = "claim"
    PAPER = "paper"
    NOTE = "note"
    PERSON = "person"
    ORGANIZATION = "organization"
    EVENT = "event"


@dataclass
class Document:
    """
    통합 문서 모델

    Obsidian 노트, 논문, 웹 문서 등을 모두 표현합니다.
    """
    content: str
    metadata: Dict[str, Any]
    file_path: str
    title: str
    tags: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    source_type: SourceType = SourceType.NOTE

    @property
    def id(self) -> str:
        return self.file_path


@dataclass
class SearchResult:
    """검색 결과"""
    document: str
    metadata: Dict[str, Any]
    distance: float
    score: float
    document_id: str = ""
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    retrieval_mode: str = "semantic"
    lexical_extras: Dict[str, Any] | None = None

    def __repr__(self):
        title = self.metadata.get("title", "Untitled")
        return (
            f"SearchResult(title='{title}', score={self.score:.3f}, "
            f"semantic={self.semantic_score:.3f}, lexical={self.lexical_score:.3f}, "
            f"mode={self.retrieval_mode})"
        )


@dataclass
class PaperInfo:
    """논문 메타데이터"""
    arxiv_id: str
    title: str
    authors: str = ""
    year: int = 0
    research_field: str = ""
    importance: int = 0
    notes: str = ""
    pdf_path: Optional[str] = None
    text_path: Optional[str] = None
    translated_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    primary_lane: str = ""
    secondary_tags: List[str] = field(default_factory=list)
    lane_review_status: str = "seeded"
    lane_updated_at: str = ""

    @property
    def has_pdf(self) -> bool:
        return self.pdf_path is not None

    @property
    def has_translation(self) -> bool:
        return self.translated_path is not None


@dataclass
class OntologyEntity:
    """
    통합 온톨로지 엔티티 모델

    Concept, Claim, Paper, Person, Organization, Event 등을 모두 표현합니다.
    """
    entity_id: str
    entity_type: EntityType
    canonical_name: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "system"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value if isinstance(self.entity_type, EntityType) else str(self.entity_type),
            "canonical_name": self.canonical_name,
            "description": self.description,
            "properties": self.properties,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class OntologyClaim:
    """
    온톨로지 주장/명제 모델

    주체-술어-객체 트리플로 표현되는 지식 주장
    예: "RAG reduces hallucination" -> (RAG, reduces, hallucination)
    """
    claim_id: str
    claim_text: str
    subject_entity_id: str
    predicate: str
    object_entity_id: Optional[str] = None
    object_literal: Optional[str] = None
    confidence: float = 0.5
    evidence_ptrs: List[Dict[str, str]] = field(default_factory=list)
    source: str = "extraction"
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "subject_entity_id": self.subject_entity_id,
            "predicate": self.predicate,
            "object_entity_id": self.object_entity_id,
            "object_literal": self.object_literal,
            "confidence": self.confidence,
            "evidence_ptrs": self.evidence_ptrs,
            "source": self.source,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "created_at": self.created_at,
        }


@dataclass
class OntologyEvent:
    """
    온톨로지 변경 이벤트

    모든 온톨로지 변경을 이벤트로 기록하여 time-travel 가능
    """
    event_id: str
    timestamp: str
    event_type: str
    entity_id: str
    entity_type: str
    actor: str
    data: Dict[str, Any]
    policy_class: str = "P2"
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "actor": self.actor,
            "data": self.data,
            "policy_class": self.policy_class,
            "run_id": self.run_id,
        }


@dataclass
class CrawlPipelineCursor:
    """파이프라인 체크포인트 커서 모델."""

    step: str
    cursor: str
    last_record_id: str
    ts: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "cursor": self.cursor,
            "lastRecordId": self.last_record_id,
            "ts": self.ts,
        }


@dataclass
class CrawlPipelineRunResult:
    """대용량 crawl 파이프라인 실행 결과 모델."""

    run_id: str
    job_id: str
    status: str
    profile: str
    source_policy: str
    storage_root: str
    requested: int
    processed: int
    normalized: int
    indexed: int
    pending_domain: int
    failed: int
    skipped: int
    dedupe_rate: float
    retry_rate: float
    memory_peak_ratio: float
    records_per_min: float
    p50_step_latency_ms: float
    cursor: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    schema_errors: List[str] = field(default_factory=list)
    ts: str = ""
    schema: str = "knowledge-hub.crawl.pipeline.run.result.v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "runId": self.run_id,
            "jobId": self.job_id,
            "status": self.status,
            "profile": self.profile,
            "sourcePolicy": self.source_policy,
            "storageRoot": self.storage_root,
            "requested": self.requested,
            "processed": self.processed,
            "normalized": self.normalized,
            "indexed": self.indexed,
            "pendingDomain": self.pending_domain,
            "failed": self.failed,
            "skipped": self.skipped,
            "dedupeRate": self.dedupe_rate,
            "retryRate": self.retry_rate,
            "memoryPeakRatio": self.memory_peak_ratio,
            "recordsPerMin": self.records_per_min,
            "p50StepLatencyMs": self.p50_step_latency_ms,
            "cursor": self.cursor,
            "warnings": list(self.warnings),
            "schemaErrors": list(self.schema_errors),
            "ts": self.ts,
        }


@dataclass
class ClaimCandidate:
    """Typed claim candidate payload shared across web/paper extractors."""

    claim_text: str
    subject: str
    predicate: str
    object_value: str
    evidence: str
    llm_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "subject": self.subject,
            "predicate": self.predicate,
            "object_value": self.object_value,
            "evidence": self.evidence,
            "llm_confidence": self.llm_confidence,
        }


@dataclass
class FeatureSnapshot:
    """Typed feature snapshot payload used at feature boundaries."""

    topic_slug: str
    feature_kind: str
    feature_key: str
    feature_name: str = ""
    entity_id: str = ""
    note_id: str = ""
    record_id: str = ""
    canonical_url: str = ""
    source_item_id: str = ""
    freshness_score: float = 0.0
    importance_score: float = 0.0
    support_doc_count: int = 0
    relation_degree: float = 0.0
    claim_density: float = 0.0
    source_trust_score: float = 0.0
    concept_activity_score: float = 0.0
    contradiction_score: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_slug": self.topic_slug,
            "feature_kind": self.feature_kind,
            "feature_key": self.feature_key,
            "feature_name": self.feature_name,
            "entity_id": self.entity_id,
            "note_id": self.note_id,
            "record_id": self.record_id,
            "canonical_url": self.canonical_url,
            "source_item_id": self.source_item_id,
            "freshness_score": self.freshness_score,
            "importance_score": self.importance_score,
            "support_doc_count": self.support_doc_count,
            "relation_degree": self.relation_degree,
            "claim_density": self.claim_density,
            "source_trust_score": self.source_trust_score,
            "concept_activity_score": self.concept_activity_score,
            "contradiction_score": self.contradiction_score,
            "payload": self.payload,
        }


@dataclass
class OntologyExtractionResult:
    """Typed ontology extraction result for ingest/extraction boundaries."""

    run_id: str
    source_type: str
    source_id: str
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    claims: List[ClaimCandidate] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "entities": self.entities,
            "relations": self.relations,
            "claims": [candidate.to_dict() for candidate in self.claims],
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass
class LearningGraphCandidate:
    """Typed candidate payload for learning graph generation and review."""

    topic_slug: str
    item_type: str
    payload: Dict[str, Any]
    confidence: float
    provenance: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_slug": self.topic_slug,
            "item_type": self.item_type,
            "payload": self.payload,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "evidence": self.evidence,
            "status": self.status,
        }
