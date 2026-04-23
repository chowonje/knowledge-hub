"""Backward-compatible repository contract import surface."""

from __future__ import annotations

from knowledge_hub.knowledge.contracts import (
    ClaimRepository,
    FeatureComputationRepository,
    FeatureRepository,
    NoteRepository,
    OntologyRepository,
)
from knowledge_hub.learning.contracts import LearningGraphDataRepository, LearningServiceRepository
from knowledge_hub.notes.contracts import CrawlPipelineRepository, EnrichmentRepository, MaterializationRepository

__all__ = [
    "ClaimRepository",
    "CrawlPipelineRepository",
    "EnrichmentRepository",
    "FeatureComputationRepository",
    "FeatureRepository",
    "LearningGraphDataRepository",
    "LearningServiceRepository",
    "MaterializationRepository",
    "NoteRepository",
    "OntologyRepository",
]
