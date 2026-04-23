"""Canonical infrastructure store import surface."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ClaimCardV1Store": "knowledge_hub.infrastructure.persistence.stores.claim_card_v1_store",
    "ClaimStore": "knowledge_hub.infrastructure.persistence.stores.claim_store",
    "CrawlPipelineStore": "knowledge_hub.infrastructure.persistence.stores.crawl_pipeline_store",
    "DocumentMemoryStore": "knowledge_hub.infrastructure.persistence.stores.document_memory_store",
    "EntityResolutionStore": "knowledge_hub.infrastructure.persistence.stores.entity_resolution_store",
    "EpistemicStore": "knowledge_hub.infrastructure.persistence.stores.epistemic_store",
    "EventStore": "knowledge_hub.infrastructure.persistence.stores.event_store",
    "KoNoteStore": "knowledge_hub.infrastructure.persistence.stores.ko_note_store",
    "LearningGraphStore": "knowledge_hub.infrastructure.persistence.stores.learning_graph_store",
    "LearningStore": "knowledge_hub.infrastructure.persistence.stores.learning_store",
    "MCPJobStore": "knowledge_hub.infrastructure.persistence.stores.mcp_job_store",
    "MemoryRelationStore": "knowledge_hub.infrastructure.persistence.stores.memory_relation_store",
    "MigrationManager": "knowledge_hub.infrastructure.persistence.stores.migrations",
    "NoteStore": "knowledge_hub.infrastructure.persistence.stores.note_store",
    "OntologyProfileStore": "knowledge_hub.infrastructure.persistence.stores.ontology_profile_store",
    "OntologyStore": "knowledge_hub.infrastructure.persistence.stores.ontology_store",
    "OpsActionQueueStore": "knowledge_hub.infrastructure.persistence.stores.ops_action_queue_store",
    "OpsActionReceiptStore": "knowledge_hub.infrastructure.persistence.stores.ops_action_receipt_store",
    "PaperCardV2Store": "knowledge_hub.infrastructure.persistence.stores.paper_card_v2_store",
    "PaperMemoryStore": "knowledge_hub.infrastructure.persistence.stores.paper_memory_store",
    "PaperStore": "knowledge_hub.infrastructure.persistence.stores.paper_store",
    "QualityModeStore": "knowledge_hub.infrastructure.persistence.stores.quality_mode_store",
    "RAGAnswerLogStore": "knowledge_hub.infrastructure.persistence.stores.rag_answer_log_store",
    "SectionCardV1Store": "knowledge_hub.infrastructure.persistence.stores.section_card_v1_store",
    "SyncConflictStore": "knowledge_hub.infrastructure.persistence.stores.sync_conflict_store",
    "VaultCardV2Store": "knowledge_hub.infrastructure.persistence.stores.vault_card_v2_store",
    "WebCardV2Store": "knowledge_hub.infrastructure.persistence.stores.web_card_v2_store",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
