from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from knowledge_hub.knowledge.feature_store import FeatureStore
from knowledge_hub.infrastructure.persistence.stores import (
    ClaimCardV1Store,
    ClaimStore,
    CrawlPipelineStore,
    DocumentMemoryStore,
    EntityResolutionStore,
    EpistemicStore,
    KoNoteStore,
    LearningGraphStore,
    LearningStore,
    MCPJobStore,
    MemoryRelationStore,
    MigrationManager,
    NoteStore,
    OntologyProfileStore,
    OntologyStore,
    OpsActionQueueStore,
    OpsActionReceiptStore,
    PaperCardV2Store,
    PaperMemoryStore,
    PaperStore,
    QualityModeStore,
    RAGAnswerLogStore,
    SyncConflictStore,
    VaultCardV2Store,
    WebCardV2Store,
)

log = logging.getLogger("khub.database")

SQLITE_BUSY_TIMEOUT_MS = 5000
_LEGACY_KG_TABLE_SUFFIX = "kg" + "_relations"
_ENSURE_LEGACY_KG_TABLE_SCHEMA = "ensure_legacy_" + _LEGACY_KG_TABLE_SUFFIX + "_schema"
_LIST_LEGACY_KG_TABLE = "list_" + _LEGACY_KG_TABLE_SUFFIX


def _delegate_map(
    store_attr: str,
    same_name: tuple[str, ...] = (),
    **aliases: str,
) -> dict[str, tuple[str, str]]:
    mapping = {name: (store_attr, name) for name in same_name}
    mapping.update({name: (store_attr, target) for name, target in aliases.items()})
    return mapping


DELEGATED_METHODS: dict[str, tuple[str, str]] = {}
DELEGATED_METHODS.update(
    _delegate_map(
        "note_store",
        (
            "upsert_note",
            "get_note",
            "list_notes",
            "delete_note",
            "delete_notes",
            "merge_note_metadata",
            "search_notes",
            "get_note_tags",
            "list_tags",
            "get_links",
            "list_para_categories",
            "get_para_stats",
            "get_graph_data",
            "get_stats",
            "replace_note_tags",
            "clear_links_for_source",
            "replace_links_for_source",
            "delete_links_for_note_ids",
        ),
        ensure_tag="ensure_tag",
        add_note_tag="add_note_tag",
        add_link="add_link",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "memory_relation_store",
        upsert_memory_relation="upsert_relation",
        get_memory_relation="get_relation",
        list_memory_relations="list_relations",
        delete_memory_relations_for_node="delete_relations_for_node",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "document_memory_store",
        replace_document_memory_units="replace_units",
        get_document_memory_unit="get_unit",
        get_document_memory_summary="get_document_summary",
        list_document_memory_units="list_document_units",
        search_document_memory_units="search_units",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "paper_store",
        (
            "upsert_paper",
            "get_paper",
            "list_papers",
            "search_papers",
            "get_concept_papers",
            "get_paper_concepts",
            "update_paper_lane_metadata",
        ),
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "paper_card_v2_store",
        upsert_paper_card_v2="upsert_card",
        get_paper_card_v2="get_card",
        list_paper_cards_v2="list_cards",
        search_paper_cards_v2="search_cards",
        replace_paper_card_claim_refs_v2="replace_claim_refs",
        list_paper_card_claim_refs_v2="list_claim_refs",
        replace_evidence_anchors_v2="replace_anchors",
        list_evidence_anchors_v2="list_anchors",
        replace_paper_card_entity_refs_v2="replace_entity_refs",
        list_paper_card_entity_refs_v2="list_entity_refs",
        list_paper_cards_v2_by_entity_ids="list_cards_by_entity_ids",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "web_card_v2_store",
        upsert_web_card_v2="upsert_card",
        get_web_card_v2="get_card",
        get_web_card_v2_by_url="get_card_by_url",
        list_web_cards_v2="list_cards",
        search_web_cards_v2="search_cards",
        replace_web_card_claim_refs_v2="replace_claim_refs",
        list_web_card_claim_refs_v2="list_claim_refs",
        replace_web_evidence_anchors_v2="replace_anchors",
        list_web_evidence_anchors_v2="list_anchors",
        replace_web_card_entity_refs_v2="replace_entity_refs",
        list_web_card_entity_refs_v2="list_entity_refs",
        list_web_cards_v2_by_entity_ids="list_cards_by_entity_ids",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "vault_card_v2_store",
        upsert_vault_card_v2="upsert_card",
        get_vault_card_v2="get_card",
        list_vault_cards_v2="list_cards",
        search_vault_cards_v2="search_cards",
        replace_vault_card_claim_refs_v2="replace_claim_refs",
        list_vault_card_claim_refs_v2="list_claim_refs",
        replace_vault_evidence_anchors_v2="replace_anchors",
        list_vault_evidence_anchors_v2="list_anchors",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "paper_memory_store",
        upsert_paper_memory_card="upsert_card",
        get_paper_memory_card="get_card",
        list_paper_memory_cards="list_cards",
        search_paper_memory_cards="search_cards",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "ontology_store",
        (
            "upsert_concept",
            "get_concept",
            "legacy_lookup_concept_by_name",
            "list_concepts",
            "add_alias",
            "get_aliases",
            "resolve_concept",
            "delete_concept",
            "upsert_ontology_entity",
            "get_ontology_entity",
            "list_ontology_entities",
            "add_entity_alias",
            "get_entity_aliases",
            "resolve_entity",
            "delete_ontology_entity",
            "migrate_concepts_to_entities",
            "create_concepts_view",
            "sync_paper_entities",
            "add_relation",
            "get_relations",
            "list_relations",
            "list_predicate_validation_issues",
            "list_ontology_claims",
            "list_ontology_events",
            "upsert_predicate",
            "get_predicate",
            "list_predicates",
            "add_ontology_pending",
            "get_ontology_pending",
            "list_ontology_pending",
            "update_ontology_pending_status",
            "add_web_ontology_pending",
            "get_web_ontology_pending",
            "list_web_ontology_pending",
            "get_related_concepts",
            "count_relations",
            "count_concepts",
            "get_kg_stats",
        ),
        get_concept_by_name="legacy_lookup_concept_by_name",
        list_pending_ontology="list_ontology_pending",
        update_web_ontology_pending_status="update_ontology_pending_status",
    )
)
DELEGATED_METHODS[_ENSURE_LEGACY_KG_TABLE_SCHEMA] = ("ontology_store", _ENSURE_LEGACY_KG_TABLE_SCHEMA)
DELEGATED_METHODS[_LIST_LEGACY_KG_TABLE] = ("ontology_store", _LIST_LEGACY_KG_TABLE)
DELEGATED_METHODS.update(
    _delegate_map(
        "claim_card_store",
        (
            "upsert_claim_card",
            "get_claim_card",
            "list_claim_cards",
            "delete_claim_cards",
            "replace_claim_card_source_refs",
            "list_claim_card_source_refs",
            "replace_claim_card_alignment_refs",
            "list_claim_card_alignment_refs",
            "upsert_normalization_alias",
            "list_normalization_aliases",
        ),
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "claim_store",
        (
            "upsert_claim_normalization",
            "get_claim_normalization",
            "list_claim_normalizations",
            "upsert_claim",
            "get_claim",
            "list_claims",
            "list_claims_by_note",
            "list_claims_by_record",
            "list_claims_by_entity",
            "delete_claim",
        ),
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "learning_store",
        (
            "upsert_learning_session",
            "get_learning_session",
            "list_learning_sessions",
            "replace_learning_session_edges",
            "list_learning_session_edges",
            "upsert_learning_progress",
            "get_learning_progress",
            "append_learning_event",
            "list_learning_events",
        ),
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "learning_graph_store",
        upsert_learning_graph_node="upsert_node",
        list_learning_graph_nodes="list_nodes",
        upsert_learning_graph_edge="upsert_edge",
        list_learning_graph_edges="list_edges",
        upsert_learning_graph_path="upsert_path",
        get_latest_learning_graph_path="get_latest_path",
        upsert_learning_graph_resource_link="upsert_resource_link",
        list_learning_graph_resource_links="list_resource_links",
        add_learning_graph_pending="add_pending",
        list_learning_graph_pending="list_pending",
        get_learning_graph_pending="get_pending",
        set_learning_graph_pending_status="set_pending_status",
        append_learning_graph_event="append_event",
        list_learning_graph_events="list_events",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "feature_store",
        upsert_feature_snapshot="upsert_snapshot",
        get_feature_snapshot="get_snapshot",
        find_source_feature_snapshot="find_source_snapshot",
        list_feature_snapshots="list_snapshots",
        list_top_feature_snapshots="list_top",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "sync_conflict_store",
        add_foundry_sync_conflict="add_conflict",
        get_foundry_sync_conflict="get_conflict",
        list_foundry_sync_conflicts="list_conflicts",
        update_foundry_sync_conflict_status="update_conflict_status",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "mcp_job_store",
        (
            "create_mcp_job",
            "update_mcp_job",
            "get_mcp_job",
            "list_mcp_jobs",
            "cancel_mcp_job",
        ),
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "crawl_pipeline_store",
        (
            "upsert_crawl_domain_policy",
            "get_crawl_domain_policy",
            "list_crawl_domain_policy",
            "create_crawl_pipeline_job",
            "get_crawl_pipeline_job",
            "update_crawl_pipeline_job",
            "upsert_crawl_pipeline_record",
            "get_crawl_pipeline_record",
            "list_crawl_pipeline_records",
            "count_crawl_pipeline_records",
            "update_crawl_pipeline_record_state",
            "upsert_crawl_pipeline_checkpoint",
            "list_crawl_pipeline_checkpoints",
            "append_crawl_pipeline_metric",
            "list_crawl_pipeline_metrics",
            "get_latest_crawl_pipeline_job",
        ),
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "ko_note_store",
        create_ko_note_run="create_run",
        get_ko_note_run="get_run",
        update_ko_note_run="update_run",
        add_ko_note_item="add_item",
        get_ko_note_item="get_item",
        list_ko_note_items="list_items",
        list_existing_ko_note_items="list_existing_items",
        update_ko_note_item_status="update_item_status",
        update_ko_note_item_payload="update_item_payload",
        find_ko_note_item_by_final_path="find_item_by_final_path",
        get_latest_ko_note_run="get_latest_run",
        list_ko_note_runs="list_runs",
        list_stale_ko_note_runs="list_stale_runs",
        create_ko_note_enrichment_run="create_enrichment_run",
        get_ko_note_enrichment_run="get_enrichment_run",
        update_ko_note_enrichment_run="update_enrichment_run",
        add_ko_note_enrichment_item="add_enrichment_item",
        list_ko_note_enrichment_items="list_enrichment_items",
        update_ko_note_enrichment_item="update_enrichment_item",
        find_matching_ko_note_enrichment_item="find_matching_enrichment_item",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "rag_answer_log_store",
        add_rag_answer_log="add_log",
        list_rag_answer_logs="list_logs",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "ops_action_queue_store",
        get_ops_action="get_action",
        get_ops_action_by_identity="get_action_by_identity",
        upsert_ops_action="upsert_action",
        list_ops_actions="list_actions",
        count_ops_actions="action_counts",
        set_ops_action_status="set_action_status",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "ops_action_receipt_store",
        get_ops_action_receipt="get_receipt",
        create_ops_action_receipt="create_receipt",
        update_ops_action_receipt="update_receipt",
        list_ops_action_receipts="list_receipts",
        get_latest_ops_action_receipt="latest_receipt",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "quality_mode_store",
        record_quality_mode_usage="record_usage",
        get_quality_mode_monthly_spend="get_monthly_spend",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "entity_resolution_store",
        (
            "add_entity_merge_proposal",
            "list_entity_merge_proposals",
            "get_entity_merge_proposal",
            "update_entity_merge_proposal_status",
            "reject_entity_merge_proposal",
            "apply_entity_merge_proposal",
        ),
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "ontology_profile_store",
        set_active_ontology_profile="set_active_profile",
        get_active_ontology_profile="get_active_profile",
        list_active_ontology_profiles="list_active_profiles",
        set_ontology_profile_runtime_json="set_runtime_json",
        get_ontology_profile_runtime_json="get_runtime_json",
        add_ontology_profile_proposal="add_profile_proposal",
        get_ontology_profile_proposal="get_profile_proposal",
        list_ontology_profile_proposals="list_profile_proposals",
        update_ontology_profile_proposal_status="update_profile_proposal_status",
        add_ontology_profile_overlay="add_profile_overlay",
        list_ontology_profile_overlays="list_profile_overlays",
    )
)
DELEGATED_METHODS.update(
    _delegate_map(
        "epistemic_store",
        (
            "upsert_belief",
            "get_belief",
            "list_beliefs",
            "list_beliefs_by_claim_ids",
            "review_belief",
            "upsert_decision",
            "get_decision",
            "list_decisions",
            "review_decision",
            "record_outcome",
            "get_outcome",
            "list_outcomes",
        ),
    )
)
_DELEGATED_METHODS = DELEGATED_METHODS


class StoreRegistry:
    """Owns SQLite bootstrapping while preserving facade-compatible access."""

    def __init__(
        self,
        db_path: str,
        enable_event_store: bool = True,
        *,
        bootstrap: bool = True,
        read_only: bool = False,
    ):
        self.db_path = Path(db_path)
        self.read_only = bool(read_only)
        self.bootstrap = bool(bootstrap) and not self.read_only
        if not self.read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        connect_target = str(self.db_path)
        connect_kwargs: dict[str, Any] = {"timeout": SQLITE_BUSY_TIMEOUT_MS / 1000}
        if self.read_only:
            connect_target = f"file:{self.db_path}?mode=ro"
            connect_kwargs["uri"] = True

        self.conn = sqlite3.connect(connect_target, **connect_kwargs)
        self.conn.row_factory = sqlite3.Row
        if not self.read_only:
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")

        self.note_store = NoteStore(self.conn)
        self.paper_store = PaperStore(self.conn)
        self.document_memory_store = DocumentMemoryStore(self.conn)
        self.paper_card_v2_store = PaperCardV2Store(self.conn)
        self.web_card_v2_store = WebCardV2Store(self.conn)
        self.vault_card_v2_store = VaultCardV2Store(self.conn)
        self.paper_memory_store = PaperMemoryStore(self.conn)
        self.memory_relation_store = MemoryRelationStore(self.conn)
        self.ontology_store = OntologyStore(self.conn, event_store=None, db_path=self.db_path)
        self.ontology_profile_store = OntologyProfileStore(self.conn)
        self.epistemic_store = EpistemicStore(self.conn)
        self.learning_store = LearningStore(self.conn)
        self.learning_graph_store = LearningGraphStore(self.conn)
        self.mcp_job_store = MCPJobStore(self.conn)
        self.ko_note_store = KoNoteStore(self.conn)
        self.sync_conflict_store = SyncConflictStore(self.conn)
        self.feature_store = FeatureStore(self.conn)
        self.claim_store = ClaimStore(self.conn, event_store=None)
        self.claim_card_store = ClaimCardV1Store(self.conn)
        self.crawl_pipeline_store = CrawlPipelineStore(self.conn)
        self.quality_mode_store = QualityModeStore(self.conn)
        self.rag_answer_log_store = RAGAnswerLogStore(self.conn)
        self.ops_action_queue_store = OpsActionQueueStore(self.conn)
        self.ops_action_receipt_store = OpsActionReceiptStore(self.conn)
        self.entity_resolution_store = EntityResolutionStore(self.conn, self)

        if self.bootstrap:
            self._init_tables()
            self.migration_manager = MigrationManager(self.conn, self.db_path)
            self.migration_manager.apply_pending_migrations()
        else:
            self.migration_manager = None

        self.event_store = None
        if enable_event_store and not self.read_only:
            try:
                from knowledge_hub.infrastructure.persistence.stores.event_store import EventStore

                jsonl_path = self.db_path.parent / "ontology_events.jsonl"
                self.event_store = EventStore(self, jsonl_path)
            except Exception as error:
                log.error("EventStore initialization failed for %s: %s", self.db_path, error)

        self.ontology_store.event_store = self.event_store
        self.claim_store.event_store = self.event_store
        if self.bootstrap:
            self.learning_store.ensure_schema()
            self.learning_graph_store.ensure_schema()
            self.mcp_job_store.ensure_schema()
            self.ko_note_store.ensure_schema()
            self.sync_conflict_store.ensure_schema()
            self.feature_store.ensure_schema()
            self.crawl_pipeline_store.ensure_schema()
            self.quality_mode_store.ensure_schema()
            self.rag_answer_log_store.ensure_schema()
            self.document_memory_store.ensure_schema()
            self.paper_card_v2_store.ensure_schema()
            self.web_card_v2_store.ensure_schema()
            self.vault_card_v2_store.ensure_schema()
            self.paper_memory_store.ensure_schema()
            self.memory_relation_store.ensure_schema()
            self.ops_action_queue_store.ensure_schema()
            self.ops_action_receipt_store.ensure_schema()
            self.entity_resolution_store.ensure_schema()
            self.ontology_profile_store.ensure_schema()
            self.epistemic_store.ensure_schema()
            self.claim_store.ensure_schema()
            self.claim_card_store.ensure_schema()
            self.learning_store.ensure_learning_events_schema()
            self.ontology_store.ensure_core_predicates()
            self.ontology_store.run_core_migration()

    @contextmanager
    def transaction(self):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _init_tables(self) -> None:
        self.note_store.ensure_schema()
        self.paper_store.ensure_schema()
        self.ontology_store.ensure_schema()
        self.claim_store.ensure_schema()
        self._ensure_default_para()

    def _ensure_default_para(self) -> None:
        self.note_store.ensure_default_para()

    def set_system_meta(self, key: str, value: str) -> None:
        self.note_store.set_system_meta(key, value)

    def get_system_meta(self, key: str, default: str = "") -> str:
        return self.note_store.get_system_meta(key, default)

    def __getattr__(self, name: str):
        delegated = DELEGATED_METHODS.get(name)
        if delegated is None:
            raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")
        store_attr, method_name = delegated
        return getattr(getattr(self, store_attr), method_name)

    def apply_ontology_pending(self, pending_id: int) -> Optional[dict]:
        item = self.get_ontology_pending(int(pending_id))
        if not item or str(item.get("status", "")) != "pending":
            return None

        pending_type = str(item.get("pending_type", ""))
        reason = item.get("reason_json") if isinstance(item.get("reason_json"), dict) else {}
        applied = False
        if pending_type == "predicate_ext":
            predicate_id = str(item.get("predicate_id") or reason.get("predicate_id") or "").strip()
            if predicate_id:
                self.upsert_predicate(
                    predicate_id=predicate_id,
                    parent_predicate_id=str(reason.get("parent_predicate_id", "")).strip() or None,
                    status="approved_ext",
                    description=str(reason.get("description", "")),
                    source="pending_apply",
                )
                applied = True
        elif pending_type == "concept":
            entity_id = str(item.get("source_entity_id", "")).strip()
            display_name = str(reason.get("display_name") or reason.get("original_term") or entity_id).strip()
            if entity_id and display_name:
                self.upsert_ontology_entity(
                    entity_id=entity_id,
                    entity_type="concept",
                    canonical_name=display_name,
                    source="pending_apply",
                )
                for alias in reason.get("aliases", []) if isinstance(reason.get("aliases"), list) else []:
                    self.add_entity_alias(str(alias), entity_id)
                applied = True
        elif pending_type == "relation":
            source_entity_id = str(item.get("source_entity_id", "")).strip()
            target_entity_id = str(item.get("target_entity_id", "")).strip()
            predicate_id = str(item.get("predicate_id", "")).strip() or "related_to"
            if source_entity_id and target_entity_id:
                self.add_relation(
                    source_type=str(reason.get("source_type", "concept") or "concept"),
                    source_id=str(reason.get("source_id", source_entity_id) or source_entity_id),
                    relation=predicate_id,
                    target_type=str(reason.get("target_type", "concept") or "concept"),
                    target_id=str(reason.get("target_id", target_entity_id) or target_entity_id),
                    evidence_text=json.dumps(
                        {
                            "source": str(reason.get("source", "pending_apply") or "pending_apply"),
                            "relation_norm": predicate_id,
                            "evidence_ptrs": item.get("evidence_ptrs_json") or [],
                            "reason": reason,
                        },
                        ensure_ascii=False,
                    ),
                    confidence=float(item.get("confidence", 0.0)),
                )
                applied = True
        elif pending_type == "claim":
            claim_id = str(reason.get("claim_id") or f"claim_pending_{pending_id}").strip()
            claim_text = str(reason.get("claim_text", "")).strip()
            subject_entity_id = str(reason.get("subject_entity_id", item.get("source_entity_id", ""))).strip()
            predicate = str(reason.get("predicate", item.get("predicate_id", ""))).strip()
            object_entity_id = str(reason.get("object_entity_id", item.get("target_entity_id", ""))).strip() or None
            object_literal = str(reason.get("object_literal", "")).strip() or None
            if claim_text and subject_entity_id and predicate:
                self.upsert_claim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    subject_entity_id=subject_entity_id,
                    predicate=predicate,
                    object_entity_id=object_entity_id,
                    object_literal=object_literal,
                    confidence=float(item.get("confidence", 0.0)),
                    evidence_ptrs=item.get("evidence_ptrs_json") if isinstance(item.get("evidence_ptrs_json"), list) else [],
                    source="pending_apply",
                )
                applied = True

        if not applied:
            return None
        self.update_ontology_pending_status(int(pending_id), "approved")
        return self.get_ontology_pending(int(pending_id))

    def apply_pending_ontology(self, pending_id: int) -> Optional[dict]:
        return self.apply_ontology_pending(pending_id)

    def reject_ontology_pending(self, pending_id: int) -> Optional[dict]:
        item = self.get_ontology_pending(int(pending_id))
        if not item or str(item.get("status", "")) != "pending":
            return None
        self.update_ontology_pending_status(int(pending_id), "rejected")
        return self.get_ontology_pending(int(pending_id))

    def close(self) -> None:
        self.conn.close()


SQLiteStoreRegistry = StoreRegistry

__all__ = ["DELEGATED_METHODS", "SQLITE_BUSY_TIMEOUT_MS", "SQLiteStoreRegistry", "StoreRegistry", "_DELEGATED_METHODS"]
