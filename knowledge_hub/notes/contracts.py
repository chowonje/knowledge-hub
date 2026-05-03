"""Notes-domain repository contracts."""

from __future__ import annotations

from typing import Any, Optional, Protocol

from knowledge_hub.knowledge.contracts import (
    ClaimRepository,
    FeatureRepository,
    NoteRepository,
    OntologyRepository,
)


class CrawlPipelineRepository(Protocol):
    def create_crawl_pipeline_job(
        self,
        job_id: str,
        run_id: str,
        profile: str,
        source_policy: str,
        storage_root: str,
        source: str = "web",
        topic: str = "",
        sources: list[str] | None = None,
        status: str = "running",
    ) -> None: ...
    def get_crawl_pipeline_job(self, job_id: str) -> Optional[dict[str, Any]]: ...
    def list_crawl_pipeline_records(
        self,
        job_id: str,
        *,
        state: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]: ...
    def list_crawl_domain_policy(
        self,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]: ...


class MaterializationRepository(
    NoteRepository,
    ClaimRepository,
    OntologyRepository,
    CrawlPipelineRepository,
    Protocol,
):
    def create_ko_note_run(self, **kwargs: Any) -> None: ...
    def update_ko_note_run(self, run_id: str, **updates: Any) -> bool: ...
    def get_ko_note_run(self, run_id: str) -> Optional[dict[str, Any]]: ...
    def get_latest_ko_note_run(self) -> Optional[dict[str, Any]]: ...
    def list_ko_note_runs(self, limit: int = 20) -> list[dict[str, Any]]: ...
    def list_stale_ko_note_runs(
        self,
        *,
        status: str = "running",
        updated_before_seconds: int,
        limit: int = 200,
    ) -> list[dict[str, Any]]: ...
    def add_ko_note_item(self, **kwargs: Any) -> int: ...
    def get_ko_note_item(self, item_id: int) -> Optional[dict[str, Any]]: ...
    def list_ko_note_items(
        self,
        *,
        run_id: str,
        item_type: str | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]: ...
    def list_existing_ko_note_items(
        self,
        *,
        item_type: str | None = None,
        statuses: tuple[str, ...] = ("staged", "approved", "applied"),
        limit: int = 5000,
    ) -> list[dict[str, Any]]: ...
    def find_ko_note_item_by_final_path(
        self,
        *,
        final_path: str,
        item_type: str | None = None,
        statuses: tuple[str, ...] = ("approved", "applied"),
    ) -> Optional[dict[str, Any]]: ...
    def update_ko_note_item_status(
        self,
        item_id: int,
        *,
        status: str,
        final_path: str | None = None,
        staging_path: str | None = None,
    ) -> bool: ...
    def update_ko_note_item_payload(
        self,
        item_id: int,
        *,
        payload: dict[str, Any],
        title_en: str | None = None,
        title_ko: str | None = None,
        staging_path: str | None = None,
        final_path: str | None = None,
    ) -> bool: ...


class EnrichmentRepository(MaterializationRepository, FeatureRepository, Protocol):
    def get_crawl_pipeline_record(self, job_id: str, record_id: str) -> Optional[dict[str, Any]]: ...
    def get_quality_mode_monthly_spend(self, month_key: str | None = None) -> float: ...
    def record_quality_mode_usage(
        self,
        item_kind: str,
        route: str,
        estimated_cost_usd: float,
        topic_slug: str = "",
    ) -> None: ...
    def create_ko_note_enrichment_run(self, **kwargs: Any) -> None: ...
    def get_ko_note_enrichment_run(self, run_id: str) -> Optional[dict[str, Any]]: ...
    def update_ko_note_enrichment_run(self, run_id: str, **updates: Any) -> bool: ...
    def add_ko_note_enrichment_item(self, **kwargs: Any) -> int: ...
    def get_ko_note_enrichment_item(self, item_id: int) -> Optional[dict[str, Any]]: ...
    def update_ko_note_enrichment_item(self, item_id: int, **updates: Any) -> bool: ...
    def list_ko_note_enrichment_items(
        self,
        *,
        run_id: str,
        item_type: str | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]: ...
    def find_matching_ko_note_enrichment_item(
        self,
        *,
        item_type: str,
        note_item_id: int = 0,
        target_path: str = "",
        evidence_pack_hash: str = "",
        model_fingerprint: str = "",
    ) -> Optional[dict[str, Any]]: ...
    def add_rag_answer_log(self, **kwargs: Any) -> int: ...
    def list_rag_answer_logs(self, *, limit: int = 100, days: int = 0) -> list[dict[str, Any]]: ...
    def get_ops_action(self, action_id: str) -> Optional[dict[str, Any]]: ...
    def upsert_ops_action(self, **kwargs: Any) -> dict[str, Any]: ...
    def get_ops_action_receipt(self, receipt_id: str) -> Optional[dict[str, Any]]: ...
    def create_ops_action_receipt(self, **kwargs: Any) -> dict[str, Any]: ...
    def update_ops_action_receipt(self, receipt_id: str, **kwargs: Any) -> Optional[dict[str, Any]]: ...
    def list_ops_action_receipts(self, *, action_id: str, limit: int = 20) -> list[dict[str, Any]]: ...
    def get_latest_ops_action_receipt(self, action_id: str) -> Optional[dict[str, Any]]: ...
    def list_ops_actions(
        self,
        *,
        status: str | None = None,
        scope: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]: ...
    def count_ops_actions(self) -> dict[str, int]: ...
    def set_ops_action_status(
        self,
        action_id: str,
        *,
        status: str,
        actor: str = "",
        note: str = "",
        changed_at: str | None = None,
    ) -> Optional[dict[str, Any]]: ...


__all__ = [
    "CrawlPipelineRepository",
    "EnrichmentRepository",
    "MaterializationRepository",
]
