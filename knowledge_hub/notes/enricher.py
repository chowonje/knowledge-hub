"""Content-focused enrichment pass for Korean Obsidian notes."""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.notes.contracts import EnrichmentRepository
from knowledge_hub.knowledge.quality_mode import (
    estimate_quality_mode_cost,
    infer_quality_topic,
    resolve_quality_mode_route,
)
from knowledge_hub.notes import enrichment_support as _enrichment_support
from knowledge_hub.notes.composer import compose_concept_note_payload, compose_source_note_payload
from knowledge_hub.notes.koreanizer import Koreanizer
from knowledge_hub.notes.materializer import (
    KoNoteMaterializer,
    _attach_concept_quality,
    _attach_review_payload,
    _attach_source_quality,
    _date_only,
    _resolve_existing_path,
)
from knowledge_hub.notes.models import KoNoteQuality, KoNoteReview
from knowledge_hub.notes.templates import (
    build_visible_frontmatter,
    render_concept_note,
    render_source_note,
)
from knowledge_hub.notes.scoring import update_review_remediation


ENRICHMENT_VERSION = "v1"
_SOURCE_QUALITY_PRIORITY = {
    "ok": 0,
    "needs_review": 1,
    "reject": 2,
    "unscored": 3,
}


@dataclass
class SourceEvidencePack:
    note_item_id: int
    title_en: str
    title_ko: str
    source_url: str
    domain: str
    metadata_lines: list[str]
    entity_names: list[str]
    relation_lines: list[str]
    claim_lines: list[str]
    related_concepts: list[str]
    key_excerpts_en: list[str]
    content_text: str
    document_type: str
    thesis: str
    top_claims: list[str]
    core_concepts: list[str]
    contributions: list[str]
    methodology: list[str]
    results_or_findings: list[str]
    limitations: list[str]
    representative_sources: list[str]
    candidate_score: float
    translation_level: str

    def stable_hash(self) -> str:
        payload = {
            "title_en": self.title_en,
            "source_url": self.source_url,
            "domain": self.domain,
            "metadata_lines": self.metadata_lines,
            "entity_names": self.entity_names,
            "relation_lines": self.relation_lines,
            "claim_lines": self.claim_lines,
            "related_concepts": self.related_concepts,
            "key_excerpts_en": self.key_excerpts_en,
            "content_text": self.content_text,
            "document_type": self.document_type,
            "thesis": self.thesis,
            "top_claims": self.top_claims,
            "core_concepts": self.core_concepts,
            "contributions": self.contributions,
            "methodology": self.methodology,
            "results_or_findings": self.results_or_findings,
            "limitations": self.limitations,
            "representative_sources": self.representative_sources,
            "candidate_score": self.candidate_score,
            "translation_level": self.translation_level,
        }
        return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class ConceptEvidencePack:
    note_item_id: int
    entity_id: str
    canonical_name: str
    aliases: list[str]
    relation_lines: list[str]
    related_concepts: list[str]
    compressed_support_docs: list[dict[str, Any]]
    existing_summary: str
    candidate_score: float
    translation_level: str

    def stable_hash(self) -> str:
        payload = {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "aliases": self.aliases,
            "relation_lines": self.relation_lines,
            "related_concepts": self.related_concepts,
            "compressed_support_docs": self.compressed_support_docs,
            "candidate_score": self.candidate_score,
            "translation_level": self.translation_level,
        }
        return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class KoNoteEnricher:
    def __init__(
        self,
        config: Config,
        *,
        sqlite_db: EnrichmentRepository | None = None,
        sqlite_db_factory: Callable[[], EnrichmentRepository] | None = None,
    ):
        self.config = config
        self._sqlite_db_factory = sqlite_db_factory or (lambda: SQLiteDatabase(self.config.sqlite_path))
        self.sqlite_db: EnrichmentRepository = sqlite_db or self._sqlite_db_factory()
        self.materializer = KoNoteMaterializer(
            config,
            sqlite_db=self.sqlite_db,
            sqlite_db_factory=self._sqlite_db_factory,
        )
        self.koreanizer: Koreanizer | None = None
        self._koreanizer_cache: dict[tuple[bool, str, int | None, bool], Koreanizer] = {}
        self._source_quality_index: dict[str, dict[str, Any]] | None = None

    def _get_koreanizer_for(
        self,
        *,
        allow_external: bool,
        llm_mode: str,
        local_timeout_sec: int | None,
        api_fallback_on_timeout: bool,
    ) -> Koreanizer:
        key = (allow_external, llm_mode, local_timeout_sec, api_fallback_on_timeout)
        cached = self._koreanizer_cache.get(key)
        if cached is not None:
            return cached
        cached = Koreanizer(
            self.config,
            allow_external=allow_external,
            llm_mode=llm_mode,
            local_timeout_sec=local_timeout_sec,
            api_fallback_on_timeout=api_fallback_on_timeout,
        )
        self._koreanizer_cache[key] = cached
        return cached

    def _get_record(self, *, job_id: str, record_id: str) -> dict[str, Any] | None:
        return _enrichment_support.get_record(self, job_id=job_id, record_id=record_id)

    def _frontmatter_title(self, content: str) -> str:
        return _enrichment_support.frontmatter_title(content)

    def _build_source_pack(self, item: dict[str, Any]) -> SourceEvidencePack:
        return SourceEvidencePack(**_enrichment_support.build_source_pack_data(self, item))

    def _get_source_quality_index(self) -> dict[str, dict[str, Any]]:
        return _enrichment_support.get_source_quality_index(self)

    def _source_quality_for_note(self, note_id: str) -> dict[str, Any]:
        return _enrichment_support.source_quality_for_note(self, note_id)

    def _candidate_from_note(self, note_id: str) -> dict[str, Any] | None:
        return _enrichment_support.candidate_from_note(self, note_id)

    def _build_concept_pack(self, item: dict[str, Any]) -> ConceptEvidencePack:
        return ConceptEvidencePack(**_enrichment_support.build_concept_pack_data(self, item))

    def _source_model_fingerprint(self, allow_external: bool, llm_mode: str) -> tuple[str, str, str, str]:
        return _enrichment_support.source_model_fingerprint(self, allow_external, llm_mode)

    def _concept_model_fingerprint(self, allow_external: bool, llm_mode: str) -> tuple[str, str, str, str]:
        return _enrichment_support.concept_model_fingerprint(self, allow_external, llm_mode)

    def _rewrite_staging_or_final(self, item: dict[str, Any], *, run_id: str) -> None:
        payload = dict(item.get("payload_json") or {})
        if str(item.get("item_type")) == "source":
            payload["frontmatter"] = self.materializer._build_source_frontmatter(
                title=str(payload.get("title_en") or item.get("title_en") or ""),
                note_id=str(payload.get("note_id") or item.get("note_id") or ""),
                entity_ids=list(payload.get("entity_ids") or []),
                knowledge_label=str(payload.get("knowledge_label") or ""),
                document_type=str(payload.get("document_type") or ""),
                domain=str(payload.get("domain") or ""),
            )
            payload = compose_source_note_payload(payload)
            markdown = render_source_note(payload)
            staging_path = _resolve_existing_path(str(item.get("staging_path") or ""))
            if staging_path.exists():
                staging_path.write_text(markdown, encoding="utf-8")
            if str(item.get("status")) == "applied":
                self.materializer._apply_source_item(dict(item), run_id)
        else:
            payload["frontmatter"] = build_visible_frontmatter(
                note_type="concept",
                status="enriched",
                title=str(payload.get("title") or item.get("title_en") or ""),
                updated=_date_only(),
            )
            payload = compose_concept_note_payload(payload)
            markdown = render_concept_note(payload)
            staging_path = _resolve_existing_path(str(item.get("staging_path") or ""))
            if staging_path.exists():
                staging_path.write_text(markdown, encoding="utf-8")
            if str(item.get("status")) == "applied":
                self.materializer._apply_concept_item(dict(item))

    def _review_guidance(
        self,
        item_type: str,
        payload: dict[str, Any],
    ) -> tuple[list[str], list[str], int, list[str], list[str], dict[str, dict[str, Any]]]:
        return _enrichment_support.review_guidance(self, item_type, payload)

    @staticmethod
    def _normalized_quality_score(quality: dict[str, Any] | None) -> float:
        return _enrichment_support.normalized_quality_score(quality)

    @staticmethod
    def _line_field_values(item_type: str, payload: dict[str, Any], section: str) -> list[str]:
        return _enrichment_support.line_field_values(item_type, payload, section)

    @classmethod
    def _field_diagnostics(
        cls,
        *,
        item_type: str,
        payload: dict[str, Any],
        target_sections: list[str],
    ) -> dict[str, dict[str, Any]]:
        return _enrichment_support.field_diagnostics(
            item_type=item_type,
            payload=payload,
            target_sections=target_sections,
            line_field_values_fn=cls._line_field_values,
        )

    @staticmethod
    def _merge_scalar_field(
        merged: dict[str, Any],
        *,
        field_name: str,
        generated_value: Any,
        section_name: str,
        missing_sections: set[str],
        patched_sections: list[str],
    ) -> tuple[int, int]:
        return _enrichment_support.merge_scalar_field(
            merged,
            field_name=field_name,
            generated_value=generated_value,
            section_name=section_name,
            missing_sections=missing_sections,
            patched_sections=patched_sections,
        )

    @staticmethod
    def _merge_list_field(
        merged: dict[str, Any],
        *,
        item_type: str,
        field_name: str,
        generated_values: list[str],
        section_name: str,
        diagnostics: dict[str, Any],
        patched_sections: list[str],
        min_count: int | None = None,
        secondary_fields: list[str] | None = None,
    ) -> tuple[int, int]:
        return _enrichment_support.merge_list_field(
            merged,
            item_type=item_type,
            field_name=field_name,
            generated_values=generated_values,
            section_name=section_name,
            diagnostics=diagnostics,
            patched_sections=patched_sections,
            min_count=min_count,
            secondary_fields=secondary_fields,
        )

    @classmethod
    def _merge_concept_section_payload(
        cls,
        payload: dict[str, Any],
        *,
        enriched: dict[str, Any],
        target_sections: list[str],
        field_diagnostics: dict[str, dict[str, Any]],
        missing_sections: set[str],
    ) -> tuple[dict[str, Any], list[str], int, int]:
        merged = dict(payload)
        patched: list[str] = []
        patched_line_count = 0
        preserved_line_count = 0
        target_set = {str(item).strip() for item in (target_sections or []) if str(item).strip()}

        if "summary" in target_set:
            patched_delta, preserved_delta = cls._merge_scalar_field(
                merged,
                field_name="summary_ko",
                generated_value=enriched.get("summary_ko") or enriched.get("core_summary") or "",
                section_name="summary",
                missing_sections=missing_sections,
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "summary_line" in target_set:
            patched_delta, preserved_delta = cls._merge_scalar_field(
                merged,
                field_name="summary_line_ko",
                generated_value=enriched.get("summary_line_ko") or "",
                section_name="summary_line",
                missing_sections=missing_sections,
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "core_summary" in target_set:
            patched_delta, preserved_delta = cls._merge_scalar_field(
                merged,
                field_name="core_summary",
                generated_value=enriched.get("core_summary") or enriched.get("summary_ko") or "",
                section_name="core_summary",
                missing_sections=missing_sections,
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "why_it_matters" in target_set:
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="concept",
                field_name="why_it_matters",
                generated_values=list(enriched.get("why_it_matters") or []),
                section_name="why_it_matters",
                diagnostics=field_diagnostics.get("why_it_matters") or {},
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "relation_lines" in target_set:
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="concept",
                field_name="relation_lines",
                generated_values=list(enriched.get("relation_lines") or []),
                section_name="relation_lines",
                diagnostics=field_diagnostics.get("relation_lines") or {},
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "claim_lines" in target_set:
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="concept",
                field_name="claim_lines",
                generated_values=list(enriched.get("claim_lines") or enriched.get("key_excerpts_ko") or []),
                section_name="claim_lines",
                diagnostics=field_diagnostics.get("claim_lines") or {},
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "support_lines" in target_set:
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="concept",
                field_name="support_lines",
                generated_values=list(enriched.get("support_lines") or []),
                section_name="support_lines",
                diagnostics=field_diagnostics.get("support_lines") or {},
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "related_sources" in target_set:
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="concept",
                field_name="related_sources",
                generated_values=list(enriched.get("related_sources") or []),
                section_name="related_sources",
                diagnostics=field_diagnostics.get("related_sources") or {},
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "related_concepts" in target_set:
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="concept",
                field_name="related_concepts",
                generated_values=list(enriched.get("related_concepts") or []),
                section_name="related_concepts",
                diagnostics=field_diagnostics.get("related_concepts") or {},
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "key_excerpts_ko" in target_set:
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="concept",
                field_name="key_excerpts_ko",
                generated_values=list(enriched.get("key_excerpts_ko") or []),
                section_name="key_excerpts_ko",
                diagnostics=field_diagnostics.get("key_excerpts_ko") or {},
                patched_sections=patched,
            )
            if enriched.get("key_excerpts_en"):
                merged["key_excerpts_en"] = list(enriched.get("key_excerpts_en") or [])
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if any(section in target_set for section in {"summary", "core_summary"}):
            merged["title_ko"] = enriched.get("title_ko") or merged.get("title_ko") or merged.get("title") or ""
        return merged, patched, patched_line_count, preserved_line_count

    @classmethod
    def _merge_source_section_payload(
        cls,
        payload: dict[str, Any],
        *,
        enriched: dict[str, Any],
        pack: SourceEvidencePack,
        target_sections: list[str],
        field_diagnostics: dict[str, dict[str, Any]],
        missing_sections: set[str],
    ) -> tuple[dict[str, Any], list[str], int, int]:
        merged = dict(payload)
        patched: list[str] = []
        patched_line_count = 0
        preserved_line_count = 0
        target_set = {str(item).strip() for item in (target_sections or []) if str(item).strip()}

        if "summary" in target_set:
            patched_delta, preserved_delta = cls._merge_scalar_field(
                merged,
                field_name="summary_ko",
                generated_value=enriched.get("summary_ko") or enriched.get("core_summary") or "",
                section_name="summary",
                missing_sections=missing_sections,
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
            merged["title_ko"] = enriched.get("title_ko") or merged.get("title_ko") or merged.get("title_en") or ""
        if "summary_line" in target_set:
            patched_delta, preserved_delta = cls._merge_scalar_field(
                merged,
                field_name="summary_line_ko",
                generated_value=enriched.get("summary_line_ko") or "",
                section_name="summary_line",
                missing_sections=missing_sections,
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "document_type" in target_set:
            patched_delta, preserved_delta = cls._merge_scalar_field(
                merged,
                field_name="document_type",
                generated_value=enriched.get("document_type") or pack.document_type,
                section_name="document_type",
                missing_sections=missing_sections,
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "thesis" in target_set:
            patched_delta, preserved_delta = cls._merge_scalar_field(
                merged,
                field_name="thesis",
                generated_value=enriched.get("thesis") or pack.thesis,
                section_name="thesis",
                missing_sections=missing_sections,
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "top_claims" in target_set:
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="source",
                field_name="top_claims",
                generated_values=list(enriched.get("top_claims") or pack.top_claims or []),
                section_name="top_claims",
                diagnostics=field_diagnostics.get("top_claims") or {},
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "contributions" in target_set:
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="source",
                field_name="contributions",
                generated_values=list(enriched.get("contributions") or []),
                section_name="contributions",
                diagnostics=field_diagnostics.get("contributions") or {},
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "methodology" in target_set:
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="source",
                field_name="methodology",
                generated_values=list(enriched.get("methodology") or []),
                section_name="methodology",
                diagnostics=field_diagnostics.get("methodology") or {},
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "results_or_findings" in target_set:
            results = list(enriched.get("results_or_findings") or enriched.get("key_results") or [])
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="source",
                field_name="results_or_findings",
                generated_values=results,
                section_name="results_or_findings",
                diagnostics=field_diagnostics.get("results_or_findings") or {},
                patched_sections=patched,
                secondary_fields=["key_results"],
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "insights" in target_set:
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="source",
                field_name="insights",
                generated_values=list(enriched.get("insights") or []),
                section_name="insights",
                diagnostics=field_diagnostics.get("insights") or {},
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "limitations" in target_set:
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="source",
                field_name="limitations",
                generated_values=list(enriched.get("limitations") or []),
                section_name="limitations",
                diagnostics=field_diagnostics.get("limitations") or {},
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "core_concepts" in target_set:
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="source",
                field_name="core_concepts",
                generated_values=list(enriched.get("core_concepts") or pack.core_concepts or []),
                section_name="core_concepts",
                diagnostics=field_diagnostics.get("core_concepts") or {},
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "key_excerpts_ko" in target_set:
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="source",
                field_name="key_excerpts_ko",
                generated_values=list(enriched.get("key_excerpts_ko") or []),
                section_name="key_excerpts_ko",
                diagnostics=field_diagnostics.get("key_excerpts_ko") or {},
                patched_sections=patched,
            )
            if enriched.get("key_excerpts_en"):
                merged["key_excerpts_en"] = list(enriched.get("key_excerpts_en") or merged.get("key_excerpts_en") or pack.key_excerpts_en)
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "related_concepts" in target_set:
            related = list(enriched.get("related_concepts") or [f"- {token}" for token in pack.related_concepts[:12]])
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="source",
                field_name="related_concepts",
                generated_values=related,
                section_name="related_concepts",
                diagnostics=field_diagnostics.get("related_concepts") or {},
                patched_sections=patched,
            )
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        if "sources" in target_set:
            patched_delta, preserved_delta = cls._merge_list_field(
                merged,
                item_type="source",
                field_name="representative_sources",
                generated_values=list(enriched.get("representative_sources") or pack.representative_sources or []),
                section_name="sources",
                diagnostics=field_diagnostics.get("sources") or {},
                patched_sections=patched,
            )
            if not list(merged.get("metadata_lines") or []):
                merged["metadata_lines"] = list(pack.metadata_lines)
            patched_line_count += patched_delta
            preserved_line_count += preserved_delta
        return merged, patched, patched_line_count, preserved_line_count

    def _remediate_source_item(
        self,
        *,
        item: dict[str, Any],
        remediation_run_id: str,
        strategy: str,
        allow_external: bool,
        llm_mode: str,
        local_timeout_sec: int | None,
        api_fallback_on_timeout: bool,
    ) -> tuple[dict[str, Any], bool, bool]:
        pack = self._build_source_pack(item)
        payload = dict(item.get("payload_json") or {})
        before_payload = deepcopy(payload)
        before_quality = KoNoteQuality.from_payload(payload).to_payload()
        (
            review_reasons,
            review_patch_hints,
            remediation_attempt,
            target_sections,
            preserve_sections,
            field_diagnostics,
        ) = self._review_guidance("source", payload)
        missing_sections = {str(item).strip() for item in (before_quality.get("missing_sections") or []) if str(item).strip()}
        inferred_topic = infer_quality_topic(
            self.config,
            None,
            pack.title_en,
            pack.title_ko,
            pack.content_text,
            pack.entity_names,
            pack.relation_lines,
            pack.claim_lines,
            pack.related_concepts,
        )
        quality_decision = resolve_quality_mode_route(
            self.config,
            item_kind="source",
            requested_allow_external=allow_external,
            requested_mode=llm_mode,
            topic=inferred_topic,
            counters={},
            monthly_spend_usd=0.0,
        )
        self.koreanizer = self._get_koreanizer_for(
            allow_external=quality_decision.allow_external,
            llm_mode=quality_decision.llm_mode,
            local_timeout_sec=local_timeout_sec,
            api_fallback_on_timeout=api_fallback_on_timeout,
        )
        if strategy == "section" and not target_sections:
            warning = "section-remediation-skipped:no-target-sections"
            payload["review"] = update_review_remediation(
                payload.get("review"),
                run_id=remediation_run_id,
                status="skipped",
                warnings=[warning],
                before_quality=before_quality,
                after_quality=before_quality,
                strategy=strategy,
                target_sections=[],
                patched_sections=[],
                preserved_sections_count=len(preserve_sections),
            )
            self.sqlite_db.update_ko_note_item_payload(
                int(item["id"]),
                payload=payload,
                title_en=pack.title_en,
                title_ko=str(payload.get("title_ko") or item.get("title_ko") or pack.title_ko),
            )
            updated_item = self.sqlite_db.get_ko_note_item(int(item["id"])) or item
            return updated_item, False, False
        enriched = self.koreanizer.build_source_enrichment(
            title_en=pack.title_en,
            content_text=pack.content_text,
            entity_names=pack.entity_names,
            relation_lines=pack.relation_lines,
            claim_lines=pack.claim_lines,
            related_concepts=pack.related_concepts,
            key_excerpts_en=pack.key_excerpts_en,
            metadata_lines=pack.metadata_lines,
            evidence_pack={
                "document_type": pack.document_type,
                "thesis": pack.thesis,
                "top_claims": pack.top_claims,
                "core_concepts": pack.core_concepts,
                "contributions": pack.contributions,
                "methodology": pack.methodology,
                "results_or_findings": pack.results_or_findings,
                "limitations": pack.limitations,
                "representative_sources": pack.representative_sources,
            },
            candidate_score=pack.candidate_score,
            translation_level=pack.translation_level,
            minimum_bullets_per_section=int(
                self.config.get_nested("materialization", "enrichment", "minimum_source_bullets_per_section", default=3) or 3
            ),
            target_sections=target_sections if strategy == "section" else [],
            preserve_sections=preserve_sections if strategy == "section" else [],
            field_diagnostics=field_diagnostics if strategy == "section" else {},
            review_reasons=review_reasons,
            review_patch_hints=review_patch_hints,
            remediation_attempt=remediation_attempt,
        )
        payload.update(
            {
                "title_en": pack.title_en,
                "source_content_text": payload.get("source_content_text") or pack.content_text,
                "source_key_excerpts_en": payload.get("source_key_excerpts_en") or pack.key_excerpts_en,
                "enrichment_meta": {
                    "version": ENRICHMENT_VERSION,
                    "route": quality_decision.llm_mode,
                    "provider": "",
                    "model": "",
                    "evidence_pack_hash": pack.stable_hash(),
                    "needs_reinforcement": bool(enriched.get("needs_reinforcement")),
                    "reinforcement_reasons": list(enriched.get("reinforcement_reasons") or []),
                    "remediationRunId": remediation_run_id,
                },
            }
        )
        if strategy == "full":
            payload.update(
                {
                    "title_ko": enriched["title_ko"],
                    "summary_ko": enriched["summary_ko"],
                    "core_summary": enriched["core_summary"],
                    "summary_line_ko": enriched["summary_line_ko"],
                    "document_type": enriched.get("document_type") or pack.document_type,
                    "thesis": enriched.get("thesis") or pack.thesis,
                    "top_claims": enriched.get("top_claims") or pack.top_claims,
                    "core_concepts": enriched.get("core_concepts") or pack.core_concepts,
                    "contributions": enriched["contributions"],
                    "methodology": enriched["methodology"],
                    "key_results": enriched["key_results"],
                    "results_or_findings": enriched.get("results_or_findings") or enriched["key_results"],
                    "limitations": enriched["limitations"],
                    "insights": enriched["insights"],
                    "representative_sources": enriched.get("representative_sources") or pack.representative_sources,
                    "entity_lines": enriched["entity_lines"],
                    "relation_lines": enriched["relation_lines"],
                    "key_excerpts_ko": enriched["key_excerpts_ko"],
                    "key_excerpts_en": enriched["key_excerpts_en"],
                    "related_concepts": [f"- {token}" for token in pack.related_concepts[:12]] or payload.get("related_concepts") or [],
                }
            )
            patched_sections = [
                "summary",
                "summary_line",
                "document_type",
                "thesis",
                "top_claims",
                "contributions",
                "methodology",
                "results_or_findings",
                "insights",
                "limitations",
                "core_concepts",
                "key_excerpts_ko",
                "related_concepts",
                "sources",
            ]
            patched_line_count = sum(
                len(list(payload.get(field) or []))
                for field in (
                    "top_claims",
                    "contributions",
                    "methodology",
                    "results_or_findings",
                    "insights",
                    "limitations",
                    "core_concepts",
                    "key_excerpts_ko",
                    "related_concepts",
                    "representative_sources",
                )
            ) + 4
            preserved_line_count = 0
        else:
            payload, patched_sections, patched_line_count, preserved_line_count = self._merge_source_section_payload(
                payload,
                enriched=enriched,
                pack=pack,
                target_sections=target_sections,
                field_diagnostics=field_diagnostics,
                missing_sections=missing_sections,
            )
        payload = compose_source_note_payload(
            payload,
            content_text=payload.get("source_content_text") or pack.content_text,
            entity_names=pack.entity_names,
            metadata_lines=pack.metadata_lines,
        )
        payload["knowledge_label"] = str(payload.get("knowledge_label") or "").strip() or self.materializer._source_knowledge_label(
            note_id=str(payload.get("note_id") or item.get("note_id") or ""),
            entity_ids=list(payload.get("entity_ids") or []),
            title=str(payload.get("title_en") or pack.title_en or ""),
            document_type=str(payload.get("document_type") or ""),
            domain=str(payload.get("domain") or pack.domain or ""),
        )
        payload["frontmatter"] = self.materializer._build_source_frontmatter(
            title=str(payload.get("title_en") or pack.title_en or ""),
            note_id=str(payload.get("note_id") or item.get("note_id") or ""),
            entity_ids=list(payload.get("entity_ids") or []),
            knowledge_label=str(payload.get("knowledge_label") or ""),
            document_type=str(payload.get("document_type") or ""),
            domain=str(payload.get("domain") or pack.domain or ""),
        )
        payload = _attach_source_quality(payload, render_source_note(payload))
        remediation_warnings = list(dict.fromkeys([*list(enriched.get("warnings") or []), *list(quality_decision.warnings)]))
        after_quality = KoNoteQuality.from_payload(payload).to_payload()
        if strategy == "section":
            before_norm = self._normalized_quality_score(before_quality)
            after_norm = self._normalized_quality_score(after_quality)
            if after_norm + 1e-9 < before_norm:
                remediation_warnings.append(
                    f"section-remediation-quality-regressed:{before_norm:.3f}->{after_norm:.3f}"
                )
        payload = _attach_review_payload(payload, item_type="source")
        payload["review"] = update_review_remediation(
            payload.get("review"),
            run_id=remediation_run_id,
            status="remediated",
            warnings=remediation_warnings,
            before_quality=before_quality,
            after_quality=after_quality,
            strategy=strategy,
            target_sections=target_sections if strategy == "section" else [],
            patched_sections=patched_sections,
            preserved_sections_count=len(preserve_sections),
            patched_line_count=patched_line_count,
            preserved_line_count=preserved_line_count,
        )
        self.sqlite_db.update_ko_note_item_payload(
            int(item["id"]),
            payload=payload,
            title_en=pack.title_en,
            title_ko=enriched["title_ko"],
        )
        updated_item = self.sqlite_db.get_ko_note_item(int(item["id"])) or item
        self._rewrite_staging_or_final(updated_item, run_id=remediation_run_id)
        improved = KoNoteReview.from_payload(payload).remediation.last_improved
        changed = payload != before_payload
        return updated_item, improved, changed

    def _remediate_concept_item(
        self,
        *,
        item: dict[str, Any],
        remediation_run_id: str,
        strategy: str,
        allow_external: bool,
        llm_mode: str,
        local_timeout_sec: int | None,
        api_fallback_on_timeout: bool,
    ) -> tuple[dict[str, Any], bool, bool]:
        payload = dict(item.get("payload_json") or {})
        before_payload = deepcopy(payload)
        before_quality = KoNoteQuality.from_payload(payload).to_payload()
        (
            review_reasons,
            review_patch_hints,
            remediation_attempt,
            target_sections,
            preserve_sections,
            field_diagnostics,
        ) = self._review_guidance("concept", payload)
        missing_sections = {str(item).strip() for item in (before_quality.get("missing_sections") or []) if str(item).strip()}
        topic_seed = infer_quality_topic(
            self.config,
            None,
            payload.get("title"),
            payload.get("summary_ko"),
            payload.get("core_summary"),
            payload.get("relation_lines"),
            payload.get("related_concepts"),
        )
        quality_decision = resolve_quality_mode_route(
            self.config,
            item_kind="concept",
            requested_allow_external=allow_external,
            requested_mode=llm_mode,
            topic=topic_seed,
            counters={},
            monthly_spend_usd=0.0,
        )
        self.koreanizer = self._get_koreanizer_for(
            allow_external=quality_decision.allow_external,
            llm_mode=quality_decision.llm_mode,
            local_timeout_sec=local_timeout_sec,
            api_fallback_on_timeout=api_fallback_on_timeout,
        )
        pack = self._build_concept_pack(item)
        if strategy == "section" and not target_sections:
            warning = "section-remediation-skipped:no-target-sections"
            payload["review"] = update_review_remediation(
                payload.get("review"),
                run_id=remediation_run_id,
                status="skipped",
                warnings=[warning],
                before_quality=before_quality,
                after_quality=before_quality,
                strategy=strategy,
                target_sections=[],
                patched_sections=[],
                preserved_sections_count=len(preserve_sections),
            )
            self.sqlite_db.update_ko_note_item_payload(
                int(item["id"]),
                payload=payload,
                title_en=str(item.get("title_en") or payload.get("title") or ""),
                title_ko=str(payload.get("title_ko") or item.get("title_ko") or pack.canonical_name),
            )
            updated_item = self.sqlite_db.get_ko_note_item(int(item["id"])) or item
            return updated_item, False, False
        enriched = self.koreanizer.build_concept_enrichment(
            canonical_name=pack.canonical_name,
            aliases=pack.aliases,
            relation_lines=pack.relation_lines,
            related_concepts=pack.related_concepts,
            compressed_support_docs=pack.compressed_support_docs,
            existing_summary=pack.existing_summary,
            candidate_score=pack.candidate_score,
            translation_level=pack.translation_level,
            minimum_support_docs=int(
                self.config.get_nested("materialization", "enrichment", "minimum_concept_support_docs", default=4) or 4
            ),
            target_sections=target_sections if strategy == "section" else [],
            preserve_sections=preserve_sections if strategy == "section" else [],
            field_diagnostics=field_diagnostics if strategy == "section" else {},
            review_reasons=review_reasons,
            review_patch_hints=review_patch_hints,
            remediation_attempt=remediation_attempt,
        )
        payload.update(
            {
                "enrichment_meta": {
                    "version": ENRICHMENT_VERSION,
                    "route": quality_decision.llm_mode,
                    "provider": "",
                    "model": "",
                    "evidence_pack_hash": pack.stable_hash(),
                    "remediationRunId": remediation_run_id,
                },
            }
        )
        if strategy == "full":
            payload.update(
                {
                    "title_ko": enriched["title_ko"],
                    "summary_ko": enriched["summary_ko"],
                    "core_summary": enriched["core_summary"],
                    "summary_line_ko": enriched["summary_line_ko"],
                    "why_it_matters": enriched["why_it_matters"],
                    "relation_lines": enriched["relation_lines"],
                    "support_lines": enriched["support_lines"],
                    "key_excerpts_ko": enriched["key_excerpts_ko"],
                    "key_excerpts_en": enriched["key_excerpts_en"],
                    "related_sources": enriched.get("related_sources") or payload.get("related_sources") or [],
                    "related_concepts": enriched.get("related_concepts") or payload.get("related_concepts") or [],
                }
            )
            patched_sections = [
                "summary",
                "summary_line",
                "core_summary",
                "why_it_matters",
                "relation_lines",
                "claim_lines",
                "support_lines",
                "related_sources",
                "related_concepts",
                "key_excerpts_ko",
            ]
            patched_line_count = sum(
                len(list(payload.get(field) or []))
                for field in (
                    "why_it_matters",
                    "relation_lines",
                    "claim_lines",
                    "support_lines",
                    "related_sources",
                    "related_concepts",
                    "key_excerpts_ko",
                )
            ) + 3
            preserved_line_count = 0
        else:
            payload, patched_sections, patched_line_count, preserved_line_count = self._merge_concept_section_payload(
                payload,
                enriched=enriched,
                target_sections=target_sections,
                field_diagnostics=field_diagnostics,
                missing_sections=missing_sections,
            )
        payload = compose_concept_note_payload(payload, aliases=pack.aliases)
        payload["frontmatter"] = build_visible_frontmatter(
            note_type="concept",
            status="enriched",
            title=str(payload.get("title") or item.get("title_en") or ""),
            updated=_date_only(),
        )
        payload = _attach_concept_quality(payload, render_concept_note(payload))
        remediation_warnings = list(dict.fromkeys([*list(enriched.get("warnings") or []), *list(quality_decision.warnings)]))
        after_quality = KoNoteQuality.from_payload(payload).to_payload()
        if strategy == "section":
            before_norm = self._normalized_quality_score(before_quality)
            after_norm = self._normalized_quality_score(after_quality)
            if after_norm + 1e-9 < before_norm:
                remediation_warnings.append(
                    f"section-remediation-quality-regressed:{before_norm:.3f}->{after_norm:.3f}"
                )
        payload = _attach_review_payload(payload, item_type="concept")
        payload["review"] = update_review_remediation(
            payload.get("review"),
            run_id=remediation_run_id,
            status="remediated",
            warnings=remediation_warnings,
            before_quality=before_quality,
            after_quality=after_quality,
            strategy=strategy,
            target_sections=target_sections if strategy == "section" else [],
            patched_sections=patched_sections,
            preserved_sections_count=len(preserve_sections),
            patched_line_count=patched_line_count,
            preserved_line_count=preserved_line_count,
        )
        self.sqlite_db.update_ko_note_item_payload(
            int(item["id"]),
            payload=payload,
            title_en=str(item.get("title_en") or payload.get("title") or ""),
            title_ko=enriched["title_ko"],
        )
        updated_item = self.sqlite_db.get_ko_note_item(int(item["id"])) or item
        self._rewrite_staging_or_final(updated_item, run_id=remediation_run_id)
        improved = KoNoteReview.from_payload(payload).remediation.last_improved
        changed = payload != before_payload
        return updated_item, improved, changed

    def _select_existing_top_items(self, *, item_type: str, limit: int) -> list[dict[str, Any]]:
        return _enrichment_support.select_existing_top_items(self, item_type=item_type, limit=limit)

    def enrich(
        self,
        *,
        scope: str = "both",
        run_id: str = "",
        item_type: str = "all",
        limit_source: int = 120,
        limit_concept: int = 80,
        allow_external: bool = True,
        llm_mode: str = "auto",
        local_timeout_sec: int | None = None,
        api_fallback_on_timeout: bool = True,
    ) -> dict[str, Any]:
        ts = _date_only()
        chosen_run_id = str(run_id or "").strip()
        if scope in {"new", "both"} and not chosen_run_id:
            latest = self.sqlite_db.get_latest_ko_note_run() or {}
            chosen_run_id = str(latest.get("run_id") or "").strip()
        enrichment_run_id = (
            f"ko_note_enrich_"
            f"{hashlib.sha1(f'{scope}:{chosen_run_id}:{item_type}:{time.time_ns()}'.encode('utf-8')).hexdigest()[:12]}"
        )
        self.koreanizer = Koreanizer(
            self.config,
            allow_external=allow_external,
            llm_mode=llm_mode,
            local_timeout_sec=local_timeout_sec,
            api_fallback_on_timeout=api_fallback_on_timeout,
        )
        usage_counts: dict[str, int] = {}

        selected: list[dict[str, Any]] = []
        if scope in {"new", "both"} and chosen_run_id:
            for current_type in (["source", "concept"] if item_type == "all" else [item_type]):
                selected.extend(
                    self.sqlite_db.list_ko_note_items(
                        run_id=chosen_run_id,
                        item_type=current_type,
                        status=None,
                        limit=2000,
                    )
                )
        if scope in {"existing-top", "both"}:
            if item_type in {"all", "source"}:
                selected.extend(self._select_existing_top_items(item_type="source", limit=limit_source))
            if item_type in {"all", "concept"}:
                selected.extend(self._select_existing_top_items(item_type="concept", limit=limit_concept))

        deduped: dict[tuple[str, str], dict[str, Any]] = {}
        for item in selected:
            key = (str(item.get("item_type") or ""), str(item.get("id") or item.get("final_path") or ""))
            deduped[key] = item
        selected = list(deduped.values())
        source_target_count = sum(1 for item in selected if str(item.get("item_type")) == "source")
        concept_target_count = sum(1 for item in selected if str(item.get("item_type")) == "concept")
        warnings: list[str] = []
        self.sqlite_db.create_ko_note_enrichment_run(
            run_id=enrichment_run_id,
            source_run_id=chosen_run_id,
            scope=scope,
            item_type=item_type,
            status="running",
            source_target_count=source_target_count,
            concept_target_count=concept_target_count,
            warnings=[],
        )

        source_enriched = 0
        concept_enriched = 0
        skipped = 0
        run_status = "completed"
        monthly_spend_usd = float(self.sqlite_db.get_quality_mode_monthly_spend() or 0.0)

        try:
            for item in selected:
                if str(item.get("item_type")) == "source":
                    pack = self._build_source_pack(item)
                    inferred_topic = infer_quality_topic(
                        self.config,
                        None,
                        pack.title_en,
                        pack.title_ko,
                        pack.content_text,
                        pack.entity_names,
                        pack.relation_lines,
                        pack.claim_lines,
                        pack.related_concepts,
                    )
                    quality_decision = resolve_quality_mode_route(
                        self.config,
                        item_kind="source",
                        requested_allow_external=allow_external,
                        requested_mode=llm_mode,
                        topic=inferred_topic,
                        counters=usage_counts,
                        monthly_spend_usd=monthly_spend_usd,
                    )
                    self.koreanizer = self._get_koreanizer_for(
                        allow_external=quality_decision.allow_external,
                        llm_mode=quality_decision.llm_mode,
                        local_timeout_sec=local_timeout_sec,
                        api_fallback_on_timeout=api_fallback_on_timeout,
                    )
                    route, provider, model, fingerprint = self._source_model_fingerprint(
                        quality_decision.allow_external, quality_decision.llm_mode
                    )
                    evidence_pack_hash = pack.stable_hash()
                    existing = self.sqlite_db.find_matching_ko_note_enrichment_item(
                        note_item_id=int(item.get("id") or 0),
                        target_path=str(item.get("final_path") or ""),
                        item_type="source",
                        evidence_pack_hash=evidence_pack_hash,
                        model_fingerprint=fingerprint,
                    )
                    if existing and str(existing.get("status")) in {"enriched", "skipped"}:
                        skipped += 1
                        continue
                    enrichment_item_id = self.sqlite_db.add_ko_note_enrichment_item(
                        run_id=enrichment_run_id,
                        note_item_id=int(item.get("id") or 0),
                        target_path=str(item.get("final_path") or ""),
                        item_type="source",
                        status="queued",
                        route=route,
                        provider=provider,
                        model=model,
                        model_fingerprint=fingerprint,
                        evidence_pack_hash=evidence_pack_hash,
                        warnings=[],
                    )
                    enriched = self.koreanizer.build_source_enrichment(
                        title_en=pack.title_en,
                        content_text=pack.content_text,
                        entity_names=pack.entity_names,
                        relation_lines=pack.relation_lines,
                        claim_lines=pack.claim_lines,
                        related_concepts=pack.related_concepts,
                        key_excerpts_en=pack.key_excerpts_en,
                        metadata_lines=pack.metadata_lines,
                        evidence_pack={
                            "document_type": pack.document_type,
                            "thesis": pack.thesis,
                            "top_claims": pack.top_claims,
                            "core_concepts": pack.core_concepts,
                            "contributions": pack.contributions,
                            "methodology": pack.methodology,
                            "results_or_findings": pack.results_or_findings,
                            "limitations": pack.limitations,
                            "representative_sources": pack.representative_sources,
                        },
                        candidate_score=pack.candidate_score,
                        translation_level=pack.translation_level,
                        minimum_bullets_per_section=int(
                            self.config.get_nested("materialization", "enrichment", "minimum_source_bullets_per_section", default=3)
                            or 3
                        ),
                    )
                    payload = dict(item.get("payload_json") or {})
                    payload.update(
                        {
                            "title_en": pack.title_en,
                            "title_ko": enriched["title_ko"],
                            "summary_ko": enriched["summary_ko"],
                            "core_summary": enriched["core_summary"],
                            "summary_line_ko": enriched["summary_line_ko"],
                            "document_type": enriched.get("document_type") or pack.document_type,
                            "thesis": enriched.get("thesis") or pack.thesis,
                            "top_claims": enriched.get("top_claims") or pack.top_claims,
                            "core_concepts": enriched.get("core_concepts") or pack.core_concepts,
                            "contributions": enriched["contributions"],
                            "methodology": enriched["methodology"],
                            "key_results": enriched["key_results"],
                            "results_or_findings": enriched.get("results_or_findings") or enriched["key_results"],
                            "limitations": enriched["limitations"],
                            "insights": enriched["insights"],
                            "representative_sources": enriched.get("representative_sources") or pack.representative_sources,
                            "entity_lines": enriched["entity_lines"],
                            "relation_lines": enriched["relation_lines"],
                            "key_excerpts_ko": enriched["key_excerpts_ko"],
                            "key_excerpts_en": enriched["key_excerpts_en"],
                            "source_content_text": payload.get("source_content_text") or pack.content_text,
                            "source_key_excerpts_en": payload.get("source_key_excerpts_en") or pack.key_excerpts_en,
                            "related_concepts": [f"- {token}" for token in pack.related_concepts[:12]] or payload.get("related_concepts") or [],
                            "enrichment_meta": {
                                "version": ENRICHMENT_VERSION,
                                "route": route,
                                "provider": provider,
                                "model": model,
                                "evidence_pack_hash": evidence_pack_hash,
                                "needs_reinforcement": bool(enriched.get("needs_reinforcement")),
                                "reinforcement_reasons": list(enriched.get("reinforcement_reasons") or []),
                            },
                        }
                    )
                    payload = compose_source_note_payload(
                        payload,
                        content_text=payload.get("source_content_text") or pack.content_text,
                        entity_names=pack.entity_names,
                        metadata_lines=pack.metadata_lines,
                    )
                    payload["knowledge_label"] = str(payload.get("knowledge_label") or "").strip() or self.materializer._source_knowledge_label(
                        note_id=str(payload.get("note_id") or item.get("note_id") or ""),
                        entity_ids=list(payload.get("entity_ids") or []),
                        title=str(payload.get("title_en") or pack.title_en or ""),
                        document_type=str(payload.get("document_type") or ""),
                        domain=str(payload.get("domain") or pack.domain or ""),
                    )
                    payload["frontmatter"] = self.materializer._build_source_frontmatter(
                        title=str(payload.get("title_en") or pack.title_en or ""),
                        note_id=str(payload.get("note_id") or item.get("note_id") or ""),
                        entity_ids=list(payload.get("entity_ids") or []),
                        knowledge_label=str(payload.get("knowledge_label") or ""),
                        document_type=str(payload.get("document_type") or ""),
                        domain=str(payload.get("domain") or pack.domain or ""),
                    )
                    payload = _attach_source_quality(payload, render_source_note(payload))
                    payload = _attach_review_payload(payload, item_type="source")
                    self.sqlite_db.update_ko_note_item_payload(
                        int(item["id"]),
                        payload=payload,
                        title_en=pack.title_en,
                        title_ko=enriched["title_ko"],
                    )
                    updated_item = self.sqlite_db.get_ko_note_item(int(item["id"])) or item
                    self._rewrite_staging_or_final(updated_item, run_id=enrichment_run_id)
                    self.sqlite_db.update_ko_note_enrichment_item(
                        enrichment_item_id,
                        status="enriched",
                        warnings_json=list((payload.get("warnings") or [])) + list(quality_decision.warnings),
                    )
                    warnings.extend(quality_decision.warnings)
                    warnings.extend(payload.get("warnings") or [])
                    if quality_decision.allow_external and quality_decision.usage_key:
                        usage_counts[quality_decision.usage_key] = int(usage_counts.get(quality_decision.usage_key, 0)) + 1
                        estimated_cost = estimate_quality_mode_cost(
                            self.config,
                            "source",
                            quality_decision.llm_mode,
                        )
                        self.sqlite_db.record_quality_mode_usage(
                            "source",
                            quality_decision.llm_mode,
                            estimated_cost,
                            topic_slug=inferred_topic or "",
                        )
                        monthly_spend_usd += estimated_cost
                    source_enriched += 1
                    self._source_quality_index = None
                else:
                    topic_seed = None
                    concept_payload = dict(item.get("payload_json") or {})
                    topic_seed = infer_quality_topic(
                        self.config,
                        None,
                        concept_payload.get("title"),
                        concept_payload.get("summary_ko"),
                        concept_payload.get("core_summary"),
                        concept_payload.get("relation_lines"),
                        concept_payload.get("related_concepts"),
                    )
                    quality_decision = resolve_quality_mode_route(
                        self.config,
                        item_kind="concept",
                        requested_allow_external=allow_external,
                        requested_mode=llm_mode,
                        topic=topic_seed,
                        counters=usage_counts,
                        monthly_spend_usd=monthly_spend_usd,
                    )
                    self.koreanizer = self._get_koreanizer_for(
                        allow_external=quality_decision.allow_external,
                        llm_mode=quality_decision.llm_mode,
                        local_timeout_sec=local_timeout_sec,
                        api_fallback_on_timeout=api_fallback_on_timeout,
                    )
                    pack = self._build_concept_pack(item)
                    route, provider, model, fingerprint = self._concept_model_fingerprint(
                        quality_decision.allow_external, quality_decision.llm_mode
                    )
                    evidence_pack_hash = pack.stable_hash()
                    existing = self.sqlite_db.find_matching_ko_note_enrichment_item(
                        note_item_id=int(item.get("id") or 0),
                        target_path=str(item.get("final_path") or ""),
                        item_type="concept",
                        evidence_pack_hash=evidence_pack_hash,
                        model_fingerprint=fingerprint,
                    )
                    if existing and str(existing.get("status")) in {"enriched", "skipped"}:
                        skipped += 1
                        continue
                    enrichment_item_id = self.sqlite_db.add_ko_note_enrichment_item(
                        run_id=enrichment_run_id,
                        note_item_id=int(item.get("id") or 0),
                        target_path=str(item.get("final_path") or ""),
                        item_type="concept",
                        status="queued",
                        route=route,
                        provider=provider,
                        model=model,
                        model_fingerprint=fingerprint,
                        evidence_pack_hash=evidence_pack_hash,
                        warnings=[],
                    )
                    enriched = self.koreanizer.build_concept_enrichment(
                        canonical_name=pack.canonical_name,
                        aliases=pack.aliases,
                        relation_lines=pack.relation_lines,
                        related_concepts=pack.related_concepts,
                        compressed_support_docs=pack.compressed_support_docs,
                        existing_summary=pack.existing_summary,
                        candidate_score=pack.candidate_score,
                        translation_level=pack.translation_level,
                        minimum_support_docs=int(
                            self.config.get_nested("materialization", "enrichment", "minimum_concept_support_docs", default=4)
                            or 4
                        ),
                    )
                    payload = dict(item.get("payload_json") or {})
                    payload.update(
                        {
                            "title_ko": enriched["title_ko"],
                            "summary_ko": enriched["summary_ko"],
                            "core_summary": enriched["core_summary"],
                            "summary_line_ko": enriched["summary_line_ko"],
                            "why_it_matters": enriched["why_it_matters"],
                            "relation_lines": enriched["relation_lines"],
                            "support_lines": enriched["support_lines"],
                            "key_excerpts_ko": enriched["key_excerpts_ko"],
                            "key_excerpts_en": enriched["key_excerpts_en"],
                            "related_sources": enriched.get("related_sources") or payload.get("related_sources") or [],
                            "related_concepts": enriched.get("related_concepts") or payload.get("related_concepts") or [],
                            "enrichment_meta": {
                                "version": ENRICHMENT_VERSION,
                                "route": route,
                                "provider": provider,
                                "model": model,
                                "evidence_pack_hash": evidence_pack_hash,
                            },
                        }
                    )
                    payload = compose_concept_note_payload(payload, aliases=pack.aliases)
                    payload["frontmatter"] = build_visible_frontmatter(
                        note_type="concept",
                        status="enriched",
                        title=str(payload.get("title") or item.get("title_en") or ""),
                        updated=_date_only(),
                    )
                    payload = _attach_concept_quality(payload, render_concept_note(payload))
                    payload = _attach_review_payload(payload, item_type="concept")
                    self.sqlite_db.update_ko_note_item_payload(
                        int(item["id"]),
                        payload=payload,
                        title_en=str(item.get("title_en") or payload.get("title") or ""),
                        title_ko=enriched["title_ko"],
                    )
                    updated_item = self.sqlite_db.get_ko_note_item(int(item["id"])) or item
                    self._rewrite_staging_or_final(updated_item, run_id=enrichment_run_id)
                    self.sqlite_db.update_ko_note_enrichment_item(
                        enrichment_item_id,
                        status="enriched",
                        warnings_json=list((payload.get("warnings") or [])) + list(quality_decision.warnings),
                    )
                    warnings.extend(quality_decision.warnings)
                    warnings.extend(payload.get("warnings") or [])
                    if quality_decision.allow_external and quality_decision.usage_key:
                        usage_counts[quality_decision.usage_key] = int(usage_counts.get(quality_decision.usage_key, 0)) + 1
                        estimated_cost = estimate_quality_mode_cost(
                            self.config,
                            "concept",
                            quality_decision.llm_mode,
                        )
                        self.sqlite_db.record_quality_mode_usage(
                            "concept",
                            quality_decision.llm_mode,
                            estimated_cost,
                            topic_slug=topic_seed or "",
                        )
                        monthly_spend_usd += estimated_cost
                    concept_enriched += 1
        except Exception as error:
            warnings.append(f"ko-note-enrichment-error: {error}")
            run_status = "partial" if (source_enriched or concept_enriched) else "failed"

        self.sqlite_db.update_ko_note_enrichment_run(
            enrichment_run_id,
            status=run_status,
            source_enriched_count=source_enriched,
            concept_enriched_count=concept_enriched,
            warnings_json=list(dict.fromkeys(warnings)),
        )
        return {
            "schema": "knowledge-hub.ko-note.enrich.result.v1",
            "status": run_status,
            "runId": enrichment_run_id,
            "sourceRunId": chosen_run_id,
            "scope": scope,
            "itemType": item_type,
            "sourceTargets": source_target_count,
            "sourceEnriched": source_enriched,
            "conceptTargets": concept_target_count,
            "conceptEnriched": concept_enriched,
            "skipped": skipped,
            "warnings": list(dict.fromkeys(warnings)),
            "ts": _date_only(),
        }

    def remediate(
        self,
        *,
        run_id: str,
        item_type: str = "all",
        quality_flag: str = "all",
        item_id: int = 0,
        limit: int = 50,
        strategy: str = "section",
        allow_external: bool = True,
        llm_mode: str = "auto",
        local_timeout_sec: int | None = None,
        api_fallback_on_timeout: bool = True,
    ) -> dict[str, Any]:
        remediation_run_id = (
            f"ko_note_remediate_"
            f"{hashlib.sha1(f'{run_id}:{item_type}:{quality_flag}:{item_id}:{strategy}:{time.time_ns()}'.encode('utf-8')).hexdigest()[:12]}"
        )
        self.koreanizer = Koreanizer(
            self.config,
            allow_external=allow_external,
            llm_mode=llm_mode,
            local_timeout_sec=local_timeout_sec,
            api_fallback_on_timeout=api_fallback_on_timeout,
        )
        selected = self.sqlite_db.list_ko_note_items(
            run_id=str(run_id or "").strip(),
            item_type=None if item_type == "all" else item_type,
            status="staged",
            limit=2000,
        )
        filtered: list[dict[str, Any]] = []
        for item in selected:
            if item_id and int(item.get("id") or 0) != int(item_id):
                continue
            payload = dict(item.get("payload_json") or {})
            review = KoNoteReview.from_payload(payload)
            if not review.queue:
                continue
            current_quality = KoNoteQuality.from_payload(payload)
            current_flag = current_quality.flag
            if quality_flag != "all" and current_flag != quality_flag:
                continue
            filtered.append(item)
        if limit > 0:
            filtered = filtered[: max(1, int(limit))]

        attempted = 0
        remediated = 0
        improved = 0
        unchanged = 0
        failed = 0
        warnings: list[str] = []
        items_out: list[dict[str, Any]] = []

        for item in filtered:
            attempted += 1
            payload = dict(item.get("payload_json") or {})
            before_quality = KoNoteQuality.from_payload(payload)
            before_flag = before_quality.flag
            before_score = float(before_quality.score or 0.0)
            try:
                if str(item.get("item_type") or "") == "source":
                    updated_item, item_improved, item_changed = self._remediate_source_item(
                        item=item,
                        remediation_run_id=remediation_run_id,
                        strategy=str(strategy or "section"),
                        allow_external=allow_external,
                        llm_mode=llm_mode,
                        local_timeout_sec=local_timeout_sec,
                        api_fallback_on_timeout=api_fallback_on_timeout,
                    )
                else:
                    updated_item, item_improved, item_changed = self._remediate_concept_item(
                        item=item,
                        remediation_run_id=remediation_run_id,
                        strategy=str(strategy or "section"),
                        allow_external=allow_external,
                        llm_mode=llm_mode,
                        local_timeout_sec=local_timeout_sec,
                        api_fallback_on_timeout=api_fallback_on_timeout,
                    )
                after_payload = dict(updated_item.get("payload_json") or {})
                after_quality = KoNoteQuality.from_payload(after_payload)
                remediation_meta = KoNoteReview.from_payload(after_payload).remediation
                remediated += 1
                if item_improved:
                    improved += 1
                else:
                    unchanged += 1
                items_out.append(
                    {
                        "id": updated_item.get("id"),
                        "itemType": updated_item.get("item_type"),
                        "status": updated_item.get("status"),
                        "titleKo": updated_item.get("title_ko"),
                        "titleEn": updated_item.get("title_en"),
                        "beforeQualityFlag": before_flag,
                        "beforeScore": before_score,
                        "afterQualityFlag": after_quality.flag,
                        "afterScore": float(after_quality.score or 0.0),
                        "improved": bool(item_improved),
                        "changed": bool(item_changed),
                        "strategy": str(remediation_meta.strategy or strategy or "section"),
                        "targetSections": list(remediation_meta.target_sections),
                        "patchedSections": list(remediation_meta.patched_sections),
                        "preservedSectionsCount": int(remediation_meta.preserved_sections_count or 0),
                        "patchedLineCount": int(remediation_meta.last_patched_line_count or 0),
                        "preservedLineCount": int(remediation_meta.last_preserved_line_count or 0),
                        "recommendedStrategy": str(remediation_meta.recommended_strategy or ""),
                    }
                )
            except Exception as error:
                failed += 1
                warning = f"ko-note-remediation-error:item={item.get('id')} error={error}"
                warnings.append(warning)
                review = update_review_remediation(
                    payload.get("review"),
                    run_id=remediation_run_id,
                    status="failed",
                    warnings=[str(error)],
                    before_quality=before_quality.to_payload(),
                    after_quality=before_quality.to_payload(),
                )
                payload["review"] = review
                self.sqlite_db.update_ko_note_item_payload(int(item["id"]), payload=payload)
                items_out.append(
                    {
                        "id": item.get("id"),
                        "itemType": item.get("item_type"),
                        "status": item.get("status"),
                        "titleKo": item.get("title_ko"),
                        "titleEn": item.get("title_en"),
                        "beforeQualityFlag": before_flag,
                        "beforeScore": before_score,
                        "afterQualityFlag": before_flag,
                        "afterScore": before_score,
                        "improved": False,
                        "changed": False,
                        "strategy": str(strategy or "section"),
                        "targetSections": [],
                        "patchedSections": [],
                        "preservedSectionsCount": 0,
                        "patchedLineCount": 0,
                        "preservedLineCount": 0,
                        "recommendedStrategy": "",
                        "error": str(error),
                    }
                )

        return {
            "schema": "knowledge-hub.ko-note.remediate.result.v1",
            "status": "ok",
            "runId": remediation_run_id,
            "sourceRunId": str(run_id or "").strip(),
            "itemType": str(item_type),
            "qualityFlag": str(quality_flag),
            "itemId": int(item_id or 0),
            "strategy": str(strategy or "section"),
            "attempted": attempted,
            "remediated": remediated,
            "improved": improved,
            "unchanged": unchanged,
            "failed": failed,
            "items": items_out,
            "warnings": warnings,
            "ts": _date_only(),
        }

    def status(self, *, run_id: str) -> dict[str, Any]:
        run = self.sqlite_db.get_ko_note_enrichment_run(run_id)
        if not run:
            return {
                "schema": "knowledge-hub.ko-note.enrich.status.result.v1",
                "status": "failed",
                "runId": str(run_id),
                "counts": {},
                "items": [],
                "ts": _date_only(),
            }
        items = self.sqlite_db.list_ko_note_enrichment_items(run_id=run_id, limit=500)
        counts: dict[str, int] = {}
        for item in items:
            token = str(item.get("status") or "unknown")
            counts[token] = counts.get(token, 0) + 1
        return {
            "schema": "knowledge-hub.ko-note.enrich.status.result.v1",
            "status": str(run.get("status") or "ok"),
            "runId": str(run_id),
            "counts": {
                **counts,
                "sourceTargets": int(run.get("source_target_count") or 0),
                "sourceEnriched": int(run.get("source_enriched_count") or 0),
                "conceptTargets": int(run.get("concept_target_count") or 0),
                "conceptEnriched": int(run.get("concept_enriched_count") or 0),
                "total": len(items),
            },
            "items": [
                {
                    "id": item.get("id"),
                    "itemType": item.get("item_type"),
                    "status": item.get("status"),
                    "noteItemId": item.get("note_item_id"),
                    "targetPath": item.get("target_path"),
                    "route": item.get("route"),
                    "provider": item.get("provider"),
                    "model": item.get("model"),
                }
                for item in items[:50]
            ],
            "warnings": list(run.get("warnings_json") or []),
            "ts": _date_only(),
        }
