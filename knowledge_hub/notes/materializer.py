"""Materialize crawl results into Korean Obsidian notes."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit
from uuid import uuid4

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.notes.contracts import MaterializationRepository
from knowledge_hub.notes.composer import compose_concept_note_payload, compose_source_note_payload
from knowledge_hub.notes.knowledge_label import summarize_ai_knowledge_label
from knowledge_hub.notes.models import KoNoteApproval, KoNoteQuality, KoNoteReview
from knowledge_hub.notes.koreanizer import Koreanizer
from knowledge_hub.notes.workflow_helpers import (
    approval_summary as _approval_summary,
    concept_quality_counts as _concept_quality_counts,
    merge_quality_counts as _merge_quality_counts,
    now_iso as _now_iso,
    remediation_summary as _remediation_summary,
    report_apply_backlog_count as _report_apply_backlog_count,
    review_item_view as _review_item_view,
    review_queue_counts as _review_queue_counts,
    should_apply_concept_item as _should_apply_concept_item,
    source_quality_counts as _source_quality_counts,
)
from knowledge_hub.notes.source_profile import filter_low_signal_evidence
from knowledge_hub.notes.scoring import (
    KO_NOTE_REVIEW_VERSION,
    build_note_review_payload,
    clamp01,
    concept_quality_warnings,
    compute_concept_score,
    compute_evidence_quality,
    compute_ontology_density,
    compute_source_novelty,
    compute_source_score,
    normalize_count,
    score_source_note_markdown,
    score_concept_note_markdown,
    source_quality_warnings,
    translation_level_for_score,
)
from knowledge_hub.notes.templates import (
    build_visible_frontmatter,
    merge_manual_concept_note,
    replace_frontmatter,
    render_concept_note,
    render_source_note,
    safe_file_name,
    slugify_title,
)
from knowledge_hub.knowledge.ai_taxonomy import classify_ai_concept
from knowledge_hub.web.ingest import make_web_note_id


DEFAULT_DOCUMENTS_VAULT = Path(
    "/Users/won/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"
)
CONCEPT_RELATION_PREDICATES = {
    "mentions",
    "uses",
    "related_to",
    "improves",
    "requires",
    "causes",
    "enables",
}


def _safe_json_loads(raw: Any, fallback: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except Exception:
        return fallback


def _attach_concept_quality(payload: dict[str, Any], markdown: str) -> dict[str, Any]:
    updated = dict(payload)
    quality = KoNoteQuality.from_payload(score_concept_note_markdown(markdown, str(updated.get("concept_type") or "generic")))
    warnings = list(dict.fromkeys([*list(updated.get("warnings") or []), *concept_quality_warnings(quality.to_payload())]))
    updated["quality"] = quality.to_payload()
    updated["warnings"] = warnings
    return updated


def _attach_source_quality(payload: dict[str, Any], markdown: str) -> dict[str, Any]:
    updated = dict(payload)
    quality = KoNoteQuality.from_payload(score_source_note_markdown(markdown, str(updated.get("document_type") or "method_paper")))
    warnings = list(dict.fromkeys([*list(updated.get("warnings") or []), *source_quality_warnings(quality.to_payload())]))
    updated["quality"] = quality.to_payload()
    updated["warnings"] = warnings
    return updated


def _attach_review_payload(payload: dict[str, Any], *, item_type: str) -> dict[str, Any]:
    updated = dict(payload)
    updated["review"] = build_note_review_payload(
        item_type=item_type,
        quality=KoNoteQuality.from_payload(updated).to_payload(),
        payload=updated,
        existing_review=KoNoteReview.from_payload(updated).to_payload(),
    )
    return updated


def _payload_quality_flag(payload: dict[str, Any]) -> str:
    return KoNoteQuality.from_payload(payload).flag


def _payload_review(payload: dict[str, Any]) -> dict[str, Any]:
    return KoNoteReview.from_payload(payload).to_payload()


def _payload_approval(payload: dict[str, Any]) -> dict[str, Any]:
    return KoNoteApproval.from_payload(payload).to_payload()


def _payload_review_queue(payload: dict[str, Any]) -> bool:
    return KoNoteReview.from_payload(payload).queue


def _concept_quality_flag(payload: dict[str, Any]) -> str:
    return _payload_quality_flag(payload)


def _word_count(text: str) -> int:
    return len([token for token in re.split(r"\s+", str(text or "").strip()) if token])


def _short_run_id(run_id: str) -> str:
    token = str(run_id or "").strip()
    return token[-8:] if len(token) > 8 else token


def _date_only(value: str | None = None) -> str:
    raw = str(value or "").strip()
    if raw:
        try:
            if raw.endswith("Z"):
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
            return datetime.fromisoformat(raw).date().isoformat()
        except Exception:
            pass
    return datetime.now(timezone.utc).date().isoformat()


def _strip_title_prefix(title: str) -> str:
    token = str(title or "").strip()
    token = re.sub(r"^\[[0-9]{4}\.[0-9]{4,5}(?:v\d+)?\]\s*", "", token).strip()
    return token


def _resolved_path_str(path_str: str) -> str:
    if not str(path_str or "").strip():
        return ""
    return str(_resolve_existing_path(path_str))


def _resolve_existing_path(path_str: str) -> Path:
    path = Path(str(path_str or "")).expanduser()
    if path.exists():
        return path
    parent = path.parent
    if not parent.exists():
        return path
    target_name = path.name
    normalized_targets = {
        unicodedata.normalize("NFC", target_name),
        unicodedata.normalize("NFD", target_name),
    }
    target_stem = path.stem
    target_prefix = target_stem.split("__", 1)[0]
    target_prefix_nfc = unicodedata.normalize("NFC", target_prefix)
    target_prefix_nfd = unicodedata.normalize("NFD", target_prefix)
    for candidate in parent.iterdir():
        candidate_name = candidate.name
        if candidate_name == target_name:
            return candidate
        if unicodedata.normalize("NFC", candidate_name) in normalized_targets:
            return candidate
        if unicodedata.normalize("NFD", candidate_name) in normalized_targets:
            return candidate
        candidate_stem = candidate.stem
        candidate_nfc = unicodedata.normalize("NFC", candidate_stem)
        candidate_nfd = unicodedata.normalize("NFD", candidate_stem)
        prefix_checks = (
            candidate_nfc.startswith(target_prefix_nfc[:32]),
            candidate_nfd.startswith(target_prefix_nfd[:32]),
            target_prefix_nfc.startswith(candidate_nfc[:32]),
            target_prefix_nfd.startswith(candidate_nfd[:32]),
        )
        if any(prefix_checks):
            return candidate
    return path


class KoNoteMaterializer:
    def __init__(
        self,
        config: Config,
        *,
        sqlite_db: MaterializationRepository | None = None,
        sqlite_db_factory: Callable[[], MaterializationRepository] | None = None,
    ):
        self.config = config
        self._sqlite_db_factory = sqlite_db_factory or (lambda: SQLiteDatabase(self.config.sqlite_path))
        self.sqlite_db: MaterializationRepository = sqlite_db or self._sqlite_db_factory()
        self.koreanizer = Koreanizer(self.config)

    def _vault_root(self) -> Path:
        configured = str(self.config.vault_path or "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return DEFAULT_DOCUMENTS_VAULT

    def _staging_root(self, run_id: str) -> Path:
        dt = datetime.now(timezone.utc)
        return (
            self._vault_root()
            / self.config.obsidian_ko_notes_staging_folder
            / dt.strftime("%Y")
            / dt.strftime("%m")
            / dt.strftime("%d")
            / run_id
        )

    def _final_sources_root(self) -> Path:
        return self._vault_root() / self.config.obsidian_web_sources_folder

    def _final_concepts_root(self) -> Path:
        return self._vault_root() / self.config.obsidian_concepts_folder

    def _is_managed_final_path(self, target_path: Path, *, item_type: str, exclude_item_id: int | None = None) -> bool:
        resolved_target = str(target_path.expanduser())
        item = self.sqlite_db.find_ko_note_item_by_final_path(
            final_path=resolved_target,
            item_type=item_type,
            statuses=("approved", "applied"),
        )
        if item and int(item.get("id") or 0) != int(exclude_item_id or 0):
            return True
        for existing in self.sqlite_db.list_existing_ko_note_items(
            item_type=item_type,
            statuses=("approved", "applied"),
            limit=10000,
        ):
            if int(existing.get("id") or 0) == int(exclude_item_id or 0):
                continue
            existing_path = _resolved_path_str(str(existing.get("final_path") or ""))
            if existing_path and existing_path == resolved_target:
                return True
        return False

    def _list_job_records(self, job_id: str, *, state: str = "indexed") -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        offset = 0
        batch_size = 1000
        while True:
            batch = self.sqlite_db.list_crawl_pipeline_records(job_id, state=state, limit=batch_size, offset=offset)
            if not batch:
                break
            items.extend(batch)
            if len(batch) < batch_size:
                break
            offset += len(batch)
        return items

    def _load_normalized(self, record: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(record.get("normalized_path") or "")).expanduser()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _document_view(self, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        normalized = self._load_normalized(record)
        canonical_url = str(
            normalized.get("canonical_url")
            or normalized.get("url")
            or record.get("canonical_url")
            or record.get("source_url")
            or ""
        ).strip()
        note = self.sqlite_db.get_note(make_web_note_id(canonical_url)) if canonical_url else None
        if normalized:
            return normalized, note

        note_meta = self._get_note_metadata(note)
        note_content = str((note or {}).get("content") or "").strip()
        note_title = str((note or {}).get("title") or canonical_url or record.get("source_url") or "Untitled").strip()
        domain = str(record.get("domain") or "").strip()
        if not domain and canonical_url:
            try:
                domain = (urlsplit(canonical_url).netloc or "").strip().lower()
            except Exception:
                domain = ""
        heuristic_quality = 0.74 if note_content else 0.55
        recovered = {
            "record_id": str(record.get("record_id") or note_meta.get("record_id") or ""),
            "url": str(note_meta.get("url") or canonical_url),
            "canonical_url": canonical_url,
            "domain": str(note_meta.get("domain") or domain),
            "fetched_at": str(record.get("fetched_at") or note_meta.get("fetched_at") or ""),
            "quality_score": float(note_meta.get("quality_score") or heuristic_quality),
            "title": note_title,
            "content_text": note_content,
            "recovered_from_note": True,
        }
        return recovered, note

    def _get_note_metadata(self, note: dict[str, Any] | None) -> dict[str, Any]:
        if not note:
            return {}
        return _safe_json_loads(note.get("metadata"), {})

    def _domain_trust(self, domain: str) -> float:
        allowlist = self.config.get_nested("pipeline", "allowlist_domains", default=[]) or []
        if str(domain or "").strip().lower() in {str(item).strip().lower() for item in allowlist}:
            return 1.0
        row = self.sqlite_db.get_crawl_domain_policy(domain)
        if row and str(row.get("status", "")).strip().lower() == "approved":
            return 0.7
        return 0.4

    def _is_official_summary_fallback(self, normalized: dict[str, Any], content_text: str) -> bool:
        domain = str(normalized.get("domain") or "").strip().lower()
        author = str(normalized.get("author") or "").strip().lower()
        text = str(content_text or "").strip()
        if not text:
            return False
        if domain != "openai.com" or author != "openai":
            return False
        has_summary_markers = "Category:" in text and "Published:" in text
        short_summary = len(text) <= 800
        return bool(has_summary_markers and short_summary)

    def _extract_note_concepts(self, note_id: str) -> tuple[list[dict[str, Any]], list[str], list[int]]:
        relations = self.sqlite_db.get_relations("note", note_id)
        concept_relations: list[dict[str, Any]] = []
        concept_ids: list[str] = []
        relation_ids: list[int] = []
        for relation in relations:
            source_type = str(relation.get("source_type", "")).strip()
            target_type = str(relation.get("target_type", "")).strip()
            predicate_id = str(relation.get("predicate_id", relation.get("relation", ""))).strip()
            if predicate_id not in CONCEPT_RELATION_PREDICATES:
                continue
            concept_id = ""
            if source_type == "concept" and target_type == "note":
                concept_id = str(relation.get("source_id", "")).strip()
            elif source_type == "note" and target_type == "concept":
                concept_id = str(relation.get("target_id", "")).strip()
            if not concept_id:
                continue
            concept_relations.append(relation)
            concept_ids.append(concept_id)
            relation_id = relation.get("relation_id", relation.get("id"))
            if relation_id is not None:
                relation_ids.append(int(relation_id))
        deduped_ids = list(dict.fromkeys(concept_ids))
        deduped_relation_ids = list(dict.fromkeys(relation_ids))
        return concept_relations, deduped_ids, deduped_relation_ids

    def _entity_name(self, entity_id: str) -> str:
        entity = self.sqlite_db.get_ontology_entity(entity_id)
        if entity:
            return str(entity.get("canonical_name") or entity_id)
        return entity_id

    def _entity_rows(self, entity_ids: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_entity_id in entity_ids:
            entity_id = str(raw_entity_id or "").strip()
            if not entity_id or entity_id in seen:
                continue
            seen.add(entity_id)
            entity = self.sqlite_db.get_ontology_entity(entity_id)
            if entity:
                rows.append(entity)
        return rows

    def _source_knowledge_label(
        self,
        *,
        note_id: str = "",
        entity_ids: list[str] | None = None,
        title: str = "",
        document_type: str = "",
        domain: str = "",
    ) -> str:
        resolved_entity_ids = list(entity_ids or [])
        if not resolved_entity_ids and note_id:
            _relations, resolved_entity_ids, _relation_ids = self._extract_note_concepts(note_id)
        label = summarize_ai_knowledge_label(self._entity_rows(resolved_entity_ids))
        if label:
            return label

        related_names = [str(entity.get("canonical_name") or "") for entity in self._entity_rows(resolved_entity_ids)]
        classification = classify_ai_concept(
            canonical_name=str(title or "").strip(),
            title=str(title or "").strip(),
            domain=str(domain or "").strip(),
            related_names=related_names,
            source_type="web",
        )
        if classification:
            label = summarize_ai_knowledge_label([{"properties": classification}])
            if label:
                return label

        fallback_kind = {
            "method_paper": "algorithm",
            "blog_tutorial": "algorithm",
            "benchmark": "evaluation",
            "survey_taxonomy": "theory",
            "system_card_safety_report": "operation",
        }.get(str(document_type or "").strip())
        return str(fallback_kind or "").strip()

    def _build_source_frontmatter(
        self,
        *,
        title: str,
        note_id: str = "",
        entity_ids: list[str] | None = None,
        knowledge_label: str = "",
        document_type: str = "",
        domain: str = "",
    ) -> dict[str, Any]:
        resolved_entity_ids = [str(token).strip() for token in (entity_ids or []) if str(token).strip()]
        if not resolved_entity_ids and str(note_id or "").strip():
            _relations, resolved_entity_ids, _relation_ids = self._extract_note_concepts(str(note_id or "").strip())
        explicit_label = summarize_ai_knowledge_label(self._entity_rows(resolved_entity_ids))
        label = str(explicit_label or "").strip()
        extra_fields = {"knowledge_label": label} if label else None
        return build_visible_frontmatter(
            note_type="web-source",
            status="summarized",
            title=title,
            updated=_date_only(),
            extra_fields=extra_fields,
        )

    def _ref_name(self, entity_type: str, entity_id: str) -> str:
        type_token = str(entity_type or "").strip()
        entity_token = str(entity_id or "").strip()
        if type_token == "note":
            note = self.sqlite_db.get_note(entity_token)
            if note:
                return str(note.get("title") or entity_token)
        return self._entity_name(entity_token)

    def _select_key_excerpts(
        self,
        *,
        title: str,
        content_text: str,
        entity_names: list[str],
        relation_lines: list[str],
        max_count: int = 2,
    ) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", str(content_text or "")) if part.strip()]
        if not paragraphs:
            paragraphs = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", str(content_text or "")) if sentence.strip()]
        title_terms = {token for token in re.findall(r"[A-Za-z0-9가-힣]+", str(title or "").lower()) if len(token) >= 3}
        entity_terms = {
            token
            for name in entity_names[:10]
            for token in re.findall(r"[A-Za-z0-9가-힣]+", str(name).lower())
            if len(token) >= 3
        }
        relation_terms = {
            token
            for line in relation_lines[:10]
            for token in re.findall(r"[A-Za-z0-9가-힣]+", str(line).lower())
            if len(token) >= 3
        }
        scored: list[tuple[float, str]] = []
        for paragraph in paragraphs:
            lowered = paragraph.lower()
            score = 0.0
            score += sum(1.0 for token in title_terms if token in lowered)
            score += sum(1.1 for token in entity_terms if token in lowered)
            score += sum(0.8 for token in relation_terms if token in lowered)
            score += min(len(paragraph) / 800.0, 0.8)
            scored.append((score, paragraph[:1200]))
        scored.sort(key=lambda item: item[0], reverse=True)
        excerpts: list[str] = []
        for _, paragraph in scored:
            if paragraph in excerpts:
                continue
            excerpts.append(paragraph)
            if len(excerpts) >= max(1, max_count):
                break
        return excerpts

    def _existing_source_items(self) -> list[dict[str, Any]]:
        return self.sqlite_db.list_existing_ko_note_items(item_type="source", limit=5000)

    def _build_source_candidates(
        self,
        *,
        job_id: str,
        records: list[dict[str, Any]],
        max_candidates: int,
        min_score: float | None = None,
        min_entities: int | None = None,
        min_relations: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        existing_items = self._existing_source_items()
        existing_urls: set[str] = set()
        for item in existing_items:
            raw_urls = item.get("source_urls_json")
            urls = _safe_json_loads(raw_urls, []) if not isinstance(raw_urls, list) else raw_urls
            for url in urls or []:
                token = str(url or "").strip()
                if token:
                    existing_urls.add(token)
        source_threshold = float(
            min_score
            if min_score is not None
            else self.config.get_nested("materialization", "source_score_threshold", default=0.72)
            or 0.72
        )
        min_entities_value = int(
            min_entities
            if min_entities is not None
            else self.config.get_nested("materialization", "source_min_entities", default=2)
            or 0
        )
        min_relations_value = int(
            min_relations
            if min_relations is not None
            else self.config.get_nested("materialization", "source_min_relations", default=0)
            or 0
        )
        summary_fallback_threshold = float(
            self.config.get_nested("materialization", "summary_fallback_source_score_threshold", default=0.46) or 0.46
        )
        candidates: list[dict[str, Any]] = []
        note_context_by_id: dict[str, dict[str, Any]] = {}
        for record in records:
            normalized, note = self._document_view(record)
            if not normalized:
                continue
            canonical_url = str(normalized.get("canonical_url") or normalized.get("url") or record.get("canonical_url") or "").strip()
            note_id = make_web_note_id(canonical_url or str(record.get("source_url") or ""))
            relations, concept_ids, relation_ids = self._extract_note_concepts(note_id)
            entity_names = [self._entity_name(entity_id) for entity_id in concept_ids]
            relation_lines: list[str] = []
            for relation in relations:
                source_name = self._ref_name(
                    str(relation.get("source_type", "")),
                    str(relation.get("source_id", relation.get("source_entity_id", ""))),
                )
                target_name = self._ref_name(
                    str(relation.get("target_type", "")),
                    str(relation.get("target_id", relation.get("target_entity_id", ""))),
                )
                predicate = str(relation.get("predicate_id", relation.get("relation", "related_to")))
                relation_lines.append(f"{source_name} -[{predicate}]-> {target_name}")
            metadata = {
                "url": canonical_url or str(normalized.get("url") or ""),
                "domain": str(normalized.get("domain") or record.get("domain") or ""),
                "record_id": str(normalized.get("record_id") or record.get("record_id") or ""),
                "fetched_at": str(normalized.get("fetched_at") or record.get("fetched_at") or ""),
                "quality_score": float(normalized.get("quality_score") or 0.0),
            }
            if metadata["url"] and metadata["url"] in existing_urls:
                continue
            content_text = str(normalized.get("content_text") or "")
            ontology_density = compute_ontology_density(
                entity_count=len(concept_ids),
                relation_count=len(relation_ids),
                claim_count=0,
                token_count=_word_count(content_text),
            )
            novelty = compute_source_novelty(
                title=str(normalized.get("title") or ""),
                source_url=metadata["url"],
                existing_items=existing_items,
            )
            evidence_quality = compute_evidence_quality(
                title=str(normalized.get("title") or ""),
                content_text=content_text,
                metadata=metadata,
            )
            score = compute_source_score(
                quality_score=float(normalized.get("quality_score") or 0.0),
                ontology_density=ontology_density,
                novelty=novelty,
                domain_trust=self._domain_trust(metadata["domain"]),
                evidence_quality=evidence_quality,
            )
            is_summary_fallback = self._is_official_summary_fallback(normalized, content_text)
            effective_threshold = source_threshold
            effective_min_entities = min_entities_value
            effective_min_relations = min_relations_value
            if is_summary_fallback and self._domain_trust(metadata["domain"]) >= 1.0:
                effective_threshold = min(source_threshold, summary_fallback_threshold)
                effective_min_entities = 0
                effective_min_relations = 0
            if (
                score < effective_threshold
                or len(concept_ids) < max(0, effective_min_entities)
                or len(relation_ids) < max(0, effective_min_relations)
            ):
                continue
            key_excerpts_en = self._select_key_excerpts(
                title=str(normalized.get("title") or ""),
                content_text=content_text,
                entity_names=entity_names,
                relation_lines=relation_lines,
                max_count=2,
            )
            candidate = {
                "job_id": job_id,
                "record_id": str(record.get("record_id") or normalized.get("record_id") or ""),
                "note_id": note_id,
                "title_en": _strip_title_prefix(str(normalized.get("title") or canonical_url or note_id)),
                "source_url": metadata["url"],
                "canonical_url": canonical_url,
                "domain": metadata["domain"],
                "content_text": content_text,
                "quality_score": float(normalized.get("quality_score") or 0.0),
                "candidate_score": score,
                "is_summary_fallback": is_summary_fallback,
                "entity_ids": concept_ids,
                "relation_ids": relation_ids,
                "entity_names": entity_names,
                "relation_lines": relation_lines,
                "key_excerpts_en": key_excerpts_en,
                "raw_path": str(record.get("raw_path") or ""),
                "normalized_path": str(record.get("normalized_path") or ""),
                "indexed_path": str(record.get("indexed_path") or ""),
                "metadata": metadata,
                "note": note or {},
                "note_metadata": self._get_note_metadata(note),
            }
            candidates.append(candidate)
            note_context_by_id[note_id] = candidate
        candidates.sort(key=lambda item: item["candidate_score"], reverse=True)
        return candidates[: max(1, int(max_candidates))], note_context_by_id

    def _build_concept_candidates(
        self,
        *,
        job_id: str,
        support_candidates: list[dict[str, Any]],
        note_context_by_id: dict[str, dict[str, Any]],
        max_candidates: int,
    ) -> list[dict[str, Any]]:
        concept_threshold = float(self.config.get_nested("materialization", "concept_score_threshold", default=0.68) or 0.68)
        support_docs: dict[str, set[str]] = defaultdict(set)
        support_domains: dict[str, set[str]] = defaultdict(set)
        support_predicates: dict[str, set[str]] = defaultdict(set)
        support_relation_ids: dict[str, list[int]] = defaultdict(list)
        support_confidences: dict[str, list[float]] = defaultdict(list)
        support_lines: dict[str, list[str]] = defaultdict(list)
        support_excerpts: dict[str, list[str]] = defaultdict(list)
        source_urls: dict[str, set[str]] = defaultdict(set)
        for candidate in support_candidates:
            note_id = str(candidate.get("note_id") or "")
            domain = str(candidate.get("domain") or "")
            for relation in self.sqlite_db.get_relations("note", note_id):
                predicate = str(relation.get("predicate_id", relation.get("relation", ""))).strip()
                if predicate not in CONCEPT_RELATION_PREDICATES:
                    continue
                concept_id = ""
                if str(relation.get("source_type", "")).strip() == "concept" and str(relation.get("target_type", "")).strip() == "note":
                    concept_id = str(relation.get("source_id", "")).strip()
                elif str(relation.get("source_type", "")).strip() == "note" and str(relation.get("target_type", "")).strip() == "concept":
                    concept_id = str(relation.get("target_id", "")).strip()
                if not concept_id:
                    continue
                support_docs[concept_id].add(note_id)
                if domain:
                    support_domains[concept_id].add(domain)
                support_predicates[concept_id].add(predicate)
                relation_id = relation.get("relation_id", relation.get("id"))
                if relation_id is not None:
                    support_relation_ids[concept_id].append(int(relation_id))
                support_confidences[concept_id].append(float(relation.get("confidence") or 0.0))
                title = str(candidate.get("title_en") or note_id)
                url = str(candidate.get("source_url") or "")
                support_lines[concept_id].append(f"{title} ({domain or 'unknown-domain'}) - {url}")
                support_excerpts[concept_id].extend(candidate.get("key_excerpts_en") or [])
                source_urls[concept_id].add(url)
        concept_candidates: list[dict[str, Any]] = []
        for concept_id, doc_ids in support_docs.items():
            if len(doc_ids) < 3:
                continue
            relation_degree_raw = len(self.sqlite_db.get_relations("concept", concept_id))
            if relation_degree_raw < 2:
                continue
            support_doc_count_norm = normalize_count(len(doc_ids), 8)
            diversity = clamp01((normalize_count(len(support_domains[concept_id]), 4) + normalize_count(len(support_predicates[concept_id]), 4)) / 2.0)
            relation_degree = normalize_count(relation_degree_raw, 8)
            avg_confidence = clamp01(sum(support_confidences[concept_id]) / max(1, len(support_confidences[concept_id])))
            score = compute_concept_score(
                support_doc_count_norm=support_doc_count_norm,
                evidence_diversity=diversity,
                relation_degree=relation_degree,
                avg_confidence=avg_confidence,
            )
            if score < concept_threshold:
                continue
            entity = self.sqlite_db.get_ontology_entity(concept_id) or {"canonical_name": concept_id}
            aliases = self.sqlite_db.get_entity_aliases(concept_id)
            relation_lines: list[str] = []
            related_concepts: list[str] = []
            for relation in self.sqlite_db.get_relations("concept", concept_id):
                predicate = str(relation.get("predicate_id", relation.get("relation", "related_to")))
                source_name = self._ref_name(
                    str(relation.get("source_type", "")),
                    str(relation.get("source_id", relation.get("source_entity_id", ""))),
                )
                target_name = self._ref_name(
                    str(relation.get("target_type", "")),
                    str(relation.get("target_id", relation.get("target_entity_id", ""))),
                )
                relation_lines.append(f"{source_name} -[{predicate}]-> {target_name}")
                if str(relation.get("source_type", "")).strip() == "concept":
                    related_concepts.append(source_name)
                if str(relation.get("target_type", "")).strip() == "concept":
                    related_concepts.append(target_name)
            concept_candidates.append(
                {
                    "job_id": job_id,
                    "entity_id": concept_id,
                    "title": str(entity.get("canonical_name") or concept_id),
                    "aliases": aliases,
                    "candidate_score": score,
                    "support_doc_count": len(doc_ids),
                    "relation_degree_raw": relation_degree_raw,
                    "entity_ids": [concept_id],
                    "relation_ids": list(dict.fromkeys(support_relation_ids[concept_id])),
                    "support_lines": list(dict.fromkeys(support_lines[concept_id]))[:10],
                    "relation_lines": list(dict.fromkeys(relation_lines))[:12],
                    "key_excerpts_en": list(dict.fromkeys(support_excerpts[concept_id]))[:2],
                    "source_urls": sorted(url for url in source_urls[concept_id] if url),
                    "related_concepts": [f"- {name}" for name in list(dict.fromkeys(item for item in related_concepts if item and item != entity.get("canonical_name")))[:10]],
                    "related_sources": [f"- {line}" for line in list(dict.fromkeys(support_lines[concept_id]))[:10]],
                }
            )
        concept_candidates.sort(key=lambda item: item["candidate_score"], reverse=True)
        return concept_candidates[: max(1, int(max_candidates))]

    def _render_source_item(self, run_id: str, candidate: dict[str, Any]) -> tuple[dict[str, Any], str, Path, Path]:
        filtered_key_excerpts = filter_low_signal_evidence(list(candidate["key_excerpts_en"]), limit=4) or list(candidate["key_excerpts_en"])
        translation_level = translation_level_for_score(
            candidate["candidate_score"],
            key_excerpt_threshold=float(self.config.get_nested("materialization", "key_excerpt_threshold", default=0.82) or 0.82),
        )
        korean = self.koreanizer.build_source_summary(
            title_en=candidate["title_en"],
            content_text=candidate["content_text"],
            entity_names=candidate["entity_names"],
            relation_lines=candidate["relation_lines"],
            key_excerpts_en=filtered_key_excerpts,
            candidate_score=candidate["candidate_score"],
            translation_level=translation_level,
        )
        knowledge_label = self._source_knowledge_label(
            note_id=str(candidate.get("note_id") or ""),
            entity_ids=list(candidate.get("entity_ids") or []),
        )
        frontmatter = self._build_source_frontmatter(
            title=candidate["title_en"],
            note_id=str(candidate.get("note_id") or ""),
            entity_ids=list(candidate.get("entity_ids") or []),
            knowledge_label=knowledge_label,
            document_type=str(candidate.get("document_type") or ""),
            domain=str(candidate.get("domain") or ""),
        )
        related_concepts = [f"- {name}" for name in candidate["entity_names"][:12]] or ["- 관련 개념 없음"]
        claim_lines = []
        for claim in self.sqlite_db.list_claims_by_note(str(candidate.get("note_id") or ""), limit=8):
            claim_text = str(claim.get("claim_text") or "").strip()
            if claim_text:
                claim_lines.append(f"- {claim_text}")
        claim_lines = filter_low_signal_evidence(claim_lines, limit=6)
        metadata_lines = [
            item
            for item in (
                f"url={candidate['source_url']}" if candidate.get("source_url") else "",
                f"domain={str(candidate.get('source_url') or '').split('/')[2]}" if "://" in str(candidate.get("source_url") or "") else "",
            )
            if item
        ]
        payload = compose_source_note_payload(
            {
            "frontmatter": frontmatter,
            "knowledge_label": knowledge_label,
            "title_en": candidate["title_en"],
            "title_ko": korean["title_ko"],
            "summary_ko": korean["summary_ko"],
            "core_summary": korean["summary_ko"],
            "summary_line_ko": korean["summary_line_ko"],
            "entity_lines": korean["entity_lines"],
            "relation_lines": korean["relation_lines"],
            "claim_lines": claim_lines,
            "key_excerpts_ko": korean["key_excerpts_ko"],
            "key_excerpts_en": filter_low_signal_evidence(korean["key_excerpts_en"], limit=4) or korean["key_excerpts_en"],
            "related_concepts": related_concepts,
            "source_url": candidate["source_url"],
            "record_id": candidate["record_id"],
            "note_id": candidate["note_id"],
            "crawl_job_id": candidate["job_id"],
            "candidate_score": round(float(candidate["candidate_score"]), 4),
            "translation_level": translation_level,
            "raw_path": candidate["raw_path"],
            "normalized_path": candidate["normalized_path"],
            "indexed_path": candidate["indexed_path"],
            "warnings": korean["warnings"],
            },
            content_text=candidate["content_text"],
            entity_names=candidate["entity_names"],
            metadata_lines=metadata_lines,
            section_fallbacks={
                "contributions": list(korean.get("contributions") or []),
                "methodology": list(korean.get("methodology") or []),
                "results_or_findings": list(korean.get("key_results") or []),
                "limitations": list(korean.get("limitations") or []),
                "insights": list(korean.get("insights") or []),
            },
        )
        markdown = render_source_note(payload)
        payload = _attach_source_quality(payload, markdown)
        payload = _attach_review_payload(payload, item_type="source")
        staging_name = f"{slugify_title(korean['title_ko'] or candidate['title_en'])}__{candidate['record_id'][:10]}.md"
        staging_path = self._staging_root(run_id) / "sources" / staging_name
        final_name = f"{safe_file_name(korean['title_ko'] or candidate['title_en'])}.md"
        final_path = self._final_sources_root() / final_name
        return payload, markdown, staging_path, final_path

    def _render_concept_item(self, run_id: str, candidate: dict[str, Any]) -> tuple[dict[str, Any], str, Path, Path]:
        translation_level = translation_level_for_score(
            candidate["candidate_score"],
            key_excerpt_threshold=float(self.config.get_nested("materialization", "key_excerpt_threshold", default=0.82) or 0.82),
        )
        korean = self.koreanizer.build_concept_summary(
            canonical_name=candidate["title"],
            aliases=candidate["aliases"],
            support_lines=candidate["support_lines"],
            relation_lines=candidate["relation_lines"],
            key_excerpts_en=candidate["key_excerpts_en"],
            candidate_score=candidate["candidate_score"],
            translation_level=translation_level,
        )
        frontmatter = build_visible_frontmatter(
            note_type="concept",
            status="enriched",
            title=candidate["title"],
            updated=_date_only(),
        )
        payload = compose_concept_note_payload(
            {
                "frontmatter": frontmatter,
                "title": candidate["title"],
                "title_ko": korean["title_ko"],
                "summary_ko": korean["summary_ko"],
                "core_summary": korean["summary_ko"],
                "relation_lines": korean["relation_lines"],
                "claim_lines": [
                    f"- {str(claim.get('claim_text') or '').strip()}"
                    for claim in self.sqlite_db.list_claims_by_entity(str(candidate.get("entity_id") or ""), limit=8)
                    if str(claim.get("claim_text") or "").strip()
                ],
                "support_lines": korean["support_lines"],
                "key_excerpts_ko": korean["key_excerpts_ko"],
                "key_excerpts_en": korean["key_excerpts_en"],
                "related_sources": candidate["related_sources"],
                "related_concepts": candidate["related_concepts"],
                "why_it_matters": korean.get("why_it_matters") or [],
                "entity_id": candidate["entity_id"],
                "candidate_score": round(float(candidate["candidate_score"]), 4),
                "support_doc_count": int(candidate["support_doc_count"]),
                "translation_level": translation_level,
                "warnings": korean["warnings"],
            },
            aliases=candidate["aliases"],
        )
        markdown = render_concept_note(payload)
        payload = _attach_concept_quality(payload, markdown)
        payload = _attach_review_payload(payload, item_type="concept")
        staging_name = f"{safe_file_name(candidate['title'])}.md"
        staging_path = self._staging_root(run_id) / "concepts" / staging_name
        final_path = self._final_concepts_root() / staging_name
        return payload, markdown, staging_path, final_path

    def generate_for_job(
        self,
        *,
        job_id: str,
        max_source_notes: int | None = None,
        max_concept_notes: int | None = None,
        allow_external: bool = False,
        llm_mode: str = "auto",
        local_timeout_sec: int | None = None,
        api_fallback_on_timeout: bool = True,
        enrich: bool | None = None,
    ) -> dict[str, Any]:
        job = self.sqlite_db.get_crawl_pipeline_job(job_id)
        ts = _now_iso()
        if not job:
            return {
                "schema": "knowledge-hub.ko-note.generate.result.v1",
                "status": "failed",
                "runId": "",
                "crawlJobId": str(job_id),
                "sourceCandidates": 0,
                "sourceGenerated": 0,
                "conceptCandidates": 0,
                "conceptGenerated": 0,
                "blocked": 1,
                "warnings": [f"crawl job not found: {job_id}"],
                "ts": ts,
            }
        run_id = f"ko_note_{uuid4().hex[:12]}"
        stale_after_sec = int(
            self.config.get_nested("materialization", "run_stale_after_sec", default=1800) or 1800
        )
        warnings: list[str] = []
        try:
            stale_rows = self.sqlite_db.list_stale_ko_note_runs(
                status="running",
                updated_before_seconds=stale_after_sec,
            )
            for row in stale_rows:
                self.sqlite_db.update_ko_note_run(
                    str(row["run_id"]),
                    status="failed",
                    warnings_json=["stale-run-recovered"],
                )
                warnings.append("stale-run-recovered")
        except Exception:
            pass

        self.koreanizer = Koreanizer(
            self.config,
            allow_external=allow_external,
            llm_mode=llm_mode,
            local_timeout_sec=local_timeout_sec,
            api_fallback_on_timeout=api_fallback_on_timeout,
        )
        source_cap = int(
            max_source_notes
            if max_source_notes is not None
            else self.config.get_nested("materialization", "max_source_notes_per_run", default=50)
        )
        concept_cap = int(
            max_concept_notes
            if max_concept_notes is not None
            else self.config.get_nested("materialization", "max_concept_notes_per_run", default=30)
        )
        records = self._list_job_records(job_id, state="indexed")
        source_candidates, note_context_by_id = self._build_source_candidates(
            job_id=job_id,
            records=records,
            max_candidates=max(1, source_cap),
        )
        concept_support_threshold = float(
            self.config.get_nested(
                "materialization",
                "concept_support_source_score_threshold",
                default=0.62,
            )
            or 0.62
        )
        concept_support_candidates, _ = self._build_source_candidates(
            job_id=job_id,
            records=records,
            max_candidates=max(1, len(records)),
            min_score=concept_support_threshold,
            min_entities=1,
            min_relations=1,
        )
        concept_candidates = self._build_concept_candidates(
            job_id=job_id,
            support_candidates=concept_support_candidates,
            note_context_by_id=note_context_by_id,
            max_candidates=max(1, concept_cap),
        )
        self.sqlite_db.create_ko_note_run(
            run_id=run_id,
            crawl_job_id=job_id,
            source_candidates=len(source_candidates),
            concept_candidates=len(concept_candidates),
            status="running",
            warnings=[],
        )

        source_generated = 0
        concept_generated = 0
        run_status = "completed"
        try:
            for candidate in source_candidates:
                payload, markdown, staging_path, final_path = self._render_source_item(run_id, candidate)
                staging_path.parent.mkdir(parents=True, exist_ok=True)
                staging_path.write_text(markdown, encoding="utf-8")
                self.sqlite_db.add_ko_note_item(
                    run_id=run_id,
                    item_type="source",
                    item_key=str(candidate["record_id"]),
                    status="staged",
                    job_id=job_id,
                    record_id=str(candidate["record_id"]),
                    note_id=str(candidate["note_id"]),
                    title_en=str(candidate["title_en"]),
                    title_ko=str(payload["title_ko"]),
                    candidate_score=float(candidate["candidate_score"]),
                    translation_level=str(payload.get("translation_level") or "T1"),
                    source_urls=[str(candidate["source_url"])],
                    evidence_ptrs=[],
                    entity_ids=list(candidate["entity_ids"]),
                    relation_ids=[str(item) for item in candidate["relation_ids"]],
                    payload=payload,
                    staging_path=str(staging_path),
                    final_path=str(final_path),
                )
                warnings.extend(payload.get("warnings") or [])
                source_generated += 1

            for candidate in concept_candidates:
                payload, markdown, staging_path, final_path = self._render_concept_item(run_id, candidate)
                staging_path.parent.mkdir(parents=True, exist_ok=True)
                staging_path.write_text(markdown, encoding="utf-8")
                self.sqlite_db.add_ko_note_item(
                    run_id=run_id,
                    item_type="concept",
                    item_key=str(candidate["entity_id"]),
                    status="staged",
                    job_id=job_id,
                    entity_id=str(candidate["entity_id"]),
                    title_en=str(candidate["title"]),
                    title_ko=str(payload["title_ko"]),
                    candidate_score=float(candidate["candidate_score"]),
                    translation_level=str(payload.get("translation_level") or "T1"),
                    source_urls=list(candidate["source_urls"]),
                    evidence_ptrs=[],
                    entity_ids=[str(candidate["entity_id"])],
                    relation_ids=[str(item) for item in candidate["relation_ids"]],
                    payload=payload,
                    staging_path=str(staging_path),
                    final_path=str(final_path),
                )
                warnings.extend(payload.get("warnings") or [])
                concept_generated += 1

            if not source_generated and not concept_generated:
                run_status = "partial"
                warnings.append("no-ko-note-candidates-generated")
        except Exception as error:
            warnings.append(f"materializer-error: {error}")
            run_status = "partial" if (source_generated or concept_generated) else "failed"
        finally:
            self.sqlite_db.update_ko_note_run(
                run_id,
                status=run_status,
                source_generated=source_generated,
                concept_generated=concept_generated,
                warnings_json=list(dict.fromkeys(warnings)),
            )
        result = {
            "schema": "knowledge-hub.ko-note.generate.result.v1",
            "status": run_status,
            "runId": run_id,
            "crawlJobId": str(job_id),
            "sourceCandidates": len(source_candidates),
            "sourceGenerated": source_generated,
            "conceptCandidates": len(concept_candidates),
            "conceptGenerated": concept_generated,
            "blocked": 0,
            "warnings": list(dict.fromkeys(warnings)),
            "ts": ts,
        }
        should_enrich = (
            bool(enrich)
            if enrich is not None
            else bool(self.config.get_nested("materialization", "enrichment", "enabled", default=True))
        )
        if should_enrich and (source_generated or concept_generated):
            from knowledge_hub.notes.enricher import KoNoteEnricher

            enrich_payload = KoNoteEnricher(self.config, sqlite_db=self.sqlite_db).enrich(
                scope="new",
                run_id=run_id,
                item_type="all",
                limit_source=max(1, source_generated),
                limit_concept=max(1, concept_generated),
                allow_external=allow_external,
                llm_mode=llm_mode,
                local_timeout_sec=local_timeout_sec,
                api_fallback_on_timeout=api_fallback_on_timeout,
            )
            result["enrichmentRunId"] = enrich_payload.get("runId", "")
            result["sourceEnriched"] = enrich_payload.get("sourceEnriched", 0)
            result["conceptEnriched"] = enrich_payload.get("conceptEnriched", 0)
            for warning in enrich_payload.get("warnings", []):
                if warning not in result["warnings"]:
                    result["warnings"].append(warning)
        return result

    def status(self, *, run_id: str) -> dict[str, Any]:
        run = self.sqlite_db.get_ko_note_run(run_id)
        ts = _now_iso()
        if not run:
            return {
                "schema": "knowledge-hub.ko-note.status.result.v1",
                "status": "failed",
                "runId": str(run_id),
                "counts": {},
                "paths": {},
                "items": [],
                "ts": ts,
            }
        items = self.sqlite_db.list_ko_note_items(run_id=run_id, limit=2000)
        counts: dict[str, int] = defaultdict(int)
        type_counts: dict[str, int] = defaultdict(int)
        paths: dict[str, str] = {}
        concept_quality_counts = _concept_quality_counts(items)
        source_quality_counts = _source_quality_counts(items)
        combined_quality_counts = _merge_quality_counts(concept_quality_counts, source_quality_counts)
        review_queue = _review_queue_counts(items)
        for item in items:
            counts[str(item.get("status", "unknown"))] += 1
            type_counts[str(item.get("item_type", "unknown"))] += 1
            if not paths.get("stagingRoot") and str(item.get("staging_path") or "").strip():
                paths["stagingRoot"] = str(Path(str(item["staging_path"])).parent.parent)
        return {
            "schema": "knowledge-hub.ko-note.status.result.v1",
            "status": str(run.get("status") or "ok"),
            "runId": str(run_id),
            "counts": {
                **counts,
                "source": type_counts.get("source", 0),
                "concept": type_counts.get("concept", 0),
                "total": len(items),
            },
            "paths": paths,
            "items": [_review_item_view(item) for item in items[:50]],
            "quality": {
                **concept_quality_counts,
                "concept": concept_quality_counts,
                "source": source_quality_counts,
                "combined": combined_quality_counts,
            },
            "reviewQueue": review_queue,
            "ts": ts,
        }

    def report(self, *, run_id: str, recent_runs: int = 10) -> dict[str, Any]:
        from knowledge_hub.application.ko_note_reports import build_ko_note_report

        return build_ko_note_report(self.sqlite_db, run_id=run_id, recent_runs=recent_runs)

    def _apply_source_item(self, item: dict[str, Any], run_id: str) -> tuple[str, str | None]:
        staging_path = _resolve_existing_path(str(item.get("staging_path") or ""))
        if not staging_path.exists():
            return "missing-staging", None
        payload = item.get("payload_json") or {}
        clean_title_en = _strip_title_prefix(str((payload or {}).get("title_en") or item.get("title_en") or ""))
        clean_title_ko = _strip_title_prefix(str((payload or {}).get("title_ko") or item.get("title_ko") or clean_title_en))
        raw_final_path = str(item.get("final_path") or "").strip()
        if raw_final_path:
            target_path = Path(raw_final_path).expanduser()
            if re.match(r"^\[[0-9]{4}\.[0-9]{4,5}(?:v\d+)?\]\s*", target_path.name):
                target_path = self._final_sources_root() / f"{safe_file_name(clean_title_ko or clean_title_en)}.md"
        else:
            target_path = self._final_sources_root() / f"{safe_file_name(clean_title_ko or clean_title_en)}.md"
        if payload:
            payload = dict(payload)
            payload["title_en"] = clean_title_en
            payload["title_ko"] = clean_title_ko or clean_title_en
            payload["frontmatter"] = self._build_source_frontmatter(
                title=clean_title_en,
                note_id=str(payload.get("note_id") or item.get("note_id") or ""),
                entity_ids=list(payload.get("entity_ids") or []),
                knowledge_label=str(payload.get("knowledge_label") or ""),
                document_type=str(payload.get("document_type") or ""),
                domain=str(payload.get("domain") or ""),
            )
            content = render_source_note(payload)
        else:
            content = staging_path.read_text(encoding="utf-8")
            content = replace_frontmatter(
                content,
                self._build_source_frontmatter(
                    title=clean_title_en,
                    note_id=str(item.get("note_id") or ""),
                    document_type=str((payload or {}).get("document_type") or ""),
                ),
            )
        if not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            return "applied", str(target_path)
        if self._is_managed_final_path(target_path, item_type="source"):
            target_path.write_text(content, encoding="utf-8")
            return "applied", str(target_path)
        suffixed = target_path.with_name(f"{target_path.stem}__run_{_short_run_id(run_id)}{target_path.suffix}")
        suffixed.parent.mkdir(parents=True, exist_ok=True)
        if suffixed.exists() and suffixed.read_text(encoding="utf-8") == content:
            return "skipped", str(suffixed)
        suffixed.write_text(content, encoding="utf-8")
        return "conflict-copy", str(suffixed)

    def _apply_concept_item(self, item: dict[str, Any]) -> tuple[str, str | None]:
        payload = item.get("payload_json") or {}
        staging_path = _resolve_existing_path(str(item.get("staging_path") or ""))
        target_path = Path(str(item.get("final_path") or "")).expanduser()
        if not staging_path.exists():
            return "missing-staging", None
        if payload:
            payload = dict(payload)
            payload["frontmatter"] = build_visible_frontmatter(
                note_type="concept",
                status="enriched",
                title=str(payload.get("title") or item.get("title_en") or ""),
                updated=_date_only(),
            )
            staged_content = render_concept_note(payload)
        else:
            staged_content = staging_path.read_text(encoding="utf-8")
            staged_content = replace_frontmatter(
                staged_content,
                build_visible_frontmatter(
                    note_type="concept",
                    status="enriched",
                    title=str(item.get("title_en") or ""),
                    updated=_date_only(),
                ),
            )
        if not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(staged_content, encoding="utf-8")
            return "applied", str(target_path)
        existing = target_path.read_text(encoding="utf-8")
        if "<!-- SECTION:khub-auto-summary:start -->" in existing:
            merged = merge_manual_concept_note(existing, payload)
            if merged == existing:
                return "skipped", str(target_path)
            target_path.write_text(merged, encoding="utf-8")
            return "merged", str(target_path)
        exclude_item_id = None
        if str(dict(payload or {}).get("approval", {}).get("mode") or "") == "auto":
            exclude_item_id = int(item.get("id") or 0)
        if self._is_managed_final_path(target_path, item_type="concept", exclude_item_id=exclude_item_id):
            if existing == staged_content:
                return "skipped", str(target_path)
            target_path.write_text(staged_content, encoding="utf-8")
            return "applied", str(target_path)
        merged = merge_manual_concept_note(existing, payload)
        if merged == existing:
            return "skipped", str(target_path)
        target_path.write_text(merged, encoding="utf-8")
        return "merged", str(target_path)

    def apply(
        self,
        *,
        run_id: str,
        item_type: str = "all",
        limit: int = 0,
        only_approved: bool = True,
    ) -> dict[str, Any]:
        from knowledge_hub.notes.applier import KoNoteApplier

        return KoNoteApplier(self).apply(
            run_id=run_id,
            item_type=item_type,
            limit=limit,
            only_approved=only_approved,
        )

    def reject(
        self,
        *,
        run_id: str,
        item_type: str = "all",
        limit: int = 0,
    ) -> dict[str, Any]:
        from knowledge_hub.notes.applier import KoNoteApplier

        return KoNoteApplier(self).reject(run_id=run_id, item_type=item_type, limit=limit)

    def review_list(
        self,
        *,
        run_id: str,
        item_type: str = "all",
        quality_flag: str = "all",
        limit: int = 50,
    ) -> dict[str, Any]:
        from knowledge_hub.notes.review import KoNoteReviewService

        return KoNoteReviewService(self.sqlite_db).review_list(
            run_id=run_id,
            item_type=item_type,
            quality_flag=quality_flag,
            limit=limit,
        )

    def review_approve(self, *, item_id: int, reviewer: str = "cli-user", note: str = "") -> dict[str, Any]:
        from knowledge_hub.notes.review import KoNoteReviewService

        return KoNoteReviewService(self.sqlite_db).review_approve(
            item_id=int(item_id),
            reviewer=str(reviewer or "cli-user"),
            note=str(note or ""),
        )

    def review_reject(self, *, item_id: int, reviewer: str = "cli-user", note: str = "") -> dict[str, Any]:
        from knowledge_hub.notes.review import KoNoteReviewService

        return KoNoteReviewService(self.sqlite_db).review_reject(
            item_id=int(item_id),
            reviewer=str(reviewer or "cli-user"),
            note=str(note or ""),
        )
