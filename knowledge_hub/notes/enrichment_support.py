"""Support helpers for `KoNoteEnricher`.

Keep `KoNoteEnricher` as the stable facade while moving low-risk read-only,
pack-assembly, and pure merge-support helpers out of the main file.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from knowledge_hub.knowledge.quality_mode import resolve_quality_mode_route
from knowledge_hub.learning.task_router import decide_task_route
from knowledge_hub.notes.materializer import _resolve_existing_path
from knowledge_hub.notes.models import KoNoteQuality, KoNoteReview
from knowledge_hub.notes.scoring import (
    build_note_remediation_targets,
    line_diagnostics_for_payload_field,
    merge_targeted_lines,
    remediation_line_minimum,
    remediation_preserve_sections,
    should_replace_scalar,
)
from knowledge_hub.notes.source_profile import (
    extract_thesis,
    filter_low_signal_evidence,
    infer_document_type,
    representative_sources,
    synthesize_evidence_sections,
)
from knowledge_hub.notes.templates import split_frontmatter


ENRICHMENT_VERSION = "v1"
_SOURCE_QUALITY_PRIORITY = {
    "ok": 0,
    "needs_review": 1,
    "reject": 2,
    "unscored": 3,
}


def get_record(service, *, job_id: str, record_id: str) -> dict[str, Any] | None:
    return service.sqlite_db.get_crawl_pipeline_record(str(job_id or ""), str(record_id or ""))


def frontmatter_title(content: str) -> str:
    frontmatter, _ = split_frontmatter(content)
    return str(frontmatter.get("title") or "").strip()


def build_source_pack_data(service, item: dict[str, Any]) -> dict[str, Any]:
    payload = dict(item.get("payload_json") or {})
    job_id = str(item.get("job_id") or payload.get("crawl_job_id") or "")
    record_id = str(item.get("record_id") or payload.get("record_id") or "")
    record = service._get_record(job_id=job_id, record_id=record_id) or {}
    normalized, note = service.materializer._document_view(record) if record else ({}, service.sqlite_db.get_note(str(item.get("note_id") or "")))
    note_id = str(item.get("note_id") or payload.get("note_id") or "")
    relation_lines = [str(line).lstrip("- ").strip() for line in (payload.get("relation_lines") or []) if str(line).strip()]
    entity_names = [str(line).lstrip("- ").strip() for line in (payload.get("entity_lines") or payload.get("related_concepts") or []) if str(line).strip()]
    if note_id and (not relation_lines or not entity_names):
        relations, concept_ids, _relation_ids = service.materializer._extract_note_concepts(note_id)
        if not relation_lines:
            for relation in relations:
                source_name = service.materializer._ref_name(
                    str(relation.get("source_type", "")),
                    str(relation.get("source_id", relation.get("source_entity_id", ""))),
                )
                target_name = service.materializer._ref_name(
                    str(relation.get("target_type", "")),
                    str(relation.get("target_id", relation.get("target_entity_id", ""))),
                )
                predicate = str(relation.get("predicate_id", relation.get("relation", "related_to")))
                relation_lines.append(f"{source_name} -[{predicate}]-> {target_name}")
        if not entity_names:
            entity_names = [service.materializer._entity_name(entity_id) for entity_id in concept_ids]
    entity_ids = [str(token) for token in (item.get("entity_ids_json") or []) if str(token).strip()]
    claim_lines: list[str] = []
    for entity_id in entity_ids[:6]:
        for claim in service.sqlite_db.list_claims(subject_id=entity_id, limit=3):
            claim_text = str(claim.get("claim_text") or "").strip()
            if claim_text:
                claim_lines.append(claim_text)
        for claim in service.sqlite_db.list_claims(object_id=entity_id, limit=3):
            claim_text = str(claim.get("claim_text") or "").strip()
            if claim_text:
                claim_lines.append(claim_text)
    claim_lines = list(dict.fromkeys(claim_lines))[:8]
    content_text = str(
        normalized.get("content_text")
        or (note or {}).get("content")
        or payload.get("source_content_text")
        or payload.get("original_content_text")
        or payload.get("core_summary")
        or ""
    )
    excerpt_count = int(service.config.get_nested("materialization", "enrichment", "source_excerpt_count", default=4) or 4)
    key_excerpts_en = [
        str(item_).strip()
        for item_ in (payload.get("source_key_excerpts_en") or payload.get("key_excerpts_en") or [])
        if str(item_).strip()
    ]
    if len(key_excerpts_en) < excerpt_count:
        key_excerpts_en = service.materializer._select_key_excerpts(
            title=str(payload.get("title_en") or item.get("title_en") or normalized.get("title") or ""),
            content_text=content_text,
            entity_names=entity_names,
            relation_lines=relation_lines,
            max_count=excerpt_count,
        )
    key_excerpts_en = filter_low_signal_evidence(key_excerpts_en, limit=excerpt_count) or key_excerpts_en[:excerpt_count]
    metadata_lines = []
    source_url = str(payload.get("source_url") or normalized.get("canonical_url") or normalized.get("url") or "")
    domain = str(normalized.get("domain") or record.get("domain") or "")
    if source_url:
        metadata_lines.append(f"url={source_url}")
    if domain:
        metadata_lines.append(f"domain={domain}")
    published = str(normalized.get("published_at") or normalized.get("fetched_at") or record.get("fetched_at") or "")
    if published:
        metadata_lines.append(f"published_or_fetched={published}")
    if normalized.get("quality_score") is not None:
        metadata_lines.append(f"quality={normalized.get('quality_score')}")
    document_type = infer_document_type(
        title=str(payload.get("title_en") or item.get("title_en") or normalized.get("title") or ""),
        source_url=source_url,
        domain=domain,
        content_text=content_text,
        key_excerpts=key_excerpts_en,
        metadata_lines=metadata_lines,
        claim_lines=claim_lines,
    )
    thesis = extract_thesis(
        title=str(payload.get("title_en") or item.get("title_en") or normalized.get("title") or ""),
        document_type=document_type,
        content_text=content_text,
        claim_lines=claim_lines,
        key_excerpts=key_excerpts_en,
    )
    synthesized = synthesize_evidence_sections(
        document_type=document_type,
        title=str(payload.get("title_en") or item.get("title_en") or normalized.get("title") or ""),
        thesis=thesis,
        content_text=content_text,
        entity_names=entity_names,
        relation_lines=relation_lines,
        claim_lines=claim_lines,
        related_concepts=[str(line).lstrip("- ").strip() for line in (payload.get("related_concepts") or []) if str(line).strip()][:12],
        key_excerpts=key_excerpts_en,
        metadata_lines=metadata_lines,
    )
    return {
        "note_item_id": int(item.get("id") or 0),
        "title_en": str(payload.get("title_en") or item.get("title_en") or normalized.get("title") or ""),
        "title_ko": str(payload.get("title_ko") or item.get("title_ko") or payload.get("title_en") or item.get("title_en") or ""),
        "source_url": source_url,
        "domain": domain,
        "metadata_lines": metadata_lines,
        "entity_names": entity_names[:12],
        "relation_lines": relation_lines[:12],
        "claim_lines": claim_lines[:8],
        "related_concepts": [str(line).lstrip("- ").strip() for line in (payload.get("related_concepts") or []) if str(line).strip()][:12],
        "key_excerpts_en": key_excerpts_en[:excerpt_count],
        "content_text": str(content_text or "")[: int(service.config.get_nested("materialization", "enrichment", "source_context_chars", default=14000) or 14000)],
        "document_type": document_type,
        "thesis": thesis,
        "top_claims": synthesized["top_claims"],
        "core_concepts": synthesized["core_concepts"],
        "contributions": synthesized["contributions"],
        "methodology": synthesized["methodology"],
        "results_or_findings": synthesized["results_or_findings"],
        "limitations": synthesized["limitations"],
        "representative_sources": representative_sources(
            title=str(payload.get("title_en") or item.get("title_en") or normalized.get("title") or ""),
            source_url=source_url,
            domain=domain,
            metadata_lines=metadata_lines,
        ),
        "candidate_score": float(item.get("candidate_score") or payload.get("candidate_score") or 0.0),
        "translation_level": str(item.get("translation_level") or payload.get("translation_level") or "T1"),
    }


def get_source_quality_index(service) -> dict[str, dict[str, Any]]:
    if service._source_quality_index is not None:
        return service._source_quality_index
    index: dict[str, dict[str, Any]] = {}
    items = service.sqlite_db.list_existing_ko_note_items(
        item_type="source",
        statuses=("staged", "approved", "applied"),
        limit=5000,
    )
    for item in items:
        note_id = str(item.get("note_id") or "")
        if not note_id:
            continue
        payload = dict(item.get("payload_json") or {})
        quality = dict(payload.get("quality") or {})
        flag = str(quality.get("flag") or "").strip() or "unscored"
        if flag not in _SOURCE_QUALITY_PRIORITY:
            flag = "unscored"
        score = float(quality.get("score") or 0.0)
        max_score = max(1.0, float(quality.get("max_score") or 1.0))
        normalized_score = score / max_score
        current = index.get(note_id)
        current_rank = _SOURCE_QUALITY_PRIORITY.get(str((current or {}).get("quality_flag") or "unscored"), 3)
        new_rank = _SOURCE_QUALITY_PRIORITY.get(flag, 3)
        if current is None or (new_rank, -normalized_score) < (current_rank, -float((current or {}).get("quality_score") or 0.0)):
            index[note_id] = {
                "quality": quality,
                "quality_flag": flag,
                "quality_score": normalized_score,
            }
    service._source_quality_index = index
    return index


def source_quality_for_note(service, note_id: str) -> dict[str, Any]:
    return dict(service._get_source_quality_index().get(str(note_id or ""), {}))


def candidate_from_note(service, note_id: str) -> dict[str, Any] | None:
    note = service.sqlite_db.get_note(note_id)
    if not note:
        return None
    metadata = service.materializer._get_note_metadata(note)
    record_id = str(metadata.get("record_id") or "")
    job_id = str(metadata.get("crawl_job_id") or "")
    record = service._get_record(job_id=job_id, record_id=record_id) if (job_id and record_id) else None
    normalized, _ = service.materializer._document_view(record or {})
    relations, concept_ids, relation_ids = service.materializer._extract_note_concepts(note_id)
    _ = relation_ids
    entity_names = [service.materializer._entity_name(entity_id) for entity_id in concept_ids]
    relation_lines: list[str] = []
    for relation in relations:
        source_name = service.materializer._ref_name(
            str(relation.get("source_type", "")),
            str(relation.get("source_id", relation.get("source_entity_id", ""))),
        )
        target_name = service.materializer._ref_name(
            str(relation.get("target_type", "")),
            str(relation.get("target_id", relation.get("target_entity_id", ""))),
        )
        predicate = str(relation.get("predicate_id", relation.get("relation", "related_to")))
        relation_lines.append(f"{source_name} -[{predicate}]-> {target_name}")
    content_text = str(normalized.get("content_text") or note.get("content") or "")
    excerpts = service.materializer._select_key_excerpts(
        title=str(note.get("title") or ""),
        content_text=content_text,
        entity_names=entity_names,
        relation_lines=relation_lines,
        max_count=2,
    )
    quality_meta = service._source_quality_for_note(note_id)
    return {
        "note_id": note_id,
        "source_title": str(note.get("title") or note_id),
        "source_url": str(metadata.get("canonical_url") or normalized.get("canonical_url") or ""),
        "domain": str(normalized.get("domain") or ""),
        "summary_text": content_text[:1800],
        "relation_lines": relation_lines[:8],
        "key_excerpts_en": excerpts[:2],
        "quality": dict(quality_meta.get("quality") or {}),
        "quality_flag": str(quality_meta.get("quality_flag") or "unscored"),
        "quality_score": float(quality_meta.get("quality_score") or 0.0),
    }


def build_concept_pack_data(service, item: dict[str, Any]) -> dict[str, Any]:
    payload = dict(item.get("payload_json") or {})
    entity_id = str(item.get("entity_id") or payload.get("entity_id") or "")
    canonical_name = str(payload.get("title") or item.get("title_en") or entity_id)
    aliases = service.sqlite_db.get_entity_aliases(entity_id) if entity_id else []
    relation_lines: list[str] = []
    related_concepts: list[str] = []
    support_note_ids: list[str] = []
    for relation in service.sqlite_db.get_relations("concept", entity_id):
        predicate = str(relation.get("predicate_id", relation.get("relation", "related_to")))
        source_type = str(relation.get("source_type", "")).strip()
        target_type = str(relation.get("target_type", "")).strip()
        source_id = str(relation.get("source_id", relation.get("source_entity_id", "")))
        target_id = str(relation.get("target_id", relation.get("target_entity_id", "")))
        source_name = service.materializer._ref_name(source_type, source_id)
        target_name = service.materializer._ref_name(target_type, target_id)
        relation_lines.append(f"{source_name} -[{predicate}]-> {target_name}")
        if source_type == "note":
            support_note_ids.append(source_id)
        if target_type == "note":
            support_note_ids.append(target_id)
        if source_type == "concept" and source_name != canonical_name:
            related_concepts.append(source_name)
        if target_type == "concept" and target_name != canonical_name:
            related_concepts.append(target_name)
    support_note_ids = list(dict.fromkeys([token for token in support_note_ids if token]))[
        : int(service.config.get_nested("materialization", "enrichment", "concept_support_docs", default=8) or 8)
    ]
    compressed_support_docs: list[dict[str, Any]] = []
    if service.koreanizer is not None:
        candidates: list[dict[str, Any]] = []
        for note_id in support_note_ids:
            candidate = service._candidate_from_note(note_id)
            if not candidate:
                continue
            candidates.append(candidate)
        candidates.sort(
            key=lambda candidate: (
                _SOURCE_QUALITY_PRIORITY.get(str(candidate.get("quality_flag") or "unscored"), 3),
                -float(candidate.get("quality_score") or 0.0),
                str(candidate.get("source_title") or ""),
            )
        )
        for candidate in candidates:
            compressed, _warnings = service.koreanizer.compress_source_evidence_for_concept(
                source_title=str(candidate.get("source_title") or candidate.get("note_id") or ""),
                source_url=str(candidate.get("source_url") or ""),
                domain=str(candidate.get("domain") or ""),
                summary_text=str(candidate.get("summary_text") or ""),
                relation_lines=list(candidate.get("relation_lines") or []),
                key_excerpts_en=list(candidate.get("key_excerpts_en") or []),
            )
            compressed["source_quality_flag"] = str(candidate.get("quality_flag") or "unscored")
            compressed["source_quality"] = dict(candidate.get("quality") or {})
            compressed_support_docs.append(compressed)
    return {
        "note_item_id": int(item.get("id") or 0),
        "entity_id": entity_id,
        "canonical_name": canonical_name,
        "aliases": aliases[:12],
        "relation_lines": list(dict.fromkeys(relation_lines))[:12],
        "related_concepts": list(dict.fromkeys(related_concepts))[:12],
        "compressed_support_docs": compressed_support_docs[: int(service.config.get_nested("materialization", "enrichment", "concept_support_docs", default=8) or 8)],
        "existing_summary": str(payload.get("core_summary") or payload.get("summary_ko") or ""),
        "candidate_score": float(item.get("candidate_score") or payload.get("candidate_score") or 0.0),
        "translation_level": str(item.get("translation_level") or payload.get("translation_level") or "T1"),
    }


def source_model_fingerprint(service, allow_external: bool, llm_mode: str) -> tuple[str, str, str, str]:
    decision = decide_task_route(
        service.config,
        task_type="materialization_source_enrichment",
        allow_external=allow_external,
        force_route=llm_mode if llm_mode in {"fallback-only", "local", "mini", "strong", "auto"} else None,
        query="source enrichment",
        context="",
        source_count=1,
    )
    fingerprint = f"{ENRICHMENT_VERSION}:{decision.route}:{decision.provider}:{decision.model}"
    return decision.route, decision.provider, decision.model, fingerprint


def concept_model_fingerprint(service, allow_external: bool, llm_mode: str) -> tuple[str, str, str, str]:
    decision = decide_task_route(
        service.config,
        task_type="materialization_concept_enrichment",
        allow_external=allow_external,
        force_route=llm_mode if llm_mode in {"fallback-only", "local", "mini", "strong", "auto"} else None,
        query="concept enrichment",
        context="",
        source_count=6,
    )
    fingerprint = f"{ENRICHMENT_VERSION}:{decision.route}:{decision.provider}:{decision.model}"
    return decision.route, decision.provider, decision.model, fingerprint


def review_guidance(service, item_type: str, payload: dict[str, Any]) -> tuple[list[str], list[str], int, list[str], list[str], dict[str, dict[str, Any]]]:
    review = KoNoteReview.from_payload(payload)
    remediation = review.remediation
    quality = KoNoteQuality.from_payload(payload).to_payload()
    target_sections = [
        str(item).strip()
        for item in (
            remediation.target_sections
            or build_note_remediation_targets(item_type=item_type, quality=quality, payload=payload)
            or []
        )
        if str(item).strip()
    ]
    preserve_sections = remediation_preserve_sections(item_type=item_type, target_sections=target_sections)
    field_diagnostics_map = service._field_diagnostics(
        item_type=item_type,
        payload=payload,
        target_sections=target_sections,
    )
    return (
        list(review.reasons)[:6],
        list(review.patch_hints)[:6],
        int(remediation.attempt_count or 0) + 1,
        target_sections,
        preserve_sections,
        field_diagnostics_map,
    )


def normalized_quality_score(quality: dict[str, Any] | None) -> float:
    payload = KoNoteQuality.from_payload(quality)
    score = float(payload.score or 0.0)
    max_score = max(1.0, float(payload.max_score or 1.0))
    return score / max_score


def line_field_values(item_type: str, payload: dict[str, Any], section: str) -> list[str]:
    if str(item_type or "") == "source":
        mapping = {
            "top_claims": "top_claims",
            "contributions": "contributions",
            "methodology": "methodology",
            "results_or_findings": "results_or_findings",
            "insights": "insights",
            "limitations": "limitations",
            "core_concepts": "core_concepts",
            "key_excerpts_ko": "key_excerpts_ko",
            "related_concepts": "related_concepts",
            "sources": "representative_sources",
            "representative_sources": "representative_sources",
        }
        field_name = mapping.get(str(section or "").strip(), "")
        if not field_name:
            return []
        values = payload.get(field_name)
        if field_name == "results_or_findings" and not values:
            values = payload.get("key_results")
        return [str(item) for item in (values or []) if str(item).strip()]
    mapping = {
        "why_it_matters": "why_it_matters",
        "relation_lines": "relation_lines",
        "claim_lines": "claim_lines",
        "support_lines": "support_lines",
        "related_sources": "related_sources",
        "related_concepts": "related_concepts",
        "key_excerpts_ko": "key_excerpts_ko",
    }
    field_name = mapping.get(str(section or "").strip(), "")
    if not field_name:
        return []
    return [str(item) for item in (payload.get(field_name) or []) if str(item).strip()]


def field_diagnostics(
    *,
    item_type: str,
    payload: dict[str, Any],
    target_sections: list[str],
    line_field_values_fn: Callable[[str, dict[str, Any], str], list[str]],
) -> dict[str, dict[str, Any]]:
    review = KoNoteReview.from_payload(payload)
    remediation = review.remediation
    quality = KoNoteQuality.from_payload(payload)
    quality_diagnostics = dict((quality.checks or {}).get("line_diagnostics") or {})
    existing = dict(remediation.field_diagnostics or {})
    field_diagnostics_map: dict[str, dict[str, Any]] = {}
    for section in target_sections:
        key = str(section or "").strip()
        if not key:
            continue
        diagnostic = dict(existing.get(key) or {})
        if not diagnostic:
            diagnostic = dict(quality_diagnostics.get(key) or {})
        if not diagnostic and key == "sources":
            diagnostic = dict(quality_diagnostics.get("representative_sources") or {})
        if not diagnostic:
            values = line_field_values_fn(item_type, payload, key)
            if values:
                diagnostic = line_diagnostics_for_payload_field(
                    item_type=item_type,
                    field_name=key,
                    values=values,
                )
        if diagnostic:
            field_diagnostics_map[key] = diagnostic
    return field_diagnostics_map


def merge_scalar_field(
    merged: dict[str, Any],
    *,
    field_name: str,
    generated_value: Any,
    section_name: str,
    missing_sections: set[str],
    patched_sections: list[str],
) -> tuple[int, int]:
    existing_value = str(merged.get(field_name) or "").strip()
    generated_token = str(generated_value or "").strip()
    if should_replace_scalar(existing_value, generated_token, missing=section_name in missing_sections):
        merged[field_name] = generated_token
        if field_name == "summary_ko" and not str(merged.get("core_summary") or "").strip():
            merged["core_summary"] = generated_token
        if field_name == "core_summary" and not str(merged.get("summary_ko") or "").strip():
            merged["summary_ko"] = generated_token
        patched_sections.append(section_name)
        return 1, 0
    return 0, int(bool(existing_value))


def merge_list_field(
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
    existing_values = [str(item) for item in (merged.get(field_name) or []) if str(item).strip()]
    merged_values, patched_count, preserved_count = merge_targeted_lines(
        existing_values,
        generated_values,
        item_type=item_type,
        weak_indexes=list(diagnostics.get("weak_line_indexes") or []),
        min_count=int(min_count or remediation_line_minimum(item_type=item_type, field_name=section_name)),
        max_count=max(len(existing_values), len(generated_values), int(min_count or 1)),
    )
    if merged_values != existing_values:
        merged[field_name] = merged_values
        patched_sections.append(section_name)
    for secondary in secondary_fields or []:
        merged[secondary] = list(merged_values)
    return patched_count, preserved_count


def select_existing_top_items(service, *, item_type: str, limit: int) -> list[dict[str, Any]]:
    items = service.sqlite_db.list_existing_ko_note_items(item_type=item_type, statuses=("applied",), limit=5000)
    feature_index: dict[str, dict[str, Any]] = {}
    if item_type == "concept":
        for snapshot in service.sqlite_db.list_feature_snapshots(feature_kind="concept", limit=5000):
            entity_id = str(snapshot.get("entity_id") or "").strip()
            if entity_id and entity_id not in feature_index:
                feature_index[entity_id] = snapshot
    ranked: list[tuple[float, dict[str, Any]]] = []
    now_ts = time.time()
    for item in items:
        payload = dict(item.get("payload_json") or {})
        final_path = str(item.get("final_path") or "")
        if not final_path or not Path(final_path).exists():
            continue
        freshness = 0.0
        try:
            age_days = max(0.0, (now_ts - Path(final_path).stat().st_mtime) / 86400.0)
            freshness = max(0.0, min(1.0, 1.0 - (age_days / 30.0)))
        except Exception:
            freshness = 0.5
        if item_type == "source":
            source_snapshot = service.sqlite_db.find_source_feature_snapshot(
                note_id=str(payload.get("note_id") or item.get("note_id") or ""),
                record_id=str(payload.get("record_id") or item.get("record_id") or ""),
                canonical_url=str(payload.get("source_url") or ""),
            ) or {}
            feature_importance = float(source_snapshot.get("importance_score") or 0.0)
            feature_freshness = float(source_snapshot.get("freshness_score") or 0.0)
            claim_density = float(source_snapshot.get("claim_density") or 0.0)
            score = (
                0.30 * max(float(item.get("candidate_score") or 0.0), feature_importance)
                + 0.20 * max(freshness, feature_freshness)
                + 0.20 * service.materializer._domain_trust(str(payload.get("source_url") or "").split("/")[2] if "://" in str(payload.get("source_url") or "") else "")
                + 0.15 * min(1.0, (len(payload.get("entity_lines") or []) + len(payload.get("relation_lines") or [])) / 12.0)
                + 0.15 * min(1.0, claim_density)
            )
        else:
            concept_snapshot = feature_index.get(str(item.get("entity_id") or payload.get("entity_id") or "").strip(), {})
            score = (
                0.30 * max(min(1.0, float(payload.get("support_doc_count") or 0.0) / 8.0), min(1.0, float(concept_snapshot.get("support_doc_count") or 0.0) / 8.0))
                + 0.20 * max(min(1.0, len(payload.get("relation_lines") or []) / 10.0), min(1.0, float(concept_snapshot.get("relation_degree") or 0.0)))
                + 0.20 * freshness
                + 0.15 * min(1.0, float(item.get("candidate_score") or 0.0))
                + 0.15 * min(1.0, float(concept_snapshot.get("importance_score") or 0.0))
            )
        ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in ranked[: max(1, int(limit))]]


__all__ = [
    "build_concept_pack_data",
    "build_source_pack_data",
    "candidate_from_note",
    "concept_model_fingerprint",
    "field_diagnostics",
    "frontmatter_title",
    "get_record",
    "get_source_quality_index",
    "line_field_values",
    "merge_list_field",
    "merge_scalar_field",
    "normalized_quality_score",
    "review_guidance",
    "select_existing_top_items",
    "source_model_fingerprint",
    "source_quality_for_note",
]
