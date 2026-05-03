"""khub claims - claim extraction and review commands."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from knowledge_hub.ai.ask_v2_verification import AskV2Verifier
from knowledge_hub.domain.ai_papers.claim_cards import ClaimCardAlignmentService
from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.knowledge.claim_normalization import (
    CLAIM_NORMALIZATION_VERSION,
    ClaimComparisonService,
    ClaimNormalizationService,
)
from knowledge_hub.knowledge.synthesis import ClaimSynthesisService
from knowledge_hub.learning.mapper import slugify_topic
from knowledge_hub.learning.resolver import EntityResolver
from knowledge_hub.learning.task_router import get_llm_for_task
from knowledge_hub.papers.claim_extractor import (
    estimate_evidence_quality,
    extract_claim_candidates,
    score_claim_with_breakdown,
)
from knowledge_hub.knowledge.features import topic_matches_text
from knowledge_hub.knowledge.quality_mode import (
    estimate_quality_mode_cost,
    resolve_quality_mode_route,
)
from knowledge_hub.web.claim_extractor import extract_web_claim_candidates
from knowledge_hub.web.ingest import make_web_note_id
from knowledge_hub.web.ontology_extractor import WebOntologyExtractor

console = Console()


def _db(ctx) -> SQLiteDatabase:
    khub = ctx.obj["khub"]
    if hasattr(khub, "sqlite_db"):
        return khub.sqlite_db()
    return SQLiteDatabase(khub.config.sqlite_path)


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]schema:[/bold] {payload.get('schema')}")
    console.print(f"[bold]status:[/bold] {payload.get('status')}")
    if payload.get("topic"):
        console.print(f"[bold]topic:[/bold] {payload.get('topic')}")
    if payload.get("jobId"):
        console.print(f"[bold]jobId:[/bold] {payload.get('jobId')}")
    if payload.get("counts"):
        for key, value in payload["counts"].items():
            console.print(f"- {key}: {value}")
    for warning in payload.get("warnings") or []:
        console.print(f"[yellow]- {warning}[/yellow]")


def _validate_cli_payload(config: Any, payload: dict[str, Any], schema_id: str) -> None:
    strict = bool(config.get_nested("validation", "schema", "strict", default=False))
    result = annotate_schema_errors(payload, schema_id, strict=strict)
    if not result.ok:
        problems = ", ".join(result.errors[:5]) or "unknown schema validation error"
        raise click.ClickException(f"schema validation failed for {schema_id}: {problems}")


def _require_selector(*, claim_ids: tuple[str, ...], paper_ids: tuple[str, ...], include_all: bool = False) -> None:
    selectors = int(bool(claim_ids)) + int(bool(paper_ids)) + int(bool(include_all))
    if selectors != 1:
        raise click.ClickException("하나의 대상만 지정해야 합니다: --claim-id | --paper-id | --all")


def _clean_tokens(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for value in values:
        token = str(value or "").strip()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        items.append(token)
    return items


def _paper_ids_from_file(path_str: str) -> list[str]:
    token = str(path_str or "").strip()
    if not token:
        return []
    path = Path(token).expanduser()
    if not path.exists():
        raise click.ClickException(f"paper id file not found: {path}")
    items: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        items.append(line.split(",", 1)[0].strip())
    return _clean_tokens(items)


def _resolve_claim_extract_papers(
    db: SQLiteDatabase,
    *,
    topic: str,
    limit: int,
    paper_ids: tuple[str, ...],
    paper_id_file: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    explicit_ids = _clean_tokens(list(paper_ids) + _paper_ids_from_file(paper_id_file))
    warnings: list[str] = []
    if explicit_ids:
        papers: list[dict[str, Any]] = []
        missing: list[str] = []
        for paper_id in explicit_ids:
            row = db.get_paper(paper_id)
            if row:
                papers.append(dict(row))
            else:
                missing.append(paper_id)
        if missing:
            warnings.append(f"missing paper ids: {', '.join(missing[:10])}")
        return papers[: max(1, int(limit))], warnings
    if not str(topic or "").strip():
        raise click.ClickException("topic 또는 --paper-id/--paper-id-file 중 하나는 필요합니다.")
    papers = db.search_papers(topic, limit=max(1, int(limit)) * 3)
    if not papers:
        papers = db.list_papers(limit=max(1, int(limit)) * 3)
    return list(papers), warnings


def _load_json_file(path_str: str) -> dict[str, Any]:
    path = Path(str(path_str or "")).expanduser()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _paper_summary_text(paper: dict[str, Any]) -> str:
    notes = str(paper.get("notes") or "").strip()
    if notes and len(notes) >= 40:
        return notes[:3500]
    for key in ("translated_path", "text_path"):
        raw_path = str(paper.get(key) or "").strip()
        if raw_path and Path(raw_path).exists():
            try:
                text = Path(raw_path).read_text(encoding="utf-8").strip()
            except Exception:
                text = ""
            if text:
                return text[:3500]
    return ""


@click.group("claims")
def claims_group():
    """주장(Claim) 추출/검토"""


@claims_group.command("extract-web")
@click.option("--job-id", required=True, help="crawl job ID")
@click.option("--topic", required=True, help="핵심 토픽")
@click.option("--limit", default=120, type=int, show_default=True)
@click.option("--allow-external/--no-allow-external", default=False, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def claims_extract_web(ctx, job_id, topic, limit, allow_external, as_json):
    """최신 web batch에서 claim만 추출"""
    khub = ctx.obj["khub"]
    db = _db(ctx)
    extractor = WebOntologyExtractor(db, khub.config)
    topic_slug = slugify_topic(topic)
    monthly_spend_usd = float(db.get_quality_mode_monthly_spend() or 0.0)

    note_rows: list[dict[str, Any]] = []
    scanned = 0
    for state in ("indexed", "normalized"):
        records = db.list_crawl_pipeline_records(job_id, state=state, limit=max(1, int(limit)) * 4)
        for record in records:
            scanned += 1
            normalized = _load_json_file(str(record.get("normalized_path") or ""))
            canonical_url = str(
                normalized.get("canonical_url")
                or normalized.get("url")
                or record.get("canonical_url")
                or record.get("source_url")
                or ""
            ).strip()
            if not canonical_url:
                continue
            note_id = make_web_note_id(canonical_url)
            note = db.get_note(note_id) or {}
            content = str(normalized.get("content_text") or note.get("content") or "").strip()
            title = str(normalized.get("title") or note.get("title") or canonical_url).strip()
            if not content:
                continue
            if not topic_matches_text(topic, title, content, normalized.get("source_name"), normalized.get("tags")):
                continue
            note_rows.append(
                {
                    "note_id": note_id,
                    "url": canonical_url,
                    "title": title,
                    "content": content,
                    "file_path": str(note.get("file_path") or ""),
                }
            )
            if len(note_rows) >= max(1, int(limit)):
                break
        if note_rows:
            break

    quality_decision = resolve_quality_mode_route(
        khub.config,
        item_kind="claim",
        requested_allow_external=bool(allow_external),
        requested_mode="auto",
        topic=topic_slug,
        monthly_spend_usd=monthly_spend_usd,
    )

    payload = extractor.extract_claims_from_notes(
        topic=topic_slug,
        note_rows=note_rows,
        run_id=f"web_claim_{job_id}",
        allow_external=quality_decision.allow_external,
    )
    payload_warnings = list(payload.get("warnings") or [])
    payload_warnings.extend(quality_decision.warnings)
    payload.update(
        {
            "schema": "knowledge-hub.claims.extract-web.result.v1",
            "status": "ok",
            "jobId": job_id,
            "qualityMode": {
                "topic": quality_decision.topic or "",
                "isCoreTopic": quality_decision.is_core_topic,
                "allowExternal": quality_decision.allow_external,
                "llmMode": quality_decision.llm_mode,
            },
            "counts": {
                "recordsScanned": scanned,
                "notesSelected": len(note_rows),
                "claimsAccepted": int(payload.get("claimsAccepted") or 0),
                "pendingCount": int(payload.get("pendingCount") or 0),
                "droppedClaims": int(payload.get("droppedClaims") or 0),
            },
            "warnings": list(dict.fromkeys(payload_warnings)),
        }
    )
    if quality_decision.allow_external and quality_decision.usage_key:
        db.record_quality_mode_usage(
            "claim",
            quality_decision.llm_mode,
            estimate_quality_mode_cost(
                khub.config,
                "claim",
                quality_decision.llm_mode,
            ),
            topic_slug=topic_slug,
        )
    _emit(payload, as_json)


@claims_group.command("extract-paper")
@click.option("--topic", default="", help="핵심 토픽 (paper subset 지정 시 optional)")
@click.option("--paper-id", "paper_ids", multiple=True, help="특정 paper id/arXiv id만 대상")
@click.option("--paper-id-file", default="", help="paper id 목록 파일 (한 줄당 1개)")
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--allow-external/--no-allow-external", default=False, show_default=True)
@click.option(
    "--llm-mode",
    default="auto",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def claims_extract_paper(ctx, topic, paper_ids, paper_id_file, limit, allow_external, llm_mode, as_json):
    """핵심 paper subset에서 claim 추출"""
    khub = ctx.obj["khub"]
    config = khub.config
    db = _db(ctx)
    resolver = EntityResolver(db)
    topic_slug = slugify_topic(topic) if str(topic or "").strip() else "paper_claim_subset"
    monthly_spend_usd = float(db.get_quality_mode_monthly_spend() or 0.0)

    papers, selection_warnings = _resolve_claim_extract_papers(
        db,
        topic=topic,
        limit=max(1, int(limit)),
        paper_ids=paper_ids,
        paper_id_file=paper_id_file,
    )

    quality_decision = resolve_quality_mode_route(
        config,
        item_kind="claim",
        requested_allow_external=bool(allow_external),
        requested_mode=str(llm_mode),
        topic=topic_slug,
        monthly_spend_usd=monthly_spend_usd,
    )

    llm, decision, warnings = get_llm_for_task(
        config,
        task_type="claim_extraction",
        allow_external=quality_decision.allow_external,
        query=topic or "paper claim subset",
        context=topic or "paper claim subset",
        source_count=1,
        force_route=quality_decision.llm_mode,
    )
    warnings = list(selection_warnings) + list(warnings or [])
    warnings.extend(quality_decision.warnings)
    accepted = 0
    pending = 0
    dropped = 0
    processed = 0
    selected = 0
    extraction_errors = 0

    for paper in papers:
        if processed >= max(1, int(limit)):
            break
        title = str(paper.get("title") or "").strip()
        summary_text = _paper_summary_text(paper)
        if not title or len(summary_text) < 40:
            continue
        if not _clean_tokens(paper_ids) and not paper_id_file and not topic_matches_text(topic, title, summary_text, paper.get("field"), paper.get("authors")):
            continue
        selected += 1
        arxiv_id = str(paper.get("arxiv_id") or "").strip()
        if not arxiv_id:
            continue
        paper_entity_id = f"paper:{arxiv_id}"
        db.upsert_ontology_entity(
            entity_id=paper_entity_id,
            entity_type="paper",
            canonical_name=title,
            properties={"arxiv_id": arxiv_id},
            source="claims_extract_paper",
        )
        if llm is None:
            warnings.append("claim route unavailable")
            break
        try:
            candidates = extract_claim_candidates(llm, title=title, text=summary_text, max_claims=8)
        except Exception as error:
            extraction_errors += 1
            warnings.append(f"claim extraction failed for {arxiv_id}: {error}")
            processed += 1
            continue
        for idx_claim, cand in enumerate(candidates):
            subj_res = resolver.resolve(cand.subject, entity_type="concept")
            subject_entity_id = subj_res.canonical_id if subj_res else paper_entity_id
            subject_conf = float(subj_res.resolve_confidence) if subj_res else 0.65

            object_entity_id = None
            object_literal = None
            object_conf = 0.6
            if cand.object_value:
                obj_res = resolver.resolve(cand.object_value, entity_type="concept")
                if obj_res:
                    object_entity_id = obj_res.canonical_id
                    object_conf = float(obj_res.resolve_confidence)
                else:
                    object_literal = cand.object_value

            entity_resolve_conf = (subject_conf + object_conf) / 2.0
            evidence_quality = estimate_evidence_quality(cand.evidence)
            claim_score, score_breakdown = score_claim_with_breakdown(
                cand.llm_confidence,
                entity_resolve_conf,
                evidence_quality,
                claim_text=cand.claim_text,
                evidence=cand.evidence,
                subject=cand.subject,
                predicate=cand.predicate,
                object_value=cand.object_value,
            )
            claim_id = (
                f"claim:{arxiv_id}:{idx_claim}:"
                f"{hashlib.sha1((cand.claim_text + cand.predicate).encode('utf-8')).hexdigest()[:10]}"
            )
            evidence_ptrs = [
                {
                    "type": "paper",
                    "path": str(paper.get("text_path") or paper.get("translated_path") or ""),
                    "snippet_hash": hashlib.sha1(cand.evidence.encode("utf-8")).hexdigest()[:16],
                    "score_breakdown": score_breakdown,
                }
            ]
            if claim_score >= 0.82:
                evidence_ptrs[0]["claim_decision"] = "accepted"
                db.upsert_claim(
                    claim_id=claim_id,
                    claim_text=cand.claim_text,
                    subject_entity_id=subject_entity_id,
                    predicate=cand.predicate,
                    object_entity_id=object_entity_id,
                    object_literal=object_literal,
                    confidence=claim_score,
                    evidence_ptrs=evidence_ptrs,
                    source="paper_claim_extractor",
                )
                accepted += 1
            elif claim_score >= 0.65:
                evidence_ptrs[0]["claim_decision"] = "pending"
                db.add_ontology_pending(
                    pending_type="claim",
                    run_id=f"paper_claim_{arxiv_id}",
                    topic_slug=topic_slug,
                    note_id=title,
                    source_url="",
                    source_entity_id=subject_entity_id,
                    predicate_id=cand.predicate,
                    target_entity_id=object_entity_id or "",
                    confidence=claim_score,
                    evidence_ptrs=evidence_ptrs,
                    reason={
                        "claim_id": claim_id,
                        "claim_text": cand.claim_text,
                        "subject_entity_id": subject_entity_id,
                        "predicate": cand.predicate,
                        "object_entity_id": object_entity_id or "",
                        "object_literal": object_literal or "",
                        "llm_confidence": cand.llm_confidence,
                        "entity_resolve_conf": entity_resolve_conf,
                        "evidence_quality": evidence_quality,
                        "entity_resolution_confidence": score_breakdown.get("entity_resolution_confidence", entity_resolve_conf),
                        "generic_claim_penalty": score_breakdown.get("generic_claim_penalty", 0.0),
                        "contradiction_hint": score_breakdown.get("contradiction_hint", 0.0),
                        "score_breakdown": score_breakdown,
                    },
                    status="pending",
                )
                pending += 1
            else:
                dropped += 1
        processed += 1

    payload = {
        "schema": "knowledge-hub.claims.extract-paper.result.v1",
        "status": "ok",
        "topic": topic_slug,
        "route": decision.to_dict() if hasattr(decision, "to_dict") else {},
        "counts": {
            "papersSelected": selected,
            "papersProcessed": processed,
            "claimsAccepted": accepted,
            "pendingCount": pending,
            "droppedClaims": dropped,
            "explicitPaperCount": len(_clean_tokens(list(paper_ids) + _paper_ids_from_file(paper_id_file))),
            "extractionErrors": extraction_errors,
        },
        "warnings": warnings,
    }
    if quality_decision.allow_external and quality_decision.usage_key:
        db.record_quality_mode_usage(
            "claim",
            quality_decision.llm_mode,
            estimate_quality_mode_cost(
                config,
                "claim",
                quality_decision.llm_mode,
            ),
            topic_slug=topic_slug,
        )
    _emit(payload, as_json)


def _latest_normalization_rows(
    db: SQLiteDatabase,
    *,
    claim_ids: list[str],
) -> dict[str, dict[str, Any]]:
    rows = db.list_claim_normalizations(claim_ids=claim_ids, limit=max(50, len(claim_ids) * 4))
    best: dict[str, dict[str, Any]] = {}
    ranking = {"normalized": 3, "partial": 2, "failed": 1}
    for row in rows:
        claim_id = str(row.get("claim_id") or "").strip()
        if not claim_id:
            continue
        existing = best.get(claim_id)
        if existing is None:
            best[claim_id] = dict(row)
            continue
        current_rank = ranking.get(str(row.get("status") or "").strip(), 0)
        existing_rank = ranking.get(str(existing.get("status") or "").strip(), 0)
        if current_rank > existing_rank or (
            current_rank == existing_rank and str(row.get("updated_at") or "") > str(existing.get("updated_at") or "")
        ):
            best[claim_id] = dict(row)
    return best


def _paper_claim_cards(db: SQLiteDatabase, *, paper_ids: list[str] | None = None) -> list[dict[str, Any]]:
    cards = list(db.list_claim_cards(source_kind="paper", limit=20000))
    if not paper_ids:
        return cards
    selected = set(_clean_tokens(paper_ids))
    return [item for item in cards if str(item.get("paper_id") or item.get("source_id") or "").strip() in selected]


def _paper_claim_anchors(db: SQLiteDatabase, claim_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for card in claim_cards:
        claim_id = str(card.get("claim_id") or "").strip()
        source_refs = db.list_claim_card_source_refs(claim_card_id=str(card.get("claim_card_id") or ""))
        source_card_id = str(source_refs[0].get("source_card_id") or "").strip() if source_refs else ""
        if not claim_id or not source_card_id:
            continue
        anchors.extend(
            dict(item)
            for item in db.list_evidence_anchors_v2(card_id=source_card_id, claim_ids=[claim_id])
        )
    return anchors


def _strict_report_payload(
    db: SQLiteDatabase,
    *,
    paper_ids: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    claim_cards = _paper_claim_cards(db, paper_ids=paper_ids)
    selected_ids = {str(item.get("claim_card_id") or "").strip() for item in claim_cards if str(item.get("claim_card_id") or "").strip()}
    if not claim_cards:
        return (
            {
                "schema": "knowledge-hub.claims.strict-report.result.v1",
                "status": "ok",
                "selectedPaperIds": [],
                "alignedGroups": [],
                "conflictCandidates": [],
                "versionSplitGroups": [],
                "verificationSummary": {},
                "counts": {
                    "selectedPapers": 0,
                    "selectedClaimCards": 0,
                    "strictReadyClaimCount": 0,
                    "conflictSurfacedCount": 0,
                    "versionSplitCount": 0,
                    "pendingReviewCount": 0,
                    "partialNormalizationCount": 0,
                    "definitiveComparisonAnswerCount": 0,
                },
                "metrics": {
                    "strictReadyClaimRate": 0.0,
                    "conflictSurfacedRate": 0.0,
                    "versionSplitDetectionRate": 0.0,
                    "definitiveComparisonAnswerRate": 0.0,
                },
            },
            [],
        )
    alignment_refs = [
        dict(item)
        for item in db.list_claim_card_alignment_refs(claim_card_ids=list(selected_ids))
        if str(item.get("aligned_claim_card_id") or "").strip() in selected_ids
    ]
    normalization_rows = _latest_normalization_rows(
        db,
        claim_ids=[str(item.get("claim_id") or "").strip() for item in claim_cards if str(item.get("claim_id") or "").strip()],
    )
    verifier = AskV2Verifier(db)
    verification_items, verification_summary = verifier.claim_verification(
        selected_claims=claim_cards,
        anchors=_paper_claim_anchors(db, claim_cards),
    )
    verification_by_card = {
        str(item.get("claimCardId") or "").strip(): dict(item)
        for item in verification_items
        if str(item.get("claimCardId") or "").strip()
    }
    grouped = ClaimCardAlignmentService.group_claim_cards(claim_cards)

    aligned_groups = [dict(item) for item in grouped if int(item.get("sourceDiversity") or 0) >= 2]
    conflict_candidates: list[dict[str, Any]] = []
    version_split_groups: list[dict[str, Any]] = []
    aligned_group_keys = {str(item.get("groupKey") or "") for item in aligned_groups if str(item.get("groupKey") or "")}
    strict_paper_ids: set[str] = set()

    card_by_id = {
        str(item.get("claim_card_id") or "").strip(): dict(item)
        for item in claim_cards
        if str(item.get("claim_card_id") or "").strip()
    }

    for group in aligned_groups:
        for claim_card_id in list(group.get("claimCardIds") or []):
            card = card_by_id.get(str(claim_card_id).strip())
            if card:
                strict_paper_ids.add(str(card.get("paper_id") or card.get("source_id") or "").strip())

    seen_conflict_keys: set[tuple[str, str, str]] = set()
    seen_version_keys: set[tuple[str, str]] = set()
    for ref in alignment_refs:
        left = card_by_id.get(str(ref.get("claim_card_id") or "").strip())
        right = card_by_id.get(str(ref.get("aligned_claim_card_id") or "").strip())
        if not left or not right:
            continue
        left_paper = str(left.get("paper_id") or left.get("source_id") or "").strip()
        right_paper = str(right.get("paper_id") or right.get("source_id") or "").strip()
        alignment_type = str(ref.get("alignment_type") or "").strip()
        if alignment_type == "family_related":
            key = tuple(sorted([left_paper, right_paper]))
            if key not in seen_version_keys:
                seen_version_keys.add(key)
                version_split_groups.append(
                    {
                        "paperIds": list(key),
                        "claimCardIds": [str(left.get("claim_card_id") or ""), str(right.get("claim_card_id") or "")],
                        "datasetFamily": str(ref.get("dataset_family") or ""),
                        "datasetVersionLeft": str(left.get("dataset_version") or ""),
                        "datasetVersionRight": str(right.get("dataset_version") or ""),
                        "reason": "dataset_family_version_split",
                    }
                )
            strict_paper_ids.update([left_paper, right_paper])
        elif alignment_type == "conflict":
            key = tuple(sorted([str(left.get("claim_card_id") or ""), str(right.get("claim_card_id") or ""), alignment_type]))
            if key not in seen_conflict_keys:
                seen_conflict_keys.add(key)
                conflict_candidates.append(
                    {
                        "paperIds": [left_paper, right_paper],
                        "claimCardIds": [str(left.get("claim_card_id") or ""), str(right.get("claim_card_id") or "")],
                        "reason": "alignment_conflict",
                    }
                )
            strict_paper_ids.update([left_paper, right_paper])

    strict_frames: list[tuple[str, str, str, str]] = []
    for group in aligned_groups:
        canonical = dict(group.get("canonicalFrame") or {})
        strict_frames.append(
            (
                str(canonical.get("task") or ""),
                str(canonical.get("dataset") or ""),
                str(canonical.get("metric") or ""),
                str(canonical.get("comparator") or ""),
            )
        )

    review_rows: list[dict[str, Any]] = []
    seen_review: set[tuple[str, str]] = set()

    def add_review_row(row: dict[str, Any]) -> None:
        key = (str(row.get("claimId") or ""), str(row.get("issueType") or ""))
        if key in seen_review:
            return
        seen_review.add(key)
        review_rows.append(row)

    for card in claim_cards:
        claim_card_id = str(card.get("claim_card_id") or "").strip()
        claim_id = str(card.get("claim_id") or "").strip()
        paper_id = str(card.get("paper_id") or card.get("source_id") or "").strip()
        verification = verification_by_card.get(claim_card_id, {})
        normalization = normalization_rows.get(claim_id, {})
        frame = (
            str(card.get("task_canonical") or ""),
            str(card.get("dataset_canonical") or ""),
            str(card.get("metric_canonical") or ""),
            str(card.get("comparator_canonical") or ""),
        )
        if any(frame) and frame in strict_frames:
            strict_paper_ids.add(paper_id)
        status = str(normalization.get("status") or "").strip()
        if status == "partial" and (frame in strict_frames or str(card.get("group_key") or "") in aligned_group_keys):
            strict_paper_ids.add(paper_id)
            add_review_row(
                {
                    "paperId": paper_id,
                    "claimId": claim_id,
                    "claimCardId": claim_card_id,
                    "issueType": "partial_normalization",
                    "stage1Status": str(verification.get("stage1Status") or ""),
                    "origin": str(card.get("origin") or ""),
                    "trustLevel": str(card.get("trust_level") or ""),
                    "task": str(card.get("task_canonical") or card.get("task") or ""),
                    "dataset": str(card.get("dataset_canonical") or card.get("dataset") or ""),
                    "metric": str(card.get("metric_canonical") or card.get("metric") or ""),
                    "comparator": str(card.get("comparator_canonical") or card.get("comparator") or ""),
                    "resultValueText": str(card.get("result_value_text") or ""),
                    "conditionText": str(card.get("condition_text") or ""),
                }
            )
        if str(verification.get("stage1Status") or "") in {"unsupported", "metric_mismatch", "numeric_mismatch", "version_mismatch", "direction_conflict"}:
            strict_paper_ids.add(paper_id)
            add_review_row(
                {
                    "paperId": paper_id,
                    "claimId": claim_id,
                    "claimCardId": claim_card_id,
                    "issueType": "stage1_mismatch",
                    "stage1Status": str(verification.get("stage1Status") or ""),
                    "origin": str(card.get("origin") or ""),
                    "trustLevel": str(card.get("trust_level") or ""),
                    "task": str(card.get("task_canonical") or card.get("task") or ""),
                    "dataset": str(card.get("dataset_canonical") or card.get("dataset") or ""),
                    "metric": str(card.get("metric_canonical") or card.get("metric") or ""),
                    "comparator": str(card.get("comparator_canonical") or card.get("comparator") or ""),
                    "resultValueText": str(card.get("result_value_text") or ""),
                    "conditionText": str(card.get("condition_text") or ""),
                }
            )

    pending_rows = db.list_ontology_pending(pending_type="claim", status="pending", limit=5000)
    for row in pending_rows:
        source_entity_id = str(row.get("source_entity_id") or "").strip()
        reason_json = row.get("reason_json") if isinstance(row.get("reason_json"), dict) else {}
        paper_id = ""
        if source_entity_id.startswith("paper:"):
            paper_id = source_entity_id.split(":", 1)[1]
        claim_id = str(reason_json.get("claim_id") or "").strip()
        if not paper_id and claim_id:
            parts = claim_id.split(":")
            if len(parts) >= 2:
                paper_id = parts[1]
        if paper_id and paper_id in strict_paper_ids:
            add_review_row(
                {
                    "paperId": paper_id,
                    "claimId": claim_id,
                    "claimCardId": "",
                    "issueType": "pending_claim",
                    "stage1Status": "",
                    "origin": "pending",
                    "trustLevel": "review",
                    "task": "",
                    "dataset": "",
                    "metric": "",
                    "comparator": "",
                    "resultValueText": "",
                    "conditionText": "",
                }
            )

    for item in conflict_candidates:
        for paper_id, claim_card_id in zip(item.get("paperIds") or [], item.get("claimCardIds") or [], strict=True):
            card = card_by_id.get(str(claim_card_id).strip(), {})
            add_review_row(
                {
                    "paperId": str(paper_id or ""),
                    "claimId": str(card.get("claim_id") or ""),
                    "claimCardId": str(claim_card_id or ""),
                    "issueType": "conflict_candidate",
                    "stage1Status": str(verification_by_card.get(str(claim_card_id or "").strip(), {}).get("stage1Status") or ""),
                    "origin": str(card.get("origin") or ""),
                    "trustLevel": str(card.get("trust_level") or ""),
                    "task": str(card.get("task_canonical") or card.get("task") or ""),
                    "dataset": str(card.get("dataset_canonical") or card.get("dataset") or ""),
                    "metric": str(card.get("metric_canonical") or card.get("metric") or ""),
                    "comparator": str(card.get("comparator_canonical") or card.get("comparator") or ""),
                    "resultValueText": str(card.get("result_value_text") or ""),
                    "conditionText": str(card.get("condition_text") or ""),
                }
            )
    for item in version_split_groups:
        for paper_id, claim_card_id in zip(item.get("paperIds") or [], item.get("claimCardIds") or [], strict=True):
            card = card_by_id.get(str(claim_card_id).strip(), {})
            add_review_row(
                {
                    "paperId": str(paper_id or ""),
                    "claimId": str(card.get("claim_id") or ""),
                    "claimCardId": str(claim_card_id or ""),
                    "issueType": "version_split",
                    "stage1Status": str(verification_by_card.get(str(claim_card_id or "").strip(), {}).get("stage1Status") or ""),
                    "origin": str(card.get("origin") or ""),
                    "trustLevel": str(card.get("trust_level") or ""),
                    "task": str(card.get("task_canonical") or card.get("task") or ""),
                    "dataset": str(card.get("dataset_canonical") or card.get("dataset") or ""),
                    "metric": str(card.get("metric_canonical") or card.get("metric") or ""),
                    "comparator": str(card.get("comparator_canonical") or card.get("comparator") or ""),
                    "resultValueText": str(card.get("result_value_text") or ""),
                    "conditionText": str(card.get("condition_text") or ""),
                }
            )

    strict_claim_cards = [item for item in claim_cards if str(item.get("paper_id") or item.get("source_id") or "").strip() in strict_paper_ids]
    strict_ready = 0
    for card in strict_claim_cards:
        claim_id = str(card.get("claim_id") or "").strip()
        verification = verification_by_card.get(str(card.get("claim_card_id") or "").strip(), {})
        normalization = normalization_rows.get(claim_id, {})
        if (
            str(card.get("origin") or "") == "extracted"
            and str(normalization.get("status") or "") == "normalized"
            and str(card.get("task_canonical") or "").strip()
            and str(card.get("dataset_canonical") or "").strip()
            and str(card.get("metric_canonical") or "").strip()
            and str(verification.get("stage1Status") or "supported") not in {"unsupported", "metric_mismatch", "numeric_mismatch", "version_mismatch"}
        ):
            strict_ready += 1

    definitive_denominator = max(1, len(aligned_groups) + len(conflict_candidates) + len(version_split_groups))
    definitive_numerator = sum(
        1 for group in aligned_groups
        if sum(
            1
            for claim_card_id in list(group.get("claimCardIds") or [])
            if any(
                card.get("claim_card_id") == str(claim_card_id).strip()
                and str(card.get("paper_id") or card.get("source_id") or "").strip() in strict_paper_ids
                and str(card.get("origin") or "") == "extracted"
                and str(normalization_rows.get(str(card.get("claim_id") or "").strip(), {}).get("status") or "") == "normalized"
                and str(verification_by_card.get(str(card.get("claim_card_id") or "").strip(), {}).get("stage1Status") or "supported") == "supported"
                for card in strict_claim_cards
            )
        ) >= 2
    )

    report = {
        "schema": "knowledge-hub.claims.strict-report.result.v1",
        "status": "ok",
        "selectedPaperIds": sorted(item for item in strict_paper_ids if item),
        "alignedGroups": aligned_groups,
        "conflictCandidates": conflict_candidates,
        "versionSplitGroups": version_split_groups,
        "verificationSummary": verification_summary,
        "counts": {
            "selectedPapers": len(strict_paper_ids),
            "selectedClaimCards": len(strict_claim_cards),
            "strictReadyClaimCount": strict_ready,
            "conflictSurfacedCount": len(conflict_candidates),
            "versionSplitCount": len(version_split_groups),
            "pendingReviewCount": sum(1 for item in review_rows if item["issueType"] == "pending_claim"),
            "partialNormalizationCount": sum(1 for item in review_rows if item["issueType"] == "partial_normalization"),
            "definitiveComparisonAnswerCount": definitive_numerator,
        },
        "metrics": {
            "strictReadyClaimRate": round(strict_ready / max(1, len(strict_claim_cards)), 4),
            "conflictSurfacedRate": round(len(conflict_candidates) / definitive_denominator, 4),
            "versionSplitDetectionRate": round(len(version_split_groups) / definitive_denominator, 4),
            "definitiveComparisonAnswerRate": round(definitive_numerator / definitive_denominator, 4),
        },
    }
    return report, review_rows


@claims_group.command("strict-report")
@click.option("--paper-id", "paper_ids", multiple=True, help="strict lane 대상으로 볼 paper id/arXiv id")
@click.option("--paper-id-file", default="", help="strict lane 대상 paper id 파일")
@click.option("--out-dir", default="", help="strict artifacts 출력 디렉터리")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def claims_strict_report(ctx, paper_ids, paper_id_file, out_dir, as_json):
    """strict comparison lane operator report 생성"""
    db = _db(ctx)
    selected_ids = _clean_tokens(list(paper_ids) + _paper_ids_from_file(paper_id_file))
    payload, review_rows = _strict_report_payload(
        db,
        paper_ids=selected_ids or None,
    )
    if out_dir:
        root = Path(str(out_dir)).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        report_path = root / "strict_lane_report.json"
        review_path = root / "strict_review_queue.csv"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        fieldnames = [
            "paperId",
            "claimId",
            "claimCardId",
            "issueType",
            "stage1Status",
            "origin",
            "trustLevel",
            "task",
            "dataset",
            "metric",
            "comparator",
            "resultValueText",
            "conditionText",
        ]
        with review_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in review_rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        payload["artifacts"] = {
            "strictLaneReport": str(report_path),
            "strictReviewQueue": str(review_path),
        }
    else:
        payload["artifacts"] = {}
    payload["reviewQueue"] = review_rows
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    click.echo(f"schema: {payload['schema']}")
    click.echo(f"selected papers: {payload['counts']['selectedPapers']}")
    if payload["artifacts"]:
        click.echo(f"strict lane report: {payload['artifacts']['strictLaneReport']}")
        click.echo(f"strict review queue: {payload['artifacts']['strictReviewQueue']}")


@claims_group.group("pending")
def claims_pending_group():
    """claim pending 큐"""


@claims_pending_group.command("list")
@click.option("--topic", default="", help="토픽 slug 또는 토픽명")
@click.option("--limit", default=50, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def claims_pending_list(ctx, topic, limit, as_json):
    db = _db(ctx)
    topic_slug = slugify_topic(topic) if str(topic or "").strip() else None
    items = db.list_ontology_pending(pending_type="claim", topic_slug=topic_slug, limit=max(1, int(limit)))
    payload = {
        "schema": "knowledge-hub.claims.pending.result.v1",
        "status": "ok",
        "topic": topic or "",
        "counts": {"pending": len(items)},
        "items": items,
        "warnings": [],
    }
    _emit(payload, as_json)


@claims_group.command("normalize")
@click.option("--claim-id", "claim_ids", multiple=True, help="target claim id")
@click.option("--paper-id", "paper_ids", multiple=True, help="target paper id / arXiv id")
@click.option("--all", "include_all", is_flag=True, default=False, help="normalize all stored claims")
@click.option("--limit", default=50, type=int, show_default=True)
@click.option("--allow-external/--no-allow-external", default=False, show_default=True)
@click.option(
    "--llm-mode",
    default="auto",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def claims_normalize(ctx, claim_ids, paper_ids, include_all, limit, allow_external, llm_mode, as_json):
    """비교 가능한 claim 정규화 레이어를 labs에 생성"""
    _require_selector(claim_ids=claim_ids, paper_ids=paper_ids, include_all=include_all)
    khub = ctx.obj["khub"]
    db = _db(ctx)
    service = ClaimNormalizationService(db, khub.config)
    result = service.normalize_claims(
        claim_ids=list(claim_ids),
        paper_ids=list(paper_ids),
        include_all=bool(include_all),
        limit=max(1, int(limit)),
        allow_external=bool(allow_external),
        llm_mode=str(llm_mode),
        persist=True,
    )
    items = list(result.get("items") or [])
    payload = {
        "schema": "knowledge-hub.claim-normalize.result.v1",
        "status": "ok",
        "version": CLAIM_NORMALIZATION_VERSION,
        "count": len(items),
        "items": items,
        "counts": {
            "normalized": sum(1 for item in items if str(item.get("status") or "") == "normalized"),
            "partial": sum(1 for item in items if str(item.get("status") or "") == "partial"),
            "failed": sum(1 for item in items if str(item.get("status") or "") == "failed"),
        },
        "warnings": list(
            dict.fromkeys(
                [
                    warning
                    for item in items
                    for warning in list(item.get("warnings") or [])
                ]
            )
        ),
    }
    _validate_cli_payload(khub.config, payload, payload["schema"])
    _emit(payload, as_json)


@claims_group.command("compare")
@click.option("--claim-id", "claim_ids", multiple=True, help="compare specific claim ids")
@click.option("--paper-id", "paper_ids", multiple=True, help="compare claims from specific paper ids")
@click.option("--task", default="", help="normalized task filter")
@click.option("--dataset", default="", help="normalized dataset filter")
@click.option("--metric", default="", help="normalized metric filter")
@click.option("--limit", default=200, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def claims_compare(ctx, claim_ids, paper_ids, task, dataset, metric, limit, as_json):
    """정규화된 claim들을 aligned/conflicting/incomparable로 비교"""
    _require_selector(claim_ids=claim_ids, paper_ids=paper_ids, include_all=False)
    khub = ctx.obj["khub"]
    db = _db(ctx)
    service = ClaimComparisonService(db, khub.config)
    result = service.compare(
        claim_ids=list(claim_ids),
        paper_ids=list(paper_ids),
        task=str(task or "").strip(),
        dataset=str(dataset or "").strip(),
        metric=str(metric or "").strip(),
        limit=max(1, int(limit)),
    )
    payload = {
        "schema": "knowledge-hub.claim-compare.result.v1",
        "status": "ok",
        "version": CLAIM_NORMALIZATION_VERSION,
        "filters": {
            "task": str(task or "").strip(),
            "dataset": str(dataset or "").strip(),
            "metric": str(metric or "").strip(),
        },
        "aligned_groups": list(result.get("alignedGroups") or []),
        "conflict_candidates": list(result.get("conflictCandidates") or []),
        "incomparable_groups": list(result.get("incomparableGroups") or []),
        "skipped_claims": list(result.get("skippedClaims") or []),
        "counts": {
            "selected": int(result.get("selectedCount") or 0),
            "alignedGroups": len(result.get("alignedGroups") or []),
            "conflictCandidates": len(result.get("conflictCandidates") or []),
            "incomparableGroups": len(result.get("incomparableGroups") or []),
            "skippedClaims": len(result.get("skippedClaims") or []),
        },
        "warnings": [],
    }
    _validate_cli_payload(khub.config, payload, payload["schema"])
    _emit(payload, as_json)


@claims_group.command("synthesize")
@click.option("--claim-id", "claim_ids", multiple=True, help="synthesize specific claim ids")
@click.option("--paper-id", "paper_ids", multiple=True, help="synthesize claims from specific paper ids")
@click.option("--task", default="", help="normalized task filter")
@click.option("--dataset", default="", help="normalized dataset filter")
@click.option("--metric", default="", help="normalized metric filter")
@click.option("--limit", default=200, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def claims_synthesize(ctx, claim_ids, paper_ids, task, dataset, metric, limit, as_json):
    """정규화된 claim들 위에서 bounded synthesis report 생성"""
    _require_selector(claim_ids=claim_ids, paper_ids=paper_ids, include_all=False)
    khub = ctx.obj["khub"]
    db = _db(ctx)
    result = ClaimSynthesisService(db, khub.config).synthesize(
        claim_ids=list(claim_ids),
        paper_ids=list(paper_ids),
        task=str(task or "").strip(),
        dataset=str(dataset or "").strip(),
        metric=str(metric or "").strip(),
        limit=max(1, int(limit)),
    )
    compare_result = dict(result.get("compareResult") or {})
    payload = {
        "schema": "knowledge-hub.claim-synthesis.result.v1",
        "status": "ok",
        "version": result.get("version") or CLAIM_NORMALIZATION_VERSION,
        "filters": {
            "task": str(task or "").strip(),
            "dataset": str(dataset or "").strip(),
            "metric": str(metric or "").strip(),
        },
        "comparisonReport": list(result.get("comparisonReport") or []),
        "commonLimitationSummary": list(result.get("commonLimitationSummary") or []),
        "conflictExplanations": list(result.get("conflictExplanations") or []),
        "diagnostics": dict(result.get("diagnostics") or {}),
        "counts": {
            "reportItems": len(result.get("comparisonReport") or []),
            "limitationSummaries": len(result.get("commonLimitationSummary") or []),
            "conflictExplanations": len(result.get("conflictExplanations") or []),
            "alignedGroups": len(compare_result.get("alignedGroups") or []),
            "conflictCandidates": len(compare_result.get("conflictCandidates") or []),
            "incomparableGroups": len(compare_result.get("incomparableGroups") or []),
            "skippedClaims": len(compare_result.get("skippedClaims") or []),
        },
        "warnings": [],
    }
    _validate_cli_payload(khub.config, payload, payload["schema"])
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]claim synthesis[/bold] reports={payload['counts']['reportItems']} "
        f"limitations={payload['counts']['limitationSummaries']} "
        f"conflicts={payload['counts']['conflictExplanations']}"
    )
