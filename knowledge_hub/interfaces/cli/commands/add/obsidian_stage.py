"""Stage-only Obsidian helpers for `khub add`."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from knowledge_hub.infrastructure.persistence.stores.derivative_lifecycle import fallback_source_hash
from knowledge_hub.web.quality import canonicalize_url

from .result import lookup_web_summary, sqlite_db


def pipeline_storage_root(khub) -> str:
    getter = getattr(khub.config, "get_nested", None)
    if callable(getter):
        configured = str(getter("pipeline", "storage_root", default="") or "").strip()
        if configured:
            return str(Path(configured).expanduser())
    sqlite_path = str(getattr(khub.config, "sqlite_path", "") or "").strip()
    if sqlite_path:
        return str(Path(sqlite_path).expanduser().resolve().parent / "pipeline")
    return str(Path.home() / ".khub" / "pipeline")


def ko_note_materializer(config):
    from knowledge_hub.notes import KoNoteMaterializer

    return KoNoteMaterializer(config)


def stage_obsidian_for_existing_source(
    *,
    khub,
    source_url: str,
    source_label: str,
    topic: str,
    indexed: bool,
    allow_external: bool,
) -> dict[str, Any]:
    if not indexed:
        return {
            "schema": "knowledge-hub.crawl.collect.result.v1",
            "status": "partial",
            "requested": 1,
            "topic": topic,
            "source": source_label,
            "jobId": "",
            "runId": "",
            "applyRequested": False,
            "onlyApproved": False,
            "crawl": {},
            "materialize": {},
            "apply": {},
            "warnings": ["ko-note staging skipped: source was not indexed"],
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    db = sqlite_db(khub)
    if db is None:
        return {
            "schema": "knowledge-hub.crawl.collect.result.v1",
            "status": "partial",
            "requested": 1,
            "topic": topic,
            "source": source_label,
            "jobId": "",
            "runId": "",
            "applyRequested": False,
            "onlyApproved": False,
            "crawl": {},
            "materialize": {},
            "apply": {},
            "warnings": ["ko-note staging skipped: sqlite runtime unavailable"],
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    canonical_url = canonicalize_url(source_url) or str(source_url or "").strip()
    record_id = fallback_source_hash(canonical_url, source_label)[:24]
    job_id = f"crawl_job_{uuid4().hex[:12]}"
    run_id = f"crawl_pipeline_{uuid4().hex[:12]}"
    summary = lookup_web_summary(khub, canonical_url)
    content_hash = str(summary.get("contentHash") or "").strip()
    try:
        db.create_crawl_pipeline_job(
            job_id=job_id,
            run_id=run_id,
            profile="safe",
            source_policy="fixed",
            storage_root=pipeline_storage_root(khub),
            source=source_label,
            topic=topic,
            sources=[canonical_url],
            status="completed",
        )
        db.upsert_crawl_pipeline_record(
            job_id=job_id,
            record_id=record_id,
            source=source_label,
            source_url=canonical_url,
            canonical_url=canonical_url,
            domain=urlsplit(canonical_url).netloc.lower(),
            canonical_url_hash=fallback_source_hash(canonical_url),
            content_sha256=content_hash,
            state="indexed",
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )
        db.update_crawl_pipeline_job(
            job_id,
            status="completed",
            requested_count=1,
            processed_count=1,
            normalized_count=1,
            indexed_count=1,
            finished=True,
        )
        materializer = ko_note_materializer(khub.config)
        materialize_payload = materializer.generate_for_job(
            job_id=job_id,
            max_source_notes=1,
            max_concept_notes=0,
            allow_external=bool(allow_external),
            llm_mode="fallback-only",
            local_timeout_sec=None,
            api_fallback_on_timeout=False,
            enrich=False,
        )
    except Exception as error:
        materialize_payload = {
            "schema": "knowledge-hub.ko-note.generate.result.v1",
            "status": "failed",
            "runId": "",
            "crawlJobId": job_id,
            "sourceCandidates": 0,
            "sourceGenerated": 0,
            "conceptCandidates": 0,
            "conceptGenerated": 0,
            "blocked": 1,
            "warnings": [f"ko-note staging failed: {error}"],
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    return {
        "schema": "knowledge-hub.crawl.collect.result.v1",
        "status": str(materialize_payload.get("status") or "partial"),
        "requested": 1,
        "topic": topic,
        "source": source_label,
        "jobId": job_id,
        "runId": str(materialize_payload.get("runId") or ""),
        "applyRequested": False,
        "onlyApproved": False,
        "crawl": {
            "schema": "knowledge-hub.crawl.pipeline.run.result.v1",
            "status": "completed",
            "jobId": job_id,
            "runId": run_id,
            "requested": 1,
            "processed": 1,
            "normalized": 1,
            "indexed": 1 if indexed else 0,
            "warnings": [],
        },
        "materialize": materialize_payload,
        "apply": {},
        "warnings": list(materialize_payload.get("warnings") or []),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
