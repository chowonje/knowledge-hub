"""Support helpers for crawl CLI commands.

The canonical command registration stays in ``crawl_cmd.py`` while payload
assembly, watchlist orchestration, and reindex-worker plumbing live here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.application.ko_note_reports import build_ko_note_report
from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.curation import (
    build_continuous_latest_batch,
    build_reference_seed_batch,
    load_latest_batch_items,
)
from knowledge_hub.learning import LearningCoachService
from knowledge_hub.notes import KoNoteMaterializer
from knowledge_hub.web import WebIngestService

CANONICAL_CLI_MODULE = "knowledge_hub.interfaces.cli.main"


def sqlite_db(khub):
    if hasattr(khub, "sqlite_db"):
        return khub.sqlite_db()
    from knowledge_hub.infrastructure.persistence import SQLiteDatabase

    return SQLiteDatabase(khub.config.sqlite_path)


def web_ingest_service(khub, *, service_factory=None):
    if hasattr(khub, "web_ingest_service"):
        return khub.web_ingest_service()
    factory = service_factory or WebIngestService
    return factory(khub.config)


def learning_service(khub, *, service_factory=None):
    if hasattr(khub, "learning_service"):
        return khub.learning_service()
    factory = service_factory or LearningCoachService
    return factory(khub.config)


def validate_cli_payload(config, payload: dict[str, Any], schema_id: str) -> None:
    strict = bool(config.get_nested("validation", "schema", "strict", default=False))
    result = annotate_schema_errors(payload, schema_id, strict=strict)
    if strict and result.errors:
        payload["status"] = "blocked"
        raise click.ClickException("; ".join(result.errors[:5]))


def collect_urls(url: tuple[str, ...], url_file: str | None) -> list[str]:
    values = [item.strip() for item in url if str(item).strip()]
    if url_file:
        path = Path(url_file).expanduser().resolve()
        if not path.exists():
            raise click.BadParameter(f"url file not found: {path}", param_hint="--url-file")
        for line in path.read_text(encoding="utf-8").splitlines():
            candidate = line.strip()
            if candidate and not candidate.startswith("#"):
                values.append(candidate)
    unique: list[str] = []
    seen = set()
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def stage_status(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("status") or "").strip().lower()


def combine_stage_warnings(*payloads: dict[str, Any] | None) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for warning in payload.get("warnings") or []:
            token = str(warning or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            warnings.append(token)
    return warnings


def collapse_workflow_status(*payloads: dict[str, Any] | None) -> str:
    statuses = [stage_status(payload) for payload in payloads if stage_status(payload)]
    if not statuses:
        return "ok"
    if "blocked" in statuses:
        return "blocked"
    if any(status in {"error", "failed"} for status in statuses):
        return "error"
    if "partial" in statuses:
        return "partial"
    if all(status in {"ok", "completed"} for status in statuses):
        return "ok"
    if all(status == "skipped" for status in statuses):
        return "skipped"
    return statuses[-1]


def collect_to_obsidian_payload(
    *,
    khub,
    urls: list[str],
    topic: str,
    source: str,
    profile: str,
    source_policy: str,
    engine: str,
    timeout: int,
    delay: float,
    index: bool,
    extract_concepts: bool,
    allow_external: bool,
    input_source: str = "web",
    transcript_language: str | None = None,
    asr_model: str = "tiny",
    max_source_notes: int,
    max_concept_notes: int,
    llm_mode: str,
    local_timeout_sec: int,
    api_fallback_on_timeout: bool,
    enrich: bool,
    apply_notes: bool,
    only_approved: bool,
    web_ingest_factory=None,
    materializer_factory=None,
) -> dict[str, Any]:
    service = web_ingest_service(khub, service_factory=web_ingest_factory)
    crawl_payload = service.run_pipeline(
        urls=urls,
        topic=topic,
        source=source,
        profile=profile,
        source_policy=source_policy,
        limit=0,
        engine=engine,
        timeout=max(1, int(timeout)),
        delay=max(0.0, float(delay)),
        index=bool(index),
        extract_concepts=bool(extract_concepts),
        allow_external=bool(allow_external),
        input_source=str(input_source or "web").strip() or "web",
        transcript_language=str(transcript_language or "").strip() or None,
        asr_model=str(asr_model or "tiny").strip() or "tiny",
    )
    validate_cli_payload(khub.config, crawl_payload, "knowledge-hub.crawl.pipeline.run.result.v1")

    materialize_payload: dict[str, Any] = {}
    apply_payload: dict[str, Any] = {}
    warnings: list[str] = []

    crawl_job_id = str(crawl_payload.get("jobId") or "").strip()
    if crawl_job_id:
        materializer_cls = materializer_factory or KoNoteMaterializer
        materializer = materializer_cls(khub.config)
        materialize_payload = materializer.generate_for_job(
            job_id=crawl_job_id,
            max_source_notes=int(max_source_notes) if int(max_source_notes) > 0 else None,
            max_concept_notes=int(max_concept_notes) if int(max_concept_notes) > 0 else None,
            allow_external=bool(allow_external),
            llm_mode=str(llm_mode),
            local_timeout_sec=int(local_timeout_sec) if int(local_timeout_sec) > 0 else None,
            api_fallback_on_timeout=bool(api_fallback_on_timeout),
            enrich=bool(enrich),
        )
        validate_cli_payload(khub.config, materialize_payload, "knowledge-hub.ko-note.generate.result.v1")
        run_id = str(materialize_payload.get("runId") or "").strip()
        if apply_notes:
            if run_id:
                apply_payload = materializer.apply(
                    run_id=run_id,
                    item_type="all",
                    limit=0,
                    only_approved=bool(only_approved),
                )
                validate_cli_payload(khub.config, apply_payload, "knowledge-hub.ko-note.apply.result.v1")
            else:
                warnings.append("apply skipped: ko-note run not available")
    else:
        warnings.append("ko-note generation skipped: crawl job not available")

    warnings = combine_stage_warnings(crawl_payload, materialize_payload, apply_payload, {"warnings": warnings})
    return {
        "schema": "knowledge-hub.crawl.collect.result.v1",
        "status": collapse_workflow_status(crawl_payload, materialize_payload, apply_payload),
        "requested": len(urls),
        "topic": str(topic or "").strip(),
        "source": str(source or "").strip(),
        "jobId": crawl_job_id,
        "runId": str(materialize_payload.get("runId") or "").strip(),
        "applyRequested": bool(apply_notes),
        "onlyApproved": bool(only_approved),
        "crawl": crawl_payload,
        "materialize": materialize_payload,
        "apply": apply_payload,
        "warnings": warnings,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def sync_watchlist_payload(
    *,
    khub,
    build_payload: dict[str, Any],
    topic: str,
    source: str,
    profile: str,
    source_policy: str,
    engine: str,
    timeout: int,
    delay: float,
    index: bool,
    extract_concepts: bool,
    materialize: bool,
    apply_notes: bool,
    max_source_notes: int,
    max_concept_notes: int,
    allow_external: bool,
    llm_mode: str,
    web_ingest_factory=None,
    materializer_factory=None,
) -> dict[str, Any]:
    txt_path = str(build_payload.get("txtPath") or "").strip()
    source_items = build_payload.get("items") if isinstance(build_payload.get("items"), list) else []
    if not source_items:
        yaml_path = str(build_payload.get("yamlPath") or "").strip()
        if yaml_path:
            source_items = load_latest_batch_items(yaml_path)
    urls = collect_urls(tuple(), txt_path)
    if not urls:
        return {
            "status": "skipped",
            "build": build_payload,
            "crawl": {},
            "materialize": {},
            "apply": {},
            "warnings": ["no fresh urls discovered"],
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    service = web_ingest_service(khub, service_factory=web_ingest_factory)
    crawl_payload = service.run_pipeline(
        urls=urls,
        source_items=source_items,
        topic=str(topic).strip(),
        source=str(source).strip(),
        profile=str(profile),
        source_policy=str(source_policy),
        limit=0,
        engine=str(engine),
        timeout=max(1, int(timeout)),
        delay=max(0.0, float(delay)),
        index=bool(index),
        extract_concepts=bool(extract_concepts),
        allow_external=bool(allow_external),
    )

    materialize_payload: dict[str, Any] = {}
    apply_payload: dict[str, Any] = {}
    if materialize and str(crawl_payload.get("jobId") or "").strip():
        materializer_cls = materializer_factory or KoNoteMaterializer
        materializer = materializer_cls(khub.config)
        materialize_payload = materializer.generate_for_job(
            job_id=str(crawl_payload.get("jobId") or "").strip(),
            max_source_notes=int(max_source_notes) if int(max_source_notes) > 0 else None,
            max_concept_notes=int(max_concept_notes) if int(max_concept_notes) > 0 else None,
            allow_external=bool(allow_external),
            llm_mode=str(llm_mode),
        )
        validate_cli_payload(khub.config, materialize_payload, "knowledge-hub.ko-note.generate.result.v1")
        run_id = str(materialize_payload.get("runId") or "").strip()
        if apply_notes and run_id:
            apply_payload = materializer.apply(run_id=run_id, item_type="all", limit=0, only_approved=False)
            validate_cli_payload(khub.config, apply_payload, "knowledge-hub.ko-note.apply.result.v1")

    return {
        "status": "ok",
        "build": build_payload,
        "crawl": crawl_payload,
        "materialize": materialize_payload,
        "apply": apply_payload,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def load_normalized_records_for_job(
    config: Config,
    job_id: str,
    *,
    sqlite_db_factory=None,
) -> list[dict[str, Any]]:
    db = None
    try:
        if sqlite_db_factory is not None:
            db = sqlite_db_factory()
        else:
            from knowledge_hub.infrastructure.persistence import SQLiteDatabase

            db = SQLiteDatabase(config.sqlite_path)
        rows = db.list_crawl_pipeline_records(job_id, limit=100000)
        items: list[dict[str, Any]] = []
        for row in rows:
            path_str = str(row.get("normalized_path") or "").strip()
            if not path_str:
                continue
            path = Path(path_str)
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            items.append(payload)
        return items
    finally:
        if db is not None:
            db.close()


def metadata_audit_payload(*, job_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_stats: dict[str, dict[str, Any]] = {}
    flag_counter: Counter[str] = Counter()
    completeness_values: list[float] = []
    for row in rows:
        source_name = str(row.get("source_name") or "unknown")
        meta_quality = dict(row.get("metadata_quality") or {})
        completeness = float(meta_quality.get("completeness") or 0.0)
        completeness_values.append(completeness)
        flags = [str(flag) for flag in meta_quality.get("consistency_flags") or [] if str(flag)]
        flag_counter.update(flags)
        bucket = source_stats.setdefault(
            source_name,
            {
                "count": 0,
                "publishedCount": 0,
                "authorCount": 0,
                "tagCount": 0,
                "itemIdCount": 0,
                "avgCompleteness": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["publishedCount"] += 1 if row.get("published_at") else 0
        bucket["authorCount"] += 1 if row.get("author") else 0
        bucket["tagCount"] += 1 if row.get("tags") else 0
        bucket["itemIdCount"] += 1 if row.get("source_item_id") else 0
        bucket["avgCompleteness"] += completeness

    for bucket in source_stats.values():
        count = max(1, int(bucket["count"]))
        bucket["avgCompleteness"] = round(float(bucket["avgCompleteness"]) / count, 6)
        bucket["publishedRatio"] = round(float(bucket["publishedCount"]) / count, 6)
        bucket["authorRatio"] = round(float(bucket["authorCount"]) / count, 6)
        bucket["tagRatio"] = round(float(bucket["tagCount"]) / count, 6)
        bucket["itemIdRatio"] = round(float(bucket["itemIdCount"]) / count, 6)

    avg_completeness = round(sum(completeness_values) / max(1, len(completeness_values)), 6)
    return {
        "schema": "knowledge-hub.crawl.metadata.audit.result.v1",
        "status": "ok",
        "jobId": job_id,
        "records": len(rows),
        "avgCompleteness": avg_completeness,
        "topFlags": [{"flag": flag, "count": count} for flag, count in flag_counter.most_common(10)],
        "bySource": source_stats,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def build_reindex_worker_cmd(
    base_ctx,
    worker_index: int,
    worker_total: int,
    topic: str,
    limit: int,
    include_unrated: bool,
) -> list[str]:
    cmd = [sys.executable, "-m", CANONICAL_CLI_MODULE]
    config_path = getattr(base_ctx.obj["khub"], "_config_path", None)
    if config_path:
        cmd += ["--config", str(config_path)]

    cmd += [
        "crawl",
        "reindex-approved",
        "--shard-index",
        str(worker_index),
        "--shard-total",
        str(worker_total),
        "--json",
    ]
    if topic:
        cmd += ["--topic", topic]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    if include_unrated:
        cmd += ["--include-unrated"]

    return cmd


def parse_reindex_worker_output(
    worker_index: int,
    worker_total: int,
    topic: str,
    result: subprocess.CompletedProcess,
) -> tuple[dict[str, Any], str]:
    if result.returncode != 0:
        return (
            {
                "schema": "knowledge-hub.crawl.reindex-approved.result.v1",
                "status": "error",
                "runShard": worker_index,
                "shardIndex": worker_index,
                "shardTotal": worker_total,
                "topic": topic,
                "scanned": 0,
                "selected": 0,
                "indexedChunks": 0,
                "failed": [
                    {
                        "url": "*reindex-worker*",
                        "error": (result.stderr or result.stdout or "worker exited with non-zero code"),
                    }
                ],
                "ts": datetime.now(timezone.utc).isoformat(),
            },
            result.stderr or result.stdout or "",
        )

    try:
        payload = json.loads((result.stdout or "").strip())
    except Exception as error:
        return (
            {
                "schema": "knowledge-hub.crawl.reindex-approved.result.v1",
                "status": "error",
                "runShard": worker_index,
                "shardIndex": worker_index,
                "shardTotal": worker_total,
                "topic": topic,
                "scanned": 0,
                "selected": 0,
                "indexedChunks": 0,
                "failed": [{"url": "*reindex-worker*", "error": f"failed to parse worker json: {error}"}],
                "ts": datetime.now(timezone.utc).isoformat(),
            },
            result.stdout or result.stderr or "",
        )

    return payload, ""


__all__ = [
    "CANONICAL_CLI_MODULE",
    "build_continuous_latest_batch",
    "build_reference_seed_batch",
    "build_reindex_worker_cmd",
    "build_ko_note_report",
    "collapse_workflow_status",
    "collect_to_obsidian_payload",
    "collect_urls",
    "combine_stage_warnings",
    "learning_service",
    "load_normalized_records_for_job",
    "metadata_audit_payload",
    "parse_reindex_worker_output",
    "sqlite_db",
    "sync_watchlist_payload",
    "validate_cli_payload",
    "web_ingest_service",
]
