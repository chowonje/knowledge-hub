"""Lane runners for the `khub add` facade."""

from __future__ import annotations

import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from uuid import uuid4

import click

from knowledge_hub.interfaces.cli.commands.crawl_support import collect_to_obsidian_payload
from knowledge_hub.interfaces.cli.commands.paper_import_support import run_import_csv
from knowledge_hub.papers.source_text import extract_pdf_text_excerpt
from knowledge_hub.web import WebIngestService
from knowledge_hub.web.crawl4ai_adapter import CrawlDocument

from .obsidian_stage import stage_obsidian_for_existing_source
from .result import redact_local_path, warning_list, wrap_result
from .route import AddRoute, _ARXIV_ID_RE, _local_pdf_path


def web_ingest_service(khub):
    if hasattr(khub, "web_ingest_service"):
        return khub.web_ingest_service()
    return WebIngestService(khub.config)


def runtime_add_dir(khub) -> Path:
    sqlite_path = str(getattr(khub.config, "sqlite_path", "") or "").strip()
    if sqlite_path:
        root = Path(sqlite_path).expanduser().resolve().parent
    else:
        root = Path.home() / ".khub"
    path = root / "runs" / "add"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_single_paper_csv(path: Path, *, source: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["bucket_ko", "title", "year", "theme_ko", "priority", "why_selected", "source_url"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "bucket_ko": "add",
                "title": title or source,
                "year": "",
                "theme_ko": "",
                "priority": "5",
                "why_selected": "khub add",
                "source_url": source,
            }
        )


def run_web_or_youtube_add(
    *,
    khub,
    route: AddRoute,
    source: str,
    topic: str,
    index: bool,
    extract_concepts: bool,
    allow_external: bool,
    to_obsidian: bool,
    timeout: int,
    delay: float,
    transcript_language: str | None,
    asr_model: str,
) -> dict:
    service = web_ingest_service(khub)
    is_youtube = route.kind == "youtube"
    engine = "youtube" if is_youtube else ("basic" if route.source_type == "pdf" else "auto")
    if to_obsidian:
        upstream = collect_to_obsidian_payload(
            khub=khub,
            urls=[source],
            topic=topic,
            source="youtube" if is_youtube else "web",
            profile="safe",
            source_policy="hybrid",
            engine="auto" if is_youtube else engine,
            timeout=max(5 if is_youtube else 1, int(timeout)),
            delay=max(0.0, float(delay)),
            index=bool(index),
            extract_concepts=bool(extract_concepts),
            allow_external=bool(allow_external),
            input_source="youtube" if is_youtube else "web",
            transcript_language=str(transcript_language or "").strip() or None,
            asr_model=str(asr_model or "tiny").strip() or "tiny",
            max_source_notes=1,
            max_concept_notes=0,
            llm_mode="fallback-only",
            local_timeout_sec=0,
            api_fallback_on_timeout=False,
            enrich=False,
            apply_notes=False,
            only_approved=False,
        )
    else:
        upstream = service.crawl_and_ingest(
            urls=[source],
            topic=topic,
            engine=engine,
            timeout=max(5 if is_youtube else 1, int(timeout)),
            delay=max(0.0, float(delay)),
            index=bool(index),
            extract_concepts=bool(extract_concepts),
            allow_external=bool(allow_external),
            writeback=False,
            input_source="youtube" if is_youtube else "web",
            transcript_language=str(transcript_language or "").strip() or None,
            asr_model=str(asr_model or "tiny").strip() or "tiny",
            index_autofix_mode="youtube_single_retry" if is_youtube else "none",
        )
    return wrap_result(
        khub=khub,
        route=route,
        source=source,
        topic=topic,
        index=index,
        upstream=upstream,
        to_obsidian=to_obsidian,
    )


def run_local_pdf_add(
    *,
    khub,
    route: AddRoute,
    source: str,
    topic: str,
    index: bool,
    extract_concepts: bool,
    allow_external: bool,
    to_obsidian: bool,
) -> dict:
    path = _local_pdf_path(source)
    if path is None:
        raise click.BadParameter("local PDF path is required")
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise click.BadParameter(f"local PDF not found: {resolved}")
    content = extract_pdf_text_excerpt(str(resolved), max_pages=12, max_chars=30_000)
    if not content.strip():
        upstream = {
            "schema": "knowledge-hub.crawl.ingest.result.v1",
            "runId": f"crawl_ingest_{uuid4().hex[:12]}",
            "status": "failed",
            "requested": 1,
            "crawled": 0,
            "stored": 0,
            "indexedChunks": 0,
            "failed": [{"url": str(resolved), "error": "pdf text extraction returned empty content"}],
            "engine": "local_pdf",
            "topic": topic,
            "warnings": ["local PDF extraction returned empty content"],
            "localPath": redact_local_path(str(resolved)),
        }
        return wrap_result(
            khub=khub,
            route=route,
            source=source,
            topic=topic,
            index=index,
            upstream=upstream,
            to_obsidian=to_obsidian,
        )

    source_url = resolved.as_uri()
    service = web_ingest_service(khub)
    doc = CrawlDocument(
        url=source_url,
        title=resolved.stem,
        content=content,
        markdown=content,
        raw_html="",
        source_metadata={
            "media_type": "pdf",
            "content_type": "application/pdf",
            "source_path": str(resolved),
        },
        engine="local_pdf",
        ok=True,
        error="",
    )
    upstream = service.ingest_documents(
        [doc],
        topic=topic,
        index=bool(index),
        extract_concepts=bool(extract_concepts),
        allow_external=bool(allow_external),
        writeback=False,
        run_id=f"crawl_ingest_{uuid4().hex[:12]}",
        index_autofix_mode="none",
    ).to_dict()
    upstream["engine"] = "local_pdf"
    upstream["localPath"] = redact_local_path(str(resolved))
    upstream["warnings"] = list(
        dict.fromkeys(
            [
                *list(upstream.get("warnings") or []),
                "local PDF ingest uses an excerpt only (max 12 pages / 30000 chars)",
            ]
        )
    )
    if to_obsidian:
        stage_payload = stage_obsidian_for_existing_source(
            khub=khub,
            source_url=source_url,
            source_label="pdf",
            topic=topic,
            indexed=bool(index and int(upstream.get("indexedChunks") or 0) > 0),
            allow_external=bool(allow_external),
        )
        upstream = {
            **stage_payload,
            "crawl": upstream,
            "warnings": warning_list(upstream, stage_payload),
        }
    return wrap_result(
        khub=khub,
        route=route,
        source=source,
        topic=topic,
        index=index,
        upstream=upstream,
        to_obsidian=to_obsidian,
    )


def run_paper_url_add(
    *,
    khub,
    route: AddRoute,
    source: str,
    topic: str,
    index: bool,
    build_memory: bool,
    to_obsidian: bool,
    paper_parser: str,
    quiet: bool = False,
) -> dict:
    run_id = uuid4().hex[:12]
    run_dir = runtime_add_dir(khub)
    csv_path = run_dir / f"add-paper-{run_id}.csv"
    manifest_path = run_dir / f"add-paper-{run_id}.manifest.json"
    source_for_import = f"https://arxiv.org/abs/{source}" if _ARXIV_ID_RE.match(str(source or "").strip()) else source
    write_single_paper_csv(csv_path, source=source_for_import, title=topic or source_for_import)
    steps = ["register", "download"]
    if index:
        steps.append("embed")
    if build_memory:
        steps.extend(["paper-memory", "document-memory"])
    kwargs = {
        "khub": khub,
        "csv_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "min_priority": 0,
        "limit": 1,
        "steps": steps,
        "fail_fast": False,
        "document_memory_parser": str(paper_parser or "raw"),
        "rebuild_memory": False,
    }
    try:
        if quiet:
            with redirect_stdout(io.StringIO()):
                upstream = run_import_csv(**kwargs)
        else:
            upstream = run_import_csv(**kwargs)
    finally:
        try:
            csv_path.unlink()
        except FileNotFoundError:
            pass
    if to_obsidian:
        upstream = {
            **upstream,
            "warnings": [
                *list(upstream.get("warnings") or []),
                "paper Obsidian staging is deferred in add v1; direct paper writeback was not executed",
            ],
        }
    upstream["csvRetained"] = False
    return wrap_result(
        khub=khub,
        route=route,
        source=source_for_import,
        topic=topic,
        index=index,
        upstream=upstream,
        to_obsidian=to_obsidian,
    )


def run_paper_query_add(
    *,
    ctx,
    khub,
    route: AddRoute,
    source: str,
    max_papers: int,
    index: bool,
    allow_external: bool,
    to_obsidian: bool,
) -> dict:
    from knowledge_hub.interfaces.cli.commands.discover_cmd import discover

    output = io.StringIO()
    with redirect_stdout(output):
        ctx.invoke(
            discover,
            topic=source,
            max_papers=max(1, int(max_papers)),
            year_start=None,
            min_citations=0,
            sort_by="relevance",
            translate=True,
            gen_summary=True,
            create_obsidian=False,
            judge_enabled=False,
            judge_threshold=0.62,
            judge_candidates=None,
            allow_external=bool(allow_external),
            auto_confirm=True,
            workers=4,
            do_index=bool(index),
            as_json=True,
        )
    raw = output.getvalue().strip()
    try:
        upstream = json.loads(raw) if raw else {}
    except Exception:
        upstream = {
            "schema": "knowledge-hub.paper.discover.result.v1",
            "status": "failed",
            "topic": source,
            "warnings": ["paper discovery did not return JSON"],
            "rawOutput": raw[-4000:],
        }
    if not upstream:
        upstream = {
            "schema": "knowledge-hub.paper.discover.result.v1",
            "status": "failed",
            "topic": source,
            "warnings": ["paper discovery returned no payload"],
        }
    if to_obsidian:
        upstream = {
            **upstream,
            "warnings": [
                *list(upstream.get("warnings") or []),
                "paper Obsidian staging is deferred in add v1; direct paper writeback was not executed",
            ],
        }
    return wrap_result(
        khub=khub,
        route=route,
        source=source,
        topic=source,
        index=index,
        upstream=upstream,
        to_obsidian=to_obsidian,
    )
