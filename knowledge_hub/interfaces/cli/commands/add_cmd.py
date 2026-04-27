"""Simple source-add facade for the canonical CLI surface."""

from __future__ import annotations

import csv
import io
import re
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import click
from rich.console import Console

from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.interfaces.cli.commands.paper_import_support import run_import_csv
from knowledge_hub.web import WebIngestService
from knowledge_hub.web.youtube_extractor import is_youtube_url

console = Console()

ADD_RESULT_SCHEMA = "knowledge-hub.add.result.v1"
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")
_PAPER_HOST_SUFFIXES = (
    "arxiv.org",
    "openreview.net",
    "semanticscholar.org",
    "paperswithcode.com",
    "aclanthology.org",
    "doi.org",
    "proceedings.mlr.press",
    "neurips.cc",
    "openaccess.thecvf.com",
)


@dataclass(frozen=True)
class AddRoute:
    kind: str
    source_type: str
    route: str
    reason: str


def _is_url(source: str) -> bool:
    parsed = urlsplit(str(source or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_paper_url(source: str) -> bool:
    parsed = urlsplit(str(source or "").strip())
    host = (parsed.netloc or "").lower()
    if not host:
        return False
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in _PAPER_HOST_SUFFIXES)


def detect_add_route(source: str, source_type: str = "auto") -> AddRoute:
    """Select the existing subsystem that should own a source add request."""
    source_token = str(source or "").strip()
    requested = str(source_type or "auto").strip().lower()
    if not source_token:
        raise click.BadParameter("source is required")

    if requested == "youtube":
        return AddRoute("youtube", "youtube", "crawl_youtube_ingest", "explicit_type")
    if requested == "web":
        if not _is_url(source_token):
            raise click.BadParameter("--type web requires an http(s) URL")
        return AddRoute("web", "web", "crawl_ingest", "explicit_type")
    if requested == "paper":
        if _is_url(source_token) or _ARXIV_ID_RE.match(source_token):
            return AddRoute("paper_url", "paper", "paper_import", "explicit_type")
        return AddRoute("paper_query", "paper", "discover", "explicit_type")
    if requested != "auto":
        raise click.BadParameter(f"unsupported source type: {source_type}")

    if is_youtube_url(source_token):
        return AddRoute("youtube", "youtube", "crawl_youtube_ingest", "youtube_url")
    if _ARXIV_ID_RE.match(source_token) or _is_paper_url(source_token):
        return AddRoute("paper_url", "paper", "paper_import", "paper_identifier")
    if _is_url(source_token):
        return AddRoute("web", "web", "crawl_ingest", "web_url")
    return AddRoute("paper_query", "paper", "discover", "non_url_query")


def _web_ingest_service(khub):
    if hasattr(khub, "web_ingest_service"):
        return khub.web_ingest_service()
    return WebIngestService(khub.config)


def _runtime_add_dir(khub) -> Path:
    sqlite_path = str(getattr(khub.config, "sqlite_path", "") or "").strip()
    if sqlite_path:
        root = Path(sqlite_path).expanduser().resolve().parent
    else:
        root = Path.home() / ".khub"
    path = root / "runs" / "add"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_single_paper_csv(path: Path, *, source: str, title: str) -> None:
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


def _status_from_upstream(payload: dict) -> str:
    upstream_status = str(payload.get("status") or "").strip().lower()
    if upstream_status in {"ok", "completed"}:
        return "ok"
    if upstream_status:
        return upstream_status
    failed = payload.get("failed")
    if isinstance(failed, list) and failed:
        return "failed"
    return "ok"


def _wrap_result(*, route: AddRoute, source: str, topic: str, index: bool, upstream: dict) -> dict:
    payload = {
        "schema": ADD_RESULT_SCHEMA,
        "status": _status_from_upstream(upstream),
        "source": source,
        "sourceType": route.source_type,
        "route": route.route,
        "routeReason": route.reason,
        "topic": topic,
        "index": bool(index),
        "upstream": upstream,
        "nextCommands": [
            'khub ask "질문"',
            'khub search "검색어"',
            "khub status",
        ],
    }
    annotate_schema_errors(payload, ADD_RESULT_SCHEMA)
    return payload


def _run_web_or_youtube_add(
    *,
    khub,
    route: AddRoute,
    source: str,
    topic: str,
    index: bool,
    extract_concepts: bool,
    allow_external: bool,
    writeback: bool,
    timeout: int,
    delay: float,
    transcript_language: str | None,
    asr_model: str,
) -> dict:
    service = _web_ingest_service(khub)
    is_youtube = route.kind == "youtube"
    upstream = service.crawl_and_ingest(
        urls=[source],
        topic=topic,
        engine="youtube" if is_youtube else "auto",
        timeout=max(5 if is_youtube else 1, int(timeout)),
        delay=max(0.0, float(delay)),
        index=bool(index),
        extract_concepts=bool(extract_concepts),
        allow_external=bool(allow_external),
        writeback=bool(writeback),
        input_source="youtube" if is_youtube else "web",
        transcript_language=str(transcript_language or "").strip() or None,
        asr_model=str(asr_model or "tiny").strip() or "tiny",
        index_autofix_mode="youtube_single_retry" if is_youtube else "none",
    )
    return _wrap_result(route=route, source=source, topic=topic, index=index, upstream=upstream)


def _run_paper_url_add(
    *,
    khub,
    route: AddRoute,
    source: str,
    topic: str,
    index: bool,
    paper_parser: str,
    quiet: bool = False,
) -> dict:
    run_id = uuid4().hex[:12]
    run_dir = _runtime_add_dir(khub)
    csv_path = run_dir / f"add-paper-{run_id}.csv"
    manifest_path = run_dir / f"add-paper-{run_id}.manifest.json"
    source_for_import = f"https://arxiv.org/abs/{source}" if _ARXIV_ID_RE.match(str(source or "").strip()) else source
    _write_single_paper_csv(csv_path, source=source_for_import, title=topic or source_for_import)
    steps = ["register", "download"]
    if index:
        steps.extend(["embed", "paper-memory", "document-memory"])
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
    if quiet:
        with redirect_stdout(io.StringIO()):
            upstream = run_import_csv(**kwargs)
    else:
        upstream = run_import_csv(**kwargs)
    return _wrap_result(route=route, source=source_for_import, topic=topic, index=index, upstream=upstream)


def _print_text_summary(payload: dict) -> None:
    upstream = dict(payload.get("upstream") or {})
    console.print(
        f"[bold]add[/bold] status={payload.get('status')} type={payload.get('sourceType')} "
        f"route={payload.get('route')} index={payload.get('index')}"
    )
    if payload.get("route") in {"crawl_ingest", "crawl_youtube_ingest"}:
        console.print(
            "  "
            f"requested={upstream.get('requested', 0)} crawled={upstream.get('crawled', 0)} "
            f"stored={upstream.get('stored', 0)} indexedChunks={upstream.get('indexedChunks', 0)} "
            f"runId={upstream.get('runId') or '-'}"
        )
        for warning in list(upstream.get("warnings") or [])[:10]:
            console.print(f"  [yellow]- {warning}[/yellow]")
        return

    if payload.get("route") == "paper_import":
        counts = dict(upstream.get("counts") or {})
        console.print(
            "  "
            f"completed={counts.get('completed', 0)} failed={counts.get('failed', 0)} "
            f"skipped={counts.get('skipped', 0)} manifest={upstream.get('manifestPath') or '-'}"
        )
        for warning in list(upstream.get("warnings") or [])[:10]:
            console.print(f"  [yellow]- {warning}[/yellow]")


@click.command("add")
@click.argument("source")
@click.option(
    "--type",
    "source_type",
    type=click.Choice(["auto", "web", "youtube", "paper"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="source route. auto detects YouTube, paper URLs, web URLs, and paper queries.",
)
@click.option("--topic", default="", help="topic label for web/youtube ingest or paper import.")
@click.option("-n", "--max-papers", default=3, show_default=True, help="paper query discovery limit.")
@click.option("--index/--no-index", default=True, show_default=True, help="embed/index added content.")
@click.option("--extract-concepts/--no-extract-concepts", default=True, show_default=True)
@click.option("--allow-external", is_flag=True, default=False, help="allow sanitized external LLM enrichment.")
@click.option("--writeback/--no-writeback", default=False, show_default=True, help="update configured learning writeback files.")
@click.option("--timeout", default=30, show_default=True, help="crawl/youtube timeout seconds.")
@click.option("--delay", default=0.5, show_default=True, help="delay between fetched URLs.")
@click.option("--transcript-language", default=None, help="preferred YouTube transcript language, e.g. ko or en.")
@click.option("--asr-model", default="tiny", show_default=True, help="local Whisper fallback model for YouTube.")
@click.option(
    "--paper-parser",
    type=click.Choice(["raw", "pymupdf", "mineru", "opendataloader"]),
    default="raw",
    show_default=True,
    help="document-memory parser for paper URL imports.",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def add_cmd(
    ctx,
    source,
    source_type,
    topic,
    max_papers,
    index,
    extract_concepts,
    allow_external,
    writeback,
    timeout,
    delay,
    transcript_language,
    asr_model,
    paper_parser,
    as_json,
):
    """Add a source with one command.

    Examples:

      khub add https://example.com/guide --topic rag
      khub add https://youtu.be/VIDEO --topic agents
      khub add https://arxiv.org/abs/2401.00001
      khub add "retrieval augmented generation" --type paper -n 3
    """
    route = detect_add_route(source, source_type)
    khub = ctx.obj["khub"]

    if route.kind == "paper_query":
        from knowledge_hub.interfaces.cli.commands.discover_cmd import discover

        return ctx.invoke(
            discover,
            topic=source,
            max_papers=max(1, int(max_papers)),
            year_start=None,
            min_citations=0,
            sort_by="relevance",
            translate=True,
            gen_summary=True,
            create_obsidian=None,
            judge_enabled=False,
            judge_threshold=0.62,
            judge_candidates=None,
            allow_external=bool(allow_external),
            auto_confirm=True,
            workers=4,
            do_index=bool(index),
            as_json=bool(as_json),
        )

    if route.kind == "paper_url":
        payload = _run_paper_url_add(
            khub=khub,
            route=route,
            source=source,
            topic=topic,
            index=bool(index),
            paper_parser=paper_parser,
            quiet=bool(as_json),
        )
    else:
        payload = _run_web_or_youtube_add(
            khub=khub,
            route=route,
            source=source,
            topic=topic,
            index=bool(index),
            extract_concepts=bool(extract_concepts),
            allow_external=bool(allow_external),
            writeback=bool(writeback),
            timeout=int(timeout),
            delay=float(delay),
            transcript_language=transcript_language,
            asr_model=asr_model,
        )

    if as_json:
        console.print_json(data=payload)
        return
    _print_text_summary(payload)
