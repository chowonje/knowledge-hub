"""Simple source-add facade for the canonical CLI surface."""

from __future__ import annotations

import click
from rich.console import Console

from knowledge_hub.interfaces.cli.commands.add.lanes import (
    run_local_pdf_add,
    run_paper_query_add,
    run_paper_url_add,
    run_web_or_youtube_add,
)
from knowledge_hub.interfaces.cli.commands.add.route import _local_pdf_path, detect_add_route

console = Console()


def _print_text_summary(payload: dict) -> None:
    upstream = dict(payload.get("upstream") or {})
    console.print(
        f"[bold]add[/bold] status={payload.get('status')} type={payload.get('sourceType')} "
        f"route={payload.get('route')} stored={payload.get('stored')} indexed={payload.get('indexed')}"
    )
    if payload.get("sourceId") or payload.get("contentHash"):
        console.print(
            "  "
            f"sourceId={payload.get('sourceId') or '-'} "
            f"contentHash={(payload.get('contentHash') or '-')[:16]}"
        )
    stage = dict(payload.get("obsidianStage") or {})
    if stage.get("requested"):
        console.print(
            "  "
            f"obsidianStage={stage.get('status')} staged={stage.get('staged', 0)} "
            f"runId={stage.get('runId') or '-'} apply=false"
        )
    if payload.get("route") in {"crawl_ingest", "crawl_youtube_ingest"}:
        crawl = dict(upstream.get("crawl") or upstream)
        console.print(
            "  "
            f"requested={crawl.get('requested', 0)} crawled={crawl.get('crawled', crawl.get('processed', 0))} "
            f"stored={crawl.get('stored', crawl.get('normalized', 0))} "
            f"indexedChunks={crawl.get('indexedChunks', crawl.get('indexed', 0))} "
            f"runId={crawl.get('runId') or upstream.get('runId') or '-'}"
        )
        for warning in list(payload.get("warnings") or [])[:10]:
            console.print(f"  [yellow]- {warning}[/yellow]")
        return

    if payload.get("route") == "paper_import":
        counts = dict(upstream.get("counts") or {})
        console.print(
            "  "
            f"completed={counts.get('completed', 0)} failed={counts.get('failed', 0)} "
            f"skipped={counts.get('skipped', 0)} manifest={upstream.get('manifestPath') or '-'}"
        )
        for warning in list(payload.get("warnings") or [])[:10]:
            console.print(f"  [yellow]- {warning}[/yellow]")


@click.command("add")
@click.argument("source")
@click.option(
    "--type",
    "source_type",
    type=click.Choice(["auto", "web", "youtube", "paper", "pdf"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="source route. auto detects paper, YouTube, PDF, web URL, or paper query.",
)
@click.option("--topic", default="", help="topic label for web/youtube ingest or paper import.")
@click.option("-n", "--max-papers", default=3, show_default=True, help="paper query discovery limit.")
@click.option("--index/--no-index", default=True, show_default=True, help="embed/index added content.")
@click.option("--extract-concepts/--no-extract-concepts", default=True, show_default=True)
@click.option("--allow-external", is_flag=True, default=False, help="allow sanitized external LLM enrichment.")
@click.option("--to-obsidian/--no-to-obsidian", default=False, show_default=True, help="stage managed Obsidian notes without final apply.")
@click.option("--timeout", default=30, show_default=True, help="crawl/youtube timeout seconds.")
@click.option("--delay", default=0.5, show_default=True, help="delay between fetched URLs.")
@click.option("--transcript-language", default=None, help="preferred YouTube transcript language, e.g. ko or en.")
@click.option("--asr-model", default="tiny", show_default=True, help="local Whisper fallback model for YouTube.")
@click.option("--build-memory/--no-build-memory", default=False, show_default=True, help="also build heavy paper/document memory artifacts.")
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
    to_obsidian,
    timeout,
    delay,
    transcript_language,
    asr_model,
    build_memory,
    paper_parser,
    as_json,
):
    """Add a source with one command.

    \b
    Examples:
      khub add https://example.com/guide --topic rag
      khub add https://youtu.be/VIDEO --topic agents
      khub add https://arxiv.org/abs/2401.00001
      khub add ./local-report.pdf --type pdf
      khub add "retrieval augmented generation" --type paper -n 3
    """
    route = detect_add_route(source, source_type)
    khub = ctx.obj["khub"]

    if route.kind == "paper_query":
        payload = run_paper_query_add(
            ctx=ctx,
            khub=khub,
            route=route,
            source=source,
            max_papers=max_papers,
            index=bool(index),
            allow_external=bool(allow_external),
            to_obsidian=bool(to_obsidian),
        )
    elif route.kind == "paper_url":
        payload = run_paper_url_add(
            khub=khub,
            route=route,
            source=source,
            topic=topic,
            index=bool(index),
            build_memory=bool(build_memory),
            to_obsidian=bool(to_obsidian),
            paper_parser=paper_parser,
            quiet=bool(as_json),
        )
    elif route.kind == "pdf" and _local_pdf_path(source) is not None:
        payload = run_local_pdf_add(
            khub=khub,
            route=route,
            source=source,
            topic=topic,
            index=bool(index),
            extract_concepts=bool(extract_concepts),
            allow_external=bool(allow_external),
            to_obsidian=bool(to_obsidian),
        )
    else:
        payload = run_web_or_youtube_add(
            khub=khub,
            route=route,
            source=source,
            topic=topic,
            index=bool(index),
            extract_concepts=bool(extract_concepts),
            allow_external=bool(allow_external),
            to_obsidian=bool(to_obsidian),
            timeout=int(timeout),
            delay=float(delay),
            transcript_language=transcript_language,
            asr_model=asr_model,
        )

    if as_json:
        console.print_json(data=payload)
        return
    _print_text_summary(payload)
