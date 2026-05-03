"""Explicit paper memory card CLI surface."""

from __future__ import annotations

import click
from rich.console import Console

from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.papers.memory_payloads import card_payload
from knowledge_hub.papers.memory_retriever import PaperMemoryRetriever
from knowledge_hub.papers.memory_runtime import build_paper_memory_builder

console = Console()


def _validate_cli_payload(config, payload: dict, schema_id: str) -> None:
    strict = bool(config.get_nested("validation", "schema", "strict", default=False))
    result = annotate_schema_errors(payload, schema_id, strict=strict)
    if not result.ok:
        problems = ", ".join(result.errors[:5]) or "unknown schema validation error"
        raise click.ClickException(f"schema validation failed for {schema_id}: {problems}")


def _builder(khub):
    return build_paper_memory_builder(khub.sqlite_db(), config=khub.config)


def _retriever(khub):
    return PaperMemoryRetriever(khub.sqlite_db())


def _compact_item(item: dict) -> dict:
    return card_payload(item)


@click.group("paper-memory")
def paper_memory_group():
    """paper memory card build/show/search"""


@paper_memory_group.command("build")
@click.option("--paper-id", required=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def build_paper_memory(ctx, paper_id, as_json):
    """하나의 paper memory card를 빌드/업데이트"""
    khub = ctx.obj["khub"]
    item = _builder(khub).build_and_store(paper_id=str(paper_id).strip())
    payload = {
        "schema": "knowledge-hub.paper-memory.build.result.v1",
        "status": "ok",
        "mode": "single",
        "count": 1,
        "items": [_compact_item(item)],
        "warnings": [],
    }
    _validate_cli_payload(khub.config, payload, "knowledge-hub.paper-memory.build.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    card = payload["items"][0]
    console.print(
        f"[bold]paper-memory build[/bold] paper={card.get('paperId')} quality={card.get('qualityFlag')} title={card.get('title')}"
    )


@paper_memory_group.command("rebuild")
@click.option("--all", "rebuild_all", is_flag=True, help="모든 paper memory card를 재빌드")
@click.option("--limit", type=int, default=5000, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def rebuild_paper_memory(ctx, rebuild_all, limit, as_json):
    """전체 paper memory card 재빌드"""
    if not rebuild_all:
        raise click.ClickException("--all 플래그가 필요합니다.")
    khub = ctx.obj["khub"]
    items = _builder(khub).rebuild_all(limit=max(1, int(limit)))
    payload = {
        "schema": "knowledge-hub.paper-memory.build.result.v1",
        "status": "ok",
        "mode": "rebuild_all",
        "count": len(items),
        "items": [_compact_item(item) for item in items],
        "warnings": [],
    }
    _validate_cli_payload(khub.config, payload, "knowledge-hub.paper-memory.build.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]paper-memory rebuild[/bold] count={payload.get('count')}")


@paper_memory_group.command("show")
@click.option("--paper-id", required=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def show_paper_memory(ctx, paper_id, as_json):
    """paper memory card 조회"""
    khub = ctx.obj["khub"]
    item = _retriever(khub).get(str(paper_id).strip())
    payload = {
        "schema": "knowledge-hub.paper-memory.card.result.v1",
        "status": "ok" if item else "failed",
        "item": item or {},
        "warnings": [] if item else [f"paper memory card not found: {paper_id}"],
    }
    _validate_cli_payload(khub.config, payload, "knowledge-hub.paper-memory.card.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    if not item:
        raise click.ClickException(f"paper memory card not found: {paper_id}")
    console.print(f"[bold]{item.get('title')}[/bold]")
    console.print(item.get("paperCore") or "")
    if item.get("conceptLinks"):
        console.print(f"[dim]concepts:[/dim] {', '.join(item.get('conceptLinks') or [])}")


@paper_memory_group.command("search")
@click.option("--query", required=True)
@click.option("--limit", type=int, default=10, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def search_paper_memory(ctx, query, limit, as_json):
    """paper memory card 검색"""
    khub = ctx.obj["khub"]
    items = _retriever(khub).search(str(query).strip(), limit=max(1, int(limit)))
    payload = {
        "schema": "knowledge-hub.paper-memory.search.result.v1",
        "status": "ok",
        "query": str(query).strip(),
        "count": len(items),
        "items": items,
        "warnings": [],
    }
    _validate_cli_payload(khub.config, payload, "knowledge-hub.paper-memory.search.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]paper-memory search[/bold] query={query} count={len(items)}")
    for item in items[:10]:
        console.print(f"- {item.get('paperId')} {item.get('title')}")
        if item.get("paperCore"):
            console.print(f"[dim]  {item.get('paperCore')}[/dim]")
