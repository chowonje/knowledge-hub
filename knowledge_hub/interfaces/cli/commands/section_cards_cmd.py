"""labs section-cards - SectionCard build/preview surface."""

from __future__ import annotations

import json
from pathlib import Path

import click

from knowledge_hub.ai.section_card_materializer import PaperSectionCardMaterializer
from knowledge_hub.ai.section_cards import assess_section_source_quality, project_section_cards, rank_section_cards, section_coverage
from knowledge_hub.papers.card_v2_builder import PaperCardV2Builder


def _db(ctx):
    return ctx.obj["khub"].sqlite_db()


@click.group("section-cards")
def section_cards_group():
    """Build and inspect paper SectionCards."""


def _parse_paper_ids(*, paper_ids: tuple[str, ...], paper_id_file: str) -> list[str]:
    items = [str(item or "").strip() for item in paper_ids if str(item or "").strip()]
    path_token = str(paper_id_file or "").strip()
    if path_token:
        lines = Path(path_token).expanduser().read_text(encoding="utf-8").splitlines()
        items.extend(str(line or "").strip() for line in lines if str(line or "").strip())
    out: list[str] = []
    seen: set[str] = set()
    for token in items:
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(token)
    return out


@section_cards_group.command("build")
@click.option("--paper-id", "paper_ids", multiple=True, help="paper id / arXiv id (repeatable)")
@click.option("--paper-id-file", default="", help="newline-delimited paper id file")
@click.option("--allow-external", is_flag=True, default=False, help="allow external provider route for labs generation")
@click.option(
    "--llm-mode",
    default="auto",
    type=click.Choice(["auto", "local", "mini", "strong", "fallback-only"], case_sensitive=False),
    show_default=True,
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def section_cards_build(ctx, paper_ids, paper_id_file, allow_external, llm_mode, as_json):
    sqlite_db = _db(ctx)
    tokens = _parse_paper_ids(paper_ids=tuple(paper_ids), paper_id_file=paper_id_file)
    if not tokens:
        raise click.ClickException("at least one --paper-id or --paper-id-file is required")
    materializer = PaperSectionCardMaterializer(sqlite_db, ctx.obj["khub"].config)
    items = [
        materializer.build_and_store(
            paper_id=token,
            allow_external=bool(allow_external),
            llm_mode=str(llm_mode or "auto"),
        )
        for token in tokens
    ]
    payload = {
        "schema": "knowledge-hub.section-cards.batch-build.result.v1",
        "status": "ok",
        "count": len(items),
        "items": items,
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    click.echo(f"schema: {payload['schema']}")
    click.echo(f"count: {payload['count']}")
    for item in items:
        click.echo(
            f"- {item.get('paperId')} status={item.get('status')} count={item.get('count')} "
            f"coverage={dict(item.get('sectionCoverage') or {}).get('status', 'unknown')}"
        )


@section_cards_group.command("show")
@click.option("--paper-id", required=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def section_cards_show(ctx, paper_id, as_json):
    sqlite_db = _db(ctx)
    materializer = PaperSectionCardMaterializer(sqlite_db, ctx.obj["khub"].config)
    items = materializer.list_materialized(paper_id=str(paper_id).strip())
    payload = {
        "schema": "knowledge-hub.section-cards.show.result.v1",
        "status": "ok" if items else "missing",
        "paperId": str(paper_id).strip(),
        "count": len(items),
        "sectionCoverage": section_coverage(section_cards=items),
        "items": items,
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    click.echo(f"schema: {payload['schema']}")
    click.echo(f"paper_id: {payload['paperId']}")
    click.echo(f"count: {payload['count']}")


@section_cards_group.command("preview")
@click.option("--paper-id", required=True)
@click.option("--query", default="", show_default=True, help="optional ranking query")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def section_cards_preview(ctx, paper_id, query, as_json):
    sqlite_db = _db(ctx)
    builder = PaperCardV2Builder(sqlite_db)
    card = sqlite_db.get_paper_card_v2(str(paper_id).strip())
    if not card:
        card = builder.build_and_store(paper_id=str(paper_id).strip())
    units = list(sqlite_db.list_document_memory_units(f"paper:{str(paper_id).strip()}", limit=200) or [])
    projected = project_section_cards(source_kind="paper", source_card=card, units=units)
    ranked = rank_section_cards(query=str(query or ""), section_cards=projected, intent="paper_lookup" if not query else "implementation")
    coverage = section_coverage(section_cards=ranked)
    quality_gate = assess_section_source_quality(section_cards=ranked, coverage=coverage)
    payload = {
        "schema": "knowledge-hub.section-cards.preview.result.v1",
        "status": "ok",
        "paperId": str(paper_id).strip(),
        "count": len(ranked),
        "sectionCoverage": coverage,
        "qualityGate": quality_gate,
        "items": [
            {
                "sectionCardId": item.get("section_card_id"),
                "role": item.get("role"),
                "unitType": item.get("unit_type"),
                "sectionPath": item.get("section_path"),
                "title": item.get("title"),
                "confidence": item.get("confidence"),
                "appendixLike": item.get("appendix_like"),
                "selectionScore": item.get("selection_score"),
                "rankingReasons": list(item.get("ranking_reasons") or []),
                "contextualSummary": item.get("contextual_summary"),
                "sourceExcerpt": item.get("source_excerpt"),
            }
            for item in ranked
        ],
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    click.echo(f"schema: {payload['schema']}")
    click.echo(f"paper_id: {payload['paperId']}")
    click.echo(f"count: {payload['count']}")
