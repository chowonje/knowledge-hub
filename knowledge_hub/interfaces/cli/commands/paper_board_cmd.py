"""Paper board export command for Obsidian-facing visualization surfaces."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.papers.public_surface import build_public_board_export

console = Console()


@click.command("board-export")
@click.option("--field", default=None, help="분야 필터")
@click.option("--limit", default=50, show_default=True, help="내보낼 최대 논문 수")
@click.option("--top-concepts", default=4, show_default=True, help="논문별 최대 concept 수")
@click.option("--top-related", default=4, show_default=True, help="논문별 최대 related paper 수")
@click.option("--json/--no-json", "as_json", default=True, show_default=True, help="JSON payload 출력")
@click.pass_context
def paper_board_export(ctx, field, limit, top_concepts, top_related, as_json):
    """Obsidian board UI용 논문 읽기 전용 export"""
    payload = build_public_board_export(
        ctx.obj["khub"],
        field=(str(field).strip() or None) if field is not None else None,
        limit=max(1, int(limit)),
        concept_limit=max(1, int(top_concepts)),
        related_limit=max(1, int(top_related)),
    )
    annotate_schema_errors(payload, payload["schema"], strict=False)
    if as_json:
        console.print_json(data=payload)
        return

    table = Table(title=f"Paper Board Export ({payload['stats']['returnedPapers']} papers)")
    table.add_column("Paper ID", style="cyan", width=14)
    table.add_column("Title", max_width=40)
    table.add_column("Year", width=6)
    table.add_column("Field", max_width=16)
    table.add_column("Artifacts", max_width=24)
    table.add_column("Summary", max_width=42)
    for item in payload.get("papers", []):
        flags = item.get("artifactFlags") or {}
        badges = [
            "PDF" if flags.get("hasPdf") else "",
            "SUM" if flags.get("hasSummary") else "",
            "KO" if flags.get("hasTranslation") else "",
            "IDX" if flags.get("isIndexed") else "",
            "MEM" if flags.get("hasMemory") else "",
        ]
        table.add_row(
            str(item.get("paperId") or ""),
            str(item.get("title") or ""),
            str(item.get("year") or ""),
            str(item.get("field") or ""),
            " ".join(part for part in badges if part) or "-",
            str((item.get("summary") or {}).get("oneLine") or "-"),
        )
    console.print(table)


__all__ = ["paper_board_export"]
