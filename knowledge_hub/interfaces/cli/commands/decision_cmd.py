from __future__ import annotations

from uuid import uuid4

import click
from rich.console import Console
from rich.table import Table

from knowledge_hub.infrastructure.persistence import SQLiteDatabase

console = Console()


def _db(ctx) -> SQLiteDatabase:
    khub = ctx.obj["khub"]
    if hasattr(khub, "sqlite_db"):
        return khub.sqlite_db()
    return SQLiteDatabase(khub.config.sqlite_path)


@click.group("decision")
def decision_group():
    """Decision ledger 운영"""


@decision_group.command("create")
@click.argument("title")
@click.option("--decision-id", default=None)
@click.option("--summary", default="", show_default=True)
@click.option("--belief-id", "belief_ids", multiple=True)
@click.option("--chosen-option", default="", show_default=True)
@click.option("--status", default="open", show_default=True)
@click.option("--review-due-at", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def decision_create(ctx, title, decision_id, summary, belief_ids, chosen_option, status, review_due_at, as_json):
    db = _db(ctx)
    token = decision_id or f"decision_{uuid4().hex[:12]}"
    db.upsert_decision(
        decision_id=token,
        title=title,
        summary=summary,
        related_belief_ids=list(belief_ids),
        chosen_option=chosen_option,
        status=status,
        review_due_at=review_due_at,
    )
    item = db.get_decision(token)
    if as_json:
        console.print_json(data={"status": "ok", "item": item})
        return
    console.print(f"[green]decision created[/green] id={token}")


@decision_group.command("list")
@click.option("--status", default=None)
@click.option("--limit", default=100, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def decision_list(ctx, status, limit, as_json):
    db = _db(ctx)
    items = db.list_decisions(status=status, limit=limit)
    if as_json:
        console.print_json(data={"status": "ok", "count": len(items), "items": items})
        return
    table = Table(title=f"Decisions ({len(items)})")
    table.add_column("decision_id", style="cyan")
    table.add_column("status", style="magenta")
    table.add_column("title", max_width=42)
    table.add_column("beliefs", justify="right")
    table.add_column("review_due_at")
    for item in items:
        table.add_row(
            str(item.get("decision_id", "")),
            str(item.get("status", "")),
            str(item.get("title", "")),
            str(len(item.get("related_belief_ids", []))),
            str(item.get("review_due_at", "") or "-"),
        )
    console.print(table)


@decision_group.command("review")
@click.argument("decision_id")
@click.option("--status", required=True, type=click.Choice(["open", "committed", "reviewed", "closed"]))
@click.option("--review-due-at", default=None)
@click.option("--successor-decision-id", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def decision_review(ctx, decision_id, status, review_due_at, successor_decision_id, as_json):
    db = _db(ctx)
    item = db.review_decision(
        decision_id,
        status=status,
        review_due_at=review_due_at,
        successor_decision_id=successor_decision_id,
    )
    if not item:
        raise click.ClickException(f"decision not found: {decision_id}")
    if as_json:
        console.print_json(data={"status": "ok", "item": item})
        return
    console.print(f"[green]decision reviewed[/green] old_id={decision_id} new_id={item['decision_id']} status={status}")
