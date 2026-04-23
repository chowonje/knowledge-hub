from __future__ import annotations

from datetime import datetime, timezone
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


@click.group("belief")
def belief_group():
    """Belief ledger 운영"""


@belief_group.command("list")
@click.option("--status", default=None)
@click.option("--scope", default=None)
@click.option("--limit", default=100, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def belief_list(ctx, status, scope, limit, as_json):
    db = _db(ctx)
    items = db.list_beliefs(status=status, scope=scope, limit=limit)
    if as_json:
        console.print_json(data={"status": "ok", "count": len(items), "items": items})
        return
    table = Table(title=f"Beliefs ({len(items)})")
    table.add_column("belief_id", style="cyan")
    table.add_column("status", style="magenta")
    table.add_column("scope")
    table.add_column("confidence", justify="right")
    table.add_column("statement", max_width=56)
    for item in items:
        table.add_row(
            str(item.get("belief_id", "")),
            str(item.get("status", "")),
            str(item.get("scope", "")),
            f"{float(item.get('confidence', 0.0)):.2f}",
            str(item.get("statement", "")),
        )
    console.print(table)


@belief_group.command("show")
@click.argument("belief_id")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def belief_show(ctx, belief_id, as_json):
    db = _db(ctx)
    item = db.get_belief(belief_id)
    if not item:
        raise click.ClickException(f"belief not found: {belief_id}")
    if as_json:
        console.print_json(data=item)
        return
    console.print_json(data=item)


@belief_group.command("upsert")
@click.argument("statement")
@click.option("--belief-id", default=None)
@click.option("--scope", default="global", show_default=True)
@click.option("--status", default="proposed", show_default=True)
@click.option("--confidence", default=0.5, type=click.FloatRange(0.0, 1.0), show_default=True)
@click.option("--claim-id", "claim_ids", multiple=True)
@click.option("--support-id", "support_ids", multiple=True)
@click.option("--contradiction-id", "contradiction_ids", multiple=True)
@click.option("--last-validated-at", default=None)
@click.option("--review-due-at", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def belief_upsert(
    ctx,
    statement,
    belief_id,
    scope,
    status,
    confidence,
    claim_ids,
    support_ids,
    contradiction_ids,
    last_validated_at,
    review_due_at,
    as_json,
):
    db = _db(ctx)
    token = belief_id or f"belief_{uuid4().hex[:12]}"
    db.upsert_belief(
        belief_id=token,
        statement=statement,
        scope=scope,
        status=status,
        confidence=confidence,
        derived_from_claim_ids=list(claim_ids),
        support_ids=list(support_ids),
        contradiction_ids=list(contradiction_ids),
        last_validated_at=last_validated_at,
        review_due_at=review_due_at,
    )
    item = db.get_belief(token)
    if as_json:
        console.print_json(data={"status": "ok", "item": item})
        return
    console.print(f"[green]belief upserted[/green] id={token}")


@belief_group.command("review")
@click.argument("belief_id")
@click.option("--status", required=True, type=click.Choice(["proposed", "reviewed", "trusted", "stale", "rejected"]))
@click.option("--last-validated-at", default=None)
@click.option("--review-due-at", default=None)
@click.option("--successor-belief-id", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def belief_review(ctx, belief_id, status, last_validated_at, review_due_at, successor_belief_id, as_json):
    db = _db(ctx)
    item = db.review_belief(
        belief_id,
        status=status,
        last_validated_at=last_validated_at or datetime.now(timezone.utc).isoformat(),
        review_due_at=review_due_at,
        successor_belief_id=successor_belief_id,
    )
    if not item:
        raise click.ClickException(f"belief not found: {belief_id}")
    if as_json:
        console.print_json(data={"status": "ok", "item": item})
        return
    console.print(f"[green]belief reviewed[/green] old_id={belief_id} new_id={item['belief_id']} status={status}")
