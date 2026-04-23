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


@click.group("outcome")
def outcome_group():
    """Outcome ledger 운영"""


@outcome_group.command("record")
@click.argument("decision_id")
@click.argument("summary")
@click.option("--outcome-id", default=None)
@click.option("--status", default="observed", show_default=True)
@click.option("--recorded-at", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def outcome_record(ctx, decision_id, summary, outcome_id, status, recorded_at, as_json):
    db = _db(ctx)
    token = outcome_id or f"outcome_{uuid4().hex[:12]}"
    db.record_outcome(
        outcome_id=token,
        decision_id=decision_id,
        status=status,
        summary=summary,
        recorded_at=recorded_at or datetime.now(timezone.utc).isoformat(),
    )
    item = db.get_outcome(token)
    if as_json:
        console.print_json(data={"status": "ok", "item": item})
        return
    console.print(f"[green]outcome recorded[/green] id={token}")


@outcome_group.command("show")
@click.argument("decision_id")
@click.option("--limit", default=50, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def outcome_show(ctx, decision_id, limit, as_json):
    db = _db(ctx)
    items = db.list_outcomes(decision_id=decision_id, limit=limit)
    if as_json:
        console.print_json(data={"status": "ok", "count": len(items), "items": items})
        return
    table = Table(title=f"Outcomes for {decision_id} ({len(items)})")
    table.add_column("outcome_id", style="cyan")
    table.add_column("status", style="magenta")
    table.add_column("recorded_at")
    table.add_column("summary", max_width=64)
    for item in items:
        table.add_row(
            str(item.get("outcome_id", "")),
            str(item.get("status", "")),
            str(item.get("recorded_at", "")),
            str(item.get("summary", "")),
        )
    console.print(table)
