"""`khub labs judge` — record and list user judgments (V0 judgment loop).

Core-only sqlite access; never initializes the search runtime.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.quarantine import QUARANTINE_REASON_CODE, quarantined_paper_ids

console = Console()

_TARGET_TYPES = ["claim", "answer", "brief"]
_DECISIONS = ["accept", "thin", "reject", "unsure", "abstain", "archive"]
_CONFIDENCES = ["low", "medium", "high"]


def _db(ctx) -> SQLiteDatabase:
    khub = ctx.obj["khub"]
    if hasattr(khub, "sqlite_db"):
        return khub.sqlite_db()
    return SQLiteDatabase(khub.config.sqlite_path)


@click.group("judge")
def judge_group():
    """Record and list user judgments (V0 judgment loop)."""


@judge_group.command("record")
@click.option("--target-type", required=True, type=click.Choice(_TARGET_TYPES))
@click.option("--target-id", required=True)
@click.option("--decision", required=True, type=click.Choice(_DECISIONS))
@click.option("--reviewer", required=True)
@click.option("--reason", required=True)
@click.option("--confidence", default="medium", type=click.Choice(_CONFIDENCES), show_default=True)
@click.option("--reviewed-at", default=None, help="ISO-8601 timestamp (default: now UTC)")
@click.option("--target-text", default=None, help="Raw text is hashed (sha1[:16]) and never stored")
@click.option("--snippet-hash", "snippet_hashes", multiple=True)
@click.option("--evidence-span-id", "evidence_span_ids", multiple=True)
@click.option("--source-id", "source_ids", multiple=True)
@click.option("--query-hash", default="")
@click.option("--query-digest", default="")
@click.option("--rag-answer-log-id", default=None, type=int)
@click.option("--runtime-used", default="")
@click.option("--supersedes", default=None, help="Existing judgment_id this record supersedes")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def judge_record(
    ctx,
    target_type,
    target_id,
    decision,
    reviewer,
    reason,
    confidence,
    reviewed_at,
    target_text,
    snippet_hashes,
    evidence_span_ids,
    source_ids,
    query_hash,
    query_digest,
    rag_answer_log_id,
    runtime_used,
    supersedes,
    as_json,
):
    flagged = quarantined_paper_ids(source_ids)
    if flagged:
        raise click.ClickException(
            f"judgment refused — quarantined_source ({QUARANTINE_REASON_CODE}): {','.join(flagged)}"
        )
    db = _db(ctx)
    fields = {
        "target_type": target_type,
        "target_id": target_id,
        "decision": decision,
        "reviewer": reviewer,
        "reason": reason,
        "confidence": confidence,
        "reviewed_at": reviewed_at,
        "target_text": target_text,
        "snippet_hashes": list(snippet_hashes),
        "evidence_span_ids": list(evidence_span_ids),
        "source_ids": list(source_ids),
        "query_hash": query_hash,
        "query_digest": query_digest,
        "rag_answer_log_id": rag_answer_log_id,
        "runtime_used": runtime_used,
        "decision_source": "cli",
    }
    try:
        if supersedes:
            record = db.supersede_judgment(supersedes, **fields)
        else:
            record = db.record_judgment(**fields)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    if as_json:
        console.print_json(data={"status": "ok", "item": record})
        return
    console.print(
        f"[green]judgment recorded[/green] id={record['judgment_id']} "
        f"target={record['target_type']}:{record['target_id']} decision={record['decision']}"
    )


@judge_group.command("list")
@click.option("--limit", default=50, show_default=True)
@click.option("--target-type", default=None, type=click.Choice(_TARGET_TYPES))
@click.option("--decision", default=None, type=click.Choice(_DECISIONS))
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def judge_list(ctx, limit, target_type, decision, as_json):
    db = _db(ctx)
    items = db.list_judgments(limit=limit, target_type=target_type, decision=decision)
    if as_json:
        console.print_json(data={"status": "ok", "count": len(items), "items": items})
        return
    table = Table(title=f"Judgments ({len(items)})")
    table.add_column("judgment_id", style="cyan")
    table.add_column("target", style="magenta")
    table.add_column("decision")
    table.add_column("confidence")
    table.add_column("reviewer")
    table.add_column("reviewed_at")
    table.add_column("reason", max_width=48)
    for item in items:
        table.add_row(
            str(item.get("judgment_id", "")),
            f"{item.get('target_type', '')}:{item.get('target_id', '')}",
            str(item.get("decision", "")),
            str(item.get("confidence", "")),
            str(item.get("reviewer", "")),
            str(item.get("reviewed_at", "")),
            str(item.get("reason", "")),
        )
    console.print(table)
