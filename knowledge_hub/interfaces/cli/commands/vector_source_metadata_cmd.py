from __future__ import annotations

import click
from rich.console import Console

from knowledge_hub.application.vector_source_metadata import audit_vector_source_metadata_for_config

console = Console()


@click.command("vector-source-metadata")
@click.option("--apply", is_flag=True, help="Backfill missing source metadata in the active vector store")
@click.option("--limit", type=int, default=10000, show_default=True)
@click.option("--sample-limit", type=int, default=10, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def vector_source_metadata_cmd(ctx, apply, limit, sample_limit, as_json):
    """Audit or backfill vector source lifecycle metadata."""

    khub = ctx.obj["khub"]
    payload = audit_vector_source_metadata_for_config(
        config=khub.config,
        limit=max(1, int(limit)),
        sample_limit=max(0, int(sample_limit)),
        apply=bool(apply),
    )
    if as_json:
        console.print_json(data=payload)
        return

    console.print(
        f"[bold]vector source metadata[/bold] status={payload.get('status')} "
        f"dryRun={payload.get('dryRun')} scanned={payload.get('scannedCount', 0)} "
        f"candidates={payload.get('updateCandidateCount', 0)} updated={payload.get('updatedCount', 0)}"
    )
    console.print(
        f"missing source hash={payload.get('missingSourceContentHashCount', 0)} "
        f"missing stale flag={payload.get('missingStaleCount', 0)}"
    )
    for warning in list(payload.get("warnings") or [])[:10]:
        console.print(f"[yellow]- {warning}[/yellow]")
    samples = list(payload.get("sampleMissing") or [])
    if samples:
        console.print("[dim]sample rows needing metadata:[/dim]")
        for item in samples[:5]:
            console.print(
                f"  - {item.get('id')} {item.get('sourceType') or '-'} "
                f"{item.get('title') or item.get('documentId') or item.get('filePath') or '-'}"
            )
    if not apply and int(payload.get("updateCandidateCount", 0) or 0) > 0:
        console.print("[dim]preview only; rerun with `--apply` to backfill metadata[/dim]")
