from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from knowledge_hub.application.vector_restore import compare_vector_backup

console = Console()


@click.command("vector-compare")
@click.option(
    "--backup-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Compare against this explicit backup directory instead of the latest sibling .corrupt.* backup",
)
@click.option("--latest-backup", is_flag=True, help="Use the most recent sibling .corrupt.* backup")
@click.option("--sample-limit", type=int, default=10, show_default=True)
@click.option("--document-limit", type=int, default=10000, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def vector_compare_cmd(ctx, backup_path, latest_backup, sample_limit, document_limit, as_json):
    """Compare the active vector store to a selected backup without modifying either path."""
    if backup_path is not None and latest_backup:
        raise click.ClickException("--backup-path and --latest-backup cannot be used together")
    khub = ctx.obj["khub"]
    payload = compare_vector_backup(
        config=khub.config,
        backup_path=backup_path,
        use_latest_backup=bool(latest_backup or backup_path is None),
        sample_limit=max(1, int(sample_limit)),
        document_limit=max(1, int(document_limit)),
    )
    if as_json:
        console.print_json(data=payload)
        return
    active = dict(payload.get("activeVector") or {})
    backup = dict(payload.get("backupVector") or {})
    diff = dict(payload.get("diff") or {})
    provenance = dict(diff.get("provenance") or {})
    console.print(
        f"[bold]vector compare[/bold] status={payload.get('status')} "
        f"active={active.get('total_documents', 0)} backup={backup.get('total_documents', 0)}"
    )
    for warning in list(payload.get("warnings") or [])[:10]:
        console.print(f"[yellow]- {warning}[/yellow]")
    for error in list(payload.get("errors") or [])[:10]:
        console.print(f"[red]- {error}[/red]")
    if diff:
        console.print(
            f"shared={diff.get('sharedCount', 0)} "
            f"active_only={diff.get('activeOnlyCount', 0)} "
            f"backup_only={diff.get('backupOnlyCount', 0)} "
            f"changed_shared={diff.get('changedSharedCount', 0)}"
        )
        backup_only = dict(provenance.get("backupOnly") or {})
        decision = dict(provenance.get("decisionHint") or {})
        source_counts = dict(backup_only.get("sourceTypeCounts") or {})
        publisher_counts = dict(backup_only.get("publisherCounts") or {})
        if source_counts:
            source_text = ", ".join(f"{key}={value}" for key, value in source_counts.items())
            console.print(f"[dim]backup_only sources:[/dim] {source_text}")
        if publisher_counts:
            publisher_text = ", ".join(f"{key}={value}" for key, value in publisher_counts.items())
            console.print(f"[dim]backup_only publishers:[/dim] {publisher_text}")
        if backup_only.get("uniqueFileCount"):
            console.print(f"[dim]backup_only unique files:[/dim] {backup_only.get('uniqueFileCount')}")
        if decision:
            console.print(
                f"[dim]heuristic:[/dim] {decision.get('recommendedAction') or '-'} "
                f"- {decision.get('summary') or '-'}"
            )
        for label, key in (
            ("active only", "activeOnlySample"),
            ("backup only", "backupOnlySample"),
        ):
            sample = list(diff.get(key) or [])
            if not sample:
                continue
            console.print(f"[dim]{label} sample:[/dim]")
            for item in sample[:5]:
                console.print(f"  - {item.get('doc_id')} {item.get('title') or '-'}")
        changed_sample = list(diff.get("changedSharedSample") or [])
        if changed_sample:
            console.print("[dim]changed shared sample:[/dim]")
            for item in changed_sample[:5]:
                active_item = dict(item.get("active") or {})
                backup_item = dict(item.get("backup") or {})
                console.print(
                    f"  - {item.get('doc_id')} active={active_item.get('title') or '-'} "
                    f"backup={backup_item.get('title') or '-'}"
                )
    command = str(dict(payload.get("action") or {}).get("recommendedRestoreCommand") or "").strip()
    if command:
        console.print(f"[dim]restore candidate: `{command}`[/dim]")
