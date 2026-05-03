from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from knowledge_hub.application.vector_restore import restore_vector_backup

console = Console()


@click.command("vector-restore")
@click.option(
    "--backup-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Restore from this explicit backup directory instead of the latest sibling .corrupt.* backup",
)
@click.option("--latest-backup", is_flag=True, help="Use the most recent sibling .corrupt.* backup")
@click.option("--apply", is_flag=True, help="Replace the active vector path with the selected backup")
@click.option("--confirm", is_flag=True, help="Required with --apply to execute the restore")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def vector_restore_cmd(ctx, backup_path, latest_backup, apply, confirm, as_json):
    """Preview or restore a vector backup into the active Chroma path."""
    if backup_path is not None and latest_backup:
        raise click.ClickException("--backup-path and --latest-backup cannot be used together")
    khub = ctx.obj["khub"]
    payload = restore_vector_backup(
        config=khub.config,
        backup_path=backup_path,
        use_latest_backup=bool(latest_backup or backup_path is None),
        apply=bool(apply),
        confirm=bool(confirm),
    )
    if as_json:
        console.print_json(data=payload)
        return

    selection = dict(payload.get("selection") or {})
    active = dict(payload.get("activeVector") or {})
    backup = dict(payload.get("backupVector") or {})
    action = dict(payload.get("action") or {})
    console.print(
        f"[bold]vector restore[/bold] status={payload.get('status')} "
        f"dryRun={payload.get('dryRun')} applied={payload.get('applied')}"
    )
    console.print(
        f"active={active.get('collection_name') or '-'} / {active.get('total_documents', 0)} "
        f"backup={selection.get('selectedPath') or '-'} / {backup.get('total_documents', 0)}"
    )
    if payload.get("activeBackupPath"):
        console.print(f"[dim]previous active moved to {payload.get('activeBackupPath')}[/dim]")
    for warning in list(payload.get("warnings") or [])[:10]:
        console.print(f"[yellow]- {warning}[/yellow]")
    for error in list(payload.get("errors") or [])[:10]:
        console.print(f"[red]- {error}[/red]")
    if not apply and action.get("canRestore"):
        console.print(f"[dim]preview only; run `{action.get('recommendedApplyCommand')}` to apply[/dim]")
