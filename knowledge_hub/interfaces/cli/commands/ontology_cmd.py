"""khub ontology - predicate/pending 운영 명령."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from knowledge_hub.application.ontology_contributor_audit import audit_ontology_contributor_hashes
from knowledge_hub.knowledge.ontology_profiles import OntologyProfileManager

console = Console()


@click.group("ontology")
def ontology_group():
    """온톨로지 predicate / pending 운영"""
    pass


def _profile_manager(ctx) -> OntologyProfileManager:
    return OntologyProfileManager(_db(ctx))


def _db(ctx):
    khub = ctx.obj["khub"]
    if hasattr(khub, "sqlite_db"):
        return khub.sqlite_db()
    from knowledge_hub.infrastructure.persistence import SQLiteDatabase

    return SQLiteDatabase(khub.config.sqlite_path)


@ontology_group.group("profile")
def profile_group():
    """온톨로지 프로필 관리"""


@profile_group.command("list")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def profile_list(ctx, as_json):
    items = _profile_manager(ctx).list_profiles()
    if as_json:
        console.print_json(data={"status": "ok", "count": len(items), "items": items})
        return

    table = Table(title=f"Ontology Profiles ({len(items)})")
    table.add_column("profile_id", style="cyan")
    table.add_column("kind", style="magenta")
    table.add_column("active", justify="center")
    table.add_column("title")
    table.add_column("source_path", max_width=56)
    for item in items:
        table.add_row(
            str(item.get("profile_id", "")),
            str(item.get("kind", "")),
            "Y" if item.get("active") else "",
            str(item.get("title", "")),
            str(item.get("source_path", "")),
        )
    console.print(table)


@profile_group.command("show")
@click.argument("profile_id")
@click.option("--compiled", is_flag=True, default=False)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def profile_show(ctx, profile_id, compiled, as_json):
    manager = _profile_manager(ctx)
    payload = manager.compile_active_profile() if compiled else manager.get_profile(profile_id)
    if not payload:
        raise click.ClickException(f"profile not found: {profile_id}")
    if as_json:
        console.print_json(data=payload)
        return
    console.print_json(data=payload)


@profile_group.command("activate")
@click.argument("profile_id")
@click.option("--kind", type=click.Choice(["core", "domain", "personal"]), required=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def profile_activate(ctx, profile_id, kind, as_json):
    payload = _profile_manager(ctx).activate_profile(kind=kind, profile_id=profile_id)
    if as_json:
        console.print_json(data={"status": "ok", "item": payload})
        return
    console.print(f"[green]activated[/green] kind={kind} profile={profile_id}")


@profile_group.command("import")
@click.argument("source_path")
@click.option("--profile-id", default=None)
@click.option("--kind", type=click.Choice(["core", "domain", "personal"]), default="personal", show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def profile_import(ctx, source_path, profile_id, kind, as_json):
    payload = _profile_manager(ctx).import_profile(source_path, profile_id=profile_id, kind=kind)
    if as_json:
        console.print_json(data={"status": "ok", "item": payload})
        return
    console.print(f"[green]imported[/green] profile={payload.get('profile_id')} path={payload.get('source_path')}")


@profile_group.command("export")
@click.argument("profile_id")
@click.argument("destination")
@click.option("--compiled", is_flag=True, default=False)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def profile_export(ctx, profile_id, destination, compiled, as_json):
    payload = _profile_manager(ctx).export_profile(profile_id, destination, compiled=compiled)
    if as_json:
        console.print_json(data={"status": "ok", "item": payload})
        return
    console.print(f"[green]exported[/green] {destination}")


@ontology_group.group("proposal")
def proposal_group():
    """온톨로지 프로필 제안/승인"""


@proposal_group.command("submit")
@click.option("--type", "proposal_type", type=click.Choice(["entity_type", "predicate", "profile_patch"]), required=True)
@click.option("--target-profile", default="personal", show_default=True)
@click.option("--file", "file_path", default=None, help="JSON/YAML proposal 파일")
@click.option("--payload", default=None, help="inline JSON payload")
@click.option("--source", default="user", show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def proposal_submit(ctx, proposal_type, target_profile, file_path, payload, source, as_json):
    manager = _profile_manager(ctx)
    proposal_payload: dict
    if file_path:
        raw = Path(file_path).expanduser().resolve().read_text(encoding="utf-8")
        if file_path.endswith((".yaml", ".yml")):
            import yaml

            proposal_payload = yaml.safe_load(raw) or {}
        else:
            proposal_payload = json.loads(raw or "{}")
    else:
        proposal_payload = json.loads(payload or "{}")
    item = manager.submit_proposal(
        proposal_type=proposal_type,
        target_profile=target_profile,
        payload=proposal_payload,
        source=source,
    )
    if as_json:
        console.print_json(data={"status": "ok", "item": item})
        return
    console.print(f"[green]proposal submitted[/green] id={item.get('proposal_id')}")


@proposal_group.command("list")
@click.option("--status", default=None)
@click.option("--type", "proposal_type", default=None, type=click.Choice(["entity_type", "predicate", "profile_patch"]))
@click.option("--target-profile", default=None)
@click.option("--limit", default=100, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def proposal_list(ctx, status, proposal_type, target_profile, limit, as_json):
    items = _profile_manager(ctx).list_proposals(
        status=status,
        proposal_type=proposal_type,
        target_profile=target_profile,
        limit=limit,
    )
    if as_json:
        console.print_json(data={"status": "ok", "count": len(items), "items": items})
        return
    table = Table(title=f"Ontology Profile Proposals ({len(items)})")
    table.add_column("id", justify="right")
    table.add_column("type", style="cyan")
    table.add_column("target")
    table.add_column("status", style="magenta")
    table.add_column("source")
    table.add_column("summary", max_width=48)
    for item in items:
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        summary = payload.get("id") or payload.get("title") or json.dumps(payload, ensure_ascii=False)[:80]
        table.add_row(
            str(item.get("proposal_id", "")),
            str(item.get("proposal_type", "")),
            str(item.get("target_profile", "")),
            str(item.get("status", "")),
            str(item.get("source", "")),
            str(summary),
        )
    console.print(table)


@proposal_group.command("apply")
@click.argument("proposal_id", type=int)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def proposal_apply(ctx, proposal_id, as_json):
    item = _profile_manager(ctx).apply_proposal(proposal_id)
    if as_json:
        console.print_json(data={"status": "ok", "item": item})
        return
    console.print(f"[green]proposal applied[/green] id={proposal_id}")


@proposal_group.command("reject")
@click.argument("proposal_id", type=int)
@click.option("--reason", default="")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def proposal_reject(ctx, proposal_id, reason, as_json):
    item = _profile_manager(ctx).reject_proposal(proposal_id, {"reason": reason})
    if as_json:
        console.print_json(data={"status": "ok", "item": item})
        return
    console.print(f"[yellow]proposal rejected[/yellow] id={proposal_id}")


@ontology_group.command("predicate-list")
@click.option("--status", default=None, type=click.Choice(["core", "approved_ext", "deprecated"]))
@click.option("--limit", default=200, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def predicate_list(ctx, status, limit, as_json):
    db = _db(ctx)
    items = db.list_predicates(status=status, limit=max(1, int(limit)))
    if as_json:
        console.print_json(
            data={
                "status": "ok",
                "count": len(items),
                "items": items,
            }
        )
        return

    table = Table(title=f"Predicates ({len(items)})")
    table.add_column("predicate_id", style="cyan")
    table.add_column("status", style="magenta")
    table.add_column("parent")
    table.add_column("description")
    for item in items:
        table.add_row(
            str(item.get("predicate_id", "")),
            str(item.get("status", "")),
            str(item.get("parent_predicate_id", "") or "-"),
            str(item.get("description", "") or "-"),
        )
    console.print(table)


@ontology_group.command("contributor-audit", hidden=True)
@click.option("--apply", "apply_changes", is_flag=True, default=False, help="Fill missing contributor hashes.")
@click.option("--limit", default=50, show_default=True, help="Maximum sample rows to print.")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def contributor_audit(ctx, apply_changes, limit, as_json):
    """Audit/backfill concept contributor hashes for historical ontology rows."""
    db = _db(ctx)
    payload = audit_ontology_contributor_hashes(db.conn, apply=bool(apply_changes), sample_limit=max(1, int(limit)))
    if as_json:
        console.print_json(data=payload)
        return
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    console.print(
        "[bold]ontology contributor-audit[/bold] "
        f"apply={payload.get('apply')} "
        f"concepts={counts.get('conceptEntityCount', 0)} "
        f"candidates={counts.get('contributorCandidateCount', 0)} "
        f"missing_entities={counts.get('missingContributorEntityCount', 0)} "
        f"missing_hashes={counts.get('missingContributorHashCount', 0)} "
        f"updated={counts.get('updatedEntityCount', 0)}"
    )
    rows = payload.get("items") if isinstance(payload.get("items"), list) else []
    if not rows:
        return
    table = Table(title=f"Contributor Hash Gaps ({len(rows)})")
    table.add_column("entity_id", style="cyan")
    table.add_column("name")
    table.add_column("missing", max_width=56)
    table.add_column("action", style="magenta")
    for item in rows:
        table.add_row(
            str(item.get("entityId", "")),
            str(item.get("canonicalName", "")),
            ", ".join(str(value) for value in item.get("missingContributorHashes", [])),
            str(item.get("action", "")),
        )
    console.print(table)


@ontology_group.command("pending-list")
@click.option("--type", "pending_type", default=None, type=click.Choice(["concept", "relation", "claim", "predicate_ext"]))
@click.option("--topic", "topic_slug", default=None)
@click.option("--status", default="pending", type=click.Choice(["pending", "approved", "rejected"]))
@click.option("--limit", default=50, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def pending_list(ctx, pending_type, topic_slug, status, limit, as_json):
    db = _db(ctx)
    items = db.list_ontology_pending(
        pending_type=pending_type,
        topic_slug=topic_slug,
        status=status,
        limit=max(1, int(limit)),
    )
    if as_json:
        console.print_json(
            data={
                "status": "ok",
                "count": len(items),
                "items": items,
            }
        )
        return

    table = Table(title=f"Ontology Pending ({len(items)})")
    table.add_column("id", justify="right")
    table.add_column("type", style="cyan")
    table.add_column("predicate")
    table.add_column("confidence", justify="right")
    table.add_column("status", style="magenta")
    table.add_column("reason", max_width=56)
    for item in items:
        reason = item.get("reason_json") if isinstance(item.get("reason_json"), dict) else {}
        table.add_row(
            str(item.get("id", "")),
            str(item.get("pending_type", "")),
            str(item.get("predicate_id", "") or "-"),
            f"{float(item.get('confidence', 0.0)):.3f}",
            str(item.get("status", "")),
            json.dumps(reason, ensure_ascii=False)[:140],
        )
    console.print(table)


@ontology_group.command("pending-apply")
@click.argument("pending_id", type=int)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def pending_apply(ctx, pending_id, as_json):
    db = _db(ctx)
    item = db.apply_ontology_pending(int(pending_id))
    if not item:
        raise click.ClickException(f"pending apply failed: {pending_id}")
    if as_json:
        console.print_json(data={"status": "ok", "item": item})
    else:
        console.print(f"[green]approved[/green] pending id={pending_id} type={item.get('pending_type')}")


@ontology_group.command("pending-reject")
@click.argument("pending_id", type=int)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def pending_reject(ctx, pending_id, as_json):
    db = _db(ctx)
    item = db.reject_ontology_pending(int(pending_id))
    if not item:
        raise click.ClickException(f"pending reject failed: {pending_id}")
    if as_json:
        console.print_json(data={"status": "ok", "item": item})
    else:
        console.print(f"[yellow]rejected[/yellow] pending id={pending_id} type={item.get('pending_type')}")


@ontology_group.command("predicate-approve")
@click.argument("pending_id", type=int)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def predicate_approve(ctx, pending_id, as_json):
    db = _db(ctx)
    item = db.get_ontology_pending(int(pending_id))
    if not item:
        raise click.ClickException(f"pending not found: {pending_id}")
    if str(item.get("pending_type", "")) != "predicate_ext":
        raise click.ClickException(f"pending id {pending_id} is not predicate_ext")
    applied = db.apply_ontology_pending(int(pending_id))
    if not applied:
        raise click.ClickException(f"predicate approve failed: {pending_id}")
    if as_json:
        console.print_json(data={"status": "ok", "item": applied})
    else:
        console.print(f"[green]predicate approved[/green] pending id={pending_id}")


@ontology_group.command("merge-list")
@click.option("--topic", "topic_slug", default=None)
@click.option("--status", default="pending", type=click.Choice(["pending", "approved", "rejected"]))
@click.option("--limit", default=50, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def merge_list(ctx, topic_slug, status, limit, as_json):
    db = _db(ctx)
    items = db.list_entity_merge_proposals(
        topic_slug=topic_slug,
        status=status,
        limit=max(1, int(limit)),
    )
    if as_json:
        console.print_json(data={"status": "ok", "count": len(items), "items": items})
        return

    table = Table(title=f"Entity Merge Proposals ({len(items)})")
    table.add_column("id", justify="right")
    table.add_column("topic", style="cyan")
    table.add_column("source")
    table.add_column("target")
    table.add_column("cluster", justify="right")
    table.add_column("confidence", justify="right")
    table.add_column("method")
    table.add_column("status", style="magenta")
    for item in items:
        source_entity = item.get("source_entity") if isinstance(item.get("source_entity"), dict) else {}
        target_entity = item.get("target_entity") if isinstance(item.get("target_entity"), dict) else {}
        duplicate_cluster = item.get("duplicate_cluster") if isinstance(item.get("duplicate_cluster"), dict) else {}
        table.add_row(
            str(item.get("id", "")),
            str(item.get("topic_slug", "") or "-"),
            str(source_entity.get("canonical_name") or item.get("source_entity_id", "")),
            str(target_entity.get("canonical_name") or item.get("target_entity_id", "")),
            str(duplicate_cluster.get("size", 0) or "-"),
            f"{float(item.get('confidence', 0.0)):.3f}",
            str(item.get("match_method", "") or "-"),
            str(item.get("status", "")),
        )
    console.print(table)


@ontology_group.command("merge-apply")
@click.argument("proposal_id", type=int)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def merge_apply(ctx, proposal_id, as_json):
    db = _db(ctx)
    ok = db.apply_entity_merge_proposal(int(proposal_id))
    if not ok:
        raise click.ClickException(f"merge apply failed: {proposal_id}")
    payload = {
        "status": "ok",
        "applied": True,
        "proposal_id": int(proposal_id),
        "item": db.get_entity_merge_proposal(int(proposal_id)),
    }
    if as_json:
        console.print_json(data=payload)
    else:
        console.print(f"[green]merge applied[/green] proposal id={proposal_id}")


@ontology_group.command("merge-reject")
@click.argument("proposal_id", type=int)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def merge_reject(ctx, proposal_id, as_json):
    db = _db(ctx)
    ok = db.reject_entity_merge_proposal(int(proposal_id))
    if not ok:
        raise click.ClickException(f"merge reject failed: {proposal_id}")
    payload = {
        "status": "ok",
        "rejected": True,
        "proposal_id": int(proposal_id),
        "item": db.get_entity_merge_proposal(int(proposal_id)),
    }
    if as_json:
        console.print_json(data=payload)
    else:
        console.print(f"[yellow]merge rejected[/yellow] proposal id={proposal_id}")
