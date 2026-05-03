"""khub labs transform - inspectable bounded transformations."""

from __future__ import annotations

import click
from rich.console import Console

from knowledge_hub.application.transformations import list_transformations, preview_transformation, run_transformation

console = Console()


@click.group("transform")
def transform_group():
    """bounded prompt-based transformations"""


@transform_group.command("list")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
def transform_list(as_json):
    payload = list_transformations()
    if as_json:
        console.print_json(data=payload)
        return
    for item in payload.get("items") or []:
        console.print(f"- {item['id']} ({item['version']}): {item['description']}")


@transform_group.command("preview")
@click.argument("transformation_id")
@click.argument("query")
@click.option("--repo-path", default=None)
@click.option("--workspace/--no-workspace", "include_workspace", default=False, show_default=True)
@click.option("--vault/--no-vault", "include_vault", default=True, show_default=True)
@click.option("--papers/--no-papers", "include_papers", default=True, show_default=True)
@click.option("--web/--no-web", "include_web", default=True, show_default=True)
@click.option("--max-sources", default=6, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def transform_preview(ctx, transformation_id, query, repo_path, include_workspace, include_vault, include_papers, include_web, max_sources, as_json):
    khub = ctx.obj["khub"]
    payload = preview_transformation(
        khub.searcher(),
        sqlite_db=khub.sqlite_db(),
        transformation_id=transformation_id,
        query=query,
        repo_path=repo_path,
        include_workspace=include_workspace,
        include_vault=include_vault,
        include_papers=include_papers,
        include_web=include_web,
        max_sources=max_sources,
    )
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]{payload['transformation']['title']}[/bold]")
    console.print(payload.get("prompt_preview") or "")
    for item in payload.get("selected_sources") or []:
        console.print(f"- [{item['normalized_source_type']}] {item['title']}")


@transform_group.command("run")
@click.argument("transformation_id")
@click.argument("query")
@click.option("--repo-path", default=None)
@click.option("--workspace/--no-workspace", "include_workspace", default=False, show_default=True)
@click.option("--vault/--no-vault", "include_vault", default=True, show_default=True)
@click.option("--papers/--no-papers", "include_papers", default=True, show_default=True)
@click.option("--web/--no-web", "include_web", default=True, show_default=True)
@click.option("--max-sources", default=6, show_default=True)
@click.option("--dry-run", is_flag=True, help="preview selection and policy without running the model")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def transform_run(ctx, transformation_id, query, repo_path, include_workspace, include_vault, include_papers, include_web, max_sources, dry_run, as_json):
    khub = ctx.obj["khub"]
    payload = run_transformation(
        khub.searcher(),
        sqlite_db=khub.sqlite_db(),
        llm=khub.app_context.get_summarizer(),
        config=khub.config,
        transformation_id=transformation_id,
        query=query,
        repo_path=repo_path,
        include_workspace=include_workspace,
        include_vault=include_vault,
        include_papers=include_papers,
        include_web=include_web,
        max_sources=max_sources,
        dry_run=dry_run,
    )
    if as_json:
        console.print_json(data=payload)
        return
    if payload["status"] == "blocked":
        console.print("[yellow]transformation blocked by policy[/yellow]")
    console.print(payload.get("output") or "(no output)")
