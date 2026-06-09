from __future__ import annotations

import json
from pathlib import Path

import click

from knowledge_hub.application.research_review_loop import build_research_review_loop_report


@click.group("review-loop")
def review_loop_group() -> None:
    """Labs-only research review loop reports."""


@review_loop_group.command("report")
@click.option("--paper-id", "paper_ids", multiple=True, help="Explicit paper/arXiv id to include")
@click.option("--decision-file", default="", help="Optional reviewed decision JSON file")
@click.option("--out-dir", default="", help="Optional output directory for the report JSON")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def review_loop_report(ctx, paper_ids, decision_file, out_dir, as_json) -> None:
    selected_ids = [str(item).strip() for item in paper_ids if str(item).strip()]
    if not selected_ids:
        raise click.ClickException("at least one --paper-id is required")
    khub = ctx.obj["khub"]
    payload = build_research_review_loop_report(
        khub.sqlite_db(),
        paper_ids=selected_ids,
        decision_file=str(decision_file or "").strip() or None,
    )
    if out_dir:
        root = Path(str(out_dir)).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        report_path = root / "research_review_loop_report.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["contextPackPreview"]["outputPath"] = str(report_path)
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    click.echo(f"schema: {payload['schema']}")
    click.echo(f"status: {payload['status']}")
    for key, value in dict(payload.get("counts") or {}).items():
        click.echo(f"- {key}: {value}")
