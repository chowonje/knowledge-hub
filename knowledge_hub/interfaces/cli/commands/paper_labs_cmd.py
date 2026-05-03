"""Labs/operator paper lane commands."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.interfaces.cli.commands.paper_shared_runtime import _resolve_vault_papers_dir
from knowledge_hub.papers.paper_lanes import (
    LANE_DEFINITIONS,
    all_lane_slugs,
    lane_hub_filename,
    normalize_lane_review_status,
    normalize_primary_lane,
    normalize_secondary_tags,
    seed_lane_metadata,
    summarize_lane_tag_counts,
)
from knowledge_hub.papers.topic_synthesis import PaperTopicSynthesisService
from knowledge_hub.papers.openai_batch_memory import OpenAIPaperMemoryBatchService

console = Console()

_REVIEW_COLUMNS = (
    "arxiv_id",
    "title",
    "year",
    "field",
    "importance",
    "primary_lane",
    "secondary_tags",
    "lane_review_status",
    "lane_updated_at",
    "review_notes",
)


@click.group("paper")
def paper_labs_group():
    """paper lane/operator workflows"""


@paper_labs_group.group("memory-batch")
def paper_memory_batch_group():
    """OpenAI Batch API operator workflow for paper-memory rebuilds."""


def _sqlite_db(khub):
    return khub.sqlite_db()


def _paper_rows(sqlite_db, *, limit: int = 5000) -> list[dict[str, Any]]:
    return list(sqlite_db.list_papers(limit=max(1, int(limit))) or [])


def _get_card(sqlite_db, paper_id: str) -> dict[str, Any]:
    getter = getattr(sqlite_db, "get_paper_memory_card", None)
    if callable(getter):
        row = getter(str(paper_id or "").strip())
        if isinstance(row, dict):
            return row
    return {}


def _searcher_for_khub(khub):
    factory = getattr(khub, "factory", None)
    if factory is not None and hasattr(factory, "searcher"):
        return factory.searcher()
    searcher_attr = getattr(khub, "searcher", None)
    if callable(searcher_attr):
        return searcher_attr()
    if searcher_attr is not None:
        return searcher_attr
    raise click.ClickException("search runtime unavailable")


def _default_review_csv_path(config) -> Path:
    root = Path(str(config.sqlite_path)).expanduser().parent / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root / "paper_lanes_review_v1.csv"


def _render_secondary_tags(tags: Any) -> str:
    return ",".join(normalize_secondary_tags(tags))


def _lane_paper_rows(sqlite_db, *, status: str | None = None) -> list[dict[str, Any]]:
    rows = _paper_rows(sqlite_db)
    if status:
        normalized = normalize_lane_review_status(status)
        rows = [row for row in rows if normalize_lane_review_status(row.get("lane_review_status")) == normalized]
    return rows


def _review_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lane_counts = {slug: 0 for slug in all_lane_slugs()}
    status_counts = {"seeded": 0, "reviewed": 0, "locked": 0}
    for row in rows:
        lane = normalize_primary_lane(row.get("primary_lane"))
        status = normalize_lane_review_status(row.get("lane_review_status") or "seeded")
        if lane:
            lane_counts[lane] += 1
        status_counts[status] += 1
    return {
        "rowCount": len(rows),
        "laneCounts": lane_counts,
        "statusCounts": status_counts,
    }


def _lane_note_path(vault_path: str, lane_slug: str) -> Path:
    vault_root = Path(vault_path).expanduser()
    preferred_ai_papers = vault_root / "Projects" / "AI" / "AI_Papers"
    papers_dir = preferred_ai_papers if preferred_ai_papers.exists() else _resolve_vault_papers_dir(vault_path)
    if papers_dir.name.lower() == "papers":
        lanes_dir = papers_dir.parent / "Lanes"
    else:
        lanes_dir = papers_dir / "Lanes"
    lanes_dir.mkdir(parents=True, exist_ok=True)
    return lanes_dir / lane_hub_filename(lane_slug)


def _paper_line(item: dict[str, Any]) -> str:
    tags = normalize_secondary_tags(item.get("secondary_tags") or item.get("secondary_tags_json") or [])
    tag_text = f" | tags: {', '.join(tags)}" if tags else ""
    year = item.get("year")
    year_text = f" | {year}" if year else ""
    importance = item.get("importance")
    importance_text = f" | importance={importance}" if importance is not None else ""
    return f"- `{item.get('arxiv_id')}` | {item.get('title')}{year_text}{importance_text}{tag_text}"


def _build_hub_content(lane_slug: str, papers: list[dict[str, Any]]) -> str:
    definition = LANE_DEFINITIONS[lane_slug]
    key_papers = sorted(
        papers,
        key=lambda item: (
            int(item.get("importance") or 0),
            int(item.get("year") or 0),
            str(item.get("title") or ""),
        ),
        reverse=True,
    )[:15]
    recent_papers = sorted(
        papers,
        key=lambda item: (
            str(item.get("lane_updated_at") or ""),
            str(item.get("created_at") or ""),
            str(item.get("arxiv_id") or ""),
        ),
        reverse=True,
    )[:10]
    tag_summary = summarize_lane_tag_counts(papers)[:12]

    lines = [
        "---",
        f"lane: {lane_slug}",
        f"title: {definition.title} Lane",
        f"paper_count: {len(papers)}",
        "---",
        "",
        f"# {definition.title} Lane",
        "",
        f"- Description: {definition.description}",
        f"- Inclusion: {definition.inclusion_criteria}",
        f"- Seed tags: {', '.join(definition.seed_tags)}",
        "",
        "## Representative Questions",
    ]
    lines.extend(f"- {question}" for question in definition.questions)
    lines.extend(["", "## Key Papers"])
    if key_papers:
        lines.extend(_paper_line(item) for item in key_papers)
    else:
        lines.append("- None yet")
    lines.extend(["", "## Recent Papers"])
    if recent_papers:
        lines.extend(_paper_line(item) for item in recent_papers)
    else:
        lines.append("- None yet")
    lines.extend(["", "## Secondary Tag Summary"])
    if tag_summary:
        lines.extend(f"- `{tag}`: {count}" for tag, count in tag_summary)
    else:
        lines.append("- None yet")
    lines.append("")
    return "\n".join(lines)


@paper_labs_group.command("lanes-backfill")
@click.option("--limit", type=int, default=5000, show_default=True)
@click.option("--force/--only-missing", default=False, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def paper_lanes_backfill(ctx, limit, force, as_json):
    """Seed primary_lane + secondary_tags for current papers."""
    khub = ctx.obj["khub"]
    sqlite_db = _sqlite_db(khub)
    rows = _paper_rows(sqlite_db, limit=limit)
    updated = 0
    skipped = 0
    unclassified = 0
    lane_counts = {slug: 0 for slug in all_lane_slugs()}
    items: list[dict[str, Any]] = []

    for row in rows:
        existing_lane = normalize_primary_lane(row.get("primary_lane"))
        existing_tags = normalize_secondary_tags(row.get("secondary_tags") or row.get("secondary_tags_json") or [])
        review_status = normalize_lane_review_status(row.get("lane_review_status") or "seeded")
        if review_status in {"reviewed", "locked"}:
            skipped += 1
            if existing_lane:
                lane_counts[existing_lane] += 1
            items.append(
                {
                    "paperId": row.get("arxiv_id"),
                    "title": row.get("title"),
                    "primaryLane": row.get("primary_lane"),
                    "secondaryTags": existing_tags,
                    "reviewStatus": review_status,
                    "status": "skipped_locked" if review_status == "locked" else "skipped_reviewed",
                }
            )
            continue
        if not force and existing_lane:
            skipped += 1
            lane_counts[existing_lane] += 1
            items.append(
                {
                    "paperId": row.get("arxiv_id"),
                    "title": row.get("title"),
                    "primaryLane": row.get("primary_lane"),
                    "secondaryTags": existing_tags,
                    "reviewStatus": review_status,
                    "status": "skipped",
                }
            )
            continue
        seeded = seed_lane_metadata(row, _get_card(sqlite_db, str(row.get("arxiv_id") or "")))
        if not seeded.get("primary_lane"):
            unclassified += 1
            if force and (existing_lane or existing_tags):
                sqlite_db.update_paper_lane_metadata(
                    arxiv_id=str(row.get("arxiv_id") or ""),
                    primary_lane=None,
                    secondary_tags=[],
                    lane_review_status="seeded",
                    lane_updated_at=str(seeded["lane_updated_at"]),
                )
            items.append(
                {
                    "paperId": row.get("arxiv_id"),
                    "title": row.get("title"),
                    "primaryLane": "",
                    "secondaryTags": [],
                    "reviewStatus": "seeded" if force and (existing_lane or existing_tags) else "",
                    "status": "cleared_non_ai" if force and (existing_lane or existing_tags) else "unclassified",
                }
            )
            continue
        sqlite_db.update_paper_lane_metadata(
            arxiv_id=str(row.get("arxiv_id") or ""),
            primary_lane=seeded["primary_lane"],
            secondary_tags=seeded["secondary_tags"],
            lane_review_status="seeded",
            lane_updated_at=str(seeded["lane_updated_at"]),
        )
        updated += 1
        lane_counts[seeded["primary_lane"]] += 1
        items.append(
            {
                "paperId": row.get("arxiv_id"),
                "title": row.get("title"),
                "primaryLane": seeded["primary_lane"],
                "secondaryTags": list(seeded["secondary_tags"]),
                "reviewStatus": "seeded",
                "status": "updated",
            }
        )

    payload = {
        "schema": "knowledge-hub.paper-lanes.backfill.result.v1",
        "status": "ok",
        "requested": len(rows),
        "updated": updated,
        "skipped": skipped,
        "unclassified": unclassified,
        "laneCounts": lane_counts,
        "items": items,
        "warnings": [],
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print(
        f"[bold]paper lanes backfill[/bold] requested={payload['requested']} updated={updated} skipped={skipped}"
    )


@paper_labs_group.command("lanes-review")
@click.option("--out", "out_path", default=None, help="CSV output path")
@click.option("--status", "status_filter", type=click.Choice(["seeded", "reviewed", "locked"]), default=None)
@click.option("--lane", "lane_filter", type=click.Choice(list(all_lane_slugs())), default=None)
@click.option("--summary-out", "summary_out", default=None, help="Optional JSON summary output path")
@click.option("--apply-csv", "apply_csv", default=None, help="Apply reviewed CSV back into SQLite")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def paper_lanes_review(ctx, out_path, status_filter, lane_filter, summary_out, apply_csv, as_json):
    """Export or apply lane review CSV."""
    khub = ctx.obj["khub"]
    sqlite_db = _sqlite_db(khub)

    if apply_csv:
        path = Path(str(apply_csv)).expanduser()
        if not path.exists():
            raise click.ClickException(f"review CSV not found: {path}")
        updated = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                paper_id = str(row.get("arxiv_id") or "").strip()
                if not paper_id:
                    continue
                sqlite_db.update_paper_lane_metadata(
                    arxiv_id=paper_id,
                    primary_lane=normalize_primary_lane(row.get("primary_lane")),
                    secondary_tags=normalize_secondary_tags(row.get("secondary_tags") or ""),
                    lane_review_status=normalize_lane_review_status(row.get("lane_review_status") or "reviewed"),
                    lane_updated_at=str(row.get("lane_updated_at") or ""),
                )
                updated += 1
        payload = {
            "schema": "knowledge-hub.paper-lanes.review.apply.result.v1",
            "status": "ok",
            "updated": updated,
            "path": str(path),
            "warnings": [],
        }
        if as_json:
            click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        console.print(f"[bold]paper lanes review apply[/bold] updated={updated} path={path}")
        return

    rows = _lane_paper_rows(sqlite_db, status=status_filter)
    if lane_filter:
        rows = [row for row in rows if normalize_primary_lane(row.get("primary_lane")) == lane_filter]
    review_path = Path(str(out_path)).expanduser() if out_path else _default_review_csv_path(khub.config)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_REVIEW_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "arxiv_id": row.get("arxiv_id"),
                    "title": row.get("title"),
                    "year": row.get("year"),
                    "field": row.get("field"),
                    "importance": row.get("importance"),
                    "primary_lane": row.get("primary_lane") or "",
                    "secondary_tags": _render_secondary_tags(row.get("secondary_tags") or row.get("secondary_tags_json") or []),
                    "lane_review_status": row.get("lane_review_status") or "seeded",
                    "lane_updated_at": row.get("lane_updated_at") or "",
                    "review_notes": "",
                }
            )
    summary_payload = _review_summary(rows)
    if summary_out:
        summary_path = Path(str(summary_out)).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "schema": "knowledge-hub.paper-lanes.review.summary.v1",
                    "status": "ok",
                    "path": str(review_path),
                    **summary_payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    payload = {
        "schema": "knowledge-hub.paper-lanes.review.export.result.v1",
        "status": "ok",
        "rowCount": len(rows),
        "path": str(review_path),
        "laneFilter": lane_filter or "",
        "summary": summary_payload,
        "summaryPath": str(Path(str(summary_out)).expanduser()) if summary_out else "",
        "warnings": [],
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print(f"[bold]paper lanes review export[/bold] rows={len(rows)} out={review_path}")


@paper_labs_group.command("lanes-sync-hubs")
@click.option("--vault-path", default=None, help="Override Obsidian vault path")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def paper_lanes_sync_hubs(ctx, vault_path, as_json):
    """Generate idempotent lane hub notes under the paper vault."""
    khub = ctx.obj["khub"]
    sqlite_db = _sqlite_db(khub)
    resolved_vault = str(vault_path or khub.config.vault_path or "").strip()
    if not resolved_vault:
        raise click.ClickException("vault_path not configured")
    all_papers = _paper_rows(sqlite_db, limit=5000)
    written_paths: list[str] = []
    lane_counts: dict[str, int] = {}
    for lane_slug in all_lane_slugs():
        lane_papers = [paper for paper in all_papers if paper.get("primary_lane") == lane_slug]
        lane_counts[lane_slug] = len(lane_papers)
        note_path = _lane_note_path(resolved_vault, lane_slug)
        note_path.write_text(_build_hub_content(lane_slug, lane_papers), encoding="utf-8")
        written_paths.append(str(note_path))
    payload = {
        "schema": "knowledge-hub.paper-lanes.sync-hubs.result.v1",
        "status": "ok",
        "writtenCount": len(written_paths),
        "laneCounts": lane_counts,
        "paths": written_paths,
        "warnings": [],
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print(f"[bold]paper lanes sync-hubs[/bold] written={len(written_paths)}")


@paper_labs_group.command("topic-synthesize")
@click.argument("query")
@click.option("--source-mode", type=click.Choice(["local", "discover", "hybrid"]), default="local", show_default=True)
@click.option("--candidate-limit", type=int, default=12, show_default=True)
@click.option("--selected-limit", type=int, default=6, show_default=True)
@click.option("--top-k", type=int, default=8, show_default=True)
@click.option("--mode", "retrieval_mode", type=click.Choice(["semantic", "keyword", "hybrid"]), default="hybrid", show_default=True)
@click.option("--alpha", type=float, default=0.7, show_default=True)
@click.option("--allow-external/--no-allow-external", default=False, show_default=True)
@click.option("--llm-mode", type=click.Choice(["auto", "local", "mini", "strong", "fallback-only"]), default="auto", show_default=True)
@click.option("--provider", "provider_override", default=None, help="Override synthesis/judge provider")
@click.option("--model", "model_override", default=None, help="Override synthesis/judge model")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def paper_topic_synthesize(
    ctx,
    query,
    source_mode,
    candidate_limit,
    selected_limit,
    top_k,
    retrieval_mode,
    alpha,
    allow_external,
    llm_mode,
    provider_override,
    model_override,
    as_json,
):
    """Synthesize a multi-paper topic answer from the local corpus."""
    khub = ctx.obj["khub"]
    payload = PaperTopicSynthesisService(
        sqlite_db=_sqlite_db(khub),
        searcher=_searcher_for_khub(khub),
        config=khub.config,
    ).synthesize(
        query=query,
        source_mode=source_mode,
        candidate_limit=max(1, int(candidate_limit)),
        selected_limit=max(1, int(selected_limit)),
        top_k=max(1, int(top_k)),
        retrieval_mode=retrieval_mode,
        alpha=float(alpha),
        allow_external=bool(allow_external),
        llm_mode=llm_mode,
        provider_override=provider_override,
        model_override=model_override,
    )
    annotate_schema_errors(payload, payload.get("schema", ""))
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print(
        f"[bold]paper topic synthesize[/bold] candidates={len(payload['candidatePapers'])} selected={len(payload['selectedPapers'])}"
    )
    if payload.get("topicSummary"):
        console.print(str(payload["topicSummary"]))
    if payload.get("warnings"):
        console.print(f"warnings: {', '.join(str(item) for item in payload['warnings'])}")


@paper_memory_batch_group.command("prepare")
@click.option("--output-dir", default=None, help="Batch artifact directory")
@click.option("--paper-id", "paper_ids", multiple=True, help="Specific paper id to include (repeatable)")
@click.option("--paper-ids-file", default=None, help="Text file with one paper id per line")
@click.option("--limit", type=int, default=50, show_default=True)
@click.option("--model", default="gpt-5.4", show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def paper_memory_batch_prepare(ctx, output_dir, paper_ids, paper_ids_file, limit, model, as_json):
    """Prepare OpenAI Batch API requests for paper-memory cards."""
    khub = ctx.obj["khub"]
    result = OpenAIPaperMemoryBatchService(khub, model=model).prepare(
        output_dir=output_dir,
        paper_ids=list(paper_ids or []),
        paper_ids_file=paper_ids_file,
        limit=max(1, int(limit)),
        model=model,
    )
    payload = {
        "schema": "knowledge-hub.paper-memory.openai-batch.prepare.result.v1",
        "status": "ok",
        "paperCount": int(result.paper_count),
        "blockedCount": int(result.blocked_count),
        "manifestPath": str(result.manifest_path),
        "requestsPath": str(result.requests_path),
        "summaryPath": str(result.summary_path),
        "warnings": [],
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print(
        f"[bold]paper-memory batch prepare[/bold] papers={payload['paperCount']} blocked={payload['blockedCount']}"
    )
    console.print(f"manifest: {payload['manifestPath']}")


def _require_allow_external(allow_external: bool) -> None:
    if not bool(allow_external):
        raise click.ClickException("--allow-external 플래그가 필요합니다.")


@paper_memory_batch_group.command("submit")
@click.option("--manifest", "manifest_path", required=True)
@click.option("--allow-external/--no-allow-external", default=False, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def paper_memory_batch_submit(ctx, manifest_path, allow_external, as_json):
    """Submit a prepared OpenAI Batch API request file."""
    _require_allow_external(allow_external)
    khub = ctx.obj["khub"]
    batch = OpenAIPaperMemoryBatchService(khub).submit(manifest_path=manifest_path)
    payload = {
        "schema": "knowledge-hub.paper-memory.openai-batch.submit.result.v1",
        "status": "ok",
        "manifestPath": str(manifest_path),
        "batch": batch,
        "warnings": [],
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print(f"[bold]paper-memory batch submit[/bold] batch={batch.get('batchId')} status={batch.get('status')}")


@paper_memory_batch_group.command("status")
@click.option("--manifest", "manifest_path", required=True)
@click.option("--allow-external/--no-allow-external", default=False, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def paper_memory_batch_status(ctx, manifest_path, allow_external, as_json):
    """Refresh OpenAI Batch API status for a prepared run."""
    _require_allow_external(allow_external)
    khub = ctx.obj["khub"]
    batch = OpenAIPaperMemoryBatchService(khub).status(manifest_path=manifest_path)
    payload = {
        "schema": "knowledge-hub.paper-memory.openai-batch.status.result.v1",
        "status": "ok",
        "manifestPath": str(manifest_path),
        "batch": batch,
        "warnings": [],
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print(f"[bold]paper-memory batch status[/bold] status={batch.get('status')}")


@paper_memory_batch_group.command("apply")
@click.option("--manifest", "manifest_path", required=True)
@click.option("--allow-external/--no-allow-external", default=False, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def paper_memory_batch_apply(ctx, manifest_path, allow_external, as_json):
    """Download completed batch results and apply them to paper-memory cards."""
    _require_allow_external(allow_external)
    khub = ctx.obj["khub"]
    summary = OpenAIPaperMemoryBatchService(khub).apply(manifest_path=manifest_path)
    payload = {
        "schema": "knowledge-hub.paper-memory.openai-batch.apply.result.v1",
        "status": "ok",
        "manifestPath": str(manifest_path),
        "summary": summary,
        "warnings": [],
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print(
        f"[bold]paper-memory batch apply[/bold] applied={summary.get('appliedCount')} failed={summary.get('failedCount')}"
    )
