"""khub feature - feature snapshot and ranking commands."""

from __future__ import annotations

from typing import Any

import click
from rich.console import Console

from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.knowledge.features import snapshot_features
from knowledge_hub.learning.mapper import slugify_topic

console = Console()


def _db(ctx) -> SQLiteDatabase:
    khub = ctx.obj["khub"]
    if hasattr(khub, "sqlite_db"):
        return khub.sqlite_db()
    return SQLiteDatabase(khub.config.sqlite_path)


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]schema:[/bold] {payload.get('schema')}")
    console.print(f"[bold]status:[/bold] {payload.get('status')}")
    if payload.get("topic"):
        console.print(f"[bold]topic:[/bold] {payload.get('topic')}")
    if payload.get("counts"):
        for key, value in payload["counts"].items():
            console.print(f"- {key}: {value}")


@click.group("feature")
def feature_group():
    """feature snapshot / ranking"""


@feature_group.command("snapshot")
@click.option("--topic", required=True, help="핵심 토픽")
@click.option("--source-limit", default=500, type=int, show_default=True)
@click.option("--top-k", default=12, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def feature_snapshot(ctx, topic, source_limit, top_k, as_json):
    db = _db(ctx)
    summary = snapshot_features(db, topic=topic, source_limit=max(1, int(source_limit)), top_k=max(1, int(top_k)))
    payload = {
        "schema": "knowledge-hub.feature.snapshot.result.v1",
        "status": "ok",
        "topic": topic,
        "topicSlug": summary["topicSlug"],
        "counts": {
            "sourceCount": int(summary.get("sourceCount") or 0),
            "conceptCount": int(summary.get("conceptCount") or 0),
        },
        "topSources": summary.get("topSources") or [],
        "topConcepts": summary.get("topConcepts") or [],
    }
    _emit(payload, as_json)


@feature_group.command("top")
@click.option("--kind", "feature_kind", required=True, type=click.Choice(["source", "concept"]))
@click.option("--topic", required=True, help="핵심 토픽")
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def feature_top(ctx, feature_kind, topic, limit, as_json):
    db = _db(ctx)
    topic_slug = slugify_topic(topic)
    items = db.list_top_feature_snapshots(
        topic_slug=topic_slug,
        feature_kind=feature_kind,
        limit=max(1, int(limit)),
    )
    payload = {
        "schema": "knowledge-hub.feature.top.result.v1",
        "status": "ok",
        "topic": topic,
        "topicSlug": topic_slug,
        "kind": feature_kind,
        "counts": {"items": len(items)},
        "items": items,
    }
    _emit(payload, as_json)
