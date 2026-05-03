"""labs claim-cards - ClaimCard backfill and operator metrics."""

from __future__ import annotations

import json
from typing import Any

import click

from knowledge_hub.domain.ai_papers.claim_cards import ClaimCardBuilder
from knowledge_hub.papers.card_v2_builder import PaperCardV2Builder
from knowledge_hub.vault.card_v2_builder import VaultCardV2Builder
from knowledge_hub.web.card_v2_builder import WebCardV2Builder


def _db(ctx):
    return ctx.obj["khub"].sqlite_db()


def _source_cards(sqlite_db: Any, source: str) -> list[dict[str, Any]]:
    if source == "paper":
        return list(sqlite_db.paper_card_v2_store.list_cards(limit=5000))
    if source == "web":
        return list(sqlite_db.web_card_v2_store.list_cards(limit=5000))
    if source == "vault":
        return list(sqlite_db.vault_card_v2_store.list_cards(limit=5000))
    return []


def _paper_card_needs_claim_ref_repair(sqlite_db: Any, card: dict[str, Any]) -> bool:
    card_id = str(card.get("card_id") or "").strip()
    paper_id = str(card.get("paper_id") or "").strip()
    if not card_id or not paper_id:
        return False
    if sqlite_db.list_paper_card_claim_refs_v2(card_id=card_id):
        return False
    note_id = f"paper:{paper_id}"
    return bool(
        list(sqlite_db.list_claims_by_note(note_id, limit=1))
        or list(sqlite_db.list_claims_by_entity(note_id, limit=1))
    )


def _build_missing_source_cards(sqlite_db: Any, source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if source == "paper":
        builder = PaperCardV2Builder(sqlite_db)
        existing_cards = list(sqlite_db.paper_card_v2_store.list_cards(limit=5000))
        by_paper_id = {
            str(item.get("paper_id") or "").strip(): dict(item)
            for item in existing_cards
            if str(item.get("paper_id") or "").strip()
        }
        cards: list[dict[str, Any]] = list(existing_cards)
        failures: list[dict[str, Any]] = []
        repaired_cards: list[dict[str, Any]] = []
        for item in cards:
            if not _paper_card_needs_claim_ref_repair(sqlite_db, item):
                repaired_cards.append(dict(item))
                continue
            paper_id = str(item.get("paper_id") or "").strip()
            try:
                repaired_cards.append(builder.build_and_store(paper_id=paper_id))
            except Exception as error:  # pragma: no cover - operator resilience
                failures.append(
                    {
                        "sourceKind": source,
                        "sourceId": paper_id,
                        "stage": "source_card_repair",
                        "error": str(error),
                    }
                )
                repaired_cards.append(dict(item))
        cards = repaired_cards
        for row in sqlite_db.list_papers(limit=5000):
            paper_id = str(row.get("arxiv_id") or "").strip()
            if not paper_id or paper_id in by_paper_id:
                continue
            try:
                built = builder.build_and_store(paper_id=paper_id)
            except Exception as error:  # pragma: no cover - operator resilience
                failures.append(
                    {
                        "sourceKind": source,
                        "sourceId": paper_id,
                        "stage": "source_card_build",
                        "error": str(error),
                    }
                )
                continue
            cards.append(built)
        return cards, failures
    return _source_cards(sqlite_db, source), []


def _metrics_for_source(sqlite_db: Any, source: str) -> dict[str, Any]:
    source_cards = _source_cards(sqlite_db, source)
    claim_cards = list(sqlite_db.list_claim_cards(source_kind=source, limit=10000))
    extracted = [item for item in claim_cards if str(item.get("origin") or "") == "extracted"]
    fallback = [item for item in claim_cards if str(item.get("origin") or "") == "synthetic_fallback"]
    source_card_ids = {
        str(item.get("card_id") or "").strip()
        for item in source_cards
        if str(item.get("card_id") or "").strip()
    }
    covered_source_card_ids = {
        str(item.get("source_card_id") or "").strip()
        for item in sqlite_db.list_claim_card_source_refs()
        if str(item.get("source_kind") or "").strip() == source and str(item.get("source_card_id") or "").strip()
    }
    aligned_ids = {
        str(item.get("claim_card_id") or "")
        for item in sqlite_db.list_claim_card_alignment_refs(
            claim_card_ids=[str(card.get("claim_card_id") or "") for card in claim_cards],
        )
        if str(item.get("aligned_claim_card_id") or "").strip()
    }
    normalized_count = sum(
        1
        for item in claim_cards
        if any(
            str(item.get(field) or "").strip()
            for field in ("task_canonical", "dataset_canonical", "metric_canonical", "comparator_canonical")
        )
    )
    unsupported_count = sum(1 for item in claim_cards if str(item.get("status") or "").strip() == "unsupported")
    return {
        "sourceCardCount": len(source_cards),
        "claimCardCount": len(claim_cards),
        "claimCardCoverage": round(len(source_card_ids & covered_source_card_ids) / max(1, len(source_cards)), 4),
        "normalizedClaimCoverage": round(normalized_count / max(1, len(claim_cards)), 4),
        "alignedComparisonCoverage": round(len(aligned_ids) / max(1, len(claim_cards)), 4),
        "syntheticFallbackRate": round(len(fallback) / max(1, len(claim_cards)), 4),
        "unsupportedClaimRate": round(unsupported_count / max(1, len(claim_cards)), 4),
        "extractedClaimRate": round(len(extracted) / max(1, len(claim_cards)), 4),
    }


@click.group("claim-cards")
def claim_cards_group():
    """ClaimCard backfill and metrics."""


@claim_cards_group.command("backfill")
@click.option("--source", type=click.Choice(["paper", "web", "vault", "all"]), default="all", show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def claim_cards_backfill(ctx, source, as_json):
    sqlite_db = _db(ctx)
    builder = ClaimCardBuilder(sqlite_db)
    requested = ["paper", "web", "vault"] if source == "all" else [source]
    items: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source_kind in requested:
        cards, build_failures = _build_missing_source_cards(sqlite_db, source_kind)
        failures.extend(build_failures)
        for card in cards:
            claim_cards = builder.build_and_store_for_source_card(source_kind=source_kind, source_card=card)
            items.append(
                {
                    "sourceKind": source_kind,
                    "sourceCardId": str(card.get("card_id") or ""),
                    "sourceId": str(card.get("paper_id") or card.get("document_id") or card.get("note_id") or ""),
                    "claimCardCount": len(claim_cards),
                }
            )
    payload = {
        "schema": "knowledge-hub.claim-cards.backfill.result.v1",
        "status": "ok" if not failures else "partial",
        "requestedSources": requested,
        "count": len(items),
        "items": items,
        "failures": failures,
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    click.echo(f"schema: {payload['schema']}")
    click.echo(f"count: {payload['count']}")


@claim_cards_group.command("metrics")
@click.option("--source", type=click.Choice(["paper", "web", "vault", "all"]), default="all", show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def claim_cards_metrics(ctx, source, as_json):
    sqlite_db = _db(ctx)
    requested = ["paper", "web", "vault"] if source == "all" else [source]
    metrics = {source_kind: _metrics_for_source(sqlite_db, source_kind) for source_kind in requested}
    payload = {
        "schema": "knowledge-hub.claim-cards.metrics.result.v1",
        "status": "ok",
        "metrics": metrics,
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    click.echo(f"schema: {payload['schema']}")
    for source_kind, values in metrics.items():
        click.echo(f"[{source_kind}] {json.dumps(values, ensure_ascii=False)}")
