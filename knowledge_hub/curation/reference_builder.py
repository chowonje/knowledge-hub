"""Curated specialist reference-source batch builder."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from knowledge_hub.curation.latest_builder import (
    BATCH_SCHEMA_V2,
    DEFAULT_BATCH_DIR,
    _existing_canonical_urls,
    _load_yaml,
    _normalize_title_key,
    _normalize_url,
)
from knowledge_hub.curation.source_registry import (
    DiscoveredSourceItem,
    DiscoveryProvenance,
    SourceIdentity,
)
from knowledge_hub.infrastructure.config import Config

DEFAULT_REFERENCE_WATCHLIST = Path("data/curation/ai_watchlists/reference_sources.yaml")


def _today_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _reference_source_item_id(source_name: str, url: str, title_hint: str, index: int) -> str:
    title_key = _normalize_title_key(title_hint or source_name)
    raw = "|".join(
        [
            str(source_name or "").strip().lower(),
            str(url or "").strip().lower(),
            title_key,
            str(index),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _reference_item(
    *,
    source: dict[str, Any],
    entry: dict[str, Any],
    watchlist_path: Path,
    rank: int,
) -> DiscoveredSourceItem | None:
    url = _normalize_url(str(entry.get("url") or ""))
    if not url:
        return None
    canonical_url = _normalize_url(str(entry.get("canonical_url") or url))
    source_name = str(source.get("source_name") or "").strip() or "Reference Source"
    source_vendor = str(source.get("source_vendor") or "").strip() or "reference"
    source_channel = str(source.get("source_channel") or "").strip()
    if not source_channel:
        source_channel = _normalize_title_key(source_name).replace("-", "_") or "reference_source"
    source_channel_type = str(source.get("source_channel_type") or "").strip() or "reference_glossary"
    source_type = str(source.get("type") or source.get("source_type") or "").strip() or "reference_glossary"
    title_hint = str(entry.get("title_hint") or source_name).strip() or source_name
    tags: list[str] = []
    for token in list(source.get("topic_tags") or []) + list(entry.get("tags") or []):
        cleaned = str(token or "").strip()
        if cleaned and cleaned not in tags:
            tags.append(cleaned)
    source_item_id = str(entry.get("source_item_id") or "").strip() or _reference_source_item_id(
        source_name,
        canonical_url or url,
        title_hint,
        rank,
    )
    published_at = str(entry.get("published_at") or source.get("published_at") or "").strip()
    author = str(entry.get("author") or source.get("author") or "").strip()
    metadata = {
        "metadata_precedence": ["curated_reference_seed"],
        "reference_role": str(source.get("reference_role") or "background_reference").strip() or "background_reference",
        "reference_tier": str(source.get("reference_tier") or "specialist").strip() or "specialist",
        "topic_tags": list(source.get("topic_tags") or []),
        "source_description": str(source.get("description") or "").strip(),
        "entry_description": str(entry.get("description") or "").strip(),
    }
    return DiscoveredSourceItem(
        identity=SourceIdentity(
            source_vendor=source_vendor,
            source_channel=source_channel,
            source_channel_type=source_channel_type,
            source_item_id=source_item_id,
        ),
        source_name=source_name,
        source_type=source_type,
        url=url,
        canonical_url=canonical_url,
        title_hint=title_hint,
        published_at=published_at,
        author=author,
        tags=tags,
        provenance=DiscoveryProvenance(
            method="static_seed",
            origin_url=str(watchlist_path),
            entry_ref=source_item_id,
            discovered_at=datetime.now(timezone.utc).isoformat(),
            rank=rank,
        ),
        metadata=metadata,
    )


def build_reference_seed_batch(
    *,
    config: Config,
    watchlist_path: str | Path = DEFAULT_REFERENCE_WATCHLIST,
    output_prefix: str = "",
    include_existing: bool = False,
) -> dict[str, Any]:
    watchlist = Path(watchlist_path).expanduser().resolve()
    payload = _load_yaml(watchlist)
    sources = payload.get("sources") or []
    existing_urls = set() if include_existing else _existing_canonical_urls(config)

    batch_items: list[DiscoveredSourceItem] = []
    warnings: list[str] = []
    failed_sources: list[dict[str, str]] = []

    for source in sources:
        source_name = str(source.get("source_name") or "").strip() or "unknown"
        entries = source.get("entries") or []
        if not isinstance(entries, list) or not entries:
            warnings.append(f"{source_name}: no reference entries configured")
            failed_sources.append({"source_name": source_name, "error": "no reference entries configured"})
            continue
        kept = 0
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            item = _reference_item(
                source=source,
                entry=entry,
                watchlist_path=watchlist,
                rank=index,
            )
            if item is None:
                warnings.append(f"{source_name}: skipped empty url entry")
                continue
            normalized_url = str(item.url or "").strip()
            normalized_canonical = str(item.canonical_url or "").strip()
            if normalized_url in existing_urls or normalized_canonical in existing_urls:
                continue
            batch_items.append(item)
            kept += 1
        if kept == 0:
            failed_sources.append({"source_name": source_name, "error": "no fresh reference urls"})

    deduped: list[DiscoveredSourceItem] = []
    seen_urls: set[str] = set()
    for item in batch_items:
        key = str(item.canonical_url or item.url or "").strip()
        if key and key in seen_urls:
            continue
        if key:
            seen_urls.add(key)
        deduped.append(item)

    category = str(payload.get("category") or "reference_sources").strip() or "reference_sources"
    stamp = output_prefix.strip() or f"reference_sources_{_today_stamp()}"
    DEFAULT_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = DEFAULT_BATCH_DIR / f"{stamp}.txt"
    yaml_path = DEFAULT_BATCH_DIR / f"{stamp}.yaml"

    txt_path.write_text("\n".join(item.url for item in deduped) + ("\n" if deduped else ""), encoding="utf-8")
    yaml_payload = {
        "schema": BATCH_SCHEMA_V2,
        "version": 2,
        "kind": "ai_watchlist_batch",
        "category": category,
        "updated_at": datetime.now(timezone.utc).date().isoformat(),
        "count": len(deduped),
        "items": [item.to_dict() for item in deduped],
    }
    yaml_path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    by_source: dict[str, int] = {}
    for item in deduped:
        by_source[item.source_name] = by_source.get(item.source_name, 0) + 1

    return {
        "schema": BATCH_SCHEMA_V2,
        "version": 2,
        "kind": "ai_watchlist_batch",
        "category": category,
        "updated_at": yaml_payload["updated_at"],
        "status": "ok",
        "watchlist": str(watchlist),
        "outputPrefix": stamp,
        "txtPath": str(txt_path),
        "yamlPath": str(yaml_path),
        "count": len(deduped),
        "bySource": by_source,
        "failedSources": failed_sources,
        "warnings": warnings,
        "items": [item.to_dict() for item in deduped],
    }
