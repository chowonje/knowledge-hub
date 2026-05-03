"""Curated latest-batch builder for continuous AI source watchlists."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.curation.source_registry import DiscoveredSourceItem, build_source_registry

DEFAULT_WATCHLIST = Path("data/curation/ai_watchlists/continuous_sources.yaml")
DEFAULT_BATCH_DIR = Path("data/curation/ai_watchlists/batches")
DEFAULT_PRIORITY_TXT = Path("data/curation/ai_watchlists/priority_documents.txt")
BATCH_SCHEMA_V2 = "knowledge-hub.ai-watchlist-batch.v2"


def _today_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_priority_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        _normalize_url(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _existing_canonical_urls(config: Config) -> set[str]:
    db_path = Path(config.sqlite_path).expanduser()
    if not db_path.exists():
        return set()

    db = SQLiteDatabase(
        str(db_path),
        enable_event_store=False,
        bootstrap=False,
        read_only=True,
    )
    try:
        try:
            rows = db.conn.execute(
                "select distinct canonical_url from crawl_pipeline_records where canonical_url != ''"
            ).fetchall()
        except sqlite3.OperationalError:
            return set()
        return {_normalize_url(str(row[0]).strip()) for row in rows if str(row[0]).strip()}
    finally:
        db.close()


def _normalize_title_key(value: str) -> str:
    lowered = _clean_text(value).lower()
    lowered = re.sub(r"[^a-z0-9가-힣]+", "-", lowered)
    lowered = re.sub(r"-+", "-", lowered).strip("-")
    return lowered


def _normalize_url(value: str) -> str:
    token = _clean_text(value)
    if not token:
        return ""
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(token)
    scheme = parsed.scheme or "https"
    path = re.sub(r"/+", "/", parsed.path or "/").rstrip("/") or "/"
    return urlunparse((scheme, parsed.netloc.lower(), path, "", parsed.query, ""))


def _lightweight_hash(item: DiscoveredSourceItem) -> str:
    parts = [item.source_name, item.title_hint, item.published_at]
    raw = "|".join(part for part in parts if part)
    if not raw:
        return ""
    import hashlib

    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _dedupe_key(item: DiscoveredSourceItem) -> tuple[str, str]:
    source_item_id = str(item.identity.source_item_id or "").strip()
    if source_item_id:
        return ("source_item_id", source_item_id)
    canonical = str(item.canonical_url or "").strip()
    if canonical:
        return ("canonical_url", canonical)
    normalized_url = str(item.url or "").strip()
    if normalized_url:
        return ("normalized_url", normalized_url)
    light_hash = _lightweight_hash(item)
    if light_hash:
        return ("lightweight_content_hash", light_hash)
    title_date = "|".join(
        [
            str(item.identity.source_channel or "").strip(),
            _normalize_title_key(item.title_hint),
            str(item.published_at or "").strip(),
        ]
    )
    return ("title_date", title_date)


def load_latest_batch_items(batch_yaml_path: str | Path) -> list[dict[str, Any]]:
    path = Path(batch_yaml_path).expanduser().resolve()
    if not path.exists():
        return []
    payload = _load_yaml(path)
    items = payload.get("items") or []
    if not isinstance(items, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            cleaned.append(dict(item))
    return cleaned


def build_continuous_latest_batch(
    *,
    config: Config,
    watchlist_path: str | Path = DEFAULT_WATCHLIST,
    output_prefix: str = "",
    per_source_limit: int = 4,
    include_existing: bool = False,
) -> dict[str, Any]:
    watchlist = Path(watchlist_path).expanduser().resolve()
    payload = _load_yaml(watchlist)
    sources = payload.get("sources") or []
    priority_urls = _load_priority_urls(DEFAULT_PRIORITY_TXT)
    existing_urls = set() if include_existing else _existing_canonical_urls(config)
    registry = build_source_registry()

    batch_items: list[DiscoveredSourceItem] = []
    warnings: list[str] = []
    failed_sources: list[dict[str, str]] = []

    for source in sources:
        source_name = str(source.get("source_name") or "").strip()
        if not source_name:
            continue
        items, item_warnings = registry.discover_latest(source, max(1, int(per_source_limit)))
        warnings.extend(item_warnings)
        if item_warnings and not items:
            failed_sources.append({"source_name": source_name, "error": "; ".join(item_warnings[:3])})
        for item in items:
            normalized_url = str(item.url or "").strip()
            normalized_canonical = str(item.canonical_url or "").strip()
            if not normalized_url:
                continue
            if normalized_url in priority_urls or normalized_canonical in priority_urls:
                continue
            if normalized_url in existing_urls or normalized_canonical in existing_urls:
                continue
            batch_items.append(item)

    deduped: list[DiscoveredSourceItem] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in batch_items:
        key = _dedupe_key(item)
        if key[1] and key in seen_keys:
            continue
        if key[1]:
            seen_keys.add(key)
        deduped.append(item)

    stamp = output_prefix.strip() or f"continuous_latest_{_today_stamp()}"
    DEFAULT_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = DEFAULT_BATCH_DIR / f"{stamp}.txt"
    yaml_path = DEFAULT_BATCH_DIR / f"{stamp}.yaml"

    txt_path.write_text("\n".join(item.url for item in deduped) + ("\n" if deduped else ""), encoding="utf-8")
    yaml_payload = {
        "schema": BATCH_SCHEMA_V2,
        "version": 2,
        "kind": "ai_watchlist_batch",
        "category": "continuous_latest",
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
        "category": "continuous_latest",
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
