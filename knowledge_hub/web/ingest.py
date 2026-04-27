"""Web ingestion service (crawl -> local store -> optional vector index)."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.persistence import SQLiteDatabase, VectorDatabase
from knowledge_hub.infrastructure.persistence.stores.derivative_lifecycle import source_hash_from_content
from knowledge_hub.core.models import CrawlPipelineRunResult
from knowledge_hub.core.sanitizer import detect_p0, redact_p0
from knowledge_hub.infrastructure.providers import get_embedder
from knowledge_hub.learning.obsidian_writeback import (
    build_paths,
    resolve_vault_write_adapter,
    write_web_concepts,
    write_web_sources,
)
from knowledge_hub.web.ontology_graph import OntologyGraphResult, OntologyGraphService
from knowledge_hub.web.crawl4ai_adapter import (
    CrawlDocument,
    crawl_urls_with_crawl4ai,
    is_crawl4ai_available,
)
from knowledge_hub.web.crawler import WebCrawler
from knowledge_hub.web.ontology_extractor import WebOntologyExtractor
from knowledge_hub.web.quality import (
    assess_quality,
    canonicalize_url,
    evaluate_batch,
    evaluate_sample_gate,
)
from knowledge_hub.web.youtube_extractor import (
    extract_youtube_document,
    is_youtube_url,
    youtube_video_id_from_url,
)
from knowledge_hub.web import ingest_chunking as _ingest_chunking
from knowledge_hub.web import ingest_pipeline_support as _ingest_pipeline
from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.knowledge.entity_resolution import build_entity_merge_proposals_for_note


def make_web_note_id(url: str) -> str:
    canonical = canonicalize_url(url) or (url or "").strip()
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
    return f"web_{digest}"


def _slugify(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9가-힣]+", "-", lowered)
    lowered = re.sub(r"-+", "-", lowered).strip("-")
    return lowered or "untitled"


def _archive_dir(config: Config) -> Path:
    base = Path(config.sqlite_path).expanduser().resolve().parent
    path = base / "web_docs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _raw_archive_dir(config: Config, raw_dir: str | None = None) -> Path:
    if raw_dir:
        path = Path(raw_dir).expanduser().resolve()
    else:
        base = Path(config.sqlite_path).expanduser().resolve().parent
        path = base / "web_raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_datetime(value: str | None, *, default_to_now: bool = True) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return datetime.now(timezone.utc) if default_to_now else None
    try:
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return datetime.fromisoformat(raw)
    except Exception:
        return datetime.now(timezone.utc) if default_to_now else None


def _pipeline_day_parts(fetched_at: str | None) -> tuple[str, str, str]:
    dt = _to_datetime(fetched_at)
    assert dt is not None
    return dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")


def _fetched_at_bucket(fetched_at: str | None) -> str:
    dt = _to_datetime(fetched_at)
    assert dt is not None
    return dt.strftime("%Y-%m-%dT%H")


def _sha256_token(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _record_id_from(canonical_url: str, fetched_at: str | None, source: str) -> str:
    bucket = _fetched_at_bucket(fetched_at)
    return _sha256_token(f"{canonical_url}|{bucket}|{source}")


def _normalize_title_key(value: str) -> str:
    lowered = (value or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9가-힣]+", "-", lowered)
    lowered = re.sub(r"-+", "-", lowered).strip("-")
    return lowered


def _freshness_days(published_at: str | None, fetched_at: str | None) -> int | None:
    published_dt = _to_datetime(published_at, default_to_now=False)
    if published_dt is None:
        return None
    fetched_dt = _to_datetime(fetched_at)
    assert fetched_dt is not None
    delta = fetched_dt - published_dt
    return max(0, int(delta.total_seconds() // 86400))


def _queue_entity_resolution_for_note(
    sqlite_db: SQLiteDatabase,
    *,
    topic_slug: str,
    note_id: str,
    source_url: str,
) -> list[dict[str, Any]]:
    return build_entity_merge_proposals_for_note(
        sqlite_db,
        topic_slug=str(topic_slug or ""),
        note_id=str(note_id or ""),
        source_url=str(source_url or ""),
        fuzzy_threshold=0.94,
        max_candidates=6,
    )


def _build_source_item_index(source_items: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in source_items or []:
        if not isinstance(item, dict):
            continue
        for key in (str(item.get("url") or "").strip(), str(item.get("canonical_url") or "").strip()):
            if key:
                index[key] = dict(item)
    return index


_DOC_MEDIA_METADATA_KEYS = (
    "media_platform",
    "media_type",
    "video_id",
    "channel_name",
    "channel_id",
    "duration_sec",
    "language",
    "transcript_source",
    "transcript_segments",
    "chapters",
    "thumbnail_url",
)


def _doc_extra_metadata(doc: Any) -> dict[str, Any]:
    source_metadata = dict(getattr(doc, "source_metadata", None) or {})
    payload: dict[str, Any] = {}
    for key in (
        "source_name",
        "source_type",
        "source_vendor",
        "source_channel",
        "source_channel_type",
        "source_item_id",
        *_DOC_MEDIA_METADATA_KEYS,
    ):
        value = source_metadata.get(key)
        if value in (None, "", [], {}):
            continue
        payload[key] = value
    warnings = source_metadata.get("warnings")
    if isinstance(warnings, list):
        clean_warnings = [str(item).strip() for item in warnings if str(item).strip()]
        if clean_warnings:
            payload["warnings"] = clean_warnings
    return payload


def _infer_source_defaults(url: str, source_type: str) -> dict[str, Any]:
    parsed = urlsplit(str(url or "").strip())
    domain = (parsed.netloc or "").lower().strip()
    path = parsed.path or ""
    youtube_video_id = youtube_video_id_from_url(url)
    defaults = {
        "source_name": domain or "web",
        "source_type": source_type or "web",
        "source_vendor": domain.split(".")[0] if domain else "web",
        "source_channel": re.sub(r"[^a-z0-9]+", "_", domain) if domain else "web",
        "source_channel_type": "generic_web",
        "source_item_id": "",
    }
    if domain == "developers.openai.com":
        defaults.update(
            {
                "source_name": "OpenAI Developer Blog",
                "source_vendor": "openai",
                "source_channel": "openai_developer_blog",
                "source_channel_type": "official_blog",
                "source_type": "official_blog",
            }
        )
    elif domain == "openai.com" and path.startswith(("/news/", "/index/")):
        defaults.update(
            {
                "source_name": "OpenAI News",
                "source_vendor": "openai",
                "source_channel": "openai_news",
                "source_channel_type": "official_news_rss",
                "source_type": "official_blog_index",
            }
        )
    elif domain == "deepmind.google":
        defaults.update(
            {
                "source_name": "Google DeepMind Blog",
                "source_vendor": "google",
                "source_channel": "deepmind_blog",
                "source_channel_type": "official_blog",
                "source_type": "official_blog",
            }
        )
    elif domain == "research.google":
        defaults.update(
            {
                "source_name": "Google Research Blog",
                "source_vendor": "google",
                "source_channel": "google_research_blog",
                "source_channel_type": "research_blog",
                "source_type": "research_blog",
            }
        )
    elif domain == "www.anthropic.com":
        defaults.update(
            {
                "source_name": "Anthropic News",
                "source_vendor": "anthropic",
                "source_channel": "anthropic_news",
                "source_channel_type": "official_blog",
                "source_type": "official_blog",
            }
        )
    elif domain == "bair.berkeley.edu":
        defaults.update(
            {
                "source_name": "BAIR Blog",
                "source_vendor": "berkeley",
                "source_channel": "bair_blog",
                "source_channel_type": "research_blog",
                "source_type": "research_blog",
            }
        )
    elif domain == "aws.amazon.com":
        defaults.update(
            {
                "source_name": "AWS Machine Learning Blog",
                "source_vendor": "amazon",
                "source_channel": "aws_ml_blog",
                "source_channel_type": "official_blog",
                "source_type": "official_blog",
            }
        )
    elif domain == "huggingface.co" and path.startswith("/blog"):
        defaults.update(
            {
                "source_name": "Hugging Face Blog",
                "source_vendor": "huggingface",
                "source_channel": "huggingface_blog",
                "source_channel_type": "official_blog",
                "source_type": "official_blog",
            }
        )
    elif domain == "arxiv.org":
        defaults.update(
            {
                "source_name": "arXiv",
                "source_vendor": "arxiv",
                "source_channel": "arxiv",
                "source_channel_type": "paper_feed",
                "source_type": "paper_index",
            }
        )
        if "/cs.LG/" in path or path.startswith("/list/cs.LG"):
            defaults["source_name"] = "arXiv cs.LG"
            defaults["source_channel"] = "arxiv_cs_lg"
        elif "/cs.CL/" in path or path.startswith("/list/cs.CL"):
            defaults["source_name"] = "arXiv cs.CL"
            defaults["source_channel"] = "arxiv_cs_cl"
        paper_match = re.search(r"/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", path)
        if paper_match:
            defaults["source_item_id"] = paper_match.group(1)
    elif domain in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
        defaults.update(
            {
                "source_name": "YouTube",
                "source_type": "web",
                "source_vendor": "youtube",
                "source_channel": "youtube_video",
                "source_channel_type": "youtube_video",
                "source_item_id": youtube_video_id,
            }
        )
    return defaults


def _compute_metadata_quality(
    *,
    source_name: str,
    source_type: str,
    source_vendor: str,
    source_channel: str,
    source_channel_type: str,
    source_item_id: str,
    canonical_url: str,
    published_at: str,
    author: str,
    tags: list[str],
    precedence_used: dict[str, str],
    title_hint: str,
    crawl_title: str,
) -> dict[str, Any]:
    weighted_fields = [
        (bool(source_name), 1.0),
        (bool(source_type), 1.0),
        (bool(source_vendor), 1.0),
        (bool(source_channel), 1.0),
        (bool(source_channel_type), 1.0),
        (bool(canonical_url), 1.0),
        (bool(published_at), 1.0),
        (bool(title_hint or crawl_title), 1.0),
        (bool(source_item_id), 0.5),
        (bool(author), 0.5),
        (bool(tags), 0.5),
    ]
    actual = sum(weight for present, weight in weighted_fields if present)
    total = sum(weight for _, weight in weighted_fields) or 1.0
    flags: list[str] = []
    if not published_at:
        flags.append("missing_published_at")
    if not author:
        flags.append("missing_author")
    if not tags:
        flags.append("missing_tags")
    if title_hint and crawl_title and _normalize_title_key(title_hint) != _normalize_title_key(crawl_title):
        flags.append("title_hint_differs_from_crawl_title")
    return {
        "completeness": round(actual / total, 6),
        "consistency_flags": flags,
        "precedence_used": dict(precedence_used),
    }


def _resolve_source_context(
    *,
    source_url: str,
    canonical_url: str,
    source_type: str,
    fetched_at: str,
    doc: CrawlDocument,
    source_item: dict[str, Any] | None,
) -> dict[str, Any]:
    fallback = _infer_source_defaults(canonical_url or source_url, source_type)
    item = dict(source_item or {})
    identity = dict(item.get("identity") or {})
    provenance = dict(item.get("provenance") or {})
    metadata = dict(item.get("metadata") or {})
    resolution = dict(metadata.get("resolution") or {})
    doc_meta = dict(doc.source_metadata or {})

    source_name = str(item.get("source_name") or doc_meta.get("source_name") or fallback["source_name"]).strip()
    resolved_source_type = str(item.get("source_type") or doc_meta.get("source_type") or fallback["source_type"] or source_type).strip()
    source_vendor = str(identity.get("source_vendor") or doc_meta.get("source_vendor") or fallback["source_vendor"]).strip()
    source_channel = str(identity.get("source_channel") or doc_meta.get("source_channel") or fallback["source_channel"]).strip()
    source_channel_type = str(
        identity.get("source_channel_type")
        or doc_meta.get("source_channel_type")
        or fallback["source_channel_type"]
    ).strip()
    source_item_id = str(identity.get("source_item_id") or doc_meta.get("source_item_id") or fallback["source_item_id"]).strip()
    title_hint = str(item.get("title_hint") or doc_meta.get("title_hint") or "").strip()
    published_at = str(item.get("published_at") or doc.published_at or "").strip()
    author = str(item.get("author") or doc.author or "").strip()
    tags_value = item.get("tags")
    if not isinstance(tags_value, list):
        tags_value = doc.tags if isinstance(doc.tags, list) else []
    tags = [str(tag).strip() for tag in tags_value if str(tag).strip()]
    canonical_value = str(item.get("canonical_url") or canonical_url or source_url).strip()
    precedence_used = {
        "title": str(resolution.get("title") or ("source_feed_metadata" if title_hint else "generic_crawler_inference")),
        "published_at": str(
            resolution.get("published_at")
            or ("source_feed_metadata" if item.get("published_at") else ("page_embedded_metadata" if doc.published_at else ""))
        ),
        "author": str(
            resolution.get("author")
            or ("source_feed_metadata" if item.get("author") else ("page_embedded_metadata" if doc.author else ""))
        ),
        "tags": str(resolution.get("tags") or ("source_feed_metadata" if item.get("tags") else "")),
        "canonical_url": str(
            resolution.get("canonical_url")
            or ("source_feed_metadata" if item.get("canonical_url") else "normalized_url")
        ),
    }
    discovery = {
        "method": str(provenance.get("method") or "direct_url"),
        "origin_url": str(provenance.get("origin_url") or source_url),
        "entry_ref": str(provenance.get("entry_ref") or source_item_id or canonical_value),
        "discovered_at": str(provenance.get("discovered_at") or fetched_at),
        "rank": max(1, int(provenance.get("rank") or 1)),
    }
    freshness_days = _freshness_days(published_at, fetched_at)
    metadata_quality = _compute_metadata_quality(
        source_name=source_name,
        source_type=resolved_source_type,
        source_vendor=source_vendor,
        source_channel=source_channel,
        source_channel_type=source_channel_type,
        source_item_id=source_item_id,
        canonical_url=canonical_value,
        published_at=published_at,
        author=author,
        tags=tags,
        precedence_used=precedence_used,
        title_hint=title_hint,
        crawl_title=str(doc.title or ""),
    )
    return {
        "source_name": source_name,
        "source_type": resolved_source_type,
        "source_vendor": source_vendor,
        "source_channel": source_channel,
        "source_channel_type": source_channel_type,
        "source_item_id": source_item_id,
        "title_hint": title_hint,
        "published_at": published_at,
        "author": author,
        "tags": tags,
        "canonical_url": canonical_value,
        "freshness_days": freshness_days,
        "discovery": discovery,
        "metadata_quality": metadata_quality,
    }


def _validate_record_payload(payload: dict[str, Any], schema_id: str, *, strict: bool) -> None:
    result = annotate_schema_errors(payload, schema_id, strict=strict)
    if strict and not result.ok:
        message = "; ".join(result.errors or [f"schema validation failed: {schema_id}"])
        raise ValueError(message)


def _domain_from_url(url: str) -> str:
    token = str(url or "").strip()
    if not token:
        return ""
    try:
        return (urlsplit(token).netloc or "").lower().strip()
    except Exception:
        return ""


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    return _ingest_chunking.chunk_text(text, chunk_size=chunk_size, overlap=overlap)


def _extract_headings(text: str) -> list[tuple[int, list[str], str]]:
    headings: list[tuple[int, list[str], str]] = []
    stack: list[tuple[int, str]] = []

    for match in re.finditer(r"^(#{1,6})\s+(.+?)\s*$", text, flags=re.MULTILINE):
        level = len(match.group(1))
        heading = match.group(2).strip()

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        headings.append((match.start(), [item[1] for item in stack], heading))

    return headings


def _section_by_offset(headings: list[tuple[int, list[str], str]], offset: int) -> tuple[str, str]:
    section_title = ""
    section_path: list[str] = []
    for heading_offset, heading_path, heading in headings:
        if heading_offset <= offset:
            section_title = heading
            section_path = heading_path
        else:
            break

    return section_title, " > ".join(section_path)


def _build_context_summary(title: str, chunk_text: str, section_title: str, section_path: str) -> str:
    return _ingest_chunking.build_context_summary(title, chunk_text, section_title, section_path)


def _format_seconds_label(value: float | int | None) -> str:
    return _ingest_chunking.format_seconds_label(value)


def _format_time_window_label(start_sec: float | int | None, end_sec: float | int | None) -> str:
    return _ingest_chunking.format_time_window_label(start_sec, end_sec)


def _youtube_context_summary(
    *,
    chunk_text: str,
    chapter_title: str,
    start_sec: float | int | None,
    end_sec: float | int | None,
) -> str:
    return _ingest_chunking.youtube_context_summary(
        chunk_text=chunk_text,
        chapter_title=chapter_title,
        start_sec=start_sec,
        end_sec=end_sec,
    )


def _build_parent_metadata(
    source_type: str,
    document_id: str,
    title: str,
    section_title: str,
    section_path: str,
) -> dict[str, str]:
    return _ingest_chunking.build_parent_metadata(source_type, document_id, title, section_title, section_path)


def _resolve_youtube_chapter(
    chapters: list[dict[str, Any]],
    *,
    point_sec: float | int | None,
) -> tuple[int, str]:
    return _ingest_chunking.resolve_youtube_chapter(chapters, point_sec=point_sec)


def _build_youtube_chunks_from_segments(
    transcript_segments: list[dict[str, Any]],
    *,
    chapters: list[dict[str, Any]] | None,
    title: str,
    source_type: str = "web",
    document_id: str = "",
    target_chars: int = 1200,
    max_chars: int = 1400,
    target_seconds: int = 60,
    max_seconds: int = 90,
    overlap_segments: int = 1,
) -> list[dict[str, Any]]:
    return _ingest_chunking.build_youtube_chunks_from_segments(
        transcript_segments,
        chapters=chapters,
        title=title,
        source_type=source_type,
        document_id=document_id,
        target_chars=target_chars,
        max_chars=max_chars,
        target_seconds=target_seconds,
        max_seconds=max_seconds,
        overlap_segments=overlap_segments,
    )


def _resolve_web_chunks(
    row: dict[str, Any],
    *,
    title: str,
    source_type: str,
    document_id: str,
    content: str,
) -> list[dict[str, Any]]:
    return _ingest_chunking.resolve_web_chunks(
        row,
        title=title,
        source_type=source_type,
        document_id=document_id,
        content=content,
    )


def _chunk_text_with_offsets(
    text: str,
    title: str,
    source_type: str = "web",
    document_id: str = "",
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    return _ingest_chunking.chunk_text_with_offsets(
        text,
        title=title,
        source_type=source_type,
        document_id=document_id,
        chunk_size=chunk_size,
        overlap=overlap,
    )


@dataclass
class WebIngestSummary:
    run_id: str
    requested: int
    crawled: int
    stored: int
    indexed_chunks: int
    failed: list[dict[str, str]]
    engine: str
    topic: str
    warnings: list[str]
    status: str = "ok"
    schema: str = "knowledge-hub.crawl.ingest.result.v1"
    ontology: dict[str, Any] | None = None
    ontology_graph: dict[str, Any] | None = None
    writeback_paths: list[str] | None = None
    quality: dict[str, Any] | None = None
    index_diagnostics: dict[str, Any] | None = None
    raw_stored: int = 0
    raw_dir: str = ""
    unchanged: int = 0
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "runId": self.run_id,
            "status": self.status,
            "requested": self.requested,
            "crawled": self.crawled,
            "stored": self.stored,
            "indexedChunks": self.indexed_chunks,
            "failed": self.failed,
            "engine": self.engine,
            "topic": self.topic,
            "warnings": self.warnings,
            "ontology": self.ontology or {},
            "ontologyGraph": self.ontology_graph or {},
            "writebackPaths": self.writeback_paths or [],
            "quality": self.quality or {},
            "indexDiagnostics": self.index_diagnostics or {},
            "rawStored": self.raw_stored,
            "rawDir": self.raw_dir,
            "unchanged": self.unchanged,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }


class WebIngestService:
    def __init__(
        self,
        config: Config,
        *,
        sqlite_db: SQLiteDatabase | None = None,
        vector_db: VectorDatabase | None = None,
        embedder: Any | None = None,
        sqlite_db_factory: Callable[[], SQLiteDatabase] | None = None,
        vector_db_factory: Callable[[], VectorDatabase] | None = None,
        embedder_factory: Callable[[], Any] | None = None,
    ):
        self.config = config
        self._sqlite_db = sqlite_db
        self._vector_db = vector_db
        self._embedder = embedder
        self._sqlite_db_factory = sqlite_db_factory
        self._vector_db_factory = vector_db_factory
        self._embedder_factory = embedder_factory

    def _create_sqlite_db(self) -> SQLiteDatabase:
        if self._sqlite_db_factory is not None:
            return self._sqlite_db_factory()
        return SQLiteDatabase(self.config.sqlite_path)

    @contextmanager
    def _sqlite_session(self) -> Iterator[SQLiteDatabase]:
        if self._sqlite_db is not None:
            yield self._sqlite_db
            return

        sqlite_db = self._create_sqlite_db()
        try:
            yield sqlite_db
        finally:
            sqlite_db.close()

    def _create_vector_db(self) -> VectorDatabase:
        if self._vector_db is not None:
            return self._vector_db
        if self._vector_db_factory is not None:
            return self._vector_db_factory()
        return VectorDatabase(self.config.vector_db_path, self.config.collection_name)

    def _create_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder
        if self._embedder_factory is not None:
            return self._embedder_factory()
        embed_cfg = self.config.get_provider_config(self.config.embedding_provider)
        return get_embedder(self.config.embedding_provider, model=self.config.embedding_model, **embed_cfg)

    def _crawl(
        self,
        urls: list[str],
        engine: str = "auto",
        timeout: int = 15,
        delay: float = 0.5,
        input_source: str = "web",
        transcript_language: str | None = None,
        asr_model: str = "tiny",
    ) -> tuple[list[CrawlDocument], str, list[str]]:
        warnings: list[str] = []
        effective_engine = engine
        requested_input_source = str(input_source or "web").strip().lower()
        use_youtube_adapter = requested_input_source in {"youtube", "auto"}

        youtube_docs: dict[str, CrawlDocument] = {}
        generic_urls: list[str] = []
        youtube_used = False

        for raw_url in urls:
            source_url = str(raw_url or "").strip()
            if not source_url:
                continue
            if use_youtube_adapter and is_youtube_url(source_url):
                result = extract_youtube_document(
                    source_url,
                    timeout=timeout,
                    transcript_language=transcript_language,
                    asr_model=asr_model,
                )
                youtube_docs[source_url] = result.document
                warnings.extend(result.warnings)
                youtube_used = True
                continue
            if requested_input_source == "youtube":
                youtube_docs[source_url] = CrawlDocument(
                    url=source_url,
                    title=source_url,
                    content="",
                    markdown="",
                    source_metadata={},
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    engine="youtube",
                    ok=False,
                    error="not a youtube url",
                )
                warnings.append("youtube_ingest_blocked_no_text")
                youtube_used = True
                continue
            generic_urls.append(source_url)

        if not generic_urls:
            if youtube_used:
                return [youtube_docs[str(url).strip()] for url in urls if str(url).strip() in youtube_docs], "youtube", warnings
            return [], effective_engine, warnings

        if engine == "auto":
            if is_crawl4ai_available():
                effective_engine = "crawl4ai"
            else:
                effective_engine = "basic"
                warnings.append("crawl4ai not installed; fallback basic crawler used")

        if effective_engine == "crawl4ai":
            crawled_docs = crawl_urls_with_crawl4ai(generic_urls)
        else:
            crawler = WebCrawler(timeout=timeout, delay=delay)
            basic_docs = crawler.crawl_urls(generic_urls)
            crawled_docs = [
                CrawlDocument(
                    url=item.url,
                    title=item.title,
                    content=item.content,
                    markdown=item.content,
                    raw_html=item.raw_html or "",
                    description=item.description or "",
                    author=item.author or "",
                    published_at=item.published_at or "",
                    tags=list(item.tags or []),
                    source_metadata=dict(item.source_metadata or {}),
                    engine="basic",
                    ok=True,
                    error="",
                )
                for item in basic_docs
            ]

        generic_index = {item.url: item for item in crawled_docs}
        missing = set(generic_urls) - set(generic_index)
        for url in sorted(missing):
            generic_index[url] = CrawlDocument(
                url=url,
                title=url,
                content="",
                markdown="",
                engine=effective_engine,
                ok=False,
                error="crawl failed",
            )

        ordered_docs: list[CrawlDocument] = []
        for raw_url in urls:
            source_url = str(raw_url or "").strip()
            if not source_url:
                continue
            if source_url in youtube_docs:
                ordered_docs.append(youtube_docs[source_url])
            elif source_url in generic_index:
                ordered_docs.append(generic_index[source_url])

        if youtube_used and generic_urls:
            return ordered_docs, "mixed", warnings
        return ordered_docs, effective_engine, warnings

    def _index_web_records(
        self,
        records: list[dict[str, Any]],
        *,
        topic: str,
        archive: Path,
    ) -> tuple[int, str]:
        if not records:
            return 0, ""

        embedder = self._create_embedder()
        vector_db = self._create_vector_db()

        docs_payload: list[str] = []
        em_meta: list[dict[str, Any]] = []
        ids: list[str] = []

        for row in records:
            note_id = str(row.get("note_id") or "").strip()
            title = str(row.get("title") or "").strip() or note_id
            content = str(row.get("content") or "")
            canonical_url = str(row.get("url") or "").strip()
            source_hash = source_hash_from_content(
                content=content,
                metadata=dict(row or {}),
                identity=note_id or canonical_url,
            )
            quality_score = float(row.get("quality_score") or 0.0)
            source_name = str(row.get("source_name") or "").strip()
            source_vendor = str(row.get("source_vendor") or "").strip()
            source_channel = str(row.get("source_channel") or "").strip()
            source_channel_type = str(row.get("source_channel_type") or "").strip()
            source_item_id = str(row.get("source_item_id") or "").strip()
            published_at = str(row.get("published_at") or "").strip()
            freshness_days = row.get("freshness_days")
            tags = row.get("tags") if isinstance(row.get("tags"), list) else []
            media_platform = str(row.get("media_platform") or "").strip()
            video_id = str(row.get("video_id") or "").strip()
            channel_name = str(row.get("channel_name") or "").strip()
            transcript_source = str(row.get("transcript_source") or "").strip()
            if not note_id or not content.strip():
                continue
            delete_by_metadata = getattr(vector_db, "delete_by_metadata", None)
            if callable(delete_by_metadata):
                for document_id in (note_id, f"web:{note_id}"):
                    delete_by_metadata({"source_type": "web", "document_id": document_id})

            chunks = _resolve_web_chunks(
                row,
                title=title,
                source_type="web",
                document_id=note_id,
                content=content,
            )
            for chunk in chunks:
                chunk_text = chunk.get("text", "").strip()
                if not chunk_text:
                    continue
                chunk_index = int(chunk.get("chunk_index", 0))
                docs_payload.append(
                    f"Title: {title}\nURL: {canonical_url}\nSection: {chunk.get('section_title', '')}\n\n{chunk_text}"
                )
                em_meta.append(
                    {
                        "title": title,
                        "url": canonical_url,
                        "source_type": "web",
                        "file_path": str(archive / f"{note_id}-{_slugify(title)}.md"),
                        "topic": topic,
                        "source_content_hash": source_hash,
                        "stale": 0,
                        "chunk_index": chunk_index,
                        "chunk_size": len(chunk_text),
                        "quality_score": quality_score,
                        "document_id": chunk.get("document_id", f"web:{note_id}"),
                        "parent_id": chunk.get("parent_id", f"web:{note_id}::document:{title}"),
                        "parent_title": chunk.get("parent_title", title),
                        "parent_type": chunk.get("parent_type", "document"),
                        "section_title": chunk.get("section_title", ""),
                        "section_path": chunk.get("section_path", ""),
                        "source_name": source_name,
                        "source_vendor": source_vendor,
                        "source_channel": source_channel,
                        "source_channel_type": source_channel_type,
                        "source_item_id": source_item_id,
                        "published_at": published_at,
                        "freshness_days": freshness_days if freshness_days is not None else -1,
                        "tags": json.dumps(tags, ensure_ascii=False),
                        "media_platform": media_platform,
                        "video_id": video_id,
                        "channel_name": channel_name,
                        "transcript_source": transcript_source,
                        "start_sec": chunk.get("start_sec", -1.0),
                        "end_sec": chunk.get("end_sec", -1.0),
                        "timestamp_label": chunk.get("timestamp_label", ""),
                        "chapter_title": chunk.get("chapter_title", ""),
                        "chapter_index": int(chunk.get("chapter_index", -1)),
                        "contextual_summary": chunk.get(
                            "summary",
                            _build_context_summary(title, chunk_text, "", ""),
                        ),
                    }
                )
                ids.append(f"{note_id}_{chunk_index}")

        if not docs_payload:
            return 0, ""

        batch_size = max(4, int(self.config.get_nested("indexing", "embed_batch_size", default=24)))
        pause_ms = max(0, int(self.config.get_nested("indexing", "embed_pause_ms", default=50)))
        external_embed_provider = self.config.embedding_provider in {
            "openai",
            "openai-compat",
            "anthropic",
            "google",
        }

        total_indexed = 0
        for start in range(0, len(docs_payload), batch_size):
            end = min(len(docs_payload), start + batch_size)
            docs_batch = docs_payload[start:end]
            meta_batch = em_meta[start:end]
            ids_batch = ids[start:end]

            payload_batch = docs_batch
            if external_embed_provider:
                payload_batch = [
                    redact_p0(text) if detect_p0(text) else text
                    for text in docs_batch
                ]

            embeddings = embedder.embed_batch(payload_batch, show_progress=False)
            valid = [
                (document, emb, meta, doc_id)
                for document, emb, meta, doc_id in zip(docs_batch, embeddings, meta_batch, ids_batch)
                if emb is not None
            ]
            if not valid:
                continue

            documents, embs, metadatas, doc_ids = zip(*valid)
            vector_db.add_documents(
                documents=list(documents),
                embeddings=list(embs),
                metadatas=list(metadatas),
                ids=list(doc_ids),
            )
            total_indexed += len(documents)

            if pause_ms:
                time.sleep(pause_ms / 1000.0)
        return total_indexed, ""

    def _pipeline_root(self) -> Path:
        return _ingest_pipeline.pipeline_root(self)

    def _pipeline_profile(self, profile: str | None = None) -> tuple[str, dict[str, int]]:
        return _ingest_pipeline.pipeline_profile(self, profile=profile)

    def _resource_thresholds(self) -> tuple[float, float, float, float]:
        return _ingest_pipeline.resource_thresholds(self)

    def _sample_resource_ratio(self) -> tuple[float, float]:
        return _ingest_pipeline.sample_resource_ratio(self)

    def _domain_allowlist(self) -> set[str]:
        return _ingest_pipeline.domain_allowlist(self)

    def _is_domain_allowed(
        self,
        sqlite_db: SQLiteDatabase,
        *,
        domain: str,
        source_policy: str,
        allowlist: set[str],
    ) -> tuple[bool, str]:
        return _ingest_pipeline.is_domain_allowed(
            self,
            sqlite_db,
            domain=domain,
            source_policy=source_policy,
            allowlist=allowlist,
        )

    def _pipeline_paths(self, source: str, fetched_at: str | None, record_id: str) -> tuple[Path, Path, Path]:
        return _ingest_pipeline.pipeline_paths(
            self,
            source,
            fetched_at,
            record_id,
            day_parts_fn=_pipeline_day_parts,
        )

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _index_records_with_meta(
        self,
        records: list[dict[str, Any]],
        *,
        topic: str,
        archive: Path,
        embed_batch_size: int,
    ) -> dict[str, Any]:
        if not records:
            return {
                "indexed_chunks": 0,
                "vector_dim": 0,
                "provider": self.config.embedding_provider,
                "model": self.config.embedding_model,
                "retry_count": 0,
                "failures": [],
            }

        embedder = self._create_embedder()
        vector_db = self._create_vector_db()

        docs_payload: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []

        for row in records:
            note_id = str(row.get("note_id") or "").strip()
            title = str(row.get("title") or "").strip() or note_id
            content = str(row.get("content") or "")
            canonical_url = str(row.get("url") or "").strip()
            source_hash = source_hash_from_content(
                content=content,
                metadata=dict(row or {}),
                identity=note_id or canonical_url,
            )
            quality_score = float(row.get("quality_score") or 0.0)
            media_platform = str(row.get("media_platform") or "").strip()
            video_id = str(row.get("video_id") or "").strip()
            channel_name = str(row.get("channel_name") or "").strip()
            transcript_source = str(row.get("transcript_source") or "").strip()
            if not note_id or not content.strip():
                continue
            delete_by_metadata = getattr(vector_db, "delete_by_metadata", None)
            if callable(delete_by_metadata):
                for document_id in (note_id, f"web:{note_id}"):
                    delete_by_metadata({"source_type": "web", "document_id": document_id})

            chunks = _resolve_web_chunks(
                row,
                title=title,
                source_type="web",
                document_id=note_id,
                content=content,
            )
            for chunk in chunks:
                chunk_text = str(chunk.get("text") or "").strip()
                if not chunk_text:
                    continue
                chunk_index = int(chunk.get("chunk_index", 0))
                docs_payload.append(
                    f"Title: {title}\nURL: {canonical_url}\nSection: {chunk.get('section_title', '')}\n\n{chunk_text}"
                )
                metadatas.append(
                    {
                        "title": title,
                        "url": canonical_url,
                        "source_type": "web",
                        "file_path": str(archive / f"{note_id}-{_slugify(title)}.md"),
                        "topic": topic,
                        "source_content_hash": source_hash,
                        "stale": 0,
                        "chunk_index": chunk_index,
                        "chunk_size": len(chunk_text),
                        "quality_score": quality_score,
                        "document_id": chunk.get("document_id", f"web:{note_id}"),
                        "parent_id": chunk.get("parent_id", f"web:{note_id}::document:{title}"),
                        "parent_title": chunk.get("parent_title", title),
                        "parent_type": chunk.get("parent_type", "document"),
                        "section_title": chunk.get("section_title", ""),
                        "section_path": chunk.get("section_path", ""),
                        "media_platform": media_platform,
                        "video_id": video_id,
                        "channel_name": channel_name,
                        "transcript_source": transcript_source,
                        "start_sec": chunk.get("start_sec", -1.0),
                        "end_sec": chunk.get("end_sec", -1.0),
                        "timestamp_label": chunk.get("timestamp_label", ""),
                        "chapter_title": chunk.get("chapter_title", ""),
                        "chapter_index": int(chunk.get("chapter_index", -1)),
                        "contextual_summary": chunk.get(
                            "summary",
                            _build_context_summary(title, chunk_text, "", ""),
                        ),
                    }
                )
                ids.append(f"{note_id}_{chunk_index}")

        if not docs_payload:
            return {
                "indexed_chunks": 0,
                "vector_dim": 0,
                "provider": self.config.embedding_provider,
                "model": self.config.embedding_model,
                "retry_count": 0,
                "failures": [],
            }

        batch_size = max(1, int(embed_batch_size))
        total_indexed = 0
        vector_dim = 0
        failures: list[dict[str, Any]] = []
        retry_count = 0
        external_embed_provider = self.config.embedding_provider in {
            "openai",
            "openai-compat",
            "anthropic",
            "google",
        }

        for start in range(0, len(docs_payload), batch_size):
            end = min(len(docs_payload), start + batch_size)
            docs_batch = docs_payload[start:end]
            meta_batch = metadatas[start:end]
            ids_batch = ids[start:end]
            try:
                payload_batch = docs_batch
                if external_embed_provider:
                    payload_batch = [
                        redact_p0(text) if detect_p0(text) else text
                        for text in docs_batch
                    ]
                embeddings = embedder.embed_batch(payload_batch, show_progress=False)
            except Exception as error:
                failures.append(
                    {
                        "stage": "embed_batch",
                        "start": start,
                        "end": end,
                        "error": str(error),
                    }
                )
                continue

            status_fn = getattr(embedder, "get_last_status", None)
            if callable(status_fn):
                try:
                    embed_status = status_fn() or {}
                    retry_count += int(embed_status.get("retries", 0) or 0)
                    failures.extend(embed_status.get("failures", []) or [])
                except Exception:
                    pass

            valid = [
                (document, emb, meta, doc_id)
                for document, emb, meta, doc_id in zip(docs_batch, embeddings, meta_batch, ids_batch)
                if emb is not None
            ]
            if not valid:
                continue

            documents, embs, md, doc_ids = zip(*valid)
            if vector_dim <= 0 and embs:
                try:
                    vector_dim = len(embs[0])
                except Exception:
                    vector_dim = 0
            vector_db.add_documents(
                documents=list(documents),
                embeddings=list(embs),
                metadatas=list(md),
                ids=list(doc_ids),
            )
            total_indexed += len(documents)

        return {
            "indexed_chunks": total_indexed,
            "vector_dim": int(vector_dim),
            "provider": self.config.embedding_provider,
            "model": self.config.embedding_model,
            "retry_count": int(retry_count),
            "failures": failures,
        }

    def run_pipeline(
        self,
        *,
        urls: list[str],
        source_items: list[dict[str, Any]] | None = None,
        topic: str = "",
        source: str = "web",
        profile: str | None = None,
        source_policy: str | None = None,
        limit: int = 0,
        engine: str = "auto",
        timeout: int = 15,
        delay: float = 0.5,
        index: bool = True,
        extract_concepts: bool = True,
        allow_external: bool = False,
        input_source: str = "web",
        transcript_language: str | None = None,
        asr_model: str = "tiny",
        resume_job_id: str | None = None,
    ) -> dict[str, Any]:
        started_mono = time.monotonic()
        now = datetime.now(timezone.utc).isoformat()
        topic_safe = str(topic or "").strip()
        source_safe = str(source or "web").strip() or "web"
        profile_name, profile_cfg = self._pipeline_profile(profile)
        policy_name = str(
            source_policy
            or self.config.get_nested("pipeline", "source_policy", default="hybrid")
            or "hybrid"
        ).strip().lower()
        storage_root = str(self._pipeline_root())
        safe_urls = [str(url).strip() for url in urls if str(url).strip()]
        unique_urls: list[str] = []
        seen_urls: set[str] = set()
        for candidate in safe_urls:
            if candidate in seen_urls:
                continue
            seen_urls.add(candidate)
            unique_urls.append(candidate)
        if limit > 0:
            unique_urls = unique_urls[: max(1, int(limit))]
        source_item_index = _build_source_item_index(source_items)

        archive = _archive_dir(self.config)
        strict_schema = bool(self.config.get_nested("validation", "schema", "strict", default=False))
        allowlist = self._domain_allowlist()
        warnings: list[str] = []

        with self._sqlite_session() as sqlite_db:
            if resume_job_id:
                job = sqlite_db.get_crawl_pipeline_job(str(resume_job_id))
                if not job:
                    return {
                        "schema": "knowledge-hub.crawl.pipeline.run.result.v1",
                        "status": "failed",
                        "error": f"job not found: {resume_job_id}",
                        "runId": "",
                        "jobId": str(resume_job_id),
                        "profile": profile_name,
                        "sourcePolicy": policy_name,
                        "storageRoot": storage_root,
                        "requested": 0,
                        "processed": 0,
                        "normalized": 0,
                        "indexed": 0,
                        "pendingDomain": 0,
                        "failed": 1,
                        "skipped": 0,
                        "dedupeRate": 0.0,
                        "retryRate": 0.0,
                        "memoryPeakRatio": 0.0,
                        "recordsPerMin": 0.0,
                        "p50StepLatencyMs": 0.0,
                        "cursor": {},
                        "warnings": [],
                        "ts": now,
                    }
                job_id = str(job["job_id"])
                run_id = str(job.get("run_id") or f"crawl_pipeline_{uuid4().hex[:12]}")
                if not unique_urls:
                    sources = job.get("sources_json") if isinstance(job.get("sources_json"), list) else []
                    unique_urls = [str(item).strip() for item in sources if str(item).strip()]
                sqlite_db.update_crawl_pipeline_job(job_id, status="running")
            else:
                job_id = f"crawl_job_{uuid4().hex[:12]}"
                run_id = f"crawl_pipeline_{uuid4().hex[:12]}"
                sqlite_db.create_crawl_pipeline_job(
                    job_id=job_id,
                    run_id=run_id,
                    profile=profile_name,
                    source_policy=policy_name,
                    storage_root=storage_root,
                    source=source_safe,
                    topic=topic_safe,
                    sources=unique_urls,
                    status="running",
                )

            existing_records = sqlite_db.list_crawl_pipeline_records(job_id=job_id, limit=200000, offset=0)
            processed_record_ids = {
                str(item.get("record_id", ""))
                for item in existing_records
                if str(item.get("state", "")) in {"indexed", "skipped", "pending_domain"}
            }
            seen_canonical_hash = {
                str(item.get("canonical_url_hash", "")).strip()
                for item in existing_records
                if str(item.get("canonical_url_hash", "")).strip()
                and str(item.get("state", "")) in {"normalized", "indexed", "skipped"}
            }
            seen_content_hash = {
                str(item.get("content_sha256", "")).strip()
                for item in existing_records
                if str(item.get("content_sha256", "")).strip()
                and str(item.get("state", "")) in {"normalized", "indexed", "skipped"}
            }

            extractor = WebOntologyExtractor(sqlite_db, self.config) if extract_concepts else None
            accepted_concepts = 0
            accepted_relations = 0
            accepted_claims = 0
            pending_count = 0
            aliases_added = 0
            current_job = sqlite_db.get_crawl_pipeline_job(job_id)
            total_retry_count = int(current_job.get("retry_count", 0) or 0) if current_job else 0
            dedupe_count = int(current_job.get("dedupe_count", 0) or 0) if current_job else 0

            mem_high, cpu_high, backoff_base, backoff_max = self._resource_thresholds()
            backoff_round = 0

            requested_count = len(unique_urls)
            processed_count = 0
            normalized_count = 0
            indexed_count = 0
            pending_domain_count = 0
            failed_count = 0
            skipped_count = 0
            warned_pending_domains: set[str] = set()

            for cursor_index, source_url in enumerate(unique_urls):
                step_started = time.monotonic()
                memory_ratio, cpu_ratio = self._sample_resource_ratio()
                if memory_ratio >= mem_high or cpu_ratio >= cpu_high:
                    sleep_sec = min(backoff_max, backoff_base * (2 ** backoff_round))
                    backoff_round = min(backoff_round + 1, 12)
                    warnings.append(
                        f"resource backoff applied at cursor={cursor_index}: mem={memory_ratio:.2f}, cpu={cpu_ratio:.2f}, sleep={sleep_sec:.1f}s"
                    )
                    sqlite_db.append_crawl_pipeline_metric(
                        job_id,
                        phase="resource_backoff",
                        memory_ratio=memory_ratio,
                        cpu_ratio=cpu_ratio,
                        step_latency_ms=0.0,
                        retry_count=total_retry_count,
                        dedupe_count=dedupe_count,
                        details={"cursor": cursor_index, "sleepSec": sleep_sec},
                    )
                    time.sleep(sleep_sec)
                else:
                    backoff_round = 0

                canonical_url = canonicalize_url(source_url) or source_url
                domain = _domain_from_url(canonical_url)
                canonical_hash = _sha256_token(canonical_url)
                provisional_record_id = _record_id_from(canonical_url, now, source_safe)
                sqlite_db.upsert_crawl_pipeline_checkpoint(
                    job_id=job_id,
                    step="crawl",
                    cursor_value=str(cursor_index),
                    last_record_id=provisional_record_id,
                )

                allowed_domain, domain_reason = self._is_domain_allowed(
                    sqlite_db,
                    domain=domain,
                    source_policy=policy_name,
                    allowlist=allowlist,
                )
                if not allowed_domain:
                    pending_domain_count += 1
                    processed_count += 1
                    if domain and domain not in warned_pending_domains:
                        warned_pending_domains.add(domain)
                        if policy_name == "fixed":
                            warnings.append(
                                f"domain-policy gate (source-policy=fixed): domain '{domain}' is not in "
                                "pipeline.allowlist_domains — blocked before fetch (no raw/normalize/index for this URL). "
                                f"Add the domain to the allowlist, or use --source-policy hybrid|keyword and approve it; "
                                f"resume with crawl resume --job-id {job_id} after approval."
                            )
                        elif domain_reason == "rejected_domain":
                            warnings.append(
                                f"domain-policy gate: domain '{domain}' is rejected in crawl domain policy — "
                                "blocked before fetch (no raw/normalize/index). "
                                "Use crawl domain-policy approve to allow, or choose a different URL."
                            )
                        else:
                            warnings.append(
                                f"domain-policy gate (source-policy={policy_name}): domain '{domain}' is pending approval — "
                                "blocked before fetch (no raw/normalize/index). "
                                f"Run crawl domain-policy approve --domain {domain}, then crawl resume --job-id {job_id}."
                            )
                    sqlite_db.upsert_crawl_pipeline_record(
                        job_id=job_id,
                        record_id=provisional_record_id,
                        source=source_safe,
                        source_url=source_url,
                        canonical_url=canonical_url,
                        domain=domain,
                        canonical_url_hash=canonical_hash,
                        content_sha256="",
                        state="pending_domain",
                        retries=0,
                        error=domain_reason,
                    )
                    sqlite_db.append_crawl_pipeline_metric(
                        job_id,
                        phase="pending_domain",
                        memory_ratio=memory_ratio,
                        cpu_ratio=cpu_ratio,
                        step_latency_ms=(time.monotonic() - step_started) * 1000.0,
                        retry_count=total_retry_count,
                        dedupe_count=dedupe_count,
                        details={"url": canonical_url, "domain": domain, "reason": domain_reason},
                    )
                    continue

                if canonical_hash in seen_canonical_hash:
                    dedupe_count += 1
                    skipped_count += 1
                    processed_count += 1
                    sqlite_db.upsert_crawl_pipeline_record(
                        job_id=job_id,
                        record_id=provisional_record_id,
                        source=source_safe,
                        source_url=source_url,
                        canonical_url=canonical_url,
                        domain=domain,
                        canonical_url_hash=canonical_hash,
                        content_sha256="",
                        state="skipped",
                        retries=0,
                        error="duplicate_canonical_url",
                    )
                    sqlite_db.append_crawl_pipeline_metric(
                        job_id,
                        phase="dedupe",
                        memory_ratio=memory_ratio,
                        cpu_ratio=cpu_ratio,
                        step_latency_ms=(time.monotonic() - step_started) * 1000.0,
                        retry_count=total_retry_count,
                        dedupe_count=dedupe_count,
                        details={"url": canonical_url, "kind": "canonical_url_hash"},
                    )
                    continue

                docs, effective_engine, crawl_warnings = self._crawl(
                    [source_url],
                    engine=engine,
                    timeout=max(1, int(timeout)),
                    delay=max(0.0, float(delay)),
                    input_source=input_source,
                    transcript_language=transcript_language,
                    asr_model=asr_model,
                )
                warnings.extend(crawl_warnings or [])
                doc = next((item for item in docs if item.ok and str(item.content or "").strip()), None)
                if not doc:
                    failed_count += 1
                    processed_count += 1
                    sqlite_db.upsert_crawl_pipeline_record(
                        job_id=job_id,
                        record_id=provisional_record_id,
                        source=source_safe,
                        source_url=source_url,
                        canonical_url=canonical_url,
                        domain=domain,
                        canonical_url_hash=canonical_hash,
                        state="failed",
                        retries=0,
                        error="crawl_failed_or_empty",
                    )
                    sqlite_db.append_crawl_pipeline_metric(
                        job_id,
                        phase="failed",
                        memory_ratio=memory_ratio,
                        cpu_ratio=cpu_ratio,
                        step_latency_ms=(time.monotonic() - step_started) * 1000.0,
                        retry_count=total_retry_count,
                        dedupe_count=dedupe_count,
                        details={"url": canonical_url, "error": "crawl_failed_or_empty"},
                    )
                    continue

                fetched_at = str(doc.fetched_at or datetime.now(timezone.utc).isoformat())
                source_item = source_item_index.get(source_url) or source_item_index.get(canonical_url)
                source_context = _resolve_source_context(
                    source_url=source_url,
                    canonical_url=canonical_url,
                    source_type=source_safe,
                    fetched_at=fetched_at,
                    doc=doc,
                    source_item=source_item,
                )
                record_id = _record_id_from(canonical_url, fetched_at, source_safe)
                if record_id in processed_record_ids:
                    skipped_count += 1
                    processed_count += 1
                    sqlite_db.append_crawl_pipeline_metric(
                        job_id,
                        phase="resume_skip",
                        memory_ratio=memory_ratio,
                        cpu_ratio=cpu_ratio,
                        step_latency_ms=(time.monotonic() - step_started) * 1000.0,
                        retry_count=total_retry_count,
                        dedupe_count=dedupe_count,
                        details={"recordId": record_id, "reason": "already_processed"},
                    )
                    continue

                cleaned_content = str(doc.content or "").strip()
                content_sha = _sha256_token(cleaned_content)
                if content_sha in seen_content_hash:
                    dedupe_count += 1
                    skipped_count += 1
                    processed_count += 1
                    sqlite_db.upsert_crawl_pipeline_record(
                        job_id=job_id,
                        record_id=record_id,
                        source=source_safe,
                        source_url=source_url,
                        canonical_url=canonical_url,
                        domain=domain,
                        canonical_url_hash=canonical_hash,
                        content_sha256=content_sha,
                        state="skipped",
                        retries=0,
                        error="duplicate_content_hash",
                        fetched_at=fetched_at,
                    )
                    sqlite_db.append_crawl_pipeline_metric(
                        job_id,
                        phase="dedupe",
                        memory_ratio=memory_ratio,
                        cpu_ratio=cpu_ratio,
                        step_latency_ms=(time.monotonic() - step_started) * 1000.0,
                        retry_count=total_retry_count,
                        dedupe_count=dedupe_count,
                        details={"url": canonical_url, "kind": "content_sha256"},
                    )
                    continue

                raw_dir, normalized_path, indexed_path = self._pipeline_paths(source_safe, fetched_at, record_id)
                sqlite_db.upsert_crawl_pipeline_record(
                    job_id=job_id,
                    record_id=record_id,
                    source=source_safe,
                    source_url=source_url,
                    canonical_url=canonical_url,
                    domain=domain,
                    canonical_url_hash=canonical_hash,
                    content_sha256=content_sha,
                    state="downloading",
                    retries=0,
                    error="",
                    raw_path=str(raw_dir),
                    normalized_path=str(normalized_path),
                    indexed_path=str(indexed_path),
                    fetched_at=fetched_at,
                )

                raw_text = str(doc.raw_html or doc.markdown or doc.content or "")
                doc_extra_metadata = _doc_extra_metadata(doc)
                (raw_dir / "content.raw").write_text(raw_text, encoding="utf-8")
                self._write_json(
                    raw_dir / "metadata.json",
                    {
                        "schema": "knowledge-hub.normalized.web-record.v2",
                        "record_id": record_id,
                        "source": source_safe,
                        "url": source_url,
                        "canonical_url": canonical_url,
                        "domain": domain,
                        "fetched_at": fetched_at,
                        "crawl_engine": effective_engine,
                        "title": str(doc.title or ""),
                        "source_name": source_context["source_name"],
                        "source_type": source_context["source_type"],
                        "source_vendor": source_context["source_vendor"],
                        "source_channel": source_context["source_channel"],
                        "source_channel_type": source_context["source_channel_type"],
                        "source_item_id": source_context["source_item_id"],
                        "published_at": source_context["published_at"],
                        "author": source_context["author"],
                        "tags": source_context["tags"],
                        "freshness_days": source_context["freshness_days"],
                        "discovery": source_context["discovery"],
                        "metadata_quality": source_context["metadata_quality"],
                        **doc_extra_metadata,
                    },
                )

                quality = assess_quality(
                    cleaned_content,
                    threshold=float(self.config.get_nested("pipeline", "quality", "threshold", default=0.62) or 0.62),
                    min_tokens=int(self.config.get_nested("pipeline", "quality", "min_tokens", default=80) or 80),
                )
                normalized_payload = {
                    "schema": "knowledge-hub.normalized.web-record.v2",
                    "record_id": record_id,
                    "source": source_safe,
                    "url": source_url,
                    "canonical_url": canonical_url,
                    "domain": domain,
                    "fetched_at": fetched_at,
                    "content_sha256": content_sha,
                    "lang": "unknown",
                    "quality_score": float(quality.score),
                    "title_hint": source_context["title_hint"],
                    "title": str(doc.title or canonical_url),
                    "description": str(doc.description or ""),
                    "author": source_context["author"],
                    "published_at": source_context["published_at"],
                    "source_name": source_context["source_name"],
                    "source_type": source_context["source_type"],
                    "source_vendor": source_context["source_vendor"],
                    "source_channel": source_context["source_channel"],
                    "source_channel_type": source_context["source_channel_type"],
                    "source_item_id": source_context["source_item_id"],
                    "tags": source_context["tags"],
                    "freshness_days": source_context["freshness_days"],
                    "discovery": source_context["discovery"],
                    "metadata_quality": source_context["metadata_quality"],
                    "content_text": cleaned_content,
                    "crawl_engine": effective_engine,
                    "job_id": job_id,
                    "run_id": run_id,
                    **doc_extra_metadata,
                }
                _validate_record_payload(
                    normalized_payload,
                    "knowledge-hub.normalized.web-record.v2",
                    strict=strict_schema,
                )
                self._write_json(normalized_path, normalized_payload)
                sqlite_db.update_crawl_pipeline_record_state(
                    job_id,
                    record_id,
                    state="normalized",
                    normalized_path=str(normalized_path),
                    raw_path=str(raw_dir),
                    fetched_at=fetched_at,
                    content_sha256=content_sha,
                )
                normalized_count += 1
                processed_count += 1
                seen_canonical_hash.add(canonical_hash)
                seen_content_hash.add(content_sha)

                note_id = make_web_note_id(canonical_url)
                note_file = archive / f"{note_id}-{_slugify(str(doc.title or canonical_url))}.md"
                note_file.write_text(cleaned_content, encoding="utf-8")
                sqlite_db.upsert_note(
                    note_id=note_id,
                    title=str(doc.title or canonical_url),
                    content=cleaned_content,
                    file_path=str(note_file),
                    source_type="web",
                    para_category="resource",
                    metadata={
                        "url": canonical_url,
                        "url_original": source_url,
                        "topic": topic_safe,
                        "record_id": record_id,
                        "content_sha256": content_sha,
                        "source_content_hash": content_sha,
                        "crawl_run_id": run_id,
                        "crawl_job_id": job_id,
                        "source_name": source_context["source_name"],
                        "source_type": source_context["source_type"],
                        "source_vendor": source_context["source_vendor"],
                        "source_channel": source_context["source_channel"],
                        "source_channel_type": source_context["source_channel_type"],
                        "source_item_id": source_context["source_item_id"],
                        "published_at": source_context["published_at"],
                        "author": source_context["author"],
                        "tags": source_context["tags"],
                        "freshness_days": source_context["freshness_days"],
                        "discovery": source_context["discovery"],
                        "metadata_quality": source_context["metadata_quality"],
                        **doc_extra_metadata,
                    },
                )

                if extractor is not None:
                    try:
                        ontology_summary = extractor.extract_from_notes(
                            topic=topic_safe or "web",
                            note_rows=[
                                {
                                    "note_id": note_id,
                                    "url": canonical_url,
                                    "title": str(doc.title or ""),
                                    "content": cleaned_content,
                                    "file_path": str(note_file),
                                }
                            ],
                            run_id=run_id,
                            allow_external=bool(allow_external),
                            concept_threshold=0.78,
                            relation_threshold=0.75,
                            concept_pending_threshold=0.60,
                            relation_pending_threshold=0.55,
                        )
                        accepted_concepts += int(ontology_summary.get("conceptsAccepted", 0) or 0)
                        accepted_relations += int(ontology_summary.get("relationsAccepted", 0) or 0)
                        accepted_claims += int(ontology_summary.get("claimsAccepted", 0) or 0)
                        pending_count += int(ontology_summary.get("pendingCount", 0) or 0)
                        aliases_added += int(ontology_summary.get("aliasesAdded", 0) or 0)
                        merge_proposals = _queue_entity_resolution_for_note(
                            sqlite_db,
                            topic_slug=topic_safe,
                            note_id=note_id,
                            source_url=canonical_url,
                        )
                        sqlite_db.merge_note_metadata(
                            note_id=note_id,
                            patch={
                                "ontology_run_id": run_id,
                                "ontology": {
                                    "conceptsAccepted": int(ontology_summary.get("conceptsAccepted") or 0),
                                    "relationsAccepted": int(ontology_summary.get("relationsAccepted") or 0),
                                    "claimsAccepted": int(ontology_summary.get("claimsAccepted") or 0),
                                    "pendingCount": int(ontology_summary.get("pendingCount") or 0),
                                    "entityMergeProposals": len(merge_proposals),
                                },
                            },
                        )
                    except Exception as error:
                        warnings.append(f"ontology extraction failed for {canonical_url}: {error}")

                indexed_meta: dict[str, Any] = {
                    "indexed_chunks": 0,
                    "vector_dim": 0,
                    "provider": self.config.embedding_provider,
                    "model": self.config.embedding_model,
                    "retry_count": 0,
                    "failures": [],
                }
                if index:
                    indexed_meta = self._index_records_with_meta(
                        [
                            {
                                "note_id": note_id,
                                "url": canonical_url,
                                "title": str(doc.title or canonical_url),
                                "content": cleaned_content,
                                "quality_score": float(quality.score),
                                "source_content_hash": content_sha,
                                "source_name": source_context["source_name"],
                                "source_vendor": source_context["source_vendor"],
                                "source_channel": source_context["source_channel"],
                                "source_channel_type": source_context["source_channel_type"],
                                "source_item_id": source_context["source_item_id"],
                                "published_at": source_context["published_at"],
                                "freshness_days": source_context["freshness_days"],
                                "tags": source_context["tags"],
                            }
                        ],
                        topic=topic_safe,
                        archive=archive,
                        embed_batch_size=max(1, int(profile_cfg.get("embed_batch_size", 4))),
                    )
                    indexed_chunks = int(indexed_meta.get("indexed_chunks", 0) or 0)
                    indexed_count += indexed_chunks
                    total_retry_count += int(indexed_meta.get("retry_count", 0) or 0)

                indexed_payload = {
                    "schema": "knowledge-hub.indexed.web-record.v2",
                    "record_id": record_id,
                    "job_id": job_id,
                    "run_id": run_id,
                    "source": source_safe,
                    "url": source_url,
                    "canonical_url": canonical_url,
                    "domain": domain,
                    "fetched_at": fetched_at,
                    "source_name": source_context["source_name"],
                    "source_type": source_context["source_type"],
                    "source_vendor": source_context["source_vendor"],
                    "source_channel": source_context["source_channel"],
                    "source_channel_type": source_context["source_channel_type"],
                    "source_item_id": source_context["source_item_id"],
                    "published_at": source_context["published_at"],
                    "author": source_context["author"],
                    "tags": source_context["tags"],
                    "freshness_days": source_context["freshness_days"],
                    "discovery": source_context["discovery"],
                    "metadata_quality": source_context["metadata_quality"],
                    "content_sha256": content_sha,
                    "embedding_provider": indexed_meta.get("provider", self.config.embedding_provider),
                    "embedding_model": indexed_meta.get("model", self.config.embedding_model),
                    "vector_dim": int(indexed_meta.get("vector_dim", 0) or 0),
                    "embedded_at": datetime.now(timezone.utc).isoformat(),
                    "indexed_chunks": int(indexed_meta.get("indexed_chunks", 0) or 0),
                    "embedding_failures": indexed_meta.get("failures", []),
                }
                _validate_record_payload(
                    indexed_payload,
                    "knowledge-hub.indexed.web-record.v2",
                    strict=strict_schema,
                )
                self._write_json(indexed_path, indexed_payload)

                final_state = "indexed" if index else "normalized"
                sqlite_db.update_crawl_pipeline_record_state(
                    job_id,
                    record_id,
                    state=final_state,
                    indexed_path=str(indexed_path),
                )
                if final_state == "indexed":
                    processed_record_ids.add(record_id)

                sqlite_db.append_crawl_pipeline_metric(
                    job_id,
                    phase=final_state,
                    memory_ratio=memory_ratio,
                    cpu_ratio=cpu_ratio,
                    step_latency_ms=(time.monotonic() - step_started) * 1000.0,
                    retry_count=total_retry_count,
                    dedupe_count=dedupe_count,
                    details={
                        "recordId": record_id,
                        "url": canonical_url,
                        "state": final_state,
                        "qualityScore": float(quality.score),
                    },
                )

            counts = sqlite_db.count_crawl_pipeline_records(job_id)
            metrics = sqlite_db.list_crawl_pipeline_metrics(job_id, limit=50000)
            latencies = [
                float(item.get("step_latency_ms", 0.0) or 0.0)
                for item in metrics
                if float(item.get("step_latency_ms", 0.0) or 0.0) > 0.0
            ]
            p50_latency = float(statistics.median(latencies)) if latencies else 0.0
            memory_peak = 0.0
            for item in metrics:
                memory_peak = max(memory_peak, float(item.get("memory_ratio", 0.0) or 0.0))
            elapsed_sec = max(1e-6, time.monotonic() - started_mono)
            records_per_min = float(processed_count / max(1e-6, elapsed_sec / 60.0))
            dedupe_rate = float(dedupe_count / max(1, requested_count))
            retry_rate = float(total_retry_count / max(1, processed_count))

            checkpoints = sqlite_db.list_crawl_pipeline_checkpoints(job_id)
            cursor_payload = {
                "steps": [
                    {
                        "step": str(row.get("step", "")),
                        "cursor": str(row.get("cursor", "")),
                        "lastRecordId": str(row.get("last_record_id", "")),
                        "ts": str(row.get("ts", "")),
                    }
                    for row in checkpoints
                ],
                "next_record_ts": datetime.now(timezone.utc).isoformat(),
            }

            final_status = "completed"
            if counts.get("failed", 0) > 0 and counts.get("normalized", 0) == 0 and counts.get("indexed", 0) == 0:
                final_status = "failed"
            elif counts.get("failed", 0) > 0:
                final_status = "partial"

            sqlite_db.update_crawl_pipeline_job(
                job_id,
                status=final_status,
                requested_count=requested_count,
                processed_count=processed_count,
                normalized_count=normalized_count,
                indexed_count=indexed_count,
                pending_domain_count=pending_domain_count,
                failed_count=failed_count,
                skipped_count=skipped_count,
                retry_count=total_retry_count,
                dedupe_count=dedupe_count,
                warnings=warnings,
                finished=True,
            )

            result = CrawlPipelineRunResult(
                run_id=run_id,
                job_id=job_id,
                status=final_status,
                profile=profile_name,
                source_policy=policy_name,
                storage_root=storage_root,
                requested=requested_count,
                processed=processed_count,
                normalized=normalized_count,
                indexed=indexed_count,
                pending_domain=pending_domain_count,
                failed=failed_count,
                skipped=skipped_count,
                dedupe_rate=round(dedupe_rate, 6),
                retry_rate=round(retry_rate, 6),
                memory_peak_ratio=round(memory_peak, 6),
                records_per_min=round(records_per_min, 4),
                p50_step_latency_ms=round(p50_latency, 4),
                cursor=cursor_payload,
                warnings=warnings,
                ts=datetime.now(timezone.utc).isoformat(),
            ).to_dict()
            result["engine"] = engine
            result["counts"] = counts
            result["ontology"] = {
                "conceptsAccepted": accepted_concepts,
                "relationsAccepted": accepted_relations,
                "claimsAccepted": accepted_claims,
                "pendingCount": pending_count,
                "aliasesAdded": aliases_added,
            }

            try:
                schema_result = annotate_schema_errors(
                    result,
                    "knowledge-hub.crawl.pipeline.run.result.v1",
                    strict=strict_schema,
                )
                if strict_schema and not schema_result.ok:
                    result["status"] = "blocked"
                    result["error"] = "schema validation failed"
                    result["verify"] = {
                        "allowed": False,
                        "schemaValid": False,
                        "schemaErrors": list(schema_result.errors or []),
                    }
                    sqlite_db.update_crawl_pipeline_job(job_id, status="blocked", finished=True)
            except Exception as error:
                if strict_schema:
                    result["status"] = "blocked"
                    result["error"] = "schema validation failed"
                    result.setdefault("schemaErrors", [])
                    if isinstance(result.get("schemaErrors"), list):
                        result["schemaErrors"].append(str(error))
                    result["verify"] = {
                        "allowed": False,
                        "schemaValid": False,
                        "schemaErrors": list(result.get("schemaErrors") or []),
                    }
                    sqlite_db.update_crawl_pipeline_job(job_id, status="blocked", finished=True)

            return result

    def resume_pipeline(
        self,
        *,
        job_id: str,
        profile: str | None = None,
        source_policy: str | None = None,
        limit: int = 0,
        engine: str = "auto",
        timeout: int = 15,
        delay: float = 0.5,
        index: bool = True,
        extract_concepts: bool = True,
        allow_external: bool = False,
    ) -> dict[str, Any]:
        with self._sqlite_session() as sqlite_db:
            job = sqlite_db.get_crawl_pipeline_job(job_id)
            if not job:
                return {
                    "schema": "knowledge-hub.crawl.pipeline.run.result.v1",
                    "status": "failed",
                    "error": f"job not found: {job_id}",
                    "runId": "",
                    "jobId": job_id,
                    "profile": profile or "safe",
                    "sourcePolicy": source_policy or "hybrid",
                    "storageRoot": str(self._pipeline_root()),
                    "requested": 0,
                    "processed": 0,
                    "normalized": 0,
                    "indexed": 0,
                    "pendingDomain": 0,
                    "failed": 1,
                    "skipped": 0,
                    "dedupeRate": 0.0,
                    "retryRate": 0.0,
                    "memoryPeakRatio": 0.0,
                    "recordsPerMin": 0.0,
                    "p50StepLatencyMs": 0.0,
                    "cursor": {},
                    "warnings": [],
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            urls = [str(item).strip() for item in (job.get("sources_json") or []) if str(item).strip()]
            return self.run_pipeline(
                urls=urls,
                topic=str(job.get("topic") or ""),
                source=str(job.get("source") or "web"),
                profile=profile or str(job.get("profile") or "safe"),
                source_policy=source_policy or str(job.get("source_policy") or "hybrid"),
                limit=limit,
                engine=engine,
                timeout=timeout,
                delay=delay,
                index=index,
                extract_concepts=extract_concepts,
                allow_external=allow_external,
                resume_job_id=job_id,
            )

    def pipeline_status(self, job_id: str) -> dict[str, Any]:
        with self._sqlite_session() as sqlite_db:
            job = sqlite_db.get_crawl_pipeline_job(job_id)
            if not job:
                return {
                    "schema": "knowledge-hub.crawl.pipeline.status.result.v1",
                    "status": "failed",
                    "error": f"job not found: {job_id}",
                    "jobId": job_id,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            counts = sqlite_db.count_crawl_pipeline_records(job_id)
            checkpoints = sqlite_db.list_crawl_pipeline_checkpoints(job_id)
            metrics = sqlite_db.list_crawl_pipeline_metrics(job_id, limit=5000)
            memory_peak = 0.0
            for item in metrics:
                memory_peak = max(memory_peak, float(item.get("memory_ratio", 0.0) or 0.0))
            return {
                "schema": "knowledge-hub.crawl.pipeline.status.result.v1",
                "status": "ok",
                "jobId": job_id,
                "runId": str(job.get("run_id", "")),
                "jobStatus": str(job.get("status", "")),
                "profile": str(job.get("profile", "safe")),
                "sourcePolicy": str(job.get("source_policy", "hybrid")),
                "storageRoot": str(job.get("storage_root", "")),
                "topic": str(job.get("topic", "")),
                "counts": counts,
                "checkpoints": checkpoints,
                "memoryPeakRatio": round(memory_peak, 6),
                "warnings": list(job.get("warnings_json") or []),
                "startedAt": str(job.get("started_at", "")),
                "updatedAt": str(job.get("updated_at", "")),
                "finishedAt": str(job.get("finished_at", "") or ""),
                "ts": datetime.now(timezone.utc).isoformat(),
            }

    def benchmark_pipeline(
        self,
        *,
        urls: list[str],
        sample: int = 20,
        profile: str = "safe",
        source_policy: str = "hybrid",
        topic: str = "",
        engine: str = "auto",
    ) -> dict[str, Any]:
        sample_n = max(1, int(sample))
        unique_urls: list[str] = []
        seen = set()
        for item in urls:
            token = str(item).strip()
            if not token or token in seen:
                continue
            seen.add(token)
            unique_urls.append(token)
            if len(unique_urls) >= sample_n:
                break

        run_result = self.run_pipeline(
            urls=unique_urls,
            topic=topic,
            profile=profile,
            source_policy=source_policy,
            limit=sample_n,
            engine=engine,
            timeout=15,
            delay=0.5,
            index=False,
            extract_concepts=False,
            allow_external=False,
        )
        payload = {
            "schema": "knowledge-hub.crawl.benchmark.result.v1",
            "status": str(run_result.get("status", "ok")),
            "jobId": str(run_result.get("jobId", "")),
            "runId": str(run_result.get("runId", "")),
            "sample": sample_n,
            "profile": profile,
            "sourcePolicy": source_policy,
            "recordsPerMin": float(run_result.get("recordsPerMin", 0.0) or 0.0),
            "p50StepLatencyMs": float(run_result.get("p50StepLatencyMs", 0.0) or 0.0),
            "retryRate": float(run_result.get("retryRate", 0.0) or 0.0),
            "dedupeRate": float(run_result.get("dedupeRate", 0.0) or 0.0),
            "memoryPeakRatio": float(run_result.get("memoryPeakRatio", 0.0) or 0.0),
            "processed": int(run_result.get("processed", 0) or 0),
            "indexed": int(run_result.get("indexed", 0) or 0),
            "failed": int(run_result.get("failed", 0) or 0),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        strict_schema = bool(self.config.get_nested("validation", "schema", "strict", default=False))
        try:
            schema_result = annotate_schema_errors(
                payload,
                "knowledge-hub.crawl.benchmark.result.v1",
                strict=strict_schema,
            )
            if strict_schema and not schema_result.ok:
                payload["status"] = "blocked"
                payload["error"] = "schema validation failed"
                payload["verify"] = {
                    "allowed": False,
                    "schemaValid": False,
                    "schemaErrors": list(schema_result.errors or []),
                }
        except Exception as error:
            if strict_schema:
                payload["status"] = "blocked"
                payload["error"] = "schema validation failed"
                payload.setdefault("schemaErrors", [])
                if isinstance(payload.get("schemaErrors"), list):
                    payload["schemaErrors"].append(str(error))
                payload["verify"] = {
                    "allowed": False,
                    "schemaValid": False,
                    "schemaErrors": list(payload.get("schemaErrors") or []),
                }
        return payload

    def list_domain_policy(self, status: str = "", limit: int = 200) -> dict[str, Any]:
        with self._sqlite_session() as sqlite_db:
            items = sqlite_db.list_crawl_domain_policy(
                status=str(status or "").strip().lower() or None,
                limit=max(1, int(limit)),
            )
            payload = {
                "schema": "knowledge-hub.crawl.domain.policy.result.v1",
                "status": "ok",
                "count": len(items),
                "items": items,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            strict_schema = bool(self.config.get_nested("validation", "schema", "strict", default=False))
            try:
                schema_result = annotate_schema_errors(
                    payload,
                    "knowledge-hub.crawl.domain.policy.result.v1",
                    strict=strict_schema,
                )
                if strict_schema and not schema_result.ok:
                    payload["status"] = "blocked"
                    payload["error"] = "schema validation failed"
                    payload["verify"] = {
                        "allowed": False,
                        "schemaValid": False,
                        "schemaErrors": list(schema_result.errors or []),
                    }
            except Exception as error:
                if strict_schema:
                    payload["status"] = "blocked"
                    payload["error"] = "schema validation failed"
                    payload.setdefault("schemaErrors", [])
                    if isinstance(payload.get("schemaErrors"), list):
                        payload["schemaErrors"].append(str(error))
                    payload["verify"] = {
                        "allowed": False,
                        "schemaValid": False,
                        "schemaErrors": list(payload.get("schemaErrors") or []),
                    }
            return payload

    def apply_domain_policy(self, domain: str, reason: str = "") -> dict[str, Any]:
        with self._sqlite_session() as sqlite_db:
            item = sqlite_db.upsert_crawl_domain_policy(str(domain).strip(), "approved", reason=reason)
            payload = {
                "schema": "knowledge-hub.crawl.domain.policy.result.v1",
                "status": "ok",
                "action": "approve",
                "item": item,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            return payload

    def reject_domain_policy(self, domain: str, reason: str = "") -> dict[str, Any]:
        with self._sqlite_session() as sqlite_db:
            item = sqlite_db.upsert_crawl_domain_policy(str(domain).strip(), "rejected", reason=reason)
            payload = {
                "schema": "knowledge-hub.crawl.domain.policy.result.v1",
                "status": "ok",
                "action": "reject",
                "item": item,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            return payload

    def ingest_documents(
        self,
        docs: list[CrawlDocument],
        topic: str = "",
        index: bool = True,
        extract_concepts: bool = True,
        allow_external: bool = False,
        writeback: bool = False,
        save_raw: bool = False,
        raw_dir: str | None = None,
        concept_threshold: float = 0.78,
        relation_threshold: float = 0.75,
        run_id: str | None = None,
        emit_ontology_graph: bool = False,
        ontology_ttl_path: str | None = None,
        validate_ontology_graph: bool = False,
        quality_first: bool = False,
        quality_threshold: float = 0.62,
        quality_min_tokens: int = 80,
        quality_sample_size: int = 12,
        quality_sample_min_pass_rate: float = 0.70,
        incremental: bool = True,
        index_autofix_mode: str = "none",
    ) -> WebIngestSummary:
        now = datetime.now(timezone.utc).isoformat()
        run_id = run_id or f"crawl_ingest_{uuid4().hex[:12]}"
        archive = _archive_dir(self.config)
        raw_archive = _raw_archive_dir(self.config, raw_dir=raw_dir) if save_raw else None

        stored = 0
        unchanged = 0
        raw_stored = 0
        failed: list[dict[str, str]] = []
        warnings: list[str] = []
        indexed_chunks = 0
        index_diagnostics: dict[str, Any] = {
            "requested": bool(index),
            "attempted": False,
            "autoRetryEligible": False,
            "autoRetryAttempted": False,
            "initialIndexedChunks": 0,
            "finalIndexedChunks": 0,
            "status": "not_requested" if not index else "indexed",
            "reason": "index disabled" if not index else "",
            "warnings": [],
        }
        writeback_paths: list[str] = []
        ontology_summary: dict[str, Any] = {
            "runId": run_id,
            "conceptsAccepted": 0,
            "relationsAccepted": 0,
            "pendingCount": 0,
            "aliasesAdded": 0,
            "writebackPaths": [],
        }
        ontology_graph_summary: dict[str, Any] | None = None
        status = "ok"
        topic_safe = str(topic or "").strip()

        quality_threshold = max(0.0, min(1.0, float(quality_threshold)))
        quality_min_tokens = max(10, int(quality_min_tokens))
        quality_sample_size = max(1, int(quality_sample_size))
        quality_sample_min_pass_rate = max(0.0, min(1.0, float(quality_sample_min_pass_rate)))

        valid_docs = [doc for doc in docs if doc.ok and doc.content.strip()]
        invalid_docs = [doc for doc in docs if not (doc.ok and doc.content.strip())]
        stored_notes: list[dict[str, Any]] = []
        index_records: list[dict[str, Any]] = []

        for doc in invalid_docs:
            failed.append({"url": doc.url, "error": doc.error or "empty content"})

        quality_batch = evaluate_batch(
            valid_docs,
            threshold=quality_threshold,
            min_tokens=quality_min_tokens,
        )
        for duplicate in quality_batch.duplicates:
            failed.append(duplicate)

        sample_gate = evaluate_sample_gate(
            quality_batch,
            sample_size=quality_sample_size,
            min_pass_rate=quality_sample_min_pass_rate,
        )
        quality_gate_allowed = bool(sample_gate.get("allowed", True))
        if quality_first and not quality_gate_allowed:
            warnings.append("quality sample gate failed; ontology/index stages skipped")
            status = "partial"

        quality_summary: dict[str, Any] = {
            "enabled": True,
            "qualityFirst": bool(quality_first),
            "threshold": quality_threshold,
            "minTokens": quality_min_tokens,
            "evaluatedCount": quality_batch.evaluated,
            "approvedCount": quality_batch.approved,
            "rejectedCount": quality_batch.rejected,
            "duplicateCount": len(quality_batch.duplicates),
            "sampleGate": sample_gate,
            "gateAllowed": (quality_gate_allowed if quality_first else True),
        }

        with self._sqlite_session() as sqlite_db:
            # 1) local note storage (P0 local-only, dedupe + cleaned content)
            for quality_doc in quality_batch.items:
                doc = quality_doc.doc
                note_id = make_web_note_id(quality_doc.canonical_url)
                filename = f"{note_id}-{_slugify(doc.title)}.md"
                file_path = archive / filename
                clean_content = quality_doc.cleaned_content.strip() or str(doc.content or "").strip()
                doc_extra_metadata = _doc_extra_metadata(doc)
                if not clean_content:
                    failed.append({"url": doc.url, "error": "clean content empty"})
                    continue

                if incremental:
                    existing_note = sqlite_db.get_note(note_id)
                    if existing_note:
                        try:
                            existing_meta = json.loads(existing_note.get("metadata") or "{}")
                        except Exception:
                            existing_meta = {}
                        existing_hash = str(existing_meta.get("content_sha1", "")).strip()
                        if existing_hash and existing_hash == quality_doc.content_hash:
                            unchanged += 1
                            continue

                file_path.write_text(clean_content, encoding="utf-8")

                if save_raw and raw_archive is not None:
                    try:
                        raw_html_path = raw_archive / f"{note_id}.html"
                        raw_meta_path = raw_archive / f"{note_id}.json"
                        raw_html = (doc.raw_html or "").strip()
                        if not raw_html:
                            raw_html = doc.markdown or doc.content or ""
                        raw_html_path.write_text(raw_html, encoding="utf-8")
                        raw_meta = {
                            "note_id": note_id,
                            "url": quality_doc.canonical_url,
                            "url_original": doc.url,
                            "title": doc.title,
                            "description": doc.description or "",
                            "author": doc.author or "",
                            "topic": topic_safe,
                            "crawl_engine": doc.engine,
                            "fetched_at": doc.fetched_at,
                            "crawl_run_id": run_id,
                            "html_path": str(raw_html_path),
                            "markdown_path": str(file_path),
                            **doc_extra_metadata,
                        }
                        raw_meta_path.write_text(json.dumps(raw_meta, ensure_ascii=False, indent=2), encoding="utf-8")
                        raw_stored += 1
                    except Exception as error:
                        failed.append({"url": doc.url, "error": f"raw save failed: {error}"})

                metadata = {
                    "url": quality_doc.canonical_url,
                    "url_original": doc.url,
                    "description": doc.description,
                    "author": doc.author,
                    "topic": topic_safe,
                    "crawl_engine": doc.engine,
                    "fetched_at": doc.fetched_at,
                    "crawl_run_id": run_id,
                    "content_sha1": quality_doc.content_hash,
                    "source_content_hash": quality_doc.content_hash,
                    "quality": quality_doc.assessment.to_dict(),
                    **doc_extra_metadata,
                }
                sqlite_db.upsert_note(
                    note_id=note_id,
                    title=doc.title,
                    content=clean_content,
                    file_path=str(file_path),
                    source_type="web",
                    para_category="resource",
                    metadata=metadata,
                )
                stored_notes.append(
                    {
                        "note_id": note_id,
                        "url": quality_doc.canonical_url,
                        "title": doc.title,
                        "content": clean_content,
                        "file_path": str(file_path),
                        "qualityApproved": bool(quality_doc.assessment.approved),
                        "qualityScore": float(quality_doc.assessment.score),
                        "source_content_hash": quality_doc.content_hash,
                        **doc_extra_metadata,
                    }
                )
                index_records.append(
                    {
                        "note_id": note_id,
                        "url": quality_doc.canonical_url,
                        "title": doc.title,
                        "content": clean_content,
                        "quality_score": float(quality_doc.assessment.score),
                        "source_content_hash": quality_doc.content_hash,
                        **doc_extra_metadata,
                    }
                )
                stored += 1
                for warning in doc_extra_metadata.get("warnings") or []:
                    token = str(warning).strip()
                    if token:
                        warnings.append(token)

            quality_summary["storedCount"] = stored
            quality_summary["unchangedCount"] = unchanged

            ontology_candidates = stored_notes
            index_candidates = index_records
            if quality_first:
                ontology_candidates = [item for item in stored_notes if bool(item.get("qualityApproved"))]
                index_candidates = [
                    item
                    for item, note in zip(index_records, stored_notes)
                    if bool(note.get("qualityApproved"))
                ]
                if not quality_gate_allowed:
                    ontology_candidates = []
                    index_candidates = []

            # 1.5) optional ontology extraction (rule-first + pending queue)
            if extract_concepts and ontology_candidates:
                extractor = WebOntologyExtractor(sqlite_db, self.config)
                ontology_summary = extractor.extract_from_notes(
                    topic=topic_safe,
                    note_rows=ontology_candidates,
                    run_id=run_id,
                    allow_external=allow_external,
                    concept_threshold=max(0.0, min(1.0, float(concept_threshold))),
                    relation_threshold=max(0.0, min(1.0, float(relation_threshold))),
                    concept_pending_threshold=0.60,
                    relation_pending_threshold=0.55,
                )
                for note_summary in ontology_summary.get("noteSummaries") or []:
                    note_id = str(note_summary.get("noteId", "")).strip()
                    if not note_id:
                        continue
                    merge_proposals = _queue_entity_resolution_for_note(
                        sqlite_db,
                        topic_slug=topic_safe,
                        note_id=note_id,
                        source_url=str(note_summary.get("url") or ""),
                    )
                    sqlite_db.merge_note_metadata(
                        note_id=note_id,
                        patch={
                            "ontology_run_id": run_id,
                            "ontology": {
                                "conceptsAccepted": int(note_summary.get("conceptsAccepted") or 0),
                                "relationsAccepted": int(note_summary.get("relationsAccepted") or 0),
                                "claimsAccepted": int(note_summary.get("claimsAccepted") or 0),
                                "pendingCount": int(note_summary.get("pendingCount") or 0),
                                "entityMergeProposals": len(merge_proposals),
                            },
                        },
                    )

            # 1.6) optional ontology graph export (RDF/TTL)
            if emit_ontology_graph:
                try:
                    graph_service = OntologyGraphService(sqlite_db, self.config)
                    if extract_concepts:
                        graph_result: OntologyGraphResult = graph_service.export_from_summary(
                            ontology_summary=ontology_summary,
                            run_id=run_id,
                            output_path=ontology_ttl_path,
                            validate=validate_ontology_graph,
                        )
                    else:
                        graph_result = graph_service.export_from_db(
                            run_id=run_id,
                            source="web",
                            output_path=ontology_ttl_path,
                            validate=validate_ontology_graph,
                        )
                    ontology_graph_summary = graph_result.to_dict()
                    if graph_result.turtle_path:
                        writeback_paths.append(graph_result.turtle_path)
                except Exception as error:
                    ontology_graph_summary = {
                        "status": "error",
                        "error": str(error),
                        "runId": run_id,
                        "createdAt": datetime.now(timezone.utc).isoformat(),
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                    }
                    status = "partial"
                    failed.append({"url": "*ontology-graph*", "error": f"ontology graph export failed: {error}"})

        # 2) optional vector indexing
        if index and index_candidates:
            try:
                index_diagnostics["attempted"] = True
                indexed_chunks, _ = self._index_web_records(
                    index_candidates,
                    topic=topic_safe,
                    archive=archive,
                )
                index_diagnostics["initialIndexedChunks"] = indexed_chunks
                index_diagnostics["finalIndexedChunks"] = indexed_chunks
                if indexed_chunks > 0:
                    index_diagnostics["status"] = "indexed"
                    index_diagnostics["reason"] = "indexing completed"
                else:
                    index_diagnostics["reason"] = "initial indexing produced zero chunks"
            except Exception as error:
                failed.append({"url": "*index*", "error": f"indexing failed: {error}"})
                index_diagnostics["status"] = "index_failed"
                index_diagnostics["reason"] = str(error)
                index_diagnostics["warnings"] = ["indexing failed"]

        if index and not index_candidates and not index_diagnostics["reason"]:
            index_diagnostics["reason"] = "no eligible index candidates"

        auto_retry_eligible = bool(
            index
            and index_autofix_mode == "youtube_single_retry"
            and stored > 0
            and index_candidates
            and indexed_chunks == 0
            and not any(str(item.get("url") or "") == "*index*" for item in failed)
        )
        index_diagnostics["autoRetryEligible"] = auto_retry_eligible

        if auto_retry_eligible:
            index_diagnostics["autoRetryAttempted"] = True
            try:
                retry_indexed_chunks, _ = self._index_web_records(
                    index_candidates,
                    topic=topic_safe,
                    archive=archive,
                )
                indexed_chunks = retry_indexed_chunks
                index_diagnostics["finalIndexedChunks"] = indexed_chunks
                if indexed_chunks > 0:
                    index_diagnostics["status"] = "retry_succeeded"
                    index_diagnostics["reason"] = "index retry succeeded"
                    index_diagnostics["warnings"] = ["youtube_index_retry_succeeded"]
                    warnings.append("youtube_index_retry_succeeded")
                else:
                    index_diagnostics["status"] = "retry_exhausted"
                    index_diagnostics["reason"] = "index retry produced zero chunks"
                    index_diagnostics["warnings"] = ["youtube_index_retry_exhausted"]
                    warnings.append("youtube_index_retry_exhausted")
                    if status == "ok":
                        status = "partial"
            except Exception as error:
                failed.append({"url": "*index*", "error": f"index retry failed: {error}"})
                index_diagnostics["status"] = "index_failed"
                index_diagnostics["reason"] = f"index retry failed: {error}"
                index_diagnostics["warnings"] = ["indexing failed"]

        # 3) optional LearningHub writeback for web sources/concepts
        if writeback and topic_safe:
            if not self.config.vault_path:
                status = "partial"
                failed.append({"url": "*writeback*", "error": "vault_path not configured"})
            else:
                try:
                    paths = build_paths(self.config.vault_path, topic_safe)
                    backend = str(
                        self.config.get_nested("obsidian", "write_backend", default="filesystem") or "filesystem"
                    )
                    cli_binary = str(self.config.get_nested("obsidian", "cli_binary", default="obsidian") or "obsidian")
                    vault_name = str(self.config.get_nested("obsidian", "vault_name", default="") or "")
                    adapter = resolve_vault_write_adapter(
                        vault_path=self.config.vault_path,
                        backend=backend,
                        cli_binary=cli_binary,
                        vault_name=vault_name,
                    )
                    write_web_sources(
                        paths=paths,
                        topic=topic_safe,
                        run_summary={
                            "runId": run_id,
                            "requested": len(docs),
                            "crawled": len(valid_docs),
                            "stored": stored,
                            "failed": failed,
                            "engine": valid_docs[0].engine if valid_docs else "none",
                            "docs": stored_notes,
                        },
                        adapter=adapter,
                    )
                    write_web_concepts(
                        paths=paths,
                        topic=topic_safe,
                        ontology_summary=ontology_summary,
                        adapter=adapter,
                    )
                    writeback_paths = [
                        str(paths.web_sources_file),
                        str(paths.web_concepts_file),
                    ]
                    ontology_summary["writebackPaths"] = writeback_paths
                except Exception as error:
                    status = "partial"
                    failed.append({"url": "*writeback*", "error": f"writeback failed: {error}"})

        if failed and status == "ok":
            status = "partial"

        return WebIngestSummary(
            run_id=run_id,
            requested=len(docs),
            crawled=len(valid_docs),
            stored=stored,
            indexed_chunks=indexed_chunks,
            failed=failed,
            engine=valid_docs[0].engine if valid_docs else "none",
            topic=topic_safe,
            warnings=warnings,
            status=status,
            ontology=ontology_summary,
            ontology_graph=ontology_graph_summary,
            writeback_paths=writeback_paths,
            quality=quality_summary,
            index_diagnostics=index_diagnostics,
            raw_stored=raw_stored,
            raw_dir=str(raw_archive) if raw_archive else "",
            unchanged=unchanged,
            created_at=now,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def crawl_and_ingest(
        self,
        urls: list[str],
        topic: str = "",
        engine: str = "auto",
        timeout: int = 15,
        delay: float = 0.5,
        index: bool = True,
        extract_concepts: bool = True,
        allow_external: bool = False,
        writeback: bool = False,
        save_raw: bool = False,
        raw_dir: str | None = None,
        concept_threshold: float = 0.78,
        relation_threshold: float = 0.75,
        emit_ontology_graph: bool = False,
        ontology_ttl_path: str | None = None,
        validate_ontology_graph: bool = False,
        quality_first: bool = False,
        quality_threshold: float = 0.62,
        quality_min_tokens: int = 80,
        quality_sample_size: int = 12,
        quality_sample_min_pass_rate: float = 0.70,
        incremental: bool = True,
        input_source: str = "web",
        transcript_language: str | None = None,
        asr_model: str = "tiny",
        index_autofix_mode: str = "none",
    ) -> dict[str, Any]:
        run_id = f"crawl_ingest_{uuid4().hex[:12]}"
        cleaned_urls = [url.strip() for url in urls if str(url).strip()]
        docs, effective_engine, warnings = self._crawl(
            cleaned_urls,
            engine=engine,
            timeout=timeout,
            delay=delay,
            input_source=input_source,
            transcript_language=transcript_language,
            asr_model=asr_model,
        )
        summary = self.ingest_documents(
            docs,
            topic=topic,
            index=index,
            extract_concepts=extract_concepts,
            allow_external=allow_external,
            writeback=writeback,
            save_raw=save_raw,
            raw_dir=raw_dir,
            concept_threshold=concept_threshold,
            relation_threshold=relation_threshold,
            run_id=run_id,
            emit_ontology_graph=emit_ontology_graph,
            ontology_ttl_path=ontology_ttl_path,
            validate_ontology_graph=validate_ontology_graph,
            quality_first=quality_first,
            quality_threshold=quality_threshold,
            quality_min_tokens=quality_min_tokens,
            quality_sample_size=quality_sample_size,
            quality_sample_min_pass_rate=quality_sample_min_pass_rate,
            incremental=incremental,
            index_autofix_mode=index_autofix_mode,
        )
        payload = summary.to_dict()
        payload["engine"] = effective_engine
        payload["warnings"] = list(dict.fromkeys([*(warnings or []), *((summary.warnings or []))]))
        strict_schema = bool(self.config.get_nested("validation", "schema", "strict", default=False))
        try:
            schema_result = annotate_schema_errors(
                payload,
                "knowledge-hub.crawl.ingest.result.v1",
                strict=strict_schema,
            )
            if strict_schema and not schema_result.ok:
                payload["status"] = "blocked"
                payload["error"] = "schema validation failed"
                payload["verify"] = {
                    "allowed": False,
                    "schemaValid": False,
                    "schemaErrors": list(schema_result.errors or []),
                }
                return payload
        except Exception as error:
            if strict_schema:
                payload["status"] = "blocked"
                payload["error"] = "schema validation failed"
                payload.setdefault("schemaErrors", [])
                if isinstance(payload.get("schemaErrors"), list):
                    payload["schemaErrors"].append(str(error))
                payload["verify"] = {
                    "allowed": False,
                    "schemaValid": False,
                    "schemaErrors": list(payload.get("schemaErrors") or []),
                }
                return payload
            # schema validation is non-blocking in ingest flow
            pass
        return payload

    def reindex_approved(
        self,
        topic: str = "",
        limit: int = 0,
        include_unrated: bool = False,
        shard_index: int = 0,
        shard_total: int = 1,
    ) -> dict[str, Any]:
        archive = _archive_dir(self.config)
        scanned = 0
        selected = 0
        indexed_chunks = 0
        failed: list[dict[str, str]] = []
        topic_safe = str(topic or "").strip()
        shard_index_i = max(0, int(shard_index))
        shard_total_i = max(1, int(shard_total))
        if shard_index_i >= shard_total_i:
            return {
                "schema": "knowledge-hub.crawl.reindex-approved.result.v1",
                "status": "error",
                "topic": topic_safe,
                "scanned": 0,
                "selected": 0,
                "indexedChunks": 0,
                "includeUnrated": bool(include_unrated),
                "shardIndex": shard_index_i,
                "shardTotal": shard_total_i,
                "failed": [
                    {"url": "*reindex-approved*", "error": "shard-index must be in [0, shard-total)"},
                ],
                "ts": datetime.now(timezone.utc).isoformat(),
            }

        offset = 0
        page_size = 500
        selected_records: list[dict[str, Any]] = []

        with self._sqlite_session() as sqlite_db:
            while True:
                batch = sqlite_db.list_notes(source_type="web", limit=page_size, offset=offset)
                if not batch:
                    break
                current_offset = offset
                offset += len(batch)

                for local_index, note in enumerate(batch):
                    global_index = current_offset + local_index
                    if global_index % shard_total_i != shard_index_i:
                        continue
                    scanned += 1
                    if limit > 0 and selected >= int(limit):
                        break

                    metadata_raw = note.get("metadata")
                    if isinstance(metadata_raw, str):
                        try:
                            metadata = json.loads(metadata_raw or "{}")
                        except Exception:
                            metadata = {}
                    else:
                        metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

                    note_topic = str(metadata.get("topic", "")).strip()
                    if topic_safe and note_topic != topic_safe:
                        continue

                    quality = metadata.get("quality") if isinstance(metadata.get("quality"), dict) else {}
                    approved = bool(quality.get("approved")) if quality else False
                    if not approved and not include_unrated:
                        continue
                    if not approved and include_unrated and quality:
                        continue

                    selected_records.append(
                        {
                            "note_id": str(note.get("id") or ""),
                            "url": str(metadata.get("url") or ""),
                            "title": str(note.get("title") or ""),
                            "content": str(note.get("content") or ""),
                            "quality_score": float(quality.get("score") or 0.0),
                            "source_content_hash": str(metadata.get("source_content_hash") or metadata.get("content_sha1") or metadata.get("content_sha256") or ""),
                        }
                    )
                    selected += 1

                if limit > 0 and selected >= int(limit):
                    break

            if selected_records:
                try:
                    indexed_chunks, _ = self._index_web_records(
                        selected_records,
                        topic=topic_safe,
                        archive=archive,
                    )
                except Exception as error:
                    failed.append({"url": "*reindex-approved*", "error": str(error)})

            return {
                "schema": "knowledge-hub.crawl.reindex-approved.result.v1",
                "status": "partial" if failed else "ok",
                "topic": topic_safe,
                "scanned": scanned,
                "selected": selected,
                "indexedChunks": indexed_chunks,
                "shardIndex": shard_index_i,
                "shardTotal": shard_total_i,
                "includeUnrated": bool(include_unrated),
                "failed": failed,
                "ts": datetime.now(timezone.utc).isoformat(),
            }

    def list_pending(self, topic: str = "", limit: int = 50) -> dict[str, Any]:
        with self._sqlite_session() as sqlite_db:
            extractor = WebOntologyExtractor(sqlite_db, self.config)
            return extractor.list_pending(topic=topic, limit=limit)

    def apply_pending(self, pending_id: int) -> dict[str, Any]:
        with self._sqlite_session() as sqlite_db:
            extractor = WebOntologyExtractor(sqlite_db, self.config)
            return extractor.apply_pending(pending_id)

    def reject_pending(self, pending_id: int) -> dict[str, Any]:
        with self._sqlite_session() as sqlite_db:
            extractor = WebOntologyExtractor(sqlite_db, self.config)
            return extractor.reject_pending(pending_id)
