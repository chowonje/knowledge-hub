"""Web ingestion service (crawl -> local store -> optional vector index)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from knowledge_hub.core.config import Config
from knowledge_hub.core.database import SQLiteDatabase, VectorDatabase
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
from knowledge_hub.core.schema_validator import annotate_schema_errors


def make_web_note_id(url: str) -> str:
    digest = hashlib.sha1(url.strip().encode("utf-8")).hexdigest()[:16]
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


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


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
    normalized = re.sub(r"\s+", " ", (chunk_text or "").strip())
    if not normalized:
        return title

    first_sentence = normalized.split(". ")[0].strip()
    if len(first_sentence) > 180:
        first_sentence = f"{first_sentence[:177]}..."

    if section_title:
        if section_path and section_path != section_title:
            return f"[{section_path}] {first_sentence}"
        return f"[{section_title}] {first_sentence}"
    return f"[{title}] {first_sentence}"


def _chunk_text_with_offsets(
    text: str,
    title: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return []

    headings = _extract_headings(text)
    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()

        if end < len(text):
            for delimiter in ["\n\n", "\n", ". ", "? ", "! "]:
                last_delim = chunk.rfind(delimiter)
                if last_delim > chunk_size * 0.5:
                    chunk = chunk[: last_delim + len(delimiter)]
                    end = start + last_delim + len(delimiter)
                    break

        if chunk:
            section_title, section_path = _section_by_offset(headings, start)
            chunks.append(
                {
                    "text": chunk,
                    "chunk_index": chunk_index,
                    "start": start,
                    "end": end,
                    "section_title": section_title,
                    "section_path": section_path,
                    "summary": _build_context_summary(
                        title=title,
                        chunk_text=chunk,
                        section_title=section_title,
                        section_path=section_path,
                    ),
                }
            )

        start = max(0, end - overlap)
        chunk_index += 1

    for idx, item in enumerate(chunks):
        item["chunk_index"] = idx
    return chunks


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
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }


class WebIngestService:
    def __init__(self, config: Config):
        self.config = config

    def _crawl(self, urls: list[str], engine: str = "auto", timeout: int = 15, delay: float = 0.5) -> tuple[list[CrawlDocument], str, list[str]]:
        warnings: list[str] = []
        effective_engine = engine

        if engine == "auto":
            if is_crawl4ai_available():
                effective_engine = "crawl4ai"
            else:
                effective_engine = "basic"
                warnings.append("crawl4ai not installed; fallback basic crawler used")

        if effective_engine == "crawl4ai":
            docs = crawl_urls_with_crawl4ai(urls)
            return docs, effective_engine, warnings

        crawler = WebCrawler(timeout=timeout, delay=delay)
        basic_docs = crawler.crawl_urls(urls)
        docs = [
            CrawlDocument(
                url=item.url,
                title=item.title,
                content=item.content,
                markdown=item.content,
                description=item.description or "",
                author=item.author or "",
                engine="basic",
                ok=True,
                error="",
            )
            for item in basic_docs
        ]

        missing = set(urls) - {item.url for item in docs}
        for url in sorted(missing):
            docs.append(
                CrawlDocument(
                    url=url,
                    title=url,
                    content="",
                    markdown="",
                    engine="basic",
                    ok=False,
                    error="crawl failed",
                )
            )
        return docs, effective_engine, warnings

    def ingest_documents(
        self,
        docs: list[CrawlDocument],
        topic: str = "",
        index: bool = True,
        extract_concepts: bool = True,
        allow_external: bool = False,
        writeback: bool = False,
        concept_threshold: float = 0.78,
        relation_threshold: float = 0.75,
        run_id: str | None = None,
        emit_ontology_graph: bool = False,
        ontology_ttl_path: str | None = None,
        validate_ontology_graph: bool = False,
    ) -> WebIngestSummary:
        now = datetime.now(timezone.utc).isoformat()
        run_id = run_id or f"crawl_ingest_{uuid4().hex[:12]}"
        sqlite_db = SQLiteDatabase(self.config.sqlite_path)
        archive = _archive_dir(self.config)

        stored = 0
        failed: list[dict[str, str]] = []
        indexed_chunks = 0
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

        valid_docs = [doc for doc in docs if doc.ok and doc.content.strip()]
        invalid_docs = [doc for doc in docs if not (doc.ok and doc.content.strip())]
        stored_notes: list[dict[str, Any]] = []

        for doc in invalid_docs:
            failed.append({"url": doc.url, "error": doc.error or "empty content"})

        # 1) local note storage (P0 local-only)
        for doc in valid_docs:
            note_id = make_web_note_id(doc.url)
            filename = f"{note_id}-{_slugify(doc.title)}.md"
            file_path = archive / filename
            file_path.write_text(doc.markdown or doc.content, encoding="utf-8")

            metadata = {
                "url": doc.url,
                "description": doc.description,
                "author": doc.author,
                "topic": topic_safe,
                "crawl_engine": doc.engine,
                "fetched_at": doc.fetched_at,
                "crawl_run_id": run_id,
            }
            sqlite_db.upsert_note(
                note_id=note_id,
                title=doc.title,
                content=doc.content,
                file_path=str(file_path),
                source_type="web",
                para_category="resource",
                metadata=metadata,
            )
            stored_notes.append(
                {
                    "note_id": note_id,
                    "url": doc.url,
                    "title": doc.title,
                    "content": doc.content,
                    "file_path": str(file_path),
                }
            )
            stored += 1

        # 1.5) optional ontology extraction (rule-first + pending queue)
        if extract_concepts and stored_notes:
            extractor = WebOntologyExtractor(sqlite_db, self.config)
            ontology_summary = extractor.extract_from_notes(
                topic=topic_safe,
                note_rows=stored_notes,
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
                sqlite_db.merge_note_metadata(
                    note_id=note_id,
                    patch={
                        "ontology_run_id": run_id,
                        "ontology": {
                            "conceptsAccepted": int(note_summary.get("conceptsAccepted") or 0),
                            "relationsAccepted": int(note_summary.get("relationsAccepted") or 0),
                            "pendingCount": int(note_summary.get("pendingCount") or 0),
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
        if index and valid_docs:
            try:
                from knowledge_hub.providers.registry import get_embedder

                embed_cfg = self.config.get_provider_config(self.config.embedding_provider)
                embedder = get_embedder(self.config.embedding_provider, model=self.config.embedding_model, **embed_cfg)
                vector_db = VectorDatabase(self.config.vector_db_path, self.config.collection_name)

                docs_payload: list[str] = []
                em_meta: list[dict[str, Any]] = []
                ids: list[str] = []

                for doc in valid_docs:
                    note_id = make_web_note_id(doc.url)
                    chunks = _chunk_text_with_offsets(doc.content, title=doc.title)
                    for chunk in chunks:
                        chunk_text = chunk.get("text", "").strip()
                        if not chunk_text:
                            continue
                        chunk_index = int(chunk.get("chunk_index", 0))
                        docs_payload.append(
                            f"Title: {doc.title}\nURL: {doc.url}\nSection: {chunk.get('section_title', '')}\n\n{chunk_text}"
                        )
                        em_meta.append(
                            {
                                "title": doc.title,
                                "url": doc.url,
                                "source_type": "web",
                                "file_path": str(archive / f"{note_id}-{_slugify(doc.title)}.md"),
                                "topic": topic,
                                "chunk_index": chunk_index,
                                "chunk_size": len(chunk_text),
                                "section_title": chunk.get("section_title", ""),
                                "section_path": chunk.get("section_path", ""),
                                "contextual_summary": chunk.get(
                                    "summary",
                                    _build_context_summary(doc.title, chunk_text, "", ""),
                                ),
                            }
                        )
                        ids.append(f"{note_id}_{chunk_index}")

                if docs_payload:
                    embeddings = embedder.embed_batch(docs_payload, show_progress=False)
                    valid = [
                        (document, emb, meta, doc_id)
                        for document, emb, meta, doc_id in zip(docs_payload, embeddings, em_meta, ids)
                        if emb is not None
                    ]
                    if valid:
                        documents, embs, metadatas, doc_ids = zip(*valid)
                        vector_db.add_documents(
                            documents=list(documents),
                            embeddings=list(embs),
                            metadatas=list(metadatas),
                            ids=list(doc_ids),
                        )
                        indexed_chunks = len(documents)
            except Exception as error:
                failed.append({"url": "*index*", "error": f"indexing failed: {error}"})

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

        sqlite_db.close()

        return WebIngestSummary(
            run_id=run_id,
            requested=len(docs),
            crawled=len(valid_docs),
            stored=stored,
            indexed_chunks=indexed_chunks,
            failed=failed,
            engine=valid_docs[0].engine if valid_docs else "none",
            topic=topic_safe,
            warnings=[],
            status=status,
            ontology=ontology_summary,
            ontology_graph=ontology_graph_summary,
            writeback_paths=writeback_paths,
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
        concept_threshold: float = 0.78,
        relation_threshold: float = 0.75,
        emit_ontology_graph: bool = False,
        ontology_ttl_path: str | None = None,
        validate_ontology_graph: bool = False,
    ) -> dict[str, Any]:
        run_id = f"crawl_ingest_{uuid4().hex[:12]}"
        cleaned_urls = [url.strip() for url in urls if str(url).strip()]
        docs, effective_engine, warnings = self._crawl(
            cleaned_urls,
            engine=engine,
            timeout=timeout,
            delay=delay,
        )
        summary = self.ingest_documents(
            docs,
            topic=topic,
            index=index,
            extract_concepts=extract_concepts,
            allow_external=allow_external,
            writeback=writeback,
            concept_threshold=concept_threshold,
            relation_threshold=relation_threshold,
            run_id=run_id,
            emit_ontology_graph=emit_ontology_graph,
            ontology_ttl_path=ontology_ttl_path,
            validate_ontology_graph=validate_ontology_graph,
        )
        payload = summary.to_dict()
        payload["engine"] = effective_engine
        payload["warnings"] = warnings
        try:
            annotate_schema_errors(payload, "knowledge-hub.crawl.ingest.result.v1")
        except Exception:
            # schema validation is non-blocking in ingest flow
            pass
        return payload

    def list_pending(self, topic: str = "", limit: int = 50) -> dict[str, Any]:
        sqlite_db = SQLiteDatabase(self.config.sqlite_path)
        try:
            extractor = WebOntologyExtractor(sqlite_db, self.config)
            return extractor.list_pending(topic=topic, limit=limit)
        finally:
            sqlite_db.close()

    def apply_pending(self, pending_id: int) -> dict[str, Any]:
        sqlite_db = SQLiteDatabase(self.config.sqlite_path)
        try:
            extractor = WebOntologyExtractor(sqlite_db, self.config)
            return extractor.apply_pending(pending_id)
        finally:
            sqlite_db.close()

    def reject_pending(self, pending_id: int) -> dict[str, Any]:
        sqlite_db = SQLiteDatabase(self.config.sqlite_path)
        try:
            extractor = WebOntologyExtractor(sqlite_db, self.config)
            return extractor.reject_pending(pending_id)
        finally:
            sqlite_db.close()
