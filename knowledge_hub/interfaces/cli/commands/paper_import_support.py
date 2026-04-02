"""CSV-driven staged paper import helpers for the paper CLI."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from rich.console import Console

from knowledge_hub.document_memory import DocumentMemoryBuilder
from knowledge_hub.interfaces.cli.commands.paper_shared_runtime import (
    _build_embedder,
    _sqlite_db,
    _vector_db,
)
from knowledge_hub.papers.downloader import PaperDownloader, _safe_filename
from knowledge_hub.papers.memory_builder import PaperMemoryBuilder
from knowledge_hub.papers.memory_runtime import build_paper_memory_builder
from knowledge_hub.papers.url_resolver import resolve_url

console = Console()
ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")

STEP_ORDER = ["register", "download", "embed", "paper-memory", "document-memory"]
REQUIRED_COLUMNS = {"title", "source_url"}
OPTIONAL_COLUMNS = {"priority", "year", "bucket_ko", "theme_ko", "why_selected"}
MANIFEST_SCHEMA = "knowledge-hub.paper-import.manifest.v1"


@dataclass
class ImportRow:
    row_index: int
    title: str
    source_url: str
    priority: int | None
    raw: dict[str, str]

    @property
    def entry_id(self) -> str:
        digest = hashlib.sha1(f"{self.row_index}|{self.source_url}|{self.title}".encode("utf-8")).hexdigest()[:12]
        return f"row-{self.row_index}-{digest}"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_steps(raw_steps: str | None) -> list[str]:
    if not raw_steps or not str(raw_steps).strip():
        return list(STEP_ORDER)
    seen: set[str] = set()
    parsed: list[str] = []
    for token in str(raw_steps).split(","):
        step = token.strip().lower()
        if not step:
            continue
        if step not in STEP_ORDER:
            raise ValueError(f"unsupported step: {step}")
        if step not in seen:
            parsed.append(step)
            seen.add(step)
    return parsed or list(STEP_ORDER)


def default_manifest_path(csv_path: str | Path) -> Path:
    path = Path(csv_path).expanduser().resolve()
    return path.with_name(f"{path.name}.import-manifest.json")


def load_csv_rows(csv_path: str | Path) -> tuple[list[ImportRow], list[str]]:
    path = Path(csv_path).expanduser().resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = {str(item or "").strip() for item in (reader.fieldnames or []) if str(item or "").strip()}
        missing_columns = [name for name in sorted(REQUIRED_COLUMNS) if name not in fieldnames]
        if missing_columns:
            raise ValueError(f"missing required CSV columns: {', '.join(missing_columns)}")

        rows: list[ImportRow] = []
        warnings: list[str] = []
        for index, raw_row in enumerate(reader, start=2):
            cleaned = {str(key or "").strip(): str(value or "").strip() for key, value in dict(raw_row or {}).items()}
            title = cleaned.get("title", "").strip()
            source_url = cleaned.get("source_url", "").strip()
            if not title or not source_url:
                warnings.append(f"row {index}: missing title/source_url; skipped")
                rows.append(
                    ImportRow(
                        row_index=index,
                        title=title,
                        source_url=source_url,
                        priority=_parse_priority(cleaned.get("priority")),
                        raw=cleaned,
                    )
                )
                continue
            rows.append(
                ImportRow(
                    row_index=index,
                    title=title,
                    source_url=source_url,
                    priority=_parse_priority(cleaned.get("priority")),
                    raw=cleaned,
                )
            )
        return rows, warnings


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        return {
            "schema": MANIFEST_SCHEMA,
            "manifestPath": str(manifest_path),
            "entries": [],
            "updatedAt": now_iso(),
        }
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "schema": MANIFEST_SCHEMA,
            "manifestPath": str(manifest_path),
            "entries": [],
            "updatedAt": now_iso(),
        }
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("schema", MANIFEST_SCHEMA)
    payload["manifestPath"] = str(manifest_path)
    payload.setdefault("entries", [])
    payload.setdefault("updatedAt", now_iso())
    return payload


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    manifest_path = Path(path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["schema"] = MANIFEST_SCHEMA
    manifest["manifestPath"] = str(manifest_path)
    manifest["updatedAt"] = now_iso()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_import_csv(
    *,
    khub,
    csv_path: str,
    manifest_path: str,
    min_priority: int = 5,
    limit: int = 0,
    steps: list[str] | None = None,
    fail_fast: bool = False,
    document_memory_parser: str = "raw",
    rebuild_memory: bool = False,
) -> dict[str, Any]:
    selected_steps = list(steps or STEP_ORDER)
    rows, csv_warnings = load_csv_rows(csv_path)
    manifest = load_manifest(manifest_path)
    entries = _index_manifest_entries(manifest)
    sqlite_db = _sqlite_db(khub.config, khub=khub)

    selected_rows: list[ImportRow] = []
    filtered_count = 0
    skipped_invalid_count = 0
    limit_token = max(0, int(limit or 0))
    for row in rows:
        entry = _merge_entry_base(entries.get(row.entry_id), row=row)
        entries[row.entry_id] = entry
        if not row.title or not row.source_url:
            entry["status"] = "skipped"
            entry["error"] = "missing title/source_url"
            entry["updatedAt"] = now_iso()
            skipped_invalid_count += 1
            continue
        if row.priority is not None and row.priority < int(min_priority):
            entry["status"] = "filtered"
            entry["error"] = ""
            entry["updatedAt"] = now_iso()
            filtered_count += 1
            continue
        selected_rows.append(row)
        if limit_token > 0 and len(selected_rows) >= limit_token:
            break

    counts = {
        "selected": len(selected_rows),
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "partial": 0,
        "filtered": filtered_count,
        "invalid": skipped_invalid_count,
    }
    run_warnings = list(csv_warnings)
    stopped_early = False

    for index, row in enumerate(selected_rows, start=1):
        entry = entries[row.entry_id]
        outcome = _process_row(
            khub=khub,
            sqlite_db=sqlite_db,
            row=row,
            entry=entry,
            selected_steps=selected_steps,
            document_memory_parser=document_memory_parser,
            rebuild_memory=rebuild_memory,
        )
        if outcome["status"] == "completed":
            counts["completed"] += 1
            if not [step for step in outcome["executedSteps"] if step != "register"]:
                counts["skipped"] += 1
        elif outcome["status"] == "failed":
            counts["failed"] += 1
            run_warnings.append(f"row {row.row_index} failed at {outcome.get('failedStep')}: {outcome.get('error')}")
            if fail_fast:
                stopped_early = True
        else:
            counts["partial"] += 1
        save_manifest(manifest_path, _manifest_with_entries(manifest, entries))
        if stopped_early:
            break

    manifest = _manifest_with_entries(manifest, entries)
    save_manifest(manifest_path, manifest)
    status = "failed" if counts["failed"] else "ok"
    if stopped_early:
        status = "failed"
    return {
        "status": status,
        "csvPath": str(Path(csv_path).expanduser().resolve()),
        "manifestPath": str(Path(manifest_path).expanduser().resolve()),
        "steps": selected_steps,
        "documentMemoryParser": str(document_memory_parser),
        "rebuildMemory": bool(rebuild_memory),
        "counts": counts,
        "stoppedEarly": bool(stopped_early),
        "warnings": run_warnings,
        "items": [
            _result_item(entries[row.entry_id], selected_steps=selected_steps)
            for row in selected_rows
        ],
    }


def render_import_summary(payload: dict[str, Any]) -> None:
    counts = dict(payload.get("counts") or {})
    console.print("[bold]paper import-csv[/bold]")
    console.print(
        "  "
        f"completed={counts.get('completed', 0)} "
        f"skipped={counts.get('skipped', 0)} "
        f"failed={counts.get('failed', 0)} "
        f"filtered={counts.get('filtered', 0)} "
        f"invalid={counts.get('invalid', 0)}"
    )
    console.print(f"  manifest={payload.get('manifestPath')}")
    for item in list(payload.get("items") or [])[:20]:
        title = str(item.get("title") or "")
        paper_id = str(item.get("resolvedPaperId") or "")
        status = str(item.get("status") or "")
        console.print(f"  - [{status}] {paper_id or 'unresolved'} {title[:80]}")
        if item.get("failedStep"):
            console.print(f"    failedStep={item.get('failedStep')} error={item.get('error')}")
    for warning in list(payload.get("warnings") or [])[:20]:
        console.print(f"[yellow]{warning}[/yellow]")


def _parse_priority(raw: str | None) -> int | None:
    token = str(raw or "").strip()
    if not token:
        return None
    try:
        return int(token)
    except ValueError:
        return None


def _index_manifest_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in list(manifest.get("entries") or []):
        if not isinstance(item, dict):
            continue
        entry_id = str(item.get("entryId") or "").strip()
        if not entry_id:
            continue
        indexed[entry_id] = dict(item)
    return indexed


def _manifest_with_entries(manifest: dict[str, Any], entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = dict(manifest)
    payload["entries"] = [entries[key] for key in sorted(entries.keys(), key=lambda item: entries[item].get("rowIndex", 0))]
    payload["updatedAt"] = now_iso()
    return payload


def _merge_entry_base(existing: dict[str, Any] | None, *, row: ImportRow) -> dict[str, Any]:
    entry = dict(existing or {})
    entry["entryId"] = row.entry_id
    entry["rowIndex"] = int(row.row_index)
    entry["title"] = row.title
    entry["sourceUrl"] = row.source_url
    entry["priority"] = row.priority
    entry["curation"] = {
        "bucket_ko": row.raw.get("bucket_ko", ""),
        "theme_ko": row.raw.get("theme_ko", ""),
        "why_selected": row.raw.get("why_selected", ""),
        "year": row.raw.get("year", ""),
    }
    entry.setdefault("resolvedPaperId", "")
    entry.setdefault("resolvedSource", "")
    entry.setdefault("resolvedPdfUrl", "")
    entry.setdefault("status", "pending")
    entry.setdefault("completedSteps", [])
    entry.setdefault("executedSteps", [])
    entry.setdefault("failedStep", "")
    entry.setdefault("error", "")
    entry.setdefault("artifacts", {})
    entry.setdefault("updatedAt", now_iso())
    return entry


def _process_row(
    *,
    khub,
    sqlite_db,
    row: ImportRow,
    entry: dict[str, Any],
    selected_steps: list[str],
    document_memory_parser: str,
    rebuild_memory: bool,
) -> dict[str, Any]:
    completed_steps = set(str(step) for step in list(entry.get("completedSteps") or []))
    executed_steps: list[str] = []
    entry["failedStep"] = ""
    entry["error"] = ""

    resolved_paper_id = str(entry.get("resolvedPaperId") or "").strip()
    if resolved_paper_id:
        completed_steps.update(_actual_completed_steps(sqlite_db, paper_id=resolved_paper_id))
    completed_steps = _normalize_resume_steps(
        completed_steps=completed_steps,
        sqlite_db=sqlite_db,
        entry=entry,
        selected_steps=selected_steps,
        document_memory_parser=document_memory_parser,
        rebuild_memory=rebuild_memory,
    )

    for step in STEP_ORDER:
        if step not in selected_steps:
            continue
        if step in completed_steps:
            continue
        try:
            if step == "register":
                resolved_paper_id, step_executed = _run_register(sqlite_db=sqlite_db, row=row, entry=entry)
            elif step == "download":
                step_executed = _run_download(khub=khub, sqlite_db=sqlite_db, paper_id=resolved_paper_id, entry=entry)
            elif step == "embed":
                step_executed = _run_embed(khub=khub, sqlite_db=sqlite_db, paper_id=resolved_paper_id, entry=entry)
            elif step == "paper-memory":
                step_executed = _run_paper_memory(
                    khub=khub,
                    sqlite_db=sqlite_db,
                    paper_id=resolved_paper_id,
                    entry=entry,
                    rebuild_memory=rebuild_memory,
                )
            elif step == "document-memory":
                step_executed = _run_document_memory(
                    khub=khub,
                    sqlite_db=sqlite_db,
                    paper_id=resolved_paper_id,
                    entry=entry,
                    parser=document_memory_parser,
                    rebuild_memory=rebuild_memory,
                )
            completed_steps.add(step)
            if step_executed:
                executed_steps.append(step)
            entry["completedSteps"] = _ordered_steps(completed_steps)
            entry["executedSteps"] = list(executed_steps)
            entry["status"] = "running"
            entry["updatedAt"] = now_iso()
        except Exception as error:
            entry["status"] = "failed"
            entry["failedStep"] = step
            entry["error"] = str(error)
            entry["completedSteps"] = _ordered_steps(completed_steps)
            entry["executedSteps"] = list(executed_steps)
            entry["updatedAt"] = now_iso()
            return {
                "status": "failed",
                "executedSteps": executed_steps,
                "failedStep": step,
                "error": str(error),
            }

    final_completed = completed_steps | set(_actual_completed_steps(sqlite_db, paper_id=str(entry.get("resolvedPaperId") or "")))
    entry["completedSteps"] = _ordered_steps(final_completed)
    entry["executedSteps"] = list(executed_steps)
    entry["status"] = "completed" if all(step in final_completed for step in selected_steps) else "partial"
    entry["updatedAt"] = now_iso()
    return {"status": entry["status"], "executedSteps": executed_steps, "failedStep": "", "error": ""}


def _run_register(*, sqlite_db, row: ImportRow, entry: dict[str, Any]) -> tuple[str, bool]:
    paper = resolve_url(row.source_url)
    if paper is None:
        paper = _fallback_resolved_paper(row)
    if paper is None:
        raise RuntimeError("could not resolve paper from source_url")
    paper_id = str(paper.arxiv_id or "").strip() or _fallback_paper_id(row.title)
    entry["resolvedPaperId"] = paper_id
    entry["resolvedSource"] = str(getattr(paper, "source", "") or "")
    entry["resolvedPdfUrl"] = str(getattr(paper, "pdf_url", "") or "")
    existing = sqlite_db.get_paper(paper_id)
    if existing:
        entry["artifacts"] = _merge_artifacts(entry.get("artifacts"), existing)
        return paper_id, True
    sqlite_db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": paper.title,
            "authors": getattr(paper, "authors", ""),
            "year": getattr(paper, "year", None),
            "field": ", ".join(list(getattr(paper, "fields_of_study", []) or [])[:3]),
            "importance": 3,
            "notes": f"citations: {getattr(paper, 'citation_count', 0)}",
            "pdf_path": None,
            "text_path": None,
            "translated_path": None,
        }
    )
    stored = sqlite_db.get_paper(paper_id) or {}
    entry["artifacts"] = _merge_artifacts(entry.get("artifacts"), stored)
    return paper_id, True


def _fallback_resolved_paper(row: ImportRow) -> Any | None:
    source_url = str(row.source_url or "").strip()
    if "arxiv.org" not in source_url:
        return None
    match = ARXIV_ID_PATTERN.search(source_url)
    if not match:
        return None
    year_token = _parse_priority(row.raw.get("year"))
    arxiv_id = match.group(1)
    return type(
        "FallbackResolvedPaper",
        (),
        {
            "arxiv_id": arxiv_id,
            "title": row.title,
            "authors": "",
            "year": int(year_token or 0),
            "citation_count": 0,
            "fields_of_study": [],
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "source": "arxiv",
        },
    )()


def _run_download(*, khub, sqlite_db, paper_id: str, entry: dict[str, Any]) -> bool:
    token = str(paper_id or "").strip()
    if not token:
        raise RuntimeError("paper id unavailable before download")
    paper = sqlite_db.get_paper(token)
    if not paper:
        raise RuntimeError(f"paper not found: {token}")
    if _has_downloaded_artifacts(paper):
        entry["artifacts"] = _merge_artifacts(entry.get("artifacts"), paper)
        return False
    resolved_source = str(entry.get("resolvedSource") or "").strip().lower()
    resolved_pdf_url = str(entry.get("resolvedPdfUrl") or "").strip()
    if resolved_source and resolved_source != "arxiv":
        if not resolved_pdf_url:
            raise RuntimeError(f"resolved source '{resolved_source}' did not provide a downloadable pdf_url")
        result = _download_resolved_pdf(
            pdf_url=resolved_pdf_url,
            papers_dir=khub.config.papers_dir,
            paper_id=token,
            title=str(paper.get("title") or token),
        )
    else:
        downloader = PaperDownloader(khub.config.papers_dir)
        result = downloader.download_single(token, str(paper.get("title") or token))
    if not result.get("success"):
        raise RuntimeError(result.get("error") or f"download failed: {token}")
    sqlite_db.upsert_paper(
        {
            "arxiv_id": token,
            "title": paper.get("title") or token,
            "authors": paper.get("authors") or "",
            "year": paper.get("year") or 0,
            "field": paper.get("field") or "",
            "importance": paper.get("importance") or 3,
            "notes": paper.get("notes") or "",
            "pdf_path": result.get("pdf"),
            "text_path": result.get("text"),
            "translated_path": paper.get("translated_path"),
        }
    )
    updated = sqlite_db.get_paper(token) or {}
    entry["artifacts"] = _merge_artifacts(entry.get("artifacts"), updated)
    return True


def _run_embed(*, khub, sqlite_db, paper_id: str, entry: dict[str, Any]) -> bool:
    token = str(paper_id or "").strip()
    if not token:
        raise RuntimeError("paper id unavailable before embed")
    paper = sqlite_db.get_paper(token)
    if not paper:
        raise RuntimeError(f"paper not found: {token}")
    if bool(paper.get("indexed")):
        entry["artifacts"] = _merge_artifacts(entry.get("artifacts"), paper)
        return False
    text = f"Title: {paper.get('title') or token}"
    if paper.get("notes"):
        text += f"\n\n{paper['notes']}"
    embedder = _build_embedder(khub.config, khub=khub)
    emb = embedder.embed_text(text)
    vector_db = _vector_db(khub.config, khub=khub)
    vector_db.add_documents(
        documents=[text],
        embeddings=[emb],
        metadatas=[
            {
                "title": paper.get("title") or "",
                "arxiv_id": token,
                "source_type": "paper",
                "field": paper.get("field") or "",
                "chunk_index": 0,
            }
        ],
        ids=[f"paper_{token}_0"],
    )
    sqlite_db.conn.execute("UPDATE papers SET indexed = 1 WHERE arxiv_id = ?", (token,))
    sqlite_db.conn.commit()
    updated = sqlite_db.get_paper(token) or {}
    entry["artifacts"] = _merge_artifacts(entry.get("artifacts"), updated)
    return True


def _run_paper_memory(*, khub, sqlite_db, paper_id: str, entry: dict[str, Any], rebuild_memory: bool) -> bool:
    token = str(paper_id or "").strip()
    if not token:
        raise RuntimeError("paper id unavailable before paper-memory")
    getter = getattr(sqlite_db, "get_paper_memory_card", None)
    existing = getter(token) if callable(getter) else None
    if existing and not rebuild_memory:
        entry["artifacts"] = _merge_artifacts(entry.get("artifacts"), {"paper_memory_id": existing.get("memory_id")})
        return False
    row = build_paper_memory_builder(sqlite_db, config=khub.config).build_and_store(paper_id=token)
    entry["artifacts"] = _merge_artifacts(entry.get("artifacts"), {"paper_memory_id": row.get("memory_id")})
    return True


def _run_document_memory(*, khub, sqlite_db, paper_id: str, entry: dict[str, Any], parser: str, rebuild_memory: bool) -> bool:
    token = str(paper_id or "").strip()
    if not token:
        raise RuntimeError("paper id unavailable before document-memory")
    document_id = f"paper:{token}"
    summary_getter = getattr(sqlite_db, "get_document_memory_summary", None)
    existing = summary_getter(document_id) if callable(summary_getter) else None
    requested_parser = str(parser or "raw").strip().lower() or "raw"
    previous_parser = _document_memory_parser_token(existing=existing, entry=entry)
    parser_matches = _document_memory_parser_matches(previous_parser=previous_parser, requested_parser=requested_parser)
    if existing and not rebuild_memory and parser_matches:
        entry["artifacts"] = _merge_artifacts(
            entry.get("artifacts"),
            {
                "document_memory_id": document_id,
                "document_memory_parser": requested_parser,
            },
        )
        return False
    items = DocumentMemoryBuilder(sqlite_db, config=khub.config).build_and_store_paper(
        paper_id=token,
        paper_parser=requested_parser,
        refresh_parse=False,
        opendataloader_options={},
    )
    if not items:
        raise RuntimeError(f"document memory build produced no units: {token}")
    entry["artifacts"] = _merge_artifacts(
        entry.get("artifacts"),
        {
            "document_memory_id": document_id,
            "document_memory_parser": requested_parser,
        },
    )
    return True


def _download_resolved_pdf(*, pdf_url: str, papers_dir: str, paper_id: str, title: str) -> dict[str, Any]:
    target_dir = Path(str(papers_dir)).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = target_dir / f"{_safe_filename(title, paper_id)}.pdf"
    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()
    content_type = str(response.headers.get("Content-Type") or "").lower()
    if "application/pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
        raise RuntimeError(f"resolved pdf_url did not return a PDF: {pdf_url}")
    pdf_path.write_bytes(response.content)
    return {
        "arxiv_id": paper_id,
        "title": title,
        "pdf": str(pdf_path),
        "text": None,
        "success": True,
    }


def _has_downloaded_artifacts(paper: dict[str, Any]) -> bool:
    pdf_path = Path(str(paper.get("pdf_path") or "")).expanduser() if str(paper.get("pdf_path") or "").strip() else None
    text_path = Path(str(paper.get("text_path") or "")).expanduser() if str(paper.get("text_path") or "").strip() else None
    return bool((pdf_path and pdf_path.exists()) or (text_path and text_path.exists()))


def _actual_completed_steps(sqlite_db, *, paper_id: str) -> list[str]:
    token = str(paper_id or "").strip()
    if not token:
        return []
    paper = sqlite_db.get_paper(token)
    if not paper:
        return []
    completed = ["register"]
    if _has_downloaded_artifacts(paper):
        completed.append("download")
    if bool(paper.get("indexed")):
        completed.append("embed")
    getter = getattr(sqlite_db, "get_paper_memory_card", None)
    if callable(getter) and getter(token):
        completed.append("paper-memory")
    summary_getter = getattr(sqlite_db, "get_document_memory_summary", None)
    if callable(summary_getter) and summary_getter(f"paper:{token}"):
        completed.append("document-memory")
    return completed


def _fallback_paper_id(title: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(title or "").strip()).strip("_")
    token = token[:30] or "paper"
    digest = hashlib.sha1(str(title or "").encode("utf-8")).hexdigest()[:8]
    return f"{token}_{digest}"


def _ordered_steps(steps: set[str]) -> list[str]:
    return [step for step in STEP_ORDER if step in steps]


def _normalize_resume_steps(
    *,
    completed_steps: set[str],
    sqlite_db,
    entry: dict[str, Any],
    selected_steps: list[str],
    document_memory_parser: str,
    rebuild_memory: bool,
) -> set[str]:
    normalized = set(completed_steps)
    if rebuild_memory:
        normalized.discard("paper-memory")
        normalized.discard("document-memory")
        return normalized
    if "document-memory" not in selected_steps:
        return normalized
    existing_summary = None
    resolved_paper_id = str(entry.get("resolvedPaperId") or "").strip()
    if resolved_paper_id:
        summary_getter = getattr(sqlite_db, "get_document_memory_summary", None)
        if callable(summary_getter):
            existing_summary = summary_getter(f"paper:{resolved_paper_id}")
    previous_parser = _document_memory_parser_token(existing=existing_summary, entry=entry)
    requested_parser = str(document_memory_parser or "raw").strip().lower() or "raw"
    parser_matches = _document_memory_parser_matches(previous_parser=previous_parser, requested_parser=requested_parser)
    if not parser_matches:
        normalized.discard("document-memory")
    return normalized


def _document_memory_parser_token(*, existing: dict[str, Any] | None, entry: dict[str, Any]) -> str:
    provenance = dict((existing or {}).get("provenance") or {})
    provenance_parser = str(provenance.get("parser") or "").strip().lower()
    if provenance_parser:
        return provenance_parser
    artifacts = dict(entry.get("artifacts") or {})
    return str(artifacts.get("documentMemoryParser") or "").strip().lower()


def _document_memory_parser_matches(*, previous_parser: str, requested_parser: str) -> bool:
    return previous_parser == requested_parser or (
        requested_parser == "raw" and previous_parser in {"", "raw"}
    )


def _merge_artifacts(existing: Any, payload: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(existing or {})
    if not payload:
        return merged
    for key in ("pdf_path", "text_path", "translated_path"):
        value = str(payload.get(key) or "").strip()
        if value:
            camel = {
                "pdf_path": "pdfPath",
                "text_path": "textPath",
                "translated_path": "translatedPath",
            }[key]
            merged[camel] = value
    if payload.get("arxiv_id"):
        merged["paperId"] = str(payload.get("arxiv_id"))
    if payload.get("paper_memory_id"):
        merged["paperMemoryId"] = str(payload.get("paper_memory_id"))
    if payload.get("document_memory_id"):
        merged["documentMemoryId"] = str(payload.get("document_memory_id"))
    if payload.get("document_memory_parser"):
        merged["documentMemoryParser"] = str(payload.get("document_memory_parser"))
    return merged


def _result_item(entry: dict[str, Any], *, selected_steps: list[str]) -> dict[str, Any]:
    return {
        "entryId": entry.get("entryId"),
        "rowIndex": entry.get("rowIndex"),
        "title": entry.get("title"),
        "sourceUrl": entry.get("sourceUrl"),
        "priority": entry.get("priority"),
        "resolvedPaperId": entry.get("resolvedPaperId"),
        "status": entry.get("status"),
        "completedSteps": [step for step in list(entry.get("completedSteps") or []) if step in selected_steps],
        "executedSteps": [step for step in list(entry.get("executedSteps") or []) if step in selected_steps],
        "failedStep": entry.get("failedStep"),
        "error": entry.get("error"),
        "artifacts": dict(entry.get("artifacts") or {}),
        "updatedAt": entry.get("updatedAt"),
    }
