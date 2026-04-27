"""Result packet normalization for `khub add`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.infrastructure.persistence.stores.derivative_lifecycle import fallback_source_hash
from knowledge_hub.papers.source_text import source_hash_for_path
from knowledge_hub.web import make_web_note_id
from knowledge_hub.web.quality import canonicalize_url

from .route import AddRoute, _ARXIV_ID_RE, _is_url, _local_pdf_path

ADD_RESULT_SCHEMA = "knowledge-hub.add.result.v1"


def status_from_upstream(payload: dict[str, Any]) -> str:
    upstream_status = str(payload.get("status") or "").strip().lower()
    if upstream_status in {"ok", "completed"}:
        return "ok"
    if upstream_status:
        return upstream_status
    failed = payload.get("failed")
    if isinstance(failed, list) and failed:
        return "failed"
    return "ok"


def json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            value = json.loads(raw)
        except Exception:
            return {}
        return dict(value) if isinstance(value, dict) else {}
    return {}


def upstream_crawl_payload(upstream: dict[str, Any]) -> dict[str, Any]:
    nested = upstream.get("crawl")
    return dict(nested) if isinstance(nested, dict) else upstream


def first_item(upstream: dict[str, Any]) -> dict[str, Any]:
    for key in ("items", "ingested", "papers", "results", "docs", "records"):
        value = upstream.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return dict(value[0])
    nested = upstream.get("crawl")
    if isinstance(nested, dict):
        return first_item(nested)
    return {}


def warning_list(*payloads: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for payload in payloads:
        for warning in list((payload or {}).get("warnings") or []):
            token = str(warning or "").strip()
            if token and token not in warnings:
                warnings.append(token)
        for failed in list((payload or {}).get("failed") or []):
            if isinstance(failed, dict):
                error = str(failed.get("error") or "").strip()
                url = str(failed.get("url") or "").strip()
                token = f"{url}: {error}" if url and error else error or url
                if token and token not in warnings:
                    warnings.append(token)
    return warnings


def sqlite_db(khub):
    if hasattr(khub, "sqlite_db"):
        return khub.sqlite_db()
    return None


def redact_local_path(path_value: str) -> str:
    token = str(path_value or "").strip()
    if not token:
        return ""
    try:
        path = Path(token).expanduser().resolve()
        home = Path.home().resolve()
    except Exception:
        return token
    try:
        return f"~/{path.relative_to(home)}"
    except ValueError:
        return str(path)


def lookup_web_summary(khub, source: str) -> dict[str, str]:
    canonical = canonicalize_url(source) or str(source or "").strip()
    note_id = make_web_note_id(canonical)
    result = {
        "sourceId": note_id,
        "canonicalUrl": canonical,
        "canonicalPath": "",
        "title": "",
        "contentHash": "",
    }
    db = sqlite_db(khub)
    if db is None:
        return result
    getter = getattr(db, "get_note", None)
    if not callable(getter):
        return result
    try:
        note = getter(note_id)
    except Exception:
        return result
    if not note:
        return result
    metadata = json_dict(note.get("metadata"))
    result["sourceId"] = str(note.get("id") or note.get("note_id") or note_id)
    result["canonicalUrl"] = str(metadata.get("url") or canonical)
    result["title"] = str(note.get("title") or "")
    result["contentHash"] = str(
        metadata.get("source_content_hash")
        or metadata.get("content_sha1")
        or metadata.get("content_sha256")
        or ""
    )
    path = str(metadata.get("source_path") or "").strip()
    if path:
        result["canonicalPath"] = redact_local_path(path)
    return result


def paper_result_steps(upstream: dict[str, Any], source_id: str) -> list[Any]:
    results = upstream.get("results")
    if not isinstance(results, list):
        return []
    token = str(source_id or "").strip()
    for item in results:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("arxiv_id") or item.get("resolvedPaperId") or item.get("sourceId") or "").strip()
        if token and item_id and item_id != token:
            continue
        return list(item.get("steps") or [])
    return []


def paper_summary(khub, source: str, topic: str, upstream: dict[str, Any]) -> dict[str, Any]:
    item = first_item(upstream)
    artifacts = dict(item.get("artifacts") or {})
    source_id = str(
        item.get("resolvedPaperId")
        or artifacts.get("paperId")
        or item.get("arxiv_id")
        or item.get("sourceId")
        or ""
    ).strip()
    canonical = str(
        item.get("sourceUrl")
        or item.get("canonicalUrl")
        or item.get("url")
        or source
        or ""
    ).strip()
    title = str(item.get("title") or topic or canonical).strip()
    db = sqlite_db(khub)
    if source_id and db is not None and hasattr(db, "get_paper"):
        try:
            paper = db.get_paper(source_id)
        except Exception:
            paper = None
        if paper:
            artifacts = {
                **artifacts,
                "paperId": source_id,
                "pdfPath": artifacts.get("pdfPath") or paper.get("pdf_path") or "",
                "textPath": artifacts.get("textPath") or paper.get("text_path") or "",
                "translatedPath": artifacts.get("translatedPath") or paper.get("translated_path") or "",
            }
            title = title or str(paper.get("title") or "")
    content_hash = ""
    for key in ("textPath", "pdfPath"):
        content_hash = source_hash_for_path(str(artifacts.get(key) or ""))
        if content_hash:
            break
    if not content_hash:
        content_hash = str(artifacts.get("sourceContentHash") or artifacts.get("contentHash") or "").strip()
    if source_id and _ARXIV_ID_RE.match(source_id) and not _is_url(canonical):
        canonical = f"https://arxiv.org/abs/{source_id}"
    elif not _is_url(canonical):
        canonical = ""
    stored = (
        int((upstream.get("counts") or {}).get("completed") or 0) > 0
        or bool(upstream.get("ingested"))
        or bool(item.get("success"))
    )
    result_steps = paper_result_steps(upstream, source_id)
    return {
        "sourceId": source_id or fallback_source_hash(canonical, title)[:16],
        "canonicalUrl": canonical,
        "canonicalPath": redact_local_path(str(artifacts.get("textPath") or artifacts.get("pdfPath") or "")),
        "title": title,
        "contentHash": content_hash,
        "stored": stored,
        "indexed": "embed" in set(item.get("completedSteps") or item.get("executedSteps") or [])
        or any("인덱싱" in str(step) for step in result_steps),
    }


def web_summary(khub, route: AddRoute, source: str, upstream: dict[str, Any]) -> dict[str, Any]:
    local_path = _local_pdf_path(source)
    lookup_source = str(local_path.expanduser().resolve().as_uri()) if local_path is not None else source
    summary = lookup_web_summary(khub, lookup_source)
    if local_path is not None:
        summary["canonicalPath"] = redact_local_path(str(local_path.expanduser().resolve()))
    crawl = upstream_crawl_payload(upstream)
    indexed_chunks = int(crawl.get("indexedChunks") or crawl.get("indexed") or 0)
    stored_count = int(crawl.get("stored") or crawl.get("normalized") or crawl.get("processed") or 0)
    if not summary["title"] and route.source_type == "pdf":
        summary["title"] = local_path.stem if local_path is not None else Path(urlsplit(source).path).stem
    return {
        **summary,
        "stored": stored_count > 0 or bool(summary.get("contentHash")),
        "indexed": indexed_chunks > 0,
    }


def obsidian_stage_payload(*, requested: bool, upstream: dict[str, Any] | None = None) -> dict[str, Any]:
    if not requested:
        return {
            "requested": False,
            "status": "skipped",
            "runId": "",
            "staged": 0,
            "sourceGenerated": 0,
            "conceptGenerated": 0,
            "applied": 0,
            "applyRequested": False,
            "applySkipped": True,
        }
    payload = dict(upstream or {})
    materialize = dict(payload.get("materialize") or payload)
    has_materialize_payload = isinstance(payload.get("materialize"), dict) or str(materialize.get("schema") or "").startswith(
        "knowledge-hub.ko-note."
    )
    if not has_materialize_payload and not str(payload.get("runId") or "").strip():
        return {
            "requested": True,
            "status": "skipped",
            "runId": "",
            "staged": 0,
            "sourceGenerated": 0,
            "conceptGenerated": 0,
            "applied": 0,
            "applyRequested": False,
            "applySkipped": True,
        }
    staged = int(materialize.get("sourceGenerated") or 0) + int(materialize.get("conceptGenerated") or 0)
    return {
        "requested": True,
        "status": str(materialize.get("status") or payload.get("status") or "unknown"),
        "runId": str(materialize.get("runId") or payload.get("runId") or ""),
        "staged": staged,
        "sourceGenerated": int(materialize.get("sourceGenerated") or 0),
        "conceptGenerated": int(materialize.get("conceptGenerated") or 0),
        "applied": int((payload.get("apply") or {}).get("applied") or 0),
        "applyRequested": bool(payload.get("applyRequested", False)),
        "applySkipped": not bool(payload.get("applyRequested", False)),
    }


def next_actions(payload: dict[str, Any]) -> list[str]:
    source_id = str(payload.get("sourceId") or "").strip()
    actions = [
        'khub search "검색어"',
        'khub ask "질문"',
        "khub status",
    ]
    stage = dict(payload.get("obsidianStage") or {})
    if stage.get("requested") and stage.get("runId"):
        actions.insert(0, f"khub labs crawl ko-note-review-list --run-id {stage['runId']}")
    if source_id and payload.get("sourceType") == "paper":
        actions.insert(0, f"khub paper info {source_id}")
    return list(dict.fromkeys(actions))


def wrap_result(
    *,
    khub,
    route: AddRoute,
    source: str,
    topic: str,
    index: bool,
    upstream: dict[str, Any],
    to_obsidian: bool = False,
) -> dict[str, Any]:
    source_summary = (
        paper_summary(khub, source, topic, upstream)
        if route.source_type == "paper"
        else web_summary(khub, route, source, upstream)
    )
    warnings = warning_list(upstream, dict(upstream.get("crawl") or {}), dict(upstream.get("materialize") or {}))
    payload = {
        "schema": ADD_RESULT_SCHEMA,
        "status": status_from_upstream(upstream),
        "source": source,
        "sourceType": route.source_type,
        "sourceId": source_summary["sourceId"],
        "canonicalUrl": source_summary["canonicalUrl"],
        "canonicalPath": source_summary["canonicalPath"],
        "title": source_summary["title"],
        "contentHash": source_summary["contentHash"],
        "stored": bool(source_summary["stored"]),
        "indexed": bool(source_summary["indexed"]),
        "route": route.route,
        "routeReason": route.reason,
        "topic": topic,
        "index": bool(index),
        "obsidianStage": obsidian_stage_payload(requested=bool(to_obsidian), upstream=upstream),
        "warnings": warnings,
        "upstream": upstream,
    }
    payload["nextActions"] = next_actions(payload)
    payload["nextCommands"] = list(payload["nextActions"])
    annotate_schema_errors(payload, ADD_RESULT_SCHEMA)
    return payload
