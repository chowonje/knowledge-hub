"""User-facing dinger facade over the existing knowledge-hub engine surfaces."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import click
from click.testing import CliRunner
from rich.console import Console

from knowledge_hub.application.agent.foundry_bridge import run_foundry_project_cli
from knowledge_hub.application.dinger_capture_cleanup import cleanup_dinger_capture_runtime
from knowledge_hub.application.dinger_capture_processor import (
    DingerCaptureProcessor,
    resolve_capture_queue_dir,
    resolve_capture_runtime_dir,
)
from knowledge_hub.application.dinger_capture_recovery import assess_capture_requeue_recovery
from knowledge_hub.application.dinger_filing import file_dinger_request, resolve_dinger_filing_request_from_payload
from knowledge_hub.application.dinger_os_bridge import bridge_dinger_result_to_os_capture, is_capture_linked_to_os
from knowledge_hub.application.ko_note_reports import build_ko_note_report
from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.learning.obsidian_writeback import _upsert_marked_section, resolve_vault_write_adapter
from knowledge_hub.notes.templates import slugify_title, split_frontmatter, yaml_frontmatter

console = Console()
SOURCE_REF_PRIMARY_KEYS = ("paperId", "url", "noteId", "stableScopeId", "documentScopeId")
CAPTURE_INSPECTABLE_STATUSES = ("queued", "processing", "filed", "linked_to_os", "failed")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_cli_payload(config, payload: dict[str, Any], schema_id: str) -> None:
    strict = bool(config.get_nested("validation", "schema", "strict", default=False))
    result = annotate_schema_errors(payload, schema_id, strict=strict)
    if not result.ok:
        problems = ", ".join(result.errors[:5]) or "unknown schema validation error"
        raise click.ClickException(f"schema validation failed for {schema_id}: {problems}")


def _invoke_json_command(command, args: list[str], *, obj: dict[str, Any], label: str) -> dict[str, Any]:
    runner = CliRunner()
    result = runner.invoke(command, [*args, "--json"], obj=obj)
    if result.exit_code != 0:
        message = (result.output or "").strip()
        if result.exception is not None and not message:
            message = str(result.exception)
        raise click.ClickException(f"{label} failed: {message or 'unknown error'}")
    text = str(result.output or "").strip()
    if not text:
        raise click.ClickException(f"{label} failed: wrapped command returned no JSON payload")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise click.ClickException(f"{label} failed: {text}") from error
    if not isinstance(payload, dict):
        raise click.ClickException(f"{label} failed: expected object payload")
    return payload


def _failed_payload(*, schema_id: str, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": schema_id,
        "status": "failed",
        "error": str(message or "").strip() or "unknown error",
        "createdAt": _now_iso(),
        **extra,
    }


def _emit_or_raise_failure(*, khub, as_json: bool, schema_id: str, message: str, **extra: Any) -> None:
    payload = _failed_payload(schema_id=schema_id, message=message, **extra)
    _validate_cli_payload(khub.config, payload, schema_id)
    if as_json:
        console.print_json(data=payload)
        return
    raise click.ClickException(payload["error"])


def _paper_summary_status(khub, *, paper_id: str) -> str:
    from knowledge_hub.papers.public_surface import _inspect_public_summary_artifact

    state = dict(_inspect_public_summary_artifact(khub, paper_id=str(paper_id).strip()) or {})
    return str(state.get("status") or "missing")


def _load_recent_papers(sqlite_db, *, limit: int) -> list[dict[str, Any]]:
    conn = getattr(sqlite_db, "conn", None)
    if conn is not None:
        try:
            rows = conn.execute(
                """
                SELECT arxiv_id, title, year, field, created_at, pdf_path, translated_path, indexed
                FROM papers
                ORDER BY created_at DESC, arxiv_id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            pass
    list_papers = getattr(sqlite_db, "list_papers", None)
    if callable(list_papers):
        rows = list(list_papers(limit=max(1, int(limit) * 3)) or [])
        rows.sort(key=lambda row: (str(row.get("created_at") or ""), str(row.get("arxiv_id") or "")), reverse=True)
        return rows[: max(1, int(limit))]
    return []


def _load_recent_ko_note_runs(sqlite_db, *, limit: int) -> list[dict[str, Any]]:
    list_runs = getattr(sqlite_db, "list_ko_note_runs", None)
    if not callable(list_runs):
        return []
    rows = list(list_runs(limit=max(1, int(limit))) or [])
    rows.sort(key=lambda row: (str(row.get("updated_at") or ""), str(row.get("run_id") or "")), reverse=True)
    return rows[: max(1, int(limit))]


def _build_recent_payload(khub, *, limit: int) -> dict[str, Any]:
    sqlite_db = khub.sqlite_db()
    recent_papers = [
        {
            "paperId": str(row.get("arxiv_id") or ""),
            "title": str(row.get("title") or ""),
            "year": int(row.get("year") or 0) if str(row.get("year") or "").strip() else 0,
            "field": str(row.get("field") or ""),
            "createdAt": str(row.get("created_at") or ""),
            "hasPdf": bool(row.get("pdf_path")),
            "hasTranslation": bool(row.get("translated_path")),
            "indexed": bool(row.get("indexed")),
        }
        for row in _load_recent_papers(sqlite_db, limit=limit)
    ]
    recent_runs = [
        {
            "runId": str(row.get("run_id") or ""),
            "status": str(row.get("status") or ""),
            "sourceGenerated": int(row.get("source_generated") or 0),
            "conceptGenerated": int(row.get("concept_generated") or 0),
            "approvedCount": int(row.get("approved_count") or 0),
            "rejectedCount": int(row.get("rejected_count") or 0),
            "updatedAt": str(row.get("updated_at") or ""),
        }
        for row in _load_recent_ko_note_runs(sqlite_db, limit=limit)
    ]
    payload = {
        "schema": "knowledge-hub.dinger.recent.result.v1",
        "status": "ok",
        "recentPapers": recent_papers,
        "recentKoNoteRuns": recent_runs,
        "createdAt": _now_iso(),
    }
    _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.recent.result.v1")
    return payload


def _build_lint_payload(khub, *, paper_limit: int, recent_runs: int) -> dict[str, Any]:
    sqlite_db = khub.sqlite_db()
    papers = list(getattr(sqlite_db, "list_papers", lambda limit=paper_limit: [])(limit=max(1, int(paper_limit))) or [])
    latest_runs = _load_recent_ko_note_runs(sqlite_db, limit=recent_runs)
    latest_run = latest_runs[0] if latest_runs else {}
    latest_report = (
        build_ko_note_report(sqlite_db, run_id=str(latest_run.get("run_id") or ""), recent_runs=max(1, int(recent_runs)))
        if latest_run
        else {}
    )
    list_cards = getattr(sqlite_db, "list_paper_memory_cards", None)
    card_rows = list(list_cards(limit=max(50, int(paper_limit) * 3)) or []) if callable(list_cards) else []
    card_ids = {
        str(row.get("paper_id") or row.get("paperId") or "").strip()
        for row in card_rows
        if str(row.get("paper_id") or row.get("paperId") or "").strip()
    }
    summary_statuses = {
        str(row.get("arxiv_id") or "").strip(): _paper_summary_status(khub, paper_id=str(row.get("arxiv_id") or "").strip())
        for row in papers
        if str(row.get("arxiv_id") or "").strip()
    }
    missing_summary = [
        row
        for row in papers
        if summary_statuses.get(str(row.get("arxiv_id") or "").strip(), "missing") in {"missing", "degraded"}
    ]
    not_indexed = [row for row in papers if not bool(row.get("indexed"))]
    missing_memory = [row for row in papers if str(row.get("arxiv_id") or "").strip() not in card_ids]
    checks = {
        "reviewQueued": int((((latest_report.get("run") or {}).get("reviewQueue") or {}).get("combined") or {}).get("total") or 0),
        "alertCount": len(list(latest_report.get("alerts") or [])),
        "papersMissingSummary": len(missing_summary),
        "papersNotIndexed": len(not_indexed),
        "papersMissingMemoryCard": len(missing_memory),
    }
    payload = {
        "schema": "knowledge-hub.dinger.lint.result.v1",
        "status": "ok",
        "checks": checks,
        "latestKoNoteRun": {
            "runId": str(latest_run.get("run_id") or ""),
            "status": str(latest_run.get("status") or ""),
            "updatedAt": str(latest_run.get("updated_at") or ""),
            "warnings": list(latest_report.get("warnings") or [])[:10],
        },
        "samples": {
            "paperIdsMissingSummary": [str(row.get("arxiv_id") or "") for row in missing_summary[:10]],
            "paperIdsMissingMemoryCard": [str(row.get("arxiv_id") or "") for row in missing_memory[:10]],
            "paperIdsNotIndexed": [str(row.get("arxiv_id") or "") for row in not_indexed[:10]],
            "recommendedActions": list(latest_report.get("recommendedActions") or [])[:5],
        },
        "createdAt": _now_iso(),
    }
    _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.lint.result.v1")
    return payload


def _paper_mode_count(*, paper_query: str | None, urls: tuple[str, ...], url_file: str | None, youtube_urls: tuple[str, ...], youtube_url_file: str | None) -> int:
    return sum(
        1
        for is_present in (
            bool(str(paper_query or "").strip()),
            bool(urls or str(url_file or "").strip()),
            bool(youtube_urls or str(youtube_url_file or "").strip()),
        )
        if is_present
    )


def _source_ref_options(function):
    function = click.option("--source-ref", "source_ref_jsons", multiple=True, default=())(function)
    function = click.option("--paper-id", "paper_ids", multiple=True, default=())(function)
    function = click.option("--url", "urls", multiple=True, default=())(function)
    function = click.option("--note-id", "note_ids", multiple=True, default=())(function)
    function = click.option("--stable-scope-id", "stable_scope_ids", multiple=True, default=())(function)
    function = click.option("--document-scope-id", "document_scope_ids", multiple=True, default=())(function)
    return function


def _collect_source_refs(
    *,
    source_ref_jsons: tuple[str, ...],
    paper_ids: tuple[str, ...],
    urls: tuple[str, ...],
    note_ids: tuple[str, ...],
    stable_scope_ids: tuple[str, ...],
    document_scope_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for raw in source_ref_jsons:
        text = str(raw or "").strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            refs.append(dict(payload))
    for value in paper_ids:
        text = str(value or "").strip()
        if text:
            refs.append({"sourceType": "paper", "paperId": text})
    for value in urls:
        text = str(value or "").strip()
        if text:
            refs.append({"sourceType": "web", "url": text})
    for value in note_ids:
        text = str(value or "").strip()
        if text:
            refs.append({"sourceType": "vault", "noteId": text})
    for value in stable_scope_ids:
        text = str(value or "").strip()
        if text:
            refs.append({"sourceType": "scope", "stableScopeId": text})
    for value in document_scope_ids:
        text = str(value or "").strip()
        if text:
            refs.append({"sourceType": "document", "documentScopeId": text})
    return refs


def _dedupe_source_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        source_type = str(ref.get("sourceType") or "").strip()
        primary_value = ""
        for key in SOURCE_REF_PRIMARY_KEYS:
            candidate = str(ref.get(key) or "").strip()
            if candidate:
                primary_value = candidate
                break
        marker = (source_type, primary_value)
        if not source_type or not primary_value or marker in seen:
            continue
        seen.add(marker)
        deduped.append({k: v for k, v in ref.items() if v not in (None, "")})
    return deduped


def _format_source_refs(refs: list[dict[str, Any]]) -> str:
    if not refs:
        return "## Source Refs\n- none"
    lines = ["## Source Refs"]
    for ref in refs:
        source_type = str(ref.get("sourceType") or "").strip() or "unknown"
        detail = ""
        for key in SOURCE_REF_PRIMARY_KEYS:
            candidate = str(ref.get(key) or "").strip()
            if candidate:
                detail = candidate
                break
        if detail:
            lines.append(f"- `{source_type}` {detail}")
        else:
            lines.append(f"- `{source_type}`")
    return "\n".join(lines)


def _build_capture_filing_input_from_payload(
    *,
    payload: dict[str, Any],
    input_path: str,
    title: str | None,
) -> dict[str, Any]:
    schema_id = str(payload.get("schema") or "").strip()
    capture_payload = payload.get("payload")
    if not isinstance(capture_payload, dict):
        capture_payload = {}
    resolved_capture_id = str(payload.get("captureId") or capture_payload.get("captureId") or "").strip()
    resolved_packet_path = str(payload.get("packetPath") or payload.get("queuePath") or input_path or "").strip()
    metadata = dict(capture_payload.get("metadata") or {})
    metadata.update(
        {
            "input": "json",
            "inputPath": str(input_path or "").strip(),
            "sourceSchema": schema_id,
            "captureUrl": str(payload.get("sourceUrl") or payload.get("captureUrl") or "").strip(),
            "captureId": resolved_capture_id,
            "packetPath": resolved_packet_path,
            "status": str(payload.get("status") or "").strip() or "captured",
            "queueStatus": str(payload.get("queueStatus") or "").strip(),
            "client": str(payload.get("client") or capture_payload.get("client") or "").strip(),
        }
    )
    resolved_title = (
        str(title or "").strip()
        or str(capture_payload.get("pageTitle") or "").strip()
        or str(payload.get("pageTitle") or payload.get("title") or "").strip()
        or Path(str(input_path or "capture")).stem
    )
    lines = []
    capture_url = str(payload.get("sourceUrl") or payload.get("captureUrl") or "").strip()
    if capture_url:
        lines.extend(["## Source", capture_url, ""])
    selection_text = str(capture_payload.get("selectionText") or payload.get("selectionText") or "").strip()
    if selection_text:
        lines.extend(["## Captured Content", selection_text, ""])
    note_text = str(capture_payload.get("note") or payload.get("note") or "").strip()
    if note_text:
        lines.extend(["## Note", note_text, ""])
    client = str(payload.get("client") or capture_payload.get("client") or "").strip()
    tags = list(payload.get("tags") or capture_payload.get("tags") or [])
    metadata_lines = []
    if client:
        metadata_lines.append(f"- client: `{client}`")
    if tags:
        metadata_lines.append(f"- tags: `{', '.join(str(tag) for tag in tags if str(tag).strip())}`")
    if metadata_lines:
        lines.extend(["## Capture Metadata", *metadata_lines, ""])
    error = str(payload.get("error") or "").strip()
    if error:
        lines.extend(["## Error", error, ""])
    content_body = "\n".join(lines).strip() or "## Capture\n- empty"
    return {
        "kind": "web_capture",
        "title": resolved_title,
        "contentBody": content_body,
        "metadata": metadata,
        "sourceRefs": list(capture_payload.get("sourceRefs") or payload.get("sourceRefs") or []),
        "trace": {
            "sourceSchema": schema_id,
            "captureId": resolved_capture_id,
            "packetPath": resolved_packet_path,
        },
    }


def build_dinger_filing_input_from_payload(
    *,
    payload: dict[str, Any],
    input_path: str = "",
    title: str | None = None,
) -> dict[str, Any]:
    return resolve_dinger_filing_request_from_payload(payload, input_path=input_path, title=title)


def _resolve_file_input(*, title: str | None, body: str | None, body_file: Path | None, from_json_path: Path | None) -> dict[str, Any]:
    provided = sum(1 for value in (str(body or "").strip(), body_file, from_json_path) if value)
    if provided != 1:
        raise click.ClickException("choose exactly one file input: --body, --body-file, or --from-json")

    if body_file is not None:
        text = body_file.read_text(encoding="utf-8")
        resolved_title = str(title or "").strip() or body_file.stem
        return {
            "kind": "note",
            "title": resolved_title,
            "contentBody": str(text).rstrip(),
            "metadata": {"input": "body_file", "inputPath": str(body_file)},
            "sourceRefs": [],
        }

    if str(body or "").strip():
        resolved_title = str(title or "").strip()
        if not resolved_title:
            raise click.ClickException("--title is required when using --body")
        return {
            "kind": "note",
            "title": resolved_title,
            "contentBody": str(body).rstrip(),
            "metadata": {"input": "body"},
            "sourceRefs": [],
        }

    assert from_json_path is not None
    payload = json.loads(from_json_path.read_text(encoding="utf-8"))
    return build_dinger_filing_input_from_payload(payload=payload, input_path=str(from_json_path), title=title)


def _resolve_capture_input(
    *,
    source_url: str | None,
    url: str | None,
    page_title: str | None,
    title: str | None,
    selection_text: str | None,
    selection: str | None,
    note: str | None,
    body: str | None,
    body_file: Path | None,
    captured_at: str | None,
    client: str | None,
    tags: tuple[str, ...],
    raw_path: str | None,
    raw_vault_note_path: str | None,
) -> dict[str, Any]:
    resolved_url = str(source_url or url or "").strip()
    if not resolved_url:
        raise click.ClickException("--source-url/--url is required")
    resolved_title = str(page_title or title or "").strip()
    if not resolved_title:
        raise click.ClickException("--page-title/--title is required")
    provided = sum(
        1
        for value in (str(selection_text or "").strip(), str(selection or "").strip(), str(body or "").strip(), body_file)
        if value
    )
    if provided != 1:
        raise click.ClickException("choose exactly one capture body: --selection-text/--selection, --body, or --body-file")
    if body_file is not None:
        capture_body = body_file.read_text(encoding="utf-8").rstrip()
    elif str(body or "").strip():
        capture_body = str(body).rstrip()
    elif str(selection_text or "").strip():
        capture_body = str(selection_text).rstrip()
    else:
        capture_body = str(selection or "").rstrip()
    resolved_client = str(client or "").strip()
    if not resolved_client:
        raise click.ClickException("--client is required")
    resolved_tags: list[str] = []
    for value in tags:
        text = str(value or "").strip()
        if text and text not in resolved_tags:
            resolved_tags.append(text)
    if not resolved_tags:
        raise click.ClickException("at least one --tag is required")
    if str(raw_path or "").strip() and str(raw_vault_note_path or "").strip():
        raise click.ClickException("choose at most one raw reference: --raw-path or --raw-vault-note-path")
    return {
        "kind": "web_capture",
        "sourceUrl": resolved_url,
        "pageTitle": resolved_title,
        "selectionText": capture_body,
        "capturedAt": str(captured_at or "").strip() or _now_iso(),
        "client": resolved_client,
        "tags": resolved_tags,
        "rawPath": str(raw_path or "").strip(),
        "rawVaultNotePath": str(raw_vault_note_path or "").strip(),
        "note": str(note or "").rstrip(),
        "metadata": {
            "input": "capture",
            "sourceType": "web",
        },
        "sourceRefs": [{"sourceType": "web", "url": resolved_url}],
    }


def _capture_attempt_payload(
    *,
    source_url: str | None,
    url: str | None,
    page_title: str | None,
    title: str | None,
    selection_text: str | None,
    selection: str | None,
    note: str | None,
    body: str | None,
    body_file: Path | None,
    captured_at: str | None,
    client: str | None,
    tags: tuple[str, ...],
    raw_path: str | None,
    raw_vault_note_path: str | None,
) -> dict[str, Any]:
    resolved_url = str(source_url or url or "").strip()
    resolved_title = str(page_title or title or "").strip()
    return {
        "kind": "web_capture",
        "sourceUrl": resolved_url,
        "pageTitle": resolved_title,
        "selectionText": str(selection_text or selection or body or "").strip(),
        "capturedAt": str(captured_at or "").strip() or _now_iso(),
        "client": str(client or "").strip(),
        "tags": [str(value).strip() for value in tags if str(value).strip()],
        "rawPath": str(raw_path or "").strip(),
        "rawVaultNotePath": str(raw_vault_note_path or "").strip(),
        "note": str(note or "").strip(),
        "sourceRefs": [{"sourceType": "web", "url": resolved_url}] if resolved_url else [],
        "raw": {
            "sourceUrl": resolved_url,
            "pageTitle": resolved_title,
            "client": str(client or "").strip(),
            "tags": [str(value).strip() for value in tags if str(value).strip()],
            "selection": str(selection or "").strip(),
            "selectionText": str(selection_text or "").strip(),
            "body": str(body or "").strip(),
            "bodyFile": str(body_file or ""),
            "note": str(note or "").strip(),
            "capturedAt": str(captured_at or "").strip(),
            "rawPath": str(raw_path or "").strip(),
            "rawVaultNotePath": str(raw_vault_note_path or "").strip(),
        },
    }


def _resolve_capture_queue_dir(config) -> Path:
    return resolve_capture_queue_dir(config)


def _normalize_capture_source_refs(source_refs: Any) -> list[dict[str, Any]]:
    if source_refs in (None, ""):
        return []
    if not isinstance(source_refs, list):
        raise click.ClickException("sourceRefs must be an array of objects")
    refs: list[dict[str, Any]] = []
    for index, item in enumerate(source_refs):
        if not isinstance(item, dict):
            raise click.ClickException(f"sourceRefs[{index}] must be an object")
        refs.append(dict(item))
    return refs


def _build_capture_result_payload(
    *,
    capture_payload: dict[str, Any],
    source_refs: list[dict[str, Any]],
    packet_path: str,
    queue_status: str,
    accepted: bool,
    capture_status: str,
    error: str = "",
) -> dict[str, Any]:
    resolved_packet_path = str(packet_path or "").strip()
    resolved_queue_status = str(queue_status or "").strip()
    resolved_status = str(capture_status or "failed").strip() or "failed"
    resolved_source_url = str(capture_payload.get("sourceUrl") or "").strip()
    resolved_page_title = str(capture_payload.get("pageTitle") or "").strip()
    resolved_slug = slugify_title(resolved_page_title or resolved_source_url, fallback="dinger-capture")
    payload = {
        "schema": "knowledge-hub.dinger.capture.result.v1",
        "status": resolved_status,
        "accepted": bool(accepted),
        "queued": resolved_queue_status == "queued",
        "captureId": str(capture_payload.get("captureId") or "").strip(),
        "sourceUrl": resolved_source_url,
        "pageTitle": resolved_page_title,
        "selectionText": str(capture_payload.get("selectionText") or "").strip(),
        "capturedAt": str(capture_payload.get("capturedAt") or "").strip(),
        "client": str(capture_payload.get("client") or "").strip(),
        "tags": list(capture_payload.get("tags") or []),
        "rawPath": str(capture_payload.get("rawPath") or "").strip(),
        "rawVaultNotePath": str(capture_payload.get("rawVaultNotePath") or "").strip(),
        "note": str(capture_payload.get("note") or "").strip(),
        "queuePath": resolved_packet_path,
        "packetPath": resolved_packet_path,
        "captureUrl": resolved_source_url,
        "title": resolved_page_title,
        "slug": resolved_slug,
        "sourceRefs": source_refs,
        "payload": capture_payload,
        "createdAt": _now_iso(),
    }
    if resolved_queue_status:
        payload["queueStatus"] = resolved_queue_status
    if str(error or "").strip():
        payload["error"] = str(error).strip()
    return payload


def _normalize_capture_request_source_refs(source_refs: Any) -> list[dict[str, Any]]:
    if source_refs is None:
        return []
    if not isinstance(source_refs, list):
        raise click.ClickException("sourceRefs must be an array")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(source_refs):
        if not isinstance(item, dict):
            raise click.ClickException(f"sourceRefs[{index}] must be an object")
        normalized.append(dict(item))
    return normalized


def _build_capture_failure_payload(
    *,
    khub,
    source_url: str | None,
    url: str | None,
    page_title: str | None,
    title: str | None,
    selection_text: str | None,
    selection: str | None,
    note: str | None,
    body: str | None,
    body_file: Path | None,
    captured_at: str | None,
    client: str | None,
    tags: tuple[str, ...],
    raw_path: str | None,
    raw_vault_note_path: str | None,
    source_refs: list[dict[str, Any]] | None,
    message: str,
) -> dict[str, Any]:
    attempted_payload = _capture_attempt_payload(
        source_url=source_url,
        url=url,
        page_title=page_title,
        title=title,
        selection_text=selection_text,
        selection=selection,
        note=note,
        body=body,
        body_file=body_file,
        captured_at=captured_at,
        client=client,
        tags=tuple(tags),
        raw_path=raw_path,
        raw_vault_note_path=raw_vault_note_path,
    )
    attempted_source_refs = _dedupe_source_refs(list(attempted_payload.get("sourceRefs") or []) + list(source_refs or []))
    attempted_payload["sourceRefs"] = attempted_source_refs
    failed_payload = _build_capture_result_payload(
        capture_payload=attempted_payload,
        source_refs=attempted_source_refs,
        packet_path="",
        queue_status="",
        accepted=False,
        capture_status="failed",
        error=message,
    )
    _validate_cli_payload(khub.config, failed_payload, "knowledge-hub.dinger.capture.result.v1")
    return failed_payload


def run_dinger_capture_intake(
    *,
    khub,
    source_url: str | None,
    page_title: str | None,
    selection_text: str | None,
    captured_at: str | None,
    client: str | None,
    tags: tuple[str, ...],
    note: str | None = None,
    raw_path: str | None = None,
    raw_vault_note_path: str | None = None,
    source_refs: list[dict[str, Any]] | None = None,
    url: str | None = None,
    title: str | None = None,
    selection: str | None = None,
    body: str | None = None,
    body_file: Path | None = None,
) -> dict[str, Any]:
    normalized_source_refs = list(source_refs or [])
    try:
        resolved = _resolve_capture_input(
            source_url=source_url,
            url=url,
            page_title=page_title,
            title=title,
            selection_text=selection_text,
            selection=selection,
            note=note,
            body=body,
            body_file=body_file,
            captured_at=captured_at,
            client=client,
            tags=tuple(tags),
            raw_path=raw_path,
            raw_vault_note_path=raw_vault_note_path,
        )
        deduped_source_refs = _dedupe_source_refs(list(resolved.get("sourceRefs") or []) + normalized_source_refs)
        capture_id = f"cap_{uuid4().hex[:12]}"
        capture_payload = {
            **resolved,
            "captureId": capture_id,
            "sourceRefs": deduped_source_refs,
        }
        packet_path = _resolve_capture_queue_dir(khub.config) / f"{capture_id}.json"
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _build_capture_result_payload(
            capture_payload=capture_payload,
            source_refs=deduped_source_refs,
            packet_path=str(packet_path),
            queue_status="queued",
            accepted=True,
            capture_status="captured",
        )
        packet_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.capture.result.v1")
        return payload
    except Exception as error:
        return _build_capture_failure_payload(
            khub=khub,
            source_url=source_url,
            url=url,
            page_title=page_title,
            title=title,
            selection_text=selection_text,
            selection=selection,
            note=note,
            body=body,
            body_file=body_file,
            captured_at=captured_at,
            client=client,
            tags=tuple(tags),
            raw_path=raw_path,
            raw_vault_note_path=raw_vault_note_path,
            source_refs=normalized_source_refs,
            message=str(error),
        )


def _build_file_failure_payload(
    *,
    message: str,
    title: str = "",
    slug: str = "",
    kind: str = "note",
    source_refs: list[dict[str, Any]] | None = None,
    vault_path: str = "",
    backend: str = "",
    source_schema: str = "",
    capture_id: str = "",
    packet_path: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "schema": "knowledge-hub.dinger.file.result.v1",
        "status": "failed",
        "title": str(title or "").strip(),
        "slug": str(slug or "").strip(),
        "kind": str(kind or "note").strip() or "note",
        "relativePath": "",
        "filePath": "",
        "indexPath": "",
        "logPath": "",
        "vaultPath": str(vault_path or "").strip(),
        "backend": str(backend or "").strip(),
        "sourceRefs": list(source_refs or []),
        "createdAt": _now_iso(),
        "error": str(message or "").strip() or "unknown error",
        "sourceSchema": str(source_schema or "").strip(),
        "captureId": str(capture_id or "").strip(),
        "packetPath": str(packet_path or "").strip(),
        "payload": dict(payload or {}),
    }
    return result


def _obsidian_link(relative_path: str, title: str) -> str:
    target = str(Path(relative_path).with_suffix("")).replace("\\", "/")
    label = str(title or "").strip() or Path(relative_path).stem
    return f"[[{target}|{label}]]"


def _load_dinger_pages(vault_root: Path, pages_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not pages_dir.exists():
        return entries
    for path in sorted(pages_dir.glob("*.md")):
        frontmatter, _body = split_frontmatter(path.read_text(encoding="utf-8"))
        if not isinstance(frontmatter, dict):
            frontmatter = {}
        relative_path = str(path.relative_to(vault_root).as_posix())
        entries.append(
            {
                "title": str(frontmatter.get("title") or path.stem),
                "slug": str(frontmatter.get("slug") or path.stem),
                "kind": str(frontmatter.get("dingerKind") or "note"),
                "createdAt": str(frontmatter.get("createdAt") or ""),
                "updatedAt": str(frontmatter.get("updatedAt") or ""),
                "relativePath": relative_path,
                "sourceType": str(frontmatter.get("sourceType") or ""),
            }
        )
    entries.sort(key=lambda item: (str(item.get("updatedAt") or ""), str(item.get("slug") or "")), reverse=True)
    return entries


def _render_dinger_index(entries: list[dict[str, Any]]) -> str:
    lines = ["## Filed Pages"]
    if not entries:
        lines.append("- none")
        return "\n".join(lines)
    for entry in entries[:50]:
        suffix_bits = [str(entry.get("kind") or "")]
        if str(entry.get("sourceType") or "").strip():
            suffix_bits.append(str(entry.get("sourceType")))
        updated_at = str(entry.get("updatedAt") or "").strip()
        if updated_at:
            suffix_bits.append(updated_at)
        suffix = f" ({', '.join(bit for bit in suffix_bits if bit)})" if any(suffix_bits) else ""
        lines.append(f"- {_obsidian_link(str(entry.get('relativePath') or ''), str(entry.get('title') or ''))}{suffix}")
    return "\n".join(lines)


def _render_dinger_log(entries: list[dict[str, Any]]) -> str:
    lines = ["## Recent Filed Items"]
    if not entries:
        lines.append("- none")
        return "\n".join(lines)
    for entry in entries[:50]:
        updated_at = str(entry.get("updatedAt") or "").strip() or str(entry.get("createdAt") or "").strip()
        lines.append(
            f"- `{updated_at}` {str(entry.get('kind') or 'note')}: "
            f"{_obsidian_link(str(entry.get('relativePath') or ''), str(entry.get('title') or ''))}"
        )
    return "\n".join(lines)


def _write_dinger_projection(
    *,
    khub,
    title: str,
    slug: str,
    kind: str,
    content_body: str,
    source_refs: list[dict[str, Any]],
    metadata: dict[str, Any],
    vault_path: str | None,
    backend: str | None,
    cli_binary: str | None,
    vault_name: str | None,
) -> dict[str, Any]:
    config = khub.config
    resolved_vault = str(vault_path or config.vault_path or "").strip()
    if not resolved_vault:
        raise click.ClickException("vault_path not configured")
    resolved_backend = str(backend or config.get_nested("obsidian", "write_backend", default="filesystem") or "filesystem").strip() or "filesystem"
    resolved_cli_binary = str(cli_binary or config.get_nested("obsidian", "cli_binary", default="obsidian") or "obsidian").strip() or "obsidian"
    resolved_vault_name = str(vault_name or config.get_nested("obsidian", "vault_name", default="") or "").strip()

    adapter = resolve_vault_write_adapter(
        resolved_vault,
        backend=resolved_backend,
        cli_binary=resolved_cli_binary,
        vault_name=resolved_vault_name,
    )
    vault_root = Path(resolved_vault).expanduser().resolve()
    dinger_root = vault_root / "KnowledgeOS" / "Dinger"
    pages_dir = dinger_root / "Pages"
    page_path = pages_dir / f"{slug}.md"
    index_path = dinger_root / "Index.md"
    log_path = dinger_root / "Log.md"

    existing = adapter.read_text(page_path)
    existing_frontmatter, existing_body = split_frontmatter(existing)
    if not isinstance(existing_frontmatter, dict):
        existing_frontmatter = {}
    content = str(existing_body or "").strip()
    if not content:
        content = f"# {title}\n"

    created_at = str(existing_frontmatter.get("createdAt") or "").strip() or _now_iso()
    updated_at = _now_iso()
    content = _upsert_marked_section(content, "khub-dinger-content", content_body)
    content = _upsert_marked_section(content, "khub-dinger-source-refs", _format_source_refs(source_refs))
    content = _upsert_marked_section(
        content,
        "khub-dinger-metadata",
        "\n".join(
            [
                "## Metadata",
                f"- kind: `{kind}`",
                f"- createdAt: `{created_at}`",
                f"- updatedAt: `{updated_at}`",
                *[f"- {key}: `{value}`" for key, value in metadata.items() if str(value or "").strip()],
            ]
        ),
    )

    frontmatter = {
        "title": title,
        "slug": slug,
        "dingerKind": kind,
        "createdAt": created_at,
        "updatedAt": updated_at,
        "sourceType": str(metadata.get("sourceType") or ""),
        "sourceRefs": source_refs,
        "managedBy": "knowledge-hub.dinger.file.v1",
    }
    adapter.write_text(page_path, yaml_frontmatter(frontmatter) + content.rstrip() + "\n")

    entries = _load_dinger_pages(vault_root, pages_dir)
    index_content = adapter.read_text(index_path) or "# Dinger Index\n"
    index_content = _upsert_marked_section(index_content, "khub-dinger-index", _render_dinger_index(entries))
    adapter.write_text(index_path, index_content.rstrip() + "\n")

    log_content = adapter.read_text(log_path) or "# Dinger Log\n"
    log_content = _upsert_marked_section(log_content, "khub-dinger-log", _render_dinger_log(entries))
    adapter.write_text(log_path, log_content.rstrip() + "\n")

    return {
        "schema": "knowledge-hub.dinger.file.result.v1",
        "status": "ok",
        "title": title,
        "slug": slug,
        "kind": kind,
        "relativePath": str(page_path.relative_to(vault_root).as_posix()),
        "filePath": str(page_path),
        "indexPath": str(index_path),
        "logPath": str(log_path),
        "vaultPath": str(vault_root),
        "backend": resolved_backend,
        "sourceRefs": source_refs,
        "createdAt": updated_at,
    }


def _apply_file_trace(payload: dict[str, Any], trace: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(payload)
    trace = dict(trace or {})
    source_schema = str(trace.get("sourceSchema") or "").strip()
    capture_id = str(trace.get("captureId") or "").strip()
    packet_path = str(trace.get("packetPath") or "").strip()
    if source_schema:
        result["sourceSchema"] = source_schema
    if capture_id:
        result["captureId"] = capture_id
    if packet_path:
        result["packetPath"] = packet_path
    return result


def run_dinger_file_projection(
    *,
    khub,
    filing_input: dict[str, Any],
    slug: str | None = None,
    source_refs: list[dict[str, Any]] | None = None,
    vault_path: str | None = None,
    backend: str | None = None,
    cli_binary: str | None = None,
    vault_name: str | None = None,
) -> dict[str, Any]:
    payload = file_dinger_request(
        khub=khub,
        request=filing_input,
        slug=slug,
        extra_source_refs=source_refs,
        vault_path=vault_path,
        backend=backend,
        cli_binary=cli_binary,
        vault_name=vault_name,
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.file.result.v1")
    return payload


@click.group("dinger")
def dinger_group():
    """Dinger - simplified personal knowledge surface over the existing engine."""


@dinger_group.command("ingest")
@click.option("--paper", "paper_query", default=None, help="논문 주제/검색어로 ingest")
@click.option("--url", "urls", multiple=True, help="웹 URL ingest (반복 사용 가능)")
@click.option("--url-file", default=None, help="웹 URL 목록(.txt)")
@click.option("--youtube-url", "youtube_urls", multiple=True, help="YouTube URL ingest (반복 사용 가능)")
@click.option("--youtube-url-file", default=None, help="YouTube URL 목록(.txt)")
@click.option("--topic", default="", help="웹/영상 ingest용 topic label")
@click.option("--max-papers", type=int, default=3, show_default=True)
@click.option("--apply/--stage-only", "apply_notes", default=False, show_default=True)
@click.option("--allow-external", is_flag=True, default=False, help="외부 LLM 보강 허용")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def dinger_ingest(ctx, paper_query, urls, url_file, youtube_urls, youtube_url_file, topic, max_papers, apply_notes, allow_external, as_json):
    """자료 넣기: paper query, web URL, youtube URL을 목적어 기준으로 ingest."""
    khub = ctx.obj["khub"]
    if _paper_mode_count(
        paper_query=paper_query,
        urls=tuple(urls),
        url_file=url_file,
        youtube_urls=tuple(youtube_urls),
        youtube_url_file=youtube_url_file,
    ) != 1:
        raise click.ClickException("choose exactly one ingest target: --paper, --url/--url-file, or --youtube-url/--youtube-url-file")

    try:
        if str(paper_query or "").strip():
            from knowledge_hub.interfaces.cli.commands.discover_cmd import discover

            nested = _invoke_json_command(
                discover,
                [
                    str(paper_query).strip(),
                    "--max-papers",
                    str(max(1, int(max_papers))),
                    "--yes",
                ],
                obj=ctx.obj,
                label="dinger ingest paper",
            )
            mode = "paper_query"
            target = {"paperQuery": str(paper_query).strip(), "maxPapers": max(1, int(max_papers))}
        elif urls or str(url_file or "").strip():
            from knowledge_hub.interfaces.cli.commands.crawl_cmd import crawl_collect

            args: list[str] = []
            for url in urls:
                args.extend(["--url", str(url)])
            if str(url_file or "").strip():
                args.extend(["--url-file", str(url_file).strip()])
            if str(topic or "").strip():
                args.extend(["--topic", str(topic).strip()])
            if apply_notes:
                args.append("--apply")
            else:
                args.append("--stage-only")
            if allow_external:
                args.append("--allow-external")
            nested = _invoke_json_command(crawl_collect, args, obj=ctx.obj, label="dinger ingest web")
            mode = "web"
            target = {
                "topic": str(topic or "").strip(),
                "urlCount": len(urls) + (1 if str(url_file or "").strip() else 0),
                "apply": bool(apply_notes),
            }
        else:
            from knowledge_hub.interfaces.cli.commands.crawl_cmd import crawl_youtube_ingest

            args = []
            for url in youtube_urls:
                args.extend(["--url", str(url)])
            if str(youtube_url_file or "").strip():
                args.extend(["--url-file", str(youtube_url_file).strip()])
            if str(topic or "").strip():
                args.extend(["--topic", str(topic).strip()])
            if allow_external:
                args.append("--allow-external")
            nested = _invoke_json_command(crawl_youtube_ingest, args, obj=ctx.obj, label="dinger ingest youtube")
            mode = "youtube"
            target = {
                "topic": str(topic or "").strip(),
                "urlCount": len(youtube_urls) + (1 if str(youtube_url_file or "").strip() else 0),
            }
    except click.ClickException as error:
        _emit_or_raise_failure(
            khub=khub,
            as_json=as_json,
            schema_id="knowledge-hub.dinger.ingest.result.v1",
            message=str(error),
            mode="paper_query" if str(paper_query or "").strip() else ("web" if (urls or str(url_file or "").strip()) else "youtube"),
            target={},
            underlyingSchema="",
            payload={},
        )
        return

    payload = {
        "schema": "knowledge-hub.dinger.ingest.result.v1",
        "status": "ok",
        "mode": mode,
        "target": target,
        "underlyingSchema": str(nested.get("schema") or ""),
        "payload": nested,
        "createdAt": _now_iso(),
    }
    _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.ingest.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]dinger ingest[/bold] mode={mode} status={payload.get('status')} "
        f"schema={payload.get('underlyingSchema') or '-'}"
    )


@dinger_group.command("ask")
@click.argument("question")
@click.option("--source", default=None, help="source filter: paper, web, vault, concept")
@click.option("--top-k", type=int, default=8, show_default=True)
@click.option("--allow-external/--no-allow-external", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def dinger_ask(ctx, question, source, top_k, allow_external, as_json):
    """질문하기: 기존 ask runtime을 dinger surface로 노출."""
    khub = ctx.obj["khub"]
    args = [str(question), "--top-k", str(max(1, int(top_k)))]
    if str(source or "").strip():
        args.extend(["--source", str(source).strip()])
    if allow_external is True:
        args.append("--allow-external")
    elif allow_external is False:
        args.append("--no-allow-external")
    try:
        from knowledge_hub.interfaces.cli.commands.search_cmd import ask as ask_cmd

        nested = _invoke_json_command(ask_cmd, args, obj=ctx.obj, label="dinger ask")
    except click.ClickException as error:
        _emit_or_raise_failure(
            khub=khub,
            as_json=as_json,
            schema_id="knowledge-hub.dinger.ask.result.v1",
            message=str(error),
            question=str(question),
            sourceType=str(source or ""),
            underlyingSchema="",
            answer="",
            payload={},
        )
        return
    payload = {
        "schema": "knowledge-hub.dinger.ask.result.v1",
        "status": "ok",
        "question": str(question),
        "sourceType": str(source or ""),
        "underlyingSchema": str(nested.get("schema") or ""),
        "answer": str(nested.get("answer") or ""),
        "payload": nested,
        "createdAt": _now_iso(),
    }
    _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.ask.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold cyan]Q:[/bold cyan] {question}")
    console.print(payload["answer"])


@dinger_group.group("capture", invoke_without_command=True)
@click.option("--source-url", default=None, help="Captured web URL")
@click.option("--url", default=None, help="Alias for --source-url")
@click.option("--page-title", default=None, help="Captured page title")
@click.option("--title", default=None, help="Alias for --page-title")
@click.option("--selection-text", default=None, help="Selected text from the page")
@click.option("--selection", default=None, help="Alias for --selection-text")
@click.option("--body", default=None, help="Captured reader/body text")
@click.option("--body-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Captured text file")
@click.option("--note", default=None, help="Optional user note")
@click.option("--captured-at", default=None, help="Capture timestamp")
@click.option("--client", default=None, help="Capture client identifier")
@click.option("--tag", "tags", multiple=True, default=(), help="Capture tag (repeatable)")
@click.option("--raw-path", default=None, help="Optional raw input path")
@click.option("--raw-vault-note-path", default=None, help="Optional raw vault note path")
@click.option("--slug", default=None, help="Override page slug")
@click.option("--vault-path", default=None, help="Override Obsidian vault path")
@click.option("--backend", default=None, help="Override vault write backend")
@click.option("--cli-binary", default=None, help="Override Obsidian CLI binary")
@click.option("--vault-name", default=None, help="Override Obsidian vault name")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.option("--source-ref", "source_ref_jsons", multiple=True, default=())
@click.option("--paper-id", "paper_ids", multiple=True, default=())
@click.option("--note-id", "note_ids", multiple=True, default=())
@click.option("--stable-scope-id", "stable_scope_ids", multiple=True, default=())
@click.option("--document-scope-id", "document_scope_ids", multiple=True, default=())
@click.pass_context
def dinger_capture(
    ctx,
    source_url,
    url,
    page_title,
    title,
    selection_text,
    selection,
    body,
    body_file,
    note,
    captured_at,
    client,
    tags,
    raw_path,
    raw_vault_note_path,
    slug,
    vault_path,
    backend,
    cli_binary,
    vault_name,
    as_json,
    source_ref_jsons,
    paper_ids,
    note_ids,
    stable_scope_ids,
    document_scope_ids,
):
    """브라우저/클립퍼가 보낼 수 있는 web capture intake surface."""
    if ctx.invoked_subcommand is not None:
        return
    khub = ctx.obj["khub"]
    _ = (slug, vault_path, backend, cli_binary, vault_name)
    extra_source_refs = _collect_source_refs(
        source_ref_jsons=source_ref_jsons,
        paper_ids=paper_ids,
        urls=(),
        note_ids=note_ids,
        stable_scope_ids=stable_scope_ids,
        document_scope_ids=document_scope_ids,
    )
    payload = run_dinger_capture_intake(
        khub=khub,
        source_url=source_url,
        page_title=page_title,
        selection_text=selection_text,
        captured_at=captured_at,
        client=client,
        tags=tuple(tags),
        note=note,
        raw_path=raw_path,
        raw_vault_note_path=raw_vault_note_path,
        source_refs=extra_source_refs,
        url=url,
        title=title,
        selection=selection,
        body=body,
        body_file=body_file,
    )
    if as_json:
        console.print_json(data=payload)
        return
    if str(payload.get("status") or "") == "failed":
        raise click.ClickException(str(payload.get("error") or "capture intake failed"))
    console.print(
        f"[bold]dinger capture[/bold] capture_id={payload.get('captureId')} "
        f"status={payload.get('status')} queue={payload.get('queueStatus')}"
    )


@dinger_capture.command("list")
@click.option(
    "--queue-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override capture queue directory (default: runtime dinger_capture_intake/queue)",
)
@click.option(
    "--status",
    "statuses",
    multiple=True,
    type=click.Choice(CAPTURE_INSPECTABLE_STATUSES, case_sensitive=False),
    default=(),
    help="Filter by derived operator status (repeatable)",
)
@click.option("--limit", type=int, default=20, show_default=True, help="Maximum number of captures to return; 0 means all")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def dinger_capture_list(ctx, queue_dir, statuses, limit, as_json):
    """List queued/processed capture packets with derived operator statuses."""
    khub = ctx.obj["khub"]
    items, resolved_queue_dir, resolved_runtime_dir = _collect_capture_read_model_items(
        khub=khub,
        queue_dir=queue_dir,
        statuses=statuses,
        limit=limit,
    )
    payload = {
        "schema": "knowledge-hub.dinger.capture-list.result.v1",
        "status": "ok",
        "queueDir": str(resolved_queue_dir),
        "runtimeDir": str(resolved_runtime_dir),
        "count": len(items),
        "items": items,
        "createdAt": _now_iso(),
    }
    _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.capture-list.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    for item in items:
        badges = _capture_list_badges(item)
        badge_text = f"  [{' | '.join(badges)}]" if badges else ""
        console.print(
            f"{item['captureId']}  {item['status']}{badge_text}  {item.get('title') or item.get('sourceUrl') or '-'}"
        )
        if str(item.get("operatorAction") or "").strip():
            console.print(f"  hint: {item.get('operatorAction')}")


@dinger_capture.command("status")
@click.option(
    "--queue-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override capture queue directory (default: runtime dinger_capture_intake/queue)",
)
@click.option(
    "--runtime-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override capture runtime directory (default: runtime dinger_capture_intake/runtime)",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def dinger_capture_status(ctx, queue_dir, runtime_dir, as_json):
    """Show aggregate operator counts and next actions for capture queue/runtime health."""
    khub = ctx.obj["khub"]
    payload = _build_capture_status_payload(khub=khub, queue_dir=queue_dir, runtime_dir=runtime_dir)
    _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.capture-status.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    counts = dict(payload.get("counts") or {})
    console.print(
        f"[bold]dinger capture status[/bold] total={counts.get('totalCaptures', 0)} "
        f"queued={counts.get('queued', 0)} processing={counts.get('processing', 0)} "
        f"filed={counts.get('filed', 0)} linked_to_os={counts.get('linkedToOs', 0)} "
        f"failed={counts.get('failed', 0)}"
    )
    console.print(
        f"orphans={counts.get('orphanedRuntime', 0)} recoverable={counts.get('recoverableOrphans', 0)} "
        f"unrecoverable={counts.get('unrecoverableOrphans', 0)} retry_ready={counts.get('retryReady', 0)} "
        f"retry_blocked={counts.get('retryBlocked', 0)} stale_claims={counts.get('staleClaimFiles', 0)} "
        f"active_claims={counts.get('activeClaimFiles', 0)} cleanup_delete_eligible={counts.get('cleanupDeleteEligibleEntries', 0)}"
    )
    for action in list(payload.get("actions") or []):
        captures = ", ".join(str(item).strip() for item in list(action.get("sampleCaptureIds") or []) if str(item).strip())
        capture_text = f" captures={captures}" if captures else ""
        console.print(
            f"action: {action.get('kind')} count={action.get('count', 0)} command={action.get('command')}{capture_text}"
        )


@dinger_capture.command("show")
@click.option(
    "--queue-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override capture queue directory (default: runtime dinger_capture_intake/queue)",
)
@click.option("--packet", "packet_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--capture-id", default=None, help="Show a specific captureId")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def dinger_capture_show(ctx, queue_dir, packet_path, capture_id, as_json):
    """Show a single capture packet plus runtime artifacts."""
    khub = ctx.obj["khub"]
    try:
        resolved_capture_id, resolved_packet = _resolve_capture_read_model_selector(
            khub=khub,
            queue_dir=queue_dir,
            packet_path=packet_path,
            capture_id=capture_id,
        )
        item = _build_capture_read_model_item(
            khub=khub,
            packet_path=resolved_packet,
            capture_id=resolved_capture_id,
            queue_dir=queue_dir,
        )
        runtime_state = _load_optional_json(Path(str(item.get("statePath") or ""))) if item.get("statePath") else None
        normalized = _load_optional_json(Path(str(item.get("normalizedPath") or ""))) if item.get("normalizedPath") else None
        filed_result = _load_optional_json(Path(str(item.get("filedResultPath") or ""))) if item.get("filedResultPath") else None
        os_result = _load_optional_json(Path(str(item.get("osResultPath") or ""))) if item.get("osResultPath") else None
        payload = {
            "schema": "knowledge-hub.dinger.capture-show.result.v1",
            "status": "ok",
            "captureId": str(item.get("captureId") or ""),
            "packetPath": str(item.get("packetPath") or ""),
            "item": item,
            "packet": _load_capture_packet(khub=khub, packet_path=resolved_packet) if resolved_packet is not None else {},
            "runtimeState": runtime_state or {},
            "normalized": normalized or {},
            "fileResult": filed_result or {},
            "osResult": os_result or {},
            "createdAt": _now_iso(),
        }
        _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.capture-show.result.v1")
    except click.ClickException as error:
        payload = _capture_show_failure_payload(
            capture_id=str(capture_id or "").strip(),
            packet_path=str(packet_path or ""),
            message=str(error),
        )
        _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.capture-show.result.v1")
        if not as_json:
            raise
    if as_json:
        console.print_json(data=payload)
        return
    if str(payload.get("status") or "") == "failed":
        raise click.ClickException(str(payload.get("error") or "capture show failed"))
    item = dict(payload.get("item") or {})
    console.print(f"[bold]capture[/bold] {item.get('captureId')} {item.get('status')}")
    if str(item.get("operatorSummary") or "").strip():
        console.print(f"operator: {item.get('operatorSummary')}")
    console.print(f"title: {item.get('title') or '-'}")
    console.print(f"source: {item.get('sourceUrl') or '-'}")
    packet_prefix = "missing " if item.get("orphanedRuntime") else ""
    console.print(f"packet: {packet_prefix}{payload.get('packetPath')}")
    if item.get("warnings"):
        console.print(f"warnings: {', '.join(str(entry) for entry in list(item.get('warnings') or []))}")
    if str(item.get("requeueReason") or "").strip():
        console.print(f"requeue: {item.get('requeueReason')}")
    if str(item.get("retryHint") or "").strip():
        console.print(f"retry: {item.get('retryHint')}")
    if str(item.get("operatorAction") or "").strip():
        console.print(f"action: {item.get('operatorAction')}")


@dinger_capture.command("requeue")
@click.option(
    "--queue-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override capture queue directory (default: runtime dinger_capture_intake/queue)",
)
@click.option("--capture-id", default=None, help="Restore the canonical queue packet for a specific orphan captureId")
@click.option(
    "--packet-snapshot",
    "packet_snapshot_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Exact packet snapshot JSON to restore into the canonical queue path",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def dinger_capture_requeue(ctx, queue_dir, capture_id, packet_snapshot_path, as_json):
    """Restore a missing queue packet from a packet snapshot without re-running the processor."""
    khub = ctx.obj["khub"]
    resolved_capture_id = str(capture_id or "").strip()
    restored_from = ""
    resolved_packet_path = ""
    warnings: list[str] = []
    try:
        explicit_capture_id = bool(resolved_capture_id)
        explicit_snapshot = packet_snapshot_path is not None
        if explicit_capture_id == explicit_snapshot:
            raise click.ClickException("choose exactly one selector: --capture-id or --packet-snapshot")

        snapshot_payload: dict[str, Any] | None = None
        resolved_queue_dir = _resolve_capture_read_model_queue_dir(khub=khub, queue_dir=queue_dir)

        if explicit_snapshot:
            snapshot_path = Path(packet_snapshot_path).expanduser().resolve()
            if not snapshot_path.exists():
                raise click.ClickException(f"packet snapshot not found: {snapshot_path}")
            snapshot_payload = _load_capture_packet_snapshot(khub=khub, packet_snapshot_path=snapshot_path)
            resolved_capture_id = str(snapshot_payload.get("captureId") or "").strip()
            restored_from = str(snapshot_path)

        target_packet_path = _default_capture_packet_path(khub.config, resolved_capture_id, queue_dir=resolved_queue_dir).resolve()
        resolved_packet_path = str(target_packet_path)
        if target_packet_path.exists():
            if not restored_from and resolved_capture_id:
                found_snapshot = _find_capture_packet_snapshot_path(
                    khub=khub,
                    capture_id=resolved_capture_id,
                    queue_dir=resolved_queue_dir,
                )
                if found_snapshot is not None:
                    restored_from = str(found_snapshot)
            warnings = ["queue packet already exists at canonical path"]
            payload = {
                "schema": "knowledge-hub.dinger.capture-requeue.result.v1",
                "status": "ok",
                "result": "already_present",
                "captureId": resolved_capture_id,
                "packetPath": str(target_packet_path),
                "restoredFrom": restored_from,
                "requeued": False,
                "warnings": warnings,
                "createdAt": _now_iso(),
            }
            _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.capture-requeue.result.v1")
        else:
            if snapshot_payload is None:
                if not _has_capture_runtime_artifacts(khub.config, resolved_capture_id, queue_dir=resolved_queue_dir):
                    raise click.ClickException(
                        f"capture requeue only supports orphan captures with runtime artifacts or explicit packet snapshots: "
                        f"captureId={resolved_capture_id}"
                    )
                found_snapshot = _find_capture_packet_snapshot_path(
                    khub=khub,
                    capture_id=resolved_capture_id,
                    queue_dir=resolved_queue_dir,
                )
                if found_snapshot is None:
                    raise click.ClickException(
                        f"capture requeue requires a packet snapshot; legacy orphan captureId={resolved_capture_id} "
                        "has runtime artifacts but no exact packet snapshot"
                    )
                restored_from = str(found_snapshot)
                snapshot_payload = _load_capture_packet_snapshot(khub=khub, packet_snapshot_path=found_snapshot)
            snapshot_capture_id = str(snapshot_payload.get("captureId") or "").strip()
            if snapshot_capture_id != resolved_capture_id:
                raise click.ClickException(
                    f"packet snapshot captureId mismatch: expected {resolved_capture_id}, found {snapshot_capture_id}"
                )
            _restore_capture_packet_from_snapshot(
                khub=khub,
                snapshot_payload=snapshot_payload,
                packet_path=target_packet_path,
            )
            payload = {
                "schema": "knowledge-hub.dinger.capture-requeue.result.v1",
                "status": "ok",
                "result": "requeued",
                "captureId": resolved_capture_id,
                "packetPath": str(target_packet_path),
                "restoredFrom": restored_from,
                "requeued": True,
                "warnings": [],
                "createdAt": _now_iso(),
            }
            _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.capture-requeue.result.v1")
    except click.ClickException as error:
        payload = _capture_requeue_failure_payload(
            capture_id=resolved_capture_id,
            packet_path=resolved_packet_path,
            restored_from=restored_from,
            warnings=warnings,
            message=str(error),
        )
        _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.capture-requeue.result.v1")
        if not as_json:
            raise
    if as_json:
        console.print_json(data=payload)
        return
    if str(payload.get("status") or "") == "failed":
        raise click.ClickException(str(payload.get("error") or "capture requeue failed"))
    console.print(
        f"[bold]dinger capture requeue[/bold] capture_id={payload.get('captureId')} "
        f"result={payload.get('result')} requeued={payload.get('requeued')}"
    )


@dinger_capture.command("cleanup")
@click.option(
    "--queue-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override capture queue directory (default: runtime dinger_capture_intake/queue)",
)
@click.option(
    "--runtime-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override capture runtime directory (default: runtime dinger_capture_intake/runtime)",
)
@click.option("--apply", is_flag=True, help="Delete cleanup candidates instead of previewing them")
@click.option("--confirm", is_flag=True, help="Required with --apply to execute deletions")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def dinger_capture_cleanup(ctx, queue_dir, runtime_dir, apply, confirm, as_json):
    """Preview or delete orphaned capture runtime artifacts using the application cleanup policy."""
    khub = ctx.obj["khub"]
    try:
        if apply and not confirm:
            raise click.ClickException("--confirm is required with --apply")
        payload = cleanup_dinger_capture_runtime(
            config=khub.config,
            dry_run=not apply,
            confirm=bool(confirm),
            queue_dir=queue_dir,
            runtime_dir=runtime_dir,
        )
        payload["schema"] = "knowledge-hub.dinger.capture-cleanup.result.v1"
        _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.capture-cleanup.result.v1")
    except click.ClickException as error:
        payload = _capture_cleanup_failure_payload(
            khub=khub,
            queue_dir=queue_dir,
            runtime_dir=runtime_dir,
            apply=bool(apply),
            confirm=bool(confirm),
            message=str(error),
        )
        _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.capture-cleanup.result.v1")
        if not as_json:
            raise
    if as_json:
        console.print_json(data=payload)
        return
    if str(payload.get("status") or "").strip() == "failed":
        messages = [
            str((entry or {}).get("message") or "").strip()
            for entry in list(payload.get("errors") or [])
            if str((entry or {}).get("message") or "").strip()
        ]
        raise click.ClickException(messages[-1] if messages else "capture cleanup failed")
    counts = dict(payload.get("counts") or {})
    console.print(
        f"[bold]dinger capture cleanup[/bold] dry_run={payload.get('dryRun')} "
        f"delete_eligible={counts.get('deleteEligibleEntries', 0)} "
        f"deleted={counts.get('deletedEntries', 0)} kept={counts.get('keptEntries', 0)} "
        f"failed={counts.get('failedEntries', 0)}"
    )
    for entry in list(payload.get("delete") or []):
        _print_capture_cleanup_entry("delete", dict(entry))
    for entry in list(payload.get("keep") or []):
        _print_capture_cleanup_entry("keep", dict(entry))
    for error in list(payload.get("errors") or []):
        message = str((error or {}).get("message") or "").strip()
        if message:
            console.print(f"error: {message}")


@dinger_capture.command("retry")
@click.option(
    "--queue-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override capture queue directory (default: runtime dinger_capture_intake/queue)",
)
@click.option("--packet", "packet_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--capture-id", default=None, help="Retry a specific failed captureId")
@click.option("--project-id", "--os-project-id", "project_id", default=None, help="Target OS project id")
@click.option("--slug", "--os-slug", "slug", default=None, help="Target OS project slug")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def dinger_capture_retry(ctx, queue_dir, packet_path, capture_id, project_id, slug, as_json):
    """Retry a failed capture packet through the existing processor pipeline."""
    khub = ctx.obj["khub"]
    try:
        if not str(project_id or slug or "").strip():
            raise click.ClickException("--project-id or --slug is required")
        resolved_packet = _resolve_capture_selector(
            khub=khub,
            queue_dir=queue_dir,
            packet_path=packet_path,
            capture_id=capture_id,
        )
        previous_item = _build_capture_read_model_item(khub=khub, packet_path=resolved_packet)
        previous_status = str(previous_item.get("status") or "").strip()
        if previous_status != "failed":
            raise click.ClickException(f"capture retry only supports failed packets; current status={previous_status or 'unknown'}")
        processor = _build_capture_processor(khub=khub, project_id=project_id, slug=slug)
        process_payload = processor.process_packets(
            packet_paths=[resolved_packet],
            queue_dir=Path(queue_dir).expanduser().resolve() if queue_dir is not None else _resolve_capture_queue_dir(khub.config),
        )
        item = dict((process_payload.get("items") or [{}])[0] or {})
        payload = {
            "schema": "knowledge-hub.dinger.capture-retry.result.v1",
            "status": "ok" if str(item.get("status") or "") != "failed" else "failed",
            "captureId": str(previous_item.get("captureId") or ""),
            "packetPath": str(resolved_packet),
            "previousStatus": previous_status,
            "projectId": str(project_id or ""),
            "slug": str(slug or ""),
            "retried": True,
            "item": item,
            "process": process_payload,
            "createdAt": _now_iso(),
        }
        if str(payload.get("status") or "") == "failed":
            payload["error"] = str(item.get("error") or process_payload.get("error") or "capture retry failed")
        _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.capture-retry.result.v1")
    except click.ClickException as error:
        payload = _capture_retry_failure_payload(
            capture_id=str(capture_id or "").strip(),
            packet_path=str(packet_path or ""),
            previous_status="",
            project_id=str(project_id or ""),
            slug=str(slug or ""),
            message=str(error),
        )
        _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.capture-retry.result.v1")
        if not as_json:
            raise
    if as_json:
        console.print_json(data=payload)
        return
    if str(payload.get("status") or "") == "failed":
        raise click.ClickException(str(payload.get("error") or "capture retry failed"))
    item = dict(payload.get("item") or {})
    console.print(
        f"[bold]dinger capture retry[/bold] capture_id={payload.get('captureId')} "
        f"status={item.get('status') or '-'}"
    )


@dinger_group.command("capture-http")
@click.option("--host", default="127.0.0.1", show_default=True, help="Local-only loopback bind host")
@click.option("--port", type=int, default=8765, show_default=True, help="Local-only loopback bind port")
@click.pass_context
def dinger_capture_http(ctx, host, port):
    """Run a local-only HTTP endpoint for browser/clipper capture intake."""
    from knowledge_hub.interfaces.http.dinger_capture_server import serve_dinger_capture_http, validate_local_capture_host

    if int(port) <= 0:
        raise click.ClickException("--port must be positive")
    serve_dinger_capture_http(khub=ctx.obj["khub"], host=validate_local_capture_host(str(host)), port=int(port))


@dinger_group.command("file")
@click.option("--title", default=None, help="Filed page title")
@click.option("--slug", default=None, help="Override page slug")
@click.option("--body", default=None, help="Direct body content to file")
@click.option("--body-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Markdown/text file to file")
@click.option("--from-json", "from_json_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Existing dinger/ask JSON payload to file")
@click.option("--vault-path", default=None, help="Override Obsidian vault path")
@click.option("--backend", default=None, help="Override vault write backend")
@click.option("--cli-binary", default=None, help="Override Obsidian CLI binary")
@click.option("--vault-name", default=None, help="Override Obsidian vault name")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@_source_ref_options
@click.pass_context
def dinger_file(
    ctx,
    title,
    slug,
    body,
    body_file,
    from_json_path,
    vault_path,
    backend,
    cli_binary,
    vault_name,
    as_json,
    source_ref_jsons,
    paper_ids,
    urls,
    note_ids,
    stable_scope_ids,
    document_scope_ids,
):
    """질문 결과나 메모를 managed Dinger page로 filing한다."""
    khub = ctx.obj["khub"]
    try:
        resolved = _resolve_file_input(title=title, body=body, body_file=body_file, from_json_path=from_json_path)
        source_refs = _collect_source_refs(
            source_ref_jsons=source_ref_jsons,
            paper_ids=paper_ids,
            urls=urls,
            note_ids=note_ids,
            stable_scope_ids=stable_scope_ids,
            document_scope_ids=document_scope_ids,
        )
        payload = run_dinger_file_projection(
            khub=khub,
            filing_input=resolved,
            slug=slug,
            source_refs=source_refs,
            vault_path=vault_path,
            backend=backend,
            cli_binary=cli_binary,
            vault_name=vault_name,
        )
        if as_json:
            console.print_json(data=payload)
            return
        console.print(
            f"[bold]dinger file[/bold] kind={payload.get('kind')} "
            f"slug={payload.get('slug')} path={payload.get('relativePath')}"
        )
    except Exception as error:
        message = str(error)
        if not as_json:
            if isinstance(error, click.ClickException):
                raise
            raise click.ClickException(message) from error
        fallback_title = str(title or "").strip()
        fallback_slug = str(slug or "").strip()
        fallback_kind = "note"
        fallback_source_refs = _dedupe_source_refs(
            _collect_source_refs(
                source_ref_jsons=source_ref_jsons,
                paper_ids=paper_ids,
                urls=urls,
                note_ids=note_ids,
                stable_scope_ids=stable_scope_ids,
                document_scope_ids=document_scope_ids,
            )
        )
        fallback_source_schema = ""
        fallback_capture_id = ""
        fallback_packet_path = ""
        fallback_payload: dict[str, Any] = {}
        if from_json_path is not None and from_json_path.exists():
            try:
                raw_payload = json.loads(from_json_path.read_text(encoding="utf-8"))
                if isinstance(raw_payload, dict):
                    fallback_resolved = build_dinger_filing_input_from_payload(
                        payload=raw_payload,
                        input_path=str(from_json_path),
                        title=fallback_title or None,
                    )
                    fallback_trace = dict(fallback_resolved.get("trace") or {})
                    fallback_source_schema = (
                        str(fallback_trace.get("sourceSchema") or raw_payload.get("schema") or "").strip()
                    )
                    fallback_capture_id = str(fallback_trace.get("captureId") or "").strip()
                    fallback_packet_path = str(fallback_trace.get("packetPath") or "").strip()
                    fallback_title = fallback_title or str(fallback_resolved.get("title") or "").strip()
                    fallback_slug = fallback_slug or slugify_title(fallback_title, fallback="dinger-note")
                    fallback_kind = str(fallback_resolved.get("kind") or fallback_kind).strip() or fallback_kind
                    fallback_source_refs = _dedupe_source_refs(
                        list(fallback_resolved.get("sourceRefs") or []) + list(fallback_source_refs or [])
                    )
                    nested_payload = raw_payload.get("payload")
                    if isinstance(nested_payload, dict):
                        fallback_payload = dict(nested_payload)
            except Exception:
                pass
        if not fallback_title and str(body or "").strip():
            fallback_title = "dinger-note"
        if not fallback_slug and fallback_title:
            fallback_slug = slugify_title(fallback_title, fallback="dinger-note")
        failed_payload = _build_file_failure_payload(
            message=message,
            title=fallback_title,
            slug=fallback_slug,
            kind=fallback_kind,
            source_refs=fallback_source_refs,
            vault_path=str(vault_path or khub.config.vault_path or "").strip(),
            backend=str(
                backend
                or khub.config.get_nested("obsidian", "write_backend", default="filesystem")
                or "filesystem"
            ).strip()
            or "filesystem",
            source_schema=fallback_source_schema,
            capture_id=fallback_capture_id,
            packet_path=fallback_packet_path,
            payload=fallback_payload,
        )
        _validate_cli_payload(khub.config, failed_payload, "knowledge-hub.dinger.file.result.v1")
        console.print_json(data=failed_payload)


_CAPTURE_STATUS_ORDER = {
    "captured": 0,
    "normalized": 1,
    "filed": 2,
    "linked_to_os": 3,
    "failed": 99,
}
_CAPTURE_RUNTIME_ARTIFACT_LABELS = {
    "sidecarPath": "runtime",
    "normalizedPath": "normalized",
    "fileResultPath": "filed",
    "osResultPath": "os-linked",
}
_CAPTURE_ORPHAN_FLAGS = ("missing_packet", "orphan_runtime_artifact")
_CAPTURE_ORPHAN_WARNING = "runtime artifacts exist but queue packet is missing"
_CAPTURE_ORPHAN_BLOCKED_RETRY_HINT = "retry is blocked until the canonical queue packet is restored"


def _capture_orphan_operator_guidance(*, config, capture_id: str, status: str) -> dict[str, Any]:
    assessment = assess_capture_requeue_recovery(config=config, capture_id=capture_id)
    if assessment.get("recoverable"):
        action = f"run khub dinger capture requeue --capture-id {capture_id}"
        if str(status or "").strip() == "failed":
            action = f"{action}, then khub dinger capture retry --capture-id {capture_id}"
            retry_hint = "blocked until requeue restores the missing queue packet"
        else:
            action = f"{action}, then resume processing from the restored queue packet"
            retry_hint = "retry is not the next step until the queue packet is restored, and only failed captures can be retried"
        return {
            "operatorSummary": "recoverable orphan: queue packet missing but an exact packet snapshot is available",
            "recoverability": "recoverable",
            "requeueable": True,
            "requeueReason": str(assessment.get("reason") or "").strip(),
            "retryBlocked": True,
            "warning": _CAPTURE_ORPHAN_BLOCKED_RETRY_HINT,
            "operatorAction": action,
            "retryHint": retry_hint,
        }
    return {
        "operatorSummary": "unrecoverable orphan: queue packet missing and no exact packet snapshot is available",
        "recoverability": "unrecoverable",
        "requeueable": False,
        "requeueReason": str(assessment.get("reason") or "").strip(),
        "retryBlocked": True,
        "warning": "retry is blocked because no exact packet snapshot is available",
        "operatorAction": "retry is blocked; inspect capture history or recapture the source, then clean up stale runtime artifacts",
        "retryHint": "blocked until a fresh capture packet exists; requeue cannot recover this orphan",
    }


def _capture_failed_retry_guidance() -> dict[str, Any]:
    return {
        "operatorSummary": "failed capture: queue packet is still present",
        "recoverability": "",
        "requeueable": False,
        "requeueReason": "",
        "retryBlocked": False,
        "warning": "",
        "operatorAction": "fix the underlying failure cause, then run khub dinger capture retry for this capture",
        "retryHint": "ready now because the canonical queue packet is still present",
    }


def _capture_list_badges(item: dict[str, Any]) -> list[str]:
    badges: list[str] = []
    summary = str(item.get("operatorSummary") or "").strip().lower()
    retry_hint = str(item.get("retryHint") or "").strip()
    if "recoverable orphan" in summary:
        badges.append("recoverable orphan")
    elif "unrecoverable orphan" in summary:
        badges.append("unrecoverable orphan")
    elif bool(item.get("orphanedRuntime")):
        badges.append("orphan runtime")
    if retry_hint:
        badges.append(f"retry {retry_hint}")
    elif item.get("retryable"):
        badges.append("retry ready")
    return badges


def _capture_cleanup_entry_hint(entry: dict[str, Any]) -> str:
    cleanup_kind = str(entry.get("cleanupKind") or "").strip()
    if cleanup_kind == "recoverable_orphan":
        return "recoverable orphan: requeue from the exact packet snapshot before cleanup; retry stays blocked until requeue"
    if cleanup_kind in {"orphan_runtime_artifact", "incomplete_runtime_junk"} and str(entry.get("entryType") or "") == "capture_runtime":
        return "unrecoverable orphan: retry cannot resume from these runtime artifacts; inspect or recapture, then clean them up"
    if cleanup_kind == "stale_claim_file":
        return "stale claim: safe to remove; the next processor or retry run can create a fresh claim"
    if cleanup_kind == "active_claim_file":
        return "active claim: leave it in place while a worker may still own this capture"
    if cleanup_kind == "queue_packet_present_capture":
        return "live capture: keep runtime artifacts attached to the current queue packet"
    return ""


def _capture_status_rank(status: str) -> int:
    return _CAPTURE_STATUS_ORDER.get(str(status or "").strip(), -1)


def _capture_runtime_artifact_path(packet_path: Path, label: str) -> Path:
    return packet_path.with_name(f"{packet_path.stem}.{label}.json")


def _capture_runtime_artifacts(packet_path: Path) -> dict[str, str]:
    return {
        key: str(_capture_runtime_artifact_path(packet_path, label))
        for key, label in _CAPTURE_RUNTIME_ARTIFACT_LABELS.items()
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise click.ClickException(f"expected object payload: {path}")
    return payload


def _write_json_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_capture_packet(*, khub, packet_path: Path) -> dict[str, Any]:
    payload = _load_json_object(packet_path)
    if str(payload.get("schema") or "").strip() != "knowledge-hub.dinger.capture.result.v1":
        raise click.ClickException(f"capture packet must use knowledge-hub.dinger.capture.result.v1: {packet_path}")
    _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.capture.result.v1")
    return payload


def _load_capture_runtime_sidecar(packet_path: Path, packet: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    artifacts = _capture_runtime_artifacts(packet_path)
    sidecar_path = Path(artifacts["sidecarPath"])
    state: dict[str, Any] = {}
    if sidecar_path.exists():
        try:
            loaded = _load_json_object(sidecar_path)
            state = dict(loaded)
        except Exception:
            state = {}
    capture_id = str(packet.get("captureId") or "").strip() or packet_path.stem
    state.setdefault("schema", "knowledge-hub.dinger.capture.processor.runtime.v1")
    state.setdefault("captureId", capture_id)
    state.setdefault("idempotencyKey", capture_id or str(packet_path))
    state["packetPath"] = str(packet_path)
    state["artifacts"] = {
        **artifacts,
        **dict(state.get("artifacts") or {}),
    }
    state.setdefault("attempts", {})
    state.setdefault("history", [])
    state.setdefault("warnings", [])
    state.setdefault("createdAt", _now_iso())
    state["status"] = str(state.get("status") or packet.get("status") or "captured").strip() or "captured"
    state["updatedAt"] = _now_iso()
    return state, sidecar_path


def _append_capture_history(state: dict[str, Any], *, status: str, detail: str = "", error: str = "") -> None:
    history = list(state.get("history") or [])
    last = history[-1] if history else {}
    if (
        str(last.get("status") or "").strip() == str(status).strip()
        and str(last.get("detail") or "").strip() == str(detail).strip()
        and str(last.get("error") or "").strip() == str(error).strip()
    ):
        return
    history.append(
        {
            "status": str(status).strip(),
            "detail": str(detail or "").strip(),
            "error": str(error or "").strip(),
            "at": _now_iso(),
        }
    )
    state["history"] = history


def _persist_capture_runtime_state(*, khub, packet_path: Path, packet: dict[str, Any], state: dict[str, Any]) -> None:
    artifacts = {
        **_capture_runtime_artifacts(packet_path),
        **dict((state.get("artifacts") or {})),
    }
    packet["runtime"] = {
        **dict(packet.get("runtime") or {}),
        **artifacts,
    }
    packet["updatedAt"] = _now_iso()
    packet["queueStatus"] = str(packet.get("queueStatus") or packet.get("status") or "").strip()
    state["artifacts"] = artifacts
    state["status"] = str(packet.get("status") or state.get("status") or "").strip() or "captured"
    state["currentStatus"] = state["status"]
    state["updatedAt"] = packet["updatedAt"]
    _validate_cli_payload(khub.config, packet, "knowledge-hub.dinger.capture.result.v1")
    _write_json_object(packet_path, packet)
    _write_json_object(Path(artifacts["sidecarPath"]), state)


def _set_capture_packet_status(
    *,
    packet: dict[str, Any],
    state: dict[str, Any],
    status: str,
    detail: str = "",
    error: str = "",
) -> None:
    resolved_status = str(status or "failed").strip() or "failed"
    packet["status"] = resolved_status
    packet["queueStatus"] = "queued" if resolved_status == "captured" else resolved_status
    if str(error or "").strip():
        packet["error"] = str(error).strip()
        state["lastError"] = str(error).strip()
    else:
        packet.pop("error", None)
        state["lastError"] = ""
    state["status"] = resolved_status
    state["currentStatus"] = resolved_status
    _append_capture_history(state, status=resolved_status, detail=detail, error=error)


def _increment_capture_attempt(state: dict[str, Any], step: str) -> None:
    attempts = dict(state.get("attempts") or {})
    attempts[str(step)] = int(attempts.get(str(step), 0) or 0) + 1
    state["attempts"] = attempts
    state["lastAttemptAt"] = _now_iso()


def _normalize_capture_packet(packet_path: Path) -> dict[str, Any]:
    normalized = _resolve_file_input(title=None, body=None, body_file=None, from_json_path=packet_path)
    capture_payload = {
        "kind": str(normalized.get("kind") or "web_capture"),
        "title": str(normalized.get("title") or "").strip(),
        "slug": slugify_title(str(normalized.get("title") or "").strip(), fallback="dinger-capture"),
        "contentBody": str(normalized.get("contentBody") or "").rstrip(),
        "metadata": dict(normalized.get("metadata") or {}),
        "sourceRefs": list(normalized.get("sourceRefs") or []),
        "trace": dict(normalized.get("trace") or {}),
        "normalizedAt": _now_iso(),
    }
    return capture_payload


def _load_existing_ok_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = _load_json_object(path)
    except Exception:
        return None
    return payload if str(payload.get("status") or "").strip().lower() == "ok" else None


def _process_capture_packet(
    *,
    khub,
    obj: dict[str, Any],
    packet_path: Path,
    os_project_id: str | None,
    os_slug: str | None,
    should_link_to_os: bool,
) -> dict[str, Any]:
    packet = _load_capture_packet(khub=khub, packet_path=packet_path)
    state, _sidecar_path = _load_capture_runtime_sidecar(packet_path, packet)
    artifacts = dict(state.get("artifacts") or {})
    final_status = str(packet.get("status") or "").strip() or "captured"
    error_message = ""
    warnings = list(state.get("warnings") or [])
    normalized_payload: dict[str, Any] | None = None
    file_result: dict[str, Any] | None = None
    os_result: dict[str, Any] | None = None

    try:
        normalized_path = Path(artifacts["normalizedPath"])
        if _capture_status_rank(packet.get("status", "")) < _capture_status_rank("normalized") or not normalized_path.exists():
            _increment_capture_attempt(state, "normalize")
            normalized_payload = _normalize_capture_packet(packet_path)
            _write_json_object(normalized_path, normalized_payload)
            state["normalized"] = {
                "title": str(normalized_payload.get("title") or "").strip(),
                "kind": str(normalized_payload.get("kind") or "").strip(),
                "sourceRefCount": len(list(normalized_payload.get("sourceRefs") or [])),
                "path": str(normalized_path),
            }
            _set_capture_packet_status(packet=packet, state=state, status="normalized", detail="normalized capture packet")
            _persist_capture_runtime_state(khub=khub, packet_path=packet_path, packet=packet, state=state)
        else:
            normalized_payload = _load_existing_ok_result(normalized_path) or _load_json_object(normalized_path)

        file_result_path = Path(artifacts["fileResultPath"])
        file_result = _load_existing_ok_result(file_result_path)
        if file_result is None:
            _increment_capture_attempt(state, "file")
            file_result = run_dinger_file_projection(
                khub=khub,
                filing_input=dict(normalized_payload or {}),
            )
            if str(file_result.get("status") or "").strip().lower() != "ok":
                raise click.ClickException(str(file_result.get("error") or "dinger file returned non-ok status"))
        if not str((file_result or {}).get("captureUrl") or "").strip():
            normalized_capture_url = str((normalized_payload or {}).get("trace", {}).get("captureUrl") or "").strip()
            if not normalized_capture_url:
                normalized_capture_url = str((normalized_payload or {}).get("metadata", {}).get("captureUrl") or "").strip()
            if normalized_capture_url:
                file_result["captureUrl"] = normalized_capture_url
        _write_json_object(file_result_path, file_result)
        state["fileResult"] = {
            "path": str(file_result_path),
            "relativePath": str(file_result.get("relativePath") or "").strip(),
            "filePath": str(file_result.get("filePath") or "").strip(),
            "kind": str(file_result.get("kind") or "").strip(),
        }
        _set_capture_packet_status(packet=packet, state=state, status="filed", detail="filed capture into dinger page")
        _persist_capture_runtime_state(khub=khub, packet_path=packet_path, packet=packet, state=state)

        if should_link_to_os:
            os_result_path = Path(artifacts["osResultPath"])
            if str(packet.get("status") or "").strip() == "linked_to_os" and not os_result_path.exists():
                warnings.append("packet already marked linked_to_os but os result artifact is missing; skipped rerun")
            else:
                os_result = _load_existing_ok_result(os_result_path)
                if os_result is None:
                    _increment_capture_attempt(state, "link_to_os")
                    from knowledge_hub.interfaces.cli.commands.os_cmd import _ops_alerts_json

                    os_result = bridge_dinger_result_to_os_capture(
                        dinger_payload=file_result,
                        project_id=os_project_id,
                        slug=os_slug,
                        summary=None,
                        kind=None,
                        severity="medium",
                        extra_source_refs=[],
                        ops_alerts_json=_ops_alerts_json(khub),
                        runner=run_foundry_project_cli,
                    )
                    if str(os_result.get("status") or "").strip().lower() != "ok":
                        raise click.ClickException(str(os_result.get("error") or "os capture returned non-ok status"))
                    _write_json_object(os_result_path, os_result)
                state["osResult"] = {
                    "path": str(os_result_path),
                    "itemId": str(((os_result or {}).get("item") or {}).get("id") or "").strip(),
                }
                state["osBridge"] = dict((os_result or {}).get("captureTrace") or {})
                _set_capture_packet_status(
                    packet=packet,
                    state=state,
                    status="linked_to_os",
                    detail="linked filed capture to os inbox/evidence",
                )
                _persist_capture_runtime_state(khub=khub, packet_path=packet_path, packet=packet, state=state)

        state["warnings"] = warnings
        final_status = str(packet.get("status") or "").strip() or "captured"
    except Exception as error:
        error_message = str(error)
        _set_capture_packet_status(packet=packet, state=state, status="failed", detail="processor failed", error=error_message)
        state["warnings"] = warnings
        _persist_capture_runtime_state(khub=khub, packet_path=packet_path, packet=packet, state=state)
        final_status = "failed"

    return {
        "captureId": str(packet.get("captureId") or "").strip(),
        "packetPath": str(packet_path),
        "status": final_status,
        "queueStatus": str(packet.get("queueStatus") or "").strip(),
        "sidecarPath": str(artifacts["sidecarPath"]),
        "normalizedPath": str(artifacts["normalizedPath"]) if Path(artifacts["normalizedPath"]).exists() else "",
        "fileResultPath": str(artifacts["fileResultPath"]) if Path(artifacts["fileResultPath"]).exists() else "",
        "osResultPath": str(artifacts["osResultPath"]) if Path(artifacts["osResultPath"]).exists() else "",
        "error": error_message,
        "warnings": warnings,
    }


def _select_capture_packet_paths(
    *,
    khub,
    queue_dir: Path | None,
    packet_paths: tuple[Path, ...],
    capture_ids: tuple[str, ...],
    limit: int,
) -> list[Path]:
    resolved_queue_dir = Path(queue_dir).expanduser().resolve() if queue_dir is not None else _resolve_capture_queue_dir(khub.config)
    selected: dict[str, Path] = {}
    for packet_path in packet_paths:
        resolved = packet_path.expanduser().resolve()
        selected[str(resolved)] = resolved
    for capture_id in capture_ids:
        text = str(capture_id or "").strip()
        if not text:
            continue
        resolved = (resolved_queue_dir / f"{text}.json").resolve()
        if not resolved.exists():
            if _has_capture_runtime_artifacts(khub.config, text, queue_dir=resolved_queue_dir):
                orphan_item = _build_capture_read_model_item(khub=khub, capture_id=text, queue_dir=resolved_queue_dir)
                orphan_guidance = _capture_orphan_operator_guidance(
                    config=khub.config,
                    capture_id=text,
                    status=str(orphan_item.get("status") or "").strip(),
                )
                raise click.ClickException(
                    f"capture packet missing for captureId={text}: runtime artifacts exist, "
                    f"but retry/process still require the queue packet ({resolved}); "
                    f"{orphan_guidance['operatorAction']}"
                )
            raise click.ClickException(f"capture packet not found for captureId={text}: {resolved}")
        selected[str(resolved)] = resolved
    if not selected:
        if resolved_queue_dir.exists():
            for candidate in sorted(resolved_queue_dir.glob("*.json")):
                if candidate.name.endswith((".runtime.json", ".normalized.json", ".filed.json", ".os-linked.json")):
                    continue
                selected[str(candidate.resolve())] = candidate.resolve()
    paths = list(selected.values())
    if int(limit or 0) > 0:
        return paths[: max(1, int(limit))]
    return paths


def _capture_runtime_paths_for_id(config, capture_id: str) -> dict[str, Path]:
    runtime_dir = resolve_capture_runtime_dir(config)
    return {
        "runtimeDir": runtime_dir,
        "statePath": runtime_dir / f"{capture_id}.state.json",
        "normalizedPath": runtime_dir / f"{capture_id}.normalized.json",
        "filedResultPath": runtime_dir / f"{capture_id}.file-result.json",
        "osResultPath": runtime_dir / f"{capture_id}.os-capture-result.json",
    }


def _resolve_capture_read_model_queue_dir(*, khub, queue_dir: Path | None) -> Path:
    return Path(queue_dir).expanduser().resolve() if queue_dir is not None else _resolve_capture_queue_dir(khub.config)


def _collect_capture_read_model_items(
    *,
    khub,
    queue_dir: Path | None,
    statuses: tuple[str, ...] | list[str] = (),
    limit: int = 0,
) -> tuple[list[dict[str, Any]], Path, Path]:
    packets = _select_capture_packet_paths(
        khub=khub,
        queue_dir=queue_dir,
        packet_paths=(),
        capture_ids=(),
        limit=0,
    )
    normalized_statuses = {str(status).strip().lower() for status in statuses if str(status).strip()}
    items = [_build_capture_read_model_item(khub=khub, packet_path=packet_path, queue_dir=queue_dir) for packet_path in packets]
    known_capture_ids = {str(item.get("captureId") or "").strip() for item in items if str(item.get("captureId") or "").strip()}
    runtime_only_capture_ids = _discover_runtime_only_capture_ids(
        khub=khub,
        known_capture_ids=known_capture_ids,
        queue_dir=queue_dir,
    )
    items.extend(
        _build_capture_read_model_item(khub=khub, capture_id=capture_id, queue_dir=queue_dir)
        for capture_id in runtime_only_capture_ids
    )
    if normalized_statuses:
        items = [item for item in items if str(item.get("status") or "").strip().lower() in normalized_statuses]
    items.sort(key=lambda item: (str(item.get("updatedAt") or ""), str(item.get("captureId") or "")), reverse=True)
    if int(limit or 0) > 0:
        items = items[: max(1, int(limit))]
    resolved_queue_dir = _resolve_capture_read_model_queue_dir(khub=khub, queue_dir=queue_dir)
    resolved_runtime_dir = resolve_capture_runtime_dir(khub.config)
    return items, resolved_queue_dir, resolved_runtime_dir


def _default_capture_packet_path(config, capture_id: str, *, queue_dir: Path | None = None) -> Path:
    resolved_queue_dir = Path(queue_dir).expanduser().resolve() if queue_dir is not None else _resolve_capture_queue_dir(config)
    return resolved_queue_dir / f"{capture_id}.json"


def _runtime_capture_suffixes() -> tuple[str, ...]:
    return (".state.json", ".normalized.json", ".file-result.json", ".os-capture-result.json")


def _legacy_runtime_capture_suffixes() -> tuple[str, ...]:
    return (".runtime.json", ".normalized.json", ".filed.json", ".os-linked.json")


def _has_capture_runtime_artifacts(config, capture_id: str, *, queue_dir: Path | None = None) -> bool:
    runtime_paths = _capture_runtime_paths_for_id(config, capture_id)
    packet_path = _default_capture_packet_path(config, capture_id, queue_dir=queue_dir)
    legacy_paths = (
        _capture_runtime_artifact_path(packet_path, "runtime"),
        _capture_runtime_artifact_path(packet_path, "normalized"),
        _capture_runtime_artifact_path(packet_path, "filed"),
        _capture_runtime_artifact_path(packet_path, "os-linked"),
    )
    return any(Path(path).exists() for key, path in runtime_paths.items() if key != "runtimeDir") or any(
        path.exists() for path in legacy_paths
    )


def _discover_runtime_only_capture_ids(*, khub, known_capture_ids: set[str], queue_dir: Path | None = None) -> list[str]:
    runtime_dir = resolve_capture_runtime_dir(khub.config)
    capture_ids: set[str] = set()
    if runtime_dir.exists():
        for suffix in _runtime_capture_suffixes():
            for candidate in runtime_dir.glob(f"*{suffix}"):
                if not candidate.is_file():
                    continue
                capture_id = candidate.name[: -len(suffix)].strip()
                if capture_id and capture_id not in known_capture_ids:
                    capture_ids.add(capture_id)
    resolved_queue_dir = Path(queue_dir).expanduser().resolve() if queue_dir is not None else _resolve_capture_queue_dir(khub.config)
    if resolved_queue_dir.exists():
        for suffix in _legacy_runtime_capture_suffixes():
            for candidate in resolved_queue_dir.glob(f"*{suffix}"):
                if not candidate.is_file():
                    continue
                capture_id = candidate.name[: -len(suffix)].strip()
                if capture_id and capture_id not in known_capture_ids:
                    capture_ids.add(capture_id)
    return sorted(capture_ids)


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _load_json_object(path)
    except Exception:
        return None


def _capture_path_or_empty(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return str(path)


def _resolve_capture_selector(
    *,
    khub,
    queue_dir: Path | None,
    packet_path: Path | None,
    capture_id: str | None,
) -> Path:
    explicit_packet = packet_path is not None
    explicit_capture_id = bool(str(capture_id or "").strip())
    if explicit_packet == explicit_capture_id:
        raise click.ClickException("choose exactly one selector: --packet or --capture-id")
    if explicit_packet:
        return Path(packet_path).expanduser().resolve()
    return _select_capture_packet_paths(
        khub=khub,
        queue_dir=queue_dir,
        packet_paths=(),
        capture_ids=(str(capture_id or "").strip(),),
        limit=1,
    )[0]


def _resolve_capture_read_model_selector(
    *,
    khub,
    queue_dir: Path | None,
    packet_path: Path | None,
    capture_id: str | None,
) -> tuple[str, Path | None]:
    explicit_packet = packet_path is not None
    explicit_capture_id = bool(str(capture_id or "").strip())
    if explicit_packet == explicit_capture_id:
        raise click.ClickException("choose exactly one selector: --packet or --capture-id")
    if explicit_packet:
        resolved = Path(packet_path).expanduser().resolve()
        packet = _load_capture_packet(khub=khub, packet_path=resolved)
        resolved_capture_id = str(packet.get("captureId") or "").strip() or resolved.stem
        return resolved_capture_id, resolved

    resolved_capture_id = str(capture_id or "").strip()
    resolved_queue_dir = _resolve_capture_read_model_queue_dir(khub=khub, queue_dir=queue_dir)
    resolved_packet = (resolved_queue_dir / f"{resolved_capture_id}.json").resolve()
    if resolved_packet.exists():
        return resolved_capture_id, resolved_packet
    if _has_capture_runtime_artifacts(khub.config, resolved_capture_id, queue_dir=resolved_queue_dir):
        return resolved_capture_id, None
    raise click.ClickException(f"capture not found for captureId={resolved_capture_id}: no packet or runtime artifacts")


def _derive_capture_read_model_status(
    *,
    packet: dict[str, Any],
    runtime_state: dict[str, Any] | None,
    normalized: dict[str, Any] | None,
    filed_result: dict[str, Any] | None,
    os_result: dict[str, Any] | None,
) -> str:
    packet_status = str(packet.get("status") or "").strip()
    runtime_status = ""
    if isinstance(runtime_state, dict):
        runtime_status = str(runtime_state.get("currentStatus") or runtime_state.get("status") or "").strip()

    if packet_status == "failed" or runtime_status == "failed":
        return "failed"
    if isinstance(os_result, dict) and is_capture_linked_to_os(os_result):
        return "linked_to_os"
    if runtime_status == "linked_to_os":
        return "linked_to_os"
    if runtime_status == "filed" or isinstance(filed_result, dict):
        return "filed"
    if runtime_status in {"captured", "normalized"} or runtime_state or normalized:
        return "processing"
    return "queued"


def _capture_attempt_count(runtime_state: dict[str, Any] | None) -> int:
    if not isinstance(runtime_state, dict):
        return 0
    explicit = int(runtime_state.get("attemptCount") or 0)
    if explicit > 0:
        return explicit
    attempts = runtime_state.get("attempts")
    if not isinstance(attempts, dict):
        return 0
    return sum(int(value or 0) for value in attempts.values())


def _build_capture_read_model_item(
    *,
    khub,
    packet_path: Path | None = None,
    capture_id: str | None = None,
    queue_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_packet_path = Path(packet_path).expanduser().resolve() if packet_path is not None else None
    packet: dict[str, Any] = {}
    if resolved_packet_path is not None and resolved_packet_path.exists():
        packet = _load_capture_packet(khub=khub, packet_path=resolved_packet_path)
    resolved_capture_id = str((packet.get("captureId") if packet else "") or capture_id or "").strip()
    if not resolved_capture_id:
        if resolved_packet_path is None:
            raise click.ClickException("captureId is required when packet_path is missing")
        resolved_capture_id = resolved_packet_path.stem

    runtime_paths = _capture_runtime_paths_for_id(khub.config, resolved_capture_id)
    legacy_paths = {
        "statePath": _capture_runtime_artifact_path(resolved_packet_path, "runtime") if resolved_packet_path is not None else None,
        "normalizedPath": _capture_runtime_artifact_path(resolved_packet_path, "normalized") if resolved_packet_path is not None else None,
        "filedResultPath": _capture_runtime_artifact_path(resolved_packet_path, "filed") if resolved_packet_path is not None else None,
        "osResultPath": _capture_runtime_artifact_path(resolved_packet_path, "os-linked") if resolved_packet_path is not None else None,
    }

    def _load_runtime_payload(primary: Path, legacy: Path | None) -> dict[str, Any] | None:
        return _load_optional_json(primary) or (_load_optional_json(legacy) if legacy is not None else None)

    runtime_state = _load_runtime_payload(runtime_paths["statePath"], legacy_paths["statePath"])
    normalized = _load_runtime_payload(runtime_paths["normalizedPath"], legacy_paths["normalizedPath"])
    filed_result = _load_runtime_payload(runtime_paths["filedResultPath"], legacy_paths["filedResultPath"])
    os_result = _load_runtime_payload(runtime_paths["osResultPath"], legacy_paths["osResultPath"])

    def _resolved_artifact_path(primary: Path, legacy: Path | None) -> Path | None:
        if primary.exists():
            return primary
        if legacy is not None and legacy.exists():
            return legacy
        return primary if legacy is None else legacy

    state_path = _resolved_artifact_path(runtime_paths["statePath"], legacy_paths["statePath"])
    normalized_path = _resolved_artifact_path(runtime_paths["normalizedPath"], legacy_paths["normalizedPath"])
    filed_result_path = _resolved_artifact_path(runtime_paths["filedResultPath"], legacy_paths["filedResultPath"])
    os_result_path = _resolved_artifact_path(runtime_paths["osResultPath"], legacy_paths["osResultPath"])

    runtime_status = ""
    attempt_count = 0
    last_error = ""
    if isinstance(runtime_state, dict):
        runtime_status = str(runtime_state.get("currentStatus") or runtime_state.get("status") or "").strip()
        attempt_count = _capture_attempt_count(runtime_state)
        last_error = str(runtime_state.get("lastError") or "").strip()
        if not last_error:
            last_error = str(((runtime_state.get("steps") or {}).get("failed") or {}).get("error") or "").strip()

    status = _derive_capture_read_model_status(
        packet=packet,
        runtime_state=runtime_state,
        normalized=normalized,
        filed_result=filed_result,
        os_result=os_result,
    )
    updated_at = str(
        (runtime_state or {}).get("updatedAt")
        or (os_result or {}).get("createdAt")
        or (filed_result or {}).get("createdAt")
        or (normalized or {}).get("normalizedAt")
        or packet.get("updatedAt")
        or packet.get("createdAt")
        or ""
    ).strip()
    packet_present = bool(packet)
    packet_path_text = str(resolved_packet_path) if resolved_packet_path is not None else ""
    if not packet_path_text:
        packet_path_text = str((runtime_state or {}).get("packetPath") or "").strip()
    if not packet_path_text:
        packet_path_text = str(_default_capture_packet_path(khub.config, resolved_capture_id, queue_dir=queue_dir))
    orphaned_runtime = (not packet_present) and any(
        path and Path(path).exists()
        for path in (state_path, normalized_path, filed_result_path, os_result_path)
    )
    artifact_kinds = [
        label
        for label, path in (
            ("runtime_state", state_path),
            ("normalized", normalized_path),
            ("file_result", filed_result_path),
            ("os_result", os_result_path),
        )
        if path and Path(path).exists()
    ]
    flags: list[str] = []
    warnings = [
        str(entry).strip()
        for entry in list((runtime_state or {}).get("warnings") or [])
        if str(entry).strip()
    ]
    operator_summary = ""
    operator_action = ""
    retry_hint = ""
    recoverability = ""
    requeueable = False
    requeue_reason = ""
    retry_blocked = False
    if orphaned_runtime:
        orphan_guidance = _capture_orphan_operator_guidance(
            config=khub.config,
            capture_id=resolved_capture_id,
            status=status,
        )
        flags.extend(_CAPTURE_ORPHAN_FLAGS)
        warnings.append(_CAPTURE_ORPHAN_WARNING)
        warnings.append(orphan_guidance["warning"])
        operator_summary = orphan_guidance["operatorSummary"]
        operator_action = orphan_guidance["operatorAction"]
        retry_hint = orphan_guidance["retryHint"]
        recoverability = str(orphan_guidance.get("recoverability") or "").strip()
        requeueable = bool(orphan_guidance.get("requeueable"))
        requeue_reason = str(orphan_guidance.get("requeueReason") or "").strip()
        retry_blocked = bool(orphan_guidance.get("retryBlocked"))
    elif status == "failed":
        failed_guidance = _capture_failed_retry_guidance()
        operator_summary = failed_guidance["operatorSummary"]
        operator_action = failed_guidance["operatorAction"]
        retry_hint = failed_guidance["retryHint"]
        recoverability = str(failed_guidance.get("recoverability") or "").strip()
        requeueable = bool(failed_guidance.get("requeueable"))
        requeue_reason = str(failed_guidance.get("requeueReason") or "").strip()
        retry_blocked = bool(failed_guidance.get("retryBlocked"))
    warnings = list(dict.fromkeys(warnings))
    title = str(
        packet.get("pageTitle")
        or packet.get("title")
        or (normalized or {}).get("title")
        or ((runtime_state or {}).get("normalized") or {}).get("title")
        or (filed_result or {}).get("title")
        or ""
    ).strip()
    source_url = str(
        packet.get("sourceUrl")
        or packet.get("captureUrl")
        or (normalized or {}).get("captureUrl")
        or (normalized or {}).get("sourceUrl")
        or ((normalized or {}).get("trace") or {}).get("captureUrl")
        or ((normalized or {}).get("metadata") or {}).get("captureUrl")
        or (filed_result or {}).get("captureUrl")
        or ""
    ).strip()
    client = str(packet.get("client") or (normalized or {}).get("client") or "").strip()
    captured_at = str(
        packet.get("capturedAt") or (normalized or {}).get("capturedAt") or (runtime_state or {}).get("capturedAt") or ""
    ).strip()
    tags = [
        str(item).strip()
        for item in list(packet.get("tags") or (normalized or {}).get("tags") or (runtime_state or {}).get("tags") or [])
        if str(item).strip()
    ]

    return {
        "captureId": resolved_capture_id,
        "status": status,
        "packetStatus": str(packet.get("status") or "").strip(),
        "queueStatus": str(packet.get("queueStatus") or "").strip(),
        "runtimeStatus": runtime_status,
        "accepted": bool(packet.get("accepted")),
        "packetPresent": packet_present,
        "packetMissing": not packet_present,
        "orphanedRuntime": orphaned_runtime,
        "artifactKinds": artifact_kinds,
        "flags": flags,
        "sourceUrl": source_url,
        "title": title,
        "client": client,
        "capturedAt": captured_at,
        "tags": tags,
        "packetPath": packet_path_text,
        "statePath": _capture_path_or_empty(state_path),
        "normalizedPath": _capture_path_or_empty(normalized_path),
        "filedResultPath": _capture_path_or_empty(filed_result_path),
        "osResultPath": _capture_path_or_empty(os_result_path),
        "filedRelativePath": str((filed_result or {}).get("relativePath") or "").strip(),
        "osItemId": str((((os_result or {}).get("item") or {}).get("id")) or "").strip(),
        "attemptCount": attempt_count,
        "lastError": last_error,
        "warnings": warnings,
        "retryable": packet_present and status == "failed",
        "operatorSummary": operator_summary,
        "recoverability": recoverability,
        "requeueable": requeueable,
        "requeueReason": requeue_reason,
        "retryBlocked": retry_blocked,
        "operatorAction": operator_action,
        "retryHint": retry_hint,
        "updatedAt": updated_at,
    }


def _append_capture_id_sample(samples: list[str], capture_id: str, *, limit: int = 5) -> None:
    token = str(capture_id or "").strip()
    if not token or token in samples or len(samples) >= limit:
        return
    samples.append(token)


def _build_capture_status_payload(*, khub, queue_dir: Path | None, runtime_dir: Path | None) -> dict[str, Any]:
    items, resolved_queue_dir, resolved_runtime_dir = _collect_capture_read_model_items(
        khub=khub,
        queue_dir=queue_dir,
        statuses=(),
        limit=0,
    )
    cleanup_payload = cleanup_dinger_capture_runtime(
        config=khub.config,
        dry_run=True,
        confirm=False,
        queue_dir=resolved_queue_dir,
        runtime_dir=runtime_dir or resolved_runtime_dir,
    )
    claim_entries = [
        dict(entry)
        for entry in [*list(cleanup_payload.get("delete") or []), *list(cleanup_payload.get("keep") or [])]
        if str((entry or {}).get("entryType") or "").strip() == "claim_file"
    ]
    counts = {
        "totalCaptures": len(items),
        "queued": 0,
        "processing": 0,
        "filed": 0,
        "linkedToOs": 0,
        "failed": 0,
        "orphanedRuntime": 0,
        "recoverableOrphans": 0,
        "unrecoverableOrphans": 0,
        "retryReady": 0,
        "retryBlocked": 0,
        "staleClaimFiles": 0,
        "activeClaimFiles": 0,
        "invalidClaimFiles": 0,
        "cleanupDeleteEligibleEntries": 0,
    }
    samples = {
        "recoverableOrphanCaptureIds": [],
        "unrecoverableOrphanCaptureIds": [],
        "retryReadyCaptureIds": [],
        "staleClaimCaptureIds": [],
    }
    for item in items:
        status = str(item.get("status") or "").strip()
        if status in CAPTURE_INSPECTABLE_STATUSES:
            key = "linkedToOs" if status == "linked_to_os" else status
            counts[key] = int(counts.get(key) or 0) + 1
        capture_id = str(item.get("captureId") or "").strip()
        if bool(item.get("orphanedRuntime")):
            counts["orphanedRuntime"] += 1
        recoverability = str(item.get("recoverability") or "").strip()
        if recoverability == "recoverable":
            counts["recoverableOrphans"] += 1
            _append_capture_id_sample(samples["recoverableOrphanCaptureIds"], capture_id)
        elif recoverability == "unrecoverable":
            counts["unrecoverableOrphans"] += 1
            _append_capture_id_sample(samples["unrecoverableOrphanCaptureIds"], capture_id)
        if bool(item.get("retryable")):
            counts["retryReady"] += 1
            _append_capture_id_sample(samples["retryReadyCaptureIds"], capture_id)
        if bool(item.get("retryBlocked")):
            counts["retryBlocked"] += 1
    for entry in claim_entries:
        cleanup_kind = str(entry.get("cleanupKind") or "").strip()
        capture_id = str(entry.get("captureId") or "").strip()
        if cleanup_kind == "stale_claim_file":
            counts["staleClaimFiles"] += 1
            _append_capture_id_sample(samples["staleClaimCaptureIds"], capture_id)
        elif cleanup_kind == "active_claim_file":
            counts["activeClaimFiles"] += 1
        elif cleanup_kind == "incomplete_runtime_junk":
            counts["invalidClaimFiles"] += 1
    counts["cleanupDeleteEligibleEntries"] = (
        counts["unrecoverableOrphans"] + counts["staleClaimFiles"] + counts["invalidClaimFiles"]
    )

    actions: list[dict[str, Any]] = []
    if counts["recoverableOrphans"] > 0:
        actions.append(
            {
                "kind": "requeue_recoverable_orphans",
                "count": counts["recoverableOrphans"],
                "command": "khub dinger capture requeue --capture-id <capture-id>",
                "summary": "restore missing queue packets for recoverable orphan captures before retry or process",
                "sampleCaptureIds": list(samples["recoverableOrphanCaptureIds"]),
            }
        )
    if counts["retryReady"] > 0:
        actions.append(
            {
                "kind": "retry_failed_captures",
                "count": counts["retryReady"],
                "command": "khub dinger capture retry --capture-id <capture-id> --project-id <id>|--slug <slug>",
                "summary": "replay failed captures that still have a canonical queue packet",
                "sampleCaptureIds": list(samples["retryReadyCaptureIds"]),
            }
        )
    cleanup_action_count = counts["unrecoverableOrphans"] + counts["staleClaimFiles"] + counts["invalidClaimFiles"]
    if cleanup_action_count > 0:
        cleanup_samples = list(
            dict.fromkeys(
                [
                    *list(samples["unrecoverableOrphanCaptureIds"]),
                    *list(samples["staleClaimCaptureIds"]),
                ]
            )
        )[:5]
        actions.append(
            {
                "kind": "cleanup_runtime_artifacts",
                "count": cleanup_action_count,
                "command": "khub dinger capture cleanup",
                "summary": "preview runtime-only cleanup targets for unrecoverable orphans, stale claims, or invalid claim files",
                "sampleCaptureIds": cleanup_samples,
            }
        )

    return {
        "schema": "knowledge-hub.dinger.capture-status.result.v1",
        "status": "ok",
        "queueDir": str(resolved_queue_dir),
        "runtimeDir": str(Path(runtime_dir).expanduser().resolve() if runtime_dir is not None else resolved_runtime_dir),
        "counts": counts,
        "samples": samples,
        "actions": actions,
        "createdAt": _now_iso(),
    }


def _capture_show_failure_payload(*, capture_id: str = "", packet_path: str = "", message: str) -> dict[str, Any]:
    return {
        "schema": "knowledge-hub.dinger.capture-show.result.v1",
        "status": "failed",
        "captureId": str(capture_id or "").strip(),
        "packetPath": str(packet_path or "").strip(),
        "error": str(message or "").strip() or "unknown error",
        "createdAt": _now_iso(),
    }


def _capture_retry_failure_payload(
    *,
    capture_id: str = "",
    packet_path: str = "",
    previous_status: str = "",
    project_id: str = "",
    slug: str = "",
    message: str,
) -> dict[str, Any]:
    return {
        "schema": "knowledge-hub.dinger.capture-retry.result.v1",
        "status": "failed",
        "captureId": str(capture_id or "").strip(),
        "packetPath": str(packet_path or "").strip(),
        "previousStatus": str(previous_status or "").strip(),
        "projectId": str(project_id or "").strip(),
        "slug": str(slug or "").strip(),
        "retried": False,
        "error": str(message or "").strip() or "unknown error",
        "createdAt": _now_iso(),
    }


def _capture_requeue_failure_payload(
    *,
    capture_id: str = "",
    packet_path: str = "",
    restored_from: str = "",
    warnings: list[str] | None = None,
    message: str,
) -> dict[str, Any]:
    return {
        "schema": "knowledge-hub.dinger.capture-requeue.result.v1",
        "status": "failed",
        "result": "failed",
        "captureId": str(capture_id or "").strip(),
        "packetPath": str(packet_path or "").strip(),
        "restoredFrom": str(restored_from or "").strip(),
        "requeued": False,
        "warnings": [str(entry).strip() for entry in list(warnings or []) if str(entry).strip()],
        "error": str(message or "").strip() or "unknown error",
        "createdAt": _now_iso(),
    }


def _capture_cleanup_failure_payload(
    *,
    khub,
    queue_dir: Path | None,
    runtime_dir: Path | None,
    apply: bool,
    confirm: bool,
    message: str,
) -> dict[str, Any]:
    try:
        payload = cleanup_dinger_capture_runtime(
            config=khub.config,
            dry_run=True,
            confirm=False,
            queue_dir=queue_dir,
            runtime_dir=runtime_dir,
        )
    except Exception:
        payload = {
            "queueDir": str(Path(queue_dir).expanduser().resolve()) if queue_dir is not None else str(resolve_capture_queue_dir(khub.config)),
            "runtimeDir": str(Path(runtime_dir).expanduser().resolve()) if runtime_dir is not None else str(resolve_capture_runtime_dir(khub.config)),
            "delete": [],
            "keep": [],
            "errors": [],
            "counts": {
                "scannedCaptureGroups": 0,
                "scannedClaimFiles": 0,
                "deleteEligibleEntries": 0,
                "deleteEligiblePaths": 0,
                "deletedEntries": 0,
                "deletedPaths": 0,
                "keptEntries": 0,
                "failedEntries": 0,
            },
        }
    payload["schema"] = "knowledge-hub.dinger.capture-cleanup.result.v1"
    payload["status"] = "failed"
    payload["dryRun"] = not bool(apply)
    payload["confirmed"] = bool(confirm)
    errors = list(payload.get("errors") or [])
    first_entry = dict((list(payload.get("delete") or []) or [{}])[0] or {})
    errors.append(
        {
            "entryType": str(first_entry.get("entryType") or "capture_runtime"),
            "captureId": str(first_entry.get("captureId") or ""),
            "message": str(message or "").strip() or "unknown error",
        }
    )
    payload["errors"] = errors
    payload["createdAt"] = _now_iso()
    return payload


def _capture_cleanup_path_count(entry: dict[str, Any]) -> int:
    if str(entry.get("entryType") or "").strip() == "capture_runtime":
        return len(list(entry.get("runtimeArtifactPaths") or []))
    deleted_paths = list(entry.get("deletedPaths") or [])
    if deleted_paths:
        return len(deleted_paths)
    return 1 if str(entry.get("claimPath") or "").strip() else 0


def _print_capture_cleanup_entry(prefix: str, entry: dict[str, Any]) -> None:
    line = (
        f"{prefix} {entry.get('entryType') or '-'} {entry.get('captureId') or '-'} "
        f"{entry.get('cleanupKind') or '-'} action={entry.get('action') or '-'} "
        f"paths={_capture_cleanup_path_count(entry)}"
    )
    hint = _capture_cleanup_entry_hint(entry)
    if hint:
        line = f"{line} hint={hint}"
    console.print(line, soft_wrap=True)


def _capture_snapshot_hint_paths(payload: dict[str, Any] | None) -> list[Path]:
    if not isinstance(payload, dict):
        return []
    candidates: list[Path] = []
    seen: set[str] = set()

    def _append(candidate: Any) -> None:
        text = str(candidate or "").strip()
        if not text:
            return
        resolved = Path(text).expanduser().resolve()
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        candidates.append(resolved)

    for key in ("packetSnapshotPath", "snapshotPath"):
        _append(payload.get(key))
    for nested_key in ("artifacts", "recovery", "traceability"):
        nested = payload.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in ("packetSnapshotPath", "snapshotPath"):
            _append(nested.get(key))
    return candidates


def _capture_snapshot_candidate_paths(*, khub, capture_id: str, queue_dir: Path | None = None) -> list[Path]:
    runtime_paths = _capture_runtime_paths_for_id(khub.config, capture_id)
    resolved_queue_dir = _resolve_capture_read_model_queue_dir(khub=khub, queue_dir=queue_dir)
    default_packet_path = _default_capture_packet_path(khub.config, capture_id, queue_dir=resolved_queue_dir)
    legacy_state_path = _capture_runtime_artifact_path(default_packet_path, "runtime")
    state_payload = _load_optional_json(runtime_paths["statePath"]) or _load_optional_json(legacy_state_path)
    packet_hint = str((state_payload or {}).get("packetPath") or "").strip()
    packet_hint_path = Path(packet_hint).expanduser().resolve() if packet_hint else None
    candidates: list[Path] = []
    seen: set[str] = set()

    def _append(candidate: Path | None) -> None:
        if candidate is None:
            return
        resolved = candidate.expanduser().resolve()
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        candidates.append(resolved)

    for candidate in _capture_snapshot_hint_paths(state_payload):
        _append(candidate)
    for base_dir in (runtime_paths["runtimeDir"], resolved_queue_dir):
        for suffix in ("packet-snapshot", "packet.snapshot", "snapshot"):
            _append(base_dir / f"{capture_id}.{suffix}.json")
    for packet_base in (default_packet_path, packet_hint_path):
        if packet_base is None:
            continue
        for suffix in ("packet-snapshot", "packet.snapshot", "snapshot"):
            _append(packet_base.with_name(f"{packet_base.stem}.{suffix}.json"))
    return candidates


def _find_capture_packet_snapshot_path(*, khub, capture_id: str, queue_dir: Path | None = None) -> Path | None:
    for candidate in _capture_snapshot_candidate_paths(khub=khub, capture_id=capture_id, queue_dir=queue_dir):
        if candidate.exists():
            return candidate
    return None


def _load_capture_packet_snapshot(*, khub, packet_snapshot_path: Path) -> dict[str, Any]:
    payload = _load_json_object(packet_snapshot_path)
    if str(payload.get("schema") or "").strip() != "knowledge-hub.dinger.capture.result.v1":
        raise click.ClickException(
            f"packet snapshot must use knowledge-hub.dinger.capture.result.v1: {packet_snapshot_path}"
        )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.capture.result.v1")
    capture_id = str(payload.get("captureId") or "").strip()
    if not capture_id:
        raise click.ClickException(f"packet snapshot is missing captureId: {packet_snapshot_path}")
    return payload


def _restore_capture_packet_from_snapshot(*, khub, snapshot_payload: dict[str, Any], packet_path: Path) -> dict[str, Any]:
    restored_payload = dict(snapshot_payload)
    restored_payload["queuePath"] = str(packet_path)
    restored_payload["packetPath"] = str(packet_path)
    _validate_cli_payload(khub.config, restored_payload, "knowledge-hub.dinger.capture.result.v1")
    _write_json_object(packet_path, restored_payload)
    return restored_payload


def _build_capture_processor(*, khub, project_id: str | None, slug: str | None) -> DingerCaptureProcessor:
    from knowledge_hub.interfaces.cli.commands.os_cmd import _ops_alerts_json

    return DingerCaptureProcessor(
        khub=khub,
        project_id=project_id,
        slug=slug,
        file_capture=lambda capture_packet_path: _invoke_json_command(
            dinger_group,
            ["file", "--from-json", str(capture_packet_path)],
            obj={"khub": khub},
            label="dinger file",
        ),
        link_to_os=lambda filed_payload, _capture_payload, resolved_project_id, resolved_slug: bridge_dinger_result_to_os_capture(
            dinger_payload=filed_payload,
            project_id=resolved_project_id,
            slug=resolved_slug,
            summary=None,
            kind=None,
            severity="medium",
            extra_source_refs=[],
            ops_alerts_json=_ops_alerts_json(khub),
            runner=run_foundry_project_cli,
        ),
    )


@dinger_group.command("capture-process")
@click.option(
    "--queue-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override capture queue directory (default: runtime dinger_capture_intake/queue)",
)
@click.option(
    "--packet",
    "packet_paths",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=(),
    help="Process a specific capture packet path (repeatable)",
)
@click.option("--capture-id", "capture_ids", multiple=True, default=(), help="Process a specific captureId (repeatable)")
@click.option("--limit", type=int, default=0, show_default=True, help="Maximum number of queue packets to process; 0 means all")
@click.option("--project-id", "--os-project-id", "project_id", default=None, help="Target OS project id")
@click.option("--slug", "--os-slug", "slug", default=None, help="Target OS project slug")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def dinger_capture_process(ctx, queue_dir, packet_paths, capture_ids, limit, project_id, slug, as_json):
    """Consume queued capture packets into filed Dinger pages and OS inbox items."""
    khub = ctx.obj["khub"]
    if not str(project_id or slug or "").strip():
        raise click.ClickException("--project-id or --slug is required")
    packets = _select_capture_packet_paths(
        khub=khub,
        queue_dir=queue_dir,
        packet_paths=tuple(packet_paths),
        capture_ids=tuple(capture_ids),
        limit=max(0, int(limit)),
    )
    resolved_queue_dir = Path(queue_dir).expanduser().resolve() if queue_dir is not None else _resolve_capture_queue_dir(khub.config)
    processor = _build_capture_processor(khub=khub, project_id=project_id, slug=slug)
    payload = processor.process_packets(packet_paths=packets, queue_dir=resolved_queue_dir)
    _validate_cli_payload(khub.config, payload, "knowledge-hub.dinger.capture-process.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    counts = dict(payload.get("counts") or {})
    if str(payload.get("status") or "") == "failed":
        raise click.ClickException(
            f"dinger capture-process failed: succeeded={counts.get('succeeded', 0)} failed={counts.get('failed', 0)}"
        )
    console.print(
        f"[bold]dinger capture-process[/bold] scanned={counts.get('scanned', 0)} "
        f"succeeded={counts.get('succeeded', 0)} idempotent={counts.get('idempotent', 0)}"
    )

# Legacy runtime processor kept as a private helper. The public CLI contract is the
# schema-backed `capture-process` command defined above.
@click.option(
    "--queue-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override capture queue directory (default: runtime dinger_capture_intake/queue)",
)
@click.option(
    "--packet",
    "packet_paths",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=(),
    help="Process a specific capture packet path (repeatable)",
)
@click.option("--capture-id", "capture_ids", multiple=True, default=(), help="Process a specific captureId (repeatable)")
@click.option("--limit", type=int, default=0, show_default=True, help="Maximum number of queue packets to process; 0 means all")
@click.option("--os-project-id", default=None, help="Optional OS project id for link_to_os")
@click.option("--slug", "--os-slug", "os_slug", default=None, help="Optional OS project slug for link_to_os")
@click.option("--link-to-os/--no-link-to-os", default=False, show_default=True, help="Run the linked_to_os step after filing")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def _legacy_dinger_capture_process_runtime(ctx, queue_dir, packet_paths, capture_ids, limit, os_project_id, os_slug, link_to_os, as_json):
    """Consume queued capture packets into normalized/filed/linked_to_os runtime states."""
    khub = ctx.obj["khub"]
    should_link_to_os = bool(link_to_os or str(os_project_id or "").strip() or str(os_slug or "").strip())
    if should_link_to_os and not str(os_project_id or os_slug or "").strip():
        raise click.ClickException("--os-project-id or --os-slug is required when linking captures to OS")
    packets = _select_capture_packet_paths(
        khub=khub,
        queue_dir=queue_dir,
        packet_paths=tuple(packet_paths),
        capture_ids=tuple(capture_ids),
        limit=max(0, int(limit)),
    )
    items = [
        _process_capture_packet(
            khub=khub,
            obj=ctx.obj,
            packet_path=packet_path,
            os_project_id=os_project_id,
            os_slug=os_slug,
            should_link_to_os=should_link_to_os,
        )
        for packet_path in packets
    ]
    counts = {
        "captured": sum(1 for item in items if item["status"] == "captured"),
        "normalized": sum(1 for item in items if item["status"] == "normalized"),
        "filed": sum(1 for item in items if item["status"] == "filed"),
        "linked_to_os": sum(1 for item in items if item["status"] == "linked_to_os"),
        "failed": sum(1 for item in items if item["status"] == "failed"),
    }
    payload = {
        "command": "khub dinger capture-process",
        "queueDir": str(Path(queue_dir).expanduser().resolve()) if queue_dir is not None else str(_resolve_capture_queue_dir(khub.config)),
        "processed": len(items),
        "counts": counts,
        "items": items,
        "createdAt": _now_iso(),
    }
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]dinger capture-process[/bold] processed={payload['processed']} "
        f"filed={counts['filed']} linked_to_os={counts['linked_to_os']} failed={counts['failed']}"
    )


@dinger_group.command("recent")
@click.option("--limit", type=int, default=10, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def dinger_recent(ctx, limit, as_json):
    """최근 들어온 paper / ko-note runs를 목적어 기준으로 보여준다."""
    khub = ctx.obj["khub"]
    payload = _build_recent_payload(khub, limit=max(1, int(limit)))
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]dinger recent[/bold] papers={len(payload.get('recentPapers') or [])} "
        f"koNoteRuns={len(payload.get('recentKoNoteRuns') or [])}"
    )


@dinger_group.command("lint")
@click.option("--paper-limit", type=int, default=200, show_default=True)
@click.option("--recent-runs", type=int, default=5, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def dinger_lint(ctx, paper_limit, recent_runs, as_json):
    """현재 knowledge surface를 운영 관점에서 점검한다."""
    khub = ctx.obj["khub"]
    payload = _build_lint_payload(khub, paper_limit=max(1, int(paper_limit)), recent_runs=max(1, int(recent_runs)))
    if as_json:
        console.print_json(data=payload)
        return
    checks = dict(payload.get("checks") or {})
    console.print(
        "[bold]dinger lint[/bold] "
        f"reviewQueued={checks.get('reviewQueued', 0)} "
        f"alerts={checks.get('alertCount', 0)} "
        f"missingSummary={checks.get('papersMissingSummary', 0)} "
        f"missingMemory={checks.get('papersMissingMemoryCard', 0)}"
    )


# Keep the default `khub dinger --help` centered on ingest/capture/file flows.
# Operator utilities remain directly invokable for compatibility.
for _hidden_command in (dinger_capture_http, dinger_recent, dinger_lint):
    _hidden_command.hidden = True
