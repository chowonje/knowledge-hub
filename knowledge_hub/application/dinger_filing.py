"""Reusable helpers for turning Dinger result payloads into filed projections."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import click

from knowledge_hub.learning.obsidian_writeback import _upsert_marked_section, resolve_vault_write_adapter
from knowledge_hub.notes.templates import slugify_title, split_frontmatter, yaml_frontmatter

SOURCE_REF_PRIMARY_KEYS = ("paperId", "url", "noteId", "stableScopeId", "documentScopeId")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        deduped.append({key: value for key, value in ref.items() if value not in (None, "")})
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


def resolve_dinger_filing_request_from_payload(
    payload: dict[str, Any],
    *,
    title: str | None = None,
    input_path: str = "",
) -> dict[str, Any]:
    schema_id = str(payload.get("schema") or "").strip()
    if not schema_id:
        raise click.ClickException("source payload must include schema")

    resolved_input_path = str(input_path or "").strip()
    if schema_id == "knowledge-hub.dinger.capture.result.v1":
        capture_payload = payload.get("payload")
        if not isinstance(capture_payload, dict):
            capture_payload = {}
        capture_id = str(payload.get("captureId") or capture_payload.get("captureId") or "").strip()
        packet_path = str(
            payload.get("packetPath")
            or payload.get("queuePath")
            or capture_payload.get("packetPath")
            or capture_payload.get("queuePath")
            or ""
        ).strip()
        metadata = dict(capture_payload.get("metadata") or {})
        metadata.update(
            {
                "input": "json",
                "inputPath": resolved_input_path,
                "sourceSchema": schema_id,
                "captureUrl": str(payload.get("sourceUrl") or payload.get("captureUrl") or "").strip(),
                "captureId": capture_id,
                "packetPath": packet_path,
                "status": str(payload.get("status") or "").strip() or "captured",
                "queueStatus": str(payload.get("queueStatus") or "").strip(),
                "client": str(payload.get("client") or capture_payload.get("client") or "").strip(),
            }
        )
        resolved_title = (
            str(title or "").strip()
            or str(capture_payload.get("pageTitle") or "").strip()
            or str(payload.get("pageTitle") or payload.get("title") or "").strip()
            or Path(resolved_input_path).stem
            or "dinger-capture"
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
                "captureId": capture_id,
                "packetPath": packet_path,
            },
        }

    if schema_id == "knowledge-hub.dinger.ask.result.v1":
        question = str(payload.get("question") or "").strip()
        answer = str(payload.get("answer") or "").strip()
        error = str(payload.get("error") or "").strip()
        resolved_title = str(title or "").strip() or question or Path(resolved_input_path).stem or "dinger-result"
        lines = []
        if question:
            lines.extend(["## Question", question, ""])
        if answer:
            lines.extend(["## Answer", answer, ""])
        if error:
            lines.extend(["## Error", error, ""])
        content_body = "\n".join(lines).strip() or "## Result\n- empty"
        return {
            "kind": "ask_result",
            "title": resolved_title,
            "contentBody": content_body,
            "metadata": {
                "input": "json",
                "inputPath": resolved_input_path,
                "sourceSchema": schema_id,
                "sourceType": str(payload.get("sourceType") or "").strip(),
                "underlyingSchema": str(payload.get("underlyingSchema") or "").strip(),
                "question": question,
                "status": str(payload.get("status") or "").strip() or "ok",
            },
            "sourceRefs": [],
            "trace": {
                "sourceSchema": schema_id,
                "captureId": "",
                "packetPath": "",
            },
        }

    resolved_title = str(title or "").strip() or schema_id or Path(resolved_input_path).stem or "dinger-result"
    pretty = json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "kind": "json_result",
        "title": resolved_title,
        "contentBody": f"## JSON Payload\n```json\n{pretty}\n```",
        "metadata": {
            "input": "json",
            "inputPath": resolved_input_path,
            "sourceSchema": schema_id,
        },
        "sourceRefs": [],
        "trace": {
            "sourceSchema": schema_id,
            "captureId": "",
            "packetPath": "",
        },
    }


def file_dinger_request(
    *,
    khub,
    request: dict[str, Any],
    slug: str | None = None,
    extra_source_refs: list[dict[str, Any]] | None = None,
    vault_path: str | None = None,
    backend: str | None = None,
    cli_binary: str | None = None,
    vault_name: str | None = None,
) -> dict[str, Any]:
    resolved_title = str(request.get("title") or "").strip()
    resolved_slug = str(slug or "").strip() or slugify_title(resolved_title, fallback="dinger-note")
    source_refs = _dedupe_source_refs(list(request.get("sourceRefs") or []) + list(extra_source_refs or []))
    payload = _write_dinger_projection(
        khub=khub,
        title=resolved_title,
        slug=resolved_slug,
        kind=str(request.get("kind") or "note"),
        content_body=str(request.get("contentBody") or "").rstrip(),
        source_refs=source_refs,
        metadata=dict(request.get("metadata") or {}),
        vault_path=vault_path,
        backend=backend,
        cli_binary=cli_binary,
        vault_name=vault_name,
    )
    trace = dict(request.get("trace") or {})
    payload["sourceSchema"] = str(trace.get("sourceSchema") or (request.get("metadata") or {}).get("sourceSchema") or "").strip()
    payload["captureId"] = str(trace.get("captureId") or (request.get("metadata") or {}).get("captureId") or "").strip()
    payload["packetPath"] = str(trace.get("packetPath") or (request.get("metadata") or {}).get("packetPath") or "").strip()
    return payload
