"""Canonical content-aware chunking helpers.

This module centralizes deterministic chunk selection for markdown, html/web,
plain text, and code-like content so ingestion and ephemeral context surfaces
do not re-implement slightly different splitting rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import re
from typing import Any


MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
HTML_HEADING_RE = re.compile(r"<h([1-6])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
CODE_SECTION_RE = re.compile(
    r"^(?P<indent>\s*)(?:(?:async\s+)?def|class|function|interface|type|enum)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|^(?P<export_indent>\s*)export\s+(?:async\s+)?(?:function|class|const|let|var)\s+(?P<export_name>[A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ChunkSection:
    offset: int
    path: tuple[str, ...]
    title: str


def infer_content_type(
    *,
    text: str = "",
    file_path: str | Path | None = None,
    hint: str | None = None,
) -> str:
    explicit = str(hint or "").strip().lower()
    if explicit in {"markdown", "html", "web", "plain", "code"}:
        return explicit

    suffix = Path(file_path).suffix.lower() if file_path else ""
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".rb", ".c", ".cc", ".cpp", ".h"}:
        return "code"

    body = str(text or "")
    if "<html" in body.lower() or "<body" in body.lower() or HTML_HEADING_RE.search(body):
        return "html"
    if MARKDOWN_HEADING_RE.search(body):
        return "markdown"
    if CODE_SECTION_RE.search(body):
        return "code"
    return "plain"


def _extract_sections(text: str, content_type: str) -> list[ChunkSection]:
    if content_type == "markdown":
        sections: list[ChunkSection] = []
        stack: list[tuple[int, str]] = []
        for match in MARKDOWN_HEADING_RE.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append(ChunkSection(offset=match.start(), path=tuple(item[1] for item in stack), title=title))
        return sections

    if content_type == "html":
        sections = []
        stack = []
        for match in HTML_HEADING_RE.finditer(text):
            level = int(match.group(1))
            raw_title = HTML_TAG_RE.sub(" ", html.unescape(match.group(2) or ""))
            title = re.sub(r"\s+", " ", raw_title).strip()
            if not title:
                continue
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append(ChunkSection(offset=match.start(), path=tuple(item[1] for item in stack), title=title))
        return sections

    if content_type == "code":
        sections = []
        for match in CODE_SECTION_RE.finditer(text):
            name = match.group("name") or match.group("export_name") or "symbol"
            sections.append(ChunkSection(offset=match.start(), path=(name,), title=name))
        return sections

    return []


def _section_for_offset(sections: list[ChunkSection], offset: int) -> tuple[str, str]:
    title = ""
    path: tuple[str, ...] = ()
    for item in sections:
        if item.offset <= offset:
            title = item.title
            path = item.path
        else:
            break
    return title, " > ".join(path)


def _candidate_split_points(chunk: str, content_type: str) -> list[int]:
    points: list[int] = []
    if content_type == "markdown":
        for match in MARKDOWN_HEADING_RE.finditer(chunk):
            if match.start() > 0:
                points.append(match.start())
    if content_type == "html":
        for match in HTML_HEADING_RE.finditer(chunk):
            if match.start() > 0:
                points.append(match.start())
    if content_type == "code":
        for match in CODE_SECTION_RE.finditer(chunk):
            if match.start() > 0:
                points.append(match.start())
        for marker in ("\nclass ", "\ndef ", "\nasync def ", "\nfunction ", "\nexport "):
            index = chunk.rfind(marker)
            if index > 0:
                points.append(index + 1)
        return points

    for delimiter in ("\n\n", "\n", ". ", "? ", "! "):
        index = chunk.rfind(delimiter)
        if index > 0:
            points.append(index + len(delimiter))
    return points


def chunk_text_with_offsets(
    text: str,
    *,
    content_type: str = "plain",
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    body = str(text or "").strip()
    if not body:
        return []

    resolved_type = infer_content_type(text=body, hint=content_type)
    sections = _extract_sections(body, resolved_type)
    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_index = 0
    size = max(32, int(chunk_size or 1200))
    overlap = max(0, min(int(overlap or 0), size // 2))
    total_len = len(body)

    while start < total_len:
        end = min(total_len, start + size)
        candidate = body[start:end]
        if end < total_len:
            for point in sorted(_candidate_split_points(candidate, resolved_type), reverse=True):
                if point > size * 0.5:
                    end = start + point
                    candidate = body[start:end]
                    break

        chunk_text = candidate.strip()
        if not chunk_text:
            break

        section_title, section_path = _section_for_offset(sections, start)
        chunks.append(
            {
                "text": chunk_text,
                "chunk_index": chunk_index,
                "start": start,
                "end": end,
                "section_title": section_title,
                "section_path": section_path,
                "content_type": resolved_type,
            }
        )

        if end >= total_len:
            break
        start = max(0, end - overlap)
        chunk_index += 1

    for idx, item in enumerate(chunks):
        item["chunk_index"] = idx
    return chunks


def snippet_for_path(path: Path, *, max_chars: int) -> str:
    try:
        body = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        body = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

    content_type = infer_content_type(text=body, file_path=path)
    chunks = chunk_text_with_offsets(
        body,
        content_type=content_type,
        chunk_size=max(200, int(max_chars)),
        overlap=min(120, max(0, int(max_chars) // 8)),
    )
    if not chunks:
        return body[:max_chars]
    return str(chunks[0]["text"])[:max_chars]
