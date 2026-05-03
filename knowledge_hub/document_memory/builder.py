"""Build generic document-memory units from notes, papers, and web records."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from knowledge_hub.core.chunking import infer_content_type
from knowledge_hub.core.models import Document, SourceType
from knowledge_hub.document_memory.extraction import DocumentMemoryExtractionV1
from knowledge_hub.document_memory.models import DocumentMemoryUnit
from knowledge_hub.papers.mineru_adapter import MinerUPDFAdapter
from knowledge_hub.papers.opendataloader_adapter import OpenDataLoaderPDFAdapter, resolve_opendataloader_convert_options
from knowledge_hub.papers.pymupdf_adapter import PyMuPDFAdapter
from knowledge_hub.papers.source_text import (
    extract_pdf_text_excerpt,
    resolve_paper_source_snapshot,
    source_hash_for_path,
)
from knowledge_hub.web.ingest import make_web_note_id

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_DEGRADED_MEMORY_HINTS = (
    "pending_summary",
    "요약본/번역본이 아직 등록되지 않았습니다",
    "번역 완료 후 다시 실행하세요",
    "요약을 바로 작성할 수 없습니다",
    "제공된 정보만으로는 요약을 작성할 수 없습니다",
    "원문이 필요합니다",
    "논문 본문이 필요합니다",
    "원문을 보내주시면",
    "original paper text is required",
    "need the original paper text",
    "cannot summarize",
    "unable to summarize",
)
_APPENDIX_SECTION_HINTS = (
    "appendix",
    "appendices",
    "supplementary",
    "supplemental",
    "reference",
    "references",
    "bibliography",
    "acknowledg",
    "부록",
    "참고문헌",
    "감사의 말",
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _iso_utc(value: Any) -> str:
    token = _clean_text(value)
    if not token:
        return ""
    try:
        if token.endswith("Z"):
            return datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        parsed = datetime.fromisoformat(token)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    if re.fullmatch(r"\d{4}", token):
        return datetime(int(token), 1, 1, tzinfo=timezone.utc).isoformat()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(token, fmt).replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            continue
    return ""


def _metadata_date(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        parsed = _iso_utc(metadata.get(key))
        if parsed:
            return parsed
    return ""


def _strip_markdown(text: str) -> str:
    body = str(text or "")
    body = re.sub(r"```.*?```", " ", body, flags=re.DOTALL)
    body = re.sub(r"`([^`]+)`", r"\1", body)
    body = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)
    body = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", body)
    body = re.sub(r"^#{1,6}\s+", "", body, flags=re.MULTILINE)
    body = re.sub(r"^\s*[-*]\s+", "", body, flags=re.MULTILINE)
    body = re.sub(r"\n{2,}", "\n", body)
    return _clean_text(body)


def _first_sentences(text: str, *, limit: int = 2) -> list[str]:
    body = _strip_markdown(text)
    if not body:
        return []
    sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(body) if item.strip()]
    return sentences[: max(1, int(limit))]


def _bounded_excerpt(text: str, *, limit: int = 320) -> str:
    body = _strip_markdown(text)
    if len(body) <= limit:
        return body
    return body[: max(0, limit - 3)].rstrip() + "..."


def _clean_lines(values: list[str], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = _clean_text(raw)
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
        if limit is not None and len(result) >= limit:
            break
    return result


def _looks_degraded_memory_text(*parts: Any) -> bool:
    haystack = " ".join(_clean_text(part).casefold() for part in parts if _clean_text(part))
    if not haystack:
        return False
    return any(token in haystack for token in _DEGRADED_MEMORY_HINTS)


def _is_appendix_like_section(*, title: Any, section_path: Any, source_excerpt: Any = "") -> bool:
    haystack = " ".join(
        [
            _clean_text(title).casefold(),
            _clean_text(section_path).casefold(),
            _clean_text(source_excerpt).casefold()[:160],
        ]
    )
    if not haystack:
        return False
    return any(token in haystack for token in _APPENDIX_SECTION_HINTS)


def _paragraph_blocks(text: str, *, soft_limit: int = 1400) -> list[tuple[str, str]]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", str(text or "")) if part.strip()]
    if not parts:
        token = _clean_text(text)
        return [("Block 1", token)] if token else []
    blocks: list[tuple[str, str]] = []
    current: list[str] = []
    current_len = 0
    index = 1
    for part in parts:
        if current and current_len + len(part) > soft_limit:
            blocks.append((f"Block {index}", "\n\n".join(current)))
            index += 1
            current = []
            current_len = 0
        current.append(part)
        current_len += len(part)
    if current:
        blocks.append((f"Block {index}", "\n\n".join(current)))
    return blocks


def _markdown_sections(text: str) -> list[tuple[str, str, str]]:
    matches = list(_HEADING_RE.finditer(str(text or "")))
    if not matches:
        return []
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = _clean_text(match.group(2))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = str(text[start:end] or "").strip()
        section_path = " > ".join(item[1] for item in stack)
        sections.append((title, section_path, body))
    return sections


def _classify_unit_type(title: str, text: str) -> str:
    key = f"{title} {_bounded_excerpt(text, limit=180)}".lower()
    if any(token in key for token in ("abstract", "요약", "summary")):
        return "summary"
    if any(token in key for token in ("method", "방법", "approach")):
        return "method"
    if any(token in key for token in ("result", "결과", "finding", "evaluation")):
        return "result"
    if any(token in key for token in ("limitation", "한계")):
        return "limitation"
    if any(token in key for token in ("background", "배경", "introduction", "소개")):
        return "background"
    if re.search(r"^\s*[-*]\s+", str(text or ""), flags=re.MULTILINE):
        return "list_block"
    if "|" in str(text or "") and "\n" in str(text or ""):
        return "table_block"
    return "section"


def _contextual_summary(document_title: str, section_path: str, text: str) -> str:
    sentences = _first_sentences(text, limit=2)
    body = " ".join(sentences) if sentences else _bounded_excerpt(text, limit=180)
    label = section_path or document_title or "document"
    if not body:
        return label
    return f"[{label}] {body}"


def _document_thesis(document_title: str, child_units: list[DocumentMemoryUnit], fallback_text: str) -> str:
    for unit in child_units[:3]:
        summary = _clean_text(unit.contextual_summary)
        if summary:
            return summary
    return _contextual_summary(document_title, document_title, fallback_text)


def _context_header(
    *,
    document_title: str,
    source_type: str,
    section_path: str,
    unit_type: str,
    document_thesis: str,
    parent_summary: str = "",
) -> str:
    header_parts = [
        f"Document: {document_title}" if document_title else "",
        f"Source: {source_type}" if source_type else "",
        f"Section: {section_path}" if section_path else "",
        f"Unit: {unit_type}" if unit_type else "",
        f"Thesis: {document_thesis}" if document_thesis else "",
        f"Parent summary: {parent_summary}" if parent_summary else "",
    ]
    return _clean_text(" | ".join(part for part in header_parts if part))


def _normalize_heading_path(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [_clean_text(item) for item in value if _clean_text(item)]
    token = _clean_text(value)
    if not token:
        return []
    return [part.strip() for part in token.split(">") if part.strip()]


def _element_matches_section(element: dict[str, Any], *, title: str, section_path: str) -> bool:
    path = _normalize_heading_path(element.get("heading_path"))
    normalized_title = _clean_text(title).casefold()
    normalized_path = _clean_text(section_path).casefold()
    if path:
        joined = " > ".join(path).casefold()
        if normalized_path and joined == normalized_path:
            return True
        if normalized_title and path[-1].casefold() == normalized_title:
            return True
    text = _clean_text(element.get("text"))
    return bool(normalized_title and text and text.casefold() == normalized_title)


def _structured_section_signals(
    *,
    title: str,
    section_path: str,
    elements: list[dict[str, Any]],
) -> dict[str, Any]:
    matches = [item for item in elements if _element_matches_section(item, title=title, section_path=section_path)]
    if not matches:
        return {}
    pages = [item.get("page") for item in matches if item.get("page") is not None]
    heading_path = _normalize_heading_path(matches[0].get("heading_path"))
    element_types = [_clean_text(item.get("type")).lower() for item in matches if _clean_text(item.get("type"))]
    bbox = next((item.get("bbox") for item in matches if item.get("bbox") is not None), None)
    return {
        "page": pages[0] if pages else None,
        "bbox": bbox,
        "heading_path": heading_path,
        "element_types": element_types,
        "match_count": len(matches),
        "reading_order": next((item.get("reading_order") for item in matches if item.get("reading_order") is not None), None),
    }


def _unit_type_with_structured_hint(default_unit_type: str, signals: dict[str, Any]) -> str:
    element_types = {str(item).strip().lower() for item in list(signals.get("element_types") or []) if str(item).strip()}
    if "table" in element_types:
        return "table_block"
    if "image" in element_types or "figure" in element_types:
        return "image_block"
    return default_unit_type


def _stable_unit_id(document_id: str, path: str, index: int) -> str:
    digest = hashlib.sha1(f"{document_id}|{path}|{index}".encode("utf-8")).hexdigest()[:12]
    return f"memory-unit:{document_id}:{digest}"


def _parse_note_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw or "{}")
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _top_concept_candidates(concepts: list[str] | None, *, limit: int = 8) -> list[str]:
    return _clean_lines(list(concepts or []), limit=limit)


def _schema_document_payload(
    *,
    document: Document,
    document_id: str,
    source_ref: str,
    records: list[dict[str, Any]],
    parser_payload: dict[str, Any] | None,
    raw_text: str,
    compact: bool = False,
) -> dict[str, Any]:
    summary = dict(records[0]) if records else {}
    sections = [dict(item) for item in records[1:]]
    section_by_type: dict[str, dict[str, Any]] = {}
    for item in sections:
        unit_type = _clean_text(item.get("unit_type"))
        if unit_type and unit_type not in section_by_type:
            section_by_type[unit_type] = item
    parser_meta = dict((parser_payload or {}).get("parser_meta") or {})

    def _section_excerpt(*unit_types: str, fallback_index: int | None = None, limit: int = 420) -> str:
        for unit_type in unit_types:
            candidate = section_by_type.get(unit_type)
            if candidate:
                text = _clean_text(candidate.get("source_excerpt") or candidate.get("contextual_summary"))
                if text:
                    return text[:limit]
        if fallback_index is not None and 0 <= fallback_index < len(sections):
            fallback = sections[fallback_index]
            text = _clean_text(fallback.get("source_excerpt") or fallback.get("contextual_summary"))
            if text:
                return text[:limit]
        return ""

    payload = {
        "documentId": document_id,
        "documentTitle": str(document.title or ""),
        "sourceType": str(document.source_type.value),
        "sourceRef": str(source_ref or ""),
        "contentType": infer_content_type(text=raw_text, file_path=document.file_path),
        "documentMetadata": dict(document.metadata or {}),
        "parserMeta": parser_meta,
        "title": str(document.title or ""),
        "summaryExcerpt": _clean_text(summary.get("source_excerpt") or summary.get("contextual_summary"))[:480],
        "methodExcerpt": _section_excerpt("method"),
        "findingsExcerpt": _section_excerpt("result", fallback_index=1),
        "limitationsExcerpt": _section_excerpt("limitation"),
        "topConceptCandidates": _top_concept_candidates(list(summary.get("concepts") or [])),
        "sectionCandidates": [
            {
                "title": _clean_text(item.get("title")),
                "sectionPath": _clean_text(item.get("section_path")),
                "unitType": _clean_text(item.get("unit_type")),
                "contextualSummary": _clean_text(item.get("contextual_summary"))[:220],
                "sourceExcerpt": _clean_text(item.get("source_excerpt"))[:220],
            }
            for item in sections[:6]
        ],
    }
    if compact:
        return {
            "documentId": payload["documentId"],
            "documentTitle": payload["documentTitle"],
            "sourceType": payload["sourceType"],
            "sourceRef": payload["sourceRef"],
            "contentType": payload["contentType"],
            "title": payload["title"],
            "summaryExcerpt": payload["summaryExcerpt"][:320],
            "methodExcerpt": payload["methodExcerpt"][:320],
            "findingsExcerpt": payload["findingsExcerpt"][:320],
            "limitationsExcerpt": payload["limitationsExcerpt"][:260],
            "topConceptCandidates": list(payload["topConceptCandidates"][:6]),
            "sectionCandidates": [
                {
                    "title": _clean_text(item.get("title")),
                    "sectionPath": _clean_text(item.get("sectionPath") or item.get("section_path")),
                    "unitType": _clean_text(item.get("unitType") or item.get("unit_type")),
                    "contextualSummary": _clean_text(item.get("contextualSummary") or item.get("contextual_summary"))[:160],
                }
                for item in list(payload["sectionCandidates"] or [])[:2]
            ],
        }
    return payload


def _stable_relation_id(src_form: str, src_id: str, dst_form: str, dst_id: str, relation_type: str) -> str:
    digest = hashlib.sha1(f"{src_form}|{src_id}|{dst_form}|{dst_id}|{relation_type}".encode("utf-8")).hexdigest()[:16]
    return f"memory-relation:{relation_type}:{digest}"


def _memory_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^0-9A-Za-z가-힣]+", str(text or "").casefold())
        if len(token) >= 3 and token not in {"paper", "note", "notes", "summary", "overview", "method"}
    }


def _memory_overlap(left: str, right: str) -> float:
    left_tokens = _memory_tokens(left)
    right_tokens = _memory_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def _update_markers(*parts: Any) -> bool:
    haystack = " ".join(str(part or "").casefold() for part in parts)
    return any(token in haystack for token in ("updated", "revised", "benchmark update", " v2", " v3", "version 2", "version 3"))


class DocumentMemoryBuilder:
    def __init__(
        self,
        sqlite_db,
        config: Any | None = None,
        pdf_parser_adapter: OpenDataLoaderPDFAdapter | None = None,
        mineru_parser_adapter: MinerUPDFAdapter | None = None,
        pymupdf_adapter: PyMuPDFAdapter | None = None,
        schema_extractor: Any | None = None,
        extraction_mode: str = "deterministic",
    ):
        self.sqlite_db = sqlite_db
        self.config = config
        self._pdf_parser_adapter = pdf_parser_adapter
        self._mineru_parser_adapter = mineru_parser_adapter
        self._pymupdf_adapter = pymupdf_adapter
        self._schema_extractor = schema_extractor
        self._extraction_mode = str(extraction_mode or "deterministic").strip().lower()
        self._last_extraction_diagnostics: dict[str, dict[str, Any]] = {}

    def get_last_extraction_diagnostics(self, document_id: str) -> dict[str, Any]:
        return dict(self._last_extraction_diagnostics.get(str(document_id or "").strip(), {}))

    def _document_from_note(self, note_id: str) -> Document:
        note = self.sqlite_db.get_note(str(note_id).strip())
        if not note:
            raise ValueError(f"note not found: {note_id}")
        metadata = _parse_note_metadata(note.get("metadata"))
        tags = list(metadata.get("tags") or [])
        tag_rows = getattr(self.sqlite_db, "get_note_tags", None)
        if callable(tag_rows):
            for tag in tag_rows(str(note_id).strip()):
                if tag not in tags:
                    tags.append(tag)
        links = list(metadata.get("links") or [])
        source_type = str(note.get("source_type") or metadata.get("source_type") or "note").strip().lower()
        try:
            source_enum = SourceType(source_type)
        except Exception:
            source_enum = SourceType.NOTE
        return Document(
            content=str(note.get("content") or ""),
            metadata=metadata,
            file_path=str(note.get("file_path") or note_id),
            title=str(note.get("title") or note_id),
            tags=[str(item) for item in tags if str(item or "").strip()],
            links=[str(item) for item in links if str(item or "").strip()],
            source_type=source_enum,
        )

    def _paper_parser_adapter(self) -> OpenDataLoaderPDFAdapter:
        if self._pdf_parser_adapter is None:
            papers_dir = getattr(self.config, "papers_dir", "") if self.config is not None else ""
            self._pdf_parser_adapter = OpenDataLoaderPDFAdapter(papers_dir=str(papers_dir or ""))
        return self._pdf_parser_adapter

    def _paper_mineru_adapter(self) -> MinerUPDFAdapter:
        if self._mineru_parser_adapter is None:
            papers_dir = getattr(self.config, "papers_dir", "") if self.config is not None else ""
            self._mineru_parser_adapter = MinerUPDFAdapter(papers_dir=str(papers_dir or ""))
        return self._mineru_parser_adapter

    def _paper_pymupdf_adapter(self) -> PyMuPDFAdapter:
        if self._pymupdf_adapter is None:
            papers_dir = getattr(self.config, "papers_dir", "") if self.config is not None else ""
            self._pymupdf_adapter = PyMuPDFAdapter(papers_dir=str(papers_dir or ""))
        return self._pymupdf_adapter

    def _document_from_paper(
        self,
        paper_id: str,
        *,
        paper_parser: str = "raw",
        refresh_parse: bool = False,
        opendataloader_options: dict[str, Any] | None = None,
    ) -> tuple[Document, dict[str, Any]]:
        note_id = f"paper:{str(paper_id).strip()}"
        token = str(paper_id).strip()
        parser_token = str(paper_parser or "raw").strip().lower()
        if parser_token in {"opendataloader", "mineru", "pymupdf"}:
            paper = self.sqlite_db.get_paper(token)
            if not paper:
                raise ValueError(f"paper not found: {paper_id}")
            pdf_path = str(paper.get("pdf_path") or "").strip()
            if not pdf_path:
                raise ValueError(f"paper pdf not found: {paper_id}")
            if parser_token == "opendataloader":
                adapter = self._paper_parser_adapter()
            elif parser_token == "mineru":
                adapter = self._paper_mineru_adapter()
            else:
                adapter = self._paper_pymupdf_adapter()
            if parser_token == "opendataloader":
                parsed = adapter.ensure_artifacts(
                    paper_id=token,
                    pdf_path=pdf_path,
                    refresh=bool(refresh_parse),
                    parser_options=resolve_opendataloader_convert_options(self.config, overrides=opendataloader_options),
                )
            else:
                parsed = adapter.ensure_artifacts(
                    paper_id=token,
                    pdf_path=pdf_path,
                    refresh=bool(refresh_parse),
                )
            document = Document(
                content=str(parsed.markdown_text or ""),
            metadata={
                "paper_id": token,
                "authors": str(paper.get("authors") or ""),
                "year": paper.get("year"),
                "field": str(paper.get("field") or ""),
                "published_at": _iso_utc(paper.get("year")),
                "source_content_hash": source_hash_for_path(pdf_path),
                "parser_meta": dict(parsed.parser_meta),
            },
                file_path=str(parsed.markdown_path),
                title=str(paper.get("title") or token),
                tags=["paper", f"parsed:{parser_token}"],
                links=[],
                source_type=SourceType.PAPER,
            )
            return document, parsed.to_payload()

        note = self.sqlite_db.get_note(note_id)
        if note:
            return self._document_from_note(note_id), {}
        paper = self.sqlite_db.get_paper(token)
        if not paper:
            raise ValueError(f"paper not found: {paper_id}")
        snapshot = resolve_paper_source_snapshot(
            paper,
            pdf_text_extractor=extract_pdf_text_excerpt,
        )
        return Document(
            content=snapshot.content,
            metadata={
                "paper_id": token,
                "authors": str(paper.get("authors") or ""),
                "year": paper.get("year"),
                "field": str(paper.get("field") or ""),
                "published_at": _iso_utc(paper.get("year")),
                "raw_fallback_source": snapshot.source_key,
                "source_content_hash": snapshot.source_content_hash,
                "source_warnings": list(snapshot.warnings),
            },
            file_path=snapshot.path or str(paper.get("pdf_path") or f"paper:{paper_id}"),
            title=str(paper.get("title") or paper_id),
            tags=["paper"],
            links=[],
            source_type=SourceType.PAPER,
        ), {}

    def _document_from_web(self, canonical_url: str) -> Document:
        token = str(canonical_url or "").strip()
        if not token:
            raise ValueError("canonical_url is required")
        note_id = make_web_note_id(token)
        note = self.sqlite_db.get_note(note_id)
        if note:
            return self._document_from_note(note_id)
        row = self.sqlite_db.conn.execute(
            """
            SELECT * FROM crawl_pipeline_records
            WHERE canonical_url = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (token,),
        ).fetchone()
        if not row:
            raise ValueError(f"web record not found: {canonical_url}")
        record = dict(row)
        normalized_path = Path(str(record.get("normalized_path") or "").strip())
        payload: dict[str, Any] = {}
        if normalized_path.exists():
            try:
                payload = json.loads(normalized_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        return Document(
            content=str(payload.get("content_text") or ""),
            metadata={
                "canonical_url": token,
                "domain": str(payload.get("domain") or record.get("domain") or ""),
                "published_at": str(payload.get("published_at") or ""),
                "observed_at": _iso_utc(record.get("updated_at") or record.get("created_at") or ""),
                "author": str(payload.get("author") or ""),
            },
            file_path=token,
            title=str(payload.get("title") or token),
            tags=list(payload.get("tags") or []),
            links=[],
            source_type=SourceType.WEB,
        )

    def _build_units(
        self,
        document: Document,
        *,
        document_id: str,
        source_ref: str,
        claims: list[str] | None = None,
        concepts: list[str] | None = None,
        structured_elements: list[dict[str, Any]] | None = None,
        parser_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        text = str(document.content or "").strip()
        if not text:
            raise ValueError(f"document content is empty: {document_id}")

        content_type = infer_content_type(text=text, file_path=document.file_path)
        sections = _markdown_sections(text) if content_type == "markdown" else []
        if not sections:
            sections = [(title, title, body) for title, body in _paragraph_blocks(text)]

        summary_unit_id = _stable_unit_id(document_id, "__document__", 0)
        child_units: list[DocumentMemoryUnit] = []
        parser_meta = dict((parser_payload or {}).get("parser_meta") or {})
        structured_elements = list(structured_elements or [])
        metadata = dict(document.metadata or {})
        source_content_hash = _clean_text(metadata.get("source_content_hash"))
        document_date = _metadata_date(
            metadata,
            "document_date",
            "published_at",
            "updated_at",
            "date",
            "created_at",
            "year",
        )
        event_date = _metadata_date(metadata, "event_date")
        observed_at = _metadata_date(metadata, "observed_at", "fetched_at", "updated_at")
        for index, (title, section_path, body) in enumerate(sections, start=1):
            body_token = str(body or "").strip()
            if not body_token:
                continue
            appendix_like = (
                document.source_type == SourceType.PAPER
                and _is_appendix_like_section(title=title or document.title or f"Block {index}", section_path=section_path or title, source_excerpt=body_token)
            )
            section_signals = _structured_section_signals(
                title=title or document.title or f"Block {index}",
                section_path=section_path or title,
                elements=structured_elements,
            )
            unit_id = _stable_unit_id(document_id, section_path or title or f"block-{index}", index)
            child_units.append(
                DocumentMemoryUnit(
                    unit_id=unit_id,
                    document_id=document_id,
                    document_title=document.title,
                    source_type=document.source_type.value,
                    source_ref=source_ref,
                    unit_type=_unit_type_with_structured_hint(
                        _classify_unit_type(title, body_token),
                        section_signals,
                    ),
                    title=title or document.title or f"Block {index}",
                    section_path=section_path or title,
                    contextual_summary=_contextual_summary(document.title, section_path or title, body_token),
                    source_excerpt=_bounded_excerpt(body_token, limit=320),
                    context_header="",
                    document_thesis="",
                    parent_unit_id=summary_unit_id,
                    scope_id=f"{document.source_type.value}:{document.file_path}:{section_path or title or index}",
                    confidence=0.18 if appendix_like else 0.72,
                    provenance={
                        "builder": "document-memory-v1",
                        "source_type": document.source_type.value,
                        "source_ref": source_ref,
                        "file_path": document.file_path,
                        "source_content_hash": source_content_hash,
                        "parser": str(parser_meta.get("parser") or ""),
                        "page": section_signals.get("page"),
                        "bbox": section_signals.get("bbox"),
                        "element_type": next(iter(section_signals.get("element_types") or []), ""),
                        "heading_path": list(section_signals.get("heading_path") or []),
                        "match_count": int(section_signals.get("match_count") or 0),
                        "reading_order": section_signals.get("reading_order"),
                        "quality_signals": {
                            "appendix_like": bool(appendix_like),
                        },
                    },
                    order_index=index,
                    content_type=content_type,
                    links=list(document.links),
                    tags=list(document.tags),
                    document_date=document_date,
                    event_date=event_date,
                    observed_at=observed_at,
                    search_text="",
                )
            )

        document_thesis = _document_thesis(document.title, child_units, text)
        for unit in child_units:
            parent_summary = _clean_text(unit.contextual_summary)
            unit.document_thesis = document_thesis
            unit.context_header = _context_header(
                document_title=document.title,
                source_type=document.source_type.value,
                section_path=unit.section_path,
                unit_type=unit.unit_type,
                document_thesis=document_thesis,
                parent_summary=parent_summary,
            )
            unit.search_text = _clean_text(
                " ".join(
                    [
                        unit.context_header,
                        document.source_type.value,
                        source_ref,
                        document.file_path,
                        unit.title,
                        unit.section_path,
                        unit.contextual_summary,
                        unit.source_excerpt,
                        unit.document_date,
                        unit.event_date,
                        unit.observed_at,
                        " ".join(unit.tags),
                        " ".join(unit.links),
                    ]
                )
            )

        summary_text = " ".join(unit.contextual_summary for unit in child_units[:3]).strip() or _contextual_summary(
            document.title,
            document.title,
            text,
        )
        summary = DocumentMemoryUnit(
            unit_id=summary_unit_id,
            document_id=document_id,
            document_title=document.title,
            source_type=document.source_type.value,
            source_ref=source_ref,
            unit_type="document_summary",
            title=document.title,
            section_path="",
            contextual_summary=summary_text,
            source_excerpt=_bounded_excerpt(text, limit=360),
            context_header=_context_header(
                document_title=document.title,
                source_type=document.source_type.value,
                section_path=document.title,
                unit_type="document_summary",
                document_thesis=document_thesis,
            ),
            document_thesis=document_thesis,
            parent_unit_id="",
            scope_id=f"{document.source_type.value}:{document.file_path}",
            confidence=0.85,
            provenance={
                "builder": "document-memory-v1",
                "source_type": document.source_type.value,
                "source_ref": source_ref,
                "file_path": document.file_path,
                "source_content_hash": source_content_hash,
                "parser": str(parser_meta.get("parser") or ""),
                "source_pdf": str(parser_meta.get("source_pdf") or ""),
                "elements_imported": len(structured_elements),
            },
            order_index=0,
            content_type=content_type,
            links=list(document.links),
            tags=list(document.tags),
            claims=list(claims or []),
            concepts=list(concepts or []),
            document_date=document_date,
            event_date=event_date,
            observed_at=observed_at,
            search_text=_clean_text(
                " ".join(
                    [
                        _context_header(
                            document_title=document.title,
                            source_type=document.source_type.value,
                            section_path=document.title,
                            unit_type="document_summary",
                            document_thesis=document_thesis,
                        ),
                        document.source_type.value,
                        source_ref,
                        document.file_path,
                        document.title,
                        summary_text,
                        _bounded_excerpt(text, limit=260),
                        document_date,
                        event_date,
                        observed_at,
                        " ".join(document.tags),
                        " ".join(document.links),
                        " ".join(claims or []),
                        " ".join(concepts or []),
                    ]
                )
            ),
        )
        records = [summary.to_record()] + [unit.to_record() for unit in child_units]
        if source_content_hash:
            for record in records:
                record["source_content_hash"] = source_content_hash
        records = self._apply_schema_extraction(
            document=document,
            document_id=document_id,
            source_ref=source_ref,
            records=records,
            parser_payload=parser_payload,
            raw_text=text,
        )
        return records

    def _apply_schema_extraction(
        self,
        *,
        document: Document,
        document_id: str,
        source_ref: str,
        records: list[dict[str, Any]],
        parser_payload: dict[str, Any] | None,
        raw_text: str,
    ) -> list[dict[str, Any]]:
        diagnostics = {
            "mode": self._extraction_mode,
            "attempted": False,
            "applied": False,
            "fallbackUsed": False,
            "schema": "knowledge-hub.document-memory-extraction.v1",
            "warnings": [],
            "latencyMs": 0,
            "latencyBucket": "none",
            "rawPayloadBytes": 0,
            "parsedFields": [],
            "parseStage": "",
            "rawOutputPreview": "",
            "coverageByField": {},
            "recoveryAttempted": False,
            "recoveryApplied": False,
            "recoveryStrategy": "",
        }

        def _warn_once(code: str) -> None:
            token = _clean_text(code)
            if token and token not in diagnostics["warnings"]:
                diagnostics["warnings"].append(token)

        def _prefer_clean_schema_text(candidate: Any, fallback: Any, *, warning_code: str) -> str:
            candidate_text = _clean_text(candidate)
            fallback_text = _clean_text(fallback)
            if not candidate_text:
                return fallback_text
            if _looks_degraded_memory_text(candidate_text):
                _warn_once(warning_code)
                return fallback_text
            return candidate_text

        if self._extraction_mode not in {"shadow", "schema"} or self._schema_extractor is None:
            self._last_extraction_diagnostics[document_id] = diagnostics
            return records
        diagnostics["attempted"] = True
        started = time.perf_counter()
        schema_input = _schema_document_payload(
            document=document,
            document_id=document_id,
            source_ref=source_ref,
            records=records,
            parser_payload=parser_payload,
            raw_text=raw_text,
        )
        compact_schema_input = _schema_document_payload(
            document=document,
            document_id=document_id,
            source_ref=source_ref,
            records=records,
            parser_payload=parser_payload,
            raw_text=raw_text,
            compact=True,
        )

        def _run_extraction(*, compact: bool) -> tuple[dict[str, Any], dict[str, Any]]:
            extract_with_metadata = getattr(self._schema_extractor, "extract_with_metadata", None)
            if callable(extract_with_metadata):
                try:
                    raw_payload, extractor_metadata = extract_with_metadata(
                        document=compact_schema_input if compact else schema_input,
                        compact=compact,
                    )
                except TypeError:
                    raw_payload, extractor_metadata = extract_with_metadata(
                        document=compact_schema_input if compact else schema_input,
                    )
            else:
                try:
                    raw_payload = self._schema_extractor.extract(
                        document=compact_schema_input if compact else schema_input,
                        compact=compact,
                    )
                except TypeError:
                    raw_payload = self._schema_extractor.extract(
                        document=compact_schema_input if compact else schema_input,
                    )
                extractor_metadata = {}
            return raw_payload, extractor_metadata

        extraction = None
        raw_payload: dict[str, Any] | None = None
        extractor_metadata: dict[str, Any] = {}
        try:
            try:
                raw_payload, extractor_metadata = _run_extraction(compact=False)
                extraction = DocumentMemoryExtractionV1.from_dict(raw_payload)
            except Exception as exc:
                extraction = None
                diagnostics["warnings"].append(f"initial_extractor_error:{type(exc).__name__}")
                diagnostics["parseStage"] = _clean_text(getattr(exc, "parse_stage", ""))
                diagnostics["rawOutputPreview"] = _clean_text(getattr(exc, "raw_preview", ""))
                diagnostics["rawPayloadBytes"] = int(getattr(exc, "raw_payload_bytes", 0) or 0)
                raw_payload = None

            if extraction is None and raw_payload is not None:
                diagnostics["warnings"].append("initial_invalid_or_empty_payload")

            if extraction is None and document.source_type == SourceType.PAPER:
                diagnostics["recoveryAttempted"] = True
                diagnostics["recoveryStrategy"] = "compact_retry"
                try:
                    raw_payload, extractor_metadata = _run_extraction(compact=True)
                    extraction = DocumentMemoryExtractionV1.from_dict(raw_payload)
                    if extraction is not None:
                        diagnostics["recoveryApplied"] = True
                        diagnostics["warnings"].append("recovery_pass:compact_retry")
                except Exception as exc:
                    diagnostics["warnings"].append(f"extractor_error:{type(exc).__name__}")
                    diagnostics["fallbackUsed"] = True
                    diagnostics["parseStage"] = _clean_text(getattr(exc, "parse_stage", "")) or diagnostics["parseStage"]
                    diagnostics["rawOutputPreview"] = _clean_text(getattr(exc, "raw_preview", "")) or diagnostics["rawOutputPreview"]
                    diagnostics["rawPayloadBytes"] = int(getattr(exc, "raw_payload_bytes", 0) or diagnostics["rawPayloadBytes"] or 0)
                    self._last_extraction_diagnostics[document_id] = diagnostics
                    return records

            if extraction is None:
                diagnostics["warnings"].append("invalid_or_empty_payload")
                diagnostics["fallbackUsed"] = True
                self._last_extraction_diagnostics[document_id] = diagnostics
                return records

            diagnostics["rawPayloadBytes"] = int(extractor_metadata.get("rawPayloadBytes") or diagnostics["rawPayloadBytes"] or 0)
            diagnostics["parsedFields"] = list(extractor_metadata.get("parsedFields") or list((raw_payload or {}).keys()))
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            diagnostics["latencyMs"] = latency_ms
            diagnostics["latencyBucket"] = (
                "fast" if latency_ms < 5000 else "steady" if latency_ms < 15000 else "slow"
            )

        diagnostics["extractorModel"] = extraction.extractor_model
        diagnostics["coverageStatus"] = extraction.coverage_status
        diagnostics["warnings"].extend(list(extraction.warnings or []))
        diagnostics["sectionCount"] = len(extraction.section_units)
        diagnostics["coverageByField"] = {
            "documentThesis": "complete" if _clean_text(extraction.document_thesis) else "missing",
            "summaryUnit": "complete" if extraction.summary_unit is not None else "missing",
            "sectionUnits": "complete" if extraction.section_units else "missing",
            "topClaims": "complete" if extraction.top_claims else "missing",
            "coreConcepts": "complete" if extraction.core_concepts else "missing",
        }
        diagnostics["applied"] = self._extraction_mode == "schema"

        if self._extraction_mode == "shadow":
            self._last_extraction_diagnostics[document_id] = diagnostics
            return records

        summary = dict(records[0]) if records else {}
        units = [dict(item) for item in records[1:]]
        document_thesis = _prefer_clean_schema_text(
            extraction.document_thesis,
            summary.get("document_thesis"),
            warning_code="filtered_degraded_document_thesis",
        )
        if document_thesis:
            summary["document_thesis"] = document_thesis
        if extraction.summary_unit is not None:
            clean_summary = _prefer_clean_schema_text(
                extraction.summary_unit.contextual_summary,
                summary.get("contextual_summary"),
                warning_code="filtered_degraded_summary_context",
            )
            clean_excerpt = _prefer_clean_schema_text(
                extraction.summary_unit.source_excerpt,
                summary.get("source_excerpt"),
                warning_code="filtered_degraded_summary_excerpt",
            )
            if clean_summary:
                summary["contextual_summary"] = clean_summary
            if clean_excerpt:
                summary["source_excerpt"] = clean_excerpt
            summary["confidence"] = max(
                float(summary.get("confidence") or 0.0),
                float(extraction.summary_unit.field_confidence or 0.0),
            )
            summary["claims"] = _clean_lines(list(summary.get("claims") or []) + list(extraction.top_claims or extraction.summary_unit.claims), limit=16)
            summary["concepts"] = _clean_lines(list(summary.get("concepts") or []) + list(extraction.core_concepts or extraction.summary_unit.concepts), limit=16)
        summary_provenance = dict(summary.get("provenance") or {})
        summary_provenance["schema_extraction"] = {
            "schema": diagnostics["schema"],
            "mode": self._extraction_mode,
            "extractor_model": diagnostics.get("extractorModel", ""),
            "coverage_status": diagnostics.get("coverageStatus", "partial"),
            "warnings": list(diagnostics["warnings"]),
        }
        summary["provenance"] = summary_provenance
        if document_thesis:
            summary["context_header"] = _context_header(
                document_title=str(summary.get("document_title") or document.title or ""),
                source_type=str(summary.get("source_type") or document.source_type.value),
                section_path=str(summary.get("document_title") or document.title or ""),
                unit_type=str(summary.get("unit_type") or "document_summary"),
                document_thesis=document_thesis,
            )

        by_key = {
            (
                _clean_text(item.get("section_path")).casefold(),
                _clean_text(item.get("title")).casefold(),
            ): item
            for item in units
        }
        for section in extraction.section_units:
            key = (section.section_path.casefold(), section.title.casefold())
            target = by_key.get(key)
            if target is None:
                for candidate in units:
                    if section.section_path and _clean_text(candidate.get("section_path")).casefold() == section.section_path.casefold():
                        target = candidate
                        break
                    if section.title and _clean_text(candidate.get("title")).casefold() == section.title.casefold():
                        target = candidate
                        break
            if target is None:
                continue
            if section.unit_type:
                target["unit_type"] = section.unit_type
            clean_section_summary = _prefer_clean_schema_text(
                section.contextual_summary,
                target.get("contextual_summary"),
                warning_code="filtered_degraded_section_context",
            )
            clean_section_excerpt = _prefer_clean_schema_text(
                section.source_excerpt,
                target.get("source_excerpt"),
                warning_code="filtered_degraded_section_excerpt",
            )
            if clean_section_summary:
                target["contextual_summary"] = clean_section_summary
            if clean_section_excerpt:
                target["source_excerpt"] = clean_section_excerpt
            target["document_thesis"] = document_thesis
            target["confidence"] = max(float(target.get("confidence") or 0.0), float(section.field_confidence or 0.0))
            target["claims"] = _clean_lines(list(target.get("claims") or []) + list(section.claims or []), limit=12)
            target["concepts"] = _clean_lines(list(target.get("concepts") or []) + list(section.concepts or []), limit=12)
            provenance = dict(target.get("provenance") or {})
            quality_signals = dict(provenance.get("quality_signals") or {})
            appendix_like = (
                document.source_type == SourceType.PAPER
                and _is_appendix_like_section(
                    title=target.get("title"),
                    section_path=target.get("section_path"),
                    source_excerpt=target.get("source_excerpt"),
                )
            )
            quality_signals["appendix_like"] = bool(appendix_like)
            provenance["schema_extraction"] = {
                "coverage_status": section.coverage_status,
                "field_confidence": round(float(section.field_confidence or 0.0), 4),
                "evidence_spans": list(section.evidence_spans or []),
            }
            if appendix_like:
                _warn_once("appendix_like_units_downranked")
                current_confidence = float(target.get("confidence") or 0.0)
                if current_confidence > 0.24:
                    target["confidence"] = 0.24
            provenance["quality_signals"] = quality_signals
            target["provenance"] = provenance
            target["context_header"] = _context_header(
                document_title=str(target.get("document_title") or document.title or ""),
                source_type=str(target.get("source_type") or document.source_type.value),
                section_path=str(target.get("section_path") or target.get("title") or ""),
                unit_type=str(target.get("unit_type") or "section"),
                document_thesis=document_thesis,
                parent_summary=_clean_text(target.get("contextual_summary")),
            )
            target["search_text"] = _clean_text(
                " ".join(
                    [
                        str(target.get("context_header") or ""),
                        str(target.get("source_type") or ""),
                        str(target.get("source_ref") or ""),
                        str(target.get("title") or ""),
                        str(target.get("section_path") or ""),
                        str(target.get("contextual_summary") or ""),
                        str(target.get("source_excerpt") or ""),
                        " ".join(list(target.get("claims") or [])),
                        " ".join(list(target.get("concepts") or [])),
                    ]
                )
            )

        summary["search_text"] = _clean_text(
            " ".join(
                [
                    str(summary.get("context_header") or ""),
                    str(summary.get("source_type") or ""),
                    str(summary.get("source_ref") or ""),
                    str(summary.get("document_title") or ""),
                    str(summary.get("contextual_summary") or ""),
                    str(summary.get("source_excerpt") or ""),
                    " ".join(list(summary.get("claims") or [])),
                    " ".join(list(summary.get("concepts") or [])),
                ]
            )
        )
        self._last_extraction_diagnostics[document_id] = diagnostics
        return [summary] + units

    def build_and_store_note(self, *, note_id: str) -> list[dict[str, Any]]:
        document = self._document_from_note(note_id)
        document_id = str(note_id).strip()
        units = self._build_units(document, document_id=document_id, source_ref=document_id)
        stored = self.sqlite_db.replace_document_memory_units(document_id=document_id, units=units)
        self._refresh_updates_relations(document_id=document_id, summary=stored[0] if stored else {})
        return stored

    def build_and_store_paper(
        self,
        *,
        paper_id: str,
        paper_parser: str = "raw",
        refresh_parse: bool = False,
        opendataloader_options: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        token = str(paper_id).strip()
        document, parser_payload = self._document_from_paper(
            token,
            paper_parser=paper_parser,
            refresh_parse=refresh_parse,
            opendataloader_options=opendataloader_options,
        )
        claims = [str(row.get("claim_id") or "") for row in self.sqlite_db.list_claims_by_entity(f"paper:{token}", limit=12)]
        concepts = [
            str(row.get("canonical_name") or row.get("entity_id") or "")
            for row in self.sqlite_db.get_paper_concepts(token)
        ]
        document_id = f"paper:{token}"
        units = self._build_units(
            document,
            document_id=document_id,
            source_ref=token,
            claims=[item for item in claims if item],
            concepts=[item for item in concepts if item],
            structured_elements=list(parser_payload.get("elements") or []),
            parser_payload=parser_payload,
        )
        stored = self.sqlite_db.replace_document_memory_units(document_id=document_id, units=units)
        self._refresh_updates_relations(document_id=document_id, summary=stored[0] if stored else {})
        return stored

    def build_and_store_web(self, *, canonical_url: str) -> list[dict[str, Any]]:
        token = str(canonical_url).strip()
        document = self._document_from_web(token)
        document_id = make_web_note_id(token)
        units = self._build_units(document, document_id=document_id, source_ref=token)
        stored = self.sqlite_db.replace_document_memory_units(document_id=document_id, units=units)
        self._refresh_updates_relations(document_id=document_id, summary=stored[0] if stored else {})
        return stored

    def _refresh_updates_relations(self, *, document_id: str, summary: dict[str, Any]) -> None:
        if not getattr(self.sqlite_db, "list_memory_relations", None):
            return
        self.sqlite_db.delete_memory_relations_for_node(
            form="document_memory",
            node_id=document_id,
            relation_type="updates",
            direction="src",
        )
        title = str(summary.get("document_title") or summary.get("title") or "").strip()
        document_date = _iso_utc(summary.get("document_date") or summary.get("updated_at") or "")
        if not title or not document_date:
            return
        current_dt = datetime.fromisoformat(document_date.replace("Z", "+00:00"))
        rows = self.sqlite_db.search_document_memory_units(title, limit=100, unit_types=["document_summary"])
        for row in rows:
            other_id = str(row.get("document_id") or "").strip()
            if not other_id or other_id == document_id:
                continue
            overlap = _memory_overlap(
                title + " " + str(summary.get("search_text") or ""),
                str(row.get("document_title") or row.get("title") or "") + " " + str(row.get("search_text") or ""),
            )
            other_date = _iso_utc(row.get("document_date") or row.get("updated_at") or "")
            if overlap < 0.45 or not other_date:
                continue
            other_dt = datetime.fromisoformat(other_date.replace("Z", "+00:00"))
            if current_dt <= other_dt:
                continue
            if not (_update_markers(title, row.get("document_title"), summary.get("search_text"), row.get("search_text")) or overlap >= 0.7):
                continue
            self.sqlite_db.upsert_memory_relation(
                relation_id=_stable_relation_id("document_memory", other_id, "document_memory", document_id, "updates"),
                src_form="document_memory",
                src_id=other_id,
                dst_form="document_memory",
                dst_id=document_id,
                relation_type="updates",
                confidence=round(min(0.99, 0.55 + (0.4 * overlap)), 4),
                provenance={
                    "rule": "document_updates_v1",
                    "older_date": other_date,
                    "newer_date": document_date,
                    "title_overlap": round(overlap, 4),
                },
            )
