"""Read-only diagnostics for existing parsed paper artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


EXTRACTION_REPORT_SCHEMA_ID = "knowledge-hub.paper.extraction-report.v1"

_PAGE_HEADING_RE = re.compile(r"^page\s+\d+$", re.IGNORECASE)
_TABLE_CAPTION_RE = re.compile(r"\btable\s+\d+\b", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(r"\b(?:fig\.?|figure)\s+\d+\b", re.IGNORECASE)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value or "").strip().casefold()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _meta_value(meta: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in meta and meta.get(name) is not None:
            return meta.get(name)
    return default


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, "missing"
    except Exception as error:
        return {}, f"invalid: {error}"
    if not isinstance(payload, dict):
        return {}, "invalid: expected object"
    return payload, ""


def _artifact_paths(papers_dir: str | Path, paper_id: str, manifest: dict[str, Any] | None = None) -> dict[str, str]:
    _ = papers_dir
    manifest = dict(manifest or {})
    base = Path("parsed") / paper_id
    document_json_path = Path(str(manifest.get("json_path") or base / "document.json"))
    document_markdown_path = Path(str(manifest.get("markdown_path") or base / "document.md"))
    if document_json_path.is_absolute():
        document_json_path = base / "document.json"
    if document_markdown_path.is_absolute():
        document_markdown_path = base / "document.md"
    return {
        "artifactDir": str(base),
        "manifestPath": str(base / "manifest.json"),
        "documentJsonPath": str(document_json_path),
        "documentMarkdownPath": str(document_markdown_path),
    }


def _absolute_artifact_paths(papers_dir: str | Path, paper_id: str, manifest: dict[str, Any] | None = None) -> dict[str, Path]:
    artifact_dir = Path(str(papers_dir)).expanduser() / "parsed" / paper_id
    manifest = dict(manifest or {})
    manifest_path = artifact_dir / "manifest.json"
    document_json_path = Path(str(manifest.get("json_path") or artifact_dir / "document.json"))
    document_markdown_path = Path(str(manifest.get("markdown_path") or artifact_dir / "document.md"))
    return {
        "artifactDir": artifact_dir,
        "manifestPath": manifest_path,
        "documentJsonPath": document_json_path,
        "documentMarkdownPath": document_markdown_path,
    }


def _element_type(element: dict[str, Any]) -> str:
    return _clean_text(element.get("type") or element.get("kind") or element.get("content_type")).casefold()


def _element_text(element: dict[str, Any]) -> str:
    return _clean_text(element.get("text") or element.get("markdown") or element.get("content"))


def _elements_with_text(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in elements if _element_text(item)]


def _max_page(elements: list[dict[str, Any]]) -> int:
    return max((_safe_int(item.get("page")) for item in elements), default=0)


def _pages_with_text(elements: list[dict[str, Any]]) -> int:
    pages = {_safe_int(item.get("page")) for item in _elements_with_text(elements)}
    return len({page for page in pages if page > 0})


def _column_count_from_distribution(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    detected = 0
    for raw_key, raw_count in value.items():
        count = _safe_int(raw_count)
        if count <= 0:
            continue
        detected = max(detected, _safe_int(raw_key))
    return detected


def _column_count_from_elements(elements: list[dict[str, Any]]) -> int:
    per_page: dict[int, set[int]] = {}
    for element in elements:
        page = _safe_int(element.get("page"))
        raw_column = element.get("column_index")
        if raw_column is None:
            locator = element.get("locator")
            if isinstance(locator, dict):
                raw_column = locator.get("column_index")
        if page <= 0 or raw_column is None:
            continue
        per_page.setdefault(page, set()).add(_safe_int(raw_column))
    return max((len(columns) for columns in per_page.values()), default=0)


def _infer_column_count(meta: dict[str, Any], elements: list[dict[str, Any]]) -> int:
    explicit = _safe_int(_meta_value(meta, "column_count_detected", "columnCountDetected", default=0))
    if explicit > 0:
        return explicit
    from_distribution = _column_count_from_distribution(
        _meta_value(meta, "column_count_distribution", "columnCountDistribution", default={})
    )
    if from_distribution > 0:
        return from_distribution
    return _column_count_from_elements(elements)


def _real_heading_paths(elements: list[dict[str, Any]]) -> list[str]:
    headings: list[str] = []
    for element in elements:
        path = element.get("heading_path") or element.get("headingPath")
        items = list(path) if isinstance(path, list) else []
        for item in items:
            token = _clean_text(item)
            if token and not _PAGE_HEADING_RE.match(token):
                headings.append(token)
        element_type = _element_type(element)
        text = _element_text(element)
        if element_type in {"heading", "section", "title"} and text and not _PAGE_HEADING_RE.match(text):
            headings.append(text)
    return headings


def _has_only_page_headings(elements: list[dict[str, Any]]) -> bool:
    if not elements:
        return False
    if _real_heading_paths(elements):
        return False
    heading_values: list[str] = []
    for element in elements:
        path = element.get("heading_path") or element.get("headingPath")
        if isinstance(path, list):
            heading_values.extend(_clean_text(item) for item in path if _clean_text(item))
    return bool(heading_values) and all(_PAGE_HEADING_RE.match(item) for item in heading_values)


def _count_tables(elements: list[dict[str, Any]]) -> int:
    table_ids: set[str] = set()
    table_like = 0
    cell_like = 0
    for index, element in enumerate(elements):
        element_type = _element_type(element)
        has_cell_coordinates = element.get("row_number") is not None or element.get("column_number") is not None
        if has_cell_coordinates:
            cell_like += 1
            linked = _clean_text(element.get("linked_content_id") or element.get("linkedContentId"))
            if linked:
                table_ids.add(linked)
                continue
        if "table" in element_type:
            table_like += 1
            token = _clean_text(element.get("id") or element.get("table_id") or element.get("tableId"))
            if token:
                table_ids.add(token)
            else:
                table_ids.add(f"table:{index}")
    if table_ids:
        return len(table_ids)
    if cell_like:
        return 1
    return table_like


def _count_figures(elements: list[dict[str, Any]]) -> int:
    figure_ids: set[str] = set()
    caption_count = 0
    for index, element in enumerate(elements):
        element_type = _element_type(element)
        text = _element_text(element)
        if any(token in element_type for token in ("figure", "image")):
            figure_ids.add(_clean_text(element.get("id") or element.get("figure_id") or element.get("figureId")) or f"figure:{index}")
            continue
        if text and _FIGURE_CAPTION_RE.search(text):
            caption_count += 1
    return len(figure_ids) if figure_ids else caption_count


def _count_equations(elements: list[dict[str, Any]]) -> int:
    count = 0
    for element in elements:
        element_type = _element_type(element)
        if "equation" in element_type or "formula" in element_type:
            count += 1
    return count


def _caption_detected(pattern: re.Pattern[str], elements: list[dict[str, Any]]) -> bool:
    return any(pattern.search(_element_text(element)) for element in elements)


def _diagnostic_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    diagnostic = meta.get("extraction_diagnostic") or meta.get("extractionDiagnostic")
    return dict(diagnostic) if isinstance(diagnostic, dict) else {}


def diagnose_paper_parse(
    *,
    paper_id: str,
    papers_dir: str | Path,
    manifest: dict[str, Any] | None = None,
    document: dict[str, Any] | None = None,
    manifest_error: str = "",
    document_error: str = "",
) -> dict[str, Any]:
    """Build a structural diagnostic from existing parsed artifacts only."""

    manifest = dict(manifest or {})
    document = dict(document or {})
    manifest_meta = dict(manifest.get("parser_meta") or {})
    document_meta = dict(document.get("parser_meta") or {})
    meta = {**manifest_meta, **document_meta}
    existing_diagnostic = _diagnostic_from_meta(meta)
    elements = [dict(item) for item in list(document.get("elements") or []) if isinstance(item, dict)]

    parser = _clean_text(
        existing_diagnostic.get("parser")
        or _meta_value(meta, "parser", "parser_name", "parserName", default="")
        or "unknown"
    )
    page_count = _safe_int(
        existing_diagnostic.get("pageCount")
        or _meta_value(meta, "page_count", "pageCount", default=0)
        or _max_page(elements)
    )
    pages_with_text = _safe_int(
        existing_diagnostic.get("pagesWithText")
        or _meta_value(meta, "pages_with_text", "pagesWithText", default=0)
        or _pages_with_text(elements)
    )
    text_layer_detected = _safe_bool(
        existing_diagnostic.get("textLayerDetected")
        if "textLayerDetected" in existing_diagnostic
        else _meta_value(meta, "text_layer_detected", "textLayerDetected", default=bool(pages_with_text > 0)),
        default=bool(pages_with_text > 0),
    )
    ocr_attempted = _safe_bool(
        existing_diagnostic.get("ocrAttempted")
        if "ocrAttempted" in existing_diagnostic
        else _meta_value(meta, "ocr_attempted", "ocrAttempted", default=False)
    )
    ocr_applied = _safe_bool(
        existing_diagnostic.get("ocrApplied")
        if "ocrApplied" in existing_diagnostic
        else _meta_value(meta, "ocr_applied", "ocrApplied", default=False)
    )
    column_count = _safe_int(existing_diagnostic.get("columnCountDetected") or _infer_column_count(meta, elements))
    tables_detected = _safe_int(existing_diagnostic.get("tablesDetected") or _meta_value(meta, "tables_detected", "tablesDetected", default=0))
    if tables_detected <= 0:
        tables_detected = _count_tables(elements)
    figures_detected = _safe_int(existing_diagnostic.get("figuresDetected") or _meta_value(meta, "figures_detected", "figuresDetected", default=0))
    if figures_detected <= 0:
        figures_detected = _count_figures(elements)
    equations_detected = _safe_int(existing_diagnostic.get("equationsDetected") or _meta_value(meta, "equations_detected", "equationsDetected", default=0))
    if equations_detected <= 0:
        equations_detected = _count_equations(elements)

    reasons: list[str] = []

    def _add_reason(reason: str) -> None:
        if reason and reason not in reasons:
            reasons.append(reason)

    for item in list(existing_diagnostic.get("degradationReasons") or []):
        _add_reason(_clean_text(item))
    if manifest_error:
        _add_reason("parsed_artifact_missing" if manifest_error == "missing" else "parsed_artifact_unreadable")
    if document_error:
        _add_reason("parsed_artifact_missing" if document_error == "missing" else "parsed_artifact_unreadable")
        _add_reason("parsed_document_missing")
    if parser.casefold() == "pymupdf" and _has_only_page_headings(elements):
        _add_reason("page_blob_sections_only")
    if page_count > 0 and pages_with_text / max(1, page_count) < 0.6:
        _add_reason("low_text_coverage")
    if page_count > 0 and not text_layer_detected:
        _add_reason("text_layer_missing")
    if ocr_attempted and not ocr_applied:
        _add_reason("ocr_attempted_not_applied")
    if parser.casefold() == "pymupdf" and column_count >= 2:
        _add_reason("multi_column_probe_only")
    if tables_detected <= 0 and _caption_detected(_TABLE_CAPTION_RE, elements):
        _add_reason("tables_caption_only")

    degraded = bool(reasons) or _safe_bool(existing_diagnostic.get("extractionDegraded"), default=False)
    paths = _artifact_paths(papers_dir, str(paper_id).strip(), manifest)
    return {
        "paperId": str(paper_id).strip(),
        "artifactPaths": paths,
        "parser": parser,
        "pageCount": page_count,
        "pagesWithText": pages_with_text,
        "textLayerDetected": bool(text_layer_detected),
        "ocrAttempted": bool(ocr_attempted),
        "ocrApplied": bool(ocr_applied),
        "columnCountDetected": column_count,
        "tablesDetected": tables_detected,
        "figuresDetected": figures_detected,
        "equationsDetected": equations_detected,
        "extractionDegraded": bool(degraded),
        "degradationReasons": reasons,
        "readingOrderMethod": _clean_text(
            existing_diagnostic.get("readingOrderMethod")
            or _meta_value(meta, "reading_order_method", "readingOrderMethod", default="")
        ),
    }


def load_paper_diagnostic(*, paper: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
    paper_id = _clean_text(paper.get("arxiv_id") or paper.get("paper_id") or paper.get("paperId"))
    absolute_paths = _absolute_artifact_paths(papers_dir, paper_id)
    manifest_path = absolute_paths["manifestPath"]
    manifest, manifest_error = _read_json(manifest_path)
    absolute_paths = _absolute_artifact_paths(papers_dir, paper_id, manifest)
    document_path = absolute_paths["documentJsonPath"]
    document, document_error = _read_json(document_path)
    diagnostic = diagnose_paper_parse(
        paper_id=paper_id,
        papers_dir=papers_dir,
        manifest=manifest,
        document=document,
        manifest_error=manifest_error,
        document_error=document_error,
    )
    return {
        "paperId": paper_id,
        "paperTitle": _clean_text(paper.get("title") or paper.get("paperTitle")),
        "artifactPaths": dict(diagnostic.get("artifactPaths") or {}),
        "diagnostic": diagnostic,
        "warnings": list(diagnostic.get("degradationReasons") or []),
    }


def _selected_papers(sqlite_db: Any, paper_ids: list[str] | tuple[str, ...], limit: int) -> list[dict[str, Any]]:
    if paper_ids:
        selected: list[dict[str, Any]] = []
        for raw in paper_ids:
            paper_id = _clean_text(raw)
            if not paper_id:
                continue
            row = sqlite_db.get_paper(paper_id) if hasattr(sqlite_db, "get_paper") else None
            if isinstance(row, dict) and row:
                selected.append(row)
            else:
                selected.append({"arxiv_id": paper_id, "title": ""})
        return selected[:limit] if limit > 0 else selected
    effective_limit = limit if limit > 0 else 200000
    return list(sqlite_db.list_papers(limit=effective_limit) or [])


def build_extraction_report(
    *,
    sqlite_db: Any,
    papers_dir: str | Path,
    paper_ids: list[str] | tuple[str, ...] | None = None,
    limit: int = 0,
    degraded_only: bool = False,
) -> dict[str, Any]:
    requested_ids = [_clean_text(item) for item in list(paper_ids or []) if _clean_text(item)]
    rows = _selected_papers(sqlite_db, requested_ids, max(0, int(limit or 0)))
    items = [load_paper_diagnostic(paper=row, papers_dir=papers_dir) for row in rows]
    if degraded_only:
        items = [item for item in items if bool(dict(item.get("diagnostic") or {}).get("extractionDegraded"))]
    degraded_count = sum(1 for item in items if bool(dict(item.get("diagnostic") or {}).get("extractionDegraded")))
    missing_count = sum(
        1
        for item in items
        if "parsed_artifact_missing" in list(dict(item.get("diagnostic") or {}).get("degradationReasons") or [])
    )
    status = "degraded" if degraded_count else "ok"
    return {
        "schema": EXTRACTION_REPORT_SCHEMA_ID,
        "status": status,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "request": {
            "paperIds": requested_ids,
            "limit": max(0, int(limit or 0)),
            "degradedOnly": bool(degraded_only),
        },
        "counts": {
            "scannedPapers": len(rows),
            "reportedPapers": len(items),
            "degradedPapers": degraded_count,
            "missingParsedArtifacts": missing_count,
        },
        "papers": items,
        "warnings": [],
    }


def build_parser_meta_diagnostic(parser_meta: dict[str, Any], elements: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the additive parser_meta.extraction_diagnostic object for future parses."""

    diagnostic = diagnose_paper_parse(
        paper_id="",
        papers_dir="",
        manifest={"parser_meta": parser_meta},
        document={"parser_meta": parser_meta, "elements": elements},
    )
    return {
        key: value
        for key, value in diagnostic.items()
        if key not in {"paperId", "artifactPaths"}
    }


__all__ = [
    "EXTRACTION_REPORT_SCHEMA_ID",
    "build_extraction_report",
    "build_parser_meta_diagnostic",
    "diagnose_paper_parse",
    "load_paper_diagnostic",
]
