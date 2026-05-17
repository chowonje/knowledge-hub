"""Report-only layout parser pilot for complex paper extraction.

The pilot compares parser candidates in an isolated output root.  It never
writes to the configured paper corpus, SQLite, vectors, or document-memory
stores.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import importlib.metadata
from pathlib import Path
import shutil
import time
from typing import Any, Callable

from knowledge_hub.papers.extraction_diagnostics import diagnose_paper_parse
from knowledge_hub.papers.mineru_adapter import MinerUPDFAdapter
from knowledge_hub.papers.opendataloader_adapter import OpenDataLoaderPDFAdapter
from knowledge_hub.papers.pymupdf_adapter import PyMuPDFAdapter


LAYOUT_PARSER_PILOT_SCHEMA_ID = "knowledge-hub.paper.layout-parser-pilot.result.v1"
DEFAULT_LAYOUT_PILOT_PAPERS = (
    "1706.03762",
    "1506.02640",
    "1412.6980",
    "1512.03385",
    "1810.04805",
    "2005.14165",
    "2005.11401",
    "alexnet-2012",
    "1312.5602",
    "2201.11903",
)
SUPPORTED_LAYOUT_PILOT_PARSERS = ("pymupdf", "opendataloader", "mineru")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _paper_id_from_row(row: dict[str, Any], requested_id: str) -> str:
    return _clean_text(row.get("arxiv_id") or row.get("paper_id") or row.get("paperId") or requested_id)


def _source_pdf_for_paper(paper: dict[str, Any], *, papers_dir: str | Path) -> Path | None:
    token = _clean_text(paper.get("pdf_path") or paper.get("pdfPath"))
    if not token:
        return None
    candidate = Path(token).expanduser()
    if not candidate.is_absolute():
        candidate = Path(str(papers_dir)).expanduser() / candidate
    return candidate


def _safe_path(path: str | Path | None, *, root: str | Path | None = None) -> str:
    if path is None:
        return ""
    candidate = Path(str(path)).expanduser()
    if root is not None:
        root_path = Path(str(root)).expanduser()
        try:
            return str(candidate.resolve().relative_to(root_path.resolve()))
        except Exception:
            pass
    return str(candidate)


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path.home() / ".khub" / "reports" / "layout-parser-pilot" / datetime.now().strftime("%Y-%m-%d") / stamp


def _adapter_for_parser(parser: str, *, parser_root: Path) -> Any:
    if parser == "pymupdf":
        return PyMuPDFAdapter(papers_dir=str(parser_root))
    if parser == "opendataloader":
        return OpenDataLoaderPDFAdapter(papers_dir=str(parser_root))
    if parser == "mineru":
        return MinerUPDFAdapter(papers_dir=str(parser_root))
    raise ValueError(f"unsupported parser: {parser}")


def _parser_availability(parser: str) -> dict[str, Any]:
    if parser == "pymupdf":
        module = "fitz"
        package = "PyMuPDF"
        command = ""
    elif parser == "opendataloader":
        module = "opendataloader_pdf"
        package = "opendataloader-pdf"
        command = ""
    elif parser == "mineru":
        module = ""
        package = "mineru"
        command = "mineru"
    else:
        return {"available": False, "reason": "unsupported_parser", "version": "", "command": ""}

    version = ""
    if package:
        try:
            version = str(importlib.metadata.version(package))
        except Exception:
            version = ""
    module_ok = True
    if module:
        try:
            importlib.import_module(module)
        except Exception:
            module_ok = False
    command_ok = True
    if command:
        command_ok = bool(shutil.which(command))
    available = module_ok and command_ok and (bool(version) or parser == "pymupdf")
    reason = ""
    if not module_ok:
        reason = f"missing_module:{module}"
    elif not command_ok:
        reason = f"missing_command:{command}"
    elif not version and parser != "pymupdf":
        reason = f"missing_package:{package}"
    return {
        "available": bool(available),
        "reason": reason,
        "version": version,
        "command": command,
    }


def _real_heading_count(elements: list[dict[str, Any]]) -> int:
    count = 0
    for element in elements:
        path = element.get("heading_path") or element.get("headingPath")
        if isinstance(path, list):
            for item in path:
                token = _clean_text(item)
                if token and not token.casefold().startswith("page "):
                    count += 1
        element_type = _clean_text(element.get("type") or element.get("kind")).casefold()
        text = _clean_text(element.get("text") or element.get("content") or element.get("markdown"))
        if element_type in {"heading", "section", "title"} and text and not text.casefold().startswith("page "):
            count += 1
    return count


def _layout_metrics(*, markdown_text: str, elements: list[dict[str, Any]], diagnostic: dict[str, Any]) -> dict[str, Any]:
    bbox_count = 0
    reading_order_count = 0
    table_cell_count = 0
    for element in elements:
        if element.get("bbox") is not None:
            bbox_count += 1
        if element.get("reading_order") is not None or element.get("readingOrder") is not None:
            reading_order_count += 1
        if element.get("row_number") is not None or element.get("column_number") is not None:
            table_cell_count += 1
    return {
        "markdownChars": len(markdown_text or ""),
        "elementCount": len(elements),
        "realHeadingCount": _real_heading_count(elements),
        "bboxElementCount": bbox_count,
        "readingOrderElementCount": reading_order_count,
        "tableCellElementCount": table_cell_count,
        "pageCount": int(diagnostic.get("pageCount") or 0),
        "columnCountDetected": int(diagnostic.get("columnCountDetected") or 0),
        "tablesDetected": int(diagnostic.get("tablesDetected") or 0),
        "figuresDetected": int(diagnostic.get("figuresDetected") or 0),
        "equationsDetected": int(diagnostic.get("equationsDetected") or 0),
        "extractionDegraded": bool(diagnostic.get("extractionDegraded")),
        "degradationReasons": list(diagnostic.get("degradationReasons") or []),
    }


@dataclass(slots=True)
class _ParserRun:
    markdown_text: str
    elements: list[dict[str, Any]]
    parser_meta: dict[str, Any]
    artifact_dir: str


def _run_parser(
    *,
    parser: str,
    paper_id: str,
    pdf_path: Path,
    parser_root: Path,
) -> _ParserRun:
    adapter = _adapter_for_parser(parser, parser_root=parser_root)
    if parser == "pymupdf":
        result = adapter.ensure_artifacts(
            paper_id=paper_id,
            pdf_path=str(pdf_path),
            refresh=True,
            allow_ocr=False,
        )
    else:
        result = adapter.ensure_artifacts(
            paper_id=paper_id,
            pdf_path=str(pdf_path),
            refresh=True,
        )
    return _ParserRun(
        markdown_text=str(result.markdown_text),
        elements=[dict(item) for item in list(result.elements or []) if isinstance(item, dict)],
        parser_meta=dict(result.parser_meta or {}),
        artifact_dir=str(result.artifact_dir),
    )


def _parser_item(
    *,
    parser: str,
    paper_id: str,
    parser_root: Path,
    status: str,
    reason: str,
    available: dict[str, Any],
    duration_ms: int = 0,
    run: _ParserRun | None = None,
    diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elements = list(run.elements if run is not None else [])
    markdown_text = str(run.markdown_text if run is not None else "")
    diagnostic = dict(diagnostic or {})
    return {
        "parser": parser,
        "status": status,
        "reason": reason,
        "available": bool(available.get("available")),
        "availability": dict(available),
        "durationMs": int(duration_ms),
        "artifactDir": _safe_path(run.artifact_dir, root=parser_root) if run is not None else "",
        "metrics": _layout_metrics(markdown_text=markdown_text, elements=elements, diagnostic=diagnostic),
    }


def _counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"planned": 0, "ok": 0, "blocked": 0, "failed": 0}
    for paper in items:
        for parser_item in list(paper.get("parsers") or []):
            status = str(parser_item.get("status") or "")
            if status in counts:
                counts[status] += 1
    return counts


def _overall_status(counts: dict[str, int]) -> str:
    if counts.get("failed", 0):
        return "failed"
    if counts.get("blocked", 0) and (counts.get("ok", 0) or counts.get("planned", 0)):
        return "partial"
    if counts.get("blocked", 0):
        return "blocked"
    return "ok"


def run_layout_parser_pilot(
    *,
    sqlite_db: Any,
    papers_dir: str | Path,
    paper_ids: list[str] | tuple[str, ...] | None = None,
    parsers: list[str] | tuple[str, ...] | None = None,
    output_dir: str | Path | None = None,
    run: bool = False,
) -> dict[str, Any]:
    """Plan or run isolated parser comparisons for explicit paper ids."""

    requested_ids = [_clean_text(item) for item in list(paper_ids or []) if _clean_text(item)]
    if not requested_ids:
        requested_ids = list(DEFAULT_LAYOUT_PILOT_PAPERS)
    parser_tokens = [_clean_text(item).casefold() for item in list(parsers or []) if _clean_text(item)]
    if not parser_tokens:
        parser_tokens = ["pymupdf", "opendataloader", "mineru"]
    output_root = Path(str(output_dir)).expanduser() if output_dir is not None else _default_output_dir()
    if run:
        output_root.mkdir(parents=True, exist_ok=True)

    availability = {parser: _parser_availability(parser) for parser in parser_tokens}
    items: list[dict[str, Any]] = []
    warnings: list[str] = []
    for requested_id in requested_ids:
        paper = sqlite_db.get_paper(requested_id) if hasattr(sqlite_db, "get_paper") else None
        if not isinstance(paper, dict) or not paper:
            items.append(
                {
                    "paperId": requested_id,
                    "paperTitle": "",
                    "sourcePdf": {"exists": False, "path": ""},
                    "parsers": [
                        _parser_item(
                            parser=parser,
                            paper_id=requested_id,
                            parser_root=output_root / parser,
                            status="blocked",
                            reason="paper_not_registered",
                            available=availability.get(parser, {"available": False}),
                        )
                        for parser in parser_tokens
                    ],
                }
            )
            continue
        paper_id = _paper_id_from_row(paper, requested_id)
        title = _clean_text(paper.get("title") or paper.get("paperTitle"))
        pdf_path = _source_pdf_for_paper(paper, papers_dir=papers_dir)
        pdf_exists = bool(pdf_path and pdf_path.exists())
        parser_items: list[dict[str, Any]] = []
        for parser in parser_tokens:
            available = dict(availability.get(parser) or {"available": False, "reason": "unsupported_parser"})
            parser_root = output_root / parser
            if parser not in SUPPORTED_LAYOUT_PILOT_PARSERS:
                parser_items.append(
                    _parser_item(
                        parser=parser,
                        paper_id=paper_id,
                        parser_root=parser_root,
                        status="blocked",
                        reason="unsupported_parser",
                        available=available,
                    )
                )
                continue
            if not pdf_exists or pdf_path is None:
                parser_items.append(
                    _parser_item(
                        parser=parser,
                        paper_id=paper_id,
                        parser_root=parser_root,
                        status="blocked",
                        reason="source_pdf_missing",
                        available=available,
                    )
                )
                continue
            if not available.get("available"):
                parser_items.append(
                    _parser_item(
                        parser=parser,
                        paper_id=paper_id,
                        parser_root=parser_root,
                        status="blocked",
                        reason=str(available.get("reason") or "parser_unavailable"),
                        available=available,
                    )
                )
                continue
            if not run:
                parser_items.append(
                    _parser_item(
                        parser=parser,
                        paper_id=paper_id,
                        parser_root=parser_root,
                        status="planned",
                        reason="run_required",
                        available=available,
                    )
                )
                continue
            started = time.perf_counter()
            try:
                parser_run = _run_parser(
                    parser=parser,
                    paper_id=paper_id,
                    pdf_path=pdf_path,
                    parser_root=parser_root,
                )
                diagnostic = diagnose_paper_parse(
                    paper_id=paper_id,
                    papers_dir=parser_root,
                    manifest={"parser_meta": parser_run.parser_meta},
                    document={
                        "parser_meta": parser_run.parser_meta,
                        "elements": parser_run.elements,
                    },
                )
                parser_items.append(
                    _parser_item(
                        parser=parser,
                        paper_id=paper_id,
                        parser_root=parser_root,
                        status="ok",
                        reason="ok",
                        available=available,
                        duration_ms=int((time.perf_counter() - started) * 1000),
                        run=parser_run,
                        diagnostic=diagnostic,
                    )
                )
            except Exception as error:
                parser_items.append(
                    _parser_item(
                        parser=parser,
                        paper_id=paper_id,
                        parser_root=parser_root,
                        status="failed",
                        reason=f"parser_failed: {_clean_text(error)}",
                        available=available,
                        duration_ms=int((time.perf_counter() - started) * 1000),
                    )
                )
        items.append(
            {
                "paperId": paper_id,
                "paperTitle": title,
                "sourcePdf": {
                    "exists": pdf_exists,
                    "path": _safe_path(pdf_path, root=papers_dir) if pdf_path else "",
                },
                "parsers": parser_items,
            }
        )

    counts = _counts(items)
    return {
        "schema": LAYOUT_PARSER_PILOT_SCHEMA_ID,
        "status": _overall_status(counts),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "request": {
            "paperIds": requested_ids,
            "parsers": parser_tokens,
            "run": bool(run),
            "outputDir": str(output_root),
        },
        "counts": counts,
        "papers": items,
        "warnings": warnings,
    }


__all__ = [
    "DEFAULT_LAYOUT_PILOT_PAPERS",
    "LAYOUT_PARSER_PILOT_SCHEMA_ID",
    "SUPPORTED_LAYOUT_PILOT_PARSERS",
    "run_layout_parser_pilot",
]
