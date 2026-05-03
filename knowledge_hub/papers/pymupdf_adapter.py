"""Optional PyMuPDF PDF adapter for lightweight local paper parsing."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _page_total(document: Any) -> int:
    total = getattr(document, "page_count", None)
    if total is None:
        try:
            total = len(document)
        except Exception:
            total = 0
    try:
        return max(0, int(total or 0))
    except Exception:
        return 0


def _extract_document_text(document: Any) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    markdown_lines: list[str] = []
    pages_with_text = 0
    char_count = 0
    for page_index in range(_page_total(document)):
        try:
            page = document.load_page(page_index)
            raw_text = str(page.get_text("text") or "")
        except Exception:
            continue
        text = _clean_text(raw_text)
        if not text:
            continue
        pages_with_text += 1
        char_count += len(text)
        page_number = page_index + 1
        markdown_lines.extend([f"## Page {page_number}", "", text, ""])
        elements.append(
            {
                "type": "paragraph",
                "text": text,
                "page": page_number,
                "heading_path": [f"Page {page_number}"],
                "reading_order": page_index,
            }
        )
    markdown_text = "\n".join(markdown_lines).strip()
    stats = {
        "page_count": _page_total(document),
        "pages_with_text": pages_with_text,
        "char_count": char_count,
    }
    return markdown_text, elements, stats


def _looks_like_scanned_pdf(stats: dict[str, Any]) -> bool:
    page_count = int(stats.get("page_count") or 0)
    pages_with_text = int(stats.get("pages_with_text") or 0)
    char_count = int(stats.get("char_count") or 0)
    if page_count <= 0:
        return False
    if pages_with_text == 0:
        return True
    return char_count < max(80, page_count * 20)


def _ocr_prerequisite_status() -> dict[str, str]:
    paths = {
        "ocrmypdf": str(shutil.which("ocrmypdf") or "").strip(),
        "tesseract": str(shutil.which("tesseract") or "").strip(),
        "gs": str(shutil.which("gs") or "").strip(),
    }
    missing = [name for name, path in paths.items() if not path]
    return {
        "status": "ok" if not missing else "missing",
        "detail": "" if not missing else f"missing {', '.join(missing)}",
        "ocrmypdf": paths["ocrmypdf"],
        "tesseract": paths["tesseract"],
        "gs": paths["gs"],
    }


@dataclass(slots=True)
class PyMuPDFParseResult:
    markdown_text: str
    elements: list[dict[str, Any]]
    parser_meta: dict[str, Any]
    artifact_dir: str
    markdown_path: str
    json_path: str
    manifest_path: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "markdown_text": self.markdown_text,
            "elements": list(self.elements),
            "parser_meta": dict(self.parser_meta),
            "artifact_dir": self.artifact_dir,
            "markdown_path": self.markdown_path,
            "json_path": self.json_path,
            "manifest_path": self.manifest_path,
        }


class PyMuPDFAdapter:
    def __init__(self, *, papers_dir: str):
        self.papers_dir = Path(str(papers_dir)).expanduser()

    def artifact_dir_for(self, *, paper_id: str) -> Path:
        return self.papers_dir / "parsed" / str(paper_id).strip()

    def _manifest_path(self, *, paper_id: str) -> Path:
        return self.artifact_dir_for(paper_id=paper_id) / "manifest.json"

    def _load_existing(self, *, paper_id: str) -> PyMuPDFParseResult | None:
        manifest_path = self._manifest_path(paper_id=paper_id)
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        parser_meta = dict(manifest.get("parser_meta") or {})
        if str(parser_meta.get("parser") or "").strip().lower() != "pymupdf":
            return None
        markdown_path = Path(str(manifest.get("markdown_path") or "").strip())
        json_path = Path(str(manifest.get("json_path") or "").strip())
        if not markdown_path.exists() or not json_path.exists():
            return None
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return PyMuPDFParseResult(
            markdown_text=str(payload.get("markdown_text") or markdown_path.read_text(encoding="utf-8")),
            elements=list(payload.get("elements") or []),
            parser_meta=dict(payload.get("parser_meta") or parser_meta),
            artifact_dir=str(self.artifact_dir_for(paper_id=paper_id)),
            markdown_path=str(markdown_path),
            json_path=str(json_path),
            manifest_path=str(manifest_path),
        )

    def ensure_artifacts(self, *, paper_id: str, pdf_path: str, refresh: bool = False) -> PyMuPDFParseResult:
        token = str(paper_id).strip()
        existing = None if refresh else self._load_existing(paper_id=token)
        if existing is not None:
            return existing

        source_pdf = Path(str(pdf_path).strip())
        if not source_pdf.exists():
            raise RuntimeError(f"paper pdf not found: {source_pdf}")

        try:
            import fitz  # type: ignore
        except Exception as error:
            raise RuntimeError("PyMuPDF is not installed; install it to use --paper-parser pymupdf") from error
        try:
            version = str(importlib.metadata.version("PyMuPDF"))
        except Exception:
            version = "unknown"

        try:
            document = fitz.open(str(source_pdf))
        except Exception as error:
            raise RuntimeError(f"pymupdf parse failed: {error}") from error

        markdown_text = ""
        elements: list[dict[str, Any]] = []
        stats: dict[str, Any] = {}
        try:
            markdown_text, elements, stats = _extract_document_text(document)
        finally:
            try:
                document.close()
            except Exception:
                pass

        ocr_attempted = False
        ocr_applied = False
        ocr_warning = ""
        ocr_output_pdf = ""
        extracted_from = str(source_pdf)
        if _looks_like_scanned_pdf(stats):
            ocr_attempted = True
            prereq = _ocr_prerequisite_status()
            if prereq["status"] != "ok":
                ocr_warning = f"OCR prerequisites unavailable: {prereq['detail']}"
            else:
                artifact_dir = self.artifact_dir_for(paper_id=token)
                artifact_dir.mkdir(parents=True, exist_ok=True)
                output_pdf = artifact_dir / "document.ocr.pdf"
                command = [
                    str(prereq["ocrmypdf"]),
                    "--skip-text",
                    "--quiet",
                    str(source_pdf),
                    str(output_pdf),
                ]
                try:
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    ocr_output_pdf = str(output_pdf)
                    ocr_document = fitz.open(str(output_pdf))
                    try:
                        ocr_markdown_text, ocr_elements, ocr_stats = _extract_document_text(ocr_document)
                    finally:
                        try:
                            ocr_document.close()
                        except Exception:
                            pass
                    if ocr_markdown_text and int(ocr_stats.get("char_count") or 0) >= int(stats.get("char_count") or 0):
                        markdown_text = ocr_markdown_text
                        elements = ocr_elements
                        stats = ocr_stats
                        ocr_applied = True
                        extracted_from = str(output_pdf)
                    else:
                        ocr_warning = "OCR completed but did not recover more usable text"
                except subprocess.CalledProcessError as error:
                    detail = _clean_text(error.stderr or error.stdout or "") or "unknown OCRmyPDF error"
                    ocr_warning = f"OCRmyPDF failed: {detail}"
                except Exception as error:
                    ocr_warning = f"OCR preprocessing failed: {error}"

        markdown_body = markdown_text.strip()
        if not markdown_body:
            reason = ocr_warning or "PyMuPDF extracted no usable text"
            raise RuntimeError(f"pymupdf parse produced no usable text: {reason}")
        markdown_text = f"# {token}\n\n{markdown_body}"
        parser_meta = {
            "parser": "pymupdf",
            "mode": "local",
            "version": version,
            "source_pdf": str(source_pdf),
            "extracted_from": extracted_from,
            "page_count": int(stats.get("page_count") or 0),
            "pages_with_text": int(stats.get("pages_with_text") or 0),
            "char_count": int(stats.get("char_count") or 0),
            "layout_mode": "text_only",
            "text_layer_detected": bool(int(stats.get("pages_with_text") or 0) > 0),
            "ocr_attempted": ocr_attempted,
            "ocr_applied": ocr_applied,
            "ocr_output_pdf": ocr_output_pdf,
            "ocr_warning": ocr_warning,
        }

        artifact_dir = self.artifact_dir_for(paper_id=token)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = artifact_dir / "document.md"
        json_path = artifact_dir / "document.json"
        manifest_path = artifact_dir / "manifest.json"
        markdown_path.write_text(markdown_text, encoding="utf-8")
        json_path.write_text(
            json.dumps(
                {
                    "markdown_text": markdown_text,
                    "elements": elements,
                    "parser_meta": parser_meta,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "paper_id": token,
                    "parser_meta": parser_meta,
                    "markdown_path": str(markdown_path),
                    "json_path": str(json_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return PyMuPDFParseResult(
            markdown_text=markdown_text,
            elements=elements,
            parser_meta=parser_meta,
            artifact_dir=str(artifact_dir),
            markdown_path=str(markdown_path),
            json_path=str(json_path),
            manifest_path=str(manifest_path),
        )


__all__ = ["PyMuPDFAdapter", "PyMuPDFParseResult"]
