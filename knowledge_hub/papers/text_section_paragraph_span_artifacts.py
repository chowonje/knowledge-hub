"""Text-derived SectionSpan and ParagraphSpan candidates for local paper PDFs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from knowledge_hub.papers.figure_caption_artifact_vertical_slice import (
    PaperSpec,
    default_paper_specs,
    default_papers_root,
    normalize_text,
    sha256_file,
    sha256_text,
    utc_now_iso,
)

TEXT_SPAN_ARTIFACT_CANDIDATE_SCHEMA_ID = "knowledge-hub.paper.text-span-artifact-candidate.v1"
TEXT_SECTION_PARAGRAPH_SPAN_REPORT_SCHEMA_ID = (
    "knowledge-hub.paper.text-section-paragraph-span-artifacts-report.v1"
)

EXTRACTION_METHOD = "pymupdf_text_block_span_candidate_v1"
CHAR_BASIS = "pymupdf_block_reading_order_normalized_text_v1"

CAPTION_PREFIX_RE = re.compile(r"^\s*(Figure|Fig\.|Table)\s+[0-9]+[A-Za-z]?\s*[:.\-]", re.IGNORECASE)
SECTION_NUMBER_RE = re.compile(r"^\s*(?:[0-9]+(?:\.[0-9]+){0,2}|[A-Z])\s+([A-Z][A-Za-z][A-Za-z0-9 ,:/-]{1,80})$")
SECTION_KEYWORDS = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "approach",
    "experiments",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "references",
    "appendix",
}
PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)


def _short_hash(value: str, *, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _slug(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return token or "unknown"


def _round_bbox(values: Sequence[Any]) -> list[float]:
    return [round(float(value), 2) for value in values[:4]]


def _paper_ref(paper: PaperSpec) -> str:
    return f"papers_dir/{paper.filename}"


def _load_pdf_blocks(pdf_path: Path) -> tuple[int, list[tuple[int, list[Sequence[Any]]]]]:
    try:
        import fitz  # type: ignore
    except Exception as error:  # pragma: no cover
        raise RuntimeError("PyMuPDF is not installed; install it to run text span extraction") from error

    try:
        document = fitz.open(str(pdf_path))
    except Exception as error:
        raise RuntimeError(f"pymupdf open failed: {error}") from error

    pages: list[tuple[int, list[Sequence[Any]]]] = []
    try:
        page_count = int(getattr(document, "page_count", 0) or len(document))
        for page_index in range(page_count):
            page = document.load_page(page_index)
            blocks = sorted(
                list(page.get_text("blocks") or []),
                key=lambda block: (round(float(block[1]), 1), round(float(block[0]), 1)),
            )
            pages.append((page_index + 1, blocks))
    finally:
        try:
            document.close()
        except Exception:
            pass
    return page_count, pages


def _first_line(text: str) -> str:
    for line in str(text or "").splitlines():
        cleaned = normalize_text(line)
        if cleaned:
            return cleaned
    return normalize_text(text)


def _is_caption(text: str) -> bool:
    return bool(CAPTION_PREFIX_RE.match(normalize_text(text)))


def _is_section_heading(text: str) -> bool:
    first = _first_line(text)
    lowered = first.lower().strip(" .:")
    if lowered in SECTION_KEYWORDS:
        return True
    if SECTION_NUMBER_RE.match(first):
        return True
    return False


def _span_artifact_id(*, paper_id: str, span_type: str, page: int, char_start: int, text: str) -> str:
    basis = json.dumps(
        {
            "paperId": paper_id,
            "spanType": span_type,
            "page": page,
            "charStart": char_start,
            "text": text,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return f"{span_type}-span:{_slug(paper_id)}:{page}:{_short_hash(basis)}"


def extract_text_span_candidates_from_blocks(
    *,
    paper_id: str,
    paper_ref: str,
    source_content_hash: str,
    blocks_by_page: Iterable[tuple[int, Iterable[Sequence[Any]]]],
    max_section_rows: int = 12,
    max_paragraph_rows: int = 24,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    section_rows = 0
    paragraph_rows = 0
    cursor = 0
    assembly_parts: list[str] = []
    current_heading = ""

    for page_number, blocks in blocks_by_page:
        for block_index, block in enumerate(blocks):
            if len(block) < 5:
                continue
            text = normalize_text(block[4])
            if len(text) < 8 or _is_caption(text):
                continue
            if assembly_parts:
                cursor += 2
            char_start = cursor
            char_end = char_start + len(text)
            cursor = char_end
            assembly_parts.append(text)

            bbox = _round_bbox(block[:4])
            is_section = _is_section_heading(str(block[4]))
            if is_section and section_rows < max_section_rows:
                current_heading = _first_line(str(block[4]))
                confidence = 0.74
                artifact_id = _span_artifact_id(
                    paper_id=paper_id,
                    span_type="section",
                    page=int(page_number),
                    char_start=char_start,
                    text=text,
                )
                candidates.append(
                    {
                        "schema": TEXT_SPAN_ARTIFACT_CANDIDATE_SCHEMA_ID,
                        "paperId": paper_id,
                        "paperRef": paper_ref,
                        "artifactId": artifact_id,
                        "spanType": "section",
                        "sourceContentHash": source_content_hash,
                        "charBasis": CHAR_BASIS,
                        "charStart": char_start,
                        "charEnd": char_end,
                        "page": int(page_number),
                        "bbox": bbox,
                        "text": text,
                        "textHash": sha256_text(text),
                        "headingPath": [current_heading],
                        "extractionMethod": EXTRACTION_METHOD,
                        "confidence": confidence,
                        "blockerReason": "",
                        "diagnostics": {
                            "source": "pymupdf_block",
                            "blockIndex": int(block_index),
                            "candidateRule": "section_heading",
                        },
                    }
                )
                section_rows += 1

            if paragraph_rows < max_paragraph_rows and len(text) >= 80 and not is_section:
                artifact_id = _span_artifact_id(
                    paper_id=paper_id,
                    span_type="paragraph",
                    page=int(page_number),
                    char_start=char_start,
                    text=text,
                )
                candidates.append(
                    {
                        "schema": TEXT_SPAN_ARTIFACT_CANDIDATE_SCHEMA_ID,
                        "paperId": paper_id,
                        "paperRef": paper_ref,
                        "artifactId": artifact_id,
                        "spanType": "paragraph",
                        "sourceContentHash": source_content_hash,
                        "charBasis": CHAR_BASIS,
                        "charStart": char_start,
                        "charEnd": char_end,
                        "page": int(page_number),
                        "bbox": bbox,
                        "text": text,
                        "textHash": sha256_text(text),
                        "headingPath": [current_heading] if current_heading else [],
                        "extractionMethod": EXTRACTION_METHOD,
                        "confidence": 0.8,
                        "blockerReason": "",
                        "diagnostics": {
                            "source": "pymupdf_block",
                            "blockIndex": int(block_index),
                            "candidateRule": "paragraph_block",
                        },
                    }
                )
                paragraph_rows += 1

    assembly_text = "\n\n".join(assembly_parts)
    diagnostics = {
        "paperId": paper_id,
        "paperRef": paper_ref,
        "textAssemblyHash": sha256_text(assembly_text) if assembly_text else "",
        "assemblyCharCount": len(assembly_text),
        "sectionSpanRows": section_rows,
        "paragraphSpanRows": paragraph_rows,
    }
    return candidates, diagnostics


def extract_text_span_candidates_from_pdf(*, paper: PaperSpec, pdf_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_content_hash = sha256_file(pdf_path)
    page_count, blocks_by_page = _load_pdf_blocks(pdf_path)
    candidates, diagnostics = extract_text_span_candidates_from_blocks(
        paper_id=paper.paper_id,
        paper_ref=_paper_ref(paper),
        source_content_hash=source_content_hash,
        blocks_by_page=blocks_by_page,
    )
    diagnostics["pageCount"] = page_count
    diagnostics["candidateRows"] = len(candidates)
    return candidates, diagnostics


def _blocker_row(*, paper: PaperSpec, reason: str, detail: str = "") -> dict[str, Any]:
    return {
        "paperId": paper.paper_id,
        "paperRef": _paper_ref(paper),
        "blockerReason": reason,
        "detail": detail,
    }


def _private_path_leak_count(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return 1 if PRIVATE_PATH_RE.search(encoded) else 0


def build_text_section_paragraph_span_report(
    *,
    papers_root: Path | None = None,
    paper_specs: Sequence[PaperSpec] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    root = papers_root or default_papers_root()
    papers = list(paper_specs or default_paper_specs())
    candidates: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    paper_diagnostics: list[dict[str, Any]] = []

    for paper in papers:
        pdf_path = root / paper.filename
        if not pdf_path.exists():
            blockers.append(_blocker_row(paper=paper, reason="source_pdf_missing", detail=_paper_ref(paper)))
            continue
        try:
            paper_candidates, diagnostics = extract_text_span_candidates_from_pdf(paper=paper, pdf_path=pdf_path)
        except Exception as error:
            blockers.append(_blocker_row(paper=paper, reason="text_span_extraction_failed", detail=normalize_text(error)))
            continue
        if not paper_candidates:
            blockers.append(_blocker_row(paper=paper, reason="text_span_candidates_not_found"))
        candidates.extend(paper_candidates)
        paper_diagnostics.append(diagnostics)

    section_rows = sum(1 for row in candidates if row.get("spanType") == "section")
    paragraph_rows = sum(1 for row in candidates if row.get("spanType") == "paragraph")
    ready_papers = len({row.get("paperId") for row in candidates})
    report: dict[str, Any] = {
        "schema": TEXT_SECTION_PARAGRAPH_SPAN_REPORT_SCHEMA_ID,
        "status": "ready" if ready_papers >= 3 and paragraph_rows > 0 else "blocked",
        "generatedAt": generated_at or utc_now_iso(),
        "scope": {
            "paperRows": len(papers),
            "paperRefs": [_paper_ref(paper) for paper in papers],
            "writes": "report_only",
            "canonicalParsedArtifactWriteRows": 0,
            "charBasis": CHAR_BASIS,
        },
        "paperDiagnostics": paper_diagnostics,
        "candidateRows": len(candidates),
        "sectionSpanRows": section_rows,
        "paragraphSpanRows": paragraph_rows,
        "candidates": candidates,
        "blockerRows": len(blockers),
        "blockers": blockers,
        "mutationCounters": {
            "canonicalParsedArtifactWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "strictEvidencePromotionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
        },
        "schemaViolationCount": 0,
        "privatePathLeakRows": 0,
        "warnings": [
            "char offsets are candidate provenance over PyMuPDF block reading-order normalized text, not canonical parsed artifact offsets"
        ],
        "schemaErrors": [],
    }
    report["privatePathLeakRows"] = _private_path_leak_count(report)
    if report["privatePathLeakRows"]:
        report["status"] = "blocked"
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Text Section/Paragraph Span Artifacts",
        "",
        f"- status: `{report.get('status')}`",
        f"- candidateRows: `{report.get('candidateRows')}`",
        f"- sectionSpanRows: `{report.get('sectionSpanRows')}`",
        f"- paragraphSpanRows: `{report.get('paragraphSpanRows')}`",
        f"- blockerRows: `{report.get('blockerRows')}`",
        f"- privatePathLeakRows: `{report.get('privatePathLeakRows')}`",
        "",
        "## Paper Diagnostics",
        "",
        "| paperId | sections | paragraphs | chars |",
        "|---|---:|---:|---:|",
    ]
    for row in report.get("paperDiagnostics", []):
        lines.append(
            f"| {row.get('paperId')} | {row.get('sectionSpanRows')} | {row.get('paragraphSpanRows')} | {row.get('assemblyCharCount')} |"
        )
    lines.extend(["", "## Sample Candidates", ""])
    for row in list(report.get("candidates", []))[:12]:
        lines.append(
            f"- `{row.get('spanType')}` `{row.get('paperId')}` page `{row.get('page')}` "
            f"chars `{row.get('charStart')}-{row.get('charEnd')}` bbox `{row.get('bbox')}`"
        )
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        for blocker in report.get("blockers", []):
            lines.append(f"- `{blocker.get('paperId')}`: `{blocker.get('blockerReason')}`")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


__all__ = [
    "TEXT_SECTION_PARAGRAPH_SPAN_REPORT_SCHEMA_ID",
    "TEXT_SPAN_ARTIFACT_CANDIDATE_SCHEMA_ID",
    "build_text_section_paragraph_span_report",
    "extract_text_span_candidates_from_blocks",
    "extract_text_span_candidates_from_pdf",
    "write_report",
]
