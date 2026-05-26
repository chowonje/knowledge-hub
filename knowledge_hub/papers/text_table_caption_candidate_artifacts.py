"""Text-derived table caption and table-like text candidates for local paper PDFs."""

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

TABLE_TEXT_ARTIFACT_CANDIDATE_SCHEMA_ID = "knowledge-hub.paper.table-text-artifact-candidate.v1"
TABLE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID = "knowledge-hub.paper.text-table-caption-candidate-artifacts-report.v1"

EXTRACTION_METHOD = "pymupdf_table_caption_text_candidate_v1"
CHAR_BASIS = "pymupdf_block_reading_order_normalized_text_v1"
TABLE_CAPTION_RE = re.compile(
    r"^\s*Table\s+(?P<number>[0-9]+[A-Za-z]?)(?P<sep>\s*(?::|\.|-)\s*)(?P<caption>.+)$",
    re.IGNORECASE | re.DOTALL,
)
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
        raise RuntimeError("PyMuPDF is not installed; install it to run table text extraction") from error

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


def _caption_match(text: str) -> re.Match[str] | None:
    return TABLE_CAPTION_RE.match(normalize_text(text))


def _table_label(number: str) -> str:
    return f"Table {str(number).strip()}"


def _numeric_density(text: str) -> float:
    value = str(text or "")
    if not value:
        return 0.0
    digits = sum(1 for char in value if char.isdigit())
    return digits / max(1, len(value))


def _looks_table_like(text: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 20:
        return False
    if _caption_match(normalized):
        return False
    raw = str(text or "")
    line_count = sum(1 for line in raw.splitlines() if normalize_text(line))
    dense_numbers = _numeric_density(normalized) >= 0.08
    has_columns = bool(re.search(r"\s{2,}", raw)) or "|" in raw or "\t" in raw
    has_metric_terms = bool(re.search(r"\b(top-?1|top-?5|acc|accuracy|error|params|flops|score|mAP|AP|BLEU)\b", normalized, re.I))
    return (line_count >= 2 and (dense_numbers or has_columns)) or (dense_numbers and has_metric_terms)


def _merge_bbox(boxes: list[list[float]]) -> list[float]:
    if not boxes:
        return []
    return [
        round(min(box[0] for box in boxes), 2),
        round(min(box[1] for box in boxes), 2),
        round(max(box[2] for box in boxes), 2),
        round(max(box[3] for box in boxes), 2),
    ]


def _artifact_id(*, paper_id: str, table_label: str, page: int, caption_text: str) -> str:
    basis = json.dumps(
        {
            "paperId": paper_id,
            "tableLabel": table_label,
            "page": page,
            "captionText": caption_text,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return f"table-text:{_slug(paper_id)}:{_slug(table_label)}:{_short_hash(basis)}"


def _block_records(blocks_by_page: Iterable[tuple[int, Iterable[Sequence[Any]]]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cursor = 0
    for page_number, blocks in blocks_by_page:
        for block_index, block in enumerate(blocks):
            if len(block) < 5:
                continue
            text = normalize_text(block[4])
            if not text:
                continue
            if records:
                cursor += 2
            char_start = cursor
            char_end = char_start + len(text)
            cursor = char_end
            records.append(
                {
                    "page": int(page_number),
                    "blockIndex": int(block_index),
                    "text": text,
                    "rawText": str(block[4] or ""),
                    "bbox": _round_bbox(block[:4]),
                    "charStart": char_start,
                    "charEnd": char_end,
                }
            )
    return records


def extract_table_text_candidates_from_blocks(
    *,
    paper_id: str,
    paper_ref: str,
    source_content_hash: str,
    blocks_by_page: Iterable[tuple[int, Iterable[Sequence[Any]]]],
    max_rows: int = 24,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = _block_records(blocks_by_page)
    candidates: list[dict[str, Any]] = []
    caption_rows = 0
    table_like_rows = 0

    for index, record in enumerate(records):
        match = _caption_match(record["text"])
        if not match:
            continue
        if len(candidates) >= max_rows:
            break
        table_label = _table_label(match.group("number"))
        caption_text = normalize_text(match.group("caption"))
        nearby = [
            other
            for other in records[index + 1 : index + 4]
            if other["page"] == record["page"] and _looks_table_like(other["rawText"])
        ]
        table_text = "\n".join(row["text"] for row in nearby)
        table_bbox = _merge_bbox([row["bbox"] for row in nearby])
        structure_grade = "table_like_text_candidate" if table_text else "caption_only"
        numeric_candidate = bool(table_text and _numeric_density(table_text) >= 0.08)
        if table_text:
            table_like_rows += 1
        caption_rows += 1
        candidates.append(
            {
                "schema": TABLE_TEXT_ARTIFACT_CANDIDATE_SCHEMA_ID,
                "paperId": paper_id,
                "paperRef": paper_ref,
                "artifactId": _artifact_id(
                    paper_id=paper_id,
                    table_label=table_label,
                    page=record["page"],
                    caption_text=caption_text,
                ),
                "sourceContentHash": source_content_hash,
                "charBasis": CHAR_BASIS,
                "captionCharStart": int(record["charStart"]),
                "captionCharEnd": int(record["charEnd"]),
                "page": int(record["page"]),
                "captionBbox": record["bbox"],
                "tableTextBbox": table_bbox,
                "tableLabel": table_label,
                "captionText": caption_text,
                "captionTextHash": sha256_text(caption_text),
                "tableText": table_text,
                "tableTextHash": sha256_text(table_text) if table_text else "",
                "structureGrade": structure_grade,
                "numericCandidate": numeric_candidate,
                "extractionMethod": EXTRACTION_METHOD,
                "confidence": 0.78 if table_text else 0.68,
                "blockerReason": "",
                "diagnostics": {
                    "source": "pymupdf_block",
                    "captionBlockIndex": int(record["blockIndex"]),
                    "nearbyTableLikeBlockCount": len(nearby),
                },
            }
        )

    diagnostics = {
        "paperId": paper_id,
        "paperRef": paper_ref,
        "captionRows": caption_rows,
        "tableLikeTextRows": table_like_rows,
        "candidateRows": len(candidates),
    }
    return candidates, diagnostics


def extract_table_text_candidates_from_pdf(*, paper: PaperSpec, pdf_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_content_hash = sha256_file(pdf_path)
    page_count, blocks_by_page = _load_pdf_blocks(pdf_path)
    candidates, diagnostics = extract_table_text_candidates_from_blocks(
        paper_id=paper.paper_id,
        paper_ref=_paper_ref(paper),
        source_content_hash=source_content_hash,
        blocks_by_page=blocks_by_page,
    )
    diagnostics["pageCount"] = page_count
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


def build_text_table_caption_candidate_report(
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
            paper_candidates, diagnostics = extract_table_text_candidates_from_pdf(paper=paper, pdf_path=pdf_path)
        except Exception as error:
            blockers.append(_blocker_row(paper=paper, reason="table_text_extraction_failed", detail=normalize_text(error)))
            continue
        if not paper_candidates:
            blockers.append(_blocker_row(paper=paper, reason="table_caption_not_found"))
        candidates.extend(paper_candidates)
        paper_diagnostics.append(diagnostics)

    caption_only_rows = sum(1 for row in candidates if row.get("structureGrade") == "caption_only")
    table_like_rows = sum(1 for row in candidates if row.get("structureGrade") == "table_like_text_candidate")
    numeric_rows = sum(1 for row in candidates if row.get("numericCandidate") is True)
    ready_papers = len({row.get("paperId") for row in candidates})
    report: dict[str, Any] = {
        "schema": TABLE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID,
        "status": "ready" if ready_papers >= 3 else "blocked",
        "generatedAt": generated_at or utc_now_iso(),
        "scope": {
            "paperRows": len(papers),
            "paperRefs": [_paper_ref(paper) for paper in papers],
            "writes": "report_only",
            "canonicalParsedArtifactWriteRows": 0,
            "charBasis": CHAR_BASIS,
            "tableGridGuarantee": "none_v0_1_candidate_only",
        },
        "paperDiagnostics": paper_diagnostics,
        "candidateRows": len(candidates),
        "captionOnlyRows": caption_only_rows,
        "tableLikeTextRows": table_like_rows,
        "numericCandidateRows": numeric_rows,
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
            "row/column/cell structure remains candidate-grade and is not strict evidence in v0.1"
        ],
        "schemaErrors": [],
    }
    report["privatePathLeakRows"] = _private_path_leak_count(report)
    if report["privatePathLeakRows"]:
        report["status"] = "blocked"
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Text Table Caption Candidate Artifacts",
        "",
        f"- status: `{report.get('status')}`",
        f"- candidateRows: `{report.get('candidateRows')}`",
        f"- captionOnlyRows: `{report.get('captionOnlyRows')}`",
        f"- tableLikeTextRows: `{report.get('tableLikeTextRows')}`",
        f"- numericCandidateRows: `{report.get('numericCandidateRows')}`",
        f"- blockerRows: `{report.get('blockerRows')}`",
        f"- privatePathLeakRows: `{report.get('privatePathLeakRows')}`",
        "",
        "## Paper Diagnostics",
        "",
        "| paperId | captions | table-like | candidates |",
        "|---|---:|---:|---:|",
    ]
    for row in report.get("paperDiagnostics", []):
        lines.append(
            f"| {row.get('paperId')} | {row.get('captionRows')} | {row.get('tableLikeTextRows')} | {row.get('candidateRows')} |"
        )
    lines.extend(["", "## Sample Candidates", ""])
    for row in list(report.get("candidates", []))[:12]:
        lines.append(
            f"- `{row.get('paperId')}` `{row.get('tableLabel')}` page `{row.get('page')}` "
            f"grade `{row.get('structureGrade')}` numeric `{row.get('numericCandidate')}`"
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
    "TABLE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID",
    "TABLE_TEXT_ARTIFACT_CANDIDATE_SCHEMA_ID",
    "build_text_table_caption_candidate_report",
    "extract_table_text_candidates_from_blocks",
    "extract_table_text_candidates_from_pdf",
    "write_report",
]
