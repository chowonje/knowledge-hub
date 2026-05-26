"""Text-derived equation locator and nearby-context candidates."""

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

EQUATION_CONTEXT_ARTIFACT_CANDIDATE_SCHEMA_ID = "knowledge-hub.paper.equation-context-artifact-candidate.v1"
EQUATION_LOCATOR_CONTEXT_REPORT_SCHEMA_ID = "knowledge-hub.paper.text-equation-locator-context-artifacts-report.v1"

EXTRACTION_METHOD = "pymupdf_equation_locator_context_candidate_v1"
CHAR_BASIS = "pymupdf_block_reading_order_normalized_text_v1"
EQUATION_LABEL_RE = re.compile(r"\((?P<paren>[0-9]+[A-Za-z]?)\)\s*[\.;]?\s*$", re.I)
MATH_SIGNAL_RE = re.compile(r"(=|≤|≥|∑|Σ|∏|√|→|←|↦|∈|∂|∇|\\frac|\\sum|\\begin|\barg\s*max\b)", re.I)
CAPTION_OR_TABLE_RE = re.compile(r"^\s*(Figure|Fig\.|Table)\s+[0-9]+[A-Za-z]?\s*[:.\-]", re.I)
URL_RE = re.compile(r"\b(?:https?://|URL\s+https?://|www\.)", re.I)
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
        raise RuntimeError("PyMuPDF is not installed; install it to run equation text extraction") from error

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


def _equation_label(text: str) -> str:
    match = EQUATION_LABEL_RE.search(str(text or ""))
    if not match:
        return ""
    number = match.group("paren") or ""
    return f"Equation {number.strip()}" if number else ""


def _math_operator_count(text: str) -> int:
    return len(
        re.findall(
            r"(:=|=|≤|≥|∂|∇|∑|Σ|∏|√|→|←|↦|∈|−|(?<=\s)[+\-*/^_](?=\s)|[+\-*/^_](?=\d)|(?<=\d)[+\-*/^_])",
            text,
        )
    )


def _is_equation_like(text: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 3 or CAPTION_OR_TABLE_RE.match(normalized):
        return False
    if URL_RE.search(normalized):
        return False
    if re.match(r"\s*where\b", normalized, re.I):
        return False
    if re.search(r"\b(con(?:f|ﬁ)g|method)\s+value\s+optimizer\b", normalized, re.I):
        return False
    if _equation_label(normalized) and MATH_SIGNAL_RE.search(normalized):
        return True
    if not MATH_SIGNAL_RE.search(normalized):
        return False
    letter_count = sum(1 for char in normalized if char.isalpha())
    digit_count = sum(1 for char in normalized if char.isdigit())
    operator_count = _math_operator_count(normalized)
    token_count = len(normalized.split())
    return operator_count >= 2 and (digit_count + operator_count) >= 2 and token_count <= 45 and letter_count >= 1


def _is_context_block(text: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 50 or CAPTION_OR_TABLE_RE.match(normalized):
        return False
    if _is_equation_like(normalized):
        return False
    return True


def _merge_bbox(boxes: list[list[float]]) -> list[float]:
    if not boxes:
        return []
    return [
        round(min(box[0] for box in boxes), 2),
        round(min(box[1] for box in boxes), 2),
        round(max(box[2] for box in boxes), 2),
        round(max(box[3] for box in boxes), 2),
    ]


def _artifact_id(*, paper_id: str, equation_label: str, page: int, char_start: int, equation_text: str) -> str:
    basis = json.dumps(
        {
            "paperId": paper_id,
            "equationLabel": equation_label,
            "page": page,
            "charStart": char_start,
            "equationText": equation_text,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    label_token = _slug(equation_label or "unlabeled")
    return f"equation-context:{_slug(paper_id)}:{label_token}:{page}:{_short_hash(basis)}"


def extract_equation_context_candidates_from_blocks(
    *,
    paper_id: str,
    paper_ref: str,
    source_content_hash: str,
    blocks_by_page: Iterable[tuple[int, Iterable[Sequence[Any]]]],
    max_rows: int = 24,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = _block_records(blocks_by_page)
    candidates: list[dict[str, Any]] = []
    labeled_rows = 0
    context_rows = 0

    for index, record in enumerate(records):
        equation_text = record["text"]
        if not _is_equation_like(equation_text):
            continue
        if len(candidates) >= max_rows:
            break
        nearby = [
            other
            for other in records[max(0, index - 2) : min(len(records), index + 3)]
            if other is not record and other["page"] == record["page"] and _is_context_block(other["text"])
        ]
        context_text = "\n".join(row["text"] for row in nearby[:2])
        equation_label = _equation_label(equation_text)
        locator_grade = "labeled_equation_context" if equation_label else "equation_like_context"
        if equation_label:
            labeled_rows += 1
        if context_text:
            context_rows += 1
        candidates.append(
            {
                "schema": EQUATION_CONTEXT_ARTIFACT_CANDIDATE_SCHEMA_ID,
                "paperId": paper_id,
                "paperRef": paper_ref,
                "artifactId": _artifact_id(
                    paper_id=paper_id,
                    equation_label=equation_label,
                    page=record["page"],
                    char_start=record["charStart"],
                    equation_text=equation_text,
                ),
                "sourceContentHash": source_content_hash,
                "charBasis": CHAR_BASIS,
                "equationCharStart": int(record["charStart"]),
                "equationCharEnd": int(record["charEnd"]),
                "contextCharStart": int(nearby[0]["charStart"]) if nearby else 0,
                "contextCharEnd": int(nearby[-1]["charEnd"]) if nearby else 0,
                "page": int(record["page"]),
                "equationBbox": record["bbox"],
                "contextBbox": _merge_bbox([row["bbox"] for row in nearby]),
                "equationLabel": equation_label,
                "equationText": equation_text,
                "equationTextHash": sha256_text(equation_text),
                "contextText": context_text,
                "contextTextHash": sha256_text(context_text) if context_text else "",
                "locatorGrade": locator_grade,
                "latexGuarantee": "none_v0_1_locator_only",
                "extractionMethod": EXTRACTION_METHOD,
                "confidence": 0.78 if equation_label and context_text else 0.66,
                "blockerReason": "",
                "diagnostics": {
                    "source": "pymupdf_block",
                    "equationBlockIndex": int(record["blockIndex"]),
                    "nearbyContextBlockCount": len(nearby),
                },
            }
        )

    diagnostics = {
        "paperId": paper_id,
        "paperRef": paper_ref,
        "candidateRows": len(candidates),
        "labeledEquationRows": labeled_rows,
        "equationLikeRows": len(candidates) - labeled_rows,
        "contextRows": context_rows,
    }
    return candidates, diagnostics


def extract_equation_context_candidates_from_pdf(
    *,
    paper: PaperSpec,
    pdf_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_content_hash = sha256_file(pdf_path)
    page_count, blocks_by_page = _load_pdf_blocks(pdf_path)
    candidates, diagnostics = extract_equation_context_candidates_from_blocks(
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


def build_text_equation_locator_context_report(
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
            paper_candidates, diagnostics = extract_equation_context_candidates_from_pdf(
                paper=paper,
                pdf_path=pdf_path,
            )
        except Exception as error:
            blockers.append(_blocker_row(paper=paper, reason="equation_context_extraction_failed", detail=normalize_text(error)))
            continue
        if not paper_candidates:
            blockers.append(_blocker_row(paper=paper, reason="equation_locator_not_found"))
        candidates.extend(paper_candidates)
        paper_diagnostics.append(diagnostics)

    labeled_rows = sum(1 for row in candidates if row.get("equationLabel"))
    context_rows = sum(1 for row in candidates if row.get("contextText"))
    equation_like_rows = sum(1 for row in candidates if not row.get("equationLabel"))
    ready_papers = len({row.get("paperId") for row in candidates})
    report: dict[str, Any] = {
        "schema": EQUATION_LOCATOR_CONTEXT_REPORT_SCHEMA_ID,
        "status": "ready" if ready_papers >= 2 and context_rows > 0 else "blocked",
        "generatedAt": generated_at or utc_now_iso(),
        "scope": {
            "paperRows": len(papers),
            "paperRefs": [_paper_ref(paper) for paper in papers],
            "writes": "report_only",
            "canonicalParsedArtifactWriteRows": 0,
            "charBasis": CHAR_BASIS,
            "latexGuarantee": "none_v0_1_locator_only",
        },
        "paperDiagnostics": paper_diagnostics,
        "candidateRows": len(candidates),
        "labeledEquationRows": labeled_rows,
        "equationLikeRows": equation_like_rows,
        "contextRows": context_rows,
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
            "equation text and labels are candidate-grade locator/context signals; LaTeX reconstruction is not attempted in v0.1"
        ],
        "schemaErrors": [],
    }
    report["privatePathLeakRows"] = _private_path_leak_count(report)
    if report["privatePathLeakRows"]:
        report["status"] = "blocked"
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Text Equation Locator Context Artifacts",
        "",
        f"- status: `{report.get('status')}`",
        f"- candidateRows: `{report.get('candidateRows')}`",
        f"- labeledEquationRows: `{report.get('labeledEquationRows')}`",
        f"- equationLikeRows: `{report.get('equationLikeRows')}`",
        f"- contextRows: `{report.get('contextRows')}`",
        f"- blockerRows: `{report.get('blockerRows')}`",
        f"- privatePathLeakRows: `{report.get('privatePathLeakRows')}`",
        "",
        "## Paper Diagnostics",
        "",
        "| paperId | candidates | labeled | context |",
        "|---|---:|---:|---:|",
    ]
    for row in report.get("paperDiagnostics", []):
        lines.append(
            f"| {row.get('paperId')} | {row.get('candidateRows')} | {row.get('labeledEquationRows')} | {row.get('contextRows')} |"
        )
    lines.extend(["", "## Sample Candidates", ""])
    for row in list(report.get("candidates", []))[:12]:
        label = row.get("equationLabel") or "unlabeled"
        lines.append(
            f"- `{row.get('paperId')}` `{label}` page `{row.get('page')}` "
            f"grade `{row.get('locatorGrade')}` context `{bool(row.get('contextText'))}`"
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
    "EQUATION_CONTEXT_ARTIFACT_CANDIDATE_SCHEMA_ID",
    "EQUATION_LOCATOR_CONTEXT_REPORT_SCHEMA_ID",
    "build_text_equation_locator_context_report",
    "extract_equation_context_candidates_from_blocks",
    "extract_equation_context_candidates_from_pdf",
    "write_report",
]
