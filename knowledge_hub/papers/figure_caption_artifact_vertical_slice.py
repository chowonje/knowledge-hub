"""Minimal FigureCaptionArtifact vertical slice for local paper PDFs.

This module intentionally reads PDF text blocks and writes only eval/report
artifacts. It does not mutate canonical parsed artifacts, indexes, or runtime
answer-visible evidence stores.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

FIGURE_CAPTION_ARTIFACT_CANDIDATE_SCHEMA_ID = "knowledge-hub.paper.figure-caption-artifact-candidate.v1"
FIGURE_CAPTION_ARTIFACT_VERTICAL_SLICE_REPORT_SCHEMA_ID = (
    "knowledge-hub.paper.figure-caption-artifact-vertical-slice-report.v1"
)

EXTRACTION_METHOD = "pymupdf_text_block_regex_v1"
ANSWER_PACKET_TYPE = "figure_caption_candidate_answer_packet.v1"

CAPTION_RE = re.compile(
    r"^\s*(?P<prefix>Figure|Fig\.)\s+(?P<number>[0-9]+[A-Za-z]?)(?P<sep>\s*(?::|\.|-)?\s*)(?P<caption>.+)$",
    re.IGNORECASE | re.DOTALL,
)
QUESTION_FIGURE_RE = re.compile(r"(?:Figure|Fig\.)\s+(?P<number>[0-9]+[A-Za-z]?)(?![0-9A-Za-z])", re.IGNORECASE)
PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)

DEFAULT_EVAL_PAPERS: tuple[dict[str, str], ...] = (
    {
        "paperId": "alexnet-2012",
        "filename": "4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf",
    },
    {
        "paperId": "resnet-2015",
        "filename": "Deep Residual Learning for Image Recognition.pdf",
    },
    {
        "paperId": "clip-2021",
        "filename": "Learning Transferable Visual Models From Natural Language Supervision.pdf",
    },
    {
        "paperId": "mae-2021",
        "filename": "Masked Autoencoders Are Scalable Vision Learners.pdf",
    },
)


@dataclass(frozen=True, slots=True)
class PaperSpec:
    paper_id: str
    filename: str

    @property
    def paper_ref(self) -> str:
        return f"papers_dir/{self.filename}"


def default_papers_root() -> Path:
    return Path.home() / ".khub" / "papers"


def default_paper_specs() -> list[PaperSpec]:
    return [
        PaperSpec(paper_id=str(row["paperId"]), filename=str(row["filename"]))
        for row in DEFAULT_EVAL_PAPERS
    ]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _short_hash(value: str, *, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _slug(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return token or "unknown"


def _round_bbox(values: Sequence[Any]) -> list[float]:
    rounded: list[float] = []
    for value in values[:4]:
        rounded.append(round(float(value), 2))
    return rounded


def _figure_label(number: str) -> str:
    return f"Figure {str(number).strip()}"


def _caption_match(text: str) -> re.Match[str] | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    match = CAPTION_RE.match(normalized)
    if not match:
        return None
    if match.group("prefix").lower() == "figure" and not re.search(r"[:.\-]", match.group("sep")):
        return None
    return match


def extract_figure_caption_candidates_from_blocks(
    *,
    paper_id: str,
    paper_ref: str,
    source_content_hash: str,
    blocks_by_page: Iterable[tuple[int, Iterable[Sequence[Any]]]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page_number, blocks in blocks_by_page:
        for block_index, block in enumerate(blocks):
            if len(block) < 5:
                continue
            text = str(block[4] or "")
            match = _caption_match(text)
            if not match:
                continue
            caption_text = normalize_text(match.group("caption"))
            if not caption_text:
                continue
            figure_label = _figure_label(match.group("number"))
            bbox = _round_bbox(block[:4])
            identity_basis = json.dumps(
                {
                    "paperId": paper_id,
                    "figureLabel": figure_label,
                    "page": page_number,
                    "bbox": bbox,
                    "captionText": caption_text,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            artifact_id = (
                f"figure-caption:{_slug(paper_id)}:{_slug(figure_label)}:{_short_hash(identity_basis)}"
            )
            dedupe_key = f"{paper_id}:{figure_label}:{page_number}:{bbox}:{caption_text}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            candidates.append(
                {
                    "schema": FIGURE_CAPTION_ARTIFACT_CANDIDATE_SCHEMA_ID,
                    "paperId": paper_id,
                    "paperRef": paper_ref,
                    "artifactId": artifact_id,
                    "sourceContentHash": source_content_hash,
                    "page": int(page_number),
                    "bbox": bbox,
                    "figureLabel": figure_label,
                    "captionText": caption_text,
                    "captionTextHash": sha256_text(caption_text),
                    "extractionMethod": EXTRACTION_METHOD,
                    "confidence": 0.86,
                    "blockerReason": "",
                    "diagnostics": {
                        "source": "pymupdf_block",
                        "blockIndex": int(block_index),
                    },
                }
            )
    return candidates


def _load_pdf_blocks(pdf_path: Path) -> tuple[int, list[tuple[int, list[Sequence[Any]]]]]:
    try:
        import fitz  # type: ignore
    except Exception as error:  # pragma: no cover - covered by environment-level smoke
        raise RuntimeError("PyMuPDF is not installed; install it to run the vertical slice") from error

    try:
        document = fitz.open(str(pdf_path))
    except Exception as error:
        raise RuntimeError(f"pymupdf open failed: {error}") from error

    pages: list[tuple[int, list[Sequence[Any]]]] = []
    try:
        page_count = int(getattr(document, "page_count", 0) or len(document))
        for page_index in range(page_count):
            page = document.load_page(page_index)
            pages.append((page_index + 1, list(page.get_text("blocks") or [])))
    finally:
        try:
            document.close()
        except Exception:
            pass
    return page_count, pages


def extract_figure_caption_candidates_from_pdf(
    *,
    paper: PaperSpec,
    pdf_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_content_hash = sha256_file(pdf_path)
    page_count, blocks_by_page = _load_pdf_blocks(pdf_path)
    candidates = extract_figure_caption_candidates_from_blocks(
        paper_id=paper.paper_id,
        paper_ref=paper.paper_ref,
        source_content_hash=source_content_hash,
        blocks_by_page=blocks_by_page,
    )
    diagnostics = {
        "paperId": paper.paper_id,
        "paperRef": paper.paper_ref,
        "pageCount": page_count,
        "captionCandidateRows": len(candidates),
    }
    return candidates, diagnostics


def _candidate_has_answerable_provenance(candidate: dict[str, Any]) -> bool:
    bbox = candidate.get("bbox")
    return (
        str(candidate.get("sourceContentHash") or "").startswith("sha256:")
        and isinstance(candidate.get("page"), int)
        and int(candidate.get("page") or 0) >= 1
        and isinstance(bbox, list)
        and len(bbox) == 4
        and bool(candidate.get("figureLabel"))
        and bool(candidate.get("captionText"))
        and str(candidate.get("captionTextHash") or "").startswith("sha256:")
        and not str(candidate.get("blockerReason") or "").strip()
    )


def _parse_figure_label_from_question(question: str) -> str:
    match = QUESTION_FIGURE_RE.search(str(question or ""))
    if not match:
        return ""
    return _figure_label(match.group("number"))


def build_figure_caption_qa_readback(
    *,
    report: dict[str, Any],
    paper_id: str,
    question: str,
) -> dict[str, Any]:
    figure_label = _parse_figure_label_from_question(question)
    if not figure_label:
        return {
            "paperId": paper_id,
            "question": question,
            "requestedFigureLabel": "",
            "answerabilityStatus": "no_answer",
            "blockerReason": "figure_label_not_requested",
            "candidateAnswerPacket": None,
        }

    matching = [
        candidate
        for candidate in report.get("candidateArtifacts", [])
        if candidate.get("paperId") == paper_id and str(candidate.get("figureLabel")) == figure_label
    ]
    if not matching:
        return {
            "paperId": paper_id,
            "question": question,
            "requestedFigureLabel": figure_label,
            "answerabilityStatus": "no_answer",
            "blockerReason": "figure_caption_not_found",
            "candidateAnswerPacket": None,
        }

    candidate = matching[0]
    if not _candidate_has_answerable_provenance(candidate):
        return {
            "paperId": paper_id,
            "question": question,
            "requestedFigureLabel": figure_label,
            "answerabilityStatus": "no_answer",
            "blockerReason": str(candidate.get("blockerReason") or "insufficient_provenance"),
            "candidateAnswerPacket": None,
        }

    packet = {
        "packetType": ANSWER_PACKET_TYPE,
        "answerVisible": False,
        "strictEvidence": False,
        "answer": f"{figure_label} shows: {candidate['captionText']}",
        "evidence": {
            "artifactId": candidate["artifactId"],
            "paperId": candidate["paperId"],
            "sourceContentHash": candidate["sourceContentHash"],
            "page": candidate["page"],
            "bbox": candidate["bbox"],
            "figureLabel": candidate["figureLabel"],
            "captionTextHash": candidate["captionTextHash"],
            "extractionMethod": candidate["extractionMethod"],
        },
    }
    return {
        "paperId": paper_id,
        "question": question,
        "requestedFigureLabel": figure_label,
        "answerabilityStatus": "answerable",
        "blockerReason": "",
        "candidateAnswerPacket": packet,
    }


def _blocker_row(*, paper: PaperSpec, reason: str, detail: str = "") -> dict[str, Any]:
    return {
        "paperId": paper.paper_id,
        "paperRef": paper.paper_ref,
        "blockerReason": reason,
        "detail": detail,
    }


def _private_path_leak_count(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return 1 if PRIVATE_PATH_RE.search(encoded) else 0


def build_vertical_slice_report(
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
            blockers.append(_blocker_row(paper=paper, reason="source_pdf_missing", detail=paper.paper_ref))
            paper_diagnostics.append(
                {
                    "paperId": paper.paper_id,
                    "paperRef": paper.paper_ref,
                    "pageCount": 0,
                    "captionCandidateRows": 0,
                }
            )
            continue
        try:
            paper_candidates, diagnostics = extract_figure_caption_candidates_from_pdf(
                paper=paper,
                pdf_path=pdf_path,
            )
        except Exception as error:
            blockers.append(
                _blocker_row(
                    paper=paper,
                    reason="caption_extraction_failed",
                    detail=normalize_text(str(error)),
                )
            )
            continue
        if not paper_candidates:
            blockers.append(_blocker_row(paper=paper, reason="figure_caption_not_found"))
        candidates.extend(paper_candidates)
        paper_diagnostics.append(diagnostics)

    report: dict[str, Any] = {
        "schema": FIGURE_CAPTION_ARTIFACT_VERTICAL_SLICE_REPORT_SCHEMA_ID,
        "status": "ready" if len({row["paperId"] for row in candidates}) >= 3 else "blocked",
        "generatedAt": generated_at or utc_now_iso(),
        "scope": {
            "paperRows": len(papers),
            "paperRefs": [paper.paper_ref for paper in papers],
            "writes": "report_only",
            "canonicalParsedArtifactWriteRows": 0,
        },
        "paperDiagnostics": paper_diagnostics,
        "candidateArtifactRows": len(candidates),
        "candidateArtifacts": candidates,
        "blockerRows": len(blockers),
        "blockers": blockers,
        "qaReadbackRows": 0,
        "answerableQaRows": 0,
        "noAnswerQaRows": 0,
        "qaReadback": [],
        "mutationCounters": {
            "canonicalParsedArtifactWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "answerVisiblePromotionRows": 0,
            "strictEvidencePromotionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
            "answerabilityGateBypassRows": 0,
        },
        "schemaViolationCount": 0,
        "privatePathLeakRows": 0,
        "warnings": [],
        "schemaErrors": [],
    }

    qa_rows: list[dict[str, Any]] = []
    if candidates:
        first_candidate = candidates[0]
        qa_rows.append(
            build_figure_caption_qa_readback(
                report=report,
                paper_id=str(first_candidate["paperId"]),
                question=f"이 논문의 {first_candidate['figureLabel']}은 무엇을 보여주는가?",
            )
        )
        qa_rows.append(
            build_figure_caption_qa_readback(
                report=report,
                paper_id=str(first_candidate["paperId"]),
                question="이 논문의 Figure 99은 무엇을 보여주는가?",
            )
        )
    elif papers:
        qa_rows.append(
            build_figure_caption_qa_readback(
                report=report,
                paper_id=papers[0].paper_id,
                question="이 논문의 Figure 1은 무엇을 보여주는가?",
            )
        )

    report["qaReadback"] = qa_rows
    report["qaReadbackRows"] = len(qa_rows)
    report["answerableQaRows"] = sum(1 for row in qa_rows if row.get("answerabilityStatus") == "answerable")
    report["noAnswerQaRows"] = sum(1 for row in qa_rows if row.get("answerabilityStatus") == "no_answer")
    report["privatePathLeakRows"] = _private_path_leak_count(report)
    if report["privatePathLeakRows"]:
        report["status"] = "blocked"
        report["warnings"].append("private_path_leak_detected")
    return report


def write_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# FigureCaptionArtifact Vertical Slice",
        "",
        f"- status: `{report.get('status')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- candidateArtifactRows: `{report.get('candidateArtifactRows')}`",
        f"- blockerRows: `{report.get('blockerRows')}`",
        f"- qaReadbackRows: `{report.get('qaReadbackRows')}`",
        f"- answerableQaRows: `{report.get('answerableQaRows')}`",
        f"- noAnswerQaRows: `{report.get('noAnswerQaRows')}`",
        f"- privatePathLeakRows: `{report.get('privatePathLeakRows')}`",
        "",
        "## Candidate Artifacts",
        "",
        "| paperId | figureLabel | page | bbox | confidence |",
        "|---|---:|---:|---|---:|",
    ]
    for candidate in report.get("candidateArtifacts", []):
        lines.append(
            "| {paperId} | {figureLabel} | {page} | `{bbox}` | {confidence} |".format(
                paperId=candidate.get("paperId"),
                figureLabel=candidate.get("figureLabel"),
                page=candidate.get("page"),
                bbox=candidate.get("bbox"),
                confidence=candidate.get("confidence"),
            )
        )
    lines.extend(["", "## QA Readback", ""])
    for row in report.get("qaReadback", []):
        lines.extend(
            [
                f"- paperId: `{row.get('paperId')}`",
                f"  - question: {row.get('question')}",
                f"  - answerabilityStatus: `{row.get('answerabilityStatus')}`",
                f"  - blockerReason: `{row.get('blockerReason')}`",
            ]
        )
        packet = row.get("candidateAnswerPacket")
        if isinstance(packet, dict):
            evidence = packet.get("evidence") or {}
            lines.append(
                "  - evidence: "
                f"`page={evidence.get('page')}`, "
                f"`bbox={evidence.get('bbox')}`, "
                f"`sourceContentHash={evidence.get('sourceContentHash')}`"
            )
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        for blocker in report.get("blockers", []):
            lines.append(
                f"- `{blocker.get('paperId')}`: `{blocker.get('blockerReason')}` {blocker.get('detail') or ''}".rstrip()
            )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "FIGURE_CAPTION_ARTIFACT_CANDIDATE_SCHEMA_ID",
    "FIGURE_CAPTION_ARTIFACT_VERTICAL_SLICE_REPORT_SCHEMA_ID",
    "PaperSpec",
    "build_figure_caption_qa_readback",
    "build_vertical_slice_report",
    "default_paper_specs",
    "default_papers_root",
    "extract_figure_caption_candidates_from_blocks",
    "extract_figure_caption_candidates_from_pdf",
    "render_markdown_report",
    "write_report",
]
