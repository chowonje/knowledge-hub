"""Report-only visual/layout candidate list for local paper PDFs.

This helper prepares retrieval-hint candidates for a future GPT/VLM request
pack. It reads only already-local PDFs and optional parsed artifacts, writes
only reports through the caller, and never promotes derived visual text to
strict, citation-grade, runtime, or answer-visible evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


VISUAL_LAYOUT_CANDIDATE_ROW_SCHEMA_ID = "knowledge-hub.paper.visual-layout-candidate-row.v1"
VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID = (
    "knowledge-hub.paper.visual-layout-candidate-list-report.v1"
)

NEXT_RECOMMENDED_TRANCHE = "visual_annotation_request_pack_design"
READY_DECISION = "ready_for_visual_annotation_request_pack_design"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)

CAPTION_RE = re.compile(
    r"^\s*(?P<label>(?:Figure|Fig\.?|Table)\s+[0-9]+[A-Za-z]?)\s*(?::|\.|-)\s*(?P<caption>.+)$",
    re.IGNORECASE | re.DOTALL,
)
EQUATION_RE = re.compile(r"(?:[A-Za-z]\s*[=<>]\s*[-+*/(). A-Za-z0-9]+|\\(?:sum|frac|alpha|beta|theta|lambda)|[∑∏√≤≥≈])")
TABLE_SEP_RE = re.compile(r"(?:\s{2,}|\t|\|)")

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

KNOWN_PAPER_IDS_BY_FILENAME: dict[str, str] = {
    str(row["filename"]): str(row["paperId"]) for row in DEFAULT_EVAL_PAPERS
}


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


def paper_id_from_filename(filename: str, *, used_paper_ids: set[str] | None = None) -> str:
    known = KNOWN_PAPER_IDS_BY_FILENAME.get(str(filename))
    if known:
        return known
    stem = Path(str(filename)).stem
    base = _slug(stem)[:96].strip("-") or "paper"
    used = used_paper_ids if used_paper_ids is not None else set()
    if base not in used:
        return base
    digest = _short_hash(str(filename), length=8)
    candidate = f"{base[:87].strip('-')}-{digest}".strip("-")
    counter = 2
    while candidate in used:
        suffix = f"{digest}-{counter}"
        candidate = f"{base[: max(1, 96 - len(suffix) - 1)].strip('-')}-{suffix}".strip("-")
        counter += 1
    return candidate


def discover_local_paper_specs(
    papers_root: Path | None = None,
    *,
    limit: int | None = None,
) -> list[PaperSpec]:
    root = papers_root or default_papers_root()
    pdf_paths = sorted(
        [path for path in root.expanduser().glob("*.pdf") if path.is_file()],
        key=lambda path: path.name.casefold(),
    )
    if limit is not None:
        pdf_paths = pdf_paths[: max(0, int(limit))]
    used: set[str] = set()
    specs: list[PaperSpec] = []
    for path in pdf_paths:
        paper_id = paper_id_from_filename(path.name, used_paper_ids=used)
        used.add(paper_id)
        specs.append(PaperSpec(paper_id=paper_id, filename=path.name))
    return specs


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _short_hash(value: str, *, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _slug(value: str) -> str:
    token = re.sub(r"[^a-z0-9_.-]+", "-", str(value or "").lower()).strip("-")
    return token or "unknown"


def _round_bbox(value: Any) -> list[float]:
    if isinstance(value, dict):
        raw = [
            value.get("x0", value.get("left")),
            value.get("y0", value.get("top")),
            value.get("x1", value.get("right")),
            value.get("y1", value.get("bottom")),
        ]
    elif isinstance(value, (list, tuple)):
        raw = list(value[:4])
    else:
        return []
    out: list[float] = []
    for item in raw:
        try:
            out.append(round(float(item), 2))
        except Exception:
            return []
    return out if len(out) == 4 else []


def _clean_heading_path(value: Any, *, page: int) -> list[str]:
    if isinstance(value, (list, tuple)):
        headings = [normalize_text(item) for item in value if normalize_text(item)]
    else:
        token = normalize_text(value)
        headings = [part.strip() for part in token.split(">") if part.strip()] if token else []
    return headings or ([f"Page {page}"] if page > 0 else [])


def _bounded_text(value: Any, *, limit: int = 420) -> str:
    text = normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _candidate_id(
    *,
    paper_id: str,
    candidate_type: str,
    page: int,
    bbox: Sequence[float],
    basis: str,
) -> str:
    digest = _short_hash(
        json.dumps(
            {
                "paperId": paper_id,
                "candidateType": candidate_type,
                "page": page,
                "bbox": list(bbox),
                "basis": basis,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        length=16,
    )
    return f"visual-layout:{_slug(paper_id)}:{candidate_type}:{page}:{digest}"


def _planned_crop_ref(*, paper_id: str, candidate_id: str, page: int) -> str:
    return f"papers_dir/visual_layout_planned_crops/{_slug(paper_id)}/page-{page}/{_slug(candidate_id)}.png"


def _retrieval_hint_plan() -> dict[str, Any]:
    return {
        "targetDerivedTextField": "derivedTextForRetrieval",
        "allowedUse": "retrieval_hint_only",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
    }


def _candidate_row(
    *,
    paper_id: str,
    paper_ref: str,
    source_content_hash: str,
    page: int,
    bbox: Sequence[float],
    candidate_type: str,
    nearby_text: str = "",
    caption_text: str = "",
    heading_path: Sequence[str] | None = None,
    image_hash: str = "",
    page_image_required: bool = True,
    extraction_method: str,
    blocker_reason: str = "",
) -> dict[str, Any]:
    rounded_bbox = _round_bbox(list(bbox))
    basis = "|".join(
        [
            paper_id,
            source_content_hash,
            str(page),
            json.dumps(rounded_bbox, ensure_ascii=True),
            candidate_type,
            caption_text,
            nearby_text[:160],
            image_hash,
        ]
    )
    candidate_id = _candidate_id(
        paper_id=paper_id,
        candidate_type=candidate_type,
        page=page,
        bbox=rounded_bbox,
        basis=basis,
    )
    return {
        "schema": VISUAL_LAYOUT_CANDIDATE_ROW_SCHEMA_ID,
        "candidateId": candidate_id,
        "paperId": paper_id,
        "paperRef": paper_ref,
        "sourceContentHash": source_content_hash,
        "page": int(page),
        "bbox": rounded_bbox,
        "candidateType": candidate_type,
        "textContext": {
            "nearbyText": _bounded_text(nearby_text),
            "captionText": _bounded_text(caption_text),
            "headingPath": [normalize_text(item) for item in list(heading_path or []) if normalize_text(item)],
        },
        "visualContext": {
            "cropRef": _planned_crop_ref(paper_id=paper_id, candidate_id=candidate_id, page=max(0, int(page))),
            "imageHash": image_hash,
            "pageImageRequired": bool(page_image_required),
        },
        "retrievalHintPlan": _retrieval_hint_plan(),
        "provenance": {
            "sourceContentHash": source_content_hash,
            "page": int(page),
            "bbox": rounded_bbox,
            "extractionMethod": extraction_method,
        },
        "blockerReason": normalize_text(blocker_reason),
    }


def _caption_match(text: str) -> re.Match[str] | None:
    return CAPTION_RE.match(normalize_text(text))


def _looks_table_like(text: str) -> bool:
    body = str(text or "")
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    normalized = normalize_text(body)
    if len(normalized) < 24:
        return False
    numeric_tokens = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", normalized))
    separator_lines = sum(1 for line in lines if TABLE_SEP_RE.search(line))
    return (len(lines) >= 3 and separator_lines >= 2 and numeric_tokens >= 3) or (
        numeric_tokens >= 8 and separator_lines >= 1
    )


def _looks_equation_like(text: str) -> bool:
    body = normalize_text(text)
    if not body or len(body) > 260:
        return False
    if body.lower().startswith(("figure ", "fig.", "table ")):
        return False
    return bool(EQUATION_RE.search(body) and re.search(r"[=<>∑∏√≤≥≈]", body))


def _looks_layout_heading(text: str) -> bool:
    body = normalize_text(text)
    if not body or len(body) > 120:
        return False
    lowered = body.casefold()
    if lowered.startswith(("abstract", "introduction", "related work", "method", "methods", "experiments", "results", "discussion", "conclusion", "appendix")):
        return True
    return bool(re.match(r"^(?:[0-9]+(?:\.[0-9]+)*)\s+[A-Z][A-Za-z0-9,;:() -]{2,}$", body))


def _nearby_text_for_block(blocks: Sequence[Sequence[Any]], block_index: int) -> str:
    parts: list[str] = []
    for index in (block_index - 1, block_index, block_index + 1):
        if index < 0 or index >= len(blocks):
            continue
        block = blocks[index]
        if len(block) >= 5:
            parts.append(str(block[4] or ""))
    return _bounded_text(" ".join(parts), limit=520)


def extract_candidates_from_blocks(
    *,
    paper_id: str,
    paper_ref: str,
    source_content_hash: str,
    blocks_by_page: Iterable[tuple[int, Sequence[Sequence[Any]]]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for page_number, raw_blocks in blocks_by_page:
        blocks = list(raw_blocks)
        for block_index, block in enumerate(blocks):
            if len(block) < 5:
                continue
            bbox = _round_bbox(block[:4])
            if not bbox:
                continue
            text = normalize_text(block[4])
            if not text:
                continue
            nearby = _nearby_text_for_block(blocks, block_index)
            caption_match = _caption_match(text)
            if caption_match:
                label = normalize_text(caption_match.group("label"))
                candidate_type = "table_region" if label.lower().startswith("table") else "figure_caption_region"
                candidates.append(
                    _candidate_row(
                        paper_id=paper_id,
                        paper_ref=paper_ref,
                        source_content_hash=source_content_hash,
                        page=page_number,
                        bbox=bbox,
                        candidate_type=candidate_type,
                        nearby_text=nearby,
                        caption_text=caption_match.group("caption"),
                        heading_path=[f"Page {page_number}"],
                        extraction_method="pymupdf_text_block_caption_regex_v1",
                    )
                )
                continue
            if _looks_table_like(str(block[4] or "")):
                candidates.append(
                    _candidate_row(
                        paper_id=paper_id,
                        paper_ref=paper_ref,
                        source_content_hash=source_content_hash,
                        page=page_number,
                        bbox=bbox,
                        candidate_type="table_region",
                        nearby_text=nearby,
                        heading_path=[f"Page {page_number}"],
                        extraction_method="pymupdf_text_block_table_density_v1",
                    )
                )
                continue
            if _looks_equation_like(text):
                candidates.append(
                    _candidate_row(
                        paper_id=paper_id,
                        paper_ref=paper_ref,
                        source_content_hash=source_content_hash,
                        page=page_number,
                        bbox=bbox,
                        candidate_type="equation_region",
                        nearby_text=nearby,
                        heading_path=[f"Page {page_number}"],
                        extraction_method="pymupdf_text_block_equation_heuristic_v1",
                    )
                )
                continue
            if _looks_layout_heading(text):
                candidates.append(
                    _candidate_row(
                        paper_id=paper_id,
                        paper_ref=paper_ref,
                        source_content_hash=source_content_hash,
                        page=page_number,
                        bbox=bbox,
                        candidate_type="layout_region",
                        nearby_text=nearby,
                        heading_path=[text],
                        extraction_method="pymupdf_text_block_layout_heading_v1",
                    )
                )
    return candidates


def extract_candidates_from_images(
    *,
    paper_id: str,
    paper_ref: str,
    source_content_hash: str,
    image_refs_by_page: Iterable[tuple[int, Sequence[dict[str, Any]]]],
    blocks_by_page: dict[int, Sequence[Sequence[Any]]] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    blocks_lookup = dict(blocks_by_page or {})
    for page_number, image_refs in image_refs_by_page:
        nearby = ""
        page_blocks = list(blocks_lookup.get(page_number) or [])
        if page_blocks:
            nearby = _nearby_text_for_block(page_blocks, 0)
        for index, image_ref in enumerate(image_refs):
            bbox = _round_bbox(image_ref.get("bbox"))
            if not bbox:
                continue
            image_hash = normalize_text(image_ref.get("imageHash"))
            method = normalize_text(image_ref.get("extractionMethod")) or "pymupdf_image_xref_v1"
            candidates.append(
                _candidate_row(
                    paper_id=paper_id,
                    paper_ref=paper_ref,
                    source_content_hash=source_content_hash,
                    page=page_number,
                    bbox=bbox,
                    candidate_type="image_region",
                    nearby_text=nearby,
                    caption_text=normalize_text(image_ref.get("captionText")),
                    heading_path=[f"Page {page_number}"],
                    image_hash=image_hash,
                    page_image_required=True,
                    extraction_method=method,
                    blocker_reason="" if image_hash else normalize_text(image_ref.get("blockerReason")),
                )
            )
            candidates[-1]["provenance"]["imageIndex"] = int(index)
    return candidates


def _candidate_type_from_element_type(element_type: str, text: str) -> str:
    token = element_type.casefold()
    if "table" in token:
        return "table_region"
    if "equation" in token or "formula" in token:
        return "equation_region"
    if "image" in token or "picture" in token:
        return "image_region"
    if "figure" in token or _caption_match(text):
        return "figure_caption_region"
    if "heading" in token or "title" in token or "section" in token:
        return "layout_region"
    return ""


def extract_candidates_from_parsed_elements(
    *,
    paper_id: str,
    paper_ref: str,
    source_content_hash: str,
    elements: Iterable[dict[str, Any]],
    extraction_method: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in elements:
        if not isinstance(item, dict):
            continue
        text = normalize_text(item.get("text"))
        element_type = normalize_text(item.get("type") or item.get("elementType"))
        candidate_type = _candidate_type_from_element_type(element_type, text)
        if not candidate_type:
            continue
        try:
            page = int(item.get("page") or 0)
        except Exception:
            page = 0
        bbox = _round_bbox(item.get("bbox"))
        if page <= 0 or not bbox:
            continue
        caption = ""
        match = _caption_match(text)
        if match:
            caption = match.group("caption")
        candidates.append(
            _candidate_row(
                paper_id=paper_id,
                paper_ref=paper_ref,
                source_content_hash=source_content_hash,
                page=page,
                bbox=bbox,
                candidate_type=candidate_type,
                nearby_text=text,
                caption_text=caption,
                heading_path=_clean_heading_path(item.get("heading_path") or item.get("headingPath"), page=page),
                image_hash="",
                page_image_required=True,
                extraction_method=extraction_method,
            )
        )
    return candidates


def _dedupe_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, int, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in candidates:
        bbox_key = json.dumps(row.get("bbox") or [], ensure_ascii=True)
        key = (
            str(row.get("paperId") or ""),
            str(row.get("candidateType") or ""),
            int(row.get("page") or 0),
            bbox_key,
            str((row.get("textContext") or {}).get("captionText") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _load_parsed_elements(papers_root: Path, paper_id: str) -> tuple[list[dict[str, Any]], str]:
    document_path = papers_root / "parsed" / paper_id / "document.json"
    if not document_path.is_file():
        return [], ""
    try:
        payload = json.loads(document_path.read_text(encoding="utf-8"))
    except Exception:
        return [], ""
    parser_meta = payload.get("parser_meta") if isinstance(payload, dict) else {}
    parser = normalize_text((parser_meta or {}).get("parser"))
    elements = payload.get("elements") if isinstance(payload, dict) else []
    if not isinstance(elements, list):
        return [], ""
    method = f"parsed_artifact_{_slug(parser) or 'unknown'}_element_metadata_v1"
    return [item for item in elements if isinstance(item, dict)], method


def _load_pdf_candidates(*, paper: PaperSpec, pdf_path: Path) -> tuple[str, int, list[dict[str, Any]], list[str]]:
    try:
        import fitz  # type: ignore
    except Exception as error:  # pragma: no cover - environment-level failure path
        raise RuntimeError("PyMuPDF is not installed; install it to run the visual/layout report") from error

    source_content_hash = sha256_file(pdf_path)
    warnings: list[str] = []
    blocks_by_page: list[tuple[int, list[Sequence[Any]]]] = []
    image_refs_by_page: list[tuple[int, list[dict[str, Any]]]] = []
    document = fitz.open(str(pdf_path))
    try:
        page_count = int(getattr(document, "page_count", 0) or len(document))
        for page_index in range(page_count):
            page = document.load_page(page_index)
            page_number = page_index + 1
            blocks = list(page.get_text("blocks") or [])
            blocks_by_page.append((page_number, blocks))
            image_refs: list[dict[str, Any]] = []
            for image_tuple in list(page.get_images(full=True) or []):
                if not image_tuple:
                    continue
                try:
                    xref = int(image_tuple[0])
                    rects = list(page.get_image_rects(xref) or [])
                except Exception:
                    continue
                image_hash = ""
                try:
                    extracted = document.extract_image(xref) or {}
                    image_bytes = extracted.get("image")
                    if isinstance(image_bytes, bytes):
                        image_hash = sha256_bytes(image_bytes)
                except Exception:
                    warnings.append(f"{paper.paper_id}:image_hash_unavailable")
                for rect in rects:
                    image_refs.append(
                        {
                            "bbox": [
                                getattr(rect, "x0", 0.0),
                                getattr(rect, "y0", 0.0),
                                getattr(rect, "x1", 0.0),
                                getattr(rect, "y1", 0.0),
                            ],
                            "imageHash": image_hash,
                            "blockerReason": "" if image_hash else "image_hash_unavailable",
                            "extractionMethod": "pymupdf_image_xref_v1",
                        }
                    )
            image_refs_by_page.append((page_number, image_refs))
    finally:
        try:
            document.close()
        except Exception:
            pass

    block_candidates = extract_candidates_from_blocks(
        paper_id=paper.paper_id,
        paper_ref=paper.paper_ref,
        source_content_hash=source_content_hash,
        blocks_by_page=blocks_by_page,
    )
    image_candidates = extract_candidates_from_images(
        paper_id=paper.paper_id,
        paper_ref=paper.paper_ref,
        source_content_hash=source_content_hash,
        image_refs_by_page=image_refs_by_page,
        blocks_by_page=dict(blocks_by_page),
    )
    return source_content_hash, page_count, [*block_candidates, *image_candidates], warnings


def _blocked_candidate_row(*, paper: PaperSpec, reason: str, detail: str = "") -> dict[str, Any]:
    row = _candidate_row(
        paper_id=paper.paper_id,
        paper_ref=paper.paper_ref,
        source_content_hash="",
        page=0,
        bbox=[],
        candidate_type="layout_region",
        nearby_text="",
        caption_text="",
        heading_path=[],
        image_hash="",
        page_image_required=False,
        extraction_method="blocked_input_v1",
        blocker_reason=reason,
    )
    row["provenance"]["detail"] = normalize_text(detail)
    return row


def _private_path_leak_count(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return 1 if PRIVATE_PATH_RE.search(encoded) else 0


def _counts(*, input_paper_rows: int, candidate_rows: list[dict[str, Any]], private_path_leak_rows: int) -> dict[str, int]:
    type_counts = {
        "figureCandidateRows": 0,
        "tableCandidateRows": 0,
        "equationCandidateRows": 0,
        "layoutCandidateRows": 0,
        "imageCandidateRows": 0,
    }
    for row in candidate_rows:
        if row.get("blockerReason"):
            continue
        candidate_type = row.get("candidateType")
        if candidate_type == "figure_caption_region":
            type_counts["figureCandidateRows"] += 1
        elif candidate_type == "table_region":
            type_counts["tableCandidateRows"] += 1
        elif candidate_type == "equation_region":
            type_counts["equationCandidateRows"] += 1
        elif candidate_type == "layout_region":
            type_counts["layoutCandidateRows"] += 1
        elif candidate_type == "image_region":
            type_counts["imageCandidateRows"] += 1
    return {
        "inputPaperRows": input_paper_rows,
        "candidateRows": len(candidate_rows),
        **type_counts,
        "blockedRows": sum(1 for row in candidate_rows if row.get("blockerReason")),
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": 0,
    }


def build_report_from_candidate_rows(
    *,
    input_paper_rows: int,
    candidate_rows: Sequence[dict[str, Any]],
    generated_at: str | None = None,
    warnings: Sequence[str] | None = None,
) -> dict[str, Any]:
    rows = _dedupe_candidates(candidate_rows)
    report: dict[str, Any] = {
        "schema": VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "scope": {
            "writes": "report_only",
            "modelCalls": False,
            "vectorIndexing": False,
            "strictEvidencePromotionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "answerabilityGateBypassRows": 0,
            "cropWriteRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
        },
        "counts": {},
        "candidateRowsDetail": rows,
        "warnings": list(dict.fromkeys(normalize_text(item) for item in list(warnings or []) if normalize_text(item))),
    }
    private_path_leak_rows = _private_path_leak_count(report)
    report["counts"] = _counts(
        input_paper_rows=input_paper_rows,
        candidate_rows=rows,
        private_path_leak_rows=private_path_leak_rows,
    )
    if not rows or private_path_leak_rows:
        report["status"] = "blocked"
        report["decision"] = "blocked"
    return report


def build_visual_layout_candidate_list_report(
    *,
    papers_root: Path | None = None,
    paper_specs: Sequence[PaperSpec] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    root = papers_root or default_papers_root()
    papers = list(paper_specs or default_paper_specs())
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []

    for paper in papers:
        pdf_path = root / paper.filename
        if not pdf_path.is_file():
            candidates.append(
                _blocked_candidate_row(
                    paper=paper,
                    reason="source_pdf_missing",
                    detail=paper.paper_ref,
                )
            )
            continue
        try:
            source_hash, _page_count, pdf_candidates, paper_warnings = _load_pdf_candidates(
                paper=paper,
                pdf_path=pdf_path,
            )
            candidates.extend(pdf_candidates)
            warnings.extend(paper_warnings)
            parsed_elements, parsed_method = _load_parsed_elements(root, paper.paper_id)
            if parsed_elements and parsed_method:
                candidates.extend(
                    extract_candidates_from_parsed_elements(
                        paper_id=paper.paper_id,
                        paper_ref=paper.paper_ref,
                        source_content_hash=source_hash,
                        elements=parsed_elements,
                        extraction_method=parsed_method,
                    )
                )
        except Exception as error:
            candidates.append(
                _blocked_candidate_row(
                    paper=paper,
                    reason="candidate_extraction_failed",
                    detail=normalize_text(str(error)),
                )
            )

    return build_report_from_candidate_rows(
        input_paper_rows=len(papers),
        candidate_rows=candidates,
        generated_at=generated_at,
        warnings=warnings,
    )


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    lines = [
        "# Visual Layout Candidate List Report",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- inputPaperRows: `{counts.get('inputPaperRows')}`",
        f"- candidateRows: `{counts.get('candidateRows')}`",
        f"- figureCandidateRows: `{counts.get('figureCandidateRows')}`",
        f"- tableCandidateRows: `{counts.get('tableCandidateRows')}`",
        f"- equationCandidateRows: `{counts.get('equationCandidateRows')}`",
        f"- layoutCandidateRows: `{counts.get('layoutCandidateRows')}`",
        f"- imageCandidateRows: `{counts.get('imageCandidateRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- modelCalls: `{scope.get('modelCalls')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        f"- databaseMutationRows: `{scope.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{scope.get('indexMutationRows')}`",
        f"- reindexOrReembedRows: `{scope.get('reindexOrReembedRows')}`",
        f"- vaultScanRows: `{scope.get('vaultScanRows')}`",
        f"- externalDownloadRows: `{scope.get('externalDownloadRows')}`",
        f"- answerabilityGateBypassRows: `{scope.get('answerabilityGateBypassRows')}`",
        f"- cropWriteRows: `{scope.get('cropWriteRows')}`",
        f"- canonicalParsedArtifactWriteRows: `{scope.get('canonicalParsedArtifactWriteRows')}`",
        "",
        "## Candidate Rows",
        "",
        "| candidateType | paperId | page | bbox | cropRef | blockerReason |",
        "|---|---|---:|---|---|---|",
    ]
    for row in report.get("candidateRowsDetail", []):
        visual = row.get("visualContext") if isinstance(row.get("visualContext"), dict) else {}
        lines.append(
            "| {candidateType} | {paperId} | {page} | `{bbox}` | `{cropRef}` | `{blockerReason}` |".format(
                candidateType=row.get("candidateType"),
                paperId=row.get("paperId"),
                page=row.get("page"),
                bbox=row.get("bbox"),
                cropRef=visual.get("cropRef"),
                blockerReason=row.get("blockerReason"),
            )
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_visual_layout_candidate_list_reports(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown_report(report), encoding="utf-8")
    return {
        "json": str(report_json),
        "markdown": str(report_md),
    }


__all__ = [
    "NEXT_RECOMMENDED_TRANCHE",
    "READY_DECISION",
    "VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID",
    "VISUAL_LAYOUT_CANDIDATE_ROW_SCHEMA_ID",
    "PaperSpec",
    "build_report_from_candidate_rows",
    "build_visual_layout_candidate_list_report",
    "default_paper_specs",
    "default_papers_root",
    "discover_local_paper_specs",
    "extract_candidates_from_blocks",
    "extract_candidates_from_images",
    "extract_candidates_from_parsed_elements",
    "paper_id_from_filename",
    "render_markdown_report",
    "write_visual_layout_candidate_list_reports",
]
