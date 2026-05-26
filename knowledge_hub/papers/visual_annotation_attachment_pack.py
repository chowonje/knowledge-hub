"""Render local PDF crops for a manual visual annotation web pack.

This helper consumes the small visual annotation web pack and writes only
sanitized eval attachment artifacts. It reads already-local PDFs, writes crop
PNGs plus reports, and never calls models or mutates DB/index/evidence state.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.visual_annotation_web_pack import (
    VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID,
    VISUAL_ANNOTATION_WEB_PACK_ROW_SCHEMA_ID,
)
from knowledge_hub.papers.visual_layout_candidate_list_report import default_papers_root


VISUAL_ANNOTATION_ATTACHMENT_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-attachment-pack.v1"
)
VISUAL_ANNOTATION_ATTACHMENT_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-attachment-row.v1"
)

DEFAULT_ATTACHMENT_PACK_ID = "visual_annotation_attachment_pack_001"
READY_DECISION = "ready_for_manual_web_vlm_run"
NEXT_RECOMMENDED_TRANCHE = "visual_annotation_web_pack_manual_calibration"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _slug(value: str) -> str:
    token = re.sub(r"[^a-z0-9_.-]+", "-", str(value or "").lower()).strip("-")
    return token or "unknown"


def _short_hash(value: str, *, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sanitized_report_ref(path: Path, *, project_root: Path | None = None) -> str:
    resolved = path.expanduser()
    if project_root is not None:
        try:
            rel = resolved.resolve().relative_to(project_root.resolve())
            return rel.as_posix()
        except Exception:
            pass
    return f"generated_reports/{resolved.name}"


def load_web_pack(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _round_bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    out: list[float] = []
    for item in list(value[:4]):
        try:
            out.append(round(float(item), 2))
        except Exception:
            return []
    return out


def _bbox_area(bbox: Sequence[float]) -> float:
    if len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _expanded_bbox(bbox: Sequence[float], candidate_type: str, page_rect: Any) -> list[float]:
    try:
        page_x0 = float(page_rect.x0)
        page_y0 = float(page_rect.y0)
        page_x1 = float(page_rect.x1)
        page_y1 = float(page_rect.y1)
    except Exception:
        return list(bbox)
    x0, y0, x1, y1 = [float(item) for item in bbox[:4]]
    if candidate_type == "figure_caption_region":
        margins = (42.0, 260.0, 42.0, 80.0)
    elif candidate_type == "table_region":
        margins = (52.0, 100.0, 52.0, 240.0)
    elif candidate_type == "equation_region":
        margins = (70.0, 90.0, 70.0, 120.0)
    else:
        margins = (48.0, 120.0, 48.0, 120.0)
    left, top, right, bottom = margins
    expanded = [
        max(page_x0, x0 - left),
        max(page_y0, y0 - top),
        min(page_x1, x1 + right),
        min(page_y1, y1 + bottom),
    ]
    return [round(item, 2) for item in expanded]


def _paper_filename_from_ref(paper_ref: str) -> str:
    prefix = "papers_dir/"
    if not paper_ref.startswith(prefix):
        return ""
    token = paper_ref[len(prefix) :].strip()
    if not token or token.startswith("/") or ".." in Path(token).parts:
        return ""
    return token


def _asset_filename(row: dict[str, Any]) -> str:
    basis = "|".join(
        [
            normalize_text(row.get("sourceCandidateId")),
            normalize_text(row.get("paperId")),
            str(row.get("page") or ""),
            json.dumps(row.get("bbox") or [], ensure_ascii=True),
        ]
    )
    return "{priority:02d}-{paper}-p{page}-{kind}-{digest}.png".format(
        priority=int(row.get("priority") or 0),
        paper=_slug(normalize_text(row.get("paperId"))),
        page=int(row.get("page") or 0),
        kind=_slug(normalize_text(row.get("candidateType"))),
        digest=_short_hash(basis, length=10),
    )


def _blocked_row(
    *,
    source_row: dict[str, Any],
    reason: str,
    detail: str = "",
) -> dict[str, Any]:
    bbox = _round_bbox(source_row.get("bbox"))
    return {
        "schema": VISUAL_ANNOTATION_ATTACHMENT_ROW_SCHEMA_ID,
        "sourcePackCandidateId": normalize_text(source_row.get("packCandidateId")),
        "sourceCandidateId": normalize_text(source_row.get("sourceCandidateId")),
        "paperId": normalize_text(source_row.get("paperId")),
        "paperRef": normalize_text(source_row.get("paperRef")),
        "sourceContentHash": normalize_text(source_row.get("sourceContentHash")),
        "page": int(source_row.get("page") or 0),
        "candidateType": normalize_text(source_row.get("candidateType")),
        "attachmentKind": "context_crop_png",
        "attachmentRef": "",
        "attachmentSha256": "",
        "attachmentBytes": 0,
        "pixelWidth": 0,
        "pixelHeight": 0,
        "sourceBbox": bbox,
        "cropBbox": [],
        "renderPolicy": {
            "renderer": "pymupdf",
            "zoom": 0.0,
            "cropExpansionPolicy": "candidate_type_context_margin_v1",
            "writeScope": "eval_report_attachment_only",
        },
        "provenance": {
            "sourceWebPackSchema": VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID,
            "sourceWebPackRowSchema": VISUAL_ANNOTATION_WEB_PACK_ROW_SCHEMA_ID,
            "sourcePackCandidateId": normalize_text(source_row.get("packCandidateId")),
            "sourceCandidateId": normalize_text(source_row.get("sourceCandidateId")),
            "sourceContentHash": normalize_text(source_row.get("sourceContentHash")),
            "page": int(source_row.get("page") or 0),
            "sourceBbox": bbox,
            "renderMethod": "blocked_attachment_render_v1",
            "detail": normalize_text(detail),
        },
        "retrievalHintUseOnly": True,
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "blockerReason": normalize_text(reason),
    }


def _render_row_attachment(
    *,
    source_row: dict[str, Any],
    papers_root: Path,
    attachment_root: Path,
    attachment_root_ref: str,
    zoom: float,
) -> dict[str, Any]:
    try:
        import fitz  # type: ignore
    except Exception as error:  # pragma: no cover - environment-level failure path
        return _blocked_row(source_row=source_row, reason="pymupdf_unavailable", detail=str(error))

    paper_ref = normalize_text(source_row.get("paperRef"))
    filename = _paper_filename_from_ref(paper_ref)
    if not filename:
        return _blocked_row(source_row=source_row, reason="invalid_paper_ref", detail=paper_ref)
    pdf_path = papers_root / filename
    if not pdf_path.is_file():
        return _blocked_row(source_row=source_row, reason="source_pdf_missing", detail=paper_ref)
    expected_hash = normalize_text(source_row.get("sourceContentHash"))
    try:
        observed_hash = sha256_file(pdf_path)
    except Exception as error:
        return _blocked_row(source_row=source_row, reason="source_hash_failed", detail=str(error))
    if expected_hash and observed_hash != expected_hash:
        return _blocked_row(
            source_row=source_row,
            reason="source_hash_mismatch",
            detail=f"observed={observed_hash}",
        )

    bbox = _round_bbox(source_row.get("bbox"))
    if len(bbox) != 4 or _bbox_area(bbox) <= 0:
        return _blocked_row(source_row=source_row, reason="invalid_bbox", detail=str(source_row.get("bbox")))
    page_number = int(source_row.get("page") or 0)
    if page_number <= 0:
        return _blocked_row(source_row=source_row, reason="invalid_page", detail=str(source_row.get("page")))

    attachment_root.mkdir(parents=True, exist_ok=True)
    asset_filename = _asset_filename(source_row)
    attachment_path = attachment_root / asset_filename
    attachment_ref = f"{attachment_root_ref.rstrip('/')}/assets/{asset_filename}"

    try:
        document = fitz.open(str(pdf_path))
        try:
            if page_number > int(getattr(document, "page_count", 0) or len(document)):
                return _blocked_row(source_row=source_row, reason="page_out_of_range", detail=str(page_number))
            page = document.load_page(page_number - 1)
            crop_bbox = _expanded_bbox(bbox, normalize_text(source_row.get("candidateType")), page.rect)
            if _bbox_area(crop_bbox) <= 0:
                return _blocked_row(source_row=source_row, reason="invalid_crop_bbox", detail=str(crop_bbox))
            clip = fitz.Rect(*crop_bbox)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(float(zoom), float(zoom)), clip=clip, alpha=False)
            pixmap.save(str(attachment_path))
            pixel_width = int(pixmap.width)
            pixel_height = int(pixmap.height)
        finally:
            try:
                document.close()
            except Exception:
                pass
    except Exception as error:
        return _blocked_row(source_row=source_row, reason="attachment_render_failed", detail=str(error))

    attachment_hash = sha256_file(attachment_path)
    attachment_bytes = attachment_path.stat().st_size
    return {
        "schema": VISUAL_ANNOTATION_ATTACHMENT_ROW_SCHEMA_ID,
        "sourcePackCandidateId": normalize_text(source_row.get("packCandidateId")),
        "sourceCandidateId": normalize_text(source_row.get("sourceCandidateId")),
        "paperId": normalize_text(source_row.get("paperId")),
        "paperRef": paper_ref,
        "sourceContentHash": observed_hash,
        "page": page_number,
        "candidateType": normalize_text(source_row.get("candidateType")),
        "attachmentKind": "context_crop_png",
        "attachmentRef": attachment_ref,
        "attachmentSha256": attachment_hash,
        "attachmentBytes": int(attachment_bytes),
        "pixelWidth": pixel_width,
        "pixelHeight": pixel_height,
        "sourceBbox": bbox,
        "cropBbox": crop_bbox,
        "renderPolicy": {
            "renderer": "pymupdf",
            "zoom": float(zoom),
            "cropExpansionPolicy": "candidate_type_context_margin_v1",
            "writeScope": "eval_report_attachment_only",
        },
        "provenance": {
            "sourceWebPackSchema": VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID,
            "sourceWebPackRowSchema": VISUAL_ANNOTATION_WEB_PACK_ROW_SCHEMA_ID,
            "sourcePackCandidateId": normalize_text(source_row.get("packCandidateId")),
            "sourceCandidateId": normalize_text(source_row.get("sourceCandidateId")),
            "sourceContentHash": observed_hash,
            "page": page_number,
            "sourceBbox": bbox,
            "cropBbox": crop_bbox,
            "renderMethod": "pymupdf_context_crop_png_v1",
        },
        "retrievalHintUseOnly": True,
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "blockerReason": "",
    }


def _scope(crop_write_rows: int) -> dict[str, Any]:
    return {
        "writes": "report_and_attachment_files_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "vectorIndexing": False,
        "strictEvidencePromotionRows": 0,
        "runtimeAnswerVisibleExposureRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "answerabilityGateBypassRows": 0,
        "cropWriteRows": int(crop_write_rows),
        "pageImageWriteRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
    }


def _private_path_leak_count(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return 1 if PRIVATE_PATH_RE.search(encoded) else 0


def _counts(rows: Sequence[dict[str, Any]], *, private_path_leak_rows: int) -> dict[str, int]:
    crop_rows = [row for row in rows if not row.get("blockerReason")]
    return {
        "sourcePackRows": len(rows),
        "attachmentRows": len(rows),
        "cropAttachmentRows": len(crop_rows),
        "pageAttachmentRows": 0,
        "blockedRows": sum(1 for row in rows if row.get("blockerReason")),
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": 0,
    }


def build_visual_annotation_attachment_pack(
    web_pack: dict[str, Any],
    *,
    papers_root: Path | None = None,
    output_dir: Path,
    output_dir_ref: str,
    attachment_pack_id: str = DEFAULT_ATTACHMENT_PACK_ID,
    source_web_pack_ref: str = "eval/knowledgeos/reports/visual_annotation_web_pack_001.v1.json",
    generated_at: str | None = None,
    zoom: float = 2.0,
) -> dict[str, Any]:
    root = papers_root or default_papers_root()
    source_rows = [row for row in list(web_pack.get("packRowsDetail") or []) if isinstance(row, dict)]
    attachment_root = output_dir / "assets"
    attachment_root_ref = output_dir_ref.rstrip("/")
    rows = [
        _render_row_attachment(
            source_row=row,
            papers_root=root.expanduser(),
            attachment_root=attachment_root,
            attachment_root_ref=attachment_root_ref,
            zoom=zoom,
        )
        for row in source_rows
    ]
    crop_write_rows = sum(1 for row in rows if not row.get("blockerReason"))
    report: dict[str, Any] = {
        "schema": VISUAL_ANNOTATION_ATTACHMENT_PACK_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "attachmentPackId": attachment_pack_id,
        "sourceWebPack": {
            "schema": normalize_text(web_pack.get("schema")),
            "status": normalize_text(web_pack.get("status")),
            "reportRef": normalize_text(source_web_pack_ref),
            "packRows": len(source_rows),
        },
        "assetRootRef": attachment_root_ref,
        "scope": _scope(crop_write_rows),
        "attachmentPolicy": {
            "attachmentKind": "context_crop_png",
            "renderSource": "already_local_pdf_only",
            "cropExpansionPolicy": "candidate_type_context_margin_v1",
            "pageImagePolicy": "not_written_in_pack_001",
            "uploadGuidance": "Upload the Markdown web pack plus these PNG attachments to the manual web/VLM session.",
            "derivedTextUse": "retrieval_hint_only",
        },
        "counts": {},
        "attachmentRowsDetail": rows,
        "warnings": [
            "Attachment PNGs are eval artifacts for manual web/VLM calibration only.",
            "Generated visual descriptions must remain derivedTextForRetrieval retrieval hints, not evidence.",
            "Full page PNGs are intentionally not written in pack 001 to keep the artifact set small.",
        ],
    }
    private_path_leak_rows = _private_path_leak_count(report)
    report["counts"] = _counts(rows, private_path_leak_rows=private_path_leak_rows)
    if (
        web_pack.get("schema") != VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID
        or web_pack.get("status") != "ready"
        or not rows
        or report["counts"]["blockedRows"]
        or private_path_leak_rows
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
    return report


def render_markdown_pack(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    source = dict(report.get("sourceWebPack") or {})
    lines = [
        "# Visual Annotation Attachment Pack 001",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- attachmentPackId: `{report.get('attachmentPackId')}`",
        f"- sourceWebPack: `{source.get('reportRef')}`",
        f"- assetRootRef: `{report.get('assetRootRef')}`",
        f"- cropAttachmentRows: `{counts.get('cropAttachmentRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- apiCalls: `{scope.get('apiCalls')}`",
        f"- modelCalls: `{scope.get('modelCalls')}`",
        f"- webModelCalls: `{scope.get('webModelCalls')}`",
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
        f"- pageImageWriteRows: `{scope.get('pageImageWriteRows')}`",
        "",
        "## Upload Set",
        "",
        "Upload the web pack Markdown and the PNG refs below to the manual web/VLM session.",
        "",
        "| # | paperId | type | page | pixels | bytes | attachmentRef | blockerReason |",
        "|---:|---|---|---:|---|---:|---|---|",
    ]
    for index, row in enumerate(report.get("attachmentRowsDetail", []), start=1):
        pixels = f"{row.get('pixelWidth')}x{row.get('pixelHeight')}"
        lines.append(
            "| {index} | {paperId} | {candidateType} | {page} | `{pixels}` | {bytes} | `{ref}` | `{blocker}` |".format(
                index=index,
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                page=row.get("page"),
                pixels=pixels,
                bytes=row.get("attachmentBytes"),
                ref=row.get("attachmentRef"),
                blocker=row.get("blockerReason"),
            )
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_visual_annotation_attachment_pack(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown_pack(report), encoding="utf-8")
    return {
        "json": str(report_json),
        "markdown": str(report_md),
    }


__all__ = [
    "DEFAULT_ATTACHMENT_PACK_ID",
    "NEXT_RECOMMENDED_TRANCHE",
    "READY_DECISION",
    "VISUAL_ANNOTATION_ATTACHMENT_PACK_SCHEMA_ID",
    "VISUAL_ANNOTATION_ATTACHMENT_ROW_SCHEMA_ID",
    "build_visual_annotation_attachment_pack",
    "load_web_pack",
    "render_markdown_pack",
    "sanitized_report_ref",
    "write_visual_annotation_attachment_pack",
]
