"""Render local PDF crops for the visual annotation expansion pack.

This Phase 8 side-track helper consumes the report-only expansion pack design,
writes only context-crop PNG eval attachments plus reports, and never calls
GPT/VLM, writes whole-page images, mutates stores, indexes vectors, or promotes
derived visual text to evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
import json
from typing import Any

from knowledge_hub.papers.visual_annotation_attachment_pack import (
    PRIVATE_PATH_RE,
    _render_row_attachment,
    default_papers_root,
    load_web_pack,
    normalize_text,
    sanitized_report_ref,
    sha256_file,
    utc_now_iso,
)
from knowledge_hub.papers.visual_annotation_expansion_pack_design import (
    VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID,
    VISUAL_ANNOTATION_EXPANSION_PACK_ROW_SCHEMA_ID,
)


VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-expansion-attachment-pack.v1"
)
VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-expansion-attachment-row.v1"
)

DEFAULT_EXPANSION_ATTACHMENT_PACK_ID = "visual_annotation_expansion_attachment_pack_002"
READY_DECISION = "ready_for_manual_web_vlm_expansion_run"
NEXT_RECOMMENDED_TRANCHE = "visual_annotation_expansion_manual_output_capture"


def load_expansion_pack(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _convert_rendered_row(row: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
    converted = deepcopy(row)
    source_pack_candidate_id = normalize_text(source_row.get("packCandidateId"))
    candidate_type = normalize_text(source_row.get("candidateType"))
    converted["schema"] = VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_ROW_SCHEMA_ID
    converted["sourcePackCandidateId"] = source_pack_candidate_id
    converted["wholeImageAttachment"] = False
    converted["pageImageAttachment"] = False
    old_provenance = dict(converted.get("provenance") or {})
    provenance = {
        "sourceExpansionPackSchema": VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID,
        "sourceExpansionPackRowSchema": VISUAL_ANNOTATION_EXPANSION_PACK_ROW_SCHEMA_ID,
        "sourcePackCandidateId": source_pack_candidate_id,
        "sourceCandidateId": normalize_text(source_row.get("sourceCandidateId")),
        "sourceContentHash": normalize_text(converted.get("sourceContentHash"))
        or normalize_text(source_row.get("sourceContentHash")),
        "page": int(converted.get("page") or source_row.get("page") or 0),
        "sourceBbox": list(converted.get("sourceBbox") or source_row.get("bbox") or []),
        "cropBbox": list(converted.get("cropBbox") or []),
        "renderMethod": normalize_text(old_provenance.get("renderMethod")),
        "attachmentSource": "visual_annotation_expansion_pack_design_v1",
    }
    if old_provenance.get("detail"):
        provenance["detail"] = normalize_text(old_provenance.get("detail"))
    converted["provenance"] = provenance
    converted["retrievalHintUseOnly"] = True
    converted["strictEvidence"] = False
    converted["citationGrade"] = False
    converted["answerableWithoutTextEvidence"] = False
    if candidate_type == "image_region":
        converted["fullImageEscalationStatus"] = "deferred_to_full_image_gate"
    else:
        converted["fullImageEscalationStatus"] = "not_required_for_this_context_crop_gate"
    return converted


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
        "wholeImageWriteRows": 0,
        "wholeImageGptRows": 0,
        "candidateStoreMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
    }


def _private_path_leak_count(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return 1 if PRIVATE_PATH_RE.search(encoded) else 0


def _counts(rows: Sequence[dict[str, Any]], *, private_path_leak_rows: int) -> dict[str, int]:
    crop_rows = [row for row in rows if not row.get("blockerReason")]
    by_type = Counter(row.get("candidateType") for row in rows if not row.get("blockerReason"))
    return {
        "sourcePackRows": len(rows),
        "attachmentRows": len(rows),
        "cropAttachmentRows": len(crop_rows),
        "pageAttachmentRows": 0,
        "wholeImageRows": 0,
        "imageCandidateRows": int(by_type.get("image_region", 0)),
        "figureCandidateRows": int(by_type.get("figure_caption_region", 0)),
        "tableCandidateRows": int(by_type.get("table_region", 0)),
        "equationCandidateRows": int(by_type.get("equation_region", 0)),
        "layoutCandidateRows": int(by_type.get("layout_region", 0)),
        "blockedRows": sum(1 for row in rows if row.get("blockerReason")),
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": 0,
    }


def build_visual_annotation_expansion_attachment_pack(
    expansion_pack: dict[str, Any],
    *,
    papers_root: Path | None = None,
    output_dir: Path,
    output_dir_ref: str,
    attachment_pack_id: str = DEFAULT_EXPANSION_ATTACHMENT_PACK_ID,
    source_expansion_pack_ref: str = "eval/knowledgeos/reports/visual_annotation_expansion_pack_design.v1.json",
    generated_at: str | None = None,
    zoom: float = 2.0,
) -> dict[str, Any]:
    root = papers_root or default_papers_root()
    source_rows = [row for row in list(expansion_pack.get("packRowsDetail") or []) if isinstance(row, dict)]
    attachment_root = output_dir / "assets"
    attachment_root_ref = output_dir_ref.rstrip("/")
    rows = [
        _convert_rendered_row(
            _render_row_attachment(
                source_row=row,
                papers_root=root.expanduser(),
                attachment_root=attachment_root,
                attachment_root_ref=attachment_root_ref,
                zoom=zoom,
            ),
            row,
        )
        for row in source_rows
    ]
    crop_write_rows = sum(1 for row in rows if not row.get("blockerReason"))
    report: dict[str, Any] = {
        "schema": VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "attachmentPackId": attachment_pack_id,
        "sourceExpansionPack": {
            "schema": normalize_text(expansion_pack.get("schema")),
            "status": normalize_text(expansion_pack.get("status")),
            "reportRef": normalize_text(source_expansion_pack_ref),
            "packRows": len(source_rows),
        },
        "assetRootRef": attachment_root_ref,
        "scope": _scope(crop_write_rows),
        "attachmentPolicy": {
            "attachmentKind": "context_crop_png",
            "renderSource": "already_local_pdf_only",
            "cropExpansionPolicy": "candidate_type_context_margin_v1",
            "pageImagePolicy": "not_written_in_expansion_pack_002",
            "wholeImagePolicy": "deferred_to_visual_full_image_annotation_pack_design",
            "uploadGuidance": (
                "Upload these 24 PNG context crops with the expansion pack row JSON to the manual web/VLM session."
            ),
            "derivedTextUse": "retrieval_hint_only",
        },
        "counts": {},
        "attachmentRowsDetail": rows,
        "warnings": [
            "Attachment PNGs are eval artifacts for manual web/VLM expansion only.",
            "Generated visual descriptions must remain derivedTextForRetrieval retrieval hints, not evidence.",
            "Whole-image and whole-page GPT/VLM work remains deferred to the full-image gate.",
        ],
    }
    private_path_leak_rows = _private_path_leak_count(report)
    report["counts"] = _counts(rows, private_path_leak_rows=private_path_leak_rows)
    if (
        expansion_pack.get("schema") != VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID
        or expansion_pack.get("status") != "ready"
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
    source = dict(report.get("sourceExpansionPack") or {})
    lines = [
        "# Visual Annotation Expansion Attachment Pack 002",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- attachmentPackId: `{report.get('attachmentPackId')}`",
        f"- sourceExpansionPack: `{source.get('reportRef')}`",
        f"- assetRootRef: `{report.get('assetRootRef')}`",
        f"- cropAttachmentRows: `{counts.get('cropAttachmentRows')}`",
        f"- imageCandidateRows: `{counts.get('imageCandidateRows')}`",
        f"- figureCandidateRows: `{counts.get('figureCandidateRows')}`",
        f"- tableCandidateRows: `{counts.get('tableCandidateRows')}`",
        f"- equationCandidateRows: `{counts.get('equationCandidateRows')}`",
        f"- wholeImageRows: `{counts.get('wholeImageRows')}`",
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
        f"- wholeImageWriteRows: `{scope.get('wholeImageWriteRows')}`",
        f"- wholeImageGptRows: `{scope.get('wholeImageGptRows')}`",
        "",
        "## Upload Set",
        "",
        "Upload the expansion pack row JSON and the PNG refs below to the manual web/VLM session.",
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


def write_visual_annotation_expansion_attachment_pack(
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
    "DEFAULT_EXPANSION_ATTACHMENT_PACK_ID",
    "NEXT_RECOMMENDED_TRANCHE",
    "READY_DECISION",
    "VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID",
    "VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_ROW_SCHEMA_ID",
    "build_visual_annotation_expansion_attachment_pack",
    "default_papers_root",
    "load_expansion_pack",
    "load_web_pack",
    "render_markdown_pack",
    "sanitized_report_ref",
    "sha256_file",
    "write_visual_annotation_expansion_attachment_pack",
]
