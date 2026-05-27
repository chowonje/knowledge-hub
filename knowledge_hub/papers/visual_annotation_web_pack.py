"""Build small web-ready visual annotation packs from candidate reports.

This helper consumes the report-only visual/layout candidate list and emits a
bounded manual web/VLM pack. It does not call models, write crops, mutate
indexes, or promote future derived visual text to evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.visual_layout_candidate_list_report import (
    VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID,
)


VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID = "knowledge-hub.paper.visual-annotation-web-pack.v1"
VISUAL_ANNOTATION_WEB_PACK_ROW_SCHEMA_ID = "knowledge-hub.paper.visual-annotation-web-pack-row.v1"

DEFAULT_PACK_ID = "visual_annotation_web_pack_001"
READY_DECISION = "ready_for_manual_web_vlm_calibration"
NEXT_RECOMMENDED_TRANCHE = "visual_annotation_web_pack_manual_calibration"

DEFAULT_MAX_CANDIDATES = 18
DEFAULT_PAPER_IDS: tuple[str, ...] = ("alexnet-2012", "resnet-2015")
DEFAULT_CANDIDATE_TYPES: tuple[str, ...] = (
    "figure_caption_region",
    "table_region",
    "equation_region",
)

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)

TYPE_PRIORITY = {
    "figure_caption_region": 0,
    "table_region": 1,
    "equation_region": 2,
    "layout_region": 3,
    "image_region": 4,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def bounded_text(value: Any, *, limit: int = 480) -> str:
    text = normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _short_hash(value: str, *, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_.-]+", "-", str(value or "").lower()).strip("-")
    return slug or "unknown"


def sanitized_report_ref(path: Path, *, project_root: Path | None = None) -> str:
    """Return a public-safe report reference without absolute local paths."""

    resolved = path.expanduser()
    if project_root is not None:
        try:
            rel = resolved.resolve().relative_to(project_root.resolve())
            return rel.as_posix()
        except Exception:
            pass
    return f"input_reports/{resolved.name}"


def load_candidate_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _retrieval_hint_plan() -> dict[str, Any]:
    return {
        "targetDerivedTextField": "derivedTextForRetrieval",
        "allowedUse": "retrieval_hint_only",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
    }


def _annotation_task(candidate_type: str) -> str:
    if candidate_type == "figure_caption_region":
        return "describe_figure_caption_region_for_retrieval"
    if candidate_type == "table_region":
        return "describe_table_region_for_retrieval"
    if candidate_type == "equation_region":
        return "describe_equation_region_for_retrieval"
    if candidate_type == "image_region":
        return "describe_image_region_for_retrieval"
    return "describe_layout_region_for_retrieval"


def _copy_paste_context(row: dict[str, Any]) -> str:
    text_context = row.get("textContext") if isinstance(row.get("textContext"), dict) else {}
    parts = [
        f"paperId={normalize_text(row.get('paperId'))}",
        f"page={row.get('page')}",
        f"candidateType={normalize_text(row.get('candidateType'))}",
        f"bbox={row.get('bbox')}",
    ]
    caption = bounded_text(text_context.get("captionText"), limit=320)
    nearby = bounded_text(text_context.get("nearbyText"), limit=520)
    headings = text_context.get("headingPath") if isinstance(text_context.get("headingPath"), list) else []
    if caption:
        parts.append(f"captionText={caption}")
    if headings:
        parts.append("headingPath=" + " > ".join(normalize_text(item) for item in headings if normalize_text(item)))
    if nearby:
        parts.append(f"nearbyText={nearby}")
    return "\n".join(parts)


def _pack_row(*, pack_id: str, source_row: dict[str, Any], priority: int) -> dict[str, Any]:
    visual_context = source_row.get("visualContext") if isinstance(source_row.get("visualContext"), dict) else {}
    text_context = source_row.get("textContext") if isinstance(source_row.get("textContext"), dict) else {}
    provenance = source_row.get("provenance") if isinstance(source_row.get("provenance"), dict) else {}
    candidate_id = normalize_text(source_row.get("candidateId"))
    pack_candidate_id = "visual-annotation-web-pack:{pack}:{digest}".format(
        pack=_slug(pack_id),
        digest=_short_hash(candidate_id or json.dumps(source_row, ensure_ascii=True, sort_keys=True), length=16),
    )
    return {
        "schema": VISUAL_ANNOTATION_WEB_PACK_ROW_SCHEMA_ID,
        "packCandidateId": pack_candidate_id,
        "sourceCandidateId": candidate_id,
        "paperId": normalize_text(source_row.get("paperId")),
        "paperRef": normalize_text(source_row.get("paperRef")),
        "sourceContentHash": normalize_text(source_row.get("sourceContentHash")),
        "page": int(source_row.get("page") or 0),
        "bbox": list(source_row.get("bbox") or []),
        "candidateType": normalize_text(source_row.get("candidateType")),
        "priority": int(priority),
        "annotationTask": _annotation_task(normalize_text(source_row.get("candidateType"))),
        "webInput": {
            "copyPasteContext": _copy_paste_context(source_row),
            "attachmentInstruction": (
                "Attach the matching page or crop image manually if available. "
                "If no image is attached, do not invent visual details; use only supplied text and bbox metadata."
            ),
            "plannedAttachmentRef": normalize_text(visual_context.get("cropRef")),
            "pageImageRequired": bool(visual_context.get("pageImageRequired", True)),
        },
        "textContext": {
            "nearbyText": bounded_text(text_context.get("nearbyText"), limit=520),
            "captionText": bounded_text(text_context.get("captionText"), limit=320),
            "headingPath": [
                normalize_text(item)
                for item in list(text_context.get("headingPath") or [])
                if normalize_text(item)
            ],
        },
        "visualContext": {
            "cropRef": normalize_text(visual_context.get("cropRef")),
            "imageHash": normalize_text(visual_context.get("imageHash")),
            "pageImageRequired": bool(visual_context.get("pageImageRequired", True)),
        },
        "retrievalHintPlan": _retrieval_hint_plan(),
        "expectedOutputContract": {
            "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
            "requiredFields": [
                "sourceCandidateId",
                "derivedTextForRetrieval",
                "visibleText",
                "retrievalKeywords",
                "uncertainty",
                "limitations",
                "strictEvidence",
                "citationGrade",
                "answerableWithoutTextEvidence",
            ],
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
        },
        "provenance": {
            "sourceReportSchema": VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID,
            "sourceCandidateId": candidate_id,
            "sourceContentHash": normalize_text(provenance.get("sourceContentHash"))
            or normalize_text(source_row.get("sourceContentHash")),
            "page": int(provenance.get("page") or source_row.get("page") or 0),
            "bbox": list(provenance.get("bbox") or source_row.get("bbox") or []),
            "extractionMethod": normalize_text(provenance.get("extractionMethod")),
        },
        "blockerReason": normalize_text(source_row.get("blockerReason")),
    }


def _paper_rank(paper_id: str, preferred_paper_ids: Sequence[str]) -> int:
    try:
        return list(preferred_paper_ids).index(paper_id)
    except ValueError:
        return len(preferred_paper_ids) + 1


def select_web_pack_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    preferred_paper_ids: Sequence[str] = DEFAULT_PAPER_IDS,
    included_candidate_types: Sequence[str] = DEFAULT_CANDIDATE_TYPES,
) -> list[dict[str, Any]]:
    included_types = set(included_candidate_types)
    preferred_ids = set(preferred_paper_ids)
    eligible = [
        row
        for row in rows
        if isinstance(row, dict)
        and not row.get("blockerReason")
        and normalize_text(row.get("candidateType")) in included_types
        and (not preferred_ids or normalize_text(row.get("paperId")) in preferred_ids)
    ]
    eligible.sort(
        key=lambda row: (
            _paper_rank(normalize_text(row.get("paperId")), preferred_paper_ids),
            int(row.get("page") or 0),
            TYPE_PRIORITY.get(normalize_text(row.get("candidateType")), 99),
            json.dumps(row.get("bbox") or [], ensure_ascii=True),
            normalize_text(row.get("candidateId")),
        )
    )
    return eligible[: max(0, int(max_candidates))]


def _scope() -> dict[str, Any]:
    return {
        "writes": "report_only",
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
        "cropWriteRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
    }


def _prompt_contract() -> dict[str, Any]:
    system_prompt = "\n".join(
        [
            "You are generating visual/layout retrieval hints only.",
            "Do not create citation-grade evidence.",
            "Do not answer scientific questions.",
            "Do not infer facts not visible in the attached image or supplied nearby text.",
            "Do not mark anything as strict evidence.",
            "Return only schema-valid JSON.",
            "If an image/page crop is not attached, set visualObservationStatus to image_not_attached.",
            "The output field derivedTextForRetrieval is retrieval_hint_only and must not be used as answer evidence.",
        ]
    )
    user_prompt = "\n".join(
        [
            "For each candidate row in this pack, produce one output row.",
            "Describe only visible layout/object/table/figure/equation structure, readable visible text, retrieval keywords, uncertainty, and limitations.",
            "Keep strictEvidence=false, citationGrade=false, and answerableWithoutTextEvidence=false for every row.",
            "Return JSON with top-level schema knowledge-hub.paper.visual-annotation-web-output.v1 and a rows array.",
        ]
    )
    output_shape = {
        "schema": "knowledge-hub.paper.visual-annotation-web-output.v1",
        "rows": [
            {
                "sourceCandidateId": "string",
                "visualObservationStatus": "image_attached | image_not_attached | unclear",
                "derivedTextForRetrieval": "string",
                "visibleText": "string",
                "retrievalKeywords": ["string"],
                "uncertainty": "string",
                "limitations": "string",
                "strictEvidence": False,
                "citationGrade": False,
                "answerableWithoutTextEvidence": False,
            }
        ],
    }
    return {
        "targetSurface": "manual_web_vlm_calibration",
        "modelGuidance": "Use a high-capability web VLM/Pro model for manual calibration; do not run this pack through product answer generation.",
        "systemPrompt": system_prompt,
        "userPrompt": user_prompt,
        "outputShape": output_shape,
    }


def _private_path_leak_count(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return 1 if PRIVATE_PATH_RE.search(encoded) else 0


def _counts(
    *,
    source_candidate_rows: int,
    eligible_candidate_rows: int,
    selected_rows: Sequence[dict[str, Any]],
    private_path_leak_rows: int,
) -> dict[str, int]:
    by_type = Counter(row.get("candidateType") for row in selected_rows)
    return {
        "sourceCandidateRows": int(source_candidate_rows),
        "eligibleCandidateRows": int(eligible_candidate_rows),
        "selectedCandidateRows": len(selected_rows),
        "selectedPaperRows": len({row.get("paperId") for row in selected_rows}),
        "excludedCandidateRows": max(0, int(source_candidate_rows) - len(selected_rows)),
        "figureCandidateRows": int(by_type.get("figure_caption_region", 0)),
        "tableCandidateRows": int(by_type.get("table_region", 0)),
        "equationCandidateRows": int(by_type.get("equation_region", 0)),
        "layoutCandidateRows": int(by_type.get("layout_region", 0)),
        "imageCandidateRows": int(by_type.get("image_region", 0)),
        "blockedRows": sum(1 for row in selected_rows if row.get("blockerReason")),
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": 0,
    }


def build_visual_annotation_web_pack(
    source_report: dict[str, Any],
    *,
    pack_id: str = DEFAULT_PACK_ID,
    source_report_ref: str = "eval/knowledgeos/reports/visual_layout_candidate_list_report.v1.json",
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    preferred_paper_ids: Sequence[str] = DEFAULT_PAPER_IDS,
    included_candidate_types: Sequence[str] = DEFAULT_CANDIDATE_TYPES,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_rows = [
        row for row in list(source_report.get("candidateRowsDetail") or []) if isinstance(row, dict)
    ]
    selected_source_rows = select_web_pack_candidates(
        source_rows,
        max_candidates=max_candidates,
        preferred_paper_ids=preferred_paper_ids,
        included_candidate_types=included_candidate_types,
    )
    eligible_rows = select_web_pack_candidates(
        source_rows,
        max_candidates=len(source_rows),
        preferred_paper_ids=preferred_paper_ids,
        included_candidate_types=included_candidate_types,
    )
    pack_rows = [
        _pack_row(pack_id=pack_id, source_row=row, priority=index + 1)
        for index, row in enumerate(selected_source_rows)
    ]
    report: dict[str, Any] = {
        "schema": VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "packId": pack_id,
        "sourceReport": {
            "schema": normalize_text(source_report.get("schema")),
            "status": normalize_text(source_report.get("status")),
            "reportRef": normalize_text(source_report_ref),
            "candidateRows": len(source_rows),
        },
        "scope": _scope(),
        "selectionPolicy": {
            "packUse": "manual_web_vlm_retrieval_hint_calibration",
            "maxCandidateRows": int(max_candidates),
            "preferredPaperIds": [normalize_text(item) for item in preferred_paper_ids if normalize_text(item)],
            "includedCandidateTypes": [
                normalize_text(item) for item in included_candidate_types if normalize_text(item)
            ],
            "excludedCandidateTypes": [
                "image_region",
                "layout_region",
            ],
            "cropPolicy": "planned_ref_only_no_crop_write",
            "imageAttachmentPolicy": "manual_page_or_crop_attachment_only",
            "rationale": (
                "Start with caption/table/equation candidates from the two smallest papers so web/Pro "
                "manual calibration stays inspectable before broader image-region processing."
            ),
        },
        "prompt": _prompt_contract(),
        "counts": {},
        "packRowsDetail": pack_rows,
        "warnings": [
            "plannedAttachmentRef values are sanitized planned refs only; this tranche writes no crop files.",
            "image_region candidates are intentionally excluded from pack 001 to avoid noisy web/VLM calibration.",
            "Any returned derivedTextForRetrieval must remain retrieval_hint_only and non-evidence.",
        ],
    }
    private_path_leak_rows = _private_path_leak_count(report)
    report["counts"] = _counts(
        source_candidate_rows=len(source_rows),
        eligible_candidate_rows=len(eligible_rows),
        selected_rows=pack_rows,
        private_path_leak_rows=private_path_leak_rows,
    )
    if (
        source_report.get("schema") != VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID
        or source_report.get("status") != "ready"
        or not pack_rows
        or private_path_leak_rows
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
    return report


def render_markdown_pack(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    prompt = dict(report.get("prompt") or {})
    source = dict(report.get("sourceReport") or {})
    lines = [
        "# Visual Annotation Web Pack 001",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- packId: `{report.get('packId')}`",
        f"- sourceReport: `{source.get('reportRef')}`",
        f"- selectedCandidateRows: `{counts.get('selectedCandidateRows')}`",
        f"- selectedPaperRows: `{counts.get('selectedPaperRows')}`",
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
        "",
        "## Copy-Paste Prompt",
        "",
        "### System",
        "",
        "```text",
        str(prompt.get("systemPrompt") or ""),
        "```",
        "",
        "### User",
        "",
        "```text",
        str(prompt.get("userPrompt") or ""),
        "```",
        "",
        "### Output Shape",
        "",
        "```json",
        json.dumps(prompt.get("outputShape") or {}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Candidate Rows",
        "",
        "| # | paperId | type | page | bbox | plannedAttachmentRef | caption/context |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for row in report.get("packRowsDetail", []):
        text_context = row.get("textContext") if isinstance(row.get("textContext"), dict) else {}
        visual_context = row.get("visualContext") if isinstance(row.get("visualContext"), dict) else {}
        caption = bounded_text(text_context.get("captionText") or text_context.get("nearbyText"), limit=160)
        lines.append(
            "| {priority} | {paperId} | {candidateType} | {page} | `{bbox}` | `{ref}` | {caption} |".format(
                priority=row.get("priority"),
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                page=row.get("page"),
                bbox=row.get("bbox"),
                ref=visual_context.get("cropRef"),
                caption=caption.replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## Row JSON",
            "",
            "```json",
            json.dumps(report.get("packRowsDetail") or [], ensure_ascii=False, indent=2),
            "```",
        ]
    )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_visual_annotation_web_pack(
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
    "DEFAULT_CANDIDATE_TYPES",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_PACK_ID",
    "DEFAULT_PAPER_IDS",
    "NEXT_RECOMMENDED_TRANCHE",
    "READY_DECISION",
    "VISUAL_ANNOTATION_WEB_PACK_ROW_SCHEMA_ID",
    "VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID",
    "build_visual_annotation_web_pack",
    "load_candidate_report",
    "render_markdown_pack",
    "sanitized_report_ref",
    "select_web_pack_candidates",
    "write_visual_annotation_web_pack",
]
