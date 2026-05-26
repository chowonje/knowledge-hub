"""Design the next bounded visual annotation expansion pack.

This report-only helper consumes the 467-row visual/layout candidate list plus
the completed first manual web/VLM loop, then selects the next small batch for
manual visual annotation. It writes no crops, calls no models, and does not
index or promote derived visual text.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.visual_annotation_web_pack import VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID
from knowledge_hub.papers.visual_layout_candidate_list_report import (
    VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_dry_run import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID,
)


VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-expansion-pack-design.v1"
)
VISUAL_ANNOTATION_EXPANSION_PACK_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-expansion-pack-row.v1"
)

DEFAULT_EXPANSION_PACK_ID = "visual_annotation_expansion_pack_002"
DEFAULT_MAX_CANDIDATES = 24
DEFAULT_TYPE_QUOTAS: dict[str, int] = {
    "image_region": 8,
    "figure_caption_region": 8,
    "table_region": 5,
    "equation_region": 3,
}
DEFAULT_PAPER_IDS: tuple[str, ...] = ("clip-2021", "mae-2021", "alexnet-2012", "resnet-2015")

READY_DECISION = "ready_for_visual_annotation_expansion_attachment_pack"
NEXT_RECOMMENDED_TRANCHE = "visual_annotation_expansion_attachment_pack"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)

TYPE_PRIORITY = {
    "image_region": 0,
    "figure_caption_region": 1,
    "table_region": 2,
    "equation_region": 3,
    "layout_region": 4,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def bounded_text(value: Any, *, limit: int = 520) -> str:
    text = normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitized_report_ref(path: Path, *, project_root: Path | None = None) -> str:
    resolved = path.expanduser()
    if project_root is not None:
        try:
            rel = resolved.resolve().relative_to(project_root.resolve())
            return rel.as_posix()
        except Exception:
            pass
    return f"input_reports/{resolved.name}"


def _short_hash(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _slug(value: str) -> str:
    token = re.sub(r"[^a-z0-9_.-]+", "-", str(value or "").lower()).strip("-")
    return token or "unknown"


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _paper_rank(paper_id: str, preferred_paper_ids: Sequence[str]) -> int:
    try:
        return list(preferred_paper_ids).index(paper_id)
    except ValueError:
        return len(preferred_paper_ids) + 1


def _previous_candidate_ids(web_pack: dict[str, Any], dry_run_report: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for row in list(web_pack.get("packRowsDetail") or []):
        if isinstance(row, dict):
            candidate_id = normalize_text(row.get("sourceCandidateId"))
            if candidate_id:
                ids.add(candidate_id)
    for row in list(dry_run_report.get("dryRunRowsDetail") or []):
        if isinstance(row, dict):
            candidate_id = normalize_text(row.get("sourceCandidateId"))
            if candidate_id:
                ids.add(candidate_id)
    return ids


def _retrieval_hint_plan() -> dict[str, Any]:
    return {
        "targetDerivedTextField": "derivedTextForRetrieval",
        "allowedUse": "retrieval_hint_only",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
    }


def _annotation_task(candidate_type: str) -> str:
    if candidate_type == "image_region":
        return "describe_image_region_for_retrieval"
    if candidate_type == "figure_caption_region":
        return "describe_figure_caption_region_for_retrieval"
    if candidate_type == "table_region":
        return "describe_table_region_for_retrieval"
    if candidate_type == "equation_region":
        return "describe_equation_region_for_retrieval"
    return "describe_layout_region_for_retrieval"


def _expansion_reason(candidate_type: str, paper_id: str) -> str:
    if candidate_type == "image_region":
        return "first_image_region_probe_after_text_caption_batch"
    if paper_id in {"clip-2021", "mae-2021"}:
        return "new_paper_visual_coverage"
    return "remaining_high_value_visual_candidate"


def _recommended_attachment_kind(candidate_type: str) -> str:
    if candidate_type == "image_region":
        return "context_crop_png"
    return "context_crop_png"


def _attachment_guidance(candidate_type: str) -> dict[str, Any]:
    return {
        "recommendedAttachmentKind": _recommended_attachment_kind(candidate_type),
        "wholeImageAllowedInThisPack": False,
        "pageImageAllowedInThisPack": False,
        "pageCropFallbackAllowed": candidate_type in {"image_region", "figure_caption_region"},
        "fullImageGateRequired": candidate_type == "image_region",
    }


def _copy_paste_context(row: dict[str, Any]) -> str:
    text_context = row.get("textContext") if isinstance(row.get("textContext"), dict) else {}
    visual_context = row.get("visualContext") if isinstance(row.get("visualContext"), dict) else {}
    parts = [
        f"paperId={normalize_text(row.get('paperId'))}",
        f"page={row.get('page')}",
        f"candidateType={normalize_text(row.get('candidateType'))}",
        f"bbox={row.get('bbox')}",
    ]
    crop_ref = normalize_text(visual_context.get("cropRef"))
    caption = bounded_text(text_context.get("captionText"), limit=280)
    nearby = bounded_text(text_context.get("nearbyText"), limit=520)
    if crop_ref:
        parts.append(f"plannedCropRef={crop_ref}")
    if caption:
        parts.append(f"captionText={caption}")
    if nearby:
        parts.append(f"nearbyText={nearby}")
    return "\n".join(parts)


def _pack_row(*, pack_id: str, source_row: dict[str, Any], priority: int) -> dict[str, Any]:
    candidate_type = normalize_text(source_row.get("candidateType"))
    paper_id = normalize_text(source_row.get("paperId"))
    candidate_id = normalize_text(source_row.get("candidateId"))
    text_context = source_row.get("textContext") if isinstance(source_row.get("textContext"), dict) else {}
    visual_context = source_row.get("visualContext") if isinstance(source_row.get("visualContext"), dict) else {}
    provenance = source_row.get("provenance") if isinstance(source_row.get("provenance"), dict) else {}
    pack_candidate_id = "visual-annotation-expansion-pack:{pack}:{digest}".format(
        pack=_slug(pack_id),
        digest=_short_hash(candidate_id or json.dumps(source_row, ensure_ascii=True, sort_keys=True)),
    )
    return {
        "schema": VISUAL_ANNOTATION_EXPANSION_PACK_ROW_SCHEMA_ID,
        "packCandidateId": pack_candidate_id,
        "sourceCandidateId": candidate_id,
        "paperId": paper_id,
        "paperRef": normalize_text(source_row.get("paperRef")),
        "sourceContentHash": normalize_text(source_row.get("sourceContentHash")),
        "page": int(source_row.get("page") or 0),
        "bbox": list(source_row.get("bbox") or []),
        "candidateType": candidate_type,
        "priority": int(priority),
        "expansionReason": _expansion_reason(candidate_type, paper_id),
        "annotationTask": _annotation_task(candidate_type),
        "webInput": {
            "copyPasteContext": _copy_paste_context(source_row),
            "plannedAttachmentRef": normalize_text(visual_context.get("cropRef")),
            "attachmentGuidance": _attachment_guidance(candidate_type),
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
            "wholeImageAllowedInThisPack": False,
        },
        "retrievalHintPlan": _retrieval_hint_plan(),
        "expectedOutputContract": {
            "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "requiredFields": [
                "sourceCandidateId",
                "visualObservationStatus",
                "derivedTextForRetrieval",
                "visibleText",
                "retrievalKeywords",
                "uncertainty",
                "limitations",
                "strictEvidence",
                "citationGrade",
                "answerableWithoutTextEvidence",
            ],
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
        "blockerReason": "",
    }


def _eligible_rows(
    candidate_rows: Iterable[dict[str, Any]],
    *,
    previous_candidate_ids: set[str],
    included_types: set[str],
    preferred_paper_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in candidate_rows
        if isinstance(row, dict)
        and not row.get("blockerReason")
        and normalize_text(row.get("candidateId")) not in previous_candidate_ids
        and normalize_text(row.get("candidateType")) in included_types
        and normalize_text(row.get("paperId")) in set(preferred_paper_ids)
    ]
    rows.sort(
        key=lambda row: (
            TYPE_PRIORITY.get(normalize_text(row.get("candidateType")), 99),
            _paper_rank(normalize_text(row.get("paperId")), preferred_paper_ids),
            int(row.get("page") or 0),
            normalize_text(row.get("candidateId")),
        )
    )
    return rows


def select_expansion_candidates(
    candidate_rows: Iterable[dict[str, Any]],
    *,
    previous_candidate_ids: set[str],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    type_quotas: dict[str, int] | None = None,
    preferred_paper_ids: Sequence[str] = DEFAULT_PAPER_IDS,
    max_per_paper_type_page: int = 2,
) -> list[dict[str, Any]]:
    quotas = dict(type_quotas or DEFAULT_TYPE_QUOTAS)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_page_counts: Counter[tuple[str, str, int]] = Counter()
    rows = _eligible_rows(
        candidate_rows,
        previous_candidate_ids=previous_candidate_ids,
        included_types=set(quotas),
        preferred_paper_ids=preferred_paper_ids,
    )
    for candidate_type, quota in quotas.items():
        type_rows_by_paper: dict[str, list[dict[str, Any]]] = {
            paper_id: [
                row
                for row in rows
                if normalize_text(row.get("candidateType")) == candidate_type
                and normalize_text(row.get("paperId")) == paper_id
            ]
            for paper_id in preferred_paper_ids
        }
        picked = 0
        while picked < quota and len(selected) < max_candidates:
            made_progress = False
            for paper_id in preferred_paper_ids:
                bucket = type_rows_by_paper.get(paper_id) or []
                while bucket:
                    row = bucket.pop(0)
                    candidate_id = normalize_text(row.get("candidateId"))
                    page_key = (
                        normalize_text(row.get("paperId")),
                        normalize_text(row.get("candidateType")),
                        int(row.get("page") or 0),
                    )
                    if candidate_id in selected_ids:
                        continue
                    if per_page_counts[page_key] >= max_per_paper_type_page:
                        continue
                    selected.append(row)
                    selected_ids.add(candidate_id)
                    per_page_counts[page_key] += 1
                    picked += 1
                    made_progress = True
                    break
                if picked >= quota or len(selected) >= max_candidates:
                    break
            if not made_progress:
                break
        if len(selected) >= max_candidates:
            break
    if len(selected) < max_candidates:
        for row in rows:
            candidate_id = normalize_text(row.get("candidateId"))
            page_key = (
                normalize_text(row.get("paperId")),
                normalize_text(row.get("candidateType")),
                int(row.get("page") or 0),
            )
            if candidate_id in selected_ids:
                continue
            if per_page_counts[page_key] >= max_per_paper_type_page:
                continue
            selected.append(row)
            selected_ids.add(candidate_id)
            per_page_counts[page_key] += 1
            if len(selected) >= max_candidates:
                break
    return selected[:max_candidates]


def _scope(selected_rows: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "selectedExpansionRows": int(selected_rows),
        "wholeImageGptRows": 0,
        "pageImageWriteRows": 0,
        "cropWriteRows": 0,
        "candidateStoreMutationRows": 0,
        "vectorIndexing": False,
        "indexMutationRows": 0,
        "runtimeAnswerVisibleExposureRows": 0,
        "strictEvidencePromotionRows": 0,
        "databaseMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "answerabilityGateBypassRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
    }


def _selection_policy(type_quotas: dict[str, int], max_candidates: int) -> dict[str, Any]:
    return {
        "packUse": "manual_web_vlm_retrieval_hint_expansion",
        "maxCandidateRows": int(max_candidates),
        "typeQuotas": type_quotas,
        "preferredPaperIds": list(DEFAULT_PAPER_IDS),
        "excludedPreviousPackRows": True,
        "maxPerPaperTypePage": 2,
        "wholeImagePolicy": "deferred_to_visual_full_image_annotation_pack_design",
        "attachmentPolicy": "context_crop_png_first_no_whole_image",
        "rationale": (
            "Expand beyond the first AlexNet/ResNet caption/table/equation calibration by probing "
            "image regions and new CLIP/MAE visual candidates while keeping the batch small enough for manual web/VLM review."
        ),
    }


def _counts(
    *,
    source_candidate_rows: int,
    previous_candidate_rows: int,
    eligible_rows: int,
    selected_rows: Sequence[dict[str, Any]],
    private_path_leak_rows: int,
) -> dict[str, int]:
    by_type = Counter(row.get("candidateType") for row in selected_rows)
    return {
        "sourceCandidateRows": int(source_candidate_rows),
        "previousAnnotatedRows": int(previous_candidate_rows),
        "eligibleExpansionRows": int(eligible_rows),
        "selectedExpansionRows": len(selected_rows),
        "selectedPaperRows": len({row.get("paperId") for row in selected_rows}),
        "imageCandidateRows": int(by_type.get("image_region", 0)),
        "figureCandidateRows": int(by_type.get("figure_caption_region", 0)),
        "tableCandidateRows": int(by_type.get("table_region", 0)),
        "equationCandidateRows": int(by_type.get("equation_region", 0)),
        "layoutCandidateRows": int(by_type.get("layout_region", 0)),
        "wholeImageRows": 0,
        "pageImageRows": 0,
        "blockedRows": sum(1 for row in selected_rows if row.get("blockerReason")),
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": 0,
    }


def build_visual_annotation_expansion_pack_design(
    candidate_report: dict[str, Any],
    web_pack: dict[str, Any],
    dry_run_report: dict[str, Any],
    *,
    pack_id: str = DEFAULT_EXPANSION_PACK_ID,
    source_candidate_report_ref: str = "eval/knowledgeos/reports/visual_layout_candidate_list_report.v1.json",
    source_web_pack_ref: str = "eval/knowledgeos/reports/visual_annotation_web_pack_001.v1.json",
    source_dry_run_report_ref: str = "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_dry_run.v1.json",
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    type_quotas: dict[str, int] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    quotas = dict(type_quotas or DEFAULT_TYPE_QUOTAS)
    source_rows = [
        row for row in list(candidate_report.get("candidateRowsDetail") or []) if isinstance(row, dict)
    ]
    previous_ids = _previous_candidate_ids(web_pack, dry_run_report)
    eligible = _eligible_rows(
        source_rows,
        previous_candidate_ids=previous_ids,
        included_types=set(quotas),
        preferred_paper_ids=DEFAULT_PAPER_IDS,
    )
    selected = select_expansion_candidates(
        source_rows,
        previous_candidate_ids=previous_ids,
        max_candidates=max_candidates,
        type_quotas=quotas,
        preferred_paper_ids=DEFAULT_PAPER_IDS,
    )
    pack_rows = [
        _pack_row(pack_id=pack_id, source_row=row, priority=index + 1)
        for index, row in enumerate(selected)
    ]
    report: dict[str, Any] = {
        "schema": VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "packId": pack_id,
        "sourceCandidateReport": {
            "schema": normalize_text(candidate_report.get("schema")),
            "status": normalize_text(candidate_report.get("status")),
            "reportRef": normalize_text(source_candidate_report_ref),
            "candidateRows": len(source_rows),
        },
        "sourceWebPack": {
            "schema": normalize_text(web_pack.get("schema")),
            "status": normalize_text(web_pack.get("status")),
            "reportRef": normalize_text(source_web_pack_ref),
            "packRows": len(list(web_pack.get("packRowsDetail") or [])),
        },
        "sourceDryRunReport": {
            "schema": normalize_text(dry_run_report.get("schema")),
            "status": normalize_text(dry_run_report.get("status")),
            "reportRef": normalize_text(source_dry_run_report_ref),
            "dryRunRows": len(list(dry_run_report.get("dryRunRowsDetail") or [])),
        },
        "scope": _scope(len(pack_rows)),
        "selectionPolicy": _selection_policy(quotas, max_candidates),
        "counts": {},
        "packRowsDetail": pack_rows,
        "warnings": [
            "This expansion design writes no crop files and sends no images to GPT/VLM.",
            "Image-region candidates are included only as bounded context-crop annotation candidates.",
            "Whole-image or whole-page annotation remains deferred to visual_full_image_annotation_pack_design.",
            "Any future derivedTextForRetrieval remains retrieval_hint_only and non-evidence.",
        ],
    }
    private_path_leak_rows = 1 if _contains_private_path(report) else 0
    report["counts"] = _counts(
        source_candidate_rows=len(source_rows),
        previous_candidate_rows=len(previous_ids),
        eligible_rows=len(eligible),
        selected_rows=pack_rows,
        private_path_leak_rows=private_path_leak_rows,
    )
    if (
        candidate_report.get("schema") != VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID
        or candidate_report.get("status") != "ready"
        or web_pack.get("schema") != VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID
        or web_pack.get("status") != "ready"
        or dry_run_report.get("schema") != VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID
        or dry_run_report.get("status") != "ready"
        or not pack_rows
        or private_path_leak_rows
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    lines = [
        "# Visual Annotation Expansion Pack Design",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- packId: `{report.get('packId')}`",
        f"- selectedExpansionRows: `{counts.get('selectedExpansionRows')}`",
        f"- imageCandidateRows: `{counts.get('imageCandidateRows')}`",
        f"- figureCandidateRows: `{counts.get('figureCandidateRows')}`",
        f"- tableCandidateRows: `{counts.get('tableCandidateRows')}`",
        f"- equationCandidateRows: `{counts.get('equationCandidateRows')}`",
        f"- wholeImageRows: `{counts.get('wholeImageRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- modelCalls: `{scope.get('modelCalls')}`",
        f"- webModelCalls: `{scope.get('webModelCalls')}`",
        f"- wholeImageGptRows: `{scope.get('wholeImageGptRows')}`",
        f"- cropWriteRows: `{scope.get('cropWriteRows')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- candidateStoreMutationRows: `{scope.get('candidateStoreMutationRows')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        f"- vaultScanRows: `{scope.get('vaultScanRows')}`",
        f"- externalDownloadRows: `{scope.get('externalDownloadRows')}`",
        "",
        "## Expansion Rows",
        "",
        "| # | paperId | type | page | sourceCandidateId | reason | attachment |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for row in report.get("packRowsDetail", []):
        guidance = dict(dict(row.get("webInput") or {}).get("attachmentGuidance") or {})
        lines.append(
            "| {priority} | {paperId} | {candidateType} | {page} | `{candidateId}` | `{reason}` | `{attachment}` |".format(
                priority=row.get("priority"),
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                page=row.get("page"),
                candidateId=row.get("sourceCandidateId"),
                reason=row.get("expansionReason"),
                attachment=guidance.get("recommendedAttachmentKind"),
            )
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_visual_annotation_expansion_pack_design(
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
    "DEFAULT_EXPANSION_PACK_ID",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_PAPER_IDS",
    "DEFAULT_TYPE_QUOTAS",
    "NEXT_RECOMMENDED_TRANCHE",
    "READY_DECISION",
    "VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID",
    "VISUAL_ANNOTATION_EXPANSION_PACK_ROW_SCHEMA_ID",
    "build_visual_annotation_expansion_pack_design",
    "load_json",
    "render_markdown_report",
    "sanitized_report_ref",
    "select_expansion_candidates",
    "write_visual_annotation_expansion_pack_design",
]
