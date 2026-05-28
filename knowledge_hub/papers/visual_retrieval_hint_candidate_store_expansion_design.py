"""Design a quarantined candidate store for visual retrieval hints.

This report-only helper consumes validated manual web/VLM visual annotations
and projects them into a future store contract. It does not write a candidate
store, index vectors, promote evidence, or expose hints at answer runtime.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.visual_annotation_expansion_manual_output_capture import (
    VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_VALIDATION_SCHEMA_ID,
)


VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-design.v1"
)
VISUAL_RETRIEVAL_HINT_CANDIDATE_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-row.v1"
)

READY_DECISION = "ready_for_visual_retrieval_hint_candidate_store_expansion_dry_run"
NEXT_RECOMMENDED_TRANCHE = "visual_retrieval_hint_candidate_store_expansion_dry_run"
PLANNED_STORE_REF = "papers_dir/visual_retrieval_hints/visual_retrieval_hint_candidates.v1.jsonl"

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


def _scope(candidate_rows: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "candidateRowsProjected": int(candidate_rows),
        "candidateStoreMutationRows": 0,
        "vectorIndexing": False,
        "indexEligibleRows": 0,
        "strictEvidencePromotionRows": 0,
        "runtimeAnswerVisibleExposureRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "answerabilityGateBypassRows": 0,
        "cropWriteRows": 0,
        "pageImageWriteRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
    }


def _policy() -> dict[str, Any]:
    return {
        "allowedUse": "retrieval_hint_only",
        "targetDerivedTextField": "derivedTextForRetrieval",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "runtimeVisible": False,
        "indexEligible": False,
        "requiresFutureIndexEligibilityGate": True,
        "answerabilityGateBypassAllowed": False,
    }


def _store_contract() -> dict[str, Any]:
    return {
        "storeName": "visual_retrieval_hint_candidate_store",
        "storeKind": "quarantined_candidate_store",
        "plannedStoreRef": PLANNED_STORE_REF,
        "writeMode": "future_explicit_apply_only",
        "intendedConsumer": "future_retrieval_query_expansion_or_rerank_hint",
        "identityFields": [
            "hintCandidateId",
            "sourceCandidateId",
            "paperId",
            "sourceContentHash",
            "page",
            "bbox",
        ],
        "requiredSourceLinks": [
            "sourceValidationReportRef",
            "sourceCandidateId",
            "sourceContentHash",
            "paperRef",
            "page",
            "bbox",
        ],
        "nonEvidencePolicyFields": [
            "allowedUse",
            "strictEvidence",
            "citationGrade",
            "answerableWithoutTextEvidence",
            "runtimeVisible",
            "indexEligible",
        ],
        "blockedUntil": [
            "candidate_store_dry_run",
            "human_review_of_storage_contract",
            "separate_index_eligibility_gate",
        ],
    }


def _full_image_escalation_policy() -> dict[str, Any]:
    return {
        "currentTrancheSendsWholeImagesToGpt": False,
        "earliestRecommendedTranche": "visual_full_image_annotation_pack_design",
        "requiresBeforeFullImageGpt": [
            "visual_retrieval_hint_candidate_store_expansion_design",
            "visual_retrieval_hint_candidate_store_expansion_dry_run",
            "visual_annotation_expansion_pack_design",
        ],
        "defaultBatchPolicy": "small_batches_not_all_candidates",
        "recommendedMaxRowsPerBatch": 24,
        "allowedImageInputsBeforeScaleUp": [
            "context_crop_png",
            "page_crop_png_when_context_crop_is_insufficient",
        ],
        "wholePageOrWholeImageUse": (
            "Only after a separate pack-design gate, and only for candidates where context crops are "
            "insufficient, uncertainty is high, or image-region candidates need object/layout inspection."
        ),
        "neverUseAsEvidence": True,
    }


def _candidate_row(row: dict[str, Any], *, source_validation_report_ref: str) -> dict[str, Any]:
    basis = "|".join(
        [
            normalize_text(row.get("sourceCandidateId")),
            normalize_text(row.get("sourceContentHash")),
            str(row.get("page") or ""),
            json.dumps(row.get("bbox") or [], ensure_ascii=True, sort_keys=True),
            normalize_text(row.get("derivedTextForRetrieval")),
        ]
    )
    hint_candidate_id = "visual-retrieval-hint:{paper}:{kind}:{page}:{digest}".format(
        paper=_slug(normalize_text(row.get("paperId"))),
        kind=_slug(normalize_text(row.get("candidateType"))),
        page=int(row.get("page") or 0),
        digest=_short_hash(basis),
    )
    policy = _policy()
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_ROW_SCHEMA_ID,
        "hintCandidateId": hint_candidate_id,
        "sourceCandidateId": normalize_text(row.get("sourceCandidateId")),
        "paperId": normalize_text(row.get("paperId")),
        "paperRef": normalize_text(row.get("paperRef")),
        "sourceContentHash": normalize_text(row.get("sourceContentHash")),
        "page": int(row.get("page") or 0),
        "bbox": list(row.get("bbox") or []),
        "candidateType": normalize_text(row.get("candidateType")),
        "sourceAttachmentRef": normalize_text(row.get("attachmentRef")),
        "derivedTextForRetrieval": normalize_text(row.get("derivedTextForRetrieval")),
        "visibleText": normalize_text(row.get("visibleText")),
        "retrievalKeywords": [
            normalize_text(item) for item in list(row.get("retrievalKeywords") or []) if normalize_text(item)
        ],
        "uncertainty": normalize_text(row.get("uncertainty")),
        "limitations": normalize_text(row.get("limitations")),
        "policy": policy,
        "storeProjection": {
            "plannedStoreRef": PLANNED_STORE_REF,
            "writeStatus": "not_written_design_only",
            "indexStatus": "not_indexed",
            "runtimeVisibility": "not_runtime_visible",
        },
        "provenance": {
            "sourceValidationReportSchema": VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_VALIDATION_SCHEMA_ID,
            "sourceValidationReportRef": normalize_text(source_validation_report_ref),
            "sourceCandidateId": normalize_text(row.get("sourceCandidateId")),
            "sourceContentHash": normalize_text(row.get("sourceContentHash")),
            "page": int(row.get("page") or 0),
            "bbox": list(row.get("bbox") or []),
            "extractionMethod": "visual_annotation_expansion_manual_output_capture_to_candidate_store_design_v1",
        },
        "blockerReason": "",
    }


def _candidate_rows(validation_report: dict[str, Any], *, source_validation_report_ref: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in list(validation_report.get("capturedRowsDetail") or [])
        if isinstance(row, dict)
        and not row.get("validation", {}).get("violationReasons")
        and row.get("validation", {}).get("policyCompliant") is True
    ]
    return [
        _candidate_row(row, source_validation_report_ref=source_validation_report_ref)
        for row in rows
    ]


def _counts(
    *,
    source_validation_rows: int,
    candidate_rows: list[dict[str, Any]],
    private_path_leak_rows: int,
    schema_violation_count: int = 0,
) -> dict[str, int]:
    return {
        "sourceValidationRows": int(source_validation_rows),
        "candidateRows": len(candidate_rows),
        "eligibleRows": len(candidate_rows),
        "blockedRows": 0,
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": int(schema_violation_count),
    }


def build_visual_retrieval_hint_candidate_store_expansion_design(
    validation_report: dict[str, Any],
    *,
    source_validation_report_ref: str = "eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.validation.v1.json",
    generated_at: str | None = None,
) -> dict[str, Any]:
    rows = _candidate_rows(
        validation_report,
        source_validation_report_ref=source_validation_report_ref,
    )
    report: dict[str, Any] = {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DESIGN_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "sourceValidationReport": {
            "schema": normalize_text(validation_report.get("schema")),
            "status": normalize_text(validation_report.get("status")),
            "reportRef": normalize_text(source_validation_report_ref),
            "capturedRows": len(list(validation_report.get("capturedRowsDetail") or [])),
            "blockedRows": int(dict(validation_report.get("counts") or {}).get("blockedRows") or 0),
        },
        "scope": _scope(len(rows)),
        "storeContract": _store_contract(),
        "fullImageEscalationPolicy": _full_image_escalation_policy(),
        "counts": {},
        "candidateRowsDetail": rows,
        "warnings": [
            "This is a design report only; the planned candidate store is not written.",
            "All projected rows remain retrieval_hint_only, not evidence.",
            "Whole-image or whole-page GPT/VLM batches require a later pack-design gate.",
        ],
    }
    private_path_leak_rows = 1 if _contains_private_path(report) else 0
    source_counts = dict(validation_report.get("counts") or {})
    report["counts"] = _counts(
        source_validation_rows=int(source_counts.get("matchedRows") or 0),
        candidate_rows=rows,
        private_path_leak_rows=private_path_leak_rows,
    )
    if (
        validation_report.get("schema") != VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_VALIDATION_SCHEMA_ID
        or validation_report.get("status") != "ready"
        or source_counts.get("blockedRows")
        or not rows
        or private_path_leak_rows
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
        report["counts"]["blockedRows"] = max(1, int(source_counts.get("blockedRows") or 0))
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    source = dict(report.get("sourceValidationReport") or {})
    escalation = dict(report.get("fullImageEscalationPolicy") or {})
    lines = [
        "# Visual Retrieval Hint Candidate Store Expansion Design",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- sourceValidationReport: `{source.get('reportRef')}`",
        f"- sourceValidationRows: `{counts.get('sourceValidationRows')}`",
        f"- candidateRows: `{counts.get('candidateRows')}`",
        f"- indexEligibleRows: `{counts.get('indexEligibleRows')}`",
        f"- runtimeVisibleRows: `{counts.get('runtimeVisibleRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- apiCalls: `{scope.get('apiCalls')}`",
        f"- modelCalls: `{scope.get('modelCalls')}`",
        f"- webModelCalls: `{scope.get('webModelCalls')}`",
        f"- candidateStoreMutationRows: `{scope.get('candidateStoreMutationRows')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- indexEligibleRows: `{scope.get('indexEligibleRows')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        f"- databaseMutationRows: `{scope.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{scope.get('indexMutationRows')}`",
        f"- reindexOrReembedRows: `{scope.get('reindexOrReembedRows')}`",
        f"- vaultScanRows: `{scope.get('vaultScanRows')}`",
        f"- externalDownloadRows: `{scope.get('externalDownloadRows')}`",
        f"- answerabilityGateBypassRows: `{scope.get('answerabilityGateBypassRows')}`",
        "",
        "## Whole-Image Timing",
        "",
        f"- currentTrancheSendsWholeImagesToGpt: `{escalation.get('currentTrancheSendsWholeImagesToGpt')}`",
        f"- earliestRecommendedTranche: `{escalation.get('earliestRecommendedTranche')}`",
        f"- recommendedMaxRowsPerBatch: `{escalation.get('recommendedMaxRowsPerBatch')}`",
        f"- rule: `{escalation.get('wholePageOrWholeImageUse')}`",
        "",
        "## Candidate Rows",
        "",
        "| # | paperId | type | page | sourceCandidateId | indexEligible | runtimeVisible |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for index, row in enumerate(report.get("candidateRowsDetail", []), start=1):
        policy = dict(row.get("policy") or {})
        lines.append(
            "| {index} | {paperId} | {candidateType} | {page} | `{candidateId}` | `{indexEligible}` | `{runtimeVisible}` |".format(
                index=index,
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                page=row.get("page"),
                candidateId=row.get("sourceCandidateId"),
                indexEligible=policy.get("indexEligible"),
                runtimeVisible=policy.get("runtimeVisible"),
            )
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_visual_retrieval_hint_candidate_store_expansion_design(
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
    "PLANNED_STORE_REF",
    "READY_DECISION",
    "VISUAL_RETRIEVAL_HINT_CANDIDATE_ROW_SCHEMA_ID",
    "VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DESIGN_SCHEMA_ID",
    "build_visual_retrieval_hint_candidate_store_expansion_design",
    "load_json",
    "render_markdown_report",
    "sanitized_report_ref",
    "write_visual_retrieval_hint_candidate_store_expansion_design",
]
