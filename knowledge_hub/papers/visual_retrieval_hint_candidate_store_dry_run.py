"""Dry-run a future visual retrieval-hint candidate store write.

This helper consumes the candidate-store design report and previews the JSONL
records that a later explicit apply tranche could write. It never writes the
candidate store, indexes vectors, promotes evidence, or exposes hints at answer
runtime.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.visual_retrieval_hint_candidate_store_design import (
    PLANNED_STORE_REF,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_ROW_SCHEMA_ID,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DESIGN_SCHEMA_ID,
)


VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-store-dry-run.v1"
)
VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-store-dry-run-row.v1"
)

READY_DECISION = "ready_for_visual_annotation_expansion_pack_design"
NEXT_RECOMMENDED_TRANCHE = "visual_annotation_expansion_pack_design"

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


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _short_hash(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _scope(row_count: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "sourceDesignRows": int(row_count),
        "plannedCandidateStoreRows": int(row_count),
        "candidateStoreWriteRows": 0,
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


def _dry_run_policy() -> dict[str, Any]:
    return {
        "plannedStoreRef": PLANNED_STORE_REF,
        "actualStoreWrite": False,
        "jsonlWritePreviewOnly": True,
        "applyRequiredForStoreMutation": True,
        "idempotencyKeyFields": [
            "hintCandidateId",
            "sourceCandidateId",
            "sourceContentHash",
            "page",
            "bbox",
        ],
        "runtimeVisibilityAfterDryRun": "not_runtime_visible",
        "indexingAfterDryRun": "not_indexed",
        "allowedUse": "retrieval_hint_only",
    }


def _store_record_preview(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_ROW_SCHEMA_ID,
        "hintCandidateId": normalize_text(row.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(row.get("sourceCandidateId")),
        "paperId": normalize_text(row.get("paperId")),
        "paperRef": normalize_text(row.get("paperRef")),
        "sourceContentHash": normalize_text(row.get("sourceContentHash")),
        "page": int(row.get("page") or 0),
        "bbox": list(row.get("bbox") or []),
        "candidateType": normalize_text(row.get("candidateType")),
        "sourceAttachmentRef": normalize_text(row.get("sourceAttachmentRef")),
        "derivedTextForRetrieval": normalize_text(row.get("derivedTextForRetrieval")),
        "visibleText": normalize_text(row.get("visibleText")),
        "retrievalKeywords": [
            normalize_text(item) for item in list(row.get("retrievalKeywords") or []) if normalize_text(item)
        ],
        "uncertainty": normalize_text(row.get("uncertainty")),
        "limitations": normalize_text(row.get("limitations")),
        "policy": dict(row.get("policy") or {}),
        "provenance": dict(row.get("provenance") or {}),
    }


def _idempotency_key(record: dict[str, Any]) -> str:
    basis = "|".join(
        [
            normalize_text(record.get("hintCandidateId")),
            normalize_text(record.get("sourceCandidateId")),
            normalize_text(record.get("sourceContentHash")),
            str(record.get("page") or ""),
            json.dumps(record.get("bbox") or [], ensure_ascii=True, sort_keys=True),
        ]
    )
    return "visual-retrieval-hint-idempotency:" + _short_hash(basis, length=24)


def _row_policy_compliant(row: dict[str, Any]) -> bool:
    policy = dict(row.get("policy") or {})
    return (
        policy.get("allowedUse") == "retrieval_hint_only"
        and policy.get("strictEvidence") is False
        and policy.get("citationGrade") is False
        and policy.get("answerableWithoutTextEvidence") is False
        and policy.get("runtimeVisible") is False
        and policy.get("indexEligible") is False
        and policy.get("answerabilityGateBypassAllowed") is False
    )


def _dry_run_row(row: dict[str, Any], *, source_design_report_ref: str) -> dict[str, Any]:
    record = _store_record_preview(row)
    canonical_line = _canonical_json(record)
    line_hash = _sha256_text(canonical_line)
    dry_run_row_id = "visual-retrieval-hint-dry-run:" + _short_hash(
        "|".join([normalize_text(record.get("hintCandidateId")), line_hash]),
        length=20,
    )
    blocker_reasons: list[str] = []
    if not normalize_text(record.get("hintCandidateId")):
        blocker_reasons.append("missing_hint_candidate_id")
    if not normalize_text(record.get("sourceCandidateId")):
        blocker_reasons.append("missing_source_candidate_id")
    if not normalize_text(record.get("derivedTextForRetrieval")):
        blocker_reasons.append("missing_derived_text_for_retrieval")
    if not _row_policy_compliant(row):
        blocker_reasons.append("policy_not_quarantined")
    if _contains_private_path(record):
        blocker_reasons.append("private_path_leak")
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_ROW_SCHEMA_ID,
        "dryRunRowId": dry_run_row_id,
        "hintCandidateId": normalize_text(record.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(record.get("sourceCandidateId")),
        "paperId": normalize_text(record.get("paperId")),
        "paperRef": normalize_text(record.get("paperRef")),
        "sourceContentHash": normalize_text(record.get("sourceContentHash")),
        "page": int(record.get("page") or 0),
        "bbox": list(record.get("bbox") or []),
        "candidateType": normalize_text(record.get("candidateType")),
        "plannedStoreRef": PLANNED_STORE_REF,
        "idempotencyKey": _idempotency_key(record),
        "plannedJsonlRecordSha256": line_hash,
        "plannedJsonlRecordPreview": record,
        "dryRunResult": {
            "wouldWriteOnApply": not blocker_reasons,
            "actualStoreWrite": False,
            "jsonlSerializable": True,
            "policyCompliant": not blocker_reasons,
            "indexEligible": False,
            "runtimeVisible": False,
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
        },
        "provenance": {
            "sourceDesignReportSchema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DESIGN_SCHEMA_ID,
            "sourceDesignReportRef": normalize_text(source_design_report_ref),
            "sourceCandidateId": normalize_text(record.get("sourceCandidateId")),
            "sourceContentHash": normalize_text(record.get("sourceContentHash")),
            "page": int(record.get("page") or 0),
            "bbox": list(record.get("bbox") or []),
            "extractionMethod": "visual_retrieval_hint_candidate_store_dry_run_v1",
        },
        "blockerReason": ";".join(blocker_reasons),
    }


def _counts(rows: list[dict[str, Any]], *, source_design_rows: int, private_path_leak_rows: int) -> dict[str, int]:
    hint_ids = [normalize_text(row.get("hintCandidateId")) for row in rows]
    source_ids = [normalize_text(row.get("sourceCandidateId")) for row in rows]
    duplicate_hint_ids = {item for item, count in Counter(hint_ids).items() if item and count > 1}
    duplicate_source_ids = {item for item, count in Counter(source_ids).items() if item and count > 1}
    blocked_rows = sum(1 for row in rows if row.get("blockerReason"))
    return {
        "sourceDesignRows": int(source_design_rows),
        "dryRunRows": len(rows),
        "plannedWriteRows": sum(1 for row in rows if row.get("dryRunResult", {}).get("wouldWriteOnApply")),
        "candidateStoreWriteRows": 0,
        "jsonlSerializableRows": sum(1 for row in rows if row.get("dryRunResult", {}).get("jsonlSerializable")),
        "duplicateHintCandidateIdRows": len(duplicate_hint_ids),
        "duplicateSourceCandidateIdRows": len(duplicate_source_ids),
        "blockedRows": int(blocked_rows + len(duplicate_hint_ids) + len(duplicate_source_ids)),
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": 0,
    }


def build_visual_retrieval_hint_candidate_store_dry_run(
    design_report: dict[str, Any],
    *,
    source_design_report_ref: str = "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_design.v1.json",
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_rows = [
        row for row in list(design_report.get("candidateRowsDetail") or []) if isinstance(row, dict)
    ]
    rows = [
        _dry_run_row(row, source_design_report_ref=source_design_report_ref)
        for row in source_rows
    ]
    private_path_leak_rows = 1 if _contains_private_path(rows) else 0
    report: dict[str, Any] = {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "sourceDesignReport": {
            "schema": normalize_text(design_report.get("schema")),
            "status": normalize_text(design_report.get("status")),
            "reportRef": normalize_text(source_design_report_ref),
            "candidateRows": len(source_rows),
            "blockedRows": int(dict(design_report.get("counts") or {}).get("blockedRows") or 0),
        },
        "scope": _scope(len(rows)),
        "dryRunPolicy": _dry_run_policy(),
        "counts": {},
        "dryRunRowsDetail": rows,
        "warnings": [
            "This dry-run previews future JSONL records but writes no candidate store.",
            "All rows remain unindexed and not runtime-visible after dry-run.",
            "A separate apply tranche is required before any store mutation, and a later gate is required before indexing.",
        ],
    }
    report["counts"] = _counts(
        rows,
        source_design_rows=len(source_rows),
        private_path_leak_rows=private_path_leak_rows,
    )
    if (
        design_report.get("schema") != VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DESIGN_SCHEMA_ID
        or design_report.get("status") != "ready"
        or dict(design_report.get("counts") or {}).get("blockedRows")
        or not rows
        or report["counts"]["blockedRows"]
        or private_path_leak_rows
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    source = dict(report.get("sourceDesignReport") or {})
    lines = [
        "# Visual Retrieval Hint Candidate Store Dry Run",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- sourceDesignReport: `{source.get('reportRef')}`",
        f"- dryRunRows: `{counts.get('dryRunRows')}`",
        f"- plannedWriteRows: `{counts.get('plannedWriteRows')}`",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- indexEligibleRows: `{counts.get('indexEligibleRows')}`",
        f"- runtimeVisibleRows: `{counts.get('runtimeVisibleRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- candidateStoreWriteRows: `{scope.get('candidateStoreWriteRows')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- indexMutationRows: `{scope.get('indexMutationRows')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        f"- databaseMutationRows: `{scope.get('databaseMutationRows')}`",
        f"- reindexOrReembedRows: `{scope.get('reindexOrReembedRows')}`",
        f"- vaultScanRows: `{scope.get('vaultScanRows')}`",
        f"- externalDownloadRows: `{scope.get('externalDownloadRows')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        f"- answerabilityGateBypassRows: `{scope.get('answerabilityGateBypassRows')}`",
        "",
        "## Dry Run Rows",
        "",
        "| # | paperId | type | page | hintCandidateId | wouldWriteOnApply | recordSha256 |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for index, row in enumerate(report.get("dryRunRowsDetail", []), start=1):
        lines.append(
            "| {index} | {paperId} | {candidateType} | {page} | `{hintCandidateId}` | `{wouldWrite}` | `{sha}` |".format(
                index=index,
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                page=row.get("page"),
                hintCandidateId=row.get("hintCandidateId"),
                wouldWrite=dict(row.get("dryRunResult") or {}).get("wouldWriteOnApply"),
                sha=row.get("plannedJsonlRecordSha256"),
            )
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_visual_retrieval_hint_candidate_store_dry_run(
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
    "VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_ROW_SCHEMA_ID",
    "VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID",
    "build_visual_retrieval_hint_candidate_store_dry_run",
    "load_json",
    "render_markdown_report",
    "sanitized_report_ref",
    "write_visual_retrieval_hint_candidate_store_dry_run",
]
