"""Human/product decision record for visual retrieval-hint expansion rows.

This report-only helper consumes the expansion review report and creates a
default hold-only decision record. If an explicit decision file is supplied, it
validates those decisions without applying them. It never writes the candidate
store, indexes vectors, promotes evidence, or exposes hints at answer runtime.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_design import (
    PLANNED_STORE_REF,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_review import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_SCHEMA_ID,
)


VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-human-product-decision-record.v1"
)
VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-human-product-decision-row.v1"
)
VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_FILE_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-human-product-decision-file.v1"
)

DEFAULT_DECISION = "hold_pending_human_product_review"
HUMAN_DECISIONS = (
    "approve_store_candidate_only",
    "hold_pending_more_context",
    "reject_visual_hint_candidate",
    "request_recrop_or_reannotation",
)
ALLOWED_DECISIONS = (DEFAULT_DECISION, *HUMAN_DECISIONS)

READY_TEMPLATE_DECISION = "manual_human_product_decisions_required"
READY_APPLY_DESIGN_DECISION = "ready_for_visual_retrieval_hint_candidate_store_expansion_apply_design"
NEXT_TEMPLATE_TRANCHE = "manual_edit_visual_retrieval_hint_candidate_store_expansion_decision_record"
NEXT_APPLY_DESIGN_TRANCHE = "visual_retrieval_hint_candidate_store_expansion_apply_design"

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


def _short_hash(value: str, *, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _scope(row_count: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "sourceReviewRows": int(row_count),
        "decisionRowsProjected": int(row_count),
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


def _decision_policy() -> dict[str, Any]:
    return {
        "plannedStoreRef": PLANNED_STORE_REF,
        "defaultDecision": DEFAULT_DECISION,
        "allowedHumanDecisions": list(HUMAN_DECISIONS),
        "applyAllowedByThisReport": False,
        "actualApprovalByThisReport": False,
        "requiresReviewerForHumanDecision": True,
        "requiresNotesForApproveDecision": True,
        "approvalMeaning": (
            "approve_store_candidate_only only marks a row as eligible for a later apply-design "
            "tranche; it does not write the candidate store or make the row index/runtime eligible."
        ),
    }


def _decision_file_template(review_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_FILE_SCHEMA_ID,
        "instructions": [
            "Edit a copy of this template for human/product decisions.",
            "Leave decision=hold_pending_human_product_review until a real decision is made.",
            "Human decisions require reviewer; approve_store_candidate_only also requires notes.",
            "This file does not authorize candidate-store writes, indexing, evidence promotion, or answer visibility.",
        ],
        "allowedHumanDecisions": list(HUMAN_DECISIONS),
        "decisions": [
            {
                "sourceReviewRowId": normalize_text(row.get("reviewRowId")),
                "hintCandidateId": normalize_text(row.get("hintCandidateId")),
                "sourceCandidateId": normalize_text(row.get("sourceCandidateId")),
                "paperId": normalize_text(row.get("paperId")),
                "candidateType": normalize_text(row.get("candidateType")),
                "decision": DEFAULT_DECISION,
                "reviewer": "",
                "notes": "",
            }
            for row in review_rows
        ],
    }


def _decision_rows(decision_file: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not decision_file:
        return []
    rows = decision_file.get("decisions")
    if rows is None:
        rows = decision_file.get("decisionRows")
    return [dict(item) for item in list(rows or []) if isinstance(item, dict)]


def _decision_key(item: dict[str, Any]) -> str:
    return normalize_text(item.get("hintCandidateId") or item.get("hint_candidate_id"))


def _submitted_decision_map(decision_file: dict[str, Any] | None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    mapped: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    seen: set[str] = set()
    for item in _decision_rows(decision_file):
        key = _decision_key(item)
        if not key:
            errors.append("decision_file_missing_hint_candidate_id")
            continue
        if key in seen:
            errors.append("decision_file_duplicate_hint_candidate_id")
        seen.add(key)
        mapped.setdefault(key, dict(item))
    return mapped, list(dict.fromkeys(errors))


def _source_review_by_hint_id(review_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {normalize_text(row.get("hintCandidateId")): row for row in review_rows}


def _unknown_decision_rows(
    decision_file: dict[str, Any] | None,
    review_rows: list[dict[str, Any]],
) -> list[str]:
    known = set(_source_review_by_hint_id(review_rows))
    return [
        _decision_key(item)
        for item in _decision_rows(decision_file)
        if _decision_key(item) and _decision_key(item) not in known
    ]


def _review_row_safe(row: dict[str, Any]) -> bool:
    checks = dict(row.get("checks") or {})
    return (
        row.get("reviewStatus") == "ready_for_human_product_review"
        and all(bool(value) for value in checks.values())
        and not normalize_text(row.get("blockerReason"))
    )


def _decision_value(submitted: dict[str, Any] | None) -> str:
    return normalize_text((submitted or {}).get("decision")) or DEFAULT_DECISION


def _decision_row(
    index: int,
    review_row: dict[str, Any],
    *,
    submitted: dict[str, Any] | None,
    source_review_report_ref: str,
) -> dict[str, Any]:
    decision = _decision_value(submitted)
    reviewer = normalize_text((submitted or {}).get("reviewer"))
    notes = normalize_text((submitted or {}).get("notes"))
    errors: list[str] = []
    if decision not in ALLOWED_DECISIONS:
        errors.append("decision_not_allowed")
    if decision in HUMAN_DECISIONS and not reviewer:
        errors.append("reviewer_required_for_human_decision")
    if decision == "approve_store_candidate_only" and not notes:
        errors.append("notes_required_for_approve_store_candidate_only")
    if decision == "approve_store_candidate_only" and not _review_row_safe(review_row):
        errors.append("source_review_row_not_safe_for_approval")
    if _contains_private_path(submitted or {}):
        errors.append("private_path_leak_in_decision")

    accepted_as_human_decision = decision in HUMAN_DECISIONS and not errors
    apply_design_candidate = decision == "approve_store_candidate_only" and accepted_as_human_decision
    if decision == DEFAULT_DECISION:
        decision_status = "default_hold_pending_manual_review"
    elif errors:
        decision_status = "invalid"
    else:
        decision_status = "human_decision_recorded"

    basis = "|".join(
        [
            normalize_text(review_row.get("reviewRowId")),
            normalize_text(review_row.get("hintCandidateId")),
            decision,
            reviewer,
            notes,
        ]
    )
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_ROW_SCHEMA_ID,
        "decisionRowId": "visual-retrieval-hint-expansion-decision:" + _short_hash(basis),
        "sourceReviewRowId": normalize_text(review_row.get("reviewRowId")),
        "hintCandidateId": normalize_text(review_row.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(review_row.get("sourceCandidateId")),
        "paperId": normalize_text(review_row.get("paperId")),
        "paperRef": normalize_text(review_row.get("paperRef")),
        "sourceContentHash": normalize_text(review_row.get("sourceContentHash")),
        "page": int(review_row.get("page") or 0),
        "bbox": list(review_row.get("bbox") or []),
        "candidateType": normalize_text(review_row.get("candidateType")),
        "plannedStoreRef": PLANNED_STORE_REF,
        "decision": decision,
        "decisionStatus": decision_status,
        "acceptedAsHumanDecision": accepted_as_human_decision,
        "reviewer": reviewer,
        "notes": notes,
        "applyDesignCandidate": apply_design_candidate,
        "candidateStoreWrite": False,
        "policySummary": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
            "applyAllowedByThisReport": False,
        },
        "provenance": {
            "sourceReviewReportSchema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_SCHEMA_ID,
            "sourceReviewReportRef": normalize_text(source_review_report_ref),
            "sourceReviewRowId": normalize_text(review_row.get("reviewRowId")),
            "hintCandidateId": normalize_text(review_row.get("hintCandidateId")),
            "sourceCandidateId": normalize_text(review_row.get("sourceCandidateId")),
            "sourceContentHash": normalize_text(review_row.get("sourceContentHash")),
            "page": int(review_row.get("page") or 0),
            "bbox": list(review_row.get("bbox") or []),
            "extractionMethod": "visual_retrieval_hint_candidate_store_expansion_human_product_decision_record_v1",
        },
        "blockerReason": ";".join(errors),
    }


def _counts(
    rows: list[dict[str, Any]],
    *,
    source_review_rows: int,
    file_errors: list[str],
    unknown_rows: list[str],
    private_path_leak_rows: int,
) -> dict[str, int]:
    by_decision = Counter(normalize_text(row.get("decision")) for row in rows)
    invalid_rows = sum(1 for row in rows if row.get("decisionStatus") == "invalid")
    default_hold_rows = by_decision.get(DEFAULT_DECISION, 0)
    human_decision_rows = sum(1 for row in rows if row.get("acceptedAsHumanDecision") is True)
    approved_rows = by_decision.get("approve_store_candidate_only", 0) - sum(
        1
        for row in rows
        if row.get("decision") == "approve_store_candidate_only" and row.get("decisionStatus") == "invalid"
    )
    return {
        "sourceReviewRows": int(source_review_rows),
        "decisionRows": len(rows),
        "templateRows": len(rows),
        "defaultHoldRows": int(default_hold_rows),
        "humanDecisionRows": int(human_decision_rows),
        "approvedRows": int(max(0, approved_rows)),
        "heldRows": by_decision.get("hold_pending_more_context", 0),
        "rejectedRows": by_decision.get("reject_visual_hint_candidate", 0),
        "recropRequestedRows": by_decision.get("request_recrop_or_reannotation", 0),
        "applyDesignCandidateRows": sum(1 for row in rows if row.get("applyDesignCandidate") is True),
        "candidateStoreWriteRows": 0,
        "invalidDecisionRows": int(invalid_rows),
        "missingDecisionRows": 0,
        "duplicateDecisionRows": 1 if "decision_file_duplicate_hint_candidate_id" in file_errors else 0,
        "unknownReviewRows": len([item for item in unknown_rows if item]),
        "policyViolationRows": int(invalid_rows + len(file_errors) + len(unknown_rows)),
        "blockedRows": int(invalid_rows + len(file_errors) + len(unknown_rows)),
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": 0,
    }


def build_visual_retrieval_hint_candidate_store_expansion_human_product_decision_record(
    review_report: dict[str, Any],
    *,
    decision_file: dict[str, Any] | None = None,
    source_review_report_ref: str = (
        "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_review.v1.json"
    ),
    decision_file_ref: str = "",
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_rows = [row for row in list(review_report.get("reviewRowsDetail") or []) if isinstance(row, dict)]
    submitted, file_errors = _submitted_decision_map(decision_file)
    unknown_rows = _unknown_decision_rows(decision_file, source_rows)
    rows = [
        _decision_row(
            index,
            review_row,
            submitted=submitted.get(normalize_text(review_row.get("hintCandidateId"))),
            source_review_report_ref=source_review_report_ref,
        )
        for index, review_row in enumerate(source_rows, start=1)
    ]
    private_path_leak_rows = 1 if _contains_private_path(rows) else 0
    review_counts = dict(review_report.get("counts") or {})
    counts = _counts(
        rows,
        source_review_rows=len(source_rows),
        file_errors=file_errors,
        unknown_rows=unknown_rows,
        private_path_leak_rows=private_path_leak_rows,
    )

    if counts["applyDesignCandidateRows"] and not counts["blockedRows"] and not private_path_leak_rows:
        status = "decision_record_validated"
        decision = READY_APPLY_DESIGN_DECISION
        next_tranche = NEXT_APPLY_DESIGN_TRANCHE
    else:
        status = "decision_record_template_ready"
        decision = READY_TEMPLATE_DECISION
        next_tranche = NEXT_TEMPLATE_TRANCHE

    report: dict[str, Any] = {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_RECORD_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": decision,
        "nextRecommendedTranche": next_tranche,
        "sourceReviewReport": {
            "schema": normalize_text(review_report.get("schema")),
            "status": normalize_text(review_report.get("status")),
            "reportRef": normalize_text(source_review_report_ref),
            "reviewRows": len(source_rows),
            "reviewReadyRows": int(review_counts.get("reviewReadyRows") or 0),
            "blockedRows": int(review_counts.get("blockedRows") or 0),
        },
        "decisionFile": {
            "schema": normalize_text((decision_file or {}).get("schema")),
            "reportRef": normalize_text(decision_file_ref),
            "provided": bool(decision_file),
            "decisionFileRows": len(_decision_rows(decision_file)),
            "fileErrors": [*file_errors, *[f"unknown_review_row:{item}" for item in unknown_rows if item]],
        },
        "scope": _scope(len(rows)),
        "decisionPolicy": _decision_policy(),
        "counts": counts,
        "decisionFileTemplate": _decision_file_template(source_rows),
        "decisionRowsDetail": rows,
        "warnings": [
            "This report is a decision-record template/validation only; it is not an apply step.",
            "The default generated record holds every row pending human/product review.",
            "Approved rows, if supplied later, only become apply-design candidates and remain unindexed, not runtime-visible, and non-evidence.",
        ],
    }

    if (
        review_report.get("schema") != VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_SCHEMA_ID
        or review_report.get("status") != "ready"
        or review_counts.get("blockedRows")
        or not rows
        or counts["blockedRows"]
        or private_path_leak_rows
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
        report["nextRecommendedTranche"] = NEXT_TEMPLATE_TRANCHE
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    source = dict(report.get("sourceReviewReport") or {})
    policy = dict(report.get("decisionPolicy") or {})
    lines = [
        "# Visual Retrieval Hint Candidate Store Expansion Human/Product Decision Record",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- sourceReviewReport: `{source.get('reportRef')}`",
        f"- decisionRows: `{counts.get('decisionRows')}`",
        f"- defaultHoldRows: `{counts.get('defaultHoldRows')}`",
        f"- humanDecisionRows: `{counts.get('humanDecisionRows')}`",
        f"- approvedRows: `{counts.get('approvedRows')}`",
        f"- applyDesignCandidateRows: `{counts.get('applyDesignCandidateRows')}`",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        "",
        "## Decision Boundary",
        "",
        f"- defaultDecision: `{policy.get('defaultDecision')}`",
        f"- applyAllowedByThisReport: `{policy.get('applyAllowedByThisReport')}`",
        f"- actualApprovalByThisReport: `{policy.get('actualApprovalByThisReport')}`",
        f"- plannedStoreRef: `{policy.get('plannedStoreRef')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- candidateStoreWriteRows: `{scope.get('candidateStoreWriteRows')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- indexMutationRows: `{scope.get('indexMutationRows')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        f"- answerabilityGateBypassRows: `{scope.get('answerabilityGateBypassRows')}`",
        "",
        "## Decision Rows",
        "",
        "| # | paperId | type | page | decision | accepted | applyDesignCandidate | hintCandidateId |",
        "|---:|---|---|---:|---|---|---|---|",
    ]
    for index, row in enumerate(report.get("decisionRowsDetail", []), start=1):
        lines.append(
            "| {index} | {paperId} | {candidateType} | {page} | `{decision}` | `{accepted}` | `{applyCandidate}` | `{hintCandidateId}` |".format(
                index=index,
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                page=row.get("page"),
                decision=row.get("decision"),
                accepted=row.get("acceptedAsHumanDecision"),
                applyCandidate=row.get("applyDesignCandidate"),
                hintCandidateId=row.get("hintCandidateId"),
            )
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_visual_retrieval_hint_candidate_store_expansion_human_product_decision_record(
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
    "ALLOWED_DECISIONS",
    "DEFAULT_DECISION",
    "HUMAN_DECISIONS",
    "NEXT_APPLY_DESIGN_TRANCHE",
    "NEXT_TEMPLATE_TRANCHE",
    "READY_APPLY_DESIGN_DECISION",
    "READY_TEMPLATE_DECISION",
    "VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_FILE_SCHEMA_ID",
    "VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_RECORD_SCHEMA_ID",
    "VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_ROW_SCHEMA_ID",
    "build_visual_retrieval_hint_candidate_store_expansion_human_product_decision_record",
    "load_json",
    "render_markdown_report",
    "sanitized_report_ref",
    "write_visual_retrieval_hint_candidate_store_expansion_human_product_decision_record",
]
