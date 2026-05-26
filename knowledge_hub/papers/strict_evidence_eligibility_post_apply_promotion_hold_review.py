"""Post-apply promotion hold review for StrictEvidence eligibility rows.

Consumes the eligibility apply readback review report and documents that the
eligibility set remains candidate-only before any citation-grade, runtime, parser,
answer, DB, or strictEligible mutation tranche. Report-only: writes only its own
reports.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_eligibility_executor_apply_readback_review import (
    EXPECTED_ELIGIBILITY_RECORD_ROWS,
    EXPECTED_INPUT_ROWS,
    EXPECTED_SOURCE_SPAN_STORE_ROWS,
    EXPECTED_STRICT_EVIDENCE_STORE_ROWS,
    READBACK_STATUS_VALIDATED,
    STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_APPLY_READBACK_REVIEW_SCHEMA_ID,
)


STRICT_EVIDENCE_ELIGIBILITY_POST_APPLY_PROMOTION_HOLD_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-eligibility-post-apply-promotion-hold-review.v1"
)

HOLD_STATUS_ACTIVE = "eligibility_post_apply_promotion_hold_active"
HOLD_STATUS_BLOCKED_READBACK_NOT_VALIDATED = "blocked_readback_not_validated"
HOLD_STATUS_BLOCKED_GATE_ENABLED = "blocked_downstream_gate_already_enabled"
HOLD_STATUS_BLOCKED_STORE_COUNT = "blocked_store_row_count_changed"
HOLD_STATUS_BLOCKED_RUNTIME_OR_CITATION = "blocked_runtime_or_citation_flag_violation"
HOLD_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

DEFAULT_READBACK_REVIEW_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-eligibility-executor-apply-readback-review"
    / "01-strict-evidence-eligibility-executor-apply-readback-review"
    / "strict-evidence-eligibility-executor-apply-readback-review.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-eligibility-post-apply-promotion-hold-review"
    / "01-strict-evidence-eligibility-post-apply-promotion-hold-review"
)

_WRITE_COUNT_FIELDS = (
    "eligibilityRecordWriteRows",
    "strictEvidenceWriteRows",
    "sourceSpanUpdatedRows",
    "strictEligibleMutationRows",
    "strictEvidenceCreatedRows",
    "citationGradeEvidenceCreatedRows",
    "runtimeEvidenceCreatedRows",
    "parserRoutingChangedRows",
    "answerIntegrationChangedRows",
    "databaseMutationRows",
    "reindexOrReembedRows",
    "manifestWriteRows",
)

_DOWNSTREAM_GATE_FIELDS = {
    "eligibilityRecordWrite": ("eligibilityRecordWriteAllowed", "eligibilityRecordWrite"),
    "strictEligibleMutation": ("strictEligibleMutationAllowed", "strictEligibleMutation"),
    "strictEvidenceStoreWrite": ("strictEvidenceStoreWriteAllowed", "strictEvidenceStoreWrite"),
    "sourceSpanStoreWrite": ("sourceSpanStoreWriteAllowed", "sourceSpanStoreWrite"),
    "citationGradeEvidence": (
        "citationReady",
        "citationGradeAllowed",
        "citationGradeEvidenceCreated",
    ),
    "runtimeEvidence": ("runtimeEvidenceReady", "runtimeEvidenceAllowed", "runtimeEvidenceCreated"),
    "parserRouting": ("parserRoutingReady", "parserRoutingAllowed", "parserRoutingChanged"),
    "answerIntegration": (
        "answerIntegrationReady",
        "answerIntegrationAllowed",
        "answerIntegrationChanged",
    ),
    "databaseMutation": ("runtimeMutationAllowed", "databaseMutationAllowed", "databaseMutation"),
    "reindexOrReembed": ("reindexOrReembedAllowed", "reindexOrReembed"),
    "manifestWrite": ("runManifestWriteAllowed", "manifestWriteAllowed", "manifestWrite"),
}

_ROW_ILLEGAL_BOOL_FIELDS = (
    "strictEligible",
    "strictEligibleMutationApplied",
    "strictEvidenceCreated",
    "citationGrade",
    "runtimeEvidence",
    "runtimeVisible",
    "parserRoutingChanged",
    "answerIntegrationChanged",
    "databaseMutation",
)

_ROW_ILLEGAL_COUNT_FIELDS = (
    "eligibilityRecordWriteRows",
    "strictEvidenceWriteRows",
    "sourceSpanUpdatedRows",
    "strictEligibleMutationRows",
)

_WRITE_MATRIX_ILLEGAL_FIELDS = (
    "eligibilityRecordWrite",
    "strictEvidenceStoreWrite",
    "sourceSpanStoreWrite",
    "strictEvidenceCreated",
    "strictEligibleMutation",
    "citationGradeEvidenceCreated",
    "runtimeEvidenceCreated",
    "parserRoutingChanged",
    "answerIntegrationChanged",
    "databaseMutation",
    "reindexOrReembed",
    "canonicalParsedArtifactsWritten",
    "manifestWrite",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = _safe_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(str(path)).expanduser()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _no_mutation_policy_matrix() -> dict[str, Any]:
    return {
        "reportOnly": True,
        "postApplyHoldReviewOnly": True,
        "eligibilityRecordWrite": False,
        "strictEvidenceStoreWrite": False,
        "sourceSpanStoreWrite": False,
        "strictEvidenceCreated": False,
        "strictEligibleMutation": False,
        "citationGradeEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "vaultScan": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
        "manifestWrite": False,
    }


def _future_promotion_readiness_checklist() -> list[dict[str, Any]]:
    return [
        {
            "id": "citation_grade_policy_gate",
            "title": "Citation-grade policy gate for eligibility records",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "no_answer_safety_eval_gate",
            "title": "No-answer safety eval before any runtime visibility",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "citation_grade_executor_dry_run",
            "title": "Citation-grade executor dry-run with zero runtime effect",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "rollback_strategy",
            "title": "Rollback strategy for eligibility/citation/runtime promotion records",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "strict_eligible_boolean_mutation_change_requires_new_decision",
            "title": "Any strictEligible boolean mutation semantics change requires a new ADR",
            "status": "blocked",
            "requiredBeforePromotion": True,
        },
        {
            "id": "runtime_binding_gate",
            "title": "Runtime binding gate after citation-grade validation",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "answer_integration_gate",
            "title": "Answer integration gate after runtime binding validation",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "post_promotion_store_readback",
            "title": "Full store readback after any promotion write",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "release_smoke_hygiene",
            "title": "Release smoke and public hygiene after any integration change",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
    ]


def _downstream_gate_matrix(readback_report: dict[str, Any]) -> dict[str, Any]:
    gate = readback_report.get("gate") if isinstance(readback_report.get("gate"), dict) else {}
    policy = readback_report.get("policy") if isinstance(readback_report.get("policy"), dict) else {}
    matrix: dict[str, Any] = {}
    for name, fields in _DOWNSTREAM_GATE_FIELDS.items():
        values = [_safe_bool(gate.get(field)) or _safe_bool(policy.get(field)) for field in fields]
        enabled = any(values)
        matrix[name] = {
            "allowed": enabled,
            "ready": enabled,
            "reason": (
                "blocked_until_explicit_post_apply_promotion_tranche"
                if not enabled
                else "input_report_downstream_gate_already_enabled"
            ),
            "holdActive": not enabled,
        }
    return matrix


def _downstream_gate_violations(gate_matrix: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for name, section in gate_matrix.items():
        if not isinstance(section, dict):
            continue
        if _safe_bool(section.get("allowed")) or _safe_bool(section.get("ready")):
            violations.append(f"downstream_gate_{name}_already_enabled")
    return violations


def _row_flag_violations(row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for field_name in _ROW_ILLEGAL_BOOL_FIELDS:
        if _safe_bool(row.get(field_name)):
            violations.append(f"{field_name}_true")
    for field_name in _ROW_ILLEGAL_COUNT_FIELDS:
        if _safe_int(row.get(field_name)) != 0:
            violations.append(f"{field_name}={_safe_int(row.get(field_name))}_expected_0")
    matrix = row.get("writeMatrix") if isinstance(row.get("writeMatrix"), dict) else {}
    for field_name in _WRITE_MATRIX_ILLEGAL_FIELDS:
        if _safe_bool(matrix.get(field_name)):
            violations.append(f"writeMatrix.{field_name}_true")
    return _dedupe(violations)


def _input_report_violations(
    *,
    readback_report: dict[str, Any],
    expected_input_rows: int,
    expected_eligibility_record_rows: int,
    expected_strict_evidence_store_rows: int,
    expected_source_span_store_rows: int,
) -> tuple[list[str], list[str], list[str]]:
    schema_violations: list[str] = []
    store_count_violations: list[str] = []
    write_count_violations: list[str] = []
    counts = readback_report.get("counts") if isinstance(readback_report.get("counts"), dict) else {}
    gate = readback_report.get("gate") if isinstance(readback_report.get("gate"), dict) else {}

    if not readback_report:
        schema_violations.append("readback_report_missing_or_unreadable")
        return schema_violations, store_count_violations, write_count_violations

    validation = validate_payload(
        readback_report,
        STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_APPLY_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        schema_violations.extend(str(error) for error in validation.errors)
    if _safe_text(readback_report.get("status")) != "ok":
        schema_violations.append(
            f"readback_report_status={_safe_text(readback_report.get('status')) or 'unknown'}"
        )
    if not _safe_bool(gate.get("readbackReviewReady")):
        schema_violations.append("readback_review_not_ready")
    if _safe_text(gate.get("decision")) != "strict_evidence_eligibility_executor_apply_readback_review_ready":
        schema_violations.append(f"readback_decision={_safe_text(gate.get('decision')) or 'unknown'}")

    expectations = {
        "inputRows": expected_input_rows,
        "eligibilityRecordRows": expected_eligibility_record_rows,
        "readbackValidatedRows": expected_eligibility_record_rows,
    }
    for field_name, expected in expectations.items():
        actual = _safe_int(counts.get(field_name))
        if actual != expected:
            schema_violations.append(f"{field_name}={actual}_expected_{expected}")

    store_expectations = {
        "strictEvidenceStoreRows": expected_strict_evidence_store_rows,
        "sourceSpanStoreRows": expected_source_span_store_rows,
    }
    for field_name, expected in store_expectations.items():
        actual = _safe_int(counts.get(field_name))
        if actual != expected:
            store_count_violations.append(f"{field_name}={actual}_expected_{expected}")

    for field_name in _WRITE_COUNT_FIELDS:
        actual = _safe_int(counts.get(field_name))
        if actual != 0:
            write_count_violations.append(f"{field_name}={actual}_expected_0")

    return _dedupe(schema_violations), _dedupe(store_count_violations), _dedupe(write_count_violations)


def _hold_rows(
    readback_rows: list[dict[str, Any]],
    *,
    schema_violations: list[str],
    store_count_violations: list[str],
    write_count_violations: list[str],
    downstream_gate_violations: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, readback_row in enumerate(readback_rows):
        source_row = dict(readback_row or {})
        hold_status = HOLD_STATUS_ACTIVE
        blockers: list[str] = []
        recommended_action = "post_apply_promotion_hold_active"

        row_violations = _row_flag_violations(source_row)
        if schema_violations:
            hold_status = HOLD_STATUS_BLOCKED_INPUT_SCHEMA
            blockers.extend(schema_violations)
            recommended_action = "blocked_input_schema_violation"
        elif store_count_violations:
            hold_status = HOLD_STATUS_BLOCKED_STORE_COUNT
            blockers.extend(store_count_violations)
            recommended_action = "blocked_store_row_count_changed"
        elif write_count_violations or downstream_gate_violations:
            hold_status = HOLD_STATUS_BLOCKED_GATE_ENABLED
            blockers.extend(write_count_violations)
            blockers.extend(downstream_gate_violations)
            recommended_action = "blocked_downstream_gate_already_enabled"
        elif (
            _safe_text(source_row.get("readback_status")) != READBACK_STATUS_VALIDATED
            or not _safe_bool(source_row.get("eligibilityReadbackValidated"))
        ):
            hold_status = HOLD_STATUS_BLOCKED_READBACK_NOT_VALIDATED
            blockers.append(f"readback_status={_safe_text(source_row.get('readback_status')) or 'unknown'}")
            recommended_action = "blocked_readback_not_validated"
        elif row_violations:
            hold_status = HOLD_STATUS_BLOCKED_RUNTIME_OR_CITATION
            blockers.extend(row_violations)
            recommended_action = "blocked_runtime_or_citation_flag_violation"

        hold_active = hold_status == HOLD_STATUS_ACTIVE
        rows.append(
            {
                "hold_row_id": f"strict-evidence-eligibility-post-apply-promotion-hold-review:{index:04d}",
                "readback_row_id": _safe_text(source_row.get("readback_row_id")),
                "apply_row_id": _safe_text(source_row.get("apply_row_id")),
                "dry_run_row_id": _safe_text(source_row.get("dry_run_row_id")),
                "decision_row_id": _safe_text(source_row.get("decision_row_id")),
                "strictEvidenceId": _safe_text(source_row.get("strictEvidenceId")),
                "sourceSpanId": _safe_text(source_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(source_row.get("candidateRecordId")),
                "eligibilityRecordId": _safe_text(source_row.get("eligibilityRecordId")),
                "paper_id": _safe_text(source_row.get("paper_id")),
                "artifact_type": _safe_text(source_row.get("artifact_type")),
                "eligibilityState": _safe_text(source_row.get("eligibilityState")),
                "eligibilityDecision": _safe_text(source_row.get("eligibilityDecision")),
                "readback_status": _safe_text(source_row.get("readback_status")),
                "hold_status": hold_status,
                "hold_blockers": _dedupe(blockers),
                "postApplyPromotionHoldActive": hold_active,
                "strictEligibleMutationAllowed": False,
                "eligibilityRecordWriteAllowed": False,
                "citationGradeAllowed": False,
                "runtimeEvidenceAllowed": False,
                "parserRoutingAllowed": False,
                "answerIntegrationAllowed": False,
                "databaseMutationAllowed": False,
                "reindexOrReembedAllowed": False,
                "strictEligible": False,
                "strictEvidenceCreated": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "recommended_action": recommended_action,
            }
        )
    return rows


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    readback_counts: dict[str, Any],
    aggregate_violations: list[str],
    gate_matrix: dict[str, Any],
) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("hold_status")) for row in rows)
    active_rows = [row for row in rows if row.get("hold_status") == HOLD_STATUS_ACTIVE]
    return {
        "inputRows": _safe_int(readback_counts.get("inputRows")),
        "eligibilityRecordRows": _safe_int(readback_counts.get("eligibilityRecordRows")),
        "readbackValidatedRows": _safe_int(readback_counts.get("readbackValidatedRows")),
        "holdActiveRows": int(by_status.get(HOLD_STATUS_ACTIVE, 0)),
        "sectionHoldRows": sum(1 for row in active_rows if _safe_text(row.get("artifact_type")) == "section"),
        "figureCaptionHoldRows": sum(1 for row in active_rows if _safe_text(row.get("artifact_type")) == "figure"),
        "blockedReadbackNotValidatedRows": int(by_status.get(HOLD_STATUS_BLOCKED_READBACK_NOT_VALIDATED, 0)),
        "blockedDownstreamGateAlreadyEnabledRows": int(by_status.get(HOLD_STATUS_BLOCKED_GATE_ENABLED, 0)),
        "blockedStoreRowCountChangedRows": int(by_status.get(HOLD_STATUS_BLOCKED_STORE_COUNT, 0)),
        "blockedRuntimeOrCitationFlagViolationRows": int(
            by_status.get(HOLD_STATUS_BLOCKED_RUNTIME_OR_CITATION, 0)
        ),
        "blockedInputSchemaViolationRows": int(by_status.get(HOLD_STATUS_BLOCKED_INPUT_SCHEMA, 0)),
        "strictEligibleMutationAllowedRows": sum(
            1 for row in rows if _safe_bool(row.get("strictEligibleMutationAllowed"))
        ),
        "eligibilityRecordWriteAllowedRows": sum(
            1 for row in rows if _safe_bool(row.get("eligibilityRecordWriteAllowed"))
        ),
        "citationGradeAllowedRows": sum(1 for row in rows if _safe_bool(row.get("citationGradeAllowed"))),
        "runtimeEvidenceAllowedRows": sum(
            1 for row in rows if _safe_bool(row.get("runtimeEvidenceAllowed"))
        ),
        "parserRoutingAllowedRows": sum(1 for row in rows if _safe_bool(row.get("parserRoutingAllowed"))),
        "answerIntegrationAllowedRows": sum(
            1 for row in rows if _safe_bool(row.get("answerIntegrationAllowed"))
        ),
        "strictEvidenceStoreRows": _safe_int(readback_counts.get("strictEvidenceStoreRows")),
        "sourceSpanStoreRows": _safe_int(readback_counts.get("sourceSpanStoreRows")),
        "eligibilityRecordWriteRows": 0,
        "strictEvidenceWriteRows": 0,
        "strictEvidenceCreatedRows": 0,
        "sourceSpanUpdatedRows": 0,
        "strictEligibleMutationRows": 0,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "manifestWriteRows": 0,
        "reindexOrReembedRows": 0,
        "schemaViolationCount": len(aggregate_violations),
        "downstreamGateAllowedCount": sum(
            1
            for section in gate_matrix.values()
            if isinstance(section, dict)
            and (_safe_bool(section.get("allowed")) or _safe_bool(section.get("ready")))
        ),
        "byPaperId": dict(Counter(_safe_text(row.get("paper_id")) for row in active_rows)),
        "byArtifactType": dict(Counter(_safe_text(row.get("artifact_type")) for row in active_rows)),
        "byHoldStatus": dict(by_status),
        "byRecommendedAction": dict(Counter(_safe_text(row.get("recommended_action")) for row in rows)),
    }


def build_strict_evidence_eligibility_post_apply_promotion_hold_review(
    *,
    readback_review_report_path: str | Path = DEFAULT_READBACK_REVIEW_REPORT_PATH,
    paper_ids: list[str] | None = None,
    expected_input_rows: int = EXPECTED_INPUT_ROWS,
    expected_eligibility_record_rows: int = EXPECTED_ELIGIBILITY_RECORD_ROWS,
    expected_strict_evidence_store_rows: int = EXPECTED_STRICT_EVIDENCE_STORE_ROWS,
    expected_source_span_store_rows: int = EXPECTED_SOURCE_SPAN_STORE_ROWS,
) -> dict[str, Any]:
    report_path = Path(str(readback_review_report_path)).expanduser()
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}
    warnings: list[str] = []

    readback_report = _read_json(report_path)
    schema_violations, store_count_violations, write_count_violations = _input_report_violations(
        readback_report=readback_report,
        expected_input_rows=expected_input_rows,
        expected_eligibility_record_rows=expected_eligibility_record_rows,
        expected_strict_evidence_store_rows=expected_strict_evidence_store_rows,
        expected_source_span_store_rows=expected_source_span_store_rows,
    )
    gate_matrix = _downstream_gate_matrix(readback_report) if readback_report else {}
    downstream_violations = _downstream_gate_violations(gate_matrix)

    readback_rows = [
        row for row in readback_report.get("rows", []) if isinstance(row, dict)
    ] if readback_report else []
    if requested_papers:
        found = {_safe_text(row.get("paper_id")) for row in readback_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found:
            warnings.append("requested_paper_ids_not_found_in_readback_report")
        readback_rows = [
            row for row in readback_rows if _safe_text(row.get("paper_id")) in requested_papers
        ]

    if not readback_rows and not schema_violations:
        schema_violations.append("readback_rows_missing")

    aggregate_violations = _dedupe(
        schema_violations + store_count_violations + write_count_violations + downstream_violations
    )
    rows = _hold_rows(
        readback_rows,
        schema_violations=schema_violations,
        store_count_violations=store_count_violations,
        write_count_violations=write_count_violations,
        downstream_gate_violations=downstream_violations,
    )
    readback_counts = (
        readback_report.get("counts") if isinstance(readback_report.get("counts"), dict) else {}
    )
    counts = _count_rows(
        rows=rows,
        readback_counts=readback_counts,
        aggregate_violations=aggregate_violations,
        gate_matrix=gate_matrix,
    )

    hold_active_rows = int(counts.get("holdActiveRows") or 0)
    status = "ok"
    if (
        aggregate_violations
        or not rows
        or hold_active_rows != expected_eligibility_record_rows
        or hold_active_rows != len(rows)
        or int(counts.get("downstreamGateAllowedCount") or 0) > 0
    ):
        status = "blocked"

    policy_matrix = _no_mutation_policy_matrix()
    return {
        "schema": STRICT_EVIDENCE_ELIGIBILITY_POST_APPLY_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "readbackReviewReportPath": str(report_path),
            "readbackReviewReportSchema": _safe_text(readback_report.get("schema")) if readback_report else "",
            "readbackReviewReportStatus": _safe_text(readback_report.get("status")) if readback_report else "",
            "readbackReviewDecision": _safe_text((readback_report.get("gate") or {}).get("decision"))
            if readback_report
            else "",
            "requestedPaperIds": sorted(requested_papers),
            "expectedInputRows": expected_input_rows,
            "expectedEligibilityRecordRows": expected_eligibility_record_rows,
            "expectedStrictEvidenceStoreRows": expected_strict_evidence_store_rows,
            "expectedSourceSpanStoreRows": expected_source_span_store_rows,
        },
        "counts": counts,
        "blockedDownstreamGateMatrix": gate_matrix,
        "futurePromotionReadinessChecklist": _future_promotion_readiness_checklist(),
        "noMutationPolicyMatrix": policy_matrix,
        "gate": {
            "eligibilityPostApplyPromotionHoldReviewReady": status == "ok",
            "holdDecision": (
                "strict_evidence_eligibility_post_apply_promotion_hold_active"
                if status == "ok"
                else "strict_evidence_eligibility_post_apply_promotion_hold_blocked"
            ),
            "eligibilityRecordWriteAllowed": False,
            "strictEligibleMutationAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "sourceSpanStoreWriteAllowed": False,
            "citationGradeAllowed": False,
            "runtimeEvidenceAllowed": False,
            "parserRoutingAllowed": False,
            "answerIntegrationAllowed": False,
            "databaseMutationAllowed": False,
            "reindexOrReembedAllowed": False,
            "runManifestWriteAllowed": False,
            "schemaViolations": aggregate_violations,
            "recommendedNextTranche": (
                "strict_evidence_citation_grade_policy_gate_design"
                if status == "ok"
                else "strict_evidence_eligibility_executor_apply_readback_repair"
            ),
        },
        "policy": policy_matrix,
        "warnings": _dedupe(warnings),
        "rows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "input",
            "counts",
            "blockedDownstreamGateMatrix",
            "futurePromotionReadinessChecklist",
            "noMutationPolicyMatrix",
            "gate",
            "policy",
            "warnings",
        )
        if key in report
    }


def render_strict_evidence_eligibility_post_apply_promotion_hold_review_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byHoldStatus") or {})).items())
    ]
    matrix_lines = [
        f"- {name}: allowed={json.dumps(section.get('allowed'))}, holdActive={json.dumps(section.get('holdActive'))}"
        for name, section in sorted((dict(report.get("blockedDownstreamGateMatrix") or {})).items())
        if isinstance(section, dict)
    ]
    checklist_lines = [
        f"- [{item.get('id', '')}] {item.get('title', '')} (status={item.get('status', '')})"
        for item in list(report.get("futurePromotionReadinessChecklist") or [])
    ]
    return "\n".join(
        [
            "# Strict Evidence Eligibility Post-Apply Promotion Hold Review",
            "",
            f"- status: {report.get('status', '')}",
            f"- hold decision: {gate.get('holdDecision', '')}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- eligibility record rows: {int(counts.get('eligibilityRecordRows') or 0)}",
            f"- readback validated rows: {int(counts.get('readbackValidatedRows') or 0)}",
            f"- hold active rows: {int(counts.get('holdActiveRows') or 0)}",
            f"- section hold rows: {int(counts.get('sectionHoldRows') or 0)}",
            f"- figure caption hold rows: {int(counts.get('figureCaptionHoldRows') or 0)}",
            f"- strict evidence store rows: {int(counts.get('strictEvidenceStoreRows') or 0)}",
            f"- source span store rows: {int(counts.get('sourceSpanStoreRows') or 0)}",
            "",
            "## Blocked downstream gate matrix",
            *matrix_lines,
            "",
            "## Future promotion readiness checklist",
            *checklist_lines,
            "",
            "## Hold status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- recommended next tranche: {gate.get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_eligibility_post_apply_promotion_hold_review_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-eligibility-post-apply-promotion-hold-review.json"
    summary_path = root / "strict-evidence-eligibility-post-apply-promotion-hold-review-summary.json"
    markdown_path = root / "strict-evidence-eligibility-post-apply-promotion-hold-review.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_eligibility_post_apply_promotion_hold_review_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Review post-apply promotion hold state for StrictEvidence eligibility rows "
            "without enabling citation, runtime, parser, answer, DB, or strictEligible mutation."
        )
    )
    parser.add_argument(
        "--readback-report",
        default=str(DEFAULT_READBACK_REVIEW_REPORT_PATH),
        help="Path to eligibility executor apply readback review JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, summary, and markdown reports.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_eligibility_post_apply_promotion_hold_review(
        readback_review_report_path=args.readback_report,
        paper_ids=args.paper_id or None,
    )
    paths = write_strict_evidence_eligibility_post_apply_promotion_hold_review_reports(
        report,
        args.output_dir,
    )
    print(f"wrote report: {paths['report']}")
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote markdown: {paths['markdown']}")
    if args.json:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_READBACK_REVIEW_REPORT_PATH",
    "HOLD_STATUS_ACTIVE",
    "HOLD_STATUS_BLOCKED_GATE_ENABLED",
    "HOLD_STATUS_BLOCKED_INPUT_SCHEMA",
    "HOLD_STATUS_BLOCKED_READBACK_NOT_VALIDATED",
    "HOLD_STATUS_BLOCKED_RUNTIME_OR_CITATION",
    "HOLD_STATUS_BLOCKED_STORE_COUNT",
    "STRICT_EVIDENCE_ELIGIBILITY_POST_APPLY_PROMOTION_HOLD_REVIEW_SCHEMA_ID",
    "build_strict_evidence_eligibility_post_apply_promotion_hold_review",
    "render_strict_evidence_eligibility_post_apply_promotion_hold_review_markdown",
    "write_strict_evidence_eligibility_post_apply_promotion_hold_review_reports",
]
