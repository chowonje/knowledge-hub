"""Citation-grade policy-gate design for StrictEvidence eligibility rows.

Consumes the post-apply eligibility promotion hold review and classifies rows
for a future citation-grade record contract. Report-only: does not create
citation-grade records, mutate existing evidence rows, or enable runtime/answer
integration.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_eligibility_post_apply_promotion_hold_review import (
    HOLD_STATUS_ACTIVE,
    STRICT_EVIDENCE_ELIGIBILITY_POST_APPLY_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
)


STRICT_EVIDENCE_CITATION_GRADE_POLICY_GATE_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-citation-grade-policy-gate-design.v1"
)

CITATION_GRADE_POLICY_DESIGN_STATUS_CANDIDATE_ONLY = (
    "citation_grade_policy_design_candidate_only"
)
CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_HOLD_NOT_ACTIVE = "blocked_post_apply_hold_not_active"
CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_DOWNSTREAM_GATE = (
    "blocked_downstream_gate_already_enabled"
)
CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_STORE_COUNT = "blocked_store_row_count_changed"
CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_RUNTIME_OR_CITATION = (
    "blocked_runtime_or_citation_flag_violation"
)
CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

EXPECTED_INPUT_ROWS = 99
EXPECTED_CITATION_GRADE_POLICY_DESIGN_ROWS = 99
EXPECTED_SECTION_ROWS = 45
EXPECTED_FIGURE_CAPTION_ROWS = 54
EXPECTED_STRICT_EVIDENCE_STORE_ROWS = 99
EXPECTED_SOURCE_SPAN_STORE_ROWS = 102

DEFAULT_HOLD_REVIEW_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-eligibility-post-apply-promotion-hold-review"
    / "01-strict-evidence-eligibility-post-apply-promotion-hold-review"
    / "strict-evidence-eligibility-post-apply-promotion-hold-review.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-citation-grade-policy-gate-design"
    / "01-strict-evidence-citation-grade-policy-gate-design"
)

_WRITE_COUNT_FIELDS = (
    "eligibilityRecordWriteRows",
    "strictEvidenceWriteRows",
    "strictEvidenceCreatedRows",
    "sourceSpanUpdatedRows",
    "strictEligibleMutationRows",
    "citationGradeEvidenceCreatedRows",
    "runtimeEvidenceCreatedRows",
    "parserRoutingChangedRows",
    "answerIntegrationChangedRows",
    "databaseMutationRows",
    "canonicalParsedArtifactWriteRows",
    "manifestWriteRows",
    "reindexOrReembedRows",
)

_DOWNSTREAM_GATE_FIELDS = {
    "eligibilityRecordWrite": ("eligibilityRecordWriteAllowed", "eligibilityRecordWrite"),
    "strictEligibleMutation": ("strictEligibleMutationAllowed", "strictEligibleMutation"),
    "strictEvidenceStoreWrite": ("strictEvidenceStoreWriteAllowed", "strictEvidenceStoreWrite"),
    "sourceSpanStoreWrite": ("sourceSpanStoreWriteAllowed", "sourceSpanStoreWrite"),
    "citationGradeEvidence": (
        "citationGradeAllowed",
        "citationGradeEvidenceCreated",
        "citationReady",
    ),
    "runtimeEvidence": ("runtimeEvidenceAllowed", "runtimeEvidenceCreated", "runtimeEvidenceReady"),
    "parserRouting": ("parserRoutingAllowed", "parserRoutingChanged", "parserRoutingReady"),
    "answerIntegration": (
        "answerIntegrationAllowed",
        "answerIntegrationChanged",
        "answerIntegrationReady",
    ),
    "databaseMutation": ("databaseMutationAllowed", "databaseMutation", "runtimeMutationAllowed"),
    "reindexOrReembed": ("reindexOrReembedAllowed", "reindexOrReembed"),
    "manifestWrite": ("runManifestWriteAllowed", "manifestWriteAllowed", "manifestWrite"),
}

_ROW_ILLEGAL_BOOL_FIELDS = (
    "strictEligible",
    "strictEvidenceCreated",
    "citationGrade",
    "runtimeEvidence",
    "runtimeVisible",
    "parserRoutingChanged",
    "answerIntegrationChanged",
    "databaseMutation",
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
        "policyGateDesignOnly": True,
        "citationGradeRecordWrite": False,
        "eligibilityRecordWrite": False,
        "strictEvidenceStoreWrite": False,
        "sourceSpanStoreWrite": False,
        "strictEvidenceCreated": False,
        "strictEligibleMutation": False,
        "citationGradeBooleanMutation": False,
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


def _citation_grade_policy_design() -> dict[str, Any]:
    return {
        "decision": "separate_append_only_citation_grade_record",
        "citationGradeRecordRequired": True,
        "citationGradeRecordAppendOnly": True,
        "citationGradeRecordContractRequired": True,
        "citationGradeStoreName": "parsed_artifact_strict_evidence_citation_grade_store",
        "strictEvidenceInPlaceMutationAllowed": False,
        "eligibilityRecordInPlaceMutationAllowed": False,
        "sourceSpanInPlaceMutationAllowed": False,
        "citationGradeBooleanMutationAllowed": False,
        "citationGradeEvidenceCreatedByThisGate": False,
        "runtimeEvidenceAllowedByThisGate": False,
        "runtimeVisibleByThisGate": False,
        "parserRoutingAllowedByThisGate": False,
        "answerIntegrationAllowedByThisGate": False,
        "noAnswerSafetyEvalRequiredBeforeRuntime": True,
        "runtimeBindingGateRequired": True,
        "supportedArtifactTypesForThisPilot": ["section", "figure"],
        "citationPolicyVersion": "strict_evidence_citation_grade_policy.v1",
        "minimumInputState": "eligibility_post_apply_promotion_hold_active",
        "minimumEligibilityState": "strict_evidence_eligible_candidate_only",
        "minimumEligibilityDecision": "eligible_for_citation_grade_gate_candidate_only",
        "readbackRequirements": [
            "eligibility_record_resolves_to_strict_evidence_record",
            "strict_evidence_record_remains_non_runtime",
            "source_span_record_remains_non_strict_non_runtime",
            "citation_grade_record_references_eligibility_record",
            "no_duplicate_citation_grade_idempotency_key",
        ],
        "blockedUntil": [
            "citation_grade_record_contract",
            "citation_grade_executor_dry_run",
            "citation_grade_executor_apply",
            "citation_grade_readback_review",
            "no_answer_safety_eval_gate",
            "runtime_binding_gate",
            "answer_integration_gate",
        ],
    }


def _future_promotion_readiness_checklist() -> list[dict[str, Any]]:
    return [
        {
            "id": "citation_grade_record_contract",
            "title": "Append-only citation-grade record contract",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "citation_grade_executor_dry_run",
            "title": "Citation-grade executor dry-run with zero writes",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "citation_grade_executor_apply",
            "title": "Explicit apply-gated citation-grade JSONL write",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "citation_grade_readback_review",
            "title": "Readback review for citation-grade records",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "no_answer_safety_eval_gate",
            "title": "No-answer safety eval before runtime binding",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "rollback_strategy",
            "title": "Rollback strategy by run id and citation-grade record id",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "runtime_binding_gate",
            "title": "Runtime evidence binding gate",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "answer_integration_gate",
            "title": "Answer integration gate after runtime binding",
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


def _downstream_gate_matrix(hold_report: dict[str, Any]) -> dict[str, Any]:
    gate = hold_report.get("gate") if isinstance(hold_report.get("gate"), dict) else {}
    policy = hold_report.get("policy") if isinstance(hold_report.get("policy"), dict) else {}
    matrix: dict[str, Any] = {}
    for name, fields in _DOWNSTREAM_GATE_FIELDS.items():
        enabled = any(_safe_bool(gate.get(field)) or _safe_bool(policy.get(field)) for field in fields)
        matrix[name] = {
            "allowed": enabled,
            "ready": enabled,
            "reason": (
                "blocked_until_explicit_citation_grade_promotion_tranche"
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
    for field_name in (
        "eligibilityRecordWriteRows",
        "strictEvidenceWriteRows",
        "strictEvidenceCreatedRows",
        "sourceSpanUpdatedRows",
        "strictEligibleMutationRows",
        "citationGradeEvidenceCreatedRows",
        "runtimeEvidenceCreatedRows",
        "parserRoutingChangedRows",
        "answerIntegrationChangedRows",
        "databaseMutationRows",
        "manifestWriteRows",
        "reindexOrReembedRows",
    ):
        if _safe_int(row.get(field_name)) != 0:
            violations.append(f"{field_name}={_safe_int(row.get(field_name))}_expected_0")
    return _dedupe(violations)


def _input_report_violations(
    *,
    hold_report: dict[str, Any],
    expected_input_rows: int,
    expected_candidate_rows: int,
    expected_section_rows: int,
    expected_figure_caption_rows: int,
    expected_strict_evidence_store_rows: int,
    expected_source_span_store_rows: int,
) -> tuple[list[str], list[str], list[str]]:
    schema_violations: list[str] = []
    store_count_violations: list[str] = []
    write_count_violations: list[str] = []
    counts = hold_report.get("counts") if isinstance(hold_report.get("counts"), dict) else {}
    gate = hold_report.get("gate") if isinstance(hold_report.get("gate"), dict) else {}

    if not hold_report:
        schema_violations.append("hold_review_report_missing_or_unreadable")
        return schema_violations, store_count_violations, write_count_violations

    validation = validate_payload(
        hold_report,
        STRICT_EVIDENCE_ELIGIBILITY_POST_APPLY_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        schema_violations.extend(str(error) for error in validation.errors)
    if _safe_text(hold_report.get("status")) != "ok":
        schema_violations.append(f"hold_report_status={_safe_text(hold_report.get('status')) or 'unknown'}")
    if not _safe_bool(gate.get("eligibilityPostApplyPromotionHoldReviewReady")):
        schema_violations.append("eligibility_post_apply_hold_review_not_ready")
    if _safe_text(gate.get("holdDecision")) != "strict_evidence_eligibility_post_apply_promotion_hold_active":
        schema_violations.append(f"hold_decision={_safe_text(gate.get('holdDecision')) or 'unknown'}")

    expectations = {
        "inputRows": expected_input_rows,
        "eligibilityRecordRows": expected_candidate_rows,
        "readbackValidatedRows": expected_candidate_rows,
        "holdActiveRows": expected_candidate_rows,
        "sectionHoldRows": expected_section_rows,
        "figureCaptionHoldRows": expected_figure_caption_rows,
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


def _design_rows(
    hold_rows: list[dict[str, Any]],
    *,
    schema_violations: list[str],
    store_count_violations: list[str],
    write_count_violations: list[str],
    downstream_gate_violations: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, hold_row in enumerate(hold_rows):
        source_row = dict(hold_row or {})
        status = CITATION_GRADE_POLICY_DESIGN_STATUS_CANDIDATE_ONLY
        blockers: list[str] = []
        recommended_action = "queue_for_citation_grade_record_contract"

        row_violations = _row_flag_violations(source_row)
        if schema_violations:
            status = CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_INPUT_SCHEMA
            blockers.extend(schema_violations)
            recommended_action = "blocked_input_schema_violation"
        elif store_count_violations:
            status = CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_STORE_COUNT
            blockers.extend(store_count_violations)
            recommended_action = "blocked_store_row_count_changed"
        elif write_count_violations or downstream_gate_violations:
            status = CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_DOWNSTREAM_GATE
            blockers.extend(write_count_violations)
            blockers.extend(downstream_gate_violations)
            recommended_action = "blocked_downstream_gate_already_enabled"
        elif (
            _safe_text(source_row.get("hold_status")) != HOLD_STATUS_ACTIVE
            or not _safe_bool(source_row.get("postApplyPromotionHoldActive"))
        ):
            status = CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_HOLD_NOT_ACTIVE
            blockers.append(f"hold_status={_safe_text(source_row.get('hold_status')) or 'unknown'}")
            recommended_action = "blocked_post_apply_hold_not_active"
        elif _safe_text(source_row.get("eligibilityState")) != "strict_evidence_eligible_candidate_only":
            status = CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_HOLD_NOT_ACTIVE
            blockers.append(
                f"eligibilityState={_safe_text(source_row.get('eligibilityState')) or 'unknown'}"
            )
            recommended_action = "blocked_post_apply_hold_not_active"
        elif _safe_text(source_row.get("eligibilityDecision")) != "eligible_for_citation_grade_gate_candidate_only":
            status = CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_HOLD_NOT_ACTIVE
            blockers.append(
                f"eligibilityDecision={_safe_text(source_row.get('eligibilityDecision')) or 'unknown'}"
            )
            recommended_action = "blocked_post_apply_hold_not_active"
        elif row_violations:
            status = CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_RUNTIME_OR_CITATION
            blockers.extend(row_violations)
            recommended_action = "blocked_runtime_or_citation_flag_violation"

        ready = status == CITATION_GRADE_POLICY_DESIGN_STATUS_CANDIDATE_ONLY
        rows.append(
            {
                "policy_design_row_id": f"strict-evidence-citation-grade-policy-gate-design:{index:04d}",
                "hold_row_id": _safe_text(source_row.get("hold_row_id")),
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
                "hold_status": _safe_text(source_row.get("hold_status")),
                "citation_grade_policy_design_status": status,
                "citation_grade_policy_design_blockers": _dedupe(blockers),
                "citationGradePolicyDesignCandidateOnly": ready,
                "citationGradePolicyVersion": "strict_evidence_citation_grade_policy.v1",
                "citationGradeRecordRequired": ready,
                "citationGradeRecordWriteAllowed": False,
                "citationGradeBooleanMutationAllowed": False,
                "citationGradeAllowed": False,
                "runtimeEvidenceAllowed": False,
                "runtimeVisibleAllowed": False,
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
    hold_counts: dict[str, Any],
    aggregate_violations: list[str],
    gate_matrix: dict[str, Any],
) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("citation_grade_policy_design_status")) for row in rows)
    candidate_rows = [
        row
        for row in rows
        if row.get("citation_grade_policy_design_status")
        == CITATION_GRADE_POLICY_DESIGN_STATUS_CANDIDATE_ONLY
    ]
    return {
        "inputRows": _safe_int(hold_counts.get("inputRows")),
        "eligibilityRecordRows": _safe_int(hold_counts.get("eligibilityRecordRows")),
        "holdActiveRows": _safe_int(hold_counts.get("holdActiveRows")),
        "citationGradePolicyDesignCandidateOnlyRows": int(
            by_status.get(CITATION_GRADE_POLICY_DESIGN_STATUS_CANDIDATE_ONLY, 0)
        ),
        "sectionCitationGradePolicyDesignRows": sum(
            1 for row in candidate_rows if _safe_text(row.get("artifact_type")) == "section"
        ),
        "figureCaptionCitationGradePolicyDesignRows": sum(
            1 for row in candidate_rows if _safe_text(row.get("artifact_type")) == "figure"
        ),
        "blockedPostApplyHoldNotActiveRows": int(
            by_status.get(CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_HOLD_NOT_ACTIVE, 0)
        ),
        "blockedDownstreamGateAlreadyEnabledRows": int(
            by_status.get(CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_DOWNSTREAM_GATE, 0)
        ),
        "blockedStoreRowCountChangedRows": int(
            by_status.get(CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_STORE_COUNT, 0)
        ),
        "blockedRuntimeOrCitationFlagViolationRows": int(
            by_status.get(CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_RUNTIME_OR_CITATION, 0)
        ),
        "blockedInputSchemaViolationRows": int(
            by_status.get(CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_INPUT_SCHEMA, 0)
        ),
        "citationGradeRecordWriteAllowedRows": sum(
            1 for row in rows if _safe_bool(row.get("citationGradeRecordWriteAllowed"))
        ),
        "citationGradeAllowedRows": sum(1 for row in rows if _safe_bool(row.get("citationGradeAllowed"))),
        "runtimeEvidenceAllowedRows": sum(
            1 for row in rows if _safe_bool(row.get("runtimeEvidenceAllowed"))
        ),
        "parserRoutingAllowedRows": sum(1 for row in rows if _safe_bool(row.get("parserRoutingAllowed"))),
        "answerIntegrationAllowedRows": sum(
            1 for row in rows if _safe_bool(row.get("answerIntegrationAllowed"))
        ),
        "strictEvidenceStoreRows": _safe_int(hold_counts.get("strictEvidenceStoreRows")),
        "sourceSpanStoreRows": _safe_int(hold_counts.get("sourceSpanStoreRows")),
        "citationGradeRecordWriteRows": 0,
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
        "byPaperId": dict(Counter(_safe_text(row.get("paper_id")) for row in candidate_rows)),
        "byArtifactType": dict(Counter(_safe_text(row.get("artifact_type")) for row in candidate_rows)),
        "byPolicyDesignStatus": dict(by_status),
        "byRecommendedAction": dict(Counter(_safe_text(row.get("recommended_action")) for row in rows)),
    }


def build_strict_evidence_citation_grade_policy_gate_design(
    *,
    hold_review_report_path: str | Path = DEFAULT_HOLD_REVIEW_REPORT_PATH,
    paper_ids: list[str] | None = None,
    expected_input_rows: int = EXPECTED_INPUT_ROWS,
    expected_candidate_rows: int = EXPECTED_CITATION_GRADE_POLICY_DESIGN_ROWS,
    expected_section_rows: int = EXPECTED_SECTION_ROWS,
    expected_figure_caption_rows: int = EXPECTED_FIGURE_CAPTION_ROWS,
    expected_strict_evidence_store_rows: int = EXPECTED_STRICT_EVIDENCE_STORE_ROWS,
    expected_source_span_store_rows: int = EXPECTED_SOURCE_SPAN_STORE_ROWS,
) -> dict[str, Any]:
    report_path = Path(str(hold_review_report_path)).expanduser()
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}
    warnings: list[str] = []

    hold_report = _read_json(report_path)
    schema_violations, store_count_violations, write_count_violations = _input_report_violations(
        hold_report=hold_report,
        expected_input_rows=expected_input_rows,
        expected_candidate_rows=expected_candidate_rows,
        expected_section_rows=expected_section_rows,
        expected_figure_caption_rows=expected_figure_caption_rows,
        expected_strict_evidence_store_rows=expected_strict_evidence_store_rows,
        expected_source_span_store_rows=expected_source_span_store_rows,
    )
    gate_matrix = _downstream_gate_matrix(hold_report) if hold_report else {}
    downstream_violations = _downstream_gate_violations(gate_matrix)

    hold_rows = [row for row in hold_report.get("rows", []) if isinstance(row, dict)] if hold_report else []
    if requested_papers:
        found = {_safe_text(row.get("paper_id")) for row in hold_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found:
            warnings.append("requested_paper_ids_not_found_in_hold_review")
        hold_rows = [row for row in hold_rows if _safe_text(row.get("paper_id")) in requested_papers]

    if not hold_rows and not schema_violations:
        schema_violations.append("hold_review_rows_missing")

    aggregate_violations = _dedupe(
        schema_violations + store_count_violations + write_count_violations + downstream_violations
    )
    rows = _design_rows(
        hold_rows,
        schema_violations=schema_violations,
        store_count_violations=store_count_violations,
        write_count_violations=write_count_violations,
        downstream_gate_violations=downstream_violations,
    )
    hold_counts = hold_report.get("counts") if isinstance(hold_report.get("counts"), dict) else {}
    counts = _count_rows(
        rows=rows,
        hold_counts=hold_counts,
        aggregate_violations=aggregate_violations,
        gate_matrix=gate_matrix,
    )

    candidate_rows = int(counts.get("citationGradePolicyDesignCandidateOnlyRows") or 0)
    status = "ok"
    if (
        aggregate_violations
        or not rows
        or candidate_rows != expected_candidate_rows
        or candidate_rows != len(rows)
        or int(counts.get("downstreamGateAllowedCount") or 0) > 0
    ):
        status = "blocked"

    policy_matrix = _no_mutation_policy_matrix()
    return {
        "schema": STRICT_EVIDENCE_CITATION_GRADE_POLICY_GATE_DESIGN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "holdReviewReportPath": str(report_path),
            "holdReviewReportSchema": _safe_text(hold_report.get("schema")) if hold_report else "",
            "holdReviewReportStatus": _safe_text(hold_report.get("status")) if hold_report else "",
            "holdDecision": _safe_text((hold_report.get("gate") or {}).get("holdDecision"))
            if hold_report
            else "",
            "requestedPaperIds": sorted(requested_papers),
            "expectedInputRows": expected_input_rows,
            "expectedCitationGradePolicyDesignRows": expected_candidate_rows,
            "expectedSectionRows": expected_section_rows,
            "expectedFigureCaptionRows": expected_figure_caption_rows,
            "expectedStrictEvidenceStoreRows": expected_strict_evidence_store_rows,
            "expectedSourceSpanStoreRows": expected_source_span_store_rows,
        },
        "counts": counts,
        "citationGradePolicyDesign": _citation_grade_policy_design(),
        "blockedDownstreamGateMatrix": gate_matrix,
        "futurePromotionReadinessChecklist": _future_promotion_readiness_checklist(),
        "noMutationPolicyMatrix": policy_matrix,
        "gate": {
            "citationGradePolicyGateDesignReady": status == "ok",
            "decision": (
                "strict_evidence_citation_grade_policy_gate_design_candidate_only"
                if status == "ok"
                else "strict_evidence_citation_grade_policy_gate_design_blocked"
            ),
            "citationGradeRecordContractRequired": status == "ok",
            "citationGradeRecordWriteAllowed": False,
            "citationGradeBooleanMutationAllowed": False,
            "citationGradeAllowed": False,
            "runtimeEvidenceAllowed": False,
            "runtimeVisibleAllowed": False,
            "parserRoutingAllowed": False,
            "answerIntegrationAllowed": False,
            "databaseMutationAllowed": False,
            "reindexOrReembedAllowed": False,
            "runManifestWriteAllowed": False,
            "strictEligibleMutationAllowed": False,
            "schemaViolations": aggregate_violations,
            "recommendedNextTranche": (
                "strict_evidence_citation_grade_record_contract"
                if status == "ok"
                else "strict_evidence_eligibility_post_apply_promotion_hold_repair"
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
            "citationGradePolicyDesign",
            "blockedDownstreamGateMatrix",
            "futurePromotionReadinessChecklist",
            "noMutationPolicyMatrix",
            "gate",
            "policy",
            "warnings",
        )
        if key in report
    }


def render_strict_evidence_citation_grade_policy_gate_design_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byPolicyDesignStatus") or {})).items())
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
            "# Strict Evidence Citation-Grade Policy Gate Design",
            "",
            f"- status: {report.get('status', '')}",
            f"- decision: {gate.get('decision', '')}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- eligibility record rows: {int(counts.get('eligibilityRecordRows') or 0)}",
            f"- hold active rows: {int(counts.get('holdActiveRows') or 0)}",
            f"- citation-grade policy design candidates: {int(counts.get('citationGradePolicyDesignCandidateOnlyRows') or 0)}",
            f"- section candidates: {int(counts.get('sectionCitationGradePolicyDesignRows') or 0)}",
            f"- figure caption candidates: {int(counts.get('figureCaptionCitationGradePolicyDesignRows') or 0)}",
            f"- strict evidence store rows: {int(counts.get('strictEvidenceStoreRows') or 0)}",
            f"- source span store rows: {int(counts.get('sourceSpanStoreRows') or 0)}",
            "",
            "## Blocked downstream gate matrix",
            *matrix_lines,
            "",
            "## Future promotion readiness checklist",
            *checklist_lines,
            "",
            "## Policy design status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- recommended next tranche: {gate.get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_citation_grade_policy_gate_design_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-citation-grade-policy-gate-design.json"
    summary_path = root / "strict-evidence-citation-grade-policy-gate-design-summary.json"
    markdown_path = root / "strict-evidence-citation-grade-policy-gate-design.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_citation_grade_policy_gate_design_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Design a citation-grade policy gate for StrictEvidence eligibility rows "
            "without creating citation/runtime evidence or integration changes."
        )
    )
    parser.add_argument(
        "--hold-review-report",
        default=str(DEFAULT_HOLD_REVIEW_REPORT_PATH),
        help="Path to eligibility post-apply promotion hold review JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, summary, and markdown reports.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_citation_grade_policy_gate_design(
        hold_review_report_path=args.hold_review_report,
        paper_ids=args.paper_id or None,
    )
    paths = write_strict_evidence_citation_grade_policy_gate_design_reports(report, args.output_dir)
    print(f"wrote report: {paths['report']}")
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote markdown: {paths['markdown']}")
    if args.json:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CITATION_GRADE_POLICY_DESIGN_STATUS_CANDIDATE_ONLY",
    "DEFAULT_HOLD_REVIEW_REPORT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "STRICT_EVIDENCE_CITATION_GRADE_POLICY_GATE_DESIGN_SCHEMA_ID",
    "build_strict_evidence_citation_grade_policy_gate_design",
    "render_strict_evidence_citation_grade_policy_gate_design_markdown",
    "write_strict_evidence_citation_grade_policy_gate_design_reports",
]
