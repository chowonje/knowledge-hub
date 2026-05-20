"""Decision record for StrictEvidence runtime binding visibility semantics.

Consumes the runtime binding post-apply promotion hold review report and records
the policy decision that runtime binding rows must not be mutated in place to
become runtime-visible. This helper is report-only: it does not write visibility
records, mutate parent stores, create runtime evidence, or enable answer paths.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_runtime_binding_executor_apply_readback_review import (
    EXPECTED_CITATION_GRADE_STORE_ROWS,
    EXPECTED_ELIGIBILITY_STORE_ROWS,
    EXPECTED_INPUT_ROWS,
    EXPECTED_RUNTIME_BINDING_RECORD_ROWS,
    EXPECTED_SOURCE_SPAN_STORE_ROWS,
    EXPECTED_STRICT_EVIDENCE_STORE_ROWS,
)
from knowledge_hub.papers.strict_evidence_runtime_binding_post_apply_promotion_hold_review import (
    DEFAULT_OUTPUT_DIR as DEFAULT_HOLD_REVIEW_OUTPUT_DIR,
    HOLD_STATUS_ACTIVE,
    STRICT_EVIDENCE_RUNTIME_BINDING_POST_APPLY_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
)


STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_DECISION_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-runtime-binding-visibility-decision-record.v1"
)

DECISION_SEPARATE_RUNTIME_VISIBILITY_RECORD = "separate_append_only_runtime_visibility_record"

DECISION_STATUS_CANDIDATE_ONLY = "runtime_binding_visibility_decision_candidate_only"
DECISION_STATUS_BLOCKED_HOLD_NOT_ACTIVE = "blocked_runtime_binding_post_apply_hold_not_active"
DECISION_STATUS_BLOCKED_DOWNSTREAM_GATE = "blocked_downstream_gate_already_enabled"
DECISION_STATUS_BLOCKED_STORE_COUNT = "blocked_store_row_count_changed"
DECISION_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
DECISION_STATUS_BLOCKED_RUNTIME_OR_ANSWER = "blocked_runtime_or_answer_flag_violation"

EXPECTED_SECTION_DECISION_ROWS = 45
EXPECTED_FIGURE_CAPTION_DECISION_ROWS = 54

DEFAULT_HOLD_REVIEW_REPORT_PATH = (
    DEFAULT_HOLD_REVIEW_OUTPUT_DIR
    / "strict-evidence-runtime-binding-post-apply-promotion-hold-review.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-runtime-binding-visibility-decision-record"
    / "01-strict-evidence-runtime-binding-visibility-decision-record"
)

_WRITE_COUNT_FIELDS = (
    "runtimeBindingRecordWriteRows",
    "citationGradeRecordWriteRows",
    "citationGradeBooleanMutationRows",
    "eligibilityRecordWriteRows",
    "strictEvidenceWriteRows",
    "sourceSpanUpdatedRows",
    "strictEligibleMutationRows",
    "strictEvidenceCreatedRows",
    "runtimeEvidenceCreatedRows",
    "runtimeVisibleRows",
    "answerIntegrationVisibleRows",
    "parserRoutingChangedRows",
    "answerIntegrationChangedRows",
    "databaseMutationRows",
    "canonicalParsedArtifactWriteRows",
    "manifestWriteRows",
    "reindexOrReembedRows",
)

_ROW_ILLEGAL_BOOL_FIELDS = (
    "runtimeVisible",
    "answerIntegrationVisible",
    "runtimeEvidence",
    "runtimeBindingMutationApplied",
    "strictEligible",
    "strictEligibleMutationApplied",
    "strictEvidenceCreated",
    "parserRoutingChanged",
    "answerIntegrationChanged",
    "databaseMutation",
)

_ROW_ILLEGAL_COUNT_FIELDS = (
    "runtimeVisibleRows",
    "answerIntegrationVisibleRows",
    "runtimeBindingRecordWriteRows",
    "citationGradeRecordWriteRows",
    "eligibilityRecordWriteRows",
    "strictEvidenceWriteRows",
    "sourceSpanUpdatedRows",
    "strictEligibleMutationRows",
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


def _runtime_visibility_semantics_decision() -> dict[str, Any]:
    return {
        "decision": DECISION_SEPARATE_RUNTIME_VISIBILITY_RECORD,
        "runtimeBindingInPlaceMutationAllowed": False,
        "runtimeVisibleBooleanMutationAllowed": False,
        "answerIntegrationVisibleBooleanMutationAllowed": False,
        "runtimeVisibilityRecordRequired": True,
        "runtimeVisibilityRecordAppendOnly": True,
        "runtimeVisibilityStoreName": "parsed_artifact_strict_evidence_runtime_visibility_store",
        "runtimeVisibilityRecordContractRequired": True,
        "runtimeVisibilityRecordRuntimeVisible": False,
        "answerIntegrationAllowedByThisDecision": False,
        "parserRoutingAllowedByThisDecision": False,
        "runtimeEvidenceAllowedByThisDecision": False,
        "rationale": [
            "preserve runtime binding records as immutable audit records",
            "make runtime visibility decisions independently reviewable and rollbackable",
            "avoid turning legacy boolean fields into runtime authority",
            "keep answer integration and no-answer safety behind separate gates",
        ],
        "alternativesRejected": [
            {
                "alternative": "mutate_runtime_binding_record_in_place",
                "reason": "would obscure when and why a runtime binding became visible",
            },
            {
                "alternative": "treat_runtime_visibility_as_answer_integration",
                "reason": "would bypass answer integration and no-answer safety gates",
            },
        ],
    }


def _no_mutation_policy_matrix() -> dict[str, Any]:
    return {
        "reportOnly": True,
        "decisionRecordOnly": True,
        "runtimeVisibilityRecordWrite": False,
        "runtimeBindingRecordWrite": False,
        "runtimeBindingStoreWrite": False,
        "citationGradeRecordWrite": False,
        "eligibilityRecordWrite": False,
        "strictEvidenceStoreWrite": False,
        "sourceSpanStoreWrite": False,
        "runtimeVisibleMutation": False,
        "answerIntegrationVisibleMutation": False,
        "runtimeEvidenceCreated": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "vaultScan": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
        "manifestWrite": False,
    }


def _downstream_gate_allowed_count(hold_review: dict[str, Any]) -> int:
    matrix = hold_review.get("blockedDownstreamGateMatrix")
    if not isinstance(matrix, dict):
        return 0
    return sum(
        1
        for section in matrix.values()
        if isinstance(section, dict) and (_safe_bool(section.get("allowed")) or _safe_bool(section.get("ready")))
    )


def _aggregate_hold_violations(
    *,
    hold_review: dict[str, Any],
    input_schema_violations: list[str],
    expected_input_rows: int,
    expected_runtime_binding_record_rows: int,
    expected_section_decision_rows: int,
    expected_figure_caption_decision_rows: int,
    expected_citation_grade_store_rows: int,
    expected_strict_evidence_store_rows: int,
    expected_eligibility_store_rows: int,
    expected_source_span_store_rows: int,
) -> list[str]:
    violations = list(input_schema_violations)
    counts = hold_review.get("counts") if isinstance(hold_review.get("counts"), dict) else {}
    gate = hold_review.get("gate") if isinstance(hold_review.get("gate"), dict) else {}

    if _safe_text(hold_review.get("status")) != "ok":
        violations.append(f"hold_review_status={_safe_text(hold_review.get('status')) or 'unknown'}")
    if _safe_text(gate.get("holdDecision")) != "strict_evidence_runtime_binding_post_apply_promotion_hold_active":
        violations.append(f"hold_decision={_safe_text(gate.get('holdDecision')) or 'unknown'}")
    if not _safe_bool(gate.get("runtimeBindingPostApplyPromotionHoldReviewReady")):
        violations.append("runtime_binding_post_apply_promotion_hold_review_not_ready")

    expectations = {
        "inputRows": expected_input_rows,
        "runtimeBindingRecordRows": expected_runtime_binding_record_rows,
        "readbackValidatedRows": expected_runtime_binding_record_rows,
        "runtimeBindingHoldRows": expected_runtime_binding_record_rows,
        "sectionHoldRows": expected_section_decision_rows,
        "figureCaptionHoldRows": expected_figure_caption_decision_rows,
        "citationGradeStoreRows": expected_citation_grade_store_rows,
        "strictEvidenceStoreRows": expected_strict_evidence_store_rows,
        "eligibilityStoreRows": expected_eligibility_store_rows,
        "sourceSpanStoreRows": expected_source_span_store_rows,
    }
    for field_name, expected in expectations.items():
        actual = _safe_int(counts.get(field_name))
        if actual != expected:
            violations.append(f"{field_name}={actual}_expected_{expected}")

    for field_name in _WRITE_COUNT_FIELDS:
        if _safe_int(counts.get(field_name)) != 0:
            violations.append(f"{field_name}={_safe_int(counts.get(field_name))}_expected_0")

    if _downstream_gate_allowed_count(hold_review) > 0:
        violations.append("downstream_gate_already_enabled")

    return _dedupe(violations)


def _row_blockers(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for field_name in _ROW_ILLEGAL_BOOL_FIELDS:
        if _safe_bool(row.get(field_name)):
            blockers.append(f"{field_name}_true")
    for field_name in _ROW_ILLEGAL_COUNT_FIELDS:
        if _safe_int(row.get(field_name)) != 0:
            blockers.append(f"{field_name}={_safe_int(row.get(field_name))}_expected_0")
    write_matrix = row.get("writeMatrix")
    if isinstance(write_matrix, dict):
        for field_name, value in write_matrix.items():
            if _safe_bool(value):
                blockers.append(f"writeMatrix.{field_name}_true")
    return _dedupe(blockers)


def _decision_rows(
    hold_rows: list[dict[str, Any]],
    *,
    aggregate_violations: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, hold_row in enumerate(hold_rows):
        source_row = dict(hold_row or {})
        hold_status = _safe_text(source_row.get("hold_status"))
        blockers: list[str] = []

        if aggregate_violations:
            blockers.extend(aggregate_violations)
        if hold_status != HOLD_STATUS_ACTIVE:
            blockers.append(f"hold_status={hold_status or 'unknown'}")
        blockers.extend(_row_blockers(source_row))

        if any(item.startswith("input_schema:") or item.startswith("hold_review_status=") for item in blockers):
            decision_status = DECISION_STATUS_BLOCKED_INPUT_SCHEMA
        elif any(item.startswith("hold_status=") for item in blockers):
            decision_status = DECISION_STATUS_BLOCKED_HOLD_NOT_ACTIVE
        elif any("StoreRows=" in item or item.startswith("inputRows=") for item in blockers):
            decision_status = DECISION_STATUS_BLOCKED_STORE_COUNT
        elif "downstream_gate_already_enabled" in blockers:
            decision_status = DECISION_STATUS_BLOCKED_DOWNSTREAM_GATE
        elif _row_blockers(source_row):
            decision_status = DECISION_STATUS_BLOCKED_RUNTIME_OR_ANSWER
        else:
            decision_status = DECISION_STATUS_CANDIDATE_ONLY

        rows.append(
            {
                "visibility_decision_record_row_id": (
                    f"strict-evidence-runtime-binding-visibility-decision-record:{index + 1:04d}"
                ),
                "hold_row_id": _safe_text(source_row.get("hold_row_id")),
                "runtimeBindingRecordId": _safe_text(source_row.get("runtimeBindingRecordId")),
                "citationGradeRecordId": _safe_text(source_row.get("citationGradeRecordId")),
                "eligibilityRecordId": _safe_text(source_row.get("eligibilityRecordId")),
                "strictEvidenceId": _safe_text(source_row.get("strictEvidenceId")),
                "sourceSpanId": _safe_text(source_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(source_row.get("candidateRecordId")),
                "paper_id": _safe_text(source_row.get("paper_id") or source_row.get("paperId")),
                "artifact_type": _safe_text(source_row.get("artifact_type") or source_row.get("artifactType")),
                "sourceContentHash": _safe_text(source_row.get("sourceContentHash")),
                "hold_status": hold_status,
                "decision_status": decision_status,
                "decision_blockers": _dedupe(blockers),
                "decision": DECISION_SEPARATE_RUNTIME_VISIBILITY_RECORD,
                "runtimeVisibilityRecordRequired": True,
                "runtimeVisibilityRecordAppendOnly": True,
                "runtimeBindingInPlaceMutationAllowed": False,
                "runtimeVisibleMutationAllowed": False,
                "answerIntegrationVisibleAllowed": False,
                "runtimeEvidenceAllowed": False,
                "parserRoutingAllowed": False,
                "answerIntegrationAllowed": False,
                "databaseMutationAllowed": False,
                "reindexOrReembedAllowed": False,
                "vaultScanAllowed": False,
                "runtimeVisibilityRecordWriteRows": 0,
                "runtimeBindingRecordWriteRows": 0,
                "runtimeVisibleRows": 0,
                "answerIntegrationVisibleRows": 0,
                "runtimeEvidenceCreatedRows": 0,
                "parserRoutingChangedRows": 0,
                "answerIntegrationChangedRows": 0,
                "databaseMutationRows": 0,
                "manifestWriteRows": 0,
                "recommended_action": (
                    "runtime_visibility_record_contract_candidate_only"
                    if decision_status == DECISION_STATUS_CANDIDATE_ONLY
                    else "repair_runtime_binding_visibility_decision_input"
                ),
            }
        )
    return rows


def _counts(
    rows: list[dict[str, Any]],
    hold_review: dict[str, Any],
    input_schema_violations: list[str],
) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("decision_status")) for row in rows)
    by_artifact_type = Counter(_safe_text(row.get("artifact_type")) for row in rows)
    by_action = Counter(_safe_text(row.get("recommended_action")) for row in rows)
    hold_counts = hold_review.get("counts") if isinstance(hold_review.get("counts"), dict) else {}

    return {
        "inputRows": len(rows),
        "runtimeBindingRecordRows": _safe_int(hold_counts.get("runtimeBindingRecordRows")),
        "runtimeBindingHoldRows": _safe_int(hold_counts.get("runtimeBindingHoldRows")),
        "visibilityDecisionCandidateOnlyRows": by_status.get(DECISION_STATUS_CANDIDATE_ONLY, 0),
        "sectionDecisionRows": by_artifact_type.get("section", 0),
        "figureCaptionDecisionRows": by_artifact_type.get("figure", 0),
        "blockedHoldNotActiveRows": by_status.get(DECISION_STATUS_BLOCKED_HOLD_NOT_ACTIVE, 0),
        "blockedDownstreamGateAlreadyEnabledRows": by_status.get(DECISION_STATUS_BLOCKED_DOWNSTREAM_GATE, 0),
        "blockedStoreRowCountChangedRows": by_status.get(DECISION_STATUS_BLOCKED_STORE_COUNT, 0),
        "blockedRuntimeOrAnswerFlagViolationRows": by_status.get(DECISION_STATUS_BLOCKED_RUNTIME_OR_ANSWER, 0),
        "blockedInputSchemaViolationRows": by_status.get(DECISION_STATUS_BLOCKED_INPUT_SCHEMA, 0),
        "runtimeVisibilityRecordWriteAllowedRows": 0,
        "runtimeVisibleMutationAllowedRows": 0,
        "answerIntegrationVisibleAllowedRows": 0,
        "runtimeEvidenceAllowedRows": 0,
        "parserRoutingAllowedRows": 0,
        "answerIntegrationAllowedRows": 0,
        "vaultScanAllowedRows": 0,
        "citationGradeStoreRows": _safe_int(hold_counts.get("citationGradeStoreRows")),
        "strictEvidenceStoreRows": _safe_int(hold_counts.get("strictEvidenceStoreRows")),
        "eligibilityStoreRows": _safe_int(hold_counts.get("eligibilityStoreRows")),
        "sourceSpanStoreRows": _safe_int(hold_counts.get("sourceSpanStoreRows")),
        "runtimeVisibilityRecordWriteRows": 0,
        "runtimeBindingRecordWriteRows": 0,
        "citationGradeRecordWriteRows": 0,
        "eligibilityRecordWriteRows": 0,
        "strictEvidenceWriteRows": 0,
        "sourceSpanUpdatedRows": 0,
        "strictEligibleMutationRows": 0,
        "strictEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "runtimeVisibleRows": 0,
        "answerIntegrationVisibleRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "manifestWriteRows": 0,
        "reindexOrReembedRows": 0,
        "schemaViolationCount": len(input_schema_violations),
        "downstreamGateAllowedCount": _downstream_gate_allowed_count(hold_review),
        "byArtifactType": dict(sorted(by_artifact_type.items())),
        "byDecisionStatus": dict(sorted(by_status.items())),
        "byRecommendedAction": dict(sorted(by_action.items())),
    }


def build_strict_evidence_runtime_binding_visibility_decision_record(
    *,
    hold_review_report_path: str | Path = DEFAULT_HOLD_REVIEW_REPORT_PATH,
    expected_input_rows: int = EXPECTED_INPUT_ROWS,
    expected_runtime_binding_record_rows: int = EXPECTED_RUNTIME_BINDING_RECORD_ROWS,
    expected_section_decision_rows: int = EXPECTED_SECTION_DECISION_ROWS,
    expected_figure_caption_decision_rows: int = EXPECTED_FIGURE_CAPTION_DECISION_ROWS,
    expected_citation_grade_store_rows: int = EXPECTED_CITATION_GRADE_STORE_ROWS,
    expected_strict_evidence_store_rows: int = EXPECTED_STRICT_EVIDENCE_STORE_ROWS,
    expected_eligibility_store_rows: int = EXPECTED_ELIGIBILITY_STORE_ROWS,
    expected_source_span_store_rows: int = EXPECTED_SOURCE_SPAN_STORE_ROWS,
) -> dict[str, Any]:
    hold_path = Path(str(hold_review_report_path)).expanduser()
    hold_review = _read_json(hold_path)
    input_schema_violations: list[str] = []

    if not hold_review:
        input_schema_violations.append("hold_review_report_missing_or_unreadable")
    elif hold_review.get("schema") != STRICT_EVIDENCE_RUNTIME_BINDING_POST_APPLY_PROMOTION_HOLD_REVIEW_SCHEMA_ID:
        input_schema_violations.append("hold_review_report_schema_mismatch")
    else:
        validation = validate_payload(
            hold_review,
            STRICT_EVIDENCE_RUNTIME_BINDING_POST_APPLY_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            input_schema_violations.extend(f"input_schema:{error}" for error in validation.errors)

    hold_rows = hold_review.get("rows") if isinstance(hold_review.get("rows"), list) else []
    aggregate_violations = _aggregate_hold_violations(
        hold_review=hold_review,
        input_schema_violations=input_schema_violations,
        expected_input_rows=expected_input_rows,
        expected_runtime_binding_record_rows=expected_runtime_binding_record_rows,
        expected_section_decision_rows=expected_section_decision_rows,
        expected_figure_caption_decision_rows=expected_figure_caption_decision_rows,
        expected_citation_grade_store_rows=expected_citation_grade_store_rows,
        expected_strict_evidence_store_rows=expected_strict_evidence_store_rows,
        expected_eligibility_store_rows=expected_eligibility_store_rows,
        expected_source_span_store_rows=expected_source_span_store_rows,
    )
    rows = _decision_rows(hold_rows, aggregate_violations=aggregate_violations)
    counts = _counts(rows, hold_review, input_schema_violations)
    status = "ok" if counts["visibilityDecisionCandidateOnlyRows"] == len(rows) and rows else "blocked"
    decision = _runtime_visibility_semantics_decision()

    return {
        "schema": STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_DECISION_RECORD_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "holdReviewReportPath": str(hold_path),
            "holdReviewReportSchema": _safe_text(hold_review.get("schema")),
            "holdReviewReportStatus": _safe_text(hold_review.get("status")),
            "holdDecision": _safe_text((hold_review.get("gate") or {}).get("holdDecision") if isinstance(hold_review.get("gate"), dict) else ""),
            "expectedInputRows": expected_input_rows,
            "expectedRuntimeBindingRecordRows": expected_runtime_binding_record_rows,
            "expectedSectionDecisionRows": expected_section_decision_rows,
            "expectedFigureCaptionDecisionRows": expected_figure_caption_decision_rows,
            "expectedCitationGradeStoreRows": expected_citation_grade_store_rows,
            "expectedStrictEvidenceStoreRows": expected_strict_evidence_store_rows,
            "expectedEligibilityStoreRows": expected_eligibility_store_rows,
            "expectedSourceSpanStoreRows": expected_source_span_store_rows,
        },
        "counts": counts,
        "decision": decision,
        "noMutationPolicyMatrix": _no_mutation_policy_matrix(),
        "gate": {
            "runtimeBindingVisibilityDecisionRecordReady": status == "ok",
            "decision": decision["decision"],
            "runtimeBindingInPlaceMutationAllowed": False,
            "runtimeVisibleBooleanMutationAllowed": False,
            "answerIntegrationVisibleBooleanMutationAllowed": False,
            "runtimeVisibilityRecordRequired": True,
            "runtimeVisibilityRecordContractRequired": True,
            "runtimeVisibilityRecordWriteAllowed": False,
            "runtimeVisibleMutationAllowed": False,
            "answerIntegrationVisibleAllowed": False,
            "runtimeEvidenceAllowed": False,
            "parserRoutingAllowed": False,
            "answerIntegrationAllowed": False,
            "databaseMutationAllowed": False,
            "reindexOrReembedAllowed": False,
            "vaultScanAllowed": False,
            "runManifestWriteAllowed": False,
            "schemaViolations": input_schema_violations,
            "recommendedNextTranche": (
                "strict_evidence_runtime_binding_visibility_record_contract"
                if status == "ok"
                else "strict_evidence_runtime_binding_visibility_decision_record_input_repair"
            ),
        },
        "policy": {
            "reportOnly": True,
            "decisionRecordOnly": True,
            "visibilityDoesNotMeanAnswerIntegration": True,
            "runtimeBindingRowsImmutable": True,
            "runtimeVisibleMustRemainFalse": True,
            "answerIntegrationVisibleMustRemainFalse": True,
        },
        "warnings": [] if status == "ok" else aggregate_violations,
        "rows": rows,
    }


def render_strict_evidence_runtime_binding_visibility_decision_record_markdown(
    report: dict[str, Any],
) -> str:
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    gate = report.get("gate") if isinstance(report.get("gate"), dict) else {}
    lines = [
        "# Strict Evidence Runtime Binding Visibility Decision Record",
        "",
        f"- status: {report.get('status', '')}",
        f"- decision: {gate.get('decision', '')}",
        f"- input rows: {counts.get('inputRows', 0)}",
        f"- visibility decision candidate-only rows: {counts.get('visibilityDecisionCandidateOnlyRows', 0)}",
        f"- section decision rows: {counts.get('sectionDecisionRows', 0)}",
        f"- figure-caption decision rows: {counts.get('figureCaptionDecisionRows', 0)}",
        f"- runtime visible mutation allowed rows: {counts.get('runtimeVisibleMutationAllowedRows', 0)}",
        f"- answer integration visible allowed rows: {counts.get('answerIntegrationVisibleAllowedRows', 0)}",
        f"- runtime visibility record writes: {counts.get('runtimeVisibilityRecordWriteRows', 0)}",
        f"- runtime evidence created rows: {counts.get('runtimeEvidenceCreatedRows', 0)}",
        f"- parser routing changed rows: {counts.get('parserRoutingChangedRows', 0)}",
        f"- answer integration changed rows: {counts.get('answerIntegrationChangedRows', 0)}",
        f"- database mutation rows: {counts.get('databaseMutationRows', 0)}",
        f"- vault scan allowed rows: {counts.get('vaultScanAllowedRows', 0)}",
        f"- recommended next tranche: {gate.get('recommendedNextTranche', '')}",
        "",
        "## Decision",
        "",
        "- Runtime binding rows remain immutable audit records.",
        "- Runtime visibility requires a later append-only visibility record contract.",
        "- Runtime visibility does not authorize answer integration.",
        "- Runtime evidence, parser routing, answer integration, DB/index/reembed, vault scans, and manifests remain blocked.",
        "",
    ]
    return "\n".join(lines)


def write_strict_evidence_runtime_binding_visibility_decision_record_reports(
    report: dict[str, Any],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "strict-evidence-runtime-binding-visibility-decision-record.json"
    summary_path = out_dir / "strict-evidence-runtime-binding-visibility-decision-record-summary.json"
    markdown_path = out_dir / "strict-evidence-runtime-binding-visibility-decision-record.md"

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": report.get("schema"),
        "status": report.get("status"),
        "generatedAt": report.get("generatedAt"),
        "counts": report.get("counts"),
        "gate": report.get("gate"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_strict_evidence_runtime_binding_visibility_decision_record_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--hold-review-report", default=str(DEFAULT_HOLD_REVIEW_REPORT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)

    report = build_strict_evidence_runtime_binding_visibility_decision_record(
        hold_review_report_path=args.hold_review_report,
    )
    paths = write_strict_evidence_runtime_binding_visibility_decision_record_reports(
        report,
        args.output_dir,
    )
    print(f"wrote report: {paths['report']}")
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote markdown: {paths['markdown']}")
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DECISION_SEPARATE_RUNTIME_VISIBILITY_RECORD",
    "DECISION_STATUS_CANDIDATE_ONLY",
    "DEFAULT_HOLD_REVIEW_REPORT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_DECISION_RECORD_SCHEMA_ID",
    "build_strict_evidence_runtime_binding_visibility_decision_record",
    "render_strict_evidence_runtime_binding_visibility_decision_record_markdown",
    "write_strict_evidence_runtime_binding_visibility_decision_record_reports",
]
