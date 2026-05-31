"""Dry-run planner for StrictEvidence runtime visibility records.

Consumes the runtime binding visibility record contract and visibility decision
record, plans in-memory runtime visibility records for candidate rows, and
validates schema plus semantic contracts with zero runtime visibility JSONL
writes and zero parent-store mutation.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_runtime_binding_visibility_decision_record import (
    DECISION_SEPARATE_RUNTIME_VISIBILITY_RECORD,
    DECISION_STATUS_CANDIDATE_ONLY,
    STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_DECISION_RECORD_SCHEMA_ID,
)
from knowledge_hub.papers.strict_evidence_runtime_binding_visibility_record_contract import (
    DEFAULT_OUTPUT_DIR as DEFAULT_VISIBILITY_RECORD_CONTRACT_OUTPUT_DIR,
    DEFAULT_VISIBILITY_DECISION_REPORT_PATH,
    RUNTIME_VISIBILITY_DECISION,
    RUNTIME_VISIBILITY_POLICY_VERSION,
    RUNTIME_VISIBILITY_STATE,
    RUNTIME_VISIBILITY_STORE,
    STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_CONTRACT_SCHEMA_ID,
    STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_SCHEMA_ID,
    build_sample_runtime_visibility_record_from_decision_row,
    validate_runtime_visibility_record_semantics,
)


STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-runtime-binding-visibility-executor-dry-run.v1"
)

DRY_RUN_STATUS_READY = "dry_run_ready_runtime_visibility_record_only"
DRY_RUN_STATUS_BLOCKED_CONTRACT = "blocked_visibility_record_contract_not_ready"
DRY_RUN_STATUS_BLOCKED_DECISION = "blocked_visibility_decision_not_ready"
DRY_RUN_STATUS_BLOCKED_UNSUPPORTED_VISIBILITY_POLICY = (
    "blocked_unsupported_visibility_policy"
)
DRY_RUN_STATUS_BLOCKED_MISSING_RUNTIME_BINDING_RECORD_ID = (
    "blocked_missing_runtime_binding_record_id"
)
DRY_RUN_STATUS_BLOCKED_MISSING_CITATION_GRADE_RECORD_ID = (
    "blocked_missing_citation_grade_record_id"
)
DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID = "blocked_missing_strict_evidence_id"
DRY_RUN_STATUS_BLOCKED_MISSING_ELIGIBILITY_RECORD_ID = (
    "blocked_missing_eligibility_record_id"
)
DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_SPAN_ID = "blocked_missing_source_span_id"
DRY_RUN_STATUS_BLOCKED_MISSING_CANDIDATE_RECORD_ID = "blocked_missing_candidate_record_id"
DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA = "blocked_planned_visibility_record_schema_violation"
DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC = (
    "blocked_planned_visibility_record_semantic_violation"
)
DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

DEFAULT_VISIBILITY_RECORD_CONTRACT_REPORT_PATH = (
    DEFAULT_VISIBILITY_RECORD_CONTRACT_OUTPUT_DIR
    / "strict-evidence-runtime-binding-visibility-record-contract.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-runtime-binding-visibility-executor-dry-run"
    / "01-strict-evidence-runtime-binding-visibility-executor-dry-run"
)

EXPECTED_VISIBILITY_DECISION_ROWS = 99
EXPECTED_SECTION_VISIBILITY_DECISION_ROWS = 45
EXPECTED_FIGURE_CAPTION_VISIBILITY_DECISION_ROWS = 54
EXPECTED_PLANNED_RUNTIME_VISIBILITY_ROWS = 99
SUPPORTED_RUNTIME_VISIBILITY_ARTIFACT_TYPES = ("section", "figure")


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
        "plannedWriteTarget": RUNTIME_VISIBILITY_STORE,
        "writeEnabled": False,
        "runtimeVisibilityRecordWrite": False,
        "runtimeBindingRecordWrite": False,
        "runtimeEvidenceCreated": False,
        "runtimeVisible": False,
        "answerIntegrationVisible": False,
        "citationGradeRecordWrite": False,
        "citationGradeBooleanMutation": False,
        "eligibilityRecordWrite": False,
        "strictEvidenceStoreWrite": False,
        "sourceSpanStoreWrite": False,
        "strictEvidenceCreated": False,
        "strictEligibleMutation": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "vaultScan": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
        "manifestWrite": False,
    }


def _visibility_decision_row_flag_violations(decision_row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for field_name in (
        "runtimeBindingInPlaceMutationAllowed",
        "runtimeVisibleMutationAllowed",
        "answerIntegrationVisibleAllowed",
        "runtimeEvidenceAllowed",
        "parserRoutingAllowed",
        "answerIntegrationAllowed",
        "databaseMutationAllowed",
        "reindexOrReembedAllowed",
        "vaultScanAllowed",
    ):
        if _safe_bool(decision_row.get(field_name)):
            violations.append(f"visibility_decision_row.{field_name}_true")
    for field_name in (
        "runtimeVisibilityRecordWriteRows",
        "runtimeBindingRecordWriteRows",
        "runtimeVisibleRows",
        "answerIntegrationVisibleRows",
        "runtimeEvidenceCreatedRows",
        "parserRoutingChangedRows",
        "answerIntegrationChangedRows",
        "databaseMutationRows",
        "manifestWriteRows",
    ):
        if _safe_int(decision_row.get(field_name)) != 0:
            violations.append(
                f"visibility_decision_row.{field_name}="
                f"{_safe_int(decision_row.get(field_name))}_expected_0"
            )
    blockers = decision_row.get("decision_blockers")
    if isinstance(blockers, list):
        violations.extend(f"visibility_decision_row.blocker={item}" for item in blockers)
    return _dedupe(violations)


def _visibility_decision_row_is_candidate(decision_row: dict[str, Any]) -> bool:
    return (
        _safe_text(decision_row.get("decision_status")) == DECISION_STATUS_CANDIDATE_ONLY
        and _safe_text(decision_row.get("decision")) == DECISION_SEPARATE_RUNTIME_VISIBILITY_RECORD
        and _safe_bool(decision_row.get("runtimeVisibilityRecordRequired"))
        and _safe_bool(decision_row.get("runtimeVisibilityRecordAppendOnly"))
        and not _safe_bool(decision_row.get("runtimeBindingInPlaceMutationAllowed"))
        and not _safe_bool(decision_row.get("runtimeVisibleMutationAllowed"))
        and not _safe_bool(decision_row.get("answerIntegrationVisibleAllowed"))
        and not _safe_bool(decision_row.get("runtimeEvidenceAllowed"))
        and not _safe_bool(decision_row.get("parserRoutingAllowed"))
        and not _safe_bool(decision_row.get("answerIntegrationAllowed"))
        and not _safe_bool(decision_row.get("databaseMutationAllowed"))
        and not _safe_bool(decision_row.get("reindexOrReembedAllowed"))
        and not _safe_bool(decision_row.get("vaultScanAllowed"))
        and not _visibility_decision_row_flag_violations(decision_row)
    )


def _classify_dry_run_row(
    decision_row: dict[str, Any],
    *,
    input_schema_violations: list[str],
    contract_ready: bool,
    decision_ready: bool,
    run_id: str,
) -> tuple[str, list[str], dict[str, Any] | None]:
    if input_schema_violations:
        return DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA, list(input_schema_violations), None
    if not contract_ready:
        return DRY_RUN_STATUS_BLOCKED_CONTRACT, ["visibility_record_contract_not_ready"], None
    if not decision_ready:
        return DRY_RUN_STATUS_BLOCKED_DECISION, ["visibility_decision_record_not_ready"], None
    artifact_type = _safe_text(decision_row.get("artifact_type") or decision_row.get("artifactType"))
    if artifact_type not in SUPPORTED_RUNTIME_VISIBILITY_ARTIFACT_TYPES:
        return (
            DRY_RUN_STATUS_BLOCKED_UNSUPPORTED_VISIBILITY_POLICY,
            [f"artifact_type={artifact_type or 'missing'}_unsupported_for_runtime_visibility"],
            None,
        )
    if not _visibility_decision_row_is_candidate(decision_row):
        blockers = [
            _safe_text(item)
            for item in (decision_row.get("decision_blockers") or [])
            if _safe_text(item)
        ]
        blockers.extend(_visibility_decision_row_flag_violations(decision_row))
        blockers.append(
            "visibility_decision_status="
            f"{_safe_text(decision_row.get('decision_status')) or 'unknown'}"
        )
        return (
            DRY_RUN_STATUS_BLOCKED_UNSUPPORTED_VISIBILITY_POLICY,
            _dedupe(blockers),
            None,
        )

    required_ids = (
        (
            "runtimeBindingRecordId",
            DRY_RUN_STATUS_BLOCKED_MISSING_RUNTIME_BINDING_RECORD_ID,
        ),
        (
            "citationGradeRecordId",
            DRY_RUN_STATUS_BLOCKED_MISSING_CITATION_GRADE_RECORD_ID,
        ),
        ("strictEvidenceId", DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID),
        ("eligibilityRecordId", DRY_RUN_STATUS_BLOCKED_MISSING_ELIGIBILITY_RECORD_ID),
        ("sourceSpanId", DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_SPAN_ID),
        ("candidateRecordId", DRY_RUN_STATUS_BLOCKED_MISSING_CANDIDATE_RECORD_ID),
    )
    for field_name, status in required_ids:
        if not _safe_text(decision_row.get(field_name)):
            return status, [f"{field_name}_missing"], None

    planned_record = build_sample_runtime_visibility_record_from_decision_row(
        decision_row,
        run_id=run_id,
    )
    schema_validation = validate_payload(
        planned_record,
        STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_SCHEMA_ID,
        strict=True,
    )
    if not schema_validation.ok:
        return (
            DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA,
            [str(error) for error in schema_validation.errors],
            planned_record,
        )

    semantic_errors = validate_runtime_visibility_record_semantics(planned_record)
    if semantic_errors:
        return DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC, semantic_errors, planned_record

    return DRY_RUN_STATUS_READY, [], planned_record


def _planned_executor_key(decision_row: dict[str, Any]) -> str:
    runtime_binding_record_id = _safe_text(decision_row.get("runtimeBindingRecordId"))
    return f"runtime-visibility-executor:{runtime_binding_record_id or 'unknown'}"


def _dry_run_rows(
    decision_rows: list[dict[str, Any]],
    *,
    input_schema_violations: list[str],
    contract_ready: bool,
    decision_ready: bool,
    run_id: str,
) -> list[dict[str, Any]]:
    matrix = _no_mutation_policy_matrix()
    rows: list[dict[str, Any]] = []
    for index, decision_row in enumerate(decision_rows):
        source_row = dict(decision_row or {})
        dry_run_status, blockers, planned_record = _classify_dry_run_row(
            source_row,
            input_schema_violations=input_schema_violations,
            contract_ready=contract_ready,
            decision_ready=decision_ready,
            run_id=run_id,
        )
        ready = dry_run_status == DRY_RUN_STATUS_READY and planned_record is not None
        record = planned_record or {}
        rows.append(
            {
                "dry_run_row_id": (
                    "strict-evidence-runtime-binding-visibility-executor-dry-run:"
                    f"{index:04d}"
                ),
                "visibility_decision_record_row_id": _safe_text(
                    source_row.get("visibility_decision_record_row_id")
                ),
                "hold_row_id": _safe_text(source_row.get("hold_row_id")),
                "runtimeBindingRecordId": _safe_text(source_row.get("runtimeBindingRecordId")),
                "citationGradeRecordId": _safe_text(source_row.get("citationGradeRecordId")),
                "strictEvidenceId": _safe_text(source_row.get("strictEvidenceId")),
                "eligibilityRecordId": _safe_text(source_row.get("eligibilityRecordId")),
                "sourceSpanId": _safe_text(source_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(source_row.get("candidateRecordId")),
                "paper_id": _safe_text(source_row.get("paper_id") or source_row.get("paperId")),
                "artifact_type": _safe_text(
                    source_row.get("artifact_type") or source_row.get("artifactType")
                ),
                "sourceContentHash": _safe_text(record.get("sourceContentHash")),
                "dry_run_status": dry_run_status,
                "dry_run_blockers": _dedupe(blockers),
                "dryRunReadyRuntimeVisibilityRecordOnly": ready,
                "plannedExecutorKey": _planned_executor_key(source_row),
                "plannedWriteTarget": RUNTIME_VISIBILITY_STORE if ready else "",
                "plannedRuntimeVisibilityPolicyVersion": (
                    RUNTIME_VISIBILITY_POLICY_VERSION if ready else ""
                ),
                "plannedRuntimeVisibilityDecision": RUNTIME_VISIBILITY_DECISION if ready else "",
                "plannedRuntimeVisibilityState": RUNTIME_VISIBILITY_STATE if ready else "",
                "plannedRuntimeVisibilityRecord": record,
                "runtimeVisibilityRecordWrite": False,
                "runtimeBindingRecordWrite": False,
                "runtimeEvidenceCreated": False,
                "runtimeVisible": False,
                "answerIntegrationVisible": False,
                "citationGradeRecordWrite": False,
                "citationGradeBooleanMutation": False,
                "eligibilityRecordWrite": False,
                "strictEvidenceStoreWrite": False,
                "sourceSpanStoreWrite": False,
                "strictEligibleMutation": False,
                "strictEvidenceCreated": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "vaultScan": False,
                "reindexOrReembed": False,
                "canonicalParsedArtifactsWritten": False,
                "manifestWrite": False,
                "policyMatrix": matrix,
                "recommended_action": (
                    "queue_for_runtime_visibility_executor_apply_dry_run_review"
                    if ready
                    else "repair_runtime_visibility_executor_dry_run_input"
                ),
            }
        )
    return rows


def _count_rows(rows: list[dict[str, Any]], *, schema_violations: list[str]) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("dry_run_status")) for row in rows)
    ready_rows = [row for row in rows if _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_READY]
    by_artifact_type = Counter(_safe_text(row.get("artifact_type")) for row in rows)
    return {
        "inputRows": len(rows),
        "plannedRuntimeVisibilityRows": sum(
            1
            for row in rows
            if _safe_text(row.get("dry_run_status")) in {
                DRY_RUN_STATUS_READY,
                DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA,
                DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC,
            }
        ),
        "dryRunReadyRuntimeVisibilityRecordOnlyRows": int(by_status.get(DRY_RUN_STATUS_READY, 0)),
        "sectionVisibilityDecisionRows": int(by_artifact_type.get("section", 0)),
        "figureCaptionVisibilityDecisionRows": int(by_artifact_type.get("figure", 0)),
        "blockedVisibilityRecordContractNotReadyRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_CONTRACT, 0)
        ),
        "blockedVisibilityDecisionNotReadyRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_DECISION, 0)
        ),
        "blockedUnsupportedVisibilityPolicyRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_UNSUPPORTED_VISIBILITY_POLICY, 0)
        ),
        "blockedMissingRuntimeBindingRecordIdRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_RUNTIME_BINDING_RECORD_ID, 0)
        ),
        "blockedMissingCitationGradeRecordIdRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_CITATION_GRADE_RECORD_ID, 0)
        ),
        "blockedMissingStrictEvidenceIdRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID, 0)
        ),
        "blockedMissingEligibilityRecordIdRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_ELIGIBILITY_RECORD_ID, 0)
        ),
        "blockedMissingSourceSpanIdRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_SPAN_ID, 0)
        ),
        "blockedMissingCandidateRecordIdRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_CANDIDATE_RECORD_ID, 0)
        ),
        "blockedPlannedVisibilityRecordSchemaViolationRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA, 0)
        ),
        "blockedPlannedVisibilityRecordSemanticViolationRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC, 0)
        ),
        "blockedInputSchemaViolationRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA, 0)),
        "runtimeVisibilityRecordWriteRows": 0,
        "runtimeBindingRecordWriteRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "runtimeVisibleRows": 0,
        "answerIntegrationVisibleRows": 0,
        "answerIntegrationChangedRows": 0,
        "parserRoutingChangedRows": 0,
        "databaseMutationRows": 0,
        "citationGradeRecordWriteRows": 0,
        "citationGradeBooleanMutationRows": 0,
        "eligibilityRecordWriteRows": 0,
        "strictEligibleMutationRows": 0,
        "strictEvidenceWriteRows": 0,
        "strictEvidenceCreatedRows": 0,
        "sourceSpanUpdatedRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "manifestWriteRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaperId": dict(Counter(_safe_text(row.get("paper_id")) for row in ready_rows)),
        "byArtifactType": dict(Counter(_safe_text(row.get("artifact_type")) for row in ready_rows)),
        "byDryRunStatus": dict(by_status),
        "byRecommendedAction": dict(
            Counter(_safe_text(row.get("recommended_action")) for row in rows)
        ),
    }


def _contract_report_ready(
    contract_report: dict[str, Any],
    *,
    expected_planned_runtime_visibility_rows: int,
    expected_section_visibility_decision_rows: int,
    expected_figure_caption_visibility_decision_rows: int,
) -> bool:
    gate = contract_report.get("gate") if isinstance(contract_report.get("gate"), dict) else {}
    counts = (
        contract_report.get("counts")
        if isinstance(contract_report.get("counts"), dict)
        else {}
    )
    return (
        _safe_text(contract_report.get("status")) == "ok"
        and _safe_text(gate.get("decision"))
        == "strict_evidence_runtime_binding_visibility_record_contract_ready"
        and not _safe_bool(gate.get("executorReady"))
        and not _safe_bool(gate.get("runtimeVisibilityRecordWriteAllowed"))
        and not _safe_bool(gate.get("runtimeBindingRecordWriteAllowed"))
        and not _safe_bool(gate.get("runtimeMutationAllowed"))
        and not _safe_bool(gate.get("answerIntegrationVisibleAllowed"))
        and _safe_int(counts.get("plannedRuntimeVisibilityRows"))
        == expected_planned_runtime_visibility_rows
        and _safe_int(counts.get("sectionVisibilityDecisionRows"))
        == expected_section_visibility_decision_rows
        and _safe_int(counts.get("figureCaptionVisibilityDecisionRows"))
        == expected_figure_caption_visibility_decision_rows
        and _safe_int(counts.get("runtimeVisibilityRecordWriteRows")) == 0
        and _safe_int(counts.get("runtimeBindingRecordWriteRows")) == 0
        and _safe_int(counts.get("runtimeVisibleRows")) == 0
        and _safe_int(counts.get("answerIntegrationVisibleRows")) == 0
        and _safe_int(counts.get("runtimeEvidenceCreatedRows")) == 0
        and _safe_int(counts.get("parserRoutingChangedRows")) == 0
        and _safe_int(counts.get("answerIntegrationChangedRows")) == 0
        and _safe_int(counts.get("databaseMutationRows")) == 0
        and _safe_int(counts.get("vaultScanRows")) == 0
        and _safe_int(counts.get("schemaViolationCount")) == 0
        and _safe_int(counts.get("sampleRecordSemanticViolationRows")) == 0
    )


def _visibility_decision_report_ready(
    decision_report: dict[str, Any],
    *,
    expected_visibility_decision_rows: int,
    expected_section_visibility_decision_rows: int,
    expected_figure_caption_visibility_decision_rows: int,
) -> bool:
    gate = decision_report.get("gate") if isinstance(decision_report.get("gate"), dict) else {}
    counts = (
        decision_report.get("counts")
        if isinstance(decision_report.get("counts"), dict)
        else {}
    )
    return (
        _safe_text(decision_report.get("status")) == "ok"
        and _safe_bool(gate.get("runtimeBindingVisibilityDecisionRecordReady"))
        and _safe_text(gate.get("decision")) == DECISION_SEPARATE_RUNTIME_VISIBILITY_RECORD
        and not _safe_bool(gate.get("runtimeVisibilityRecordWriteAllowed"))
        and not _safe_bool(gate.get("runtimeVisibleMutationAllowed"))
        and not _safe_bool(gate.get("answerIntegrationVisibleAllowed"))
        and not _safe_bool(gate.get("runtimeEvidenceAllowed"))
        and not _safe_bool(gate.get("parserRoutingAllowed"))
        and not _safe_bool(gate.get("answerIntegrationAllowed"))
        and not _safe_bool(gate.get("databaseMutationAllowed"))
        and not _safe_bool(gate.get("vaultScanAllowed"))
        and _safe_int(counts.get("visibilityDecisionCandidateOnlyRows"))
        == expected_visibility_decision_rows
        and _safe_int(counts.get("sectionDecisionRows"))
        == expected_section_visibility_decision_rows
        and _safe_int(counts.get("figureCaptionDecisionRows"))
        == expected_figure_caption_visibility_decision_rows
        and _safe_int(counts.get("runtimeVisibilityRecordWriteRows")) == 0
        and _safe_int(counts.get("runtimeVisibleRows")) == 0
        and _safe_int(counts.get("answerIntegrationVisibleRows")) == 0
        and _safe_int(counts.get("runtimeEvidenceCreatedRows")) == 0
        and _safe_int(counts.get("parserRoutingChangedRows")) == 0
        and _safe_int(counts.get("answerIntegrationChangedRows")) == 0
        and _safe_int(counts.get("databaseMutationRows")) == 0
        and _safe_int(counts.get("schemaViolationCount")) == 0
    )


def build_strict_evidence_runtime_binding_visibility_executor_dry_run(
    *,
    visibility_record_contract_report_path: str | Path = (
        DEFAULT_VISIBILITY_RECORD_CONTRACT_REPORT_PATH
    ),
    visibility_decision_report_path: str | Path = DEFAULT_VISIBILITY_DECISION_REPORT_PATH,
    paper_ids: list[str] | None = None,
    run_id: str = "strict-evidence-runtime-binding-visibility-executor-dry-run-20260520",
    expected_visibility_decision_rows: int = EXPECTED_VISIBILITY_DECISION_ROWS,
    expected_section_visibility_decision_rows: int = EXPECTED_SECTION_VISIBILITY_DECISION_ROWS,
    expected_figure_caption_visibility_decision_rows: int = (
        EXPECTED_FIGURE_CAPTION_VISIBILITY_DECISION_ROWS
    ),
    expected_planned_runtime_visibility_rows: int = EXPECTED_PLANNED_RUNTIME_VISIBILITY_ROWS,
) -> dict[str, Any]:
    contract_path = Path(str(visibility_record_contract_report_path)).expanduser()
    decision_path = Path(str(visibility_decision_report_path)).expanduser()
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}
    warnings: list[str] = []
    input_schema_violations: list[str] = []

    contract_report = _read_json(contract_path)
    decision_report = _read_json(decision_path)

    if contract_report:
        validation = validate_payload(
            contract_report,
            STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_CONTRACT_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            input_schema_violations.extend(str(error) for error in validation.errors)
    else:
        input_schema_violations.append(
            "visibility_record_contract_report_missing_or_unreadable"
        )

    if decision_report:
        validation = validate_payload(
            decision_report,
            STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_DECISION_RECORD_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            input_schema_violations.extend(str(error) for error in validation.errors)
    else:
        input_schema_violations.append("visibility_decision_report_missing_or_unreadable")

    contract_ready = _contract_report_ready(
        contract_report,
        expected_planned_runtime_visibility_rows=expected_planned_runtime_visibility_rows,
        expected_section_visibility_decision_rows=expected_section_visibility_decision_rows,
        expected_figure_caption_visibility_decision_rows=(
            expected_figure_caption_visibility_decision_rows
        ),
    )
    decision_ready = _visibility_decision_report_ready(
        decision_report,
        expected_visibility_decision_rows=expected_visibility_decision_rows,
        expected_section_visibility_decision_rows=expected_section_visibility_decision_rows,
        expected_figure_caption_visibility_decision_rows=(
            expected_figure_caption_visibility_decision_rows
        ),
    )

    decision_rows = [
        row for row in decision_report.get("rows", []) if isinstance(row, dict)
    ] if decision_report else []
    if requested_papers:
        found = {_safe_text(row.get("paper_id")) for row in decision_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found:
            warnings.append("requested_paper_ids_not_found_in_visibility_decision_report")
        decision_rows = [
            row for row in decision_rows if _safe_text(row.get("paper_id")) in requested_papers
        ]

    schema_violations = _dedupe(input_schema_violations)
    rows = _dry_run_rows(
        decision_rows,
        input_schema_violations=schema_violations,
        contract_ready=contract_ready,
        decision_ready=decision_ready,
        run_id=run_id,
    )
    counts = _count_rows(rows, schema_violations=schema_violations)
    status = "ok"
    if (
        schema_violations
        or not rows
        or counts["dryRunReadyRuntimeVisibilityRecordOnlyRows"] != len(rows)
        or counts["dryRunReadyRuntimeVisibilityRecordOnlyRows"]
        != expected_visibility_decision_rows
        or counts["plannedRuntimeVisibilityRows"] != expected_planned_runtime_visibility_rows
        or counts["sectionVisibilityDecisionRows"] != expected_section_visibility_decision_rows
        or counts["figureCaptionVisibilityDecisionRows"]
        != expected_figure_caption_visibility_decision_rows
    ):
        status = "blocked"

    matrix = _no_mutation_policy_matrix()
    return {
        "schema": STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "visibilityRecordContractReportPath": str(contract_path),
            "visibilityRecordContractReportSchema": _safe_text(contract_report.get("schema"))
            if contract_report
            else "",
            "visibilityRecordContractReportStatus": _safe_text(contract_report.get("status"))
            if contract_report
            else "",
            "visibilityDecisionReportPath": str(decision_path),
            "visibilityDecisionReportSchema": _safe_text(decision_report.get("schema"))
            if decision_report
            else "",
            "visibilityDecisionReportStatus": _safe_text(decision_report.get("status"))
            if decision_report
            else "",
            "requestedPaperIds": sorted(requested_papers),
            "runId": run_id,
            "expectedVisibilityDecisionRows": expected_visibility_decision_rows,
            "expectedSectionVisibilityDecisionRows": expected_section_visibility_decision_rows,
            "expectedFigureCaptionVisibilityDecisionRows": (
                expected_figure_caption_visibility_decision_rows
            ),
            "expectedPlannedRuntimeVisibilityRows": expected_planned_runtime_visibility_rows,
        },
        "counts": counts,
        "dryRunOnlyPolicyMatrix": matrix,
        "gate": {
            "readyForRuntimeBindingVisibilityExecutorDryRun": status == "ok",
            "readyForRuntimeBindingVisibilityExecutorApply": False,
            "runtimeVisibilityRecordWriteAllowed": False,
            "runtimeBindingRecordWriteAllowed": False,
            "runtimeEvidenceReady": False,
            "runtimeVisibleAllowed": False,
            "answerIntegrationVisibleAllowed": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "citationGradeRecordWriteAllowed": False,
            "parentRecordMutationAllowed": False,
            "runManifestWriteAllowed": False,
            "runtimeMutationAllowed": False,
            "databaseMutationAllowed": False,
            "vaultScanAllowed": False,
            "schemaViolations": schema_violations,
            "decision": (
                "strict_evidence_runtime_binding_visibility_executor_dry_run_ready"
                if status == "ok"
                else "strict_evidence_runtime_binding_visibility_executor_dry_run_blocked"
            ),
            "recommendedNextTranche": (
                "strict_evidence_runtime_binding_visibility_executor_apply"
                if status == "ok"
                else "strict_evidence_runtime_binding_visibility_executor_dry_run_repair"
            ),
        },
        "policy": {"reportOnly": True, "dryRunOnly": True, **matrix},
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
            "dryRunOnlyPolicyMatrix",
            "gate",
            "policy",
            "warnings",
        )
        if key in report
    }


def render_strict_evidence_runtime_binding_visibility_executor_dry_run_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byDryRunStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Strict Evidence Runtime Binding Visibility Executor Dry Run",
            "",
            f"- status: {report.get('status', '')}",
            f"- decision: {gate.get('decision', '')}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- planned runtime visibility rows: {int(counts.get('plannedRuntimeVisibilityRows') or 0)}",
            f"- dry-run ready rows: {int(counts.get('dryRunReadyRuntimeVisibilityRecordOnlyRows') or 0)}",
            f"- section decision rows: {int(counts.get('sectionVisibilityDecisionRows') or 0)}",
            f"- figure-caption decision rows: {int(counts.get('figureCaptionVisibilityDecisionRows') or 0)}",
            f"- runtime visibility record writes: {int(counts.get('runtimeVisibilityRecordWriteRows') or 0)}",
            f"- runtime visible rows: {int(counts.get('runtimeVisibleRows') or 0)}",
            f"- answer integration visible rows: {int(counts.get('answerIntegrationVisibleRows') or 0)}",
            f"- runtime evidence rows: {int(counts.get('runtimeEvidenceCreatedRows') or 0)}",
            f"- parser routing changed rows: {int(counts.get('parserRoutingChangedRows') or 0)}",
            f"- answer integration changed rows: {int(counts.get('answerIntegrationChangedRows') or 0)}",
            f"- database mutation rows: {int(counts.get('databaseMutationRows') or 0)}",
            f"- vault scan rows: {int(counts.get('vaultScanRows') or 0)}",
            "",
            "## Dry-run status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- recommended next tranche: {gate.get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_runtime_binding_visibility_executor_dry_run_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-runtime-binding-visibility-executor-dry-run.json"
    summary_path = root / "strict-evidence-runtime-binding-visibility-executor-dry-run-summary.json"
    markdown_path = root / "strict-evidence-runtime-binding-visibility-executor-dry-run.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_runtime_binding_visibility_executor_dry_run_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Plan StrictEvidence runtime visibility records in dry-run mode without "
            "writing runtime visibility JSONL or mutating evidence stores."
        )
    )
    parser.add_argument(
        "--visibility-record-contract-report",
        default=str(DEFAULT_VISIBILITY_RECORD_CONTRACT_REPORT_PATH),
        help="Path to the runtime binding visibility record contract JSON report.",
    )
    parser.add_argument(
        "--visibility-decision-report",
        default=str(DEFAULT_VISIBILITY_DECISION_REPORT_PATH),
        help="Path to the runtime binding visibility decision JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument(
        "--run-id",
        default="strict-evidence-runtime-binding-visibility-executor-dry-run-20260520",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, summary, and markdown reports.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_runtime_binding_visibility_executor_dry_run(
        visibility_record_contract_report_path=args.visibility_record_contract_report,
        visibility_decision_report_path=args.visibility_decision_report,
        paper_ids=args.paper_id or None,
        run_id=args.run_id,
    )
    paths = write_strict_evidence_runtime_binding_visibility_executor_dry_run_reports(
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
    "DEFAULT_VISIBILITY_DECISION_REPORT_PATH",
    "DEFAULT_VISIBILITY_RECORD_CONTRACT_REPORT_PATH",
    "DRY_RUN_STATUS_BLOCKED_CONTRACT",
    "DRY_RUN_STATUS_BLOCKED_DECISION",
    "DRY_RUN_STATUS_BLOCKED_MISSING_CITATION_GRADE_RECORD_ID",
    "DRY_RUN_STATUS_BLOCKED_MISSING_RUNTIME_BINDING_RECORD_ID",
    "DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID",
    "DRY_RUN_STATUS_BLOCKED_UNSUPPORTED_VISIBILITY_POLICY",
    "DRY_RUN_STATUS_READY",
    "STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID",
    "build_strict_evidence_runtime_binding_visibility_executor_dry_run",
    "render_strict_evidence_runtime_binding_visibility_executor_dry_run_markdown",
    "write_strict_evidence_runtime_binding_visibility_executor_dry_run_reports",
]
