"""Dry-run planner for StrictEvidence runtime binding record writes.

Consumes the runtime binding record contract report and the citation-grade
runtime binding gate design report, plans in-memory runtime binding records for
candidate rows, and validates schema plus semantic contracts with zero
filesystem writes.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_citation_grade_runtime_binding_gate_design import (
    RUNTIME_BINDING_GATE_DESIGN_STATUS_CANDIDATE_ONLY,
    STRICT_EVIDENCE_CITATION_GRADE_RUNTIME_BINDING_GATE_DESIGN_SCHEMA_ID,
)
from knowledge_hub.papers.strict_evidence_runtime_binding_record_contract import (
    RUNTIME_BINDING_DECISION,
    RUNTIME_BINDING_POLICY_VERSION,
    RUNTIME_BINDING_STATE,
    STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID,
    STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID,
    RUNTIME_BINDING_STORE,
    build_sample_runtime_binding_record_from_gate_design_row,
    validate_runtime_binding_record_semantics,
)


STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-runtime-binding-executor-dry-run.v1"
)

DRY_RUN_STATUS_READY = "dry_run_ready_runtime_binding_record_only"
DRY_RUN_STATUS_BLOCKED_CONTRACT = "blocked_contract_not_ready"
DRY_RUN_STATUS_BLOCKED_GATE_DESIGN = "blocked_runtime_binding_gate_design_not_ready"
DRY_RUN_STATUS_BLOCKED_UNSUPPORTED_RUNTIME_POLICY = "blocked_unsupported_runtime_policy"
DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID = "blocked_missing_strict_evidence_id"
DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_SPAN_ID = "blocked_missing_source_span_id"
DRY_RUN_STATUS_BLOCKED_MISSING_CANDIDATE_RECORD_ID = "blocked_missing_candidate_record_id"
DRY_RUN_STATUS_BLOCKED_MISSING_ELIGIBILITY_RECORD_ID = "blocked_missing_eligibility_record_id"
DRY_RUN_STATUS_BLOCKED_MISSING_CITATION_GRADE_RECORD_ID = "blocked_missing_citation_grade_record_id"
DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA = "blocked_planned_record_schema_violation"
DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC = "blocked_planned_record_semantic_violation"
DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

DEFAULT_RUNTIME_BINDING_RECORD_CONTRACT_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-runtime-binding-record-contract"
    / "01-strict-evidence-runtime-binding-record-contract"
    / "strict-evidence-runtime-binding-record-contract.json"
)

DEFAULT_RUNTIME_BINDING_GATE_DESIGN_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-citation-grade-runtime-binding-gate-design"
    / "01-strict-evidence-citation-grade-runtime-binding-gate-design"
    / "strict-evidence-citation-grade-runtime-binding-gate-design.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-runtime-binding-executor-dry-run"
    / "01-strict-evidence-runtime-binding-executor-dry-run"
)

EXPECTED_RUNTIME_BINDING_GATE_DESIGN_ROWS = 99
EXPECTED_SECTION_GATE_DESIGN_ROWS = 45
EXPECTED_FIGURE_CAPTION_GATE_DESIGN_ROWS = 54
EXPECTED_PLANNED_RUNTIME_BINDING_ROWS = 99


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
        "plannedWriteTarget": RUNTIME_BINDING_STORE,
        "writeEnabled": False,
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


def _gate_design_row_is_candidate(design_row: dict[str, Any]) -> bool:
    return (
        _safe_text(design_row.get("runtime_binding_gate_design_status"))
        == RUNTIME_BINDING_GATE_DESIGN_STATUS_CANDIDATE_ONLY
        and _safe_bool(design_row.get("runtimeBindingGateDesignCandidateOnly"))
        and _safe_bool(design_row.get("runtimeBindingRecordRequired"))
        and not _safe_bool(design_row.get("runtimeBindingRecordWriteAllowed"))
        and not _safe_bool(design_row.get("runtimeEvidenceAllowed"))
        and not _safe_bool(design_row.get("runtimeVisibleAllowed"))
        and not _safe_bool(design_row.get("parserRoutingAllowed"))
        and not _safe_bool(design_row.get("answerIntegrationAllowed"))
    )


def _gate_design_row_flag_violations(design_row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for field_name in (
        "strictEligible",
        "strictEvidenceCreated",
        "citationGrade",
        "runtimeEvidence",
        "runtimeVisible",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "databaseMutation",
    ):
        if _safe_bool(design_row.get(field_name)):
            violations.append(f"gate_design_row.{field_name}_true")
    for field_name in (
        "runtimeBindingRecordWriteAllowed",
        "runtimeEvidenceAllowed",
        "runtimeVisibleAllowed",
        "parserRoutingAllowed",
        "answerIntegrationAllowed",
        "databaseMutationAllowed",
        "vaultScanAllowed",
    ):
        if _safe_bool(design_row.get(field_name)):
            violations.append(f"gate_design_row.{field_name}_true")
    return violations


def _classify_dry_run_row(
    design_row: dict[str, Any],
    *,
    input_schema_violations: list[str],
    contract_ready: bool,
    gate_design_ready: bool,
    run_id: str,
) -> tuple[str, list[str], dict[str, Any] | None]:
    if input_schema_violations:
        return DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA, list(input_schema_violations), None
    if not contract_ready:
        return DRY_RUN_STATUS_BLOCKED_CONTRACT, ["runtime_binding_record_contract_not_ready"], None
    if not gate_design_ready:
        return (
            DRY_RUN_STATUS_BLOCKED_GATE_DESIGN,
            ["runtime_binding_gate_design_not_ready"],
            None,
        )
    if not _gate_design_row_is_candidate(design_row):
        blockers = [
            _safe_text(item)
            for item in (design_row.get("runtime_binding_gate_design_blockers") or [])
        ]
        blockers.append(
            "runtime_binding_gate_design_status="
            f"{_safe_text(design_row.get('runtime_binding_gate_design_status')) or 'unknown'}"
        )
        return DRY_RUN_STATUS_BLOCKED_UNSUPPORTED_RUNTIME_POLICY, _dedupe(blockers), None

    strict_evidence_id = _safe_text(design_row.get("strictEvidenceId"))
    source_span_id = _safe_text(design_row.get("sourceSpanId"))
    candidate_record_id = _safe_text(design_row.get("candidateRecordId"))
    eligibility_record_id = _safe_text(design_row.get("eligibilityRecordId"))
    citation_grade_record_id = _safe_text(design_row.get("citationGradeRecordId"))
    if not strict_evidence_id:
        return DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID, ["strictEvidenceId_missing"], None
    if not source_span_id:
        return DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_SPAN_ID, ["sourceSpanId_missing"], None
    if not candidate_record_id:
        return DRY_RUN_STATUS_BLOCKED_MISSING_CANDIDATE_RECORD_ID, ["candidateRecordId_missing"], None
    if not eligibility_record_id:
        return DRY_RUN_STATUS_BLOCKED_MISSING_ELIGIBILITY_RECORD_ID, ["eligibilityRecordId_missing"], None
    if not citation_grade_record_id:
        return (
            DRY_RUN_STATUS_BLOCKED_MISSING_CITATION_GRADE_RECORD_ID,
            ["citationGradeRecordId_missing"],
            None,
        )

    flag_violations = _gate_design_row_flag_violations(design_row)
    if flag_violations:
        return DRY_RUN_STATUS_BLOCKED_UNSUPPORTED_RUNTIME_POLICY, flag_violations, None

    planned_record = build_sample_runtime_binding_record_from_gate_design_row(
        design_row,
        run_id=run_id,
    )
    schema_validation = validate_payload(
        planned_record,
        STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID,
        strict=True,
    )
    if not schema_validation.ok:
        return (
            DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA,
            [str(error) for error in schema_validation.errors],
            planned_record,
        )

    semantic_errors = validate_runtime_binding_record_semantics(planned_record)
    if semantic_errors:
        return DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC, semantic_errors, planned_record

    return DRY_RUN_STATUS_READY, [], planned_record


def _planned_executor_key(design_row: dict[str, Any]) -> str:
    strict_evidence_id = _safe_text(design_row.get("strictEvidenceId"))
    return f"runtime-binding-executor:{strict_evidence_id or 'unknown'}"


def _dry_run_rows(
    design_rows: list[dict[str, Any]],
    *,
    input_schema_violations: list[str],
    contract_ready: bool,
    gate_design_ready: bool,
    run_id: str,
) -> list[dict[str, Any]]:
    matrix = _no_mutation_policy_matrix()
    rows: list[dict[str, Any]] = []
    for index, design_row in enumerate(design_rows):
        source_row = dict(design_row or {})
        dry_run_status, blockers, planned_record = _classify_dry_run_row(
            source_row,
            input_schema_violations=input_schema_violations,
            contract_ready=contract_ready,
            gate_design_ready=gate_design_ready,
            run_id=run_id,
        )
        ready = dry_run_status == DRY_RUN_STATUS_READY and planned_record is not None
        record = planned_record or {}
        rows.append(
            {
                "dry_run_row_id": f"strict-evidence-runtime-binding-executor-dry-run:{index:04d}",
                "runtime_binding_gate_design_row_id": _safe_text(
                    source_row.get("runtime_binding_gate_design_row_id")
                ),
                "hold_row_id": _safe_text(source_row.get("hold_row_id")),
                "readback_row_id": _safe_text(source_row.get("readback_row_id")),
                "apply_row_id": _safe_text(source_row.get("apply_row_id")),
                "strictEvidenceId": _safe_text(source_row.get("strictEvidenceId")),
                "sourceSpanId": _safe_text(source_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(source_row.get("candidateRecordId")),
                "eligibilityRecordId": _safe_text(source_row.get("eligibilityRecordId")),
                "citationGradeRecordId": _safe_text(source_row.get("citationGradeRecordId")),
                "paper_id": _safe_text(source_row.get("paper_id")),
                "artifact_type": _safe_text(source_row.get("artifact_type")),
                "dry_run_status": dry_run_status,
                "dry_run_blockers": _dedupe(blockers),
                "dryRunReadyRuntimeBindingRecordOnly": ready,
                "plannedExecutorKey": _planned_executor_key(source_row),
                "plannedWriteTarget": RUNTIME_BINDING_STORE if ready else "",
                "plannedRuntimeBindingPolicyVersion": (
                    RUNTIME_BINDING_POLICY_VERSION if ready else ""
                ),
                "plannedRuntimeBindingDecision": RUNTIME_BINDING_DECISION if ready else "",
                "plannedRuntimeBindingState": RUNTIME_BINDING_STATE if ready else "",
                "plannedRuntimeBindingRecord": record,
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
                    "queue_for_runtime_binding_executor_apply_dry_run_review"
                    if ready
                    else "repair_runtime_binding_executor_dry_run_input"
                ),
            }
        )
    return rows


def _count_rows(rows: list[dict[str, Any]], *, schema_violations: list[str]) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("dry_run_status")) for row in rows)
    ready_rows = [row for row in rows if _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_READY]
    return {
        "inputRows": len(rows),
        "plannedRuntimeBindingRows": sum(
            1
            for row in rows
            if _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_READY
            or _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA
            or _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC
        ),
        "dryRunReadyRuntimeBindingRecordOnlyRows": int(by_status.get(DRY_RUN_STATUS_READY, 0)),
        "blockedContractNotReadyRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_CONTRACT, 0)),
        "blockedRuntimeBindingGateDesignNotReadyRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_GATE_DESIGN, 0)
        ),
        "blockedUnsupportedRuntimePolicyRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_UNSUPPORTED_RUNTIME_POLICY, 0)
        ),
        "blockedMissingStrictEvidenceIdRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID, 0)
        ),
        "blockedMissingSourceSpanIdRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_SPAN_ID, 0)
        ),
        "blockedMissingCandidateRecordIdRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_CANDIDATE_RECORD_ID, 0)
        ),
        "blockedMissingEligibilityRecordIdRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_ELIGIBILITY_RECORD_ID, 0)
        ),
        "blockedMissingCitationGradeRecordIdRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_CITATION_GRADE_RECORD_ID, 0)
        ),
        "blockedPlannedRecordSchemaViolationRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA, 0)
        ),
        "blockedPlannedRecordSemanticViolationRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC, 0)
        ),
        "blockedInputSchemaViolationRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA, 0)),
        "runtimeBindingRecordWriteRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "runtimeVisibleRows": 0,
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
        "byRecommendedAction": dict(Counter(_safe_text(row.get("recommended_action")) for row in rows)),
    }


def build_strict_evidence_runtime_binding_executor_dry_run(
    *,
    runtime_binding_record_contract_report_path: str | Path = (
        DEFAULT_RUNTIME_BINDING_RECORD_CONTRACT_REPORT_PATH
    ),
    runtime_binding_gate_design_report_path: str | Path = (
        DEFAULT_RUNTIME_BINDING_GATE_DESIGN_REPORT_PATH
    ),
    paper_ids: list[str] | None = None,
    run_id: str = "strict-evidence-runtime-binding-executor-dry-run-20260520",
    expected_runtime_binding_gate_design_rows: int = EXPECTED_RUNTIME_BINDING_GATE_DESIGN_ROWS,
    expected_section_gate_design_rows: int = EXPECTED_SECTION_GATE_DESIGN_ROWS,
    expected_figure_caption_gate_design_rows: int = EXPECTED_FIGURE_CAPTION_GATE_DESIGN_ROWS,
    expected_planned_runtime_binding_rows: int = EXPECTED_PLANNED_RUNTIME_BINDING_ROWS,
) -> dict[str, Any]:
    contract_path = Path(str(runtime_binding_record_contract_report_path)).expanduser()
    gate_design_path = Path(str(runtime_binding_gate_design_report_path)).expanduser()
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}
    warnings: list[str] = []
    input_schema_violations: list[str] = []

    contract_report = _read_json(contract_path)
    gate_design_report = _read_json(gate_design_path)

    contract_ready = False
    if contract_report:
        validation = validate_payload(
            contract_report,
            STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            input_schema_violations.extend(str(error) for error in validation.errors)
        gate = contract_report.get("gate") if isinstance(contract_report.get("gate"), dict) else {}
        contract_counts = (
            contract_report.get("counts") if isinstance(contract_report.get("counts"), dict) else {}
        )
        contract_ready = (
            _safe_text(contract_report.get("status")) == "ok"
            and _safe_text(gate.get("decision"))
            == "strict_evidence_runtime_binding_record_contract_ready"
            and not _safe_bool(gate.get("executorReady"))
            and not _safe_bool(gate.get("runtimeBindingRecordWriteAllowed"))
            and _safe_int(contract_counts.get("plannedRuntimeBindingRows"))
            == expected_planned_runtime_binding_rows
        )
    else:
        input_schema_violations.append(
            "runtime_binding_record_contract_report_missing_or_unreadable"
        )

    gate_design_ready = False
    design_rows: list[dict[str, Any]] = []
    if gate_design_report:
        validation = validate_payload(
            gate_design_report,
            STRICT_EVIDENCE_CITATION_GRADE_RUNTIME_BINDING_GATE_DESIGN_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            input_schema_violations.extend(str(error) for error in validation.errors)
        gate = gate_design_report.get("gate") if isinstance(gate_design_report.get("gate"), dict) else {}
        counts = gate_design_report.get("counts") if isinstance(gate_design_report.get("counts"), dict) else {}
        gate_design_ready = (
            _safe_text(gate_design_report.get("status")) == "ok"
            and _safe_bool(gate.get("runtimeBindingGateDesignReady"))
            and _safe_text(gate.get("decision"))
            == "strict_evidence_runtime_binding_gate_design_candidate_only"
            and _safe_int(counts.get("runtimeBindingGateDesignCandidateOnlyRows"))
            == expected_runtime_binding_gate_design_rows
            and _safe_int(counts.get("sectionRuntimeBindingGateDesignRows"))
            == expected_section_gate_design_rows
            and _safe_int(counts.get("figureCaptionRuntimeBindingGateDesignRows"))
            == expected_figure_caption_gate_design_rows
            and _safe_int(counts.get("runtimeBindingRecordWriteRows")) == 0
            and _safe_int(counts.get("runtimeEvidenceCreatedRows")) == 0
            and _safe_int(counts.get("answerIntegrationChangedRows")) == 0
        )
        design_rows = [row for row in gate_design_report.get("rows", []) if isinstance(row, dict)]
    else:
        input_schema_violations.append(
            "runtime_binding_gate_design_report_missing_or_unreadable"
        )

    if requested_papers:
        found = {_safe_text(row.get("paper_id")) for row in design_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found:
            warnings.append("requested_paper_ids_not_found_in_runtime_binding_gate_design")
        design_rows = [row for row in design_rows if _safe_text(row.get("paper_id")) in requested_papers]

    rows = _dry_run_rows(
        design_rows,
        input_schema_violations=_dedupe(input_schema_violations),
        contract_ready=contract_ready,
        gate_design_ready=gate_design_ready,
        run_id=run_id,
    )
    counts = _count_rows(rows, schema_violations=_dedupe(input_schema_violations))
    status = "ok"
    if (
        input_schema_violations
        or not rows
        or counts["dryRunReadyRuntimeBindingRecordOnlyRows"] != len(rows)
        or counts["dryRunReadyRuntimeBindingRecordOnlyRows"] != expected_runtime_binding_gate_design_rows
    ):
        status = "blocked"

    matrix = _no_mutation_policy_matrix()
    return {
        "schema": STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "runtimeBindingRecordContractReportPath": str(contract_path),
            "runtimeBindingRecordContractReportSchema": _safe_text(contract_report.get("schema"))
            if contract_report
            else "",
            "runtimeBindingRecordContractReportStatus": _safe_text(contract_report.get("status"))
            if contract_report
            else "",
            "runtimeBindingGateDesignReportPath": str(gate_design_path),
            "runtimeBindingGateDesignReportSchema": _safe_text(gate_design_report.get("schema"))
            if gate_design_report
            else "",
            "runtimeBindingGateDesignReportStatus": _safe_text(gate_design_report.get("status"))
            if gate_design_report
            else "",
            "requestedPaperIds": sorted(requested_papers),
            "runId": run_id,
            "expectedRuntimeBindingGateDesignRows": expected_runtime_binding_gate_design_rows,
            "expectedSectionGateDesignRows": expected_section_gate_design_rows,
            "expectedFigureCaptionGateDesignRows": expected_figure_caption_gate_design_rows,
            "expectedPlannedRuntimeBindingRows": expected_planned_runtime_binding_rows,
        },
        "counts": counts,
        "dryRunOnlyPolicyMatrix": matrix,
        "gate": {
            "readyForRuntimeBindingExecutorDryRun": status == "ok",
            "readyForRuntimeBindingExecutorApply": False,
            "runtimeBindingRecordWriteAllowed": False,
            "runtimeEvidenceReady": False,
            "runtimeVisibleAllowed": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "citationGradeRecordWriteAllowed": False,
            "parentRecordMutationAllowed": False,
            "runManifestWriteAllowed": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "strict_evidence_runtime_binding_executor_dry_run_ready"
                if status == "ok"
                else "strict_evidence_runtime_binding_executor_dry_run_blocked"
            ),
            "recommendedNextTranche": (
                "strict_evidence_runtime_binding_executor_apply"
                if status == "ok"
                else "strict_evidence_runtime_binding_executor_dry_run_repair"
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


def render_strict_evidence_runtime_binding_executor_dry_run_markdown(
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
            "# Strict Evidence Runtime Binding Executor Dry Run",
            "",
            f"- status: {report.get('status', '')}",
            f"- decision: {gate.get('decision', '')}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- planned runtime binding rows: {int(counts.get('plannedRuntimeBindingRows') or 0)}",
            f"- dry-run ready rows: {int(counts.get('dryRunReadyRuntimeBindingRecordOnlyRows') or 0)}",
            f"- runtime binding record writes: {int(counts.get('runtimeBindingRecordWriteRows') or 0)}",
            f"- runtime evidence rows: {int(counts.get('runtimeEvidenceCreatedRows') or 0)}",
            f"- runtime visible rows: {int(counts.get('runtimeVisibleRows') or 0)}",
            "",
            "## Dry-run status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- recommended next tranche: {gate.get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_runtime_binding_executor_dry_run_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-runtime-binding-executor-dry-run.json"
    summary_path = root / "strict-evidence-runtime-binding-executor-dry-run-summary.json"
    markdown_path = root / "strict-evidence-runtime-binding-executor-dry-run.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_runtime_binding_executor_dry_run_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Plan StrictEvidence runtime binding record writes in dry-run mode without "
            "writing runtime binding records or mutating evidence stores."
        )
    )
    parser.add_argument(
        "--runtime-binding-record-contract-report",
        default=str(DEFAULT_RUNTIME_BINDING_RECORD_CONTRACT_REPORT_PATH),
        help="Path to the runtime binding record contract JSON report.",
    )
    parser.add_argument(
        "--runtime-binding-gate-design-report",
        default=str(DEFAULT_RUNTIME_BINDING_GATE_DESIGN_REPORT_PATH),
        help="Path to the citation-grade runtime binding gate design JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--run-id", default="strict-evidence-runtime-binding-executor-dry-run-20260520")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, summary, and markdown reports.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_runtime_binding_executor_dry_run(
        runtime_binding_record_contract_report_path=args.runtime_binding_record_contract_report,
        runtime_binding_gate_design_report_path=args.runtime_binding_gate_design_report,
        paper_ids=args.paper_id or None,
        run_id=args.run_id,
    )
    paths = write_strict_evidence_runtime_binding_executor_dry_run_reports(report, args.output_dir)
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
    "DEFAULT_RUNTIME_BINDING_GATE_DESIGN_REPORT_PATH",
    "DEFAULT_RUNTIME_BINDING_RECORD_CONTRACT_REPORT_PATH",
    "DRY_RUN_STATUS_BLOCKED_CONTRACT",
    "DRY_RUN_STATUS_BLOCKED_GATE_DESIGN",
    "DRY_RUN_STATUS_BLOCKED_MISSING_CITATION_GRADE_RECORD_ID",
    "DRY_RUN_STATUS_BLOCKED_MISSING_ELIGIBILITY_RECORD_ID",
    "DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID",
    "DRY_RUN_STATUS_BLOCKED_UNSUPPORTED_RUNTIME_POLICY",
    "DRY_RUN_STATUS_READY",
    "STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_DRY_RUN_SCHEMA_ID",
    "build_strict_evidence_runtime_binding_executor_dry_run",
    "render_strict_evidence_runtime_binding_executor_dry_run_markdown",
    "write_strict_evidence_runtime_binding_executor_dry_run_reports",
]
