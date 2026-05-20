from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_runtime_binding_executor_dry_run import (
    DRY_RUN_STATUS_BLOCKED_CONTRACT,
    DRY_RUN_STATUS_BLOCKED_GATE_DESIGN,
    DRY_RUN_STATUS_BLOCKED_MISSING_CITATION_GRADE_RECORD_ID,
    DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID,
    DRY_RUN_STATUS_BLOCKED_UNSUPPORTED_RUNTIME_POLICY,
    DRY_RUN_STATUS_READY,
    STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_DRY_RUN_SCHEMA_ID,
    build_strict_evidence_runtime_binding_executor_dry_run,
    write_strict_evidence_runtime_binding_executor_dry_run_reports,
)
from knowledge_hub.papers.strict_evidence_runtime_binding_record_contract import (
    RUNTIME_BINDING_DECISION,
    RUNTIME_BINDING_POLICY_VERSION,
    RUNTIME_BINDING_STATE,
    RUNTIME_BINDING_STORE,
    STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID,
    build_strict_evidence_runtime_binding_record_contract,
    write_strict_evidence_runtime_binding_record_contract_reports,
)
from tests.test_strict_evidence_runtime_binding_record_contract import (
    _runtime_binding_gate_design_report_path,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _contract_report(gate_design_path: Path, tmp_path: Path) -> Path:
    contract = build_strict_evidence_runtime_binding_record_contract(
        runtime_binding_gate_design_report_path=gate_design_path,
        expected_input_rows=1,
        expected_planned_runtime_binding_rows=1,
    )
    contract_dir = tmp_path / "contract"
    write_strict_evidence_runtime_binding_record_contract_reports(contract, contract_dir)
    return contract_dir / "strict-evidence-runtime-binding-record-contract.json"


def test_runtime_binding_executor_dry_run_plans_records_without_writes(tmp_path: Path) -> None:
    gate_design_path = _runtime_binding_gate_design_report_path(tmp_path)
    contract_path = _contract_report(gate_design_path, tmp_path)

    payload = build_strict_evidence_runtime_binding_executor_dry_run(
        runtime_binding_record_contract_report_path=contract_path,
        runtime_binding_gate_design_report_path=gate_design_path,
        run_id="dry-run-test",
        expected_runtime_binding_gate_design_rows=1,
        expected_section_gate_design_rows=1,
        expected_figure_caption_gate_design_rows=0,
        expected_planned_runtime_binding_rows=1,
    )

    assert payload["schema"] == STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert payload["status"] == "ok"
    assert payload["counts"]["inputRows"] == 1
    assert payload["counts"]["plannedRuntimeBindingRows"] == 1
    assert payload["counts"]["dryRunReadyRuntimeBindingRecordOnlyRows"] == 1
    assert payload["counts"]["runtimeBindingRecordWriteRows"] == 0
    assert payload["counts"]["runtimeVisibleRows"] == 0
    assert payload["gate"]["readyForRuntimeBindingExecutorDryRun"] is True
    assert payload["gate"]["readyForRuntimeBindingExecutorApply"] is False
    assert payload["gate"]["recommendedNextTranche"] == "strict_evidence_runtime_binding_executor_apply"

    ready_row = payload["rows"][0]
    assert ready_row["dry_run_status"] == DRY_RUN_STATUS_READY
    assert ready_row["plannedWriteTarget"] == RUNTIME_BINDING_STORE
    assert ready_row["plannedRuntimeBindingPolicyVersion"] == RUNTIME_BINDING_POLICY_VERSION
    assert ready_row["plannedRuntimeBindingDecision"] == RUNTIME_BINDING_DECISION
    assert ready_row["plannedRuntimeBindingState"] == RUNTIME_BINDING_STATE

    planned = ready_row["plannedRuntimeBindingRecord"]
    assert planned["schema"] == STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID
    assert planned["runtimeBindingMutationApplied"] is False
    assert planned["runtimeEvidence"] is False
    assert planned["runtimeVisible"] is False
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_binding_executor_dry_run_blocks_when_contract_not_ready(tmp_path: Path) -> None:
    gate_design_path = _runtime_binding_gate_design_report_path(tmp_path)
    contract_path = _contract_report(gate_design_path, tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["status"] = "blocked"
    contract["gate"]["decision"] = "strict_evidence_runtime_binding_record_contract_blocked"
    _write_json(contract_path, contract)

    payload = build_strict_evidence_runtime_binding_executor_dry_run(
        runtime_binding_record_contract_report_path=contract_path,
        runtime_binding_gate_design_report_path=gate_design_path,
        expected_runtime_binding_gate_design_rows=1,
        expected_section_gate_design_rows=1,
        expected_figure_caption_gate_design_rows=0,
        expected_planned_runtime_binding_rows=1,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["blockedContractNotReadyRows"] == 1
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_CONTRACT


def test_runtime_binding_executor_dry_run_blocks_non_candidate_gate_design_rows(
    tmp_path: Path,
) -> None:
    gate_design_path = _runtime_binding_gate_design_report_path(tmp_path)
    gate_design = json.loads(gate_design_path.read_text(encoding="utf-8"))
    gate_design["rows"][0]["runtime_binding_gate_design_status"] = "blocked_citation_grade_hold_not_active"
    gate_design["rows"][0]["runtimeBindingGateDesignCandidateOnly"] = False
    _write_json(gate_design_path, gate_design)
    contract_path = _contract_report(gate_design_path, tmp_path)

    payload = build_strict_evidence_runtime_binding_executor_dry_run(
        runtime_binding_record_contract_report_path=contract_path,
        runtime_binding_gate_design_report_path=gate_design_path,
        expected_runtime_binding_gate_design_rows=1,
        expected_section_gate_design_rows=1,
        expected_figure_caption_gate_design_rows=0,
        expected_planned_runtime_binding_rows=1,
    )

    assert payload["status"] == "blocked"
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_UNSUPPORTED_RUNTIME_POLICY


def test_runtime_binding_executor_dry_run_blocks_when_gate_design_not_ready(tmp_path: Path) -> None:
    gate_design_path = _runtime_binding_gate_design_report_path(tmp_path)
    contract_path = _contract_report(gate_design_path, tmp_path)
    gate_design = json.loads(gate_design_path.read_text(encoding="utf-8"))
    gate_design["status"] = "blocked"
    gate_design["gate"]["runtimeBindingGateDesignReady"] = False
    _write_json(gate_design_path, gate_design)

    payload = build_strict_evidence_runtime_binding_executor_dry_run(
        runtime_binding_record_contract_report_path=contract_path,
        runtime_binding_gate_design_report_path=gate_design_path,
        expected_runtime_binding_gate_design_rows=1,
        expected_section_gate_design_rows=1,
        expected_figure_caption_gate_design_rows=0,
        expected_planned_runtime_binding_rows=1,
    )

    assert payload["status"] == "blocked"
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_GATE_DESIGN


def test_runtime_binding_executor_dry_run_blocks_missing_parent_ids(tmp_path: Path) -> None:
    gate_design_path = _runtime_binding_gate_design_report_path(tmp_path)
    gate_design = json.loads(gate_design_path.read_text(encoding="utf-8"))
    gate_design["rows"][0]["strictEvidenceId"] = ""
    _write_json(gate_design_path, gate_design)
    contract_path = _contract_report(gate_design_path, tmp_path)

    payload = build_strict_evidence_runtime_binding_executor_dry_run(
        runtime_binding_record_contract_report_path=contract_path,
        runtime_binding_gate_design_report_path=gate_design_path,
        expected_runtime_binding_gate_design_rows=1,
        expected_section_gate_design_rows=1,
        expected_figure_caption_gate_design_rows=0,
        expected_planned_runtime_binding_rows=1,
    )

    assert payload["status"] == "blocked"
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID

    gate_design["rows"][0]["strictEvidenceId"] = "strict-evidence:paper-1:section:1"
    gate_design["rows"][0]["citationGradeRecordId"] = ""
    _write_json(gate_design_path, gate_design)
    payload = build_strict_evidence_runtime_binding_executor_dry_run(
        runtime_binding_record_contract_report_path=contract_path,
        runtime_binding_gate_design_report_path=gate_design_path,
        expected_runtime_binding_gate_design_rows=1,
        expected_section_gate_design_rows=1,
        expected_figure_caption_gate_design_rows=0,
        expected_planned_runtime_binding_rows=1,
    )
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_MISSING_CITATION_GRADE_RECORD_ID


def test_runtime_binding_executor_dry_run_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    gate_design_path = _runtime_binding_gate_design_report_path(tmp_path)
    contract_path = _contract_report(gate_design_path, tmp_path)

    payload = build_strict_evidence_runtime_binding_executor_dry_run(
        runtime_binding_record_contract_report_path=contract_path,
        runtime_binding_gate_design_report_path=gate_design_path,
        expected_runtime_binding_gate_design_rows=1,
        expected_section_gate_design_rows=1,
        expected_figure_caption_gate_design_rows=0,
        expected_planned_runtime_binding_rows=1,
    )
    paths = write_strict_evidence_runtime_binding_executor_dry_run_reports(
        payload,
        tmp_path / "reports",
    )

    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert report["schema"] == STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert summary["counts"]["dryRunReadyRuntimeBindingRecordOnlyRows"] == 1
    assert "Strict Evidence Runtime Binding Executor Dry Run" in markdown
    assert validate_payload(
        report,
        STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_binding_executor_dry_run_integrated_measured_local_report() -> None:
    payload = build_strict_evidence_runtime_binding_executor_dry_run()
    assert payload["status"] == "ok"
    assert payload["counts"]["inputRows"] == 99
    assert payload["counts"]["plannedRuntimeBindingRows"] == 99
    assert payload["counts"]["dryRunReadyRuntimeBindingRecordOnlyRows"] == 99
    assert payload["counts"]["blockedContractNotReadyRows"] == 0
    assert payload["counts"]["blockedRuntimeBindingGateDesignNotReadyRows"] == 0
    assert payload["counts"]["blockedUnsupportedRuntimePolicyRows"] == 0
    assert payload["counts"]["blockedMissingStrictEvidenceIdRows"] == 0
    assert payload["counts"]["blockedMissingSourceSpanIdRows"] == 0
    assert payload["counts"]["blockedMissingCandidateRecordIdRows"] == 0
    assert payload["counts"]["blockedMissingEligibilityRecordIdRows"] == 0
    assert payload["counts"]["blockedMissingCitationGradeRecordIdRows"] == 0
    assert payload["counts"]["blockedPlannedRecordSchemaViolationRows"] == 0
    assert payload["counts"]["blockedPlannedRecordSemanticViolationRows"] == 0
    assert payload["counts"]["blockedInputSchemaViolationRows"] == 0
    assert payload["counts"]["runtimeBindingRecordWriteRows"] == 0
    assert payload["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert payload["counts"]["runtimeVisibleRows"] == 0
    assert payload["counts"]["answerIntegrationChangedRows"] == 0
    assert payload["counts"]["parserRoutingChangedRows"] == 0
    assert payload["counts"]["databaseMutationRows"] == 0
    assert payload["counts"]["reindexOrReembedRows"] == 0
    assert payload["counts"]["vaultScanRows"] == 0
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok
