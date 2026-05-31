from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_runtime_binding_visibility_executor_dry_run import (
    DRY_RUN_STATUS_BLOCKED_CONTRACT,
    DRY_RUN_STATUS_BLOCKED_DECISION,
    DRY_RUN_STATUS_BLOCKED_MISSING_RUNTIME_BINDING_RECORD_ID,
    DRY_RUN_STATUS_BLOCKED_UNSUPPORTED_VISIBILITY_POLICY,
    DRY_RUN_STATUS_READY,
    STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID,
    build_strict_evidence_runtime_binding_visibility_executor_dry_run,
    write_strict_evidence_runtime_binding_visibility_executor_dry_run_reports,
)
from knowledge_hub.papers.strict_evidence_runtime_binding_visibility_record_contract import (
    RUNTIME_VISIBILITY_DECISION,
    RUNTIME_VISIBILITY_POLICY_VERSION,
    RUNTIME_VISIBILITY_STATE,
    RUNTIME_VISIBILITY_STORE,
    STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_SCHEMA_ID,
    build_strict_evidence_runtime_binding_visibility_record_contract,
    write_strict_evidence_runtime_binding_visibility_record_contract_reports,
)
from tests.test_strict_evidence_runtime_binding_visibility_record_contract import (
    _visibility_decision_report_path,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _contract_report(decision_path: Path, tmp_path: Path) -> Path:
    contract = build_strict_evidence_runtime_binding_visibility_record_contract(
        visibility_decision_report_path=decision_path,
        expected_input_rows=1,
        expected_planned_runtime_visibility_rows=1,
    )
    contract_dir = tmp_path / "contract"
    write_strict_evidence_runtime_binding_visibility_record_contract_reports(
        contract,
        contract_dir,
    )
    return contract_dir / "strict-evidence-runtime-binding-visibility-record-contract.json"


def test_runtime_visibility_executor_dry_run_plans_records_without_writes(
    tmp_path: Path,
) -> None:
    decision_path = _visibility_decision_report_path(tmp_path)
    contract_path = _contract_report(decision_path, tmp_path)

    payload = build_strict_evidence_runtime_binding_visibility_executor_dry_run(
        visibility_record_contract_report_path=contract_path,
        visibility_decision_report_path=decision_path,
        run_id="visibility-dry-run-test",
        expected_visibility_decision_rows=1,
        expected_section_visibility_decision_rows=1,
        expected_figure_caption_visibility_decision_rows=0,
        expected_planned_runtime_visibility_rows=1,
    )

    assert payload["schema"] == STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert payload["status"] == "ok"
    assert payload["counts"]["inputRows"] == 1
    assert payload["counts"]["plannedRuntimeVisibilityRows"] == 1
    assert payload["counts"]["dryRunReadyRuntimeVisibilityRecordOnlyRows"] == 1
    assert payload["counts"]["sectionVisibilityDecisionRows"] == 1
    assert payload["counts"]["figureCaptionVisibilityDecisionRows"] == 0
    assert payload["counts"]["runtimeVisibilityRecordWriteRows"] == 0
    assert payload["counts"]["runtimeVisibleRows"] == 0
    assert payload["counts"]["answerIntegrationVisibleRows"] == 0
    assert payload["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert payload["gate"]["readyForRuntimeBindingVisibilityExecutorDryRun"] is True
    assert payload["gate"]["readyForRuntimeBindingVisibilityExecutorApply"] is False
    assert payload["gate"]["recommendedNextTranche"] == (
        "strict_evidence_runtime_binding_visibility_executor_apply"
    )

    ready_row = payload["rows"][0]
    assert ready_row["dry_run_status"] == DRY_RUN_STATUS_READY
    assert ready_row["plannedWriteTarget"] == RUNTIME_VISIBILITY_STORE
    assert ready_row["plannedRuntimeVisibilityPolicyVersion"] == RUNTIME_VISIBILITY_POLICY_VERSION
    assert ready_row["plannedRuntimeVisibilityDecision"] == RUNTIME_VISIBILITY_DECISION
    assert ready_row["plannedRuntimeVisibilityState"] == RUNTIME_VISIBILITY_STATE
    assert ready_row["sourceContentHash"]

    planned = ready_row["plannedRuntimeVisibilityRecord"]
    assert planned["schema"] == STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_SCHEMA_ID
    assert planned["runtimeVisibilityMutationApplied"] is False
    assert planned["runtimeEvidence"] is False
    assert planned["runtimeVisible"] is False
    assert planned["answerIntegrationVisible"] is False
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_visibility_executor_dry_run_blocks_when_contract_not_ready(
    tmp_path: Path,
) -> None:
    decision_path = _visibility_decision_report_path(tmp_path)
    contract_path = _contract_report(decision_path, tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["status"] = "blocked"
    contract["gate"]["decision"] = "strict_evidence_runtime_binding_visibility_record_contract_blocked"
    _write_json(contract_path, contract)

    payload = build_strict_evidence_runtime_binding_visibility_executor_dry_run(
        visibility_record_contract_report_path=contract_path,
        visibility_decision_report_path=decision_path,
        expected_visibility_decision_rows=1,
        expected_section_visibility_decision_rows=1,
        expected_figure_caption_visibility_decision_rows=0,
        expected_planned_runtime_visibility_rows=1,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["blockedVisibilityRecordContractNotReadyRows"] == 1
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_CONTRACT


def test_runtime_visibility_executor_dry_run_blocks_when_decision_not_ready(
    tmp_path: Path,
) -> None:
    decision_path = _visibility_decision_report_path(tmp_path)
    contract_path = _contract_report(decision_path, tmp_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["status"] = "blocked"
    decision["gate"]["runtimeBindingVisibilityDecisionRecordReady"] = False
    _write_json(decision_path, decision)

    payload = build_strict_evidence_runtime_binding_visibility_executor_dry_run(
        visibility_record_contract_report_path=contract_path,
        visibility_decision_report_path=decision_path,
        expected_visibility_decision_rows=1,
        expected_section_visibility_decision_rows=1,
        expected_figure_caption_visibility_decision_rows=0,
        expected_planned_runtime_visibility_rows=1,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["blockedVisibilityDecisionNotReadyRows"] == 1
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_DECISION


def test_runtime_visibility_executor_dry_run_blocks_non_candidate_decision_rows(
    tmp_path: Path,
) -> None:
    decision_path = _visibility_decision_report_path(tmp_path)
    contract_path = _contract_report(decision_path, tmp_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["rows"][0]["decision_blockers"] = ["hold_status=inactive"]
    _write_json(decision_path, decision)

    payload = build_strict_evidence_runtime_binding_visibility_executor_dry_run(
        visibility_record_contract_report_path=contract_path,
        visibility_decision_report_path=decision_path,
        expected_visibility_decision_rows=1,
        expected_section_visibility_decision_rows=1,
        expected_figure_caption_visibility_decision_rows=0,
        expected_planned_runtime_visibility_rows=1,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["blockedUnsupportedVisibilityPolicyRows"] == 1
    assert payload["rows"][0]["dry_run_status"] == (
        DRY_RUN_STATUS_BLOCKED_UNSUPPORTED_VISIBILITY_POLICY
    )


def test_runtime_visibility_executor_dry_run_blocks_unsupported_artifact_type_with_schema_valid_report(
    tmp_path: Path,
) -> None:
    decision_path = _visibility_decision_report_path(tmp_path)
    contract_path = _contract_report(decision_path, tmp_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["rows"][0]["artifact_type"] = "table"
    _write_json(decision_path, decision)

    payload = build_strict_evidence_runtime_binding_visibility_executor_dry_run(
        visibility_record_contract_report_path=contract_path,
        visibility_decision_report_path=decision_path,
        expected_visibility_decision_rows=1,
        expected_section_visibility_decision_rows=1,
        expected_figure_caption_visibility_decision_rows=0,
        expected_planned_runtime_visibility_rows=1,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["plannedRuntimeVisibilityRows"] == 0
    assert payload["counts"]["dryRunReadyRuntimeVisibilityRecordOnlyRows"] == 0
    assert payload["counts"]["blockedUnsupportedVisibilityPolicyRows"] == 1
    assert payload["counts"]["runtimeVisibilityRecordWriteRows"] == 0
    assert payload["counts"]["runtimeVisibleRows"] == 0
    assert payload["counts"]["answerIntegrationVisibleRows"] == 0
    assert payload["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert payload["counts"]["parserRoutingChangedRows"] == 0
    assert payload["counts"]["answerIntegrationChangedRows"] == 0
    assert payload["counts"]["databaseMutationRows"] == 0
    assert payload["counts"]["reindexOrReembedRows"] == 0
    assert payload["counts"]["vaultScanRows"] == 0

    row = payload["rows"][0]
    assert row["artifact_type"] == "table"
    assert row["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_UNSUPPORTED_VISIBILITY_POLICY
    assert row["dry_run_blockers"] == [
        "artifact_type=table_unsupported_for_runtime_visibility"
    ]
    assert row["plannedRuntimeVisibilityRecord"] == {}
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_visibility_executor_dry_run_blocks_missing_parent_ids(
    tmp_path: Path,
) -> None:
    decision_path = _visibility_decision_report_path(tmp_path)
    contract_path = _contract_report(decision_path, tmp_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["rows"][0]["runtimeBindingRecordId"] = ""
    _write_json(decision_path, decision)

    payload = build_strict_evidence_runtime_binding_visibility_executor_dry_run(
        visibility_record_contract_report_path=contract_path,
        visibility_decision_report_path=decision_path,
        expected_visibility_decision_rows=1,
        expected_section_visibility_decision_rows=1,
        expected_figure_caption_visibility_decision_rows=0,
        expected_planned_runtime_visibility_rows=1,
    )

    assert payload["status"] == "blocked"
    assert payload["rows"][0]["dry_run_status"] == (
        DRY_RUN_STATUS_BLOCKED_MISSING_RUNTIME_BINDING_RECORD_ID
    )


def test_runtime_visibility_executor_dry_run_writer_outputs_schema_valid_reports(
    tmp_path: Path,
) -> None:
    decision_path = _visibility_decision_report_path(tmp_path)
    contract_path = _contract_report(decision_path, tmp_path)

    payload = build_strict_evidence_runtime_binding_visibility_executor_dry_run(
        visibility_record_contract_report_path=contract_path,
        visibility_decision_report_path=decision_path,
        expected_visibility_decision_rows=1,
        expected_section_visibility_decision_rows=1,
        expected_figure_caption_visibility_decision_rows=0,
        expected_planned_runtime_visibility_rows=1,
    )
    paths = write_strict_evidence_runtime_binding_visibility_executor_dry_run_reports(
        payload,
        tmp_path / "reports",
    )

    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert report["schema"] == STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert summary["counts"]["dryRunReadyRuntimeVisibilityRecordOnlyRows"] == 1
    assert "Strict Evidence Runtime Binding Visibility Executor Dry Run" in markdown
    assert validate_payload(
        report,
        STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_visibility_executor_dry_run_integrated_measured_local_report() -> None:
    payload = build_strict_evidence_runtime_binding_visibility_executor_dry_run()

    assert payload["status"] == "ok"
    assert payload["counts"]["inputRows"] == 99
    assert payload["counts"]["plannedRuntimeVisibilityRows"] == 99
    assert payload["counts"]["dryRunReadyRuntimeVisibilityRecordOnlyRows"] == 99
    assert payload["counts"]["sectionVisibilityDecisionRows"] == 45
    assert payload["counts"]["figureCaptionVisibilityDecisionRows"] == 54
    assert payload["counts"]["blockedVisibilityRecordContractNotReadyRows"] == 0
    assert payload["counts"]["blockedVisibilityDecisionNotReadyRows"] == 0
    assert payload["counts"]["blockedUnsupportedVisibilityPolicyRows"] == 0
    assert payload["counts"]["blockedMissingRuntimeBindingRecordIdRows"] == 0
    assert payload["counts"]["blockedMissingCitationGradeRecordIdRows"] == 0
    assert payload["counts"]["blockedMissingStrictEvidenceIdRows"] == 0
    assert payload["counts"]["blockedMissingEligibilityRecordIdRows"] == 0
    assert payload["counts"]["blockedMissingSourceSpanIdRows"] == 0
    assert payload["counts"]["blockedMissingCandidateRecordIdRows"] == 0
    assert payload["counts"]["blockedPlannedVisibilityRecordSchemaViolationRows"] == 0
    assert payload["counts"]["blockedPlannedVisibilityRecordSemanticViolationRows"] == 0
    assert payload["counts"]["blockedInputSchemaViolationRows"] == 0
    assert payload["counts"]["runtimeVisibilityRecordWriteRows"] == 0
    assert payload["counts"]["runtimeBindingRecordWriteRows"] == 0
    assert payload["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert payload["counts"]["runtimeVisibleRows"] == 0
    assert payload["counts"]["answerIntegrationVisibleRows"] == 0
    assert payload["counts"]["answerIntegrationChangedRows"] == 0
    assert payload["counts"]["parserRoutingChangedRows"] == 0
    assert payload["counts"]["databaseMutationRows"] == 0
    assert payload["counts"]["reindexOrReembedRows"] == 0
    assert payload["counts"]["vaultScanRows"] == 0
    assert payload["counts"]["schemaViolationCount"] == 0
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok
