from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_eligibility_executor_dry_run import (
    DRY_RUN_STATUS_BLOCKED_CONTRACT,
    DRY_RUN_STATUS_BLOCKED_DECISION_RECORD,
    DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID,
    DRY_RUN_STATUS_READY,
    STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID,
    build_strict_evidence_eligibility_executor_dry_run,
    write_strict_evidence_eligibility_executor_dry_run_reports,
)
from knowledge_hub.papers.strict_evidence_eligibility_record_contract import (
    ELIGIBILITY_DECISION,
    ELIGIBILITY_POLICY_VERSION,
    ELIGIBILITY_STATE_CANDIDATE_ONLY,
    STRICT_EVIDENCE_ELIGIBILITY_RECORD_SCHEMA_ID,
    STRICT_EVIDENCE_ELIGIBILITY_STORE,
    build_strict_evidence_eligibility_record_contract,
    write_strict_evidence_eligibility_record_contract_reports,
)
from knowledge_hub.papers.strict_evidence_strict_eligible_mutation_decision_record import (
    DECISION_SEPARATE_ELIGIBILITY_RECORD,
    DECISION_STATUS_CANDIDATE_ONLY,
)
from tests.test_strict_evidence_eligibility_record_contract import (
    _decision_report,
    _decision_row,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _contract_report(decision_path: Path, tmp_path: Path) -> Path:
    contract = build_strict_evidence_eligibility_record_contract(
        decision_record_report_path=decision_path,
    )
    contract_dir = tmp_path / "contract"
    write_strict_evidence_eligibility_record_contract_reports(contract, contract_dir)
    return contract_dir / "strict-evidence-eligibility-record-contract.json"


def test_eligibility_executor_dry_run_plans_records_without_writes(tmp_path: Path) -> None:
    rows = [_decision_row(index=1, artifact_type="section"), _decision_row(index=2, artifact_type="figure")]
    decision_path = tmp_path / "decision-record.json"
    _write_json(decision_path, _decision_report(rows))
    contract_path = _contract_report(decision_path, tmp_path)

    payload = build_strict_evidence_eligibility_executor_dry_run(
        eligibility_record_contract_report_path=contract_path,
        decision_record_report_path=decision_path,
        run_id="dry-run-test",
    )

    assert payload["schema"] == STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert payload["status"] == "ok"
    assert payload["counts"]["inputRows"] == 2
    assert payload["counts"]["decisionCandidateRows"] == 2
    assert payload["counts"]["dryRunReadyEligibilityRecordOnlyRows"] == 2
    assert payload["counts"]["eligibilityRecordWriteRows"] == 0
    assert payload["counts"]["strictEligibleMutationRows"] == 0
    assert payload["gate"]["readyForEligibilityExecutorDryRun"] is True
    assert payload["gate"]["readyForEligibilityExecutorApply"] is False
    assert payload["gate"]["recommendedNextTranche"] == "strict_evidence_eligibility_executor_apply"

    ready_row = payload["rows"][0]
    assert ready_row["dry_run_status"] == DRY_RUN_STATUS_READY
    assert ready_row["plannedWriteTarget"] == STRICT_EVIDENCE_ELIGIBILITY_STORE
    assert ready_row["plannedEligibilityPolicyVersion"] == ELIGIBILITY_POLICY_VERSION
    assert ready_row["plannedEligibilityDecision"] == ELIGIBILITY_DECISION
    assert ready_row["plannedEligibilityState"] == ELIGIBILITY_STATE_CANDIDATE_ONLY

    planned = ready_row["plannedEligibilityRecord"]
    assert planned["schema"] == STRICT_EVIDENCE_ELIGIBILITY_RECORD_SCHEMA_ID
    assert planned["strictEligibleMutationApplied"] is False
    assert planned["citationGrade"] is False
    assert planned["runtimeEvidence"] is False
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_eligibility_executor_dry_run_blocks_when_contract_not_ready(tmp_path: Path) -> None:
    decision_path = tmp_path / "decision-record.json"
    _write_json(decision_path, _decision_report([_decision_row()]))
    contract_path = _contract_report(decision_path, tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["status"] = "blocked"
    contract["gate"]["decision"] = "strict_evidence_eligibility_record_contract_blocked"
    _write_json(contract_path, contract)

    payload = build_strict_evidence_eligibility_executor_dry_run(
        eligibility_record_contract_report_path=contract_path,
        decision_record_report_path=decision_path,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["blockedContractNotReadyRows"] == 1
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_CONTRACT


def test_eligibility_executor_dry_run_blocks_non_candidate_decision_rows(tmp_path: Path) -> None:
    row = _decision_row()
    row["decision_status"] = "blocked"
    decision_path = tmp_path / "decision-record.json"
    _write_json(decision_path, _decision_report([row]))
    contract_path = _contract_report(decision_path, tmp_path)

    payload = build_strict_evidence_eligibility_executor_dry_run(
        eligibility_record_contract_report_path=contract_path,
        decision_record_report_path=decision_path,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["decisionCandidateRows"] == 0
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_DECISION_RECORD


def test_eligibility_executor_dry_run_blocks_missing_strict_evidence_id(tmp_path: Path) -> None:
    row = _decision_row()
    row["strictEvidenceId"] = ""
    decision_path = tmp_path / "decision-record.json"
    _write_json(decision_path, _decision_report([row]))
    contract_path = _contract_report(decision_path, tmp_path)

    payload = build_strict_evidence_eligibility_executor_dry_run(
        eligibility_record_contract_report_path=contract_path,
        decision_record_report_path=decision_path,
    )

    assert payload["status"] == "blocked"
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID


def test_eligibility_executor_dry_run_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    decision_path = tmp_path / "decision-record.json"
    _write_json(decision_path, _decision_report([_decision_row()]))
    contract_path = _contract_report(decision_path, tmp_path)

    payload = build_strict_evidence_eligibility_executor_dry_run(
        eligibility_record_contract_report_path=contract_path,
        decision_record_report_path=decision_path,
    )
    paths = write_strict_evidence_eligibility_executor_dry_run_reports(
        payload,
        tmp_path / "reports",
    )

    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert report["schema"] == STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert summary["counts"]["dryRunReadyEligibilityRecordOnlyRows"] == 1
    assert "Strict Evidence Eligibility Executor Dry Run" in markdown
    assert validate_payload(
        report,
        STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_eligibility_executor_dry_run_integrated_measured_local_report() -> None:
    payload = build_strict_evidence_eligibility_executor_dry_run()
    assert payload["status"] == "ok"
    assert payload["counts"]["inputRows"] == 99
    assert payload["counts"]["decisionCandidateRows"] == 99
    assert payload["counts"]["dryRunReadyEligibilityRecordOnlyRows"] == 99
    assert payload["counts"]["blockedContractNotReadyRows"] == 0
    assert payload["counts"]["blockedDecisionRecordNotReadyRows"] == 0
    assert payload["counts"]["blockedMissingStrictEvidenceIdRows"] == 0
    assert payload["counts"]["blockedMissingSourceSpanIdRows"] == 0
    assert payload["counts"]["blockedMissingCandidateRecordIdRows"] == 0
    assert payload["counts"]["blockedPlannedRecordSchemaViolationRows"] == 0
    assert payload["counts"]["blockedPlannedRecordSemanticViolationRows"] == 0
    assert payload["counts"]["blockedInputSchemaViolationRows"] == 0
    assert payload["counts"]["eligibilityRecordWriteRows"] == 0
    assert payload["counts"]["strictEligibleMutationRows"] == 0
    assert payload["counts"]["citationGradeEvidenceCreatedRows"] == 0
    assert payload["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert payload["counts"]["parserRoutingChangedRows"] == 0
    assert payload["counts"]["answerIntegrationChangedRows"] == 0
    assert payload["counts"]["databaseMutationRows"] == 0
    assert payload["counts"]["reindexOrReembedRows"] == 0
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok
