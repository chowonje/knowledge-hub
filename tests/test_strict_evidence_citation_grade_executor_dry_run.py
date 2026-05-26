from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_citation_grade_executor_dry_run import (
    DRY_RUN_STATUS_BLOCKED_CONTRACT,
    DRY_RUN_STATUS_BLOCKED_MISSING_ELIGIBILITY_RECORD_ID,
    DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID,
    DRY_RUN_STATUS_BLOCKED_POLICY_GATE,
    DRY_RUN_STATUS_READY,
    STRICT_EVIDENCE_CITATION_GRADE_EXECUTOR_DRY_RUN_SCHEMA_ID,
    build_strict_evidence_citation_grade_executor_dry_run,
    write_strict_evidence_citation_grade_executor_dry_run_reports,
)
from knowledge_hub.papers.strict_evidence_citation_grade_record_contract import (
    CITATION_GRADE_DECISION,
    CITATION_GRADE_POLICY_VERSION,
    CITATION_GRADE_STATE_CANDIDATE_ONLY,
    STRICT_EVIDENCE_CITATION_GRADE_RECORD_CONTRACT_SCHEMA_ID,
    STRICT_EVIDENCE_CITATION_GRADE_RECORD_SCHEMA_ID,
    STRICT_EVIDENCE_CITATION_GRADE_STORE,
    build_strict_evidence_citation_grade_record_contract,
    write_strict_evidence_citation_grade_record_contract_reports,
)
from tests.test_strict_evidence_citation_grade_record_contract import (
    _policy_gate_design_report_path,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _contract_report(policy_path: Path, tmp_path: Path) -> Path:
    contract = build_strict_evidence_citation_grade_record_contract(
        policy_gate_design_report_path=policy_path,
    )
    contract_dir = tmp_path / "contract"
    write_strict_evidence_citation_grade_record_contract_reports(contract, contract_dir)
    return contract_dir / "strict-evidence-citation-grade-record-contract.json"


def test_citation_grade_executor_dry_run_plans_records_without_writes(tmp_path: Path) -> None:
    policy_path = _policy_gate_design_report_path(tmp_path)
    contract_path = _contract_report(policy_path, tmp_path)

    payload = build_strict_evidence_citation_grade_executor_dry_run(
        citation_grade_record_contract_report_path=contract_path,
        policy_gate_design_report_path=policy_path,
        run_id="dry-run-test",
        expected_policy_candidate_rows=1,
        expected_section_policy_rows=1,
        expected_figure_caption_policy_rows=0,
    )

    assert payload["schema"] == STRICT_EVIDENCE_CITATION_GRADE_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert payload["status"] == "ok"
    assert payload["counts"]["inputRows"] == 1
    assert payload["counts"]["policyCandidateRows"] == 1
    assert payload["counts"]["dryRunReadyCitationGradeRecordOnlyRows"] == 1
    assert payload["counts"]["citationGradeRecordWriteRows"] == 0
    assert payload["counts"]["citationGradeBooleanMutationRows"] == 0
    assert payload["gate"]["readyForCitationGradeExecutorDryRun"] is True
    assert payload["gate"]["readyForCitationGradeExecutorApply"] is False
    assert payload["gate"]["recommendedNextTranche"] == "strict_evidence_citation_grade_executor_apply"

    ready_row = payload["rows"][0]
    assert ready_row["dry_run_status"] == DRY_RUN_STATUS_READY
    assert ready_row["plannedWriteTarget"] == STRICT_EVIDENCE_CITATION_GRADE_STORE
    assert ready_row["plannedCitationGradePolicyVersion"] == CITATION_GRADE_POLICY_VERSION
    assert ready_row["plannedCitationGradeDecision"] == CITATION_GRADE_DECISION
    assert ready_row["plannedCitationGradeState"] == CITATION_GRADE_STATE_CANDIDATE_ONLY

    planned = ready_row["plannedCitationGradeRecord"]
    assert planned["schema"] == STRICT_EVIDENCE_CITATION_GRADE_RECORD_SCHEMA_ID
    assert planned["citationGradeMutationApplied"] is False
    assert planned["runtimeEvidence"] is False
    assert planned["runtimeVisible"] is False
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_CITATION_GRADE_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_citation_grade_executor_dry_run_blocks_when_contract_not_ready(tmp_path: Path) -> None:
    policy_path = _policy_gate_design_report_path(tmp_path)
    contract_path = _contract_report(policy_path, tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["status"] = "blocked"
    contract["gate"]["decision"] = "strict_evidence_citation_grade_record_contract_blocked"
    _write_json(contract_path, contract)

    payload = build_strict_evidence_citation_grade_executor_dry_run(
        citation_grade_record_contract_report_path=contract_path,
        policy_gate_design_report_path=policy_path,
        expected_policy_candidate_rows=1,
        expected_section_policy_rows=1,
        expected_figure_caption_policy_rows=0,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["blockedContractNotReadyRows"] == 1
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_CONTRACT


def test_citation_grade_executor_dry_run_blocks_non_candidate_policy_rows(tmp_path: Path) -> None:
    policy_path = _policy_gate_design_report_path(tmp_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["rows"][0]["citation_grade_policy_design_status"] = "blocked_post_apply_hold_not_active"
    policy["rows"][0]["citationGradePolicyDesignCandidateOnly"] = False
    _write_json(policy_path, policy)
    contract_path = _contract_report(policy_path, tmp_path)

    payload = build_strict_evidence_citation_grade_executor_dry_run(
        citation_grade_record_contract_report_path=contract_path,
        policy_gate_design_report_path=policy_path,
        expected_policy_candidate_rows=1,
        expected_section_policy_rows=1,
        expected_figure_caption_policy_rows=0,
    )

    assert payload["status"] == "blocked"
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_POLICY_GATE


def test_citation_grade_executor_dry_run_blocks_missing_parent_ids(tmp_path: Path) -> None:
    policy_path = _policy_gate_design_report_path(tmp_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["rows"][0]["strictEvidenceId"] = ""
    _write_json(policy_path, policy)
    contract_path = _contract_report(policy_path, tmp_path)

    payload = build_strict_evidence_citation_grade_executor_dry_run(
        citation_grade_record_contract_report_path=contract_path,
        policy_gate_design_report_path=policy_path,
        expected_policy_candidate_rows=1,
        expected_section_policy_rows=1,
        expected_figure_caption_policy_rows=0,
    )

    assert payload["status"] == "blocked"
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID

    policy["rows"][0]["strictEvidenceId"] = "strict-evidence:paper-1:section:1"
    policy["rows"][0]["eligibilityRecordId"] = ""
    _write_json(policy_path, policy)
    payload = build_strict_evidence_citation_grade_executor_dry_run(
        citation_grade_record_contract_report_path=contract_path,
        policy_gate_design_report_path=policy_path,
        expected_policy_candidate_rows=1,
        expected_section_policy_rows=1,
        expected_figure_caption_policy_rows=0,
    )
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_MISSING_ELIGIBILITY_RECORD_ID


def test_citation_grade_executor_dry_run_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    policy_path = _policy_gate_design_report_path(tmp_path)
    contract_path = _contract_report(policy_path, tmp_path)

    payload = build_strict_evidence_citation_grade_executor_dry_run(
        citation_grade_record_contract_report_path=contract_path,
        policy_gate_design_report_path=policy_path,
        expected_policy_candidate_rows=1,
        expected_section_policy_rows=1,
        expected_figure_caption_policy_rows=0,
    )
    paths = write_strict_evidence_citation_grade_executor_dry_run_reports(
        payload,
        tmp_path / "reports",
    )

    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert report["schema"] == STRICT_EVIDENCE_CITATION_GRADE_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert summary["counts"]["dryRunReadyCitationGradeRecordOnlyRows"] == 1
    assert "Strict Evidence Citation-Grade Executor Dry Run" in markdown
    assert validate_payload(
        report,
        STRICT_EVIDENCE_CITATION_GRADE_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_citation_grade_executor_dry_run_integrated_measured_local_report() -> None:
    payload = build_strict_evidence_citation_grade_executor_dry_run()
    assert payload["status"] == "ok"
    assert payload["counts"]["inputRows"] == 99
    assert payload["counts"]["policyCandidateRows"] == 99
    assert payload["counts"]["dryRunReadyCitationGradeRecordOnlyRows"] == 99
    assert payload["counts"]["blockedContractNotReadyRows"] == 0
    assert payload["counts"]["blockedPolicyGateDesignNotReadyRows"] == 0
    assert payload["counts"]["blockedMissingStrictEvidenceIdRows"] == 0
    assert payload["counts"]["blockedMissingSourceSpanIdRows"] == 0
    assert payload["counts"]["blockedMissingCandidateRecordIdRows"] == 0
    assert payload["counts"]["blockedMissingEligibilityRecordIdRows"] == 0
    assert payload["counts"]["blockedPlannedRecordSchemaViolationRows"] == 0
    assert payload["counts"]["blockedPlannedRecordSemanticViolationRows"] == 0
    assert payload["counts"]["blockedInputSchemaViolationRows"] == 0
    assert payload["counts"]["citationGradeRecordWriteRows"] == 0
    assert payload["counts"]["citationGradeBooleanMutationRows"] == 0
    assert payload["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert payload["counts"]["parserRoutingChangedRows"] == 0
    assert payload["counts"]["answerIntegrationChangedRows"] == 0
    assert payload["counts"]["databaseMutationRows"] == 0
    assert payload["counts"]["reindexOrReembedRows"] == 0
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_CITATION_GRADE_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok
