from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_authority_record_contract import (
    EQUATION_AUTHORITY_RECORD_SCHEMA_ID,
    EQUATION_AUTHORITY_RECORD_STORE,
    build_equation_authority_record_contract,
    write_equation_authority_record_contract_reports,
)
from knowledge_hub.papers.equation_authority_record_executor_dry_run import (
    DRY_RUN_STATUS_BLOCKED_CONTRACT,
    DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA,
    DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA,
    DRY_RUN_STATUS_BLOCKED_RECORD_CONTRACT_STATUS,
    DRY_RUN_STATUS_READY,
    EQUATION_AUTHORITY_RECORD_EXECUTOR_DRY_RUN_SCHEMA_ID,
    build_equation_authority_record_executor_dry_run,
    write_equation_authority_record_executor_dry_run_reports,
)
from knowledge_hub.papers.equation_authority_hash_identity_design import (
    STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH as HASH_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
    STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT as HASH_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT,
    STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY,
)
from tests.test_equation_authority_record_contract import _contract_design_report_path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _record_contract_report_path(tmp_path: Path, rows: list[dict]) -> Path:
    contract_design_path = _contract_design_report_path(tmp_path, rows)
    payload = build_equation_authority_record_contract(contract_design_path)
    output_dir = tmp_path / "record-contract"
    write_equation_authority_record_contract_reports(payload, output_dir)
    return output_dir / "equation-authority-record-contract-report.json"


def test_equation_authority_record_executor_dry_run_plans_records_without_writes(
    tmp_path: Path,
) -> None:
    from tests.test_equation_authority_contract_design import _hash_row

    contract_path = _record_contract_report_path(
        tmp_path,
        [
            _hash_row("candidate", design_status=STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY),
            _hash_row(
                "missing-hash",
                design_status=HASH_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT,
                selected_hash="",
            ),
            _hash_row(
                "ambiguous",
                design_status=HASH_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
                ambiguous=True,
            ),
        ],
    )

    payload = build_equation_authority_record_executor_dry_run(
        equation_authority_record_contract_report=contract_path,
        run_id="dry-run-test",
        expected_input_rows=3,
        expected_planned_equation_authority_record_rows=1,
    )

    assert payload["schema"] == EQUATION_AUTHORITY_RECORD_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert payload["status"] == "ok"
    counts = payload["counts"]
    assert counts["inputRows"] == 3
    assert counts["plannedEquationAuthorityRecordRows"] == 1
    assert counts["dryRunReadyEquationAuthorityRecordOnlyRows"] == 1
    assert counts["blockedRecordContractStatusNotCandidateRows"] == 2
    assert counts["equationAuthorityRecordWriteRows"] == 0
    assert counts["equationArtifactCreatedRows"] == 0
    assert counts["strictEvidenceCreatedRows"] == 0
    assert counts["sourceSpanMutatedRows"] == 0
    assert payload["gate"]["readyForEquationAuthorityRecordExecutorDryRun"] is True
    assert payload["gate"]["readyForEquationAuthorityRecordExecutorApply"] is False
    assert payload["gate"]["recommendedNextTranche"] == "equation_authority_record_executor_apply"

    ready_row = next(row for row in payload["rows"] if row["dry_run_status"] == DRY_RUN_STATUS_READY)
    assert ready_row["plannedWriteTarget"] == EQUATION_AUTHORITY_RECORD_STORE
    assert ready_row["plannedContractVersion"] == "equation_authority_record_contract_v1"
    assert ready_row["plannedAuthorityDecision"] == "equation_authority_record_contract_candidate_only"
    assert ready_row["plannedAuthorityState"] == "record_contract_candidate_only"
    planned = ready_row["plannedEquationAuthorityRecord"]
    assert planned["schema"] == EQUATION_AUTHORITY_RECORD_SCHEMA_ID
    assert planned["runId"] == "dry-run-test"
    assert planned["runtimeVisible"] is False
    assert planned["answerIntegrationVisible"] is False
    assert validate_payload(planned, EQUATION_AUTHORITY_RECORD_SCHEMA_ID, strict=True).ok
    assert validate_payload(payload, EQUATION_AUTHORITY_RECORD_EXECUTOR_DRY_RUN_SCHEMA_ID, strict=True).ok


def test_equation_authority_record_executor_dry_run_blocks_when_contract_not_ready(
    tmp_path: Path,
) -> None:
    from tests.test_equation_authority_contract_design import _hash_row

    contract_path = _record_contract_report_path(
        tmp_path,
        [_hash_row("candidate", design_status=STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY)],
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["status"] = "blocked"
    contract["gate"]["decision"] = "equation_authority_record_contract_blocked"
    _write_json(contract_path, contract)

    payload = build_equation_authority_record_executor_dry_run(
        equation_authority_record_contract_report=contract_path,
        expected_input_rows=1,
        expected_planned_equation_authority_record_rows=1,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["blockedContractNotReadyRows"] == 1
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_CONTRACT


def test_equation_authority_record_executor_dry_run_blocks_non_candidate_record_contract_row(
    tmp_path: Path,
) -> None:
    from tests.test_equation_authority_contract_design import _hash_row

    contract_path = _record_contract_report_path(
        tmp_path,
        [
            _hash_row(
                "missing-hash",
                design_status=HASH_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT,
                selected_hash="",
            )
        ],
    )

    payload = build_equation_authority_record_executor_dry_run(
        equation_authority_record_contract_report=contract_path,
        expected_input_rows=1,
        expected_planned_equation_authority_record_rows=0,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["blockedRecordContractStatusNotCandidateRows"] == 1
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_RECORD_CONTRACT_STATUS


def test_equation_authority_record_executor_dry_run_blocks_planned_schema_violation(
    tmp_path: Path,
) -> None:
    from tests.test_equation_authority_contract_design import _hash_row

    contract_path = _record_contract_report_path(
        tmp_path,
        [_hash_row("candidate", design_status=STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY)],
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["rows"][0]["equation_authority_record_preview"]["runtimeVisible"] = True
    _write_json(contract_path, contract)

    payload = build_equation_authority_record_executor_dry_run(
        equation_authority_record_contract_report=contract_path,
        expected_input_rows=1,
        expected_planned_equation_authority_record_rows=1,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["blockedPlannedRecordSchemaViolationRows"] == 1
    assert payload["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA


def test_equation_authority_record_executor_dry_run_blocks_missing_input_report(
    tmp_path: Path,
) -> None:
    payload = build_equation_authority_record_executor_dry_run(
        equation_authority_record_contract_report=tmp_path / "missing-record-contract-report.json"
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["blockedInputSchemaViolationRows"] == 0
    assert payload["counts"]["schemaViolationCount"] == 1
    assert payload["rows"] == []
    assert payload["gate"]["decision"] == "equation_authority_record_executor_dry_run_blocked"
    assert validate_payload(payload, EQUATION_AUTHORITY_RECORD_EXECUTOR_DRY_RUN_SCHEMA_ID, strict=True).ok


def test_equation_authority_record_executor_dry_run_writer_outputs_valid_report(
    tmp_path: Path,
) -> None:
    from tests.test_equation_authority_contract_design import _hash_row

    contract_path = _record_contract_report_path(
        tmp_path,
        [_hash_row("candidate", design_status=STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY)],
    )
    payload = build_equation_authority_record_executor_dry_run(
        equation_authority_record_contract_report=contract_path,
        expected_input_rows=1,
        expected_planned_equation_authority_record_rows=1,
    )
    paths = write_equation_authority_record_executor_dry_run_reports(payload, tmp_path / "reports")

    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert set(paths) == {"report", "summary", "markdown"}
    assert report["schema"] == EQUATION_AUTHORITY_RECORD_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert summary["counts"]["dryRunReadyEquationAuthorityRecordOnlyRows"] == 1
    assert "Equation Authority Record Executor Dry Run" in markdown
    assert validate_payload(report, EQUATION_AUTHORITY_RECORD_EXECUTOR_DRY_RUN_SCHEMA_ID, strict=True).ok
