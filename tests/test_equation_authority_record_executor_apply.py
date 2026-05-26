from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_authority_hash_identity_design import (
    STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH as HASH_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
    STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT as HASH_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT,
    STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY,
)
from knowledge_hub.papers.equation_authority_record_contract import (
    EQUATION_AUTHORITY_RECORD_SCHEMA_ID,
    EQUATION_AUTHORITY_RECORD_STORE,
    build_equation_authority_record_contract,
    write_equation_authority_record_contract_reports,
)
from knowledge_hub.papers.equation_authority_record_executor_apply import (
    APPLY_STATUS_APPLIED,
    APPLY_STATUS_READY,
    EQUATION_AUTHORITY_RECORD_EXECUTOR_APPLY_SCHEMA_ID,
    build_equation_authority_record_executor_apply,
    default_output_dir,
    write_equation_authority_record_executor_apply_reports,
)
from knowledge_hub.papers.equation_authority_record_executor_dry_run import (
    build_equation_authority_record_executor_dry_run,
    write_equation_authority_record_executor_dry_run_reports,
)
from tests.test_equation_authority_record_contract import _contract_design_report_path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fixture_reports(tmp_path: Path) -> tuple[Path, Path]:
    from tests.test_equation_authority_contract_design import _hash_row

    contract_design_path = _contract_design_report_path(
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
    contract = build_equation_authority_record_contract(contract_design_path)
    contract_dir = tmp_path / "record-contract"
    write_equation_authority_record_contract_reports(contract, contract_dir)
    contract_path = contract_dir / "equation-authority-record-contract-report.json"

    dry_run = build_equation_authority_record_executor_dry_run(
        equation_authority_record_contract_report=contract_path,
        run_id="dry-run-test",
        expected_input_rows=3,
        expected_planned_equation_authority_record_rows=1,
    )
    dry_run_dir = tmp_path / "dry-run"
    write_equation_authority_record_executor_dry_run_reports(dry_run, dry_run_dir)
    dry_run_path = dry_run_dir / "equation-authority-record-executor-dry-run-report.json"
    return dry_run_path, contract_path


def test_equation_authority_record_executor_apply_plans_without_writes(tmp_path: Path) -> None:
    dry_run_path, contract_path = _fixture_reports(tmp_path)

    report = build_equation_authority_record_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        equation_authority_record_contract_report_path=contract_path,
        apply=False,
    )

    assert report["schema"] == EQUATION_AUTHORITY_RECORD_EXECUTOR_APPLY_SCHEMA_ID
    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 3
    assert report["counts"]["dryRunReadyEquationAuthorityRecordRows"] == 1
    assert report["counts"]["plannedApplyRows"] == 1
    assert report["counts"]["heldInputRows"] == 2
    assert report["counts"]["equationAuthorityRecordWriteRows"] == 0
    assert report["counts"]["equationArtifactCreatedRows"] == 0
    assert report["counts"]["strictEvidenceCreatedRows"] == 0
    assert report["counts"]["sourceSpanMutatedRows"] == 0
    assert report["gate"]["readyForDryRunApplyPlanning"] is True
    assert report["gate"]["readyForEquationAuthorityRecordApply"] is True
    assert report["gate"]["equationAuthorityRecordWriteAllowed"] is False

    ready_row = next(row for row in report["rows"] if row["apply_status"] == APPLY_STATUS_READY)
    assert ready_row["would_write_equation_authority_record"] is True
    assert ready_row["applied_equation_authority_record"] is False
    assert ready_row["planned_write_target"] == EQUATION_AUTHORITY_RECORD_STORE
    assert validate_payload(report, EQUATION_AUTHORITY_RECORD_EXECUTOR_APPLY_SCHEMA_ID, strict=True).ok


def test_equation_authority_record_executor_apply_writes_idempotent_jsonl(
    tmp_path: Path,
) -> None:
    dry_run_path, contract_path = _fixture_reports(tmp_path)
    papers_dir = tmp_path / "papers"

    report = build_equation_authority_record_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        equation_authority_record_contract_report_path=contract_path,
        papers_dir=papers_dir,
        run_id="test-equation-authority-apply-run",
        apply=True,
    )
    repeat = build_equation_authority_record_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        equation_authority_record_contract_report_path=contract_path,
        papers_dir=papers_dir,
        run_id="test-equation-authority-apply-run",
        apply=True,
    )

    assert report["status"] == "ok"
    assert report["counts"]["equationAuthorityRecordWriteRows"] == 1
    assert report["counts"]["readbackValidatedRows"] == 1
    assert report["counts"]["appliedEquationAuthorityRecordRows"] == 1
    assert report["counts"]["equationArtifactCreatedRows"] == 0
    assert report["counts"]["strictEvidenceCreatedRows"] == 0
    assert repeat["status"] == "ok"
    assert repeat["counts"]["readbackValidatedRows"] == 1

    apply_row = next(row for row in report["rows"] if row["apply_status"] == APPLY_STATUS_APPLIED)
    assert apply_row["equationAuthorityRecordWriteRows"] == 1
    jsonl_path = Path(apply_row["equation_authority_record_store_path"])
    assert jsonl_path == papers_dir / "structured_evidence" / "equation_authority" / "paper-1.jsonl"
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert validate_payload(stored, EQUATION_AUTHORITY_RECORD_SCHEMA_ID, strict=True).ok
    assert stored["runId"] == "test-equation-authority-apply-run"
    assert stored["plannedWriteTarget"] == EQUATION_AUTHORITY_RECORD_STORE
    assert stored["runtimeVisible"] is False
    assert stored["answerIntegrationVisible"] is False
    assert stored["equationArtifactCreated"] is False
    assert stored["strictEvidenceCreated"] is False
    assert stored["sourceSpanMutationAllowed"] is False
    assert validate_payload(report, EQUATION_AUTHORITY_RECORD_EXECUTOR_APPLY_SCHEMA_ID, strict=True).ok


def test_equation_authority_record_executor_apply_blocks_non_ready_dry_run_row(
    tmp_path: Path,
) -> None:
    dry_run_path, contract_path = _fixture_reports(tmp_path)
    dry_run = json.loads(dry_run_path.read_text(encoding="utf-8"))
    dry_run["rows"][0]["dry_run_status"] = "blocked_input_schema_violation"
    dry_run["rows"][0]["dryRunReadyEquationAuthorityRecordOnly"] = False
    _write_json(dry_run_path, dry_run)

    report = build_equation_authority_record_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        equation_authority_record_contract_report_path=contract_path,
        apply=False,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["blockedDryRunNotReadyRows"] >= 1
    assert report["counts"]["equationAuthorityRecordWriteRows"] == 0
    assert validate_payload(report, EQUATION_AUTHORITY_RECORD_EXECUTOR_APPLY_SCHEMA_ID, strict=True).ok


def test_equation_authority_record_executor_apply_rejects_missing_papers_dir(
    tmp_path: Path,
) -> None:
    dry_run_path, contract_path = _fixture_reports(tmp_path)

    report = build_equation_authority_record_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        equation_authority_record_contract_report_path=contract_path,
        apply=True,
    )

    assert report["status"] == "blocked"
    assert "apply_requires_papers_dir" in report["gate"]["schemaViolations"]
    assert report["counts"]["equationAuthorityRecordWriteRows"] == 0
    assert validate_payload(report, EQUATION_AUTHORITY_RECORD_EXECUTOR_APPLY_SCHEMA_ID, strict=True).ok


def test_equation_authority_record_executor_apply_writer_outputs_valid_report(
    tmp_path: Path,
) -> None:
    dry_run_path, contract_path = _fixture_reports(tmp_path)

    report = build_equation_authority_record_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        equation_authority_record_contract_report_path=contract_path,
        apply=False,
    )
    paths = write_equation_authority_record_executor_apply_reports(report, tmp_path / "reports")

    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert written["schema"] == EQUATION_AUTHORITY_RECORD_EXECUTOR_APPLY_SCHEMA_ID
    assert summary["counts"]["plannedApplyRows"] == 1
    assert "Equation Authority Record Executor Apply" in markdown
    assert validate_payload(written, EQUATION_AUTHORITY_RECORD_EXECUTOR_APPLY_SCHEMA_ID, strict=True).ok


def test_equation_authority_record_executor_apply_default_output_dirs_are_mode_specific() -> None:
    assert default_output_dir(apply=False) != default_output_dir(apply=True)
    assert "apply-plan" in str(default_output_dir(apply=False))
