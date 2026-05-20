from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_runtime_binding_executor_apply import (
    APPLY_STATUS_APPLIED,
    APPLY_STATUS_READY,
    RUNTIME_BINDING_STORE,
    STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_SCHEMA_ID,
    build_strict_evidence_runtime_binding_executor_apply,
    default_output_dir,
    write_strict_evidence_runtime_binding_executor_apply_reports,
)
from knowledge_hub.papers.strict_evidence_runtime_binding_executor_dry_run import (
    build_strict_evidence_runtime_binding_executor_dry_run,
    write_strict_evidence_runtime_binding_executor_dry_run_reports,
)
from knowledge_hub.papers.strict_evidence_runtime_binding_record_contract import (
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


def _fixture_reports(tmp_path: Path) -> tuple[Path, Path, Path]:
    gate_design_path = _runtime_binding_gate_design_report_path(tmp_path)
    contract = build_strict_evidence_runtime_binding_record_contract(
        runtime_binding_gate_design_report_path=gate_design_path,
        expected_input_rows=1,
        expected_planned_runtime_binding_rows=1,
    )
    contract_dir = tmp_path / "contract"
    write_strict_evidence_runtime_binding_record_contract_reports(contract, contract_dir)
    contract_path = contract_dir / "strict-evidence-runtime-binding-record-contract.json"

    dry_run = build_strict_evidence_runtime_binding_executor_dry_run(
        runtime_binding_record_contract_report_path=contract_path,
        runtime_binding_gate_design_report_path=gate_design_path,
        run_id="dry-run-test",
        expected_runtime_binding_gate_design_rows=1,
        expected_section_gate_design_rows=1,
        expected_figure_caption_gate_design_rows=0,
        expected_planned_runtime_binding_rows=1,
    )
    dry_run_dir = tmp_path / "dry-run"
    write_strict_evidence_runtime_binding_executor_dry_run_reports(dry_run, dry_run_dir)
    dry_run_path = dry_run_dir / "strict-evidence-runtime-binding-executor-dry-run.json"
    return dry_run_path, contract_path, gate_design_path


def test_runtime_binding_executor_apply_plans_without_writes(tmp_path: Path) -> None:
    dry_run_path, contract_path, _ = _fixture_reports(tmp_path)

    report = build_strict_evidence_runtime_binding_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        runtime_binding_record_contract_report_path=contract_path,
        apply=False,
    )

    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 1
    assert report["counts"]["plannedApplyRows"] == 1
    assert report["counts"]["runtimeBindingRecordWriteRows"] == 0
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert report["counts"]["answerIntegrationVisibleRows"] == 0
    assert report["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert report["rows"][0]["apply_status"] == APPLY_STATUS_READY
    assert report["rows"][0]["would_write_runtime_binding_record"] is True
    assert validate_payload(
        report,
        STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_binding_executor_apply_writes_runtime_binding_jsonl(tmp_path: Path) -> None:
    dry_run_path, contract_path, _ = _fixture_reports(tmp_path)
    papers_dir = tmp_path / "papers"

    report = build_strict_evidence_runtime_binding_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        runtime_binding_record_contract_report_path=contract_path,
        papers_dir=papers_dir,
        run_id="test-runtime-binding-apply-run",
        apply=True,
    )

    assert report["status"] == "ok"
    assert report["counts"]["runtimeBindingRecordWriteRows"] == 1
    assert report["counts"]["readbackValidatedRows"] == 1
    assert report["counts"]["appliedRuntimeBindingRecordRows"] == 1
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert report["counts"]["answerIntegrationVisibleRows"] == 0
    assert report["rows"][0]["apply_status"] == APPLY_STATUS_APPLIED

    jsonl_path = (
        papers_dir
        / "structured_evidence"
        / "strict_evidence_runtime_binding"
        / "paper-1.jsonl"
    )
    assert jsonl_path.is_file()
    stored = json.loads(jsonl_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert validate_payload(
        stored,
        STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID,
        strict=True,
    ).ok
    assert stored["runId"] == "test-runtime-binding-apply-run"
    assert stored["plannedWriteTarget"] == RUNTIME_BINDING_STORE
    assert stored["runtimeBindingMutationApplied"] is False
    assert stored["runtimeEvidence"] is False
    assert stored["runtimeVisible"] is False
    assert stored["answerIntegrationVisible"] is False


def test_runtime_binding_executor_apply_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    dry_run_path, contract_path, _ = _fixture_reports(tmp_path)

    report = build_strict_evidence_runtime_binding_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        runtime_binding_record_contract_report_path=contract_path,
        apply=False,
    )
    paths = write_strict_evidence_runtime_binding_executor_apply_reports(
        report,
        tmp_path / "reports",
    )

    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert written["schema"] == STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_SCHEMA_ID
    assert summary["counts"]["plannedApplyRows"] == 1
    assert "Strict Evidence Runtime Binding Executor Apply" in markdown
    assert validate_payload(
        written,
        STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_binding_executor_apply_integrated_measured_local_plan() -> None:
    report = build_strict_evidence_runtime_binding_executor_apply(apply=False)

    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 99
    assert report["counts"]["plannedApplyRows"] == 99
    assert report["counts"]["runtimeBindingRecordWriteRows"] == 0
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert report["counts"]["answerIntegrationVisibleRows"] == 0
    assert report["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert validate_payload(
        report,
        STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_binding_executor_apply_blocks_non_ready_dry_run_row(tmp_path: Path) -> None:
    dry_run_path, contract_path, _ = _fixture_reports(tmp_path)
    dry_run = json.loads(dry_run_path.read_text(encoding="utf-8"))
    dry_run["rows"][0]["dry_run_status"] = "blocked_input_schema_violation"
    dry_run["rows"][0]["dryRunReadyRuntimeBindingRecordOnly"] = False
    _write_json(dry_run_path, dry_run)

    report = build_strict_evidence_runtime_binding_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        runtime_binding_record_contract_report_path=contract_path,
        apply=False,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["blockedDryRunNotReadyRows"] >= 1


def test_runtime_binding_executor_apply_rejects_missing_papers_dir(tmp_path: Path) -> None:
    dry_run_path, contract_path, _ = _fixture_reports(tmp_path)

    report = build_strict_evidence_runtime_binding_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        runtime_binding_record_contract_report_path=contract_path,
        apply=True,
    )

    assert report["status"] == "blocked"
    assert "apply_requires_papers_dir" in report["gate"]["schemaViolations"]
    assert report["counts"]["runtimeBindingRecordWriteRows"] == 0


def test_runtime_binding_executor_apply_default_output_dirs_are_mode_specific() -> None:
    assert default_output_dir(apply=False) != default_output_dir(apply=True)
    assert "dry-run" in str(default_output_dir(apply=False))
