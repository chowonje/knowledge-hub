from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run import (
    PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_DESIGN_SCHEMA_ID,
    PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_EXECUTOR_DRY_RUN_SCHEMA_ID,
    SIDECAR_ORACLE_PACK_SCHEMA_ID,
    ZERO_COUNTER_KEYS,
    build_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run,
    render_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run_markdown,
    write_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run_reports,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESIGN_FIXTURE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "schemas"
    / "fixtures"
    / "paper-parsed-artifact-pymupdf-quality-repair-sidecar-oracle-candidate-pack-design.v1.fixture.json"
)
EXECUTOR_FIXTURE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "schemas"
    / "fixtures"
    / "paper-parsed-artifact-pymupdf-quality-repair-sidecar-oracle-candidate-pack-executor-dry-run.v1.fixture.json"
)


def _design_fixture() -> dict:
    return json.loads(DESIGN_FIXTURE_PATH.read_text(encoding="utf-8"))


def test_candidate_pack_executor_dry_run_builds_from_design_fixture() -> None:
    report = build_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run(
        design_report=_design_fixture()
    )

    assert report["schema"] == PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert report["status"] == "executor_dry_run_complete"
    assert report["gate"]["decision"] == "ready_for_sidecar_oracle_candidate_pack_local_dependency_probe"
    assert report["gate"]["nextTranche"] == (
        "parsed_artifact_pymupdf_quality_repair_sidecar_oracle_candidate_pack_local_dependency_probe"
    )
    assert report["request"]["inputSchema"] == PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_DESIGN_SCHEMA_ID
    assert report["request"]["targetPackSchemaId"] == SIDECAR_ORACLE_PACK_SCHEMA_ID
    assert report["request"]["sidecarParserInvocationAllowed"] is False
    assert report["counts"]["inputDesignRows"] == 1
    assert report["counts"]["dryRunPlanRows"] == 1
    assert report["counts"]["plannedOraclePackRows"] == 1
    assert report["counts"]["plannedLocalProbeRows"] == 1
    assert report["counts"]["plannedCandidateParserReferenceRows"] == 2
    assert report["counts"]["localParserDependencyProbeRows"] == 0
    assert report["counts"]["sidecarParserInvokedRows"] == 0
    assert report["counts"]["oraclePackWriteRows"] == 0
    assert report["counts"]["actualOracleRows"] == 0
    assert report["counts"]["safeApplyCandidateRows"] == 0
    for key in ZERO_COUNTER_KEYS:
        assert report["counts"][key] == 0

    row = report["dryRunPlanRows"][0]
    assert row["requirement"] == "table_body_cell_oracle"
    assert row["candidateParsers"] == ["camelot_local", "pdfplumber_local"]
    assert row["plannedExecutionMode"] == "dry_run_only_no_sidecar_parser_invocation"
    assert row["localParserDependencyProbeRequired"] is True
    assert row["sidecarParserInvoked"] is False
    assert row["oraclePackRowWritePlanned"] is False
    assert row["safeApplyCandidate"] is False

    result = validate_payload(
        report,
        PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    assert result.ok, result.errors


def test_candidate_pack_executor_dry_run_blocks_on_unexpected_parent_gate() -> None:
    design = _design_fixture()
    design["gate"]["decision"] = "blocked_until_sidecar_oracle_candidate_pack_design_gap_resolved"

    report = build_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run(
        design_report=design
    )

    assert report["status"] == "blocked"
    assert report["gate"]["decision"] == (
        "blocked_until_sidecar_oracle_candidate_pack_executor_dry_run_gap_resolved"
    )
    assert "candidate_pack_design_gate_not_ready_for_executor_dry_run" in report["gate"]["schemaViolations"]
    assert report["counts"]["schemaViolationCount"] == 1


def test_candidate_pack_executor_dry_run_blocks_on_unsafe_upstream_counter() -> None:
    design = copy.deepcopy(_design_fixture())
    design["counts"]["databaseMutationRows"] = 1

    report = build_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run(
        design_report=design
    )

    assert report["status"] == "blocked"
    assert "databaseMutationRows_nonzero" in report["gate"]["unsafeUpstreamFlags"]
    assert report["counts"]["dryRunPlanRows"] == 1
    assert report["counts"]["databaseMutationRows"] == 0


def test_candidate_pack_executor_dry_run_schema_fixture_validates() -> None:
    fixture = json.loads(EXECUTOR_FIXTURE_PATH.read_text(encoding="utf-8"))

    result = validate_payload(
        fixture,
        PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    )

    assert result.ok, result.errors


def test_candidate_pack_executor_dry_run_writer_outputs_path_safe_reports(tmp_path: Path) -> None:
    report = build_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run(
        design_report=_design_fixture()
    )
    paths = write_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run_reports(
        report,
        tmp_path / "reports",
    )
    written = Path(paths["json"]).read_text(encoding="utf-8")
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    payload = json.loads(written)

    result = validate_payload(
        payload,
        PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    assert result.ok, result.errors
    assert "ready_for_sidecar_oracle_candidate_pack_local_dependency_probe" in markdown
    assert "Report-only executor dry-run" in markdown
    assert str(tmp_path) not in written
    assert str(tmp_path) not in markdown
    assert str(tmp_path) not in render_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run_markdown(
        report
    )
