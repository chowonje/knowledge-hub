from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design import (
    PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_DESIGN_SCHEMA_ID,
    SIDECAR_ORACLE_PACK_SCHEMA_ID,
    ZERO_COUNTER_KEYS,
    build_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design,
    render_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design_markdown,
    write_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design_reports,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_FIXTURE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "schemas"
    / "fixtures"
    / "paper-parsed-artifact-pymupdf-quality-repair-sidecar-oracle-comparison.v1.fixture.json"
)
DESIGN_FIXTURE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "schemas"
    / "fixtures"
    / "paper-parsed-artifact-pymupdf-quality-repair-sidecar-oracle-candidate-pack-design.v1.fixture.json"
)
PACK_FIXTURE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "schemas"
    / "fixtures"
    / "paper-parsed-artifact-pymupdf-quality-repair-sidecar-oracle-pack.v1.fixture.json"
)


def _comparison_fixture() -> dict:
    return json.loads(COMPARISON_FIXTURE_PATH.read_text(encoding="utf-8"))


def test_candidate_pack_design_builds_from_missing_oracle_comparison_fixture() -> None:
    report = build_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design(
        comparison_report=_comparison_fixture()
    )

    assert report["schema"] == PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_DESIGN_SCHEMA_ID
    assert report["status"] == "design_ready"
    assert report["gate"]["decision"] == "ready_for_sidecar_oracle_candidate_pack_executor_dry_run"
    assert report["gate"]["nextTranche"] == (
        "parsed_artifact_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run"
    )
    assert report["request"]["designedPackSchemaId"] == SIDECAR_ORACLE_PACK_SCHEMA_ID
    assert report["counts"]["inputPlanRows"] == 1
    assert report["counts"]["oracleRequirementRows"] == 1
    assert report["counts"]["oracleMissingSignalRows"] == 1
    assert report["counts"]["candidatePackDesignRows"] == 1
    assert report["counts"]["sidecarOraclePackSchemaDefinedRows"] == 1
    assert report["counts"]["sidecarParserProfileRows"] == 1
    assert report["counts"]["futureExecutorDryRunReadyRows"] == 1
    assert report["counts"]["futureParserInvocationRows"] == 0
    assert report["counts"]["safeApplyCandidateRows"] == 0
    for key in ZERO_COUNTER_KEYS:
        assert report["counts"][key] == 0

    row = report["candidatePackDesignRows"][0]
    assert row["requirement"] == "table_body_cell_oracle"
    assert row["component"] == "table_numeric_region_locator"
    assert row["candidateParsers"] == ["camelot_local", "pdfplumber_local"]
    assert row["bboxOrLocatorRequiredForAgreement"] is True
    assert row["futureExecutorAction"] == "populate_local_oracle_candidate_row"
    assert validate_payload(
        report,
        PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok


def test_candidate_pack_design_blocks_on_unexpected_parent_gate() -> None:
    comparison = _comparison_fixture()
    comparison["gate"]["decision"] = "ready_for_pymupdf_quality_repair_safe_apply_subset_review"

    report = build_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design(
        comparison_report=comparison
    )

    assert report["status"] == "blocked"
    assert report["gate"]["decision"] == "blocked_until_sidecar_oracle_candidate_pack_design_gap_resolved"
    assert "sidecar_oracle_comparison_gate_not_waiting_for_candidate_pack" in report["gate"]["schemaViolations"]
    assert report["counts"]["schemaViolationCount"] == 1


def test_candidate_pack_design_blocks_on_unsafe_upstream_counter() -> None:
    comparison = copy.deepcopy(_comparison_fixture())
    comparison["counts"]["databaseMutationRows"] = 1

    report = build_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design(
        comparison_report=comparison
    )

    assert report["status"] == "blocked"
    assert "databaseMutationRows_nonzero" in report["gate"]["unsafeUpstreamFlags"]
    assert report["counts"]["candidatePackDesignRows"] == 1
    assert report["counts"]["databaseMutationRows"] == 0


def test_candidate_pack_design_and_pack_schema_fixtures_validate() -> None:
    design_fixture = json.loads(DESIGN_FIXTURE_PATH.read_text(encoding="utf-8"))
    pack_fixture = json.loads(PACK_FIXTURE_PATH.read_text(encoding="utf-8"))

    design_result = validate_payload(
        design_fixture,
        PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_DESIGN_SCHEMA_ID,
        strict=True,
    )
    pack_result = validate_payload(pack_fixture, SIDECAR_ORACLE_PACK_SCHEMA_ID, strict=True)

    assert design_result.ok, design_result.errors
    assert pack_result.ok, pack_result.errors


def test_candidate_pack_design_writer_outputs_path_safe_reports(tmp_path: Path) -> None:
    report = build_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design(
        comparison_report=_comparison_fixture()
    )
    paths = write_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design_reports(
        report,
        tmp_path / "reports",
    )
    written = Path(paths["json"]).read_text(encoding="utf-8")
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    payload = json.loads(written)

    result = validate_payload(
        payload,
        PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_DESIGN_SCHEMA_ID,
        strict=True,
    )
    assert result.ok, result.errors
    assert "ready_for_sidecar_oracle_candidate_pack_executor_dry_run" in markdown
    assert "Report-only design" in markdown
    assert str(tmp_path) not in written
    assert str(tmp_path) not in markdown
    assert str(tmp_path) not in render_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design_markdown(report)
