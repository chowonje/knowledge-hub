from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_structured_evidence_execution_plan import (
    EXECUTION_STATUS_BLOCKED_MISSING_LOCATION,
    EXECUTION_STATUS_BLOCKED_MISSING_SOURCE_HASH,
    EXECUTION_STATUS_BLOCKED_NON_READY_INPUT,
    EXECUTION_STATUS_BLOCKED_UNKNOWN_WRITE_TARGET,
    EXECUTION_STATUS_DRY_RUN_READY,
    PARSED_ARTIFACT_STRUCTURED_EVIDENCE_EXECUTION_PLAN_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_structured_evidence_write_target_contract_audit import (
    PARSED_ARTIFACT_STRUCTURED_EVIDENCE_WRITE_TARGET_CONTRACT_AUDIT_SCHEMA_ID,
    build_parsed_artifact_structured_evidence_write_target_contract_audit,
    write_parsed_artifact_structured_evidence_write_target_contract_audit_reports,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _execution_row(
    *,
    status: str,
    artifact_type: str,
    source_hash: str,
    source_candidate_id: str,
    planned_write_target: str,
    page: int | None,
    chars_start: int | None = None,
    chars_end: int | None = None,
    row_id: int = 0,
) -> dict:
    return {
        "plan_id": f"plan-{row_id:04d}",
        "source_readiness_row_id": f"readiness-{row_id:04d}",
        "paper_id": "paper-1",
        "artifact_type": artifact_type,
        "source_candidate_id": source_candidate_id,
        "sourceContentHash": source_hash,
        "source_file": f"paper-1.pdf",
        "page": page,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "blockIndexes": [1, 2],
        "planned_operation": "plan_source_span_creation",
        "planned_write_target": planned_write_target,
        "would_create_source_span": status == EXECUTION_STATUS_DRY_RUN_READY,
        "would_create_strict_evidence": False,
        "would_change_runtime": False,
        "would_mutate_database": False,
        "execution_status": status,
        "execution_blockers": [],
        "chars_start": chars_start,
        "chars_end": chars_end,
    }


def _build_execution_plan_report(tmp_path: Path, *, wrong_schema: bool = False) -> Path:
    schema = PARSED_ARTIFACT_STRUCTURED_EVIDENCE_EXECUTION_PLAN_SCHEMA_ID
    if wrong_schema:
        schema = "invalid.schema.v1"
    return _write_json(
        tmp_path / "execution-plan.json",
        {
            "schema": schema,
            "status": "ok",
            "rows": [
                _execution_row(
                    row_id=1,
                    status=EXECUTION_STATUS_DRY_RUN_READY,
                    artifact_type="section",
                    source_hash="known-hash",
                    source_candidate_id="ready-known-target",
                    planned_write_target="parsed_artifact_source_span_candidate_store",
                    page=1,
                    chars_start=10,
                    chars_end=20,
                ),
                _execution_row(
                    row_id=2,
                    status=EXECUTION_STATUS_DRY_RUN_READY,
                    artifact_type="figure",
                    source_hash="unknown-contract-hash",
                    source_candidate_id="ready-unknown-target",
                    planned_write_target="structured_evidence_candidate_store",
                    page=None,
                    chars_start=1,
                    chars_end=2,
                ),
                _execution_row(
                    row_id=3,
                    status=EXECUTION_STATUS_BLOCKED_MISSING_SOURCE_HASH,
                    artifact_type="table",
                    source_hash="",
                    source_candidate_id="blocked-hash",
                    planned_write_target="parsed_artifact_source_span_candidate_store",
                    page=3,
                ),
                _execution_row(
                    row_id=4,
                    status=EXECUTION_STATUS_BLOCKED_MISSING_LOCATION,
                    artifact_type="equation",
                    source_hash="missing-location-hash",
                    source_candidate_id="blocked-location",
                    planned_write_target="parsed_artifact_source_span_candidate_store",
                    page=None,
                    chars_start=None,
                    chars_end=None,
                ),
                _execution_row(
                    row_id=5,
                    status=EXECUTION_STATUS_BLOCKED_UNKNOWN_WRITE_TARGET,
                    artifact_type="section",
                    source_hash="blocked-unknown",
                    source_candidate_id="blocked-unknown-target",
                    planned_write_target="structured_evidence_candidate_store",
                    page=5,
                ),
                _execution_row(
                    row_id=6,
                    status=EXECUTION_STATUS_BLOCKED_NON_READY_INPUT,
                    artifact_type="equation",
                    source_hash="ok-hash",
                    source_candidate_id="blocked-non-ready",
                    planned_write_target="parsed_artifact_source_span_candidate_store",
                    page=6,
                ),
            ],
        },
    )


def test_parsed_artifact_structured_evidence_write_target_contract_audit_classifies_contract_readiness(
    tmp_path: Path,
) -> None:
    execution_plan_path = _build_execution_plan_report(tmp_path)

    payload = build_parsed_artifact_structured_evidence_write_target_contract_audit(
        execution_plan_report=execution_plan_path
    )

    assert payload["schema"] == PARSED_ARTIFACT_STRUCTURED_EVIDENCE_WRITE_TARGET_CONTRACT_AUDIT_SCHEMA_ID
    assert payload["status"] == "blocked"
    assert payload["counts"]["inputRows"] == 6
    assert payload["counts"]["readyInputRows"] == 2
    assert payload["counts"]["plannedRows"] == 6
    assert payload["counts"]["blockedUnknownWriteTargetRows"] == 1
    assert payload["counts"]["blockedMissingSourceHashRows"] == 1
    assert payload["counts"]["blockedMissingLocationRows"] == 1
    assert payload["counts"]["blockedNonReadyInputRows"] == 1
    assert payload["counts"]["writeTargetContractKnownRows"] == 6
    assert payload["counts"]["sourceSpanCreatedRows"] == 0
    assert payload["counts"]["strictEvidenceCreatedRows"] == 0
    assert payload["counts"]["databaseMutationRows"] == 0
    assert payload["counts"]["parserRoutingChangedRows"] == 0
    assert payload["counts"]["schemaViolationCount"] == 0
    assert len(payload["rows"]) == 6

    status_by_id = {
        row["plan_id"]: row["execution_status"] for row in payload["rows"]
    }
    assert status_by_id["plan-0001"] == EXECUTION_STATUS_DRY_RUN_READY
    assert status_by_id["plan-0002"] == EXECUTION_STATUS_DRY_RUN_READY
    assert status_by_id["plan-0003"] == EXECUTION_STATUS_BLOCKED_MISSING_SOURCE_HASH
    assert status_by_id["plan-0004"] == EXECUTION_STATUS_BLOCKED_MISSING_LOCATION
    assert status_by_id["plan-0005"] == EXECUTION_STATUS_BLOCKED_UNKNOWN_WRITE_TARGET
    assert status_by_id["plan-0006"] == EXECUTION_STATUS_BLOCKED_NON_READY_INPUT
    ready_rows = [
        row
        for row in payload["rows"]
        if row["execution_status"] == EXECUTION_STATUS_DRY_RUN_READY
    ]
    assert all(row["write_target_contract_known"] is True for row in ready_rows)
    assert all(row["write_target_contract_reference"] for row in ready_rows)
    assert all(row["write_target_contract_known"] is True for row in payload["rows"])
    assert payload["rows"][3]["rollback_strategy"] == "hold row until non-write-target blockers are resolved"
    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["dryRunOnly"] is True
    assert payload["policy"]["sourceSpanCreated"] is False
    assert payload["policy"]["canonicalParsedArtifactsWritten"] is False
    assert validate_payload(payload, PARSED_ARTIFACT_STRUCTURED_EVIDENCE_WRITE_TARGET_CONTRACT_AUDIT_SCHEMA_ID, strict=True).ok


def test_parsed_artifact_structured_evidence_write_target_contract_audit_blocks_wrong_execution_plan_schema(
    tmp_path: Path,
) -> None:
    execution_plan_path = _build_execution_plan_report(tmp_path, wrong_schema=True)

    payload = build_parsed_artifact_structured_evidence_write_target_contract_audit(
        execution_plan_report=execution_plan_path
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "execution_plan_schema_mismatch" in payload["gate"]["schemaViolations"]
    assert payload["counts"]["inputRows"] == 0
    assert payload["counts"]["plannedRows"] == 0
    assert payload["counts"]["schemaViolationCount"] == 1
    assert validate_payload(payload, PARSED_ARTIFACT_STRUCTURED_EVIDENCE_WRITE_TARGET_CONTRACT_AUDIT_SCHEMA_ID, strict=True).ok


def test_parsed_artifact_structured_evidence_write_target_contract_audit_writer_outputs_schema_valid_report_summary_markdown(
    tmp_path: Path,
) -> None:
    execution_plan_path = _build_execution_plan_report(tmp_path)
    payload = build_parsed_artifact_structured_evidence_write_target_contract_audit(
        execution_plan_report=execution_plan_path
    )
    paths = write_parsed_artifact_structured_evidence_write_target_contract_audit_reports(
        payload,
        tmp_path / "reports",
    )

    assert set(paths.keys()) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert report["schema"] == PARSED_ARTIFACT_STRUCTURED_EVIDENCE_WRITE_TARGET_CONTRACT_AUDIT_SCHEMA_ID
    assert summary["counts"]["plannedRows"] == 6
    assert validate_payload(report, PARSED_ARTIFACT_STRUCTURED_EVIDENCE_WRITE_TARGET_CONTRACT_AUDIT_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, PARSED_ARTIFACT_STRUCTURED_EVIDENCE_WRITE_TARGET_CONTRACT_AUDIT_SCHEMA_ID, strict=True).ok
    assert "Parsed Artifact Structured-Evidence Write Target Contract Audit" in markdown
    assert "input rows: 6" in markdown
    assert "contract-known rows:" in markdown
