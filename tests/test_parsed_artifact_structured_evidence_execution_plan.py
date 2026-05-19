from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_structured_evidence_execution_plan import (
    EXECUTION_STATUS_BLOCKED_MISSING_LOCATION,
    EXECUTION_STATUS_BLOCKED_MISSING_SOURCE_HASH,
    EXECUTION_STATUS_DRY_RUN_READY,
    EXECUTION_STATUS_BLOCKED_NON_READY_INPUT,
    PARSED_ARTIFACT_STRUCTURED_EVIDENCE_EXECUTION_PLAN_SCHEMA_ID,
    build_parsed_artifact_structured_evidence_execution_plan,
    write_parsed_artifact_structured_evidence_execution_plan_reports,
)
from knowledge_hub.papers.parsed_artifact_structured_evidence_readiness_audit import (
    PARSED_ARTIFACT_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID,
    STATUS_READY,
    STATUS_BLOCKED_AMBIGUOUS_PDF_REGION,
    STATUS_BLOCKED_MANUAL_OR_LATER_EXTRACTOR,
    STATUS_BLOCKED_MISSING_LABEL_NUMBER_DISAMBIGUATION,
    STATUS_BLOCKED_MISSING_SOURCE_HASH,
    STATUS_BLOCKED_PDF_REGION_ONLY,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _readiness_report_path(tmp_path: Path) -> Path:
    return tmp_path / "parsed-artifact-readiness.json"


def _readiness_row(
    *,
    paper_id: str,
    artifact_type: str,
    status: str,
    source_content_hash: str,
    source_candidate_id: str,
    page: int | None,
    chars_start: int | None,
    chars_end: int | None,
    bbox: list[float],
    block_indexes: list[int],
) -> dict:
    return {
        "paper_id": paper_id,
        "artifact_type": artifact_type,
        "readiness_status": status,
        "source_candidate_id": source_candidate_id,
        "sourceContentHash": source_content_hash,
        "source_file": f"{paper_id}.pdf",
        "page": page,
        "bbox": bbox,
        "blockIndexes": block_indexes,
        "chars_start": chars_start,
        "chars_end": chars_end,
    }


def _build_readiness_report(tmp_path: Path, *, wrong_schema: bool = False) -> Path:
    schema = "invalid.schema.v1" if wrong_schema else PARSED_ARTIFACT_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID
    return _write_json(
        _readiness_report_path(tmp_path),
        {
            "schema": schema,
            "status": "ok",
            "rows": [
                _readiness_row(
                    paper_id="paper-1",
                    artifact_type="section",
                    status=STATUS_READY,
                    source_content_hash="ready-hash",
                    source_candidate_id="ready-section-1",
                    page=1,
                    chars_start=10,
                    chars_end=20,
                    bbox=[0.1, 0.2, 10.0, 12.5],
                    block_indexes=[1, 2],
                ),
                _readiness_row(
                    paper_id="paper-1",
                    artifact_type="table",
                    status=STATUS_BLOCKED_MISSING_SOURCE_HASH,
                    source_content_hash="",
                    source_candidate_id="missing-hash-table-1",
                    page=2,
                    chars_start=1,
                    chars_end=2,
                    bbox=[1.0, 1.0, 1.0, 1.0],
                    block_indexes=[3],
                ),
                _readiness_row(
                    paper_id="paper-1",
                    artifact_type="figure",
                    status=STATUS_BLOCKED_AMBIGUOUS_PDF_REGION,
                    source_content_hash="figure-hash-amb",
                    source_candidate_id="ambiguous-figure-1",
                    page=3,
                    chars_start=None,
                    chars_end=None,
                    bbox=[],
                    block_indexes=[],
                ),
                _readiness_row(
                    paper_id="paper-2",
                    artifact_type="equation",
                    status=STATUS_BLOCKED_MISSING_LABEL_NUMBER_DISAMBIGUATION,
                    source_content_hash="eq-hash",
                    source_candidate_id="missing-label-eq-1",
                    page=4,
                    chars_start=11,
                    chars_end=22,
                    bbox=[4.0, 5.0, 6.0, 7.0],
                    block_indexes=[7],
                ),
                _readiness_row(
                    paper_id="paper-2",
                    artifact_type="figure",
                    status=STATUS_BLOCKED_PDF_REGION_ONLY,
                    source_content_hash="figure-region-hash",
                    source_candidate_id="pdf-region-only-fig-1",
                    page=5,
                    chars_start=1,
                    chars_end=2,
                    bbox=[0.0, 1.0, 2.0, 3.0],
                    block_indexes=[2, 3],
                ),
                _readiness_row(
                    paper_id="paper-3",
                    artifact_type="equation",
                    status=STATUS_BLOCKED_MANUAL_OR_LATER_EXTRACTOR,
                    source_content_hash="manual-hash",
                    source_candidate_id="manual-eq-1",
                    page=6,
                    chars_start=33,
                    chars_end=44,
                    bbox=[2.0, 3.0, 4.0, 5.0],
                    block_indexes=[8],
                ),
            ],
        },
    )


def test_parsed_artifact_structured_evidence_execution_plan_classifies_all_readiness_statuses(tmp_path: Path) -> None:
    readiness_path = _build_readiness_report(tmp_path)
    payload = build_parsed_artifact_structured_evidence_execution_plan(
        readiness_report=readiness_path,
    )

    assert payload["schema"] == PARSED_ARTIFACT_STRUCTURED_EVIDENCE_EXECUTION_PLAN_SCHEMA_ID
    assert payload["status"] == "ok"
    assert validate_payload(payload, PARSED_ARTIFACT_STRUCTURED_EVIDENCE_EXECUTION_PLAN_SCHEMA_ID, strict=True).ok
    assert payload["input"]["readinessReportSchema"] == PARSED_ARTIFACT_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID
    assert payload["counts"]["inputRows"] == 6
    assert payload["counts"]["plannedRows"] == 6

    by_readiness = {
        row["readiness_status"]: row["execution_status"] for row in payload["rows"]
    }
    assert by_readiness[STATUS_READY] == EXECUTION_STATUS_DRY_RUN_READY
    assert by_readiness[STATUS_BLOCKED_MISSING_SOURCE_HASH] == EXECUTION_STATUS_BLOCKED_MISSING_SOURCE_HASH
    assert by_readiness[STATUS_BLOCKED_AMBIGUOUS_PDF_REGION] == EXECUTION_STATUS_BLOCKED_NON_READY_INPUT
    assert by_readiness[STATUS_BLOCKED_MISSING_LABEL_NUMBER_DISAMBIGUATION] == EXECUTION_STATUS_BLOCKED_NON_READY_INPUT
    assert by_readiness[STATUS_BLOCKED_PDF_REGION_ONLY] == EXECUTION_STATUS_BLOCKED_NON_READY_INPUT
    assert by_readiness[STATUS_BLOCKED_MANUAL_OR_LATER_EXTRACTOR] == EXECUTION_STATUS_BLOCKED_NON_READY_INPUT
    assert payload["counts"]["dryRunReadyRows"] == 1
    assert payload["counts"]["blockedMissingSourceHashRows"] == 1
    assert payload["counts"]["blockedMissingLocationRows"] == 0
    assert payload["counts"]["blockedUnknownWriteTargetRows"] == 0
    assert payload["counts"]["blockedNonReadyInputRows"] == 4
    assert payload["counts"]["sourceSpanCreatedRows"] == 0
    assert payload["counts"]["strictEvidenceCreatedRows"] == 0
    assert payload["counts"]["databaseMutationRows"] == 0
    assert payload["counts"]["schemaViolationCount"] == 0
    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["dryRunOnly"] is True
    assert payload["policy"]["sourceSpanCreated"] is False
    assert payload["policy"]["canonicalParsedArtifactsWritten"] is False
    assert payload["rows"][0]["execution_status"] == EXECUTION_STATUS_DRY_RUN_READY
    assert payload["rows"][0]["would_create_source_span"] is True
    assert payload["rows"][0]["planned_operation"] == "plan_source_span_creation"
    assert payload["rows"][0]["planned_write_target"] == "parsed_artifact_source_span_candidate_store"


def test_parsed_artifact_structured_evidence_execution_plan_blocks_wrong_readiness_schema(tmp_path: Path) -> None:
    readiness_path = _build_readiness_report(tmp_path, wrong_schema=True)

    payload = build_parsed_artifact_structured_evidence_execution_plan(
        readiness_report=readiness_path,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "readiness_report_schema_mismatch" in payload["gate"]["schemaViolations"]
    assert "readiness_report_unreadable" not in payload["warnings"]
    assert payload["counts"]["inputRows"] == 0
    assert payload["counts"]["plannedRows"] == 0
    assert payload["counts"]["schemaViolationCount"] == 1
    assert validate_payload(payload, PARSED_ARTIFACT_STRUCTURED_EVIDENCE_EXECUTION_PLAN_SCHEMA_ID, strict=True).ok


def test_parsed_artifact_structured_evidence_execution_plan_writer_outputs_schema_valid_report_summary_markdown(tmp_path: Path) -> None:
    readiness_path = _build_readiness_report(tmp_path)
    payload = build_parsed_artifact_structured_evidence_execution_plan(
        readiness_report=readiness_path,
    )
    paths = write_parsed_artifact_structured_evidence_execution_plan_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert validate_payload(report, PARSED_ARTIFACT_STRUCTURED_EVIDENCE_EXECUTION_PLAN_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, PARSED_ARTIFACT_STRUCTURED_EVIDENCE_EXECUTION_PLAN_SCHEMA_ID, strict=True).ok
    assert "Parsed Artifact Structured-Evidence Execution Plan" in markdown
    assert "ready plan rows: 1" in markdown
