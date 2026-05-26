from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.table_cell_provenance_feasibility_audit import (
    TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID,
)
from knowledge_hub.papers.table_cell_structured_evidence_readiness_audit import (
    TABLE_CELL_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID,
    STATUS_READY,
    STATUS_BLOCKED_INPUT_REPORT_MISSING,
    STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
    STATUS_BLOCKED_MISSING_CELL_COORDINATES,
    STATUS_BLOCKED_MISSING_CELL_TEXT,
    STATUS_BLOCKED_MISSING_LOCATOR,
    STATUS_BLOCKED_MISSING_SOURCE_HASH,
    STATUS_BLOCKED_MISSING_TABLE_ID,
    STATUS_BLOCKED_NUMERIC_NORMALIZATION_UNVERIFIED,
    build_table_cell_structured_evidence_readiness_audit,
    write_table_cell_structured_evidence_readiness_audit_reports,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _feasibility_row(
    *,
    source_candidate_id: str,
    table_region_candidate_id: str,
    paper_id: str,
    source_hash: str,
    strict_blockers: list[str] | None = None,
    row_text: str = "Cell A, Cell B",
    row_column_text_available: bool = True,
    cell_bbox_available: bool = True,
    has_locator: bool = True,
    non_empty_table_cell_count: int | None = None,
    cell_bbox_count: int | None = None,
    bbox: list[float] | None = None,
    page: int | None = 1,
    block_indexes: list[int] | None = None,
) -> dict:
    chars_start = 1 if has_locator else None
    chars_end = 10 if has_locator else None
    if not has_locator:
        page = None
        chars_start = None
        chars_end = None
    if not row_column_text_available:
        row_text = ""
    return {
        "audit_id": f"table-cell-provenance-feasibility:{source_candidate_id}",
        "table_region_candidate_id": table_region_candidate_id,
        "source_candidate_id": source_candidate_id,
        "paper_id": paper_id,
        "candidate_type": "table_cell_provenance_feasibility_candidate",
        "source_parser": "mineru+pymupdf_alignment",
        "candidate_text": row_text,
        "table_label": "Table 1",
        "caption_text": "Table metrics",
        "caption_alignment_status": "aligned",
        "caption_alignment_method": "exact",
        "caption_chars_start": chars_start,
        "caption_chars_end": chars_end,
        "caption_page": page,
        "sourceContentHash": source_hash,
        "sourceContentHashSource": "mineru",
        "caption_source_span_available": True,
        "table_region_bbox_available": True,
        "layout_element_count": 1,
        "normalizer_candidate_id": source_candidate_id,
        "normalizer_report_path": "/tmp/normalizer.json",
        "normalizer_schema": "knowledge-hub.paper.mineru-normalizer-audit.v1",
        "normalizer_match": True,
        "table_structure_available": True,
        "row_column_text_available": row_column_text_available,
        "table_row_count": 2,
        "table_max_column_count": 2,
        "table_cell_count": 4,
        "non_empty_table_cell_count": 4 if non_empty_table_cell_count is None else non_empty_table_cell_count,
        "header_like_cell_count": 1,
        "rowspan_cell_count": 0,
        "colspan_cell_count": 0,
        "cell_bbox_count": 4 if cell_bbox_count is None else cell_bbox_count,
        "cell_source_span_count": 4,
        "cell_source_hash_count": 4,
        "sample_cells": [],
        "cell_bbox_available": cell_bbox_available,
        "cell_source_span_available": True,
        "cell_source_hash_backed": True,
        "table_cell_citation_grade": False,
        "feasibility_status": "cell_text_structure_candidate_non_strict",
        "confidence": 0.98,
        "evidence_tier": "table_cell_provenance_feasibility_candidate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": strict_blockers or [],
        "non_strict_reason": [],
        "bbox": [1.0, 2.0, 5.0, 6.0] if bbox is None else bbox,
        "blockIndexes": [3] if block_indexes is None else block_indexes,
    }


def _report_path(tmp_path: Path, *rows: dict) -> Path:
    return _write(
        tmp_path / "table-cell-provenance-feasibility.json",
        {
            "schema": TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID,
            "rows": [dict(item) for item in rows],
        },
    )


def test_table_cell_structured_evidence_readiness_audit_classifies_all_readiness_statuses(tmp_path: Path) -> None:
    report_path = _report_path(
        tmp_path,
        _feasibility_row(
            source_candidate_id="ready-1",
            table_region_candidate_id="tbl-1",
            paper_id="paper-1",
            source_hash="hash-ready",
        ),
        _feasibility_row(
            source_candidate_id="missing-table-id",
            table_region_candidate_id="",
            paper_id="paper-1",
            source_hash="hash-table-id",
        ),
        _feasibility_row(
            source_candidate_id="missing-cell-text",
            table_region_candidate_id="tbl-3",
            paper_id="paper-1",
            source_hash="hash-text",
            row_column_text_available=False,
            non_empty_table_cell_count=0,
        ),
        _feasibility_row(
            source_candidate_id="missing-cell-coords",
            table_region_candidate_id="tbl-4",
            paper_id="paper-2",
            source_hash="hash-coords",
            cell_bbox_available=False,
            cell_bbox_count=0,
            bbox=[],
        ),
        _feasibility_row(
            source_candidate_id="missing-source-hash",
            table_region_candidate_id="tbl-5",
            paper_id="paper-2",
            source_hash="",
            strict_blockers=["source hash missing"],
        ),
        _feasibility_row(
            source_candidate_id="missing-locator",
            table_region_candidate_id="tbl-6",
            paper_id="paper-2",
            source_hash="hash-locator",
            has_locator=False,
            block_indexes=[],
            page=None,
            bbox=[],
        ),
        _feasibility_row(
            source_candidate_id="unverified-number",
            table_region_candidate_id="tbl-7",
            paper_id="paper-3",
            source_hash="hash-number",
            strict_blockers=["numeric_normalization_unverified"],
        ),
    )

    payload = build_table_cell_structured_evidence_readiness_audit(
        table_cell_provenance_feasibility_report=report_path
    )

    assert payload["schema"] == TABLE_CELL_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID
    assert payload["status"] == "ok"
    assert validate_payload(payload, TABLE_CELL_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["tableCellStructuredAuthorityCandidateOnlyRows"] == 1
    assert payload["counts"]["blockedMissingTableIdRows"] == 1
    assert payload["counts"]["blockedMissingCellTextRows"] == 1
    assert payload["counts"]["blockedMissingCellCoordinatesRows"] == 1
    assert payload["counts"]["blockedMissingSourceHashRows"] == 1
    assert payload["counts"]["blockedMissingLocatorRows"] == 1
    assert payload["counts"]["blockedNumericNormalizationUnverifiedRows"] == 1
    assert payload["counts"]["blockedInputReportMissingRows"] == 0
    assert payload["counts"]["blockedInputSchemaViolationRows"] == 0
    assert payload["input"]["tableCellProvenanceFeasibilitySchema"] == TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID

    by_status = {
        row["readiness_status"]: row["recommended_action"] for row in payload["rows"]
    }
    assert by_status[STATUS_READY] == "route_for_table_cell_structured_evidence_execution_plan"
    assert (
        by_status[STATUS_BLOCKED_MISSING_TABLE_ID]
        == "recover_table_region_candidate_id_before_table_cell_readiness"
    )
    assert (
        by_status[STATUS_BLOCKED_MISSING_CELL_TEXT]
        == "recover_table_cell_row_column_text_before_table_cell_readiness"
    )
    assert (
        by_status[STATUS_BLOCKED_MISSING_CELL_COORDINATES]
        == "recover_cell_bbox_or_coordinate_data_before_table_cell_readiness"
    )
    assert by_status[STATUS_BLOCKED_MISSING_SOURCE_HASH] == "recover_source_content_hash_before_table_cell_readiness"
    assert by_status[STATUS_BLOCKED_MISSING_LOCATOR] == "recover_table_cell_locator_before_table_cell_readiness"
    assert by_status[STATUS_BLOCKED_NUMERIC_NORMALIZATION_UNVERIFIED] == "verify_numeric_normalization_before_table_cell_readiness"
    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["designOnly"] is True
    assert payload["policy"]["databaseMutation"] is False


def test_table_cell_structured_evidence_readiness_audit_blocks_missing_input_report(tmp_path: Path) -> None:
    payload = build_table_cell_structured_evidence_readiness_audit(
        table_cell_provenance_feasibility_report=None
    )
    assert payload["status"] == "blocked"
    assert payload["counts"]["blockedInputReportMissingRows"] == 1
    assert payload["counts"]["blockedInputSchemaViolationRows"] == 0
    assert payload["counts"]["inputRows"] == 0
    assert payload["counts"]["targetRows"] == 0
    assert payload["counts"]["schemaViolationCount"] == 0
    assert payload["gate"]["decision"] == "blocked"
    assert "table_cell_provenance_feasibility_report_missing_or_unreadable" in payload["warnings"]
    assert STATUS_BLOCKED_INPUT_REPORT_MISSING in payload["warnings"]


def test_table_cell_structured_evidence_readiness_audit_blocks_wrong_input_schema(tmp_path: Path) -> None:
    payload = {
        "schema": "knowledge-hub.paper.wrong.schema.v1",
        "rows": [
            _feasibility_row(
                source_candidate_id="ready-1",
                table_region_candidate_id="tbl-1",
                paper_id="paper-1",
                source_hash="hash-1",
            )
        ],
    }
    report_path = _write(tmp_path / "wrong-schema.json", payload)
    result = build_table_cell_structured_evidence_readiness_audit(
        table_cell_provenance_feasibility_report=report_path
    )
    assert result["status"] == "blocked"
    assert result["counts"]["blockedInputSchemaViolationRows"] == 1
    assert result["counts"]["blockedInputReportMissingRows"] == 0
    assert result["counts"]["schemaViolationCount"] == 1
    assert result["counts"]["inputRows"] == 0
    assert result["counts"]["targetRows"] == 0
    assert STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION in result["warnings"]
    assert result["gate"]["decision"] == "blocked"
    assert validate_payload(result, TABLE_CELL_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID, strict=True).ok


def test_table_cell_structured_evidence_readiness_audit_writer_outputs_schema_valid_report_and_markdown(tmp_path: Path) -> None:
    report_path = _report_path(
        tmp_path,
        _feasibility_row(
            source_candidate_id="ready-1",
            table_region_candidate_id="tbl-1",
            paper_id="paper-1",
            source_hash="hash-ready",
        ),
    )
    payload = build_table_cell_structured_evidence_readiness_audit(
        table_cell_provenance_feasibility_report=report_path
    )
    paths = write_table_cell_structured_evidence_readiness_audit_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, TABLE_CELL_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, TABLE_CELL_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID, strict=True).ok
    assert "TableCell Structured-Evidence Readiness Audit" in markdown
    assert "authority-ready rows: 1" in markdown
