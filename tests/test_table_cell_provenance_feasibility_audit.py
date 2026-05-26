from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.table_cell_provenance_feasibility_audit import (
    TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID,
    build_table_cell_provenance_feasibility_audit,
    write_table_cell_provenance_feasibility_audit_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _reports(root: Path, *, wrong_schema: bool = False, with_source_cell: bool = False) -> tuple[Path, Path]:
    table_schema = "knowledge-hub.paper.table-region-candidate-report.v1"
    source_schema = "knowledge-hub.paper.mineru-source-alignment-audit.v1"
    if wrong_schema:
        table_schema = "example.wrong.table"
        source_schema = "example.wrong.source"
    normalizer_path = _write(
        root,
        "normalizer/paper-1/mineru-normalizer-candidates.json",
        {
            "schema": "knowledge-hub.paper.mineru-normalizer-audit.v1",
            "candidates": [
                {
                    "candidate_id": "paper-1:table:0001",
                    "paper_id": "paper-1",
                    "candidate_type": "table_candidate",
                    "text": "Table 1: Model results.",
                    "tableRows": [
                        [
                            {
                                "text": "Model",
                                "rowspan": 1,
                                "colspan": 1,
                                **({"bbox": [1, 2, 3, 4], "chars_start": 10, "chars_end": 15, "sourceContentHash": "cell-hash"} if with_source_cell else {}),
                            },
                            {"text": "Score", "rowspan": 1, "colspan": 1},
                        ],
                        [{"text": "A", "rowspan": 1, "colspan": 1}, {"text": "1.0", "rowspan": 1, "colspan": 1}],
                    ],
                }
            ],
        },
    )
    table_path = _write(
        root,
        "table-region-candidates.json",
        {
            "schema": table_schema,
            "candidates": [
                {
                    "candidate_id": "tableregion:paper-1:0001",
                    "candidate_type": "table_region_candidate",
                    "source_candidate_id": "paper-1:table:0001",
                    "paper_id": "paper-1",
                    "candidate_text": "Table 1: Model results.",
                    "table_label": "Table 1",
                    "caption_text": "Model results.",
                    "canonical_alignment_status": "aligned",
                    "alignment_method": "exact",
                    "chars_start": 100,
                    "chars_end": 123,
                    "page": 4,
                    "sourceContentHash": "source-hash",
                    "sourceContentHashSource": "manifest",
                    "layout_element_ids": ["mineru:1"],
                    "bbox": [10, 20, 200, 220],
                    "confidence": 0.99,
                    "strict_blockers": ["table_cell_provenance_missing"],
                    "strict_eligible": False,
                    "citation_grade": False,
                },
                {
                    "candidate_id": "tableregion:paper-1:0002",
                    "candidate_type": "table_region_candidate",
                    "source_candidate_id": "paper-1:table:9999",
                    "paper_id": "paper-1",
                    "candidate_text": "Table 2: Missing normalizer.",
                    "table_label": "Table 2",
                    "caption_text": "Missing normalizer.",
                    "canonical_alignment_status": "failed",
                    "alignment_method": "none",
                    "chars_start": None,
                    "chars_end": None,
                    "page": None,
                    "sourceContentHash": "source-hash",
                    "layout_element_ids": ["mineru:2"],
                    "bbox": [30, 40, 250, 280],
                    "confidence": 0.0,
                    "strict_blockers": ["missing_chars_start_end", "missing_page"],
                    "strict_eligible": False,
                    "citation_grade": False,
                },
            ],
        },
    )
    source_path = _write(
        root,
        "mineru-source-alignment-report.json",
        {
            "schema": source_schema,
            "papers": [
                {
                    "paperId": "paper-1",
                    "input": {"mineruNormalizerCandidatesPath": str(normalizer_path)},
                }
            ],
        },
    )
    return table_path, source_path


def test_table_cell_provenance_feasibility_reports_rows_but_no_cell_evidence(tmp_path: Path) -> None:
    table_path, source_path = _reports(tmp_path)

    payload = build_table_cell_provenance_feasibility_audit(
        table_region_report=table_path,
        mineru_source_alignment_report=source_path,
    )

    assert payload["schema"] == TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID
    assert validate_payload(payload, TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["auditedTableRegionCandidates"] == 2
    assert payload["counts"]["normalizerTableMatches"] == 1
    assert payload["counts"]["tableStructureCandidates"] == 1
    assert payload["counts"]["rowColumnTextCandidates"] == 1
    assert payload["counts"]["totalTableRows"] == 2
    assert payload["counts"]["totalTableCells"] == 4
    assert payload["counts"]["cellBboxCandidates"] == 0
    assert payload["counts"]["cellSourceSpanCandidates"] == 0
    assert payload["counts"]["tableCellCitationGradeCandidates"] == 0


def test_table_cell_rows_are_candidate_only_even_when_cells_exist(tmp_path: Path) -> None:
    table_path, source_path = _reports(tmp_path)

    payload = build_table_cell_provenance_feasibility_audit(
        table_region_report=table_path,
        mineru_source_alignment_report=source_path,
    )
    row = next(item for item in payload["rows"] if item["normalizer_match"])

    assert row["table_structure_available"] is True
    assert row["row_column_text_available"] is True
    assert row["cell_bbox_available"] is False
    assert row["cell_source_span_available"] is False
    assert row["cell_source_hash_backed"] is False
    assert row["table_cell_citation_grade"] is False
    assert row["strict_eligible"] is False
    assert row["citation_grade"] is False
    assert row["runtime_evidence"] is False
    assert row["evidence_tier"] == "table_cell_provenance_feasibility_candidate_only"
    assert row["feasibility_status"] == "table_structure_candidate_no_cell_provenance"
    assert "table_cell_bbox_missing" in row["strict_blockers"]
    assert "table_cell_chars_start_end_missing" in row["strict_blockers"]
    assert payload["policy"]["tableCellEvidenceCreated"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["runtimePromotionAllowed"] is False


def test_missing_normalizer_table_is_blocked_not_green(tmp_path: Path) -> None:
    table_path, source_path = _reports(tmp_path)

    payload = build_table_cell_provenance_feasibility_audit(
        table_region_report=table_path,
        mineru_source_alignment_report=source_path,
    )
    blocked = next(item for item in payload["rows"] if item["source_candidate_id"] == "paper-1:table:9999")

    assert blocked["normalizer_match"] is False
    assert blocked["table_structure_available"] is False
    assert blocked["feasibility_status"] == "blocked_missing_normalizer_table"
    assert "missing_normalizer_table_candidate" in blocked["strict_blockers"]
    assert blocked["strict_eligible"] is False


def test_source_span_like_cell_still_does_not_create_citation_grade_evidence(tmp_path: Path) -> None:
    table_path, source_path = _reports(tmp_path, with_source_cell=True)

    payload = build_table_cell_provenance_feasibility_audit(
        table_region_report=table_path,
        mineru_source_alignment_report=source_path,
    )
    row = next(item for item in payload["rows"] if item["normalizer_match"])

    assert row["cell_bbox_available"] is True
    assert row["cell_source_span_available"] is True
    assert row["cell_source_hash_backed"] is True
    assert row["feasibility_status"] == "cell_text_structure_candidate_non_strict"
    assert row["table_cell_citation_grade"] is False
    assert row["strict_eligible"] is False
    assert row["citation_grade"] is False
    assert "table_cell_provenance_feasibility_audit_only" in row["strict_blockers"]


def test_table_cell_provenance_feasibility_blocks_wrong_input_schema_ids(tmp_path: Path) -> None:
    table_path, source_path = _reports(tmp_path, wrong_schema=True)

    payload = build_table_cell_provenance_feasibility_audit(
        table_region_report=table_path,
        mineru_source_alignment_report=source_path,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert set(payload["gate"]["schemaViolations"]) == {
        "table_region_candidate_report_schema_mismatch",
        "mineru_source_alignment_report_schema_mismatch",
    }


def test_table_cell_provenance_feasibility_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    table_path, source_path = _reports(tmp_path / "input")
    payload = build_table_cell_provenance_feasibility_audit(
        table_region_report=table_path,
        mineru_source_alignment_report=source_path,
    )

    paths = write_table_cell_provenance_feasibility_audit_reports(payload, tmp_path / "reports")

    assert set(paths) == {"audit", "summary", "markdown"}
    audit = json.loads(Path(paths["audit"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(audit, TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID, strict=True).ok
    assert "not table-cell citation-grade evidence" in markdown
