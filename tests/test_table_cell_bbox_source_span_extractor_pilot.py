from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.table_cell_bbox_source_span_extractor_pilot import (
    TABLE_CELL_BBOX_SOURCE_SPAN_EXTRACTOR_PILOT_SCHEMA_ID,
    build_table_cell_bbox_source_span_extractor_pilot,
    write_table_cell_bbox_source_span_extractor_pilot_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _design(*, wrong_schema: bool = False, unsafe: bool = False) -> dict:
    return {
        "schema": (
            "example.wrong.design"
            if wrong_schema
            else "knowledge-hub.paper.table-cell-bbox-source-span-authority-design.v1"
        ),
        "status": "design_ready",
        "counts": {
            "tableCellEvidenceReadyRows": 1 if unsafe else 0,
            "tableCellCitationGradeRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "authorityDecisionMade": False,
            "tableCellEvidenceReady": False,
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "authorityDecisionMade": False,
            "tableCellEvidenceCreated": False,
            "tableCellCitationGradeEvidenceCreated": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "designRows": [
            {
                "design_id": "table-cell-authority-design:0001",
                "source_review_card_id": "table-cell-provenance-review:0001",
                "source_table_region_candidate_id": "tableregion:paper-1:0001",
                "paper_id": "paper-1",
                "table_label": "Table 1",
                "candidate_text": "Table 1: Model results.",
                "caption_original_pdf_offset_recovered": True,
                "original_pdf_span": {
                    "page": 3,
                    "sourceContentHash": "source-hash",
                },
                "authority_design_status": "ready_for_cell_bbox_source_span_authority_design",
                "strict_blockers": ["table_cell_bbox_source_span_authority_design_only"],
            },
            {
                "design_id": "table-cell-authority-design:0002",
                "source_review_card_id": "table-cell-provenance-review:0002",
                "source_table_region_candidate_id": "tableregion:paper-1:0002",
                "paper_id": "paper-1",
                "table_label": "Table 2",
                "candidate_text": "Table 2: Missing caption.",
                "caption_original_pdf_offset_recovered": False,
                "original_pdf_span": {"page": None, "sourceContentHash": "source-hash"},
                "authority_design_status": "blocked_caption_source_offset_missing",
                "strict_blockers": ["held_out_caption_source_offset_missing"],
            },
        ],
    }


def _source_report(*, wrong_schema: bool = False) -> dict:
    return {
        "schema": (
            "example.wrong.source"
            if wrong_schema
            else "knowledge-hub.paper.table-region-pdf-offset-feasibility.v1"
        ),
        "status": "feasibility_complete",
        "feasibilityRows": [
            {
                "source_table_region_candidate_id": "tableregion:paper-1:0001",
                "paper_id": "paper-1",
                "source_pdf_path": "/tmp/paper-1.pdf",
                "sourceContentHash": "source-hash",
            },
            {
                "source_table_region_candidate_id": "tableregion:paper-1:0002",
                "paper_id": "paper-1",
                "source_pdf_path": "/tmp/paper-1.pdf",
                "sourceContentHash": "source-hash",
            },
        ],
    }


def _paths(root: Path, *, wrong_schema: bool = False, unsafe: bool = False) -> tuple[Path, Path]:
    design_path = _write(root, "table-cell-authority-design.json", _design(wrong_schema=wrong_schema, unsafe=unsafe))
    source_path = _write(root, "table-region-pdf-offset-feasibility.json", _source_report(wrong_schema=wrong_schema))
    return design_path, source_path


def _table_probe(_source_pdf: str | Path, page_number: int) -> dict[str, Any]:
    assert page_number == 3
    return {
        "status": "ok",
        "failureReason": "",
        "tables": [
            {
                "table_index": 1,
                "bbox": [10.0, 20.0, 200.0, 240.0],
                "row_count": 2,
                "column_count": 2,
                "cell_bbox_count": 4,
                "cell_bboxes_sample": [[10.0, 20.0, 50.0, 40.0]],
                "extracted_rows": [["Model", "Score"], ["A", "1.0"]],
            }
        ],
    }


def _no_table_probe(_source_pdf: str | Path, _page_number: int) -> dict[str, Any]:
    return {"status": "ok", "failureReason": "", "tables": []}


def _page_text_loader(_source_pdf: str | Path) -> list[dict[str, Any]]:
    return [{"page": 3, "text": "Model Score\nA 1.0\n"}]


def test_extractor_pilot_reports_cell_bbox_candidates_but_no_evidence(tmp_path: Path) -> None:
    design_path, source_path = _paths(tmp_path)

    payload = build_table_cell_bbox_source_span_extractor_pilot(
        table_cell_authority_design_report=design_path,
        table_region_pdf_offset_feasibility_report=source_path,
        table_probe_loader=_table_probe,
        page_text_loader=_page_text_loader,
    )

    assert payload["schema"] == TABLE_CELL_BBOX_SOURCE_SPAN_EXTRACTOR_PILOT_SCHEMA_ID
    assert validate_payload(payload, TABLE_CELL_BBOX_SOURCE_SPAN_EXTRACTOR_PILOT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "pilot_complete"
    assert payload["counts"]["pilotRows"] == 2
    assert payload["counts"]["probeAttemptedRows"] == 1
    assert payload["counts"]["tableDetectedRows"] == 1
    assert payload["counts"]["cellBboxCandidateRows"] == 1
    assert payload["counts"]["selectedTableCellBboxCandidates"] == 4
    assert payload["counts"]["cellSourceSpanCreatedRows"] == 0
    assert payload["gate"]["cellBboxCandidatesObserved"] is True
    assert payload["gate"]["cellSourceSpansCreated"] is False
    assert payload["gate"]["strictEvidenceReady"] is False


def test_probe_row_stays_diagnostic_only_even_with_text_matches(tmp_path: Path) -> None:
    design_path, source_path = _paths(tmp_path)

    payload = build_table_cell_bbox_source_span_extractor_pilot(
        table_cell_authority_design_report=design_path,
        table_region_pdf_offset_feasibility_report=source_path,
        table_probe_loader=_table_probe,
        page_text_loader=_page_text_loader,
    )
    row = next(item for item in payload["pilotRows"] if item["table_label"] == "Table 1")

    assert row["probe_status"] == "cell_bbox_candidates_detected_non_strict"
    assert row["table_cell_bbox_candidates_found"] is True
    assert row["diagnostic_unique_cell_text_matches"] > 0
    assert row["diagnostic_cell_text_matches_are_strict"] is False
    assert row["cell_bbox_text_pairing_verified"] is False
    assert row["cell_source_spans_created"] == 0
    assert row["table_cell_evidence_created"] is False
    assert row["strict_eligible"] is False
    assert "cell_source_spans_not_created" in row["strict_blockers"]


def test_no_tables_detected_is_not_green(tmp_path: Path) -> None:
    design_path, source_path = _paths(tmp_path)

    payload = build_table_cell_bbox_source_span_extractor_pilot(
        table_cell_authority_design_report=design_path,
        table_region_pdf_offset_feasibility_report=source_path,
        table_probe_loader=_no_table_probe,
        page_text_loader=_page_text_loader,
    )
    row = next(item for item in payload["pilotRows"] if item["table_label"] == "Table 1")

    assert row["probe_status"] == "no_tables_detected"
    assert row["table_cell_bbox_candidates_found"] is False
    assert payload["counts"]["noTableDetectedRows"] == 1
    assert payload["counts"]["cellBboxCandidateRows"] == 0


def test_held_out_authority_design_row_is_not_probed(tmp_path: Path) -> None:
    design_path, source_path = _paths(tmp_path)

    payload = build_table_cell_bbox_source_span_extractor_pilot(
        table_cell_authority_design_report=design_path,
        table_region_pdf_offset_feasibility_report=source_path,
        table_probe_loader=_table_probe,
        page_text_loader=_page_text_loader,
    )
    row = next(item for item in payload["pilotRows"] if item["table_label"] == "Table 2")

    assert row["probe_status"] == "held_out_authority_design_not_ready"
    assert row["probe_attempted"] is False
    assert row["strict_eligible"] is False


def test_max_ready_rows_keeps_extra_ready_rows_held_out(tmp_path: Path) -> None:
    design_path, source_path = _paths(tmp_path)
    design = json.loads(design_path.read_text(encoding="utf-8"))
    second_ready = dict(design["designRows"][0])
    second_ready["design_id"] = "table-cell-authority-design:0003"
    second_ready["source_table_region_candidate_id"] = "tableregion:paper-1:0003"
    second_ready["table_label"] = "Table 3"
    design["designRows"].append(second_ready)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["feasibilityRows"].append(
        {
            "source_table_region_candidate_id": "tableregion:paper-1:0003",
            "paper_id": "paper-1",
            "source_pdf_path": "/tmp/paper-1.pdf",
            "sourceContentHash": "source-hash",
        }
    )
    design_path.write_text(json.dumps(design), encoding="utf-8")
    source_path.write_text(json.dumps(source), encoding="utf-8")

    payload = build_table_cell_bbox_source_span_extractor_pilot(
        table_cell_authority_design_report=design_path,
        table_region_pdf_offset_feasibility_report=source_path,
        max_ready_rows=1,
        table_probe_loader=_table_probe,
        page_text_loader=_page_text_loader,
    )

    assert payload["counts"]["probeAttemptedRows"] == 1
    assert payload["counts"]["byProbeStatus"]["held_out_max_ready_rows_exceeded"] == 1


def test_wrong_or_unsafe_inputs_block_pilot(tmp_path: Path) -> None:
    design_path, source_path = _paths(tmp_path, wrong_schema=True, unsafe=True)

    payload = build_table_cell_bbox_source_span_extractor_pilot(
        table_cell_authority_design_report=design_path,
        table_region_pdf_offset_feasibility_report=source_path,
        table_probe_loader=_table_probe,
        page_text_loader=_page_text_loader,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert set(payload["gate"]["schemaViolations"]) == {
        "table_cell_bbox_source_span_authority_design_schema_mismatch",
        "table_region_pdf_offset_feasibility_schema_mismatch",
    }
    assert "tableCellEvidenceReadyRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert validate_payload(payload, TABLE_CELL_BBOX_SOURCE_SPAN_EXTRACTOR_PILOT_SCHEMA_ID, strict=True).ok


def test_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    design_path, source_path = _paths(tmp_path / "input")
    payload = build_table_cell_bbox_source_span_extractor_pilot(
        table_cell_authority_design_report=design_path,
        table_region_pdf_offset_feasibility_report=source_path,
        table_probe_loader=_table_probe,
        page_text_loader=_page_text_loader,
    )

    paths = write_table_cell_bbox_source_span_extractor_pilot_reports(payload, tmp_path / "reports")

    assert set(paths) == {"pilot", "summary", "markdown"}
    pilot = json.loads(Path(paths["pilot"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(pilot, TABLE_CELL_BBOX_SOURCE_SPAN_EXTRACTOR_PILOT_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, TABLE_CELL_BBOX_SOURCE_SPAN_EXTRACTOR_PILOT_SCHEMA_ID, strict=True).ok
    assert "not source-aligned table-cell evidence" in markdown
