from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.table_cell_provenance_review_pack import (
    TABLE_CELL_PROVENANCE_REVIEW_PACK_SCHEMA_ID,
    build_table_cell_provenance_review_pack,
    write_table_cell_provenance_review_pack_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _cell_report(root: Path, *, wrong_schema: bool = False, with_cell_provenance: bool = False) -> Path:
    return _write(
        root,
        "table-cell-provenance-feasibility-audit.json",
        {
            "schema": (
                "example.wrong.table-cell"
                if wrong_schema
                else "knowledge-hub.paper.table-cell-provenance-feasibility-audit.v1"
            ),
            "status": "ok",
            "rows": [
                {
                    "audit_id": "table-cell-provenance-feasibility:0001",
                    "table_region_candidate_id": "tableregion:paper-1:0001",
                    "source_candidate_id": "paper-1:table:0001",
                    "paper_id": "paper-1",
                    "candidate_type": "table_cell_provenance_feasibility_candidate",
                    "source_parser": "mineru+pymupdf_alignment",
                    "candidate_text": "Table 1: Model results.",
                    "table_label": "Table 1",
                    "caption_text": "Model results.",
                    "caption_alignment_status": "aligned",
                    "caption_alignment_method": "exact",
                    "caption_chars_start": 100,
                    "caption_chars_end": 123,
                    "caption_page": 4,
                    "sourceContentHash": "source-hash",
                    "caption_source_span_available": True,
                    "table_region_bbox_available": True,
                    "layout_element_count": 3,
                    "normalizer_candidate_id": "paper-1:table:0001",
                    "normalizer_report_path": "/tmp/mineru-normalizer-candidates.json",
                    "normalizer_match": True,
                    "table_structure_available": True,
                    "row_column_text_available": True,
                    "table_row_count": 2,
                    "table_max_column_count": 2,
                    "table_cell_count": 4,
                    "non_empty_table_cell_count": 4,
                    "header_like_cell_count": 0,
                    "rowspan_cell_count": 0,
                    "colspan_cell_count": 0,
                    "cell_bbox_count": 1 if with_cell_provenance else 0,
                    "cell_source_span_count": 1 if with_cell_provenance else 0,
                    "cell_source_hash_count": 1 if with_cell_provenance else 0,
                    "sample_cells": [{"text": "mAP", "has_bbox": with_cell_provenance}],
                    "cell_bbox_available": with_cell_provenance,
                    "cell_source_span_available": with_cell_provenance,
                    "cell_source_hash_backed": with_cell_provenance,
                    "table_cell_citation_grade": False,
                    "feasibility_status": (
                        "cell_text_structure_candidate_non_strict"
                        if with_cell_provenance
                        else "table_structure_candidate_no_cell_provenance"
                    ),
                    "strict_eligible": False,
                    "citation_grade": False,
                    "runtime_evidence": False,
                    "strict_blockers": ["table_cell_provenance_feasibility_audit_only"],
                    "non_strict_reason": ["table_cell_provenance_feasibility_rows_are_not_evidence"],
                },
                {
                    "audit_id": "table-cell-provenance-feasibility:0002",
                    "table_region_candidate_id": "tableregion:paper-1:0002",
                    "source_candidate_id": "paper-1:table:0002",
                    "paper_id": "paper-1",
                    "candidate_type": "table_cell_provenance_feasibility_candidate",
                    "source_parser": "mineru+pymupdf_alignment",
                    "candidate_text": "Table 2: Caption missing.",
                    "table_label": "Table 2",
                    "caption_text": "Caption missing.",
                    "caption_alignment_status": "failed",
                    "caption_alignment_method": "none",
                    "caption_chars_start": None,
                    "caption_chars_end": None,
                    "caption_page": None,
                    "sourceContentHash": "source-hash",
                    "caption_source_span_available": False,
                    "table_region_bbox_available": True,
                    "layout_element_count": 2,
                    "normalizer_candidate_id": "paper-1:table:0002",
                    "normalizer_match": True,
                    "table_structure_available": True,
                    "row_column_text_available": True,
                    "table_row_count": 1,
                    "table_max_column_count": 2,
                    "table_cell_count": 2,
                    "non_empty_table_cell_count": 2,
                    "header_like_cell_count": 0,
                    "rowspan_cell_count": 0,
                    "colspan_cell_count": 0,
                    "cell_bbox_count": 0,
                    "cell_source_span_count": 0,
                    "cell_source_hash_count": 0,
                    "sample_cells": [{"text": "blocked"}],
                    "cell_bbox_available": False,
                    "cell_source_span_available": False,
                    "cell_source_hash_backed": False,
                    "table_cell_citation_grade": False,
                    "feasibility_status": "table_structure_candidate_caption_alignment_blocked",
                    "strict_eligible": False,
                    "citation_grade": False,
                    "runtime_evidence": False,
                    "strict_blockers": ["table_caption_source_span_incomplete"],
                    "non_strict_reason": ["table_cell_provenance_feasibility_rows_are_not_evidence"],
                },
            ],
        },
    )


def _pdf_report(root: Path, *, wrong_schema: bool = False) -> Path:
    return _write(
        root,
        "table-region-pdf-offset-feasibility.json",
        {
            "schema": (
                "example.wrong.table-region"
                if wrong_schema
                else "knowledge-hub.paper.table-region-pdf-offset-feasibility.v1"
            ),
            "status": "feasibility_complete",
            "feasibilityRows": [
                {
                    "feasibility_row_id": "tableregion-pdf-offset:paper-1:tableregion:paper-1:0001",
                    "source_table_region_candidate_id": "tableregion:paper-1:0001",
                    "paper_id": "paper-1",
                    "candidate_text": "Table 1: Model results.",
                    "caption_text": "Model results.",
                    "table_label": "Table 1",
                    "original_pdf_offset_recovered": True,
                    "original_pdf_span": {
                        "originalPdfCharsStart": 200,
                        "originalPdfCharsEnd": 223,
                        "page": 4,
                        "sourceContentHash": "source-hash",
                        "matchMethod": "exact",
                        "matchConfidence": 1.0,
                    },
                    "page_agrees_with_canonical": True,
                    "source_hash_agrees_with_canonical": True,
                    "strict_blockers": [
                        "table_cell_row_column_bbox_provenance_missing",
                        "runtime_promotion_disabled_for_tranche",
                    ],
                },
                {
                    "feasibility_row_id": "tableregion-pdf-offset:paper-1:tableregion:paper-1:0002",
                    "source_table_region_candidate_id": "tableregion:paper-1:0002",
                    "paper_id": "paper-1",
                    "candidate_text": "Table 2: Caption missing.",
                    "caption_text": "Caption missing.",
                    "table_label": "Table 2",
                    "original_pdf_offset_recovered": False,
                    "original_pdf_span": {
                        "originalPdfCharsStart": None,
                        "originalPdfCharsEnd": None,
                        "page": None,
                        "sourceContentHash": "source-hash",
                        "matchMethod": "",
                        "matchConfidence": 0.0,
                    },
                    "page_agrees_with_canonical": False,
                    "source_hash_agrees_with_canonical": False,
                    "strict_blockers": [
                        "original_pdf_offset_not_recovered",
                        "table_cell_provenance_missing",
                    ],
                },
            ],
        },
    )


def test_table_cell_review_pack_emits_ready_and_held_out_cards_with_valid_schema(tmp_path: Path) -> None:
    payload = build_table_cell_provenance_review_pack(
        table_cell_provenance_feasibility_report=_cell_report(tmp_path),
        table_region_pdf_offset_feasibility_report=_pdf_report(tmp_path),
    )

    assert payload["schema"] == TABLE_CELL_PROVENANCE_REVIEW_PACK_SCHEMA_ID
    assert validate_payload(payload, TABLE_CELL_PROVENANCE_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "review_pack_ready"
    assert payload["counts"]["reviewCardRows"] == 2
    assert payload["counts"]["readyForCellProvenanceReviewRows"] == 1
    assert payload["counts"]["heldOutRows"] == 1
    assert payload["counts"]["tableCellCitationGradeRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["decision"] == "ready_for_table_cell_provenance_human_review"


def test_ready_card_keeps_table_cells_non_strict_even_with_caption_offset(tmp_path: Path) -> None:
    payload = build_table_cell_provenance_review_pack(
        table_cell_provenance_feasibility_report=_cell_report(tmp_path),
        table_region_pdf_offset_feasibility_report=_pdf_report(tmp_path),
    )

    card = next(item for item in payload["reviewCards"] if item["table_label"] == "Table 1")
    assert card["review_status"] == "ready_for_cell_provenance_review"
    assert card["caption_original_pdf_offset_recovered"] is True
    assert card["original_pdf_span"]["page"] == 4
    assert card["table_structure_available"] is True
    assert card["row_column_text_available"] is True
    assert card["cell_bbox_available"] is False
    assert card["cell_source_span_available"] is False
    assert card["cell_source_hash_backed"] is False
    assert card["table_cell_citation_grade"] is False
    assert card["table_cell_evidence_verified"] is False
    assert card["strict_eligible"] is False
    assert card["citation_grade"] is False
    assert "caption_offset_does_not_create_table_cell_evidence" in card["strict_blockers"]
    assert "table_cell_provenance_review_pack_only" in card["strict_blockers"]
    assert payload["policy"]["tableCellEvidenceCreated"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["parserRoutingChanged"] is False


def test_missing_caption_offset_is_held_out_not_green(tmp_path: Path) -> None:
    payload = build_table_cell_provenance_review_pack(
        table_cell_provenance_feasibility_report=_cell_report(tmp_path),
        table_region_pdf_offset_feasibility_report=_pdf_report(tmp_path),
    )

    card = next(item for item in payload["reviewCards"] if item["table_label"] == "Table 2")
    assert card["review_status"] == "held_out_caption_source_offset_missing"
    assert card["caption_original_pdf_offset_recovered"] is False
    assert card["strict_eligible"] is False
    assert "original_pdf_offset_not_recovered" in card["strict_blockers"]
    assert "held_out_caption_source_offset_missing" in card["strict_blockers"]


def test_cell_like_provenance_still_requires_review_not_runtime_promotion(tmp_path: Path) -> None:
    payload = build_table_cell_provenance_review_pack(
        table_cell_provenance_feasibility_report=_cell_report(tmp_path, with_cell_provenance=True),
        table_region_pdf_offset_feasibility_report=_pdf_report(tmp_path),
    )

    card = next(item for item in payload["reviewCards"] if item["table_label"] == "Table 1")
    assert card["cell_bbox_available"] is True
    assert card["cell_source_span_available"] is True
    assert card["cell_source_hash_backed"] is True
    assert card["table_cell_citation_grade"] is False
    assert card["table_cell_evidence_verified"] is False
    assert card["strict_eligible"] is False
    assert card["runtime_evidence"] is False
    assert card["evidence_tier"] == "table_cell_provenance_review_card_only"


def test_wrong_input_schema_blocks_review_pack(tmp_path: Path) -> None:
    payload = build_table_cell_provenance_review_pack(
        table_cell_provenance_feasibility_report=_cell_report(tmp_path, wrong_schema=True),
        table_region_pdf_offset_feasibility_report=_pdf_report(tmp_path, wrong_schema=True),
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert set(payload["gate"]["schemaViolations"]) == {
        "table_cell_provenance_feasibility_schema_mismatch",
        "table_region_pdf_offset_feasibility_schema_mismatch",
    }


def test_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    payload = build_table_cell_provenance_review_pack(
        table_cell_provenance_feasibility_report=_cell_report(tmp_path / "input"),
        table_region_pdf_offset_feasibility_report=_pdf_report(tmp_path / "input"),
    )

    paths = write_table_cell_provenance_review_pack_reports(payload, tmp_path / "reports")

    assert set(paths) == {"cards", "summary", "markdown"}
    cards = json.loads(Path(paths["cards"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(cards, TABLE_CELL_PROVENANCE_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, TABLE_CELL_PROVENANCE_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert "Recovered table-caption offsets" in markdown
