from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.table_cell_bbox_source_span_authority_design import (
    TABLE_CELL_BBOX_SOURCE_SPAN_AUTHORITY_DESIGN_SCHEMA_ID,
    build_table_cell_bbox_source_span_authority_design,
    write_table_cell_bbox_source_span_authority_design_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _review_pack(*, wrong_schema: bool = False, unsafe: bool = False, status: str = "review_pack_ready") -> dict:
    return {
        "schema": "example.wrong.review" if wrong_schema else "knowledge-hub.paper.table-cell-provenance-review-pack.v1",
        "status": status,
        "counts": {
            "reviewCardRows": 2,
            "readyForCellProvenanceReviewRows": 1,
            "heldOutRows": 1,
            "tableCellEvidenceVerifiedRows": 1 if unsafe else 0,
            "tableCellCitationGradeRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "reviewPackReady": status == "review_pack_ready",
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "reviewOnly": True,
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
        "reviewCards": [
            {
                "review_card_id": "table-cell-provenance-review:0001",
                "source_table_region_candidate_id": "tableregion:paper-1:0001",
                "paper_id": "paper-1",
                "candidate_type": "table_cell_provenance_review_card",
                "source_parser": "mineru+pymupdf_alignment",
                "table_label": "Table 1",
                "candidate_text": "Table 1: Model results.",
                "caption_text": "Model results.",
                "caption_original_pdf_offset_recovered": True,
                "original_pdf_span": {
                    "originalPdfCharsStart": 10,
                    "originalPdfCharsEnd": 33,
                    "page": 4,
                    "sourceContentHash": "source-hash",
                    "matchMethod": "exact",
                    "matchConfidence": 1.0,
                },
                "table_structure_available": True,
                "row_column_text_available": True,
                "table_row_count": 2,
                "table_cell_count": 4,
                "non_empty_table_cell_count": 4,
                "cell_bbox_count": 0,
                "cell_source_span_count": 0,
                "cell_source_hash_count": 0,
                "review_status": "ready_for_cell_provenance_review",
                "strict_blockers": ["table_cell_provenance_review_pack_only"],
                "strict_eligible": False,
                "citation_grade": False,
                "runtime_evidence": False,
            },
            {
                "review_card_id": "table-cell-provenance-review:0002",
                "source_table_region_candidate_id": "tableregion:paper-1:0002",
                "paper_id": "paper-1",
                "candidate_type": "table_cell_provenance_review_card",
                "source_parser": "mineru+pymupdf_alignment",
                "table_label": "Table 2",
                "candidate_text": "Table 2: Missing caption.",
                "caption_text": "Missing caption.",
                "caption_original_pdf_offset_recovered": False,
                "original_pdf_span": {
                    "originalPdfCharsStart": None,
                    "originalPdfCharsEnd": None,
                    "page": None,
                    "sourceContentHash": "source-hash",
                    "matchMethod": "",
                    "matchConfidence": 0.0,
                },
                "table_structure_available": True,
                "row_column_text_available": True,
                "table_row_count": 1,
                "table_cell_count": 2,
                "non_empty_table_cell_count": 2,
                "cell_bbox_count": 0,
                "cell_source_span_count": 0,
                "cell_source_hash_count": 0,
                "review_status": "held_out_caption_source_offset_missing",
                "strict_blockers": ["held_out_caption_source_offset_missing"],
                "strict_eligible": False,
                "citation_grade": False,
                "runtime_evidence": False,
            },
        ],
    }


def _report_path(root: Path, payload: dict | None = None) -> Path:
    return _write(root, "table-cell-provenance-review-cards.json", payload or _review_pack())


def test_table_cell_authority_design_lists_options_and_validates_schema(tmp_path: Path) -> None:
    payload = build_table_cell_bbox_source_span_authority_design(
        table_cell_provenance_review_pack=_report_path(tmp_path)
    )

    assert payload["schema"] == TABLE_CELL_BBOX_SOURCE_SPAN_AUTHORITY_DESIGN_SCHEMA_ID
    assert validate_payload(payload, TABLE_CELL_BBOX_SOURCE_SPAN_AUTHORITY_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "design_ready"
    assert payload["counts"]["authorityDesignRows"] == 2
    assert payload["counts"]["readyForCellAuthorityDesignRows"] == 1
    assert payload["counts"]["heldOutRows"] == 1
    assert payload["counts"]["tableCellEvidenceReadyRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert [item["option_id"] for item in payload["authorityOptions"]] == [
        "keep_table_region_caption_only",
        "recover_per_cell_bbox_source_spans",
        "explicitly_authorize_generated_markdown_cells",
    ]


def test_authority_design_rows_keep_table_cells_non_strict(tmp_path: Path) -> None:
    payload = build_table_cell_bbox_source_span_authority_design(
        table_cell_provenance_review_pack=_report_path(tmp_path)
    )

    row = next(item for item in payload["designRows"] if item["table_label"] == "Table 1")
    assert row["authority_design_status"] == "ready_for_cell_bbox_source_span_authority_design"
    assert row["recommended_authority_path"] == "require_per_cell_bbox_source_span_and_hash_before_strict_table_evidence"
    assert "per_cell_bbox_coordinates" in row["required_before_strict_table_cell_use"]
    assert "per_cell_original_pdf_chars_start_end" in row["required_before_strict_table_cell_use"]
    assert row["authority_decision_made"] is False
    assert row["table_cell_evidence_ready"] is False
    assert row["strict_promotion_ready"] is False
    assert row["strict_eligible"] is False
    assert row["citation_grade"] is False
    assert row["runtime_evidence"] is False
    assert "table_cell_bbox_source_span_authority_design_only" in row["strict_blockers"]
    assert payload["policy"]["tableCellEvidenceCreated"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["databaseMutation"] is False


def test_held_out_caption_offset_remains_blocked_for_authority_design(tmp_path: Path) -> None:
    payload = build_table_cell_bbox_source_span_authority_design(
        table_cell_provenance_review_pack=_report_path(tmp_path)
    )

    row = next(item for item in payload["designRows"] if item["table_label"] == "Table 2")
    assert row["authority_design_status"] == "blocked_caption_source_offset_missing"
    assert row["recommended_authority_path"] == "hold_until_caption_source_offset_and_table_structure_are_available"
    assert row["strict_eligible"] is False


def test_generated_markdown_cell_authority_option_requires_policy_decision(tmp_path: Path) -> None:
    payload = build_table_cell_bbox_source_span_authority_design(
        table_cell_provenance_review_pack=_report_path(tmp_path)
    )

    option = next(item for item in payload["authorityOptions"] if item["option_id"] == "explicitly_authorize_generated_markdown_cells")
    assert option["recommendation"] == "not_recommended_without_durable_policy_decision"
    assert "ADR_or_project_state_policy_record" in option["required_before_use"]
    assert option["authority_decision_made"] is False
    assert option["strict_promotion_ready"] is False
    assert option["runtime_promotion_allowed"] is False
    assert "runtime_answer_citation" in option["blocked_actions"]


def test_authority_design_blocks_wrong_or_unsafe_review_pack(tmp_path: Path) -> None:
    payload = build_table_cell_bbox_source_span_authority_design(
        table_cell_provenance_review_pack=_report_path(tmp_path, _review_pack(wrong_schema=True, unsafe=True))
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert payload["gate"]["schemaViolations"] == [
        "table_cell_provenance_review_pack_schema_mismatch"
    ]
    assert "tableCellEvidenceVerifiedRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert validate_payload(payload, TABLE_CELL_BBOX_SOURCE_SPAN_AUTHORITY_DESIGN_SCHEMA_ID, strict=True).ok


def test_authority_design_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    payload = build_table_cell_bbox_source_span_authority_design(
        table_cell_provenance_review_pack=_report_path(tmp_path / "input")
    )

    paths = write_table_cell_bbox_source_span_authority_design_reports(payload, tmp_path / "reports")

    assert set(paths) == {"design", "summary", "markdown"}
    design = json.loads(Path(paths["design"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(design, TABLE_CELL_BBOX_SOURCE_SPAN_AUTHORITY_DESIGN_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, TABLE_CELL_BBOX_SOURCE_SPAN_AUTHORITY_DESIGN_SCHEMA_ID, strict=True).ok
    assert "does not extract per-cell bboxes" in markdown
