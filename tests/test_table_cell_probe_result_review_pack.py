from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.table_cell_probe_result_review_pack import (
    TABLE_CELL_PROBE_RESULT_REVIEW_PACK_SCHEMA_ID,
    build_table_cell_probe_result_review_pack,
    write_table_cell_probe_result_review_pack_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pilot(*, wrong_schema: bool = False, unsafe: bool = False) -> dict:
    return {
        "schema": (
            "example.wrong.extractor"
            if wrong_schema
            else "knowledge-hub.paper.table-cell-bbox-source-span-extractor-pilot.v1"
        ),
        "status": "pilot_complete",
        "counts": {
            "cellBboxTextPairingVerifiedRows": 1 if unsafe else 0,
            "cellSourceSpanCreatedRows": 0,
            "cellSourceHashLinkedRows": 0,
            "tableCellEvidenceCreatedRows": 0,
            "tableCellCitationGradeRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "cellBboxTextPairingVerified": False,
            "cellSourceSpansCreated": False,
            "cellSourceHashLinked": False,
            "tableCellEvidenceReady": False,
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
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
        "pilotRows": [
            {
                "pilot_row_id": "table-cell-extractor-pilot:0001",
                "source_authority_design_id": "table-cell-authority-design:0001",
                "source_review_card_id": "table-cell-provenance-review:0001",
                "source_table_region_candidate_id": "tableregion:paper-1:0001",
                "paper_id": "paper-1",
                "source_parser": "pymupdf_table_probe",
                "table_label": "Table 1",
                "candidate_text": "Table 1: Results.",
                "page": 3,
                "sourceContentHash": "source-hash",
                "probe_attempted": True,
                "probe_status": "cell_bbox_candidates_detected_non_strict",
                "probe_failure_reason": "",
                "detected_table_count": 1,
                "selected_table_index": 1,
                "selected_table_bbox": [10.0, 20.0, 200.0, 240.0],
                "selected_table_row_count": 2,
                "selected_table_column_count": 2,
                "selected_table_cell_bbox_count": 4,
                "selected_table_cell_text_count": 4,
                "diagnostic_unique_cell_text_matches": 2,
                "diagnostic_ambiguous_cell_text_matches": 1,
                "diagnostic_no_match_cell_texts": 1,
                "sample_cell_bboxes": [[10.0, 20.0, 50.0, 40.0]],
                "sample_cell_text_matches": [{"cell_text": "Score", "strict_source_span": False}],
                "table_cell_bbox_candidates_found": True,
                "cell_bbox_text_pairing_verified": False,
                "cell_source_spans_created": 0,
                "cell_source_hash_linkages_created": 0,
                "table_cell_evidence_created": False,
                "table_cell_citation_grade": False,
                "strict_eligible": False,
                "citation_grade": False,
                "runtime_evidence": False,
                "strict_blockers": ["table_cell_extractor_pilot_only"],
            },
            {
                "pilot_row_id": "table-cell-extractor-pilot:0002",
                "source_table_region_candidate_id": "tableregion:paper-1:0002",
                "paper_id": "paper-1",
                "table_label": "Table 2",
                "candidate_text": "Table 2: Missing table detection.",
                "page": 4,
                "sourceContentHash": "source-hash",
                "probe_attempted": True,
                "probe_status": "no_tables_detected",
                "probe_failure_reason": "",
                "detected_table_count": 0,
                "selected_table_cell_bbox_count": 0,
                "selected_table_cell_text_count": 0,
                "diagnostic_unique_cell_text_matches": 0,
                "diagnostic_ambiguous_cell_text_matches": 0,
                "diagnostic_no_match_cell_texts": 0,
                "table_cell_bbox_candidates_found": False,
                "strict_blockers": ["no_tables_detected"],
            },
            {
                "pilot_row_id": "table-cell-extractor-pilot:0003",
                "source_table_region_candidate_id": "tableregion:paper-2:0001",
                "paper_id": "paper-2",
                "table_label": "Table 1",
                "candidate_text": "Table 1: Held out.",
                "page": None,
                "sourceContentHash": "source-hash-2",
                "probe_attempted": False,
                "probe_status": "held_out_authority_design_not_ready",
                "probe_failure_reason": "",
                "detected_table_count": 0,
                "selected_table_cell_bbox_count": 0,
                "selected_table_cell_text_count": 0,
                "diagnostic_unique_cell_text_matches": 0,
                "diagnostic_ambiguous_cell_text_matches": 0,
                "diagnostic_no_match_cell_texts": 0,
                "table_cell_bbox_candidates_found": False,
                "strict_blockers": ["held_out_caption_source_offset_missing"],
            },
        ],
    }


def _path(root: Path, payload: dict | None = None) -> Path:
    return _write(root, "table-cell-bbox-source-span-extractor-pilot.json", payload or _pilot())


def test_probe_result_review_pack_classifies_probe_outcomes(tmp_path: Path) -> None:
    payload = build_table_cell_probe_result_review_pack(table_cell_extractor_pilot_report=_path(tmp_path))

    assert payload["schema"] == TABLE_CELL_PROBE_RESULT_REVIEW_PACK_SCHEMA_ID
    assert validate_payload(payload, TABLE_CELL_PROBE_RESULT_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "review_pack_ready"
    assert payload["counts"]["reviewCardRows"] == 3
    assert payload["counts"]["pymupdfOverlayCandidateRows"] == 1
    assert payload["counts"]["alternativeExtractorRequiredRows"] == 1
    assert payload["counts"]["heldOutRows"] == 1
    assert payload["gate"]["pymupdfOverlayCandidateObserved"] is True
    assert payload["gate"]["alternativeExtractorRequired"] is True
    assert payload["gate"]["extractorChoiceMade"] is False


def test_overlay_candidate_remains_non_strict_and_not_evidence(tmp_path: Path) -> None:
    payload = build_table_cell_probe_result_review_pack(table_cell_extractor_pilot_report=_path(tmp_path))
    card = next(
        item
        for item in payload["reviewCards"]
        if item["review_status"] == "pymupdf_overlay_candidate_needs_pairing_review"
    )

    assert card["recommended_review_action"] == "review_pymupdf_cell_bbox_overlay_before_source_span_work"
    assert card["cell_bbox_candidates_observed"] is True
    assert card["cell_bbox_text_pairing_verified"] is False
    assert card["cell_source_spans_created"] == 0
    assert card["cell_source_hash_linkages_created"] == 0
    assert card["table_cell_evidence_created"] is False
    assert card["table_cell_citation_grade"] is False
    assert card["strict_eligible"] is False
    assert card["runtime_evidence"] is False
    assert "extractor_choice_not_made" in card["strict_blockers"]


def test_no_table_detected_routes_to_alternative_extractor_review(tmp_path: Path) -> None:
    payload = build_table_cell_probe_result_review_pack(table_cell_extractor_pilot_report=_path(tmp_path))
    card = next(item for item in payload["reviewCards"] if item["probe_status"] == "no_tables_detected")

    assert card["review_status"] == "requires_alternative_table_extractor"
    assert card["recommended_review_action"] == "pilot_alternative_table_extractor_for_this_row"
    assert card["cell_bbox_candidates_observed"] is False
    assert card["strict_eligible"] is False


def test_held_out_row_stays_held_out(tmp_path: Path) -> None:
    payload = build_table_cell_probe_result_review_pack(table_cell_extractor_pilot_report=_path(tmp_path))
    card = next(item for item in payload["reviewCards"] if item["paper_id"] == "paper-2")

    assert card["review_status"] == "held_out_authority_design_not_ready"
    assert card["recommended_review_action"] == "resolve_caption_source_offset_or_keep_held_out"
    assert card["probe_attempted"] is False
    assert card["strict_eligible"] is False


def test_wrong_or_unsafe_input_blocks_review_pack(tmp_path: Path) -> None:
    payload = build_table_cell_probe_result_review_pack(
        table_cell_extractor_pilot_report=_path(tmp_path, _pilot(wrong_schema=True, unsafe=True))
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert payload["gate"]["schemaViolations"] == [
        "table_cell_bbox_source_span_extractor_pilot_schema_mismatch"
    ]
    assert "cellBboxTextPairingVerifiedRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert validate_payload(payload, TABLE_CELL_PROBE_RESULT_REVIEW_PACK_SCHEMA_ID, strict=True).ok


def test_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    payload = build_table_cell_probe_result_review_pack(table_cell_extractor_pilot_report=_path(tmp_path / "input"))

    paths = write_table_cell_probe_result_review_pack_reports(payload, tmp_path / "reports")

    assert set(paths) == {"cards", "summary", "markdown"}
    cards = json.loads(Path(paths["cards"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(cards, TABLE_CELL_PROBE_RESULT_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, TABLE_CELL_PROBE_RESULT_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert "do not verify cell/text pairing" in markdown
