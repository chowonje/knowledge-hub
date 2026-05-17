from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.table_cell_pymupdf_overlay_review_pack import (
    TABLE_CELL_PYMUPDF_OVERLAY_REVIEW_PACK_SCHEMA_ID,
    build_table_cell_pymupdf_overlay_review_pack,
    write_table_cell_pymupdf_overlay_review_pack_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _review_pack(*, wrong_schema: bool = False, unsafe: bool = False, all_unique: bool = False) -> dict:
    unique = 4 if all_unique else 2
    ambiguous = 0 if all_unique else 1
    missing = 0 if all_unique else 1
    return {
        "schema": (
            "example.wrong.probe-review"
            if wrong_schema
            else "knowledge-hub.paper.table-cell-probe-result-review-pack.v1"
        ),
        "status": "review_pack_ready",
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
            "extractorChoiceMade": False,
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
            "extractorChoiceMade": False,
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
                "review_card_id": "table-cell-probe-result-review:0001",
                "source_table_region_candidate_id": "tableregion:paper-1:0001",
                "paper_id": "paper-1",
                "table_label": "Table 1",
                "candidate_text": "Table 1: Results.",
                "page": 3,
                "sourceContentHash": "source-hash",
                "review_status": "pymupdf_overlay_candidate_needs_pairing_review",
                "selected_table_bbox": [10.0, 20.0, 200.0, 240.0],
                "selected_table_row_count": 2,
                "selected_table_column_count": 2,
                "selected_table_cell_bbox_count": 4,
                "selected_table_cell_text_count": 4,
                "diagnostic_unique_cell_text_matches": unique,
                "diagnostic_ambiguous_cell_text_matches": ambiguous,
                "diagnostic_no_match_cell_texts": missing,
                "sample_cell_bboxes": [[10.0, 20.0, 50.0, 40.0]],
                "sample_cell_text_matches": [{"cell_text": "Score", "strict_source_span": False}],
                "cell_bbox_candidates_observed": True,
                "strict_blockers": ["table_cell_probe_result_review_pack_only"],
            },
            {
                "review_card_id": "table-cell-probe-result-review:0002",
                "source_table_region_candidate_id": "tableregion:paper-1:0002",
                "paper_id": "paper-1",
                "table_label": "Table 2",
                "candidate_text": "Table 2: Alternative extractor needed.",
                "page": 4,
                "sourceContentHash": "source-hash",
                "review_status": "requires_alternative_table_extractor",
                "selected_table_cell_bbox_count": 0,
                "selected_table_cell_text_count": 0,
                "diagnostic_unique_cell_text_matches": 0,
                "diagnostic_ambiguous_cell_text_matches": 0,
                "diagnostic_no_match_cell_texts": 0,
                "cell_bbox_candidates_observed": False,
                "strict_blockers": ["requires_alternative_table_extractor"],
            },
        ],
    }


def _path(root: Path, payload: dict | None = None) -> Path:
    return _write(root, "table-cell-probe-result-review-cards.json", payload or _review_pack())


def test_overlay_review_pack_emits_only_pymupdf_overlay_rows(tmp_path: Path) -> None:
    payload = build_table_cell_pymupdf_overlay_review_pack(
        table_cell_probe_result_review_pack=_path(tmp_path)
    )

    assert payload["schema"] == TABLE_CELL_PYMUPDF_OVERLAY_REVIEW_PACK_SCHEMA_ID
    assert validate_payload(payload, TABLE_CELL_PYMUPDF_OVERLAY_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "review_pack_ready"
    assert payload["counts"]["overlayReviewCardRows"] == 1
    assert payload["counts"]["manualOverlayPairingReviewRequiredRows"] == 1
    assert payload["counts"]["selectedTableCellBboxCandidates"] == 4
    assert payload["counts"]["diagnosticUniqueCellTextMatches"] == 2
    assert payload["gate"]["manualOverlayPairingReviewRequired"] is True


def test_overlay_review_card_remains_non_strict_without_pairing_verification(tmp_path: Path) -> None:
    payload = build_table_cell_pymupdf_overlay_review_pack(
        table_cell_probe_result_review_pack=_path(tmp_path)
    )
    card = payload["overlayReviewCards"][0]

    assert card["overlay_review_status"] == "manual_overlay_pairing_review_required"
    assert card["recommended_review_action"] == "manually_compare_cell_bboxes_to_page_text_before_any_extractor_choice"
    assert card["cell_bbox_candidates_observed"] is True
    assert card["diagnostic_unique_match_ratio"] == 0.5
    assert card["diagnostic_ambiguous_match_ratio"] == 0.25
    assert card["diagnostic_no_match_ratio"] == 0.25
    assert card["cell_bbox_text_pairing_verified"] is False
    assert card["visual_pairing_review_completed"] is False
    assert card["cell_source_spans_created"] == 0
    assert card["table_cell_evidence_created"] is False
    assert card["table_cell_citation_grade"] is False
    assert card["strict_eligible"] is False
    assert "visual_cell_bbox_text_pairing_not_verified" in card["strict_blockers"]


def test_all_unique_diagnostic_matches_still_do_not_create_evidence(tmp_path: Path) -> None:
    payload = build_table_cell_pymupdf_overlay_review_pack(
        table_cell_probe_result_review_pack=_path(tmp_path, _review_pack(all_unique=True))
    )
    card = payload["overlayReviewCards"][0]

    assert card["overlay_review_status"] == "overlay_candidate_ready_for_visual_pairing_review"
    assert card["diagnostic_unique_match_ratio"] == 1.0
    assert card["cell_bbox_text_pairing_verified"] is False
    assert card["cell_source_spans_created"] == 0
    assert card["strict_eligible"] is False
    assert payload["counts"]["visualPairingReviewReadyRows"] == 1
    assert payload["counts"]["cellBboxTextPairingVerifiedRows"] == 0


def test_wrong_or_unsafe_input_blocks_overlay_pack(tmp_path: Path) -> None:
    payload = build_table_cell_pymupdf_overlay_review_pack(
        table_cell_probe_result_review_pack=_path(tmp_path, _review_pack(wrong_schema=True, unsafe=True))
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert payload["gate"]["schemaViolations"] == [
        "table_cell_probe_result_review_pack_schema_mismatch"
    ]
    assert "cellBboxTextPairingVerifiedRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert validate_payload(payload, TABLE_CELL_PYMUPDF_OVERLAY_REVIEW_PACK_SCHEMA_ID, strict=True).ok


def test_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    payload = build_table_cell_pymupdf_overlay_review_pack(
        table_cell_probe_result_review_pack=_path(tmp_path / "input")
    )

    paths = write_table_cell_pymupdf_overlay_review_pack_reports(payload, tmp_path / "reports")

    assert set(paths) == {"cards", "summary", "markdown"}
    cards = json.loads(Path(paths["cards"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(cards, TABLE_CELL_PYMUPDF_OVERLAY_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, TABLE_CELL_PYMUPDF_OVERLAY_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert "do not verify cell/text pairing" in markdown
