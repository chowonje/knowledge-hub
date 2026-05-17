from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.table_cell_pymupdf_pairing_diagnostic import (
    TABLE_CELL_PYMUPDF_PAIRING_DIAGNOSTIC_SCHEMA_ID,
    build_table_cell_pymupdf_pairing_diagnostic,
    write_table_cell_pymupdf_pairing_diagnostic_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _overlay_pack(*, wrong_schema: bool = False, unsafe: bool = False) -> dict:
    return {
        "schema": (
            "example.wrong.overlay"
            if wrong_schema
            else "knowledge-hub.paper.table-cell-pymupdf-overlay-review-pack.v1"
        ),
        "status": "review_pack_ready",
        "counts": {
            "cellBboxTextPairingVerifiedRows": 1 if unsafe else 0,
            "visualPairingReviewCompletedRows": 0,
            "cellSourceSpanCreatedRows": 0,
            "cellSourceHashLinkedRows": 0,
            "tableCellEvidenceCreatedRows": 0,
            "tableCellCitationGradeRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "visualPairingReviewCompleted": False,
            "cellBboxTextPairingVerified": False,
            "cellSourceSpansCreated": False,
            "cellSourceHashLinked": False,
            "extractorChoiceMade": False,
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
        "overlayReviewCards": [
            {
                "overlay_review_card_id": "table-cell-pymupdf-overlay-review:0001",
                "source_table_region_candidate_id": "tableregion:paper-1:0001",
                "paper_id": "paper-1",
                "table_label": "Table 1",
                "candidate_text": "Table 1: Results.",
                "page": 3,
                "sourceContentHash": "source-hash",
                "overlay_review_status": "manual_overlay_pairing_review_required",
                "strict_blockers": ["table_cell_pymupdf_overlay_review_pack_only"],
            }
        ],
    }


def _pilot_report(*, wrong_schema: bool = False) -> dict:
    return {
        "schema": (
            "example.wrong.pilot"
            if wrong_schema
            else "knowledge-hub.paper.table-cell-bbox-source-span-extractor-pilot.v1"
        ),
        "status": "pilot_complete",
        "counts": {
            "cellBboxTextPairingVerifiedRows": 0,
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
                "source_table_region_candidate_id": "tableregion:paper-1:0001",
                "paper_id": "paper-1",
                "table_label": "Table 1",
                "source_pdf_path": "/tmp/paper-1.pdf",
                "page": 3,
                "selected_table_index": 1,
            }
        ],
    }


def _paths(root: Path, *, wrong_schema: bool = False, unsafe: bool = False) -> tuple[Path, Path]:
    overlay = _write(root, "table-cell-pymupdf-overlay-review-cards.json", _overlay_pack(wrong_schema=wrong_schema, unsafe=unsafe))
    pilot = _write(root, "table-cell-bbox-source-span-extractor-pilot.json", _pilot_report(wrong_schema=wrong_schema))
    return overlay, pilot


def _probe(_source_pdf: str | Path, page_number: int) -> dict[str, Any]:
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
                "cell_bboxes": [
                    [10.0, 20.0, 50.0, 40.0],
                    [50.0, 20.0, 90.0, 40.0],
                    [10.0, 40.0, 50.0, 60.0],
                    [50.0, 40.0, 90.0, 60.0],
                ],
                "extracted_rows": [["Model", "Score"], ["A", "1.0"]],
            }
        ],
    }


def _probe_no_table(_source_pdf: str | Path, _page_number: int) -> dict[str, Any]:
    return {"status": "ok", "failureReason": "", "tables": []}


def _page_text_loader(_source_pdf: str | Path) -> list[dict[str, Any]]:
    return [{"page": 3, "text": "Model Score\nA 1.0\n"}]


def test_pairing_diagnostic_records_cell_bbox_text_candidates_without_evidence(tmp_path: Path) -> None:
    overlay, pilot = _paths(tmp_path)

    payload = build_table_cell_pymupdf_pairing_diagnostic(
        table_cell_pymupdf_overlay_review_pack=overlay,
        table_cell_extractor_pilot_report=pilot,
        pairing_probe_loader=_probe,
        page_text_loader=_page_text_loader,
    )

    assert payload["schema"] == TABLE_CELL_PYMUPDF_PAIRING_DIAGNOSTIC_SCHEMA_ID
    assert validate_payload(payload, TABLE_CELL_PYMUPDF_PAIRING_DIAGNOSTIC_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "diagnostic_ready"
    assert payload["counts"]["diagnosticRows"] == 1
    assert payload["counts"]["pairingDiagnosticReadyRows"] == 1
    assert payload["counts"]["cellBboxCandidateCount"] == 4
    assert payload["counts"]["cellTextCandidateCount"] == 4
    assert payload["counts"]["cellPairingCandidateCount"] == 4
    assert payload["gate"]["cellBboxTextPairingVerified"] is False
    assert payload["gate"]["strictEvidenceReady"] is False


def test_cell_candidates_are_non_strict_even_with_unique_text_matches(tmp_path: Path) -> None:
    overlay, pilot = _paths(tmp_path)

    payload = build_table_cell_pymupdf_pairing_diagnostic(
        table_cell_pymupdf_overlay_review_pack=overlay,
        table_cell_extractor_pilot_report=pilot,
        pairing_probe_loader=_probe,
        page_text_loader=_page_text_loader,
    )
    row = payload["diagnosticRows"][0]

    assert row["diagnostic_status"] == "pairing_diagnostic_ready_non_strict"
    assert row["cell_bbox_text_pairing_verified"] is False
    assert row["cell_source_spans_created"] == 0
    assert row["table_cell_evidence_created"] is False
    assert row["strict_eligible"] is False
    assert all(cell["diagnostic_match_is_strict"] is False for cell in row["cell_candidates"])
    assert all(cell["strict_eligible"] is False for cell in row["cell_candidates"])
    assert "diagnostic_matches_are_not_cell_source_spans" in row["strict_blockers"]


def test_missing_selected_table_blocks_without_green_status(tmp_path: Path) -> None:
    overlay, pilot = _paths(tmp_path)

    payload = build_table_cell_pymupdf_pairing_diagnostic(
        table_cell_pymupdf_overlay_review_pack=overlay,
        table_cell_extractor_pilot_report=pilot,
        pairing_probe_loader=_probe_no_table,
        page_text_loader=_page_text_loader,
    )
    row = payload["diagnosticRows"][0]

    assert row["diagnostic_status"] == "blocked_selected_table_missing"
    assert row["cell_pairing_candidate_count"] == 0
    assert payload["counts"]["blockedRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0


def test_wrong_or_unsafe_input_blocks_pairing_diagnostic(tmp_path: Path) -> None:
    overlay, pilot = _paths(tmp_path, wrong_schema=True, unsafe=True)

    payload = build_table_cell_pymupdf_pairing_diagnostic(
        table_cell_pymupdf_overlay_review_pack=overlay,
        table_cell_extractor_pilot_report=pilot,
        pairing_probe_loader=_probe,
        page_text_loader=_page_text_loader,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert set(payload["gate"]["schemaViolations"]) == {
        "table_cell_pymupdf_overlay_review_pack_schema_mismatch",
        "table_cell_bbox_source_span_extractor_pilot_schema_mismatch",
    }
    assert "cellBboxTextPairingVerifiedRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert validate_payload(payload, TABLE_CELL_PYMUPDF_PAIRING_DIAGNOSTIC_SCHEMA_ID, strict=True).ok


def test_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    overlay, pilot = _paths(tmp_path / "input")
    payload = build_table_cell_pymupdf_pairing_diagnostic(
        table_cell_pymupdf_overlay_review_pack=overlay,
        table_cell_extractor_pilot_report=pilot,
        pairing_probe_loader=_probe,
        page_text_loader=_page_text_loader,
    )

    paths = write_table_cell_pymupdf_pairing_diagnostic_reports(payload, tmp_path / "reports")

    assert set(paths) == {"diagnostic", "summary", "markdown"}
    diagnostic = json.loads(Path(paths["diagnostic"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(diagnostic, TABLE_CELL_PYMUPDF_PAIRING_DIAGNOSTIC_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, TABLE_CELL_PYMUPDF_PAIRING_DIAGNOSTIC_SCHEMA_ID, strict=True).ok
    assert "does not verify cell/text pairing" in markdown
