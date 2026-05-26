from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.table_cell_next_action_gate import (
    TABLE_CELL_NEXT_ACTION_GATE_SCHEMA_ID,
    build_table_cell_next_action_gate,
    write_table_cell_next_action_gate_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pairing_report(*, wrong_schema: bool = False, unsafe: bool = False) -> dict:
    return {
        "schema": (
            "example.wrong.pairing"
            if wrong_schema
            else "knowledge-hub.paper.table-cell-pymupdf-pairing-diagnostic.v1"
        ),
        "status": "diagnostic_ready",
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
        "diagnosticRows": [
            {
                "diagnostic_row_id": "table-cell-pymupdf-pairing-diagnostic:0001",
                "source_table_region_candidate_id": "tableregion:paper-1:0001",
                "paper_id": "paper-1",
                "table_label": "Table 1",
                "cell_bbox_candidate_count": 4,
                "cell_text_candidate_count": 4,
                "cell_pairing_candidate_count": 4,
                "diagnostic_unique_cell_text_matches": 1,
                "diagnostic_ambiguous_cell_text_matches": 2,
                "diagnostic_no_match_cell_texts": 1,
            }
        ],
    }


def _alternative_report() -> dict:
    return {
        "schema": "knowledge-hub.local.table-cell-alternative-extractor-availability.v1",
        "status": "blocked",
        "counts": {
            "availableAlternativeExtractorCount": 0,
            "tableCellEvidenceCreatedRows": 0,
            "tableCellCitationGradeRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "requiresExplicitApprovalForDependencyPilot": True,
            "extractorChoiceMade": False,
            "tableCellEvidenceReady": False,
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {},
        "rows": [
            {
                "availability_row_id": "table-cell-alternative-extractor:0001",
                "source_table_region_candidate_id": "tableregion:paper-1:0002",
                "paper_id": "paper-1",
                "table_label": "Table 2",
                "blocked_reason": "pdfplumber/camelot/tabula unavailable",
            }
        ],
    }


def _visual_report() -> dict:
    return {
        "schema": "knowledge-hub.local.table-cell-pymupdf-overlay-visual-review.v1",
        "status": "visual_review_assets_ready",
        "counts": {
            "visualPairingReviewCompletedRows": 0,
            "cellBboxTextPairingVerifiedRows": 0,
            "cellSourceSpanCreatedRows": 0,
            "tableCellEvidenceCreatedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "visualPairingReviewCompleted": False,
            "cellBboxTextPairingVerified": False,
            "cellSourceSpansCreated": False,
            "tableCellEvidenceReady": False,
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {},
        "rows": [
            {
                "visual_review_row_id": "table-cell-visual-review:0001",
                "source_table_region_candidate_id": "tableregion:paper-1:0001",
                "paper_id": "paper-1",
                "table_label": "Table 1",
                "page_image_path": "/tmp/table-1.png",
                "svg_overlay_path": "/tmp/table-1.svg",
            }
        ],
    }


def _paths(root: Path, *, wrong_schema: bool = False, unsafe: bool = False) -> tuple[Path, Path, Path]:
    pairing = _write(root, "pairing.json", _pairing_report(wrong_schema=wrong_schema, unsafe=unsafe))
    alternative = _write(root, "alternative.json", _alternative_report())
    visual = _write(root, "visual.json", _visual_report())
    return pairing, alternative, visual


def test_next_action_gate_combines_human_review_and_dependency_actions(tmp_path: Path) -> None:
    pairing, alternative, visual = _paths(tmp_path)

    payload = build_table_cell_next_action_gate(
        table_cell_pymupdf_pairing_diagnostic=pairing,
        table_cell_alternative_extractor_availability=alternative,
        table_cell_pymupdf_overlay_visual_review=visual,
    )

    assert payload["schema"] == TABLE_CELL_NEXT_ACTION_GATE_SCHEMA_ID
    assert validate_payload(payload, TABLE_CELL_NEXT_ACTION_GATE_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "next_action_ready"
    assert payload["counts"]["nextActionCards"] == 3
    assert payload["counts"]["humanReviewCards"] == 2
    assert payload["counts"]["dependencyApprovalRequiredCards"] == 1
    assert payload["gate"]["humanReviewRequired"] is True
    assert payload["gate"]["dependencyApprovalRequired"] is True
    assert payload["gate"]["strictEvidenceReady"] is False
    assert payload["gate"]["parserRoutingReady"] is False


def test_visual_assets_do_not_verify_table_cell_pairing(tmp_path: Path) -> None:
    pairing, alternative, visual = _paths(tmp_path)

    payload = build_table_cell_next_action_gate(
        table_cell_pymupdf_pairing_diagnostic=pairing,
        table_cell_alternative_extractor_availability=alternative,
        table_cell_pymupdf_overlay_visual_review=visual,
    )
    visual_card = [
        item for item in payload["nextActionCards"] if item["action_type"] == "inspect_visual_overlay_asset"
    ][0]

    assert visual_card["action_status"] == "ready_for_human_review"
    assert visual_card["strict_eligible"] is False
    assert visual_card["runtime_evidence"] is False
    assert "visual_pairing_review_not_completed" in visual_card["strict_blockers"]
    assert payload["gate"]["cellBboxTextPairingVerified"] is False
    assert payload["gate"]["tableCellEvidenceReady"] is False


def test_unavailable_alternative_extractors_require_explicit_approval(tmp_path: Path) -> None:
    pairing, alternative, visual = _paths(tmp_path)

    payload = build_table_cell_next_action_gate(
        table_cell_pymupdf_pairing_diagnostic=pairing,
        table_cell_alternative_extractor_availability=alternative,
        table_cell_pymupdf_overlay_visual_review=visual,
    )
    card = [
        item
        for item in payload["nextActionCards"]
        if item["action_type"] == "approve_isolated_alternative_extractor_dependency_pilot"
    ][0]

    assert card["action_status"] == "blocked_requires_explicit_approval"
    assert card["strict_eligible"] is False
    assert "dependency_pilot_requires_explicit_approval" in card["strict_blockers"]
    assert payload["counts"]["alternativeExtractorApprovalCards"] == 1


def test_unsafe_upstream_evidence_flags_block_the_gate(tmp_path: Path) -> None:
    pairing, alternative, visual = _paths(tmp_path, unsafe=True)

    payload = build_table_cell_next_action_gate(
        table_cell_pymupdf_pairing_diagnostic=pairing,
        table_cell_alternative_extractor_availability=alternative,
        table_cell_pymupdf_overlay_visual_review=visual,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["nextActionReady"] is False
    assert payload["gate"]["decision"] == "blocked"
    assert "pairing_cellBboxTextPairingVerifiedRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert validate_payload(payload, TABLE_CELL_NEXT_ACTION_GATE_SCHEMA_ID, strict=True).ok


def test_schema_mismatch_blocks_the_gate(tmp_path: Path) -> None:
    pairing, alternative, visual = _paths(tmp_path, wrong_schema=True)

    payload = build_table_cell_next_action_gate(
        table_cell_pymupdf_pairing_diagnostic=pairing,
        table_cell_alternative_extractor_availability=alternative,
        table_cell_pymupdf_overlay_visual_review=visual,
    )

    assert payload["status"] == "blocked"
    assert "table_cell_pymupdf_pairing_diagnostic_schema_mismatch" in payload["gate"]["schemaViolations"]
    assert validate_payload(payload, TABLE_CELL_NEXT_ACTION_GATE_SCHEMA_ID, strict=True).ok


def test_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    pairing, alternative, visual = _paths(tmp_path / "input")
    payload = build_table_cell_next_action_gate(
        table_cell_pymupdf_pairing_diagnostic=pairing,
        table_cell_alternative_extractor_availability=alternative,
        table_cell_pymupdf_overlay_visual_review=visual,
    )

    paths = write_table_cell_next_action_gate_reports(payload, tmp_path / "reports")

    assert set(paths) == {"gate", "summary", "markdown"}
    gate = json.loads(Path(paths["gate"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(gate, TABLE_CELL_NEXT_ACTION_GATE_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, TABLE_CELL_NEXT_ACTION_GATE_SCHEMA_ID, strict=True).ok
    assert "does not verify pairing" in markdown
