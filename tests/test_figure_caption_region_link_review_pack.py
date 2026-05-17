from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.figure_caption_region_link_review_pack import (
    FIGURE_CAPTION_REGION_LINK_REVIEW_PACK_SCHEMA_ID,
    build_figure_caption_region_link_review_pack,
    write_figure_caption_region_link_review_pack_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pdf_offset_report(root: Path) -> Path:
    return _write(
        root,
        "figure-caption-pdf-offset-feasibility.json",
        {
            "schema": "knowledge-hub.paper.figure-caption-pdf-offset-feasibility.v1",
            "status": "feasibility_complete",
            "feasibilityRows": [
                {
                    "source_figure_caption_candidate_id": "figurecaption:paper-1:0001",
                    "paper_id": "paper-1",
                    "candidate_text": "Figure 1: The model architecture.",
                    "caption_text": "The model architecture.",
                    "figure_label": "Figure 1",
                    "original_pdf_offset_recovered": True,
                    "original_pdf_span": {
                        "originalPdfCharsStart": 10,
                        "originalPdfCharsEnd": 42,
                        "page": 2,
                        "sourceContentHash": "hash-source",
                        "matchMethod": "exact",
                        "matchConfidence": 1.0,
                    },
                    "page_agrees_with_canonical": True,
                    "source_hash_agrees_with_canonical": True,
                    "strict_blockers": [
                        "report_only",
                        "figure_region_link_incomplete",
                        "runtime_promotion_disabled_for_tranche",
                    ],
                },
                {
                    "source_figure_caption_candidate_id": "figurecaption:paper-1:0002",
                    "paper_id": "paper-1",
                    "candidate_text": "Figure 2: Missing caption.",
                    "caption_text": "Missing caption.",
                    "figure_label": "Figure 2",
                    "original_pdf_offset_recovered": False,
                    "original_pdf_span": {
                        "originalPdfCharsStart": None,
                        "originalPdfCharsEnd": None,
                        "page": None,
                        "sourceContentHash": "hash-source",
                        "matchMethod": "",
                        "matchConfidence": 0.0,
                    },
                    "page_agrees_with_canonical": False,
                    "source_hash_agrees_with_canonical": False,
                    "strict_blockers": [
                        "report_only",
                        "original_pdf_offset_not_recovered",
                        "figure_region_link_incomplete",
                    ],
                },
            ],
        },
    )


def _region_report(root: Path) -> Path:
    return _write(
        root,
        "figure-region-link-feasibility-audit.json",
        {
            "schema": "knowledge-hub.paper.figure-region-link-feasibility-audit.v1",
            "status": "ok",
            "rows": [
                {
                    "audit_id": "figure-region-link-feasibility:0001",
                    "figure_caption_candidate_id": "figurecaption:paper-1:0001",
                    "paper_id": "paper-1",
                    "layout_region_candidate_present": True,
                    "layout_element_ids": ["mineru:element:1", "mineru:element:2"],
                    "layout_element_count": 2,
                    "bbox": [10.0, 20.0, 200.0, 240.0],
                    "layout_link_reason": "ordinal_figure_like_layout_match_without_page",
                    "normalizer_candidate_id": "paper-1:figure-caption:0001",
                    "normalizer_region_page": None,
                    "region_page_recovered": False,
                    "caption_page_matches_region_page": False,
                    "figure_region_type_verified": False,
                    "figure_region_link_verified": False,
                    "strict_blockers": [
                        "figure_region_page_missing",
                        "figure_region_type_unverified",
                    ],
                },
                {
                    "audit_id": "figure-region-link-feasibility:0002",
                    "figure_caption_candidate_id": "figurecaption:paper-1:0002",
                    "paper_id": "paper-1",
                    "layout_region_candidate_present": True,
                    "layout_element_ids": ["mineru:element:3"],
                    "layout_element_count": 1,
                    "bbox": [30.0, 40.0, 300.0, 360.0],
                    "normalizer_region_page": None,
                    "region_page_recovered": False,
                    "caption_page_matches_region_page": False,
                    "figure_region_type_verified": False,
                    "figure_region_link_verified": False,
                    "strict_blockers": [
                        "figure_region_page_missing",
                        "figure_region_type_unverified",
                    ],
                },
            ],
        },
    )


def test_region_link_review_pack_emits_ready_and_held_out_cards_with_valid_schema(tmp_path: Path) -> None:
    pdf_path = _pdf_offset_report(tmp_path)
    region_path = _region_report(tmp_path)

    payload = build_figure_caption_region_link_review_pack(
        figure_caption_pdf_offset_feasibility_report=pdf_path,
        figure_region_link_feasibility_report=region_path,
    )

    assert payload["schema"] == FIGURE_CAPTION_REGION_LINK_REVIEW_PACK_SCHEMA_ID
    assert validate_payload(payload, FIGURE_CAPTION_REGION_LINK_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "review_pack_ready"
    assert payload["counts"]["reviewCardRows"] == 2
    assert payload["counts"]["readyForRegionReviewRows"] == 1
    assert payload["counts"]["heldOutRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["counts"]["figureRegionLinkVerifiedRows"] == 0
    assert payload["gate"]["decision"] == "ready_for_region_link_human_review"


def test_ready_card_keeps_caption_offset_and_layout_region_non_strict(tmp_path: Path) -> None:
    payload = build_figure_caption_region_link_review_pack(
        figure_caption_pdf_offset_feasibility_report=_pdf_offset_report(tmp_path),
        figure_region_link_feasibility_report=_region_report(tmp_path),
    )

    card = next(item for item in payload["reviewCards"] if item["figure_label"] == "Figure 1")
    assert card["review_status"] == "ready_for_region_link_review"
    assert card["original_pdf_offset_recovered"] is True
    assert card["original_pdf_span"]["page"] == 2
    assert card["layout_region_candidate_present"] is True
    assert card["layout_element_count"] == 2
    assert card["figure_region_type_verified"] is False
    assert card["figure_region_link_verified"] is False
    assert card["strict_eligible"] is False
    assert card["citation_grade"] is False
    assert "figure_region_link_incomplete" in card["strict_blockers"]
    assert "figure_region_page_missing" in card["strict_blockers"]
    assert "region_link_review_pack_only" in card["strict_blockers"]
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["databaseMutation"] is False


def test_missing_original_pdf_offset_is_held_out_not_green(tmp_path: Path) -> None:
    payload = build_figure_caption_region_link_review_pack(
        figure_caption_pdf_offset_feasibility_report=_pdf_offset_report(tmp_path),
        figure_region_link_feasibility_report=_region_report(tmp_path),
    )

    card = next(item for item in payload["reviewCards"] if item["figure_label"] == "Figure 2")
    assert card["review_status"] == "held_out_original_pdf_offset_missing"
    assert card["original_pdf_offset_recovered"] is False
    assert card["strict_eligible"] is False
    assert "original_pdf_offset_not_recovered" in card["strict_blockers"]
    assert "held_out_original_pdf_offset_missing" in card["strict_blockers"]


def test_wrong_input_schema_blocks_review_pack(tmp_path: Path) -> None:
    pdf_path = _write(tmp_path, "bad-pdf.json", {"schema": "wrong", "status": "ok", "feasibilityRows": []})
    region_path = _write(tmp_path, "bad-region.json", {"schema": "wrong", "status": "ok", "rows": []})

    payload = build_figure_caption_region_link_review_pack(
        figure_caption_pdf_offset_feasibility_report=pdf_path,
        figure_region_link_feasibility_report=region_path,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert set(payload["gate"]["schemaViolations"]) == {
        "figure_caption_pdf_offset_feasibility_schema_mismatch",
        "figure_caption_pdf_offset_feasibility_not_complete",
        "figure_region_link_feasibility_schema_mismatch",
    }


def test_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    payload = build_figure_caption_region_link_review_pack(
        figure_caption_pdf_offset_feasibility_report=_pdf_offset_report(tmp_path / "input"),
        figure_region_link_feasibility_report=_region_report(tmp_path / "input"),
    )

    paths = write_figure_caption_region_link_review_pack_reports(payload, tmp_path / "reports")

    assert set(paths) == {"cards", "summary", "markdown"}
    cards = json.loads(Path(paths["cards"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(cards, FIGURE_CAPTION_REGION_LINK_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, FIGURE_CAPTION_REGION_LINK_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert "A recovered caption source span does not verify" in markdown
