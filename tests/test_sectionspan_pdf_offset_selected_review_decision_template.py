from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_pdf_offset_selected_review_decision_template import (
    SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_TEMPLATE_SCHEMA_ID,
    build_sectionspan_pdf_offset_selected_review_decision_template,
    write_sectionspan_pdf_offset_selected_review_decision_template_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _selected_card(card_id: str, *, priority: str = "P0") -> dict:
    return {
        "selected_review_card_id": f"selected:{card_id}",
        "source_priority_card_id": f"priority:{card_id}",
        "source_decision_record_row_id": f"record:{card_id}",
        "source_decision_row_id": f"decision:{card_id}",
        "source_gate_row_id": f"gate:{card_id}",
        "source_review_card_id": f"review:{card_id}",
        "source_sectionspan_candidate_id": f"sectionspan:1706.03762:{card_id}",
        "paper_id": "1706.03762",
        "candidate_text": "1. Introduction",
        "section_type": "numbered_section",
        "section_level": 1,
        "review_priority": priority,
        "priority_reasons": ["top_level_numbered_section_boundary"],
        "canonical_span": {
            "chars_start": 10,
            "chars_end": 25,
            "page": 1,
            "sourceContentHash": "hash1",
        },
        "original_pdf_span": {
            "originalPdfCharsStart": 9,
            "originalPdfCharsEnd": 24,
            "page": 1,
            "sourceContentHash": "hash1",
            "matchMethod": "exact",
            "matchConfidence": 1.0,
        },
        "page_agreement": True,
        "source_hash_agreement": True,
        "review_checklist": ["verify_heading_text_on_original_pdf_page"],
        "evidence_tier": "sectionspan_pdf_offset_selected_review_card_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
    }


def _selected_packet(cards: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-packet.v1",
        "status": "selected_review_packet_ready",
        "counts": {
            "selectedReviewCards": len(cards),
            "approvedRows": 0,
            "rejectedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
            "unsafeUpstreamFlagCount": 0,
            "byPaper": {"1706.03762": len(cards)},
            "bySectionType": {"numbered_section": len(cards)},
            "byReviewPriority": {"P0": len(cards)},
        },
        "gate": {
            "selectedReviewPacketReady": True,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "selectedReviewPacketOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "selectedReviewCards": cards,
    }
    payload.update(overrides)
    return payload


def test_selected_decision_template_emits_pending_rows_and_validates_schema(tmp_path: Path) -> None:
    packet = _write(
        tmp_path,
        "selected-packet.json",
        _selected_packet([
            _selected_card("card-1"),
            _selected_card("card-2", priority="P1"),
        ]),
    )

    payload = build_sectionspan_pdf_offset_selected_review_decision_template(
        sectionspan_pdf_offset_selected_review_packet_report=packet
    )

    assert payload["schema"] == SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_TEMPLATE_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_TEMPLATE_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "selected_decision_template_ready"
    assert payload["counts"]["templateRows"] == 2
    assert payload["counts"]["pendingDecisionRows"] == 2
    assert payload["counts"]["approvedRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["humanReviewComplete"] is False
    assert payload["gate"]["runtimePromotionAllowed"] is False
    assert [item["source_selected_review_card_id"] for item in payload["decisionRows"]] == ["selected:card-1", "selected:card-2"]
    assert all(item["default_decision"] == "needs_review" for item in payload["decisionRows"])
    assert all(item["strict_eligible"] is False for item in payload["decisionRows"])
    assert all("needs_review" in item["allowed_decisions"] for item in payload["decisionRows"])


def test_selected_decision_template_reports_no_selected_cards_without_creating_evidence(tmp_path: Path) -> None:
    packet = _write(tmp_path, "selected-packet.json", _selected_packet([]))

    payload = build_sectionspan_pdf_offset_selected_review_decision_template(
        sectionspan_pdf_offset_selected_review_packet_report=packet
    )

    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_TEMPLATE_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "no_selected_decisions"
    assert payload["counts"]["templateRows"] == 0
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["gate"]["selectedDecisionTemplateReady"] is False


def test_selected_decision_template_blocks_unsafe_selected_packet(tmp_path: Path) -> None:
    packet = _write(
        tmp_path,
        "selected-packet.json",
        _selected_packet(
            [_selected_card("card-1")],
            schema="example.wrong.selected-packet.v1",
            policy={
                "reportOnly": True,
                "selectedReviewPacketOnly": True,
                "strictEvidenceCreated": True,
                "runtimePromotionAllowed": False,
                "parserRoutingChanged": False,
                "canonicalParsedArtifactsWritten": False,
                "databaseMutation": False,
                "reindexOrReembed": False,
                "answerIntegrationChanged": False,
            },
        ),
    )

    payload = build_sectionspan_pdf_offset_selected_review_decision_template(
        sectionspan_pdf_offset_selected_review_packet_report=packet
    )

    assert payload["status"] == "blocked"
    assert "sectionspan_pdf_offset_selected_review_packet_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "selectedReviewPacket_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_selected_decision_template_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    packet = _write(tmp_path / "input", "selected-packet.json", _selected_packet([_selected_card("card-1")]))
    payload = build_sectionspan_pdf_offset_selected_review_decision_template(
        sectionspan_pdf_offset_selected_review_packet_report=packet
    )

    paths = write_sectionspan_pdf_offset_selected_review_decision_template_reports(payload, tmp_path / "reports")

    assert set(paths) == {"template", "summary", "markdown"}
    template = json.loads(Path(paths["template"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(template, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_TEMPLATE_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["pendingDecisionRows"] == 1
    assert "This template is a selected manual-review worksheet only" in markdown
