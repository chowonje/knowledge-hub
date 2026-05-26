from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_pdf_offset_selected_review_packet import (
    SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_PACKET_SCHEMA_ID,
    build_sectionspan_pdf_offset_selected_review_packet,
    write_sectionspan_pdf_offset_selected_review_packet_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _priority_card(card_id: str, *, selected: bool = True, priority: str = "P0") -> dict:
    return {
        "priority_card_id": card_id,
        "source_decision_record_row_id": f"record:{card_id}",
        "source_decision_row_id": f"decision:{card_id}",
        "source_gate_row_id": f"gate:{card_id}",
        "source_review_card_id": f"review:{card_id}",
        "source_sectionspan_candidate_id": f"sectionspan:1706.03762:{card_id}",
        "paper_id": "1706.03762",
        "candidate_text": "1. Introduction",
        "section_type": "numbered_section",
        "section_level": 1,
        "recorded_decision": "needs_review",
        "review_priority": priority,
        "selected_for_initial_review": selected,
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
        "evidence_tier": "sectionspan_pdf_offset_review_priority_card_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
    }


def _priority_pack(cards: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-review-priority-pack.v1",
        "status": "priority_pack_ready",
        "counts": {
            "priorityCardRows": len(cards),
            "selectedInitialReviewRows": sum(1 for item in cards if item.get("selected_for_initial_review")),
            "approvedRows": 0,
            "rejectedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "priorityPackReady": True,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "priorityPackOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "priorityCards": cards,
    }
    payload.update(overrides)
    return payload


def test_selected_review_packet_emits_only_selected_cards_and_validates_schema(tmp_path: Path) -> None:
    priority_pack = _write(
        tmp_path,
        "priority-pack.json",
        _priority_pack([
            _priority_card("selected-1", selected=True),
            _priority_card("selected-2", selected=True, priority="P1"),
            _priority_card("not-selected", selected=False),
        ]),
    )

    payload = build_sectionspan_pdf_offset_selected_review_packet(
        sectionspan_pdf_offset_review_priority_pack_report=priority_pack
    )

    assert payload["schema"] == SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_PACKET_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_PACKET_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "selected_review_packet_ready"
    assert payload["counts"]["selectedReviewCards"] == 2
    assert payload["counts"]["approvedRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["runtimePromotionAllowed"] is False
    assert [item["source_priority_card_id"] for item in payload["selectedReviewCards"]] == ["selected-1", "selected-2"]
    assert all("approve_for_later_promotion_design" in item["allowed_decisions"] for item in payload["selectedReviewCards"])


def test_selected_review_packet_reports_no_selected_cards_without_creating_evidence(tmp_path: Path) -> None:
    priority_pack = _write(tmp_path, "priority-pack.json", _priority_pack([_priority_card("card-1", selected=False)]))

    payload = build_sectionspan_pdf_offset_selected_review_packet(
        sectionspan_pdf_offset_review_priority_pack_report=priority_pack
    )

    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_PACKET_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "no_selected_review_cards"
    assert payload["counts"]["selectedReviewCards"] == 0
    assert payload["policy"]["strictEvidenceCreated"] is False


def test_selected_review_packet_blocks_unsafe_priority_pack(tmp_path: Path) -> None:
    priority_pack = _write(
        tmp_path,
        "priority-pack.json",
        _priority_pack(
            [_priority_card("card-1")],
            schema="example.wrong.priority-pack.v1",
            policy={
                "reportOnly": True,
                "priorityPackOnly": True,
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

    payload = build_sectionspan_pdf_offset_selected_review_packet(
        sectionspan_pdf_offset_review_priority_pack_report=priority_pack
    )

    assert payload["status"] == "blocked"
    assert "sectionspan_pdf_offset_review_priority_pack_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "priorityPack_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_selected_review_packet_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    priority_pack = _write(tmp_path / "input", "priority-pack.json", _priority_pack([_priority_card("card-1")]))
    payload = build_sectionspan_pdf_offset_selected_review_packet(
        sectionspan_pdf_offset_review_priority_pack_report=priority_pack
    )

    paths = write_sectionspan_pdf_offset_selected_review_packet_reports(payload, tmp_path / "reports")

    assert set(paths) == {"packet", "summary", "markdown"}
    packet = json.loads(Path(paths["packet"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(packet, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_PACKET_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["selectedReviewCards"] == 1
    assert "This packet is a selected manual-review input only" in markdown
