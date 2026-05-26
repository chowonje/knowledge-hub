from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_pdf_offset_human_review_gate import (
    SECTIONSPAN_PDF_OFFSET_HUMAN_REVIEW_GATE_SCHEMA_ID,
    build_sectionspan_pdf_offset_human_review_gate,
    write_sectionspan_pdf_offset_human_review_gate_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _card(card_id: str = "sectionspan-pdf-offset-review-card:0001", **overrides: object) -> dict:
    payload = {
        "review_card_id": card_id,
        "source_sectionspan_candidate_id": "sectionspan:1706.03762:0001",
        "paper_id": "1706.03762",
        "candidate_text": "1. Introduction",
        "section_type": "numbered_section",
        "section_level": 1,
        "canonical_span": {
            "chars_start": 10,
            "chars_end": 25,
            "page": 1,
            "sourceContentHash": "hash1",
            "locatorKind": "canonical_generated_markdown",
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
        "review_status": "ready_for_human_review",
        "evidence_tier": "sectionspan_pdf_offset_recovery_review_card_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
    }
    payload.update(overrides)
    return payload


def _review_pack(cards: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-recovery-review-pack.v1",
        "status": "review_pack_ready",
        "counts": {
            "reviewCardRows": len(cards),
            "readyForHumanReviewRows": sum(1 for item in cards if item.get("review_status") == "ready_for_human_review"),
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "reviewPackReady": True,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "reviewCards": cards,
    }
    payload.update(overrides)
    return payload


def test_human_review_gate_defaults_ready_cards_to_pending_and_validates_schema(tmp_path: Path) -> None:
    review_pack = _write(tmp_path, "review-pack.json", _review_pack([_card(), _card("sectionspan-pdf-offset-review-card:0002")]))

    payload = build_sectionspan_pdf_offset_human_review_gate(
        sectionspan_pdf_offset_recovery_review_pack_report=review_pack
    )

    assert payload["schema"] == SECTIONSPAN_PDF_OFFSET_HUMAN_REVIEW_GATE_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_HUMAN_REVIEW_GATE_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "review_required"
    assert payload["counts"]["gateRows"] == 2
    assert payload["counts"]["pendingHumanReviewRows"] == 2
    assert payload["counts"]["approvedForLaterPromotionDesignRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["humanReviewComplete"] is False
    assert payload["gate"]["runtimePromotionAllowed"] is False
    assert all(row["human_review_status"] == "pending_human_review" for row in payload["gateRows"])


def test_human_review_gate_records_decisions_without_creating_strict_evidence(tmp_path: Path) -> None:
    approved_id = "sectionspan-pdf-offset-review-card:0001"
    rejected_id = "sectionspan-pdf-offset-review-card:0002"
    review_pack = _write(tmp_path, "review-pack.json", _review_pack([_card(approved_id), _card(rejected_id)]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "approvedReviewCardIds": [approved_id],
            "rejectedReviewCardIds": [rejected_id],
        },
    )

    payload = build_sectionspan_pdf_offset_human_review_gate(
        sectionspan_pdf_offset_recovery_review_pack_report=review_pack,
        review_decisions_report=decisions,
    )

    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_HUMAN_REVIEW_GATE_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "review_recorded"
    assert payload["counts"]["pendingHumanReviewRows"] == 0
    assert payload["counts"]["approvedForLaterPromotionDesignRows"] == 1
    assert payload["counts"]["rejectedRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["policy"]["strictEvidenceCreated"] is False
    approved = next(row for row in payload["gateRows"] if row["source_review_card_id"] == approved_id)
    assert approved["review_decision_scope"] == "later_sectionspan_promotion_design_only"
    assert approved["strict_eligible"] is False
    assert "approval_is_for_later_design_only_not_runtime_evidence" in approved["strict_blockers"]


def test_human_review_gate_does_not_approve_held_out_upstream_rows(tmp_path: Path) -> None:
    held_out_id = "sectionspan-pdf-offset-review-card:0001"
    review_pack = _write(
        tmp_path,
        "review-pack.json",
        _review_pack([_card(held_out_id, review_status="held_out_page_conflict", page_agreement=False)]),
    )
    decisions = _write(tmp_path, "decisions.json", {"approvedReviewCardIds": [held_out_id]})

    payload = build_sectionspan_pdf_offset_human_review_gate(
        sectionspan_pdf_offset_recovery_review_pack_report=review_pack,
        review_decisions_report=decisions,
    )

    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_HUMAN_REVIEW_GATE_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "review_recorded"
    assert payload["counts"]["approvedForLaterPromotionDesignRows"] == 0
    assert payload["counts"]["heldOutRows"] == 1
    row = payload["gateRows"][0]
    assert row["human_review_status"] == "held_out_upstream_blocked"
    assert "upstream_review_card_not_ready" in row["strict_blockers"]


def test_human_review_gate_blocks_unsafe_upstream_or_overlapping_decisions(tmp_path: Path) -> None:
    card_id = "sectionspan-pdf-offset-review-card:0001"
    review_pack = _write(
        tmp_path,
        "review-pack.json",
        _review_pack([_card(card_id)], schema="example.wrong.review-pack.v1"),
    )
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "approvedReviewCardIds": [card_id],
            "rejectedReviewCardIds": [card_id],
        },
    )

    payload = build_sectionspan_pdf_offset_human_review_gate(
        sectionspan_pdf_offset_recovery_review_pack_report=review_pack,
        review_decisions_report=decisions,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "sectionspan_pdf_offset_recovery_review_pack_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "review_decisions_approved_and_rejected_overlap" in payload["gate"]["unsafeUpstreamFlags"]


def test_human_review_gate_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    review_pack = _write(tmp_path / "input", "review-pack.json", _review_pack([_card()]))
    payload = build_sectionspan_pdf_offset_human_review_gate(
        sectionspan_pdf_offset_recovery_review_pack_report=review_pack
    )

    paths = write_sectionspan_pdf_offset_human_review_gate_reports(payload, tmp_path / "reports")

    assert set(paths) == {"gate", "summary", "markdown"}
    gate = json.loads(Path(paths["gate"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(gate, SECTIONSPAN_PDF_OFFSET_HUMAN_REVIEW_GATE_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["pendingHumanReviewRows"] == 1
    assert "This gate records review state only" in markdown
