from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_pdf_offset_selected_review_decision_proposal import (
    SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_PROPOSAL_SCHEMA_ID,
    build_sectionspan_pdf_offset_selected_review_decision_proposal,
    write_sectionspan_pdf_offset_selected_review_decision_proposal_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _evidence_row(row_id: str = "row-1", **overrides: object) -> dict:
    payload = {
        "review_evidence_row_id": f"evidence:{row_id}",
        "source_decision_row_id": f"decision:{row_id}",
        "source_selected_review_card_id": f"selected:{row_id}",
        "source_sectionspan_candidate_id": f"sectionspan:1706.03762:{row_id}",
        "paper_id": "1706.03762",
        "candidate_text": "1. Introduction",
        "section_type": "numbered_section",
        "section_level": 1,
        "review_priority": "P0",
        "review_context_status": "review_context_ready",
        "page_text_match": True,
        "context_match_method": "exact",
        "matched_text": "1. Introduction",
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
        "sourceContentHash": "hash1",
        "review_suggestion": "approve_for_later_promotion_design",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
    }
    payload.update(overrides)
    return payload


def _evidence_pack(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-evidence-pack.v1",
        "status": "selected_review_evidence_pack_ready",
        "counts": {
            "evidenceRows": len(rows),
            "reviewContextReadyRows": len(rows),
            "blockedRows": 0,
            "pageTextMatchRows": len(rows),
            "exactTextRows": len(rows),
            "normalizedTextRows": 0,
            "suggestedApproveForLaterPromotionDesignRows": len(rows),
            "suggestedNeedsReviewRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "selectedReviewEvidencePackReady": True,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "selectedReviewEvidencePackOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "evidenceRows": rows,
    }
    payload.update(overrides)
    return payload


def test_decision_proposal_emits_non_binding_approve_suggestion_without_human_decision(tmp_path: Path) -> None:
    evidence = _write(tmp_path, "evidence-pack.json", _evidence_pack([_evidence_row()]))

    payload = build_sectionspan_pdf_offset_selected_review_decision_proposal(
        sectionspan_pdf_offset_selected_review_evidence_pack_report=evidence
    )

    assert payload["schema"] == SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_PROPOSAL_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_PROPOSAL_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "decision_proposal_ready"
    assert payload["counts"]["proposalRows"] == 1
    assert payload["counts"]["proposedApproveForLaterPromotionDesignRows"] == 1
    assert payload["counts"]["acceptedHumanDecisionRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["humanReviewComplete"] is False
    row = payload["proposalRows"][0]
    assert row["proposed_decision"] == "approve_for_later_promotion_design"
    assert row["accepted_as_human_decision"] is False
    assert row["human_decision_required"] is True
    assert row["strict_eligible"] is False
    assert "decision_proposals_are_not_human_review_decisions" in row["non_strict_reason"]
    assert "proposalRows" in payload
    assert "decisionRows" not in payload
    assert "decisions" not in payload


def test_decision_proposal_keeps_non_ready_context_as_needs_review(tmp_path: Path) -> None:
    evidence = _write(
        tmp_path,
        "evidence-pack.json",
        _evidence_pack([
            _evidence_row(
                review_context_status="blocked_page_text_mismatch",
                page_text_match=False,
                review_suggestion="needs_review",
            )
        ]),
    )

    payload = build_sectionspan_pdf_offset_selected_review_decision_proposal(
        sectionspan_pdf_offset_selected_review_evidence_pack_report=evidence
    )

    row = payload["proposalRows"][0]
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_PROPOSAL_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["proposedNeedsReviewRows"] == 1
    assert row["proposed_decision"] == "needs_review"
    assert row["accepted_as_human_decision"] is False


def test_decision_proposal_blocks_unsafe_evidence_pack(tmp_path: Path) -> None:
    evidence = _write(
        tmp_path,
        "evidence-pack.json",
        _evidence_pack(
            [_evidence_row()],
            schema="example.wrong.evidence-pack.v1",
            policy={
                "reportOnly": True,
                "selectedReviewEvidencePackOnly": True,
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

    payload = build_sectionspan_pdf_offset_selected_review_decision_proposal(
        sectionspan_pdf_offset_selected_review_evidence_pack_report=evidence
    )

    assert payload["status"] == "blocked"
    assert "sectionspan_pdf_offset_selected_review_evidence_pack_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "selectedReviewEvidencePack_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_decision_proposal_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    evidence = _write(tmp_path / "input", "evidence-pack.json", _evidence_pack([_evidence_row()]))
    payload = build_sectionspan_pdf_offset_selected_review_decision_proposal(
        sectionspan_pdf_offset_selected_review_evidence_pack_report=evidence
    )

    paths = write_sectionspan_pdf_offset_selected_review_decision_proposal_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_PROPOSAL_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["proposedApproveForLaterPromotionDesignRows"] == 1
    assert "This proposal is not a human decision file" in markdown
