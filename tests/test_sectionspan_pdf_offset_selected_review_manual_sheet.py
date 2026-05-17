from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_pdf_offset_selected_review_manual_sheet import (
    SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_MANUAL_SHEET_SCHEMA_ID,
    build_sectionspan_pdf_offset_selected_review_manual_sheet,
    write_sectionspan_pdf_offset_selected_review_manual_sheet_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _template_row(row_id: str = "sectionspan-pdf-offset-selected-review-decision:0001", **overrides: object) -> dict:
    payload = {
        "decision_row_id": row_id,
        "source_selected_review_card_id": "sectionspan-pdf-offset-selected-review-card:0001",
        "source_priority_card_id": "sectionspan-pdf-offset-review-priority-card:0001",
        "source_decision_record_row_id": "sectionspan-pdf-offset-review-decision-record:0001",
        "source_decision_row_id": "sectionspan-pdf-offset-review-decision:0001",
        "source_gate_row_id": "sectionspan-pdf-offset-human-review-gate:0001",
        "source_review_card_id": "sectionspan-pdf-offset-review-card:0001",
        "source_sectionspan_candidate_id": "sectionspan:1706.03762:0001",
        "paper_id": "1706.03762",
        "candidate_text": "1 Introduction",
        "section_type": "numbered_section",
        "section_level": 1,
        "review_priority": "P0",
        "canonical_span": {
            "chars_start": 10,
            "chars_end": 24,
            "page": 1,
            "sourceContentHash": "hash1",
            "locatorKind": "canonical_generated_markdown",
        },
        "original_pdf_span": {
            "originalPdfCharsStart": 8,
            "originalPdfCharsEnd": 22,
            "page": 1,
            "sourceContentHash": "hash1",
            "matchMethod": "exact",
            "matchConfidence": 1.0,
        },
        "page_agreement": True,
        "source_hash_agreement": True,
        "allowed_decisions": [
            "needs_review",
            "approve_for_later_promotion_design",
            "reject_keep_candidate_only",
        ],
        "required_review_checks": ["confirm_heading_text_matches_original_pdf_at_recorded_page_and_offset"],
    }
    payload.update(overrides)
    return payload


def _template(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-template.v1",
        "status": "selected_decision_template_ready",
        "counts": {
            "templateRows": len(rows),
            "pendingDecisionRows": len(rows),
            "approvedRows": 0,
            "rejectedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "selectedDecisionTemplateReady": True,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "selectedDecisionTemplateOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "decisionRows": rows,
    }
    payload.update(overrides)
    return payload


def _evidence_row(row_id: str = "sectionspan-pdf-offset-selected-review-decision:0001", **overrides: object) -> dict:
    payload = {
        "review_evidence_row_id": "sectionspan-pdf-offset-selected-review-evidence:0001",
        "source_decision_row_id": row_id,
        "source_selected_review_card_id": "sectionspan-pdf-offset-selected-review-card:0001",
        "source_sectionspan_candidate_id": "sectionspan:1706.03762:0001",
        "paper_id": "1706.03762",
        "candidate_text": "1 Introduction",
        "section_type": "numbered_section",
        "section_level": 1,
        "review_priority": "P0",
        "canonical_span": {},
        "original_pdf_span": {},
        "review_context_status": "review_context_ready",
        "page_text_match": True,
        "context_match_method": "exact",
        "matched_text": "1 Introduction",
        "context_before": "Abstract text before.",
        "context_after": "Section text after.",
        "review_suggestion": "approve_for_later_promotion_design",
        "review_suggestion_reason": "candidate_text_exactly_matches_original_pdf_page_offset",
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


def _validation_row(row_id: str = "sectionspan-pdf-offset-selected-review-decision:0001", **overrides: object) -> dict:
    payload = {
        "validation_row_id": "sectionspan-pdf-offset-selected-review-decision-file-validation:0001",
        "source_decision_row_id": row_id,
        "paper_id": "1706.03762",
        "candidate_text": "1 Introduction",
        "section_type": "numbered_section",
        "section_level": 1,
        "review_priority": "P0",
        "submitted_decision": "needs_review",
        "reviewer": "",
        "notes": "",
        "validation_status": "valid",
        "validation_errors": [],
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
    }
    payload.update(overrides)
    return payload


def _validation_report(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-file-validation.v1",
        "status": "selected_decision_file_validated",
        "counts": {
            "validationRows": len(rows),
            "validRows": len(rows),
            "invalidRows": 0,
            "missingRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "selectedDecisionFileValidationReady": True,
            "selectedDecisionFileComplete": True,
            "containsRecordedDecisions": False,
            "humanReviewRecordComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "selectedDecisionFileValidationOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "validationRows": rows,
    }
    payload.update(overrides)
    return payload


def _decisions(rows: list[dict]) -> dict:
    return {"draftOnly": True, "decisions": rows}


def test_selected_manual_sheet_consolidates_evidence_draft_and_validation(tmp_path: Path) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row()]))
    evidence = _write(tmp_path, "evidence.json", _evidence_pack([_evidence_row()]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        {"decisions": [{"source_decision_row_id": "sectionspan-pdf-offset-selected-review-decision:0001", "decision": "needs_review"}]},
    )
    validation = _write(tmp_path, "validation.json", _validation_report([_validation_row()]))

    payload = build_sectionspan_pdf_offset_selected_review_manual_sheet(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        sectionspan_pdf_offset_selected_review_evidence_pack_report=evidence,
        selected_review_decisions_file=decisions,
        sectionspan_pdf_offset_selected_review_decision_file_validation_report=validation,
    )

    assert payload["schema"] == SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_MANUAL_SHEET_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_MANUAL_SHEET_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "selected_manual_sheet_ready"
    assert payload["counts"]["manualSheetRows"] == 1
    assert payload["counts"]["needsReviewRows"] == 1
    assert payload["counts"]["validationValidRows"] == 1
    assert payload["counts"]["reviewContextReadyRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["runtimePromotionAllowed"] is False
    row = payload["manualRows"][0]
    assert row["current_decision"] == "needs_review"
    assert row["review_suggestion"] == "approve_for_later_promotion_design"
    assert row["validation_status"] == "valid"
    assert row["strict_eligible"] is False


def test_selected_manual_sheet_does_not_promote_non_needs_review_draft_values(tmp_path: Path) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row("approve-row")]))
    evidence = _write(tmp_path, "evidence.json", _evidence_pack([_evidence_row("approve-row")]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        _decisions(
            [
                {
                    "source_decision_row_id": "approve-row",
                    "decision": "approve_for_later_promotion_design",
                    "reviewer": "reviewer",
                    "notes": "checked",
                }
            ]
        ),
    )
    validation = _write(
        tmp_path,
        "validation.json",
        _validation_report(
            [
                _validation_row(
                    "approve-row",
                    submitted_decision="approve_for_later_promotion_design",
                    reviewer="reviewer",
                    notes="checked",
                )
            ]
        ),
    )

    payload = build_sectionspan_pdf_offset_selected_review_manual_sheet(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        sectionspan_pdf_offset_selected_review_evidence_pack_report=evidence,
        selected_review_decisions_file=decisions,
        sectionspan_pdf_offset_selected_review_decision_file_validation_report=validation,
    )

    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_MANUAL_SHEET_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["needsReviewRows"] == 0
    assert payload["counts"]["nonNeedsReviewRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["counts"]["runtimeEvidenceRows"] == 0
    assert payload["gate"]["containsNonNeedsReviewDraftValues"] is True
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["manualRows"][0]["strict_eligible"] is False


def test_selected_manual_sheet_blocks_unsafe_inputs(tmp_path: Path) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row()], schema="example.wrong.template.v1"))
    evidence = _write(
        tmp_path,
        "evidence.json",
        _evidence_pack(
            [_evidence_row()],
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
    decisions = _write(tmp_path, "decisions.json", _decisions([]))
    validation = _write(tmp_path, "validation.json", _validation_report([_validation_row()]))

    payload = build_sectionspan_pdf_offset_selected_review_manual_sheet(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        sectionspan_pdf_offset_selected_review_evidence_pack_report=evidence,
        selected_review_decisions_file=decisions,
        sectionspan_pdf_offset_selected_review_decision_file_validation_report=validation,
    )

    assert payload["status"] == "blocked"
    assert "selectedDecisionTemplate_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "selectedReviewEvidencePack_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_selected_manual_sheet_writer_outputs_schema_valid_files(tmp_path: Path) -> None:
    template = _write(tmp_path / "input", "template.json", _template([_template_row()]))
    evidence = _write(tmp_path / "input", "evidence.json", _evidence_pack([_evidence_row()]))
    decisions = _write(
        tmp_path / "input",
        "decisions.json",
        _decisions([{"source_decision_row_id": "sectionspan-pdf-offset-selected-review-decision:0001", "decision": "needs_review"}]),
    )
    validation = _write(tmp_path / "input", "validation.json", _validation_report([_validation_row()]))
    payload = build_sectionspan_pdf_offset_selected_review_manual_sheet(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        sectionspan_pdf_offset_selected_review_evidence_pack_report=evidence,
        selected_review_decisions_file=decisions,
        sectionspan_pdf_offset_selected_review_decision_file_validation_report=validation,
    )

    paths = write_sectionspan_pdf_offset_selected_review_manual_sheet_reports(payload, tmp_path / "reports")

    assert set(paths) == {"sheet", "summary", "markdown"}
    sheet = json.loads(Path(paths["sheet"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(sheet, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_MANUAL_SHEET_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["manualSheetRows"] == 1
    assert "This sheet is local review metadata only" in markdown
