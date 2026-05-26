from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_pdf_offset_selected_review_next_action_brief import (
    SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_NEXT_ACTION_BRIEF_SCHEMA_ID,
    build_sectionspan_pdf_offset_selected_review_next_action_brief,
    write_sectionspan_pdf_offset_selected_review_next_action_brief_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manual_row(row_id: str = "sectionspan-pdf-offset-selected-review-decision:0001", **overrides: object) -> dict:
    payload = {
        "manual_sheet_row_id": "sectionspan-pdf-offset-selected-review-manual-sheet:0001",
        "source_decision_row_id": row_id,
        "source_sectionspan_candidate_id": "sectionspan:1706.03762:0001",
        "paper_id": "1706.03762",
        "candidate_text": "1 Introduction",
        "section_type": "numbered_section",
        "review_priority": "P0",
        "current_decision": "needs_review",
        "review_suggestion": "approve_for_later_promotion_design",
        "review_suggestion_reason": "candidate_text_exactly_matches_original_pdf_page_offset",
        "validation_status": "valid",
        "review_context_status": "review_context_ready",
        "page_text_match": True,
        "context_match_method": "exact",
        "matched_text": "1 Introduction",
        "context_before": "Abstract text.",
        "context_after": "Section text.",
        "allowed_decisions": [
            "needs_review",
            "approve_for_later_promotion_design",
            "reject_keep_candidate_only",
        ],
        "required_review_checks": ["confirm_heading_text_matches_original_pdf_at_recorded_page_and_offset"],
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
    }
    payload.update(overrides)
    return payload


def _manual_sheet(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-manual-sheet.v1",
        "status": "selected_manual_sheet_ready",
        "inputs": {
            "selectedReviewDecisionsFile": "/tmp/sectionspan-pdf-offset-selected-review-decisions.draft.json",
        },
        "counts": {
            "manualSheetRows": len(rows),
            "needsReviewRows": len(rows),
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "selectedManualSheetReady": True,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "selectedManualSheetOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "manualRows": rows,
    }
    payload.update(overrides)
    return payload


def _validation_report(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-file-validation.v1",
        "status": "selected_decision_file_validated",
        "counts": {
            "validRows": 1,
            "invalidRows": 0,
            "missingRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "selectedDecisionFileValidationReady": True,
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
    }
    payload.update(overrides)
    return payload


def _decision_record(needs_review: int = 1, **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-record.v1",
        "status": "selected_decision_record_required" if needs_review else "selected_decision_recorded",
        "counts": {
            "needsReviewRows": needs_review,
            "approvedForLaterPromotionDesignRows": 0 if needs_review else 1,
            "rejectedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "selectedDecisionRecordReady": True,
            "humanReviewComplete": needs_review == 0,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "selectedDecisionRecordOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
    }
    payload.update(overrides)
    return payload


def test_next_action_brief_keeps_pending_rows_manual_only(tmp_path: Path) -> None:
    manual = _write(tmp_path, "manual.json", _manual_sheet([_manual_row()]))
    validation = _write(tmp_path, "validation.json", _validation_report())
    record = _write(tmp_path, "record.json", _decision_record(needs_review=1))

    payload = build_sectionspan_pdf_offset_selected_review_next_action_brief(
        sectionspan_pdf_offset_selected_review_manual_sheet_report=manual,
        sectionspan_pdf_offset_selected_review_decision_file_validation_report=validation,
        sectionspan_pdf_offset_selected_review_decision_record_report=record,
    )

    assert payload["schema"] == SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_NEXT_ACTION_BRIEF_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_NEXT_ACTION_BRIEF_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "manual_review_required"
    assert payload["gate"]["manualReviewRequired"] is True
    assert payload["gate"]["autoApprovalAllowed"] is False
    assert payload["gate"]["humanReviewComplete"] is False
    assert payload["counts"]["briefRows"] == 1
    assert payload["counts"]["needsReviewRows"] == 1
    assert payload["counts"]["decisionRecordNeedsReviewRows"] == 1
    assert payload["counts"]["suggestedApproveForLaterPromotionDesignRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["policy"]["strictEvidenceCreated"] is False
    row = payload["briefRows"][0]
    assert row["safe_default_decision"] == "needs_review"
    assert row["strict_eligible"] is False
    assert row["runtime_promotion_allowed"] is False
    assert "reviewer" in row["required_for_non_needs_review_decision"]


def test_next_action_brief_does_not_treat_recorded_decisions_as_runtime_evidence(tmp_path: Path) -> None:
    manual = _write(
        tmp_path,
        "manual.json",
        _manual_sheet([_manual_row(current_decision="approve_for_later_promotion_design")]),
    )
    validation = _write(tmp_path, "validation.json", _validation_report())
    record = _write(tmp_path, "record.json", _decision_record(needs_review=0))

    payload = build_sectionspan_pdf_offset_selected_review_next_action_brief(
        sectionspan_pdf_offset_selected_review_manual_sheet_report=manual,
        sectionspan_pdf_offset_selected_review_decision_file_validation_report=validation,
        sectionspan_pdf_offset_selected_review_decision_record_report=record,
    )

    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_NEXT_ACTION_BRIEF_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "manual_review_recorded_non_runtime"
    assert payload["gate"]["humanReviewComplete"] is True
    assert payload["gate"]["runtimePromotionAllowed"] is False
    assert payload["counts"]["nonNeedsReviewRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["briefRows"][0]["runtime_evidence"] is False


def test_next_action_brief_blocks_unsafe_upstream_reports(tmp_path: Path) -> None:
    manual = _write(tmp_path, "manual.json", _manual_sheet([_manual_row()], schema="example.wrong"))
    validation = _write(tmp_path, "validation.json", _validation_report())
    record = _write(
        tmp_path,
        "record.json",
        _decision_record(
            needs_review=1,
            policy={
                "reportOnly": True,
                "selectedDecisionRecordOnly": True,
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

    payload = build_sectionspan_pdf_offset_selected_review_next_action_brief(
        sectionspan_pdf_offset_selected_review_manual_sheet_report=manual,
        sectionspan_pdf_offset_selected_review_decision_file_validation_report=validation,
        sectionspan_pdf_offset_selected_review_decision_record_report=record,
    )

    assert payload["status"] == "blocked"
    assert "selectedReviewManualSheet_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "selectedDecisionRecord_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_next_action_brief_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    manual = _write(tmp_path / "input", "manual.json", _manual_sheet([_manual_row()]))
    validation = _write(tmp_path / "input", "validation.json", _validation_report())
    record = _write(tmp_path / "input", "record.json", _decision_record(needs_review=1))
    payload = build_sectionspan_pdf_offset_selected_review_next_action_brief(
        sectionspan_pdf_offset_selected_review_manual_sheet_report=manual,
        sectionspan_pdf_offset_selected_review_decision_file_validation_report=validation,
        sectionspan_pdf_offset_selected_review_decision_record_report=record,
    )

    paths = write_sectionspan_pdf_offset_selected_review_next_action_brief_reports(payload, tmp_path / "out")

    written = json.loads(Path(paths["brief"]).read_text(encoding="utf-8"))
    assert validate_payload(written, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_NEXT_ACTION_BRIEF_SCHEMA_ID, strict=True).ok
    assert Path(paths["summary"]).exists()
    assert "does not approve rows" in Path(paths["markdown"]).read_text(encoding="utf-8")
