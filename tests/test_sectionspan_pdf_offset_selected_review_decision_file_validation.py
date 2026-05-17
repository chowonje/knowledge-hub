from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_pdf_offset_selected_review_decision_file_validation import (
    SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_VALIDATION_SCHEMA_ID,
    build_sectionspan_pdf_offset_selected_review_decision_file_validation,
    write_sectionspan_pdf_offset_selected_review_decision_file_validation_reports,
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
        "default_decision": "needs_review",
        "evidence_tier": "sectionspan_pdf_offset_selected_review_decision_template_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
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


def test_selected_decision_file_validation_reports_missing_file_without_recording(tmp_path: Path) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row()]))

    payload = build_sectionspan_pdf_offset_selected_review_decision_file_validation(
        sectionspan_pdf_offset_selected_review_decision_template_report=template
    )

    assert payload["schema"] == SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_VALIDATION_SCHEMA_ID
    assert validate_payload(
        payload,
        SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_VALIDATION_SCHEMA_ID,
        strict=True,
    ).ok
    assert payload["status"] == "selected_decision_file_required"
    assert payload["counts"]["missingRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["containsRecordedDecisions"] is False
    assert payload["gate"]["runtimePromotionAllowed"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False


def test_selected_decision_file_validation_accepts_needs_review_draft_as_pending(tmp_path: Path) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row()]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        {"decisions": [{"source_decision_row_id": "sectionspan-pdf-offset-selected-review-decision:0001", "decision": "needs_review"}]},
    )

    payload = build_sectionspan_pdf_offset_selected_review_decision_file_validation(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        review_decisions_report=decisions,
    )

    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_VALIDATION_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "selected_decision_file_validated"
    assert payload["counts"]["validRows"] == 1
    assert payload["counts"]["needsReviewRows"] == 1
    assert payload["counts"]["submittedApprovedForLaterPromotionDesignRows"] == 0
    assert payload["counts"]["runtimeEvidenceRows"] == 0
    assert payload["gate"]["selectedDecisionFileComplete"] is True
    assert payload["gate"]["humanReviewRecordComplete"] is False


def test_selected_decision_file_validation_accepts_reviewed_rows_without_promoting_runtime(tmp_path: Path) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row("approve-row"), _template_row("reject-row")]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "decisions": [
                {
                    "source_decision_row_id": "approve-row",
                    "decision": "approve_for_later_promotion_design",
                    "reviewer": "reviewer",
                    "notes": "offset verified for later design review",
                },
                {
                    "source_decision_row_id": "reject-row",
                    "decision": "reject_keep_candidate_only",
                    "reviewer": "reviewer",
                    "notes": "boundary ambiguous",
                },
            ]
        },
    )

    payload = build_sectionspan_pdf_offset_selected_review_decision_file_validation(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        review_decisions_report=decisions,
    )

    assert payload["status"] == "selected_decision_file_validated"
    assert payload["counts"]["validRows"] == 2
    assert payload["counts"]["submittedApprovedForLaterPromotionDesignRows"] == 1
    assert payload["counts"]["submittedRejectedRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["policy"]["runtimePromotionAllowed"] is False
    assert all(row["strict_eligible"] is False for row in payload["validationRows"])


def test_selected_decision_file_validation_blocks_unknown_duplicate_or_invalid_decisions(tmp_path: Path) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row("known")]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "decisions": [
                {"source_decision_row_id": "known", "decision": "approve_for_later_promotion_design"},
                {"source_decision_row_id": "known", "decision": "needs_review"},
                {"source_decision_row_id": "unknown", "decision": "needs_review"},
            ]
        },
    )

    payload = build_sectionspan_pdf_offset_selected_review_decision_file_validation(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        review_decisions_report=decisions,
    )

    assert payload["status"] == "blocked"
    assert "selected_review_decision_duplicate_row_id" in payload["gate"]["fileValidationErrors"]
    assert "selected_review_decision_unknown_template_row_id" in payload["gate"]["fileValidationErrors"]


def test_selected_decision_file_validation_requires_reviewer_and_notes_for_non_needs_review(
    tmp_path: Path,
) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row("known")]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        {"decisions": [{"source_decision_row_id": "known", "decision": "approve_for_later_promotion_design"}]},
    )

    payload = build_sectionspan_pdf_offset_selected_review_decision_file_validation(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        review_decisions_report=decisions,
    )

    assert payload["status"] == "selected_decision_file_incomplete"
    assert payload["counts"]["invalidRows"] == 1
    assert "reviewer_required_for_non_needs_review_decision" in payload["validationRows"][0]["validation_errors"]
    assert "notes_required_for_non_needs_review_decision" in payload["validationRows"][0]["validation_errors"]


def test_selected_decision_file_validation_blocks_unsafe_template(tmp_path: Path) -> None:
    template = _write(
        tmp_path,
        "template.json",
        _template(
            [_template_row()],
            schema="example.wrong.template.v1",
            policy={
                "reportOnly": True,
                "selectedDecisionTemplateOnly": True,
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

    payload = build_sectionspan_pdf_offset_selected_review_decision_file_validation(
        sectionspan_pdf_offset_selected_review_decision_template_report=template
    )

    assert payload["status"] == "blocked"
    assert (
        "sectionspan_pdf_offset_selected_review_decision_template_schema_mismatch"
        in payload["gate"]["unsafeUpstreamFlags"]
    )
    assert "selectedDecisionTemplate_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_selected_decision_file_validation_writer_outputs_schema_valid_files(tmp_path: Path) -> None:
    template = _write(tmp_path / "input", "template.json", _template([_template_row()]))
    payload = build_sectionspan_pdf_offset_selected_review_decision_file_validation(
        sectionspan_pdf_offset_selected_review_decision_template_report=template
    )

    paths = write_sectionspan_pdf_offset_selected_review_decision_file_validation_reports(
        payload,
        tmp_path / "reports",
    )

    assert set(paths) == {"validation", "summary", "markdown"}
    validation = json.loads(Path(paths["validation"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(
        validation,
        SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_VALIDATION_SCHEMA_ID,
        strict=True,
    ).ok
    assert summary["counts"]["missingRows"] == 1
    assert "This validation report is report-only" in markdown
