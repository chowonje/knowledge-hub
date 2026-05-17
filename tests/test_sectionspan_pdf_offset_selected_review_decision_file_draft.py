from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_pdf_offset_selected_review_decision_file_draft import (
    SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_DRAFT_SCHEMA_ID,
    build_sectionspan_pdf_offset_selected_review_decision_file_draft,
    write_sectionspan_pdf_offset_selected_review_decision_file_draft_reports,
)
from knowledge_hub.papers.sectionspan_pdf_offset_selected_review_decision_record import (
    SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_RECORD_SCHEMA_ID,
    build_sectionspan_pdf_offset_selected_review_decision_record,
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


def test_selected_review_decision_file_draft_emits_only_needs_review_rows(tmp_path: Path) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row(), _template_row("row-2")]))

    payload = build_sectionspan_pdf_offset_selected_review_decision_file_draft(
        sectionspan_pdf_offset_selected_review_decision_template_report=template
    )

    assert payload["schema"] == SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_DRAFT_SCHEMA_ID
    assert validate_payload(
        payload,
        SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_DRAFT_SCHEMA_ID,
        strict=True,
    ).ok
    assert payload["status"] == "selected_decision_file_draft_ready"
    assert payload["counts"]["draftRows"] == 2
    assert payload["counts"]["needsReviewRows"] == 2
    assert payload["counts"]["approvedForLaterPromotionDesignRows"] == 0
    assert payload["counts"]["rejectedRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["containsOnlyNeedsReviewDefaults"] is True
    assert payload["gate"]["strictEvidenceReady"] is False
    assert all(row["decision"] == "needs_review" for row in payload["draftRows"])
    assert all(row["strict_eligible"] is False for row in payload["draftRows"])
    assert all(row["decision"] == "needs_review" for row in payload["decisionFileDraft"]["decisions"])


def test_selected_review_decision_file_draft_can_be_consumed_as_pending_decision_file(tmp_path: Path) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row()]))
    payload = build_sectionspan_pdf_offset_selected_review_decision_file_draft(
        sectionspan_pdf_offset_selected_review_decision_template_report=template
    )
    decisions_file = _write(tmp_path, "selected-review-decisions.draft.json", payload["decisionFileDraft"])

    record = build_sectionspan_pdf_offset_selected_review_decision_record(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        review_decisions_report=decisions_file,
    )

    assert validate_payload(record, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_RECORD_SCHEMA_ID, strict=True).ok
    assert record["status"] == "selected_decision_record_required"
    assert record["counts"]["recordRows"] == 1
    assert record["counts"]["needsReviewRows"] == 1
    assert record["counts"]["approvedForLaterPromotionDesignRows"] == 0
    assert record["counts"]["rejectedRows"] == 0
    assert record["counts"]["strictEligibleRows"] == 0
    assert record["gate"]["humanReviewComplete"] is False
    assert record["gate"]["runtimePromotionAllowed"] is False
    assert record["decisionRecords"][0]["recorded_decision"] == "needs_review"


def test_selected_review_decision_file_draft_blocks_unsafe_template(tmp_path: Path) -> None:
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

    payload = build_sectionspan_pdf_offset_selected_review_decision_file_draft(
        sectionspan_pdf_offset_selected_review_decision_template_report=template
    )

    assert payload["status"] == "blocked"
    assert (
        "sectionspan_pdf_offset_selected_review_decision_template_schema_mismatch"
        in payload["gate"]["unsafeUpstreamFlags"]
    )
    assert "selectedDecisionTemplate_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_selected_review_decision_file_draft_writer_outputs_schema_valid_files(tmp_path: Path) -> None:
    template = _write(tmp_path / "input", "template.json", _template([_template_row()]))
    payload = build_sectionspan_pdf_offset_selected_review_decision_file_draft(
        sectionspan_pdf_offset_selected_review_decision_template_report=template
    )

    paths = write_sectionspan_pdf_offset_selected_review_decision_file_draft_reports(payload, tmp_path / "reports")

    assert set(paths) == {"draftReport", "decisionFileDraft", "summary", "markdown"}
    report = json.loads(Path(paths["draftReport"]).read_text(encoding="utf-8"))
    decision_file = json.loads(Path(paths["decisionFileDraft"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_FILE_DRAFT_SCHEMA_ID, strict=True).ok
    assert decision_file["draftOnly"] is True
    assert decision_file["decisions"][0]["decision"] == "needs_review"
    assert summary["counts"]["draftRows"] == 1
    assert "This draft is an editable starting point only" in markdown
