from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_pdf_offset_selected_review_decision_record import (
    SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_RECORD_SCHEMA_ID,
    build_sectionspan_pdf_offset_selected_review_decision_record,
    write_sectionspan_pdf_offset_selected_review_decision_record_reports,
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
        "candidate_text": "1. Introduction",
        "section_type": "numbered_section",
        "section_level": 1,
        "review_priority": "P0",
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


def test_selected_decision_record_defaults_to_needs_review_without_decision_file(tmp_path: Path) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row(), _template_row("row-2")]))

    payload = build_sectionspan_pdf_offset_selected_review_decision_record(
        sectionspan_pdf_offset_selected_review_decision_template_report=template
    )

    assert payload["schema"] == SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_RECORD_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_RECORD_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "selected_decision_record_required"
    assert payload["counts"]["recordRows"] == 2
    assert payload["counts"]["needsReviewRows"] == 2
    assert payload["counts"]["approvedForLaterPromotionDesignRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["humanReviewComplete"] is False
    assert payload["gate"]["runtimePromotionAllowed"] is False
    assert all(row["recorded_decision"] == "needs_review" for row in payload["decisionRecords"])


def test_selected_decision_record_records_valid_decisions_without_creating_strict_evidence(tmp_path: Path) -> None:
    approve_id = "row-approve"
    reject_id = "row-reject"
    reject_source_id = "source-reject"
    template = _write(
        tmp_path,
        "template.json",
        _template([
            _template_row(approve_id),
            _template_row(reject_id, source_decision_row_id=reject_source_id),
        ]),
    )
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "decisions": [
                {"source_decision_row_id": approve_id, "decision": "approve_for_later_promotion_design", "reviewer": "operator"},
                {"source_decision_row_id": reject_source_id, "decision": "reject_keep_candidate_only", "notes": "ambiguous"},
            ]
        },
    )

    payload = build_sectionspan_pdf_offset_selected_review_decision_record(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        review_decisions_report=decisions,
    )

    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_RECORD_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "selected_decision_recorded"
    assert payload["counts"]["needsReviewRows"] == 0
    assert payload["counts"]["approvedForLaterPromotionDesignRows"] == 1
    assert payload["counts"]["rejectedRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["policy"]["strictEvidenceCreated"] is False
    approved = next(row for row in payload["decisionRecords"] if row["source_decision_row_id"] == approve_id)
    assert approved["recorded_decision"] == "approved_for_later_promotion_design"
    assert approved["strict_eligible"] is False
    assert "approval_is_for_later_design_only_not_runtime_evidence" in approved["strict_blockers"]


def test_selected_decision_record_blocks_invalid_unknown_or_duplicate_decisions(tmp_path: Path) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row("known", source_decision_row_id="known-source")]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "decisions": [
                {"source_decision_row_id": "known", "decision": "approve_for_later_promotion_design"},
                {"source_decision_row_id": "known-source", "decision": "approve_for_later_promotion_design"},
                {"source_decision_row_id": "unknown", "decision": "approve_for_later_promotion_design"},
                {"source_decision_row_id": "known", "decision": "invalid"},
            ]
        },
    )

    payload = build_sectionspan_pdf_offset_selected_review_decision_record(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        review_decisions_report=decisions,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "selected_review_decision_unknown_template_row_id" in payload["gate"]["unsafeUpstreamFlags"]
    assert "selected_review_decision_duplicate_row_id" in payload["gate"]["unsafeUpstreamFlags"]
    assert "selected_review_decision_invalid_value" in payload["gate"]["unsafeUpstreamFlags"]


def test_selected_decision_record_blocks_unsafe_template(tmp_path: Path) -> None:
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

    payload = build_sectionspan_pdf_offset_selected_review_decision_record(
        sectionspan_pdf_offset_selected_review_decision_template_report=template
    )

    assert payload["status"] == "blocked"
    assert "sectionspan_pdf_offset_selected_review_decision_template_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "selectedDecisionTemplate_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_selected_decision_record_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    template = _write(tmp_path / "input", "template.json", _template([_template_row()]))
    payload = build_sectionspan_pdf_offset_selected_review_decision_record(
        sectionspan_pdf_offset_selected_review_decision_template_report=template
    )

    paths = write_sectionspan_pdf_offset_selected_review_decision_record_reports(payload, tmp_path / "reports")

    assert set(paths) == {"record", "summary", "markdown"}
    record = json.loads(Path(paths["record"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(record, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_RECORD_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["needsReviewRows"] == 1
    assert "This record is report-only" in markdown
