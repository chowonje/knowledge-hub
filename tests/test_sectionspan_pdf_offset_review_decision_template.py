from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_pdf_offset_review_decision_template import (
    SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_TEMPLATE_SCHEMA_ID,
    build_sectionspan_pdf_offset_review_decision_template,
    write_sectionspan_pdf_offset_review_decision_template_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _gate_row(row_id: str = "sectionspan-pdf-offset-human-review-gate:0001", **overrides: object) -> dict:
    payload = {
        "gate_row_id": row_id,
        "source_review_card_id": "sectionspan-pdf-offset-review-card:0001",
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
        "human_review_status": "pending_human_review",
        "evidence_tier": "sectionspan_pdf_offset_human_review_gate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
    }
    payload.update(overrides)
    return payload


def _human_review_gate(rows: list[dict], **overrides: object) -> dict:
    pending = [row for row in rows if row.get("human_review_status") == "pending_human_review"]
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-human-review-gate.v1",
        "status": "review_required" if pending else "review_recorded",
        "counts": {
            "gateRows": len(rows),
            "pendingHumanReviewRows": len(pending),
            "approvedForLaterPromotionDesignRows": 0,
            "rejectedRows": 0,
            "heldOutRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "humanReviewGateReady": True,
            "humanReviewComplete": not pending,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "humanReviewGateOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "gateRows": rows,
    }
    payload.update(overrides)
    return payload


def test_decision_template_emits_pending_rows_and_validates_schema(tmp_path: Path) -> None:
    gate = _write(
        tmp_path,
        "gate.json",
        _human_review_gate([_gate_row(), _gate_row("sectionspan-pdf-offset-human-review-gate:0002")]),
    )

    payload = build_sectionspan_pdf_offset_review_decision_template(
        sectionspan_pdf_offset_human_review_gate_report=gate
    )

    assert payload["schema"] == SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_TEMPLATE_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_TEMPLATE_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "decision_template_ready"
    assert payload["counts"]["templateRows"] == 2
    assert payload["counts"]["pendingDecisionRows"] == 2
    assert payload["counts"]["approvedRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["decisionTemplateReady"] is True
    assert payload["gate"]["runtimePromotionAllowed"] is False
    assert all(row["default_decision"] == "needs_review" for row in payload["decisionRows"])


def test_decision_template_does_not_emit_already_reviewed_rows(tmp_path: Path) -> None:
    gate = _write(
        tmp_path,
        "gate.json",
        _human_review_gate(
            [
                _gate_row("pending"),
                _gate_row("approved", human_review_status="approved_for_later_promotion_design"),
                _gate_row("rejected", human_review_status="rejected_keep_candidate_only"),
            ]
        ),
    )

    payload = build_sectionspan_pdf_offset_review_decision_template(
        sectionspan_pdf_offset_human_review_gate_report=gate
    )

    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_TEMPLATE_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["templateRows"] == 1
    assert payload["decisionRows"][0]["source_gate_row_id"] == "pending"
    assert payload["decisionRows"][0]["strict_eligible"] is False
    assert "approve_for_later_promotion_design" in payload["decisionRows"][0]["allowed_decisions"]


def test_decision_template_reports_no_pending_rows_without_approving_anything(tmp_path: Path) -> None:
    gate = _write(
        tmp_path,
        "gate.json",
        _human_review_gate([_gate_row(human_review_status="approved_for_later_promotion_design")]),
    )

    payload = build_sectionspan_pdf_offset_review_decision_template(
        sectionspan_pdf_offset_human_review_gate_report=gate
    )

    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_TEMPLATE_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "no_pending_decisions"
    assert payload["counts"]["templateRows"] == 0
    assert payload["counts"]["approvedRows"] == 0
    assert payload["gate"]["humanReviewComplete"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False


def test_decision_template_blocks_unsafe_upstream_gate(tmp_path: Path) -> None:
    gate = _write(
        tmp_path,
        "gate.json",
        _human_review_gate(
            [_gate_row()],
            schema="example.wrong.human-review-gate.v1",
            policy={
                "reportOnly": True,
                "humanReviewGateOnly": True,
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

    payload = build_sectionspan_pdf_offset_review_decision_template(
        sectionspan_pdf_offset_human_review_gate_report=gate
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "sectionspan_pdf_offset_human_review_gate_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "humanReviewGate_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_decision_template_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    gate = _write(tmp_path / "input", "gate.json", _human_review_gate([_gate_row()]))
    payload = build_sectionspan_pdf_offset_review_decision_template(
        sectionspan_pdf_offset_human_review_gate_report=gate
    )

    paths = write_sectionspan_pdf_offset_review_decision_template_reports(payload, tmp_path / "reports")

    assert set(paths) == {"template", "summary", "markdown"}
    template = json.loads(Path(paths["template"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(template, SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_TEMPLATE_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["pendingDecisionRows"] == 1
    assert "This template is a review worksheet only" in markdown
