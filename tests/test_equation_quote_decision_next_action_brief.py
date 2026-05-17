from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_quote_decision_next_action_brief import (
    EQUATION_QUOTE_DECISION_NEXT_ACTION_BRIEF_SCHEMA_ID,
    build_equation_quote_decision_next_action_brief,
    write_equation_quote_decision_next_action_brief_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _review_row(index: int, action_type: str = "review_diagnostic_page_context") -> dict:
    return {
        "review_sheet_row_id": f"equation-quote-manual-review:{index:04d}",
        "source_action_card_id": f"equation-quote-next-action:{index:04d}",
        "source_equation_quote_candidate_id": f"equationquote:1706.03762:{index:04d}",
        "paper_id": "1706.03762",
        "candidate_text": "softmax(QK^T / sqrt(d_k))V",
        "equation_label": f"eq-{index}",
        "action_type": action_type,
        "action_status": "ready_for_human_review",
        "priority": "high",
        "current_decision": "needs_review",
        "allowed_decisions": [
            "needs_review",
            "accept_diagnostic_context_for_later_reextract_design",
            "reject_equation_quote_candidate",
            "request_equation_quote_reextraction",
            "keep_blocked",
        ],
        "recommended_review_action": "review_diagnostic_page_context_before_any_later_reextract_or_source_span_design",
        "review_prompt": "Inspect diagnostic context before choosing a non-default decision.",
        "canonical_alignment_status": "failed",
        "canonical_alignment_method": "none",
        "alignment_feasibility_status": "diagnostic_term_context_candidate_only",
        "pdf_offset_feasibility_status": "diagnostic_page_context_candidate_only",
        "diagnostic_terms": ["softmax", "sqrt"],
        "diagnostic_page_candidates": [{"page": 3, "coverage": 1.0}],
        "best_diagnostic_page_coverage": 1.0,
        "layout_element_count": 2,
        "bbox_available": True,
        "sourceContentHash": "hash",
        "source_span_created": False,
        "original_pdf_offset_recovered": False,
        "equation_semantics_interpreted": False,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
    }


def _manual_sheet(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.equation-quote-manual-review-sheet.v1",
        "status": "manual_review_sheet_ready",
        "counts": {
            "reviewRows": len(rows),
            "needsReviewRows": len(rows),
            "sourceSpanCreatedRows": 0,
            "originalPdfOffsetRecoveredRows": 0,
            "equationSemanticsInterpretedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "manualReviewSheetOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "reviewRows": rows,
    }
    payload.update(overrides)
    return payload


def _validation_report(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.equation-quote-decision-file-validation.v1",
        "status": "decision_file_validated",
        "counts": {
            "validRows": 1,
            "invalidRows": 0,
            "missingRows": 0,
            "sourceSpanCreatedRows": 0,
            "originalPdfOffsetRecoveredRows": 0,
            "equationSemanticsInterpretedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "humanReviewRecordComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "decisionFileValidationOnly": True,
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
    decision = "needs_review" if needs_review else "accepted_diagnostic_context_for_later_reextract_design"
    payload = {
        "schema": "knowledge-hub.paper.equation-quote-decision-record.v1",
        "status": "decision_record_required" if needs_review else "decision_recorded",
        "inputs": {"reviewDecisionsReport": "/tmp/equation-quote-decisions.draft.json"},
        "counts": {
            "recordRows": 1,
            "needsReviewRows": needs_review,
            "acceptedDiagnosticContextRows": 0 if needs_review else 1,
            "rejectedRows": 0,
            "reextractRequestRows": 0,
            "keptBlockedRows": 0,
            "sourceSpanCreatedRows": 0,
            "originalPdfOffsetRecoveredRows": 0,
            "equationSemanticsInterpretedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "decisionRecordReady": True,
            "humanReviewComplete": needs_review == 0,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "decisionRecordOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "decisionRecords": [
            {
                "source_review_sheet_row_id": "equation-quote-manual-review:0001",
                "recorded_decision": decision,
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_equation_quote_next_action_brief_keeps_pending_rows_manual_only(tmp_path: Path) -> None:
    manual = _write(tmp_path, "manual.json", _manual_sheet([_review_row(1)]))
    validation = _write(tmp_path, "validation.json", _validation_report())
    record = _write(tmp_path, "record.json", _decision_record(needs_review=1))

    payload = build_equation_quote_decision_next_action_brief(
        equation_quote_manual_review_sheet_report=manual,
        equation_quote_decision_file_validation_report=validation,
        equation_quote_decision_record_report=record,
    )

    assert payload["schema"] == EQUATION_QUOTE_DECISION_NEXT_ACTION_BRIEF_SCHEMA_ID
    assert validate_payload(payload, EQUATION_QUOTE_DECISION_NEXT_ACTION_BRIEF_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "manual_review_required"
    assert payload["gate"]["manualReviewRequired"] is True
    assert payload["gate"]["autoApprovalAllowed"] is False
    assert payload["gate"]["humanReviewComplete"] is False
    assert payload["gate"]["nextEditTarget"] == "/tmp/equation-quote-decisions.draft.json"
    assert payload["counts"]["briefRows"] == 1
    assert payload["counts"]["needsReviewRows"] == 1
    assert payload["counts"]["decisionRecordNeedsReviewRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["policy"]["strictEvidenceCreated"] is False
    row = payload["briefRows"][0]
    assert row["safe_default_decision"] == "needs_review"
    assert row["source_span_created"] is False
    assert row["equation_semantics_interpreted"] is False
    assert row["runtime_promotion_allowed"] is False
    assert "reviewer" in row["required_for_non_needs_review_decision"]


def test_equation_quote_next_action_brief_does_not_treat_recorded_decisions_as_runtime_evidence(
    tmp_path: Path,
) -> None:
    manual = _write(tmp_path, "manual.json", _manual_sheet([_review_row(1)]))
    validation = _write(tmp_path, "validation.json", _validation_report())
    record = _write(tmp_path, "record.json", _decision_record(needs_review=0))

    payload = build_equation_quote_decision_next_action_brief(
        equation_quote_manual_review_sheet_report=manual,
        equation_quote_decision_file_validation_report=validation,
        equation_quote_decision_record_report=record,
    )

    assert validate_payload(payload, EQUATION_QUOTE_DECISION_NEXT_ACTION_BRIEF_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "manual_review_recorded_non_runtime"
    assert payload["gate"]["humanReviewComplete"] is True
    assert payload["gate"]["runtimePromotionAllowed"] is False
    assert payload["counts"]["acceptedDiagnosticContextRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["briefRows"][0]["runtime_evidence"] is False


def test_equation_quote_next_action_brief_blocks_unsafe_upstream_reports(tmp_path: Path) -> None:
    manual = _write(tmp_path, "manual.json", _manual_sheet([_review_row(1)], schema="example.wrong"))
    validation = _write(
        tmp_path,
        "validation.json",
        _validation_report(status="decision_file_incomplete", counts={"validRows": 0, "invalidRows": 1}),
    )
    record = _write(
        tmp_path,
        "record.json",
        _decision_record(
            needs_review=1,
            policy={
                "reportOnly": True,
                "decisionRecordOnly": True,
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

    payload = build_equation_quote_decision_next_action_brief(
        equation_quote_manual_review_sheet_report=manual,
        equation_quote_decision_file_validation_report=validation,
        equation_quote_decision_record_report=record,
    )

    assert payload["status"] == "blocked"
    assert "equationQuoteManualReviewSheet_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "equationQuoteDecisionFileValidation_status_unexpected" in payload["gate"]["unsafeUpstreamFlags"]
    assert "equationQuoteDecisionFileValidation_invalidRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert "equationQuoteDecisionRecord_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_equation_quote_next_action_brief_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    manual = _write(tmp_path / "input", "manual.json", _manual_sheet([_review_row(1)]))
    validation = _write(tmp_path / "input", "validation.json", _validation_report())
    record = _write(tmp_path / "input", "record.json", _decision_record(needs_review=1))
    payload = build_equation_quote_decision_next_action_brief(
        equation_quote_manual_review_sheet_report=manual,
        equation_quote_decision_file_validation_report=validation,
        equation_quote_decision_record_report=record,
    )

    paths = write_equation_quote_decision_next_action_brief_reports(payload, tmp_path / "out")

    written = json.loads(Path(paths["brief"]).read_text(encoding="utf-8"))
    assert validate_payload(written, EQUATION_QUOTE_DECISION_NEXT_ACTION_BRIEF_SCHEMA_ID, strict=True).ok
    assert Path(paths["summary"]).exists()
    assert "does not approve rows" in Path(paths["markdown"]).read_text(encoding="utf-8")
