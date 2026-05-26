from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_quote_decision_record import (
    EQUATION_QUOTE_DECISION_RECORD_SCHEMA_ID,
    build_equation_quote_decision_record,
    write_equation_quote_decision_record_reports,
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
        "source_equation_quote_candidate_id": f"equation_quote:{index:04d}",
        "paper_id": "1706.03762",
        "candidate_text": "softmax(QK^T / sqrt(d_k))V",
        "equation_label": f"eq-{index}",
        "action_type": action_type,
        "action_status": "ready_for_human_review",
        "priority": "high",
        "allowed_decisions": [
            "needs_review",
            "accept_diagnostic_context_for_later_reextract_design",
            "reject_equation_quote_candidate",
            "request_equation_quote_reextraction",
            "keep_blocked",
        ],
        "evidence_tier": "equation_quote_manual_review_sheet_only",
        "report_only": True,
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
            "nonNeedsReviewRows": 0,
            "humanReviewRows": len(rows),
            "blockedRows": 0,
            "diagnosticPageContextRows": len(rows),
            "unmatchedEquationQuoteRows": 0,
            "sourceSpanCreatedRows": 0,
            "originalPdfOffsetRecoveredRows": 0,
            "equationSemanticsInterpretedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "manualReviewSheetReady": True,
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


def _validation_report() -> dict:
    return {
        "schema": "knowledge-hub.paper.equation-quote-decision-file-validation.v1",
        "status": "decision_file_validated",
        "counts": {
            "missingRows": 0,
            "invalidRows": 0,
            "sourceSpanCreatedRows": 0,
            "originalPdfOffsetRecoveredRows": 0,
            "equationSemanticsInterpretedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
    }


def _decision(row_id: str, decision: str = "needs_review", reviewer: str = "", notes: str = "") -> dict:
    return {
        "source_review_sheet_row_id": row_id,
        "decision": decision,
        "reviewer": reviewer,
        "notes": notes,
    }


def test_equation_quote_decision_record_keeps_needs_review_rows_pending_non_runtime(tmp_path: Path) -> None:
    sheet = _write(tmp_path, "manual-sheet.json", _manual_sheet([_review_row(1), _review_row(2)]))
    validation = _write(tmp_path, "validation.json", _validation_report())
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "decisions": [
                _decision("equation-quote-manual-review:0001"),
                _decision("equation-quote-manual-review:0002"),
            ]
        },
    )

    payload = build_equation_quote_decision_record(
        equation_quote_manual_review_sheet_report=sheet,
        equation_quote_decision_file_validation_report=validation,
        review_decisions_report=decisions,
    )

    assert payload["schema"] == EQUATION_QUOTE_DECISION_RECORD_SCHEMA_ID
    assert validate_payload(payload, EQUATION_QUOTE_DECISION_RECORD_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "decision_record_required"
    assert payload["counts"]["recordRows"] == 2
    assert payload["counts"]["needsReviewRows"] == 2
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["counts"]["runtimeEvidenceRows"] == 0
    assert payload["gate"]["humanReviewComplete"] is False
    assert payload["gate"]["runtimePromotionAllowed"] is False
    assert all(row["recorded_decision"] == "needs_review" for row in payload["decisionRecords"])


def test_equation_quote_decision_record_maps_non_needs_review_decisions_without_runtime_promotion(
    tmp_path: Path,
) -> None:
    sheet = _write(tmp_path, "manual-sheet.json", _manual_sheet([_review_row(1), _review_row(2)]))
    validation = _write(tmp_path, "validation.json", _validation_report())
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "decisions": [
                _decision(
                    "equation-quote-manual-review:0001",
                    "accept_diagnostic_context_for_later_reextract_design",
                    reviewer="operator",
                    notes="diagnostic page context looks useful for later re-extraction design",
                ),
                _decision(
                    "equation-quote-manual-review:0002",
                    "request_equation_quote_reextraction",
                    reviewer="operator",
                    notes="quote should be re-extracted with a better equation parser",
                ),
            ]
        },
    )

    payload = build_equation_quote_decision_record(
        equation_quote_manual_review_sheet_report=sheet,
        equation_quote_decision_file_validation_report=validation,
        review_decisions_report=decisions,
    )

    assert validate_payload(payload, EQUATION_QUOTE_DECISION_RECORD_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "decision_recorded"
    assert payload["counts"]["needsReviewRows"] == 0
    assert payload["counts"]["acceptedDiagnosticContextRows"] == 1
    assert payload["counts"]["reextractRequestRows"] == 1
    assert payload["gate"]["humanReviewComplete"] is True
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["answerIntegrationChanged"] is False
    assert all(row["strict_eligible"] is False for row in payload["decisionRecords"])
    assert all(row["runtime_evidence"] is False for row in payload["decisionRecords"])


def test_equation_quote_decision_record_requires_validation_report_for_submitted_decisions(
    tmp_path: Path,
) -> None:
    sheet = _write(tmp_path, "manual-sheet.json", _manual_sheet([_review_row(1)]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "decisions": [
                _decision(
                    "equation-quote-manual-review:0001",
                    "request_equation_quote_reextraction",
                    reviewer="operator",
                    notes="looks like the equation should be extracted again",
                )
            ]
        },
    )

    payload = build_equation_quote_decision_record(
        equation_quote_manual_review_sheet_report=sheet,
        review_decisions_report=decisions,
    )

    assert payload["status"] == "blocked"
    assert "equation_quote_decision_file_validation_missing" in payload["gate"]["unsafeUpstreamFlags"]
    assert payload["gate"]["humanReviewComplete"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False


def test_equation_quote_decision_record_blocks_unknown_duplicate_or_invalid_decisions(tmp_path: Path) -> None:
    sheet = _write(tmp_path, "manual-sheet.json", _manual_sheet([_review_row(1)]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "decisions": [
                _decision("equation-quote-manual-review:0001"),
                _decision("equation-quote-manual-review:0001"),
                _decision("missing-row"),
                _decision("equation-quote-manual-review:0001", "approve_runtime_evidence"),
            ]
        },
    )

    payload = build_equation_quote_decision_record(
        equation_quote_manual_review_sheet_report=sheet,
        review_decisions_report=decisions,
    )

    assert payload["status"] == "blocked"
    assert "equation_quote_decision_duplicate_row_id" in payload["gate"]["unsafeUpstreamFlags"]
    assert "equation_quote_decision_unknown_review_row_id" in payload["gate"]["unsafeUpstreamFlags"]
    assert "equation_quote_decision_invalid_value" in payload["gate"]["unsafeUpstreamFlags"]
    assert payload["gate"]["decisionRecordReady"] is False
    assert payload["policy"]["parserRoutingChanged"] is False


def test_equation_quote_decision_record_blocks_unsafe_manual_sheet_or_validation_report(tmp_path: Path) -> None:
    sheet = _write(
        tmp_path,
        "manual-sheet.json",
        _manual_sheet(
            [_review_row(1)],
            counts={
                "reviewRows": 1,
                "needsReviewRows": 1,
                "nonNeedsReviewRows": 0,
                "humanReviewRows": 1,
                "blockedRows": 0,
                "diagnosticPageContextRows": 1,
                "unmatchedEquationQuoteRows": 0,
                "sourceSpanCreatedRows": 1,
                "originalPdfOffsetRecoveredRows": 0,
                "equationSemanticsInterpretedRows": 0,
                "strictEligibleRows": 0,
                "citationGradeRows": 0,
                "runtimeEvidenceRows": 0,
            },
        ),
    )
    validation = _write(
        tmp_path,
        "validation.json",
        {
            **_validation_report(),
            "counts": {
                "missingRows": 0,
                "invalidRows": 0,
                "sourceSpanCreatedRows": 0,
                "originalPdfOffsetRecoveredRows": 0,
                "equationSemanticsInterpretedRows": 1,
                "strictEligibleRows": 0,
                "citationGradeRows": 0,
                "runtimeEvidenceRows": 0,
            },
        },
    )

    payload = build_equation_quote_decision_record(
        equation_quote_manual_review_sheet_report=sheet,
        equation_quote_decision_file_validation_report=validation,
    )

    assert payload["status"] == "blocked"
    assert "manualReviewSheet_sourceSpanCreatedRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert "decisionValidation_equationSemanticsInterpretedRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["policy"]["databaseMutation"] is False


def test_equation_quote_decision_record_blocks_incomplete_validation_report(tmp_path: Path) -> None:
    sheet = _write(tmp_path, "manual-sheet.json", _manual_sheet([_review_row(1)]))
    validation = _write(
        tmp_path,
        "validation.json",
        {
            **_validation_report(),
            "status": "decision_file_incomplete",
            "counts": {
                "missingRows": 0,
                "invalidRows": 1,
                "sourceSpanCreatedRows": 0,
                "originalPdfOffsetRecoveredRows": 0,
                "equationSemanticsInterpretedRows": 0,
                "strictEligibleRows": 0,
                "citationGradeRows": 0,
                "runtimeEvidenceRows": 0,
            },
        },
    )
    decisions = _write(
        tmp_path,
        "decisions.json",
        {"decisions": [_decision("equation-quote-manual-review:0001")]},
    )

    payload = build_equation_quote_decision_record(
        equation_quote_manual_review_sheet_report=sheet,
        equation_quote_decision_file_validation_report=validation,
        review_decisions_report=decisions,
    )

    assert payload["status"] == "blocked"
    assert "equation_quote_decision_file_validation_not_validated" in payload["gate"]["unsafeUpstreamFlags"]
    assert "decisionValidation_invalidRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert payload["gate"]["decisionRecordReady"] is False
    assert payload["policy"]["runtimePromotionAllowed"] is False


def test_equation_quote_decision_record_writer_outputs_schema_valid_files(tmp_path: Path) -> None:
    sheet = _write(tmp_path / "input", "manual-sheet.json", _manual_sheet([_review_row(1)]))
    validation = _write(tmp_path / "input", "validation.json", _validation_report())
    decisions = _write(
        tmp_path / "input",
        "decisions.json",
        {"decisions": [_decision("equation-quote-manual-review:0001")]},
    )
    payload = build_equation_quote_decision_record(
        equation_quote_manual_review_sheet_report=sheet,
        equation_quote_decision_file_validation_report=validation,
        review_decisions_report=decisions,
    )

    paths = write_equation_quote_decision_record_reports(payload, tmp_path / "reports")

    assert set(paths) == {"record", "summary", "markdown"}
    record = json.loads(Path(paths["record"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(record, EQUATION_QUOTE_DECISION_RECORD_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["recordRows"] == 1
    assert summary["counts"]["strictEligibleRows"] == 0
    assert "This record is report-only" in markdown
