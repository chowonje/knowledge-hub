from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_quote_decision_file_validation import (
    EQUATION_QUOTE_DECISION_FILE_VALIDATION_SCHEMA_ID,
    build_equation_quote_decision_file_validation,
    write_equation_quote_decision_file_validation_reports,
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


def _decision(row_id: str, decision: str = "needs_review", reviewer: str = "", notes: str = "") -> dict:
    return {
        "source_review_sheet_row_id": row_id,
        "decision": decision,
        "reviewer": reviewer,
        "notes": notes,
    }


def test_equation_quote_decision_file_validation_accepts_needs_review_draft_as_structurally_valid(
    tmp_path: Path,
) -> None:
    sheet = _write(tmp_path, "manual-sheet.json", _manual_sheet([_review_row(1), _review_row(2)]))
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

    payload = build_equation_quote_decision_file_validation(
        equation_quote_manual_review_sheet_report=sheet,
        review_decisions_report=decisions,
    )

    assert payload["schema"] == EQUATION_QUOTE_DECISION_FILE_VALIDATION_SCHEMA_ID
    assert validate_payload(payload, EQUATION_QUOTE_DECISION_FILE_VALIDATION_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "decision_file_validated"
    assert payload["counts"]["validationRows"] == 2
    assert payload["counts"]["validRows"] == 2
    assert payload["counts"]["needsReviewRows"] == 2
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["decisionFileComplete"] is True
    assert payload["gate"]["humanReviewRecordComplete"] is False
    assert payload["policy"]["runtimePromotionAllowed"] is False


def test_equation_quote_decision_file_validation_requires_reviewer_and_notes_for_non_needs_review(
    tmp_path: Path,
) -> None:
    sheet = _write(tmp_path, "manual-sheet.json", _manual_sheet([_review_row(1)]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        {"decisions": [_decision("equation-quote-manual-review:0001", "reject_equation_quote_candidate")]},
    )

    payload = build_equation_quote_decision_file_validation(
        equation_quote_manual_review_sheet_report=sheet,
        review_decisions_report=decisions,
    )

    assert payload["status"] == "decision_file_incomplete"
    assert payload["counts"]["invalidRows"] == 1
    row = payload["validationRows"][0]
    assert "reviewer_required_for_non_needs_review_decision" in row["validation_errors"]
    assert "notes_required_for_non_needs_review_decision" in row["validation_errors"]
    assert row["strict_eligible"] is False


def test_equation_quote_decision_file_validation_keeps_reject_decision_non_runtime(
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
                    "reject_equation_quote_candidate",
                    reviewer="reviewer",
                    notes="not an alignable equation quote",
                )
            ]
        },
    )

    payload = build_equation_quote_decision_file_validation(
        equation_quote_manual_review_sheet_report=sheet,
        review_decisions_report=decisions,
    )

    assert validate_payload(payload, EQUATION_QUOTE_DECISION_FILE_VALIDATION_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "decision_file_validated"
    assert payload["counts"]["submittedRejectRows"] == 1
    assert payload["counts"]["nonNeedsReviewRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    row = payload["validationRows"][0]
    assert row["submitted_decision"] == "reject_equation_quote_candidate"
    assert row["source_span_created"] is False
    assert row["equation_semantics_interpreted"] is False
    assert row["runtime_evidence"] is False
    assert payload["policy"]["answerIntegrationChanged"] is False


def test_equation_quote_decision_file_validation_blocks_unknown_or_duplicate_rows(tmp_path: Path) -> None:
    sheet = _write(tmp_path, "manual-sheet.json", _manual_sheet([_review_row(1)]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "decisions": [
                _decision("equation-quote-manual-review:0001"),
                _decision("equation-quote-manual-review:0001"),
                _decision("unknown-row"),
            ]
        },
    )

    payload = build_equation_quote_decision_file_validation(
        equation_quote_manual_review_sheet_report=sheet,
        review_decisions_report=decisions,
    )

    assert payload["status"] == "blocked"
    assert "equation_quote_decision_duplicate_row_id" in payload["gate"]["fileValidationErrors"]
    assert "equation_quote_decision_unknown_review_row_id" in payload["gate"]["fileValidationErrors"]


def test_equation_quote_decision_file_validation_blocks_unsafe_manual_sheet(tmp_path: Path) -> None:
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

    payload = build_equation_quote_decision_file_validation(equation_quote_manual_review_sheet_report=sheet)

    assert payload["status"] == "blocked"
    assert "manualReviewSheet_sourceSpanCreatedRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]


def test_equation_quote_decision_file_validation_writer_outputs_schema_valid_files(tmp_path: Path) -> None:
    sheet = _write(tmp_path / "input", "manual-sheet.json", _manual_sheet([_review_row(1)]))
    decisions = _write(
        tmp_path / "input",
        "decisions.json",
        {"decisions": [_decision("equation-quote-manual-review:0001")]},
    )
    payload = build_equation_quote_decision_file_validation(
        equation_quote_manual_review_sheet_report=sheet,
        review_decisions_report=decisions,
    )

    paths = write_equation_quote_decision_file_validation_reports(payload, tmp_path / "reports")

    assert set(paths) == {"validation", "summary", "markdown"}
    validation = json.loads(Path(paths["validation"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(validation, EQUATION_QUOTE_DECISION_FILE_VALIDATION_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["validationRows"] == 1
    assert "This validation report is report-only" in markdown
