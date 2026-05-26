from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_quote_decision_file_draft import (
    EQUATION_QUOTE_DECISION_FILE_DRAFT_SCHEMA_ID,
    build_equation_quote_decision_file_draft,
    write_equation_quote_decision_file_draft_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _review_row(index: int, action_type: str = "review_diagnostic_page_context") -> dict:
    status = "ready_for_human_review"
    if action_type == "reject_or_reextract_unmatched_equation_quote":
        status = "blocked_no_source_span_or_diagnostic_context"
    return {
        "review_sheet_row_id": f"equation-quote-manual-review:{index:04d}",
        "source_action_card_id": f"equation-quote-next-action:{index:04d}",
        "source_equation_quote_candidate_id": f"equation_quote:{index:04d}",
        "paper_id": "1706.03762",
        "candidate_text": "softmax(QK^T / sqrt(d_k))V",
        "equation_label": f"eq-{index}",
        "action_type": action_type,
        "action_status": status,
        "priority": "high",
        "current_decision": "needs_review",
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
            "humanReviewRows": sum(1 for row in rows if row["action_status"] == "ready_for_human_review"),
            "blockedRows": sum(1 for row in rows if str(row["action_status"]).startswith("blocked")),
            "diagnosticPageContextRows": sum(1 for row in rows if row["action_type"] == "review_diagnostic_page_context"),
            "unmatchedEquationQuoteRows": sum(
                1 for row in rows if row["action_type"] == "reject_or_reextract_unmatched_equation_quote"
            ),
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


def test_equation_quote_decision_file_draft_defaults_all_rows_to_needs_review(tmp_path: Path) -> None:
    sheet = _write(
        tmp_path,
        "manual-sheet.json",
        _manual_sheet([
            _review_row(1),
            _review_row(2, "reject_or_reextract_unmatched_equation_quote"),
        ]),
    )

    payload = build_equation_quote_decision_file_draft(equation_quote_manual_review_sheet_report=sheet)

    assert payload["schema"] == EQUATION_QUOTE_DECISION_FILE_DRAFT_SCHEMA_ID
    assert validate_payload(payload, EQUATION_QUOTE_DECISION_FILE_DRAFT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "decision_file_draft_ready"
    assert payload["counts"]["draftRows"] == 2
    assert payload["counts"]["needsReviewRows"] == 2
    assert payload["counts"]["rejectDecisionRows"] == 0
    assert payload["counts"]["reextractRequestRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["containsOnlyNeedsReviewDefaults"] is True
    assert payload["gate"]["humanReviewComplete"] is False
    assert all(row["decision"] == "needs_review" for row in payload["draftRows"])
    assert all(decision["decision"] == "needs_review" for decision in payload["decisionFileDraft"]["decisions"])


def test_equation_quote_decision_file_draft_does_not_record_decisions_or_runtime_effects(tmp_path: Path) -> None:
    sheet = _write(tmp_path, "manual-sheet.json", _manual_sheet([_review_row(1)]))

    payload = build_equation_quote_decision_file_draft(equation_quote_manual_review_sheet_report=sheet)

    row = payload["draftRows"][0]
    assert row["draft_only"] is True
    assert row["strict_eligible"] is False
    assert row["citation_grade"] is False
    assert row["runtime_evidence"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["answerIntegrationChanged"] is False
    assert "draft_rows_are_not_human_review_decisions" in row["non_strict_reason"]


def test_equation_quote_decision_file_draft_blocks_unsafe_manual_sheet(tmp_path: Path) -> None:
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

    payload = build_equation_quote_decision_file_draft(equation_quote_manual_review_sheet_report=sheet)

    assert payload["status"] == "blocked"
    assert "manualReviewSheet_sourceSpanCreatedRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert payload["policy"]["runtimePromotionAllowed"] is False


def test_equation_quote_decision_file_draft_writer_outputs_schema_valid_files(tmp_path: Path) -> None:
    sheet = _write(tmp_path / "input", "manual-sheet.json", _manual_sheet([_review_row(1)]))
    payload = build_equation_quote_decision_file_draft(equation_quote_manual_review_sheet_report=sheet)

    paths = write_equation_quote_decision_file_draft_reports(payload, tmp_path / "reports")

    assert set(paths) == {"draftReport", "decisionFileDraft", "summary", "markdown"}
    draft_report = json.loads(Path(paths["draftReport"]).read_text(encoding="utf-8"))
    decision_file = json.loads(Path(paths["decisionFileDraft"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(draft_report, EQUATION_QUOTE_DECISION_FILE_DRAFT_SCHEMA_ID, strict=True).ok
    assert decision_file["draftOnly"] is True
    assert decision_file["decisions"][0]["decision"] == "needs_review"
    assert summary["counts"]["draftRows"] == 1
    assert "This draft is an editable starting point only" in markdown
