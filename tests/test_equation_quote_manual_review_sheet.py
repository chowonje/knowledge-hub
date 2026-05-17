from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_quote_manual_review_sheet import (
    EQUATION_QUOTE_MANUAL_REVIEW_SHEET_SCHEMA_ID,
    build_equation_quote_manual_review_sheet,
    write_equation_quote_manual_review_sheet_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _action_card(index: int, action_type: str = "review_diagnostic_page_context") -> dict:
    status = "ready_for_human_review"
    if action_type == "reject_or_reextract_unmatched_equation_quote":
        status = "blocked_no_source_span_or_diagnostic_context"
    return {
        "action_card_id": f"equation-quote-next-action:{index:04d}",
        "source_equation_quote_candidate_id": f"equation_quote:{index:04d}",
        "paper_id": "1706.03762",
        "candidate_text": "softmax(QK^T / sqrt(d_k))V",
        "equation_label": f"eq-{index}",
        "action_type": action_type,
        "action_status": status,
        "priority": "high",
        "canonical_alignment_status": "diagnostic_only",
        "canonical_alignment_method": "diagnostic_terms",
        "alignment_feasibility_status": "diagnostic_term_context_candidate_only",
        "pdf_offset_feasibility_status": "diagnostic_page_context_candidate_only",
        "diagnostic_terms": ["softmax", "sqrt"],
        "diagnostic_term_coverage": 1.0,
        "diagnostic_page_candidates": [{"page": 4, "coverage": 1.0}],
        "best_diagnostic_page_coverage": 1.0,
        "layout_element_count": 2,
        "bbox_available": True,
        "sourceContentHash": "sha256:test",
        "source_span_created": False,
        "original_pdf_offset_recovered": False,
        "equation_semantics_interpreted": False,
        "quote_only": True,
        "strict_blockers": [
            "equation_quote_next_action_gate_only",
            "equation_alignment_missing",
            "original_pdf_offset_not_recovered",
        ],
    }


def _next_action_gate(cards: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.equation-quote-next-action-gate.v1",
        "status": "next_action_ready",
        "counts": {
            "nextActionCards": len(cards),
            "humanReviewCards": sum(1 for card in cards if card["action_status"] == "ready_for_human_review"),
            "blockedCards": sum(1 for card in cards if str(card["action_status"]).startswith("blocked")),
            "diagnosticPageContextCards": sum(1 for card in cards if card["action_type"] == "review_diagnostic_page_context"),
            "diagnosticTermContextCards": 0,
            "unmatchedEquationQuoteCards": sum(
                1 for card in cards if card["action_type"] == "reject_or_reextract_unmatched_equation_quote"
            ),
            "sourceSpanCreatedCards": 0,
            "originalPdfOffsetRecoveredCards": 0,
            "equationSemanticsInterpretedCards": 0,
            "strictEligibleCards": 0,
            "citationGradeCards": 0,
            "runtimeEvidenceCards": 0,
        },
        "gate": {
            "nextActionGateReady": True,
            "humanReviewRequired": True,
            "sourceSpanCreationReady": False,
            "equationSemanticsReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "nextActionGateOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "actionCards": cards,
    }
    payload.update(overrides)
    return payload


def test_equation_quote_manual_review_sheet_projects_action_cards_as_non_strict_rows(tmp_path: Path) -> None:
    gate = _write(
        tmp_path,
        "gate.json",
        _next_action_gate([
            _action_card(1),
            _action_card(2, "reject_or_reextract_unmatched_equation_quote"),
        ]),
    )

    payload = build_equation_quote_manual_review_sheet(equation_quote_next_action_gate_report=gate)

    assert payload["schema"] == EQUATION_QUOTE_MANUAL_REVIEW_SHEET_SCHEMA_ID
    assert validate_payload(payload, EQUATION_QUOTE_MANUAL_REVIEW_SHEET_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "manual_review_sheet_ready"
    assert payload["counts"]["reviewRows"] == 2
    assert payload["counts"]["needsReviewRows"] == 2
    assert payload["counts"]["humanReviewRows"] == 1
    assert payload["counts"]["blockedRows"] == 1
    assert payload["counts"]["diagnosticPageContextRows"] == 1
    assert payload["counts"]["unmatchedEquationQuoteRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert all(row["evidence_tier"] == "equation_quote_manual_review_sheet_only" for row in payload["reviewRows"])
    assert all(row["current_decision"] == "needs_review" for row in payload["reviewRows"])


def test_equation_quote_manual_review_sheet_does_not_promote_diagnostic_context(tmp_path: Path) -> None:
    gate = _write(tmp_path, "gate.json", _next_action_gate([_action_card(1)]))

    payload = build_equation_quote_manual_review_sheet(equation_quote_next_action_gate_report=gate)

    row = payload["reviewRows"][0]
    assert row["sourceContentHash"] == "sha256:test"
    assert row["bbox_available"] is True
    assert row["source_span_created"] is False
    assert row["original_pdf_offset_recovered"] is False
    assert row["equation_semantics_interpreted"] is False
    assert row["quote_only"] is True
    assert row["strict_eligible"] is False
    assert row["citation_grade"] is False
    assert row["runtime_evidence"] is False
    assert "diagnostic_context_does_not_create_source_spans" in row["non_strict_reason"]


def test_equation_quote_manual_review_sheet_blocks_unsafe_next_action_gate(tmp_path: Path) -> None:
    unsafe = _next_action_gate(
        [_action_card(1)],
        counts={
            "nextActionCards": 1,
            "humanReviewCards": 1,
            "blockedCards": 0,
            "diagnosticPageContextCards": 1,
            "diagnosticTermContextCards": 0,
            "unmatchedEquationQuoteCards": 0,
            "sourceSpanCreatedCards": 1,
            "originalPdfOffsetRecoveredCards": 0,
            "equationSemanticsInterpretedCards": 0,
            "strictEligibleCards": 0,
            "citationGradeCards": 0,
            "runtimeEvidenceCards": 0,
        },
    )
    gate = _write(tmp_path, "gate.json", unsafe)

    payload = build_equation_quote_manual_review_sheet(equation_quote_next_action_gate_report=gate)

    assert payload["status"] == "blocked"
    assert "nextActionGate_sourceSpanCreatedCards_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert payload["policy"]["runtimePromotionAllowed"] is False


def test_equation_quote_manual_review_sheet_writer_outputs_schema_valid_files(tmp_path: Path) -> None:
    gate = _write(tmp_path / "input", "gate.json", _next_action_gate([_action_card(1)]))
    payload = build_equation_quote_manual_review_sheet(equation_quote_next_action_gate_report=gate)

    paths = write_equation_quote_manual_review_sheet_reports(payload, tmp_path / "reports")

    assert set(paths) == {"sheet", "summary", "markdown"}
    sheet = json.loads(Path(paths["sheet"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(sheet, EQUATION_QUOTE_MANUAL_REVIEW_SHEET_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["reviewRows"] == 1
    assert "This sheet is local review metadata only" in markdown
