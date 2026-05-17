from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_quote_next_action_gate import (
    EQUATION_QUOTE_NEXT_ACTION_GATE_SCHEMA_ID,
    build_equation_quote_next_action_gate,
    write_equation_quote_next_action_gate_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _alignment_row(candidate_id: str = "equationquote:1", **overrides: object) -> dict:
    payload = {
        "audit_id": "equation-alignment-feasibility:0001",
        "candidate_id": candidate_id,
        "paper_id": "1706.03762",
        "candidate_text": "softmax(QK^T)",
        "existing_alignment_status": "failed",
        "existing_alignment_method": "none",
        "feasibility_status": "diagnostic_term_context_candidate_only",
        "diagnostic_terms": ["softmax"],
        "diagnostic_term_matches": ["softmax"],
        "diagnostic_term_coverage": 1.0,
        "layout_element_count": 1,
        "bbox_available": True,
        "sourceContentHash": "hash1",
        "source_span_created": False,
        "equation_semantics_interpreted": False,
        "strict_eligible": False,
        "citation_grade": False,
    }
    payload.update(overrides)
    return payload


def _alignment_report(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.equation-alignment-feasibility-audit.v1",
        "status": "ok",
        "counts": {
            "auditedEquationQuoteCandidates": len(rows),
            "canonicalSourceSpanCreatedCandidates": 0,
            "equationSemanticsInterpretedCandidates": 0,
            "strictEligibleCandidates": 0,
            "citationGradeCandidates": 0,
            "runtimeEvidenceCandidates": 0,
        },
        "gate": {
            "equationAlignmentFeasibilityReviewed": True,
            "sourceSpanCreationReady": False,
            "equationSemanticsReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "rows": rows,
    }
    payload.update(overrides)
    return payload


def _pdf_row(candidate_id: str = "equationquote:1", **overrides: object) -> dict:
    payload = {
        "feasibility_row_id": "equationquote-pdf-offset:1706.03762:equationquote:1",
        "source_equation_quote_candidate_id": candidate_id,
        "paper_id": "1706.03762",
        "candidate_text": "softmax(QK^T)",
        "equation_label": "eq:1",
        "feasibility_status": "diagnostic_page_context_candidate_only",
        "original_pdf_offset_recovered": False,
        "source_span_created": False,
        "equation_semantics_interpreted": False,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "diagnostic_page_candidates": [
            {"page": 3, "matchedTerms": ["softmax"], "coverage": 1.0},
        ],
    }
    payload.update(overrides)
    return payload


def _pdf_report(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.equation-quote-pdf-offset-feasibility.v1",
        "status": "feasibility_complete",
        "counts": {
            "feasibilityRows": len(rows),
            "originalPdfOffsetRecoveredRows": 0,
            "sourceSpanCreatedRows": 0,
            "equationSemanticsInterpretedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "feasibilityComplete": True,
            "sourceSpanCreationReady": False,
            "equationSemanticsReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "feasibilityRows": rows,
    }
    payload.update(overrides)
    return payload


def test_equation_quote_next_action_gate_creates_diagnostic_review_cards(tmp_path: Path) -> None:
    alignment = _write(tmp_path, "alignment.json", _alignment_report([_alignment_row()]))
    pdf = _write(tmp_path, "pdf.json", _pdf_report([_pdf_row()]))

    payload = build_equation_quote_next_action_gate(
        equation_alignment_feasibility_audit=alignment,
        equation_quote_pdf_offset_feasibility=pdf,
    )

    assert payload["schema"] == EQUATION_QUOTE_NEXT_ACTION_GATE_SCHEMA_ID
    assert validate_payload(payload, EQUATION_QUOTE_NEXT_ACTION_GATE_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "next_action_ready"
    assert payload["counts"]["nextActionCards"] == 1
    assert payload["counts"]["humanReviewCards"] == 1
    assert payload["counts"]["diagnosticPageContextCards"] == 1
    assert payload["counts"]["sourceSpanCreatedCards"] == 0
    assert payload["counts"]["strictEligibleCards"] == 0
    assert payload["gate"]["sourceSpanCreationReady"] is False
    assert payload["gate"]["equationSemanticsReady"] is False
    card = payload["actionCards"][0]
    assert card["action_type"] == "review_diagnostic_page_context"
    assert card["strict_eligible"] is False
    assert card["quote_only"] is True
    assert "equation_semantics_not_interpreted" in card["strict_blockers"]


def test_equation_quote_next_action_gate_marks_unmatched_rows_blocked_not_strict(tmp_path: Path) -> None:
    alignment = _write(
        tmp_path,
        "alignment.json",
        _alignment_report([
            _alignment_row(feasibility_status="blocked_no_canonical_equation_text_match", diagnostic_terms=[]),
        ]),
    )
    pdf = _write(
        tmp_path,
        "pdf.json",
        _pdf_report([
            _pdf_row(feasibility_status="blocked_no_match", diagnostic_page_candidates=[]),
        ]),
    )

    payload = build_equation_quote_next_action_gate(
        equation_alignment_feasibility_audit=alignment,
        equation_quote_pdf_offset_feasibility=pdf,
    )

    assert validate_payload(payload, EQUATION_QUOTE_NEXT_ACTION_GATE_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["blockedCards"] == 1
    assert payload["counts"]["unmatchedEquationQuoteCards"] == 1
    assert payload["actionCards"][0]["action_status"] == "blocked_no_source_span_or_diagnostic_context"
    assert payload["actionCards"][0]["runtime_evidence"] is False


def test_equation_quote_next_action_gate_blocks_unsafe_upstream_reports(tmp_path: Path) -> None:
    alignment = _write(
        tmp_path,
        "alignment.json",
        _alignment_report([_alignment_row()], schema="example.wrong.alignment.v1"),
    )
    pdf = _write(
        tmp_path,
        "pdf.json",
        _pdf_report(
            [_pdf_row()],
            counts={
                "feasibilityRows": 1,
                "originalPdfOffsetRecoveredRows": 1,
                "sourceSpanCreatedRows": 0,
                "equationSemanticsInterpretedRows": 0,
                "strictEligibleRows": 0,
                "citationGradeRows": 0,
                "runtimeEvidenceRows": 0,
            },
        ),
    )

    payload = build_equation_quote_next_action_gate(
        equation_alignment_feasibility_audit=alignment,
        equation_quote_pdf_offset_feasibility=pdf,
    )

    assert payload["status"] == "blocked"
    assert "equation_alignment_feasibility_audit_schema_mismatch" in payload["gate"]["schemaViolations"]
    assert "pdfOffset_originalPdfOffsetRecoveredRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]


def test_equation_quote_next_action_gate_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    alignment = _write(tmp_path / "input", "alignment.json", _alignment_report([_alignment_row()]))
    pdf = _write(tmp_path / "input", "pdf.json", _pdf_report([_pdf_row()]))
    payload = build_equation_quote_next_action_gate(
        equation_alignment_feasibility_audit=alignment,
        equation_quote_pdf_offset_feasibility=pdf,
    )

    paths = write_equation_quote_next_action_gate_reports(payload, tmp_path / "reports")

    written = json.loads(Path(paths["gate"]).read_text(encoding="utf-8"))
    assert validate_payload(written, EQUATION_QUOTE_NEXT_ACTION_GATE_SCHEMA_ID, strict=True).ok
    assert Path(paths["summary"]).exists()
    assert "does not interpret equations" in Path(paths["markdown"]).read_text(encoding="utf-8")
