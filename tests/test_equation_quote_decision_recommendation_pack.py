from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_quote_decision_recommendation_pack import (
    EQUATION_QUOTE_DECISION_RECOMMENDATION_PACK_SCHEMA_ID,
    build_equation_quote_decision_recommendation_pack,
    write_equation_quote_decision_recommendation_pack_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _brief_row(index: int, action_type: str = "review_diagnostic_page_context", **overrides: object) -> dict:
    payload = {
        "brief_row_id": f"equation-quote-decision-next-action:{index:04d}",
        "source_review_sheet_row_id": f"equation-quote-manual-review:{index:04d}",
        "source_action_card_id": f"equation-quote-next-action:{index:04d}",
        "source_equation_quote_candidate_id": f"equationquote:1706.03762:{index:04d}",
        "paper_id": "1706.03762",
        "candidate_text": "softmax(QK^T / sqrt(d_k))V",
        "equation_label": f"eq-{index}",
        "action_type": action_type,
        "action_status": "ready_for_human_review",
        "priority": "high",
        "current_decision": "needs_review",
        "diagnostic_terms": ["softmax", "sqrt"],
        "diagnostic_page_candidates": [{"page": 3, "coverage": 1.0}],
        "best_diagnostic_page_coverage": 1.0,
        "canonical_alignment_status": "failed",
        "alignment_feasibility_status": "diagnostic_term_context_candidate_only",
        "pdf_offset_feasibility_status": "diagnostic_page_context_candidate_only",
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
    payload.update(overrides)
    return payload


def _brief(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.equation-quote-decision-next-action-brief.v1",
        "status": "manual_review_required",
        "counts": {
            "briefRows": len(rows),
            "needsReviewRows": len(rows),
            "sourceSpanCreatedRows": 0,
            "originalPdfOffsetRecoveredRows": 0,
            "equationSemanticsInterpretedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "manualReviewRequired": True,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "nextActionBriefOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "briefRows": rows,
    }
    payload.update(overrides)
    return payload


def test_recommendation_pack_accepts_diagnostic_context_as_nonbinding_later_reextract_design(
    tmp_path: Path,
) -> None:
    brief = _write(tmp_path, "brief.json", _brief([_brief_row(1)]))

    payload = build_equation_quote_decision_recommendation_pack(
        equation_quote_decision_next_action_brief_report=brief
    )

    assert payload["schema"] == EQUATION_QUOTE_DECISION_RECOMMENDATION_PACK_SCHEMA_ID
    assert validate_payload(payload, EQUATION_QUOTE_DECISION_RECOMMENDATION_PACK_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "recommendation_pack_ready"
    assert payload["counts"]["recommendationRows"] == 1
    assert payload["counts"]["proposedAcceptDiagnosticContextRows"] == 1
    assert payload["counts"]["acceptedHumanDecisionRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["humanReviewComplete"] is False
    row = payload["recommendationRows"][0]
    assert row["recommended_decision"] == "accept_diagnostic_context_for_later_reextract_design"
    assert row["accepted_as_human_decision"] is False
    assert row["human_decision_required"] is True
    assert row["source_span_created"] is False
    assert row["equation_semantics_interpreted"] is False
    assert "recommendation_rows_are_not_human_review_decisions" in row["non_strict_reason"]
    assert "recommendationRows" in payload
    assert "decisionRows" not in payload
    assert "decisions" not in payload


def test_recommendation_pack_requests_reextract_for_unmatched_layout_signal(tmp_path: Path) -> None:
    brief = _write(
        tmp_path,
        "brief.json",
        _brief([
            _brief_row(
                1,
                "reject_or_reextract_unmatched_equation_quote",
                action_status="blocked_no_source_span_or_diagnostic_context",
                best_diagnostic_page_coverage=0.0,
                diagnostic_page_candidates=[],
                layout_element_count=1,
                bbox_available=True,
            )
        ]),
    )

    payload = build_equation_quote_decision_recommendation_pack(
        equation_quote_decision_next_action_brief_report=brief
    )

    row = payload["recommendationRows"][0]
    assert validate_payload(payload, EQUATION_QUOTE_DECISION_RECOMMENDATION_PACK_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["proposedReextractRequestRows"] == 1
    assert row["recommended_decision"] == "request_equation_quote_reextraction"
    assert row["recommendation_confidence"] == "low"
    assert row["runtime_evidence"] is False


def test_recommendation_pack_keeps_unknown_or_low_context_rows_as_needs_review(tmp_path: Path) -> None:
    brief = _write(
        tmp_path,
        "brief.json",
        _brief([
            _brief_row(
                1,
                action_type="review_diagnostic_page_context",
                best_diagnostic_page_coverage=0.1,
                diagnostic_page_candidates=[],
            )
        ]),
    )

    payload = build_equation_quote_decision_recommendation_pack(
        equation_quote_decision_next_action_brief_report=brief
    )

    row = payload["recommendationRows"][0]
    assert validate_payload(payload, EQUATION_QUOTE_DECISION_RECOMMENDATION_PACK_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["proposedNeedsReviewRows"] == 1
    assert row["recommended_decision"] == "needs_review"
    assert row["accepted_as_human_decision"] is False


def test_recommendation_pack_blocks_unsafe_next_action_brief(tmp_path: Path) -> None:
    brief = _write(
        tmp_path,
        "brief.json",
        _brief(
            [_brief_row(1)],
            schema="example.wrong.brief.v1",
            policy={
                "reportOnly": True,
                "nextActionBriefOnly": True,
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

    payload = build_equation_quote_decision_recommendation_pack(
        equation_quote_decision_next_action_brief_report=brief
    )

    assert payload["status"] == "blocked"
    assert "equation_quote_decision_next_action_brief_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "decisionNextActionBrief_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_recommendation_pack_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    brief = _write(tmp_path / "input", "brief.json", _brief([_brief_row(1)]))
    payload = build_equation_quote_decision_recommendation_pack(
        equation_quote_decision_next_action_brief_report=brief
    )

    paths = write_equation_quote_decision_recommendation_pack_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, EQUATION_QUOTE_DECISION_RECOMMENDATION_PACK_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["proposedAcceptDiagnosticContextRows"] == 1
    assert "This recommendation pack is not a human decision file" in markdown
