from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_quote_decision_edit_plan import (
    EQUATION_QUOTE_DECISION_EDIT_PLAN_SCHEMA_ID,
    build_equation_quote_decision_edit_plan,
    write_equation_quote_decision_edit_plan_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _decision_file(rows: list[dict]) -> dict:
    return {
        "draftOnly": True,
        "instructions": ["manual review required"],
        "decisions": rows,
    }


def _decision_row(index: int, **overrides: object) -> dict:
    payload = {
        "source_review_sheet_row_id": f"equation-quote-manual-review:{index:04d}",
        "source_action_card_id": f"equation-quote-next-action:{index:04d}",
        "source_equation_quote_candidate_id": f"equationquote:1706.03762:{index:04d}",
        "paper_id": "1706.03762",
        "candidate_text": "softmax(QK^T / sqrt(d_k))V",
        "action_type": "review_diagnostic_page_context",
        "decision": "needs_review",
        "reviewer": "",
        "notes": "",
        "allowed_decisions": [
            "needs_review",
            "accept_diagnostic_context_for_later_reextract_design",
            "reject_equation_quote_candidate",
            "request_equation_quote_reextraction",
            "keep_blocked",
        ],
    }
    payload.update(overrides)
    return payload


def _recommendation_pack(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.equation-quote-decision-recommendation-pack.v1",
        "status": "recommendation_pack_ready",
        "counts": {
            "recommendationRows": len(rows),
            "proposedAcceptDiagnosticContextRows": sum(
                1
                for row in rows
                if row["recommended_decision"] == "accept_diagnostic_context_for_later_reextract_design"
            ),
            "proposedRejectRows": sum(1 for row in rows if row["recommended_decision"] == "reject_equation_quote_candidate"),
            "proposedReextractRequestRows": sum(
                1 for row in rows if row["recommended_decision"] == "request_equation_quote_reextraction"
            ),
            "proposedKeepBlockedRows": sum(1 for row in rows if row["recommended_decision"] == "keep_blocked"),
            "proposedNeedsReviewRows": sum(1 for row in rows if row["recommended_decision"] == "needs_review"),
            "acceptedHumanDecisionRows": 0,
            "sourceSpanCreatedRows": 0,
            "originalPdfOffsetRecoveredRows": 0,
            "equationSemanticsInterpretedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
            "unsafeUpstreamFlagCount": 0,
            "byPaper": {"1706.03762": len(rows)},
            "byActionType": {"review_diagnostic_page_context": len(rows)},
            "byActionStatus": {"ready_for_human_review": len(rows)},
            "byRecommendedDecision": {},
            "byRecommendationConfidence": {"medium": len(rows)},
        },
        "gate": {
            "recommendationPackReady": True,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "manual_decision_file_still_required",
            "unsafeUpstreamFlags": [],
            "recommendedNextTranche": "manual_edit_equation_quote_decision_file",
        },
        "policy": {
            "reportOnly": True,
            "decisionRecommendationOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [],
        "recommendationRows": rows,
    }
    payload.update(overrides)
    return payload


def _recommendation_row(index: int, decision: str, **overrides: object) -> dict:
    payload = {
        "recommendation_row_id": f"equation-quote-decision-recommendation:{index:04d}",
        "source_brief_row_id": f"equation-quote-decision-next-action:{index:04d}",
        "source_review_sheet_row_id": f"equation-quote-manual-review:{index:04d}",
        "source_action_card_id": f"equation-quote-next-action:{index:04d}",
        "source_equation_quote_candidate_id": f"equationquote:1706.03762:{index:04d}",
        "paper_id": "1706.03762",
        "candidate_text": "softmax(QK^T / sqrt(d_k))V",
        "equation_label": f"eq-{index}",
        "action_type": "review_diagnostic_page_context",
        "action_status": "ready_for_human_review",
        "priority": "high",
        "current_decision": "needs_review",
        "recommended_decision": decision,
        "recommended_decision_reason": "diagnostic_page_context_is_available",
        "recommendation_confidence": "medium",
        "diagnostic_terms": ["softmax"],
        "diagnostic_page_candidates": [{"page": 3, "coverage": 1.0}],
        "best_diagnostic_page_coverage": 1.0,
        "canonical_alignment_status": "failed",
        "alignment_feasibility_status": "diagnostic_term_context_candidate_only",
        "pdf_offset_feasibility_status": "diagnostic_page_context_candidate_only",
        "layout_element_count": 1,
        "bbox_available": True,
        "sourceContentHash": "hash",
        "human_decision_required": True,
        "accepted_as_human_decision": False,
        "decision_record_input_hint": {},
        "source_span_created": False,
        "original_pdf_offset_recovered": False,
        "equation_semantics_interpreted": False,
        "evidence_tier": "equation_quote_decision_recommendation_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": ["recommendation_only"],
        "non_strict_reason": ["not_a_human_decision"],
    }
    payload.update(overrides)
    return payload


def test_equation_quote_decision_edit_plan_projects_recommendations_as_manual_hints_only(
    tmp_path: Path,
) -> None:
    decisions = _write(tmp_path, "decisions.json", _decision_file([_decision_row(1), _decision_row(2)]))
    recommendations = _write(
        tmp_path,
        "recommendations.json",
        _recommendation_pack([
            _recommendation_row(1, "accept_diagnostic_context_for_later_reextract_design"),
            _recommendation_row(2, "request_equation_quote_reextraction"),
        ]),
    )

    payload = build_equation_quote_decision_edit_plan(
        equation_quote_decision_recommendation_pack_report=recommendations,
        equation_quote_decisions_file=decisions,
    )

    assert payload["schema"] == EQUATION_QUOTE_DECISION_EDIT_PLAN_SCHEMA_ID
    assert validate_payload(payload, EQUATION_QUOTE_DECISION_EDIT_PLAN_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "edit_plan_ready"
    assert payload["counts"]["editRows"] == 2
    assert payload["counts"]["readyForManualEditRows"] == 2
    assert payload["counts"]["currentNeedsReviewRows"] == 2
    assert payload["counts"]["proposedAcceptDiagnosticContextRows"] == 1
    assert payload["counts"]["proposedReextractRequestRows"] == 1
    assert payload["counts"]["acceptedHumanDecisionRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["manualDecisionFileEditRequired"] is True
    assert payload["gate"]["decisionFileModified"] is False
    assert payload["policy"]["databaseMutation"] is False
    assert "decisions" not in payload
    row = payload["editRows"][0]
    assert row["edit_status"] == "ready_for_manual_edit"
    assert row["accepted_as_human_decision"] is False
    assert row["decision_file_patch_hint"]["reviewer"] == ""
    assert "edit_plan_rows_do_not_modify_the_decision_file" in row["non_strict_reason"]


def test_equation_quote_decision_edit_plan_blocks_missing_or_disallowed_rows(tmp_path: Path) -> None:
    decisions = _write(
        tmp_path,
        "decisions.json",
        _decision_file([
            _decision_row(1, allowed_decisions=["needs_review", "reject_equation_quote_candidate"]),
        ]),
    )
    recommendations = _write(
        tmp_path,
        "recommendations.json",
        _recommendation_pack([
            _recommendation_row(1, "accept_diagnostic_context_for_later_reextract_design"),
            _recommendation_row(2, "request_equation_quote_reextraction"),
        ]),
    )

    payload = build_equation_quote_decision_edit_plan(
        equation_quote_decision_recommendation_pack_report=recommendations,
        equation_quote_decisions_file=decisions,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["blockedRecommendationNotAllowedRows"] == 1
    assert payload["counts"]["blockedMissingDecisionFileRows"] == 1
    assert payload["gate"]["editPlanReady"] is False


def test_equation_quote_decision_edit_plan_blocks_unsafe_recommendation_pack(tmp_path: Path) -> None:
    decisions = _write(tmp_path, "decisions.json", _decision_file([_decision_row(1)]))
    recommendations = _write(
        tmp_path,
        "recommendations.json",
        _recommendation_pack(
            [_recommendation_row(1, "accept_diagnostic_context_for_later_reextract_design")],
            counts={
                "recommendationRows": 1,
                "proposedAcceptDiagnosticContextRows": 1,
                "proposedRejectRows": 0,
                "proposedReextractRequestRows": 0,
                "proposedKeepBlockedRows": 0,
                "proposedNeedsReviewRows": 0,
                "acceptedHumanDecisionRows": 1,
                "sourceSpanCreatedRows": 0,
                "originalPdfOffsetRecoveredRows": 0,
                "equationSemanticsInterpretedRows": 0,
                "strictEligibleRows": 0,
                "citationGradeRows": 0,
                "runtimeEvidenceRows": 0,
                "unsafeUpstreamFlagCount": 0,
                "byPaper": {"1706.03762": 1},
                "byActionType": {"review_diagnostic_page_context": 1},
                "byActionStatus": {"ready_for_human_review": 1},
                "byRecommendedDecision": {"accept_diagnostic_context_for_later_reextract_design": 1},
                "byRecommendationConfidence": {"medium": 1},
            },
        ),
    )

    payload = build_equation_quote_decision_edit_plan(
        equation_quote_decision_recommendation_pack_report=recommendations,
        equation_quote_decisions_file=decisions,
    )

    assert payload["status"] == "blocked"
    assert "recommendationPack_acceptedHumanDecisionRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]


def test_equation_quote_decision_edit_plan_writer_outputs_schema_valid_files(tmp_path: Path) -> None:
    decisions = _write(tmp_path / "input", "decisions.json", _decision_file([_decision_row(1)]))
    recommendations = _write(
        tmp_path / "input",
        "recommendations.json",
        _recommendation_pack([
            _recommendation_row(1, "accept_diagnostic_context_for_later_reextract_design"),
        ]),
    )
    payload = build_equation_quote_decision_edit_plan(
        equation_quote_decision_recommendation_pack_report=recommendations,
        equation_quote_decisions_file=decisions,
    )

    paths = write_equation_quote_decision_edit_plan_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, EQUATION_QUOTE_DECISION_EDIT_PLAN_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["readyForManualEditRows"] == 1
    assert "This edit plan is not a decision file" in markdown
