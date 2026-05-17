from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.candidate_layer_blocker_backlog import (
    CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID,
    build_candidate_layer_blocker_backlog,
    write_candidate_layer_blocker_backlog_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _gate_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-review-gate.v1",
        "status": "ready_for_candidate_layer_review",
        "counts": {
            "totalCandidates": 86,
            "questionCount": 20,
            "strictEligibleCandidates": 0,
            "citationGradeCandidates": 0,
            "runtimeEvidenceCandidates": 0,
            "currentRuntimeAnswerableQuestions": 0,
        },
        "gate": {
            "candidateLayerReviewReady": True,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "blockers": [
                "equation_quote_alignment_missing",
                "table_cell_row_column_bbox_provenance_missing",
                "figure_region_link_unverified",
                "generated_markdown_offsets_are_not_original_pdf_offsets",
                "candidate_layers_are_report_only",
                "runtime_promotion_disabled_for_tranche",
            ],
        },
    }
    payload.update(overrides)
    return payload


def _summary_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.structured-candidate-summary.v1",
        "counts": {
            "totalCandidates": 86,
            "byLayer": {
                "sectionspan": 61,
                "figure_caption": 11,
                "equation_quote": 9,
                "table_region": 5,
            },
            "strictEligibleCandidates": 0,
            "citationGradeCandidates": 0,
            "runtimeEvidenceCandidates": 0,
        },
        "releaseCandidateAssessment": {
            "candidateLayerReviewReady": True,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "mainBlockers": [
                "equation_quote_alignment_missing",
                "table_cell_row_column_bbox_provenance_missing",
                "figure_region_link_unverified",
            ],
        },
    }
    payload.update(overrides)
    return payload


def _source_aligned_summary_payload() -> dict:
    return _summary_payload(
        counts={
            "totalCandidates": 86,
            "byLayer": {
                "sectionspan": 61,
                "figure_caption": 11,
                "equation_quote": 9,
                "table_region": 5,
            },
            "strictEligibleCandidates": 0,
            "citationGradeCandidates": 0,
            "runtimeEvidenceCandidates": 0,
            "sectionspanOriginalPdfOffsetReadyForReviewRows": 61,
        },
        releaseCandidateAssessment={
            "candidateLayerReviewReady": True,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "mainBlockers": [
                "equation_quote_alignment_missing",
                "table_cell_row_column_bbox_provenance_missing",
                "figure_region_link_unverified",
                "sectionspan_pdf_offsets_require_human_review_before_strict_promotion",
                "non_sectionspan_layers_lack_original_pdf_offsets",
            ],
        },
    )


def _eval_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.complex-qa-eval-design.v1",
        "counts": {
            "questionCount": 20,
            "currentRuntimeAnswerableQuestions": 0,
            "executedQuestions": 0,
            "strictEvidenceCreated": 0,
        },
        "questions": [
            {
                "question_id": "q1",
                "target_candidate_layers": ["equation_quote"],
                "blocked_by_current_candidates": ["equation_quote_alignment_missing"],
            },
            {
                "question_id": "q2",
                "target_candidate_layers": ["table_region"],
                "blocked_by_current_candidates": ["table_cell_row_column_bbox_provenance_missing"],
            },
            {
                "question_id": "q3",
                "target_candidate_layers": ["figure_caption"],
                "blocked_by_current_candidates": ["figure_region_link_unverified"],
            },
        ],
    }
    payload.update(overrides)
    return payload


def _table_cell_result_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.table-cell-isolated-extractor-pilot-result.v1",
        "status": "approval_required",
        "counts": {
            "targetRows": 2,
            "probeAttemptedRows": 0,
            "approvalRequiredRows": 2,
            "blockedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "pilotExecuted": False,
            "approvalRequiredBeforeInstallOrRun": True,
            "extractorAvailable": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
        },
    }
    payload.update(overrides)
    return payload


def _sectionspan_human_review_gate_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-human-review-gate.v1",
        "status": "review_required",
        "counts": {
            "gateRows": 61,
            "pendingHumanReviewRows": 61,
            "approvedForLaterPromotionDesignRows": 0,
            "rejectedRows": 0,
            "heldOutRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "humanReviewGateReady": True,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
    }
    payload.update(overrides)
    return payload


def _sectionspan_selected_decision_proposal_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-proposal.v1",
        "status": "decision_proposal_ready",
        "counts": {
            "proposalRows": 12,
            "proposedApproveForLaterPromotionDesignRows": 12,
            "proposedNeedsReviewRows": 0,
            "acceptedHumanDecisionRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "decisionProposalReady": True,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "decisionProposalOnly": True,
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


def _sectionspan_selected_next_action_brief_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-next-action-brief.v1",
        "status": "manual_review_required",
        "counts": {
            "briefRows": 12,
            "needsReviewRows": 12,
            "nonNeedsReviewRows": 0,
            "suggestedApproveForLaterPromotionDesignRows": 12,
            "suggestedNeedsReviewRows": 0,
            "validationValidRows": 12,
            "validationInvalidRows": 0,
            "validationMissingRows": 0,
            "decisionRecordNeedsReviewRows": 12,
            "decisionRecordApprovedForLaterPromotionDesignRows": 0,
            "decisionRecordRejectedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "nextActionBriefReady": True,
            "manualReviewRequired": True,
            "autoApprovalAllowed": False,
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
    }
    payload.update(overrides)
    return payload


def _non_sectionspan_pdf_offset_audit_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.non-sectionspan-pdf-offset-feasibility-audit.v1",
        "status": "ok",
        "counts": {
            "totalRows": 25,
            "recoveredRows": 13,
            "blockedRows": 12,
            "diagnosticPageContextRows": 8,
            "readyForRegionReviewRows": 13,
            "needsFigureRegionReviewRows": 9,
            "needsTableCellProvenanceReviewRows": 4,
            "needsEquationAlignmentReviewRows": 9,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "auditComplete": True,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "allCandidatesNonStrict": True,
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


def _equation_alignment_audit_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.equation-alignment-feasibility-audit.v1",
        "status": "ok",
        "counts": {
            "auditedEquationQuoteCandidates": 9,
            "canonicalSourceSpanCreatedCandidates": 0,
            "diagnosticTermContextCandidates": 8,
            "strictEligibleCandidates": 0,
            "citationGradeCandidates": 0,
            "runtimeEvidenceCandidates": 0,
            "schemaViolationCount": 0,
        },
        "gate": {
            "equationAlignmentFeasibilityReviewed": True,
            "sourceSpanCreationReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
        },
    }
    payload.update(overrides)
    return payload


def _table_cell_provenance_audit_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.table-cell-provenance-feasibility-audit.v1",
        "status": "ok",
        "counts": {
            "auditedTableRegionCandidates": 5,
            "totalTableCells": 681,
            "cellSourceSpanCandidates": 0,
            "tableCellCitationGradeCandidates": 0,
            "strictEligibleCandidates": 0,
            "citationGradeCandidates": 0,
            "runtimeEvidenceCandidates": 0,
            "schemaViolationCount": 0,
        },
        "gate": {
            "tableCellProvenanceReviewed": True,
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
        },
    }
    payload.update(overrides)
    return payload


def _figure_region_link_audit_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.figure-region-link-feasibility-audit.v1",
        "status": "ok",
        "counts": {
            "auditedFigureCaptionCandidates": 11,
            "captionSourceSpanCandidates": 9,
            "figureRegionLinkVerifiedCandidates": 0,
            "strictEligibleCandidates": 0,
            "citationGradeCandidates": 0,
            "runtimeEvidenceCandidates": 0,
            "schemaViolationCount": 0,
        },
        "gate": {
            "figureRegionLinkReviewed": True,
            "figureRegionCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
        },
    }
    payload.update(overrides)
    return payload


def _candidate_layer_blocker_decision_record_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-decision-record.v1",
        "status": "decision_record_required",
        "counts": {
            "recordRows": 12,
            "needsReviewRows": 12,
            "manualApprovalRows": 0,
            "manualRejectionRows": 0,
            "operatorApprovedRows": 0,
            "operatorDeclinedRows": 0,
            "technicalAcceptedOpenRows": 0,
            "technicalDeferredRows": 0,
            "policyAcceptedGuardrailRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
            "unsafeUpstreamFlagCount": 0,
        },
        "gate": {
            "decisionRecordReady": True,
            "allDecisionRowsComplete": False,
            "humanReviewComplete": False,
            "operatorApprovalComplete": False,
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
    }
    payload.update(overrides)
    return payload


def _reports(root: Path, *, gate: dict | None = None, summary: dict | None = None, eval_design: dict | None = None) -> tuple[Path, Path, Path]:
    gate_path = _write(root, "candidate-layer-review-gate.json", gate or _gate_payload())
    summary_path = _write(root, "structured-candidate-summary.json", summary or _summary_payload())
    eval_path = _write(root, "complex-paper-qa-eval-design.json", eval_design or _eval_payload())
    return gate_path, summary_path, eval_path


def test_candidate_layer_blocker_backlog_classifies_open_blockers_and_validates_schema(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path)

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
    )

    assert payload["schema"] == CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID
    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["backlogItemCount"] == 6
    assert payload["counts"]["byPriority"]["P0"] == 3
    assert payload["counts"]["byLayer"]["equation_quote"] >= 1
    equation = next(item for item in payload["backlog"] if item["blocker"] == "equation_quote_alignment_missing")
    assert equation["priority"] == "P0"
    assert equation["affected_layers"] == ["equation_quote"]
    assert equation["affected_candidate_count"] == 9
    assert equation["affected_eval_question_count"] == 1


def test_candidate_layer_blocker_backlog_classifies_source_aligned_sectionspan_blockers(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(
        tmp_path,
        gate=_gate_payload(
            gate={
                "candidateLayerReviewReady": True,
                "strictEvidenceReady": False,
                "parserRoutingReady": False,
                "answerIntegrationReady": False,
                "blockers": [
                    "sectionspan_pdf_offsets_require_human_review_before_strict_promotion",
                    "non_sectionspan_layers_lack_original_pdf_offsets",
                    "candidate_layers_are_report_only",
                    "runtime_promotion_disabled_for_tranche",
                ],
            }
        ),
        summary=_source_aligned_summary_payload(),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    sectionspan = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "sectionspan_pdf_offsets_require_human_review_before_strict_promotion"
    )
    non_sectionspan = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "non_sectionspan_layers_lack_original_pdf_offsets"
    )
    assert sectionspan["priority"] == "P0"
    assert sectionspan["affected_layers"] == ["sectionspan"]
    assert sectionspan["affected_candidate_count"] == 61
    assert sectionspan["recommendedNextTranche"] == "sectionspan_pdf_offset_human_review_gate"
    assert non_sectionspan["priority"] == "P0"
    assert non_sectionspan["affected_layers"] == ["figure_caption", "equation_quote", "table_region"]
    assert non_sectionspan["affected_candidate_count"] == 25
    assert non_sectionspan["recommendedNextTranche"] == "non_sectionspan_original_pdf_offset_feasibility_audit"


def test_candidate_layer_blocker_backlog_uses_figure_caption_pdf_offset_supplement_counts(tmp_path: Path) -> None:
    summary = _source_aligned_summary_payload()
    summary["counts"] = {
        **summary["counts"],
        "figureCaptionOriginalPdfOffsetFeasibilityRows": 11,
        "figureCaptionOriginalPdfOffsetRecoveredRows": 9,
        "figureCaptionOriginalPdfOffsetBlockedRows": 2,
    }
    summary["releaseCandidateAssessment"] = {
        **summary["releaseCandidateAssessment"],
        "mainBlockers": [
            *summary["releaseCandidateAssessment"]["mainBlockers"],
            "figure_caption_pdf_offsets_require_region_link_review",
        ],
    }
    gate_path, summary_path, eval_path = _reports(
        tmp_path,
        gate=_gate_payload(
            gate={
                "candidateLayerReviewReady": True,
                "strictEvidenceReady": False,
                "parserRoutingReady": False,
                "answerIntegrationReady": False,
                "blockers": [
                    "non_sectionspan_layers_lack_original_pdf_offsets",
                    "figure_caption_pdf_offsets_require_region_link_review",
                    "candidate_layers_are_report_only",
                    "runtime_promotion_disabled_for_tranche",
                ],
            }
        ),
        summary=summary,
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    non_sectionspan = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "non_sectionspan_layers_lack_original_pdf_offsets"
    )
    figure_region_review = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "figure_caption_pdf_offsets_require_region_link_review"
    )
    assert non_sectionspan["affected_candidate_count"] == 16
    assert figure_region_review["priority"] == "P1"
    assert figure_region_review["affected_layers"] == ["figure_caption"]
    assert figure_region_review["affected_candidate_count"] == 9
    assert figure_region_review["recommendedNextTranche"] == "figure_caption_region_link_review_pack"


def test_candidate_layer_blocker_backlog_uses_table_region_pdf_offset_supplement_counts(tmp_path: Path) -> None:
    summary = _source_aligned_summary_payload()
    summary["counts"] = {
        **summary["counts"],
        "figureCaptionOriginalPdfOffsetFeasibilityRows": 11,
        "figureCaptionOriginalPdfOffsetRecoveredRows": 9,
        "figureCaptionOriginalPdfOffsetBlockedRows": 2,
        "tableRegionOriginalPdfOffsetFeasibilityRows": 5,
        "tableRegionOriginalPdfOffsetRecoveredRows": 4,
        "tableRegionOriginalPdfOffsetBlockedRows": 1,
    }
    summary["releaseCandidateAssessment"] = {
        **summary["releaseCandidateAssessment"],
        "mainBlockers": [
            *summary["releaseCandidateAssessment"]["mainBlockers"],
            "figure_caption_pdf_offsets_require_region_link_review",
            "table_caption_pdf_offsets_require_cell_provenance_review",
        ],
    }
    gate_path, summary_path, eval_path = _reports(
        tmp_path,
        gate=_gate_payload(
            gate={
                "candidateLayerReviewReady": True,
                "strictEvidenceReady": False,
                "parserRoutingReady": False,
                "answerIntegrationReady": False,
                "blockers": [
                    "non_sectionspan_layers_lack_original_pdf_offsets",
                    "table_caption_pdf_offsets_require_cell_provenance_review",
                    "candidate_layers_are_report_only",
                    "runtime_promotion_disabled_for_tranche",
                ],
            }
        ),
        summary=summary,
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    non_sectionspan = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "non_sectionspan_layers_lack_original_pdf_offsets"
    )
    table_review = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "table_caption_pdf_offsets_require_cell_provenance_review"
    )
    assert non_sectionspan["affected_candidate_count"] == 12
    assert table_review["priority"] == "P0"
    assert table_review["affected_layers"] == ["table_region"]
    assert table_review["affected_candidate_count"] == 4
    assert table_review["recommendedNextTranche"] == "table_cell_provenance_review_pack"


def test_candidate_layer_blocker_backlog_uses_equation_quote_pdf_offset_supplement_counts(tmp_path: Path) -> None:
    summary = _source_aligned_summary_payload()
    summary["counts"] = {
        **summary["counts"],
        "figureCaptionOriginalPdfOffsetFeasibilityRows": 11,
        "figureCaptionOriginalPdfOffsetRecoveredRows": 9,
        "figureCaptionOriginalPdfOffsetBlockedRows": 2,
        "tableRegionOriginalPdfOffsetFeasibilityRows": 5,
        "tableRegionOriginalPdfOffsetRecoveredRows": 4,
        "tableRegionOriginalPdfOffsetBlockedRows": 1,
        "equationQuoteOriginalPdfOffsetFeasibilityRows": 9,
        "equationQuoteOriginalPdfOffsetRecoveredRows": 2,
        "equationQuoteOriginalPdfOffsetBlockedRows": 7,
    }
    summary["releaseCandidateAssessment"] = {
        **summary["releaseCandidateAssessment"],
        "mainBlockers": [
            *summary["releaseCandidateAssessment"]["mainBlockers"],
            "equation_quote_pdf_offsets_require_quote_review",
        ],
    }
    gate_path, summary_path, eval_path = _reports(
        tmp_path,
        gate=_gate_payload(
            gate={
                "candidateLayerReviewReady": True,
                "strictEvidenceReady": False,
                "parserRoutingReady": False,
                "answerIntegrationReady": False,
                "blockers": [
                    "non_sectionspan_layers_lack_original_pdf_offsets",
                    "equation_quote_pdf_offsets_require_quote_review",
                    "candidate_layers_are_report_only",
                    "runtime_promotion_disabled_for_tranche",
                ],
            }
        ),
        summary=summary,
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    non_sectionspan = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "non_sectionspan_layers_lack_original_pdf_offsets"
    )
    equation_review = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "equation_quote_pdf_offsets_require_quote_review"
    )
    assert non_sectionspan["affected_candidate_count"] == 10
    assert equation_review["priority"] == "P0"
    assert equation_review["affected_layers"] == ["equation_quote"]
    assert equation_review["affected_candidate_count"] == 2
    assert equation_review["recommendedNextTranche"] == "equation_quote_offset_review_pack"


def test_candidate_layer_blocker_backlog_uses_non_sectionspan_pdf_offset_audit_counts(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(
        tmp_path / "inputs",
        gate=_gate_payload(
            gate={
                "candidateLayerReviewReady": True,
                "strictEvidenceReady": False,
                "parserRoutingReady": False,
                "answerIntegrationReady": False,
                "blockers": [
                    "candidate_layers_are_report_only",
                    "runtime_promotion_disabled_for_tranche",
                ],
            }
        ),
        summary=_source_aligned_summary_payload(),
    )
    audit = _write(
        tmp_path / "inputs",
        "non-sectionspan-pdf-offset-feasibility-audit.json",
        _non_sectionspan_pdf_offset_audit_payload(),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        non_sectionspan_pdf_offset_feasibility_audit_report=audit,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    non_sectionspan = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "non_sectionspan_layers_lack_original_pdf_offsets"
    )
    figure_review = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "figure_caption_pdf_offsets_require_region_link_review"
    )
    table_review = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "table_caption_pdf_offsets_require_cell_provenance_review"
    )
    equation_alignment = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "equation_quote_alignment_missing"
    )
    assert non_sectionspan["affected_candidate_count"] == 12
    assert figure_review["affected_candidate_count"] == 9
    assert table_review["affected_candidate_count"] == 4
    assert equation_alignment["affected_candidate_count"] == 9
    assert payload["counts"]["nonSectionspanPdfOffsetAuditRows"] == 25
    assert payload["counts"]["nonSectionspanPdfOffsetRecoveredRows"] == 13
    assert payload["counts"]["nonSectionspanPdfOffsetBlockedRows"] == 12
    assert payload["counts"]["nonSectionspanPdfOffsetDiagnosticPageContextRows"] == 8
    assert payload["counts"]["nonSectionspanPdfOffsetReadyForRegionReviewRows"] == 13


def test_candidate_layer_blocker_backlog_blocks_wrong_non_sectionspan_audit_schema(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "inputs")
    audit = _write(
        tmp_path / "inputs",
        "non-sectionspan-pdf-offset-feasibility-audit.json",
        _non_sectionspan_pdf_offset_audit_payload(schema="example.wrong.non-sectionspan-audit.v1"),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        non_sectionspan_pdf_offset_feasibility_audit_report=audit,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "non_sectionspan_pdf_offset_feasibility_audit_schema_mismatch" in payload["gate"]["schemaViolations"]


def test_candidate_layer_blocker_backlog_uses_downstream_feasibility_audit_counts(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(
        tmp_path / "inputs",
        gate=_gate_payload(
            gate={
                "candidateLayerReviewReady": True,
                "strictEvidenceReady": False,
                "parserRoutingReady": False,
                "answerIntegrationReady": False,
                "blockers": [
                    "candidate_layers_are_report_only",
                    "runtime_promotion_disabled_for_tranche",
                ],
            }
        ),
        summary=_source_aligned_summary_payload(),
    )
    equation = _write(tmp_path / "inputs", "equation-alignment.json", _equation_alignment_audit_payload())
    table = _write(tmp_path / "inputs", "table-cell-provenance.json", _table_cell_provenance_audit_payload())
    figure = _write(tmp_path / "inputs", "figure-region-link.json", _figure_region_link_audit_payload())

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        equation_alignment_feasibility_audit_report=equation,
        table_cell_provenance_feasibility_audit_report=table,
        figure_region_link_feasibility_audit_report=figure,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    equation_item = next(item for item in payload["backlog"] if item["blocker"] == "equation_quote_alignment_missing")
    table_item = next(
        item for item in payload["backlog"] if item["blocker"] == "table_cell_row_column_bbox_provenance_missing"
    )
    figure_item = next(item for item in payload["backlog"] if item["blocker"] == "figure_region_link_unverified")
    assert equation_item["affected_candidate_count"] == 9
    assert table_item["affected_candidate_count"] == 5
    assert figure_item["affected_candidate_count"] == 11
    assert payload["counts"]["equationAlignmentAuditRows"] == 9
    assert payload["counts"]["equationAlignmentCanonicalSourceSpanCreatedRows"] == 0
    assert payload["counts"]["equationAlignmentDiagnosticTermContextRows"] == 8
    assert payload["counts"]["tableCellProvenanceAuditRows"] == 5
    assert payload["counts"]["tableCellProvenanceTotalTableCells"] == 681
    assert payload["counts"]["tableCellProvenanceCellSourceSpanRows"] == 0
    assert payload["counts"]["tableCellProvenanceCitationGradeRows"] == 0
    assert payload["counts"]["figureRegionLinkAuditRows"] == 11
    assert payload["counts"]["figureRegionLinkCaptionSourceSpanRows"] == 9
    assert payload["counts"]["figureRegionLinkVerifiedRows"] == 0


def test_candidate_layer_blocker_backlog_blocks_wrong_downstream_audit_schema(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "inputs")
    equation = _write(
        tmp_path / "inputs",
        "equation-alignment.json",
        _equation_alignment_audit_payload(schema="example.wrong.equation-alignment.v1"),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        equation_alignment_feasibility_audit_report=equation,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "equation_alignment_feasibility_audit_schema_mismatch" in payload["gate"]["schemaViolations"]


def test_candidate_layer_blocker_backlog_uses_blocker_decision_record_pending_counts(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(
        tmp_path / "inputs",
        gate=_gate_payload(
            gate={
                "candidateLayerReviewReady": True,
                "strictEvidenceReady": False,
                "parserRoutingReady": False,
                "answerIntegrationReady": False,
                "blockers": [
                    "candidate_layers_are_report_only",
                    "runtime_promotion_disabled_for_tranche",
                ],
            }
        ),
        summary=_source_aligned_summary_payload(),
    )
    decision_record = _write(
        tmp_path / "inputs",
        "candidate-layer-blocker-decision-record.json",
        _candidate_layer_blocker_decision_record_payload(),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        candidate_layer_blocker_decision_record_report=decision_record,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    item = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "candidate_layer_blocker_decision_record_pending"
    )
    assert item["priority"] == "P0"
    assert item["affected_layers"] == ["sectionspan", "figure_caption", "equation_quote", "table_region"]
    assert item["affected_candidate_count"] == 12
    assert item["recommendedNextTranche"] == "manual_record_candidate_layer_blocker_decisions"
    assert payload["counts"]["candidateLayerBlockerDecisionRecordRows"] == 12
    assert payload["counts"]["candidateLayerBlockerDecisionNeedsReviewRows"] == 12
    assert payload["counts"]["candidateLayerBlockerManualApprovalRows"] == 0
    assert payload["counts"]["candidateLayerBlockerOperatorApprovedRows"] == 0


def test_candidate_layer_blocker_backlog_skips_completed_blocker_decision_record(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "inputs")
    decision_record = _write(
        tmp_path / "inputs",
        "candidate-layer-blocker-decision-record.json",
        _candidate_layer_blocker_decision_record_payload(
            status="decision_recorded",
            counts={
                "recordRows": 12,
                "needsReviewRows": 0,
                "manualApprovalRows": 3,
                "manualRejectionRows": 0,
                "operatorApprovedRows": 1,
                "operatorDeclinedRows": 0,
                "technicalAcceptedOpenRows": 6,
                "technicalDeferredRows": 0,
                "policyAcceptedGuardrailRows": 2,
                "strictEligibleRows": 0,
                "citationGradeRows": 0,
                "runtimeEvidenceRows": 0,
                "unsafeUpstreamFlagCount": 0,
            },
            gate={
                "decisionRecordReady": True,
                "allDecisionRowsComplete": True,
                "humanReviewComplete": True,
                "operatorApprovalComplete": True,
                "strictEvidenceReady": False,
                "parserRoutingReady": False,
                "answerIntegrationReady": False,
                "runtimePromotionAllowed": False,
            },
        ),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        candidate_layer_blocker_decision_record_report=decision_record,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    blockers = {item["blocker"] for item in payload["backlog"]}
    assert "candidate_layer_blocker_decision_record_pending" not in blockers
    assert payload["counts"]["candidateLayerBlockerDecisionNeedsReviewRows"] == 0
    assert payload["counts"]["candidateLayerBlockerManualApprovalRows"] == 3
    assert payload["counts"]["candidateLayerBlockerOperatorApprovedRows"] == 1


def test_candidate_layer_blocker_backlog_blocks_wrong_blocker_decision_record_schema(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "inputs")
    decision_record = _write(
        tmp_path / "inputs",
        "candidate-layer-blocker-decision-record.json",
        _candidate_layer_blocker_decision_record_payload(schema="example.wrong.blocker-decision-record.v1"),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        candidate_layer_blocker_decision_record_report=decision_record,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "candidate_layer_blocker_decision_record_schema_mismatch" in payload["gate"]["schemaViolations"]


def test_candidate_layer_blocker_backlog_includes_table_cell_isolated_extractor_approval_gate(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "inputs")
    table_cell_result = _write(tmp_path / "inputs", "table-cell-result.json", _table_cell_result_payload())

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        table_cell_isolated_extractor_pilot_result_report=table_cell_result,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    item = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "table_cell_isolated_extractor_approval_required"
    )
    assert item["priority"] == "P0"
    assert item["affected_layers"] == ["table_region"]
    assert item["affected_candidate_count"] == 2
    assert item["recommendedNextTranche"] == "table_cell_isolated_extractor_pilot_requires_explicit_approval"
    assert "strict_evidence_promotion" in item["disallowedActions"]
    assert payload["counts"]["tableCellIsolatedExtractorTargetRows"] == 2
    assert payload["counts"]["tableCellIsolatedExtractorApprovalRequiredRows"] == 2
    assert payload["counts"]["tableCellIsolatedExtractorProbeAttemptedRows"] == 0


def test_candidate_layer_blocker_backlog_includes_table_cell_blocked_extractor_gate(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "inputs")
    table_cell_result = _write(
        tmp_path / "inputs",
        "table-cell-result.json",
        _table_cell_result_payload(
            status="blocked",
            counts={
                "targetRows": 2,
                "probeAttemptedRows": 0,
                "approvalRequiredRows": 0,
                "blockedRows": 2,
                "strictEligibleRows": 0,
                "citationGradeRows": 0,
                "runtimeEvidenceRows": 0,
            },
            gate={
                "pilotExecuted": False,
                "approvalRequiredBeforeInstallOrRun": False,
                "extractorAvailable": False,
                "strictEvidenceReady": False,
                "parserRoutingReady": False,
                "answerIntegrationReady": False,
            },
        ),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        table_cell_isolated_extractor_pilot_result_report=table_cell_result,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    item = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "table_cell_isolated_extractor_unavailable_or_blocked"
    )
    assert item["priority"] == "P0"
    assert item["affected_candidate_count"] == 2
    assert item["recommendedNextTranche"] == "table_cell_isolated_extractor_dependency_repair_or_alternative_review"
    assert payload["counts"]["tableCellIsolatedExtractorBlockedRows"] == 2


def test_candidate_layer_blocker_backlog_blocks_wrong_table_cell_result_schema(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "inputs")
    table_cell_result = _write(
        tmp_path / "inputs",
        "table-cell-result.json",
        _table_cell_result_payload(schema="example.wrong.table-cell-result.v1"),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        table_cell_isolated_extractor_pilot_result_report=table_cell_result,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "table_cell_isolated_extractor_pilot_result_schema_mismatch" in payload["gate"]["schemaViolations"]


def test_candidate_layer_blocker_backlog_includes_sectionspan_human_review_pending_gate(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "inputs")
    human_review_gate = _write(
        tmp_path / "inputs",
        "sectionspan-human-review-gate.json",
        _sectionspan_human_review_gate_payload(),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        sectionspan_pdf_offset_human_review_gate_report=human_review_gate,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    item = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "sectionspan_pdf_offset_human_review_pending"
    )
    assert item["priority"] == "P0"
    assert item["affected_layers"] == ["sectionspan"]
    assert item["affected_candidate_count"] == 61
    assert item["recommendedNextTranche"] == "sectionspan_pdf_offset_human_review_execution"
    assert payload["counts"]["sectionspanHumanReviewGateRows"] == 61
    assert payload["counts"]["sectionspanHumanReviewPendingRows"] == 61
    assert payload["counts"]["sectionspanHumanReviewApprovedRows"] == 0


def test_candidate_layer_blocker_backlog_skips_completed_sectionspan_human_review_gate(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "inputs")
    human_review_gate = _write(
        tmp_path / "inputs",
        "sectionspan-human-review-gate.json",
        _sectionspan_human_review_gate_payload(
            status="review_recorded",
            counts={
                "gateRows": 61,
                "pendingHumanReviewRows": 0,
                "approvedForLaterPromotionDesignRows": 61,
                "rejectedRows": 0,
                "heldOutRows": 0,
                "strictEligibleRows": 0,
                "citationGradeRows": 0,
                "runtimeEvidenceRows": 0,
            },
            gate={
                "humanReviewGateReady": True,
                "humanReviewComplete": True,
                "strictEvidenceReady": False,
                "parserRoutingReady": False,
                "answerIntegrationReady": False,
                "runtimePromotionAllowed": False,
            },
        ),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        sectionspan_pdf_offset_human_review_gate_report=human_review_gate,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    blockers = {item["blocker"] for item in payload["backlog"]}
    assert "sectionspan_pdf_offset_human_review_pending" not in blockers
    assert payload["counts"]["sectionspanHumanReviewPendingRows"] == 0
    assert payload["counts"]["sectionspanHumanReviewApprovedRows"] == 61


def test_candidate_layer_blocker_backlog_blocks_wrong_sectionspan_human_review_gate_schema(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "inputs")
    human_review_gate = _write(
        tmp_path / "inputs",
        "sectionspan-human-review-gate.json",
        _sectionspan_human_review_gate_payload(schema="example.wrong.sectionspan-human-review-gate.v1"),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        sectionspan_pdf_offset_human_review_gate_report=human_review_gate,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "sectionspan_pdf_offset_human_review_gate_schema_mismatch" in payload["gate"]["schemaViolations"]


def test_candidate_layer_blocker_backlog_includes_selected_decision_file_required_gate(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "inputs")
    proposal = _write(
        tmp_path / "inputs",
        "sectionspan-selected-decision-proposal.json",
        _sectionspan_selected_decision_proposal_payload(),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        sectionspan_pdf_offset_selected_review_decision_proposal_report=proposal,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    item = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "sectionspan_selected_review_decision_file_required"
    )
    assert item["priority"] == "P0"
    assert item["affected_layers"] == ["sectionspan"]
    assert item["affected_candidate_count"] == 12
    assert item["recommendedNextTranche"] == "manual_record_selected_sectionspan_review_decisions"
    assert payload["counts"]["sectionspanSelectedDecisionProposalRows"] == 12
    assert payload["counts"]["sectionspanSelectedDecisionProposalApproveRows"] == 12
    assert payload["counts"]["sectionspanSelectedDecisionAcceptedRows"] == 0


def test_candidate_layer_blocker_backlog_blocks_wrong_selected_decision_proposal_schema(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "inputs")
    proposal = _write(
        tmp_path / "inputs",
        "sectionspan-selected-decision-proposal.json",
        _sectionspan_selected_decision_proposal_payload(schema="example.wrong.selected-decision-proposal.v1"),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        sectionspan_pdf_offset_selected_review_decision_proposal_report=proposal,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "sectionspan_pdf_offset_selected_review_decision_proposal_schema_mismatch" in payload["gate"]["schemaViolations"]


def test_candidate_layer_blocker_backlog_includes_selected_next_action_manual_edit_gate(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "inputs")
    next_action = _write(
        tmp_path / "inputs",
        "sectionspan-selected-next-action-brief.json",
        _sectionspan_selected_next_action_brief_payload(),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        sectionspan_pdf_offset_selected_review_next_action_brief_report=next_action,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    item = next(
        item
        for item in payload["backlog"]
        if item["blocker"] == "sectionspan_selected_review_manual_edit_required"
    )
    assert item["priority"] == "P0"
    assert item["affected_layers"] == ["sectionspan"]
    assert item["affected_candidate_count"] == 12
    assert item["recommendedNextTranche"] == "manual_edit_selected_sectionspan_review_decision_file"
    assert payload["counts"]["sectionspanSelectedNextActionBriefRows"] == 12
    assert payload["counts"]["sectionspanSelectedNextActionNeedsReviewRows"] == 12
    assert payload["counts"]["sectionspanSelectedNextActionSuggestedApproveRows"] == 12
    assert payload["counts"]["sectionspanSelectedNextActionDecisionRecordNeedsReviewRows"] == 12


def test_candidate_layer_blocker_backlog_omits_selected_next_action_gate_when_manual_review_recorded(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "inputs")
    next_action = _write(
        tmp_path / "inputs",
        "sectionspan-selected-next-action-brief.json",
        _sectionspan_selected_next_action_brief_payload(
            status="manual_review_recorded_non_runtime",
            counts={
                "briefRows": 12,
                "needsReviewRows": 0,
                "nonNeedsReviewRows": 12,
                "suggestedApproveForLaterPromotionDesignRows": 12,
                "suggestedNeedsReviewRows": 0,
                "validationValidRows": 12,
                "validationInvalidRows": 0,
                "validationMissingRows": 0,
                "decisionRecordNeedsReviewRows": 0,
                "decisionRecordApprovedForLaterPromotionDesignRows": 12,
                "decisionRecordRejectedRows": 0,
                "strictEligibleRows": 0,
                "citationGradeRows": 0,
                "runtimeEvidenceRows": 0,
            },
            gate={
                "nextActionBriefReady": True,
                "manualReviewRequired": False,
                "autoApprovalAllowed": False,
                "humanReviewComplete": True,
                "strictEvidenceReady": False,
                "parserRoutingReady": False,
                "answerIntegrationReady": False,
                "runtimePromotionAllowed": False,
            },
        ),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        sectionspan_pdf_offset_selected_review_next_action_brief_report=next_action,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    assert "sectionspan_selected_review_manual_edit_required" not in {
        item["blocker"] for item in payload["backlog"]
    }
    assert payload["counts"]["sectionspanSelectedNextActionDecisionRecordNeedsReviewRows"] == 0
    assert payload["counts"]["strictEligibleCandidates"] == 0


def test_candidate_layer_blocker_backlog_blocks_wrong_selected_next_action_schema(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "inputs")
    next_action = _write(
        tmp_path / "inputs",
        "sectionspan-selected-next-action-brief.json",
        _sectionspan_selected_next_action_brief_payload(schema="example.wrong.selected-next-action.v1"),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
        sectionspan_pdf_offset_selected_review_next_action_brief_report=next_action,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "sectionspan_pdf_offset_selected_review_next_action_brief_schema_mismatch" in payload["gate"]["schemaViolations"]


def test_candidate_layer_blocker_backlog_remains_non_strict_and_blocks_runtime_actions(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path)

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
    )

    assert payload["policy"]["backlogOnly"] is True
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["runtimePromotionAllowed"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["canonicalParsedArtifactsWritten"] is False
    assert payload["policy"]["databaseMutation"] is False
    assert payload["policy"]["reindexOrReembed"] is False
    assert payload["policy"]["answerIntegrationChanged"] is False
    for item in payload["backlog"]:
        assert item["evidence_tier"] == "candidate_backlog_only"
        assert item["strict_eligible"] is False
        assert "strict_evidence_promotion" in item["disallowedActions"]
        assert "parser_routing" in item["disallowedActions"]
        assert "database_mutation" in item["disallowedActions"]


def test_candidate_layer_blocker_backlog_blocks_wrong_input_schema_ids(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(
        tmp_path,
        gate=_gate_payload(schema="example.wrong.gate.v1"),
        summary=_summary_payload(schema="example.wrong.summary.v1"),
        eval_design=_eval_payload(schema="example.wrong.eval.v1"),
    )

    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert set(payload["gate"]["schemaViolations"]) == {
        "candidate_layer_review_gate_schema_mismatch",
        "structured_candidate_summary_schema_mismatch",
        "complex_qa_eval_design_schema_mismatch",
    }


def test_candidate_layer_blocker_backlog_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    gate_path, summary_path, eval_path = _reports(tmp_path / "input")
    payload = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=gate_path,
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
    )

    paths = write_candidate_layer_blocker_backlog_reports(payload, tmp_path / "reports")

    assert set(paths) == {"backlog", "markdown"}
    backlog = json.loads(Path(paths["backlog"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(backlog, CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID, strict=True).ok
    assert "This backlog is report-only" in markdown
    assert "table_cell_row_column_bbox_provenance_missing" in markdown
