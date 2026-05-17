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
