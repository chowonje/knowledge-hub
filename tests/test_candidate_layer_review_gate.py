from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.candidate_layer_review_gate import (
    CANDIDATE_LAYER_REVIEW_GATE_SCHEMA_ID,
    build_candidate_layer_review_gate,
    write_candidate_layer_review_gate_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.structured-candidate-summary.v1",
        "counts": {
            "totalCandidates": 86,
            "strictEligibleCandidates": 0,
            "citationGradeCandidates": 0,
            "runtimeEvidenceCandidates": 0,
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
        "releaseCandidateAssessment": {
            "candidateLayerReviewReady": True,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "mainBlockers": [
                "equation_quote_alignment_missing",
                "table_cell_row_column_bbox_provenance_missing",
            ],
        },
    }
    payload.update(overrides)
    return payload


def _eval_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.complex-qa-eval-design.v1",
        "counts": {
            "questionCount": 20,
            "currentRuntimeAnswerableQuestions": 0,
            "executedQuestions": 0,
            "strictEvidenceCreated": 0,
        },
        "policy": {
            "questionsExecuted": False,
            "answerGenerationRun": False,
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


def _reports(root: Path, *, summary: dict | None = None, eval_design: dict | None = None) -> tuple[Path, Path]:
    summary_path = _write(root, "structured-candidate-summary.json", summary or _summary_payload())
    eval_path = _write(root, "complex-paper-qa-eval-design.json", eval_design or _eval_payload())
    return summary_path, eval_path


def test_candidate_layer_review_gate_reports_ready_without_runtime_promotion(tmp_path: Path) -> None:
    summary_path, eval_path = _reports(tmp_path)

    payload = build_candidate_layer_review_gate(
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
    )

    assert payload["schema"] == CANDIDATE_LAYER_REVIEW_GATE_SCHEMA_ID
    assert validate_payload(payload, CANDIDATE_LAYER_REVIEW_GATE_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ready_for_candidate_layer_review"
    assert payload["gate"]["candidateLayerReviewReady"] is True
    assert payload["gate"]["strictEvidenceReady"] is False
    assert payload["gate"]["parserRoutingReady"] is False
    assert payload["gate"]["strictPromotionDecision"] == "blocked"
    assert payload["gate"]["violations"] == []
    assert payload["counts"]["totalCandidates"] == 86
    assert payload["counts"]["questionCount"] == 20


def test_candidate_layer_review_gate_blocks_if_eval_was_executed_or_runtime_answerable(tmp_path: Path) -> None:
    summary_path, eval_path = _reports(
        tmp_path,
        eval_design=_eval_payload(
            counts={
                "questionCount": 20,
                "currentRuntimeAnswerableQuestions": 2,
                "executedQuestions": 1,
                "strictEvidenceCreated": 0,
            },
            policy={
                "questionsExecuted": True,
                "answerGenerationRun": True,
                "strictEvidenceCreated": False,
                "runtimePromotionAllowed": False,
                "parserRoutingChanged": False,
                "canonicalParsedArtifactsWritten": False,
                "databaseMutation": False,
                "reindexOrReembed": False,
                "answerIntegrationChanged": False,
            },
        ),
    )

    payload = build_candidate_layer_review_gate(
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["candidateLayerReviewReady"] is False
    assert "eval_design_runtime_answerable_questions_nonzero" in payload["gate"]["violations"]
    assert "eval_design_questions_executed" in payload["gate"]["violations"]
    assert "eval_policy_answerGenerationRun_true" in payload["gate"]["violations"]


def test_candidate_layer_review_gate_blocks_if_summary_promotes_strict_candidates(tmp_path: Path) -> None:
    summary_path, eval_path = _reports(
        tmp_path,
        summary=_summary_payload(
            counts={
                "totalCandidates": 86,
                "strictEligibleCandidates": 1,
                "citationGradeCandidates": 0,
                "runtimeEvidenceCandidates": 0,
            }
        ),
    )

    payload = build_candidate_layer_review_gate(
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["candidateLayerReviewReady"] is False
    assert "summary_strict_eligible_candidates_nonzero" in payload["gate"]["violations"]
    assert payload["gate"]["strictPromotionDecision"] == "blocked"


def test_candidate_layer_review_gate_blocks_wrong_upstream_schema_ids(tmp_path: Path) -> None:
    summary_path, eval_path = _reports(
        tmp_path,
        summary=_summary_payload(schema="example.wrong.summary.v1"),
        eval_design=_eval_payload(schema="example.wrong.eval.v1"),
    )

    payload = build_candidate_layer_review_gate(
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["candidateLayerReviewReady"] is False
    assert "structured_summary_schema_mismatch" in payload["gate"]["violations"]
    assert "complex_qa_eval_design_schema_mismatch" in payload["gate"]["violations"]


def test_candidate_layer_review_gate_blocks_mutation_and_runtime_policy_flags(tmp_path: Path) -> None:
    unsafe_policy = {
        "strictEvidenceCreated": True,
        "runtimePromotionAllowed": True,
        "parserRoutingChanged": True,
        "canonicalParsedArtifactsWritten": True,
        "databaseMutation": True,
        "reindexOrReembed": True,
        "answerIntegrationChanged": True,
    }
    summary_path, eval_path = _reports(
        tmp_path,
        summary=_summary_payload(policy=unsafe_policy),
        eval_design=_eval_payload(
            policy={
                **unsafe_policy,
                "questionsExecuted": False,
                "answerGenerationRun": False,
            }
        ),
    )

    payload = build_candidate_layer_review_gate(
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
    )

    assert payload["status"] == "blocked"
    violations = set(payload["gate"]["violations"])
    assert "summary_policy_strictEvidenceCreated_true" in violations
    assert "eval_policy_strictEvidenceCreated_true" in violations
    assert "summary_policy_databaseMutation_true" in violations
    assert "eval_policy_databaseMutation_true" in violations
    assert "summary_policy_reindexOrReembed_true" in violations
    assert "eval_policy_reindexOrReembed_true" in violations
    assert "summary_policy_canonicalParsedArtifactsWritten_true" in violations
    assert "eval_policy_canonicalParsedArtifactsWritten_true" in violations
    assert "summary_policy_parserRoutingChanged_true" in violations
    assert "eval_policy_parserRoutingChanged_true" in violations
    assert "summary_policy_answerIntegrationChanged_true" in violations
    assert "eval_policy_answerIntegrationChanged_true" in violations


def test_candidate_layer_review_gate_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    summary_path, eval_path = _reports(tmp_path / "input")
    payload = build_candidate_layer_review_gate(
        structured_summary_report=summary_path,
        complex_qa_eval_design_report=eval_path,
    )

    paths = write_candidate_layer_review_gate_reports(payload, tmp_path / "reports")

    assert set(paths) == {"gate", "markdown"}
    gate = json.loads(Path(paths["gate"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(gate, CANDIDATE_LAYER_REVIEW_GATE_SCHEMA_ID, strict=True).ok
    assert "does not create strict evidence" in markdown
    assert "ready_for_candidate_layer_review" in markdown
