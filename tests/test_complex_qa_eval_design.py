from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.complex_qa_eval_design import (
    COMPLEX_QA_EVAL_DESIGN_SCHEMA_ID,
    build_complex_qa_eval_design,
    write_complex_qa_eval_design_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _input_reports(root: Path) -> dict[str, Path]:
    return {
        "summary": _write(
            root,
            "summary.json",
            {
                "schema": "knowledge-hub.paper.structured-candidate-summary.v1",
                "counts": {
                    "totalCandidates": 8,
                    "strictEligibleCandidates": 0,
                    "citationGradeCandidates": 0,
                },
            },
        ),
        "section": _write(
            root,
            "section.json",
            {
                "schema": "knowledge-hub.paper.sectionspan-candidate-report.v1",
                "candidates": [
                    {
                        "candidate_id": "sectionspan:paper-a:0001",
                        "paper_id": "paper-a",
                        "candidate_text": "Abstract",
                        "section_type": "abstract",
                        "section_title": "Abstract",
                        "page": 1,
                        "chars_start": 0,
                        "chars_end": 8,
                        "sourceContentHash": "hash-a",
                    },
                    {
                        "candidate_id": "sectionspan:paper-a:0002",
                        "paper_id": "paper-a",
                        "candidate_text": "2. Method",
                        "section_type": "numbered_section",
                        "section_title": "Method",
                        "page": 2,
                        "chars_start": 20,
                        "chars_end": 29,
                        "sourceContentHash": "hash-a",
                    },
                    {
                        "candidate_id": "sectionspan:paper-b:0001",
                        "paper_id": "paper-b",
                        "candidate_text": "1. Introduction",
                        "section_type": "numbered_section",
                        "section_title": "Introduction",
                        "page": 1,
                        "chars_start": 0,
                        "chars_end": 15,
                        "sourceContentHash": "hash-b",
                    },
                ],
            },
        ),
        "figure": _write(
            root,
            "figure.json",
            {
                "schema": "knowledge-hub.paper.figure-caption-candidate-report.v1",
                "candidates": [
                    {
                        "candidate_id": "figurecaption:paper-a:0001",
                        "paper_id": "paper-a",
                        "figure_label": "Figure 1",
                        "candidate_text": "Figure 1: The model.",
                        "canonical_alignment_status": "aligned",
                        "page": 3,
                        "chars_start": 100,
                        "chars_end": 120,
                        "sourceContentHash": "hash-a",
                    },
                    {
                        "candidate_id": "figurecaption:paper-b:0001",
                        "paper_id": "paper-b",
                        "figure_label": "Figure 2",
                        "candidate_text": "Figure 2: Blocked.",
                        "canonical_alignment_status": "failed",
                        "page": None,
                        "chars_start": None,
                        "chars_end": None,
                        "sourceContentHash": "hash-b",
                    },
                ],
            },
        ),
        "equation": _write(
            root,
            "equation.json",
            {
                "schema": "knowledge-hub.paper.equation-quote-candidate-report.v1",
                "candidates": [
                    {
                        "candidate_id": "equationquote:paper-a:0001",
                        "paper_id": "paper-a",
                        "equation_label": "tag:1",
                        "candidate_text": "x = y",
                        "canonical_alignment_status": "failed",
                        "page": None,
                        "chars_start": None,
                        "chars_end": None,
                        "sourceContentHash": "hash-a",
                    }
                ],
            },
        ),
        "table": _write(
            root,
            "table.json",
            {
                "schema": "knowledge-hub.paper.table-region-candidate-report.v1",
                "candidates": [
                    {
                        "candidate_id": "tableregion:paper-a:0001",
                        "paper_id": "paper-a",
                        "table_label": "Table 1",
                        "candidate_text": "Table 1: Results.",
                        "canonical_alignment_status": "aligned",
                        "page": 4,
                        "chars_start": 200,
                        "chars_end": 216,
                        "sourceContentHash": "hash-a",
                    }
                ],
            },
        ),
    }


def _build(paths: dict[str, Path]) -> dict:
    return build_complex_qa_eval_design(
        structured_summary_report=paths["summary"],
        sectionspan_report=paths["section"],
        figure_caption_report=paths["figure"],
        equation_quote_report=paths["equation"],
        table_region_report=paths["table"],
    )


def test_complex_qa_eval_design_generates_design_only_questions_and_validates_schema(tmp_path: Path) -> None:
    paths = _input_reports(tmp_path)

    payload = _build(paths)

    assert payload["schema"] == COMPLEX_QA_EVAL_DESIGN_SCHEMA_ID
    assert validate_payload(payload, COMPLEX_QA_EVAL_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["questionCount"] >= 5
    assert payload["counts"]["byQuestionType"]["section_evidence_question"] >= 2
    assert payload["counts"]["byQuestionType"]["figure_caption_question"] == 2
    assert payload["counts"]["byQuestionType"]["table_region_question"] == 1
    assert payload["counts"]["byQuestionType"]["equation_quote_question"] == 1
    assert payload["policy"]["designOnly"] is True
    assert payload["policy"]["questionsExecuted"] is False


def test_complex_qa_eval_design_keeps_all_questions_non_runtime_and_strict_required(tmp_path: Path) -> None:
    paths = _input_reports(tmp_path)

    payload = _build(paths)

    assert payload["counts"]["currentRuntimeAnswerableQuestions"] == 0
    assert payload["counts"]["executedQuestions"] == 0
    assert payload["counts"]["strictEvidenceCreated"] == 0
    for question in payload["questions"]:
        assert question["execution_status"] == "designed_not_executed"
        assert question["current_runtime_expectation"] == "abstain_until_strict_candidate_runtime_exists"
        assert question["strict_evidence_required_before_answering"] is True
        assert question["strict_eligible_now"] is False
        assert question["citation_grade_now"] is False
        assert question["candidate_only"] is True


def test_complex_qa_eval_design_marks_table_and_equation_blockers(tmp_path: Path) -> None:
    paths = _input_reports(tmp_path)

    payload = _build(paths)

    equation = next(item for item in payload["questions"] if item["question_type"] == "equation_quote_question")
    table = next(item for item in payload["questions"] if item["question_type"] == "table_region_question")
    assert "equation_quote_alignment_missing" in equation["blocked_by_current_candidates"]
    assert "table_cell_row_column_bbox_provenance_missing" in table["blocked_by_current_candidates"]


def test_complex_qa_eval_design_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    paths = _input_reports(tmp_path / "input")
    payload = _build(paths)

    report_paths = write_complex_qa_eval_design_reports(payload, tmp_path / "reports")

    assert set(report_paths) == {"design", "markdown"}
    design = json.loads(Path(report_paths["design"]).read_text(encoding="utf-8"))
    markdown = Path(report_paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(design, COMPLEX_QA_EVAL_DESIGN_SCHEMA_ID, strict=True).ok
    assert "design-only eval set" in markdown
    assert "designed_not_executed" in markdown
