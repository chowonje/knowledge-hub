from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate import (
    BLOCKED_DECISION,
    DEFAULT_CONTROLLED_EXECUTION_REPORT,
    DEFAULT_POSITIVE_SEED_REPORT,
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_ANSWER_EXECUTION_GATE_SCHEMA_ID,
    READY_DECISION,
    build_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate,
    write_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate,
)


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _positive_seed_report() -> dict[str, Any]:
    return _read(DEFAULT_POSITIVE_SEED_REPORT)


def _controlled_report() -> dict[str, Any]:
    return _read(DEFAULT_CONTROLLED_EXECUTION_REPORT)


def _fake_execution(*, seed_row: dict[str, Any], seed_question: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
    _ = seed_question, papers_dir
    paper_ids = [str(item) for item in list(seed_row.get("paperIds") or [])]
    citation_min = max(2, len(paper_ids) * 2)
    support_groups = list(seed_row.get("supportTermGroups") or [])
    return {
        "caseIndex": int(seed_row.get("caseIndex") or 0),
        "caseId": str(seed_row.get("caseId") or ""),
        "questionCategory": str(seed_row.get("questionCategory") or ""),
        "expectedEvidenceType": "section_paragraph",
        "answerabilityExpectation": "answerable",
        "paperIds": paper_ids,
        "questionSha256": str(seed_row.get("questionSha256") or ""),
        "observedStatus": "ok",
        "observedAnswerable": True,
        "adapterStatus": "applied",
        "adapterRowsAdded": citation_min,
        "adapterCandidateRowsConsidered": citation_min,
        "selectedEvidenceCount": citation_min,
        "citationCount": citation_min,
        "evidencePacketContractSpanRows": citation_min,
        "localFakeLlmCallRows": 1,
        "observedSourceIds": paper_ids,
        "missingSourceIds": [],
        "supportTermGroups": support_groups,
        "matchedSupportTermGroups": support_groups,
        "missingSupportTermGroups": [],
        "dimensionStatuses": {
            "schemaValid": True,
            "answerPayloadStatusOk": True,
            "answerabilityExpectation": True,
            "sourceCoverage": True,
            "citationProvenance": True,
            "supportTermCoverage": True,
            "answerGenerated": True,
            "publicDefaultHeld": True,
        },
        "qualityScore": 1.0,
        "qualityGrade": "pass",
        "pass": True,
        "answerTextSha256": "sha256:" + "a" * 64,
        "answerTextByteLength": 64,
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "failureReasons": [],
    }


def _build(**updates: Any) -> dict[str, Any]:
    return build_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate(
        positive_seed_report=updates.pop("positive_seed_report", _positive_seed_report()),
        controlled_execution_report=updates.pop("controlled_execution_report", _controlled_report()),
        execute_case=updates.pop("execute_case", _fake_execution),
        generated_at="2026-05-30T00:00:00Z",
        **updates,
    )


def test_positive_answer_execution_gate_ready_for_seven_seed_rows() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["nextRecommendedTranche"] == "corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution"
    assert report["counts"]["inputPositiveSeedRows"] == 7
    assert report["counts"]["attemptedPositiveAnswerRows"] == 7
    assert report["counts"]["positiveAnswerPassRows"] == 7
    assert report["counts"]["positiveAnswerFailRows"] == 0
    assert report["counts"]["heldExpectedNoAnswerRows"] == 17
    assert report["counts"]["heldStructuredModalityRows"] == 34
    assert report["counts"]["controlledExecutionUnexpectedAnswerableRows"] == 0
    assert report["counts"]["controlledExecutionNoAnswerSafetyFailRows"] == 0
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["counts"]["schemaViolationCount"] == 0
    assert report["gate"]["noAnswerSafetyStillGreen"] is True
    assert report["gate"]["allPositiveAnswersPassed"] is True
    assert validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_ANSWER_EXECUTION_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_positive_answer_execution_blocks_when_seed_report_not_ready() -> None:
    seed = _positive_seed_report()
    seed["status"] = "blocked"

    report = _build(positive_seed_report=seed)

    assert report["status"] == "blocked"
    assert report["decision"] == BLOCKED_DECISION
    assert "positive_section_paragraph_seed_not_ready" in report["gate"]["semanticViolations"]


def test_positive_answer_execution_blocks_when_no_answer_safety_regresses() -> None:
    controlled = deepcopy(_controlled_report())
    controlled["counts"]["unexpectedAnswerableRows"] = 1

    report = _build(controlled_execution_report=controlled)

    assert report["status"] == "blocked"
    assert "controlled_execution_unexpected_answerable_rows_present" in report["gate"]["semanticViolations"]
    assert "no_answer_safety_unexpected_answerable_rows_present" in report["gate"]["semanticViolations"]


def test_positive_answer_execution_blocks_failed_positive_row() -> None:
    def fail_one(*, seed_row: dict[str, Any], seed_question: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
        row = _fake_execution(seed_row=seed_row, seed_question=seed_question, papers_dir=papers_dir)
        if row["caseId"] == "complex-paper-qa-seed-20260520-q028":
            row["dimensionStatuses"]["supportTermCoverage"] = False
            row["qualityScore"] = 0.875
            row["qualityGrade"] = "partial"
            row["pass"] = False
            row["failureReasons"] = ["support_term_gap"]
        return row

    report = _build(execute_case=fail_one)

    assert report["status"] == "blocked"
    assert report["counts"]["positiveAnswerPassRows"] == 6
    assert report["counts"]["positiveAnswerFailRows"] == 1
    assert "positive_answer_execution_fail_rows:1" in report["gate"]["semanticViolations"]


def test_positive_answer_execution_blocks_when_pass_rows_below_minimum() -> None:
    def fail_all(*, seed_row: dict[str, Any], seed_question: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
        row = _fake_execution(seed_row=seed_row, seed_question=seed_question, papers_dir=papers_dir)
        row["dimensionStatuses"]["answerGenerated"] = False
        row["qualityScore"] = 0.875
        row["qualityGrade"] = "partial"
        row["pass"] = False
        row["failureReasons"] = ["answer_text_empty"]
        return row

    report = _build(execute_case=fail_all)

    assert report["status"] == "blocked"
    assert report["counts"]["positiveAnswerPassRows"] == 0
    assert "positive_answer_execution_pass_rows_below_minimum" in report["gate"]["semanticViolations"]


def test_positive_answer_execution_rows_exclude_raw_question_answer_citation_source_and_excerpt() -> None:
    report = _build()

    for row in report["executionRows"]:
        assert "question" not in row
        assert "answer" not in row
        assert "citations" not in row
        assert "sources" not in row
        assert "excerpt" not in row
        assert row["answerTextIncludedInReport"] is False
        assert row["citationPayloadIncludedInReport"] is False
        assert row["sourcePayloadIncludedInReport"] is False
        assert row["excerptIncludedInReport"] is False
        assert row["answerTextSha256"].startswith("sha256:")


def test_positive_answer_execution_blocks_private_path_marker() -> None:
    def leaky_row(*, seed_row: dict[str, Any], seed_question: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
        row = _fake_execution(seed_row=seed_row, seed_question=seed_question, papers_dir=papers_dir)
        row["observedSourceIds"] = ["/" + "Users" + "/example/private"]
        return row

    report = _build(execute_case=leaky_row)

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] >= 1
    assert "positive_answer_execution_private_path_marker" in report["gate"]["semanticViolations"]


def test_positive_answer_execution_blocks_unsafe_counter_in_inputs() -> None:
    seed = deepcopy(_positive_seed_report())
    seed["counts"]["databaseMutationRows"] = 1

    report = _build(positive_seed_report=seed)

    assert report["status"] == "blocked"
    assert "unsafe_counter_nonzero:positive_seed:databaseMutationRows" in report["gate"]["semanticViolations"]


def test_positive_answer_execution_write_report_validates_schema(tmp_path: Path) -> None:
    report = _build()
    paths = write_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )
    loaded = _read(paths["json"])

    assert validate_payload(
        loaded,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_ANSWER_EXECUTION_GATE_SCHEMA_ID,
        strict=True,
    ).ok
    assert "positiveAnswerPassRows" in Path(paths["markdown"]).read_text(encoding="utf-8")
