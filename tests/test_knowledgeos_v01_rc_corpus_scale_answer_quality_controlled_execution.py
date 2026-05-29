from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution import (
    BLOCKED_DECISION,
    DEFAULT_LIVE_RUNNER_DRY_RUN_REPORT,
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_CONTROLLED_EXECUTION_SCHEMA_ID,
    READY_DECISION,
    build_knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution,
    write_knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution,
)


def _dry_run_report() -> dict[str, Any]:
    return json.loads(Path(DEFAULT_LIVE_RUNNER_DRY_RUN_REPORT).read_text(encoding="utf-8"))


def _passing_no_evidence_executor(*, case: dict[str, Any], seed_question: dict[str, Any], papers_dir: Path) -> dict[str, Any]:
    _ = seed_question, papers_dir
    return {
        "caseIndex": int(case["caseIndex"]),
        "caseId": str(case["caseId"]),
        "questionCategory": str(case["questionCategory"]),
        "expectedEvidenceType": str(case["expectedEvidenceType"]),
        "answerabilityExpectation": str(case["answerabilityExpectation"]),
        "paperIds": list(case.get("paperIds") or []),
        "questionSha256": str(case["questionSha256"]),
        "observedStatus": "no_evidence",
        "observedAnswerable": False,
        "adapterStatus": "skipped",
        "adapterRowsAdded": 0,
        "adapterCandidateRowsConsidered": 0,
        "selectedEvidenceCount": 0,
        "citationCount": 0,
        "evidencePacketContractSpanRows": 0,
        "localFakeLlmCallRows": 0,
        "schemaValid": True,
        "dimensionStatuses": {
            "schemaValid": True,
            "answerability": True,
            "noAnswerSafety": True,
            "citationProvenance": True,
            "sourceCoverage": True,
            "answerSupport": True,
        },
        "qualityScore": 1.0,
        "qualityGrade": "pass",
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "pass": True,
        "failureReasons": [],
    }


def _unexpected_answerable_executor(*, case: dict[str, Any], seed_question: dict[str, Any], papers_dir: Path) -> dict[str, Any]:
    _ = seed_question, papers_dir
    return {
        "caseIndex": int(case["caseIndex"]),
        "caseId": str(case["caseId"]),
        "questionCategory": str(case["questionCategory"]),
        "expectedEvidenceType": str(case["expectedEvidenceType"]),
        "answerabilityExpectation": str(case["answerabilityExpectation"]),
        "paperIds": list(case.get("paperIds") or []),
        "questionSha256": str(case["questionSha256"]),
        "observedStatus": "ok",
        "observedAnswerable": True,
        "adapterStatus": "applied",
        "adapterRowsAdded": 2,
        "adapterCandidateRowsConsidered": 2,
        "selectedEvidenceCount": 2,
        "citationCount": 2,
        "evidencePacketContractSpanRows": 2,
        "localFakeLlmCallRows": 1,
        "schemaValid": True,
        "dimensionStatuses": {
            "schemaValid": True,
            "answerability": False,
            "noAnswerSafety": False,
            "citationProvenance": False,
            "sourceCoverage": True,
            "answerSupport": False,
        },
        "qualityScore": 0.333333,
        "qualityGrade": "fail",
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "pass": False,
        "failureReasons": [
            "answerability_expectation_failed",
            "answer_support_failed",
            "citation_provenance_failed",
            "no_answer_safety_failed",
            "unexpected_answerable_for_no_answer_or_blocked_case",
            "unexpected_llm_call_for_no_answer_or_blocked_case",
        ],
    }


def _build(
    *,
    dry_run_report: dict[str, Any] | None = None,
    execute_case: Any = _passing_no_evidence_executor,
) -> dict[str, Any]:
    return build_knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution(
        live_runner_dry_run_report=dry_run_report or _dry_run_report(),
        execute_case=execute_case,
        generated_at="2026-05-29T00:00:00Z",
    )


def test_controlled_execution_ready_when_all_cases_abstain_safely() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["nextRecommendedTranche"] == "corpus_scale_answer_quality_positive_section_paragraph_seed"
    assert report["counts"]["attemptedCaseRows"] == 50
    assert report["counts"]["executionPassRows"] == 50
    assert report["counts"]["executionFailRows"] == 0
    assert report["counts"]["unexpectedAnswerableRows"] == 0
    assert report["counts"]["noAnswerSafetyFailRows"] == 0
    assert report["counts"]["liveAnswerExecutionRows"] == 50
    assert report["counts"]["answerPathInvokedRows"] == 50
    assert report["counts"]["localFakeLlmCallRows"] == 0
    assert report["counts"]["externalLlmCallRows"] == 0
    assert report["counts"]["modelApiCallRows"] == 0
    assert report["counts"]["judgeModelCallRows"] == 0
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["gate"]["controlledExecutionReady"] is True
    assert validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_CONTROLLED_EXECUTION_SCHEMA_ID,
        strict=True,
    ).ok


def test_controlled_execution_blocks_when_no_answer_cases_become_answerable() -> None:
    report = _build(execute_case=_unexpected_answerable_executor)

    assert report["status"] == "blocked"
    assert report["decision"] == BLOCKED_DECISION
    assert report["nextRecommendedTranche"] == "corpus_scale_answer_quality_answerability_gate_repair"
    assert report["counts"]["executionFailRows"] == 50
    assert report["counts"]["unexpectedAnswerableRows"] == 50
    assert report["counts"]["noAnswerSafetyFailRows"] == 50
    assert "unexpected_answerable_rows:50" in report["gate"]["semanticViolations"]
    assert "no_answer_safety_fail_rows:50" in report["gate"]["semanticViolations"]


def test_controlled_execution_blocks_when_dry_run_is_not_ready() -> None:
    dry_run = _dry_run_report()
    dry_run["status"] = "blocked"

    report = _build(dry_run_report=dry_run)

    assert report["status"] == "blocked"
    assert report["counts"]["attemptedCaseRows"] == 0
    assert "live_runner_dry_run_not_ready" in report["gate"]["semanticViolations"]


def test_controlled_execution_blocks_when_dry_run_next_tranche_is_wrong() -> None:
    dry_run = _dry_run_report()
    dry_run["nextRecommendedTranche"] = "unexpected_next_tranche"

    report = _build(dry_run_report=dry_run)

    assert report["status"] == "blocked"
    assert "live_runner_dry_run_next_tranche_not_controlled_execution" in report["gate"]["semanticViolations"]


def test_controlled_execution_blocks_private_path_marker_in_dry_run() -> None:
    dry_run = _dry_run_report()
    marker = "/" + "Users" + "/example/private"
    dry_run["warnings"] = list(dry_run.get("warnings") or []) + [marker]

    report = _build(dry_run_report=dry_run)

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "corpus_scale_controlled_execution_private_path_marker" in report["gate"]["semanticViolations"]


def test_controlled_execution_blocks_unsafe_dry_run_counter() -> None:
    dry_run = _dry_run_report()
    dry_run["counts"] = deepcopy(dry_run["counts"])
    dry_run["counts"]["externalLlmCallRows"] = 1

    report = _build(dry_run_report=dry_run)

    assert report["status"] == "blocked"
    assert "unsafe_counter_nonzero:dry_run:externalLlmCallRows" in report["gate"]["semanticViolations"]


def test_controlled_execution_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = _build()

    paths = write_knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert parsed["status"] == "ready"
    assert markdown.startswith("# KnowledgeOS v0.1 RC Corpus-Scale Answer Quality Controlled Execution")
    assert validate_payload(
        parsed,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_CONTROLLED_EXECUTION_SCHEMA_ID,
        strict=True,
    ).ok


def test_controlled_execution_rows_do_not_persist_raw_payloads() -> None:
    report = _build(execute_case=_unexpected_answerable_executor)
    row = report["executionRows"][0]

    for forbidden_key in ("question", "answer", "answerText", "citations", "sources", "excerpt", "excerpts"):
        assert forbidden_key not in row
    assert row["answerTextIncludedInReport"] is False
    assert row["citationPayloadIncludedInReport"] is False
    assert row["sourcePayloadIncludedInReport"] is False
    assert row["excerptIncludedInReport"] is False
