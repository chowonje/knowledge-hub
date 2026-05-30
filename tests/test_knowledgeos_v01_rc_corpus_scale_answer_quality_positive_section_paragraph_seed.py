from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed import (
    BLOCKED_DECISION,
    DEFAULT_CONTROLLED_EXECUTION_REPORT,
    DEFAULT_LIVE_RUNNER_DRY_RUN_REPORT,
    DEFAULT_POST_MERGE_REPORT,
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_SECTION_PARAGRAPH_SEED_SCHEMA_ID,
    READY_DECISION,
    build_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed,
    write_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed,
)


PASS_CASE_IDS = {
    "complex-paper-qa-seed-20260520-q028",
    "complex-paper-qa-seed-20260520-q029",
    "complex-paper-qa-seed-20260520-q034",
    "complex-paper-qa-seed-20260520-q035",
    "complex-paper-qa-seed-20260520-q036",
    "complex-paper-qa-seed-20260520-q037",
    "complex-paper-qa-seed-20260520-q039",
}


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _post_merge_report() -> dict[str, Any]:
    return _read(DEFAULT_POST_MERGE_REPORT)


def _controlled_report() -> dict[str, Any]:
    return _read(DEFAULT_CONTROLLED_EXECUTION_REPORT)


def _dry_run_report() -> dict[str, Any]:
    return _read(DEFAULT_LIVE_RUNNER_DRY_RUN_REPORT)


def _fake_probe(*, case: dict[str, Any], seed_question: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
    case_id = str(case.get("caseId") or "")
    paper_ids = [str(item) for item in list(case.get("paperIds") or [])]
    selected = case_id in PASS_CASE_IDS
    observed_source_ids = paper_ids if selected else paper_ids[:1]
    missing_source_ids = [] if selected else paper_ids[1:]
    support_groups = ["support-a", "support-b"]
    matched_groups = support_groups if selected else ["support-a"]
    missing_groups = [] if selected else ["support-b"]
    citation_min = max(2, len(paper_ids) * 2)
    return {
        "caseIndex": int(case.get("caseIndex") or 0),
        "caseId": case_id,
        "questionCategory": str(case.get("questionCategory") or ""),
        "originalExpectedEvidenceType": str(case.get("expectedEvidenceType") or ""),
        "originalAnswerabilityExpectation": str(case.get("answerabilityExpectation") or ""),
        "proposedExpectedEvidenceType": "section_paragraph",
        "proposedAnswerabilityExpectation": "answerable",
        "paperIds": paper_ids,
        "questionSha256": str(case.get("questionSha256") or ""),
        "observedStatus": "ok",
        "observedAnswerable": True,
        "adapterStatus": "applied",
        "adapterRowsAdded": citation_min,
        "adapterCandidateRowsConsidered": citation_min,
        "selectedEvidenceCount": citation_min if selected else max(1, citation_min - 1),
        "citationCount": citation_min if selected else max(1, citation_min - 1),
        "evidencePacketContractSpanRows": citation_min if selected else max(1, citation_min - 1),
        "localFakeLlmCallRows": 1,
        "observedSourceIds": observed_source_ids,
        "missingSourceIds": missing_source_ids,
        "supportTermGroups": support_groups,
        "matchedSupportTermGroups": matched_groups,
        "missingSupportTermGroups": missing_groups,
        "dimensionStatuses": {
            "schemaValid": True,
            "answerabilityExpectation": True,
            "sourceCoverage": selected,
            "citationProvenance": selected,
            "supportTermCoverage": selected,
            "publicDefaultHeld": True,
        },
        "qualityScore": 1.0 if selected else 0.5,
        "qualityGrade": "pass" if selected else "fail",
        "selectedPositiveSeed": selected,
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "failureReasons": [] if selected else ["source_coverage_gap", "support_term_gap"],
    }


def _build(**updates: Any) -> dict[str, Any]:
    return build_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed(
        post_merge_report=updates.pop("post_merge_report", _post_merge_report()),
        controlled_execution_report=updates.pop("controlled_execution_report", _controlled_report()),
        live_runner_dry_run_report=updates.pop("live_runner_dry_run_report", _dry_run_report()),
        execute_probe=updates.pop("execute_probe", _fake_probe),
        generated_at="2026-05-30T00:00:00Z",
        **updates,
    )


def test_positive_section_paragraph_seed_ready_with_seven_supported_rows() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["nextRecommendedTranche"] == "corpus_scale_answer_quality_positive_answer_execution_gate"
    assert report["counts"]["inputCaseRows"] == 50
    assert report["counts"]["eligibleSectionParagraphProbeRows"] == 13
    assert report["counts"]["positiveSeedRows"] == 7
    assert report["counts"]["positiveProbeHeldRows"] == 6
    assert report["counts"]["heldExpectedNoAnswerRows"] == 17
    assert report["counts"]["heldStructuredModalityRows"] == 34
    assert report["counts"]["controlledExecutionUnexpectedAnswerableRows"] == 0
    assert report["counts"]["controlledExecutionNoAnswerSafetyFailRows"] == 0
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["counts"]["schemaViolationCount"] == 0
    assert report["gate"]["noAnswerSafetyStillGreen"] is True
    assert validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_SECTION_PARAGRAPH_SEED_SCHEMA_ID,
        strict=True,
    ).ok


def test_positive_seed_blocks_when_post_merge_report_is_not_ready() -> None:
    post_merge = _post_merge_report()
    post_merge["status"] = "blocked"

    report = _build(post_merge_report=post_merge)

    assert report["status"] == "blocked"
    assert report["decision"] == BLOCKED_DECISION
    assert "answerability_gate_post_merge_not_ready" in report["gate"]["semanticViolations"]


def test_positive_seed_blocks_when_no_answer_safety_regresses() -> None:
    controlled = _controlled_report()
    controlled["counts"]["unexpectedAnswerableRows"] = 1

    report = _build(controlled_execution_report=controlled)

    assert report["status"] == "blocked"
    assert "controlled_execution_unexpected_answerable_rows_present" in report["gate"]["semanticViolations"]
    assert "no_answer_safety_unexpected_answerable_rows_present" in report["gate"]["semanticViolations"]


def test_positive_seed_blocks_when_positive_rows_below_minimum() -> None:
    def fail_probe(*, case: dict[str, Any], seed_question: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
        row = _fake_probe(case=case, seed_question=seed_question, papers_dir=papers_dir)
        row["selectedPositiveSeed"] = False
        row["dimensionStatuses"]["supportTermCoverage"] = False
        row["qualityScore"] = 0.5
        row["qualityGrade"] = "fail"
        row["failureReasons"] = ["support_term_gap"]
        return row

    report = _build(execute_probe=fail_probe)

    assert report["status"] == "blocked"
    assert report["counts"]["positiveSeedRows"] == 0
    assert "positive_seed_rows_below_minimum" in report["gate"]["semanticViolations"]


def test_positive_seed_keeps_expected_no_answer_rows_out_of_positive_probes() -> None:
    report = _build()
    all_rows = list(report["positiveSeedRows"]) + list(report["heldProbeRows"])

    assert all(row["originalAnswerabilityExpectation"] == "blocked_until_structured_evidence" for row in all_rows)
    assert all(row["questionCategory"] in {"method_comparison_qa", "limitation_qa"} for row in all_rows)


def test_positive_seed_blocks_private_path_marker() -> None:
    def leaky_probe(*, case: dict[str, Any], seed_question: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
        row = _fake_probe(case=case, seed_question=seed_question, papers_dir=papers_dir)
        if row["selectedPositiveSeed"]:
            row["observedSourceIds"] = ["/" + "Users" + "/example/private"]
        return row

    report = _build(execute_probe=leaky_probe)

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] >= 1
    assert "positive_section_paragraph_seed_private_path_marker" in report["gate"]["semanticViolations"]


def test_positive_seed_blocks_unsafe_counter_in_inputs() -> None:
    controlled = deepcopy(_controlled_report())
    controlled["counts"]["databaseMutationRows"] = 1

    report = _build(controlled_execution_report=controlled)

    assert report["status"] == "blocked"
    assert "unsafe_counter_nonzero:controlled_execution:databaseMutationRows" in report["gate"]["semanticViolations"]


def test_positive_seed_rows_exclude_raw_question_answer_citation_source_and_excerpt() -> None:
    report = _build()

    for row in list(report["positiveSeedRows"]) + list(report["heldProbeRows"]):
        assert "question" not in row
        assert "answer" not in row
        assert "citations" not in row
        assert "sources" not in row
        assert "excerpt" not in row
        assert row["answerTextIncludedInReport"] is False
        assert row["citationPayloadIncludedInReport"] is False
        assert row["sourcePayloadIncludedInReport"] is False
        assert row["excerptIncludedInReport"] is False


def test_positive_seed_write_outputs_schema_valid_report(tmp_path: Path) -> None:
    report = _build()
    paths = write_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert parsed["status"] == "ready"
    assert "Positive Section/Paragraph Seed" in markdown
    assert validate_payload(
        parsed,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_SECTION_PARAGRAPH_SEED_SCHEMA_ID,
        strict=True,
    ).ok


def test_positive_seed_mutation_counters_remain_zero() -> None:
    report = _build()
    counts = report["counts"]

    for field in (
        "candidateStoreWriteRows",
        "sourceSpanCreatedRows",
        "strictEvidenceRows",
        "citationGradeEvidenceRows",
        "runtimeEvidenceRows",
        "parserExecutionRows",
        "databaseMutationRows",
        "indexMutationRows",
        "reindexOrReembedRows",
        "canonicalParsedArtifactWriteRows",
        "vaultScanRows",
        "externalDownloadRows",
        "publicCliFlagRows",
        "defaultOnRows",
        "externalLlmCallRows",
        "modelApiCallRows",
        "judgeModelCallRows",
    ):
        assert counts[field] == 0
