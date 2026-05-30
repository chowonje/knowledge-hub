from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution import (
    BLOCKED_DECISION,
    DEFAULT_POSITIVE_EXECUTION_REPORT,
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_REFRESH_AFTER_POSITIVE_ANSWER_EXECUTION_SCHEMA_ID,
    READY_DECISION,
    build_knowledgeos_v01_rc_corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution,
    write_knowledgeos_v01_rc_corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution,
)


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _positive_execution_report() -> dict[str, Any]:
    return _read(DEFAULT_POSITIVE_EXECUTION_REPORT)


def _fake_provenance_row(
    *,
    execution_row: dict[str, Any],
    seed_question: dict[str, Any],
    papers_dir: str | Path,
) -> dict[str, Any]:
    _ = seed_question, papers_dir
    paper_ids = [str(item) for item in list(execution_row.get("paperIds") or [])]
    min_required = max(2, len(paper_ids) * 2)
    span_rows: list[dict[str, Any]] = []
    for index in range(min_required):
        paper_id = paper_ids[index % len(paper_ids)] if paper_ids else "paper"
        span_rows.append(
            {
                "spanRef": f"span:{index + 1}",
                "sourceId": paper_id,
                "sourceRef": f"papers_dir/parsed/{paper_id}/document.md",
                "sourceContentHash": "sha256:" + str(index % 10) * 64,
                "sourceContentHashAvailable": True,
                "spanLocator": f"chars:{index * 100}-{index * 100 + 80}",
                "spanOffsetAvailable": True,
                "charStart": index * 100,
                "charEnd": index * 100 + 80,
                "evidenceKind": "parsed_artifact_evidence_chunk",
                "candidateRecordId": f"candidate:{paper_id}:{index}",
                "candidateStoreRef": f"papers_dir/structured_evidence_candidates/evidence_chunk/{paper_id}.jsonl",
                "strictProvenance": True,
                "excerptIncludedInReport": False,
            }
        )
    return {
        "caseIndex": int(execution_row.get("caseIndex") or 0),
        "caseId": str(execution_row.get("caseId") or ""),
        "questionCategory": str(execution_row.get("questionCategory") or ""),
        "paperIds": paper_ids,
        "questionSha256": str(execution_row.get("questionSha256") or ""),
        "observedStatus": "ok",
        "localFakeLlmCallRows": 1,
        "observedSourceIds": paper_ids,
        "missingSourceIds": [],
        "minRequiredProvenanceRows": min_required,
        "evidencePacketContractSpanRows": min_required,
        "strictProvenanceSpanRows": min_required,
        "sourceContentHashRows": min_required,
        "charsLocatorRows": min_required,
        "answerContractCitationRows": min_required,
        "answerContractCitationProvenanceRows": min_required,
        "dimensionStatuses": {
            "answerPayloadStatusOk": True,
            "evidencePacketContractPresent": True,
            "answerContractPresent": True,
            "sourceCoverage": True,
            "sourceHashCoverage": True,
            "charsLocatorCoverage": True,
            "answerContractCitationProvenance": True,
            "publicDefaultHeld": True,
        },
        "qualityScore": 1.0,
        "qualityGrade": "pass",
        "pass": True,
        "spanProofRows": span_rows,
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "failureReasons": [],
    }


def _build(**updates: Any) -> dict[str, Any]:
    return build_knowledgeos_v01_rc_corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution(
        positive_execution_report=updates.pop("positive_execution_report", _positive_execution_report()),
        execute_case=updates.pop("execute_case", _fake_provenance_row),
        generated_at="2026-05-30T00:00:00Z",
        **updates,
    )


def test_positive_provenance_refresh_ready_for_seven_rows() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["nextRecommendedTranche"] == "knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review"
    assert report["counts"]["inputPositiveExecutionRows"] == 7
    assert report["counts"]["attemptedProvenanceRows"] == 7
    assert report["counts"]["provenancePassRows"] == 7
    assert report["counts"]["provenanceFailRows"] == 0
    assert report["counts"]["strictProvenanceSpanRows"] == 20
    assert report["counts"]["sourceContentHashRows"] == 20
    assert report["counts"]["charsLocatorRows"] == 20
    assert report["counts"]["answerContractCitationProvenanceRows"] == 20
    assert report["counts"]["heldExpectedNoAnswerRows"] == 17
    assert report["counts"]["heldStructuredModalityRows"] == 34
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["counts"]["schemaViolationCount"] == 0
    assert report["gate"]["positiveSectionParagraphQualityComplete"] is True
    assert validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_REFRESH_AFTER_POSITIVE_ANSWER_EXECUTION_SCHEMA_ID,
        strict=True,
    ).ok


def test_positive_provenance_refresh_blocks_when_execution_report_not_ready() -> None:
    positive = _positive_execution_report()
    positive["status"] = "blocked"

    report = _build(positive_execution_report=positive)

    assert report["status"] == "blocked"
    assert report["decision"] == BLOCKED_DECISION
    assert "positive_answer_execution_not_ready" in report["gate"]["semanticViolations"]


def test_positive_provenance_refresh_blocks_when_execution_fail_rows_present() -> None:
    positive = deepcopy(_positive_execution_report())
    positive["counts"]["positiveAnswerFailRows"] = 1

    report = _build(positive_execution_report=positive)

    assert report["status"] == "blocked"
    assert "positive_answer_failures_present" in report["gate"]["semanticViolations"]


def test_positive_provenance_refresh_blocks_row_without_strict_provenance() -> None:
    def missing_provenance(
        *,
        execution_row: dict[str, Any],
        seed_question: dict[str, Any],
        papers_dir: str | Path,
    ) -> dict[str, Any]:
        row = _fake_provenance_row(execution_row=execution_row, seed_question=seed_question, papers_dir=papers_dir)
        if row["caseId"] == "complex-paper-qa-seed-20260520-q028":
            row["spanProofRows"][0]["strictProvenance"] = False
            row["strictProvenanceSpanRows"] = row["minRequiredProvenanceRows"] - 1
            row["dimensionStatuses"]["sourceHashCoverage"] = False
            row["qualityScore"] = 0.875
            row["qualityGrade"] = "partial"
            row["pass"] = False
            row["failureReasons"] = ["strict_provenance_span_rows_below_minimum"]
        return row

    report = _build(execute_case=missing_provenance)

    assert report["status"] == "blocked"
    assert report["counts"]["provenanceFailRows"] == 1
    assert "provenance_fail_rows:1" in report["gate"]["semanticViolations"]


def test_positive_provenance_refresh_blocks_when_pass_rows_below_minimum() -> None:
    def fail_all(
        *,
        execution_row: dict[str, Any],
        seed_question: dict[str, Any],
        papers_dir: str | Path,
    ) -> dict[str, Any]:
        row = _fake_provenance_row(execution_row=execution_row, seed_question=seed_question, papers_dir=papers_dir)
        row["dimensionStatuses"]["answerContractCitationProvenance"] = False
        row["qualityScore"] = 0.875
        row["qualityGrade"] = "partial"
        row["pass"] = False
        row["failureReasons"] = ["answer_contract_citation_provenance_below_minimum"]
        return row

    report = _build(execute_case=fail_all)

    assert report["status"] == "blocked"
    assert report["counts"]["provenancePassRows"] == 0
    assert "provenance_pass_rows_below_minimum" in report["gate"]["semanticViolations"]


def test_positive_provenance_refresh_blocks_private_path_marker() -> None:
    def leaky_row(
        *,
        execution_row: dict[str, Any],
        seed_question: dict[str, Any],
        papers_dir: str | Path,
    ) -> dict[str, Any]:
        row = _fake_provenance_row(execution_row=execution_row, seed_question=seed_question, papers_dir=papers_dir)
        row["spanProofRows"][0]["sourceRef"] = "/" + "Users" + "/example/private/document.md"
        return row

    report = _build(execute_case=leaky_row)

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] >= 1
    assert "positive_provenance_refresh_private_path_marker" in report["gate"]["semanticViolations"]


def test_positive_provenance_refresh_blocks_unsafe_counter_in_input() -> None:
    positive = deepcopy(_positive_execution_report())
    positive["counts"]["databaseMutationRows"] = 1

    report = _build(positive_execution_report=positive)

    assert report["status"] == "blocked"
    assert "unsafe_counter_nonzero:positive_execution:databaseMutationRows" in report["gate"]["semanticViolations"]


def test_positive_provenance_refresh_rows_exclude_raw_payloads() -> None:
    report = _build()

    forbidden_keys = {"question", "answer", "citations", "sources", "excerpt", "quote", "text"}
    for row in report["provenanceRows"]:
        assert forbidden_keys.isdisjoint(row)
        assert row["answerTextIncludedInReport"] is False
        assert row["citationPayloadIncludedInReport"] is False
        assert row["sourcePayloadIncludedInReport"] is False
        assert row["excerptIncludedInReport"] is False
        for span in row["spanProofRows"]:
            assert forbidden_keys.isdisjoint(span)
            assert span["excerptIncludedInReport"] is False
            assert span["sourceContentHash"].startswith("sha256:")
            assert span["spanLocator"].startswith("chars:")


def test_positive_provenance_refresh_write_report_validates_schema(tmp_path: Path) -> None:
    report = _build()
    paths = write_knowledgeos_v01_rc_corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )
    loaded = _read(paths["json"])

    assert validate_payload(
        loaded,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_REFRESH_AFTER_POSITIVE_ANSWER_EXECUTION_SCHEMA_ID,
        strict=True,
    ).ok
    assert "strictProvenanceSpanRows" in Path(paths["markdown"]).read_text(encoding="utf-8")
