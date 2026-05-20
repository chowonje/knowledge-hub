from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.complex_qa_abstain_baseline_runner import build_complex_qa_abstain_baseline
from knowledge_hub.papers.complex_qa_seed_pack import build_complex_qa_seed_pack
from knowledge_hub.papers.complex_qa_strict_evidence_answer_quality_dry_run import (
    build_complex_qa_strict_evidence_answer_quality_dry_run,
)
from knowledge_hub.papers.complex_qa_strict_evidence_answer_quality_grader_design import (
    COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_GRADER_DESIGN_SCHEMA_ID,
    NEXT_RECOMMENDED_TRANCHE,
    build_complex_qa_strict_evidence_answer_quality_grader_design,
    write_complex_qa_strict_evidence_answer_quality_grader_design_reports,
)
from knowledge_hub.papers.complex_qa_structured_evidence_comparison_runner import (
    build_complex_qa_structured_evidence_comparison,
)


def _write_manifest(root: Path, count: int = 19) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    seed_ids = [
        "2005.11401",
        "2007.01282",
        "2404.16130",
        "2410.05779",
        "alexnet-2012",
        "2010.11929",
        "1706.03762",
        "2312.00752",
        "1810.04805",
        "2005.14165",
        "1312.5602",
        "1707.06347",
        "1406.2661",
        "2006.11239",
        "1512.03385",
        "2310.11511",
        "1502.03167",
        "1409.3215",
        "2201.11903",
        "2501.12948",
    ][:count]
    artifacts = [
        {
            "artifactId": f"paper_{paper_id.replace('.', '_').replace('-', '_')}",
            "sourceIds": [paper_id],
            "expectedFilename": f"Seed Paper {paper_id}.pdf",
            "expectedSourceContentHash": f"sha256:{index:064d}",
            "corpusTier": "local_corpus",
        }
        for index, paper_id in enumerate(seed_ids, start=1)
    ]
    path = root / "corpus_manifest.json"
    path.write_text(json.dumps({"schema": "knowledge-hub.corpus-manifest.v1", "artifacts": artifacts}), encoding="utf-8")
    return path


def _write_seed_and_baseline(tmp_path: Path) -> tuple[Path, Path, dict]:
    manifest = _write_manifest(tmp_path / "input", count=19)
    seed_pack = build_complex_qa_seed_pack(corpus_manifest=manifest)
    seed_path = tmp_path / "complex-paper-qa-seed-pack.json"
    seed_path.write_text(json.dumps(seed_pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    baseline = build_complex_qa_abstain_baseline(seed_pack_report=seed_path)
    baseline_path = tmp_path / "complex-qa-abstain-baseline-runner.json"
    baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return seed_path, baseline_path, seed_pack


def _write_dry_run(tmp_path: Path, *, with_strict_evidence: bool = False) -> tuple[Path, dict | None]:
    seed_path, baseline_path, seed_pack = _write_seed_and_baseline(tmp_path)
    evidence_path: Path | None = None
    target_question = None
    if with_strict_evidence:
        target_question = next(
            question
            for question in seed_pack["questions"]
            if question["answerabilityExpectation"] == "blocked_until_structured_evidence"
        )
        evidence_path = tmp_path / "structured-evidence-availability.json"
        evidence_path.write_text(
            json.dumps(
                {
                    "schema": "knowledge-hub.paper.test-structured-evidence-availability.v1",
                    "rows": [
                        {
                            "questionId": target_question["questionId"],
                            "evidenceContractSatisfied": True,
                            "structuredEvidenceRefs": [
                                f"strict-evidence://question/{target_question['questionId']}/source-span-001"
                            ],
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    comparison = build_complex_qa_structured_evidence_comparison(
        seed_pack_report=seed_path,
        abstain_baseline_report=baseline_path,
        structured_evidence_report=evidence_path,
    )
    comparison_path = tmp_path / "complex-qa-structured-evidence-comparison-runner.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dry_run = build_complex_qa_strict_evidence_answer_quality_dry_run(comparison_report=comparison_path)
    dry_run_path = tmp_path / "complex-qa-strict-evidence-answer-quality-dry-run.json"
    dry_run_path.write_text(json.dumps(dry_run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dry_run_path, target_question


def test_grader_design_defaults_to_zero_future_grading_rows_without_strict_evidence(tmp_path: Path) -> None:
    dry_run_path, _target_question = _write_dry_run(tmp_path, with_strict_evidence=False)

    payload = build_complex_qa_strict_evidence_answer_quality_grader_design(dry_run_report=dry_run_path)

    assert payload["schema"] == COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_GRADER_DESIGN_SCHEMA_ID
    assert validate_payload(payload, COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_GRADER_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["questionRows"] == 50
    assert payload["counts"]["dryRunRows"] == 50
    assert payload["counts"]["rubricDesignRows"] == 50
    assert payload["counts"]["readyForFutureGradingRows"] == 0
    assert payload["counts"]["expectedNoAnswerRows"] == 17
    assert payload["counts"]["blockedMissingStrictEvidenceRows"] == 33
    assert payload["counts"]["schemaViolationCount"] == 0
    assert payload["counts"]["readyForFutureGradingRate"] == 0.0
    table_row = next(row for row in payload["rows"] if row["questionCategory"] == "table_numeric_qa")
    assert "exact_cell_value_check" in table_row["categorySpecificChecks"]
    assert "no_rounding_drift_check" in table_row["categorySpecificChecks"]


def test_grader_design_marks_synthetic_strict_evidence_ready_row_eligible(tmp_path: Path) -> None:
    dry_run_path, target_question = _write_dry_run(tmp_path, with_strict_evidence=True)
    assert target_question is not None

    payload = build_complex_qa_strict_evidence_answer_quality_grader_design(dry_run_report=dry_run_path)

    assert validate_payload(payload, COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_GRADER_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["readyForFutureGradingRows"] == 1
    assert payload["counts"]["blockedMissingStrictEvidenceRows"] == 32
    row = next(item for item in payload["rows"] if item["questionId"] == target_question["questionId"])
    assert row["graderExecutionExpectation"] == "eligible_for_future_grading"
    assert row["readyForAnswerQualityComparison"] is True
    assert row["structuredEvidenceRefs"]
    assert "future_candidate_answer_text" in row["requiredGraderInputs"]
    assert "future_answer_citation_map" in row["requiredGraderInputs"]
    assert row["plannedScoringContract"]["scoreComputedNow"] is False


def test_grader_design_never_executes_expected_no_answer_rows_even_if_ready_flag_is_bad(tmp_path: Path) -> None:
    dry_run_path, _target_question = _write_dry_run(tmp_path, with_strict_evidence=False)
    dry_run = json.loads(dry_run_path.read_text(encoding="utf-8"))
    expected_no_answer_row = next(row for row in dry_run["rows"] if row["answerabilityExpectation"] == "expected_no_answer")
    expected_no_answer_row["readyForAnswerQualityComparison"] = True
    expected_no_answer_row["structuredEvidenceRefs"] = ["strict-evidence://bad/expected-no-answer"]
    dry_run_path.write_text(json.dumps(dry_run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    payload = build_complex_qa_strict_evidence_answer_quality_grader_design(dry_run_report=dry_run_path)

    row = next(item for item in payload["rows"] if item["questionId"] == expected_no_answer_row["questionId"])
    assert row["graderExecutionExpectation"] == "not_run_expected_no_answer"
    assert row["readyForAnswerQualityComparison"] is False
    assert payload["counts"]["readyForFutureGradingRows"] == 0


def test_grader_design_keeps_report_only_no_mutation_policy(tmp_path: Path) -> None:
    dry_run_path, _target_question = _write_dry_run(tmp_path, with_strict_evidence=False)

    payload = build_complex_qa_strict_evidence_answer_quality_grader_design(dry_run_report=dry_run_path)

    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["designOnly"] is True
    assert payload["policy"]["graderDesignOnly"] is True
    assert payload["policy"]["strictEvidenceReadOnly"] is True
    assert payload["policy"]["questionExecutionRun"] is False
    assert payload["policy"]["answerGenerationRun"] is False
    assert payload["policy"]["answerQualityScoringRun"] is False
    assert payload["policy"]["scoreComputed"] is False
    assert payload["policy"]["llmCalls"] is False
    assert payload["policy"]["judgeModelCalls"] is False
    assert payload["policy"]["answerPathChanged"] is False
    assert payload["policy"]["answerPathInvoked"] is False
    assert payload["policy"]["searchIndexQueried"] is False
    assert payload["policy"]["databaseMutation"] is False
    assert payload["policy"]["indexMutation"] is False
    assert payload["policy"]["reindexOrReembed"] is False
    assert payload["policy"]["vaultScan"] is False
    assert payload["policy"]["runtimeEvidenceCreated"] is False
    assert payload["policy"]["citationEvidenceCreated"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["canonicalParsedArtifactsWritten"] is False
    assert payload["counts"]["answerGeneratedRows"] == 0
    assert payload["counts"]["scoreComputedRows"] == 0
    assert payload["counts"]["llmCallRows"] == 0
    assert payload["counts"]["judgeModelCallRows"] == 0
    assert payload["counts"]["answerPathInvokedRows"] == 0
    assert payload["counts"]["databaseMutationRows"] == 0
    assert payload["counts"]["indexMutationRows"] == 0
    assert payload["counts"]["vaultScanRows"] == 0
    assert payload["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert payload["counts"]["citationEvidenceCreatedRows"] == 0
    assert payload["counts"]["strictEvidenceCreatedRows"] == 0


def test_grader_design_writer_outputs_schema_valid_report_and_summary(tmp_path: Path) -> None:
    dry_run_path, _target_question = _write_dry_run(tmp_path, with_strict_evidence=False)
    payload = build_complex_qa_strict_evidence_answer_quality_grader_design(dry_run_report=dry_run_path)

    report_paths = write_complex_qa_strict_evidence_answer_quality_grader_design_reports(payload, tmp_path / "reports")

    assert set(report_paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(report_paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(report_paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(report_paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_GRADER_DESIGN_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["questionRows"] == 50
    assert summary["nextRecommendedTranche"] == NEXT_RECOMMENDED_TRANCHE
    assert "Report-only grader design" in markdown
