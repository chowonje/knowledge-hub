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
    build_complex_qa_strict_evidence_answer_quality_grader_design,
)
from knowledge_hub.papers.complex_qa_structured_evidence_comparison_runner import (
    build_complex_qa_structured_evidence_comparison,
)
from knowledge_hub.papers.complex_qa_supplied_strict_evidence_grader_baseline_runner import (
    COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_GRADER_BASELINE_RUNNER_SCHEMA_ID,
    NEXT_RECOMMENDED_TRANCHE,
    build_complex_qa_supplied_strict_evidence_grader_baseline,
    write_complex_qa_supplied_strict_evidence_grader_baseline_reports,
)
from knowledge_hub.papers.complex_qa_supplied_strict_evidence_synthetic_grading_fixture_runner import (
    build_complex_qa_supplied_strict_evidence_synthetic_grading_fixture_runner,
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


def _write_synthetic_fixture_report(tmp_path: Path) -> Path:
    manifest = _write_manifest(tmp_path / "input", count=19)
    seed_pack = build_complex_qa_seed_pack(corpus_manifest=manifest)
    seed_path = tmp_path / "complex-paper-qa-seed-pack.json"
    seed_path.write_text(json.dumps(seed_pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    baseline = build_complex_qa_abstain_baseline(seed_pack_report=seed_path)
    baseline_path = tmp_path / "complex-qa-abstain-baseline-runner.json"
    baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    comparison = build_complex_qa_structured_evidence_comparison(
        seed_pack_report=seed_path,
        abstain_baseline_report=baseline_path,
    )
    comparison_path = tmp_path / "complex-qa-structured-evidence-comparison-runner.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dry_run = build_complex_qa_strict_evidence_answer_quality_dry_run(comparison_report=comparison_path)
    dry_run_path = tmp_path / "complex-qa-strict-evidence-answer-quality-dry-run.json"
    dry_run_path.write_text(json.dumps(dry_run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    design = build_complex_qa_strict_evidence_answer_quality_grader_design(dry_run_report=dry_run_path)
    design_path = tmp_path / "complex-qa-strict-evidence-answer-quality-grader-design.json"
    design_path.write_text(json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fixture = build_complex_qa_supplied_strict_evidence_synthetic_grading_fixture_runner(
        grader_design_report=design_path
    )
    fixture_path = tmp_path / "complex-qa-supplied-strict-evidence-synthetic-grading-fixture-runner.json"
    fixture_path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return fixture_path


def test_grader_baseline_default_counts_and_schema(tmp_path: Path) -> None:
    fixture_path = _write_synthetic_fixture_report(tmp_path)

    payload = build_complex_qa_supplied_strict_evidence_grader_baseline(
        synthetic_fixture_report=fixture_path
    )

    assert payload["schema"] == COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_GRADER_BASELINE_RUNNER_SCHEMA_ID
    assert validate_payload(
        payload,
        COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_GRADER_BASELINE_RUNNER_SCHEMA_ID,
        strict=True,
    ).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["inputFixtureRows"] == 9
    assert payload["counts"]["baselineGradeRows"] == 9
    assert payload["counts"]["baselineVerdictComputedRows"] == 9
    assert payload["counts"]["baselinePassRows"] == 5
    assert payload["counts"]["baselineFailRows"] == 3
    assert payload["counts"]["baselineBlockedRows"] == 1
    assert payload["counts"]["expectedPassRows"] == 5
    assert payload["counts"]["expectedFailRows"] == 3
    assert payload["counts"]["expectedBlockedRows"] == 1
    assert payload["counts"]["baselineExpectedMatchRows"] == 9
    assert payload["counts"]["baselineExpectedMismatchRows"] == 0
    assert payload["counts"]["baselineSourceFixtureMatchRows"] == 9
    assert payload["counts"]["baselineSourceFixtureMismatchRows"] == 0
    assert payload["counts"]["suppliedAnswerRows"] == 8
    assert payload["counts"]["suppliedCitationRows"] == 7
    assert payload["counts"]["syntheticStrictEvidenceRefRows"] == 8
    assert payload["counts"]["schemaViolationCount"] == 0
    assert payload["counts"]["baselineExpectedMatchRate"] == 1.0
    assert payload["counts"]["baselineSourceFixtureMatchRate"] == 1.0


def test_grader_baseline_recomputes_negative_fixture_guards(tmp_path: Path) -> None:
    fixture_path = _write_synthetic_fixture_report(tmp_path)

    payload = build_complex_qa_supplied_strict_evidence_grader_baseline(
        synthetic_fixture_report=fixture_path
    )

    by_kind = {
        row["fixtureKind"]: row
        for row in payload["rows"]
        if row["fixtureKind"] != "synthetic_good_strict_evidence_answer"
    }
    assert by_kind["synthetic_unsupported_claim"]["baselineGraderVerdict"] == "fail"
    assert by_kind["synthetic_unsupported_claim"]["baselineFailureReasons"] == ["unsupported_claim_present"]
    assert by_kind["synthetic_missing_citation"]["baselineGraderVerdict"] == "fail"
    assert by_kind["synthetic_missing_citation"]["baselineFailureReasons"] == ["missing_citation_coverage"]
    assert by_kind["synthetic_missing_strict_evidence"]["baselineGraderVerdict"] == "blocked"
    assert by_kind["synthetic_missing_strict_evidence"]["baselineFailureReasons"] == [
        "missing_strict_structured_evidence"
    ]
    assert by_kind["synthetic_expected_no_answer_answered"]["baselineGraderVerdict"] == "fail"
    assert by_kind["synthetic_expected_no_answer_answered"]["baselineFailureReasons"] == [
        "expected_no_answer_question_answered"
    ]


def test_grader_baseline_keeps_expected_no_answer_rows_non_passing(tmp_path: Path) -> None:
    fixture_path = _write_synthetic_fixture_report(tmp_path)

    payload = build_complex_qa_supplied_strict_evidence_grader_baseline(
        synthetic_fixture_report=fixture_path
    )

    expected_no_answer_rows = [
        row for row in payload["rows"] if row["answerabilityExpectation"] == "expected_no_answer"
    ]
    assert len(expected_no_answer_rows) == 1
    assert expected_no_answer_rows[0]["baselineGraderVerdict"] == "fail"
    assert expected_no_answer_rows[0]["baselineVerdictMatchedExpected"] is True
    assert expected_no_answer_rows[0]["baselineGradeContract"]["realAnswerQualityScoreComputed"] is False


def test_grader_baseline_keeps_no_llm_no_answer_path_no_mutation_policy(tmp_path: Path) -> None:
    fixture_path = _write_synthetic_fixture_report(tmp_path)

    payload = build_complex_qa_supplied_strict_evidence_grader_baseline(
        synthetic_fixture_report=fixture_path
    )

    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["baselineOnly"] is True
    assert payload["policy"]["deterministicGraderBaselineOnly"] is True
    assert payload["policy"]["suppliedFixtureOnly"] is True
    assert payload["policy"]["syntheticStrictEvidenceRefsOnly"] is True
    assert payload["policy"]["strictEvidenceReadOnly"] is True
    assert payload["policy"]["deterministicBaselineVerdictRun"] is True
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
    assert payload["counts"]["answerQualityScoreComputedRows"] == 0
    assert payload["counts"]["scoreComputedRows"] == 0
    assert payload["counts"]["llmCallRows"] == 0
    assert payload["counts"]["judgeModelCallRows"] == 0
    assert payload["counts"]["answerPathInvokedRows"] == 0
    assert payload["counts"]["searchIndexQueriedRows"] == 0
    assert payload["counts"]["databaseMutationRows"] == 0
    assert payload["counts"]["indexMutationRows"] == 0
    assert payload["counts"]["vaultScanRows"] == 0
    assert payload["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert payload["counts"]["citationEvidenceCreatedRows"] == 0
    assert payload["counts"]["strictEvidenceCreatedRows"] == 0
    assert all(row["answerGenerated"] is False for row in payload["rows"])
    assert all(row["answerQualityScoreComputed"] is False for row in payload["rows"])
    assert all(row["llmCall"] is False for row in payload["rows"])
    assert all(row["judgeModelCall"] is False for row in payload["rows"])
    assert all(row["answerPathInvoked"] is False for row in payload["rows"])
    assert all(row["databaseMutation"] is False for row in payload["rows"])
    assert all(row["vaultScan"] is False for row in payload["rows"])
    assert all(row["strictEvidenceCreated"] is False for row in payload["rows"])


def test_grader_baseline_writer_outputs_schema_valid_report_and_summary(tmp_path: Path) -> None:
    fixture_path = _write_synthetic_fixture_report(tmp_path)
    payload = build_complex_qa_supplied_strict_evidence_grader_baseline(
        synthetic_fixture_report=fixture_path
    )

    report_paths = write_complex_qa_supplied_strict_evidence_grader_baseline_reports(
        payload,
        tmp_path / "reports",
    )

    assert set(report_paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(report_paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(report_paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(report_paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(
        report,
        COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_GRADER_BASELINE_RUNNER_SCHEMA_ID,
        strict=True,
    ).ok
    assert summary["counts"]["baselineGradeRows"] == 9
    assert summary["nextRecommendedTranche"] == NEXT_RECOMMENDED_TRANCHE
    assert "Report-only deterministic grader baseline" in markdown
