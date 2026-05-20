from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.complex_qa_abstain_baseline_runner import build_complex_qa_abstain_baseline
from knowledge_hub.papers.complex_qa_real_strict_evidence_availability_bridge_audit import (
    COMPLEX_QA_REAL_STRICT_EVIDENCE_AVAILABILITY_BRIDGE_AUDIT_SCHEMA_ID,
    NEXT_RECOMMENDED_TRANCHE,
    build_complex_qa_real_strict_evidence_availability_bridge_audit,
    write_complex_qa_real_strict_evidence_availability_bridge_audit_reports,
)
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
    build_complex_qa_supplied_strict_evidence_grader_baseline,
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


def _write_reports(
    tmp_path: Path,
    *,
    with_real_evidence: bool = False,
    expected_no_answer_evidence: bool = False,
) -> tuple[Path, Path, dict | None]:
    manifest = _write_manifest(tmp_path / "input", count=19)
    seed_pack = build_complex_qa_seed_pack(corpus_manifest=manifest)
    seed_path = tmp_path / "complex-paper-qa-seed-pack.json"
    seed_path.write_text(json.dumps(seed_pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    abstain = build_complex_qa_abstain_baseline(seed_pack_report=seed_path)
    abstain_path = tmp_path / "complex-qa-abstain-baseline-runner.json"
    abstain_path.write_text(json.dumps(abstain, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    target_question = None
    evidence_path: Path | None = None
    if with_real_evidence or expected_no_answer_evidence:
        target_question = next(
            question
            for question in seed_pack["questions"]
            if question["answerabilityExpectation"]
            == ("expected_no_answer" if expected_no_answer_evidence else "blocked_until_structured_evidence")
        )
        evidence_path = tmp_path / "real-strict-evidence-availability.json"
        evidence_path.write_text(
            json.dumps(
                {
                    "schema": "knowledge-hub.paper.test-real-strict-evidence-availability.v1",
                    "rows": [
                        {
                            "questionId": target_question["questionId"],
                            "evidenceContractSatisfied": True,
                            "structuredEvidenceRefs": [
                                f"strict-evidence://real/{target_question['questionId']}/source-span-001"
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
        abstain_baseline_report=abstain_path,
        structured_evidence_report=evidence_path,
    )
    comparison_path = tmp_path / "complex-qa-structured-evidence-comparison-runner.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    baseline_comparison = build_complex_qa_structured_evidence_comparison(
        seed_pack_report=seed_path,
        abstain_baseline_report=abstain_path,
    )
    baseline_comparison_path = tmp_path / "complex-qa-structured-evidence-comparison-runner-baseline.json"
    baseline_comparison_path.write_text(
        json.dumps(baseline_comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dry_run = build_complex_qa_strict_evidence_answer_quality_dry_run(comparison_report=baseline_comparison_path)
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
    grader_baseline = build_complex_qa_supplied_strict_evidence_grader_baseline(
        synthetic_fixture_report=fixture_path
    )
    grader_baseline_path = tmp_path / "complex-qa-supplied-strict-evidence-grader-baseline-runner.json"
    grader_baseline_path.write_text(
        json.dumps(grader_baseline, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return comparison_path, grader_baseline_path, target_question


def test_bridge_audit_defaults_to_zero_real_strict_evidence_ready_rows(tmp_path: Path) -> None:
    comparison_path, grader_baseline_path, _target = _write_reports(tmp_path)

    payload = build_complex_qa_real_strict_evidence_availability_bridge_audit(
        comparison_report=comparison_path,
        grader_baseline_report=grader_baseline_path,
    )

    assert payload["schema"] == COMPLEX_QA_REAL_STRICT_EVIDENCE_AVAILABILITY_BRIDGE_AUDIT_SCHEMA_ID
    assert validate_payload(
        payload,
        COMPLEX_QA_REAL_STRICT_EVIDENCE_AVAILABILITY_BRIDGE_AUDIT_SCHEMA_ID,
        strict=True,
    ).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["comparisonQuestionRows"] == 50
    assert payload["counts"]["baselineFixtureRows"] == 9
    assert payload["counts"]["bridgeAuditRows"] == 50
    assert payload["counts"]["syntheticBaselineCoveredQuestionRows"] == 8
    assert payload["counts"]["syntheticStrictEvidenceObservedRows"] == 7
    assert payload["counts"]["syntheticStrictEvidencePromotedToRealRows"] == 0
    assert payload["counts"]["realStrictEvidenceAvailableRows"] == 0
    assert payload["counts"]["realStrictEvidenceRefRows"] == 0
    assert payload["counts"]["readyForFutureRealGradingRows"] == 0
    assert payload["counts"]["expectedNoAnswerRows"] == 17
    assert payload["counts"]["blockedMissingRealStrictEvidenceRows"] == 33
    assert payload["counts"]["notApplicableExpectedNoAnswerRows"] == 17
    assert payload["counts"]["schemaViolationCount"] == 0
    assert payload["counts"]["realStrictEvidenceAvailabilityRate"] == 0.0
    assert payload["counts"]["futureRealGradingReadinessRate"] == 0.0


def test_bridge_audit_real_strict_evidence_row_becomes_future_grading_ready(tmp_path: Path) -> None:
    comparison_path, grader_baseline_path, target_question = _write_reports(tmp_path, with_real_evidence=True)
    assert target_question is not None

    payload = build_complex_qa_real_strict_evidence_availability_bridge_audit(
        comparison_report=comparison_path,
        grader_baseline_report=grader_baseline_path,
    )

    assert validate_payload(
        payload,
        COMPLEX_QA_REAL_STRICT_EVIDENCE_AVAILABILITY_BRIDGE_AUDIT_SCHEMA_ID,
        strict=True,
    ).ok
    assert payload["counts"]["realStrictEvidenceAvailableRows"] == 1
    assert payload["counts"]["realStrictEvidenceRefRows"] == 1
    assert payload["counts"]["readyForFutureRealGradingRows"] == 1
    assert payload["counts"]["blockedMissingRealStrictEvidenceRows"] == 32
    row = next(item for item in payload["rows"] if item["questionId"] == target_question["questionId"])
    assert row["availabilityBridgeStatus"] == "ready_for_future_real_supplied_answer_grading"
    assert row["readyForFutureRealGrading"] is True
    assert row["realStrictEvidenceAvailable"] is True
    assert row["realStrictEvidenceRefs"] == [f"strict-evidence://real/{target_question['questionId']}/source-span-001"]
    assert row["syntheticRefsCountAsRealEvidence"] is False
    assert row["bridgeBlockers"] == []


def test_bridge_audit_expected_no_answer_stays_not_applicable_even_with_real_evidence(tmp_path: Path) -> None:
    comparison_path, grader_baseline_path, target_question = _write_reports(
        tmp_path,
        expected_no_answer_evidence=True,
    )
    assert target_question is not None

    payload = build_complex_qa_real_strict_evidence_availability_bridge_audit(
        comparison_report=comparison_path,
        grader_baseline_report=grader_baseline_path,
    )

    row = next(item for item in payload["rows"] if item["questionId"] == target_question["questionId"])
    assert row["answerabilityExpectation"] == "expected_no_answer"
    assert row["realStrictEvidenceAvailable"] is True
    assert row["availabilityBridgeStatus"] == "not_applicable_expected_no_answer"
    assert row["readyForFutureRealGrading"] is False
    assert row["bridgeBlockers"] == ["expected_no_answer_policy_guard"]
    assert payload["counts"]["readyForFutureRealGradingRows"] == 0


def test_bridge_audit_keeps_report_only_no_mutation_policy(tmp_path: Path) -> None:
    comparison_path, grader_baseline_path, _target = _write_reports(tmp_path)

    payload = build_complex_qa_real_strict_evidence_availability_bridge_audit(
        comparison_report=comparison_path,
        grader_baseline_report=grader_baseline_path,
    )

    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["bridgeAuditOnly"] is True
    assert payload["policy"]["realStrictEvidenceReadOnly"] is True
    assert payload["policy"]["syntheticFixtureRefsReadOnly"] is True
    assert payload["policy"]["syntheticRefsPromotedToRealEvidence"] is False
    assert payload["policy"]["storeScanPerformed"] is False
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
    assert all(row["syntheticRefsCountAsRealEvidence"] is False for row in payload["rows"])
    assert all(row["answerGenerated"] is False for row in payload["rows"])
    assert all(row["answerQualityScoreComputed"] is False for row in payload["rows"])
    assert all(row["answerPathInvoked"] is False for row in payload["rows"])
    assert all(row["databaseMutation"] is False for row in payload["rows"])
    assert all(row["vaultScan"] is False for row in payload["rows"])
    assert all(row["strictEvidenceCreated"] is False for row in payload["rows"])


def test_bridge_audit_writer_outputs_schema_valid_report_and_summary(tmp_path: Path) -> None:
    comparison_path, grader_baseline_path, _target = _write_reports(tmp_path)
    payload = build_complex_qa_real_strict_evidence_availability_bridge_audit(
        comparison_report=comparison_path,
        grader_baseline_report=grader_baseline_path,
    )

    report_paths = write_complex_qa_real_strict_evidence_availability_bridge_audit_reports(
        payload,
        tmp_path / "reports",
    )

    assert set(report_paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(report_paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(report_paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(report_paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(
        report,
        COMPLEX_QA_REAL_STRICT_EVIDENCE_AVAILABILITY_BRIDGE_AUDIT_SCHEMA_ID,
        strict=True,
    ).ok
    assert summary["counts"]["bridgeAuditRows"] == 50
    assert summary["nextRecommendedTranche"] == NEXT_RECOMMENDED_TRANCHE
    assert "Report-only bridge audit" in markdown
