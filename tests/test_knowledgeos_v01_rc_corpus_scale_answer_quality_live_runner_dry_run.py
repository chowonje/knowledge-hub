from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.complex_qa_seed_pack import DEFAULT_CORPUS_MANIFEST, build_complex_qa_seed_pack
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design import (
    DEFAULT_CORPUS_SCALE_GATE_REPORT,
    build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design,
)
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run import (
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DRY_RUN_SCHEMA_ID,
    READY_DECISION,
    build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run,
    write_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _design_report() -> dict[str, Any]:
    corpus_gate = _json(DEFAULT_CORPUS_SCALE_GATE_REPORT)
    return build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design(
        corpus_scale_gate_report=corpus_gate,
        generated_at="2026-05-29T00:00:00Z",
    )


def _seed_report() -> dict[str, Any]:
    return build_complex_qa_seed_pack(corpus_manifest=DEFAULT_CORPUS_MANIFEST, target_paper_count=20)


def _build(**updates: Any) -> dict[str, Any]:
    design = _design_report()
    seed = _seed_report()
    design.update(updates.pop("design_updates", {}))
    seed.update(updates.pop("seed_updates", {}))
    return build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run(
        live_runner_design_report=design,
        seed_pack_report=seed,
        generated_at="2026-05-29T00:00:00Z",
        **updates,
    )


def test_live_runner_dry_run_ready_plans_50_cases_without_invoking_answers() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["nextRecommendedTranche"] == "corpus_scale_answer_quality_live_runner_controlled_execution"
    assert report["dryRunPlan"]["mode"] == "dry_run"
    assert report["dryRunPlan"]["answerExecutionInThisTranche"] is False
    assert report["counts"]["dryRunCaseRows"] == 50
    assert report["counts"]["plannedAnswerPathInvocationRows"] == 50
    assert report["counts"]["plannedScoreAxisRows"] == 5
    assert report["counts"]["plannedScoreRows"] == 250
    assert report["counts"]["plannedAnswerableRows"] == 0
    assert report["counts"]["plannedExpectedNoAnswerRows"] == 17
    assert report["counts"]["plannedBlockedUntilStructuredEvidenceRows"] == 33
    assert report["counts"]["runnerDryRunRows"] == 1
    assert report["counts"]["liveAnswerExecutionRows"] == 0
    assert report["counts"]["answerPathInvokedRows"] == 0
    assert report["counts"]["answerGeneratedRows"] == 0
    assert report["counts"]["answerQualityMeasuredRows"] == 0
    assert report["counts"]["llmCallRows"] == 0
    assert report["counts"]["publicDefaultPromotionReadyRows"] == 0
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["counts"]["schemaViolationCount"] == 0
    assert report["gate"]["dryRunReady"] is True
    assert report["gate"]["liveExecutionAllowedInThisTranche"] is False
    assert report["gate"]["modelOrJudgeCallsAllowedInThisTranche"] is False
    assert report["gate"]["publicDefaultPromotionAllowed"] is False
    assert len(report["caseRows"]) == 50
    first = report["caseRows"][0]
    assert "question" not in first
    assert len(first["questionSha256"]) == 64
    assert first["plannedQueryPlan"]["requiresExplicitPaperIds"] is True
    assert first["plannedQueryPlan"]["surface"] == "labs_internal_only"
    assert validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_live_runner_dry_run_blocks_when_design_not_ready() -> None:
    report = _build(design_updates={"status": "blocked"})

    assert report["status"] == "blocked"
    assert "live_runner_design_not_ready" in report["gate"]["semanticViolations"]


def test_live_runner_dry_run_blocks_when_design_next_tranche_is_wrong() -> None:
    report = _build(design_updates={"nextRecommendedTranche": "unexpected_next_tranche"})

    assert report["status"] == "blocked"
    assert "live_runner_design_next_tranche_not_dry_run" in report["gate"]["semanticViolations"]


def test_live_runner_dry_run_blocks_when_seed_has_less_than_50_questions() -> None:
    seed = _seed_report()
    seed["counts"] = deepcopy(seed["counts"])
    seed["counts"]["questionRows"] = 49

    report = build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run(
        live_runner_design_report=_design_report(),
        seed_pack_report=seed,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert "complex_qa_seed_pack_less_than_50_questions" in report["gate"]["semanticViolations"]


def test_live_runner_dry_run_blocks_unsafe_design_counter() -> None:
    design = _design_report()
    design["counts"] = deepcopy(design["counts"])
    design["counts"]["answerGeneratedRows"] = 1

    report = build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run(
        live_runner_design_report=design,
        seed_pack_report=_seed_report(),
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert "unsafe_counter_nonzero:design:answerGeneratedRows" in report["gate"]["semanticViolations"]


def test_live_runner_dry_run_blocks_private_path_marker() -> None:
    design = _design_report()
    marker = "/" + "Users" + "/example/private"
    design["warnings"] = list(design.get("warnings") or []) + [marker]

    report = build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run(
        live_runner_design_report=design,
        seed_pack_report=_seed_report(),
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "corpus_scale_live_runner_dry_run_private_path_marker" in report["gate"]["semanticViolations"]


def test_live_runner_dry_run_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = _build()

    paths = write_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert parsed["status"] == "ready"
    assert markdown.startswith("# KnowledgeOS v0.1 RC Corpus-Scale Answer Quality Live Runner Dry Run")
    assert validate_payload(
        parsed,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok
