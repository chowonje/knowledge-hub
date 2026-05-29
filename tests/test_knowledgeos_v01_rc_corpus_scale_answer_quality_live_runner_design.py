from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design import (
    DEFAULT_CORPUS_SCALE_GATE_REPORT,
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DESIGN_SCHEMA_ID,
    READY_DECISION,
    build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design,
    write_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate_report() -> dict[str, Any]:
    return _json(DEFAULT_CORPUS_SCALE_GATE_REPORT)


def _build(**updates: Any) -> dict[str, Any]:
    gate = _gate_report()
    gate.update(updates)
    return build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design(
        corpus_scale_gate_report=gate,
        generated_at="2026-05-29T00:00:00Z",
    )


def test_live_runner_design_ready_from_held_corpus_scale_gate() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["nextRecommendedTranche"] == "corpus_scale_answer_quality_live_runner_dry_run"
    assert report["runnerDesign"]["mode"] == "design_only"
    assert report["runnerDesign"]["executionSurface"] == "labs_internal_only"
    assert report["counts"]["seedPaperRows"] == 20
    assert report["counts"]["seedQuestionRows"] == 50
    assert report["counts"]["plannedRunnerPaperRows"] == 20
    assert report["counts"]["plannedRunnerQuestionRows"] == 50
    assert report["counts"]["plannedExecutionPhaseRows"] == 5
    assert report["counts"]["plannedScoreAxisRows"] == 5
    assert report["counts"]["liveAnswerExecutionRows"] == 0
    assert report["counts"]["answerPathInvokedRows"] == 0
    assert report["counts"]["answerGeneratedRows"] == 0
    assert report["counts"]["answerQualityMeasuredRows"] == 0
    assert report["counts"]["publicDefaultPromotionReadyRows"] == 0
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["counts"]["schemaViolationCount"] == 0
    assert report["gate"]["designReady"] is True
    assert report["gate"]["liveExecutionAllowedInThisTranche"] is False
    assert report["gate"]["modelOrJudgeCallsAllowedInThisTranche"] is False
    assert report["gate"]["publicDefaultPromotionAllowed"] is False
    assert validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok


def test_live_runner_design_blocks_when_corpus_gate_not_ready() -> None:
    report = _build(status="blocked")

    assert report["status"] == "blocked"
    assert "corpus_scale_answer_quality_gate_not_ready" in report["gate"]["semanticViolations"]


def test_live_runner_design_blocks_when_gate_is_not_held() -> None:
    report = _build(decision="knowledgeos_v01_rc_corpus_scale_answer_quality_gate_ready")

    assert report["status"] == "blocked"
    assert "corpus_scale_answer_quality_gate_not_held" in report["gate"]["semanticViolations"]


def test_live_runner_design_blocks_when_public_default_is_marked_ready() -> None:
    gate = _gate_report()
    gate["counts"] = deepcopy(gate["counts"])
    gate["counts"]["publicDefaultPromotionReadyRows"] = 1

    report = build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design(
        corpus_scale_gate_report=gate,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert "public_default_promotion_ready_unexpected" in report["gate"]["semanticViolations"]


def test_live_runner_design_blocks_when_live_execution_already_opened() -> None:
    gate = _gate_report()
    gate["counts"] = deepcopy(gate["counts"])
    gate["counts"]["liveAnswerExecutionRows"] = 1

    report = build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design(
        corpus_scale_gate_report=gate,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert "live_answer_execution_already_opened" in report["gate"]["semanticViolations"]
    assert any(
        blocker == "unsafe_counter_nonzero:liveAnswerExecutionRows"
        for blocker in report["gate"]["semanticViolations"]
    )


def test_live_runner_design_blocks_private_path_marker() -> None:
    gate = _gate_report()
    marker = "/" + "Users" + "/example/private"
    gate["warnings"] = list(gate.get("warnings") or []) + [marker]

    report = build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design(
        corpus_scale_gate_report=gate,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "corpus_scale_live_runner_design_private_path_marker" in report["gate"]["semanticViolations"]


def test_live_runner_design_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = _build()

    paths = write_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert parsed["status"] == "ready"
    assert markdown.startswith("# KnowledgeOS v0.1 RC Corpus-Scale Answer Quality Live Runner Design")
    assert validate_payload(
        parsed,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok
