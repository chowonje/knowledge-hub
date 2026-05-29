"""Design gate for the KnowledgeOS v0.1 RC corpus-scale live answer runner."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_gate import (
    HELD_DECISION as CORPUS_GATE_HELD_DECISION,
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed import (
    ZERO_COUNTER_FIELDS,
    _clean_text,
    _contains_private_path,
    _int,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture import (
    _read_json,
)


KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DESIGN_SCHEMA_ID = (
    "knowledge-hub.product.knowledgeos-v01-rc-corpus-scale-answer-quality-live-runner-design.v1"
)

READY_DECISION = "knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design_ready"
BLOCKED_DECISION = "knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design_blocked"
NEXT_TRANCHE_READY = "corpus_scale_answer_quality_live_runner_dry_run"
NEXT_TRANCHE_BLOCKED = "corpus_scale_answer_quality_live_runner_design_input_repair"
DEFAULT_CORPUS_SCALE_GATE_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_corpus_scale_answer_quality_gate.v1.json"
)

EXTRA_ZERO_COUNTER_FIELDS = (
    "liveAnswerExecutionRows",
    "answerPathInvokedRows",
    "answerGeneratedRows",
    "answerQualityMeasuredRows",
    "answerQualityScoreComputedRows",
    "runnerDryRunRows",
    "llmCallRows",
    "externalLlmCallRows",
    "modelApiCallRows",
    "judgeModelCallRows",
    "githubPrMutationRows",
    "mergeRows",
    "branchDeletionRows",
    "releaseTagRows",
    "packagePublishRows",
    "rawPayloadPersistedRows",
    "defaultMcpToolRows",
    "defaultKhubAskRouteRows",
)

REQUIRED_HELD_BLOCKERS = (
    "corpus_scale_live_answer_quality_execution_missing",
    "answer_quality_scores_not_computed",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _zero_counter_fields() -> tuple[str, ...]:
    return tuple(dict.fromkeys((*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS)))


def _schema_blockers(report: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if report.get("schema") != KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_SCHEMA_ID:
        blockers.append("corpus_scale_answer_quality_gate_schema_mismatch")
        return blockers
    validation = validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        blockers.append("corpus_scale_answer_quality_gate_schema_validation_failed")
    return blockers


def _corpus_gate_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    quality_decision = dict(report.get("qualityGateDecision") or {})
    gate = dict(report.get("gate") or {})
    blockers = _schema_blockers(report)

    if report.get("status") != "ready":
        blockers.append("corpus_scale_answer_quality_gate_not_ready")
    if report.get("decision") != CORPUS_GATE_HELD_DECISION:
        blockers.append("corpus_scale_answer_quality_gate_not_held")
    if report.get("nextRecommendedTranche") != "corpus_scale_answer_quality_live_runner_design":
        blockers.append("corpus_scale_answer_quality_gate_next_tranche_not_live_runner_design")
    if quality_decision.get("corpusScaleGate") != "held":
        blockers.append("corpus_scale_answer_quality_gate_decision_not_held")
    if quality_decision.get("publicDefaultDecision") != "hold_public_default_promotion":
        blockers.append("public_default_promotion_not_held")
    for required in REQUIRED_HELD_BLOCKERS:
        if required not in list(quality_decision.get("gateBlockers") or []):
            blockers.append(f"missing_expected_gate_blocker:{required}")

    if _int(counts.get("seedPaperRows")) < 20:
        blockers.append("seed_paper_rows_below_20")
    if _int(counts.get("seedQuestionRows")) < 50:
        blockers.append("seed_question_rows_below_50")
    if _int(counts.get("abstainNoAnswerPassRows")) < _int(counts.get("abstainNoAnswerExpectedRows")):
        blockers.append("abstain_no_answer_pass_rows_below_expected")
    if _int(counts.get("publicDefaultPromotionReadyRows")) != 0:
        blockers.append("public_default_promotion_ready_unexpected")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("public_default_promotion_hold_missing")
    if _int(counts.get("generalRcReadyRows")) != 0:
        blockers.append("general_rc_ready_unexpected")
    if _int(counts.get("liveAnswerExecutionRows")) != 0:
        blockers.append("live_answer_execution_already_opened")
    if _int(counts.get("answerQualityMeasuredRows")) != 0:
        blockers.append("answer_quality_already_measured")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("corpus_scale_answer_quality_gate_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("corpus_scale_answer_quality_gate_schema_violations")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("public_default_promotion_allowed_unexpected")
    if gate.get("generalRcReady") is not False:
        blockers.append("general_rc_ready_gate_unexpected")
    if gate.get("liveAnswerQualityMeasured") is not False:
        blockers.append("live_answer_quality_measured_unexpected")
    return sorted(set(blockers))


def _unsafe_counter_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    for field in _zero_counter_fields():
        if _int(counts.get(field)) != 0:
            blockers.append(f"unsafe_counter_nonzero:{field}")
    return sorted(set(blockers))


def _phase_rows() -> list[dict[str, Any]]:
    return [
        {
            "phaseId": "load_seed_cases",
            "status": "planned",
            "summary": "Load the fixed 20-paper / 50-question corpus-scale seed and preserve case ids.",
        },
        {
            "phaseId": "invoke_labs_answer_path",
            "status": "planned",
            "summary": "Invoke only an explicit labs/internal answer path with paper ids and adapter opt-in.",
        },
        {
            "phaseId": "capture_sanitized_outputs",
            "status": "planned",
            "summary": "Persist only schema-backed, sanitized per-case scores and redacted diagnostics.",
        },
        {
            "phaseId": "score_deterministic_axes",
            "status": "planned",
            "summary": "Score answerability, no-answer safety, provenance, source coverage, and support without a judge model.",
        },
        {
            "phaseId": "hold_public_default",
            "status": "planned",
            "summary": "Keep public khub ask and default MCP promotion blocked until the measured gate is green.",
        },
    ]


def _score_axis_rows() -> list[dict[str, Any]]:
    return [
        {
            "axisId": "answerability",
            "status": "planned",
            "required": "Every answerable decision must be backed by eligible evidence; unsupported cases must abstain.",
            "minimumRows": 50,
        },
        {
            "axisId": "no_answer_safety",
            "status": "planned",
            "required": "Expected no-answer cases must remain no-answer with no fallback-only elevation.",
            "minimumRows": 50,
        },
        {
            "axisId": "citation_provenance",
            "status": "planned",
            "required": "Answer citations must carry source hash and chars/page/bbox provenance before scoring as supported.",
            "minimumRows": 50,
        },
        {
            "axisId": "source_coverage",
            "status": "planned",
            "required": "Resolved paper ids and cited source ids must cover each case's expected source set.",
            "minimumRows": 50,
        },
        {
            "axisId": "answer_support",
            "status": "planned",
            "required": "Claim support is measured against selected evidence terms and citation spans, not final-answer fluency.",
            "minimumRows": 50,
        },
    ]


def _threshold_rows() -> list[dict[str, Any]]:
    return [
        {"thresholdId": "minimum_seed_papers", "value": "20"},
        {"thresholdId": "minimum_seed_questions", "value": "50"},
        {"thresholdId": "expected_no_answer_pass_rate", "value": "1.0"},
        {"thresholdId": "non_evidence_leak_count", "value": "0"},
        {"thresholdId": "private_path_leak_rows", "value": "0"},
        {"thresholdId": "schema_violation_count", "value": "0"},
        {"thresholdId": "public_default_promotion", "value": "held"},
    ]


def build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design(
    *,
    corpus_scale_gate_report_path: str | Path = DEFAULT_CORPUS_SCALE_GATE_REPORT,
    corpus_scale_gate_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    corpus_gate = dict(corpus_scale_gate_report or _read_json(corpus_scale_gate_report_path))
    corpus_counts = dict(corpus_gate.get("counts") or {})
    corpus_decision = dict(corpus_gate.get("qualityGateDecision") or {})

    corpus_blockers = _corpus_gate_blockers(corpus_gate)
    unsafe_blockers = _unsafe_counter_blockers(corpus_gate)
    semantic_violations = sorted(set(corpus_blockers + unsafe_blockers))
    private_path_leak_rows = 1 if _contains_private_path(corpus_gate) else 0
    if private_path_leak_rows:
        semantic_violations.append("corpus_scale_live_runner_design_private_path_marker")
    semantic_violations = sorted(set(semantic_violations))

    status = "ready" if not semantic_violations else "blocked"
    phases = _phase_rows()
    axes = _score_axis_rows()
    thresholds = _threshold_rows()
    counts = {
        "liveRunnerDesignRows": 1,
        "corpusScaleGateReadyRows": 1 if not corpus_blockers else 0,
        "heldCorpusScaleGateRows": 1 if corpus_decision.get("corpusScaleGate") == "held" and not corpus_blockers else 0,
        "seedPaperRows": _int(corpus_counts.get("seedPaperRows")),
        "seedQuestionRows": _int(corpus_counts.get("seedQuestionRows")),
        "plannedRunnerPaperRows": 20,
        "plannedRunnerQuestionRows": 50,
        "plannedExecutionPhaseRows": len(phases),
        "plannedScoreAxisRows": len(axes),
        "plannedThresholdRows": len(thresholds),
        "liveAnswerExecutionRows": 0,
        "answerPathInvokedRows": 0,
        "answerGeneratedRows": 0,
        "answerQualityMeasuredRows": 0,
        "answerQualityScoreComputedRows": 0,
        "runnerDryRunRows": 0,
        "publicDefaultPromotionReadyRows": 0,
        "publicDefaultPromotionHeldRows": 1 if status == "ready" else 0,
        "generalRcReadyRows": 0,
        "blockedRows": len(semantic_violations),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        **{field: 0 for field in EXTRA_ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(semantic_violations),
    }
    return {
        "schema": KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DESIGN_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "corpusScaleGateReportRef": DEFAULT_CORPUS_SCALE_GATE_REPORT.as_posix(),
            "corpusScaleGateStatus": _clean_text(corpus_gate.get("status")),
            "corpusScaleGateDecision": _clean_text(corpus_gate.get("decision")),
            "corpusScaleGateNextRecommendedTranche": _clean_text(corpus_gate.get("nextRecommendedTranche")),
        },
        "runnerDesign": {
            "runnerId": "knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner",
            "mode": "design_only",
            "caseSet": "complex_qa_seed_pack_20_paper_50_question",
            "executionSurface": "labs_internal_only",
            "capturePolicy": "schema_backed_sanitized_scores_only",
            "publicDefaultPromotionPolicy": "hold_until_measured_gate_green",
        },
        "gate": {
            "designReady": status == "ready",
            "corpusScaleGateHeld": corpus_decision.get("corpusScaleGate") == "held" and not corpus_blockers,
            "liveExecutionAllowedInThisTranche": False,
            "modelOrJudgeCallsAllowedInThisTranche": False,
            "publicDefaultPromotionAllowed": False,
            "generalRcReady": False,
            "semanticViolations": semantic_violations,
        },
        "counts": counts,
        "executionPhaseRows": phases,
        "scoreAxisRows": axes,
        "thresholdRows": thresholds,
        "checkRows": [
            {
                "checkId": "corpus_scale_answer_quality_gate",
                "status": "pass" if not corpus_blockers else "fail",
                "blockers": corpus_blockers,
            },
            {
                "checkId": "unsafe_counters",
                "status": "pass" if not unsafe_blockers else "fail",
                "blockers": unsafe_blockers,
            },
            {
                "checkId": "public_default_hold",
                "status": "pass" if counts["publicDefaultPromotionReadyRows"] == 0 else "fail",
                "blockers": [] if counts["publicDefaultPromotionReadyRows"] == 0 else ["public_default_ready_unexpected"],
            },
        ],
        "warnings": [
            "design_only_no_live_answer_execution_or_scoring",
            "next_tranche_must_be_dry_run_before_any_measured_execution",
            "public_default_promotion_remains_held",
        ],
    }


def render_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    design = dict(report.get("runnerDesign") or {})
    lines = [
        "# KnowledgeOS v0.1 RC Corpus-Scale Answer Quality Live Runner Design",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- runnerId: `{design.get('runnerId')}`",
        f"- mode: `{design.get('mode')}`",
        f"- executionSurface: `{design.get('executionSurface')}`",
        f"- seedPaperRows: `{counts.get('seedPaperRows')}`",
        f"- seedQuestionRows: `{counts.get('seedQuestionRows')}`",
        f"- plannedRunnerQuestionRows: `{counts.get('plannedRunnerQuestionRows')}`",
        f"- liveAnswerExecutionRows: `{counts.get('liveAnswerExecutionRows')}`",
        f"- answerQualityMeasuredRows: `{counts.get('answerQualityMeasuredRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Execution Phases",
        "",
    ]
    for row in list(report.get("executionPhaseRows") or []):
        lines.append(f"- `{row.get('phaseId')}`: `{row.get('status')}`; {row.get('summary')}")
    lines.extend(["", "## Score Axes", ""])
    for row in list(report.get("scoreAxisRows") or []):
        lines.append(
            f"- `{row.get('axisId')}`: `{row.get('status')}`; minimumRows=`{row.get('minimumRows')}`; {row.get('required')}"
        )
    lines.extend(["", "## Checks", ""])
    for row in list(report.get("checkRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(f"- `{row.get('checkId')}`: `{row.get('status')}`; blockers=`{blockers}`")
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in _zero_counter_fields():
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DESIGN_SCHEMA_ID",
    "READY_DECISION",
    "BLOCKED_DECISION",
    "build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design",
    "write_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design",
]
