"""Corpus-scale answer quality gate for the KnowledgeOS v0.1 RC."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.complex_qa_abstain_baseline_runner import (
    COMPLEX_QA_ABSTAIN_BASELINE_SCHEMA_ID,
)
from knowledge_hub.papers.complex_qa_seed_pack import COMPLEX_QA_SEED_PACK_SCHEMA_ID
from knowledge_hub.papers.complex_qa_strict_evidence_answer_quality_dry_run import (
    COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_DRY_RUN_SCHEMA_ID,
)
from knowledge_hub.papers.complex_qa_structured_evidence_comparison_runner import (
    COMPLEX_QA_STRUCTURED_EVIDENCE_COMPARISON_SCHEMA_ID,
)
from knowledge_hub.papers.knowledgeos_v01_rc_post_merge_convergence_cleanup_decision import (
    KNOWLEDGEOS_V01_RC_POST_MERGE_CONVERGENCE_CLEANUP_DECISION_SCHEMA_ID,
    READY_DECISION as POST_MERGE_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_DEFAULT_OFF_NO_ANSWER_REGRESSION_SMOKE_SCHEMA_ID,
    READY_DECISION as DEFAULT_OFF_NO_ANSWER_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_RUNNER_SCHEMA_ID,
    READY_DECISION as LABS_QUALITY_READY_DECISION,
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


KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_SCHEMA_ID = (
    "knowledge-hub.product.knowledgeos-v01-rc-corpus-scale-answer-quality-gate.v1"
)

READY_DECISION = "knowledgeos_v01_rc_corpus_scale_answer_quality_gate_ready"
HELD_DECISION = "knowledgeos_v01_rc_corpus_scale_answer_quality_gate_held"
BLOCKED_DECISION = "knowledgeos_v01_rc_corpus_scale_answer_quality_gate_blocked"
NEXT_TRANCHE_GREEN = "public_default_promotion_gate_after_corpus_scale_quality"
NEXT_TRANCHE_HELD = "corpus_scale_answer_quality_live_runner_design"
NEXT_TRANCHE_BLOCKED = "corpus_scale_answer_quality_gate_input_repair"

DEFAULT_POST_MERGE_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_post_merge_convergence_cleanup_decision.v1.json"
)
DEFAULT_NO_ANSWER_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke.v1.json"
)
DEFAULT_LABS_QUALITY_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner.v1.json"
)

EXTRA_ZERO_COUNTER_FIELDS = (
    "answerGeneratedRows",
    "answerQualityScoreComputedRows",
    "liveAnswerExecutionRows",
    "answerPathInvokedRows",
    "llmCallRows",
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _zero_counter_fields() -> tuple[str, ...]:
    return tuple(dict.fromkeys((*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS)))


def _schema_blockers(report: dict[str, Any], schema_id: str, prefix: str) -> list[str]:
    blockers: list[str] = []
    if report.get("schema") != schema_id:
        blockers.append(f"{prefix}_schema_mismatch")
        return blockers
    validation = validate_payload(report, schema_id, strict=True)
    if not validation.ok:
        blockers.append(f"{prefix}_schema_validation_failed")
    return blockers


def _post_merge_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers = _schema_blockers(
        report,
        KNOWLEDGEOS_V01_RC_POST_MERGE_CONVERGENCE_CLEANUP_DECISION_SCHEMA_ID,
        "post_merge_convergence_cleanup",
    )
    if report.get("status") != "ready":
        blockers.append("post_merge_convergence_cleanup_not_ready")
    if report.get("decision") != POST_MERGE_READY_DECISION:
        blockers.append("post_merge_convergence_cleanup_decision_not_ready")
    if _int(counts.get("prMergedRows")) != 1:
        blockers.append("post_merge_pr_merged_missing")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("post_merge_public_default_hold_missing")
    if _int(counts.get("branchCleanupAppliedRows")) != 0:
        blockers.append("post_merge_branch_cleanup_already_applied")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("post_merge_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("post_merge_schema_violations")
    return sorted(set(blockers))


def _seed_pack_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers = _schema_blockers(report, COMPLEX_QA_SEED_PACK_SCHEMA_ID, "complex_qa_seed_pack")
    if report.get("status") != "ok":
        blockers.append("complex_qa_seed_pack_not_ok")
    if _int(counts.get("paperRows")) < 20:
        blockers.append("complex_qa_seed_pack_less_than_20_papers")
    if _int(counts.get("questionRows")) < 50:
        blockers.append("complex_qa_seed_pack_less_than_50_questions")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("complex_qa_seed_pack_schema_violations")
    return sorted(set(blockers))


def _abstain_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers = _schema_blockers(report, COMPLEX_QA_ABSTAIN_BASELINE_SCHEMA_ID, "complex_qa_abstain_baseline")
    if report.get("status") != "ok":
        blockers.append("complex_qa_abstain_baseline_not_ok")
    if _int(counts.get("questionRows")) < 50:
        blockers.append("complex_qa_abstain_baseline_less_than_50_questions")
    if float(counts.get("abstainNoAnswerPassRate") or 0.0) < 1.0:
        blockers.append("complex_qa_abstain_no_answer_pass_rate_below_1")
    if _int(counts.get("unsafeAnsweredRows")) != 0:
        blockers.append("complex_qa_abstain_unsafe_answered")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("complex_qa_abstain_schema_violations")
    return sorted(set(blockers))


def _comparison_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers = _schema_blockers(
        report,
        COMPLEX_QA_STRUCTURED_EVIDENCE_COMPARISON_SCHEMA_ID,
        "complex_qa_structured_evidence_comparison",
    )
    if report.get("status") != "ok":
        blockers.append("complex_qa_structured_evidence_comparison_not_ok")
    if _int(counts.get("questionRows")) < 50:
        blockers.append("complex_qa_structured_evidence_comparison_less_than_50_questions")
    if _int(counts.get("unsafeAnsweredRows")) != 0:
        blockers.append("complex_qa_structured_evidence_comparison_unsafe_answered")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("complex_qa_structured_evidence_comparison_schema_violations")
    return sorted(set(blockers))


def _dry_run_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers = _schema_blockers(
        report,
        COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_DRY_RUN_SCHEMA_ID,
        "complex_qa_strict_evidence_answer_quality_dry_run",
    )
    if report.get("status") != "ok":
        blockers.append("complex_qa_strict_evidence_answer_quality_dry_run_not_ok")
    if _int(counts.get("questionRows")) < 50:
        blockers.append("complex_qa_strict_evidence_answer_quality_dry_run_less_than_50_questions")
    if _int(counts.get("unsafeAnsweredRows")) != 0:
        blockers.append("complex_qa_strict_evidence_answer_quality_dry_run_unsafe_answered")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("complex_qa_strict_evidence_answer_quality_dry_run_schema_violations")
    return sorted(set(blockers))


def _no_answer_smoke_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers = _schema_blockers(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_DEFAULT_OFF_NO_ANSWER_REGRESSION_SMOKE_SCHEMA_ID,
        "default_off_no_answer_smoke",
    )
    if report.get("status") != "ready":
        blockers.append("default_off_no_answer_smoke_not_ready")
    if report.get("decision") != DEFAULT_OFF_NO_ANSWER_READY_DECISION:
        blockers.append("default_off_no_answer_smoke_decision_not_ready")
    if _int(counts.get("passRows")) < 3:
        blockers.append("default_off_no_answer_smoke_pass_rows_below_3")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("default_off_no_answer_smoke_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("default_off_no_answer_smoke_schema_violations")
    return sorted(set(blockers))


def _labs_quality_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers = _schema_blockers(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_RUNNER_SCHEMA_ID,
        "labs_opt_in_quality_eval_runner",
    )
    if report.get("status") != "ready":
        blockers.append("labs_opt_in_quality_eval_runner_not_ready")
    if report.get("decision") != LABS_QUALITY_READY_DECISION:
        blockers.append("labs_opt_in_quality_eval_runner_decision_not_ready")
    if _int(counts.get("qualityPassRows")) != _int(counts.get("inputCaseRows")):
        blockers.append("labs_opt_in_quality_eval_runner_not_all_passed")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("labs_opt_in_quality_eval_runner_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("labs_opt_in_quality_eval_runner_schema_violations")
    return sorted(set(blockers))


def _unsafe_counter_blockers(*reports: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for index, report in enumerate(reports, start=1):
        counts = dict(report.get("counts") or {})
        for field in _zero_counter_fields():
            if _int(counts.get(field)) != 0:
                blockers.append(f"unsafe_counter_nonzero:report{index}:{field}")
    return sorted(set(blockers))


def _quality_axis_rows(*, no_answer_ready: bool, seed_ready: bool, dry_counts: dict[str, Any]) -> list[dict[str, Any]]:
    planned_rows = _int(dry_counts.get("plannedAnswerQualityDryRunRows"))
    measured_rows = _int(dry_counts.get("answerQualityScoreComputedRows"))
    strict_rows = _int(dry_counts.get("strictEvidenceAvailableRows"))
    return [
        {
            "axisId": "corpus_breadth",
            "status": "pass" if seed_ready else "fail",
            "required": "20 papers and 50 seed questions",
            "observedRows": _int(dry_counts.get("questionRows")),
            "blockers": [] if seed_ready else ["seed_pack_breadth_not_ready"],
        },
        {
            "axisId": "answerability_no_answer_safety",
            "status": "pass" if no_answer_ready else "fail",
            "required": "expected no-answer remains no-answer",
            "observedRows": _int(dry_counts.get("stableExpectedNoAnswerRows")),
            "blockers": [] if no_answer_ready else ["no_answer_safety_not_ready"],
        },
        {
            "axisId": "citation_provenance",
            "status": "blocked",
            "required": "citation/provenance scored on corpus-scale live answer outputs",
            "observedRows": planned_rows,
            "blockers": ["corpus_scale_live_answer_quality_execution_missing"],
        },
        {
            "axisId": "source_coverage",
            "status": "blocked",
            "required": "source coverage scored on corpus-scale live answer outputs",
            "observedRows": planned_rows,
            "blockers": ["corpus_scale_live_answer_quality_execution_missing"],
        },
        {
            "axisId": "answer_support",
            "status": "blocked",
            "required": "answer support measured against strict evidence",
            "observedRows": measured_rows,
            "blockers": [
                "answer_quality_scores_not_computed",
                "strict_evidence_ready_rows_below_threshold" if strict_rows < 1 else "live_answer_outputs_missing",
            ],
        },
    ]


def build_knowledgeos_v01_rc_corpus_scale_answer_quality_gate(
    *,
    post_merge_report_path: str | Path = DEFAULT_POST_MERGE_REPORT,
    no_answer_report_path: str | Path = DEFAULT_NO_ANSWER_REPORT,
    labs_quality_report_path: str | Path = DEFAULT_LABS_QUALITY_REPORT,
    post_merge_report: dict[str, Any] | None = None,
    seed_pack_report: dict[str, Any] | None = None,
    abstain_baseline_report: dict[str, Any] | None = None,
    structured_evidence_comparison_report: dict[str, Any] | None = None,
    answer_quality_dry_run_report: dict[str, Any] | None = None,
    no_answer_report: dict[str, Any] | None = None,
    labs_quality_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    post_report = dict(post_merge_report or _read_json(post_merge_report_path))
    seed_report = dict(seed_pack_report or {})
    abstain_report = dict(abstain_baseline_report or {})
    comparison_report = dict(structured_evidence_comparison_report or {})
    dry_run_report = dict(answer_quality_dry_run_report or {})
    no_answer = dict(no_answer_report or _read_json(no_answer_report_path))
    labs_quality = dict(labs_quality_report or _read_json(labs_quality_report_path))

    post_blockers = _post_merge_blockers(post_report)
    seed_blockers = _seed_pack_blockers(seed_report)
    abstain_blockers = _abstain_blockers(abstain_report)
    comparison_blockers = _comparison_blockers(comparison_report)
    dry_run_blockers = _dry_run_blockers(dry_run_report)
    no_answer_blockers = _no_answer_smoke_blockers(no_answer)
    labs_quality_blockers = _labs_quality_blockers(labs_quality)
    unsafe_blockers = _unsafe_counter_blockers(post_report, seed_report, abstain_report, comparison_report, dry_run_report)
    semantic_violations = sorted(
        set(
            post_blockers
            + seed_blockers
            + abstain_blockers
            + comparison_blockers
            + dry_run_blockers
            + no_answer_blockers
            + labs_quality_blockers
            + unsafe_blockers
        )
    )
    private_path_leak_rows = (
        1
        if any(
            _contains_private_path(report)
            for report in (post_report, seed_report, abstain_report, comparison_report, dry_run_report, no_answer, labs_quality)
        )
        else 0
    )
    if private_path_leak_rows:
        semantic_violations.append("corpus_scale_answer_quality_gate_private_path_marker")
    semantic_violations = sorted(set(semantic_violations))

    seed_counts = dict(seed_report.get("counts") or {})
    abstain_counts = dict(abstain_report.get("counts") or {})
    comparison_counts = dict(comparison_report.get("counts") or {})
    dry_counts = dict(dry_run_report.get("counts") or {})
    no_answer_counts = dict(no_answer.get("counts") or {})
    labs_counts = dict(labs_quality.get("counts") or {})

    seed_ready = not seed_blockers
    no_answer_ready = not no_answer_blockers and not abstain_blockers
    planned_quality_rows = _int(dry_counts.get("plannedAnswerQualityDryRunRows"))
    measured_quality_rows = _int(dry_counts.get("answerQualityScoreComputedRows"))
    strict_ready_rows = _int(dry_counts.get("strictEvidenceAvailableRows"))
    gate_green = (
        not semantic_violations
        and _int(seed_counts.get("paperRows")) >= 20
        and _int(seed_counts.get("questionRows")) >= 50
        and no_answer_ready
        and planned_quality_rows >= 20
        and measured_quality_rows >= 50
    )
    report_ready = not semantic_violations
    held = report_ready and not gate_green
    status = "ready" if report_ready else "blocked"
    decision = READY_DECISION if gate_green else (HELD_DECISION if held else BLOCKED_DECISION)

    quality_axes = _quality_axis_rows(no_answer_ready=no_answer_ready, seed_ready=seed_ready, dry_counts=dry_counts)
    gate_blockers = [] if gate_green else [
        "corpus_scale_live_answer_quality_execution_missing",
        "answer_quality_scores_not_computed",
        "strict_evidence_ready_rows_below_threshold" if strict_ready_rows < 20 else "corpus_scale_measured_rows_below_threshold",
    ]
    counts = {
        "corpusScaleAnswerQualityGateRows": 1,
        "seedPaperRows": _int(seed_counts.get("paperRows")),
        "seedQuestionRows": _int(seed_counts.get("questionRows")),
        "abstainBaselineRows": _int(abstain_counts.get("questionRows")),
        "abstainNoAnswerPassRows": _int(abstain_counts.get("abstainBaselinePassRows")),
        "abstainNoAnswerExpectedRows": _int(abstain_counts.get("abstainExpectedRows")),
        "defaultOffNoAnswerPassRows": _int(no_answer_counts.get("passRows")),
        "labsQualityInputRows": _int(labs_counts.get("inputCaseRows")),
        "labsQualityPassRows": _int(labs_counts.get("qualityPassRows")),
        "structuredEvidenceComparisonRows": _int(comparison_counts.get("questionRows")),
        "strictEvidenceAvailableRows": strict_ready_rows,
        "plannedAnswerQualityDryRunRows": planned_quality_rows,
        "liveAnswerExecutionRows": 0,
        "answerQualityMeasuredRows": measured_quality_rows,
        "qualityAxisRows": len(quality_axes),
        "qualityAxisPassRows": sum(1 for row in quality_axes if row.get("status") == "pass"),
        "qualityAxisBlockedRows": sum(1 for row in quality_axes if row.get("status") == "blocked"),
        "corpusScaleAnswerQualityGateGreenRows": 1 if gate_green else 0,
        "corpusScaleAnswerQualityGateHeldRows": 1 if held else 0,
        "publicDefaultPromotionReadyRows": 0,
        "publicDefaultPromotionHeldRows": 1,
        "generalRcReadyRows": 0,
        "blockedRows": len(semantic_violations),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        **{field: 0 for field in EXTRA_ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(semantic_violations),
    }
    return {
        "schema": KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": decision,
        "nextRecommendedTranche": (
            NEXT_TRANCHE_GREEN if gate_green else (NEXT_TRANCHE_HELD if held else NEXT_TRANCHE_BLOCKED)
        ),
        "inputs": {
            "postMergeReportRef": DEFAULT_POST_MERGE_REPORT.as_posix(),
            "seedSource": "complex_qa_seed_pack_20_paper_50_question",
            "noAnswerReportRef": DEFAULT_NO_ANSWER_REPORT.as_posix(),
            "labsQualityReportRef": DEFAULT_LABS_QUALITY_REPORT.as_posix(),
        },
        "qualityGateDecision": {
            "corpusScaleGate": "green" if gate_green else ("held" if held else "blocked"),
            "publicDefaultDecision": "hold_public_default_promotion",
            "generalRcDecision": "blocked_pending_corpus_scale_answer_quality",
            "nextGate": NEXT_TRANCHE_GREEN if gate_green else NEXT_TRANCHE_HELD,
            "gateBlockers": gate_blockers if held else semantic_violations,
        },
        "counts": counts,
        "gate": {
            "reportReady": report_ready,
            "corpusScaleAnswerQualityGateGreen": gate_green,
            "seedBreadthReady": seed_ready,
            "noAnswerSafetyReady": no_answer_ready,
            "strictEvidenceReadyRowsMeetThreshold": strict_ready_rows >= 20,
            "liveAnswerQualityMeasured": measured_quality_rows >= 50,
            "publicDefaultPromotionAllowed": False,
            "generalRcReady": False,
            "semanticViolations": semantic_violations,
        },
        "qualityAxisRows": quality_axes,
        "checkRows": [
            {"checkId": "post_merge_convergence_cleanup", "status": "pass" if not post_blockers else "fail", "blockers": post_blockers},
            {"checkId": "complex_qa_seed_pack", "status": "pass" if not seed_blockers else "fail", "blockers": seed_blockers},
            {"checkId": "complex_qa_abstain_baseline", "status": "pass" if not abstain_blockers else "fail", "blockers": abstain_blockers},
            {"checkId": "structured_evidence_comparison", "status": "pass" if not comparison_blockers else "fail", "blockers": comparison_blockers},
            {"checkId": "answer_quality_dry_run", "status": "pass" if not dry_run_blockers else "fail", "blockers": dry_run_blockers},
            {"checkId": "default_off_no_answer_smoke", "status": "pass" if not no_answer_blockers else "fail", "blockers": no_answer_blockers},
            {"checkId": "labs_opt_in_quality_eval_runner", "status": "pass" if not labs_quality_blockers else "fail", "blockers": labs_quality_blockers},
            {"checkId": "unsafe_counters", "status": "pass" if not unsafe_blockers else "fail", "blockers": unsafe_blockers},
        ],
        "warnings": [
            "this_gate_defines_corpus_scale_thresholds_but_does_not_call_models_or_judges",
            "public_default_promotion_remains_held_until_corpus_scale_gate_is_green",
            "current_report_expected_to_hold_because_live_answer_quality_execution_is_not_yet_open",
        ],
    }


def render_knowledgeos_v01_rc_corpus_scale_answer_quality_gate_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    decision = dict(report.get("qualityGateDecision") or {})
    lines = [
        "# KnowledgeOS v0.1 RC Corpus-Scale Answer Quality Gate",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- corpusScaleGate: `{decision.get('corpusScaleGate')}`",
        f"- publicDefaultDecision: `{decision.get('publicDefaultDecision')}`",
        f"- seedPaperRows: `{counts.get('seedPaperRows')}`",
        f"- seedQuestionRows: `{counts.get('seedQuestionRows')}`",
        f"- abstainNoAnswerPassRows: `{counts.get('abstainNoAnswerPassRows')}`",
        f"- defaultOffNoAnswerPassRows: `{counts.get('defaultOffNoAnswerPassRows')}`",
        f"- labsQualityPassRows: `{counts.get('labsQualityPassRows')}`",
        f"- strictEvidenceAvailableRows: `{counts.get('strictEvidenceAvailableRows')}`",
        f"- plannedAnswerQualityDryRunRows: `{counts.get('plannedAnswerQualityDryRunRows')}`",
        f"- liveAnswerExecutionRows: `{counts.get('liveAnswerExecutionRows')}`",
        f"- answerQualityMeasuredRows: `{counts.get('answerQualityMeasuredRows')}`",
        f"- corpusScaleAnswerQualityGateGreenRows: `{counts.get('corpusScaleAnswerQualityGateGreenRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Quality Axes",
        "",
    ]
    for row in list(report.get("qualityAxisRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(f"- `{row.get('axisId')}`: `{row.get('status')}`; observedRows=`{row.get('observedRows')}`; blockers=`{blockers}`")
    lines.extend(["", "## Checks", ""])
    for row in list(report.get("checkRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(f"- `{row.get('checkId')}`: `{row.get('status')}`; blockers=`{blockers}`")
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in _zero_counter_fields():
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_knowledgeos_v01_rc_corpus_scale_answer_quality_gate(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_knowledgeos_v01_rc_corpus_scale_answer_quality_gate_markdown(report), encoding="utf-8")
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_SCHEMA_ID",
    "HELD_DECISION",
    "build_knowledgeos_v01_rc_corpus_scale_answer_quality_gate",
    "write_knowledgeos_v01_rc_corpus_scale_answer_quality_gate",
]
