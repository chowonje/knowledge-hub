"""Dry-run plan for the KnowledgeOS v0.1 RC corpus-scale live answer runner."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.complex_qa_seed_pack import (
    COMPLEX_QA_SEED_PACK_SCHEMA_ID,
    DEFAULT_CORPUS_MANIFEST,
    build_complex_qa_seed_pack,
)
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design import (
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DESIGN_SCHEMA_ID,
    READY_DECISION as DESIGN_READY_DECISION,
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


KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.product.knowledgeos-v01-rc-corpus-scale-answer-quality-live-runner-dry-run.v1"
)

READY_DECISION = "knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run_ready"
BLOCKED_DECISION = "knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run_blocked"
NEXT_TRANCHE_READY = "corpus_scale_answer_quality_live_runner_controlled_execution"
NEXT_TRANCHE_BLOCKED = "corpus_scale_answer_quality_live_runner_dry_run_input_repair"
DEFAULT_LIVE_RUNNER_DESIGN_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design.v1.json"
)

EXTRA_ZERO_COUNTER_FIELDS = (
    "liveAnswerExecutionRows",
    "answerPathInvokedRows",
    "answerGeneratedRows",
    "answerQualityMeasuredRows",
    "answerQualityScoreComputedRows",
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

SCORE_AXIS_IDS = (
    "answerability",
    "no_answer_safety",
    "citation_provenance",
    "source_coverage",
    "answer_support",
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


def _design_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    design = dict(report.get("runnerDesign") or {})
    blockers = _schema_blockers(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DESIGN_SCHEMA_ID,
        "live_runner_design",
    )
    if report.get("status") != "ready":
        blockers.append("live_runner_design_not_ready")
    if report.get("decision") != DESIGN_READY_DECISION:
        blockers.append("live_runner_design_decision_not_ready")
    if report.get("nextRecommendedTranche") != "corpus_scale_answer_quality_live_runner_dry_run":
        blockers.append("live_runner_design_next_tranche_not_dry_run")
    if design.get("mode") != "design_only":
        blockers.append("live_runner_design_mode_not_design_only")
    if design.get("executionSurface") != "labs_internal_only":
        blockers.append("live_runner_design_execution_surface_not_labs_internal_only")
    if gate.get("liveExecutionAllowedInThisTranche") is not False:
        blockers.append("live_runner_design_allows_live_execution")
    if gate.get("modelOrJudgeCallsAllowedInThisTranche") is not False:
        blockers.append("live_runner_design_allows_model_or_judge_calls")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("live_runner_design_allows_public_default_promotion")
    if _int(counts.get("seedPaperRows")) < 20:
        blockers.append("live_runner_design_seed_paper_rows_below_20")
    if _int(counts.get("seedQuestionRows")) < 50:
        blockers.append("live_runner_design_seed_question_rows_below_50")
    if _int(counts.get("runnerDryRunRows")) != 0:
        blockers.append("live_runner_design_already_has_runner_dry_run_rows")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("live_runner_design_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("live_runner_design_schema_violations")
    return sorted(set(blockers))


def _seed_blockers(report: dict[str, Any]) -> list[str]:
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


def _unsafe_counter_blockers(report: dict[str, Any], prefix: str) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    for field in _zero_counter_fields():
        if _int(counts.get(field)) != 0:
            blockers.append(f"unsafe_counter_nonzero:{prefix}:{field}")
    return sorted(set(blockers))


def _question_hash(question: Any) -> str:
    text = _clean_text(question)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _expected_runtime_disposition(expectation: str) -> str:
    if expectation == "answerable":
        return "answer_only_if_evidence_contract_passes"
    if expectation == "expected_no_answer":
        return "must_abstain"
    return "abstain_until_structured_evidence_available"


def _case_rows(seed_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, question in enumerate(list(seed_report.get("questions") or []), start=1):
        if not isinstance(question, dict):
            continue
        expectation = _clean_text(question.get("answerabilityExpectation"))
        paper_ids = [_clean_text(item) for item in list(question.get("paperIds") or []) if _clean_text(item)]
        rows.append(
            {
                "caseIndex": index,
                "caseId": _clean_text(question.get("questionId")) or f"corpus-scale-live-runner-case-{index:03d}",
                "questionCategory": _clean_text(question.get("questionCategory")),
                "expectedEvidenceType": _clean_text(question.get("expectedEvidenceType")),
                "answerabilityExpectation": expectation,
                "paperIds": paper_ids,
                "questionSha256": _question_hash(question.get("question")),
                "plannedQueryPlan": {
                    "sourceType": "paper",
                    "requiresExplicitPaperIds": True,
                    "adapter": "parsed_artifact_evidence_chunk_adapter_runtime_v1",
                    "surface": "labs_internal_only",
                },
                "plannedScoreAxes": list(SCORE_AXIS_IDS),
                "expectedRuntimeDisposition": _expected_runtime_disposition(expectation),
                "status": "planned",
                "blockers": [],
            }
        )
    return rows


def _seed_output_contains_private_path(seed_report: dict[str, Any]) -> bool:
    return _contains_private_path(
        {
            "warnings": seed_report.get("warnings"),
            "papers": seed_report.get("papers"),
            "questions": seed_report.get("questions"),
        }
    )


def build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run(
    *,
    live_runner_design_report_path: str | Path = DEFAULT_LIVE_RUNNER_DESIGN_REPORT,
    corpus_manifest: str | Path = DEFAULT_CORPUS_MANIFEST,
    live_runner_design_report: dict[str, Any] | None = None,
    seed_pack_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    design_report = dict(live_runner_design_report or _read_json(live_runner_design_report_path))
    seed_report = dict(seed_pack_report or build_complex_qa_seed_pack(corpus_manifest=corpus_manifest, target_paper_count=20))

    design_blockers = _design_blockers(design_report)
    seed_blockers = _seed_blockers(seed_report)
    unsafe_blockers = sorted(
        set(_unsafe_counter_blockers(design_report, "design") + _unsafe_counter_blockers(seed_report, "seed"))
    )
    semantic_violations = sorted(set(design_blockers + seed_blockers + unsafe_blockers))
    private_path_leak_rows = 1 if _contains_private_path(design_report) or _seed_output_contains_private_path(seed_report) else 0
    if private_path_leak_rows:
        semantic_violations.append("corpus_scale_live_runner_dry_run_private_path_marker")
    semantic_violations = sorted(set(semantic_violations))

    cases = _case_rows(seed_report)
    category_counts = Counter(row.get("questionCategory") for row in cases)
    expectation_counts = Counter(row.get("answerabilityExpectation") for row in cases)
    status = "ready" if not semantic_violations else "blocked"
    planned_score_rows = len(cases) * len(SCORE_AXIS_IDS)
    counts = {
        "liveRunnerDryRunRows": 1,
        "sourceDesignReadyRows": 1 if not design_blockers else 0,
        "seedPaperRows": _int((seed_report.get("counts") or {}).get("paperRows")),
        "seedQuestionRows": _int((seed_report.get("counts") or {}).get("questionRows")),
        "dryRunCaseRows": len(cases),
        "plannedRunnerPaperRows": 20,
        "plannedRunnerQuestionRows": 50,
        "plannedAnswerPathInvocationRows": len(cases),
        "plannedSanitizedOutputRows": len(cases),
        "plannedScoreAxisRows": len(SCORE_AXIS_IDS),
        "plannedScoreRows": planned_score_rows,
        "plannedAnswerableRows": expectation_counts.get("answerable", 0),
        "plannedExpectedNoAnswerRows": expectation_counts.get("expected_no_answer", 0),
        "plannedBlockedUntilStructuredEvidenceRows": expectation_counts.get("blocked_until_structured_evidence", 0),
        "plannedNoAnswerOrBlockedRows": expectation_counts.get("expected_no_answer", 0)
        + expectation_counts.get("blocked_until_structured_evidence", 0),
        "runnerDryRunRows": 1 if status == "ready" else 0,
        "liveAnswerExecutionRows": 0,
        "answerPathInvokedRows": 0,
        "answerGeneratedRows": 0,
        "answerQualityMeasuredRows": 0,
        "answerQualityScoreComputedRows": 0,
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
        "schema": KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "liveRunnerDesignReportRef": DEFAULT_LIVE_RUNNER_DESIGN_REPORT.as_posix(),
            "corpusManifestRef": "eval/knowledgeos/fixtures/corpus_manifest.json",
            "caseSet": "complex_qa_seed_pack_20_paper_50_question",
        },
        "dryRunPlan": {
            "runnerId": "knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner",
            "mode": "dry_run",
            "executionSurface": "labs_internal_only",
            "caseSelection": "fixed_20_paper_50_question_seed",
            "outputPolicy": "schema_backed_sanitized_case_scores_only",
            "answerExecutionInThisTranche": False,
            "modelOrJudgeCallsInThisTranche": False,
        },
        "gate": {
            "dryRunReady": status == "ready",
            "sourceDesignReady": not design_blockers,
            "seedPackReady": not seed_blockers,
            "caseRowsReady": len(cases) == 50,
            "liveExecutionAllowedInThisTranche": False,
            "modelOrJudgeCallsAllowedInThisTranche": False,
            "publicDefaultPromotionAllowed": False,
            "generalRcReady": False,
            "semanticViolations": semantic_violations,
        },
        "counts": counts,
        "caseSummary": {
            "byQuestionCategory": dict(sorted((str(key), value) for key, value in category_counts.items())),
            "byAnswerabilityExpectation": dict(sorted((str(key), value) for key, value in expectation_counts.items())),
        },
        "caseRows": cases,
        "checkRows": [
            {"checkId": "live_runner_design", "status": "pass" if not design_blockers else "fail", "blockers": design_blockers},
            {"checkId": "complex_qa_seed_pack", "status": "pass" if not seed_blockers else "fail", "blockers": seed_blockers},
            {"checkId": "unsafe_counters", "status": "pass" if not unsafe_blockers else "fail", "blockers": unsafe_blockers},
            {
                "checkId": "public_default_hold",
                "status": "pass" if counts["publicDefaultPromotionReadyRows"] == 0 else "fail",
                "blockers": [] if counts["publicDefaultPromotionReadyRows"] == 0 else ["public_default_ready_unexpected"],
            },
        ],
        "warnings": [
            "dry_run_plans_runner_cases_but_does_not_invoke_answer_path",
            "generated_report_uses_question_hashes_not_raw_question_text",
            "public_default_promotion_remains_held",
        ],
    }


def render_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# KnowledgeOS v0.1 RC Corpus-Scale Answer Quality Live Runner Dry Run",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- dryRunCaseRows: `{counts.get('dryRunCaseRows')}`",
        f"- plannedRunnerQuestionRows: `{counts.get('plannedRunnerQuestionRows')}`",
        f"- plannedAnswerPathInvocationRows: `{counts.get('plannedAnswerPathInvocationRows')}`",
        f"- plannedScoreRows: `{counts.get('plannedScoreRows')}`",
        f"- runnerDryRunRows: `{counts.get('runnerDryRunRows')}`",
        f"- liveAnswerExecutionRows: `{counts.get('liveAnswerExecutionRows')}`",
        f"- answerPathInvokedRows: `{counts.get('answerPathInvokedRows')}`",
        f"- answerGeneratedRows: `{counts.get('answerGeneratedRows')}`",
        f"- answerQualityMeasuredRows: `{counts.get('answerQualityMeasuredRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Case Summary",
        "",
    ]
    summary = dict(report.get("caseSummary") or {})
    for key, values in summary.items():
        lines.append(f"- `{key}`: `{values}`")
    lines.extend(["", "## Checks", ""])
    for row in list(report.get("checkRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(f"- `{row.get('checkId')}`: `{row.get('status')}`; blockers=`{blockers}`")
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in _zero_counter_fields():
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DRY_RUN_SCHEMA_ID",
    "READY_DECISION",
    "BLOCKED_DECISION",
    "build_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run",
    "write_knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run",
]
