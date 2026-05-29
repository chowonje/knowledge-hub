"""Controlled execution for the KnowledgeOS v0.1 RC corpus-scale answer-quality runner."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.complex_qa_seed_pack import DEFAULT_CORPUS_MANIFEST, build_complex_qa_seed_pack
from knowledge_hub.papers.evidence_chunk_answer_preview import (
    PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID,
    build_paper_evidence_chunk_answer_preview,
)
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run import (
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DRY_RUN_SCHEMA_ID,
    READY_DECISION as DRY_RUN_READY_DECISION,
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
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke import (
    DEFAULT_PAPERS_DIR,
    _build_searcher,
)


KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_CONTROLLED_EXECUTION_SCHEMA_ID = (
    "knowledge-hub.product.knowledgeos-v01-rc-corpus-scale-answer-quality-controlled-execution.v1"
)

READY_DECISION = "knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution_ready"
BLOCKED_DECISION = "knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution_blocked"
NEXT_TRANCHE_READY = "corpus_scale_answer_quality_positive_section_paragraph_seed"
NEXT_TRANCHE_BLOCKED = "corpus_scale_answer_quality_answerability_gate_repair"
DEFAULT_LIVE_RUNNER_DRY_RUN_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run.v1.json"
)

EXTRA_ZERO_COUNTER_FIELDS = (
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


def _dry_run_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers = _schema_blockers(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DRY_RUN_SCHEMA_ID,
        "live_runner_dry_run",
    )
    if report.get("status") != "ready":
        blockers.append("live_runner_dry_run_not_ready")
    if report.get("decision") != DRY_RUN_READY_DECISION:
        blockers.append("live_runner_dry_run_decision_not_ready")
    if report.get("nextRecommendedTranche") != "corpus_scale_answer_quality_live_runner_controlled_execution":
        blockers.append("live_runner_dry_run_next_tranche_not_controlled_execution")
    if _int(counts.get("dryRunCaseRows")) != 50:
        blockers.append("live_runner_dry_run_case_rows_not_50")
    if _int(counts.get("runnerDryRunRows")) != 1:
        blockers.append("live_runner_dry_run_row_missing")
    if _int(counts.get("liveAnswerExecutionRows")) != 0:
        blockers.append("live_runner_dry_run_already_executed_answers")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("live_runner_dry_run_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("live_runner_dry_run_schema_violations")
    if gate.get("liveExecutionAllowedInThisTranche") is not False:
        blockers.append("live_runner_dry_run_allows_live_execution_in_source_tranche")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("live_runner_dry_run_allows_public_default_promotion")
    return sorted(set(blockers))


def _unsafe_counter_blockers(report: dict[str, Any], prefix: str) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    for field in _zero_counter_fields():
        if _int(counts.get(field)) != 0:
            blockers.append(f"unsafe_counter_nonzero:{prefix}:{field}")
    return sorted(set(blockers))


def _seed_questions_by_id(seed_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in list(seed_report.get("questions") or []):
        if not isinstance(row, dict):
            continue
        question_id = _clean_text(row.get("questionId"))
        if question_id:
            rows[question_id] = row
    return rows


def _preview_dimension_statuses(*, case: dict[str, Any], preview: dict[str, Any], schema_valid: bool, llm_calls: int) -> dict[str, bool]:
    expectation = _clean_text(case.get("answerabilityExpectation"))
    expected_answerable = expectation == "answerable"
    observed_answerable = bool(preview.get("answerable"))
    summary = dict(preview.get("evidencePacketSummary") or {})
    contract = dict(preview.get("evidencePacketContractSummary") or {})
    citations = _int(summary.get("citationCount"))
    spans = _int(contract.get("spanRows"))
    no_answer_safe = (
        not observed_answerable
        and _clean_text(preview.get("status")) == "no_evidence"
        and citations == 0
        and spans == 0
        and llm_calls == 0
    )
    answerability_ok = observed_answerable if expected_answerable else no_answer_safe
    citation_ok = citations > 0 and spans > 0 if expected_answerable else citations == 0 and spans == 0
    source_coverage_ok = bool(case.get("paperIds")) and list(preview.get("paperIds") or []) == list(case.get("paperIds") or [])
    return {
        "schemaValid": bool(schema_valid),
        "answerability": bool(answerability_ok),
        "noAnswerSafety": bool(no_answer_safe if not expected_answerable else True),
        "citationProvenance": bool(citation_ok),
        "sourceCoverage": bool(source_coverage_ok),
        "answerSupport": bool(observed_answerable and citation_ok) if expected_answerable else bool(no_answer_safe),
    }


def _execution_row(
    *,
    case: dict[str, Any],
    seed_question: dict[str, Any],
    papers_dir: str | Path,
) -> dict[str, Any]:
    question = _clean_text(seed_question.get("question"))
    searcher, llm = _build_searcher(papers_dir=papers_dir)
    preview = build_paper_evidence_chunk_answer_preview(
        searcher,
        question=question,
        paper_ids=list(case.get("paperIds") or []),
        allow_external=False,
    )
    validation = validate_payload(preview, PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID, strict=True)
    summary = dict(preview.get("evidencePacketSummary") or {})
    contract = dict(preview.get("evidencePacketContractSummary") or {})
    dimensions = _preview_dimension_statuses(
        case=case,
        preview=preview,
        schema_valid=validation.ok,
        llm_calls=int(llm.calls),
    )
    passed_dimensions = sum(1 for ok in dimensions.values() if ok)
    quality_score = round(passed_dimensions / len(dimensions), 6) if dimensions else 0.0
    expectation = _clean_text(case.get("answerabilityExpectation"))
    expected_answerable = expectation == "answerable"
    observed_answerable = bool(preview.get("answerable"))
    failure_reasons: list[str] = []
    if not validation.ok:
        failure_reasons.append("preview_payload_schema_invalid")
    if not dimensions["answerability"]:
        failure_reasons.append("answerability_expectation_failed")
    if not dimensions["noAnswerSafety"]:
        failure_reasons.append("no_answer_safety_failed")
    if not dimensions["citationProvenance"]:
        failure_reasons.append("citation_provenance_failed")
    if not dimensions["sourceCoverage"]:
        failure_reasons.append("source_coverage_failed")
    if not dimensions["answerSupport"]:
        failure_reasons.append("answer_support_failed")
    if not expected_answerable and observed_answerable:
        failure_reasons.append("unexpected_answerable_for_no_answer_or_blocked_case")
    if not expected_answerable and int(llm.calls) > 0:
        failure_reasons.append("unexpected_llm_call_for_no_answer_or_blocked_case")
    row = {
        "caseIndex": _int(case.get("caseIndex")),
        "caseId": _clean_text(case.get("caseId")),
        "questionCategory": _clean_text(case.get("questionCategory")),
        "expectedEvidenceType": _clean_text(case.get("expectedEvidenceType")),
        "answerabilityExpectation": expectation,
        "paperIds": list(case.get("paperIds") or []),
        "questionSha256": _clean_text(case.get("questionSha256")),
        "observedStatus": _clean_text(preview.get("status")),
        "observedAnswerable": observed_answerable,
        "adapterStatus": _clean_text(summary.get("adapterStatus")),
        "adapterRowsAdded": _int(summary.get("adapterRowsAdded")),
        "adapterCandidateRowsConsidered": _int(summary.get("adapterCandidateRowsConsidered")),
        "selectedEvidenceCount": _int(summary.get("selectedEvidenceCount")),
        "citationCount": _int(summary.get("citationCount")),
        "evidencePacketContractSpanRows": _int(contract.get("spanRows")),
        "localFakeLlmCallRows": int(llm.calls),
        "schemaValid": bool(validation.ok),
        "dimensionStatuses": dimensions,
        "qualityScore": quality_score,
        "qualityGrade": "pass" if not failure_reasons else ("partial" if quality_score >= 0.5 else "fail"),
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "pass": not failure_reasons,
        "failureReasons": sorted(set(failure_reasons)),
    }
    if _contains_private_path(row):
        row["failureReasons"] = sorted(set([*row["failureReasons"], "private_path_leak"]))
        row["pass"] = False
    return row


def _missing_question_row(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "caseIndex": _int(case.get("caseIndex")),
        "caseId": _clean_text(case.get("caseId")),
        "questionCategory": _clean_text(case.get("questionCategory")),
        "expectedEvidenceType": _clean_text(case.get("expectedEvidenceType")),
        "answerabilityExpectation": _clean_text(case.get("answerabilityExpectation")),
        "paperIds": list(case.get("paperIds") or []),
        "questionSha256": _clean_text(case.get("questionSha256")),
        "observedStatus": "not_executed",
        "observedAnswerable": False,
        "adapterStatus": "",
        "adapterRowsAdded": 0,
        "adapterCandidateRowsConsidered": 0,
        "selectedEvidenceCount": 0,
        "citationCount": 0,
        "evidencePacketContractSpanRows": 0,
        "localFakeLlmCallRows": 0,
        "schemaValid": False,
        "dimensionStatuses": {
            "schemaValid": False,
            "answerability": False,
            "noAnswerSafety": False,
            "citationProvenance": False,
            "sourceCoverage": False,
            "answerSupport": False,
        },
        "qualityScore": 0.0,
        "qualityGrade": "fail",
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "pass": False,
        "failureReasons": ["seed_question_missing_for_case"],
    }


def build_knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution(
    *,
    live_runner_dry_run_report_path: str | Path = DEFAULT_LIVE_RUNNER_DRY_RUN_REPORT,
    corpus_manifest: str | Path = DEFAULT_CORPUS_MANIFEST,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    live_runner_dry_run_report: dict[str, Any] | None = None,
    seed_pack_report: dict[str, Any] | None = None,
    execute_case: Callable[..., dict[str, Any]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    dry_run_report = dict(live_runner_dry_run_report or _read_json(live_runner_dry_run_report_path))
    seed_report = dict(seed_pack_report or build_complex_qa_seed_pack(corpus_manifest=corpus_manifest, target_paper_count=20))
    cases = [dict(row or {}) for row in list(dry_run_report.get("caseRows") or [])]
    seed_by_id = _seed_questions_by_id(seed_report)

    dry_run_blockers = _dry_run_blockers(dry_run_report)
    unsafe_blockers = _unsafe_counter_blockers(dry_run_report, "dry_run")
    executor = execute_case or _execution_row
    rows: list[dict[str, Any]] = []
    if not dry_run_blockers:
        for case in cases:
            seed_question = seed_by_id.get(_clean_text(case.get("caseId")))
            if not seed_question:
                rows.append(_missing_question_row(case))
                continue
            if _clean_text(seed_question.get("questionId")) != _clean_text(case.get("caseId")):
                rows.append(_missing_question_row(case))
                continue
            rows.append(executor(case=case, seed_question=seed_question, papers_dir=papers_dir))

    fail_rows = sum(1 for row in rows if not bool(row.get("pass")))
    pass_rows = len(rows) - fail_rows
    expected_answerable_rows = sum(1 for row in rows if row.get("answerabilityExpectation") == "answerable")
    expected_no_answer_rows = sum(1 for row in rows if row.get("answerabilityExpectation") == "expected_no_answer")
    blocked_until_rows = sum(1 for row in rows if row.get("answerabilityExpectation") == "blocked_until_structured_evidence")
    unexpected_answerable_rows = sum(
        1
        for row in rows
        if row.get("answerabilityExpectation") != "answerable" and bool(row.get("observedAnswerable"))
    )
    no_answer_safety_fail_rows = sum(
        1
        for row in rows
        if row.get("answerabilityExpectation") != "answerable"
        and not bool(dict(row.get("dimensionStatuses") or {}).get("noAnswerSafety"))
    )
    private_path_leak_rows = sum(1 for row in rows if _contains_private_path(row))
    scores = [float(row.get("qualityScore") or 0.0) for row in rows]
    semantic_violations = sorted(set(dry_run_blockers + unsafe_blockers))
    if fail_rows:
        semantic_violations.append(f"controlled_execution_case_failures:{fail_rows}")
    if unexpected_answerable_rows:
        semantic_violations.append(f"unexpected_answerable_rows:{unexpected_answerable_rows}")
    if no_answer_safety_fail_rows:
        semantic_violations.append(f"no_answer_safety_fail_rows:{no_answer_safety_fail_rows}")
    if private_path_leak_rows or _contains_private_path(dry_run_report):
        semantic_violations.append("corpus_scale_controlled_execution_private_path_marker")
        private_path_leak_rows = max(private_path_leak_rows, 1)
    semantic_violations = sorted(set(semantic_violations))

    status = "ready" if not semantic_violations and pass_rows == len(cases) else "blocked"
    local_fake_llm_call_rows = sum(_int(row.get("localFakeLlmCallRows")) for row in rows)
    citation_count = sum(_int(row.get("citationCount")) for row in rows)
    span_rows = sum(_int(row.get("evidencePacketContractSpanRows")) for row in rows)
    counts = {
        "controlledExecutionRows": 1,
        "inputDryRunRows": 1 if dry_run_report else 0,
        "dryRunReadyInputRows": 1 if dry_run_report and not dry_run_blockers else 0,
        "inputCaseRows": len(cases),
        "attemptedCaseRows": len(rows),
        "executionPassRows": pass_rows,
        "executionFailRows": fail_rows,
        "expectedAnswerableRows": expected_answerable_rows,
        "expectedNoAnswerRows": expected_no_answer_rows,
        "blockedUntilStructuredEvidenceRows": blocked_until_rows,
        "observedAnswerableRows": sum(1 for row in rows if bool(row.get("observedAnswerable"))),
        "observedNoEvidenceRows": sum(1 for row in rows if row.get("observedStatus") == "no_evidence"),
        "unexpectedAnswerableRows": unexpected_answerable_rows,
        "noAnswerSafetyPassRows": sum(
            1
            for row in rows
            if row.get("answerabilityExpectation") != "answerable"
            and bool(dict(row.get("dimensionStatuses") or {}).get("noAnswerSafety"))
        ),
        "noAnswerSafetyFailRows": no_answer_safety_fail_rows,
        "schemaValidRows": sum(1 for row in rows if bool(row.get("schemaValid"))),
        "averageQualityScore": round(mean(scores), 6) if scores else 0.0,
        "minQualityScore": round(min(scores), 6) if scores else 0.0,
        "adapterRowsAdded": sum(_int(row.get("adapterRowsAdded")) for row in rows),
        "selectedEvidenceCount": sum(_int(row.get("selectedEvidenceCount")) for row in rows),
        "citationCount": citation_count,
        "evidencePacketContractSpanRows": span_rows,
        "localFakeLlmCallRows": local_fake_llm_call_rows,
        "noAnswerOrBlockedLlmCallRows": sum(
            _int(row.get("localFakeLlmCallRows"))
            for row in rows
            if row.get("answerabilityExpectation") != "answerable"
        ),
        "liveAnswerExecutionRows": len(rows),
        "answerPathInvokedRows": len(rows),
        "answerGeneratedRows": local_fake_llm_call_rows,
        "answerQualityMeasuredRows": len(rows),
        "answerQualityScoreComputedRows": len(rows),
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
        "schema": KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_CONTROLLED_EXECUTION_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "liveRunnerDryRunReportRef": DEFAULT_LIVE_RUNNER_DRY_RUN_REPORT.as_posix(),
            "corpusManifestRef": "eval/knowledgeos/fixtures/corpus_manifest.json",
            "papersDirRef": "papers_dir",
        },
        "policy": {
            "labsInternalOnly": True,
            "localFakeLlmOnly": True,
            "externalModelCallsAllowed": False,
            "judgeModelCallsAllowed": False,
            "answerTextExcludedFromReport": True,
            "citationPayloadExcludedFromReport": True,
            "sourcePayloadExcludedFromReport": True,
            "excerptExcludedFromReport": True,
            "publicDefaultPromotionAllowed": False,
        },
        "counts": counts,
        "gate": {
            "controlledExecutionReady": status == "ready",
            "dryRunReady": not dry_run_blockers,
            "allCasesPassed": fail_rows == 0 and len(rows) == len(cases),
            "noAnswerSafetyPassed": no_answer_safety_fail_rows == 0,
            "unexpectedAnswerableRowsZero": unexpected_answerable_rows == 0,
            "publicDefaultPromotionAllowed": False,
            "generalRcReady": False,
            "semanticViolations": semantic_violations,
        },
        "executionRows": rows,
        "checkRows": [
            {"checkId": "live_runner_dry_run", "status": "pass" if not dry_run_blockers else "fail", "blockers": dry_run_blockers},
            {"checkId": "unsafe_counters", "status": "pass" if not unsafe_blockers else "fail", "blockers": unsafe_blockers},
            {
                "checkId": "no_answer_safety",
                "status": "pass" if no_answer_safety_fail_rows == 0 else "fail",
                "blockers": [] if no_answer_safety_fail_rows == 0 else [f"no_answer_safety_fail_rows:{no_answer_safety_fail_rows}"],
            },
            {
                "checkId": "public_default_hold",
                "status": "pass",
                "blockers": [],
            },
        ],
        "warnings": [
            "controlled_execution_invokes_labs_internal_answer_path_with_local_fake_llm_only",
            "raw_question_answer_citation_source_and_excerpt_payloads_are_not_included",
            "public_default_promotion_remains_held",
        ],
    }


def render_knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# KnowledgeOS v0.1 RC Corpus-Scale Answer Quality Controlled Execution",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- attemptedCaseRows: `{counts.get('attemptedCaseRows')}`",
        f"- executionPassRows: `{counts.get('executionPassRows')}`",
        f"- executionFailRows: `{counts.get('executionFailRows')}`",
        f"- unexpectedAnswerableRows: `{counts.get('unexpectedAnswerableRows')}`",
        f"- noAnswerSafetyFailRows: `{counts.get('noAnswerSafetyFailRows')}`",
        f"- localFakeLlmCallRows: `{counts.get('localFakeLlmCallRows')}`",
        f"- liveAnswerExecutionRows: `{counts.get('liveAnswerExecutionRows')}`",
        f"- answerPathInvokedRows: `{counts.get('answerPathInvokedRows')}`",
        f"- externalLlmCallRows: `{counts.get('externalLlmCallRows')}`",
        f"- modelApiCallRows: `{counts.get('modelApiCallRows')}`",
        f"- judgeModelCallRows: `{counts.get('judgeModelCallRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Checks",
        "",
    ]
    for row in list(report.get("checkRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(f"- `{row.get('checkId')}`: `{row.get('status')}`; blockers=`{blockers}`")
    lines.extend(["", "## Failure Preview", ""])
    for row in list(report.get("executionRows") or []):
        if bool(row.get("pass")):
            continue
        lines.append(
            f"- `{row.get('caseId')}` category=`{row.get('questionCategory')}` "
            f"expectation=`{row.get('answerabilityExpectation')}` observedAnswerable=`{row.get('observedAnswerable')}` "
            f"failures=`{row.get('failureReasons')}`"
        )
        if len([line for line in lines if line.startswith("- `complex-paper-qa-seed")]) >= 12:
            break
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in _zero_counter_fields():
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_CONTROLLED_EXECUTION_SCHEMA_ID",
    "READY_DECISION",
    "BLOCKED_DECISION",
    "build_knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution",
    "write_knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution",
]
