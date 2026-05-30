"""Positive answer execution gate for KnowledgeOS v0.1 RC corpus-scale QA."""

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
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution import (
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_CONTROLLED_EXECUTION_SCHEMA_ID,
    READY_DECISION as CONTROLLED_EXECUTION_READY_DECISION,
)
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed import (
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_SECTION_PARAGRAPH_SEED_SCHEMA_ID,
    PROPOSED_ANSWERABILITY_EXPECTATION,
    PROPOSED_EXPECTED_EVIDENCE_TYPE,
    READY_DECISION as POSITIVE_SEED_READY_DECISION,
    SUPPORT_TERM_GROUPS_BY_CASE_ID,
    _evidence_text,
    _observed_source_ids,
    _quality_grade,
    _support_group_matches,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed import (
    ZERO_COUNTER_FIELDS,
    _clean_text,
    _contains_private_path,
    _int,
    _read_json,
    _sha256_text,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke import (
    DEFAULT_PAPERS_DIR,
    _build_searcher,
)


KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_ANSWER_EXECUTION_GATE_SCHEMA_ID = (
    "knowledge-hub.product.knowledgeos-v01-rc-corpus-scale-answer-quality-positive-answer-execution-gate.v1"
)

READY_DECISION = "knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate_ready"
BLOCKED_DECISION = "knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate_blocked"
NEXT_TRANCHE_READY = "corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution"
NEXT_TRANCHE_BLOCKED = "corpus_scale_answer_quality_positive_answer_execution_gate_repair"

DEFAULT_POSITIVE_SEED_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed.v1.json"
)
DEFAULT_CONTROLLED_EXECUTION_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution.v1.json"
)

MIN_POSITIVE_EXECUTION_PASS_ROWS = 4

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


def _unsafe_counter_blockers(report: dict[str, Any], prefix: str) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    for field in _zero_counter_fields():
        if _int(counts.get(field)) != 0:
            blockers.append(f"unsafe_counter_nonzero:{prefix}:{field}")
    return sorted(set(blockers))


def _positive_seed_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers = _schema_blockers(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_SECTION_PARAGRAPH_SEED_SCHEMA_ID,
        "positive_section_paragraph_seed",
    )
    if report.get("status") != "ready":
        blockers.append("positive_section_paragraph_seed_not_ready")
    if report.get("decision") != POSITIVE_SEED_READY_DECISION:
        blockers.append("positive_section_paragraph_seed_decision_not_ready")
    if report.get("nextRecommendedTranche") != "corpus_scale_answer_quality_positive_answer_execution_gate":
        blockers.append("positive_section_paragraph_seed_next_tranche_not_positive_answer_execution_gate")
    if _int(counts.get("positiveSeedRows")) < MIN_POSITIVE_EXECUTION_PASS_ROWS:
        blockers.append("positive_section_paragraph_seed_rows_below_minimum")
    if _int(counts.get("controlledExecutionUnexpectedAnswerableRows")) != 0:
        blockers.append("positive_section_paragraph_seed_no_answer_safety_regressed")
    if _int(counts.get("controlledExecutionNoAnswerSafetyFailRows")) != 0:
        blockers.append("positive_section_paragraph_seed_no_answer_safety_failures_present")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("positive_section_paragraph_seed_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("positive_section_paragraph_seed_schema_violations")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("positive_section_paragraph_seed_allows_public_default_promotion")
    if gate.get("noAnswerSafetyStillGreen") is not True:
        blockers.append("positive_section_paragraph_seed_no_answer_safety_not_green")
    return sorted(set(blockers))


def _controlled_execution_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers = _schema_blockers(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_CONTROLLED_EXECUTION_SCHEMA_ID,
        "controlled_execution",
    )
    if report.get("status") != "ready":
        blockers.append("controlled_execution_not_ready")
    if report.get("decision") != CONTROLLED_EXECUTION_READY_DECISION:
        blockers.append("controlled_execution_decision_not_ready")
    if _int(counts.get("attemptedCaseRows")) != 50:
        blockers.append("controlled_execution_attempted_rows_not_50")
    if _int(counts.get("executionFailRows")) != 0:
        blockers.append("controlled_execution_failures_present")
    if _int(counts.get("unexpectedAnswerableRows")) != 0:
        blockers.append("controlled_execution_unexpected_answerable_rows_present")
    if _int(counts.get("noAnswerSafetyFailRows")) != 0:
        blockers.append("controlled_execution_no_answer_safety_failures_present")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("controlled_execution_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("controlled_execution_schema_violations")
    return sorted(set(blockers))


def _seed_questions_by_id(seed_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in list(seed_report.get("questions") or []):
        if not isinstance(row, dict):
            continue
        question_id = _clean_text(row.get("questionId"))
        if question_id:
            out[question_id] = dict(row)
    return out


def _execution_row(
    *,
    seed_row: dict[str, Any],
    seed_question: dict[str, Any],
    papers_dir: str | Path,
) -> dict[str, Any]:
    paper_ids = [_clean_text(item) for item in list(seed_row.get("paperIds") or []) if _clean_text(item)]
    searcher, llm = _build_searcher(papers_dir=papers_dir)
    payload = build_paper_evidence_chunk_answer_preview(
        searcher,
        question=_clean_text(seed_question.get("question")),
        paper_ids=paper_ids,
        question_category=_clean_text(seed_row.get("questionCategory")),
        expected_evidence_type=PROPOSED_EXPECTED_EVIDENCE_TYPE,
        answerability_expectation=PROPOSED_ANSWERABILITY_EXPECTATION,
        allow_external=False,
    )
    validation = validate_payload(payload, PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID, strict=True)
    summary = dict(payload.get("evidencePacketSummary") or {})
    contract = dict(payload.get("evidencePacketContractSummary") or {})
    observed_ids = _observed_source_ids(payload)
    missing_ids = [paper_id for paper_id in paper_ids if paper_id not in observed_ids]
    evidence_text = _evidence_text(payload)
    support_groups = list(SUPPORT_TERM_GROUPS_BY_CASE_ID.get(_clean_text(seed_row.get("caseId")), ()))
    matched_groups = [group for group in support_groups if _support_group_matches(evidence_text, group)]
    missing_groups = [group for group in support_groups if group not in matched_groups]
    citation_count = _int(summary.get("citationCount"))
    span_rows = _int(contract.get("spanRows"))
    expected_min = max(2, len(paper_ids) * 2)
    answer_text = _clean_text(payload.get("answer"))
    dimensions = {
        "schemaValid": bool(validation.ok),
        "answerPayloadStatusOk": _clean_text(payload.get("status")) == "ok",
        "answerabilityExpectation": bool(payload.get("answerable")),
        "sourceCoverage": not missing_ids,
        "citationProvenance": citation_count >= expected_min and span_rows >= expected_min,
        "supportTermCoverage": not missing_groups,
        "answerGenerated": bool(answer_text) and int(llm.calls) == 1,
        "publicDefaultHeld": True,
    }
    passed = sum(1 for ok in dimensions.values() if ok)
    score = round(passed / len(dimensions), 6) if dimensions else 0.0
    blockers: list[str] = []
    if not validation.ok:
        blockers.append("preview_payload_schema_invalid")
    if _clean_text(payload.get("status")) != "ok":
        blockers.append("answer_payload_status_not_ok")
    if not bool(payload.get("answerable")):
        blockers.append("preview_not_answerable")
    if missing_ids:
        blockers.append("source_coverage_gap")
    if citation_count < expected_min or span_rows < expected_min:
        blockers.append("citation_or_span_count_below_minimum")
    if missing_groups:
        blockers.append("support_term_gap")
    if not bool(answer_text):
        blockers.append("answer_text_empty")
    if int(llm.calls) != 1:
        blockers.append("local_fake_llm_call_count_unexpected")
    row = {
        "caseIndex": _int(seed_row.get("caseIndex")),
        "caseId": _clean_text(seed_row.get("caseId")),
        "questionCategory": _clean_text(seed_row.get("questionCategory")),
        "expectedEvidenceType": PROPOSED_EXPECTED_EVIDENCE_TYPE,
        "answerabilityExpectation": PROPOSED_ANSWERABILITY_EXPECTATION,
        "paperIds": paper_ids,
        "questionSha256": _clean_text(seed_row.get("questionSha256")),
        "observedStatus": _clean_text(payload.get("status")),
        "observedAnswerable": bool(payload.get("answerable")),
        "adapterStatus": _clean_text(summary.get("adapterStatus")),
        "adapterRowsAdded": _int(summary.get("adapterRowsAdded")),
        "adapterCandidateRowsConsidered": _int(summary.get("adapterCandidateRowsConsidered")),
        "selectedEvidenceCount": _int(summary.get("selectedEvidenceCount")),
        "citationCount": citation_count,
        "evidencePacketContractSpanRows": span_rows,
        "localFakeLlmCallRows": int(llm.calls),
        "observedSourceIds": observed_ids,
        "missingSourceIds": missing_ids,
        "supportTermGroups": support_groups,
        "matchedSupportTermGroups": matched_groups,
        "missingSupportTermGroups": missing_groups,
        "dimensionStatuses": dimensions,
        "qualityScore": score,
        "qualityGrade": _quality_grade(score),
        "pass": not blockers,
        "answerTextSha256": _sha256_text(answer_text),
        "answerTextByteLength": len(answer_text.encode("utf-8")),
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "failureReasons": sorted(set(blockers)),
    }
    if _contains_private_path(row):
        row["failureReasons"] = sorted(set([*row["failureReasons"], "private_path_leak"]))
        row["pass"] = False
    return row


def _missing_seed_question_row(seed_row: dict[str, Any]) -> dict[str, Any]:
    support_groups = list(SUPPORT_TERM_GROUPS_BY_CASE_ID.get(_clean_text(seed_row.get("caseId")), ()))
    return {
        "caseIndex": _int(seed_row.get("caseIndex")),
        "caseId": _clean_text(seed_row.get("caseId")),
        "questionCategory": _clean_text(seed_row.get("questionCategory")),
        "expectedEvidenceType": PROPOSED_EXPECTED_EVIDENCE_TYPE,
        "answerabilityExpectation": PROPOSED_ANSWERABILITY_EXPECTATION,
        "paperIds": [_clean_text(item) for item in list(seed_row.get("paperIds") or []) if _clean_text(item)],
        "questionSha256": _clean_text(seed_row.get("questionSha256")),
        "observedStatus": "not_executed",
        "observedAnswerable": False,
        "adapterStatus": "",
        "adapterRowsAdded": 0,
        "adapterCandidateRowsConsidered": 0,
        "selectedEvidenceCount": 0,
        "citationCount": 0,
        "evidencePacketContractSpanRows": 0,
        "localFakeLlmCallRows": 0,
        "observedSourceIds": [],
        "missingSourceIds": [_clean_text(item) for item in list(seed_row.get("paperIds") or []) if _clean_text(item)],
        "supportTermGroups": support_groups,
        "matchedSupportTermGroups": [],
        "missingSupportTermGroups": support_groups,
        "dimensionStatuses": {
            "schemaValid": False,
            "answerPayloadStatusOk": False,
            "answerabilityExpectation": False,
            "sourceCoverage": False,
            "citationProvenance": False,
            "supportTermCoverage": False,
            "answerGenerated": False,
            "publicDefaultHeld": True,
        },
        "qualityScore": 0.0,
        "qualityGrade": "fail",
        "pass": False,
        "answerTextSha256": _sha256_text(""),
        "answerTextByteLength": 0,
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "failureReasons": ["seed_question_missing_for_case"],
    }


def build_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate(
    *,
    positive_seed_report_path: str | Path = DEFAULT_POSITIVE_SEED_REPORT,
    controlled_execution_report_path: str | Path = DEFAULT_CONTROLLED_EXECUTION_REPORT,
    corpus_manifest: str | Path = DEFAULT_CORPUS_MANIFEST,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    positive_seed_report: dict[str, Any] | None = None,
    controlled_execution_report: dict[str, Any] | None = None,
    seed_pack_report: dict[str, Any] | None = None,
    execute_case: Callable[..., dict[str, Any]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    positive_seed = dict(positive_seed_report or _read_json(positive_seed_report_path))
    controlled = dict(controlled_execution_report or _read_json(controlled_execution_report_path))
    seed_pack = dict(seed_pack_report or build_complex_qa_seed_pack(corpus_manifest=corpus_manifest, target_paper_count=20))
    seed_by_id = _seed_questions_by_id(seed_pack)
    seed_rows = [dict(row or {}) for row in list(positive_seed.get("positiveSeedRows") or [])]

    positive_seed_blockers = _positive_seed_blockers(positive_seed)
    controlled_blockers = _controlled_execution_blockers(controlled)
    unsafe_blockers = sorted(
        set(
            _unsafe_counter_blockers(positive_seed, "positive_seed")
            + _unsafe_counter_blockers(controlled, "controlled_execution")
        )
    )
    input_blockers = sorted(set(positive_seed_blockers + controlled_blockers + unsafe_blockers))

    executor = execute_case or _execution_row
    rows: list[dict[str, Any]] = []
    if not input_blockers:
        for seed_row in seed_rows:
            seed_question = seed_by_id.get(_clean_text(seed_row.get("caseId")))
            if not seed_question:
                rows.append(_missing_seed_question_row(seed_row))
            else:
                rows.append(executor(seed_row=seed_row, seed_question=seed_question, papers_dir=papers_dir))

    pass_rows = sum(1 for row in rows if bool(row.get("pass")))
    fail_rows = len(rows) - pass_rows
    partial_rows = sum(1 for row in rows if row.get("qualityGrade") == "partial")
    scores = [float(row.get("qualityScore") or 0.0) for row in rows]
    seed_counts = dict(positive_seed.get("counts") or {})
    controlled_counts = dict(controlled.get("counts") or {})
    private_path_leak_rows = sum(1 for row in rows if _contains_private_path(row))
    if _contains_private_path(positive_seed) or _contains_private_path(controlled):
        private_path_leak_rows = max(private_path_leak_rows, 1)

    semantic_violations = sorted(set(input_blockers))
    if len(seed_rows) < MIN_POSITIVE_EXECUTION_PASS_ROWS:
        semantic_violations.append("positive_execution_input_seed_rows_below_minimum")
    if pass_rows < MIN_POSITIVE_EXECUTION_PASS_ROWS:
        semantic_violations.append("positive_answer_execution_pass_rows_below_minimum")
    if fail_rows:
        semantic_violations.append(f"positive_answer_execution_fail_rows:{fail_rows}")
    if _int(controlled_counts.get("unexpectedAnswerableRows")) != 0:
        semantic_violations.append("no_answer_safety_unexpected_answerable_rows_present")
    if _int(controlled_counts.get("noAnswerSafetyFailRows")) != 0:
        semantic_violations.append("no_answer_safety_fail_rows_present")
    if private_path_leak_rows:
        semantic_violations.append("positive_answer_execution_private_path_marker")
    semantic_violations = sorted(set(semantic_violations))
    status = "ready" if not semantic_violations else "blocked"

    counts = {
        "positiveAnswerExecutionGateRows": 1,
        "inputPositiveSeedReportRows": 1 if positive_seed else 0,
        "positiveSeedReadyInputRows": 1 if positive_seed and not positive_seed_blockers else 0,
        "controlledExecutionReadyRows": 1 if controlled and not controlled_blockers else 0,
        "inputPositiveSeedRows": len(seed_rows),
        "attemptedPositiveAnswerRows": len(rows),
        "positiveAnswerPassRows": pass_rows,
        "positiveAnswerPartialRows": partial_rows,
        "positiveAnswerFailRows": fail_rows,
        "positiveMethodComparisonRows": sum(1 for row in rows if row.get("questionCategory") == "method_comparison_qa"),
        "positiveLimitationRows": sum(1 for row in rows if row.get("questionCategory") == "limitation_qa"),
        "heldExpectedNoAnswerRows": _int(seed_counts.get("heldExpectedNoAnswerRows")),
        "heldStructuredModalityRows": _int(seed_counts.get("heldStructuredModalityRows")),
        "heldProbeRows": _int(seed_counts.get("positiveProbeHeldRows")),
        "controlledExecutionUnexpectedAnswerableRows": _int(controlled_counts.get("unexpectedAnswerableRows")),
        "controlledExecutionNoAnswerSafetyFailRows": _int(controlled_counts.get("noAnswerSafetyFailRows")),
        "sourceCoveragePassRows": sum(1 for row in rows if bool(dict(row.get("dimensionStatuses") or {}).get("sourceCoverage"))),
        "supportTermCoveragePassRows": sum(
            1 for row in rows if bool(dict(row.get("dimensionStatuses") or {}).get("supportTermCoverage"))
        ),
        "citationProvenancePassRows": sum(
            1 for row in rows if bool(dict(row.get("dimensionStatuses") or {}).get("citationProvenance"))
        ),
        "answerGeneratedPassRows": sum(
            1 for row in rows if bool(dict(row.get("dimensionStatuses") or {}).get("answerGenerated"))
        ),
        "averageQualityScore": round(mean(scores), 6) if scores else 0.0,
        "minQualityScore": round(min(scores), 6) if scores else 0.0,
        "adapterRowsAdded": sum(_int(row.get("adapterRowsAdded")) for row in rows),
        "selectedEvidenceCount": sum(_int(row.get("selectedEvidenceCount")) for row in rows),
        "citationCount": sum(_int(row.get("citationCount")) for row in rows),
        "evidencePacketContractSpanRows": sum(_int(row.get("evidencePacketContractSpanRows")) for row in rows),
        "localFakeLlmCallRows": sum(_int(row.get("localFakeLlmCallRows")) for row in rows),
        "answerTextHashRows": sum(1 for row in rows if _clean_text(row.get("answerTextSha256"))),
        "answerTextIncludedRows": 0,
        "liveAnswerExecutionRows": len(rows),
        "answerPathInvokedRows": len(rows),
        "answerGeneratedRows": sum(_int(row.get("localFakeLlmCallRows")) for row in rows),
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
        "schema": KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_ANSWER_EXECUTION_GATE_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "positiveSeedReportRef": DEFAULT_POSITIVE_SEED_REPORT.as_posix(),
            "controlledExecutionReportRef": DEFAULT_CONTROLLED_EXECUTION_REPORT.as_posix(),
            "corpusManifestRef": "eval/knowledgeos/fixtures/corpus_manifest.json",
            "papersDirRef": "papers_dir",
        },
        "policy": {
            "labsInternalOnly": True,
            "reportOnly": True,
            "localFakeLlmOnly": True,
            "externalModelCallsAllowed": False,
            "judgeModelCallsAllowed": False,
            "answerStyleJudged": False,
            "answerTextExcludedFromReport": True,
            "citationPayloadExcludedFromReport": True,
            "sourcePayloadExcludedFromReport": True,
            "excerptExcludedFromReport": True,
            "publicDefaultPromotionAllowed": False,
            "tableEquationFigurePromotionAllowed": False,
        },
        "qualityDimensions": [
            "schema_valid",
            "answer_payload_status_ok",
            "answerability_expectation",
            "source_coverage",
            "citation_provenance",
            "support_term_coverage",
            "answer_generated",
            "public_default_held",
        ],
        "counts": counts,
        "gate": {
            "positiveAnswerExecutionGateReady": status == "ready",
            "positiveSeedReady": not positive_seed_blockers,
            "controlledExecutionReady": not controlled_blockers,
            "noAnswerSafetyStillGreen": _int(controlled_counts.get("unexpectedAnswerableRows")) == 0
            and _int(controlled_counts.get("noAnswerSafetyFailRows")) == 0,
            "allPositiveAnswersPassed": pass_rows == len(seed_rows) and bool(seed_rows),
            "positiveAnswerPassRowsPresent": pass_rows >= MIN_POSITIVE_EXECUTION_PASS_ROWS,
            "expectedNoAnswerRowsHeld": _int(seed_counts.get("heldExpectedNoAnswerRows")) == _int(controlled_counts.get("expectedNoAnswerRows")),
            "structuredModalityRowsHeld": _int(seed_counts.get("heldStructuredModalityRows")) > 0,
            "publicDefaultPromotionAllowed": False,
            "generalRcReady": False,
            "semanticViolations": semantic_violations,
        },
        "executionRows": rows,
        "checkRows": [
            {"checkId": "positive_section_paragraph_seed", "status": "pass" if not positive_seed_blockers else "fail", "blockers": positive_seed_blockers},
            {"checkId": "controlled_execution_no_answer_safety", "status": "pass" if not controlled_blockers else "fail", "blockers": controlled_blockers},
            {"checkId": "unsafe_counters", "status": "pass" if not unsafe_blockers else "fail", "blockers": unsafe_blockers},
            {
                "checkId": "positive_answer_execution",
                "status": "pass" if pass_rows == len(seed_rows) and pass_rows >= MIN_POSITIVE_EXECUTION_PASS_ROWS else "fail",
                "blockers": []
                if pass_rows == len(seed_rows) and pass_rows >= MIN_POSITIVE_EXECUTION_PASS_ROWS
                else ["positive_answer_execution_not_all_rows_passed"],
            },
            {"checkId": "public_default_hold", "status": "pass", "blockers": []},
        ],
        "warnings": [
            "positive_answer_execution_uses_labs_internal_opt_in_answer_path_with_local_fake_llm_only",
            "runner_scores_evidence_support_citation_contract_and_answer_generation_not_human_answer_style",
            "raw_question_answer_citation_source_and_excerpt_payloads_are_not_included",
            "expected_no_answer_and_structured_modality_rows_remain_held",
            "public_default_promotion_remains_held",
        ],
    }


def render_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# KnowledgeOS v0.1 RC Corpus-Scale Positive Answer Execution Gate",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- inputPositiveSeedRows: `{counts.get('inputPositiveSeedRows')}`",
        f"- attemptedPositiveAnswerRows: `{counts.get('attemptedPositiveAnswerRows')}`",
        f"- positiveAnswerPassRows: `{counts.get('positiveAnswerPassRows')}`",
        f"- positiveAnswerPartialRows: `{counts.get('positiveAnswerPartialRows')}`",
        f"- positiveAnswerFailRows: `{counts.get('positiveAnswerFailRows')}`",
        f"- averageQualityScore: `{counts.get('averageQualityScore')}`",
        f"- minQualityScore: `{counts.get('minQualityScore')}`",
        f"- heldExpectedNoAnswerRows: `{counts.get('heldExpectedNoAnswerRows')}`",
        f"- heldStructuredModalityRows: `{counts.get('heldStructuredModalityRows')}`",
        f"- controlledExecutionUnexpectedAnswerableRows: `{counts.get('controlledExecutionUnexpectedAnswerableRows')}`",
        f"- controlledExecutionNoAnswerSafetyFailRows: `{counts.get('controlledExecutionNoAnswerSafetyFailRows')}`",
        f"- citationCount: `{counts.get('citationCount')}`",
        f"- evidencePacketContractSpanRows: `{counts.get('evidencePacketContractSpanRows')}`",
        f"- localFakeLlmCallRows: `{counts.get('localFakeLlmCallRows')}`",
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
    lines.extend(["", "## Execution Rows", ""])
    for row in list(report.get("executionRows") or []):
        lines.append(
            f"- `{row.get('caseId')}` category=`{row.get('questionCategory')}` papers=`{row.get('paperIds')}` "
            f"pass=`{row.get('pass')}` score=`{row.get('qualityScore')}` "
            f"citations=`{row.get('citationCount')}` spans=`{row.get('evidencePacketContractSpanRows')}` "
            f"failures=`{row.get('failureReasons')}`"
        )
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in _zero_counter_fields():
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_ANSWER_EXECUTION_GATE_SCHEMA_ID",
    "READY_DECISION",
    "BLOCKED_DECISION",
    "build_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate",
    "write_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate",
]
