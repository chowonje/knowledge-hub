"""Refresh the corpus-scale quality gate after positive answer execution."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from typing import Any

from knowledge_hub.papers.complex_qa_seed_pack import DEFAULT_CORPUS_MANIFEST, build_complex_qa_seed_pack
from knowledge_hub.papers.evidence_chunk_answer_preview import build_evidence_chunk_query_plan
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate import (
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_ANSWER_EXECUTION_GATE_SCHEMA_ID,
    READY_DECISION as POSITIVE_EXECUTION_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed import (
    ZERO_COUNTER_FIELDS,
    _clean_text,
    _contains_private_path,
    _int,
    _read_json,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke import (
    DEFAULT_PAPERS_DIR,
    _build_searcher,
)
from knowledge_hub.core.schema_validator import validate_payload


KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_REFRESH_AFTER_POSITIVE_ANSWER_EXECUTION_SCHEMA_ID = (
    "knowledge-hub.product.knowledgeos-v01-rc-corpus-scale-answer-quality-gate-refresh-after-positive-answer-execution.v1"
)

READY_DECISION = "knowledgeos_v01_rc_corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution_ready"
BLOCKED_DECISION = "knowledgeos_v01_rc_corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution_blocked"
NEXT_TRANCHE_READY = "knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review"
NEXT_TRANCHE_BLOCKED = "corpus_scale_answer_quality_positive_provenance_refresh_repair"

DEFAULT_POSITIVE_EXECUTION_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate.v1.json"
)

MIN_PROVENANCE_PASS_ROWS = 4
EVIDENCE_PACKET_CONTRACT_SCHEMA_IDS = frozenset(
    {
        "knowledge-hub.evidence-packet.v1",
        "knowledge-hub.evidence-packet-contract.v1",
    }
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


def _unsafe_counter_blockers(report: dict[str, Any], prefix: str) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    for field in _zero_counter_fields():
        if _int(counts.get(field)) != 0:
            blockers.append(f"unsafe_counter_nonzero:{prefix}:{field}")
    return sorted(set(blockers))


def _positive_execution_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_ANSWER_EXECUTION_GATE_SCHEMA_ID:
        blockers.append("positive_answer_execution_schema_mismatch")
        return blockers
    validation = validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_ANSWER_EXECUTION_GATE_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        blockers.append("positive_answer_execution_schema_validation_failed")
    if report.get("status") != "ready":
        blockers.append("positive_answer_execution_not_ready")
    if report.get("decision") != POSITIVE_EXECUTION_READY_DECISION:
        blockers.append("positive_answer_execution_decision_not_ready")
    if report.get("nextRecommendedTranche") != "corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution":
        blockers.append("positive_answer_execution_next_tranche_not_refresh")
    if _int(counts.get("positiveAnswerPassRows")) < MIN_PROVENANCE_PASS_ROWS:
        blockers.append("positive_answer_pass_rows_below_minimum")
    if _int(counts.get("positiveAnswerFailRows")) != 0:
        blockers.append("positive_answer_failures_present")
    if _int(counts.get("controlledExecutionUnexpectedAnswerableRows")) != 0:
        blockers.append("controlled_execution_unexpected_answerable_rows_present")
    if _int(counts.get("controlledExecutionNoAnswerSafetyFailRows")) != 0:
        blockers.append("controlled_execution_no_answer_safety_failures_present")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("positive_answer_execution_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("positive_answer_execution_schema_violations")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("positive_answer_execution_allows_public_default_promotion")
    if gate.get("noAnswerSafetyStillGreen") is not True:
        blockers.append("positive_answer_execution_no_answer_safety_not_green")
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


def _valid_hash(value: Any) -> bool:
    token = _clean_text(value)
    return token.startswith("sha256:") and len(token) == 71


def _valid_chars(start: Any, end: Any) -> bool:
    try:
        start_i = int(start)
        end_i = int(end)
    except Exception:
        return False
    return start_i >= 0 and end_i > start_i


def _span_proof_row(span: dict[str, Any]) -> dict[str, Any]:
    derivative = dict(span.get("derivativeSource") or span.get("derivative_source") or {})
    source_hash = _clean_text(span.get("sourceContentHash"))
    locator = _clean_text(span.get("spanLocator"))
    char_start = span.get("charStart")
    char_end = span.get("charEnd")
    source_ref = _clean_text(span.get("sourceRef"))
    return {
        "spanRef": _clean_text(span.get("spanRef")),
        "sourceId": _clean_text(span.get("sourceId") or span.get("source_id")),
        "sourceRef": source_ref,
        "sourceContentHash": source_hash,
        "sourceContentHashAvailable": bool(span.get("sourceContentHashAvailable")),
        "spanLocator": locator,
        "spanOffsetAvailable": bool(span.get("spanOffsetAvailable")),
        "charStart": _int(char_start),
        "charEnd": _int(char_end),
        "evidenceKind": _clean_text(span.get("evidenceKind")),
        "candidateRecordId": _clean_text(derivative.get("candidateRecordId")),
        "candidateStoreRef": _clean_text(derivative.get("candidateStoreRef")),
        "strictProvenance": bool(span.get("sourceContentHashAvailable"))
        and bool(span.get("spanOffsetAvailable"))
        and _valid_hash(source_hash)
        and locator.startswith("chars:")
        and _valid_chars(char_start, char_end)
        and not _contains_private_path({"sourceRef": source_ref, "candidateStoreRef": derivative.get("candidateStoreRef")}),
        "excerptIncludedInReport": False,
    }


def _answer_contract_citation_provenance(citation: dict[str, Any]) -> bool:
    return _valid_hash(citation.get("content_hash")) and _valid_chars(citation.get("char_start"), citation.get("char_end"))


def _provenance_row(
    *,
    execution_row: dict[str, Any],
    seed_question: dict[str, Any],
    papers_dir: str | Path,
) -> dict[str, Any]:
    paper_ids = [_clean_text(item) for item in list(execution_row.get("paperIds") or []) if _clean_text(item)]
    searcher, llm = _build_searcher(papers_dir=papers_dir)
    payload = searcher.generate_answer(
        _clean_text(seed_question.get("question")),
        top_k=8,
        source_type="paper",
        retrieval_mode="semantic",
        alpha=0.7,
        allow_external=False,
        ask_v2_mode="claim_first",
        query_plan=build_evidence_chunk_query_plan(
            paper_ids,
            question_category=_clean_text(execution_row.get("questionCategory")),
            expected_evidence_type="section_paragraph",
            answerability_expectation="answerable",
        ),
    )
    evidence_contract = dict(payload.get("evidencePacketContract") or {})
    answer_contract = dict(payload.get("answerContract") or {})
    spans = [dict(span or {}) for span in list(evidence_contract.get("spans") or [])]
    span_rows = [_span_proof_row(span) for span in spans]
    strict_span_rows = [row for row in span_rows if bool(row.get("strictProvenance"))]
    answer_citations = [dict(row or {}) for row in list(answer_contract.get("citations") or [])]
    answer_citation_provenance_rows = sum(1 for row in answer_citations if _answer_contract_citation_provenance(row))
    observed_source_ids = sorted({row.get("sourceId") for row in span_rows if _clean_text(row.get("sourceId"))})
    missing_source_ids = [paper_id for paper_id in paper_ids if paper_id not in observed_source_ids]
    min_required = max(2, len(paper_ids) * 2)
    dimensions = {
        "answerPayloadStatusOk": _clean_text(payload.get("status")) == "ok",
        "evidencePacketContractPresent": evidence_contract.get("schema") in EVIDENCE_PACKET_CONTRACT_SCHEMA_IDS,
        "answerContractPresent": answer_contract.get("schema") == "knowledge-hub.answer-contract.v1",
        "sourceCoverage": not missing_source_ids,
        "sourceHashCoverage": len(strict_span_rows) >= min_required,
        "charsLocatorCoverage": len(strict_span_rows) >= min_required,
        "answerContractCitationProvenance": answer_citation_provenance_rows >= min_required,
        "publicDefaultHeld": True,
    }
    blockers: list[str] = []
    if not dimensions["answerPayloadStatusOk"]:
        blockers.append("answer_payload_status_not_ok")
    if not dimensions["evidencePacketContractPresent"]:
        blockers.append("evidence_packet_contract_missing")
    if not dimensions["answerContractPresent"]:
        blockers.append("answer_contract_missing")
    if missing_source_ids:
        blockers.append("source_coverage_gap")
    if len(strict_span_rows) < min_required:
        blockers.append("strict_provenance_span_rows_below_minimum")
    if answer_citation_provenance_rows < min_required:
        blockers.append("answer_contract_citation_provenance_below_minimum")
    if int(llm.calls) != 1:
        blockers.append("local_fake_llm_call_count_unexpected")
    if _contains_private_path({"row": execution_row, "spanRows": span_rows}):
        blockers.append("private_path_leak")
    passed = sum(1 for ok in dimensions.values() if ok)
    score = round(passed / len(dimensions), 6) if dimensions else 0.0
    return {
        "caseIndex": _int(execution_row.get("caseIndex")),
        "caseId": _clean_text(execution_row.get("caseId")),
        "questionCategory": _clean_text(execution_row.get("questionCategory")),
        "paperIds": paper_ids,
        "questionSha256": _clean_text(execution_row.get("questionSha256")),
        "observedStatus": _clean_text(payload.get("status")),
        "localFakeLlmCallRows": int(llm.calls),
        "observedSourceIds": observed_source_ids,
        "missingSourceIds": missing_source_ids,
        "minRequiredProvenanceRows": min_required,
        "evidencePacketContractSpanRows": len(span_rows),
        "strictProvenanceSpanRows": len(strict_span_rows),
        "sourceContentHashRows": sum(1 for row in span_rows if _valid_hash(row.get("sourceContentHash"))),
        "charsLocatorRows": sum(1 for row in span_rows if str(row.get("spanLocator") or "").startswith("chars:")),
        "answerContractCitationRows": len(answer_citations),
        "answerContractCitationProvenanceRows": answer_citation_provenance_rows,
        "dimensionStatuses": dimensions,
        "qualityScore": score,
        "qualityGrade": "pass" if score == 1.0 else ("partial" if score >= 0.6 else "fail"),
        "pass": not blockers,
        "spanProofRows": span_rows,
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "failureReasons": sorted(set(blockers)),
    }


def _missing_seed_question_row(execution_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "caseIndex": _int(execution_row.get("caseIndex")),
        "caseId": _clean_text(execution_row.get("caseId")),
        "questionCategory": _clean_text(execution_row.get("questionCategory")),
        "paperIds": list(execution_row.get("paperIds") or []),
        "questionSha256": _clean_text(execution_row.get("questionSha256")),
        "observedStatus": "not_executed",
        "localFakeLlmCallRows": 0,
        "observedSourceIds": [],
        "missingSourceIds": list(execution_row.get("paperIds") or []),
        "minRequiredProvenanceRows": max(2, len(list(execution_row.get("paperIds") or [])) * 2),
        "evidencePacketContractSpanRows": 0,
        "strictProvenanceSpanRows": 0,
        "sourceContentHashRows": 0,
        "charsLocatorRows": 0,
        "answerContractCitationRows": 0,
        "answerContractCitationProvenanceRows": 0,
        "dimensionStatuses": {
            "answerPayloadStatusOk": False,
            "evidencePacketContractPresent": False,
            "answerContractPresent": False,
            "sourceCoverage": False,
            "sourceHashCoverage": False,
            "charsLocatorCoverage": False,
            "answerContractCitationProvenance": False,
            "publicDefaultHeld": True,
        },
        "qualityScore": 0.0,
        "qualityGrade": "fail",
        "pass": False,
        "spanProofRows": [],
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "failureReasons": ["seed_question_missing_for_case"],
    }


def build_knowledgeos_v01_rc_corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution(
    *,
    positive_execution_report_path: str | Path = DEFAULT_POSITIVE_EXECUTION_REPORT,
    corpus_manifest: str | Path = DEFAULT_CORPUS_MANIFEST,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    positive_execution_report: dict[str, Any] | None = None,
    seed_pack_report: dict[str, Any] | None = None,
    execute_case: Callable[..., dict[str, Any]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    positive_execution = dict(positive_execution_report or _read_json(positive_execution_report_path))
    seed_pack = dict(seed_pack_report or build_complex_qa_seed_pack(corpus_manifest=corpus_manifest, target_paper_count=20))
    seed_by_id = _seed_questions_by_id(seed_pack)
    execution_seed_rows = [dict(row or {}) for row in list(positive_execution.get("executionRows") or [])]

    input_blockers = sorted(
        set(_positive_execution_blockers(positive_execution) + _unsafe_counter_blockers(positive_execution, "positive_execution"))
    )
    executor = execute_case or _provenance_row
    rows: list[dict[str, Any]] = []
    if not input_blockers:
        for execution_row in execution_seed_rows:
            seed_question = seed_by_id.get(_clean_text(execution_row.get("caseId")))
            if not seed_question:
                rows.append(_missing_seed_question_row(execution_row))
            else:
                rows.append(executor(execution_row=execution_row, seed_question=seed_question, papers_dir=papers_dir))

    pass_rows = sum(1 for row in rows if bool(row.get("pass")))
    fail_rows = len(rows) - pass_rows
    scores = [float(row.get("qualityScore") or 0.0) for row in rows]
    positive_counts = dict(positive_execution.get("counts") or {})
    private_path_leak_rows = sum(1 for row in rows if _contains_private_path(row))
    if _contains_private_path(positive_execution):
        private_path_leak_rows = max(private_path_leak_rows, 1)
    semantic_violations = sorted(set(input_blockers))
    if pass_rows < MIN_PROVENANCE_PASS_ROWS:
        semantic_violations.append("provenance_pass_rows_below_minimum")
    if fail_rows:
        semantic_violations.append(f"provenance_fail_rows:{fail_rows}")
    if private_path_leak_rows:
        semantic_violations.append("positive_provenance_refresh_private_path_marker")
    semantic_violations = sorted(set(semantic_violations))
    status = "ready" if not semantic_violations else "blocked"

    counts = {
        "qualityGateRefreshRows": 1,
        "inputPositiveExecutionReportRows": 1 if positive_execution else 0,
        "positiveExecutionReadyInputRows": 1 if positive_execution and not input_blockers else 0,
        "inputPositiveExecutionRows": len(execution_seed_rows),
        "attemptedProvenanceRows": len(rows),
        "provenancePassRows": pass_rows,
        "provenanceFailRows": fail_rows,
        "positiveSectionParagraphQualityCompleteRows": 1 if status == "ready" else 0,
        "positiveMethodComparisonRows": sum(1 for row in rows if row.get("questionCategory") == "method_comparison_qa"),
        "positiveLimitationRows": sum(1 for row in rows if row.get("questionCategory") == "limitation_qa"),
        "heldExpectedNoAnswerRows": _int(positive_counts.get("heldExpectedNoAnswerRows")),
        "heldStructuredModalityRows": _int(positive_counts.get("heldStructuredModalityRows")),
        "controlledExecutionUnexpectedAnswerableRows": _int(positive_counts.get("controlledExecutionUnexpectedAnswerableRows")),
        "controlledExecutionNoAnswerSafetyFailRows": _int(positive_counts.get("controlledExecutionNoAnswerSafetyFailRows")),
        "sourceCoveragePassRows": sum(1 for row in rows if bool(dict(row.get("dimensionStatuses") or {}).get("sourceCoverage"))),
        "sourceHashCoveragePassRows": sum(
            1 for row in rows if bool(dict(row.get("dimensionStatuses") or {}).get("sourceHashCoverage"))
        ),
        "charsLocatorCoveragePassRows": sum(
            1 for row in rows if bool(dict(row.get("dimensionStatuses") or {}).get("charsLocatorCoverage"))
        ),
        "answerContractCitationProvenancePassRows": sum(
            1 for row in rows if bool(dict(row.get("dimensionStatuses") or {}).get("answerContractCitationProvenance"))
        ),
        "averageQualityScore": round(mean(scores), 6) if scores else 0.0,
        "minQualityScore": round(min(scores), 6) if scores else 0.0,
        "evidencePacketContractSpanRows": sum(_int(row.get("evidencePacketContractSpanRows")) for row in rows),
        "strictProvenanceSpanRows": sum(_int(row.get("strictProvenanceSpanRows")) for row in rows),
        "sourceContentHashRows": sum(_int(row.get("sourceContentHashRows")) for row in rows),
        "charsLocatorRows": sum(_int(row.get("charsLocatorRows")) for row in rows),
        "answerContractCitationRows": sum(_int(row.get("answerContractCitationRows")) for row in rows),
        "answerContractCitationProvenanceRows": sum(_int(row.get("answerContractCitationProvenanceRows")) for row in rows),
        "localFakeLlmCallRows": sum(_int(row.get("localFakeLlmCallRows")) for row in rows),
        "liveAnswerExecutionRows": len(rows),
        "answerPathInvokedRows": len(rows),
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
        "schema": KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_REFRESH_AFTER_POSITIVE_ANSWER_EXECUTION_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "positiveExecutionReportRef": DEFAULT_POSITIVE_EXECUTION_REPORT.as_posix(),
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
            "answerContractRawCitationPayloadExcluded": True,
            "sourcePayloadExcludedFromReport": True,
            "excerptExcludedFromReport": True,
            "publicDefaultPromotionAllowed": False,
            "tableEquationFigurePromotionAllowed": False,
        },
        "qualityDimensions": [
            "answer_payload_status_ok",
            "evidence_packet_contract_present",
            "answer_contract_present",
            "source_coverage",
            "source_hash_coverage",
            "chars_locator_coverage",
            "answer_contract_citation_provenance",
            "public_default_held",
        ],
        "counts": counts,
        "gate": {
            "qualityGateRefreshReady": status == "ready",
            "positiveExecutionReady": not input_blockers,
            "noAnswerSafetyStillGreen": _int(positive_counts.get("controlledExecutionUnexpectedAnswerableRows")) == 0
            and _int(positive_counts.get("controlledExecutionNoAnswerSafetyFailRows")) == 0,
            "allPositiveProvenancePassed": pass_rows == len(execution_seed_rows) and bool(execution_seed_rows),
            "positiveSectionParagraphQualityComplete": status == "ready",
            "expectedNoAnswerRowsHeld": _int(positive_counts.get("heldExpectedNoAnswerRows")) == 17,
            "structuredModalityRowsHeld": _int(positive_counts.get("heldStructuredModalityRows")) > 0,
            "publicDefaultPromotionAllowed": False,
            "generalRcReady": False,
            "semanticViolations": semantic_violations,
        },
        "provenanceRows": rows,
        "checkRows": [
            {"checkId": "positive_answer_execution", "status": "pass" if not input_blockers else "fail", "blockers": input_blockers},
            {
                "checkId": "hash_and_chars_provenance",
                "status": "pass" if pass_rows == len(execution_seed_rows) and pass_rows >= MIN_PROVENANCE_PASS_ROWS else "fail",
                "blockers": []
                if pass_rows == len(execution_seed_rows) and pass_rows >= MIN_PROVENANCE_PASS_ROWS
                else ["positive_provenance_not_all_rows_passed"],
            },
            {"checkId": "public_default_hold", "status": "pass", "blockers": []},
        ],
        "warnings": [
            "refresh_scores_hash_chars_locator_and_citation_contract_provenance_not_human_answer_style",
            "raw_answer_text_answer_contract_quotes_citation_sources_and_excerpts_are_not_included",
            "expected_no_answer_and_structured_modality_rows_remain_held",
            "public_default_promotion_remains_held",
        ],
    }


def render_knowledgeos_v01_rc_corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# KnowledgeOS v0.1 RC Corpus-Scale Quality Gate Refresh After Positive Answer Execution",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- inputPositiveExecutionRows: `{counts.get('inputPositiveExecutionRows')}`",
        f"- provenancePassRows: `{counts.get('provenancePassRows')}`",
        f"- provenanceFailRows: `{counts.get('provenanceFailRows')}`",
        f"- positiveSectionParagraphQualityCompleteRows: `{counts.get('positiveSectionParagraphQualityCompleteRows')}`",
        f"- heldExpectedNoAnswerRows: `{counts.get('heldExpectedNoAnswerRows')}`",
        f"- heldStructuredModalityRows: `{counts.get('heldStructuredModalityRows')}`",
        f"- strictProvenanceSpanRows: `{counts.get('strictProvenanceSpanRows')}`",
        f"- sourceContentHashRows: `{counts.get('sourceContentHashRows')}`",
        f"- charsLocatorRows: `{counts.get('charsLocatorRows')}`",
        f"- answerContractCitationProvenanceRows: `{counts.get('answerContractCitationProvenanceRows')}`",
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
    lines.extend(["", "## Provenance Rows", ""])
    for row in list(report.get("provenanceRows") or []):
        lines.append(
            f"- `{row.get('caseId')}` papers=`{row.get('paperIds')}` pass=`{row.get('pass')}` "
            f"strictSpans=`{row.get('strictProvenanceSpanRows')}` "
            f"answerContractCitationProvenance=`{row.get('answerContractCitationProvenanceRows')}` "
            f"failures=`{row.get('failureReasons')}`"
        )
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in _zero_counter_fields():
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_knowledgeos_v01_rc_corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_knowledgeos_v01_rc_corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_REFRESH_AFTER_POSITIVE_ANSWER_EXECUTION_SCHEMA_ID",
    "READY_DECISION",
    "BLOCKED_DECISION",
    "build_knowledgeos_v01_rc_corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution",
    "write_knowledgeos_v01_rc_corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution",
]
