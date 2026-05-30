"""Positive section/paragraph seed gate for the KnowledgeOS v0.1 RC corpus-scale runner."""

from __future__ import annotations

from collections import Counter
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
from knowledge_hub.papers.knowledgeos_v01_rc_answerability_gate_repair_post_merge_convergence import (
    KNOWLEDGEOS_V01_RC_ANSWERABILITY_GATE_REPAIR_POST_MERGE_CONVERGENCE_SCHEMA_ID,
    READY_DECISION as POST_MERGE_READY_DECISION,
)
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution import (
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_CONTROLLED_EXECUTION_SCHEMA_ID,
    READY_DECISION as CONTROLLED_EXECUTION_READY_DECISION,
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


KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_SECTION_PARAGRAPH_SEED_SCHEMA_ID = (
    "knowledge-hub.product.knowledgeos-v01-rc-corpus-scale-answer-quality-positive-section-paragraph-seed.v1"
)

READY_DECISION = "knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed_ready"
BLOCKED_DECISION = "knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed_blocked"
NEXT_TRANCHE_READY = "corpus_scale_answer_quality_positive_answer_execution_gate"
NEXT_TRANCHE_BLOCKED = "corpus_scale_answer_quality_positive_section_paragraph_seed_repair"

DEFAULT_POST_MERGE_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_answerability_gate_repair_post_merge_convergence.v1.json"
)
DEFAULT_CONTROLLED_EXECUTION_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution.v1.json"
)
DEFAULT_LIVE_RUNNER_DRY_RUN_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_dry_run.v1.json"
)

PROPOSED_EXPECTED_EVIDENCE_TYPE = "section_paragraph"
PROPOSED_ANSWERABILITY_EXPECTATION = "answerable"
MIN_POSITIVE_SEED_ROWS = 4

EXTRA_ZERO_COUNTER_FIELDS = (
    "githubPrMutationRows",
    "mergeRows",
    "branchDeletionRows",
    "releaseTagRows",
    "packagePublishRows",
    "rawPayloadPersistedRows",
    "defaultMcpToolRows",
    "defaultKhubAskRouteRows",
)

ELIGIBLE_PROBE_CATEGORIES = {"method_comparison_qa", "limitation_qa"}
ELIGIBLE_ORIGINAL_EVIDENCE_TYPES = {"cross_paper_method_spans", "limitation_section_span"}
STRUCTURED_MODALITY_CATEGORIES = {
    "table_numeric_qa",
    "equation_citation_qa",
    "figure_caption_qa",
    "appendix_table_lookup_qa",
}

SUPPORT_TERM_GROUPS_BY_CASE_ID: dict[str, tuple[str, ...]] = {
    "complex-paper-qa-seed-20260520-q027": ("retrieval", "generation", "RAG", "FiD"),
    "complex-paper-qa-seed-20260520-q028": ("GraphRAG", "LightRAG", "graph", "retrieval"),
    "complex-paper-qa-seed-20260520-q029": ("Transformer", "Mamba", "sequence", "selective"),
    "complex-paper-qa-seed-20260520-q030": ("AlexNet", "ViT", "image", "architecture"),
    "complex-paper-qa-seed-20260520-q031": ("DQN", "PPO", "objective", "update"),
    "complex-paper-qa-seed-20260520-q032": ("GAN", "DDPM", "generative", "training"),
    "complex-paper-qa-seed-20260520-q033": ("BERT", "GPT-3", "pretraining", "prompting"),
    "complex-paper-qa-seed-20260520-q034": ("RAG", "Self-RAG", "retrieval", "critique"),
    "complex-paper-qa-seed-20260520-q035": (
        "limitation|caveat|risk|boundary|future|however",
        "model|experiment|training|compute",
    ),
    "complex-paper-qa-seed-20260520-q036": (
        "limitation|caveat|risk|boundary|however",
        "selective|state|space|Mamba",
    ),
    "complex-paper-qa-seed-20260520-q037": (
        "limitation|caveat|risk|boundary|data|scale|compute",
        "ViT|image|Transformer",
    ),
    "complex-paper-qa-seed-20260520-q038": (
        "limitation|risk|bias|misuse|harm|caveat",
        "few-shot|language|GPT-3",
    ),
    "complex-paper-qa-seed-20260520-q039": (
        "limitation|failure|risk|caveat|retrieval",
        "Self-RAG|reflection|critique",
    ),
}


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


def _post_merge_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers = _schema_blockers(
        report,
        KNOWLEDGEOS_V01_RC_ANSWERABILITY_GATE_REPAIR_POST_MERGE_CONVERGENCE_SCHEMA_ID,
        "answerability_gate_post_merge",
    )
    if report.get("status") != "ready":
        blockers.append("answerability_gate_post_merge_not_ready")
    if report.get("decision") != POST_MERGE_READY_DECISION:
        blockers.append("answerability_gate_post_merge_decision_not_ready")
    if report.get("nextRecommendedTranche") != "corpus_scale_answer_quality_positive_section_paragraph_seed":
        blockers.append("answerability_gate_post_merge_next_tranche_not_positive_seed")
    if _int(counts.get("answerabilityGateRepairCompleteRows")) != 1:
        blockers.append("answerability_gate_repair_not_complete")
    if _int(counts.get("positiveSectionParagraphSeedRecommendedRows")) != 1:
        blockers.append("positive_section_paragraph_seed_not_recommended")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("answerability_gate_post_merge_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("answerability_gate_post_merge_schema_violations")
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
    if report.get("nextRecommendedTranche") != "corpus_scale_answer_quality_positive_section_paragraph_seed":
        blockers.append("controlled_execution_next_tranche_not_positive_seed")
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


def _dry_run_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers = _schema_blockers(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_LIVE_RUNNER_DRY_RUN_SCHEMA_ID,
        "live_runner_dry_run",
    )
    if report.get("status") != "ready":
        blockers.append("live_runner_dry_run_not_ready")
    if report.get("decision") != DRY_RUN_READY_DECISION:
        blockers.append("live_runner_dry_run_decision_not_ready")
    if _int(counts.get("dryRunCaseRows")) != 50:
        blockers.append("live_runner_dry_run_case_rows_not_50")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("live_runner_dry_run_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("live_runner_dry_run_schema_violations")
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


def _eligible_for_positive_probe(case: dict[str, Any]) -> bool:
    return (
        _clean_text(case.get("answerabilityExpectation")) == "blocked_until_structured_evidence"
        and _clean_text(case.get("questionCategory")) in ELIGIBLE_PROBE_CATEGORIES
        and _clean_text(case.get("expectedEvidenceType")) in ELIGIBLE_ORIGINAL_EVIDENCE_TYPES
        and _clean_text(case.get("caseId")) in SUPPORT_TERM_GROUPS_BY_CASE_ID
    )


def _support_group_matches(text: str, group: str) -> bool:
    haystack = str(text or "").casefold()
    return any(token.casefold() in haystack for token in str(group or "").split("|") if token.strip())


def _observed_source_ids(payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for source in list(payload.get("sources") or []):
        item = dict(source or {})
        source_id = _clean_text(item.get("source_id") or item.get("sourceId") or item.get("paper_id"))
        if source_id and source_id not in out:
            out.append(source_id)
    return sorted(out)


def _evidence_text(payload: dict[str, Any]) -> str:
    return " ".join(
        _clean_text(dict(source or {}).get("text") or dict(source or {}).get("excerpt"))
        for source in list(payload.get("sources") or [])
    )


def _quality_grade(score: float) -> str:
    if score >= 1.0:
        return "pass"
    if score >= 0.6:
        return "partial"
    return "fail"


def _probe_row(
    *,
    case: dict[str, Any],
    seed_question: dict[str, Any],
    papers_dir: str | Path,
) -> dict[str, Any]:
    paper_ids = [_clean_text(item) for item in list(case.get("paperIds") or []) if _clean_text(item)]
    searcher, llm = _build_searcher(papers_dir=papers_dir)
    payload = build_paper_evidence_chunk_answer_preview(
        searcher,
        question=_clean_text(seed_question.get("question")),
        paper_ids=paper_ids,
        question_category=_clean_text(case.get("questionCategory")),
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
    support_groups = list(SUPPORT_TERM_GROUPS_BY_CASE_ID.get(_clean_text(case.get("caseId")), ()))
    matched_groups = [group for group in support_groups if _support_group_matches(evidence_text, group)]
    missing_groups = [group for group in support_groups if group not in matched_groups]
    citation_count = _int(summary.get("citationCount"))
    span_rows = _int(contract.get("spanRows"))
    expected_min = max(2, len(paper_ids) * 2)
    dimensions = {
        "schemaValid": bool(validation.ok),
        "answerabilityExpectation": bool(payload.get("answerable")),
        "sourceCoverage": not missing_ids,
        "citationProvenance": citation_count >= expected_min and span_rows >= expected_min,
        "supportTermCoverage": not missing_groups,
        "publicDefaultHeld": True,
    }
    passed = sum(1 for ok in dimensions.values() if ok)
    score = round(passed / len(dimensions), 6) if dimensions else 0.0
    blockers: list[str] = []
    if not validation.ok:
        blockers.append("preview_payload_schema_invalid")
    if not bool(payload.get("answerable")):
        blockers.append("preview_not_answerable")
    if missing_ids:
        blockers.append("source_coverage_gap")
    if citation_count < expected_min or span_rows < expected_min:
        blockers.append("citation_or_span_count_below_minimum")
    if missing_groups:
        blockers.append("support_term_gap")
    if int(llm.calls) != 1:
        blockers.append("local_fake_llm_call_count_unexpected")
    if _contains_private_path(
        {
            "caseId": case.get("caseId"),
            "paperIds": paper_ids,
            "observedSourceIds": observed_ids,
            "supportTermGroups": support_groups,
        }
    ):
        blockers.append("private_path_leak")
    return {
        "caseIndex": _int(case.get("caseIndex")),
        "caseId": _clean_text(case.get("caseId")),
        "questionCategory": _clean_text(case.get("questionCategory")),
        "originalExpectedEvidenceType": _clean_text(case.get("expectedEvidenceType")),
        "originalAnswerabilityExpectation": _clean_text(case.get("answerabilityExpectation")),
        "proposedExpectedEvidenceType": PROPOSED_EXPECTED_EVIDENCE_TYPE,
        "proposedAnswerabilityExpectation": PROPOSED_ANSWERABILITY_EXPECTATION,
        "paperIds": paper_ids,
        "questionSha256": _clean_text(case.get("questionSha256")),
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
        "selectedPositiveSeed": not blockers,
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "failureReasons": sorted(set(blockers)),
    }


def _missing_seed_question_row(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "caseIndex": _int(case.get("caseIndex")),
        "caseId": _clean_text(case.get("caseId")),
        "questionCategory": _clean_text(case.get("questionCategory")),
        "originalExpectedEvidenceType": _clean_text(case.get("expectedEvidenceType")),
        "originalAnswerabilityExpectation": _clean_text(case.get("answerabilityExpectation")),
        "proposedExpectedEvidenceType": PROPOSED_EXPECTED_EVIDENCE_TYPE,
        "proposedAnswerabilityExpectation": PROPOSED_ANSWERABILITY_EXPECTATION,
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
        "observedSourceIds": [],
        "missingSourceIds": list(case.get("paperIds") or []),
        "supportTermGroups": list(SUPPORT_TERM_GROUPS_BY_CASE_ID.get(_clean_text(case.get("caseId")), ())),
        "matchedSupportTermGroups": [],
        "missingSupportTermGroups": list(SUPPORT_TERM_GROUPS_BY_CASE_ID.get(_clean_text(case.get("caseId")), ())),
        "dimensionStatuses": {
            "schemaValid": False,
            "answerabilityExpectation": False,
            "sourceCoverage": False,
            "citationProvenance": False,
            "supportTermCoverage": False,
            "publicDefaultHeld": True,
        },
        "qualityScore": 0.0,
        "qualityGrade": "fail",
        "selectedPositiveSeed": False,
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "failureReasons": ["seed_question_missing_for_case"],
    }


def build_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed(
    *,
    post_merge_report_path: str | Path = DEFAULT_POST_MERGE_REPORT,
    controlled_execution_report_path: str | Path = DEFAULT_CONTROLLED_EXECUTION_REPORT,
    live_runner_dry_run_report_path: str | Path = DEFAULT_LIVE_RUNNER_DRY_RUN_REPORT,
    corpus_manifest: str | Path = DEFAULT_CORPUS_MANIFEST,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    post_merge_report: dict[str, Any] | None = None,
    controlled_execution_report: dict[str, Any] | None = None,
    live_runner_dry_run_report: dict[str, Any] | None = None,
    seed_pack_report: dict[str, Any] | None = None,
    execute_probe: Callable[..., dict[str, Any]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    post_merge = dict(post_merge_report or _read_json(post_merge_report_path))
    controlled = dict(controlled_execution_report or _read_json(controlled_execution_report_path))
    dry_run = dict(live_runner_dry_run_report or _read_json(live_runner_dry_run_report_path))
    seed_report = dict(seed_pack_report or build_complex_qa_seed_pack(corpus_manifest=corpus_manifest, target_paper_count=20))
    seed_by_id = _seed_questions_by_id(seed_report)
    cases = [dict(row or {}) for row in list(dry_run.get("caseRows") or [])]

    post_merge_blockers = _post_merge_blockers(post_merge)
    controlled_blockers = _controlled_execution_blockers(controlled)
    dry_run_blockers = _dry_run_blockers(dry_run)
    unsafe_blockers = sorted(
        set(
            _unsafe_counter_blockers(post_merge, "post_merge")
            + _unsafe_counter_blockers(controlled, "controlled_execution")
            + _unsafe_counter_blockers(dry_run, "live_runner_dry_run")
        )
    )
    input_blockers = sorted(set(post_merge_blockers + controlled_blockers + dry_run_blockers + unsafe_blockers))

    eligible_cases = [case for case in cases if _eligible_for_positive_probe(case)]
    executor = execute_probe or _probe_row
    rows: list[dict[str, Any]] = []
    if not input_blockers:
        for case in eligible_cases:
            seed_question = seed_by_id.get(_clean_text(case.get("caseId")))
            if not seed_question:
                rows.append(_missing_seed_question_row(case))
            else:
                rows.append(executor(case=case, seed_question=seed_question, papers_dir=papers_dir))

    selected_rows = [row for row in rows if bool(row.get("selectedPositiveSeed"))]
    held_probe_rows = [row for row in rows if not bool(row.get("selectedPositiveSeed"))]
    scores = [float(row.get("qualityScore") or 0.0) for row in rows]
    selected_scores = [float(row.get("qualityScore") or 0.0) for row in selected_rows]
    category_counts = Counter(row.get("questionCategory") for row in rows)
    selected_category_counts = Counter(row.get("questionCategory") for row in selected_rows)
    expected_no_answer_rows = sum(1 for row in cases if _clean_text(row.get("answerabilityExpectation")) == "expected_no_answer")
    structured_modality_rows = sum(1 for row in cases if _clean_text(row.get("questionCategory")) in STRUCTURED_MODALITY_CATEGORIES)
    controlled_counts = dict(controlled.get("counts") or {})

    private_path_leak_rows = sum(1 for row in rows if _contains_private_path(row))
    if _contains_private_path(post_merge) or _contains_private_path(controlled) or _contains_private_path(dry_run):
        private_path_leak_rows = max(private_path_leak_rows, 1)

    semantic_violations = sorted(set(input_blockers))
    if len(cases) != 50:
        semantic_violations.append("input_case_rows_not_50")
    if len(eligible_cases) != 13:
        semantic_violations.append("eligible_probe_rows_not_13")
    if len(selected_rows) < MIN_POSITIVE_SEED_ROWS:
        semantic_violations.append("positive_seed_rows_below_minimum")
    if _int(controlled_counts.get("unexpectedAnswerableRows")) != 0:
        semantic_violations.append("no_answer_safety_unexpected_answerable_rows_present")
    if _int(controlled_counts.get("noAnswerSafetyFailRows")) != 0:
        semantic_violations.append("no_answer_safety_fail_rows_present")
    if private_path_leak_rows:
        semantic_violations.append("positive_section_paragraph_seed_private_path_marker")
    semantic_violations = sorted(set(semantic_violations))
    status = "ready" if not semantic_violations else "blocked"

    counts = {
        "positiveSectionParagraphSeedRows": 1,
        "inputCaseRows": len(cases),
        "controlledExecutionReadyRows": 1 if controlled.get("status") == "ready" else 0,
        "controlledExecutionPassRows": _int(controlled_counts.get("executionPassRows")),
        "controlledExecutionFailRows": _int(controlled_counts.get("executionFailRows")),
        "controlledExecutionUnexpectedAnswerableRows": _int(controlled_counts.get("unexpectedAnswerableRows")),
        "controlledExecutionNoAnswerSafetyFailRows": _int(controlled_counts.get("noAnswerSafetyFailRows")),
        "eligibleSectionParagraphProbeRows": len(eligible_cases),
        "positiveProbeAttemptedRows": len(rows),
        "positiveProbePassRows": len(selected_rows),
        "positiveProbeHeldRows": len(held_probe_rows),
        "positiveSeedRows": len(selected_rows),
        "positiveMethodComparisonRows": selected_category_counts.get("method_comparison_qa", 0),
        "positiveLimitationRows": selected_category_counts.get("limitation_qa", 0),
        "eligibleMethodComparisonRows": category_counts.get("method_comparison_qa", 0),
        "eligibleLimitationRows": category_counts.get("limitation_qa", 0),
        "heldExpectedNoAnswerRows": expected_no_answer_rows,
        "heldStructuredModalityRows": structured_modality_rows,
        "heldRows": len(cases) - len(selected_rows),
        "sourceCoveragePassRows": sum(1 for row in rows if bool(dict(row.get("dimensionStatuses") or {}).get("sourceCoverage"))),
        "supportTermCoveragePassRows": sum(
            1 for row in rows if bool(dict(row.get("dimensionStatuses") or {}).get("supportTermCoverage"))
        ),
        "citationProvenancePassRows": sum(
            1 for row in rows if bool(dict(row.get("dimensionStatuses") or {}).get("citationProvenance"))
        ),
        "answerabilityPassRows": sum(
            1 for row in rows if bool(dict(row.get("dimensionStatuses") or {}).get("answerabilityExpectation"))
        ),
        "averageProbeQualityScore": round(mean(scores), 6) if scores else 0.0,
        "minProbeQualityScore": round(min(scores), 6) if scores else 0.0,
        "averageSelectedQualityScore": round(mean(selected_scores), 6) if selected_scores else 0.0,
        "minSelectedQualityScore": round(min(selected_scores), 6) if selected_scores else 0.0,
        "adapterRowsAdded": sum(_int(row.get("adapterRowsAdded")) for row in rows),
        "selectedEvidenceCount": sum(_int(row.get("selectedEvidenceCount")) for row in selected_rows),
        "citationCount": sum(_int(row.get("citationCount")) for row in selected_rows),
        "evidencePacketContractSpanRows": sum(_int(row.get("evidencePacketContractSpanRows")) for row in selected_rows),
        "localFakeLlmCallRows": sum(_int(row.get("localFakeLlmCallRows")) for row in rows),
        "positiveSeedLocalFakeLlmCallRows": sum(_int(row.get("localFakeLlmCallRows")) for row in selected_rows),
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
        "schema": KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_SECTION_PARAGRAPH_SEED_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "postMergeReportRef": DEFAULT_POST_MERGE_REPORT.as_posix(),
            "controlledExecutionReportRef": DEFAULT_CONTROLLED_EXECUTION_REPORT.as_posix(),
            "liveRunnerDryRunReportRef": DEFAULT_LIVE_RUNNER_DRY_RUN_REPORT.as_posix(),
            "corpusManifestRef": "eval/knowledgeos/fixtures/corpus_manifest.json",
            "papersDirRef": "papers_dir",
        },
        "policy": {
            "labsInternalOnly": True,
            "reportOnly": True,
            "localFakeLlmOnly": True,
            "externalModelCallsAllowed": False,
            "judgeModelCallsAllowed": False,
            "answerTextExcludedFromReport": True,
            "citationPayloadExcludedFromReport": True,
            "sourcePayloadExcludedFromReport": True,
            "excerptExcludedFromReport": True,
            "publicDefaultPromotionAllowed": False,
            "tableEquationFigurePromotionAllowed": False,
        },
        "selectionPolicy": {
            "minimumPositiveSeedRows": MIN_POSITIVE_SEED_ROWS,
            "eligibleQuestionCategories": sorted(ELIGIBLE_PROBE_CATEGORIES),
            "eligibleOriginalEvidenceTypes": sorted(ELIGIBLE_ORIGINAL_EVIDENCE_TYPES),
            "proposedExpectedEvidenceType": PROPOSED_EXPECTED_EVIDENCE_TYPE,
            "proposedAnswerabilityExpectation": PROPOSED_ANSWERABILITY_EXPECTATION,
            "requiresSourceCoverageForAllPaperIds": True,
            "requiresCitationAndSpanRowsPerPaper": 2,
            "requiresSupportTermCoverage": True,
        },
        "counts": counts,
        "gate": {
            "positiveSectionParagraphSeedReady": status == "ready",
            "postMergeReady": not post_merge_blockers,
            "controlledExecutionReady": not controlled_blockers,
            "noAnswerSafetyStillGreen": _int(controlled_counts.get("unexpectedAnswerableRows")) == 0
            and _int(controlled_counts.get("noAnswerSafetyFailRows")) == 0,
            "positiveSeedRowsPresent": len(selected_rows) >= MIN_POSITIVE_SEED_ROWS,
            "expectedNoAnswerRowsHeld": expected_no_answer_rows == _int(controlled_counts.get("expectedNoAnswerRows")),
            "structuredModalityRowsHeld": structured_modality_rows > 0,
            "publicDefaultPromotionAllowed": False,
            "generalRcReady": False,
            "semanticViolations": semantic_violations,
        },
        "positiveSeedRows": selected_rows,
        "heldProbeRows": held_probe_rows,
        "checkRows": [
            {"checkId": "post_merge_convergence", "status": "pass" if not post_merge_blockers else "fail", "blockers": post_merge_blockers},
            {"checkId": "controlled_execution_no_answer_safety", "status": "pass" if not controlled_blockers else "fail", "blockers": controlled_blockers},
            {"checkId": "live_runner_dry_run", "status": "pass" if not dry_run_blockers else "fail", "blockers": dry_run_blockers},
            {"checkId": "unsafe_counters", "status": "pass" if not unsafe_blockers else "fail", "blockers": unsafe_blockers},
            {
                "checkId": "positive_seed_selection",
                "status": "pass" if len(selected_rows) >= MIN_POSITIVE_SEED_ROWS else "fail",
                "blockers": [] if len(selected_rows) >= MIN_POSITIVE_SEED_ROWS else ["positive_seed_rows_below_minimum"],
            },
        ],
        "warnings": [
            "positive_seed_uses_labs_internal_opt_in_answer_path_with_local_fake_llm_only",
            "raw_question_answer_citation_source_and_excerpt_payloads_are_not_included",
            "expected_no_answer_and_structured_modality_rows_remain_held",
            "public_default_promotion_remains_held",
        ],
    }


def render_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# KnowledgeOS v0.1 RC Corpus-Scale Positive Section/Paragraph Seed",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- inputCaseRows: `{counts.get('inputCaseRows')}`",
        f"- eligibleSectionParagraphProbeRows: `{counts.get('eligibleSectionParagraphProbeRows')}`",
        f"- positiveSeedRows: `{counts.get('positiveSeedRows')}`",
        f"- positiveMethodComparisonRows: `{counts.get('positiveMethodComparisonRows')}`",
        f"- positiveLimitationRows: `{counts.get('positiveLimitationRows')}`",
        f"- positiveProbeHeldRows: `{counts.get('positiveProbeHeldRows')}`",
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
    lines.extend(["", "## Positive Seed Rows", ""])
    for row in list(report.get("positiveSeedRows") or []):
        lines.append(
            f"- `{row.get('caseId')}` category=`{row.get('questionCategory')}` papers=`{row.get('paperIds')}` "
            f"citations=`{row.get('citationCount')}` spans=`{row.get('evidencePacketContractSpanRows')}` "
            f"score=`{row.get('qualityScore')}`"
        )
    lines.extend(["", "## Held Probe Rows", ""])
    for row in list(report.get("heldProbeRows") or []):
        lines.append(
            f"- `{row.get('caseId')}` category=`{row.get('questionCategory')}` "
            f"qualityGrade=`{row.get('qualityGrade')}` failures=`{row.get('failureReasons')}`"
        )
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in _zero_counter_fields():
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_SECTION_PARAGRAPH_SEED_SCHEMA_ID",
    "READY_DECISION",
    "BLOCKED_DECISION",
    "build_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed",
    "write_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed",
]
