"""Quality eval runner for the labs parsed-artifact evidence chunk surface."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any

from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_SEED_SCHEMA_ID,
    READY_DECISION as QUALITY_EVAL_SEED_READY_DECISION,
    ZERO_COUNTER_FIELDS,
    _case_row as _run_seed_case,
    _clean_text,
    _contains_private_path,
    _int,
    _read_json,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke import (
    DEFAULT_PAPERS_DIR,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_RUNNER_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-labs-opt-in-quality-eval-runner.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner_ready"
BLOCKED_DECISION = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner_blocked"
NEXT_TRANCHE_READY = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet"
NEXT_TRANCHE_BLOCKED = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner_repair"
DEFAULT_QUALITY_EVAL_SEED_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed.v1.json"
)
PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _seed_report_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_SEED_SCHEMA_ID:
        blockers.append("quality_eval_seed_schema_mismatch")
    if report.get("status") != "ready":
        blockers.append("quality_eval_seed_not_ready")
    if report.get("decision") != QUALITY_EVAL_SEED_READY_DECISION:
        blockers.append("quality_eval_seed_decision_not_ready")
    if gate.get("readyForLabsOptInQualityEvalRunner") is not True:
        blockers.append("quality_eval_seed_gate_not_ready_for_runner")
    if _int(counts.get("inputCaseRows")) <= 0:
        blockers.append("quality_eval_seed_has_no_cases")
    if _int(counts.get("passRows")) != _int(counts.get("inputCaseRows")):
        blockers.append("quality_eval_seed_cases_not_all_passed")
    if _int(counts.get("noEvidenceLlmCallRows")) != 0:
        blockers.append("quality_eval_seed_no_evidence_called_llm")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("quality_eval_seed_schema_violations_present")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("quality_eval_seed_private_path_leak")
    if _contains_private_path(report):
        blockers.append("quality_eval_seed_private_path_marker")
    return sorted(set(blockers))


def _case_from_seed_row(row: dict[str, Any]) -> dict[str, Any]:
    expected_source_ids = [_clean_text(item) for item in list(row.get("expectedSourceIds") or []) if _clean_text(item)]
    expected_answerable = bool(row.get("expectedAnswerable"))
    minimum = 0
    if expected_answerable:
        minimum = max(2, len(expected_source_ids) * 2)
    return {
        "caseId": _clean_text(row.get("caseId")),
        "evalFocus": _clean_text(row.get("evalFocus")),
        "question": _clean_text(row.get("question")),
        "paperIds": [_clean_text(item) for item in list(row.get("paperIds") or []) if _clean_text(item)],
        "expectedStatus": _clean_text(row.get("expectedStatus")),
        "expectedAnswerable": expected_answerable,
        "expectedSourceIds": expected_source_ids,
        "requiredEvidenceTerms": [
            _clean_text(item) for item in list(row.get("requiredEvidenceTerms") or []) if _clean_text(item)
        ],
        "minCitations": minimum,
        "minSpanRows": minimum,
    }


def _dimension_statuses(row: dict[str, Any]) -> dict[str, Any]:
    expected_answerable = bool(row.get("expectedAnswerable"))
    answerability_ok = bool(row.get("observedAnswerable")) is expected_answerable
    schema_ok = bool(row.get("schemaValid"))
    source_ok = not list(row.get("missingSourceIds") or [])
    evidence_terms_ok = not list(row.get("missingEvidenceTerms") or [])
    citation_span_ok = True
    no_evidence_ok = True
    if expected_answerable:
        citation_span_ok = _int(row.get("citationCount")) > 0 and _int(row.get("evidencePacketContractSpanRows")) > 0
    else:
        no_evidence_ok = (
            row.get("observedStatus") == "no_evidence"
            and not bool(row.get("observedAnswerable"))
            and _int(row.get("citationCount")) == 0
            and _int(row.get("evidencePacketContractSpanRows")) == 0
            and _int(row.get("localFakeLlmCallRows")) == 0
        )
    applicable = [
        ("schema_valid", schema_ok),
        ("answerability_expectation", answerability_ok),
        ("source_coverage", source_ok),
        ("evidence_term_support", evidence_terms_ok),
    ]
    if expected_answerable:
        applicable.append(("citation_and_span_presence", citation_span_ok))
    else:
        applicable.append(("expected_no_evidence_safety", no_evidence_ok))
    passed = sum(1 for _, ok in applicable if ok)
    score = round(passed / len(applicable), 6) if applicable else 0.0
    return {
        "schemaValid": schema_ok,
        "answerabilityExpectation": answerability_ok,
        "sourceCoverage": source_ok,
        "evidenceTermSupport": evidence_terms_ok,
        "citationAndSpanPresence": citation_span_ok,
        "expectedNoEvidenceSafety": no_evidence_ok,
        "qualityScore": score,
        "qualityGrade": "pass" if score == 1.0 else ("partial" if score >= 0.6 else "fail"),
    }


def _runner_row(*, seed_row: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
    rerun = _run_seed_case(case=_case_from_seed_row(seed_row), papers_dir=papers_dir)
    dimensions = _dimension_statuses(rerun)
    failure_reasons = list(rerun.get("failureReasons") or [])
    if dimensions["qualityGrade"] != "pass":
        failure_reasons.append("quality_dimensions_not_all_passed")
    if _contains_private_path(rerun):
        failure_reasons.append("private_path_leak")
    return {
        "caseId": _clean_text(rerun.get("caseId")),
        "evalFocus": _clean_text(rerun.get("evalFocus")),
        "expectedStatus": _clean_text(rerun.get("expectedStatus")),
        "observedStatus": _clean_text(rerun.get("observedStatus")),
        "expectedAnswerable": bool(rerun.get("expectedAnswerable")),
        "observedAnswerable": bool(rerun.get("observedAnswerable")),
        "paperIds": list(rerun.get("paperIds") or []),
        "observedSourceIds": list(rerun.get("observedSourceIds") or []),
        "missingSourceIds": list(rerun.get("missingSourceIds") or []),
        "requiredEvidenceTerms": list(rerun.get("requiredEvidenceTerms") or []),
        "observedEvidenceTerms": list(rerun.get("observedEvidenceTerms") or []),
        "missingEvidenceTerms": list(rerun.get("missingEvidenceTerms") or []),
        "adapterStatus": _clean_text(rerun.get("adapterStatus")),
        "adapterRowsAdded": _int(rerun.get("adapterRowsAdded")),
        "selectedEvidenceCount": _int(rerun.get("selectedEvidenceCount")),
        "citationCount": _int(rerun.get("citationCount")),
        "evidencePacketContractSpanRows": _int(rerun.get("evidencePacketContractSpanRows")),
        "localFakeLlmCallRows": _int(rerun.get("localFakeLlmCallRows")),
        "dimensionStatuses": {
            "schemaValid": bool(dimensions["schemaValid"]),
            "answerabilityExpectation": bool(dimensions["answerabilityExpectation"]),
            "sourceCoverage": bool(dimensions["sourceCoverage"]),
            "evidenceTermSupport": bool(dimensions["evidenceTermSupport"]),
            "citationAndSpanPresence": bool(dimensions["citationAndSpanPresence"]),
            "expectedNoEvidenceSafety": bool(dimensions["expectedNoEvidenceSafety"]),
        },
        "qualityScore": float(dimensions["qualityScore"]),
        "qualityGrade": _clean_text(dimensions["qualityGrade"]),
        "answerTextHash": _clean_text(rerun.get("answerTextHash")),
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "candidateRecordIds": list(rerun.get("candidateRecordIds") or []),
        "failureReasons": sorted(set(failure_reasons)),
        "pass": not failure_reasons,
    }


def build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner(
    *,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    quality_eval_seed_report_path: str | Path = DEFAULT_QUALITY_EVAL_SEED_REPORT,
    quality_eval_seed_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    seed_report = dict(quality_eval_seed_report or _read_json(quality_eval_seed_report_path))
    source_blockers = _seed_report_blockers(seed_report)
    seed_rows = [dict(row or {}) for row in list(seed_report.get("rows") or [])]
    rows = [] if source_blockers else [_runner_row(seed_row=row, papers_dir=papers_dir) for row in seed_rows]
    pass_rows = sum(1 for row in rows if bool(row.get("pass")))
    partial_rows = sum(1 for row in rows if row.get("qualityGrade") == "partial")
    fail_rows = sum(1 for row in rows if row.get("qualityGrade") == "fail" or not bool(row.get("pass")))
    scores = [float(row.get("qualityScore") or 0.0) for row in rows]
    private_path_leak_rows = sum(1 for row in rows if _contains_private_path(row))
    semantic_violations = list(source_blockers)
    if fail_rows:
        semantic_violations.append(f"quality_eval_runner_failures:{fail_rows}")
    if private_path_leak_rows:
        semantic_violations.append("private_path_leak")
    expected_answerable_rows = sum(1 for row in rows if bool(row.get("expectedAnswerable")))
    expected_no_evidence_rows = len(rows) - expected_answerable_rows
    counts = {
        "inputSeedRows": 1 if seed_report else 0,
        "seedReadyInputRows": 1 if seed_report and not source_blockers else 0,
        "inputCaseRows": len(seed_rows),
        "attemptedCaseRows": len(rows),
        "qualityPassRows": pass_rows,
        "qualityPartialRows": partial_rows,
        "qualityFailRows": fail_rows,
        "blockedRows": len(seed_rows) if source_blockers else 0,
        "expectedAnswerableRows": expected_answerable_rows,
        "expectedNoEvidenceRows": expected_no_evidence_rows,
        "observedAnswerableRows": sum(1 for row in rows if bool(row.get("observedAnswerable"))),
        "observedNoEvidenceRows": sum(1 for row in rows if row.get("observedStatus") == "no_evidence"),
        "sourceCoveragePassRows": sum(1 for row in rows if bool(row.get("dimensionStatuses", {}).get("sourceCoverage"))),
        "evidenceTermSupportPassRows": sum(
            1 for row in rows if bool(row.get("dimensionStatuses", {}).get("evidenceTermSupport"))
        ),
        "citationSpanPassRows": sum(
            1 for row in rows if bool(row.get("dimensionStatuses", {}).get("citationAndSpanPresence"))
        ),
        "noEvidenceSafetyPassRows": sum(
            1
            for row in rows
            if not bool(row.get("expectedAnswerable"))
            and bool(row.get("dimensionStatuses", {}).get("expectedNoEvidenceSafety"))
        ),
        "averageQualityScore": round(mean(scores), 6) if scores else 0.0,
        "minQualityScore": round(min(scores), 6) if scores else 0.0,
        "adapterRowsAdded": sum(_int(row.get("adapterRowsAdded")) for row in rows),
        "selectedEvidenceCount": sum(_int(row.get("selectedEvidenceCount")) for row in rows),
        "citationCount": sum(_int(row.get("citationCount")) for row in rows),
        "evidencePacketContractSpanRows": sum(_int(row.get("evidencePacketContractSpanRows")) for row in rows),
        "localFakeLlmCallRows": sum(_int(row.get("localFakeLlmCallRows")) for row in rows),
        "noEvidenceLlmCallRows": sum(
            _int(row.get("localFakeLlmCallRows")) for row in rows if not bool(row.get("expectedAnswerable"))
        ),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(set(semantic_violations)),
    }
    status = "ready" if not semantic_violations and pass_rows == len(seed_rows) else "blocked"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_RUNNER_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "qualityEvalSeedReportRef": (
                "eval/knowledgeos/reports/"
                "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed.v1.json"
            ),
            "qualityEvalSeedSchema": _clean_text(seed_report.get("schema")),
            "qualityEvalSeedStatus": _clean_text(seed_report.get("status")),
            "qualityEvalSeedDecision": _clean_text(seed_report.get("decision")),
            "papersDirRef": "papers_dir",
        },
        "policy": {
            "reportOnly": True,
            "labsOnly": True,
            "localFakeLlmOnly": True,
            "answerTextExcludedFromReport": True,
            "citationPayloadExcludedFromReport": True,
            "sourcePayloadExcludedFromReport": True,
            "excerptExcludedFromReport": True,
            "externalModelCallsAllowed": False,
            "judgeModelCallsAllowed": False,
            "candidateStoreWrites": False,
            "sourceSpanCreation": False,
            "strictEvidenceCreation": False,
            "runtimeDefaultChange": False,
        },
        "qualityDimensions": [
            "schema_valid",
            "answerability_expectation",
            "source_coverage",
            "evidence_term_support",
            "citation_and_span_presence",
            "expected_no_evidence_safety",
        ],
        "counts": counts,
        "gate": {
            "readyForLabsOptInUserTestPacket": status == "ready",
            "qualityEvalSeedReady": not source_blockers,
            "allCasesPassed": pass_rows == len(seed_rows) and not fail_rows and not source_blockers,
            "expectedNoEvidenceCasesStayedNoEvidence": counts["noEvidenceSafetyPassRows"] == expected_no_evidence_rows,
            "noEvidenceCasesAvoidedLlm": counts["noEvidenceLlmCallRows"] == 0,
            "qualityScoreThresholdPassed": counts["minQualityScore"] >= 1.0,
            "semanticViolations": sorted(set(semantic_violations)),
        },
        "rows": rows,
        "warnings": [
            "runner_scores_evidence_and_answerability_contracts_not_human_answer_style",
            "answer_text_citations_sources_and_raw_excerpts_are_not_included_in_this_report",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in Quality Eval Runner",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- inputCaseRows: `{counts.get('inputCaseRows')}`",
        f"- qualityPassRows: `{counts.get('qualityPassRows')}`",
        f"- qualityPartialRows: `{counts.get('qualityPartialRows')}`",
        f"- qualityFailRows: `{counts.get('qualityFailRows')}`",
        f"- averageQualityScore: `{counts.get('averageQualityScore')}`",
        f"- minQualityScore: `{counts.get('minQualityScore')}`",
        f"- expectedNoEvidenceRows: `{counts.get('expectedNoEvidenceRows')}`",
        f"- noEvidenceLlmCallRows: `{counts.get('noEvidenceLlmCallRows')}`",
        f"- citationCount: `{counts.get('citationCount')}`",
        f"- evidencePacketContractSpanRows: `{counts.get('evidencePacketContractSpanRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Policy",
        "",
        "Report-only quality runner for the labs preview surface. It scores evidence support, source coverage, citations/spans, and no-evidence safety; it does not judge final answer prose.",
        "",
        "## Mutation Guarantees",
        "",
    ]
    for field in ZERO_COUNTER_FIELDS:
        lines.append(f"- {field}: `{counts.get(field)}`")
    lines.extend(["", "## Cases", ""])
    for row in list(report.get("rows") or []):
        lines.extend(
            [
                f"### `{row.get('caseId')}`",
                "",
                f"- pass: `{row.get('pass')}`",
                f"- qualityGrade: `{row.get('qualityGrade')}`",
                f"- qualityScore: `{row.get('qualityScore')}`",
                f"- expected/observed status: `{row.get('expectedStatus')}` / `{row.get('observedStatus')}`",
                f"- expected/observed answerable: `{row.get('expectedAnswerable')}` / `{row.get('observedAnswerable')}`",
                f"- observedEvidenceTerms: `{row.get('observedEvidenceTerms')}`",
                f"- citations: `{row.get('citationCount')}`",
                f"- spans: `{row.get('evidencePacketContractSpanRows')}`",
                f"- localFakeLlmCallRows: `{row.get('localFakeLlmCallRows')}`",
                f"- failureReasons: `{row.get('failureReasons')}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_RUNNER_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner",
    "write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner",
]
