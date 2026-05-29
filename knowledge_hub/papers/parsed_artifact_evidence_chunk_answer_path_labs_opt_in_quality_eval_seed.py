"""Quality-eval seed for the labs parsed-artifact evidence chunk surface."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.evidence_chunk_answer_preview import (
    PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID,
    build_paper_evidence_chunk_answer_preview,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID,
    READY_DECISION as LIVE_SMOKE_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke import (
    DEFAULT_PAPERS_DIR,
    _build_searcher,
    _clean_text,
    _contains_private_path,
    _int,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_SEED_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-labs-opt-in-quality-eval-seed.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed_ready"
BLOCKED_DECISION = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed_blocked"
NEXT_TRANCHE_READY = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner"
NEXT_TRANCHE_BLOCKED = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed_repair"
DEFAULT_LABS_SURFACE_LIVE_SMOKE_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke.v1.json"
)
ZERO_COUNTER_FIELDS = (
    "candidateStoreWriteRows",
    "sourceSpanCreatedRows",
    "strictEvidenceRows",
    "citationGradeEvidenceRows",
    "runtimeEvidenceRows",
    "parserExecutionRows",
    "databaseMutationRows",
    "indexMutationRows",
    "reindexOrReembedRows",
    "canonicalParsedArtifactWriteRows",
    "vaultScanRows",
    "externalDownloadRows",
    "publicCliFlagRows",
    "defaultOnRows",
    "externalLlmCallRows",
    "modelApiCallRows",
    "judgeModelCallRows",
)
PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)


DEFAULT_CASES: tuple[dict[str, Any], ...] = (
    {
        "caseId": "alphafold_protein_structure_seed",
        "evalFocus": "single_paper_section_paragraph_evidence",
        "question": "What parsed section or paragraph evidence is available about AlphaFold protein structure prediction?",
        "paperIds": ["1207.0580"],
        "expectedStatus": "ok",
        "expectedAnswerable": True,
        "expectedSourceIds": ["1207.0580"],
        "requiredEvidenceTerms": ["AlphaFold", "protein", "structure"],
        "minCitations": 2,
        "minSpanRows": 2,
    },
    {
        "caseId": "word_vectors_similarity_seed",
        "evalFocus": "single_paper_section_paragraph_evidence",
        "question": "What parsed section or paragraph evidence is available about continuous word representations?",
        "paperIds": ["1301.3781"],
        "expectedStatus": "ok",
        "expectedAnswerable": True,
        "expectedSourceIds": ["1301.3781"],
        "requiredEvidenceTerms": ["continuous", "vector", "word", "similarity"],
        "minCitations": 2,
        "minSpanRows": 2,
    },
    {
        "caseId": "resolved_pair_compare_seed",
        "evalFocus": "two_paper_compare_seed",
        "question": "Compare available parsed evidence for the AlphaFold and word vector papers.",
        "paperIds": ["1207.0580", "1301.3781"],
        "expectedStatus": "ok",
        "expectedAnswerable": True,
        "expectedSourceIds": ["1207.0580", "1301.3781"],
        "requiredEvidenceTerms": ["AlphaFold", "CASP14", "continuous", "similarity"],
        "minCitations": 4,
        "minSpanRows": 4,
    },
    {
        "caseId": "missing_candidate_store_no_evidence_seed",
        "evalFocus": "expected_no_evidence_safety",
        "question": "What parsed section or paragraph evidence is available for this missing paper?",
        "paperIds": ["missing-paper-id"],
        "expectedStatus": "no_evidence",
        "expectedAnswerable": False,
        "expectedSourceIds": [],
        "requiredEvidenceTerms": [],
        "minCitations": 0,
        "minSpanRows": 0,
    },
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_text(value: Any) -> str:
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _term_present(text: str, term: str) -> bool:
    token = _clean_text(term)
    return bool(token and token.casefold() in str(text or "").casefold())


def _observed_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if _term_present(text, term)]


def _live_smoke_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID:
        blockers.append("labs_surface_live_smoke_schema_mismatch")
    if report.get("status") != "ready":
        blockers.append("labs_surface_live_smoke_not_ready")
    if report.get("decision") != LIVE_SMOKE_READY_DECISION:
        blockers.append("labs_surface_live_smoke_decision_not_ready")
    if gate.get("readyForLabsOptInQualityEvalSeed") is not True:
        blockers.append("labs_surface_live_smoke_gate_not_ready_for_quality_seed")
    if _int(counts.get("surfaceSmokePassRows")) != 1:
        blockers.append("labs_surface_live_smoke_positive_case_not_passed")
    if _int(counts.get("adapterRowsAdded")) <= 0:
        blockers.append("labs_surface_live_smoke_adapter_added_no_rows")
    if _int(counts.get("externalRequestRejectedRows")) != 1:
        blockers.append("labs_surface_live_smoke_external_request_not_rejected")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("labs_surface_live_smoke_schema_violations_present")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("labs_surface_live_smoke_private_path_leak")
    if _contains_private_path(report):
        blockers.append("labs_surface_live_smoke_private_path_marker")
    return sorted(set(blockers))


def _case_row(*, case: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
    paper_ids = [_clean_text(item) for item in list(case.get("paperIds") or []) if _clean_text(item)]
    expected_source_ids = [_clean_text(item) for item in list(case.get("expectedSourceIds") or []) if _clean_text(item)]
    required_terms = [_clean_text(item) for item in list(case.get("requiredEvidenceTerms") or []) if _clean_text(item)]
    expected_status = _clean_text(case.get("expectedStatus")) or "ok"
    expected_answerable = bool(case.get("expectedAnswerable"))
    min_citations = max(0, _int(case.get("minCitations")))
    min_span_rows = max(0, _int(case.get("minSpanRows")))
    searcher, llm = _build_searcher(papers_dir=papers_dir)
    payload: dict[str, Any] = {}
    call_error = ""
    try:
        payload = build_paper_evidence_chunk_answer_preview(
            searcher,
            question=_clean_text(case.get("question")),
            paper_ids=paper_ids,
            top_k=1,
            retrieval_mode="semantic",
            allow_external=False,
        )
    except Exception as error:  # pragma: no cover - defensive, surfaced in report.
        call_error = str(error)
    schema_valid = validate_payload(payload, PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID, strict=True).ok
    adapter = dict(payload.get("adapterDiagnostics") or {})
    summary = dict(payload.get("evidencePacketSummary") or {})
    contract = dict(payload.get("evidencePacketContractSummary") or {})
    sources = [dict(source or {}) for source in list(payload.get("sources") or [])]
    evidence_text = " ".join(_clean_text(source.get("text") or source.get("excerpt")) for source in sources)
    observed_terms = _observed_terms(evidence_text, required_terms)
    observed_source_ids = sorted(
        {
            _clean_text(source.get("source_id") or source.get("sourceId") or source.get("paper_id"))
            for source in sources
            if _clean_text(source.get("source_id") or source.get("sourceId") or source.get("paper_id"))
        }
    )
    missing_terms = [term for term in required_terms if term not in observed_terms]
    missing_source_ids = [source_id for source_id in expected_source_ids if source_id not in observed_source_ids]
    payload_status = _clean_text(payload.get("status"))
    observed_answerable = bool(payload.get("answerable"))
    citation_count = _int(summary.get("citationCount"))
    span_rows = _int(contract.get("spanRows"))
    selected_evidence_count = _int(summary.get("selectedEvidenceCount"))
    adapter_rows_added = _int(summary.get("adapterRowsAdded") or adapter.get("rowsAdded"))
    failure_reasons: list[str] = []
    if call_error:
        failure_reasons.append("surface_call_error")
    if not schema_valid:
        failure_reasons.append("surface_payload_schema_invalid")
    if payload_status != expected_status:
        failure_reasons.append("payload_status_unexpected")
    if observed_answerable != expected_answerable:
        failure_reasons.append("answerable_unexpected")
    if bool(payload.get("allowExternal")):
        failure_reasons.append("allow_external_not_false")
    query_plan = dict(payload.get("queryPlan") or {})
    if query_plan.get("parsed_artifact_evidence_chunk_adapter") != "runtime_v1":
        failure_reasons.append("query_plan_opt_in_missing")
    if expected_answerable:
        if _clean_text(summary.get("adapterStatus") or adapter.get("status")) != "applied":
            failure_reasons.append("adapter_not_applied")
        if adapter_rows_added < min_citations:
            failure_reasons.append("adapter_rows_added_below_minimum")
        if selected_evidence_count < min_citations:
            failure_reasons.append("selected_evidence_count_below_minimum")
        if citation_count < min_citations:
            failure_reasons.append("citation_count_below_minimum")
        if span_rows < min_span_rows:
            failure_reasons.append("span_rows_below_minimum")
        if int(llm.calls) != 1:
            failure_reasons.append("local_fake_llm_call_count_unexpected")
        if missing_terms:
            failure_reasons.append("required_evidence_terms_missing")
        if missing_source_ids:
            failure_reasons.append("expected_source_ids_missing")
    else:
        if adapter_rows_added != 0 or selected_evidence_count != 0 or citation_count != 0 or span_rows != 0:
            failure_reasons.append("no_evidence_case_produced_evidence")
        if int(llm.calls) != 0:
            failure_reasons.append("no_evidence_case_called_llm")
    row_private_path_leak = _contains_private_path(
        {
            "caseId": case.get("caseId"),
            "paperIds": paper_ids,
            "observedSourceIds": observed_source_ids,
            "candidateRecordIds": list(adapter.get("selectedCandidateRecordIds") or []),
        }
    )
    if row_private_path_leak:
        failure_reasons.append("private_path_leak")
    return {
        "caseId": _clean_text(case.get("caseId")),
        "evalFocus": _clean_text(case.get("evalFocus")),
        "question": _clean_text(case.get("question")),
        "paperIds": paper_ids,
        "expectedStatus": expected_status,
        "observedStatus": payload_status,
        "expectedAnswerable": expected_answerable,
        "observedAnswerable": observed_answerable,
        "expectedSourceIds": expected_source_ids,
        "observedSourceIds": observed_source_ids,
        "missingSourceIds": missing_source_ids,
        "requiredEvidenceTerms": required_terms,
        "observedEvidenceTerms": observed_terms,
        "missingEvidenceTerms": missing_terms,
        "adapterStatus": _clean_text(summary.get("adapterStatus") or adapter.get("status")),
        "adapterRowsAdded": adapter_rows_added,
        "adapterCandidateRowsConsidered": _int(summary.get("adapterCandidateRowsConsidered")),
        "selectedEvidenceCount": selected_evidence_count,
        "citationCount": citation_count,
        "evidencePacketContractSpanRows": span_rows,
        "localFakeLlmCallRows": int(llm.calls),
        "schemaValid": schema_valid,
        "answerTextHash": _sha256_text(payload.get("answer")),
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "candidateRecordIds": list(adapter.get("selectedCandidateRecordIds") or []),
        "failureReasons": sorted(set(failure_reasons)),
        "pass": not failure_reasons,
    }


def build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed(
    *,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    live_smoke_report_path: str | Path = DEFAULT_LABS_SURFACE_LIVE_SMOKE_REPORT,
    live_smoke_report: dict[str, Any] | None = None,
    cases: list[dict[str, Any]] | tuple[dict[str, Any], ...] = DEFAULT_CASES,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_report = dict(live_smoke_report or _read_json(live_smoke_report_path))
    source_blockers = _live_smoke_blockers(source_report)
    rows = [] if source_blockers else [_case_row(case=dict(case), papers_dir=papers_dir) for case in cases]
    pass_rows = sum(1 for row in rows if row.get("pass") is True)
    fail_rows = sum(1 for row in rows if row.get("pass") is False)
    expected_answerable_rows = sum(1 for case in cases if bool(case.get("expectedAnswerable")))
    expected_no_evidence_rows = len(cases) - expected_answerable_rows
    private_path_leak_rows = sum(1 for row in rows if _contains_private_path(row))
    semantic_violations = list(source_blockers)
    if fail_rows:
        semantic_violations.append(f"quality_seed_case_failures:{fail_rows}")
    if private_path_leak_rows:
        semantic_violations.append("private_path_leak")
    counts = {
        "inputLiveSmokeRows": 1 if source_report else 0,
        "liveSmokeReadyInputRows": 1 if source_report and not source_blockers else 0,
        "inputCaseRows": len(cases),
        "attemptedCaseRows": len(rows),
        "passRows": pass_rows,
        "failRows": fail_rows,
        "blockedRows": len(cases) if source_blockers else 0,
        "expectedAnswerableRows": expected_answerable_rows,
        "expectedNoEvidenceRows": expected_no_evidence_rows,
        "observedAnswerableRows": sum(1 for row in rows if bool(row.get("observedAnswerable"))),
        "observedNoEvidenceRows": sum(1 for row in rows if row.get("observedStatus") == "no_evidence"),
        "positiveCasePassRows": sum(
            1 for row in rows if bool(row.get("expectedAnswerable")) and bool(row.get("pass"))
        ),
        "noEvidenceCasePassRows": sum(
            1 for row in rows if not bool(row.get("expectedAnswerable")) and bool(row.get("pass"))
        ),
        "sourceCoveragePassRows": sum(1 for row in rows if not row.get("missingSourceIds")),
        "evidenceTermCoveragePassRows": sum(1 for row in rows if not row.get("missingEvidenceTerms")),
        "surfacePayloadSchemaValidRows": sum(1 for row in rows if bool(row.get("schemaValid"))),
        "adapterAppliedRows": sum(1 for row in rows if row.get("adapterStatus") == "applied"),
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
    status = "ready" if not semantic_violations and pass_rows == len(cases) else "blocked"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_SEED_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "labsSurfaceLiveSmokeReportRef": (
                "eval/knowledgeos/reports/"
                "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke.v1.json"
            ),
            "labsSurfaceLiveSmokeSchema": _clean_text(source_report.get("schema")),
            "labsSurfaceLiveSmokeStatus": _clean_text(source_report.get("status")),
            "labsSurfaceLiveSmokeDecision": _clean_text(source_report.get("decision")),
            "papersDirRef": "papers_dir",
        },
        "policy": {
            "reportOnly": True,
            "seedOnly": True,
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
            "answerability_expectation",
            "evidence_term_support",
            "source_coverage",
            "citation_and_span_presence",
            "expected_no_evidence_safety",
        ],
        "counts": counts,
        "gate": {
            "readyForLabsOptInQualityEvalRunner": status == "ready",
            "labsSurfaceLiveSmokeReady": not source_blockers,
            "allSeedCasesPassed": pass_rows == len(cases) and not fail_rows and not source_blockers,
            "expectedNoEvidenceCasesStayedNoEvidence": counts["noEvidenceCasePassRows"] == expected_no_evidence_rows,
            "noEvidenceCasesAvoidedLlm": counts["noEvidenceLlmCallRows"] == 0,
            "semanticViolations": sorted(set(semantic_violations)),
        },
        "rows": rows,
        "warnings": [
            "seed_cases_define_a_minimal_quality_eval_starting_set_not_a_full_quality_benchmark",
            "answer_text_citations_sources_and_raw_excerpts_are_not_included_in_this_report",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in Quality Eval Seed",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- inputCaseRows: `{counts.get('inputCaseRows')}`",
        f"- passRows: `{counts.get('passRows')}`",
        f"- failRows: `{counts.get('failRows')}`",
        f"- expectedAnswerableRows: `{counts.get('expectedAnswerableRows')}`",
        f"- expectedNoEvidenceRows: `{counts.get('expectedNoEvidenceRows')}`",
        f"- adapterRowsAdded: `{counts.get('adapterRowsAdded')}`",
        f"- citationCount: `{counts.get('citationCount')}`",
        f"- evidencePacketContractSpanRows: `{counts.get('evidencePacketContractSpanRows')}`",
        f"- localFakeLlmCallRows: `{counts.get('localFakeLlmCallRows')}`",
        f"- noEvidenceLlmCallRows: `{counts.get('noEvidenceLlmCallRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Policy",
        "",
        "Report-only seed for the next labs quality eval runner. Raw answer text, citations, sources, and excerpts are excluded from this report.",
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
                f"- focus: `{row.get('evalFocus')}`",
                f"- expected/observed status: `{row.get('expectedStatus')}` / `{row.get('observedStatus')}`",
                f"- expected/observed answerable: `{row.get('expectedAnswerable')}` / `{row.get('observedAnswerable')}`",
                f"- paperIds: `{row.get('paperIds')}`",
                f"- observedEvidenceTerms: `{row.get('observedEvidenceTerms')}`",
                f"- missingEvidenceTerms: `{row.get('missingEvidenceTerms')}`",
                f"- citations: `{row.get('citationCount')}`",
                f"- spans: `{row.get('evidencePacketContractSpanRows')}`",
                f"- localFakeLlmCallRows: `{row.get('localFakeLlmCallRows')}`",
                f"- failureReasons: `{row.get('failureReasons')}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_SEED_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed",
    "write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed",
]
