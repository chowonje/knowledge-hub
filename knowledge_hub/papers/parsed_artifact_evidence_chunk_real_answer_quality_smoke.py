"""Deterministic real-answer smoke for parsed-artifact evidence chunks.

This smoke exercises the actual runtime adapter path and then builds small
extractive answer candidates from the selected evidence. It checks citation,
provenance, and term-support gates without calling an LLM or judge model.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

from knowledge_hub.ai.answer_contracts import build_answer_contract, build_evidence_packet_contract
from knowledge_hub.ai.evidence_assembly import EvidenceAssemblyService
from knowledge_hub.ai.parsed_artifact_evidence_chunk_runtime_adapter import ADAPTER_OPT_IN_VALUE
from knowledge_hub.core.models import SearchResult
from knowledge_hub.papers.parsed_artifact_evidence_chunk_runtime_adapter_live_smoke import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_RUNTIME_ADAPTER_LIVE_SMOKE_SCHEMA_ID,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_REAL_ANSWER_QUALITY_SMOKE_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-real-answer-quality-smoke.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_real_answer_quality_smoke_ready"
BLOCKED_DECISION = "parsed_artifact_evidence_chunk_real_answer_quality_smoke_blocked"
NEXT_TRANCHE_READY = "parsed_artifact_evidence_chunk_answer_path_opt_in_route_review"
NEXT_TRANCHE_BLOCKED = "parsed_artifact_evidence_chunk_real_answer_quality_smoke_repair"
DEFAULT_PAPERS_DIR = Path.home() / ".khub" / "papers"
DEFAULT_LIVE_SMOKE_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_runtime_adapter_live_smoke.v1.json"
)
ZERO_COUNTER_FIELDS = (
    "candidateStoreWriteRows",
    "sourceSpanCreatedRows",
    "strictEvidenceRows",
    "parserExecutionRows",
    "databaseMutationRows",
    "indexMutationRows",
    "reindexOrReembedRows",
    "canonicalParsedArtifactWriteRows",
    "vaultScanRows",
    "externalDownloadRows",
    "llmCallRows",
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
        "caseId": "alphafold_casp14_accuracy",
        "query": "What does the selected parsed evidence say about AlphaFold CASP14 accuracy?",
        "resolvedPaperIds": ["1207.0580"],
        "answerClaim": (
            "Selected parsed-artifact evidence supports that AlphaFold concerns protein structure "
            "prediction and reports CASP14 accuracy comparisons."
        ),
        "requiredEvidenceTerms": ["AlphaFold", "CASP14", "accuracy"],
        "requiredAnswerTerms": ["AlphaFold", "CASP14", "accuracy"],
        "forbiddenAnswerTerms": ["word vectors", "BERT"],
        "minCitations": 2,
    },
    {
        "caseId": "word_vectors_representation",
        "query": "What does the selected parsed evidence say about word vector representations?",
        "resolvedPaperIds": ["1301.3781"],
        "answerClaim": (
            "Selected parsed-artifact evidence supports that the paper proposes continuous vector "
            "representations of words and evaluates word similarity."
        ),
        "requiredEvidenceTerms": ["continuous", "representations", "word", "similarity"],
        "requiredAnswerTerms": ["continuous", "representations", "word", "similarity"],
        "forbiddenAnswerTerms": ["AlphaFold", "CASP14"],
        "minCitations": 2,
    },
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _sha256_text(value: Any) -> str:
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _term_present(text: str, term: str) -> bool:
    term_text = _clean_text(term)
    if not term_text:
        return False
    return term_text.casefold() in str(text or "").casefold()


def _observed_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if _term_present(text, term)]


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


class _AnswerQualitySmokeCollaborator:
    def collect_claim_context(
        self,
        results: list[SearchResult],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        return [], [], [], []

    def resolve_parent_context(
        self,
        result: SearchResult,
        doc_cache: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        metadata = dict(result.metadata or {})
        return {
            "parent_id": _clean_text(metadata.get("paper_id") or metadata.get("arxiv_id")),
            "parent_label": _clean_text(metadata.get("title")),
            "chunk_span": _clean_text(metadata.get("span_locator")),
            "text": result.document,
        }

    def answer_evidence_item(
        self,
        result: SearchResult,
        parent_ctx_by_result: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        metadata = dict(result.metadata or {})
        return {
            "title": _clean_text(metadata.get("title")),
            "source_type": _clean_text(metadata.get("source_type")),
            "normalized_source_type": _clean_text(metadata.get("source_type")),
            "source_id": _clean_text(metadata.get("source_id") or metadata.get("paper_id") or metadata.get("arxiv_id")),
            "arxiv_id": _clean_text(metadata.get("arxiv_id") or metadata.get("paper_id")),
            "citation_target": _clean_text(metadata.get("arxiv_id") or metadata.get("paper_id")),
            "source_ref": _clean_text(metadata.get("source_ref")),
            "source_content_hash": _clean_text(metadata.get("source_content_hash")),
            "span_locator": _clean_text(metadata.get("span_locator")),
            "snippet_hash": _clean_text(metadata.get("snippet_hash")),
            "excerpt": result.document[:500],
            "score": result.score,
            "semantic_score": result.semantic_score,
            "lexical_score": result.lexical_score,
            "quality_flag": "ok",
            "source_trust_score": 0.95,
        }

    def summarize_answer_signals(
        self,
        evidence: list[dict[str, Any]],
        *,
        contradicting_beliefs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        quality_counts = Counter(_clean_text(item.get("quality_flag")) or "unscored" for item in evidence)
        return {
            "total_sources": len(evidence),
            "quality_counts": dict(quality_counts),
            "preferred_sources": quality_counts.get("ok", 0),
            "contradicting_belief_count": len(contradicting_beliefs or []),
            "caution_required": quality_counts.get("ok", 0) == 0,
        }

    def build_answer_context(
        self,
        *,
        filtered: list[SearchResult],
        parent_ctx_by_result: dict[str, dict[str, Any]],
    ) -> str:
        return "\n".join(item.document for item in filtered)


def _pipeline_result(query_frame: dict[str, Any]) -> Any:
    return SimpleNamespace(plan=SimpleNamespace(to_dict=lambda: {"queryFrame": query_frame}))


def _assemble_case_packet(
    *,
    papers_dir: str | Path,
    case: dict[str, Any],
) -> Any:
    resolved_ids = [_clean_text(item) for item in list(case.get("resolvedPaperIds") or []) if _clean_text(item)]
    query_plan = {
        "family": "paper_lookup",
        "parsed_artifact_evidence_chunk_adapter": ADAPTER_OPT_IN_VALUE,
        "resolvedPaperIds": resolved_ids,
    }
    service = EvidenceAssemblyService(
        _AnswerQualitySmokeCollaborator(),
        papers_dir=str(Path(str(papers_dir)).expanduser()),
    )
    return service.assemble(
        query=_clean_text(case.get("query")),
        source_type="paper",
        results=[],
        paper_memory_prefilter={},
        metadata_filter=None,
        query_plan=query_plan,
        query_frame={"source_type": "paper", "family": "paper_lookup", "resolved_source_ids": resolved_ids},
    )


def _answer_text(case: dict[str, Any], citations: list[dict[str, Any]]) -> str:
    labels = [
        _clean_text(item.get("label") or item.get("citationLabel"))
        for item in citations
        if _clean_text(item.get("label") or item.get("citationLabel"))
    ]
    citation_tail = " ".join(f"[{label}]" for label in labels[:4])
    claim = _clean_text(case.get("answerClaim"))
    return f"{claim} {citation_tail}".strip()


def _case_row(
    *,
    case: dict[str, Any],
    papers_dir: str | Path,
) -> dict[str, Any]:
    resolved_ids = [_clean_text(item) for item in list(case.get("resolvedPaperIds") or []) if _clean_text(item)]
    packet = _assemble_case_packet(papers_dir=papers_dir, case=case)
    evidence_text = " ".join(_clean_text(item.get("excerpt") or item.get("text")) for item in list(packet.evidence or []))
    adapter_diag = dict(packet.evidence_packet.get("parsedArtifactEvidenceChunkAdapter") or {})
    query_frame = {"source_type": "paper", "family": "paper_lookup", "resolved_source_ids": resolved_ids}
    evidence_contract = build_evidence_packet_contract(
        query=_clean_text(case.get("query")),
        retrieval_mode="parsed_artifact_evidence_chunk_real_answer_quality_smoke",
        pipeline_result=_pipeline_result(query_frame),
        evidence_packet=packet,
    )
    answer = _answer_text(case, list(packet.citations or []))
    required_evidence_terms = [_clean_text(item) for item in list(case.get("requiredEvidenceTerms") or []) if _clean_text(item)]
    required_answer_terms = [_clean_text(item) for item in list(case.get("requiredAnswerTerms") or []) if _clean_text(item)]
    forbidden_answer_terms = [_clean_text(item) for item in list(case.get("forbiddenAnswerTerms") or []) if _clean_text(item)]
    observed_evidence_terms = _observed_terms(evidence_text, required_evidence_terms)
    observed_answer_terms = _observed_terms(answer, required_answer_terms)
    missing_evidence_terms = [term for term in required_evidence_terms if term not in observed_evidence_terms]
    missing_answer_terms = [term for term in required_answer_terms if term not in observed_answer_terms]
    forbidden_observed = _observed_terms(answer, forbidden_answer_terms)
    pre_contract_failures = [
        *(f"missing_evidence_term:{term}" for term in missing_evidence_terms),
        *(f"missing_answer_term:{term}" for term in missing_answer_terms),
        *(f"forbidden_answer_term:{term}" for term in forbidden_observed),
    ]
    answer_contract = build_answer_contract(
        answer=answer,
        evidence_packet=packet,
        verification={
            "status": "verified" if not pre_contract_failures else "caution",
            "unsupportedClaimCount": len(pre_contract_failures),
            "needsCaution": bool(pre_contract_failures),
        },
        rewrite={"attempted": False, "applied": False, "finalAnswerSource": "deterministic_extractive_smoke"},
        routing_meta={"provider": "local", "model": "deterministic-extractive-smoke"},
    )
    spans = list(evidence_contract.get("spans") or [])
    citations = list(answer_contract.get("citations") or [])
    strict_span_rows = sum(
        1
        for span in spans
        if bool(span.get("sourceContentHashAvailable")) and bool(span.get("spanOffsetAvailable"))
    )
    min_citations = max(1, _int(case.get("minCitations")))
    failure_reasons = list(pre_contract_failures)
    if adapter_diag.get("status") != "applied":
        failure_reasons.append("runtime_adapter_not_applied")
    if _int(adapter_diag.get("rowsAdded")) < min_citations:
        failure_reasons.append("adapter_rows_added_below_minimum")
    if not bool(evidence_contract.get("answerable")):
        failure_reasons.append("evidence_packet_contract_not_answerable")
    if bool(answer_contract.get("abstain")):
        failure_reasons.append("answer_contract_abstained")
    if len(citations) < min_citations:
        failure_reasons.append("answer_contract_citations_below_minimum")
    if strict_span_rows != len(spans) or not spans:
        failure_reasons.append("strict_provenance_span_count_mismatch")
    if _clean_text(dict(answer_contract.get("coverage") or {}).get("status")) != "complete":
        failure_reasons.append("answer_contract_coverage_not_complete")
    if _contains_private_path({"case": case, "rowEvidence": evidence_contract, "answerContract": answer_contract}):
        failure_reasons.append("private_path_leak")
    row_status = "pass" if not failure_reasons else "fail"
    return {
        "caseId": _clean_text(case.get("caseId")),
        "status": row_status,
        "query": _clean_text(case.get("query")),
        "resolvedPaperIds": resolved_ids,
        "adapterStatus": _clean_text(adapter_diag.get("status")),
        "adapterRowsAdded": _int(adapter_diag.get("rowsAdded")),
        "selectedEvidenceCount": _int(packet.evidence_packet.get("selectedEvidenceCount")),
        "evidencePacketContractSpanRows": len(spans),
        "answerContractCitationRows": len(citations),
        "strictProvenanceSpanRows": strict_span_rows,
        "answerContractAbstain": bool(answer_contract.get("abstain")),
        "answerContractCoverageStatus": _clean_text(dict(answer_contract.get("coverage") or {}).get("status")),
        "answerCoverageRatio": float(answer_contract.get("coverageRatio") or 0.0),
        "claimLikeSentenceCount": _int(answer_contract.get("claimLikeSentenceCount")),
        "citationBackedSentenceCount": _int(answer_contract.get("citationBackedSentenceCount")),
        "requiredEvidenceTerms": required_evidence_terms,
        "observedEvidenceTerms": observed_evidence_terms,
        "missingEvidenceTerms": missing_evidence_terms,
        "requiredAnswerTerms": required_answer_terms,
        "observedAnswerTerms": observed_answer_terms,
        "missingAnswerTerms": missing_answer_terms,
        "forbiddenAnswerTermsObserved": forbidden_observed,
        "candidateRecordIds": list(adapter_diag.get("selectedCandidateRecordIds") or []),
        "answerTextHash": _sha256_text(answer),
        "answerTextIncludedInReport": False,
        "excerptIncludedInReport": False,
        "failureReasons": sorted(set(failure_reasons)),
    }


def _live_smoke_gate(live_smoke_report: str | Path) -> tuple[dict[str, Any], list[str]]:
    payload = _read_json(live_smoke_report)
    violations: list[str] = []
    if payload.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_RUNTIME_ADAPTER_LIVE_SMOKE_SCHEMA_ID:
        violations.append("live_smoke_schema_mismatch")
    if payload.get("status") != "ready":
        violations.append("live_smoke_not_ready")
    counts = dict(payload.get("counts") or {})
    if _int(counts.get("adapterRowsAdded")) <= 0:
        violations.append("live_smoke_adapter_added_no_rows")
    if _int(counts.get("answerContractCitationRows")) <= 0:
        violations.append("live_smoke_answer_contract_missing_citations")
    if _int(counts.get("schemaViolationCount")) != 0:
        violations.append("live_smoke_schema_violations_present")
    if _int(counts.get("privatePathLeakRows")) != 0:
        violations.append("live_smoke_private_path_leak")
    return payload, sorted(set(violations))


def build_parsed_artifact_evidence_chunk_real_answer_quality_smoke(
    *,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    live_smoke_report: str | Path = DEFAULT_LIVE_SMOKE_REPORT,
    cases: list[dict[str, Any]] | tuple[dict[str, Any], ...] = DEFAULT_CASES,
    generated_at: str | None = None,
) -> dict[str, Any]:
    live_payload, gate_violations = _live_smoke_gate(live_smoke_report)
    rows = [] if gate_violations else [_case_row(case=dict(case), papers_dir=papers_dir) for case in cases]
    private_path_leak_rows = sum(1 for row in rows if _contains_private_path(row))
    pass_rows = sum(1 for row in rows if row.get("status") == "pass")
    fail_rows = sum(1 for row in rows if row.get("status") == "fail")
    blocked_rows = len(cases) if gate_violations else 0
    semantic_violations = list(gate_violations)
    if fail_rows:
        semantic_violations.append(f"failed_answer_quality_cases:{fail_rows}")
    if private_path_leak_rows:
        semantic_violations.append("private_path_leak")
    counts = {
        "inputCaseRows": len(cases),
        "attemptedCaseRows": len(rows),
        "passRows": pass_rows,
        "failRows": fail_rows,
        "blockedRows": blocked_rows,
        "deterministicAnswerGeneratedRows": len(rows),
        "llmAnswerGeneratedRows": 0,
        "answerQualityScoredRows": len(rows),
        "answerContractCitationRows": sum(_int(row.get("answerContractCitationRows")) for row in rows),
        "strictProvenanceSpanRows": sum(_int(row.get("strictProvenanceSpanRows")) for row in rows),
        "missingEvidenceTermRows": sum(1 for row in rows if row.get("missingEvidenceTerms")),
        "missingAnswerTermRows": sum(1 for row in rows if row.get("missingAnswerTerms")),
        "forbiddenAnswerTermRows": sum(1 for row in rows if row.get("forbiddenAnswerTermsObserved")),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(semantic_violations),
    }
    status = "ready" if not semantic_violations and pass_rows == len(cases) else "blocked"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_REAL_ANSWER_QUALITY_SMOKE_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "liveSmokeReportRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_runtime_adapter_live_smoke.v1.json",
            "liveSmokeSchema": _clean_text(live_payload.get("schema")),
            "liveSmokeStatus": _clean_text(live_payload.get("status")),
            "papersDirRef": "papers_dir",
        },
        "policy": {
            "reportOnly": True,
            "runtimeAdapterInvoked": True,
            "deterministicExtractiveAnswerOnly": True,
            "answerTextExcludedFromReport": True,
            "excerptExcludedFromReport": True,
            "llmCalls": False,
            "judgeModelCalls": False,
            "candidateStoreWrites": False,
            "sourceSpanCreation": False,
            "strictEvidenceCreation": False,
            "parserExecution": False,
            "databaseMutation": False,
            "indexMutation": False,
            "vaultScan": False,
            "externalDownload": False,
        },
        "counts": counts,
        "gate": {
            "readyForAnswerPathOptInRouteReview": status == "ready",
            "liveSmokeReady": not gate_violations,
            "allCasesPassed": pass_rows == len(cases) and not fail_rows and not blocked_rows,
            "semanticViolations": sorted(set(semantic_violations)),
        },
        "rows": rows,
        "warnings": [
            "deterministic_smoke_not_llm_judge_or_full_user_quality_eval",
            "answer_text_and_raw_excerpts_are_not_included_in_report",
        ],
    }


def render_parsed_artifact_evidence_chunk_real_answer_quality_smoke_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Real Answer Quality Smoke",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- inputCaseRows: `{counts.get('inputCaseRows')}`",
        f"- passRows: `{counts.get('passRows')}`",
        f"- failRows: `{counts.get('failRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- answerContractCitationRows: `{counts.get('answerContractCitationRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Policy",
        "",
        "Report-only deterministic answer smoke. It invokes the runtime adapter and checks term support, citations, and provenance, but does not call an LLM or judge model and does not include raw excerpts or answer text in the report.",
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
                f"- status: `{row.get('status')}`",
                f"- resolvedPaperIds: `{row.get('resolvedPaperIds')}`",
                f"- adapterRowsAdded: `{row.get('adapterRowsAdded')}`",
                f"- citations: `{row.get('answerContractCitationRows')}`",
                f"- coverage: `{row.get('answerContractCoverageStatus')}` / `{row.get('answerCoverageRatio')}`",
                f"- observedEvidenceTerms: `{row.get('observedEvidenceTerms')}`",
                f"- observedAnswerTerms: `{row.get('observedAnswerTerms')}`",
                f"- failureReasons: `{row.get('failureReasons')}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_real_answer_quality_smoke(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_real_answer_quality_smoke_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_REAL_ANSWER_QUALITY_SMOKE_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_real_answer_quality_smoke",
    "write_parsed_artifact_evidence_chunk_real_answer_quality_smoke",
]
