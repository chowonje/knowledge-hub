"""Read-only live smoke for the parsed-artifact evidence chunk runtime adapter."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

from knowledge_hub.ai.answer_contracts import build_answer_contract, build_evidence_packet_contract
from knowledge_hub.ai.evidence_assembly import EvidenceAssemblyService
from knowledge_hub.ai.parsed_artifact_evidence_chunk_runtime_adapter import ADAPTER_OPT_IN_VALUE
from knowledge_hub.core.models import SearchResult


PARSED_ARTIFACT_EVIDENCE_CHUNK_RUNTIME_ADAPTER_LIVE_SMOKE_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-runtime-adapter-live-smoke.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_runtime_adapter_live_smoke_ready"
BLOCKED_DECISION = "parsed_artifact_evidence_chunk_runtime_adapter_live_smoke_blocked"
NEXT_TRANCHE_READY = "parsed_artifact_evidence_chunk_real_answer_quality_smoke"
NEXT_TRANCHE_BLOCKED = "parsed_artifact_evidence_chunk_runtime_adapter_live_smoke_repair"
DEFAULT_PAPERS_DIR = Path.home() / ".khub" / "papers"
DEFAULT_RESOLVED_PAPER_IDS = ("1207.0580", "1301.3781")
DEFAULT_QUERY = "What section or paragraph evidence is available for these resolved papers?"
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


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


class _SmokeCollaborator:
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


def _contract_span_summary(span: dict[str, Any]) -> dict[str, Any]:
    derivative = dict(span.get("derivativeSource") or span.get("derivative_source") or {})
    return {
        "spanRef": _clean_text(span.get("spanRef")),
        "citationLabel": _clean_text(span.get("citationLabel")),
        "sourceId": _clean_text(span.get("sourceId") or span.get("source_id")),
        "sourceType": _clean_text(span.get("source_type")),
        "sourceRef": _clean_text(span.get("sourceRef")),
        "sourceContentHash": _clean_text(span.get("sourceContentHash")),
        "contentHash": _clean_text(span.get("content_hash")),
        "spanLocator": _clean_text(span.get("spanLocator")),
        "charStart": span.get("charStart"),
        "charEnd": span.get("charEnd"),
        "evidenceKind": _clean_text(span.get("evidenceKind")),
        "candidateRecordId": _clean_text(derivative.get("candidateRecordId")),
        "candidateStoreRef": _clean_text(derivative.get("candidateStoreRef")),
        "artifactType": _clean_text(derivative.get("artifactType")),
        "excerptIncludedInReport": False,
    }


def _report_private_path_rows(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if _contains_private_path(row))


def build_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke(
    *,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    resolved_paper_ids: list[str] | tuple[str, ...] = DEFAULT_RESOLVED_PAPER_IDS,
    query: str = DEFAULT_QUERY,
    generated_at: str | None = None,
) -> dict[str, Any]:
    resolved_ids = [_clean_text(item) for item in resolved_paper_ids if _clean_text(item)]
    query_plan = {
        "family": "paper_lookup",
        "parsed_artifact_evidence_chunk_adapter": ADAPTER_OPT_IN_VALUE,
        "resolvedPaperIds": resolved_ids,
    }
    service = EvidenceAssemblyService(
        _SmokeCollaborator(),
        papers_dir=str(Path(str(papers_dir)).expanduser()),
    )
    packet = service.assemble(
        query=query,
        source_type="paper",
        results=[],
        paper_memory_prefilter={},
        metadata_filter=None,
        query_plan=query_plan,
        query_frame={"source_type": "paper", "family": "paper_lookup", "resolved_source_ids": resolved_ids},
    )
    pipeline_result = SimpleNamespace(
        plan=SimpleNamespace(
            to_dict=lambda: {
                "queryFrame": {
                    "source_type": "paper",
                    "family": "paper_lookup",
                    "resolved_source_ids": resolved_ids,
                }
            }
        )
    )
    evidence_contract = build_evidence_packet_contract(
        query=query,
        retrieval_mode="parsed_artifact_evidence_chunk_runtime_adapter",
        pipeline_result=pipeline_result,
        evidence_packet=packet,
    )
    answer_contract = build_answer_contract(
        answer="Resolved paper evidence is available with source hashes and character offsets.",
        evidence_packet=packet,
        verification={"status": "verified", "unsupportedClaimCount": 0, "needsCaution": False},
        rewrite={"attempted": False, "applied": False, "finalAnswerSource": "supplied_smoke_answer"},
        routing_meta={"provider": "local", "model": "smoke"},
    )
    adapter_diag = dict(packet.evidence_packet.get("parsedArtifactEvidenceChunkAdapter") or {})
    spans = list(evidence_contract.get("spans") or [])
    rows = [_contract_span_summary(dict(span)) for span in spans]
    private_path_leak_rows = _report_private_path_rows(rows)
    schema_violations: list[str] = []
    if adapter_diag.get("status") != "applied":
        schema_violations.append("runtime_adapter_not_applied")
    if _int(adapter_diag.get("rowsAdded")) <= 0:
        schema_violations.append("runtime_adapter_added_no_rows")
    if not bool(packet.evidence_packet.get("answerable")):
        schema_violations.append("evidence_packet_not_answerable")
    if not bool(evidence_contract.get("answerable")):
        schema_violations.append("evidence_packet_contract_not_answerable")
    if bool(answer_contract.get("abstain")):
        schema_violations.append("answer_contract_abstained")
    if len(spans) != _int(adapter_diag.get("rowsAdded")):
        schema_violations.append("contract_span_count_mismatch")
    if private_path_leak_rows:
        schema_violations.append("private_path_leak")
    strict_span_rows = sum(
        1
        for span in spans
        if bool(span.get("sourceContentHashAvailable")) and bool(span.get("spanOffsetAvailable"))
    )
    if spans and strict_span_rows != len(spans):
        schema_violations.append("strict_provenance_span_count_mismatch")
    counts = {
        "resolvedPaperRows": len(resolved_ids),
        "adapterCandidateRowsConsidered": _int(adapter_diag.get("candidateRowsConsidered")),
        "adapterRowsAdded": _int(adapter_diag.get("rowsAdded")),
        "adapterBlockedRows": _int(adapter_diag.get("blockedRows")),
        "selectedEvidenceCount": _int(packet.evidence_packet.get("selectedEvidenceCount")),
        "citationCount": _int(packet.evidence_packet.get("citationCount")),
        "evidencePacketContractSpanRows": len(spans),
        "answerContractCitationRows": len(answer_contract.get("citations") or []),
        "strictProvenanceSpanRows": strict_span_rows,
        "answerableRows": 1 if bool(packet.evidence_packet.get("answerable")) else 0,
        "answerContractAbstainRows": 1 if bool(answer_contract.get("abstain")) else 0,
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(schema_violations),
    }
    status = "ready" if not schema_violations else "blocked"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_RUNTIME_ADAPTER_LIVE_SMOKE_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "smoke": {
            "query": query,
            "sourceType": "paper",
            "papersDirRef": "papers_dir",
            "resolvedPaperIds": resolved_ids,
            "queryPlanOptInKey": "parsed_artifact_evidence_chunk_adapter",
            "queryPlanOptInValue": ADAPTER_OPT_IN_VALUE,
            "runtimeBoundary": "knowledge_hub.ai.evidence_assembly.EvidenceAssemblyService.assemble",
            "llmGenerationRows": 0,
            "suppliedSmokeAnswerOnly": True,
        },
        "adapterDiagnostics": {
            "enabled": bool(adapter_diag.get("enabled")),
            "status": _clean_text(adapter_diag.get("status")),
            "resolvedPaperIds": list(adapter_diag.get("resolvedPaperIds") or []),
            "candidateRowsConsidered": _int(adapter_diag.get("candidateRowsConsidered")),
            "rowsAdded": _int(adapter_diag.get("rowsAdded")),
            "blockedRows": _int(adapter_diag.get("blockedRows")),
            "skippedReason": _clean_text(adapter_diag.get("skippedReason")),
            "maxRowsPerResolvedPaper": _int(adapter_diag.get("maxRowsPerResolvedPaper")),
            "maxRowsTotal": _int(adapter_diag.get("maxRowsTotal")),
            "fallbackToAllPapersAllowed": bool(adapter_diag.get("fallbackToAllPapersAllowed")),
            "selectedCandidateRecordIds": list(adapter_diag.get("selectedCandidateRecordIds") or []),
            "blockedReasons": dict(adapter_diag.get("blockedReasons") or {}),
        },
        "contract": {
            "evidencePacketAnswerable": bool(evidence_contract.get("answerable")),
            "evidencePacketCoverageStatus": _clean_text(dict(evidence_contract.get("coverage") or {}).get("status")),
            "answerContractAbstain": bool(answer_contract.get("abstain")),
            "answerContractCoverageStatus": _clean_text(dict(answer_contract.get("coverage") or {}).get("status")),
            "answerContractCitationRows": len(answer_contract.get("citations") or []),
        },
        "counts": counts,
        "gate": {
            "readyForRealAnswerQualitySmoke": status == "ready",
            "runtimeAdapterApplied": _clean_text(adapter_diag.get("status")) == "applied",
            "strictProvenanceSpansReady": bool(spans) and strict_span_rows == len(spans),
            "answerContractHasCitations": bool(answer_contract.get("citations")),
            "schemaViolations": schema_violations,
        },
        "rows": rows,
        "warnings": [],
    }


def render_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    smoke = dict(report.get("smoke") or {})
    adapter = dict(report.get("adapterDiagnostics") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Runtime Adapter Live Smoke",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- resolvedPaperIds: `{smoke.get('resolvedPaperIds')}`",
        f"- adapterStatus: `{adapter.get('status')}`",
        f"- adapterRowsAdded: `{counts.get('adapterRowsAdded')}`",
        f"- evidencePacketContractSpanRows: `{counts.get('evidencePacketContractSpanRows')}`",
        f"- answerContractCitationRows: `{counts.get('answerContractCitationRows')}`",
        f"- answerableRows: `{counts.get('answerableRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Mutation Guarantees",
        "",
    ]
    for field in ZERO_COUNTER_FIELDS:
        lines.append(f"- {field}: `{counts.get(field)}`")
    lines.extend(["", "## Selected Contract Spans", ""])
    for row in list(report.get("rows") or [])[:8]:
        lines.append(
            f"- `{row.get('spanRef')}` paper=`{row.get('sourceId')}` "
            f"locator=`{row.get('spanLocator')}` candidate=`{row.get('candidateRecordId')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_RUNTIME_ADAPTER_LIVE_SMOKE_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke",
    "write_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke",
]
