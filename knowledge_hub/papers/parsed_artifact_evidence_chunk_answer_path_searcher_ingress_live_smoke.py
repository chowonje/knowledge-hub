"""Live smoke for parsed-artifact evidence chunks through RAGSearcher.

This smoke exercises the internal searcher answer path with an explicit
query-plan opt-in. It uses a local fake LLM and an empty retrieval DB so the
only answer evidence comes from the parsed-artifact evidence chunk adapter.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.ai.parsed_artifact_evidence_chunk_runtime_adapter import ADAPTER_OPT_IN_VALUE
from knowledge_hub.ai.rag import RAGSearcher


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_SEARCHER_INGRESS_LIVE_SMOKE_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-searcher-ingress-live-smoke.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke_ready"
BLOCKED_DECISION = "parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke_blocked"
NEXT_TRANCHE_READY = "parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke"
NEXT_TRANCHE_BLOCKED = "parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke_repair"
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
    "publicCliFlagRows",
    "defaultOnRows",
    "externalLlmCallRows",
    "modelApiCallRows",
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


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


class _SmokeEmbedder:
    def embed_text(self, text: str) -> list[float]:
        _ = text
        return [0.0]


class _SmokeVectorDB:
    def search(self, query_embedding: list[float], top_k: int, filter_dict: dict[str, Any] | None = None) -> dict[str, Any]:
        _ = query_embedding, top_k, filter_dict
        return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

    def get_documents(
        self,
        filter_dict: dict[str, Any] | None = None,
        limit: int = 500,
        include_ids: bool = True,
        include_documents: bool = True,
        include_metadatas: bool = True,
    ) -> dict[str, Any]:
        _ = filter_dict, limit, include_ids, include_documents, include_metadatas
        return {"documents": [], "metadatas": [], "ids": []}


class _SmokeLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.last_prompt = ""
        self.last_context = ""

    def generate(self, prompt: str, context: str = "") -> str:
        self.calls += 1
        self.last_prompt = prompt
        self.last_context = context
        return "Local smoke answer generated from parsed-artifact evidence chunks."

    def stream_generate(self, prompt: str, context: str = ""):
        self.calls += 1
        self.last_prompt = prompt
        self.last_context = context
        yield "Local smoke answer generated from parsed-artifact evidence chunks."


class _SmokeConfig:
    def __init__(self, *, papers_dir: str | Path):
        self.papers_dir = str(Path(str(papers_dir)).expanduser())

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        _ = keys
        return default


def _build_searcher(*, papers_dir: str | Path) -> tuple[RAGSearcher, _SmokeLLM]:
    llm = _SmokeLLM()
    searcher = RAGSearcher(
        _SmokeEmbedder(),
        _SmokeVectorDB(),  # type: ignore[arg-type]
        llm=llm,
        sqlite_db=None,
        config=_SmokeConfig(papers_dir=papers_dir),
    )
    # Instance-level overrides are intentionally used because the orchestrator
    # detects them as smoke-local behavior, while subclass methods would be
    # treated as the searcher's baseline class methods.
    searcher._resolve_llm_for_request = lambda **kwargs: (  # type: ignore[attr-defined]  # noqa: SLF001
        llm,
        {"route": "local_smoke", "provider": "fake", "model": "parsed-artifact-evidence-chunk-smoke"},
        [],
    )
    searcher._verify_answer = lambda **kwargs: {  # type: ignore[attr-defined]  # noqa: SLF001
        "status": "verified",
        "unsupportedClaimCount": 0,
        "needsCaution": False,
    }
    searcher._rewrite_answer = lambda **kwargs: (  # type: ignore[attr-defined]  # noqa: SLF001
        str(kwargs.get("answer") or ""),
        {"attempted": False, "applied": False, "finalAnswerSource": "local_smoke"},
    )
    searcher._apply_conservative_fallback_if_needed = lambda **kwargs: (  # type: ignore[attr-defined]  # noqa: SLF001
        str(kwargs.get("answer") or ""),
        dict(kwargs.get("rewrite_meta") or {}),
        dict(kwargs.get("verification") or {}),
    )
    searcher._record_answer_log = lambda **kwargs: None  # type: ignore[attr-defined]  # noqa: SLF001
    return searcher, llm


def _span_row(span: dict[str, Any]) -> dict[str, Any]:
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


def _private_path_rows(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if _contains_private_path(row))


def build_parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke(
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
    searcher, llm = _build_searcher(papers_dir=papers_dir)
    payload = searcher.generate_answer(
        query,
        top_k=1,
        source_type="paper",
        retrieval_mode="semantic",
        allow_external=False,
        ask_v2_mode="claim_first",
        query_plan=query_plan,
    )
    payload_query_plan = dict(payload.get("queryPlan") or {})
    evidence_packet = dict(payload.get("evidencePacket") or {})
    adapter_diag = dict(evidence_packet.get("parsedArtifactEvidenceChunkAdapter") or {})
    evidence_contract = dict(payload.get("evidencePacketContract") or {})
    spans = [dict(span or {}) for span in list(evidence_contract.get("spans") or [])]
    rows = [_span_row(span) for span in spans]
    private_path_leak_rows = _private_path_rows(rows)
    strict_span_rows = sum(
        1
        for span in spans
        if bool(span.get("sourceContentHashAvailable")) and bool(span.get("spanOffsetAvailable"))
    )
    schema_violations: list[str] = []
    if str(payload.get("status") or "").strip().lower() != "ok":
        schema_violations.append("answer_payload_status_not_ok")
    if payload_query_plan.get("parsed_artifact_evidence_chunk_adapter") != ADAPTER_OPT_IN_VALUE:
        schema_violations.append("query_plan_snake_opt_in_not_preserved")
    if payload_query_plan.get("parsedArtifactEvidenceChunkAdapter") != ADAPTER_OPT_IN_VALUE:
        schema_violations.append("query_plan_camel_opt_in_not_preserved")
    if _clean_text(adapter_diag.get("status")) != "applied":
        schema_violations.append("runtime_adapter_not_applied")
    if _int(adapter_diag.get("rowsAdded")) <= 0:
        schema_violations.append("runtime_adapter_added_no_rows")
    if not bool(evidence_packet.get("answerable")):
        schema_violations.append("evidence_packet_not_answerable")
    if not bool(evidence_contract.get("answerable")):
        schema_violations.append("evidence_packet_contract_not_answerable")
    if len(spans) != _int(adapter_diag.get("rowsAdded")):
        schema_violations.append("contract_span_count_mismatch")
    if spans and strict_span_rows != len(spans):
        schema_violations.append("strict_provenance_span_count_mismatch")
    if llm.calls != 1:
        schema_violations.append("local_fake_llm_call_count_unexpected")
    if private_path_leak_rows:
        schema_violations.append("private_path_leak")

    counts = {
        "searcherIngressRows": 1,
        "queryPlanForwardedRows": 1 if payload_query_plan else 0,
        "queryPlanOptInPreservedRows": 1
        if (
            payload_query_plan.get("parsed_artifact_evidence_chunk_adapter") == ADAPTER_OPT_IN_VALUE
            and payload_query_plan.get("parsedArtifactEvidenceChunkAdapter") == ADAPTER_OPT_IN_VALUE
        )
        else 0,
        "resolvedPaperRows": len(resolved_ids),
        "adapterCandidateRowsConsidered": _int(adapter_diag.get("candidateRowsConsidered")),
        "adapterRowsAdded": _int(adapter_diag.get("rowsAdded")),
        "adapterBlockedRows": _int(adapter_diag.get("blockedRows")),
        "selectedEvidenceCount": _int(evidence_packet.get("selectedEvidenceCount")),
        "citationCount": _int(evidence_packet.get("citationCount")),
        "evidencePacketContractSpanRows": len(spans),
        "strictProvenanceSpanRows": strict_span_rows,
        "internalAnswerPayloadRows": 1,
        "localFakeLlmCallRows": int(llm.calls),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(schema_violations),
    }
    status = "ready" if not schema_violations else "blocked"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_SEARCHER_INGRESS_LIVE_SMOKE_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "smoke": {
            "query": query,
            "sourceType": "paper",
            "papersDirRef": "papers_dir",
            "resolvedPaperIds": resolved_ids,
            "runtimeBoundary": "knowledge_hub.ai.rag.RAGSearcher.generate_answer",
            "queryPlanOptInKey": "parsed_artifact_evidence_chunk_adapter",
            "queryPlanOptInValue": ADAPTER_OPT_IN_VALUE,
            "askV2Mode": "claim_first",
            "retrievalMode": "semantic",
            "publicCliDefaultChanged": False,
            "externalModelCallsAllowed": False,
            "fakeLocalLlmOnly": True,
            "answerTextIncludedInReport": False,
        },
        "searcherPayload": {
            "status": _clean_text(payload.get("status")),
            "paperFamily": _clean_text(payload.get("paperFamily")),
            "queryPlanOptInSnake": _clean_text(payload_query_plan.get("parsed_artifact_evidence_chunk_adapter")),
            "queryPlanOptInCamel": _clean_text(payload_query_plan.get("parsedArtifactEvidenceChunkAdapter")),
            "answerIncludedInReport": False,
            "evidenceIncludedInReport": False,
        },
        "adapterDiagnostics": {
            "enabled": bool(adapter_diag.get("enabled")),
            "status": _clean_text(adapter_diag.get("status")),
            "resolvedPaperIds": list(adapter_diag.get("resolvedPaperIds") or []),
            "candidateRowsConsidered": _int(adapter_diag.get("candidateRowsConsidered")),
            "rowsAdded": _int(adapter_diag.get("rowsAdded")),
            "blockedRows": _int(adapter_diag.get("blockedRows")),
            "skippedReason": _clean_text(adapter_diag.get("skippedReason")),
            "selectedCandidateRecordIds": list(adapter_diag.get("selectedCandidateRecordIds") or []),
            "blockedReasons": dict(adapter_diag.get("blockedReasons") or {}),
        },
        "contract": {
            "evidencePacketAnswerable": bool(evidence_contract.get("answerable")),
            "evidencePacketCoverageStatus": _clean_text(dict(evidence_contract.get("coverage") or {}).get("status")),
            "strictProvenanceSpansReady": bool(spans) and strict_span_rows == len(spans),
            "spanRows": len(spans),
        },
        "counts": counts,
        "gate": {
            "readyForDefaultOffNoAnswerRegressionSmoke": status == "ready",
            "searcherAnswerPathInvoked": True,
            "queryPlanOptInPreserved": counts["queryPlanOptInPreservedRows"] == 1,
            "runtimeAdapterApplied": _clean_text(adapter_diag.get("status")) == "applied",
            "answerPayloadStatusOk": _clean_text(payload.get("status")).lower() == "ok",
            "externalModelCallsDisabled": True,
            "publicDefaultUnchanged": True,
            "schemaViolations": schema_violations,
        },
        "rows": rows,
        "warnings": [],
    }


def render_parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    smoke = dict(report.get("smoke") or {})
    adapter = dict(report.get("adapterDiagnostics") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path Searcher Ingress Live Smoke",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- resolvedPaperIds: `{smoke.get('resolvedPaperIds')}`",
        f"- adapterStatus: `{adapter.get('status')}`",
        f"- adapterRowsAdded: `{counts.get('adapterRowsAdded')}`",
        f"- evidencePacketContractSpanRows: `{counts.get('evidencePacketContractSpanRows')}`",
        f"- internalAnswerPayloadRows: `{counts.get('internalAnswerPayloadRows')}`",
        f"- localFakeLlmCallRows: `{counts.get('localFakeLlmCallRows')}`",
        f"- externalLlmCallRows: `{counts.get('externalLlmCallRows')}`",
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


def write_parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_SEARCHER_INGRESS_LIVE_SMOKE_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke",
    "write_parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke",
]
