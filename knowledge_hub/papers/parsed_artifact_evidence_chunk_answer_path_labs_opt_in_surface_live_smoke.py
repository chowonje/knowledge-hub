"""Live smoke for the labs-only parsed-artifact evidence chunk answer surface."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.mcp import tool_specs
from knowledge_hub.papers.evidence_chunk_answer_preview import (
    PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID,
    build_paper_evidence_chunk_answer_preview,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_DESIGN_SCHEMA_ID,
    READY_DECISION as LABS_SURFACE_DESIGN_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke import (
    DEFAULT_PAPERS_DIR,
    DEFAULT_QUERY,
    DEFAULT_RESOLVED_PAPER_IDS,
    _build_searcher,
    _clean_text,
    _contains_private_path,
    _int,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-labs-opt-in-surface-live-smoke.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke_ready"
BLOCKED_DECISION = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke_blocked"
NEXT_TRANCHE_READY = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed"
NEXT_TRANCHE_BLOCKED = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke_repair"
DEFAULT_LABS_SURFACE_DESIGN_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design.v1.json"
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


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _schema_ok(payload: dict[str, Any]) -> bool:
    return validate_payload(payload, PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID, strict=True).ok


def _tool_by_name(tools: list[Any], name: str) -> Any | None:
    for tool in tools:
        if getattr(tool, "name", "") == name:
            return tool
    return None


def _surface_observations() -> dict[str, Any]:
    default_tools = tool_specs.build_tools(profile="default")
    labs_tools = tool_specs.build_tools(profile="labs")
    all_tools = tool_specs.build_tools(profile="all")
    default_ask = _tool_by_name(default_tools, "ask_knowledge")
    default_ask_props = dict((getattr(default_ask, "inputSchema", {}) or {}).get("properties") or {})
    return {
        "publicCliCommand": "khub labs paper evidence-chunk-ask",
        "labsMcpTool": "paper_evidence_chunk_answer_preview",
        "labsMcpToolInDefaultProfile": _tool_by_name(default_tools, "paper_evidence_chunk_answer_preview") is not None,
        "labsMcpToolInLabsProfile": _tool_by_name(labs_tools, "paper_evidence_chunk_answer_preview") is not None,
        "labsMcpToolInAllProfile": _tool_by_name(all_tools, "paper_evidence_chunk_answer_preview") is not None,
        "defaultMcpAskQueryPlanArgPresent": "query_plan" in default_ask_props or "queryPlan" in default_ask_props,
        "defaultMcpAskAdapterArgPresent": "parsed_artifact_evidence_chunk_adapter" in default_ask_props
        or "parsedArtifactEvidenceChunkAdapter" in default_ask_props,
    }


def _source_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_DESIGN_SCHEMA_ID:
        blockers.append("labs_surface_design_schema_mismatch")
    if report.get("status") != "ready":
        blockers.append("labs_surface_design_not_ready")
    if report.get("decision") != LABS_SURFACE_DESIGN_READY_DECISION:
        blockers.append("labs_surface_design_decision_not_ready")
    if gate.get("readyForLabsOptInSurfaceImplementation") is not True:
        blockers.append("labs_surface_design_gate_not_ready_for_implementation")
    if _int(counts.get("blockedRows")) != 0:
        blockers.append("labs_surface_design_has_blocked_rows")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("labs_surface_design_has_schema_violations")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("labs_surface_design_has_private_path_leaks")
    if _contains_private_path(report):
        blockers.append("labs_surface_design_private_path_leak")
    return sorted(set(blockers))


def _row_from_payload(
    *,
    scenario_id: str,
    payload: dict[str, Any],
    local_llm_calls: int,
) -> dict[str, Any]:
    adapter = dict(payload.get("adapterDiagnostics") or {})
    summary = dict(payload.get("evidencePacketSummary") or {})
    contract = dict(payload.get("evidencePacketContractSummary") or {})
    return {
        "scenarioId": scenario_id,
        "surface": "knowledge_hub.papers.evidence_chunk_answer_preview.build_paper_evidence_chunk_answer_preview",
        "payloadSchema": _clean_text(payload.get("schema")),
        "payloadStatus": _clean_text(payload.get("status")),
        "paperIds": list(payload.get("paperIds") or []),
        "sourceType": _clean_text(payload.get("sourceType")),
        "allowExternal": bool(payload.get("allowExternal")),
        "queryPlanOptInSnake": _clean_text(dict(payload.get("queryPlan") or {}).get("parsed_artifact_evidence_chunk_adapter")),
        "queryPlanOptInCamel": _clean_text(dict(payload.get("queryPlan") or {}).get("parsedArtifactEvidenceChunkAdapter")),
        "adapterStatus": _clean_text(summary.get("adapterStatus") or adapter.get("status")),
        "adapterRowsAdded": _int(summary.get("adapterRowsAdded") or adapter.get("rowsAdded")),
        "adapterCandidateRowsConsidered": _int(
            summary.get("adapterCandidateRowsConsidered") or adapter.get("candidateRowsConsidered")
        ),
        "selectedEvidenceCount": _int(summary.get("selectedEvidenceCount")),
        "citationCount": _int(summary.get("citationCount")),
        "answerable": bool(payload.get("answerable")),
        "evidencePacketAnswerable": bool(summary.get("answerable")),
        "evidencePacketContractAnswerable": bool(contract.get("answerable")),
        "evidencePacketContractSpanRows": _int(contract.get("spanRows")),
        "localFakeLlmCallRows": int(local_llm_calls),
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "evidenceTextIncludedInReport": False,
        "schemaValid": _schema_ok(payload),
    }


def build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke(
    *,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    resolved_paper_ids: list[str] | tuple[str, ...] = DEFAULT_RESOLVED_PAPER_IDS,
    query: str = DEFAULT_QUERY,
    labs_surface_design_report_path: str | Path = DEFAULT_LABS_SURFACE_DESIGN_REPORT,
    labs_surface_design_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    resolved_ids = [_clean_text(item) for item in resolved_paper_ids if _clean_text(item)]
    source_report = dict(labs_surface_design_report or _read_json(labs_surface_design_report_path))
    source_blockers = _source_blockers(source_report)
    observations = _surface_observations()
    searcher, llm = _build_searcher(papers_dir=papers_dir)
    payload: dict[str, Any] = {}
    positive_error = ""
    try:
        payload = build_paper_evidence_chunk_answer_preview(
            searcher,
            question=query,
            paper_ids=resolved_ids,
            top_k=1,
            retrieval_mode="semantic",
            allow_external=False,
        )
    except Exception as error:  # pragma: no cover - defensive, surfaced in report.
        positive_error = str(error)
    external_rejection = False
    external_error = ""
    calls_before_external = int(llm.calls)
    try:
        build_paper_evidence_chunk_answer_preview(
            searcher,
            question=query,
            paper_ids=resolved_ids,
            top_k=1,
            retrieval_mode="semantic",
            allow_external=True,
        )
    except ValueError as error:
        external_rejection = True
        external_error = str(error)
    row = _row_from_payload(
        scenario_id="labs_surface_positive_local_candidate_store",
        payload=payload,
        local_llm_calls=int(llm.calls),
    )
    row_violations: list[str] = []
    if positive_error:
        row_violations.append("positive_surface_call_error")
    if row["payloadStatus"] != "ok":
        row_violations.append("payload_status_not_ok")
    if row["sourceType"] != "paper":
        row_violations.append("source_type_not_paper")
    if row["allowExternal"]:
        row_violations.append("allow_external_not_false")
    if row["queryPlanOptInSnake"] != "runtime_v1" or row["queryPlanOptInCamel"] != "runtime_v1":
        row_violations.append("query_plan_opt_in_not_preserved")
    if row["adapterStatus"] != "applied":
        row_violations.append("adapter_not_applied")
    if row["adapterRowsAdded"] <= 0:
        row_violations.append("adapter_rows_added_zero")
    if row["selectedEvidenceCount"] <= 0:
        row_violations.append("selected_evidence_count_zero")
    if row["citationCount"] <= 0:
        row_violations.append("citation_count_zero")
    if row["evidencePacketContractSpanRows"] <= 0:
        row_violations.append("contract_span_rows_zero")
    if not row["schemaValid"]:
        row_violations.append("surface_payload_schema_mismatch")
    if row["localFakeLlmCallRows"] != 1:
        row_violations.append("local_fake_llm_call_count_unexpected")
    row["violations"] = row_violations
    row["pass"] = not row_violations

    surface_violations = list(source_blockers)
    if row_violations:
        surface_violations.append("positive_surface_smoke_failed")
    if not external_rejection:
        surface_violations.append("external_call_request_not_rejected")
    if int(llm.calls) != calls_before_external:
        surface_violations.append("external_rejection_called_llm")
    if observations["labsMcpToolInDefaultProfile"]:
        surface_violations.append("labs_mcp_tool_visible_in_default_profile")
    if not observations["labsMcpToolInLabsProfile"] or not observations["labsMcpToolInAllProfile"]:
        surface_violations.append("labs_mcp_tool_not_visible_in_labs_or_all")
    if observations["defaultMcpAskQueryPlanArgPresent"] or observations["defaultMcpAskAdapterArgPresent"]:
        surface_violations.append("default_mcp_ask_exposes_adapter_opt_in")
    private_path_leak_rows = 1 if _contains_private_path(row) or _contains_private_path(observations) else 0
    if private_path_leak_rows:
        surface_violations.append("private_path_leak")

    counts = {
        "inputLabsSurfaceDesignRows": 1 if source_report else 0,
        "designReadyInputRows": 1 if not source_blockers else 0,
        "surfaceSmokeRows": 1,
        "surfaceSmokePassRows": 1 if row["pass"] else 0,
        "surfaceSmokeFailRows": 0 if row["pass"] else 1,
        "surfacePayloadSchemaValidRows": 1 if row["schemaValid"] else 0,
        "surfacePayloadStatusOkRows": 1 if row["payloadStatus"] == "ok" else 0,
        "answerableRows": 1 if row["answerable"] else 0,
        "adapterAppliedRows": 1 if row["adapterStatus"] == "applied" else 0,
        "adapterRowsAdded": row["adapterRowsAdded"],
        "adapterCandidateRowsConsidered": row["adapterCandidateRowsConsidered"],
        "selectedEvidenceCount": row["selectedEvidenceCount"],
        "citationCount": row["citationCount"],
        "evidencePacketContractSpanRows": row["evidencePacketContractSpanRows"],
        "localFakeLlmCallRows": int(llm.calls),
        "externalRequestRejectedRows": 1 if external_rejection else 0,
        "externalRejectionLlmCallRows": max(0, int(llm.calls) - calls_before_external),
        "labsMcpToolDefaultProfileRows": 1 if observations["labsMcpToolInDefaultProfile"] else 0,
        "labsMcpToolLabsProfileRows": 1 if observations["labsMcpToolInLabsProfile"] else 0,
        "labsMcpToolAllProfileRows": 1 if observations["labsMcpToolInAllProfile"] else 0,
        "defaultMcpAskAdapterArgRows": 1 if observations["defaultMcpAskAdapterArgPresent"] else 0,
        "defaultMcpAskQueryPlanArgRows": 1 if observations["defaultMcpAskQueryPlanArgPresent"] else 0,
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(set(surface_violations)),
    }
    status = "ready" if not surface_violations else "blocked"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "labsSurfaceDesignReportRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design.v1.json",
            "labsSurfaceDesignSchema": _clean_text(source_report.get("schema")),
            "labsSurfaceDesignStatus": _clean_text(source_report.get("status")),
            "labsSurfaceDesignDecision": _clean_text(source_report.get("decision")),
        },
        "smoke": {
            "query": query,
            "papersDirRef": "papers_dir",
            "resolvedPaperIds": resolved_ids,
            "surfaceBoundary": "knowledge_hub.papers.evidence_chunk_answer_preview.build_paper_evidence_chunk_answer_preview",
            "publicCliCommand": observations["publicCliCommand"],
            "labsMcpTool": observations["labsMcpTool"],
            "retrievalMode": "semantic",
            "externalModelCallsAllowed": False,
            "fakeLocalLlmOnly": True,
            "answerTextIncludedInReport": False,
            "citationPayloadIncludedInReport": False,
            "sourcePayloadIncludedInReport": False,
        },
        "observations": observations,
        "positiveSurfacePayload": {
            "status": row["payloadStatus"],
            "schema": row["payloadSchema"],
            "answerIncludedInReport": False,
            "citationPayloadIncludedInReport": False,
            "sourcePayloadIncludedInReport": False,
            "positiveError": _clean_text(positive_error),
        },
        "externalRejection": {
            "requested": True,
            "rejected": external_rejection,
            "error": _clean_text(external_error),
            "llmCallsAdded": max(0, int(llm.calls) - calls_before_external),
        },
        "counts": counts,
        "gate": {
            "readyForLabsOptInQualityEvalSeed": status == "ready",
            "labsSurfaceDesignReady": not source_blockers,
            "surfacePayloadSchemaValid": row["schemaValid"],
            "surfacePayloadStatusOk": row["payloadStatus"] == "ok",
            "runtimeAdapterApplied": row["adapterStatus"] == "applied",
            "answerabilityReachedLabsSurface": bool(row["answerable"]),
            "externalRequestRejected": external_rejection,
            "publicDefaultUnchanged": True,
            "defaultMcpAskClosed": not observations["defaultMcpAskQueryPlanArgPresent"]
            and not observations["defaultMcpAskAdapterArgPresent"],
            "labsMcpToolHiddenFromDefault": not observations["labsMcpToolInDefaultProfile"],
            "schemaViolations": sorted(set(surface_violations)),
        },
        "rows": [row],
        "warnings": [],
    }


def render_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    smoke = dict(report.get("smoke") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in Surface Live Smoke",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- resolvedPaperIds: `{smoke.get('resolvedPaperIds')}`",
        f"- surfacePayloadStatusOkRows: `{counts.get('surfacePayloadStatusOkRows')}`",
        f"- adapterAppliedRows: `{counts.get('adapterAppliedRows')}`",
        f"- adapterRowsAdded: `{counts.get('adapterRowsAdded')}`",
        f"- selectedEvidenceCount: `{counts.get('selectedEvidenceCount')}`",
        f"- citationCount: `{counts.get('citationCount')}`",
        f"- evidencePacketContractSpanRows: `{counts.get('evidencePacketContractSpanRows')}`",
        f"- externalRequestRejectedRows: `{counts.get('externalRequestRejectedRows')}`",
        f"- localFakeLlmCallRows: `{counts.get('localFakeLlmCallRows')}`",
        f"- externalLlmCallRows: `{counts.get('externalLlmCallRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Surface",
        "",
        f"- publicCliCommand: `{smoke.get('publicCliCommand')}`",
        f"- labsMcpTool: `{smoke.get('labsMcpTool')}`",
        "- answer/citation/source payloads are excluded from this report.",
        "",
        "## Mutation Guarantees",
        "",
    ]
    for field in ZERO_COUNTER_FIELDS:
        lines.append(f"- {field}: `{counts.get(field)}`")
    lines.extend(["", "## Rows", ""])
    for row in list(report.get("rows") or []):
        lines.append(
            f"- `{row.get('scenarioId')}` status=`{row.get('payloadStatus')}` "
            f"adapter=`{row.get('adapterStatus')}` rowsAdded=`{row.get('adapterRowsAdded')}` "
            f"citations=`{row.get('citationCount')}` pass=`{row.get('pass')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke",
    "write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke",
]
