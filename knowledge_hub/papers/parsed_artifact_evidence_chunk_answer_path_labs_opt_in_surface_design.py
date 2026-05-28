"""Report-only labs opt-in surface design for parsed-artifact evidence chunks."""

from __future__ import annotations

from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.ai import rag as rag_module
from knowledge_hub.ai.parsed_artifact_evidence_chunk_runtime_adapter import (
    ADAPTER_OPT_IN_VALUE,
    MAX_ROWS_PER_RESOLVED_PAPER,
    MAX_ROWS_TOTAL,
    OPT_IN_KEYS,
)
from knowledge_hub.application.mcp.responses import DEFAULT_TOOL_NAMES
from knowledge_hub.interfaces.cli import main as cli_main
from knowledge_hub.interfaces.cli.commands import paper_labs_cmd, search_cmd
from knowledge_hub.mcp import tool_specs
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_DEFAULT_OFF_NO_ANSWER_REGRESSION_SMOKE_SCHEMA_ID,
    READY_DECISION as DEFAULT_OFF_READY_DECISION,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-labs-opt-in-surface-design.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design_ready"
BLOCKED_DECISION = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design_blocked"
NEXT_TRANCHE_READY = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_implementation"
NEXT_TRANCHE_BLOCKED = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design_repair"
DEFAULT_DEFAULT_OFF_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke.v1.json"
)
FUTURE_LABS_CLI_COMMAND = "khub labs paper evidence-chunk-ask"
FUTURE_LABS_MCP_TOOL = "paper_evidence_chunk_answer_preview"

ZERO_COUNTER_FIELDS = (
    "runtimeRouteWriteRows",
    "publicCliFlagRows",
    "defaultMcpSchemaChangeRows",
    "defaultOnRows",
    "answerGenerationRows",
    "answerVisibleRows",
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
    "llmCallRows",
    "judgeModelCallRows",
    "modelApiCallRows",
)
UPSTREAM_ZERO_FIELDS = (
    "failRows",
    "answerableRows",
    "evidencePacketContractAnswerableRows",
    "adapterAppliedRows",
    "adapterRowsAdded",
    "selectedEvidenceCount",
    "citationCount",
    "evidencePacketContractSpanRows",
    "localFakeLlmCallRows",
    "externalLlmCallRows",
    "modelApiCallRows",
    "privatePathLeakRows",
    "schemaViolationCount",
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


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _source_text(target: Any) -> str:
    try:
        return inspect.getsource(target)
    except Exception:
        return ""


def _source_has_all(target: Any, *needles: str) -> bool:
    source = _source_text(target)
    return bool(source) and all(needle in source for needle in needles)


def _signature_has_param(target: Any, name: str) -> bool:
    try:
        return name in inspect.signature(target).parameters
    except Exception:
        return False


def _tool_by_name(tools: list[Any], name: str) -> Any | None:
    for tool in tools:
        if getattr(tool, "name", "") == name:
            return tool
    return None


def _tool_properties(tool: Any | None) -> dict[str, Any]:
    schema = dict(getattr(tool, "inputSchema", {}) or {})
    props = dict(schema.get("properties") or {})
    return props


def _surface_observations(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    default_tools = tool_specs.build_tools(profile="default")
    labs_tools = tool_specs.build_tools(profile="labs")
    default_ask_props = _tool_properties(_tool_by_name(default_tools, "ask_knowledge"))
    labs_tool_names = {str(getattr(tool, "name", "")) for tool in labs_tools}
    default_tool_names = {str(getattr(tool, "name", "")) for tool in default_tools}
    ask_source = _source_text(search_cmd.ask)
    main_source = _source_text(cli_main)
    paper_labs_source = _source_text(paper_labs_cmd.paper_labs_group)
    public_opt_in_needles = (
        "--parsed-artifact-evidence-chunk",
        "parsed_artifact_evidence_chunk_adapter",
        "parsedArtifactEvidenceChunkAdapter",
    )
    observations = {
        "internalGenerateAnswerQueryPlanParam": _signature_has_param(
            rag_module.RAGSearcher.generate_answer,
            "query_plan",
        ),
        "internalStreamAnswerQueryPlanParam": _signature_has_param(
            rag_module.RAGSearcher.stream_answer,
            "query_plan",
        ),
        "khubAskPublicOptInFlagPresent": any(needle in ask_source for needle in public_opt_in_needles),
        "khubAskQueryPlanPublicArgPresent": "query_plan" in ask_source or "--query-plan" in ask_source,
        "defaultMcpAskAdapterArgPresent": any(key in default_ask_props for key in OPT_IN_KEYS),
        "defaultMcpAskQueryPlanArgPresent": "query_plan" in default_ask_props or "queryPlan" in default_ask_props,
        "defaultMcpAskToolInDefaultProfile": "ask_knowledge" in default_tool_names,
        "futureLabsMcpToolAlreadyPresent": FUTURE_LABS_MCP_TOOL in labs_tool_names,
        "futureLabsMcpToolInDefaultProfile": FUTURE_LABS_MCP_TOOL in default_tool_names,
        "labsMcpProfileAvailable": len(labs_tool_names) > len(default_tool_names),
        "defaultToolSetContainsOnlyAllowedDefaultNames": default_tool_names.issubset(set(DEFAULT_TOOL_NAMES)),
        "paperLabsCliGroupImportable": callable(getattr(paper_labs_cmd, "paper_labs_group", None)),
        "paperLabsCliRegisteredUnderLabs": "paper_labs_cmd" in main_source and '"paper"' in main_source,
        "futureLabsCliCommandAlreadyPresent": "evidence-chunk-ask" in paper_labs_source,
    }
    observations.update(dict(overrides or {}))
    return observations


def _source_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_DEFAULT_OFF_NO_ANSWER_REGRESSION_SMOKE_SCHEMA_ID:
        blockers.append("default_off_smoke_schema_mismatch")
    if report.get("status") != "ready":
        blockers.append("default_off_smoke_not_ready")
    if report.get("decision") != DEFAULT_OFF_READY_DECISION:
        blockers.append("default_off_smoke_decision_not_ready")
    if gate.get("readyForLabsOptInSurfaceDesign") is not True:
        blockers.append("default_off_smoke_gate_not_ready_for_labs_design")
    if gate.get("publicDefaultUnchanged") is not True:
        blockers.append("public_default_not_unchanged")
    if _int(counts.get("inputScenarioRows")) <= 0:
        blockers.append("default_off_smoke_has_no_scenarios")
    if _int(counts.get("passRows")) != _int(counts.get("inputScenarioRows")):
        blockers.append("default_off_smoke_not_all_passed")
    if _int(counts.get("noAnswerRows")) != _int(counts.get("inputScenarioRows")):
        blockers.append("default_off_smoke_not_all_no_answer")
    for field_name in UPSTREAM_ZERO_FIELDS:
        if _int(counts.get(field_name)) != 0:
            blockers.append(f"default_off_smoke_has_{field_name}")
    if _contains_private_path(report):
        blockers.append("default_off_smoke_private_path_leak")
    return sorted(set(blockers))


def _row(
    *,
    row_id: str,
    surface_layer: str,
    surface_ref: str,
    planned_surface: str,
    current_observed: bool,
    required_for_next: bool,
    ready: bool,
    safety_contract: str,
    blockers: list[str] | None = None,
    next_check: str = "",
) -> dict[str, Any]:
    row_blockers = sorted(set(blockers or []))
    status = "design_ready" if ready and not row_blockers else "blocked"
    return {
        "rowId": row_id,
        "surfaceLayer": surface_layer,
        "surfaceRef": surface_ref,
        "plannedSurface": _clean_text(planned_surface),
        "currentObserved": bool(current_observed),
        "requiredForNext": bool(required_for_next),
        "status": status,
        "safetyContract": _clean_text(safety_contract),
        "blockers": row_blockers,
        "nextCheck": _clean_text(next_check),
    }


def _surface_rows(source_blockers: list[str], observations: dict[str, Any]) -> list[dict[str, Any]]:
    inherited = list(source_blockers)
    public_cli_closed = not bool(observations.get("khubAskPublicOptInFlagPresent")) and not bool(
        observations.get("khubAskQueryPlanPublicArgPresent")
    )
    default_mcp_closed = not bool(observations.get("defaultMcpAskAdapterArgPresent")) and not bool(
        observations.get("defaultMcpAskQueryPlanArgPresent")
    )
    internal_ingress_ready = bool(observations.get("internalGenerateAnswerQueryPlanParam")) and bool(
        observations.get("internalStreamAnswerQueryPlanParam")
    )
    labs_cli_host_ready = bool(observations.get("paperLabsCliGroupImportable")) and bool(
        observations.get("paperLabsCliRegisteredUnderLabs")
    )
    labs_mcp_host_ready = bool(observations.get("labsMcpProfileAvailable")) and bool(
        observations.get("defaultToolSetContainsOnlyAllowedDefaultNames")
    )
    return [
        _row(
            row_id="default_off_no_answer_smoke_gate",
            surface_layer="upstream_gate",
            surface_ref="eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke.v1.json",
            planned_surface="Consume the default-off/no-answer smoke as the authority before designing any opt-in surface.",
            current_observed=not source_blockers,
            required_for_next=True,
            ready=not source_blockers,
            safety_contract="labs design is blocked unless the default answer path remains fail-closed.",
            blockers=inherited,
            next_check="Generated report must be ready with all scenarios no-answer and zero public/default/LLM/evidence mutation counters.",
        ),
        _row(
            row_id="internal_python_query_plan_ingress",
            surface_layer="internal_python_api",
            surface_ref="knowledge_hub/ai/rag.py",
            planned_surface="Future labs callers use RAGSearcher.generate_answer/stream_answer with query_plan opt-in instead of adding a public khub ask flag.",
            current_observed=internal_ingress_ready,
            required_for_next=True,
            ready=internal_ingress_ready,
            safety_contract="activation remains owned by the runtime adapter; absent query_plan keeps existing behavior.",
            blockers=inherited + ([] if internal_ingress_ready else ["internal_query_plan_ingress_missing"]),
            next_check="Implementation should pass query_plan keys through unchanged and keep defaults None.",
        ),
        _row(
            row_id="public_khub_ask_stays_closed",
            surface_layer="public_cli",
            surface_ref="knowledge_hub/interfaces/cli/commands/search_cmd.py",
            planned_surface="Do not add parsed-artifact evidence chunk flags to public khub ask in the labs opt-in tranche.",
            current_observed=public_cli_closed,
            required_for_next=True,
            ready=public_cli_closed,
            safety_contract="public CLI remains default-off; labs-only command is the only planned user-visible opt-in surface.",
            blockers=inherited + ([] if public_cli_closed else ["public_khub_ask_opt_in_surface_present"]),
            next_check="Focused CLI test should assert khub ask help has no parsed-artifact evidence chunk flag.",
        ),
        _row(
            row_id="default_mcp_ask_stays_closed",
            surface_layer="default_mcp",
            surface_ref="knowledge_hub/mcp/tool_specs.py",
            planned_surface="Do not add query_plan or adapter opt-in fields to the default ask_knowledge MCP schema.",
            current_observed=default_mcp_closed,
            required_for_next=True,
            ready=default_mcp_closed,
            safety_contract="default MCP profile stays retrieval-assistant-first and cannot activate evidence chunks by accident.",
            blockers=inherited + ([] if default_mcp_closed else ["default_mcp_ask_opt_in_arg_present"]),
            next_check="MCP tests should assert ask_knowledge default schema has no query_plan or adapter opt-in fields.",
        ),
        _row(
            row_id="labs_cli_paper_group_host",
            surface_layer="labs_cli",
            surface_ref="knowledge_hub/interfaces/cli/commands/paper_labs_cmd.py",
            planned_surface=f"Use future command `{FUTURE_LABS_CLI_COMMAND}` as the human/operator opt-in host.",
            current_observed=labs_cli_host_ready,
            required_for_next=True,
            ready=labs_cli_host_ready,
            safety_contract="command must require explicit --paper-id and --source paper, default --no-allow-external, and JSON diagnostics.",
            blockers=inherited + ([] if labs_cli_host_ready else ["labs_paper_cli_group_missing"]),
            next_check="Next tranche may add this labs subcommand only under khub labs paper.",
        ),
        _row(
            row_id="labs_mcp_profile_host",
            surface_layer="labs_mcp",
            surface_ref="knowledge_hub/mcp/tool_specs.py",
            planned_surface=f"Use future labs/all-only MCP tool `{FUTURE_LABS_MCP_TOOL}` for tool callers.",
            current_observed=labs_mcp_host_ready,
            required_for_next=True,
            ready=labs_mcp_host_ready,
            safety_contract="future tool must be absent from default profile and blocked by default MCP profile direct-call enforcement.",
            blockers=inherited + ([] if labs_mcp_host_ready else ["labs_mcp_profile_host_missing"]),
            next_check="Next tranche may add the tool only if default profile tests prove it is hidden/blocked.",
        ),
        _row(
            row_id="future_labs_cli_command_contract",
            surface_layer="future_labs_cli_contract",
            surface_ref="knowledge_hub/interfaces/cli/commands/paper_labs_cmd.py",
            planned_surface=(
                f"`{FUTURE_LABS_CLI_COMMAND} QUESTION --paper-id <id> --json` builds a paper-only query_plan "
                "with parsed_artifact_evidence_chunk_adapter=runtime_v1 and resolvedPaperIds."
            ),
            current_observed=bool(observations.get("futureLabsCliCommandAlreadyPresent")),
            required_for_next=False,
            ready=True,
            safety_contract="future command may expose answer payload only in labs and only after no-answer/default-off gates stay green.",
            blockers=inherited,
            next_check="Implementation must keep public khub ask unchanged and add focused labs CLI tests.",
        ),
        _row(
            row_id="future_labs_mcp_tool_contract",
            surface_layer="future_labs_mcp_contract",
            surface_ref="knowledge_hub/mcp/tool_specs.py",
            planned_surface=(
                f"`{FUTURE_LABS_MCP_TOOL}` accepts question plus explicit paper_ids and forwards the same "
                "query_plan opt-in through the internal searcher API."
            ),
            current_observed=bool(observations.get("futureLabsMcpToolAlreadyPresent")),
            required_for_next=False,
            ready=not bool(observations.get("futureLabsMcpToolInDefaultProfile")),
            safety_contract="future labs MCP tool must not be discoverable or callable from KHUB_MCP_PROFILE=default.",
            blockers=inherited
            + ([] if not bool(observations.get("futureLabsMcpToolInDefaultProfile")) else ["future_labs_mcp_tool_in_default"]),
            next_check="Implementation must extend default-profile block tests before returning answer-visible data.",
        ),
        _row(
            row_id="explicit_paper_scope_policy",
            surface_layer="answerability_policy",
            surface_ref="knowledge_hub/ai/parsed_artifact_evidence_chunk_runtime_adapter.py",
            planned_surface="Require source_type=paper and explicit resolvedPaperIds/paper_ids for every labs opt-in invocation.",
            current_observed=True,
            required_for_next=True,
            ready=True,
            safety_contract="no fallback to all papers, visual hints, table/equation/figure artifacts, or unresolved retrieval aliases.",
            blockers=inherited,
            next_check="Next tranche should test missing paper ids and non-paper sources still no-answer.",
        ),
        _row(
            row_id="local_first_call_policy",
            surface_layer="provider_policy",
            surface_ref="knowledge_hub/interfaces/cli/commands/paper_labs_cmd.py",
            planned_surface="Default labs opt-in answer runs with allow_external=false; external calls require an explicit later policy decision.",
            current_observed=True,
            required_for_next=True,
            ready=True,
            safety_contract="local-first remains the default; report-only design makes zero LLM/model/API calls.",
            blockers=inherited,
            next_check="Next tranche should keep default --no-allow-external and verify no API calls in smoke tests.",
        ),
    ]


def build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design(
    *,
    default_off_report: dict[str, Any] | None = None,
    default_off_report_path: str | Path = DEFAULT_DEFAULT_OFF_REPORT,
    surface_observation_overrides: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_report = dict(default_off_report or _read_json(default_off_report_path))
    source_blockers = _source_blockers(source_report)
    observations = _surface_observations(surface_observation_overrides)
    rows = _surface_rows(source_blockers, observations)
    blocked_rows = [row for row in rows if row.get("status") != "design_ready"]
    private_path_leak_rows = sum(1 for row in rows if _contains_private_path(row))
    semantic_violations = list(source_blockers)
    semantic_violations.extend(str(row.get("rowId")) for row in blocked_rows)
    if private_path_leak_rows:
        semantic_violations.append("private_path_leak")
    counts = {
        "inputDefaultOffSmokeRows": 1 if source_report else 0,
        "defaultOffSmokeScenarioRows": _int(dict(source_report.get("counts") or {}).get("inputScenarioRows")),
        "defaultOffSmokePassRows": _int(dict(source_report.get("counts") or {}).get("passRows")),
        "defaultOffSmokeNoAnswerRows": _int(dict(source_report.get("counts") or {}).get("noAnswerRows")),
        "designRows": len(rows),
        "designReadyRows": sum(1 for row in rows if row.get("status") == "design_ready"),
        "blockedRows": len(blocked_rows),
        "plannedLabsCliRows": 1,
        "plannedLabsMcpRows": 1,
        "plannedInternalApiRows": 1,
        "plannedPublicCliRows": 0,
        "plannedDefaultMcpRows": 0,
        "defaultPublicSurfaceUnchangedRows": int(not observations.get("khubAskPublicOptInFlagPresent")),
        "defaultMcpSurfaceUnchangedRows": int(not observations.get("defaultMcpAskAdapterArgPresent")),
        "requiredExplicitPaperIdRows": 1,
        "requiredSourcePaperRows": 1,
        "requiredAllowExternalDefaultFalseRows": 1,
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(set(semantic_violations)),
    }
    status = "ready" if not semantic_violations else "blocked"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_DESIGN_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "defaultOffNoAnswerSmokeReportRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke.v1.json",
            "defaultOffNoAnswerSmokeSchema": _clean_text(source_report.get("schema")),
            "defaultOffNoAnswerSmokeStatus": _clean_text(source_report.get("status")),
            "defaultOffNoAnswerSmokeDecision": _clean_text(source_report.get("decision")),
        },
        "policy": {
            "reportOnly": True,
            "surfaceDesignOnly": True,
            "runtimeCodeChanged": False,
            "publicCliFlagAdded": False,
            "defaultMcpSchemaChanged": False,
            "defaultOn": False,
            "answerGenerationRun": False,
            "answerVisibleRowsGenerated": False,
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
        "observations": observations,
        "counts": counts,
        "gate": {
            "readyForLabsOptInSurfaceImplementation": status == "ready",
            "defaultOffNoAnswerSmokeReady": not source_blockers,
            "internalSearcherIngressAvailable": bool(observations.get("internalGenerateAnswerQueryPlanParam"))
            and bool(observations.get("internalStreamAnswerQueryPlanParam")),
            "publicKhubAskClosed": not bool(observations.get("khubAskPublicOptInFlagPresent"))
            and not bool(observations.get("khubAskQueryPlanPublicArgPresent")),
            "defaultMcpAskClosed": not bool(observations.get("defaultMcpAskAdapterArgPresent"))
            and not bool(observations.get("defaultMcpAskQueryPlanArgPresent")),
            "labsCliHostAvailable": bool(observations.get("paperLabsCliGroupImportable"))
            and bool(observations.get("paperLabsCliRegisteredUnderLabs")),
            "labsMcpProfileAvailable": bool(observations.get("labsMcpProfileAvailable")),
            "plannedDefaultSurfaceChange": False,
            "plannedLabsOnly": True,
            "semanticViolations": sorted(set(semantic_violations)),
        },
        "design": {
            "surfaceMode": "labs_only_explicit_opt_in",
            "futureLabsCliCommand": FUTURE_LABS_CLI_COMMAND,
            "futureLabsMcpTool": FUTURE_LABS_MCP_TOOL,
            "allowedProfiles": ["labs", "all"],
            "defaultProfileAllowed": False,
            "publicKhubAskFlag": None,
            "defaultAskKnowledgeSchemaChange": False,
            "queryPlanOptInKeys": list(OPT_IN_KEYS),
            "queryPlanOptInValue": ADAPTER_OPT_IN_VALUE,
            "requiredSourceType": "paper",
            "requiredResolvedPaperIds": True,
            "allowExternalDefault": False,
            "maxRowsPerResolvedPaper": MAX_ROWS_PER_RESOLVED_PAPER,
            "maxRowsTotal": MAX_ROWS_TOTAL,
            "plannedCliOptions": ["question", "--paper-id", "--top-k", "--mode", "--json", "--no-allow-external"],
            "plannedMcpInputFields": ["question", "paper_ids", "top_k", "mode", "allow_external"],
            "forbiddenDefaultEvidence": [
                "visual_retrieval_hint",
                "fallback_chunk",
                "locator_only_anchor",
                "table",
                "equation",
                "figure_caption",
            ],
            "plannedTests": [
                "khub ask help has no parsed-artifact evidence chunk flag",
                "default MCP ask_knowledge has no query_plan or adapter opt-in fields",
                "labs CLI command requires explicit paper id",
                "labs MCP tool is hidden/blocked in KHUB_MCP_PROFILE=default",
                "labs opt-in smoke keeps missing-paper/non-paper scenarios no-answer",
            ],
        },
        "rows": rows,
        "warnings": [
            "report_only_no_surface_implementation",
            "next_tranche_must_keep_public_khub_ask_unchanged",
            "next_tranche_must_hide_labs_mcp_tool_from_default_profile",
            "next_tranche_must_require_explicit_paper_ids",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    design = dict(report.get("design") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in Surface Design",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- designRows: `{counts.get('designRows')}`",
        f"- designReadyRows: `{counts.get('designReadyRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- plannedLabsCliRows: `{counts.get('plannedLabsCliRows')}`",
        f"- plannedLabsMcpRows: `{counts.get('plannedLabsMcpRows')}`",
        f"- plannedPublicCliRows: `{counts.get('plannedPublicCliRows')}`",
        f"- plannedDefaultMcpRows: `{counts.get('plannedDefaultMcpRows')}`",
        f"- publicKhubAskClosed: `{gate.get('publicKhubAskClosed')}`",
        f"- defaultMcpAskClosed: `{gate.get('defaultMcpAskClosed')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Design",
        "",
        f"- surfaceMode: `{design.get('surfaceMode')}`",
        f"- futureLabsCliCommand: `{design.get('futureLabsCliCommand')}`",
        f"- futureLabsMcpTool: `{design.get('futureLabsMcpTool')}`",
        f"- allowedProfiles: `{', '.join(str(item) for item in list(design.get('allowedProfiles') or []))}`",
        f"- queryPlanOptInValue: `{design.get('queryPlanOptInValue')}`",
        f"- requiredSourceType: `{design.get('requiredSourceType')}`",
        f"- requiredResolvedPaperIds: `{design.get('requiredResolvedPaperIds')}`",
        f"- allowExternalDefault: `{design.get('allowExternalDefault')}`",
        "",
        "## Policy",
        "",
        "Report-only surface design. It plans a labs-only explicit opt-in surface and keeps public `khub ask` plus default MCP `ask_knowledge` closed. It does not implement a command, generate answers, call LLMs, write stores, create evidence records, mutate indexes, or scan the vault.",
        "",
        "## Rows",
        "",
    ]
    for row in list(report.get("rows") or []):
        blockers = ", ".join(str(item) for item in list(row.get("blockers") or []))
        lines.extend(
            [
                f"### `{row.get('rowId')}`",
                "",
                f"- layer: `{row.get('surfaceLayer')}`",
                f"- status: `{row.get('status')}`",
                f"- surface: `{row.get('surfaceRef')}`",
                f"- currentObserved: `{row.get('currentObserved')}`",
                f"- requiredForNext: `{row.get('requiredForNext')}`",
                f"- plannedSurface: {row.get('plannedSurface')}",
                f"- safetyContract: {row.get('safetyContract')}",
                f"- blockers: `{blockers}`",
                f"- nextCheck: {row.get('nextCheck')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_DESIGN_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design",
    "write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design",
]
