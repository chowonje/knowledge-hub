"""Report-only design for parsed-artifact evidence chunk answer-path opt-in ingress."""

from __future__ import annotations

from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.ai import rag as rag_module
from knowledge_hub.interfaces.cli.commands import search_cmd
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_opt_in_route_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_ROUTE_REVIEW_SCHEMA_ID,
    READY_DECISION as ROUTE_REVIEW_READY_DECISION,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_IMPLEMENTATION_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-opt-in-implementation-design.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design_ready"
BLOCKED_DECISION = "parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design_blocked"
NEXT_TRANCHE_READY = "parsed_artifact_evidence_chunk_answer_path_opt_in_searcher_ingress_implementation"
NEXT_TRANCHE_BLOCKED = "parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design_repair"
DEFAULT_ROUTE_REVIEW_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_opt_in_route_review.v1.json"
)

ZERO_COUNTER_FIELDS = (
    "runtimeRouteWriteRows",
    "publicCliFlagRows",
    "defaultOnRows",
    "answerVisibleDefaultRows",
    "answerGenerationRows",
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


def _signature_has_param(target: Any, name: str) -> bool:
    try:
        return name in inspect.signature(target).parameters
    except Exception:
        return False


def _source_contains(target: Any, *needles: str) -> bool:
    try:
        source = inspect.getsource(target)
    except Exception:
        return False
    return all(needle in source for needle in needles)


def _source_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_ROUTE_REVIEW_SCHEMA_ID:
        blockers.append("route_review_schema_mismatch")
    if report.get("status") != "ready":
        blockers.append("route_review_not_ready")
    if report.get("decision") != ROUTE_REVIEW_READY_DECISION:
        blockers.append("route_review_decision_not_ready")
    if gate.get("readyForOptInImplementationDesign") is not True:
        blockers.append("route_review_gate_not_ready_for_implementation_design")
    if gate.get("answerQualitySmokeReady") is not True:
        blockers.append("answer_quality_smoke_not_ready")
    if gate.get("internalRuntimeQueryPlanIngressReady") is not True:
        blockers.append("internal_runtime_query_plan_ingress_not_ready")
    if gate.get("publicSearcherIngressGap") is not True:
        blockers.append("public_searcher_ingress_gap_not_confirmed")
    if gate.get("publicCliDefaultUnchanged") is not True:
        blockers.append("public_cli_default_changed")
    for field_name in (
        "routeReviewFailRows",
        "privatePathLeakRows",
        "schemaViolationCount",
        *ZERO_COUNTER_FIELDS,
    ):
        if _int(counts.get(field_name)) != 0:
            blockers.append(f"route_review_has_{field_name}")
    if _contains_private_path(report):
        blockers.append("route_review_private_path_leak")
    return sorted(set(blockers))


def _row(
    *,
    row_id: str,
    design_layer: str,
    file_ref: str,
    planned_change: str,
    current_observed: bool,
    ready: bool,
    blockers: list[str] | None = None,
    next_check: str = "",
) -> dict[str, Any]:
    row_blockers = sorted(set(blockers or []))
    status = "design_ready" if ready and not row_blockers else "blocked"
    return {
        "rowId": row_id,
        "designLayer": design_layer,
        "fileRef": file_ref,
        "plannedChange": _clean_text(planned_change),
        "currentObserved": bool(current_observed),
        "status": status,
        "blockers": row_blockers,
        "nextCheck": _clean_text(next_check),
    }


def _implementation_rows(source_blockers: list[str]) -> list[dict[str, Any]]:
    generate_has_query_plan = _signature_has_param(rag_module.RAGSearcher.generate_answer, "query_plan")
    stream_has_query_plan = _signature_has_param(rag_module.RAGSearcher.stream_answer, "query_plan")
    generate_forwards_query_plan = _source_contains(
        rag_module.RAGSearcher.generate_answer,
        "generate_answer_runtime",
        "query_plan=query_plan",
    )
    stream_forwards_query_plan = _source_contains(
        rag_module.RAGSearcher.stream_answer,
        "stream_answer_runtime",
        "query_plan=query_plan",
    )
    cli_has_public_flag = _source_contains(search_cmd.ask, "--parsed-artifact-evidence-chunk", "parsed_artifact_evidence_chunk")
    inherited_blockers = list(source_blockers)
    return [
        _row(
            row_id="rag_searcher_generate_answer_add_query_plan_param",
            design_layer="public_python_searcher_api",
            file_ref="knowledge_hub/ai/rag.py",
            planned_change="Add keyword-only query_plan: Optional[Dict[str, Any]] = None to RAGSearcher.generate_answer.",
            current_observed=not generate_has_query_plan,
            ready=not generate_has_query_plan,
            blockers=inherited_blockers + ([] if not generate_has_query_plan else ["generate_answer_query_plan_already_present"]),
            next_check="Signature keeps default None so existing callers remain compatible.",
        ),
        _row(
            row_id="rag_searcher_generate_answer_forward_query_plan",
            design_layer="runtime_forwarding",
            file_ref="knowledge_hub/ai/rag.py",
            planned_change="Forward query_plan=query_plan from RAGSearcher.generate_answer to rag_answer_runtime.generate_answer.",
            current_observed=not generate_forwards_query_plan,
            ready=not generate_forwards_query_plan,
            blockers=inherited_blockers + ([] if not generate_forwards_query_plan else ["generate_answer_forwarding_already_present"]),
            next_check="Unit test should monkeypatch generate_answer_runtime and assert object identity or equality for query_plan.",
        ),
        _row(
            row_id="rag_searcher_stream_answer_add_query_plan_param",
            design_layer="public_python_searcher_api",
            file_ref="knowledge_hub/ai/rag.py",
            planned_change="Add keyword-only query_plan: Optional[Dict[str, Any]] = None to RAGSearcher.stream_answer for parity.",
            current_observed=not stream_has_query_plan,
            ready=not stream_has_query_plan,
            blockers=inherited_blockers + ([] if not stream_has_query_plan else ["stream_answer_query_plan_already_present"]),
            next_check="Streaming remains opt-in and default behavior is unchanged when query_plan is None.",
        ),
        _row(
            row_id="rag_searcher_stream_answer_forward_query_plan",
            design_layer="runtime_forwarding",
            file_ref="knowledge_hub/ai/rag.py",
            planned_change="Forward query_plan=query_plan from RAGSearcher.stream_answer to rag_answer_runtime.stream_answer.",
            current_observed=not stream_forwards_query_plan,
            ready=not stream_forwards_query_plan,
            blockers=inherited_blockers + ([] if not stream_forwards_query_plan else ["stream_answer_forwarding_already_present"]),
            next_check="Unit test should monkeypatch stream_answer_runtime and assert query_plan is passed through.",
        ),
        _row(
            row_id="khub_ask_no_public_flag",
            design_layer="public_cli",
            file_ref="knowledge_hub/interfaces/cli/commands/search_cmd.py",
            planned_change="Keep khub ask public flags unchanged; no parsed-artifact evidence chunk CLI flag in this tranche.",
            current_observed=not cli_has_public_flag,
            ready=not cli_has_public_flag,
            blockers=inherited_blockers + ([] if not cli_has_public_flag else ["public_cli_flag_already_present"]),
            next_check="CLI tests should verify no public opt-in flag is introduced by the searcher ingress implementation.",
        ),
        _row(
            row_id="default_none_preserves_behavior",
            design_layer="compatibility",
            file_ref="knowledge_hub/ai/rag.py",
            planned_change="Keep query_plan optional and default None so absent opt-in preserves existing search/ask behavior.",
            current_observed=True,
            ready=True,
            blockers=inherited_blockers,
            next_check="Regression test should call generate_answer without query_plan and assert forwarded value is None.",
        ),
        _row(
            row_id="activation_remains_adapter_owned",
            design_layer="answerability_policy",
            file_ref="knowledge_hub/ai/parsed_artifact_evidence_chunk_runtime_adapter.py",
            planned_change="Do not duplicate activation policy in RAGSearcher; adapter remains responsible for opt-in, source type, resolved-paper, hash, locator, and row-cap checks.",
            current_observed=True,
            ready=True,
            blockers=inherited_blockers,
            next_check="Implementation should only pass query_plan through and leave adapter gating unchanged.",
        ),
    ]


def build_parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design(
    *,
    route_review_report: dict[str, Any] | None = None,
    route_review_report_path: str | Path = DEFAULT_ROUTE_REVIEW_REPORT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_report = dict(route_review_report or _read_json(route_review_report_path))
    source_blockers = _source_blockers(source_report)
    rows = _implementation_rows(source_blockers)
    blocked_rows = [row for row in rows if row.get("status") != "design_ready"]
    private_path_leak_rows = sum(1 for row in rows if _contains_private_path(row))
    semantic_violations = list(source_blockers)
    semantic_violations.extend(str(row.get("rowId")) for row in blocked_rows)
    if private_path_leak_rows:
        semantic_violations.append("private_path_leak")
    counts = {
        "inputRouteReviewRows": 1 if source_report else 0,
        "routeReviewPassRows": _int(dict(source_report.get("counts") or {}).get("routeReviewPassRows")),
        "routeReviewGapRows": _int(dict(source_report.get("counts") or {}).get("routeReviewGapRows")),
        "designRows": len(rows),
        "designReadyRows": sum(1 for row in rows if row.get("status") == "design_ready"),
        "blockedRows": len(blocked_rows),
        "plannedSearcherIngressRows": 2,
        "plannedRuntimeForwardingRows": 2,
        "plannedTestRows": 4,
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(set(semantic_violations)),
    }
    status = "ready" if not semantic_violations else "blocked"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_IMPLEMENTATION_DESIGN_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "routeReviewReportRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_opt_in_route_review.v1.json",
            "routeReviewSchema": _clean_text(source_report.get("schema")),
            "routeReviewStatus": _clean_text(source_report.get("status")),
            "routeReviewDecision": _clean_text(source_report.get("decision")),
        },
        "policy": {
            "reportOnly": True,
            "implementationDesignOnly": True,
            "runtimeCodeChanged": False,
            "publicCliFlagAdded": False,
            "defaultOn": False,
            "answerGenerationRun": False,
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
            "readyForSearcherIngressImplementation": status == "ready",
            "routeReviewReady": not source_blockers,
            "internalRuntimeQueryPlanIngressReady": dict(source_report.get("gate") or {}).get(
                "internalRuntimeQueryPlanIngressReady"
            )
            is True,
            "publicSearcherIngressGapConfirmed": dict(source_report.get("gate") or {}).get("publicSearcherIngressGap")
            is True,
            "publicCliDefaultUnchanged": dict(source_report.get("gate") or {}).get("publicCliDefaultUnchanged") is True,
            "plannedPublicCliChange": False,
            "semanticViolations": sorted(set(semantic_violations)),
        },
        "design": {
            "ingressMode": "internal_python_api_opt_in",
            "targetClass": "knowledge_hub.ai.rag.RAGSearcher",
            "targetMethods": ["generate_answer", "stream_answer"],
            "queryPlanParameter": "query_plan",
            "queryPlanType": "Optional[Dict[str, Any]]",
            "queryPlanDefault": None,
            "forwardTo": [
                "knowledge_hub.ai.rag_answer_runtime.generate_answer",
                "knowledge_hub.ai.rag_answer_runtime.stream_answer",
            ],
            "activationOwnedBy": "knowledge_hub.ai.parsed_artifact_evidence_chunk_runtime_adapter",
            "activationKeys": ["parsed_artifact_evidence_chunk_adapter", "parsedArtifactEvidenceChunkAdapter"],
            "activationValue": "runtime_v1",
            "publicCliFlag": None,
            "defaultBehavior": "unchanged_when_query_plan_is_none",
            "plannedTests": [
                "RAGSearcher.generate_answer forwards query_plan",
                "RAGSearcher.generate_answer defaults query_plan to None",
                "RAGSearcher.stream_answer forwards query_plan",
                "khub ask has no public parsed-artifact evidence chunk flag",
            ],
        },
        "rows": rows,
        "warnings": [
            "report_only_no_runtime_code_change",
            "next_tranche_must_keep_query_plan_default_none",
            "next_tranche_must_not_add_public_cli_flag",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    design = dict(report.get("design") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path Opt-in Implementation Design",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- designRows: `{counts.get('designRows')}`",
        f"- designReadyRows: `{counts.get('designReadyRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- plannedSearcherIngressRows: `{counts.get('plannedSearcherIngressRows')}`",
        f"- plannedRuntimeForwardingRows: `{counts.get('plannedRuntimeForwardingRows')}`",
        f"- publicSearcherIngressGapConfirmed: `{gate.get('publicSearcherIngressGapConfirmed')}`",
        f"- plannedPublicCliChange: `{gate.get('plannedPublicCliChange')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Design",
        "",
        f"- ingressMode: `{design.get('ingressMode')}`",
        f"- targetClass: `{design.get('targetClass')}`",
        f"- targetMethods: `{', '.join(str(item) for item in list(design.get('targetMethods') or []))}`",
        f"- queryPlanParameter: `{design.get('queryPlanParameter')}`",
        f"- defaultBehavior: `{design.get('defaultBehavior')}`",
        f"- publicCliFlag: `{design.get('publicCliFlag')}`",
        "",
        "## Policy",
        "",
        "Report-only implementation design. It fixes the future searcher ingress contract; it does not change runtime code, add public CLI flags, generate answers, call LLMs, mutate stores, or scan the vault.",
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
                f"- layer: `{row.get('designLayer')}`",
                f"- status: `{row.get('status')}`",
                f"- file: `{row.get('fileRef')}`",
                f"- currentObserved: `{row.get('currentObserved')}`",
                f"- plannedChange: {row.get('plannedChange')}",
                f"- blockers: `{blockers}`",
                f"- nextCheck: {row.get('nextCheck')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_IMPLEMENTATION_DESIGN_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design",
    "write_parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design",
]
