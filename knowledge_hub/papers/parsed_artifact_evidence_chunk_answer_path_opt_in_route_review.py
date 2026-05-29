"""Report-only review for routing parsed-artifact evidence chunk opt-in through ask."""

from __future__ import annotations

from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.ai import rag_answer_runtime
from knowledge_hub.ai import rag as rag_module
from knowledge_hub.ai import ask_v2
from knowledge_hub.ai import answer_payload_builder
from knowledge_hub.ai import evidence_assembly
from knowledge_hub.interfaces.cli.commands import search_cmd
from knowledge_hub.papers.parsed_artifact_evidence_chunk_real_answer_quality_smoke import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_REAL_ANSWER_QUALITY_SMOKE_SCHEMA_ID,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_ROUTE_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-opt-in-route-review.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_answer_path_opt_in_route_review_ready"
BLOCKED_DECISION = "parsed_artifact_evidence_chunk_answer_path_opt_in_route_review_blocked"
NEXT_TRANCHE_READY = "parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design"
NEXT_TRANCHE_BLOCKED = "parsed_artifact_evidence_chunk_answer_path_opt_in_route_review_repair"
DEFAULT_ANSWER_QUALITY_SMOKE_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_real_answer_quality_smoke.v1.json"
)
ZERO_COUNTER_FIELDS = (
    "runtimeRouteWriteRows",
    "publicCliFlagRows",
    "defaultOnRows",
    "answerVisibleDefaultRows",
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


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


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


def _row(
    *,
    row_id: str,
    route_layer: str,
    file_ref: str,
    expected: str,
    observed: bool,
    required_for_current_review: bool = True,
    gap_reason: str = "",
    recommendation: str = "",
) -> dict[str, Any]:
    status = "pass" if observed else ("gap" if not required_for_current_review else "fail")
    return {
        "rowId": row_id,
        "routeLayer": route_layer,
        "fileRef": file_ref,
        "expected": expected,
        "observed": bool(observed),
        "requiredForCurrentReview": bool(required_for_current_review),
        "status": status,
        "gapReason": _clean_text(gap_reason),
        "recommendation": _clean_text(recommendation),
    }


def _answer_quality_gate(answer_quality_smoke_report: str | Path) -> tuple[dict[str, Any], list[str]]:
    payload = _read_json(answer_quality_smoke_report)
    counts = dict(payload.get("counts") or {})
    violations: list[str] = []
    if payload.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_REAL_ANSWER_QUALITY_SMOKE_SCHEMA_ID:
        violations.append("answer_quality_smoke_schema_mismatch")
    if payload.get("status") != "ready":
        violations.append("answer_quality_smoke_not_ready")
    if _int(counts.get("passRows")) <= 0:
        violations.append("answer_quality_smoke_has_no_pass_rows")
    if _int(counts.get("failRows")) != 0 or _int(counts.get("blockedRows")) != 0:
        violations.append("answer_quality_smoke_has_failed_or_blocked_rows")
    if _int(counts.get("privatePathLeakRows")) != 0:
        violations.append("answer_quality_smoke_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        violations.append("answer_quality_smoke_schema_violations_present")
    return payload, sorted(set(violations))


def _route_rows() -> list[dict[str, Any]]:
    return [
        _row(
            row_id="answer_runtime_request_query_plan",
            route_layer="runtime_request",
            file_ref="knowledge_hub/ai/rag_answer_runtime.py",
            expected="AnswerRuntimeRequest and build_request accept query_plan.",
            observed=_signature_has_param(rag_answer_runtime.AnswerRuntimeRequest, "query_plan")
            and _signature_has_param(rag_answer_runtime.RAGAnswerRuntime.build_request, "query_plan"),
        ),
        _row(
            row_id="runtime_wrapper_query_plan",
            route_layer="runtime_wrapper",
            file_ref="knowledge_hub/ai/rag_answer_runtime.py",
            expected="generate_answer and stream_answer wrappers accept query_plan.",
            observed=_signature_has_param(rag_answer_runtime.generate_answer, "query_plan")
            and _signature_has_param(rag_answer_runtime.stream_answer, "query_plan"),
        ),
        _row(
            row_id="legacy_execution_passes_query_plan_to_evidence_assembly",
            route_layer="legacy_runtime_execution",
            file_ref="knowledge_hub/ai/rag_answer_runtime.py",
            expected="_build_non_ask_v2_execution passes request.query_plan into RetrievalPipelineService and EvidenceAssemblyService.",
            observed=_source_contains(
                rag_answer_runtime.RAGAnswerRuntime._build_non_ask_v2_execution,
                "query_plan=request.query_plan",
                "EvidenceAssemblyService.from_searcher",
            ),
        ),
        _row(
            row_id="ask_v2_execution_passes_query_plan_to_evidence_assembly",
            route_layer="ask_v2_runtime_execution",
            file_ref="knowledge_hub/ai/ask_v2.py",
            expected="AskV2 execution passes normalized query_plan_payload into EvidenceAssemblyService.",
            observed=_source_contains(ask_v2.AskV2Service.execute, "query_plan_payload", "EvidenceAssemblyService.from_searcher")
            and _source_contains(ask_v2.AskV2Service.execute, "query_plan=query_plan_payload"),
        ),
        _row(
            row_id="evidence_assembly_adapter_wired",
            route_layer="evidence_assembly",
            file_ref="knowledge_hub/ai/evidence_assembly.py",
            expected="EvidenceAssemblyService invokes collect_parsed_artifact_evidence_chunk_runtime_evidence.",
            observed=_source_contains(
                evidence_assembly.EvidenceAssemblyService.assemble,
                "collect_parsed_artifact_evidence_chunk_runtime_evidence",
                "parsedArtifactEvidenceChunkAdapter",
            ),
        ),
        _row(
            row_id="answer_payload_exposes_contracts",
            route_layer="answer_payload",
            file_ref="knowledge_hub/ai/answer_payload_builder.py",
            expected="Answer payload exposes queryPlan, evidencePacket, and evidencePacketContract.",
            observed=_source_contains(
                answer_payload_builder.AnswerPayloadBuilder.base_payload,
                '"queryPlan"',
                '"evidencePacket"',
                '"evidencePacketContract"',
            ),
        ),
        _row(
            row_id="public_searcher_generate_answer_query_plan_ingress",
            route_layer="public_searcher_api",
            file_ref="knowledge_hub/ai/rag.py",
            expected="Public RAGSearcher.generate_answer exposes an internal query_plan ingress for opt-in route tests.",
            observed=_signature_has_param(rag_module.RAGSearcher.generate_answer, "query_plan"),
            required_for_current_review=False,
            gap_reason="public_searcher_generate_answer_does_not_accept_query_plan",
            recommendation="Add an internal/labs-only opt-in ingress before attempting end-to-end khub ask route smoke.",
        ),
        _row(
            row_id="khub_ask_has_no_public_strict_chunk_flag",
            route_layer="public_cli",
            file_ref="knowledge_hub/interfaces/cli/commands/search_cmd.py",
            expected="khub ask has no public parsed-artifact evidence chunk flag in this tranche.",
            observed=not _source_contains(search_cmd.ask, "--parsed-artifact-evidence-chunk", "parsed_artifact_evidence_chunk"),
            required_for_current_review=True,
        ),
    ]


def build_parsed_artifact_evidence_chunk_answer_path_opt_in_route_review(
    *,
    answer_quality_smoke_report: str | Path = DEFAULT_ANSWER_QUALITY_SMOKE_REPORT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    smoke_payload, smoke_violations = _answer_quality_gate(answer_quality_smoke_report)
    rows = _route_rows()
    required_fail_rows = [row for row in rows if row.get("requiredForCurrentReview") and row.get("status") != "pass"]
    gap_rows = [row for row in rows if row.get("status") == "gap"]
    public_searcher_gap = any(
        row.get("rowId") == "public_searcher_generate_answer_query_plan_ingress" and row.get("status") == "gap"
        for row in rows
    )
    private_path_leak_rows = sum(1 for row in rows if _contains_private_path(row))
    semantic_violations = list(smoke_violations)
    semantic_violations.extend(str(row.get("rowId")) for row in required_fail_rows)
    if private_path_leak_rows:
        semantic_violations.append("private_path_leak")
    counts = {
        "inputAnswerQualitySmokeRows": 1,
        "answerQualitySmokePassRows": _int(dict(smoke_payload.get("counts") or {}).get("passRows")),
        "routeReviewRows": len(rows),
        "routeReviewPassRows": sum(1 for row in rows if row.get("status") == "pass"),
        "routeReviewGapRows": len(gap_rows),
        "routeReviewFailRows": len(required_fail_rows),
        "requiredRouteRows": sum(1 for row in rows if row.get("requiredForCurrentReview")),
        "optionalGapRows": len(gap_rows),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(set(semantic_violations)),
    }
    status = "ready" if not semantic_violations else "blocked"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_ROUTE_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "answerQualitySmokeReportRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_real_answer_quality_smoke.v1.json",
            "answerQualitySmokeSchema": _clean_text(smoke_payload.get("schema")),
            "answerQualitySmokeStatus": _clean_text(smoke_payload.get("status")),
        },
        "policy": {
            "reportOnly": True,
            "routeReviewOnly": True,
            "defaultAskPathChanged": False,
            "publicCliFlagAdded": False,
            "runtimeRouteWrite": False,
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
            "readyForOptInImplementationDesign": status == "ready",
            "answerQualitySmokeReady": not smoke_violations,
            "internalRuntimeQueryPlanIngressReady": all(
                row.get("status") == "pass"
                for row in rows
                if row.get("rowId")
                in {
                    "answer_runtime_request_query_plan",
                    "runtime_wrapper_query_plan",
                    "legacy_execution_passes_query_plan_to_evidence_assembly",
                    "ask_v2_execution_passes_query_plan_to_evidence_assembly",
                    "evidence_assembly_adapter_wired",
                    "answer_payload_exposes_contracts",
                }
            ),
            "publicSearcherIngressGap": public_searcher_gap,
            "publicCliDefaultUnchanged": any(
                row.get("rowId") == "khub_ask_has_no_public_strict_chunk_flag" and row.get("status") == "pass"
                for row in rows
            ),
            "semanticViolations": sorted(set(semantic_violations)),
        },
        "implementationNotes": [
            "internal_runtime_can_accept_query_plan_opt_in",
            "public_rag_searcher_generate_answer_lacks_query_plan_ingress"
            if public_searcher_gap
            else "public_rag_searcher_generate_answer_has_query_plan_ingress",
            "keep_khub_ask_public_flag_out_until_labs_internal_route_is_verified",
        ],
        "rows": rows,
        "warnings": [
            "route_review_only_no_runtime_behavior_change",
            "public_searcher_query_plan_ingress_gap_is_expected_input_for_next_tranche"
            if public_searcher_gap
            else "public_searcher_query_plan_ingress_present_ready_for_opt_in_smoke",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_opt_in_route_review_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path Opt-in Route Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- answerQualitySmokePassRows: `{counts.get('answerQualitySmokePassRows')}`",
        f"- routeReviewRows: `{counts.get('routeReviewRows')}`",
        f"- routeReviewPassRows: `{counts.get('routeReviewPassRows')}`",
        f"- routeReviewGapRows: `{counts.get('routeReviewGapRows')}`",
        f"- routeReviewFailRows: `{counts.get('routeReviewFailRows')}`",
        f"- publicSearcherIngressGap: `{gate.get('publicSearcherIngressGap')}`",
        f"- publicCliDefaultUnchanged: `{gate.get('publicCliDefaultUnchanged')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Policy",
        "",
        "Report-only route review. It inspects the current answer path and records where the opt-in can flow next; it does not change runtime behavior, add public CLI flags, generate answers, call LLMs, mutate stores, or scan the vault.",
        "",
        "## Route Rows",
        "",
    ]
    for row in list(report.get("rows") or []):
        lines.extend(
            [
                f"### `{row.get('rowId')}`",
                "",
                f"- layer: `{row.get('routeLayer')}`",
                f"- status: `{row.get('status')}`",
                f"- file: `{row.get('fileRef')}`",
                f"- observed: `{row.get('observed')}`",
                f"- gapReason: `{row.get('gapReason')}`",
                f"- recommendation: `{row.get('recommendation')}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_opt_in_route_review(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_opt_in_route_review_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_ROUTE_REVIEW_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_opt_in_route_review",
    "write_parsed_artifact_evidence_chunk_answer_path_opt_in_route_review",
]
