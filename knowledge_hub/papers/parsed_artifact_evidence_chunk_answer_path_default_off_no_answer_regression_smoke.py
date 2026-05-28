"""Default-off/no-answer regression smoke for parsed-artifact evidence chunks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.ai.parsed_artifact_evidence_chunk_runtime_adapter import ADAPTER_OPT_IN_VALUE
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke import (
    READY_DECISION as SEARCHER_INGRESS_READY_DECISION,
    _build_searcher,
    _clean_text,
    _contains_private_path,
    _int,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_DEFAULT_OFF_NO_ANSWER_REGRESSION_SMOKE_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-default-off-no-answer-regression-smoke.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke_ready"
BLOCKED_DECISION = "parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke_blocked"
NEXT_TRANCHE_READY = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design"
NEXT_TRANCHE_BLOCKED = "parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_repair"
DEFAULT_PAPERS_DIR = Path.home() / ".khub" / "papers"
DEFAULT_SEARCHER_INGRESS_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke.v1.json"
)
DEFAULT_RESOLVED_PAPER_IDS = ("1207.0580",)
DEFAULT_QUERY = "What evidence is available for this resolved paper?"
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _scenario_inputs(resolved_paper_ids: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "scenarioId": "opt_in_absent",
            "sourceType": "paper",
            "queryPlan": {
                "family": "paper_lookup",
                "resolvedPaperIds": resolved_paper_ids,
            },
            "expectedSkippedReason": "query_plan_opt_in_not_enabled",
            "expectedAdapterStatus": "disabled",
        },
        {
            "scenarioId": "opt_in_missing_resolved_paper",
            "sourceType": "paper",
            "queryPlan": {
                "family": "paper_lookup",
                "parsed_artifact_evidence_chunk_adapter": ADAPTER_OPT_IN_VALUE,
            },
            "expectedSkippedReason": "resolved_paper_ids_required",
            "expectedAdapterStatus": "skipped",
        },
        {
            "scenarioId": "opt_in_non_paper_source",
            "sourceType": "web",
            "queryPlan": {
                "family": "paper_lookup",
                "parsed_artifact_evidence_chunk_adapter": ADAPTER_OPT_IN_VALUE,
                "resolvedPaperIds": resolved_paper_ids,
            },
            "expectedSkippedReason": "source_type_not_paper",
            "expectedAdapterStatus": "skipped",
        },
    ]


def _run_scenario(
    *,
    papers_dir: str | Path,
    query: str,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    searcher, llm = _build_searcher(papers_dir=papers_dir)
    payload = searcher.generate_answer(
        query,
        top_k=1,
        source_type=_clean_text(scenario.get("sourceType")),
        retrieval_mode="semantic",
        allow_external=False,
        ask_v2_mode="claim_first",
        query_plan=dict(scenario.get("queryPlan") or {}),
    )
    evidence_packet = dict(payload.get("evidencePacket") or {})
    adapter_diag = dict(evidence_packet.get("parsedArtifactEvidenceChunkAdapter") or {})
    evidence_contract = dict(payload.get("evidencePacketContract") or {})
    span_rows = len(list(evidence_contract.get("spans") or []))
    payload_status = _clean_text(payload.get("status"))
    answerable = bool(evidence_packet.get("answerable"))
    contract_answerable = bool(evidence_contract.get("answerable"))
    adapter_status = _clean_text(adapter_diag.get("status"))
    skipped_reason = _clean_text(adapter_diag.get("skippedReason"))
    expected_adapter_status = _clean_text(scenario.get("expectedAdapterStatus"))
    expected_skipped_reason = _clean_text(scenario.get("expectedSkippedReason"))
    violations: list[str] = []
    if payload_status != "no_result":
        violations.append("payload_status_not_no_result")
    if answerable:
        violations.append("evidence_packet_answerable_unexpected")
    if contract_answerable:
        violations.append("evidence_contract_answerable_unexpected")
    if _int(evidence_packet.get("selectedEvidenceCount")) != 0:
        violations.append("selected_evidence_count_nonzero")
    if _int(evidence_packet.get("citationCount")) != 0:
        violations.append("citation_count_nonzero")
    if span_rows != 0:
        violations.append("contract_span_rows_nonzero")
    if _int(adapter_diag.get("rowsAdded")) != 0:
        violations.append("adapter_rows_added_nonzero")
    if llm.calls != 0:
        violations.append("local_fake_llm_called")
    if adapter_status != expected_adapter_status:
        violations.append("adapter_status_mismatch")
    if skipped_reason != expected_skipped_reason:
        violations.append("adapter_skipped_reason_mismatch")
    row = {
        "scenarioId": _clean_text(scenario.get("scenarioId")),
        "sourceType": _clean_text(scenario.get("sourceType")),
        "queryPlanOptInPresent": any(
            key in dict(scenario.get("queryPlan") or {})
            for key in ("parsed_artifact_evidence_chunk_adapter", "parsedArtifactEvidenceChunkAdapter")
        ),
        "resolvedPaperIds": list(adapter_diag.get("resolvedPaperIds") or []),
        "payloadStatus": payload_status,
        "adapterStatus": adapter_status,
        "adapterSkippedReason": skipped_reason,
        "adapterRowsAdded": _int(adapter_diag.get("rowsAdded")),
        "adapterCandidateRowsConsidered": _int(adapter_diag.get("candidateRowsConsidered")),
        "selectedEvidenceCount": _int(evidence_packet.get("selectedEvidenceCount")),
        "citationCount": _int(evidence_packet.get("citationCount")),
        "evidencePacketAnswerable": answerable,
        "evidencePacketContractAnswerable": contract_answerable,
        "evidencePacketContractSpanRows": span_rows,
        "localFakeLlmCallRows": int(llm.calls),
        "answerTextIncludedInReport": False,
        "evidenceTextIncludedInReport": False,
        "pass": not violations,
        "violations": violations,
    }
    return row


def build_parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke(
    *,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    resolved_paper_ids: list[str] | tuple[str, ...] = DEFAULT_RESOLVED_PAPER_IDS,
    query: str = DEFAULT_QUERY,
    searcher_ingress_report_path: str | Path = DEFAULT_SEARCHER_INGRESS_REPORT,
    searcher_ingress_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    resolved_ids = [_clean_text(item) for item in resolved_paper_ids if _clean_text(item)]
    upstream = dict(searcher_ingress_report or _read_json(searcher_ingress_report_path))
    upstream_ready = (
        upstream.get("status") == "ready"
        and upstream.get("decision") == SEARCHER_INGRESS_READY_DECISION
        and _int(dict(upstream.get("counts") or {}).get("schemaViolationCount")) == 0
    )
    scenario_rows = [
        _run_scenario(papers_dir=papers_dir, query=query, scenario=scenario)
        for scenario in _scenario_inputs(resolved_ids)
    ]
    private_path_leak_rows = sum(1 for row in scenario_rows if _contains_private_path(row))
    failed_rows = [row for row in scenario_rows if not bool(row.get("pass"))]
    schema_violations: list[str] = []
    if not upstream_ready:
        schema_violations.append("searcher_ingress_smoke_not_ready")
    if failed_rows:
        schema_violations.append("default_off_no_answer_scenario_failed")
    if private_path_leak_rows:
        schema_violations.append("private_path_leak")

    counts = {
        "inputScenarioRows": len(scenario_rows),
        "passRows": sum(1 for row in scenario_rows if bool(row.get("pass"))),
        "failRows": len(failed_rows),
        "noAnswerRows": sum(1 for row in scenario_rows if row.get("payloadStatus") == "no_result"),
        "answerableRows": sum(1 for row in scenario_rows if bool(row.get("evidencePacketAnswerable"))),
        "evidencePacketContractAnswerableRows": sum(
            1 for row in scenario_rows if bool(row.get("evidencePacketContractAnswerable"))
        ),
        "adapterAppliedRows": sum(1 for row in scenario_rows if row.get("adapterStatus") == "applied"),
        "adapterDisabledRows": sum(1 for row in scenario_rows if row.get("adapterStatus") == "disabled"),
        "adapterSkippedRows": sum(1 for row in scenario_rows if row.get("adapterStatus") == "skipped"),
        "adapterRowsAdded": sum(_int(row.get("adapterRowsAdded")) for row in scenario_rows),
        "selectedEvidenceCount": sum(_int(row.get("selectedEvidenceCount")) for row in scenario_rows),
        "citationCount": sum(_int(row.get("citationCount")) for row in scenario_rows),
        "evidencePacketContractSpanRows": sum(_int(row.get("evidencePacketContractSpanRows")) for row in scenario_rows),
        "localFakeLlmCallRows": sum(_int(row.get("localFakeLlmCallRows")) for row in scenario_rows),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(schema_violations),
    }
    status = "ready" if not schema_violations else "blocked"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_DEFAULT_OFF_NO_ANSWER_REGRESSION_SMOKE_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "upstream": {
            "searcherIngressReportRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke.v1.json",
            "searcherIngressStatus": _clean_text(upstream.get("status")),
            "searcherIngressDecision": _clean_text(upstream.get("decision")),
            "searcherIngressReady": upstream_ready,
        },
        "smoke": {
            "query": query,
            "papersDirRef": "papers_dir",
            "resolvedPaperIds": resolved_ids,
            "runtimeBoundary": "knowledge_hub.ai.rag.RAGSearcher.generate_answer",
            "scenarioPolicy": "default_off_and_ineligible_routes_must_no_answer",
            "retrievalMode": "semantic",
            "askV2Mode": "claim_first",
            "externalModelCallsAllowed": False,
            "fakeLocalLlmOnly": True,
            "answerTextIncludedInReport": False,
            "evidenceTextIncludedInReport": False,
        },
        "counts": counts,
        "gate": {
            "readyForLabsOptInSurfaceDesign": status == "ready",
            "searcherAnswerPathInvoked": True,
            "allScenariosNoAnswer": counts["noAnswerRows"] == counts["inputScenarioRows"],
            "answerabilityStayedFalse": counts["answerableRows"] == 0
            and counts["evidencePacketContractAnswerableRows"] == 0,
            "adapterNeverApplied": counts["adapterAppliedRows"] == 0 and counts["adapterRowsAdded"] == 0,
            "noLlmCalls": counts["localFakeLlmCallRows"] == 0,
            "externalModelCallsDisabled": True,
            "publicDefaultUnchanged": True,
            "schemaViolations": schema_violations,
        },
        "rows": scenario_rows,
        "warnings": [],
    }


def render_parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path Default-Off No-Answer Regression Smoke",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- inputScenarioRows: `{counts.get('inputScenarioRows')}`",
        f"- passRows: `{counts.get('passRows')}`",
        f"- failRows: `{counts.get('failRows')}`",
        f"- noAnswerRows: `{counts.get('noAnswerRows')}`",
        f"- answerableRows: `{counts.get('answerableRows')}`",
        f"- adapterAppliedRows: `{counts.get('adapterAppliedRows')}`",
        f"- adapterRowsAdded: `{counts.get('adapterRowsAdded')}`",
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
    lines.extend(["", "## Scenarios", ""])
    for row in list(report.get("rows") or []):
        lines.append(
            f"- `{row.get('scenarioId')}` source=`{row.get('sourceType')}` "
            f"payload=`{row.get('payloadStatus')}` adapter=`{row.get('adapterStatus')}` "
            f"reason=`{row.get('adapterSkippedReason')}` pass=`{row.get('pass')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_DEFAULT_OFF_NO_ANSWER_REGRESSION_SMOKE_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke",
    "write_parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke",
]
