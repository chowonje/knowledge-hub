"""Runtime adapter design for parsed-artifact evidence chunk candidates.

This report consumes the runtime contract review and fixes the future adapter
boundary for section/paragraph evidence chunks. It intentionally does not
change ``EvidenceAssemblyService`` or make candidate rows answer-visible.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.ai.answer_contracts import parse_span_offsets
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    _contains_private_path,
    normalize_text,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_runtime_contract_review import (
    CONTRACT_STATUS_READY_CANDIDATE_ONLY,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_CONTRACT_REVIEW_SCHEMA_ID,
    READY_DECISION as CONTRACT_REVIEW_READY_DECISION,
    load_json,
    sanitized_report_ref,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-runtime-adapter-design.v1"
)

ADAPTER_ID = "parsed_artifact_evidence_chunk_runtime_adapter_v1"
ADAPTER_MODE = "opt_in_runtime_design"
INTEGRATION_BOUNDARY = "knowledge_hub.ai.evidence_assembly.EvidenceAssemblyService.assemble"
INSERTION_POINT = (
    "after _reselect_top1_if_needed and before _build_citations, answer_signals, "
    "evaluate_answerability, context assembly, and answer payload construction"
)
READY_DECISION = "parsed_artifact_evidence_chunk_candidate_runtime_adapter_design_ready"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE = "parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run"

ADAPTER_STATUS_READY_CANDIDATE_ONLY = "runtime_adapter_design_ready_candidate_only"
ADAPTER_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
ADAPTER_STATUS_BLOCKED_CONTRACT_NOT_READY = "blocked_contract_review_not_ready"
ADAPTER_STATUS_BLOCKED_MISSING_REQUIRED_FIELD = "blocked_missing_required_adapter_field"
ADAPTER_STATUS_BLOCKED_INVALID_LOCATOR = "blocked_invalid_locator"
ADAPTER_STATUS_BLOCKED_POLICY_QUARANTINE = "blocked_policy_quarantine_violation"
ADAPTER_STATUS_BLOCKED_PRIVATE_PATH = "blocked_private_path_leak"

ALLOWED_ARTIFACT_TYPES = {"section", "paragraph"}
ZERO_COUNTER_FIELDS = (
    "candidateStoreWriteRows",
    "sourceSpanCreatedRows",
    "strictEvidenceRows",
    "citationGradeRows",
    "runtimeEvidenceRows",
    "answerVisibleRows",
    "answerGenerationRows",
    "databaseMutationRows",
    "indexMutationRows",
    "reindexOrReembedRows",
    "parserExecutionRows",
    "canonicalParsedArtifactWriteRows",
    "vaultScanRows",
    "externalDownloadRows",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _source_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_CONTRACT_REVIEW_SCHEMA_ID:
        blockers.append("invalid_runtime_contract_review_schema")
    if report.get("status") != "ready":
        blockers.append("runtime_contract_review_not_ready")
    if report.get("decision") != CONTRACT_REVIEW_READY_DECISION:
        blockers.append("runtime_contract_review_invalid_decision")
    if gate.get("readyForRuntimeAdapterDesign") is not True:
        blockers.append("runtime_contract_review_gate_not_ready_for_adapter_design")
    input_rows = _int(counts.get("inputRows"))
    if input_rows <= 0:
        blockers.append("runtime_contract_review_has_no_input_rows")
    if _int(counts.get("contractReviewReadyRows")) != input_rows:
        blockers.append("contract_review_ready_count_mismatch")
    if _int(counts.get("runtimeAdapterDesignCandidateRows")) != input_rows:
        blockers.append("runtime_adapter_design_candidate_count_mismatch")
    for field_name in (
        "answerableRows",
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
        *ZERO_COUNTER_FIELDS,
    ):
        if _int(counts.get(field_name)) != 0:
            blockers.append(f"runtime_contract_review_has_{field_name}")
    if _contains_private_path(report):
        blockers.append("runtime_contract_review_has_private_path_leak")
    return sorted(set(blockers))


def _source_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "nextRecommendedTranche": normalize_text(report.get("nextRecommendedTranche")),
        "reportRef": normalize_text(report_ref),
        "inputRows": _int(counts.get("inputRows")),
        "contractReviewReadyRows": _int(counts.get("contractReviewReadyRows")),
        "runtimeAdapterDesignCandidateRows": _int(counts.get("runtimeAdapterDesignCandidateRows")),
        "answerableRows": _int(counts.get("answerableRows")),
        "blockedRows": _int(counts.get("blockedRows")),
        "privatePathLeakRows": _int(counts.get("privatePathLeakRows")),
        "schemaViolationCount": _int(counts.get("schemaViolationCount")),
    }


def _row_blockers(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if normalize_text(row.get("contractReviewStatus")) != CONTRACT_STATUS_READY_CANDIDATE_ONLY:
        blockers.append("contract_review_status_not_ready_candidate_only")
    for flag_name in (
        "contractReviewReady",
        "runtimeAdapterDesignCandidate",
        "evidencePacketSpanCompatible",
        "answerContractCitationShapeCompatible",
        "strictProvenanceShapePresent",
    ):
        if row.get(flag_name) is not True:
            blockers.append(f"{flag_name}_not_true")
    for field_name in (
        "contractReviewRowId",
        "candidateRecordId",
        "paperId",
        "artifactType",
        "sourceRef",
        "sourceContentHash",
        "spanLocator",
        "snippetHash",
        "candidateStoreRef",
    ):
        if not normalize_text(row.get(field_name)):
            blockers.append(f"{field_name}_missing")
    if normalize_text(row.get("artifactType")) not in ALLOWED_ARTIFACT_TYPES:
        blockers.append("unsupported_artifact_type")
    span_locator = normalize_text(row.get("spanLocator"))
    parsed_start, parsed_end = parse_span_offsets(span_locator, span_locator)
    if parsed_start is None or parsed_end is None:
        blockers.append("chars_locator_missing_or_invalid")
    if _int(row.get("charStart")) != parsed_start:
        blockers.append("char_start_mismatch")
    if _int(row.get("charEnd")) != parsed_end:
        blockers.append("char_end_mismatch")
    for flag_name in ("runtimeVisibilityAllowed", "answerVisibleAllowed", "answerableAllowed"):
        if row.get(flag_name) is not False:
            blockers.append(f"{flag_name}_not_false")
    for flag_name in ("strictEvidence", "citationGrade", "runtimeEvidence", "answerVisible", "answerable"):
        if row.get(flag_name) is not False:
            blockers.append(f"{flag_name}_not_false")
    if row.get("candidateOnly") is not True:
        blockers.append("candidateOnly_not_true")
    span = dict(row.get("plannedEvidencePacketSpan") or {})
    derivative = dict(span.get("derivativeSource") or {})
    if normalize_text(derivative.get("candidateRecordId")) != normalize_text(row.get("candidateRecordId")):
        blockers.append("planned_span_derivative_candidate_record_id_mismatch")
    if normalize_text(derivative.get("candidateStoreRef")) != normalize_text(row.get("candidateStoreRef")):
        blockers.append("planned_span_derivative_candidate_store_ref_mismatch")
    if _contains_private_path(row):
        blockers.append("private_path_leak")
    return sorted(set(blockers))


def _adapter_status(blockers: list[str], source_blockers: list[str]) -> str:
    if source_blockers:
        return ADAPTER_STATUS_BLOCKED_INPUT_SCHEMA
    if not blockers:
        return ADAPTER_STATUS_READY_CANDIDATE_ONLY
    if any(item.startswith("contractReview") or item.startswith("contract_review") for item in blockers):
        return ADAPTER_STATUS_BLOCKED_CONTRACT_NOT_READY
    if any("locator" in item or item in {"char_start_mismatch", "char_end_mismatch"} for item in blockers):
        return ADAPTER_STATUS_BLOCKED_INVALID_LOCATOR
    if "private_path_leak" in blockers:
        return ADAPTER_STATUS_BLOCKED_PRIVATE_PATH
    if any(item.endswith("_not_false") or item.endswith("_not_true") for item in blockers):
        return ADAPTER_STATUS_BLOCKED_POLICY_QUARANTINE
    return ADAPTER_STATUS_BLOCKED_MISSING_REQUIRED_FIELD


def _adapter_design() -> dict[str, Any]:
    return {
        "adapterId": ADAPTER_ID,
        "adapterMode": ADAPTER_MODE,
        "integrationBoundary": INTEGRATION_BOUNDARY,
        "insertionPoint": INSERTION_POINT,
        "activationPolicy": {
            "sourceTypeRequired": "paper",
            "queryPlanOptInKeys": [
                "parsed_artifact_evidence_chunk_adapter",
                "parsedArtifactEvidenceChunkAdapter",
            ],
            "queryPlanOptInValue": "runtime_v1",
            "resolvedPaperIdsRequired": True,
            "fallbackToAllPapersAllowed": False,
            "defaultEnabled": False,
        },
        "selectionPolicy": {
            "maxRowsPerResolvedPaper": 2,
            "maxRowsTotal": 4,
            "order": ["resolved_paper_order", "candidate_store_readback_order"],
            "allowedArtifactTypes": sorted(ALLOWED_ARTIFACT_TYPES),
            "requiresSourceContentHash": True,
            "requiresCharsLocator": True,
            "requiresSnippetHash": True,
            "requiresExcerptReadback": True,
        },
        "candidateStoreRead": {
            "storeRefTemplate": "papers_dir/structured_evidence_candidates/evidence_chunk/{paperId}.jsonl",
            "readResolvedPaperIdsOnly": True,
            "recordSchema": "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-record.v1",
            "noVaultScan": True,
            "noParserExecution": True,
            "noExternalDownload": True,
        },
        "futureEvidenceItemShape": {
            "source_type": "paper",
            "sourceType": "paper",
            "source_id": "{paperId}",
            "sourceId": "{paperId}",
            "source_ref": "{sourceRef}",
            "sourceRef": "{sourceRef}",
            "source_content_hash": "{sourceContentHash}",
            "sourceContentHash": "{sourceContentHash}",
            "span_locator": "chars:{start}-{end}",
            "spanLocator": "chars:{start}-{end}",
            "char_start": "{charStart}",
            "charStart": "{charStart}",
            "char_end": "{charEnd}",
            "charEnd": "{charEnd}",
            "excerpt": "{candidateStore.excerpt}",
            "excerptSource": "candidate_store_readback",
            "snippet_hash": "{snippetHash}",
            "snippetHash": "{snippetHash}",
            "content_hash": "{snippetHash}",
            "evidence_kind": "parsed_artifact_evidence_chunk",
            "evidenceKind": "parsed_artifact_evidence_chunk",
            "derivative_source": {
                "candidateRecordId": "{candidateRecordId}",
                "candidateStoreRef": "{candidateStoreRef}",
                "artifactType": "{artifactType}",
            },
            "derivativeSource": {
                "candidateRecordId": "{candidateRecordId}",
                "candidateStoreRef": "{candidateStoreRef}",
                "artifactType": "{artifactType}",
            },
        },
        "futureDiagnosticsShape": {
            "evidencePacketKey": "parsedArtifactEvidenceChunkAdapter",
            "fields": [
                "enabled",
                "status",
                "resolvedPaperIds",
                "candidateRowsConsidered",
                "rowsAdded",
                "blockedRows",
                "skippedReason",
            ],
        },
        "answerabilityPolicy": {
            "currentTrancheAnswerabilityAllowed": False,
            "futureAdapterMayAffectAnswerabilityOnlyAfterDryRunAndImplementationGate": True,
            "legacyRetrievalBypassAllowed": False,
        },
    }


def _planned_evidence_item_shape(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_type": "paper",
        "sourceType": "paper",
        "source_id": normalize_text(row.get("paperId")),
        "sourceId": normalize_text(row.get("paperId")),
        "source_ref": normalize_text(row.get("sourceRef")),
        "sourceRef": normalize_text(row.get("sourceRef")),
        "source_content_hash": normalize_text(row.get("sourceContentHash")),
        "sourceContentHash": normalize_text(row.get("sourceContentHash")),
        "span_locator": normalize_text(row.get("spanLocator")),
        "spanLocator": normalize_text(row.get("spanLocator")),
        "char_start": _int(row.get("charStart")),
        "charStart": _int(row.get("charStart")),
        "char_end": _int(row.get("charEnd")),
        "charEnd": _int(row.get("charEnd")),
        "excerptSource": "candidate_store_readback",
        "excerptIncludedInThisReport": False,
        "snippet_hash": normalize_text(row.get("snippetHash")),
        "snippetHash": normalize_text(row.get("snippetHash")),
        "content_hash": normalize_text(row.get("snippetHash")),
        "evidence_kind": "parsed_artifact_evidence_chunk",
        "evidenceKind": "parsed_artifact_evidence_chunk",
        "derivative_source": {
            "candidateRecordId": normalize_text(row.get("candidateRecordId")),
            "candidateStoreRef": normalize_text(row.get("candidateStoreRef")),
            "artifactType": normalize_text(row.get("artifactType")),
        },
        "derivativeSource": {
            "candidateRecordId": normalize_text(row.get("candidateRecordId")),
            "candidateStoreRef": normalize_text(row.get("candidateStoreRef")),
            "artifactType": normalize_text(row.get("artifactType")),
        },
    }


def build_parsed_artifact_evidence_chunk_candidate_runtime_adapter_design(
    *,
    runtime_contract_review_report: dict[str, Any],
    source_runtime_contract_review_report_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_rows = [
        dict(row)
        for row in list(runtime_contract_review_report.get("rows") or [])
        if isinstance(row, dict)
    ]
    source_blockers = _source_blockers(runtime_contract_review_report)
    rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(source_rows, start=1):
        row_blockers = _row_blockers(source_row)
        status = _adapter_status(row_blockers, source_blockers)
        blockers = sorted(set([*source_blockers, *row_blockers]))
        ready = status == ADAPTER_STATUS_READY_CANDIDATE_ONLY and not blockers
        rows.append(
            {
                "adapterDesignRowId": f"parsed-artifact-evidence-chunk-candidate-runtime-adapter-design:{index:04d}",
                "sourceContractReviewRowId": normalize_text(source_row.get("contractReviewRowId")),
                "candidateRecordId": normalize_text(source_row.get("candidateRecordId")),
                "sourceCandidateRowId": normalize_text(source_row.get("sourceCandidateRowId")),
                "paperId": normalize_text(source_row.get("paperId")),
                "artifactType": normalize_text(source_row.get("artifactType")),
                "sourceRef": normalize_text(source_row.get("sourceRef")),
                "sourceContentHash": normalize_text(source_row.get("sourceContentHash")),
                "spanLocator": normalize_text(source_row.get("spanLocator")),
                "charStart": source_row.get("charStart"),
                "charEnd": source_row.get("charEnd"),
                "snippetHash": normalize_text(source_row.get("snippetHash")),
                "candidateStoreRef": normalize_text(source_row.get("candidateStoreRef")),
                "adapterDesignStatus": status,
                "adapterDesignBlockers": blockers,
                "adapterDesignReady": ready,
                "futureAdapterCandidate": ready,
                "futureCandidateStoreReadRequired": ready,
                "futureRuntimeEvidenceAllowed": False,
                "futureAnswerVisibleAllowed": False,
                "futureAnswerabilityAllowed": False,
                "candidateOnly": True,
                "strictEvidence": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "answerVisible": False,
                "answerable": False,
                "plannedAdapterInputs": {
                    "adapterId": ADAPTER_ID,
                    "adapterMode": ADAPTER_MODE,
                    "sourceType": "paper",
                    "candidateStoreRef": normalize_text(source_row.get("candidateStoreRef")),
                    "queryPlanOptInValue": "runtime_v1",
                    "requiresResolvedPaperIds": True,
                    "fallbackToAllPapersAllowed": False,
                },
                "plannedEvidenceItemShape": _planned_evidence_item_shape(source_row),
                "plannedDiagnosticsShape": {
                    "evidencePacketKey": "parsedArtifactEvidenceChunkAdapter",
                    "enabled": "{bool}",
                    "status": "{applied|skipped|blocked}",
                    "resolvedPaperIds": "{list[str]}",
                    "candidateRowsConsidered": "{int}",
                    "rowsAdded": "{int}",
                    "blockedRows": "{int}",
                    "skippedReason": "{string}",
                },
                "checks": {
                    "contractReviewReady": normalize_text(source_row.get("contractReviewStatus")) == CONTRACT_STATUS_READY_CANDIDATE_ONLY
                    and source_row.get("contractReviewReady") is True,
                    "evidencePacketSpanCompatible": source_row.get("evidencePacketSpanCompatible") is True,
                    "answerContractCitationShapeCompatible": source_row.get("answerContractCitationShapeCompatible") is True,
                    "strictProvenanceShapePresent": source_row.get("strictProvenanceShapePresent") is True,
                    "policyQuarantineMaintained": not any(item.endswith("_not_false") or item.endswith("_not_true") for item in blockers),
                    "futureRuntimeStillBlocked": True,
                    "answerVisibleStillBlocked": True,
                    "answerableStillBlocked": True,
                },
            }
        )

    by_status = Counter(row["adapterDesignStatus"] for row in rows)
    by_type = Counter(row["artifactType"] for row in rows)
    ready_rows = by_status.get(ADAPTER_STATUS_READY_CANDIDATE_ONLY, 0) if not source_blockers else 0
    blocked_rows = len(rows) - ready_rows
    private_path_leak_rows = sum(1 for row in rows if "private_path_leak" in row["adapterDesignBlockers"])
    schema_violations = list(source_blockers)
    if private_path_leak_rows:
        schema_violations = sorted(set([*schema_violations, "private_path_leak"]))
    counts = {
        "inputRows": len(source_rows),
        "contractReadyInputRows": sum(1 for row in source_rows if row.get("contractReviewReady") is True),
        "runtimeAdapterDesignReadyRows": ready_rows,
        "futureAdapterCandidateRows": ready_rows,
        "plannedCandidateStoreReadRows": ready_rows,
        "plannedEvidenceItemShapeRows": ready_rows,
        "plannedDiagnosticsRows": ready_rows,
        "answerableRows": 0,
        "blockedRows": blocked_rows,
        "blockedInputSchemaViolationRows": len(rows) if source_blockers and rows else int(bool(source_blockers)),
        "blockedContractNotReadyRows": by_status.get(ADAPTER_STATUS_BLOCKED_CONTRACT_NOT_READY, 0),
        "blockedMissingRequiredFieldRows": by_status.get(ADAPTER_STATUS_BLOCKED_MISSING_REQUIRED_FIELD, 0),
        "blockedInvalidLocatorRows": by_status.get(ADAPTER_STATUS_BLOCKED_INVALID_LOCATOR, 0),
        "blockedPolicyQuarantineRows": by_status.get(ADAPTER_STATUS_BLOCKED_POLICY_QUARANTINE, 0),
        "blockedPrivatePathRows": by_status.get(ADAPTER_STATUS_BLOCKED_PRIVATE_PATH, 0),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(schema_violations),
        "byArtifactType": dict(by_type),
        "byAdapterDesignStatus": dict(by_status),
    }
    status = "blocked" if schema_violations or blocked_rows else "ready"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DESIGN_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE if status == "ready" else "parsed_artifact_evidence_chunk_candidate_runtime_adapter_design_repair",
        "sourceRuntimeContractReview": _source_summary(
            runtime_contract_review_report,
            report_ref=source_runtime_contract_review_report_ref,
        ),
        "policy": {
            "reportOnly": True,
            "runtimeAdapterDesignOnly": True,
            "runtimeCodeChanged": False,
            "evidenceAssemblyChanged": False,
            "candidateStoreReadOnlyPlanned": True,
            "candidateStoreWrite": False,
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "answerVisibleExposure": False,
            "answerGeneration": False,
            "answerableRowsAllowedInThisTranche": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "parserExecution": False,
            "canonicalParsedArtifactsWritten": False,
            "futureRuntimeAdapterRequired": True,
            "nextGateRequiredBeforeRuntimeUse": True,
        },
        "adapterDesign": _adapter_design(),
        "counts": counts,
        "gate": {
            "readyForRuntimeAdapterDryRun": status == "ready",
            "readyForRuntimeApply": False,
            "runtimeCodeChangeAllowed": False,
            "candidateStoreWriteAllowed": False,
            "sourceSpanCreationAllowed": False,
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "answerVisibleAllowed": False,
            "answerableAllowed": False,
            "schemaViolations": schema_violations,
        },
        "rows": rows,
        "warnings": [],
    }


def render_parsed_artifact_evidence_chunk_candidate_runtime_adapter_design_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    design = dict(report.get("adapterDesign") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Candidate Runtime Adapter Design",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- adapterId: `{design.get('adapterId')}`",
        f"- integrationBoundary: `{design.get('integrationBoundary')}`",
        f"- insertionPoint: `{design.get('insertionPoint')}`",
        f"- inputRows: `{counts.get('inputRows')}`",
        f"- runtimeAdapterDesignReadyRows: `{counts.get('runtimeAdapterDesignReadyRows')}`",
        f"- futureAdapterCandidateRows: `{counts.get('futureAdapterCandidateRows')}`",
        f"- plannedCandidateStoreReadRows: `{counts.get('plannedCandidateStoreReadRows')}`",
        f"- plannedEvidenceItemShapeRows: `{counts.get('plannedEvidenceItemShapeRows')}`",
        f"- answerableRows: `{counts.get('answerableRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Mutation Guarantees",
        "",
    ]
    for field in ZERO_COUNTER_FIELDS:
        lines.append(f"- {field}: `{counts.get(field)}`")
    lines.extend(["", "## Adapter Design Status", ""])
    for status, count in sorted(dict(counts.get("byAdapterDesignStatus") or {}).items()):
        lines.append(f"- `{status}`: `{count}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_candidate_runtime_adapter_design(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_candidate_runtime_adapter_design_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DESIGN_SCHEMA_ID",
    "ADAPTER_STATUS_READY_CANDIDATE_ONLY",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_candidate_runtime_adapter_design",
    "load_json",
    "sanitized_report_ref",
    "write_parsed_artifact_evidence_chunk_candidate_runtime_adapter_design",
]
