"""Runtime integration dry-run for parsed-artifact evidence chunk candidates.

This report consumes the answerability policy gate and projects candidate rows
into EvidencePacket-compatible span previews. It deliberately keeps the rows
candidate-only: no SourceSpan, StrictEvidence, citation-grade, runtime-visible,
or answer-visible evidence is created here.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any

from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    _contains_private_path,
    normalize_text,
    sanitized_report_ref,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_answerability_policy_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_ANSWERABILITY_POLICY_GATE_SCHEMA_ID,
    POLICY_STATUS_READY_CANDIDATE_ONLY,
    READY_DECISION as ANSWERABILITY_POLICY_READY_DECISION,
    load_json,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_INTEGRATION_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-runtime-integration-dry-run.v1"
)

RUNTIME_STATUS_READY_CANDIDATE_ONLY = "runtime_integration_dry_run_ready_candidate_only"
RUNTIME_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
RUNTIME_STATUS_BLOCKED_POLICY_NOT_READY = "blocked_answerability_policy_not_ready"
RUNTIME_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE = "blocked_unsupported_artifact_type"
RUNTIME_STATUS_BLOCKED_MISSING_REQUIRED_FIELD = "blocked_missing_required_runtime_field"
RUNTIME_STATUS_BLOCKED_INVALID_LOCATOR = "blocked_invalid_chars_locator"
RUNTIME_STATUS_BLOCKED_POLICY_QUARANTINE = "blocked_policy_quarantine_violation"
RUNTIME_STATUS_BLOCKED_PRIVATE_PATH = "blocked_private_path_leak"

READY_DECISION = "parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run_ready"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE = "parsed_artifact_evidence_chunk_candidate_runtime_contract_review"

ALLOWED_ARTIFACT_TYPES = {"section", "paragraph"}
CHARS_LOCATOR_RE = re.compile(r"^chars:(\d+)-(\d+)$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _parse_chars_locator(value: Any) -> tuple[int | None, int | None]:
    match = CHARS_LOCATOR_RE.match(normalize_text(value))
    if not match:
        return None, None
    start = int(match.group(1))
    end = int(match.group(2))
    if end <= start:
        return None, None
    return start, end


def _source_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_ANSWERABILITY_POLICY_GATE_SCHEMA_ID:
        blockers.append("invalid_answerability_policy_schema")
    if report.get("status") != "ready":
        blockers.append("answerability_policy_gate_not_ready")
    if report.get("decision") != ANSWERABILITY_POLICY_READY_DECISION:
        blockers.append("answerability_policy_gate_invalid_decision")
    if gate.get("readyForRuntimeIntegrationDryRun") is not True:
        blockers.append("answerability_policy_gate_not_ready_for_runtime_dry_run")
    input_rows = _int(counts.get("inputRows"))
    if input_rows <= 0:
        blockers.append("answerability_policy_gate_has_no_input_rows")
    if _int(counts.get("answerabilityPolicyReadyRows")) != input_rows:
        blockers.append("answerability_policy_ready_count_mismatch")
    if _int(counts.get("runtimeIntegrationDryRunReadyRows")) != input_rows:
        blockers.append("runtime_integration_dry_run_ready_count_mismatch")
    for field_name in (
        "answerableRows",
        "blockedRows",
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
        "privatePathLeakRows",
        "schemaViolationCount",
    ):
        if _int(counts.get(field_name)) != 0:
            blockers.append(f"answerability_policy_has_{field_name}")
    if _contains_private_path(report):
        blockers.append("answerability_policy_has_private_path_leak")
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
        "answerabilityPolicyReadyRows": _int(counts.get("answerabilityPolicyReadyRows")),
        "runtimeIntegrationDryRunReadyRows": _int(counts.get("runtimeIntegrationDryRunReadyRows")),
        "blockedRows": _int(counts.get("blockedRows")),
        "privatePathLeakRows": _int(counts.get("privatePathLeakRows")),
        "schemaViolationCount": _int(counts.get("schemaViolationCount")),
    }


def _row_blockers(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if normalize_text(row.get("answerabilityPolicyStatus")) != POLICY_STATUS_READY_CANDIDATE_ONLY:
        blockers.append("answerability_policy_status_not_ready_candidate_only")
    if row.get("answerabilityPolicyReady") is not True:
        blockers.append("answerability_policy_ready_not_true")
    if row.get("runtimeIntegrationDryRunReady") is not True:
        blockers.append("runtime_integration_dry_run_ready_not_true")
    if normalize_text(row.get("artifactType")) not in ALLOWED_ARTIFACT_TYPES:
        blockers.append("unsupported_artifact_type")
    for field_name in ("candidateRecordId", "paperId", "sourceRef", "sourceContentHash", "spanLocator", "snippetHash"):
        if not normalize_text(row.get(field_name)):
            blockers.append(f"{field_name}_missing")
    start, end = _parse_chars_locator(row.get("spanLocator"))
    if start is None or end is None:
        blockers.append("chars_locator_missing_or_invalid")
    if row.get("candidateOnly") is not True:
        blockers.append("candidate_only_not_true")
    if row.get("answerable") is not False:
        blockers.append("answerable_not_false")
    if row.get("strictEvidence") is not False:
        blockers.append("strict_evidence_not_false")
    if row.get("citationGrade") is not False:
        blockers.append("citation_grade_not_false")
    if row.get("runtimeEvidence") is not False:
        blockers.append("runtime_evidence_not_false")
    if row.get("answerVisible") is not False:
        blockers.append("answer_visible_not_false")
    if _contains_private_path(row):
        blockers.append("private_path_leak")
    return sorted(set(blockers))


def _runtime_status(blockers: list[str], source_blockers: list[str]) -> str:
    if source_blockers:
        return RUNTIME_STATUS_BLOCKED_INPUT_SCHEMA
    if not blockers:
        return RUNTIME_STATUS_READY_CANDIDATE_ONLY
    if any(item.startswith("answerability_policy_") or item.startswith("runtime_integration_dry_run_ready") for item in blockers):
        return RUNTIME_STATUS_BLOCKED_POLICY_NOT_READY
    if "unsupported_artifact_type" in blockers:
        return RUNTIME_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE
    if "chars_locator_missing_or_invalid" in blockers:
        return RUNTIME_STATUS_BLOCKED_INVALID_LOCATOR
    if "private_path_leak" in blockers:
        return RUNTIME_STATUS_BLOCKED_PRIVATE_PATH
    if any(item.endswith("_not_false") or item.endswith("_not_true") for item in blockers):
        return RUNTIME_STATUS_BLOCKED_POLICY_QUARANTINE
    return RUNTIME_STATUS_BLOCKED_MISSING_REQUIRED_FIELD


def _planned_span(*, row: dict[str, Any], index: int, start: int | None, end: int | None) -> dict[str, Any]:
    paper_id = normalize_text(row.get("paperId"))
    span_ref = f"span:{index}"
    source_hash = normalize_text(row.get("sourceContentHash"))
    snippet_hash = normalize_text(row.get("snippetHash"))
    span_locator = normalize_text(row.get("spanLocator"))
    return {
        "spanRef": span_ref,
        "span_id": span_ref,
        "sourceType": "paper",
        "source_type": "paper",
        "sourceId": paper_id,
        "source_id": paper_id,
        "sourceRef": normalize_text(row.get("sourceRef")),
        "source_ref": normalize_text(row.get("sourceRef")),
        "sourceContentHash": source_hash,
        "source_content_hash": source_hash,
        "content_hash": snippet_hash,
        "snippetHash": snippet_hash,
        "snippet_hash": snippet_hash,
        "spanLocator": span_locator,
        "span_locator": span_locator,
        "locator": span_locator,
        "charStart": start,
        "char_start": start,
        "charEnd": end,
        "char_end": end,
        "spanOffsetAvailable": start is not None and end is not None,
        "artifactType": normalize_text(row.get("artifactType")),
        "evidenceKind": "parsed_artifact_evidence_chunk_candidate_runtime_preview",
        "evidence_kind": "parsed_artifact_evidence_chunk_candidate_runtime_preview",
        "derivativeSource": {
            "candidateRecordId": normalize_text(row.get("candidateRecordId")),
            "policyGateRowId": normalize_text(row.get("policyGateRowId")),
            "sourceCandidateRowId": normalize_text(row.get("sourceCandidateRowId")),
            "candidateStoreRef": normalize_text(row.get("candidateStoreRef")),
        },
        "textPreviewIncluded": False,
        "candidateOnly": True,
        "runtimeEvidence": False,
        "answerVisible": False,
        "answerable": False,
        "strictEvidence": False,
        "citationGrade": False,
    }


def build_parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run(
    *,
    answerability_policy_gate_report: dict[str, Any],
    source_answerability_policy_gate_report_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_rows = [
        dict(row)
        for row in list(answerability_policy_gate_report.get("rows") or [])
        if isinstance(row, dict)
    ]
    source_blockers = _source_blockers(answerability_policy_gate_report)
    rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(source_rows, start=1):
        row_blockers = _row_blockers(source_row)
        status = _runtime_status(row_blockers, source_blockers)
        blockers = sorted(set([*source_blockers, *row_blockers]))
        start, end = _parse_chars_locator(source_row.get("spanLocator"))
        ready = status == RUNTIME_STATUS_READY_CANDIDATE_ONLY and not blockers
        planned_span = _planned_span(row=source_row, index=index, start=start, end=end)
        rows.append(
            {
                "runtimeDryRunRowId": f"parsed-artifact-evidence-chunk-candidate-runtime-integration-dry-run:{index:04d}",
                "sourcePolicyGateRowId": normalize_text(source_row.get("policyGateRowId")),
                "candidateRecordId": normalize_text(source_row.get("candidateRecordId")),
                "sourceCandidateRowId": normalize_text(source_row.get("sourceCandidateRowId")),
                "paperId": normalize_text(source_row.get("paperId")),
                "artifactType": normalize_text(source_row.get("artifactType")),
                "sourceRef": normalize_text(source_row.get("sourceRef")),
                "sourceContentHash": normalize_text(source_row.get("sourceContentHash")),
                "spanLocator": normalize_text(source_row.get("spanLocator")),
                "charStart": start,
                "charEnd": end,
                "snippetHash": normalize_text(source_row.get("snippetHash")),
                "candidateStoreRef": normalize_text(source_row.get("candidateStoreRef")),
                "runtimeIntegrationStatus": status,
                "runtimeIntegrationBlockers": blockers,
                "runtimeIntegrationDryRunReady": ready,
                "plannedRuntimeSpanId": f"parsed-artifact-evidence-chunk-runtime-preview:{index:04d}",
                "plannedEvidencePacketSpan": planned_span,
                "plannedEvidencePacketSpanReady": ready,
                "plannedAnswerContextCandidate": ready,
                "runtimeVisibilityAllowed": False,
                "answerVisibleAllowed": False,
                "answerableAllowed": False,
                "futureRuntimeIntegrationRequired": True,
                "candidateOnly": True,
                "strictEvidence": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "answerVisible": False,
                "answerable": False,
                "recommendedAction": (
                    "queue_for_runtime_contract_review"
                    if ready
                    else "repair_policy_gate_candidate_before_runtime_integration"
                ),
                "checks": {
                    "answerabilityPolicyReady": source_row.get("answerabilityPolicyReady") is True,
                    "runtimeIntegrationDryRunReadyInput": source_row.get("runtimeIntegrationDryRunReady") is True,
                    "artifactTypeAllowed": normalize_text(source_row.get("artifactType")) in ALLOWED_ARTIFACT_TYPES,
                    "sourceContentHashPresent": bool(normalize_text(source_row.get("sourceContentHash"))),
                    "snippetHashPresent": bool(normalize_text(source_row.get("snippetHash"))),
                    "charsLocatorStrict": start is not None and end is not None,
                    "spanOffsetAvailable": start is not None and end is not None,
                    "evidencePacketPreviewOnly": True,
                    "textPreviewIncluded": False,
                    "answerableStillFalse": True,
                    "runtimeVisibilityStillFalse": True,
                },
            }
        )

    by_status = Counter(row["runtimeIntegrationStatus"] for row in rows)
    by_type = Counter(row["artifactType"] for row in rows)
    ready_rows = by_status.get(RUNTIME_STATUS_READY_CANDIDATE_ONLY, 0) if not source_blockers else 0
    blocked_rows = len(rows) - ready_rows
    private_path_leak_rows = sum(1 for row in rows if "private_path_leak" in row["runtimeIntegrationBlockers"])
    schema_violations = list(source_blockers)
    if private_path_leak_rows:
        schema_violations = sorted(set([*schema_violations, "private_path_leak"]))
    counts = {
        "inputRows": len(source_rows),
        "policyReadyInputRows": sum(1 for row in source_rows if row.get("answerabilityPolicyReady") is True),
        "runtimeIntegrationDryRunReadyRows": ready_rows,
        "plannedRuntimeSpanRows": ready_rows,
        "plannedEvidencePacketSpanRows": ready_rows,
        "plannedAnswerContextCandidateRows": ready_rows,
        "answerableRows": 0,
        "blockedRows": blocked_rows,
        "blockedInputSchemaViolationRows": len(rows) if source_blockers and rows else int(bool(source_blockers)),
        "blockedPolicyNotReadyRows": by_status.get(RUNTIME_STATUS_BLOCKED_POLICY_NOT_READY, 0),
        "blockedUnsupportedArtifactTypeRows": by_status.get(RUNTIME_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE, 0),
        "blockedMissingRequiredFieldRows": by_status.get(RUNTIME_STATUS_BLOCKED_MISSING_REQUIRED_FIELD, 0),
        "blockedInvalidLocatorRows": by_status.get(RUNTIME_STATUS_BLOCKED_INVALID_LOCATOR, 0),
        "blockedPolicyQuarantineRows": by_status.get(RUNTIME_STATUS_BLOCKED_POLICY_QUARANTINE, 0),
        "candidateStoreWriteRows": 0,
        "sourceSpanCreatedRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "answerVisibleRows": 0,
        "answerGenerationRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "parserExecutionRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(schema_violations),
        "byArtifactType": dict(by_type),
        "byRuntimeIntegrationStatus": dict(by_status),
    }
    status = "blocked" if schema_violations or blocked_rows else "ready"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_INTEGRATION_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE if status == "ready" else "parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run_repair",
        "sourceAnswerabilityPolicyGate": _source_summary(
            answerability_policy_gate_report,
            report_ref=source_answerability_policy_gate_report_ref,
        ),
        "policy": {
            "reportOnly": True,
            "runtimeIntegrationDryRunOnly": True,
            "evidencePacketPreviewOnly": True,
            "allowedArtifactTypes": sorted(ALLOWED_ARTIFACT_TYPES),
            "textPreviewIncluded": False,
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
            "futureRuntimeIntegrationRequired": True,
            "nextGateRequiredBeforeRuntimeUse": True,
        },
        "counts": counts,
        "gate": {
            "readyForRuntimeContractReview": status == "ready",
            "readyForRuntimeApply": False,
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


def render_parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Candidate Runtime Integration Dry Run",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- inputRows: `{counts.get('inputRows')}`",
        f"- policyReadyInputRows: `{counts.get('policyReadyInputRows')}`",
        f"- runtimeIntegrationDryRunReadyRows: `{counts.get('runtimeIntegrationDryRunReadyRows')}`",
        f"- plannedRuntimeSpanRows: `{counts.get('plannedRuntimeSpanRows')}`",
        f"- plannedEvidencePacketSpanRows: `{counts.get('plannedEvidencePacketSpanRows')}`",
        f"- plannedAnswerContextCandidateRows: `{counts.get('plannedAnswerContextCandidateRows')}`",
        f"- answerableRows: `{counts.get('answerableRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- sourceSpanCreatedRows: `{counts.get('sourceSpanCreatedRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        f"- runtimeEvidenceRows: `{counts.get('runtimeEvidenceRows')}`",
        f"- answerVisibleRows: `{counts.get('answerVisibleRows')}`",
        f"- answerGenerationRows: `{counts.get('answerGenerationRows')}`",
        f"- databaseMutationRows: `{counts.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{counts.get('indexMutationRows')}`",
        f"- vaultScanRows: `{counts.get('vaultScanRows')}`",
        f"- externalDownloadRows: `{counts.get('externalDownloadRows')}`",
        "",
        "## Runtime Integration Status",
        "",
    ]
    for status, count in sorted(dict(counts.get("byRuntimeIntegrationStatus") or {}).items()):
        lines.append(f"- `{status}`: `{count}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_INTEGRATION_DRY_RUN_SCHEMA_ID",
    "RUNTIME_STATUS_READY_CANDIDATE_ONLY",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run",
    "load_json",
    "sanitized_report_ref",
    "write_parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run",
]
