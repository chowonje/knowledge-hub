"""Contract review for parsed-artifact evidence chunk runtime previews.

The review consumes the runtime integration dry-run and checks that each
previewed span has the fields the EvidencePacket/AnswerContract path will need
later. It is still report-only: no preview row becomes runtime evidence,
answer-visible text, StrictEvidence, or citation-grade evidence here.
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
    sanitized_report_ref,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_INTEGRATION_DRY_RUN_SCHEMA_ID,
    READY_DECISION as RUNTIME_DRY_RUN_READY_DECISION,
    RUNTIME_STATUS_READY_CANDIDATE_ONLY,
    load_json,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_CONTRACT_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-runtime-contract-review.v1"
)

CONTRACT_STATUS_READY_CANDIDATE_ONLY = "runtime_contract_review_ready_candidate_only"
CONTRACT_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
CONTRACT_STATUS_BLOCKED_RUNTIME_PREVIEW_NOT_READY = "blocked_runtime_preview_not_ready"
CONTRACT_STATUS_BLOCKED_SOURCE_ID_MISMATCH = "blocked_source_id_mismatch"
CONTRACT_STATUS_BLOCKED_HASH_MISMATCH = "blocked_hash_mismatch"
CONTRACT_STATUS_BLOCKED_LOCATOR_MISMATCH = "blocked_locator_mismatch"
CONTRACT_STATUS_BLOCKED_MISSING_REQUIRED_FIELD = "blocked_missing_required_contract_field"
CONTRACT_STATUS_BLOCKED_POLICY_QUARANTINE = "blocked_policy_quarantine_violation"
CONTRACT_STATUS_BLOCKED_PRIVATE_PATH = "blocked_private_path_leak"

READY_DECISION = "parsed_artifact_evidence_chunk_candidate_runtime_contract_review_ready"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE = "parsed_artifact_evidence_chunk_candidate_runtime_adapter_design"

ALLOWED_ARTIFACT_TYPES = {"section", "paragraph"}
EXPECTED_EVIDENCE_KIND = "parsed_artifact_evidence_chunk_candidate_runtime_preview"


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
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_INTEGRATION_DRY_RUN_SCHEMA_ID:
        blockers.append("invalid_runtime_integration_dry_run_schema")
    if report.get("status") != "ready":
        blockers.append("runtime_integration_dry_run_not_ready")
    if report.get("decision") != RUNTIME_DRY_RUN_READY_DECISION:
        blockers.append("runtime_integration_dry_run_invalid_decision")
    if gate.get("readyForRuntimeContractReview") is not True:
        blockers.append("runtime_integration_dry_run_gate_not_ready_for_contract_review")
    input_rows = _int(counts.get("inputRows"))
    if input_rows <= 0:
        blockers.append("runtime_integration_dry_run_has_no_input_rows")
    if _int(counts.get("runtimeIntegrationDryRunReadyRows")) != input_rows:
        blockers.append("runtime_integration_ready_count_mismatch")
    if _int(counts.get("plannedEvidencePacketSpanRows")) != input_rows:
        blockers.append("planned_evidence_packet_span_count_mismatch")
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
            blockers.append(f"runtime_integration_dry_run_has_{field_name}")
    if _contains_private_path(report):
        blockers.append("runtime_integration_dry_run_has_private_path_leak")
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
        "runtimeIntegrationDryRunReadyRows": _int(counts.get("runtimeIntegrationDryRunReadyRows")),
        "plannedEvidencePacketSpanRows": _int(counts.get("plannedEvidencePacketSpanRows")),
        "answerableRows": _int(counts.get("answerableRows")),
        "blockedRows": _int(counts.get("blockedRows")),
        "privatePathLeakRows": _int(counts.get("privatePathLeakRows")),
        "schemaViolationCount": _int(counts.get("schemaViolationCount")),
    }


def _required_span_text(span: dict[str, Any], *names: str) -> str:
    for name in names:
        value = normalize_text(span.get(name))
        if value:
            return value
    return ""


def _all_span_aliases_equal(span: dict[str, Any], expected: str, *names: str) -> bool:
    values = [normalize_text(span.get(name)) for name in names]
    return all(value == expected for value in values)


def _row_blockers(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    span = dict(row.get("plannedEvidencePacketSpan") or {})
    if normalize_text(row.get("runtimeIntegrationStatus")) != RUNTIME_STATUS_READY_CANDIDATE_ONLY:
        blockers.append("runtime_integration_status_not_ready_candidate_only")
    if row.get("runtimeIntegrationDryRunReady") is not True:
        blockers.append("runtime_integration_dry_run_ready_not_true")
    if row.get("plannedEvidencePacketSpanReady") is not True:
        blockers.append("planned_evidence_packet_span_ready_not_true")
    if normalize_text(row.get("artifactType")) not in ALLOWED_ARTIFACT_TYPES:
        blockers.append("unsupported_artifact_type")
    for field_name in ("candidateRecordId", "paperId", "sourceRef", "sourceContentHash", "spanLocator", "snippetHash"):
        if not normalize_text(row.get(field_name)):
            blockers.append(f"{field_name}_missing")
    for span_field in (
        "source_id",
        "source_ref",
        "source_content_hash",
        "content_hash",
        "span_locator",
        "char_start",
        "char_end",
        "evidence_kind",
        "derivativeSource",
    ):
        if span.get(span_field) in (None, "", {}):
            blockers.append(f"planned_span_{span_field}_missing")
    paper_id = normalize_text(row.get("paperId"))
    if not _all_span_aliases_equal(span, "paper", "sourceType", "source_type"):
        blockers.append("planned_span_source_type_not_paper")
    if not _all_span_aliases_equal(span, paper_id, "sourceId", "source_id"):
        blockers.append("planned_span_source_id_mismatch")
    if not _all_span_aliases_equal(span, normalize_text(row.get("sourceRef")), "sourceRef", "source_ref"):
        blockers.append("planned_span_source_ref_mismatch")
    if not _all_span_aliases_equal(span, normalize_text(row.get("sourceContentHash")), "sourceContentHash", "source_content_hash"):
        blockers.append("planned_span_source_hash_mismatch")
    if not _all_span_aliases_equal(span, normalize_text(row.get("snippetHash")), "snippetHash", "snippet_hash", "content_hash"):
        blockers.append("planned_span_snippet_hash_mismatch")
    row_locator = normalize_text(row.get("spanLocator"))
    if not _all_span_aliases_equal(span, row_locator, "spanLocator", "span_locator", "locator"):
        blockers.append("planned_span_locator_mismatch")
    span_locator = _required_span_text(span, "spanLocator", "span_locator", "locator")
    parsed_start, parsed_end = parse_span_offsets(row_locator, span_locator)
    if parsed_start is None or parsed_end is None:
        blockers.append("chars_locator_missing_or_invalid")
    if _int(row.get("charStart")) != parsed_start or _int(span.get("charStart") or span.get("char_start")) != parsed_start:
        blockers.append("char_start_mismatch")
    if _int(row.get("charEnd")) != parsed_end or _int(span.get("charEnd") or span.get("char_end")) != parsed_end:
        blockers.append("char_end_mismatch")
    if span.get("spanOffsetAvailable") is not True:
        blockers.append("span_offset_available_not_true")
    if _required_span_text(span, "evidenceKind", "evidence_kind") != EXPECTED_EVIDENCE_KIND:
        blockers.append("evidence_kind_mismatch")
    derivative = dict(span.get("derivativeSource") or {})
    if normalize_text(derivative.get("candidateRecordId")) != normalize_text(row.get("candidateRecordId")):
        blockers.append("derivative_candidate_record_id_mismatch")
    if normalize_text(derivative.get("candidateStoreRef")) != normalize_text(row.get("candidateStoreRef")):
        blockers.append("derivative_candidate_store_ref_mismatch")
    for item, label in ((row, "row"), (span, "planned_span")):
        if item.get("candidateOnly") is not True:
            blockers.append(f"{label}_candidate_only_not_true")
        if item.get("strictEvidence") is not False:
            blockers.append(f"{label}_strict_evidence_not_false")
        if item.get("citationGrade") is not False:
            blockers.append(f"{label}_citation_grade_not_false")
        if item.get("runtimeEvidence") is not False:
            blockers.append(f"{label}_runtime_evidence_not_false")
        if item.get("answerVisible") is not False:
            blockers.append(f"{label}_answer_visible_not_false")
        if item.get("answerable") is not False:
            blockers.append(f"{label}_answerable_not_false")
    if span.get("textPreviewIncluded") is not False:
        blockers.append("planned_span_text_preview_included_not_false")
    if row.get("runtimeVisibilityAllowed") is not False:
        blockers.append("runtime_visibility_allowed_not_false")
    if row.get("answerVisibleAllowed") is not False:
        blockers.append("answer_visible_allowed_not_false")
    if row.get("answerableAllowed") is not False:
        blockers.append("answerable_allowed_not_false")
    if _contains_private_path(row):
        blockers.append("private_path_leak")
    return sorted(set(blockers))


def _contract_status(blockers: list[str], source_blockers: list[str]) -> str:
    if source_blockers:
        return CONTRACT_STATUS_BLOCKED_INPUT_SCHEMA
    if not blockers:
        return CONTRACT_STATUS_READY_CANDIDATE_ONLY
    if any(item.startswith("runtime_integration_") or item.startswith("planned_evidence_packet_span_ready") for item in blockers):
        return CONTRACT_STATUS_BLOCKED_RUNTIME_PREVIEW_NOT_READY
    if any("source_id_mismatch" in item or "source_ref_mismatch" in item for item in blockers):
        return CONTRACT_STATUS_BLOCKED_SOURCE_ID_MISMATCH
    if any("hash_mismatch" in item for item in blockers):
        return CONTRACT_STATUS_BLOCKED_HASH_MISMATCH
    if any("locator" in item or item in {"char_start_mismatch", "char_end_mismatch"} for item in blockers):
        return CONTRACT_STATUS_BLOCKED_LOCATOR_MISMATCH
    if "private_path_leak" in blockers:
        return CONTRACT_STATUS_BLOCKED_PRIVATE_PATH
    if any(item.endswith("_not_false") or item.endswith("_not_true") for item in blockers):
        return CONTRACT_STATUS_BLOCKED_POLICY_QUARANTINE
    return CONTRACT_STATUS_BLOCKED_MISSING_REQUIRED_FIELD


def build_parsed_artifact_evidence_chunk_candidate_runtime_contract_review(
    *,
    runtime_integration_dry_run_report: dict[str, Any],
    source_runtime_integration_dry_run_report_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_rows = [
        dict(row)
        for row in list(runtime_integration_dry_run_report.get("rows") or [])
        if isinstance(row, dict)
    ]
    source_blockers = _source_blockers(runtime_integration_dry_run_report)
    rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(source_rows, start=1):
        row_blockers = _row_blockers(source_row)
        status = _contract_status(row_blockers, source_blockers)
        blockers = sorted(set([*source_blockers, *row_blockers]))
        ready = status == CONTRACT_STATUS_READY_CANDIDATE_ONLY and not blockers
        span = dict(source_row.get("plannedEvidencePacketSpan") or {})
        rows.append(
            {
                "contractReviewRowId": f"parsed-artifact-evidence-chunk-candidate-runtime-contract-review:{index:04d}",
                "sourceRuntimeDryRunRowId": normalize_text(source_row.get("runtimeDryRunRowId")),
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
                "contractReviewStatus": status,
                "contractReviewBlockers": blockers,
                "contractReviewReady": ready,
                "evidencePacketSpanCompatible": ready,
                "answerContractCitationShapeCompatible": ready,
                "strictProvenanceShapePresent": ready,
                "plannedEvidencePacketSpan": span,
                "runtimeAdapterDesignCandidate": ready,
                "runtimeVisibilityAllowed": False,
                "answerVisibleAllowed": False,
                "answerableAllowed": False,
                "candidateOnly": True,
                "strictEvidence": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "answerVisible": False,
                "answerable": False,
                "recommendedAction": (
                    "queue_for_runtime_adapter_design"
                    if ready
                    else "repair_runtime_preview_before_contract_review"
                ),
                "checks": {
                    "runtimePreviewReady": normalize_text(source_row.get("runtimeIntegrationStatus")) == RUNTIME_STATUS_READY_CANDIDATE_ONLY
                    and source_row.get("runtimeIntegrationDryRunReady") is True,
                    "sourceIdentityMatches": "planned_span_source_id_mismatch" not in blockers
                    and "planned_span_source_ref_mismatch" not in blockers,
                    "sourceHashMatches": "planned_span_source_hash_mismatch" not in blockers,
                    "snippetHashMatches": "planned_span_snippet_hash_mismatch" not in blockers,
                    "strictCharsLocatorMatches": not any(
                        item in blockers
                        for item in (
                            "planned_span_locator_mismatch",
                            "chars_locator_missing_or_invalid",
                            "char_start_mismatch",
                            "char_end_mismatch",
                            "span_offset_available_not_true",
                        )
                    ),
                    "evidenceKindMatches": "evidence_kind_mismatch" not in blockers,
                    "policyQuarantineMaintained": not any(item.endswith("_not_false") or item.endswith("_not_true") for item in blockers),
                    "textPreviewStillExcluded": span.get("textPreviewIncluded") is False,
                    "runtimeEvidenceStillFalse": True,
                    "answerVisibleStillFalse": True,
                    "answerableStillFalse": True,
                },
            }
        )

    by_status = Counter(row["contractReviewStatus"] for row in rows)
    by_type = Counter(row["artifactType"] for row in rows)
    ready_rows = by_status.get(CONTRACT_STATUS_READY_CANDIDATE_ONLY, 0) if not source_blockers else 0
    blocked_rows = len(rows) - ready_rows
    private_path_leak_rows = sum(1 for row in rows if "private_path_leak" in row["contractReviewBlockers"])
    schema_violations = list(source_blockers)
    if private_path_leak_rows:
        schema_violations = sorted(set([*schema_violations, "private_path_leak"]))
    counts = {
        "inputRows": len(source_rows),
        "runtimeDryRunReadyInputRows": sum(1 for row in source_rows if row.get("runtimeIntegrationDryRunReady") is True),
        "contractReviewReadyRows": ready_rows,
        "evidencePacketSpanCompatibleRows": ready_rows,
        "answerContractCitationShapeCompatibleRows": ready_rows,
        "strictProvenanceShapeRows": ready_rows,
        "runtimeAdapterDesignCandidateRows": ready_rows,
        "answerableRows": 0,
        "blockedRows": blocked_rows,
        "blockedInputSchemaViolationRows": len(rows) if source_blockers and rows else int(bool(source_blockers)),
        "blockedRuntimePreviewNotReadyRows": by_status.get(CONTRACT_STATUS_BLOCKED_RUNTIME_PREVIEW_NOT_READY, 0),
        "blockedSourceIdMismatchRows": by_status.get(CONTRACT_STATUS_BLOCKED_SOURCE_ID_MISMATCH, 0),
        "blockedHashMismatchRows": by_status.get(CONTRACT_STATUS_BLOCKED_HASH_MISMATCH, 0),
        "blockedLocatorMismatchRows": by_status.get(CONTRACT_STATUS_BLOCKED_LOCATOR_MISMATCH, 0),
        "blockedMissingRequiredFieldRows": by_status.get(CONTRACT_STATUS_BLOCKED_MISSING_REQUIRED_FIELD, 0),
        "blockedPolicyQuarantineRows": by_status.get(CONTRACT_STATUS_BLOCKED_POLICY_QUARANTINE, 0),
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
        "byContractReviewStatus": dict(by_status),
    }
    status = "blocked" if schema_violations or blocked_rows else "ready"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_CONTRACT_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE if status == "ready" else "parsed_artifact_evidence_chunk_candidate_runtime_contract_review_repair",
        "sourceRuntimeIntegrationDryRun": _source_summary(
            runtime_integration_dry_run_report,
            report_ref=source_runtime_integration_dry_run_report_ref,
        ),
        "policy": {
            "reportOnly": True,
            "runtimeContractReviewOnly": True,
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
            "futureRuntimeAdapterRequired": True,
            "nextGateRequiredBeforeRuntimeUse": True,
        },
        "counts": counts,
        "gate": {
            "readyForRuntimeAdapterDesign": status == "ready",
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


def render_parsed_artifact_evidence_chunk_candidate_runtime_contract_review_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Candidate Runtime Contract Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- inputRows: `{counts.get('inputRows')}`",
        f"- runtimeDryRunReadyInputRows: `{counts.get('runtimeDryRunReadyInputRows')}`",
        f"- contractReviewReadyRows: `{counts.get('contractReviewReadyRows')}`",
        f"- evidencePacketSpanCompatibleRows: `{counts.get('evidencePacketSpanCompatibleRows')}`",
        f"- answerContractCitationShapeCompatibleRows: `{counts.get('answerContractCitationShapeCompatibleRows')}`",
        f"- strictProvenanceShapeRows: `{counts.get('strictProvenanceShapeRows')}`",
        f"- runtimeAdapterDesignCandidateRows: `{counts.get('runtimeAdapterDesignCandidateRows')}`",
        f"- answerableRows: `{counts.get('answerableRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Mutation Guarantees",
        "",
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
        "## Contract Review Status",
        "",
    ]
    for status, count in sorted(dict(counts.get("byContractReviewStatus") or {}).items()):
        lines.append(f"- `{status}`: `{count}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_candidate_runtime_contract_review(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_candidate_runtime_contract_review_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_CONTRACT_REVIEW_SCHEMA_ID",
    "CONTRACT_STATUS_READY_CANDIDATE_ONLY",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_candidate_runtime_contract_review",
    "load_json",
    "sanitized_report_ref",
    "write_parsed_artifact_evidence_chunk_candidate_runtime_contract_review",
]
