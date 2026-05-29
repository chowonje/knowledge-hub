"""Answerability policy gate for parsed-artifact evidence chunk candidates.

The gate consumes the full-apply readback review and classifies validated
section/paragraph candidate-store records as ready for a later runtime
integration dry-run. It deliberately does not create runtime evidence, mark
anything answerable, or expose candidate text to answer generation.
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
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_full_apply_readback_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READBACK_REVIEW_SCHEMA_ID,
    READBACK_STATUS_VALIDATED,
    READY_DECISION as READBACK_REVIEW_READY_DECISION,
    load_json,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_ANSWERABILITY_POLICY_GATE_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-answerability-policy-gate.v1"
)

POLICY_STATUS_READY_CANDIDATE_ONLY = "answerability_policy_ready_candidate_only"
POLICY_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
POLICY_STATUS_BLOCKED_READBACK_NOT_READY = "blocked_readback_not_ready"
POLICY_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE = "blocked_unsupported_artifact_type"
POLICY_STATUS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_content_hash"
POLICY_STATUS_BLOCKED_MISSING_SNIPPET_HASH = "blocked_missing_snippet_hash"
POLICY_STATUS_BLOCKED_MISSING_LOCATOR = "blocked_missing_chars_locator"
POLICY_STATUS_BLOCKED_MISSING_EXCERPT = "blocked_missing_verbatim_excerpt"
POLICY_STATUS_BLOCKED_POLICY_VIOLATION = "blocked_policy_quarantine_violation"

READY_DECISION = "parsed_artifact_evidence_chunk_candidate_answerability_policy_gate_ready"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE = "parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run"

ALLOWED_ARTIFACT_TYPES = {"section", "paragraph"}
CHARS_LOCATOR_RE = re.compile(r"^chars:(\d+)-(\d+)$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _row_blockers(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if normalize_text(row.get("readbackStatus")) != READBACK_STATUS_VALIDATED:
        blockers.append("readback_status_not_validated")
    if row.get("readbackValidated") is not True:
        blockers.append("readback_validated_not_true")
    if normalize_text(row.get("artifactType")) not in ALLOWED_ARTIFACT_TYPES:
        blockers.append("unsupported_artifact_type")
    if not normalize_text(row.get("sourceContentHash")):
        blockers.append("source_content_hash_missing")
    if not normalize_text(row.get("snippetHash")):
        blockers.append("snippet_hash_missing")
    match = CHARS_LOCATOR_RE.match(normalize_text(row.get("spanLocator")))
    if not match or int(match.group(2)) <= int(match.group(1)):
        blockers.append("chars_locator_missing_or_invalid")
    record_checks = dict(row.get("checks") or {})
    if record_checks.get("requiredFieldsPresent") is not True:
        blockers.append("required_fields_not_confirmed")
    if record_checks.get("candidatePolicyQuarantined") is not True:
        blockers.append("candidate_policy_not_quarantined")
    if record_checks.get("recordBytesMatchCanonicalJson") is not True:
        blockers.append("record_bytes_not_confirmed")
    if row.get("candidateOnly") is not True:
        blockers.append("candidate_only_not_true")
    if row.get("answerEvidenceCandidate") is not True:
        blockers.append("answer_evidence_candidate_not_true")
    if row.get("answerabilityCandidate") is not True:
        blockers.append("answerability_candidate_not_true")
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


def _policy_status(blockers: list[str]) -> str:
    if not blockers:
        return POLICY_STATUS_READY_CANDIDATE_ONLY
    if "readback_status_not_validated" in blockers or "readback_validated_not_true" in blockers:
        return POLICY_STATUS_BLOCKED_READBACK_NOT_READY
    if "unsupported_artifact_type" in blockers:
        return POLICY_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE
    if "source_content_hash_missing" in blockers:
        return POLICY_STATUS_BLOCKED_MISSING_SOURCE_HASH
    if "snippet_hash_missing" in blockers:
        return POLICY_STATUS_BLOCKED_MISSING_SNIPPET_HASH
    if "chars_locator_missing_or_invalid" in blockers:
        return POLICY_STATUS_BLOCKED_MISSING_LOCATOR
    if "required_fields_not_confirmed" in blockers:
        return POLICY_STATUS_BLOCKED_MISSING_EXCERPT
    return POLICY_STATUS_BLOCKED_POLICY_VIOLATION


def _source_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READBACK_REVIEW_SCHEMA_ID:
        blockers.append("invalid_readback_review_schema")
    if report.get("status") != "ready":
        blockers.append("readback_review_not_ready")
    if report.get("decision") != READBACK_REVIEW_READY_DECISION:
        blockers.append("readback_review_invalid_decision")
    if dict(report.get("gate") or {}).get("readyForAnswerabilityPolicyGate") is not True:
        blockers.append("readback_review_gate_not_ready")
    expected = _int(counts.get("expectedCandidateRows"))
    if expected <= 0:
        blockers.append("readback_review_has_no_expected_candidates")
    if _int(counts.get("readbackValidatedRows")) != expected:
        blockers.append("readback_review_validated_count_mismatch")
    for field_name in (
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
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
    ):
        if _int(counts.get(field_name)) != 0:
            blockers.append(f"readback_review_has_{field_name}")
    if _contains_private_path(report):
        blockers.append("readback_review_has_private_path_leak")
    return sorted(set(blockers))


def _source_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "nextRecommendedTranche": normalize_text(report.get("nextRecommendedTranche")),
        "reportRef": normalize_text(report_ref),
        "expectedCandidateRows": _int(counts.get("expectedCandidateRows")),
        "readbackValidatedRows": _int(counts.get("readbackValidatedRows")),
        "blockedRows": _int(counts.get("blockedRows")),
        "privatePathLeakRows": _int(counts.get("privatePathLeakRows")),
        "schemaViolationCount": _int(counts.get("schemaViolationCount")),
    }


def build_parsed_artifact_evidence_chunk_candidate_answerability_policy_gate(
    *,
    full_apply_readback_review_report: dict[str, Any],
    source_readback_review_report_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_rows = [
        dict(row)
        for row in list(full_apply_readback_review_report.get("rows") or [])
        if isinstance(row, dict)
    ]
    schema_violations = _source_blockers(full_apply_readback_review_report)
    rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(source_rows, start=1):
        blockers = _row_blockers(source_row)
        status = _policy_status(blockers)
        ready = status == POLICY_STATUS_READY_CANDIDATE_ONLY and not schema_violations
        rows.append(
            {
                "policyGateRowId": f"parsed-artifact-evidence-chunk-candidate-answerability-policy-gate:{index:04d}",
                "sourceReadbackReviewRowId": normalize_text(source_row.get("readbackReviewRowId")),
                "candidateRecordId": normalize_text(source_row.get("candidateRecordId")),
                "sourceCandidateRowId": normalize_text(source_row.get("sourceCandidateRowId")),
                "paperId": normalize_text(source_row.get("paperId")),
                "artifactType": normalize_text(source_row.get("artifactType")),
                "sourceRef": normalize_text(source_row.get("sourceRef")),
                "sourceContentHash": normalize_text(source_row.get("sourceContentHash")),
                "spanLocator": normalize_text(source_row.get("spanLocator")),
                "snippetHash": normalize_text(source_row.get("snippetHash")),
                "candidateStoreRef": normalize_text(source_row.get("candidateStoreRef")),
                "answerabilityPolicyStatus": status if not schema_violations else POLICY_STATUS_BLOCKED_INPUT_SCHEMA,
                "answerabilityPolicyBlockers": sorted(set([*blockers, *schema_violations])),
                "answerEvidenceCandidate": True,
                "answerabilityCandidate": True,
                "answerabilityPolicyReady": ready,
                "runtimeIntegrationDryRunReady": ready,
                "answerEvidenceEligible": ready,
                "answerabilityEligible": ready,
                "answerable": False,
                "candidateOnly": True,
                "strictEvidence": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "answerVisible": False,
                "recommendedAction": (
                    "queue_for_candidate_runtime_integration_dry_run"
                    if ready
                    else "repair_candidate_or_readback_before_answerability_policy_gate"
                ),
                "checks": {
                    "readbackValidated": normalize_text(source_row.get("readbackStatus")) == READBACK_STATUS_VALIDATED
                    and source_row.get("readbackValidated") is True,
                    "artifactTypeAllowed": normalize_text(source_row.get("artifactType")) in ALLOWED_ARTIFACT_TYPES,
                    "sourceContentHashPresent": bool(normalize_text(source_row.get("sourceContentHash"))),
                    "snippetHashPresent": bool(normalize_text(source_row.get("snippetHash"))),
                    "charsLocatorPresent": bool(CHARS_LOCATOR_RE.match(normalize_text(source_row.get("spanLocator")))),
                    "candidatePolicyQuarantined": dict(source_row.get("checks") or {}).get("candidatePolicyQuarantined") is True,
                    "answerableStillFalse": True,
                    "runtimeVisibilityStillFalse": True,
                },
            }
        )

    by_status = Counter(row["answerabilityPolicyStatus"] for row in rows)
    by_type = Counter(row["artifactType"] for row in rows)
    ready_rows = by_status.get(POLICY_STATUS_READY_CANDIDATE_ONLY, 0) if not schema_violations else 0
    blocked_rows = len(rows) - ready_rows
    counts = {
        "inputRows": len(source_rows),
        "readbackValidatedInputRows": sum(1 for row in source_rows if normalize_text(row.get("readbackStatus")) == READBACK_STATUS_VALIDATED),
        "answerabilityPolicyReadyRows": ready_rows,
        "answerEvidenceEligibleCandidateRows": ready_rows,
        "answerabilityEligibleCandidateRows": ready_rows,
        "runtimeIntegrationDryRunReadyRows": ready_rows,
        "answerableRows": 0,
        "blockedRows": blocked_rows,
        "blockedInputSchemaViolationRows": len(rows) if schema_violations and rows else int(bool(schema_violations)),
        "blockedReadbackNotReadyRows": by_status.get(POLICY_STATUS_BLOCKED_READBACK_NOT_READY, 0),
        "blockedUnsupportedArtifactTypeRows": by_status.get(POLICY_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE, 0),
        "blockedMissingSourceHashRows": by_status.get(POLICY_STATUS_BLOCKED_MISSING_SOURCE_HASH, 0),
        "blockedMissingSnippetHashRows": by_status.get(POLICY_STATUS_BLOCKED_MISSING_SNIPPET_HASH, 0),
        "blockedMissingLocatorRows": by_status.get(POLICY_STATUS_BLOCKED_MISSING_LOCATOR, 0),
        "blockedMissingExcerptRows": by_status.get(POLICY_STATUS_BLOCKED_MISSING_EXCERPT, 0),
        "blockedPolicyViolationRows": by_status.get(POLICY_STATUS_BLOCKED_POLICY_VIOLATION, 0),
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
        "privatePathLeakRows": 1 if _contains_private_path(rows) else 0,
        "schemaViolationCount": len(schema_violations),
        "byArtifactType": dict(by_type),
        "byAnswerabilityPolicyStatus": dict(by_status),
    }
    if counts["privatePathLeakRows"]:
        schema_violations = sorted(set([*schema_violations, "private_path_leak"]))
        counts["schemaViolationCount"] = len(schema_violations)
    status = "blocked" if schema_violations or blocked_rows else "ready"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_ANSWERABILITY_POLICY_GATE_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE if status == "ready" else "parsed_artifact_evidence_chunk_candidate_answerability_policy_gate_repair",
        "sourceFullApplyReadbackReview": _source_summary(
            full_apply_readback_review_report,
            report_ref=source_readback_review_report_ref,
        ),
        "policy": {
            "reportOnly": True,
            "answerabilityPolicyGateOnly": True,
            "allowedArtifactTypes": sorted(ALLOWED_ARTIFACT_TYPES),
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
            "nextGateRequiredBeforeRuntimeUse": True,
        },
        "counts": counts,
        "gate": {
            "readyForRuntimeIntegrationDryRun": status == "ready",
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


def render_parsed_artifact_evidence_chunk_candidate_answerability_policy_gate_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Candidate Answerability Policy Gate",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- inputRows: `{counts.get('inputRows')}`",
        f"- answerabilityPolicyReadyRows: `{counts.get('answerabilityPolicyReadyRows')}`",
        f"- runtimeIntegrationDryRunReadyRows: `{counts.get('runtimeIntegrationDryRunReadyRows')}`",
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
        "## Policy Status",
        "",
    ]
    for status, count in sorted(dict(counts.get("byAnswerabilityPolicyStatus") or {}).items()):
        lines.append(f"- `{status}`: `{count}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_candidate_answerability_policy_gate(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_candidate_answerability_policy_gate_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_ANSWERABILITY_POLICY_GATE_SCHEMA_ID",
    "POLICY_STATUS_READY_CANDIDATE_ONLY",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_candidate_answerability_policy_gate",
    "load_json",
    "sanitized_report_ref",
    "write_parsed_artifact_evidence_chunk_candidate_answerability_policy_gate",
]
