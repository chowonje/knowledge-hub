"""Readback review for parsed-artifact evidence chunk candidate-store records.

The review consumes the full-apply executor report, rereads the local
candidate-store JSONL files, and verifies every expected candidate record is
present exactly once and byte-equivalent under canonical JSON. The records
remain candidate-only: this review does not create SourceSpan, StrictEvidence,
citation-grade evidence, runtime evidence, or answer-visible payloads.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    _contains_private_path,
    normalize_text,
    sanitized_report_ref,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_canary_apply_readback import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
    _read_jsonl,
    _record_hash,
    _store_path,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback import (
    APPLIED_DECISION as FULL_APPLY_APPLIED_DECISION,
    NEXT_TRANCHE_APPLIED as FULL_APPLY_NEXT_TRANCHE,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_EXECUTOR_APPLY_READBACK_SCHEMA_ID,
    load_json,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READBACK_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-full-apply-readback-review.v1"
)

READBACK_STATUS_VALIDATED = "readback_validated_candidate_record"
READBACK_STATUS_BLOCKED_MISSING = "blocked_missing_store_record"
READBACK_STATUS_BLOCKED_DUPLICATE = "blocked_duplicate_store_record"
READBACK_STATUS_BLOCKED_HASH_MISMATCH = "blocked_hash_mismatch"
READBACK_STATUS_BLOCKED_POLICY_VIOLATION = "blocked_policy_violation"

READY_DECISION = "parsed_artifact_evidence_chunk_candidate_full_apply_readback_review_ready"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE = "parsed_artifact_evidence_chunk_candidate_answerability_policy_gate"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _idempotency_key(record: dict[str, Any]) -> str:
    return normalize_text(record.get("idempotencyKey"))


def _candidate_policy_ok(record: dict[str, Any]) -> bool:
    write_policy = dict(record.get("writePolicy") or {})
    return (
        record.get("schema") == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID
        and record.get("candidateOnly") is True
        and record.get("answerEvidenceCandidate") is True
        and record.get("answerabilityCandidate") is True
        and record.get("strictEvidence") is False
        and record.get("citationGrade") is False
        and record.get("runtimeEvidence") is False
        and record.get("answerVisible") is False
        and normalize_text(record.get("evidenceTier")) == "parsed_artifact_evidence_chunk_candidate_only"
        and write_policy.get("candidateStoreWrite") is True
        and write_policy.get("sourceSpanCreated") is False
        and write_policy.get("strictEvidenceCreated") is False
        and write_policy.get("citationGradeEvidenceCreated") is False
        and write_policy.get("runtimeEvidenceCreated") is False
        and write_policy.get("parserRoutingChanged") is False
        and write_policy.get("answerIntegrationChanged") is False
        and write_policy.get("databaseMutation") is False
        and write_policy.get("vaultScan") is False
        and write_policy.get("reindexOrReembed") is False
        and write_policy.get("canonicalParsedArtifactsWritten") is False
    )


def _record_required_fields_ok(record: dict[str, Any]) -> bool:
    for field_name in (
        "candidateRecordId",
        "sourceCandidateRowId",
        "paperId",
        "sourceType",
        "artifactType",
        "sourceRef",
        "sourceContentHash",
        "spanLocator",
        "excerpt",
        "snippetHash",
        "idempotencyKey",
        "candidateRecordHash",
    ):
        if not normalize_text(record.get(field_name)):
            return False
    locator = dict(record.get("locator") or {})
    chars = dict(locator.get("chars") or {})
    return (
        normalize_text(record.get("sourceType")) == "paper"
        and normalize_text(record.get("artifactType")) in {"section", "paragraph"}
        and locator.get("kind") == "parsed_document_chars"
        and _int(chars.get("end")) > _int(chars.get("start"))
    )


def _source_apply_report_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "nextRecommendedTranche": normalize_text(report.get("nextRecommendedTranche")),
        "reportRef": normalize_text(report_ref),
        "plannedApplyRows": _int(counts.get("plannedApplyRows")),
        "appliedCandidateRecordRows": _int(counts.get("appliedCandidateRecordRows")),
        "alreadyCorrectRows": _int(counts.get("alreadyCorrectRows")),
        "candidateStoreWriteRows": _int(counts.get("candidateStoreWriteRows")),
        "readbackValidatedRows": _int(counts.get("readbackValidatedRows")),
        "runManifestWriteRows": _int(counts.get("runManifestWriteRows")),
        "blockedRows": _int(counts.get("blockedRows")),
        "privatePathLeakRows": _int(counts.get("privatePathLeakRows")),
        "schemaViolationCount": _int(counts.get("schemaViolationCount")),
    }


def _source_blockers(report: dict[str, Any], *, expected_records: list[dict[str, Any]]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_EXECUTOR_APPLY_READBACK_SCHEMA_ID:
        blockers.append("invalid_full_apply_executor_schema")
    if report.get("status") != "applied":
        blockers.append("full_apply_executor_not_applied")
    if report.get("decision") != FULL_APPLY_APPLIED_DECISION:
        blockers.append("full_apply_executor_invalid_decision")
    if report.get("nextRecommendedTranche") != FULL_APPLY_NEXT_TRANCHE:
        blockers.append("full_apply_executor_unexpected_next_tranche")
    if not expected_records:
        blockers.append("full_apply_executor_candidate_records_missing")
    if _int(counts.get("plannedApplyRows")) != len(expected_records):
        blockers.append("full_apply_executor_planned_count_mismatch")
    if _int(counts.get("appliedCandidateRecordRows")) != len(expected_records):
        blockers.append("full_apply_executor_applied_count_mismatch")
    if _int(counts.get("readbackValidatedRows")) != len(expected_records):
        blockers.append("full_apply_executor_readback_count_mismatch")
    if _int(counts.get("runManifestWriteRows")) != 1:
        blockers.append("full_apply_executor_run_manifest_not_written")
    for field_name in (
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
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
            blockers.append(f"full_apply_executor_has_{field_name}")
    if _contains_private_path(report):
        blockers.append("full_apply_executor_has_private_path_leak")
    return sorted(set(blockers))


def _scope(candidate_rows: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "expectedCandidateRows": int(candidate_rows),
        "candidateStoreWriteRows": 0,
        "sourceSpanCreatedRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "answerVisibleRows": 0,
        "answerGenerationRows": 0,
        "parserExecutionRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
    }


def _policy() -> dict[str, Any]:
    return {
        "readbackReviewOnly": True,
        "candidateStoreWrite": False,
        "sourceSpanCreated": False,
        "strictEvidenceCreated": False,
        "citationGradeEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "answerVisibleExposure": False,
        "answerGeneration": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "vaultScan": False,
        "reindexOrReembed": False,
        "parserExecution": False,
        "canonicalParsedArtifactsWritten": False,
        "allowedUse": "candidate_store_only",
        "strictEvidence": False,
        "citationGrade": False,
        "runtimeVisible": False,
        "answerVisible": False,
        "answerabilityPolicyGateRequired": True,
    }


def _store_rows_for_expected_records(
    expected_records: list[dict[str, Any]],
    *,
    papers_dir: str | Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for record in expected_records:
        path = _store_path(papers_dir, normalize_text(record.get("paperId")))
        if path in seen_paths:
            continue
        seen_paths.add(path)
        rows.extend(_read_jsonl(path))
    return rows


def build_parsed_artifact_evidence_chunk_candidate_full_apply_readback_review(
    *,
    full_apply_report: dict[str, Any],
    source_full_apply_report_ref: str,
    papers_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    expected_records = [
        dict(record)
        for record in list(full_apply_report.get("candidateRecords") or [])
        if isinstance(record, dict)
    ]
    store_rows = _store_rows_for_expected_records(expected_records, papers_dir=papers_dir)
    store_rows_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for store_row in store_rows:
        store_rows_by_key[_idempotency_key(store_row)].append(store_row)

    schema_violations = _source_blockers(full_apply_report, expected_records=expected_records)
    rows: list[dict[str, Any]] = []
    for index, expected in enumerate(expected_records, start=1):
        key = _idempotency_key(expected)
        matches = store_rows_by_key.get(key, [])
        stored = matches[0] if matches else {}
        expected_hash = _record_hash(expected)
        stored_hash = _record_hash(stored) if stored else ""
        checks = {
            "storeRecordPresent": bool(matches),
            "singleStoreRecordForIdempotencyKey": len(matches) == 1,
            "recordHashMatches": bool(stored) and expected_hash == stored_hash,
            "recordBytesMatchCanonicalJson": bool(stored) and expected == stored,
            "candidateRecordIdMatches": normalize_text(expected.get("candidateRecordId"))
            == normalize_text(stored.get("candidateRecordId")),
            "sourceCandidateRowIdMatches": normalize_text(expected.get("sourceCandidateRowId"))
            == normalize_text(stored.get("sourceCandidateRowId")),
            "paperIdMatches": normalize_text(expected.get("paperId")) == normalize_text(stored.get("paperId")),
            "artifactTypeMatches": normalize_text(expected.get("artifactType"))
            == normalize_text(stored.get("artifactType")),
            "sourceContentHashMatches": normalize_text(expected.get("sourceContentHash"))
            == normalize_text(stored.get("sourceContentHash")),
            "spanLocatorMatches": normalize_text(expected.get("spanLocator"))
            == normalize_text(stored.get("spanLocator")),
            "snippetHashMatches": normalize_text(expected.get("snippetHash"))
            == normalize_text(stored.get("snippetHash")),
            "requiredFieldsPresent": bool(stored) and _record_required_fields_ok(stored),
            "candidatePolicyQuarantined": bool(stored) and _candidate_policy_ok(stored),
            "recordNotPrivatePathLeaking": bool(stored) and not _contains_private_path(stored),
        }
        blockers = [name for name, passed in checks.items() if not passed]
        if not matches:
            readback_status = READBACK_STATUS_BLOCKED_MISSING
        elif len(matches) != 1:
            readback_status = READBACK_STATUS_BLOCKED_DUPLICATE
        elif not checks["candidatePolicyQuarantined"] or not checks["recordNotPrivatePathLeaking"]:
            readback_status = READBACK_STATUS_BLOCKED_POLICY_VIOLATION
        elif blockers:
            readback_status = READBACK_STATUS_BLOCKED_HASH_MISMATCH
        else:
            readback_status = READBACK_STATUS_VALIDATED
        rows.append(
            {
                "readbackReviewRowId": f"parsed-artifact-evidence-chunk-candidate-full-apply-readback-review:{index:04d}",
                "candidateRecordId": normalize_text(expected.get("candidateRecordId")),
                "sourceCandidateRowId": normalize_text(expected.get("sourceCandidateRowId")),
                "paperId": normalize_text(expected.get("paperId")),
                "artifactType": normalize_text(expected.get("artifactType")),
                "sourceRef": normalize_text(expected.get("sourceRef")),
                "sourceContentHash": normalize_text(expected.get("sourceContentHash")),
                "spanLocator": normalize_text(expected.get("spanLocator")),
                "snippetHash": normalize_text(expected.get("snippetHash")),
                "candidateStoreRef": f"papers_dir/structured_evidence_candidates/evidence_chunk/{normalize_text(expected.get('paperId'))}.jsonl",
                "idempotencyKey": key,
                "expectedRecordSha256": expected_hash,
                "storedRecordSha256": stored_hash,
                "matchingStoreRecordRows": len(matches),
                "readbackValidated": readback_status == READBACK_STATUS_VALIDATED,
                "candidateOnly": True,
                "answerEvidenceCandidate": True,
                "answerabilityCandidate": True,
                "strictEvidence": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "answerVisible": False,
                "readbackStatus": readback_status,
                "readbackBlockers": sorted(set(blockers)),
                "checks": checks,
            }
        )

    expected_keys = {_idempotency_key(record) for record in expected_records}
    matching_store_rows = sum(1 for row in store_rows if _idempotency_key(row) in expected_keys)
    private_path_leak_rows = 1 if _contains_private_path(rows) else 0
    if private_path_leak_rows:
        schema_violations.append("private_path_leak")
    schema_violations = sorted(set(schema_violations))
    by_status = Counter(row["readbackStatus"] for row in rows)
    by_type = Counter(row["artifactType"] for row in rows)
    blocked_rows = sum(1 for row in rows if row["readbackStatus"] != READBACK_STATUS_VALIDATED)
    counts = {
        "sourceAppliedRows": _int(dict(full_apply_report.get("counts") or {}).get("appliedCandidateRecordRows")),
        "expectedCandidateRows": len(expected_records),
        "storeRows": len(store_rows),
        "matchingStoreRows": matching_store_rows,
        "unmatchedStoreRows": len(store_rows) - matching_store_rows,
        "readbackValidatedRows": by_status.get(READBACK_STATUS_VALIDATED, 0),
        "missingStoreRows": by_status.get(READBACK_STATUS_BLOCKED_MISSING, 0),
        "duplicateStoreRows": by_status.get(READBACK_STATUS_BLOCKED_DUPLICATE, 0),
        "hashMismatchRows": by_status.get(READBACK_STATUS_BLOCKED_HASH_MISMATCH, 0),
        "policyViolationRows": by_status.get(READBACK_STATUS_BLOCKED_POLICY_VIOLATION, 0),
        "blockedRows": blocked_rows,
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
        "byReadbackStatus": dict(by_status),
    }
    status = "blocked" if schema_violations or blocked_rows else "ready"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READBACK_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE,
        "sourceFullApplyExecutorReport": _source_apply_report_summary(
            full_apply_report,
            report_ref=source_full_apply_report_ref,
        ),
        "input": {
            "papersDirRef": "papers_dir",
            "candidateStoreRefTemplate": "papers_dir/structured_evidence_candidates/evidence_chunk/{paperId}.jsonl",
        },
        "scope": _scope(len(expected_records)),
        "policy": _policy(),
        "counts": counts,
        "gate": {
            "readyForAnswerabilityPolicyGate": status == "ready",
            "candidateStoreWriteAllowed": False,
            "sourceSpanCreationAllowed": False,
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "answerVisibleAllowed": False,
            "schemaViolations": schema_violations,
        },
        "rows": rows,
        "warnings": [],
    }


def render_parsed_artifact_evidence_chunk_candidate_full_apply_readback_review_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Candidate Full Apply Readback Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- expectedCandidateRows: `{counts.get('expectedCandidateRows')}`",
        f"- storeRows: `{counts.get('storeRows')}`",
        f"- matchingStoreRows: `{counts.get('matchingStoreRows')}`",
        f"- readbackValidatedRows: `{counts.get('readbackValidatedRows')}`",
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
        f"- databaseMutationRows: `{counts.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{counts.get('indexMutationRows')}`",
        f"- vaultScanRows: `{counts.get('vaultScanRows')}`",
        f"- externalDownloadRows: `{counts.get('externalDownloadRows')}`",
        "",
        "## Readback Status",
        "",
    ]
    for status, count in sorted(dict(counts.get("byReadbackStatus") or {}).items()):
        lines.append(f"- `{status}`: `{count}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_candidate_full_apply_readback_review(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_candidate_full_apply_readback_review_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "BLOCKED_DECISION",
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READBACK_REVIEW_SCHEMA_ID",
    "READBACK_STATUS_BLOCKED_DUPLICATE",
    "READBACK_STATUS_BLOCKED_HASH_MISMATCH",
    "READBACK_STATUS_BLOCKED_MISSING",
    "READBACK_STATUS_BLOCKED_POLICY_VIOLATION",
    "READBACK_STATUS_VALIDATED",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_candidate_full_apply_readback_review",
    "load_json",
    "sanitized_report_ref",
    "write_parsed_artifact_evidence_chunk_candidate_full_apply_readback_review",
]
