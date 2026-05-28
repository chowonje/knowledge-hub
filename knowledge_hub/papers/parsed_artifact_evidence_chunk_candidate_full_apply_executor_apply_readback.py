"""Apply-gated full executor for parsed-artifact evidence chunk candidates.

The executor consumes the candidate dry-run and the full-apply readiness gate.
It writes candidate JSONL records only under explicit apply mode and validates
readback. Candidate records remain candidate-only and are not SourceSpan,
StrictEvidence, citation-grade, runtime evidence, or answer-visible payloads.
"""

from __future__ import annotations

from collections import Counter
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
    EXECUTOR_STATUS_ALREADY_CORRECT,
    EXECUTOR_STATUS_APPLIED,
    EXECUTOR_STATUS_BLOCKED_IDEMPOTENCY_CONFLICT,
    EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH,
    EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION,
    EXECUTOR_STATUS_DRY_RUN_READY,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
    _apply_records,
    _candidate_record,
    _candidate_row_ready,
    _read_jsonl,
    _record_hash,
    _store_ref,
    _store_path,
    load_json,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_dry_run import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID,
    READY_DECISION as CANDIDATE_DRY_RUN_READY_DECISION,
    ZERO_COUNTER_FIELDS,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READINESS_GATE_SCHEMA_ID,
    READY_DECISION as FULL_APPLY_READINESS_READY_DECISION,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_EXECUTOR_APPLY_READBACK_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-full-apply-executor-apply-readback.v1"
)

PENDING_APPLY_DECISION = "parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback_pending_apply"
APPLIED_DECISION = "parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback_ready"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE_PENDING = "parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback_apply"
NEXT_TRANCHE_APPLIED = "parsed_artifact_evidence_chunk_candidate_full_apply_readback_review"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _safe_filename(value: str) -> str:
    import re

    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("._") or "unknown"


def _run_manifest_ref(run_id: str) -> str:
    return f"papers_dir/structured_evidence_candidates/evidence_chunk/runs/{_safe_filename(run_id)}.json"


def _run_manifest_path(papers_dir: str | Path, run_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "structured_evidence_candidates"
        / "evidence_chunk"
        / "runs"
        / f"{_safe_filename(run_id)}.json"
    )


def _dry_run_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID:
        blockers.append("invalid_candidate_dry_run_schema")
    if report.get("status") != "ready":
        blockers.append("candidate_dry_run_not_ready")
    if report.get("decision") != CANDIDATE_DRY_RUN_READY_DECISION:
        blockers.append("candidate_dry_run_invalid_decision")
    if dict(report.get("gate") or {}).get("passed") is not True:
        blockers.append("candidate_dry_run_gate_not_passed")
    if _int(counts.get("selectedCandidateRows")) <= 0:
        blockers.append("candidate_dry_run_has_no_selected_rows")
    for field_name in ("blockedRows", "privatePathLeakRows", "schemaViolationCount", *ZERO_COUNTER_FIELDS):
        if _int(counts.get(field_name)) != 0:
            blockers.append(f"candidate_dry_run_has_{field_name}")
    if _contains_private_path(report):
        blockers.append("candidate_dry_run_has_private_path_leak")
    return sorted(set(blockers))


def _readiness_blockers(report: dict[str, Any], *, expected_candidate_rows: int) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READINESS_GATE_SCHEMA_ID:
        blockers.append("invalid_full_apply_readiness_schema")
    if report.get("status") != "ready":
        blockers.append("full_apply_readiness_not_ready")
    if report.get("decision") != FULL_APPLY_READINESS_READY_DECISION:
        blockers.append("full_apply_readiness_invalid_decision")
    if dict(report.get("gate") or {}).get("readyForFullApplyExecutor") is not True:
        blockers.append("full_apply_readiness_gate_not_ready")
    if _int(counts.get("fullApplyCandidateRows")) != int(expected_candidate_rows):
        blockers.append("full_apply_readiness_candidate_count_mismatch")
    if _int(counts.get("storeValidatedCanaryRows")) != 10:
        blockers.append("full_apply_readiness_canary_not_10")
    for field_name in (
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
            blockers.append(f"full_apply_readiness_has_{field_name}")
    if _contains_private_path(report):
        blockers.append("full_apply_readiness_has_private_path_leak")
    return sorted(set(blockers))


def _canary_records_by_source_row(canary_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    if canary_report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID:
        return by_source
    for record in list(canary_report.get("candidateRecords") or []):
        if isinstance(record, dict) and record.get("schema") == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID:
            by_source[normalize_text(record.get("sourceCandidateRowId"))] = dict(record)
    return by_source


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "decision",
            "nextRecommendedTranche",
            "input",
            "counts",
            "gate",
            "policy",
            "warnings",
            "rows",
            "candidateRecords",
        )
        if key in report
    }


def _existing_correct_record_ids(records: list[dict[str, Any]], *, papers_dir: str | Path) -> set[str]:
    correct_ids: set[str] = set()
    existing_by_path: dict[Path, dict[str, dict[str, Any]]] = {}
    for record in records:
        path = _store_path(papers_dir, normalize_text(record.get("paperId")))
        if path not in existing_by_path:
            existing_by_path[path] = {
                normalize_text(row.get("idempotencyKey")): row
                for row in _read_jsonl(path)
            }
        stored = existing_by_path[path].get(normalize_text(record.get("idempotencyKey")))
        if stored == record:
            correct_ids.add(normalize_text(record.get("candidateRecordId")))
    return correct_ids


def build_parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback(
    *,
    candidate_dry_run_report: dict[str, Any],
    source_candidate_dry_run_report_ref: str,
    full_apply_readiness_report: dict[str, Any],
    source_full_apply_readiness_report_ref: str,
    canary_apply_readback_report: dict[str, Any],
    source_canary_apply_readback_report_ref: str,
    papers_dir: str | Path | None = None,
    run_id: str | None = None,
    apply: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    run_id = normalize_text(run_id) or f"parsed-artifact-evidence-chunk-full-apply-{utc_now_iso()}"
    source_rows = [
        dict(row)
        for row in list(candidate_dry_run_report.get("candidateRows") or [])
        if isinstance(row, dict)
    ]
    schema_violations = sorted(
        set(
            [
                *_dry_run_blockers(candidate_dry_run_report),
                *_readiness_blockers(full_apply_readiness_report, expected_candidate_rows=len(source_rows)),
            ]
        )
    )
    warnings: list[str] = []
    if apply and not papers_dir:
        warnings.append("apply_requires_papers_dir")
        schema_violations.append("apply_requires_papers_dir")

    canary_records = _canary_records_by_source_row(canary_apply_readback_report)
    candidate_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(source_rows, start=1):
        ready, blockers = _candidate_row_ready(source_row)
        record: dict[str, Any] | None = None
        execution_status = EXECUTOR_STATUS_DRY_RUN_READY if ready else EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION
        source_row_id = normalize_text(source_row.get("candidateRowId"))
        reused_canary_record = False
        if ready:
            if source_row_id in canary_records:
                record = dict(canary_records[source_row_id])
                reused_canary_record = True
            else:
                record = _candidate_record(
                    source_row,
                    run_id=run_id,
                    source_report_ref=source_candidate_dry_run_report_ref,
                )
            if _contains_private_path(record):
                blockers.append("candidate_record_has_private_path_leak")
                execution_status = EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION
                record = None
            else:
                candidate_records.append(record)
        rows.append(
            {
                "executorRowId": f"parsed-artifact-evidence-chunk-candidate-full-apply-executor-apply-readback:{index:04d}",
                "sourceCandidateRowId": source_row_id,
                "candidateRecordId": normalize_text(record.get("candidateRecordId")) if record else "",
                "paperId": normalize_text(source_row.get("paperId")),
                "artifactType": normalize_text(source_row.get("artifactType")),
                "sourceRef": normalize_text(source_row.get("sourceRef")),
                "sourceContentHash": normalize_text(source_row.get("sourceContentHash")),
                "spanLocator": normalize_text(source_row.get("spanLocator")),
                "snippetHash": normalize_text(source_row.get("snippetHash")),
                "candidateStoreRef": _store_ref(normalize_text(source_row.get("paperId"))),
                "idempotencyKey": normalize_text(record.get("idempotencyKey")) if record else "",
                "candidateRecordHash": _record_hash(record) if record else "",
                "reusedCanaryRecord": reused_canary_record,
                "wouldWriteCandidateRecord": bool(record) and not apply,
                "appliedCandidateRecord": False,
                "alreadyCorrectCandidateRecord": False,
                "readbackValidated": False,
                "candidateOnly": True,
                "answerEvidenceCandidate": True,
                "answerabilityCandidate": True,
                "strictEvidence": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "answerVisible": False,
                "executionStatus": execution_status,
                "executionBlockers": sorted(set(blockers)),
            }
        )

    if not candidate_records and not schema_violations:
        warnings.append("candidate_records_missing")
        schema_violations.append("candidate_records_missing")

    applied_rows = 0
    already_correct_rows = 0
    readback_rows = 0
    manifest_write_rows = 0
    if apply and candidate_records and not schema_violations and papers_dir:
        already_correct_before_apply = _existing_correct_record_ids(candidate_records, papers_dir=papers_dir)
        applied_rows, already_correct_rows, readback_rows, apply_warnings, path_by_record_id = _apply_records(
            candidate_records,
            papers_dir=papers_dir,
        )
        warnings.extend(apply_warnings)
        conflict_record_ids = {
            warning.split(":", 1)[1]
            for warning in apply_warnings
            if warning.startswith("idempotency_conflict:")
        }
        if conflict_record_ids:
            schema_violations.append("apply_idempotency_conflict")
        if readback_rows != len(candidate_records) and not conflict_record_ids:
            schema_violations.append("apply_readback_incomplete")
        for row in rows:
            record_id = normalize_text(row.get("candidateRecordId"))
            if record_id in path_by_record_id and "apply_readback_incomplete" not in schema_violations:
                row["wouldWriteCandidateRecord"] = False
                row["readbackValidated"] = True
                if record_id in already_correct_before_apply:
                    row["alreadyCorrectCandidateRecord"] = True
                    row["executionStatus"] = EXECUTOR_STATUS_ALREADY_CORRECT
                else:
                    row["appliedCandidateRecord"] = True
                    row["executionStatus"] = EXECUTOR_STATUS_APPLIED
            elif record_id in conflict_record_ids:
                row["executionStatus"] = EXECUTOR_STATUS_BLOCKED_IDEMPOTENCY_CONFLICT
                row["executionBlockers"] = sorted(set([*row["executionBlockers"], "apply_idempotency_conflict"]))
        if schema_violations:
            for row in rows:
                if row["executionStatus"] == EXECUTOR_STATUS_DRY_RUN_READY:
                    if "apply_readback_incomplete" in schema_violations:
                        row["executionStatus"] = EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH
                        row["executionBlockers"] = sorted(set([*row["executionBlockers"], *schema_violations]))

    private_path_leak_rows = 1 if _contains_private_path(rows) or _contains_private_path(candidate_records) else 0
    if private_path_leak_rows:
        schema_violations.append("private_path_leak")
    schema_violations = sorted(set(schema_violations))

    blocked_rows = sum(1 for row in rows if str(row.get("executionStatus") or "").startswith("blocked_"))
    by_status = Counter(row["executionStatus"] for row in rows)
    by_type = Counter(row["artifactType"] for row in rows)
    counts = {
        "inputRows": len(source_rows),
        "readinessFullApplyCandidateRows": _int(dict(full_apply_readiness_report.get("counts") or {}).get("fullApplyCandidateRows")),
        "plannedApplyRows": len(candidate_records),
        "reusedCanaryRecordRows": sum(1 for row in rows if row.get("reusedCanaryRecord")),
        "applyRequestedRows": len(candidate_records) if apply else 0,
        "dryRunCandidateRecordRows": 0 if apply else len(candidate_records),
        "appliedCandidateRecordRows": (applied_rows + already_correct_rows) if apply else 0,
        "alreadyCorrectRows": already_correct_rows if apply else 0,
        "candidateStoreWriteRows": applied_rows if apply else 0,
        "readbackRows": readback_rows if apply else 0,
        "readbackValidatedRows": readback_rows if apply else 0,
        "runManifestWriteRows": manifest_write_rows,
        "blockedRows": blocked_rows,
        "blockedSchemaViolationRows": by_status.get(EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION, 0),
        "blockedReadbackMismatchRows": by_status.get(EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH, 0),
        "blockedIdempotencyConflictRows": by_status.get(EXECUTOR_STATUS_BLOCKED_IDEMPOTENCY_CONFLICT, 0),
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(schema_violations),
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
        "byArtifactType": dict(by_type),
        "byExecutionStatus": dict(by_status),
    }
    status = "blocked" if schema_violations or blocked_rows else ("applied" if apply else "ready")
    decision = BLOCKED_DECISION if status == "blocked" else (APPLIED_DECISION if apply else PENDING_APPLY_DECISION)
    report: dict[str, Any] = {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_EXECUTOR_APPLY_READBACK_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": decision,
        "nextRecommendedTranche": NEXT_TRANCHE_APPLIED if status == "applied" else NEXT_TRANCHE_PENDING,
        "input": {
            "sourceCandidateDryRunReportRef": normalize_text(source_candidate_dry_run_report_ref),
            "sourceFullApplyReadinessReportRef": normalize_text(source_full_apply_readiness_report_ref),
            "sourceCanaryApplyReadbackReportRef": normalize_text(source_canary_apply_readback_report_ref),
            "papersDirRef": "papers_dir" if papers_dir else "",
            "runId": run_id,
            "apply": bool(apply),
            "runManifestRef": _run_manifest_ref(run_id),
        },
        "sourceCandidateDryRun": {
            "schema": normalize_text(candidate_dry_run_report.get("schema")),
            "status": normalize_text(candidate_dry_run_report.get("status")),
            "decision": normalize_text(candidate_dry_run_report.get("decision")),
            "selectedCandidateRows": _int(dict(candidate_dry_run_report.get("counts") or {}).get("selectedCandidateRows")),
            "blockedRows": _int(dict(candidate_dry_run_report.get("counts") or {}).get("blockedRows")),
            "privatePathLeakRows": _int(dict(candidate_dry_run_report.get("counts") or {}).get("privatePathLeakRows")),
            "schemaViolationCount": _int(dict(candidate_dry_run_report.get("counts") or {}).get("schemaViolationCount")),
        },
        "sourceFullApplyReadiness": {
            "schema": normalize_text(full_apply_readiness_report.get("schema")),
            "status": normalize_text(full_apply_readiness_report.get("status")),
            "decision": normalize_text(full_apply_readiness_report.get("decision")),
            "fullApplyCandidateRows": _int(dict(full_apply_readiness_report.get("counts") or {}).get("fullApplyCandidateRows")),
            "storeValidatedCanaryRows": _int(dict(full_apply_readiness_report.get("counts") or {}).get("storeValidatedCanaryRows")),
            "blockedRows": _int(dict(full_apply_readiness_report.get("counts") or {}).get("blockedRows")),
            "privatePathLeakRows": _int(dict(full_apply_readiness_report.get("counts") or {}).get("privatePathLeakRows")),
            "schemaViolationCount": _int(dict(full_apply_readiness_report.get("counts") or {}).get("schemaViolationCount")),
        },
        "policy": {
            "dryRunByDefault": True,
            "applyRequiredForCandidateStoreWrites": True,
            "applyMode": bool(apply),
            "candidateStoreWrite": bool(counts["candidateStoreWriteRows"]),
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
        },
        "counts": counts,
        "gate": {
            "readyForApply": status == "ready",
            "applyMode": bool(apply),
            "candidateStoreWriteAllowed": bool(apply and papers_dir and not schema_violations),
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "schemaViolations": schema_violations,
        },
        "rows": rows,
        "candidateRecords": candidate_records,
        "warnings": sorted(set(warnings)),
    }

    if apply and status == "applied" and papers_dir:
        manifest_path = _run_manifest_path(papers_dir, run_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        report["counts"]["runManifestWriteRows"] = 1
        manifest_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return report


def render_parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    by_status = dict(counts.get("byExecutionStatus") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Candidate Full Apply Executor Apply/Readback",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- apply: `{dict(report.get('input') or {}).get('apply')}`",
        f"- inputRows: `{counts.get('inputRows')}`",
        f"- plannedApplyRows: `{counts.get('plannedApplyRows')}`",
        f"- reusedCanaryRecordRows: `{counts.get('reusedCanaryRecordRows')}`",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- alreadyCorrectRows: `{counts.get('alreadyCorrectRows')}`",
        f"- readbackValidatedRows: `{counts.get('readbackValidatedRows')}`",
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
        f"- databaseMutationRows: `{counts.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{counts.get('indexMutationRows')}`",
        f"- vaultScanRows: `{counts.get('vaultScanRows')}`",
        f"- externalDownloadRows: `{counts.get('externalDownloadRows')}`",
        "",
        "## Execution Status",
        "",
    ]
    for status, count in sorted(by_status.items()):
        lines.append(f"- `{status}`: `{count}`")
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback_markdown(report), encoding="utf-8")
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "APPLIED_DECISION",
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_EXECUTOR_APPLY_READBACK_SCHEMA_ID",
    "PENDING_APPLY_DECISION",
    "build_parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback",
    "load_json",
    "sanitized_report_ref",
    "write_parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback",
]
