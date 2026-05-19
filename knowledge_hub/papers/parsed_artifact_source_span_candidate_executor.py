"""Dry-run/apply executor for parsed-artifact SourceSpan candidate records.

The executor only writes candidate-store JSONL records when explicitly invoked
with ``apply=True``. Candidate records are not strict evidence, citation-grade
evidence, runtime evidence, parser-routing input, or answer-runtime input.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import re

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_candidate_store_contract import (
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE,
)
from knowledge_hub.papers.parsed_artifact_structured_evidence_execution_plan import (
    EXECUTION_STATUS_DRY_RUN_READY,
)
from knowledge_hub.papers.parsed_artifact_structured_evidence_write_target_contract_audit import (
    PARSED_ARTIFACT_STRUCTURED_EVIDENCE_WRITE_TARGET_CONTRACT_AUDIT_SCHEMA_ID,
)


PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_EXECUTOR_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-candidate-executor.v1"
)

EXECUTOR_STATUS_DRY_RUN_READY = "dry_run_ready_candidate_record"
EXECUTOR_STATUS_APPLIED = "applied_candidate_record"
EXECUTOR_STATUS_BLOCKED_NON_READY_INPUT = "blocked_non_ready_input_row"
EXECUTOR_STATUS_BLOCKED_UNKNOWN_WRITE_TARGET = "blocked_unknown_write_target"
EXECUTOR_STATUS_BLOCKED_UNSUPPORTED_WRITE_TARGET = "blocked_unsupported_write_target"
EXECUTOR_STATUS_BLOCKED_MISSING_REQUIRED_FIELD = "blocked_missing_required_record_field"
EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION = "blocked_schema_violation"

DEFAULT_WRITE_TARGET_AUDIT_REPORT_PATH = str(
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-structured-evidence-write-target-contract-audit"
    / "02-after-source-span-candidate-store-contract"
    / "parsed-artifact-structured-evidence-write-target-contract-audit.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-candidate-executor"
    / "01-parsed-artifact-source-span-candidate-executor"
)

NO_RUNTIME_WRITE_POLICY = {
    "executorRequired": True,
    "databaseMutation": False,
    "parserRoutingChanged": False,
    "answerIntegrationChanged": False,
    "reindexOrReembed": False,
    "canonicalParsedArtifactsWritten": False,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _normalize_bbox(value: Any) -> list[float]:
    if value is None:
        return []
    out: list[float] = []
    try:
        for item in list(value):
            out.append(float(item))
    except Exception:
        return []
    return out


def _normalize_indexes(value: Any) -> list[int]:
    if value is None:
        return []
    out: list[int] = []
    try:
        for item in list(value):
            out.append(int(item))
    except Exception:
        return []
    return out


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = _safe_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(str(path)).expanduser()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    return [dict(item) for item in rows if isinstance(item, dict)]


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("._") or "unknown-paper"


def _locator_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "page": _safe_int(row.get("page")),
        "bbox": _normalize_bbox(row.get("bbox") or row.get("selected_bbox")),
        "blockIndexes": _normalize_indexes(row.get("blockIndexes") or row.get("block_indexes")),
        "chars": {
            "start": _safe_int(row.get("chars_start")),
            "end": _safe_int(row.get("chars_end")),
        },
    }


def _idempotency_key(row: dict[str, Any]) -> str:
    payload = {
        "plannedWriteTarget": _safe_text(row.get("planned_write_target")),
        "paperId": _safe_text(row.get("paper_id")),
        "artifactType": _safe_text(row.get("artifact_type")),
        "sourceCandidateId": _safe_text(row.get("source_candidate_id")),
        "sourceContentHash": _safe_text(row.get("sourceContentHash")),
        "sourceFile": _safe_text(row.get("source_file")),
        "locator": _locator_from_row(row),
    }
    return _sha256_text(_stable_json(payload))


def _candidate_record_id(row: dict[str, Any], idempotency_key: str) -> str:
    paper_id = _safe_filename(_safe_text(row.get("paper_id")))
    artifact_type = _safe_filename(_safe_text(row.get("artifact_type")))
    return f"source-span-candidate:{paper_id}:{artifact_type}:{idempotency_key[:16]}"


def _source_span_candidate_record(row: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    key = _idempotency_key(row)
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID,
        "candidateRecordId": _candidate_record_id(row, key),
        "runId": run_id,
        "plannedWriteTarget": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE,
        "paperId": _safe_text(row.get("paper_id")),
        "artifactType": _safe_text(row.get("artifact_type")),
        "sourceCandidateId": _safe_text(row.get("source_candidate_id")),
        "sourceReadinessRowId": _safe_text(row.get("source_readiness_row_id")),
        "sourceContentHash": _safe_text(row.get("sourceContentHash")),
        "sourceFile": _safe_text(row.get("source_file")),
        "locator": _locator_from_row(row),
        "idempotencyKey": key,
        "evidenceTier": "source_span_candidate_only",
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "strictBlockers": [
            "candidate_store_record_not_strict_evidence",
            "source_span_promotion_gate_not_run",
            "runtime_integration_not_allowed",
        ],
        "writePolicy": dict(NO_RUNTIME_WRITE_POLICY),
    }


def _record_path(papers_dir: str | Path, paper_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "structured_evidence_candidates"
        / "source_span"
        / f"{_safe_filename(paper_id)}.jsonl"
    )


def _run_manifest_path(papers_dir: str | Path, run_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "structured_evidence_candidates"
        / "runs"
        / f"{_safe_filename(run_id)}.json"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_jsonl_idempotent(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming_by_key = {
        str(record.get("idempotencyKey") or record.get("candidateRecordId")): record
        for record in records
    }
    retained: list[dict[str, Any]] = []
    for existing in _read_jsonl(path):
        key = str(existing.get("idempotencyKey") or existing.get("candidateRecordId"))
        if key and key in incoming_by_key:
            continue
        retained.append(existing)
    output = retained + list(incoming_by_key.values())
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in output),
        encoding="utf-8",
    )
    return len(incoming_by_key)


def _apply_records(
    records: list[dict[str, Any]],
    *,
    papers_dir: str | Path,
) -> tuple[int, int, list[str], dict[str, str]]:
    records_by_path: dict[Path, list[dict[str, Any]]] = {}
    for record in records:
        path = _record_path(papers_dir, _safe_text(record.get("paperId")))
        records_by_path.setdefault(path, []).append(record)

    applied_rows = 0
    readback_rows = 0
    warnings: list[str] = []
    path_by_record_id: dict[str, str] = {}
    for path, path_records in sorted(records_by_path.items(), key=lambda item: str(item[0])):
        applied_rows += _write_jsonl_idempotent(path, path_records)
        readback_by_key = {
            str(record.get("idempotencyKey") or ""): record
            for record in _read_jsonl(path)
        }
        for record in path_records:
            key = _safe_text(record.get("idempotencyKey"))
            stored = readback_by_key.get(key)
            if stored == record:
                readback_rows += 1
                path_by_record_id[_safe_text(record.get("candidateRecordId"))] = str(path)
            else:
                warnings.append(f"readback_mismatch:{record.get('candidateRecordId')}")
    return applied_rows, readback_rows, _dedupe(warnings), path_by_record_id


def _row_is_source_span_candidate_ready(row: dict[str, Any]) -> tuple[bool, str, list[str]]:
    blockers: list[str] = []
    if _safe_text(row.get("execution_status")) != EXECUTION_STATUS_DRY_RUN_READY:
        return False, EXECUTOR_STATUS_BLOCKED_NON_READY_INPUT, [
            f"execution_status={_safe_text(row.get('execution_status')) or 'unknown'}"
        ]
    if not _safe_bool(row.get("write_target_contract_known")):
        return False, EXECUTOR_STATUS_BLOCKED_UNKNOWN_WRITE_TARGET, [
            "write_target_contract_unknown"
        ]
    if _safe_text(row.get("planned_write_target")) != PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE:
        return False, EXECUTOR_STATUS_BLOCKED_UNSUPPORTED_WRITE_TARGET, [
            f"unsupported_write_target={_safe_text(row.get('planned_write_target')) or 'unknown'}"
        ]
    for field_name in ("paper_id", "artifact_type", "source_candidate_id", "sourceContentHash"):
        if not _safe_text(row.get(field_name)):
            blockers.append(f"{field_name}_missing")
    artifact_type = _safe_text(row.get("artifact_type"))
    if artifact_type not in {"section", "table", "figure"}:
        blockers.append(f"unsupported_source_span_artifact_type={artifact_type or 'unknown'}")
    if blockers:
        return False, EXECUTOR_STATUS_BLOCKED_MISSING_REQUIRED_FIELD, blockers
    return True, EXECUTOR_STATUS_DRY_RUN_READY, []


def execute_parsed_artifact_source_span_candidate_executor(
    *,
    write_target_contract_audit_report: str | Path | None = DEFAULT_WRITE_TARGET_AUDIT_REPORT_PATH,
    papers_dir: str | Path | None = None,
    run_id: str | None = None,
    apply: bool = False,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    audit_path = (
        Path(str(write_target_contract_audit_report)).expanduser()
        if write_target_contract_audit_report
        else None
    )
    input_payload = _read_json(audit_path)
    input_schema = _safe_text(input_payload.get("schema"))
    requested = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}
    run_id = _safe_text(run_id) or f"parsed-artifact-source-span-candidate-executor-{_now_iso()}"

    warnings: list[str] = []
    schema_violations: list[str] = []

    if not write_target_contract_audit_report:
        warnings.append("write_target_contract_audit_report_not_provided")
        schema_violations.append("write_target_contract_audit_report_missing")
    elif not input_payload:
        warnings.append("write_target_contract_audit_report_unreadable")
        schema_violations.append("write_target_contract_audit_report_unreadable")
    elif input_schema != PARSED_ARTIFACT_STRUCTURED_EVIDENCE_WRITE_TARGET_CONTRACT_AUDIT_SCHEMA_ID:
        warnings.append("write_target_contract_audit_report_schema_mismatch")
        schema_violations.append("write_target_contract_audit_report_schema_mismatch")

    if apply and not papers_dir:
        warnings.append("apply_requires_papers_dir")
        schema_violations.append("apply_requires_papers_dir")

    input_rows: list[dict[str, Any]] = []
    if not schema_violations:
        all_rows = _extract_rows(input_payload)
        if requested:
            found = {_safe_text(row.get("paper_id")) for row in all_rows if _safe_text(row.get("paper_id"))}
            if requested - found:
                warnings.append("requested_paper_ids_not_found")
            input_rows = [row for row in all_rows if _safe_text(row.get("paper_id")) in requested]
        else:
            input_rows = all_rows
        if not input_rows:
            warnings.append("write_target_contract_audit_rows_missing")
            schema_violations.append("write_target_contract_audit_rows_missing")

    candidate_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(input_rows):
        ready, status, blockers = _row_is_source_span_candidate_ready(row)
        candidate_record: dict[str, Any] | None = None
        candidate_record_id = ""
        candidate_store_path = ""
        if ready:
            candidate_record = _source_span_candidate_record(row, run_id=run_id)
            candidate_record_id = _safe_text(candidate_record.get("candidateRecordId"))
            if papers_dir:
                candidate_store_path = str(
                    _record_path(papers_dir, _safe_text(candidate_record.get("paperId")))
                )
            validation = validate_payload(
                candidate_record,
                PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID,
                strict=True,
            )
            if validation.ok:
                candidate_records.append(candidate_record)
            else:
                status = EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION
                blockers.extend(str(error) for error in validation.errors)
                schema_violations.extend(f"candidate_record_schema_violation:{candidate_record_id}:{error}" for error in validation.errors)

        rows.append(
            {
                "executor_row_id": f"parsed-artifact-source-span-candidate-executor:{index:04d}",
                "plan_id": _safe_text(row.get("plan_id")),
                "source_readiness_row_id": _safe_text(row.get("source_readiness_row_id")),
                "candidateRecordId": candidate_record_id,
                "paper_id": _safe_text(row.get("paper_id")),
                "artifact_type": _safe_text(row.get("artifact_type")),
                "source_candidate_id": _safe_text(row.get("source_candidate_id")),
                "sourceContentHash": _safe_text(row.get("sourceContentHash")),
                "source_file": _safe_text(row.get("source_file")),
                "locator": _locator_from_row(row),
                "planned_write_target": _safe_text(row.get("planned_write_target")),
                "candidate_record_schema": (
                    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID if ready else ""
                ),
                "candidate_store_path": candidate_store_path,
                "would_write_candidate_record": ready and not apply,
                "applied_candidate_record": False,
                "readback_validated": False,
                "sourceSpanCreated": False,
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "canonicalParsedArtifactsWritten": False,
                "execution_status": status,
                "execution_blockers": _dedupe(blockers),
                "rollback_strategy": (
                    f"delete candidate record {candidate_record_id} from this run_id before promotion"
                    if ready
                    else "no-op"
                ),
            }
        )

    schema_violations = _dedupe(schema_violations)
    applied_rows = 0
    readback_rows = 0
    manifest_write_rows = 0
    path_by_record_id: dict[str, str] = {}
    if apply and candidate_records and not schema_violations and papers_dir:
        applied_rows, readback_rows, readback_warnings, path_by_record_id = _apply_records(
            candidate_records,
            papers_dir=papers_dir,
        )
        warnings.extend(readback_warnings)
        for row in rows:
            candidate_record_id = _safe_text(row.get("candidateRecordId"))
            if candidate_record_id in path_by_record_id:
                row["execution_status"] = EXECUTOR_STATUS_APPLIED
                row["would_write_candidate_record"] = False
                row["applied_candidate_record"] = True
                row["readback_validated"] = True
                row["candidate_store_path"] = path_by_record_id[candidate_record_id]

    counts = _count_rows(
        input_rows=input_rows,
        executor_rows=rows,
        candidate_records=candidate_records,
        applied_rows=applied_rows,
        readback_rows=readback_rows,
        manifest_write_rows=manifest_write_rows,
        schema_violations=schema_violations,
    )

    status = "ok"
    if schema_violations or not rows or not candidate_records:
        status = "blocked"

    report = {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_EXECUTOR_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "writeTargetContractAuditReportPath": str(audit_path) if audit_path else "",
            "writeTargetContractAuditSchema": input_schema,
            "writeTargetContractAuditStatus": _safe_text(input_payload.get("status")),
            "requestedPaperIds": sorted(requested),
            "papersDir": str(Path(str(papers_dir)).expanduser()) if papers_dir else "",
            "runId": run_id,
            "apply": bool(apply),
        },
        "counts": counts,
        "gate": {
            "readyForDryRun": status == "ok",
            "readyForApply": status == "ok" and bool(candidate_records),
            "applyMode": bool(apply),
            "candidateStoreWriteAllowed": bool(apply and papers_dir and not schema_violations),
            "sourceSpanCreated": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": schema_violations,
            "decision": (
                "parsed_artifact_source_span_candidate_executor_ready"
                if status == "ok" and not apply
                else (
                    "parsed_artifact_source_span_candidate_executor_applied"
                    if status == "ok" and apply
                    else "blocked"
                )
            ),
            "recommendedNextTranche": (
                "parsed_artifact_source_span_candidate_readback_review"
                if apply
                else "parsed_artifact_source_span_candidate_executor_apply_review"
            ),
        },
        "policy": {
            "dryRunByDefault": True,
            "applyRequiredForCandidateStoreWrites": True,
            "candidateStoreWrite": bool(applied_rows),
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "warnings": _dedupe(warnings),
        "rows": rows,
        "candidateRecords": candidate_records,
    }

    if apply and status == "ok" and papers_dir:
        manifest_path = _run_manifest_path(papers_dir, run_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_write_rows = 1
        report["counts"]["runManifestWriteRows"] = manifest_write_rows
        report["gate"]["runManifestPath"] = str(manifest_path)
        manifest = _summary_payload(report)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return report


def _count_rows(
    *,
    input_rows: list[dict[str, Any]],
    executor_rows: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    applied_rows: int,
    readback_rows: int,
    manifest_write_rows: int,
    schema_violations: list[str],
) -> dict[str, Any]:
    return {
        "inputRows": len(input_rows),
        "candidateInputRows": len(candidate_records),
        "heldInputRows": len(input_rows) - len(candidate_records),
        "dryRunCandidateRecordRows": sum(
            1 for row in executor_rows if _safe_text(row.get("execution_status")) == EXECUTOR_STATUS_DRY_RUN_READY
        ),
        "appliedCandidateRecordRows": applied_rows,
        "candidateStoreWriteRows": applied_rows,
        "readbackValidatedRows": readback_rows,
        "runManifestWriteRows": manifest_write_rows,
        "blockedNonReadyInputRows": sum(
            1 for row in executor_rows if _safe_text(row.get("execution_status")) == EXECUTOR_STATUS_BLOCKED_NON_READY_INPUT
        ),
        "blockedUnknownWriteTargetRows": sum(
            1 for row in executor_rows if _safe_text(row.get("execution_status")) == EXECUTOR_STATUS_BLOCKED_UNKNOWN_WRITE_TARGET
        ),
        "blockedUnsupportedWriteTargetRows": sum(
            1 for row in executor_rows if _safe_text(row.get("execution_status")) == EXECUTOR_STATUS_BLOCKED_UNSUPPORTED_WRITE_TARGET
        ),
        "blockedMissingRequiredFieldRows": sum(
            1 for row in executor_rows if _safe_text(row.get("execution_status")) == EXECUTOR_STATUS_BLOCKED_MISSING_REQUIRED_FIELD
        ),
        "blockedSchemaViolationRows": sum(
            1 for row in executor_rows if _safe_text(row.get("execution_status")) == EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION
        ),
        "sourceSpanCandidateRecordRows": len(candidate_records),
        "structuredEvidenceCandidateRecordRows": 0,
        "sourceSpanCreatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in executor_rows)),
        "byExecutionStatus": dict(Counter(str(row.get("execution_status") or "") for row in executor_rows)),
        "byPlannedWriteTarget": dict(Counter(str(row.get("planned_write_target") or "") for row in executor_rows)),
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
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


def render_parsed_artifact_source_span_candidate_executor_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byExecutionStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact SourceSpan Candidate Executor",
            "",
            f"- status: {report.get('status', '')}",
            f"- apply mode: {json.dumps(report.get('input', {}).get('apply'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- candidate input rows: {int(counts.get('candidateInputRows') or 0)}",
            f"- held input rows: {int(counts.get('heldInputRows') or 0)}",
            f"- dry-run candidate records: {int(counts.get('dryRunCandidateRecordRows') or 0)}",
            f"- applied candidate records: {int(counts.get('appliedCandidateRecordRows') or 0)}",
            f"- readback validated rows: {int(counts.get('readbackValidatedRows') or 0)}",
            f"- source spans created: {int(counts.get('sourceSpanCreatedRows') or 0)}",
            f"- strict evidence created: {int(counts.get('strictEvidenceCreatedRows') or 0)}",
            f"- database mutations: {int(counts.get('databaseMutationRows') or 0)}",
            "",
            "## Execution status breakdown",
            *[f"- {item}" for item in by_status],
        ]
    )


def write_parsed_artifact_source_span_candidate_executor_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-span-candidate-executor.json"
    summary_path = root / "parsed-artifact-source-span-candidate-executor-summary.json"
    markdown_path = root / "parsed-artifact-source-span-candidate-executor.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_source_span_candidate_executor_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Dry-run or explicitly apply parsed-artifact SourceSpan candidate-store "
            "JSONL records from a write-target contract audit report."
        )
    )
    parser.add_argument(
        "--write-target-contract-audit-report",
        default=DEFAULT_WRITE_TARGET_AUDIT_REPORT_PATH,
        help="Path to parsed-artifact structured-evidence write-target contract audit report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--papers-dir", default="", help="Local papers_dir root. Required with --apply.")
    parser.add_argument("--run-id", default="", help="Run id recorded into candidate records.")
    parser.add_argument("--apply", action="store_true", help="Write candidate-store JSONL records.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = execute_parsed_artifact_source_span_candidate_executor(
        write_target_contract_audit_report=args.write_target_contract_audit_report,
        papers_dir=args.papers_dir or None,
        run_id=args.run_id or None,
        apply=bool(args.apply),
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_parsed_artifact_source_span_candidate_executor_reports(report, args.output_dir)
        print(f"wrote report: {paths['report']}")
        print(f"wrote summary: {paths['summary']}")
        print(f"wrote markdown: {paths['markdown']}")

    if args.json or not args.output_dir:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_WRITE_TARGET_AUDIT_REPORT_PATH",
    "EXECUTOR_STATUS_APPLIED",
    "EXECUTOR_STATUS_DRY_RUN_READY",
    "PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_EXECUTOR_SCHEMA_ID",
    "execute_parsed_artifact_source_span_candidate_executor",
    "render_parsed_artifact_source_span_candidate_executor_markdown",
    "write_parsed_artifact_source_span_candidate_executor_reports",
]
