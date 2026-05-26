"""Apply-gated executor for parsed-artifact StrictEvidence store JSONL records.

Writes StrictEvidence records only when explicitly invoked with ``apply=True`` and
``papers_dir``. Records remain non-citation-grade and non-runtime evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_strict_evidence_executor_dry_run import (
    DRY_RUN_STATUS_READY,
    PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_DRY_RUN_SCHEMA_ID,
    RECOMMENDED_ACTION_READY,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_record_contract import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID,
    PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_STRICT_EVIDENCE_STORE,
    validate_strict_evidence_record_semantics,
)


PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_APPLY_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-strict-evidence-executor-apply.v1"
)

APPLY_STATUS_PLANNED = "planned_apply_strict_evidence"
APPLY_STATUS_APPLIED = "applied_strict_evidence"
APPLY_STATUS_BLOCKED_NON_READY_INPUT = "blocked_non_ready_dry_run_row"
APPLY_STATUS_BLOCKED_STORE_CONTRACT = "blocked_store_contract_not_ready"
APPLY_STATUS_BLOCKED_SCHEMA_VIOLATION = "blocked_strict_evidence_record_schema_violation"

DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-executor-dry-run"
    / "02-parsed-artifact-strict-evidence-executor-dry-run-repaired-design-packet"
    / "parsed-artifact-strict-evidence-executor-dry-run.json"
)

DEFAULT_CONTRACT_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-record-contract"
    / "01-parsed-artifact-strict-evidence-record-contract"
    / "parsed-artifact-strict-evidence-record-contract.json"
)

DEFAULT_REPORT_ROOT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-executor-apply"
)

DEFAULT_DRY_RUN_OUTPUT_DIR = DEFAULT_REPORT_ROOT / "01-parsed-artifact-strict-evidence-executor-apply-dry-run"
DEFAULT_APPLY_OUTPUT_DIR = DEFAULT_REPORT_ROOT / "02-parsed-artifact-strict-evidence-executor-apply"

DEFAULT_OUTPUT_DIR = DEFAULT_DRY_RUN_OUTPUT_DIR

NO_RUNTIME_WRITE_POLICY = {
    "executorRequired": True,
    "sourceSpanStoreWrite": False,
    "databaseMutation": False,
    "parserRoutingChanged": False,
    "answerIntegrationChanged": False,
    "reindexOrReembed": False,
    "canonicalParsedArtifactsWritten": False,
}

ROLLBACK_STRATEGY = (
    "delete strict_evidence records written by the explicit run_id only while no "
    "downstream citation-grade or runtime evidence record references them"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_bool(value: Any) -> bool:
    return bool(value)


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


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("._") or "unknown-paper"


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


def default_output_dir(*, apply: bool) -> Path:
    return DEFAULT_APPLY_OUTPUT_DIR if apply else DEFAULT_DRY_RUN_OUTPUT_DIR


def resolve_output_dir(output_dir: str | Path | None, *, apply: bool) -> Path:
    if output_dir is not None and str(output_dir).strip():
        return Path(str(output_dir)).expanduser()
    return default_output_dir(apply=apply)


def _mutation_flag_violation(row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if _safe_bool(row.get("strictEvidenceCreated")):
        violations.append("strictEvidenceCreated_true")
    if int(row.get("strictEvidenceWriteRows") or 0) > 0:
        violations.append("strictEvidenceWriteRows_non_zero")
    if int(row.get("sourceSpanUpdatedRows") or 0) > 0:
        violations.append("sourceSpanUpdatedRows_non_zero")
    for field_name in (
        "citationGrade",
        "runtimeEvidence",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "databaseMutation",
        "canonicalParsedArtifactsWritten",
    ):
        if _safe_bool(row.get(field_name)):
            violations.append(f"{field_name}_true")
    write_matrix = row.get("writeMatrix")
    if isinstance(write_matrix, dict):
        if _safe_bool(write_matrix.get("strictEvidenceStoreWrite")):
            violations.append("writeMatrix_strictEvidenceStoreWrite_true")
        if _safe_bool(write_matrix.get("sourceSpanStoreWrite")):
            violations.append("writeMatrix_sourceSpanStoreWrite_true")
    return violations


def _contract_ready(contract_payload: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if _safe_text(contract_payload.get("schema")) != PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID:
        blockers.append("strict_evidence_contract_schema_mismatch")
    if _safe_text(contract_payload.get("status")) != "ok":
        blockers.append(
            f"strict_evidence_contract_status={_safe_text(contract_payload.get('status')) or 'unknown'}"
        )
    write_targets = contract_payload.get("writeTargets")
    if not isinstance(write_targets, list) or not write_targets:
        blockers.append("strict_evidence_contract_write_targets_missing")
    else:
        target = write_targets[0] if isinstance(write_targets[0], dict) else {}
        if _safe_text(target.get("plannedWriteTarget")) != PARSED_ARTIFACT_STRICT_EVIDENCE_STORE:
            blockers.append("strict_evidence_contract_write_target_mismatch")
    return not blockers, blockers


def _planned_record(dry_row: dict[str, Any]) -> dict[str, Any]:
    planned = dry_row.get("plannedStrictEvidenceRecord")
    return deepcopy(planned) if isinstance(planned, dict) else {}


def _row_is_apply_ready(dry_row: dict[str, Any]) -> tuple[bool, str, list[str]]:
    blockers: list[str] = []
    if _safe_text(dry_row.get("dry_run_status")) != DRY_RUN_STATUS_READY:
        blockers.append(
            f"dry_run_status={_safe_text(dry_row.get('dry_run_status')) or 'unknown'}"
        )
        return False, APPLY_STATUS_BLOCKED_NON_READY_INPUT, _dedupe(blockers)

    if _safe_text(dry_row.get("recommended_action")) != RECOMMENDED_ACTION_READY:
        blockers.append(
            f"recommended_action={_safe_text(dry_row.get('recommended_action')) or 'unknown'}"
        )

    planned = _planned_record(dry_row)
    if not planned:
        blockers.append("plannedStrictEvidenceRecord_missing")

    if _safe_text(planned.get("plannedWriteTarget")) != PARSED_ARTIFACT_STRICT_EVIDENCE_STORE:
        blockers.append(
            f"plannedWriteTarget={_safe_text(planned.get('plannedWriteTarget')) or 'unknown'}"
        )

    blockers.extend(_mutation_flag_violation(dry_row))

    for field_name in ("strictEvidenceId", "idempotencyKey"):
        if not _safe_text(planned.get(field_name)):
            blockers.append(f"{field_name}_missing")

    authority = planned.get("authority") if isinstance(planned.get("authority"), dict) else {}
    chars = authority.get("chars") if isinstance(authority.get("chars"), dict) else {}
    for field_name in ("start", "end", "basis", "normalization", "expectedSubstringSha256"):
        if field_name in ("start", "end"):
            if chars.get(field_name) is None:
                blockers.append(f"authority_chars_{field_name}_missing")
        elif not _safe_text(chars.get(field_name)):
            blockers.append(f"authority_chars_{field_name}_missing")

    if blockers:
        return False, APPLY_STATUS_BLOCKED_NON_READY_INPUT, _dedupe(blockers)
    return True, APPLY_STATUS_PLANNED, []


def _strict_evidence_record(dry_row: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    record = _planned_record(dry_row)
    record["runId"] = run_id
    record["writePolicy"] = dict(NO_RUNTIME_WRITE_POLICY)
    record["strictEligible"] = False
    record["citationGrade"] = False
    record["runtimeEvidence"] = False
    return record


def _record_path(papers_dir: str | Path, paper_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "structured_evidence"
        / "strict_evidence"
        / f"{_safe_filename(paper_id)}.jsonl"
    )


def _run_manifest_path(papers_dir: str | Path, run_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "structured_evidence"
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
        str(record.get("idempotencyKey") or record.get("strictEvidenceId")): record
        for record in records
    }
    retained: list[dict[str, Any]] = []
    for existing in _read_jsonl(path):
        key = str(existing.get("idempotencyKey") or existing.get("strictEvidenceId"))
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
    path_by_strict_evidence_id: dict[str, str] = {}
    for path, path_records in sorted(records_by_path.items(), key=lambda item: str(item[0])):
        applied_rows += _write_jsonl_idempotent(path, path_records)
        readback_by_key = {
            str(record.get("idempotencyKey") or ""): record for record in _read_jsonl(path)
        }
        for record in path_records:
            key = _safe_text(record.get("idempotencyKey"))
            stored = readback_by_key.get(key)
            if stored == record:
                readback_rows += 1
                path_by_strict_evidence_id[_safe_text(record.get("strictEvidenceId"))] = str(path)
            else:
                warnings.append(f"readback_mismatch:{record.get('strictEvidenceId')}")
    return applied_rows, readback_rows, _dedupe(warnings), path_by_strict_evidence_id


def execute_parsed_artifact_strict_evidence_executor_apply(
    *,
    executor_dry_run_report: str | Path = DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH,
    contract_report: str | Path = DEFAULT_CONTRACT_REPORT_PATH,
    papers_dir: str | Path | None = None,
    run_id: str | None = None,
    apply: bool = False,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    dry_run_path = Path(str(executor_dry_run_report)).expanduser()
    contract_path = Path(str(contract_report)).expanduser()
    requested = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}
    run_id = _safe_text(run_id) or f"parsed-artifact-strict-evidence-executor-apply-{_now_iso()}"

    warnings: list[str] = []
    schema_violations: list[str] = []

    dry_run_payload = _read_json(dry_run_path)
    contract_payload = _read_json(contract_path)

    if not dry_run_payload:
        schema_violations.append("executor_dry_run_report_missing_or_unreadable")
    else:
        validation = validate_payload(
            dry_run_payload,
            PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_DRY_RUN_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            schema_violations.extend(str(error) for error in validation.errors)

    contract_ready, contract_blockers = _contract_ready(contract_payload)
    if not contract_payload:
        schema_violations.append("strict_evidence_contract_report_missing_or_unreadable")
    elif not contract_ready:
        schema_violations.extend(contract_blockers)
    else:
        validation = validate_payload(
            contract_payload,
            PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            schema_violations.extend(str(error) for error in validation.errors)

    if apply and not papers_dir:
        schema_violations.append("apply_requires_papers_dir")

    gate = dry_run_payload.get("gate") if isinstance(dry_run_payload.get("gate"), dict) else {}
    if dry_run_payload and not _safe_bool(gate.get("readyForStrictEvidenceExecutorApply")):
        schema_violations.append("executor_dry_run_not_ready_for_apply")

    input_rows: list[dict[str, Any]] = []
    if not schema_violations:
        all_rows = _extract_rows(dry_run_payload)
        ready_rows = [
            row
            for row in all_rows
            if _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_READY
        ]
        if requested:
            found = {_safe_text(row.get("paper_id")) for row in ready_rows if _safe_text(row.get("paper_id"))}
            if requested - found:
                warnings.append("requested_paper_ids_not_found")
            input_rows = [row for row in ready_rows if _safe_text(row.get("paper_id")) in requested]
        else:
            input_rows = ready_rows
        if not input_rows:
            warnings.append("executor_dry_run_ready_rows_missing")
            schema_violations.append("executor_dry_run_ready_rows_missing")

    strict_evidence_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, dry_row in enumerate(input_rows):
        ready, status, blockers = _row_is_apply_ready(dry_row)
        if not contract_ready and ready:
            status = APPLY_STATUS_BLOCKED_STORE_CONTRACT
            blockers = _dedupe([*blockers, *contract_blockers])
            ready = False

        strict_evidence_record: dict[str, Any] | None = None
        strict_evidence_id = ""
        strict_evidence_store_path = ""
        if ready:
            strict_evidence_record = _strict_evidence_record(dry_row, run_id=run_id)
            strict_evidence_id = _safe_text(strict_evidence_record.get("strictEvidenceId"))
            if papers_dir:
                strict_evidence_store_path = str(
                    _record_path(papers_dir, _safe_text(strict_evidence_record.get("paperId")))
                )
            validation = validate_payload(
                strict_evidence_record,
                PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID,
                strict=True,
            )
            semantic_errors = validate_strict_evidence_record_semantics(strict_evidence_record)
            if validation.ok and not semantic_errors:
                strict_evidence_records.append(strict_evidence_record)
            else:
                status = APPLY_STATUS_BLOCKED_SCHEMA_VIOLATION
                blockers.extend(str(error) for error in validation.errors)
                blockers.extend(semantic_errors)
                schema_violations.extend(
                    f"strict_evidence_record_schema_violation:{strict_evidence_id}:{error}"
                    for error in validation.errors
                )
                schema_violations.extend(
                    f"strict_evidence_record_semantic_violation:{strict_evidence_id}:{error}"
                    for error in semantic_errors
                )

        rows.append(
            {
                "apply_row_id": f"parsed-artifact-strict-evidence-executor-apply:{index:04d}",
                "dry_run_row_id": _safe_text(dry_row.get("dry_run_row_id")),
                "packet_review_row_id": _safe_text(dry_row.get("packet_review_row_id")),
                "strictEvidenceId": strict_evidence_id,
                "sourceSpanId": _safe_text(dry_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(dry_row.get("candidateRecordId")),
                "paper_id": _safe_text(dry_row.get("paper_id")),
                "artifact_type": _safe_text(dry_row.get("artifact_type")),
                "idempotencyKey": _safe_text(
                    (strict_evidence_record or {}).get("idempotencyKey")
                ),
                "planned_write_target": PARSED_ARTIFACT_STRICT_EVIDENCE_STORE,
                "strict_evidence_record_schema": (
                    PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID if ready else ""
                ),
                "strict_evidence_store_path": strict_evidence_store_path,
                "would_write_strict_evidence_record": ready and not apply,
                "applied_strict_evidence_record": False,
                "readback_validated": False,
                "strictEvidenceCreated": False,
                "strictEvidenceWriteRows": 0,
                "sourceSpanUpdatedRows": 0,
                "citationGrade": False,
                "runtimeEvidence": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "canonicalParsedArtifactsWritten": False,
                "apply_status": status,
                "apply_blockers": _dedupe(blockers),
                "rollback_strategy": ROLLBACK_STRATEGY if ready else "no-op",
                "rollback_eligible": ready and not apply,
                "rollback_implemented": False,
            }
        )

    schema_violations = _dedupe(schema_violations)
    applied_rows = 0
    readback_rows = 0
    manifest_write_rows = 0
    path_by_strict_evidence_id: dict[str, str] = {}
    if apply and strict_evidence_records and not schema_violations and papers_dir:
        applied_rows, readback_rows, readback_warnings, path_by_strict_evidence_id = _apply_records(
            strict_evidence_records,
            papers_dir=papers_dir,
        )
        warnings.extend(readback_warnings)
        for row in rows:
            strict_evidence_id = _safe_text(row.get("strictEvidenceId"))
            if strict_evidence_id in path_by_strict_evidence_id:
                row["apply_status"] = APPLY_STATUS_APPLIED
                row["would_write_strict_evidence_record"] = False
                row["applied_strict_evidence_record"] = True
                row["readback_validated"] = True
                row["strictEvidenceCreated"] = True
                row["strictEvidenceWriteRows"] = 1
                row["strict_evidence_store_path"] = path_by_strict_evidence_id[strict_evidence_id]
                row["rollback_eligible"] = True

    counts = _count_rows(
        input_rows=input_rows,
        apply_rows=rows,
        strict_evidence_records=strict_evidence_records,
        applied_rows=applied_rows,
        readback_rows=readback_rows,
        manifest_write_rows=manifest_write_rows,
        schema_violations=schema_violations,
        apply_mode=bool(apply),
    )

    status = "ok"
    if schema_violations or not rows or len(strict_evidence_records) != len(input_rows):
        status = "blocked"
    elif apply and (
        applied_rows != len(strict_evidence_records) or readback_rows != len(strict_evidence_records)
    ):
        status = "blocked"
        schema_violations.append("apply_readback_incomplete")

    report = {
        "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_APPLY_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "executorDryRunReportPath": str(dry_run_path),
            "executorDryRunSchema": _safe_text(dry_run_payload.get("schema")),
            "executorDryRunStatus": _safe_text(dry_run_payload.get("status")),
            "contractReportPath": str(contract_path),
            "contractSchema": _safe_text(contract_payload.get("schema")),
            "contractStatus": _safe_text(contract_payload.get("status")),
            "requestedPaperIds": sorted(requested),
            "papersDir": str(Path(str(papers_dir)).expanduser()) if papers_dir else "",
            "runId": run_id,
            "apply": bool(apply),
        },
        "counts": counts,
        "gate": {
            "readyForDryRun": status == "ok" and not apply,
            "readyForApply": status == "ok" and bool(strict_evidence_records),
            "applyMode": bool(apply),
            "strictEvidenceStoreWriteAllowed": bool(apply and papers_dir and not schema_violations),
            "rollbackImplemented": False,
            "rollbackRequiresExplicitRunId": True,
            "strictEvidenceReady": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": schema_violations,
            "decision": (
                "parsed_artifact_strict_evidence_executor_apply_ready"
                if status == "ok" and not apply
                else (
                    "parsed_artifact_strict_evidence_executor_applied"
                    if status == "ok" and apply
                    else "blocked"
                )
            ),
            "recommendedNextTranche": (
                "parsed_artifact_strict_evidence_promotion_readback_review"
                if apply and status == "ok"
                else "parsed_artifact_strict_evidence_executor_apply_review"
            ),
        },
        "policy": {
            "dryRunByDefault": True,
            "applyRequiredForStrictEvidenceStoreWrites": True,
            "strictEvidenceStoreWrite": bool(applied_rows),
            "sourceSpanStoreWrite": False,
            "designPacketReviewReportMutated": False,
            "normalizationHashRepairReportMutated": False,
            "strictEvidenceCreated": bool(applied_rows),
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
        "strictEvidenceRecords": strict_evidence_records,
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
    apply_rows: list[dict[str, Any]],
    strict_evidence_records: list[dict[str, Any]],
    applied_rows: int,
    readback_rows: int,
    manifest_write_rows: int,
    schema_violations: list[str],
    apply_mode: bool,
) -> dict[str, Any]:
    planned_apply_rows = sum(
        1 for row in apply_rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_PLANNED
    )
    strict_evidence_created_rows = readback_rows if apply_mode else 0
    return {
        "inputRows": len(input_rows),
        "plannedApplyRows": planned_apply_rows,
        "heldInputRows": len(input_rows) - len(strict_evidence_records),
        "strictEvidenceRecordRows": len(strict_evidence_records),
        "strictEvidenceWriteRows": applied_rows,
        "readbackValidatedRows": readback_rows,
        "runManifestWriteRows": manifest_write_rows,
        "blockedNonReadyInputRows": sum(
            1
            for row in apply_rows
            if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_NON_READY_INPUT
        ),
        "blockedStoreContractRows": sum(
            1
            for row in apply_rows
            if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_STORE_CONTRACT
        ),
        "blockedSchemaViolationRows": sum(
            1
            for row in apply_rows
            if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_SCHEMA_VIOLATION
        ),
        "strictEvidenceCreatedRows": strict_evidence_created_rows,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "sourceSpanUpdatedRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaperId": dict(Counter(str(row.get("paper_id") or "") for row in apply_rows)),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in apply_rows)),
        "byApplyStatus": dict(Counter(str(row.get("apply_status") or "") for row in apply_rows)),
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
            "strictEvidenceRecords",
        )
        if key in report
    }


def render_parsed_artifact_strict_evidence_executor_apply_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byApplyStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact StrictEvidence Executor Apply",
            "",
            f"- status: {report.get('status', '')}",
            f"- apply mode: {json.dumps(report.get('input', {}).get('apply'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- planned apply rows: {int(counts.get('plannedApplyRows') or 0)}",
            f"- strict evidence write rows: {int(counts.get('strictEvidenceWriteRows') or 0)}",
            f"- readback validated rows: {int(counts.get('readbackValidatedRows') or 0)}",
            f"- strict evidence created: {int(counts.get('strictEvidenceCreatedRows') or 0)}",
            f"- citation-grade evidence created: {int(counts.get('citationGradeEvidenceCreatedRows') or 0)}",
            f"- runtime evidence created: {int(counts.get('runtimeEvidenceCreatedRows') or 0)}",
            f"- database mutations: {int(counts.get('databaseMutationRows') or 0)}",
            f"- rollback implemented: {json.dumps(report.get('gate', {}).get('rollbackImplemented'))}",
            "",
            "## Apply status breakdown",
            *[f"- {item}" for item in by_status],
        ]
    )


def write_parsed_artifact_strict_evidence_executor_apply_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-strict-evidence-executor-apply.json"
    summary_path = root / "parsed-artifact-strict-evidence-executor-apply-summary.json"
    markdown_path = root / "parsed-artifact-strict-evidence-executor-apply.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_strict_evidence_executor_apply_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Dry-run or explicitly apply parsed-artifact StrictEvidence store JSONL records "
            "from an executor dry-run report."
        )
    )
    parser.add_argument(
        "--executor-dry-run-report",
        default=str(DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH),
        help="Path to executor dry-run JSON report.",
    )
    parser.add_argument(
        "--contract-report",
        default=str(DEFAULT_CONTRACT_REPORT_PATH),
        help="Path to StrictEvidence record contract JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--papers-dir", default="", help="Local papers_dir root. Required with --apply.")
    parser.add_argument("--run-id", default="", help="Run id recorded into StrictEvidence records.")
    parser.add_argument("--apply", action="store_true", help="Write StrictEvidence store JSONL records.")
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Report output directory. Defaults to a dry-run-specific directory without --apply, "
            "or an apply-specific directory with --apply."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    apply_mode = bool(args.apply)
    report = execute_parsed_artifact_strict_evidence_executor_apply(
        executor_dry_run_report=args.executor_dry_run_report,
        contract_report=args.contract_report,
        papers_dir=args.papers_dir or None,
        run_id=args.run_id or None,
        apply=apply_mode,
        paper_ids=args.paper_id or None,
    )

    output_dir = resolve_output_dir(args.output_dir or None, apply=apply_mode)
    paths = write_parsed_artifact_strict_evidence_executor_apply_reports(report, output_dir)
    print(f"wrote report: {paths['report']}")
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote markdown: {paths['markdown']}")

    if args.json:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_APPLY_OUTPUT_DIR",
    "DEFAULT_DRY_RUN_OUTPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REPORT_ROOT",
    "DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH",
    "DEFAULT_CONTRACT_REPORT_PATH",
    "APPLY_STATUS_APPLIED",
    "APPLY_STATUS_PLANNED",
    "PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_APPLY_SCHEMA_ID",
    "default_output_dir",
    "execute_parsed_artifact_strict_evidence_executor_apply",
    "render_parsed_artifact_strict_evidence_executor_apply_markdown",
    "resolve_output_dir",
    "write_parsed_artifact_strict_evidence_executor_apply_reports",
]
