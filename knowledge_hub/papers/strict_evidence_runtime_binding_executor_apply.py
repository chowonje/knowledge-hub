"""Apply-gated executor for StrictEvidence runtime binding JSONL records.

Consumes the runtime binding executor dry-run report, validates dry-run-ready
rows, and appends runtime binding records only when explicitly invoked with
``--apply`` and ``--papers-dir``. Runtime binding records remain non-visible to
answer paths: parent evidence stores are read-only, ``runtimeVisible`` stays
false, and parser/answer/DB/index/vault surfaces stay disabled.
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
from knowledge_hub.papers.strict_evidence_runtime_binding_executor_dry_run import (
    DRY_RUN_STATUS_READY,
    STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_DRY_RUN_SCHEMA_ID,
)
from knowledge_hub.papers.strict_evidence_runtime_binding_record_contract import (
    RUNTIME_BINDING_STORE,
    RUNTIME_BINDING_STORE_CONTRACT,
    STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID,
    STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID,
    validate_runtime_binding_record_semantics,
)


STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-runtime-binding-executor-apply.v1"
)

APPLY_STATUS_READY = "apply_ready_runtime_binding_record_only"
APPLY_STATUS_APPLIED = "applied_runtime_binding_record"
APPLY_STATUS_BLOCKED_DRY_RUN_NOT_READY = "blocked_dry_run_not_ready"
APPLY_STATUS_BLOCKED_MISSING_IDENTITY = "blocked_missing_record_identity"
APPLY_STATUS_BLOCKED_RUNTIME_OR_ANSWER = "blocked_runtime_or_answer_flag_violation"
APPLY_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
APPLY_STATUS_BLOCKED_STORE_CONTRACT = "blocked_runtime_binding_store_contract_not_ready"
APPLY_STATUS_BLOCKED_SCHEMA_VIOLATION = "blocked_runtime_binding_record_schema_violation"

DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-runtime-binding-executor-dry-run"
    / "01-strict-evidence-runtime-binding-executor-dry-run"
    / "strict-evidence-runtime-binding-executor-dry-run.json"
)

DEFAULT_CONTRACT_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-runtime-binding-record-contract"
    / "01-strict-evidence-runtime-binding-record-contract"
    / "strict-evidence-runtime-binding-record-contract.json"
)

DEFAULT_REPORT_ROOT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-runtime-binding-executor-apply"
)

DEFAULT_DRY_RUN_OUTPUT_DIR = (
    DEFAULT_REPORT_ROOT / "01-strict-evidence-runtime-binding-executor-apply-dry-run"
)
DEFAULT_APPLY_OUTPUT_DIR = DEFAULT_REPORT_ROOT / "02-strict-evidence-runtime-binding-executor-apply"
DEFAULT_OUTPUT_DIR = DEFAULT_DRY_RUN_OUTPUT_DIR

ROLLBACK_STRATEGY = RUNTIME_BINDING_STORE_CONTRACT.get(
    "rollbackStrategy",
    "delete runtime binding records written by the explicit run_id only",
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


def _runtime_binding_only_policy_matrix() -> dict[str, Any]:
    return {
        "plannedWriteTarget": RUNTIME_BINDING_STORE,
        "writeEnabled": False,
        "runtimeBindingRecordWrite": False,
        "runtimeEvidenceCreated": False,
        "runtimeVisible": False,
        "answerIntegrationVisible": False,
        "citationGradeRecordWrite": False,
        "citationGradeBooleanMutation": False,
        "eligibilityRecordWrite": False,
        "strictEvidenceStoreWrite": False,
        "sourceSpanStoreWrite": False,
        "strictEvidenceCreated": False,
        "strictEligibleMutation": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "vaultScan": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
        "manifestWrite": False,
    }


def _mutation_flag_violation(dry_row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for field_name in (
        "runtimeBindingRecordWrite",
        "runtimeEvidenceCreated",
        "runtimeVisible",
        "answerIntegrationVisible",
        "citationGradeRecordWrite",
        "citationGradeBooleanMutation",
        "eligibilityRecordWrite",
        "strictEvidenceStoreWrite",
        "sourceSpanStoreWrite",
        "strictEligibleMutation",
        "strictEvidenceCreated",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "databaseMutation",
        "vaultScan",
        "reindexOrReembed",
        "canonicalParsedArtifactsWritten",
        "manifestWrite",
    ):
        if _safe_bool(dry_row.get(field_name)):
            violations.append(f"dry_run_row.{field_name}_true")
    matrix = dry_row.get("policyMatrix") if isinstance(dry_row.get("policyMatrix"), dict) else {}
    for field_name in (
        "runtimeBindingRecordWrite",
        "runtimeEvidenceCreated",
        "runtimeVisible",
        "answerIntegrationVisible",
        "citationGradeRecordWrite",
        "citationGradeBooleanMutation",
        "eligibilityRecordWrite",
        "strictEvidenceStoreWrite",
        "sourceSpanStoreWrite",
        "strictEligibleMutation",
        "strictEvidenceCreated",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "databaseMutation",
        "vaultScan",
        "reindexOrReembed",
        "canonicalParsedArtifactsWritten",
        "manifestWrite",
    ):
        if _safe_bool(matrix.get(field_name)):
            violations.append(f"policyMatrix.{field_name}_true")
    return violations


def _contract_ready(contract_payload: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if _safe_text(contract_payload.get("schema")) != STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID:
        blockers.append("runtime_binding_contract_schema_mismatch")
    if _safe_text(contract_payload.get("status")) != "ok":
        blockers.append(
            f"runtime_binding_contract_status={_safe_text(contract_payload.get('status')) or 'unknown'}"
        )
    write_targets = contract_payload.get("writeTargets")
    if not isinstance(write_targets, list) or not write_targets:
        blockers.append("runtime_binding_contract_write_targets_missing")
    else:
        target = write_targets[0] if isinstance(write_targets[0], dict) else {}
        if _safe_text(target.get("plannedWriteTarget")) != RUNTIME_BINDING_STORE:
            blockers.append("runtime_binding_contract_write_target_mismatch")
    gate = contract_payload.get("gate") if isinstance(contract_payload.get("gate"), dict) else {}
    if _safe_text(gate.get("decision")) != "strict_evidence_runtime_binding_record_contract_ready":
        blockers.append(
            f"runtime_binding_contract_decision={_safe_text(gate.get('decision')) or 'unknown'}"
        )
    return not blockers, blockers


def _planned_record(dry_row: dict[str, Any]) -> dict[str, Any]:
    planned = dry_row.get("plannedRuntimeBindingRecord")
    return deepcopy(planned) if isinstance(planned, dict) else {}


def _row_is_apply_ready(dry_row: dict[str, Any]) -> tuple[bool, str, list[str]]:
    blockers: list[str] = []
    if _safe_text(dry_row.get("dry_run_status")) != DRY_RUN_STATUS_READY:
        blockers.append(f"dry_run_status={_safe_text(dry_row.get('dry_run_status')) or 'unknown'}")
        return False, APPLY_STATUS_BLOCKED_DRY_RUN_NOT_READY, _dedupe(blockers)

    if not _safe_bool(dry_row.get("dryRunReadyRuntimeBindingRecordOnly")):
        blockers.append("dryRunReadyRuntimeBindingRecordOnly_false")

    if _safe_text(dry_row.get("recommended_action")) != "queue_for_runtime_binding_executor_apply_dry_run_review":
        blockers.append(
            f"recommended_action={_safe_text(dry_row.get('recommended_action')) or 'unknown'}"
        )

    planned = _planned_record(dry_row)
    if not planned:
        blockers.append("plannedRuntimeBindingRecord_missing")

    if _safe_text(planned.get("plannedWriteTarget")) != RUNTIME_BINDING_STORE:
        blockers.append(
            f"plannedWriteTarget={_safe_text(planned.get('plannedWriteTarget')) or 'unknown'}"
        )

    flag_violations = _mutation_flag_violation(dry_row)
    if flag_violations:
        blockers.extend(flag_violations)

    for field_name in (
        "strictEvidenceId",
        "sourceSpanId",
        "candidateRecordId",
        "eligibilityRecordId",
        "citationGradeRecordId",
    ):
        if not _safe_text(dry_row.get(field_name)):
            blockers.append(f"{field_name}_missing")
    if not _safe_text(planned.get("runtimeBindingRecordId")):
        blockers.append("runtimeBindingRecordId_missing")
    if not _safe_text(planned.get("idempotencyKey")):
        blockers.append("idempotencyKey_missing")
    for field_name in ("runtimeVisible", "answerIntegrationVisible", "runtimeEvidence"):
        if _safe_bool(planned.get(field_name)):
            blockers.append(f"plannedRuntimeBindingRecord.{field_name}_true")

    if blockers:
        status = APPLY_STATUS_BLOCKED_MISSING_IDENTITY
        if any(
            item.startswith("dry_run_status=")
            or item == "dryRunReadyRuntimeBindingRecordOnly_false"
            or item.startswith("recommended_action=")
            for item in blockers
        ):
            status = APPLY_STATUS_BLOCKED_DRY_RUN_NOT_READY
        elif any(
            "true" in item
            or "runtime" in item
            or "answerIntegration" in item
            or "policyMatrix" in item
            or "RecordWrite" in item
            for item in blockers
        ):
            status = APPLY_STATUS_BLOCKED_RUNTIME_OR_ANSWER
        return False, status, _dedupe(blockers)

    return True, APPLY_STATUS_READY, []


def _runtime_binding_record(dry_row: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    record = _planned_record(dry_row)
    record["runId"] = run_id
    write_policy = record.get("writePolicy") if isinstance(record.get("writePolicy"), dict) else {}
    record["writePolicy"] = {
        **write_policy,
        "executorRequired": True,
        "runtimeBindingRecordWrite": False,
        "runtimeEvidenceCreated": False,
        "runtimeVisible": False,
        "answerIntegrationVisible": False,
        "citationGradeRecordWrite": False,
        "citationGradeBooleanMutation": False,
        "eligibilityRecordWrite": False,
        "strictEvidenceStoreWrite": False,
        "sourceSpanStoreWrite": False,
        "strictEligibleMutation": False,
        "databaseMutation": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
        "manifestWrite": False,
        "vaultScan": False,
    }
    record["citationGradeRecordInPlaceMutationAllowed"] = False
    record["strictEvidenceInPlaceMutationAllowed"] = False
    record["eligibilityRecordInPlaceMutationAllowed"] = False
    record["sourceSpanInPlaceMutationAllowed"] = False
    record["runtimeBindingMutationApplied"] = False
    record["runtimeEvidence"] = False
    record["runtimeVisible"] = False
    record["answerIntegrationVisible"] = False
    return record


def _record_path(papers_dir: str | Path, paper_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "structured_evidence"
        / "strict_evidence_runtime_binding"
        / f"{_safe_filename(paper_id)}.jsonl"
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
        str(record.get("idempotencyKey") or record.get("runtimeBindingRecordId")): record
        for record in records
    }
    retained: list[dict[str, Any]] = []
    for existing in _read_jsonl(path):
        key = str(existing.get("idempotencyKey") or existing.get("runtimeBindingRecordId"))
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
            str(record.get("idempotencyKey") or ""): record for record in _read_jsonl(path)
        }
        for record in path_records:
            key = _safe_text(record.get("idempotencyKey"))
            stored = readback_by_key.get(key)
            if stored == record:
                readback_rows += 1
                path_by_record_id[_safe_text(record.get("runtimeBindingRecordId"))] = str(path)
            else:
                warnings.append(f"readback_mismatch:{record.get('runtimeBindingRecordId')}")
    return applied_rows, readback_rows, _dedupe(warnings), path_by_record_id


def build_strict_evidence_runtime_binding_executor_apply(
    *,
    executor_dry_run_report_path: str | Path = DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH,
    runtime_binding_record_contract_report_path: str | Path = DEFAULT_CONTRACT_REPORT_PATH,
    papers_dir: str | Path | None = None,
    run_id: str | None = None,
    apply: bool = False,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    dry_run_path = Path(str(executor_dry_run_report_path)).expanduser()
    contract_path = Path(str(runtime_binding_record_contract_report_path)).expanduser()
    requested = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}
    run_id = _safe_text(run_id) or f"strict-evidence-runtime-binding-executor-apply-{_now_iso()}"

    warnings: list[str] = []
    schema_violations: list[str] = []

    dry_run_payload = _read_json(dry_run_path)
    contract_payload = _read_json(contract_path)

    if not dry_run_payload:
        schema_violations.append("executor_dry_run_report_missing_or_unreadable")
    else:
        validation = validate_payload(
            dry_run_payload,
            STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_DRY_RUN_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            schema_violations.extend(str(error) for error in validation.errors)

    contract_ready, contract_blockers = _contract_ready(contract_payload)
    if not contract_payload:
        schema_violations.append("runtime_binding_contract_report_missing_or_unreadable")
    elif not contract_ready:
        schema_violations.extend(contract_blockers)
    else:
        validation = validate_payload(
            contract_payload,
            STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            schema_violations.extend(str(error) for error in validation.errors)

    if apply and not papers_dir:
        schema_violations.append("apply_requires_papers_dir")

    gate = dry_run_payload.get("gate") if isinstance(dry_run_payload.get("gate"), dict) else {}
    if dry_run_payload:
        if _safe_text(dry_run_payload.get("status")) != "ok":
            schema_violations.append(
                f"executor_dry_run_report_status={_safe_text(dry_run_payload.get('status')) or 'unknown'}"
            )
        if not _safe_bool(gate.get("readyForRuntimeBindingExecutorDryRun")):
            schema_violations.append("executor_dry_run_not_ready_for_apply")

    dry_run_ready_rows = (
        int(
            (dry_run_payload.get("counts") or {}).get("dryRunReadyRuntimeBindingRecordOnlyRows")
            or 0
        )
        if dry_run_payload
        else 0
    )

    input_rows: list[dict[str, Any]] = []
    if not schema_violations:
        all_rows = _extract_rows(dry_run_payload)
        if requested:
            found = {_safe_text(row.get("paper_id")) for row in all_rows if _safe_text(row.get("paper_id"))}
            if requested - found:
                warnings.append("requested_paper_ids_not_found")
            input_rows = [row for row in all_rows if _safe_text(row.get("paper_id")) in requested]
        else:
            input_rows = all_rows
        ready_row_count = sum(
            1 for row in input_rows if _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_READY
        )
        if not input_rows:
            warnings.append("executor_dry_run_rows_missing")
            schema_violations.append("executor_dry_run_rows_missing")
        elif ready_row_count != dry_run_ready_rows:
            schema_violations.append("executor_dry_run_ready_row_count_mismatch")

    runtime_binding_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, dry_row in enumerate(input_rows):
        ready, status, blockers = _row_is_apply_ready(dry_row)
        if not contract_ready and ready:
            status = APPLY_STATUS_BLOCKED_STORE_CONTRACT
            blockers = _dedupe([*blockers, *contract_blockers])
            ready = False

        runtime_binding_record: dict[str, Any] | None = None
        runtime_binding_record_id = ""
        runtime_binding_store_path = ""
        if ready:
            runtime_binding_record = _runtime_binding_record(dry_row, run_id=run_id)
            runtime_binding_record_id = _safe_text(
                runtime_binding_record.get("runtimeBindingRecordId")
            )
            if papers_dir:
                runtime_binding_store_path = str(
                    _record_path(papers_dir, _safe_text(runtime_binding_record.get("paperId")))
                )
            validation = validate_payload(
                runtime_binding_record,
                STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID,
                strict=True,
            )
            semantic_errors = validate_runtime_binding_record_semantics(runtime_binding_record)
            if validation.ok and not semantic_errors:
                runtime_binding_records.append(runtime_binding_record)
            else:
                status = APPLY_STATUS_BLOCKED_SCHEMA_VIOLATION
                blockers.extend(str(error) for error in validation.errors)
                blockers.extend(semantic_errors)
                schema_violations.extend(
                    f"runtime_binding_record_schema_violation:{runtime_binding_record_id}:{error}"
                    for error in validation.errors
                )
                schema_violations.extend(
                    f"runtime_binding_record_semantic_violation:{runtime_binding_record_id}:{error}"
                    for error in semantic_errors
                )

        rows.append(
            {
                "apply_row_id": f"strict-evidence-runtime-binding-executor-apply:{index:04d}",
                "dry_run_row_id": _safe_text(dry_row.get("dry_run_row_id")),
                "runtime_binding_gate_design_row_id": _safe_text(
                    dry_row.get("runtime_binding_gate_design_row_id")
                ),
                "hold_row_id": _safe_text(dry_row.get("hold_row_id")),
                "readback_row_id": _safe_text(dry_row.get("readback_row_id")),
                "citation_grade_apply_row_id": _safe_text(dry_row.get("apply_row_id")),
                "strictEvidenceId": _safe_text(dry_row.get("strictEvidenceId")),
                "sourceSpanId": _safe_text(dry_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(dry_row.get("candidateRecordId")),
                "eligibilityRecordId": _safe_text(dry_row.get("eligibilityRecordId")),
                "citationGradeRecordId": _safe_text(dry_row.get("citationGradeRecordId")),
                "paper_id": _safe_text(dry_row.get("paper_id")),
                "artifact_type": _safe_text(dry_row.get("artifact_type")),
                "runtimeBindingRecordId": runtime_binding_record_id,
                "idempotencyKey": _safe_text((runtime_binding_record or {}).get("idempotencyKey")),
                "planned_write_target": RUNTIME_BINDING_STORE,
                "runtime_binding_record_schema": (
                    STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID if ready else ""
                ),
                "runtime_binding_store_path": runtime_binding_store_path,
                "would_write_runtime_binding_record": ready and not apply,
                "applied_runtime_binding_record": False,
                "readback_validated": False,
                "runtimeBindingRecordWriteRows": 0,
                "runtimeVisibleRows": 0,
                "answerIntegrationVisibleRows": 0,
                "runtimeEvidenceCreatedRows": 0,
                "citationGradeRecordWriteRows": 0,
                "citationGradeBooleanMutationRows": 0,
                "eligibilityRecordWriteRows": 0,
                "strictEvidenceWriteRows": 0,
                "strictEvidenceCreated": False,
                "strictEligibleMutation": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "canonicalParsedArtifactsWritten": False,
                "sourceSpanUpdatedRows": 0,
                "manifestWriteRows": 0,
                "apply_status": status,
                "apply_blockers": _dedupe(blockers),
                "rollback_strategy": ROLLBACK_STRATEGY if ready else "no-op",
                "rollback_eligible": ready and not apply,
                "rollback_implemented": False,
                "recommended_action": (
                    "run_runtime_binding_record_apply_with_explicit_apply_flag"
                    if ready and not apply
                    else (
                        "runtime_binding_record_apply_recorded"
                        if ready and apply
                        else "repair_dry_run_row_before_runtime_binding_executor_apply"
                    )
                ),
            }
        )

    schema_violations = _dedupe(schema_violations)
    applied_rows = 0
    readback_rows = 0
    path_by_record_id: dict[str, str] = {}
    if apply and runtime_binding_records and not schema_violations and papers_dir:
        applied_rows, readback_rows, readback_warnings, path_by_record_id = _apply_records(
            runtime_binding_records,
            papers_dir=papers_dir,
        )
        warnings.extend(readback_warnings)
        for row in rows:
            runtime_binding_record_id = _safe_text(row.get("runtimeBindingRecordId"))
            if runtime_binding_record_id in path_by_record_id:
                row["apply_status"] = APPLY_STATUS_APPLIED
                row["would_write_runtime_binding_record"] = False
                row["applied_runtime_binding_record"] = True
                row["readback_validated"] = True
                row["runtimeBindingRecordWriteRows"] = 1
                row["runtime_binding_store_path"] = path_by_record_id[runtime_binding_record_id]
                row["rollback_eligible"] = True

    counts = _count_rows(
        input_rows=input_rows,
        apply_rows=rows,
        runtime_binding_records=runtime_binding_records,
        applied_rows=applied_rows,
        readback_rows=readback_rows,
        schema_violations=schema_violations,
        apply_mode=bool(apply),
        dry_run_ready_rows=dry_run_ready_rows,
    )

    ready_input_rows = [
        row for row in input_rows if _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_READY
    ]

    status = "ok"
    if (
        schema_violations
        or not rows
        or len(runtime_binding_records) != len(ready_input_rows)
        or len(ready_input_rows) != dry_run_ready_rows
    ):
        status = "blocked"
    elif apply and (
        applied_rows != len(runtime_binding_records) or readback_rows != len(runtime_binding_records)
    ):
        status = "blocked"
        schema_violations.append("apply_readback_incomplete")

    policy_matrix = _runtime_binding_only_policy_matrix()
    return {
        "schema": STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "executorDryRunReportPath": str(dry_run_path),
            "executorDryRunSchema": _safe_text(dry_run_payload.get("schema")),
            "executorDryRunStatus": _safe_text(dry_run_payload.get("status")),
            "runtimeBindingRecordContractReportPath": str(contract_path),
            "runtimeBindingRecordContractSchema": _safe_text(contract_payload.get("schema")),
            "runtimeBindingRecordContractStatus": _safe_text(contract_payload.get("status")),
            "requestedPaperIds": sorted(requested),
            "papersDir": str(Path(str(papers_dir)).expanduser()) if papers_dir else "",
            "runId": run_id,
            "apply": bool(apply),
        },
        "counts": counts,
        "runtimeBindingOnlyPolicyMatrix": policy_matrix,
        "gate": {
            "readyForDryRunApplyPlanning": status == "ok" and not apply,
            "readyForRuntimeBindingRecordApply": status == "ok" and bool(runtime_binding_records),
            "applyMode": bool(apply),
            "runtimeBindingRecordWriteAllowed": bool(apply and papers_dir and not schema_violations),
            "runtimeVisibleAllowed": False,
            "answerIntegrationVisibleAllowed": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "citationGradeRecordWriteAllowed": False,
            "citationGradeBooleanMutationAllowed": False,
            "eligibilityRecordWriteAllowed": False,
            "strictEligibleMutationAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "sourceSpanStoreWriteAllowed": False,
            "runManifestWriteAllowed": False,
            "strictEvidenceCreated": False,
            "runtimeMutationAllowed": False,
            "rollbackImplemented": False,
            "rollbackRequiresExplicitRunId": True,
            "schemaViolations": schema_violations,
            "decision": (
                "strict_evidence_runtime_binding_executor_apply_ready"
                if status == "ok" and not apply
                else (
                    "strict_evidence_runtime_binding_executor_applied"
                    if status == "ok" and apply
                    else "strict_evidence_runtime_binding_executor_apply_blocked"
                )
            ),
            "recommendedNextTranche": (
                "strict_evidence_runtime_binding_executor_apply_readback_review"
                if status == "ok" and apply
                else "strict_evidence_runtime_binding_executor_apply_review"
            ),
        },
        "policy": {
            "reportOnly": not apply,
            "dryRunByDefault": True,
            "applyRequiredForRuntimeBindingStoreWrites": True,
            "runtimeBindingRecordWrite": bool(applied_rows),
            "runtimeVisible": False,
            "answerIntegrationVisible": False,
            "runtimeEvidenceCreated": False,
            "citationGradeRecordWrite": False,
            "citationGradeBooleanMutation": False,
            "eligibilityRecordWrite": False,
            "strictEvidenceStoreWrite": False,
            "sourceSpanStoreWrite": False,
            "strictEvidenceCreated": False,
            "strictEligibleMutation": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
            "manifestWrite": False,
        },
        "warnings": _dedupe(warnings),
        "rows": rows,
        "runtimeBindingRecords": runtime_binding_records,
    }


def _count_rows(
    *,
    input_rows: list[dict[str, Any]],
    apply_rows: list[dict[str, Any]],
    runtime_binding_records: list[dict[str, Any]],
    applied_rows: int,
    readback_rows: int,
    schema_violations: list[str],
    apply_mode: bool,
    dry_run_ready_rows: int,
) -> dict[str, Any]:
    planned_apply_rows = sum(
        1 for row in apply_rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_READY
    )
    applied_runtime_binding_rows = sum(
        1 for row in apply_rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_APPLIED
    )
    return {
        "inputRows": len(input_rows),
        "dryRunReadyRuntimeBindingRecordRows": dry_run_ready_rows,
        "plannedApplyRows": planned_apply_rows,
        "appliedRuntimeBindingRecordRows": applied_runtime_binding_rows,
        "heldInputRows": len(input_rows) - len(runtime_binding_records),
        "runtimeBindingRecordRows": len(runtime_binding_records),
        "runtimeBindingRecordWriteRows": applied_rows if apply_mode else 0,
        "readbackValidatedRows": readback_rows if apply_mode else 0,
        "blockedDryRunNotReadyRows": sum(
            1
            for row in apply_rows
            if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_DRY_RUN_NOT_READY
        ),
        "blockedMissingRecordIdentityRows": sum(
            1
            for row in apply_rows
            if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_MISSING_IDENTITY
        ),
        "blockedRuntimeOrAnswerFlagViolationRows": sum(
            1
            for row in apply_rows
            if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_RUNTIME_OR_ANSWER
        ),
        "blockedStoreContractRows": sum(
            1 for row in apply_rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_STORE_CONTRACT
        ),
        "blockedSchemaViolationRows": sum(
            1
            for row in apply_rows
            if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_SCHEMA_VIOLATION
        ),
        "blockedInputSchemaViolationRows": sum(
            1
            for row in apply_rows
            if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_INPUT_SCHEMA
        ),
        "runtimeVisibleRows": 0,
        "answerIntegrationVisibleRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "citationGradeRecordWriteRows": 0,
        "citationGradeBooleanMutationRows": 0,
        "eligibilityRecordWriteRows": 0,
        "strictEligibleMutationRows": 0,
        "strictEvidenceWriteRows": 0,
        "strictEvidenceCreatedRows": 0,
        "sourceSpanUpdatedRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "reindexOrReembedRows": 0,
        "manifestWriteRows": 0,
        "vaultScanRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaperId": dict(Counter(_safe_text(row.get("paper_id")) for row in apply_rows)),
        "byArtifactType": dict(Counter(_safe_text(row.get("artifact_type")) for row in apply_rows)),
        "byApplyStatus": dict(Counter(_safe_text(row.get("apply_status")) for row in apply_rows)),
        "byRecommendedAction": dict(Counter(_safe_text(row.get("recommended_action")) for row in apply_rows)),
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
            "runtimeBindingOnlyPolicyMatrix",
            "gate",
            "policy",
            "warnings",
            "rows",
            "runtimeBindingRecords",
        )
        if key in report
    }


def render_strict_evidence_runtime_binding_executor_apply_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    matrix = dict(report.get("runtimeBindingOnlyPolicyMatrix") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byApplyStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Strict Evidence Runtime Binding Executor Apply",
            "",
            f"- status: {report.get('status', '')}",
            f"- apply mode: {json.dumps(report.get('input', {}).get('apply'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- planned apply rows: {int(counts.get('plannedApplyRows') or 0)}",
            f"- applied runtime binding record rows: {int(counts.get('appliedRuntimeBindingRecordRows') or 0)}",
            f"- runtime binding record writes: {int(counts.get('runtimeBindingRecordWriteRows') or 0)}",
            f"- runtime visible rows: {int(counts.get('runtimeVisibleRows') or 0)}",
            f"- answer integration visible rows: {int(counts.get('answerIntegrationVisibleRows') or 0)}",
            f"- readback validated rows: {int(counts.get('readbackValidatedRows') or 0)}",
            f"- runtime evidence rows: {int(counts.get('runtimeEvidenceCreatedRows') or 0)}",
            "",
            "## Runtime-binding-only policy matrix",
            f"- planned write target: {matrix.get('plannedWriteTarget', '')}",
            f"- runtime binding record write: {json.dumps(matrix.get('runtimeBindingRecordWrite'))}",
            f"- runtime visible: {json.dumps(matrix.get('runtimeVisible'))}",
            f"- answer integration visible: {json.dumps(matrix.get('answerIntegrationVisible'))}",
            "",
            "## Apply status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- recommended next tranche: {report.get('gate', {}).get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_runtime_binding_executor_apply_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-runtime-binding-executor-apply.json"
    summary_path = root / "strict-evidence-runtime-binding-executor-apply-summary.json"
    markdown_path = root / "strict-evidence-runtime-binding-executor-apply.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_runtime_binding_executor_apply_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Dry-run or explicitly apply StrictEvidence runtime binding JSONL records "
            "without making records runtime-visible or mutating parent stores."
        )
    )
    parser.add_argument(
        "--executor-dry-run-report",
        default=str(DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH),
        help="Path to runtime binding executor dry-run JSON report.",
    )
    parser.add_argument(
        "--runtime-binding-record-contract-report",
        default=str(DEFAULT_CONTRACT_REPORT_PATH),
        help="Path to runtime binding record contract JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument(
        "--papers-dir",
        default="",
        help="Local papers_dir root. Required with --apply for runtime binding JSONL writes.",
    )
    parser.add_argument("--run-id", default="", help="Run id recorded on runtime binding records.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Append runtime binding records to structured_evidence/strict_evidence_runtime_binding/.",
    )
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
    report = build_strict_evidence_runtime_binding_executor_apply(
        executor_dry_run_report_path=args.executor_dry_run_report,
        runtime_binding_record_contract_report_path=args.runtime_binding_record_contract_report,
        papers_dir=args.papers_dir or None,
        run_id=args.run_id or None,
        apply=apply_mode,
        paper_ids=args.paper_id or None,
    )
    output_dir = resolve_output_dir(args.output_dir or None, apply=apply_mode)
    paths = write_strict_evidence_runtime_binding_executor_apply_reports(report, output_dir)
    print(f"wrote report: {paths['report']}")
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote markdown: {paths['markdown']}")
    if args.json:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "APPLY_STATUS_APPLIED",
    "APPLY_STATUS_READY",
    "DEFAULT_APPLY_OUTPUT_DIR",
    "DEFAULT_CONTRACT_REPORT_PATH",
    "DEFAULT_DRY_RUN_OUTPUT_DIR",
    "DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REPORT_ROOT",
    "RUNTIME_BINDING_STORE",
    "STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_SCHEMA_ID",
    "build_strict_evidence_runtime_binding_executor_apply",
    "default_output_dir",
    "render_strict_evidence_runtime_binding_executor_apply_markdown",
    "resolve_output_dir",
    "write_strict_evidence_runtime_binding_executor_apply_reports",
]
