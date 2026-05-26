"""Apply-gated executor for Equation Authority record JSONL writes.

Consumes the Equation Authority record executor dry-run report and appends
equation authority records only when explicitly invoked with ``--apply`` and
``--papers-dir``. The write is limited to the future candidate store under
``structured_evidence/equation_authority/``; EquationArtifact, StrictEvidence,
SourceSpan, DB/index state, vault content, parser routing, answer integration,
and authority policy mutation remain disabled.
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
from knowledge_hub.papers.equation_authority_record_contract import (
    EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID,
    EQUATION_AUTHORITY_RECORD_SCHEMA_ID,
    EQUATION_AUTHORITY_RECORD_STORE,
    EQUATION_AUTHORITY_RECORD_STORE_CONTRACT,
    validate_equation_authority_record_semantics,
)
from knowledge_hub.papers.equation_authority_record_executor_dry_run import (
    DRY_RUN_STATUS_READY,
    EQUATION_AUTHORITY_RECORD_EXECUTOR_DRY_RUN_SCHEMA_ID,
)


EQUATION_AUTHORITY_RECORD_EXECUTOR_APPLY_SCHEMA_ID = (
    "knowledge-hub.paper.equation-authority-record-executor-apply.v1"
)

APPLY_STATUS_READY = "apply_ready_equation_authority_record_only"
APPLY_STATUS_APPLIED = "applied_equation_authority_record"
APPLY_STATUS_BLOCKED_DRY_RUN_NOT_READY = "blocked_dry_run_not_ready"
APPLY_STATUS_BLOCKED_MISSING_IDENTITY = "blocked_missing_record_identity"
APPLY_STATUS_BLOCKED_RUNTIME_OR_ANSWER = "blocked_runtime_or_answer_flag_violation"
APPLY_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
APPLY_STATUS_BLOCKED_STORE_CONTRACT = "blocked_equation_authority_record_store_contract_not_ready"
APPLY_STATUS_BLOCKED_SCHEMA_VIOLATION = "blocked_equation_authority_record_schema_violation"

DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "equation-authority-record-executor-dry-run"
    / "equation-authority-record-executor-dry-run-report.json"
)

DEFAULT_CONTRACT_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "equation-authority-record-contract"
    / "equation-authority-record-contract-report.json"
)

DEFAULT_REPORT_ROOT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "equation-authority-record-executor-apply"
)

DEFAULT_DRY_RUN_OUTPUT_DIR = DEFAULT_REPORT_ROOT / "01-equation-authority-record-executor-apply-plan"
DEFAULT_APPLY_OUTPUT_DIR = DEFAULT_REPORT_ROOT / "02-equation-authority-record-executor-apply"
DEFAULT_OUTPUT_DIR = DEFAULT_DRY_RUN_OUTPUT_DIR

ROLLBACK_STRATEGY = EQUATION_AUTHORITY_RECORD_STORE_CONTRACT.get(
    "rollbackStrategy",
    "delete equation authority records written by the explicit run_id only",
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


def _equation_authority_record_only_policy_matrix() -> dict[str, Any]:
    return {
        "plannedWriteTarget": EQUATION_AUTHORITY_RECORD_STORE,
        "writeEnabled": False,
        "equationAuthorityRecordWrite": False,
        "authorityPolicyCreated": False,
        "equationIdentityPromoted": False,
        "equationHashPromoted": False,
        "equationArtifactCreated": False,
        "strictEvidenceCreated": False,
        "sourceSpanMutation": False,
        "runtimeVisible": False,
        "answerIntegrationVisible": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "indexMutation": False,
        "vaultScan": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
        "manifestWrite": False,
    }


def _mutation_flag_violations(dry_row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for field_name in (
        "equationAuthorityRecordWrite",
        "authorityPolicyCreated",
        "equationIdentityPromoted",
        "equationHashPromoted",
        "equationArtifactCreated",
        "strictEvidenceCreated",
        "sourceSpanMutated",
        "runtimeVisible",
        "answerIntegrationVisible",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "databaseMutation",
        "indexMutation",
        "vaultScan",
        "reindexOrReembed",
        "canonicalParsedArtifactsWritten",
        "manifestWrite",
    ):
        if _safe_bool(dry_row.get(field_name)):
            violations.append(f"dry_run_row.{field_name}_true")
    matrix = dry_row.get("policyMatrix") if isinstance(dry_row.get("policyMatrix"), dict) else {}
    for field_name in (
        "equationAuthorityRecordWrite",
        "authorityPolicyCreated",
        "equationIdentityPromoted",
        "equationHashPromoted",
        "equationArtifactCreated",
        "strictEvidenceCreated",
        "sourceSpanMutation",
        "runtimeVisible",
        "answerIntegrationVisible",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "databaseMutation",
        "indexMutation",
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
    if _safe_text(contract_payload.get("schema")) != EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID:
        blockers.append("equation_authority_record_contract_schema_mismatch")
    if _safe_text(contract_payload.get("status")) != "ok":
        blockers.append(
            f"equation_authority_record_contract_status={_safe_text(contract_payload.get('status')) or 'unknown'}"
        )
    write_targets = contract_payload.get("writeTargets")
    if not isinstance(write_targets, list) or not write_targets:
        blockers.append("equation_authority_record_contract_write_targets_missing")
    else:
        target = write_targets[0] if isinstance(write_targets[0], dict) else {}
        if _safe_text(target.get("plannedWriteTarget")) != EQUATION_AUTHORITY_RECORD_STORE:
            blockers.append("equation_authority_record_contract_write_target_mismatch")
    gate = contract_payload.get("gate") if isinstance(contract_payload.get("gate"), dict) else {}
    if _safe_text(gate.get("decision")) != "equation_authority_record_contract_ready":
        blockers.append(
            f"equation_authority_record_contract_decision={_safe_text(gate.get('decision')) or 'unknown'}"
        )
    return not blockers, blockers


def _planned_record(dry_row: dict[str, Any]) -> dict[str, Any]:
    planned = dry_row.get("plannedEquationAuthorityRecord")
    return deepcopy(planned) if isinstance(planned, dict) else {}


def _row_is_apply_ready(dry_row: dict[str, Any]) -> tuple[bool, str, list[str]]:
    blockers: list[str] = []
    if _safe_text(dry_row.get("dry_run_status")) != DRY_RUN_STATUS_READY:
        blockers.append(f"dry_run_status={_safe_text(dry_row.get('dry_run_status')) or 'unknown'}")
        return False, APPLY_STATUS_BLOCKED_DRY_RUN_NOT_READY, _dedupe(blockers)

    if not _safe_bool(dry_row.get("dryRunReadyEquationAuthorityRecordOnly")):
        blockers.append("dryRunReadyEquationAuthorityRecordOnly_false")
    if _safe_text(dry_row.get("recommended_action")) != "queue_for_equation_authority_record_executor_apply_review":
        blockers.append(
            f"recommended_action={_safe_text(dry_row.get('recommended_action')) or 'unknown'}"
        )
    if _safe_text(dry_row.get("plannedWriteTarget")) != EQUATION_AUTHORITY_RECORD_STORE:
        blockers.append(f"plannedWriteTarget={_safe_text(dry_row.get('plannedWriteTarget')) or 'unknown'}")

    planned = _planned_record(dry_row)
    if not planned:
        blockers.append("plannedEquationAuthorityRecord_missing")
    for field_name in (
        "equationAuthorityRecordId",
        "paperId",
        "sourceContentHash",
        "equationIdentityDigestSha256",
        "equationHashSha256",
        "idempotencyKey",
    ):
        if not _safe_text(planned.get(field_name)):
            blockers.append(f"{field_name}_missing")
    for field_name in (
        "runtimeVisible",
        "answerIntegrationVisible",
        "equationArtifactCreated",
        "strictEvidenceCreated",
        "sourceSpanMutationAllowed",
    ):
        if _safe_bool(planned.get(field_name)):
            blockers.append(f"plannedEquationAuthorityRecord.{field_name}_true")

    flag_violations = _mutation_flag_violations(dry_row)
    blockers.extend(flag_violations)
    if blockers:
        status = APPLY_STATUS_BLOCKED_MISSING_IDENTITY
        if any(
            item.startswith("dry_run_status=")
            or item == "dryRunReadyEquationAuthorityRecordOnly_false"
            or item.startswith("recommended_action=")
            or item.startswith("plannedWriteTarget=")
            for item in blockers
        ):
            status = APPLY_STATUS_BLOCKED_DRY_RUN_NOT_READY
        elif any("true" in item or "runtime" in item or "answerIntegration" in item for item in blockers):
            status = APPLY_STATUS_BLOCKED_RUNTIME_OR_ANSWER
        return False, status, _dedupe(blockers)

    return True, APPLY_STATUS_READY, []


def _equation_authority_record(dry_row: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    record = _planned_record(dry_row)
    record["runId"] = run_id
    write_policy = record.get("writePolicy") if isinstance(record.get("writePolicy"), dict) else {}
    record["writePolicy"] = {
        **write_policy,
        "executorRequired": True,
        "equationAuthorityRecordWrite": False,
        "authorityPolicyCreated": False,
        "equationIdentityPromoted": False,
        "equationHashPromoted": False,
        "equationArtifactCreated": False,
        "strictEvidenceCreated": False,
        "sourceSpanMutation": False,
        "runtimeVisible": False,
        "answerIntegrationVisible": False,
        "databaseMutation": False,
        "indexMutation": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
        "manifestWrite": False,
        "vaultScan": False,
    }
    record["runtimeVisible"] = False
    record["answerIntegrationVisible"] = False
    record["equationArtifactCreated"] = False
    record["strictEvidenceCreated"] = False
    record["sourceSpanMutationAllowed"] = False
    return record


def _record_path(papers_dir: str | Path, paper_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "structured_evidence"
        / "equation_authority"
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
        str(record.get("idempotencyKey") or record.get("equationAuthorityRecordId")): record
        for record in records
    }
    retained: list[dict[str, Any]] = []
    for existing in _read_jsonl(path):
        key = str(existing.get("idempotencyKey") or existing.get("equationAuthorityRecordId"))
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
        readback_by_key = {str(record.get("idempotencyKey") or ""): record for record in _read_jsonl(path)}
        for record in path_records:
            key = _safe_text(record.get("idempotencyKey"))
            stored = readback_by_key.get(key)
            if stored == record:
                readback_rows += 1
                path_by_record_id[_safe_text(record.get("equationAuthorityRecordId"))] = str(path)
            else:
                warnings.append(f"readback_mismatch:{record.get('equationAuthorityRecordId')}")
    return applied_rows, readback_rows, _dedupe(warnings), path_by_record_id


def build_equation_authority_record_executor_apply(
    *,
    executor_dry_run_report_path: str | Path = DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH,
    equation_authority_record_contract_report_path: str | Path = DEFAULT_CONTRACT_REPORT_PATH,
    papers_dir: str | Path | None = None,
    run_id: str | None = None,
    apply: bool = False,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    dry_run_path = Path(str(executor_dry_run_report_path)).expanduser()
    contract_path = Path(str(equation_authority_record_contract_report_path)).expanduser()
    requested = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}
    run_id = _safe_text(run_id) or f"equation-authority-record-executor-apply-{_now_iso()}"

    warnings: list[str] = []
    schema_violations: list[str] = []

    dry_run_payload = _read_json(dry_run_path)
    contract_payload = _read_json(contract_path)

    if not dry_run_payload:
        schema_violations.append("executor_dry_run_report_missing_or_unreadable")
    else:
        validation = validate_payload(
            dry_run_payload,
            EQUATION_AUTHORITY_RECORD_EXECUTOR_DRY_RUN_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            schema_violations.extend(str(error) for error in validation.errors)

    contract_ready, contract_blockers = _contract_ready(contract_payload)
    if not contract_payload:
        schema_violations.append("equation_authority_record_contract_report_missing_or_unreadable")
    elif not contract_ready:
        schema_violations.extend(contract_blockers)
    else:
        validation = validate_payload(
            contract_payload,
            EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID,
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
        if not _safe_bool(gate.get("readyForEquationAuthorityRecordExecutorDryRun")):
            schema_violations.append("executor_dry_run_not_ready_for_apply")

    all_rows = _extract_rows(dry_run_payload) if dry_run_payload else []
    if requested:
        found = {_safe_text(row.get("paper_id")) for row in all_rows if _safe_text(row.get("paper_id"))}
        if requested - found:
            warnings.append("requested_paper_ids_not_found")
        input_rows = [row for row in all_rows if _safe_text(row.get("paper_id")) in requested]
    else:
        input_rows = all_rows
    dry_run_ready_rows = sum(1 for row in input_rows if _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_READY)

    if not schema_violations:
        if not input_rows:
            warnings.append("executor_dry_run_rows_missing")
            schema_violations.append("executor_dry_run_rows_missing")
        expected_ready_rows = int((dry_run_payload.get("counts") or {}).get("dryRunReadyEquationAuthorityRecordOnlyRows") or 0)
        if not requested and dry_run_ready_rows != expected_ready_rows:
            schema_violations.append("executor_dry_run_ready_row_count_mismatch")

    equation_authority_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, dry_row in enumerate(input_rows, start=1):
        ready, status, blockers = _row_is_apply_ready(dry_row)
        if not contract_ready and ready:
            status = APPLY_STATUS_BLOCKED_STORE_CONTRACT
            blockers = _dedupe([*blockers, *contract_blockers])
            ready = False

        record: dict[str, Any] | None = None
        record_id = ""
        store_path = ""
        if ready:
            record = _equation_authority_record(dry_row, run_id=run_id)
            record_id = _safe_text(record.get("equationAuthorityRecordId"))
            if papers_dir:
                store_path = str(_record_path(papers_dir, _safe_text(record.get("paperId"))))
            validation = validate_payload(record, EQUATION_AUTHORITY_RECORD_SCHEMA_ID, strict=True)
            semantic_errors = validate_equation_authority_record_semantics(record)
            if validation.ok and not semantic_errors:
                equation_authority_records.append(record)
            else:
                status = APPLY_STATUS_BLOCKED_SCHEMA_VIOLATION
                blockers.extend(str(error) for error in validation.errors)
                blockers.extend(semantic_errors)
                schema_violations.extend(
                    f"equation_authority_record_schema_violation:{record_id}:{error}"
                    for error in validation.errors
                )
                schema_violations.extend(
                    f"equation_authority_record_semantic_violation:{record_id}:{error}"
                    for error in semantic_errors
                )

        rows.append(
            {
                "apply_row_id": f"equation-authority-record-executor-apply:{index:04d}",
                "dry_run_row_id": _safe_text(dry_row.get("dry_run_row_id")),
                "record_contract_row_id": _safe_text(dry_row.get("record_contract_row_id")),
                "contract_design_row_id": _safe_text(dry_row.get("contract_design_row_id")),
                "hash_identity_design_row_id": _safe_text(dry_row.get("hash_identity_design_row_id")),
                "readiness_audit_row_id": _safe_text(dry_row.get("readiness_audit_row_id")),
                "paper_id": _safe_text(dry_row.get("paper_id")),
                "source_candidate_id": _safe_text(dry_row.get("source_candidate_id")),
                "source_file": _safe_text(dry_row.get("source_file")),
                "equation_environment": _safe_text(dry_row.get("equation_environment")),
                "equationAuthorityRecordId": record_id,
                "idempotencyKey": _safe_text((record or {}).get("idempotencyKey")),
                "planned_write_target": EQUATION_AUTHORITY_RECORD_STORE,
                "equation_authority_record_schema": EQUATION_AUTHORITY_RECORD_SCHEMA_ID if ready else "",
                "equation_authority_record_store_path": store_path,
                "would_write_equation_authority_record": ready and not apply,
                "applied_equation_authority_record": False,
                "readback_validated": False,
                "equationAuthorityRecordWriteRows": 0,
                "authorityPolicyCreatedRows": 0,
                "equationArtifactCreatedRows": 0,
                "strictEvidenceCreatedRows": 0,
                "sourceSpanMutatedRows": 0,
                "runtimeVisibleRows": 0,
                "answerIntegrationVisibleRows": 0,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "indexMutation": False,
                "canonicalParsedArtifactsWritten": False,
                "manifestWriteRows": 0,
                "apply_status": status,
                "apply_blockers": _dedupe(blockers),
                "rollback_strategy": ROLLBACK_STRATEGY if ready else "no-op",
                "rollback_eligible": ready and not apply,
                "rollback_implemented": False,
                "recommended_action": (
                    "run_equation_authority_record_apply_with_explicit_apply_flag"
                    if ready and not apply
                    else (
                        "equation_authority_record_apply_recorded"
                        if ready and apply
                        else "repair_dry_run_row_before_equation_authority_record_apply"
                    )
                ),
            }
        )

    schema_violations = _dedupe(schema_violations)
    applied_rows = 0
    readback_rows = 0
    path_by_record_id: dict[str, str] = {}
    if apply and equation_authority_records and not schema_violations and papers_dir:
        applied_rows, readback_rows, readback_warnings, path_by_record_id = _apply_records(
            equation_authority_records,
            papers_dir=papers_dir,
        )
        warnings.extend(readback_warnings)
        for row in rows:
            record_id = _safe_text(row.get("equationAuthorityRecordId"))
            if record_id in path_by_record_id:
                row["apply_status"] = APPLY_STATUS_APPLIED
                row["would_write_equation_authority_record"] = False
                row["applied_equation_authority_record"] = True
                row["readback_validated"] = True
                row["equationAuthorityRecordWriteRows"] = 1
                row["equation_authority_record_store_path"] = path_by_record_id[record_id]
                row["rollback_eligible"] = True

    counts = _count_rows(
        input_rows=input_rows,
        apply_rows=rows,
        equation_authority_records=equation_authority_records,
        applied_rows=applied_rows,
        readback_rows=readback_rows,
        schema_violations=schema_violations,
        apply_mode=bool(apply),
        dry_run_ready_rows=dry_run_ready_rows,
    )

    ready_input_rows = [row for row in input_rows if _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_READY]
    status = "ok"
    if (
        schema_violations
        or not rows
        or len(equation_authority_records) != len(ready_input_rows)
        or len(ready_input_rows) != dry_run_ready_rows
    ):
        status = "blocked"
    elif apply and (
        applied_rows != len(equation_authority_records)
        or readback_rows != len(equation_authority_records)
    ):
        status = "blocked"
        schema_violations.append("apply_readback_incomplete")

    policy_matrix = _equation_authority_record_only_policy_matrix()
    return {
        "schema": EQUATION_AUTHORITY_RECORD_EXECUTOR_APPLY_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "executorDryRunReportPath": str(dry_run_path),
            "executorDryRunSchema": _safe_text(dry_run_payload.get("schema")),
            "executorDryRunStatus": _safe_text(dry_run_payload.get("status")),
            "equationAuthorityRecordContractReportPath": str(contract_path),
            "equationAuthorityRecordContractSchema": _safe_text(contract_payload.get("schema")),
            "equationAuthorityRecordContractStatus": _safe_text(contract_payload.get("status")),
            "requestedPaperIds": sorted(requested),
            "papersDir": str(Path(str(papers_dir)).expanduser()) if papers_dir else "",
            "runId": run_id,
            "apply": bool(apply),
        },
        "counts": counts,
        "equationAuthorityRecordOnlyPolicyMatrix": policy_matrix,
        "gate": {
            "readyForDryRunApplyPlanning": status == "ok" and not apply,
            "readyForEquationAuthorityRecordApply": status == "ok" and bool(equation_authority_records),
            "applyMode": bool(apply),
            "equationAuthorityRecordWriteAllowed": bool(apply and papers_dir and not schema_violations),
            "authorityPolicyMutationAllowed": False,
            "equationIdentityPromotionAllowed": False,
            "equationHashPromotionAllowed": False,
            "equationArtifactCreationReady": False,
            "strictEvidenceReady": False,
            "sourceSpanMutationReady": False,
            "runtimeVisibleAllowed": False,
            "answerIntegrationVisibleAllowed": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runManifestWriteAllowed": False,
            "databaseMutationAllowed": False,
            "indexMutationAllowed": False,
            "runtimeMutationAllowed": False,
            "rollbackImplemented": False,
            "rollbackRequiresExplicitRunId": True,
            "schemaViolations": schema_violations,
            "decision": (
                "equation_authority_record_executor_apply_ready"
                if status == "ok" and not apply
                else (
                    "equation_authority_record_executor_applied"
                    if status == "ok" and apply
                    else "equation_authority_record_executor_apply_blocked"
                )
            ),
            "recommendedNextTranche": (
                "equation_authority_record_apply_readback_review"
                if status == "ok" and apply
                else "equation_authority_record_executor_apply_review"
            ),
        },
        "policy": {
            "reportOnly": not apply,
            "dryRunByDefault": True,
            "applyRequiredForEquationAuthorityRecordWrites": True,
            "equationAuthorityRecordWrite": bool(applied_rows),
            "authorityPolicyCreated": False,
            "equationIdentityPromoted": False,
            "equationHashPromoted": False,
            "equationArtifactCreated": False,
            "strictEvidenceCreated": False,
            "sourceSpanMutation": False,
            "runtimeVisible": False,
            "answerIntegrationVisible": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "indexMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
            "manifestWrite": False,
        },
        "warnings": _dedupe(warnings),
        "rows": rows,
        "equationAuthorityRecords": equation_authority_records,
    }


def _count_rows(
    *,
    input_rows: list[dict[str, Any]],
    apply_rows: list[dict[str, Any]],
    equation_authority_records: list[dict[str, Any]],
    applied_rows: int,
    readback_rows: int,
    schema_violations: list[str],
    apply_mode: bool,
    dry_run_ready_rows: int,
) -> dict[str, Any]:
    planned_apply_rows = sum(1 for row in apply_rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_READY)
    applied_record_rows = sum(1 for row in apply_rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_APPLIED)
    return {
        "inputRows": len(input_rows),
        "dryRunReadyEquationAuthorityRecordRows": dry_run_ready_rows,
        "plannedApplyRows": planned_apply_rows,
        "appliedEquationAuthorityRecordRows": applied_record_rows,
        "heldInputRows": len(input_rows) - len(equation_authority_records),
        "equationAuthorityRecordRows": len(equation_authority_records),
        "equationAuthorityRecordWriteRows": applied_rows if apply_mode else 0,
        "readbackValidatedRows": readback_rows if apply_mode else 0,
        "blockedDryRunNotReadyRows": sum(
            1 for row in apply_rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_DRY_RUN_NOT_READY
        ),
        "blockedMissingRecordIdentityRows": sum(
            1 for row in apply_rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_MISSING_IDENTITY
        ),
        "blockedRuntimeOrAnswerFlagViolationRows": sum(
            1 for row in apply_rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_RUNTIME_OR_ANSWER
        ),
        "blockedStoreContractRows": sum(
            1 for row in apply_rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_STORE_CONTRACT
        ),
        "blockedSchemaViolationRows": sum(
            1 for row in apply_rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_SCHEMA_VIOLATION
        ),
        "blockedInputSchemaViolationRows": sum(
            1 for row in apply_rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_INPUT_SCHEMA
        ),
        "authorityPolicyCreatedRows": 0,
        "equationIdentityPromotedRows": 0,
        "equationHashPromotedRows": 0,
        "equationArtifactCreatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "sourceSpanMutatedRows": 0,
        "runtimeVisibleRows": 0,
        "answerIntegrationVisibleRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "manifestWriteRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaperId": dict(Counter(_safe_text(row.get("paper_id")) for row in apply_rows)),
        "byEnvironment": dict(Counter(_safe_text(row.get("equation_environment")) for row in apply_rows)),
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
            "equationAuthorityRecordOnlyPolicyMatrix",
            "gate",
            "policy",
            "warnings",
            "rows",
            "equationAuthorityRecords",
        )
        if key in report
    }


def render_equation_authority_record_executor_apply_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    matrix = dict(report.get("equationAuthorityRecordOnlyPolicyMatrix") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byApplyStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Equation Authority Record Executor Apply",
            "",
            f"- Status: `{report.get('status', '')}`",
            f"- Decision: `{report.get('gate', {}).get('decision', '')}`",
            f"- Apply mode: `{json.dumps(report.get('input', {}).get('apply'))}`",
            f"- Input rows: `{int(counts.get('inputRows') or 0)}`",
            f"- Planned apply rows: `{int(counts.get('plannedApplyRows') or 0)}`",
            f"- Applied equation authority record rows: `{int(counts.get('appliedEquationAuthorityRecordRows') or 0)}`",
            f"- Equation authority record writes: `{int(counts.get('equationAuthorityRecordWriteRows') or 0)}`",
            f"- Readback validated rows: `{int(counts.get('readbackValidatedRows') or 0)}`",
            f"- EquationArtifact created rows: `{int(counts.get('equationArtifactCreatedRows') or 0)}`",
            f"- StrictEvidence created rows: `{int(counts.get('strictEvidenceCreatedRows') or 0)}`",
            f"- SourceSpan mutated rows: `{int(counts.get('sourceSpanMutatedRows') or 0)}`",
            "",
            "## Equation-authority-record-only policy matrix",
            f"- Planned write target: `{matrix.get('plannedWriteTarget', '')}`",
            f"- Matrix write enabled: `{json.dumps(matrix.get('writeEnabled'))}`",
            f"- Runtime visible: `{json.dumps(matrix.get('runtimeVisible'))}`",
            f"- Answer integration visible: `{json.dumps(matrix.get('answerIntegrationVisible'))}`",
            "",
            "## Apply Status Breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- Recommended next tranche: `{report.get('gate', {}).get('recommendedNextTranche', '')}`",
        ]
    )


def write_equation_authority_record_executor_apply_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "equation-authority-record-executor-apply-report.json"
    summary_path = root / "equation-authority-record-executor-apply-summary.json"
    markdown_path = root / "equation-authority-record-executor-apply.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_equation_authority_record_executor_apply_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(
        description=(
            "Plan or explicitly apply Equation Authority record JSONL writes "
            "without creating EquationArtifact/StrictEvidence or runtime/answer integration."
        )
    )
    parser.add_argument(
        "--executor-dry-run-report",
        default=str(DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH),
        help="Path to equation authority record executor dry-run JSON report.",
    )
    parser.add_argument(
        "--equation-authority-record-contract-report",
        default=str(DEFAULT_CONTRACT_REPORT_PATH),
        help="Path to equation authority record contract JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument(
        "--papers-dir",
        default="",
        help="Local papers_dir root. Required with --apply for equation authority JSONL writes.",
    )
    parser.add_argument("--run-id", default="", help="Run id recorded on equation authority records.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Append equation authority records to structured_evidence/equation_authority/.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Report output directory. Defaults to an apply-plan directory without --apply, "
            "or an apply-specific directory with --apply."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    apply_mode = bool(args.apply)
    report = build_equation_authority_record_executor_apply(
        executor_dry_run_report_path=args.executor_dry_run_report,
        equation_authority_record_contract_report_path=args.equation_authority_record_contract_report,
        papers_dir=args.papers_dir or None,
        run_id=args.run_id or None,
        apply=apply_mode,
        paper_ids=args.paper_id or None,
    )
    output_dir = resolve_output_dir(args.output_dir or None, apply=apply_mode)
    paths = write_equation_authority_record_executor_apply_reports(report, output_dir)
    print(f"wrote report: {paths['report']}")
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote markdown: {paths['markdown']}")
    if args.json:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
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
    "EQUATION_AUTHORITY_RECORD_EXECUTOR_APPLY_SCHEMA_ID",
    "EQUATION_AUTHORITY_RECORD_STORE",
    "build_equation_authority_record_executor_apply",
    "default_output_dir",
    "render_equation_authority_record_executor_apply_markdown",
    "resolve_output_dir",
    "write_equation_authority_record_executor_apply_reports",
]
