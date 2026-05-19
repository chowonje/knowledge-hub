"""Apply-gated executor for parsed-artifact SourceSpan store JSONL records.

Writes SourceSpan records only when explicitly invoked with ``apply=True`` and
``papers_dir``. SourceSpan records remain non-strict and non-runtime evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_promotion_executor_dry_run import (
    EXECUTOR_STATUS_PLANNED,
    PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_DRY_RUN_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_source_span_store_contract import (
    PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_SOURCE_SPAN_STORE,
    PARSED_ARTIFACT_SOURCE_SPAN_STORE_CONTRACT_SCHEMA_ID,
)


PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_APPLY_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-promotion-executor-apply.v1"
)

APPLY_STATUS_PLANNED = "planned_apply_source_span"
APPLY_STATUS_APPLIED = "applied_source_span"
APPLY_STATUS_BLOCKED_NON_READY_INPUT = "blocked_non_ready_dry_run_row"
APPLY_STATUS_BLOCKED_STORE_CONTRACT = "blocked_store_contract_not_ready"
APPLY_STATUS_BLOCKED_SCHEMA_VIOLATION = "blocked_source_span_record_schema_violation"

DEFAULT_PROMOTION_EXECUTOR_DRY_RUN_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-promotion-executor-dry-run"
    / "01-parsed-artifact-source-span-promotion-executor-dry-run"
    / "parsed-artifact-source-span-promotion-executor-dry-run.json"
)

DEFAULT_STORE_CONTRACT_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-store-contract"
    / "01-parsed-artifact-source-span-store-contract"
    / "parsed-artifact-source-span-store-contract.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-promotion-executor-apply"
    / "01-parsed-artifact-source-span-promotion-executor-apply"
)

NO_RUNTIME_WRITE_POLICY = {
    "executorRequired": True,
    "databaseMutation": False,
    "parserRoutingChanged": False,
    "answerIntegrationChanged": False,
    "reindexOrReembed": False,
    "canonicalParsedArtifactsWritten": False,
}

ROLLBACK_STRATEGY = (
    "delete records written by the explicit run_id only while no downstream "
    "strict or runtime evidence record references them"
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


def _mutation_flag_violation(row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if _safe_bool(row.get("sourceSpanPromotionApproved")):
        violations.append("sourceSpanPromotionApproved_true")
    for field_name in (
        "sourceSpanCreated",
        "strictEligible",
        "citationGrade",
        "runtimeEvidence",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "databaseMutation",
        "canonicalParsedArtifactsWritten",
    ):
        if _safe_bool(row.get(field_name)):
            violations.append(f"{field_name}_true")
    return violations


def _store_contract_ready(contract_payload: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if _safe_text(contract_payload.get("schema")) != PARSED_ARTIFACT_SOURCE_SPAN_STORE_CONTRACT_SCHEMA_ID:
        blockers.append("store_contract_schema_mismatch")
    if _safe_text(contract_payload.get("status")) != "ok":
        blockers.append(f"store_contract_status={_safe_text(contract_payload.get('status')) or 'unknown'}")
    write_targets = contract_payload.get("writeTargets")
    if not isinstance(write_targets, list) or not write_targets:
        blockers.append("store_contract_write_targets_missing")
    else:
        target = write_targets[0] if isinstance(write_targets[0], dict) else {}
        if _safe_text(target.get("plannedWriteTarget")) != PARSED_ARTIFACT_SOURCE_SPAN_STORE:
            blockers.append("store_contract_write_target_mismatch")
    return not blockers, blockers


def _row_is_apply_ready(row: dict[str, Any]) -> tuple[bool, str, list[str]]:
    blockers: list[str] = []
    if _safe_text(row.get("executor_dry_run_status")) != EXECUTOR_STATUS_PLANNED:
        blockers.append(
            f"executor_dry_run_status={_safe_text(row.get('executor_dry_run_status')) or 'unknown'}"
        )
        return False, APPLY_STATUS_BLOCKED_NON_READY_INPUT, _dedupe(blockers)
    if not _safe_bool(row.get("dryRunPlannedSourceSpan")):
        blockers.append("dryRunPlannedSourceSpan_false")
    if _safe_text(row.get("plannedWriteTarget")) != PARSED_ARTIFACT_SOURCE_SPAN_STORE:
        blockers.append(
            f"plannedWriteTarget={_safe_text(row.get('plannedWriteTarget')) or 'unknown'}"
        )
    blockers.extend(_mutation_flag_violation(row))
    for field_name in ("paper_id", "artifact_type", "source_candidate_id", "sourceContentHash", "idempotencyKey"):
        if not _safe_text(row.get(field_name)):
            blockers.append(f"{field_name}_missing")
    if not _safe_text(row.get("plannedSourceSpanId") or row.get("plannedSourceSpanKey")):
        blockers.append("plannedSourceSpanId_missing")
    if blockers:
        return False, APPLY_STATUS_BLOCKED_NON_READY_INPUT, _dedupe(blockers)
    return True, APPLY_STATUS_PLANNED, []


def _source_span_record(dry_row: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    source_span_id = _safe_text(dry_row.get("plannedSourceSpanId") or dry_row.get("plannedSourceSpanKey"))
    locator = dry_row.get("locator") if isinstance(dry_row.get("locator"), dict) else {}
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID,
        "sourceSpanId": source_span_id,
        "candidateRecordId": _safe_text(dry_row.get("candidateRecordId")),
        "runId": run_id,
        "plannedWriteTarget": PARSED_ARTIFACT_SOURCE_SPAN_STORE,
        "paperId": _safe_text(dry_row.get("paper_id")),
        "artifactType": _safe_text(dry_row.get("artifact_type")),
        "sourceCandidateId": _safe_text(dry_row.get("source_candidate_id")),
        "sourceContentHash": _safe_text(dry_row.get("sourceContentHash")),
        "sourceFile": _safe_text(dry_row.get("source_file")),
        "locator": locator,
        "idempotencyKey": _safe_text(dry_row.get("idempotencyKey")),
        "evidenceTier": "parsed_artifact_source_span",
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "strictBlockers": [
            "source_span_store_record_not_strict_evidence",
            "runtime_integration_not_allowed",
        ],
        "writePolicy": dict(NO_RUNTIME_WRITE_POLICY),
    }


def _record_path(papers_dir: str | Path, paper_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "structured_evidence"
        / "source_span"
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
        str(record.get("idempotencyKey") or record.get("sourceSpanId")): record for record in records
    }
    retained: list[dict[str, Any]] = []
    for existing in _read_jsonl(path):
        key = str(existing.get("idempotencyKey") or existing.get("sourceSpanId"))
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
    path_by_source_span_id: dict[str, str] = {}
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
                path_by_source_span_id[_safe_text(record.get("sourceSpanId"))] = str(path)
            else:
                warnings.append(f"readback_mismatch:{record.get('sourceSpanId')}")
    return applied_rows, readback_rows, _dedupe(warnings), path_by_source_span_id


def execute_parsed_artifact_source_span_promotion_executor_apply(
    *,
    promotion_executor_dry_run_report: str | Path = DEFAULT_PROMOTION_EXECUTOR_DRY_RUN_REPORT_PATH,
    store_contract_report: str | Path = DEFAULT_STORE_CONTRACT_REPORT_PATH,
    papers_dir: str | Path | None = None,
    run_id: str | None = None,
    apply: bool = False,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    dry_run_path = Path(str(promotion_executor_dry_run_report)).expanduser()
    contract_path = Path(str(store_contract_report)).expanduser()
    requested = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}
    run_id = _safe_text(run_id) or f"parsed-artifact-source-span-promotion-apply-{_now_iso()}"

    warnings: list[str] = []
    schema_violations: list[str] = []

    dry_run_payload = _read_json(dry_run_path)
    contract_payload = _read_json(contract_path)

    if not dry_run_payload:
        schema_violations.append("promotion_executor_dry_run_report_missing_or_unreadable")
    else:
        validation = validate_payload(
            dry_run_payload,
            PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_DRY_RUN_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            schema_violations.extend(str(error) for error in validation.errors)

    contract_ready, contract_blockers = _store_contract_ready(contract_payload)
    if not contract_payload:
        schema_violations.append("store_contract_report_missing_or_unreadable")
    elif not contract_ready:
        schema_violations.extend(contract_blockers)
    else:
        validation = validate_payload(
            contract_payload,
            PARSED_ARTIFACT_SOURCE_SPAN_STORE_CONTRACT_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            schema_violations.extend(str(error) for error in validation.errors)

    if apply and not papers_dir:
        schema_violations.append("apply_requires_papers_dir")

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
        if not input_rows:
            warnings.append("promotion_executor_dry_run_rows_missing")
            schema_violations.append("promotion_executor_dry_run_rows_missing")

    source_span_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, dry_row in enumerate(input_rows):
        ready, status, blockers = _row_is_apply_ready(dry_row)
        if not contract_ready and ready:
            status = APPLY_STATUS_BLOCKED_STORE_CONTRACT
            blockers = _dedupe([*blockers, *contract_blockers])
            ready = False

        source_span_record: dict[str, Any] | None = None
        source_span_id = ""
        source_span_store_path = ""
        if ready:
            source_span_record = _source_span_record(dry_row, run_id=run_id)
            source_span_id = _safe_text(source_span_record.get("sourceSpanId"))
            if papers_dir:
                source_span_store_path = str(
                    _record_path(papers_dir, _safe_text(source_span_record.get("paperId")))
                )
            validation = validate_payload(
                source_span_record,
                PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID,
                strict=True,
            )
            if validation.ok:
                source_span_records.append(source_span_record)
            else:
                status = APPLY_STATUS_BLOCKED_SCHEMA_VIOLATION
                blockers.extend(str(error) for error in validation.errors)
                schema_violations.extend(
                    f"source_span_record_schema_violation:{source_span_id}:{error}"
                    for error in validation.errors
                )

        rows.append(
            {
                "apply_row_id": f"parsed-artifact-source-span-promotion-executor-apply:{index:04d}",
                "promotion_executor_dry_run_row_id": _safe_text(
                    dry_row.get("promotion_executor_dry_run_row_id")
                ),
                "policy_gate_row_id": _safe_text(dry_row.get("policy_gate_row_id")),
                "candidateRecordId": _safe_text(dry_row.get("candidateRecordId")),
                "sourceSpanId": source_span_id,
                "paper_id": _safe_text(dry_row.get("paper_id")),
                "artifact_type": _safe_text(dry_row.get("artifact_type")),
                "source_candidate_id": _safe_text(dry_row.get("source_candidate_id")),
                "sourceContentHash": _safe_text(dry_row.get("sourceContentHash")),
                "source_file": _safe_text(dry_row.get("source_file")),
                "locator": dry_row.get("locator") if isinstance(dry_row.get("locator"), dict) else {},
                "idempotencyKey": _safe_text(dry_row.get("idempotencyKey")),
                "planned_write_target": PARSED_ARTIFACT_SOURCE_SPAN_STORE,
                "source_span_record_schema": (
                    PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID if ready else ""
                ),
                "source_span_store_path": source_span_store_path,
                "would_write_source_span_record": ready and not apply,
                "applied_source_span_record": False,
                "readback_validated": False,
                "sourceSpanCreated": False,
                "strictEligible": False,
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
    path_by_source_span_id: dict[str, str] = {}
    if apply and source_span_records and not schema_violations and papers_dir:
        applied_rows, readback_rows, readback_warnings, path_by_source_span_id = _apply_records(
            source_span_records,
            papers_dir=papers_dir,
        )
        warnings.extend(readback_warnings)
        for row in rows:
            source_span_id = _safe_text(row.get("sourceSpanId"))
            if source_span_id in path_by_source_span_id:
                row["apply_status"] = APPLY_STATUS_APPLIED
                row["would_write_source_span_record"] = False
                row["applied_source_span_record"] = True
                row["readback_validated"] = True
                row["sourceSpanCreated"] = True
                row["source_span_store_path"] = path_by_source_span_id[source_span_id]
                row["rollback_eligible"] = True

    counts = _count_rows(
        input_rows=input_rows,
        apply_rows=rows,
        source_span_records=source_span_records,
        applied_rows=applied_rows,
        readback_rows=readback_rows,
        manifest_write_rows=manifest_write_rows,
        schema_violations=schema_violations,
        apply_mode=bool(apply),
    )

    status = "ok"
    if schema_violations or not rows or len(source_span_records) != len(input_rows):
        status = "blocked"
    elif apply and (applied_rows != len(source_span_records) or readback_rows != len(source_span_records)):
        status = "blocked"
        schema_violations.append("apply_readback_incomplete")

    report = {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_APPLY_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "promotionExecutorDryRunReportPath": str(dry_run_path),
            "promotionExecutorDryRunSchema": _safe_text(dry_run_payload.get("schema")),
            "promotionExecutorDryRunStatus": _safe_text(dry_run_payload.get("status")),
            "storeContractReportPath": str(contract_path),
            "storeContractSchema": _safe_text(contract_payload.get("schema")),
            "storeContractStatus": _safe_text(contract_payload.get("status")),
            "requestedPaperIds": sorted(requested),
            "papersDir": str(Path(str(papers_dir)).expanduser()) if papers_dir else "",
            "runId": run_id,
            "apply": bool(apply),
        },
        "counts": counts,
        "gate": {
            "readyForDryRun": status == "ok" and not apply,
            "readyForApply": status == "ok" and bool(source_span_records),
            "applyMode": bool(apply),
            "sourceSpanStoreWriteAllowed": bool(apply and papers_dir and not schema_violations),
            "rollbackImplemented": False,
            "rollbackRequiresExplicitRunId": True,
            "sourceSpanPromotionApproved": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": schema_violations,
            "decision": (
                "parsed_artifact_source_span_promotion_executor_apply_ready"
                if status == "ok" and not apply
                else (
                    "parsed_artifact_source_span_promotion_executor_applied"
                    if status == "ok" and apply
                    else "blocked"
                )
            ),
            "recommendedNextTranche": (
                "parsed_artifact_source_span_promotion_readback_review"
                if apply and status == "ok"
                else "parsed_artifact_source_span_promotion_executor_apply_review"
            ),
        },
        "policy": {
            "dryRunByDefault": True,
            "applyRequiredForSourceSpanStoreWrites": True,
            "sourceSpanStoreWrite": bool(applied_rows),
            "candidateStoreWrite": False,
            "sourceSpanPromotionApproved": False,
            "sourceSpanCreated": bool(applied_rows),
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
        "sourceSpanRecords": source_span_records,
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
    source_span_records: list[dict[str, Any]],
    applied_rows: int,
    readback_rows: int,
    manifest_write_rows: int,
    schema_violations: list[str],
    apply_mode: bool,
) -> dict[str, Any]:
    planned_apply_rows = sum(
        1 for row in apply_rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_PLANNED
    )
    source_span_created_rows = readback_rows if apply_mode else 0
    return {
        "inputRows": len(input_rows),
        "plannedApplyRows": planned_apply_rows,
        "heldInputRows": len(input_rows) - len(source_span_records),
        "sourceSpanRecordRows": len(source_span_records),
        "sourceSpanWriteRows": applied_rows,
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
        "sourceSpanCreatedRows": source_span_created_rows,
        "strictEvidenceCreatedRows": 0,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
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
            "sourceSpanRecords",
        )
        if key in report
    }


def render_parsed_artifact_source_span_promotion_executor_apply_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byApplyStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact SourceSpan Promotion Executor Apply",
            "",
            f"- status: {report.get('status', '')}",
            f"- apply mode: {json.dumps(report.get('input', {}).get('apply'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- planned apply rows: {int(counts.get('plannedApplyRows') or 0)}",
            f"- source span write rows: {int(counts.get('sourceSpanWriteRows') or 0)}",
            f"- readback validated rows: {int(counts.get('readbackValidatedRows') or 0)}",
            f"- source spans created: {int(counts.get('sourceSpanCreatedRows') or 0)}",
            f"- strict evidence created: {int(counts.get('strictEvidenceCreatedRows') or 0)}",
            f"- database mutations: {int(counts.get('databaseMutationRows') or 0)}",
            f"- rollback implemented: {json.dumps(report.get('gate', {}).get('rollbackImplemented'))}",
            "",
            "## Apply status breakdown",
            *[f"- {item}" for item in by_status],
        ]
    )


def write_parsed_artifact_source_span_promotion_executor_apply_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-span-promotion-executor-apply.json"
    summary_path = root / "parsed-artifact-source-span-promotion-executor-apply-summary.json"
    markdown_path = root / "parsed-artifact-source-span-promotion-executor-apply.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_source_span_promotion_executor_apply_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Dry-run or explicitly apply parsed-artifact SourceSpan store JSONL records "
            "from a promotion executor dry-run report."
        )
    )
    parser.add_argument(
        "--promotion-executor-dry-run-report",
        default=str(DEFAULT_PROMOTION_EXECUTOR_DRY_RUN_REPORT_PATH),
        help="Path to promotion executor dry-run JSON report.",
    )
    parser.add_argument(
        "--store-contract-report",
        default=str(DEFAULT_STORE_CONTRACT_REPORT_PATH),
        help="Path to SourceSpan store contract JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--papers-dir", default="", help="Local papers_dir root. Required with --apply.")
    parser.add_argument("--run-id", default="", help="Run id recorded into SourceSpan records.")
    parser.add_argument("--apply", action="store_true", help="Write SourceSpan store JSONL records.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = execute_parsed_artifact_source_span_promotion_executor_apply(
        promotion_executor_dry_run_report=args.promotion_executor_dry_run_report,
        store_contract_report=args.store_contract_report,
        papers_dir=args.papers_dir or None,
        run_id=args.run_id or None,
        apply=bool(args.apply),
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_parsed_artifact_source_span_promotion_executor_apply_reports(
            report,
            args.output_dir,
        )
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
    "DEFAULT_PROMOTION_EXECUTOR_DRY_RUN_REPORT_PATH",
    "DEFAULT_STORE_CONTRACT_REPORT_PATH",
    "APPLY_STATUS_APPLIED",
    "APPLY_STATUS_PLANNED",
    "PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_APPLY_SCHEMA_ID",
    "execute_parsed_artifact_source_span_promotion_executor_apply",
    "render_parsed_artifact_source_span_promotion_executor_apply_markdown",
    "write_parsed_artifact_source_span_promotion_executor_apply_reports",
]
