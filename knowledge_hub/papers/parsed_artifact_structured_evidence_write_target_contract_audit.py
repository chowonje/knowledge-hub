"""Report-only write-target contract audit for parsed-artifact structured evidence.

The helper consumes the measured execution-plan report and checks whether each row's
planned write target has an explicit, inspectable product contract today.
No mutation is performed.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_hub.papers.parsed_artifact_structured_evidence_execution_plan import (
    EXECUTION_STATUS_BLOCKED_MISSING_LOCATION,
    EXECUTION_STATUS_BLOCKED_MISSING_SOURCE_HASH,
    EXECUTION_STATUS_BLOCKED_NON_READY_INPUT,
    EXECUTION_STATUS_BLOCKED_UNKNOWN_WRITE_TARGET,
    EXECUTION_STATUS_DRY_RUN_READY,
    PARSED_ARTIFACT_STRUCTURED_EVIDENCE_EXECUTION_PLAN_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_source_span_candidate_store_contract import (
    KNOWN_WRITE_TARGET_CONTRACTS,
)


PARSED_ARTIFACT_STRUCTURED_EVIDENCE_WRITE_TARGET_CONTRACT_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-structured-evidence-write-target-contract-audit.v1"
)

DEFAULT_EXECUTION_PLAN_REPORT_PATH = str(
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-structured-evidence-execution-plan"
    / "01-parsed-artifact-structured-evidence-execution-plan"
    / "parsed-artifact-structured-evidence-execution-plan.json"
)

RECOMMENDED_ACTION_BY_EXECUTION_STATUS = {
    EXECUTION_STATUS_DRY_RUN_READY: "queue_for_explicit_source_span_promotion_executor_review",
    EXECUTION_STATUS_BLOCKED_MISSING_SOURCE_HASH: "recover_source_content_hash_before_promotion",
    EXECUTION_STATUS_BLOCKED_MISSING_LOCATION: "recover_page_or_location_context_before_plan",
    EXECUTION_STATUS_BLOCKED_UNKNOWN_WRITE_TARGET: "confirm_artifact_source_write_target_before_plan",
    EXECUTION_STATUS_BLOCKED_NON_READY_INPUT: "resolve_readiness_input_before_plan",
}

VALID_EXECUTION_STATUSES = (
    EXECUTION_STATUS_DRY_RUN_READY,
    EXECUTION_STATUS_BLOCKED_MISSING_SOURCE_HASH,
    EXECUTION_STATUS_BLOCKED_MISSING_LOCATION,
    EXECUTION_STATUS_BLOCKED_UNKNOWN_WRITE_TARGET,
    EXECUTION_STATUS_BLOCKED_NON_READY_INPUT,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(str(path)).expanduser()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _normalize_execution_blockers(value: Any) -> list[str]:
    out: list[str] = []
    for item in list(value or []):
        text = _safe_text(item)
        if text:
            out.append(text)
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


def _extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    return [dict(item) for item in rows if isinstance(item, dict)]


def _has_location(row: dict[str, Any]) -> bool:
    page = _safe_int(row.get("page"))
    bbox = _normalize_bbox(row.get("bbox") or row.get("selected_bbox"))
    block_indexes = _normalize_indexes(
        row.get("blockIndexes")
        or row.get("block_indexes")
        or (row.get("selected_pdf_region") or {}).get("block_indexes")
    )
    chars_start = _safe_int(row.get("chars_start"))
    chars_end = _safe_int(row.get("chars_end"))
    return page is not None or bool(bbox) or bool(block_indexes) or (
        chars_start is not None and chars_end is not None
    )


def _audit_rows(
    execution_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str], int]:
    rows: list[dict[str, Any]] = []
    schema_violations: list[str] = []
    ready_contract_count = 0

    for index, execution_row in enumerate(execution_rows):
        execution_status = _safe_text(execution_row.get("execution_status"))
        if execution_status not in VALID_EXECUTION_STATUSES:
            if execution_status:
                schema_violations.append(f"execution_status_unknown:{execution_status}")
                execution_status = EXECUTION_STATUS_BLOCKED_NON_READY_INPUT
            else:
                execution_status = EXECUTION_STATUS_BLOCKED_NON_READY_INPUT
        source_content_hash = _safe_text(execution_row.get("sourceContentHash"))
        planned_write_target = _safe_text(execution_row.get("planned_write_target"))
        execution_blockers = _normalize_execution_blockers(execution_row.get("execution_blockers"))
        write_target_contract_reference = _safe_text(KNOWN_WRITE_TARGET_CONTRACTS.get(planned_write_target))
        write_target_contract_known = bool(write_target_contract_reference)

        if execution_status == EXECUTION_STATUS_DRY_RUN_READY:
            if not source_content_hash:
                execution_status = EXECUTION_STATUS_BLOCKED_MISSING_SOURCE_HASH
                execution_blockers.append("sourceContentHash_missing")
            elif not _has_location(execution_row):
                execution_status = EXECUTION_STATUS_BLOCKED_MISSING_LOCATION
                execution_blockers.append("location_context_missing")
            else:
                if write_target_contract_reference:
                    ready_contract_count += 1
                else:
                    execution_status = EXECUTION_STATUS_BLOCKED_UNKNOWN_WRITE_TARGET
                    execution_blockers.append("write_target_contract_reference_missing")
                    if planned_write_target:
                        execution_blockers.append(
                            f"write_target_contract_unknown:{planned_write_target}"
                        )

        is_ready = execution_status == EXECUTION_STATUS_DRY_RUN_READY
        source_readiness_row_id = _safe_text(
            execution_row.get("source_readiness_row_id")
            or execution_row.get("source_candidate_id")
            or execution_row.get("sourceCandidateId")
            or execution_row.get("source_readiness_row")
            or execution_row.get("row_id")
        )
        if not source_readiness_row_id:
            source_readiness_row_id = f"execution-row:{index:04d}"

        rows.append(
            {
                "plan_id": _safe_text(execution_row.get("plan_id")) or f"parsed-artifact-plan-audit:{index:04d}",
                "source_readiness_row_id": source_readiness_row_id,
                "paper_id": _safe_text(execution_row.get("paper_id")),
                "artifact_type": _safe_text(execution_row.get("artifact_type")),
                "source_candidate_id": _safe_text(
                    execution_row.get("source_candidate_id") or execution_row.get("candidate_id")
                ),
                "sourceContentHash": source_content_hash,
                "source_file": _safe_text(
                    execution_row.get("source_file")
                    or execution_row.get("source_pdf_path")
                    or execution_row.get("selected_source_file")
                ),
                "page": _safe_int(execution_row.get("page")),
                "bbox": _normalize_bbox(execution_row.get("bbox") or execution_row.get("selected_bbox")),
                "blockIndexes": _normalize_indexes(
                    execution_row.get("blockIndexes")
                    or execution_row.get("block_indexes")
                    or (execution_row.get("selected_pdf_region") or {}).get("block_indexes")
                ),
                "planned_operation": _safe_text(
                    execution_row.get("planned_operation")
                    or ("plan_source_span_creation" if is_ready else "hold_for_upstream_repair")
                ),
                "planned_write_target": planned_write_target,
                "would_create_source_span": is_ready,
                "would_create_strict_evidence": False,
                "would_change_runtime": False,
                "would_mutate_database": False,
                "write_target_contract_known": write_target_contract_known,
                "write_target_contract_reference": write_target_contract_reference,
                "rollback_strategy": (
                    f"discard planned source-span operation against {planned_write_target}"
                    if is_ready
                    else (
                        "hold row until non-write-target blockers are resolved"
                        if write_target_contract_known
                        else "hold row until write-target contract is confirmed"
                    )
                ),
                "execution_status": execution_status,
                "execution_blockers": _dedupe(execution_blockers),
                "recommended_action": RECOMMENDED_ACTION_BY_EXECUTION_STATUS.get(
                    execution_status,
                    _safe_text(execution_row.get("recommended_action")) or "resolve_readiness_input_before_plan",
                ),
            }
        )

    return rows, _dedupe(schema_violations), ready_contract_count


def _count_rows(
    *,
    execution_rows: list[dict[str, Any]],
    audited_rows: list[dict[str, Any]],
    schema_violations: list[str],
) -> dict[str, Any]:
    return {
        "inputRows": len(execution_rows),
        "readyInputRows": sum(
            1 for row in execution_rows if _safe_text(row.get("execution_status")) == EXECUTION_STATUS_DRY_RUN_READY
        ),
        "plannedRows": len(audited_rows),
        "writeTargetContractKnownRows": sum(
            1 for row in audited_rows if _safe_bool(row.get("write_target_contract_known"))
        ),
        "blockedMissingSourceHashRows": sum(
            1 for row in audited_rows if _safe_text(row.get("execution_status")) == EXECUTION_STATUS_BLOCKED_MISSING_SOURCE_HASH
        ),
        "blockedMissingLocationRows": sum(
            1 for row in audited_rows if _safe_text(row.get("execution_status")) == EXECUTION_STATUS_BLOCKED_MISSING_LOCATION
        ),
        "blockedUnknownWriteTargetRows": sum(
            1 for row in audited_rows if _safe_text(row.get("execution_status")) == EXECUTION_STATUS_BLOCKED_UNKNOWN_WRITE_TARGET
        ),
        "blockedNonReadyInputRows": sum(
            1 for row in audited_rows if _safe_text(row.get("execution_status")) == EXECUTION_STATUS_BLOCKED_NON_READY_INPUT
        ),
        "sourceSpanCreatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in audited_rows)),
        "byExecutionStatus": dict(Counter(str(row.get("execution_status") or "") for row in audited_rows)),
        "byPlannedWriteTarget": dict(Counter(str(row.get("planned_write_target") or "") for row in audited_rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in audited_rows)),
    }


def build_parsed_artifact_structured_evidence_write_target_contract_audit(
    *,
    execution_plan_report: str | Path | None = DEFAULT_EXECUTION_PLAN_REPORT_PATH,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    execution_plan_path = Path(str(execution_plan_report)).expanduser() if execution_plan_report else None
    input_payload = _read_json(execution_plan_path)
    input_schema = _safe_text(input_payload.get("schema"))
    input_status = _safe_text(input_payload.get("status"))
    requested = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    warnings: list[str] = []
    schema_violations: list[str] = []

    if not execution_plan_report:
        warnings.append("execution_plan_report_not_provided")
        schema_violations.append("execution_plan_report_missing")
    elif not input_payload:
        warnings.append("execution_plan_report_unreadable")
        schema_violations.append("execution_plan_report_unreadable")

    if input_payload and input_schema != PARSED_ARTIFACT_STRUCTURED_EVIDENCE_EXECUTION_PLAN_SCHEMA_ID:
        warnings.append("execution_plan_schema_mismatch")
        schema_violations.append("execution_plan_schema_mismatch")

    execution_rows: list[dict[str, Any]] = []
    if not schema_violations:
        all_rows = [
            dict(item)
            for item in _extract_rows(input_payload)
            if isinstance(item, dict)
        ]
        if not all_rows:
            warnings.append("execution_plan_rows_missing")
        elif requested:
            requested_found = {
                _safe_text(item.get("paper_id"))
                for item in all_rows
                if _safe_text(item.get("paper_id"))
            }
            missing_requested = requested - requested_found
            if missing_requested:
                warnings.append("requested_paper_ids_not_found")
            execution_rows = [
                item for item in all_rows if _safe_text(item.get("paper_id")) in requested
            ]
        else:
            execution_rows = all_rows

        if not execution_rows:
            warnings.append("execution_plan_rows_missing")

    audited_rows, status_warnings, ready_contract_count = _audit_rows(execution_rows)
    schema_violations.extend(status_warnings)
    schema_violations = _dedupe(schema_violations)

    counts = _count_rows(
        execution_rows=execution_rows,
        audited_rows=audited_rows,
        schema_violations=schema_violations,
    )

    status = "ok"
    if schema_violations or not audited_rows:
        status = "blocked"
    elif any(row.get("execution_status") != EXECUTION_STATUS_DRY_RUN_READY for row in audited_rows):
        status = "blocked"

    return {
        "schema": PARSED_ARTIFACT_STRUCTURED_EVIDENCE_WRITE_TARGET_CONTRACT_AUDIT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "executionPlanReportPath": str(execution_plan_path) if execution_plan_path else "",
            "executionPlanSchema": input_schema,
            "executionPlanStatus": input_status,
            "requestedPaperIds": sorted(requested),
        },
        "counts": counts,
        "gate": {
            "readyForWriteTargetContractAudit": status == "ok",
            "writeTargetContractKnown": bool(ready_contract_count),
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": schema_violations,
            "decision": (
                "parsed_artifact_structured_evidence_write_target_contract_audit_ready"
                if status == "ok"
                else "blocked"
            ),
            "recommendedNextTranche": "parsed_artifact_structured_evidence_source_span_execution_plan",
        },
        "policy": {
            "reportOnly": True,
            "dryRunOnly": True,
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
        "rows": audited_rows,
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
        )
        if key in report
    }


def render_parsed_artifact_structured_evidence_write_target_contract_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byExecutionStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact Structured-Evidence Write Target Contract Audit",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- dry-run-only: {json.dumps(report.get('policy', {}).get('dryRunOnly'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- ready input rows: {int(counts.get('readyInputRows') or 0)}",
            f"- planned rows: {int(counts.get('plannedRows') or 0)}",
            f"- contract-known rows: {int(counts.get('writeTargetContractKnownRows') or 0)}",
            f"- blocked unknown write-target rows: {int(counts.get('blockedUnknownWriteTargetRows') or 0)}",
            "",
            "## Execution status breakdown",
            *[f"- {item}" for item in by_status],
        ]
    )


def write_parsed_artifact_structured_evidence_write_target_contract_audit_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-structured-evidence-write-target-contract-audit.json"
    summary_path = root / "parsed-artifact-structured-evidence-write-target-contract-audit-summary.json"
    markdown_path = root / "parsed-artifact-structured-evidence-write-target-contract-audit.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_structured_evidence_write_target_contract_audit_markdown(report),
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Audit parsed-artifact structured-evidence execution-plan write targets "
            "for inspectable product contract readiness."
        )
    )
    parser.add_argument(
        "--execution-plan-report",
        default=DEFAULT_EXECUTION_PLAN_REPORT_PATH,
        help="Path to measured parsed-artifact structured-evidence execution-plan report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--output-dir", default="", help="Directory for report/summary/markdown outputs.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    payload = build_parsed_artifact_structured_evidence_write_target_contract_audit(
        execution_plan_report=args.execution_plan_report,
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_parsed_artifact_structured_evidence_write_target_contract_audit_reports(
            payload,
            args.output_dir,
        )
        print(f"wrote report: {paths['report']}")
        print(f"wrote summary: {paths['summary']}")
        print(f"wrote markdown: {paths['markdown']}")

    if args.json or not args.output_dir:
        print(json.dumps(_summary_payload(payload), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PARSED_ARTIFACT_STRUCTURED_EVIDENCE_WRITE_TARGET_CONTRACT_AUDIT_SCHEMA_ID",
    "KNOWN_WRITE_TARGET_CONTRACTS",
    "build_parsed_artifact_structured_evidence_write_target_contract_audit",
    "write_parsed_artifact_structured_evidence_write_target_contract_audit_reports",
    "render_parsed_artifact_structured_evidence_write_target_contract_audit_markdown",
]
