"""Report-only execution-plan dry-run for parsed-artifact structured-evidence readiness."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_hub.papers.parsed_artifact_structured_evidence_readiness_audit import (
    PARSED_ARTIFACT_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID,
    STATUS_BLOCKED_MISSING_SOURCE_HASH,
    STATUS_READY,
)


PARSED_ARTIFACT_STRUCTURED_EVIDENCE_EXECUTION_PLAN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-structured-evidence-execution-plan.v1"
)

EXECUTION_STATUS_DRY_RUN_READY = "dry_run_ready_source_span_plan_only"
EXECUTION_STATUS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_hash"
EXECUTION_STATUS_BLOCKED_MISSING_LOCATION = "blocked_missing_location"
EXECUTION_STATUS_BLOCKED_UNKNOWN_WRITE_TARGET = "blocked_unknown_write_target"
EXECUTION_STATUS_BLOCKED_NON_READY_INPUT = "blocked_non_ready_input_row"

READY_TO_PLAN_OPERATION = "plan_source_span_creation"
BLOCKED_OPERATION = "hold_for_upstream_repair"

PLAN_TARGETS_BY_ARTIFACT = {
    "section": "parsed_artifact_source_span_candidate_store",
    "table": "parsed_artifact_source_span_candidate_store",
    "figure": "parsed_artifact_source_span_candidate_store",
    "equation": "structured_evidence_candidate_store",
}

RECOMMENDED_ACTION_BY_EXECUTION_STATUS = {
    EXECUTION_STATUS_DRY_RUN_READY: "queue_for_explicit_source_span_promotion_executor_review",
    EXECUTION_STATUS_BLOCKED_MISSING_SOURCE_HASH: "recover_source_content_hash_before_promotion",
    EXECUTION_STATUS_BLOCKED_MISSING_LOCATION: "recover_page_or_location_context_before_plan",
    EXECUTION_STATUS_BLOCKED_UNKNOWN_WRITE_TARGET: "confirm_artifact_source_write_target_before_plan",
    EXECUTION_STATUS_BLOCKED_NON_READY_INPUT: "resolve_readiness_input_before_plan",
}

DEFAULT_READINESS_REPORT_PATH = (
    "/Users/won/.khub/reports/layout-parser-pilot/2026-05-19/"
    "parsed-artifact-structured-evidence-readiness-measured-run/"
    "01-parsed-artifact-structured-evidence-readiness/"
    "parsed-artifact-structured-evidence-readiness-audit.json"
)


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


def _normalize_indexes(values: Any) -> list[int]:
    if values is None:
        return []
    try:
        return [int(item) for item in list(values)]
    except Exception:
        return []


def _normalize_bbox(values: Any) -> list[float]:
    if values is None:
        return []
    out: list[float] = []
    for item in list(values):
        try:
            out.append(float(item))
        except Exception:
            continue
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
    if not payload_path.exists():
        return {}
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


def _build_execution_status(readiness_status: str, readiness_row: dict[str, Any]) -> tuple[str, list[str]]:
    source_content_hash = _safe_text(readiness_row.get("sourceContentHash"))
    page = _safe_int(readiness_row.get("page"))
    bbox = _normalize_bbox(readiness_row.get("bbox") or readiness_row.get("selected_bbox"))
    block_indexes = _normalize_indexes(
        readiness_row.get("blockIndexes")
        or readiness_row.get("block_indexes")
        or (readiness_row.get("selected_pdf_region") or {}).get("block_indexes")
    )
    chars_start = _safe_int(readiness_row.get("chars_start"))
    chars_end = _safe_int(readiness_row.get("chars_end"))
    has_location = page is not None or bool(bbox) or bool(block_indexes) or (
        chars_start is not None and chars_end is not None
    )

    artifact_type = _safe_text(readiness_row.get("artifact_type"))
    planned_target = PLAN_TARGETS_BY_ARTIFACT.get(artifact_type)

    if readiness_status == STATUS_BLOCKED_MISSING_SOURCE_HASH:
        return (
            EXECUTION_STATUS_BLOCKED_MISSING_SOURCE_HASH,
            ["readiness_status=blocked_missing_source_hash"],
        )
    if readiness_status != STATUS_READY:
        return EXECUTION_STATUS_BLOCKED_NON_READY_INPUT, [f"readiness_status={readiness_status or 'unknown'}"]
    if not source_content_hash:
        return EXECUTION_STATUS_BLOCKED_MISSING_SOURCE_HASH, ["sourceContentHash missing"]
    if not has_location:
        return EXECUTION_STATUS_BLOCKED_MISSING_LOCATION, ["page/bbox/blockIndexes/chars offsets missing"]
    if not planned_target:
        return EXECUTION_STATUS_BLOCKED_UNKNOWN_WRITE_TARGET, [
            f"unknown artifact write target for artifact_type={artifact_type or 'unknown'}"
        ]
    return EXECUTION_STATUS_DRY_RUN_READY, []


def _plan_rows(readiness_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, readiness_row in enumerate(readiness_rows):
        readiness_status = _safe_text(readiness_row.get("readiness_status"))
        source_readiness_row_id = _safe_text(
            readiness_row.get("source_readiness_row_id")
            or readiness_row.get("source_candidate_id")
            or readiness_row.get("sourceCandidateId")
            or readiness_row.get("candidate_id")
            or readiness_row.get("row_id")
        )
        if not source_readiness_row_id:
            source_readiness_row_id = f"readiness-row:{index:04d}"

        source_candidate_id = _safe_text(readiness_row.get("source_candidate_id") or readiness_row.get("candidate_id"))
        artifact_type = _safe_text(readiness_row.get("artifact_type"))
        planned_target = PLAN_TARGETS_BY_ARTIFACT.get(artifact_type, "")
        source_content_hash = _safe_text(readiness_row.get("sourceContentHash"))
        source_file = _safe_text(
            readiness_row.get("source_file")
            or readiness_row.get("sourceFile")
            or readiness_row.get("source_pdf_path")
            or readiness_row.get("sourcePdfPath")
        )
        page = _safe_int(readiness_row.get("page"))
        bbox = _normalize_bbox(readiness_row.get("bbox") or readiness_row.get("selected_bbox"))
        block_indexes = _normalize_indexes(
            readiness_row.get("blockIndexes")
            or readiness_row.get("block_indexes")
            or (readiness_row.get("selected_pdf_region") or {}).get("block_indexes")
        )

        execution_status, execution_blockers = _build_execution_status(readiness_status, readiness_row)
        is_dry_run_ready = execution_status == EXECUTION_STATUS_DRY_RUN_READY

        rows.append(
            {
                "plan_id": f"parsed-artifact-structured-evidence-execution-plan:{source_readiness_row_id}:{index:04d}",
                "source_readiness_row_id": source_readiness_row_id,
                "paper_id": _safe_text(readiness_row.get("paper_id")),
                "artifact_type": artifact_type,
                "source_candidate_id": source_candidate_id,
                "sourceContentHash": source_content_hash,
                "source_file": source_file,
                "page": page,
                "bbox": bbox,
                "blockIndexes": block_indexes,
                "planned_operation": READY_TO_PLAN_OPERATION if is_dry_run_ready else BLOCKED_OPERATION,
                "planned_write_target": planned_target,
                "would_create_source_span": is_dry_run_ready,
                "would_create_strict_evidence": False,
                "would_change_runtime": False,
                "would_mutate_database": False,
                "rollback_strategy": "discard planned source-span candidate row" if is_dry_run_ready else "no-op",
                "execution_status": execution_status,
                "execution_blockers": execution_blockers,
                "recommended_action": RECOMMENDED_ACTION_BY_EXECUTION_STATUS.get(
                    execution_status,
                    _safe_text(readiness_row.get("recommended_action")) or "resolve_readiness_input_before_plan",
                ),
                "readiness_status": readiness_status,
            }
        )

    return rows


def _count_rows(
    *,
    execution_rows: list[dict[str, Any]],
    plan_rows: list[dict[str, Any]],
    schema_violations: list[str],
) -> dict[str, Any]:
    return {
        "inputRows": len(execution_rows),
        "readyInputRows": sum(1 for row in execution_rows if _safe_text(row.get("readiness_status")) == STATUS_READY),
        "plannedRows": len(plan_rows),
        "dryRunReadyRows": sum(
            1 for row in plan_rows if _safe_text(row.get("execution_status")) == EXECUTION_STATUS_DRY_RUN_READY
        ),
        "blockedMissingSourceHashRows": sum(
            1
            for row in plan_rows
            if _safe_text(row.get("execution_status")) == EXECUTION_STATUS_BLOCKED_MISSING_SOURCE_HASH
        ),
        "blockedMissingLocationRows": sum(
            1
            for row in plan_rows
            if _safe_text(row.get("execution_status")) == EXECUTION_STATUS_BLOCKED_MISSING_LOCATION
        ),
        "blockedUnknownWriteTargetRows": sum(
            1
            for row in plan_rows
            if _safe_text(row.get("execution_status")) == EXECUTION_STATUS_BLOCKED_UNKNOWN_WRITE_TARGET
        ),
        "blockedNonReadyInputRows": sum(
            1
            for row in plan_rows
            if _safe_text(row.get("execution_status")) == EXECUTION_STATUS_BLOCKED_NON_READY_INPUT
        ),
        "sourceSpanCreatedRows": sum(1 for row in plan_rows if _safe_bool(row.get("would_create_source_span"))),
        "strictEvidenceCreatedRows": sum(
            1 for row in plan_rows if _safe_bool(row.get("would_create_strict_evidence"))
        ),
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": sum(1 for row in plan_rows if _safe_bool(row.get("would_mutate_database"))),
        "schemaViolationCount": len(schema_violations),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in plan_rows)),
        "byExecutionStatus": dict(Counter(str(row.get("execution_status") or "") for row in plan_rows)),
        "byPlannedWriteTarget": dict(Counter(str(row.get("planned_write_target") or "") for row in plan_rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in plan_rows)),
    }


def build_parsed_artifact_structured_evidence_execution_plan(
    *,
    readiness_report: str | Path | None = DEFAULT_READINESS_REPORT_PATH,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    readiness_path = Path(str(readiness_report)).expanduser() if readiness_report else None
    input_payload = _read_json(readiness_path)
    input_schema = _safe_text(input_payload.get("schema"))

    warnings: list[str] = []
    schema_violations: list[str] = []
    requested = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    if not readiness_path:
        warnings.append("readiness_report_not_provided")
        schema_violations.append("readiness_report_missing")
    elif not input_payload:
        warnings.append("readiness_report_unreadable")
        schema_violations.append("readiness_report_unreadable")

    if input_payload and input_schema != PARSED_ARTIFACT_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID:
        warnings.append("readiness_report_schema_mismatch")
        schema_violations.append("readiness_report_schema_mismatch")

    readiness_rows: list[dict[str, Any]] = []
    if not schema_violations:
        readiness_rows = [
            dict(item, **{"readiness_status": _safe_text(item.get("readiness_status"))})
            for item in _extract_rows(input_payload)
            if isinstance(item, dict)
        ]
        if not readiness_rows:
            warnings.append("readiness_report_rows_missing")

        if requested:
            readiness_rows = [
                row for row in readiness_rows if _safe_text(row.get("paper_id")) in requested
            ]

    plan_rows = _plan_rows(readiness_rows)

    if not plan_rows:
        warnings.append("no_readiness_rows")

    if not schema_violations and plan_rows and requested and not readiness_rows:
        # Filtered all rows away.
        warnings.append("requested_paper_ids_not_found")

    status = "ok"
    if schema_violations or not plan_rows:
        status = "blocked"
        if "execution_plan_not_ready" not in warnings:
            warnings.append("execution_plan_not_ready")

    counts = _count_rows(
        execution_rows=readiness_rows,
        plan_rows=plan_rows,
        schema_violations=schema_violations,
    )

    return {
        "schema": PARSED_ARTIFACT_STRUCTURED_EVIDENCE_EXECUTION_PLAN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "readinessReportPath": str(readiness_path) if readiness_path else "",
            "readinessReportSchema": input_schema,
            "requestedPaperIds": sorted(requested),
            "readinessReportStatus": _safe_text(input_payload.get("status")),
        },
        "counts": counts,
        "gate": {
            "readyForExecutionPlan": status == "ok",
            "sourceSpanPlanReady": bool(counts.get("dryRunReadyRows") or 0) > 0,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": schema_violations,
            "decision": "parsed_artifact_structured_evidence_execution_plan_ready"
            if status == "ok"
            else "blocked",
            "recommendedNextTranche": "parsed_artifact_structured_evidence_source_span_execution",
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
        "rows": plan_rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "input", "counts", "gate", "policy", "warnings", "rows")
    }


def render_parsed_artifact_structured_evidence_execution_plan_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    rows = list(report.get("rows") or [])
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byExecutionStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact Structured-Evidence Execution Plan",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- dry-run-only: {json.dumps(report.get('policy', {}).get('dryRunOnly'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- planned rows: {int(counts.get('plannedRows') or 0)}",
            f"- ready plan rows: {int(counts.get('dryRunReadyRows') or 0)}",
            f"- blocked missing source hash rows: {int(counts.get('blockedMissingSourceHashRows') or 0)}",
            f"- blocked missing location rows: {int(counts.get('blockedMissingLocationRows') or 0)}",
            f"- blocked unknown write-target rows: {int(counts.get('blockedUnknownWriteTargetRows') or 0)}",
            f"- blocked non-ready input rows: {int(counts.get('blockedNonReadyInputRows') or 0)}",
            "",
            "## Execution status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            "## Rows",
            *(
                f"- paper={row.get('paper_id','')} artifact={row.get('artifact_type','')} "
                f"status={row.get('execution_status','')} operation={row.get('planned_operation','')}"
                for row in rows
            ),
        ]
    )


def write_parsed_artifact_structured_evidence_execution_plan_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-structured-evidence-execution-plan.json"
    summary_path = root / "parsed-artifact-structured-evidence-execution-plan-summary.json"
    markdown_path = root / "parsed-artifact-structured-evidence-execution-plan.md"

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_parsed_artifact_structured_evidence_execution_plan_markdown(report),
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
            "Build a report-only parsed-artifact structured-evidence execution plan "
            "for measured readiness inputs."
        )
    )
    parser.add_argument(
        "--readiness-report",
        default=DEFAULT_READINESS_REPORT_PATH,
        help="Path to measured parsed-artifact structured-evidence readiness report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable")
    parser.add_argument("--output-dir")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON")
    args = parser.parse_args(argv)

    report = build_parsed_artifact_structured_evidence_execution_plan(
        readiness_report=args.readiness_report,
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_parsed_artifact_structured_evidence_execution_plan_reports(
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
    "PARSED_ARTIFACT_STRUCTURED_EVIDENCE_EXECUTION_PLAN_SCHEMA_ID",
    "build_parsed_artifact_structured_evidence_execution_plan",
    "write_parsed_artifact_structured_evidence_execution_plan_reports",
    "render_parsed_artifact_structured_evidence_execution_plan_markdown",
    "main",
]
