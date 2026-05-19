"""Dry-run planner for parsed-artifact SourceSpan promotion.

This helper consumes the promotion policy gate report and emits planned
SourceSpan creation rows only. It does not write SourceSpan records, candidate
store records, strict evidence, citation-grade evidence, or runtime evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_candidate_promotion_policy_gate import (
    ALLOWED_SOURCE_SPAN_ARTIFACT_TYPES,
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_PROMOTION_POLICY_GATE_SCHEMA_ID,
    POLICY_STATUS_READY_CANDIDATE_ONLY,
    _has_locator,
)


PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-promotion-executor-dry-run.v1"
)

PARSED_ARTIFACT_SOURCE_SPAN_STORE = "parsed_artifact_source_span_store"

EXECUTOR_STATUS_PLANNED = "dry_run_planned_source_span"
EXECUTOR_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
EXECUTOR_STATUS_BLOCKED_POLICY_GATE_NOT_READY = "blocked_policy_gate_not_ready"
EXECUTOR_STATUS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_hash"
EXECUTOR_STATUS_BLOCKED_MISSING_LOCATOR = "blocked_missing_locator"

DEFAULT_POLICY_GATE_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-candidate-promotion-policy-gate"
    / "01-parsed-artifact-source-span-candidate-promotion-policy-gate"
    / "parsed-artifact-source-span-candidate-promotion-policy-gate.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-promotion-executor-dry-run"
    / "01-parsed-artifact-source-span-promotion-executor-dry-run"
)

ROLLBACK_NOTE = (
    "dry-run only; explicit apply tranche required before any source span store write"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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


def _planned_source_span_key(
    *,
    candidate_record_id: str,
    paper_id: str,
    artifact_type: str,
    idempotency_key: str,
) -> str:
    record_id = _safe_text(candidate_record_id)
    if record_id.startswith("source-span-candidate:"):
        return "source-span:" + record_id[len("source-span-candidate:") :]
    suffix = idempotency_key[:16] if idempotency_key else "unknown"
    return f"source-span:{paper_id}:{artifact_type}:{suffix}"


def _write_matrix() -> dict[str, Any]:
    return {
        "plannedWriteTarget": PARSED_ARTIFACT_SOURCE_SPAN_STORE,
        "writeEnabled": False,
        "sourceSpanStoreWrite": False,
        "candidateStoreWrite": False,
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
    }


def _classify_executor_row(row: dict[str, Any]) -> tuple[str, list[str]]:
    blockers: list[str] = []

    if _safe_text(row.get("policy_gate_status")) != POLICY_STATUS_READY_CANDIDATE_ONLY:
        blockers.extend(_safe_list(row.get("policy_blockers")))
        blockers.append(
            f"policy_gate_status={_safe_text(row.get('policy_gate_status')) or 'unknown'}"
        )
        return EXECUTOR_STATUS_BLOCKED_POLICY_GATE_NOT_READY, _dedupe(blockers)

    if not _safe_bool(row.get("sourceSpanPromotionExecutorDryRunReady")):
        blockers.append("sourceSpanPromotionExecutorDryRunReady_false")
        return EXECUTOR_STATUS_BLOCKED_POLICY_GATE_NOT_READY, _dedupe(blockers)

    flag_violations = _mutation_flag_violation(row)
    if flag_violations:
        blockers.extend(flag_violations)
        return EXECUTOR_STATUS_BLOCKED_POLICY_GATE_NOT_READY, _dedupe(blockers)

    if _safe_text(row.get("artifact_type")) not in ALLOWED_SOURCE_SPAN_ARTIFACT_TYPES:
        return EXECUTOR_STATUS_BLOCKED_POLICY_GATE_NOT_READY, [
            f"artifact_type={_safe_text(row.get('artifact_type')) or 'unknown'}"
        ]

    if not _safe_text(row.get("sourceContentHash")):
        return EXECUTOR_STATUS_BLOCKED_MISSING_SOURCE_HASH, ["sourceContentHash_missing"]

    if not _has_locator(row):
        return EXECUTOR_STATUS_BLOCKED_MISSING_LOCATOR, [
            "locator_missing_page_bbox_blockIndexes_or_chars"
        ]

    return EXECUTOR_STATUS_PLANNED, []


def _executor_rows(policy_gate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, policy_row in enumerate(policy_gate_rows):
        source_row = dict(policy_row or {})
        status, blockers = _classify_executor_row(source_row)
        planned = status == EXECUTOR_STATUS_PLANNED
        paper_id = _safe_text(source_row.get("paper_id"))
        artifact_type = _safe_text(source_row.get("artifact_type"))
        idempotency_key = _safe_text(source_row.get("idempotencyKey"))
        candidate_record_id = _safe_text(source_row.get("candidateRecordId"))
        planned_key = (
            _planned_source_span_key(
                candidate_record_id=candidate_record_id,
                paper_id=paper_id,
                artifact_type=artifact_type,
                idempotency_key=idempotency_key,
            )
            if planned
            else ""
        )
        rows.append(
            {
                "promotion_executor_dry_run_row_id": (
                    f"parsed-artifact-source-span-promotion-executor-dry-run:{index:04d}"
                ),
                "policy_gate_row_id": _safe_text(source_row.get("policy_gate_row_id")),
                "readback_review_row_id": _safe_text(source_row.get("readback_review_row_id")),
                "candidateRecordId": candidate_record_id,
                "runId": _safe_text(source_row.get("runId")),
                "paper_id": paper_id,
                "artifact_type": artifact_type,
                "source_candidate_id": _safe_text(source_row.get("source_candidate_id")),
                "source_readiness_row_id": _safe_text(source_row.get("source_readiness_row_id")),
                "sourceContentHash": _safe_text(source_row.get("sourceContentHash")),
                "source_file": _safe_text(source_row.get("source_file")),
                "locator": (
                    source_row.get("locator") if isinstance(source_row.get("locator"), dict) else {}
                ),
                "idempotencyKey": idempotency_key,
                "candidate_store_path": _safe_text(source_row.get("candidate_store_path")),
                "candidate_store_line": _safe_int(source_row.get("candidate_store_line")) or 0,
                "policy_gate_status": _safe_text(source_row.get("policy_gate_status")),
                "executor_dry_run_status": status,
                "executor_blockers": _dedupe(blockers),
                "plannedSourceSpanKey": planned_key,
                "plannedSourceSpanId": planned_key,
                "plannedWriteTarget": PARSED_ARTIFACT_SOURCE_SPAN_STORE if planned else "",
                "writeMatrix": _write_matrix() if planned else {},
                "rollbackNote": ROLLBACK_NOTE if planned else "",
                "dryRunPlannedSourceSpan": planned,
                "sourceSpanPromotionApproved": False,
                "sourceSpanCreated": False,
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "canonicalParsedArtifactsWritten": False,
                "recommended_action": (
                    "queue_for_explicit_source_span_promotion_executor_apply"
                    if planned
                    else "repair_policy_gate_row_before_promotion_executor_dry_run"
                ),
            }
        )
    return rows


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    input_schema_violations: list[str],
) -> dict[str, Any]:
    return {
        "inputRows": len(rows),
        "dryRunPlannedSourceSpanRows": sum(
            1 for row in rows if row.get("executor_dry_run_status") == EXECUTOR_STATUS_PLANNED
        ),
        "blockedInputSchemaViolationRows": (
            len(rows) if input_schema_violations and rows else int(bool(input_schema_violations))
        ),
        "blockedPolicyGateNotReadyRows": sum(
            1
            for row in rows
            if row.get("executor_dry_run_status") == EXECUTOR_STATUS_BLOCKED_POLICY_GATE_NOT_READY
        ),
        "blockedMissingSourceHashRows": sum(
            1
            for row in rows
            if row.get("executor_dry_run_status") == EXECUTOR_STATUS_BLOCKED_MISSING_SOURCE_HASH
        ),
        "blockedMissingLocatorRows": sum(
            1
            for row in rows
            if row.get("executor_dry_run_status") == EXECUTOR_STATUS_BLOCKED_MISSING_LOCATOR
        ),
        "sourceSpanCreatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "schemaViolationCount": len(input_schema_violations),
        "byPaperId": dict(Counter(str(row.get("paper_id") or "") for row in rows)),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in rows)),
        "byExecutorDryRunStatus": dict(
            Counter(str(row.get("executor_dry_run_status") or "") for row in rows)
        ),
        "byRecommendedAction": dict(
            Counter(str(row.get("recommended_action") or "") for row in rows)
        ),
    }


def build_parsed_artifact_source_span_promotion_executor_dry_run(
    *,
    policy_gate_report_path: str | Path = DEFAULT_POLICY_GATE_REPORT_PATH,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    report_path = Path(str(policy_gate_report_path)).expanduser()
    warnings: list[str] = []
    input_schema_violations: list[str] = []
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    policy_gate_report = _read_json(report_path)
    if not policy_gate_report:
        warnings.append("policy_gate_report_missing_or_unreadable")

    validation = validate_payload(
        policy_gate_report,
        PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_PROMOTION_POLICY_GATE_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        input_schema_violations = [str(error) for error in validation.errors]
        if not policy_gate_report:
            input_schema_violations.append("policy_gate_report_missing_or_unreadable")

    policy_gate_rows = [
        row for row in policy_gate_report.get("rows", []) if isinstance(row, dict)
    ] if isinstance(policy_gate_report, dict) else []

    if requested_papers:
        found_papers = {
            _safe_text(row.get("paper_id")) for row in policy_gate_rows if _safe_text(row.get("paper_id"))
        }
        if requested_papers - found_papers:
            warnings.append("requested_paper_ids_not_found")
        policy_gate_rows = [
            row for row in policy_gate_rows if _safe_text(row.get("paper_id")) in requested_papers
        ]

    if not policy_gate_rows:
        warnings.append("policy_gate_rows_missing")

    rows = _executor_rows(policy_gate_rows)
    if input_schema_violations:
        for row in rows:
            row["executor_dry_run_status"] = EXECUTOR_STATUS_BLOCKED_INPUT_SCHEMA
            row["executor_blockers"] = _dedupe(
                [*row.get("executor_blockers", []), *input_schema_violations]
            )
            row["dryRunPlannedSourceSpan"] = False
            row["plannedSourceSpanKey"] = ""
            row["plannedSourceSpanId"] = ""
            row["plannedWriteTarget"] = ""
            row["writeMatrix"] = {}
            row["rollbackNote"] = ""
            row["recommended_action"] = "repair_policy_gate_report_schema_before_executor_dry_run"

    counts = _count_rows(rows=rows, input_schema_violations=_dedupe(input_schema_violations))
    planned_rows = int(counts.get("dryRunPlannedSourceSpanRows") or 0)
    status = "ok"
    if input_schema_violations or not rows or planned_rows != len(rows):
        status = "blocked"

    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "policyGateReportPath": str(report_path),
            "policyGateSchema": (
                _safe_text(policy_gate_report.get("schema")) if policy_gate_report else ""
            ),
            "requestedPaperIds": sorted(requested_papers),
        },
        "counts": counts,
        "gate": {
            "readyForSourceSpanPromotionApply": False,
            "sourceSpanPromotionApproved": False,
            "sourceSpanCreated": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "parsed_artifact_source_span_promotion_executor_dry_run_ready"
                if status == "ok"
                else "blocked"
            ),
            "recommendedNextTranche": (
                "parsed_artifact_source_span_promotion_executor_apply"
                if status == "ok"
                else "parsed_artifact_source_span_candidate_promotion_policy_gate_repair"
            ),
        },
        "policy": {
            "reportOnly": True,
            "promotionExecutorDryRunOnly": True,
            "sourceSpanStoreWrite": False,
            "candidateStoreWrite": False,
            "sourceSpanPromotionApproved": False,
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


def render_parsed_artifact_source_span_promotion_executor_dry_run_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byExecutorDryRunStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact SourceSpan Promotion Executor Dry Run",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- promotion-executor-dry-run-only: {json.dumps(report.get('policy', {}).get('promotionExecutorDryRunOnly'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- dry-run planned source span rows: {int(counts.get('dryRunPlannedSourceSpanRows') or 0)}",
            f"- source spans created: {int(counts.get('sourceSpanCreatedRows') or 0)}",
            f"- strict evidence created: {int(counts.get('strictEvidenceCreatedRows') or 0)}",
            f"- database mutations: {int(counts.get('databaseMutationRows') or 0)}",
            "",
            "## Executor dry-run status breakdown",
            *[f"- {item}" for item in by_status],
        ]
    )


def write_parsed_artifact_source_span_promotion_executor_dry_run_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-span-promotion-executor-dry-run.json"
    summary_path = root / "parsed-artifact-source-span-promotion-executor-dry-run-summary.json"
    markdown_path = root / "parsed-artifact-source-span-promotion-executor-dry-run.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_source_span_promotion_executor_dry_run_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Plan parsed-artifact SourceSpan promotion rows from the promotion policy gate "
            "report without writing SourceSpan records."
        )
    )
    parser.add_argument(
        "--policy-gate-report",
        default=str(DEFAULT_POLICY_GATE_REPORT_PATH),
        help="SourceSpan candidate promotion policy gate JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_parsed_artifact_source_span_promotion_executor_dry_run(
        policy_gate_report_path=args.policy_gate_report,
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_parsed_artifact_source_span_promotion_executor_dry_run_reports(
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
    "PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_DRY_RUN_SCHEMA_ID",
    "EXECUTOR_STATUS_PLANNED",
    "build_parsed_artifact_source_span_promotion_executor_dry_run",
    "render_parsed_artifact_source_span_promotion_executor_dry_run_markdown",
    "write_parsed_artifact_source_span_promotion_executor_dry_run_reports",
]
