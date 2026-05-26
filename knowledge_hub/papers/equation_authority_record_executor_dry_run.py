"""Dry-run planner for future Equation Authority record writes.

Consumes the Equation Authority record contract report, plans in-memory
equation authority records for candidate rows, and validates the planned record
schema plus semantic invariants. Report-only: no equation authority records,
EquationArtifact, StrictEvidence, SourceSpan, DB/index state, vault content,
parser routing, answer integration, manifest, or authority policy is written.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_authority_record_contract import (
    EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID,
    EQUATION_AUTHORITY_RECORD_CONTRACT_VERSION,
    EQUATION_AUTHORITY_RECORD_DECISION,
    EQUATION_AUTHORITY_RECORD_SCHEMA_ID,
    EQUATION_AUTHORITY_RECORD_STATE,
    EQUATION_AUTHORITY_RECORD_STORE,
    STATUS_EQUATION_AUTHORITY_RECORD_CONTRACT_CANDIDATE_ONLY,
    validate_equation_authority_record_semantics,
)


EQUATION_AUTHORITY_RECORD_EXECUTOR_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.equation-authority-record-executor-dry-run.v1"
)

DRY_RUN_STATUS_READY = "dry_run_ready_equation_authority_record_only"
DRY_RUN_STATUS_BLOCKED_CONTRACT = "blocked_contract_not_ready"
DRY_RUN_STATUS_BLOCKED_RECORD_CONTRACT_STATUS = "blocked_record_contract_status_not_candidate"
DRY_RUN_STATUS_BLOCKED_MISSING_RECORD_PREVIEW = "blocked_missing_equation_authority_record_preview"
DRY_RUN_STATUS_BLOCKED_MISSING_RECORD_ID = "blocked_missing_equation_authority_record_id"
DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_content_hash"
DRY_RUN_STATUS_BLOCKED_MISSING_IDENTITY_DIGEST = "blocked_missing_equation_identity_digest"
DRY_RUN_STATUS_BLOCKED_MISSING_EQUATION_HASH = "blocked_missing_equation_hash"
DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA = "blocked_planned_record_schema_violation"
DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC = "blocked_planned_record_semantic_violation"
DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

DEFAULT_EQUATION_AUTHORITY_RECORD_CONTRACT_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "equation-authority-record-contract"
    / "equation-authority-record-contract-report.json"
)

DEFAULT_EQUATION_AUTHORITY_RECORD_EXECUTOR_DRY_RUN_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "equation-authority-record-executor-dry-run"
)

EXPECTED_INPUT_ROWS = 16
EXPECTED_PLANNED_EQUATION_AUTHORITY_RECORD_ROWS = 13


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(_safe_text(item) for item in items if _safe_text(item)))


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        return {}, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, "unreadable"
    if not isinstance(payload, dict):
        return {}, "unreadable"
    return payload, ""


def _no_mutation_policy_matrix() -> dict[str, Any]:
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


def _contract_schema_violations(report_path: Path, report: dict[str, Any], load_error: str) -> list[str]:
    if load_error == "missing":
        return [f"equation_authority_record_contract_report_missing:{report_path}"]
    if load_error:
        return [f"equation_authority_record_contract_report_unreadable:{report_path}"]
    schema = _safe_text(report.get("schema"))
    if schema != EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID:
        return [
            "schema mismatch: expected "
            f"{EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID}, got {schema or 'missing'}"
        ]
    validation = validate_payload(report, EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID, strict=True)
    return [] if validation.ok else [str(error) for error in validation.errors]


def _contract_ready(report: dict[str, Any], *, expected_planned_rows: int) -> bool:
    gate = report.get("gate") if isinstance(report.get("gate"), dict) else {}
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    return (
        _safe_text(report.get("status")) == "ok"
        and _safe_text(gate.get("decision")) == "equation_authority_record_contract_ready"
        and _safe_bool(gate.get("recordContractRows"))
        and _safe_bool(gate.get("writeTargetContractsDefined"))
        and _safe_bool(gate.get("equationAuthorityRecordSchemaDefined"))
        and not _safe_bool(gate.get("executorReady"))
        and not _safe_bool(gate.get("equationAuthorityRecordWriteAllowed"))
        and _safe_int(counts.get("plannedEquationAuthorityRecordRows")) == expected_planned_rows
        and _safe_int(counts.get("equationAuthorityRecordWrittenRows")) == 0
        and _safe_int(counts.get("equationArtifactCreatedRows")) == 0
        and _safe_int(counts.get("strictEvidenceCreatedRows")) == 0
        and _safe_int(counts.get("sourceSpanMutatedRows")) == 0
    )


def _record_preview(row: dict[str, Any]) -> dict[str, Any]:
    preview = row.get("equation_authority_record_preview")
    return dict(preview) if isinstance(preview, dict) else {}


def _planned_record_from_row(row: dict[str, Any], run_id: str) -> dict[str, Any]:
    record = _record_preview(row)
    if record:
        record["runId"] = run_id
    return record


def _classify_dry_run_row(
    row: dict[str, Any],
    *,
    input_schema_violations: list[str],
    contract_ready: bool,
    run_id: str,
) -> tuple[str, list[str], dict[str, Any] | None]:
    if input_schema_violations:
        return DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA, list(input_schema_violations), None
    if not contract_ready:
        return DRY_RUN_STATUS_BLOCKED_CONTRACT, ["equation_authority_record_contract_not_ready"], None
    input_status = _safe_text(row.get("record_contract_status"))
    if input_status != STATUS_EQUATION_AUTHORITY_RECORD_CONTRACT_CANDIDATE_ONLY:
        blockers = [
            f"record_contract_status={input_status or 'missing'}",
            *[str(item) for item in list(row.get("blockers") or [])],
        ]
        return DRY_RUN_STATUS_BLOCKED_RECORD_CONTRACT_STATUS, _dedupe(blockers), None
    planned_record = _planned_record_from_row(row, run_id)
    if not planned_record:
        return DRY_RUN_STATUS_BLOCKED_MISSING_RECORD_PREVIEW, ["equation_authority_record_preview_missing"], None
    if not _safe_text(planned_record.get("equationAuthorityRecordId")):
        return DRY_RUN_STATUS_BLOCKED_MISSING_RECORD_ID, ["equationAuthorityRecordId_missing"], planned_record
    if not _safe_text(planned_record.get("sourceContentHash")):
        return DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_HASH, ["sourceContentHash_missing"], planned_record
    if not _safe_text(planned_record.get("equationIdentityDigestSha256")):
        return (
            DRY_RUN_STATUS_BLOCKED_MISSING_IDENTITY_DIGEST,
            ["equationIdentityDigestSha256_missing"],
            planned_record,
        )
    if not _safe_text(planned_record.get("equationHashSha256")):
        return DRY_RUN_STATUS_BLOCKED_MISSING_EQUATION_HASH, ["equationHashSha256_missing"], planned_record

    schema_validation = validate_payload(planned_record, EQUATION_AUTHORITY_RECORD_SCHEMA_ID, strict=True)
    if not schema_validation.ok:
        return (
            DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA,
            [str(error) for error in schema_validation.errors],
            planned_record,
        )

    semantic_errors = validate_equation_authority_record_semantics(planned_record)
    if semantic_errors:
        return DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC, semantic_errors, planned_record

    return DRY_RUN_STATUS_READY, [], planned_record


def _planned_executor_key(row: dict[str, Any]) -> str:
    record = _record_preview(row)
    record_id = _safe_text(record.get("equationAuthorityRecordId"))
    return f"equation-authority-record-executor:{record_id or 'unknown'}"


def _dry_run_rows(
    contract_rows: list[dict[str, Any]],
    *,
    input_schema_violations: list[str],
    contract_ready: bool,
    run_id: str,
) -> list[dict[str, Any]]:
    matrix = _no_mutation_policy_matrix()
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(contract_rows, start=1):
        source_row = dict(source or {})
        dry_run_status, blockers, planned_record = _classify_dry_run_row(
            source_row,
            input_schema_violations=input_schema_violations,
            contract_ready=contract_ready,
            run_id=run_id,
        )
        ready = dry_run_status == DRY_RUN_STATUS_READY and planned_record is not None
        record = planned_record or {}
        rows.append(
            {
                "dry_run_row_id": f"equation-authority-record-executor-dry-run:{index:04d}",
                "record_contract_row_id": _safe_text(source_row.get("record_contract_row_id")),
                "contract_design_row_id": _safe_text(source_row.get("contract_design_row_id")),
                "hash_identity_design_row_id": _safe_text(source_row.get("hash_identity_design_row_id")),
                "readiness_audit_row_id": _safe_text(source_row.get("readiness_audit_row_id")),
                "paper_id": _safe_text(source_row.get("paper_id")),
                "source_candidate_id": _safe_text(source_row.get("source_candidate_id")),
                "source_file": _safe_text(source_row.get("source_file")),
                "equation_environment": _safe_text(source_row.get("equation_environment")),
                "input_record_contract_status": _safe_text(source_row.get("record_contract_status")),
                "input_contract_status": _safe_text(source_row.get("input_contract_status")),
                "input_design_status": _safe_text(source_row.get("input_design_status")),
                "readiness_status": _safe_text(source_row.get("readiness_status")),
                "dry_run_status": dry_run_status,
                "dry_run_blockers": _dedupe(blockers),
                "dryRunReadyEquationAuthorityRecordOnly": ready,
                "plannedExecutorKey": _planned_executor_key(source_row),
                "plannedWriteTarget": EQUATION_AUTHORITY_RECORD_STORE if ready else "",
                "plannedContractVersion": EQUATION_AUTHORITY_RECORD_CONTRACT_VERSION if ready else "",
                "plannedAuthorityDecision": EQUATION_AUTHORITY_RECORD_DECISION if ready else "",
                "plannedAuthorityState": EQUATION_AUTHORITY_RECORD_STATE if ready else "",
                "plannedEquationAuthorityRecord": record,
                "equationAuthorityRecordWrite": False,
                "authorityPolicyCreated": False,
                "equationIdentityPromoted": False,
                "equationHashPromoted": False,
                "equationArtifactCreated": False,
                "strictEvidenceCreated": False,
                "sourceSpanMutated": False,
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
                "policyMatrix": matrix,
                "recommended_action": (
                    "queue_for_equation_authority_record_executor_apply_review"
                    if ready
                    else "repair_equation_authority_record_executor_dry_run_input"
                ),
            }
        )
    return rows


def _count_rows(rows: list[dict[str, Any]], *, schema_violations: list[str]) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("dry_run_status")) for row in rows)
    ready_rows = [row for row in rows if _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_READY]
    planned_statuses = {
        DRY_RUN_STATUS_READY,
        DRY_RUN_STATUS_BLOCKED_MISSING_RECORD_ID,
        DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_HASH,
        DRY_RUN_STATUS_BLOCKED_MISSING_IDENTITY_DIGEST,
        DRY_RUN_STATUS_BLOCKED_MISSING_EQUATION_HASH,
        DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA,
        DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC,
    }
    return {
        "inputRows": len(rows),
        "plannedEquationAuthorityRecordRows": sum(
            1 for row in rows if _safe_text(row.get("dry_run_status")) in planned_statuses
        ),
        "dryRunReadyEquationAuthorityRecordOnlyRows": int(by_status.get(DRY_RUN_STATUS_READY, 0)),
        "blockedContractNotReadyRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_CONTRACT, 0)),
        "blockedRecordContractStatusNotCandidateRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_RECORD_CONTRACT_STATUS, 0)
        ),
        "blockedMissingEquationAuthorityRecordPreviewRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_RECORD_PREVIEW, 0)
        ),
        "blockedMissingEquationAuthorityRecordIdRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_RECORD_ID, 0)
        ),
        "blockedMissingSourceContentHashRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_HASH, 0)),
        "blockedMissingEquationIdentityDigestRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_IDENTITY_DIGEST, 0)
        ),
        "blockedMissingEquationHashRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_EQUATION_HASH, 0)),
        "blockedPlannedRecordSchemaViolationRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA, 0)
        ),
        "blockedPlannedRecordSemanticViolationRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC, 0)
        ),
        "blockedInputSchemaViolationRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA, 0)),
        "equationAuthorityRecordWriteRows": 0,
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
        "byPaperId": dict(Counter(_safe_text(row.get("paper_id")) for row in ready_rows)),
        "byEnvironment": dict(Counter(_safe_text(row.get("equation_environment")) for row in ready_rows)),
        "byDryRunStatus": dict(by_status),
        "byRecommendedAction": dict(Counter(_safe_text(row.get("recommended_action")) for row in rows)),
    }


def build_equation_authority_record_executor_dry_run(
    *,
    equation_authority_record_contract_report: str | Path = DEFAULT_EQUATION_AUTHORITY_RECORD_CONTRACT_REPORT,
    paper_ids: list[str] | None = None,
    run_id: str = "equation-authority-record-executor-dry-run-20260520",
    expected_input_rows: int = EXPECTED_INPUT_ROWS,
    expected_planned_equation_authority_record_rows: int = EXPECTED_PLANNED_EQUATION_AUTHORITY_RECORD_ROWS,
) -> dict[str, Any]:
    report_path = Path(str(equation_authority_record_contract_report)).expanduser()
    requested = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}
    warnings: list[str] = []
    contract_report, load_error = _load_json(report_path)
    schema_violations = _contract_schema_violations(report_path, contract_report, load_error)
    contract_rows = [
        dict(row) for row in list(contract_report.get("rows") or []) if isinstance(row, dict)
    ]
    if requested:
        found = {_safe_text(row.get("paper_id")) for row in contract_rows if _safe_text(row.get("paper_id"))}
        if requested - found:
            warnings.append("requested_paper_ids_not_found_in_equation_authority_record_contract")
        contract_rows = [row for row in contract_rows if _safe_text(row.get("paper_id")) in requested]

    contract_is_ready = _contract_ready(
        contract_report,
        expected_planned_rows=expected_planned_equation_authority_record_rows,
    )
    if not contract_is_ready and not schema_violations:
        warnings.append("equation_authority_record_contract_not_ready")

    rows = _dry_run_rows(
        contract_rows,
        input_schema_violations=_dedupe(schema_violations),
        contract_ready=contract_is_ready,
        run_id=run_id,
    )
    counts = _count_rows(rows, schema_violations=_dedupe(schema_violations))
    if requested:
        expected_rows_ok = (
            counts["dryRunReadyEquationAuthorityRecordOnlyRows"]
            <= expected_planned_equation_authority_record_rows
        )
        expected_ready_ok = counts["dryRunReadyEquationAuthorityRecordOnlyRows"] > 0
    else:
        expected_rows_ok = len(contract_rows) == expected_input_rows
        expected_ready_ok = (
            counts["dryRunReadyEquationAuthorityRecordOnlyRows"]
            == expected_planned_equation_authority_record_rows
            and counts["dryRunReadyEquationAuthorityRecordOnlyRows"] > 0
        )
    status = "ok"
    if (
        schema_violations
        or not rows
        or not contract_is_ready
        or not expected_rows_ok
        or not expected_ready_ok
    ):
        status = "blocked"

    matrix = _no_mutation_policy_matrix()
    return {
        "schema": EQUATION_AUTHORITY_RECORD_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "input": {
            "equationAuthorityRecordContractReportPath": str(report_path),
            "equationAuthorityRecordContractReportSchema": _safe_text(contract_report.get("schema"))
            if contract_report
            else "",
            "equationAuthorityRecordContractReportStatus": _safe_text(contract_report.get("status"))
            if contract_report
            else "",
            "requestedPaperIds": sorted(requested),
            "runId": run_id,
            "expectedInputRows": expected_input_rows,
            "expectedPlannedEquationAuthorityRecordRows": expected_planned_equation_authority_record_rows,
        },
        "counts": counts,
        "dryRunOnlyPolicyMatrix": matrix,
        "gate": {
            "readyForEquationAuthorityRecordExecutorDryRun": status == "ok",
            "readyForEquationAuthorityRecordExecutorApply": False,
            "equationAuthorityRecordWriteAllowed": False,
            "authorityPolicyMutationAllowed": False,
            "equationArtifactCreationReady": False,
            "strictEvidenceReady": False,
            "sourceSpanMutationReady": False,
            "runtimeVisibleAllowed": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runManifestWriteAllowed": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(schema_violations),
            "decision": (
                "equation_authority_record_executor_dry_run_ready"
                if status == "ok"
                else "equation_authority_record_executor_dry_run_blocked"
            ),
            "recommendedNextTranche": (
                "equation_authority_record_executor_apply"
                if status == "ok"
                else "equation_authority_record_executor_dry_run_repair"
            ),
        },
        "policy": {"reportOnly": True, "dryRunOnly": True, **matrix},
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
            "dryRunOnlyPolicyMatrix",
            "gate",
            "policy",
            "warnings",
        )
        if key in report
    }


def render_equation_authority_record_executor_dry_run_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byDryRunStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Equation Authority Record Executor Dry Run",
            "",
            f"- Status: `{report.get('status', '')}`",
            f"- Decision: `{gate.get('decision', '')}`",
            f"- Input rows: `{int(counts.get('inputRows') or 0)}`",
            f"- Planned equation authority record rows: `{int(counts.get('plannedEquationAuthorityRecordRows') or 0)}`",
            f"- Dry-run ready rows: `{int(counts.get('dryRunReadyEquationAuthorityRecordOnlyRows') or 0)}`",
            f"- Record writes: `{int(counts.get('equationAuthorityRecordWriteRows') or 0)}`",
            f"- EquationArtifact created rows: `{int(counts.get('equationArtifactCreatedRows') or 0)}`",
            f"- StrictEvidence created rows: `{int(counts.get('strictEvidenceCreatedRows') or 0)}`",
            "",
            "## Dry-run Status Breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- Recommended next tranche: `{gate.get('recommendedNextTranche', '')}`",
        ]
    )


def write_equation_authority_record_executor_dry_run_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "equation-authority-record-executor-dry-run-report.json"
    summary_path = root / "equation-authority-record-executor-dry-run-summary.json"
    markdown_path = root / "equation-authority-record-executor-dry-run.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_authority_record_executor_dry_run_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--equation-authority-record-contract-report",
        default=str(DEFAULT_EQUATION_AUTHORITY_RECORD_CONTRACT_REPORT),
        help="Path to an existing equation authority record contract report.",
    )
    parser.add_argument(
        "--paper-id",
        dest="paper_ids",
        action="append",
        default=[],
        help="Optional paper id filter. May be supplied multiple times.",
    )
    parser.add_argument("--run-id", default="equation-authority-record-executor-dry-run-20260520")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_EQUATION_AUTHORITY_RECORD_EXECUTOR_DRY_RUN_OUTPUT_DIR),
        help="Directory for report, summary, and Markdown outputs.",
    )
    parser.add_argument("--json", action="store_true", help="Print output paths as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_equation_authority_record_executor_dry_run(
        equation_authority_record_contract_report=args.equation_authority_record_contract_report,
        paper_ids=args.paper_ids,
        run_id=args.run_id,
    )
    paths = write_equation_authority_record_executor_dry_run_reports(report, args.output_dir)
    if args.json:
        print(json.dumps({"status": report["status"], "paths": paths}, ensure_ascii=False, indent=2))
    else:
        print(paths["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_EQUATION_AUTHORITY_RECORD_CONTRACT_REPORT",
    "DEFAULT_EQUATION_AUTHORITY_RECORD_EXECUTOR_DRY_RUN_OUTPUT_DIR",
    "DRY_RUN_STATUS_BLOCKED_CONTRACT",
    "DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA",
    "DRY_RUN_STATUS_BLOCKED_MISSING_EQUATION_HASH",
    "DRY_RUN_STATUS_BLOCKED_MISSING_IDENTITY_DIGEST",
    "DRY_RUN_STATUS_BLOCKED_MISSING_RECORD_ID",
    "DRY_RUN_STATUS_BLOCKED_MISSING_RECORD_PREVIEW",
    "DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_HASH",
    "DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA",
    "DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC",
    "DRY_RUN_STATUS_BLOCKED_RECORD_CONTRACT_STATUS",
    "DRY_RUN_STATUS_READY",
    "EQUATION_AUTHORITY_RECORD_EXECUTOR_DRY_RUN_SCHEMA_ID",
    "build_equation_authority_record_executor_dry_run",
    "render_equation_authority_record_executor_dry_run_markdown",
    "write_equation_authority_record_executor_dry_run_reports",
]
