"""Dry-run planner for StrictEvidence eligibility record writes.

Consumes the eligibility record contract report and the strictEligible mutation
decision-record report, plans in-memory eligibility records for candidate rows, and
validates schema plus semantic contracts with zero filesystem writes.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_eligibility_record_contract import (
    STRICT_EVIDENCE_ELIGIBILITY_RECORD_CONTRACT_SCHEMA_ID,
    STRICT_EVIDENCE_ELIGIBILITY_RECORD_SCHEMA_ID,
    STRICT_EVIDENCE_ELIGIBILITY_STORE,
    build_sample_eligibility_record_from_decision_row,
    validate_eligibility_record_semantics,
)
from knowledge_hub.papers.strict_evidence_strict_eligible_mutation_decision_record import (
    DECISION_SEPARATE_ELIGIBILITY_RECORD,
    DECISION_STATUS_CANDIDATE_ONLY,
    STRICT_EVIDENCE_STRICT_ELIGIBLE_MUTATION_DECISION_RECORD_SCHEMA_ID,
)


STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-eligibility-executor-dry-run.v1"
)

DRY_RUN_STATUS_READY = "dry_run_ready_eligibility_record_only"
DRY_RUN_STATUS_BLOCKED_CONTRACT = "blocked_contract_not_ready"
DRY_RUN_STATUS_BLOCKED_DECISION_RECORD = "blocked_decision_record_not_ready"
DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID = "blocked_missing_strict_evidence_id"
DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_SPAN_ID = "blocked_missing_source_span_id"
DRY_RUN_STATUS_BLOCKED_MISSING_CANDIDATE_RECORD_ID = "blocked_missing_candidate_record_id"
DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA = "blocked_planned_record_schema_violation"
DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC = "blocked_planned_record_semantic_violation"
DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

DEFAULT_ELIGIBILITY_RECORD_CONTRACT_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-eligibility-record-contract"
    / "01-strict-evidence-eligibility-record-contract"
    / "strict-evidence-eligibility-record-contract.json"
)

DEFAULT_DECISION_RECORD_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-strict-eligible-mutation-decision-record"
    / "01-strict-evidence-strict-eligible-mutation-decision-record"
    / "strict-evidence-strict-eligible-mutation-decision-record.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-eligibility-executor-dry-run"
    / "01-strict-evidence-eligibility-executor-dry-run"
)

EXPECTED_POLICY_CANDIDATE_ROWS = 99
EXPECTED_SECTION_DECISION_ROWS = 45
EXPECTED_FIGURE_CAPTION_DECISION_ROWS = 54


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _no_mutation_policy_matrix() -> dict[str, Any]:
    return {
        "plannedWriteTarget": STRICT_EVIDENCE_ELIGIBILITY_STORE,
        "writeEnabled": False,
        "eligibilityRecordWrite": False,
        "strictEvidenceStoreWrite": False,
        "sourceSpanStoreWrite": False,
        "strictEvidenceCreated": False,
        "strictEligibleMutation": False,
        "citationGradeEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "vaultScan": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
        "manifestWrite": False,
    }


def _decision_row_is_candidate(decision_row: dict[str, Any]) -> bool:
    return (
        _safe_text(decision_row.get("decision_status")) == DECISION_STATUS_CANDIDATE_ONLY
        and _safe_text(decision_row.get("strictEligibleMutationDecision"))
        == DECISION_SEPARATE_ELIGIBILITY_RECORD
        and not _safe_bool(decision_row.get("strictEligibleMutationAllowed"))
        and _safe_bool(decision_row.get("eligibilityRecordRequired"))
    )


def _decision_row_flag_violations(decision_row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for field_name in (
        "strictEligible",
        "strictEvidenceCreated",
        "citationGrade",
        "runtimeEvidence",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "databaseMutation",
    ):
        if _safe_bool(decision_row.get(field_name)):
            violations.append(f"decision_row.{field_name}_true")
    if _safe_bool(decision_row.get("strictEligibleMutationAllowed")):
        violations.append("decision_row.strictEligibleMutationAllowed_true")
    if _safe_bool(decision_row.get("eligibilityRecordWriteAllowed")):
        violations.append("decision_row.eligibilityRecordWriteAllowed_true")
    return violations


def _classify_dry_run_row(
    decision_row: dict[str, Any],
    *,
    input_schema_violations: list[str],
    contract_ready: bool,
    decision_report_ready: bool,
    run_id: str,
) -> tuple[str, list[str], dict[str, Any] | None]:
    if input_schema_violations:
        return DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA, list(input_schema_violations), None
    if not contract_ready:
        return DRY_RUN_STATUS_BLOCKED_CONTRACT, ["eligibility_record_contract_not_ready"], None
    if not decision_report_ready:
        return DRY_RUN_STATUS_BLOCKED_DECISION_RECORD, ["decision_record_report_not_ready"], None
    if not _decision_row_is_candidate(decision_row):
        blockers = [_safe_text(item) for item in (decision_row.get("decision_blockers") or [])]
        blockers.append(
            f"decision_status={_safe_text(decision_row.get('decision_status')) or 'unknown'}"
        )
        return DRY_RUN_STATUS_BLOCKED_DECISION_RECORD, _dedupe(blockers), None

    strict_evidence_id = _safe_text(decision_row.get("strictEvidenceId"))
    source_span_id = _safe_text(decision_row.get("sourceSpanId"))
    candidate_record_id = _safe_text(decision_row.get("candidateRecordId"))
    if not strict_evidence_id:
        return DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID, ["strictEvidenceId_missing"], None
    if not source_span_id:
        return DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_SPAN_ID, ["sourceSpanId_missing"], None
    if not candidate_record_id:
        return DRY_RUN_STATUS_BLOCKED_MISSING_CANDIDATE_RECORD_ID, ["candidateRecordId_missing"], None

    flag_violations = _decision_row_flag_violations(decision_row)
    if flag_violations:
        return DRY_RUN_STATUS_BLOCKED_DECISION_RECORD, flag_violations, None

    planned_record = build_sample_eligibility_record_from_decision_row(
        decision_row,
        run_id=run_id,
    )
    schema_validation = validate_payload(
        planned_record,
        STRICT_EVIDENCE_ELIGIBILITY_RECORD_SCHEMA_ID,
        strict=True,
    )
    if not schema_validation.ok:
        return (
            DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA,
            [str(error) for error in schema_validation.errors],
            planned_record,
        )

    semantic_errors = validate_eligibility_record_semantics(planned_record)
    if semantic_errors:
        return DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC, semantic_errors, planned_record

    return DRY_RUN_STATUS_READY, [], planned_record


def _planned_executor_key(decision_row: dict[str, Any]) -> str:
    strict_evidence_id = _safe_text(decision_row.get("strictEvidenceId"))
    return f"eligibility-executor:{strict_evidence_id or 'unknown'}"


def _dry_run_rows(
    decision_rows: list[dict[str, Any]],
    *,
    input_schema_violations: list[str],
    contract_ready: bool,
    decision_report_ready: bool,
    run_id: str,
) -> list[dict[str, Any]]:
    matrix = _no_mutation_policy_matrix()
    rows: list[dict[str, Any]] = []
    for index, decision_row in enumerate(decision_rows):
        source_row = dict(decision_row or {})
        dry_run_status, blockers, planned_record = _classify_dry_run_row(
            source_row,
            input_schema_violations=input_schema_violations,
            contract_ready=contract_ready,
            decision_report_ready=decision_report_ready,
            run_id=run_id,
        )
        ready = dry_run_status == DRY_RUN_STATUS_READY and planned_record is not None
        record = planned_record or {}
        rows.append(
            {
                "dry_run_row_id": f"strict-evidence-eligibility-executor-dry-run:{index:04d}",
                "decision_row_id": _safe_text(source_row.get("decision_row_id")),
                "hold_row_id": _safe_text(source_row.get("hold_row_id")),
                "completion_row_id": _safe_text(source_row.get("completion_row_id")),
                "readback_row_id": _safe_text(source_row.get("readback_row_id")),
                "strictEvidenceId": _safe_text(source_row.get("strictEvidenceId")),
                "sourceSpanId": _safe_text(source_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(source_row.get("candidateRecordId")),
                "paper_id": _safe_text(source_row.get("paper_id")),
                "artifact_type": _safe_text(source_row.get("artifact_type")),
                "decision_status": _safe_text(source_row.get("decision_status")),
                "strictEligibleMutationDecision": _safe_text(
                    source_row.get("strictEligibleMutationDecision")
                ),
                "eligibilityRecordRequired": _safe_bool(source_row.get("eligibilityRecordRequired")),
                "planned_executor_key": _planned_executor_key(source_row),
                "plannedWriteTarget": STRICT_EVIDENCE_ELIGIBILITY_STORE if ready else "",
                "plannedEligibilityRecordId": _safe_text(record.get("eligibilityRecordId")),
                "plannedIdempotencyKey": _safe_text(record.get("idempotencyKey")),
                "plannedEligibilityPolicyVersion": _safe_text(record.get("eligibilityPolicyVersion")),
                "plannedEligibilityDecision": _safe_text(record.get("eligibilityDecision")),
                "plannedEligibilityState": _safe_text(record.get("eligibilityState")),
                "dry_run_status": dry_run_status,
                "dry_run_blockers": _dedupe(blockers),
                "dryRunReadyEligibilityRecordOnly": ready,
                "plannedEligibilityRecord": record,
                "writeMatrix": dict(matrix),
                "strictEligible": False,
                "strictEligibleMutation": False,
                "eligibilityRecordWriteRows": 0,
                "strictEvidenceWriteRows": 0,
                "strictEvidenceCreated": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "canonicalParsedArtifactsWritten": False,
                "sourceSpanUpdatedRows": 0,
                "recommended_action": (
                    "queue_for_strict_evidence_eligibility_executor_apply"
                    if ready
                    else "repair_decision_row_before_eligibility_executor_dry_run"
                ),
            }
        )
    return rows


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    input_schema_violations: list[str],
    decision_candidate_rows: int,
) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("dry_run_status")) for row in rows)
    return {
        "inputRows": len(rows),
        "decisionCandidateRows": decision_candidate_rows,
        "dryRunReadyEligibilityRecordOnlyRows": int(by_status.get(DRY_RUN_STATUS_READY, 0)),
        "blockedContractNotReadyRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_CONTRACT, 0)),
        "blockedDecisionRecordNotReadyRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_DECISION_RECORD, 0)
        ),
        "blockedMissingStrictEvidenceIdRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID, 0)
        ),
        "blockedMissingSourceSpanIdRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_SPAN_ID, 0)
        ),
        "blockedMissingCandidateRecordIdRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_CANDIDATE_RECORD_ID, 0)
        ),
        "blockedPlannedRecordSchemaViolationRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA, 0)
        ),
        "blockedPlannedRecordSemanticViolationRows": int(
            by_status.get(DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC, 0)
        ),
        "blockedInputSchemaViolationRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA, 0)),
        "eligibilityRecordWriteRows": 0,
        "strictEligibleMutationRows": 0,
        "strictEvidenceWriteRows": 0,
        "strictEvidenceCreatedRows": 0,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "sourceSpanUpdatedRows": 0,
        "reindexOrReembedRows": 0,
        "manifestWriteRows": 0,
        "schemaViolationCount": len(input_schema_violations),
        "byPaperId": dict(
            Counter(
                _safe_text(row.get("paper_id"))
                for row in rows
                if row.get("dry_run_status") == DRY_RUN_STATUS_READY
            )
        ),
        "byArtifactType": dict(
            Counter(
                _safe_text(row.get("artifact_type"))
                for row in rows
                if row.get("dry_run_status") == DRY_RUN_STATUS_READY
            )
        ),
        "byDryRunStatus": dict(by_status),
        "byRecommendedAction": dict(Counter(_safe_text(row.get("recommended_action")) for row in rows)),
    }


def build_strict_evidence_eligibility_executor_dry_run(
    *,
    eligibility_record_contract_report_path: str | Path = (
        DEFAULT_ELIGIBILITY_RECORD_CONTRACT_REPORT_PATH
    ),
    decision_record_report_path: str | Path = DEFAULT_DECISION_RECORD_REPORT_PATH,
    run_id: str | None = None,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    contract_path = Path(str(eligibility_record_contract_report_path)).expanduser()
    decision_path = Path(str(decision_record_report_path)).expanduser()
    warnings: list[str] = []
    input_schema_violations: list[str] = []
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    contract_report = _read_json(contract_path)
    decision_report = _read_json(decision_path)

    if not contract_report:
        input_schema_violations.append("eligibility_record_contract_report_missing_or_unreadable")
    if not decision_report:
        input_schema_violations.append("decision_record_report_missing_or_unreadable")

    if contract_report:
        validation = validate_payload(
            contract_report,
            STRICT_EVIDENCE_ELIGIBILITY_RECORD_CONTRACT_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            input_schema_violations.extend(
                f"contract_report:{error}" for error in validation.errors
            )

    if decision_report:
        validation = validate_payload(
            decision_report,
            STRICT_EVIDENCE_STRICT_ELIGIBLE_MUTATION_DECISION_RECORD_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            input_schema_violations.extend(
                f"decision_record_report:{error}" for error in validation.errors
            )

    contract_gate = contract_report.get("gate") if isinstance(contract_report.get("gate"), dict) else {}
    contract_ready = (
        bool(contract_report)
        and _safe_text(contract_report.get("status")) == "ok"
        and _safe_text(contract_gate.get("decision"))
        == "strict_evidence_eligibility_record_contract_ready"
    )

    decision_report_ready = bool(decision_report) and _safe_text(decision_report.get("status")) == "ok"

    decision_semantics = (
        decision_report.get("strictEligibleSemanticsDecision")
        if isinstance(decision_report.get("strictEligibleSemanticsDecision"), dict)
        else {}
    )
    if decision_report and _safe_text(decision_semantics.get("decision")) != DECISION_SEPARATE_ELIGIBILITY_RECORD:
        input_schema_violations.append(
            "decision_semantics="
            f"{_safe_text(decision_semantics.get('decision')) or 'unknown'}"
        )

    decision_rows = [
        row for row in decision_report.get("rows", []) if isinstance(row, dict)
    ] if decision_report else []

    if requested_papers:
        found_papers = {_safe_text(row.get("paper_id")) for row in decision_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found_papers:
            warnings.append("requested_paper_ids_not_found")
        decision_rows = [row for row in decision_rows if _safe_text(row.get("paper_id")) in requested_papers]

    if not decision_rows and not input_schema_violations:
        warnings.append("decision_record_rows_missing")

    decision_candidate_rows = sum(1 for row in decision_rows if _decision_row_is_candidate(row))
    decision_input = (
        decision_report.get("input") if isinstance(decision_report.get("input"), dict) else {}
    )
    decision_counts = (
        decision_report.get("counts") if isinstance(decision_report.get("counts"), dict) else {}
    )
    expected_policy_rows = (
        _safe_int(decision_input.get("expectedPolicyCandidateRows"))
        or _safe_int(decision_counts.get("decisionRecordCandidateOnlyRows"))
        or len(decision_rows)
    )

    run_id = _safe_text(run_id) or f"strict-evidence-eligibility-executor-dry-run-{_now_iso()}"

    input_schema_violations = _dedupe(input_schema_violations)
    rows = _dry_run_rows(
        decision_rows,
        input_schema_violations=input_schema_violations,
        contract_ready=contract_ready,
        decision_report_ready=decision_report_ready,
        run_id=run_id,
    )
    counts = _count_rows(
        rows=rows,
        input_schema_violations=input_schema_violations,
        decision_candidate_rows=decision_candidate_rows,
    )

    ready_rows = int(counts.get("dryRunReadyEligibilityRecordOnlyRows") or 0)
    status = "ok"
    if (
        input_schema_violations
        or not rows
        or decision_candidate_rows != expected_policy_rows
        or ready_rows != expected_policy_rows
        or ready_rows != len(rows)
    ):
        status = "blocked"

    policy_matrix = _no_mutation_policy_matrix()

    return {
        "schema": STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "eligibilityRecordContractReportPath": str(contract_path),
            "eligibilityRecordContractReportSchema": _safe_text(contract_report.get("schema"))
            if contract_report
            else "",
            "eligibilityRecordContractReportStatus": _safe_text(contract_report.get("status"))
            if contract_report
            else "",
            "decisionRecordReportPath": str(decision_path),
            "decisionRecordReportSchema": _safe_text(decision_report.get("schema"))
            if decision_report
            else "",
            "decisionRecordReportStatus": _safe_text(decision_report.get("status"))
            if decision_report
            else "",
            "decision": _safe_text(decision_semantics.get("decision")),
            "requestedPaperIds": sorted(requested_papers),
            "runId": run_id,
            "expectedPolicyCandidateRows": expected_policy_rows,
            "expectedSectionDecisionRows": EXPECTED_SECTION_DECISION_ROWS,
            "expectedFigureCaptionDecisionRows": EXPECTED_FIGURE_CAPTION_DECISION_ROWS,
        },
        "counts": counts,
        "dryRunOnlyPolicyMatrix": policy_matrix,
        "gate": {
            "readyForEligibilityExecutorDryRun": status == "ok",
            "readyForEligibilityExecutorApply": False,
            "strictEligibleMutationAllowed": False,
            "eligibilityRecordWriteAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "sourceSpanStoreWriteAllowed": False,
            "runManifestWriteAllowed": False,
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": input_schema_violations,
            "decision": (
                "strict_evidence_eligibility_executor_dry_run_ready"
                if status == "ok"
                else "strict_evidence_eligibility_executor_dry_run_blocked"
            ),
            "recommendedNextTranche": (
                "strict_evidence_eligibility_executor_apply"
                if status == "ok"
                else "strict_evidence_eligibility_record_contract_repair"
            ),
        },
        "policy": {
            "reportOnly": True,
            "dryRunOnly": True,
            **policy_matrix,
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
            "dryRunOnlyPolicyMatrix",
            "gate",
            "policy",
            "warnings",
        )
        if key in report
    }


def render_strict_evidence_eligibility_executor_dry_run_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    matrix = dict(report.get("dryRunOnlyPolicyMatrix") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byDryRunStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Strict Evidence Eligibility Executor Dry Run",
            "",
            f"- status: {report.get('status', '')}",
            f"- decision: {gate.get('decision', '')}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- decision candidate rows: {int(counts.get('decisionCandidateRows') or 0)}",
            f"- dry-run ready eligibility record rows: {int(counts.get('dryRunReadyEligibilityRecordOnlyRows') or 0)}",
            f"- eligibility record writes: {int(counts.get('eligibilityRecordWriteRows') or 0)}",
            f"- strictEligible mutation rows: {int(counts.get('strictEligibleMutationRows') or 0)}",
            "",
            "## Dry-run policy matrix",
            f"- planned write target: {matrix.get('plannedWriteTarget', '')}",
            f"- write enabled: {json.dumps(matrix.get('writeEnabled'))}",
            f"- eligibility record write: {json.dumps(matrix.get('eligibilityRecordWrite'))}",
            "",
            "## Dry-run status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- recommended next tranche: {gate.get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_eligibility_executor_dry_run_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-eligibility-executor-dry-run.json"
    summary_path = root / "strict-evidence-eligibility-executor-dry-run-summary.json"
    markdown_path = root / "strict-evidence-eligibility-executor-dry-run.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_eligibility_executor_dry_run_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Dry-run planner for StrictEvidence eligibility records without writing "
            "eligibility JSONL or mutating evidence stores."
        )
    )
    parser.add_argument(
        "--eligibility-record-contract-report",
        default=str(DEFAULT_ELIGIBILITY_RECORD_CONTRACT_REPORT_PATH),
        help="Path to the eligibility record contract JSON report.",
    )
    parser.add_argument(
        "--decision-record-report",
        default=str(DEFAULT_DECISION_RECORD_REPORT_PATH),
        help="Path to the strictEligible mutation decision-record JSON report.",
    )
    parser.add_argument("--run-id", default="", help="Run id recorded on planned eligibility records.")
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, summary, and markdown reports.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_eligibility_executor_dry_run(
        eligibility_record_contract_report_path=args.eligibility_record_contract_report,
        decision_record_report_path=args.decision_record_report,
        run_id=args.run_id or None,
        paper_ids=args.paper_id or None,
    )
    paths = write_strict_evidence_eligibility_executor_dry_run_reports(report, args.output_dir)
    print(f"wrote report: {paths['report']}")
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote markdown: {paths['markdown']}")
    if args.json:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DECISION_RECORD_REPORT_PATH",
    "DEFAULT_ELIGIBILITY_RECORD_CONTRACT_REPORT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DRY_RUN_STATUS_READY",
    "STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID",
    "build_strict_evidence_eligibility_executor_dry_run",
    "render_strict_evidence_eligibility_executor_dry_run_markdown",
    "write_strict_evidence_eligibility_executor_dry_run_reports",
]
