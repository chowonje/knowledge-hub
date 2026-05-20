"""Dry-run planner for StrictEvidence citation-grade record writes.

Consumes the citation-grade record contract report and the citation-grade policy
gate design report, plans in-memory citation-grade records for candidate rows,
and validates schema plus semantic contracts with zero filesystem writes.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_citation_grade_policy_gate_design import (
    CITATION_GRADE_POLICY_DESIGN_STATUS_CANDIDATE_ONLY,
    STRICT_EVIDENCE_CITATION_GRADE_POLICY_GATE_DESIGN_SCHEMA_ID,
)
from knowledge_hub.papers.strict_evidence_citation_grade_record_contract import (
    CITATION_GRADE_DECISION,
    CITATION_GRADE_POLICY_VERSION,
    CITATION_GRADE_STATE_CANDIDATE_ONLY,
    STRICT_EVIDENCE_CITATION_GRADE_RECORD_CONTRACT_SCHEMA_ID,
    STRICT_EVIDENCE_CITATION_GRADE_RECORD_SCHEMA_ID,
    STRICT_EVIDENCE_CITATION_GRADE_STORE,
    build_sample_citation_grade_record_from_policy_row,
    validate_citation_grade_record_semantics,
)


STRICT_EVIDENCE_CITATION_GRADE_EXECUTOR_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-citation-grade-executor-dry-run.v1"
)

DRY_RUN_STATUS_READY = "dry_run_ready_citation_grade_record_only"
DRY_RUN_STATUS_BLOCKED_CONTRACT = "blocked_contract_not_ready"
DRY_RUN_STATUS_BLOCKED_POLICY_GATE = "blocked_policy_gate_design_not_ready"
DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID = "blocked_missing_strict_evidence_id"
DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_SPAN_ID = "blocked_missing_source_span_id"
DRY_RUN_STATUS_BLOCKED_MISSING_CANDIDATE_RECORD_ID = "blocked_missing_candidate_record_id"
DRY_RUN_STATUS_BLOCKED_MISSING_ELIGIBILITY_RECORD_ID = "blocked_missing_eligibility_record_id"
DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA = "blocked_planned_record_schema_violation"
DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC = "blocked_planned_record_semantic_violation"
DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

DEFAULT_CITATION_GRADE_RECORD_CONTRACT_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-citation-grade-record-contract"
    / "01-strict-evidence-citation-grade-record-contract"
    / "strict-evidence-citation-grade-record-contract.json"
)

DEFAULT_POLICY_GATE_DESIGN_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-citation-grade-policy-gate-design"
    / "01-strict-evidence-citation-grade-policy-gate-design"
    / "strict-evidence-citation-grade-policy-gate-design.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-citation-grade-executor-dry-run"
    / "01-strict-evidence-citation-grade-executor-dry-run"
)

EXPECTED_POLICY_CANDIDATE_ROWS = 99
EXPECTED_SECTION_POLICY_ROWS = 45
EXPECTED_FIGURE_CAPTION_POLICY_ROWS = 54


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
        "plannedWriteTarget": STRICT_EVIDENCE_CITATION_GRADE_STORE,
        "writeEnabled": False,
        "citationGradeRecordWrite": False,
        "citationGradeBooleanMutation": False,
        "eligibilityRecordWrite": False,
        "strictEvidenceStoreWrite": False,
        "sourceSpanStoreWrite": False,
        "strictEvidenceCreated": False,
        "strictEligibleMutation": False,
        "runtimeEvidenceCreated": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "vaultScan": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
        "manifestWrite": False,
    }


def _policy_row_is_candidate(policy_row: dict[str, Any]) -> bool:
    return (
        _safe_text(policy_row.get("citation_grade_policy_design_status"))
        == CITATION_GRADE_POLICY_DESIGN_STATUS_CANDIDATE_ONLY
        and _safe_bool(policy_row.get("citationGradePolicyDesignCandidateOnly"))
        and _safe_bool(policy_row.get("citationGradeRecordRequired"))
        and not _safe_bool(policy_row.get("citationGradeRecordWriteAllowed"))
        and not _safe_bool(policy_row.get("citationGradeBooleanMutationAllowed"))
        and not _safe_bool(policy_row.get("citationGradeAllowed"))
        and not _safe_bool(policy_row.get("runtimeEvidenceAllowed"))
    )


def _policy_row_flag_violations(policy_row: dict[str, Any]) -> list[str]:
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
        if _safe_bool(policy_row.get(field_name)):
            violations.append(f"policy_row.{field_name}_true")
    for field_name in (
        "citationGradeRecordWriteAllowed",
        "citationGradeBooleanMutationAllowed",
        "citationGradeAllowed",
        "runtimeEvidenceAllowed",
        "runtimeVisibleAllowed",
        "parserRoutingAllowed",
        "answerIntegrationAllowed",
        "databaseMutationAllowed",
        "reindexOrReembedAllowed",
    ):
        if _safe_bool(policy_row.get(field_name)):
            violations.append(f"policy_row.{field_name}_true")
    return violations


def _classify_dry_run_row(
    policy_row: dict[str, Any],
    *,
    input_schema_violations: list[str],
    contract_ready: bool,
    policy_gate_ready: bool,
    run_id: str,
) -> tuple[str, list[str], dict[str, Any] | None]:
    if input_schema_violations:
        return DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA, list(input_schema_violations), None
    if not contract_ready:
        return DRY_RUN_STATUS_BLOCKED_CONTRACT, ["citation_grade_record_contract_not_ready"], None
    if not policy_gate_ready:
        return DRY_RUN_STATUS_BLOCKED_POLICY_GATE, ["citation_grade_policy_gate_design_not_ready"], None
    if not _policy_row_is_candidate(policy_row):
        blockers = [_safe_text(item) for item in (policy_row.get("citation_grade_policy_design_blockers") or [])]
        blockers.append(
            "citation_grade_policy_design_status="
            f"{_safe_text(policy_row.get('citation_grade_policy_design_status')) or 'unknown'}"
        )
        return DRY_RUN_STATUS_BLOCKED_POLICY_GATE, _dedupe(blockers), None

    strict_evidence_id = _safe_text(policy_row.get("strictEvidenceId"))
    source_span_id = _safe_text(policy_row.get("sourceSpanId"))
    candidate_record_id = _safe_text(policy_row.get("candidateRecordId"))
    eligibility_record_id = _safe_text(policy_row.get("eligibilityRecordId"))
    if not strict_evidence_id:
        return DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID, ["strictEvidenceId_missing"], None
    if not source_span_id:
        return DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_SPAN_ID, ["sourceSpanId_missing"], None
    if not candidate_record_id:
        return DRY_RUN_STATUS_BLOCKED_MISSING_CANDIDATE_RECORD_ID, ["candidateRecordId_missing"], None
    if not eligibility_record_id:
        return DRY_RUN_STATUS_BLOCKED_MISSING_ELIGIBILITY_RECORD_ID, ["eligibilityRecordId_missing"], None

    flag_violations = _policy_row_flag_violations(policy_row)
    if flag_violations:
        return DRY_RUN_STATUS_BLOCKED_POLICY_GATE, flag_violations, None

    planned_record = build_sample_citation_grade_record_from_policy_row(
        policy_row,
        run_id=run_id,
    )
    schema_validation = validate_payload(
        planned_record,
        STRICT_EVIDENCE_CITATION_GRADE_RECORD_SCHEMA_ID,
        strict=True,
    )
    if not schema_validation.ok:
        return (
            DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA,
            [str(error) for error in schema_validation.errors],
            planned_record,
        )

    semantic_errors = validate_citation_grade_record_semantics(planned_record)
    if semantic_errors:
        return DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC, semantic_errors, planned_record

    return DRY_RUN_STATUS_READY, [], planned_record


def _planned_executor_key(policy_row: dict[str, Any]) -> str:
    strict_evidence_id = _safe_text(policy_row.get("strictEvidenceId"))
    return f"citation-grade-executor:{strict_evidence_id or 'unknown'}"


def _dry_run_rows(
    policy_rows: list[dict[str, Any]],
    *,
    input_schema_violations: list[str],
    contract_ready: bool,
    policy_gate_ready: bool,
    run_id: str,
) -> list[dict[str, Any]]:
    matrix = _no_mutation_policy_matrix()
    rows: list[dict[str, Any]] = []
    for index, policy_row in enumerate(policy_rows):
        source_row = dict(policy_row or {})
        dry_run_status, blockers, planned_record = _classify_dry_run_row(
            source_row,
            input_schema_violations=input_schema_violations,
            contract_ready=contract_ready,
            policy_gate_ready=policy_gate_ready,
            run_id=run_id,
        )
        ready = dry_run_status == DRY_RUN_STATUS_READY and planned_record is not None
        record = planned_record or {}
        rows.append(
            {
                "dry_run_row_id": f"strict-evidence-citation-grade-executor-dry-run:{index:04d}",
                "policy_design_row_id": _safe_text(source_row.get("policy_design_row_id")),
                "hold_row_id": _safe_text(source_row.get("hold_row_id")),
                "strictEvidenceId": _safe_text(source_row.get("strictEvidenceId")),
                "sourceSpanId": _safe_text(source_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(source_row.get("candidateRecordId")),
                "eligibilityRecordId": _safe_text(source_row.get("eligibilityRecordId")),
                "paper_id": _safe_text(source_row.get("paper_id")),
                "artifact_type": _safe_text(source_row.get("artifact_type")),
                "dry_run_status": dry_run_status,
                "dry_run_blockers": _dedupe(blockers),
                "dryRunReadyCitationGradeRecordOnly": ready,
                "plannedExecutorKey": _planned_executor_key(source_row),
                "plannedWriteTarget": STRICT_EVIDENCE_CITATION_GRADE_STORE if ready else "",
                "plannedCitationGradePolicyVersion": (
                    CITATION_GRADE_POLICY_VERSION if ready else ""
                ),
                "plannedCitationGradeDecision": CITATION_GRADE_DECISION if ready else "",
                "plannedCitationGradeState": CITATION_GRADE_STATE_CANDIDATE_ONLY if ready else "",
                "plannedCitationGradeRecord": record,
                "citationGradeRecordWrite": False,
                "citationGradeBooleanMutation": False,
                "eligibilityRecordWrite": False,
                "strictEvidenceStoreWrite": False,
                "sourceSpanStoreWrite": False,
                "strictEligibleMutation": False,
                "strictEvidenceCreated": False,
                "runtimeEvidenceCreated": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "vaultScan": False,
                "reindexOrReembed": False,
                "canonicalParsedArtifactsWritten": False,
                "manifestWrite": False,
                "policyMatrix": matrix,
                "recommended_action": (
                    "queue_for_citation_grade_executor_apply_dry_run_review"
                    if ready
                    else "repair_citation_grade_executor_dry_run_input"
                ),
            }
        )
    return rows


def _count_rows(rows: list[dict[str, Any]], *, schema_violations: list[str]) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("dry_run_status")) for row in rows)
    ready_rows = [row for row in rows if _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_READY]
    return {
        "inputRows": len(rows),
        "policyCandidateRows": sum(
            1
            for row in rows
            if _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_READY
            or _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA
            or _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC
        ),
        "dryRunReadyCitationGradeRecordOnlyRows": int(by_status.get(DRY_RUN_STATUS_READY, 0)),
        "blockedContractNotReadyRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_CONTRACT, 0)),
        "blockedPolicyGateDesignNotReadyRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_POLICY_GATE, 0)),
        "blockedMissingStrictEvidenceIdRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID, 0)),
        "blockedMissingSourceSpanIdRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_SOURCE_SPAN_ID, 0)),
        "blockedMissingCandidateRecordIdRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_CANDIDATE_RECORD_ID, 0)),
        "blockedMissingEligibilityRecordIdRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_ELIGIBILITY_RECORD_ID, 0)),
        "blockedPlannedRecordSchemaViolationRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_PLANNED_SCHEMA, 0)),
        "blockedPlannedRecordSemanticViolationRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_PLANNED_SEMANTIC, 0)),
        "blockedInputSchemaViolationRows": int(by_status.get(DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA, 0)),
        "citationGradeRecordWriteRows": 0,
        "citationGradeBooleanMutationRows": 0,
        "eligibilityRecordWriteRows": 0,
        "strictEligibleMutationRows": 0,
        "strictEvidenceWriteRows": 0,
        "strictEvidenceCreatedRows": 0,
        "sourceSpanUpdatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "manifestWriteRows": 0,
        "reindexOrReembedRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaperId": dict(Counter(_safe_text(row.get("paper_id")) for row in ready_rows)),
        "byArtifactType": dict(Counter(_safe_text(row.get("artifact_type")) for row in ready_rows)),
        "byDryRunStatus": dict(by_status),
        "byRecommendedAction": dict(Counter(_safe_text(row.get("recommended_action")) for row in rows)),
    }


def build_strict_evidence_citation_grade_executor_dry_run(
    *,
    citation_grade_record_contract_report_path: str | Path = DEFAULT_CITATION_GRADE_RECORD_CONTRACT_REPORT_PATH,
    policy_gate_design_report_path: str | Path = DEFAULT_POLICY_GATE_DESIGN_REPORT_PATH,
    paper_ids: list[str] | None = None,
    run_id: str = "strict-evidence-citation-grade-executor-dry-run-20260520",
    expected_policy_candidate_rows: int = EXPECTED_POLICY_CANDIDATE_ROWS,
    expected_section_policy_rows: int = EXPECTED_SECTION_POLICY_ROWS,
    expected_figure_caption_policy_rows: int = EXPECTED_FIGURE_CAPTION_POLICY_ROWS,
) -> dict[str, Any]:
    contract_path = Path(str(citation_grade_record_contract_report_path)).expanduser()
    policy_path = Path(str(policy_gate_design_report_path)).expanduser()
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}
    warnings: list[str] = []
    input_schema_violations: list[str] = []

    contract_report = _read_json(contract_path)
    policy_report = _read_json(policy_path)

    contract_ready = False
    if contract_report:
        validation = validate_payload(
            contract_report,
            STRICT_EVIDENCE_CITATION_GRADE_RECORD_CONTRACT_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            input_schema_violations.extend(str(error) for error in validation.errors)
        gate = contract_report.get("gate") if isinstance(contract_report.get("gate"), dict) else {}
        contract_ready = (
            _safe_text(contract_report.get("status")) == "ok"
            and _safe_text(gate.get("decision")) == "strict_evidence_citation_grade_record_contract_ready"
            and not _safe_bool(gate.get("executorReady"))
            and not _safe_bool(gate.get("citationGradeRecordWriteAllowed"))
            and not _safe_bool(gate.get("citationGradeBooleanMutationAllowed"))
        )
    else:
        input_schema_violations.append("citation_grade_record_contract_report_missing_or_unreadable")

    policy_gate_ready = False
    policy_rows: list[dict[str, Any]] = []
    if policy_report:
        validation = validate_payload(
            policy_report,
            STRICT_EVIDENCE_CITATION_GRADE_POLICY_GATE_DESIGN_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            input_schema_violations.extend(str(error) for error in validation.errors)
        gate = policy_report.get("gate") if isinstance(policy_report.get("gate"), dict) else {}
        counts = policy_report.get("counts") if isinstance(policy_report.get("counts"), dict) else {}
        policy_gate_ready = (
            _safe_text(policy_report.get("status")) == "ok"
            and _safe_bool(gate.get("citationGradePolicyGateDesignReady"))
            and _safe_text(gate.get("decision"))
            == "strict_evidence_citation_grade_policy_gate_design_candidate_only"
            and _safe_int(counts.get("citationGradePolicyDesignCandidateOnlyRows"))
            == expected_policy_candidate_rows
            and _safe_int(counts.get("sectionCitationGradePolicyDesignRows"))
            == expected_section_policy_rows
            and _safe_int(counts.get("figureCaptionCitationGradePolicyDesignRows"))
            == expected_figure_caption_policy_rows
            and _safe_int(counts.get("citationGradeRecordWriteRows")) == 0
            and _safe_int(counts.get("citationGradeEvidenceCreatedRows")) == 0
            and _safe_int(counts.get("runtimeEvidenceCreatedRows")) == 0
        )
        policy_rows = [row for row in policy_report.get("rows", []) if isinstance(row, dict)]
    else:
        input_schema_violations.append("citation_grade_policy_gate_design_report_missing_or_unreadable")

    if requested_papers:
        found = {_safe_text(row.get("paper_id")) for row in policy_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found:
            warnings.append("requested_paper_ids_not_found_in_policy_gate_design")
        policy_rows = [row for row in policy_rows if _safe_text(row.get("paper_id")) in requested_papers]

    rows = _dry_run_rows(
        policy_rows,
        input_schema_violations=_dedupe(input_schema_violations),
        contract_ready=contract_ready,
        policy_gate_ready=policy_gate_ready,
        run_id=run_id,
    )
    counts = _count_rows(rows, schema_violations=_dedupe(input_schema_violations))
    status = "ok"
    if (
        input_schema_violations
        or not rows
        or counts["dryRunReadyCitationGradeRecordOnlyRows"] != len(rows)
        or counts["dryRunReadyCitationGradeRecordOnlyRows"] != expected_policy_candidate_rows
    ):
        status = "blocked"

    matrix = _no_mutation_policy_matrix()
    return {
        "schema": STRICT_EVIDENCE_CITATION_GRADE_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "citationGradeRecordContractReportPath": str(contract_path),
            "citationGradeRecordContractReportSchema": _safe_text(contract_report.get("schema"))
            if contract_report
            else "",
            "citationGradeRecordContractReportStatus": _safe_text(contract_report.get("status"))
            if contract_report
            else "",
            "policyGateDesignReportPath": str(policy_path),
            "policyGateDesignReportSchema": _safe_text(policy_report.get("schema"))
            if policy_report
            else "",
            "policyGateDesignReportStatus": _safe_text(policy_report.get("status"))
            if policy_report
            else "",
            "requestedPaperIds": sorted(requested_papers),
            "runId": run_id,
            "expectedPolicyCandidateRows": expected_policy_candidate_rows,
            "expectedSectionPolicyRows": expected_section_policy_rows,
            "expectedFigureCaptionPolicyRows": expected_figure_caption_policy_rows,
        },
        "counts": counts,
        "dryRunOnlyPolicyMatrix": matrix,
        "gate": {
            "readyForCitationGradeExecutorDryRun": status == "ok",
            "readyForCitationGradeExecutorApply": False,
            "citationGradeRecordWriteAllowed": False,
            "citationGradeBooleanMutationAllowed": False,
            "eligibilityRecordWriteAllowed": False,
            "strictEligibleMutationAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "sourceSpanStoreWriteAllowed": False,
            "runManifestWriteAllowed": False,
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "strict_evidence_citation_grade_executor_dry_run_ready"
                if status == "ok"
                else "strict_evidence_citation_grade_executor_dry_run_blocked"
            ),
            "recommendedNextTranche": (
                "strict_evidence_citation_grade_executor_apply"
                if status == "ok"
                else "strict_evidence_citation_grade_executor_dry_run_repair"
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


def render_strict_evidence_citation_grade_executor_dry_run_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byDryRunStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Strict Evidence Citation-Grade Executor Dry Run",
            "",
            f"- status: {report.get('status', '')}",
            f"- decision: {gate.get('decision', '')}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- policy candidate rows: {int(counts.get('policyCandidateRows') or 0)}",
            f"- dry-run ready rows: {int(counts.get('dryRunReadyCitationGradeRecordOnlyRows') or 0)}",
            f"- citation-grade record writes: {int(counts.get('citationGradeRecordWriteRows') or 0)}",
            f"- citationGrade boolean mutation rows: {int(counts.get('citationGradeBooleanMutationRows') or 0)}",
            f"- runtime evidence rows: {int(counts.get('runtimeEvidenceCreatedRows') or 0)}",
            "",
            "## Dry-run status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- recommended next tranche: {gate.get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_citation_grade_executor_dry_run_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-citation-grade-executor-dry-run.json"
    summary_path = root / "strict-evidence-citation-grade-executor-dry-run-summary.json"
    markdown_path = root / "strict-evidence-citation-grade-executor-dry-run.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_citation_grade_executor_dry_run_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Plan StrictEvidence citation-grade record writes in dry-run mode without "
            "writing citation-grade records or mutating evidence stores."
        )
    )
    parser.add_argument(
        "--citation-grade-record-contract-report",
        default=str(DEFAULT_CITATION_GRADE_RECORD_CONTRACT_REPORT_PATH),
        help="Path to the citation-grade record contract JSON report.",
    )
    parser.add_argument(
        "--policy-gate-design-report",
        default=str(DEFAULT_POLICY_GATE_DESIGN_REPORT_PATH),
        help="Path to the citation-grade policy gate design JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--run-id", default="strict-evidence-citation-grade-executor-dry-run-20260520")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, summary, and markdown reports.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_citation_grade_executor_dry_run(
        citation_grade_record_contract_report_path=args.citation_grade_record_contract_report,
        policy_gate_design_report_path=args.policy_gate_design_report,
        paper_ids=args.paper_id or None,
        run_id=args.run_id,
    )
    paths = write_strict_evidence_citation_grade_executor_dry_run_reports(report, args.output_dir)
    print(f"wrote report: {paths['report']}")
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote markdown: {paths['markdown']}")
    if args.json:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CITATION_GRADE_RECORD_CONTRACT_REPORT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_POLICY_GATE_DESIGN_REPORT_PATH",
    "DRY_RUN_STATUS_BLOCKED_CONTRACT",
    "DRY_RUN_STATUS_BLOCKED_MISSING_ELIGIBILITY_RECORD_ID",
    "DRY_RUN_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE_ID",
    "DRY_RUN_STATUS_BLOCKED_POLICY_GATE",
    "DRY_RUN_STATUS_READY",
    "STRICT_EVIDENCE_CITATION_GRADE_EXECUTOR_DRY_RUN_SCHEMA_ID",
    "build_strict_evidence_citation_grade_executor_dry_run",
    "render_strict_evidence_citation_grade_executor_dry_run_markdown",
    "write_strict_evidence_citation_grade_executor_dry_run_reports",
]
