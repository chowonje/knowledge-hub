"""Contract-only write target definition for StrictEvidence citation-grade records.

Defines the append-only citation-grade record and store contract after the
citation-grade policy gate design. This helper is report-only: it does not
write citation-grade JSONL, mutate eligibility/StrictEvidence/SourceSpan
stores, set citationGrade=true, or enable runtime/answer surfaces.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_citation_grade_policy_gate_design import (
    CITATION_GRADE_POLICY_DESIGN_STATUS_CANDIDATE_ONLY,
    STRICT_EVIDENCE_CITATION_GRADE_POLICY_GATE_DESIGN_SCHEMA_ID,
)


STRICT_EVIDENCE_CITATION_GRADE_RECORD_CONTRACT_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-citation-grade-record-contract.v1"
)
STRICT_EVIDENCE_CITATION_GRADE_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-citation-grade-record.v1"
)

STRICT_EVIDENCE_CITATION_GRADE_STORE = "parsed_artifact_strict_evidence_citation_grade_store"
CITATION_GRADE_POLICY_VERSION = "strict_evidence_citation_grade_policy.v1"
CITATION_GRADE_STATE_CANDIDATE_ONLY = "citation_grade_candidate_only"
CITATION_GRADE_DECISION = "citation_grade_policy_candidate_only"

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
    / "strict-evidence-citation-grade-record-contract"
    / "01-strict-evidence-citation-grade-record-contract"
)

NO_MUTATION_POLICY = {
    "contractOnly": True,
    "executorImplemented": False,
    "citationGradeRecordWrite": False,
    "citationGradeBooleanMutation": False,
    "eligibilityRecordWrite": False,
    "strictEligibleMutation": False,
    "strictEvidenceStoreWrite": False,
    "sourceSpanStoreWrite": False,
    "strictEvidenceCreated": False,
    "runtimeEvidenceCreated": False,
    "parserRoutingChanged": False,
    "answerIntegrationChanged": False,
    "databaseMutation": False,
    "vaultScan": False,
    "reindexOrReembed": False,
    "canonicalParsedArtifactsWritten": False,
    "manifestWrite": False,
}

REQUIRED_RECORD_FIELDS = [
    "schema",
    "citationGradeRecordId",
    "runId",
    "plannedWriteTarget",
    "paperId",
    "artifactType",
    "strictEvidenceId",
    "sourceSpanId",
    "candidateRecordId",
    "eligibilityRecordId",
    "policyDesignRowId",
    "holdRowId",
    "citationGradePolicyVersion",
    "citationGradeDecision",
    "citationGradeState",
    "strictEvidenceInPlaceMutationAllowed",
    "eligibilityRecordInPlaceMutationAllowed",
    "sourceSpanInPlaceMutationAllowed",
    "citationGradeBooleanMutationAllowed",
    "citationGradeMutationApplied",
    "runtimeEvidence",
    "runtimeVisible",
    "idempotencyKey",
    "provenanceTrace",
    "writePolicy",
]

IDEMPOTENCY_KEY_FIELDS = [
    "plannedWriteTarget",
    "paperId",
    "artifactType",
    "strictEvidenceId",
    "eligibilityRecordId",
    "sourceSpanId",
    "candidateRecordId",
    "citationGradePolicyVersion",
    "citationGradeDecision",
    "idempotencyKey",
]

CITATION_GRADE_STORE_CONTRACT: dict[str, Any] = {
    "plannedWriteTarget": STRICT_EVIDENCE_CITATION_GRADE_STORE,
    "contractReference": STRICT_EVIDENCE_CITATION_GRADE_RECORD_CONTRACT_SCHEMA_ID,
    "citationGradeRecordSchema": STRICT_EVIDENCE_CITATION_GRADE_RECORD_SCHEMA_ID,
    "storeKind": "local_papers_dir_jsonl_strict_evidence_citation_grade_store",
    "storeRootTemplate": "{papers_dir}/structured_evidence/strict_evidence_citation_grade",
    "recordPathTemplate": "{papers_dir}/structured_evidence/strict_evidence_citation_grade/{paper_id}.jsonl",
    "runManifestPathTemplate": "{papers_dir}/structured_evidence/runs/{run_id}.json",
    "allowedArtifactTypes": ["section", "figure"],
    "requiredRecordFields": REQUIRED_RECORD_FIELDS,
    "idempotencyKeyFields": IDEMPOTENCY_KEY_FIELDS,
    "writeSemantics": "explicit_apply_executor_appends_or_replaces_same_idempotency_key",
    "readbackChecks": [
        "citation_grade_record_schema_validates",
        "idempotency_key_stable",
        "strictEvidenceId_resolves_to_existing_strict_evidence_jsonl",
        "eligibilityRecordId_resolves_to_existing_eligibility_jsonl",
        "sourceSpanId_resolves_to_existing_source_span_jsonl",
        "candidateRecordId_preserved",
        "strict_evidence_record_remains_unmutated",
        "eligibility_record_remains_unmutated",
        "citationGrade_boolean_remains_false_on_parent_records",
        "citation_grade_record_remains_non_runtime",
    ],
    "rollbackStrategy": (
        "delete or invalidate citation-grade records written by the explicit run_id "
        "while no downstream runtime binding or answer integration references them"
    ),
    "rollbackImplemented": False,
    "executorImplemented": False,
    "strictEvidenceInPlaceMutationAllowed": False,
    "eligibilityRecordInPlaceMutationAllowed": False,
    "sourceSpanInPlaceMutationAllowed": False,
    "citationGradeBooleanMutationAllowed": False,
    "runtimeUseAllowed": False,
    "parserRoutingAllowed": False,
    "answerIntegrationAllowed": False,
    "databaseMutationAllowed": False,
    "sourceSpanMutationAllowed": False,
}

KNOWN_WRITE_TARGET_CONTRACTS: dict[str, str] = {
    STRICT_EVIDENCE_CITATION_GRADE_STORE: STRICT_EVIDENCE_CITATION_GRADE_RECORD_CONTRACT_SCHEMA_ID,
}


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


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(str(path)).expanduser()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_policy() -> dict[str, Any]:
    return {
        "executorRequired": True,
        "citationGradeRecordWrite": False,
        "citationGradeBooleanMutation": False,
        "eligibilityRecordWrite": False,
        "strictEvidenceStoreWrite": False,
        "sourceSpanStoreWrite": False,
        "strictEligibleMutation": False,
        "runtimeEvidenceCreated": False,
        "databaseMutation": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
        "manifestWrite": False,
    }


def validate_citation_grade_record_semantics(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if _safe_text(record.get("plannedWriteTarget")) != STRICT_EVIDENCE_CITATION_GRADE_STORE:
        errors.append("plannedWriteTarget_must_be_strict_evidence_citation_grade_store")
    if _safe_text(record.get("citationGradePolicyVersion")) != CITATION_GRADE_POLICY_VERSION:
        errors.append("citationGradePolicyVersion_mismatch")
    if _safe_text(record.get("citationGradeDecision")) != CITATION_GRADE_DECISION:
        errors.append("citationGradeDecision_mismatch")
    if _safe_bool(record.get("strictEvidenceInPlaceMutationAllowed")):
        errors.append("strictEvidenceInPlaceMutationAllowed_must_be_false")
    if _safe_bool(record.get("eligibilityRecordInPlaceMutationAllowed")):
        errors.append("eligibilityRecordInPlaceMutationAllowed_must_be_false")
    if _safe_bool(record.get("sourceSpanInPlaceMutationAllowed")):
        errors.append("sourceSpanInPlaceMutationAllowed_must_be_false")
    if _safe_bool(record.get("citationGradeBooleanMutationAllowed")):
        errors.append("citationGradeBooleanMutationAllowed_must_be_false")
    if _safe_bool(record.get("citationGradeMutationApplied")):
        errors.append("citationGradeMutationApplied_must_be_false")
    if _safe_bool(record.get("runtimeEvidence")):
        errors.append("runtimeEvidence_must_be_false")
    if _safe_bool(record.get("runtimeVisible")):
        errors.append("runtimeVisible_must_be_false")
    for field_name in (
        "strictEvidenceId",
        "sourceSpanId",
        "candidateRecordId",
        "eligibilityRecordId",
    ):
        if not _safe_text(record.get(field_name)):
            errors.append(f"{field_name}_must_be_non_empty")
    return errors


def build_sample_citation_grade_record_from_policy_row(
    policy_row: dict[str, Any],
    *,
    run_id: str = "citation-grade-contract-sample-run",
    citation_grade_record_id: str | None = None,
) -> dict[str, Any]:
    paper_id = _safe_text(policy_row.get("paper_id"))
    artifact_type = _safe_text(policy_row.get("artifact_type"))
    strict_evidence_id = _safe_text(policy_row.get("strictEvidenceId"))
    source_span_id = _safe_text(policy_row.get("sourceSpanId"))
    candidate_record_id = _safe_text(policy_row.get("candidateRecordId"))
    eligibility_record_id = _safe_text(policy_row.get("eligibilityRecordId"))
    policy_row_id = _safe_text(policy_row.get("policy_design_row_id"))
    hold_row_id = _safe_text(policy_row.get("hold_row_id"))
    record_id = citation_grade_record_id or f"strict-evidence-citation-grade:{strict_evidence_id}"
    idempotency_key = (
        f"strict-citation-grade:{strict_evidence_id}:{eligibility_record_id}:"
        f"{CITATION_GRADE_POLICY_VERSION}:{CITATION_GRADE_DECISION}"
    )

    return {
        "schema": STRICT_EVIDENCE_CITATION_GRADE_RECORD_SCHEMA_ID,
        "citationGradeRecordId": record_id,
        "runId": run_id,
        "plannedWriteTarget": STRICT_EVIDENCE_CITATION_GRADE_STORE,
        "paperId": paper_id,
        "artifactType": artifact_type,
        "strictEvidenceId": strict_evidence_id,
        "sourceSpanId": source_span_id,
        "candidateRecordId": candidate_record_id,
        "eligibilityRecordId": eligibility_record_id,
        "policyDesignRowId": policy_row_id,
        "holdRowId": hold_row_id,
        "citationGradePolicyVersion": CITATION_GRADE_POLICY_VERSION,
        "citationGradeDecision": CITATION_GRADE_DECISION,
        "citationGradeState": CITATION_GRADE_STATE_CANDIDATE_ONLY,
        "strictEvidenceInPlaceMutationAllowed": False,
        "eligibilityRecordInPlaceMutationAllowed": False,
        "sourceSpanInPlaceMutationAllowed": False,
        "citationGradeBooleanMutationAllowed": False,
        "citationGradeMutationApplied": False,
        "runtimeEvidence": False,
        "runtimeVisible": False,
        "idempotencyKey": idempotency_key,
        "provenanceTrace": {
            "strictEvidenceId": strict_evidence_id,
            "sourceSpanId": source_span_id,
            "candidateRecordId": candidate_record_id,
            "eligibilityRecordId": eligibility_record_id,
            "policyDesignRowId": policy_row_id,
            "holdRowId": hold_row_id,
            "citationGradePolicyDesignStatus": _safe_text(
                policy_row.get("citation_grade_policy_design_status")
            ),
        },
        "writePolicy": _write_policy(),
    }


def build_strict_evidence_citation_grade_record_contract(
    *,
    policy_gate_design_report_path: str | Path = DEFAULT_POLICY_GATE_DESIGN_REPORT_PATH,
) -> dict[str, Any]:
    report_path = Path(str(policy_gate_design_report_path)).expanduser()
    policy_gate = _read_json(report_path)
    warnings: list[str] = []
    schema_violations: list[str] = []

    if policy_gate:
        validation = validate_payload(
            policy_gate,
            STRICT_EVIDENCE_CITATION_GRADE_POLICY_GATE_DESIGN_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            schema_violations.extend(str(error) for error in validation.errors)
    else:
        warnings.append("policy_gate_design_report_missing_or_unreadable")

    counts = policy_gate.get("counts") if isinstance(policy_gate.get("counts"), dict) else {}
    policy_rows = [
        row for row in policy_gate.get("rows", []) if isinstance(row, dict)
    ] if policy_gate else []
    design = (
        policy_gate.get("citationGradePolicyDesign")
        if isinstance(policy_gate.get("citationGradePolicyDesign"), dict)
        else {}
    )

    status = "ok"
    if (
        schema_violations
        or not policy_gate
        or _safe_text(design.get("decision")) != "separate_append_only_citation_grade_record"
        or not _safe_bool(design.get("citationGradeRecordContractRequired"))
        or _safe_int(counts.get("citationGradePolicyDesignCandidateOnlyRows")) <= 0
        or _safe_int(counts.get("citationGradeRecordWriteRows")) != 0
        or _safe_int(counts.get("citationGradeEvidenceCreatedRows")) != 0
        or _safe_int(counts.get("runtimeEvidenceCreatedRows")) != 0
    ):
        status = "blocked"

    candidate_rows = _safe_int(counts.get("citationGradePolicyDesignCandidateOnlyRows"))
    section_rows = _safe_int(counts.get("sectionCitationGradePolicyDesignRows"))
    figure_rows = _safe_int(counts.get("figureCaptionCitationGradePolicyDesignRows"))
    sample_record = (
        build_sample_citation_grade_record_from_policy_row(policy_rows[0])
        if policy_rows
        and _safe_text(policy_rows[0].get("citation_grade_policy_design_status"))
        == CITATION_GRADE_POLICY_DESIGN_STATUS_CANDIDATE_ONLY
        else {}
    )
    sample_record_semantic_errors = (
        validate_citation_grade_record_semantics(sample_record) if sample_record else []
    )
    sample_record_schema_ok = False
    if sample_record:
        sample_record_schema_ok = validate_payload(
            sample_record,
            STRICT_EVIDENCE_CITATION_GRADE_RECORD_SCHEMA_ID,
            strict=True,
        ).ok

    return {
        "schema": STRICT_EVIDENCE_CITATION_GRADE_RECORD_CONTRACT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "policyGateDesignReportPath": str(report_path),
            "policyGateDesignSchema": _safe_text(policy_gate.get("schema")) if policy_gate else "",
            "policyGateDesignStatus": _safe_text(policy_gate.get("status")) if policy_gate else "",
            "policyGateDecision": _safe_text(design.get("decision")),
            "citationGradePolicyDesignCandidateOnlyRows": candidate_rows,
            "sectionCitationGradePolicyDesignRows": section_rows,
            "figureCaptionCitationGradePolicyDesignRows": figure_rows,
        },
        "counts": {
            "writeTargetContracts": 1,
            "citationGradeStoreContracts": 1,
            "citationGradeRecordSchemas": 1,
            "citationGradePolicyDesignCandidateOnlyRows": candidate_rows,
            "sectionCitationGradePolicyDesignRows": section_rows,
            "figureCaptionCitationGradePolicyDesignRows": figure_rows,
            "executorImplementedRows": 0,
            "citationGradeRecordWriteRows": 0,
            "citationGradeBooleanMutationRows": 0,
            "eligibilityRecordWriteRows": 0,
            "strictEligibleMutationRows": 0,
            "strictEvidenceWriteRows": 0,
            "sourceSpanUpdatedRows": 0,
            "strictEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "manifestWriteRows": 0,
            "reindexOrReembedRows": 0,
            "schemaViolationCount": len(schema_violations),
            "sampleRecordSchemaValidRows": 1 if sample_record_schema_ok else 0,
            "sampleRecordSemanticViolationRows": len(sample_record_semantic_errors),
        },
        "gate": {
            "writeTargetContractsDefined": True,
            "citationGradeStoreContractDefined": True,
            "citationGradeRecordSchemaDefined": True,
            "executorReady": False,
            "runtimeMutationAllowed": False,
            "citationGradeRecordWriteAllowed": False,
            "citationGradeBooleanMutationAllowed": False,
            "strictEligibleMutationAllowed": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "decision": (
                "strict_evidence_citation_grade_record_contract_ready"
                if status == "ok"
                else "strict_evidence_citation_grade_record_contract_blocked"
            ),
            "schemaViolations": schema_violations,
            "sampleRecordSemanticViolations": sample_record_semantic_errors,
            "recommendedNextTranche": (
                "strict_evidence_citation_grade_executor_dry_run"
                if status == "ok"
                else "strict_evidence_citation_grade_policy_gate_design_repair"
            ),
        },
        "policy": dict(NO_MUTATION_POLICY),
        "contractPrinciples": [
            "citation-grade records are append-only promotion metadata",
            "StrictEvidence, eligibility, and SourceSpan rows remain immutable under this contract",
            "citationGrade booleans remain false on parent records",
            "citation-grade is not runtime evidence, parser routing, or answer integration",
            "rollback targets citation-grade records by explicit run_id before downstream references exist",
        ],
        "writeTargets": [dict(CITATION_GRADE_STORE_CONTRACT)],
        "sampleCitationGradeRecord": sample_record,
        "warnings": warnings,
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
            "contractPrinciples",
            "writeTargets",
            "warnings",
        )
        if key in report
    }


def render_strict_evidence_citation_grade_record_contract_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    write_target = (report.get("writeTargets") or [{}])[0]
    principles = [f"- {item}" for item in list(report.get("contractPrinciples") or [])]
    return "\n".join(
        [
            "# Strict Evidence Citation-Grade Record Contract",
            "",
            f"- status: {report.get('status', '')}",
            f"- decision: {gate.get('decision', '')}",
            f"- planned write target: {write_target.get('plannedWriteTarget', '')}",
            f"- record path template: {write_target.get('recordPathTemplate', '')}",
            f"- policy design candidate rows: {int(counts.get('citationGradePolicyDesignCandidateOnlyRows') or 0)}",
            f"- section design rows: {int(counts.get('sectionCitationGradePolicyDesignRows') or 0)}",
            f"- figure caption design rows: {int(counts.get('figureCaptionCitationGradePolicyDesignRows') or 0)}",
            f"- citation-grade record writes: {int(counts.get('citationGradeRecordWriteRows') or 0)}",
            f"- citationGrade boolean mutation rows: {int(counts.get('citationGradeBooleanMutationRows') or 0)}",
            "",
            "## Contract principles",
            *principles,
            "",
            "## Readback checks",
            *[f"- {item}" for item in list(write_target.get("readbackChecks") or [])],
            "",
            f"- recommended next tranche: {gate.get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_citation_grade_record_contract_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-citation-grade-record-contract.json"
    summary_path = root / "strict-evidence-citation-grade-record-contract-summary.json"
    markdown_path = root / "strict-evidence-citation-grade-record-contract.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_citation_grade_record_contract_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Define the StrictEvidence citation-grade record/store contract without writing "
            "citation-grade records or mutating evidence stores."
        )
    )
    parser.add_argument(
        "--policy-gate-design-report",
        default=str(DEFAULT_POLICY_GATE_DESIGN_REPORT_PATH),
        help="Path to the citation-grade policy gate design JSON report.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, summary, and markdown reports.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_citation_grade_record_contract(
        policy_gate_design_report_path=args.policy_gate_design_report,
    )
    paths = write_strict_evidence_citation_grade_record_contract_reports(report, args.output_dir)
    print(f"wrote report: {paths['report']}")
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote markdown: {paths['markdown']}")
    if args.json:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CITATION_GRADE_DECISION",
    "CITATION_GRADE_POLICY_VERSION",
    "CITATION_GRADE_STATE_CANDIDATE_ONLY",
    "CITATION_GRADE_STORE_CONTRACT",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_POLICY_GATE_DESIGN_REPORT_PATH",
    "KNOWN_WRITE_TARGET_CONTRACTS",
    "STRICT_EVIDENCE_CITATION_GRADE_RECORD_CONTRACT_SCHEMA_ID",
    "STRICT_EVIDENCE_CITATION_GRADE_RECORD_SCHEMA_ID",
    "STRICT_EVIDENCE_CITATION_GRADE_STORE",
    "build_sample_citation_grade_record_from_policy_row",
    "build_strict_evidence_citation_grade_record_contract",
    "render_strict_evidence_citation_grade_record_contract_markdown",
    "validate_citation_grade_record_semantics",
    "write_strict_evidence_citation_grade_record_contract_reports",
]
