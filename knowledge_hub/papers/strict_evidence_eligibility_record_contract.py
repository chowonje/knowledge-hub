"""Contract-only write target definition for StrictEvidence eligibility records.

Defines the append-only eligibility record and store contract after the
strictEligible mutation decision record. This helper is report-only: it does
not write eligibility JSONL, mutate StrictEvidence/SourceSpan stores, set
strictEligible=true, or enable citation/runtime/answer surfaces.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_strict_eligible_mutation_decision_record import (
    DECISION_SEPARATE_ELIGIBILITY_RECORD,
    STRICT_EVIDENCE_STRICT_ELIGIBLE_MUTATION_DECISION_RECORD_SCHEMA_ID,
)


STRICT_EVIDENCE_ELIGIBILITY_RECORD_CONTRACT_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-eligibility-record-contract.v1"
)
STRICT_EVIDENCE_ELIGIBILITY_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-eligibility-record.v1"
)

STRICT_EVIDENCE_ELIGIBILITY_STORE = "parsed_artifact_strict_evidence_eligibility_store"
ELIGIBILITY_POLICY_VERSION = "strict_evidence_eligibility_policy.v1"
ELIGIBILITY_STATE_CANDIDATE_ONLY = "strict_evidence_eligible_candidate_only"
ELIGIBILITY_DECISION = "eligible_for_citation_grade_gate_candidate_only"

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
    / "strict-evidence-eligibility-record-contract"
    / "01-strict-evidence-eligibility-record-contract"
)

NO_MUTATION_POLICY = {
    "contractOnly": True,
    "executorImplemented": False,
    "eligibilityRecordWrite": False,
    "strictEligibleMutation": False,
    "strictEvidenceStoreWrite": False,
    "sourceSpanStoreWrite": False,
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

REQUIRED_RECORD_FIELDS = [
    "schema",
    "eligibilityRecordId",
    "runId",
    "plannedWriteTarget",
    "paperId",
    "artifactType",
    "strictEvidenceId",
    "sourceSpanId",
    "candidateRecordId",
    "decisionRowId",
    "holdRowId",
    "eligibilityPolicyVersion",
    "eligibilityDecision",
    "eligibilityState",
    "strictEvidenceInPlaceMutationAllowed",
    "strictEligibleBooleanMutationAllowed",
    "strictEligibleMutationApplied",
    "citationGrade",
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
    "sourceSpanId",
    "candidateRecordId",
    "eligibilityPolicyVersion",
    "eligibilityDecision",
    "idempotencyKey",
]

ELIGIBILITY_STORE_CONTRACT: dict[str, Any] = {
    "plannedWriteTarget": STRICT_EVIDENCE_ELIGIBILITY_STORE,
    "contractReference": STRICT_EVIDENCE_ELIGIBILITY_RECORD_CONTRACT_SCHEMA_ID,
    "eligibilityRecordSchema": STRICT_EVIDENCE_ELIGIBILITY_RECORD_SCHEMA_ID,
    "storeKind": "local_papers_dir_jsonl_strict_evidence_eligibility_store",
    "storeRootTemplate": "{papers_dir}/structured_evidence/strict_evidence_eligibility",
    "recordPathTemplate": "{papers_dir}/structured_evidence/strict_evidence_eligibility/{paper_id}.jsonl",
    "runManifestPathTemplate": "{papers_dir}/structured_evidence/runs/{run_id}.json",
    "allowedArtifactTypes": ["section", "figure"],
    "requiredRecordFields": REQUIRED_RECORD_FIELDS,
    "idempotencyKeyFields": IDEMPOTENCY_KEY_FIELDS,
    "writeSemantics": "explicit_apply_executor_appends_or_replaces_same_idempotency_key",
    "readbackChecks": [
        "eligibility_record_schema_validates",
        "idempotency_key_stable",
        "strictEvidenceId_resolves_to_existing_strict_evidence_jsonl",
        "sourceSpanId_resolves_to_existing_source_span_jsonl",
        "candidateRecordId_preserved",
        "strict_evidence_record_remains_unmutated",
        "strictEligible_boolean_remains_false_on_strict_evidence_record",
        "eligibility_record_remains_non_citation_and_non_runtime",
    ],
    "rollbackStrategy": (
        "delete or invalidate eligibility records written by the explicit run_id "
        "while no downstream citation-grade or runtime binding references them"
    ),
    "rollbackImplemented": False,
    "executorImplemented": False,
    "strictEvidenceInPlaceMutationAllowed": False,
    "strictEligibleBooleanMutationAllowed": False,
    "runtimeUseAllowed": False,
    "citationUseAllowed": False,
    "parserRoutingAllowed": False,
    "answerIntegrationAllowed": False,
    "databaseMutationAllowed": False,
    "sourceSpanMutationAllowed": False,
}

KNOWN_WRITE_TARGET_CONTRACTS: dict[str, str] = {
    STRICT_EVIDENCE_ELIGIBILITY_STORE: STRICT_EVIDENCE_ELIGIBILITY_RECORD_CONTRACT_SCHEMA_ID,
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
        "eligibilityRecordWrite": False,
        "strictEvidenceStoreWrite": False,
        "sourceSpanStoreWrite": False,
        "strictEligibleMutation": False,
        "citationGradeEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "databaseMutation": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
    }


def validate_eligibility_record_semantics(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if _safe_text(record.get("plannedWriteTarget")) != STRICT_EVIDENCE_ELIGIBILITY_STORE:
        errors.append("plannedWriteTarget_must_be_strict_evidence_eligibility_store")
    if _safe_text(record.get("eligibilityPolicyVersion")) != ELIGIBILITY_POLICY_VERSION:
        errors.append("eligibilityPolicyVersion_mismatch")
    if _safe_text(record.get("eligibilityDecision")) != ELIGIBILITY_DECISION:
        errors.append("eligibilityDecision_mismatch")
    if _safe_bool(record.get("strictEvidenceInPlaceMutationAllowed")):
        errors.append("strictEvidenceInPlaceMutationAllowed_must_be_false")
    if _safe_bool(record.get("strictEligibleBooleanMutationAllowed")):
        errors.append("strictEligibleBooleanMutationAllowed_must_be_false")
    if _safe_bool(record.get("strictEligibleMutationApplied")):
        errors.append("strictEligibleMutationApplied_must_be_false")
    if _safe_bool(record.get("citationGrade")):
        errors.append("citationGrade_must_be_false")
    if _safe_bool(record.get("runtimeEvidence")):
        errors.append("runtimeEvidence_must_be_false")
    if _safe_bool(record.get("runtimeVisible")):
        errors.append("runtimeVisible_must_be_false")
    for field_name in ("strictEvidenceId", "sourceSpanId", "candidateRecordId"):
        if not _safe_text(record.get(field_name)):
            errors.append(f"{field_name}_must_be_non_empty")
    return errors


def build_sample_eligibility_record_from_decision_row(
    decision_row: dict[str, Any],
    *,
    run_id: str = "eligibility-contract-sample-run",
    eligibility_record_id: str | None = None,
) -> dict[str, Any]:
    paper_id = _safe_text(decision_row.get("paper_id"))
    artifact_type = _safe_text(decision_row.get("artifact_type"))
    strict_evidence_id = _safe_text(decision_row.get("strictEvidenceId"))
    source_span_id = _safe_text(decision_row.get("sourceSpanId"))
    candidate_record_id = _safe_text(decision_row.get("candidateRecordId"))
    decision_row_id = _safe_text(decision_row.get("decision_row_id"))
    hold_row_id = _safe_text(decision_row.get("hold_row_id"))
    record_id = eligibility_record_id or f"strict-evidence-eligibility:{strict_evidence_id}"
    idempotency_key = (
        f"strict-eligibility:{strict_evidence_id}:{ELIGIBILITY_POLICY_VERSION}:{ELIGIBILITY_DECISION}"
    )

    return {
        "schema": STRICT_EVIDENCE_ELIGIBILITY_RECORD_SCHEMA_ID,
        "eligibilityRecordId": record_id,
        "runId": run_id,
        "plannedWriteTarget": STRICT_EVIDENCE_ELIGIBILITY_STORE,
        "paperId": paper_id,
        "artifactType": artifact_type,
        "strictEvidenceId": strict_evidence_id,
        "sourceSpanId": source_span_id,
        "candidateRecordId": candidate_record_id,
        "decisionRowId": decision_row_id,
        "holdRowId": hold_row_id,
        "eligibilityPolicyVersion": ELIGIBILITY_POLICY_VERSION,
        "eligibilityDecision": ELIGIBILITY_DECISION,
        "eligibilityState": ELIGIBILITY_STATE_CANDIDATE_ONLY,
        "strictEvidenceInPlaceMutationAllowed": False,
        "strictEligibleBooleanMutationAllowed": False,
        "strictEligibleMutationApplied": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "runtimeVisible": False,
        "idempotencyKey": idempotency_key,
        "provenanceTrace": {
            "strictEvidenceId": strict_evidence_id,
            "sourceSpanId": source_span_id,
            "candidateRecordId": candidate_record_id,
            "decisionRowId": decision_row_id,
            "holdRowId": hold_row_id,
            "decision": _safe_text(decision_row.get("strictEligibleMutationDecision")),
        },
        "writePolicy": _write_policy(),
    }


def build_strict_evidence_eligibility_record_contract(
    *,
    decision_record_report_path: str | Path = DEFAULT_DECISION_RECORD_REPORT_PATH,
) -> dict[str, Any]:
    report_path = Path(str(decision_record_report_path)).expanduser()
    decision_record = _read_json(report_path)
    warnings: list[str] = []
    schema_violations: list[str] = []

    if decision_record:
        validation = validate_payload(
            decision_record,
            STRICT_EVIDENCE_STRICT_ELIGIBLE_MUTATION_DECISION_RECORD_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            schema_violations.extend(str(error) for error in validation.errors)
    else:
        warnings.append("decision_record_report_missing_or_unreadable")

    counts = decision_record.get("counts") if isinstance(decision_record.get("counts"), dict) else {}
    decision_rows = [
        row for row in decision_record.get("rows", []) if isinstance(row, dict)
    ] if decision_record else []
    decision = (
        decision_record.get("strictEligibleSemanticsDecision")
        if isinstance(decision_record.get("strictEligibleSemanticsDecision"), dict)
        else {}
    )
    status = "ok"
    if (
        schema_violations
        or not decision_record
        or _safe_text(decision.get("decision")) != DECISION_SEPARATE_ELIGIBILITY_RECORD
        or _safe_int(counts.get("decisionRecordCandidateOnlyRows")) <= 0
        or _safe_int(counts.get("strictEligibleMutationAllowedRows")) != 0
    ):
        status = "blocked"

    candidate_rows = _safe_int(counts.get("decisionRecordCandidateOnlyRows"))
    section_rows = _safe_int(counts.get("sectionDecisionRows"))
    figure_rows = _safe_int(counts.get("figureCaptionDecisionRows"))
    sample_record = (
        build_sample_eligibility_record_from_decision_row(decision_rows[0])
        if decision_rows
        else {}
    )
    sample_record_semantic_errors = (
        validate_eligibility_record_semantics(sample_record) if sample_record else []
    )
    sample_record_schema_ok = False
    if sample_record:
        sample_record_schema_ok = validate_payload(
            sample_record,
            STRICT_EVIDENCE_ELIGIBILITY_RECORD_SCHEMA_ID,
            strict=True,
        ).ok

    return {
        "schema": STRICT_EVIDENCE_ELIGIBILITY_RECORD_CONTRACT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "decisionRecordReportPath": str(report_path),
            "decisionRecordSchema": _safe_text(decision_record.get("schema")) if decision_record else "",
            "decisionRecordStatus": _safe_text(decision_record.get("status")) if decision_record else "",
            "decision": _safe_text(decision.get("decision")),
            "decisionRecordCandidateOnlyRows": candidate_rows,
            "sectionDecisionRows": section_rows,
            "figureCaptionDecisionRows": figure_rows,
        },
        "counts": {
            "writeTargetContracts": 1,
            "eligibilityStoreContracts": 1,
            "eligibilityRecordSchemas": 1,
            "decisionRecordCandidateOnlyRows": candidate_rows,
            "sectionDecisionRows": section_rows,
            "figureCaptionDecisionRows": figure_rows,
            "executorImplementedRows": 0,
            "eligibilityRecordWriteRows": 0,
            "strictEligibleMutationRows": 0,
            "strictEvidenceWriteRows": 0,
            "sourceSpanUpdatedRows": 0,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "reindexOrReembedRows": 0,
            "schemaViolationCount": len(schema_violations),
            "sampleRecordSchemaValidRows": 1 if sample_record_schema_ok else 0,
            "sampleRecordSemanticViolationRows": len(sample_record_semantic_errors),
        },
        "gate": {
            "writeTargetContractsDefined": True,
            "eligibilityStoreContractDefined": True,
            "eligibilityRecordSchemaDefined": True,
            "executorReady": False,
            "runtimeMutationAllowed": False,
            "strictEligibleMutationAllowed": False,
            "eligibilityRecordWriteAllowed": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "decision": (
                "strict_evidence_eligibility_record_contract_ready"
                if status == "ok"
                else "strict_evidence_eligibility_record_contract_blocked"
            ),
            "schemaViolations": schema_violations,
            "sampleRecordSemanticViolations": sample_record_semantic_errors,
            "recommendedNextTranche": (
                "strict_evidence_eligibility_executor_dry_run"
                if status == "ok"
                else "strict_evidence_strict_eligible_mutation_decision_record_repair"
            ),
        },
        "policy": dict(NO_MUTATION_POLICY),
        "contractPrinciples": [
            "eligibility records are append-only promotion metadata",
            "StrictEvidence and SourceSpan rows remain immutable under this contract",
            "strictEligible booleans remain false until a later explicit cleanup or schema revision",
            "eligibility is not citation-grade, runtime evidence, parser routing, or answer integration",
            "rollback targets eligibility records by explicit run_id before downstream references exist",
        ],
        "writeTargets": [dict(ELIGIBILITY_STORE_CONTRACT)],
        "sampleEligibilityRecord": sample_record,
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


def render_strict_evidence_eligibility_record_contract_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    write_target = (report.get("writeTargets") or [{}])[0]
    principles = [f"- {item}" for item in list(report.get("contractPrinciples") or [])]
    return "\n".join(
        [
            "# Strict Evidence Eligibility Record Contract",
            "",
            f"- status: {report.get('status', '')}",
            f"- decision: {gate.get('decision', '')}",
            f"- planned write target: {write_target.get('plannedWriteTarget', '')}",
            f"- record path template: {write_target.get('recordPathTemplate', '')}",
            f"- decision candidate rows: {int(counts.get('decisionRecordCandidateOnlyRows') or 0)}",
            f"- section decision rows: {int(counts.get('sectionDecisionRows') or 0)}",
            f"- figure caption decision rows: {int(counts.get('figureCaptionDecisionRows') or 0)}",
            f"- eligibility record writes: {int(counts.get('eligibilityRecordWriteRows') or 0)}",
            f"- strictEligible mutation rows: {int(counts.get('strictEligibleMutationRows') or 0)}",
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


def write_strict_evidence_eligibility_record_contract_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-eligibility-record-contract.json"
    summary_path = root / "strict-evidence-eligibility-record-contract-summary.json"
    markdown_path = root / "strict-evidence-eligibility-record-contract.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_eligibility_record_contract_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Define the StrictEvidence eligibility record/store contract without writing "
            "eligibility records or mutating evidence stores."
        )
    )
    parser.add_argument(
        "--decision-record-report",
        default=str(DEFAULT_DECISION_RECORD_REPORT_PATH),
        help="Path to the strictEligible mutation decision-record JSON report.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, summary, and markdown reports.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_eligibility_record_contract(
        decision_record_report_path=args.decision_record_report,
    )
    paths = write_strict_evidence_eligibility_record_contract_reports(report, args.output_dir)
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
    "DEFAULT_OUTPUT_DIR",
    "ELIGIBILITY_DECISION",
    "ELIGIBILITY_POLICY_VERSION",
    "ELIGIBILITY_STATE_CANDIDATE_ONLY",
    "ELIGIBILITY_STORE_CONTRACT",
    "KNOWN_WRITE_TARGET_CONTRACTS",
    "STRICT_EVIDENCE_ELIGIBILITY_RECORD_CONTRACT_SCHEMA_ID",
    "STRICT_EVIDENCE_ELIGIBILITY_RECORD_SCHEMA_ID",
    "STRICT_EVIDENCE_ELIGIBILITY_STORE",
    "build_sample_eligibility_record_from_decision_row",
    "build_strict_evidence_eligibility_record_contract",
    "render_strict_evidence_eligibility_record_contract_markdown",
    "validate_eligibility_record_semantics",
    "write_strict_evidence_eligibility_record_contract_reports",
]
