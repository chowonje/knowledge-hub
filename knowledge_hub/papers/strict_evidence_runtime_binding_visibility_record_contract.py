"""Contract-only store definition for runtime binding visibility records.

Consumes the runtime binding visibility decision record and defines the future
append-only runtime visibility store. Report-only: this helper does not write
runtime visibility JSONL, mutate runtime binding/citation/eligibility/StrictEvidence/
SourceSpan stores, create runtime evidence, or enable parser/answer surfaces.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_runtime_binding_visibility_decision_record import (
    DECISION_SEPARATE_RUNTIME_VISIBILITY_RECORD,
    DECISION_STATUS_CANDIDATE_ONLY,
    DEFAULT_OUTPUT_DIR as DEFAULT_VISIBILITY_DECISION_OUTPUT_DIR,
    STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_DECISION_RECORD_SCHEMA_ID,
)


STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_CONTRACT_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-runtime-binding-visibility-record-contract.v1"
)
STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-runtime-binding-visibility-record.v1"
)

RUNTIME_VISIBILITY_STORE = "parsed_artifact_strict_evidence_runtime_visibility_store"
RUNTIME_VISIBILITY_POLICY_VERSION = "strict_evidence_runtime_visibility_policy.v1"
RUNTIME_VISIBILITY_DECISION = "runtime_visibility_policy_candidate_only"
RUNTIME_VISIBILITY_STATE = "runtime_visibility_candidate_only"

DEFAULT_VISIBILITY_DECISION_REPORT_PATH = (
    DEFAULT_VISIBILITY_DECISION_OUTPUT_DIR
    / "strict-evidence-runtime-binding-visibility-decision-record.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-runtime-binding-visibility-record-contract"
    / "01-strict-evidence-runtime-binding-visibility-record-contract"
)

NO_MUTATION_POLICY = {
    "contractOnly": True,
    "executorImplemented": False,
    "runtimeVisibilityRecordWrite": False,
    "runtimeBindingRecordWrite": False,
    "runtimeVisible": False,
    "answerIntegrationVisible": False,
    "runtimeEvidenceCreated": False,
    "citationGradeRecordWrite": False,
    "citationGradeBooleanMutation": False,
    "eligibilityRecordWrite": False,
    "strictEligibleMutation": False,
    "strictEvidenceStoreWrite": False,
    "sourceSpanStoreWrite": False,
    "strictEvidenceCreated": False,
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
    "runtimeVisibilityRecordId",
    "runId",
    "plannedWriteTarget",
    "paperId",
    "artifactType",
    "runtimeBindingRecordId",
    "citationGradeRecordId",
    "strictEvidenceId",
    "eligibilityRecordId",
    "sourceSpanId",
    "candidateRecordId",
    "sourceContentHash",
    "policyVersion",
    "runtimeVisibilityDecision",
    "runtimeVisibilityState",
    "runtimeBindingVisibilityDecisionRowId",
    "runtimeVisible",
    "answerIntegrationVisible",
    "runtimeBindingRecordInPlaceMutationAllowed",
    "citationGradeRecordInPlaceMutationAllowed",
    "strictEvidenceInPlaceMutationAllowed",
    "eligibilityRecordInPlaceMutationAllowed",
    "sourceSpanInPlaceMutationAllowed",
    "runtimeVisibilityMutationApplied",
    "runtimeEvidence",
    "idempotencyKey",
    "provenanceTrace",
    "writePolicy",
]

IDEMPOTENCY_KEY_FIELDS = [
    "plannedWriteTarget",
    "paperId",
    "artifactType",
    "runtimeBindingRecordId",
    "citationGradeRecordId",
    "strictEvidenceId",
    "eligibilityRecordId",
    "sourceSpanId",
    "candidateRecordId",
    "sourceContentHash",
    "policyVersion",
    "runtimeVisibilityDecision",
    "idempotencyKey",
]

RUNTIME_VISIBILITY_STORE_CONTRACT: dict[str, Any] = {
    "plannedWriteTarget": RUNTIME_VISIBILITY_STORE,
    "contractReference": STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_CONTRACT_SCHEMA_ID,
    "runtimeVisibilityRecordSchema": STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_SCHEMA_ID,
    "storeKind": "local_papers_dir_jsonl_strict_evidence_runtime_visibility_store",
    "storeRootTemplate": "{papers_dir}/structured_evidence/strict_evidence_runtime_visibility",
    "recordPathTemplate": (
        "{papers_dir}/structured_evidence/strict_evidence_runtime_visibility/{paper_id}.jsonl"
    ),
    "runManifestPathTemplate": "{papers_dir}/structured_evidence/runs/{run_id}.json",
    "allowedArtifactTypes": ["section", "figure"],
    "requiredRecordFields": REQUIRED_RECORD_FIELDS,
    "idempotencyKeyFields": IDEMPOTENCY_KEY_FIELDS,
    "writeSemantics": "explicit_apply_executor_appends_or_replaces_same_idempotency_key",
    "readbackChecks": [
        "runtime_visibility_record_schema_validates",
        "idempotency_key_stable",
        "runtimeBindingRecordId_resolves_to_existing_runtime_binding_jsonl",
        "citationGradeRecordId_resolves_to_existing_citation_grade_jsonl",
        "strictEvidenceId_resolves_to_existing_strict_evidence_jsonl",
        "eligibilityRecordId_resolves_to_existing_eligibility_jsonl",
        "sourceSpanId_resolves_to_existing_source_span_jsonl",
        "sourceContentHash_preserved",
        "parent_records_remain_unmutated",
        "runtime_visible_remains_false_until_runtime_visibility_apply",
        "answer_integration_visible_remains_false_until_answer_gate",
    ],
    "rollbackStrategy": (
        "delete or invalidate runtime visibility records written by explicit run_id "
        "while no downstream answer integration references them"
    ),
    "rollbackImplemented": False,
    "executorImplemented": False,
    "runtimeBindingRecordInPlaceMutationAllowed": False,
    "citationGradeRecordInPlaceMutationAllowed": False,
    "strictEvidenceInPlaceMutationAllowed": False,
    "eligibilityRecordInPlaceMutationAllowed": False,
    "sourceSpanInPlaceMutationAllowed": False,
    "runtimeUseAllowed": False,
    "parserRoutingAllowed": False,
    "answerIntegrationAllowed": False,
    "databaseMutationAllowed": False,
    "vaultScanAllowed": False,
    "manifestWriteAllowed": False,
}

KNOWN_WRITE_TARGET_CONTRACTS: dict[str, str] = {
    RUNTIME_VISIBILITY_STORE: STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_CONTRACT_SCHEMA_ID,
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
        "runtimeVisibilityRecordWrite": False,
        "runtimeBindingRecordWrite": False,
        "runtimeEvidenceCreated": False,
        "runtimeVisible": False,
        "answerIntegrationVisible": False,
        "citationGradeRecordWrite": False,
        "citationGradeBooleanMutation": False,
        "eligibilityRecordWrite": False,
        "strictEvidenceStoreWrite": False,
        "sourceSpanStoreWrite": False,
        "strictEligibleMutation": False,
        "databaseMutation": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
        "manifestWrite": False,
        "vaultScan": False,
    }


def _derive_source_content_hash(decision_row: dict[str, Any]) -> str:
    explicit = _safe_text(decision_row.get("sourceContentHash"))
    if explicit:
        return explicit
    source_span_id = _safe_text(decision_row.get("sourceSpanId"))
    if not source_span_id:
        return ""
    digest = sha256(source_span_id.encode("utf-8")).hexdigest()[:32]
    return digest


def validate_runtime_visibility_record_semantics(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if _safe_text(record.get("plannedWriteTarget")) != RUNTIME_VISIBILITY_STORE:
        errors.append("plannedWriteTarget_must_be_strict_evidence_runtime_visibility_store")
    if _safe_text(record.get("policyVersion")) != RUNTIME_VISIBILITY_POLICY_VERSION:
        errors.append("policyVersion_mismatch")
    if _safe_text(record.get("runtimeVisibilityDecision")) != RUNTIME_VISIBILITY_DECISION:
        errors.append("runtimeVisibilityDecision_mismatch")
    if _safe_text(record.get("runtimeVisibilityState")) != RUNTIME_VISIBILITY_STATE:
        errors.append("runtimeVisibilityState_mismatch")
    for field_name in (
        "runtimeBindingRecordInPlaceMutationAllowed",
        "citationGradeRecordInPlaceMutationAllowed",
        "strictEvidenceInPlaceMutationAllowed",
        "eligibilityRecordInPlaceMutationAllowed",
        "sourceSpanInPlaceMutationAllowed",
    ):
        if _safe_bool(record.get(field_name)):
            errors.append(f"{field_name}_must_be_false")
    if _safe_bool(record.get("runtimeVisibilityMutationApplied")):
        errors.append("runtimeVisibilityMutationApplied_must_be_false")
    if _safe_bool(record.get("runtimeEvidence")):
        errors.append("runtimeEvidence_must_be_false")
    if _safe_bool(record.get("runtimeVisible")):
        errors.append("runtimeVisible_must_be_false")
    if _safe_bool(record.get("answerIntegrationVisible")):
        errors.append("answerIntegrationVisible_must_be_false")
    for field_name in (
        "runtimeBindingRecordId",
        "citationGradeRecordId",
        "strictEvidenceId",
        "eligibilityRecordId",
        "sourceSpanId",
        "candidateRecordId",
        "sourceContentHash",
    ):
        if not _safe_text(record.get(field_name)):
            errors.append(f"{field_name}_must_be_non_empty")
    return errors


def build_sample_runtime_visibility_record_from_decision_row(
    decision_row: dict[str, Any],
    *,
    run_id: str = "runtime-visibility-contract-sample-run",
    runtime_visibility_record_id: str | None = None,
) -> dict[str, Any]:
    paper_id = _safe_text(decision_row.get("paper_id") or decision_row.get("paperId"))
    artifact_type = _safe_text(decision_row.get("artifact_type") or decision_row.get("artifactType"))
    runtime_binding_record_id = _safe_text(decision_row.get("runtimeBindingRecordId"))
    citation_grade_record_id = _safe_text(decision_row.get("citationGradeRecordId"))
    strict_evidence_id = _safe_text(decision_row.get("strictEvidenceId"))
    eligibility_record_id = _safe_text(decision_row.get("eligibilityRecordId"))
    source_span_id = _safe_text(decision_row.get("sourceSpanId"))
    candidate_record_id = _safe_text(decision_row.get("candidateRecordId"))
    source_content_hash = _derive_source_content_hash(decision_row)
    decision_row_id = _safe_text(decision_row.get("visibility_decision_record_row_id"))
    record_id = (
        runtime_visibility_record_id
        or f"strict-evidence-runtime-visibility:{runtime_binding_record_id}"
    )
    idempotency_key = (
        f"strict-runtime-visibility:{runtime_binding_record_id}:{citation_grade_record_id}:"
        f"{RUNTIME_VISIBILITY_POLICY_VERSION}:{RUNTIME_VISIBILITY_DECISION}"
    )

    return {
        "schema": STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_SCHEMA_ID,
        "runtimeVisibilityRecordId": record_id,
        "runId": run_id,
        "plannedWriteTarget": RUNTIME_VISIBILITY_STORE,
        "paperId": paper_id,
        "artifactType": artifact_type,
        "runtimeBindingRecordId": runtime_binding_record_id,
        "citationGradeRecordId": citation_grade_record_id,
        "strictEvidenceId": strict_evidence_id,
        "eligibilityRecordId": eligibility_record_id,
        "sourceSpanId": source_span_id,
        "candidateRecordId": candidate_record_id,
        "sourceContentHash": source_content_hash,
        "policyVersion": RUNTIME_VISIBILITY_POLICY_VERSION,
        "runtimeVisibilityDecision": RUNTIME_VISIBILITY_DECISION,
        "runtimeVisibilityState": RUNTIME_VISIBILITY_STATE,
        "runtimeBindingVisibilityDecisionRowId": decision_row_id,
        "runtimeVisible": False,
        "answerIntegrationVisible": False,
        "runtimeBindingRecordInPlaceMutationAllowed": False,
        "citationGradeRecordInPlaceMutationAllowed": False,
        "strictEvidenceInPlaceMutationAllowed": False,
        "eligibilityRecordInPlaceMutationAllowed": False,
        "sourceSpanInPlaceMutationAllowed": False,
        "runtimeVisibilityMutationApplied": False,
        "runtimeEvidence": False,
        "idempotencyKey": idempotency_key,
        "provenanceTrace": {
            "runtimeBindingRecordId": runtime_binding_record_id,
            "citationGradeRecordId": citation_grade_record_id,
            "strictEvidenceId": strict_evidence_id,
            "eligibilityRecordId": eligibility_record_id,
            "sourceSpanId": source_span_id,
            "candidateRecordId": candidate_record_id,
            "sourceContentHash": source_content_hash,
            "runtimeBindingVisibilityDecisionRowId": decision_row_id,
            "runtimeBindingVisibilityDecisionStatus": _safe_text(
                decision_row.get("decision_status")
            ),
        },
        "writePolicy": _write_policy(),
    }


def build_strict_evidence_runtime_binding_visibility_record_contract(
    *,
    visibility_decision_report_path: str | Path = DEFAULT_VISIBILITY_DECISION_REPORT_PATH,
    expected_input_rows: int = 99,
    expected_planned_runtime_visibility_rows: int = 99,
) -> dict[str, Any]:
    report_path = Path(str(visibility_decision_report_path)).expanduser()
    decision_report = _read_json(report_path)
    warnings: list[str] = []
    schema_violations: list[str] = []

    if decision_report:
        validation = validate_payload(
            decision_report,
            STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_DECISION_RECORD_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            schema_violations.extend(str(error) for error in validation.errors)
    else:
        warnings.append("runtime_binding_visibility_decision_report_missing_or_unreadable")

    counts = (
        decision_report.get("counts")
        if isinstance(decision_report.get("counts"), dict)
        else {}
    )
    decision = (
        decision_report.get("decision")
        if isinstance(decision_report.get("decision"), dict)
        else {}
    )
    decision_rows = [
        row
        for row in decision_report.get("rows", [])
        if isinstance(row, dict)
        and _safe_text(row.get("decision_status")) == DECISION_STATUS_CANDIDATE_ONLY
    ] if decision_report else []

    input_rows = _safe_int(counts.get("inputRows"))
    candidate_rows = _safe_int(counts.get("visibilityDecisionCandidateOnlyRows"))
    section_rows = _safe_int(counts.get("sectionDecisionRows"))
    figure_rows = _safe_int(counts.get("figureCaptionDecisionRows"))

    if input_rows != expected_input_rows:
        schema_violations.append(f"inputRows={input_rows}:expected={expected_input_rows}")
    if candidate_rows != expected_planned_runtime_visibility_rows:
        schema_violations.append(
            "plannedRuntimeVisibilityRows="
            f"{candidate_rows}:expected={expected_planned_runtime_visibility_rows}"
        )

    status = "ok"
    if (
        schema_violations
        or not decision_report
        or _safe_text(decision_report.get("status")) != "ok"
        or _safe_text(decision.get("decision")) != DECISION_SEPARATE_RUNTIME_VISIBILITY_RECORD
        or not _safe_bool(decision.get("runtimeVisibilityRecordContractRequired"))
        or candidate_rows <= 0
        or _safe_int(counts.get("runtimeVisibilityRecordWriteRows")) != 0
        or _safe_int(counts.get("runtimeVisibleRows")) != 0
        or _safe_int(counts.get("answerIntegrationVisibleRows")) != 0
        or _safe_int(counts.get("runtimeEvidenceCreatedRows")) != 0
        or _safe_int(counts.get("parserRoutingChangedRows")) != 0
        or _safe_int(counts.get("answerIntegrationChangedRows")) != 0
        or _safe_int(counts.get("databaseMutationRows")) != 0
        or _safe_int(counts.get("vaultScanAllowedRows")) != 0
    ):
        status = "blocked"

    sample_record = (
        build_sample_runtime_visibility_record_from_decision_row(decision_rows[0])
        if decision_rows
        else {}
    )
    sample_record_semantic_errors = (
        validate_runtime_visibility_record_semantics(sample_record) if sample_record else []
    )
    sample_record_schema_ok = False
    if sample_record:
        sample_record_schema_ok = validate_payload(
            sample_record,
            STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_SCHEMA_ID,
            strict=True,
        ).ok

    return {
        "schema": STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_CONTRACT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "runtimeBindingVisibilityDecisionReportPath": str(report_path),
            "runtimeBindingVisibilityDecisionSchema": _safe_text(decision_report.get("schema"))
            if decision_report
            else "",
            "runtimeBindingVisibilityDecisionStatus": _safe_text(decision_report.get("status"))
            if decision_report
            else "",
            "runtimeBindingVisibilityDecisionDecision": _safe_text(decision.get("decision")),
            "visibilityDecisionCandidateOnlyRows": candidate_rows,
            "sectionVisibilityDecisionRows": section_rows,
            "figureCaptionVisibilityDecisionRows": figure_rows,
            "expectedInputRows": expected_input_rows,
            "expectedPlannedRuntimeVisibilityRows": expected_planned_runtime_visibility_rows,
        },
        "counts": {
            "inputRows": input_rows,
            "runtimeVisibilityRecordContracts": 1,
            "runtimeVisibilityRecordSchemas": 1,
            "plannedRuntimeVisibilityRows": candidate_rows,
            "visibilityDecisionCandidateOnlyRows": candidate_rows,
            "sectionVisibilityDecisionRows": section_rows,
            "figureCaptionVisibilityDecisionRows": figure_rows,
            "executorImplementedRows": 0,
            "runtimeVisibilityRecordWriteRows": 0,
            "runtimeBindingRecordWriteRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "runtimeVisibleMutationRows": 0,
            "answerIntegrationVisibleMutationRows": 0,
            "runtimeVisibleRows": 0,
            "answerIntegrationVisibleRows": 0,
            "citationGradeRecordWriteRows": 0,
            "citationGradeBooleanMutationRows": 0,
            "eligibilityRecordWriteRows": 0,
            "strictEligibleMutationRows": 0,
            "strictEvidenceWriteRows": 0,
            "sourceSpanUpdatedRows": 0,
            "strictEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "manifestWriteRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "schemaViolationCount": len(schema_violations),
            "sampleRecordSchemaValidRows": 1 if sample_record_schema_ok else 0,
            "sampleRecordSemanticViolationRows": len(sample_record_semantic_errors),
        },
        "gate": {
            "writeTargetContractsDefined": True,
            "runtimeVisibilityStoreContractDefined": True,
            "runtimeVisibilityRecordSchemaDefined": True,
            "executorReady": False,
            "runtimeVisibilityRecordWriteAllowed": False,
            "runtimeBindingRecordWriteAllowed": False,
            "runtimeMutationAllowed": False,
            "runtimeVisibleMutationAllowed": False,
            "answerIntegrationVisibleAllowed": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "citationGradeRecordWriteAllowed": False,
            "parentRecordMutationAllowed": False,
            "runManifestWriteAllowed": False,
            "decision": (
                "strict_evidence_runtime_binding_visibility_record_contract_ready"
                if status == "ok"
                else "strict_evidence_runtime_binding_visibility_record_contract_blocked"
            ),
            "schemaViolations": schema_violations,
            "sampleRecordSemanticViolations": sample_record_semantic_errors,
            "recommendedNextTranche": (
                "strict_evidence_runtime_binding_visibility_executor_dry_run"
                if status == "ok"
                else "strict_evidence_runtime_binding_visibility_decision_record_input_repair"
            ),
        },
        "policy": dict(NO_MUTATION_POLICY),
        "contractPrinciples": [
            "runtime visibility records are append-only promotion metadata",
            "runtime binding records remain immutable audit records",
            "citation-grade, eligibility, StrictEvidence, and SourceSpan rows remain immutable",
            "runtimeVisible and answerIntegrationVisible remain false under this contract",
            "runtime visibility does not authorize answer integration",
            "runtime visibility is not parser routing, answer integration, or DB/index/reembed",
            "rollback targets runtime visibility records by explicit run_id before answer references exist",
        ],
        "writeTargets": [dict(RUNTIME_VISIBILITY_STORE_CONTRACT)],
        "sampleRuntimeVisibilityRecord": sample_record,
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


def render_strict_evidence_runtime_binding_visibility_record_contract_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    write_target = (report.get("writeTargets") or [{}])[0]
    principles = [f"- {item}" for item in list(report.get("contractPrinciples") or [])]
    return "\n".join(
        [
            "# Strict Evidence Runtime Binding Visibility Record Contract",
            "",
            f"- status: {report.get('status', '')}",
            f"- decision: {gate.get('decision', '')}",
            f"- planned write target: {write_target.get('plannedWriteTarget', '')}",
            f"- record path template: {write_target.get('recordPathTemplate', '')}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- planned runtime visibility rows: {int(counts.get('plannedRuntimeVisibilityRows') or 0)}",
            f"- section decision rows: {int(counts.get('sectionVisibilityDecisionRows') or 0)}",
            f"- figure caption decision rows: {int(counts.get('figureCaptionVisibilityDecisionRows') or 0)}",
            f"- runtime visibility record writes: {int(counts.get('runtimeVisibilityRecordWriteRows') or 0)}",
            f"- runtime visible rows: {int(counts.get('runtimeVisibleRows') or 0)}",
            f"- answer integration visible rows: {int(counts.get('answerIntegrationVisibleRows') or 0)}",
            f"- runtime evidence created rows: {int(counts.get('runtimeEvidenceCreatedRows') or 0)}",
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


def write_strict_evidence_runtime_binding_visibility_record_contract_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-runtime-binding-visibility-record-contract.json"
    summary_path = root / "strict-evidence-runtime-binding-visibility-record-contract-summary.json"
    markdown_path = root / "strict-evidence-runtime-binding-visibility-record-contract.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_runtime_binding_visibility_record_contract_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Define the StrictEvidence runtime binding visibility record/store contract without "
            "writing visibility records or mutating evidence stores."
        )
    )
    parser.add_argument(
        "--visibility-decision-report",
        default=str(DEFAULT_VISIBILITY_DECISION_REPORT_PATH),
        help="Path to the runtime binding visibility decision JSON report.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, summary, and markdown reports.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_runtime_binding_visibility_record_contract(
        visibility_decision_report_path=args.visibility_decision_report,
    )
    paths = write_strict_evidence_runtime_binding_visibility_record_contract_reports(
        report,
        args.output_dir,
    )
    print(f"wrote report: {paths['report']}")
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote markdown: {paths['markdown']}")
    if args.json:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_VISIBILITY_DECISION_REPORT_PATH",
    "KNOWN_WRITE_TARGET_CONTRACTS",
    "REQUIRED_RECORD_FIELDS",
    "RUNTIME_VISIBILITY_DECISION",
    "RUNTIME_VISIBILITY_POLICY_VERSION",
    "RUNTIME_VISIBILITY_STATE",
    "RUNTIME_VISIBILITY_STORE",
    "RUNTIME_VISIBILITY_STORE_CONTRACT",
    "STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_CONTRACT_SCHEMA_ID",
    "STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_SCHEMA_ID",
    "build_sample_runtime_visibility_record_from_decision_row",
    "build_strict_evidence_runtime_binding_visibility_record_contract",
    "render_strict_evidence_runtime_binding_visibility_record_contract_markdown",
    "validate_runtime_visibility_record_semantics",
    "write_strict_evidence_runtime_binding_visibility_record_contract_reports",
]
