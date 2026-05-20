"""Contract-only write target definition for StrictEvidence runtime binding records.

Defines the append-only runtime binding record and store contract after the
citation-grade runtime binding gate design. Report-only: does not write runtime
binding JSONL, mutate citation-grade/eligibility/StrictEvidence/SourceSpan stores,
or enable runtime/answer surfaces.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_citation_grade_runtime_binding_gate_design import (
    RUNTIME_BINDING_GATE_DESIGN_STATUS_CANDIDATE_ONLY,
    RUNTIME_BINDING_POLICY_VERSION,
    RUNTIME_BINDING_STORE,
    STRICT_EVIDENCE_CITATION_GRADE_RUNTIME_BINDING_GATE_DESIGN_SCHEMA_ID,
)


STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-runtime-binding-record-contract.v1"
)
STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-runtime-binding-record.v1"
)

RUNTIME_BINDING_DECISION = "runtime_binding_policy_candidate_only"
RUNTIME_BINDING_STATE = "runtime_binding_candidate_only"

DEFAULT_RUNTIME_BINDING_GATE_DESIGN_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-citation-grade-runtime-binding-gate-design"
    / "01-strict-evidence-citation-grade-runtime-binding-gate-design"
    / "strict-evidence-citation-grade-runtime-binding-gate-design.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-runtime-binding-record-contract"
    / "01-strict-evidence-runtime-binding-record-contract"
)

NO_MUTATION_POLICY = {
    "contractOnly": True,
    "executorImplemented": False,
    "runtimeBindingRecordWrite": False,
    "runtimeEvidenceCreated": False,
    "runtimeVisible": False,
    "answerIntegrationVisible": False,
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
    "runtimeBindingRecordId",
    "runId",
    "plannedWriteTarget",
    "paperId",
    "artifactType",
    "citationGradeRecordId",
    "strictEvidenceId",
    "eligibilityRecordId",
    "sourceSpanId",
    "candidateRecordId",
    "sourceContentHash",
    "policyVersion",
    "runtimeBindingDecision",
    "runtimeBindingState",
    "runtimeBindingGateDesignRowId",
    "runtimeVisible",
    "answerIntegrationVisible",
    "citationGradeRecordInPlaceMutationAllowed",
    "strictEvidenceInPlaceMutationAllowed",
    "eligibilityRecordInPlaceMutationAllowed",
    "sourceSpanInPlaceMutationAllowed",
    "runtimeBindingMutationApplied",
    "runtimeEvidence",
    "idempotencyKey",
    "provenanceTrace",
    "writePolicy",
]

IDEMPOTENCY_KEY_FIELDS = [
    "plannedWriteTarget",
    "paperId",
    "artifactType",
    "citationGradeRecordId",
    "strictEvidenceId",
    "eligibilityRecordId",
    "sourceSpanId",
    "candidateRecordId",
    "sourceContentHash",
    "policyVersion",
    "runtimeBindingDecision",
    "idempotencyKey",
]

RUNTIME_BINDING_STORE_CONTRACT: dict[str, Any] = {
    "plannedWriteTarget": RUNTIME_BINDING_STORE,
    "contractReference": STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID,
    "runtimeBindingRecordSchema": STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID,
    "storeKind": "local_papers_dir_jsonl_strict_evidence_runtime_binding_store",
    "storeRootTemplate": "{papers_dir}/structured_evidence/strict_evidence_runtime_binding",
    "recordPathTemplate": (
        "{papers_dir}/structured_evidence/strict_evidence_runtime_binding/{paper_id}.jsonl"
    ),
    "runManifestPathTemplate": "{papers_dir}/structured_evidence/runs/{run_id}.json",
    "allowedArtifactTypes": ["section", "figure"],
    "requiredRecordFields": REQUIRED_RECORD_FIELDS,
    "idempotencyKeyFields": IDEMPOTENCY_KEY_FIELDS,
    "writeSemantics": "explicit_apply_executor_appends_or_replaces_same_idempotency_key",
    "readbackChecks": [
        "runtime_binding_record_schema_validates",
        "idempotency_key_stable",
        "citationGradeRecordId_resolves_to_existing_citation_grade_jsonl",
        "strictEvidenceId_resolves_to_existing_strict_evidence_jsonl",
        "eligibilityRecordId_resolves_to_existing_eligibility_jsonl",
        "sourceSpanId_resolves_to_existing_source_span_jsonl",
        "sourceContentHash_preserved",
        "parent_records_remain_unmutated",
        "runtime_visible_remains_false_until_runtime_binding_apply",
        "answer_integration_visible_remains_false_until_answer_gate",
    ],
    "rollbackStrategy": (
        "delete or invalidate runtime binding records written by the explicit run_id "
        "while no downstream answer integration references them"
    ),
    "rollbackImplemented": False,
    "executorImplemented": False,
    "citationGradeRecordInPlaceMutationAllowed": False,
    "strictEvidenceInPlaceMutationAllowed": False,
    "eligibilityRecordInPlaceMutationAllowed": False,
    "sourceSpanInPlaceMutationAllowed": False,
    "runtimeUseAllowed": False,
    "parserRoutingAllowed": False,
    "answerIntegrationAllowed": False,
    "databaseMutationAllowed": False,
    "vaultScanAllowed": False,
}

KNOWN_WRITE_TARGET_CONTRACTS: dict[str, str] = {
    RUNTIME_BINDING_STORE: STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID,
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


def _derive_source_content_hash(design_row: dict[str, Any]) -> str:
    explicit = _safe_text(design_row.get("sourceContentHash"))
    if explicit:
        return explicit
    source_span_id = _safe_text(design_row.get("sourceSpanId"))
    if not source_span_id:
        return ""
    digest = sha256(source_span_id.encode("utf-8")).hexdigest()[:32]
    return digest


def validate_runtime_binding_record_semantics(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if _safe_text(record.get("plannedWriteTarget")) != RUNTIME_BINDING_STORE:
        errors.append("plannedWriteTarget_must_be_strict_evidence_runtime_binding_store")
    if _safe_text(record.get("policyVersion")) != RUNTIME_BINDING_POLICY_VERSION:
        errors.append("policyVersion_mismatch")
    if _safe_text(record.get("runtimeBindingDecision")) != RUNTIME_BINDING_DECISION:
        errors.append("runtimeBindingDecision_mismatch")
    if _safe_text(record.get("runtimeBindingState")) != RUNTIME_BINDING_STATE:
        errors.append("runtimeBindingState_mismatch")
    for field_name in (
        "citationGradeRecordInPlaceMutationAllowed",
        "strictEvidenceInPlaceMutationAllowed",
        "eligibilityRecordInPlaceMutationAllowed",
        "sourceSpanInPlaceMutationAllowed",
    ):
        if _safe_bool(record.get(field_name)):
            errors.append(f"{field_name}_must_be_false")
    if _safe_bool(record.get("runtimeBindingMutationApplied")):
        errors.append("runtimeBindingMutationApplied_must_be_false")
    if _safe_bool(record.get("runtimeEvidence")):
        errors.append("runtimeEvidence_must_be_false")
    if _safe_bool(record.get("runtimeVisible")):
        errors.append("runtimeVisible_must_be_false")
    if _safe_bool(record.get("answerIntegrationVisible")):
        errors.append("answerIntegrationVisible_must_be_false")
    for field_name in (
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


def build_sample_runtime_binding_record_from_gate_design_row(
    design_row: dict[str, Any],
    *,
    run_id: str = "runtime-binding-contract-sample-run",
    runtime_binding_record_id: str | None = None,
) -> dict[str, Any]:
    paper_id = _safe_text(design_row.get("paper_id"))
    artifact_type = _safe_text(design_row.get("artifact_type"))
    strict_evidence_id = _safe_text(design_row.get("strictEvidenceId"))
    source_span_id = _safe_text(design_row.get("sourceSpanId"))
    candidate_record_id = _safe_text(design_row.get("candidateRecordId"))
    eligibility_record_id = _safe_text(design_row.get("eligibilityRecordId"))
    citation_grade_record_id = _safe_text(design_row.get("citationGradeRecordId"))
    gate_design_row_id = _safe_text(design_row.get("runtime_binding_gate_design_row_id"))
    source_content_hash = _derive_source_content_hash(design_row)
    record_id = runtime_binding_record_id or f"strict-evidence-runtime-binding:{strict_evidence_id}"
    idempotency_key = (
        f"strict-runtime-binding:{citation_grade_record_id}:{strict_evidence_id}:"
        f"{RUNTIME_BINDING_POLICY_VERSION}:{RUNTIME_BINDING_DECISION}"
    )

    return {
        "schema": STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID,
        "runtimeBindingRecordId": record_id,
        "runId": run_id,
        "plannedWriteTarget": RUNTIME_BINDING_STORE,
        "paperId": paper_id,
        "artifactType": artifact_type,
        "citationGradeRecordId": citation_grade_record_id,
        "strictEvidenceId": strict_evidence_id,
        "eligibilityRecordId": eligibility_record_id,
        "sourceSpanId": source_span_id,
        "candidateRecordId": candidate_record_id,
        "sourceContentHash": source_content_hash,
        "policyVersion": RUNTIME_BINDING_POLICY_VERSION,
        "runtimeBindingDecision": RUNTIME_BINDING_DECISION,
        "runtimeBindingState": RUNTIME_BINDING_STATE,
        "runtimeBindingGateDesignRowId": gate_design_row_id,
        "runtimeVisible": False,
        "answerIntegrationVisible": False,
        "citationGradeRecordInPlaceMutationAllowed": False,
        "strictEvidenceInPlaceMutationAllowed": False,
        "eligibilityRecordInPlaceMutationAllowed": False,
        "sourceSpanInPlaceMutationAllowed": False,
        "runtimeBindingMutationApplied": False,
        "runtimeEvidence": False,
        "idempotencyKey": idempotency_key,
        "provenanceTrace": {
            "citationGradeRecordId": citation_grade_record_id,
            "strictEvidenceId": strict_evidence_id,
            "sourceSpanId": source_span_id,
            "candidateRecordId": candidate_record_id,
            "eligibilityRecordId": eligibility_record_id,
            "sourceContentHash": source_content_hash,
            "runtimeBindingGateDesignRowId": gate_design_row_id,
            "runtimeBindingGateDesignStatus": _safe_text(
                design_row.get("runtime_binding_gate_design_status")
            ),
        },
        "writePolicy": _write_policy(),
    }


def build_strict_evidence_runtime_binding_record_contract(
    *,
    runtime_binding_gate_design_report_path: str | Path = (
        DEFAULT_RUNTIME_BINDING_GATE_DESIGN_REPORT_PATH
    ),
    expected_input_rows: int = 99,
    expected_planned_runtime_binding_rows: int = 99,
) -> dict[str, Any]:
    report_path = Path(str(runtime_binding_gate_design_report_path)).expanduser()
    gate_design = _read_json(report_path)
    warnings: list[str] = []
    schema_violations: list[str] = []

    if gate_design:
        validation = validate_payload(
            gate_design,
            STRICT_EVIDENCE_CITATION_GRADE_RUNTIME_BINDING_GATE_DESIGN_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            schema_violations.extend(str(error) for error in validation.errors)
    else:
        warnings.append("runtime_binding_gate_design_report_missing_or_unreadable")

    counts = gate_design.get("counts") if isinstance(gate_design.get("counts"), dict) else {}
    design_rows = [
        row for row in gate_design.get("rows", []) if isinstance(row, dict)
    ] if gate_design else []
    design = (
        gate_design.get("runtimeBindingGateDesign")
        if isinstance(gate_design.get("runtimeBindingGateDesign"), dict)
        else {}
    )

    input_rows = _safe_int(counts.get("inputRows"))
    candidate_rows = _safe_int(counts.get("runtimeBindingGateDesignCandidateOnlyRows"))
    section_rows = _safe_int(counts.get("sectionRuntimeBindingGateDesignRows"))
    figure_rows = _safe_int(counts.get("figureCaptionRuntimeBindingGateDesignRows"))

    if input_rows != expected_input_rows:
        schema_violations.append(f"inputRows={input_rows}:expected={expected_input_rows}")
    if candidate_rows != expected_planned_runtime_binding_rows:
        schema_violations.append(
            "plannedRuntimeBindingRows="
            f"{candidate_rows}:expected={expected_planned_runtime_binding_rows}"
        )

    status = "ok"
    if (
        schema_violations
        or not gate_design
        or _safe_text(gate_design.get("status")) != "ok"
        or _safe_text(design.get("decision")) != "separate_append_only_runtime_binding_record"
        or not _safe_bool(design.get("runtimeBindingRecordContractRequired"))
        or candidate_rows <= 0
        or _safe_int(counts.get("runtimeBindingRecordWriteRows")) != 0
        or _safe_int(counts.get("runtimeEvidenceCreatedRows")) != 0
        or _safe_int(counts.get("answerIntegrationChangedRows")) != 0
    ):
        status = "blocked"

    sample_record = (
        build_sample_runtime_binding_record_from_gate_design_row(design_rows[0])
        if design_rows
        and _safe_text(design_rows[0].get("runtime_binding_gate_design_status"))
        == RUNTIME_BINDING_GATE_DESIGN_STATUS_CANDIDATE_ONLY
        else {}
    )
    sample_record_semantic_errors = (
        validate_runtime_binding_record_semantics(sample_record) if sample_record else []
    )
    sample_record_schema_ok = False
    if sample_record:
        sample_record_schema_ok = validate_payload(
            sample_record,
            STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID,
            strict=True,
        ).ok

    return {
        "schema": STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "runtimeBindingGateDesignReportPath": str(report_path),
            "runtimeBindingGateDesignSchema": _safe_text(gate_design.get("schema"))
            if gate_design
            else "",
            "runtimeBindingGateDesignStatus": _safe_text(gate_design.get("status"))
            if gate_design
            else "",
            "runtimeBindingGateDesignDecision": _safe_text(design.get("decision")),
            "runtimeBindingGateDesignCandidateOnlyRows": candidate_rows,
            "sectionRuntimeBindingGateDesignRows": section_rows,
            "figureCaptionRuntimeBindingGateDesignRows": figure_rows,
            "expectedInputRows": expected_input_rows,
            "expectedPlannedRuntimeBindingRows": expected_planned_runtime_binding_rows,
        },
        "counts": {
            "inputRows": input_rows,
            "runtimeBindingRecordContracts": 1,
            "runtimeBindingRecordSchemas": 1,
            "plannedRuntimeBindingRows": candidate_rows,
            "runtimeBindingGateDesignCandidateOnlyRows": candidate_rows,
            "sectionRuntimeBindingGateDesignRows": section_rows,
            "figureCaptionRuntimeBindingGateDesignRows": figure_rows,
            "executorImplementedRows": 0,
            "runtimeBindingRecordWriteRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "runtimeVisibleMutationRows": 0,
            "answerIntegrationVisibleMutationRows": 0,
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
            "runtimeBindingStoreContractDefined": True,
            "runtimeBindingRecordSchemaDefined": True,
            "executorReady": False,
            "runtimeMutationAllowed": False,
            "runtimeBindingRecordWriteAllowed": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "citationGradeRecordWriteAllowed": False,
            "parentRecordMutationAllowed": False,
            "decision": (
                "strict_evidence_runtime_binding_record_contract_ready"
                if status == "ok"
                else "strict_evidence_runtime_binding_record_contract_blocked"
            ),
            "schemaViolations": schema_violations,
            "sampleRecordSemanticViolations": sample_record_semantic_errors,
            "recommendedNextTranche": (
                "strict_evidence_runtime_binding_executor_dry_run"
                if status == "ok"
                else "strict_evidence_citation_grade_runtime_binding_gate_design_repair"
            ),
        },
        "policy": dict(NO_MUTATION_POLICY),
        "contractPrinciples": [
            "runtime binding records are append-only promotion metadata",
            "citation-grade, eligibility, StrictEvidence, and SourceSpan rows remain immutable",
            "runtimeVisible and answerIntegrationVisible remain false under this contract",
            "runtime binding is not parser routing, answer integration, or DB/index/reembed",
            "rollback targets runtime binding records by explicit run_id before answer references exist",
        ],
        "writeTargets": [dict(RUNTIME_BINDING_STORE_CONTRACT)],
        "sampleRuntimeBindingRecord": sample_record,
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


def render_strict_evidence_runtime_binding_record_contract_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    write_target = (report.get("writeTargets") or [{}])[0]
    principles = [f"- {item}" for item in list(report.get("contractPrinciples") or [])]
    return "\n".join(
        [
            "# Strict Evidence Runtime Binding Record Contract",
            "",
            f"- status: {report.get('status', '')}",
            f"- decision: {gate.get('decision', '')}",
            f"- planned write target: {write_target.get('plannedWriteTarget', '')}",
            f"- record path template: {write_target.get('recordPathTemplate', '')}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- planned runtime binding rows: {int(counts.get('plannedRuntimeBindingRows') or 0)}",
            f"- section design rows: {int(counts.get('sectionRuntimeBindingGateDesignRows') or 0)}",
            f"- figure caption design rows: {int(counts.get('figureCaptionRuntimeBindingGateDesignRows') or 0)}",
            f"- runtime binding record writes: {int(counts.get('runtimeBindingRecordWriteRows') or 0)}",
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


def write_strict_evidence_runtime_binding_record_contract_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-runtime-binding-record-contract.json"
    summary_path = root / "strict-evidence-runtime-binding-record-contract-summary.json"
    markdown_path = root / "strict-evidence-runtime-binding-record-contract.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_runtime_binding_record_contract_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Define the StrictEvidence runtime binding record/store contract without writing "
            "runtime binding records or mutating evidence stores."
        )
    )
    parser.add_argument(
        "--runtime-binding-gate-design-report",
        default=str(DEFAULT_RUNTIME_BINDING_GATE_DESIGN_REPORT_PATH),
        help="Path to the runtime binding gate design JSON report.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, summary, and markdown reports.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_runtime_binding_record_contract(
        runtime_binding_gate_design_report_path=args.runtime_binding_gate_design_report,
    )
    paths = write_strict_evidence_runtime_binding_record_contract_reports(report, args.output_dir)
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
    "DEFAULT_RUNTIME_BINDING_GATE_DESIGN_REPORT_PATH",
    "KNOWN_WRITE_TARGET_CONTRACTS",
    "RUNTIME_BINDING_DECISION",
    "RUNTIME_BINDING_STATE",
    "RUNTIME_BINDING_STORE",
    "RUNTIME_BINDING_STORE_CONTRACT",
    "STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID",
    "STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID",
    "build_sample_runtime_binding_record_from_gate_design_row",
    "build_strict_evidence_runtime_binding_record_contract",
    "render_strict_evidence_runtime_binding_record_contract_markdown",
    "validate_runtime_binding_record_semantics",
    "write_strict_evidence_runtime_binding_record_contract_reports",
]
