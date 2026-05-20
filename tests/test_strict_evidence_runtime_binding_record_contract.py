from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_citation_grade_runtime_binding_gate_design import (
    RUNTIME_BINDING_STORE,
    build_strict_evidence_citation_grade_runtime_binding_gate_design,
    write_strict_evidence_citation_grade_runtime_binding_gate_design_reports,
)
from knowledge_hub.papers.strict_evidence_runtime_binding_record_contract import (
    KNOWN_WRITE_TARGET_CONTRACTS,
    RUNTIME_BINDING_DECISION,
    RUNTIME_BINDING_STORE_CONTRACT,
    STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID,
    STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID,
    build_sample_runtime_binding_record_from_gate_design_row,
    build_strict_evidence_runtime_binding_record_contract,
    validate_runtime_binding_record_semantics,
    write_strict_evidence_runtime_binding_record_contract_reports,
)
from tests.test_strict_evidence_citation_grade_runtime_binding_gate_design import (
    _hold_report_path,
)


def _runtime_binding_gate_design_report_path(tmp_path: Path) -> Path:
    hold_path = _hold_report_path(tmp_path)
    report = build_strict_evidence_citation_grade_runtime_binding_gate_design(
        hold_review_report_path=hold_path,
        expected_input_rows=1,
        expected_runtime_binding_gate_design_rows=1,
        expected_section_rows=1,
        expected_figure_caption_rows=0,
        expected_citation_grade_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=1,
        expected_source_span_store_rows=1,
    )
    output_dir = tmp_path / "runtime-binding-gate-design"
    write_strict_evidence_citation_grade_runtime_binding_gate_design_reports(
        report,
        output_dir,
    )
    return output_dir / "strict-evidence-citation-grade-runtime-binding-gate-design.json"


def test_runtime_binding_record_contract_declares_separate_store_target(tmp_path: Path) -> None:
    gate_design_path = _runtime_binding_gate_design_report_path(tmp_path)

    payload = build_strict_evidence_runtime_binding_record_contract(
        runtime_binding_gate_design_report_path=gate_design_path,
        expected_input_rows=1,
        expected_planned_runtime_binding_rows=1,
    )

    assert payload["schema"] == STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID
    assert payload["status"] == "ok"
    assert payload["counts"]["runtimeBindingRecordContracts"] == 1
    assert payload["counts"]["runtimeBindingRecordSchemas"] == 1
    assert payload["counts"]["plannedRuntimeBindingRows"] == 1
    assert payload["counts"]["executorImplementedRows"] == 0
    assert payload["counts"]["runtimeBindingRecordWriteRows"] == 0
    assert payload["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert payload["gate"]["executorReady"] is False
    assert payload["gate"]["recommendedNextTranche"] == "strict_evidence_runtime_binding_executor_dry_run"

    write_target = payload["writeTargets"][0]
    assert write_target["plannedWriteTarget"] == RUNTIME_BINDING_STORE
    assert write_target["recordPathTemplate"] == (
        "{papers_dir}/structured_evidence/strict_evidence_runtime_binding/{paper_id}.jsonl"
    )
    assert write_target["citationGradeRecordInPlaceMutationAllowed"] is False
    assert write_target["vaultScanAllowed"] is False
    assert "citationGradeRecordId_resolves_to_existing_citation_grade_jsonl" in write_target[
        "readbackChecks"
    ]
    assert KNOWN_WRITE_TARGET_CONTRACTS[RUNTIME_BINDING_STORE] == (
        STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID
    )
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_binding_record_schema_keeps_runtime_and_parent_mutation_disabled() -> None:
    row = {
        "runtime_binding_gate_design_row_id": "gate:0001",
        "strictEvidenceId": "strict-evidence:paper-1:section:1",
        "sourceSpanId": "source-span:paper-1:section:1",
        "candidateRecordId": "candidate:paper-1:section:1",
        "eligibilityRecordId": "eligibility:strict-evidence:paper-1:section:1",
        "citationGradeRecordId": "strict-evidence-citation-grade:strict-evidence:paper-1:section:1",
        "paper_id": "paper-1",
        "artifact_type": "section",
        "sourceContentHash": "abc123def456",
        "runtime_binding_gate_design_status": "runtime_binding_gate_design_candidate_only",
    }
    record = build_sample_runtime_binding_record_from_gate_design_row(row)

    assert record["plannedWriteTarget"] == RUNTIME_BINDING_STORE
    assert record["policyVersion"] == "strict_evidence_runtime_binding_policy.v1"
    assert record["runtimeBindingDecision"] == RUNTIME_BINDING_DECISION
    assert record["runtimeVisible"] is False
    assert record["answerIntegrationVisible"] is False
    assert record["citationGradeRecordInPlaceMutationAllowed"] is False
    assert record["runtimeBindingMutationApplied"] is False
    assert record["runtimeEvidence"] is False
    assert record["writePolicy"]["runtimeBindingRecordWrite"] is False
    assert record["writePolicy"]["vaultScan"] is False
    assert record["writePolicy"]["answerIntegrationChanged"] is False

    assert validate_payload(record, STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID, strict=True).ok
    assert validate_runtime_binding_record_semantics(record) == []


def test_runtime_binding_record_semantics_rejects_mutation_flags() -> None:
    record = build_sample_runtime_binding_record_from_gate_design_row(
        {
            "runtime_binding_gate_design_row_id": "gate:0001",
            "strictEvidenceId": "strict-evidence:paper-1:section:1",
            "sourceSpanId": "source-span:paper-1:section:1",
            "candidateRecordId": "candidate:paper-1:section:1",
            "eligibilityRecordId": "eligibility:strict-evidence:paper-1:section:1",
            "citationGradeRecordId": "strict-evidence-citation-grade:strict-evidence:paper-1:section:1",
            "paper_id": "paper-1",
            "artifact_type": "section",
            "sourceContentHash": "abc123def456",
        }
    )
    record["runtimeVisible"] = True

    assert validate_payload(record, STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID, strict=True).ok is False
    assert "runtimeVisible_must_be_false" in validate_runtime_binding_record_semantics(record)


def test_contract_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    gate_design_path = _runtime_binding_gate_design_report_path(tmp_path)

    payload = build_strict_evidence_runtime_binding_record_contract(
        runtime_binding_gate_design_report_path=gate_design_path,
        expected_input_rows=1,
        expected_planned_runtime_binding_rows=1,
    )
    paths = write_strict_evidence_runtime_binding_record_contract_reports(
        payload,
        tmp_path / "reports",
    )

    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert report["schema"] == STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID
    assert summary["counts"]["runtimeBindingRecordSchemas"] == 1
    assert "Strict Evidence Runtime Binding Record Contract" in markdown
    assert validate_payload(
        report,
        STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok


def test_contract_integrated_measured_local_report() -> None:
    payload = build_strict_evidence_runtime_binding_record_contract()
    assert payload["status"] == "ok"
    assert payload["counts"]["inputRows"] == 99
    assert payload["counts"]["runtimeBindingRecordContracts"] == 1
    assert payload["counts"]["runtimeBindingRecordSchemas"] == 1
    assert payload["counts"]["plannedRuntimeBindingRows"] == 99
    assert payload["counts"]["sectionRuntimeBindingGateDesignRows"] == 45
    assert payload["counts"]["figureCaptionRuntimeBindingGateDesignRows"] == 54
    assert payload["counts"]["runtimeBindingRecordWriteRows"] == 0
    assert payload["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert payload["counts"]["answerIntegrationChangedRows"] == 0
    assert payload["counts"]["vaultScanRows"] == 0
    assert payload["writeTargets"][0] == RUNTIME_BINDING_STORE_CONTRACT
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok
