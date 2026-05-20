from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_runtime_binding_visibility_decision_record import (
    DECISION_STATUS_CANDIDATE_ONLY,
    build_strict_evidence_runtime_binding_visibility_decision_record,
    write_strict_evidence_runtime_binding_visibility_decision_record_reports,
)
from knowledge_hub.papers.strict_evidence_runtime_binding_visibility_record_contract import (
    KNOWN_WRITE_TARGET_CONTRACTS,
    RUNTIME_VISIBILITY_DECISION,
    RUNTIME_VISIBILITY_STORE,
    RUNTIME_VISIBILITY_STORE_CONTRACT,
    STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_CONTRACT_SCHEMA_ID,
    STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_SCHEMA_ID,
    build_sample_runtime_visibility_record_from_decision_row,
    build_strict_evidence_runtime_binding_visibility_record_contract,
    validate_runtime_visibility_record_semantics,
    write_strict_evidence_runtime_binding_visibility_record_contract_reports,
)
from tests.test_strict_evidence_runtime_binding_visibility_decision_record import (
    _hold_review_report_path,
)


def _visibility_decision_report_path(tmp_path: Path) -> Path:
    hold_path = _hold_review_report_path(tmp_path)
    report = build_strict_evidence_runtime_binding_visibility_decision_record(
        hold_review_report_path=hold_path,
        expected_input_rows=1,
        expected_runtime_binding_record_rows=1,
        expected_section_decision_rows=1,
        expected_figure_caption_decision_rows=0,
        expected_citation_grade_store_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=1,
        expected_source_span_store_rows=1,
    )
    output_dir = tmp_path / "visibility-decision"
    write_strict_evidence_runtime_binding_visibility_decision_record_reports(
        report,
        output_dir,
    )
    return output_dir / "strict-evidence-runtime-binding-visibility-decision-record.json"


def test_runtime_visibility_record_contract_declares_separate_store_target(tmp_path: Path) -> None:
    decision_path = _visibility_decision_report_path(tmp_path)

    payload = build_strict_evidence_runtime_binding_visibility_record_contract(
        visibility_decision_report_path=decision_path,
        expected_input_rows=1,
        expected_planned_runtime_visibility_rows=1,
    )

    assert payload["schema"] == STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_CONTRACT_SCHEMA_ID
    assert payload["status"] == "ok"
    assert payload["counts"]["runtimeVisibilityRecordContracts"] == 1
    assert payload["counts"]["runtimeVisibilityRecordSchemas"] == 1
    assert payload["counts"]["plannedRuntimeVisibilityRows"] == 1
    assert payload["counts"]["executorImplementedRows"] == 0
    assert payload["counts"]["runtimeVisibilityRecordWriteRows"] == 0
    assert payload["counts"]["runtimeVisibleRows"] == 0
    assert payload["counts"]["answerIntegrationVisibleRows"] == 0
    assert payload["gate"]["executorReady"] is False
    assert payload["gate"]["recommendedNextTranche"] == (
        "strict_evidence_runtime_binding_visibility_executor_dry_run"
    )

    write_target = payload["writeTargets"][0]
    assert write_target["plannedWriteTarget"] == RUNTIME_VISIBILITY_STORE
    assert write_target["recordPathTemplate"] == (
        "{papers_dir}/structured_evidence/strict_evidence_runtime_visibility/{paper_id}.jsonl"
    )
    assert write_target["runtimeBindingRecordInPlaceMutationAllowed"] is False
    assert write_target["vaultScanAllowed"] is False
    assert "runtimeBindingRecordId_resolves_to_existing_runtime_binding_jsonl" in write_target[
        "readbackChecks"
    ]
    assert KNOWN_WRITE_TARGET_CONTRACTS[RUNTIME_VISIBILITY_STORE] == (
        STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_CONTRACT_SCHEMA_ID
    )
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_visibility_record_schema_keeps_runtime_and_parent_mutation_disabled() -> None:
    row = {
        "visibility_decision_record_row_id": "visibility-decision:0001",
        "runtimeBindingRecordId": "runtime-binding:paper-1:section:1",
        "strictEvidenceId": "strict-evidence:paper-1:section:1",
        "sourceSpanId": "source-span:paper-1:section:1",
        "candidateRecordId": "candidate:paper-1:section:1",
        "eligibilityRecordId": "eligibility:strict-evidence:paper-1:section:1",
        "citationGradeRecordId": "strict-evidence-citation-grade:strict-evidence:paper-1:section:1",
        "paper_id": "paper-1",
        "artifact_type": "section",
        "sourceContentHash": "abc123def456",
        "decision_status": DECISION_STATUS_CANDIDATE_ONLY,
    }
    record = build_sample_runtime_visibility_record_from_decision_row(row)

    assert record["plannedWriteTarget"] == RUNTIME_VISIBILITY_STORE
    assert record["policyVersion"] == "strict_evidence_runtime_visibility_policy.v1"
    assert record["runtimeVisibilityDecision"] == RUNTIME_VISIBILITY_DECISION
    assert record["runtimeVisible"] is False
    assert record["answerIntegrationVisible"] is False
    assert record["runtimeBindingRecordInPlaceMutationAllowed"] is False
    assert record["runtimeVisibilityMutationApplied"] is False
    assert record["runtimeEvidence"] is False
    assert record["writePolicy"]["runtimeVisibilityRecordWrite"] is False
    assert record["writePolicy"]["runtimeBindingRecordWrite"] is False
    assert record["writePolicy"]["vaultScan"] is False
    assert record["writePolicy"]["answerIntegrationChanged"] is False

    assert validate_payload(
        record,
        STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_SCHEMA_ID,
        strict=True,
    ).ok
    assert validate_runtime_visibility_record_semantics(record) == []


def test_runtime_visibility_record_semantics_rejects_mutation_flags() -> None:
    record = build_sample_runtime_visibility_record_from_decision_row(
        {
            "visibility_decision_record_row_id": "visibility-decision:0001",
            "runtimeBindingRecordId": "runtime-binding:paper-1:section:1",
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

    assert validate_payload(
        record,
        STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_SCHEMA_ID,
        strict=True,
    ).ok is False
    assert "runtimeVisible_must_be_false" in validate_runtime_visibility_record_semantics(record)


def test_contract_blocks_if_visibility_decision_report_enables_writes(tmp_path: Path) -> None:
    decision_path = _visibility_decision_report_path(tmp_path)
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    payload["counts"]["runtimeVisibilityRecordWriteRows"] = 1
    decision_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = build_strict_evidence_runtime_binding_visibility_record_contract(
        visibility_decision_report_path=decision_path,
        expected_input_rows=1,
        expected_planned_runtime_visibility_rows=1,
    )

    assert report["status"] == "blocked"
    assert report["gate"]["recommendedNextTranche"] == (
        "strict_evidence_runtime_binding_visibility_decision_record_input_repair"
    )


def test_contract_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    decision_path = _visibility_decision_report_path(tmp_path)

    payload = build_strict_evidence_runtime_binding_visibility_record_contract(
        visibility_decision_report_path=decision_path,
        expected_input_rows=1,
        expected_planned_runtime_visibility_rows=1,
    )
    paths = write_strict_evidence_runtime_binding_visibility_record_contract_reports(
        payload,
        tmp_path / "reports",
    )

    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert report["schema"] == STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_CONTRACT_SCHEMA_ID
    assert summary["counts"]["runtimeVisibilityRecordSchemas"] == 1
    assert "Strict Evidence Runtime Binding Visibility Record Contract" in markdown
    assert validate_payload(
        report,
        STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok


def test_contract_integrated_measured_local_report() -> None:
    payload = build_strict_evidence_runtime_binding_visibility_record_contract()
    assert payload["status"] == "ok"
    assert payload["counts"]["inputRows"] == 99
    assert payload["counts"]["runtimeVisibilityRecordContracts"] == 1
    assert payload["counts"]["runtimeVisibilityRecordSchemas"] == 1
    assert payload["counts"]["plannedRuntimeVisibilityRows"] == 99
    assert payload["counts"]["sectionVisibilityDecisionRows"] == 45
    assert payload["counts"]["figureCaptionVisibilityDecisionRows"] == 54
    assert payload["counts"]["runtimeVisibilityRecordWriteRows"] == 0
    assert payload["counts"]["runtimeBindingRecordWriteRows"] == 0
    assert payload["counts"]["runtimeVisibleRows"] == 0
    assert payload["counts"]["answerIntegrationVisibleRows"] == 0
    assert payload["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert payload["counts"]["answerIntegrationChangedRows"] == 0
    assert payload["counts"]["vaultScanRows"] == 0
    assert payload["writeTargets"][0] == RUNTIME_VISIBILITY_STORE_CONTRACT
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_RECORD_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok
