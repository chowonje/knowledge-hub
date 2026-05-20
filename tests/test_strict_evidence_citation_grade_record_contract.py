from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_citation_grade_policy_gate_design import (
    build_strict_evidence_citation_grade_policy_gate_design,
    write_strict_evidence_citation_grade_policy_gate_design_reports,
)
from knowledge_hub.papers.strict_evidence_citation_grade_record_contract import (
    CITATION_GRADE_DECISION,
    CITATION_GRADE_POLICY_VERSION,
    CITATION_GRADE_STORE_CONTRACT,
    KNOWN_WRITE_TARGET_CONTRACTS,
    STRICT_EVIDENCE_CITATION_GRADE_RECORD_CONTRACT_SCHEMA_ID,
    STRICT_EVIDENCE_CITATION_GRADE_RECORD_SCHEMA_ID,
    STRICT_EVIDENCE_CITATION_GRADE_STORE,
    build_sample_citation_grade_record_from_policy_row,
    build_strict_evidence_citation_grade_record_contract,
    validate_citation_grade_record_semantics,
    write_strict_evidence_citation_grade_record_contract_reports,
)
from tests.test_strict_evidence_citation_grade_policy_gate_design import (
    _hold_review_report_path,
)


def _policy_gate_design_report_path(tmp_path: Path) -> Path:
    hold_path = _hold_review_report_path(tmp_path)
    report = build_strict_evidence_citation_grade_policy_gate_design(
        hold_review_report_path=hold_path,
        expected_input_rows=1,
        expected_candidate_rows=1,
        expected_section_rows=1,
        expected_figure_caption_rows=0,
        expected_strict_evidence_store_rows=1,
        expected_source_span_store_rows=1,
    )
    output_dir = tmp_path / "policy-gate"
    write_strict_evidence_citation_grade_policy_gate_design_reports(
        report,
        output_dir,
    )
    return output_dir / "strict-evidence-citation-grade-policy-gate-design.json"


def test_citation_grade_record_contract_declares_separate_store_target(tmp_path: Path) -> None:
    policy_path = _policy_gate_design_report_path(tmp_path)

    payload = build_strict_evidence_citation_grade_record_contract(
        policy_gate_design_report_path=policy_path,
    )

    assert payload["schema"] == STRICT_EVIDENCE_CITATION_GRADE_RECORD_CONTRACT_SCHEMA_ID
    assert payload["status"] == "ok"
    assert payload["counts"]["writeTargetContracts"] == 1
    assert payload["counts"]["citationGradeStoreContracts"] == 1
    assert payload["counts"]["citationGradeRecordSchemas"] == 1
    assert payload["counts"]["executorImplementedRows"] == 0
    assert payload["counts"]["citationGradeRecordWriteRows"] == 0
    assert payload["counts"]["citationGradeBooleanMutationRows"] == 0
    assert payload["gate"]["executorReady"] is False
    assert payload["gate"]["recommendedNextTranche"] == "strict_evidence_citation_grade_executor_dry_run"

    write_target = payload["writeTargets"][0]
    assert write_target["plannedWriteTarget"] == STRICT_EVIDENCE_CITATION_GRADE_STORE
    assert write_target["recordPathTemplate"] == (
        "{papers_dir}/structured_evidence/strict_evidence_citation_grade/{paper_id}.jsonl"
    )
    assert write_target["strictEvidenceInPlaceMutationAllowed"] is False
    assert write_target["eligibilityRecordInPlaceMutationAllowed"] is False
    assert write_target["citationGradeBooleanMutationAllowed"] is False
    assert "eligibilityRecordId_resolves_to_existing_eligibility_jsonl" in write_target[
        "readbackChecks"
    ]
    assert KNOWN_WRITE_TARGET_CONTRACTS[STRICT_EVIDENCE_CITATION_GRADE_STORE] == (
        STRICT_EVIDENCE_CITATION_GRADE_RECORD_CONTRACT_SCHEMA_ID
    )
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_CITATION_GRADE_RECORD_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok


def test_citation_grade_record_schema_keeps_runtime_and_parent_mutation_disabled() -> None:
    row = {
        "policy_design_row_id": "policy:0001",
        "hold_row_id": "hold:0001",
        "strictEvidenceId": "strict-evidence:paper-1:section:1",
        "sourceSpanId": "source-span:paper-1:section:1",
        "candidateRecordId": "candidate:paper-1:section:1",
        "eligibilityRecordId": "eligibility:strict-evidence:paper-1:section:1",
        "paper_id": "paper-1",
        "artifact_type": "section",
        "citation_grade_policy_design_status": "citation_grade_policy_design_candidate_only",
    }
    record = build_sample_citation_grade_record_from_policy_row(row)

    assert record["plannedWriteTarget"] == STRICT_EVIDENCE_CITATION_GRADE_STORE
    assert record["citationGradePolicyVersion"] == CITATION_GRADE_POLICY_VERSION
    assert record["citationGradeDecision"] == CITATION_GRADE_DECISION
    assert record["strictEvidenceInPlaceMutationAllowed"] is False
    assert record["eligibilityRecordInPlaceMutationAllowed"] is False
    assert record["sourceSpanInPlaceMutationAllowed"] is False
    assert record["citationGradeBooleanMutationAllowed"] is False
    assert record["citationGradeMutationApplied"] is False
    assert record["runtimeEvidence"] is False
    assert record["runtimeVisible"] is False
    assert record["writePolicy"]["citationGradeRecordWrite"] is False
    assert record["writePolicy"]["citationGradeBooleanMutation"] is False
    assert record["writePolicy"]["answerIntegrationChanged"] is False

    assert validate_payload(record, STRICT_EVIDENCE_CITATION_GRADE_RECORD_SCHEMA_ID, strict=True).ok
    assert validate_citation_grade_record_semantics(record) == []


def test_citation_grade_record_semantics_rejects_mutation_flags() -> None:
    record = build_sample_citation_grade_record_from_policy_row(
        {
            "policy_design_row_id": "policy:0001",
            "hold_row_id": "hold:0001",
            "strictEvidenceId": "strict-evidence:paper-1:section:1",
            "sourceSpanId": "source-span:paper-1:section:1",
            "candidateRecordId": "candidate:paper-1:section:1",
            "eligibilityRecordId": "eligibility:strict-evidence:paper-1:section:1",
            "paper_id": "paper-1",
            "artifact_type": "section",
        }
    )
    record["citationGradeMutationApplied"] = True

    assert validate_payload(record, STRICT_EVIDENCE_CITATION_GRADE_RECORD_SCHEMA_ID, strict=True).ok is False
    assert "citationGradeMutationApplied_must_be_false" in validate_citation_grade_record_semantics(record)


def test_contract_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    policy_path = _policy_gate_design_report_path(tmp_path)

    payload = build_strict_evidence_citation_grade_record_contract(
        policy_gate_design_report_path=policy_path,
    )
    paths = write_strict_evidence_citation_grade_record_contract_reports(
        payload,
        tmp_path / "reports",
    )

    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert report["schema"] == STRICT_EVIDENCE_CITATION_GRADE_RECORD_CONTRACT_SCHEMA_ID
    assert summary["counts"]["citationGradeRecordSchemas"] == 1
    assert "Strict Evidence Citation-Grade Record Contract" in markdown
    assert validate_payload(
        report,
        STRICT_EVIDENCE_CITATION_GRADE_RECORD_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok


def test_contract_integrated_measured_local_report() -> None:
    payload = build_strict_evidence_citation_grade_record_contract()
    assert payload["status"] == "ok"
    assert payload["counts"]["citationGradePolicyDesignCandidateOnlyRows"] == 99
    assert payload["counts"]["sectionCitationGradePolicyDesignRows"] == 45
    assert payload["counts"]["figureCaptionCitationGradePolicyDesignRows"] == 54
    assert payload["writeTargets"][0] == CITATION_GRADE_STORE_CONTRACT
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_CITATION_GRADE_RECORD_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok
