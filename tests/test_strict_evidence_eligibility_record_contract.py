from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_eligibility_record_contract import (
    ELIGIBILITY_DECISION,
    ELIGIBILITY_POLICY_VERSION,
    ELIGIBILITY_STORE_CONTRACT,
    KNOWN_WRITE_TARGET_CONTRACTS,
    STRICT_EVIDENCE_ELIGIBILITY_RECORD_CONTRACT_SCHEMA_ID,
    STRICT_EVIDENCE_ELIGIBILITY_RECORD_SCHEMA_ID,
    STRICT_EVIDENCE_ELIGIBILITY_STORE,
    build_sample_eligibility_record_from_decision_row,
    build_strict_evidence_eligibility_record_contract,
    validate_eligibility_record_semantics,
    write_strict_evidence_eligibility_record_contract_reports,
)
from knowledge_hub.papers.strict_evidence_strict_eligible_mutation_decision_record import (
    DECISION_SEPARATE_ELIGIBILITY_RECORD,
    DECISION_STATUS_CANDIDATE_ONLY,
    STRICT_EVIDENCE_STRICT_ELIGIBLE_MUTATION_DECISION_RECORD_SCHEMA_ID,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _decision_row(*, index: int = 1, artifact_type: str = "section") -> dict:
    return {
        "decision_row_id": f"strict-evidence-strict-eligible-mutation-decision-record:{index:04d}",
        "hold_row_id": f"strict-evidence-post-pilot-promotion-hold-review:{index:04d}",
        "completion_row_id": f"completion:{index:04d}",
        "readback_row_id": f"readback:{index:04d}",
        "strictEvidenceId": f"strict-evidence:paper-1:{artifact_type}:{index}",
        "sourceSpanId": f"source-span:paper-1:{artifact_type}:{index}",
        "candidateRecordId": f"candidate:paper-1:{artifact_type}:{index}",
        "paper_id": "paper-1",
        "artifact_type": artifact_type,
        "hold_status": "post_pilot_promotion_hold_active",
        "decision_status": DECISION_STATUS_CANDIDATE_ONLY,
        "decision_blockers": [],
        "strictEligibleMutationDecision": DECISION_SEPARATE_ELIGIBILITY_RECORD,
        "strictEvidenceInPlaceMutationAllowed": False,
        "strictEligibleBooleanMutationAllowed": False,
        "eligibilityRecordRequired": True,
        "eligibilityRecordWriteAllowed": False,
        "strictEligibleMutationAllowed": False,
        "citationGradeAllowed": False,
        "runtimeEvidenceAllowed": False,
        "parserRoutingAllowed": False,
        "answerIntegrationAllowed": False,
        "databaseMutationAllowed": False,
        "reindexOrReembedAllowed": False,
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "recommended_action": "queue_for_strict_evidence_eligibility_record_contract",
    }


def _decision_report(rows: list[dict]) -> dict:
    section_rows = sum(1 for row in rows if row.get("artifact_type") == "section")
    figure_rows = sum(1 for row in rows if row.get("artifact_type") == "figure")
    return {
        "schema": STRICT_EVIDENCE_STRICT_ELIGIBLE_MUTATION_DECISION_RECORD_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-20T00:00:00Z",
        "input": {
            "holdReviewReportPath": "/tmp/hold.json",
            "holdReviewReportSchema": "knowledge-hub.paper.strict-evidence-post-pilot-promotion-hold-review.v1",
            "holdReviewReportStatus": "ok",
            "holdDecision": "strict_evidence_post_pilot_promotion_hold_active",
            "requestedPaperIds": [],
            "expectedPolicyCandidateRows": len(rows),
            "expectedDecisionCandidateRows": len(rows),
            "expectedSectionDecisionRows": section_rows,
            "expectedFigureCaptionDecisionRows": figure_rows,
        },
        "counts": {
            "inputPolicyCandidateRows": len(rows),
            "inputHoldRows": len(rows),
            "decisionRecordCandidateOnlyRows": len(rows),
            "sectionDecisionRows": section_rows,
            "figureCaptionDecisionRows": figure_rows,
            "blockedPostPilotHoldNotActiveRows": 0,
            "blockedDownstreamGateAlreadyEnabledRows": 0,
            "blockedStoreRowCountChangedRows": 0,
            "blockedInputSchemaViolationRows": 0,
            "strictEvidenceStoreRows": 99 if len(rows) == 99 else len(rows),
            "sourceSpanStoreRows": 102 if len(rows) == 99 else len(rows) + 3,
            "strictEligibleMutationAllowedRows": 0,
            "eligibilityRecordWriteRows": 0,
            "strictEvidenceWriteRows": 0,
            "strictEvidenceCreatedRows": 0,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "sourceSpanUpdatedRows": 0,
            "manifestWriteRows": 0,
            "reindexOrReembedRows": 0,
            "schemaViolationCount": 0,
            "byArtifactType": {"section": section_rows, "figure": figure_rows},
            "byDecisionStatus": {DECISION_STATUS_CANDIDATE_ONLY: len(rows)},
            "byRecommendedAction": {
                "queue_for_strict_evidence_eligibility_record_contract": len(rows)
            },
        },
        "strictEligibleSemanticsDecision": {
            "decision": DECISION_SEPARATE_ELIGIBILITY_RECORD,
            "strictEvidenceInPlaceMutationAllowed": False,
            "strictEligibleBooleanMutationAllowed": False,
            "strictEligibleFlagMeaning": "legacy_compatibility_flag_must_remain_false_on_strict_evidence_records",
            "eligibilityRecordRequired": True,
            "eligibilityRecordAppendOnly": True,
            "eligibilityStoreName": STRICT_EVIDENCE_ELIGIBILITY_STORE,
            "eligibilityRecordContractRequired": True,
            "eligibilityRecordRuntimeVisible": False,
            "citationGradeAllowedByThisDecision": False,
            "runtimeEvidenceAllowedByThisDecision": False,
            "answerIntegrationAllowedByThisDecision": False,
            "rationale": ["preserve auditability"],
            "alternativesRejected": [],
        },
        "blockedDownstreamGateMatrix": {},
        "noMutationPolicyMatrix": {
            "reportOnly": True,
            "decisionRecordOnly": True,
            "manifestWrite": False,
            "eligibilityRecordWrite": False,
            "strictEvidenceStoreWrite": False,
            "sourceSpanStoreWrite": False,
            "strictEligibleMutation": False,
            "strictEvidenceCreated": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "gate": {
            "strictEligibleMutationDecisionRecordReady": True,
            "decision": DECISION_SEPARATE_ELIGIBILITY_RECORD,
            "strictEvidenceInPlaceMutationAllowed": False,
            "strictEligibleBooleanMutationAllowed": False,
            "strictEligibleMutationAllowed": False,
            "eligibilityRecordContractRequired": True,
            "eligibilityRecordWriteAllowed": False,
            "citationGradeAllowed": False,
            "runtimeEvidenceAllowed": False,
            "parserRoutingAllowed": False,
            "answerIntegrationAllowed": False,
            "databaseMutationAllowed": False,
            "reindexOrReembedAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "sourceSpanStoreWriteAllowed": False,
            "runManifestWriteAllowed": False,
            "schemaViolations": [],
            "recommendedNextTranche": "strict_evidence_eligibility_record_contract",
        },
        "policy": {"reportOnly": True, "decisionRecordOnly": True},
        "warnings": [],
        "rows": rows,
    }


def test_eligibility_record_contract_declares_separate_store_target(tmp_path: Path) -> None:
    rows = [_decision_row(index=1, artifact_type="section"), _decision_row(index=2, artifact_type="figure")]
    decision_path = tmp_path / "decision-record.json"
    _write_json(decision_path, _decision_report(rows))

    payload = build_strict_evidence_eligibility_record_contract(
        decision_record_report_path=decision_path,
    )

    assert payload["schema"] == STRICT_EVIDENCE_ELIGIBILITY_RECORD_CONTRACT_SCHEMA_ID
    assert payload["status"] == "ok"
    assert payload["counts"]["writeTargetContracts"] == 1
    assert payload["counts"]["eligibilityStoreContracts"] == 1
    assert payload["counts"]["eligibilityRecordSchemas"] == 1
    assert payload["counts"]["executorImplementedRows"] == 0
    assert payload["counts"]["eligibilityRecordWriteRows"] == 0
    assert payload["counts"]["strictEligibleMutationRows"] == 0
    assert payload["gate"]["executorReady"] is False
    assert payload["gate"]["recommendedNextTranche"] == "strict_evidence_eligibility_executor_dry_run"

    write_target = payload["writeTargets"][0]
    assert write_target["plannedWriteTarget"] == STRICT_EVIDENCE_ELIGIBILITY_STORE
    assert write_target["recordPathTemplate"] == (
        "{papers_dir}/structured_evidence/strict_evidence_eligibility/{paper_id}.jsonl"
    )
    assert write_target["strictEvidenceInPlaceMutationAllowed"] is False
    assert write_target["strictEligibleBooleanMutationAllowed"] is False
    assert "strictEvidenceId_resolves_to_existing_strict_evidence_jsonl" in write_target["readbackChecks"]
    assert KNOWN_WRITE_TARGET_CONTRACTS[STRICT_EVIDENCE_ELIGIBILITY_STORE] == (
        STRICT_EVIDENCE_ELIGIBILITY_RECORD_CONTRACT_SCHEMA_ID
    )
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_ELIGIBILITY_RECORD_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok


def test_eligibility_record_schema_keeps_runtime_and_mutation_disabled() -> None:
    record = build_sample_eligibility_record_from_decision_row(_decision_row())

    assert record["plannedWriteTarget"] == STRICT_EVIDENCE_ELIGIBILITY_STORE
    assert record["eligibilityPolicyVersion"] == ELIGIBILITY_POLICY_VERSION
    assert record["eligibilityDecision"] == ELIGIBILITY_DECISION
    assert record["strictEvidenceInPlaceMutationAllowed"] is False
    assert record["strictEligibleBooleanMutationAllowed"] is False
    assert record["strictEligibleMutationApplied"] is False
    assert record["citationGrade"] is False
    assert record["runtimeEvidence"] is False
    assert record["runtimeVisible"] is False
    assert record["writePolicy"]["eligibilityRecordWrite"] is False
    assert record["writePolicy"]["strictEligibleMutation"] is False
    assert record["writePolicy"]["answerIntegrationChanged"] is False

    assert validate_payload(record, STRICT_EVIDENCE_ELIGIBILITY_RECORD_SCHEMA_ID, strict=True).ok
    assert validate_eligibility_record_semantics(record) == []


def test_eligibility_record_semantics_rejects_mutation_flags() -> None:
    record = build_sample_eligibility_record_from_decision_row(_decision_row())
    record["strictEligibleMutationApplied"] = True

    assert validate_payload(record, STRICT_EVIDENCE_ELIGIBILITY_RECORD_SCHEMA_ID, strict=True).ok is False
    assert "strictEligibleMutationApplied_must_be_false" in validate_eligibility_record_semantics(record)


def test_contract_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    decision_path = tmp_path / "decision-record.json"
    _write_json(decision_path, _decision_report([_decision_row()]))

    payload = build_strict_evidence_eligibility_record_contract(
        decision_record_report_path=decision_path,
    )
    paths = write_strict_evidence_eligibility_record_contract_reports(
        payload,
        tmp_path / "reports",
    )

    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert report["schema"] == STRICT_EVIDENCE_ELIGIBILITY_RECORD_CONTRACT_SCHEMA_ID
    assert summary["counts"]["eligibilityRecordSchemas"] == 1
    assert "Strict Evidence Eligibility Record Contract" in markdown
    assert validate_payload(
        report,
        STRICT_EVIDENCE_ELIGIBILITY_RECORD_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok


def test_contract_integrated_measured_local_report() -> None:
    payload = build_strict_evidence_eligibility_record_contract()
    assert payload["status"] == "ok"
    assert payload["counts"]["decisionRecordCandidateOnlyRows"] == 99
    assert validate_payload(
        payload,
        STRICT_EVIDENCE_ELIGIBILITY_RECORD_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok
