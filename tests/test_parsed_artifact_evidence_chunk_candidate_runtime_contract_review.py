from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_runtime_contract_review import (
    CONTRACT_STATUS_READY_CANDIDATE_ONLY,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_CONTRACT_REVIEW_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_candidate_runtime_contract_review,
    write_parsed_artifact_evidence_chunk_candidate_runtime_contract_review,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_INTEGRATION_DRY_RUN_SCHEMA_ID,
    READY_DECISION as RUNTIME_DRY_RUN_READY_DECISION,
    RUNTIME_STATUS_READY_CANDIDATE_ONLY,
)


RUNTIME_DRY_RUN_REF = (
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run.v1.json"
)


def _runtime_row(index: int, *, artifact_type: str = "paragraph") -> dict[str, object]:
    start = index * 100
    end = start + 80
    paper_id = f"paper-{index}"
    source_ref = f"papers_dir/parsed/{paper_id}/document.md"
    source_hash = "sha256:" + f"{index:064d}"[-64:]
    snippet_hash = "sha256:" + f"{index + 1:064d}"[-64:]
    span_locator = f"chars:{start}-{end}"
    candidate_record_id = f"parsed-artifact-evidence-chunk-candidate:{paper_id}:paragraph:key"
    candidate_store_ref = f"papers_dir/structured_evidence_candidates/evidence_chunk/{paper_id}.jsonl"
    return {
        "runtimeDryRunRowId": f"parsed-artifact-evidence-chunk-candidate-runtime-integration-dry-run:{index:04d}",
        "sourcePolicyGateRowId": f"parsed-artifact-evidence-chunk-candidate-answerability-policy-gate:{index:04d}",
        "candidateRecordId": candidate_record_id,
        "sourceCandidateRowId": f"parsed-artifact-evidence-chunk:{paper_id}:{index}",
        "paperId": paper_id,
        "artifactType": artifact_type,
        "sourceRef": source_ref,
        "sourceContentHash": source_hash,
        "spanLocator": span_locator,
        "charStart": start,
        "charEnd": end,
        "snippetHash": snippet_hash,
        "candidateStoreRef": candidate_store_ref,
        "runtimeIntegrationStatus": RUNTIME_STATUS_READY_CANDIDATE_ONLY,
        "runtimeIntegrationBlockers": [],
        "runtimeIntegrationDryRunReady": True,
        "plannedRuntimeSpanId": f"parsed-artifact-evidence-chunk-runtime-preview:{index:04d}",
        "plannedEvidencePacketSpan": {
            "spanRef": f"span:{index}",
            "span_id": f"span:{index}",
            "sourceType": "paper",
            "source_type": "paper",
            "sourceId": paper_id,
            "source_id": paper_id,
            "sourceRef": source_ref,
            "source_ref": source_ref,
            "sourceContentHash": source_hash,
            "source_content_hash": source_hash,
            "content_hash": snippet_hash,
            "snippetHash": snippet_hash,
            "snippet_hash": snippet_hash,
            "spanLocator": span_locator,
            "span_locator": span_locator,
            "locator": span_locator,
            "charStart": start,
            "char_start": start,
            "charEnd": end,
            "char_end": end,
            "spanOffsetAvailable": True,
            "artifactType": artifact_type,
            "evidenceKind": "parsed_artifact_evidence_chunk_candidate_runtime_preview",
            "evidence_kind": "parsed_artifact_evidence_chunk_candidate_runtime_preview",
            "derivativeSource": {
                "candidateRecordId": candidate_record_id,
                "policyGateRowId": f"parsed-artifact-evidence-chunk-candidate-answerability-policy-gate:{index:04d}",
                "sourceCandidateRowId": f"parsed-artifact-evidence-chunk:{paper_id}:{index}",
                "candidateStoreRef": candidate_store_ref,
            },
            "textPreviewIncluded": False,
            "candidateOnly": True,
            "runtimeEvidence": False,
            "answerVisible": False,
            "answerable": False,
            "strictEvidence": False,
            "citationGrade": False,
        },
        "plannedEvidencePacketSpanReady": True,
        "plannedAnswerContextCandidate": True,
        "runtimeVisibilityAllowed": False,
        "answerVisibleAllowed": False,
        "answerableAllowed": False,
        "futureRuntimeIntegrationRequired": True,
        "candidateOnly": True,
        "strictEvidence": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "answerVisible": False,
        "answerable": False,
        "recommendedAction": "queue_for_runtime_contract_review",
        "checks": {
            "answerabilityPolicyReady": True,
            "runtimeIntegrationDryRunReadyInput": True,
            "artifactTypeAllowed": True,
            "sourceContentHashPresent": True,
            "snippetHashPresent": True,
            "charsLocatorStrict": True,
            "spanOffsetAvailable": True,
            "evidencePacketPreviewOnly": True,
            "textPreviewIncluded": False,
            "answerableStillFalse": True,
            "runtimeVisibilityStillFalse": True,
        },
    }


def _runtime_report(rows: list[dict[str, object]] | None = None, *, status: str = "ready") -> dict[str, object]:
    dry_run_rows = rows or [_runtime_row(1), _runtime_row(2, artifact_type="section")]
    ready = status == "ready"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_INTEGRATION_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": "2026-05-29T00:00:00Z",
        "decision": RUNTIME_DRY_RUN_READY_DECISION if ready else "blocked",
        "nextRecommendedTranche": "parsed_artifact_evidence_chunk_candidate_runtime_contract_review",
        "sourceAnswerabilityPolicyGate": {
            "schema": "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-answerability-policy-gate.v1",
            "status": "ready" if ready else "blocked",
            "decision": "parsed_artifact_evidence_chunk_candidate_answerability_policy_gate_ready" if ready else "blocked",
            "nextRecommendedTranche": "parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run",
            "reportRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_answerability_policy_gate.v1.json",
            "inputRows": len(dry_run_rows),
            "answerabilityPolicyReadyRows": len(dry_run_rows) if ready else 0,
            "runtimeIntegrationDryRunReadyRows": len(dry_run_rows) if ready else 0,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "policy": {
            "reportOnly": True,
            "runtimeIntegrationDryRunOnly": True,
            "evidencePacketPreviewOnly": True,
            "allowedArtifactTypes": ["paragraph", "section"],
            "textPreviewIncluded": False,
            "candidateStoreWrite": False,
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "answerVisibleExposure": False,
            "answerGeneration": False,
            "answerableRowsAllowedInThisTranche": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "parserExecution": False,
            "canonicalParsedArtifactsWritten": False,
            "futureRuntimeIntegrationRequired": True,
            "nextGateRequiredBeforeRuntimeUse": True,
        },
        "counts": {
            "inputRows": len(dry_run_rows),
            "policyReadyInputRows": len(dry_run_rows) if ready else 0,
            "runtimeIntegrationDryRunReadyRows": len(dry_run_rows) if ready else 0,
            "plannedRuntimeSpanRows": len(dry_run_rows) if ready else 0,
            "plannedEvidencePacketSpanRows": len(dry_run_rows) if ready else 0,
            "plannedAnswerContextCandidateRows": len(dry_run_rows) if ready else 0,
            "answerableRows": 0,
            "blockedRows": 0,
            "blockedInputSchemaViolationRows": 0,
            "blockedPolicyNotReadyRows": 0,
            "blockedUnsupportedArtifactTypeRows": 0,
            "blockedMissingRequiredFieldRows": 0,
            "blockedInvalidLocatorRows": 0,
            "blockedPolicyQuarantineRows": 0,
            "candidateStoreWriteRows": 0,
            "sourceSpanCreatedRows": 0,
            "strictEvidenceRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
            "answerVisibleRows": 0,
            "answerGenerationRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "parserExecutionRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
            "byArtifactType": {"paragraph": 1, "section": 1},
            "byRuntimeIntegrationStatus": {RUNTIME_STATUS_READY_CANDIDATE_ONLY: len(dry_run_rows)},
        },
        "gate": {
            "readyForRuntimeContractReview": ready,
            "readyForRuntimeApply": False,
            "candidateStoreWriteAllowed": False,
            "sourceSpanCreationAllowed": False,
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "answerVisibleAllowed": False,
            "answerableAllowed": False,
            "schemaViolations": [],
        },
        "rows": dry_run_rows,
        "warnings": [],
    }


def _build(runtime_report: dict[str, object]) -> dict[str, object]:
    return build_parsed_artifact_evidence_chunk_candidate_runtime_contract_review(
        runtime_integration_dry_run_report=runtime_report,
        source_runtime_integration_dry_run_report_ref=RUNTIME_DRY_RUN_REF,
        generated_at="2026-05-29T00:00:00Z",
    )


def test_runtime_contract_review_accepts_evidence_packet_compatible_preview_rows() -> None:
    report = _build(_runtime_report())

    assert report["schema"] == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_CONTRACT_REVIEW_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["inputRows"] == 2
    assert report["counts"]["contractReviewReadyRows"] == 2
    assert report["counts"]["evidencePacketSpanCompatibleRows"] == 2
    assert report["counts"]["answerContractCitationShapeCompatibleRows"] == 2
    assert report["counts"]["runtimeEvidenceRows"] == 0
    assert report["counts"]["answerVisibleRows"] == 0
    assert report["counts"]["answerableRows"] == 0
    assert {row["contractReviewStatus"] for row in report["rows"]} == {
        CONTRACT_STATUS_READY_CANDIDATE_ONLY
    }
    assert report["rows"][0]["strictProvenanceShapePresent"] is True
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_CONTRACT_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_contract_review_blocks_non_ready_source_report() -> None:
    report = _build(_runtime_report(status="blocked"))

    assert report["status"] == "blocked"
    assert "runtime_integration_dry_run_not_ready" in report["gate"]["schemaViolations"]
    assert report["counts"]["contractReviewReadyRows"] == 0
    assert report["counts"]["blockedInputSchemaViolationRows"] == 2


def test_runtime_contract_review_blocks_source_identity_mismatch() -> None:
    row = copy.deepcopy(_runtime_row(1))
    row["plannedEvidencePacketSpan"]["source_id"] = "wrong-paper"

    report = _build(_runtime_report([row]))

    assert report["status"] == "blocked"
    assert report["counts"]["blockedSourceIdMismatchRows"] == 1
    assert "planned_span_source_id_mismatch" in report["rows"][0]["contractReviewBlockers"]


def test_runtime_contract_review_blocks_hash_mismatch() -> None:
    row = copy.deepcopy(_runtime_row(1))
    row["plannedEvidencePacketSpan"]["source_content_hash"] = "sha256:" + "f" * 64

    report = _build(_runtime_report([row]))

    assert report["status"] == "blocked"
    assert report["counts"]["blockedHashMismatchRows"] == 1
    assert "planned_span_source_hash_mismatch" in report["rows"][0]["contractReviewBlockers"]


def test_runtime_contract_review_blocks_locator_mismatch() -> None:
    row = copy.deepcopy(_runtime_row(1))
    row["plannedEvidencePacketSpan"]["span_locator"] = "chars:1-2"

    report = _build(_runtime_report([row]))

    assert report["status"] == "blocked"
    assert report["counts"]["blockedLocatorMismatchRows"] == 1
    assert "planned_span_locator_mismatch" in report["rows"][0]["contractReviewBlockers"]


def test_runtime_contract_review_blocks_policy_quarantine_violation() -> None:
    row = copy.deepcopy(_runtime_row(1))
    row["plannedEvidencePacketSpan"]["answerVisible"] = True

    report = _build(_runtime_report([row]))

    assert report["status"] == "blocked"
    assert report["counts"]["blockedPolicyQuarantineRows"] == 1
    assert "planned_span_answer_visible_not_false" in report["rows"][0]["contractReviewBlockers"]


def test_runtime_contract_review_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = _build(_runtime_report())
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_parsed_artifact_evidence_chunk_candidate_runtime_contract_review(
        report,
        report_json=json_path,
        report_md=md_path,
    )

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["sourceRuntimeIntegrationDryRun"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
