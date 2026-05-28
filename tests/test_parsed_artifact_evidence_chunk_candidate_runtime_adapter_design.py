from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_runtime_adapter_design import (
    ADAPTER_STATUS_READY_CANDIDATE_ONLY,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DESIGN_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_candidate_runtime_adapter_design,
    write_parsed_artifact_evidence_chunk_candidate_runtime_adapter_design,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_runtime_contract_review import (
    CONTRACT_STATUS_READY_CANDIDATE_ONLY,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_CONTRACT_REVIEW_SCHEMA_ID,
    READY_DECISION as CONTRACT_REVIEW_READY_DECISION,
)


CONTRACT_REVIEW_REF = (
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_runtime_contract_review.v1.json"
)


def _contract_row(index: int, *, artifact_type: str = "paragraph") -> dict[str, object]:
    start = index * 100
    end = start + 80
    paper_id = f"paper-{index}"
    source_ref = f"papers_dir/parsed/{paper_id}/document.md"
    source_hash = "sha256:" + f"{index:064d}"[-64:]
    snippet_hash = "sha256:" + f"{index + 1:064d}"[-64:]
    span_locator = f"chars:{start}-{end}"
    candidate_record_id = f"parsed-artifact-evidence-chunk-candidate:{paper_id}:{artifact_type}:key"
    candidate_store_ref = f"papers_dir/structured_evidence_candidates/evidence_chunk/{paper_id}.jsonl"
    return {
        "contractReviewRowId": f"parsed-artifact-evidence-chunk-candidate-runtime-contract-review:{index:04d}",
        "sourceRuntimeDryRunRowId": f"parsed-artifact-evidence-chunk-candidate-runtime-integration-dry-run:{index:04d}",
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
        "contractReviewStatus": CONTRACT_STATUS_READY_CANDIDATE_ONLY,
        "contractReviewBlockers": [],
        "contractReviewReady": True,
        "evidencePacketSpanCompatible": True,
        "answerContractCitationShapeCompatible": True,
        "strictProvenanceShapePresent": True,
        "plannedEvidencePacketSpan": {
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
                "candidateStoreRef": candidate_store_ref,
                "sourceCandidateRowId": f"parsed-artifact-evidence-chunk:{paper_id}:{index}",
            },
            "textPreviewIncluded": False,
            "candidateOnly": True,
            "runtimeEvidence": False,
            "answerVisible": False,
            "answerable": False,
            "strictEvidence": False,
            "citationGrade": False,
        },
        "runtimeAdapterDesignCandidate": True,
        "runtimeVisibilityAllowed": False,
        "answerVisibleAllowed": False,
        "answerableAllowed": False,
        "candidateOnly": True,
        "strictEvidence": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "answerVisible": False,
        "answerable": False,
        "recommendedAction": "queue_for_runtime_adapter_design",
        "checks": {},
    }


def _contract_report(rows: list[dict[str, object]] | None = None, *, status: str = "ready") -> dict[str, object]:
    contract_rows = rows or [_contract_row(1), _contract_row(2, artifact_type="section")]
    ready = status == "ready"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_CONTRACT_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": "2026-05-29T00:00:00Z",
        "decision": CONTRACT_REVIEW_READY_DECISION if ready else "blocked",
        "nextRecommendedTranche": "parsed_artifact_evidence_chunk_candidate_runtime_adapter_design",
        "sourceRuntimeIntegrationDryRun": {
            "schema": "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-runtime-integration-dry-run.v1",
            "status": "ready",
            "decision": "parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run_ready",
            "nextRecommendedTranche": "parsed_artifact_evidence_chunk_candidate_runtime_contract_review",
            "reportRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run.v1.json",
            "inputRows": len(contract_rows),
            "runtimeIntegrationDryRunReadyRows": len(contract_rows),
            "plannedEvidencePacketSpanRows": len(contract_rows),
            "answerableRows": 0,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "policy": {
            "reportOnly": True,
            "runtimeContractReviewOnly": True,
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
            "futureRuntimeAdapterRequired": True,
            "nextGateRequiredBeforeRuntimeUse": True,
        },
        "counts": {
            "inputRows": len(contract_rows),
            "runtimeDryRunReadyInputRows": len(contract_rows) if ready else 0,
            "contractReviewReadyRows": len(contract_rows) if ready else 0,
            "evidencePacketSpanCompatibleRows": len(contract_rows) if ready else 0,
            "answerContractCitationShapeCompatibleRows": len(contract_rows) if ready else 0,
            "strictProvenanceShapeRows": len(contract_rows) if ready else 0,
            "runtimeAdapterDesignCandidateRows": len(contract_rows) if ready else 0,
            "answerableRows": 0,
            "blockedRows": 0,
            "blockedInputSchemaViolationRows": 0,
            "blockedRuntimePreviewNotReadyRows": 0,
            "blockedSourceIdMismatchRows": 0,
            "blockedHashMismatchRows": 0,
            "blockedLocatorMismatchRows": 0,
            "blockedMissingRequiredFieldRows": 0,
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
            "byContractReviewStatus": {CONTRACT_STATUS_READY_CANDIDATE_ONLY: len(contract_rows)},
        },
        "gate": {
            "readyForRuntimeAdapterDesign": ready,
            "readyForRuntimeApply": False,
            "candidateStoreWriteAllowed": False,
            "sourceSpanCreationAllowed": False,
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "answerVisibleAllowed": False,
            "answerableAllowed": False,
            "schemaViolations": [],
        },
        "rows": contract_rows,
        "warnings": [],
    }


def _build(contract_report: dict[str, object]) -> dict[str, object]:
    return build_parsed_artifact_evidence_chunk_candidate_runtime_adapter_design(
        runtime_contract_review_report=contract_report,
        source_runtime_contract_review_report_ref=CONTRACT_REVIEW_REF,
        generated_at="2026-05-29T00:00:00Z",
    )


def test_runtime_adapter_design_accepts_ready_contract_rows() -> None:
    report = _build(_contract_report())

    assert report["schema"] == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DESIGN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["inputRows"] == 2
    assert report["counts"]["runtimeAdapterDesignReadyRows"] == 2
    assert report["counts"]["plannedCandidateStoreReadRows"] == 2
    assert report["counts"]["runtimeEvidenceRows"] == 0
    assert report["counts"]["answerVisibleRows"] == 0
    assert report["counts"]["answerableRows"] == 0
    assert report["gate"]["readyForRuntimeAdapterDryRun"] is True
    assert report["adapterDesign"]["integrationBoundary"] == (
        "knowledge_hub.ai.evidence_assembly.EvidenceAssemblyService.assemble"
    )
    assert report["adapterDesign"]["selectionPolicy"]["maxRowsTotal"] == 4
    assert {row["adapterDesignStatus"] for row in report["rows"]} == {
        ADAPTER_STATUS_READY_CANDIDATE_ONLY
    }
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_adapter_design_blocks_non_ready_source_report() -> None:
    report = _build(_contract_report(status="blocked"))

    assert report["status"] == "blocked"
    assert "runtime_contract_review_not_ready" in report["gate"]["schemaViolations"]
    assert report["counts"]["runtimeAdapterDesignReadyRows"] == 0
    assert report["counts"]["blockedInputSchemaViolationRows"] == 2


def test_runtime_adapter_design_blocks_non_ready_contract_row() -> None:
    row = copy.deepcopy(_contract_row(1))
    row["contractReviewReady"] = False

    report = _build(_contract_report([row]))

    assert report["status"] == "blocked"
    assert report["counts"]["blockedContractNotReadyRows"] == 1
    assert "contractReviewReady_not_true" in report["rows"][0]["adapterDesignBlockers"]


def test_runtime_adapter_design_blocks_missing_required_field() -> None:
    row = copy.deepcopy(_contract_row(1))
    row["sourceContentHash"] = ""

    report = _build(_contract_report([row]))

    assert report["status"] == "blocked"
    assert report["counts"]["blockedMissingRequiredFieldRows"] == 1
    assert "sourceContentHash_missing" in report["rows"][0]["adapterDesignBlockers"]


def test_runtime_adapter_design_blocks_invalid_locator() -> None:
    row = copy.deepcopy(_contract_row(1))
    row["spanLocator"] = "page:1"

    report = _build(_contract_report([row]))

    assert report["status"] == "blocked"
    assert report["counts"]["blockedInvalidLocatorRows"] == 1
    assert "chars_locator_missing_or_invalid" in report["rows"][0]["adapterDesignBlockers"]


def test_runtime_adapter_design_blocks_policy_flag_violation() -> None:
    row = copy.deepcopy(_contract_row(1))
    row["answerVisible"] = True

    report = _build(_contract_report([row]))

    assert report["status"] == "blocked"
    assert report["counts"]["blockedPolicyQuarantineRows"] == 1
    assert "answerVisible_not_false" in report["rows"][0]["adapterDesignBlockers"]


def test_runtime_adapter_design_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = _build(_contract_report())
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_parsed_artifact_evidence_chunk_candidate_runtime_adapter_design(
        report,
        report_json=json_path,
        report_md=md_path,
    )

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["sourceRuntimeContractReview"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
