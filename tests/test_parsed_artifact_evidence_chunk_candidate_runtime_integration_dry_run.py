from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_answerability_policy_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_ANSWERABILITY_POLICY_GATE_SCHEMA_ID,
    POLICY_STATUS_READY_CANDIDATE_ONLY,
    READY_DECISION as ANSWERABILITY_POLICY_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_INTEGRATION_DRY_RUN_SCHEMA_ID,
    READY_DECISION,
    RUNTIME_STATUS_READY_CANDIDATE_ONLY,
    build_parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run,
    write_parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run,
)


ANSWERABILITY_POLICY_GATE_REF = (
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_answerability_policy_gate.v1.json"
)


def _policy_row(index: int, *, artifact_type: str = "paragraph") -> dict[str, object]:
    start = index * 100
    return {
        "policyGateRowId": f"parsed-artifact-evidence-chunk-candidate-answerability-policy-gate:{index:04d}",
        "sourceReadbackReviewRowId": f"parsed-artifact-evidence-chunk-candidate-full-apply-readback-review:{index:04d}",
        "candidateRecordId": f"parsed-artifact-evidence-chunk-candidate:paper-{index}:paragraph:key",
        "sourceCandidateRowId": f"parsed-artifact-evidence-chunk:paper-{index}:{index}",
        "paperId": f"paper-{index}",
        "artifactType": artifact_type,
        "sourceRef": f"papers_dir/parsed/paper-{index}/document.md",
        "sourceContentHash": "sha256:" + f"{index:064d}"[-64:],
        "spanLocator": f"chars:{start}-{start + 80}",
        "snippetHash": "sha256:" + f"{index + 1:064d}"[-64:],
        "candidateStoreRef": f"papers_dir/structured_evidence_candidates/evidence_chunk/paper-{index}.jsonl",
        "answerabilityPolicyStatus": POLICY_STATUS_READY_CANDIDATE_ONLY,
        "answerabilityPolicyBlockers": [],
        "answerEvidenceCandidate": True,
        "answerabilityCandidate": True,
        "answerabilityPolicyReady": True,
        "runtimeIntegrationDryRunReady": True,
        "answerEvidenceEligible": True,
        "answerabilityEligible": True,
        "answerable": False,
        "candidateOnly": True,
        "strictEvidence": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "answerVisible": False,
        "recommendedAction": "queue_for_candidate_runtime_integration_dry_run",
        "checks": {
            "readbackValidated": True,
            "artifactTypeAllowed": True,
            "sourceContentHashPresent": True,
            "snippetHashPresent": True,
            "charsLocatorPresent": True,
            "candidatePolicyQuarantined": True,
            "answerableStillFalse": True,
            "runtimeVisibilityStillFalse": True,
        },
    }


def _policy_report(rows: list[dict[str, object]] | None = None, *, status: str = "ready") -> dict[str, object]:
    policy_rows = rows or [_policy_row(1), _policy_row(2, artifact_type="section")]
    ready = status == "ready"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_ANSWERABILITY_POLICY_GATE_SCHEMA_ID,
        "status": status,
        "generatedAt": "2026-05-29T00:00:00Z",
        "decision": ANSWERABILITY_POLICY_READY_DECISION if ready else "blocked",
        "nextRecommendedTranche": "parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run",
        "sourceFullApplyReadbackReview": {
            "schema": "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-full-apply-readback-review.v1",
            "status": "ready" if ready else "blocked",
            "decision": "parsed_artifact_evidence_chunk_candidate_full_apply_readback_review_ready" if ready else "blocked",
            "nextRecommendedTranche": "parsed_artifact_evidence_chunk_candidate_answerability_policy_gate",
            "reportRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_full_apply_readback_review.v1.json",
            "expectedCandidateRows": len(policy_rows),
            "readbackValidatedRows": len(policy_rows) if ready else 0,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "policy": {
            "reportOnly": True,
            "answerabilityPolicyGateOnly": True,
            "allowedArtifactTypes": ["paragraph", "section"],
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
            "nextGateRequiredBeforeRuntimeUse": True,
        },
        "counts": {
            "inputRows": len(policy_rows),
            "readbackValidatedInputRows": len(policy_rows) if ready else 0,
            "answerabilityPolicyReadyRows": len(policy_rows) if ready else 0,
            "answerEvidenceEligibleCandidateRows": len(policy_rows) if ready else 0,
            "answerabilityEligibleCandidateRows": len(policy_rows) if ready else 0,
            "runtimeIntegrationDryRunReadyRows": len(policy_rows) if ready else 0,
            "answerableRows": 0,
            "blockedRows": 0,
            "blockedInputSchemaViolationRows": 0,
            "blockedReadbackNotReadyRows": 0,
            "blockedUnsupportedArtifactTypeRows": 0,
            "blockedMissingSourceHashRows": 0,
            "blockedMissingSnippetHashRows": 0,
            "blockedMissingLocatorRows": 0,
            "blockedMissingExcerptRows": 0,
            "blockedPolicyViolationRows": 0,
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
            "byAnswerabilityPolicyStatus": {POLICY_STATUS_READY_CANDIDATE_ONLY: len(policy_rows)},
        },
        "gate": {
            "readyForRuntimeIntegrationDryRun": ready,
            "candidateStoreWriteAllowed": False,
            "sourceSpanCreationAllowed": False,
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "answerVisibleAllowed": False,
            "answerableAllowed": False,
            "schemaViolations": [],
        },
        "rows": policy_rows,
        "warnings": [],
    }


def _build(policy_report: dict[str, object]) -> dict[str, object]:
    return build_parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run(
        answerability_policy_gate_report=policy_report,
        source_answerability_policy_gate_report_ref=ANSWERABILITY_POLICY_GATE_REF,
        generated_at="2026-05-29T00:00:00Z",
    )


def test_runtime_integration_dry_run_projects_ready_rows_to_evidence_packet_preview_only() -> None:
    report = _build(_policy_report())

    assert report["schema"] == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_INTEGRATION_DRY_RUN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["inputRows"] == 2
    assert report["counts"]["runtimeIntegrationDryRunReadyRows"] == 2
    assert report["counts"]["plannedEvidencePacketSpanRows"] == 2
    assert report["counts"]["runtimeEvidenceRows"] == 0
    assert report["counts"]["answerVisibleRows"] == 0
    assert report["counts"]["answerableRows"] == 0
    assert {row["runtimeIntegrationStatus"] for row in report["rows"]} == {
        RUNTIME_STATUS_READY_CANDIDATE_ONLY
    }
    span = report["rows"][0]["plannedEvidencePacketSpan"]
    assert span["source_type"] == "paper"
    assert span["source_id"] == "paper-1"
    assert span["source_content_hash"].startswith("sha256:")
    assert span["span_locator"] == "chars:100-180"
    assert span["charStart"] == 100
    assert span["charEnd"] == 180
    assert span["spanOffsetAvailable"] is True
    assert span["textPreviewIncluded"] is False
    assert span["runtimeEvidence"] is False
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_INTEGRATION_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_integration_dry_run_blocks_non_ready_policy_gate() -> None:
    report = _build(_policy_report(status="blocked"))

    assert report["status"] == "blocked"
    assert "answerability_policy_gate_not_ready" in report["gate"]["schemaViolations"]
    assert report["counts"]["runtimeIntegrationDryRunReadyRows"] == 0
    assert report["counts"]["blockedInputSchemaViolationRows"] == 2


def test_runtime_integration_dry_run_blocks_unsupported_artifact_type() -> None:
    report = _build(_policy_report([_policy_row(1, artifact_type="table")]))

    assert report["status"] == "blocked"
    assert report["counts"]["blockedUnsupportedArtifactTypeRows"] == 1
    assert "unsupported_artifact_type" in report["rows"][0]["runtimeIntegrationBlockers"]


def test_runtime_integration_dry_run_blocks_missing_required_fields_and_invalid_locator() -> None:
    row = copy.deepcopy(_policy_row(1))
    row["sourceContentHash"] = ""
    row["spanLocator"] = "page:1"

    report = _build(_policy_report([row]))

    assert report["status"] == "blocked"
    assert report["counts"]["blockedInvalidLocatorRows"] == 1
    assert "sourceContentHash_missing" in report["rows"][0]["runtimeIntegrationBlockers"]
    assert "chars_locator_missing_or_invalid" in report["rows"][0]["runtimeIntegrationBlockers"]


def test_runtime_integration_dry_run_blocks_policy_quarantine_violation() -> None:
    row = copy.deepcopy(_policy_row(1))
    row["answerVisible"] = True

    report = _build(_policy_report([row]))

    assert report["status"] == "blocked"
    assert report["counts"]["blockedPolicyQuarantineRows"] == 1
    assert "answer_visible_not_false" in report["rows"][0]["runtimeIntegrationBlockers"]


def test_runtime_integration_dry_run_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = _build(_policy_report())
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_parsed_artifact_evidence_chunk_candidate_runtime_integration_dry_run(
        report,
        report_json=json_path,
        report_md=md_path,
    )

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["sourceAnswerabilityPolicyGate"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
