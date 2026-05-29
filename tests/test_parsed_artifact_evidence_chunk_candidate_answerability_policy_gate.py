from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_answerability_policy_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_ANSWERABILITY_POLICY_GATE_SCHEMA_ID,
    POLICY_STATUS_READY_CANDIDATE_ONLY,
    build_parsed_artifact_evidence_chunk_candidate_answerability_policy_gate,
    write_parsed_artifact_evidence_chunk_candidate_answerability_policy_gate,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_full_apply_readback_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READBACK_REVIEW_SCHEMA_ID,
    READY_DECISION as READBACK_REVIEW_READY_DECISION,
)


READBACK_REVIEW_REF = (
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_full_apply_readback_review.v1.json"
)


def _readback_row(index: int, *, artifact_type: str = "paragraph") -> dict[str, object]:
    return {
        "readbackReviewRowId": f"parsed-artifact-evidence-chunk-candidate-full-apply-readback-review:{index:04d}",
        "candidateRecordId": f"parsed-artifact-evidence-chunk-candidate:paper-{index}:paragraph:key",
        "sourceCandidateRowId": f"parsed-artifact-evidence-chunk:paper-{index}:{index}",
        "paperId": f"paper-{index}",
        "artifactType": artifact_type,
        "sourceRef": f"papers_dir/parsed/paper-{index}/document.md",
        "sourceContentHash": "sha256:" + f"{index:064d}"[-64:],
        "spanLocator": f"chars:{index * 100}-{index * 100 + 80}",
        "snippetHash": "sha256:" + f"{index + 1:064d}"[-64:],
        "candidateStoreRef": f"papers_dir/structured_evidence_candidates/evidence_chunk/paper-{index}.jsonl",
        "idempotencyKey": f"parsed-artifact-evidence-chunk-candidate:key:{index}",
        "expectedRecordSha256": "sha256:" + "a" * 64,
        "storedRecordSha256": "sha256:" + "a" * 64,
        "matchingStoreRecordRows": 1,
        "readbackValidated": True,
        "candidateOnly": True,
        "answerEvidenceCandidate": True,
        "answerabilityCandidate": True,
        "strictEvidence": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "answerVisible": False,
        "readbackStatus": "readback_validated_candidate_record",
        "readbackBlockers": [],
        "checks": {
            "storeRecordPresent": True,
            "singleStoreRecordForIdempotencyKey": True,
            "recordHashMatches": True,
            "recordBytesMatchCanonicalJson": True,
            "candidateRecordIdMatches": True,
            "sourceCandidateRowIdMatches": True,
            "paperIdMatches": True,
            "artifactTypeMatches": True,
            "sourceContentHashMatches": True,
            "spanLocatorMatches": True,
            "snippetHashMatches": True,
            "requiredFieldsPresent": True,
            "candidatePolicyQuarantined": True,
            "recordNotPrivatePathLeaking": True,
        },
    }


def _review_report(rows: list[dict[str, object]] | None = None, *, status: str = "ready") -> dict[str, object]:
    review_rows = rows or [_readback_row(1), _readback_row(2, artifact_type="section")]
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READBACK_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": "2026-05-29T00:00:00Z",
        "decision": READBACK_REVIEW_READY_DECISION if status == "ready" else "blocked",
        "nextRecommendedTranche": "parsed_artifact_evidence_chunk_candidate_answerability_policy_gate",
        "counts": {
            "expectedCandidateRows": len(review_rows),
            "readbackValidatedRows": len(review_rows),
            "blockedRows": 0,
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
        },
        "gate": {"readyForAnswerabilityPolicyGate": status == "ready"},
        "rows": review_rows,
    }


def _build(review_report: dict[str, object]) -> dict[str, object]:
    return build_parsed_artifact_evidence_chunk_candidate_answerability_policy_gate(
        full_apply_readback_review_report=review_report,
        source_readback_review_report_ref=READBACK_REVIEW_REF,
        generated_at="2026-05-29T00:00:00Z",
    )


def test_answerability_policy_gate_marks_valid_rows_ready_without_answerability() -> None:
    report = _build(_review_report())

    assert report["schema"] == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_ANSWERABILITY_POLICY_GATE_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == "parsed_artifact_evidence_chunk_candidate_answerability_policy_gate_ready"
    assert report["counts"]["inputRows"] == 2
    assert report["counts"]["answerabilityPolicyReadyRows"] == 2
    assert report["counts"]["runtimeIntegrationDryRunReadyRows"] == 2
    assert report["counts"]["answerableRows"] == 0
    assert report["counts"]["answerVisibleRows"] == 0
    assert {row["answerabilityPolicyStatus"] for row in report["rows"]} == {
        POLICY_STATUS_READY_CANDIDATE_ONLY
    }
    assert all(row["answerable"] is False for row in report["rows"])
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_ANSWERABILITY_POLICY_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_answerability_policy_gate_blocks_non_ready_source_report() -> None:
    report = _build(_review_report(status="blocked"))

    assert report["status"] == "blocked"
    assert "readback_review_not_ready" in report["gate"]["schemaViolations"]
    assert report["counts"]["answerabilityPolicyReadyRows"] == 0
    assert report["counts"]["blockedInputSchemaViolationRows"] == 2


def test_answerability_policy_gate_blocks_unsupported_artifact_type() -> None:
    report = _build(_review_report([_readback_row(1, artifact_type="table")]))

    assert report["status"] == "blocked"
    assert report["counts"]["blockedUnsupportedArtifactTypeRows"] == 1
    assert "unsupported_artifact_type" in report["rows"][0]["answerabilityPolicyBlockers"]


def test_answerability_policy_gate_blocks_missing_hash_and_locator() -> None:
    row = copy.deepcopy(_readback_row(1))
    row["sourceContentHash"] = ""
    row["spanLocator"] = "page:1"

    report = _build(_review_report([row]))

    assert report["status"] == "blocked"
    assert report["counts"]["blockedMissingSourceHashRows"] == 1
    assert "source_content_hash_missing" in report["rows"][0]["answerabilityPolicyBlockers"]
    assert "chars_locator_missing_or_invalid" in report["rows"][0]["answerabilityPolicyBlockers"]


def test_answerability_policy_gate_blocks_policy_quarantine_violation() -> None:
    row = copy.deepcopy(_readback_row(1))
    row["answerVisible"] = True

    report = _build(_review_report([row]))

    assert report["status"] == "blocked"
    assert report["counts"]["blockedPolicyViolationRows"] == 1
    assert "answer_visible_not_false" in report["rows"][0]["answerabilityPolicyBlockers"]


def test_answerability_policy_gate_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = _build(_review_report())
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_parsed_artifact_evidence_chunk_candidate_answerability_policy_gate(
        report,
        report_json=json_path,
        report_md=md_path,
    )

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["sourceFullApplyReadbackReview"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
