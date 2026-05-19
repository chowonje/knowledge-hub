from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_candidate_store_contract import (
    KNOWN_WRITE_TARGET_CONTRACTS,
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE,
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE_CONTRACT_SCHEMA_ID,
    STRUCTURED_EVIDENCE_CANDIDATE_RECORD_SCHEMA_ID,
    STRUCTURED_EVIDENCE_CANDIDATE_STORE,
    build_parsed_artifact_source_span_candidate_store_contract,
    write_parsed_artifact_source_span_candidate_store_contract_reports,
)


def _write_policy() -> dict:
    return {
        "executorRequired": True,
        "databaseMutation": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
    }


def test_parsed_artifact_source_span_candidate_store_contract_declares_known_targets() -> None:
    payload = build_parsed_artifact_source_span_candidate_store_contract()

    assert payload["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE_CONTRACT_SCHEMA_ID
    assert payload["status"] == "ok"
    assert payload["counts"]["writeTargetContracts"] == 2
    assert payload["counts"]["sourceSpanCandidateStoreContracts"] == 1
    assert payload["counts"]["structuredEvidenceCandidateStoreContracts"] == 1
    assert payload["counts"]["executorImplementedRows"] == 0
    assert payload["counts"]["databaseMutationRows"] == 0
    assert payload["gate"]["executorReady"] is False
    assert payload["gate"]["runtimeMutationAllowed"] is False
    assert payload["policy"]["contractOnly"] is True
    assert payload["policy"]["executorImplemented"] is False
    assert payload["policy"]["sourceSpanCreated"] is False
    assert payload["policy"]["canonicalParsedArtifactsWritten"] is False

    by_target = {row["plannedWriteTarget"]: row for row in payload["writeTargets"]}
    assert set(by_target) == {
        PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE,
        STRUCTURED_EVIDENCE_CANDIDATE_STORE,
    }
    assert by_target[PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE]["candidateRecordSchema"] == (
        PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID
    )
    assert by_target[STRUCTURED_EVIDENCE_CANDIDATE_STORE]["candidateRecordSchema"] == (
        STRUCTURED_EVIDENCE_CANDIDATE_RECORD_SCHEMA_ID
    )
    assert all(row["executorImplemented"] is False for row in by_target.values())
    assert all(row["runtimeUseAllowed"] is False for row in by_target.values())
    assert KNOWN_WRITE_TARGET_CONTRACTS[PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE] == (
        PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE_CONTRACT_SCHEMA_ID
    )
    assert validate_payload(payload, PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE_CONTRACT_SCHEMA_ID, strict=True).ok


def test_candidate_record_schemas_remain_candidate_only() -> None:
    source_span_record = {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID,
        "candidateRecordId": "source-span-candidate:paper-1:0001",
        "runId": "run-1",
        "plannedWriteTarget": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE,
        "paperId": "paper-1",
        "artifactType": "section",
        "sourceCandidateId": "section-candidate-1",
        "sourceReadinessRowId": "readiness-1",
        "sourceContentHash": "source-hash",
        "sourceFile": "paper-1.pdf",
        "locator": {
            "page": 1,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "blockIndexes": [1, 2],
            "chars": {"start": 10, "end": 20},
        },
        "idempotencyKey": "key-1",
        "evidenceTier": "source_span_candidate_only",
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "strictBlockers": ["candidate_store_record_not_strict_evidence"],
        "writePolicy": _write_policy(),
    }
    structured_record = {
        "schema": STRUCTURED_EVIDENCE_CANDIDATE_RECORD_SCHEMA_ID,
        "candidateRecordId": "structured-candidate:paper-1:0001",
        "runId": "run-1",
        "plannedWriteTarget": STRUCTURED_EVIDENCE_CANDIDATE_STORE,
        "paperId": "paper-1",
        "artifactType": "equation",
        "sourceCandidateId": "equation-candidate-1",
        "sourceReadinessRowId": "readiness-2",
        "sourceContentHash": "source-hash",
        "sourceFile": "paper-1.pdf",
        "sourceSpanCandidateRecordId": "source-span-candidate:paper-1:0001",
        "locator": {
            "page": 2,
            "bbox": [5.0, 6.0, 7.0, 8.0],
            "blockIndexes": [3],
            "chars": {"start": None, "end": None},
        },
        "idempotencyKey": "key-2",
        "evidenceTier": "structured_evidence_candidate_only",
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "strictBlockers": ["structured_candidate_record_not_strict_evidence"],
        "writePolicy": _write_policy(),
    }

    assert validate_payload(source_span_record, PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID, strict=True).ok
    assert validate_payload(structured_record, STRUCTURED_EVIDENCE_CANDIDATE_RECORD_SCHEMA_ID, strict=True).ok


def test_parsed_artifact_source_span_candidate_store_contract_writer_outputs_schema_valid_reports(
    tmp_path: Path,
) -> None:
    payload = build_parsed_artifact_source_span_candidate_store_contract()
    paths = write_parsed_artifact_source_span_candidate_store_contract_reports(
        payload,
        tmp_path / "reports",
    )

    assert set(paths.keys()) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert report["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE_CONTRACT_SCHEMA_ID
    assert summary["counts"]["writeTargetContracts"] == 2
    assert "Parsed Artifact SourceSpan Candidate Store Contract" in markdown
    assert "executor implemented: false" in markdown
    assert validate_payload(report, PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE_CONTRACT_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE_CONTRACT_SCHEMA_ID, strict=True).ok
