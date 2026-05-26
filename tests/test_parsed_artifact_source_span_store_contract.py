from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_store_contract import (
    KNOWN_WRITE_TARGET_CONTRACTS,
    PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_SOURCE_SPAN_STORE,
    PARSED_ARTIFACT_SOURCE_SPAN_STORE_CONTRACT_SCHEMA_ID,
    build_parsed_artifact_source_span_store_contract,
    write_parsed_artifact_source_span_store_contract_reports,
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


def test_parsed_artifact_source_span_store_contract_declares_source_span_target() -> None:
    payload = build_parsed_artifact_source_span_store_contract()

    assert payload["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_STORE_CONTRACT_SCHEMA_ID
    assert payload["status"] == "ok"
    assert payload["counts"]["writeTargetContracts"] == 1
    assert payload["counts"]["sourceSpanStoreContracts"] == 1
    assert payload["counts"]["recordSchemas"] == 1
    assert payload["counts"]["executorImplementedRows"] == 0
    assert payload["counts"]["sourceSpanCreatedRows"] == 0
    assert payload["counts"]["databaseMutationRows"] == 0
    assert payload["gate"]["executorReady"] is False
    assert payload["gate"]["sourceSpanStoreContractDefined"] is True
    assert payload["gate"]["recommendedNextTranche"] == (
        "parsed_artifact_source_span_promotion_executor_apply"
    )
    assert payload["policy"]["contractOnly"] is True
    assert payload["policy"]["executorImplemented"] is False
    assert payload["policy"]["vaultScan"] is False

    write_target = payload["writeTargets"][0]
    assert write_target["plannedWriteTarget"] == PARSED_ARTIFACT_SOURCE_SPAN_STORE
    assert write_target["sourceSpanRecordSchema"] == PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID
    assert write_target["recordPathTemplate"] == (
        "{papers_dir}/structured_evidence/source_span/{paper_id}.jsonl"
    )
    assert write_target["executorImplemented"] is False
    assert write_target["runtimeUseAllowed"] is False
    assert "candidateRecordId_linked" in write_target["readbackChecks"]
    assert KNOWN_WRITE_TARGET_CONTRACTS[PARSED_ARTIFACT_SOURCE_SPAN_STORE] == (
        PARSED_ARTIFACT_SOURCE_SPAN_STORE_CONTRACT_SCHEMA_ID
    )
    assert validate_payload(
        payload,
        PARSED_ARTIFACT_SOURCE_SPAN_STORE_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok


def test_source_span_record_schema_remains_non_strict_and_non_runtime() -> None:
    record = {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID,
        "sourceSpanId": "source-span:paper-1:section:idem-1",
        "candidateRecordId": "source-span-candidate:paper-1:section:idem-1",
        "runId": "run-1",
        "plannedWriteTarget": PARSED_ARTIFACT_SOURCE_SPAN_STORE,
        "paperId": "paper-1",
        "artifactType": "section",
        "sourceCandidateId": "section-candidate-1",
        "sourceContentHash": "source-hash",
        "sourceFile": "paper-1.pdf",
        "locator": {
            "page": 1,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "blockIndexes": [1, 2],
            "chars": {"start": 10, "end": 20},
        },
        "idempotencyKey": "key-1",
        "evidenceTier": "parsed_artifact_source_span",
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "strictBlockers": ["source_span_record_not_strict_evidence"],
        "writePolicy": _write_policy(),
    }

    assert validate_payload(record, PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID, strict=True).ok


def test_parsed_artifact_source_span_store_contract_writer_outputs_schema_valid_reports(
    tmp_path: Path,
) -> None:
    payload = build_parsed_artifact_source_span_store_contract()
    paths = write_parsed_artifact_source_span_store_contract_reports(
        payload,
        tmp_path / "reports",
    )

    assert set(paths.keys()) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert report["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_STORE_CONTRACT_SCHEMA_ID
    assert summary["counts"]["sourceSpanStoreContracts"] == 1
    assert "Parsed Artifact SourceSpan Store Contract" in markdown
    assert "executor implemented: false" in markdown
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_STORE_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok
    assert validate_payload(
        summary,
        PARSED_ARTIFACT_SOURCE_SPAN_STORE_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok
