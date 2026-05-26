from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_strict_evidence_record_contract import (
    CHARS_BASIS,
    CHARS_NORMALIZATION_LABEL,
    KNOWN_WRITE_TARGET_CONTRACTS,
    PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID,
    PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_STRICT_EVIDENCE_STORE,
    STRICT_EVIDENCE_STORE_CONTRACT,
    build_parsed_artifact_strict_evidence_record_contract,
    build_sample_strict_evidence_record_from_packet_row,
    validate_strict_evidence_record_semantics,
    write_parsed_artifact_strict_evidence_record_contract_reports,
)


def _packet_row() -> dict:
    return {
        "packet_review_row_id": "parsed-artifact-source-span-strict-evidence-design-packet-review:0000",
        "reconciliation_row_id": "reconciliation:ready:0000",
        "source": "original_design_review",
        "review_row_id": "review:0000",
        "design_row_id": "offset-design:0000",
        "sourceSpanId": "source-span:paper-1:section:abc",
        "candidateRecordId": "source-span-candidate:paper-1:section:abc",
        "paper_id": "paper-1",
        "artifact_type": "section",
        "sourceContentHash": "hash-paper-1",
        "source_file": "",
        "text_surface": "Introduction",
        "proposed_chars": {
            "start": 10,
            "end": 22,
            "basis": CHARS_BASIS,
            "normalization": CHARS_NORMALIZATION_LABEL,
            "expectedSubstringSha256": "abc123expectedhashvalue00000000000000000000000000000000000001",
            "sourceContentHash": "hash-paper-1",
        },
    }


def test_strict_evidence_record_contract_declares_store_target() -> None:
    payload = build_parsed_artifact_strict_evidence_record_contract()

    assert payload["schema"] == PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID
    assert payload["status"] == "ok"
    assert payload["counts"]["writeTargetContracts"] == 1
    assert payload["counts"]["strictEvidenceStoreContracts"] == 1
    assert payload["counts"]["strictEvidenceRecordSchemas"] == 1
    assert payload["counts"]["executorImplementedRows"] == 0
    assert payload["counts"]["strictEvidenceCreatedRows"] == 0
    assert payload["counts"]["databaseMutationRows"] == 0
    assert payload["gate"]["executorReady"] is False
    assert payload["gate"]["recommendedNextTranche"] == "parsed_artifact_strict_evidence_executor_dry_run"
    assert payload["policy"]["contractOnly"] is True
    assert payload["policy"]["sourceSpanStoreWrite"] is False

    write_target = payload["writeTargets"][0]
    assert write_target["plannedWriteTarget"] == PARSED_ARTIFACT_STRICT_EVIDENCE_STORE
    assert write_target["strictEvidenceRecordSchema"] == PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID
    assert write_target["recordPathTemplate"] == (
        "{papers_dir}/structured_evidence/strict_evidence/{paper_id}.jsonl"
    )
    assert write_target["rollbackImplemented"] is False
    assert write_target["sourceSpanMutationAllowed"] is False
    assert "verbatimSubstringSha256_equals_authority_chars_expectedSubstringSha256" in (
        write_target["readbackChecks"]
    )
    assert (
        write_target["normalizationHashContract"]["executorBlockerOnMismatch"]
        == "blocked_normalization_hash_contract_mismatch"
    )
    assert KNOWN_WRITE_TARGET_CONTRACTS[PARSED_ARTIFACT_STRICT_EVIDENCE_STORE] == (
        PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID
    )
    assert validate_payload(
        payload,
        PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok


def test_strict_evidence_record_schema_requires_hash_equality_fields() -> None:
    record = build_sample_strict_evidence_record_from_packet_row(_packet_row())
    expected_hash = record["authority"]["chars"]["expectedSubstringSha256"]

    assert record["verbatimSubstringSha256"] == expected_hash
    assert record["plannedWriteTarget"] == PARSED_ARTIFACT_STRICT_EVIDENCE_STORE
    assert record["evidenceTier"] == "parsed_artifact_strict_evidence"
    assert record["authority"]["type"] == "text_offset"
    assert record["authority"]["chars"]["basis"] == CHARS_BASIS
    assert record["authority"]["chars"]["normalization"] == CHARS_NORMALIZATION_LABEL
    assert len(record["sourceSpanIds"]) >= 1
    assert len(record["candidateRecordIds"]) >= 1
    assert record["strictEligible"] is False
    assert record["citationGrade"] is False
    assert record["runtimeEvidence"] is False
    assert record["writePolicy"]["parserRoutingChanged"] is False
    assert record["writePolicy"]["answerIntegrationChanged"] is False
    assert record["writePolicy"]["databaseMutation"] is False

    assert validate_payload(record, PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID, strict=True).ok
    assert validate_strict_evidence_record_semantics(record) == []


def test_strict_evidence_record_schema_rejects_hash_mismatch() -> None:
    record = build_sample_strict_evidence_record_from_packet_row(_packet_row())
    record["verbatimSubstringSha256"] = "deadbeef"

    assert validate_payload(record, PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID, strict=True).ok
    assert "verbatimSubstringSha256_must_equal_authority_chars_expectedSubstringSha256" in (
        validate_strict_evidence_record_semantics(record)
    )


def test_contract_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    payload = build_parsed_artifact_strict_evidence_record_contract()
    paths = write_parsed_artifact_strict_evidence_record_contract_reports(
        payload,
        tmp_path / "reports",
    )

    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert report["schema"] == PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID
    assert summary["counts"]["strictEvidenceRecordSchemas"] == 1
    assert "Normalization / hash contract" in markdown
    assert validate_payload(
        report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID,
        strict=True,
    ).ok
