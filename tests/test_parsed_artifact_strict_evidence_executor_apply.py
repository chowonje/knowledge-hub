from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_original_source_offset_authority_design import (
    CHARS_BASIS,
    CHARS_NORMALIZATION,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_executor_apply import (
    APPLY_STATUS_APPLIED,
    APPLY_STATUS_PLANNED,
    PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_APPLY_SCHEMA_ID,
    execute_parsed_artifact_strict_evidence_executor_apply,
    write_parsed_artifact_strict_evidence_executor_apply_reports,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_executor_dry_run import (
    DRY_RUN_STATUS_READY,
    PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_DRY_RUN_SCHEMA_ID,
    RECOMMENDED_ACTION_READY,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_record_contract import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID,
    PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_STRICT_EVIDENCE_STORE,
    build_parsed_artifact_strict_evidence_record_contract,
    build_sample_strict_evidence_record_from_packet_row,
)


def _packet_row(*, expected_hash: str = "abc123") -> dict:
    return {
        "packet_review_row_id": "packet:0001",
        "paper_id": "paper-1",
        "artifact_type": "section",
        "sourceSpanId": "source-span:paper-1:section:1",
        "candidateRecordId": "source-span-candidate:paper-1:section:1",
        "sourceContentHash": "hash-paper-1",
        "source_file": "/tmp/paper-1.pdf",
        "text_surface": "Introduction",
        "proposed_chars": {
            "start": 0,
            "end": 12,
            "basis": CHARS_BASIS,
            "normalization": CHARS_NORMALIZATION,
            "expectedSubstringSha256": expected_hash,
        },
    }


def _dry_run_report(*, planned_record: dict) -> dict:
    return {
        "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "designPacketReviewReportPath": "/tmp/packet.json",
            "designPacketReviewSchema": "knowledge-hub.paper.parsed-artifact-source-span-strict-evidence-design-packet-review.v1",
            "contractReportPath": "/tmp/contract.json",
            "contractSchema": PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID,
            "papersDir": "/tmp/papers",
            "parsedRoot": "/tmp/papers/parsed",
            "runId": "test-dry-run",
            "requestedPaperIds": [],
        },
        "counts": {
            "inputRows": 1,
            "packetReadyRows": 1,
            "plannedStrictEvidenceRows": 1,
            "dryRunReadyStrictEvidenceRecordOnlyRows": 1,
            "blockedSourceTextUnavailableRows": 0,
            "blockedNormalizationHashContractMismatchRows": 0,
            "blockedPlannedRecordSchemaViolationRows": 0,
            "blockedPlannedRecordSemanticViolationRows": 0,
            "blockedInputSchemaViolationRows": 0,
            "strictEvidenceWriteRows": 0,
            "strictEvidenceCreatedRows": 0,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "schemaViolationCount": 0,
            "byPaperId": {},
            "byArtifactType": {},
            "byDryRunStatus": {DRY_RUN_STATUS_READY: 1},
            "byRecommendedAction": {},
        },
        "gate": {
            "readyForStrictEvidenceExecutorApply": True,
            "strictEvidenceCreated": False,
            "strictEvidenceReady": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "parsed_artifact_strict_evidence_executor_dry_run_ready",
            "recommendedNextTranche": "parsed_artifact_strict_evidence_executor_apply",
        },
        "policy": {
            "reportOnly": True,
            "executorImplemented": False,
            "strictEvidenceStoreWrite": False,
            "sourceSpanStoreWrite": False,
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
        "warnings": [],
        "rows": [
            {
                "dry_run_row_id": "parsed-artifact-strict-evidence-executor-dry-run:0000",
                "packet_review_row_id": "packet:0001",
                "paper_id": "paper-1",
                "artifact_type": "section",
                "sourceSpanId": "source-span:paper-1:section:1",
                "candidateRecordId": "source-span-candidate:paper-1:section:1",
                "dry_run_status": DRY_RUN_STATUS_READY,
                "dry_run_blockers": ["strict_evidence_executor_dry_run_only"],
                "recommended_action": RECOMMENDED_ACTION_READY,
                "hashVerification": {},
                "plannedStrictEvidenceRecord": planned_record,
                "writeMatrix": {
                    "plannedWriteTarget": PARSED_ARTIFACT_STRICT_EVIDENCE_STORE,
                    "writeEnabled": False,
                    "strictEvidenceStoreWrite": False,
                    "sourceSpanStoreWrite": False,
                },
                "strictEvidenceWriteRows": 0,
                "strictEvidenceCreated": False,
                "runtimeEvidenceCreated": False,
                "sourceSpanUpdatedRows": 0,
            }
        ],
    }


def test_apply_dry_run_plans_without_writes(tmp_path: Path) -> None:
    planned = build_sample_strict_evidence_record_from_packet_row(_packet_row())
    dry_run_path = tmp_path / "dry-run.json"
    dry_run_path.write_text(json.dumps(_dry_run_report(planned_record=planned)), encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(build_parsed_artifact_strict_evidence_record_contract()),
        encoding="utf-8",
    )

    report = execute_parsed_artifact_strict_evidence_executor_apply(
        executor_dry_run_report=dry_run_path,
        contract_report=contract_path,
        apply=False,
    )

    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 1
    assert report["counts"]["plannedApplyRows"] == 1
    assert report["counts"]["strictEvidenceWriteRows"] == 0
    assert report["counts"]["strictEvidenceCreatedRows"] == 0
    assert report["rows"][0]["apply_status"] == APPLY_STATUS_PLANNED
    assert report["rows"][0]["would_write_strict_evidence_record"] is True


def test_apply_writes_strict_evidence_jsonl(tmp_path: Path) -> None:
    planned = build_sample_strict_evidence_record_from_packet_row(_packet_row())
    dry_run_path = tmp_path / "dry-run.json"
    dry_run_path.write_text(json.dumps(_dry_run_report(planned_record=planned)), encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(build_parsed_artifact_strict_evidence_record_contract()),
        encoding="utf-8",
    )
    papers_dir = tmp_path / "papers"

    report = execute_parsed_artifact_strict_evidence_executor_apply(
        executor_dry_run_report=dry_run_path,
        contract_report=contract_path,
        papers_dir=papers_dir,
        run_id="test-apply-run",
        apply=True,
    )

    assert report["status"] == "ok"
    assert report["counts"]["strictEvidenceWriteRows"] == 1
    assert report["counts"]["strictEvidenceCreatedRows"] == 1
    assert report["counts"]["readbackValidatedRows"] == 1
    assert report["rows"][0]["apply_status"] == APPLY_STATUS_APPLIED

    jsonl_path = papers_dir / "structured_evidence" / "strict_evidence" / "paper-1.jsonl"
    assert jsonl_path.is_file()
    stored = json.loads(jsonl_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert validate_payload(stored, PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID, strict=True).ok
    assert stored["runId"] == "test-apply-run"
    assert stored["citationGrade"] is False
    assert stored["runtimeEvidence"] is False


def test_apply_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    planned = build_sample_strict_evidence_record_from_packet_row(_packet_row())
    dry_run_path = tmp_path / "dry-run.json"
    dry_run_path.write_text(json.dumps(_dry_run_report(planned_record=planned)), encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(build_parsed_artifact_strict_evidence_record_contract()),
        encoding="utf-8",
    )

    report = execute_parsed_artifact_strict_evidence_executor_apply(
        executor_dry_run_report=dry_run_path,
        contract_report=contract_path,
        apply=False,
    )
    paths = write_parsed_artifact_strict_evidence_executor_apply_reports(
        report,
        tmp_path / "reports",
    )
    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert written["schema"] == PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_APPLY_SCHEMA_ID
    assert validate_payload(
        written,
        PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_APPLY_SCHEMA_ID,
        strict=True,
    ).ok
