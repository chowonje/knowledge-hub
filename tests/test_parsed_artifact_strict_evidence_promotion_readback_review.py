from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_store_contract import (
    PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_SOURCE_SPAN_STORE,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_executor_apply import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_APPLY_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_promotion_readback_review import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
    READBACK_STATUS_BLOCKED_MISSING_SOURCE_SPAN,
    READBACK_STATUS_BLOCKED_RUNTIME_OR_CITATION,
    READBACK_STATUS_BLOCKED_SOURCE_HASH_MISMATCH,
    READBACK_STATUS_VALIDATED,
    build_parsed_artifact_strict_evidence_promotion_readback_review,
    write_parsed_artifact_strict_evidence_promotion_readback_review_reports,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_record_contract import (
    build_sample_strict_evidence_record_from_packet_row,
)


def _source_span_record(
    *,
    paper_id: str = "paper-1",
    index: int = 1,
    source_content_hash: str = "hash-1",
    run_id: str = "run-1",
) -> dict:
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID,
        "sourceSpanId": f"source-span:{paper_id}:section:{index}",
        "candidateRecordId": f"source-span-candidate:{paper_id}:section:{index}",
        "runId": run_id,
        "plannedWriteTarget": PARSED_ARTIFACT_SOURCE_SPAN_STORE,
        "paperId": paper_id,
        "artifactType": "section",
        "sourceCandidateId": f"candidate-{index}",
        "sourceContentHash": source_content_hash,
        "sourceFile": f"{paper_id}.pdf",
        "locator": {"page": 1, "bbox": [0, 0, 1, 1]},
        "idempotencyKey": f"span-idem-{index}",
        "evidenceTier": "parsed_artifact_source_span",
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "strictBlockers": [],
        "writePolicy": {
            "executorRequired": True,
            "databaseMutation": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
    }


def _strict_evidence_record(
    *,
    paper_id: str = "paper-1",
    index: int = 1,
    source_content_hash: str = "hash-1",
    run_id: str = "run-1",
) -> dict:
    packet_row = {
        "packet_review_row_id": f"packet:{index:04d}",
        "paper_id": paper_id,
        "artifact_type": "section",
        "sourceSpanId": f"source-span:{paper_id}:section:{index}",
        "candidateRecordId": f"source-span-candidate:{paper_id}:section:{index}",
        "sourceContentHash": source_content_hash,
        "source_file": f"{paper_id}.pdf",
        "text_surface": "Introduction",
        "proposed_chars": {
            "start": 0,
            "end": 12,
            "basis": "sourceContentHash",
            "normalization": "nfkc_whitespace_casefold_v1",
            "expectedSubstringSha256": "abc123",
        },
    }
    record = build_sample_strict_evidence_record_from_packet_row(packet_row, run_id=run_id)
    record["idempotencyKey"] = f"strict-idem-{index}"
    return record


def _apply_report(*, run_id: str = "run-1", input_rows: int = 2) -> dict:
    return {
        "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_APPLY_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "executorDryRunReportPath": "/tmp/dry-run.json",
            "executorDryRunSchema": "knowledge-hub.paper.parsed-artifact-strict-evidence-executor-dry-run.v1",
            "executorDryRunStatus": "ok",
            "contractReportPath": "/tmp/contract.json",
            "contractSchema": "knowledge-hub.paper.parsed-artifact-strict-evidence-record-contract.v1",
            "contractStatus": "ok",
            "requestedPaperIds": [],
            "papersDir": "/tmp/papers",
            "runId": run_id,
            "apply": True,
        },
        "counts": {
            "inputRows": input_rows,
            "plannedApplyRows": input_rows,
            "heldInputRows": 0,
            "strictEvidenceRecordRows": input_rows,
            "strictEvidenceWriteRows": input_rows,
            "readbackValidatedRows": input_rows,
            "runManifestWriteRows": 1,
            "blockedNonReadyInputRows": 0,
            "blockedStoreContractRows": 0,
            "blockedSchemaViolationRows": 0,
            "strictEvidenceCreatedRows": input_rows,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "sourceSpanUpdatedRows": 0,
            "schemaViolationCount": 0,
            "byPaperId": {},
            "byArtifactType": {},
            "byApplyStatus": {},
        },
        "gate": {
            "readyForDryRun": False,
            "readyForApply": True,
            "applyMode": True,
            "strictEvidenceStoreWriteAllowed": True,
            "rollbackImplemented": False,
            "rollbackRequiresExplicitRunId": True,
            "strictEvidenceReady": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "parsed_artifact_strict_evidence_executor_applied",
            "recommendedNextTranche": "parsed_artifact_strict_evidence_promotion_readback_review",
        },
        "policy": {
            "dryRunByDefault": True,
            "applyRequiredForStrictEvidenceStoreWrites": True,
            "strictEvidenceStoreWrite": True,
            "sourceSpanStoreWrite": False,
            "designPacketReviewReportMutated": False,
            "normalizationHashRepairReportMutated": False,
            "strictEvidenceCreated": True,
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
        "rows": [],
        "strictEvidenceRecords": [],
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_readback_review_validates_linked_records(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    run_id = "run-1"
    _write_jsonl(
        papers_dir / "structured_evidence" / "source_span" / "paper-1.jsonl",
        [_source_span_record(index=1), _source_span_record(index=2)],
    )
    _write_jsonl(
        papers_dir / "structured_evidence" / "strict_evidence" / "paper-1.jsonl",
        [_strict_evidence_record(index=1, run_id=run_id), _strict_evidence_record(index=2, run_id=run_id)],
    )

    apply_path = tmp_path / "apply.json"
    apply_path.write_text(json.dumps(_apply_report(run_id=run_id, input_rows=2)), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input": {"runId": run_id},
                "strictEvidenceRecords": [
                    _strict_evidence_record(index=1, run_id=run_id),
                    _strict_evidence_record(index=2, run_id=run_id),
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_parsed_artifact_strict_evidence_promotion_readback_review(
        apply_report_path=apply_path,
        run_manifest_path=manifest_path,
        papers_dir=papers_dir,
        run_id=run_id,
    )

    assert report["status"] == "ok"
    assert report["counts"]["readbackValidatedRows"] == 2
    assert report["counts"]["strictEvidenceCreatedRows"] == 0
    assert all(row["source_span_reference_found"] for row in report["rows"])
    assert {row["readback_status"] for row in report["rows"]} == {READBACK_STATUS_VALIDATED}


def test_readback_review_blocks_missing_source_span_and_hash_mismatch(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    run_id = "run-1"
    missing_span = _strict_evidence_record(index=1, run_id=run_id)
    hash_mismatch = _strict_evidence_record(index=2, run_id=run_id)
    hash_mismatch["sourceContentHash"] = "different-hash"

    _write_jsonl(
        papers_dir / "structured_evidence" / "strict_evidence" / "paper-1.jsonl",
        [missing_span, hash_mismatch],
    )
    _write_jsonl(
        papers_dir / "structured_evidence" / "source_span" / "paper-1.jsonl",
        [_source_span_record(index=2, source_content_hash="hash-1")],
    )

    apply_path = tmp_path / "apply.json"
    apply_path.write_text(json.dumps(_apply_report(run_id=run_id, input_rows=2)), encoding="utf-8")

    report = build_parsed_artifact_strict_evidence_promotion_readback_review(
        apply_report_path=apply_path,
        papers_dir=papers_dir,
        run_id=run_id,
    )

    assert report["status"] == "blocked"
    statuses = {row["readback_status"] for row in report["rows"]}
    assert READBACK_STATUS_BLOCKED_MISSING_SOURCE_SPAN in statuses
    assert READBACK_STATUS_BLOCKED_SOURCE_HASH_MISMATCH in statuses


def test_readback_review_blocks_runtime_flags(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    run_id = "run-1"
    record = _strict_evidence_record(index=1, run_id=run_id)
    record["citationGrade"] = True
    _write_jsonl(
        papers_dir / "structured_evidence" / "strict_evidence" / "paper-1.jsonl",
        [record],
    )
    _write_jsonl(
        papers_dir / "structured_evidence" / "source_span" / "paper-1.jsonl",
        [_source_span_record(index=1)],
    )
    apply_path = tmp_path / "apply.json"
    apply_path.write_text(json.dumps(_apply_report(run_id=run_id, input_rows=1)), encoding="utf-8")

    report = build_parsed_artifact_strict_evidence_promotion_readback_review(
        apply_report_path=apply_path,
        papers_dir=papers_dir,
        run_id=run_id,
    )

    assert report["rows"][0]["readback_status"] == READBACK_STATUS_BLOCKED_RUNTIME_OR_CITATION


def test_readback_writer_outputs_schema_valid_report(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    run_id = "run-1"
    _write_jsonl(
        papers_dir / "structured_evidence" / "source_span" / "paper-1.jsonl",
        [_source_span_record(index=1)],
    )
    _write_jsonl(
        papers_dir / "structured_evidence" / "strict_evidence" / "paper-1.jsonl",
        [_strict_evidence_record(index=1, run_id=run_id)],
    )
    apply_path = tmp_path / "apply.json"
    apply_path.write_text(json.dumps(_apply_report(run_id=run_id, input_rows=1)), encoding="utf-8")

    report = build_parsed_artifact_strict_evidence_promotion_readback_review(
        apply_report_path=apply_path,
        papers_dir=papers_dir,
        run_id=run_id,
    )
    paths = write_parsed_artifact_strict_evidence_promotion_readback_review_reports(
        report,
        tmp_path / "reports",
    )
    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert validate_payload(
        written,
        PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
