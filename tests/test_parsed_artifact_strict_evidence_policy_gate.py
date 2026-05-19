from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_store_contract import (
    PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_SOURCE_SPAN_STORE,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_policy_gate import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
    POLICY_STATUS_BLOCKED_MISSING_VERBATIM_HASH,
    POLICY_STATUS_BLOCKED_RUNTIME_OR_CITATION,
    POLICY_STATUS_CANDIDATE_ONLY,
    build_parsed_artifact_strict_evidence_policy_gate,
    write_parsed_artifact_strict_evidence_policy_gate_reports,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_executor_apply import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_APPLY_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_promotion_readback_review import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
    READBACK_STATUS_VALIDATED,
    build_parsed_artifact_strict_evidence_promotion_readback_review,
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


def _write_readback_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ready_readback_report(tmp_path: Path) -> dict:
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
    return build_parsed_artifact_strict_evidence_promotion_readback_review(
        apply_report_path=apply_path,
        run_manifest_path=manifest_path,
        papers_dir=papers_dir,
        run_id=run_id,
    )


def test_strict_evidence_policy_gate_marks_readback_validated_rows(tmp_path: Path) -> None:
    readback_report_path = tmp_path / "readback.json"
    _write_readback_report(readback_report_path, _ready_readback_report(tmp_path))

    report = build_parsed_artifact_strict_evidence_policy_gate(
        readback_report_path=readback_report_path,
    )

    assert report["schema"] == PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID
    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 2
    assert report["counts"]["strictEvidencePolicyCandidateOnlyRows"] == 2
    assert report["counts"]["strictEvidenceCreatedRows"] == 0
    assert report["gate"]["strictEvidencePolicyGateReady"] is True
    assert report["gate"]["strictEligibleMutationAllowed"] is False
    assert {row["policy_gate_status"] for row in report["rows"]} == {POLICY_STATUS_CANDIDATE_ONLY}
    assert validate_payload(
        report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_strict_evidence_policy_gate_blocks_source_span_and_readback_status(tmp_path: Path) -> None:
    readback_report = _ready_readback_report(tmp_path)
    readback_report["rows"][0]["source_span_reference_hash_match"] = False
    readback_report["rows"][1]["readback_status"] = "blocked_readback_not_validated"
    readback_report_path = tmp_path / "readback.json"
    _write_readback_report(readback_report_path, readback_report)

    report = build_parsed_artifact_strict_evidence_policy_gate(
        readback_report_path=readback_report_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["blockedMissingSourceSpanReferenceRows"] == 1
    assert report["counts"]["blockedReadbackNotValidatedRows"] == 1
    assert report["counts"]["strictEvidencePolicyCandidateOnlyRows"] == 0
    assert validate_payload(
        report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_strict_evidence_policy_gate_blocks_invalid_input_schema(tmp_path: Path) -> None:
    readback_report = _ready_readback_report(tmp_path)
    readback_report["schema"] = "wrong.schema"
    readback_report_path = tmp_path / "readback.json"
    _write_readback_report(readback_report_path, readback_report)

    report = build_parsed_artifact_strict_evidence_policy_gate(
        readback_report_path=readback_report_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0
    assert report["counts"]["strictEvidencePolicyCandidateOnlyRows"] == 0
    assert report["gate"]["strictEvidencePolicyGateReady"] is False
    assert validate_payload(
        report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def _synthetic_readback_row(
    *,
    store_path: Path,
    store_line: int = 1,
    index: int = 1,
) -> dict:
    return {
        "review_row_id": f"parsed-artifact-strict-evidence-promotion-readback-review:{index:04d}",
        "strictEvidenceId": f"strict-evidence:paper-1:section:{index}",
        "sourceSpanId": f"source-span:paper-1:section:{index}",
        "candidateRecordId": f"source-span-candidate:paper-1:section:{index}",
        "runId": "run-1",
        "paper_id": "paper-1",
        "artifact_type": "section",
        "sourceContentHash": "hash-1",
        "idempotencyKey": f"strict-idem-{index}",
        "strict_evidence_store_path": str(store_path),
        "strict_evidence_store_line": store_line,
        "source_span_reference_found": True,
        "source_span_reference_hash_match": True,
        "readback_status": READBACK_STATUS_VALIDATED,
        "review_blockers": [],
        "readback_validated": True,
        "strictEvidenceCreated": False,
        "strictEvidenceWriteRows": 0,
        "sourceSpanUpdatedRows": 0,
        "citationGrade": False,
        "runtimeEvidence": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "canonicalParsedArtifactsWritten": False,
        "recommended_action": "queue_for_explicit_strict_evidence_policy_gate",
    }


def _synthetic_readback_report(rows: list[dict]) -> dict:
    return {
        "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "applyReportPath": "/tmp/apply.json",
            "applyReportSchema": PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_APPLY_SCHEMA_ID,
            "applyReportStatus": "ok",
            "runManifestPath": "/tmp/manifest.json",
            "papersDir": "/tmp/papers",
            "strictEvidenceStoreRoot": "/tmp/papers/structured_evidence/strict_evidence",
            "sourceSpanStoreRoot": "/tmp/papers/structured_evidence/source_span",
            "requestedRunId": "run-1",
            "requestedPaperIds": [],
            "runIdentity": {},
            "expectedInputRowsFromApplyReport": len(rows),
        },
        "counts": {
            "inputRows": len(rows),
            "strictEvidenceRecordRows": len(rows),
            "readbackValidatedRows": len(rows),
            "blockedRecordSchemaViolationRows": 0,
            "blockedRecordSemanticViolationRows": 0,
            "blockedMissingSourceSpanReferenceRows": 0,
            "blockedSourceHashMismatchRows": 0,
            "blockedIdempotencyDuplicateRows": 0,
            "blockedRuntimeOrCitationFlagViolationRows": 0,
            "blockedInputReportSchemaViolationRows": 0,
            "strictEvidenceCreatedRows": 0,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "sourceSpanUpdatedRows": 0,
            "schemaViolationCount": 0,
            "byPaperId": {"paper-1": len(rows)},
            "byArtifactType": {"section": len(rows)},
            "byReadbackStatus": {READBACK_STATUS_VALIDATED: len(rows)},
            "byRecommendedAction": {"queue_for_explicit_strict_evidence_policy_gate": len(rows)},
        },
        "gate": {
            "readbackReviewReady": True,
            "strictEvidencePolicyGateReady": True,
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "parsed_artifact_strict_evidence_promotion_readback_review_ready",
            "recommendedNextTranche": "parsed_artifact_strict_evidence_policy_gate",
        },
        "policy": {
            "reportOnly": True,
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
        "rows": rows,
    }


def test_strict_evidence_policy_gate_blocks_verbatim_hash_mismatch(tmp_path: Path) -> None:
    store_path = tmp_path / "strict.jsonl"
    record = _strict_evidence_record(index=1)
    record["verbatimSubstringSha256"] = "mismatch"
    _write_jsonl(store_path, [record])
    readback_report = _synthetic_readback_report([_synthetic_readback_row(store_path=store_path)])
    readback_report_path = tmp_path / "readback.json"
    _write_readback_report(readback_report_path, readback_report)

    report = build_parsed_artifact_strict_evidence_policy_gate(
        readback_report_path=readback_report_path,
    )

    assert report["status"] == "blocked"
    assert report["rows"][0]["policy_gate_status"] == POLICY_STATUS_BLOCKED_MISSING_VERBATIM_HASH
    assert validate_payload(
        report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_strict_evidence_policy_gate_blocks_runtime_flags_on_record(tmp_path: Path) -> None:
    store_path = tmp_path / "strict.jsonl"
    record = _strict_evidence_record(index=1)
    record["runtimeEvidence"] = True
    _write_jsonl(store_path, [record])
    readback_report = _synthetic_readback_report([_synthetic_readback_row(store_path=store_path)])
    readback_report_path = tmp_path / "readback.json"
    _write_readback_report(readback_report_path, readback_report)

    report = build_parsed_artifact_strict_evidence_policy_gate(
        readback_report_path=readback_report_path,
    )

    assert report["rows"][0]["policy_gate_status"] == POLICY_STATUS_BLOCKED_RUNTIME_OR_CITATION
    assert validate_payload(
        report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_strict_evidence_policy_gate_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    readback_report_path = tmp_path / "readback.json"
    _write_readback_report(readback_report_path, _ready_readback_report(tmp_path))

    report = build_parsed_artifact_strict_evidence_policy_gate(
        readback_report_path=readback_report_path,
    )
    paths = write_parsed_artifact_strict_evidence_policy_gate_reports(report, tmp_path / "reports")
    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert validate_payload(
        written,
        PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
        strict=True,
    ).ok
