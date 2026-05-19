from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_candidate_executor import (
    EXECUTOR_STATUS_APPLIED,
    EXECUTOR_STATUS_DRY_RUN_READY,
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_EXECUTOR_SCHEMA_ID,
    execute_parsed_artifact_source_span_candidate_executor,
    write_parsed_artifact_source_span_candidate_executor_reports,
)
from knowledge_hub.papers.parsed_artifact_source_span_candidate_store_contract import (
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE,
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE_CONTRACT_SCHEMA_ID,
    STRUCTURED_EVIDENCE_CANDIDATE_STORE,
)
from knowledge_hub.papers.parsed_artifact_structured_evidence_execution_plan import (
    EXECUTION_STATUS_BLOCKED_MISSING_LOCATION,
    EXECUTION_STATUS_DRY_RUN_READY,
)
from knowledge_hub.papers.parsed_artifact_structured_evidence_write_target_contract_audit import (
    PARSED_ARTIFACT_STRUCTURED_EVIDENCE_WRITE_TARGET_CONTRACT_AUDIT_SCHEMA_ID,
)


def _ready_row(*, paper_id: str = "paper-1", index: int = 1) -> dict:
    return {
        "plan_id": f"plan-{index}",
        "source_readiness_row_id": f"readiness-{index}",
        "paper_id": paper_id,
        "artifact_type": "section",
        "source_candidate_id": f"candidate-{index}",
        "sourceContentHash": f"source-hash-{index}",
        "source_file": f"{paper_id}.pdf",
        "page": index,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "blockIndexes": [index],
        "planned_operation": "plan_source_span_creation",
        "planned_write_target": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE,
        "would_create_source_span": True,
        "would_create_strict_evidence": False,
        "would_change_runtime": False,
        "would_mutate_database": False,
        "write_target_contract_known": True,
        "write_target_contract_reference": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE_CONTRACT_SCHEMA_ID,
        "rollback_strategy": "discard planned source-span operation",
        "execution_status": EXECUTION_STATUS_DRY_RUN_READY,
        "execution_blockers": [],
        "recommended_action": "queue_for_explicit_source_span_promotion_executor_review",
    }


def _audit_report(rows: list[dict]) -> dict:
    return {
        "schema": PARSED_ARTIFACT_STRUCTURED_EVIDENCE_WRITE_TARGET_CONTRACT_AUDIT_SCHEMA_ID,
        "status": "blocked",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "executionPlanReportPath": "execution-plan.json",
            "executionPlanSchema": "knowledge-hub.paper.parsed-artifact-structured-evidence-execution-plan.v1",
            "executionPlanStatus": "ok",
            "requestedPaperIds": [],
        },
        "counts": {
            "inputRows": len(rows),
            "readyInputRows": sum(
                1 for row in rows if row["execution_status"] == EXECUTION_STATUS_DRY_RUN_READY
            ),
            "plannedRows": len(rows),
            "writeTargetContractKnownRows": sum(
                1 for row in rows if row.get("write_target_contract_known")
            ),
            "blockedMissingSourceHashRows": 0,
            "blockedMissingLocationRows": sum(
                1 for row in rows if row["execution_status"] == EXECUTION_STATUS_BLOCKED_MISSING_LOCATION
            ),
            "blockedUnknownWriteTargetRows": 0,
            "blockedNonReadyInputRows": 0,
            "sourceSpanCreatedRows": 0,
            "strictEvidenceCreatedRows": 0,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "schemaViolationCount": 0,
            "byArtifactType": {},
            "byExecutionStatus": {},
            "byPlannedWriteTarget": {},
            "byRecommendedAction": {},
        },
        "gate": {
            "readyForWriteTargetContractAudit": False,
            "writeTargetContractKnown": True,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "blocked",
            "recommendedNextTranche": "parsed_artifact_structured_evidence_source_span_execution_plan",
        },
        "policy": {
            "reportOnly": True,
            "dryRunOnly": True,
            "sourceSpanCreated": False,
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


def _write_audit_report(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "write-target-audit.json"
    path.write_text(json.dumps(_audit_report(rows), ensure_ascii=False), encoding="utf-8")
    return path


def test_source_span_candidate_executor_dry_run_builds_schema_valid_candidate_records(
    tmp_path: Path,
) -> None:
    blocked_row = {
        **_ready_row(paper_id="paper-2", index=2),
        "planned_write_target": STRUCTURED_EVIDENCE_CANDIDATE_STORE,
        "execution_status": EXECUTION_STATUS_BLOCKED_MISSING_LOCATION,
        "would_create_source_span": False,
        "write_target_contract_known": True,
    }
    audit_path = _write_audit_report(tmp_path, [_ready_row(), blocked_row])

    report = execute_parsed_artifact_source_span_candidate_executor(
        write_target_contract_audit_report=audit_path,
        papers_dir=tmp_path / "papers",
        run_id="run-1",
        apply=False,
    )

    assert report["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_EXECUTOR_SCHEMA_ID
    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 2
    assert report["counts"]["candidateInputRows"] == 1
    assert report["counts"]["heldInputRows"] == 1
    assert report["counts"]["dryRunCandidateRecordRows"] == 1
    assert report["counts"]["appliedCandidateRecordRows"] == 0
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["counts"]["sourceSpanCreatedRows"] == 0
    assert report["counts"]["strictEvidenceCreatedRows"] == 0
    assert report["policy"]["candidateStoreWrite"] is False
    assert report["rows"][0]["execution_status"] == EXECUTOR_STATUS_DRY_RUN_READY
    assert len(report["candidateRecords"]) == 1
    assert not (tmp_path / "papers" / "structured_evidence_candidates").exists()

    assert validate_payload(report, PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_EXECUTOR_SCHEMA_ID, strict=True).ok
    assert validate_payload(
        report["candidateRecords"][0],
        PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID,
        strict=True,
    ).ok


def test_source_span_candidate_executor_apply_writes_idempotent_jsonl_and_manifest(
    tmp_path: Path,
) -> None:
    audit_path = _write_audit_report(
        tmp_path,
        [_ready_row(index=1), _ready_row(index=2)],
    )
    papers_dir = tmp_path / "papers"

    first_report = execute_parsed_artifact_source_span_candidate_executor(
        write_target_contract_audit_report=audit_path,
        papers_dir=papers_dir,
        run_id="run-apply",
        apply=True,
    )
    second_report = execute_parsed_artifact_source_span_candidate_executor(
        write_target_contract_audit_report=audit_path,
        papers_dir=papers_dir,
        run_id="run-apply",
        apply=True,
    )

    record_path = papers_dir / "structured_evidence_candidates" / "source_span" / "paper-1.jsonl"
    manifest_path = papers_dir / "structured_evidence_candidates" / "runs" / "run-apply.json"
    records = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert first_report["status"] == "ok"
    assert first_report["counts"]["appliedCandidateRecordRows"] == 2
    assert first_report["counts"]["candidateStoreWriteRows"] == 2
    assert first_report["counts"]["readbackValidatedRows"] == 2
    assert first_report["counts"]["runManifestWriteRows"] == 1
    assert first_report["policy"]["candidateStoreWrite"] is True
    assert {row["execution_status"] for row in first_report["rows"]} == {EXECUTOR_STATUS_APPLIED}
    assert len(records) == 2
    assert len({record["idempotencyKey"] for record in records}) == 2
    assert second_report["counts"]["readbackValidatedRows"] == 2
    assert len(record_path.read_text(encoding="utf-8").splitlines()) == 2
    assert manifest["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_EXECUTOR_SCHEMA_ID
    assert validate_payload(first_report, PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_EXECUTOR_SCHEMA_ID, strict=True).ok


def test_source_span_candidate_executor_apply_requires_papers_dir(tmp_path: Path) -> None:
    audit_path = _write_audit_report(tmp_path, [_ready_row()])

    report = execute_parsed_artifact_source_span_candidate_executor(
        write_target_contract_audit_report=audit_path,
        run_id="run-1",
        apply=True,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["inputRows"] == 0
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert "apply_requires_papers_dir" in report["warnings"]
    assert "apply_requires_papers_dir" in report["gate"]["schemaViolations"]
    assert validate_payload(report, PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_EXECUTOR_SCHEMA_ID, strict=True).ok


def test_source_span_candidate_executor_report_writer_outputs_schema_valid_reports(
    tmp_path: Path,
) -> None:
    audit_path = _write_audit_report(tmp_path, [_ready_row()])
    report = execute_parsed_artifact_source_span_candidate_executor(
        write_target_contract_audit_report=audit_path,
        papers_dir=tmp_path / "papers",
        run_id="run-1",
        apply=False,
    )

    paths = write_parsed_artifact_source_span_candidate_executor_reports(
        report,
        tmp_path / "reports",
    )

    written_report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    written_summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert written_report["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_EXECUTOR_SCHEMA_ID
    assert written_summary["counts"]["candidateInputRows"] == 1
    assert "Parsed Artifact SourceSpan Candidate Executor" in markdown
    assert "applied candidate records: 0" in markdown
    assert validate_payload(written_report, PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_EXECUTOR_SCHEMA_ID, strict=True).ok
    assert validate_payload(written_summary, PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_EXECUTOR_SCHEMA_ID, strict=True).ok
