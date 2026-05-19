from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_promotion_executor_apply import (
    APPLY_STATUS_APPLIED,
    APPLY_STATUS_PLANNED,
    PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_APPLY_SCHEMA_ID,
    execute_parsed_artifact_source_span_promotion_executor_apply,
    write_parsed_artifact_source_span_promotion_executor_apply_reports,
)
from knowledge_hub.papers.parsed_artifact_source_span_promotion_executor_dry_run import (
    EXECUTOR_STATUS_PLANNED,
)
from knowledge_hub.papers.parsed_artifact_source_span_store_contract import (
    build_parsed_artifact_source_span_store_contract,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _minimal_dry_run_row(*, paper_id: str = "paper-1", index: int = 1) -> dict:
    return {
        "promotion_executor_dry_run_row_id": f"parsed-artifact-source-span-promotion-executor-dry-run:{index:04d}",
        "policy_gate_row_id": f"gate:{index:04d}",
        "readback_review_row_id": f"readback:{index:04d}",
        "candidateRecordId": f"source-span-candidate:{paper_id}:section:{index}",
        "runId": "run-1",
        "paper_id": paper_id,
        "artifact_type": "section",
        "source_candidate_id": f"candidate-{index}",
        "source_readiness_row_id": f"readiness-{index}",
        "sourceContentHash": f"source-hash-{index}",
        "source_file": f"{paper_id}.pdf",
        "locator": {
            "page": index,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "blockIndexes": [index],
            "chars": {"start": None, "end": None},
        },
        "idempotencyKey": f"idem-{index}",
        "candidate_store_path": "",
        "candidate_store_line": 0,
        "policy_gate_status": "policy_gate_ready_candidate_only",
        "executor_dry_run_status": EXECUTOR_STATUS_PLANNED,
        "executor_blockers": [],
        "plannedSourceSpanKey": f"source-span:{paper_id}:section:{index}",
        "plannedSourceSpanId": f"source-span:{paper_id}:section:{index}",
        "plannedWriteTarget": "parsed_artifact_source_span_store",
        "writeMatrix": {"writeEnabled": False},
        "rollbackNote": "dry-run only",
        "dryRunPlannedSourceSpan": True,
        "sourceSpanPromotionApproved": False,
        "sourceSpanCreated": False,
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "canonicalParsedArtifactsWritten": False,
        "recommended_action": "queue_for_explicit_source_span_promotion_executor_apply",
    }


def _fixture_reports(tmp_path: Path) -> tuple[Path, Path]:
    dry_run_report = {
        "schema": "knowledge-hub.paper.parsed-artifact-source-span-promotion-executor-dry-run.v1",
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "policyGateReportPath": "",
            "policyGateSchema": "",
            "requestedPaperIds": [],
        },
        "counts": {
            "inputRows": 2,
            "dryRunPlannedSourceSpanRows": 2,
            "blockedInputSchemaViolationRows": 0,
            "blockedPolicyGateNotReadyRows": 0,
            "blockedMissingSourceHashRows": 0,
            "blockedMissingLocatorRows": 0,
            "sourceSpanCreatedRows": 0,
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
            "byExecutorDryRunStatus": {},
            "byRecommendedAction": {},
        },
        "gate": {
            "readyForSourceSpanPromotionApply": False,
            "sourceSpanPromotionApproved": False,
            "sourceSpanCreated": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "parsed_artifact_source_span_promotion_executor_dry_run_ready",
            "recommendedNextTranche": "parsed_artifact_source_span_promotion_executor_apply",
        },
        "policy": {
            "reportOnly": True,
            "promotionExecutorDryRunOnly": True,
            "sourceSpanStoreWrite": False,
            "candidateStoreWrite": False,
            "sourceSpanPromotionApproved": False,
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
        "rows": [_minimal_dry_run_row(index=1), _minimal_dry_run_row(paper_id="paper-2", index=2)],
    }
    contract_report = build_parsed_artifact_source_span_store_contract()
    dry_run_path = tmp_path / "dry-run.json"
    contract_path = tmp_path / "store-contract.json"
    _write_json(dry_run_path, dry_run_report)
    _write_json(contract_path, contract_report)
    return dry_run_path, contract_path


def test_promotion_executor_apply_dry_run_plans_rows_without_writes(tmp_path: Path) -> None:
    dry_run_path, contract_path = _fixture_reports(tmp_path)

    report = execute_parsed_artifact_source_span_promotion_executor_apply(
        promotion_executor_dry_run_report=dry_run_path,
        store_contract_report=contract_path,
        apply=False,
    )

    assert report["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_APPLY_SCHEMA_ID
    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 2
    assert report["counts"]["plannedApplyRows"] == 2
    assert report["counts"]["sourceSpanWriteRows"] == 0
    assert report["counts"]["readbackValidatedRows"] == 0
    assert report["counts"]["sourceSpanCreatedRows"] == 0
    assert report["gate"]["rollbackImplemented"] is False
    assert len(report["sourceSpanRecords"]) == 2
    assert {row["apply_status"] for row in report["rows"]} == {APPLY_STATUS_PLANNED}
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_APPLY_SCHEMA_ID,
        strict=True,
    ).ok


def test_promotion_executor_apply_writes_source_span_jsonl(tmp_path: Path) -> None:
    dry_run_path, contract_path = _fixture_reports(tmp_path)
    papers_dir = tmp_path / "papers"

    report = execute_parsed_artifact_source_span_promotion_executor_apply(
        promotion_executor_dry_run_report=dry_run_path,
        store_contract_report=contract_path,
        papers_dir=papers_dir,
        run_id="apply-test-run",
        apply=True,
    )

    assert report["status"] == "ok"
    assert report["counts"]["sourceSpanWriteRows"] == 2
    assert report["counts"]["readbackValidatedRows"] == 2
    assert report["counts"]["sourceSpanCreatedRows"] == 2
    assert report["policy"]["sourceSpanStoreWrite"] is True
    assert report["gate"]["rollbackImplemented"] is False
    assert {row["apply_status"] for row in report["rows"]} == {APPLY_STATUS_APPLIED}

    paper_one_path = papers_dir / "structured_evidence" / "source_span" / "paper-1.jsonl"
    paper_two_path = papers_dir / "structured_evidence" / "source_span" / "paper-2.jsonl"
    assert paper_one_path.exists()
    assert paper_two_path.exists()
    assert len(_read_jsonl(paper_one_path)) == 1
    assert len(_read_jsonl(paper_two_path)) == 1
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_APPLY_SCHEMA_ID,
        strict=True,
    ).ok


def test_promotion_executor_apply_requires_papers_dir_for_apply(tmp_path: Path) -> None:
    dry_run_path, contract_path = _fixture_reports(tmp_path)

    report = execute_parsed_artifact_source_span_promotion_executor_apply(
        promotion_executor_dry_run_report=dry_run_path,
        store_contract_report=contract_path,
        apply=True,
    )

    assert report["status"] == "blocked"
    assert "apply_requires_papers_dir" in report["gate"]["schemaViolations"]


def test_promotion_executor_apply_blocks_non_ready_dry_run_rows(tmp_path: Path) -> None:
    dry_run_path, contract_path = _fixture_reports(tmp_path)
    dry_run_report = json.loads(dry_run_path.read_text(encoding="utf-8"))
    dry_run_report["rows"][0]["executor_dry_run_status"] = "blocked_policy_gate_not_ready"
    dry_run_report["rows"][0]["dryRunPlannedSourceSpan"] = False
    _write_json(dry_run_path, dry_run_report)

    report = execute_parsed_artifact_source_span_promotion_executor_apply(
        promotion_executor_dry_run_report=dry_run_path,
        store_contract_report=contract_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["plannedApplyRows"] == 1
    assert report["counts"]["blockedNonReadyInputRows"] == 1


def test_promotion_executor_apply_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    dry_run_path, contract_path = _fixture_reports(tmp_path)
    report = execute_parsed_artifact_source_span_promotion_executor_apply(
        promotion_executor_dry_run_report=dry_run_path,
        store_contract_report=contract_path,
    )

    paths = write_parsed_artifact_source_span_promotion_executor_apply_reports(
        report,
        tmp_path / "reports",
    )

    written_report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert written_report["counts"]["plannedApplyRows"] == 2
    assert validate_payload(
        written_report,
        PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_APPLY_SCHEMA_ID,
        strict=True,
    ).ok


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
