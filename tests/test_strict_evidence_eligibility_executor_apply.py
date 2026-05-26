from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_eligibility_executor_apply import (
    APPLY_STATUS_APPLIED,
    APPLY_STATUS_READY,
    STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_APPLY_SCHEMA_ID,
    STRICT_EVIDENCE_ELIGIBILITY_STORE,
    build_strict_evidence_eligibility_executor_apply,
    write_strict_evidence_eligibility_executor_apply_reports,
)
from knowledge_hub.papers.strict_evidence_eligibility_executor_dry_run import (
    DRY_RUN_STATUS_READY,
    STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID,
    build_strict_evidence_eligibility_executor_dry_run,
    write_strict_evidence_eligibility_executor_dry_run_reports,
)
from knowledge_hub.papers.strict_evidence_eligibility_record_contract import (
    STRICT_EVIDENCE_ELIGIBILITY_RECORD_SCHEMA_ID,
    build_strict_evidence_eligibility_record_contract,
    write_strict_evidence_eligibility_record_contract_reports,
)
from tests.test_strict_evidence_eligibility_record_contract import (
    _decision_report,
    _decision_row,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fixture_reports(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = [_decision_row(index=1, artifact_type="section")]
    decision_path = tmp_path / "decision-record.json"
    _write_json(decision_path, _decision_report(rows))
    contract = build_strict_evidence_eligibility_record_contract(
        decision_record_report_path=decision_path,
    )
    contract_dir = tmp_path / "contract"
    write_strict_evidence_eligibility_record_contract_reports(contract, contract_dir)
    contract_path = contract_dir / "strict-evidence-eligibility-record-contract.json"
    dry_run = build_strict_evidence_eligibility_executor_dry_run(
        eligibility_record_contract_report_path=contract_path,
        decision_record_report_path=decision_path,
        run_id="dry-run-test",
    )
    dry_run_dir = tmp_path / "dry-run"
    write_strict_evidence_eligibility_executor_dry_run_reports(dry_run, dry_run_dir)
    dry_run_path = dry_run_dir / "strict-evidence-eligibility-executor-dry-run.json"
    return dry_run_path, contract_path, decision_path


def test_eligibility_executor_apply_plans_without_writes(tmp_path: Path) -> None:
    dry_run_path, contract_path, _ = _fixture_reports(tmp_path)

    report = build_strict_evidence_eligibility_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        eligibility_record_contract_report_path=contract_path,
        apply=False,
    )

    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 1
    assert report["counts"]["plannedApplyRows"] == 1
    assert report["counts"]["eligibilityRecordWriteRows"] == 0
    assert report["counts"]["strictEvidenceWriteRows"] == 0
    assert report["counts"]["strictEligibleMutationRows"] == 0
    assert report["rows"][0]["apply_status"] == APPLY_STATUS_READY
    assert report["rows"][0]["would_write_eligibility_record"] is True
    assert validate_payload(
        report,
        STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_APPLY_SCHEMA_ID,
        strict=True,
    ).ok


def test_eligibility_executor_apply_writes_eligibility_jsonl(tmp_path: Path) -> None:
    dry_run_path, contract_path, _ = _fixture_reports(tmp_path)
    papers_dir = tmp_path / "papers"

    report = build_strict_evidence_eligibility_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        eligibility_record_contract_report_path=contract_path,
        papers_dir=papers_dir,
        run_id="test-apply-run",
        apply=True,
    )

    assert report["status"] == "ok"
    assert report["counts"]["eligibilityRecordWriteRows"] == 1
    assert report["counts"]["readbackValidatedRows"] == 1
    assert report["counts"]["appliedEligibilityRecordRows"] == 1
    assert report["counts"]["strictEvidenceWriteRows"] == 0
    assert report["rows"][0]["apply_status"] == APPLY_STATUS_APPLIED

    jsonl_path = papers_dir / "structured_evidence" / "strict_evidence_eligibility" / "paper-1.jsonl"
    assert jsonl_path.is_file()
    stored = json.loads(jsonl_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert validate_payload(stored, STRICT_EVIDENCE_ELIGIBILITY_RECORD_SCHEMA_ID, strict=True).ok
    assert stored["runId"] == "test-apply-run"
    assert stored["plannedWriteTarget"] == STRICT_EVIDENCE_ELIGIBILITY_STORE
    assert stored["citationGrade"] is False
    assert stored["runtimeEvidence"] is False
    assert stored["strictEligibleMutationApplied"] is False


def test_eligibility_executor_apply_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    dry_run_path, contract_path, _ = _fixture_reports(tmp_path)

    report = build_strict_evidence_eligibility_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        eligibility_record_contract_report_path=contract_path,
        apply=False,
    )
    paths = write_strict_evidence_eligibility_executor_apply_reports(
        report,
        tmp_path / "reports",
    )

    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert written["schema"] == STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_APPLY_SCHEMA_ID
    assert validate_payload(
        written,
        STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_APPLY_SCHEMA_ID,
        strict=True,
    ).ok


def test_eligibility_executor_apply_integrated_measured_local_plan() -> None:
    report = build_strict_evidence_eligibility_executor_apply(apply=False)
    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 99
    assert report["counts"]["plannedApplyRows"] == 99
    assert report["counts"]["eligibilityRecordWriteRows"] == 0
    assert report["counts"]["strictEligibleMutationRows"] == 0
    assert validate_payload(
        report,
        STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_APPLY_SCHEMA_ID,
        strict=True,
    ).ok


def _synthetic_dry_run_report(dry_run_rows: list[dict]) -> dict:
    ready_rows = sum(1 for row in dry_run_rows if row.get("dry_run_status") == DRY_RUN_STATUS_READY)
    return {
        "schema": STRICT_EVIDENCE_ELIGIBILITY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-20T00:00:00Z",
        "input": {
            "eligibilityRecordContractReportPath": "/tmp/contract.json",
            "decisionRecordReportPath": "/tmp/decision.json",
            "requestedPaperIds": [],
            "runId": "dry-run-test",
            "expectedPolicyCandidateRows": len(dry_run_rows),
        },
        "counts": {
            "inputRows": len(dry_run_rows),
            "decisionCandidateRows": ready_rows,
            "dryRunReadyEligibilityRecordOnlyRows": ready_rows,
            "blockedContractNotReadyRows": 0,
            "blockedDecisionRecordNotReadyRows": 0,
            "blockedMissingStrictEvidenceIdRows": 0,
            "blockedMissingSourceSpanIdRows": 0,
            "blockedMissingCandidateRecordIdRows": 0,
            "blockedPlannedRecordSchemaViolationRows": 0,
            "blockedPlannedRecordSemanticViolationRows": 0,
            "blockedInputSchemaViolationRows": 0,
            "eligibilityRecordWriteRows": 0,
            "strictEligibleMutationRows": 0,
            "strictEvidenceWriteRows": 0,
            "strictEvidenceCreatedRows": 0,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "sourceSpanUpdatedRows": 0,
            "reindexOrReembedRows": 0,
            "manifestWriteRows": 0,
            "schemaViolationCount": 0,
            "byPaperId": {},
            "byDryRunStatus": {DRY_RUN_STATUS_READY: ready_rows},
            "byRecommendedAction": {},
        },
        "dryRunOnlyPolicyMatrix": {
            "plannedWriteTarget": STRICT_EVIDENCE_ELIGIBILITY_STORE,
            "writeEnabled": False,
            "eligibilityRecordWrite": False,
            "strictEvidenceStoreWrite": False,
            "sourceSpanStoreWrite": False,
        },
        "gate": {
            "readyForEligibilityExecutorDryRun": True,
            "readyForEligibilityExecutorApply": False,
            "strictEligibleMutationAllowed": False,
            "eligibilityRecordWriteAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "sourceSpanStoreWriteAllowed": False,
            "runManifestWriteAllowed": False,
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "strict_evidence_eligibility_executor_dry_run_ready",
            "recommendedNextTranche": "strict_evidence_eligibility_executor_apply",
        },
        "policy": {"reportOnly": True, "dryRunOnly": True},
        "warnings": [],
        "rows": dry_run_rows,
    }


def test_eligibility_executor_apply_blocks_non_ready_dry_run_row(tmp_path: Path) -> None:
    dry_run_path, contract_path, decision_path = _fixture_reports(tmp_path)
    dry_run = json.loads(dry_run_path.read_text(encoding="utf-8"))
    dry_run["rows"][0]["dry_run_status"] = "blocked_input_schema_violation"
    dry_run["rows"][0]["dryRunReadyEligibilityRecordOnly"] = False
    _write_json(dry_run_path, dry_run)

    report = build_strict_evidence_eligibility_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        eligibility_record_contract_report_path=contract_path,
        apply=False,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["blockedDryRunNotReadyRows"] >= 1
    _ = decision_path
