from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_strict_evidence_promotion_tranche_plan import (
    TRANCHE_FIGURE_CAPTION_PILOT,
    TRANCHE_TEXT_SECTION_PILOT,
)
from knowledge_hub.papers.strict_evidence_text_section_pilot_executor_apply import (
    APPLY_STATUS_APPLIED,
    APPLY_STATUS_BLOCKED_DRY_RUN_NOT_READY,
    APPLY_STATUS_HELD_OUT,
    APPLY_STATUS_READY,
    STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
    build_strict_evidence_text_section_pilot_executor_apply,
    write_strict_evidence_text_section_pilot_executor_apply_reports,
)
from knowledge_hub.papers.strict_evidence_text_section_pilot_executor_dry_run import (
    DRY_RUN_STATUS_HELD_OUT,
    DRY_RUN_STATUS_READY,
    PLANNED_ACTION_VALIDATE_ONLY,
    PLANNED_WRITE_TARGET_NONE,
    STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_DRY_RUN_SCHEMA_ID,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _synthetic_dry_run_row(
    *,
    index: int = 1,
    artifact_type: str = "section",
    dry_run_status: str = DRY_RUN_STATUS_READY,
) -> dict:
    planned_tranche = (
        TRANCHE_TEXT_SECTION_PILOT
        if artifact_type == "section"
        else TRANCHE_FIGURE_CAPTION_PILOT
    )
    ready = dry_run_status == DRY_RUN_STATUS_READY
    return {
        "dry_run_row_id": f"strict-evidence-text-section-pilot-executor-dry-run:{index:04d}",
        "pilot_row_id": f"strict-evidence-text-section-pilot:{index:04d}",
        "plan_row_id": f"parsed-artifact-strict-evidence-promotion-tranche-plan:{index:04d}",
        "strictEvidenceId": f"strict-evidence:paper-1:{artifact_type}:{index}",
        "sourceSpanId": f"source-span:paper-1:{artifact_type}:{index}",
        "candidateRecordId": f"source-span-candidate:paper-1:{artifact_type}:{index}",
        "runId": "run-1",
        "paper_id": "paper-1",
        "artifact_type": artifact_type,
        "sourceContentHash": "hash-1",
        "idempotencyKey": f"strict-idem-{index}",
        "strict_evidence_store_path": "/tmp/strict.jsonl",
        "strict_evidence_store_line": index,
        "planned_tranche": planned_tranche,
        "pilot_status": "text_section_pilot_candidate_only",
        "textSectionPilotCandidateOnly": artifact_type == "section",
        "verbatimSubstringSha256": "abc123",
        "authority_chars": {
            "start": 0,
            "end": 12,
            "basis": "sourceContentHash",
            "normalization": "nfkc_whitespace_casefold_v1",
            "expectedSubstringSha256": "abc123",
        },
        "planned_executor_key": f"section-pilot-executor:strict-evidence:paper-1:{artifact_type}:{index}",
        "plannedAction": PLANNED_ACTION_VALIDATE_ONLY if ready else "",
        "plannedWriteTarget": PLANNED_WRITE_TARGET_NONE,
        "plannedRuntimeEffect": "none",
        "plannedAnswerEffect": "none",
        "dry_run_status": dry_run_status,
        "dry_run_blockers": [],
        "dryRunReadyTextSectionPilotExecutorOnly": ready,
        "writeMatrix": {
            "plannedAction": PLANNED_ACTION_VALIDATE_ONLY,
            "plannedWriteTarget": PLANNED_WRITE_TARGET_NONE,
            "writeEnabled": False,
            "strictEvidenceStoreWrite": False,
            "sourceSpanStoreWrite": False,
        },
        "strictEligible": False,
        "strictEvidenceWriteRows": 0,
        "strictEvidenceCreated": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "canonicalParsedArtifactsWritten": False,
        "sourceSpanUpdatedRows": 0,
        "recommended_action": "queue_for_strict_evidence_text_section_pilot_executor_apply",
    }


def _synthetic_dry_run_report(rows: list[dict]) -> dict:
    ready_rows = sum(1 for row in rows if row.get("dry_run_status") == DRY_RUN_STATUS_READY)
    held_rows = sum(1 for row in rows if row.get("dry_run_status") == DRY_RUN_STATUS_HELD_OUT)
    return {
        "schema": STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "sectionPilotReportPath": "/tmp/section-pilot.json",
            "sectionPilotSchema": "knowledge-hub.paper.strict-evidence-text-section-pilot.v1",
            "requestedPaperIds": [],
            "targetTranche": TRANCHE_TEXT_SECTION_PILOT,
        },
        "counts": {
            "inputRows": len(rows),
            "sectionPilotCandidateRows": ready_rows,
            "dryRunReadyTextSectionPilotExecutorOnlyRows": ready_rows,
            "heldOutNonSectionRows": held_rows,
            "blockedInputReportSchemaViolationRows": 0,
            "blockedSectionPilotNotCandidateRows": 0,
            "blockedMissingRecordIdentityRows": 0,
            "blockedInvalidArtifactTypeRows": 0,
            "blockedMissingAuthorityCharsRows": 0,
            "blockedHashContractViolationRows": 0,
            "blockedRuntimeOrCitationFlagViolationRows": 0,
            "strictEvidenceWriteRows": 0,
            "strictEvidenceCreatedRows": 0,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "sourceSpanUpdatedRows": 0,
            "schemaViolationCount": 0,
            "byPaperId": {},
            "byDryRunStatus": {},
            "byRecommendedAction": {},
        },
        "heldOutNonSection": {
            "heldOutNonSectionRows": held_rows,
            "byPlannedTranche": {},
            "byArtifactType": {},
            "diagnosticOnly": True,
            "activeExecutorProcessing": False,
        },
        "noMutationPolicyMatrix": {
            "plannedAction": PLANNED_ACTION_VALIDATE_ONLY,
            "plannedWriteTarget": PLANNED_WRITE_TARGET_NONE,
            "plannedRuntimeEffect": "none",
            "plannedAnswerEffect": "none",
            "writeEnabled": False,
            "strictEvidenceStoreWrite": False,
            "sourceSpanStoreWrite": False,
        },
        "gate": {
            "dryRunReadyForTextSectionPilotExecutor": ready_rows > 0,
            "strictEligibleMutationAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "applyMode": False,
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "figureRowsProcessedAsActiveExecutor": False,
            "schemaViolations": [],
            "decision": "strict_evidence_text_section_pilot_executor_dry_run_ready",
            "recommendedNextTranche": "strict_evidence_text_section_pilot_executor_apply",
        },
        "policy": {
            "reportOnly": True,
            "dryRunOnly": True,
            "strictEvidenceStoreWrite": False,
            "sourceSpanStoreWrite": False,
            "strictEvidenceCreated": False,
            "strictEligibleMutation": False,
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


def test_apply_plans_section_ready_and_holds_out_figure(tmp_path: Path) -> None:
    dry_run_path = tmp_path / "dry-run.json"
    _write_json(
        dry_run_path,
        _synthetic_dry_run_report(
            [
                _synthetic_dry_run_row(index=1, artifact_type="section"),
                _synthetic_dry_run_row(
                    index=2,
                    artifact_type="figure",
                    dry_run_status=DRY_RUN_STATUS_HELD_OUT,
                ),
            ]
        ),
    )

    report = build_strict_evidence_text_section_pilot_executor_apply(
        executor_dry_run_report_path=dry_run_path,
    )

    assert report["schema"] == STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID
    assert report["status"] == "ok"
    assert report["counts"]["plannedApplyRows"] == 1
    assert report["counts"]["manifestWriteRows"] == 0
    assert report["counts"]["heldOutNonSectionRows"] == 1
    assert report["gate"]["readyForDryRunApplyPlanning"] is True
    assert report["gate"]["figureRowsProcessedAsActiveApply"] is False
    assert {row["apply_status"] for row in report["rows"]} == {
        APPLY_STATUS_READY,
        APPLY_STATUS_HELD_OUT,
    }
    assert validate_payload(
        report,
        STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
        strict=True,
    ).ok


def test_apply_writes_manifest_only_when_explicit_apply(tmp_path: Path) -> None:
    dry_run_path = tmp_path / "dry-run.json"
    _write_json(
        dry_run_path,
        _synthetic_dry_run_report([_synthetic_dry_run_row(index=1, artifact_type="section")]),
    )
    papers_dir = tmp_path / "papers"

    report = build_strict_evidence_text_section_pilot_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        papers_dir=papers_dir,
        run_id="section-pilot-apply-test",
        apply=True,
    )

    assert report["status"] == "ok"
    assert report["counts"]["appliedManifestOnlyRows"] == 1
    assert report["counts"]["manifestWriteRows"] == 1
    assert report["counts"]["strictEvidenceWriteRows"] == 0
    manifest_path = Path(report["gate"]["runManifestPath"])
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID
    assert report["rows"][0]["apply_status"] == APPLY_STATUS_APPLIED
    assert validate_payload(
        report,
        STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
        strict=True,
    ).ok


def test_apply_blocks_non_ready_dry_run_row() -> None:
    from knowledge_hub.papers.strict_evidence_text_section_pilot_executor_apply import (
        _classify_apply_row,
    )

    dry_row = _synthetic_dry_run_row(
        index=1,
        artifact_type="section",
        dry_run_status="blocked_hash_contract_violation",
    )
    status, blockers, ready = _classify_apply_row(dry_row, input_schema_violations=[])

    assert ready is False
    assert status == APPLY_STATUS_BLOCKED_DRY_RUN_NOT_READY
    assert blockers


def test_apply_blocks_invalid_input_schema(tmp_path: Path) -> None:
    dry_run_report = _synthetic_dry_run_report([_synthetic_dry_run_row(index=1)])
    dry_run_report["schema"] = "wrong.schema"
    dry_run_path = tmp_path / "dry-run.json"
    _write_json(dry_run_path, dry_run_report)

    report = build_strict_evidence_text_section_pilot_executor_apply(
        executor_dry_run_report_path=dry_run_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0
    assert report["gate"]["readyForDryRunApplyPlanning"] is False


def test_apply_integrated_from_executor_dry_run_builder(tmp_path: Path) -> None:
    from knowledge_hub.papers.strict_evidence_text_section_pilot_executor_dry_run import (
        build_strict_evidence_text_section_pilot_executor_dry_run,
    )
    from tests.test_strict_evidence_text_section_pilot_executor_dry_run import (
        _synthetic_section_pilot_report,
        _synthetic_section_pilot_row,
        _write_section_pilot_report,
    )
    from tests.test_strict_evidence_text_section_pilot_executor_dry_run import (
        _strict_evidence_record,
        _write_jsonl,
    )

    store_path = tmp_path / "strict.jsonl"
    _write_jsonl(
        store_path,
        [
            _strict_evidence_record(index=1, artifact_type="section"),
            _strict_evidence_record(index=2, artifact_type="section"),
        ],
    )
    pilot_path = tmp_path / "section-pilot.json"
    _write_section_pilot_report(
        pilot_path,
        _synthetic_section_pilot_report(
            [
                _synthetic_section_pilot_row(store_path=store_path, index=1, artifact_type="section"),
                _synthetic_section_pilot_row(store_path=store_path, index=2, artifact_type="section"),
            ]
        ),
    )
    dry_run_report = build_strict_evidence_text_section_pilot_executor_dry_run(
        section_pilot_report_path=pilot_path,
    )
    dry_run_path = tmp_path / "dry-run.json"
    _write_json(dry_run_path, dry_run_report)

    report = build_strict_evidence_text_section_pilot_executor_apply(
        executor_dry_run_report_path=dry_run_path,
    )

    assert report["status"] == "ok"
    assert report["counts"]["plannedApplyRows"] == 2
    assert report["counts"]["strictEvidenceWriteRows"] == 0


def test_apply_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    dry_run_path = tmp_path / "dry-run.json"
    _write_json(
        dry_run_path,
        _synthetic_dry_run_report([_synthetic_dry_run_row(index=1, artifact_type="section")]),
    )
    report = build_strict_evidence_text_section_pilot_executor_apply(
        executor_dry_run_report_path=dry_run_path,
    )
    paths = write_strict_evidence_text_section_pilot_executor_apply_reports(
        report,
        tmp_path / "reports",
    )
    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert validate_payload(
        written,
        STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
        strict=True,
    ).ok
