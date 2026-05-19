from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_strict_evidence_promotion_tranche_plan import (
    TRANCHE_FIGURE_CAPTION_PILOT,
    TRANCHE_TEXT_SECTION_PILOT,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_record_contract import (
    build_sample_strict_evidence_record_from_packet_row,
)
from knowledge_hub.papers.strict_evidence_figure_caption_pilot import (
    PILOT_STATUS_CANDIDATE_ONLY,
    PILOT_STATUS_HELD_OUT_SECTION,
    STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_SCHEMA_ID,
)
from knowledge_hub.papers.strict_evidence_figure_caption_pilot_executor_dry_run import (
    DRY_RUN_STATUS_BLOCKED_HASH_CONTRACT,
    DRY_RUN_STATUS_HELD_OUT,
    DRY_RUN_STATUS_READY,
    PLANNED_ACTION_VALIDATE_ONLY,
    PLANNED_ANSWER_EFFECT_NONE,
    PLANNED_RUNTIME_EFFECT_NONE,
    PLANNED_WRITE_TARGET_NONE,
    STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_DRY_RUN_SCHEMA_ID,
    build_strict_evidence_figure_caption_pilot_executor_dry_run,
    write_strict_evidence_figure_caption_pilot_executor_dry_run_reports,
)


def _strict_evidence_record(
    *,
    paper_id: str = "paper-1",
    index: int = 1,
    artifact_type: str = "figure",
) -> dict:
    packet_row = {
        "packet_review_row_id": f"packet:{index:04d}",
        "paper_id": paper_id,
        "artifact_type": artifact_type,
        "sourceSpanId": f"source-span:{paper_id}:{artifact_type}:{index}",
        "candidateRecordId": f"source-span-candidate:{paper_id}:{artifact_type}:{index}",
        "sourceContentHash": "hash-1",
        "source_file": f"{paper_id}.pdf",
        "text_surface": "Figure 1 caption",
        "proposed_chars": {
            "start": 0,
            "end": 12,
            "basis": "sourceContentHash",
            "normalization": "nfkc_whitespace_casefold_v1",
            "expectedSubstringSha256": "abc123",
        },
    }
    return build_sample_strict_evidence_record_from_packet_row(packet_row, run_id="run-1")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_figure_caption_pilot_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _synthetic_figure_caption_pilot_row(
    *,
    store_path: Path,
    index: int = 1,
    artifact_type: str = "figure",
    paper_id: str = "paper-1",
) -> dict:
    planned_tranche = (
        TRANCHE_TEXT_SECTION_PILOT
        if artifact_type == "section"
        else TRANCHE_FIGURE_CAPTION_PILOT
    )
    pilot_status = (
        PILOT_STATUS_CANDIDATE_ONLY
        if artifact_type == "figure"
        else PILOT_STATUS_HELD_OUT_SECTION
    )
    return {
        "pilot_row_id": f"strict-evidence-figure-caption-pilot:{index:04d}",
        "plan_row_id": f"parsed-artifact-strict-evidence-promotion-tranche-plan:{index:04d}",
        "strictEvidenceId": f"strict-evidence:{paper_id}:{artifact_type}:{index}",
        "sourceSpanId": f"source-span:{paper_id}:{artifact_type}:{index}",
        "candidateRecordId": f"source-span-candidate:{paper_id}:{artifact_type}:{index}",
        "runId": "run-1",
        "paper_id": paper_id,
        "artifact_type": artifact_type,
        "sourceContentHash": "hash-1",
        "idempotencyKey": f"strict-idem-{index}",
        "strict_evidence_store_path": str(store_path),
        "strict_evidence_store_line": index,
        "planned_tranche": planned_tranche,
        "planned_tranche_scope": "figure caption text strict evidence pilot (not figure region)",
        "plan_status": "promotion_tranche_plan_candidate_only",
        "promotionTranchePlanCandidateOnly": True,
        "pilot_status": pilot_status,
        "pilot_blockers": [],
        "figureCaptionPilotCandidateOnly": artifact_type == "figure",
        "figureCaptionTextOnly": artifact_type == "figure",
        "verbatimSubstringSha256": "abc123",
        "authority_chars": {
            "start": 0,
            "end": 12,
            "basis": "sourceContentHash",
            "normalization": "nfkc_whitespace_casefold_v1",
            "expectedSubstringSha256": "abc123",
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
        "recommended_action": "queue_for_strict_evidence_figure_caption_pilot_executor_dry_run",
    }


def _synthetic_figure_caption_pilot_report(rows: list[dict]) -> dict:
    figure_rows = sum(1 for row in rows if row.get("pilot_status") == PILOT_STATUS_CANDIDATE_ONLY)
    held_rows = sum(1 for row in rows if row.get("pilot_status") == PILOT_STATUS_HELD_OUT_SECTION)
    return {
        "schema": STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "tranchePlanReportPath": "/tmp/tranche-plan.json",
            "tranchePlanSchema": "knowledge-hub.paper.parsed-artifact-strict-evidence-promotion-tranche-plan.v1",
            "tranchePlanReportStatus": "ok",
            "requestedPaperIds": [],
            "targetTranche": TRANCHE_FIGURE_CAPTION_PILOT,
        },
        "counts": {
            "inputRows": len(rows),
            "figureCaptionPilotInputRows": figure_rows,
            "figureCaptionPilotCandidateOnlyRows": figure_rows,
            "heldOutSectionRows": held_rows,
            "blockedInputReportSchemaViolationRows": 0,
            "blockedTrancheNotAssignedRows": 0,
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
            "byArtifactType": {},
            "byPilotStatus": {},
            "byRecommendedAction": {},
        },
        "heldOutSection": {
            "heldOutSectionRows": held_rows,
            "byPlannedTranche": {},
            "byArtifactType": {},
            "diagnosticOnly": True,
            "activePilotProcessing": False,
        },
        "gate": {
            "figureCaptionPilotReady": figure_rows > 0,
            "strictEligibleMutationAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "sectionRowsProcessedAsActivePilot": False,
            "schemaViolations": [],
            "decision": "strict_evidence_figure_caption_pilot_ready",
            "recommendedNextTranche": TRANCHE_FIGURE_CAPTION_PILOT,
        },
        "policy": {
            "reportOnly": True,
            "figureCaptionPilotOnly": True,
            "figureCaptionTextOnlyNotFigureRegion": True,
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


def test_executor_dry_run_marks_figure_ready_and_holds_out_section(tmp_path: Path) -> None:
    store_path = tmp_path / "strict.jsonl"
    _write_jsonl(
        store_path,
        [
            _strict_evidence_record(index=1, artifact_type="figure"),
            _strict_evidence_record(index=2, artifact_type="section"),
        ],
    )
    pilot_path = tmp_path / "figure-caption-pilot.json"
    _write_figure_caption_pilot_report(
        pilot_path,
        _synthetic_figure_caption_pilot_report(
            [
                _synthetic_figure_caption_pilot_row(store_path=store_path, index=1, artifact_type="figure"),
                _synthetic_figure_caption_pilot_row(store_path=store_path, index=2, artifact_type="section"),
            ]
        ),
    )

    report = build_strict_evidence_figure_caption_pilot_executor_dry_run(
        figure_caption_pilot_report_path=pilot_path,
    )

    assert report["schema"] == STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert report["status"] == "ok"
    assert report["counts"]["dryRunReadyFigureCaptionPilotExecutorOnlyRows"] == 1
    assert report["counts"]["heldOutSectionRows"] == 1
    assert report["gate"]["dryRunReadyForFigureCaptionPilotExecutor"] is True
    assert report["gate"]["sectionRowsProcessedAsActiveExecutor"] is False
    ready_row = next(row for row in report["rows"] if row["dry_run_status"] == DRY_RUN_STATUS_READY)
    assert ready_row["plannedAction"] == PLANNED_ACTION_VALIDATE_ONLY
    assert ready_row["plannedWriteTarget"] == PLANNED_WRITE_TARGET_NONE
    assert ready_row["plannedRuntimeEffect"] == PLANNED_RUNTIME_EFFECT_NONE
    assert ready_row["plannedAnswerEffect"] == PLANNED_ANSWER_EFFECT_NONE
    assert {row["dry_run_status"] for row in report["rows"]} == {
        DRY_RUN_STATUS_READY,
        DRY_RUN_STATUS_HELD_OUT,
    }
    assert validate_payload(
        report,
        STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_executor_dry_run_blocks_hash_contract_violation() -> None:
    from knowledge_hub.papers.strict_evidence_figure_caption_pilot_executor_dry_run import (
        _classify_dry_run_row,
    )

    pilot_row = _synthetic_figure_caption_pilot_row(
        store_path=Path("/tmp/strict.jsonl"),
        index=1,
        artifact_type="figure",
    )
    pilot_row["verbatimSubstringSha256"] = "mismatch"
    status, blockers = _classify_dry_run_row(
        pilot_row,
        strict_evidence_record={
            "authority": {"chars": pilot_row["authority_chars"]},
            "verbatimSubstringSha256": "mismatch",
        },
    )
    assert status == DRY_RUN_STATUS_BLOCKED_HASH_CONTRACT
    assert blockers


def test_executor_dry_run_blocks_invalid_input_schema(tmp_path: Path) -> None:
    pilot_report = _synthetic_figure_caption_pilot_report(
        [_synthetic_figure_caption_pilot_row(store_path=tmp_path / "strict.jsonl", index=1)]
    )
    pilot_report["schema"] = "wrong.schema"
    pilot_path = tmp_path / "figure-caption-pilot.json"
    _write_figure_caption_pilot_report(pilot_path, pilot_report)

    report = build_strict_evidence_figure_caption_pilot_executor_dry_run(
        figure_caption_pilot_report_path=pilot_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0
    assert report["gate"]["dryRunReadyForFigureCaptionPilotExecutor"] is False
    assert validate_payload(
        report,
        STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_executor_dry_run_integrated_from_figure_caption_pilot_builder(tmp_path: Path) -> None:
    from knowledge_hub.papers.strict_evidence_figure_caption_pilot import (
        build_strict_evidence_figure_caption_pilot,
        write_strict_evidence_figure_caption_pilot_reports,
    )
    from tests.test_strict_evidence_figure_caption_pilot import (
        _synthetic_tranche_plan_report,
        _synthetic_tranche_plan_row,
        _write_tranche_plan_report,
    )

    store_path = tmp_path / "strict.jsonl"
    _write_jsonl(
        store_path,
        [
            _strict_evidence_record(index=1, artifact_type="figure"),
            _strict_evidence_record(index=2, artifact_type="figure"),
        ],
    )
    tranche_plan_path = tmp_path / "tranche-plan.json"
    _write_tranche_plan_report(
        tranche_plan_path,
        _synthetic_tranche_plan_report(
            [
                _synthetic_tranche_plan_row(store_path=store_path, index=1, artifact_type="figure"),
                _synthetic_tranche_plan_row(store_path=store_path, index=2, artifact_type="figure"),
            ]
        ),
    )
    figure_caption_pilot_report = build_strict_evidence_figure_caption_pilot(
        tranche_plan_report_path=tranche_plan_path,
    )
    figure_caption_pilot_dir = tmp_path / "figure-caption-pilot"
    write_strict_evidence_figure_caption_pilot_reports(
        figure_caption_pilot_report,
        figure_caption_pilot_dir,
    )

    report = build_strict_evidence_figure_caption_pilot_executor_dry_run(
        figure_caption_pilot_report_path=figure_caption_pilot_dir
        / "strict-evidence-figure-caption-pilot.json",
    )

    assert report["status"] == "ok"
    assert report["counts"]["dryRunReadyFigureCaptionPilotExecutorOnlyRows"] == 2
    assert report["counts"]["strictEvidenceWriteRows"] == 0


def test_executor_dry_run_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    store_path = tmp_path / "strict.jsonl"
    _write_jsonl(store_path, [_strict_evidence_record(index=1, artifact_type="figure")])
    pilot_path = tmp_path / "figure-caption-pilot.json"
    _write_figure_caption_pilot_report(
        pilot_path,
        _synthetic_figure_caption_pilot_report(
            [_synthetic_figure_caption_pilot_row(store_path=store_path, index=1, artifact_type="figure")]
        ),
    )
    report = build_strict_evidence_figure_caption_pilot_executor_dry_run(
        figure_caption_pilot_report_path=pilot_path,
    )
    paths = write_strict_evidence_figure_caption_pilot_executor_dry_run_reports(
        report,
        tmp_path / "reports",
    )
    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert validate_payload(
        written,
        STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok
