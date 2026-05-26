from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.candidate_layer_blocker_resolution_preview import (
    CANDIDATE_LAYER_BLOCKER_RESOLUTION_PREVIEW_SCHEMA_ID,
    build_candidate_layer_blocker_resolution_preview,
    write_candidate_layer_blocker_resolution_preview_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _decision_record(needs_review: int, **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-decision-record.v1",
        "status": "decision_record_required" if needs_review else "decision_recorded",
        "counts": {
            "recordRows": 2,
            "needsReviewRows": needs_review,
            "manualApprovalRows": 1 if not needs_review else 0,
            "operatorApprovedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "decisionRecordOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
    }
    payload.update(overrides)
    return payload


def _backlog(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-backlog.v1",
        "status": "ok",
        "counts": {
            "backlogItemCount": 2,
            "openBacklogItemCount": 2,
            "strictEligibleCandidates": 0,
            "citationGradeCandidates": 0,
            "runtimeEvidenceCandidates": 0,
        },
        "gate": {
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
        },
        "policy": {
            "backlogOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "backlog": [
            {
                "backlog_id": "candidate-layer-blocker-v1-001",
                "blocker": "candidate_layer_blocker_decision_record_pending",
                "priority": "P0",
                "category": "blocker_decision_review",
                "affected_layers": ["sectionspan"],
                "affected_candidate_count": 12,
                "affected_eval_question_count": 20,
                "recommendedNextTranche": "manual_record_candidate_layer_blocker_decisions",
            },
            {
                "backlog_id": "candidate-layer-blocker-v1-002",
                "blocker": "table_cell_row_column_bbox_provenance_missing",
                "priority": "P0",
                "category": "table_cell_provenance",
                "affected_layers": ["table_region"],
                "affected_candidate_count": 5,
                "affected_eval_question_count": 5,
                "recommendedNextTranche": "table_cell_provenance_feasibility_audit",
            },
        ],
    }
    payload.update(overrides)
    return payload


def test_resolution_preview_keeps_needs_review_decision_record_blocked(tmp_path: Path) -> None:
    record = _write(tmp_path, "record.json", _decision_record(needs_review=2))
    backlog = _write(tmp_path, "backlog.json", _backlog())

    payload = build_candidate_layer_blocker_resolution_preview(
        candidate_layer_blocker_decision_record_report=record,
        candidate_layer_blocker_backlog_report=backlog,
    )

    assert payload["schema"] == CANDIDATE_LAYER_BLOCKER_RESOLUTION_PREVIEW_SCHEMA_ID
    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_RESOLUTION_PREVIEW_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "resolution_preview_ready"
    assert payload["counts"]["previewRows"] == 2
    assert payload["counts"]["decisionRecordNeedsReviewRows"] == 2
    assert payload["counts"]["stillBlockedRows"] == 1
    assert payload["gate"]["decisionRecordPending"] is True
    assert payload["gate"]["strictEvidenceReady"] is False
    assert payload["policy"]["answerIntegrationChanged"] is False
    pending = next(row for row in payload["previewRows"] if row["blocker"] == "candidate_layer_blocker_decision_record_pending")
    assert pending["preview_status"] == "still_blocked"
    assert pending["strict_eligible"] is False


def test_resolution_preview_complete_decision_record_still_report_only(tmp_path: Path) -> None:
    record = _write(tmp_path, "record.json", _decision_record(needs_review=0))
    backlog = _write(tmp_path, "backlog.json", _backlog())

    payload = build_candidate_layer_blocker_resolution_preview(
        candidate_layer_blocker_decision_record_report=record,
        candidate_layer_blocker_backlog_report=backlog,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_RESOLUTION_PREVIEW_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["decisionRecordNeedsReviewRows"] == 0
    assert payload["counts"]["decisionRecordManualApprovalRows"] == 1
    assert payload["counts"]["decisionRecordCompleteReportOnlyRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["counts"]["runtimeEvidenceRows"] == 0
    assert payload["gate"]["decisionRecordCompleteReportOnly"] is True
    assert payload["gate"]["runtimePromotionAllowed"] is False
    pending = next(row for row in payload["previewRows"] if row["blocker"] == "candidate_layer_blocker_decision_record_pending")
    assert pending["preview_status"] == "decision_record_complete_report_only"
    assert "resolution_preview_rows_are_not_backlog_mutations" in pending["non_strict_reason"]


def test_resolution_preview_blocks_unsafe_inputs(tmp_path: Path) -> None:
    record = _write(
        tmp_path,
        "record.json",
        _decision_record(
            needs_review=0,
            schema="example.wrong.record.v1",
            policy={
                "reportOnly": True,
                "decisionRecordOnly": True,
                "strictEvidenceCreated": True,
                "runtimePromotionAllowed": False,
                "parserRoutingChanged": False,
                "canonicalParsedArtifactsWritten": False,
                "databaseMutation": False,
                "reindexOrReembed": False,
                "answerIntegrationChanged": False,
            },
        ),
    )
    backlog = _write(tmp_path, "backlog.json", _backlog())

    payload = build_candidate_layer_blocker_resolution_preview(
        candidate_layer_blocker_decision_record_report=record,
        candidate_layer_blocker_backlog_report=backlog,
    )

    assert payload["status"] == "blocked"
    assert "candidate_layer_blocker_decision_record_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "decisionRecord_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_resolution_preview_writer_outputs_schema_valid_files(tmp_path: Path) -> None:
    record = _write(tmp_path / "input", "record.json", _decision_record(needs_review=2))
    backlog = _write(tmp_path / "input", "backlog.json", _backlog())
    payload = build_candidate_layer_blocker_resolution_preview(
        candidate_layer_blocker_decision_record_report=record,
        candidate_layer_blocker_backlog_report=backlog,
    )

    paths = write_candidate_layer_blocker_resolution_preview_reports(payload, tmp_path / "reports")

    assert set(paths) == {"preview", "summary", "markdown"}
    preview = json.loads(Path(paths["preview"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(preview, CANDIDATE_LAYER_BLOCKER_RESOLUTION_PREVIEW_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["stillBlockedRows"] == 1
    assert "This preview is report-only" in markdown
