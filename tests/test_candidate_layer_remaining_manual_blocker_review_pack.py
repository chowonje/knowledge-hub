from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.candidate_layer_remaining_manual_blocker_review_pack import (
    CANDIDATE_LAYER_REMAINING_MANUAL_BLOCKER_REVIEW_PACK_SCHEMA_ID,
    build_candidate_layer_remaining_manual_blocker_review_pack,
    write_candidate_layer_remaining_manual_blocker_review_pack_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _record_row(
    row_id: str,
    blocker: str,
    *,
    recorded_decision: str = "needs_review",
    bucket: str = "manual_decision_required",
    layers: list[str] | None = None,
) -> dict:
    return {
        "record_row_id": f"candidate-layer-blocker-decision-record:{row_id}",
        "source_decision_row_id": f"candidate-layer-blocker-decision:{row_id}",
        "source_review_card_id": f"candidate-layer-blocker-review:{row_id}",
        "source_backlog_id": f"candidate-layer-blocker-v1-{row_id}",
        "blocker": blocker,
        "priority": "P0",
        "review_bucket": bucket,
        "affected_layers": layers or ["sectionspan"],
        "affected_candidate_count": 1,
        "affected_eval_question_count": 0,
        "recommended_next_tranche": "manual_next_step",
        "recommended_review_action": "manual_review_action",
        "raw_decision": "needs_review",
        "recorded_decision": recorded_decision,
        "decision_scope": "candidate_layer_blocker_decision_record_only_no_runtime_or_strict_promotion",
        "reviewer": "",
        "notes": "",
        "evidence_tier": "candidate_layer_blocker_decision_record_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": ["candidate_layer_blocker_decision_missing"],
        "non_strict_reason": ["decision_record_rows_are_review_metadata_only"],
    }


def _decision_record(rows: list[dict], **overrides: object) -> dict:
    needs_review = sum(1 for row in rows if row["recorded_decision"] == "needs_review")
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-decision-record.v1",
        "status": "decision_record_required" if needs_review else "decision_recorded",
        "counts": {
            "recordRows": len(rows),
            "needsReviewRows": needs_review,
            "technicalAcceptedOpenRows": 1,
            "policyAcceptedGuardrailRows": 1,
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
        "decisionRecords": rows,
    }
    payload.update(overrides)
    return payload


def _resolution_preview(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-resolution-preview.v1",
        "status": "resolution_preview_ready",
        "counts": {
            "stillBlockedRows": 1,
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
            "resolutionPreviewOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "previewRows": [
            {
                "source_backlog_id": "candidate-layer-blocker-v1-0012",
                "blocker": "candidate_layer_blocker_decision_record_pending",
                "preview_status": "still_blocked",
                "preview_reason": "decision_record_rows_still_need_review",
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_remaining_manual_blocker_review_pack_extracts_needs_review_rows(tmp_path: Path) -> None:
    record = _write(
        tmp_path,
        "record.json",
        _decision_record(
            [
                _record_row("0004", "sectionspan_pdf_offsets_require_human_review_before_strict_promotion"),
                _record_row("0010", "sectionspan_selected_review_manual_edit_required"),
                _record_row("0011", "equation_quote_decision_manual_edit_required", layers=["equation_quote"]),
                _record_row(
                    "0012",
                    "candidate_layer_blocker_decision_record_pending",
                    layers=["sectionspan", "figure_caption", "equation_quote", "table_region"],
                ),
                _record_row(
                    "0001",
                    "equation_quote_alignment_missing",
                    recorded_decision="technical_blocker_accepted_open",
                    bucket="technical_feasibility_blocked",
                    layers=["equation_quote"],
                ),
            ]
        ),
    )
    preview = _write(tmp_path, "preview.json", _resolution_preview())

    payload = build_candidate_layer_remaining_manual_blocker_review_pack(
        candidate_layer_blocker_decision_record_report=record,
        candidate_layer_blocker_resolution_preview_report=preview,
    )

    assert payload["schema"] == CANDIDATE_LAYER_REMAINING_MANUAL_BLOCKER_REVIEW_PACK_SCHEMA_ID
    assert validate_payload(payload, CANDIDATE_LAYER_REMAINING_MANUAL_BLOCKER_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "manual_review_required"
    assert payload["counts"]["reviewRows"] == 4
    assert payload["counts"]["remainingNeedsReviewRows"] == 4
    assert payload["counts"]["manualDecisionRows"] == 4
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["runtimePromotionAllowed"] is False
    assert payload["policy"]["answerIntegrationChanged"] is False
    blockers = {row["blocker"] for row in payload["reviewRows"]}
    assert "equation_quote_alignment_missing" not in blockers
    assert "candidate_layer_blocker_decision_record_pending" in blockers
    pending = next(row for row in payload["reviewRows"] if row["blocker"] == "candidate_layer_blocker_decision_record_pending")
    assert pending["resolution_preview_status"] == "still_blocked"
    assert pending["decision_input_hint"]["decision"] == "needs_review"
    assert pending["strict_eligible"] is False
    assert "record_manual_approval_in_separate_decision_file" in pending["allowed_decisions"]


def test_remaining_manual_blocker_review_pack_is_report_only_when_no_rows_remain(tmp_path: Path) -> None:
    record = _write(
        tmp_path,
        "record.json",
        _decision_record(
            [
                _record_row(
                    "0001",
                    "equation_quote_alignment_missing",
                    recorded_decision="technical_blocker_accepted_open",
                    bucket="technical_feasibility_blocked",
                    layers=["equation_quote"],
                )
            ]
        ),
    )

    payload = build_candidate_layer_remaining_manual_blocker_review_pack(
        candidate_layer_blocker_decision_record_report=record
    )

    assert validate_payload(payload, CANDIDATE_LAYER_REMAINING_MANUAL_BLOCKER_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "no_remaining_manual_blockers"
    assert payload["counts"]["remainingNeedsReviewRows"] == 0
    assert payload["gate"]["allDecisionRowsComplete"] is True
    assert payload["gate"]["strictEvidenceReady"] is False


def test_remaining_manual_blocker_review_pack_blocks_unsafe_inputs(tmp_path: Path) -> None:
    record = _write(
        tmp_path,
        "record.json",
        _decision_record(
            [_record_row("0004", "sectionspan_pdf_offsets_require_human_review_before_strict_promotion")],
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

    payload = build_candidate_layer_remaining_manual_blocker_review_pack(
        candidate_layer_blocker_decision_record_report=record
    )

    assert payload["status"] == "blocked"
    assert "candidate_layer_blocker_decision_record_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "decisionRecord_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]
    assert payload["gate"]["runtimePromotionAllowed"] is False


def test_remaining_manual_blocker_review_pack_writer_outputs_schema_valid_files(tmp_path: Path) -> None:
    record = _write(
        tmp_path / "input",
        "record.json",
        _decision_record(
            [_record_row("0010", "sectionspan_selected_review_manual_edit_required")]
        ),
    )
    payload = build_candidate_layer_remaining_manual_blocker_review_pack(
        candidate_layer_blocker_decision_record_report=record
    )

    paths = write_candidate_layer_remaining_manual_blocker_review_pack_reports(payload, tmp_path / "reports")

    assert set(paths) == {"reviewPack", "summary", "markdown"}
    review_pack = json.loads(Path(paths["reviewPack"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(review_pack, CANDIDATE_LAYER_REMAINING_MANUAL_BLOCKER_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["remainingNeedsReviewRows"] == 1
    assert "This pack is report-only" in markdown
