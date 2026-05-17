from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.candidate_layer_blocker_decision_file_validation import (
    CANDIDATE_LAYER_BLOCKER_DECISION_FILE_VALIDATION_SCHEMA_ID,
    build_candidate_layer_blocker_decision_file_validation,
    write_candidate_layer_blocker_decision_file_validation_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _input_row(row_id: str, bucket: str = "manual_decision_required") -> dict:
    allowed = {
        "manual_decision_required": [
            "needs_review",
            "record_manual_approval_in_separate_decision_file",
            "record_manual_rejection_in_separate_decision_file",
            "keep_blocked",
        ],
        "operator_approval_required": [
            "needs_review",
            "approve_diagnostic_operator_action_in_separate_decision_file",
            "decline_diagnostic_operator_action_keep_blocked",
            "keep_blocked",
        ],
        "technical_feasibility_blocked": [
            "needs_review",
            "accept_technical_blocker_as_open",
            "defer_technical_followup",
            "close_as_not_needed",
        ],
    }[bucket]
    return {
        "input_row_id": f"input:{row_id}",
        "source_decision_row_id": row_id,
        "source_review_card_id": f"review:{row_id}",
        "source_backlog_id": f"backlog:{row_id}",
        "blocker": f"blocker:{row_id}",
        "priority": "P0",
        "review_bucket": bucket,
        "affected_layers": ["sectionspan"],
        "allowed_decisions": allowed,
        "decision": "needs_review",
        "reviewer": "",
        "notes": "",
        "decision_scope": "candidate_layer_blocker_decision_input_pack_only_no_runtime_or_strict_promotion",
        "evidence_tier": "candidate_layer_blocker_decision_input_pack_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
    }


def _input_pack(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-decision-input-pack.v1",
        "status": "decision_input_pack_ready",
        "counts": {
            "inputRows": len(rows),
            "pendingSourceDecisionRows": len(rows),
            "defaultNeedsReviewRows": len(rows),
            "acceptedDecisionRows": 0,
            "operatorApprovedRows": 0,
            "strictEligibleRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "decisionInputPackReady": True,
            "containsAcceptedDecisions": False,
            "containsOperatorApprovals": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "decisionInputPackOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "decisionInputs": rows,
    }
    payload.update(overrides)
    return payload


def test_validation_reports_missing_decision_file_without_recording_decisions(tmp_path: Path) -> None:
    input_pack = _write(tmp_path, "input-pack.json", _input_pack([_input_row("manual")]))

    payload = build_candidate_layer_blocker_decision_file_validation(
        candidate_layer_blocker_decision_input_pack_report=input_pack
    )

    assert payload["schema"] == CANDIDATE_LAYER_BLOCKER_DECISION_FILE_VALIDATION_SCHEMA_ID
    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_DECISION_FILE_VALIDATION_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "decision_file_required"
    assert payload["counts"]["missingRows"] == 1
    assert payload["counts"]["acceptedDecisionRows"] == 0
    assert payload["counts"]["operatorApprovedRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["containsRecordedDecisions"] is False
    assert payload["gate"]["runtimePromotionAllowed"] is False


def test_validation_accepts_complete_file_as_non_runtime_validation_only(tmp_path: Path) -> None:
    input_pack = _write(
        tmp_path,
        "input-pack.json",
        _input_pack([_input_row("manual"), _input_row("operator", "operator_approval_required")]),
    )
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "decisions": [
                {
                    "source_decision_row_id": "manual",
                    "decision": "record_manual_approval_in_separate_decision_file",
                    "reviewer": "reviewer",
                },
                {
                    "source_decision_row_id": "operator",
                    "decision": "approve_diagnostic_operator_action_in_separate_decision_file",
                    "reviewer": "operator",
                },
            ]
        },
    )

    payload = build_candidate_layer_blocker_decision_file_validation(
        candidate_layer_blocker_decision_input_pack_report=input_pack,
        candidate_layer_blocker_decisions_file=decisions,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_DECISION_FILE_VALIDATION_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "decision_file_validated"
    assert payload["counts"]["validRows"] == 2
    assert payload["counts"]["nonNeedsReviewRows"] == 2
    assert payload["counts"]["acceptedDecisionRows"] == 0
    assert payload["counts"]["operatorApprovedRows"] == 0
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert all(row["validation_status"] == "valid" for row in payload["validationRows"])


def test_validation_blocks_unknown_duplicate_or_bucket_invalid_decisions(tmp_path: Path) -> None:
    input_pack = _write(tmp_path, "input-pack.json", _input_pack([_input_row("manual")]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "decisions": [
                {"source_decision_row_id": "manual", "decision": "accept_technical_blocker_as_open"},
                {"source_decision_row_id": "manual", "decision": "needs_review"},
                {"source_decision_row_id": "unknown", "decision": "needs_review"},
            ]
        },
    )

    payload = build_candidate_layer_blocker_decision_file_validation(
        candidate_layer_blocker_decision_input_pack_report=input_pack,
        candidate_layer_blocker_decisions_file=decisions,
    )

    assert payload["status"] == "blocked"
    assert "decision_file_duplicate_row_id" in payload["gate"]["fileValidationErrors"]
    assert "decision_file_unknown_input_row_id" in payload["gate"]["fileValidationErrors"]
    manual = payload["validationRows"][0]
    assert manual["validation_status"] == "invalid"
    assert "decision_not_allowed_for_review_bucket" in manual["validation_errors"]


def test_validation_requires_reviewer_for_non_needs_review_decisions(tmp_path: Path) -> None:
    input_pack = _write(tmp_path, "input-pack.json", _input_pack([_input_row("manual")]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        {"decisions": [{"source_decision_row_id": "manual", "decision": "record_manual_approval_in_separate_decision_file"}]},
    )

    payload = build_candidate_layer_blocker_decision_file_validation(
        candidate_layer_blocker_decision_input_pack_report=input_pack,
        candidate_layer_blocker_decisions_file=decisions,
    )

    assert payload["status"] == "decision_file_incomplete"
    assert payload["counts"]["invalidRows"] == 1
    assert "reviewer_required_for_non_needs_review_decision" in payload["validationRows"][0]["validation_errors"]


def test_validation_blocks_unsafe_input_pack(tmp_path: Path) -> None:
    input_pack = _write(
        tmp_path,
        "input-pack.json",
        _input_pack(
            [_input_row("manual")],
            schema="example.wrong.input-pack.v1",
            policy={
                "reportOnly": True,
                "decisionInputPackOnly": True,
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

    payload = build_candidate_layer_blocker_decision_file_validation(
        candidate_layer_blocker_decision_input_pack_report=input_pack
    )

    assert payload["status"] == "blocked"
    assert "candidate_layer_blocker_decision_input_pack_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "decisionInputPack_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_validation_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    input_pack = _write(tmp_path / "input", "input-pack.json", _input_pack([_input_row("manual")]))
    payload = build_candidate_layer_blocker_decision_file_validation(
        candidate_layer_blocker_decision_input_pack_report=input_pack
    )

    paths = write_candidate_layer_blocker_decision_file_validation_reports(payload, tmp_path / "reports")

    assert set(paths) == {"validation", "summary", "markdown"}
    validation = json.loads(Path(paths["validation"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(validation, CANDIDATE_LAYER_BLOCKER_DECISION_FILE_VALIDATION_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["missingRows"] == 1
    assert "This validation report is report-only" in markdown
