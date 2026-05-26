from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.candidate_layer_blocker_manual_decision_review_sheet import (
    CANDIDATE_LAYER_BLOCKER_MANUAL_DECISION_REVIEW_SHEET_SCHEMA_ID,
    build_candidate_layer_blocker_manual_decision_review_sheet,
    write_candidate_layer_blocker_manual_decision_review_sheet_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _template_row(row_id: str, bucket: str = "manual_decision_required") -> dict:
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
    }[bucket]
    return {
        "decision_row_id": row_id,
        "source_review_card_id": f"review:{row_id}",
        "source_backlog_id": f"backlog:{row_id}",
        "blocker": f"blocker:{row_id}",
        "priority": "P0",
        "review_bucket": bucket,
        "affected_layers": ["sectionspan"],
        "affected_candidate_count": 2,
        "affected_eval_question_count": 1,
        "recommended_next_tranche": "next_tranche",
        "recommended_review_action": "review_action",
        "allowed_decisions": allowed,
        "required_review_checks": ["confirm_no_runtime_or_strict_evidence_is_authorized_by_this_row"],
    }


def _template(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-decision-template.v1",
        "status": "decision_template_ready",
        "counts": {
            "templateRows": len(rows),
            "pendingDecisionRows": len(rows),
            "acceptedDecisionRows": 0,
            "rejectedDecisionRows": 0,
            "operatorApprovedRows": 0,
            "operatorDeclinedRows": 0,
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
            "decisionTemplateOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "decisionRows": rows,
    }
    payload.update(overrides)
    return payload


def _record(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-decision-record.v1",
        "status": "decision_record_required",
        "counts": {
            "recordRows": len(rows),
            "needsReviewRows": len(rows),
            "manualApprovalRows": 0,
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
        "decisionRecords": rows,
    }
    payload.update(overrides)
    return payload


def _record_row(row_id: str, decision: str = "needs_review") -> dict:
    return {
        "source_decision_row_id": row_id,
        "recorded_decision": decision,
    }


def test_manual_decision_review_sheet_consolidates_needs_review_rows(tmp_path: Path) -> None:
    template = _write(
        tmp_path,
        "template.json",
        _template(
            [
                _template_row("manual"),
                _template_row("operator", "operator_approval_required"),
            ]
        ),
    )
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "decisions": [
                {"source_decision_row_id": "manual", "decision": "needs_review", "reviewer": "", "notes": ""},
                {"source_decision_row_id": "operator", "decision": "needs_review", "reviewer": "", "notes": ""},
            ]
        },
    )
    record = _write(tmp_path, "record.json", _record([_record_row("manual"), _record_row("operator")]))

    payload = build_candidate_layer_blocker_manual_decision_review_sheet(
        candidate_layer_blocker_decision_template_report=template,
        candidate_layer_blocker_decisions_file=decisions,
        candidate_layer_blocker_decision_record_report=record,
    )

    assert payload["schema"] == CANDIDATE_LAYER_BLOCKER_MANUAL_DECISION_REVIEW_SHEET_SCHEMA_ID
    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_MANUAL_DECISION_REVIEW_SHEET_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "manual_review_sheet_ready"
    assert payload["counts"]["reviewRows"] == 2
    assert payload["counts"]["needsReviewRows"] == 2
    assert payload["counts"]["nonNeedsReviewRows"] == 0
    assert payload["counts"]["manualBucketRows"] == 1
    assert payload["counts"]["operatorBucketRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["containsNonNeedsReviewDraftValues"] is False
    assert payload["gate"]["runtimePromotionAllowed"] is False
    assert all(row["evidence_tier"] == "candidate_layer_blocker_manual_decision_review_sheet_only" for row in payload["reviewRows"])


def test_manual_decision_review_sheet_does_not_promote_non_needs_review_draft_values(tmp_path: Path) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row("manual")]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "decisions": [
                {
                    "source_decision_row_id": "manual",
                    "decision": "record_manual_approval_in_separate_decision_file",
                    "reviewer": "reviewer",
                    "notes": "reviewed",
                }
            ]
        },
    )
    record = _write(
        tmp_path,
        "record.json",
        _record([_record_row("manual", "manual_approval_recorded_for_later_design_only")]),
    )

    payload = build_candidate_layer_blocker_manual_decision_review_sheet(
        candidate_layer_blocker_decision_template_report=template,
        candidate_layer_blocker_decisions_file=decisions,
        candidate_layer_blocker_decision_record_report=record,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_MANUAL_DECISION_REVIEW_SHEET_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["needsReviewRows"] == 0
    assert payload["counts"]["nonNeedsReviewRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["counts"]["citationGradeRows"] == 0
    assert payload["counts"]["runtimeEvidenceRows"] == 0
    assert payload["gate"]["containsNonNeedsReviewDraftValues"] is True
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["answerIntegrationChanged"] is False
    row = payload["reviewRows"][0]
    assert row["current_decision"] == "record_manual_approval_in_separate_decision_file"
    assert row["strict_eligible"] is False
    assert "manual_review_sheet_rows_are_not_decisions" in row["non_strict_reason"]


def test_manual_decision_review_sheet_blocks_unsafe_inputs(tmp_path: Path) -> None:
    template = _write(
        tmp_path,
        "template.json",
        _template(
            [_template_row("manual")],
            schema="example.wrong.template.v1",
            policy={
                "reportOnly": True,
                "decisionTemplateOnly": True,
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
    decisions = _write(tmp_path, "decisions.json", {"decisions": []})

    payload = build_candidate_layer_blocker_manual_decision_review_sheet(
        candidate_layer_blocker_decision_template_report=template,
        candidate_layer_blocker_decisions_file=decisions,
    )

    assert payload["status"] == "blocked"
    assert "candidate_layer_blocker_decision_template_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "decisionTemplate_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_manual_decision_review_sheet_writer_outputs_schema_valid_files(tmp_path: Path) -> None:
    template = _write(tmp_path / "input", "template.json", _template([_template_row("manual")]))
    decisions = _write(
        tmp_path / "input",
        "decisions.json",
        {"decisions": [{"source_decision_row_id": "manual", "decision": "needs_review"}]},
    )
    payload = build_candidate_layer_blocker_manual_decision_review_sheet(
        candidate_layer_blocker_decision_template_report=template,
        candidate_layer_blocker_decisions_file=decisions,
    )

    paths = write_candidate_layer_blocker_manual_decision_review_sheet_reports(payload, tmp_path / "reports")

    assert set(paths) == {"sheet", "summary", "markdown"}
    sheet = json.loads(Path(paths["sheet"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(sheet, CANDIDATE_LAYER_BLOCKER_MANUAL_DECISION_REVIEW_SHEET_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["reviewRows"] == 1
    assert "This sheet is local review metadata only" in markdown
