from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.candidate_layer_blocker_decision_record import (
    CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID,
    build_candidate_layer_blocker_decision_record,
    write_candidate_layer_blocker_decision_record_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _template_row(
    row_id: str,
    bucket: str = "manual_decision_required",
    *,
    priority: str = "P0",
    layers: list[str] | None = None,
) -> dict:
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
        "policy_review_only": [
            "needs_review",
            "accept_policy_blocker_as_guardrail",
            "defer_policy_review",
        ],
    }[bucket]
    return {
        "decision_row_id": row_id,
        "source_review_card_id": f"review:{row_id}",
        "source_backlog_id": f"backlog:{row_id}",
        "blocker": f"blocker:{row_id}",
        "priority": priority,
        "review_bucket": bucket,
        "affected_layers": layers or ["sectionspan"],
        "affected_candidate_count": 2,
        "affected_eval_question_count": 1,
        "recommended_next_tranche": "next_tranche",
        "recommended_review_action": "review_action",
        "default_decision": "needs_review",
        "allowed_decisions": allowed,
        "manual_decision_input": {"source_review_card_id": f"review:{row_id}", "decision": "needs_review"},
        "required_review_checks": ["confirm_no_runtime_or_strict_evidence_is_authorized_by_this_row"],
        "decision_scope": "candidate_layer_blocker_decision_template_only_no_runtime_or_strict_promotion",
        "evidence_tier": "candidate_layer_blocker_decision_template_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": ["decision_not_recorded"],
        "non_strict_reason": ["decision_template_rows_are_not_human_or_operator_decisions"],
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
            "unsafeUpstreamFlagCount": 0,
        },
        "gate": {
            "decisionTemplateReady": True,
            "humanReviewComplete": False,
            "operatorApprovalComplete": False,
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


def test_blocker_decision_record_defaults_all_rows_to_needs_review(tmp_path: Path) -> None:
    template = _write(
        tmp_path,
        "template.json",
        _template(
            [
                _template_row("manual"),
                _template_row("operator", "operator_approval_required", layers=["table_region"]),
            ]
        ),
    )

    payload = build_candidate_layer_blocker_decision_record(
        candidate_layer_blocker_decision_template_report=template
    )

    assert payload["schema"] == CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID
    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "decision_record_required"
    assert payload["counts"]["recordRows"] == 2
    assert payload["counts"]["needsReviewRows"] == 2
    assert payload["counts"]["manualApprovalRows"] == 0
    assert payload["counts"]["operatorApprovedRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["allDecisionRowsComplete"] is False
    assert payload["gate"]["runtimePromotionAllowed"] is False
    assert all(row["recorded_decision"] == "needs_review" for row in payload["decisionRecords"])


def test_blocker_decision_record_records_valid_decisions_without_runtime_authority(tmp_path: Path) -> None:
    template = _write(
        tmp_path,
        "template.json",
        _template(
            [
                _template_row("manual"),
                _template_row("operator", "operator_approval_required", layers=["table_region"]),
                _template_row("technical", "technical_feasibility_blocked", layers=["equation_quote"]),
                _template_row("policy", "policy_review_only", priority="P2", layers=["figure_caption"]),
            ]
        ),
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
                },
                {"source_decision_row_id": "technical", "decision": "accept_technical_blocker_as_open"},
                {"source_decision_row_id": "policy", "decision": "accept_policy_blocker_as_guardrail"},
            ]
        },
    )

    payload = build_candidate_layer_blocker_decision_record(
        candidate_layer_blocker_decision_template_report=template,
        blocker_decisions_report=decisions,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "decision_recorded"
    assert payload["counts"]["needsReviewRows"] == 0
    assert payload["counts"]["manualApprovalRows"] == 1
    assert payload["counts"]["operatorApprovedRows"] == 1
    assert payload["counts"]["technicalAcceptedOpenRows"] == 1
    assert payload["counts"]["policyAcceptedGuardrailRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["gate"]["runtimePromotionAllowed"] is False
    operator = next(row for row in payload["decisionRecords"] if row["source_decision_row_id"] == "operator")
    assert operator["recorded_decision"] == "operator_diagnostic_action_approved_report_only"
    assert "operator_approval_is_report_only_and_does_not_execute_action" in operator["strict_blockers"]


def test_blocker_decision_record_blocks_unknown_duplicate_or_bucket_invalid_decisions(tmp_path: Path) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row("known")]))
    decisions = _write(
        tmp_path,
        "decisions.json",
        {
            "decisions": [
                {"source_decision_row_id": "known", "decision": "record_manual_approval_in_separate_decision_file"},
                {"source_decision_row_id": "unknown", "decision": "record_manual_approval_in_separate_decision_file"},
                {"source_decision_row_id": "known", "decision": "accept_technical_blocker_as_open"},
            ]
        },
    )

    payload = build_candidate_layer_blocker_decision_record(
        candidate_layer_blocker_decision_template_report=template,
        blocker_decisions_report=decisions,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "blocker_decision_unknown_template_row_id" in payload["gate"]["unsafeUpstreamFlags"]
    assert "blocker_decision_duplicate_row_id" in payload["gate"]["unsafeUpstreamFlags"]
    assert "blocker_decision_invalid_for_review_bucket" in payload["gate"]["unsafeUpstreamFlags"]


def test_blocker_decision_record_blocks_unsafe_template(tmp_path: Path) -> None:
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

    payload = build_candidate_layer_blocker_decision_record(
        candidate_layer_blocker_decision_template_report=template
    )

    assert payload["status"] == "blocked"
    assert "candidate_layer_blocker_decision_template_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "decisionTemplate_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_blocker_decision_record_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    template = _write(tmp_path / "input", "template.json", _template([_template_row("manual")]))
    payload = build_candidate_layer_blocker_decision_record(
        candidate_layer_blocker_decision_template_report=template
    )

    paths = write_candidate_layer_blocker_decision_record_reports(payload, tmp_path / "reports")

    assert set(paths) == {"record", "summary", "markdown"}
    record = json.loads(Path(paths["record"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(record, CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["needsReviewRows"] == 1
    assert "This record is report-only" in markdown
