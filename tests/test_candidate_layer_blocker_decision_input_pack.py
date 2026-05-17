from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.candidate_layer_blocker_decision_input_pack import (
    CANDIDATE_LAYER_BLOCKER_DECISION_INPUT_PACK_SCHEMA_ID,
    build_candidate_layer_blocker_decision_input_pack,
    write_candidate_layer_blocker_decision_input_pack_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _template_row(row_id: str, bucket: str = "manual_decision_required") -> dict:
    allowed_by_bucket = {
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
    }
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
        "default_decision": "needs_review",
        "allowed_decisions": allowed_by_bucket[bucket],
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
            "operatorApprovedRows": 0,
            "strictEligibleRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "decisionTemplateReady": True,
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
            "needsReviewRows": sum(1 for row in rows if row["recorded_decision"] == "needs_review"),
            "manualApprovalRows": 0,
            "operatorApprovedRows": 0,
            "strictEligibleRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "decisionRecordReady": True,
            "allDecisionRowsComplete": False,
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


def _record_row(source_id: str, decision: str = "needs_review") -> dict:
    return {
        "record_row_id": f"record:{source_id}",
        "source_decision_row_id": source_id,
        "recorded_decision": decision,
        "strict_eligible": False,
        "runtime_evidence": False,
    }


def test_decision_input_pack_defaults_pending_rows_to_needs_review(tmp_path: Path) -> None:
    template = _write(
        tmp_path,
        "template.json",
        _template([_template_row("manual"), _template_row("operator", "operator_approval_required")]),
    )

    payload = build_candidate_layer_blocker_decision_input_pack(
        candidate_layer_blocker_decision_template_report=template
    )

    assert payload["schema"] == CANDIDATE_LAYER_BLOCKER_DECISION_INPUT_PACK_SCHEMA_ID
    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_DECISION_INPUT_PACK_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "decision_input_pack_ready"
    assert payload["counts"]["inputRows"] == 2
    assert payload["counts"]["defaultNeedsReviewRows"] == 2
    assert payload["counts"]["acceptedDecisionRows"] == 0
    assert payload["counts"]["operatorApprovedRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["containsAcceptedDecisions"] is False
    assert payload["gate"]["runtimePromotionAllowed"] is False
    assert all(row["decision"] == "needs_review" for row in payload["decisionInputs"])


def test_decision_input_pack_emits_only_record_pending_rows(tmp_path: Path) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row("pending"), _template_row("done")]))
    record = _write(
        tmp_path,
        "record.json",
        _record(
            [
                _record_row("pending", "needs_review"),
                _record_row("done", "manual_approval_recorded_for_later_design_only"),
            ]
        ),
    )

    payload = build_candidate_layer_blocker_decision_input_pack(
        candidate_layer_blocker_decision_template_report=template,
        candidate_layer_blocker_decision_record_report=record,
    )

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_DECISION_INPUT_PACK_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["inputRows"] == 1
    assert payload["decisionInputs"][0]["source_decision_row_id"] == "pending"
    assert payload["decisionInputs"][0]["decision"] == "needs_review"
    assert payload["counts"]["acceptedDecisionRows"] == 0


def test_decision_input_pack_blocks_unsafe_template_or_record(tmp_path: Path) -> None:
    template = _write(tmp_path, "template.json", _template([_template_row("manual")], schema="example.wrong.template"))
    record = _write(
        tmp_path,
        "record.json",
        _record(
            [_record_row("manual")],
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

    payload = build_candidate_layer_blocker_decision_input_pack(
        candidate_layer_blocker_decision_template_report=template,
        candidate_layer_blocker_decision_record_report=record,
    )

    assert payload["status"] == "blocked"
    assert "candidate_layer_blocker_decision_template_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "decisionRecord_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_decision_input_pack_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    template = _write(tmp_path / "input", "template.json", _template([_template_row("manual")]))
    payload = build_candidate_layer_blocker_decision_input_pack(
        candidate_layer_blocker_decision_template_report=template
    )

    paths = write_candidate_layer_blocker_decision_input_pack_reports(payload, tmp_path / "reports")

    assert set(paths) == {"inputPack", "summary", "markdown"}
    pack = json.loads(Path(paths["inputPack"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(pack, CANDIDATE_LAYER_BLOCKER_DECISION_INPUT_PACK_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["inputRows"] == 1
    assert "This input pack is a worksheet source only" in markdown
