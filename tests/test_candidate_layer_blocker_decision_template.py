from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.candidate_layer_blocker_decision_template import (
    CANDIDATE_LAYER_BLOCKER_DECISION_TEMPLATE_SCHEMA_ID,
    build_candidate_layer_blocker_decision_template,
    write_candidate_layer_blocker_decision_template_reports,
)


def _write(root: Path, payload: dict) -> Path:
    path = root / "candidate-layer-blocker-review-cards.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _card(card_id: str, bucket: str, *, priority: str = "P0", layers: list[str] | None = None) -> dict:
    return {
        "review_card_id": f"review:{card_id}",
        "source_backlog_id": f"backlog:{card_id}",
        "blocker": card_id,
        "priority": priority,
        "review_bucket": bucket,
        "affected_layers": layers or ["sectionspan"],
        "affected_candidate_count": 2,
        "affected_eval_question_count": 1,
        "recommended_next_tranche": "next_tranche",
        "recommended_review_action": "review_action",
        "requires_human_decision": bucket == "manual_decision_required",
        "requires_operator_approval": bucket == "operator_approval_required",
        "requires_technical_followup": bucket == "technical_feasibility_blocked",
        "policy_only": bucket == "policy_review_only",
        "allowed_actions": ["report_only_audit"],
        "disallowed_actions": ["strict_evidence_promotion", "parser_routing", "database_mutation"],
        "stop_rule": "stop_if_needed",
        "evidence_needed_before_promotion": ["evidence needed"],
        "evidence_tier": "candidate_layer_blocker_review_card_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": [card_id],
        "non_strict_reason": ["review_cards_are_not_evidence"],
    }


def _review_pack(cards: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-review-pack.v1",
        "status": "review_pack_ready",
        "counts": {
            "reviewCardCount": len(cards),
            "manualDecisionRequiredCards": sum(1 for card in cards if card["review_bucket"] == "manual_decision_required"),
            "operatorApprovalRequiredCards": sum(1 for card in cards if card["review_bucket"] == "operator_approval_required"),
            "technicalFeasibilityBlockedCards": sum(1 for card in cards if card["review_bucket"] == "technical_feasibility_blocked"),
            "policyReviewOnlyCards": sum(1 for card in cards if card["review_bucket"] == "policy_review_only"),
            "strictEligibleCards": 0,
            "citationGradeCards": 0,
            "runtimeEvidenceCards": 0,
        },
        "gate": {
            "reviewPackReady": True,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "schemaViolations": [],
        },
        "policy": {
            "reviewPackOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "reviewCards": cards,
    }
    payload.update(overrides)
    return payload


def test_blocker_decision_template_emits_pending_rows_without_accepting_decisions(tmp_path: Path) -> None:
    review_pack = _write(
        tmp_path,
        _review_pack(
            [
                _card("manual", "manual_decision_required"),
                _card("operator", "operator_approval_required", layers=["table_region"]),
                _card("technical", "technical_feasibility_blocked", layers=["equation_quote"]),
                _card("policy", "policy_review_only", priority="P2", layers=["sectionspan", "figure_caption"]),
            ]
        ),
    )

    payload = build_candidate_layer_blocker_decision_template(
        candidate_layer_blocker_review_pack_report=review_pack
    )

    assert payload["schema"] == CANDIDATE_LAYER_BLOCKER_DECISION_TEMPLATE_SCHEMA_ID
    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_DECISION_TEMPLATE_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "decision_template_ready"
    assert payload["counts"]["templateRows"] == 4
    assert payload["counts"]["pendingDecisionRows"] == 4
    assert payload["counts"]["acceptedDecisionRows"] == 0
    assert payload["counts"]["operatorApprovedRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["humanReviewComplete"] is False
    assert payload["gate"]["operatorApprovalComplete"] is False
    assert payload["gate"]["runtimePromotionAllowed"] is False
    assert all(row["default_decision"] == "needs_review" for row in payload["decisionRows"])
    assert all(row["strict_eligible"] is False for row in payload["decisionRows"])


def test_template_decision_options_depend_on_review_bucket(tmp_path: Path) -> None:
    review_pack = _write(
        tmp_path,
        _review_pack(
            [
                _card("manual", "manual_decision_required"),
                _card("operator", "operator_approval_required", layers=["table_region"]),
                _card("technical", "technical_feasibility_blocked", layers=["equation_quote"]),
                _card("policy", "policy_review_only", priority="P2"),
            ]
        ),
    )

    payload = build_candidate_layer_blocker_decision_template(
        candidate_layer_blocker_review_pack_report=review_pack
    )

    rows = {row["review_bucket"]: row for row in payload["decisionRows"]}
    assert "record_manual_approval_in_separate_decision_file" in rows["manual_decision_required"]["allowed_decisions"]
    assert "approve_diagnostic_operator_action_in_separate_decision_file" in rows["operator_approval_required"]["allowed_decisions"]
    assert "accept_technical_blocker_as_open" in rows["technical_feasibility_blocked"]["allowed_decisions"]
    assert "accept_policy_blocker_as_guardrail" in rows["policy_review_only"]["allowed_decisions"]
    assert rows["operator_approval_required"]["runtime_evidence"] is False


def test_blocker_decision_template_blocks_unsafe_review_pack(tmp_path: Path) -> None:
    review_pack = _write(
        tmp_path,
        _review_pack(
            [_card("manual", "manual_decision_required")],
            schema="example.wrong.review-pack.v1",
            policy={
                "reviewPackOnly": True,
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

    payload = build_candidate_layer_blocker_decision_template(
        candidate_layer_blocker_review_pack_report=review_pack
    )

    assert payload["status"] == "blocked"
    assert "candidate_layer_blocker_review_pack_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "reviewPack_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]
    assert payload["counts"]["acceptedDecisionRows"] == 0


def test_blocker_decision_template_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    review_pack = _write(tmp_path / "input", _review_pack([_card("manual", "manual_decision_required")]))
    payload = build_candidate_layer_blocker_decision_template(
        candidate_layer_blocker_review_pack_report=review_pack
    )

    paths = write_candidate_layer_blocker_decision_template_reports(payload, tmp_path / "reports")

    assert set(paths) == {"template", "summary", "markdown"}
    template = json.loads(Path(paths["template"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(template, CANDIDATE_LAYER_BLOCKER_DECISION_TEMPLATE_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["pendingDecisionRows"] == 1
    assert "This template is a worksheet only" in markdown
