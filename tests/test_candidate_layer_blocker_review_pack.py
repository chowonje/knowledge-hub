from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.candidate_layer_blocker_review_pack import (
    CANDIDATE_LAYER_BLOCKER_REVIEW_PACK_SCHEMA_ID,
    build_candidate_layer_blocker_review_pack,
    write_candidate_layer_blocker_review_pack_reports,
)


def _write(root: Path, payload: dict) -> Path:
    path = root / "candidate-layer-blocker-backlog.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _backlog_item(blocker: str, *, priority: str = "P0", layers: list[str] | None = None) -> dict:
    return {
        "backlog_id": f"candidate-layer-blocker-v1-{blocker}",
        "blocker": blocker,
        "status": "open",
        "priority": priority,
        "category": "test",
        "affected_layers": layers or ["sectionspan"],
        "affected_candidate_count": 3,
        "affected_eval_question_count": 2,
        "evidenceNeededBeforePromotion": ["evidence needed"],
        "recommendedNextTranche": "next_tranche",
        "allowedActions": ["report_only_audit"],
        "disallowedActions": ["strict_evidence_promotion", "parser_routing", "database_mutation"],
        "stopRule": "stop_if_required",
        "evidence_tier": "candidate_backlog_only",
        "strict_eligible": False,
        "strict_blockers": [blocker],
        "non_strict_reason": ["backlog_items_are_not_evidence"],
    }


def _backlog_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-backlog.v1",
        "status": "ok",
        "counts": {
            "backlogItemCount": 5,
            "strictEligibleCandidates": 0,
            "citationGradeCandidates": 0,
            "runtimeEvidenceCandidates": 0,
        },
        "gate": {
            "decision": "blocker_backlog_ready",
            "schemaViolations": [],
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
            _backlog_item("sectionspan_selected_review_decision_file_required", layers=["sectionspan"]),
            _backlog_item("table_cell_isolated_extractor_approval_required", layers=["table_region"]),
            _backlog_item("equation_quote_alignment_missing", layers=["equation_quote"]),
            _backlog_item("figure_region_link_unverified", priority="P1", layers=["figure_caption"]),
            _backlog_item("candidate_layers_are_report_only", priority="P2", layers=["sectionspan", "table_region"]),
        ],
    }
    payload.update(overrides)
    return payload


def test_blocker_review_pack_classifies_cards_without_promoting_evidence(tmp_path: Path) -> None:
    backlog = _write(tmp_path, _backlog_payload())

    payload = build_candidate_layer_blocker_review_pack(candidate_layer_blocker_backlog_report=backlog)

    assert payload["schema"] == CANDIDATE_LAYER_BLOCKER_REVIEW_PACK_SCHEMA_ID
    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "review_pack_ready"
    assert payload["counts"]["reviewCardCount"] == 5
    assert payload["counts"]["manualDecisionRequiredCards"] == 1
    assert payload["counts"]["operatorApprovalRequiredCards"] == 1
    assert payload["counts"]["technicalFeasibilityBlockedCards"] == 2
    assert payload["counts"]["policyReviewOnlyCards"] == 1
    assert payload["counts"]["strictEligibleCards"] == 0
    assert payload["counts"]["citationGradeCards"] == 0
    assert payload["counts"]["runtimeEvidenceCards"] == 0
    assert payload["gate"]["manualDecisionRequired"] is True
    assert payload["gate"]["operatorApprovalRequired"] is True
    assert payload["gate"]["strictEvidenceReady"] is False
    assert payload["gate"]["parserRoutingReady"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["databaseMutation"] is False
    assert all(card["strict_eligible"] is False for card in payload["reviewCards"])
    assert all(card["evidence_tier"] == "candidate_layer_blocker_review_card_only" for card in payload["reviewCards"])


def test_manual_and_operator_cards_are_not_accepted_decisions(tmp_path: Path) -> None:
    backlog = _write(tmp_path, _backlog_payload())

    payload = build_candidate_layer_blocker_review_pack(candidate_layer_blocker_backlog_report=backlog)

    manual = next(card for card in payload["reviewCards"] if card["review_bucket"] == "manual_decision_required")
    operator = next(card for card in payload["reviewCards"] if card["review_bucket"] == "operator_approval_required")
    assert manual["requires_human_decision"] is True
    assert manual["recommended_review_action"] == "provide_selected_sectionspan_decision_file_or_keep_pending"
    assert manual["runtime_evidence"] is False
    assert operator["requires_operator_approval"] is True
    assert operator["recommended_review_action"] == "approve_or_decline_isolated_table_cell_extractor_pilot"
    assert "strict_evidence_promotion" in operator["disallowed_actions"]


def test_latest_manual_edit_blockers_are_classified_as_manual_decisions(tmp_path: Path) -> None:
    backlog = _write(
        tmp_path,
        _backlog_payload(
            backlog=[
                _backlog_item("sectionspan_selected_review_manual_edit_required", layers=["sectionspan"]),
                _backlog_item("equation_quote_decision_manual_edit_required", layers=["equation_quote"]),
                _backlog_item(
                    "candidate_layer_blocker_decision_record_pending",
                    layers=["sectionspan", "figure_caption", "equation_quote", "table_region"],
                ),
            ],
        ),
    )

    payload = build_candidate_layer_blocker_review_pack(candidate_layer_blocker_backlog_report=backlog)

    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["reviewCardCount"] == 3
    assert payload["counts"]["manualDecisionRequiredCards"] == 3
    assert payload["counts"]["technicalFeasibilityBlockedCards"] == 0
    actions = {card["blocker"]: card["recommended_review_action"] for card in payload["reviewCards"]}
    assert (
        actions["sectionspan_selected_review_manual_edit_required"]
        == "manually_edit_selected_sectionspan_decision_file_or_keep_pending"
    )
    assert (
        actions["equation_quote_decision_manual_edit_required"]
        == "manually_edit_equation_quote_decision_file_or_keep_pending"
    )
    assert (
        actions["candidate_layer_blocker_decision_record_pending"]
        == "record_candidate_layer_blocker_decisions_or_keep_pending"
    )
    assert all(card["requires_human_decision"] is True for card in payload["reviewCards"])
    assert all(card["runtime_evidence"] is False for card in payload["reviewCards"])


def test_blocker_review_pack_blocks_wrong_or_failed_backlog(tmp_path: Path) -> None:
    backlog = _write(
        tmp_path,
        _backlog_payload(
            schema="example.wrong.backlog.v1",
            status="blocked",
            gate={"schemaViolations": ["upstream_bad"]},
            backlog=[],
        ),
    )

    payload = build_candidate_layer_blocker_review_pack(candidate_layer_blocker_backlog_report=backlog)

    assert payload["status"] == "blocked"
    assert payload["gate"]["reviewPackReady"] is False
    assert "candidate_layer_blocker_backlog_schema_mismatch" in payload["gate"]["schemaViolations"]
    assert "candidate_layer_blocker_backlog_not_ok" in payload["gate"]["schemaViolations"]
    assert "candidate_layer_blocker_backlog_upstream:upstream_bad" in payload["gate"]["schemaViolations"]


def test_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    backlog = _write(tmp_path, _backlog_payload())
    payload = build_candidate_layer_blocker_review_pack(candidate_layer_blocker_backlog_report=backlog)

    paths = write_candidate_layer_blocker_review_pack_reports(payload, tmp_path / "reports")

    assert set(paths) == {"cards", "summary", "markdown"}
    cards = json.loads(Path(paths["cards"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(cards, CANDIDATE_LAYER_BLOCKER_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["reviewCardCount"] == 5
    assert "Candidate Layer Blocker Review Pack" in markdown
