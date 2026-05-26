from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.candidate_layer_blocker_decision_edit_plan import (
    CANDIDATE_LAYER_BLOCKER_DECISION_EDIT_PLAN_SCHEMA_ID,
    build_candidate_layer_blocker_decision_edit_plan,
    main,
    write_candidate_layer_blocker_decision_edit_plan_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _decision_file(rows: list[dict]) -> dict:
    return {"draftOnly": True, "instructions": ["manual review required"], "decisions": rows}


def _decision_row(index: int, **overrides: object) -> dict:
    payload = {
        "source_decision_row_id": f"candidate-layer-blocker-decision:{index:04d}",
        "decision": "needs_review",
        "reviewer": "",
        "notes": "",
        "allowed_decisions": [
            "needs_review",
            "accept_technical_blocker_as_open",
            "accept_policy_blocker_as_guardrail",
            "record_manual_approval_in_separate_decision_file",
            "record_manual_rejection_in_separate_decision_file",
            "keep_blocked",
        ],
    }
    payload.update(overrides)
    return payload


def _recommendation_row(index: int, decision: str, *, bucket: str = "technical_feasibility_blocked") -> dict:
    return {
        "recommendation_row_id": f"candidate-layer-blocker-decision-recommendation:{index:04d}",
        "source_input_row_id": f"candidate-layer-blocker-decision-input:{index:04d}",
        "source_decision_row_id": f"candidate-layer-blocker-decision:{index:04d}",
        "source_review_card_id": f"candidate-layer-blocker-review:{index:04d}",
        "source_backlog_id": f"candidate-layer-blocker-v1-{index:03d}",
        "blocker": f"blocker_{index}",
        "priority": "P0",
        "review_bucket": bucket,
        "affected_layers": ["sectionspan"],
        "current_decision": "needs_review",
        "recommended_decision": decision,
        "recommendation_rationale": "test_rationale",
        "recommendation_confidence": "medium",
        "allowed_decisions": [
            "needs_review",
            "accept_technical_blocker_as_open",
            "accept_policy_blocker_as_guardrail",
            "record_manual_approval_in_separate_decision_file",
            "record_manual_rejection_in_separate_decision_file",
            "keep_blocked",
        ],
        "manual_edit_required": decision != "needs_review",
        "decision_pointer": f"/decisions/{index - 1}/decision",
        "reviewer_pointer": f"/decisions/{index - 1}/reviewer",
        "notes_pointer": f"/decisions/{index - 1}/notes",
        "decision_file_patch_hint": {
            "source_decision_row_id": f"candidate-layer-blocker-decision:{index:04d}",
            "decision": decision,
            "reviewer": "",
            "notes": "",
        },
        "evidence_tier": "candidate_layer_blocker_nonbinding_recommendation_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": ["recommendation_only"],
        "non_strict_reason": ["not_a_decision"],
    }


def _recommendations(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-nonbinding-decision-recommendations.v1",
        "status": "nonbinding_recommendations_ready",
        "generatedAt": "2026-05-18T00:00:00Z",
        "inputs": {},
        "counts": {
            "recommendationRows": len(rows),
            "manualEditSuggestedRows": sum(1 for row in rows if row["recommended_decision"] != "needs_review"),
            "leaveNeedsReviewRows": sum(1 for row in rows if row["recommended_decision"] == "needs_review"),
            "decisionRowsModified": 0,
            "recommendationOnlyRows": len(rows),
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
            "unsafeUpstreamFlagCount": 0,
            "byRecommendation": {},
            "byBucket": {},
            "byPriority": {},
            "byLayer": {},
        },
        "gate": {
            "recommendationsReady": True,
            "containsRecordedDecisions": False,
            "reviewCopyModified": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "recommendations_ready",
            "unsafeUpstreamFlags": [],
            "recommendedNextTranche": "manual_edit_candidate_layer_blocker_decisions_review_json",
        },
        "policy": {
            "reportOnly": True,
            "recommendationsOnly": True,
            "decisionFileModified": False,
            "decisionRecordCreated": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [],
        "recommendationRows": rows,
    }
    payload.update(overrides)
    return payload


def test_blocker_decision_edit_plan_projects_recommendations_as_manual_hints_only(tmp_path: Path) -> None:
    decisions = _write(
        tmp_path,
        "decisions.json",
        _decision_file([
            _decision_row(1),
            _decision_row(2),
            _decision_row(3),
        ]),
    )
    recommendations = _write(
        tmp_path,
        "recommendations.json",
        _recommendations([
            _recommendation_row(1, "accept_technical_blocker_as_open"),
            _recommendation_row(2, "accept_policy_blocker_as_guardrail", bucket="policy_review_only"),
            _recommendation_row(3, "needs_review", bucket="manual_decision_required"),
        ]),
    )

    payload = build_candidate_layer_blocker_decision_edit_plan(
        candidate_layer_blocker_nonbinding_decision_recommendations_report=recommendations,
        candidate_layer_blocker_decisions_file=decisions,
    )

    assert payload["schema"] == CANDIDATE_LAYER_BLOCKER_DECISION_EDIT_PLAN_SCHEMA_ID
    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_DECISION_EDIT_PLAN_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "edit_plan_ready"
    assert payload["counts"]["editRows"] == 3
    assert payload["counts"]["readyForManualEditRows"] == 3
    assert payload["counts"]["manualEditRequiredRows"] == 2
    assert payload["counts"]["noEditRequiredRows"] == 1
    assert payload["counts"]["proposedAcceptTechnicalOpenRows"] == 1
    assert payload["counts"]["proposedAcceptPolicyGuardrailRows"] == 1
    assert payload["counts"]["proposedNeedsReviewRows"] == 1
    assert payload["counts"]["decisionRowsModified"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["manualDecisionFileEditRequired"] is True
    assert payload["gate"]["decisionFileModified"] is False
    assert payload["gate"]["decisionsRecorded"] is False
    assert payload["policy"]["databaseMutation"] is False
    assert "decisions" not in payload
    first = payload["editRows"][0]
    assert first["edit_status"] == "ready_for_manual_edit"
    assert first["manual_edit_required"] is True
    assert first["accepted_as_human_decision"] is False
    assert first["decision_file_patch_hint"]["reviewer"] == ""
    assert "edit_plan_rows_do_not_modify_the_decision_file" in first["non_strict_reason"]


def test_blocker_decision_edit_plan_blocks_missing_or_disallowed_rows(tmp_path: Path) -> None:
    decisions = _write(
        tmp_path,
        "decisions.json",
        _decision_file([
            _decision_row(1, allowed_decisions=["needs_review", "keep_blocked"]),
        ]),
    )
    recommendations = _write(
        tmp_path,
        "recommendations.json",
        _recommendations([
            _recommendation_row(1, "accept_technical_blocker_as_open"),
            _recommendation_row(2, "accept_policy_blocker_as_guardrail", bucket="policy_review_only"),
        ]),
    )

    payload = build_candidate_layer_blocker_decision_edit_plan(
        candidate_layer_blocker_nonbinding_decision_recommendations_report=recommendations,
        candidate_layer_blocker_decisions_file=decisions,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["blockedRecommendationNotAllowedRows"] == 1
    assert payload["counts"]["blockedMissingDecisionFileRows"] == 1
    assert payload["gate"]["editPlanReady"] is False


def test_blocker_decision_edit_plan_blocks_unsafe_recommendations(tmp_path: Path) -> None:
    decisions = _write(tmp_path, "decisions.json", _decision_file([_decision_row(1)]))
    recommendations = _write(
        tmp_path,
        "recommendations.json",
        _recommendations(
            [_recommendation_row(1, "accept_technical_blocker_as_open")],
            counts={
                "recommendationRows": 1,
                "manualEditSuggestedRows": 1,
                "leaveNeedsReviewRows": 0,
                "decisionRowsModified": 1,
                "recommendationOnlyRows": 1,
                "strictEligibleRows": 0,
                "citationGradeRows": 0,
                "runtimeEvidenceRows": 0,
                "unsafeUpstreamFlagCount": 0,
                "byRecommendation": {},
                "byBucket": {},
                "byPriority": {},
                "byLayer": {},
            },
        ),
    )

    payload = build_candidate_layer_blocker_decision_edit_plan(
        candidate_layer_blocker_nonbinding_decision_recommendations_report=recommendations,
        candidate_layer_blocker_decisions_file=decisions,
    )

    assert payload["status"] == "blocked"
    assert "recommendations_decisionRowsModified_nonzero" in payload["gate"]["unsafeUpstreamFlags"]


def test_blocker_decision_edit_plan_writer_outputs_schema_valid_files(tmp_path: Path) -> None:
    decisions = _write(tmp_path / "input", "decisions.json", _decision_file([_decision_row(1)]))
    recommendations = _write(
        tmp_path / "input",
        "recommendations.json",
        _recommendations([_recommendation_row(1, "accept_technical_blocker_as_open")]),
    )
    payload = build_candidate_layer_blocker_decision_edit_plan(
        candidate_layer_blocker_nonbinding_decision_recommendations_report=recommendations,
        candidate_layer_blocker_decisions_file=decisions,
    )

    paths = write_candidate_layer_blocker_decision_edit_plan_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, CANDIDATE_LAYER_BLOCKER_DECISION_EDIT_PLAN_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["manualEditRequiredRows"] == 1
    assert "This edit plan is not a decision file" in markdown


def test_blocker_decision_edit_plan_cli_writes_report_paths(tmp_path: Path, capsys) -> None:
    decisions = _write(tmp_path / "input", "decisions.json", _decision_file([_decision_row(1)]))
    recommendations = _write(
        tmp_path / "input",
        "recommendations.json",
        _recommendations([_recommendation_row(1, "accept_technical_blocker_as_open")]),
    )
    output_dir = tmp_path / "reports"

    result = main(
        [
            "--candidate-layer-blocker-nonbinding-decision-recommendations-report",
            str(recommendations),
            "--candidate-layer-blocker-decisions-file",
            str(decisions),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "edit_plan_ready"
    assert payload["reportPaths"]["report"] == str(output_dir / "candidate-layer-blocker-decision-edit-plan.json")
    assert Path(payload["reportPaths"]["report"]).exists()
