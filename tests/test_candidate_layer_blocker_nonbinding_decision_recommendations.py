from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.candidate_layer_blocker_nonbinding_decision_recommendations import (
    CANDIDATE_LAYER_BLOCKER_NONBINDING_DECISION_RECOMMENDATIONS_SCHEMA_ID,
    build_candidate_layer_blocker_nonbinding_decision_recommendations,
    main,
    write_candidate_layer_blocker_nonbinding_decision_recommendations_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _decision_input(
    index: int,
    *,
    bucket: str,
    allowed_decisions: list[str],
    blocker: str | None = None,
    layer: str = "sectionspan",
) -> dict:
    return {
        "input_row_id": f"candidate-layer-blocker-decision-input:{index:04d}",
        "source_decision_row_id": f"candidate-layer-blocker-decision:{index:04d}",
        "source_review_card_id": f"candidate-layer-blocker-review-card:{index:04d}",
        "source_backlog_id": f"candidate-layer-blocker-backlog:{index:04d}",
        "blocker": blocker or f"blocker_{index}",
        "priority": "P0",
        "review_bucket": bucket,
        "affected_layers": [layer],
        "affected_candidate_count": index,
        "affected_eval_question_count": 0,
        "recommended_next_tranche": "manual_decision_review",
        "recommended_review_action": "review_manually",
        "allowed_decisions": allowed_decisions,
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
        "strict_blockers": ["manual_review_required"],
        "non_strict_reason": ["candidate_layer_only"],
    }


def _input_pack_payload(**overrides: object) -> dict:
    rows = [
        _decision_input(
            1,
            bucket="technical_feasibility_blocked",
            allowed_decisions=[
                "needs_review",
                "accept_technical_blocker_as_open",
                "defer_technical_followup",
            ],
            blocker="table_cell_provenance_missing",
            layer="table_region",
        ),
        _decision_input(
            2,
            bucket="policy_review_only",
            allowed_decisions=[
                "needs_review",
                "accept_policy_blocker_as_guardrail",
                "defer_policy_review",
            ],
            blocker="runtime_promotion_policy_guardrail",
            layer="figure_caption",
        ),
        _decision_input(
            3,
            bucket="manual_decision_required",
            allowed_decisions=["needs_review", "approve_for_later_design", "reject_candidate"],
            blocker="sectionspan_pdf_offsets_require_human_review_before_strict_promotion",
        ),
        _decision_input(
            4,
            bucket="operator_approval_required",
            allowed_decisions=["needs_review", "operator_approved_for_next_tranche"],
            blocker="candidate_layer_blocker_decision_record_pending",
            layer="equation_quote",
        ),
    ]
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-decision-input-pack.v1",
        "status": "decision_input_pack_ready",
        "generatedAt": "2026-05-18T00:00:00Z",
        "inputs": {},
        "counts": {
            "inputRows": len(rows),
            "pendingSourceDecisionRows": len(rows),
            "defaultNeedsReviewRows": len(rows),
            "manualDecisionInputRows": 1,
            "operatorApprovalInputRows": 1,
            "technicalDecisionInputRows": 1,
            "policyDecisionInputRows": 1,
            "acceptedDecisionRows": 0,
            "operatorApprovedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
            "unsafeUpstreamFlagCount": 0,
            "byBucket": {
                "technical_feasibility_blocked": 1,
                "policy_review_only": 1,
                "manual_decision_required": 1,
                "operator_approval_required": 1,
            },
            "byPriority": {"P0": len(rows)},
            "byLayer": {"sectionspan": 1, "table_region": 1, "figure_caption": 1, "equation_quote": 1},
        },
        "gate": {
            "decisionInputPackReady": True,
            "containsAcceptedDecisions": False,
            "containsOperatorApprovals": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "pending_decision_inputs_ready",
            "unsafeUpstreamFlags": [],
            "recommendedNextTranche": "manual_edit_candidate_layer_blocker_decisions_review_json",
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
        "warnings": [],
        "decisionInputs": rows,
    }
    payload.update(overrides)
    return payload


def _decision_file_payload() -> dict:
    return {
        "draftOnly": True,
        "decisions": [
            {
                "source_decision_row_id": "candidate-layer-blocker-decision:0001",
                "decision": "needs_review",
                "reviewer": "",
                "notes": "",
            },
            {
                "source_decision_row_id": "candidate-layer-blocker-decision:0002",
                "decision": "accept_policy_blocker_as_guardrail",
                "reviewer": "operator",
                "notes": "Already recorded in a copied local review file.",
            },
            {
                "source_decision_row_id": "candidate-layer-blocker-decision:0003",
                "decision": "needs_review",
                "reviewer": "",
                "notes": "",
            },
            {
                "source_decision_row_id": "candidate-layer-blocker-decision:0004",
                "decision": "needs_review",
                "reviewer": "",
                "notes": "",
            },
        ],
    }


def test_nonbinding_decision_recommendations_bucket_mapping_and_policy(tmp_path: Path) -> None:
    input_pack = _write(tmp_path, "input-pack.json", _input_pack_payload())

    payload = build_candidate_layer_blocker_nonbinding_decision_recommendations(
        candidate_layer_blocker_decision_input_pack_report=input_pack,
    )

    assert payload["schema"] == CANDIDATE_LAYER_BLOCKER_NONBINDING_DECISION_RECOMMENDATIONS_SCHEMA_ID
    assert validate_payload(
        payload,
        CANDIDATE_LAYER_BLOCKER_NONBINDING_DECISION_RECOMMENDATIONS_SCHEMA_ID,
        strict=True,
    ).ok
    assert payload["status"] == "nonbinding_recommendations_ready"
    assert payload["counts"]["recommendationRows"] == 4
    assert payload["counts"]["decisionRowsModified"] == 0
    assert payload["counts"]["recommendationOnlyRows"] == 4
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["counts"]["runtimeEvidenceRows"] == 0
    assert payload["gate"]["containsRecordedDecisions"] is False
    assert payload["gate"]["reviewCopyModified"] is False
    assert payload["policy"]["decisionFileModified"] is False
    assert payload["policy"]["databaseMutation"] is False
    assert payload["policy"]["canonicalParsedArtifactsWritten"] is False

    recommendations = {
        row["review_bucket"]: row["recommended_decision"]
        for row in payload["recommendationRows"]
    }
    assert recommendations == {
        "technical_feasibility_blocked": "accept_technical_blocker_as_open",
        "policy_review_only": "accept_policy_blocker_as_guardrail",
        "manual_decision_required": "needs_review",
        "operator_approval_required": "needs_review",
    }
    assert payload["counts"]["byRecommendation"] == {
        "accept_technical_blocker_as_open": 1,
        "accept_policy_blocker_as_guardrail": 1,
        "needs_review": 2,
    }
    assert all(row["evidence_tier"] == "candidate_layer_blocker_nonbinding_recommendation_only" for row in payload["recommendationRows"])
    assert all(row["strict_eligible"] is False for row in payload["recommendationRows"])
    assert all(row["runtime_evidence"] is False for row in payload["recommendationRows"])


def test_nonbinding_decision_recommendations_reads_review_copy_without_modifying_it(tmp_path: Path) -> None:
    input_pack = _write(tmp_path, "input-pack.json", _input_pack_payload())
    decision_file = _write(tmp_path, "candidate-layer-blocker-decisions.review.json", _decision_file_payload())
    before = decision_file.read_text(encoding="utf-8")

    payload = build_candidate_layer_blocker_nonbinding_decision_recommendations(
        candidate_layer_blocker_decision_input_pack_report=input_pack,
        blocker_decisions_file=decision_file,
    )

    after = decision_file.read_text(encoding="utf-8")
    assert after == before
    assert payload["inputs"]["blockerDecisionRows"] == 4
    policy_row = next(row for row in payload["recommendationRows"] if row["review_bucket"] == "policy_review_only")
    assert policy_row["current_decision"] == "accept_policy_blocker_as_guardrail"
    assert policy_row["manual_edit_required"] is False
    technical_row = next(row for row in payload["recommendationRows"] if row["review_bucket"] == "technical_feasibility_blocked")
    assert technical_row["manual_edit_required"] is True
    assert technical_row["decision_pointer"] == "/decisions/0/decision"


def test_nonbinding_decision_recommendations_block_schema_mismatch(tmp_path: Path) -> None:
    bad_input_pack = _write(tmp_path, "bad-input-pack.json", _input_pack_payload(schema="example.bad.v1"))

    payload = build_candidate_layer_blocker_nonbinding_decision_recommendations(
        candidate_layer_blocker_decision_input_pack_report=bad_input_pack,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["recommendationsReady"] is False
    assert payload["counts"]["unsafeUpstreamFlagCount"] == 1
    assert "candidate_layer_blocker_decision_input_pack_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert validate_payload(
        payload,
        CANDIDATE_LAYER_BLOCKER_NONBINDING_DECISION_RECOMMENDATIONS_SCHEMA_ID,
        strict=True,
    ).ok


def test_nonbinding_decision_recommendations_writer_outputs_schema_valid_json_and_markdown(
    tmp_path: Path,
) -> None:
    input_pack = _write(tmp_path, "input-pack.json", _input_pack_payload())
    payload = build_candidate_layer_blocker_nonbinding_decision_recommendations(
        candidate_layer_blocker_decision_input_pack_report=input_pack,
    )

    paths = write_candidate_layer_blocker_nonbinding_decision_recommendations_reports(payload, tmp_path / "out")

    assert set(paths) == {"recommendations", "summary", "markdown"}
    recommendations = json.loads(Path(paths["recommendations"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(
        recommendations,
        CANDIDATE_LAYER_BLOCKER_NONBINDING_DECISION_RECOMMENDATIONS_SCHEMA_ID,
        strict=True,
    ).ok
    assert summary["counts"]["recommendationRows"] == 4
    assert "Candidate Layer Blocker Nonbinding Decision Recommendations" in markdown


def test_nonbinding_decision_recommendations_cli_accepts_canonical_decisions_file_alias(
    tmp_path: Path,
    capsys,
) -> None:
    input_pack = _write(tmp_path, "input-pack.json", _input_pack_payload())
    decision_file = _write(tmp_path, "decisions.json", _decision_file_payload())
    output_dir = tmp_path / "out"

    result = main(
        [
            "--candidate-layer-blocker-decision-input-pack-report",
            str(input_pack),
            "--candidate-layer-blocker-decisions-file",
            str(decision_file),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert result == 0
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["status"] == "nonbinding_recommendations_ready"
    assert payload["reportPaths"]["recommendations"] == str(
        output_dir / "candidate-layer-blocker-nonbinding-decision-recommendations.json"
    )
    assert Path(payload["reportPaths"]["recommendations"]).exists()
