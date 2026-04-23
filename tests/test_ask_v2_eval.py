from __future__ import annotations

from knowledge_hub.application.ask_v2_eval import serialize_ask_v2_eval_row


def test_serialize_ask_v2_eval_row_preserves_v2_diagnostics():
    row = serialize_ask_v2_eval_row(
        {
            "query": "Explain project architecture",
            "source": "project",
            "query_type": "architecture",
            "expected_primary_source": "project",
            "expected_answer_style": "implementation_steps",
            "difficulty": "medium",
            "regression_bucket": "project_architecture",
        },
        {
            "status": "ok",
            "answer": "Project answers use structure cards and evidence anchors.",
            "answerVerification": {"needsCaution": True},
            "answerRewrite": {"finalAnswerSource": "rewritten"},
            "claimConsensus": {
                "claimVerificationSummary": "weak",
                "conflictCount": 1,
                "weakClaimCount": 2,
                "unsupportedClaimCount": 0,
            },
            "claimCards": [{"claimCardId": "claim-card-1"}],
            "claimAlignment": {"groups": [{"frame": {"dataset": "MemoryBench"}}]},
            "answerProvenance": {"mode": "claim_cards_conflicted"},
            "v2": {
                "routing": {
                    "mode": "card-first",
                    "intent": "architecture",
                    "sourceKind": "project",
                    "matched_entities": [{"canonical_name": "entrypoint"}],
                    "selected_card_ids": ["project-card-1", "project-card-2"],
                },
                "evidenceVerification": {
                    "anchorIdsUsed": ["anchor-1", "anchor-2", "anchor-3"],
                    "unsupportedFields": ["weak_project_slots"],
                    "verificationStatus": "weak",
                },
                "fallback": {"used": True, "reason": "weak_slots"},
            },
        },
        top_k=4,
        retrieval_mode="hybrid",
        latency_ms=12.345,
    )

    assert row["query"] == "Explain project architecture"
    assert row["selected_source_kind"] == "project"
    assert row["selected_card_ids"] == "project-card-1 | project-card-2"
    assert row["matched_entities"] == "entrypoint"
    assert row["routing_mode"] == "card-first"
    assert row["intent"] == "architecture"
    assert row["anchor_count"] == "3"
    assert row["unsupported_fields"] == "weak_project_slots"
    assert row["claim_verification_summary"] == "weak"
    assert row["claim_conflict_count"] == "1"
    assert row["claim_weak_count"] == "2"
    assert row["claim_card_count"] == "1"
    assert row["claim_alignment_group_count"] == "1"
    assert row["answer_provenance_mode"] == "claim_cards_conflicted"
    assert row["fallback_used"] == "1"
    assert row["weak_evidence"] == "1"
    assert row["needs_caution"] == "1"
