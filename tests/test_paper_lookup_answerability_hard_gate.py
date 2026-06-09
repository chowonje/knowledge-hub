from __future__ import annotations

from types import SimpleNamespace

from knowledge_hub.ai.ask_v2 import _ask_v2_hard_gate_reason


def test_hard_gate_allows_single_paper_lookup_when_direct_evidence_is_present() -> None:
    # Given: fallback claim cards are unsupported, but the single-paper lookup
    # evidence packet has direct substantive evidence from the resolved paper.
    evidence_packet = SimpleNamespace(
        evidence_packet={
            "validation": {
                "substantiveEvidenceCount": 2,
                "directAnswerEvidenceCount": 1,
                "sourceMismatchCount": 0,
            },
            "uniquePaperCount": 1,
        }
    )

    # When: ask-v2 computes the hard gate reason for a paper lookup route.
    reason = _ask_v2_hard_gate_reason(
        verification={"verificationStatus": "weak", "unsupportedFields": []},
        claim_consensus={"unsupportedClaimCount": 3},
        paper_family="paper_lookup",
        evidence_packet=evidence_packet,
    )

    # Then: unsupported fallback claim cards do not override direct paper evidence.
    assert reason == ""


def test_hard_gate_still_blocks_unsupported_claims_without_direct_paper_lookup_evidence() -> None:
    # Given: unsupported claim cards are present and no direct answer evidence exists.
    evidence_packet = SimpleNamespace(
        evidence_packet={
            "validation": {
                "substantiveEvidenceCount": 1,
                "directAnswerEvidenceCount": 0,
                "sourceMismatchCount": 0,
            },
            "uniquePaperCount": 1,
        }
    )

    # When: ask-v2 computes the hard gate reason for the same paper lookup route.
    reason = _ask_v2_hard_gate_reason(
        verification={"verificationStatus": "weak", "unsupportedFields": []},
        claim_consensus={"unsupportedClaimCount": 3},
        paper_family="paper_lookup",
        evidence_packet=evidence_packet,
    )

    # Then: the original unsupported-claim hard gate remains active.
    assert reason == "ask_v2_unsupported_claim_cards"
