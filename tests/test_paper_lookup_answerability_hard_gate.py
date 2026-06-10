from __future__ import annotations

from types import SimpleNamespace

from knowledge_hub.ai.evidence_assembly import _refresh_validation_counts
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


def test_hard_gate_allows_source_backed_paper_lookup_anchor_from_live_transformer_smoke() -> None:
    # Given: the live Transformer lookup resolves the right paper and selects a
    # source-backed paper-card anchor from the paper abstract/introduction.
    validation = _refresh_validation_counts(
        {},
        query="Explain the core idea of the Transformer paper.",
        source_type="paper",
        evidence=[
            {
                "title": "Attention Is All You Need",
                "source_type": "paper",
                "retrieval_mode": "paper-card-v2",
                "arxiv_id": "1706.03762",
                "source_id": "1706.03762",
                "citation_target": "1706.03762",
                "evidence_kind": "memory_hint",
                "section_path": "arXiv:1706.03762v7[cs.CL]2 Aug 2023 > 1 Introduction",
                "excerpt": (
                    "[arXiv:1706.03762v7[cs.CL]2 Aug 2023 > 1 Introduction] "
                    "Recurrent neural networks, long short-term memory [13] and gated recurrent "
                    "[7] neural networks in particular, have been firmly established as sta..."
                ),
            }
        ],
    )
    evidence_packet = SimpleNamespace(evidence_packet={"validation": validation, "uniquePaperCount": 1})

    # When: unsupported fallback claim cards are present for the paper lookup.
    reason = _ask_v2_hard_gate_reason(
        verification={"verificationStatus": "weak", "unsupportedFields": []},
        claim_consensus={"unsupportedClaimCount": 3},
        paper_family="paper_lookup",
        evidence_packet=evidence_packet,
    )

    # Then: direct source-backed paper evidence is enough to bypass the fallback
    # claim-card hard block for this single-paper lookup.
    assert validation["substantiveEvidenceCount"] == 1
    assert validation["directAnswerEvidenceCount"] == 1
    assert validation["sourceMismatchCount"] == 0
    assert reason == ""
