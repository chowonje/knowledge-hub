from __future__ import annotations

from knowledge_hub.ai.answer_verification import heuristic_answer_verification
from knowledge_hub.ai.rag_support import (
    build_paper_answer_readiness_p1_conservative_answer,
    split_answer_claims,
)


class _VerifierSearcher:
    @staticmethod
    def _split_answer_claims(answer: str) -> list[str]:
        return split_answer_claims(answer)

    @staticmethod
    def _answer_mentions_conflict(answer: str) -> bool:
        return "conflict" in answer.lower() or "상충" in answer


def test_evidence_only_p1_fallback_reduces_unsupported_but_can_still_need_caution():
    evidence = [
        {
            "title": "Attention Is All You Need",
            "excerpt": "The Transformer is based solely on attention mechanisms.",
            "source_type": "paper",
        }
    ]
    unsupported = heuristic_answer_verification(
        _VerifierSearcher(),
        answer="The paper proves a 99 percent ImageNet accuracy improvement with convolutional recurrence.",
        evidence=evidence,
        answer_signals={},
        contradicting_beliefs=[],
        route_meta={"route": "local"},
    )
    fallback_answer = build_paper_answer_readiness_p1_conservative_answer(evidence=evidence)
    fallback = heuristic_answer_verification(
        _VerifierSearcher(),
        answer=fallback_answer,
        evidence=evidence,
        answer_signals={},
        contradicting_beliefs=[],
        route_meta={"route": "local"},
    )

    assert unsupported["unsupportedClaimCount"] > fallback["unsupportedClaimCount"]
    assert fallback["unsupportedClaimCount"] == 0
    assert fallback["uncertainClaimCount"] > 0
    # The current local heuristic has no supported verdict path, so exact evidence bullets can still require caution.
    assert fallback["needsCaution"] is True
