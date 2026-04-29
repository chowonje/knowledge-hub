from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnswerEarlyExitDeps:
    base_payload_fn: Any
    adjudicate_claims_fn: Any
    comparison_insufficient_answer_fn: Any


@dataclass(frozen=True)
class AnswerEarlyExitResult:
    answer: str
    payload: dict[str, Any]


class AnswerEarlyExit:
    def __init__(self, deps: AnswerEarlyExitDeps) -> None:
        self._deps = deps

    def build_result(
        self,
        *,
        query: str,
        retrieval_mode: str,
        pipeline_result: Any,
        evidence_packet: Any,
    ) -> AnswerEarlyExitResult | None:
        evidence_packet_payload = dict(evidence_packet.evidence_packet or {})
        answerable_reason = str(evidence_packet_payload.get("answerableDecisionReason") or "")
        if not evidence_packet.filtered_results:
            answer = "관련된 문서를 찾을 수 없습니다."
        elif answerable_reason == "need_multiple_papers":
            answer = self._deps.comparison_insufficient_answer_fn(evidence_packet)
        elif evidence_packet_payload.get("askV2HardGate") is True:
            answer = "제공된 근거만으로는 검증 가능한 답변을 생성하기 어렵습니다."
        else:
            return None

        claim_verification, claim_consensus, _claim_context = self._deps.adjudicate_claims_fn(
            evidence_packet=evidence_packet
        )
        payload = self._deps.base_payload_fn(
            query=query,
            retrieval_mode=retrieval_mode,
            pipeline_result=pipeline_result,
            evidence_packet=evidence_packet,
            answer=answer,
            status="no_result",
        )
        payload["claimVerification"] = claim_verification
        payload["claim_verification"] = claim_verification
        payload["claimConsensus"] = claim_consensus
        payload["claim_consensus"] = claim_consensus
        return AnswerEarlyExitResult(answer=answer, payload=payload)


__all__ = [
    "AnswerEarlyExit",
    "AnswerEarlyExitDeps",
    "AnswerEarlyExitResult",
]
