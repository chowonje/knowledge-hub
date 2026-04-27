from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from knowledge_hub.ai.answer_contracts import (
    build_answer_contract,
    build_evidence_packet_contract,
    build_verification_verdict,
)


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
        contract_packet = self._contract_evidence_packet(evidence_packet, reason=answerable_reason or "no_result")
        answer_verification = {
            "status": "abstain",
            "supportedClaimCount": 0,
            "unsupportedClaimCount": 0,
            "uncertainClaimCount": 0,
            "needsCaution": False,
            "summary": answerable_reason or "no_result",
        }
        answer_rewrite = {"attempted": False, "applied": False, "finalAnswerSource": "early_exit"}
        payload.setdefault(
            "evidencePacketContract",
            build_evidence_packet_contract(
                query=query,
                retrieval_mode=retrieval_mode,
                pipeline_result=pipeline_result,
                evidence_packet=contract_packet,
            ),
        )
        payload["answerVerification"] = answer_verification
        payload["verificationVerdict"] = build_verification_verdict(answer_verification)
        payload["answerRewrite"] = answer_rewrite
        payload["answerContract"] = build_answer_contract(
            answer=answer,
            evidence_packet=contract_packet,
            verification=answer_verification,
            rewrite=answer_rewrite,
            routing_meta={"provider": "local", "model": "early_exit"},
        )
        return AnswerEarlyExitResult(answer=answer, payload=payload)

    @staticmethod
    def _contract_evidence_packet(evidence_packet: Any, *, reason: str) -> Any:
        payload = dict(getattr(evidence_packet, "evidence_packet", {}) or {})
        payload["answerable"] = False
        payload.setdefault("answerableDecisionReason", reason or "no_result")
        return SimpleNamespace(
            evidence=list(getattr(evidence_packet, "evidence", []) or []),
            citations=list(getattr(evidence_packet, "citations", []) or []),
            evidence_packet=payload,
            evidence_policy=dict(getattr(evidence_packet, "evidence_policy", {}) or {}),
        )


__all__ = [
    "AnswerEarlyExit",
    "AnswerEarlyExitDeps",
    "AnswerEarlyExitResult",
]
