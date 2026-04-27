from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from knowledge_hub.ai.answer_contracts import build_answer_contract, build_verification_verdict


@dataclass(frozen=True)
class AnswerPayloadSlicesDeps:
    base_payload_fn: Any
    policy_payload_fn: Any


class AnswerPayloadSlices:
    def __init__(self, deps: AnswerPayloadSlicesDeps) -> None:
        self._deps = deps

    def build_success_payload(
        self,
        *,
        query: str,
        retrieval_mode: str,
        pipeline_result: Any,
        evidence_packet: Any,
        answer: str,
        claim_verification: list[dict[str, Any]],
        claim_consensus: dict[str, Any],
        answer_verification: dict[str, Any],
        answer_rewrite: dict[str, Any],
        original_classification: str,
        external_policy: Any,
        safe_context: str,
        allow_external: bool,
        routing_meta: dict[str, Any],
        router_warnings: list[str] | None = None,
        initial_answer_verification: dict[str, Any] | None = None,
        verification_warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = self._deps.base_payload_fn(
            query=query,
            retrieval_mode=retrieval_mode,
            pipeline_result=pipeline_result,
            evidence_packet=evidence_packet,
            answer=answer,
        )
        payload["claimVerification"] = claim_verification
        payload["claim_verification"] = claim_verification
        payload["claimConsensus"] = claim_consensus
        payload["claim_consensus"] = claim_consensus
        payload["answerVerification"] = answer_verification
        payload["verificationVerdict"] = build_verification_verdict(answer_verification)
        payload["answerRewrite"] = answer_rewrite
        payload["answerContract"] = build_answer_contract(
            answer=answer,
            evidence_packet=evidence_packet,
            evidence_packet_contract=payload.get("evidencePacketContract"),
            verification=answer_verification,
            rewrite=answer_rewrite,
            routing_meta=routing_meta,
        )
        payload["policy"] = self._deps.policy_payload_fn(
            original_classification=original_classification,
            effective_policy=external_policy,
            safe_context=safe_context,
            original_context=evidence_packet.context,
            allow_external=allow_external,
        )
        router = {"selected": routing_meta}
        if router_warnings is not None:
            router["warnings"] = list(router_warnings)
        payload["router"] = router
        if initial_answer_verification is not None:
            payload["initialAnswerVerification"] = initial_answer_verification
        if verification_warnings:
            payload["warnings"] = list(verification_warnings)
        return payload

    def build_blocked_payload(
        self,
        *,
        query: str,
        retrieval_mode: str,
        pipeline_result: Any,
        evidence_packet: Any,
        answer: str,
        claim_verification: list[dict[str, Any]],
        claim_consensus: dict[str, Any],
        original_classification: str,
        external_policy: Any,
        safe_context: str,
        allow_external: bool,
        routing_meta: dict[str, Any],
        router_warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = self._deps.base_payload_fn(
            query=query,
            retrieval_mode=retrieval_mode,
            pipeline_result=pipeline_result,
            evidence_packet=evidence_packet,
            answer=answer,
            status="blocked",
        )
        payload["policy"] = self._deps.policy_payload_fn(
            original_classification=original_classification,
            effective_policy=external_policy,
            safe_context=safe_context,
            original_context=evidence_packet.context,
            allow_external=allow_external,
        )
        router = {"selected": routing_meta}
        if router_warnings is not None:
            router["warnings"] = list(router_warnings)
        payload["router"] = router
        payload["claimVerification"] = claim_verification
        payload["claim_verification"] = claim_verification
        payload["claimConsensus"] = claim_consensus
        payload["claim_consensus"] = claim_consensus
        payload["verificationVerdict"] = build_verification_verdict({"status": "abstain", "summary": "policy_blocked"})
        payload["answerContract"] = build_answer_contract(
            answer=answer,
            evidence_packet=evidence_packet,
            evidence_packet_contract=payload.get("evidencePacketContract"),
            verification=payload["verificationVerdict"],
            rewrite={"attempted": False, "applied": False, "finalAnswerSource": "policy_blocked"},
            routing_meta=routing_meta,
        )
        return payload

    def build_generation_fallback_payload(
        self,
        *,
        query: str,
        retrieval_mode: str,
        pipeline_result: Any,
        evidence_packet: Any,
        answer: str,
        answer_generation: dict[str, Any],
        answer_verification: dict[str, Any],
        answer_rewrite: dict[str, Any],
        claim_verification: list[dict[str, Any]],
        claim_consensus: dict[str, Any],
        original_classification: str,
        external_policy: Any,
        safe_context: str,
        allow_external: bool,
        routing_meta: dict[str, Any],
        router_warnings: list[str] | None = None,
        payload_warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = self._deps.base_payload_fn(
            query=query,
            retrieval_mode=retrieval_mode,
            pipeline_result=pipeline_result,
            evidence_packet=evidence_packet,
            answer=answer,
        )
        payload["answerVerification"] = answer_verification
        payload["verificationVerdict"] = build_verification_verdict(answer_verification)
        payload["answerRewrite"] = answer_rewrite
        payload["answerGeneration"] = answer_generation
        payload["answerContract"] = build_answer_contract(
            answer=answer,
            evidence_packet=evidence_packet,
            evidence_packet_contract=payload.get("evidencePacketContract"),
            verification=answer_verification,
            rewrite=answer_rewrite,
            routing_meta=routing_meta,
        )
        payload["claimVerification"] = claim_verification
        payload["claim_verification"] = claim_verification
        payload["claimConsensus"] = claim_consensus
        payload["claim_consensus"] = claim_consensus
        payload["policy"] = self._deps.policy_payload_fn(
            original_classification=original_classification,
            effective_policy=external_policy,
            safe_context=safe_context,
            original_context=evidence_packet.context,
            allow_external=allow_external,
        )
        router = {"selected": routing_meta}
        if router_warnings is not None:
            router["warnings"] = list(router_warnings)
        payload["router"] = router
        if payload_warnings:
            payload["warnings"] = list(payload_warnings)
        return payload


__all__ = [
    "AnswerPayloadSlices",
    "AnswerPayloadSlicesDeps",
]
