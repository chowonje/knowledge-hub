from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_BLOCKED_ANSWER = "정책상 민감 정보(P0)가 포함되어 외부 모델 호출을 차단했습니다."


@dataclass(frozen=True)
class AnswerRuntimeFlowDeps:
    build_blocked_payload_fn: Any
    build_generation_fallback_payload_fn: Any
    build_success_payload_fn: Any
    initial_generation_result_fn: Any
    postprocess_result_fn: Any


@dataclass(frozen=True)
class AnswerRuntimeFlowResult:
    answer_text: str
    payload: dict[str, Any]


class AnswerRuntimeFlow:
    def __init__(self, deps: AnswerRuntimeFlowDeps) -> None:
        self._deps = deps

    def run(
        self,
        *,
        query: str,
        retrieval_mode: str,
        pipeline_result: Any,
        evidence_packet: Any,
        selected_llm: Any,
        claim_verification: list[dict[str, Any]],
        claim_consensus: dict[str, Any],
        claim_consensus_merge_mode: str,
        answer_prompt: str,
        safe_context: str,
        external_policy: Any,
        original_classification: str,
        allow_external: bool,
        routing_meta: dict[str, Any],
        routing_warnings: list[str],
        stream: bool = False,
        answer_max_tokens: int | None = None,
    ) -> AnswerRuntimeFlowResult:
        if not external_policy.allowed:
            payload = self._deps.build_blocked_payload_fn(
                query=query,
                retrieval_mode=retrieval_mode,
                pipeline_result=pipeline_result,
                evidence_packet=evidence_packet,
                answer=_BLOCKED_ANSWER,
                claim_verification=claim_verification,
                claim_consensus=claim_consensus,
                original_classification=original_classification,
                external_policy=external_policy,
                safe_context=safe_context,
                allow_external=allow_external,
                routing_meta=routing_meta,
                router_warnings=None if stream else routing_warnings,
            )
            return AnswerRuntimeFlowResult(answer_text=_BLOCKED_ANSWER, payload=payload)

        stage = "initial_stream_answer" if stream else "initial_answer"
        generation_result = self._deps.initial_generation_result_fn(
            query=query,
            selected_llm=selected_llm,
            answer_prompt=answer_prompt,
            safe_context=safe_context,
            evidence_packet=evidence_packet,
            routing_meta=routing_meta,
            stage=stage,
            stream=stream,
            answer_max_tokens=answer_max_tokens,
        )
        if generation_result.is_fallback:
            if generation_result.fallback_kind == "no_route":
                fallback_router_warnings: list[str] | None = [
                    *routing_warnings,
                    *generation_result.generation_warnings,
                ]
                fallback_payload_warnings: list[str] | None = [
                    *routing_warnings,
                    *generation_result.generation_warnings,
                ]
            else:
                fallback_router_warnings = None if stream else routing_warnings
                fallback_payload_warnings = None if stream else generation_result.generation_warnings
            payload = self._deps.build_generation_fallback_payload_fn(
                query=query,
                retrieval_mode=retrieval_mode,
                pipeline_result=pipeline_result,
                evidence_packet=evidence_packet,
                answer=str(generation_result.fallback_answer or ""),
                answer_generation=generation_result.generation_meta or {},
                answer_verification=generation_result.fallback_verification or {},
                answer_rewrite=generation_result.fallback_rewrite or {},
                claim_verification=claim_verification,
                claim_consensus=claim_consensus,
                original_classification=original_classification,
                external_policy=external_policy,
                safe_context=safe_context,
                allow_external=allow_external,
                routing_meta=routing_meta,
                router_warnings=fallback_router_warnings,
                payload_warnings=fallback_payload_warnings,
            )
            return AnswerRuntimeFlowResult(
                answer_text=str(generation_result.fallback_answer or ""),
                payload=payload,
            )

        initial_answer = str(generation_result.initial_answer or "")
        postprocess = self._deps.postprocess_result_fn(
            query=query,
            initial_answer=initial_answer,
            evidence_packet=evidence_packet,
            claim_consensus=claim_consensus,
            claim_consensus_merge_mode=claim_consensus_merge_mode,
            allow_external=allow_external,
            routing_meta=routing_meta,
        )
        final_answer = postprocess.final_answer or initial_answer if stream else postprocess.final_answer
        payload = self._deps.build_success_payload_fn(
            query=query,
            retrieval_mode=retrieval_mode,
            pipeline_result=pipeline_result,
            evidence_packet=evidence_packet,
            answer=final_answer,
            claim_verification=claim_verification,
            claim_consensus=claim_consensus,
            answer_verification=postprocess.final_answer_verification,
            answer_rewrite=postprocess.final_answer_rewrite,
            original_classification=original_classification,
            external_policy=external_policy,
            safe_context=safe_context,
            allow_external=allow_external,
            routing_meta=routing_meta,
            router_warnings=None if stream else routing_warnings,
            initial_answer_verification=(
                None
                if stream or not bool(postprocess.final_answer_rewrite.get("attempted"))
                else postprocess.initial_answer_verification
            ),
            verification_warnings=None if stream else postprocess.verification_warnings,
        )
        return AnswerRuntimeFlowResult(answer_text=final_answer, payload=payload)


__all__ = [
    "AnswerRuntimeFlow",
    "AnswerRuntimeFlowDeps",
    "AnswerRuntimeFlowResult",
]
