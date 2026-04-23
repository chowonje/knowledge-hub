from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnswerExecutionSetupDeps:
    claim_consensus_merge_mode_fn: Any
    section_native_inputs_fn: Any
    claim_native_inputs_fn: Any
    default_answer_inputs_fn: Any
    evaluate_policy_fn: Any


@dataclass(frozen=True)
class PreparedAnswerInputs:
    claim_verification: list[dict[str, Any]]
    claim_consensus: dict[str, Any]
    claim_consensus_merge_mode: str
    answer_prompt: str
    safe_context: str
    external_policy: Any
    original_classification: str


class AnswerExecutionSetup:
    def __init__(self, deps: AnswerExecutionSetupDeps) -> None:
        self._deps = deps

    def prepare_inputs(
        self,
        *,
        query: str,
        pipeline_result: Any,
        evidence_packet: Any,
        selected_llm: Any,
        claim_verification: list[dict[str, Any]],
        claim_consensus: dict[str, Any],
        claim_context: str,
        allow_external: bool,
        route_mode: str,
    ) -> PreparedAnswerInputs:
        deps = self._deps
        claim_consensus_merge_mode = deps.claim_consensus_merge_mode_fn(
            claim_native_used=False,
            pipeline_result=pipeline_result,
        )
        section_native = deps.section_native_inputs_fn(
            query=query,
            pipeline_result=pipeline_result,
            evidence_packet=evidence_packet,
        )
        if section_native is not None:
            answer_prompt, answer_context, _section_coverage = section_native
        else:
            claim_native = deps.claim_native_inputs_fn(
                query=query,
                pipeline_result=pipeline_result,
                evidence_packet=evidence_packet,
                llm=selected_llm,
            )
            if claim_native is not None:
                answer_prompt, answer_context, claim_verification, claim_consensus, _scope_warnings = claim_native
                claim_consensus_merge_mode = deps.claim_consensus_merge_mode_fn(
                    claim_native_used=True,
                    pipeline_result=pipeline_result,
                )
            else:
                answer_prompt, answer_context = deps.default_answer_inputs_fn(
                    query=query,
                    evidence_packet=evidence_packet,
                    claim_context=claim_context,
                )
        safe_context, external_policy, original_classification = deps.evaluate_policy_fn(
            context=answer_context,
            allow_external=allow_external,
            route_mode=route_mode,
        )
        return PreparedAnswerInputs(
            claim_verification=claim_verification,
            claim_consensus=claim_consensus,
            claim_consensus_merge_mode=claim_consensus_merge_mode,
            answer_prompt=answer_prompt,
            safe_context=safe_context,
            external_policy=external_policy,
            original_classification=original_classification,
        )


__all__ = [
    "AnswerExecutionSetup",
    "AnswerExecutionSetupDeps",
    "PreparedAnswerInputs",
]
