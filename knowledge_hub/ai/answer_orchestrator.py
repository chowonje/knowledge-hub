from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from knowledge_hub.ai.answer_policy_support import (
    apply_claim_consensus_to_verification,
    evaluate_policy,
    policy_payload,
    with_claim_context,
)
from knowledge_hub.ai.answer_execution_setup import (
    AnswerExecutionSetup,
    AnswerExecutionSetupDeps,
    PreparedAnswerInputs,
)
from knowledge_hub.ai.answer_orchestrator_initial_generation import (
    AnswerInitialGeneration,
    AnswerInitialGenerationDeps,
    AnswerInitialGenerationResult,
)
from knowledge_hub.ai.answer_orchestrator_runtime_flow import (
    AnswerRuntimeFlow,
    AnswerRuntimeFlowDeps,
    AnswerRuntimeFlowResult,
)
from knowledge_hub.ai.answer_orchestrator_early_exit import (
    AnswerEarlyExit,
    AnswerEarlyExitDeps,
    AnswerEarlyExitResult,
)
from knowledge_hub.ai.answer_orchestrator_postprocess import (
    AnswerPostprocess,
    AnswerPostprocessDeps,
    AnswerPostprocessResult,
)
from knowledge_hub.ai.answer_orchestrator_payload_slices import (
    AnswerPayloadSlices,
    AnswerPayloadSlicesDeps,
)
from knowledge_hub.ai.answer_rewrite import (
    apply_conservative_fallback_if_needed as _apply_conservative_fallback_if_needed_impl,
    rewrite_answer as _rewrite_answer_impl,
)
from knowledge_hub.ai.answer_native_inputs import AnswerNativeInputBuilder
from knowledge_hub.ai.answer_payload_builder import AnswerPayloadBuilder
from knowledge_hub.ai.answer_verification import verify_answer as _verify_answer_impl
from knowledge_hub.ai.rag_answer_route_resolver import resolve_llm_for_request as _resolve_llm_for_request_impl
from knowledge_hub.ai.rag_support import (
    build_answer_generation_fallback as _build_answer_generation_fallback_impl,
    build_answer_prompt as _build_answer_prompt_impl,
    build_paper_definition_context as _build_paper_definition_context_impl,
    record_answer_log as _record_answer_log_impl,
)
from knowledge_hub.learning.model_router import get_llm_for_hybrid_routing


@dataclass(frozen=True)
class AnswerHelperDeps:
    build_answer_generation_fallback: Any
    build_answer_prompt: Any
    build_paper_definition_context: Any
    record_answer_log: Any


@dataclass(frozen=True)
class AnswerRuntimeDeps:
    resolve_llm_for_request: Any
    verify_answer: Any
    rewrite_answer: Any
    apply_conservative_fallback_if_needed: Any


class AnswerOrchestrator:
    def __init__(self, searcher: Any):
        self.searcher = searcher
        self.payload_builder = AnswerPayloadBuilder(searcher)
        self.native_inputs = AnswerNativeInputBuilder(searcher)

    def _base_payload(
        self,
        *,
        query: str,
        retrieval_mode: str,
        pipeline_result: Any,
        evidence_packet: Any,
        answer: str,
        status: str = "ok",
    ) -> dict[str, Any]:
        return self.payload_builder.base_payload(
            query=query,
            retrieval_mode=retrieval_mode,
            pipeline_result=pipeline_result,
            evidence_packet=evidence_packet,
            answer=answer,
            status=status,
        )

    def _adjudicate_claims(self, *, evidence_packet: Any) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
        return self.native_inputs.adjudicate_claims(evidence_packet=evidence_packet)

    def _method_overridden(self, name: str) -> bool:
        bound = getattr(self.searcher, name, None)
        class_attr = getattr(type(self.searcher), name, None)
        func = getattr(bound, "__func__", None)
        if callable(bound) and func is None:
            return True
        return callable(bound) and func is not None and func is not class_attr

    @staticmethod
    def _helper_deps() -> AnswerHelperDeps:
        return AnswerHelperDeps(
            build_answer_generation_fallback=_build_answer_generation_fallback_impl,
            build_answer_prompt=_build_answer_prompt_impl,
            build_paper_definition_context=_build_paper_definition_context_impl,
            record_answer_log=_record_answer_log_impl,
        )

    @staticmethod
    def _runtime_deps() -> AnswerRuntimeDeps:
        return AnswerRuntimeDeps(
            resolve_llm_for_request=_resolve_llm_for_request_impl,
            verify_answer=_verify_answer_impl,
            rewrite_answer=_rewrite_answer_impl,
            apply_conservative_fallback_if_needed=_apply_conservative_fallback_if_needed_impl,
        )

    def _early_exit_result(
        self,
        *,
        query: str,
        source_type: str | None,
        retrieval_mode: str,
        allow_external: bool,
        pipeline_result: Any,
        evidence_packet: Any,
    ) -> AnswerEarlyExitResult | None:
        result = AnswerEarlyExit(
            AnswerEarlyExitDeps(
                base_payload_fn=self._base_payload,
                adjudicate_claims_fn=self._adjudicate_claims,
                comparison_insufficient_answer_fn=self._comparison_insufficient_answer,
            )
        ).build_result(
            query=query,
            retrieval_mode=retrieval_mode,
            pipeline_result=pipeline_result,
            evidence_packet=evidence_packet,
        )
        if result is None:
            return None
        self._record_answer_log(
            query=query,
            payload=result.payload,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            allow_external=allow_external,
        )
        return result

    def _build_answer_prompt(
        self,
        *,
        query: str,
        answer_signals: dict[str, Any],
    ) -> str:
        if self._method_overridden("_build_answer_prompt"):
            return self.searcher._build_answer_prompt(
                query=query,
                answer_signals=answer_signals,
            )
        return self._helper_deps().build_answer_prompt(
            query=query,
            answer_signals=answer_signals,
        )

    def _build_answer_generation_fallback(
        self,
        *,
        query: str,
        error: Exception,
        stage: str,
        evidence: list[dict[str, Any]],
        citations: list[dict[str, Any]],
        routing_meta: dict[str, Any],
    ) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
        if self._method_overridden("_build_answer_generation_fallback"):
            return self.searcher._build_answer_generation_fallback(
                query=query,
                error=error,
                stage=stage,
                evidence=evidence,
                citations=citations,
                routing_meta=routing_meta,
            )
        return self._helper_deps().build_answer_generation_fallback(
            query=query,
            error=error,
            stage=stage,
            evidence=evidence,
            citations=citations,
            routing_meta=routing_meta,
        )

    def _build_paper_definition_context(
        self,
        *,
        query: str,
        filtered: list[Any],
        evidence: list[dict[str, Any]],
        answer_signals: dict[str, Any],
        claim_context: str,
    ) -> str:
        if self._method_overridden("_build_paper_definition_context"):
            return self.searcher._build_paper_definition_context(
                query=query,
                filtered=filtered,
                evidence=evidence,
                answer_signals=answer_signals,
                claim_context=claim_context,
            )
        return self._helper_deps().build_paper_definition_context(
            query=query,
            filtered=filtered,
            evidence=evidence,
            answer_signals=answer_signals,
            claim_context=claim_context,
        )

    def _record_answer_log(
        self,
        *,
        query: str,
        payload: dict[str, Any],
        source_type: str | None,
        retrieval_mode: str,
        allow_external: bool,
    ) -> None:
        if self._method_overridden("_record_answer_log"):
            self.searcher._record_answer_log(
                query=query,
                payload=payload,
                source_type=source_type,
                retrieval_mode=retrieval_mode,
                allow_external=allow_external,
            )
            return
        self._helper_deps().record_answer_log(
            getattr(self.searcher.sqlite_db, "add_rag_answer_log", None),
            query=query,
            payload=payload,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            allow_external=allow_external,
        )

    def _resolve_llm_for_request(
        self,
        *,
        query: str,
        context: str,
        source_count: int,
        allow_external: bool,
        force_route: str | None = None,
    ) -> tuple[Any, dict[str, Any], list[str]]:
        if self._method_overridden("_resolve_llm_for_request"):
            return self.searcher._resolve_llm_for_request(
                query=query,
                context=context,
                source_count=source_count,
                allow_external=allow_external,
                force_route=force_route,
            )
        cached_local_llm, cached_local_llm_signature = self.searcher._caches.route_llm_cache()
        hybrid_router_fn = get_llm_for_hybrid_routing
        try:
            from knowledge_hub.ai import rag as rag_module

            hybrid_router_fn = getattr(rag_module, "get_llm_for_hybrid_routing", hybrid_router_fn)
        except Exception:
            pass
        llm, decision, warnings, next_cached_local_llm, next_cached_local_llm_signature = self._runtime_deps().resolve_llm_for_request(
            config=self.searcher.config,
            fixed_llm=self.searcher.llm,
            query=query,
            context=context,
            source_count=source_count,
            allow_external=allow_external,
            force_route=force_route,
            cached_local_llm=cached_local_llm,
            cached_local_llm_signature=cached_local_llm_signature,
            get_llm_for_hybrid_routing_fn=hybrid_router_fn,
        )
        self.searcher._caches.writeback_route_llm_cache(
            self.searcher,
            cached_local_llm=next_cached_local_llm,
            cached_local_llm_signature=next_cached_local_llm_signature,
        )
        return llm, decision, warnings

    def _verify_answer(
        self,
        *,
        query: str,
        answer: str,
        evidence: list[dict[str, Any]],
        answer_signals: dict[str, Any],
        contradicting_beliefs: list[dict[str, Any]],
        allow_external: bool,
    ) -> dict[str, Any]:
        if self._method_overridden("_verify_answer"):
            return self.searcher._verify_answer(
                query=query,
                answer=answer,
                evidence=evidence,
                answer_signals=answer_signals,
                contradicting_beliefs=contradicting_beliefs,
                allow_external=allow_external,
            )
        return self._runtime_deps().verify_answer(
            self.searcher,
            query=query,
            answer=answer,
            evidence=evidence,
            answer_signals=answer_signals,
            contradicting_beliefs=contradicting_beliefs,
            allow_external=allow_external,
        )

    def _rewrite_answer(
        self,
        *,
        query: str,
        answer: str,
        evidence: list[dict[str, Any]],
        answer_signals: dict[str, Any],
        verification: dict[str, Any],
        contradicting_beliefs: list[dict[str, Any]],
        allow_external: bool,
    ) -> tuple[str, dict[str, Any]]:
        if self._method_overridden("_rewrite_answer"):
            return self.searcher._rewrite_answer(
                query=query,
                answer=answer,
                evidence=evidence,
                answer_signals=answer_signals,
                verification=verification,
                contradicting_beliefs=contradicting_beliefs,
                allow_external=allow_external,
            )
        return self._runtime_deps().rewrite_answer(
            self.searcher,
            query=query,
            answer=answer,
            evidence=evidence,
            answer_signals=answer_signals,
            verification=verification,
            contradicting_beliefs=contradicting_beliefs,
            allow_external=allow_external,
        )

    def _apply_conservative_fallback_if_needed(
        self,
        *,
        query: str,
        answer: str,
        rewrite_meta: dict[str, Any],
        verification: dict[str, Any],
        evidence: list[dict[str, Any]],
        answer_signals: dict[str, Any],
        contradicting_beliefs: list[dict[str, Any]],
        allow_external: bool,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        if self._method_overridden("_apply_conservative_fallback_if_needed"):
            return self.searcher._apply_conservative_fallback_if_needed(
                query=query,
                answer=answer,
                rewrite_meta=rewrite_meta,
                verification=verification,
                evidence=evidence,
                answer_signals=answer_signals,
                contradicting_beliefs=contradicting_beliefs,
                allow_external=allow_external,
            )
        return self._runtime_deps().apply_conservative_fallback_if_needed(
            self.searcher,
            query=query,
            answer=answer,
            rewrite_meta=rewrite_meta,
            verification=verification,
            evidence=evidence,
            answer_signals=answer_signals,
            contradicting_beliefs=contradicting_beliefs,
            allow_external=allow_external,
        )

    @staticmethod
    def _comparison_insufficient_answer(evidence_packet: Any) -> str:
        unique_count = int((evidence_packet.evidence_packet or {}).get("uniquePaperCount") or 0)
        if unique_count <= 1:
            return "비교 가능한 논문 2편 이상을 찾지 못했습니다."
        return "비교에 필요한 논문 근거가 충분하지 않습니다."

    @staticmethod
    def _claim_consensus_from_verification(claim_verification: list[dict[str, Any]]) -> dict[str, Any]:
        return AnswerNativeInputBuilder.claim_consensus_from_verification(claim_verification)

    @staticmethod
    def _v2_claim_bundle(pipeline_result: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        return AnswerNativeInputBuilder.v2_claim_bundle(pipeline_result)

    @staticmethod
    def _v2_section_bundle(pipeline_result: Any) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
        return AnswerNativeInputBuilder.v2_section_bundle(pipeline_result)

    def _structured_verify_selected_claims(
        self,
        *,
        llm: Any,
        claim_cards: list[dict[str, Any]],
        claim_verification: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        return self.native_inputs.structured_verify_selected_claims(
            llm=llm,
            claim_cards=claim_cards,
            claim_verification=claim_verification,
        )

    @staticmethod
    def _comparison_verification(*, claim_alignment: list[dict[str, Any]]) -> dict[str, Any]:
        return AnswerNativeInputBuilder.comparison_verification(claim_alignment=claim_alignment)

    def _claim_native_inputs(
        self,
        *,
        query: str,
        pipeline_result: Any,
        evidence_packet: Any,
        llm: Any,
    ) -> tuple[str, str, list[dict[str, Any]], dict[str, Any], list[str]] | None:
        return self.native_inputs.claim_native_inputs(
            query=query,
            pipeline_result=pipeline_result,
            evidence_packet=evidence_packet,
            llm=llm,
        )

    def _section_native_inputs(
        self,
        *,
        query: str,
        pipeline_result: Any,
        evidence_packet: Any,
    ) -> tuple[str, str, dict[str, Any]] | None:
        return self.native_inputs.section_native_inputs(
            query=query,
            pipeline_result=pipeline_result,
            evidence_packet=evidence_packet,
        )

    @staticmethod
    def _with_claim_context(context: str, claim_context: str) -> str:
        return with_claim_context(context, claim_context)

    def _default_answer_inputs(
        self,
        *,
        query: str,
        evidence_packet: Any,
        claim_context: str,
    ) -> tuple[str, str]:
        answer_prompt = self._build_answer_prompt(
            query=query,
            answer_signals=evidence_packet.answer_signals,
        )
        if bool(dict(evidence_packet.answer_signals or {}).get("paper_definition_mode")):
            answer_context = self._build_paper_definition_context(
                query=query,
                filtered=evidence_packet.filtered_results,
                evidence=evidence_packet.evidence,
                answer_signals=evidence_packet.answer_signals,
                claim_context=claim_context,
            )
        else:
            answer_context = self._with_claim_context(evidence_packet.context, claim_context)
        return answer_prompt, answer_context

    @staticmethod
    def _apply_claim_consensus_to_verification(
        verification: dict[str, Any],
        claim_consensus: dict[str, Any],
        *,
        force_weak_caution: bool = True,
        merge_mode: str = "strict",
    ) -> dict[str, Any]:
        return apply_claim_consensus_to_verification(
            verification,
            claim_consensus,
            force_weak_caution=force_weak_caution,
            merge_mode=merge_mode,
        )

    @staticmethod
    def _claim_consensus_merge_mode(*, claim_native_used: bool, pipeline_result: Any) -> str:
        if not claim_native_used:
            return "advisory"
        try:
            plan_payload = dict(getattr(getattr(pipeline_result, "plan", None), "to_dict", lambda: {})() or {})
        except Exception:
            plan_payload = {}
        diagnostics = dict(getattr(pipeline_result, "v2_diagnostics", {}) or {})
        query_frame = dict(plan_payload.get("queryFrame") or {})
        family = str(
            query_frame.get("family")
            or plan_payload.get("paperFamily")
            or ""
        ).strip().lower()
        if family != "paper_compare":
            return "advisory"
        answer_provenance = dict(diagnostics.get("answerProvenance") or {})
        provenance_mode = str(answer_provenance.get("mode") or "").strip().lower()
        if provenance_mode == "weak_claim_fallback":
            return "advisory"
        return "strict"

    def _evaluate_policy(
        self,
        *,
        context: str,
        allow_external: bool,
        route_mode: str,
    ) -> tuple[str, Any, str]:
        return evaluate_policy(
            context=context,
            allow_external=allow_external,
            route_mode=route_mode,
        )

    def _prepare_answer_execution_inputs(
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
        setup = AnswerExecutionSetup(
            AnswerExecutionSetupDeps(
                claim_consensus_merge_mode_fn=self._claim_consensus_merge_mode,
                section_native_inputs_fn=self._section_native_inputs,
                claim_native_inputs_fn=self._claim_native_inputs,
                default_answer_inputs_fn=self._default_answer_inputs,
                evaluate_policy_fn=self._evaluate_policy,
            )
        )
        return setup.prepare_inputs(
            query=query,
            pipeline_result=pipeline_result,
            evidence_packet=evidence_packet,
            selected_llm=selected_llm,
            claim_verification=claim_verification,
            claim_consensus=claim_consensus,
            claim_context=claim_context,
            allow_external=allow_external,
            route_mode=route_mode,
        )

    def _postprocess_result(
        self,
        *,
        query: str,
        initial_answer: str,
        evidence_packet: Any,
        claim_consensus: dict[str, Any],
        claim_consensus_merge_mode: str,
        allow_external: bool,
    ) -> AnswerPostprocessResult:
        return AnswerPostprocess(
            AnswerPostprocessDeps(
                verify_answer_fn=self._verify_answer,
                rewrite_answer_fn=self._rewrite_answer,
                apply_conservative_fallback_if_needed_fn=self._apply_conservative_fallback_if_needed,
                apply_claim_consensus_to_verification_fn=self._apply_claim_consensus_to_verification,
            )
        ).run(
            query=query,
            initial_answer=initial_answer,
            evidence_packet=evidence_packet,
            claim_consensus=claim_consensus,
            claim_consensus_merge_mode=claim_consensus_merge_mode,
            allow_external=allow_external,
        )

    def _payload_slices(self) -> AnswerPayloadSlices:
        return AnswerPayloadSlices(
            AnswerPayloadSlicesDeps(
                base_payload_fn=self._base_payload,
                policy_payload_fn=policy_payload,
            )
        )

    def _initial_generation_result(
        self,
        *,
        query: str,
        selected_llm: Any,
        answer_prompt: str,
        safe_context: str,
        evidence_packet: Any,
        routing_meta: dict[str, Any],
        stage: str,
        stream: bool = False,
    ) -> AnswerInitialGenerationResult:
        return AnswerInitialGeneration(
            AnswerInitialGenerationDeps(
                build_answer_generation_fallback_fn=self._build_answer_generation_fallback,
            )
        ).run(
            query=query,
            selected_llm=selected_llm,
            answer_prompt=answer_prompt,
            safe_context=safe_context,
            evidence_packet=evidence_packet,
            routing_meta=routing_meta,
            stage=stage,
            stream=stream,
        )

    def _runtime_flow_result(
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
    ) -> AnswerRuntimeFlowResult:
        payload_slices = self._payload_slices()
        return AnswerRuntimeFlow(
            AnswerRuntimeFlowDeps(
                build_blocked_payload_fn=payload_slices.build_blocked_payload,
                build_generation_fallback_payload_fn=payload_slices.build_generation_fallback_payload,
                build_success_payload_fn=payload_slices.build_success_payload,
                initial_generation_result_fn=self._initial_generation_result,
                postprocess_result_fn=self._postprocess_result,
            )
        ).run(
            query=query,
            retrieval_mode=retrieval_mode,
            pipeline_result=pipeline_result,
            evidence_packet=evidence_packet,
            selected_llm=selected_llm,
            claim_verification=claim_verification,
            claim_consensus=claim_consensus,
            claim_consensus_merge_mode=claim_consensus_merge_mode,
            answer_prompt=answer_prompt,
            safe_context=safe_context,
            external_policy=external_policy,
            original_classification=original_classification,
            allow_external=allow_external,
            routing_meta=routing_meta,
            routing_warnings=routing_warnings,
            stream=stream,
        )

    def generate(
        self,
        *,
        query: str,
        source_type: str | None,
        retrieval_mode: str,
        allow_external: bool,
        answer_route_override: str | None = None,
        pipeline_result: Any,
        evidence_packet: Any,
    ) -> dict[str, Any]:
        early_exit = self._early_exit_result(
            query=query,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            allow_external=allow_external,
            pipeline_result=pipeline_result,
            evidence_packet=evidence_packet,
        )
        if early_exit is not None:
            return early_exit.payload

        claim_verification, claim_consensus, claim_context = self._adjudicate_claims(evidence_packet=evidence_packet)
        selected_llm, routing_meta, routing_warnings = self._resolve_llm_for_request(
            query=query,
            context=self._with_claim_context(evidence_packet.context, claim_context),
            source_count=len(evidence_packet.filtered_results),
            allow_external=allow_external,
            force_route=answer_route_override,
        )
        self.searcher._caches.sync_active_request_llm_from_searcher(self.searcher)
        previous_active_request_llm = self.searcher._caches.active_request_llm_value()
        self.searcher._caches.write_active_request_llm(self.searcher, selected_llm)
        try:
            route_mode = str(routing_meta.get("route", "fixed"))
            prepared_inputs = self._prepare_answer_execution_inputs(
                query=query,
                pipeline_result=pipeline_result,
                evidence_packet=evidence_packet,
                selected_llm=selected_llm,
                claim_verification=claim_verification,
                claim_consensus=claim_consensus,
                claim_context=claim_context,
                allow_external=allow_external,
                route_mode=route_mode,
            )
            claim_verification = prepared_inputs.claim_verification
            claim_consensus = prepared_inputs.claim_consensus
            claim_consensus_merge_mode = prepared_inputs.claim_consensus_merge_mode
            answer_prompt = prepared_inputs.answer_prompt
            safe_context = prepared_inputs.safe_context
            external_policy = prepared_inputs.external_policy
            original_classification = prepared_inputs.original_classification
            flow_result = self._runtime_flow_result(
                query=query,
                retrieval_mode=retrieval_mode,
                pipeline_result=pipeline_result,
                evidence_packet=evidence_packet,
                selected_llm=selected_llm,
                claim_verification=claim_verification,
                claim_consensus=claim_consensus,
                claim_consensus_merge_mode=claim_consensus_merge_mode,
                answer_prompt=answer_prompt,
                safe_context=safe_context,
                external_policy=external_policy,
                original_classification=original_classification,
                allow_external=allow_external,
                routing_meta=routing_meta,
                routing_warnings=routing_warnings,
            )
            self._record_answer_log(
                query=query,
                payload=flow_result.payload,
                source_type=source_type,
                retrieval_mode=retrieval_mode,
                allow_external=allow_external,
            )
            return flow_result.payload
        finally:
            self.searcher._caches.write_active_request_llm(self.searcher, previous_active_request_llm)

    def stream(
        self,
        *,
        query: str,
        source_type: str | None,
        retrieval_mode: str,
        allow_external: bool,
        answer_route_override: str | None = None,
        pipeline_result: Any,
        evidence_packet: Any,
    ):
        early_exit = self._early_exit_result(
            query=query,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            allow_external=allow_external,
            pipeline_result=pipeline_result,
            evidence_packet=evidence_packet,
        )
        if early_exit is not None:
            yield early_exit.answer
            return

        claim_verification, claim_consensus, claim_context = self._adjudicate_claims(evidence_packet=evidence_packet)
        selected_llm, routing_meta, _routing_warnings = self._resolve_llm_for_request(
            query=query,
            context=self._with_claim_context(evidence_packet.context, claim_context),
            source_count=len(evidence_packet.filtered_results),
            allow_external=allow_external,
            force_route=answer_route_override,
        )
        self.searcher._caches.sync_active_request_llm_from_searcher(self.searcher)
        previous_active_request_llm = self.searcher._caches.active_request_llm_value()
        self.searcher._caches.write_active_request_llm(self.searcher, selected_llm)
        try:
            route_mode = str(routing_meta.get("route", "fixed"))
            prepared_inputs = self._prepare_answer_execution_inputs(
                query=query,
                pipeline_result=pipeline_result,
                evidence_packet=evidence_packet,
                selected_llm=selected_llm,
                claim_verification=claim_verification,
                claim_consensus=claim_consensus,
                claim_context=claim_context,
                allow_external=allow_external,
                route_mode=route_mode,
            )
            claim_verification = prepared_inputs.claim_verification
            claim_consensus = prepared_inputs.claim_consensus
            claim_consensus_merge_mode = prepared_inputs.claim_consensus_merge_mode
            answer_prompt = prepared_inputs.answer_prompt
            safe_context = prepared_inputs.safe_context
            external_policy = prepared_inputs.external_policy
            original_classification = prepared_inputs.original_classification
            flow_result = self._runtime_flow_result(
                query=query,
                retrieval_mode=retrieval_mode,
                pipeline_result=pipeline_result,
                evidence_packet=evidence_packet,
                selected_llm=selected_llm,
                claim_verification=claim_verification,
                claim_consensus=claim_consensus,
                claim_consensus_merge_mode=claim_consensus_merge_mode,
                answer_prompt=answer_prompt,
                safe_context=safe_context,
                external_policy=external_policy,
                original_classification=original_classification,
                allow_external=allow_external,
                routing_meta=routing_meta,
                routing_warnings=_routing_warnings,
                stream=True,
            )
            self._record_answer_log(
                query=query,
                payload=flow_result.payload,
                source_type=source_type,
                retrieval_mode=retrieval_mode,
                allow_external=allow_external,
            )
            for chunk in re.findall(r".{1,256}", flow_result.answer_text, flags=re.DOTALL):
                yield chunk
        finally:
            self.searcher._caches.write_active_request_llm(self.searcher, previous_active_request_llm)


__all__ = ["AnswerOrchestrator"]
