from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Optional

from knowledge_hub.application.query_frame import NormalizedQueryFrame
from knowledge_hub.ai.ask_v2_support import classify_intent as classify_ask_v2_intent
from knowledge_hub.ai.ask_v2_support import paper_scope_from_filter, paper_scope_from_query
from knowledge_hub.ai.rag_decision_log import emit_rag_decision_log
from knowledge_hub.ai.memory_prefilter import MEMORY_ROUTE_MODE_OFF
from knowledge_hub.ai.retrieval_fit import normalize_source_type
from knowledge_hub.domain.registry import get_domain_pack
from knowledge_hub.domain.ai_papers.query_plan import (
    PAPER_FAMILY_COMPARE,
    PAPER_FAMILY_CONCEPT,
    PAPER_FAMILY_DISCOVER,
    PAPER_FAMILY_LOOKUP,
    build_rule_based_query_frame,
    build_rule_query_plan,
    maybe_apply_planner_fallback,
)
from knowledge_hub.domain.ai_papers.representative import local_title_prefix_hints
from knowledge_hub.papers.prefilter import PAPER_MEMORY_MODE_OFF


_PAPER_HINT_STOPWORDS = {
    "about",
    "concept",
    "define",
    "definition",
    "describe",
    "explain",
    "explainer",
    "idea",
    "ideas",
    "main",
    "meaning",
    "what",
    "개념",
    "대해",
    "대해서",
    "설명",
    "설명해",
    "설명해줘",
    "설명해주세요",
    "쉽게",
    "아이디어",
    "원리",
    "의미",
    "정의",
    "핵심",
}
_KOREAN_PARTICLE_SUFFIXES = (
    "으로",
    "에서",
    "에게",
    "한테",
    "처럼",
    "보다",
    "까지",
    "부터",
    "와",
    "과",
    "의",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "도",
    "만",
    "로",
)
_ASK_V2_COMPATIBILITY_METHODS = {
    "get_paper_card_v2",
    "get_related_concepts",
    "list_claim_card_source_refs",
    "list_claim_cards",
    "list_document_memory_units",
    "list_evidence_anchors_v2",
    "list_normalization_aliases",
    "list_paper_card_entity_refs_v2",
    "list_paper_cards_v2_by_entity_ids",
    "replace_paper_card_claim_refs_v2",
    "search_document_memory_units",
    "search_paper_cards_v2",
    "upsert_normalization_alias",
}
_ASK_V2_MISSING_ATTR_RE = re.compile(r"has no attribute '([^']+)'")


def _strip_korean_particle(token: str) -> str:
    value = str(token or "").strip()
    for suffix in _KOREAN_PARTICLE_SUFFIXES:
        if len(value) > len(suffix) + 1 and value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _paper_hint_entities(query: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9.+-]+|[가-힣]+", str(query or "").strip()):
        token = _strip_korean_particle(raw)
        lowered = token.casefold()
        if not token or lowered in _PAPER_HINT_STOPWORDS:
            continue
        if len(token) < 2:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        values.append(token)
        if len(values) >= 4:
            break
    return values


@dataclass(frozen=True)
class AnswerRuntimeRequest:
    query: str
    top_k: int = 5
    min_score: float = 0.0
    source_type: Optional[str] = None
    retrieval_mode: str = "hybrid"
    alpha: float = 0.7
    allow_external: bool = False
    memory_route_mode: str = MEMORY_ROUTE_MODE_OFF
    paper_memory_mode: str = PAPER_MEMORY_MODE_OFF
    metadata_filter: Optional[dict[str, Any]] = None
    ask_v2_mode: Optional[str] = None
    answer_route_override: Optional[str] = None
    query_plan: Optional[dict[str, Any]] = None
    query_frame: Optional[NormalizedQueryFrame] = None


@dataclass(frozen=True)
class AnswerRuntimeExecution:
    pipeline_result: Any
    evidence_packet: Any


class RAGAnswerRuntime:
    def __init__(self, searcher: Any):
        self.searcher = searcher

    @staticmethod
    def build_request(
        *,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        source_type: Optional[str] = None,
        retrieval_mode: str = "hybrid",
        alpha: float = 0.7,
        allow_external: bool = False,
        memory_route_mode: str = MEMORY_ROUTE_MODE_OFF,
        paper_memory_mode: str = PAPER_MEMORY_MODE_OFF,
        metadata_filter: Optional[dict[str, Any]] = None,
        ask_v2_mode: Optional[str] = None,
        answer_route_override: Optional[str] = None,
        query_plan: Optional[dict[str, Any]] = None,
        query_frame: Optional[NormalizedQueryFrame] = None,
    ) -> AnswerRuntimeRequest:
        return AnswerRuntimeRequest(
            query=query,
            top_k=top_k,
            min_score=min_score,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            allow_external=allow_external,
            memory_route_mode=memory_route_mode,
            paper_memory_mode=paper_memory_mode,
            metadata_filter=metadata_filter,
            ask_v2_mode=ask_v2_mode,
            answer_route_override=answer_route_override,
            query_plan=query_plan,
            query_frame=query_frame,
        )

    @staticmethod
    def _normalize_ask_v2_mode(mode: str | None) -> str:
        return str(mode or "").strip().lower()

    @staticmethod
    def _legacy_mode_error() -> ValueError:
        return ValueError("ask_v2_mode='legacy' was removed; use auto/default routing instead.")

    @staticmethod
    def _has_single_paper_scope(request: AnswerRuntimeRequest) -> bool:
        return bool(paper_scope_from_filter(request.metadata_filter))

    def _maybe_infer_paper_context(self, request: AnswerRuntimeRequest) -> AnswerRuntimeRequest:
        if normalize_source_type(request.source_type):
            return request
        if request.query_frame is not None or (isinstance(request.query_plan, dict) and request.query_plan):
            return request
        if classify_ask_v2_intent(request.query, request.metadata_filter) != "definition":
            return request
        entities = _paper_hint_entities(request.query)
        if not entities:
            return request
        hints = local_title_prefix_hints(entities, sqlite_db=self.searcher.sqlite_db)
        if len(hints) != 1:
            return request
        query_frame = build_rule_based_query_frame(
            request.query,
            source_type="paper",
            metadata_filter=request.metadata_filter,
            sqlite_db=self.searcher.sqlite_db,
        )
        if len(list(query_frame.resolved_source_ids or [])) != 1:
            return request
        inferred_filter = dict(request.metadata_filter or {})
        inferred_filter.setdefault("arxiv_id", str(list(query_frame.resolved_source_ids)[0]).strip())
        return replace(
            request,
            source_type="paper",
            metadata_filter=inferred_filter,
            query_plan=query_frame.to_query_plan_dict(),
            query_frame=query_frame,
        )

    def _maybe_infer_vault_context(self, request: AnswerRuntimeRequest) -> AnswerRuntimeRequest:
        if normalize_source_type(request.source_type):
            return request
        if request.query_frame is not None or (isinstance(request.query_plan, dict) and request.query_plan):
            return request
        from knowledge_hub.domain.vault_knowledge.families import explicit_vault_scope
        from knowledge_hub.domain.vault_knowledge.query_plan import build_rule_based_query_frame as build_vault_query_frame

        if not explicit_vault_scope(request.query, metadata_filter=request.metadata_filter):
            return request
        query_frame = build_vault_query_frame(
            request.query,
            source_type="vault",
            metadata_filter=request.metadata_filter,
            sqlite_db=self.searcher.sqlite_db,
        )
        return replace(
            request,
            source_type="vault",
            metadata_filter=dict(query_frame.metadata_filter or request.metadata_filter or {}),
            query_plan=query_frame.to_query_plan_dict(),
            query_frame=query_frame,
        )

    def _ensure_query_context(self, request: AnswerRuntimeRequest) -> AnswerRuntimeRequest:
        request = self._maybe_infer_vault_context(request)
        request = self._maybe_infer_paper_context(request)
        if request.query_frame is not None and isinstance(request.query_plan, dict) and request.query_plan:
            return request
        domain_pack = get_domain_pack(source_type=request.source_type)
        normalized_source = normalize_source_type(request.source_type)
        query_frame = request.query_frame
        query_plan = dict(request.query_plan or {})

        if query_frame is None:
            if domain_pack is not None and hasattr(domain_pack, "normalize"):
                try:
                    query_frame = domain_pack.normalize(
                        request.query,
                        source_type=request.source_type,
                        metadata_filter=request.metadata_filter,
                        sqlite_db=self.searcher.sqlite_db,
                        query_plan=query_plan or None,
                    )
                except Exception:
                    query_frame = None
            else:
                if query_plan:
                    from knowledge_hub.domain.ai_papers.query_plan import query_frame_from_query_plan

                    query_frame = query_frame_from_query_plan(
                        query_plan,
                        query=request.query,
                        source_type=request.source_type,
                        metadata_filter=request.metadata_filter,
                        sqlite_db=self.searcher.sqlite_db,
                    )
                elif normalized_source == "paper":
                    query_frame = build_rule_based_query_frame(
                        request.query,
                        source_type=request.source_type,
                        metadata_filter=request.metadata_filter,
                        sqlite_db=self.searcher.sqlite_db,
                    )

        if not query_plan:
            if isinstance(query_frame, NormalizedQueryFrame):
                query_plan = query_frame.to_query_plan_dict()
            elif normalized_source == "paper":
                query_plan = build_rule_query_plan(
                    request.query,
                    source_type=request.source_type,
                    metadata_filter=request.metadata_filter,
                    sqlite_db=self.searcher.sqlite_db,
                ).to_dict()
        return replace(request, query_plan=query_plan, query_frame=query_frame)

    def _maybe_apply_planner_before_runtime(self, request: AnswerRuntimeRequest) -> AnswerRuntimeRequest:
        query_plan = dict(request.query_plan or {})
        if not query_plan:
            return request
        if normalize_source_type(request.source_type) != "paper":
            return request
        updated = maybe_apply_planner_fallback(
            query=request.query,
            query_plan=query_plan,
            searcher=self.searcher,
            allow_external=request.allow_external,
            reason="low_confidence",
        )
        if updated == query_plan:
            return request
        domain_pack = get_domain_pack(source_type=request.source_type)
        query_frame = request.query_frame
        if domain_pack is not None and hasattr(domain_pack, "normalize"):
            query_frame = domain_pack.normalize(
                request.query,
                source_type=request.source_type,
                metadata_filter=request.metadata_filter,
                sqlite_db=self.searcher.sqlite_db,
                query_plan=updated,
            )
        return replace(request, query_plan=updated, query_frame=query_frame)

    def _family_prefers_ask_v2(self, request: AnswerRuntimeRequest) -> bool:
        _ = (
            (request.query_frame.family if request.query_frame else "")
            or (request.query_plan or {}).get("family")
            or classify_ask_v2_intent(request.query, request.metadata_filter)
            or PAPER_FAMILY_LOOKUP
            or PAPER_FAMILY_COMPARE
            or PAPER_FAMILY_CONCEPT
            or PAPER_FAMILY_DISCOVER
        )
        return True

    def _ask_v2_gate_decision(self, request: AnswerRuntimeRequest) -> tuple[bool, str]:
        ask_v2_mode = self._normalize_ask_v2_mode(request.ask_v2_mode)
        if ask_v2_mode == "legacy":
            raise self._legacy_mode_error()
        if bool(
            self.searcher._should_use_ask_v2(
                source_type=request.source_type,
                metadata_filter=request.metadata_filter,
                sqlite_db=self.searcher.sqlite_db,
            )
        ):
            return True, ""
        if normalize_source_type(request.source_type) == "paper":
            # Paper is promoted by default. Keep compatibility missing as a
            # runtime fallback, not a family-level pre-route to legacy.
            return True, ""
        return False, "ask_v2_capability_missing"

    @staticmethod
    def _ask_v2_capability_fallback_reason(error: Exception) -> str:
        if isinstance(error, AttributeError):
            match = _ASK_V2_MISSING_ATTR_RE.search(str(error or ""))
            if match and match.group(1) in _ASK_V2_COMPATIBILITY_METHODS:
                return "ask_v2_capability_missing"
        if isinstance(error, NotImplementedError):
            message = str(error or "")
            if any(method in message for method in _ASK_V2_COMPATIBILITY_METHODS):
                return "ask_v2_capability_missing"
        return ""

    def _should_use_ask_v2(self, request: AnswerRuntimeRequest) -> bool:
        should_use, _fallback_reason = self._ask_v2_gate_decision(request)
        return should_use

    def _try_ask_v2_execution(self, request: AnswerRuntimeRequest) -> tuple[AnswerRuntimeExecution | None, str]:
        should_use, fallback_reason = self._ask_v2_gate_decision(request)
        if not should_use:
            return None, fallback_reason

        from knowledge_hub.ai.ask_v2 import AskV2FallbackToLegacy, AskV2Service

        ask_v2_mode = self._normalize_ask_v2_mode(request.ask_v2_mode) or None
        try:
            pipeline_result, evidence_packet = AskV2Service(self.searcher).execute(
                query=request.query,
                top_k=request.top_k,
                source_type=request.source_type,
                retrieval_mode=request.retrieval_mode,
                alpha=request.alpha,
                allow_external=request.allow_external,
                metadata_filter=request.metadata_filter,
                ask_v2_mode=ask_v2_mode,
                query_plan=request.query_plan,
                query_frame=request.query_frame,
            )
        except AskV2FallbackToLegacy as error:
            return None, str(error or "").strip() or "ask_v2_not_used"
        except (AttributeError, NotImplementedError) as error:
            fallback_reason = self._ask_v2_capability_fallback_reason(error)
            if fallback_reason:
                return None, fallback_reason
            raise
        return (
            AnswerRuntimeExecution(
                pipeline_result=pipeline_result,
                evidence_packet=evidence_packet,
            ),
            "",
        )

    def _generate_via_orchestrator(
        self,
        *,
        request: AnswerRuntimeRequest,
        execution: AnswerRuntimeExecution,
    ) -> dict[str, Any]:
        from knowledge_hub.ai.answer_orchestrator import AnswerOrchestrator

        return AnswerOrchestrator(self.searcher).generate(
            query=request.query,
            source_type=request.source_type,
            retrieval_mode=request.retrieval_mode,
            allow_external=request.allow_external,
            answer_route_override=request.answer_route_override,
            pipeline_result=execution.pipeline_result,
            evidence_packet=execution.evidence_packet,
        )

    def _stream_via_orchestrator(
        self,
        *,
        request: AnswerRuntimeRequest,
        execution: AnswerRuntimeExecution,
    ):
        from knowledge_hub.ai.answer_orchestrator import AnswerOrchestrator

        yield from AnswerOrchestrator(self.searcher).stream(
            query=request.query,
            source_type=request.source_type,
            retrieval_mode=request.retrieval_mode,
            allow_external=request.allow_external,
            answer_route_override=request.answer_route_override,
            pipeline_result=execution.pipeline_result,
            evidence_packet=execution.evidence_packet,
        )

    @staticmethod
    def _mark_non_ask_v2_runtime(
        pipeline_result: Any,
        *,
        fallback_reason: str = "ask_v2_not_used",
    ) -> None:
        diagnostics = dict(getattr(pipeline_result, "v2_diagnostics", {}) or {})
        diagnostics["runtimeExecution"] = {
            "used": "legacy",
            "sectionDecision": "skipped",
            "sectionBlockReason": "",
            "fallbackReason": str(fallback_reason or "ask_v2_not_used").strip() or "ask_v2_not_used",
        }
        pipeline_result.v2_diagnostics = diagnostics

    def _build_non_ask_v2_execution(
        self,
        *,
        request: AnswerRuntimeRequest,
        fallback_reason: str = "ask_v2_not_used",
    ) -> AnswerRuntimeExecution:
        from knowledge_hub.ai.evidence_assembly import EvidenceAssemblyService
        from knowledge_hub.ai.retrieval_pipeline import RetrievalPipelineService

        pipeline_result = RetrievalPipelineService(self.searcher).execute(
            query=request.query,
            top_k=request.top_k,
            source_type=request.source_type,
            retrieval_mode=request.retrieval_mode,
            alpha=request.alpha,
            metadata_filter=request.metadata_filter,
            memory_route_mode=request.memory_route_mode,
            paper_memory_mode=request.paper_memory_mode,
            min_score=request.min_score,
            query_plan=request.query_plan,
            query_frame=request.query_frame,
        )
        evidence_packet = EvidenceAssemblyService.from_searcher(self.searcher).assemble(
            query=request.query,
            source_type=request.source_type,
            results=pipeline_result.results,
            paper_memory_prefilter=pipeline_result.paper_memory_prefilter,
            metadata_filter=dict(
                getattr(getattr(pipeline_result, "plan", None), "metadata_filter_applied", {})
                or request.metadata_filter
                or {}
            ),
            query_plan=request.query_plan,
            query_frame=request.query_frame,
        )
        self._mark_non_ask_v2_runtime(pipeline_result, fallback_reason=fallback_reason)
        return AnswerRuntimeExecution(
            pipeline_result=pipeline_result,
            evidence_packet=evidence_packet,
        )

    @staticmethod
    def _normalized_runtime_execution(runtime_execution: dict[str, Any] | None) -> dict[str, Any]:
        runtime = dict(runtime_execution or {})
        return {
            "used": str(runtime.get("used") or "").strip(),
            "fallbackReason": str(runtime.get("fallbackReason") or "").strip(),
            "sectionDecision": str(runtime.get("sectionDecision") or "").strip(),
            "sectionBlockReason": str(runtime.get("sectionBlockReason") or "").strip(),
        }

    def _runtime_execution_for_log(
        self,
        *,
        payload: dict[str, Any] | None,
        execution: AnswerRuntimeExecution | None,
        fallback_reason: str = "",
    ) -> dict[str, Any]:
        payload = dict(payload or {})
        payload_runtime = dict(dict(payload.get("v2") or {}).get("runtimeExecution") or {})
        if payload_runtime:
            return self._normalized_runtime_execution(payload_runtime)
        if execution is not None:
            diagnostics = dict(getattr(execution.pipeline_result, "v2_diagnostics", {}) or {})
            runtime = dict(diagnostics.get("runtimeExecution") or {})
            if runtime:
                return self._normalized_runtime_execution(runtime)
        if str(fallback_reason or "").strip():
            return self._normalized_runtime_execution(
                {
                    "used": "legacy",
                    "fallbackReason": fallback_reason,
                    "sectionDecision": "skipped",
                    "sectionBlockReason": "",
                }
            )
        return self._normalized_runtime_execution(None)

    def _emit_decision_log(
        self,
        *,
        stage: str,
        request: AnswerRuntimeRequest,
        payload: dict[str, Any] | None = None,
        execution: AnswerRuntimeExecution | None = None,
        fallback_reason: str = "",
    ) -> None:
        payload = dict(payload or {})
        diagnostics = execution.pipeline_result.diagnostics() if execution is not None else {}
        retrieval_plan = dict(diagnostics.get("retrievalPlan") or {})
        emit_rag_decision_log(
            stage=stage,
            query=request.query,
            source_type=request.source_type,
            query_plan=payload.get("queryPlan") or retrieval_plan.get("queryPlan") or request.query_plan,
            query_frame=payload.get("queryFrame") or retrieval_plan.get("queryFrame") or request.query_frame,
            memory_route=payload.get("memoryRoute") or diagnostics.get("memoryRoute"),
            memory_prefilter=payload.get("memoryPrefilter") or diagnostics.get("memoryPrefilter"),
            runtime_execution=self._runtime_execution_for_log(
                payload=payload,
                execution=execution,
                fallback_reason=fallback_reason,
            ),
        )

    def generate(self, *, request: AnswerRuntimeRequest) -> dict[str, Any]:
        request = self._ensure_query_context(request)
        request = self._maybe_apply_planner_before_runtime(request)
        execution, fallback_reason = self._try_ask_v2_execution(request)
        if execution is not None:
            payload = self._generate_via_orchestrator(request=request, execution=execution)
        else:
            execution = self._build_non_ask_v2_execution(request=request, fallback_reason=fallback_reason)
            payload = self._generate_via_orchestrator(request=request, execution=execution)

        if normalize_source_type(request.source_type) == "paper" and str(payload.get("status") or "").strip().lower() == "no_result":
            replanned = maybe_apply_planner_fallback(
                query=request.query,
                query_plan=request.query_plan or {},
                searcher=self.searcher,
                allow_external=request.allow_external,
                reason="no_result",
            )
            if bool(replanned.get("plannerUsed") or replanned.get("planner_used")) and replanned != (request.query_plan or {}):
                retry_request = self._ensure_query_context(replace(request, query_plan=replanned, query_frame=None))
                retry_execution, retry_fallback_reason = self._try_ask_v2_execution(retry_request)
                if retry_execution is not None:
                    payload = self._generate_via_orchestrator(request=retry_request, execution=retry_execution)
                    self._emit_decision_log(
                        stage="generate_answer_runtime",
                        request=retry_request,
                        payload=payload,
                        execution=retry_execution,
                    )
                    return payload
                retry_execution = self._build_non_ask_v2_execution(
                    request=retry_request,
                    fallback_reason=retry_fallback_reason,
                )
                payload = self._generate_via_orchestrator(request=retry_request, execution=retry_execution)
                self._emit_decision_log(
                    stage="generate_answer_runtime",
                    request=retry_request,
                    payload=payload,
                    execution=retry_execution,
                    fallback_reason=retry_fallback_reason,
                )
                return payload
        self._emit_decision_log(
            stage="generate_answer_runtime",
            request=request,
            payload=payload,
            execution=execution,
            fallback_reason=fallback_reason,
        )
        return payload

    def stream(self, *, request: AnswerRuntimeRequest):
        request = self._ensure_query_context(request)
        request = self._maybe_apply_planner_before_runtime(request)
        execution, fallback_reason = self._try_ask_v2_execution(request)
        if execution is None:
            execution = self._build_non_ask_v2_execution(request=request, fallback_reason=fallback_reason)
        self._emit_decision_log(
            stage="stream_answer_runtime",
            request=request,
            execution=execution,
            fallback_reason=fallback_reason,
        )
        yield from self._stream_via_orchestrator(request=request, execution=execution)


def generate_answer(
    searcher: Any,
    *,
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    source_type: Optional[str] = None,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.7,
    allow_external: bool = False,
    memory_route_mode: str = MEMORY_ROUTE_MODE_OFF,
    paper_memory_mode: str = PAPER_MEMORY_MODE_OFF,
    metadata_filter: Optional[dict[str, Any]] = None,
    ask_v2_mode: Optional[str] = None,
    answer_route_override: Optional[str] = None,
    query_plan: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    runtime = RAGAnswerRuntime(searcher)
    request = runtime.build_request(
        query=query,
        top_k=top_k,
        min_score=min_score,
        source_type=source_type,
        retrieval_mode=retrieval_mode,
        alpha=alpha,
        allow_external=allow_external,
        memory_route_mode=memory_route_mode,
        paper_memory_mode=paper_memory_mode,
        metadata_filter=metadata_filter,
        ask_v2_mode=ask_v2_mode,
        answer_route_override=answer_route_override,
        query_plan=query_plan,
    )
    return runtime.generate(request=request)


def stream_answer(
    searcher: Any,
    *,
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    source_type: Optional[str] = None,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.7,
    allow_external: bool = False,
    memory_route_mode: str = MEMORY_ROUTE_MODE_OFF,
    paper_memory_mode: str = PAPER_MEMORY_MODE_OFF,
    metadata_filter: Optional[dict[str, Any]] = None,
    ask_v2_mode: Optional[str] = None,
    answer_route_override: Optional[str] = None,
    query_plan: Optional[dict[str, Any]] = None,
):
    runtime = RAGAnswerRuntime(searcher)
    request = runtime.build_request(
        query=query,
        top_k=top_k,
        min_score=min_score,
        source_type=source_type,
        retrieval_mode=retrieval_mode,
        alpha=alpha,
        allow_external=allow_external,
        memory_route_mode=memory_route_mode,
        paper_memory_mode=paper_memory_mode,
        metadata_filter=metadata_filter,
        ask_v2_mode=ask_v2_mode,
        answer_route_override=answer_route_override,
        query_plan=query_plan,
    )
    yield from runtime.stream(request=request)


__all__ = [
    "AnswerRuntimeExecution",
    "AnswerRuntimeRequest",
    "RAGAnswerRuntime",
    "generate_answer",
    "stream_answer",
]
