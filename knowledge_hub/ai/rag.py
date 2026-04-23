"""
RAG (Retrieval-Augmented Generation) 파이프라인

통합 지식 검색 + 답변 생성
- Obsidian 노트
- 논문
- 웹 문서
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from knowledge_hub.core.models import SearchResult
from knowledge_hub.infrastructure.persistence import VectorDatabase
from knowledge_hub.knowledge.graph_signals import analyze_graph_query  # noqa: F401 - compatibility patch point
from knowledge_hub.ai.memory_prefilter import (
    MEMORY_ROUTE_MODE_OFF,
    execute_memory_prefilter,
)
from knowledge_hub.ai.rag_answer_evidence import (
    answer_evidence_item,
    summarize_answer_signals,
)
from knowledge_hub.ai.rag_ask_v2_gate import should_use_ask_v2
from knowledge_hub.ai.rag_answer_route_resolver import (
    resolve_llm_for_request,
    resolve_llm_for_rewrite,
    resolve_llm_for_verification,
)
from knowledge_hub.ai.rag_context import RAGCaches, RAGContext
from knowledge_hub.ai.rag_answer_runtime import (
    generate_answer as generate_answer_runtime,
    stream_answer as stream_answer_runtime,
)
from knowledge_hub.ai.rag_evidence_context import (
    collect_claim_context,
    resolve_result_note_row,
    resolve_result_quality,
)
from knowledge_hub.ai.rag_paper_prefilter import search_with_paper_memory_prefilter
from knowledge_hub.ai.rag_retrieval_signals import build_retrieval_ranking_signals
from knowledge_hub.ai.rag_ranking import retrieval_sort_key, retrieval_sort_score
from knowledge_hub.ai.rag_scope import (
    apply_profile_and_cluster_scope,
    get_active_profile,
    load_topology_index,
    resolve_query_entities,
    resolve_topology_snapshot_path,
)
from knowledge_hub.learning.model_router import get_llm_for_hybrid_routing
from knowledge_hub.papers.prefilter import (
    PAPER_MEMORY_MODE_OFF,
    resolve_paper_memory_prefilter,
)
from knowledge_hub.ai.rag_support import (
    answer_mentions_conflict as _answer_mentions_conflict_impl,
    build_answer_context as _build_answer_context_impl,
    build_paper_definition_context as _build_paper_definition_context_impl,
    build_answer_generation_fallback as _build_answer_generation_fallback_impl,
    build_answer_prompt as _build_answer_prompt_impl,
    build_claim_native_context as _build_claim_native_context_impl,
    build_claim_native_prompt as _build_claim_native_prompt_impl,
    build_section_native_context as _build_section_native_context_impl,
    build_section_native_prompt as _build_section_native_prompt_impl,
    build_answer_rewrite_context as _build_answer_rewrite_context_impl,
    build_answer_rewrite_prompt as _build_answer_rewrite_prompt_impl,
    build_answer_verification_context as _build_answer_verification_context_impl,
    build_answer_verification_prompt as _build_answer_verification_prompt_impl,
    build_conservative_answer as _build_conservative_answer_impl,
    default_answer_rewrite as _default_answer_rewrite_impl,
    json_load_dict as _json_load_dict,
    merge_search_results as _merge_search_results_impl,
    merge_top_signal_items as _merge_top_signal_items,
    normalize_source_type as _normalize_source_type,
    note_id_for_result as _note_id_for_result,
    record_answer_log as _record_answer_log_impl,
    result_id as _result_id,
    rewrite_route_stub as _rewrite_route_stub_impl,
    route_summary as _route_summary_impl,
    safe_float as _safe_float,
    should_rewrite_answer as _should_rewrite_answer_impl,
    source_label_for_result as _source_label_for_result,
    split_answer_claims as _split_answer_claims_impl,
    tokenize as _tokenize,
    verification_route_stub as _verification_route_stub_impl,
)


class RAGSearcher:
    """통합 RAG 검색 및 답변 생성

    embedder: embed_text(str) -> List[float] 를 제공하는 객체
    llm: generate(prompt, context) 를 제공하는 객체
    """

    def __init__(
        self,
        embedder,
        database: VectorDatabase,
        llm=None,
        sqlite_db=None,
        config=None,
    ):
        self.embedder = embedder
        self.database = database
        self.llm = llm
        self.sqlite_db = sqlite_db  # SQLite DB for ontology access
        self.config = config
        self.parent_fetch_limit = 256
        self.parent_window_chunks = 3
        self._cached_local_llm = None
        self._cached_local_llm_signature: tuple[str, str, int] | None = None
        self._topology_cache: dict[str, Any] | None = None
        self._profile_cache: dict[str, Any] | None = None
        self._active_request_llm = None
        self._ctx = RAGContext.from_searcher(self)
        self._caches = RAGCaches.from_searcher(self)

    @staticmethod
    def _route_summary(route: dict[str, Any] | None) -> dict[str, Any]:
        return _route_summary_impl(route)

    @staticmethod
    def _merge_search_results(results: list[SearchResult], *, top_k: int) -> list[SearchResult]:
        return _merge_search_results_impl(results, top_k=top_k)

    @staticmethod
    def _retrieval_sort_score(result: SearchResult) -> float:
        return retrieval_sort_score(result)

    @classmethod
    def _retrieval_sort_key(
        cls,
        result: SearchResult,
        *,
        prefer_lexical: bool = False,
    ) -> tuple[float, float, float, float]:
        return retrieval_sort_key(result, prefer_lexical=prefer_lexical)

    def _search_with_memory_prefilter(
        self,
        *,
        query: str,
        top_k: int,
        source_type: Optional[str],
        retrieval_mode: str,
        alpha: float,
        min_score: float,
        requested_mode: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        execution = execute_memory_prefilter(
            self,
            query=query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            min_score=min_score,
            requested_mode=requested_mode,
            metadata_filter=metadata_filter,
            result_id_fn=_result_id,
        )
        return execution.results, execution.diagnostics

    def _search_with_paper_memory_prefilter(
        self,
        *,
        query: str,
        top_k: int,
        source_type: Optional[str],
        retrieval_mode: str,
        alpha: float,
        min_score: float,
        requested_mode: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        return search_with_paper_memory_prefilter(
            self,
            query=query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            min_score=min_score,
            requested_mode=requested_mode,
            metadata_filter=metadata_filter,
            resolve_paper_memory_prefilter_fn=resolve_paper_memory_prefilter,
            merge_search_results_fn=lambda results: self._merge_search_results(results, top_k=top_k),
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
        _record_answer_log_impl(
            getattr(self.sqlite_db, "add_rag_answer_log", None),
            query=query,
            payload=payload,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            allow_external=allow_external,
        )

    def _get_active_profile(self) -> dict[str, Any] | None:
        self._profile_cache = get_active_profile(
            sqlite_db=self.sqlite_db,
            cached_profile=self._profile_cache,
        )
        return self._profile_cache

    def _resolve_topology_snapshot_path(self) -> Path | None:
        return resolve_topology_snapshot_path(self.config)

    def _load_topology_index(self) -> dict[str, Any] | None:
        self._topology_cache = load_topology_index(
            config=self.config,
            cached_topology=self._topology_cache,
        )
        return self._topology_cache

    def _apply_profile_and_cluster_scope(
        self,
        results: list[SearchResult],
        query: str,
        top_k: int,
        *,
        apply_score_boosts: bool = True,
    ) -> tuple[list[SearchResult], list[dict[str, Any]], dict[str, Any] | None]:
        _ = query
        return apply_profile_and_cluster_scope(
            results,
            profile=self._get_active_profile(),
            topology=self._load_topology_index(),
            top_k=top_k,
            apply_score_boosts=apply_score_boosts,
            safe_float_fn=_safe_float,
            source_label_for_result_fn=_source_label_for_result,
            merge_top_signal_items_fn=_merge_top_signal_items,
            retrieval_sort_key_fn=self._retrieval_sort_key,
        )

    def _resolve_query_entities(self, query: str, max_related: int = 8) -> list[dict[str, Any]]:
        return resolve_query_entities(
            query,
            sqlite_db=self.sqlite_db,
            tokenize_fn=_tokenize,
            max_related=max_related,
        )

    def _collect_claim_context(self, results: list[SearchResult]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        return collect_claim_context(
            results,
            sqlite_db=self.sqlite_db,
            note_id_for_result_fn=_note_id_for_result,
        )

    def _resolve_result_note_row(self, result: SearchResult) -> dict[str, Any] | None:
        return resolve_result_note_row(
            result,
            sqlite_db=self.sqlite_db,
            note_id_for_result_fn=_note_id_for_result,
        )

    def _resolve_result_quality(self, result: SearchResult) -> tuple[str, dict[str, Any]]:
        return resolve_result_quality(
            result,
            sqlite_db=self.sqlite_db,
            config=self.config,
            resolve_result_note_row_fn=self._resolve_result_note_row,
            json_load_dict_fn=_json_load_dict,
        )

    def _build_retrieval_ranking_signals(self, result: SearchResult, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        return build_retrieval_ranking_signals(
            result,
            snapshot=snapshot,
            resolve_result_note_row_fn=self._resolve_result_note_row,
            resolve_result_quality_fn=self._resolve_result_quality,
            json_load_dict_fn=_json_load_dict,
            safe_float_fn=_safe_float,
        )

    def _answer_evidence_item(self, result: SearchResult, parent_ctx_by_result: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return answer_evidence_item(
            result,
            parent_ctx_by_result=parent_ctx_by_result,
            result_id_fn=_result_id,
            normalize_source_type_fn=_normalize_source_type,
            safe_float_fn=_safe_float,
        )

    def _summarize_answer_signals(
        self,
        evidence: list[dict[str, Any]],
        *,
        contradicting_beliefs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return summarize_answer_signals(
            evidence,
            contradicting_beliefs=contradicting_beliefs,
            safe_float_fn=_safe_float,
        )

    def _build_answer_prompt(
        self,
        *,
        query: str,
        answer_signals: dict[str, Any],
    ) -> str:
        return _build_answer_prompt_impl(query=query, answer_signals=answer_signals)

    def _build_answer_context(
        self,
        *,
        filtered: list[SearchResult],
        parent_ctx_by_result: dict[str, dict[str, Any]],
    ) -> str:
        return _build_answer_context_impl(
            filtered=filtered,
            parent_ctx_by_result=parent_ctx_by_result,
        )

    def _build_paper_definition_context(
        self,
        *,
        query: str,
        filtered: list[SearchResult],
        evidence: list[dict[str, Any]],
        answer_signals: dict[str, Any],
        claim_context: str = "",
    ) -> str:
        return _build_paper_definition_context_impl(
            query=query,
            filtered=filtered,
            evidence=evidence,
            answer_signals=answer_signals,
            claim_context=claim_context,
        )

    def _build_claim_native_prompt(self, *, query: str, answer_provenance: str) -> str:
        return _build_claim_native_prompt_impl(query=query, answer_provenance=answer_provenance)

    def _build_claim_native_context(
        self,
        *,
        claim_cards: list[dict[str, Any]],
        claim_alignment: list[dict[str, Any]],
        claim_verification: list[dict[str, Any]],
        comparison_verification: dict[str, Any] | None = None,
        scope_warnings: list[str] | None = None,
        abstention_conditions: list[str] | None = None,
        supplemental_context: str = "",
    ) -> str:
        return _build_claim_native_context_impl(
            claim_cards=claim_cards,
            claim_alignment=claim_alignment,
            claim_verification=claim_verification,
            comparison_verification=comparison_verification,
            scope_warnings=scope_warnings,
            abstention_conditions=abstention_conditions,
            supplemental_context=supplemental_context,
        )

    def _build_section_native_prompt(self, *, query: str, answer_provenance: str) -> str:
        return _build_section_native_prompt_impl(query=query, answer_provenance=answer_provenance)

    def _build_section_native_context(
        self,
        *,
        section_cards: list[dict[str, Any]],
        section_coverage: dict[str, Any] | None = None,
        supplemental_context: str = "",
    ) -> str:
        return _build_section_native_context_impl(
            section_cards=section_cards,
            section_coverage=section_coverage,
            supplemental_context=supplemental_context,
        )

    @staticmethod
    def _answer_mentions_conflict(answer: str) -> bool:
        return _answer_mentions_conflict_impl(answer)

    @staticmethod
    def _split_answer_claims(answer: str) -> list[str]:
        return _split_answer_claims_impl(answer)

    def _verification_route_stub(self, *, reason: str, route: str = "fallback-only") -> dict[str, Any]:
        return _verification_route_stub_impl(reason=reason, route=route)

    def _rewrite_route_stub(self, *, reason: str, route: str = "fallback-only") -> dict[str, Any]:
        return _rewrite_route_stub_impl(reason=reason, route=route)

    def _resolve_llm_for_verification(
        self,
        *,
        query: str,
        context: str,
        source_count: int,
        allow_external: bool,
    ) -> tuple[Any | None, dict[str, Any], list[str]]:
        return resolve_llm_for_verification(
            config=self.config,
            query=query,
            context=context,
            source_count=source_count,
            allow_external=allow_external,
            verification_route_stub_fn=self._verification_route_stub,
        )

    def _resolve_llm_for_rewrite(
        self,
        *,
        query: str,
        context: str,
        source_count: int,
        allow_external: bool,
    ) -> tuple[Any | None, dict[str, Any], list[str]]:
        return resolve_llm_for_rewrite(
            config=self.config,
            query=query,
            context=context,
            source_count=source_count,
            allow_external=allow_external,
            rewrite_route_stub_fn=self._rewrite_route_stub,
        )

    def _build_answer_verification_context(
        self,
        *,
        evidence: list[dict[str, Any]],
        answer_signals: dict[str, Any],
        contradicting_beliefs: list[dict[str, Any]],
    ) -> str:
        return _build_answer_verification_context_impl(
            evidence=evidence,
            answer_signals=answer_signals,
            contradicting_beliefs=contradicting_beliefs,
        )

    def _build_answer_verification_prompt(
        self,
        *,
        query: str,
        answer: str,
    ) -> str:
        return _build_answer_verification_prompt_impl(query=query, answer=answer)

    def _default_answer_rewrite(self, *, answer: str, route: dict[str, Any] | None = None, summary: str = "") -> dict[str, Any]:
        return _default_answer_rewrite_impl(answer=answer, route=route, summary=summary)

    def _should_rewrite_answer(self, verification: dict[str, Any]) -> list[str]:
        return _should_rewrite_answer_impl(verification)

    def _build_answer_rewrite_context(
        self,
        *,
        evidence: list[dict[str, Any]],
        answer_signals: dict[str, Any],
        contradicting_beliefs: list[dict[str, Any]],
        verification: dict[str, Any],
    ) -> str:
        return _build_answer_rewrite_context_impl(
            evidence=evidence,
            answer_signals=answer_signals,
            contradicting_beliefs=contradicting_beliefs,
            verification=verification,
        )

    def _build_answer_rewrite_prompt(
        self,
        *,
        query: str,
        original_answer: str,
        verification: dict[str, Any],
        triggered_by: list[str],
    ) -> str:
        return _build_answer_rewrite_prompt_impl(
            query=query,
            original_answer=original_answer,
            verification=verification,
            triggered_by=triggered_by,
        )

    def _build_conservative_answer(
        self,
        *,
        verification: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> str:
        return _build_conservative_answer_impl(
            verification=verification,
            evidence=evidence,
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
        return _build_answer_generation_fallback_impl(
            query=query,
            error=error,
            stage=stage,
            evidence=evidence,
            citations=citations,
            routing_meta=routing_meta,
        )

    def _should_apply_conservative_fallback(self, verification: dict[str, Any]) -> bool:
        if not bool(verification.get("needsCaution")):
            return False
        if int(verification.get("unsupportedClaimCount") or 0) > 0:
            return True
        if int(verification.get("supportedClaimCount") or 0) == 0:
            return True
        return False

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
        from knowledge_hub.ai.answer_rewrite import apply_conservative_fallback_if_needed

        return apply_conservative_fallback_if_needed(
            self,
            query=query,
            answer=answer,
            rewrite_meta=rewrite_meta,
            verification=verification,
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
        from knowledge_hub.ai.answer_rewrite import rewrite_answer

        return rewrite_answer(
            self,
            query=query,
            answer=answer,
            evidence=evidence,
            answer_signals=answer_signals,
            verification=verification,
            contradicting_beliefs=contradicting_beliefs,
            allow_external=allow_external,
        )

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
        from knowledge_hub.ai.answer_verification import verify_answer

        return verify_answer(
            self,
            query=query,
            answer=answer,
            evidence=evidence,
            answer_signals=answer_signals,
            contradicting_beliefs=contradicting_beliefs,
            allow_external=allow_external,
        )

    def _apply_parent_context(
        self,
        results: List[SearchResult],
        include_parent_text: bool = True,
    ) -> List[SearchResult]:
        from knowledge_hub.ai.rag_parent_context import ParentContextService

        return ParentContextService.from_ctx(self._ctx).apply_parent_context(
            results,
            include_parent_text=include_parent_text,
        )

    @staticmethod
    def _build_document_filter(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        from knowledge_hub.ai.rag_parent_context import ParentContextService

        return ParentContextService.build_document_filter(metadata)

    @staticmethod
    def _doc_cache_key(filter_dict: Dict[str, Any]) -> str:
        from knowledge_hub.ai.rag_parent_context import ParentContextService

        return ParentContextService.doc_cache_key(filter_dict)

    @staticmethod
    def _parent_group_key(metadata: Dict[str, Any]) -> str:
        from knowledge_hub.ai.rag_parent_context import ParentContextService

        return ParentContextService.parent_group_key(metadata)

    @staticmethod
    def _parent_label(metadata: Dict[str, Any]) -> str:
        from knowledge_hub.ai.rag_parent_context import ParentContextService

        return ParentContextService.parent_label(metadata)

    def _load_document_chunks(
        self,
        metadata: Dict[str, Any],
        cache: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        from knowledge_hub.ai.rag_parent_context import ParentContextService

        return ParentContextService.from_ctx(self._ctx).load_document_chunks(metadata, cache)

    def _resolve_parent_context(
        self,
        result: SearchResult,
        cache: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        from knowledge_hub.ai.rag_parent_context import ParentContextService

        return ParentContextService.from_ctx(self._ctx).resolve_parent_context(result, cache)

    def _build_evidence_collaborator(self):
        from knowledge_hub.ai.evidence_collaborator import RAGEvidenceCollaborator

        return RAGEvidenceCollaborator(self)

    def search_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
        source_type: Optional[str] = None,
        retrieval_mode: str = "hybrid",
        alpha: float = 0.7,
        semantic_top_k: Optional[int] = None,
        lexical_top_k: Optional[int] = None,
        expand_parent_context: bool = False,
        use_ontology_expansion: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        from knowledge_hub.ai.rag_search_runtime import RAGSearchRuntime

        return RAGSearchRuntime(self).search_with_diagnostics(
            query=query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            semantic_top_k=semantic_top_k,
            lexical_top_k=lexical_top_k,
            expand_parent_context=expand_parent_context,
            use_ontology_expansion=use_ontology_expansion,
            metadata_filter=metadata_filter,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_type: Optional[str] = None,
        retrieval_mode: str = "hybrid",
        alpha: float = 0.7,
        semantic_top_k: Optional[int] = None,
        lexical_top_k: Optional[int] = None,
        expand_parent_context: bool = False,
        use_ontology_expansion: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        payload = self.search_with_diagnostics(
            query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            semantic_top_k=semantic_top_k,
            lexical_top_k=lexical_top_k,
            expand_parent_context=expand_parent_context,
            use_ontology_expansion=use_ontology_expansion,
            metadata_filter=metadata_filter,
        )
        return list(payload.get("results") or [])

    def _resolve_llm_for_request(
        self,
        query: str,
        context: str,
        source_count: int,
        allow_external: bool,
        force_route: str | None = None,
    ) -> tuple[Any, dict[str, Any], list[str]]:
        cached_local_llm, cached_local_llm_signature = self._caches.route_llm_cache()
        llm, decision, warnings, next_cached_local_llm, next_cached_local_llm_signature = resolve_llm_for_request(
            config=self.config,
            fixed_llm=self.llm,
            query=query,
            context=context,
            source_count=source_count,
            allow_external=allow_external,
            force_route=force_route,
            cached_local_llm=cached_local_llm,
            cached_local_llm_signature=cached_local_llm_signature,
            get_llm_for_hybrid_routing_fn=get_llm_for_hybrid_routing,
        )
        self._caches.writeback_route_llm_cache(
            self,
            cached_local_llm=next_cached_local_llm,
            cached_local_llm_signature=next_cached_local_llm_signature,
        )
        return llm, decision, warnings

    @staticmethod
    def _should_use_ask_v2(
        *,
        source_type: str | None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        sqlite_db: Any | None = None,
    ) -> bool:
        from knowledge_hub.ai.ask_v2 import AskV2Service

        return should_use_ask_v2(
            source_type=source_type,
            metadata_filter=metadata_filter,
            sqlite_db=sqlite_db,
            supports_fn=AskV2Service.supports,
            normalize_source_type_fn=_normalize_source_type,
        )

    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        source_type: Optional[str] = None,
        retrieval_mode: str = "hybrid",
        alpha: float = 0.7,
        allow_external: bool = False,
        memory_route_mode: str = MEMORY_ROUTE_MODE_OFF,
        paper_memory_mode: str = PAPER_MEMORY_MODE_OFF,
        metadata_filter: Optional[Dict[str, Any]] = None,
        ask_v2_mode: Optional[str] = None,
        answer_route_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """RAG 답변 생성: 공통 retrieval pipeline + evidence assembly + answer orchestration"""
        return generate_answer_runtime(
            self,
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
        )

    def stream_answer(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        source_type: Optional[str] = None,
        retrieval_mode: str = "hybrid",
        alpha: float = 0.7,
        allow_external: bool = False,
        memory_route_mode: str = MEMORY_ROUTE_MODE_OFF,
        paper_memory_mode: str = PAPER_MEMORY_MODE_OFF,
        metadata_filter: Optional[Dict[str, Any]] = None,
        ask_v2_mode: Optional[str] = None,
        answer_route_override: Optional[str] = None,
    ):
        """스트리밍 RAG 답변"""
        yield from stream_answer_runtime(
            self,
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
        )
