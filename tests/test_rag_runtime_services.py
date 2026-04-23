from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from knowledge_hub.ai.ask_v2 import AskV2FallbackToLegacy
from knowledge_hub.ai.rag import RAGSearcher
from knowledge_hub.ai.rag_answer_route_resolver import resolve_llm_for_request as resolve_llm_for_request_impl
from knowledge_hub.ai.rag_answer_runtime import generate_answer as generate_answer_runtime
from knowledge_hub.ai.rag_answer_runtime import stream_answer as stream_answer_runtime
from knowledge_hub.ai.rag_search_runtime import RAGSearchRuntime
from knowledge_hub.core.models import SearchResult
from tests.test_rag_search import DummyEmbedder, DummyVectorDB, FakeLLM


class _Config:
    def __init__(self, values: dict | None = None):
        self._values = values or {}

    def get_nested(self, *keys, default=None):  # noqa: ANN002, ANN003
        current = self._values
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current


def test_resolve_llm_for_request_reuses_cached_local_llm_only_for_matching_signature():
    class _Decision:
        def __init__(self, *, provider: str, model: str, timeout_sec: int):
            self.route = "local"
            self.provider = provider
            self.model = model
            self.timeout_sec = timeout_sec

        def to_dict(self):
            return {
                "route": self.route,
                "provider": self.provider,
                "model": self.model,
                "timeout_sec": self.timeout_sec,
            }

    first_candidate = object()
    second_candidate = object()
    third_candidate = object()
    calls = {"count": 0}

    def _router(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return first_candidate, _Decision(provider="ollama", model="qwen3:14b", timeout_sec=45), []
        if calls["count"] == 2:
            return second_candidate, _Decision(provider="ollama", model="qwen3:14b", timeout_sec=45), []
        return third_candidate, _Decision(provider="ollama", model="qwen3:32b", timeout_sec=45), []

    llm1, decision1, warnings1, cached1, signature1 = resolve_llm_for_request_impl(
        config=object(),
        fixed_llm=None,
        query="rag query",
        context="ctx",
        source_count=1,
        allow_external=False,
        cached_local_llm=None,
        cached_local_llm_signature=None,
        get_llm_for_hybrid_routing_fn=_router,
    )
    llm2, decision2, warnings2, cached2, signature2 = resolve_llm_for_request_impl(
        config=object(),
        fixed_llm=None,
        query="rag query",
        context="ctx",
        source_count=1,
        allow_external=False,
        cached_local_llm=cached1,
        cached_local_llm_signature=signature1,
        get_llm_for_hybrid_routing_fn=_router,
    )
    llm3, decision3, warnings3, cached3, signature3 = resolve_llm_for_request_impl(
        config=object(),
        fixed_llm=None,
        query="rag query",
        context="ctx",
        source_count=1,
        allow_external=False,
        cached_local_llm=cached2,
        cached_local_llm_signature=signature2,
        get_llm_for_hybrid_routing_fn=_router,
    )

    assert llm1 is first_candidate
    assert decision1["model"] == "qwen3:14b"
    assert warnings1 == []
    assert cached1 is first_candidate
    assert signature1 == ("ollama", "qwen3:14b", 45)

    assert llm2 is first_candidate
    assert decision2["model"] == "qwen3:14b"
    assert warnings2 == []
    assert cached2 is first_candidate
    assert signature2 == ("ollama", "qwen3:14b", 45)

    assert llm3 is third_candidate
    assert decision3["model"] == "qwen3:32b"
    assert warnings3 == []
    assert cached3 is third_candidate
    assert signature3 == ("ollama", "qwen3:32b", 45)


def test_resolve_llm_for_request_prefers_codex_backend_when_requested(monkeypatch):
    sentinel_llm = object()
    decision = {
        "route": "api",
        "provider": "codex_mcp",
        "model": "gpt-5.4-codex",
        "reasons": ["force_route=codex", "transport=exec"],
        "timeoutSec": 180,
        "fallbackUsed": False,
    }
    monkeypatch.setattr(
        "knowledge_hub.ai.rag_answer_route_resolver.resolve_preferred_codex_backend",
        lambda **kwargs: (sentinel_llm, decision, []),  # noqa: ARG005
    )

    llm, route_meta, warnings, cached_llm, cached_signature = resolve_llm_for_request_impl(
        config=_Config(),
        fixed_llm=None,
        query="rag query",
        context="ctx",
        source_count=1,
        allow_external=True,
        force_route="codex",
        cached_local_llm=None,
        cached_local_llm_signature=None,
        get_llm_for_hybrid_routing_fn=lambda *_args, **_kwargs: pytest.fail("hybrid router should not run"),
    )

    assert llm is sentinel_llm
    assert route_meta == decision
    assert warnings == []
    assert cached_llm is None
    assert cached_signature is None


def test_resolve_llm_for_request_keeps_codex_skip_warning_when_config_missing(monkeypatch):
    monkeypatch.setattr(
        "knowledge_hub.ai.rag_answer_route_resolver.resolve_preferred_codex_backend",
        lambda **kwargs: (None, None, ["codex_mcp backend unavailable: codex command not found: codex"]),  # noqa: ARG005
    )
    fixed_llm = FakeLLM()

    llm, route_meta, warnings, cached_llm, cached_signature = resolve_llm_for_request_impl(
        config=None,
        fixed_llm=fixed_llm,
        query="rag query",
        context="ctx",
        source_count=1,
        allow_external=True,
        cached_local_llm=None,
        cached_local_llm_signature=None,
        get_llm_for_hybrid_routing_fn=lambda *_args, **_kwargs: pytest.fail("hybrid router should not run"),
    )

    assert llm is fixed_llm
    assert route_meta["route"] == "fixed"
    assert route_meta["reasons"] == ["config_missing"]
    assert warnings == ["codex_mcp backend unavailable: codex command not found: codex"]
    assert cached_llm is None
    assert cached_signature is None


def test_rag_searcher_resolve_llm_for_request_syncs_route_cache_mirror_and_attrs():
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM())
    searcher._cached_local_llm = "stale-attr-cache"
    searcher._cached_local_llm_signature = ("stale", "attr", 1)
    searcher._caches.cached_local_llm = "authoritative-cache"
    searcher._caches.cached_local_llm_signature = ("provider", "model", 30)
    observed: dict[str, object] = {}

    def _direct_resolve(**kwargs):
        observed["cached_local_llm"] = kwargs["cached_local_llm"]
        observed["cached_local_llm_signature"] = kwargs["cached_local_llm_signature"]
        return FakeLLM(), {"route": "fixed"}, [], "next-cache", ("provider", "next", 45)

    with patch("knowledge_hub.ai.rag.resolve_llm_for_request", side_effect=_direct_resolve):
        llm, decision, warnings = searcher._resolve_llm_for_request(
            query="rag query",
            context="evidence context",
            source_count=1,
            allow_external=False,
        )

    assert isinstance(llm, FakeLLM)
    assert decision == {"route": "fixed"}
    assert warnings == []
    assert observed["cached_local_llm"] == "authoritative-cache"
    assert observed["cached_local_llm_signature"] == ("provider", "model", 30)
    assert searcher._cached_local_llm == "next-cache"
    assert searcher._cached_local_llm_signature == ("provider", "next", 45)
    assert searcher._caches.cached_local_llm == "next-cache"
    assert searcher._caches.cached_local_llm_signature == ("provider", "next", 45)


class _PrefixTitleSQLite:
    def search_papers(self, query, limit=20):
        _ = limit
        if str(query or "").strip().casefold() == "dino":
            return [{"arxiv_id": "dinov3-local", "title": "DINOv3"}]
        return []


def _ask_v2_pipeline_result(**overrides):
    diagnostics = overrides.pop("diagnostics", None)
    result = SimpleNamespace(
        results=overrides.pop("results", []),
        v2_diagnostics=overrides.pop("v2_diagnostics", {}),
        plan=overrides.pop("plan", None),
        **overrides,
    )
    result.diagnostics = diagnostics or (lambda: {})
    return result


def test_rag_search_runtime_applies_parent_context_when_requested():
    result = SearchResult(
        document="child chunk",
        metadata={"title": "RAG Child", "source_type": "vault", "file_path": "AI/rag.md"},
        distance=0.1,
        score=0.9,
        semantic_score=0.9,
        lexical_score=0.0,
        retrieval_mode="semantic",
        document_id="doc-1",
    )
    pipeline_result = SimpleNamespace(results=[result], diagnostics=lambda: {"status": "ok"})
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM())

    with patch(
        "knowledge_hub.ai.rag_search_runtime.RetrievalPipelineService.execute",
        return_value=pipeline_result,
    ), patch.object(
        searcher,
        "_apply_parent_context",
        return_value=[
            SearchResult(
                document="parent context",
                metadata=dict(result.metadata or {}),
                distance=result.distance,
                score=result.score,
                semantic_score=result.semantic_score,
                lexical_score=result.lexical_score,
                retrieval_mode=result.retrieval_mode,
                lexical_extras=dict(result.lexical_extras or {}),
                document_id=result.document_id,
            )
        ],
    ) as apply_parent:
        payload = RAGSearchRuntime(searcher).search_with_diagnostics(
            "rag query",
            top_k=1,
            expand_parent_context=True,
        )

    apply_parent.assert_called_once()
    assert payload["diagnostics"] == {"status": "ok"}
    assert payload["results"][0].document == "parent context"


def test_default_runtime_marks_legacy_execution_before_generate():
    pipeline_result = _ask_v2_pipeline_result(
        results=[],
        paper_memory_prefilter={},
        v2_diagnostics={"preserved": True},
    )
    evidence_packet = SimpleNamespace()
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM())
    searcher._should_use_ask_v2 = lambda **_kwargs: False  # type: ignore[method-assign]

    with patch(
        "knowledge_hub.ai.retrieval_pipeline.RetrievalPipelineService.execute",
        return_value=pipeline_result,
    ), patch(
        "knowledge_hub.ai.evidence_assembly.EvidenceAssemblyService.assemble",
        return_value=evidence_packet,
    ), patch(
        "knowledge_hub.ai.answer_orchestrator.AnswerOrchestrator.generate",
        return_value={"status": "ok"},
    ) as generate:
        payload = generate_answer_runtime(
            searcher,
            query="rag query",
            top_k=3,
            source_type="vault",
        )

    assert pipeline_result.v2_diagnostics["preserved"] is True
    assert pipeline_result.v2_diagnostics["runtimeExecution"]["used"] == "legacy"
    generate.assert_called_once()
    assert payload == {"status": "ok"}


def test_default_runtime_stream_delegates_with_marked_execution():
    pipeline_result = _ask_v2_pipeline_result(
        results=[],
        paper_memory_prefilter={},
        v2_diagnostics={},
    )
    evidence_packet = SimpleNamespace()
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM())
    searcher._should_use_ask_v2 = lambda **_kwargs: False  # type: ignore[method-assign]

    with patch(
        "knowledge_hub.ai.retrieval_pipeline.RetrievalPipelineService.execute",
        return_value=pipeline_result,
    ), patch(
        "knowledge_hub.ai.evidence_assembly.EvidenceAssemblyService.assemble",
        return_value=evidence_packet,
    ), patch(
        "knowledge_hub.ai.answer_orchestrator.AnswerOrchestrator.stream",
        return_value=iter(["a", "b"]),
    ) as stream:
        chunks = list(
            stream_answer_runtime(
                searcher,
                query="rag query",
                top_k=3,
                source_type="vault",
            )
        )

    assert pipeline_result.v2_diagnostics["runtimeExecution"]["fallbackReason"] == "ask_v2_capability_missing"
    stream.assert_called_once()
    assert chunks == ["a", "b"]


def test_rag_answer_runtime_prefers_ask_v2_execution_for_paper_lookup_queries():
    pipeline_result = _ask_v2_pipeline_result(
        v2_diagnostics={
            "runtimeExecution": {
                "used": "ask_v2",
                "fallbackReason": "",
                "sectionDecision": "paper_card",
                "sectionBlockReason": "",
            }
        }
    )
    evidence_packet = SimpleNamespace()
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM(), sqlite_db=object())
    searcher._should_use_ask_v2 = lambda **_kwargs: True  # type: ignore[method-assign]
    observed: dict[str, object] = {}

    class _FakeAskV2Service:
        def __init__(self, bound_searcher):
            assert bound_searcher is searcher

        def execute(self, **kwargs):
            assert kwargs["ask_v2_mode"] == "claim_first"
            observed["query_plan"] = kwargs["query_plan"]
            observed["query_frame"] = kwargs["query_frame"]
            return pipeline_result, evidence_packet

    with patch(
        "knowledge_hub.ai.ask_v2.AskV2Service",
        _FakeAskV2Service,
    ), patch(
        "knowledge_hub.ai.answer_orchestrator.AnswerOrchestrator.generate",
        return_value={"status": "ok", "path": "ask_v2"},
    ) as orchestrator_generate, patch(
        "knowledge_hub.ai.rag_answer_runtime.emit_rag_decision_log",
    ) as emit_log, patch(
        "knowledge_hub.ai.retrieval_pipeline.RetrievalPipelineService.execute",
    ) as default_generate:
        payload = generate_answer_runtime(
            searcher,
            query="AlexNet 논문 요약해줘",
            source_type="paper",
            ask_v2_mode="claim_first",
        )

    orchestrator_generate.assert_called_once()
    emit_log.assert_called_once()
    default_generate.assert_not_called()
    assert observed["query_plan"]["family"] == "paper_lookup"
    assert observed["query_frame"].family == "paper_lookup"
    assert isinstance(observed["query_frame"].resolved_source_ids, tuple)
    assert emit_log.call_args.kwargs["runtime_execution"]["used"] == "ask_v2"
    assert payload == {"status": "ok", "path": "ask_v2"}


def test_rag_answer_runtime_prefers_ask_v2_execution_for_paper_discover_queries():
    pipeline_result = _ask_v2_pipeline_result()
    evidence_packet = SimpleNamespace()
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM(), sqlite_db=object())
    searcher._should_use_ask_v2 = lambda **_kwargs: True  # type: ignore[method-assign]
    observed: dict[str, object] = {}

    class _FakeAskV2Service:
        def __init__(self, bound_searcher):
            assert bound_searcher is searcher

        def execute(self, **kwargs):
            observed["query_plan"] = kwargs["query_plan"]
            observed["query_frame"] = kwargs["query_frame"]
            return pipeline_result, evidence_packet

    with patch(
        "knowledge_hub.ai.ask_v2.AskV2Service",
        _FakeAskV2Service,
    ), patch(
        "knowledge_hub.ai.answer_orchestrator.AnswerOrchestrator.generate",
        return_value={"status": "ok", "path": "ask_v2"},
    ), patch(
        "knowledge_hub.ai.retrieval_pipeline.RetrievalPipelineService.execute",
    ) as default_generate:
        payload = generate_answer_runtime(
            searcher,
            query="RAG 관련 논문 찾아줘",
            source_type="paper",
        )

    default_generate.assert_not_called()
    assert observed["query_plan"]["family"] == "paper_discover"
    assert observed["query_frame"].family == "paper_discover"
    assert payload == {"status": "ok", "path": "ask_v2"}


def test_rag_answer_runtime_rejects_removed_legacy_mode():
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM(), sqlite_db=object())
    searcher._should_use_ask_v2 = lambda **_kwargs: True  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="ask_v2_mode='legacy' was removed"):
        generate_answer_runtime(
            searcher,
            query="rag query",
            source_type="paper",
            ask_v2_mode="legacy",
        )


def test_rag_answer_runtime_non_scoped_paper_definition_prefers_ask_v2():
    pipeline_result = _ask_v2_pipeline_result()
    evidence_packet = SimpleNamespace()
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM(), sqlite_db=object())
    searcher._should_use_ask_v2 = lambda **_kwargs: True  # type: ignore[method-assign]
    observed: dict[str, object] = {}

    class _FakeAskV2Service:
        def __init__(self, bound_searcher):
            assert bound_searcher is searcher

        def execute(self, **kwargs):
            observed["query_plan"] = kwargs["query_plan"]
            observed["query_frame"] = kwargs["query_frame"]
            return pipeline_result, evidence_packet

    with patch(
        "knowledge_hub.ai.ask_v2.AskV2Service",
        _FakeAskV2Service,
    ), patch(
        "knowledge_hub.ai.answer_orchestrator.AnswerOrchestrator.generate",
        return_value={"status": "ok", "path": "ask_v2"},
    ), patch(
        "knowledge_hub.ai.retrieval_pipeline.RetrievalPipelineService.execute",
    ) as default_generate:
        payload = generate_answer_runtime(
            searcher,
            query="Transformer의 핵심 아이디어를 설명해줘",
            source_type="paper",
        )

    default_generate.assert_not_called()
    assert observed["query_plan"]["family"] == "concept_explainer"
    assert observed["query_frame"].family == "concept_explainer"
    assert payload == {"status": "ok", "path": "ask_v2"}


def test_rag_answer_runtime_maps_paper_capability_gap_to_default_runtime_after_execution_fails():
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM(), sqlite_db=object())
    pipeline_result = _ask_v2_pipeline_result(results=[], paper_memory_prefilter={}, v2_diagnostics={})
    evidence_packet = SimpleNamespace()

    class _CapabilityGapAskV2Service:
        @staticmethod
        def supports(**_kwargs):
            return True

        def __init__(self, bound_searcher):
            assert bound_searcher is searcher

        def execute(self, **_kwargs):
            raise AttributeError("'object' object has no attribute 'search_paper_cards_v2'")

    with patch(
        "knowledge_hub.ai.ask_v2.AskV2Service",
        _CapabilityGapAskV2Service,
    ), patch(
        "knowledge_hub.ai.retrieval_pipeline.RetrievalPipelineService.execute",
        return_value=pipeline_result,
    ) as execute, patch(
        "knowledge_hub.ai.evidence_assembly.EvidenceAssemblyService.assemble",
        return_value=evidence_packet,
    ), patch(
        "knowledge_hub.ai.answer_orchestrator.AnswerOrchestrator.generate",
        return_value={"status": "ok", "path": "default_runtime"},
    ) as generate:
        payload = generate_answer_runtime(
            searcher,
            query="Transformer의 핵심 아이디어를 설명해줘",
            source_type="paper",
        )

    execute.assert_called_once()
    generate.assert_called_once()
    assert pipeline_result.v2_diagnostics["runtimeExecution"]["fallbackReason"] == "ask_v2_capability_missing"
    assert payload == {"status": "ok", "path": "default_runtime"}


def test_rag_answer_runtime_preserves_internal_ask_v2_fallback_reason_when_default_runtime_runs():
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM(), sqlite_db=object())
    searcher._should_use_ask_v2 = lambda **_kwargs: True  # type: ignore[method-assign]
    pipeline_result = _ask_v2_pipeline_result(results=[], paper_memory_prefilter={}, v2_diagnostics={})
    evidence_packet = SimpleNamespace()

    class _FallbackAskV2Service:
        def __init__(self, bound_searcher):
            assert bound_searcher is searcher

        def execute(self, **_kwargs):
            raise AskV2FallbackToLegacy("no_paper_shortlist_candidates")

    with patch(
        "knowledge_hub.ai.ask_v2.AskV2Service",
        _FallbackAskV2Service,
    ), patch(
        "knowledge_hub.ai.retrieval_pipeline.RetrievalPipelineService.execute",
        return_value=pipeline_result,
    ), patch(
        "knowledge_hub.ai.evidence_assembly.EvidenceAssemblyService.assemble",
        return_value=evidence_packet,
    ), patch(
        "knowledge_hub.ai.answer_orchestrator.AnswerOrchestrator.generate",
        return_value={"status": "ok", "path": "default_runtime"},
    ) as generate:
        payload = generate_answer_runtime(
            searcher,
            query="Transformer 대안 아키텍처 논문들을 추천해줘",
            source_type="paper",
        )

    generate.assert_called_once()
    assert pipeline_result.v2_diagnostics["runtimeExecution"]["fallbackReason"] == "no_paper_shortlist_candidates"
    assert payload == {"status": "ok", "path": "default_runtime"}


def test_rag_answer_runtime_infers_single_paper_scoped_concept_queries_into_ask_v2():
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM(), sqlite_db=_PrefixTitleSQLite())
    searcher._should_use_ask_v2 = lambda **_kwargs: True  # type: ignore[method-assign]
    observed: dict[str, object] = {}

    class _FakeAskV2Service:
        @staticmethod
        def supports(**_kwargs):
            return True

        def __init__(self, *_args, **_kwargs):
            pass

        def execute(self, **kwargs):
            observed["query_plan"] = kwargs["query_plan"]
            observed["query_frame"] = kwargs["query_frame"]
            return (
                _ask_v2_pipeline_result(plan=SimpleNamespace()),
                SimpleNamespace(),
            )

    with patch(
        "knowledge_hub.ai.ask_v2.AskV2Service",
        _FakeAskV2Service,
    ), patch(
        "knowledge_hub.ai.answer_orchestrator.AnswerOrchestrator.generate",
        return_value={"status": "ok", "path": "ask_v2"},
    ) as generate, patch(
        "knowledge_hub.ai.retrieval_pipeline.RetrievalPipelineService.execute",
    ) as default_generate:
        payload = generate_answer_runtime(
            searcher,
            query="DINO에 대해서 설명해줘",
        )

    generate.assert_called_once()
    default_generate.assert_not_called()
    assert payload == {"status": "ok", "path": "ask_v2"}
    assert observed["query_plan"]["family"] == "concept_explainer"
    assert observed["query_plan"]["resolved_paper_ids"] == ["dinov3-local"]


def test_rag_answer_runtime_paper_discover_uses_ask_v2_after_promotion():
    pipeline_result = _ask_v2_pipeline_result()
    evidence_packet = SimpleNamespace()
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM(), sqlite_db=object())
    searcher._should_use_ask_v2 = lambda **_kwargs: True  # type: ignore[method-assign]
    observed: dict[str, object] = {}

    class _FakeAskV2Service:
        def __init__(self, bound_searcher):
            assert bound_searcher is searcher

        def execute(self, **kwargs):
            observed["query_plan"] = kwargs["query_plan"]
            observed["query_frame"] = kwargs["query_frame"]
            return pipeline_result, evidence_packet

    with patch(
        "knowledge_hub.ai.ask_v2.AskV2Service",
        _FakeAskV2Service,
    ), patch(
        "knowledge_hub.ai.answer_orchestrator.AnswerOrchestrator.generate",
        return_value={"status": "ok", "path": "ask_v2"},
    ), patch(
        "knowledge_hub.ai.retrieval_pipeline.RetrievalPipelineService.execute",
    ) as default_generate:
        payload = generate_answer_runtime(
            searcher,
            query="RAG 관련 논문 찾아줘",
            source_type="paper",
        )

    default_generate.assert_not_called()
    assert observed["query_plan"]["family"] == "paper_discover"
    assert observed["query_frame"].family == "paper_discover"
    assert payload == {"status": "ok", "path": "ask_v2"}


def test_rag_answer_runtime_retries_no_result_with_planner_fallback():
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM(), sqlite_db=object())

    class _PlannerLLM:
        def generate(self, prompt: str, context: str = ""):
            _ = (prompt, context)
            return (
                '{"family":"concept_explainer","entities":["CNN"],'
                '"expanded_terms":["convolutional neural network","AlexNet"],'
                '"resolved_paper_ids":[],"answer_mode":"representative_paper_explainer","confidence":0.93}'
            )

    searcher._resolve_llm_for_request = lambda **_kwargs: (  # type: ignore[method-assign]
        _PlannerLLM(),
        {"route": "fixed", "provider": "openai", "model": "gpt-5.4"},
        [],
    )

    class _CapabilityGapAskV2Service:
        @staticmethod
        def supports(**_kwargs):
            return True

        def __init__(self, bound_searcher):
            assert bound_searcher is searcher

        def execute(self, **_kwargs):
            raise AttributeError("'object' object has no attribute 'upsert_normalization_alias'")

    first_pipeline = _ask_v2_pipeline_result(results=[], paper_memory_prefilter={}, v2_diagnostics={})
    second_pipeline = _ask_v2_pipeline_result(results=[], paper_memory_prefilter={}, v2_diagnostics={})
    with patch(
        "knowledge_hub.ai.ask_v2.AskV2Service",
        _CapabilityGapAskV2Service,
    ), patch(
        "knowledge_hub.ai.retrieval_pipeline.RetrievalPipelineService.execute",
        side_effect=[first_pipeline, second_pipeline],
    ) as execute, patch(
        "knowledge_hub.ai.evidence_assembly.EvidenceAssemblyService.assemble",
        return_value=SimpleNamespace(),
    ), patch(
        "knowledge_hub.ai.answer_orchestrator.AnswerOrchestrator.generate",
        side_effect=[
            {"status": "no_result", "answer": "관련된 문서를 찾을 수 없습니다."},
            {"status": "ok", "path": "default_retry"},
        ],
    ) as generate:
        payload = generate_answer_runtime(
            searcher,
            query="CNN을 쉽게 설명해줘",
            source_type="paper",
            allow_external=True,
        )

    assert execute.call_count == 2
    assert generate.call_count == 2
    retry_kwargs = execute.call_args_list[1].kwargs
    assert retry_kwargs["query_plan"]["plannerUsed"] is True
    assert "AlexNet" in retry_kwargs["query_plan"]["expandedTerms"]
    assert second_pipeline.v2_diagnostics["runtimeExecution"]["fallbackReason"] == "ask_v2_capability_missing"
    assert payload == {"status": "ok", "path": "default_retry"}


def test_generate_answer_passes_ask_v2_mode_to_runtime(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM())
    captured: dict[str, object] = {}

    def _fake_runtime(_searcher, **kwargs):
        assert _searcher is searcher
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr("knowledge_hub.ai.rag.generate_answer_runtime", _fake_runtime)
    payload = searcher.generate_answer("rag query", ask_v2_mode="claim_first")

    assert payload == {"status": "ok"}
    assert captured["ask_v2_mode"] == "claim_first"


def test_stream_answer_passes_ask_v2_mode_to_runtime(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM())
    captured: dict[str, object] = {}

    def _fake_runtime(_searcher, **kwargs):
        assert _searcher is searcher
        captured.update(kwargs)
        yield "chunk"

    monkeypatch.setattr("knowledge_hub.ai.rag.stream_answer_runtime", _fake_runtime)
    chunks = list(searcher.stream_answer("rag query", ask_v2_mode="section_first"))

    assert chunks == ["chunk"]
    assert captured["ask_v2_mode"] == "section_first"


def test_should_use_ask_v2_treats_paper_gate_as_warn_only_but_keeps_web_strict():
    sqlite_db = SimpleNamespace()
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB([]), llm=FakeLLM(), sqlite_db=sqlite_db)

    with patch("knowledge_hub.ai.ask_v2.AskV2Service.supports", return_value=True):
        assert searcher._should_use_ask_v2(source_type="paper", sqlite_db=sqlite_db) is True
        assert searcher._should_use_ask_v2(
            source_type=None,
            metadata_filter={"paper_id": "2401.00001"},
            sqlite_db=sqlite_db,
        ) is True
        assert searcher._should_use_ask_v2(
            source_type=None,
            metadata_filter={"arxiv_id": "2401.00001"},
            sqlite_db=sqlite_db,
        ) is True
        assert searcher._should_use_ask_v2(
            source_type="web",
            metadata_filter={"url": "https://example.com"},
            sqlite_db=sqlite_db,
        ) is False
        assert searcher._should_use_ask_v2(
            source_type="web",
            metadata_filter={"paper_id": "2401.00001"},
            sqlite_db=sqlite_db,
        ) is False
        assert searcher._should_use_ask_v2(
            source_type="vault",
            metadata_filter={"arxiv_id": "2401.00001"},
            sqlite_db=sqlite_db,
        ) is False
