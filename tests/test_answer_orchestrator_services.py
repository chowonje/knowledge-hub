from __future__ import annotations

from types import SimpleNamespace

from knowledge_hub.ai.answer_orchestrator import AnswerOrchestrator
from knowledge_hub.ai.answer_orchestrator_runtime_flow import AnswerRuntimeFlow, AnswerRuntimeFlowDeps
from knowledge_hub.ai.rag import RAGSearcher
from knowledge_hub.core.models import SearchResult
from tests.test_rag_search import (
    DummyEmbedder,
    DummyFeatureSQLiteWithLogs,
    DummyVectorDB,
    FailingLLM,
    FakeLLM,
    StaticLLM,
    _build_records,
)


class _AllowedPolicy:
    allowed = True
    classification = "P1"
    warnings: list[str] = []

    @staticmethod
    def to_dict():
        return {"allowed": True, "classification": "P1", "warnings": []}


class _BlockedPolicy:
    allowed = False
    classification = "P0"
    warnings: list[str] = []

    @staticmethod
    def to_dict():
        return {"allowed": False, "classification": "P0", "warnings": []}


def _paper_result() -> SearchResult:
    return SearchResult(
        document="paper evidence",
        metadata={"title": "Paper A", "source_type": "paper"},
        distance=0.1,
        score=0.95,
        semantic_score=0.94,
        lexical_score=0.93,
        retrieval_mode="hybrid",
        lexical_extras={},
        document_id="paper-a",
    )


def _evidence_packet(*, answer_signals=None, context="context", evidence=None, filtered_results=None, evidence_packet_payload=None):
    return SimpleNamespace(
        filtered_results=list(filtered_results) if filtered_results is not None else [_paper_result()],
        evidence=evidence or [{"title": "Paper A", "excerpt": "attention is needed"}],
        answer_signals=answer_signals or {},
        contradicting_beliefs=[],
        citations=[{"target": "Paper A"}],
        context=context,
        evidence_packet=evidence_packet_payload or {},
    )


def _pipeline_result():
    return SimpleNamespace(v2_diagnostics={}, plan=None)


def test_answer_orchestrator_default_inputs_use_direct_builder_path_without_override(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    observed: dict[str, object] = {}

    def _direct_prompt(*, query, answer_signals):
        observed["prompt"] = (query, dict(answer_signals or {}))
        return "direct-prompt"

    def _direct_context(*, query, filtered, evidence, answer_signals, claim_context):
        observed["context"] = {
            "query": query,
            "filtered_titles": [item.metadata["title"] for item in filtered],
            "evidence_count": len(evidence),
            "answer_signals": dict(answer_signals or {}),
            "claim_context": claim_context,
        }
        return "direct-context"

    monkeypatch.setattr("knowledge_hub.ai.answer_orchestrator._build_answer_prompt_impl", _direct_prompt)
    monkeypatch.setattr("knowledge_hub.ai.answer_orchestrator._build_paper_definition_context_impl", _direct_context)

    answer_prompt, answer_context = orchestrator._default_answer_inputs(
        query="attention mechanism",
        evidence_packet=SimpleNamespace(
            answer_signals={"paper_definition_mode": True},
            filtered_results=[_paper_result()],
            evidence=[{"title": "Paper A"}],
            context="unused-context",
        ),
        claim_context="claim-context",
    )

    assert answer_prompt == "direct-prompt"
    assert answer_context == "direct-context"
    assert observed["prompt"] == ("attention mechanism", {"paper_definition_mode": True})
    assert observed["context"] == {
        "query": "attention mechanism",
        "filtered_titles": ["Paper A"],
        "evidence_count": 1,
        "answer_signals": {"paper_definition_mode": True},
        "claim_context": "claim-context",
    }


def test_answer_orchestrator_default_inputs_preserve_searcher_builder_overrides(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    searcher._build_answer_prompt = lambda **kwargs: f"override-prompt:{kwargs['query']}"  # type: ignore[method-assign]
    searcher._build_paper_definition_context = lambda **kwargs: f"override-context:{kwargs['claim_context']}"  # type: ignore[method-assign]

    monkeypatch.setattr(
        "knowledge_hub.ai.answer_orchestrator._build_answer_prompt_impl",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("direct prompt helper should not run")),
    )
    monkeypatch.setattr(
        "knowledge_hub.ai.answer_orchestrator._build_paper_definition_context_impl",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("direct context helper should not run")),
    )

    answer_prompt, answer_context = orchestrator._default_answer_inputs(
        query="attention mechanism",
        evidence_packet=SimpleNamespace(
            answer_signals={"paper_definition_mode": True},
            filtered_results=[_paper_result()],
            evidence=[{"title": "Paper A"}],
            context="unused-context",
        ),
        claim_context="claim-context",
    )

    assert answer_prompt == "override-prompt:attention mechanism"
    assert answer_context == "override-context:claim-context"


def test_answer_orchestrator_build_fallback_uses_direct_helper_without_override(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    observed: dict[str, object] = {}

    def _direct_fallback(*, query, error, stage, evidence, citations, routing_meta):
        observed.update(
            {
                "query": query,
                "error": error,
                "stage": stage,
                "evidence": evidence,
                "citations": citations,
                "routing_meta": routing_meta,
            }
        )
        return (
            "fallback answer",
            {"stage": stage},
            {"status": "verified"},
            {"finalAnswerSource": "generation_fallback"},
            ["fallback warning"],
        )

    monkeypatch.setattr("knowledge_hub.ai.answer_orchestrator._build_answer_generation_fallback_impl", _direct_fallback)

    result = orchestrator._build_answer_generation_fallback(
        query="attention mechanism",
        error=RuntimeError("boom"),
        stage="initial_answer",
        evidence=[{"title": "Paper A"}],
        citations=[{"target": "Paper A"}],
        routing_meta={"route": "fixed"},
    )

    assert result[0] == "fallback answer"
    assert result[1] == {"stage": "initial_answer"}
    assert result[2] == {"status": "verified"}
    assert result[3] == {"finalAnswerSource": "generation_fallback"}
    assert result[4] == ["fallback warning"]
    assert observed["query"] == "attention mechanism"
    assert observed["stage"] == "initial_answer"
    assert observed["routing_meta"] == {"route": "fixed"}


def test_answer_orchestrator_build_fallback_preserves_searcher_override(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    searcher._build_answer_generation_fallback = lambda **kwargs: (  # type: ignore[method-assign]
        "override fallback",
        {"stage": kwargs["stage"]},
        {"status": "override"},
        {"finalAnswerSource": "override"},
        ["override warning"],
    )
    monkeypatch.setattr(
        "knowledge_hub.ai.answer_orchestrator._build_answer_generation_fallback_impl",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("direct fallback helper should not run")),
    )

    result = orchestrator._build_answer_generation_fallback(
        query="attention mechanism",
        error=RuntimeError("boom"),
        stage="initial_stream_answer",
        evidence=[{"title": "Paper A"}],
        citations=[{"target": "Paper A"}],
        routing_meta={"route": "fixed"},
    )

    assert result == (
        "override fallback",
        {"stage": "initial_stream_answer"},
        {"status": "override"},
        {"finalAnswerSource": "override"},
        ["override warning"],
    )


def test_answer_orchestrator_record_answer_log_uses_direct_helper_without_override():
    sqlite_db = DummyFeatureSQLiteWithLogs({})
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM(), sqlite_db=sqlite_db)
    orchestrator = AnswerOrchestrator(searcher)

    orchestrator._record_answer_log(
        query="attention mechanism",
        payload={
            "status": "ok",
            "sources": [{"title": "Paper A"}],
            "evidence": [{"title": "Paper A"}],
            "answerVerification": {
                "status": "verified",
                "needsCaution": False,
                "supportedClaimCount": 1,
                "uncertainClaimCount": 0,
                "unsupportedClaimCount": 0,
                "conflictMentioned": True,
                "route": {"route": "strong"},
            },
            "answerRewrite": {
                "attempted": False,
                "applied": False,
                "finalAnswerSource": "original",
                "route": {"route": "fixed"},
            },
            "router": {"selected": {"route": "fixed"}},
        },
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
    )

    rows = sqlite_db.list_rag_answer_logs(limit=10, days=7)
    assert len(rows) == 1
    assert rows[0]["result_status"] == "ok"
    assert rows[0]["verification_status"] == "verified"
    assert rows[0]["final_answer_source"] == "original"


def test_answer_orchestrator_record_answer_log_preserves_searcher_override(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    observed: dict[str, object] = {}

    searcher._record_answer_log = lambda **kwargs: observed.update(kwargs)  # type: ignore[method-assign]
    monkeypatch.setattr(
        "knowledge_hub.ai.answer_orchestrator._record_answer_log_impl",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct log helper should not run")),
    )

    orchestrator._record_answer_log(
        query="attention mechanism",
        payload={"status": "ok"},
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
    )

    assert observed["query"] == "attention mechanism"
    assert observed["payload"] == {"status": "ok"}
    assert observed["source_type"] == "paper"
    assert observed["retrieval_mode"] == "hybrid"
    assert observed["allow_external"] is False


def test_answer_orchestrator_resolve_runtime_path_uses_direct_helper_without_override(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    llm = StaticLLM("direct-answer")
    observed: dict[str, object] = {}

    def _direct_resolve(**kwargs):
        observed.update(kwargs)
        return llm, {"route": "fixed", "provider": "", "model": ""}, [], "cached-llm", ("provider", "model", 10)

    monkeypatch.setattr("knowledge_hub.ai.answer_orchestrator._resolve_llm_for_request_impl", _direct_resolve)

    selected_llm, routing_meta, warnings = orchestrator._resolve_llm_for_request(
        query="attention mechanism",
        context="evidence context",
        source_count=1,
        allow_external=False,
        force_route=None,
    )

    assert selected_llm is llm
    assert routing_meta == {"route": "fixed", "provider": "", "model": ""}
    assert warnings == []
    assert observed["query"] == "attention mechanism"
    assert observed["context"] == "evidence context"
    assert searcher._cached_local_llm == "cached-llm"
    assert searcher._cached_local_llm_signature == ("provider", "model", 10)
    assert searcher._caches.cached_local_llm == "cached-llm"
    assert searcher._caches.cached_local_llm_signature == ("provider", "model", 10)


def test_answer_orchestrator_resolve_runtime_path_prefers_cache_mirror_over_stale_searcher_attrs(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    searcher._cached_local_llm = "stale-attr-cache"
    searcher._cached_local_llm_signature = ("stale", "attr", 1)
    searcher._caches.cached_local_llm = "authoritative-cache"
    searcher._caches.cached_local_llm_signature = ("provider", "model", 30)
    observed: dict[str, object] = {}

    def _direct_resolve(**kwargs):
        observed["cached_local_llm"] = kwargs["cached_local_llm"]
        observed["cached_local_llm_signature"] = kwargs["cached_local_llm_signature"]
        return StaticLLM("direct-answer"), {"route": "fixed"}, [], "next-cache", ("provider", "next", 45)

    monkeypatch.setattr("knowledge_hub.ai.answer_orchestrator._resolve_llm_for_request_impl", _direct_resolve)

    orchestrator._resolve_llm_for_request(
        query="attention mechanism",
        context="evidence context",
        source_count=1,
        allow_external=False,
        force_route=None,
    )

    assert observed["cached_local_llm"] == "authoritative-cache"
    assert observed["cached_local_llm_signature"] == ("provider", "model", 30)
    assert searcher._cached_local_llm == "next-cache"
    assert searcher._cached_local_llm_signature == ("provider", "next", 45)
    assert searcher._caches.cached_local_llm == "next-cache"
    assert searcher._caches.cached_local_llm_signature == ("provider", "next", 45)


def test_answer_orchestrator_resolve_runtime_path_reuses_cached_local_llm_on_matching_signature(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    first_llm = StaticLLM("first")
    second_llm = StaticLLM("second")
    calls = {"count": 0}

    def _direct_resolve(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return first_llm, {"route": "local", "provider": "ollama", "model": "qwen3:14b"}, [], first_llm, (
                "ollama",
                "qwen3:14b",
                45,
            )
        assert kwargs["cached_local_llm"] is first_llm
        assert kwargs["cached_local_llm_signature"] == ("ollama", "qwen3:14b", 45)
        return second_llm, {"route": "local", "provider": "ollama", "model": "qwen3:14b"}, [], first_llm, (
            "ollama",
            "qwen3:14b",
            45,
        )

    monkeypatch.setattr("knowledge_hub.ai.answer_orchestrator._resolve_llm_for_request_impl", _direct_resolve)

    llm1, _, _ = orchestrator._resolve_llm_for_request(
        query="attention mechanism",
        context="evidence context",
        source_count=1,
        allow_external=False,
        force_route=None,
    )
    llm2, _, _ = orchestrator._resolve_llm_for_request(
        query="attention mechanism",
        context="evidence context",
        source_count=1,
        allow_external=False,
        force_route=None,
    )

    assert llm1 is first_llm
    assert llm2 is second_llm
    assert searcher._caches.cached_local_llm is first_llm
    assert searcher._cached_local_llm is first_llm


def test_answer_orchestrator_resolve_runtime_path_preserves_searcher_override(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    expected = (StaticLLM("override-answer"), {"route": "override"}, ["warn"])
    searcher._resolve_llm_for_request = lambda **kwargs: expected  # type: ignore[method-assign]
    monkeypatch.setattr(
        "knowledge_hub.ai.answer_orchestrator._resolve_llm_for_request_impl",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("direct resolve helper should not run")),
    )

    actual = orchestrator._resolve_llm_for_request(
        query="attention mechanism",
        context="evidence context",
        source_count=1,
        allow_external=False,
        force_route=None,
    )

    assert actual == expected


def test_answer_orchestrator_verify_runtime_path_uses_direct_helper_without_override(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    observed: dict[str, object] = {}

    def _direct_verify(bound_searcher, **kwargs):
        observed["searcher"] = bound_searcher
        observed.update(kwargs)
        return {"status": "verified", "supportedClaimCount": 1, "unsupportedClaimCount": 0, "uncertainClaimCount": 0}

    monkeypatch.setattr("knowledge_hub.ai.answer_orchestrator._verify_answer_impl", _direct_verify)

    verification = orchestrator._verify_answer(
        query="attention mechanism",
        answer="answer",
        evidence=[{"title": "Paper A"}],
        answer_signals={},
        contradicting_beliefs=[],
        allow_external=False,
    )

    assert verification["status"] == "verified"
    assert observed["searcher"] is searcher
    assert observed["answer"] == "answer"


def test_answer_orchestrator_verify_runtime_path_preserves_searcher_override(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    searcher._verify_answer = lambda **kwargs: {"status": "override", "answer": kwargs["answer"]}  # type: ignore[method-assign]
    monkeypatch.setattr(
        "knowledge_hub.ai.answer_orchestrator._verify_answer_impl",
        lambda *args, **_kwargs: (_ for _ in ()).throw(AssertionError("direct verify helper should not run")),
    )

    verification = orchestrator._verify_answer(
        query="attention mechanism",
        answer="answer",
        evidence=[{"title": "Paper A"}],
        answer_signals={},
        contradicting_beliefs=[],
        allow_external=False,
    )

    assert verification == {"status": "override", "answer": "answer"}


def test_answer_orchestrator_rewrite_runtime_path_uses_direct_helper_without_override(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    observed: dict[str, object] = {}

    def _direct_rewrite(bound_searcher, **kwargs):
        observed["searcher"] = bound_searcher
        observed.update(kwargs)
        return "rewritten", {"attempted": True, "applied": True, "finalAnswerSource": "rewritten"}

    monkeypatch.setattr("knowledge_hub.ai.answer_orchestrator._rewrite_answer_impl", _direct_rewrite)

    rewritten, meta = orchestrator._rewrite_answer(
        query="attention mechanism",
        answer="answer",
        evidence=[{"title": "Paper A"}],
        answer_signals={},
        verification={"needsCaution": True},
        contradicting_beliefs=[],
        allow_external=False,
    )

    assert rewritten == "rewritten"
    assert meta["finalAnswerSource"] == "rewritten"
    assert observed["searcher"] is searcher


def test_answer_orchestrator_rewrite_runtime_path_preserves_searcher_override(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    searcher._rewrite_answer = lambda **kwargs: ("override-rewrite", {"finalAnswerSource": kwargs["answer"]})  # type: ignore[method-assign]
    monkeypatch.setattr(
        "knowledge_hub.ai.answer_orchestrator._rewrite_answer_impl",
        lambda *args, **_kwargs: (_ for _ in ()).throw(AssertionError("direct rewrite helper should not run")),
    )

    rewritten, meta = orchestrator._rewrite_answer(
        query="attention mechanism",
        answer="answer",
        evidence=[{"title": "Paper A"}],
        answer_signals={},
        verification={"needsCaution": True},
        contradicting_beliefs=[],
        allow_external=False,
    )

    assert rewritten == "override-rewrite"
    assert meta == {"finalAnswerSource": "answer"}


def test_answer_orchestrator_fallback_runtime_path_uses_direct_helper_without_override(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    observed: dict[str, object] = {}

    def _direct_fallback(bound_searcher, **kwargs):
        observed["searcher"] = bound_searcher
        observed.update(kwargs)
        return "fallback", {"finalAnswerSource": "conservative_fallback"}, {"status": "verified"}

    monkeypatch.setattr("knowledge_hub.ai.answer_orchestrator._apply_conservative_fallback_if_needed_impl", _direct_fallback)

    answer, rewrite_meta, verification = orchestrator._apply_conservative_fallback_if_needed(
        query="attention mechanism",
        answer="answer",
        rewrite_meta={"applied": True},
        verification={"needsCaution": True},
        evidence=[{"title": "Paper A"}],
        answer_signals={},
        contradicting_beliefs=[],
        allow_external=False,
    )

    assert answer == "fallback"
    assert rewrite_meta["finalAnswerSource"] == "conservative_fallback"
    assert verification["status"] == "verified"
    assert observed["searcher"] is searcher


def test_answer_orchestrator_fallback_runtime_path_preserves_searcher_override(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    searcher._apply_conservative_fallback_if_needed = lambda **kwargs: ("override-fallback", {"source": "override"}, {"status": kwargs["verification"]["status"]})  # type: ignore[method-assign]
    monkeypatch.setattr(
        "knowledge_hub.ai.answer_orchestrator._apply_conservative_fallback_if_needed_impl",
        lambda *args, **_kwargs: (_ for _ in ()).throw(AssertionError("direct fallback helper should not run")),
    )

    answer, rewrite_meta, verification = orchestrator._apply_conservative_fallback_if_needed(
        query="attention mechanism",
        answer="answer",
        rewrite_meta={"applied": True},
        verification={"status": "caution"},
        evidence=[{"title": "Paper A"}],
        answer_signals={},
        contradicting_beliefs=[],
        allow_external=False,
    )

    assert answer == "override-fallback"
    assert rewrite_meta == {"source": "override"}
    assert verification == {"status": "caution"}


def test_answer_orchestrator_prepare_answer_execution_inputs_prefers_section_native(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)

    monkeypatch.setattr(orchestrator, "_section_native_inputs", lambda **_kwargs: ("section-prompt", "section-context", {"covered": True}))
    monkeypatch.setattr(
        orchestrator,
        "_claim_native_inputs",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("claim-native path should not run")),
    )
    monkeypatch.setattr(
        orchestrator,
        "_default_answer_inputs",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("default input path should not run")),
    )
    monkeypatch.setattr(
        orchestrator,
        "_evaluate_policy",
        lambda **kwargs: (f"safe::{kwargs['context']}", _AllowedPolicy(), "P1"),
    )

    prepared = orchestrator._prepare_answer_execution_inputs(
        query="attention mechanism",
        pipeline_result=_pipeline_result(),
        evidence_packet=_evidence_packet(),
        selected_llm=StaticLLM("unused"),
        claim_verification=[{"claim": "base"}],
        claim_consensus={"status": "base"},
        claim_context="claim-context",
        allow_external=False,
        route_mode="fixed",
    )

    assert prepared.answer_prompt == "section-prompt"
    assert prepared.safe_context == "safe::section-context"
    assert prepared.claim_verification == [{"claim": "base"}]
    assert prepared.claim_consensus == {"status": "base"}
    assert prepared.claim_consensus_merge_mode == "advisory"
    assert prepared.original_classification == "P1"


def test_answer_orchestrator_prepare_answer_execution_inputs_updates_claim_native_bundle(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    pipeline_result = SimpleNamespace(
        v2_diagnostics={"answerProvenance": {"mode": "claim_native"}},
        plan=SimpleNamespace(to_dict=lambda: {"queryFrame": {"family": "paper_compare"}}),
    )

    monkeypatch.setattr(orchestrator, "_section_native_inputs", lambda **_kwargs: None)
    monkeypatch.setattr(
        orchestrator,
        "_claim_native_inputs",
        lambda **_kwargs: (
            "claim-prompt",
            "claim-context",
            [{"claim": "native"}],
            {"status": "native"},
            [],
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_default_answer_inputs",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("default input path should not run")),
    )
    monkeypatch.setattr(
        orchestrator,
        "_evaluate_policy",
        lambda **kwargs: (f"safe::{kwargs['context']}", _AllowedPolicy(), "P1"),
    )

    prepared = orchestrator._prepare_answer_execution_inputs(
        query="attention mechanism",
        pipeline_result=pipeline_result,
        evidence_packet=_evidence_packet(),
        selected_llm=StaticLLM("unused"),
        claim_verification=[{"claim": "base"}],
        claim_consensus={"status": "base"},
        claim_context="claim-context",
        allow_external=False,
        route_mode="fixed",
    )

    assert prepared.answer_prompt == "claim-prompt"
    assert prepared.safe_context == "safe::claim-context"
    assert prepared.claim_verification == [{"claim": "native"}]
    assert prepared.claim_consensus == {"status": "native"}
    assert prepared.claim_consensus_merge_mode == "strict"
    assert prepared.original_classification == "P1"


def test_answer_orchestrator_generate_and_stream_no_result_early_exit_stay_in_parity(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    evidence_packet = _evidence_packet(filtered_results=[])
    logged_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(
        orchestrator,
        "_resolve_llm_for_request",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("early-exit path must bypass llm resolution")),
    )
    monkeypatch.setattr(
        orchestrator,
        "_adjudicate_claims",
        lambda **_kwargs: ([{"claim": "missing"}], {"status": "empty"}, ""),
    )
    monkeypatch.setattr(
        orchestrator,
        "_base_payload",
        lambda **kwargs: {"answer": kwargs["answer"], "status": kwargs.get("status", "ok")},
    )

    def _record(**kwargs):
        payload = dict(kwargs["payload"])
        assert payload["status"] == "no_result"
        assert "claimVerification" in payload
        assert "claimConsensus" in payload
        logged_payloads.append(payload)

    monkeypatch.setattr(orchestrator, "_record_answer_log", _record)

    generated = orchestrator.generate(
        query="attention mechanism",
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
        pipeline_result=_pipeline_result(),
        evidence_packet=evidence_packet,
    )
    streamed = "".join(
        orchestrator.stream(
            query="attention mechanism",
            source_type="paper",
            retrieval_mode="hybrid",
            allow_external=False,
            pipeline_result=_pipeline_result(),
            evidence_packet=evidence_packet,
        )
    )

    assert generated["answer"] == "관련된 문서를 찾을 수 없습니다."
    assert generated["status"] == "no_result"
    assert generated["claimVerification"] == [{"claim": "missing"}]
    assert generated["claimConsensus"] == {"status": "empty"}
    assert streamed == generated["answer"]
    assert logged_payloads == [generated, generated]


def test_answer_orchestrator_generate_and_stream_need_multiple_papers_early_exit_stay_in_parity(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    evidence_packet = _evidence_packet(
        evidence_packet_payload={"answerableDecisionReason": "need_multiple_papers", "uniquePaperCount": 1}
    )
    logged_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(
        orchestrator,
        "_resolve_llm_for_request",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("early-exit path must bypass llm resolution")),
    )
    monkeypatch.setattr(
        orchestrator,
        "_adjudicate_claims",
        lambda **_kwargs: ([{"claim": "compare"}], {"status": "need_more"}, ""),
    )
    monkeypatch.setattr(
        orchestrator,
        "_base_payload",
        lambda **kwargs: {"answer": kwargs["answer"], "status": kwargs.get("status", "ok")},
    )

    def _record(**kwargs):
        payload = dict(kwargs["payload"])
        assert payload["status"] == "no_result"
        assert "claimVerification" in payload
        assert "claimConsensus" in payload
        logged_payloads.append(payload)

    monkeypatch.setattr(orchestrator, "_record_answer_log", _record)

    generated = orchestrator.generate(
        query="attention mechanism",
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
        pipeline_result=_pipeline_result(),
        evidence_packet=evidence_packet,
    )
    streamed = "".join(
        orchestrator.stream(
            query="attention mechanism",
            source_type="paper",
            retrieval_mode="hybrid",
            allow_external=False,
            pipeline_result=_pipeline_result(),
            evidence_packet=evidence_packet,
        )
    )

    assert generated["answer"] == "비교 가능한 논문 2편 이상을 찾지 못했습니다."
    assert generated["status"] == "no_result"
    assert generated["claimVerification"] == [{"claim": "compare"}]
    assert generated["claimConsensus"] == {"status": "need_more"}
    assert streamed == generated["answer"]
    assert logged_payloads == [generated, generated]


def test_answer_orchestrator_generate_and_stream_plain_answer_stay_in_parity(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    llm = StaticLLM("plain answer")
    evidence_packet = _evidence_packet()

    monkeypatch.setattr(orchestrator, "_resolve_llm_for_request", lambda **_kwargs: (llm, {"route": "fixed"}, []))
    monkeypatch.setattr(orchestrator, "_verify_answer", lambda **_kwargs: {"status": "verified", "supportedClaimCount": 1, "unsupportedClaimCount": 0, "uncertainClaimCount": 0, "conflictMentioned": True, "needsCaution": False, "warnings": []})
    monkeypatch.setattr(orchestrator, "_rewrite_answer", lambda **kwargs: (kwargs["answer"], {"attempted": False, "applied": False, "finalAnswerSource": "original", "warnings": []}))
    monkeypatch.setattr(orchestrator, "_apply_conservative_fallback_if_needed", lambda **kwargs: (kwargs["answer"], kwargs["rewrite_meta"], kwargs["verification"]))
    monkeypatch.setattr(orchestrator, "_record_answer_log", lambda **_kwargs: None)
    monkeypatch.setattr(orchestrator, "_default_answer_inputs", lambda **_kwargs: ("prompt", "context"))
    monkeypatch.setattr(orchestrator, "_adjudicate_claims", lambda **_kwargs: ([], {}, ""))
    monkeypatch.setattr(orchestrator, "_evaluate_policy", lambda **_kwargs: ("context", _AllowedPolicy(), "P1"))
    monkeypatch.setattr(orchestrator, "_base_payload", lambda **kwargs: {"answer": kwargs["answer"], "status": kwargs.get("status", "ok")})

    generated = orchestrator.generate(
        query="attention mechanism",
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
        pipeline_result=_pipeline_result(),
        evidence_packet=evidence_packet,
    )["answer"]
    streamed = "".join(
        orchestrator.stream(
            query="attention mechanism",
            source_type="paper",
            retrieval_mode="hybrid",
            allow_external=False,
            pipeline_result=_pipeline_result(),
            evidence_packet=evidence_packet,
        )
    )

    assert generated == streamed == "plain answer"


def test_answer_orchestrator_generate_and_stream_rewritten_answer_stay_in_parity(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    llm = StaticLLM("initial answer")
    evidence_packet = _evidence_packet()

    monkeypatch.setattr(orchestrator, "_resolve_llm_for_request", lambda **_kwargs: (llm, {"route": "fixed"}, []))
    monkeypatch.setattr(orchestrator, "_verify_answer", lambda **_kwargs: {"status": "verified", "supportedClaimCount": 1, "unsupportedClaimCount": 0, "uncertainClaimCount": 0, "conflictMentioned": True, "needsCaution": False, "warnings": []})
    monkeypatch.setattr(orchestrator, "_rewrite_answer", lambda **_kwargs: ("rewritten answer", {"attempted": True, "applied": True, "finalAnswerSource": "rewritten", "warnings": []}))
    monkeypatch.setattr(orchestrator, "_apply_conservative_fallback_if_needed", lambda **kwargs: (kwargs["answer"], kwargs["rewrite_meta"], kwargs["verification"]))
    monkeypatch.setattr(orchestrator, "_record_answer_log", lambda **_kwargs: None)
    monkeypatch.setattr(orchestrator, "_default_answer_inputs", lambda **_kwargs: ("prompt", "context"))
    monkeypatch.setattr(orchestrator, "_adjudicate_claims", lambda **_kwargs: ([], {}, ""))
    monkeypatch.setattr(orchestrator, "_evaluate_policy", lambda **_kwargs: ("context", _AllowedPolicy(), "P1"))
    monkeypatch.setattr(orchestrator, "_base_payload", lambda **kwargs: {"answer": kwargs["answer"], "status": kwargs.get("status", "ok")})

    generated = orchestrator.generate(
        query="attention mechanism",
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
        pipeline_result=_pipeline_result(),
        evidence_packet=evidence_packet,
    )["answer"]
    streamed = "".join(
        orchestrator.stream(
            query="attention mechanism",
            source_type="paper",
            retrieval_mode="hybrid",
            allow_external=False,
            pipeline_result=_pipeline_result(),
            evidence_packet=evidence_packet,
        )
    )

    assert generated == streamed == "rewritten answer"


def test_answer_orchestrator_generate_and_stream_conservative_fallback_stay_in_parity(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    llm = StaticLLM("initial answer")
    evidence_packet = _evidence_packet()

    monkeypatch.setattr(orchestrator, "_resolve_llm_for_request", lambda **_kwargs: (llm, {"route": "fixed"}, []))
    monkeypatch.setattr(orchestrator, "_verify_answer", lambda **_kwargs: {"status": "caution", "supportedClaimCount": 0, "unsupportedClaimCount": 1, "uncertainClaimCount": 0, "conflictMentioned": True, "needsCaution": True, "warnings": []})
    monkeypatch.setattr(orchestrator, "_rewrite_answer", lambda **_kwargs: ("rewritten answer", {"attempted": True, "applied": True, "finalAnswerSource": "rewritten", "warnings": []}))
    monkeypatch.setattr(orchestrator, "_apply_conservative_fallback_if_needed", lambda **_kwargs: ("fallback answer", {"attempted": True, "applied": True, "finalAnswerSource": "conservative_fallback", "warnings": []}, {"status": "verified", "supportedClaimCount": 1, "unsupportedClaimCount": 0, "uncertainClaimCount": 0, "conflictMentioned": True, "needsCaution": False, "warnings": []}))
    monkeypatch.setattr(orchestrator, "_record_answer_log", lambda **_kwargs: None)
    monkeypatch.setattr(orchestrator, "_default_answer_inputs", lambda **_kwargs: ("prompt", "context"))
    monkeypatch.setattr(orchestrator, "_adjudicate_claims", lambda **_kwargs: ([], {}, ""))
    monkeypatch.setattr(orchestrator, "_evaluate_policy", lambda **_kwargs: ("context", _AllowedPolicy(), "P1"))
    monkeypatch.setattr(orchestrator, "_base_payload", lambda **kwargs: {"answer": kwargs["answer"], "status": kwargs.get("status", "ok")})

    generated = orchestrator.generate(
        query="attention mechanism",
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
        pipeline_result=_pipeline_result(),
        evidence_packet=evidence_packet,
    )["answer"]
    streamed = "".join(
        orchestrator.stream(
            query="attention mechanism",
            source_type="paper",
            retrieval_mode="hybrid",
            allow_external=False,
            pipeline_result=_pipeline_result(),
            evidence_packet=evidence_packet,
        )
    )

    assert generated == streamed == "fallback answer"


def test_answer_orchestrator_generate_and_stream_blocked_path_stay_in_answer_parity(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    evidence_packet = _evidence_packet()
    logged_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(
        orchestrator,
        "_resolve_llm_for_request",
        lambda **_kwargs: (StaticLLM("unused"), {"route": "fixed"}, ["route warning"]),
    )
    monkeypatch.setattr(
        orchestrator,
        "_prepare_answer_execution_inputs",
        lambda **_kwargs: SimpleNamespace(
            claim_verification=[{"claim": "blocked"}],
            claim_consensus={"status": "blocked"},
            claim_consensus_merge_mode="advisory",
            answer_prompt="prompt",
            safe_context="safe-context",
            external_policy=_BlockedPolicy(),
            original_classification="P0",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_verify_answer",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("blocked path must bypass verification")),
    )
    monkeypatch.setattr(
        orchestrator,
        "_rewrite_answer",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("blocked path must bypass rewrite")),
    )
    monkeypatch.setattr(
        orchestrator,
        "_apply_conservative_fallback_if_needed",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("blocked path must bypass fallback")),
    )
    monkeypatch.setattr(
        orchestrator,
        "_base_payload",
        lambda **kwargs: {"answer": kwargs["answer"], "status": kwargs.get("status", "ok")},
    )

    def _record(**kwargs):
        logged_payloads.append(dict(kwargs["payload"]))

    monkeypatch.setattr(orchestrator, "_record_answer_log", _record)

    generated = orchestrator.generate(
        query="contact info",
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
        pipeline_result=_pipeline_result(),
        evidence_packet=evidence_packet,
    )
    streamed = "".join(
        orchestrator.stream(
            query="contact info",
            source_type="paper",
            retrieval_mode="hybrid",
            allow_external=False,
            pipeline_result=_pipeline_result(),
            evidence_packet=evidence_packet,
        )
    )

    assert generated["answer"] == "정책상 민감 정보(P0)가 포함되어 외부 모델 호출을 차단했습니다."
    assert generated["status"] == "blocked"
    assert generated["claimVerification"] == [{"claim": "blocked"}]
    assert generated["claimConsensus"] == {"status": "blocked"}
    assert generated["policy"]["originalClassification"] == "P0"
    assert generated["router"] == {"selected": {"route": "fixed"}, "warnings": ["route warning"]}
    assert streamed == generated["answer"]
    assert logged_payloads[0]["router"] == {"selected": {"route": "fixed"}, "warnings": ["route warning"]}
    assert logged_payloads[1]["router"] == {"selected": {"route": "fixed"}}


def test_answer_orchestrator_generate_and_stream_success_payload_shape_stays_distinct(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    evidence_packet = _evidence_packet()
    logged_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(
        orchestrator,
        "_resolve_llm_for_request",
        lambda **_kwargs: (StaticLLM("initial answer"), {"route": "fixed"}, ["route warning"]),
    )
    monkeypatch.setattr(
        orchestrator,
        "_prepare_answer_execution_inputs",
        lambda **_kwargs: SimpleNamespace(
            claim_verification=[{"claim": "supported"}],
            claim_consensus={"status": "strict"},
            claim_consensus_merge_mode="strict",
            answer_prompt="prompt",
            safe_context="safe-context",
            external_policy=_AllowedPolicy(),
            original_classification="P1",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_postprocess_result",
        lambda **kwargs: SimpleNamespace(
            final_answer="rewritten answer",
            final_answer_verification={
                "status": "verified",
                "supportedClaimCount": 1,
                "unsupportedClaimCount": 0,
                "uncertainClaimCount": 0,
                "conflictMentioned": False,
                "needsCaution": False,
                "warnings": ["final warning"],
            },
            final_answer_rewrite={
                "attempted": True,
                "applied": True,
                "finalAnswerSource": "rewritten",
                "warnings": [],
            },
            initial_answer_verification={
                "status": "caution",
                "supportedClaimCount": 0,
                "unsupportedClaimCount": 0,
                "uncertainClaimCount": 1,
                "conflictMentioned": False,
                "needsCaution": True,
                "warnings": ["initial warning"],
            },
            verification_warnings=["final warning"],
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_base_payload",
        lambda **kwargs: {"answer": kwargs["answer"], "status": kwargs.get("status", "ok")},
    )

    def _record(**kwargs):
        logged_payloads.append(dict(kwargs["payload"]))

    monkeypatch.setattr(orchestrator, "_record_answer_log", _record)

    generated = orchestrator.generate(
        query="attention mechanism",
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
        pipeline_result=_pipeline_result(),
        evidence_packet=evidence_packet,
    )
    streamed = "".join(
        orchestrator.stream(
            query="attention mechanism",
            source_type="paper",
            retrieval_mode="hybrid",
            allow_external=False,
            pipeline_result=_pipeline_result(),
            evidence_packet=evidence_packet,
        )
    )

    assert generated["answer"] == "rewritten answer"
    assert streamed == generated["answer"]
    assert generated["router"] == {"selected": {"route": "fixed"}, "warnings": ["route warning"]}
    assert generated["initialAnswerVerification"]["status"] == "caution"
    assert generated["warnings"] == ["final warning"]
    assert logged_payloads[0]["router"] == {"selected": {"route": "fixed"}, "warnings": ["route warning"]}
    assert logged_payloads[0]["initialAnswerVerification"]["status"] == "caution"
    assert logged_payloads[0]["warnings"] == ["final warning"]
    assert logged_payloads[1]["router"] == {"selected": {"route": "fixed"}}
    assert "initialAnswerVerification" not in logged_payloads[1]
    assert "warnings" not in logged_payloads[1]


def test_answer_orchestrator_initial_generation_returns_no_route_fallback(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    evidence_packet = _evidence_packet()

    monkeypatch.setattr(
        orchestrator,
        "_build_answer_generation_fallback",
        lambda **kwargs: (
            f"fallback::{kwargs['stage']}",
            {"stage": kwargs["stage"]},
            {"status": "verified"},
            {"finalAnswerSource": "generation_fallback"},
            ["fallback warning"],
        ),
    )

    result = orchestrator._initial_generation_result(
        query="attention mechanism",
        selected_llm=None,
        answer_prompt="prompt",
        safe_context="context",
        evidence_packet=evidence_packet,
        routing_meta={"route": "fixed"},
        stage="initial_answer",
    )

    assert result.is_fallback is True
    assert result.fallback_kind == "no_route"
    assert result.fallback_answer == "fallback::initial_answer"
    assert result.generation_meta == {"stage": "initial_answer"}
    assert result.fallback_verification == {"status": "verified"}
    assert result.fallback_rewrite == {"finalAnswerSource": "generation_fallback"}
    assert result.generation_warnings == ["fallback warning"]
    assert result.initial_answer is None


def test_answer_orchestrator_initial_generation_stream_error_returns_fallback(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    evidence_packet = _evidence_packet()

    monkeypatch.setattr(
        orchestrator,
        "_build_answer_generation_fallback",
        lambda **kwargs: (
            f"fallback::{kwargs['stage']}",
            {"stage": kwargs["stage"]},
            {"status": "verified"},
            {"finalAnswerSource": "generation_fallback"},
            ["stream fallback warning"],
        ),
    )

    result = orchestrator._initial_generation_result(
        query="attention mechanism",
        selected_llm=FailingLLM(RuntimeError("stream boom")),
        answer_prompt="prompt",
        safe_context="context",
        evidence_packet=evidence_packet,
        routing_meta={"route": "fixed"},
        stage="initial_stream_answer",
        stream=True,
    )

    assert result.is_fallback is True
    assert result.fallback_kind == "generation_error"
    assert result.fallback_answer == "fallback::initial_stream_answer"
    assert result.generation_meta == {"stage": "initial_stream_answer"}
    assert result.generation_warnings == ["stream fallback warning"]
    assert result.initial_answer is None


def test_answer_runtime_flow_no_route_merges_warnings_for_generate_and_stream():
    observed: list[dict[str, object]] = []

    def _build_generation_fallback_payload(**kwargs):
        observed.append(
            {
                "router_warnings": kwargs["router_warnings"],
                "payload_warnings": kwargs["payload_warnings"],
            }
        )
        return {
            "answer": kwargs["answer"],
            "router_warnings": kwargs["router_warnings"],
            "payload_warnings": kwargs["payload_warnings"],
        }

    flow = AnswerRuntimeFlow(
        AnswerRuntimeFlowDeps(
            build_blocked_payload_fn=lambda **kwargs: kwargs,
            build_generation_fallback_payload_fn=_build_generation_fallback_payload,
            build_success_payload_fn=lambda **kwargs: kwargs,
            initial_generation_result_fn=lambda **_kwargs: SimpleNamespace(
                is_fallback=True,
                fallback_kind="no_route",
                fallback_answer="fallback",
                generation_meta={"stage": "initial"},
                fallback_verification={"status": "verified"},
                fallback_rewrite={"finalAnswerSource": "generation_fallback"},
                generation_warnings=["gen warning"],
            ),
            postprocess_result_fn=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("postprocess should not run")),
        )
    )

    base_kwargs = dict(
        query="attention mechanism",
        retrieval_mode="hybrid",
        pipeline_result=_pipeline_result(),
        evidence_packet=_evidence_packet(),
        selected_llm=None,
        claim_verification=[],
        claim_consensus={},
        claim_consensus_merge_mode="advisory",
        answer_prompt="prompt",
        safe_context="context",
        external_policy=_AllowedPolicy(),
        original_classification="P1",
        allow_external=False,
        routing_meta={"route": "fixed"},
        routing_warnings=["route warning"],
    )

    generate_result = flow.run(**base_kwargs)
    stream_result = flow.run(**base_kwargs, stream=True)

    assert generate_result.answer_text == "fallback"
    assert stream_result.answer_text == "fallback"
    assert observed[0] == {
        "router_warnings": ["route warning", "gen warning"],
        "payload_warnings": ["route warning", "gen warning"],
    }
    assert observed[1] == {
        "router_warnings": ["route warning", "gen warning"],
        "payload_warnings": ["route warning", "gen warning"],
    }


def test_answer_runtime_flow_stream_generation_error_omits_warning_payloads():
    observed: list[dict[str, object]] = []

    def _build_generation_fallback_payload(**kwargs):
        observed.append(
            {
                "router_warnings": kwargs["router_warnings"],
                "payload_warnings": kwargs["payload_warnings"],
            }
        )
        return {"answer": kwargs["answer"]}

    flow = AnswerRuntimeFlow(
        AnswerRuntimeFlowDeps(
            build_blocked_payload_fn=lambda **kwargs: kwargs,
            build_generation_fallback_payload_fn=_build_generation_fallback_payload,
            build_success_payload_fn=lambda **kwargs: kwargs,
            initial_generation_result_fn=lambda **_kwargs: SimpleNamespace(
                is_fallback=True,
                fallback_kind="generation_error",
                fallback_answer="stream fallback",
                generation_meta={"stage": "initial_stream_answer"},
                fallback_verification={"status": "verified"},
                fallback_rewrite={"finalAnswerSource": "generation_fallback"},
                generation_warnings=["ignored warning"],
            ),
            postprocess_result_fn=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("postprocess should not run")),
        )
    )

    result = flow.run(
        query="attention mechanism",
        retrieval_mode="hybrid",
        pipeline_result=_pipeline_result(),
        evidence_packet=_evidence_packet(),
        selected_llm=FailingLLM(RuntimeError("boom")),
        claim_verification=[],
        claim_consensus={},
        claim_consensus_merge_mode="advisory",
        answer_prompt="prompt",
        safe_context="context",
        external_policy=_AllowedPolicy(),
        original_classification="P1",
        allow_external=False,
        routing_meta={"route": "fixed"},
        routing_warnings=["route warning"],
        stream=True,
    )

    assert result.answer_text == "stream fallback"
    assert observed == [{"router_warnings": None, "payload_warnings": None}]


def test_answer_orchestrator_generate_restores_active_request_llm_state(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    previous_llm = object()
    selected_llm = StaticLLM("plain answer")
    searcher._active_request_llm = previous_llm
    searcher._caches.active_request_llm = previous_llm
    orchestrator = AnswerOrchestrator(searcher)
    evidence_packet = _evidence_packet()

    monkeypatch.setattr(orchestrator, "_resolve_llm_for_request", lambda **_kwargs: (selected_llm, {"route": "fixed"}, []))
    monkeypatch.setattr(orchestrator, "_default_answer_inputs", lambda **_kwargs: ("prompt", "context"))
    monkeypatch.setattr(orchestrator, "_adjudicate_claims", lambda **_kwargs: ([], {}, ""))
    monkeypatch.setattr(orchestrator, "_evaluate_policy", lambda **_kwargs: ("context", _AllowedPolicy(), "P1"))
    monkeypatch.setattr(orchestrator, "_record_answer_log", lambda **_kwargs: None)
    monkeypatch.setattr(orchestrator, "_base_payload", lambda **kwargs: {"answer": kwargs["answer"], "status": kwargs.get("status", "ok")})

    def _verify(**_kwargs):
        assert searcher._active_request_llm is selected_llm
        assert searcher._caches.active_request_llm is selected_llm
        return {
            "status": "verified",
            "supportedClaimCount": 1,
            "unsupportedClaimCount": 0,
            "uncertainClaimCount": 0,
            "conflictMentioned": False,
            "needsCaution": False,
            "warnings": [],
        }

    monkeypatch.setattr(orchestrator, "_verify_answer", _verify)
    monkeypatch.setattr(orchestrator, "_rewrite_answer", lambda **kwargs: (kwargs["answer"], {"attempted": False, "applied": False, "finalAnswerSource": "original", "warnings": []}))
    monkeypatch.setattr(orchestrator, "_apply_conservative_fallback_if_needed", lambda **kwargs: (kwargs["answer"], kwargs["rewrite_meta"], kwargs["verification"]))

    payload = orchestrator.generate(
        query="attention mechanism",
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
        pipeline_result=_pipeline_result(),
        evidence_packet=evidence_packet,
    )

    assert payload["answer"] == "plain answer"
    assert searcher._active_request_llm is previous_llm
    assert searcher._caches.active_request_llm is previous_llm


def test_answer_orchestrator_stream_restores_active_request_llm_state(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    previous_llm = object()
    selected_llm = StaticLLM("plain answer")
    searcher._active_request_llm = previous_llm
    searcher._caches.active_request_llm = previous_llm
    orchestrator = AnswerOrchestrator(searcher)
    evidence_packet = _evidence_packet()

    monkeypatch.setattr(orchestrator, "_resolve_llm_for_request", lambda **_kwargs: (selected_llm, {"route": "fixed"}, []))
    monkeypatch.setattr(orchestrator, "_default_answer_inputs", lambda **_kwargs: ("prompt", "context"))
    monkeypatch.setattr(orchestrator, "_adjudicate_claims", lambda **_kwargs: ([], {}, ""))
    monkeypatch.setattr(orchestrator, "_evaluate_policy", lambda **_kwargs: ("context", _AllowedPolicy(), "P1"))
    monkeypatch.setattr(orchestrator, "_record_answer_log", lambda **_kwargs: None)
    monkeypatch.setattr(orchestrator, "_base_payload", lambda **kwargs: {"answer": kwargs["answer"], "status": kwargs.get("status", "ok")})

    def _verify(**_kwargs):
        assert searcher._active_request_llm is selected_llm
        assert searcher._caches.active_request_llm is selected_llm
        return {
            "status": "verified",
            "supportedClaimCount": 1,
            "unsupportedClaimCount": 0,
            "uncertainClaimCount": 0,
            "conflictMentioned": False,
            "needsCaution": False,
            "warnings": [],
        }

    monkeypatch.setattr(orchestrator, "_verify_answer", _verify)
    monkeypatch.setattr(orchestrator, "_rewrite_answer", lambda **kwargs: (kwargs["answer"], {"attempted": False, "applied": False, "finalAnswerSource": "original", "warnings": []}))
    monkeypatch.setattr(orchestrator, "_apply_conservative_fallback_if_needed", lambda **kwargs: (kwargs["answer"], kwargs["rewrite_meta"], kwargs["verification"]))

    streamed = "".join(
        orchestrator.stream(
            query="attention mechanism",
            source_type="paper",
            retrieval_mode="hybrid",
            allow_external=False,
            pipeline_result=_pipeline_result(),
            evidence_packet=evidence_packet,
        )
    )

    assert streamed == "plain answer"
    assert searcher._active_request_llm is previous_llm
    assert searcher._caches.active_request_llm is previous_llm


def test_answer_orchestrator_generate_restores_active_request_llm_after_exception(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    previous_llm = object()
    selected_llm = StaticLLM("plain answer")
    searcher._active_request_llm = previous_llm
    searcher._caches.active_request_llm = previous_llm
    orchestrator = AnswerOrchestrator(searcher)
    evidence_packet = _evidence_packet()

    monkeypatch.setattr(orchestrator, "_resolve_llm_for_request", lambda **_kwargs: (selected_llm, {"route": "fixed"}, []))
    monkeypatch.setattr(orchestrator, "_adjudicate_claims", lambda **_kwargs: ([], {}, ""))
    monkeypatch.setattr(orchestrator, "_evaluate_policy", lambda **_kwargs: ("context", _AllowedPolicy(), "P1"))
    monkeypatch.setattr(orchestrator, "_default_answer_inputs", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    try:
        orchestrator.generate(
            query="attention mechanism",
            source_type="paper",
            retrieval_mode="hybrid",
            allow_external=False,
            pipeline_result=_pipeline_result(),
            evidence_packet=evidence_packet,
        )
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected RuntimeError")

    assert searcher._active_request_llm is previous_llm
    assert searcher._caches.active_request_llm is previous_llm


def test_answer_orchestrator_stream_restores_active_request_llm_after_exception(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    previous_llm = object()
    selected_llm = StaticLLM("plain answer")
    searcher._active_request_llm = previous_llm
    searcher._caches.active_request_llm = previous_llm
    orchestrator = AnswerOrchestrator(searcher)
    evidence_packet = _evidence_packet()

    monkeypatch.setattr(orchestrator, "_resolve_llm_for_request", lambda **_kwargs: (selected_llm, {"route": "fixed"}, []))
    monkeypatch.setattr(orchestrator, "_adjudicate_claims", lambda **_kwargs: ([], {}, ""))
    monkeypatch.setattr(orchestrator, "_evaluate_policy", lambda **_kwargs: ("context", _AllowedPolicy(), "P1"))
    monkeypatch.setattr(orchestrator, "_default_answer_inputs", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom-stream")))

    try:
        list(
            orchestrator.stream(
                query="attention mechanism",
                source_type="paper",
                retrieval_mode="hybrid",
                allow_external=False,
                pipeline_result=_pipeline_result(),
                evidence_packet=evidence_packet,
            )
        )
    except RuntimeError as exc:
        assert str(exc) == "boom-stream"
    else:
        raise AssertionError("expected RuntimeError")

    assert searcher._active_request_llm is previous_llm
    assert searcher._caches.active_request_llm is previous_llm


def test_answer_orchestrator_generate_does_not_leak_active_request_llm_between_requests(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    evidence_packet = _evidence_packet()
    first_llm = StaticLLM("first answer")
    second_llm = StaticLLM("second answer")
    selected_llms = iter([first_llm, second_llm])
    seen: list[object] = []

    monkeypatch.setattr(orchestrator, "_resolve_llm_for_request", lambda **_kwargs: (next(selected_llms), {"route": "fixed"}, []))
    monkeypatch.setattr(orchestrator, "_default_answer_inputs", lambda **_kwargs: ("prompt", "context"))
    monkeypatch.setattr(orchestrator, "_adjudicate_claims", lambda **_kwargs: ([], {}, ""))
    monkeypatch.setattr(orchestrator, "_evaluate_policy", lambda **_kwargs: ("context", _AllowedPolicy(), "P1"))
    monkeypatch.setattr(orchestrator, "_record_answer_log", lambda **_kwargs: None)
    monkeypatch.setattr(orchestrator, "_base_payload", lambda **kwargs: {"answer": kwargs["answer"], "status": kwargs.get("status", "ok")})

    def _verify(**_kwargs):
        seen.append(searcher._active_request_llm)
        return {
            "status": "verified",
            "supportedClaimCount": 1,
            "unsupportedClaimCount": 0,
            "uncertainClaimCount": 0,
            "conflictMentioned": False,
            "needsCaution": False,
            "warnings": [],
        }

    monkeypatch.setattr(orchestrator, "_verify_answer", _verify)
    monkeypatch.setattr(orchestrator, "_rewrite_answer", lambda **kwargs: (kwargs["answer"], {"attempted": False, "applied": False, "finalAnswerSource": "original", "warnings": []}))
    monkeypatch.setattr(orchestrator, "_apply_conservative_fallback_if_needed", lambda **kwargs: (kwargs["answer"], kwargs["rewrite_meta"], kwargs["verification"]))

    first_payload = orchestrator.generate(
        query="attention mechanism",
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
        pipeline_result=_pipeline_result(),
        evidence_packet=evidence_packet,
    )
    second_payload = orchestrator.generate(
        query="attention mechanism",
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
        pipeline_result=_pipeline_result(),
        evidence_packet=evidence_packet,
    )

    assert first_payload["answer"] == "first answer"
    assert second_payload["answer"] == "second answer"
    assert seen == [first_llm, second_llm]
    assert searcher._active_request_llm is None
    assert searcher._caches.active_request_llm is None


def test_answer_orchestrator_generate_uses_fallback_wrapper_when_no_llm_route(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    evidence_packet = _evidence_packet()
    observed: dict[str, object] = {}

    monkeypatch.setattr(orchestrator, "_resolve_llm_for_request", lambda **_kwargs: (None, {"route": "fixed"}, []))
    monkeypatch.setattr(orchestrator, "_default_answer_inputs", lambda **_kwargs: ("prompt", "context"))
    monkeypatch.setattr(orchestrator, "_adjudicate_claims", lambda **_kwargs: ([], {}, ""))
    monkeypatch.setattr(orchestrator, "_evaluate_policy", lambda **_kwargs: ("context", _AllowedPolicy(), "P1"))
    monkeypatch.setattr(orchestrator, "_record_answer_log", lambda **_kwargs: None)
    monkeypatch.setattr(orchestrator, "_base_payload", lambda **kwargs: {"answer": kwargs["answer"], "status": kwargs.get("status", "ok")})

    def _fallback(**kwargs):
        observed.update(kwargs)
        return (
            "generated fallback",
            {"stage": kwargs["stage"]},
            {"status": "verified"},
            {"finalAnswerSource": "generation_fallback"},
            ["fallback warning"],
        )

    monkeypatch.setattr(orchestrator, "_build_answer_generation_fallback", _fallback)

    payload = orchestrator.generate(
        query="attention mechanism",
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
        pipeline_result=_pipeline_result(),
        evidence_packet=evidence_packet,
    )

    assert payload["answer"] == "generated fallback"
    assert payload["answerGeneration"] == {"stage": "initial_answer"}
    assert payload["answerRewrite"] == {"finalAnswerSource": "generation_fallback"}
    assert observed["stage"] == "initial_answer"
    assert observed["routing_meta"] == {"route": "fixed"}


def test_answer_orchestrator_stream_uses_fallback_wrapper_when_initial_stream_generation_errors(monkeypatch):
    searcher = RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=FakeLLM())
    orchestrator = AnswerOrchestrator(searcher)
    evidence_packet = _evidence_packet()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        orchestrator,
        "_resolve_llm_for_request",
        lambda **_kwargs: (FailingLLM(RuntimeError("stream boom")), {"route": "fixed"}, []),
    )
    monkeypatch.setattr(orchestrator, "_default_answer_inputs", lambda **_kwargs: ("prompt", "context"))
    monkeypatch.setattr(orchestrator, "_adjudicate_claims", lambda **_kwargs: ([], {}, ""))
    monkeypatch.setattr(orchestrator, "_evaluate_policy", lambda **_kwargs: ("context", _AllowedPolicy(), "P1"))
    monkeypatch.setattr(orchestrator, "_record_answer_log", lambda **_kwargs: None)
    monkeypatch.setattr(orchestrator, "_base_payload", lambda **kwargs: {"answer": kwargs["answer"], "status": kwargs.get("status", "ok")})

    def _fallback(**kwargs):
        observed.update(kwargs)
        return (
            "stream fallback",
            {"stage": kwargs["stage"]},
            {"status": "verified"},
            {"finalAnswerSource": "generation_fallback"},
            ["fallback warning"],
        )

    monkeypatch.setattr(orchestrator, "_build_answer_generation_fallback", _fallback)

    streamed = "".join(
        orchestrator.stream(
            query="attention mechanism",
            source_type="paper",
            retrieval_mode="hybrid",
            allow_external=False,
            pipeline_result=_pipeline_result(),
            evidence_packet=evidence_packet,
        )
    )

    assert streamed == "stream fallback"
    assert observed["stage"] == "initial_stream_answer"
    assert observed["routing_meta"] == {"route": "fixed"}
