from __future__ import annotations

import json

from click.testing import CliRunner

from knowledge_hub.interfaces.cli.commands import search_cmd


class _FakeFactory:
    def get_searcher(self):
        return _FakeSearcher()


class _ClaimSQLite:
    def get_note(self, note_id):  # noqa: ANN001
        if note_id == "note:rag":
            return {"id": "note:rag"}
        return None

    def list_notes(self, limit=50, offset=0, source_type=None, para_category=None):  # noqa: ANN001
        _ = (limit, offset, source_type, para_category)
        return []

    def list_claims_by_note(self, note_id, limit=100):  # noqa: ANN001
        _ = limit
        if note_id == "note:rag":
            return [
                {"claim_id": "claim:rag-f1", "confidence": 0.91},
            ]
        return []

    def list_claims_by_record(self, record_id, limit=100):  # noqa: ANN001
        _ = limit
        if record_id == "2501.00001":
            return [
                {"claim_id": "claim:paper-acc", "confidence": 0.88},
            ]
        return []

    def list_claim_normalizations(self, claim_ids=None, status=None, limit=100, **kwargs):  # noqa: ANN001
        _ = (status, limit, kwargs)
        rows = []
        for claim_id in claim_ids or []:
            if claim_id == "claim:rag-f1":
                rows.append(
                    {
                        "claim_id": claim_id,
                        "dataset": "HotpotQA",
                        "metric": "F1",
                        "comparator": "baseline",
                        "result_direction": "better",
                        "result_value_numeric": 82.1,
                    }
                )
            if claim_id == "claim:paper-acc":
                rows.append(
                    {
                        "claim_id": claim_id,
                        "dataset": "MMLU",
                        "metric": "Accuracy",
                        "comparator": "prior-work",
                        "result_direction": "better",
                        "result_value_numeric": 71.5,
                    }
                )
        return rows


class _FakeSearcher:
    def __init__(self):
        self.sqlite_db = object()
        self.last_allow_external = None
        self.config = type("Cfg", (), {"summarization_provider": "openai"})()

    def search(self, *args, **kwargs):  # noqa: ANN001
        _ = (args, kwargs)
        return [
            type(
                "Result",
                (),
                {
                    "metadata": {
                        "title": "RAG Note",
                        "source_type": "note",
                        "file_path": "Projects/AI/RAG Note.md",
                        "links": ["Projects/AI/RAG Architecture.md"],
                        "cluster_id": "cluster-rag",
                        "resolved_parent_id": "p1",
                        "resolved_parent_label": "Section A",
                        "resolved_parent_chunk_span": "0-2",
                    },
                    "score": 0.92,
                    "semantic_score": 0.9,
                    "lexical_score": 0.8,
                    "retrieval_mode": "hybrid",
                    "lexical_extras": {
                        "quality_flag": "ok",
                        "source_trust_score": 0.94,
                        "reference_role": "glossary_reference",
                        "reference_tier": "specialist",
                        "ranking_signals": {
                            "quality_flag": "ok",
                            "reference_role": "glossary_reference",
                            "reference_tier": "specialist",
                        },
                        "top_ranking_signals": [{"name": "exact_title_overlap", "weight": 0.2}],
                    },
                    "document": "retrieval augmented generation",
                    "distance": 0.1,
                    "document_id": "doc1",
                },
            )(),
            type(
                "Result",
                (),
                {
                    "metadata": {
                        "title": "RAG Architecture",
                        "source_type": "vault",
                        "file_path": "Projects/AI/RAG Architecture.md",
                        "cluster_id": "cluster-rag",
                        "resolved_parent_id": "p2",
                        "resolved_parent_label": "Section B",
                        "resolved_parent_chunk_span": "0-1",
                    },
                    "score": 0.79,
                    "semantic_score": 0.78,
                    "lexical_score": 0.52,
                    "retrieval_mode": "hybrid",
                    "lexical_extras": {},
                    "document": "rag architecture tradeoffs",
                    "distance": 0.2,
                    "document_id": "doc2",
                },
            )(),
        ]

    def build_ops_report(self, *args, **kwargs):
        _ = (args, kwargs)
        return {
            "schema": "knowledge-hub.rag.report.result.v1",
            "status": "ok",
            "days": 7,
            "limit": 100,
            "counts": {
                "total": 2,
                "needsCaution": 1,
                "rewriteAttempted": 1,
                "rewriteApplied": 1,
                "conservativeFallback": 0,
                "unsupportedClaimLogs": 1,
            },
            "verification": {"verified": 1, "caution": 1, "failed": 0, "skipped": 0, "unknown": 0},
            "rates": {
                "needsCautionRate": 0.5,
                "rewriteAttemptedRate": 0.5,
                "rewriteAppliedRate": 0.5,
                "conservativeFallbackRate": 0.0,
                "unsupportedClaimRate": 0.5,
            },
            "topWarningPatterns": [{"warning": "answer verification caution", "count": 1}],
            "alerts": [
                {
                    "code": "rag_verification_failed_or_skipped",
                    "severity": "warning",
                    "scope": "rag",
                    "summary": "verification failed/skipped 사례가 1건 있습니다.",
                    "metric": "verification.failed+skipped",
                    "observed": 1,
                    "threshold": "> 0",
                }
            ],
            "recommendedActions": [
                {
                    "actionType": "inspect_verification_routes",
                    "scope": "rag",
                    "summary": "verification route와 policy/config 경로를 점검하세요.",
                    "reasonCodes": ["rag_verification_failed_or_skipped"],
                    "command": "khub",
                    "args": ["config", "list"],
                }
            ],
            "samples": [
                {
                    "id": 1,
                    "createdAt": "2026-03-16T10:00:00Z",
                    "queryHash": "abc",
                    "queryDigest": "RAG란 무엇인가...",
                    "resultStatus": "ok",
                    "verificationStatus": "caution",
                    "needsCaution": True,
                    "unsupportedClaimCount": 1,
                    "rewriteApplied": True,
                    "finalAnswerSource": "rewritten",
                    "warningCount": 1,
                }
            ],
            "warnings": [],
        }

    def generate_answer(self, *args, **kwargs):
        self.last_allow_external = kwargs.get("allow_external")
        source_type = kwargs.get("source_type")
        paper_memory_mode = str(kwargs.get("paper_memory_mode", "off") or "off").strip().lower()
        effective_paper_memory_mode = "compat" if paper_memory_mode == "prefilter" else paper_memory_mode
        paper_memory_enabled = source_type == "paper" and effective_paper_memory_mode in {"compat", "on"}
        return {
            "answer": "재작성된 RAG 답변 본문",
            "answerRewrite": {
                "attempted": True,
                "applied": True,
                "triggeredBy": ["unsupported_claim"],
                "attemptCount": 1,
                "summary": "unsupported claim을 제거하기 위해 답변을 재작성했습니다.",
                "originalAnswer": "원본 RAG 답변 본문",
                "finalAnswerSource": "rewritten",
            },
            "answerVerification": {
                "status": "caution",
                "supportedClaimCount": 1,
                "unsupportedClaimCount": 1,
                "uncertainClaimCount": 1,
                "conflictMentioned": False,
                "needsCaution": True,
                "summary": "근거 밖의 주장과 불확실한 claim이 섞여 있습니다.",
            },
            "warnings": [
                "answer verification caution: unsupported=1 uncertain=1 conflict_mentioned=False"
            ],
            "sources": [
                {
                    "title": "RAG Note",
                    "source_type": "note",
                    "score": 0.92,
                    "semantic_score": 0.9,
                    "lexical_score": 0.8,
                }
            ],
            "citations": [
                {"label": "S1", "title": "RAG Note", "target": "Projects/AI/RAG Note.md", "kind": "file"}
            ],
            "paperMemoryPrefilter": {
                "requestedMode": paper_memory_mode,
                "effectiveMode": effective_paper_memory_mode,
                "modeAliasApplied": paper_memory_mode == "prefilter",
                "applied": paper_memory_enabled,
                "fallbackUsed": False,
                "matchedPaperIds": ["2501.00001"] if paper_memory_enabled else [],
                "matchedMemoryIds": ["paper-memory:2501.00001:abc"] if paper_memory_enabled else [],
                "reason": "matched_cards" if paper_memory_enabled else "source_not_paper" if effective_paper_memory_mode in {"compat", "on"} else "disabled",
            },
        }


class _DiagnosticSearcher(_FakeSearcher):
    def search_with_diagnostics(self, *args, **kwargs):  # noqa: ANN001
        results = self.search(*args, **kwargs)
        return {
            "results": results,
            "diagnostics": {
                "retrievalPlan": {"queryIntent": "paper_lookup", "retrievalMode": "hybrid"},
                "rerankSignals": {
                    "rerankerApplied": False,
                    "rerankerModel": "BAAI/bge-reranker-v2-m3",
                    "rerankerWindow": 8,
                    "rerankerLatencyMs": 1201,
                    "rerankerFallbackUsed": True,
                    "rerankerReason": "timeout",
                },
            },
        }


class _LegacySearchMethodSearcher(_FakeSearcher):
    def __init__(self):
        super().__init__()
        self.last_search_kwargs = None

    def search(self, query, top_k=10, source_type=None):  # noqa: ANN001
        self.last_search_kwargs = {
            "query": query,
            "top_k": top_k,
            "source_type": source_type,
        }
        return super().search(query, top_k=top_k, source_type=source_type)


class _EmptySearcher(_FakeSearcher):
    def search(self, *args, **kwargs):  # noqa: ANN001
        _ = (args, kwargs)
        return []


class _DiagnosticFactory:
    def get_searcher(self):
        return _DiagnosticSearcher()


class _LegacySearchMethodFactory:
    def __init__(self):
        self.searcher = _LegacySearchMethodSearcher()

    def get_searcher(self):
        return self.searcher


class _LegacyAskMethodSearcher(_FakeSearcher):
    def __init__(self):
        super().__init__()
        self.last_generate_kwargs = None

    def generate_answer(self, query, top_k=5, min_score=0.0, source_type=None):  # noqa: ANN001
        self.last_generate_kwargs = {
            "query": query,
            "top_k": top_k,
            "min_score": min_score,
            "source_type": source_type,
        }
        return {
            "answer": "legacy ask answer",
            "sources": [
                {
                    "title": "Alpha",
                    "source_type": "vault",
                    "score": 0.91,
                }
            ],
            "citations": [],
        }


class _LegacyAskMethodFactory:
    def __init__(self):
        self.searcher = _LegacyAskMethodSearcher()

    def get_searcher(self):
        return self.searcher


class _ClaimSignalSearcher(_FakeSearcher):
    def __init__(self):
        super().__init__()
        self.sqlite_db = _ClaimSQLite()

    def search(self, *args, **kwargs):  # noqa: ANN001
        _ = (args, kwargs)
        return [
            type(
                "Result",
                (),
                {
                    "metadata": {
                        "title": "RAG Note",
                        "source_type": "note",
                        "note_id": "note:rag",
                        "file_path": "Projects/AI/RAG Note.md",
                        "resolved_parent_id": "note:rag",
                        "resolved_parent_label": "RAG Note",
                    },
                    "score": 0.92,
                    "semantic_score": 0.9,
                    "lexical_score": 0.8,
                    "retrieval_mode": "hybrid",
                    "lexical_extras": {},
                    "document": "retrieval augmented generation",
                    "distance": 0.1,
                    "document_id": "doc1",
                },
            )(),
            type(
                "Result",
                (),
                {
                    "metadata": {
                        "title": "Paper Memory",
                        "source_type": "paper",
                        "arxiv_id": "2501.00001",
                        "resolved_parent_id": "paper:2501.00001",
                        "resolved_parent_label": "Paper Memory",
                    },
                    "score": 0.89,
                    "semantic_score": 0.84,
                    "lexical_score": 0.61,
                    "retrieval_mode": "hybrid",
                    "lexical_extras": {},
                    "document": "paper evidence",
                    "distance": 0.2,
                    "document_id": "doc2",
                },
            )(),
        ]


class _ClaimSignalFactory:
    def get_searcher(self):
        return _ClaimSignalSearcher()


class _BrokenFactory:
    def get_searcher(self):
        raise RuntimeError("broken vector store")


def test_ask_command_prints_answer_verification_and_warnings():
    runner = CliRunner()
    khub = type("Ctx", (), {"factory": _FakeFactory(), "config": type("Cfg", (), {"summarization_provider": "openai"})()})()
    result = runner.invoke(
        search_cmd.ask,
        ["RAG란?"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0
    assert "재작성: rewritten" in result.output
    assert "검증: caution" in result.output
    assert "unsupported=1" in result.output
    assert "경고:" in result.output
    assert "answer verification caution" in result.output
    assert "allow_external=True" in result.output


def test_ask_command_prints_paper_memory_prefilter_when_enabled():
    runner = CliRunner()
    result = runner.invoke(
        search_cmd.ask,
        ["RAG란?", "--source", "paper", "--paper-memory-mode", "prefilter"],
        obj={"khub": type("Ctx", (), {"factory": _FakeFactory()})()},
    )

    assert result.exit_code == 0
    assert "paper-memory route" in result.output
    assert "effective=compat" in result.output
    assert "matched papers: 2501.00001" in result.output


def test_search_and_ask_json_payloads_include_runtime_diagnostics(monkeypatch):
    runtime = {
        "schema": "knowledge-hub.runtime.diagnostics.v1",
        "status": "degraded",
        "degraded": True,
        "providers": [],
        "semanticRetrieval": {"provider": "ollama", "model": "nomic-embed-text", "available": True},
        "warnings": ["embedder_runtime_warnings"],
    }
    monkeypatch.setattr(search_cmd, "build_runtime_diagnostics", lambda config, searcher=None, searcher_error="": runtime)
    monkeypatch.setattr(
        search_cmd,
        "_graph_query_signal",
        lambda searcher, query: {"is_graph_heavy": query.lower().startswith("rag"), "recommended_mode": "graph"},
    )

    runner = CliRunner()
    search_result = runner.invoke(
        search_cmd.search,
        ["rag", "--json"],
        obj={"khub": type("Ctx", (), {"factory": _FakeFactory()})()},
    )
    assert search_result.exit_code == 0
    search_payload = json.loads(search_result.output)
    assert search_payload["runtimeDiagnostics"]["status"] == "degraded"
    assert search_payload["runtimeDiagnostics"]["semanticRetrieval"]["provider"] == "ollama"
    assert search_payload["results"][0]["normalized_source_type"] == "vault"
    assert search_payload["results"][0]["normalizedSourceType"] == "vault"
    assert isinstance(search_payload["results"][0]["top_ranking_signals"], list)
    assert isinstance(search_payload["results"][0]["topRankingSignals"], list)
    assert search_payload["related_notes"][0]["title"] == "RAG Architecture"
    assert search_payload["relatedNotes"][0]["title"] == "RAG Architecture"
    assert search_payload["graph_query_signal"]["is_graph_heavy"] is True
    assert search_payload["graphQuerySignal"]["recommended_mode"] == "graph"

    ask_result = runner.invoke(
        search_cmd.ask,
        ["RAG란?", "--json"],
        obj={"khub": type("Ctx", (), {"factory": _FakeFactory(), "config": type("Cfg", (), {"summarization_provider": "openai"})()})()},
    )
    assert ask_result.exit_code == 0
    ask_payload = json.loads(ask_result.output)
    assert ask_payload["runtimeDiagnostics"]["degraded"] is True
    assert ask_payload["runtimeDiagnostics"]["warnings"] == ["embedder_runtime_warnings"]
    assert ask_payload["citations"][0]["label"] == "S1"
    assert ask_payload["graph_query_signal"]["is_graph_heavy"] is True
    assert ask_payload["allowExternal"] is True
    assert ask_payload["externalPolicy"]["contractRole"] == "answer_generation_external_policy"
    assert ask_payload["externalPolicy"]["decisionSource"] == "configured_summarization_provider"
    assert ask_payload["externalPolicy"]["policyMode"] == "external-allowed"


def test_search_command_includes_runtime_diagnostics_when_no_results(monkeypatch):
    runtime = {
        "schema": "knowledge-hub.runtime.diagnostics.v1",
        "status": "ok",
        "degraded": False,
        "providers": [],
        "semanticRetrieval": {"provider": "ollama", "model": "nomic-embed-text", "available": True},
        "warnings": [],
    }
    monkeypatch.setattr(search_cmd, "build_runtime_diagnostics", lambda config, searcher=None, searcher_error="": runtime)
    monkeypatch.setattr(
        search_cmd,
        "_graph_query_signal",
        lambda searcher, query: {"is_graph_heavy": False, "recommended_mode": "baseline"},
    )
    monkeypatch.setattr(search_cmd, "_get_searcher", lambda khub_ctx: _EmptySearcher())

    runner = CliRunner()
    result = runner.invoke(
        search_cmd.search,
        ["missing", "--json"],
        obj={"khub": type("Ctx", (), {"factory": _FakeFactory()})()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["results"] == []
    assert payload["runtimeDiagnostics"]["status"] == "ok"
    assert payload["graph_query_signal"]["recommended_mode"] == "baseline"


def test_search_json_keeps_contract_on_searcher_init_failure(monkeypatch):
    runtime = {
        "schema": "knowledge-hub.runtime.diagnostics.v1",
        "status": "degraded",
        "degraded": True,
        "providers": [],
        "semanticRetrieval": {"provider": "ollama", "model": "bge-m3:latest", "available": False},
        "warnings": ["searcher init failed: broken vector store"],
    }
    monkeypatch.setattr(search_cmd, "build_runtime_diagnostics", lambda config, searcher=None, searcher_error="": runtime)

    runner = CliRunner()
    result = runner.invoke(
        search_cmd.search,
        ["retrieval", "--json"],
        obj={"khub": type("Ctx", (), {"factory": _BrokenFactory(), "config": object()})()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.search.result.v1"
    assert payload["status"] == "init_error"
    assert payload["query"] == "retrieval"
    assert payload["results"] == []
    assert payload["initError"] == "broken vector store"
    assert payload["runtimeDiagnostics"]["status"] == "degraded"


def test_search_json_payload_includes_reranker_diagnostics_when_searcher_supports_it(monkeypatch):
    runtime = {
        "schema": "knowledge-hub.runtime.diagnostics.v1",
        "status": "ok",
        "degraded": False,
        "providers": [],
        "semanticRetrieval": {"provider": "ollama", "model": "bge-m3:latest", "available": True},
        "warnings": [],
    }
    monkeypatch.setattr(search_cmd, "build_runtime_diagnostics", lambda config, searcher=None, searcher_error="": runtime)
    monkeypatch.setattr(search_cmd, "_graph_query_signal", lambda searcher, query: {"is_graph_heavy": False})

    runner = CliRunner()
    result = runner.invoke(
        search_cmd.search,
        ["paper memory card", "--json"],
        obj={"khub": type("Ctx", (), {"factory": _DiagnosticFactory()})()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["rerankSignals"]["rerankerReason"] == "timeout"
    assert payload["rerank_signals"]["rerankerFallbackUsed"] is True
    assert payload["retrievalPlan"]["queryIntent"] == "paper_lookup"
    assert payload["retrieval_plan"]["retrievalMode"] == "hybrid"


def test_ask_json_keeps_contract_on_searcher_init_failure(monkeypatch):
    runtime = {
        "schema": "knowledge-hub.runtime.diagnostics.v1",
        "status": "degraded",
        "degraded": True,
        "providers": [],
        "semanticRetrieval": {"provider": "ollama", "model": "bge-m3:latest", "available": False},
        "warnings": ["searcher init failed: broken vector store"],
    }
    monkeypatch.setattr(search_cmd, "build_runtime_diagnostics", lambda config, searcher=None, searcher_error="": runtime)

    runner = CliRunner()
    result = runner.invoke(
        search_cmd.ask,
        ["RAG란?", "--json"],
        obj={"khub": type("Ctx", (), {"factory": _BrokenFactory(), "config": object()})()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.ask.result.v1"
    assert payload["status"] == "init_error"
    assert payload["question"] == "RAG란?"
    assert payload["answer"] == ""
    assert payload["initError"] == "broken vector store"
    assert payload["warnings"] == ["searcher init failed: broken vector store"]


def test_search_json_payload_includes_claim_signals():
    runner = CliRunner()
    result = runner.invoke(
        search_cmd.search,
        ["rag", "--json"],
        obj={"khub": type("Ctx", (), {"factory": _ClaimSignalFactory()})()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["claimSignals"]["resultsWithClaims"] == 2
    assert payload["claim_signals"]["totalNormalizedClaimCount"] == 2
    assert payload["results"][0]["claimSignals"]["claimCount"] == 1
    assert payload["results"][0]["claim_signals"]["topMetrics"][0]["name"] == "F1"
    assert payload["results"][1]["claimSignals"]["topDatasets"][0]["name"] == "MMLU"


def test_search_command_filters_unknown_kwargs_for_legacy_searcher():
    runner = CliRunner()
    factory = _LegacySearchMethodFactory()
    result = runner.invoke(
        search_cmd.search,
        ["rag", "--source", "vault", "--mode", "keyword", "--json"],
        obj={"khub": type("Ctx", (), {"factory": factory})()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.search.result.v1"
    assert factory.searcher.last_search_kwargs == {
        "query": "rag",
        "top_k": 10,
        "source_type": "vault",
    }


def test_ask_command_filters_unknown_kwargs_for_legacy_generate_answer():
    runner = CliRunner()
    factory = _LegacyAskMethodFactory()
    result = runner.invoke(
        search_cmd.ask,
        ["rag", "--source", "vault", "--mode", "keyword", "--json", "--no-allow-external"],
        obj={"khub": type("Ctx", (), {"factory": factory, "config": type("Cfg", (), {"summarization_provider": "openai"})()})()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.ask.result.v1"
    assert payload["question"] == "rag"
    assert payload["allowExternal"] is False
    assert payload["sources"][0]["source_type"] == "vault"
    assert factory.searcher.last_generate_kwargs == {
        "query": "rag",
        "top_k": 8,
        "min_score": 0.0,
        "source_type": "vault",
    }


def test_ask_command_keeps_paper_memory_prefilter_disabled_for_non_paper_source():
    runner = CliRunner()
    result = runner.invoke(
        search_cmd.ask,
        ["RAG란?", "--source", "web", "--paper-memory-mode", "prefilter"],
        obj={"khub": type("Ctx", (), {"factory": _FakeFactory(), "config": type("Cfg", (), {"summarization_provider": "openai"})()})()},
    )

    assert result.exit_code == 0
    assert "paper-memory route" in result.output
    assert "effective=compat" in result.output
    assert "applied=False" in result.output
    assert "reason=source_not_paper" in result.output
    assert "matched papers:" not in result.output


def test_ask_json_payload_normalizes_memory_modes():
    runner = CliRunner()
    result = runner.invoke(
        search_cmd.ask,
        ["RAG란?", "--source", "paper", "--memory-route-mode", "prefilter", "--paper-memory-mode", "prefilter", "--json"],
        obj={"khub": type("Ctx", (), {"factory": _FakeFactory(), "config": type("Cfg", (), {"summarization_provider": "openai"})()})()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["memoryRouteMode"] == "compat"
    assert payload["paperMemoryMode"] == "compat"
    assert payload["memoryRoute"]["contractRole"] == "ask_retrieval_memory_prefilter"
    assert payload["memoryRoute"]["requestedMode"] == "prefilter"
    assert payload["memoryRoute"]["effectiveMode"] == "compat"
    assert payload["memoryRoute"]["modeAliasApplied"] is True
    assert payload["memoryRoute"]["aliasDeprecated"] is True
    assert payload["memoryPrefilter"]["contractRole"] == "retrieval_memory_prefilter"
    assert payload["memoryPrefilter"]["requestedMode"] == "prefilter"
    assert payload["memoryPrefilter"]["effectiveMode"] == "compat"
    assert payload["paperMemoryPrefilter"]["contractRole"] == "paper_source_memory_prefilter"
    assert payload["paperMemoryPrefilter"]["requestedMode"] == "prefilter"
    assert payload["paperMemoryPrefilter"]["effectiveMode"] == "compat"
    assert payload["paperMemoryPrefilter"]["modeAliasApplied"] is True


def test_ask_command_can_force_no_allow_external():
    runner = CliRunner()
    result = runner.invoke(
        search_cmd.ask,
        ["RAG란?", "--no-allow-external"],
        obj={"khub": type("Ctx", (), {"factory": _FakeFactory(), "config": type("Cfg", (), {"summarization_provider": "openai"})()})()},
    )

    assert result.exit_code == 0
    assert "allow_external=False" in result.output
    assert "external policy=local-only" in result.output


def test_ask_command_passes_codex_answer_route_override(monkeypatch):
    observed = {}

    def _fake_generate(searcher, query, **kwargs):  # noqa: ANN001
        _ = searcher
        observed["query"] = query
        observed["kwargs"] = dict(kwargs)
        return {
            "answer": "codex route answer",
            "sources": [],
            "citations": [],
            "warnings": [],
            "router": {"selected": {"route": "api", "provider": "codex_mcp", "model": "gpt-5.4-codex"}},
        }

    monkeypatch.setattr(search_cmd, "_generate_answer_compat", _fake_generate)
    runner = CliRunner()
    result = runner.invoke(
        search_cmd.ask,
        ["RAG란?", "--answer-route", "codex", "--json"],
        obj={"khub": type("Ctx", (), {"factory": _FakeFactory(), "config": type("Cfg", (), {"summarization_provider": "openai"})()})()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert observed["query"] == "RAG란?"
    assert observed["kwargs"]["answer_route_override"] == "codex"
    assert payload["answerRouteRequested"] == "codex"
    assert payload["answerRouteApplied"] == "api"
    assert payload["answerProviderApplied"] == "codex_mcp"
    assert payload["answerModelApplied"] == "gpt-5.4-codex"
    assert payload["allowExternal"] is True


def test_ask_command_prints_applied_answer_route_details(monkeypatch):
    monkeypatch.setattr(
        search_cmd,
        "_generate_answer_compat",
        lambda searcher, query, **kwargs: {  # noqa: ARG005
            "answer": "codex route answer",
            "sources": [],
            "citations": [],
            "warnings": [],
            "router": {"selected": {"route": "api", "provider": "codex_mcp", "model": "gpt-5.4-codex"}},
        },
    )
    runner = CliRunner()
    result = runner.invoke(
        search_cmd.ask,
        ["RAG란?", "--answer-route", "codex"],
        obj={"khub": type("Ctx", (), {"factory": _FakeFactory(), "config": type("Cfg", (), {"summarization_provider": "openai"})()})()},
    )

    assert result.exit_code == 0
    assert "answer_route=api provider=codex_mcp model=gpt-5.4-codex" in result.output


def test_rag_report_command_returns_schema_backed_json(monkeypatch):
    monkeypatch.setattr(
        search_cmd,
        "build_rag_ops_report",
        lambda sqlite_db, *, limit, days: _FakeSearcher().build_ops_report(limit=limit, days=days),
    )
    runner = CliRunner()
    result = runner.invoke(
        search_cmd.rag_report,
        ["--json"],
        obj={"khub": type("Ctx", (), {"factory": _FakeFactory(), "config": type("Cfg", (), {"get_nested": lambda *args, **kwargs: False})()})()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.rag.report.result.v1"
    assert payload["counts"]["needsCaution"] == 1
    assert payload["samples"][0]["finalAnswerSource"] == "rewritten"
    assert payload["alerts"][0]["code"] == "rag_verification_failed_or_skipped"
    assert payload["recommendedActions"][0]["actionType"] == "inspect_verification_routes"


def test_rag_report_command_prints_alert_and_action_summary(monkeypatch):
    monkeypatch.setattr(
        search_cmd,
        "build_rag_ops_report",
        lambda sqlite_db, *, limit, days: _FakeSearcher().build_ops_report(limit=limit, days=days),
    )
    runner = CliRunner()
    result = runner.invoke(
        search_cmd.rag_report,
        [],
        obj={"khub": type("Ctx", (), {"factory": _FakeFactory(), "config": type("Cfg", (), {"get_nested": lambda *args, **kwargs: False})()})()},
    )

    assert result.exit_code == 0
    assert "rag_verification_failed_or_skipped" in result.output
    assert "verification route와 policy/config 경로를 점검하세요." in result.output
    assert "khub config list" in result.output
