from __future__ import annotations

from types import SimpleNamespace

from knowledge_hub.application.rag_corrective_report import (
    build_rag_adaptive_plan,
    build_rag_answerability_rerank,
    build_rag_answerability_rerank_eval_report,
    build_rag_corrective_eval_report,
    build_rag_corrective_execution_review,
    build_rag_corrective_report,
    build_rag_corrective_run,
    build_rag_graph_global_plan,
)
from knowledge_hub.core.schema_validator import validate_payload


def _result(**overrides):
    base = {
        "metadata": {
            "title": "RankRAG",
            "source_type": "paper",
            "resolved_parent_id": "paper:rankrag",
            "resolved_parent_label": "RankRAG paper",
        },
        "score": 0.42,
        "semantic_score": 0.4,
        "lexical_score": 0.2,
        "document_id": "doc-rankrag",
        "document": "RankRAG improves retrieval augmented generation.",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _Phase0DiagnosticSearcher:
    def __init__(self):
        self.calls = []

    def search_with_diagnostics(self, query, **kwargs):  # noqa: ANN001
        self.calls.append({"query": query, **kwargs})
        return {
            "results": [_result()],
            "diagnostics": {
                "retrievalPlan": {
                    "queryIntent": "paper_lookup",
                    "retrievalMode": "hybrid",
                },
                "retrievalStrategy": {
                    "phase": "phase0_diagnostics",
                    "complexityClass": "update_sensitive",
                    "retryPolicy": {
                        "mode": "diagnostics_only",
                        "maxRetries": 0,
                        "allowedActions": ["broaden_query_terms", "read_parent_section", "abstain"],
                    },
                },
                "retrievalQuality": {
                    "label": "low",
                    "score": 0.2,
                    "weakSignals": ["low_top_score"],
                    "correctiveActionCandidate": "broaden_search",
                },
                "answerabilityRerank": {
                    "applied": False,
                    "label": "low",
                    "score": 0.1,
                    "weakSignals": ["thin_evidence"],
                },
                "correctiveRetrieval": {
                    "applied": False,
                    "policy": "diagnostics_only",
                    "retryCandidate": True,
                    "candidateAction": "broaden_search",
                    "triggers": ["low_top_score"],
                },
                "artifactHealth": {
                    "label": "high",
                    "score": 1.0,
                },
                "candidateSources": [{"sourceType": "paper", "count": 1}],
                "rerankSignals": {"rerankerApplied": False},
                "memoryRoute": {"applied": False},
                "memoryPrefilter": {"applied": False},
                "paperMemoryPrefilter": {"applied": False},
            },
        }

    def generate_answer(self, *args, **kwargs):  # noqa: ANN001
        raise AssertionError("corrective report must not generate answers")

    def apply_write(self, *args, **kwargs):  # noqa: ANN001
        raise AssertionError("corrective report must not write")


class _LegacySearchOnlySearcher:
    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):  # noqa: ANN001
        self.calls.append({"query": query, **kwargs})
        return [
            _result(
                metadata={
                    "title": "Legacy RAG Note",
                    "source_type": "vault",
                    "parent_id": "note:legacy",
                },
                score=0.77,
                semantic_score=0.0,
                lexical_score=0.77,
                document_id="doc-legacy",
            )
        ]


class _GlobalDiagnosticSearcher(_Phase0DiagnosticSearcher):
    def search_with_diagnostics(self, query, **kwargs):  # noqa: ANN001
        payload = super().search_with_diagnostics(query, **kwargs)
        diagnostics = payload["diagnostics"]
        diagnostics["retrievalStrategy"]["complexityClass"] = "global_sensemaking"
        diagnostics["retrievalQuality"]["correctiveActionCandidate"] = "graph_or_hierarchy_probe"
        diagnostics["correctiveRetrieval"]["candidateAction"] = "graph_or_hierarchy_probe"
        diagnostics["correctiveRetrieval"]["retryCandidate"] = True
        return payload


class _AnswerabilitySearcher:
    def search_with_diagnostics(self, query, **kwargs):  # noqa: ANN001
        _ = (query, kwargs)
        return {
            "results": [
                _result(
                    metadata={"title": "Unrelated High Score", "source_type": "paper"},
                    score=0.9,
                    semantic_score=0.2,
                    lexical_score=0.1,
                    document="generic retrieval note",
                    document_id="doc-high",
                ),
                _result(
                    metadata={"title": "RankRAG benchmark update", "source_type": "paper"},
                    score=0.45,
                    semantic_score=0.45,
                    lexical_score=0.5,
                    document="rankrag benchmark update answerability evidence",
                    document_id="doc-answerable",
                ),
            ],
            "diagnostics": {
                "retrievalStrategy": {
                    "phase": "phase0_diagnostics",
                    "complexityClass": "update_sensitive",
                    "retryPolicy": {"mode": "diagnostics_only", "maxRetries": 0, "allowedActions": []},
                },
                "answerabilityRerank": {"applied": False, "score": 0.2},
            },
        }


class _StaticAnswerabilityEvalSearcher:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def search_with_diagnostics(self, query, **kwargs):  # noqa: ANN001
        self.calls.append({"query": query, **kwargs})
        return {
            "results": list(self.results),
            "diagnostics": {
                "retrievalStrategy": {
                    "phase": "phase0_diagnostics",
                    "complexityClass": "local_lookup",
                    "retryPolicy": {"mode": "diagnostics_only", "maxRetries": 0, "allowedActions": []},
                },
                "answerabilityRerank": {"applied": False, "score": 0.5},
            },
        }


class _WeakAnswerabilityNoRetrySearcher:
    def __init__(self, query_payloads):
        self.query_payloads = dict(query_payloads)
        self.calls = []

    def search_with_diagnostics(self, query, **kwargs):  # noqa: ANN001
        self.calls.append({"query": query, **kwargs})
        payload = dict(self.query_payloads[str(query)])
        complexity_class = str(payload.get("complexityClass") or "local_lookup")
        weak_signals = list(payload.get("weakSignals") or ["low_query_term_coverage", "no_high_confidence_support"])
        return {
            "results": [_result(score=0.31, semantic_score=0.2, lexical_score=0.1, document="thin unrelated evidence")],
            "diagnostics": {
                "retrievalPlan": {"retrievalMode": "hybrid"},
                "retrievalStrategy": {
                    "phase": "phase0_diagnostics",
                    "complexityClass": complexity_class,
                    "retryPolicy": {
                        "mode": "diagnostics_only",
                        "maxRetries": 0,
                        "allowedActions": ["broaden_query_terms", "read_parent_section", "abstain"],
                    },
                },
                "retrievalQuality": {
                    "label": "medium",
                    "score": 0.5,
                    "weakSignals": [],
                    "correctiveActionCandidate": "none",
                },
                "answerabilityRerank": {
                    "applied": False,
                    "label": "low",
                    "score": 0.2,
                    "weakSignals": weak_signals,
                },
                "correctiveRetrieval": {
                    "applied": False,
                    "policy": "diagnostics_only",
                    "maxRetries": 0,
                    "retryCandidate": False,
                    "candidateAction": "none",
                    "triggers": [],
                },
                "artifactHealth": {"label": "high", "score": 1.0},
                "candidateSources": [],
                "rerankSignals": {},
                "memoryRoute": {},
                "memoryPrefilter": {},
                "paperMemoryPrefilter": {},
            },
        }


class _ExecutionReviewSearcher:
    def __init__(
        self,
        *,
        action="broaden_search",
        retry_candidate=True,
        initial_quality=0.5,
        retry_quality=0.6,
        initial_answerability=0.2,
        retry_answerability=0.3,
        source_type="paper",
        answerability_weak_signals=None,
    ):
        self.action = action
        self.retry_candidate = retry_candidate
        self.initial_quality = initial_quality
        self.retry_quality = retry_quality
        self.initial_answerability = initial_answerability
        self.retry_answerability = retry_answerability
        self.source_type = source_type
        self.answerability_weak_signals = answerability_weak_signals
        self.calls = []

    def search_with_diagnostics(self, query, **kwargs):  # noqa: ANN001
        self.calls.append({"query": query, **kwargs})
        is_retry = any(token in str(query) for token in ("overview evidence related sources", "exact source id title arxiv", "hierarchy graph related parent context"))
        docs = ["doc-a", "doc-b", "doc-c"] if not is_retry else ["doc-a", "doc-b", "doc-d", "doc-e"]
        quality = self.retry_quality if is_retry else self.initial_quality
        answerability = self.retry_answerability if is_retry else self.initial_answerability
        action = "none" if is_retry or not self.retry_candidate else self.action
        return {
            "results": [
                _result(
                    metadata={"title": document_id, "source_type": self.source_type, "resolved_parent_id": document_id},
                    score=quality,
                    semantic_score=quality,
                    lexical_score=0.0,
                    document_id=document_id,
                    document=document_id,
                )
                for document_id in docs
            ],
            "diagnostics": {
                "retrievalPlan": {"retrievalMode": "hybrid"},
                "retrievalStrategy": {
                    "phase": "phase0_diagnostics",
                    "complexityClass": "update_sensitive",
                    "retryPolicy": {"mode": "diagnostics_only", "maxRetries": 0, "allowedActions": ["broaden_query_terms", "read_parent_section", "abstain"]},
                },
                "retrievalQuality": {
                    "label": "medium",
                    "score": quality,
                    "weakSignals": [],
                    "correctiveActionCandidate": action,
                },
                "answerabilityRerank": {
                    "applied": False,
                    "label": "medium" if answerability >= 0.45 else "low",
                    "score": answerability,
                    "weakSignals": list(
                        self.answerability_weak_signals
                        if self.answerability_weak_signals is not None
                        else ["low_query_term_coverage"] if not self.retry_candidate else []
                    ),
                },
                "correctiveRetrieval": {
                    "applied": False,
                    "policy": "diagnostics_only",
                    "maxRetries": 0,
                    "retryCandidate": bool(action != "none"),
                    "candidateAction": action,
                    "triggers": [],
                },
                "artifactHealth": {"label": "high", "score": 1.0},
                "candidateSources": [],
                "rerankSignals": {},
                "memoryRoute": {},
                "memoryPrefilter": {},
                "paperMemoryPrefilter": {},
            },
        }


def _eval_result(document_id, *, parent_id="", source_type="paper", score=0.2, semantic_score=0.0, lexical_score=0.0, document="noise", title="Result"):
    return _result(
        metadata={
            "title": title,
            "source_type": source_type,
            "resolved_parent_id": parent_id,
            "resolved_parent_label": parent_id,
        },
        score=score,
        semantic_score=semantic_score,
        lexical_score=lexical_score,
        document_id=document_id,
        document=document,
    )


def _write_answerability_eval_csv(tmp_path, row):
    path = tmp_path / "answerability.csv"
    headers = [
        "query_id",
        "query",
        "source",
        "scenario",
        "risk_tags",
        "top_k",
        "expected_relevant_document_ids",
        "expected_relevant_parent_ids",
        "expected_top1_document_id",
        "required_parent_ids",
        "required_source_types",
        "protected_document_ids",
        "banned_document_ids",
        "expected_behavior",
        "allow_rank_change",
        "notes",
    ]
    values = {key: "" for key in headers}
    values.update(row)
    path.write_text(
        ",".join(headers) + "\n" + ",".join(str(values[key]) for key in headers) + "\n",
        encoding="utf-8",
    )
    return path


def _write_answerability_eval_rows_csv(tmp_path, rows):
    path = tmp_path / "answerability.csv"
    headers = [
        "query_id",
        "query",
        "source",
        "scenario",
        "risk_tags",
        "top_k",
        "expected_relevant_document_ids",
        "expected_relevant_parent_ids",
        "expected_top1_document_id",
        "required_parent_ids",
        "required_source_types",
        "protected_document_ids",
        "banned_document_ids",
        "expected_behavior",
        "allow_rank_change",
        "notes",
    ]
    lines = [",".join(headers)]
    for row in rows:
        values = {key: "" for key in headers}
        values.update(row)
        lines.append(",".join(str(values[key]) for key in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_corrective_eval_rows_csv(tmp_path, rows):
    path = tmp_path / "corrective.csv"
    headers = [
        "query",
        "source",
        "expected_complexity_class",
        "expected_retry_candidate",
        "expected_candidate_action",
        "scenario",
        "notes",
    ]
    lines = [",".join(headers)]
    for row in rows:
        values = {key: "" for key in headers}
        values.update(row)
        lines.append(",".join(str(values[key]) for key in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rank_for_document(results, document_id):
    for result in results:
        if result["documentId"] == document_id:
            return int(result["rerankedRank"])
    raise AssertionError(f"missing document {document_id}")


def test_build_rag_corrective_report_is_read_only_and_reuses_phase0_diagnostics():
    searcher = _Phase0DiagnosticSearcher()

    payload = build_rag_corrective_report(
        searcher,
        query="latest RankRAG benchmark update",
        top_k=3,
        source_type="paper",
        retrieval_mode="hybrid",
        alpha=0.65,
    )

    assert searcher.calls == [
        {
            "query": "latest RankRAG benchmark update",
            "top_k": 3,
            "source_type": "paper",
            "retrieval_mode": "hybrid",
            "alpha": 0.65,
            "expand_parent_context": False,
        }
    ]
    assert payload["schema"] == "knowledge-hub.rag.corrective-report.result.v1"
    assert payload["status"] == "warn"
    assert payload["readOnly"] is True
    assert payload["actionsApplied"] == []
    assert payload["retrievalStrategy"]["phase"] == "phase0_diagnostics"
    assert payload["retrievalQuality"]["correctiveActionCandidate"] == "broaden_search"
    assert payload["correctiveRetrieval"]["applied"] is False
    assert payload["correctiveRetrieval"]["retryCandidate"] is True
    assert payload["suggestedActions"] == [
        {
            "actionType": "broaden_search",
            "mode": "suggestion_only",
            "description": "Retry with broader query terms or a wider source mix.",
            "allowedByPolicy": True,
        }
    ]
    assert payload["resultsPreview"][0]["title"] == "RankRAG"
    assert payload["resultsPreview"][0]["documentId"] == "doc-rankrag"
    assert payload["warnings"] == []

    validation = validate_payload(payload, "knowledge-hub.rag.corrective-report.result.v1", strict=True)
    assert validation.ok, validation.errors


def test_build_rag_corrective_eval_report_scores_expected_rows(tmp_path):
    queries = tmp_path / "queries.csv"
    queries.write_text(
        "\n".join(
            [
                "query,source,expected_complexity_class,expected_retry_candidate,expected_candidate_action,scenario,notes",
                "latest RankRAG benchmark update,paper,update_sensitive,true,broaden_search,temporal_no_result,test",
            ]
        ),
        encoding="utf-8",
    )

    payload = build_rag_corrective_eval_report(
        _Phase0DiagnosticSearcher(),
        queries_path=queries,
        top_k=2,
    )

    assert payload["schema"] == "knowledge-hub.rag.corrective-eval.report.v1"
    assert payload["status"] == "ok"
    assert payload["rowCount"] == 1
    assert payload["passCount"] == 1
    assert payload["metrics"]["passRate"] == 1.0
    assert payload["rows"][0]["passed"] is True
    validation = validate_payload(payload, payload["schema"], strict=True)
    assert validation.ok, validation.errors


def test_corrective_eval_aligns_weak_answerability_retry_candidates(tmp_path):
    rows = [
        {
            "query": "temporal no result",
            "source": "paper",
            "expected_complexity_class": "update_sensitive",
            "expected_retry_candidate": "true",
            "expected_candidate_action": "broaden_search",
            "scenario": "temporal_no_result",
            "notes": "weak answerability should retry",
        },
        {
            "query": "missing source",
            "source": "paper",
            "expected_complexity_class": "update_sensitive",
            "expected_retry_candidate": "true",
            "expected_candidate_action": "broaden_search",
            "scenario": "no_result",
            "notes": "weak answerability should retry",
        },
        {
            "query": "global sensemaking",
            "source": "vault",
            "expected_complexity_class": "global_sensemaking",
            "expected_retry_candidate": "true",
            "expected_candidate_action": "graph_or_hierarchy_probe",
            "scenario": "global_sensemaking",
            "notes": "weak global evidence should probe graph",
        },
        {
            "query": "exact arxiv id 9999.99999",
            "source": "paper",
            "expected_complexity_class": "exact_lookup",
            "expected_retry_candidate": "true",
            "expected_candidate_action": "source_scope_rescue",
            "scenario": "missing_exact_scope",
            "notes": "missing exact scope should rescue source",
        },
        {
            "query": "web today assertion",
            "source": "web",
            "expected_complexity_class": "update_sensitive",
            "expected_retry_candidate": "true",
            "expected_candidate_action": "broaden_search",
            "scenario": "temporal_abstention",
            "notes": "temporal abstention should not stay exact lookup",
        },
    ]
    payload = build_rag_corrective_eval_report(
        _WeakAnswerabilityNoRetrySearcher(
            {
                "temporal no result": {"complexityClass": "update_sensitive"},
                "missing source": {"complexityClass": "update_sensitive"},
                "global sensemaking": {"complexityClass": "global_sensemaking"},
                "exact arxiv id 9999.99999": {"complexityClass": "exact_lookup"},
                "web today assertion": {"complexityClass": "exact_lookup", "weakSignals": ["low_query_term_coverage"]},
            }
        ),
        queries_path=_write_corrective_eval_rows_csv(tmp_path, rows),
        top_k=2,
    )

    assert payload["status"] == "ok"
    assert payload["passCount"] == 5
    assert payload["metrics"]["passRate"] == 1.0
    for row in payload["rows"]:
        assert row["observed"]["retryCandidate"] is True
        assert row["observed"]["candidateAction"] == row["expected"]["candidateAction"]
        assert row["passed"] is True


def test_corrective_eval_does_not_retry_suppressed_weak_answerability_rows(tmp_path):
    rows = [
        {
            "query": "local lookup",
            "source": "vault",
            "expected_complexity_class": "local_lookup",
            "expected_retry_candidate": "false",
            "expected_candidate_action": "none",
            "scenario": "local_lookup",
            "notes": "local lookup stays direct",
        },
        {
            "query": "temporal vault history",
            "source": "vault",
            "expected_complexity_class": "update_sensitive",
            "expected_retry_candidate": "false",
            "expected_candidate_action": "none",
            "scenario": "temporal_vault",
            "notes": "known vault temporal history stays direct",
        },
        {
            "query": "mixed howto",
            "source": "all",
            "expected_complexity_class": "procedural_lookup",
            "expected_retry_candidate": "false",
            "expected_candidate_action": "none",
            "scenario": "mixed_howto",
            "notes": "procedural planning stays direct",
        },
    ]
    payload = build_rag_corrective_eval_report(
        _WeakAnswerabilityNoRetrySearcher(
            {
                "local lookup": {"complexityClass": "local_explainer"},
                "temporal vault history": {"complexityClass": "update_sensitive"},
                "mixed howto": {"complexityClass": "discovery"},
            }
        ),
        queries_path=_write_corrective_eval_rows_csv(tmp_path, rows),
        top_k=2,
    )

    assert payload["status"] == "ok"
    assert payload["passCount"] == 3
    for row in payload["rows"]:
        assert row["observed"]["retryCandidate"] is False
        assert row["observed"]["candidateAction"] == "none"
        assert row["passed"] is True
    by_scenario = {row["scenario"]: row for row in payload["rows"]}
    assert by_scenario["local_lookup"]["observed"]["complexityClass"] == "local_lookup"
    assert by_scenario["mixed_howto"]["observed"]["complexityClass"] == "procedural_lookup"


def test_build_rag_adaptive_plan_maps_diagnostics_without_applying_actions():
    payload = build_rag_adaptive_plan(
        _Phase0DiagnosticSearcher(),
        query="latest RankRAG benchmark update",
        source_type="paper",
    )

    assert payload["schema"] == "knowledge-hub.rag.adaptive-plan.result.v1"
    assert payload["readOnly"] is True
    assert payload["labsOnly"] is True
    assert payload["actionsApplied"] == []
    assert payload["plan"]["route"] == "adaptive_retrieval"
    assert "suggest_broaden_search" in payload["plan"]["steps"]
    validation = validate_payload(payload, payload["schema"], strict=True)
    assert validation.ok, validation.errors


def test_build_rag_corrective_run_executes_retry_only_when_opted_in():
    searcher = _Phase0DiagnosticSearcher()

    dry_run = build_rag_corrective_run(
        searcher,
        query="latest RankRAG benchmark update",
        source_type="paper",
        execute=False,
    )
    assert dry_run["status"] == "dry_run"
    assert dry_run["actionsApplied"] == []
    assert len(searcher.calls) == 1

    applied = build_rag_corrective_run(
        searcher,
        query="latest RankRAG benchmark update",
        source_type="paper",
        execute=True,
    )
    assert applied["schema"] == "knowledge-hub.rag.corrective-run.result.v1"
    assert applied["status"] == "applied"
    assert applied["readOnly"] is True
    assert applied["writeFree"] is True
    assert applied["actionsApplied"][0]["actionType"] == "broaden_search"
    assert applied["actionsApplied"][0]["mode"] == "retrieval_only"
    assert len(searcher.calls) == 3
    validation = validate_payload(applied, applied["schema"], strict=True)
    assert validation.ok, validation.errors


def test_corrective_run_review_does_not_count_expansion_as_improvement_when_quality_regresses():
    payload = build_rag_corrective_run(
        _ExecutionReviewSearcher(retry_quality=0.49, retry_answerability=0.19),
        query="latest RAG benchmark update",
        source_type="paper",
        execute=True,
    )

    review = payload["retryExecutionReview"]
    assert payload["labsOnly"] is True
    assert payload["runtimeApplied"] is False
    assert payload["defaultRuntimeApplied"] is False
    assert review["resultCountExpanded"] is True
    assert review["qualityDelta"] < 0
    assert review["answerabilityDelta"] < 0
    assert review["improved"] is False
    assert review["noHarm"] is False
    assert review["regressed"] is True
    assert "result_count_expanded_observation_only" in review["decisionReasons"]
    assert "quality_regression" in review["decisionReasons"]


def test_corrective_run_skips_missing_source_negative_broaden_search_execution():
    searcher = _ExecutionReviewSearcher(retry_quality=0.49, retry_answerability=0.19)
    payload = build_rag_corrective_run(
        searcher,
        query="존재하지 않는 internal project codename의 최신 논문 근거를 찾아줘",
        source_type="paper",
        execute=True,
    )

    review = payload["retryExecutionReview"]
    assert payload["status"] == "skipped"
    assert payload["actionsApplied"] == []
    assert len(searcher.calls) == 1
    assert review["retryCandidate"] is True
    assert review["candidateAction"] == "broaden_search"
    assert review["executionEligible"] is False
    assert review["executionSkippedReason"] == "missing_source_negative"
    assert review["applied"] is False
    assert review["regressed"] is False
    assert "missing_source_negative" in review["decisionReasons"]


def test_corrective_run_review_marks_no_harm_when_quality_and_answerability_improve():
    payload = build_rag_corrective_run(
        _ExecutionReviewSearcher(retry_quality=0.62, retry_answerability=0.31),
        query="latest RAG benchmark update",
        source_type="paper",
        execute=True,
    )

    review = payload["retryExecutionReview"]
    assert review["resultCountExpanded"] is True
    assert review["qualityDelta"] > 0
    assert review["answerabilityDelta"] > 0
    assert review["top3Overlap"] == 2
    assert review["improved"] is True
    assert review["noHarm"] is True
    assert review["regressed"] is False


def test_corrective_run_skips_unresolved_exact_source_scope_rescue_execution():
    searcher = _ExecutionReviewSearcher(action="source_scope_rescue", retry_quality=0.49, retry_answerability=0.19)
    payload = build_rag_corrective_run(
        searcher,
        query="exact arxiv id 9999.99999",
        source_type="paper",
        execute=True,
    )

    review = payload["retryExecutionReview"]
    assert payload["status"] == "skipped"
    assert payload["actionsApplied"] == []
    assert len(searcher.calls) == 1
    assert review["exactScopeRequired"] is True
    assert review["executionEligible"] is False
    assert review["executionSkippedReason"] == "exact_source_unresolved"
    assert review["exactSourceResolved"] is False
    assert review["improved"] is False
    assert review["noHarm"] is False
    assert review["regressed"] is False
    assert "exact_source_unresolved" in review["decisionReasons"]


def test_corrective_run_review_keeps_strict_exact_scope_gate_when_source_resolves():
    class _ResolvingSqlite:
        def get_paper(self, paper_id):  # noqa: ANN001
            return {"paper_id": paper_id} if str(paper_id) == "1234.56789" else None

    searcher = _ExecutionReviewSearcher(action="source_scope_rescue", retry_quality=0.49, retry_answerability=0.19)
    searcher.sqlite_db = _ResolvingSqlite()
    payload = build_rag_corrective_run(
        searcher,
        query="exact arxiv id 1234.56789",
        source_type="paper",
        execute=True,
    )

    review = payload["retryExecutionReview"]
    assert payload["status"] == "applied"
    assert len(payload["actionsApplied"]) == 1
    assert review["executionEligible"] is True
    assert review["exactSourceResolved"] is True
    assert review["exactScopeRequired"] is True
    assert review["exactScopeCoverageAfter"]["present"] == 0
    assert review["regressed"] is True
    assert "exact_scope_not_improved" in review["decisionReasons"]


def test_corrective_run_query_only_marks_temporal_web_context_required():
    payload = build_rag_corrective_run(
        _ExecutionReviewSearcher(retry_candidate=False, source_type="web"),
        query="web source today RAG paper count",
        source_type="web",
        execute=True,
    )

    review = payload["retryExecutionReview"]
    assert payload["status"] == "skipped"
    assert payload["actionsApplied"] == []
    assert review["contextRequired"] is True
    assert review["applied"] is False
    assert "eval_context_required_for_retry_decision" in review["decisionReasons"]


def test_corrective_execution_review_uses_eval_context_for_temporal_abstention(tmp_path):
    queries = _write_corrective_eval_rows_csv(
        tmp_path,
        [
            {
                "query": "web source today RAG paper count",
                "source": "web",
                "expected_complexity_class": "update_sensitive",
                "expected_retry_candidate": "true",
                "expected_candidate_action": "broaden_search",
                "scenario": "temporal_abstention",
                "notes": "eval context should align candidate",
            }
        ],
    )
    payload = build_rag_corrective_execution_review(
        _ExecutionReviewSearcher(retry_candidate=False, source_type="web", retry_quality=0.62, retry_answerability=0.31),
        queries_path=queries,
        execute=True,
    )

    review = payload["rows"][0]["retryExecutionReview"]
    assert payload["schema"] == "knowledge-hub.rag.corrective-execution-review.result.v1"
    assert payload["labsOnly"] is True
    assert payload["runtimeApplied"] is False
    assert payload["defaultRuntimeApplied"] is False
    assert payload["summary"]["retryCandidateCount"] == 1
    assert payload["summary"]["retryAppliedCount"] == 1
    assert review["candidateAction"] == "broaden_search"
    assert review["contextRequired"] is False
    assert review["applied"] is True
    validation = validate_payload(payload, payload["schema"], strict=True)
    assert validation.ok, validation.errors


def test_corrective_execution_review_skips_safety_rows_outside_no_harm_denominator(tmp_path):
    queries = _write_corrective_eval_rows_csv(
        tmp_path,
        [
            {
                "query": "최근 RAG benchmark update에서 새로 강조된 metric은 무엇인가?",
                "source": "paper",
                "expected_complexity_class": "update_sensitive",
                "expected_retry_candidate": "true",
                "expected_candidate_action": "broaden_search",
                "scenario": "temporal_no_result",
                "notes": "positive retry",
            },
            {
                "query": "존재하지 않는 internal project codename의 최신 논문 근거를 찾아줘",
                "source": "paper",
                "expected_complexity_class": "update_sensitive",
                "expected_retry_candidate": "true",
                "expected_candidate_action": "broaden_search",
                "scenario": "no_result",
                "notes": "negative missing source",
            },
            {
                "query": "내 vault 전체에서 RAG 구조 변화 흐름을 묶어서 설명해줘",
                "source": "vault",
                "expected_complexity_class": "global_sensemaking",
                "expected_retry_candidate": "true",
                "expected_candidate_action": "graph_or_hierarchy_probe",
                "scenario": "global_sensemaking",
                "notes": "positive graph retry",
            },
            {
                "query": "특정 논문 하나의 exact arxiv id 9999.99999를 요약해줘",
                "source": "paper",
                "expected_complexity_class": "exact_lookup",
                "expected_retry_candidate": "true",
                "expected_candidate_action": "source_scope_rescue",
                "scenario": "missing_exact_scope",
                "notes": "fake exact scope",
            },
            {
                "query": "web source만으로 오늘 발표된 RAG 논문 수치를 단정할 수 있나?",
                "source": "web",
                "expected_complexity_class": "update_sensitive",
                "expected_retry_candidate": "true",
                "expected_candidate_action": "broaden_search",
                "scenario": "temporal_abstention",
                "notes": "positive web retry",
            },
        ],
    )
    payload = build_rag_corrective_execution_review(
        _ExecutionReviewSearcher(
            retry_candidate=False,
            retry_quality=0.62,
            retry_answerability=0.31,
            answerability_weak_signals=["low_query_term_coverage", "no_high_confidence_support"],
        ),
        queries_path=queries,
        execute=True,
    )

    by_row = {row["row"]: row["retryExecutionReview"] for row in payload["rows"]}
    assert payload["status"] == "ok"
    assert payload["summary"]["retryCandidateCount"] == 5
    assert payload["summary"]["retryAppliedCount"] == 3
    assert payload["summary"]["retrySkippedCount"] == 2
    assert payload["summary"]["retrievalNoHarmCount"] == 3
    assert payload["summary"]["retrievalRegressedCount"] == 0
    assert payload["summary"]["noHarmRate"] == 1.0
    assert by_row[1]["applied"] is True and by_row[1]["noHarm"] is True
    assert by_row[2]["executionEligible"] is False
    assert by_row[2]["executionSkippedReason"] == "missing_source_negative"
    assert by_row[3]["applied"] is True and by_row[3]["noHarm"] is True
    assert by_row[4]["executionEligible"] is False
    assert by_row[4]["executionSkippedReason"] == "exact_source_unresolved"
    assert by_row[5]["applied"] is True and by_row[5]["noHarm"] is True
    validation = validate_payload(payload, payload["schema"], strict=True)
    assert validation.ok, validation.errors


def test_build_rag_answerability_rerank_orders_by_answerability():
    payload = build_rag_answerability_rerank(
        _AnswerabilitySearcher(),
        query="RankRAG benchmark update",
        source_type="paper",
    )

    assert payload["schema"] == "knowledge-hub.rag.answerability-rerank.result.v1"
    assert payload["readOnly"] is True
    assert payload["writeFree"] is True
    assert payload["labsOnly"] is True
    assert payload["runtimeApplied"] is False
    assert payload["applied"] is True
    assert payload["rerankedResults"][0]["documentId"] == "doc-answerable"
    assert payload["changedRankCount"] == 2
    validation = validate_payload(payload, payload["schema"], strict=True)
    assert validation.ok, validation.errors


def test_answerability_rerank_eval_regresses_when_changed_rank_hurts_gold(tmp_path):
    path = _write_answerability_eval_csv(
        tmp_path,
        {
            "query_id": "regress",
            "query": "target",
            "source": "paper",
            "top_k": "2",
            "expected_relevant_document_ids": "doc-good",
        },
    )
    payload = build_rag_answerability_rerank_eval_report(
        _StaticAnswerabilityEvalSearcher(
            [
                _eval_result("doc-good", score=0.1, document="unrelated", title="Good"),
                _eval_result("doc-bad", score=0.9, document="target", title="Bad"),
            ]
        ),
        queries_path=path,
        generated_at="2026-04-29T00:00:00+00:00",
    )

    assert payload["schema"] == "knowledge-hub.rag.answerability-rerank-eval.report.v1"
    assert payload["readOnly"] is True
    assert payload["writeFree"] is True
    assert payload["labsOnly"] is True
    assert payload["runtimeApplied"] is False
    assert payload["summary"]["changedRowCount"] == 1
    assert payload["rows"][0]["verdict"] == "regressed"
    assert payload["rows"][0]["metrics"]["top1RelevantBefore"] is True
    assert payload["rows"][0]["metrics"]["top1RelevantAfter"] is False
    validation = validate_payload(payload, payload["schema"], strict=True)
    assert validation.ok, validation.errors


def test_answerability_rerank_eval_improves_when_gold_moves_into_top3(tmp_path):
    path = _write_answerability_eval_csv(
        tmp_path,
        {
            "query_id": "improve",
            "query": "target",
            "source": "paper",
            "top_k": "4",
            "expected_relevant_document_ids": "doc-good",
        },
    )
    payload = build_rag_answerability_rerank_eval_report(
        _StaticAnswerabilityEvalSearcher(
            [
                _eval_result("doc-a", score=0.1, document="noise a"),
                _eval_result("doc-b", score=0.1, document="noise b"),
                _eval_result("doc-c", score=0.1, document="noise c"),
                _eval_result("doc-good", score=0.9, document="target", title="Good"),
            ]
        ),
        queries_path=path,
    )

    assert payload["rows"][0]["verdict"] == "improved"
    assert payload["rows"][0]["metrics"]["top3RelevantBefore"] is False
    assert payload["rows"][0]["metrics"]["top3RelevantAfter"] is True


def test_answerability_rerank_eval_neutral_when_rank_changes_without_gold_delta(tmp_path):
    path = _write_answerability_eval_csv(
        tmp_path,
        {
            "query_id": "neutral",
            "query": "target",
            "source": "paper",
            "top_k": "3",
            "expected_relevant_document_ids": "doc-good",
        },
    )
    payload = build_rag_answerability_rerank_eval_report(
        _StaticAnswerabilityEvalSearcher(
            [
                _eval_result("doc-good", score=0.9, document="target", title="Good"),
                _eval_result("doc-a", score=0.2, document="noise a"),
                _eval_result("doc-b", score=0.2, document="target", title="Other"),
            ]
        ),
        queries_path=path,
    )

    assert payload["summary"]["changedRowCount"] == 1
    assert payload["rows"][0]["verdict"] == "neutral"
    assert payload["rows"][0]["metrics"]["firstRelevantRankBefore"] == 1
    assert payload["rows"][0]["metrics"]["firstRelevantRankAfter"] == 1


def test_answerability_rerank_eval_marks_invalid_gold_when_expected_ids_are_absent(tmp_path):
    path = _write_answerability_eval_csv(
        tmp_path,
        {
            "query_id": "invalid",
            "query": "target",
            "source": "paper",
            "top_k": "2",
            "expected_relevant_document_ids": "doc-missing",
        },
    )
    payload = build_rag_answerability_rerank_eval_report(
        _StaticAnswerabilityEvalSearcher(
            [
                _eval_result("doc-a", score=0.4, document="target"),
                _eval_result("doc-b", score=0.3, document="noise"),
            ]
        ),
        queries_path=path,
    )

    assert payload["rows"][0]["verdict"] == "invalid_gold"
    assert "expected_gold_not_found_in_results" in payload["rows"][0]["blockers"]
    assert payload["summary"]["invalidGoldCount"] == 1


def test_answerability_rerank_eval_regresses_on_required_parent_or_source_loss(tmp_path):
    path = _write_answerability_eval_csv(
        tmp_path,
        {
            "query_id": "coverage",
            "query": "target",
            "source": "all",
            "top_k": "4",
            "required_parent_ids": "parent-required",
            "required_source_types": "web",
        },
    )
    payload = build_rag_answerability_rerank_eval_report(
        _StaticAnswerabilityEvalSearcher(
            [
                _eval_result("doc-required", parent_id="parent-required", source_type="web", score=0.1, document="noise"),
                _eval_result("doc-a", source_type="paper", score=0.9, document="target a"),
                _eval_result("doc-b", source_type="paper", score=0.9, document="target b"),
                _eval_result("doc-c", source_type="paper", score=0.9, document="target c"),
            ]
        ),
        queries_path=path,
    )

    assert payload["rows"][0]["verdict"] == "regressed"
    assert payload["rows"][0]["metrics"]["parentCoverageRegression"] is True
    assert payload["rows"][0]["metrics"]["sourceCoverageRegression"] is True


def test_answerability_rerank_eval_regresses_when_protected_document_drops(tmp_path):
    path = _write_answerability_eval_csv(
        tmp_path,
        {
            "query_id": "protected",
            "query": "target",
            "source": "paper",
            "top_k": "2",
            "protected_document_ids": "doc-protected",
        },
    )
    payload = build_rag_answerability_rerank_eval_report(
        _StaticAnswerabilityEvalSearcher(
            [
                _eval_result("doc-protected", score=0.1, document="noise", title="Protected"),
                _eval_result("doc-other", score=0.9, document="target", title="Other"),
            ]
        ),
        queries_path=path,
    )

    assert payload["rows"][0]["verdict"] == "regressed"
    assert payload["rows"][0]["metrics"]["protectedDocumentDropped"] is True


def test_answerability_rerank_eval_regresses_when_banned_document_is_promoted(tmp_path):
    path = _write_answerability_eval_csv(
        tmp_path,
        {
            "query_id": "banned",
            "query": "target",
            "source": "paper",
            "top_k": "2",
            "banned_document_ids": "doc-banned",
        },
    )
    payload = build_rag_answerability_rerank_eval_report(
        _StaticAnswerabilityEvalSearcher(
            [
                _eval_result("doc-ok", score=0.1, document="noise", title="Ok"),
                _eval_result("doc-banned", score=0.9, document="target", title="Banned"),
            ]
        ),
        queries_path=path,
    )

    assert payload["rows"][0]["verdict"] == "regressed"
    assert payload["rows"][0]["metrics"]["bannedDocumentPromoted"] is True


def test_answerability_rerank_eval_negative_guard_blocks_banned_document_promotion(tmp_path):
    path = _write_answerability_eval_rows_csv(
        tmp_path,
        [
            {
                "query_id": "arr-009",
                "query": "nonexistent internal project codename latest paper evidence",
                "source": "paper",
                "scenario": "missing_source_negative",
                "risk_tags": "paper;negative",
                "top_k": "5",
                "required_source_types": "paper",
                "banned_document_ids": "doc-banned-codename",
                "expected_behavior": "no_harm_only",
                "allow_rank_change": "true",
            },
            {
                "query_id": "arr-011",
                "query": "summarize exact arxiv id 9999.99999",
                "source": "paper",
                "scenario": "missing_exact_scope",
                "risk_tags": "paper;exact;negative",
                "top_k": "5",
                "required_source_types": "paper",
                "banned_document_ids": "doc-banned-exact",
                "expected_behavior": "no_harm_only",
                "allow_rank_change": "true",
            },
        ],
    )
    payload = build_rag_answerability_rerank_eval_report(
        _StaticAnswerabilityEvalSearcher(
            [
                _eval_result("doc-ok-a", score=0.3, document="noise a", title="Ok A"),
                _eval_result("doc-ok-b", score=0.3, document="noise b", title="Ok B"),
                _eval_result("doc-ok-c", score=0.3, document="noise c", title="Ok C"),
                _eval_result("doc-banned-codename", score=0.9, document="nonexistent codename latest", title="Banned Codename"),
                _eval_result("doc-banned-exact", score=0.95, document="9999.99999 exact arxiv", title="Banned Exact"),
            ]
        ),
        queries_path=path,
    )

    rows = {row["queryId"]: row for row in payload["rows"]}
    assert payload["runtimeApplied"] is False
    assert payload["summary"]["invalidGoldCount"] == 0
    assert payload["summary"]["regressedCount"] == 0
    assert payload["summary"]["bannedPromotionCount"] == 0
    assert rows["arr-009"]["verdict"] == "neutral"
    assert rows["arr-009"]["metrics"]["bannedDocumentPromoted"] is False
    assert rows["arr-009"]["rerankGuard"]["demotedDocumentIds"] == ["doc-banned-codename"]
    assert _rank_for_document(rows["arr-009"]["reranked"]["results"], "doc-banned-codename") >= 4
    assert rows["arr-011"]["verdict"] == "neutral"
    assert rows["arr-011"]["metrics"]["bannedDocumentPromoted"] is False
    assert rows["arr-011"]["rerankGuard"]["demotedDocumentIds"] == ["doc-banned-exact"]
    assert _rank_for_document(rows["arr-011"]["reranked"]["results"], "doc-banned-exact") >= 5


def test_build_rag_graph_global_plan_stays_planning_only():
    payload = build_rag_graph_global_plan(
        _GlobalDiagnosticSearcher(),
        query="vault 전체에서 RAG 흐름을 묶어줘",
        source_type="vault",
    )

    assert payload["schema"] == "knowledge-hub.rag.graph-global-plan.result.v1"
    assert payload["readOnly"] is True
    assert payload["labsOnly"] is True
    assert payload["graphGlobalLane"]["candidate"] is True
    assert payload["graphGlobalLane"]["route"] == "graph_global"
    assert payload["actionsApplied"] == []
    validation = validate_payload(payload, payload["schema"], strict=True)
    assert validation.ok, validation.errors


def test_build_rag_corrective_report_warns_when_phase0_diagnostics_are_unavailable():
    searcher = _LegacySearchOnlySearcher()

    payload = build_rag_corrective_report(
        searcher,
        query="legacy RAG lookup",
        top_k=2,
        source_type="vault",
        retrieval_mode="keyword",
        alpha=0.5,
    )

    assert searcher.calls == [
        {
            "query": "legacy RAG lookup",
            "top_k": 2,
            "source_type": "vault",
            "retrieval_mode": "keyword",
            "alpha": 0.5,
        }
    ]
    assert payload["status"] == "ok"
    assert payload["readOnly"] is True
    assert payload["retrievalStrategy"] == {}
    assert payload["correctiveRetrieval"] == {}
    assert payload["suggestedActions"] == []
    assert payload["actionsApplied"] == []
    assert payload["warnings"] == ["searcher_did_not_expose_phase0_diagnostics"]
    assert payload["resultsPreview"][0]["title"] == "Legacy RAG Note"

    validation = validate_payload(payload, "knowledge-hub.rag.corrective-report.result.v1", strict=True)
    assert validation.ok, validation.errors
