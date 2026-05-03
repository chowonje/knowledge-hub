from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from knowledge_hub.interfaces.cli.commands.rag_labs_cmd import rag_labs_group


class _Config:
    def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
        _ = args
        return default


class _Factory:
    def __init__(self, searcher):
        self.searcher = searcher

    def get_searcher(self):
        return self.searcher


class _DiagnosticSearcher:
    def __init__(self, *, retry_candidate: bool = True, complexity_class: str = "update_sensitive"):
        self.retry_candidate = retry_candidate
        self.complexity_class = complexity_class
        self.calls = []

    def search_with_diagnostics(self, query, **kwargs):  # noqa: ANN001
        self.calls.append({"query": query, **kwargs})
        action = "graph_or_hierarchy_probe" if self.complexity_class == "global_sensemaking" else "broaden_search" if self.retry_candidate else "none"
        return {
            "results": [],
            "diagnostics": {
                "retrievalPlan": {
                    "complexityClass": self.complexity_class,
                    "retryPolicy": {
                        "mode": "diagnostics_only",
                        "maxRetries": 0,
                        "allowedActions": ["broaden_query_terms", "read_parent_section", "abstain"],
                    },
                },
                "retrievalStrategy": {
                    "complexityClass": self.complexity_class,
                    "retryPolicy": {
                        "mode": "diagnostics_only",
                        "maxRetries": 0,
                        "allowedActions": ["broaden_query_terms", "read_parent_section", "abstain"],
                    },
                    "phase": "phase0_diagnostics",
                },
                "retrievalQuality": {
                    "label": "low" if self.retry_candidate else "high",
                    "score": 0.1 if self.retry_candidate else 0.9,
                    "weakSignals": ["no_results"] if self.retry_candidate else [],
                    "correctiveActionCandidate": action,
                },
                "answerabilityRerank": {
                    "applied": False,
                    "label": "low" if self.retry_candidate else "high",
                    "score": 0.0 if self.retry_candidate else 0.8,
                    "weakSignals": ["no_evidence"] if self.retry_candidate else [],
                },
                "correctiveRetrieval": {
                    "applied": False,
                    "policy": "diagnostics_only",
                    "retryCandidate": self.retry_candidate,
                    "candidateAction": action,
                    "triggers": ["no_results"] if self.retry_candidate else [],
                },
                "artifactHealth": {"label": "low" if self.retry_candidate else "high", "score": 0.0 if self.retry_candidate else 1.0},
                "candidateSources": [],
                "rerankSignals": {},
                "memoryRoute": {},
                "memoryPrefilter": {},
                "paperMemoryPrefilter": {},
            },
        }


class _RerankSearcher:
    def search_with_diagnostics(self, query, **kwargs):  # noqa: ANN001
        _ = (query, kwargs)
        return {
            "results": [
                SimpleNamespace(
                    metadata={"title": "Generic Note", "source_type": "paper"},
                    score=0.9,
                    semantic_score=0.1,
                    lexical_score=0.1,
                    document_id="generic",
                    document="unrelated text",
                ),
                SimpleNamespace(
                    metadata={"title": "RankRAG benchmark", "source_type": "paper"},
                    score=0.45,
                    semantic_score=0.5,
                    lexical_score=0.5,
                    document_id="rankrag",
                    document="RankRAG benchmark update evidence",
                ),
            ],
            "diagnostics": {
                "retrievalStrategy": {"phase": "phase0_diagnostics", "complexityClass": "update_sensitive"},
                "answerabilityRerank": {"applied": False},
            },
        }


class _VisualSearcher:
    def __init__(self):
        self.calls = []

    def search_with_diagnostics(self, query, **kwargs):  # noqa: ANN001
        self.calls.append({"query": query, **kwargs})
        source = kwargs.get("source_type")
        if source == "paper":
            return {
                "results": [
                    SimpleNamespace(
                        metadata={"title": "Paper Evidence", "source_type": "paper"},
                        score=0.74,
                        semantic_score=0.8,
                        lexical_score=0.2,
                        distance=0.26,
                        document_id="doc-paper-evidence",
                        document="Paper evidence about RAG grounding.",
                    )
                ],
                "diagnostics": {"retrievalQuality": {"label": "medium", "score": 0.74}},
            }
        if source == "web":
            return {
                "results": [
                    SimpleNamespace(
                        metadata={"title": "Web Evidence", "source_type": "web"},
                        score=0.52,
                        semantic_score=0.6,
                        lexical_score=0.1,
                        distance=0.48,
                        document_id="doc-web-evidence",
                        document="Web evidence about retrieval workflows.",
                    )
                ],
                "diagnostics": {"retrievalQuality": {"label": "low", "score": 0.52}},
            }
        if source == "concept":
            return {
                "results": [
                    SimpleNamespace(
                        metadata={"title": "Concept Evidence", "source_type": "concept"},
                        score=0.49,
                        semantic_score=0.5,
                        lexical_score=0.2,
                        distance=0.51,
                        document_id="doc-concept-evidence",
                        document="Concept evidence.",
                    )
                ],
                "diagnostics": {"retrievalQuality": {"label": "low", "score": 0.49}},
            }
        return {
            "results": [
                SimpleNamespace(
                    metadata={"title": "RAG Note", "source_type": "vault", "resolved_parent_id": "note-1"},
                    score=0.82,
                    semantic_score=0.9,
                    lexical_score=0.4,
                    distance=0.18,
                    document_id="doc-vault-1",
                    document="RAG systems retrieve local evidence before answer generation.",
                ),
                SimpleNamespace(
                    metadata={"title": "Attention Paper", "source_type": "paper"},
                    score=0.61,
                    semantic_score=0.7,
                    lexical_score=0.5,
                    distance=0.39,
                    document_id="doc-paper-1",
                    document="Transformer attention is adjacent background but not the top RAG evidence.",
                ),
            ],
            "diagnostics": {
                "retrievalStrategy": {"phase": "phase0_diagnostics"},
                "retrievalQuality": {"label": "medium", "score": 0.72},
                "answerabilityRerank": {"applied": False},
                "correctiveRetrieval": {"retryCandidate": False},
                "artifactHealth": {"label": "ok"},
            },
        }


def _ctx(searcher):
    return {"khub": SimpleNamespace(factory=_Factory(searcher), config=_Config())}


def test_rag_labs_corrective_report_json_is_read_only_and_schema_backed():
    searcher = _DiagnosticSearcher(retry_candidate=True)
    runner = CliRunner()

    result = runner.invoke(
        rag_labs_group,
        ["corrective-report", "latest RAG benchmark update", "--source", "paper", "--top-k", "2", "--json"],
        obj=_ctx(searcher),
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.rag.corrective-report.result.v1"
    assert payload["status"] == "warn"
    assert payload["readOnly"] is True
    assert payload["actionsApplied"] == []
    assert payload["sourceType"] == "paper"
    assert payload["topK"] == 2
    assert payload["retrievalStrategy"]["complexityClass"] == "update_sensitive"
    assert payload["correctiveRetrieval"]["retryCandidate"] is True
    assert payload["suggestedActions"][0]["actionType"] == "broaden_search"
    assert searcher.calls[0]["source_type"] == "paper"


def test_rag_labs_visualize_json_builds_threejs_graph_payload():
    searcher = _VisualSearcher()
    runner = CliRunner()

    result = runner.invoke(
        rag_labs_group,
        ["visualize", "how does my RAG use evidence", "--source", "vault", "--top-k", "2", "--json"],
        obj=_ctx(searcher),
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.rag.visualization.result.v1"
    assert payload["readOnly"] is True
    assert payload["labsOnly"] is True
    assert payload["runtimeApplied"] is False
    assert payload["artifactWritten"] is False
    assert payload["resultCount"] == 2
    assert payload["summary"]["nodeCount"] == 3
    assert payload["summary"]["edgeCount"] == 2
    assert payload["graph"]["nodes"][0]["kind"] == "query"
    assert payload["graph"]["nodes"][1]["sourceType"] == "vault"
    assert payload["graph"]["edges"][0]["source"] == "query"
    assert searcher.calls[0]["source_type"] == "vault"


def test_rag_labs_visualize_writes_standalone_html(tmp_path):
    output = tmp_path / "rag-map.html"
    runner = CliRunner()

    result = runner.invoke(
        rag_labs_group,
        ["visualize", "RAG evidence map", "--output", str(output), "--json"],
        obj=_ctx(_VisualSearcher()),
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["artifactWritten"] is True
    assert payload["artifactPath"] == str(output)
    html = output.read_text(encoding="utf-8")
    assert "three.module.js" in html
    assert "knowledge-hub.rag.visualization.result.v1" in html
    assert "RAG evidence map" in html
    assert "Top Evidence" in html
    assert "scorebar" in html
    assert "labelSprite" in html


def test_rag_labs_visualize_all_merges_source_scoped_searches():
    searcher = _VisualSearcher()
    runner = CliRunner()

    result = runner.invoke(
        rag_labs_group,
        ["visualize", "RAG evidence map", "--source", "all", "--top-k", "6", "--json"],
        obj=_ctx(searcher),
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    result_sources = {node["sourceType"] for node in payload["graph"]["nodes"] if node["kind"] == "result"}
    assert {"vault", "paper", "web", "concept"} <= result_sources
    assert [call["source_type"] for call in searcher.calls] == ["vault", "paper", "web", "concept"]
    assert payload["retrievalStrategy"]["phase"] == "rag_visualization_all_source_merge"


def test_rag_labs_corrective_report_text_prints_summary():
    runner = CliRunner()

    result = runner.invoke(
        rag_labs_group,
        ["corrective-report", "latest RAG benchmark update", "--source", "paper"],
        obj=_ctx(_DiagnosticSearcher(retry_candidate=True)),
    )

    assert result.exit_code == 0
    assert "rag corrective-report" in result.output
    assert "retryCandidate=True" in result.output
    assert "broaden_search" in result.output


def test_rag_vnext_phase1_eval_seed_has_required_scenarios():
    path = Path("eval/knowledgeos/queries/rag_vnext_phase1_corrective_eval_queries_v1.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))

    assert len(rows) >= 10
    assert {"query", "source", "expected_complexity_class", "expected_retry_candidate", "expected_candidate_action", "scenario", "notes"} <= set(rows[0])
    scenarios = {row["scenario"] for row in rows}
    assert {"compare_coverage", "temporal_no_result", "global_sensemaking", "no_result"} <= scenarios


def test_rag_labs_eval_corrective_json_scores_seed_subset(tmp_path):
    path = tmp_path / "queries.csv"
    path.write_text(
        "\n".join(
            [
                "query,source,expected_complexity_class,expected_retry_candidate,expected_candidate_action,scenario,notes",
                "latest RAG benchmark update,paper,update_sensitive,true,broaden_search,temporal_no_result,test",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        rag_labs_group,
        ["eval-corrective", "--queries", str(path), "--json"],
        obj=_ctx(_DiagnosticSearcher(retry_candidate=True)),
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.rag.corrective-eval.report.v1"
    assert payload["status"] == "ok"
    assert payload["rowCount"] == 1
    assert payload["passCount"] == 1


def test_rag_labs_adaptive_plan_json_maps_complexity_to_route():
    runner = CliRunner()

    result = runner.invoke(
        rag_labs_group,
        ["adaptive-plan", "latest RAG benchmark update", "--source", "paper", "--json"],
        obj=_ctx(_DiagnosticSearcher(retry_candidate=True)),
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.rag.adaptive-plan.result.v1"
    assert payload["readOnly"] is True
    assert payload["labsOnly"] is True
    assert payload["actionsApplied"] == []
    assert payload["plan"]["route"] == "adaptive_retrieval"


def test_rag_labs_corrective_run_requires_execute_for_retry():
    searcher = _DiagnosticSearcher(retry_candidate=True)
    runner = CliRunner()

    dry_run = runner.invoke(
        rag_labs_group,
        ["corrective-run", "latest RAG benchmark update", "--source", "paper", "--json"],
        obj=_ctx(searcher),
    )
    assert dry_run.exit_code == 0
    dry_payload = json.loads(dry_run.output)
    assert dry_payload["status"] == "dry_run"
    assert dry_payload["actionsApplied"] == []
    assert len(searcher.calls) == 1

    applied = runner.invoke(
        rag_labs_group,
        ["corrective-run", "latest RAG benchmark update", "--source", "paper", "--execute", "--json"],
        obj=_ctx(searcher),
    )
    assert applied.exit_code == 0
    applied_payload = json.loads(applied.output)
    assert applied_payload["schema"] == "knowledge-hub.rag.corrective-run.result.v1"
    assert applied_payload["status"] == "applied"
    assert applied_payload["actionsApplied"][0]["actionType"] == "broaden_search"
    assert applied_payload["actionsApplied"][0]["mode"] == "retrieval_only"
    assert len(searcher.calls) == 3


def test_rag_labs_corrective_execution_review_json_is_labs_only(tmp_path):
    path = tmp_path / "queries.csv"
    path.write_text(
        "\n".join(
            [
                "query,source,expected_complexity_class,expected_retry_candidate,expected_candidate_action,scenario,notes",
                "latest RAG benchmark update,paper,update_sensitive,true,broaden_search,temporal_no_result,test",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        rag_labs_group,
        ["corrective-execution-review", "--queries", str(path), "--execute", "--json"],
        obj=_ctx(_DiagnosticSearcher(retry_candidate=True)),
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.rag.corrective-execution-review.result.v1"
    assert payload["readOnly"] is True
    assert payload["writeFree"] is True
    assert payload["labsOnly"] is True
    assert payload["runtimeApplied"] is False
    assert payload["defaultRuntimeApplied"] is False
    assert payload["summary"]["retryCandidateCount"] == 1
    assert payload["summary"]["retryAppliedCount"] == 1
    assert payload["rows"][0]["retryExecutionReview"]["labsOnly"] is True
    assert payload["rows"][0]["retryExecutionReview"]["runtimeApplied"] is False


def test_rag_labs_answerability_rerank_json_is_labs_only():
    runner = CliRunner()

    result = runner.invoke(
        rag_labs_group,
        ["answerability-rerank", "RankRAG benchmark update", "--source", "paper", "--json"],
        obj=_ctx(_RerankSearcher()),
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.rag.answerability-rerank.result.v1"
    assert payload["readOnly"] is True
    assert payload["writeFree"] is True
    assert payload["labsOnly"] is True
    assert payload["runtimeApplied"] is False
    assert payload["rerankedResults"][0]["documentId"] == "rankrag"


def test_rag_labs_eval_answerability_rerank_json_is_shadow_only(tmp_path):
    path = tmp_path / "answerability.csv"
    path.write_text(
        "\n".join(
            [
                "query_id,query,source,scenario,risk_tags,top_k,expected_relevant_document_ids,expected_relevant_parent_ids,expected_top1_document_id,required_parent_ids,required_source_types,protected_document_ids,banned_document_ids,expected_behavior,allow_rank_change,notes",
                "seed,RankRAG benchmark update,paper,answerable,paper,2,rankrag,,rankrag,,paper,,,improve,true,test",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        rag_labs_group,
        ["eval-answerability-rerank", "--queries", str(path), "--limit", "1", "--json"],
        obj=_ctx(_RerankSearcher()),
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.rag.answerability-rerank-eval.report.v1"
    assert payload["readOnly"] is True
    assert payload["writeFree"] is True
    assert payload["labsOnly"] is True
    assert payload["runtimeApplied"] is False
    assert payload["summary"]["rowCount"] == 1
    assert payload["rows"][0]["verdict"] == "improved"


def test_rag_labs_graph_global_plan_json_marks_candidate():
    runner = CliRunner()

    result = runner.invoke(
        rag_labs_group,
        ["graph-global-plan", "vault 전체에서 RAG 흐름을 묶어줘", "--source", "vault", "--json"],
        obj=_ctx(_DiagnosticSearcher(retry_candidate=True, complexity_class="global_sensemaking")),
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.rag.graph-global-plan.result.v1"
    assert payload["readOnly"] is True
    assert payload["labsOnly"] is True
    assert payload["graphGlobalLane"]["candidate"] is True
    assert payload["actionsApplied"] == []


def test_rag_labs_observe_loop_json_collects_daily_metrics(tmp_path):
    path = tmp_path / "queries.csv"
    path.write_text(
        "\n".join(
            [
                "query,source,expected_complexity_class,expected_retry_candidate,expected_candidate_action,scenario,notes",
                "latest RAG benchmark update,paper,update_sensitive,true,broaden_search,temporal_no_result,test",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        rag_labs_group,
        [
            "observe-loop",
            "--queries",
            str(path),
            "--limit",
            "1",
            "--retry-limit",
            "1",
            "--rerank-limit",
            "1",
            "--graph-limit",
            "1",
            "--json",
        ],
        obj=_ctx(_DiagnosticSearcher(retry_candidate=True)),
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.rag.vnext-observation.report.v1"
    assert payload["readOnly"] is True
    assert payload["writeFree"] is True
    assert payload["summary"]["rowCount"] == 1
    assert payload["summary"]["retryCandidateCount"] == 1
    assert payload["summary"]["retryAppliedCount"] == 0
    assert payload["correctiveRuns"][0]["status"] == "dry_run"
    assert payload["correctiveRuns"][0]["actionsApplied"] == []
    assert payload["promotionReadiness"]["status"] == "not_ready"
