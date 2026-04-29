from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from knowledge_hub.application.rag_observation_loop import build_rag_vnext_observation_report
from knowledge_hub.core.schema_validator import validate_payload


def _write_queries(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "query,source,expected_complexity_class,expected_retry_candidate,expected_candidate_action,scenario,notes",
                "latest RAG benchmark update,paper,update_sensitive,true,broaden_search,temporal_no_result,test",
            ]
        ),
        encoding="utf-8",
    )
    return path


class _ObservationSearcher:
    def __init__(self):
        self.calls = []

    def search_with_diagnostics(self, query, **kwargs):  # noqa: ANN001
        self.calls.append({"query": query, **kwargs})
        return {
            "results": [
                SimpleNamespace(
                    metadata={"title": "RAG Benchmark Update", "source_type": "paper"},
                    score=0.2,
                    semantic_score=0.2,
                    lexical_score=0.1,
                    document_id="doc-rag-update",
                    document="latest RAG benchmark update evidence",
                )
            ],
            "diagnostics": {
                "retrievalPlan": {"queryIntent": "paper_lookup", "retrievalMode": "hybrid"},
                "retrievalStrategy": {
                    "phase": "phase0_diagnostics",
                    "complexityClass": "update_sensitive",
                    "retrievalBudget": {"topK": kwargs.get("top_k", 5)},
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
                    "score": 0.2,
                    "weakSignals": ["thin_evidence"],
                },
                "correctiveRetrieval": {
                    "applied": False,
                    "policy": "diagnostics_only",
                    "retryCandidate": True,
                    "candidateAction": "broaden_search",
                    "triggers": ["low_top_score"],
                },
                "artifactHealth": {"label": "high", "score": 1.0},
                "candidateSources": [],
                "rerankSignals": {},
                "memoryRoute": {},
                "memoryPrefilter": {},
                "paperMemoryPrefilter": {},
            },
        }


def test_build_rag_vnext_observation_report_collects_loop_metrics(tmp_path: Path):
    queries = _write_queries(tmp_path / "queries.csv")
    searcher = _ObservationSearcher()

    payload = build_rag_vnext_observation_report(
        searcher,
        queries_path=queries,
        top_k=2,
        retry_limit=1,
        rerank_limit=1,
        graph_limit=1,
        observation_count=3,
        generated_at="2026-04-28T00:00:00+00:00",
    )

    assert payload["schema"] == "knowledge-hub.rag.vnext-observation.report.v1"
    assert payload["readOnly"] is True
    assert payload["writeFree"] is True
    assert payload["summary"]["rowCount"] == 1
    assert payload["summary"]["correctivePassRate"] == 1.0
    assert payload["summary"]["retryCandidateCount"] == 1
    assert payload["summary"]["retryAppliedCount"] == 0
    assert payload["correctiveRuns"][0]["status"] == "dry_run"
    assert payload["correctiveRuns"][0]["execute"] is False
    assert payload["correctiveRuns"][0]["actionsApplied"] == []
    assert payload["correctiveRuns"][0]["retryExecutionReview"]["applied"] is False
    assert payload["correctiveRuns"][0]["retryExecutionReview"]["labsOnly"] is True
    assert payload["summary"]["rerankReportCount"] == 1
    assert "answerabilityRerankShadowEval" in payload["summary"]
    assert payload["answerabilityRerankShadowEval"]["runtimeApplied"] is False
    assert payload["summary"]["graphReportCount"] == 1
    assert payload["promotionReadiness"]["status"] == "not_ready"
    assert "needs_at_least_7_observations" in payload["promotionReadiness"]["blockers"]
    assert len(searcher.calls) >= 4
    validation = validate_payload(payload, payload["schema"], strict=True)
    assert validation.ok, validation.errors
