from __future__ import annotations

import csv
from pathlib import Path

from knowledge_hub.ai.retrieval_eval import build_eval_report, compute_precision_metrics, read_eval_queries
from knowledge_hub.core.schema_validator import validate_payload


def test_read_eval_queries_uses_existing_fixture():
    queries = read_eval_queries(Path(__file__).resolve().parents[1] / "docs" / "eval_queries_ko_20.txt")

    assert len(queries) == 20
    assert queries[0]


def test_compute_precision_metrics_handles_sparse_labels(tmp_path):
    csv_path = tmp_path / "eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query",
                "rank",
                "context_source",
                "label_context",
            ],
        )
        writer.writeheader()
        writer.writerow({"query": "rag", "rank": "1", "context_source": "vault", "label_context": "1"})
        writer.writerow({"query": "rag", "rank": "2", "context_source": "paper", "label_context": "0"})
        writer.writerow({"query": "rag", "rank": "3", "context_source": "web", "label_context": ""})
        writer.writerow({"query": "transformer", "rank": "1", "context_source": "paper", "label_context": "0"})
        writer.writerow({"query": "transformer", "rank": "2", "context_source": "vault", "label_context": "1"})

    metrics = compute_precision_metrics(csv_path, label_col="label_context", source_col="context_source")

    assert metrics["queryCount"] == 2
    assert metrics["labeledQueryCount"] == 2
    assert metrics["precisionAt5"] > 0.0
    assert metrics["hitAt3"] == 1.0
    assert metrics["vaultHitRatio"] == 1.0


def test_build_eval_report_includes_runtime_diagnostics(tmp_path):
    csv_path = tmp_path / "eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query",
                "rank",
                "context_source",
                "label_context",
            ],
        )
        writer.writeheader()
        writer.writerow({"query": "rag", "rank": "1", "context_source": "vault", "label_context": "1"})

    runtime = {
        "schema": "knowledge-hub.runtime.diagnostics.v1",
        "status": "degraded",
        "degraded": True,
        "providers": [],
        "semanticRetrieval": {"provider": "openai", "model": "gpt-4o-mini", "available": False},
        "warnings": ["missing_api_key"],
    }

    report = build_eval_report(
        csv_path,
        label_col="label_context",
        source_col="context_source",
        runtime_diagnostics=runtime,
    )

    assert report["schema"] == "knowledge-hub.retrieval.eval.report.v1"
    assert report["status"] == "degraded"
    assert report["runtimeDiagnostics"]["status"] == "degraded"
    assert report["metrics"]["queryCount"] == 1
    assert validate_payload(report, "knowledge-hub.retrieval.eval.report.v1").ok
