from __future__ import annotations

from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_daily_rag_vnext.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("daily_rag_vnext_runner_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_queries(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    def search_with_diagnostics(self, query, **kwargs):  # noqa: ANN001
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
                "retrievalPlan": {},
                "retrievalStrategy": {
                    "phase": "phase0_diagnostics",
                    "complexityClass": "update_sensitive",
                    "retrievalBudget": {"topK": kwargs.get("top_k", 5)},
                    "retryPolicy": {"mode": "diagnostics_only", "maxRetries": 0, "allowedActions": []},
                },
                "retrievalQuality": {
                    "label": "low",
                    "score": 0.2,
                    "weakSignals": ["low_top_score"],
                    "correctiveActionCandidate": "broaden_search",
                },
                "answerabilityRerank": {"applied": False, "label": "low", "score": 0.2, "weakSignals": []},
                "correctiveRetrieval": {
                    "applied": False,
                    "policy": "diagnostics_only",
                    "retryCandidate": True,
                    "candidateAction": "broaden_search",
                    "triggers": ["low_top_score"],
                },
                "artifactHealth": {"label": "high", "score": 1.0},
            },
        }


def test_parser_default_runs_root_is_externalized():
    module = _load_script()
    args = module._build_parser().parse_args([])

    assert args.runs_root == "~/.khub/eval/knowledgeos/runs/rag_vnext"


def test_run_daily_observation_writes_snapshot_and_latest(tmp_path: Path):
    module = _load_script()
    repo_root = tmp_path / "repo"
    queries = _write_queries(repo_root / "eval" / "knowledgeos" / "queries" / "rag.csv")
    runs_root = tmp_path / "runs"
    args = module._build_parser().parse_args(
        [
            "--repo-root",
            str(repo_root),
            "--runs-root",
            str(runs_root),
            "--queries",
            str(queries),
            "--limit",
            "1",
            "--retry-limit",
            "1",
            "--rerank-limit",
            "1",
            "--graph-limit",
            "1",
        ]
    )

    result = module.run_daily_observation(
        args,
        now=datetime.fromisoformat("2026-04-28T10:00:00+09:00"),
        searcher=_ObservationSearcher(),
    )

    assert result["status"] == "ok"
    assert Path(result["snapshotJsonPath"]).exists()
    assert Path(result["snapshotMarkdownPath"]).exists()
    assert Path(result["latestJsonPath"]).exists()
    assert Path(result["latestMarkdownPath"]).exists()
    latest = json.loads(Path(result["latestJsonPath"]).read_text(encoding="utf-8"))
    assert latest["schema"] == "knowledge-hub.rag.vnext-observation.report.v1"
    assert latest["summary"]["rowCount"] == 1
    markdown = Path(result["latestMarkdownPath"]).read_text(encoding="utf-8")
    assert "# RAG vNext Observation" in markdown
    assert "## Readiness Blockers" in markdown


def test_run_daily_observation_skips_when_local_date_already_covered(tmp_path: Path):
    module = _load_script()
    repo_root = tmp_path / "repo"
    queries = _write_queries(repo_root / "eval" / "knowledgeos" / "queries" / "rag.csv")
    runs_root = tmp_path / "runs"
    args = module._build_parser().parse_args(
        [
            "--repo-root",
            str(repo_root),
            "--runs-root",
            str(runs_root),
            "--queries",
            str(queries),
            "--skip-if-local-date-already-covered",
            "--local-timezone",
            "Asia/Seoul",
        ]
    )
    module.run_daily_observation(
        args,
        now=datetime.fromisoformat("2026-04-28T10:00:00+09:00"),
        searcher=_ObservationSearcher(),
    )

    result = module.run_daily_observation(
        args,
        now=datetime.fromisoformat("2026-04-28T20:00:00+09:00"),
        searcher=_ObservationSearcher(),
    )

    assert result["skipped"] is True
    assert result["skipReason"] == "already_ran_for_local_date"
    assert result["latestSnapshot"]["localDate"] == "2026-04-28"
