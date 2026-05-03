from __future__ import annotations

from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_daily_eval_center.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("daily_eval_center_runner_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _seed_eval_center_inputs(repo_root: Path) -> tuple[Path, Path]:
    runs_root = repo_root / "eval" / "knowledgeos" / "runs"
    queries_dir = repo_root / "eval" / "knowledgeos" / "queries"
    latest = _write_json(
        runs_root / "source_quality_battery_20260426_010143" / "source_quality_battery_summary.json",
        {
            "created_at": "2026-04-26T01:28:13+00:00",
            "gate_mode": "stub_hard",
            "retrieval_mode": "hybrid",
            "top_k": 6,
            "per_source": {
                "paper": {
                    "rows": 24,
                    "route_correctness": 1.0,
                    "no_result_rate": 0.375,
                    "runtime_used_counts": {"ask_v2": 24},
                    "fallback_reason_counts": {"section_blocked_to_claim_cards": 5},
                    "legacy_runtime_rate": 0.0,
                    "capability_missing_rate": 0.0,
                    "forced_legacy_rate": 0.0,
                    "citation_correctness_soft": 1.0,
                }
            },
        },
    )
    reports = runs_root / "reports"
    _write_json(
        reports / "source_quality_observation_latest.json",
        {
            "schema": "knowledge-hub.source-quality-observation.report.v1",
            "decision": "ready_for_hard_gate_review",
            "blockers": [],
            "run_count": 7,
            "required_runs": 7,
            "latest_run_dir": str(latest.parent),
        },
    )
    _write_json(
        reports / "source_quality_detail_observation_latest.json",
        {
            "schema": "knowledge-hub.source-quality-detail-observation.report.v1",
            "decision": "not_ready_for_detail_gate_review",
            "blockers": ["vault_abstention_correctness_need_7_numeric_points_have_6"],
        },
    )
    _write_json(reports / "source_quality_trend_latest.json", {"schema": "trend", "status": "ok"})
    _write_json(reports / "legacy_runtime_readiness_latest.json", {"schema": "legacy", "decision": "ready"})

    run_dir = runs_root / "answer_loop" / "manual-smoke-20260412" / "baseline_run_02"
    _write_json(
        run_dir / "answer_loop_collect_manifest.json",
        {
            "schema": "knowledge-hub.answer-loop.collect.result.v1",
            "status": "ok",
            "rowCount": 5,
            "packetCount": 5,
            "request": {
                "queriesPath": "eval/knowledgeos/runs/answer_loop/manual-smoke-20260412/queries_small_v1.csv",
                "answerBackends": ["codex_mcp"],
                "backendModels": {"codex_mcp": "gpt-5.4"},
                "retrievalMode": "hybrid",
                "topK": 8,
            },
        },
    )
    _write_json(
        run_dir / "answer_loop_judge_manifest.json",
        {
            "schema": "knowledge-hub.answer-loop.judge.result.v1",
            "status": "ok",
            "judgeProvider": "openai",
            "judgeModel": "gpt-4.1-nano",
            "rowCount": 5,
        },
    )
    _write_json(
        run_dir / "answer_loop_summary.json",
        {
            "schema": "knowledge-hub.answer-loop.summary.result.v1",
            "status": "ok",
            "rowCount": 5,
            "overall": {
                "predLabelScore": 0.8,
                "predGroundednessScore": 0.8,
                "predSourceAccuracyScore": 0.8,
                "abstainAgreement": 1.0,
            },
            "failureBucketCounts": {"groundedness_failure": 1},
            "failureCardCount": 1,
        },
    )

    _write_text(queries_dir / "paper_default_eval_queries_v1.csv", "query,source\np1,paper\n")
    _write_text(queries_dir / "user_answer_eval_queries_v1.csv", "query,source\n\"a,b\",paper,extra\n")
    return runs_root, queries_dir


def test_run_daily_snapshot_writes_snapshot_and_latest_reports(tmp_path: Path):
    module = _load_script()
    repo_root = tmp_path / "repo"
    runs_root, queries_dir = _seed_eval_center_inputs(repo_root)

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--repo-root",
            str(repo_root),
            "--runs-root",
            str(runs_root),
            "--queries-dir",
            str(queries_dir),
            "--failure-bank-path",
            str(repo_root / "missing_failure_bank.jsonl"),
        ]
    )

    result = module.run_daily_snapshot(args, now=datetime.fromisoformat("2026-04-26T11:00:00+09:00"))

    assert result["status"] == "ok"
    assert Path(result["snapshotJsonPath"]).exists()
    assert Path(result["snapshotMarkdownPath"]).exists()
    assert Path(result["latestJsonPath"]).exists()
    assert Path(result["latestMarkdownPath"]).exists()
    markdown = Path(result["latestMarkdownPath"]).read_text(encoding="utf-8")
    assert "# Daily Eval Center Brief" in markdown
    assert "## Part Status" in markdown
    assert "## Findings" in markdown
    assert "Detail-quality promotion is still blocked" in markdown
    assert "summary_modified_at=" in markdown
    assert result["summary"]["sourceQualityBaseDecision"] == "ready_for_hard_gate_review"
    assert result["summary"]["sourceQualityDetailDecision"] == "not_ready_for_detail_gate_review"
    assert result["summary"]["sourceQualityFreshnessStatus"] == "fresh"
    assert result["summary"]["answerLoopStatus"] == "ok"
    assert result["summary"]["answerLoopRowCount"] == 5
    assert "failure_bank" in result["summary"]["gapIds"]


def test_parser_default_runs_root_is_externalized():
    module = _load_script()
    parser = module._build_parser()
    args = parser.parse_args([])

    assert args.runs_root == "~/.khub/eval/knowledgeos/runs"


def test_run_daily_snapshot_skips_when_local_date_already_covered(tmp_path: Path):
    module = _load_script()
    repo_root = tmp_path / "repo"
    runs_root, queries_dir = _seed_eval_center_inputs(repo_root)
    reports_root = runs_root / "reports"
    _write_json(
        reports_root / "eval_center_latest.json",
        module.build_eval_center_summary(
            runs_root=runs_root,
            queries_dir=queries_dir,
            failure_bank_path=repo_root / "missing_failure_bank.jsonl",
            repo_root=repo_root,
            generated_at="2026-04-26T00:30:00+00:00",
        ),
    )

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--repo-root",
            str(repo_root),
            "--runs-root",
            str(runs_root),
            "--queries-dir",
            str(queries_dir),
            "--failure-bank-path",
            str(repo_root / "missing_failure_bank.jsonl"),
            "--skip-if-local-date-already-covered",
            "--local-timezone",
            "Asia/Seoul",
        ]
    )

    result = module.run_daily_snapshot(args, now=datetime.fromisoformat("2026-04-26T11:00:00+09:00"))

    assert result["skipped"] is True
    assert result["skipReason"] == "already_ran_for_local_date"
    assert result["latestSnapshot"]["localDate"] == "2026-04-26"


def test_run_daily_snapshot_refreshes_stale_same_day_snapshot(tmp_path: Path):
    module = _load_script()
    repo_root = tmp_path / "repo"
    runs_root, queries_dir = _seed_eval_center_inputs(repo_root)
    stale_payload = module.build_eval_center_summary(
        runs_root=runs_root,
        queries_dir=queries_dir,
        failure_bank_path=repo_root / "missing_failure_bank.jsonl",
        repo_root=repo_root,
        generated_at="2026-04-26T00:30:00+00:00",
        expected_source_quality_local_date="2026-04-27",
        freshness_timezone="Asia/Seoul",
    )
    _write_json(runs_root / "reports" / "eval_center_latest.json", stale_payload)

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--repo-root",
            str(repo_root),
            "--runs-root",
            str(runs_root),
            "--queries-dir",
            str(queries_dir),
            "--failure-bank-path",
            str(repo_root / "missing_failure_bank.jsonl"),
            "--skip-if-local-date-already-covered",
            "--local-timezone",
            "Asia/Seoul",
        ]
    )

    result = module.run_daily_snapshot(args, now=datetime.fromisoformat("2026-04-26T11:05:00+09:00"))

    assert result.get("skipped") is not True
    assert result["summary"]["sourceQualityFreshnessStatus"] == "fresh"
    assert Path(result["latestJsonPath"]).exists()


def test_run_daily_snapshot_marks_stale_source_quality(tmp_path: Path):
    module = _load_script()
    repo_root = tmp_path / "repo"
    runs_root, queries_dir = _seed_eval_center_inputs(repo_root)
    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--repo-root",
            str(repo_root),
            "--runs-root",
            str(runs_root),
            "--queries-dir",
            str(queries_dir),
            "--failure-bank-path",
            str(repo_root / "missing_failure_bank.jsonl"),
            "--local-timezone",
            "Asia/Seoul",
        ]
    )

    result = module.run_daily_snapshot(args, now=datetime.fromisoformat("2026-04-27T11:10:00+09:00"))

    assert result["summary"]["status"] == "warn"
    assert result["summary"]["sourceQualityFreshnessStatus"] == "stale"
    assert result["summary"]["sourceQualityFreshnessExpectedDate"] == "2026-04-27"
    payload = json.loads(Path(result["latestJsonPath"]).read_text(encoding="utf-8"))
    assert any("source-quality latest run is stale" in warning for warning in payload["warnings"])
    markdown = Path(result["latestMarkdownPath"]).read_text(encoding="utf-8")
    assert "source_quality_freshness: `stale`" in markdown


def test_run_daily_snapshot_waits_for_today_source_quality(tmp_path: Path):
    module = _load_script()
    repo_root = tmp_path / "repo"
    runs_root, queries_dir = _seed_eval_center_inputs(repo_root)
    calls: list[float] = []
    clock = [0.0]

    def sleep_fn(seconds: float) -> None:
        calls.append(seconds)
        clock[0] += seconds
        latest = _write_json(
            runs_root / "source_quality_battery_20260427_010000" / "source_quality_battery_summary.json",
            {
                "created_at": "2026-04-27T01:00:00+00:00",
                "gate_mode": "stub_hard",
                "retrieval_mode": "hybrid",
                "top_k": 6,
                "per_source": {
                    "paper": {
                        "rows": 24,
                        "route_correctness": 1.0,
                        "runtime_used_counts": {"ask_v2": 24},
                        "legacy_runtime_rate": 0.0,
                        "capability_missing_rate": 0.0,
                        "forced_legacy_rate": 0.0,
                    }
                },
            },
        )
        _write_json(
            runs_root / "reports" / "source_quality_observation_latest.json",
            {
                "schema": "knowledge-hub.source-quality-observation.report.v1",
                "decision": "ready_for_hard_gate_review",
                "blockers": [],
                "run_count": 7,
                "required_runs": 7,
                "latest_run_dir": str(latest.parent),
            },
        )

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--repo-root",
            str(repo_root),
            "--runs-root",
            str(runs_root),
            "--queries-dir",
            str(queries_dir),
            "--failure-bank-path",
            str(repo_root / "missing_failure_bank.jsonl"),
            "--local-timezone",
            "Asia/Seoul",
            "--wait-for-today-source-quality-seconds",
            "10",
            "--source-quality-wait-poll-seconds",
            "1",
        ]
    )

    result = module.run_daily_snapshot(
        args,
        now=datetime.fromisoformat("2026-04-27T11:10:00+09:00"),
        sleep_fn=sleep_fn,
        monotonic_fn=lambda: clock[0],
    )

    assert calls == [1.0]
    assert result["sourceQualityWait"]["status"] == "fresh"
    assert result["sourceQualityWait"]["latestRunLocalDate"] == "2026-04-27"
    assert result["summary"]["sourceQualityFreshnessStatus"] == "fresh"
    assert result["summary"]["sourceQualityBaseDecision"] == "ready_for_hard_gate_review"


def test_run_daily_snapshot_does_not_skip_invalid_latest_snapshot(tmp_path: Path):
    module = _load_script()
    repo_root = tmp_path / "repo"
    runs_root, queries_dir = _seed_eval_center_inputs(repo_root)
    _write_json(
        runs_root / "reports" / "eval_center_latest.json",
        {
            "schema": "knowledge-hub.eval-center.summary.result.v1",
            "generatedAt": "2026-04-26T00:30:00+00:00",
        },
    )

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--repo-root",
            str(repo_root),
            "--runs-root",
            str(runs_root),
            "--queries-dir",
            str(queries_dir),
            "--failure-bank-path",
            str(repo_root / "missing_failure_bank.jsonl"),
            "--skip-if-local-date-already-covered",
            "--local-timezone",
            "Asia/Seoul",
        ]
    )

    result = module.run_daily_snapshot(args, now=datetime.fromisoformat("2026-04-26T11:10:00+09:00"))

    assert result["status"] == "ok"
    assert result.get("skipped") is not True
    assert Path(result["latestJsonPath"]).exists()


def test_run_daily_snapshot_rejects_invalid_payload_before_writing(tmp_path: Path, monkeypatch):
    module = _load_script()
    repo_root = tmp_path / "repo"
    runs_root, queries_dir = _seed_eval_center_inputs(repo_root)
    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--repo-root",
            str(repo_root),
            "--runs-root",
            str(runs_root),
            "--queries-dir",
            str(queries_dir),
            "--failure-bank-path",
            str(repo_root / "missing_failure_bank.jsonl"),
        ]
    )

    monkeypatch.setattr(
        module,
        "build_eval_center_summary",
        lambda **_: {"schema": "knowledge-hub.eval-center.summary.result.v1", "generatedAt": "2026-04-26T00:00:00+00:00"},
    )

    with pytest.raises(ValueError, match="failed schema validation"):
        module.run_daily_snapshot(args, now=datetime.fromisoformat("2026-04-26T11:20:00+09:00"))
