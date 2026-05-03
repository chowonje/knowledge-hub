from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "eval/knowledgeos/scripts/report_source_quality_trend.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("source_quality_trend_report_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_summary(
    run_dir: Path,
    *,
    created_at: str,
    paper: float,
    vault: float,
    web: float,
    stale: float,
    paper_soft: float,
    web_soft: float,
    paper_legacy: float = 0.0,
    vault_legacy: float = 0.0,
    web_legacy: float = 0.0,
    paper_capability: float = 0.0,
    vault_capability: float = 0.0,
    web_capability: float = 0.0,
    paper_forced: float = 0.0,
    vault_forced: float = 0.0,
    web_forced: float = 0.0,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": created_at,
        "per_source": {
            "paper": {
                "route_correctness": paper,
                "citation_correctness_soft": paper_soft,
                "legacy_runtime_rate": paper_legacy,
                "capability_missing_rate": paper_capability,
                "forced_legacy_rate": paper_forced,
                "runtime_used_counts": {"ask_v2": 1},
                "fallback_reason_counts": {},
            },
            "vault": {
                "route_correctness": vault,
                "stale_citation_rate": stale,
                "abstention_correctness_soft": None,
                "legacy_runtime_rate": vault_legacy,
                "capability_missing_rate": vault_capability,
                "forced_legacy_rate": vault_forced,
                "runtime_used_counts": {"legacy": 1} if vault_legacy else {},
                "fallback_reason_counts": {"ask_v2_capability_missing": 1} if vault_capability else {},
            },
            "web": {
                "route_correctness": web,
                "recency_violation_soft": web_soft,
                "legacy_runtime_rate": web_legacy,
                "capability_missing_rate": web_capability,
                "forced_legacy_rate": web_forced,
                "runtime_used_counts": {"legacy": 1} if web_legacy else {"ask_v2": 1},
                "fallback_reason_counts": {"ask_v2_not_used": 1} if web_forced else {},
            },
        },
        "hard_gates": {
            "route_correctness": {
                "paper": {"passed": paper >= 0.8},
                "vault": {"passed": vault >= 0.8},
                "web": {"passed": web >= 0.8},
            },
            "vault_stale_citation_rate": {
                "vault": {"passed": stale <= 0.0},
            },
        },
    }
    path = run_dir / "source_quality_battery_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_trend_report_tracks_latest_and_delta(tmp_path: Path):
    module = _load_script()
    runs_root = tmp_path / "runs"
    _write_summary(
        runs_root / "source_quality_battery_20260417_000000",
        created_at="2026-04-17T00:00:00+00:00",
        paper=0.9,
        vault=1.0,
        web=0.8,
        stale=0.0,
        paper_soft=0.7,
        web_soft=0.4,
    )
    _write_summary(
        runs_root / "source_quality_battery_20260418_000000",
        created_at="2026-04-18T00:00:00+00:00",
        paper=1.0,
        vault=1.0,
        web=0.95,
        stale=0.0,
        paper_soft=0.9,
        web_soft=0.2,
    )

    summaries = module._load_run_summaries(runs_root, limit=7)
    report = module.build_trend_report(summaries)

    assert report["schema"] == "knowledge-hub.source-quality-trend.report.v1"
    assert report["run_count"] == 2
    assert report["latest_run_dir"].endswith("source_quality_battery_20260418_000000")
    assert report["sources"]["paper"]["hard"]["route_correctness"]["latest"] == 1.0
    assert report["sources"]["paper"]["hard"]["route_correctness"]["delta_vs_previous"] == 0.1
    assert report["sources"]["web"]["hard"]["route_correctness"]["latest"] == 0.95
    assert report["sources"]["web"]["hard"]["route_correctness"]["direction"] == "up"
    assert report["sources"]["vault"]["hard"]["vault_stale_citation_rate"]["latest"] == 0.0
    assert report["sources"]["paper"]["soft"]["paper_citation_correctness"]["latest"] == 0.9
    assert report["sources"]["paper"]["readiness"]["legacy_runtime_rate"]["latest"] == 0.0
    assert report["sources"]["web"]["readiness"]["forced_legacy_rate"]["latest"] == 0.0
    assert report["readiness_latest_by_source"]["paper"]["legacy_runtime_rate"] == 0.0


def test_load_run_summaries_keeps_latest_run_per_day(tmp_path: Path):
    module = _load_script()
    runs_root = tmp_path / "runs"
    _write_summary(
        runs_root / "source_quality_battery_20260417_010000",
        created_at="2026-04-17T01:00:00+00:00",
        paper=1.0,
        vault=1.0,
        web=1.0,
        stale=0.0,
        paper_soft=1.0,
        web_soft=0.0,
        vault_legacy=0.5,
    )
    latest_same_day = _write_summary(
        runs_root / "source_quality_battery_20260417_020000",
        created_at="2026-04-17T02:00:00+00:00",
        paper=1.0,
        vault=1.0,
        web=1.0,
        stale=0.0,
        paper_soft=1.0,
        web_soft=0.0,
        vault_legacy=0.0,
    )
    _write_summary(
        runs_root / "source_quality_battery_20260418_000000",
        created_at="2026-04-18T00:00:00+00:00",
        paper=1.0,
        vault=1.0,
        web=1.0,
        stale=0.0,
        paper_soft=1.0,
        web_soft=0.0,
    )

    summaries = module._load_run_summaries(runs_root, limit=7)
    report = module.build_trend_report(summaries)

    assert report["run_count"] == 2
    assert report["runs"][0]["summary_path"] == str(latest_same_day)
    vault_legacy = report["sources"]["vault"]["readiness"]["legacy_runtime_rate"]
    assert vault_legacy["points"][0]["value"] == 0.0


def test_render_markdown_includes_sources_and_metric_lines(tmp_path: Path):
    module = _load_script()
    runs_root = tmp_path / "runs"
    _write_summary(
        runs_root / "source_quality_battery_20260417_000000",
        created_at="2026-04-17T00:00:00+00:00",
        paper=1.0,
        vault=1.0,
        web=1.0,
        stale=0.0,
        paper_soft=1.0,
        web_soft=0.0,
    )
    report = module.build_trend_report(module._load_run_summaries(runs_root, limit=7))

    markdown = module.render_markdown(report)

    assert "# Source Quality Trend" in markdown
    assert "## paper" in markdown
    assert "## vault" in markdown
    assert "## web" in markdown
    assert "route_correctness" in markdown
    assert "vault_stale_citation_rate" in markdown
    assert "paper_citation_correctness" in markdown
    assert "web_recency_violation" in markdown
    assert "legacy_runtime_rate" in markdown
    assert "capability_missing_rate" in markdown
    assert "forced_legacy_rate" in markdown


def test_main_writes_json_and_markdown_outputs(tmp_path: Path):
    module = _load_script()
    runs_root = tmp_path / "runs"
    _write_summary(
        runs_root / "source_quality_battery_20260417_000000",
        created_at="2026-04-17T00:00:00+00:00",
        paper=1.0,
        vault=1.0,
        web=1.0,
        stale=0.0,
        paper_soft=1.0,
        web_soft=0.0,
    )
    out_json = tmp_path / "reports" / "trend.json"
    out_md = tmp_path / "reports" / "trend.md"

    exit_code = module.main(
        [
            "--runs-root",
            str(runs_root),
            "--limit",
            "5",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert exit_code == 0
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_count"] == 1
    assert payload["sources"]["web"]["hard"]["route_correctness"]["latest"] == 1.0
    assert payload["sources"]["web"]["readiness"]["legacy_runtime_rate"]["latest"] == 0.0
    assert "# Source Quality Trend" in out_md.read_text(encoding="utf-8")
