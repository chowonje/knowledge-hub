from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "eval/knowledgeos/scripts/report_source_quality_observation.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("source_quality_observation_report_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trend_payload(*, run_count: int, route: float = 1.0, stale: float = 0.0, legacy: float = 0.0, capability: float = 0.0, forced: float = 0.0):
    route_points = [{"created_at": f"2026-04-{idx+10:02d}T00:00:00+00:00", "run_dir": f"/runs/{idx}", "value": route, "passed": route >= 0.8} for idx in range(run_count)]
    stale_points = [{"created_at": f"2026-04-{idx+10:02d}T00:00:00+00:00", "run_dir": f"/runs/{idx}", "value": stale, "passed": stale <= 0.0} for idx in range(run_count)]
    return {
        "schema": "knowledge-hub.source-quality-trend.report.v1",
        "run_count": run_count,
        "latest_run_dir": "/runs/latest",
        "sources": {
            "paper": {
                "hard": {"route_correctness": {"latest": route, "points": route_points}},
            },
            "vault": {
                "hard": {
                    "route_correctness": {"latest": route, "points": route_points},
                    "vault_stale_citation_rate": {"latest": stale, "points": stale_points},
                },
            },
            "web": {
                "hard": {"route_correctness": {"latest": route, "points": route_points}},
            },
        },
        "readiness_latest_by_source": {
            source: {
                "legacy_runtime_rate": legacy,
                "capability_missing_rate": capability,
                "forced_legacy_rate": forced,
                "runtime_used_counts": {"ask_v2": 1},
                "fallback_reason_counts": {},
            }
            for source in ("paper", "vault", "web")
        },
    }


def _readiness_payload(*, decision: str, run_count: int, legacy: float = 0.0, capability: float = 0.0):
    values = [legacy] * run_count
    capability_values = [capability] * run_count
    return {
        "schema": "knowledge-hub.legacy-runtime-readiness.report.v1",
        "decision": decision,
        "run_count": run_count,
        "readiness_trends": {
            source: {
                "legacy_runtime_rate": values,
                "capability_missing_rate": capability_values,
                "forced_legacy_rate": [0.0] * run_count,
            }
            for source in ("paper", "vault", "web")
        },
    }


def test_build_report_observes_more_until_seven_runs():
    module = _load_script()

    report = module.build_report(
        trend_report=_trend_payload(run_count=5),
        readiness_report=_readiness_payload(decision="observe_more", run_count=5),
        required_runs=7,
    )

    assert report["decision"] == "observe_more"
    assert "need_7_runs_have_5" in report["blockers"]


def test_build_report_ready_when_hard_metrics_and_readiness_are_clean():
    module = _load_script()

    report = module.build_report(
        trend_report=_trend_payload(run_count=7),
        readiness_report=_readiness_payload(decision="ready_for_removal_tranche", run_count=7),
        required_runs=7,
    )

    assert report["decision"] == "ready_for_hard_gate_review"
    assert report["blockers"] == []


def test_build_report_blocks_on_route_or_readiness_drift():
    module = _load_script()

    report = module.build_report(
        trend_report=_trend_payload(run_count=7, route=0.95, legacy=0.1),
        readiness_report=_readiness_payload(decision="not_ready", run_count=7, legacy=0.1),
        required_runs=7,
    )

    assert report["decision"] == "not_ready_for_hard_gate_review"
    assert "paper_route_correctness_not_stable" in report["blockers"]
    assert "legacy_readiness_not_ready" in report["blockers"]
    assert "paper_legacy_runtime_rate_nonzero" in report["blockers"]


def test_render_markdown_includes_decision_and_blockers():
    module = _load_script()
    report = module.build_report(
        trend_report=_trend_payload(run_count=7, route=0.9),
        readiness_report=_readiness_payload(decision="not_ready", run_count=7),
        required_runs=7,
    )

    markdown = module.render_markdown(report)

    assert "# Source Quality Observation" in markdown
    assert "not_ready_for_hard_gate_review" in markdown
    assert "paper_route_correctness_not_stable" in markdown


def test_main_writes_outputs(tmp_path: Path):
    module = _load_script()
    runs_root = tmp_path / "runs"
    reports_root = runs_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "source_quality_trend_latest.json").write_text(
        json.dumps(_trend_payload(run_count=7), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_root / "legacy_runtime_readiness_latest.json").write_text(
        json.dumps(_readiness_payload(decision="ready_for_removal_tranche", run_count=7), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_json = reports_root / "observation.json"
    out_md = reports_root / "observation.md"

    exit_code = module.main(
        [
            "--runs-root",
            str(runs_root),
            "--required-runs",
            "7",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["decision"] == "ready_for_hard_gate_review"
    assert "# Source Quality Observation" in out_md.read_text(encoding="utf-8")
