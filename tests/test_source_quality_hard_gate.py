from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "knowledgeos" / "scripts" / "check_source_quality_hard_gate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("source_quality_hard_gate_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _observation_payload(
    *,
    decision: str = "ready_for_hard_gate_review",
    blockers: list[str] | None = None,
    run_count: int = 7,
    required_runs: int = 7,
    route: float = 1.0,
    stale: float = 0.0,
    legacy: float = 0.0,
    capability: float = 0.0,
) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.source-quality-observation.report.v1",
        "required_runs": required_runs,
        "run_count": run_count,
        "latest_run_dir": "/tmp/runs/source_quality_battery_20260421_010000",
        "decision": decision,
        "blockers": blockers or [],
        "sources": {
            "paper": {
                "route_correctness": route,
                "legacy_runtime_rate": legacy,
                "capability_missing_rate": capability,
                "forced_legacy_rate": 0.0,
            },
            "vault": {
                "route_correctness": route,
                "stale_citation_rate": stale,
                "legacy_runtime_rate": legacy,
                "capability_missing_rate": capability,
                "forced_legacy_rate": 0.0,
            },
            "web": {
                "route_correctness": route,
                "legacy_runtime_rate": legacy,
                "capability_missing_rate": capability,
                "forced_legacy_rate": 0.0,
            },
        },
    }


def test_build_gate_result_passes_ready_observation():
    module = _load_script()

    result = module.build_gate_result(_observation_payload(), observation_path="/tmp/observation.json")

    assert result["schema"] == "knowledge-hub.source-quality-hard-gate.result.v1"
    assert result["status"] == "ok"
    assert result["errors"] == []
    assert result["decision"] == "ready_for_hard_gate_review"
    assert result["observationReportPath"] == "/tmp/observation.json"


def test_build_gate_result_fails_on_not_ready_and_metric_drift():
    module = _load_script()

    result = module.build_gate_result(
        _observation_payload(
            decision="not_ready_for_hard_gate_review",
            blockers=["web_route_correctness_not_stable"],
            route=0.95,
            stale=0.25,
            capability=0.1,
        )
    )

    assert result["status"] == "failed"
    assert "decision_not_ready:not_ready_for_hard_gate_review" in result["errors"]
    assert "blockers_present" in result["errors"]
    assert "paper_route_correctness_below_gate" in result["errors"]
    assert "vault_stale_citation_rate_above_gate" in result["errors"]
    assert "web_capability_missing_rate_above_gate" in result["errors"]


def test_main_exits_nonzero_when_observation_is_not_ready(tmp_path: Path, capsys):
    module = _load_script()
    runs_root = tmp_path / "runs"
    reports_root = runs_root / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "source_quality_observation_latest.json").write_text(
        json.dumps(
            _observation_payload(
                decision="observe_more",
                blockers=["need_7_runs_have_6"],
                run_count=6,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    exit_code = module.main(["--runs-root", str(runs_root), "--json"])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert "insufficient_run_count:6/7" in payload["errors"]
