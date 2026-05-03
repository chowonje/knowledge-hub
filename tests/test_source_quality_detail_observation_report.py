from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "knowledgeos" / "scripts" / "report_source_quality_detail_observation.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("source_quality_detail_observation_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _points(run_count: int, value):
    return [
        {
            "created_at": f"2026-04-{idx+10:02d}T00:00:00+00:00",
            "run_dir": f"/runs/{idx}",
            "value": value,
        }
        for idx in range(run_count)
    ]


def _trend_payload(*, run_count: int, paper_citation=1.0, vault_abstention=1.0, web_recency=0.0):
    return {
        "schema": "knowledge-hub.source-quality-trend.report.v1",
        "run_count": run_count,
        "latest_run_dir": "/runs/latest",
        "sources": {
            "paper": {
                "soft": {
                    "paper_citation_correctness": {
                        "latest": paper_citation,
                        "min": paper_citation,
                        "max": paper_citation,
                        "points": _points(run_count, paper_citation),
                    }
                }
            },
            "vault": {
                "soft": {
                    "vault_abstention_correctness": {
                        "latest": vault_abstention,
                        "min": vault_abstention,
                        "max": vault_abstention,
                        "points": _points(run_count, vault_abstention),
                    }
                }
            },
            "web": {
                "soft": {
                    "web_recency_violation": {
                        "latest": web_recency,
                        "min": web_recency,
                        "max": web_recency,
                        "points": _points(run_count, web_recency),
                    }
                }
            },
        },
    }


def _base_observation_payload(*, decision: str = "ready_for_hard_gate_review", blockers: list[str] | None = None):
    return {
        "schema": "knowledge-hub.source-quality-observation.report.v1",
        "decision": decision,
        "blockers": blockers or [],
    }


def test_build_report_ready_when_detail_metrics_are_stable():
    module = _load_script()

    report = module.build_report(
        trend_report=_trend_payload(run_count=7),
        base_observation_report=_base_observation_payload(),
        required_runs=7,
    )

    assert report["schema"] == "knowledge-hub.source-quality-detail-observation.report.v1"
    assert report["decision"] == "ready_for_detail_gate_review"
    assert report["blockers"] == []
    assert all(check["status"] == "pass" for check in report["checks"])


def test_build_report_blocks_unobserved_vault_abstention():
    module = _load_script()

    report = module.build_report(
        trend_report=_trend_payload(run_count=7, vault_abstention=None),
        base_observation_report=_base_observation_payload(),
        required_runs=7,
    )

    assert report["decision"] == "not_ready_for_detail_gate_review"
    assert "vault_vault_abstention_correctness_unobserved" in report["blockers"]
    vault_check = next(check for check in report["checks"] if check["source"] == "vault")
    assert vault_check["numericPointCount"] == 0
    assert vault_check["status"] == "blocked"


def test_build_report_blocks_when_base_source_quality_not_ready():
    module = _load_script()

    report = module.build_report(
        trend_report=_trend_payload(run_count=7),
        base_observation_report=_base_observation_payload(
            decision="not_ready_for_hard_gate_review",
            blockers=["web_route_correctness_not_stable"],
        ),
        required_runs=7,
    )

    assert report["decision"] == "not_ready_for_detail_gate_review"
    assert "base_source_quality_not_ready:not_ready_for_hard_gate_review" in report["blockers"]
    assert "base_source_quality_blockers_present" in report["blockers"]


def test_main_writes_outputs(tmp_path: Path):
    module = _load_script()
    runs_root = tmp_path / "runs"
    reports_root = runs_root / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "source_quality_trend_latest.json").write_text(
        json.dumps(_trend_payload(run_count=7, vault_abstention=None), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_root / "source_quality_observation_latest.json").write_text(
        json.dumps(_base_observation_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_json = reports_root / "detail.json"
    out_md = reports_root / "detail.md"

    exit_code = module.main(
        [
            "--runs-root",
            str(runs_root),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["decision"] == "not_ready_for_detail_gate_review"
    assert "# Source Quality Detail Observation" in out_md.read_text(encoding="utf-8")
