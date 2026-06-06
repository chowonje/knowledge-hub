from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "knowledgeos" / "scripts" / "check_source_quality_detail_gate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("source_quality_detail_gate_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _detail_observation_payload(
    *,
    decision: str = "ready_for_detail_gate_review",
    blockers: list[str] | None = None,
    run_count: int = 7,
    required_runs: int = 7,
    paper_citation: float = 1.0,
    vault_abstention: float = 1.0,
    web_recency: float = 0.0,
    base_decision: str = "ready_for_hard_gate_review",
) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.source-quality-detail-observation.report.v1",
        "required_runs": required_runs,
        "run_count": run_count,
        "latest_run_dir": "/tmp/runs/source_quality_battery_20260421_010000",
        "decision": decision,
        "blockers": blockers or [],
        "base_observation_decision": base_decision,
        "checks": [
            {
                "source": "paper",
                "metric": "paper_citation_correctness",
                "status": "pass",
                "latest": paper_citation,
                "threshold": 1.0,
                "operator": ">=",
                "numericPointCount": required_runs,
            },
            {
                "source": "vault",
                "metric": "vault_abstention_correctness",
                "status": "pass",
                "latest": vault_abstention,
                "threshold": 1.0,
                "operator": ">=",
                "numericPointCount": required_runs,
            },
            {
                "source": "web",
                "metric": "web_recency_violation",
                "status": "pass",
                "latest": web_recency,
                "threshold": 0.0,
                "operator": "<=",
                "numericPointCount": required_runs,
            },
        ],
    }


def test_build_gate_result_passes_ready_detail_observation():
    module = _load_script()

    result = module.build_gate_result(_detail_observation_payload(), observation_path="/tmp/detail-observation.json")

    assert result["schema"] == "knowledge-hub.source-quality-detail-gate.result.v1"
    assert result["status"] == "ok"
    assert result["errors"] == []
    assert result["decision"] == "ready_for_detail_gate_review"
    assert result["baseObservationDecision"] == "ready_for_hard_gate_review"
    assert result["observationReportPath"] == "/tmp/detail-observation.json"


def test_build_gate_result_fails_on_blockers_not_ready_and_metric_drift():
    module = _load_script()

    result = module.build_gate_result(
        _detail_observation_payload(
            decision="not_ready_for_detail_gate_review",
            blockers=["vault_vault_abstention_correctness_unobserved"],
            paper_citation=0.9,
            vault_abstention=0.8,
            web_recency=0.2,
            base_decision="observe_more",
        )
    )

    assert result["status"] == "failed"
    assert "decision_not_ready:not_ready_for_detail_gate_review" in result["errors"]
    assert "blockers_present" in result["errors"]
    assert "base_observation_not_ready:observe_more" in result["errors"]
    assert "paper_paper_citation_correctness_below_gate" in result["errors"]
    assert "vault_vault_abstention_correctness_below_gate" in result["errors"]
    assert "web_web_recency_violation_above_gate" in result["errors"]


def test_build_gate_result_fails_on_check_status_blockers():
    module = _load_script()
    payload = _detail_observation_payload()
    checks = list(payload["checks"])
    checks[0] = {
        **dict(checks[0]),
        "status": "blocked",
        "blockers": ["paper_paper_citation_correctness_not_stable"],
    }
    payload["checks"] = checks

    result = module.build_gate_result(payload)

    assert result["status"] == "failed"
    assert "paper_paper_citation_correctness_status_blocked" in result["errors"]
    assert "paper_paper_citation_correctness_blockers_present" in result["errors"]
    assert result["checks"][0]["status"] == "fail"


def test_main_exits_nonzero_when_detail_observation_is_not_ready(tmp_path: Path, capsys):
    module = _load_script()
    runs_root = tmp_path / "runs"
    reports_root = runs_root / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "source_quality_detail_observation_latest.json").write_text(
        json.dumps(
            _detail_observation_payload(
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
