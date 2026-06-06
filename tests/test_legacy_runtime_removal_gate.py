from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "knowledgeos" / "scripts" / "check_legacy_runtime_removal_gate.py"
UNEXPECTED_TEST_LITERAL_ERROR = "ask_v2_mode_legacy_literal_tests_unexpected_hits_present"


def _load_script():
    spec = importlib.util.spec_from_file_location("legacy_runtime_removal_gate_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _allowed_guard_hit() -> dict[str, str | int]:
    return {
        "path": "tests/test_rag_runtime_services.py",
        "line": 438,
        "text": "ask_v2_mode=\"legacy\",",
    }


def _readiness_payload(
    *,
    decision: str = "ready_for_removal_tranche",
    run_count: int = 7,
    required_run_count: int = 7,
    runtime_hit: bool = False,
    script_hit: bool = False,
    legacy_rate: float = 0.0,
    test_literal_hits: list[dict[str, str | int]] | None = None,
) -> dict[str, object]:
    runtime_hits = [{"path": "knowledge_hub/ai/rag_legacy_runtime.py", "line": 1, "text": "LegacyRAGRuntime"}] if runtime_hit else []
    script_hits = [{"path": "scripts/example.py", "line": 1, "text": "ask_v2_mode=\"legacy\""}] if script_hit else []
    ask_v2_test_hits = test_literal_hits if test_literal_hits is not None else [_allowed_guard_hit()]
    return {
        "schema": "knowledge-hub.legacy-runtime-readiness.report.v1",
        "decision": decision,
        "run_count": run_count,
        "required_run_count": required_run_count,
        "trend_report_path": "/tmp/runs/reports/source_quality_trend_latest.json",
        "callsites": {
            "legacy_runtime_symbol": {
                "runtime": runtime_hits,
                "tests": [],
                "eval": [],
                "scripts": [],
            },
            "ask_v2_mode_legacy_literal": {
                "runtime": [],
                "tests": ask_v2_test_hits,
                "eval": [],
                "scripts": script_hits,
            },
        },
        "readiness_trends": {
            source: {
                "legacy_runtime_rate": [legacy_rate for _ in range(run_count)],
                "capability_missing_rate": [0.0 for _ in range(run_count)],
                "forced_legacy_rate": [0.0 for _ in range(run_count)],
            }
            for source in ("paper", "vault", "web")
        },
    }


def test_build_gate_result_passes_ready_removed_runtime():
    module = _load_script()

    result = module.build_gate_result(_readiness_payload(), readiness_path="/tmp/legacy.json")

    assert result["schema"] == "knowledge-hub.legacy-runtime-removal-gate.result.v1"
    assert result["status"] == "ok"
    assert result["decision"] == "ready_for_removal_tranche"
    assert result["errors"] == []
    assert result["readinessReportPath"] == "/tmp/legacy.json"
    assert result["testOnlyLegacyLiteralHits"] == 1


def test_build_gate_result_fails_on_unexpected_test_legacy_literal():
    module = _load_script()

    # Given: readiness has the allowed removal-guard literal plus an unrelated tests literal.
    unexpected_hit = {
        "path": "tests/test_unrelated_legacy_mode.py",
        "line": 12,
        "text": "ask_v2_mode=\"legacy\"",
    }
    readiness = _readiness_payload(test_literal_hits=[_allowed_guard_hit(), unexpected_hit])

    # When: the removal gate evaluates the readiness report.
    result = module.build_gate_result(readiness, readiness_path="/tmp/legacy.json")

    # Then: the gate fails and exposes the unexpected tests hit.
    assert result["status"] == "failed"
    assert UNEXPECTED_TEST_LITERAL_ERROR in result["errors"]
    assert result["testOnlyLegacyLiteralHits"] == 2
    assert result["allowedTestLegacyLiteralHits"] == 1
    assert result["unexpectedTestLegacyLiteralHits"] == 1
    assert result["unexpectedTestLegacyLiteralHitDetails"] == [unexpected_hit]


def test_build_gate_result_fails_on_malformed_test_legacy_literal_hit():
    module = _load_script()

    # Given: readiness has a tests literal hit that cannot be matched by path and text.
    malformed_hit = {
        "line": 99,
        "text": "ask_v2_mode=\"legacy\"",
    }
    readiness = _readiness_payload(test_literal_hits=[_allowed_guard_hit(), malformed_hit])

    # When: the removal gate evaluates the readiness report.
    result = module.build_gate_result(readiness, readiness_path="/tmp/legacy.json")

    # Then: the gate fails closed instead of treating all test hits as acceptable.
    assert result["status"] == "failed"
    assert UNEXPECTED_TEST_LITERAL_ERROR in result["errors"]
    assert result["testOnlyLegacyLiteralHits"] == 2
    assert result["allowedTestLegacyLiteralHits"] == 1
    assert result["unexpectedTestLegacyLiteralHits"] == 1
    assert result["unexpectedTestLegacyLiteralHitDetails"] == [malformed_hit]


def test_build_gate_result_fails_on_runtime_or_script_callsites_and_metric_drift():
    module = _load_script()

    result = module.build_gate_result(
        _readiness_payload(
            decision="not_ready",
            runtime_hit=True,
            script_hit=True,
            legacy_rate=0.25,
        )
    )

    assert result["status"] == "failed"
    assert "decision_not_ready:not_ready" in result["errors"]
    assert "legacy_runtime_symbol_runtime_hits_present" in result["errors"]
    assert "ask_v2_mode_legacy_literal_scripts_hits_present" in result["errors"]
    assert "paper_legacy_runtime_rate_above_gate" in result["errors"]
    assert "web_legacy_runtime_rate_above_gate" in result["errors"]


def test_main_exits_nonzero_when_readiness_has_insufficient_runs(tmp_path: Path, capsys):
    module = _load_script()
    runs_root = tmp_path / "runs"
    reports_root = runs_root / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "legacy_runtime_readiness_latest.json").write_text(
        json.dumps(_readiness_payload(decision="observe_more", run_count=6), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    exit_code = module.main(["--runs-root", str(runs_root), "--json"])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert "insufficient_run_count:6/7" in payload["errors"]
