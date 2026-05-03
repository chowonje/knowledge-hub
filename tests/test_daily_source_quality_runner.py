from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_daily_source_quality.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("daily_source_quality_runner_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _observation_payload() -> dict[str, object]:
    return {
        "schema": "knowledge-hub.source-quality-observation.report.v1",
        "required_runs": 7,
        "run_count": 6,
        "latest_run_dir": "/tmp/runs/source_quality_battery_20260419_000000",
        "decision": "observe_more",
        "blockers": ["need_7_runs_have_6"],
        "legacy_readiness_decision": "observe_more",
        "sources": {
            "paper": {
                "route_correctness": 1.0,
                "legacy_runtime_rate": 0.0,
                "capability_missing_rate": 0.0,
                "forced_legacy_rate": 0.0,
            },
            "vault": {
                "route_correctness": 1.0,
                "stale_citation_rate": 0.0,
                "legacy_runtime_rate": 0.0,
                "capability_missing_rate": 0.0,
                "forced_legacy_rate": 0.0,
            },
            "web": {
                "route_correctness": 0.8,
                "legacy_runtime_rate": 0.0,
                "capability_missing_rate": 0.0,
                "forced_legacy_rate": 0.0,
            },
        },
    }


def _detail_observation_payload() -> dict[str, object]:
    return {
        "schema": "knowledge-hub.source-quality-detail-observation.report.v1",
        "required_runs": 7,
        "run_count": 7,
        "latest_run_dir": "/tmp/runs/source_quality_battery_20260419_000000",
        "decision": "not_ready_for_detail_gate_review",
        "blockers": ["vault_vault_abstention_correctness_unobserved"],
        "base_observation_decision": "ready_for_hard_gate_review",
        "checks": [
            {
                "source": "paper",
                "metric": "paper_citation_correctness",
                "status": "pass",
                "latest": 1.0,
                "numericPointCount": 7,
            },
            {
                "source": "vault",
                "metric": "vault_abstention_correctness",
                "status": "blocked",
                "latest": None,
                "numericPointCount": 0,
            },
            {
                "source": "web",
                "metric": "web_recency_violation",
                "status": "pass",
                "latest": 0.0,
                "numericPointCount": 7,
            },
        ],
    }


def test_build_observation_summary_pulls_daily_fields():
    module = _load_script()

    summary = module._build_observation_summary(_observation_payload())

    assert summary["decision"] == "observe_more"
    assert summary["blockers"] == ["need_7_runs_have_6"]
    assert summary["runCount"] == 6
    assert summary["requiredRuns"] == 7
    assert summary["sources"]["paper"]["routeCorrectness"] == 1.0
    assert summary["sources"]["vault"]["staleCitationRate"] == 0.0
    assert summary["sources"]["web"]["routeCorrectness"] == 0.8


def test_build_detail_observation_summary_pulls_detail_fields():
    module = _load_script()

    summary = module._build_detail_observation_summary(_detail_observation_payload())

    assert summary["decision"] == "not_ready_for_detail_gate_review"
    assert summary["blockers"] == ["vault_vault_abstention_correctness_unobserved"]
    assert summary["checks"][0]["metric"] == "paper_citation_correctness"
    assert summary["checks"][1]["status"] == "blocked"


def test_build_writeback_goal_includes_required_metrics():
    module = _load_script()
    summary = module._build_observation_summary(_observation_payload())

    goal = module.build_writeback_goal(summary)

    assert "docs/status와 worklog에 정리해줘" in goal
    assert "decision=observe_more" in goal
    assert "blockers=need_7_runs_have_6" in goal
    assert "paper route_correctness=1.0" in goal
    assert "vault route_correctness=1.0" in goal
    assert "web route_correctness=0.8" in goal
    assert "vault stale_citation_rate=0.0" in goal


def test_build_writeback_argv_respects_apply_and_workspace_flags():
    module = _load_script()
    parser = module._build_parser()
    args = parser.parse_args(["--writeback", "--apply-writeback", "--include-workspace", "--actor", "won"])

    argv = module.build_writeback_argv(args, goal="Update docs/status and worklog", repo_root=Path("/tmp/repo"))

    assert argv[1].endswith("scripts/run_agent_docs_writeback_loop.py")
    assert "--repo-path" in argv
    assert "/tmp/repo" in argv
    assert "--include-workspace" in argv
    assert "--apply" in argv
    assert argv[-1] == "--apply"


def test_build_hard_gate_argv_targets_source_quality_gate():
    module = _load_script()

    argv = module.build_hard_gate_argv(repo_root=Path("/tmp/repo"), runs_root=Path("/tmp/runs"))

    assert argv[1].endswith("eval/knowledgeos/scripts/check_source_quality_hard_gate.py")
    assert "--runs-root" in argv
    assert "/tmp/runs" in argv
    assert argv[-1] == "--json"


def test_main_runs_commands_and_writeback(monkeypatch, tmp_path: Path, capsys):
    module = _load_script()
    repo_root = tmp_path / "repo"
    reports_root = repo_root / "eval" / "knowledgeos" / "runs" / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    observation_payload = _observation_payload()
    (reports_root / "source_quality_observation_latest.json").write_text(
        json.dumps(observation_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_root / "source_quality_detail_observation_latest.json").write_text(
        json.dumps(_detail_observation_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def _fake_run_json_command(argv, *, cwd):  # noqa: ANN001
        calls.append(list(argv))
        if "run_agent_docs_writeback_loop.py" in str(argv[1]):
            return {"summary": {"applied": True, "targets": ["docs/status/today.md", "worklog/2026-04-19.md"], "receiptId": "receipt_1"}}
        return {"status": "ok"}

    monkeypatch.setattr(module, "_run_json_command", _fake_run_json_command)

    exit_code = module.main(
        [
            "--repo-root",
            str(repo_root),
            "--runs-root",
            str(repo_root / "eval" / "knowledgeos" / "runs"),
            "--writeback",
            "--apply-writeback",
            "--include-workspace",
            "--actor",
            "won",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["observationSummary"]["decision"] == "observe_more"
    assert payload["detailObservationSummary"]["decision"] == "not_ready_for_detail_gate_review"
    assert payload["writeback"]["summary"]["applied"] is True
    assert any("run_source_quality_battery.py" in " ".join(call) for call in calls)
    assert any("report_source_quality_trend.py" in " ".join(call) for call in calls)
    assert any("report_legacy_runtime_readiness.py" in " ".join(call) for call in calls)
    assert any("report_source_quality_observation.py" in " ".join(call) for call in calls)
    assert any("report_source_quality_detail_observation.py" in " ".join(call) for call in calls)
    assert any("run_agent_docs_writeback_loop.py" in " ".join(call) for call in calls)


def test_main_enforces_hard_gate_after_observation(monkeypatch, tmp_path: Path, capsys):
    module = _load_script()
    repo_root = tmp_path / "repo"
    reports_root = repo_root / "eval" / "knowledgeos" / "runs" / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    observation_payload = _observation_payload()
    observation_payload["decision"] = "ready_for_hard_gate_review"
    observation_payload["run_count"] = 7
    observation_payload["blockers"] = []
    (reports_root / "source_quality_observation_latest.json").write_text(
        json.dumps(observation_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_root / "source_quality_detail_observation_latest.json").write_text(
        json.dumps(_detail_observation_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def _fake_run_json_command(argv, *, cwd):  # noqa: ANN001
        _ = cwd
        calls.append(list(argv))
        if "check_source_quality_hard_gate.py" in str(argv[1]):
            return {"schema": "knowledge-hub.source-quality-hard-gate.result.v1", "status": "ok", "errors": []}
        return {"status": "ok"}

    monkeypatch.setattr(module, "_run_json_command", _fake_run_json_command)

    exit_code = module.main(
        [
            "--repo-root",
            str(repo_root),
            "--runs-root",
            str(repo_root / "eval" / "knowledgeos" / "runs"),
            "--enforce-hard-gate",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hardGate"]["status"] == "ok"
    assert payload["detailObservationSummary"]["decision"] == "not_ready_for_detail_gate_review"
    assert any("check_source_quality_hard_gate.py" in " ".join(call) for call in calls)


def test_run_json_command_accepts_prefixed_non_json_stdout(monkeypatch):
    module = _load_script()

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(  # noqa: ARG005
            returncode=0,
            stdout='Wrote paper default eval sheet: /tmp/paper.csv\n{\n  "status": "ok"\n}\n',
            stderr="",
        ),
    )

    payload = module._run_json_command(["python", "fake.py"], cwd=Path("/tmp"))

    assert payload["status"] == "ok"


def test_latest_run_local_date_uses_requested_timezone(tmp_path: Path):
    module = _load_script()
    runs_root = tmp_path / "runs"
    summary_path = runs_root / "source_quality_battery_20260419_154500" / "source_quality_battery_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"created_at": "2026-04-19T23:30:00+00:00"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    latest = module._latest_run_local_date(runs_root, tz_name="Asia/Seoul")

    assert latest is not None
    assert latest["localDate"] == "2026-04-20"
    assert latest["timezone"] == "Asia/Seoul"


def test_main_skips_when_local_date_already_has_run(monkeypatch, tmp_path: Path, capsys):
    module = _load_script()
    repo_root = tmp_path / "repo"
    runs_root = repo_root / "eval" / "knowledgeos" / "runs"
    reports_root = runs_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    latest_run_dir = runs_root / "source_quality_battery_20260420_010000"
    latest_run_dir.mkdir(parents=True, exist_ok=True)
    (latest_run_dir / "source_quality_battery_summary.json").write_text(
        json.dumps({"created_at": "2026-04-19T23:30:00+00:00"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_root / "source_quality_observation_latest.json").write_text(
        json.dumps({"latest_run_dir": str(latest_run_dir)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_root / "source_quality_detail_observation_latest.json").write_text(
        json.dumps(_detail_observation_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    class _FixedDateTime(module.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001
            base = cls.fromisoformat("2026-04-20T04:46:54+09:00")
            return base if tz is None else base.astimezone(tz)

    monkeypatch.setattr(module, "datetime", _FixedDateTime)

    exit_code = module.main(
        [
            "--repo-root",
            str(repo_root),
            "--runs-root",
            str(runs_root),
            "--skip-if-local-date-already-covered",
            "--local-timezone",
            "Asia/Seoul",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["skipped"] is True
    assert payload["skipReason"] == "already_ran_for_local_date"
    assert payload["localDate"] == "2026-04-20"
    assert payload["detailObservationSummary"]["decision"] == "not_ready_for_detail_gate_review"


def test_main_enforces_hard_gate_when_skipping_same_day(monkeypatch, tmp_path: Path, capsys):
    module = _load_script()
    repo_root = tmp_path / "repo"
    runs_root = repo_root / "eval" / "knowledgeos" / "runs"
    reports_root = runs_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    latest_run_dir = runs_root / "source_quality_battery_20260420_010000"
    latest_run_dir.mkdir(parents=True, exist_ok=True)
    (latest_run_dir / "source_quality_battery_summary.json").write_text(
        json.dumps({"created_at": "2026-04-19T23:30:00+00:00"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_root / "source_quality_observation_latest.json").write_text(
        json.dumps({"latest_run_dir": str(latest_run_dir)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_root / "source_quality_detail_observation_latest.json").write_text(
        json.dumps(_detail_observation_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def _fake_run_json_command(argv, *, cwd):  # noqa: ANN001
        _ = cwd
        calls.append(list(argv))
        return {"schema": "knowledge-hub.source-quality-hard-gate.result.v1", "status": "ok", "errors": []}

    class _FixedDateTime(module.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001
            base = cls.fromisoformat("2026-04-20T04:46:54+09:00")
            return base if tz is None else base.astimezone(tz)

    monkeypatch.setattr(module, "datetime", _FixedDateTime)
    monkeypatch.setattr(module, "_run_json_command", _fake_run_json_command)

    exit_code = module.main(
        [
            "--repo-root",
            str(repo_root),
            "--runs-root",
            str(runs_root),
            "--skip-if-local-date-already-covered",
            "--local-timezone",
            "Asia/Seoul",
            "--enforce-hard-gate",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["skipped"] is True
    assert payload["hardGate"]["status"] == "ok"
    assert payload["detailObservationSummary"]["decision"] == "not_ready_for_detail_gate_review"
    assert len(calls) == 1
    assert "check_source_quality_hard_gate.py" in " ".join(calls[0])
