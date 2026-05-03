from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from click.testing import CliRunner

from knowledge_hub.interfaces.cli.main import cli


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_release_smoke.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("check_release_smoke_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _command_lines(output: str) -> set[str]:
    commands: set[str] = set()
    in_commands = False
    for line in output.splitlines():
        if line.strip() == "Commands:":
            in_commands = True
            continue
        if not in_commands:
            continue
        if not line.startswith("  "):
            if line.strip():
                break
            continue
        stripped = line.strip()
        if stripped and not stripped.startswith("-"):
            commands.add(stripped.split()[0])
    return commands


def test_validate_doctor_payload_accepts_local_blocked_status():
    module = _load_script_module()

    validation = module.validate_doctor_payload(
        {
            "schema": "knowledge-hub.doctor.result.v1",
            "status": "blocked",
            "checks": [
                {"area": "settings", "status": "ok"},
                {"area": "Ollama", "status": "blocked"},
                {"area": "vector corpus", "status": "needs_setup"},
            ],
            "nextActions": ["ollama serve"],
            "warnings": ["vector corpus degraded: vector_corpus_empty"],
        }
    )

    assert validation.ok is True
    assert validation.details["doctorStatus"] == "blocked"


def test_run_command_marks_timeout_and_preserves_partial_output(monkeypatch):
    module = _load_script_module()

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=kwargs.get("args") or args[0],
            timeout=module.SMOKE_COMMAND_TIMEOUT_SEC,
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_command(["python", "-m", "knowledge_hub.interfaces.cli.main", "status"], env={}, cwd=Path("/tmp"), name="status")

    assert result.timed_out is True
    assert result.returncode == module.SMOKE_TIMEOUT_RETURN_CODE
    assert result.stdout == "partial stdout"
    assert result.stderr == "partial stderr"
    assert result.timeout_sec == module.SMOKE_COMMAND_TIMEOUT_SEC


def test_extract_trailing_json_object_parses_index_prelude():
    module = _load_script_module()

    payload = module.extract_trailing_json_object(
        "[dim]run_id=idx_123[/dim]\n"
        "[dim]failure report: /tmp/index-idx_123.json[/dim]\n"
        '{\n  "schema": "knowledge-hub.index.result.v1",\n  "status": "ok"\n}\n'
    )

    assert payload["schema"] == "knowledge-hub.index.result.v1"
    assert payload["status"] == "ok"


def test_validate_status_result_requires_runtime_markers():
    module = _load_script_module()
    result = module.CommandResult(
        name="status",
        argv=["python", "-m", "knowledge_hub.interfaces.cli.main", "status"],
        returncode=0,
        stdout="Knowledge Hub v0.1.0\n",
        stderr="",
        duration_sec=0.1,
    )

    validation = module.validate_status_result(result, config_path=Path("/tmp/config.yaml"))

    assert validation.ok is False
    assert any("Retrieval Runtime" in error for error in validation.errors)


def test_run_release_smoke_collects_full_plan_after_first_failure(monkeypatch):
    module = _load_script_module()
    calls: list[str] = []

    def fake_run_command(argv, *, env, cwd, name):
        calls.append(name)
        return module.CommandResult(
            name=name,
            argv=list(argv),
            returncode=0,
            stdout=f"{name} ok",
            stderr="",
            duration_sec=0.01,
        )

    def ok_validation(result, **_kwargs):
        return module.ValidationResult(ok=True, summary=f"{result.name} ok", details={"name": result.name}, errors=[])

    monkeypatch.setattr(module, "run_command", fake_run_command)
    monkeypatch.setattr(module, "validate_help_result", ok_validation)
    monkeypatch.setattr(module, "validate_status_result", ok_validation)
    monkeypatch.setattr(module, "validate_doctor_result", ok_validation)
    monkeypatch.setattr(module, "validate_invalid_command_result", ok_validation)
    monkeypatch.setattr(
        module,
        "validate_setup_result",
        lambda result, **_kwargs: module.ValidationResult(ok=False, summary="setup contract failed", details={"name": result.name}, errors=["setup failed"]),
    )

    payload = module.run_release_smoke()

    assert calls == [
        "top_help",
        "setup",
        "capture_help",
        "status",
        "doctor",
        "invalid_command",
    ]
    assert payload["status"] == "failed"
    assert payload["checkedCount"] == 6
    assert payload["passedCount"] == 5
    assert [item["name"] for item in payload["commands"]] == calls
    assert payload["commands"][1]["status"] == "failed"
    assert payload["commands"][-1]["status"] == "ok"


def test_run_weekly_core_loop_smoke_collects_full_plan_after_failure(monkeypatch):
    module = _load_script_module()
    calls: list[str] = []
    config_path = Path("/tmp/weekly-core-loop-config.yaml")
    note_path = Path("/tmp/weekly-core-loop-alpha.md")

    monkeypatch.setattr(
        module,
        "prepare_weekly_core_loop_fixture",
        lambda home_dir: (config_path, note_path),
    )

    outputs = {
        "top_help": module.CommandResult(
            name="top_help",
            argv=["python", "-m", "knowledge_hub.interfaces.cli.main", "--help"],
            returncode=0,
            stdout="Usage\nCommands:\n  discover\n  index\n  search\n  ask\n  doctor\n  status\n",
            stderr="",
            duration_sec=0.01,
        ),
        "doctor": module.CommandResult(
            name="doctor",
            argv=["python", "-m", "knowledge_hub.interfaces.cli.main", "doctor", "--json"],
            returncode=0,
            stdout=json.dumps(
                {
                    "schema": "knowledge-hub.doctor.result.v1",
                    "status": "needs_setup",
                    "checks": [
                        {"area": "settings", "status": "ok"},
                        {"area": "Ollama", "status": "ok"},
                        {"area": "vector corpus", "status": "needs_setup"},
                    ],
                    "nextActions": ["khub index --all"],
                    "warnings": [],
                }
            ),
            stderr="",
            duration_sec=0.01,
        ),
        "status": module.CommandResult(
            name="status",
            argv=["python", "-m", "knowledge_hub.interfaces.cli.main", "status"],
            returncode=0,
            stdout=f"Knowledge Hub v0.1.0\n설정: {config_path}\nRetrieval Runtime\nvector corpus\n",
            stderr="",
            duration_sec=0.01,
        ),
        "index": module.CommandResult(
            name="index",
            argv=["python", "-m", "knowledge_hub.interfaces.cli.main", "index", "--vault-all", "--json"],
            returncode=0,
            stdout=(
                "[dim]run_id=idx_123[/dim]\n"
                '{\n'
                '  "schema": "knowledge-hub.index.result.v1",\n'
                '  "status": "ok",\n'
                '  "vectorDbCount": 1,\n'
                '  "processedBreakdown": {"vault": {"processed": 1, "succeeded": 1, "failed": 0}},\n'
                '  "reportPath": "/tmp/index-idx_123.json"\n'
                '}\n'
            ),
            stderr="",
            duration_sec=0.02,
            timeout_sec=module.CORE_LOOP_COMMAND_TIMEOUT_SEC,
        ),
        "ask": module.CommandResult(
            name="ask",
            argv=["python", "-m", "knowledge_hub.interfaces.cli.main", "ask", "alpha retrieval", "--json"],
            returncode=0,
            stdout=json.dumps(
                {
                    "schema": "knowledge-hub.ask.result.v1",
                    "status": "ok",
                    "question": "alpha retrieval",
                    "answer": "Alpha retrieval is a grounded vault smoke note.",
                    "allowExternal": False,
                    "citations": [
                        {
                            "label": "S1",
                            "title": "Alpha retrieval note",
                            "source_type": "vault",
                            "target": "alpha.md",
                            "kind": "file",
                        }
                    ],
                    "sources": [
                        {
                            "title": "Alpha retrieval note",
                            "source_type": "vault",
                        }
                    ],
                    "warnings": [
                        "answer verification used heuristic fallback",
                        "answer rewrite skipped: rewrite route unavailable",
                    ],
                }
            ),
            stderr="",
            duration_sec=0.02,
            timeout_sec=module.CORE_LOOP_COMMAND_TIMEOUT_SEC,
        ),
        "search": module.CommandResult(
            name="search",
            argv=["python", "-m", "knowledge_hub.interfaces.cli.main", "search", "alpha retrieval", "--json"],
            returncode=0,
            stdout=json.dumps(
                {
                    "schema": "knowledge-hub.search.result.v1",
                    "query": "alpha retrieval",
                    "results": [
                        {
                            "title": "Alpha retrieval note",
                            "sourceType": "vault",
                        }
                    ],
                }
            ),
            stderr="",
            duration_sec=0.02,
            timeout_sec=module.CORE_LOOP_COMMAND_TIMEOUT_SEC,
        ),
    }

    def fake_run_command(argv, *, env, cwd, name, timeout_sec=module.SMOKE_COMMAND_TIMEOUT_SEC):
        calls.append(name)
        return outputs[name]

    monkeypatch.setattr(module, "run_command", fake_run_command)

    payload = module.run_weekly_core_loop_smoke()

    assert calls == ["top_help", "doctor", "status", "index", "ask", "search"]
    assert payload["status"] == "ok"
    assert payload["checkedCount"] == 6
    assert payload["passedCount"] == 6
    assert payload["mode"] == "weekly_core_loop"


def test_validate_help_result_requires_surface_markers():
    module = _load_script_module()
    result = module.CommandResult(
        name="top_help",
        argv=["python", "-m", "knowledge_hub.interfaces.cli.main", "--help"],
        returncode=0,
        stdout="Usage\nCommands:\n  doctor\n",
        stderr="",
        duration_sec=0.1,
    )

    validation = module.validate_help_result(
        result,
        required_markers=("Commands:", "doctor", "status"),
        summary="top-level help surface is present",
    )

    assert validation.ok is False
    assert any("status" in error for error in validation.errors)


def test_top_level_help_hides_operator_surfaces():
    runner = CliRunner()

    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    command_lines = _command_lines(result.output)
    for token in ("dinger", "os", "eval", "paper-memory", "math-memory", "vector-compare", "vector-restore"):
        assert token not in command_lines
    for token in ("discover", "index", "search", "ask", "doctor", "status", "paper", "labs"):
        assert token in command_lines


def test_hidden_eval_compat_alias_remains_directly_invokable():
    runner = CliRunner()

    root_result = runner.invoke(cli, ["eval", "--help"])
    labs_result = runner.invoke(cli, ["labs", "eval", "--help"])

    assert root_result.exit_code == 0
    assert labs_result.exit_code == 0
    assert "Compatibility alias" in root_result.output
    assert "answer-loop" in root_result.output
    assert _command_lines(root_result.output) == _command_lines(labs_result.output)


def test_labs_help_keeps_eval_visible():
    runner = CliRunner()

    result = runner.invoke(cli, ["labs", "--help"])

    assert result.exit_code == 0
    assert "eval" in _command_lines(result.output)


def test_labs_help_keeps_rag_visible():
    runner = CliRunner()

    result = runner.invoke(cli, ["labs", "--help"])

    assert result.exit_code == 0
    assert "rag" in _command_lines(result.output)


def test_labs_help_keeps_foundry_visible():
    runner = CliRunner()

    result = runner.invoke(cli, ["labs", "--help"])

    assert result.exit_code == 0
    assert "foundry" in _command_lines(result.output)


def test_os_help_hides_low_level_record_groups():
    runner = CliRunner()

    result = runner.invoke(cli, ["os", "--help"])

    assert result.exit_code == 0
    command_lines = _command_lines(result.output)
    for token in ("goal", "decision"):
        assert token not in command_lines
    for token in ("capture", "decide", "project", "evidence", "task", "inbox", "next"):
        assert token in command_lines


def test_hidden_os_record_groups_remain_directly_invokable():
    runner = CliRunner()

    result = runner.invoke(cli, ["os", "goal", "--help"])

    assert result.exit_code == 0
    assert "Goal records" in result.output


def test_dinger_help_hides_operator_utility_commands():
    runner = CliRunner()

    result = runner.invoke(cli, ["dinger", "--help"])

    assert result.exit_code == 0
    command_lines = _command_lines(result.output)
    for token in ("capture-http", "recent", "lint"):
        assert token not in command_lines
    for token in ("ingest", "ask", "capture", "file", "capture-process"):
        assert token in command_lines


def test_hidden_dinger_operator_utility_commands_remain_directly_invokable():
    runner = CliRunner()

    result = runner.invoke(cli, ["dinger", "recent", "--help"])

    assert result.exit_code == 0
    assert "--limit" in result.output


def test_validate_invalid_command_result_requires_non_zero_error():
    module = _load_script_module()
    result = module.CommandResult(
        name="invalid_command",
        argv=["python", "-m", "knowledge_hub.interfaces.cli.main", "missing"],
        returncode=1,
        stdout="오류: No such command 'missing'.\n",
        stderr="",
        duration_sec=0.1,
    )

    validation = module.validate_invalid_command_result(result)

    assert validation.ok is True
    assert validation.details["returncode"] == 1


def test_validate_invalid_command_result_rejects_traceback_output():
    module = _load_script_module()
    result = module.CommandResult(
        name="invalid_command",
        argv=["python", "-m", "knowledge_hub.interfaces.cli.main", "missing"],
        returncode=1,
        stdout="",
        stderr="예상치 못한 오류\nTraceback (most recent call last):\nNo such command 'missing'.\n",
        duration_sec=0.1,
    )

    validation = module.validate_invalid_command_result(result)

    assert validation.ok is False
    assert any("Traceback" in error for error in validation.errors)
    assert any("예상치 못한 오류" in error for error in validation.errors)


def test_payload_exit_code_reflects_pass_fail_behavior():
    module = _load_script_module()

    assert module.payload_exit_code({"status": "ok"}) == 0
    assert module.payload_exit_code({"status": "failed"}) == 1


def test_release_smoke_script_passes_with_local_contract():
    module = _load_script_module()
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=str(SCRIPT.parent.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["checkedCount"] == 6
    assert payload["passedCount"] == 6
    assert [item["name"] for item in payload["commands"]] == [
        "top_help",
        "setup",
        "capture_help",
        "status",
        "doctor",
        "invalid_command",
    ]
    assert all(item["status"] == "ok" for item in payload["commands"])
    assert all(item["timedOut"] is False for item in payload["commands"])
    assert all(item["timeoutSec"] == module.SMOKE_COMMAND_TIMEOUT_SEC for item in payload["commands"])
    doctor = next(item for item in payload["commands"] if item["name"] == "doctor")
    assert doctor["details"]["doctorStatus"] in {"ok", "blocked", "degraded", "needs_setup"}
    invalid_command = next(item for item in payload["commands"] if item["name"] == "invalid_command")
    assert invalid_command["details"]["returncode"] != 0
    invalid_output = f"{invalid_command['stdout']}\n{invalid_command['stderr']}"
    assert "No such command" in invalid_output
    assert "Traceback" not in invalid_output
    assert "예상치 못한 오류" not in invalid_output


def test_validate_index_result_requires_non_empty_vault_success():
    module = _load_script_module()
    result = module.CommandResult(
        name="index",
        argv=["python", "-m", "knowledge_hub.interfaces.cli.main", "index", "--vault-all", "--json"],
        returncode=0,
        stdout='{"schema":"knowledge-hub.index.result.v1","status":"ok","vectorDbCount":0,"processedBreakdown":{"vault":{"processed":0,"succeeded":0,"failed":0}}}',
        stderr="",
        duration_sec=0.1,
        timeout_sec=module.CORE_LOOP_COMMAND_TIMEOUT_SEC,
    )

    validation = module.validate_index_result(result)

    assert validation.ok is False
    assert any("vector documents" in error for error in validation.errors)


def test_validate_ask_result_requires_grounded_vault_output():
    module = _load_script_module()
    result = module.CommandResult(
        name="ask",
        argv=["python", "-m", "knowledge_hub.interfaces.cli.main", "ask", "alpha retrieval", "--json"],
        returncode=0,
        stdout=json.dumps(
            {
                "schema": "knowledge-hub.ask.result.v1",
                "status": "ok",
                "question": "alpha retrieval",
                "answer": "Alpha retrieval is a grounded vault smoke note.",
                "allowExternal": False,
                "citations": [{"source_type": "vault", "target": "alpha.md"}],
                "sources": [],
                "warnings": [
                    "answer verification used heuristic fallback",
                    "answer rewrite skipped: rewrite route unavailable",
                ],
            }
        ),
        stderr="",
        duration_sec=0.1,
        timeout_sec=module.CORE_LOOP_COMMAND_TIMEOUT_SEC,
    )

    validation = module.validate_ask_result(result)

    assert validation.ok is True
    assert validation.details["hasVaultCitation"] is True
    assert validation.details["warningCount"] == 2
