from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_release_smoke.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("check_release_smoke_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
