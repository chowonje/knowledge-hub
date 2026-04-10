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


def test_release_smoke_script_passes_with_local_contract():
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
    assert [item["name"] for item in payload["commands"]] == [
        "top_help",
        "setup",
        "capture_help",
        "status",
        "doctor",
        "invalid_command",
    ]
    assert all(item["status"] == "ok" for item in payload["commands"])
    doctor = next(item for item in payload["commands"] if item["name"] == "doctor")
    assert doctor["details"]["doctorStatus"] in {"ok", "blocked", "degraded", "needs_setup"}
    invalid_command = next(item for item in payload["commands"] if item["name"] == "invalid_command")
    assert invalid_command["details"]["returncode"] != 0
