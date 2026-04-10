#!/usr/bin/env python3
"""Run a narrow, repeatable CLI smoke gate for local release trust.

This gate is intentionally narrower than the approval verification set:
- no hosted provider dependency
- no full pytest sweep
- expected local blocked/degraded results still pass when they match contract
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import yaml


DOCTOR_ALLOWED_STATUSES = {"ok", "blocked", "degraded", "needs_setup"}
STATUS_REQUIRED_MARKERS = ("Knowledge Hub v", "Retrieval Runtime", "vector corpus")
DOCTOR_REQUIRED_AREAS = {"settings", "Ollama", "vector corpus"}
TOP_HELP_REQUIRED_MARKERS = ("Commands:", "doctor", "status", "setup")
CAPTURE_HELP_REQUIRED_MARKERS = ("Commands:", "cleanup", "requeue", "status")
INVALID_COMMAND_REQUIRED_MARKER = "No such command"


@dataclass
class CommandResult:
    name: str
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float


@dataclass
class ValidationResult:
    ok: bool
    summary: str
    details: dict[str, Any]
    errors: list[str]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def temp_config_path(home_dir: Path) -> Path:
    return home_dir / ".khub" / "config.yaml"


def build_env(home_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["HOME"] = str(home_dir)
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    env["COLUMNS"] = "120"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_command(argv: list[str], *, env: dict[str, str], cwd: Path, name: str) -> CommandResult:
    started = time.monotonic()
    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return CommandResult(
        name=name,
        argv=list(argv),
        returncode=int(proc.returncode),
        stdout=str(proc.stdout or ""),
        stderr=str(proc.stderr or ""),
        duration_sec=round(time.monotonic() - started, 3),
    )


def validate_setup_result(result: CommandResult, *, config_path: Path) -> ValidationResult:
    errors: list[str] = []
    config_data: dict[str, Any] = {}
    if result.returncode != 0:
        errors.append(f"setup exited with {result.returncode}")
    if not config_path.exists():
        errors.append(f"setup did not write config: {config_path}")
    else:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            errors.append("setup config is not a mapping")
        else:
            config_data = loaded
            if loaded.get("translation", {}).get("provider") != "ollama":
                errors.append("translation provider is not ollama")
            if loaded.get("summarization", {}).get("provider") != "ollama":
                errors.append("summarization provider is not ollama")
            if loaded.get("embedding", {}).get("provider") != "ollama":
                errors.append("embedding provider is not ollama")
            if loaded.get("paper", {}).get("summary", {}).get("parser") != "auto":
                errors.append("paper.summary.parser is not auto")
    if "프로필: local" not in result.stdout:
        errors.append("setup output does not confirm local profile")
    summary = "local setup profile created" if not errors else "setup contract failed"
    return ValidationResult(
        ok=not errors,
        summary=summary,
        details={
            "configPath": str(config_path),
            "translationProvider": config_data.get("translation", {}).get("provider"),
            "summarizationProvider": config_data.get("summarization", {}).get("provider"),
            "embeddingProvider": config_data.get("embedding", {}).get("provider"),
            "paperSummaryParser": config_data.get("paper", {}).get("summary", {}).get("parser"),
        },
        errors=errors,
    )


def validate_status_result(result: CommandResult, *, config_path: Path) -> ValidationResult:
    errors: list[str] = []
    if result.returncode != 0:
        errors.append(f"status exited with {result.returncode}")
    for marker in STATUS_REQUIRED_MARKERS:
        if marker not in result.stdout:
            errors.append(f"status output missing marker: {marker}")
    if str(config_path) not in result.stdout:
        errors.append("status output does not show the isolated config path")
    summary = "status surface rendered runtime diagnostics" if not errors else "status contract failed"
    return ValidationResult(
        ok=not errors,
        summary=summary,
        details={"configPath": str(config_path)},
        errors=errors,
    )


def validate_doctor_payload(payload: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    status = str(payload.get("status") or "")
    if str(payload.get("schema") or "") != "knowledge-hub.doctor.result.v1":
        errors.append("doctor schema mismatch")
    if status not in DOCTOR_ALLOWED_STATUSES:
        errors.append(f"doctor status not allowed for local smoke: {status or 'missing'}")
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("doctor checks missing")
        checks = []
    areas = {str(item.get("area") or "") for item in checks if isinstance(item, dict)}
    missing_areas = sorted(DOCTOR_REQUIRED_AREAS - areas)
    if missing_areas:
        errors.append(f"doctor checks missing required areas: {', '.join(missing_areas)}")
    next_actions = payload.get("nextActions")
    if not isinstance(next_actions, list):
        errors.append("doctor nextActions missing")
    summary = f"doctor returned accepted local status={status}" if not errors else "doctor contract failed"
    return ValidationResult(
        ok=not errors,
        summary=summary,
        details={
            "doctorStatus": status,
            "checkAreas": sorted(areas),
            "warningCount": len(payload.get("warnings") or []),
        },
        errors=errors,
    )


def validate_doctor_result(result: CommandResult) -> ValidationResult:
    errors: list[str] = []
    payload: dict[str, Any] = {}
    if result.returncode != 0:
        errors.append(f"doctor exited with {result.returncode}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        errors.append(f"doctor did not emit valid json: {exc}")
    if errors:
        return ValidationResult(
            ok=False,
            summary="doctor contract failed",
            details={},
            errors=errors,
        )
    payload_validation = validate_doctor_payload(payload)
    return ValidationResult(
        ok=payload_validation.ok,
        summary=payload_validation.summary,
        details={"payload": payload, **payload_validation.details},
        errors=payload_validation.errors,
    )


def validate_help_result(result: CommandResult, *, required_markers: tuple[str, ...], summary: str) -> ValidationResult:
    errors: list[str] = []
    if result.returncode != 0:
        errors.append(f"{result.name} exited with {result.returncode}")
    haystack = result.stdout
    for marker in required_markers:
        if marker not in haystack:
            errors.append(f"{result.name} output missing marker: {marker}")
    return ValidationResult(
        ok=not errors,
        summary=summary if not errors else f"{result.name} contract failed",
        details={"markers": list(required_markers)},
        errors=errors,
    )


def validate_invalid_command_result(result: CommandResult) -> ValidationResult:
    errors: list[str] = []
    if result.returncode == 0:
        errors.append("invalid command exited with 0")
    combined = f"{result.stdout}\n{result.stderr}"
    if INVALID_COMMAND_REQUIRED_MARKER not in combined:
        errors.append(f"invalid command output missing marker: {INVALID_COMMAND_REQUIRED_MARKER}")
    return ValidationResult(
        ok=not errors,
        summary="invalid command exits non-zero with a user-facing error" if not errors else "invalid command contract failed",
        details={"returncode": result.returncode},
        errors=errors,
    )


def run_release_smoke(*, keep_temp_dir: bool = False) -> dict[str, Any]:
    root = repo_root()
    cli_argv = [sys.executable, "-m", "knowledge_hub.interfaces.cli.main"]
    with tempfile.TemporaryDirectory(prefix="khub-release-smoke-") as tmp_home_str:
        home_dir = Path(tmp_home_str)
        env = build_env(home_dir)
        config_path = temp_config_path(home_dir)
        plan = [
            ("top_help", cli_argv + ["--help"]),
            ("setup", cli_argv + ["setup", "--quick", "--non-interactive"]),
            ("capture_help", cli_argv + ["dinger", "capture", "--help"]),
            ("status", cli_argv + ["--config", str(config_path), "status"]),
            ("doctor", cli_argv + ["--config", str(config_path), "doctor", "--json"]),
            ("invalid_command", cli_argv + ["definitely-missing"]),
        ]
        validations: list[dict[str, Any]] = []
        failed = False

        for name, argv in plan:
            result = run_command(argv, env=env, cwd=root, name=name)
            if name == "top_help":
                validation = validate_help_result(
                    result,
                    required_markers=TOP_HELP_REQUIRED_MARKERS,
                    summary="top-level help surface is present",
                )
            elif name == "setup":
                validation = validate_setup_result(result, config_path=config_path)
            elif name == "capture_help":
                validation = validate_help_result(
                    result,
                    required_markers=CAPTURE_HELP_REQUIRED_MARKERS,
                    summary="dinger capture help surface is present",
                )
            elif name == "status":
                validation = validate_status_result(result, config_path=config_path)
            elif name == "doctor":
                validation = validate_doctor_result(result)
            else:
                validation = validate_invalid_command_result(result)
            validations.append(
                {
                    "name": name,
                    "argv": result.argv,
                    "returncode": result.returncode,
                    "durationSec": result.duration_sec,
                    "status": "ok" if validation.ok else "failed",
                    "summary": validation.summary,
                    "details": validation.details,
                    "errors": validation.errors,
                    "stdout": result.stdout if not validation.ok else "",
                    "stderr": result.stderr if result.stderr else "",
                }
            )
            if not validation.ok:
                failed = True
                break

        payload = {
            "status": "ok" if not failed else "failed",
            "repoRoot": str(root),
            "tempHome": str(home_dir) if (keep_temp_dir or failed) else "",
            "commands": validations,
        }
        if keep_temp_dir and not failed:
            persisted_home = root / ".tmp_release_smoke_home"
            if persisted_home.exists():
                raise RuntimeError(f"refusing to overwrite existing temp dir: {persisted_home}")
            persisted_home.mkdir(parents=True, exist_ok=False)
            for source in home_dir.rglob("*"):
                relative = source.relative_to(home_dir)
                target = persisted_home / relative
                if source.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
            payload["tempHome"] = str(persisted_home)
        return payload


def print_human(payload: dict[str, Any]) -> None:
    print("[release smoke]")
    for item in payload.get("commands", []):
        mark = "PASS" if item.get("status") == "ok" else "FAIL"
        print(f"{mark} {item.get('name')} {item.get('durationSec')}s {item.get('summary')}")
        for error in item.get("errors", []):
            print(f"  - {error}")
    print(f"release smoke: {payload.get('status')}")
    temp_home = str(payload.get("tempHome") or "")
    if temp_home:
        print(f"temp HOME: {temp_home}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a narrow local CLI smoke gate for release trust.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--keep-temp-dir",
        action="store_true",
        help="Keep the generated temp HOME under .tmp_release_smoke_home for inspection.",
    )
    args = parser.parse_args()

    payload = run_release_smoke(keep_temp_dir=bool(args.keep_temp_dir))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_human(payload)
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
