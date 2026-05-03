"""Shared bridge helpers for Foundry subprocess execution.

Authority contract:
- Python is the final authority for validation, policy gating, normalized bridge
  payload acceptance, and product-facing CLI/MCP outputs.
- TypeScript is the authority for delegated runtime orchestration and
  `.khub/personal-foundry/*` state inside `foundry-core`.
- The bridge is subprocess + JSON stdout only. Python may retry outer bridge
  calls, but TypeScript-to-Python inner bridge calls must not add their own
  retry loop.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FOUNDRY_SCRIPT = PROJECT_ROOT / "foundry-core" / "src" / "cli-agent.ts"
FOUNDRY_DIST_SCRIPT = PROJECT_ROOT / "foundry-core" / "dist" / "cli-agent.js"
FOUNDRY_PROJECT_SCRIPT = PROJECT_ROOT / "foundry-core" / "src" / "personal-foundry" / "project-cli.ts"
FOUNDRY_PROJECT_DIST_SCRIPT = PROJECT_ROOT / "foundry-core" / "dist" / "personal-foundry" / "project-cli.js"
FOUNDRY_LOCAL_BIN_DIR = PROJECT_ROOT / "foundry-core" / "node_modules" / ".bin"
FOUNDRY_LOCAL_TSX_BIN = FOUNDRY_LOCAL_BIN_DIR / "tsx"
DEFAULT_FOUNDRY_RETRY_ATTEMPTS = 2


def _foundry_retry_count(default: int = DEFAULT_FOUNDRY_RETRY_ATTEMPTS) -> int:
    raw = os.getenv("KHUB_FOUNDRY_RETRY_ATTEMPTS", str(default))
    try:
        value = int((raw or "").strip())
    except Exception:
        return default
    if value < 1:
        return 1
    return min(value, 8)


def coerce_json_output(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _bridge_candidates(
    script_args: Sequence[str],
    *,
    script_path: Path,
    dist_path: Path,
) -> list[list[str]]:
    script_exists = script_path.exists()
    dist_exists = dist_path.exists()
    if not script_exists and not dist_exists:
        return []

    candidates: list[list[str]] = []
    if script_exists:
        candidates.append(["node", str(script_path), *script_args])
        local_tsx = str(FOUNDRY_LOCAL_TSX_BIN) if FOUNDRY_LOCAL_TSX_BIN.exists() else None
        if local_tsx is not None:
            candidates.append([local_tsx, str(script_path), *script_args])
        global_tsx = shutil.which("tsx")
        if global_tsx and global_tsx != local_tsx:
            candidates.append([global_tsx, str(script_path), *script_args])
        if shutil.which("ts-node"):
            candidates.append(["ts-node", str(script_path), *script_args])
        if shutil.which("npx"):
            candidates.append(["npx", "tsx", str(script_path), *script_args])
    if dist_exists:
        candidates.append(["node", str(dist_path), *script_args])
    return candidates


def run_cli_command(
    command: str,
    command_args: Sequence[str],
    timeout_sec: int = 120,
) -> tuple[dict[str, Any] | None, str | None]:
    script_args = [str(PROJECT_ROOT), "python", command, *[str(value) for value in command_args]]
    candidates = _bridge_candidates(
        script_args,
        script_path=FOUNDRY_SCRIPT,
        dist_path=FOUNDRY_DIST_SCRIPT,
    )
    if not candidates:
        return None, "foundry-core cli-agent.ts not found"

    last_error = "foundry-core bridge execution failed"
    for candidate in candidates:
        try:
            result = subprocess.run(
                candidate,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(30, int(timeout_sec)),
                cwd=str(PROJECT_ROOT),
            )
        except FileNotFoundError:
            continue
        except Exception as error:
            last_error = f"foundry bridge exception: {error}"
            continue

        stdout = str(result.stdout or "").strip()
        stderr = str(result.stderr or "").strip()
        if result.returncode == 0:
            parsed = coerce_json_output(stdout)
            if parsed is not None:
                return parsed, None
            if stdout:
                return {"raw": stdout}, None
            last_error = "foundry bridge returned empty output"
            continue

        if stderr:
            last_error = stderr.splitlines()[-1]
        else:
            last_error = f"foundry bridge failed with code {result.returncode}"

    return None, last_error


def run_foundry_cli(
    command: str,
    command_args: Sequence[str],
    timeout_sec: int = 120,
) -> tuple[dict[str, Any] | None, str | None]:
    return run_cli_command(command, command_args, timeout_sec=timeout_sec)


def run_foundry_project_cli(
    command_args: Sequence[str],
    timeout_sec: int = 120,
) -> tuple[dict[str, Any] | None, str | None]:
    script_args = [str(PROJECT_ROOT), "python", *[str(value) for value in command_args]]
    candidates = _bridge_candidates(
        script_args,
        script_path=FOUNDRY_PROJECT_SCRIPT,
        dist_path=FOUNDRY_PROJECT_DIST_SCRIPT,
    )
    if not candidates:
        return None, "foundry-core personal-foundry project-cli.ts not found"

    last_error = "foundry-core personal-foundry bridge execution failed"
    for candidate in candidates:
        try:
            result = subprocess.run(
                candidate,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(30, int(timeout_sec)),
                cwd=str(PROJECT_ROOT),
            )
        except FileNotFoundError:
            continue
        except Exception as error:
            last_error = f"foundry project bridge exception: {error}"
            continue

        stdout = str(result.stdout or "").strip()
        stderr = str(result.stderr or "").strip()
        if result.returncode == 0:
            parsed = coerce_json_output(stdout)
            if parsed is not None:
                return parsed, None
            if stdout:
                return {"raw": stdout}, None
            last_error = "foundry project bridge returned empty output"
            continue

        if stderr:
            last_error = stderr.splitlines()[-1]
        else:
            last_error = f"foundry project bridge failed with code {result.returncode}"
    return None, last_error


def run_foundry_agent_goal(
    *,
    goal: str,
    max_rounds: int,
    dry_run: bool,
    dump_json: bool = False,
    role: str | None = None,
    report_path: str | None = None,
    orchestrator_mode: str | None = None,
    repo_path: str | None = None,
    include_workspace: bool | None = None,
    max_workspace_files: int | None = None,
    timeout_sec: int = 90,
) -> tuple[str | None, str | None]:
    script_args = [
        str(PROJECT_ROOT),
        "python",
        "run",
        "--goal",
        goal,
        "--max-rounds",
        str(max_rounds),
        "--role",
        (role or "planner"),
        "--orchestrator-mode",
        (orchestrator_mode or "adaptive"),
    ]
    if report_path:
        script_args.extend(["--report-path", report_path])
    if repo_path:
        script_args.extend(["--repo-path", repo_path])
    if include_workspace is True:
        script_args.append("--include-workspace")
    elif include_workspace is False:
        script_args.append("--no-include-workspace")
    if max_workspace_files:
        script_args.extend(["--max-workspace-files", str(max_workspace_files)])
    if dry_run:
        script_args.append("--dry-run")
    if dump_json:
        script_args.append("--dump-json")

    candidates = _bridge_candidates(
        script_args,
        script_path=FOUNDRY_SCRIPT,
        dist_path=FOUNDRY_DIST_SCRIPT,
    )
    if not candidates:
        return None, "foundry-core cli-agent.ts not found"

    last_error = "foundry-core execution failed"
    for _ in range(_foundry_retry_count()):
        for candidate in candidates:
            try:
                result = subprocess.run(
                    candidate,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=max(30, int(timeout_sec)),
                    cwd=str(PROJECT_ROOT),
                )
            except FileNotFoundError:
                continue
            except Exception as error:
                last_error = f"foundry bridge exception: {error}"
                continue

            stdout = str(result.stdout or "").strip()
            stderr = str(result.stderr or "").strip()
            if result.returncode == 0 and stdout:
                return stdout, None
            if stderr:
                last_error = f"foundry bridge error: {stderr.splitlines()[-1]}"
            elif result.returncode != 0:
                last_error = f"foundry bridge failed with code {result.returncode}"
    return None, last_error
