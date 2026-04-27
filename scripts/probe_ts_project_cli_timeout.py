#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from knowledge_hub.application.agent.foundry_bridge import (
    FOUNDRY_PROJECT_DIST_SCRIPT,
    FOUNDRY_PROJECT_SCRIPT,
    PROJECT_ROOT,
    _bridge_candidates,
    coerce_json_output,
)


def _candidate_label(candidate: list[str]) -> str:
    if candidate[:2] == ["npx", "tsx"]:
        return "npx-tsx-source"
    if candidate and candidate[0] == "tsx":
        return "tsx-source"
    if candidate and candidate[0] == "ts-node":
        return "ts-node-source"
    if len(candidate) >= 2 and candidate[0] == "node" and candidate[1].endswith(".ts"):
        return "node-ts-source"
    if len(candidate) >= 2 and candidate[0] == "node" and candidate[1].endswith(".js"):
        return "node-dist"
    return candidate[0] if candidate else "unknown"


def _output_excerpt(raw: str | bytes | None, limit: int = 240) -> str:
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw or "")
    collapsed = " | ".join(line.strip() for line in text.strip().splitlines() if line.strip())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3]}..."


def _stderr_summary(raw: str | bytes | None) -> str:
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw or "")
    lines = [line for line in text.strip().splitlines() if line.strip()]
    for line in reversed(lines):
        if "ERR_" in line or "Error" in line or "Cannot find module" in line:
            return line
    return lines[-1] if lines else ""


def _error_summary(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    combined = "\n".join(part for part in [_stderr_summary(stderr), _output_excerpt(stdout)] if part)
    if not combined:
        return ""
    for pattern in ("Cannot find module", "ModuleNotFoundError", "ERR_UNKNOWN_FILE_EXTENSION", "Error", "ERR_"):
        if pattern in combined:
            return combined
    return combined


def _failure_category(*, returncode: int | str, payload: dict[str, Any] | None, error_summary: str) -> str:
    if returncode == "missing-binary":
        return "missing-binary"
    if returncode == "timeout":
        return "timeout"
    if "Cannot find module" in error_summary or "ModuleNotFoundError" in error_summary:
        return "module-resolution-failure"
    if "ERR_UNKNOWN_FILE_EXTENSION" in error_summary:
        return "ts-loader-failure"
    if isinstance(returncode, int) and returncode == 0 and payload is None:
        return "no-json-payload"
    if isinstance(returncode, int) and returncode == 0:
        return "ok"
    return "nonzero-exit"


def _tool_probe(command: list[str], *, cwd: Path, timeout_sec: float = 5.0) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=timeout_sec,
        )
    except FileNotFoundError:
        return {
            "command": command,
            "resolved_binary": shutil.which(command[0]) if command else None,
            "returncode": "missing-binary",
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "stdout": "",
            "stderr": "",
        }
    except subprocess.TimeoutExpired as error:
        return {
            "command": command,
            "resolved_binary": shutil.which(command[0]) if command else None,
            "returncode": "timeout",
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "stdout": _output_excerpt(error.stdout),
            "stderr": _output_excerpt(error.stderr),
        }

    return {
        "command": command,
        "resolved_binary": shutil.which(command[0]) if command else None,
        "returncode": result.returncode,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "stdout": _output_excerpt(result.stdout),
        "stderr": _output_excerpt(result.stderr),
    }


def _npx_tsx_candidate(script_args: list[str]) -> list[str]:
    for candidate in _bridge_candidates(
        script_args,
        script_path=FOUNDRY_PROJECT_SCRIPT,
        dist_path=FOUNDRY_PROJECT_DIST_SCRIPT,
    ):
        if candidate[:2] == ["npx", "tsx"]:
            return candidate
    raise RuntimeError("`npx tsx` candidate is unavailable in the current bridge candidate set")


def _run_probe(
    *,
    label: str,
    cwd: Path,
    candidate: list[str],
    timeout_sec: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            candidate,
            check=False,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=timeout_sec,
        )
    except FileNotFoundError as error:
        error_summary = str(error)
        return {
            "kind": "probe-result",
            "label": label,
            "cwd": str(cwd),
            "candidate_label": _candidate_label(candidate),
            "argv": candidate,
            "returncode": "missing-binary",
            "category": "missing-binary",
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "stdout_kind": None,
            "stdout_schema": None,
            "stdout_status": None,
            "stdout_bytes": 0,
            "stdout_excerpt": "",
            "stderr_tail": error_summary,
            "error_summary": error_summary,
        }
    except subprocess.TimeoutExpired as error:
        error_text = _error_summary(error.stdout, error.stderr) or "timed out"
        return {
            "kind": "probe-result",
            "label": label,
            "cwd": str(cwd),
            "candidate_label": _candidate_label(candidate),
            "argv": candidate,
            "returncode": "timeout",
            "category": "timeout",
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "stdout_kind": None,
            "stdout_schema": None,
            "stdout_status": None,
            "stdout_bytes": len(error.stdout or ""),
            "stdout_excerpt": _output_excerpt(error.stdout),
            "stderr_tail": _stderr_summary(error.stderr),
            "error_summary": error_text,
        }

    payload = coerce_json_output(result.stdout or "")
    error_text = _error_summary(result.stdout, result.stderr)
    return {
        "kind": "probe-result",
        "label": label,
        "cwd": str(cwd),
        "candidate_label": _candidate_label(candidate),
        "argv": candidate,
        "returncode": result.returncode,
        "category": _failure_category(returncode=result.returncode, payload=payload, error_summary=error_text),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "stdout_kind": payload.get("kind") if isinstance(payload, dict) else None,
        "stdout_schema": payload.get("schema") if isinstance(payload, dict) else None,
        "stdout_status": payload.get("status") if isinstance(payload, dict) else None,
        "stdout_bytes": len(result.stdout or ""),
        "stdout_excerpt": _output_excerpt(result.stdout),
        "stderr_tail": _stderr_summary(result.stderr),
        "error_summary": error_text,
    }


def _run_tsx_eval_matrix(*, repo_root: Path, foundry_root: Path, timeout_sec: float, sink: list[str]) -> None:
    eval_candidate = [
        "npx",
        "tsx",
        "--eval",
        'console.log(JSON.stringify({"kind":"tsx-eval-ok"}))',
    ]
    _emit(
        {
            "kind": "candidate",
            "command": "tsx --eval",
            "argv": eval_candidate,
        },
        sink=sink,
    )
    _emit(
        _run_probe(
            label="tsx-eval-repo-root-cwd",
            cwd=repo_root,
            candidate=eval_candidate,
            timeout_sec=timeout_sec,
        ),
        sink=sink,
    )
    _emit(
        _run_probe(
            label="tsx-eval-foundry-core-cwd",
            cwd=foundry_root,
            candidate=eval_candidate,
            timeout_sec=timeout_sec,
        ),
        sink=sink,
    )


def _run_python_module_matrix(*, repo_root: Path, foundry_root: Path, timeout_sec: float, sink: list[str]) -> None:
    candidate = [
        "python",
        "-c",
        (
            "import importlib.util, json, os, sys; "
            "module='knowledge_hub.interfaces.cli.main'; "
            "spec=importlib.util.find_spec(module); "
            "print(json.dumps({'kind':'python-module-probe','module':module,'found': spec is not None,"
            "'cwd': os.getcwd(),'path0': sys.path[0]}))"
        ),
    ]
    _emit(
        {
            "kind": "candidate",
            "command": "python module probe",
            "argv": candidate,
        },
        sink=sink,
    )
    repo_result = _run_probe(
        label="python-module-repo-root-cwd",
        cwd=repo_root,
        candidate=candidate,
        timeout_sec=timeout_sec,
    )
    foundry_result = _run_probe(
        label="python-module-foundry-core-cwd",
        cwd=foundry_root,
        candidate=candidate,
        timeout_sec=timeout_sec,
    )
    _emit(repo_result, sink=sink)
    _emit(foundry_result, sink=sink)
    _emit(
        {
            "kind": "probe-summary",
            "label": "python-module-cwd-parity",
            "repo_root_category": repo_result.get("category"),
            "foundry_core_category": foundry_result.get("category"),
            "repo_root_probe": repo_result.get("stdout_excerpt", ""),
            "foundry_core_probe": foundry_result.get("stdout_excerpt", ""),
        },
        sink=sink,
    )


def _emit(row: dict[str, Any], *, sink: list[str]) -> None:
    line = json.dumps(row, ensure_ascii=False)
    sink.append(line)
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe the repo-root vs foundry-core cwd behavior of the `npx tsx` project CLI bridge path."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-command timeout in seconds. Defaults to 20.0 for a narrow slow-path probe.",
    )
    parser.add_argument(
        "--slug",
        default="authority-hardening",
        help="Slug to use for the temporary probe project. Defaults to authority-hardening.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write newline-delimited JSON results.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary probe project root instead of removing it after the run.",
    )
    args = parser.parse_args()

    output_lines: list[str] = []
    temp_dir = tempfile.mkdtemp(prefix="khub-ts-project-cli-probe-")
    temp_root = Path(temp_dir)
    repo_root = PROJECT_ROOT
    foundry_root = PROJECT_ROOT / "foundry-core"

    _emit(
        {
            "kind": "meta",
            "repo_root": str(repo_root),
            "foundry_root": str(foundry_root),
            "project_root": str(temp_root),
            "timeout_sec": args.timeout,
            "script": str(FOUNDRY_PROJECT_SCRIPT),
            "bridge_candidates": [
                {
                    "label": _candidate_label(candidate),
                    "argv": candidate,
                }
                for candidate in _bridge_candidates(
                    [
                        str(temp_root),
                        "python",
                        "project",
                        "create",
                        "--title",
                        "Authority Hardening",
                        "--slug",
                        args.slug,
                    ],
                    script_path=FOUNDRY_PROJECT_SCRIPT,
                    dist_path=FOUNDRY_PROJECT_DIST_SCRIPT,
                )
            ],
            "tooling": {
                "node": _tool_probe(["node", "--version"], cwd=repo_root),
                "npx": _tool_probe(["npx", "--version"], cwd=repo_root),
                "tsx": _tool_probe(["tsx", "--version"], cwd=repo_root),
                "python": _tool_probe(["python", "--version"], cwd=repo_root),
            },
        },
        sink=output_lines,
    )

    _run_tsx_eval_matrix(
        repo_root=repo_root,
        foundry_root=foundry_root,
        timeout_sec=args.timeout,
        sink=output_lines,
    )
    _run_python_module_matrix(
        repo_root=repo_root,
        foundry_root=foundry_root,
        timeout_sec=args.timeout,
        sink=output_lines,
    )

    create_script_args = [
        str(temp_root),
        "python",
        "project",
        "create",
        "--title",
        "Authority Hardening",
        "--slug",
        args.slug,
        "--summary",
        "authority contract probe",
        "--owner",
        "operator",
    ]
    create_candidate = _npx_tsx_candidate(create_script_args)
    _emit(
        {
            "kind": "candidate",
            "command": "project create",
            "argv": create_candidate,
        },
        sink=output_lines,
    )

    create_repo_root = _run_probe(
        label="create-repo-root-cwd",
        cwd=repo_root,
        candidate=create_candidate,
        timeout_sec=args.timeout,
    )
    _emit(create_repo_root, sink=output_lines)

    create_foundry_root = _run_probe(
        label="create-foundry-core-cwd",
        cwd=foundry_root,
        candidate=create_candidate,
        timeout_sec=args.timeout,
    )
    _emit(create_foundry_root, sink=output_lines)

    show_script_args = [
        str(temp_root),
        "python",
        "project",
        "show",
        "--slug",
        args.slug,
    ]
    show_candidate = _npx_tsx_candidate(show_script_args)
    _emit(
        {
            "kind": "candidate",
            "command": "project show",
            "argv": show_candidate,
        },
        sink=output_lines,
    )

    create_succeeded = any(
        row.get("returncode") == 0 for row in (create_repo_root, create_foundry_root)
    )
    if create_succeeded:
        _emit(
            _run_probe(
                label="show-repo-root-cwd",
                cwd=repo_root,
                candidate=show_candidate,
                timeout_sec=args.timeout,
            ),
            sink=output_lines,
        )
        _emit(
            _run_probe(
                label="show-foundry-core-cwd",
                cwd=foundry_root,
                candidate=show_candidate,
                timeout_sec=args.timeout,
            ),
            sink=output_lines,
        )
    else:
        _emit(
            {
                "kind": "probe-result",
                "label": "show-skipped",
                "cwd": "",
                "returncode": "skipped",
                "elapsed_sec": 0.0,
                "stdout_schema": None,
                "stdout_bytes": 0,
                "stderr_tail": "show probe skipped because project creation never succeeded",
            },
            sink=output_lines,
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    if not args.keep_temp:
        shutil.rmtree(temp_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
