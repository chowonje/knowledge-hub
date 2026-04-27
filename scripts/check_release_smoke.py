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
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import yaml

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.config import PUBLIC_PARSER_DEFAULT
from knowledge_hub.infrastructure.config import apply_public_setup_profile


DOCTOR_ALLOWED_STATUSES = {"ok", "blocked", "degraded", "needs_setup"}
STATUS_REQUIRED_MARKERS = ("Knowledge Hub v", "Retrieval Runtime", "vector corpus")
DOCTOR_REQUIRED_AREAS = {"settings", "Ollama", "vector corpus"}
TOP_HELP_REQUIRED_MARKERS = ("Commands:", "add", "provider", "doctor", "status", "init")
WEEKLY_TOP_HELP_REQUIRED_MARKERS = ("Commands:", "add", "provider", "index", "search", "ask", "doctor", "status")
ADD_HELP_REQUIRED_MARKERS = ("Add a source with one command", "--type", "--index", "--allow-external")
CAPTURE_HELP_REQUIRED_MARKERS = ("Commands:", "cleanup", "requeue", "status")
INVALID_COMMAND_REQUIRED_MARKER = "No such command"
INVALID_COMMAND_FORBIDDEN_MARKERS = ("Traceback (most recent call last)", "예상치 못한 오류")
SMOKE_COMMAND_TIMEOUT_SEC = 20.0
CORE_LOOP_COMMAND_TIMEOUT_SEC = 60.0
SMOKE_TIMEOUT_RETURN_CODE = 124
INDEX_REQUIRED_SCHEMA = "knowledge-hub.index.result.v1"
SEARCH_REQUIRED_SCHEMA = "knowledge-hub.search.result.v1"
ASK_REQUIRED_SCHEMA = "knowledge-hub.ask.result.v1"
WEEKLY_SEARCH_QUERY = "alpha retrieval"
PROVIDER_REQUIRED_SCHEMA = "knowledge-hub.provider.result.v1"


@dataclass
class CommandResult:
    name: str
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False
    timeout_sec: float = SMOKE_COMMAND_TIMEOUT_SEC


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


def _ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run_command(
    argv: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    name: str,
    timeout_sec: float = SMOKE_COMMAND_TIMEOUT_SEC,
) -> CommandResult:
    started = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_sec,
        )
        returncode = int(proc.returncode)
        stdout = _ensure_text(proc.stdout)
        stderr = _ensure_text(proc.stderr)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = SMOKE_TIMEOUT_RETURN_CODE
        stdout = _ensure_text(getattr(exc, "output", None))
        stderr = _ensure_text(getattr(exc, "stderr", None))
    return CommandResult(
        name=name,
        argv=list(argv),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_sec=round(time.monotonic() - started, 3),
        timed_out=timed_out,
        timeout_sec=timeout_sec,
    )


def validate_setup_result(result: CommandResult, *, config_path: Path) -> ValidationResult:
    errors: list[str] = []
    config_data: dict[str, Any] = {}
    if result.timed_out:
        errors.append(f"setup timed out after {result.timeout_sec:.0f}s")
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
    if result.timed_out:
        errors.append(f"status timed out after {result.timeout_sec:.0f}s")
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
    if result.timed_out:
        errors.append(f"doctor timed out after {result.timeout_sec:.0f}s")
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
    if result.timed_out:
        errors.append(f"{result.name} timed out after {result.timeout_sec:.0f}s")
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
    if result.timed_out:
        errors.append(f"invalid command timed out after {result.timeout_sec:.0f}s")
    if result.returncode == 0:
        errors.append("invalid command exited with 0")
    combined = f"{result.stdout}\n{result.stderr}"
    if INVALID_COMMAND_REQUIRED_MARKER not in combined:
        errors.append(f"invalid command output missing marker: {INVALID_COMMAND_REQUIRED_MARKER}")
    for marker in INVALID_COMMAND_FORBIDDEN_MARKERS:
        if marker in combined:
            errors.append(f"invalid command output included internal error marker: {marker}")
    return ValidationResult(
        ok=not errors,
        summary="invalid command exits non-zero with a user-facing error" if not errors else "invalid command contract failed",
        details={"returncode": result.returncode},
        errors=errors,
    )


def validate_provider_recommend_result(result: CommandResult) -> ValidationResult:
    errors: list[str] = []
    payload: dict[str, Any] = {}
    if result.timed_out:
        errors.append(f"provider recommend timed out after {result.timeout_sec:.0f}s")
    if result.returncode != 0:
        errors.append(f"provider recommend exited with {result.returncode}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        errors.append(f"provider recommend did not emit valid json: {exc}")
    if errors:
        return ValidationResult(ok=False, summary="provider recommend contract failed", details={}, errors=errors)
    if str(payload.get("schema") or "") != PROVIDER_REQUIRED_SCHEMA:
        errors.append("provider recommend schema mismatch")
    if str(payload.get("status") or "") != "ok":
        errors.append(f"provider recommend status is not ok: {payload.get('status')}")
    recommendations = payload.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        errors.append("provider recommend returned no recommendations")
        recommendations = []
    profiles = {str(dict(item or {}).get("profile") or "") for item in recommendations if isinstance(item, dict)}
    for required in ("local", "balanced", "quality", "codex-mcp"):
        if required not in profiles:
            errors.append(f"provider recommend missing profile: {required}")
    return ValidationResult(
        ok=not errors,
        summary="provider recommendations expose the public setup profiles" if not errors else "provider recommend contract failed",
        details={"profiles": sorted(profiles)},
        errors=errors,
    )


def extract_trailing_json_object(text: str) -> dict[str, Any]:
    content = str(text or "").strip()
    errors: list[str] = []
    for match in re.finditer(r"(?m)^\{", content):
        try:
            return dict(json.loads(content[match.start() :]))
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
    if errors:
        raise json.JSONDecodeError(errors[-1], content, 0)
    raise json.JSONDecodeError("no JSON object found in command output", content, 0)


def prepare_weekly_core_loop_fixture(home_dir: Path) -> tuple[Path, Path]:
    config_path = temp_config_path(home_dir)
    vault_dir = home_dir / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)
    note_path = vault_dir / "alpha.md"
    note_path.write_text(
        "# Alpha retrieval note\n\n"
        "alpha retrieval memory keeps a grounded source trace for the weekly smoke gate.\n",
        encoding="utf-8",
    )

    config = Config()
    apply_public_setup_profile(config, "local")
    config.set_nested("paper", "summary", "parser", PUBLIC_PARSER_DEFAULT)
    config.set_nested("obsidian", "enabled", True)
    config.set_nested("obsidian", "vault_path", str(vault_dir))
    config.set_nested("storage", "sqlite", str(home_dir / ".khub" / "knowledge.db"))
    config.set_nested("storage", "vector_db", str(home_dir / ".khub" / "chroma_db"))
    config.set_nested("storage", "collection_name", "knowledge_hub_weekly_core_loop_smoke")
    config.set_nested("indexing", "failure_report_dir", str(home_dir / ".khub" / "runs"))
    config.save(str(config_path))
    return config_path, note_path


def validate_index_result(result: CommandResult) -> ValidationResult:
    errors: list[str] = []
    payload: dict[str, Any] = {}
    if result.timed_out:
        errors.append(f"index timed out after {result.timeout_sec:.0f}s")
    if result.returncode != 0:
        errors.append(f"index exited with {result.returncode}")
    try:
        payload = extract_trailing_json_object(result.stdout)
    except json.JSONDecodeError as exc:
        errors.append(f"index did not emit trailing json payload: {exc}")
    if errors:
        return ValidationResult(ok=False, summary="index contract failed", details={}, errors=errors)
    if str(payload.get("schema") or "") != INDEX_REQUIRED_SCHEMA:
        errors.append("index schema mismatch")
    if str(payload.get("status") or "") != "ok":
        errors.append(f"index status is not ok: {payload.get('status')}")
    vector_count = int(payload.get("vectorDbCount", 0) or 0)
    vault_breakdown = dict(dict(payload.get("processedBreakdown") or {}).get("vault") or {})
    if vector_count <= 0:
        errors.append("index did not produce vector documents")
    if int(vault_breakdown.get("succeeded", 0) or 0) <= 0:
        errors.append("index did not succeed on any vault chunk")
    return ValidationResult(
        ok=not errors,
        summary="small vault index produced a non-empty vector corpus" if not errors else "index contract failed",
        details={
            "status": payload.get("status"),
            "vectorDbCount": vector_count,
            "vaultProcessed": int(vault_breakdown.get("processed", 0) or 0),
            "vaultSucceeded": int(vault_breakdown.get("succeeded", 0) or 0),
            "reportPath": str(payload.get("reportPath") or ""),
        },
        errors=errors,
    )


def validate_search_result(result: CommandResult) -> ValidationResult:
    errors: list[str] = []
    payload: dict[str, Any] = {}
    if result.timed_out:
        errors.append(f"search timed out after {result.timeout_sec:.0f}s")
    if result.returncode != 0:
        errors.append(f"search exited with {result.returncode}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        errors.append(f"search did not emit valid json: {exc}")
    if errors:
        return ValidationResult(ok=False, summary="search contract failed", details={}, errors=errors)
    if str(payload.get("schema") or "") != SEARCH_REQUIRED_SCHEMA:
        errors.append("search schema mismatch")
    if str(payload.get("query") or "") != WEEKLY_SEARCH_QUERY:
        errors.append("search query echo mismatch")
    if str(payload.get("status") or "") == "init_error":
        errors.append(f"search init_error: {payload.get('initError')}")
    results = list(payload.get("results") or [])
    if not results:
        errors.append("search returned no results")
    if not isinstance(payload.get("runtimeDiagnostics"), dict):
        errors.append("search missing runtimeDiagnostics object")
    if not isinstance(payload.get("graphQuerySignal"), dict):
        errors.append("search missing graphQuerySignal object")
    if results:
        first = dict(results[0] or {})
        if str(first.get("sourceType") or first.get("source_type") or "") != "vault":
            errors.append("top search result is not a vault document")
    return ValidationResult(
        ok=not errors,
        summary="keyword search returned a grounded vault result" if not errors else "search contract failed",
        details={
            "resultCount": len(results),
            "topTitle": str(dict(results[0] or {}).get("title") or "") if results else "",
            "topSourceType": str(dict(results[0] or {}).get("sourceType") or dict(results[0] or {}).get("source_type") or "") if results else "",
            "hasRuntimeDiagnostics": isinstance(payload.get("runtimeDiagnostics"), dict),
            "hasGraphQuerySignal": isinstance(payload.get("graphQuerySignal"), dict),
        },
        errors=errors,
    )


def validate_ask_result(result: CommandResult) -> ValidationResult:
    errors: list[str] = []
    payload: dict[str, Any] = {}
    if result.timed_out:
        errors.append(f"ask timed out after {result.timeout_sec:.0f}s")
    if result.returncode != 0:
        errors.append(f"ask exited with {result.returncode}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        errors.append(f"ask did not emit valid json: {exc}")
    if errors:
        return ValidationResult(ok=False, summary="ask contract failed", details={}, errors=errors)
    if str(payload.get("schema") or "") != ASK_REQUIRED_SCHEMA:
        errors.append("ask schema mismatch")
    if str(payload.get("status") or "") == "init_error":
        errors.append(f"ask init_error: {payload.get('initError')}")
    if str(payload.get("question") or "") != WEEKLY_SEARCH_QUERY:
        errors.append("ask question echo mismatch")
    if bool(payload.get("allowExternal", True)):
        errors.append("ask allowExternal is not false")
    external_policy = dict(payload.get("externalPolicy") or {})
    if not external_policy:
        errors.append("ask missing externalPolicy diagnostics")
    elif str(external_policy.get("policyMode") or "") != "local-only":
        errors.append("ask externalPolicy is not local-only")
    answer = str(payload.get("answer") or "").strip()
    if not answer:
        errors.append("ask returned an empty answer")
    citations = list(payload.get("citations") or [])
    sources = list(payload.get("sources") or [])
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        errors.append("ask evidence field is not a list")
    memory_route = dict(payload.get("memoryRoute") or {})
    memory_prefilter = dict(payload.get("memoryPrefilter") or {})
    paper_memory_prefilter = dict(payload.get("paperMemoryPrefilter") or {})
    if str(memory_route.get("contractRole") or "") != "ask_retrieval_memory_prefilter":
        errors.append("ask missing memoryRoute contract role")
    if str(memory_prefilter.get("contractRole") or "") != "retrieval_memory_prefilter":
        errors.append("ask missing memoryPrefilter contract role")
    if str(paper_memory_prefilter.get("contractRole") or "") != "paper_source_memory_prefilter":
        errors.append("ask missing paperMemoryPrefilter contract role")
    if not isinstance(payload.get("runtimeDiagnostics"), dict):
        errors.append("ask missing runtimeDiagnostics object")
    has_vault_citation = any(str(dict(item or {}).get("source_type") or "") == "vault" for item in citations)
    has_vault_source = any(
        str(dict(item or {}).get("source_type") or dict(item or {}).get("sourceType") or "") == "vault"
        for item in sources
    )
    if not has_vault_citation and not has_vault_source:
        errors.append("ask returned no vault citation/source")
    return ValidationResult(
        ok=not errors,
        summary="keyword ask returned a grounded vault answer" if not errors else "ask contract failed",
        details={
            "answerLength": len(answer),
            "citationCount": len(citations),
            "sourceCount": len(sources),
            "evidenceCount": len(evidence) if isinstance(evidence, list) else 0,
            "hasVaultCitation": has_vault_citation,
            "hasVaultSource": has_vault_source,
            "hasExternalPolicy": bool(external_policy),
            "hasMemoryRoute": bool(memory_route),
            "hasMemoryPrefilter": bool(memory_prefilter),
            "hasPaperMemoryPrefilter": bool(paper_memory_prefilter),
            "hasRuntimeDiagnostics": isinstance(payload.get("runtimeDiagnostics"), dict),
            "warningCount": len(payload.get("warnings") or []),
        },
        errors=errors,
    )


def run_release_smoke(*, keep_temp_dir: bool = False) -> dict[str, Any]:
    root = repo_root()
    cli_argv = [sys.executable, "-m", "knowledge_hub.interfaces.cli.main"]
    home_dir = Path(tempfile.mkdtemp(prefix="khub-release-smoke-"))
    env = build_env(home_dir)
    config_path = temp_config_path(home_dir)
    plan = [
        ("top_help", cli_argv + ["--help"]),
        ("setup", cli_argv + ["setup", "--quick", "--non-interactive"]),
        ("add_help", cli_argv + ["add", "--help"]),
        ("provider_recommend", cli_argv + ["provider", "recommend", "--json"]),
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
        elif name == "add_help":
            validation = validate_help_result(
                result,
                required_markers=ADD_HELP_REQUIRED_MARKERS,
                summary="add help exposes the public intake contract",
            )
        elif name == "provider_recommend":
            validation = validate_provider_recommend_result(result)
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
                "timedOut": result.timed_out,
                "timeoutSec": result.timeout_sec,
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

    checked_count = len(validations)
    passed_count = sum(1 for item in validations if item.get("status") == "ok")
    payload = {
        "status": "ok" if not failed else "failed",
        "repoRoot": str(root),
        "tempHome": str(home_dir) if (keep_temp_dir or failed) else "",
        "checkedCount": checked_count,
        "passedCount": passed_count,
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
        shutil.rmtree(home_dir, ignore_errors=True)
    elif not keep_temp_dir and not failed:
        shutil.rmtree(home_dir, ignore_errors=True)
    return payload


def run_weekly_core_loop_smoke(*, keep_temp_dir: bool = False) -> dict[str, Any]:
    root = repo_root()
    cli_argv = [sys.executable, "-m", "knowledge_hub.interfaces.cli.main"]
    home_dir = Path(tempfile.mkdtemp(prefix="khub-weekly-core-loop-smoke-"))
    env = build_env(home_dir)
    config_path, note_path = prepare_weekly_core_loop_fixture(home_dir)
    plan = [
        ("top_help", cli_argv + ["--help"], SMOKE_COMMAND_TIMEOUT_SEC),
        ("doctor", cli_argv + ["--config", str(config_path), "doctor", "--json"], SMOKE_COMMAND_TIMEOUT_SEC),
        ("status", cli_argv + ["--config", str(config_path), "status"], SMOKE_COMMAND_TIMEOUT_SEC),
        ("index", cli_argv + ["--config", str(config_path), "index", "--vault-all", "--json"], CORE_LOOP_COMMAND_TIMEOUT_SEC),
        (
            "ask",
            cli_argv
            + [
                "--config",
                str(config_path),
                "ask",
                WEEKLY_SEARCH_QUERY,
                "--source",
                "vault",
                "--mode",
                "keyword",
                "--no-allow-external",
                "--json",
            ],
            CORE_LOOP_COMMAND_TIMEOUT_SEC,
        ),
        (
            "search",
            cli_argv
            + [
                "--config",
                str(config_path),
                "search",
                WEEKLY_SEARCH_QUERY,
                "--source",
                "vault",
                "--mode",
                "keyword",
                "--json",
            ],
            CORE_LOOP_COMMAND_TIMEOUT_SEC,
        ),
    ]
    validations: list[dict[str, Any]] = []
    failed = False

    for name, argv, timeout_sec in plan:
        result = run_command(argv, env=env, cwd=root, name=name, timeout_sec=timeout_sec)
        if name == "top_help":
            validation = validate_help_result(
                result,
                required_markers=WEEKLY_TOP_HELP_REQUIRED_MARKERS,
                summary="top-level product help surface is present",
            )
        elif name == "doctor":
            validation = validate_doctor_result(result)
        elif name == "status":
            validation = validate_status_result(result, config_path=config_path)
        elif name == "index":
            validation = validate_index_result(result)
        elif name == "ask":
            validation = validate_ask_result(result)
        else:
            validation = validate_search_result(result)
        validations.append(
            {
                "name": name,
                "argv": result.argv,
                "returncode": result.returncode,
                "durationSec": result.duration_sec,
                "timedOut": result.timed_out,
                "timeoutSec": result.timeout_sec,
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

    checked_count = len(validations)
    passed_count = sum(1 for item in validations if item.get("status") == "ok")
    payload = {
        "mode": "weekly_core_loop",
        "status": "ok" if not failed else "failed",
        "repoRoot": str(root),
        "tempHome": str(home_dir) if (keep_temp_dir or failed) else "",
        "configPath": str(config_path),
        "fixtureNote": str(note_path),
        "checkedCount": checked_count,
        "passedCount": passed_count,
        "commands": validations,
    }
    if not keep_temp_dir and not failed:
        shutil.rmtree(home_dir, ignore_errors=True)
    return payload


def run_smoke(*, mode: str, keep_temp_dir: bool = False) -> dict[str, Any]:
    if mode == "release":
        payload = run_release_smoke(keep_temp_dir=keep_temp_dir)
        payload.setdefault("mode", "release")
        return payload
    if mode == "weekly_core_loop":
        return run_weekly_core_loop_smoke(keep_temp_dir=keep_temp_dir)
    raise ValueError(f"unknown smoke mode: {mode}")


def payload_exit_code(payload: dict[str, Any]) -> int:
    return 0 if payload.get("status") == "ok" else 1


def print_human(payload: dict[str, Any]) -> None:
    print(f"[{payload.get('mode') or 'release'} smoke]")
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
    parser.add_argument(
        "--mode",
        choices=["release", "weekly_core_loop"],
        default="release",
        help="Smoke mode: release keeps the narrow local release-trust gate, weekly_core_loop exercises the current product loop.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--keep-temp-dir",
        action="store_true",
        help="Keep the generated temp HOME under .tmp_release_smoke_home for inspection.",
    )
    args = parser.parse_args()

    payload = run_smoke(mode=str(args.mode), keep_temp_dir=bool(args.keep_temp_dir))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_human(payload)
    return payload_exit_code(payload)


if __name__ == "__main__":
    raise SystemExit(main())
