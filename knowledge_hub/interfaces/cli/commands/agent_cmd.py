"""Agent Gateway CLI commands with compatibility aliases for legacy Foundry paths."""

from __future__ import annotations

from copy import copy
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import click
from rich.console import Console

from knowledge_hub.application.agent_gateway import (
    build_gateway_metadata,
)
from knowledge_hub.application.agent_writeback_preview import (
    DEFAULT_DOCS_ONLY_PATH_PREFIXES,
)
from knowledge_hub.application.agent.foundry_bridge import (
    FOUNDRY_DIST_SCRIPT as _FOUNDRY_DIST_SCRIPT,
    FOUNDRY_SCRIPT as _FOUNDRY_SCRIPT,
    PROJECT_ROOT as _PROJECT_ROOT,
    run_foundry_cli as _bridge_run_foundry_cli,
)
from knowledge_hub.application.mcp.responses import evaluate_policy_gate
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands.agent_writeback_runtime import (
    AgentWritebackRequestDeps,
    build_agent_writeback_request_payload as _build_agent_writeback_request_payload_direct,
    request_command_args as _request_command_args_direct,
    writeback_request_target_key as _writeback_request_target_key_direct,
)
from knowledge_hub.interfaces.cli.commands.foundry_cmd import (
    agent_discover,
    agent_discover_validate,
    agent_foundry_conflict_apply,
    agent_foundry_conflict_list,
    agent_foundry_conflict_reject,
    agent_sync,
    foundry_group,
)
from knowledge_hub.core.schema_validator import validate_payload

console = Console()
log = logging.getLogger("khub.cli.agent")

PROJECT_ROOT = _PROJECT_ROOT
FOUNDRY_SCRIPT = _FOUNDRY_SCRIPT
FOUNDRY_DIST_SCRIPT = _FOUNDRY_DIST_SCRIPT
AGENT_WRITEBACK_TARGET_POLICY = "docs_only"
POLICY_REDACTION_TEXT = "[REDACTED_BY_POLICY]"
AGENT_ROLES = ("planner", "researcher", "analyst", "summarizer", "auditor", "coach")
ORCHESTRATOR_MODES = ("single-pass", "adaptive", "strict")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    bool_text = str(value or "").strip().lower()
    if bool_text in {"1", "true", "yes", "on"}:
        return True
    if bool_text in {"0", "false", "no", "off"}:
        return False
    return False


def _coerce_int(value: Any, fallback: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = fallback
    return max(minimum, parsed)


def _normalize_status(value: Any, dry_run: bool) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {"running", "completed", "blocked", "failed"}:
        return lowered
    if lowered in {"ok", "done", "success", "verify_ok"}:
        return "completed"
    if lowered in {"denied", "deny", "blocked_by_policy"}:
        return "blocked"
    if "fail" in lowered or "error" in lowered:
        return "failed"
    return "blocked" if dry_run else "completed"


def _default_repo_path(repo_path: str | None) -> str | None:
    raw = str(repo_path or "").strip()
    if raw:
        return str(Path(raw).expanduser().resolve())
    try:
        return str(Path.cwd().resolve())
    except Exception:
        return None


def _classify_task_mode(goal: str) -> str:
    from knowledge_hub.application.task_context import classify_task_mode

    return classify_task_mode(goal)


def _build_task_context_payload(searcher, **kwargs):
    from knowledge_hub.application.task_context import build_task_context

    return build_task_context(searcher, **kwargs)


def _effective_include_workspace(goal: str, repo_path: str | None, include_workspace: bool | None) -> bool:
    if include_workspace is not None:
        return include_workspace
    mode = _classify_task_mode(goal)
    return bool(repo_path and mode in {"coding", "design", "debug"})


def _default_plan(goal: str, role: str, include_workspace: bool) -> list[str]:
    mode = _classify_task_mode(goal)
    if mode in {"coding", "design", "debug"}:
        return ["build_task_context", "ask_knowledge"]
    fallback_tokens = goal.lower()
    include_search = (
        role in {"planner", "researcher", "analyst", "coach"}
        or any(
            keyword in fallback_tokens
            for keyword in ["비교", "compare", "차이", "대조", "찾아", "검색", "search", "목록", "리스트", "추천"]
        )
    )
    return ["search_knowledge", "ask_knowledge"] if include_search else ["ask_knowledge"]


def _normalize_transition(item: dict[str, Any], now: str, default_stage: str = "PLAN") -> dict[str, Any]:
    stage = str(item.get("stage", default_stage) or default_stage).upper()
    status = str(item.get("status", item.get("action", item.get("step", "STEP"))) or "STEP").upper()
    message = str(item.get("message", item.get("action", item.get("step", ""))) or "")
    if not message:
        message = f"{stage}.{status}".lower()
    transition = {
        "stage": stage,
        "status": status,
        "message": message,
        "at": str(item.get("at", now) or now),
    }
    tool = item.get("tool")
    if tool:
        transition["tool"] = str(tool)
    if "code" in item:
        transition["code"] = str(item.get("code", ""))
    return transition


def _policy_gate_artifact(artifact: Any) -> tuple[dict[str, Any] | None, bool, list[str]]:
    if artifact is None:
        return None, True, []
    if not isinstance(artifact, dict):
        artifact = {"jsonContent": artifact, "classification": "P2", "generatedAt": _now_iso()}
    artifact = dict(artifact)

    policy_allowed, policy_errors, classification = evaluate_policy_gate(artifact)
    artifact["classification"] = classification

    if not policy_allowed or classification == "P0":
        blocked_artifact = {
            "jsonContent": POLICY_REDACTION_TEXT,
            "classification": "P0",
            "generatedAt": str(artifact.get("generatedAt") or _now_iso()),
            "metadata": {},
        }
        if artifact.get("id") is not None:
            blocked_artifact["id"] = str(artifact.get("id"))
        return (
            blocked_artifact,
            False,
            list(policy_errors or ["policy denied: P0 artifact blocked"]),
        )
    return artifact, True, []


def _validate_cli_payload(config: Any, payload: dict[str, Any], schema_id: str) -> None:
    strict = bool(config.get_nested("validation", "schema", "strict", default=False))
    result = validate_payload(payload, schema_id, strict=strict)
    if not result.ok:
        problems = ", ".join(result.errors[:5]) or "unknown schema validation error"
        raise click.ClickException(f"schema validation failed for {schema_id}: {problems}")


def _latest_ops_action_receipt(sqlite_db: SQLiteDatabase, action_id: str) -> dict[str, Any]:
    getter = getattr(sqlite_db, "get_latest_ops_action_receipt", None)
    if not callable(getter):
        return {}
    return getter(str(action_id)) or {}


def _run_foundry_cli(command: str, command_args: Sequence[str], timeout_sec: int = 120) -> tuple[dict[str, Any] | None, str | None]:
    return _bridge_run_foundry_cli(command, command_args, timeout_sec=timeout_sec)


def _extract_agent_plan(payload: dict[str, Any]) -> list[str]:
    raw_plan = payload.get("plan")
    if isinstance(raw_plan, list):
        values = [str(item).strip() for item in raw_plan if str(item).strip()]
        if values:
            return values
    raw_trace = payload.get("transitions", payload.get("trace"))
    if isinstance(raw_trace, list):
        values = []
        for item in raw_trace:
            if not isinstance(item, dict):
                continue
            step = str(item.get("step", item.get("action", item.get("tool", ""))) or "").strip()
            if step and step not in values:
                values.append(step)
        if values:
            return values
    return ["search_knowledge", "ask_knowledge"]


def _default_playbook(
    goal: str,
    role: str,
    orchestrator_mode: str,
    max_rounds: int,
    *,
    repo_path: str | None = None,
    include_workspace: bool | None = None,
    max_workspace_files: int = 8,
) -> dict[str, Any]:
    effective_repo_path = _default_repo_path(repo_path)
    workspace_enabled = _effective_include_workspace(goal, effective_repo_path, include_workspace)
    mode = _classify_task_mode(goal)
    plan = _default_plan(goal, role, workspace_enabled)
    return {
        "schema": "knowledge-hub.foundry.agent.run.playbook.v1",
        "source": "knowledge-hub/cli.agent.run",
        "goal": goal,
        "role": role,
        "orchestratorMode": orchestrator_mode,
        "maxRounds": max_rounds,
        "assumptions": [
            f"role={role}",
            f"orchestratorMode={orchestrator_mode}",
            "external calls use sanitized facts only",
            f"taskMode={mode}",
            f"workspaceIncluded={str(workspace_enabled).lower()}",
        ],
        "warnings": [] if effective_repo_path or not workspace_enabled else ["workspace context requested but repo_path unavailable"],
        "steps": [
            {
                "order": index + 1,
                "tool": tool,
                "objective": (
                    "assemble read-only task context"
                    if tool == "build_task_context"
                    else "collect evidence"
                    if tool == "search_knowledge"
                    else "synthesize answer"
                ),
                "rationale": (
                    "combine persistent knowledge with ephemeral workspace evidence before synthesis"
                    if tool == "build_task_context"
                    else "gather candidate sources before synthesis"
                    if tool == "search_knowledge"
                    else "build answer from collected evidence"
                ),
                **(
                    {
                        "inputs": {
                            "repoPath": effective_repo_path or "",
                            "includeWorkspace": workspace_enabled,
                            "maxWorkspaceFiles": max_workspace_files,
                        }
                    }
                    if tool == "build_task_context"
                    else {}
                ),
            }
            for index, tool in enumerate(plan)
        ],
        "generatedAt": _now_iso(),
    }


def _normalize_run_payload(
    payload: dict[str, Any],
    *,
    goal: str,
    max_rounds: int,
    dry_run: bool,
    role: str,
    orchestrator_mode: str,
) -> dict[str, Any]:
    now = _now_iso()
    run_id = str(payload.get("runId") or payload.get("run_id") or f"agent_run_{uuid4().hex[:12]}")
    status = _normalize_status(payload.get("status"), dry_run)
    stage = str(payload.get("stage", "DONE" if status in {"completed", "blocked", "failed"} else "PLAN")).upper()
    source = str(payload.get("source") or "knowledge-hub/cli.agent.run")
    plan = _extract_agent_plan(payload)
    transitions_raw = payload.get("transitions", payload.get("trace"))
    transitions: list[dict[str, Any]] = []
    if isinstance(transitions_raw, list):
        for item in transitions_raw:
            if isinstance(item, dict):
                transitions.append(_normalize_transition(item, now))
    if not transitions:
        transitions = [
            {
                "stage": "PLAN",
                "status": "PLAN",
                "message": "fallback plan generated",
                "at": now,
            }
        ]

    verify_raw = payload.get("verify")
    if isinstance(verify_raw, dict):
        verify = {
            "allowed": bool(verify_raw.get("allowed", status == "completed")),
            "schemaValid": bool(verify_raw.get("schemaValid", status == "completed")),
            "policyAllowed": bool(verify_raw.get("policyAllowed", status == "completed")),
            "schemaErrors": [str(item) for item in (verify_raw.get("schemaErrors") or [])],
        }
    else:
        verify = {
            "allowed": status == "completed",
            "schemaValid": status == "completed",
            "policyAllowed": True,
            "schemaErrors": [],
        }

    writeback_raw = payload.get("writeback")
    if isinstance(writeback_raw, dict):
        writeback = {
            "ok": bool(writeback_raw.get("ok", status == "completed")),
            "detail": str(writeback_raw.get("detail", "")),
        }
    else:
        writeback = {
            "ok": status == "completed",
            "detail": "dry-run: writeback skipped" if dry_run else "",
        }

    artifact, policy_allowed, policy_errors = _policy_gate_artifact(payload.get("artifact"))
    if not policy_allowed:
        status = "blocked"
        stage = "VERIFY"
        verify["allowed"] = False
        verify["schemaValid"] = False
        verify["policyAllowed"] = False
        verify["schemaErrors"] = list(verify["schemaErrors"]) + policy_errors
        writeback["ok"] = False
        writeback["detail"] = "policy gate blocked"
    else:
        verify["policyAllowed"] = bool(verify.get("policyAllowed", True))

    if status != "completed":
        verify["allowed"] = False

    playbook_raw = payload.get("playbook")
    if isinstance(playbook_raw, dict):
        playbook = dict(playbook_raw)
        playbook["schema"] = "knowledge-hub.foundry.agent.run.playbook.v1"
        playbook.setdefault("source", source)
        playbook.setdefault("goal", str(payload.get("goal", goal)))
        playbook.setdefault("role", role)
        playbook.setdefault("orchestratorMode", orchestrator_mode)
        playbook.setdefault("maxRounds", max_rounds)
        playbook.setdefault("assumptions", [])
        playbook.setdefault("warnings", [])
        playbook.setdefault("steps", [])
        playbook.setdefault("generatedAt", now)
    else:
        playbook = _default_playbook(str(payload.get("goal", goal)), role, orchestrator_mode, max_rounds)

    normalized = {
        "schema": "knowledge-hub.foundry.agent.run.result.v1",
        "source": source,
        "runId": run_id,
        "status": status,
        "goal": str(payload.get("goal", goal)),
        "role": str(payload.get("role", role)),
        "orchestratorMode": str(payload.get("orchestratorMode", payload.get("orchestrator_mode", orchestrator_mode))),
        "stage": stage,
        "plan": plan,
        "playbook": playbook,
        "transitions": transitions,
        "verify": verify,
        "writeback": writeback,
        "artifact": artifact,
        "maxRounds": _coerce_int(payload.get("maxRounds", payload.get("max_rounds", max_rounds)), max_rounds),
        "createdAt": str(payload.get("createdAt", now)),
        "updatedAt": str(payload.get("updatedAt", now)),
        "dryRun": _coerce_bool(payload.get("dryRun", payload.get("dry_run", dry_run))),
    }
    if bool(normalized["dryRun"]):
        normalized["gateway"] = build_gateway_metadata(surface="agent_run", mode="dry_run")
    tool_value = payload.get("tool")
    if tool_value is not None and str(tool_value).strip():
        normalized["tool"] = str(tool_value).strip()

    result = validate_payload(normalized, "knowledge-hub.foundry.agent.run.result.v1", strict=True)
    if not result.ok:
        normalized["status"] = "failed"
        normalized["stage"] = "VERIFY"
        verify["allowed"] = False
        verify["schemaValid"] = False
        for item in result.errors:
            if item not in verify["schemaErrors"]:
                verify["schemaErrors"].append(item)

    return normalized


def _build_fallback_run_payload(
    *,
    goal: str,
    max_rounds: int,
    dry_run: bool,
    role: str,
    orchestrator_mode: str,
    error: str | None,
    repo_path: str | None = None,
    include_workspace: bool | None = None,
    max_workspace_files: int = 8,
) -> dict[str, Any]:
    now = _now_iso()
    status = "blocked" if dry_run else "failed"
    schema_errors = [error] if error else []
    plan = _default_plan(goal, role, _effective_include_workspace(goal, _default_repo_path(repo_path), include_workspace))
    payload = {
        "schema": "knowledge-hub.foundry.agent.run.result.v1",
        "source": "knowledge-hub/cli.agent.run.fallback",
        "runId": f"agent_run_{uuid4().hex[:12]}",
        "status": status,
        "goal": goal,
        "role": role,
        "orchestratorMode": orchestrator_mode,
        "stage": "VERIFY" if dry_run else "FAILED",
        "tool": "knowledge-hub-fallback",
        "plan": plan,
        "playbook": _default_playbook(
            goal,
            role,
            orchestrator_mode,
            max_rounds,
            repo_path=repo_path,
            include_workspace=include_workspace,
            max_workspace_files=max_workspace_files,
        ),
        "transitions": [
            {
                "stage": "PLAN",
                "status": "PLAN",
                "message": "foundry bridge unavailable; fallback plan only",
                "tool": "knowledge-hub-fallback",
                "at": now,
            },
            {
                "stage": "VERIFY",
                "status": "BLOCK" if dry_run else "FAIL",
                "message": error or "fallback mode",
                "tool": "knowledge-hub-fallback",
                "at": now,
            },
        ],
        "verify": {
            "allowed": False,
            "schemaValid": False,
            "policyAllowed": True,
            "schemaErrors": schema_errors,
        },
        "writeback": {
            "ok": False,
            "detail": "dry-run: writeback skipped" if dry_run else "fallback run failed",
        },
        "artifact": None,
        "maxRounds": max_rounds,
        "createdAt": now,
        "updatedAt": now,
        "dryRun": dry_run,
    }
    return _normalize_run_payload(
        payload,
        goal=goal,
        max_rounds=max_rounds,
        dry_run=dry_run,
        role=role,
        orchestrator_mode=orchestrator_mode,
    )


def _build_agent_run_payload(
    *,
    goal: str,
    max_rounds: int,
    dry_run: bool,
    role: str,
    orchestrator_mode: str,
    repo_path: str | None = None,
    include_workspace: bool | None = None,
    max_workspace_files: int = 8,
    report_path: str = "",
) -> dict[str, Any]:
    max_rounds_safe = max(1, int(max_rounds))
    resolved_repo_path = _default_repo_path(repo_path or None)
    delegated, delegated_error = _run_foundry_cli(
        "run",
        [
            "--goal",
            goal,
            "--max-rounds",
            str(max_rounds_safe),
            "--role",
            role,
            "--orchestrator-mode",
            orchestrator_mode,
            "--dump-json",
            *(["--repo-path", resolved_repo_path] if resolved_repo_path else []),
            *(["--include-workspace"] if include_workspace is True else []),
            *(["--no-include-workspace"] if include_workspace is False else []),
            "--max-workspace-files",
            str(max(1, int(max_workspace_files))),
            *(["--dry-run"] if dry_run else []),
            *(["--report-path", report_path] if report_path else []),
        ],
        timeout_sec=180,
    )

    if delegated is None:
        return _build_fallback_run_payload(
            goal=goal,
            max_rounds=max_rounds_safe,
            dry_run=dry_run,
            role=role,
            orchestrator_mode=orchestrator_mode,
            error=delegated_error,
            repo_path=resolved_repo_path,
            include_workspace=include_workspace,
            max_workspace_files=max(1, int(max_workspace_files)),
        )

    return _normalize_run_payload(
        delegated,
        goal=goal,
        max_rounds=max_rounds_safe,
        dry_run=dry_run,
        role=role,
        orchestrator_mode=orchestrator_mode,
    )


def _writeback_request_target_key(
    *,
    goal: str,
    repo_path: str,
    role: str,
    orchestrator_mode: str,
) -> str:
    return _writeback_request_target_key_direct(
        goal=goal,
        repo_path=repo_path,
        role=role,
        orchestrator_mode=orchestrator_mode,
    )


def _request_command_args(
    *,
    goal: str,
    repo_path: str,
    role: str,
    orchestrator_mode: str,
    max_rounds: int,
    include_workspace: bool | None,
    max_workspace_files: int,
) -> list[str]:
    return _request_command_args_direct(
        goal=goal,
        repo_path=repo_path,
        role=role,
        orchestrator_mode=orchestrator_mode,
        max_rounds=max_rounds,
        include_workspace=include_workspace,
        max_workspace_files=max_workspace_files,
    )


def _build_agent_writeback_request_payload(
    *,
    khub: Any,
    goal: str,
    role: str,
    orchestrator_mode: str,
    max_rounds: int,
    repo_path: str | None,
    include_workspace: bool | None,
    max_workspace_files: int,
) -> dict[str, Any]:
    return _build_agent_writeback_request_payload_direct(
        deps=AgentWritebackRequestDeps(
            default_repo_path=_default_repo_path,
            build_agent_run_payload=_build_agent_run_payload,
            effective_include_workspace=_effective_include_workspace,
            build_task_context_payload=_build_task_context_payload,
            latest_ops_action_receipt=_latest_ops_action_receipt,
            validate_cli_payload=_validate_cli_payload,
        ),
        config=khub.config,
        sqlite_db=khub.sqlite_db(),
        goal=goal,
        role=role,
        orchestrator_mode=orchestrator_mode,
        max_rounds=max_rounds,
        repo_path=repo_path,
        include_workspace=include_workspace,
        max_workspace_files=max_workspace_files,
        target_policy=AGENT_WRITEBACK_TARGET_POLICY,
        allowed_path_prefixes=DEFAULT_DOCS_ONLY_PATH_PREFIXES,
    )


def _resolve_cli_searcher(khub: Any):
    if khub is not None:
        getter = getattr(khub, "searcher", None)
        if callable(getter):
            return getter()
        factory = getattr(khub, "factory", None)
        if factory is not None and hasattr(factory, "get_searcher"):
            return factory.get_searcher()
    raise click.ClickException("search runtime is unavailable for task context assembly")


@click.group("agent")
def agent_group():
    """Agent Gateway commands."""


@agent_group.command("context")
@click.argument("goal_arg", required=False)
@click.option("--goal", "goal_opt", default="", help="Task goal text")
@click.option("--repo-path", default="", help="Workspace repo path (defaults to current cwd)")
@click.option("--include-workspace/--no-include-workspace", default=True, show_default=True)
@click.option("--include-vault/--no-include-vault", default=True, show_default=True)
@click.option("--include-papers/--no-include-papers", default=True, show_default=True)
@click.option("--include-web/--no-include-web", default=True, show_default=True)
@click.option("--max-workspace-files", default=8, type=int, show_default=True)
@click.option("--max-knowledge-hits", default=5, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def agent_context(
    ctx,
    goal_arg,
    goal_opt,
    repo_path,
    include_workspace,
    include_vault,
    include_papers,
    include_web,
    max_workspace_files,
    max_knowledge_hits,
    as_json,
):
    """Assemble read-only task context from knowledge + workspace evidence."""
    goal = str(goal_opt or goal_arg or "").strip()
    if not goal:
        raise click.BadParameter("goal is required (argument or --goal)")

    payload = _build_task_context_payload(
        _resolve_cli_searcher(ctx.obj["khub"]),
        goal=goal,
        repo_path=repo_path or None,
        include_workspace=include_workspace,
        include_vault=include_vault,
        include_papers=include_papers,
        include_web=include_web,
        max_workspace_files=max(1, int(max_workspace_files)),
        max_knowledge_hits=max(1, int(max_knowledge_hits)),
    )

    if as_json:
        console.print_json(data=payload)
        return

    console.print(f"[bold]mode:[/bold] {payload.get('mode')}")
    console.print(f"[bold]goal:[/bold] {payload.get('goal')}")
    console.print(
        f"[bold]evidence:[/bold] knowledge={len(payload.get('knowledge_hits', []))}, "
        f"workspace={len(payload.get('workspace_files', []))}"
    )
    for item in payload.get("workspace_files", [])[:6]:
        console.print(f"- {item.get('relative_path')} [{item.get('reason')}]")
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    if warnings:
        console.print("[yellow]warnings:[/yellow]")
        for warning in warnings[:5]:
            console.print(f"  - {warning}")


context_cmd = copy(agent_context)
context_cmd.help = "Assemble read-only task context from knowledge + workspace evidence."
context_cmd.short_help = "Assemble read-only task context."


@agent_group.command("run")
@click.argument("goal_arg", required=False)
@click.option("--goal", "goal_opt", default="", help="Agent goal text")
@click.option("--max-rounds", default=3, type=int, show_default=True)
@click.option("--role", default="planner", type=click.Choice(AGENT_ROLES), show_default=True)
@click.option("--orchestrator-mode", default="adaptive", type=click.Choice(ORCHESTRATOR_MODES), show_default=True)
@click.option("--repo-path", default="", help="Workspace repo path passed to task-context step")
@click.option("--include-workspace/--no-include-workspace", default=None, help="Override workspace inclusion for coding goals")
@click.option("--max-workspace-files", default=8, type=int, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--dump-json", is_flag=True, default=False, help="Alias for --json")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.option("--compact", is_flag=True, default=False, help="Compact human-readable output")
@click.option("--report-path", default="", help="Write run report file path")
def agent_run(
    goal_arg,
    goal_opt,
    max_rounds,
    role,
    orchestrator_mode,
    repo_path,
    include_workspace,
    max_workspace_files,
    dry_run,
    dump_json,
    as_json,
    compact,
    report_path,
):
    """Run Plan -> Act -> Verify loop via Foundry bridge."""
    goal = str(goal_opt or goal_arg or "").strip()
    if not goal:
        raise click.BadParameter("goal is required (argument or --goal)")

    payload = _build_agent_run_payload(
        goal=goal,
        max_rounds=max_rounds,
        dry_run=dry_run,
        role=role,
        orchestrator_mode=orchestrator_mode,
        repo_path=repo_path,
        include_workspace=include_workspace,
        max_workspace_files=max_workspace_files,
        report_path=report_path,
    )

    if as_json or dump_json:
        console.print_json(data=payload)
        return

    console.print(f"[bold]runId:[/bold] {payload.get('runId')}")
    console.print(f"[bold]status:[/bold] {payload.get('status')}")
    console.print(f"[bold]stage:[/bold] {payload.get('stage')}")
    console.print(f"[bold]goal:[/bold] {payload.get('goal')}")
    plan = payload.get("plan") if isinstance(payload.get("plan"), list) else []
    if plan:
        console.print(f"[bold]plan:[/bold] {' -> '.join([str(item) for item in plan])}")
    verify = payload.get("verify") if isinstance(payload.get("verify"), dict) else {}
    if verify:
        console.print(f"[bold]verify.allowed:[/bold] {verify.get('allowed')}")
        errors = verify.get("schemaErrors")
        if errors:
            if compact:
                console.print(f"[red]{'; '.join([str(item) for item in errors[:3]])}[/red]")
            else:
                console.print("[red]verify errors:[/red]")
                for item in errors:
                    console.print(f"  - {item}")


@agent_group.command("writeback-request")
@click.argument("goal_arg", required=False)
@click.option("--goal", "goal_opt", default="", help="Goal text for the requested repo-local writeback")
@click.option("--max-rounds", default=3, type=int, show_default=True)
@click.option("--role", default="planner", type=click.Choice(AGENT_ROLES), show_default=True)
@click.option("--orchestrator-mode", default="adaptive", type=click.Choice(ORCHESTRATOR_MODES), show_default=True)
@click.option("--repo-path", default="", help="Repository path for the requested writeback scope")
@click.option("--include-workspace/--no-include-workspace", default=None, help="Override workspace inclusion for the embedded dry-run")
@click.option("--max-workspace-files", default=8, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def agent_writeback_request(
    ctx,
    goal_arg,
    goal_opt,
    max_rounds,
    role,
    orchestrator_mode,
    repo_path,
    include_workspace,
    max_workspace_files,
    as_json,
):
    """Create an approval-gated repo-local writeback request from the current agent dry-run surface."""
    goal = str(goal_opt or goal_arg or "").strip()
    if not goal:
        raise click.BadParameter("goal is required (argument or --goal)")

    khub = ctx.obj["khub"]
    payload = _build_agent_writeback_request_payload(
        khub=khub,
        goal=goal,
        role=role,
        orchestrator_mode=orchestrator_mode,
        max_rounds=max_rounds,
        repo_path=repo_path,
        include_workspace=include_workspace,
        max_workspace_files=max_workspace_files,
    )

    if as_json:
        console.print_json(data=payload)
        return

    request = dict(payload.get("request") or {})
    approval = dict(payload.get("approval") or {})
    console.print(f"[bold]requestId:[/bold] {request.get('actionId')}")
    console.print(f"[bold]status:[/bold] {payload.get('status')}")
    console.print(f"[bold]repo:[/bold] {payload.get('repoPath')}")
    console.print(f"[bold]goal:[/bold] {payload.get('goal')}")
    console.print(f"[bold]requestOperation:[/bold] {payload.get('requestOperation')}")
    console.print(f"[bold]approval:[/bold] {approval.get('status')}")
    console.print(f"[bold]dry-run stage:[/bold] {((payload.get('dryRun') or {}).get('stage'))}")
    console.print(f"- ack: {((approval.get('commands') or {}).get('ack'))}")
    console.print(f"- execute: {((approval.get('commands') or {}).get('execute'))}")
    console.print(f"- resolve: {((approval.get('commands') or {}).get('resolve'))}")


# Keep the default `khub agent --help` surface gateway-focused while preserving
# direct invocation for legacy Foundry/operator command paths. The canonical
# operator surface now lives under `khub labs foundry ...`.
for _alias_name, _canonical_command in (
    ("sync", agent_sync),
    ("foundry-conflict-list", agent_foundry_conflict_list),
    ("foundry-conflict-apply", agent_foundry_conflict_apply),
    ("foundry-conflict-reject", agent_foundry_conflict_reject),
    ("discover", agent_discover),
    ("discover-validate", agent_discover_validate),
):
    _alias = copy(_canonical_command)
    _alias.hidden = True
    agent_group.add_command(_alias, _alias_name)
