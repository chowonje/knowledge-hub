"""Private runtime helpers for the agent writeback-request CLI lane."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Sequence

import click

from knowledge_hub.application.agent_gateway import build_writeback_request_gateway_metadata
from knowledge_hub.application.agent_writeback_preview import build_writeback_preview
from knowledge_hub.application.ops_actions import queue_item_view


@dataclass(frozen=True)
class AgentWritebackRequestDeps:
    default_repo_path: Callable[[str | None], str | None]
    build_agent_run_payload: Callable[..., dict[str, Any]]
    effective_include_workspace: Callable[[str, str | None, bool | None], bool]
    build_task_context_payload: Callable[..., dict[str, Any]]
    latest_ops_action_receipt: Callable[[Any, str], dict[str, Any]]
    validate_cli_payload: Callable[[Any, dict[str, Any], str], None]


def writeback_request_target_key(
    *,
    goal: str,
    repo_path: str,
    role: str,
    orchestrator_mode: str,
) -> str:
    goal_hash = hashlib.sha256(
        "::".join([str(repo_path).strip(), str(role).strip(), str(orchestrator_mode).strip(), str(goal).strip()]).encode("utf-8")
    ).hexdigest()[:16]
    return f"repo_goal:{repo_path}:{goal_hash}"


def request_command_args(
    *,
    goal: str,
    repo_path: str,
    role: str,
    orchestrator_mode: str,
    max_rounds: int,
    include_workspace: bool | None,
    max_workspace_files: int,
) -> list[str]:
    args = [
        "agent",
        "run",
        "--goal",
        goal,
        "--repo-path",
        repo_path,
        "--role",
        role,
        "--orchestrator-mode",
        orchestrator_mode,
        "--max-rounds",
        str(max(1, int(max_rounds))),
        "--max-workspace-files",
        str(max(1, int(max_workspace_files))),
    ]
    if include_workspace is True:
        args.append("--include-workspace")
    elif include_workspace is False:
        args.append("--no-include-workspace")
    return args


def build_agent_writeback_request_payload(
    *,
    deps: AgentWritebackRequestDeps,
    config: Any,
    sqlite_db: Any,
    goal: str,
    role: str,
    orchestrator_mode: str,
    max_rounds: int,
    repo_path: str | None,
    include_workspace: bool | None,
    max_workspace_files: int,
    target_policy: str,
    allowed_path_prefixes: Sequence[str],
) -> dict[str, Any]:
    resolved_repo_path = deps.default_repo_path(repo_path or None)
    if not resolved_repo_path:
        raise click.ClickException("repo-local writeback request requires a repo path or resolvable current working directory")

    allowed_prefixes = [str(item) for item in allowed_path_prefixes]
    dry_run_payload = deps.build_agent_run_payload(
        goal=goal,
        max_rounds=max_rounds,
        dry_run=True,
        role=role,
        orchestrator_mode=orchestrator_mode,
        repo_path=resolved_repo_path,
        include_workspace=include_workspace,
        max_workspace_files=max_workspace_files,
        report_path="",
    )
    if dry_run_payload.get("dryRun") is not True:
        raise click.ClickException("writeback request requires a dry-run agent payload")

    effective_include_workspace = deps.effective_include_workspace(goal, resolved_repo_path, include_workspace)
    preview_context_error = ""
    task_context_payload: dict[str, Any] = {}
    try:
        task_context_payload = deps.build_task_context_payload(
            None,
            goal=goal,
            repo_path=resolved_repo_path,
            include_workspace=effective_include_workspace,
            include_vault=False,
            include_papers=False,
            include_web=False,
            max_workspace_files=max(1, int(max_workspace_files)),
            max_knowledge_hits=0,
        )
    except Exception as error:
        preview_context_error = f"writeback preview context unavailable: {error}"

    writeback_preview = build_writeback_preview(
        goal=goal,
        repo_path=resolved_repo_path,
        dry_run_payload=dry_run_payload,
        task_context_payload=task_context_payload,
        include_workspace=include_workspace,
        effective_include_workspace=effective_include_workspace,
        context_error=preview_context_error,
        max_targets=max(1, int(max_workspace_files)),
        allowed_path_prefixes=allowed_prefixes,
    )

    command_args = request_command_args(
        goal=goal,
        repo_path=resolved_repo_path,
        role=role,
        orchestrator_mode=orchestrator_mode,
        max_rounds=max_rounds,
        include_workspace=include_workspace,
        max_workspace_files=max_workspace_files,
    )
    request_summary = f"Agent repo-local writeback request: {goal.strip()}"
    upserted = sqlite_db.upsert_ops_action(
        scope="agent",
        action_type="agent_repo_writeback_request",
        target_kind="repo_goal",
        target_key=writeback_request_target_key(
            goal=goal,
            repo_path=resolved_repo_path,
            role=role,
            orchestrator_mode=orchestrator_mode,
        ),
        summary=request_summary,
        reason_codes=["agent_gateway_repo_local_writeback_request"],
        command="khub",
        args=command_args,
        alerts=[
            {
                "severity": "info",
                "code": "approval_required",
                "summary": "writeback request created; explicit approval is required before any execution lane is enabled",
            }
        ],
        action={
            "goal": goal,
            "repoPath": resolved_repo_path,
            "role": role,
            "orchestratorMode": orchestrator_mode,
            "maxRounds": max(1, int(max_rounds)),
            "includeWorkspace": include_workspace,
            "maxWorkspaceFiles": max(1, int(max_workspace_files)),
            "requestKind": "repo_local_writeback",
            "gatewayVersion": "v2",
            "targetPolicy": target_policy,
            "allowedPathPrefixes": allowed_prefixes,
            "dryRunStatus": str(dry_run_payload.get("status") or ""),
            "dryRunStage": str(dry_run_payload.get("stage") or ""),
            "dryRunPlan": [str(item) for item in (dry_run_payload.get("plan") or []) if str(item).strip()],
            "writebackPreview": writeback_preview,
            "writebackPreviewFingerprint": str(writeback_preview.get("previewFingerprint") or ""),
        },
    )
    queue_item = dict(upserted.get("item") or {})
    request_view = queue_item_view(
        queue_item,
        latest_receipt=deps.latest_ops_action_receipt(sqlite_db, str(queue_item.get("action_id") or "")),
    )
    action_id = str(request_view.get("actionId") or "")
    payload = {
        "schema": "knowledge-hub.agent.writeback.request.result.v1",
        "status": "ok",
        "goal": goal,
        "repoPath": resolved_repo_path,
        "requestOperation": str(upserted.get("operation") or "created"),
        "request": request_view,
        "approval": {
            "required": True,
            "status": "pending",
            "note": "repo-local writeback stays blocked until an explicit approval decision exists",
            "commands": {
                "ack": f"khub labs ops action-ack --action-id {action_id} --actor cli-user",
                "execute": f"khub labs ops action-execute --action-id {action_id} --actor cli-user",
                "resolve": f"khub labs ops action-resolve --action-id {action_id} --actor cli-user",
            },
        },
        "dryRun": dry_run_payload,
        "writebackPreview": writeback_preview,
        "gateway": build_writeback_request_gateway_metadata(),
        "warnings": [
            *[str(item) for item in (dry_run_payload.get("warnings") or []) if str(item).strip()],
            *([preview_context_error] if preview_context_error else []),
        ],
    }
    deps.validate_cli_payload(config, payload, "knowledge-hub.agent.writeback.request.result.v1")
    return payload
