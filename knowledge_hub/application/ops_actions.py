"""Safe execution helpers for queued ops actions."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from knowledge_hub.application.paper_source_repairs import (
    summarize_paper_source_repair_queue_result,
    summarize_paper_source_repair_result,
)
from knowledge_hub.application.paper_reports import verify_paper_source_state

SAFE_ACTION_POLICY_VERSION = "ops-safe-runner-v1"
_JSON_CAPABLE_ACTIONS = {
    "agent_repo_writeback_request",
    "inspect_review_queue",
    "remediate_section",
    "inspect_rag_samples",
    "review_answer_routes",
    "repair_paper_source",
}
_SAFE_ACTION_TYPES = {
    "agent_repo_writeback_request",
    "inspect_review_queue",
    "remediate_section",
    "inspect_rag_samples",
    "inspect_verification_routes",
    "review_answer_routes",
    "repair_paper_source",
}


@dataclass
class _ValidationResult:
    action_type: str
    command: str
    args: list[str]
    json_capable: bool


def _truncate(text: Any, limit: int = 280) -> str:
    value = " ".join(str(text or "").strip().split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _flag_value(args: list[str], flag: str) -> str:
    for index, item in enumerate(args):
        if item != flag:
            continue
        if index + 1 < len(args):
            return str(args[index + 1])
        return ""
    return ""


def _has_flag(args: list[str], flag: str) -> bool:
    return any(str(item) == str(flag) for item in args)


def _flag_values(args: list[str], flag: str) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(args):
        if item != flag:
            continue
        if index + 1 < len(args):
            values.append(str(args[index + 1]))
    return values


def _normalize_rel_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    return raw.strip("/")


def _normalize_prefixes(prefixes: Any) -> list[str]:
    normalized: list[str] = []
    for value in list(prefixes or []):
        token = _normalize_rel_path(value)
        if not token:
            continue
        if not token.endswith("/"):
            token = token + "/"
        if token not in normalized:
            normalized.append(token)
    return normalized


def _coerce_artifact(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"items": value}
    return {"textExcerpt": _truncate(value, 1200)}


def _summary_from_artifact(artifact: Any) -> str:
    if isinstance(artifact, dict):
        schema = str(artifact.get("schema") or "").strip()
        if schema == "knowledge-hub.foundry.agent.run.result.v1":
            status = str(artifact.get("status") or "").strip()
            stage = str(artifact.get("stage") or "").strip()
            writeback = dict(artifact.get("writeback") or {})
            parts = [schema]
            if status:
                parts.append(f"status={status}")
            if stage:
                parts.append(f"stage={stage}")
            if writeback:
                parts.append(f"writeback={'ok' if bool(writeback.get('ok')) else 'blocked'}")
                detail = str(writeback.get("detail") or "").strip()
                if detail:
                    parts.append(detail)
            return _truncate(" ".join(parts))
        if schema == "knowledge-hub.paper.source-repair.result.v1":
            return _truncate(summarize_paper_source_repair_result(artifact))
        if schema == "knowledge-hub.paper.source-repair.queue.result.v1":
            return _truncate(summarize_paper_source_repair_queue_result(artifact))
        status = str(artifact.get("status") or "").strip()
        count = artifact.get("count")
        attempted = artifact.get("attempted")
        summary = str(artifact.get("summary") or "").strip()
        if summary:
            return _truncate(summary)
        if schema and status:
            suffix = f" count={count}" if isinstance(count, int) else f" attempted={attempted}" if isinstance(attempted, int) else ""
            return _truncate(f"{schema} status={status}{suffix}")
        if artifact.get("summary"):
            return _truncate(artifact.get("summary"))
    return _truncate(artifact)


def queue_item_view(item: dict[str, Any], *, latest_receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    receipt = latest_receipt or {}
    action_payload = dict(item.get("action_json") or item.get("action") or {})
    return {
        "actionId": str(item.get("action_id") or ""),
        "scope": str(item.get("scope") or ""),
        "actionType": str(item.get("action_type") or ""),
        "status": str(item.get("status") or ""),
        "targetKind": str(item.get("target_kind") or ""),
        "targetKey": str(item.get("target_key") or ""),
        "summary": str(item.get("summary") or ""),
        "reasonCodes": [str(code) for code in (item.get("reason_codes_json") or [])],
        "command": str(item.get("command") or ""),
        "args": [str(arg) for arg in (item.get("args_json") or [])],
        "alerts": list(item.get("alert_json") or []),
        "action": action_payload,
        "seenCount": int(item.get("seen_count") or 0),
        "firstSeenAt": str(item.get("first_seen_at") or ""),
        "lastSeenAt": str(item.get("last_seen_at") or ""),
        "ackedAt": str(item.get("acked_at") or ""),
        "ackedBy": str(item.get("acked_by") or ""),
        "resolvedAt": str(item.get("resolved_at") or ""),
        "resolvedBy": str(item.get("resolved_by") or ""),
        "note": str(item.get("note") or ""),
        "lastExecutionAt": str(receipt.get("executed_at") or ""),
        "lastExecutionStatus": str(receipt.get("status") or ""),
        "lastResultSummary": str(receipt.get("result_summary") or ""),
        "lastReceiptId": str(receipt.get("receipt_id") or ""),
        "lastMcpJobId": str(receipt.get("mcp_job_id") or ""),
    }


def receipt_view(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "receiptId": str(receipt.get("receipt_id") or ""),
        "actionId": str(receipt.get("action_id") or ""),
        "executedAt": str(receipt.get("executed_at") or ""),
        "mode": str(receipt.get("mode") or ""),
        "status": str(receipt.get("status") or ""),
        "runner": str(receipt.get("runner") or ""),
        "command": str(receipt.get("command") or ""),
        "args": [str(arg) for arg in (receipt.get("args_json") or [])],
        "mcpJobId": str(receipt.get("mcp_job_id") or ""),
        "resultSummary": str(receipt.get("result_summary") or ""),
        "errorSummary": str(receipt.get("error_summary") or ""),
        "artifact": dict(receipt.get("artifact_json") or {}),
        "actor": str(receipt.get("actor") or ""),
        "updatedAt": str(receipt.get("updated_at") or ""),
    }


def _paper_action_config(item: dict[str, Any]) -> tuple[str, str]:
    action_payload = dict(item.get("action_json") or item.get("action") or {})
    paper_id = str(action_payload.get("paperId") or "").strip()
    parser = str(action_payload.get("documentMemoryParser") or "").strip() or "raw"
    args = [str(arg) for arg in (item.get("args_json") or item.get("args") or [])]
    if not paper_id and "--paper-id" in args:
        index = args.index("--paper-id")
        if index + 1 < len(args):
            paper_id = str(args[index + 1]).strip()
    if "--document-memory-parser" in args:
        index = args.index("--document-memory-parser")
        if index + 1 < len(args):
            parser = str(args[index + 1]).strip() or parser
    return paper_id, parser


def finalize_executed_action(*, sqlite_db, item: dict[str, Any], result: dict[str, Any], actor: str) -> tuple[dict, dict, list[str]]:
    artifact = dict(result.get("artifact") or {})
    warnings = [str(w) for w in (result.get("warnings") or [])]
    updated_item = dict(item or {})
    action_scope = str(item.get("scope") or "")
    action_type = str(item.get("action_type") or "")
    result_status = str(result.get("status") or "")
    resolver = getattr(sqlite_db, "set_ops_action_status", None)

    if action_scope == "paper" and action_type == "repair_paper_source" and result_status == "ok":
        paper_id, parser = _paper_action_config(item)
        if not paper_id:
            warnings.append("post-execution verification skipped: missing paperId")
            return updated_item, artifact, warnings
        verification = verify_paper_source_state(
            sqlite_db,
            paper_id=paper_id,
            document_memory_parser=parser,
        )
        artifact["verification"] = verification
        if bool(verification.get("resolved")) and callable(resolver):
            resolved_item = resolver(
                str(item.get("action_id") or ""),
                status="resolved",
                actor=str(actor).strip(),
                note=str(verification.get("summary") or ""),
            )
            if resolved_item:
                updated_item = dict(resolved_item)
        else:
            summary = str(verification.get("summary") or "").strip()
            if summary:
                warnings.append(f"post-execution verification: {summary}")
        return updated_item, artifact, warnings

    if action_scope == "agent" and action_type == "agent_repo_writeback_request" and result_status == "ok":
        writeback = dict(artifact.get("writeback") or {})
        if bool(writeback.get("ok")) and callable(resolver):
            note = _summary_from_artifact(artifact)
            resolved_item = resolver(
                str(item.get("action_id") or ""),
                status="resolved",
                actor=str(actor).strip(),
                note=note,
            )
            if resolved_item:
                updated_item = dict(resolved_item)
        else:
            detail = str(writeback.get("detail") or "").strip()
            if detail:
                warnings.append(f"post-execution writeback: {detail}")
        return updated_item, artifact, warnings

    return updated_item, artifact, warnings


class OpsActionExecutor:
    def __init__(self, *, invoke_cli: Any = None):
        self._invoke_cli = invoke_cli or self._default_invoke_cli

    def validate(self, action_item: dict[str, Any]) -> _ValidationResult:
        action_type = str(action_item.get("action_type") or action_item.get("actionType") or "").strip()
        command = str(action_item.get("command") or "").strip()
        args = [str(arg) for arg in (action_item.get("args_json") or action_item.get("args") or [])]
        if action_type not in _SAFE_ACTION_TYPES:
            raise ValueError(f"action type is not safe to execute: {action_type or 'unknown'}")
        if command != "khub":
            raise ValueError(f"unsupported command for safe runner: {command or 'unknown'}")
        if action_type == "inspect_review_queue":
            if args[:3] != ["labs", "crawl", "ko-note-review-list"] or not _flag_value(args, "--run-id"):
                raise ValueError("inspect_review_queue args do not match safe allowlist")
        elif action_type == "remediate_section":
            if args[:3] != ["labs", "crawl", "ko-note-remediate"]:
                raise ValueError("remediate_section args do not match safe allowlist")
            if _flag_value(args, "--strategy") not in {"", "section"}:
                raise ValueError("safe runner only allows section remediation")
            if "full" in args or _has_flag(args, "--allow-external"):
                raise ValueError("safe runner blocks full remediation and explicit external execution")
            if "--run-id" not in args:
                raise ValueError("safe runner requires run_id for remediation")
            normalized_args: list[str] = []
            skip_next = False
            for index, token in enumerate(args):
                if skip_next:
                    skip_next = False
                    continue
                if token == "--strategy":
                    skip_next = True
                    continue
                normalized_args.append(token)
            args = normalized_args + ["--strategy", "section"]
            if "--no-allow-external" not in args:
                args.append("--no-allow-external")
        elif action_type in {"inspect_rag_samples", "review_answer_routes"}:
            if args[:1] != ["rag-report"]:
                raise ValueError(f"{action_type} args do not match rag-report safe allowlist")
        elif action_type == "inspect_verification_routes":
            if args[:2] != ["config", "list"]:
                raise ValueError("inspect_verification_routes args do not match config list safe allowlist")
        elif action_type == "repair_paper_source":
            if args[:2] != ["paper", "repair-source"]:
                raise ValueError("repair_paper_source args do not match paper repair-source safe allowlist")
            if _has_flag(args, "--paper-id-file"):
                raise ValueError("safe runner blocks file-based paper source repair actions")
            paper_ids = _flag_values(args, "--paper-id")
            if len(paper_ids) != 1 or not str(paper_ids[0]).strip():
                raise ValueError("safe runner requires exactly one --paper-id for paper source repair")
            if _has_flag(args, "--allow-external"):
                raise ValueError("safe runner blocks external paper source repair execution")
            for blocked_flag in ("--provider", "--model", "--llm-mode"):
                if _has_flag(args, blocked_flag):
                    raise ValueError(f"safe runner blocks {blocked_flag} override for paper source repair")
            parser = _flag_value(args, "--document-memory-parser")
            if parser and parser not in {"raw", "pymupdf", "mineru", "opendataloader"}:
                raise ValueError("safe runner only allows known document-memory parsers")
            if not parser:
                args = list(args) + ["--document-memory-parser", "raw"]
            if "--no-allow-external" not in args:
                args.append("--no-allow-external")
        elif action_type == "agent_repo_writeback_request":
            if args[:2] != ["agent", "run"]:
                raise ValueError("agent_repo_writeback_request args do not match agent run safe allowlist")
            if str(action_item.get("status") or action_item.get("status", "")).strip() != "acked":
                raise ValueError("agent_repo_writeback_request requires explicit ack before execution")
            if _has_flag(args, "--dry-run") or _has_flag(args, "--dump-json"):
                raise ValueError("agent_repo_writeback_request must execute the non-dry-run path")
            if _has_flag(args, "--report-path"):
                raise ValueError("safe runner blocks explicit --report-path for agent_repo_writeback_request")
            if _has_flag(args, "--include-workspace") and _has_flag(args, "--no-include-workspace"):
                raise ValueError("agent_repo_writeback_request cannot include both workspace override flags")
            goal = _flag_value(args, "--goal")
            repo_path = _flag_value(args, "--repo-path")
            if not goal:
                raise ValueError("agent_repo_writeback_request requires --goal")
            if not repo_path:
                raise ValueError("agent_repo_writeback_request requires --repo-path")
            action_payload = dict(action_item.get("action_json") or action_item.get("action") or {})
            resolved_repo = Path(repo_path).expanduser()
            if not resolved_repo.exists() or not resolved_repo.is_dir():
                raise ValueError("agent_repo_writeback_request repo path must exist and be a directory")
            allowed_flags = {
                "--goal",
                "--repo-path",
                "--role",
                "--orchestrator-mode",
                "--max-rounds",
                "--max-workspace-files",
                "--include-workspace",
                "--no-include-workspace",
                "--json",
            }
            for token in args[2:]:
                if not str(token).startswith("--"):
                    continue
                if token not in allowed_flags:
                    raise ValueError(f"safe runner blocks unsupported agent writeback flag: {token}")
            target_policy = str(action_payload.get("targetPolicy") or "").strip()
            if target_policy == "docs_only":
                allowed_prefixes = _normalize_prefixes(action_payload.get("allowedPathPrefixes") or [])
                if not allowed_prefixes:
                    raise ValueError("docs_only agent writeback request requires allowed path prefixes")
                preview = dict(action_payload.get("writebackPreview") or {})
                targets = list(preview.get("targets") or [])
                if not targets:
                    raise ValueError("docs_only agent writeback request requires at least one predicted target")
                for target in targets:
                    rel_path = _normalize_rel_path(
                        target.get("relativePath") if isinstance(target, dict) else ""
                    )
                    if not rel_path:
                        raise ValueError("docs_only agent writeback request contains an empty preview target")
                    if not any(rel_path.startswith(prefix) for prefix in allowed_prefixes):
                        raise ValueError(
                            f"docs_only agent writeback request preview target is outside allowed prefixes: {rel_path}"
                        )
        return _ValidationResult(
            action_type=action_type,
            command=command,
            args=args,
            json_capable=action_type in _JSON_CAPABLE_ACTIONS,
        )

    def execute_sync(
        self,
        *,
        action_item: dict[str, Any],
        khub: Any | None = None,
        config_path: str | None = None,
    ) -> dict[str, Any]:
        validated = self.validate(action_item)
        args = list(validated.args)
        if validated.json_capable and "--json" not in args:
            args.append("--json")
        invocation = self._invoke_cli(args, khub=khub, config_path=config_path)
        raw_exit_code = invocation.get("exit_code", 1)
        exit_code = int(raw_exit_code if raw_exit_code is not None else 1)
        output = str(invocation.get("output") or "")
        parsed_artifact: Any = {}
        if validated.json_capable and output.strip():
            try:
                parsed_artifact = json.loads(output)
            except Exception:
                parsed_artifact = {"textExcerpt": _truncate(output, 1200)}
        elif output.strip():
            parsed_artifact = {"textExcerpt": _truncate(output, 1200)}

        if exit_code != 0:
            error_summary = _truncate(
                invocation.get("error")
                or (invocation.get("exception") and str(invocation.get("exception")))
                or output
                or "execution failed"
            )
            return {
                "status": "failed",
                "command": validated.command,
                "args": args,
                "resultSummary": "",
                "errorSummary": error_summary,
                "artifact": _coerce_artifact(parsed_artifact),
                "warnings": [error_summary],
            }
        artifact = _coerce_artifact(parsed_artifact)
        return {
            "status": "ok",
            "command": validated.command,
            "args": args,
            "resultSummary": _summary_from_artifact(parsed_artifact or output),
            "errorSummary": "",
            "artifact": artifact,
            "warnings": [],
        }

    def _default_invoke_cli(self, args: list[str], *, khub: Any | None = None, config_path: str | None = None) -> dict[str, Any]:
        from click.testing import CliRunner

        from knowledge_hub.interfaces.cli.main import KhubContext, cli

        runner = CliRunner()
        khub_obj = khub if khub is not None else KhubContext(config_path)
        result = runner.invoke(cli, args, obj={"khub": khub_obj})
        return {
            "exit_code": int(result.exit_code),
            "output": str(result.output or ""),
            "exception": result.exception,
            "error": str(result.exception) if result.exception else "",
        }
