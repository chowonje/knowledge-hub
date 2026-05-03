from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Any

import anyio


CODEX_BACKEND_NAME = "codex_mcp"
MCP_DEPENDENCY_MISSING_SUMMARY = "mcp dependency unavailable; install knowledge-hub-cli[mcp]"


class CodexMCPDependencyMissing(RuntimeError):
    pass


def _load_mcp_stdio_client() -> tuple[Any, Any, Any]:
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ModuleNotFoundError as error:
        if str(getattr(error, "name", "")) == "mcp":
            raise CodexMCPDependencyMissing(MCP_DEPENDENCY_MISSING_SUMMARY) from error
        raise
    return ClientSession, StdioServerParameters, stdio_client


def _config_value(config: Any, *keys: str, default: Any = None) -> Any:
    getter = getattr(config, "get_nested", None)
    if callable(getter):
        return getter(*keys, default=default)
    return default


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _preview_text(value: Any, *, limit: int = 1200) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)].rstrip()}…"


def sanitize_answer_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s*\[(?:\d+(?:\s*,\s*\d+)*)\]", "", text)
    text = re.sub(r"\s*\((?:source|sources?)\s+\d+(?:\s*,\s*\d+)*\)", "", text, flags=re.IGNORECASE)
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        line = line.replace(" .", ".").replace(" ,", ",")
        line = line.replace(" !", "!").replace(" ?", "?").replace(" :", ":").replace(" ;", ";")
        lines.append(line)
    sanitized = "\n".join(lines)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized.strip()


def default_codex_env() -> dict[str, str]:
    env = dict(os.environ)
    config_pairs = os.getenv("KHUB_CODEX_ENV", "").strip() or os.getenv("KHUB_CODEX_MCP_ENV", "")
    if config_pairs:
        for pair in config_pairs.split(","):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            key, _, value = pair.partition("=")
            env[key.strip()] = value.strip()
    return env


def resolve_codex_transport(config: Any, *, task_type: str = "rag_answer") -> str:
    token = (
        os.getenv("KHUB_CODEX_TRANSPORT", "").strip().lower()
        or str(
            _config_value(
                config,
                "routing",
                "llm",
                "tasks",
                task_type,
                "codex",
                "transport",
                default="",
            )
            or ""
        ).strip().lower()
        or str(_config_value(config, "eval", "answer_loop", "codex", "transport", default="") or "").strip().lower()
        or "exec"
    )
    if token not in {"exec", "mcp"}:
        raise ValueError(f"unsupported codex transport: {token}")
    return token


def resolve_codex_server_config(config: Any, *, task_type: str = "rag_answer") -> dict[str, Any]:
    command = (
        os.getenv("KHUB_CODEX_MCP_COMMAND", "").strip()
        or str(
            _config_value(
                config,
                "routing",
                "llm",
                "tasks",
                task_type,
                "codex",
                "command",
                default="",
            )
            or ""
        ).strip()
        or str(_config_value(config, "eval", "answer_loop", "codex", "command", default="") or "").strip()
        or "codex"
    )
    raw_args = (
        os.getenv("KHUB_CODEX_MCP_ARGS", "").strip()
        or str(
            _config_value(
                config,
                "routing",
                "llm",
                "tasks",
                task_type,
                "codex",
                "args",
                default="",
            )
            or ""
        ).strip()
        or str(_config_value(config, "eval", "answer_loop", "codex", "args", default="") or "").strip()
        or "mcp-server"
    )
    timeout_seconds = int(
        os.getenv("KHUB_CODEX_MCP_TIMEOUT_SECONDS", "").strip()
        or _config_value(
            config,
            "routing",
            "llm",
            "tasks",
            task_type,
            "codex",
            "timeout_seconds",
            default=0,
        )
        or _config_value(config, "eval", "answer_loop", "codex", "timeout_seconds", default=300)
        or 300
    )
    return {
        "command": command,
        "args": shlex.split(raw_args),
        "env": default_codex_env(),
        "timeoutSeconds": timeout_seconds,
    }


def resolve_codex_exec_config(config: Any, *, task_type: str = "rag_answer") -> dict[str, Any]:
    command = (
        os.getenv("KHUB_CODEX_EXEC_COMMAND", "").strip()
        or os.getenv("KHUB_CODEX_MCP_COMMAND", "").strip()
        or str(
            _config_value(
                config,
                "routing",
                "llm",
                "tasks",
                task_type,
                "codex",
                "exec_command",
                default="",
            )
            or ""
        ).strip()
        or str(
            _config_value(
                config,
                "routing",
                "llm",
                "tasks",
                task_type,
                "codex",
                "command",
                default="",
            )
            or ""
        ).strip()
        or str(_config_value(config, "eval", "answer_loop", "codex", "exec_command", default="") or "").strip()
        or str(_config_value(config, "eval", "answer_loop", "codex", "command", default="") or "").strip()
        or "codex"
    )
    raw_args = (
        os.getenv("KHUB_CODEX_EXEC_ARGS", "").strip()
        or str(
            _config_value(
                config,
                "routing",
                "llm",
                "tasks",
                task_type,
                "codex",
                "exec_args",
                default="",
            )
            or ""
        ).strip()
        or str(_config_value(config, "eval", "answer_loop", "codex", "exec_args", default="") or "").strip()
    )
    timeout_seconds = int(
        os.getenv("KHUB_CODEX_EXEC_TIMEOUT_SECONDS", "").strip()
        or os.getenv("KHUB_CODEX_MCP_TIMEOUT_SECONDS", "").strip()
        or _config_value(
            config,
            "routing",
            "llm",
            "tasks",
            task_type,
            "codex",
            "timeout_seconds",
            default=0,
        )
        or _config_value(config, "eval", "answer_loop", "codex", "timeout_seconds", default=300)
        or 300
    )
    return {
        "command": command,
        "args": shlex.split(raw_args),
        "env": default_codex_env(),
        "timeoutSeconds": timeout_seconds,
    }


async def _call_codex_tool(
    *,
    config: Any,
    prompt: str,
    cwd: str,
    sandbox: str,
    approval_policy: str,
    model: str = "",
    include_plan_tool: bool = False,
    task_type: str = "rag_answer",
) -> dict[str, Any]:
    server = resolve_codex_server_config(config, task_type=task_type)
    ClientSession, StdioServerParameters, stdio_client = _load_mcp_stdio_client()
    params = StdioServerParameters(
        command=str(server["command"]),
        args=list(server["args"]),
        env=dict(server["env"]),
        cwd=str(cwd),
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            arguments = {
                "prompt": prompt,
                "cwd": str(cwd),
                "sandbox": str(sandbox),
                "approval-policy": str(approval_policy),
                "include-plan-tool": bool(include_plan_tool),
            }
            if model:
                arguments["model"] = str(model)
            result = await session.call_tool(
                "codex",
                arguments=arguments,
                read_timeout_seconds=timedelta(seconds=int(server["timeoutSeconds"])),
            )
    text_content = []
    for item in list(result.content or []):
        text = getattr(item, "text", "")
        if text:
            text_content.append(str(text))
    structured = dict(result.structuredContent or {})
    return {
        "isError": bool(getattr(result, "isError", False)),
        "threadId": str(structured.get("threadId") or ""),
        "content": "\n".join(part for part in text_content if part).strip(),
        "structuredContent": structured,
    }


def _session_id_from_output(text: str) -> str:
    match = re.search(r"session id:\s*([^\s]+)", str(text or ""), flags=re.IGNORECASE)
    return str(match.group(1)) if match else ""


def _run_codex_exec_sync(
    *,
    config: Any,
    prompt: str,
    cwd: str,
    sandbox: str,
    approval_policy: str,
    model: str = "",
    include_plan_tool: bool = False,
    task_type: str = "rag_answer",
) -> dict[str, Any]:
    _ = include_plan_tool
    runtime = resolve_codex_exec_config(config, task_type=task_type)
    output_file = tempfile.NamedTemporaryFile(prefix="khub-codex-", suffix=".txt", delete=False)
    output_path = Path(output_file.name)
    output_file.close()
    command = [
        str(runtime["command"]),
        "exec",
        "--ephemeral",
        "-C",
        str(cwd),
        "-s",
        str(sandbox),
        "-c",
        f'approval_policy="{approval_policy}"',
        "-o",
        str(output_path),
    ]
    if model:
        command.extend(["-m", str(model)])
    command.extend(list(runtime["args"]))
    command.append("-")
    completed = subprocess.run(
        command,
        input=str(prompt),
        capture_output=True,
        text=True,
        env=dict(runtime["env"]),
        timeout=int(runtime["timeoutSeconds"]),
        cwd=str(cwd),
        check=False,
    )
    stdout = str(completed.stdout or "")
    stderr = str(completed.stderr or "")
    output_text = output_path.read_text(encoding="utf-8", errors="ignore").strip() if output_path.exists() else ""
    try:
        output_path.unlink(missing_ok=True)
    except Exception:
        pass
    warnings: list[str] = []
    if completed.returncode != 0:
        detail = _preview_text(stderr or stdout or f"codex exec exited {completed.returncode}", limit=1200)
        warnings.append(detail)
    return {
        "isError": bool(completed.returncode != 0 and not output_text),
        "threadId": _session_id_from_output(stdout + "\n" + stderr),
        "content": output_text,
        "structuredContent": {
            "transport": "exec",
            "returncode": int(completed.returncode),
            "stdoutPreview": _preview_text(stdout, limit=1200),
            "stderrPreview": _preview_text(stderr, limit=1200),
            "warnings": warnings,
        },
    }


def run_codex_tool_sync(
    *,
    config: Any,
    prompt: str,
    cwd: str,
    sandbox: str,
    approval_policy: str,
    model: str = "",
    include_plan_tool: bool = False,
    task_type: str = "rag_answer",
) -> dict[str, Any]:
    transport = resolve_codex_transport(config, task_type=task_type)
    if transport == "mcp":
        return anyio.run(
            lambda: _call_codex_tool(
                config=config,
                prompt=prompt,
                cwd=cwd,
                sandbox=sandbox,
                approval_policy=approval_policy,
                model=model,
                include_plan_tool=include_plan_tool,
                task_type=task_type,
            )
        )
    return _run_codex_exec_sync(
        config=config,
        prompt=prompt,
        cwd=cwd,
        sandbox=sandbox,
        approval_policy=approval_policy,
        model=model,
        include_plan_tool=include_plan_tool,
        task_type=task_type,
    )


def codex_backend_readiness(config: Any, *, task_type: str = "rag_answer") -> dict[str, Any]:
    try:
        transport = resolve_codex_transport(config, task_type=task_type)
        if transport == "mcp":
            _load_mcp_stdio_client()
        runtime = (
            resolve_codex_server_config(config, task_type=task_type)
            if transport == "mcp"
            else resolve_codex_exec_config(config, task_type=task_type)
        )
    except CodexMCPDependencyMissing as error:
        return {
            "available": False,
            "provider": CODEX_BACKEND_NAME,
            "transport": "mcp",
            "command": "",
            "reason": "dependency_missing",
            "summary": str(error),
        }
    except Exception as error:
        return {
            "available": False,
            "provider": CODEX_BACKEND_NAME,
            "transport": "",
            "command": "",
            "reason": "config_error",
            "summary": f"{type(error).__name__}: {error}",
        }

    command = str(runtime.get("command") or "").strip()
    if not command:
        return {
            "available": False,
            "provider": CODEX_BACKEND_NAME,
            "transport": transport,
            "command": "",
            "reason": "missing_command",
            "summary": "codex command not configured",
        }

    exists = False
    if os.path.sep in command or command.startswith("."):
        exists = Path(command).expanduser().exists()
    else:
        exists = shutil.which(command) is not None

    if not exists:
        return {
            "available": False,
            "provider": CODEX_BACKEND_NAME,
            "transport": transport,
            "command": command,
            "reason": "command_not_found",
            "summary": f"codex command not found: {command}",
        }

    return {
        "available": True,
        "provider": CODEX_BACKEND_NAME,
        "transport": transport,
        "command": command,
        "reason": "",
        "summary": "ready",
        "timeoutSeconds": int(runtime.get("timeoutSeconds") or 0),
    }


def _preferred_backend(config: Any, *, task_type: str = "rag_answer") -> str:
    return str(_config_value(config, "routing", "llm", "tasks", task_type, "preferred_backend", default="") or "").strip().lower()


def _preferred_backend_model(config: Any, *, task_type: str = "rag_answer") -> str:
    return str(
        _config_value(config, "routing", "llm", "tasks", task_type, "preferred_backend_model", default="") or ""
    ).strip()


def resolve_preferred_codex_backend(
    *,
    config: Any,
    allow_external: bool,
    force_route: str | None = None,
    task_type: str = "rag_answer",
) -> tuple[Any | None, dict[str, Any] | None, list[str]]:
    requested_reason = ""
    forced = str(force_route or "").strip().lower()
    if forced == "codex":
        requested_reason = "force_route=codex"
    elif forced not in {"", "auto", "default", "api", "local"}:
        return None, None, []
    elif _preferred_backend(config, task_type=task_type) == CODEX_BACKEND_NAME:
        requested_reason = f"preferred_backend={CODEX_BACKEND_NAME}"

    if not requested_reason:
        return None, None, []

    if not allow_external:
        return None, None, [f"{CODEX_BACKEND_NAME} backend skipped: allow_external disabled"]

    readiness = codex_backend_readiness(config, task_type=task_type)
    if not readiness.get("available"):
        return None, None, [f"{CODEX_BACKEND_NAME} backend unavailable: {readiness.get('summary') or readiness.get('reason') or 'unknown'}"]

    model = _preferred_backend_model(config, task_type=task_type)
    llm = CodexPromptLLM(config=config, model=model, task_type=task_type)
    decision = {
        "route": "api",
        "provider": CODEX_BACKEND_NAME,
        "model": model,
        "reasons": [
            requested_reason,
            f"transport={readiness.get('transport') or 'unknown'}",
        ],
        "timeoutSec": int(readiness.get("timeoutSeconds") or 0),
        "fallbackUsed": False,
    }
    return llm, decision, []


@dataclass
class CodexPromptLLM:
    config: Any
    model: str = ""
    task_type: str = "rag_answer"

    def __post_init__(self) -> None:
        self.model = str(self.model or "").strip()
        self.last_response: dict[str, Any] = {}
        self.last_policy: dict[str, Any] = {}

    def _prompt(self, prompt: str, context: str) -> str:
        body = str(prompt or "").strip()
        ctx = str(context or "").strip()
        if not ctx:
            return body
        if not body:
            return ctx
        return f"{body}\n\nContext:\n{ctx}"

    def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        _ = max_tokens
        response = run_codex_tool_sync(
            config=self.config,
            prompt=self._prompt(prompt, context),
            cwd=os.getcwd(),
            sandbox="read-only",
            approval_policy="never",
            model=self.model,
            include_plan_tool=False,
            task_type=self.task_type,
        )
        self.last_response = dict(response or {})
        self.last_policy = {
            "provider": CODEX_BACKEND_NAME,
            "transport": str((response.get("structuredContent") or {}).get("transport") or ""),
            "threadId": str(response.get("threadId") or ""),
        }
        if response.get("isError"):
            detail = _preview_text(
                (response.get("structuredContent") or {}).get("stderrPreview")
                or (response.get("structuredContent") or {}).get("warnings")
                or "codex backend returned isError=true",
                limit=400,
            )
            raise RuntimeError(detail or "codex backend returned isError=true")
        return sanitize_answer_text(response.get("content"))

    def stream_generate(self, prompt: str, context: str = "", max_tokens: int | None = None):
        text = self.generate(prompt, context, max_tokens=max_tokens)
        for chunk in re.findall(r".{1,256}", text, flags=re.DOTALL):
            yield chunk


__all__ = [
    "CODEX_BACKEND_NAME",
    "CodexPromptLLM",
    "codex_backend_readiness",
    "default_codex_env",
    "resolve_codex_exec_config",
    "resolve_codex_server_config",
    "resolve_codex_transport",
    "resolve_preferred_codex_backend",
    "run_codex_tool_sync",
    "sanitize_answer_text",
]
