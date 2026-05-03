from __future__ import annotations

import asyncio
import inspect
import os
import time
from typing import Any
from uuid import uuid4

from knowledge_hub.application.mcp.responses import (
    DEFAULT_MCP_TOOL_TIMEOUT_SECONDS,
    MCP_TOOL_STATUS_BLOCKED,
    MCP_TOOL_STATUS_DONE,
    MCP_TOOL_STATUS_EXPIRED,
    MCP_TOOL_STATUS_FAILED,
    MCP_TOOL_STATUS_QUEUED,
    MCP_TOOL_STATUS_RUNNING,
    build_mcp_tool_response,
    build_verify_block,
    collect_source_refs,
    infer_classification,
    now_iso,
    to_float,
)

ACTIVE_MCP_JOBS: dict[str, asyncio.Task[None]] = {}


async def run_async_tool(
    *,
    sqlite_db: Any,
    active_jobs: dict[str, asyncio.Task[None]] | None = None,
    tool: str,
    request_echo: dict[str, Any],
    sync_job: Any,
    started_at: str | None = None,
    build_verify_block_fn: Any = None,
    build_mcp_tool_response_fn: Any = None,
    collect_source_refs_fn: Any = None,
    infer_classification_fn: Any = None,
    to_float_fn: Any = None,
    now_iso_fn: Any = None,
    status_done: str = MCP_TOOL_STATUS_DONE,
    status_failed: str = MCP_TOOL_STATUS_FAILED,
    status_blocked: str = MCP_TOOL_STATUS_BLOCKED,
    status_queued: str = MCP_TOOL_STATUS_QUEUED,
    status_running: str = MCP_TOOL_STATUS_RUNNING,
    status_expired: str = MCP_TOOL_STATUS_EXPIRED,
    on_queued: Any = None,
    on_finished: Any = None,
) -> tuple[str, dict[str, Any]]:
    if sqlite_db is None:
        raise RuntimeError("sqlite_db is not initialized")

    jobs = active_jobs if active_jobs is not None else ACTIVE_MCP_JOBS
    verify_block = build_verify_block_fn or build_verify_block
    response_builder = build_mcp_tool_response_fn or build_mcp_tool_response
    source_ref_builder = collect_source_refs_fn or collect_source_refs
    classification_builder = infer_classification_fn or infer_classification
    float_builder = to_float_fn or to_float
    clock = now_iso_fn or now_iso
    started = started_at or clock()
    job_id = str(uuid4())
    sqlite_db.create_mcp_job(
        job_id=job_id,
        tool=tool,
        request=request_echo,
        request_echo=request_echo,
        status=status_queued,
        classification=classification_builder(request_echo),
        progress=0,
        actor="codex",
    )
    if callable(on_queued):
        queued_result = on_queued(job_id, started)
        if inspect.isawaitable(queued_result):
            await queued_result

    async def _runner():
        run_started = time.perf_counter()
        started_iso = clock()
        sqlite_db.update_mcp_job(
            job_id,
            status=status_running,
            started_at=started_iso,
            updated_at=started_iso,
            progress=10,
        )
        tool_timeout = int(
            float_builder(
                os.getenv("KHUB_MCP_TOOL_TIMEOUT_SECONDS"),
                DEFAULT_MCP_TOOL_TIMEOUT_SECONDS,
                minimum=30,
                maximum=86400,
            )
        )
        try:
            if inspect.iscoroutinefunction(sync_job):
                payload = await asyncio.wait_for(sync_job(), timeout=tool_timeout)
            else:
                payload = await asyncio.wait_for(asyncio.to_thread(sync_job), timeout=tool_timeout)
                if inspect.isawaitable(payload):
                    payload = await payload
            if not isinstance(payload, dict):
                payload = {"status": "ok", "data": payload}

            verify = verify_block(payload, "ok", tool)
            final_status = status_done
            error_message = None
            final_artifact = payload

            if not verify["policyAllowed"]:
                final_status = status_blocked
                error_message = "policy denied"
                final_artifact = "[REDACTED_BY_POLICY]"
            elif not verify["schemaValid"]:
                final_status = status_failed
                if verify["schemaErrors"]:
                    error_message = "; ".join(verify["schemaErrors"])

            finished_iso = clock()
            elapsed_ms = int((time.perf_counter() - run_started) * 1000)
            sqlite_db.update_mcp_job(
                job_id=job_id,
                status=final_status,
                started_at=started_iso,
                finished_at=finished_iso,
                updated_at=finished_iso,
                run_time_ms=elapsed_ms,
                progress=100,
                source_refs=source_ref_builder(payload),
                policy_result=(
                    "blocked"
                    if final_status == status_blocked
                    else "failed"
                    if final_status == status_failed
                    else "allowed"
                ),
                error=error_message,
                artifact=final_artifact,
                classification=str(verify.get("classification") or classification_builder(final_artifact)),
            )
            if callable(on_finished):
                finished_result = on_finished(
                    job_id,
                    final_status,
                    final_artifact,
                    error_message,
                    started_iso,
                    finished_iso,
                )
                if inspect.isawaitable(finished_result):
                    await finished_result
        except asyncio.TimeoutError:
            finished_iso = clock()
            elapsed_ms = int((time.perf_counter() - run_started) * 1000)
            sqlite_db.update_mcp_job(
                job_id=job_id,
                status=status_failed,
                started_at=started_iso,
                finished_at=finished_iso,
                updated_at=finished_iso,
                run_time_ms=elapsed_ms,
                progress=100,
                policy_result="failed",
                error=f"tool timeout after {tool_timeout}s",
            )
            if callable(on_finished):
                finished_result = on_finished(
                    job_id,
                    status_failed,
                    None,
                    f"tool timeout after {tool_timeout}s",
                    started_iso,
                    finished_iso,
                )
                if inspect.isawaitable(finished_result):
                    await finished_result
        except asyncio.CancelledError:
            finished_iso = clock()
            elapsed_ms = int((time.perf_counter() - run_started) * 1000)
            sqlite_db.update_mcp_job(
                job_id=job_id,
                status=status_expired,
                started_at=started_iso,
                finished_at=finished_iso,
                updated_at=finished_iso,
                run_time_ms=elapsed_ms,
                progress=100,
                policy_result="expired",
                error="job cancelled",
            )
            if callable(on_finished):
                finished_result = on_finished(
                    job_id,
                    status_expired,
                    None,
                    "job cancelled",
                    started_iso,
                    finished_iso,
                )
                if inspect.isawaitable(finished_result):
                    await finished_result
            return
        except Exception as error:
            finished_iso = clock()
            elapsed_ms = int((time.perf_counter() - run_started) * 1000)
            sqlite_db.update_mcp_job(
                job_id=job_id,
                status=status_failed,
                started_at=started,
                finished_at=finished_iso,
                updated_at=finished_iso,
                run_time_ms=elapsed_ms,
                progress=100,
                policy_result="failed",
                error=str(error),
            )
            if callable(on_finished):
                finished_result = on_finished(
                    job_id,
                    status_failed,
                    None,
                    str(error),
                    started,
                    finished_iso,
                )
                if inspect.isawaitable(finished_result):
                    await finished_result

    task = asyncio.create_task(_runner())
    jobs[job_id] = task
    _ = task.add_done_callback(lambda _: jobs.pop(job_id, None))
    return job_id, response_builder(
        tool=tool,
        status=status_queued,
        payload={"message": "queued"},
        started_at=started,
        request_echo=request_echo,
        job_id=job_id,
    )
