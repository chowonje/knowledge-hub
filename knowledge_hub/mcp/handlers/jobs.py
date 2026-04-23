from __future__ import annotations

from typing import Any

from knowledge_hub.application.ops_actions import OpsActionExecutor, finalize_executed_action, queue_item_view, receipt_view


async def handle_tool(name: str, arguments: dict[str, Any], ctx: dict[str, Any]):
    emit = ctx["emit"]
    sqlite_db = ctx["sqlite_db"]
    get_sqlite_db = ctx["get_sqlite_db"]
    initialize_core_only = ctx["initialize_core_only"]
    initialize = ctx["initialize"]
    redact_payload = ctx["redact_payload"]
    request_echo = ctx["request_echo"]
    to_bool = ctx["to_bool"]
    to_int = ctx["to_int"]
    to_tool_status = ctx["to_tool_status"]
    now_iso = ctx["now_iso"]
    uuid4 = ctx["uuid4"]

    status_failed = ctx["MCP_TOOL_STATUS_FAILED"]
    status_ok = ctx["MCP_TOOL_STATUS_OK"]
    status_queued = ctx["MCP_TOOL_STATUS_QUEUED"]
    run_async_tool = ctx["run_async_tool"]
    config_path = ctx.get("config_path")

    def _safe_get_merge_item(proposal_id: int):
        getter = getattr(sqlite_db, "get_entity_merge_proposal", None)
        if callable(getter):
            return getter(proposal_id)
        lister = getattr(sqlite_db, "list_entity_merge_proposals", None)
        if not callable(lister):
            return None
        for item in lister(status="", limit=500):
            if int(item.get("id", 0) or 0) == int(proposal_id):
                return item
        return None

    if name == "mcp_job_status":
        job_id = str(arguments.get("job_id", "")).strip()
        if not job_id:
            return emit(status_failed, {"error": "job_id가 필요합니다."}, status_message="missing job_id")
        include_payload = to_bool(arguments.get("include_payload"), default=True)
        if sqlite_db is None:
            initialize_core_only()
            sqlite_db = get_sqlite_db()
        row = sqlite_db.get_mcp_job(job_id)
        if row is None:
            return emit(status_failed, {"error": f"job을 찾을 수 없습니다. {job_id}"}, status_message="job not found")
        status = to_tool_status(str(row.get("status", status_failed)))
        payload_data = row.get("artifact_json") or {}
        if not include_payload:
            payload_data = {}
        payload = {
            "job": {
                "jobId": row.get("job_id"),
                "tool": row.get("tool"),
                "status": status,
                "classification": row.get("classification"),
                "progress": row.get("progress"),
                "error": row.get("error"),
                "runTimeMs": row.get("run_time_ms"),
                "policyResult": row.get("policy_result"),
                "startedAt": row.get("started_at"),
                "finishedAt": row.get("finished_at"),
                "artifact": payload_data,
            },
        }
        return emit(
            status,
            payload=payload,
            started_at=row.get("created_at", ctx["started_at"]),
            finished_at=row.get("finished_at"),
            job_id=job_id,
            request_echo=redact_payload(row.get("request_echo_json", ctx["request_echo"])),
            source_refs=row.get("source_refs_json") or [],
            status_message=row.get("error") or None,
        )

    if name == "mcp_job_list":
        if sqlite_db is None:
            initialize_core_only()
            sqlite_db = get_sqlite_db()
        status = str(arguments.get("status", "")).strip().lower() or None
        if status not in {None, "queued", "running", "done", "failed", "blocked", "expired"}:
            return emit(status_failed, {"error": "invalid status filter"}, status_message="invalid status")
        tool_filter = str(arguments.get("tool", "")).strip() or None
        limit = to_int(arguments.get("limit"), 20, minimum=1)
        jobs = sqlite_db.list_mcp_jobs(status=status, tool=tool_filter, limit=limit)
        payload = {
            "jobs": [
                {
                    "jobId": job.get("job_id"),
                    "tool": job.get("tool"),
                    "status": to_tool_status(str(job.get("status", status_failed))),
                    "classification": job.get("classification"),
                    "progress": job.get("progress"),
                    "runTimeMs": job.get("run_time_ms"),
                    "error": job.get("error"),
                    "request": redact_payload(job.get("request_json")),
                    "requestEcho": redact_payload(job.get("request_echo_json")),
                    "createdAt": job.get("created_at"),
                    "updatedAt": job.get("updated_at"),
                    "sourceRefs": job.get("source_refs_json") or [],
                    "policy": {"result": job.get("policy_result")},
                    "finishedAt": job.get("finished_at"),
                    "startedAt": job.get("started_at"),
                    "policyResult": job.get("policy_result"),
                }
                for job in jobs
            ]
        }
        return emit(status_ok, payload, request_echo=ctx["request_echo"], status_message="jobs listed")

    if name == "mcp_job_cancel":
        job_id = str(arguments.get("job_id", "")).strip()
        if not job_id:
            return emit(status_failed, {"error": "job_id가 필요합니다."}, status_message="missing job_id")
        if sqlite_db is None:
            initialize_core_only()
            sqlite_db = get_sqlite_db()
        canceled = sqlite_db.cancel_mcp_job(job_id)
        active_jobs = ctx["active_jobs"]
        task = active_jobs.pop(job_id, None)
        if task:
            task.cancel()
        return emit(status_ok if canceled else status_failed, {"canceled": canceled}, job_id=job_id, status_message="job cancel requested")

    if name == "ops_action_list":
        if sqlite_db is None:
            initialize_core_only()
            sqlite_db = get_sqlite_db()
        status_filter = str(arguments.get("status", "pending")).strip().lower() or "pending"
        scope_filter = str(arguments.get("scope", "all")).strip().lower() or "all"
        if status_filter not in {"pending", "acked", "resolved", "all"}:
            return emit(status_failed, {"error": "invalid status filter"}, status_message="invalid status")
        if scope_filter not in {"ko_note", "rag", "all"}:
            return emit(status_failed, {"error": "invalid scope filter"}, status_message="invalid scope")
        items = sqlite_db.list_ops_actions(
            status=None if status_filter == "all" else status_filter,
            scope=None if scope_filter == "all" else scope_filter,
            limit=to_int(arguments.get("limit"), 50, minimum=1),
        )
        payload = {
            "schema": "knowledge-hub.ops.action.list.result.v1",
            "status": "ok",
            "count": len(items),
            "counts": sqlite_db.count_ops_actions(),
            "filters": {"status": status_filter, "scope": scope_filter},
            "items": [
                queue_item_view(
                    item,
                    latest_receipt=(
                        getattr(sqlite_db, "get_latest_ops_action_receipt", lambda *_args, **_kwargs: {})(
                            str(item.get("action_id") or "")
                        )
                        or {}
                    ),
                )
                for item in items
            ],
        }
        return emit(status_ok, payload)

    if name in {"ops_action_ack", "ops_action_resolve"}:
        if sqlite_db is None:
            initialize_core_only()
            sqlite_db = get_sqlite_db()
        action_id = str(arguments.get("action_id", "")).strip()
        if not action_id:
            return emit(status_failed, {"error": "action_id가 필요합니다."}, status_message="missing action_id")
        status_value = "acked" if name == "ops_action_ack" else "resolved"
        item = sqlite_db.set_ops_action_status(
            action_id,
            status=status_value,
            actor=str(arguments.get("actor", "mcp-user")).strip() or "mcp-user",
            note=str(arguments.get("note", "")).strip(),
        )
        payload = {
            "schema": "knowledge-hub.ops.action.result.v1",
            "status": "ok" if item else "failed",
            "actionId": action_id,
            "decision": status_value,
            "item": queue_item_view(
                item or {},
                latest_receipt=(
                    getattr(sqlite_db, "get_latest_ops_action_receipt", lambda *_args, **_kwargs: {})(
                        str((item or {}).get("action_id") or "")
                    )
                    or {}
                ),
            ),
        }
        return emit(status_ok if item else status_failed, payload)

    if name == "ops_action_receipts":
        if sqlite_db is None:
            initialize_core_only()
            sqlite_db = get_sqlite_db()
        action_id = str(arguments.get("action_id", "")).strip()
        if not action_id:
            return emit(status_failed, {"error": "action_id가 필요합니다."}, status_message="missing action_id")
        lister = getattr(sqlite_db, "list_ops_action_receipts", None)
        if not callable(lister):
            return emit(status_failed, {"error": "ops action receipt store unavailable"}, status_message="receipt store unavailable")
        receipts = lister(action_id=action_id, limit=to_int(arguments.get("limit"), 20, minimum=1))
        payload = {
            "schema": "knowledge-hub.ops.action.receipts.result.v1",
            "status": "ok",
            "actionId": action_id,
            "count": len(receipts),
            "receipts": [receipt_view(item) for item in receipts],
        }
        return emit(status_ok, payload)

    if name == "ops_action_execute":
        if sqlite_db is None:
            initialize_core_only()
            sqlite_db = get_sqlite_db()
        action_id = str(arguments.get("action_id", "")).strip()
        if not action_id:
            return emit(status_failed, {"error": "action_id가 필요합니다."}, status_message="missing action_id")
        item = sqlite_db.get_ops_action(action_id) if callable(getattr(sqlite_db, "get_ops_action", None)) else None
        if not item:
            return emit(status_failed, {"error": f"ops action을 찾을 수 없습니다. {action_id}"}, status_message="action not found")

        actor = str(arguments.get("actor", "mcp-user")).strip() or "mcp-user"
        receipt_holder: dict[str, Any] = {}

        async def _runner() -> dict[str, Any]:
            executor = OpsActionExecutor()
            try:
                result = executor.execute_sync(action_item=item, config_path=config_path)
            except Exception as error:
                result = {
                    "status": "failed",
                    "command": str(item.get("command") or ""),
                    "args": [str(arg) for arg in (item.get("args_json") or [])],
                    "resultSummary": "",
                    "errorSummary": str(error),
                    "artifact": {},
                    "warnings": [str(error)],
                }
            updated_item, artifact, warnings = finalize_executed_action(
                sqlite_db=sqlite_db,
                item=item,
                result=result,
                actor=actor,
            )
            updater = getattr(sqlite_db, "update_ops_action_receipt", None)
            receipt = receipt_holder.get("receipt") or {}
            receipt_id = str(receipt.get("receipt_id") or "")
            if callable(updater) and receipt_id:
                updated = updater(
                    receipt_id,
                    status="succeeded" if result.get("status") == "ok" else "failed",
                    result_summary=str(result.get("resultSummary") or ""),
                    error_summary=str(result.get("errorSummary") or ""),
                    artifact=artifact,
                    actor=actor,
                    updated_at=now_iso(),
                )
                if updated:
                    receipt_holder["receipt"] = updated
            latest_receipt = receipt_holder.get("receipt") or receipt
            latest_item = sqlite_db.get_ops_action(action_id) if callable(getattr(sqlite_db, "get_ops_action", None)) else updated_item
            return {
                "schema": "knowledge-hub.ops.action.execute.result.v1",
                "status": "ok" if result.get("status") == "ok" else "failed",
                "actionId": action_id,
                "executionMode": "async",
                "item": queue_item_view(latest_item or updated_item, latest_receipt=latest_receipt),
                "receipt": receipt_view(latest_receipt),
                "result": {
                    "summary": str(result.get("resultSummary") or ""),
                    "artifact": artifact,
                    "warnings": warnings,
                },
                "warnings": warnings,
            }

        def _on_queued(job_id: str, started_at_value: str) -> None:
            creator = getattr(sqlite_db, "create_ops_action_receipt", None)
            if not callable(creator):
                return
            receipt = creator(
                action_id=action_id,
                mode="async",
                status="started",
                runner="mcp",
                command=str(item.get("command") or ""),
                args=[str(arg) for arg in (item.get("args_json") or [])],
                mcp_job_id=job_id,
                actor=actor,
                executed_at=started_at_value,
            )
            receipt_holder["receipt"] = receipt

        def _on_finished(
            job_id: str,
            final_status: str,
            final_artifact: Any,
            error_message: str | None,
            _started_at_value: str,
            finished_at_value: str,
        ) -> None:
            updater = getattr(sqlite_db, "update_ops_action_receipt", None)
            receipt = receipt_holder.get("receipt") or {}
            receipt_id = str(receipt.get("receipt_id") or "")
            if not callable(updater) or not receipt_id:
                return
            result_summary = ""
            error_summary = str(error_message or "")
            artifact_payload = {}
            final_receipt_status = "succeeded" if final_status in {status_ok, "done"} else "failed"
            if isinstance(final_artifact, dict):
                if str(final_artifact.get("schema") or "") == "knowledge-hub.ops.action.execute.result.v1":
                    artifact_payload = dict((final_artifact.get("result") or {}).get("artifact") or {})
                    result_summary = str((final_artifact.get("result") or {}).get("summary") or "")
                    if str(final_artifact.get("status") or "") == "failed":
                        final_receipt_status = "failed"
                        if not error_summary:
                            warnings = [str(item) for item in (final_artifact.get("warnings") or []) if str(item).strip()]
                            error_summary = "; ".join(warnings[:3])
                else:
                    artifact_payload = dict(final_artifact.get("artifact") or final_artifact)
                    result_summary = str(final_artifact.get("resultSummary") or "")
                    if not error_summary:
                        error_summary = str(final_artifact.get("errorSummary") or "")
            updater(
                receipt_id,
                status=final_receipt_status,
                mcp_job_id=job_id,
                result_summary=result_summary,
                error_summary=error_summary,
                artifact=artifact_payload,
                actor=actor,
                updated_at=finished_at_value,
            )
            receipt_holder["receipt"] = getattr(sqlite_db, "get_ops_action_receipt", lambda *_args, **_kwargs: receipt)(receipt_id) or receipt

        job_id, _queued = await run_async_tool(
            name=name,
            request_echo=request_echo,
            sync_job=_runner,
            on_queued=_on_queued,
            on_finished=_on_finished,
        )
        queued_receipt = receipt_view(receipt_holder.get("receipt") or {})
        payload = {
            "schema": "knowledge-hub.ops.action.execute.result.v1",
            "status": "queued",
            "actionId": action_id,
            "executionMode": "async",
            "item": queue_item_view(
                item,
                latest_receipt=(
                    getattr(sqlite_db, "get_latest_ops_action_receipt", lambda *_args, **_kwargs: receipt_holder.get("receipt") or {})(
                        action_id
                    )
                    or receipt_holder.get("receipt")
                    or {}
                ),
            ),
            "receipt": queued_receipt,
            "result": {"summary": "queued", "artifact": {}, "warnings": []},
            "warnings": [],
        }
        return emit(status_queued, payload, job_id=job_id, status_message="queued")

    if name == "foundry_conflict_list":
        if sqlite_db is None:
            initialize_core_only()
            sqlite_db = get_sqlite_db()
        payload = {
            "schema": "knowledge-hub.foundry.conflict.list.result.v1",
            "runId": f"mcp_foundry_conflict_list_{uuid4().hex[:10]}",
            "status": "ok",
            "items": sqlite_db.list_foundry_sync_conflicts(
                status=str(arguments.get("status", "pending")).strip() or "pending",
                connector_id=str(arguments.get("connector_id", "")).strip() or None,
                source_filter=str(arguments.get("source_filter", "")).strip() or None,
                limit=to_int(arguments.get("limit"), 50, minimum=1, maximum=500),
            ),
            "ts": now_iso(),
        }
        payload["count"] = len(payload["items"])
        return emit(status_ok, payload)

    if name == "foundry_conflict_apply":
        if sqlite_db is None:
            initialize_core_only()
            sqlite_db = get_sqlite_db()
        if "id" not in arguments:
            return emit(status_failed, {"error": "id가 필요합니다."}, status_message="missing id")
        conflict_id = to_int(arguments.get("id"), 0, minimum=1)
        ok = sqlite_db.update_foundry_sync_conflict_status(
            conflict_id,
            status="approved",
            reviewer=str(arguments.get("reviewer", "mcp")).strip() or "mcp",
            resolution_note=str(arguments.get("note", "")).strip(),
        )
        payload = {
            "schema": "knowledge-hub.foundry.conflict.apply.result.v1",
            "runId": f"mcp_foundry_conflict_apply_{uuid4().hex[:10]}",
            "status": "ok" if ok else "error",
            "applied": bool(ok),
            "item": sqlite_db.get_foundry_sync_conflict(conflict_id),
            "ts": now_iso(),
        }
        return emit(status_ok if ok else status_failed, payload)

    if name == "foundry_conflict_reject":
        if sqlite_db is None:
            initialize_core_only()
            sqlite_db = get_sqlite_db()
        if "id" not in arguments:
            return emit(status_failed, {"error": "id가 필요합니다."}, status_message="missing id")
        conflict_id = to_int(arguments.get("id"), 0, minimum=1)
        ok = sqlite_db.update_foundry_sync_conflict_status(
            conflict_id,
            status="rejected",
            reviewer=str(arguments.get("reviewer", "mcp")).strip() or "mcp",
            resolution_note=str(arguments.get("note", "")).strip(),
        )
        payload = {
            "schema": "knowledge-hub.foundry.conflict.reject.result.v1",
            "runId": f"mcp_foundry_conflict_reject_{uuid4().hex[:10]}",
            "status": "ok" if ok else "error",
            "rejected": bool(ok),
            "item": sqlite_db.get_foundry_sync_conflict(conflict_id),
            "ts": now_iso(),
        }
        return emit(status_ok if ok else status_failed, payload)

    if name == "entity_merge_list":
        if sqlite_db is None:
            initialize_core_only()
            sqlite_db = get_sqlite_db()
        payload = {
            "schema": "knowledge-hub.entity.merge.list.result.v1",
            "runId": f"mcp_entity_merge_list_{uuid4().hex[:10]}",
            "status": "ok",
            "items": sqlite_db.list_entity_merge_proposals(
                topic_slug=str(arguments.get("topic", "")).strip() or None,
                status=str(arguments.get("status", "pending")).strip() or "pending",
                limit=to_int(arguments.get("limit"), 50, minimum=1, maximum=500),
            ),
            "ts": now_iso(),
        }
        payload["count"] = len(payload["items"])
        return emit(status_ok, payload)

    if name == "entity_merge_apply":
        if sqlite_db is None:
            initialize_core_only()
            sqlite_db = get_sqlite_db()
        if "id" not in arguments:
            return emit(status_failed, {"error": "id가 필요합니다."}, status_message="missing id")
        proposal_id = to_int(arguments.get("id"), 0, minimum=1)
        ok = sqlite_db.apply_entity_merge_proposal(proposal_id)
        payload = {
            "schema": "knowledge-hub.entity.merge.apply.result.v1",
            "runId": f"mcp_entity_merge_apply_{uuid4().hex[:10]}",
            "status": "ok" if ok else "error",
            "applied": bool(ok),
            "proposalId": proposal_id,
            "item": _safe_get_merge_item(proposal_id),
            "ts": now_iso(),
        }
        return emit(status_ok if ok else status_failed, payload)

    if name == "entity_merge_reject":
        if sqlite_db is None:
            initialize_core_only()
            sqlite_db = get_sqlite_db()
        if "id" not in arguments:
            return emit(status_failed, {"error": "id가 필요합니다."}, status_message="missing id")
        proposal_id = to_int(arguments.get("id"), 0, minimum=1)
        ok = sqlite_db.reject_entity_merge_proposal(proposal_id)
        payload = {
            "schema": "knowledge-hub.entity.merge.reject.result.v1",
            "runId": f"mcp_entity_merge_reject_{uuid4().hex[:10]}",
            "status": "ok" if ok else "error",
            "rejected": bool(ok),
            "proposalId": proposal_id,
            "item": _safe_get_merge_item(proposal_id),
            "ts": now_iso(),
        }
        return emit(status_ok if ok else status_failed, payload)

    return None
