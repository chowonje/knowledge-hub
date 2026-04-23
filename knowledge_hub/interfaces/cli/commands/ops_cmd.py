"""Scheduled ops report command."""

from __future__ import annotations

import click
from rich.console import Console

from knowledge_hub.application.ops_actions import OpsActionExecutor, finalize_executed_action, queue_item_view, receipt_view
from knowledge_hub.application.ops_reports import OpsReportRunner
from knowledge_hub.core.schema_validator import annotate_schema_errors

console = Console()


def _validate_cli_payload(config, payload: dict, schema_id: str) -> None:
    strict = bool(config.get_nested("validation", "schema", "strict", default=False))
    result = annotate_schema_errors(payload, schema_id, strict=strict)
    if not result.ok:
        problems = ", ".join(result.errors[:5]) or "unknown schema validation error"
        raise click.ClickException(f"schema validation failed for {schema_id}: {problems}")


@click.command("ops-report-run")
@click.option("--run-id", default=None, help="대상 ko note run ID (기본: latest completed run)")
@click.option("--recent-runs", type=int, default=10, show_default=True)
@click.option("--days", type=int, default=7, show_default=True)
@click.option("--limit", type=int, default=100, show_default=True)
@click.option("--retention", type=int, default=30, show_default=True, help="보관할 snapshot 수")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def ops_report_run(ctx, run_id, recent_runs, days, limit, retention, as_json):
    """ko-note + RAG 운영 리포트를 결합 실행하고 snapshot/note를 생성"""
    khub = ctx.obj["khub"]
    runner = OpsReportRunner(khub.config, sqlite_db=khub.sqlite_db())
    payload = runner.run(
        run_id=str(run_id).strip() or None if run_id is not None else None,
        recent_runs=max(1, int(recent_runs)),
        rag_days=max(0, int(days)),
        rag_limit=max(1, int(limit)),
        retention=max(1, int(retention)),
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.ops.report.run.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    counts = payload.get("alertCounts") or {}
    queue = payload.get("actionQueue") or {}
    console.print(
        f"[bold]ops-report-run[/bold] status={payload.get('status')} "
        f"run={payload.get('koNoteRunId') or 'none'} "
        f"alerts={counts.get('total', 0)} warning={counts.get('warning', 0)} critical={counts.get('critical', 0)}"
    )
    paths = payload.get("artifactPaths") or {}
    if paths.get("opsReportJson"):
        console.print(f"snapshot: {paths.get('opsReportJson')}")
    if payload.get("notePath"):
        console.print(f"note: {payload.get('notePath')}")
    console.print(
        f"queue: created={queue.get('created', 0)} updated={queue.get('updated', 0)} reopened={queue.get('reopened', 0)} "
        f"pending={((queue.get('counts') or {}).get('pending', 0))}"
    )
    for alert in list(payload.get("alerts") or [])[:5]:
        console.print(f"[yellow]! {alert.get('severity')} {alert.get('code')}: {alert.get('summary')}[/yellow]")
    for action in list(payload.get("recommendedActions") or [])[:3]:
        command = " ".join([str(action.get("command") or ""), *[str(item) for item in (action.get("args") or [])]]).strip()
        console.print(f"[cyan]> {action.get('summary')}[/cyan]")
        if command:
            console.print(f"[dim]  {command}[/dim]")
    for item in list(queue.get("pendingActions") or [])[:3]:
        command = " ".join([str(item.get("command") or ""), *[str(arg) for arg in (item.get("args") or [])]]).strip()
        console.print(f"[magenta]# {item.get('actionId')} {item.get('summary')}[/magenta]")
        if command:
            console.print(f"[dim]  {command}[/dim]")
    for warning in list(payload.get("warnings") or [])[:10]:
        console.print(f"[yellow]- {warning}[/yellow]")


def _latest_receipt(sqlite_db, action_id: str) -> dict:
    getter = getattr(sqlite_db, "get_latest_ops_action_receipt", None)
    if not callable(getter):
        return {}
    return getter(str(action_id)) or {}


def _queue_item_view(sqlite_db, item: dict) -> dict:
    return queue_item_view(item, latest_receipt=_latest_receipt(sqlite_db, str(item.get("action_id") or "")))


def _receipt_payload_view(receipt: dict) -> dict:
    return receipt_view(receipt or {})


@click.command("ops-action-list")
@click.option("--status", "status_filter", type=click.Choice(["pending", "acked", "resolved", "all"]), default="pending", show_default=True)
@click.option(
    "--scope",
    "scope_filter",
    type=click.Choice(["ko_note", "rag", "paper", "agent", "all"]),
    default="all",
    show_default=True,
)
@click.option("--limit", type=int, default=50, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def ops_action_list(ctx, status_filter, scope_filter, limit, as_json):
    """ops action queue 조회"""
    khub = ctx.obj["khub"]
    sqlite_db = khub.sqlite_db()
    status_value = None if status_filter == "all" else str(status_filter)
    scope_value = None if scope_filter == "all" else str(scope_filter)
    items = [
        _queue_item_view(sqlite_db, item)
        for item in sqlite_db.list_ops_actions(status=status_value, scope=scope_value, limit=max(1, int(limit)))
    ]
    payload = {
        "schema": "knowledge-hub.ops.action.list.result.v1",
        "status": "ok",
        "count": len(items),
        "counts": sqlite_db.count_ops_actions(),
        "filters": {"status": status_filter, "scope": scope_filter},
        "items": items,
    }
    _validate_cli_payload(khub.config, payload, "knowledge-hub.ops.action.list.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]ops-action-list[/bold] count={payload.get('count')} status={status_filter} scope={scope_filter}")
    for item in items[:10]:
        command = " ".join([str(item.get("command") or ""), *[str(arg) for arg in (item.get("args") or [])]]).strip()
        execution = str(item.get("lastExecutionStatus") or "never-run")
        console.print(
            f"- {item.get('actionId')} [{item.get('status')}] {item.get('scope')} {item.get('summary')} "
            f"(last={execution})"
        )
        if item.get("lastResultSummary"):
            console.print(f"[dim]  {item.get('lastResultSummary')}[/dim]")
        if item.get("scope") == "paper":
            action_payload = dict(item.get("action") or {})
            paper_id = str(action_payload.get("paperId") or item.get("targetKey") or "").strip()
            parser = str(action_payload.get("documentMemoryParser") or "").strip()
            rebuild = action_payload.get("rebuild")
            detail_bits = [paper_id]
            if parser:
                detail_bits.append(f"parser={parser}")
            if isinstance(rebuild, bool):
                detail_bits.append(f"rebuild={'yes' if rebuild else 'no'}")
            if any(bit for bit in detail_bits):
                console.print(f"[dim]  {' '.join(bit for bit in detail_bits if bit)}[/dim]")
        if command:
            console.print(f"[dim]  {command}[/dim]")


def _set_action_status(ctx, *, action_id: str, status: str, actor: str, note: str, as_json: bool) -> None:
    khub = ctx.obj["khub"]
    sqlite_db = khub.sqlite_db()
    item = sqlite_db.set_ops_action_status(
        str(action_id).strip(),
        status=str(status),
        actor=str(actor).strip(),
        note=str(note or "").strip(),
    )
    payload = {
        "schema": "knowledge-hub.ops.action.result.v1",
        "status": "ok" if item else "failed",
        "actionId": str(action_id).strip(),
        "decision": str(status),
        "item": _queue_item_view(sqlite_db, item or {}),
    }
    _validate_cli_payload(khub.config, payload, "knowledge-hub.ops.action.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    if not item:
        raise click.ClickException(f"ops action not found: {action_id}")
    console.print(f"[bold]ops-action-{status}[/bold] action={action_id} status={payload.get('status')}")


@click.command("ops-action-ack")
@click.option("--action-id", required=True)
@click.option("--actor", default="cli-user", show_default=True)
@click.option("--note", default="")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def ops_action_ack(ctx, action_id, actor, note, as_json):
    """ops action을 acked로 전환"""
    _set_action_status(ctx, action_id=action_id, status="acked", actor=actor, note=note, as_json=as_json)


@click.command("ops-action-resolve")
@click.option("--action-id", required=True)
@click.option("--actor", default="cli-user", show_default=True)
@click.option("--note", default="")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def ops_action_resolve(ctx, action_id, actor, note, as_json):
    """ops action을 resolved로 전환"""
    _set_action_status(ctx, action_id=action_id, status="resolved", actor=actor, note=note, as_json=as_json)


@click.command("ops-action-execute")
@click.option("--action-id", required=True)
@click.option("--actor", default="cli-user", show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def ops_action_execute(ctx, action_id, actor, as_json):
    """safe ops action을 동기 실행하고 receipt를 기록"""
    khub = ctx.obj["khub"]
    sqlite_db = khub.sqlite_db()
    item = sqlite_db.get_ops_action(str(action_id).strip())
    if not item:
        raise click.ClickException(f"ops action not found: {action_id}")
    executor = OpsActionExecutor()
    command = str(item.get("command") or "")
    args = [str(arg) for arg in (item.get("args_json") or [])]
    try:
        result = executor.execute_sync(action_item=item, khub=khub)
    except Exception as error:
        receipt = sqlite_db.create_ops_action_receipt(
            action_id=str(action_id).strip(),
            mode="sync",
            status="failed",
            runner="cli",
            command=command,
            args=args,
            error_summary=str(error),
            actor=str(actor).strip(),
        )
        payload = {
            "schema": "knowledge-hub.ops.action.execute.result.v1",
            "status": "failed",
            "actionId": str(action_id).strip(),
            "executionMode": "sync",
            "item": _queue_item_view(sqlite_db, item),
            "receipt": _receipt_payload_view(receipt),
            "result": {"summary": "", "artifact": {}, "warnings": [str(error)]},
            "warnings": [str(error)],
        }
        _validate_cli_payload(khub.config, payload, "knowledge-hub.ops.action.execute.result.v1")
        if as_json:
            console.print_json(data=payload)
            return
        raise click.ClickException(str(error))
    item, receipt_artifact, warnings = finalize_executed_action(
        sqlite_db=sqlite_db,
        item=item,
        result=result,
        actor=str(actor).strip(),
    )
    result_summary = str(result.get("resultSummary") or "")
    verification = dict(receipt_artifact.get("verification") or {})
    verification_summary = str(verification.get("summary") or "").strip()
    if verification_summary:
        result_summary = f"{result_summary} | {verification_summary}".strip(" |")
    receipt = sqlite_db.create_ops_action_receipt(
        action_id=str(action_id).strip(),
        mode="sync",
        status="succeeded" if result.get("status") == "ok" else "failed",
        runner="cli",
        command=str(result.get("command") or command),
        args=[str(arg) for arg in (result.get("args") or args)],
        result_summary=result_summary,
        error_summary=str(result.get("errorSummary") or ""),
        artifact=receipt_artifact,
        actor=str(actor).strip(),
    )
    latest_item = sqlite_db.get_ops_action(str(action_id).strip()) or item
    payload = {
        "schema": "knowledge-hub.ops.action.execute.result.v1",
        "status": "ok" if result.get("status") == "ok" else "failed",
        "actionId": str(action_id).strip(),
        "executionMode": "sync",
        "item": _queue_item_view(sqlite_db, latest_item),
        "receipt": _receipt_payload_view(receipt),
        "result": {
            "summary": result_summary,
            "artifact": receipt_artifact,
            "warnings": warnings,
        },
        "warnings": warnings,
    }
    _validate_cli_payload(khub.config, payload, "knowledge-hub.ops.action.execute.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]ops-action-execute[/bold] action={action_id} status={payload.get('status')} "
        f"receipt={payload['receipt'].get('receiptId')}"
    )
    if payload["result"].get("summary"):
        console.print(f"- {payload['result'].get('summary')}")
    artifact = dict(payload["result"].get("artifact") or {})
    if str(payload["item"].get("scope") or "") == "paper" and artifact:
        counts = dict(artifact.get("counts") or {})
        if counts:
            console.print(
                f"[dim]  ok={int(counts.get('ok') or 0)} blocked={int(counts.get('blocked') or 0)} "
                f"failed={int(counts.get('failed') or 0)} missing={int(counts.get('missing') or 0)}[/dim]"
            )
        verification = dict(artifact.get("verification") or {})
        if verification.get("summary"):
            console.print(f"[dim]  {verification.get('summary')}[/dim]")
    for warning in payload.get("warnings", [])[:10]:
        console.print(f"[yellow]- {warning}[/yellow]")


@click.command("ops-action-receipts")
@click.option("--action-id", required=True)
@click.option("--limit", type=int, default=20, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def ops_action_receipts(ctx, action_id, limit, as_json):
    """ops action execution receipt 조회"""
    khub = ctx.obj["khub"]
    sqlite_db = khub.sqlite_db()
    lister = getattr(sqlite_db, "list_ops_action_receipts", None)
    if not callable(lister):
        raise click.ClickException("ops action receipt store unavailable")
    receipts = [_receipt_payload_view(item) for item in lister(action_id=str(action_id).strip(), limit=max(1, int(limit)))]
    payload = {
        "schema": "knowledge-hub.ops.action.receipts.result.v1",
        "status": "ok",
        "actionId": str(action_id).strip(),
        "count": len(receipts),
        "receipts": receipts,
    }
    _validate_cli_payload(khub.config, payload, "knowledge-hub.ops.action.receipts.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]ops-action-receipts[/bold] action={action_id} count={len(receipts)}")
    for receipt in receipts[:20]:
        console.print(
            f"- {receipt.get('receiptId')} [{receipt.get('status')}] "
            f"{receipt.get('runner')} {receipt.get('mode')} {receipt.get('executedAt')}"
        )
        if receipt.get("resultSummary"):
            console.print(f"[dim]  {receipt.get('resultSummary')}[/dim]")
        artifact = dict(receipt.get("artifact") or {})
        if artifact.get("schema") == "knowledge-hub.paper.source-repair.result.v1":
            counts = dict(artifact.get("counts") or {})
            console.print(
                f"[dim]  ok={int(counts.get('ok') or 0)} blocked={int(counts.get('blocked') or 0)} "
                f"failed={int(counts.get('failed') or 0)} missing={int(counts.get('missing') or 0)}[/dim]"
            )
            verification = dict(artifact.get("verification") or {})
            if verification.get("summary"):
                console.print(f"[dim]  {verification.get('summary')}[/dim]")
        if receipt.get("errorSummary"):
            console.print(f"[yellow]  {receipt.get('errorSummary')}[/yellow]")
