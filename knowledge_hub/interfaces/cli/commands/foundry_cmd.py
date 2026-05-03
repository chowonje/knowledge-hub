"""Foundry/operator CLI commands separated from the Agent Gateway surface."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import click
from click.core import ParameterSource
from rich.console import Console

from knowledge_hub.application.agent.foundry_bridge import (
    run_foundry_cli as _bridge_run_foundry_cli,
)
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase

console = Console()

SYNC_SOURCES = ("all", "note", "paper", "web", "expense", "sleep", "schedule", "behavior")
FEATURE_INTENTS = ("read", "analyze", "forecast", "summarize", "compare", "alert")
DEFAULT_DISCOVER_FEATURES = ("daily_coach", "focus_analytics", "risk_alert")
SYNC_CONNECTOR_ID_BY_SOURCE = {
    "all": "knowledge-hub",
    "note": "knowledge-hub-vault",
    "web": "knowledge-hub-web",
    "paper": "knowledge-hub-arxiv",
    "expense": "knowledge-hub-expense",
    "sleep": "knowledge-hub-sleep",
    "schedule": "knowledge-hub-schedule",
    "behavior": "knowledge-hub-behavior",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sqlite_db(*, khub=None, sqlite_path: str | None = None) -> SQLiteDatabase:
    if khub is not None and hasattr(khub, "sqlite_db"):
        return khub.sqlite_db()
    if sqlite_path:
        return SQLiteDatabase(sqlite_path)
    if khub is None:
        raise ValueError("khub or sqlite_path is required")
    return SQLiteDatabase(khub.config.sqlite_path)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_ts(value: str | None) -> str:
    parsed = _parse_ts(value)
    if parsed:
        return parsed.isoformat()
    return _now_iso()


def _note_source(source_type: str) -> str:
    normalized = str(source_type or "").strip().lower()
    if normalized == "web":
        return "web"
    if normalized in {"expense", "sleep", "schedule", "behavior"}:
        return normalized
    return "note"


def _classification_for_source(source: str) -> str:
    if source in {"paper", "expense", "sleep", "schedule", "behavior"}:
        return "P1"
    if source == "web":
        return "P2"
    return "P2"


def _connector_id_for_source(source_filter: str) -> str:
    return SYNC_CONNECTOR_ID_BY_SOURCE.get(str(source_filter).strip().lower(), "knowledge-hub")


def _source_bucket(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token in {"note", "paper", "web", "expense", "sleep", "schedule", "behavior"}:
        return token
    if "arxiv" in token or "paper" in token:
        return "paper"
    if "web" in token or "crawl" in token:
        return "web"
    if "expense" in token:
        return "expense"
    if "sleep" in token:
        return "sleep"
    if "schedule" in token:
        return "schedule"
    if "behavior" in token:
        return "behavior"
    return "note"


def _source_filter_matches(source_filter: str, *, source_hint: str = "", entity_type: str = "") -> bool:
    if source_filter == "all":
        return True
    bucket = _source_bucket(source_hint)
    if bucket == source_filter:
        return True

    etype = str(entity_type or "").strip().lower()
    if source_filter == "paper" and etype == "paper":
        return True
    if source_filter == "note" and etype in {"concept", "person", "organization", "event"}:
        return bucket not in {"web", "paper", "expense", "sleep", "schedule", "behavior"}
    return False


def _extract_tags(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("tags")
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return []


def _run_foundry_cli(command: str, command_args: Sequence[str], timeout_sec: int = 120) -> tuple[dict[str, Any] | None, str | None]:
    return _bridge_run_foundry_cli(command, command_args, timeout_sec=timeout_sec)


def _collect_records(db: SQLiteDatabase) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    notes = db.list_notes(limit=200000)
    for row in notes:
        raw_meta = row.get("metadata")
        meta: dict[str, Any] = {}
        if isinstance(raw_meta, str) and raw_meta.strip():
            try:
                parsed = json.loads(raw_meta)
                if isinstance(parsed, dict):
                    meta = parsed
            except Exception:
                meta = {}
        elif isinstance(raw_meta, dict):
            meta = raw_meta

        source = _note_source(str(row.get("source_type", "note")))
        records.append(
            {
                "id": str(row.get("id", "")),
                "source": source,
                "title": str(row.get("title", "")).strip(),
                "content": str(row.get("content", "") or ""),
                "filePath": str(row.get("file_path", "") or ""),
                "metadata": meta,
                "updatedAt": _format_ts(str(row.get("updated_at", "") or "")),
                "tags": _extract_tags(meta),
                "classification": _classification_for_source(source),
            }
        )

    papers = db.list_papers(limit=200000)
    for row in papers:
        arxiv_id = str(row.get("arxiv_id", "")).strip()
        records.append(
            {
                "id": arxiv_id,
                "source": "paper",
                "title": str(row.get("title", "")).strip(),
                "content": str(row.get("notes", "") or ""),
                "filePath": str(row.get("translated_path") or row.get("text_path") or row.get("pdf_path") or ""),
                "metadata": {
                    "authors": str(row.get("authors", "") or ""),
                    "year": row.get("year"),
                    "field": str(row.get("field", "") or ""),
                    "pdfPath": str(row.get("pdf_path", "") or ""),
                    "textPath": str(row.get("text_path", "") or ""),
                    "translatedPath": str(row.get("translated_path", "") or ""),
                },
                "updatedAt": _format_ts(str(row.get("created_at", "") or "")),
                "tags": [str(row.get("field", "") or "").strip()] if str(row.get("field", "") or "").strip() else [],
                "classification": "P1",
            }
        )

    return [row for row in records if row.get("id") and row.get("source")]


def _parse_sync_cursor(cursor: str) -> tuple[datetime | None, datetime | None]:
    raw = str(cursor or "").strip()
    if not raw:
        return None, None

    if raw.startswith("{") and raw.endswith("}"):
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            next_record_ts = _parse_ts(str(payload.get("next_record_ts", "") or payload.get("next", "")))
            next_event_ts = _parse_ts(str(payload.get("next_event_ts", "") or payload.get("next", "")))
            return next_record_ts, next_event_ts

    parsed = _parse_ts(raw)
    return parsed, parsed


def _relation_matches_source_filter(source_filter: str, relation: dict[str, Any]) -> bool:
    if source_filter == "all":
        return True
    source_type = str(relation.get("source_type", "")).strip().lower()
    target_type = str(relation.get("target_type", "")).strip().lower()
    source_id = str(relation.get("source_id", "")).strip().lower()
    target_id = str(relation.get("target_id", "")).strip().lower()

    if source_filter == "paper":
        return (
            source_type == "paper"
            or target_type == "paper"
            or source_id.startswith("paper_")
            or target_id.startswith("paper_")
            or source_id.startswith("paper:")
            or target_id.startswith("paper:")
        )
    if source_filter == "web":
        return source_type == "web" or target_type == "web" or "web" in source_id or "web" in target_id
    if source_filter in {"expense", "sleep", "schedule", "behavior"}:
        return source_type == source_filter or target_type == source_filter

    return not (
        source_type in {"web", "paper", "expense", "sleep", "schedule", "behavior"}
        or target_type in {"web", "paper", "expense", "sleep", "schedule", "behavior"}
    )


def _collect_ontology_delta(
    db: SQLiteDatabase,
    *,
    source_filter: str,
    state_cursor_ts: datetime | None,
    event_cursor_ts: datetime | None,
    limit: int,
) -> tuple[dict[str, list[dict[str, Any]]], bool, str, str]:
    state_limit = max(50, min(limit * 3, 5000))
    event_limit = max(50, min(limit * 5, 5000))

    state_after = state_cursor_ts.isoformat() if state_cursor_ts else None
    event_after = event_cursor_ts.isoformat() if event_cursor_ts else None

    entities_all = db.list_ontology_entities(limit=state_limit * 2)
    entities: list[dict[str, Any]] = []
    for row in entities_all:
        if state_cursor_ts:
            updated_at = _parse_ts(str(row.get("updated_at", "") or row.get("created_at", "")))
            if not updated_at or updated_at <= state_cursor_ts:
                continue
        if not _source_filter_matches(
            source_filter,
            source_hint=str(row.get("source", "")),
            entity_type=str(row.get("entity_type", "")),
        ):
            continue
        entities.append(row)
        if len(entities) >= state_limit:
            break

    relations_all = db.list_relations(limit=state_limit * 2, updated_after=state_after)
    relations = [row for row in relations_all if _relation_matches_source_filter(source_filter, row)][:state_limit]

    claims_all = db.list_ontology_claims(limit=state_limit * 2, updated_after=state_after)
    claims = [
        row
        for row in claims_all
        if _source_filter_matches(
            source_filter,
            source_hint=str(row.get("source", "")),
        )
    ][:state_limit]

    events_all = db.list_ontology_events(limit=event_limit * 2, updated_after=event_after)
    events = [
        row
        for row in events_all
        if _source_filter_matches(
            source_filter,
            source_hint=str(row.get("actor", "")),
            entity_type=str(row.get("entity_type", "")),
        )
    ][:event_limit]

    record_candidates = []
    for entity in entities:
        record_candidates.append(str(entity.get("updated_at", "") or entity.get("created_at", "")))
    for relation in relations:
        record_candidates.append(str(relation.get("created_at", "")))
    for claim in claims:
        record_candidates.append(str(claim.get("created_at", "")))
    next_record_ts = max([item for item in record_candidates if item], default=state_after or "")

    event_candidates = [str(event.get("created_at", "")) for event in events if str(event.get("created_at", ""))]
    next_event_ts = max(event_candidates, default=event_after or "")

    has_more = (
        len(entities) >= state_limit
        or len(relations) >= state_limit
        or len(claims) >= state_limit
        or len(events) >= event_limit
    )

    return {
        "entities": entities,
        "relations": relations,
        "claims": claims,
        "events": events,
    }, has_more, next_record_ts, next_event_ts


def _enqueue_sync_conflicts(
    db: SQLiteDatabase,
    *,
    connector_id: str,
    source_filter: str,
    items: list[dict[str, Any]],
    ontology_delta: dict[str, list[dict[str, Any]]],
) -> int:
    conflict_count = 0

    for item in items:
        source = str(item.get("source", "")).strip().lower()
        item_id = str(item.get("id", "")).strip()
        expected_classification = _classification_for_source(source)
        actual_classification = str(item.get("classification", "")).strip().upper()
        if actual_classification and actual_classification != expected_classification:
            db.add_foundry_sync_conflict(
                conflict_key=f"classification:item:{source}:{item_id}",
                conflict_type="classification_conflict",
                connector_id=connector_id,
                source_filter=source_filter,
                reason="classification level mismatch",
                payload={"classification": actual_classification, "item": item},
                existing_payload={"classification": expected_classification},
            )
            conflict_count += 1

        if source != "paper":
            continue

        paper_entity_id = f"paper:{item_id}"
        existing_entity = db.get_ontology_entity(paper_entity_id)
        if not existing_entity:
            continue

        payload_title = str(item.get("title", "")).strip()
        existing_title = str(existing_entity.get("canonical_name", "")).strip()
        if payload_title and existing_title and payload_title != existing_title:
            db.add_foundry_sync_conflict(
                conflict_key=f"entity:{paper_entity_id}",
                conflict_type="entity_conflict",
                connector_id=connector_id,
                source_filter=source_filter,
                reason="payload hash mismatch for entity id",
                payload=item,
                existing_payload=existing_entity,
            )
            conflict_count += 1

        payload_ts = _parse_ts(str(item.get("updatedAt", "")))
        existing_ts = _parse_ts(str(existing_entity.get("updated_at", "") or existing_entity.get("created_at", "")))
        if payload_ts and existing_ts and payload_ts < existing_ts:
            db.add_foundry_sync_conflict(
                conflict_key=f"time:{paper_entity_id}",
                conflict_type="time_conflict",
                connector_id=connector_id,
                source_filter=source_filter,
                reason="stale payload timestamp against latest entity state",
                payload={"updatedAt": payload_ts.isoformat(), "item": item},
                existing_payload={"updatedAt": existing_ts.isoformat(), "entity": existing_entity},
            )
            conflict_count += 1

    for entity in ontology_delta.get("entities", []):
        entity_id = str(entity.get("entity_id", "") or entity.get("id", "")).strip()
        if not entity_id:
            continue
        existing = db.get_ontology_entity(entity_id)
        if not existing:
            continue
        incoming_type = str(entity.get("entity_type", "") or entity.get("type", "")).strip().lower()
        existing_type = str(existing.get("entity_type", "")).strip().lower()
        if incoming_type and existing_type and incoming_type != existing_type:
            db.add_foundry_sync_conflict(
                conflict_key=f"entity_type:{entity_id}",
                conflict_type="entity_conflict",
                connector_id=connector_id,
                source_filter=source_filter,
                reason="entity type mismatch",
                payload=entity,
                existing_payload=existing,
            )
            conflict_count += 1

    return conflict_count


def _build_sync_payload(
    sqlite_path: str,
    source_filter: str,
    limit: int,
    cursor: str = "",
) -> dict[str, Any]:
    limit_safe = max(1, min(int(limit or 200), 2000))
    record_cursor_ts, event_cursor_ts = _parse_sync_cursor(cursor)
    connector_id = _connector_id_for_source(source_filter)

    db = _sqlite_db(sqlite_path=sqlite_path)
    try:
        records = _collect_records(db)
        if source_filter != "all":
            records = [row for row in records if row.get("source") == source_filter]

        if record_cursor_ts:
            records = [
                row
                for row in records
                if (_parse_ts(str(row.get("updatedAt", ""))) or datetime.min.replace(tzinfo=timezone.utc)) > record_cursor_ts
            ]

        records.sort(key=lambda row: (str(row.get("updatedAt", "")), str(row.get("id", ""))))
        has_more_records = len(records) > limit_safe
        items = records[:limit_safe]
        record_item_ts = str(items[-1].get("updatedAt", "")) if items else (record_cursor_ts.isoformat() if record_cursor_ts else "")

        ontology_delta, has_more_ontology, delta_record_ts, delta_event_ts = _collect_ontology_delta(
            db,
            source_filter=source_filter,
            state_cursor_ts=record_cursor_ts,
            event_cursor_ts=event_cursor_ts,
            limit=limit_safe,
        )
        conflict_count = _enqueue_sync_conflicts(
            db,
            connector_id=connector_id,
            source_filter=source_filter,
            items=items,
            ontology_delta=ontology_delta,
        )
    finally:
        db.close()

    next_record_ts = max([value for value in [record_item_ts, delta_record_ts] if value], default="")
    next_event_ts = max([value for value in [delta_event_ts, event_cursor_ts.isoformat() if event_cursor_ts else ""] if value], default="")

    return {
        "schema": "knowledge-hub.foundry.connector.sync.result.v2",
        "source": "knowledge-hub/cli.agent.sync",
        "runId": f"agent_sync_{uuid4().hex[:12]}",
        "connectorId": connector_id,
        "status": "ok",
        "ts": _now_iso(),
        "source_filter": source_filter,
        "items": items,
        "ontologyDelta": ontology_delta,
        "pendingConflicts": {
            "count": int(conflict_count),
        },
        "cursor": {
            "next_record_ts": next_record_ts,
            "next_event_ts": next_event_ts,
            "hasMore": bool(has_more_records or has_more_ontology),
        },
        "verify": {
            "allowed": True,
            "schemaValid": True,
            "policyAllowed": True,
            "schemaErrors": [],
        },
        "writeback": {
            "ok": False,
            "detail": "sync is read-only; no writeback stage",
        },
    }


def _default_discover_request() -> dict[str, Any]:
    resolution = {
        "source": "cli",
        "days": "cli",
        "from": "cli",
        "to": "cli",
        "topK": "cli",
        "limit": "cli",
        "intent": "cli",
        "features": "cli",
        "expenseThreshold": "cli",
        "minSleepHours": "cli",
        "eventLogPath": "cli",
        "stateFile": "cli",
        "saveState": "cli",
    }
    return {
        "source": "all",
        "days": 7,
        "from": None,
        "to": None,
        "topK": 8,
        "limit": None,
        "intent": "analyze",
        "features": list(DEFAULT_DISCOVER_FEATURES),
        "expenseThreshold": None,
        "minSleepHours": None,
        "eventLogPath": None,
        "stateFile": None,
        "saveState": True,
        "resumeSource": False,
        "resolution": resolution,
    }


def _load_discover_resume(path: str) -> dict[str, Any]:
    resume_path = Path(path).expanduser().resolve()
    if not resume_path.exists():
        raise click.BadParameter(f"resume file not found: {resume_path}", param_hint="--resume")
    try:
        payload = json.loads(resume_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise click.BadParameter(f"invalid resume json: {error}", param_hint="--resume") from error
    if not isinstance(payload, dict):
        raise click.BadParameter("resume file must contain JSON object", param_hint="--resume")
    return payload


def _is_cli_override(ctx: click.Context, param_name: str) -> bool:
    source = ctx.get_parameter_source(param_name)
    return source in {
        ParameterSource.COMMANDLINE,
        ParameterSource.ENVIRONMENT,
        ParameterSource.PROMPT,
    }


def _discover_available_features(state_file: str | None, event_log_path: str | None) -> list[str]:
    args = ["list", "--json"]
    if state_file:
        args.extend(["--state-file", state_file])
    if event_log_path:
        args.extend(["--event-log", event_log_path])
    payload, error = _run_foundry_cli("feature", args, timeout_sec=90)
    if error:
        return list(DEFAULT_DISCOVER_FEATURES)
    if isinstance(payload, dict):
        feature_names = payload.get("featureNames")
        if isinstance(feature_names, list):
            normalized = [str(item).strip() for item in feature_names if str(item).strip()]
            if normalized:
                return normalized
    return list(DEFAULT_DISCOVER_FEATURES)


def _expand_discover_features(selected: Sequence[str], available: Sequence[str]) -> list[str]:
    normalized = [str(item).strip() for item in selected if str(item).strip()]
    if not normalized:
        return list(available)
    lowered = {item.lower() for item in normalized}
    if "all" in lowered:
        return list(available)
    deduped: list[str] = []
    for item in normalized:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _build_discover_request(
    ctx: click.Context,
    *,
    source_filter: str,
    days: int,
    from_ts: str | None,
    to_ts: str | None,
    top_k: int,
    limit: int | None,
    intent: str,
    features: Sequence[str],
    expense_threshold: float | None,
    min_sleep_hours: float | None,
    event_log: str | None,
    state_file: str | None,
    save_state: bool,
    resume_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    request = _default_discover_request()
    if resume_payload and isinstance(resume_payload.get("request"), dict):
        resume_request = resume_payload["request"]
        request["resumeSource"] = True
        for field in [
            "source",
            "days",
            "from",
            "to",
            "topK",
            "limit",
            "intent",
            "features",
            "expenseThreshold",
            "minSleepHours",
            "eventLogPath",
            "stateFile",
            "saveState",
        ]:
            if field in resume_request:
                request[field] = resume_request[field]
                request["resolution"][field] = "resume"

    overrides = {
        "source_filter": ("source", source_filter),
        "days": ("days", max(0, int(days))),
        "from_ts": ("from", from_ts or None),
        "to_ts": ("to", to_ts or None),
        "top_k": ("topK", max(0, int(top_k))),
        "limit": ("limit", max(0, int(limit)) if limit is not None else None),
        "intent": ("intent", intent),
        "expense_threshold": ("expenseThreshold", expense_threshold if expense_threshold is not None else None),
        "min_sleep_hours": ("minSleepHours", min_sleep_hours if min_sleep_hours is not None else None),
        "event_log": ("eventLogPath", event_log or None),
        "state_file": ("stateFile", state_file or None),
        "save_state": ("saveState", bool(save_state)),
    }
    for param_name, (field_name, value) in overrides.items():
        if _is_cli_override(ctx, param_name):
            request[field_name] = value
            request["resolution"][field_name] = "cli"

    available_features = _discover_available_features(
        state_file=str(request.get("stateFile") or "") or None,
        event_log_path=str(request.get("eventLogPath") or "") or None,
    )
    if _is_cli_override(ctx, "features"):
        request["features"] = _expand_discover_features(features, available_features)
        request["resolution"]["features"] = "cli"
    elif not request.get("features"):
        request["features"] = list(available_features)
        request["resolution"]["features"] = "cli"
    else:
        request["features"] = _expand_discover_features(request.get("features", []), available_features)

    return request


@click.group("foundry")
def foundry_group():
    """Foundry/operator maintenance commands."""


@foundry_group.command("sync")
@click.option("--source", "source_filter", default="all", type=click.Choice(SYNC_SOURCES), show_default=True)
@click.option("--limit", default=200, type=int, show_default=True)
@click.option("--cursor", default="", help="ISO timestamp cursor (exclusive)")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.option("--foundry", is_flag=True, default=False, help="Foundry bridge mode (reserved)")
@click.option("--event-log", default="", help="Foundry event log path (reserved)")
@click.pass_context
def agent_sync(ctx, source_filter, limit, cursor, as_json, foundry, event_log):
    """Emit source records for Foundry connector sync."""
    del foundry, event_log
    khub = ctx.obj["khub"]
    payload = _build_sync_payload(
        sqlite_path=khub.config.sqlite_path,
        source_filter=source_filter,
        limit=limit,
        cursor=cursor,
    )
    strict_schema = bool(khub.config.get_nested("validation", "schema", "strict", default=False))
    schema_result = validate_payload(payload, "knowledge-hub.foundry.connector.sync.result.v2", strict=strict_schema)
    if schema_result.errors:
        payload["schemaErrors"] = schema_result.errors
        verify = payload.get("verify") if isinstance(payload.get("verify"), dict) else {}
        verify["schemaErrors"] = list(schema_result.errors)
        verify["schemaValid"] = False
        verify["allowed"] = False
        payload["verify"] = verify
    if strict_schema and not schema_result.ok:
        payload["status"] = "blocked"
        verify = payload.get("verify") if isinstance(payload.get("verify"), dict) else {}
        verify["allowed"] = False
        verify["schemaValid"] = False
        payload["verify"] = verify
        payload["writeback"] = {"ok": False, "detail": "verify blocked by schema validation"}
        if as_json:
            console.print_json(data=payload)
            ctx.exit(1)
        details = "; ".join(schema_result.errors[:3]) or "schema validation failed"
        raise click.ClickException(f"agent sync blocked: schema validation failed ({details})")

    if as_json:
        console.print_json(data=payload)
        return

    console.print(f"[green]runId[/green]: {payload['runId']}")
    console.print(f"[green]items[/green]: {len(payload['items'])}")
    ontology_delta = payload.get("ontologyDelta", {}) if isinstance(payload.get("ontologyDelta"), dict) else {}
    console.print(
        "[green]ontologyDelta[/green]: "
        f"entities={len(ontology_delta.get('entities', []))}, "
        f"relations={len(ontology_delta.get('relations', []))}, "
        f"claims={len(ontology_delta.get('claims', []))}, "
        f"events={len(ontology_delta.get('events', []))}"
    )
    console.print(f"[green]source_filter[/green]: {source_filter}")
    console.print(f"[green]next_record_ts[/green]: {payload['cursor']['next_record_ts'] or '-'}")
    console.print(f"[green]next_event_ts[/green]: {payload['cursor']['next_event_ts'] or '-'}")
    console.print(f"[green]hasMore[/green]: {payload['cursor']['hasMore']}")


@foundry_group.command("conflict-list")
@click.option("--status", default="pending", type=click.Choice(["pending", "approved", "rejected"]), show_default=True)
@click.option("--connector-id", default="", help="Filter by connector id")
@click.option("--source-filter", default="", type=click.Choice(SYNC_SOURCES), help="Filter by source")
@click.option("--limit", default=50, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def agent_foundry_conflict_list(ctx, status, connector_id, source_filter, limit, as_json):
    """List pending/apply/rejected dual-write conflicts."""
    khub = ctx.obj["khub"]
    db = _sqlite_db(khub=khub)
    try:
        rows = db.list_foundry_sync_conflicts(
            status=status,
            connector_id=connector_id.strip() or None,
            source_filter=source_filter.strip() or None,
            limit=max(1, int(limit)),
        )
    finally:
        db.close()

    payload = {
        "schema": "knowledge-hub.foundry.conflict.list.result.v1",
        "runId": f"agent_conflict_list_{uuid4().hex[:12]}",
        "status": "ok",
        "count": len(rows),
        "items": rows,
        "ts": _now_iso(),
    }
    if as_json:
        console.print_json(data=payload)
        return

    console.print(f"[green]count[/green]: {payload['count']}")
    for row in rows:
        console.print(
            f"- id={row.get('id')} status={row.get('status')} "
            f"type={row.get('conflict_type')} key={row.get('conflict_key')} "
            f"connector={row.get('connector_id')}"
        )


@foundry_group.command("conflict-apply")
@click.argument("conflict_id", type=int)
@click.option("--reviewer", default="cli-user", show_default=True)
@click.option("--note", default="", help="Optional resolution note")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def agent_foundry_conflict_apply(ctx, conflict_id, reviewer, note, as_json):
    """Approve a pending dual-write conflict."""
    khub = ctx.obj["khub"]
    db = _sqlite_db(khub=khub)
    try:
        ok = db.update_foundry_sync_conflict_status(
            conflict_id,
            status="approved",
            reviewer=reviewer,
            resolution_note=note,
        )
        item = db.get_foundry_sync_conflict(conflict_id)
    finally:
        db.close()

    payload = {
        "schema": "knowledge-hub.foundry.conflict.apply.result.v1",
        "runId": f"agent_conflict_apply_{uuid4().hex[:12]}",
        "status": "ok" if ok else "error",
        "applied": bool(ok),
        "item": item,
        "ts": _now_iso(),
    }
    if as_json:
        console.print_json(data=payload)
        return
    if not ok:
        raise click.ClickException(f"conflict id not found: {conflict_id}")
    console.print(f"[green]approved[/green]: {conflict_id}")


@foundry_group.command("conflict-reject")
@click.argument("conflict_id", type=int)
@click.option("--reviewer", default="cli-user", show_default=True)
@click.option("--note", default="", help="Optional rejection note")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def agent_foundry_conflict_reject(ctx, conflict_id, reviewer, note, as_json):
    """Reject a pending dual-write conflict."""
    khub = ctx.obj["khub"]
    db = _sqlite_db(khub=khub)
    try:
        ok = db.update_foundry_sync_conflict_status(
            conflict_id,
            status="rejected",
            reviewer=reviewer,
            resolution_note=note,
        )
        item = db.get_foundry_sync_conflict(conflict_id)
    finally:
        db.close()

    payload = {
        "schema": "knowledge-hub.foundry.conflict.reject.result.v1",
        "runId": f"agent_conflict_reject_{uuid4().hex[:12]}",
        "status": "ok" if ok else "error",
        "rejected": bool(ok),
        "item": item,
        "ts": _now_iso(),
    }
    if as_json:
        console.print_json(data=payload)
        return
    if not ok:
        raise click.ClickException(f"conflict id not found: {conflict_id}")
    console.print(f"[yellow]rejected[/yellow]: {conflict_id}")


@foundry_group.command("discover")
@click.option("--source", "source_filter", default="all", type=click.Choice(SYNC_SOURCES), show_default=True)
@click.option("--days", default=7, type=int, show_default=True)
@click.option("--from", "from_ts", default=None, help="Start timestamp (ISO8601)")
@click.option("--to", "to_ts", default=None, help="End timestamp (ISO8601)")
@click.option("--top-k", "top_k", default=8, type=int, show_default=True)
@click.option("--limit", default=None, type=int, help="Optional per-feature limit")
@click.option("--intent", default="analyze", type=click.Choice(FEATURE_INTENTS), show_default=True)
@click.option("--feature", "features", multiple=True, help="Feature name (repeatable). Use 'all' for all features.")
@click.option("--expense-threshold", default=None, type=float)
@click.option("--min-sleep-hours", default=None, type=float)
@click.option("--event-log", default=None, help="Foundry event log path")
@click.option("--state-file", default=None, help="Foundry sync state file path")
@click.option("--save-state/--no-save-state", default=True, show_default=True)
@click.option("--resume", default=None, help="Resume previous discover result JSON")
@click.option("--output", default=None, help="Write discover result JSON file")
@click.option("--fail-on-error/--no-fail-on-error", default=True, show_default=True)
@click.option("--fail-on-partial/--no-fail-on-partial", default=False, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def agent_discover(
    ctx,
    source_filter,
    days,
    from_ts,
    to_ts,
    top_k,
    limit,
    intent,
    features,
    expense_threshold,
    min_sleep_hours,
    event_log,
    state_file,
    save_state,
    resume,
    output,
    fail_on_error,
    fail_on_partial,
    as_json,
):
    """Run sync + feature diagnostics and emit discover envelope."""
    resume_payload = _load_discover_resume(resume) if resume else None
    request = _build_discover_request(
        ctx,
        source_filter=source_filter,
        days=days,
        from_ts=from_ts,
        to_ts=to_ts,
        top_k=top_k,
        limit=limit,
        intent=intent,
        features=features,
        expense_threshold=expense_threshold,
        min_sleep_hours=min_sleep_hours,
        event_log=event_log,
        state_file=state_file,
        save_state=save_state,
        resume_payload=resume_payload,
    )

    sync_limit = int(request.get("limit") or 200)
    sync_args = [
        "--source",
        str(request.get("source", "all")),
        "--limit",
        str(max(1, sync_limit)),
    ]
    if request.get("stateFile"):
        sync_args.extend(["--state-file", str(request.get("stateFile"))])
    if request.get("eventLogPath"):
        sync_args.extend(["--event-log", str(request.get("eventLogPath"))])
    if not bool(request.get("saveState", True)):
        sync_args.append("--no-save-state")

    sync_payload, sync_error = _run_foundry_cli("sync", sync_args, timeout_sec=180)
    if sync_payload is None:
        khub = ctx.obj["khub"]
        try:
            sync_payload = _build_sync_payload(
                sqlite_path=khub.config.sqlite_path,
                source_filter=str(request.get("source", "all")),
                limit=sync_limit,
                cursor="",
            )
            sync_payload["source"] = "knowledge-hub/cli.agent.sync.fallback"
            sync_error = sync_error or "foundry sync bridge unavailable; used local fallback"
        except Exception as error:
            sync_payload = sync_error or "sync failed"
            sync_error = f"sync failed: {error}"

    sync_ok = not sync_error
    if isinstance(sync_payload, dict):
        sync_status = str(sync_payload.get("status", "")).lower()
        if sync_status in {"failed", "error"}:
            sync_ok = False

    feature_results: list[dict[str, Any]] = []
    errors: list[str] = []
    if sync_error:
        errors.append(sync_error)

    for feature_name in [str(item) for item in request.get("features", [])]:
        feature_args = [feature_name, "--intent", str(request.get("intent", "analyze")), "--json"]
        feature_args.extend(["--source", str(request.get("source", "all"))])
        feature_args.extend(["--days", str(max(0, int(request.get("days", 7))))])
        feature_args.extend(["--top-k", str(max(0, int(request.get("topK", 8))))])

        if request.get("from"):
            feature_args.extend(["--from", str(request.get("from"))])
        if request.get("to"):
            feature_args.extend(["--to", str(request.get("to"))])
        if request.get("limit") is not None:
            feature_args.extend(["--limit", str(int(request.get("limit")))])
        if request.get("expenseThreshold") is not None:
            feature_args.extend(["--expense-threshold", str(float(request.get("expenseThreshold")))])
        if request.get("minSleepHours") is not None:
            feature_args.extend(["--min-sleep-hours", str(float(request.get("minSleepHours")))])
        if request.get("stateFile"):
            feature_args.extend(["--state-file", str(request.get("stateFile"))])
        if request.get("eventLogPath"):
            feature_args.extend(["--event-log", str(request.get("eventLogPath"))])

        result_payload, feature_error = _run_foundry_cli("feature", feature_args, timeout_sec=180)
        feature_ok = feature_error is None and result_payload is not None
        if isinstance(result_payload, dict):
            raw_status = str(result_payload.get("status", "")).lower()
            if raw_status in {"failed", "error"}:
                feature_ok = False
        if feature_error and feature_error not in errors:
            errors.append(f"{feature_name}: {feature_error}")

        feature_results.append(
            {
                "feature": feature_name,
                "ok": feature_ok,
                "result": result_payload if result_payload is not None else None,
                "error": feature_error if feature_error else None,
            }
        )

    feature_ok_count = sum(1 for item in feature_results if bool(item.get("ok")))
    if sync_ok and not errors and feature_ok_count == len(feature_results):
        status = "ok"
    elif sync_ok or feature_ok_count > 0:
        status = "partial"
    else:
        status = "error"

    payload = {
        "schema": "knowledge-hub.agent.discover.result.v1",
        "runId": f"agent_discover_{uuid4().hex[:12]}",
        "source": "knowledge-hub/cli.agent.discover",
        "status": status,
        "sync": sync_payload if sync_payload is not None else "sync unavailable",
        "features": feature_results,
        "request": request,
    }
    if errors:
        payload["errors"] = errors

    validate_result = validate_payload(payload, "knowledge-hub.agent.discover.result.v1", strict=True)
    if not validate_result.ok:
        payload["status"] = "error"
        existing = list(payload.get("errors", []))
        for item in validate_result.errors:
            if item not in existing:
                existing.append(item)
        payload["errors"] = existing

    if output:
        out_path = Path(output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if as_json:
        console.print_json(data=payload)
    else:
        console.print(f"[bold]runId:[/bold] {payload['runId']}")
        console.print(f"[bold]status:[/bold] {payload['status']}")
        console.print(f"[bold]source:[/bold] {request.get('source')}")
        console.print(f"[bold]features:[/bold] {len(feature_results)}")
        for item in feature_results:
            mark = "OK" if item.get("ok") else "FAIL"
            console.print(f"- {item.get('feature')}: {mark}")
        if payload.get("errors"):
            console.print("[red]errors:[/red]")
            for item in payload["errors"]:
                console.print(f"  - {item}")
        if output:
            console.print(f"[bold]output:[/bold] {str(Path(output).expanduser().resolve())}")

    if payload["status"] == "error" and fail_on_error:
        ctx.exit(1)
    if payload["status"] == "partial" and fail_on_partial:
        ctx.exit(1)


@foundry_group.command("discover-validate")
@click.option("--input", "input_path", required=True, help="Discover result JSON file path")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def agent_discover_validate(ctx, input_path, as_json):
    """Validate discover envelope against schema contract."""
    del ctx
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise click.BadParameter(f"input file not found: {path}", param_hint="--input")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise click.BadParameter(f"invalid JSON: {error}", param_hint="--input") from error
    if not isinstance(payload, dict):
        raise click.BadParameter("input JSON must be an object", param_hint="--input")

    result = validate_payload(payload, "knowledge-hub.agent.discover.result.v1", strict=True)
    output = {
        "schema": "knowledge-hub.agent.discover.validate.result.v1",
        "source": "knowledge-hub/cli.agent.discover-validate",
        "status": "ok" if result.ok else "error",
        "input": str(path),
        "valid": bool(result.ok),
        "schemaId": result.schema,
        "schemaFound": result.schema_found,
        "errors": result.errors,
        "ts": _now_iso(),
    }

    if as_json:
        console.print_json(data=output)
    else:
        console.print(f"[bold]input:[/bold] {output['input']}")
        console.print(f"[bold]status:[/bold] {output['status']}")
        if output["errors"]:
            console.print("[red]errors:[/red]")
            for item in output["errors"]:
                console.print(f"  - {item}")

    if not result.ok:
        raise click.ClickException("discover result schema validation failed")
