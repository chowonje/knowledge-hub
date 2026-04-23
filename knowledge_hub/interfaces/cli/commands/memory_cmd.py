"""Experimental document-memory CLI surface."""

from __future__ import annotations

import click
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console

from knowledge_hub.agent_memory import EpisodeMemoryService
from knowledge_hub.ai.retrieval_pipeline import _context_budget_config
from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.document_memory import DocumentMemoryBuilder, DocumentMemoryRetriever
from knowledge_hub.document_memory.payloads import semantic_units_payload
from knowledge_hub.knowledge.memory_router import build_memory_route
from knowledge_hub.papers.memory_builder import PaperMemoryBuilder
from knowledge_hub.papers.memory_projection import audit_payload as paper_projection_audit_payload
from knowledge_hub.papers.opendataloader_adapter import ODL_READING_ORDER_CHOICES, ODL_TABLE_METHOD_CHOICES

console = Console()


def _validate_cli_payload(config, payload: dict, schema_id: str) -> None:
    strict = bool(config.get_nested("validation", "schema", "strict", default=False))
    result = annotate_schema_errors(payload, schema_id, strict=strict)
    if not result.ok:
        problems = ", ".join(result.errors[:5]) or "unknown schema validation error"
        raise click.ClickException(f"schema validation failed for {schema_id}: {problems}")


def _builder(khub):
    return DocumentMemoryBuilder(khub.sqlite_db(), config=khub.config)


def _retriever(khub):
    return DocumentMemoryRetriever(khub.sqlite_db())


def _route_budget_hints(*, query: str, payload: dict) -> dict[str, object]:
    intent = str(payload.get("queryIntent") or "general")
    primary = str(((payload.get("route") or {}).get("primaryForm")) or "chunk")
    token_budget, compression_target, threshold = _context_budget_config(intent, top_k=4)
    return {
        "tokenBudget": int(token_budget),
        "memoryCompressionTarget": float(compression_target),
        "chunkExpansionThreshold": float(threshold),
        "preferredPrimaryForm": primary,
        "queryDigest": " ".join(str(query or "").strip().split())[:120],
    }


def _episode_service(store_path: str | Path) -> EpisodeMemoryService:
    return EpisodeMemoryService(store_path=store_path)


def _load_turns(turns_json: str, turns_file: str) -> list[dict]:
    payload = str(turns_json or "").strip()
    if turns_file:
        payload = Path(turns_file).read_text(encoding="utf-8")
    if not payload:
        raise click.ClickException("turns 입력이 필요합니다: --turns-json 또는 --turns-file")
    try:
        data = json.loads(payload)
    except Exception as error:
        raise click.ClickException(f"episode turns JSON 파싱 실패: {error}") from error
    if not isinstance(data, list):
        raise click.ClickException("turns는 JSON 배열이어야 합니다.")
    return [dict(item) for item in data if isinstance(item, dict)]


def _stale_memory_report(sqlite_db, *, limit: int) -> dict[str, object]:
    conn = getattr(sqlite_db, "conn", None)
    if conn is None:
        return {"status": "blocked", "count": 0, "items": [], "warnings": ["sqlite connection unavailable"]}

    items: list[dict[str, object]] = []
    rows = conn.execute(
        """
        SELECT document_id, title, document_title, updated_at, source_type, contextual_summary, document_date, observed_at, search_text
        FROM document_memory_units
        WHERE unit_type = 'document_summary'
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (max(10, int(limit) * 3),),
    ).fetchall()
    for row in rows:
        summary = dict(row)
        document_id = str(summary.get("document_id") or "").strip()
        reasons: list[str] = []
        score = 0.0
        search_text = str(summary.get("search_text") or "")
        contextual_summary = str(summary.get("contextual_summary") or "")
        if not contextual_summary or len(contextual_summary.strip()) < 80:
            reasons.append("summary_too_shallow")
            score += 0.45
        if "pending_summary" in search_text or "metadata" in str(summary.get("title") or "").lower():
            reasons.append("placeholder_heavy_summary")
            score += 0.35
        if str(summary.get("observed_at") or "").strip() and not str(summary.get("document_date") or "").strip():
            reasons.append("temporal_grounding_weak")
            score += 0.2
        if getattr(sqlite_db, "list_memory_relations", None):
            relations = sqlite_db.list_memory_relations(src_form="document_memory", src_id=document_id, relation_type="updates", limit=5)
            if relations:
                reasons.append("updates_relation_present")
                score += 0.2
        if not reasons:
            continue
        items.append(
            {
                "form": "document_memory",
                "id": document_id,
                "title": str(summary.get("document_title") or summary.get("title") or document_id),
                "rebuildReason": reasons[0],
                "reasons": reasons,
                "fallbackRate": round(min(0.95, 0.2 + score), 3),
                "tokenWasteEstimate": int((len(search_text) + len(contextual_summary)) / 6),
                "lastSeenQueryIntent": "unknown",
                "updatedAt": str(summary.get("updated_at") or ""),
            }
        )

    for row in getattr(sqlite_db, "list_paper_memory_cards", lambda limit=20: [])(limit=max(10, int(limit) * 2)):
        reasons = []
        score = 0.0
        if str(row.get("quality_flag") or "").strip().lower() not in {"", "ok"}:
            reasons.append("quality_flag_not_ok")
            score += 0.4
        core_fields = [str(row.get(name) or "").strip() for name in ("paper_core", "method_core", "evidence_core", "limitations")]
        if sum(1 for item in core_fields if item) < 2:
            reasons.append("core_fields_sparse")
            score += 0.4
        memory_id = str(row.get("memory_id") or "").strip()
        if getattr(sqlite_db, "list_memory_relations", None) and memory_id:
            relations = sqlite_db.list_memory_relations(src_form="paper_memory", src_id=memory_id, relation_type="updates", limit=5)
            if relations:
                reasons.append("updates_relation_present")
                score += 0.2
        if not reasons:
            continue
        items.append(
            {
                "form": "paper_memory",
                "id": str(row.get("paper_id") or ""),
                "title": str(row.get("title") or row.get("paper_id") or ""),
                "rebuildReason": reasons[0],
                "reasons": reasons,
                "fallbackRate": round(min(0.95, 0.25 + score), 3),
                "tokenWasteEstimate": int(sum(len(item) for item in core_fields) / 6),
                "lastSeenQueryIntent": "paper_lookup",
                "updatedAt": str(row.get("updated_at") or ""),
            }
        )

    items.sort(key=lambda item: (float(item.get("fallbackRate") or 0.0), int(item.get("tokenWasteEstimate") or 0)), reverse=True)
    return {"status": "ok", "count": min(len(items), max(1, int(limit))), "items": items[: max(1, int(limit))], "warnings": []}


def _paper_memory_builder(khub):
    return PaperMemoryBuilder(khub.sqlite_db())


def _paper_ids_for_backfill(sqlite_db, *, limit: int) -> list[str]:
    rows = list(sqlite_db.list_papers(limit=max(1, int(limit))) or [])
    items: list[str] = []
    seen: set[str] = set()
    for row in rows:
        token = str(row.get("arxiv_id") or "").strip()
        if not token or token.casefold() in seen:
            continue
        seen.add(token.casefold())
        items.append(token)
    return items


def _document_memory_exists(sqlite_db, *, paper_id: str) -> bool:
    return bool(sqlite_db.get_document_memory_summary(f"paper:{str(paper_id).strip()}"))


def _paper_projection_targets(sqlite_db, *, audit_all: bool, paper_id: str, limit: int) -> list[str]:
    token = str(paper_id or "").strip()
    if audit_all and token:
        raise click.ClickException("하나만 지정해야 합니다: --paper-id 또는 --all")
    if token:
        return [token]
    if not audit_all:
        raise click.ClickException("--paper-id 또는 --all 이 필요합니다.")
    return _paper_ids_for_backfill(sqlite_db, limit=limit)


def _runs_exports_dir() -> Path:
    path = Path("runs") / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_audit_artifacts(*, rows: list[dict[str, object]]) -> dict[str, str]:
    exports_dir = _runs_exports_dir()
    stem = f"paper_projection_audit_{_timestamp_slug()}"
    json_path = exports_dir / f"{stem}.json"
    csv_path = exports_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "paperId",
        "hasDocumentMemory",
        "hasCurrentPaperMemory",
        "currentVersion",
        "projectedVersion",
        "paperCoreChanged",
        "problemContextChanged",
        "methodCoreChanged",
        "evidenceCoreChanged",
        "limitationsChanged",
        "conceptLinkDeltaCount",
        "claimRefDeltaCount",
        "searchTextChanged",
        "coverageBefore",
        "coverageAfter",
        "recommendation",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            slot_diffs = dict(row.get("slotDiffs") or {})
            coverage_before = dict(row.get("coverageBefore") or {})
            coverage_after = dict(row.get("coverageAfter") or {})
            writer.writerow(
                {
                    "paperId": row.get("paperId"),
                    "hasDocumentMemory": row.get("hasDocumentMemory"),
                    "hasCurrentPaperMemory": row.get("hasCurrentPaperMemory"),
                    "currentVersion": row.get("currentVersion"),
                    "projectedVersion": row.get("projectedVersion"),
                    "paperCoreChanged": bool((slot_diffs.get("paper_core") or {}).get("changed")),
                    "problemContextChanged": bool((slot_diffs.get("problem_context") or {}).get("changed")),
                    "methodCoreChanged": bool((slot_diffs.get("method_core") or {}).get("changed")),
                    "evidenceCoreChanged": bool((slot_diffs.get("evidence_core") or {}).get("changed")),
                    "limitationsChanged": bool((slot_diffs.get("limitations") or {}).get("changed")),
                    "conceptLinkDeltaCount": int(row.get("conceptLinkDeltaCount") or 0),
                    "claimRefDeltaCount": int(row.get("claimRefDeltaCount") or 0),
                    "searchTextChanged": bool(row.get("searchTextChanged")),
                    "coverageBefore": coverage_before.get("filledCoreSlots"),
                    "coverageAfter": coverage_after.get("filledCoreSlots"),
                    "recommendation": row.get("recommendation"),
                }
            )
    return {"json": str(json_path), "csv": str(csv_path)}


def _audit_paper_projection(sqlite_db, *, paper_id: str) -> dict[str, object]:
    token = str(paper_id or "").strip()
    current = sqlite_db.get_paper_memory_card(token)
    projector = PaperMemoryBuilder(sqlite_db)
    projected = projector.build_projected_card(paper_id=token, ensure_document_memory=False) if _document_memory_exists(sqlite_db, paper_id=token) else None
    return paper_projection_audit_payload(
        paper_id=token,
        current=current,
        projected=projected,
        has_document_memory=bool(projected),
    )


def _health_paper_materialization(sqlite_db) -> dict[str, object]:
    builder = PaperMemoryBuilder(sqlite_db)
    paper_rows = list(sqlite_db.list_paper_memory_cards(limit=5000) or [])
    missing_document_memory: list[str] = []
    stale_paper_memory: list[str] = []
    adapter_path_papers: list[str] = []
    for row in paper_rows:
        paper_id = str(row.get("paper_id") or "").strip()
        if not paper_id:
            continue
        has_document_memory = _document_memory_exists(sqlite_db, paper_id=paper_id)
        if not has_document_memory:
            missing_document_memory.append(paper_id)
            adapter_path_papers.append(paper_id)
            continue
        if builder.needs_rebuild(paper_id):
            stale_paper_memory.append(paper_id)
    missing_document_memory = sorted(set(missing_document_memory))
    stale_paper_memory = sorted(set(stale_paper_memory))
    adapter_path_papers = sorted(set(adapter_path_papers))
    return {
        "status": "ok",
        "missingDocumentMemoryPaperIds": missing_document_memory,
        "stalePaperMemoryPaperIds": stale_paper_memory,
        "adapterPathPaperIds": adapter_path_papers,
        "counts": {
            "missingDocumentMemory": len(missing_document_memory),
            "stalePaperMemory": len(stale_paper_memory),
            "adapterPathUsage": len(adapter_path_papers),
        },
    }


@click.group("memory")
def memory_group():
    """document memory build/show/search"""


def _resolve_build_target(note_id: str, paper_id: str, canonical_url: str) -> tuple[str, str]:
    provided = [(name, value) for name, value in (("note", note_id), ("paper", paper_id), ("web", canonical_url)) if str(value or "").strip()]
    if len(provided) != 1:
        raise click.ClickException("하나의 대상만 지정해야 합니다: --note-id | --paper-id | --canonical-url")
    return provided[0][0], str(provided[0][1]).strip()


def _odl_cli_overrides(*, reading_order: str | None, use_struct_tree: bool | None, table_method: str | None) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if reading_order is not None:
        overrides["reading_order"] = str(reading_order).strip().lower()
    if use_struct_tree is not None:
        overrides["use_struct_tree"] = bool(use_struct_tree)
    if table_method is not None:
        overrides["table_method"] = str(table_method).strip().lower()
    return overrides


@memory_group.command("build")
@click.option("--note-id", default="", help="SQLite note id")
@click.option("--paper-id", default="", help="paper id / arXiv id")
@click.option("--canonical-url", default="", help="canonical web url")
@click.option(
    "--paper-parser",
    default="raw",
    type=click.Choice(["raw", "pymupdf", "mineru", "opendataloader"]),
    show_default=True,
    help="paper source parsing mode for labs build",
)
@click.option("--refresh-parse", is_flag=True, default=False, help="force re-parse parser artifacts for paper builds")
@click.option(
    "--odl-reading-order",
    type=click.Choice(list(ODL_READING_ORDER_CHOICES)),
    default=None,
    help="OpenDataLoader reading-order override (opendataloader only)",
)
@click.option(
    "--odl-use-struct-tree/--no-odl-use-struct-tree",
    default=None,
    help="OpenDataLoader struct-tree override (opendataloader only)",
)
@click.option(
    "--odl-table-method",
    type=click.Choice(list(ODL_TABLE_METHOD_CHOICES)),
    default=None,
    help="OpenDataLoader table-method override (opendataloader only)",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def build_memory(
    ctx,
    note_id,
    paper_id,
    canonical_url,
    paper_parser,
    refresh_parse,
    odl_reading_order,
    odl_use_struct_tree,
    odl_table_method,
    as_json,
):
    """문서 메모리 유닛 빌드"""
    khub = ctx.obj["khub"]
    mode, value = _resolve_build_target(note_id, paper_id, canonical_url)
    parser_used = str(paper_parser if mode == "paper" else "raw")
    parse_artifact_path = (
        str(Path(khub.config.papers_dir) / "parsed" / value)
        if mode == "paper" and parser_used in {"opendataloader", "mineru", "pymupdf"}
        else ""
    )
    try:
        if mode == "note":
            items = _builder(khub).build_and_store_note(note_id=value)
            document_id = value
        elif mode == "paper":
            items = _builder(khub).build_and_store_paper(
                paper_id=value,
                paper_parser=parser_used,
                refresh_parse=bool(refresh_parse),
                opendataloader_options=_odl_cli_overrides(
                    reading_order=odl_reading_order,
                    use_struct_tree=odl_use_struct_tree,
                    table_method=odl_table_method,
                ),
            )
            document_id = f"paper:{value}"
        else:
            items = _builder(khub).build_and_store_web(canonical_url=value)
            document_id = items[0]["document_id"] if items else ""
    except Exception as error:
        if mode == "paper" and parser_used in {"opendataloader", "mineru", "pymupdf"}:
            payload = {
                "schema": "knowledge-hub.document-memory.build.result.v1",
                "status": "blocked",
                "mode": mode,
                "documentId": f"paper:{value}",
                "document": {},
                "count": 0,
                "items": [],
                "warnings": [str(error)],
                "parserUsed": parser_used,
                "parseArtifactPath": parse_artifact_path,
                "structuredSectionsDetected": 0,
                "elementsImported": 0,
            }
            _validate_cli_payload(khub.config, payload, payload["schema"])
            if as_json:
                console.print_json(data=payload)
                return
        raise
    document = _retriever(khub).get_document(document_id) or {}
    units = list(document.get("units") or [])
    summary = dict(document.get("summary") or {})
    elements_imported = int((summary.get("provenance") or {}).get("elements_imported") or 0)
    structured_sections_detected = sum(
        1
        for item in units
        if list((item.get("provenance") or {}).get("heading_path") or [])
    )
    payload = {
        "schema": "knowledge-hub.document-memory.build.result.v1",
        "status": "ok",
        "mode": mode,
        "documentId": document_id,
        "document": document,
        "semanticUnits": semantic_units_payload(document),
        "count": len(items),
        "items": units,
        "parserUsed": parser_used,
        "parseArtifactPath": parse_artifact_path,
        "structuredSectionsDetected": structured_sections_detected,
        "elementsImported": elements_imported,
        "warnings": [],
    }
    _validate_cli_payload(khub.config, payload, payload["schema"])
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]document-memory build[/bold] mode={mode} document={document_id} units={len(items)}")


@memory_group.command("show")
@click.option("--document-id", required=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def show_memory(ctx, document_id, as_json):
    """문서 메모리 유닛 조회"""
    khub = ctx.obj["khub"]
    document = _retriever(khub).get_document(str(document_id).strip())
    payload = {
        "schema": "knowledge-hub.document-memory.card.result.v1",
        "status": "ok" if document else "failed",
        "document": document or {},
        "semanticUnits": semantic_units_payload(document),
        "warnings": [] if document else [f"document memory not found: {document_id}"],
    }
    _validate_cli_payload(khub.config, payload, payload["schema"])
    if as_json:
        console.print_json(data=payload)
        return
    if not document:
        raise click.ClickException(f"document memory not found: {document_id}")
    console.print(f"[bold]{document.get('documentTitle')}[/bold]")
    summary = (document.get("summary") or {}).get("contextualSummary") or ""
    if summary:
        console.print(summary)
    for item in (document.get("units") or [])[:8]:
        if item.get("unitType") == "document_summary":
            continue
        console.print(f"- {item.get('title')} :: {item.get('contextualSummary')}")


@memory_group.command("search")
@click.option("--query", required=True)
@click.option("--limit", type=int, default=10, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def search_memory(ctx, query, limit, as_json):
    """summary-first document memory 검색"""
    khub = ctx.obj["khub"]
    items = _retriever(khub).search(str(query).strip(), limit=max(1, int(limit)))
    payload = {
        "schema": "knowledge-hub.document-memory.search.result.v1",
        "status": "ok",
        "query": str(query).strip(),
        "count": len(items),
        "items": items,
        "semanticUnits": [
            semantic_units_payload(
                {
                    "documentId": item.get("documentId"),
                    "documentTitle": item.get("documentTitle"),
                    "sourceType": item.get("sourceType"),
                    "summary": item.get("documentSummary") or {},
                    "units": [
                        item.get("matchedUnit") or {},
                        *list((item.get("matchedSegment") or {}).get("units") or []),
                    ],
                }
            )
            for item in items
        ],
        "warnings": [],
    }
    _validate_cli_payload(khub.config, payload, payload["schema"])
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]document-memory search[/bold] query={query} count={len(items)}")
    for item in items:
        console.print(f"- {item.get('documentId')} {item.get('documentTitle')}")
        matched = item.get("matchedUnit") or {}
        if matched.get("contextualSummary"):
            console.print(f"[dim]  {matched.get('contextualSummary')}[/dim]")


@memory_group.command("audit-paper-projection")
@click.option("--paper-id", default="", help="paper id / arXiv id")
@click.option("--all", "audit_all", is_flag=True, help="audit a bounded batch of paper projections")
@click.option("--limit", type=int, default=20, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def audit_paper_projection(ctx, paper_id, audit_all, limit, as_json):
    """read-only paper-memory projection audit"""
    khub = ctx.obj["khub"]
    sqlite_db = khub.sqlite_db()
    targets = _paper_projection_targets(sqlite_db, audit_all=bool(audit_all), paper_id=str(paper_id).strip(), limit=max(1, int(limit)))
    rows = [_audit_paper_projection(sqlite_db, paper_id=token) for token in targets]
    payload: dict[str, object] = {
        "status": "ok",
        "mode": "batch" if audit_all else "single",
        "count": len(rows),
        "items": rows,
    }
    if audit_all:
        payload["artifacts"] = _write_audit_artifacts(rows=rows)
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]paper projection audit[/bold] mode={payload['mode']} count={payload['count']}")
    if payload.get("artifacts"):
        artifacts = dict(payload.get("artifacts") or {})
        console.print(f"[dim]json={artifacts.get('json')} csv={artifacts.get('csv')}[/dim]")
    for item in rows[:10]:
        console.print(
            f"- {item.get('paperId')} :: {item.get('recommendation')} "
            f"coverage={dict(item.get('coverageAfter') or {}).get('filledCoreSlots', 0)}"
        )


@memory_group.command("backfill-paper-document")
@click.option("--paper-id", default="", help="paper id / arXiv id")
@click.option("--all", "build_all", is_flag=True, help="backfill a bounded batch of papers")
@click.option("--limit", type=int, default=100, show_default=True)
@click.option("--force", is_flag=True, default=False, help="rebuild even when document memory already exists")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def backfill_paper_document(ctx, paper_id, build_all, limit, force, as_json):
    """materialize paper document-memory only when missing or forced"""
    khub = ctx.obj["khub"]
    sqlite_db = khub.sqlite_db()
    targets = _paper_projection_targets(sqlite_db, audit_all=bool(build_all), paper_id=str(paper_id).strip(), limit=max(1, int(limit)))
    builder = DocumentMemoryBuilder(sqlite_db, config=khub.config)
    built: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for token in targets:
        if not force and _document_memory_exists(sqlite_db, paper_id=token):
            skipped.append({"paperId": token, "reason": "already_materialized"})
            continue
        items = builder.build_and_store_paper(paper_id=token)
        built.append({"paperId": token, "documentId": f"paper:{token}", "unitCount": len(items)})
    payload = {
        "status": "ok",
        "mode": "batch" if build_all else "single",
        "builtCount": len(built),
        "skippedCount": len(skipped),
        "built": built,
        "skipped": skipped,
        "forced": bool(force),
    }
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]paper document backfill[/bold] built={payload['builtCount']} skipped={payload['skippedCount']} forced={payload['forced']}"
    )


@memory_group.command("health-paper-materialization")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def health_paper_materialization(ctx, as_json):
    """report paper/document-memory cutover readiness"""
    khub = ctx.obj["khub"]
    payload = _health_paper_materialization(khub.sqlite_db())
    if as_json:
        console.print_json(data=payload)
        return
    counts = dict(payload.get("counts") or {})
    console.print(
        "[bold]paper materialization health[/bold] "
        f"missing={counts.get('missingDocumentMemory', 0)} "
        f"stale={counts.get('stalePaperMemory', 0)} "
        f"adapter={counts.get('adapterPathUsage', 0)}"
    )


@memory_group.command("route")
@click.option("--query", required=True)
@click.option(
    "--source-type",
    default="all",
    type=click.Choice(["all", "paper", "vault", "web", "concept"]),
    show_default=True,
    help="bounded source hint for memory-form routing",
)
@click.option("--explain-budget", is_flag=True, default=False, help="include context-budget hints for labs")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def route_memory(ctx, query, source_type, explain_budget, as_json):
    """질문 유형에 맞는 preferred memory form을 inspectable하게 제안"""
    khub = ctx.obj["khub"]
    payload = build_memory_route(query=str(query).strip(), source_type=str(source_type).strip())
    _validate_cli_payload(khub.config, payload, payload["schema"])
    if explain_budget:
        payload["budgetHints"] = _route_budget_hints(query=str(query).strip(), payload=payload)
    if as_json:
        console.print_json(data=payload)
        return
    route = dict(payload.get("route") or {})
    console.print(
        "[bold]memory-route[/bold] "
        f"intent={payload.get('queryIntent')} "
        f"recommended={payload.get('recommendedDecisionOrder')} "
        f"primary={route.get('primaryForm')} "
        f"verifier={route.get('verifierForm')}"
    )
    for item in route.get("preferredForms") or []:
        console.print(
            f"- {item.get('priority')}. {item.get('name')} :: "
            f"{', '.join(str(reason) for reason in (item.get('why') or [])[:2])}"
        )
    if explain_budget:
        budget = dict(payload.get("budgetHints") or {})
        console.print(
            f"[dim]budget token={budget.get('tokenBudget')} "
            f"compression={budget.get('memoryCompressionTarget')} "
            f"threshold={budget.get('chunkExpansionThreshold')}[/dim]"
        )


@memory_group.command("stale-report")
@click.option("--limit", type=int, default=10, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def stale_report(ctx, limit, as_json):
    """rebuild candidate memory report for labs"""
    khub = ctx.obj["khub"]
    payload = {
        "schema": "knowledge-hub.memory.stale-report.result.v1",
        **_stale_memory_report(khub.sqlite_db(), limit=max(1, int(limit))),
    }
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]memory stale-report[/bold] count={payload.get('count')}")
    for item in payload.get("items") or []:
        console.print(
            f"- {item.get('form')} {item.get('id')} :: "
            f"{item.get('rebuildReason')} fallback={item.get('fallbackRate')}"
        )


@memory_group.group("episode")
def episode_group():
    """experimental agent/session episode memory"""


@episode_group.command("build")
@click.option("--session-id", required=True)
@click.option("--turns-json", default="", help="JSON array of turns")
@click.option("--turns-file", default="", help="path to turns JSON file")
@click.option("--store-path", default=".tmp/episode_memory.json", show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
def build_episode(session_id, turns_json, turns_file, store_path, as_json):
    turns = _load_turns(turns_json, turns_file)
    payload = _episode_service(store_path).build_episode(str(session_id).strip(), turns)
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]episode build[/bold] session={session_id} episode={payload.get('episodeId')}")


@episode_group.command("search")
@click.option("--session-id", required=True)
@click.option("--query", required=True)
@click.option("--depth", type=click.Choice(["theme", "semantic", "episode"]), default="theme", show_default=True)
@click.option("--store-path", default=".tmp/episode_memory.json", show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
def search_episode(session_id, query, depth, store_path, as_json):
    service = _episode_service(store_path)
    payload = {
        "status": "ok",
        "query": str(query).strip(),
        "sessionId": str(session_id).strip(),
        "depth": str(depth).strip(),
        "route": service.explain_episode_route(str(query).strip(), str(session_id).strip()),
        "items": service.search_episode_memory(str(query).strip(), str(session_id).strip(), depth=str(depth).strip()),
    }
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]episode search[/bold] depth={depth} count={len(payload['items'])}")
    for item in payload["items"]:
        console.print(f"- {item.get('episodeId')} :: {item.get('score')} :: {item.get('summary')}")
