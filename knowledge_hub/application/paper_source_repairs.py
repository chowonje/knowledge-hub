"""Application helpers for paper source cleanup and repair flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge_hub.document_memory import DocumentMemoryBuilder
from knowledge_hub.papers.card_v2_builder import PaperCardV2Builder
from knowledge_hub.papers.memory_runtime import build_paper_memory_builder
from knowledge_hub.papers.source_cleanup import (
    apply_source_cleanup_plan,
    build_source_cleanup_plan,
    write_source_cleanup_artifacts,
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _repair_item_brief(item: dict[str, Any]) -> str:
    paper_id = _clean_text(item.get("paperId"))
    repair_status = _clean_text(item.get("repairStatus"))
    action = _clean_text(item.get("action"))
    canonical = _clean_text(item.get("canonicalPaperId"))
    if canonical:
        return f"{paper_id or '-'} {repair_status or 'unknown'} {action or 'noop'} -> {canonical}"
    return f"{paper_id or '-'} {repair_status or 'unknown'} {action or 'noop'}"


def summarize_paper_source_repair_result(payload: dict[str, Any]) -> str:
    schema = _clean_text(payload.get("schema")) or "knowledge-hub.paper.source-repair.result.v1"
    status = _clean_text(payload.get("status")) or "unknown"
    counts = dict(payload.get("counts") or {})
    target_count = int(payload.get("targetCount") or 0)
    ok = int(counts.get("ok") or 0)
    blocked = int(counts.get("blocked") or 0)
    failed = int(counts.get("failed") or 0)
    missing = int(counts.get("missing") or 0)
    prefix = f"{schema} status={status} targets={target_count} ok={ok} blocked={blocked} failed={failed} missing={missing}"
    items = list(payload.get("items") or [])
    if len(items) == 1:
        return f"{prefix} {_repair_item_brief(dict(items[0] or {}))}".strip()
    return prefix


def summarize_paper_source_repair_queue_result(payload: dict[str, Any]) -> str:
    schema = _clean_text(payload.get("schema")) or "knowledge-hub.paper.source-repair.queue.result.v1"
    status = _clean_text(payload.get("status")) or "unknown"
    counts = dict(payload.get("counts") or {})
    target_count = int(payload.get("targetCount") or 0)
    created = int(counts.get("created") or 0)
    updated = int(counts.get("updated") or 0)
    reopened = int(counts.get("reopened") or 0)
    missing = int(counts.get("missing") or 0)
    prefix = (
        f"{schema} status={status} targets={target_count} created={created} "
        f"updated={updated} reopened={reopened} missing={missing}"
    )
    items = list(payload.get("items") or [])
    if len(items) == 1:
        item = dict(items[0] or {})
        paper_id = _clean_text(item.get("paperId"))
        operation = _clean_text(item.get("operation"))
        return f"{prefix} {paper_id or '-'} {operation or 'unknown'}".strip()
    return prefix


def build_source_cleanup_rows_for_papers(
    sqlite_db,
    *,
    paper_ids: list[str] | tuple[str, ...],
    default_parser: str = "raw",
) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    missing_ids: list[str] = []
    seen: set[str] = set()
    parser_token = _clean_text(default_parser).lower() or "raw"
    for raw in list(paper_ids or []):
        paper_id = _clean_text(raw)
        if not paper_id or paper_id.casefold() in seen:
            continue
        seen.add(paper_id.casefold())
        paper = sqlite_db.get_paper(paper_id)
        if not paper:
            missing_ids.append(paper_id)
            continue
        rows.append(
            {
                "paperId": paper_id,
                "title": _clean_text(paper.get("title")),
                "oldPdfPath": _clean_text(paper.get("pdf_path")),
                "oldTextPath": _clean_text(paper.get("text_path")),
                "recommendedParser": parser_token,
            }
        )
    return rows, missing_ids


def run_source_cleanup_queue(
    *,
    sqlite_db,
    queue_rows: list[dict[str, str]],
    artifact_dir: str | Path,
    pass_b_ids: list[str] | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    decisions = build_source_cleanup_plan(queue_rows, sqlite_db=sqlite_db)
    apply_summary = {"applied": 0, "skipped": len(decisions)}
    if apply:
        apply_summary = apply_source_cleanup_plan(sqlite_db=sqlite_db, decisions=decisions)
    artifact_paths = write_source_cleanup_artifacts(
        artifact_dir=artifact_dir,
        decisions=decisions,
        pass_b_ids=list(pass_b_ids or []),
    )
    return {
        "status": "ok",
        "cleanup": {
            "total": len(decisions),
            "applySummary": apply_summary,
        },
        "artifactPaths": artifact_paths,
    }


def repair_paper_sources(
    *,
    sqlite_db,
    config: Any,
    paper_ids: list[str] | tuple[str, ...],
    document_memory_parser: str = "raw",
    allow_external: bool | None = None,
    llm_mode: str = "auto",
    dry_run: bool = False,
    rebuild: bool = True,
) -> dict[str, Any]:
    rows, missing_ids = build_source_cleanup_rows_for_papers(
        sqlite_db,
        paper_ids=paper_ids,
        default_parser=document_memory_parser,
    )
    decisions = build_source_cleanup_plan(rows, sqlite_db=sqlite_db)
    counts = {"ok": 0, "blocked": 0, "failed": 0, "missing": len(missing_ids)}
    items: list[dict[str, Any]] = []

    for decision in decisions:
        paper_id = _clean_text(decision.get("paperId"))
        item = {
            "paperId": paper_id,
            "title": _clean_text(decision.get("title")),
            "action": _clean_text(decision.get("action")),
            "decisionStatus": _clean_text(decision.get("status")),
            "resolutionReason": _clean_text(decision.get("resolutionReason")),
            "canonicalPaperId": _clean_text(decision.get("canonicalPaperId")),
            "oldPdfPath": _clean_text(decision.get("oldPdfPath")),
            "oldTextPath": _clean_text(decision.get("oldTextPath")),
            "newPdfPath": _clean_text(decision.get("newPdfPath")),
            "newTextPath": _clean_text(decision.get("newTextPath")),
            "sourceApplied": False,
            "sourceChanged": False,
            "rebuildApplied": False,
            "artifactRefresh": {},
        }
        action = item["action"]
        decision_status = item["decisionStatus"]
        if action in {"exclude_until_manual_fix", "manual_review_required"} or (
            action == "relink_to_canonical" and decision_status != "resolved"
        ):
            item["repairStatus"] = "blocked"
            counts["blocked"] += 1
            items.append(item)
            continue

        if dry_run:
            item["repairStatus"] = "planned"
            items.append(item)
            continue

        try:
            if action == "relink_to_canonical":
                summary = apply_source_cleanup_plan(sqlite_db=sqlite_db, decisions=[decision])
                item["sourceApplied"] = bool(summary.get("applied"))
                item["sourceChanged"] = bool(summary.get("applied"))
            elif action == "keep_current_source":
                item["sourceApplied"] = True
                item["sourceChanged"] = False

            if rebuild:
                memory_builder = build_paper_memory_builder(
                    sqlite_db,
                    config=config,
                    allow_external=allow_external,
                    llm_mode=llm_mode,
                )
                memory_row = dict(memory_builder.build_and_store(paper_id=paper_id) or {})
                document_rows = list(
                    DocumentMemoryBuilder(sqlite_db, config=config).build_and_store_paper(
                        paper_id=paper_id,
                        paper_parser=document_memory_parser,
                    )
                    or []
                )
                card_row = dict(PaperCardV2Builder(sqlite_db).build_and_store(paper_id=paper_id) or {})
                item["rebuildApplied"] = True
                item["artifactRefresh"] = {
                    "paperMemory": {
                        "status": "ok",
                        "qualityFlag": _clean_text(memory_row.get("quality_flag") or memory_row.get("qualityFlag")),
                    },
                    "documentMemory": {
                        "status": "ok",
                        "count": len(document_rows),
                        "documentId": _clean_text((document_rows[0] if document_rows else {}).get("document_id")),
                    },
                    "paperCardV2": {
                        "status": "ok",
                        "qualityFlag": _clean_text(card_row.get("quality_flag") or card_row.get("qualityFlag")),
                    },
                }
            item["repairStatus"] = "ok"
            counts["ok"] += 1
        except Exception as error:  # pragma: no cover - defensive operator path
            item["repairStatus"] = "failed"
            item["error"] = str(error)
            counts["failed"] += 1
        items.append(item)

    for missing_id in missing_ids:
        items.append(
            {
                "paperId": missing_id,
                "title": "",
                "action": "missing",
                "decisionStatus": "missing",
                "resolutionReason": "paper not found in sqlite store",
                "canonicalPaperId": "",
                "oldPdfPath": "",
                "oldTextPath": "",
                "newPdfPath": "",
                "newTextPath": "",
                "sourceApplied": False,
                "sourceChanged": False,
                "rebuildApplied": False,
                "artifactRefresh": {},
                "repairStatus": "missing",
            }
        )

    status = "ok"
    if counts["failed"]:
        status = "failed"
    elif counts["blocked"] and not counts["ok"]:
        status = "blocked"
    payload = {
        "schema": "knowledge-hub.paper.source-repair.result.v1",
        "status": status,
        "dryRun": bool(dry_run),
        "rebuild": bool(rebuild),
        "documentMemoryParser": _clean_text(document_memory_parser) or "raw",
        "targetCount": len(items),
        "counts": counts,
        "items": items,
    }
    payload["summary"] = summarize_paper_source_repair_result(payload)
    return payload


def queue_paper_source_repairs(
    *,
    sqlite_db,
    paper_ids: list[str] | tuple[str, ...],
    document_memory_parser: str = "raw",
    rebuild: bool = True,
) -> dict[str, Any]:
    rows, missing_ids = build_source_cleanup_rows_for_papers(
        sqlite_db,
        paper_ids=paper_ids,
        default_parser=document_memory_parser,
    )
    items: list[dict[str, Any]] = []
    counts = {"created": 0, "updated": 0, "reopened": 0, "missing": len(missing_ids)}
    parser_token = _clean_text(document_memory_parser).lower() or "raw"
    for row in rows:
        paper_id = _clean_text(row.get("paperId"))
        title = _clean_text(row.get("title"))
        args = ["paper", "repair-source", "--paper-id", paper_id]
        if parser_token != "raw":
            args.extend(["--document-memory-parser", parser_token])
        if not rebuild:
            args.append("--no-rebuild")
        action = sqlite_db.upsert_ops_action(
            scope="paper",
            action_type="repair_paper_source",
            target_kind="paper",
            target_key=f"paper:{paper_id}",
            summary=f"repair paper source for {title or paper_id}",
            reason_codes=["paper_source_repair_requested"],
            command="khub",
            args=args,
            alerts=[],
            action={
                "paperId": paper_id,
                "title": title,
                "documentMemoryParser": parser_token,
                "rebuild": bool(rebuild),
            },
        )
        operation = str(action.get("operation") or "")
        if operation in counts:
            counts[operation] += 1
        items.append(
            {
                "paperId": paper_id,
                "title": title,
                "operation": operation,
                "action": action.get("item") or {},
            }
        )
    for paper_id in missing_ids:
        items.append(
            {
                "paperId": paper_id,
                "title": "",
                "operation": "missing",
                "action": {},
                "reason": "paper not found in sqlite store",
            }
        )
    payload = {
        "schema": "knowledge-hub.paper.source-repair.queue.result.v1",
        "status": "ok",
        "targetCount": len(items),
        "queuedCount": len(rows),
        "documentMemoryParser": parser_token,
        "rebuild": bool(rebuild),
        "counts": counts,
        "items": items,
    }
    payload["summary"] = summarize_paper_source_repair_queue_result(payload)
    return payload


__all__ = [
    "build_source_cleanup_rows_for_papers",
    "queue_paper_source_repairs",
    "repair_paper_sources",
    "run_source_cleanup_queue",
    "summarize_paper_source_repair_queue_result",
    "summarize_paper_source_repair_result",
]
