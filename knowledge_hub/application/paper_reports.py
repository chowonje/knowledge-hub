from __future__ import annotations

from typing import Any

from knowledge_hub.application.ops_alerts import evaluate_paper_report_alerts
from knowledge_hub.application.paper_source_repairs import build_source_cleanup_rows_for_papers
from knowledge_hub.papers.source_cleanup import (
    EXCLUDE_UNTIL_MANUAL_FIX,
    RELINK_TO_CANONICAL,
    REVIEWED_KEEP_CURRENT_SOURCE,
    build_source_cleanup_plan,
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _known_cleanup_paper_ids() -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for mapping in (RELINK_TO_CANONICAL, EXCLUDE_UNTIL_MANUAL_FIX, REVIEWED_KEEP_CURRENT_SOURCE):
        for raw in mapping.keys():
            paper_id = _clean_text(raw)
            if not paper_id or paper_id.casefold() in seen:
                continue
            seen.add(paper_id.casefold())
            ordered.append(paper_id)
    return ordered


def _decision_to_report_item(decision: dict[str, Any]) -> dict[str, Any]:
    paper_id = _clean_text(decision.get("paperId"))
    action = _clean_text(decision.get("action"))
    decision_status = _clean_text(decision.get("status"))
    current_pdf = _clean_text(decision.get("oldPdfPath"))
    current_text = _clean_text(decision.get("oldTextPath"))
    target_pdf = _clean_text(decision.get("newPdfPath"))
    target_text = _clean_text(decision.get("newTextPath"))
    repair_eligible = (
        action == "relink_to_canonical"
        and decision_status == "resolved"
        and (current_pdf != target_pdf or current_text != target_text)
    )
    already_aligned = (
        action == "relink_to_canonical"
        and decision_status == "resolved"
        and not repair_eligible
    )
    manual_fix_required = action in {"exclude_until_manual_fix", "manual_review_required"}
    canonical_missing = action == "relink_to_canonical" and decision_status != "resolved"
    keep_current = action == "keep_current_source"
    verification_status = "unknown"
    if repair_eligible:
        verification_status = "repair_pending"
    elif manual_fix_required:
        verification_status = "manual_fix_required"
    elif canonical_missing:
        verification_status = "canonical_missing"
    elif already_aligned:
        verification_status = "already_aligned"
    elif keep_current:
        verification_status = "keep_current"
    return {
        "paperId": paper_id,
        "title": _clean_text(decision.get("title")),
        "action": action,
        "decisionStatus": decision_status,
        "status": decision_status,
        "resolutionReason": _clean_text(decision.get("resolutionReason")),
        "canonicalPaperId": _clean_text(decision.get("canonicalPaperId")),
        "canonicalTitle": _clean_text(decision.get("canonicalTitle")),
        "currentPdfPath": current_pdf,
        "currentTextPath": current_text,
        "targetPdfPath": target_pdf,
        "targetTextPath": target_text,
        "repairEligible": repair_eligible,
        "needsRepair": repair_eligible,
        "manualFixRequired": manual_fix_required,
        "canonicalMissing": canonical_missing,
        "keepCurrent": keep_current,
        "alreadyAligned": already_aligned,
        "verificationStatus": verification_status,
    }


def summarize_paper_source_verification(payload: dict[str, Any]) -> str:
    paper_id = _clean_text(payload.get("paperId"))
    verification_status = _clean_text(payload.get("verificationStatus")) or "unknown"
    decision_reason = _clean_text(payload.get("resolutionReason"))
    if decision_reason:
        return f"paper={paper_id or '-'} verify={verification_status} reason={decision_reason}"
    return f"paper={paper_id or '-'} verify={verification_status}"


def verify_paper_source_state(
    sqlite_db,
    *,
    paper_id: str,
    document_memory_parser: str = "raw",
) -> dict[str, Any]:
    normalized_paper_id = _clean_text(paper_id)
    rows, missing_ids = build_source_cleanup_rows_for_papers(
        sqlite_db,
        paper_ids=[normalized_paper_id],
        default_parser=document_memory_parser,
    )
    if missing_ids:
        payload = {
            "schema": "knowledge-hub.paper.source-repair.verify.result.v1",
            "status": "missing",
            "paperId": normalized_paper_id,
            "documentMemoryParser": _clean_text(document_memory_parser) or "raw",
            "resolved": False,
            "verificationStatus": "missing",
            "resolutionReason": "paper not found in sqlite store",
            "item": {},
        }
        payload["summary"] = summarize_paper_source_verification(payload)
        return payload
    decisions = build_source_cleanup_plan(rows, sqlite_db=sqlite_db)
    decision = dict((decisions or [{}])[0] or {})
    item = _decision_to_report_item(decision)
    resolved = bool(item.get("alreadyAligned") or item.get("keepCurrent"))
    payload = {
        "schema": "knowledge-hub.paper.source-repair.verify.result.v1",
        "status": "ok",
        "paperId": normalized_paper_id,
        "documentMemoryParser": _clean_text(document_memory_parser) or "raw",
        "resolved": resolved,
        "verificationStatus": _clean_text(item.get("verificationStatus")),
        "resolutionReason": _clean_text(item.get("resolutionReason")),
        "item": item,
    }
    payload["summary"] = summarize_paper_source_verification(payload)
    return payload


def build_paper_ops_report(
    sqlite_db,
    *,
    document_memory_parser: str = "raw",
    rebuild: bool = True,
    max_items: int = 20,
    limit: int | None = None,
) -> dict[str, Any]:
    if not callable(getattr(sqlite_db, "get_paper", None)):
        return {
            "schema": "knowledge-hub.paper.ops.report.result.v1",
            "status": "ok",
            "counts": {
                "knownRuleCount": len(_known_cleanup_paper_ids()),
                "presentInStore": 0,
                "tracked": 0,
                "repairEligible": 0,
                "alreadyAligned": 0,
                "manualFixRequired": 0,
                "canonicalMissing": 0,
                "keepCurrent": 0,
                "keepCurrentReviewed": 0,
                "missingKnownIds": 0,
                "missingFromStore": 0,
                "repairablePending": 0,
                "blockedManual": 0,
                "blockedMissingCanonical": 0,
            },
            "items": [],
            "alerts": [],
            "recommendedActions": [],
            "warnings": [],
        }

    known_ids = _known_cleanup_paper_ids()
    rows, missing_ids = build_source_cleanup_rows_for_papers(
        sqlite_db,
        paper_ids=known_ids,
        default_parser=document_memory_parser,
    )
    decisions = build_source_cleanup_plan(rows, sqlite_db=sqlite_db)
    items: list[dict[str, Any]] = []
    counts = {
        "knownRuleCount": len(known_ids),
        "tracked": len(rows),
        "presentInStore": len(rows),
        "repairEligible": 0,
        "alreadyAligned": 0,
        "manualFixRequired": 0,
        "canonicalMissing": 0,
        "keepCurrent": 0,
        "keepCurrentReviewed": 0,
        "missingKnownIds": len(missing_ids),
        "missingFromStore": len(missing_ids),
        "repairablePending": 0,
        "blockedManual": 0,
        "blockedMissingCanonical": 0,
    }

    for decision in decisions:
        item = _decision_to_report_item(decision)
        repair_eligible = bool(item.get("repairEligible"))
        already_aligned = bool(item.get("alreadyAligned"))
        manual_fix_required = bool(item.get("manualFixRequired"))
        canonical_missing = bool(item.get("canonicalMissing"))
        keep_current = bool(item.get("keepCurrent"))
        counts["repairEligible"] += int(repair_eligible)
        counts["alreadyAligned"] += int(already_aligned)
        counts["manualFixRequired"] += int(manual_fix_required)
        counts["canonicalMissing"] += int(canonical_missing)
        counts["keepCurrent"] += int(keep_current)
        counts["keepCurrentReviewed"] += int(keep_current)
        counts["repairablePending"] += int(repair_eligible)
        counts["blockedManual"] += int(manual_fix_required)
        counts["blockedMissingCanonical"] += int(canonical_missing)
        items.append(item)

    items.sort(
        key=lambda item: (
            0 if bool(item.get("repairEligible")) else 1,
            0 if bool(item.get("manualFixRequired")) else 1,
            str(item.get("paperId") or ""),
        )
    )
    alerts, recommended_actions = evaluate_paper_report_alerts(
        counts=counts,
        items=items,
        document_memory_parser=document_memory_parser,
        rebuild=rebuild,
    )
    return {
        "schema": "knowledge-hub.paper.ops.report.result.v1",
        "status": "ok",
        "counts": counts,
        "items": items[: max(1, int(limit if limit is not None else max_items))],
        "alerts": alerts,
        "recommendedActions": recommended_actions,
        "warnings": [],
    }


def build_paper_source_ops_report(
    sqlite_db,
    *,
    document_memory_parser: str = "raw",
    rebuild: bool = True,
    max_items: int = 20,
    limit: int | None = None,
) -> dict[str, Any]:
    return build_paper_ops_report(
        sqlite_db,
        document_memory_parser=document_memory_parser,
        rebuild=rebuild,
        max_items=max_items,
        limit=limit,
    )


__all__ = [
    "build_paper_ops_report",
    "build_paper_source_ops_report",
    "summarize_paper_source_verification",
    "verify_paper_source_state",
]
