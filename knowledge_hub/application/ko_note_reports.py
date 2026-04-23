from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from knowledge_hub.application.ops_alerts import evaluate_ko_note_report_alerts
from knowledge_hub.notes.workflow_helpers import (
    approval_summary,
    concept_quality_counts,
    merge_quality_counts,
    remediation_summary,
    report_apply_backlog_count,
    review_queue_counts,
    source_quality_counts,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_ko_note_report(sqlite_db, *, run_id: str, recent_runs: int = 10) -> dict[str, Any]:
    ts = _now_iso()
    run = sqlite_db.get_ko_note_run(run_id)
    if not run:
        return {
            "schema": "knowledge-hub.ko-note.report.result.v1",
            "status": "failed",
            "runId": str(run_id),
            "run": {},
            "recentRuns": [],
            "recentSummary": {},
            "alerts": [],
            "recommendedActions": [],
            "warnings": [f"ko note run not found: {run_id}"],
            "ts": ts,
        }

    items = sqlite_db.list_ko_note_items(run_id=run_id, limit=2000)
    counts: dict[str, int] = defaultdict(int)
    type_counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[str(item.get("status", "unknown"))] += 1
        type_counts[str(item.get("item_type", "unknown"))] += 1
    concept_quality = concept_quality_counts(items)
    source_quality = source_quality_counts(items)
    review_queue = review_queue_counts(items)
    approvals = approval_summary(items)
    remediation = remediation_summary(items)
    run_warnings = list(run.get("warnings_json") or [])
    apply_backlog_count = report_apply_backlog_count(items)

    recent_rows = list(sqlite_db.list_ko_note_runs(limit=max(1, int(recent_runs))))
    recent_payloads: list[dict[str, Any]] = []
    recent_summary = {
        "totalRuns": 0,
        "sourceGenerated": 0,
        "conceptGenerated": 0,
        "approved": 0,
        "rejected": 0,
        "autoApproved": 0,
        "approvedFromReview": 0,
        "reviewQueued": 0,
    }
    for row in recent_rows:
        recent_items = sqlite_db.list_ko_note_items(run_id=str(row.get("run_id") or ""), limit=2000)
        recent_approvals = approval_summary(recent_items)
        recent_review_queue = review_queue_counts(recent_items)
        recent_payload = {
            "runId": str(row.get("run_id") or ""),
            "status": str(row.get("status") or ""),
            "sourceGenerated": int(row.get("source_generated") or 0),
            "conceptGenerated": int(row.get("concept_generated") or 0),
            "approvedCount": int(row.get("approved_count") or 0),
            "rejectedCount": int(row.get("rejected_count") or 0),
            "autoApproved": int((recent_approvals.get("autoApproved") or {}).get("total") or 0),
            "approvedFromReview": int((recent_approvals.get("approvedFromReview") or {}).get("total") or 0),
            "reviewQueued": int((recent_review_queue.get("combined") or {}).get("total") or 0),
            "updatedAt": str(row.get("updated_at") or ""),
            "warnings": list(row.get("warnings_json") or [])[:5],
        }
        recent_payloads.append(recent_payload)
        recent_summary["totalRuns"] += 1
        recent_summary["sourceGenerated"] += recent_payload["sourceGenerated"]
        recent_summary["conceptGenerated"] += recent_payload["conceptGenerated"]
        recent_summary["approved"] += recent_payload["approvedCount"]
        recent_summary["rejected"] += recent_payload["rejectedCount"]
        recent_summary["autoApproved"] += recent_payload["autoApproved"]
        recent_summary["approvedFromReview"] += recent_payload["approvedFromReview"]
        recent_summary["reviewQueued"] += recent_payload["reviewQueued"]

    run_payload = {
        "runId": str(run_id),
        "status": str(run.get("status") or ""),
        "crawlJobId": str(run.get("crawl_job_id") or ""),
        "sourceGenerated": int(run.get("source_generated") or 0),
        "conceptGenerated": int(run.get("concept_generated") or 0),
        "counts": {
            **counts,
            "source": type_counts.get("source", 0),
            "concept": type_counts.get("concept", 0),
            "total": len(items),
        },
        "quality": {
            "concept": concept_quality,
            "source": source_quality,
            "combined": merge_quality_counts(concept_quality, source_quality),
        },
        "reviewQueue": review_queue,
        "remediation": remediation,
        **approvals,
        "warnings": run_warnings[:20],
        "createdAt": str(run.get("created_at") or ""),
        "updatedAt": str(run.get("updated_at") or ""),
    }
    alerts, recommended_actions = evaluate_ko_note_report_alerts(
        run_id=str(run_id),
        run_payload=run_payload,
        apply_backlog_count=apply_backlog_count,
    )
    return {
        "schema": "knowledge-hub.ko-note.report.result.v1",
        "status": "ok",
        "runId": str(run_id),
        "run": run_payload,
        "recentRuns": recent_payloads,
        "recentSummary": recent_summary,
        "alerts": alerts,
        "recommendedActions": recommended_actions,
        "warnings": run_warnings[:20],
        "ts": ts,
    }
