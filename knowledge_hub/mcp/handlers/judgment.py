"""MCP record surfaces for the persistent user-judgment ledger (labs-gated)."""

from __future__ import annotations

from typing import Any

from knowledge_hub.papers.quarantine import QUARANTINE_REASON_CODE, quarantined_paper_ids

_JUDGMENT_TOOLS = {"record_judgment", "list_judgments"}


async def handle_tool(name: str, arguments: dict[str, Any], ctx: dict[str, Any]):
    if name not in _JUDGMENT_TOOLS:
        return None

    emit = ctx["emit"]
    sqlite_db = ctx["sqlite_db"]
    to_int = ctx["to_int"]
    status_ok = ctx["MCP_TOOL_STATUS_OK"]
    status_failed = ctx["MCP_TOOL_STATUS_FAILED"]

    if name == "list_judgments":
        items = sqlite_db.list_judgments(
            limit=to_int(arguments.get("limit"), 50, minimum=1, maximum=500),
            target_type=str(arguments.get("target_type", "")).strip() or None,
            decision=str(arguments.get("decision", "")).strip() or None,
        )
        return emit(status_ok, {"count": len(items), "items": items})

    # record_judgment
    source_ids = [str(item).strip() for item in arguments.get("source_ids", []) or [] if str(item or "").strip()]
    flagged = quarantined_paper_ids(source_ids)
    if flagged:
        return emit(
            status_failed,
            {
                "error": f"{QUARANTINE_REASON_CODE}: {','.join(flagged)}",
                "decision": "refused",
                "reason": "quarantined_source",
                "quarantinedSourceIds": flagged,
            },
            status_message="quarantined source refused",
        )

    raw_log_id = arguments.get("rag_answer_log_id")
    fields: dict[str, Any] = {
        "target_type": str(arguments.get("target_type", "")).strip(),
        "target_id": str(arguments.get("target_id", "")).strip(),
        "decision": str(arguments.get("decision", "")).strip(),
        "reviewer": str(arguments.get("reviewer", "")).strip(),
        "reason": str(arguments.get("reason", "")).strip(),
        "confidence": str(arguments.get("confidence", "")).strip() or "medium",
        "reviewed_at": str(arguments.get("reviewed_at", "")).strip() or None,
        "target_text": str(arguments.get("target_text", "")) or None,
        "snippet_hashes": [str(item) for item in arguments.get("snippet_hashes", []) or []],
        "evidence_span_ids": [str(item) for item in arguments.get("evidence_span_ids", []) or []],
        "source_ids": source_ids,
        "query_hash": str(arguments.get("query_hash", "")).strip(),
        "rag_answer_log_id": to_int(raw_log_id, None) if raw_log_id is not None else None,
        "runtime_used": str(arguments.get("runtime_used", "")).strip(),
        "decision_source": "mcp",
    }
    supersedes = str(arguments.get("supersedes", "")).strip()
    try:
        if supersedes:
            record = sqlite_db.supersede_judgment(supersedes, **fields)
        else:
            record = sqlite_db.record_judgment(**fields)
    except ValueError as error:
        return emit(status_failed, {"error": str(error)}, status_message="judgment validation failed")
    return emit(status_ok, {"item": record})
