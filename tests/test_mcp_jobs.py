from __future__ import annotations

import asyncio
import sqlite3

from knowledge_hub.application.mcp.jobs import run_async_tool
from knowledge_hub.infrastructure.persistence.stores.mcp_job_store import MCPJobStore


def _store() -> MCPJobStore:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store = MCPJobStore(conn)
    store.ensure_schema()
    return store


def test_run_async_tool_updates_blocked_job_classification_to_p0():
    store = _store()
    active_jobs = {}

    async def _exercise():
        job_id, _queued = await run_async_tool(
            sqlite_db=store,
            active_jobs=active_jobs,
            tool="ask_knowledge",
            request_echo={"tool": "ask_knowledge", "arguments": {"question": "q"}},
            sync_job=lambda: {"answer": "contact secret@example.com"},
        )
        await active_jobs[job_id]
        return job_id

    job_id = asyncio.run(_exercise())
    job = store.get_mcp_job(job_id)

    assert job is not None
    assert job["status"] == "blocked"
    assert job["policy_result"] == "blocked"
    assert job["classification"] == "P0"
    assert job["artifact_json"] == "[REDACTED_BY_POLICY]"
