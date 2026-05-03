from __future__ import annotations

import asyncio
from pathlib import Path

from knowledge_hub.core.database import SQLiteDatabase
from knowledge_hub.mcp.handlers import epistemic


def _emit(status, payload, **kwargs):  # noqa: ANN001
    return {"status": status, "payload": payload, **kwargs}


def test_epistemic_handler_lists_profiles_and_upserts_belief(tmp_path: Path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    ctx = {
        "emit": _emit,
        "sqlite_db": db,
        "to_int": lambda value, default, minimum=0, maximum=10_000: max(minimum, min(int(value if value is not None else default), maximum)),
        "to_float": lambda value, default, minimum=0.0, maximum=1.0: max(minimum, min(float(value if value is not None else default), maximum)),
        "MCP_TOOL_STATUS_OK": "ok",
        "MCP_TOOL_STATUS_FAILED": "failed",
    }

    listed = asyncio.run(epistemic.handle_tool("ontology_profile_list", {}, ctx))
    assert listed["status"] == "ok"
    assert any(item["profile_id"] == "core" for item in listed["payload"]["items"])

    upserted = asyncio.run(
        epistemic.handle_tool(
            "belief_upsert",
            {"statement": "RAG helps grounding", "derived_from_claim_ids": ["claim_1"]},
            ctx,
        )
    )
    assert upserted["status"] == "ok"
    belief_id = upserted["payload"]["item"]["belief_id"]

    shown = asyncio.run(epistemic.handle_tool("belief_show", {"belief_id": belief_id}, ctx))
    assert shown["status"] == "ok"
    assert shown["payload"]["derived_from_claim_ids"] == ["claim_1"]
