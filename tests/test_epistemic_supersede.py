from __future__ import annotations

import asyncio
from pathlib import Path

from knowledge_hub.core.database import SQLiteDatabase
from knowledge_hub.mcp.handlers import epistemic


def _emit(status, payload, **kwargs):  # noqa: ANN001
    return {"status": status, "payload": payload, **kwargs}


def _ctx(db: SQLiteDatabase) -> dict:
    return {
        "emit": _emit,
        "sqlite_db": db,
        "to_int": lambda value, default, minimum=0, maximum=10_000: max(minimum, min(int(value if value is not None else default), maximum)),
        "to_float": lambda value, default, minimum=0.0, maximum=1.0: max(minimum, min(float(value if value is not None else default), maximum)),
        "MCP_TOOL_STATUS_OK": "ok",
        "MCP_TOOL_STATUS_FAILED": "failed",
    }


def test_belief_review_supersedes_prior_row_and_hides_history_by_default(tmp_path: Path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    db.upsert_belief(
        belief_id="belief_a",
        statement="RAG helps grounding.",
        scope="global",
        status="proposed",
        derived_from_claim_ids=["claim_1"],
    )

    reviewed = db.review_belief(
        "belief_a",
        status="reviewed",
        last_validated_at="2026-04-24T00:00:00+00:00",
        successor_belief_id="belief_b",
    )
    assert reviewed is not None
    assert reviewed["belief_id"] == "belief_b"
    assert reviewed["supersedes"] == "belief_a"
    assert reviewed["status"] == "reviewed"

    original = db.get_belief("belief_a")
    assert original is not None
    assert original["status"] == "superseded"
    assert original["superseded_by"] == "belief_b"

    visible_ids = [item["belief_id"] for item in db.list_beliefs()]
    assert visible_ids == ["belief_b"]

    all_ids = [item["belief_id"] for item in db.list_beliefs(include_superseded=True)]
    assert set(all_ids) == {"belief_a", "belief_b"}

    claim_ids = [item["belief_id"] for item in db.list_beliefs_by_claim_ids(["claim_1"])]
    assert claim_ids == ["belief_b"]


def test_belief_review_from_old_id_follows_latest_chain(tmp_path: Path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    db.upsert_belief(belief_id="belief_a", statement="RAG helps grounding.", status="proposed")
    first = db.review_belief("belief_a", status="reviewed", successor_belief_id="belief_b")
    assert first is not None and first["belief_id"] == "belief_b"

    second = db.review_belief("belief_a", status="trusted", successor_belief_id="belief_c")
    assert second is not None
    assert second["belief_id"] == "belief_c"
    assert second["supersedes"] == "belief_b"

    middle = db.get_belief("belief_b")
    assert middle is not None
    assert middle["status"] == "superseded"
    assert middle["superseded_by"] == "belief_c"


def test_decision_review_creates_successor_and_hides_superseded_rows(tmp_path: Path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    db.upsert_decision(
        decision_id="decision_a",
        title="Choose retrieval strategy",
        summary="baseline",
        related_belief_ids=["belief_a"],
        chosen_option="hybrid",
        status="open",
    )

    reviewed = db.review_decision(
        "decision_a",
        status="reviewed",
        review_due_at="2026-05-01",
        successor_decision_id="decision_b",
    )
    assert reviewed is not None
    assert reviewed["decision_id"] == "decision_b"
    assert reviewed["supersedes"] == "decision_a"
    assert reviewed["status"] == "reviewed"

    original = db.get_decision("decision_a")
    assert original is not None
    assert original["superseded_by"] == "decision_b"

    visible_ids = [item["decision_id"] for item in db.list_decisions()]
    assert visible_ids == ["decision_b"]

    all_ids = [item["decision_id"] for item in db.list_decisions(include_superseded=True)]
    assert set(all_ids) == {"decision_a", "decision_b"}


def test_belief_review_handler_returns_successor_id(tmp_path: Path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    ctx = _ctx(db)

    created = asyncio.run(
        epistemic.handle_tool(
            "belief_upsert",
            {"belief_id": "belief_a", "statement": "RAG helps grounding", "derived_from_claim_ids": ["claim_1"]},
            ctx,
        )
    )
    assert created["status"] == "ok"

    reviewed = asyncio.run(
        epistemic.handle_tool(
            "belief_review",
            {"belief_id": "belief_a", "status": "reviewed", "successor_belief_id": "belief_b"},
            ctx,
        )
    )
    assert reviewed["status"] == "ok"
    assert reviewed["payload"]["item"]["belief_id"] == "belief_b"
    assert reviewed["payload"]["item"]["supersedes"] == "belief_a"

    original = asyncio.run(epistemic.handle_tool("belief_show", {"belief_id": "belief_a"}, ctx))
    assert original["status"] == "ok"
    assert original["payload"]["status"] == "superseded"
    assert original["payload"]["superseded_by"] == "belief_b"
