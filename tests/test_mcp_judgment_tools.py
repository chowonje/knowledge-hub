"""MCP + CLI record surfaces for the persistent judgment ledger (labs-gated)."""

from __future__ import annotations

import asyncio
import importlib
import json

import pytest
from click.testing import CliRunner

from knowledge_hub.application.mcp.responses import (
    CORE_RUNTIME_TOOL_NAMES,
    JUDGMENT_TOOL_NAMES,
    LABS_TOOL_NAMES,
    to_int,
)
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.mcp.handlers import judgment as judgment_handler


def _import_mcp_server():
    try:
        return importlib.import_module("knowledge_hub.interfaces.mcp.server")
    except SystemExit:
        pytest.skip("mcp dependency is unavailable in test environment")


def _import_tool_specs():
    try:
        return importlib.import_module("knowledge_hub.mcp.tool_specs")
    except SystemExit:
        pytest.skip("mcp dependency is unavailable in test environment")


def _decode_response(contents):
    assert contents
    return json.loads(contents[0].text)


def _db(tmp_path) -> SQLiteDatabase:
    return SQLiteDatabase(str(tmp_path / "knowledge.db"), enable_event_store=False)


def _ctx(sqlite_db):
    def emit(status, payload, **kwargs):
        return {"status": status, "payload": payload, "statusMessage": kwargs.get("status_message")}

    return {
        "emit": emit,
        "sqlite_db": sqlite_db,
        "to_int": to_int,
        "MCP_TOOL_STATUS_OK": "ok",
        "MCP_TOOL_STATUS_FAILED": "failed",
    }


def _record_arguments(**overrides):
    arguments = {
        "target_type": "answer",
        "target_id": "answer_001",
        "decision": "accept",
        "reviewer": "won",
        "reason": "spans verified",
    }
    arguments.update(overrides)
    return arguments


def test_record_judgment_handler_records_with_mcp_decision_source(tmp_path):
    db = _db(tmp_path)
    try:
        result = asyncio.run(
            judgment_handler.handle_tool("record_judgment", _record_arguments(), _ctx(db))
        )
        assert result["status"] == "ok"
        item = result["payload"]["item"]
        assert item["decision_source"] == "mcp"
        assert (tmp_path / "judgments.jsonl").exists()
        assert db.get_judgment(item["judgment_id"]) is not None
    finally:
        db.close()


def test_record_judgment_handler_validation_failure_emits_failed(tmp_path):
    db = _db(tmp_path)
    try:
        result = asyncio.run(
            judgment_handler.handle_tool(
                "record_judgment", _record_arguments(decision="approve"), _ctx(db)
            )
        )
        assert result["status"] == "failed"
        assert "decision" in result["payload"]["error"]
        assert not (tmp_path / "judgments.jsonl").exists()
    finally:
        db.close()


def test_record_judgment_handler_refuses_quarantined_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "knowledge_hub.papers.quarantine.QUARANTINED_PAPER_IDS",
        frozenset({"paper_quarantined_x"}),
    )
    db = _db(tmp_path)
    try:
        result = asyncio.run(
            judgment_handler.handle_tool(
                "record_judgment",
                _record_arguments(source_ids=["paper_quarantined_x", "2203.15556"]),
                _ctx(db),
            )
        )
        assert result["status"] == "failed"
        assert result["payload"]["decision"] == "refused"
        assert result["payload"]["reason"] == "quarantined_source"
        assert result["payload"]["quarantinedSourceIds"] == ["paper_quarantined_x"]
        assert not (tmp_path / "judgments.jsonl").exists()
        assert db.list_judgments() == []
    finally:
        db.close()


def test_record_judgment_handler_supports_supersedes(tmp_path):
    db = _db(tmp_path)
    try:
        first = asyncio.run(
            judgment_handler.handle_tool(
                "record_judgment", _record_arguments(decision="thin"), _ctx(db)
            )
        )
        old_id = first["payload"]["item"]["judgment_id"]
        second = asyncio.run(
            judgment_handler.handle_tool(
                "record_judgment",
                _record_arguments(decision="accept", supersedes=old_id),
                _ctx(db),
            )
        )
        new_item = second["payload"]["item"]
        assert new_item["supersedes"] == old_id
        assert db.get_judgment(old_id)["superseded_by"] == new_item["judgment_id"]
    finally:
        db.close()


def test_list_judgments_handler_lists_and_filters(tmp_path):
    db = _db(tmp_path)
    try:
        asyncio.run(judgment_handler.handle_tool("record_judgment", _record_arguments(), _ctx(db)))
        asyncio.run(
            judgment_handler.handle_tool(
                "record_judgment",
                _record_arguments(target_type="brief", target_id="brief_1", decision="thin"),
                _ctx(db),
            )
        )
        result = asyncio.run(
            judgment_handler.handle_tool("list_judgments", {"target_type": "brief"}, _ctx(db))
        )
        assert result["status"] == "ok"
        assert result["payload"]["count"] == 1
        assert result["payload"]["items"][0]["target_id"] == "brief_1"
    finally:
        db.close()


def test_judgment_handler_ignores_other_tools(tmp_path):
    result = asyncio.run(judgment_handler.handle_tool("search_knowledge", {}, {}))
    assert result is None


def test_judgment_tools_are_labs_and_core_runtime_gated():
    assert JUDGMENT_TOOL_NAMES == {"record_judgment", "list_judgments"}
    assert JUDGMENT_TOOL_NAMES <= LABS_TOOL_NAMES
    assert JUDGMENT_TOOL_NAMES <= CORE_RUNTIME_TOOL_NAMES


def test_default_profile_hides_judgment_tools(monkeypatch):
    tool_specs = _import_tool_specs()
    monkeypatch.setenv("KHUB_MCP_PROFILE", "default")
    names = {tool.name for tool in tool_specs.build_tools(profile="default")}
    assert "record_judgment" not in names
    assert "list_judgments" not in names

    labs_names = {tool.name for tool in tool_specs.build_tools(profile="labs")}
    assert "record_judgment" in labs_names
    assert "list_judgments" in labs_names


def test_default_profile_blocks_direct_judgment_calls(monkeypatch):
    monkeypatch.setenv("KHUB_MCP_PROFILE", "default")
    module = _import_mcp_server()

    for tool_name, arguments in (
        ("record_judgment", _record_arguments()),
        ("list_judgments", {"limit": 5}),
    ):
        blocked = _decode_response(asyncio.run(module.call_tool(tool_name, arguments)))
        assert blocked["status"] == "failed"
        assert blocked["statusMessage"] == "tool blocked by MCP profile"
        assert blocked["payload"]["profile"] == "default"
        assert blocked["payload"]["allowedProfiles"] == ["labs", "all"]


class _JudgeKhub:
    def __init__(self, db):
        self._db = db

    def sqlite_db(self):
        return self._db


def _invoke_judge(db, args):
    from knowledge_hub.interfaces.cli.commands.judge_cmd import judge_group

    return CliRunner().invoke(judge_group, args, obj={"khub": _JudgeKhub(db)})


def test_cli_judge_record_writes_jsonl_and_mirror(tmp_path):
    db = _db(tmp_path)
    try:
        result = _invoke_judge(
            db,
            [
                "record",
                "--target-type", "brief",
                "--target-id", "brief_001",
                "--decision", "accept",
                "--reviewer", "won",
                "--reason", "first judged brief",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        item = payload["item"]
        assert item["decision_source"] == "cli"
        assert (tmp_path / "judgments.jsonl").exists()
        assert db.get_judgment(item["judgment_id"]) is not None

        listed = _invoke_judge(db, ["list", "--target-type", "brief", "--json"])
        assert listed.exit_code == 0, listed.output
        assert json.loads(listed.output)["count"] == 1
    finally:
        db.close()


def test_cli_judge_record_refuses_quarantined_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "knowledge_hub.papers.quarantine.QUARANTINED_PAPER_IDS",
        frozenset({"paper_quarantined_x"}),
    )
    db = _db(tmp_path)
    try:
        result = _invoke_judge(
            db,
            [
                "record",
                "--target-type", "answer",
                "--target-id", "answer_001",
                "--decision", "accept",
                "--reviewer", "won",
                "--reason", "ok",
                "--source-id", "paper_quarantined_x",
            ],
        )
        assert result.exit_code != 0
        assert "quarantined_source" in result.output
        assert not (tmp_path / "judgments.jsonl").exists()
    finally:
        db.close()
