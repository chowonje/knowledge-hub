from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

import knowledge_hub.interfaces.cli.commands.agent_cmd as agent_cmd_module
import knowledge_hub.interfaces.cli.commands.foundry_cmd as foundry_cmd_module
from knowledge_hub.interfaces.cli.commands.agent_cmd import agent_group, foundry_group
from knowledge_hub.core.config import Config
from knowledge_hub.core.database import SQLiteDatabase


class _StubKhub:
    def __init__(self, config: Config):
        self.config = config
        self._searcher = None

    def searcher(self):
        if self._searcher is None:
            raise RuntimeError("searcher not configured")
        return self._searcher

    def sqlite_db(self):
        return SQLiteDatabase(self.config.sqlite_path)


def _make_config(tmp_path) -> Config:
    config = Config()
    config.set_nested("storage", "sqlite", str(tmp_path / "knowledge.db"))
    return config


def _seed_db(sqlite_path: str) -> None:
    db = SQLiteDatabase(sqlite_path)
    try:
        db.upsert_note(
            note_id="note_alpha",
            title="Alpha Note",
            content="alpha content",
            file_path="/tmp/alpha.md",
            source_type="note",
            metadata={"tags": ["alpha", "core"]},
        )
        db.upsert_note(
            note_id="note_web_beta",
            title="Beta Web",
            content="beta web content",
            file_path="/tmp/beta.md",
            source_type="web",
            metadata={"url": "https://example.com/beta"},
        )
        db.upsert_note(
            note_id="note_gamma",
            title="Gamma Note",
            content="gamma content",
            file_path="/tmp/gamma.md",
            source_type="note",
            metadata={"tags": ["gamma"]},
        )
        db.upsert_paper(
            {
                "arxiv_id": "2501.00001",
                "title": "Paper One",
                "authors": "Author A",
                "year": 2025,
                "field": "AI",
                "importance": 3,
                "notes": "paper summary",
                "pdf_path": "/tmp/paper.pdf",
                "text_path": "/tmp/paper.txt",
                "translated_path": "/tmp/paper.ko.md",
            }
        )

        db.conn.execute("UPDATE notes SET updated_at = ? WHERE id = ?", ("2026-01-01T00:00:00+00:00", "note_alpha"))
        db.conn.execute("UPDATE notes SET updated_at = ? WHERE id = ?", ("2026-01-02T00:00:00+00:00", "note_web_beta"))
        db.conn.execute("UPDATE notes SET updated_at = ? WHERE id = ?", ("2026-01-03T00:00:00+00:00", "note_gamma"))
        db.conn.execute("UPDATE papers SET created_at = ? WHERE arxiv_id = ?", ("2026-01-04T00:00:00+00:00", "2501.00001"))
        db.conn.commit()
    finally:
        db.close()


def test_agent_sync_json_returns_foundry_compatible_payload(tmp_path):
    config = _make_config(tmp_path)
    _seed_db(config.sqlite_path)
    runner = CliRunner()

    result = runner.invoke(
        agent_group,
        ["sync", "--json", "--source", "all", "--limit", "10"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.foundry.connector.sync.result.v2"
    assert payload["connectorId"] == "knowledge-hub"
    assert payload["source_filter"] == "all"
    assert isinstance(payload["items"], list)
    sources = {item["source"] for item in payload["items"]}
    assert {"note", "web", "paper"}.issubset(sources)
    assert "ontologyDelta" in payload
    assert "cursor" in payload
    assert "next_record_ts" in payload["cursor"]
    assert "next_event_ts" in payload["cursor"]
    assert "hasMore" in payload["cursor"]


def test_agent_sync_cursor_and_limit_work_for_incremental_note_sync(tmp_path):
    config = _make_config(tmp_path)
    _seed_db(config.sqlite_path)
    runner = CliRunner()

    limited = runner.invoke(
        agent_group,
        ["sync", "--json", "--source", "note", "--limit", "1"],
        obj={"khub": _StubKhub(config)},
    )
    assert limited.exit_code == 0
    limited_payload = json.loads(limited.output)
    assert len(limited_payload["items"]) == 1
    assert limited_payload["cursor"]["hasMore"] is True
    first_cursor = limited_payload["cursor"]["next_record_ts"]

    incremental = runner.invoke(
        agent_group,
        ["sync", "--json", "--source", "note", "--limit", "10", "--cursor", first_cursor],
        obj={"khub": _StubKhub(config)},
    )
    assert incremental.exit_code == 0
    incremental_payload = json.loads(incremental.output)
    assert all(item["source"] == "note" for item in incremental_payload["items"])
    assert all(item["updatedAt"] > first_cursor for item in incremental_payload["items"])


def test_agent_run_json_normalizes_foundry_payload(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    runner = CliRunner()

    def _fake_run(command, command_args, timeout_sec=120):  # noqa: ANN001
        assert command == "run"
        assert "--dump-json" in command_args
        return (
            {
                "schema": "knowledge-hub.foundry.agent.run.result.v1",
                "source": "foundry-core/cli-agent",
                "runId": "run_test_001",
                "status": "completed",
                "goal": "RAG 요약",
                "role": "planner",
                "orchestratorMode": "adaptive",
                "stage": "DONE",
                "transitions": [
                    {
                        "stage": "PLAN",
                        "status": "PLAN",
                        "message": "plan",
                        "at": "2026-01-01T00:00:00+00:00",
                    }
                ],
                "verify": {"allowed": True, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
                "writeback": {"ok": True, "detail": "ok"},
                "artifact": {
                    "jsonContent": {"answer": "ok"},
                    "classification": "P2",
                    "generatedAt": "2026-01-01T00:00:00+00:00",
                },
                "createdAt": "2026-01-01T00:00:00+00:00",
                "updatedAt": "2026-01-01T00:00:01+00:00",
                "dryRun": False,
            },
            None,
        )

    monkeypatch.setattr("knowledge_hub.interfaces.cli.commands.agent_cmd._run_foundry_cli", _fake_run)
    result = runner.invoke(
        agent_group,
        ["run", "--goal", "RAG 요약", "--json"],
        obj={"khub": _StubKhub(config)},
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.foundry.agent.run.result.v1"
    assert payload["status"] == "completed"
    assert payload["role"] == "planner"
    assert payload["orchestratorMode"] == "adaptive"
    assert isinstance(payload["transitions"], list)
    assert "gateway" not in payload


def test_agent_run_policy_gate_reclassifies_sensitive_artifact(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    runner = CliRunner()

    def _fake_run(command, command_args, timeout_sec=120):  # noqa: ANN001
        assert command == "run"
        return (
            {
                "schema": "knowledge-hub.foundry.agent.run.result.v1",
                "source": "foundry-core/cli-agent",
                "runId": "run_policy_gate_001",
                "status": "completed",
                "goal": "RAG 요약",
                "role": "planner",
                "orchestratorMode": "adaptive",
                "stage": "DONE",
                "transitions": [],
                "verify": {"allowed": True, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
                "writeback": {"ok": True, "detail": "ok"},
                "artifact": {
                    "jsonContent": {"answer": "api_key: test-secret-value"},
                    "classification": "P2",
                    "generatedAt": "2026-01-01T00:00:00+00:00",
                    "metadata": {"trace": "private@example.com"},
                },
                "createdAt": "2026-01-01T00:00:00+00:00",
                "updatedAt": "2026-01-01T00:00:01+00:00",
                "dryRun": False,
            },
            None,
        )

    monkeypatch.setattr("knowledge_hub.interfaces.cli.commands.agent_cmd._run_foundry_cli", _fake_run)
    result = runner.invoke(
        agent_group,
        ["run", "--goal", "RAG 요약", "--json"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "blocked"
    assert payload["stage"] == "VERIFY"
    assert payload["verify"]["allowed"] is False
    assert payload["verify"]["policyAllowed"] is False
    assert payload["writeback"]["ok"] is False
    assert payload["artifact"]["classification"] == "P0"
    assert payload["artifact"]["jsonContent"] == agent_cmd_module.POLICY_REDACTION_TEXT
    assert payload["artifact"]["metadata"] == {}
    assert any("P0" in item for item in payload["verify"]["schemaErrors"])


def test_agent_run_dry_run_json_adds_gateway_metadata(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    runner = CliRunner()

    def _fake_run(command, command_args, timeout_sec=120):  # noqa: ANN001
        assert command == "run"
        assert "--dry-run" in command_args
        return (
            {
                "schema": "knowledge-hub.foundry.agent.run.result.v1",
                "source": "foundry-core/cli-agent",
                "runId": "run_test_dry_001",
                "status": "blocked",
                "goal": "RAG 요약",
                "role": "planner",
                "orchestratorMode": "adaptive",
                "stage": "VERIFY",
                "transitions": [
                    {
                        "stage": "PLAN",
                        "status": "PLAN",
                        "message": "plan",
                        "at": "2026-01-01T00:00:00+00:00",
                    }
                ],
                "verify": {"allowed": False, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
                "writeback": {"ok": False, "detail": "dry-run: writeback skipped"},
                "artifact": None,
                "createdAt": "2026-01-01T00:00:00+00:00",
                "updatedAt": "2026-01-01T00:00:01+00:00",
                "dryRun": True,
            },
            None,
        )

    monkeypatch.setattr("knowledge_hub.interfaces.cli.commands.agent_cmd._run_foundry_cli", _fake_run)
    result = runner.invoke(
        agent_group,
        ["run", "--goal", "RAG 요약", "--json", "--dry-run"],
        obj={"khub": _StubKhub(config)},
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dryRun"] is True
    assert payload["gateway"]["surface"] == "agent_run"
    assert payload["gateway"]["mode"] == "dry_run"
    assert payload["gateway"]["executionAllowed"] is False


def test_agent_run_fallback_when_foundry_unavailable(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    runner = CliRunner()
    monkeypatch.setattr("knowledge_hub.interfaces.cli.commands.agent_cmd._run_foundry_cli", lambda *_args, **_kwargs: (None, "bridge unavailable"))

    result = runner.invoke(
        agent_group,
        ["run", "--goal", "RAG 요약", "--json", "--dry-run"],
        obj={"khub": _StubKhub(config)},
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["source"] == "knowledge-hub/cli.agent.run.fallback"
    assert payload["status"] == "blocked"
    assert payload["dryRun"] is True
    assert payload["gateway"]["surface"] == "agent_run"
    assert payload["gateway"]["mode"] == "dry_run"
    assert "bridge unavailable" in "\n".join(payload["verify"]["schemaErrors"])


def test_agent_writeback_request_creates_schema_backed_pending_request(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    config.set_nested("validation", "schema", "strict", True)
    runner = CliRunner()

    def _fake_run(command, command_args, timeout_sec=120):  # noqa: ANN001
        assert command == "run"
        assert "--dry-run" in command_args
        return (
            {
                "schema": "knowledge-hub.foundry.agent.run.result.v1",
                "source": "foundry-core/cli-agent",
                "runId": "run_writeback_req_001",
                "status": "blocked",
                "goal": "Apply repo-local patch",
                "role": "planner",
                "orchestratorMode": "adaptive",
                "stage": "VERIFY",
                "transitions": [],
                "verify": {"allowed": False, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
                "writeback": {"ok": False, "detail": "dry-run: writeback skipped"},
                "artifact": None,
                "createdAt": "2026-01-01T00:00:00+00:00",
                "updatedAt": "2026-01-01T00:00:01+00:00",
                "dryRun": True,
                "plan": ["build_task_context", "ask_knowledge"],
            },
            None,
        )

    monkeypatch.setattr("knowledge_hub.interfaces.cli.commands.agent_cmd._run_foundry_cli", _fake_run)
    result = runner.invoke(
        agent_group,
        ["writeback-request", "--goal", "Apply repo-local patch", "--repo-path", str(tmp_path), "--json"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.agent.writeback.request.result.v1"
    assert payload["status"] == "ok"
    assert payload["repoPath"] == str(tmp_path.resolve())
    assert payload["requestOperation"] == "created"
    assert payload["request"]["scope"] == "agent"
    assert payload["request"]["actionType"] == "agent_repo_writeback_request"
    assert payload["request"]["status"] == "pending"
    assert payload["approval"]["required"] is True
    assert payload["approval"]["status"] == "pending"
    assert "ops action-ack" in payload["approval"]["commands"]["ack"]
    assert "ops action-execute" in payload["approval"]["commands"]["execute"]
    assert payload["dryRun"]["dryRun"] is True
    assert payload["dryRun"]["gateway"]["surface"] == "agent_run"
    assert payload["writebackPreview"]["kind"] == "repo_local_predicted_write_set"
    assert payload["writebackPreview"]["advisory"] is True
    assert payload["request"]["action"]["targetPolicy"] == "docs_only"
    assert payload["request"]["action"]["allowedPathPrefixes"] == ["docs/adr/", "docs/status/", "reviews/", "worklog/"]
    assert payload["writebackPreview"]["constraints"]["allowedPathPrefixes"] == ["docs/adr/", "docs/status/", "reviews/", "worklog/"]
    assert payload["request"]["action"]["writebackPreviewFingerprint"] == payload["writebackPreview"]["previewFingerprint"]
    assert payload["gateway"]["version"] == "v2"
    assert payload["gateway"]["surface"] == "agent_writeback_request"
    assert payload["gateway"]["approvalRequired"] is True


def test_agent_writeback_request_reuses_same_queue_identity(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    config.set_nested("validation", "schema", "strict", True)
    runner = CliRunner()

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.agent_cmd._run_foundry_cli",
        lambda command, command_args, timeout_sec=120: (  # noqa: ARG005, ANN001
            {
                "schema": "knowledge-hub.foundry.agent.run.result.v1",
                "source": "foundry-core/cli-agent",
                "runId": "run_writeback_req_002",
                "status": "blocked",
                "goal": "Apply repo-local patch",
                "role": "planner",
                "orchestratorMode": "adaptive",
                "stage": "VERIFY",
                "transitions": [],
                "verify": {"allowed": False, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
                "writeback": {"ok": False, "detail": "dry-run: writeback skipped"},
                "artifact": None,
                "createdAt": "2026-01-01T00:00:00+00:00",
                "updatedAt": "2026-01-01T00:00:01+00:00",
                "dryRun": True,
                "plan": ["build_task_context", "ask_knowledge"],
            },
            None,
        ),
    )

    first = runner.invoke(
        agent_group,
        ["writeback-request", "--goal", "Apply repo-local patch", "--repo-path", str(tmp_path), "--json"],
        obj={"khub": _StubKhub(config)},
    )
    second = runner.invoke(
        agent_group,
        ["writeback-request", "--goal", "Apply repo-local patch", "--repo-path", str(tmp_path), "--json"],
        obj={"khub": _StubKhub(config)},
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    first_payload = json.loads(first.output)
    second_payload = json.loads(second.output)
    assert first_payload["request"]["actionId"] == second_payload["request"]["actionId"]
    assert second_payload["requestOperation"] == "updated"
    assert second_payload["request"]["status"] == "pending"
    assert first_payload["writebackPreview"]["previewFingerprint"] == second_payload["writebackPreview"]["previewFingerprint"]


def test_agent_writeback_request_includes_workspace_writeback_preview(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    config.set_nested("validation", "schema", "strict", True)
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("- Preserve boundaries\n", encoding="utf-8")
    (repo / "docs" / "status").mkdir(parents=True)
    (repo / "worklog").mkdir(parents=True)
    (repo / "docs" / "status" / "consumer.md").write_text("status note\n", encoding="utf-8")
    (repo / "worklog" / "2026-04-18.md").write_text("worklog note\n", encoding="utf-8")

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.agent_cmd._run_foundry_cli",
        lambda command, command_args, timeout_sec=120: (  # noqa: ARG005, ANN001
            {
                "schema": "knowledge-hub.foundry.agent.run.result.v1",
                "source": "foundry-core/cli-agent",
                "runId": "run_writeback_req_preview_001",
                "status": "blocked",
                "goal": "Update docs/status/consumer.md and worklog/2026-04-18.md",
                "role": "planner",
                "orchestratorMode": "adaptive",
                "stage": "VERIFY",
                "transitions": [],
                "verify": {"allowed": False, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
                "writeback": {"ok": False, "detail": "dry-run: writeback skipped"},
                "artifact": None,
                "createdAt": "2026-01-01T00:00:00+00:00",
                "updatedAt": "2026-01-01T00:00:01+00:00",
                "dryRun": True,
                "plan": ["build_task_context", "ask_knowledge"],
            },
            None,
        ),
    )

    result = runner.invoke(
        agent_group,
        [
            "writeback-request",
            "--goal",
            "Update docs/status/consumer.md and worklog/2026-04-18.md",
            "--repo-path",
            str(repo),
            "--include-workspace",
            "--max-workspace-files",
            "4",
            "--json",
        ],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    preview = payload["writebackPreview"]
    assert preview["status"] == "ok"
    assert preview["workspaceContext"]["included"] is True
    assert preview["workspaceContext"]["mode"] == "explicit"
    assert preview["targetCount"] >= 1
    assert any(item["relativePath"] == "docs/status/consumer.md" for item in preview["targets"])
    assert preview == payload["request"]["action"]["writebackPreview"]
    assert payload["request"]["action"]["writebackPreviewFingerprint"] == preview["previewFingerprint"]


def test_agent_writeback_request_filters_non_docs_targets_from_preview(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    config.set_nested("validation", "schema", "strict", True)
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "agent.ts").write_text("export const enabled = true;\n", encoding="utf-8")

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.agent_cmd._run_foundry_cli",
        lambda command, command_args, timeout_sec=120: (  # noqa: ARG005, ANN001
            {
                "schema": "knowledge-hub.foundry.agent.run.result.v1",
                "source": "foundry-core/cli-agent",
                "runId": "run_writeback_req_preview_002",
                "status": "blocked",
                "goal": "Implement src/agent.ts change",
                "role": "planner",
                "orchestratorMode": "adaptive",
                "stage": "VERIFY",
                "transitions": [],
                "verify": {"allowed": False, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
                "writeback": {"ok": False, "detail": "dry-run: writeback skipped"},
                "artifact": None,
                "createdAt": "2026-01-01T00:00:00+00:00",
                "updatedAt": "2026-01-01T00:00:01+00:00",
                "dryRun": True,
                "plan": ["build_task_context", "ask_knowledge"],
            },
            None,
        ),
    )

    result = runner.invoke(
        agent_group,
        [
            "writeback-request",
            "--goal",
            "Implement src/agent.ts change",
            "--repo-path",
            str(repo),
            "--include-workspace",
            "--json",
        ],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["writebackPreview"]["status"] == "partial"
    assert payload["writebackPreview"]["targetCount"] == 0
    assert all(item["relativePath"] != "src/agent.ts" for item in payload["writebackPreview"]["targets"])


def test_agent_context_json_returns_task_context_payload(tmp_path):
    config = _make_config(tmp_path)
    runner = CliRunner()
    khub = _StubKhub(config)
    khub._searcher = type(
        "_Searcher",
        (),
        {
            "search": staticmethod(
                lambda *_args, **_kwargs: [
                    type(
                        "_Result",
                        (),
                        {
                            "metadata": {"title": "Vault Note", "source_type": "note"},
                            "score": 0.9,
                            "document": "vault evidence",
                        },
                    )()
                ]
            )
        },
    )()

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("- Preserve boundaries\n", encoding="utf-8")
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "agent.ts").write_text("export const enabled = true;\n", encoding="utf-8")

    result = runner.invoke(
        agent_group,
        ["context", "Implement src/agent.ts change", "--repo-path", str(repo), "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.task-context.result.v1"
    assert payload["mode"] == "coding"
    assert payload["workspace_files"][0]["source_type"] == "project"
    assert payload["gateway"]["surface"] == "task_context"
    assert payload["gateway"]["mode"] == "context"


def test_agent_run_forwards_task_context_options(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    runner = CliRunner()
    captured = {}

    def _fake_run(command, command_args, timeout_sec=120):  # noqa: ANN001
        captured["command"] = command
        captured["args"] = list(command_args)
        return (
            {
                "schema": "knowledge-hub.foundry.agent.run.result.v1",
                "source": "foundry-core/cli-agent",
                "runId": "run_task_context_001",
                "status": "completed",
                "goal": "Implement context",
                "role": "planner",
                "orchestratorMode": "adaptive",
                "stage": "DONE",
                "transitions": [],
                "verify": {"allowed": True, "schemaValid": True, "policyAllowed": True, "schemaErrors": []},
                "writeback": {"ok": True, "detail": "ok"},
                "artifact": {
                    "jsonContent": {"answer": "ok"},
                    "classification": "P2",
                    "generatedAt": "2026-01-01T00:00:00+00:00",
                },
                "createdAt": "2026-01-01T00:00:00+00:00",
                "updatedAt": "2026-01-01T00:00:01+00:00",
                "dryRun": False,
            },
            None,
        )

    monkeypatch.setattr("knowledge_hub.interfaces.cli.commands.agent_cmd._run_foundry_cli", _fake_run)
    repo = tmp_path / "repo"
    repo.mkdir()

    result = runner.invoke(
        agent_group,
        [
            "run",
            "--goal",
            "Implement context",
            "--repo-path",
            str(repo),
            "--no-include-workspace",
            "--max-workspace-files",
            "3",
            "--json",
        ],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0
    assert captured["command"] == "run"
    assert "--repo-path" in captured["args"]
    assert "--no-include-workspace" in captured["args"]
    assert "--max-workspace-files" in captured["args"]


def test_agent_discover_json_success_with_feature_results(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    _seed_db(config.sqlite_path)
    runner = CliRunner()

    def _fake_foundry(command, command_args, timeout_sec=120):  # noqa: ANN001
        if command == "sync":
            return (
                {
                    "schema": "knowledge-hub.foundry.connector.sync.result.v2",
                    "status": "done",
                    "emittedEventCount": 3,
                },
                None,
            )
        if command == "feature":
            feature_name = str(command_args[0])
            if feature_name == "list":
                return ({"featureNames": ["daily_coach", "focus_analytics", "risk_alert"]}, None)
            return ({"schema": "knowledge-hub.feature.result.v1", "feature": feature_name, "status": "ok"}, None)
        return (None, "unexpected command")

    monkeypatch.setattr("knowledge_hub.interfaces.cli.commands.foundry_cmd._run_foundry_cli", _fake_foundry)

    result = runner.invoke(
        agent_group,
        [
            "discover",
            "--source",
            "all",
            "--days",
            "7",
            "--feature",
            "daily_coach",
            "--feature",
            "risk_alert",
            "--json",
            "--no-fail-on-error",
        ],
        obj={"khub": _StubKhub(config)},
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.agent.discover.result.v1"
    assert payload["status"] == "ok"
    assert len(payload["features"]) == 2
    assert all(item["ok"] for item in payload["features"])
    assert payload["request"]["resolution"]["features"] == "cli"


def test_agent_discover_fail_on_partial_returns_nonzero(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    _seed_db(config.sqlite_path)
    runner = CliRunner()

    def _fake_foundry(command, command_args, timeout_sec=120):  # noqa: ANN001
        if command == "sync":
            return ({"status": "done"}, None)
        if command == "feature":
            feature_name = str(command_args[0])
            if feature_name == "risk_alert":
                return (None, "feature runtime error")
            return ({"status": "ok", "feature": feature_name}, None)
        return (None, "unexpected command")

    monkeypatch.setattr("knowledge_hub.interfaces.cli.commands.foundry_cmd._run_foundry_cli", _fake_foundry)
    result = runner.invoke(
        agent_group,
        [
            "discover",
            "--feature",
            "daily_coach",
            "--feature",
            "risk_alert",
            "--json",
            "--fail-on-partial",
            "--no-fail-on-error",
        ],
        obj={"khub": _StubKhub(config)},
    )
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "partial"
    assert any(item["ok"] is False for item in payload["features"])


def test_agent_discover_resume_merges_request_with_cli_priority(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    _seed_db(config.sqlite_path)
    runner = CliRunner()

    def _fake_foundry(command, command_args, timeout_sec=120):  # noqa: ANN001
        if command == "sync":
            return ({"status": "done"}, None)
        if command == "feature":
            feature_name = str(command_args[0])
            if feature_name == "list":
                return ({"featureNames": ["daily_coach", "focus_analytics", "risk_alert"]}, None)
            return ({"status": "ok", "feature": feature_name}, None)
        return (None, "unexpected command")

    monkeypatch.setattr("knowledge_hub.interfaces.cli.commands.foundry_cmd._run_foundry_cli", _fake_foundry)

    resume_payload = {
        "schema": "knowledge-hub.agent.discover.result.v1",
        "request": {
            "source": "web",
            "days": 21,
            "from": "2026-01-01T00:00:00+00:00",
            "to": None,
            "topK": 10,
            "limit": 100,
            "intent": "summarize",
            "features": ["risk_alert"],
            "expenseThreshold": None,
            "minSleepHours": None,
            "eventLogPath": ".foundry-ontology-events.jsonl",
            "stateFile": ".foundry-sync-state.json",
            "saveState": False,
            "resumeSource": False,
            "resolution": {
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
            },
        },
    }
    resume_path = Path(tmp_path / "resume-discover.json")
    resume_path.write_text(json.dumps(resume_payload), encoding="utf-8")

    result = runner.invoke(
        agent_group,
        [
            "discover",
            "--resume",
            str(resume_path),
            "--days",
            "3",
            "--feature",
            "daily_coach",
            "--json",
            "--no-fail-on-error",
        ],
        obj={"khub": _StubKhub(config)},
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    request = payload["request"]
    assert request["source"] == "web"
    assert request["resolution"]["source"] == "resume"
    assert request["days"] == 3
    assert request["resolution"]["days"] == "cli"
    assert request["features"] == ["daily_coach"]
    assert request["resolution"]["features"] == "cli"
    assert request["resumeSource"] is True


def test_agent_discover_validate_accepts_valid_payload(tmp_path):
    runner = CliRunner()
    valid_payload = {
        "schema": "knowledge-hub.agent.discover.result.v1",
        "runId": "discover_001",
        "source": "knowledge-hub/cli.agent.discover",
        "status": "ok",
        "sync": {},
        "features": [{"feature": "daily_coach", "ok": True, "result": {}, "error": None}],
        "request": {
            "source": "all",
            "days": 7,
            "from": None,
            "to": None,
            "topK": 8,
            "limit": None,
            "intent": "analyze",
            "features": ["daily_coach"],
            "expenseThreshold": None,
            "minSleepHours": None,
            "eventLogPath": None,
            "stateFile": None,
            "saveState": True,
            "resumeSource": False,
            "resolution": {
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
            },
        },
    }
    input_path = Path(tmp_path / "discover.json")
    input_path.write_text(json.dumps(valid_payload), encoding="utf-8")

    result = runner.invoke(
        agent_group,
        ["discover-validate", "--input", str(input_path), "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["valid"] is True


def test_agent_help_hides_foundry_operator_subcommands():
    runner = CliRunner()

    result = runner.invoke(agent_group, ["--help"])

    assert result.exit_code == 0
    command_lines = {
        line.strip().split()[0]
        for line in result.output.splitlines()
        if line.startswith("  ") and line.strip() and not line.lstrip().startswith("-")
    }
    assert {"context", "run", "writeback-request"}.issubset(command_lines)
    assert {
        "sync",
        "discover",
        "discover-validate",
        "foundry-conflict-list",
        "foundry-conflict-apply",
        "foundry-conflict-reject",
    }.isdisjoint(command_lines)


def test_hidden_agent_operator_subcommands_remain_directly_invokable():
    runner = CliRunner()

    for command_name in [
        "sync",
        "discover",
        "discover-validate",
        "foundry-conflict-list",
        "foundry-conflict-apply",
        "foundry-conflict-reject",
    ]:
        result = runner.invoke(agent_group, [command_name, "--help"])

        assert result.exit_code == 0, command_name
        assert "Usage:" in result.output


def test_agent_foundry_group_reexport_points_to_dedicated_foundry_module():
    assert foundry_group is foundry_cmd_module.foundry_group
    assert agent_cmd_module.foundry_group is foundry_cmd_module.foundry_group


def test_foundry_help_exposes_operator_subcommands():
    runner = CliRunner()

    result = runner.invoke(foundry_group, ["--help"])

    assert result.exit_code == 0
    command_lines = {
        line.strip().split()[0]
        for line in result.output.splitlines()
        if line.startswith("  ") and line.strip() and not line.lstrip().startswith("-")
    }
    assert {
        "sync",
        "discover",
        "discover-validate",
        "conflict-list",
        "conflict-apply",
        "conflict-reject",
    }.issubset(command_lines)
