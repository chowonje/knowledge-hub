from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.interfaces.cli.commands import add_cmd as add_module
from knowledge_hub.interfaces.cli.commands.add_cmd import add_cmd, detect_add_route
from knowledge_hub.interfaces.cli.main import cli


class _StubConfig:
    def __init__(self, tmp_path: Path):
        self.sqlite_path = str(tmp_path / "knowledge.db")


class _StubWebIngestService:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def crawl_and_ingest(self, **kwargs):  # noqa: ANN003
        self.calls.append(dict(kwargs))
        return {
            "status": "ok",
            "requested": len(kwargs.get("urls") or []),
            "crawled": len(kwargs.get("urls") or []),
            "stored": len(kwargs.get("urls") or []),
            "indexedChunks": 2 if kwargs.get("index") else 0,
            "runId": "crawl_ingest_test",
            "warnings": [],
        }


class _StubKhub:
    def __init__(self, tmp_path: Path):
        self.config = _StubConfig(tmp_path)
        self.service = _StubWebIngestService()

    def web_ingest_service(self):
        return self.service


def test_detect_add_route_auto_selects_common_source_types():
    assert detect_add_route("https://youtu.be/abc123").kind == "youtube"
    assert detect_add_route("https://arxiv.org/abs/2401.00001").kind == "paper_url"
    assert detect_add_route("https://example.com/guide").kind == "web"
    assert detect_add_route("retrieval augmented generation").kind == "paper_query"


def test_root_help_promotes_add_and_hides_ingest_internals():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "add" in result.output
    assert "discover" not in result.output
    assert "crawl" not in result.output
    assert "health" not in result.output


def test_add_help_shows_single_source_examples():
    result = CliRunner().invoke(add_cmd, ["--help"])

    assert result.exit_code == 0
    assert "Add a source with one command" in result.output
    assert "--type" in result.output
    assert "--paper-parser" in result.output


def test_add_web_outputs_schema_backed_facade_payload(tmp_path):
    khub = _StubKhub(tmp_path)
    result = CliRunner().invoke(
        add_cmd,
        ["https://example.com/guide", "--topic", "rag", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.add.result.v1"
    validation = validate_payload(payload, payload["schema"], strict=True)
    assert validation.schema_found is True
    assert validation.ok is True, validation.errors
    assert payload["sourceType"] == "web"
    assert payload["route"] == "crawl_ingest"
    assert payload["upstream"]["indexedChunks"] == 2
    assert khub.service.calls[0]["urls"] == ["https://example.com/guide"]
    assert khub.service.calls[0]["topic"] == "rag"


def test_add_paper_url_uses_single_source_import_manifest(monkeypatch, tmp_path):
    captured = {}

    def _fake_run_import_csv(**kwargs):  # noqa: ANN003
        print("resolver noise")
        captured.update(kwargs)
        return {
            "status": "ok",
            "counts": {"completed": 1, "failed": 0, "skipped": 0},
            "manifestPath": kwargs["manifest_path"],
            "warnings": [],
        }

    monkeypatch.setattr(add_module, "run_import_csv", _fake_run_import_csv)
    khub = _StubKhub(tmp_path)
    result = CliRunner().invoke(
        add_cmd,
        ["https://arxiv.org/abs/2401.00001", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    assert "resolver noise" not in result.output
    payload = json.loads(result.output)
    assert payload["route"] == "paper_import"
    assert captured["steps"] == ["register", "download", "embed"]
    csv_path = Path(captured["csv_path"])
    assert not csv_path.exists()
    assert payload["upstream"]["csvRetained"] is False
