from __future__ import annotations

import json
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.web import make_web_note_id
from knowledge_hub.interfaces.cli.commands.add import lanes as add_lanes
from knowledge_hub.interfaces.cli.commands.add import obsidian_stage as add_obsidian_stage
from knowledge_hub.interfaces.cli.commands.add_cmd import add_cmd, detect_add_route
from knowledge_hub.interfaces.cli.main import cli


class _StubConfig:
    def __init__(self, tmp_path: Path):
        self.sqlite_path = str(tmp_path / "knowledge.db")


class _StubWebIngestService:
    def __init__(self, db=None):
        self.calls: list[dict[str, object]] = []
        self.db = db

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

    def ingest_documents(self, docs, **kwargs):  # noqa: ANN003
        self.calls.append({"docs": docs, **dict(kwargs)})
        doc = docs[0]
        note_id = make_web_note_id(doc.url)
        if self.db is not None:
            self.db.notes[note_id] = {
                "id": note_id,
                "title": doc.title,
                "metadata": json.dumps(
                    {
                        "url": doc.url,
                        "source_content_hash": "hash-local-pdf",
                        "source_path": doc.source_metadata.get("source_path", ""),
                    }
                ),
            }

        class _Summary:
            def to_dict(self):
                return {
                    "status": "ok",
                    "requested": 1,
                    "crawled": 1,
                    "stored": 1,
                    "indexedChunks": 2 if kwargs.get("index") else 0,
                    "runId": "crawl_ingest_pdf_test",
                    "warnings": [],
                }

        return _Summary()


class _StubSqliteDB:
    def __init__(self):
        self.notes: dict[str, dict[str, object]] = {}
        self.pipeline_calls: list[dict[str, object]] = []

    def get_note(self, note_id):
        return self.notes.get(note_id)

    def create_crawl_pipeline_job(self, **kwargs):  # noqa: ANN003
        self.pipeline_calls.append({"fn": "create_crawl_pipeline_job", **dict(kwargs)})

    def upsert_crawl_pipeline_record(self, **kwargs):  # noqa: ANN003
        self.pipeline_calls.append({"fn": "upsert_crawl_pipeline_record", **dict(kwargs)})

    def update_crawl_pipeline_job(self, job_id, **kwargs):  # noqa: ANN003
        self.pipeline_calls.append({"fn": "update_crawl_pipeline_job", "job_id": job_id, **dict(kwargs)})


class _StubKhub:
    def __init__(self, tmp_path: Path, *, db=None):
        self.config = _StubConfig(tmp_path)
        self.db = db
        self.service = _StubWebIngestService(db=db)

    def web_ingest_service(self):
        return self.service

    def sqlite_db(self):
        return self.db


def test_detect_add_route_auto_selects_common_source_types():
    assert detect_add_route("https://youtu.be/abc123").kind == "youtube"
    assert detect_add_route("https://arxiv.org/abs/2401.00001").kind == "paper_url"
    assert detect_add_route("https://huggingface.co/papers/2401.00001").kind == "paper_url"
    assert detect_add_route("https://huggingface.co/openai/gpt-oss-20b").kind == "web"
    assert detect_add_route("https://huggingface.co/datasets/beans").kind == "web"
    assert detect_add_route("https://example.com/report.pdf").kind == "pdf"
    assert detect_add_route(str(Path("local-report.pdf"))).kind == "pdf"
    assert detect_add_route("https://example.com/guide").kind == "web"
    assert detect_add_route("retrieval augmented generation").kind == "paper_query"
    assert detect_add_route("https://example.com/report.pdf", "paper").kind == "paper_url"


def test_detect_add_route_rejects_invalid_explicit_source_types():
    with pytest.raises(click.BadParameter):
        detect_add_route("https://example.com/blog", "paper")
    with pytest.raises(click.BadParameter):
        detect_add_route("https://example.com/blog", "youtube")
    with pytest.raises(click.BadParameter):
        detect_add_route(str(Path("local-paper.pdf")), "paper")


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
    assert "--to-obsidian" in result.output
    assert "--build-memory" in result.output
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
    assert payload["sourceId"].startswith("web_")
    assert "contentHash" in payload
    assert payload["stored"] is True
    assert payload["indexed"] is True
    assert payload["obsidianStage"]["requested"] is False
    assert payload["nextActions"] == payload["nextCommands"]
    assert payload["upstream"]["indexedChunks"] == 2
    assert khub.service.calls[0]["urls"] == ["https://example.com/guide"]
    assert khub.service.calls[0]["topic"] == "rag"
    assert khub.service.calls[0]["writeback"] is False


def test_add_paper_url_uses_single_source_import_manifest(monkeypatch, tmp_path):
    captured = {}

    def _fake_run_import_csv(**kwargs):  # noqa: ANN003
        print("resolver noise")
        captured.update(kwargs)
        Path(kwargs["manifest_path"]).write_text(
            json.dumps({"source_url": "https://arxiv.org/abs/2401.00001"}),
            encoding="utf-8",
        )
        return {
            "status": "ok",
            "counts": {"completed": 1, "failed": 0, "skipped": 0},
            "manifestPath": kwargs["manifest_path"],
            "warnings": [],
            "items": [
                {
                    "title": "Paper",
                    "sourceUrl": "https://arxiv.org/abs/2401.00001",
                    "resolvedPaperId": "2401.00001",
                    "status": "completed",
                    "completedSteps": kwargs["steps"],
                    "executedSteps": kwargs["steps"],
                    "artifacts": {"paperId": "2401.00001"},
                }
            ],
        }

    monkeypatch.setattr(add_lanes, "run_import_csv", _fake_run_import_csv)
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
    assert payload["sourceId"] == "2401.00001"
    assert payload["stored"] is True
    assert captured["steps"] == ["register", "download", "embed"]
    csv_path = Path(captured["csv_path"])
    manifest_path = Path(captured["manifest_path"])
    assert not csv_path.exists()
    assert not manifest_path.exists()
    assert payload["upstream"]["csvRetained"] is False
    assert payload["upstream"]["manifestRetained"] is False
    assert str(captured["manifest_path"]) not in result.output
    assert payload["upstream"]["manifestPath"].startswith("<local>/")


def test_add_paper_url_build_memory_is_explicit(monkeypatch, tmp_path):
    captured = {}

    def _fake_run_import_csv(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {
            "status": "ok",
            "counts": {"completed": 1, "failed": 0, "skipped": 0},
            "manifestPath": kwargs["manifest_path"],
            "warnings": [],
            "items": [
                {
                    "title": "Paper",
                    "sourceUrl": "https://arxiv.org/abs/2401.00001",
                    "resolvedPaperId": "2401.00001",
                    "status": "completed",
                    "completedSteps": kwargs["steps"],
                    "executedSteps": kwargs["steps"],
                    "artifacts": {"paperId": "2401.00001"},
                }
            ],
        }

    monkeypatch.setattr(add_lanes, "run_import_csv", _fake_run_import_csv)
    khub = _StubKhub(tmp_path)
    result = CliRunner().invoke(
        add_cmd,
        ["https://arxiv.org/abs/2401.00001", "--build-memory", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    assert captured["steps"] == ["register", "download", "embed", "paper-memory", "document-memory"]


def test_add_to_obsidian_uses_stage_only_collect_payload(monkeypatch, tmp_path):
    captured = {}

    def _fake_collect_to_obsidian_payload(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {
            "schema": "knowledge-hub.crawl.collect.result.v1",
            "status": "completed",
            "requested": 1,
            "topic": "rag",
            "source": "web",
            "jobId": "crawl_job_test",
            "runId": "ko_note_test",
            "applyRequested": False,
            "onlyApproved": False,
            "crawl": {
                "status": "completed",
                "requested": 1,
                "processed": 1,
                "normalized": 1,
                "indexed": 3,
                "warnings": [],
            },
            "materialize": {
                "schema": "knowledge-hub.ko-note.generate.result.v1",
                "status": "completed",
                "runId": "ko_note_test",
                "sourceGenerated": 1,
                "conceptGenerated": 0,
                "warnings": [],
            },
            "apply": {},
            "warnings": [],
        }

    monkeypatch.setattr(add_lanes, "collect_to_obsidian_payload", _fake_collect_to_obsidian_payload)
    khub = _StubKhub(tmp_path)
    result = CliRunner().invoke(
        add_cmd,
        ["https://example.com/guide", "--topic", "rag", "--to-obsidian", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert captured["apply_notes"] is False
    assert captured["max_source_notes"] == 1
    assert captured["input_source"] == "web"
    assert payload["obsidianStage"]["requested"] is True
    assert payload["obsidianStage"]["staged"] == 1
    assert payload["obsidianStage"]["applied"] == 0
    assert payload["obsidianStage"]["applySkipped"] is True


def test_add_paper_query_returns_first_discovered_paper_summary(monkeypatch, tmp_path):
    from knowledge_hub.interfaces.cli.commands import discover_cmd

    def _fake_discover(**kwargs):  # noqa: ANN003
        assert kwargs["create_obsidian"] is False
        click.echo(
            json.dumps(
                {
                    "schema": "knowledge-hub.paper.discover.result.v1",
                    "status": "ok",
                    "topic": kwargs["topic"],
                    "ingested": [
                        {
                            "arxiv_id": "2401.00001",
                            "title": "Paper Query Result",
                            "summary": "summary",
                        }
                    ],
                    "results": [
                        {
                            "arxiv_id": "2401.00001",
                            "title": "Paper Query Result",
                            "success": True,
                            "steps": ["인덱싱(1)"],
                        }
                    ],
                    "warnings": [],
                }
            )
        )

    monkeypatch.setattr(discover_cmd, "discover", _fake_discover)
    khub = _StubKhub(tmp_path)
    result = CliRunner().invoke(
        add_cmd,
        ["retrieval augmented generation", "--type", "paper", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["route"] == "discover"
    assert payload["sourceId"] == "2401.00001"
    assert payload["canonicalUrl"] == "https://arxiv.org/abs/2401.00001"
    assert payload["title"] == "Paper Query Result"
    assert payload["stored"] is True
    assert payload["indexed"] is True


def test_add_local_pdf_outputs_document_lane_payload(monkeypatch, tmp_path):
    pdf_path = tmp_path / "local-report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    db = _StubSqliteDB()
    khub = _StubKhub(tmp_path, db=db)

    monkeypatch.setattr(add_lanes, "extract_pdf_text_excerpt", lambda *args, **kwargs: "Local PDF text")
    result = CliRunner().invoke(
        add_cmd,
        [str(pdf_path), "--topic", "pdf-topic", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    validation = validate_payload(payload, payload["schema"], strict=True)
    assert validation.ok is True, validation.errors
    assert payload["sourceType"] == "pdf"
    assert payload["route"] == "crawl_ingest"
    assert payload["routeReason"] == "pdf_source"
    assert payload["source"] == "<local>/local-report.pdf"
    assert payload["canonicalUrl"] == ""
    assert payload["canonicalPath"] == "<local>/local-report.pdf"
    assert payload["contentHash"] == "hash-local-pdf"
    assert payload["stored"] is True
    assert payload["indexed"] is True
    assert payload["upstream"]["localPath"] == "<local>/local-report.pdf"
    assert str(pdf_path.resolve()) not in result.output
    assert "excerpt only" in " ".join(payload["warnings"])


def test_add_local_pdf_to_obsidian_uses_stage_only_synthetic_job(monkeypatch, tmp_path):
    pdf_path = tmp_path / "local-report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    db = _StubSqliteDB()
    khub = _StubKhub(tmp_path, db=db)
    materializer_calls = []

    class _FakeMaterializer:
        def __init__(self, config):
            self.config = config

        def generate_for_job(self, **kwargs):  # noqa: ANN003
            materializer_calls.append(dict(kwargs))
            return {
                "schema": "knowledge-hub.ko-note.generate.result.v1",
                "status": "completed",
                "runId": "ko_note_pdf_test",
                "crawlJobId": kwargs["job_id"],
                "sourceCandidates": 1,
                "sourceGenerated": 1,
                "conceptCandidates": 0,
                "conceptGenerated": 0,
                "blocked": 0,
                "warnings": [],
            }

    monkeypatch.setattr(add_lanes, "extract_pdf_text_excerpt", lambda *args, **kwargs: "Local PDF text")
    monkeypatch.setattr(add_obsidian_stage, "ko_note_materializer", lambda config: _FakeMaterializer(config))
    result = CliRunner().invoke(
        add_cmd,
        [str(pdf_path), "--topic", "pdf-topic", "--to-obsidian", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["obsidianStage"]["requested"] is True
    assert payload["obsidianStage"]["runId"] == "ko_note_pdf_test"
    assert payload["obsidianStage"]["staged"] == 1
    assert payload["obsidianStage"]["applied"] == 0
    assert payload["obsidianStage"]["applySkipped"] is True
    assert materializer_calls[0]["max_source_notes"] == 1
    assert materializer_calls[0]["max_concept_notes"] == 0
    assert materializer_calls[0]["llm_mode"] == "fallback-only"
    assert materializer_calls[0]["api_fallback_on_timeout"] is False
    assert any(call["fn"] == "upsert_crawl_pipeline_record" and call["state"] == "indexed" for call in db.pipeline_calls)


def test_add_local_pdf_empty_extraction_returns_failed_packet(monkeypatch, tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    db = _StubSqliteDB()
    khub = _StubKhub(tmp_path, db=db)

    monkeypatch.setattr(add_lanes, "extract_pdf_text_excerpt", lambda *args, **kwargs: "")
    result = CliRunner().invoke(
        add_cmd,
        [str(pdf_path), "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["stored"] is False
    assert payload["indexed"] is False
    assert payload["canonicalUrl"] == ""
    assert payload["source"] == "<local>/empty.pdf"
    assert payload["upstream"]["failed"][0]["url"] == "<local>/empty.pdf"
    assert str(pdf_path.resolve()) not in result.output
    assert "empty content" in " ".join(payload["warnings"])
