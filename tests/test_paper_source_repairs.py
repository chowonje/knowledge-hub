from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

import knowledge_hub.application.paper_source_repairs as repair_module
from knowledge_hub.application.paper_source_repairs import (
    queue_paper_source_repairs,
    repair_paper_sources,
    run_source_cleanup_queue,
)
from knowledge_hub.core.config import Config
from knowledge_hub.core.database import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group


class _FakeSQLite:
    def __init__(self):
        self.rows = {
            "Batch_Normalization_c72acd36": {
                "arxiv_id": "Batch_Normalization_c72acd36",
                "title": "Batch Normalization",
                "pdf_path": "/tmp/bad-bn.pdf",
                "text_path": "",
            },
            "1502.03167": {
                "arxiv_id": "1502.03167",
                "title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
                "pdf_path": "/tmp/good-bn.pdf",
                "text_path": "/tmp/good-bn.txt",
            },
        }
        self.upserts: list[dict[str, str]] = []

    def get_paper(self, paper_id: str):
        return dict(self.rows.get(paper_id) or {}) or None

    def upsert_paper(self, payload):
        self.rows[str(payload["arxiv_id"])] = dict(payload)
        self.upserts.append(dict(payload))


class _StubConfig:
    def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
        return default


class _MemoryBuilder:
    def __init__(self, calls: list[tuple[str, str]]):
        self.calls = calls

    def build_and_store(self, *, paper_id: str):
        self.calls.append(("paper-memory", paper_id))
        return {"paper_id": paper_id, "quality_flag": "ok"}


class _DocumentMemoryBuilder:
    calls: list[tuple[str, str, str]] = []

    def __init__(self, sqlite_db, config=None):  # noqa: ANN001
        self.sqlite_db = sqlite_db
        self.config = config

    def build_and_store_paper(self, *, paper_id: str, paper_parser: str = "raw"):
        self.calls.append(("document-memory", paper_id, paper_parser))
        return [{"document_id": f"paper:{paper_id}"}]


class _PaperCardBuilder:
    calls: list[tuple[str, str]] = []

    def __init__(self, sqlite_db):  # noqa: ANN001
        self.sqlite_db = sqlite_db

    def build_and_store(self, *, paper_id: str):
        self.calls.append(("paper-card-v2", paper_id))
        return {"paper_id": paper_id, "quality_flag": "ok"}


class _StubKhub:
    def __init__(self, config: Config):
        self.config = config

    def sqlite_db(self):
        return SQLiteDatabase(self.config.sqlite_path)


def _config(tmp_path: Path) -> Config:
    config = Config()
    config.set_nested("storage", "sqlite", str(tmp_path / "knowledge.db"))
    config.set_nested("storage", "papers_dir", str(tmp_path / "papers"))
    config.set_nested("storage", "vector_db", str(tmp_path / "vector_db"))
    return config


def test_run_source_cleanup_queue_applies_and_writes_artifacts(tmp_path: Path):
    sqlite_db = _FakeSQLite()
    payload = run_source_cleanup_queue(
        sqlite_db=sqlite_db,
        queue_rows=[
            {
                "paperId": "Batch_Normalization_c72acd36",
                "title": "Batch Normalization",
                "oldPdfPath": "/tmp/bad-bn.pdf",
                "oldTextPath": "",
                "recommendedParser": "raw",
            }
        ],
        artifact_dir=tmp_path,
        apply=True,
    )

    assert payload["status"] == "ok"
    assert payload["cleanup"]["total"] == 1
    assert payload["cleanup"]["applySummary"]["applied"] == 1
    assert Path(payload["artifactPaths"]["summary"]).exists()
    updated = sqlite_db.get_paper("Batch_Normalization_c72acd36")
    assert updated["pdf_path"] == "/tmp/good-bn.pdf"
    assert updated["text_path"] == "/tmp/good-bn.txt"


def test_repair_paper_sources_relinks_and_rebuilds(monkeypatch):
    sqlite_db = _FakeSQLite()
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        repair_module,
        "build_paper_memory_builder",
        lambda sqlite_db, **kwargs: _MemoryBuilder(calls),
    )
    _DocumentMemoryBuilder.calls = []
    _PaperCardBuilder.calls = []
    monkeypatch.setattr(repair_module, "DocumentMemoryBuilder", _DocumentMemoryBuilder)
    monkeypatch.setattr(repair_module, "PaperCardV2Builder", _PaperCardBuilder)

    payload = repair_paper_sources(
        sqlite_db=sqlite_db,
        config=_StubConfig(),
        paper_ids=["Batch_Normalization_c72acd36"],
        document_memory_parser="opendataloader",
        dry_run=False,
        rebuild=True,
    )

    assert payload["status"] == "ok"
    assert payload["counts"]["ok"] == 1
    item = payload["items"][0]
    assert item["repairStatus"] == "ok"
    assert item["sourceChanged"] is True
    assert item["artifactRefresh"]["paperMemory"]["status"] == "ok"
    assert item["artifactRefresh"]["documentMemory"]["count"] == 1
    assert item["artifactRefresh"]["paperCardV2"]["status"] == "ok"
    updated = sqlite_db.get_paper("Batch_Normalization_c72acd36")
    assert updated["pdf_path"] == "/tmp/good-bn.pdf"
    assert updated["text_path"] == "/tmp/good-bn.txt"
    assert ("paper-memory", "Batch_Normalization_c72acd36") in calls
    assert ("document-memory", "Batch_Normalization_c72acd36", "opendataloader") in _DocumentMemoryBuilder.calls
    assert ("paper-card-v2", "Batch_Normalization_c72acd36") in _PaperCardBuilder.calls


def test_paper_repair_source_cli_reports_dry_run_plan(tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_paper(
        {
            "arxiv_id": "Batch_Normalization_c72acd36",
            "title": "Batch Normalization",
            "authors": "A",
            "year": 2015,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": "/tmp/bad-bn.pdf",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_paper(
        {
            "arxiv_id": "1502.03167",
            "title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
            "authors": "A",
            "year": 2015,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": "/tmp/good-bn.pdf",
            "text_path": "/tmp/good-bn.txt",
            "translated_path": "",
        }
    )
    db.close()

    result = CliRunner().invoke(
        paper_group,
        ["repair-source", "--paper-id", "Batch_Normalization_c72acd36", "--dry-run", "--json"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["dryRun"] is True
    assert payload["counts"]["ok"] == 0
    assert payload["counts"]["blocked"] == 0
    item = payload["items"][0]
    assert item["action"] == "relink_to_canonical"
    assert item["decisionStatus"] == "resolved"
    assert item["repairStatus"] == "planned"


def test_queue_paper_source_repairs_creates_paper_ops_action(tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_paper(
        {
            "arxiv_id": "Batch_Normalization_c72acd36",
            "title": "Batch Normalization",
            "authors": "A",
            "year": 2015,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": "/tmp/bad-bn.pdf",
            "text_path": "",
            "translated_path": "",
        }
    )

    payload = queue_paper_source_repairs(
        sqlite_db=db,
        paper_ids=["Batch_Normalization_c72acd36"],
        document_memory_parser="raw",
        rebuild=True,
    )

    assert payload["status"] == "ok"
    assert payload["counts"]["created"] == 1
    item = payload["items"][0]
    assert item["paperId"] == "Batch_Normalization_c72acd36"
    action = dict(item["action"])
    assert action["scope"] == "paper"
    assert action["action_type"] == "repair_paper_source"
    assert action["target_key"] == "paper:Batch_Normalization_c72acd36"
    assert action["args_json"] == ["paper", "repair-source", "--paper-id", "Batch_Normalization_c72acd36"]


def test_paper_repair_source_queue_cli_writes_ops_actions(tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_paper(
        {
            "arxiv_id": "Batch_Normalization_c72acd36",
            "title": "Batch Normalization",
            "authors": "A",
            "year": 2015,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": "/tmp/bad-bn.pdf",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.close()

    result = CliRunner().invoke(
        paper_group,
        ["repair-source-queue", "--paper-id", "Batch_Normalization_c72acd36", "--json"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["counts"]["created"] == 1
    assert payload["items"][0]["operation"] == "created"
