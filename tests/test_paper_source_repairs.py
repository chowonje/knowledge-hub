from __future__ import annotations

import hashlib
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


class _ConfigWithPapersDir:
    def __init__(self, papers_dir: Path):
        self.papers_dir = str(papers_dir)

    def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
        if tuple(args) == ("storage", "papers_dir"):
            return self.papers_dir
        return default


class _MemoryBuilder:
    def __init__(self, calls: list[tuple[str, str, bool]]):
        self.calls = calls

    def build_and_store(self, *, paper_id: str, materialize_card: bool = True):
        self.calls.append(("paper-memory", paper_id, bool(materialize_card)))
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


def _manifest_for_pdf(source_pdf: Path, *, expected_hash: str | None = None) -> dict:
    if expected_hash is None:
        expected_hash = "sha256:" + hashlib.sha256(source_pdf.read_bytes()).hexdigest()
    return {
        "schema": "knowledge-hub.corpus-manifest.v1",
        "artifacts": [
            {
                "artifactId": "alexnet_krizhevsky_2012",
                "sourceIds": ["alexnet-2012"],
                "expectedFilename": source_pdf.name,
                "expectedSourceContentHash": expected_hash,
                "byteLength": source_pdf.stat().st_size if source_pdf.exists() else 0,
                "corpusTier": "local_corpus",
            }
        ],
    }


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
    calls: list[tuple[str, str, bool]] = []

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
    assert ("paper-memory", "Batch_Normalization_c72acd36", False) in calls
    assert ("document-memory", "Batch_Normalization_c72acd36", "opendataloader") in _DocumentMemoryBuilder.calls
    assert ("paper-card-v2", "Batch_Normalization_c72acd36") in _PaperCardBuilder.calls


def test_repair_paper_sources_plans_alexnet_configured_source_attach(tmp_path: Path, monkeypatch):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    source_pdf = papers_dir / "4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 alexnet")
    sqlite_db = _FakeSQLite()
    sqlite_db.rows["alexnet-2012"] = {
        "arxiv_id": "alexnet-2012",
        "title": "ImageNet Classification with Deep Convolutional Neural Networks",
        "pdf_path": "",
        "text_path": "",
    }
    monkeypatch.setattr(repair_module, "load_corpus_manifest", lambda: _manifest_for_pdf(source_pdf))

    payload = repair_paper_sources(
        sqlite_db=sqlite_db,
        config=_ConfigWithPapersDir(papers_dir),
        paper_ids=["alexnet-2012"],
        document_memory_parser="raw",
        dry_run=True,
        rebuild=True,
    )

    assert payload["status"] == "ok"
    item = payload["items"][0]
    assert item["paperId"] == "alexnet-2012"
    assert item["action"] == "attach_manifest_source_artifact"
    assert item["decisionStatus"] == "resolved"
    assert item["repairStatus"] == "planned"
    assert item["newPdfPath"] == str(source_pdf)
    assert item["newTextPath"] == ""
    assert item["artifact"]["artifactId"] == "alexnet_krizhevsky_2012"
    assert sqlite_db.get_paper("alexnet-2012")["pdf_path"] == ""


def test_repair_paper_sources_attaches_alexnet_source_and_rebuilds(tmp_path: Path, monkeypatch):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    source_pdf = papers_dir / "4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 alexnet")
    sqlite_db = _FakeSQLite()
    sqlite_db.rows["alexnet-2012"] = {
        "arxiv_id": "alexnet-2012",
        "title": "ImageNet Classification with Deep Convolutional Neural Networks",
        "pdf_path": "",
        "text_path": "",
    }
    monkeypatch.setattr(repair_module, "load_corpus_manifest", lambda: _manifest_for_pdf(source_pdf))
    calls: list[tuple[str, str, bool]] = []
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
        config=_ConfigWithPapersDir(papers_dir),
        paper_ids=["alexnet-2012"],
        document_memory_parser="raw",
        allow_external=False,
        llm_mode="fallback-only",
        dry_run=False,
        rebuild=True,
    )

    assert payload["status"] == "ok"
    assert payload["counts"]["ok"] == 1
    item = payload["items"][0]
    assert item["repairStatus"] == "ok"
    assert item["sourceChanged"] is True
    assert item["artifact"]["status"] == "ok"
    assert item["artifact"]["expectedSourceContentHash"].startswith("sha256:")
    updated = sqlite_db.get_paper("alexnet-2012")
    assert updated["pdf_path"] == str(source_pdf)
    assert updated["text_path"] == ""
    assert ("paper-memory", "alexnet-2012", False) in calls
    assert ("document-memory", "alexnet-2012", "raw") in _DocumentMemoryBuilder.calls
    assert ("paper-card-v2", "alexnet-2012") in _PaperCardBuilder.calls


def test_repair_paper_sources_reports_missing_manifest_artifact_without_write(tmp_path: Path, monkeypatch):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    missing_pdf = papers_dir / "4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf"
    sqlite_db = _FakeSQLite()
    sqlite_db.rows["alexnet-2012"] = {
        "arxiv_id": "alexnet-2012",
        "title": "ImageNet Classification with Deep Convolutional Neural Networks",
        "pdf_path": "",
        "text_path": "",
    }
    monkeypatch.setattr(repair_module, "load_corpus_manifest", lambda: _manifest_for_pdf(missing_pdf, expected_hash="sha256:" + "0" * 64))

    payload = repair_paper_sources(
        sqlite_db=sqlite_db,
        config=_ConfigWithPapersDir(papers_dir),
        paper_ids=["alexnet-2012"],
        dry_run=False,
        rebuild=True,
    )

    item = payload["items"][0]
    assert payload["status"] == "blocked"
    assert payload["counts"]["blocked"] == 1
    assert item["repairStatus"] == "missing_artifact"
    assert item["sourceChanged"] is False
    assert item["rebuildApplied"] is False
    assert item["artifact"]["artifactId"] == "alexnet_krizhevsky_2012"
    assert item["artifact"]["expectedSourceContentHash"] == "sha256:" + "0" * 64
    assert item["artifact"]["searchedPaths"] == [f"papers_dir/{missing_pdf.name}"]
    assert str(papers_dir) not in json.dumps(item["artifact"])
    assert sqlite_db.get_paper("alexnet-2012")["pdf_path"] == ""
    assert sqlite_db.upserts == []


def test_repair_paper_sources_reports_manifest_hash_mismatch_without_write(tmp_path: Path, monkeypatch):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    source_pdf = papers_dir / "4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 alexnet different")
    sqlite_db = _FakeSQLite()
    sqlite_db.rows["alexnet-2012"] = {
        "arxiv_id": "alexnet-2012",
        "title": "ImageNet Classification with Deep Convolutional Neural Networks",
        "pdf_path": "",
        "text_path": "",
    }
    monkeypatch.setattr(repair_module, "load_corpus_manifest", lambda: _manifest_for_pdf(source_pdf, expected_hash="sha256:" + "1" * 64))

    payload = repair_paper_sources(
        sqlite_db=sqlite_db,
        config=_ConfigWithPapersDir(papers_dir),
        paper_ids=["alexnet-2012"],
        dry_run=False,
        rebuild=True,
    )

    item = payload["items"][0]
    assert payload["status"] == "blocked"
    assert item["repairStatus"] == "hash_mismatch"
    assert item["sourceChanged"] is False
    assert item["rebuildApplied"] is False
    assert item["artifact"]["expectedSourceContentHash"] == "sha256:" + "1" * 64
    assert item["artifact"]["observedSourceContentHash"].startswith("sha256:")
    assert sqlite_db.get_paper("alexnet-2012")["pdf_path"] == ""
    assert sqlite_db.upserts == []


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


def test_paper_repair_source_cli_reports_alexnet_configured_source_plan(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    papers_dir = Path(config.papers_dir)
    papers_dir.mkdir(parents=True)
    source_pdf = papers_dir / "4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 alexnet")
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_paper(
        {
            "arxiv_id": "alexnet-2012",
            "title": "ImageNet Classification with Deep Convolutional Neural Networks",
            "authors": "A",
            "year": 2012,
            "field": "AI",
            "importance": 5,
            "notes": "AlexNet은 ImageNet CNN 논문이다.",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.close()
    monkeypatch.setattr(repair_module, "load_corpus_manifest", lambda: _manifest_for_pdf(source_pdf))

    result = CliRunner().invoke(
        paper_group,
        ["repair-source", "--paper-id", "alexnet-2012", "--dry-run", "--json"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    item = payload["items"][0]
    assert item["action"] == "attach_manifest_source_artifact"
    assert item["decisionStatus"] == "resolved"
    assert item["repairStatus"] == "planned"
    assert item["newPdfPath"] == str(source_pdf)
    assert item["artifact"]["artifactId"] == "alexnet_krizhevsky_2012"
    db = SQLiteDatabase(config.sqlite_path)
    assert db.get_paper("alexnet-2012")["pdf_path"] == ""


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
