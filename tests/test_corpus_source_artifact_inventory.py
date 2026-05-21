from __future__ import annotations

import hashlib
import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group
from knowledge_hub.papers.corpus_source_artifact_inventory import (
    CORPUS_SOURCE_ARTIFACT_INVENTORY_SCHEMA_ID,
    build_corpus_source_artifact_inventory,
)


class _ConfigWithPapersDir:
    def __init__(self, papers_dir: Path | None):
        self.papers_dir = str(papers_dir) if papers_dir is not None else ""

    def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
        if tuple(args) == ("storage", "papers_dir"):
            return self.papers_dir
        return default


class _StubKhub:
    def __init__(self, config):
        self.config = config


def _hash(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _write_manifest(path: Path, artifacts: list[dict]) -> Path:
    path.write_text(
        json.dumps({"schema": "knowledge-hub.corpus-manifest.v1", "artifacts": artifacts}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def test_corpus_source_artifact_inventory_reports_registered_and_unregistered_without_absolute_paths(
    tmp_path: Path,
) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    registered = b"%PDF-1.4 registered"
    unregistered = b"%PDF-1.4 unregistered"
    (papers_dir / "registered.pdf").write_bytes(registered)
    (papers_dir / "unregistered.pdf").write_bytes(unregistered)
    (papers_dir / "parsed").mkdir()
    (papers_dir / "parsed" / "ignored.pdf").write_bytes(b"ignored")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "artifactId": "paper_registered",
                "sourceIds": ["registered"],
                "expectedFilename": "registered.pdf",
                "expectedSourceContentHash": _hash(registered),
                "byteLength": len(registered),
                "provenanceUrl": "https://example.test/registered.pdf",
                "corpusTier": "local_corpus",
            }
        ],
    )

    payload = build_corpus_source_artifact_inventory(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
    )

    assert payload["status"] == "ok"
    assert validate_payload(payload, CORPUS_SOURCE_ARTIFACT_INVENTORY_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["inventoryRows"] == 2
    assert payload["counts"]["alreadyRegisteredRows"] == 1
    assert payload["counts"]["unregisteredAvailableRows"] == 1
    assert str(papers_dir) not in json.dumps(payload)
    statuses = {item["filename"]: item["registrationStatus"] for item in payload["items"]}
    assert statuses["registered.pdf"] == "already_registered"
    assert statuses["unregistered.pdf"] == "unregistered_available"
    assert all("/Users/" not in item["corpusLocationRef"] for item in payload["items"])


def test_corpus_source_artifact_inventory_flags_hash_mismatch_registered(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "paper-a.pdf").write_bytes(b"different-bytes")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "artifactId": "paper_a",
                "sourceIds": ["paper-a"],
                "expectedFilename": "paper-a.pdf",
                "expectedSourceContentHash": "sha256:" + "0" * 64,
                "byteLength": 16,
                "corpusTier": "local_corpus",
            }
        ],
    )

    payload = build_corpus_source_artifact_inventory(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
    )

    assert payload["counts"]["hashMismatchRegisteredRows"] == 1
    assert payload["counts"]["manifestHashMismatchRows"] == 1
    item = payload["items"][0]
    assert item["registrationStatus"] == "hash_mismatch_registered"
    assert item["manifestRegistered"] is True


def test_corpus_source_artifact_inventory_scans_recovered_source_roots(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    recovered_arxiv = papers_dir / "recovered_sources" / "arxiv"
    recovered_arxiv.mkdir(parents=True)
    content = b"%PDF-1.4 recovered arxiv source"
    (recovered_arxiv / "2501.00001.pdf").write_bytes(content)
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "artifactId": "paper_2501_00001",
                "sourceIds": ["2501.00001"],
                "expectedFilename": "2501.00001.pdf",
                "expectedSourceContentHash": _hash(content),
                "byteLength": len(content),
                "provenanceUrl": "https://arxiv.org/pdf/2501.00001",
                "corpusTier": "local_corpus",
            }
        ],
    )

    payload = build_corpus_source_artifact_inventory(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
    )

    assert payload["status"] == "ok"
    assert payload["counts"]["inventoryRows"] == 1
    assert payload["counts"]["alreadyRegisteredRows"] == 1
    assert "recovered_sources" not in payload["checks"]["derivativeSubdirsExcluded"]
    assert "papers_dir/recovered_sources/arxiv" in payload["checks"]["scanRoots"]
    item = payload["items"][0]
    assert item["sourceId"] == "2501.00001"
    assert item["registrationStatus"] == "already_registered"
    assert item["corpusLocationRef"] == "papers_dir/recovered_sources/arxiv/2501.00001.pdf"


def test_paper_corpus_source_artifact_inventory_cli_reports_json(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    content = b"%PDF-1.4 paper-a"
    (papers_dir / "paper-a.pdf").write_bytes(content)
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "artifactId": "paper_a",
                "sourceIds": ["paper-a"],
                "expectedFilename": "paper-a.pdf",
                "expectedSourceContentHash": _hash(content),
                "byteLength": len(content),
                "corpusTier": "local_corpus",
            }
        ],
    )

    result = CliRunner().invoke(
        paper_group,
        [
            "corpus-source-artifact-inventory",
            "--manifest",
            str(manifest_path),
            "--papers-dir",
            str(papers_dir),
            "--json",
        ],
        obj={"khub": _StubKhub(_ConfigWithPapersDir(papers_dir))},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == CORPUS_SOURCE_ARTIFACT_INVENTORY_SCHEMA_ID
    assert payload["counts"]["alreadyRegisteredRows"] == 1
