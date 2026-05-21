from __future__ import annotations

import hashlib
import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group
from knowledge_hub.papers.corpus_manifest_validation import (
    CORPUS_MANIFEST_VALIDATION_SCHEMA_ID,
    validate_corpus_manifest,
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


def _artifact(
    *,
    artifact_id: str = "paper_a",
    source_id: str = "paper-a",
    filename: str = "paper-a.pdf",
    content: bytes = b"%PDF-1.4 paper-a",
    expected_hash: str | None = None,
) -> dict:
    return {
        "artifactId": artifact_id,
        "sourceIds": [source_id],
        "expectedFilename": filename,
        "expectedSourceContentHash": expected_hash or _hash(content),
        "byteLength": len(content),
        "provenanceUrl": "https://example.test/paper-a.pdf",
        "corpusTier": "local_corpus",
    }


def test_corpus_manifest_validation_reports_source_and_parsed_available_without_paths(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    content = b"%PDF-1.4 paper-a"
    (papers_dir / "paper-a.pdf").write_bytes(content)
    parsed_dir = papers_dir / "parsed" / "paper-a"
    parsed_dir.mkdir(parents=True)
    (parsed_dir / "manifest.json").write_text("{}", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path / "manifest.json", [_artifact(content=content)])

    payload = validate_corpus_manifest(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
    )

    assert payload["status"] == "ok"
    assert validate_payload(payload, CORPUS_MANIFEST_VALIDATION_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["sourceAvailableRows"] == 1
    assert payload["counts"]["parsedAvailableRows"] == 1
    item = payload["items"][0]
    assert item["sourceArtifactStatus"] == "available"
    assert item["parsedArtifactStatus"] == "available"
    assert str(papers_dir) not in json.dumps(payload)
    assert "_resolvedPath" not in json.dumps(payload)


def test_corpus_manifest_validation_finds_localpdf_subdir_artifact(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    localpdf_dir = papers_dir / "localpdf_pdfs"
    localpdf_dir.mkdir(parents=True)
    content = b"%PDF-1.4 localpdf"
    (localpdf_dir / "localpdf-paper.pdf").write_bytes(content)
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            _artifact(
                artifact_id="paper_localpdf",
                source_id="localpdf-paper",
                filename="localpdf-paper.pdf",
                content=content,
            )
        ],
    )

    payload = validate_corpus_manifest(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
        check_parsed=False,
    )

    assert payload["status"] == "ok"
    assert payload["counts"]["sourceAvailableRows"] == 1
    item = payload["items"][0]
    assert item["sourceArtifactStatus"] == "available"
    assert item["artifact"]["path"] == "papers_dir/localpdf_pdfs/localpdf-paper.pdf"
    assert str(papers_dir) not in json.dumps(payload)


def test_corpus_manifest_validation_blocks_hash_mismatch_without_green_source(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "paper-a.pdf").write_bytes(b"different")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [_artifact(expected_hash="sha256:" + "0" * 64)],
    )

    payload = validate_corpus_manifest(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["hashMismatchRows"] == 1
    assert payload["counts"]["sourceAvailableRows"] == 0
    assert payload["items"][0]["sourceArtifactStatus"] == "hash_mismatch"
    assert "source_content_hash_mismatch" in payload["items"][0]["blockers"]


def test_corpus_manifest_validation_separates_source_missing_metadata_only_and_parsed_missing(
    tmp_path: Path,
) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    available = b"%PDF-1.4 available"
    (papers_dir / "available.pdf").write_bytes(available)
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            _artifact(
                artifact_id="available",
                source_id="available",
                filename="available.pdf",
                content=available,
            ),
            _artifact(
                artifact_id="missing",
                source_id="missing",
                filename="missing.pdf",
                expected_hash="sha256:" + "1" * 64,
            ),
            {
                "artifactId": "metadata_only",
                "sourceIds": ["metadata-only"],
                "corpusTier": "local_corpus",
                "provenanceUrl": "https://example.test/metadata-only",
            },
        ],
    )

    payload = validate_corpus_manifest(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["sourceAvailableRows"] == 1
    assert payload["counts"]["sourceMissingRows"] == 1
    assert payload["counts"]["metadataOnlyRows"] == 1
    assert payload["counts"]["parsedMissingRows"] == 1
    assert payload["counts"]["parsedSourceUnavailableRows"] == 2
    statuses = {item["artifactId"]: item["sourceArtifactStatus"] for item in payload["items"]}
    parsed = {item["artifactId"]: item["parsedArtifactStatus"] for item in payload["items"]}
    assert statuses["missing"] == "source_missing"
    assert statuses["metadata_only"] == "metadata_only"
    assert parsed["available"] == "parsed_missing"
    assert parsed["metadata_only"] == "source_unavailable"


def test_paper_corpus_manifest_validate_cli_reports_json(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    content = b"%PDF-1.4 paper-a"
    (papers_dir / "paper-a.pdf").write_bytes(content)
    manifest_path = _write_manifest(tmp_path / "manifest.json", [_artifact(content=content)])

    result = CliRunner().invoke(
        paper_group,
        [
            "corpus-manifest-validate",
            "--manifest",
            str(manifest_path),
            "--papers-dir",
            str(papers_dir),
            "--no-check-parsed",
            "--json",
        ],
        obj={"khub": _StubKhub(_ConfigWithPapersDir(papers_dir))},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == CORPUS_MANIFEST_VALIDATION_SCHEMA_ID
    assert payload["status"] == "ok"
    assert payload["counts"]["sourceAvailableRows"] == 1
    assert payload["counts"]["parsedNotCheckedRows"] == 1
