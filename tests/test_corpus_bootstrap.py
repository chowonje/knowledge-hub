from __future__ import annotations

import hashlib
import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.application.corpus_artifacts import inspect_corpus_artifact
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group
from knowledge_hub.papers import corpus_bootstrap


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


class _FakeResponse:
    def __init__(self, content: bytes, *, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise corpus_bootstrap.requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):  # noqa: ARG002
        yield self.content


def _hash(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _write_manifest(path: Path, artifacts: list[dict]) -> Path:
    path.write_text(
        json.dumps({"schema": "knowledge-hub.corpus-manifest.v1", "artifacts": artifacts}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _artifact(*, content: bytes = b"%PDF-1.4 corpus") -> dict:
    return {
        "artifactId": "alexnet_krizhevsky_2012",
        "sourceIds": ["alexnet-2012"],
        "expectedFilename": "alexnet.pdf",
        "expectedSourceContentHash": _hash(content),
        "byteLength": len(content),
        "provenanceUrl": "https://example.test/alexnet.pdf",
        "corpusTier": "local_corpus",
    }


def test_corpus_bootstrap_dry_run_plans_missing_artifact_without_network(tmp_path: Path, monkeypatch):
    content = b"%PDF-1.4 corpus"
    manifest_path = _write_manifest(tmp_path / "manifest.json", [_artifact(content=content)])
    papers_dir = tmp_path / "papers"

    monkeypatch.setattr(
        corpus_bootstrap.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network should not be called")),
    )

    payload = corpus_bootstrap.bootstrap_corpus_artifacts(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
        artifact_ids=["alexnet_krizhevsky_2012"],
    )

    assert payload["status"] == "ok"
    assert payload["dryRun"] is True
    assert payload["counts"]["plannedDownload"] == 1
    item = payload["items"][0]
    assert item["status"] == "planned_download"
    assert item["targetPath"] == "papers_dir/alexnet.pdf"
    assert item["repairHints"][0]["command"] == [
        "khub",
        "paper",
        "repair-source",
        "--paper-id",
        "alexnet-2012",
        "--dry-run",
        "--json",
    ]
    assert not (papers_dir / "alexnet.pdf").exists()


def test_corpus_bootstrap_apply_requires_explicit_network(tmp_path: Path, monkeypatch):
    content = b"%PDF-1.4 corpus"
    manifest_path = _write_manifest(tmp_path / "manifest.json", [_artifact(content=content)])
    papers_dir = tmp_path / "papers"

    monkeypatch.setattr(
        corpus_bootstrap.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network should not be called")),
    )

    payload = corpus_bootstrap.bootstrap_corpus_artifacts(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
        source_ids=["alexnet-2012"],
        apply=True,
    )

    assert payload["status"] == "blocked"
    assert payload["items"][0]["status"] == "network_not_allowed"
    assert not (papers_dir / "alexnet.pdf").exists()


def test_corpus_bootstrap_requires_manifest_source_hash(tmp_path: Path, monkeypatch):
    artifact = _artifact()
    artifact.pop("expectedSourceContentHash")
    manifest_path = _write_manifest(tmp_path / "manifest.json", [artifact])
    papers_dir = tmp_path / "papers"
    monkeypatch.setattr(
        corpus_bootstrap.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network should not be called")),
    )

    payload = corpus_bootstrap.bootstrap_corpus_artifacts(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
        artifact_ids=["alexnet_krizhevsky_2012"],
        apply=True,
        allow_network=True,
    )

    assert payload["status"] == "blocked"
    assert payload["items"][0]["status"] == "missing_expected_hash"
    assert not (papers_dir / "alexnet.pdf").exists()


def test_corpus_bootstrap_downloads_and_verifies_hash_before_promotion(tmp_path: Path, monkeypatch):
    content = b"%PDF-1.4 verified corpus"
    manifest_path = _write_manifest(tmp_path / "manifest.json", [_artifact(content=content)])
    papers_dir = tmp_path / "papers"
    monkeypatch.setattr(
        corpus_bootstrap.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(content),
    )

    payload = corpus_bootstrap.bootstrap_corpus_artifacts(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
        artifact_ids=["alexnet_krizhevsky_2012"],
        apply=True,
        allow_network=True,
    )

    assert payload["status"] == "ok"
    assert payload["counts"]["downloaded"] == 1
    item = payload["items"][0]
    assert item["status"] == "downloaded"
    assert item["artifact"]["status"] == "ok"
    assert item["artifact"]["path"] == "papers_dir/alexnet.pdf"
    assert item["observedSourceContentHash"] == _hash(content)
    assert (papers_dir / "alexnet.pdf").read_bytes() == content
    assert str(papers_dir) not in json.dumps(payload)


def test_corpus_bootstrap_hash_mismatch_does_not_promote_download(tmp_path: Path, monkeypatch):
    manifest_path = _write_manifest(tmp_path / "manifest.json", [_artifact(content=b"expected")])
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    monkeypatch.setattr(
        corpus_bootstrap.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(b"different"),
    )

    payload = corpus_bootstrap.bootstrap_corpus_artifacts(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
        artifact_ids=["alexnet_krizhevsky_2012"],
        apply=True,
        allow_network=True,
    )

    assert payload["status"] == "blocked"
    item = payload["items"][0]
    assert item["status"] == "hash_mismatch"
    assert item["observedSourceContentHash"] == _hash(b"different")
    assert not (papers_dir / "alexnet.pdf").exists()
    assert list(papers_dir.iterdir()) == []


def test_corpus_bootstrap_existing_mismatch_is_not_replaced(tmp_path: Path, monkeypatch):
    expected = b"expected"
    manifest_path = _write_manifest(tmp_path / "manifest.json", [_artifact(content=expected)])
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    target = papers_dir / "alexnet.pdf"
    target.write_bytes(b"existing wrong content")
    monkeypatch.setattr(
        corpus_bootstrap.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network should not be called")),
    )

    payload = corpus_bootstrap.bootstrap_corpus_artifacts(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
        artifact_ids=["alexnet_krizhevsky_2012"],
        apply=True,
        allow_network=True,
    )

    assert payload["status"] == "blocked"
    assert payload["items"][0]["status"] == "hash_mismatch"
    assert target.read_bytes() == b"existing wrong content"


def test_corpus_bootstrap_skips_repo_fixture_without_papers_dir(tmp_path: Path):
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("fixture", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "artifactId": "fixture_artifact",
                "sourceIds": ["paper:fixture#0"],
                "fixturePath": "fixture.txt",
                "expectedSourceContentHash": _hash(fixture.read_bytes()),
                "corpusTier": "repo_fixture",
            }
        ],
    )

    payload = corpus_bootstrap.bootstrap_corpus_artifacts(
        config=_ConfigWithPapersDir(None),
        manifest_path=manifest_path,
        all_artifacts=True,
        apply=True,
        allow_network=True,
    )

    assert payload["status"] == "ok"
    assert payload["items"][0]["status"] == "skipped_repo_fixture"


def test_repo_fixture_tier_does_not_resolve_from_papers_dir(tmp_path: Path):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "fixture.txt").write_text("fixture", encoding="utf-8")
    entry = {
        "_manifestPath": str(tmp_path / "manifest.json"),
        "artifactId": "fixture_artifact",
        "sourceIds": ["paper:fixture#0"],
        "expectedFilename": "fixture.txt",
        "expectedSourceContentHash": _hash(b"fixture"),
        "corpusTier": "repo_fixture",
    }

    result = inspect_corpus_artifact(entry, config=_ConfigWithPapersDir(papers_dir))

    assert result["status"] == "missing_artifact"
    assert result["searchedPaths"] == ["repo_fixture/fixture.txt"]


def test_local_corpus_tier_does_not_resolve_from_repo_fixture_path(tmp_path: Path):
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("fixture", encoding="utf-8")
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    entry = {
        "_manifestPath": str(tmp_path / "manifest.json"),
        "artifactId": "local_artifact",
        "sourceIds": ["paper:local#0"],
        "expectedFilename": "fixture.txt",
        "expectedSourceContentHash": _hash(fixture.read_bytes()),
        "corpusTier": "local_corpus",
    }

    result = inspect_corpus_artifact(entry, config=_ConfigWithPapersDir(papers_dir))

    assert result["status"] == "missing_artifact"
    assert result["searchedPaths"] == ["papers_dir/fixture.txt"]


def test_local_corpus_first_candidate_hash_mismatch_does_not_fall_through(tmp_path: Path):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "first.pdf").write_bytes(b"wrong")
    second = papers_dir / "second.pdf"
    second.write_bytes(b"expected")
    entry = {
        "artifactId": "multi_candidate",
        "sourceIds": ["paper:multi#0"],
        "expectedFilenames": ["first.pdf", "second.pdf"],
        "expectedSourceContentHash": _hash(second.read_bytes()),
        "corpusTier": "local_corpus",
    }

    result = inspect_corpus_artifact(entry, config=_ConfigWithPapersDir(papers_dir))

    assert result["status"] == "hash_mismatch"
    assert result["path"] == "papers_dir/first.pdf"
    assert result["searchedPaths"] == ["papers_dir/first.pdf"]


def test_paper_corpus_bootstrap_cli_json_is_hidden_but_invokable(tmp_path: Path):
    content = b"%PDF-1.4 corpus"
    manifest_path = _write_manifest(tmp_path / "manifest.json", [_artifact(content=content)])
    papers_dir = tmp_path / "papers"

    result = CliRunner().invoke(
        paper_group,
        [
            "corpus-bootstrap",
            "--manifest",
            str(manifest_path),
            "--artifact-id",
            "alexnet_krizhevsky_2012",
            "--papers-dir",
            str(papers_dir),
            "--json",
        ],
        obj={"khub": _StubKhub(_ConfigWithPapersDir(None))},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["items"][0]["status"] == "planned_download"
    help_result = CliRunner().invoke(paper_group, ["--help"], obj={"khub": _StubKhub(_ConfigWithPapersDir(None))})
    assert "corpus-bootstrap" not in help_result.output
