from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sync_worktree_slice.py"


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(repo_root: Path) -> None:
    _run(["git", "init"], cwd=repo_root)
    _run(["git", "config", "user.name", "Test User"], cwd=repo_root)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo_root)


def _write(repo_root: Path, rel_path: str, content: str) -> None:
    path = repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_identical_repo(repo_root: Path) -> None:
    _init_repo(repo_root)
    _write(repo_root, "README.md", "base\n")
    _write(repo_root, "docs/old.md", "old\n")
    _run(["git", "add", "."], cwd=repo_root)
    _run(["git", "commit", "-m", "initial"], cwd=repo_root)


def test_sync_worktree_slice_copies_untracked_and_deleted_paths(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    _seed_identical_repo(source_root)
    _seed_identical_repo(target_root)

    _write(source_root, "README.md", "changed\n")
    _write(source_root, "docs/new.md", "new\n")
    (source_root / "docs/old.md").unlink()

    manifest = tmp_path / "core-loop.pathspec"
    manifest.write_text("README.md\ndocs/new.md\ndocs/old.md\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--source-repo-root",
            str(source_root),
            "--target-repo-root",
            str(target_root),
            "--pathspec-file",
            str(manifest),
            "--stage",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["status"] == "ok"
    assert payload["changedPathCount"] == 3
    assert payload["copiedPathCount"] == 2
    assert payload["deletedPathCount"] == 1
    assert (target_root / "README.md").read_text(encoding="utf-8") == "changed\n"
    assert (target_root / "docs/new.md").read_text(encoding="utf-8") == "new\n"
    assert not (target_root / "docs/old.md").exists()

    staged = _run(["git", "diff", "--cached", "--name-status"], cwd=target_root).stdout.splitlines()
    assert "M\tREADME.md" in staged
    assert "A\tdocs/new.md" in staged
    assert "D\tdocs/old.md" in staged


def test_sync_worktree_slice_rejects_pathspec_escape(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    _seed_identical_repo(source_root)
    _seed_identical_repo(target_root)

    manifest = tmp_path / "invalid.pathspec"
    manifest.write_text("../outside.txt\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--source-repo-root",
            str(source_root),
            "--target-repo-root",
            str(target_root),
            "--pathspec-file",
            str(manifest),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "must stay inside the repo root" in result.stderr
