from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "report_checkpoint_split.py"
)


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


def test_checkpoint_split_report_classifies_first_match_and_excludes(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    checkpoints_dir = repo_root / "ops" / "checkpoints"
    export_dir = tmp_path / "exported"
    repo_root.mkdir()
    checkpoints_dir.mkdir(parents=True)
    _init_repo(repo_root)

    _write(repo_root, "package.json", "{}\n")
    _write(
        repo_root,
        "knowledge_hub/interfaces/cli/commands/paper_cmd.py",
        "print('paper')\n",
    )
    _write(repo_root, "knowledge_hub/cli/main.py", "print('cli')\n")
    _write(
        repo_root,
        "ops/checkpoints/checkpoints.json",
        json.dumps(
            {
                "version": 1,
                "buckets": [
                    {
                        "order": 1,
                        "slug": "contract",
                        "title": "contract",
                        "patterns": [
                            "package.json",
                            "knowledge_hub/interfaces/**",
                        ],
                        "exclude_patterns": [
                            "knowledge_hub/interfaces/cli/commands/paper*",
                        ],
                    },
                    {
                        "order": 2,
                        "slug": "paper",
                        "title": "paper",
                        "patterns": [
                            "knowledge_hub/interfaces/cli/commands/paper*",
                        ],
                    },
                    {
                        "order": 3,
                        "slug": "cli",
                        "title": "cli",
                        "patterns": [
                            "knowledge_hub/cli/**",
                        ],
                    },
                ],
            },
            indent=2,
        )
        + "\n",
    )
    _run(["git", "add", "."], cwd=repo_root)
    _run(["git", "commit", "-m", "initial"], cwd=repo_root)

    _write(repo_root, "package.json", '{"name": "demo"}\n')
    _write(
        repo_root,
        "knowledge_hub/interfaces/cli/commands/paper_cmd.py",
        "print('paper changed')\n",
    )
    _write(repo_root, "knowledge_hub/cli/main.py", "print('cli changed')\n")
    _write(repo_root, "docs/note.md", "carryover\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--repo-root",
            str(repo_root),
            "--write-pathspec-dir",
            str(export_dir),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)

    buckets = {bucket["name"]: bucket for bucket in report["buckets"]}

    assert buckets["01-contract"]["changedCount"] == 1
    assert [item["path"] for item in buckets["01-contract"]["paths"]] == ["package.json"]

    assert buckets["02-paper"]["changedCount"] == 1
    assert [item["path"] for item in buckets["02-paper"]["paths"]] == [
        "knowledge_hub/interfaces/cli/commands/paper_cmd.py"
    ]

    assert buckets["03-cli"]["changedCount"] == 1
    assert [item["path"] for item in buckets["03-cli"]["paths"]] == [
        "knowledge_hub/cli/main.py"
    ]

    assert report["unmatchedChangedPaths"] == 1
    assert report["unmatched"][0]["path"] == "docs/note.md"

    contract_pathspec = (export_dir / "01-contract.pathspec").read_text(encoding="utf-8")
    paper_pathspec = (export_dir / "02-paper.pathspec").read_text(encoding="utf-8")
    cli_pathspec = (export_dir / "03-cli.pathspec").read_text(encoding="utf-8")
    unmatched_pathspec = (export_dir / "unmatched.pathspec").read_text(encoding="utf-8")

    assert "package.json" in contract_pathspec
    assert "knowledge_hub/interfaces/cli/commands/paper_cmd.py" in paper_pathspec
    assert "knowledge_hub/cli/main.py" in cli_pathspec
    assert "docs/note.md" in unmatched_pathspec


def test_checkpoint_split_report_can_fail_on_unmatched(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    checkpoints_dir = repo_root / "ops" / "checkpoints"
    repo_root.mkdir()
    checkpoints_dir.mkdir(parents=True)
    _init_repo(repo_root)

    _write(repo_root, "tracked.txt", "base\n")
    _write(
        repo_root,
        "ops/checkpoints/checkpoints.json",
        json.dumps(
            {
                "version": 1,
                "buckets": [
                    {
                        "order": 1,
                        "slug": "empty",
                        "title": "empty",
                        "patterns": ["missing.txt"],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
    )
    _run(["git", "add", "."], cwd=repo_root)
    _run(["git", "commit", "-m", "initial"], cwd=repo_root)

    _write(repo_root, "tracked.txt", "changed\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--repo-root",
            str(repo_root),
            "--fail-on-unmatched",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Unmatched: 1" in result.stdout
