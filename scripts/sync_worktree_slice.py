#!/usr/bin/env python3
"""Sync a selected dirty-tree slice into a clean target worktree."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a selected slice of changed paths from one git worktree to another."
    )
    parser.add_argument("--source-repo-root", type=Path, required=True, help="Dirty source worktree root.")
    parser.add_argument("--target-repo-root", type=Path, required=True, help="Clean target worktree root.")
    parser.add_argument(
        "--pathspec-file",
        type=Path,
        required=True,
        help="Static manifest containing the relative paths to sync.",
    )
    parser.add_argument(
        "--stage",
        action="store_true",
        help="Stage the synced paths in the target worktree with `git add -A`.",
    )
    parser.add_argument(
        "--allow-dirty-target",
        action="store_true",
        help="Allow syncing into a target worktree that already has local changes.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable result payload.")
    return parser.parse_args()


def _git(args: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _git_root(repo_root: Path) -> Path:
    proc = _git(["rev-parse", "--show-toplevel"], cwd=repo_root)
    return Path(proc.stdout.strip()).resolve()


def _validate_relative_path(raw_line: str) -> str:
    path = raw_line.strip()
    if not path or path.startswith("#"):
        return ""
    if path.startswith(":("):
        raise ValueError(f"magic pathspecs are not supported in sync manifests: {path}")
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"pathspec entries must stay inside the repo root: {path}")
    if path.endswith("/"):
        raise ValueError(f"directory-only entries are not supported; list files explicitly: {path}")
    return candidate.as_posix()


def _load_manifest(pathspec_file: Path) -> list[str]:
    paths: list[str] = []
    for raw_line in pathspec_file.read_text(encoding="utf-8").splitlines():
        path = _validate_relative_path(raw_line)
        if path:
            paths.append(path)
    return paths


def _chunked(values: list[str], *, size: int = 200) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _status_map(repo_root: Path, rel_paths: list[str]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for chunk in _chunked(rel_paths):
        proc = _git(["status", "--short", "--untracked-files=all", "--", *chunk], cwd=repo_root)
        for raw_line in proc.stdout.splitlines():
            if not raw_line:
                continue
            status = raw_line[:2]
            path = raw_line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path.startswith('"') and path.endswith('"'):
                path = bytes(path[1:-1], "utf-8").decode("unicode_escape")
            statuses[path] = status
    return statuses


def _target_is_dirty(target_root: Path) -> bool:
    proc = _git(["status", "--short", "--untracked-files=all"], cwd=target_root)
    return bool(proc.stdout.strip())


def _sync_path(source_root: Path, target_root: Path, rel_path: str, status: str) -> dict[str, str]:
    source_path = source_root / rel_path
    target_path = target_root / rel_path
    action = "skipped"

    if source_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)
        action = "copied"
    elif "D" in status:
        if target_path.is_dir():
            shutil.rmtree(target_path)
        elif target_path.exists():
            target_path.unlink()
        action = "deleted"
    else:
        raise FileNotFoundError(
            f"source path is missing but not reported as deleted: {rel_path} ({status.strip() or 'clean'})"
        )

    return {"path": rel_path, "status": status, "action": action}


def _stage_paths(target_root: Path, rel_paths: list[str]) -> None:
    for chunk in _chunked(rel_paths):
        _git(["add", "-A", "--", *chunk], cwd=target_root)


def build_payload(
    *,
    source_root: Path,
    target_root: Path,
    pathspec_file: Path,
    stage: bool,
    allow_dirty_target: bool,
) -> dict[str, object]:
    resolved_source = _git_root(source_root)
    resolved_target = _git_root(target_root)
    if resolved_source == resolved_target:
        raise ValueError("source and target worktrees must be different directories")
    if not allow_dirty_target and _target_is_dirty(resolved_target):
        raise RuntimeError(f"target worktree is dirty: {resolved_target}")

    rel_paths = _load_manifest(pathspec_file.resolve())
    status_map = _status_map(resolved_source, rel_paths)
    changed_paths = [path for path in rel_paths if status_map.get(path)]
    actions = [
        _sync_path(resolved_source, resolved_target, rel_path, status_map[rel_path])
        for rel_path in changed_paths
    ]
    if stage and changed_paths:
        _stage_paths(resolved_target, changed_paths)

    copied = [item["path"] for item in actions if item["action"] == "copied"]
    deleted = [item["path"] for item in actions if item["action"] == "deleted"]
    skipped = [path for path in rel_paths if path not in changed_paths]
    return {
        "status": "ok",
        "sourceRepoRoot": str(resolved_source),
        "targetRepoRoot": str(resolved_target),
        "pathspecFile": str(pathspec_file.resolve()),
        "stage": bool(stage),
        "allowDirtyTarget": bool(allow_dirty_target),
        "manifestPathCount": len(rel_paths),
        "changedPathCount": len(changed_paths),
        "copiedPathCount": len(copied),
        "deletedPathCount": len(deleted),
        "skippedPathCount": len(skipped),
        "copiedPaths": copied,
        "deletedPaths": deleted,
        "skippedPaths": skipped,
        "actions": actions,
    }


def main() -> int:
    args = parse_args()
    payload = build_payload(
        source_root=args.source_repo_root.resolve(),
        target_root=args.target_repo_root.resolve(),
        pathspec_file=args.pathspec_file.resolve(),
        stage=bool(args.stage),
        allow_dirty_target=bool(args.allow_dirty_target),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            "synced slice "
            f"changed={payload['changedPathCount']} copied={payload['copiedPathCount']} "
            f"deleted={payload['deletedPathCount']} skipped={payload['skippedPathCount']}"
        )
        for item in payload["actions"]:
            print(f"{item['action']}: {item['path']} ({item['status'].strip() or 'clean'})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # pragma: no cover - CLI guard
        print(f"sync_worktree_slice.py failed: {error}", file=sys.stderr)
        raise SystemExit(1)
