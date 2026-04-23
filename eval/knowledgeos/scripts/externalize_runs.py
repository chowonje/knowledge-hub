#!/usr/bin/env python3
"""Move local eval run artifacts out of the product repo and leave a symlink."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any


DEFAULT_TARGET_ROOT = Path.home() / ".khub" / "eval" / "knowledgeos" / "runs"


def default_target_root() -> Path:
    raw = str(os.environ.get("KHUB_EVAL_RUNS_ROOT", "") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_TARGET_ROOT


def build_plan(*, repo_root: Path, target_root: Path) -> dict[str, Any]:
    repo_runs = (repo_root / "eval" / "knowledgeos" / "runs").resolve(strict=False)
    target_root = target_root.expanduser().resolve(strict=False)
    return {
        "repoRoot": str(repo_root),
        "repoEvalRunsPath": str(repo_runs),
        "targetRoot": str(target_root),
        "repoEvalRunsExists": repo_runs.exists(),
        "repoEvalRunsIsSymlink": repo_runs.is_symlink(),
        "targetExists": target_root.exists(),
    }


def externalize_eval_runs(*, repo_root: Path, target_root: Path, dry_run: bool = False) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    repo_runs = repo_root / "eval" / "knowledgeos" / "runs"
    target_root = target_root.expanduser().resolve(strict=False)
    plan = build_plan(repo_root=repo_root, target_root=target_root)

    if repo_runs.is_symlink():
        current_target = repo_runs.resolve(strict=False)
        payload = {
            "status": "already_externalized" if current_target == target_root else "unexpected_symlink",
            "repoRoot": str(repo_root),
            "repoEvalRunsPath": str(repo_runs),
            "targetRoot": str(target_root),
            "currentTarget": str(current_target),
            "dryRun": dry_run,
        }
        if payload["status"] != "already_externalized":
            payload["error"] = "repo eval/knowledgeos/runs already points somewhere else"
        return payload

    if repo_runs.exists() and target_root.exists():
        raise RuntimeError(f"both source and target already exist: {repo_runs} and {target_root}")

    if dry_run:
        return {
            "status": "planned",
            "repoRoot": str(repo_root),
            "repoEvalRunsPath": str(repo_runs),
            "targetRoot": str(target_root),
            "moveRequired": bool(repo_runs.exists()),
            "linkRequired": not repo_runs.is_symlink(),
            "plan": plan,
            "dryRun": True,
        }

    target_root.parent.mkdir(parents=True, exist_ok=True)
    repo_runs.parent.mkdir(parents=True, exist_ok=True)

    if repo_runs.is_symlink():
        current_target = repo_runs.resolve(strict=False)
        return {
            "status": "already_externalized" if current_target == target_root else "unexpected_symlink",
            "repoRoot": str(repo_root),
            "repoEvalRunsPath": str(repo_runs),
            "targetRoot": str(target_root),
            "currentTarget": str(current_target),
            "dryRun": dry_run,
        }

    if not repo_runs.exists() and target_root.exists():
        repo_runs.symlink_to(target_root, target_is_directory=True)
        return {
            "status": "externalized",
            "repoRoot": str(repo_root),
            "repoEvalRunsPath": str(repo_runs),
            "targetRoot": str(target_root),
            "movedExisting": False,
            "symlinkCreated": True,
            "recoveredMissingSource": True,
            "dryRun": False,
        }

    moved_existing = False
    if repo_runs.exists():
        shutil.move(str(repo_runs), str(target_root))
        moved_existing = True
    else:
        target_root.mkdir(parents=True, exist_ok=True)

    repo_runs.symlink_to(target_root, target_is_directory=True)
    return {
        "status": "externalized",
        "repoRoot": str(repo_root),
        "repoEvalRunsPath": str(repo_runs),
        "targetRoot": str(target_root),
        "movedExisting": moved_existing,
        "symlinkCreated": True,
        "dryRun": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Move repo-local eval/knowledgeos/runs artifacts to ~/.khub and leave a symlink."
    )
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[3], help="knowledge-hub repo root")
    parser.add_argument(
        "--target-root",
        default=str(default_target_root()),
        help="external artifact directory for eval/knowledgeos/runs",
    )
    parser.add_argument("--dry-run", action="store_true", help="show the move/link plan without changing the filesystem")
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    args = parser.parse_args(argv)

    try:
        payload = externalize_eval_runs(
            repo_root=Path(args.repo_root),
            target_root=Path(args.target_root),
            dry_run=bool(args.dry_run),
        )
    except Exception as error:  # pragma: no cover - exercised through CLI only
        payload = {
            "status": "failed",
            "repoRoot": str(Path(args.repo_root).expanduser()),
            "targetRoot": str(Path(args.target_root).expanduser()),
            "error": str(error),
            "dryRun": bool(args.dry_run),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 1
        raise SystemExit(str(error)) from error

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            f"status={payload['status']} repo_eval_runs={payload.get('repoEvalRunsPath', '-')}"
            f" target={payload.get('targetRoot', '-')}"
        )
    return 0 if payload.get("status") in {"planned", "externalized", "already_externalized"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
