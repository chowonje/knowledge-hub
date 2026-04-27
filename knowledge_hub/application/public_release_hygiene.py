"""Checks for public-release hygiene before opening the repo."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import subprocess
from typing import Any


PUBLIC_RELEASE_HYGIENE_SCHEMA = "knowledge-hub.public-release-hygiene.result.v1"

_TRACKED_PATH_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "tracked_local_config",
        re.compile(r"(^|/)config\.yaml$", re.IGNORECASE),
        "config.yaml should stay local; keep only config.yaml.example in the public branch",
    ),
    (
        "tracked_config_backup",
        re.compile(r"(^|/)config\.yaml\.bak_[^/]+$", re.IGNORECASE),
        "config backup files should not be tracked in the public branch",
    ),
    (
        "tracked_dotenv",
        re.compile(r"(^|/)\.env(?:\.[^/]+)?$", re.IGNORECASE),
        ".env files should not be tracked; keep only .env.example",
    ),
    (
        "tracked_runtime_database",
        re.compile(r"(^|/).+\.(?:db|sqlite|sqlite3)(?:-shm|-wal)?$", re.IGNORECASE),
        "runtime database files should not be tracked in the public branch",
    ),
    (
        "tracked_runtime_state",
        re.compile(r"(^|/)\.khub(/|$)", re.IGNORECASE),
        "runtime state under .khub/ should not be tracked",
    ),
    (
        "tracked_generated_eval_run",
        re.compile(r"(^|/)eval/knowledgeos/runs/", re.IGNORECASE),
        "generated eval run artifacts should be curated or removed before release",
    ),
    (
        "tracked_generated_eval_failure_bank",
        re.compile(r"(^|/)eval/knowledgeos/failures/", re.IGNORECASE),
        "local failure-bank records can contain raw queries and paths; keep them outside the public branch",
    ),
    (
        "tracked_generated_ab_run",
        re.compile(r"(^|/)runs/ab/", re.IGNORECASE),
        "generated A/B run artifacts should be externalized before release",
    ),
)

_HIGH_CONFIDENCE_SECRET_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "openai_style_key",
        re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
        "high-confidence OpenAI-style key literal found",
    ),
    (
        "github_pat",
        re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
        "high-confidence GitHub token literal found",
    ),
    (
        "github_pat_fine_grained",
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
        "high-confidence fine-grained GitHub token literal found",
    ),
)

_LOCAL_PATH_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "absolute_user_path",
        re.compile(r"(" + re.escape("/" + "Users/") + r"[^\"'`<>\n\r]+)"),
        "absolute macOS user path found; prefer relative paths in public docs/examples",
    ),
    (
        "absolute_home_path",
        re.compile(r"(" + re.escape("/" + "home/") + r"[^\"'`<>\n\r]+)"),
        "absolute home-directory path found; prefer relative paths in public docs/examples",
    ),
    (
        "absolute_volume_path",
        re.compile(r"(" + re.escape("/" + "Volumes/") + r"[^\"'`<>\n\r]+)"),
        "absolute external-volume path found; prefer config/env/default home paths in public docs/examples",
    ),
)

_TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".css",
    ".csv",
    ".env",
    ".example",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".py",
    ".sh",
    ".sql",
    ".text",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
_SKIP_SUFFIXES = {
    ".db",
    ".dylib",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".lock",
    ".pdf",
    ".png",
    ".pyc",
    ".so",
    ".sqlite",
    ".svg",
    ".whl",
    ".zip",
}


@dataclass(frozen=True)
class HygieneIssue:
    kind: str
    path: str
    detail: str
    match: str = ""


def list_tracked_files(repo_root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "git ls-files failed").strip())
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _looks_textual(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in _SKIP_SUFFIXES:
        return False
    if suffix in _TEXT_SUFFIXES:
        return True
    if not suffix:
        return True
    return False


def _read_text(path: Path, *, max_bytes: int = 250_000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if b"\x00" in data[:4096]:
        return ""
    return data[:max_bytes].decode("utf-8", errors="ignore")


def _path_issues(tracked_files: list[str]) -> list[HygieneIssue]:
    issues: list[HygieneIssue] = []
    for rel_path in tracked_files:
        if rel_path == ".env.example":
            continue
        for kind, pattern, detail in _TRACKED_PATH_RULES:
            if pattern.search(rel_path):
                issues.append(HygieneIssue(kind=kind, path=rel_path, detail=detail))
    return issues


def _content_issues(repo_root: Path, tracked_files: list[str]) -> list[HygieneIssue]:
    issues: list[HygieneIssue] = []
    for rel_path in tracked_files:
        path = repo_root / rel_path
        if not _looks_textual(path):
            continue
        text = _read_text(path)
        if not text:
            continue
        for kind, pattern, detail in _HIGH_CONFIDENCE_SECRET_RULES:
            match = pattern.search(text)
            if match:
                issues.append(HygieneIssue(kind=kind, path=rel_path, detail=detail, match=match.group(0)[:80]))
        for kind, pattern, detail in _LOCAL_PATH_RULES:
            match = pattern.search(text)
            if match:
                issues.append(HygieneIssue(kind=kind, path=rel_path, detail=detail, match=match.group(1)[:120]))
    return issues


def check_public_release_hygiene(repo_root: Path, *, tracked_files: list[str] | None = None) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    files = list(tracked_files) if tracked_files is not None else list_tracked_files(root)
    issues = [*_path_issues(files), *_content_issues(root, files)]
    issues.sort(key=lambda item: (item.kind, item.path, item.match))
    grouped: dict[str, int] = {}
    for issue in issues:
        grouped[issue.kind] = grouped.get(issue.kind, 0) + 1
    ok = not issues
    return {
        "schema": PUBLIC_RELEASE_HYGIENE_SCHEMA,
        "status": "ok" if ok else "failed",
        "repoRoot": str(root),
        "trackedFileCount": len(files),
        "issueCount": len(issues),
        "issueCountsByKind": grouped,
        "issues": [asdict(item) for item in issues],
        "nextActions": (
            []
            if ok
            else [
                "remove tracked local/runtime files from the public branch",
                "replace literal secrets with environment-variable placeholders",
                "rewrite absolute user paths in docs/examples to relative paths",
            ]
        ),
    }


__all__ = ["PUBLIC_RELEASE_HYGIENE_SCHEMA", "check_public_release_hygiene", "list_tracked_files"]
