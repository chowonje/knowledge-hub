"""Advisory writeback preview helpers for Agent Gateway v2 request payloads."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Any

_PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_./\\-]+")
_DATED_NOTE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}.*\.md$")
DEFAULT_DOCS_ONLY_PATH_PREFIXES = (
    "docs/adr/",
    "docs/status/",
    "reviews/",
    "worklog/",
)
_LIKELY_CODE_SUFFIXES = (
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sql",
    ".sh",
    ".swift",
    ".java",
    ".go",
    ".rs",
)


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _normalize_rel_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    return raw.strip("/")


def _workspace_mode(*, include_workspace: bool | None, effective_include_workspace: bool) -> str:
    if not effective_include_workspace:
        return "disabled"
    if include_workspace is True:
        return "explicit"
    return "inferred"


def _target_confidence(item: dict[str, Any]) -> str:
    if str(item.get("source_type") or "").strip().lower() == "project":
        return "high"
    return "medium"


def _normalize_prefixes(prefixes: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in list(prefixes or []):
        token = _normalize_rel_path(value)
        if not token:
            continue
        if not token.endswith("/"):
            token = token + "/"
        if token not in normalized:
            normalized.append(token)
    return tuple(normalized)


def _is_allowed_path(rel_path: str, *, allowed_path_prefixes: tuple[str, ...]) -> bool:
    if not allowed_path_prefixes:
        return True
    return any(rel_path.startswith(prefix) for prefix in allowed_path_prefixes)


def _preferred_prefixes(goal: str, *, allowed_path_prefixes: tuple[str, ...]) -> tuple[str, ...]:
    text = str(goal or "").lower()
    preferred: list[str] = []
    candidates = (
        ("docs/status", "docs/status/"),
        ("worklog", "worklog/"),
        ("docs/adr", "docs/adr/"),
        ("reviews/", "reviews/"),
        ("review", "reviews/"),
    )
    for token, prefix in candidates:
        if token in text and prefix in allowed_path_prefixes and prefix not in preferred:
            preferred.append(prefix)
    return tuple(preferred)


def _matches_preferred_prefixes(rel_path: str, *, preferred_path_prefixes: tuple[str, ...]) -> bool:
    if not preferred_path_prefixes:
        return True
    return any(rel_path.startswith(prefix) for prefix in preferred_path_prefixes)


def _task_context_targets(
    repo_root: Path,
    task_context_payload: dict[str, Any],
    *,
    limit: int,
    allowed_path_prefixes: tuple[str, ...],
    preferred_path_prefixes: tuple[str, ...],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(task_context_payload.get("workspace_files") or []):
        rel_path = _normalize_rel_path(
            item.get("relative_path") or item.get("relativePath") or item.get("path") or item.get("title") or ""
        )
        if (
            not rel_path
            or rel_path in seen
            or not _is_allowed_path(rel_path, allowed_path_prefixes=allowed_path_prefixes)
            or not _matches_preferred_prefixes(rel_path, preferred_path_prefixes=preferred_path_prefixes)
        ):
            continue
        seen.add(rel_path)
        targets.append(
            {
                "relativePath": rel_path,
                "reason": str(item.get("reason") or "workspace evidence selected this file for the requested change").strip(),
                "source": "task_context",
                "confidence": _target_confidence(item),
                "exists": (repo_root / rel_path).exists(),
            }
        )
        if len(targets) >= limit:
            break
    return targets


def _goal_inference_targets(
    repo_root: Path,
    goal: str,
    *,
    seen: set[str],
    limit: int,
    allowed_path_prefixes: tuple[str, ...],
    preferred_path_prefixes: tuple[str, ...],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for token in _PATH_TOKEN_RE.findall(str(goal or "")):
        normalized = _normalize_rel_path(token.strip(".,:;)]}\"'"))
        if not normalized or normalized in seen:
            continue
        if "/" not in normalized and not any(normalized.endswith(suffix) for suffix in _LIKELY_CODE_SUFFIXES):
            continue
        candidate_path = repo_root / normalized
        if not candidate_path.exists():
            continue
        if not _is_allowed_path(normalized, allowed_path_prefixes=allowed_path_prefixes):
            continue
        if not _matches_preferred_prefixes(normalized, preferred_path_prefixes=preferred_path_prefixes):
            continue
        seen.add(normalized)
        candidates.append(
            {
                "relativePath": normalized,
                "reason": "goal text directly references this path",
                "source": "goal_inference",
                "confidence": "low",
                "exists": True,
            }
        )
        if len(candidates) >= limit:
            break
    return candidates


def _latest_markdown_file(root: Path, relative_dir: str) -> str | None:
    dir_path = root / relative_dir
    if not dir_path.exists() or not dir_path.is_dir():
        return None
    files = [path for path in dir_path.iterdir() if path.is_file() and path.suffix.lower() == ".md"]
    if not files:
        return None
    dated = sorted(path for path in files if _DATED_NOTE_RE.match(path.name))
    picked = dated[-1] if dated else sorted(files, key=lambda path: path.name)[-1]
    return _normalize_rel_path(str(picked.relative_to(root)))


def _preferred_prefix_fallback_targets(
    repo_root: Path,
    *,
    seen: set[str],
    limit: int,
    preferred_path_prefixes: tuple[str, ...],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for prefix in preferred_path_prefixes:
        if len(targets) >= limit:
            break
        rel_path = ""
        if prefix == "worklog/":
            # Prefer today's worklog note when it exists; otherwise fall back to the latest dated worklog note.
            candidate = repo_root / "worklog" / f"{datetime.now().date().isoformat()}.md"
            if candidate.exists():
                rel_path = _normalize_rel_path(str(candidate.relative_to(repo_root)))
            else:
                rel_path = _latest_markdown_file(repo_root, "worklog") or ""
        elif prefix == "docs/status/":
            rel_path = _latest_markdown_file(repo_root, "docs/status") or ""
        if not rel_path or rel_path in seen:
            continue
        seen.add(rel_path)
        targets.append(
            {
                "relativePath": rel_path,
                "reason": f"goal explicitly points to the {prefix.rstrip('/')} lane; selected the latest matching note",
                "source": "goal_inference",
                "confidence": "medium",
                "exists": True,
            }
        )
    return targets


def build_writeback_preview(
    *,
    goal: str,
    repo_path: str,
    dry_run_payload: dict[str, Any] | None,
    task_context_payload: dict[str, Any] | None,
    include_workspace: bool | None,
    effective_include_workspace: bool,
    context_error: str = "",
    max_targets: int = 8,
    allowed_path_prefixes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_path).expanduser().resolve()
    context_payload = task_context_payload or {}
    normalized_prefixes = _normalize_prefixes(allowed_path_prefixes)
    preferred_prefixes = _preferred_prefixes(goal, allowed_path_prefixes=normalized_prefixes)
    targets = _task_context_targets(
        repo_root,
        context_payload,
        limit=max_targets,
        allowed_path_prefixes=normalized_prefixes,
        preferred_path_prefixes=preferred_prefixes,
    )
    seen = {item["relativePath"] for item in targets}
    if len(targets) < max_targets:
        targets.extend(
            _goal_inference_targets(
                repo_root,
                goal,
                seen=seen,
                limit=max_targets - len(targets),
                allowed_path_prefixes=normalized_prefixes,
                preferred_path_prefixes=preferred_prefixes,
            )
        )
    if len(targets) < max_targets:
        targets.extend(
            _preferred_prefix_fallback_targets(
                repo_root,
                seen=seen,
                limit=max_targets - len(targets),
                preferred_path_prefixes=preferred_prefixes,
            )
        )

    unknowns: list[str] = []
    if context_error:
        unknowns.append(str(context_error).strip())
    if normalized_prefixes:
        unknowns.append(
            "writeback preview is narrowed to allowed path prefixes: "
            + ", ".join(normalized_prefixes)
        )
    if not targets:
        unknowns.append("No repo-local write targets were predicted from workspace context or goal text.")

    workspace_files = list(context_payload.get("workspace_files") or [])
    workspace_context = {
        "included": bool(effective_include_workspace),
        "mode": _workspace_mode(
            include_workspace=include_workspace,
            effective_include_workspace=effective_include_workspace,
        ),
        "workspaceFileCount": len(workspace_files),
    }

    preview = {
        "status": "ok" if targets else "partial",
        "kind": "repo_local_predicted_write_set",
        "intentSummary": f"Advisory repo-local writeback preview for goal: {str(goal or '').strip()}",
        "targetCount": len(targets),
        "targets": targets,
        "workspaceContext": workspace_context,
        "constraints": {
            "repoLocalOnly": True,
            "vaultWrite": False,
            "externalSideEffects": False,
            "hiddenAutomation": False,
            "allowedPathPrefixes": list(normalized_prefixes),
        },
        "unknowns": unknowns,
        "advisory": True,
    }
    preview["previewFingerprint"] = _stable_fingerprint(
        {
            "goal": str(goal or "").strip(),
            "repoPath": str(repo_root),
            "dryRunStatus": str((dry_run_payload or {}).get("status") or ""),
            "dryRunStage": str((dry_run_payload or {}).get("stage") or ""),
            "status": preview["status"],
            "targets": preview["targets"],
            "workspaceContext": preview["workspaceContext"],
            "constraints": preview["constraints"],
            "unknowns": preview["unknowns"],
        }
    )
    return preview
