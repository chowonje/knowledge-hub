from __future__ import annotations

from collections.abc import Iterable


DEFAULT_VAULT_EXCLUDE_FOLDERS: tuple[str, ...] = (
    ".obsidian",
    ".trash",
    "templates",
    ".local-rag",
    "node_modules",
)


def resolve_vault_exclude_folders(excludes: Iterable[str] | None = None) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in list(DEFAULT_VAULT_EXCLUDE_FOLDERS) + list(excludes or []):
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        merged.append(token)
        seen.add(token)
    return merged
