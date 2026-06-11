"""Authoritative gate for Obsidian vault writes.

Every vault writer must consult :func:`ensure_vault_writes_allowed` (directly
or via ``resolve_config_vault_write_adapter``) before touching vault files.
The gate fails closed: writes are blocked unless ``obsidian.enabled`` resolves
true AND ``obsidian.vault_path`` is configured. There is intentionally no
fallback vault location.
"""

from __future__ import annotations

from typing import Any


class VaultWriteError(RuntimeError):
    """Base error for blocked vault writes."""


class VaultWritesDisabledError(VaultWriteError):
    """Raised when a vault write is attempted while obsidian.enabled is false."""


class VaultWriteNotConfiguredError(VaultWriteError):
    """Raised when a vault write is attempted without obsidian.vault_path."""


def ensure_vault_writes_allowed(config: Any, *, vault_path: str | None = None) -> str:
    """Fail closed unless vault writes are explicitly enabled and configured.

    Returns the resolved vault path. ``vault_path`` overrides the configured
    path (CLI flags) but never bypasses the enabled check.
    """
    if not bool(getattr(config, "vault_enabled", False)):
        raise VaultWritesDisabledError(
            "vault writes are disabled (obsidian.enabled=false); "
            "set obsidian.enabled=true to allow vault writes"
        )
    resolved = str(vault_path if vault_path is not None else getattr(config, "vault_path", "") or "").strip()
    if not resolved:
        raise VaultWriteNotConfiguredError(
            "obsidian.vault_path is not configured; refusing to write to a default vault location"
        )
    return resolved
