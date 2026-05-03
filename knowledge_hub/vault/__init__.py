"""Vault package exports.

Keep imports optional so utility modules can be used in minimal environments
where heavy parsing dependencies (e.g. python-frontmatter) are not installed.
"""

from __future__ import annotations

try:  # pragma: no cover - optional dependency path
    from knowledge_hub.vault.parser import ObsidianParser
except Exception:  # pragma: no cover
    ObsidianParser = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency path
    from knowledge_hub.vault.indexer import VaultIndexer
except Exception:  # pragma: no cover
    VaultIndexer = None  # type: ignore[assignment]

from knowledge_hub.vault.organizer import VaultOrganizer
from knowledge_hub.vault.ai_organizer import AIVaultOrganizer
from knowledge_hub.vault.topology import VaultTopologyBuilder, TopologyBuildOptions, TopologyBuildError

__all__ = [
    "ObsidianParser",
    "VaultIndexer",
    "VaultOrganizer",
    "AIVaultOrganizer",
    "VaultTopologyBuilder",
    "TopologyBuildOptions",
    "TopologyBuildError",
]
