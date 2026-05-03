"""Lazy exports for paper surfaces.

This package used to eagerly import heavy modules, which creates circular
imports once lightweight helpers inside ``knowledge_hub.papers`` are reused
from lower layers such as the SQLite stores.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "PaperDownloader",
    "PaperTranslator",
    "PaperManager",
    "discover_papers",
    "DiscoveredPaper",
]


def __getattr__(name: str) -> Any:
    if name == "PaperDownloader":
        from knowledge_hub.papers.downloader import PaperDownloader

        return PaperDownloader
    if name == "PaperTranslator":
        from knowledge_hub.papers.translator import PaperTranslator

        return PaperTranslator
    if name == "PaperManager":
        from knowledge_hub.papers.manager import PaperManager

        return PaperManager
    if name in {"discover_papers", "DiscoveredPaper"}:
        from knowledge_hub.papers.discoverer import DiscoveredPaper, discover_papers

        return {"discover_papers": discover_papers, "DiscoveredPaper": DiscoveredPaper}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
