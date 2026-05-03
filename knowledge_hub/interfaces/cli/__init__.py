"""Canonical CLI package surface.

Avoid importing ``knowledge_hub.interfaces.cli.main`` eagerly so
``python -m knowledge_hub.interfaces.cli.main`` does not preload the module.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["KhubContext", "cli", "main"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    module = import_module("knowledge_hub.interfaces.cli.main")
    return getattr(module, name)
