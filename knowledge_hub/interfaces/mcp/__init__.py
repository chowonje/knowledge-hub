"""Canonical MCP package surface.

Keep package exports lazy so direct module execution does not preload the
server module as a side effect of package import.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["app", "call_tool", "list_tools", "main"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    module = import_module("knowledge_hub.interfaces.mcp.server")
    return getattr(module, name)
