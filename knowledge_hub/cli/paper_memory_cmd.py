"""Legacy paper-memory command shim."""

from knowledge_hub.interfaces.cli.commands.paper_memory_cmd import (
    build_paper_memory,
    paper_memory_group,
    rebuild_paper_memory,
    search_paper_memory,
    show_paper_memory,
)

__all__ = [
    "build_paper_memory",
    "paper_memory_group",
    "rebuild_paper_memory",
    "search_paper_memory",
    "show_paper_memory",
]
