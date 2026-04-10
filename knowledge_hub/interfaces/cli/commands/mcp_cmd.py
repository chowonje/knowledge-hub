"""MCP command for launching the local MCP server."""

from __future__ import annotations

import click

from knowledge_hub.interfaces.mcp.server import main as mcp_main
from knowledge_hub.version import get_version


@click.command("mcp")
@click.version_option(version=get_version(), prog_name="khub mcp")
def mcp_cmd():
    """Start MCP server for Cursor/Codex integration."""
    mcp_main()
