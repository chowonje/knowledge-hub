from __future__ import annotations

from typing import Any

from knowledge_hub.mcp.handlers import agent, crawl, epistemic, jobs, learn, paper, search, transform


async def dispatch_tool(name: str, arguments: dict[str, Any], ctx: dict[str, Any]):
    for handler in (
        agent.handle_tool,
        jobs.handle_tool,
        search.handle_tool,
        transform.handle_tool,
        epistemic.handle_tool,
        learn.handle_tool,
        crawl.handle_tool,
        paper.handle_tool,
    ):
        result = await handler(name, arguments, ctx)
        if result is not None:
            return result
    return None
