"""Legacy crawl command shim."""

from knowledge_hub.interfaces.cli.commands.crawl_cmd import crawl_group, labs_crawl_group

__all__ = [
    "crawl_group",
    "labs_crawl_group",
]
