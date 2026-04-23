"""Legacy CLI entrypoint shim.

The canonical CLI entrypoint now lives at ``knowledge_hub.interfaces.cli.main``.
Keep this module for compatibility with tests, subprocess calls, and older
imports that still reference ``knowledge_hub.cli.main``.
"""

from knowledge_hub.interfaces.cli.main import KhubContext, cli, main

__all__ = ["KhubContext", "cli", "main"]


if __name__ == "__main__":
    main()
