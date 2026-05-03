"""Legacy index worker shim."""

from knowledge_hub.interfaces.cli.index_worker import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
