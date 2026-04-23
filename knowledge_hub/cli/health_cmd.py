"""Legacy health command shim."""

from knowledge_hub.interfaces.cli.commands.health_cmd import (
    health_cmd,
    run_health,
)

__all__ = ["health_cmd", "run_health"]
