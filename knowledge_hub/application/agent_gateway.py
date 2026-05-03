from __future__ import annotations

from typing import Literal

GatewaySurface = Literal["task_context", "agent_run", "agent_writeback_request"]
GatewayMode = Literal["context", "dry_run", "request"]

GATEWAY_VERSION = "v1"
GATEWAY_CONTRACT = "read_only_dry_run"


def build_gateway_metadata(*, surface: GatewaySurface, mode: GatewayMode) -> dict[str, object]:
    return {
        "version": GATEWAY_VERSION,
        "surface": surface,
        "mode": mode,
        "contract": GATEWAY_CONTRACT,
        "executionAllowed": False,
        "writebackAllowed": False,
        "repoContextEphemeral": True,
    }


def build_writeback_request_gateway_metadata() -> dict[str, object]:
    return {
        "version": "v2",
        "surface": "agent_writeback_request",
        "mode": "request",
        "contract": "approval_gated_repo_local_writeback",
        "executionAllowed": False,
        "writebackAllowed": False,
        "repoContextEphemeral": True,
        "approvalRequired": True,
    }
