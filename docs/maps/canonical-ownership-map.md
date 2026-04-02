# Canonical Ownership Map

This map shows who owns what in the current repo and which surfaces are
canonical versus compatibility-only.

Diagram source: `canonical-ownership-map.mmd`

## Reading Guide

- `knowledge_hub.interfaces.*` is the canonical human/tool entry surface.
- `knowledge_hub.application.*` owns orchestration and runtime composition.
- `knowledge_hub.infrastructure.*` owns runtime-facing config, persistence, and
  provider lookup surfaces.
- `knowledge_hub.core.*` owns low-level local primitives and some compatibility
  exports.
- `foundry-core/` is the stricter TypeScript runtime, policy, and audit
  boundary.

## Key Rules

- `khub` resolves to `knowledge_hub.interfaces.cli.main:cli`.
- `khub-mcp` resolves to `knowledge_hub.interfaces.mcp.server:main`.
- Legacy shims such as `knowledge_hub.cli.*` and `knowledge_hub.mcp_server`
  remain compatibility-only. They are not peer architecture layers.
- Python remains the final authority for mode and payload decisions even when
  Foundry is used as the preferred delegated runtime.
- The Python/TypeScript boundary is additive:
  `foundry-core` orchestrates preferred agent flows, but Python owns the default
  product runtime and fallback behavior.

## Implementation Notes

- `Interfaces` should delegate into `Application`, not grow into a second
  orchestration platform.
- CLI command facades under `knowledge_hub.interfaces.cli.commands.*` may still
  host compatibility wrappers, but durable helper/runtime bodies should move
  into narrower command-local runtime modules. Current examples include
  `paper_public_reading_cmd.py`, `paper_admin_runtime.py`,
  `paper_materialization_runtime.py`, `paper_maintenance_runtime.py`, and
  `paper_shared_runtime.py` behind the stable `paper_cmd.py` surface.
- `Compatibility` is a cross-cutting concern rather than a single package.
  Some shims live under top-level legacy modules, and some compatibility
  exports live inside `core` or `infrastructure`.
- Current docs describe `Interfaces` as thin, but the MCP server still contains
  non-trivial wiring logic. Treat thinness as the target contract, not a fully
  achieved property.

## Provenance

Primary docs:

- `docs/ARCHITECTURE.md`
- `docs/PROJECT_STATE.md`
- `docs/repo-layout.md`
- `AGENTS.md`

Primary implementation references:

- `knowledge_hub/interfaces/cli/main.py`
- `knowledge_hub/interfaces/mcp/server.py`
- `knowledge_hub/application/agent/foundry_bridge.py`
- `knowledge_hub/interfaces/cli/commands/agent_cmd.py`
- `knowledge_hub/mcp/handlers/agent.py`
- `knowledge_hub/cli/main.py`
- `knowledge_hub/mcp_server.py`

Supporting verification:

- `foundry-core/tests/knowledge-hub-cli-entrypoint.test.ts`
