# Agent Execution Map

This map shows how agent execution works today for both the CLI and MCP
surfaces.

Diagram source: `agent-execution-map.mmd`

## Reading Guide

- Foundry is the preferred runtime for both CLI and MCP agent execution.
- Foundry is not the exclusive runtime. Python fallback orchestration still
  exists and is part of the real current implementation.
- CLI and MCP are intentionally not identical:
  CLI is synchronous and report-oriented, while MCP is queued through the async
  job layer and persists MCP-specific job state.

## CLI Lane

- `khub agent run` resolves goal, role, orchestration flags, and repo/workspace
  context.
- It tries the Foundry subprocess first through the shared bridge helpers.
- If Foundry returns a usable envelope, the CLI normalizes and prints it.
- If delegation fails, the CLI builds a local fallback envelope.
- The CLI path applies an extra artifact policy/redaction guard before final
  output.

## MCP Lane

- `run_agentic_query` enters through the MCP server and is executed through the
  async job layer.
- The job is created in `mcp_jobs`, marked queued, then moved to running/done or
  blocked/failed.
- MCP also prefers Foundry delegation first.
- If Foundry is unavailable, fallback branches by task mode:
  - coding/design/debug -> build task context, synthesize from task context, use
    workspace evidence as ephemeral context
  - general -> `search_knowledge` and/or `ask_knowledge`
- The MCP path applies its own verify/persistence gate before finalizing the job
  artifact.

## Current Reality

- Foundry runtime owns the clearest Plan -> Act -> Verify -> Writeback loop.
- Python fallback still carries meaningful orchestration and cannot be omitted
  from the map.
- `mcp_jobs` is an MCP transport/persistence layer, not a general agent session
  store.
- Writeback semantics are strongest in Foundry. Python fallback is more
  envelope/report oriented unless it delegates successfully.

## Provenance

Primary implementation references:

- `knowledge_hub/interfaces/cli/commands/agent_cmd.py`
- `knowledge_hub/mcp/handlers/agent.py`
- `knowledge_hub/application/agent/foundry_bridge.py`
- `knowledge_hub/application/task_context.py`
- `knowledge_hub/application/mcp/agent_payloads.py`
- `knowledge_hub/application/mcp/jobs.py`
- `knowledge_hub/core/mcp_job_store.py`
- `foundry-core/src/personal-foundry/agent-runtime.ts`

Supporting docs/tests:

- `docs/foundry-knowledge-hub-integration.md`
- `foundry-core/docs/personal-foundry-terminal-architecture.md`
- `foundry-core/tests/personal-foundry.test.ts`
