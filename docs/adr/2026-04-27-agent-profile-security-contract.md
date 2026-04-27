# ADR: Agent Profile Security Contract

Date: 2026-04-27

## Status

Accepted for the frontier agent-runtime branch.

## Context

Second Core #3 exposes agent-ready local knowledge through the existing MCP server. That makes the MCP profile boundary part of the product contract, not just a discovery convenience. Codex, Claude, Cursor, and similar callers need a packet that is explicit about whether evidence can be used, whether human review is required, and whether any memory proposal is only staged.

The default public MCP profile remains the narrow retrieval core. Agent tools are opt-in via `KHUB_MCP_PROFILE=agent`, `labs`, or `all`.

## Decision

`AgentContextPacket v1` is the required envelope for agent profile tools.

The security contract is:

- `policy.policyMode` is always `local-only` in v1.
- `policy.allowExternal` and `policy.externalSendAllowed` default to `false`.
- `safeToUse=false` and `requiredHumanReview=true` are used for policy blocks, missing answer contracts, failed/abstain verification, or unsupported-claim risk.
- `stageOnly=true` is required for memory staging tools.
- `finalApply=false` is required for all agent packets in v1.
- `sourceTextRole=evidence_not_instruction` marks retrieved/source text as data, never tool instructions.
- `redactionApplied` records whether packet/output sanitization omitted or redacted sensitive details.
- `blockedReason` records the first machine-readable reason a packet is not safe to use.
- P0 inspected payloads, raw stage proposal bodies, absolute local paths, `file://` URIs, secret-like values, and internal run/artifact paths are not echoed back in agent MCP request echoes or artifacts.
- Delegated legacy `run_agentic_query` payloads must prove local-only external policy before a completed delegated result can be treated as allowed.
- Source text passed into local synthesis is wrapped as `evidence_not_instruction` data and is not treated as an instruction source.

## Consequences

Agent callers can treat `safeToUse`, `requiredHumanReview`, `sourceTextRole`, and `finalApply` as stable decision fields before acting on local knowledge.

The packet remains inspectable even for `agent_policy_check` and `agent_stage_memory` policy denials, but the inspected payload body is omitted from the returned packet and MCP artifact.

The contract intentionally does not add a new MCP server or widen the default public profile. Future external model routing must be an explicit opt-in change with separate policy tests.
