# Second Core #3: Agent-ready Local Knowledge Runtime

Date: 2026-04-27
Branch: `frontier/agent-ready-local-runtime-20260427`
Base: current Evidence-contract frontier branch / PR #5 lineage

## Objective

Add an agent-safe wrapper layer on top of the existing MCP server so Codex/Claude/Cursor-style callers can consume local knowledge through explicit context, evidence, verification, policy, and stage-only memory contracts.

This does not introduce a second MCP server and does not widen the public default MCP profile.

## Affected Paths

- `docs/schemas/agent-context-packet.v1.json`
- `knowledge_hub/application/mcp/agent_payloads.py`
- `knowledge_hub/application/mcp/responses.py`
- `knowledge_hub/core/schema_validator.py`
- `knowledge_hub/interfaces/mcp/server.py`
- `knowledge_hub/mcp/tool_specs.py`
- `knowledge_hub/mcp/handlers/agent.py`
- `tests/test_mcp_agent_runtime.py`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`

## Implementation Notes

- Added `AgentContextPacket v1` as the common agent-facing payload envelope.
- Added agent-safe MCP wrapper tools:
  - `agent_build_context`
  - `agent_search_knowledge`
  - `agent_ask_knowledge`
  - `agent_get_evidence`
  - `agent_policy_check`
  - `agent_stage_memory`
- The wrappers reuse existing task-context, search, and ask paths instead of adding a new RAG runtime.
- `agent_ask_knowledge` and `agent_get_evidence` reuse the existing answer contract normalization and expose `evidencePacketContract`, `answerContract`, and `verificationVerdict` at the packet top level.
- All agent packets default to `allowExternal=false` and `policyMode=local-only`.
- `safeToUse=false` / `requiredHumanReview=true` is set for policy blocks, missing answer-mode contracts, failed/abstain verification, and unsupported-claim risk.
- `agent_policy_check` classifies payloads before returning a redacted local-only packet.
- Review follow-up: `agent_policy_check` now keeps the schema-backed decision packet even when the inspected payload is P0, while omitting inspected payload bodies from MCP request echo and artifact content.
- `agent_stage_memory` is proposal/stage-only and never performs final Obsidian/vault apply.
- The default MCP profile continues to expose only the public retrieval core. Agent tools require `KHUB_MCP_PROFILE=agent`, `labs`, or `all`; default direct calls return a profile-blocked hint.
- Follow-up branch `frontier/agent-runtime-security-contract-20260427` hardens the agent security contract without merging it into `main`.
- `AgentContextPacket v1` now explicitly records `stageOnly`, `finalApply=false`, `sourceTextRole=evidence_not_instruction`, `redactionApplied`, and `blockedReason`.
- Agent MCP request echoes and artifacts now use common agent sanitization for local paths, `file://` URIs, secret-like values, and raw stage/policy payload bodies.
- `agent_stage_memory` now returns a schema-backed stage-only decision packet for sensitive inputs while still omitting the inspected proposal body and forbidding final apply.
- Delegated legacy `run_agentic_query` payloads now fail closed unless they carry explicit local-only external-policy proof.
- Local task-context synthesis now wraps source text as `evidence_not_instruction` data before passing it to a synthesis call.
- The durable security decision is recorded in `docs/adr/2026-04-27-agent-profile-security-contract.md`.

## Verification

- `python -m py_compile knowledge_hub/application/mcp/agent_payloads.py knowledge_hub/mcp/handlers/agent.py knowledge_hub/mcp/tool_specs.py knowledge_hub/interfaces/mcp/server.py`
- `python -m pytest tests/test_mcp_agent_runtime.py -q` (`9 passed`)
- `python -m pytest tests/test_mcp_agent_runtime.py tests/test_mcp_server.py tests/test_mcp_search_handler.py tests/test_mcp_server_helpers.py -q` (`57 passed`)
- `python -m pytest tests/test_answer_contracts_runtime.py tests/test_answer_contract_schemas.py tests/test_answer_quality_gate.py tests/test_retrieval_span_golden.py tests/test_evidence_contract_perf_gate.py -q` (`37 passed`)
- `python -m pytest tests/test_cli_add_facade.py tests/test_cli_smoke_contract.py tests/test_interfaces_cli_main.py -q` (`49 passed`)
- `python -m pytest tests/test_mcp_agent_runtime.py tests/test_cli_add_facade.py -q` (`21 passed`)
- `ruff check knowledge_hub/application/mcp/agent_payloads.py knowledge_hub/mcp/handlers/agent.py knowledge_hub/mcp/tool_specs.py knowledge_hub/interfaces/mcp/server.py tests/test_mcp_agent_runtime.py` (`passed`)
- `git diff --check` (`passed`)

Latest security-contract follow-up checks:

- `python -m py_compile knowledge_hub/application/mcp/agent_payloads.py knowledge_hub/mcp/handlers/agent.py knowledge_hub/interfaces/mcp/server.py knowledge_hub/application/mcp/responses.py` (`passed`)
- `ruff check knowledge_hub/application/mcp/agent_payloads.py knowledge_hub/mcp/handlers/agent.py knowledge_hub/interfaces/mcp/server.py knowledge_hub/application/mcp/responses.py tests/test_mcp_agent_runtime.py tests/test_mcp_agent_handler.py tests/test_mcp_server_helpers.py` (`passed`)
- `python -m pytest tests/test_mcp_agent_runtime.py tests/test_mcp_agent_handler.py tests/test_mcp_server_helpers.py -q` (`44 passed`)
- `python -m pytest tests/test_mcp_agent_runtime.py tests/test_mcp_agent_handler.py tests/test_mcp_server_helpers.py tests/test_mcp_server.py tests/test_mcp_search_handler.py -q` (`78 passed`)
- `python -m pytest tests/test_answer_contracts_runtime.py tests/test_answer_contract_schemas.py tests/test_evidence_contract_perf_gate.py -q` (`32 passed`)
- `python -m pytest tests/test_authority_contract.py -q` (`56 passed`)
- `python scripts/check_public_release_hygiene.py` (`status: ok`, `issues: 0`)
- `python scripts/check_release_smoke.py --mode release` (`release smoke: ok`)
- `git diff --check` (`passed`)

## Follow-up

- Run the broader MCP and answer-contract regressions before pushing this branch.
- Decide whether `agent_get_evidence` should later support persisted answer/evidence lookup instead of deriving evidence by rerunning the local ask path.
- Decide whether a small deterministic agent profile smoke should become part of the frontier gate after the packet contract stabilizes.
