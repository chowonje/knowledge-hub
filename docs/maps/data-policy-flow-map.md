# Data + Policy Flow Map

This map shows where data lives, which context is persistent versus ephemeral,
and where policy gates apply.

Diagram source: `data-policy-flow-map.mmd`

## Reading Guide

- Vault, paper, and web knowledge are persistent local sources.
- Repo/project context is read-only and ephemeral by default.
- External notebook systems are adapters, not canonical stores.
- Policy is layered:
  - outbound provider policy guard
  - Foundry runtime policy at ACT, VERIFY, and WRITEBACK
  - MCP job-layer verify and persistence gate

## Classification Model

- `P0`
  - raw sensitive or non-sanitized content that must stay local by default
- `P1`
  - structured facts and bounded records
- `P2`
  - summaries and synthesized outputs
- `P3`
  - public or low-sensitivity material

The main hard rule is that outbound provider calls must keep `P0` blocked unless
the payload has already been transformed into an allowed sanitized form.

## Persistent vs Ephemeral

Persistent local stores:

- SQLite metadata and job/state records
- vector DB collections
- parser artifacts
- note, paper, and web source records

Ephemeral by default:

- repo/project context
- workspace file snippets in task-context assembly
- bounded runtime prompt context built for coding/design/debug assistance

Non-canonical consumers:

- external notebook and workbench integrations
- exported artifacts derived from the local runtime

## Policy and Audit Implications

- Outbound provider adapters enforce a second-line policy guard before external
  model calls.
- Foundry runtime applies policy before tool act, during verify, and before
  writeback.
- MCP async jobs re-check schema and policy before persisting final artifacts.
- Auditability is strongest when run payloads remain inspectable and repo
  context stays out of canonical persistent knowledge stores.

## Provenance

Primary docs:

- `docs/ARCHITECTURE.md`
- `docs/PROJECT_STATE.md`
- `docs/foundry-knowledge-hub-integration.md`

Primary implementation references:

- `knowledge_hub/application/task_context.py`
- `knowledge_hub/application/context_pack.py`
- `knowledge_hub/providers/policy_guard.py`
- `knowledge_hub/application/mcp/jobs.py`
- `knowledge_hub/core/mcp_job_store.py`
- `foundry-core/src/personal-foundry/agent-runtime.ts`
- `foundry-core/src/personal-foundry/policy-engine.ts`
