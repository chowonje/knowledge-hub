# Personal Foundry Implementation Plan (Current Repo)

Date: 2026-02-28
Scope: `foundry-core` in the current `knowledge-hub` repository

## 1) Goal

Build a practical, local-first "personal Foundry-like" runtime with this flow:

`connectors -> event bus -> ontology store -> feature layer -> agent runtime -> apps/ui`

The implementation in this plan is incremental and keeps compatibility with the current `knowledge-hub` bridge.

## 2) Core Principles

- Ontology-first
  - Personal data is modeled as `entity / relation / event`.
  - Event sourcing is the ground truth.
  - Snapshots and time series are derived projections.
- Local-first privacy
  - P0 raw sensitive data stays local.
  - External model/API calls are optional and only receive P1/P2/P3 sanitized payloads.
- Security by default
  - Policy engine and audit logs run on every critical action.
  - Classification levels: `P0 raw sensitive`, `P1 structured facts`, `P2 summary`, `P3 public`.
- Agent runtime safety
  - `Plan -> Act -> Verify -> Writeback`.
  - Verify must include policy and schema checks before writeback.
- Connector contract
  - SDK shape: `authorize()`, `sync()`, `mapToOntology()`, `emitEvents()`.
  - Must support idempotency and incremental sync.

## 3) What Is Implemented In This Iteration

This iteration adds a clear skeleton under `foundry-core/src/personal-foundry`:

- contracts/interfaces for each layer
- file-backed local event bus (append/read/subscribe)
- file-backed local audit log (append/query)
- local policy engine with classification and outbound P0 deny
- local ontology store (events, entities, relations, snapshots, time series)
- connector runner with idempotency + incremental cursor state
- agent runtime skeleton with `Plan -> Act -> Verify -> Writeback`
- `cli-agent` wiring: `run` and `sync` paths use the personal-foundry runtime/runner bridge
- `sync` path now appends mapped ontology batches into local ontology store
- `feature` path now defaults to ontology event source (`ontology-store/ontology.events.jsonl`)
- standalone project CLI added: `init`, `status`, `pipeline` (dry-run compatible)

## 4) Phased Delivery Plan

1. Phase 1: Local core skeleton
   - Build the modules above with tests.
   - Keep existing CLI bridge untouched.
2. Phase 2: Connector migration
   - Wrap existing `KnowledgeHubConnector` with the new runner.
   - Persist cursor state and idempotency for each source.
3. Phase 3: Feature alignment
   - Feature functions read from ontology/snapshot/time series, not ad-hoc payloads.
   - Add provenance on feature outputs.
4. Phase 4: Agent hardening
   - Enforce schema registry for writeback artifacts.
   - Add policy reason codes + stronger audit linkage by runId.
5. Phase 5: App/UI integration
   - Keep local API server minimal.
   - Add operator view for runs, policies, and audit traces.

## 5) Assumptions

- Single-user local environment is primary (multi-tenant can be added later).
- Existing `foundry-core` runtime/CLI behavior should remain compatible.
- File-backed storage is sufficient for MVP and local reliability.
- External LLM calls are optional and disabled by default in local mode.

## 6) Unclear Points and 2 Alternatives

### Unclear point A: Primary runtime boundary

- Alternative A (recommended): TypeScript runtime as control plane
  - Keep `foundry-core` as orchestrator and policy boundary.
  - Use `knowledge-hub` as connector/tool provider.
- Alternative B: Python-first runtime with thin TypeScript facade
  - Keep most orchestration in Python and minimize TS runtime logic.

Decision for this iteration: Alternative A.

### Unclear point B: Persistent store backend

- Alternative A (recommended for now): JSONL + local files
  - Fast iteration, easy debugging, local portability.
- Alternative B: SQLite-native event/snapshot/time-series tables
  - Better queryability and stronger consistency in one DB.

Decision for this iteration: Alternative A, with interfaces designed for later SQLite swap.

## 7) Mapping To Existing Files

- New skeleton root:
  - `foundry-core/src/personal-foundry/`
- Existing bridge left intact:
  - `foundry-core/src/cli-agent.ts`
  - `foundry-core/src/connector-sdk.ts`
  - `foundry-core/src/runtime.ts`

## 8) Completion Criteria

- All new modules exist and are test-covered for critical behavior:
  - P0 outbound deny
  - connector idempotency
  - incremental cursor update
  - agent verify gate before writeback
- Existing tests remain green.
