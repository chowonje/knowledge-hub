# ADR: Evidence Registry Authority

Date: 2026-05-15

## Status

Accepted for the evidence-substrate contract worktree.

## Context

The Codex/MCP evidence-substrate surface now exposes stable URI shapes such as `khub://packet/{id}` and `khub://context/{id}`. Returning `not_found` was acceptable while the project only froze the URI contract, but durable Codex handoff needs those ids to resolve when a packet or context pack has been explicitly saved.

The risk is that a packet/context registry can accidentally become a second source of truth. Evidence still has to come from source ledgers, normalized documents, chunks, and source-backed spans. Embeddings remain retrieval artifacts, and answer/context payloads remain derived projections.

## Decision

Add a single SQLite-backed `evidence_registry_records` table as the lookup authority for persisted packet, context, and trace records.

Registry records are derived lookup projections. Each row stores:

- `record_kind`: `packet`, `context`, or `trace`
- `registry_id`: the URI id
- payload schema, payload hash, and payload snapshot
- source refs and aggregate source revision hash
- lineage across source, document, chunk, claim, evidence span, and citation labels
- token count, expiry, stale marker, stale reason, and deletion policy
- an authority block that states the payload and embeddings are not source truth

Writes are explicit only:

- `khub trace --save-registry`
- `khub compare --save-registry`
- application helper calls for `ContextPack` records

Default CLI/MCP reads may resolve existing registry rows, but they must not create packet/context records or call external providers.

## Invariants

- `khub://packet/{id}` and `khub://context/{id}` return stable `not_found` when no registry row exists.
- Registry payloads must carry lineage and source revision metadata.
- A source revision mismatch marks a lookup result stale instead of silently trusting the saved payload.
- Deleting or pruning a registry record removes only the lookup projection. It must not delete sources, chunks, vector rows, lexical rows, cards, claims, or answer logs.
- The registry does not promote claim cards, graph/memory rows, or ontology projections into citation endpoints.

## Consequences

- Codex/MCP clients can dereference saved packet/context ids without requiring a live answer run.
- The registry adds a small persistence model, so schema and lifecycle tests are required.
- Saved payloads can still go stale; callers must inspect `status`, `staleReason`, `sourceRefs`, and `sourceRevisionHash` before treating a lookup as current.

## Follow-Ups

1. Add live source-revision checks for MCP registry reads when a cheap current-source resolver is available.
2. Add stricter eval cases for inspect/compare/context/trace registry behavior.
3. Decide whether ContextPack registration should get an explicit public CLI option after usage proves the helper path is not enough.
