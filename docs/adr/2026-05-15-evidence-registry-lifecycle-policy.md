# ADR: Evidence Registry Lifecycle Policy

Date: 2026-05-15

## Status

Accepted for the evidence-substrate follow-up tranche.

## Context

The evidence registry now resolves explicitly saved packet, context, and trace records for `khub://packet/{id}` and `khub://context/{id}`. Those records are useful only if clients can distinguish saved state from current source state.

The registry must remain a lookup ledger. It must not become canonical evidence, a cleanup daemon, a hidden writeback system, or a replacement for source/document/chunk lineage.

## Decision

Keep registry lifecycle policy narrow:

- Registry records are derived lookup projections.
- Expiry and deletion remove only registry rows.
- Source ledgers, normalized documents, chunks, lexical/vector indexes, cards, claims, and answer logs are not deleted by registry cleanup.
- Current source freshness is a read-time interpretation, not a registry mutation.
- MCP reads stay read-only and must not create, update, prune, or repair registry records.
- Cleanup/list/query CLI remains a later operator surface and is not promoted in this tranche.

Lookup responses separate two concepts:

- `storedStaleness`: the stale/status state stored with the registry row.
- `currentStaleness`: the read-time source revision check when a cheap resolver can inspect current local source hashes.

`currentStaleness.status` may be:

- `fresh`: current source hashes match the saved source refs.
- `stale`: at least one current source hash mismatches the saved source ref.
- `unchecked`: the lookup path did not have enough information to check freshness.
- `source_missing`: the saved source id was present but the current local stores could not resolve a current source hash.

## Consequences

Codex/MCP clients can dereference packet/context ids and tell whether the saved payload is current enough to trust. A registry record can still be useful when current freshness is `unchecked` or `source_missing`, but clients should treat it as a snapshot requiring verification.

Future lifecycle work should add migration/version policy and explicit operator cleanup/list/query commands before any pruning behavior is exposed.
