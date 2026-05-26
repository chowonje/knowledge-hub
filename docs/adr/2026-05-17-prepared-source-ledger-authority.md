# ADR: Prepared Source and Source Ledger Authority

Date: 2026-05-17

## Status

Accepted for the prepared-source-ledger clean tranche.

## Context

The project already distinguishes canonical source content from derived
retrieval, card, packet, and answer artifacts. Web-family indexing still had a
gap: cleaned text, source-content hashes, quality decisions, fallback metadata,
and reconstruction references could be implicit in local rows or transient
indexing code instead of a schema-backed handoff.

That gap makes stale checks, privacy review, and future cross-source
preparation harder to audit. It can also let indexing consume text without a
durable record of which source artifact, parser, fallback route, and quality
gate produced it.

## Decision

Add two schema-backed local JSON contracts:

- `SourceLedgerRecord` for the Storage / Source Ledger stage.
- `PreparedSourceRecord` for the Processing / Preparation stage.

The web-family ingestion path writes both records next to the configured
SQLite store. Indexing prefers fresh, quality-passing prepared records and
falls back to legacy cleaned content only when a prepared record is missing,
stale, invalid, or explicitly non-authoritative.

`khub crawl reindex-approved --prepared-metadata-only` may backfill prepared
metadata onto existing web vector rows, but it must not re-embed. If a
prepared record cannot be paired with a durable source ledger record, the
backfill is reported as partial/failed rather than as prepared-source success.

`khub doctor` / index-freshness diagnostics may report prepared-record vector
coverage and source-specific repair commands, but local path diagnostics must
be redacted in JSON and text output.

## Authority Rules

- Source ledger records describe source artifacts and reconstruction refs.
  They are not evidence spans and are not answer citations.
- Prepared source records are derived preparation artifacts. They are valid
  indexing inputs only when schema, quality, and lifecycle checks pass.
- `source_content_hash` is the stale-detection source hash. Snippet hashes,
  vector ids, and prepared-record ids are not source identity substitutes.
- Ledger policy fails closed by default: unknown classification means external
  use is disabled and redaction is required.
- Prepared records keep source families consistent. A YouTube prepared record
  must write a YouTube ledger record rather than being collapsed to generic
  web.

## Consequences

- Web-family source preparation becomes reproducible from repository-controlled
  contracts and local artifacts instead of transient runtime state.
- Index freshness can explain missing source-id coverage and missing prepared
  metadata without leaking local file paths.
- The prepared/source-ledger layer adds persistence files and schema tests, so
  future source adapters must preserve the same fail-closed authority rules.
- Non-web producer writes are intentionally not promoted in this tranche.
  Cross-source enforcement remains a follow-up decision.

## Follow-Ups

1. Decide whether paper and Obsidian producers should emit the same contracts.
2. Add live-corpus operator checks for prepared-record coverage after the
   metadata-only web backfill has run on a real local corpus.
3. Revisit doctor/index-freshness cost if large vector corpora make the
   coverage scan too heavy for routine diagnostics.
