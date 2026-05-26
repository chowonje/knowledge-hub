# Prepared Source Record v1

`PreparedSourceRecord` is the common output contract for the
`Processing / Preparation` stage.

It is intentionally source-agnostic: paper parsers, web cleanup, YouTube
transcripts, vault/document-memory preparation, and future source-specific
preparers should all be able to emit this shape before indexing or evidence
assembly consumes the result.

Schema:

- `knowledge-hub.prepared-source-record.v1`
- JSON Schema: `docs/schemas/prepared-source-record.v1.json`
- Fixture: `docs/schemas/fixtures/prepared-source-record.v1.fixture.json`

## Boundary

This contract sits after source intake has resolved a canonical source identity
and fetched/extracted candidate content, but before indexing, card building,
evidence assembly, or answer generation.

The contract answers seven questions:

- What source did this prepared material come from?
- Which source ledger/raw artifact can reconstruct it?
- Which source-content hash controls stale detection?
- What cleaned text and segment locators are safe to index?
- Which processor/parser/extraction route produced it?
- Did fallback happen, and why?
- Did quality gates pass strongly enough for index handoff?

## Required Envelope

- `schema`: fixed schema id.
- `record_id`: stable prepared-record id.
- `source_id`: canonical source id used by storage/retrieval layers.
- `source_type`: broad source family such as `paper`, `web`, `youtube`, or
  `vault`.
- `canonical_uri`: canonical URL, DOI/arXiv URI, vault URI, or another stable
  source locator.
- `source_content_hash`: hash of the source content that downstream derivatives
  must compare for stale detection.
- `ledger_id` and `raw_ref`: source ledger id plus the raw or normalized
  artifact reference that can reconstruct the prepared text.
- `storage_ref`: prepared-record artifact path when the producer persists this
  envelope as JSON.
- `prepared`: cleaned text plus segment locators.
- `processing`: processor/parser route, fallback chain, diagnostics, warnings,
  and errors.
- `quality`: pass/fail score and quality flags.
- `lifecycle`: stale state for the prepared record itself.
- `created_at`: creation timestamp.

## Storage Rule

The web-family adapter persists each prepared record as JSON under the runtime
store next to the SQLite database:

- cleaned markdown: `<sqlite parent>/web_docs/`
- optional raw capture: `<sqlite parent>/web_raw/`
- prepared-source JSON: `<sqlite parent>/prepared_sources/<source_type>/<record_id>.json`

`raw_ref` points to the reconstructable cleaned/raw source artifact used by the
prepared record. `storage_ref` points to the prepared-record JSON itself.
Public ingest summaries expose prepared-record counts and ids, not local
filesystem paths. Note metadata stores `prepared_record_path` so later indexing,
audits, and stale checks can find the preparation output without rerunning
extraction.

## Handoff Rule

Indexing must only consume records whose `schema` is
`knowledge-hub.prepared-source-record.v1`, whose `quality.passed` is true, and
whose `lifecycle.stale` is false.

For web-family ingest, indexing resolves either the in-memory
`prepared_source_record` or the persisted `prepared_source_record_path` before
chunking. When a prepared record is available, `prepared.text` and
`source_content_hash` are the indexing inputs of record, and vector metadata
carries the prepared record id/path/text hash so later retrieval audits can trace
chunks back to the preparation artifact. Rows without a prepared record remain
on the legacy cleaned-content path for compatibility during migration.
Unreadable, unknown-schema, failed-quality, or stale prepared records also fall
back to the legacy row instead of overriding indexed content.
For persisted YouTube records, prepared segment locators with timestamps are
rehydrated into transcript-style segments before chunking so retry/reindex paths
keep timestamp windows.

Failed or low-quality preparation can still be represented by the contract for
audit/debugging, but it should not be silently promoted into retrieval material.

## Segment Rule

Segments are not citation-grade evidence by themselves. They are locator-bearing
prepared spans that make later evidence reconstruction possible.

Evidence assembly must still build source-backed evidence packets and enforce
its own policy, stale, and citation-grade checks.

The first deterministic promotion gate is implemented by
`prepared_segment_evidence_gate(...)`: a prepared segment is only eligible as an
evidence candidate when the prepared record passed quality, is not stale, carries
`source_id` and `source_content_hash`, and the segment has text, char span, and a
locator with a source reference. This gate marks eligibility; it does not bypass
the later `EvidencePacket` / `AnswerContract` checks.

## Migration Rule

Adoption should be additive:

- keep existing paper/web/vault/YouTube builders intact
- add thin adapters that emit `PreparedSourceRecord`
- preserve source-specific chunking details such as YouTube timestamps
- validate representative fixtures first
- only then move shared indexing/search handoff code onto the common contract
