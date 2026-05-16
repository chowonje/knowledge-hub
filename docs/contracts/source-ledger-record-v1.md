# Source Ledger Record v1

`SourceLedgerRecord` is the source-of-truth contract for the
`Storage / Source Ledger` stage.

It records the canonical source identity and the reconstructable artifact
references that later preparation, indexing, evidence, and audit layers depend
on. `PreparedSourceRecord.ledger_id` should point back to this ledger identity
when a concrete ledger record exists.

Schema:

- `knowledge-hub.source-ledger-record.v1`
- JSON Schema: `docs/schemas/source-ledger-record.v1.json`
- Fixture: `docs/schemas/fixtures/source-ledger-record.v1.fixture.json`

## Required Envelope

- `schema`: fixed schema id.
- `ledger_id`: stable source ledger id.
- `source_id`: canonical source id used by storage/retrieval layers.
- `source_type`: broad source family such as `paper`, `web`, `youtube`, or
  `vault`.
- `canonical_uri`: canonical URL, DOI/arXiv URI, vault URI, or another stable
  source locator.
- `source_content_hash`: hash of the source content used for stale detection.
- `artifacts`: reconstructable raw, normalized, prepared, and indexed artifact
  references.
- `policy`: local-first classification and external-use decision.
- `created_at`: creation timestamp.

## Policy Rule

Ledger policy must fail closed. If a producer has not evaluated a source, the
record uses `classification=UNKNOWN`, `external_allowed=false`, and
`redaction_required=true` rather than pretending the source is public. Source
adapters may set a narrower explicit policy when that decision is owned by the
adapter, for example public arXiv paper metadata as `P3` with external use still
disabled, or vault notes as `P1` with redaction required.

## Invariants

- Ledger records describe source artifacts; they are not retrieval evidence by
  themselves.
- `raw_ref` / `normalized_ref` must be enough for a local operator to rerun
  preparation without rediscovering the source.
- `prepared_ref` points to a persisted `PreparedSourceRecord` when preparation
  has completed.
- Evidence layers must still verify prepared quality, stale state, segment
  locator, and source hash before citing a source.
