# Research Review Loop MVP Plan

Date: 2026-06-10

## Objective

Prove a small, user-visible research loop before expanding answer generation,
graph/ontology features, or onboarding:

```text
paper -> proposed claims -> evidence spans -> user review
      -> weak concepts / open questions -> learning or judgment context pack
```

## Constraints

- Labs-first and explicit-source only.
- No default `khub ask` promotion.
- No vault scan or vault write.
- No vector, Chroma, DB, or index mutation in the first report-only pass.
- No external-agent canonical writes.
- Proposed artifacts cannot become memory until a user review decision exists.
- Reuse existing `claims`, `claim-cards`, `review-card`, and `context-pack`
  surfaces before adding new persistence.

## Current Reusable Surfaces

- `khub labs claims extract-paper --paper-id <id> --json`
- `khub labs claims pending list --json`
- `khub labs claims strict-report --paper-id <id> --out-dir <dir> --json`
- `khub labs claims normalize --paper-id <id> --json`
- `khub labs claims compare --paper-id <id> --json`
- `khub labs claims synthesize --paper-id <id> --json`
- `khub labs claim-cards metrics --source paper --json`

These are inputs and diagnostics, not a complete user-review loop.

## Proposed Report Contract

The first implementation should emit
`knowledge-hub.research-review-loop.result.v1` with:

- explicit source scope
- proposed claim cards
- evidence spans or blocked evidence rows
- review decisions
- weak concepts
- open questions
- a learning or judgment context pack preview
- counters that prove unreviewed proposed artifacts were excluded from
  canonical context

The draft schema is `docs/schemas/research-review-loop-result.v1.json`.

## First Human Test

Use one operator-supplied paper id or a very small explicit paper set.

Done condition:

- `sourceCount >= 1`
- `proposedClaimRows` between 3 and 7 for the first paper slice
- `unsupportedCanonicalRows = 0`
- at least three review decisions captured
- weak concepts and open questions are derived only from rejected or unsure
  decisions, blocked evidence rows, or missing evidence
- final pack preview includes reviewed claim ids and excludes unreviewed
  proposed claim ids

## Stop Rules

Stop and keep the report blocked when:

- source scope is implicit or broad
- evidence spans are missing for the main claim
- review decisions are absent
- an external report attempts to write canonical memory
- unreviewed proposed artifacts appear in the canonical pack
- the output requires a vault scan, vector mutation, or answer-path promotion

## Next Implementation Slice

Add a report-only builder behind a labs command, for example:

```text
khub labs review-loop report --paper-id <id> --out-dir <dir> --json
```

The command should compose existing claim/strict-report outputs first. A later
apply slice can add explicit review-decision writes after the report contract is
reviewed.
