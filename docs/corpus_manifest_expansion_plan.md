# Corpus Manifest Expansion Plan

Date: 2026-05-21

## Scope

This plan covers the paper source artifact layer used by the local corpus
manifest, repair-source, corpus-bootstrap, live compare coverage gates, and
complex-paper QA seed selection.

It does not cover structured evidence extraction, SourceSpan promotion,
runtime answer integration, parser routing, indexing, re-embedding, vault
content, or external PDF acquisition.

## Current Contract

- Public manifest: `eval/knowledgeos/fixtures/corpus_manifest.json`
- Schema id: `knowledge-hub.corpus-manifest.v1`
- Entry authority:
  - `artifactId`: stable source artifact id, currently `paper_<source-id>` for
    arXiv-style ids with dots converted to underscores.
  - `sourceIds`: canonical paper ids that repair/eval routes match against.
  - `expectedFilename`: safe filename under the configured local paper corpus.
  - `expectedSourceContentHash`: SHA-256 of the source bytes.
  - `byteLength`: source byte length observed with the hash.
  - `provenanceUrl`: public origin locator for operator reacquisition.
  - `license`: redistribution note; source bytes are not committed.
  - `corpusTier`: `local_corpus`, `optional_local_corpus`, or `repo_fixture`.

The manifest is metadata only. It is not a source store, prepared-source store,
evidence store, parsed artifact, or answerability signal by itself.

## First Tranche (19 → 50)

This tranche expanded the public manifest from 19 to 50 entries. The 31 added
entries were included only because a local source PDF existed, SHA-256 was
computed from the source bytes, byte length was recorded, and the public entry
needed no local absolute path.

No metadata-only or source-missing rows were added as available source
artifacts. Entries that lack source bytes, a known source hash, or a matching
local artifact must stay out of `local_corpus` availability until a later
operator verification step.

## Second Tranche (50 → 100, verified allowlist)

Date applied: 2026-05-21

Input artifacts:

- `eval/knowledgeos/fixtures/priority_corpus_manifest_expansion_allowlist.v1.json`
- `eval/knowledgeos/reports/priority_corpus_source_join_report.v1.json`

Added exactly 50 rows from the join report first-tranche recommendation, in
tier order (`eval_critical` → `foundational` → `recent_ai` → `local_corpus_candidate`).
Each row was re-verified locally for source bytes, SHA-256, and byte length before
manifest registration.

Explicitly excluded from this tranche:

- allowlist rows outside the first tranche 50
- join-report `source_missing` (86)
- join-report `ambiguous` (9)
- hash-missing or hash-mismatch rows

Remaining verified allowlist rows after this tranche: 251.

## Third Tranche (100 -> 150, verified allowlist)

Date applied: 2026-05-21

Input artifact:

- `eval/knowledgeos/reports/corpus_manifest_100_to_150_dry_run_plan.v1.json`

Added exactly 50 rows from the dry-run plan after every selected row was
rechecked against configured local source bytes. The applied rows had no
already-in-manifest collisions, source-missing rows, ambiguous rows, hash-missing
rows, or hash mismatches.

Post-apply join/allowlist regeneration records:

- already in manifest: 150
- verified expansion allowlist remaining: 201 before recovered-source scan expansion
- excluded before recovered-source scan expansion: 86 `source_missing`, 9 `ambiguous`
- hash missing: 0
- hash mismatch: 0

The subsequent recovered-source scan-root correction included
`papers_dir/recovered_sources/arxiv` and `papers_dir/recovered_sources/url` in
the report-only source inventory and the shared artifact resolver used by
`corpus-manifest-validate` and `corpus-bootstrap`. The current regenerated join
now records:

- already in manifest: 150
- available: 396
- verified expansion allowlist remaining: 246
- excluded: 41 `source_missing`, 9 `ambiguous`
- hash missing: 0
- hash mismatch: 0

Next registration tranche should repeat the same gate: allowlist slice only, local hash
re-verification, `corpus-manifest-validate`, and `corpus-bootstrap --all --dry-run`.

## Fourth Tranche (150 -> 200, verified allowlist)

Date applied: 2026-05-21

Input artifacts:

- `eval/knowledgeos/fixtures/priority_corpus_manifest_expansion_allowlist.v1.json`
- `eval/knowledgeos/reports/corpus_manifest_150_to_200_tranche_plan.v1.json`

Added exactly 50 rows via `eval/knowledgeos/scripts/build_corpus_manifest_expansion_tranche.py --target-count 200 --apply`.
The tranche planner selected only allowlist rows not already present in
`corpus_manifest.json`, recomputed local source SHA-256 and byte length from
`papers_dir/...` refs, and checked that the shared artifact resolver can find
the proposed manifest entry before writing.

Post-apply join/allowlist regeneration records:

- already in manifest: 200
- available: 396
- verified expansion allowlist remaining: 196
- excluded: 41 `source_missing`, 9 `ambiguous`
- hash missing: 0
- hash mismatch: 0

Next registration tranche should run the same script with `--target-count 250`
after the current allowlist has been regenerated against the 200-row manifest.

## Two Additional Batches (200 -> 300, verified allowlist)

Date applied: 2026-05-21

Input artifacts:

- `eval/knowledgeos/reports/corpus_manifest_200_to_250_tranche_plan.v1.json`
- `eval/knowledgeos/reports/corpus_manifest_250_to_300_tranche_plan.v1.json`

Applied two additional 50-row batches with the same gate:

```bash
python eval/knowledgeos/scripts/build_corpus_manifest_expansion_tranche.py --target-count 250
python eval/knowledgeos/scripts/build_corpus_manifest_expansion_tranche.py --target-count 250 --apply
python eval/knowledgeos/scripts/build_priority_corpus_source_join_report.py
python eval/knowledgeos/scripts/build_corpus_manifest_expansion_tranche.py --target-count 300
python eval/knowledgeos/scripts/build_corpus_manifest_expansion_tranche.py --target-count 300 --apply
python eval/knowledgeos/scripts/build_priority_corpus_source_join_report.py
```

Both tranche plans recorded:

- selected rows: 50
- ready rows: 50
- blocked rows: 0
- already-in-manifest skipped rows: 0

Current post-apply join/allowlist records:

- already in manifest: 300
- available: 396
- verified expansion allowlist remaining: 96
- excluded: 41 `source_missing`, 9 `ambiguous`
- hash missing: 0
- hash mismatch: 0

The verified-only lower bound of the 300-500 target is now met. Reaching the
upper end of the target still requires new candidates and/or resolving held
source-missing and ambiguous rows.

## Validation Gate

Use the hidden report-only validator:

```bash
khub paper corpus-manifest-validate --json
```

The validator checks:

- duplicate `artifactId` and duplicate `sourceIds`
- missing or malformed `expectedSourceContentHash`
- source artifact presence under the configured local paper corpus
- observed source hash versus manifest hash
- parsed artifact presence as a separate status from source availability
- no network, DB/index mutation, vault scan, source registration write, parsed
  artifact write, or evidence promotion

Important status split:

- `available`: source artifact exists and hash matches
- `metadata_only`: metadata exists but no source artifact/hash is declared
- `source_missing`: manifest declares a source artifact but the local file is
  absent
- `hash_missing`: source artifact cannot be treated as verified because the
  expected source hash is absent
- `hash_mismatch`: local bytes do not match the manifest hash and must block
  promotion
- `parsed_missing`: source availability may be valid, but the parsed derivative
  is absent and must not be conflated with source-missing

## 300-500 Entry Expansion Path

1. Build a candidate list from the paper registry or curated source lists.
   Candidate rows may include title, source id, public provenance URL, filename
   hint, and priority reason, but they are not available corpus entries yet.
2. For each candidate, verify source bytes locally and compute SHA-256 plus byte
   length. If the source bytes are absent, classify the row as `source_missing`
   or `metadata_only` in an operator report, not as available manifest state.
3. Register only source-verified rows in the public manifest. Keep local
   absolute paths out of the manifest and use safe filename refs only.
4. Run `khub paper corpus-manifest-validate --json` and
   `khub paper corpus-bootstrap --all --dry-run --json` after each tranche.
5. Keep tranche size at 30-50 rows until the validator stays boring: no hash
   mismatch, no duplicate ids, no source-missing rows accidentally registered
   as available, and parsed-missing reported separately.
6. Only after source coverage is stable should structured evidence vertical
   slices consume the expanded corpus.

## Stop Rules

- Do not mark metadata-only rows as source-available.
- Do not accept hash mismatch as green.
- Do not use parsed artifacts as source artifact proof.
- Do not put private local absolute paths in public docs or manifest entries.
- Do not use this manifest expansion as permission for PDF download,
  parser-routing, SourceSpan, strict evidence, runtime answer, indexing, or
  vault work.
