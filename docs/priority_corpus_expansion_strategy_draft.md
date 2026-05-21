# Priority Corpus Expansion Strategy (Draft)

Date: 2026-05-21

## Objective

Expand the priority AI paper corpus from the current **100** verified `corpus_manifest.json`
entries toward a **300–500** paper target. The candidate ledger remains metadata-first;
source availability is confirmed only by the join report and manifest validator.

## Current Baseline (Surveyed)

| Source pool | Location | Count | Role in this tranche |
| --- | --- | ---: | --- |
| Local papers registry | `local_operator_registry::papers` | 446 | Primary candidate seed list |
| Public corpus manifest | `eval/knowledgeos/fixtures/corpus_manifest.json` | 100 artifacts | Eval-critical/local-corpus baseline; hash-declared and validator-checked |
| Eval-critical references | `eval/knowledgeos/queries/*.csv`, paper eval fixtures, complex QA seed ids | 55 unique source ids | Tier elevation to `eval_critical` |
| Local PDF inventory hint | `local_operator_papers_dir/*.pdf` | 338 files | Inventory signal only; **not** availability proof |
| Paper memory eval fixture | `tests/fixtures/paper_memory_eval/cases.json` | 3 cases | Eval-critical reference |
| Retrieval span golden cases | `eval/knowledgeos/fixtures/retrieval_span_golden_cases.json` | scanned | Eval-critical reference |

## Candidate Ledger Output

- JSON: `eval/knowledgeos/fixtures/priority_corpus_candidate_ledger.v1.json`
- Markdown summary: `eval/knowledgeos/fixtures/priority_corpus_candidate_ledger.v1.md`

Ledger row count: **446** (within the 300–500 target band).

### Tier mix (446 rows)

| Tier | Count | Meaning |
| --- | ---: | --- |
| `eval_critical` | 55 | Manifest, eval queries, or complex-QA seed references |
| `foundational` | 16 | High-importance or pre-2016 foundational papers in local registry |
| `recent_ai` | 210 | 2024+ papers in local registry |
| `local_corpus_candidate` | 165 | Remaining local registry rows with metadata/pdf hints |

### Manifest overlap

- Already in manifest: **100**
- Not yet in manifest: **346**
- Source verification pending (`unknown` or `join_pending`): **446** (all rows)

## Expansion Path (Tranche Sequence)

### Tranche 0 — Complete (this work)

1. Survey local DB, eval fixtures, and current manifest.
2. Build candidate ledger with metadata and priority tiers.
3. Keep every row at `source_artifact_status: unknown | join_pending`.
4. Do **not** write `available`, do **not** add unverified rows to manifest.

### Tranche 1 — Source artifact join (complete)

Input file: `eval/knowledgeos/fixtures/priority_corpus_candidate_ledger.v1.json`

Latest output:

- Join report: `eval/knowledgeos/reports/priority_corpus_source_join_report.v1.json`
- Expansion allowlist: `eval/knowledgeos/fixtures/priority_corpus_manifest_expansion_allowlist.v1.json`
- Already in manifest: **100**
- Verified expansion allowlist remaining: **251**
- Excluded: **86** `source_missing`, **9** `ambiguous`

For each regeneration:

1. Resolve local source bytes under the configured paper corpus root.
2. Compute SHA-256 and byte length from source bytes.
3. Compare against manifest hash when `current_manifest_status == in_manifest`.
4. Emit join report with per-row status:
   - `available` — bytes present and hash matches
   - `source_missing` — manifest or candidate expects bytes but file absent
   - `hash_missing` — bytes present but no expected hash to compare
   - `hash_mismatch` — bytes present but hash differs; blocks promotion
   - `metadata_only` — title/id only; no bytes

Only rows that pass join with `available` may be considered for manifest registration.

### Tranche 2 — Manifest registration (30–50 rows per batch)

**Applied 2026-05-21:** first batch registered 50 rows (manifest 50 → 100) from
`priority_corpus_manifest_expansion_allowlist.v1.json` first tranche, with local
hash re-verification. Excluded: 86 `source_missing`, 9 `ambiguous`. Remaining
verified allowlist: 251 rows.

For each subsequent batch:

1. Take join-verified `available` rows not yet in manifest.
2. Add manifest entries with safe filename, hash, byte length, provenance URL, license note.
3. Run `khub paper corpus-manifest-validate --json`.
4. Run `khub paper corpus-bootstrap --all --dry-run --json`.
5. Stop if duplicates, hash mismatch, or source-missing rows appear.

Repeat until manifest reaches 300–500 entries with boring validator output.

### Tranche 3 — Parsed derivative coverage (out of scope for ledger tranche)

Track `parsed_missing` separately from source availability. Do not treat parsed
artifacts as source proof.

## Stop Rules

- Never mark a ledger row `available` in this tranche.
- Never register metadata-only or hash-missing rows into `corpus_manifest.json`.
- Never use local absolute paths in public ledger or manifest fields.
- No vault scan, no bulk PDF download, no structured-evidence/parser/runtime work.
- No git commit/push as part of this tranche unless explicitly requested later.

## Source Artifact Verification: Before vs After Join

This tranche intentionally separates **candidate metadata** from **source artifact proof**.

### Before join (this tranche — candidate ledger)

| Field | Allowed values | Meaning |
| --- | --- | --- |
| `current_manifest_status` | `in_manifest`, `not_in_manifest` | Whether the public manifest already lists the source id |
| `source_artifact_status` | `unknown`, `join_pending` | No confirmed source bytes in this tranche |

Interpretation:

- `in_manifest` + `join_pending` means the manifest declares a hash, but this ledger
  tranche did not re-verify bytes. Treat as **not green**.
- `not_in_manifest` + `join_pending` means the local registry has a pdf path hint;
  bytes and hash are still unconfirmed.
- `not_in_manifest` + `unknown` means metadata only; no local artifact signal.

**No row in the candidate ledger is source-available.**

### After join (operator join report)

| Status | Meaning | Manifest action |
| --- | --- | --- |
| `available` | Local bytes exist and SHA-256 matches expected hash | Eligible for retention or new registration |
| `source_missing` | Expected file absent | Block; repair-source or manual lookup |
| `hash_missing` | Bytes exist but no expected hash recorded | Block promotion until hash captured |
| `hash_mismatch` | Bytes exist but hash differs | Block; never green |
| `metadata_only` | Candidate metadata without bytes | Stay out of manifest |

After join, update a **join result artifact** (separate from the candidate ledger).
Only allowlist rows that are still `not_in_manifest` may be used for subsequent
`local_corpus` registration tranches.

## Next Registration Inputs

Primary input:

- `eval/knowledgeos/fixtures/priority_corpus_manifest_expansion_allowlist.v1.json`

Supporting references:

- `eval/knowledgeos/reports/priority_corpus_source_join_report.v1.json`
- `eval/knowledgeos/fixtures/corpus_manifest.json` (existing hash declarations)
- `docs/corpus_manifest_expansion_plan.md` (manifest contract and validator gates)
- Local corpus root from config (`storage.papers_dir`)

Suggested registration priority order:

1. `candidate_tier == recent_ai`
2. `candidate_tier == local_corpus_candidate`
3. Re-run join before each new 30-50 row registration batch
