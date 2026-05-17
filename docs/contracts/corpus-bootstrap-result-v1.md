# Corpus Bootstrap Result v1

`knowledge-hub.corpus-bootstrap.result.v1` is the hidden operator result
contract for explicit local paper corpus acquisition.

It is not a source-of-truth contract. The source artifact policy remains the
corpus manifest plus the configured local `papers_dir`; this payload only
reports what the operator helper planned, downloaded, skipped, or blocked.

## Boundary

`khub paper corpus-bootstrap` may read the corpus manifest and selected
`local_corpus` / `optional_local_corpus` entries. It may download from
`provenanceUrl` only when both `--apply` and `--allow-network` are present.

The helper must not:

- run during eval, repair-source, import, or CI by default
- write SQLite rows
- attach paper source paths
- rebuild paper memory, document memory, cards, or indexes
- overwrite an existing hash-mismatched artifact automatically
- download `repo_fixture` artifacts

## Required Envelope

- `schema`: fixed schema id.
- `generatedAt`: UTC timestamp.
- `status`: `ok` when all selected items are successful/planned/skipped,
  otherwise `blocked`.
- `dryRun`: true unless `--apply` was used.
- `networkAllowed`: true only when `--allow-network` was used.
- `manifestRef`: non-sensitive manifest reference.
- `selectedCount`: number of manifest entries selected.
- `counts`: summary counts for downloaded, already-present, planned, skipped,
  and blocked entries.
- `selectionErrors`: selector misses such as an unknown artifact id/source id.
- `items`: one result object per selected manifest entry.

## Item Statuses

Successful or non-mutating statuses:

- `already_present`: local artifact exists and matches manifest hash.
- `planned_download`: dry-run found a missing artifact that could be acquired.
- `downloaded`: helper downloaded and verified the artifact before promotion.
- `skipped_repo_fixture`: entry is repo/CI-owned and is not acquired.

Blocked statuses:

- `papers_dir_unconfigured`
- `unsupported_corpus_tier`
- `missing_expected_hash`
- `missing_filename`
- `missing_provenance_url`
- `unsupported_provenance_url`
- `network_not_allowed`
- `download_failed`
- `hash_mismatch`
- `byte_length_mismatch`
- `post_write_verification_failed`

Blocked statuses must not be treated as green release signals.

## Diagnostics

Paths in payload items must use safe refs such as `papers_dir/<filename>` or
`repo_fixture/<path>`, not local absolute paths. Download mismatch payloads may
include observed hash and byte length so operators can decide whether the
manifest or the local file is wrong.

`repairHints` may include `khub paper repair-source --paper-id ... --dry-run
--json` commands after an artifact is present or planned. These are hints only;
the bootstrap helper does not invoke repair or rebuild derivatives.
