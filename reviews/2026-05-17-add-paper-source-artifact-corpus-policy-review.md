# Review: Add paper source artifact corpus policy

## Findings

- Code review found four issues and this tranche addressed them before final verification:
  - `repo_fixture` artifacts now resolve from manifest-relative fixture paths, with a passing fixture test.
  - Corpus diagnostics now expose safe path refs rather than personal absolute paths in durable eval/report payloads.
  - `optional_local_corpus` misses no longer block case execution.
  - The corpus manifest schema is validated and unsupported schemas fail closed.
- Commit-readiness review then found that required local-corpus skips could still yield a green gate if the remaining evaluable rows passed. The live compare gate now defaults to full corpus coverage and fails below that threshold unless explicitly lowered; focused tests, full pytest, release smoke, public hygiene, and wide live compare passed after this fix.
- Follow-up review found that the ignored wide live compare case file had no explicit `corpusRequirements`, which made the new coverage gate depend on manual operator-local edits. The eval CLI now derives requirements from `expected_source_ids` and the manifest, and missing manifest mappings fail visibly as `missingCorpusRequirements`.
- No hidden DB writes, network/provider calls, registry writes, MCP changes, or strict-evidence gate weakening were found in the final reviewed scope.
- The provider help blocker was resolved by aligning the regression with the existing compact public help policy: `provider` stays hidden from root help and remains documented through `khub help advanced`.

## Risks

- The manifest records hashes derived from the current local corpus; future paper revisions must be handled as explicit hash updates rather than silently accepting drift.
- Live compare coverage skipping must not be advertised as a full `15/15` without the declared/evaluable coverage counts.
- A derived requirement only proves local artifact presence and hash identity. Strict answerability still depends on source-content hashes and `chars:start-end` spans in the compare packet.
- This tranche intentionally does not solve artifact acquisition; operators still need the local corpus files.
- Provider CLI help remains a hidden/operator surface. This is intentional, but future public surface changes should update `khub --help`, `khub help advanced`, tests, and docs together.
- No full-test blocker remains after the provider help test alignment; `pytest -q` passed.

## Missing Tests

- Focused tests cover missing-artifact and hash-mismatch repair diagnostics, no-write behavior, live compare corpus skip/coverage metrics, passing repo fixtures, optional local corpus behavior, safe corpus path refs, and unsupported manifest schema rejection.
- Additional live compare tests cover manifest-derived requirements from expected source ids, default gate failure for missing derived local artifacts, and hard failure for expected source ids without manifest mappings.
- Provider surface tests now cover compact root help plus advanced hidden/operator inventory instead of expecting `provider` in default help.
