# Checkpoint Manifests

This directory defines a non-destructive split plan for the current dirty
`knowledge-hub` worktree.

Goals:
- keep the green-baseline stabilization slice narrow and reproducible
- stage or inspect later buckets without rewriting git history
- surface carryover paths explicitly instead of hiding them in a giant diff

Canonical bucket order lives in `checkpoints.json`.

Runtime rules:
- Buckets are evaluated in order.
- The first matching bucket wins.
- `scripts/report_checkpoint_split.py` classifies the live `git status` against
  `checkpoints.json`.
- `--write-pathspec-dir` exports exact current changed paths per bucket, which is
  safer for staging than broad globs on a dirty branch.
- The committed `*.pathspec` files are static review manifests. They are useful
  for understanding intended boundaries, but the exported exact pathspec files
  are the safer staging primitive.

Recommended workflow:

```bash
# inspect the current split
python scripts/report_checkpoint_split.py

# inspect the bucket paths in text mode
python scripts/report_checkpoint_split.py --show-paths

# inspect the full machine-readable report
python scripts/report_checkpoint_split.py --json

# export exact bucket pathspec files for this worktree
python scripts/report_checkpoint_split.py --write-pathspec-dir /tmp/khub-checkpoints

# stage only the current green-baseline slice
git add --pathspec-from-file=/tmp/khub-checkpoints/01-contract-entrypoints.pathspec
```

Checkpoint order:
- `01-contract-entrypoints`
- `02-cli-mcp-surface-migration`
- `03-infrastructure-core-ownership`
- `04-paper-public-reading`
- `05-retrieval-ask-v2`
- `06-ingest-learning-vault-web`
- `07-foundry-satellites`

Additional tranche manifests may live next to those canonical buckets when a
stabilization slice cuts across multiple buckets. The current auxiliary
manifests are:

- `00-replay-control-plane.pathspec`: replay helper + checkpoint definitions +
  dirty-tree split notes only
- `09-core-loop-stabilization.pathspec`: the frozen representative weekly
  core-loop payload to replay from the clean sibling
- `11-weekly-ask-ci-release-hygiene.pathspec`: the `ask` smoke promotion plus
  hermetic CI/public-release overlay to replay on top of the frozen `09` slice
- `12-opus-architecture-hardening.pathspec`: the evidence-first answer-contract,
  derivative lifecycle, source-hash/staleness, RAG compatibility, policy-gate,
  Foundry self-protection, and clean-replay closure overlay

Representative replay flow:

```bash
git worktree add ../knowledge-hub-replay-control-plane -b tranche/2026-04-17-replay-control-plane HEAD
python scripts/sync_worktree_slice.py \
  --source-repo-root . \
  --target-repo-root ../knowledge-hub-replay-control-plane \
  --pathspec-file ops/checkpoints/00-replay-control-plane.pathspec \
  --stage --json

git worktree add ../knowledge-hub-core-loop-09 -b tranche/2026-04-17-core-loop-09 HEAD
python scripts/sync_worktree_slice.py \
  --source-repo-root ../knowledge-hub-core-loop-stabilization-head-v11 \
  --target-repo-root ../knowledge-hub-core-loop-09 \
  --pathspec-file ops/checkpoints/09-core-loop-stabilization.pathspec \
  --stage --json

git worktree add ../knowledge-hub-core-loop-ask-ci -b tranche/2026-04-17-core-loop-ask-ci HEAD
python scripts/sync_worktree_slice.py \
  --source-repo-root ../knowledge-hub-core-loop-stabilization-head-v11 \
  --target-repo-root ../knowledge-hub-core-loop-ask-ci \
  --pathspec-file ops/checkpoints/09-core-loop-stabilization.pathspec \
  --stage --json
python scripts/sync_worktree_slice.py \
  --source-repo-root . \
  --target-repo-root ../knowledge-hub-core-loop-ask-ci \
  --pathspec-file ops/checkpoints/11-weekly-ask-ci-release-hygiene.pathspec \
  --allow-dirty-target --stage --json
```

Opus architecture-hardening clean replay:

```bash
git worktree add ../knowledge-hub-opus-hardening-clean -b opus-architecture-hardening-clean-20260423 HEAD

# Apply the exact exported/static stack, not only the final overlay:
# 01-07 canonical buckets, then 00, 09, 10, 11, and 12.
python scripts/sync_worktree_slice.py --source-repo-root ../knowledge-hub --target-repo-root ../knowledge-hub-opus-hardening-clean --pathspec-file ops/checkpoints/01-contract-entrypoints-green-baseline.pathspec --stage --json
python scripts/sync_worktree_slice.py --source-repo-root ../knowledge-hub --target-repo-root ../knowledge-hub-opus-hardening-clean --pathspec-file ops/checkpoints/02-cli-mcp-surface-migration.pathspec --allow-dirty-target --stage --json
python scripts/sync_worktree_slice.py --source-repo-root ../knowledge-hub --target-repo-root ../knowledge-hub-opus-hardening-clean --pathspec-file ops/checkpoints/03-core-infra-ownership-transition.pathspec --allow-dirty-target --stage --json
python scripts/sync_worktree_slice.py --source-repo-root ../knowledge-hub --target-repo-root ../knowledge-hub-opus-hardening-clean --pathspec-file ops/checkpoints/04-paper-public-reading.pathspec --allow-dirty-target --stage --json
python scripts/sync_worktree_slice.py --source-repo-root ../knowledge-hub --target-repo-root ../knowledge-hub-opus-hardening-clean --pathspec-file ops/checkpoints/05-retrieval-ask-v2.pathspec --allow-dirty-target --stage --json
python scripts/sync_worktree_slice.py --source-repo-root ../knowledge-hub --target-repo-root ../knowledge-hub-opus-hardening-clean --pathspec-file ops/checkpoints/06-learning-vault-web-ingest.pathspec --allow-dirty-target --stage --json
python scripts/sync_worktree_slice.py --source-repo-root ../knowledge-hub --target-repo-root ../knowledge-hub-opus-hardening-clean --pathspec-file ops/checkpoints/07-foundry-satellites-and-ops.pathspec --allow-dirty-target --stage --json
python scripts/sync_worktree_slice.py --source-repo-root ../knowledge-hub --target-repo-root ../knowledge-hub-opus-hardening-clean --pathspec-file ops/checkpoints/00-replay-control-plane.pathspec --allow-dirty-target --stage --json
python scripts/sync_worktree_slice.py --source-repo-root ../knowledge-hub --target-repo-root ../knowledge-hub-opus-hardening-clean --pathspec-file ops/checkpoints/09-core-loop-stabilization.pathspec --allow-dirty-target --stage --json
python scripts/sync_worktree_slice.py --source-repo-root ../knowledge-hub --target-repo-root ../knowledge-hub-opus-hardening-clean --pathspec-file ops/checkpoints/10-ask-runtime-first.pathspec --allow-dirty-target --stage --json
python scripts/sync_worktree_slice.py --source-repo-root ../knowledge-hub --target-repo-root ../knowledge-hub-opus-hardening-clean --pathspec-file ops/checkpoints/11-weekly-ask-ci-release-hygiene.pathspec --allow-dirty-target --stage --json
python scripts/sync_worktree_slice.py --source-repo-root ../knowledge-hub --target-repo-root ../knowledge-hub-opus-hardening-clean --pathspec-file ops/checkpoints/12-opus-architecture-hardening.pathspec --allow-dirty-target --stage --json
```

Important:
- The target worktree should start from the same clean branch/base commit as the
  dirty source worktree. Replaying a cross-cut stabilization slice onto an
  older base such as `origin/main` can fail even when the helper itself copied
  the requested paths correctly.
- `scripts/sync_worktree_slice.py` only replays the selected file set. It does
  not guarantee that the current tranche manifest is dependency-closed; use a
  focused smoke/test pass in the target worktree to confirm that.
- `12-opus-architecture-hardening.pathspec` is an overlay, not an independent
  minimal branch. The verified clean branch required earlier bucket content,
  packaged ontology profile data, paper-memory fixtures, Foundry source files,
  and the `.gitignore` root-anchor fix for nested `src/` and `data/` paths.

Anything not covered by those seven buckets is reported as `unmatched` and
should be reviewed deliberately instead of being auto-staged.
