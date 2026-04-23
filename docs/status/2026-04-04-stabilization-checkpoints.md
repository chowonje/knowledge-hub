# Stabilization Checkpoints

Date: 2026-04-04

## Goal

Provide a safe, non-destructive split path for the current large dirty worktree while keeping the green-baseline stabilization slice isolated.

## Why this is not an automatic branch rewrite

- The current worktree contains hundreds of changed paths across Python, TypeScript, docs, and ops surfaces.
- In that state, automatic branch splitting or history rewriting has a high chance of mixing unrelated paths or trampling local-only work.
- The safer approach is to classify the *current* worktree, export exact pathspec files, and stage buckets intentionally.

## Canonical split workflow

1. Run `python scripts/report_checkpoint_split.py --show-paths`
2. Review `unmatched` paths
3. Export exact bucket pathspecs with `python scripts/report_checkpoint_split.py --write-pathspec-dir /tmp/khub-checkpoints`
4. Stage bucket `01-contract-entrypoints` first
5. Keep later buckets parked until the green baseline is committed and verified

## Bucket order

1. `contract-entrypoints`
2. `cli-mcp-surface-migration`
3. `infrastructure-core-ownership`
4. `paper-public-reading`
5. `retrieval-ask-v2`
6. `ingest-learning-vault-web`
7. `foundry-satellites`

## Notes

- Bucket `01` is intentionally narrow: packaging, canonical entrypoints, health semantics, and explicit event-integrity repair.
- The checkpoint manifest lives in `ops/checkpoints/checkpoints.json`.
- The exported pathspec files contain exact current changed paths only, which is safer than broad `git add dir/**` staging on this branch.
- Any `unmatched` paths require manual review before they are staged or parked.
