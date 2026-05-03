# Dirty Tree Split Plan For Core-Loop Stabilization

Date: 2026-04-16

## Current evidence

- `git status --short --untracked-files=all | wc -l` on the main worktree currently reports `1146`.
- `python scripts/report_checkpoint_split.py --repo-root . --show-paths --max-paths-per-bucket 8` currently reports:
  - `913 matched`
  - `233 unmatched`
- The largest active buckets are:
  - `03-infrastructure-core-ownership`: `336`
  - `05-retrieval-ask-v2`: `236`
  - `02-cli-mcp-surface-migration`: `95`
  - `06-ingest-learning-vault-web`: `79`
  - `07-foundry-satellites`: `65`

This is not a reviewable stabilization branch as-is. The next narrow branch must be split from a clean ref and staged by exact pathset, not by broad directory.

## Stabilization branch scope

The current release-blocking stabilization slice should include only the paths that keep the representative core loop trustworthy:

- product authority docs
  - `README.md`
  - `docs/PROJECT_STATE.md`
  - `CHANGELOG.md`
- narrow status/doctor/search/vector runtime paths
  - `knowledge_hub/application/context.py`
  - `knowledge_hub/application/runtime_diagnostics.py`
  - `knowledge_hub/application/vector_restore.py`
  - `knowledge_hub/infrastructure/persistence/vector.py`
  - `knowledge_hub/interfaces/cli/commands/status_cmd.py`
  - `knowledge_hub/interfaces/cli/commands/doctor_cmd.py`
  - `knowledge_hub/interfaces/cli/commands/search_cmd.py`
  - `knowledge_hub/interfaces/cli/commands/vector_cmd.py`
  - `knowledge_hub/interfaces/cli/commands/vector_compare_cmd.py`
  - `knowledge_hub/interfaces/cli/main.py`
- fixed smoke / regression assets
  - `scripts/check_release_smoke.py`
  - `tests/test_cli_smoke_contract.py`
  - `tests/test_core_loop_read_safety.py`
  - `tests/test_doctor_cmd.py`
  - `tests/test_runtime_diagnostics.py`
  - `tests/test_search_cmd.py`
  - `tests/test_vector_db_fts.py`
  - `tests/test_vector_restore_cmd.py`
- tranche-local handoff notes
  - `docs/status/2026-04-16-dirty-tree-split-plan.md`

## Hold out of the stabilization branch

Do not mix the following into the current stabilization PR:

- broad `ask-v2` / answer-assembly expansion
- foundry-core and `.github` satellite changes
- larger ingest / learning / vault productization work
- unrelated docs import / repo migration churn
- generated eval artifacts and runtime outputs

Those belong in separate follow-up branches or remain in the dirty tree until they have their own reviewed split.

## Branching recommendation

1. Start from a clean ref or clean sibling worktree, not from the current dirty index.
2. Use `scripts/report_checkpoint_split.py --write-pathspec-dir <dir>` to export exact current bucket pathspecs.
3. Stage only the stabilization paths above plus any immediately necessary dependency paths.
4. Keep unmatched paths out unless they are required to make the narrowed slice build or test.

## Suggested execution

```bash
git worktree list
git worktree add ../knowledge-hub-core-loop-stabilization-head -b tranche/2026-04-16-core-loop-stabilization-head HEAD
```

Then replay only the stabilization slice into the clean worktree:

```bash
python scripts/sync_worktree_slice.py \
  --source-repo-root . \
  --target-repo-root ../knowledge-hub-core-loop-stabilization-head \
  --pathspec-file ops/checkpoints/09-core-loop-stabilization.pathspec \
  --stage --json
git -C ../knowledge-hub-core-loop-stabilization-head status --short
```

## Executed setup

- `git fetch origin --prune` completed successfully on `2026-04-16`.
- Created clean sibling worktree `../knowledge-hub-core-loop-stabilization` on branch `tranche/2026-04-16-core-loop-stabilization`, tracking `origin/main` at `06f4ec0`.
- Created a second clean sibling worktree `../knowledge-hub-core-loop-stabilization-head` on branch `tranche/2026-04-16-core-loop-stabilization-head`, based on the current local dirty-branch clean base `1975a97`.
- Added `ops/checkpoints/09-core-loop-stabilization.pathspec` as the tranche-local exact manifest for the representative weekly core-loop stabilization slice.
- Added `scripts/sync_worktree_slice.py` so the selected slice can be replayed from the dirty source worktree into the clean target worktree even when the source includes untracked files.
- Live replay into the `1975a97`-based target succeeded mechanically: `scripts/sync_worktree_slice.py --stage --json` copied and staged `25` exact paths.
- The replay is not yet dependency-closed as a standalone clean branch. Focused pytest collection and `python scripts/check_release_smoke.py --mode weekly_core_loop --json` both failed in the clean target with import-level gaps such as:
  - `knowledge_hub.interfaces.cli.commands.agent_cmd`
  - `knowledge_hub.interfaces.cli.commands.index_cmd`
  - `knowledge_hub.application.claim_signals`
  - `PUBLIC_DEFAULT_EMBEDDING_MODEL` drift between `knowledge_hub.infrastructure.config` and `knowledge_hub.core.config`
  - `OntologyEvent` drift between `knowledge_hub.core.models` and the persistence/store path

## Replay iterations after initial closure gap

- Widened the representative manifest from the initial `25` paths to `86` exact paths, carrying the direct runtime closure needed by `doctor`, `status`, `index`, and the current `search` import path.
- Replayed that widened manifest into fresh clean sibling worktrees on the current clean base:
  - `../knowledge-hub-core-loop-stabilization-head-v7`
  - `../knowledge-hub-core-loop-stabilization-head-v8`
  - `../knowledge-hub-core-loop-stabilization-head-v9`
- The helper remained mechanically stable across each replay: staged copy worked for untracked files, modified tracked files, and doc/schema additions without touching the dirty source tree.
- The weekly core-loop smoke now passes `4/5` checks in the clean target:
  - `top_help`: pass
  - `doctor --json`: pass
  - `status`: pass
  - `index --vault-all --json`: pass
  - `search --source vault --mode keyword --json`: still blocked
- Closure work that was required to reach `4/5` included:
  - lazy CLI help rendering so top-level help no longer imports every lazy command surface
  - schema artifacts for `doctor` / runtime diagnostics
  - `reranker`, `memory_prefilter`, `paper prefilter`, `rag_reports`, `graph_signals`, and the `vault/indexer` authoritative-resync path

## Current blocker after replay widening

- The remaining failure is no longer a missing-file import in the current manifest. In `../knowledge-hub-core-loop-stabilization-head-v9`, `search` now initializes far enough to emit a JSON payload, but it reports:
  - `initError: RAGSearcher.__init__() got an unexpected keyword argument 'sqlite_db'`
- This means the clean-branch split is now exposing an API-contract drift between the narrowed `search` surface and the older `RAGSearcher` constructor that still exists on the clean base outside the selected slice.
- At this point the next move should not be blind manifest widening. The branch owner needs to choose one of:
  1. include the compatible `RAGSearcher` construction path and its direct runtime dependencies as part of the core-loop slice, or
  2. redefine the representative weekly slice so it stops claiming standalone `search` readiness.

## Current blocker

The representative `09-core-loop-stabilization` manifest is now executable as a file-replay artifact, but it is still narrower than the actual import/runtime closure needed for a clean standalone branch. The next split pass must either:

1. widen the manifest to include the compatible `search` runtime constructor path and any direct retrieval-service dependencies, or
2. explicitly reduce the smoke/help contract so the tranche no longer claims standalone `search` readiness.

## Decision rule

- If a path is not required to keep `doctor/status/index/search` trustworthy, leave it out of the stabilization branch.
- If a path is required but brings in a wider subsystem, record that dependency explicitly before staging it.

## 2026-04-17 slice layering

The split is now tracked as three reproducible overlays instead of one mixed
dirty-tree replay:

1. `ops/checkpoints/00-replay-control-plane.pathspec`
   - source of truth: current dirty/staged worktree
   - scope: checkpoint definitions, replay/report helpers, protecting tests,
     split-status note, and matching changelog/project-state records
2. `ops/checkpoints/09-core-loop-stabilization.pathspec`
   - source of truth: clean sibling
     `../knowledge-hub-core-loop-stabilization-head-v11`
   - scope: frozen weekly `doctor/status/index/search/vector` runtime plus the
     direct schemas/tests needed to keep that tranche trustworthy
3. `ops/checkpoints/11-weekly-ask-ci-release-hygiene.pathspec`
   - source of truth: current dirty/staged worktree
   - scope: `weekly_core_loop` `ask` overlay, hermetic `.github/workflows/ci.yml`
     replacement, authority/public-release support files, and their docs/tests

This layering avoids rebuilding `09` from the mixed tree after it was already
validated in the clean sibling. It also makes the file-level overlaps explicit:
`scripts/check_release_smoke.py`, `tests/test_cli_smoke_contract.py`,
`README.md`, `CHANGELOG.md`, and `docs/PROJECT_STATE.md` are intentionally
replayed again in the `11` overlay because that tranche promotes the weekly
contract from `5/5` to `6/6` and narrows required PR CI.
