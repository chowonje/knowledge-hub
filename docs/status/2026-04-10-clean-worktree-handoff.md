# Clean Worktree Handoff For Next Tranche

Date: 2026-04-10

## Current footing

- Live state on 2026-04-10:
  - main worktree: `<repo-root>`
  - branch: `feat/obsidian-cli-adapter`
  - `HEAD`: `1975a97`
  - live worktrees:
    - `<repo-root>` at `1975a97`
    - `<repo-root>-wt1-obsidian-cli-adapter` at `7fdc12b`
    - `<repo-root>-wt1-obsidian-cli-adapter-next-tranche` at `d184c8c`
    - `<repo-root>-wt1-obsidian-cli-adapter-next-tranche-2` at `a505d67`
- `git status --short --untracked-files=all | wc -l` currently reports `1483`, so the main tree is still heavily dirty across product code, docs, tests, and generated/runtime artifacts.
- That makes the current tree a bad place to start the next tranche. The goal here is a clean starting surface, not a cleanup of the existing in-flight tree.

## Why this recommendation can stale

- `7fdc12b` is now historical bootstrap context for the older clean sibling worktree, not the current main-worktree `HEAD`.
- `origin/main` moves whenever the repository baseline advances.
- `git worktree list` can change whenever a sibling worktree is created or closed.
- Because of that, do not reuse this note blindly. Re-check the execution inputs immediately before running `git worktree add`.

## Before Running `git worktree add`

Run these commands immediately before creating the new worktree:

```bash
git fetch origin --prune
git worktree list
git rev-parse --short HEAD
git rev-parse --abbrev-ref HEAD
git status --short --untracked-files=all | wc -l
```

Required re-check:

- `git worktree list` should be treated as the source of truth for currently occupied sibling names and paths. Do not assume `wt1` naming is still free.
- `git rev-parse --short HEAD` should still be `1975a97` if you plan to continue from the current main-worktree line.
- `git rev-parse --abbrev-ref HEAD` should still be `feat/obsidian-cli-adapter` if you plan to continue from that same reviewed line.
- `git status --short --untracked-files=all | wc -l` should still show that the current tree is heavily dirty, which is the reason to avoid starting the tranche here directly.
- If any of those checks differ from this note, stop and choose the base again deliberately. Do not cargo-cult `1975a97` or `7fdc12b`.

## Go / No-Go Base Choice

- Go with current `HEAD` only if the re-check still shows `HEAD=1975a97`, `branch=feat/obsidian-cli-adapter`, and the tranche is meant to continue the live main-worktree line.
- Go with `7fdc12b` only if you deliberately want to restart from the older clean sibling baseline rather than continue the current main-worktree line.
- No-go for `7fdc12b` as a default choice if you are only using it because it appears in older handoff notes.
- Go with `origin/main` only if you intentionally want a repository-baseline reset for the next tranche after `git fetch origin --prune`.
- No-go for `origin/main` if you still need the current local continuation line and are only choosing `origin/main` because it is newer by default.

## Recommended start point

1. Start the next tranche from a clean ref, never from the current dirty index or working tree state.
2. If the tranche is continuing the current local line of work after the re-check above, use the live main-worktree `HEAD`.
3. If the tranche is intentionally restarting from the older clean sibling context, use `7fdc12b` deliberately rather than by assumption.
4. If the tranche is intentionally resetting to repository baseline after the re-check above, use `origin/main`.
4. Do not infer the base from staged or untracked files in the dirty tree.

Example:

```bash
git fetch origin --prune
git worktree list
git branch --list 'wt*' 'tranche/*'
git worktree add ../knowledge-hub-<next-sibling> -b tranche/2026-04-10-<slice>-<next-sibling> 1975a97
```

Repository-baseline variant:

```bash
git fetch origin --prune
git worktree list
git branch --list 'wt*' 'tranche/*'
git worktree add ../knowledge-hub-<next-sibling> -b tranche/2026-04-10-<slice>-<next-sibling> origin/main
```

## Recommended naming

- Branch: `tranche/2026-04-10-<slice>-<next-sibling>`
- Worktree directory: `../knowledge-hub-<next-sibling>`

Why this branch pattern:

- `tranche` marks the branch as a narrow split from a dirty larger line of work
- the date makes the handoff sortable and easier to trace later
- `<slice>` forces a concrete scope name
- `<next-sibling>` forces an explicit collision check against currently occupied sibling names and paths

Do not assume `wt1` is available. In the current live topology, `wt1` naming is already occupied by existing sibling worktrees.

If the tranche later becomes a stable product branch, that rename can happen after the clean split is committed and reviewed.

## Record ownership

Product-durable records:
- `CHANGELOG.md` for the minimum durable note of what changed.
- `docs/PROJECT_STATE.md` only when the tranche changes architecture, runtime behavior, policy semantics, tool contracts, or retrieval/eval strategy.
- ADR or protecting test/eval only when the tranche makes a durable design decision or changes a guarded behavior boundary.

Process-only records:
- `docs/status/*.md` for tranche-local handoff, verification, or freeze notes.
- `tasks/*.md` for task intent and done condition.
- `reviews/*.md` for reviewer findings, risks, and missing-test notes.
- `worklog/*.md` for day-by-day execution trace.

Rule:
- Do not treat `docs/status/`, `tasks/`, `reviews/`, or `worklog/` as the source of truth for product behavior when `CHANGELOG.md` or `docs/PROJECT_STATE.md` should carry that claim.

## Safe check and staging commands

Inspect first:

```bash
git worktree list
git status --short --untracked-files=all
git status --short --untracked-files=all | wc -l
git rev-parse --short HEAD
git rev-parse --abbrev-ref HEAD
git diff --stat
git diff --name-only
```

Stage narrowly:

```bash
git add -p <path>
git diff --cached --stat
git diff --cached
```

If the tranche is still being split from a dirty tree, prefer exact pathspec exports over broad directory staging:

```bash
python scripts/report_checkpoint_split.py --show-paths
python scripts/report_checkpoint_split.py --write-pathspec-dir /tmp/khub-checkpoints
git add --pathspec-from-file=/tmp/khub-checkpoints/<bucket>.pathspec
git diff --cached --stat
git diff --cached
```

## Do not do this from the current dirty tree

- Do not branch the next tranche from the current dirty worktree and assume later staging will stay isolated.
- Do not use broad staging commands such as `git add .`, `git add -A`, `git add docs/`, `git add knowledge_hub/`, or `git commit -a`.
- Do not try to "clean up first" with destructive reset/checkout/revert commands.
- Do not treat a partially staged index in this tree as a reliable tranche boundary.
- Do not move unrelated existing changes into the next tranche just because they already exist locally.

## Handoff summary

- Preferred path for continuation work: create a fresh worktree from the live current `HEAD` `1975a97` on a narrow `tranche/2026-04-10-<slice>-<next-sibling>` branch and keep the dirty tree untouched.
- Treat `7fdc12b` as historical bootstrap context for the older clean sibling, not as the default continuation base.
- If the tranche is supposed to reset to repository baseline instead, create the same clean worktree from `origin/main` deliberately, not by assumption.
- Immediately before execution, rerun the five-command checklist above and treat any mismatch as a no-go until the base choice is re-decided.
