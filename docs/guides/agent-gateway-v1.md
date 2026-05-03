# Agent Gateway v1

`Agent Gateway v1` is the narrow official bridge between the local Knowledge Hub runtime and future agentic runtimes.

It does **not** introduce a new CLI or MCP namespace. Instead, it formalizes two existing surfaces.

## Official surfaces

### 1. Read-only context packing

- CLI: `khub agent context "goal" --repo-path .`
- MCP: `build_task_context`

Use this surface when an external planner or coding agent needs grounded context without mutating repository state.

Contract:

- `gateway.surface = "task_context"`
- `gateway.mode = "context"`
- `gateway.contract = "read_only_dry_run"`
- `executionAllowed = false`
- `writebackAllowed = false`
- `repoContextEphemeral = true`

## 2. Dry-run plan envelopes

- CLI: `khub agent run --goal "..." --dry-run --json`
- MCP: `run_agentic_query` with `dry_run=true`

Use this surface when an external runtime needs a structured plan/result envelope without action execution or writeback.

Contract:

- `gateway.surface = "agent_run"`
- `gateway.mode = "dry_run"`
- `gateway.contract = "read_only_dry_run"`
- `executionAllowed = false`
- `writebackAllowed = false`
- `repoContextEphemeral = true`

## Out of scope for v1

The following are intentionally excluded from `Agent Gateway v1`:

- execution-enabled agent flows
- writeback approvals
- receipts for real actions
- new `khub gateway` commands
- new MCP gateway tool families

## Notes

- `task-context.result.v1` remains the public context-pack contract.
- `agent-run-result.v1` remains the public dry-run envelope contract.
- `context_pack.result.v1` remains an internal foundation, not the public gateway schema.

## First consumer preview

The first concrete consumer of `Agent Gateway v1` is a repo-local preview script:

```bash
python scripts/preview_agent_gateway.py "How should I refactor the RAG flow?" --repo-path .
python scripts/preview_agent_gateway.py "How should I refactor the RAG flow?" --repo-path . --json
```

This script does not add a new product surface. It simply consumes the existing official gateway contracts:

- `khub agent context --json`
- `khub agent run --dry-run --json`

and fails fast if either payload breaks the documented `gateway` contract.

The first request-only consumer for the writeback lane is intentionally even narrower:

```bash
python scripts/request_agent_docs_writeback.py "Update docs/status and worklog for the latest gateway tranche" --repo-path .
python scripts/request_agent_docs_writeback.py "Update docs/status and worklog for the latest gateway tranche" --repo-path . --json
```

This consumer uses the existing `khub agent writeback-request --json` surface, but it adds a stricter local check:

- only `docs/status/` and `worklog/` targets are accepted
- `docs/adr/` and `reviews/` remain valid for the broader docs-only lane, but this first consumer rejects them
- approval and execution still happen separately through `khub labs ops action-ack` and `khub labs ops action-execute`

The next thin layer on top is a semi-automated operator loop:

```bash
python scripts/run_agent_docs_writeback_loop.py "Update docs/status and worklog for the latest gateway tranche" --repo-path .
python scripts/run_agent_docs_writeback_loop.py "Update docs/status and worklog for the latest gateway tranche" --repo-path . --apply
```

Behavior:

- default mode stays preview-only and prints the existing `ack` / `execute` commands
- `--apply` is explicit and reuses the same official CLI surfaces for `ack` and `execute`
- the loop still fails closed unless the predicted targets stay inside `docs/status/` and `worklog/`
- this remains a repo-local script, not a new product command family

## What comes next

The next gateway tranche is intentionally **not** another broadening of read-only surfaces.

That direction is now implemented as one narrow execution-adjacent lane only:

- `approval-gated repo-local writeback request`

The current CLI/operator surface is:

```bash
khub agent writeback-request "Refactor the RAG fallback flow" --repo-path . --json
khub labs ops action-ack --action-id <id> --actor cli-user
khub labs ops action-execute --action-id <id> --actor cli-user --json
khub labs ops action-resolve --action-id <id> --actor cli-user
```

It creates a pending request envelope with:

- `gateway.version = "v2"`
- `gateway.surface = "agent_writeback_request"`
- `gateway.mode = "request"`
- `gateway.contract = "approval_gated_repo_local_writeback"`
- `executionAllowed = false`
- `writebackAllowed = false`
- `approvalRequired = true`

The request payload also carries an additive advisory preview:

- `writebackPreview.kind = "repo_local_predicted_write_set"`
- `writebackPreview.advisory = true`
- `writebackPreview.targets[]` is a predicted repo-local write set derived from workspace context and explicit goal-path hints
- in the current first-consumer tranche, those targets are narrowed to docs-only prefixes: `docs/adr/`, `docs/status/`, `reviews/`, `worklog/`
- the first real request-only consumer narrows further to `docs/status/` and `worklog/` only
- `writebackPreview.previewFingerprint` is a stable summary hash for operator inspection and queue traces

This preview is intentionally not an exact diff promise. It is a narrow operator-facing prediction of likely repo-local write targets before approval/execution.

This means the implemented v2 MVP can:

- propose a bounded repo-local change
- keep execution blocked until an explicit `ack` exists
- execute only through the existing ops-action queue surface after approval
- auto-resolve the queued agent action when the approved execution succeeds with `writeback.ok = true`
- keep the request and execution receipt inspectable through the existing ops-action queue surface

Still out of scope for this v2 MVP:

- general agent execution beyond this one queue-backed lane
- vault mutation
- external side effects
- hidden automation
- new broad gateway command families

In other words, `Agent Gateway v1` remains the read-only/dry-run bridge, and `v2` currently adds only a single approval-gated repo-local writeback lane rather than a general agent platform.
