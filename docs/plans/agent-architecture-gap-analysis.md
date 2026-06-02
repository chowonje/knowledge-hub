# Agent Architecture Gap Analysis

Date: 2026-06-03

## Scope

This gap analysis compares current `origin/main` in the clean worktree with the
dirty LLM CLI experiment at:

```text
KnowledgeOS/.worktrees/knowledge-hub-khub-llm-cli-v1-chat-models-20260602
```

No code was merged in this tranche.

## Baseline Evidence

- Selected product worktree:
  `KnowledgeOS/.worktrees/knowledge-hub-agent-architecture-design-tranche-1-20260603`
- Selected base: `origin/main` at `f8b3144d17c578c43a168337f3199608c9436208`
- Product symlink: `KnowledgeOS/knowledge-hub -> <allinone>/knowledge-hub`
- Vault symlink: `KnowledgeOS/vault -> <iCloud Obsidian vault>`
- Vault scanned: no
- Worktree inventory after preflight: 85 registered worktrees
- LLM CLI worktree status: dirty, with modified docs/code/tests and untracked `chat`, `tui`, `auth`, `models`, `enrich`, session, slash, and Codex provider files
- Active Python package evidence: `python -m pip show knowledge-hub` returned `Package(s) not found`; `khub` resolves through the operator's pyenv shim, so no current editable checkout target was confirmed from pip.

## Current Canonical Main

Canonical CLI entrypoint:

```text
pyproject.toml: khub = "knowledge_hub.interfaces.cli.main:cli"
```

Current `knowledge_hub/interfaces/cli/main.py`:

- keeps `khub` as a normal Click lazy command group
- shows public default commands focused on the core loop
- registers `agent` as hidden advanced/gateway surface
- does not register `chat`, `tui`, `auth`, `models`, or `enrich`
- keeps `khub labs foundry` as the foundry/operator group

Current Agent Gateway docs:

- `khub agent context "goal" --repo-path .`
- `khub agent run --goal "goal" --repo-path . --dry-run --json`
- `khub agent writeback-request "goal" --repo-path . --json`
- no new `khub gateway` command family

## LLM CLI Worktree Diff Summary

The LLM CLI worktree changes `knowledge_hub/interfaces/cli/main.py` by:

- setting `@click.group(cls=_LazyCommandGroup, invoke_without_command=True)`
- making bare `khub` launch `run_tui(...)`
- adding public default help entries for `chat`, `tui`, `auth`, `models`, and `enrich`
- lazy-loading:
  - `knowledge_hub.interfaces.cli.commands.chat_cmd`
  - `knowledge_hub.interfaces.cli.commands.tui_cmd`
  - `knowledge_hub.interfaces.cli.commands.auth_cmd`
  - `knowledge_hub.interfaces.cli.commands.models_cmd`
  - `knowledge_hub.interfaces.cli.commands.enrich_cmd`

Observed dirty/untracked files include:

- docs: `CHANGELOG.md`, `README.md`, `docs/ARCHITECTURE.md`, `docs/PROJECT_STATE.md`, `docs/guides/cli-commands.md`, `docs/adr/2026-06-02-assistant-enrichment-boundary.md`
- CLI/runtime: `chat_cmd.py`, `tui_cmd.py`, `auth_cmd.py`, `models_cmd.py`, `enrich_cmd.py`, `assistant_runtime.py`, `session_runtime.py`, `slash_registry.py`
- provider/config: `codex_provider.py`, provider registry changes, config/context changes
- tests: `test_chat_cmd.py`, `test_tui_cmd.py`, `test_auth_cmd.py`, `test_models_cmd.py`, `test_enrich_cmd.py`, `test_session_cmd.py`, `test_slash_registry.py`, `test_codex_provider.py`, and CLI smoke updates

## MCP Tool Catalog vs Chat Needs

Current default MCP profile already covers the embedded-agent read path:

- `search_knowledge`
- `ask_knowledge`
- `build_task_context`
- paper lookup/read helpers
- paper memory read helpers

Labs/all profiles cover heavier or operator surfaces:

- `run_agentic_query`
- heavy ingest and paper build/index helpers
- crawl, learning, ops actions, async jobs, and workbench helpers

Chat/TUI needs are therefore not a new tool plane. They map to existing layers:

- plain chat turn: Interface plus provider/model route
- evidence slash routes such as `/paper`: existing paper/RAG runtime
- coding/design context: Gateway `context` or MCP `build_task_context`
- dry-run plan: Gateway `run --dry-run` or MCP `run_agentic_query(dry_run=true)`
- writeback: Gateway `writeback-request` plus `labs ops action-*`
- derivative writes: future `khub enrich` dry-run/apply path, not hidden chat mutation

## Merge Risks

- Bare `khub` launching TUI is a high-risk default surface change. It changes
  command ergonomics and may break existing scripts expecting help or no-op behavior.
- Making `chat`, `tui`, `auth`, `models`, and `enrich` public default commands
  expands the product promise beyond the current Research Preview core loop.
- `auth` and `models` touch provider/config behavior and need policy/privacy tests.
- `chat` and `tui` need tests proving they reuse existing runtime and do not
  duplicate RAG or bypass outbound-provider guards.
- `session_runtime` needs explicit local storage and redaction rules before
  transcripts become durable artifacts.
- `enrich` should not merge until plan/dry-run/apply/readback semantics are fixed.

## Recommended Merge Order

### P1: Keep the interface boundary, merge no default-TUI behavior

Do not merge bare `khub -> TUI`. Keep TUI behind explicit `khub tui` until
operator evidence proves the default should change.

### P2: Land provider/model setup only after focused policy tests

Consider `auth`, `models`, and Codex provider support first if tests prove:

- secrets are never printed
- `routing.llm.tasks.chat` does not alter `ask` routing by default
- external provider calls remain policy-gated
- config writes are explicit and reversible

### P3: Land `khub chat` as an explicit Interface command

`khub chat` can merge only if it:

- reuses existing search/ask/paper runtime for evidence skills
- emits layer usage diagnostics
- does not create memory, claims, ontology, cluster, parser, DB, index, or vault writes during ordinary turns
- includes non-interactive tests for command behavior and route metadata

### P4: Land `khub tui` after chat contract stabilizes

TUI should be a presentation layer over the same assistant runtime. It needs
tests for non-interactive entry/help behavior and must not replace default
`khub` behavior in the first merge.

### P5: Defer `khub enrich`

`enrich` is useful but mutation-adjacent. It should wait for a separate tranche
that defines:

- layer registry status
- deterministic plan IDs
- dry-run output schema
- explicit apply command
- readback checks
- coverage/freshness diagnostics

## Recommendation

Proceed with documentation and schema only for this tranche. The next safe
implementation tranche is a narrow Interface merge plan for explicit `khub chat`
or provider/model setup, not a wholesale merge of the dirty worktree and not a
new `khub gateway` command family.
