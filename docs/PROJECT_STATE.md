# Project State

Last updated: 2026-06-08

## What this project is

`knowledge-hub` is a local-first, policy-first, retrieval-assistant-first knowledge runtime.

It combines notes, papers, web ingestion, local stores, retrieval, MCP/CLI surfaces, and bounded agent workflows. The main architectural priority is to keep the default runtime retrieval-assistant-first while promoting new capabilities through additive, inspectable labs surfaces first.

## Current public release posture

- Public-facing posture is **Research Preview**, not stable-release language.
- The supported default path for public docs is `discover -> index -> search/ask -> evidence review`.
- Public release trust comes from a narrow smoke gate / approval slice, not a full-repo green claim.
- The default product promise is intentionally smaller than the repository's full implementation surface. `khub labs ...`, Agent Gateway, answer-loop evals, learning workflows, Foundry delegation, and OS/decision surfaces remain experimental or operator-facing unless explicitly promoted.

## Current green signal

- Core docs and CLI guidance now align around the representative local-first path: `discover -> index -> search/ask -> evidence review`.
- The weekly core-loop smoke exists as the repeatable product-loop gate: `python scripts/check_release_smoke.py --mode weekly_core_loop --json`.
- Recent local-first security hardening blocks two concrete privacy regressions:
  - task-context workspace assembly requires `repo_path` to resolve inside a git worktree, skips symlink escapes, redacts P0-like workspace snippets before prompt construction, and blocks MCP task-context LLM synthesis of P0 context before invoking an unknown or external summarizer.
  - RAG answer routing treats `allow_external=false` as a hard block for configured non-local `fixed_llm` fallback after local routing failure.
- Focused verification for that hardening covered task context, MCP agent fallback, RAG route resolution, representative RAG answer policy behavior, original repro scripts, and public-release hygiene.

## Active stabilization lanes

### Default Runtime And Evidence Review

- Keep the default path narrow and inspectable.
- `khub index` remains the retrieval-index builder for lexical, vector, and metadata surfaces.
- Claim cards, evidence links, answer traces, semantic cards, graph/ontology projections, and registry records remain derivative unless explicitly backed by source-span evidence and eval gates.
- Registry writes remain explicit facade options or application helper calls; default CLI/MCP reads must not silently create packet/context records or call external providers.

### Local-First Policy And Task Context

- `P0` outbound must remain blocked by default.
- If classification cannot be determined, default to `P0` and no external call.
- Repo/project context is read-only and ephemeral unless a feature explicitly promotes it into a persistent store.
- Task-context snippets must not leak secrets, private absolute local paths, or symlink-escaped workspace content into prompts or external summarization paths.

### Visual Retrieval-Hint Lane

- Status: **hold for promotion decision**.
- The visual retrieval-hint work has useful labs evidence and applied local labs artifacts, but the latest runtime candidate-discovery route design is still report-only.
- Current boundary: retrieval hints are not strict evidence, not citation-grade evidence, not answer-visible text, and not default runtime search integration.
- Next decision must be one of:
  - freeze as labs-only retrieval-hint tooling, or
  - run a tightly scoped runtime candidate-discovery promotion tranche with explicit filters, answer-path exclusion rules, and eval gates.
- Do not add another report-only visual retrieval-hint phase unless it removes a named blocker for that decision.

### Candidate Layer And Structured Evidence

- Status: **hold for runtime-answer integration decision**.
- SectionSpan, FigureCaption, EquationQuote, TableRegion/TableCell, StrictEvidence, eligibility, citation-grade, and runtime-binding work has built many schema-backed candidate and apply-gated artifacts.
- These records are valuable as provenance and promotion prep, but they are not automatically answer-visible evidence.
- Runtime exposure requires a separate promotion gate that proves source-span authority, citation-grade behavior, parser/routing safety, and answer-path integration.

### Answer Path, Ask V2, And Source Quality

- Ask-path and source-specific quality work should be evaluated against the current default path, not by historical status claims alone.
- Source-quality observation remains distinct from hard-gate promotion. A trend or observation report is evidence for a later decision; it is not itself a release gate unless promoted.
- Any claim that paper/web/vault routes are ready should name the exact collector, run directory, query set, and current route/fallback metrics.

### Foundry, Agent, And OS Surfaces

- Python remains the policy gate and default end-to-end executor.
- `foundry-core` may be the preferred delegated agent runtime, but it consumes Python-owned mode/payload decisions rather than duplicating product policy.
- `khub agent`, `khub os`, and evidence-candidate review surfaces remain advanced/operator-facing unless explicitly promoted into the default product promise.

### Palantir And PDF Translation

- Status: **hold as labs sidecar**.
- The Palantir AIP PDF translation experiment proved external-call wiring and derivative artifact generation, but not a product-ready full-PDF translation workflow.
- Palantir may translate or process explicitly approved low-risk chunks only as an external sidecar. It must not become citation-grade evidence, default answer context, canonical source truth, or a silent indexing path without a separate promotion decision and eval gate.
- Layout-faithful translated PDF cloning is not a default product target. The preferred Korean reading direction is a source-linked reading pack: local source parsing, page/section/chunk anchors, Korean Markdown/HTML, optional derivative PDF, then explicit review/import.

## Active blockers and risks

- The project has a strong habit of producing schema-backed report-only phases. That is useful for safety, but it can become duplicate work when a tranche only creates another `ready_for_*` artifact without changing runtime behavior, reducing a named blocker, or improving a measured default-path metric.
- Visual retrieval-hint and structured-evidence lanes should not continue by default. Each new tranche must close a named blocker or make a documented hold/freeze decision.
- Palantir/PDF translation work should not continue by default. Any new tranche must prove a source-linked Korean reading pack improves actual reading/review, or remove a named labs blocker without widening the default product promise.
- `docs/PROJECT_STATE.md` is now the current-state surface. Long phase history belongs in `docs/project-state-archive/` so this file does not become an unreadable work ledger again.
- Release/readiness claims must be current-run claims. Historical green sheets, archived evals, and old branch/worktree notes are not sufficient without rerun evidence or a stated reason they still apply.

## Stop Rules For New Work

Do not start a new tranche when its expected output is only:

- another report-only helper,
- another dry-run wrapper,
- another `ready_for_*` status,
- another zero-mutation counter matrix,
- another manual-review worksheet,
- or another design layer that does not remove a named blocker.

A new tranche is justified only when it does at least one of:

- improves or verifies the default path `discover -> index -> search/ask -> evidence review`,
- reduces a policy/security/privacy risk,
- removes a named blocker from this file,
- makes a clear hold/freeze/promotion decision,
- or converts a prior report-only result into a bounded runtime-visible behavior with explicit rollback and eval gates.

## Archive Index

The full pre-restructure project-state ledger is preserved at:

- [2026-06-03 pre-current-summary project state](project-state-archive/2026-06-03-pre-current-summary-project-state.md)

Use that archive for detailed historical phase evidence, including:

- visual retrieval-hint Phase 8-15 history,
- candidate-layer and structured-evidence report-only history,
- source-quality and ask-path stabilization history,
- Foundry/OS/agent-surface notes,
- paper/concept quality history,
- AI paper math expansion history.

Future archive additions should be topic-specific and linked here rather than appended as long dated bullet streams in this file.
