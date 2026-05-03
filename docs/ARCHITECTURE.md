# Architecture

## What this repo is

`knowledge-hub` is a local-first research knowledge system. It ingests papers, vault notes, and web documents; stores inspectable local artifacts and indices; and exposes grounded retrieval plus bounded labs workflows over those sources.

The main architectural priority is to keep the default runtime retrieval-assistant-first while promoting new capabilities through additive, inspectable labs surfaces first.

See also:
- `docs/maps/README.md`
- `docs/maps/canonical-ownership-map.md`
- `docs/maps/agent-execution-map.md`
- `docs/maps/data-policy-flow-map.md`
- `docs/adr/2026-04-24-store-authority-inventory.md`
- `docs/adr/2026-04-24-mixed-store-schema-prep.md`

## Major boundaries

### Product/runtime boundary
- `knowledge_hub/` is the default product/runtime: ingestion, local stores, retrieval, answer generation, task-context assembly, and CLI/MCP surfaces.
- `foundry-core/` is the stricter TypeScript agent-runtime boundary: policy/audit contracts, runtime abstractions, and connector interfaces.
- Python remains the policy gate and default end-to-end executor.
- `foundry-core` may orchestrate delegated agent flows, but it should consume Python-owned mode/payload decisions rather than duplicating them.

### Canonical package boundary
- `knowledge_hub.interfaces.*` is the canonical human/tool entry surface: CLI entrypoints, MCP transport, and command registration.
- `knowledge_hub.application.*` owns runtime composition, task-context assembly, Foundry bridging, and cross-cutting application services.
- `knowledge_hub.infrastructure.*` owns config/persistence/provider lookup surfaces used by the runtime.
- `knowledge_hub.core.*` owns low-level local primitives such as stores, models, validators, chunking, and compatibility exports for external callers.
- `knowledge_hub.cli.*` and `knowledge_hub.mcp_server` are compatibility shims only. Internal package code must not import them directly.

### Policy/privacy boundary
- Local-first and policy-first are hard constraints.
- Sensitive knowledge stays in local stores by default.
- Outbound model/provider calls are policy-gated; `P0` outbound must remain blocked.

### Persistence and context boundary
- SQLite, vector collections, parser artifacts, and note/paper/web source records remain canonical local stores.
- Store authority is explicit: source/audit stores can be canonical in their own domain, semantic cards/memory/graph/ontology projections are derivative unless they resolve back to source-backed spans, and operational queues/logs are not factual answer evidence.
- Mixed semantic stores now use table-level authority plus row-level prep columns (`origin`, `contributor_hashes`, `supersedes`, `superseded_by`) so later lifecycle work does not confuse manual rows, aggregated projections, and epistemic review state.
- Mixed derivative read paths now also enforce a first lifecycle boundary: stale `ontology_claims`, `ontology_relations`, legacy `kg_relations`, `learning_graph_*`, and derived `memory_relations` are excluded by default, while `origin='manual'` rows stay visible and mixed-store invalidation runs only for explicit source-document hash changes rather than every document-memory rebuild.
- Epistemic review history is now append-style at the review boundary: `belief_review` / `decision_review` create successor rows linked by `supersedes` / `superseded_by`, and default list surfaces hide superseded history unless a caller explicitly asks for it.
- Aggregated ontology concept/entity projections now track `contributor_hashes`. Their invalidation rule is AND-gated rather than single-hash: a concept/entity row only goes stale when an affected contributor hash intersects its contributor set and no live claim/relation derivative remains for any contributor hash in that set.
- Answer-contract evidence is now narrower than retrieval input by construction: blocked mixed-store source schemes/types (`belief`, `decision`, `outcome`, `ontology`, `learning_*`, `memory_relation`, `entity_merge`, `entity_split`) are treated as retrieval or post-check signals only, never as citation targets. When such rows survive into answer assembly diagnostics they are surfaced under `retrievalSignals`, not under `citations`.
- Verification now treats rejected-belief conflicts and signal-only grounding as structural gates, not stylistic cautions. A rejected belief that is contradicted without explicit conflict framing fails verification, and retrieval signals without citation-grade evidence fail verification instead of passing as grounded success.
- Local workbench helpers may reshape or scope existing sources, but they must not become the system of record or silently add external sync paths.
- Repo/project context is read-only and ephemeral unless a feature explicitly promotes it into a persistent store.

### Default vs labs boundary
- Default surfaces stay retrieval-assistant-first: ingest, search, ask, task-context, health.
- Experimental capabilities ship under `khub labs ...` or profile-gated MCP surfaces first and are promoted only after explicit eval gates.
- Root-level compatibility commands may still exist for direct invocation, but the default `khub --help` surface should prefer the representative core loop and hide personal/eval/maintenance-heavy commands unless they are explicitly promoted.

## KnowledgeOS vNext layering

The current repo direction is:

`Source -> Parse -> Store/Index -> Semantic Units -> Narrow Normalization -> Retrieve/Compare/Synthesize -> Epistemic/Agent`

Definitions:
- `Source`: discover, crawl, download, translate, judge.
- `Parse`: layout-aware paper parsing plus heading/table/figure provenance.
- `Store/Index`: SQLite, vector DB, parser artifacts, metadata indices.
- `Semantic Units`: `Document`, `Element`, `Claim`, `MemoryCard`, `EvidenceLink`.
- `Narrow Normalization`: domain-limited comparison axes such as `task`, `dataset`, `metric`, `comparator`, `condition`, `scope`.
- `Retrieve/Compare/Synthesize`: bounded retrieval, compare reports, limitation summaries, conflict explanations.
- `Epistemic/Agent`: belief/decision/outcome and bounded research-assistant workflows.

## Durable implementation rules

- Keep chunks as the low-level retrieval substrate; semantic units and normalized units sit above them.
- Prefer extending existing handlers, stores, and schema-backed payloads over inventing parallel stacks.
- Keep compatibility shims thin: re-export canonical surfaces or delegate directly, and avoid duplicating transport/runtime logic in legacy modules.
- Keep large refactors additive and facade-preserving: move helper/support bodies into narrower modules first, and defer public contract changes until later phases.
- Keep parser, semantic-unit, normalization, and synthesis outputs inspectable as JSON or CSV.
- Do not tightly couple notebook/workbench consumers to persistence internals.
- Do not let repo/project context persist into the main knowledge stores unless a feature explicitly requires it.

## Current promotion strategy

- Default runtime stays retrieval-first.
- `document-memory`, `claim-normalization`, `claim-synthesis`, deeper memory-route controls, and parser/debug-heavy memory builders remain labs-first.
- Promoted user-facing paper reading surfaces (`khub paper summary|evidence|memory|related`) should stay thin adapters over additive internals rather than forcing a store redesign.
- Promotion path is `labs -> opt-in -> promoted default`, backed by eval evidence rather than architectural intent alone.
- Evidence-first promotion now has a dedicated required CI slice: the fixture-backed golden pack plus answer-contract, verification, epistemic-supersede, and mixed-store lifecycle tests run under the `evidence-first-gates` workflow job before merge.

## Current canonical entrypoints

- `khub` -> `knowledge_hub.interfaces.cli.main:cli`
- `khub-mcp` -> `knowledge_hub.interfaces.mcp.server:main`
- Legacy `knowledge_hub.cli.main` and `knowledge_hub.mcp_server` remain import-compatible shims for external callers and focused shim tests only.
