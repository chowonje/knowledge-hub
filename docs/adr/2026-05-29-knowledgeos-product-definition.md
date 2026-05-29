# ADR: KnowledgeOS Product Definition And v0.1 Scope

Date: 2026-05-29

## Status

Accepted for the KnowledgeOS v0.1 release-candidate track.

## Context

The repository has several useful but uneven surfaces: CLI, MCP, local stores,
paper parsing, evidence reports, labs workflows, Foundry integration,
Obsidian-facing projections, and experimental visual/table/equation paths. If
all of these are treated as equal product promises, the release target becomes
too broad and the project cannot make a clear Research Preview claim.

The product needs a durable boundary between the long-term KnowledgeOS vision
and the first release-candidate promise.

## Decision

1. **The final product is a local-first, evidence-first research knowledge
   runtime for auditable AI research workflows.**
2. **The shipped product surface is runtime-first, not GUI-first.** The core
   deliverable is `khub`, `khub-mcp`, local stores, structured evidence
   reports, and documentation.
3. **Obsidian remains a source/consumer/projection, not the canonical product
   body.**
4. **v0.1 is scoped to a Research Preview:** section/paragraph evidence-first
   paper QA and comparison over a local AI-paper corpus.
5. **v0.1 does not promise complete table, equation, figure, or image
   understanding.** Those lanes remain labs or limited support until their
   evidence and answerability gates pass.
6. **Answerability is stricter than fluency.** Retrieval hints, fallback chunks,
   locator-only anchors, paraphrases, and visual hints must not become answer
   evidence unless a later gate explicitly promotes them with source hash,
   locator, excerpt, and snippet hash.

## Consequences

- Release work should prioritize the default flow:
  `discover -> index -> search/ask -> evidence review`.
- v0.1 readiness should be judged against a 300-500 paper local corpus,
  80-90% priority parsed-artifact coverage, section/paragraph evidence chunks
  connected to answerability, conservative no-answer behavior, public CLI/MCP
  surface cleanup, green smoke/hygiene/core eval gates, current release docs,
  and a reviewable release-candidate branch/PR state.
- Visual, table, equation, and figure-caption work can continue, but default
  product promotion requires explicit evidence gates rather than architectural
  intent.
- New features should state whether they support the v0.1 runtime promise or
  remain labs/report-only work.

## Rejected Alternatives

### GUI-first KnowledgeOS

Rejected for v0.1 because the stable product substrate is local runtime,
evidence contracts, and CLI/MCP integration. A GUI can be added later as a
consumer of that substrate.

### Obsidian plugin as the canonical product

Rejected because the vault is a valuable source and projection, but the product
authority must live in local stores, evidence artifacts, runtime contracts, and
release gates that Codex/Cursor/Claude-style tools can call directly.

### Full structured-evidence OS in v0.1

Rejected because table numeric QA, equation QA, and figure/image understanding
still need separate parser, provenance, and answerability gates before they are
safe default-surface promises.

## References

- `docs/knowledge_os_definition.md`
- `docs/ARCHITECTURE.md`
- `docs/PROJECT_STATE.md`
- `README.md`
