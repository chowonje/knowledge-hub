# Canonical Project Maps

This directory is the canonical architecture-map surface for `knowledge-hub`.

Use these maps when you need a compact, implementation-grounded view of the repo
without reading every long-form design document first.

## Maps

- `canonical-ownership-map.md`
  - Read this first when you need package ownership, canonical entrypoints, shim
    status, or the Python/TypeScript boundary.
- `agent-execution-map.md`
  - Read this when you need the actual execution flow for `khub agent run` and
    MCP `run_agentic_query`, including Foundry-first delegation and fallback.
- `data-policy-flow-map.md`
  - Read this when you need to understand local stores, ephemeral context,
    outbound policy gates, and writeback/audit implications.

Each Markdown file is the human-readable explanation. Each matching `.mmd` file
is the Mermaid diagram source.

## Recommended Read Order

For a new engineer or agent, read these in order:

1. `docs/maps/README.md`
2. `canonical-ownership-map.md`
3. `agent-execution-map.md`
4. `data-policy-flow-map.md`
5. `docs/PROJECT_STATE.md`

## Source-of-Truth Relationship

These maps summarize implementation reality. They do not replace the existing
long-form docs.

- `docs/ARCHITECTURE.md`
  - High-level architectural boundaries and durable implementation rules.
- `docs/PROJECT_STATE.md`
  - Current runtime behavior, promoted surfaces, known risks, and active
    implementation reality.
- `docs/repo-layout.md`
  - Canonical package ownership and root-level layout rules.

`/Users/won/Desktop/allinone/KnowledgeOS/PROJECT_MINDMAP.md` remains a planning
workbench, not the canonical architecture-map source.
