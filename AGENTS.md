Project instructions for `knowledge-hub`.

Purpose:
- This repository is a local-first knowledge system that combines notes, papers, web ingestion, ontology storage, RAG search, MCP tools, and a Foundry-style agent runtime.
- Protect local-first and policy-first behavior. Do not trade privacy or auditability for convenience without saying so explicitly.

Read this first:
- `README.md`
- `docs/PROJECT_STATE.md`
- `docs/foundry-knowledge-hub-integration.md`
- `docs/guides/cli-commands.md`

Common CLI:
- `khub status`
- `khub search "query"`
- `khub ask "question"`
- `khub agent context "goal" --repo-path .`
- `khub discover "topic" -n 5 --judge`
- `khub paper list`
- `khub index`
- Full command inventory: `docs/guides/cli-commands.md`

Working rules:
1. Preserve architectural boundaries
- `knowledge_hub/` is the Python product/runtime and MCP surface.
- `foundry-core/` is the TypeScript agent-runtime and policy boundary.
- Keep bridge contracts explicit. Do not couple modules across the Python/TypeScript boundary with ad hoc payloads.

2. Protect policy guarantees
- Treat `P0` as blocked from external model calls by default.
- Prefer local processing for sensitive data.
- If a change affects classification, policy gating, or outbound provider behavior, update tests and call out the risk.

3. Keep agent flows inspectable
- Plan/Act/Verify/Writeback behavior must remain traceable.
- New agent behavior should emit structured outputs, preserve auditability, and fail clearly when policy or schema checks block execution.

4. Favor incremental changes
- Prefer small patches over large rewrites.
- Reuse existing MCP handlers, CLI commands, schemas, and provider abstractions before introducing new layers.

5. Leave a durable change record when code changes
- Any code change or feature addition must leave a project record.
- Default minimum record: update `CHANGELOG.md` under `Unreleased`.
- Also update `docs/PROJECT_STATE.md` when the change affects architecture, runtime behavior, policy semantics, tool contracts, retrieval/eval strategy, or current stabilization direction.
- Add or extend an ADR when the change is a durable design decision rather than a local implementation detail.
- For regressions or production-facing bug fixes, add or update the protecting test/eval and mention the fix in `CHANGELOG.md`.

6. Update project memory when behavior changes
- If you change architecture, runtime behavior, tool contracts, policy semantics, or core workflows, update `docs/PROJECT_STATE.md`.
- If you make a durable design decision, add a short entry under the Decisions section in `docs/PROJECT_STATE.md` or split it into a dedicated decision doc when it grows.

7. Verify the relevant surface
- For Python changes, run targeted `pytest` coverage for the affected area when feasible.
- For TypeScript runtime changes in `foundry-core/`, run the relevant tests.
- If you cannot verify, say so clearly.

8. Prefer project sources over assumptions
- Search the codebase before concluding how a subsystem works.
- When repository docs and implementation differ, trust implementation and note the mismatch.

High-signal areas:
- `knowledge_hub/mcp/`
- `knowledge_hub/cli/`
- `knowledge_hub/providers/`
- `knowledge_hub/core/`
- `foundry-core/src/personal-foundry/`
- `tests/` and `foundry-core/tests/`

Preferred skills:
- Use `dev_workflow` when implementing a feature or bug fix that should also leave verification and changelog records.
- Use `code-navigation` before editing when the ownership or call path is not already clear.
- Use `feature-dev` for multi-file implementation work.
- Use `code-review` after implementation to check for regressions, contract drift, and missing tests.
- Use `verification-gate` before calling a change ready.
- Use `project-mindmap` only when the user needs a system map or subsystem placement discussion rather than immediate implementation.

When adding new work:
- Prefer schema-backed payloads for MCP and agent outputs.
- Keep external-call behavior explicit, especially `allow_external`, provider policy guards, and classification handling.
- Avoid introducing hosted storage assumptions that conflict with local-first design unless the user explicitly asks for that tradeoff.
- In the final response, name which project record was updated: `CHANGELOG.md`, `docs/PROJECT_STATE.md`, ADR, or the protecting test/eval.
