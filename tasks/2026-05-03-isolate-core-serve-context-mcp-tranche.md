# Isolate core-serve-context-mcp tranche

## Objective

Split the salvage snapshot into a clean worktree that keeps only the core context-serving route:
`context_pack`, `task_context`, MCP `search_knowledge`, MCP `build_task_context`, and the small Codex-facing MCP server entrypoint.

## Scope

- Included: bounded workspace/knowledge context assembly, task-context payloads, MCP search/context serving, matching schemas, focused tests, and product records.
- Excluded: side layer, eval center, failure bank, RAG vNext, Foundry, Dinger, OS bridge, learning, and generated artifacts.

## Verification

- Passed: `pytest tests/test_task_context.py tests/test_mcp_server.py -q` (`7 passed in 0.91s`).
- Passed: `python -c "... assert khub-mcp == 'knowledge_hub.interfaces.mcp.server:main' ..."`
- Passed: `python -c "... assert not inspect.iscoroutinefunction(s.main) ..."`
- Passed: strict schema validation for a representative `knowledge-hub.task-context.result.v1` payload.
- Passed: `python -m compileall -q` for the staged context/MCP Python modules.
