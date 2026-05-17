# Review: Public CLI MCP final audit

## Findings

- Subagent review found one real contract gap: default MCP profile filtering only applied to `list_tools`, so a caller that already knew a labs/operator tool name could call it under the default profile.
- Patched `call_tool_impl` to block known out-of-profile tool names before runtime initialization or dispatch, while preserving `KHUB_MCP_PROFILE=labs|all` as the explicit opt-in.
- Added regression coverage for direct-call blocking of learning, crawl, agentic, paper-build, and async job tools under the default MCP profile.
- Expanded release smoke so the audited surfaces are checked directly: `help advanced`, `labs --help`, `papers --help`, and a hidden paper operator help path.
- Documentation now separates public default commands from hidden/labs/operator examples and splits default MCP tools from labs/all tools.

## Risks

- This is intentionally stricter than the previous hidden-by-discovery behavior. Any external MCP client that relied on calling known labs/operator tool names while launching the server without `KHUB_MCP_PROFILE=labs|all` must opt in explicitly.
- `khub labs eval` remains visible under `khub labs --help` because current project state names it as the canonical eval surface; this tranche did not demote it to hidden.
- Real connected MCP clients were not exercised end-to-end; coverage is through server unit tests, direct enumeration, and local smoke gates.

## Missing Tests

- No live MCP client session was run against Cursor/Codex.
- No vault-backed runtime scenario was run; this tranche did not scan `vault/` and did not need live vault content.
