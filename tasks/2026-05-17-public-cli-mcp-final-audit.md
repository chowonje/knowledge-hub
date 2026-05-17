# Public CLI MCP final audit

## Goal

- Close the Public CLI / MCP final audit tranche after PR #35 by verifying the rendered CLI/MCP surfaces and applying only small consistency fixes needed for release-candidate stabilization.

## Scope

- Keep the root `khub --help` public surface compact.
- Ensure default MCP profile behavior matches the documented public contract.
- Tighten release smoke coverage for public/advanced help surfaces.
- Align README, CLI guide, changelog, project state, and review/worklog records with the verified behavior.

## Non-scope

- No new Artifact Acquisition, Source Alias, ingestion, evidence, provenance, answerability, or corpus-policy work.
- No vault scan, SQLite mutation, public surface expansion, broad CLI rewrite, or MCP tool redesign.
- No push unless explicitly instructed.

## Done Condition

- Default MCP profile hides and blocks known labs/operator tool names unless `KHUB_MCP_PROFILE=labs|all` is used.
- Rendered CLI help and docs agree on public default vs hidden/labs/operator surfaces.
- Focused MCP/CLI tests, smoke/hygiene, full pytest, and diff checks pass or any gap is explicitly recorded.
- Durable records are updated.

## Planned Files

- `knowledge_hub/interfaces/mcp/server.py`
- `knowledge_hub/mcp/tool_specs.py`
- `scripts/check_release_smoke.py`
- `tests/test_mcp_server.py`
- `README.md`
- `docs/guides/cli-commands.md`
- `docs/PROJECT_STATE.md`
- `CHANGELOG.md`
- `worklog/2026-05-17.md`
- `reviews/2026-05-17-public-cli-mcp-final-audit-review.md`

## Verification Plan

- Rendered CLI help checks: root help, advanced help, labs help, papers help.
- Focused tests: MCP profile/tool tests, MCP resources, CLI public surface tests, release smoke contract tests.
- `python -m py_compile` for changed Python files.
- `python scripts/check_release_smoke.py`
- `python scripts/check_public_release_hygiene.py`
- `pytest -q`
- `git diff --check`
