# Second Core #1: Research Intake + Obsidian Memory

Date: 2026-04-27
Branch: `frontier/research-intake-obsidian-memory-20260427`

## Objective

Implement a router-first `khub add <source>` path for the first Second Core expansion track, with source-agnostic result packets and safe Obsidian staging.

This frontier branch is separate from the public-preview release branch, but `khub add` itself remains the visible intake facade in this worktree. The frontier-only expansion is the broader PDF/Obsidian staging behavior, not a hidden command surface.

## Affected Paths

- `knowledge_hub/interfaces/cli/commands/add_cmd.py`
- `knowledge_hub/interfaces/cli/commands/add/route.py`
- `knowledge_hub/interfaces/cli/commands/add/result.py`
- `knowledge_hub/interfaces/cli/commands/add/lanes.py`
- `knowledge_hub/interfaces/cli/commands/add/obsidian_stage.py`
- `knowledge_hub/interfaces/cli/commands/crawl_support.py`
- `knowledge_hub/web/ingest.py`
- `docs/schemas/add-result.v1.json`
- `tests/test_cli_add_facade.py`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`

## Implementation Notes

- `khub add` now routes paper, web, YouTube, generic PDF URL, and local PDF inputs behind one facade.
- Add result payloads now include `sourceId`, `canonicalUrl`, `canonicalPath`, `title`, `contentHash`, `stored`, `indexed`, `obsidianStage`, `warnings`, and `nextActions`.
- Paper URL adds no longer run paper-memory/document-memory by default; `--build-memory` is explicit.
- `--to-obsidian` is stage-only and never calls final apply.
- Generic/local PDFs default to the web/document lane; paper-style hosts stay in the paper lane.
- Explicit `--type paper` rejects non-paper URLs instead of passing them to the paper resolver, while generic PDF URLs can still be forced into the paper lane.
- `add_cmd.py` is now a thin Click shell; routing, packet normalization, lane execution, and Obsidian staging live in dedicated `commands/add/` modules.
- Hugging Face routing is path-aware: only `/papers/...` pages auto-route to paper import; model and dataset pages remain web sources.
- Add result output redacts local file paths, `file://` URIs, and internal single-paper import artifact paths before the JSON/text packet is emitted.

## Verification

- `python -m py_compile knowledge_hub/interfaces/cli/commands/add_cmd.py knowledge_hub/interfaces/cli/commands/add/route.py knowledge_hub/interfaces/cli/commands/add/result.py knowledge_hub/interfaces/cli/commands/add/obsidian_stage.py knowledge_hub/interfaces/cli/commands/add/lanes.py knowledge_hub/interfaces/cli/commands/crawl_support.py knowledge_hub/web/ingest.py`
- `python -m ruff check knowledge_hub/interfaces/cli/commands/add_cmd.py knowledge_hub/interfaces/cli/commands/add knowledge_hub/interfaces/cli/commands/crawl_support.py knowledge_hub/web/ingest.py tests/test_cli_add_facade.py`
- `python -m pytest tests/test_cli_add_facade.py tests/test_paper_import_csv.py tests/test_paper_source_freshness.py tests/test_mixed_store_lifecycle.py -q`
- `python -m pytest tests/test_cli_smoke_contract.py tests/test_interfaces_cli_main.py tests/test_public_release_hygiene.py tests/test_provider_custom_surface.py -q`
- `python -m pytest tests/test_mcp_server.py tests/test_mcp_server_helpers.py tests/test_mcp_jobs.py -q`
- `python -m pytest tests/test_rag_search.py tests/test_search_cmd.py tests/test_doctor_cmd.py tests/test_index_freshness.py -q`
- `python -m pytest -q` (`998 passed, 5 warnings`)

## Review Follow-up

- Addressed Opus review blockers by aligning the frontier/public-preview wording, tightening explicit paper/youtube routing validation, adding paper-query and local-PDF tests, warning on local-PDF excerpt-only ingest, and deleting temporary add CSV plus manifest files after paper import dispatch.
- Completed the Opus follow-up split so future lanes can be added without growing the command entrypoint again.
- Addressed the follow-up subagent review by fixing Hugging Face non-paper page routing and path privacy leaks in local-PDF and paper-import add results.

## Follow-up

- Paper Obsidian staging still needs a managed ko-note-compatible path before it should replace direct paper writeback.
- Local PDFs can stage only after successful local extraction/indexing; local paper-PDF import remains a separate resolver/import enhancement.
