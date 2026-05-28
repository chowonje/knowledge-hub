# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in Surface Design

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-labs-opt-in-surface-design.v1`
- status: `ready`
- decision: `parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design_ready`
- nextRecommendedTranche: `parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_implementation`
- designRows: `10`
- designReadyRows: `10`
- blockedRows: `0`
- plannedLabsCliRows: `1`
- plannedLabsMcpRows: `1`
- plannedPublicCliRows: `0`
- plannedDefaultMcpRows: `0`
- publicKhubAskClosed: `True`
- defaultMcpAskClosed: `True`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Design

- surfaceMode: `labs_only_explicit_opt_in`
- futureLabsCliCommand: `khub labs paper evidence-chunk-ask`
- futureLabsMcpTool: `paper_evidence_chunk_answer_preview`
- allowedProfiles: `labs, all`
- queryPlanOptInValue: `runtime_v1`
- requiredSourceType: `paper`
- requiredResolvedPaperIds: `True`
- allowExternalDefault: `False`

## Policy

Report-only surface design. It plans a labs-only explicit opt-in surface and keeps public `khub ask` plus default MCP `ask_knowledge` closed. It does not implement a command, generate answers, call LLMs, write stores, create evidence records, mutate indexes, or scan the vault.

## Rows

### `default_off_no_answer_smoke_gate`

- layer: `upstream_gate`
- status: `design_ready`
- surface: `eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke.v1.json`
- currentObserved: `True`
- requiredForNext: `True`
- plannedSurface: Consume the default-off/no-answer smoke as the authority before designing any opt-in surface.
- safetyContract: labs design is blocked unless the default answer path remains fail-closed.
- blockers: ``
- nextCheck: Generated report must be ready with all scenarios no-answer and zero public/default/LLM/evidence mutation counters.

### `internal_python_query_plan_ingress`

- layer: `internal_python_api`
- status: `design_ready`
- surface: `knowledge_hub/ai/rag.py`
- currentObserved: `True`
- requiredForNext: `True`
- plannedSurface: Future labs callers use RAGSearcher.generate_answer/stream_answer with query_plan opt-in instead of adding a public khub ask flag.
- safetyContract: activation remains owned by the runtime adapter; absent query_plan keeps existing behavior.
- blockers: ``
- nextCheck: Implementation should pass query_plan keys through unchanged and keep defaults None.

### `public_khub_ask_stays_closed`

- layer: `public_cli`
- status: `design_ready`
- surface: `knowledge_hub/interfaces/cli/commands/search_cmd.py`
- currentObserved: `True`
- requiredForNext: `True`
- plannedSurface: Do not add parsed-artifact evidence chunk flags to public khub ask in the labs opt-in tranche.
- safetyContract: public CLI remains default-off; labs-only command is the only planned user-visible opt-in surface.
- blockers: ``
- nextCheck: Focused CLI test should assert khub ask help has no parsed-artifact evidence chunk flag.

### `default_mcp_ask_stays_closed`

- layer: `default_mcp`
- status: `design_ready`
- surface: `knowledge_hub/mcp/tool_specs.py`
- currentObserved: `True`
- requiredForNext: `True`
- plannedSurface: Do not add query_plan or adapter opt-in fields to the default ask_knowledge MCP schema.
- safetyContract: default MCP profile stays retrieval-assistant-first and cannot activate evidence chunks by accident.
- blockers: ``
- nextCheck: MCP tests should assert ask_knowledge default schema has no query_plan or adapter opt-in fields.

### `labs_cli_paper_group_host`

- layer: `labs_cli`
- status: `design_ready`
- surface: `knowledge_hub/interfaces/cli/commands/paper_labs_cmd.py`
- currentObserved: `True`
- requiredForNext: `True`
- plannedSurface: Use future command `khub labs paper evidence-chunk-ask` as the human/operator opt-in host.
- safetyContract: command must require explicit --paper-id and --source paper, default --no-allow-external, and JSON diagnostics.
- blockers: ``
- nextCheck: Next tranche may add this labs subcommand only under khub labs paper.

### `labs_mcp_profile_host`

- layer: `labs_mcp`
- status: `design_ready`
- surface: `knowledge_hub/mcp/tool_specs.py`
- currentObserved: `True`
- requiredForNext: `True`
- plannedSurface: Use future labs/all-only MCP tool `paper_evidence_chunk_answer_preview` for tool callers.
- safetyContract: future tool must be absent from default profile and blocked by default MCP profile direct-call enforcement.
- blockers: ``
- nextCheck: Next tranche may add the tool only if default profile tests prove it is hidden/blocked.

### `future_labs_cli_command_contract`

- layer: `future_labs_cli_contract`
- status: `design_ready`
- surface: `knowledge_hub/interfaces/cli/commands/paper_labs_cmd.py`
- currentObserved: `False`
- requiredForNext: `False`
- plannedSurface: `khub labs paper evidence-chunk-ask QUESTION --paper-id <id> --json` builds a paper-only query_plan with parsed_artifact_evidence_chunk_adapter=runtime_v1 and resolvedPaperIds.
- safetyContract: future command may expose answer payload only in labs and only after no-answer/default-off gates stay green.
- blockers: ``
- nextCheck: Implementation must keep public khub ask unchanged and add focused labs CLI tests.

### `future_labs_mcp_tool_contract`

- layer: `future_labs_mcp_contract`
- status: `design_ready`
- surface: `knowledge_hub/mcp/tool_specs.py`
- currentObserved: `False`
- requiredForNext: `False`
- plannedSurface: `paper_evidence_chunk_answer_preview` accepts question plus explicit paper_ids and forwards the same query_plan opt-in through the internal searcher API.
- safetyContract: future labs MCP tool must not be discoverable or callable from KHUB_MCP_PROFILE=default.
- blockers: ``
- nextCheck: Implementation must extend default-profile block tests before returning answer-visible data.

### `explicit_paper_scope_policy`

- layer: `answerability_policy`
- status: `design_ready`
- surface: `knowledge_hub/ai/parsed_artifact_evidence_chunk_runtime_adapter.py`
- currentObserved: `True`
- requiredForNext: `True`
- plannedSurface: Require source_type=paper and explicit resolvedPaperIds/paper_ids for every labs opt-in invocation.
- safetyContract: no fallback to all papers, visual hints, table/equation/figure artifacts, or unresolved retrieval aliases.
- blockers: ``
- nextCheck: Next tranche should test missing paper ids and non-paper sources still no-answer.

### `local_first_call_policy`

- layer: `provider_policy`
- status: `design_ready`
- surface: `knowledge_hub/interfaces/cli/commands/paper_labs_cmd.py`
- currentObserved: `True`
- requiredForNext: `True`
- plannedSurface: Default labs opt-in answer runs with allow_external=false; external calls require an explicit later policy decision.
- safetyContract: local-first remains the default; report-only design makes zero LLM/model/API calls.
- blockers: ``
- nextCheck: Next tranche should keep default --no-allow-external and verify no API calls in smoke tests.
