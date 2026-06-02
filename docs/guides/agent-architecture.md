# Agent Architecture Guide

Knowledge Hub agent work uses four layers:

1. **Core Runtime**: `knowledge_hub/` owns local stores, RAG, ingest, policy, and evidence.
2. **MCP Tool Plane**: `khub-mcp` is the canonical tool list for agents.
3. **Agent Gateway**: `khub agent ...` exposes context, dry-run, and approval-gated writeback contracts.
4. **Interface**: chat, TUI, IDEs, KnowledgeOS skills, and external handoffs orchestrate the other layers.

The default product promise remains:

```text
discover -> index -> search/ask -> evidence review
```

## Claude Code처럼 쓰려면

- Treat Codex, Cursor, `khub chat`, and Foundry as embedded agents.
- Start with a read-only context pack:

```bash
khub agent context "Refactor the RAG fallback flow" --repo-path . --json
```

- Preview an agent run without execution or writeback:

```bash
khub agent run --goal "Update docs for the latest gateway tranche" --repo-path . --dry-run --json
```

- Request a bounded repo-local docs writeback only through Gateway:

```bash
khub agent writeback-request "Update docs/status and worklog for the latest gateway tranche" --repo-path . --json
khub labs ops action-ack --action-id <id> --actor cli-user
khub labs ops action-execute --action-id <id> --actor cli-user --json
```

- Use MCP as the tool plane. Default MCP tools include `search_knowledge`,
  `ask_knowledge`, `build_task_context`, and paper lookup/read helpers.
- Keep chat/TUI as interface surfaces. They should call existing runtime and
  MCP/Gateway contracts instead of creating a second RAG engine.

## Hermes처럼 쓰려면

- Treat Hermes and similar workers as external agents.
- Build a package under `artifacts/<agent>/<YYYY-MM-DD>/`.
- Include only sanitized inputs by default.
- Write a prompt that states `report_only`, `mutationRows=0`, `vaultScanRows=0`, and no `khub` command execution.
- Ask the external agent to produce a report and optional import manifest.
- Review the import manifest locally before any `khub` add, download, embed, index, or writeback command.

Example package:

```text
KnowledgeOS/artifacts/hermes_public_source_radar/2026-06-01/
```

## Boundary Checklist

- Core Runtime owns policy and evidence authority.
- MCP is the canonical agent tool catalog.
- Agent Gateway is a JSON contract and approval lane, not a new command family.
- Interface work must not hide mutations inside ordinary chat or TUI turns.
- External agents are report-only until an operator imports a reviewed manifest.
- Do not add a top-level `khub gateway` family.

## Related Docs

- `docs/adr/2026-06-03-four-layer-agent-architecture.md`
- `docs/adr/2026-06-03-external-agent-handoff-v1.md`
- `docs/guides/embedded-agent-e2e-runbook.md`
- `docs/guides/agent-gateway-v1.md`
- `docs/guides/cli-commands.md`
- `docs/schemas/knowledge-hub.external-agent-handoff.v1.json`
